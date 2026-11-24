"""Provisional workflow grouping using task text and tool overlap.

These groups are review aids, not executable workflow specifications."""

from __future__ import annotations

from collections import defaultdict
import math
import re

from switchit.redaction import EVIDENCE_CAP, excerpt
from switchit.usage import merge_usage

WORKFLOW_MAP_VERSION = 1


STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "you", "your", "are", "was", "were",
    "can", "could", "would", "should", "have", "has", "had", "not", "but", "all",
    "please", "need", "want", "from", "into", "about", "there", "their", "them",
    "they", "what", "when", "where", "which", "who", "how", "why", "will", "just",
    "get", "got", "does", "did", "done", "any", "our", "out", "too", "also", "very",
}


def tokenize(text: str) -> list:
    return [
        token for token in re.findall(r"[a-z][a-z0-9_]{2,}", (text or "").lower())
        if token not in STOPWORDS and len(token) < 32
    ]


class TfidfIndex:
    """Small deterministic TF-IDF + cosine similarity index (pure stdlib)."""

    def __init__(self):
        self.idf = {}
        self.vectors = {}

    def fit(self, documents: list) -> "TfidfIndex":
        document_frequency = defaultdict(int)
        for document in documents:
            for token in set(tokenize(document)):
                document_frequency[token] += 1
        total = max(1, len(documents))
        self.idf = {
            token: math.log((1 + total) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        self.vectors = {index: self.vector(document) for index, document in enumerate(documents)}
        return self

    def vector(self, text: str) -> dict:
        counts = defaultdict(float)
        for token in tokenize(text):
            counts[token] += 1.0
        vector = {token: (1.0 + math.log(count)) * self.idf.get(token, 1.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm > 0:
            vector = {token: value / norm for token, value in vector.items()}
        return vector

    @staticmethod
    def cosine(a: dict, b: dict) -> float:
        if len(b) < len(a):
            a, b = b, a
        return sum(value * b.get(token, 0.0) for token, value in a.items())

    def similarity_to_text(self, vector: dict, text: str) -> float:
        return self.cosine(vector, self.vector(text))


def tool_jaccard(a: list, b: list) -> float:
    set_a, set_b = set(a or []), set(b or [])
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def trace_evidence_level(trace: dict) -> str:
    if trace["user_requests"]:
        return "user_request"
    if trace["tool_names"]:
        return "tool_call"
    return "assistant_only"


def task_document(trace: dict) -> str:
    parts = list(trace["user_requests"])
    if not parts:
        parts.append(" ".join(trace["tool_names"]))
    return " ".join(parts)


def group_workflows(store: dict, threshold: float = 0.34) -> dict:
    """Provisionally group traces into workflows.

    Similarity = 0.65 * TF-IDF cosine over task text + 0.35 * tool-set Jaccard
    (text-only when neither side has tools). Greedy, deterministic assignment.
    Every output field that a reviewer might correct lives in the workflow map,
    and every cluster carries confidence and uncertainty signals.
    """
    traces = store["traces"]
    calls_by_id = {call["call_id"]: call for call in store["calls"]}
    documents = [task_document(trace) for trace in traces]
    index = TfidfIndex().fit(documents)
    vectors = [index.vectors[i] for i in range(len(traces))]

    clusters = []  # each: {"members": [indices], "centroid_terms": {}}
    for position, trace in enumerate(traces):
        best_cluster, best_score = None, 0.0
        for cluster in clusters:
            representative = max(
                index.cosine(vectors[position], vectors[member]) for member in cluster["members"]
            )
            member = traces[cluster["members"][0]]
            tool_score = tool_jaccard(trace["tool_names"], member["tool_names"])
            if not trace["tool_names"] and not member["tool_names"]:
                score = representative
            else:
                score = 0.65 * representative + 0.35 * tool_score
            if score > best_score:
                best_cluster, best_score = cluster, score
        if best_cluster is not None and best_score >= threshold:
            best_cluster["members"].append(position)
        else:
            clusters.append({"members": [position], "seed_score": best_score})

    workflows = []
    total_traces = max(1, len(traces))
    for cluster_position, cluster in enumerate(clusters):
        members = [traces[i] for i in cluster["members"]]
        member_vectors = [vectors[i] for i in cluster["members"]]
        tool_counter = defaultdict(int)
        for trace in members:
            for name in trace["tool_names"]:
                tool_counter[name] += 1
        term_mass = defaultdict(float)
        for vector in member_vectors:
            for token, value in vector.items():
                term_mass[token] += value
        top_terms = [term for term, _ in sorted(term_mass.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        top_tools = [name for name, _ in sorted(tool_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        parts = []
        if top_terms:
            parts.append(" ".join(top_terms).title())
        if top_tools:
            parts.append(top_tools[0])
        name = " — ".join(parts) if parts else f"Unlabeled activity {cluster_position + 1}"
        name = name[:80]

        usage = merge_usage(*[trace["usage"] for trace in members])
        call_count = sum(trace["call_count"] for trace in members)
        error_traces = sum(1 for trace in members if trace["error_count"])
        evidence_requests = []
        for trace in members:
            for request in trace["user_requests"]:
                evidence_requests.append({"trace_id": trace["trace_id"], "text": excerpt(request, EVIDENCE_CAP)})
        evidence_requests = evidence_requests[:5]
        example_trace_ids = [trace["trace_id"] for trace in members[:5]]

        sizes = [len(members)]
        uncertainties = []
        evidence_level = trace_evidence_level(members[0]) if len(members) == 1 else (
            "user_request" if any(trace["user_requests"] for trace in members) else "tool_call"
        )
        if len(members) == 1:
            uncertainties.append("single_trace_cluster")
        if evidence_level != "user_request":
            uncertainties.append("no_user_request_evidence")
        if error_traces:
            uncertainties.append("contains_failed_calls")
        intra = []
        if len(member_vectors) > 1:
            for i in range(len(member_vectors)):
                for j in range(i + 1, len(member_vectors)):
                    intra.append(index.cosine(member_vectors[i], member_vectors[j]))
            mean_intra = sum(intra) / len(intra)
            if mean_intra < 0.2:
                uncertainties.append("low_internal_similarity")
        else:
            mean_intra = 1.0
        if len(members) >= 2 and evidence_level == "user_request" and mean_intra >= 0.2:
            confidence = "high"
        elif len(members) >= 2 or evidence_level == "user_request":
            confidence = "medium"
        else:
            confidence = "low"

        workflows.append(
            {
                "id": f"wf_{cluster_position + 1:03d}",
                "name": name,
                "description": (
                    f"Provisional group of {len(members)} trace(s) around terms: "
                    f"{', '.join(top_terms) if top_terms else '(no dominant terms)'}"
                    + (f"; tools: {', '.join(top_tools)}" if top_tools else "")
                    + "."
                ),
                "trace_ids": sorted(trace["trace_id"] for trace in members),
                "tool_names": sorted(tool_counter),
                "evidence": {"example_trace_ids": example_trace_ids, "user_requests": evidence_requests},
                "confidence": confidence,
                "uncertainty": uncertainties,
                "evidence_level": evidence_level,
                "label_source": "heuristic",
                "edited": False,
                "stats": {
                    "traces": len(members),
                    "calls": call_count,
                    "error_traces": error_traces,
                    "usage": usage,
                    "share_of_traces": round(len(members) / total_traces, 4),
                    "representative_latency_ms": _representative_latency(members, calls_by_id),
                },
            }
        )
    assigned = set()
    for workflow in workflows:
        assigned.update(workflow["trace_ids"])
    unassigned = sorted(trace["trace_id"] for trace in traces if trace["trace_id"] not in assigned)
    return {
        "workflow_map_version": WORKFLOW_MAP_VERSION,
        "params": {"similarity_threshold": threshold},
        "workflows": workflows,
        "unassigned_trace_ids": unassigned,
        "notes": [
            "Workflow labels and similarity scores are triage aids, not measured accuracy.",
            "Names are derived from dominant TF-IDF terms and tool names; a reviewer (human or agent) should correct them.",
        ],
    }


def _representative_latency(members: list, calls_by_id: dict):
    values = [trace["latency_ms_max"] for trace in members if trace["latency_ms_max"] is not None]
    if not values:
        for trace in members:
            for call_id in trace["call_ids"]:
                latency = calls_by_id[call_id]["latency_ms"]
                if latency is not None:
                    values.append(latency)
    return max(values) if values else None
