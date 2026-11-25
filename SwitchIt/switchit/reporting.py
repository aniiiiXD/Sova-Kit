"""Build and persist migration reports from evaluated run evidence."""

from __future__ import annotations

from pathlib import Path

from switchit._run_utils import clip, utc_now_iso
from switchit.report_html import render_html
from switchit.storage import atomic_write_json

BOUNDARIES = [
    "Workflow labels and similarity scores are triage aids, not measured accuracy.",
    "Proposed tool calls are captured for comparison; the harness never executes a business tool.",
    "A historical model response is reference behavior, not proof of success; checks define success explicitly.",
    "Token usage is kept separate by provenance: recorded (source-reported), estimated, and unknown. These are telemetry, not invoices.",
    "End-to-end outcomes for workflows that need a real tool remain unverified until a safe tool fixture or authorized test environment exists.",
]


def build_report(results_doc: dict, summary: dict, cases_doc: dict, meta: dict | None = None) -> dict:
    """Assemble the full migration report document."""
    meta = dict(meta or {})
    report = {
        "report_version": 1,
        "generated_at": utc_now_iso(),
        "runner_version": results_doc.get("runner_version"),
        "fingerprint": results_doc.get("fingerprint"),
        "baseline_spec": results_doc.get("baseline_spec"),
        "target_spec": results_doc.get("target_spec"),
        "verdicts": summary["verdicts"],
        "capability_gaps": summary["capability_gaps"],
        "failure_groups": summary["failure_groups"],
        "prioritized_fixes": summary["prioritized_fixes"],
        "usage": summary["usage"],
        "cases": [],
        "boundaries": BOUNDARIES,
        "notes": [],
        "meta": meta,
    }
    for result in results_doc.get("case_results") or []:
        case = meta.get("cases_by_id", {}).get(result["case_id"], {})
        report["cases"].append(
            {
                "case_id": result["case_id"],
                "workflow_id": result.get("workflow_id"),
                "trace_id": result.get("trace_id"),
                "origins": result.get("origins"),
                "verdict": result["verdict"],
                "failures": [
                    {
                        "kind": failure["kind"],
                        "check": failure.get("check"),
                        "detail": (failure.get("check") or {}).get("detail", ""),
                    }
                    for failure in result.get("failures") or []
                ],
                "uncertainty": result.get("uncertainty") or [],
                "notes": result.get("notes") or [],
                "tool_findings": result.get("tool_findings") or [],
                "baseline_excerpt": clip((result.get("baseline") or {}).get("text")),
                "target_excerpt": clip((result.get("target") or {}).get("text")),
                "baseline_usage": (result.get("baseline") or {}).get("usage"),
                "target_usage": (result.get("target") or {}).get("usage"),
                "baseline_latency_ms": (result.get("baseline") or {}).get("latency_ms"),
                "target_latency_ms": (result.get("target") or {}).get("latency_ms"),
                "target_error": (result.get("target") or {}).get("error"),
                "checks_target": result.get("checks_target") or [],
                "chain": result.get("chain"),
                "request_messages": [
                    {"role": message.get("role"), "content": clip(message.get("content"), 300)}
                    for message in (case.get("request") or {}).get("messages") or []
                ],
            }
        )
    report["notes"].append(
        "All historical content in this report was redacted at import time; excerpts are additionally clipped."
    )
    report["notes"].append(
        "Declared adapter capabilities are descriptions to verify against provider documentation, not measured results."
    )
    return report


def write_report(report: dict, run_dir) -> dict:
    """Write report.json and report.html into the run directory."""
    run_dir = Path(run_dir)
    json_path = run_dir / "report.json"
    html_path = run_dir / "report.html"
    atomic_write_json(json_path, report)
    html_path.write_text(render_html(report), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}
