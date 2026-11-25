"""Client example for two local vLLM workers, using actual prompt token IDs."""
from dataclasses import dataclass
import json
from time import perf_counter
from urllib.request import Request, urlopen
from uuid import uuid4

from kvverify.vllm_handoff import CacheStore


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    api_key: str | None = None

    def first_token(self, tokens, scope, *, handoff_id=None, op=None, prefix_tokens=None):
        body = {"model": self.model, "prompt": list(tokens), "max_tokens": 1, "temperature": 0,
                "add_special_tokens": False, "logprobs": 5, "stream": True, "cache_salt": scope}
        if op is not None:
            body["kv_transfer_params"] = {"sova": {"op": op, "id": handoff_id,
                                                     "prefix_tokens": prefix_tokens}}
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url.rstrip("/") + "/completions",
                          data=json.dumps(body).encode(), headers=headers)
        start = perf_counter()
        result = None
        with urlopen(request, timeout=180) as stream:
            for raw_line in stream:
                elapsed_ms = (perf_counter() - start) * 1_000
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                event = line.partition(":")[2].strip()
                if not event:
                    continue
                if event == "[DONE]":
                    break
                chunk = json.loads(event)
                if "error" in chunk:
                    raise RuntimeError(f"vLLM completion failed: {chunk['error']}")
                for choice in chunk.get("choices", []):
                    probabilities = choice.get("logprobs") or {}
                    if result is None and (choice.get("text") or probabilities.get("tokens")):
                        result = {"text": choice.get("text"), "logprobs": probabilities, "ttft_ms": elapsed_ms}
        if result is None:
            raise RuntimeError("vLLM returned no token")
        return result


def compare(source, target, cache_directory, scope, prefix, source_suffix, target_suffix, *, mapper=None):
    """Compare a handoff and native prefill. Use mapper=None for the identity control."""
    prefix, source_suffix, target_suffix = tuple(prefix), tuple(source_suffix), tuple(target_suffix)
    if not prefix or not source_suffix or not target_suffix:
        raise ValueError("The prefix and both suffixes must contain tokens")
    handoff_id = uuid4().hex
    common = {"handoff_id": handoff_id}
    target_tokens = prefix + target_suffix
    started = perf_counter()
    source_result = source.first_token(prefix + source_suffix, scope, op="export",
                                       prefix_tokens=len(prefix), **common)
    if mapper is not None:
        head = mapper.native_head_tokens
        if head:
            target.first_token(prefix[:head + 1], scope, op="export", prefix_tokens=head, **common)
        mapper.prepare(CacheStore(cache_directory), scope, handoff_id, prefix)
    prepared_ms = (perf_counter() - started) * 1_000
    handoff = target.first_token(target_tokens, scope, op="load", prefix_tokens=len(prefix), **common)
    native = target.first_token(target_tokens, scope)
    # vLLM's worker log confirms whether the load was served or recomputed.
    return {"handoff_id": handoff_id, "source": source_result, "handoff": handoff,
            "native": native, "preparation_ms": prepared_ms}
