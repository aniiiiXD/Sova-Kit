"""Execute cases through adapters and checkpoint complete runs."""

from __future__ import annotations

from pathlib import Path

from switchit._run_utils import clip, utc_now_iso
from switchit.checkpoints import (
    RUNNER_VERSION,
    _checkpoint_path,
    load_checkpoint,
    run_fingerprint,
)
from switchit.evaluation import finalize_case
from switchit.storage import atomic_write_json
from switchit.fixtures import FixtureSession

MAX_CHAIN_TURNS = 3


def _chain_target_turns(case: dict, target_adapter, max_turns: int = MAX_CHAIN_TURNS):
    """Chained replay: continue the conversation through recorded tool fixtures.

    When the target proposes a business-tool call (Supermemory, an internal API,
    anything), the recorded result from history is fed back so the model can
    continue to a final answer. No real business tool is ever executed. If the
    target proposes a tool history never used, it receives an explicit
    placeholder result and the gap is reported.
    """
    request = case.get("request") or {}
    exact = FixtureSession(case["fixture_set"], case.get("trace_id")) if case.get("fixture_set") else None
    fixtures = case.get("tool_fixtures") or {}
    messages = list(request.get("messages") or [])
    tools = list(request.get("tools") or [])
    fixtures_used, fixtures_missing = [], []
    proposals = []
    turns = []
    final = None
    for turn in range(max_turns):
        result = dict(target_adapter.replay({"messages": messages, "tools": tools}, case))
        result["tool_calls"] = [dict(call, id=call.get("id") or f"call_{turn}_{position}")
                                for position, call in enumerate(result.get("tool_calls") or [])]
        proposals.extend(result.get("tool_calls") or [])
        turns.append({
            "turn": turn + 1,
            "text": clip(result.get("text"), 300),
            "tool_calls": [call.get("name", "") for call in result.get("tool_calls") or []],
            "error": result.get("error"),
            "usage": result.get("usage"),
            "latency_ms": result.get("latency_ms"),
        })
        final = result
        if result.get("error") or not result.get("tool_calls"):
            break
        proposed = result["tool_calls"]
        messages = messages + [{
            "role": "assistant",
            "content": result.get("text") or "",
            "tool_calls": proposed,
        }]
        for position, tool_call in enumerate(proposed):
            name = tool_call.get("name", "")
            if exact:
                match = exact.resolve(name, tool_call.get("arguments", {}))
                if match["status"] != "matched":
                    fixtures_missing.append(name)
                    final = dict(final, error=f"exact fixture {match['status']} for {name}")
                    break
                import json
                fixture_text = match["result"] if isinstance(match["result"], str) else json.dumps(match["result"])
                fixtures_used.append(match["fixture_id"])
            elif name in fixtures:
                fixture_text = fixtures[name]
                if name not in fixtures_used:
                    fixtures_used.append(name)
            else:
                fixture_text = (
                    f"[switchit fixture] no recorded result for tool '{name}'; "
                    "the historical trace never used it."
                )
                if name not in fixtures_missing:
                    fixtures_missing.append(name)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": fixture_text,
            })
        if final.get("error"):
            break
    if final is not None:
        bases = {(t.get("usage") or {}).get("basis", "unknown") for t in turns}
        usage = {"basis": next(iter(bases)) if len(bases) == 1 else "unknown", "estimator": None}
        for field in ("input_tokens", "output_tokens"):
            values = [(t.get("usage") or {}).get(field) for t in turns]
            usage[field] = sum(values) if all(v is not None for v in values) else None
        latencies = [t.get("latency_ms") for t in turns]
        final = dict(final, usage=usage,
                     latency_ms=sum(latencies) if all(v is not None for v in latencies) else None)
    return final, {
        "completed": bool(final and not final.get("error") and not final.get("tool_calls")),
        "fixture_matching": "exact" if exact else "legacy_name_only",
        "turns": turns,
        "proposals": proposals,
        "fixtures_used": fixtures_used,
        "fixtures_missing": fixtures_missing,
        "max_turns": max_turns,
    }


def replay_case(case: dict, baseline_adapter, target_adapter, chain: bool = False) -> dict:
    """Replay one case through baseline and target adapters.

    With ``chain=True`` and recorded tool fixtures attached to the case, the
    target's tool proposals are continued with the recorded results so checks
    can evaluate the final answer of the workflow, not just the first proposal.
    """
    request = case.get("request") or {}
    baseline_result = baseline_adapter.replay(request, case)
    if chain and (case.get("tool_fixtures") or case.get("fixture_set")):
        target_result, chain_meta = _chain_target_turns(case, target_adapter)
        # Preserve all proposals, including repeated calls and their order.
        target_result = {**target_result, "tool_calls": chain_meta["proposals"]}
    else:
        target_result = target_adapter.replay(request, case)
        chain_meta = None
    case_result = finalize_case(case, baseline_result, target_result)
    if chain_meta:
        case_result["chain"] = chain_meta
        if not chain_meta["completed"] or chain_meta["fixture_matching"] == "legacy_name_only":
            case_result["uncertainty"].append("chain incomplete or fixtures matched only by name; full workflow validity is unverified")
            if case_result["verdict"] == "pass":
                case_result["verdict"] = "uncertain"
        if chain_meta["fixtures_missing"]:
            case_result["uncertainty"].append(
                "target proposed tool(s) with no recorded fixture ("
                + ", ".join(chain_meta["fixtures_missing"])
                + "); the continuation received a placeholder result"
            )
        used = ", ".join(chain_meta["fixtures_used"]) or "none"
        case_result["uncertainty"].append(
            f"chained replay used recorded tool fixtures ({used}); end-to-end validity "
            "depends on fixture fidelity, and no real business tool was executed"
        )
    return case_result


def run_replay(cases_doc: dict, baseline_adapter, target_adapter, baseline_spec: dict,
               target_spec: dict, run_dir, resume: bool = True) -> dict:
    """Replay all cases with per-case checkpointing.

    The checkpoint records completed case results plus the run fingerprint.
    If the fingerprint changes (cases, adapter specs, or runner version), the
    previous state is refused: stale results must never masquerade as fresh.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(cases_doc, baseline_spec, target_spec)
    state = None
    if resume:
        state = load_checkpoint(run_dir)
        if state and state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Run state does not match the current cases, adapters, or runner version. "
                "Use a fresh --run-dir or restore the matching cases.json and adapter settings."
            )
    if not state:
        state = {
            "runner_version": RUNNER_VERSION,
            "fingerprint": fingerprint,
            "started_at": utc_now_iso(),
            "baseline_spec": baseline_spec,
            "target_spec": target_spec,
            "completed": {},
        }

    results = []
    chain = bool((target_spec or {}).get("chain"))
    for case in cases_doc.get("cases") or []:
        case_id = case["case_id"]
        if case_id in state["completed"]:
            results.append(state["completed"][case_id])
            continue
        case_result = replay_case(case, baseline_adapter, target_adapter, chain=chain)
        state["completed"][case_id] = case_result
        state["updated_at"] = utc_now_iso()
        atomic_write_json(_checkpoint_path(run_dir), state)
        results.append(case_result)

    document = {
        "runner_version": RUNNER_VERSION,
        "fingerprint": fingerprint,
        "started_at": state.get("started_at"),
        "finished_at": utc_now_iso(),
        "baseline_spec": baseline_spec,
        "target_spec": target_spec,
        "case_results": results,
    }
    atomic_write_json(run_dir / "results.json", document)
    return document
