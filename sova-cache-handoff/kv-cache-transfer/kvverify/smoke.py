"""Offline integration smoke: synthetic cache math and evidence plumbing only."""
from .data import Example, assert_disjoint, json_digest
from .evidence import ARMS, write_report, validate_trace
from .mathops import affine, fit_affine, rotate


def example(example_id="eval-1", document_id="heldout-1", context="Revenue was 120, then 150.", split="eval"):
    return Example.parse({"id": example_id, "document_id": document_id, "context": context,
                          "question": "Percentage growth?", "choices": {"A": "25%", "B": "50%"},
                          "answer": "A", "split": split})


def trace(arm="transfer_cold", case=None, repeat=0):
    case = case or example()
    row = {"id": case.id, "document_id": case.document_id, "context_hash": case.context_hash,
           "prompt_hash": json_digest(case.prompt()), "run_config_hash": "synthetic-smoke-config",
           "repeat": repeat, "arm": arm, "evidence_kind": "synthetic_fixture", "status": "ok",
           "prompt_tokens": 8, "ttft_ms": 1., "total_ms": 2., "correct": True, "answer": "A"}
    if arm.startswith("transfer"):
        row.update(source_prefill_ms=.25, map_ms=.25, import_ms=.25, target_first_ms=.25,
                   fallback=False, imported_tokens=7, target_prefill_tokens=1)
        if arm == "transfer_warm":
            row["ttft_ms"] = .75
    return row


def run(output):
    calibration = example("cal-1", "cal-doc", "A separate calibration document.", "calibration")
    cases = [example(), example("eval-2", "heldout-2", "Another unrelated held-out document.")]
    assert_disjoint([calibration], cases)
    x = [[-2., 1.], [-1., 3.], [0., -1.], [1., 2.], [2., -3.], [3., 4.]]
    expected_w, expected_b = [[2., -.5], [1., 3.]], [.25, -2.]
    y = [affine(v, expected_w, expected_b) for v in x]
    w, b = fit_affine(x, y, alpha=1e-8)
    heldout = [.7, -1.2]
    predicted = affine(heldout, w, b)
    expected = affine(heldout, expected_w, expected_b)
    if max(abs(a-b) for a, b in zip(predicted, expected)) > 1e-6:
        raise AssertionError("Affine fit failed held-out reconstruction")
    for position in (0, 1, 31, 32768):
        v = [1., 2., -3., 4.]
        recovered = rotate(rotate(v, position), position, inverse=True)
        if max(abs(a-b) for a, b in zip(v, recovered)) > 1e-10:
            raise AssertionError("RoPE inverse failed")
    rows = [trace(arm, case, repeat) for repeat in range(2) for case in cases for arm in ARMS]
    for row in rows:
        validate_trace(row)
    summary = write_report(output, rows, {"empirical": False, "uses_models": False, "uses_network": False,
                                          "uses_gpu": False, "timings_are_fixture_constants": True,
                                          "checks": ["held-out affine reconstruction", "RoPE inverse at long positions", "split isolation", "all five evidence arms", "paired report generation"]})
    if summary["status"] != "smoke_only_no_thesis_claim" or summary["paired_quality"]["examples"] != 2:
        raise AssertionError("Synthetic evidence mislabeled or repeated samples inflated")
    return summary
