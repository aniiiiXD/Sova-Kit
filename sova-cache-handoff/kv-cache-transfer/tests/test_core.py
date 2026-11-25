import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from kvverify.data import Example, assert_disjoint, check_heldout, load_examples, validate_pair
from kvverify.evidence import ARMS, paired_quality, summarize, validate_trace, write_report
from kvverify.mathops import affine, fit_affine, rotate, solve
from kvverify.smoke import example, run, trace


class MathTests(unittest.TestCase):
    def test_ridge_generalizes_to_unseen_vector(self):
        x = [[-2., 1.], [-1., 3.], [0., -1.], [1., 2.], [2., -3.]]
        w, b = [[2., -.5], [1., 3.]], [.25, -2.]
        y = [affine(v, w, b) for v in x]
        fitted, bias = fit_affine(x, y, alpha=1e-8)
        for a, z in zip(affine([.7, -1.2], fitted, bias), affine([.7, -1.2], w, b)):
            self.assertAlmostEqual(a, z, places=6)

    def test_ridge_handles_rank_deficiency_with_regularization(self):
        w, b = fit_affine([[1., 1.], [2., 2.], [3., 3.]], [[2.], [4.], [6.]], alpha=.01)
        self.assertAlmostEqual(affine([2., 2.], w, b)[0], 4., places=6)

    def test_ridge_rejects_negative_regularization(self):
        with self.assertRaises(ValueError):
            fit_affine([[1.], [2.]], [[1.], [2.]], alpha=-1)

    def test_solver_pivots(self):
        result = solve([[0., 1.], [2., 3.]], [[2.], [8.]])
        self.assertAlmostEqual(result[0][0], 1.)
        self.assertAlmostEqual(result[1][0], 2.)

    def test_solver_rejects_singular_and_nonfinite(self):
        for a in ([[1., 1.], [1., 1.]], [[float("nan"), 0.], [0., 1.]]):
            with self.assertRaises(ValueError):
                solve(a, [[1.], [2.]])

    def test_rope_roundtrip_long_positions(self):
        v = [1., -2., 3., .2, -1., 4., 5., -6.]
        for position in (0, 1, 31, 32768, 1_000_000):
            out = rotate(rotate(v, position), position, inverse=True)
            for a, b in zip(out, v):
                self.assertAlmostEqual(a, b, places=10)

    def test_wrong_position_changes_keys(self):
        v = [1., 2., 3., 4.]
        self.assertNotEqual(rotate(v, 5), rotate(v, 6))


class DataTests(unittest.TestCase):
    def test_reference_not_in_prompt(self):
        row = example()
        self.assertNotIn('"answer"', row.prompt())
        self.assertIn("A. 25%", row.prompt())

    def test_split_leakage_document_id(self):
        with self.assertRaises(ValueError):
            assert_disjoint([example("cal", "same", "cal", "calibration")], [example("eval", "same", "different")])

    def test_split_leakage_whitespace_copy(self):
        with self.assertRaises(ValueError):
            assert_disjoint([example("cal", "a", "the  same context", "calibration")], [example("eval", "b", "the same\ncontext")])

    def test_mapper_must_have_leakage_registry(self):
        with self.assertRaises(ValueError):
            check_heldout({}, [example()])

    def test_bad_answer_rejected(self):
        row = asdict(example());row["answer"] = "Z"
        with self.assertRaises(ValueError):
            Example.parse(row)

    def test_duplicate_context_must_have_same_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"eval.jsonl"
            rows = [asdict(example()), asdict(example("eval-2", "different-cluster"))]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaises(ValueError):
                load_examples(path, "eval")

    def test_hybrid_geometry_rejected(self):
        config = {"model_type": "qwen3_5", "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 28}
        with self.assertRaises(ValueError):
            validate_pair(config, config)

    def test_scaled_rope_rejected(self):
        config = {"model_type": "qwen3", "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 28, "rope_scaling": {"type": "dynamic"}}
        with self.assertRaises(ValueError):
            validate_pair(config, config)


class EvidenceTests(unittest.TestCase):
    def test_cold_ttft_cannot_omit_source_prefill(self):
        row = trace();row["ttft_ms"] = .75
        with self.assertRaises(ValueError):
            validate_trace(row)

    def test_warm_ttft_may_exclude_explicit_source_setup(self):
        validate_trace(trace("transfer_warm"))

    def test_silent_fallback_rejected(self):
        row = trace();row["fallback"] = True
        with self.assertRaises(ValueError):
            validate_trace(row)

    def test_full_target_recompute_rejected_as_transfer_hit(self):
        row = trace();row["target_prefill_tokens"] = 8
        with self.assertRaises(ValueError):
            validate_trace(row)

    def test_nan_timing_rejected(self):
        row = trace();row["ttft_ms"] = float("nan")
        with self.assertRaises(ValueError):
            validate_trace(row)

    def test_duplicate_attempt_rejected(self):
        row = trace()
        with self.assertRaises(ValueError):
            summarize([row, copy.deepcopy(row)])

    def test_unpaired_examples_rejected(self):
        with self.assertRaises(ValueError):
            summarize([trace()])

    def test_mismatched_prompts_rejected(self):
        a, b = trace("native_cold"), trace();b["prompt_hash"] = "another"
        with self.assertRaises(ValueError):
            summarize([a, b])

    def test_repeated_timing_rows_are_not_extra_quality_examples(self):
        rows = [trace(arm, repeat=r) for r in range(5) for arm in ARMS]
        result = summarize(rows)
        self.assertEqual(result["paired_quality"]["examples"], 1)
        self.assertEqual(result["successful_latency_pairs"], 5)

    def test_errors_count_as_wrong(self):
        rows = [trace("native_cold"), trace()]
        rows[1].update(status="error", correct=False)
        result = summarize(rows)
        self.assertEqual(result["paired_quality"]["difference"], -1)
        self.assertEqual(result["arms"]["transfer_cold"]["accuracy"], 0)
        self.assertEqual(result["arms"]["transfer_cold"]["errors_or_over_limit"], 1)

    def test_synthetic_data_never_establishes_thesis(self):
        result = summarize([trace(arm) for arm in ARMS])
        self.assertEqual(result["acceptance"], "not_established")
        self.assertEqual(result["status"], "smoke_only_no_thesis_claim")

    def test_mixed_fixture_model_evidence_rejected(self):
        a, b = trace("native_cold"), trace();b["evidence_kind"] = "hf_model"
        with self.assertRaises(ValueError):
            summarize([a, b])

    def test_offline_integration_smoke_writes_labeled_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"smoke"
            result = run(output)
            self.assertEqual(result["paired_quality"]["examples"], 2)
            saved = json.loads((output/"summary.json").read_text())
            self.assertFalse(saved["metadata"]["empirical"])
            self.assertTrue((output/"traces.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
