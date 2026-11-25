# Larger mapper-training and development experiment

**Completed corrective job:** `sova-kv-full-20261118-132341-7daf`, submitted after **91 integrated tests passed**. [Final results](reports/sova-kv-full-20261118-132341-7daf/RESULTS.md). Training completed 400 updates; validation selected checkpoint 200. All 1,037 development decisions completed: neural 50.82% versus native 59.11%, near the always-A 50.05% baseline. Contract timing median paired cold speedup was 0.953×; synthetic longer prompts were approximately 1.06–1.07×. Acceptance is not established. The worker stopped; estimated compute $0.78485, excluding extras. Receipt and previous failure evidence are preserved below.

Fast live status, without downloading weights or checkpoint files:

```bash
python3 cloud/full_collect.py --receipt reports/sova-kv-full-20261118-132341-7daf/submission.json --status-only --progress
```

18 November 2026. First larger job: **`sova-kv-full-20261118-131018-22a9`**.
It stopped after **84 completed optimizer updates**, before update 85, because FP16 gradients became nonfinite and the guard raised before GradScaler could adapt. No development evaluation was reached. Checkpoints at steps 0 and 50, all training logs and the failure record are retained. Its 360 SageMaker billable seconds imply approximately **$0.1515 compute**, excluding extras. A corrective bounded retry is being prepared; this is not a completed model result.

The numerical fix retries the same token window at a lower loss scale, records skipped overflow attempts separately, and requires finite gradients for an actual optimizer update. Persistent overflow, invalid gradient norm or exhausted time budget still stops the worker. This follows the [PyTorch 2.5.1 GradScaler step-skipping contract](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/amp_examples.rst). A retry restarts from the ridge initialization with the same seed; it does not claim optimizer-state resumption from step 50.

- [Submission receipt](reports/sova-kv-full-20261118-131018-22a9/submission.json)
- [Current full-suite evidence: 91 tests, zero skips/failures/errors](reports/smoke-full-v1/test-results.json)
- [Frozen prepared data manifest](datasets/prepared-v1/manifest.json)
- [Pre-launch plan](reports/full-plan-v1.json)

## What is trained

The public pinned Qwen3-0.6B source and Qwen3-1.7B target remain frozen. Our approximately 11.13M-parameter cache mapper is trained for up to 400 optimizer updates, using 1,024-token prefixes and eight continuation positions. It starts with an independently fitted per-head ridge map and adds a nonlinear cross-head residual reading three neighboring source layers. Those layers are chosen by fixed depth proximity, not a learned top-k layer search. Key RMS normalization and target RoPE are applied to mapped keys.

The objective combines normalized target-cache reconstruction and full-vocabulary target-logit KL. CUDA loss scaling protects small gradients through FP16 cached states. Training windows do not assume a cropped contract retains the original gold answer. No gold-label fine-tuning of either base model is performed.

The selected mapper minimizes the objective on a separate mapper-validation split. Training/selection document IDs and context hashes are recorded. The selected saved artifact is reloaded before evaluation. Nonfinite caches/gradients and incompatible checkpoint/tokenizer hashes fail explicitly.

## Frozen data and prospective scoring

| Role | Documents | QA records |
|---|---:|---:|
| Mapper training, official train only | 64 | 1,088 |
| Mapper validation, official train only | 16 | 272 |
| Development evaluation | 61 | 1,037 |

Training deduplicates documents before sampling context windows; 17 questions do not become 17 independent contracts. Official test files are excluded. The development set was already used in the earlier pilot, so it is not a final untouched test.

The main comparison covers source-only, native target, the newly fitted ridge baseline, and our trained nonlinear mapper. The decision metric is argmax over single-token A/B/C logits, with full-vocabulary option probability mass reported. This is a prospective choice-scoring task and does not prove free-generation quality. Errors and over-limit examples stay in the denominator. Class distributions, majority baseline, per-class/macro scores and paired document-cluster intervals are reported.

Unconstrained greedy generation is measured separately on one seeded question per development document, with a 16-token output budget. Strict one-letter correctness, invalid formatting and truncation remain separate from prefix diagnostics. The original pilot's scores are preserved.

## Timing and memory

- Five inference arms on a correctness-independent, length-stratified sample of up to 24 development examples, five repetitions each.
- Additional synthetic timing controls near 8K/16K/32K, ten repetitions per arm where native context, memory and time budget allow. Exact token counts are logged; these do not establish long-context real-task quality.
- Cold transfer includes source prefill, mapper/movement/import, target tail and first-token selection. Warm-cache timings remain separate.
- GPU peak allocated/reserved memory and component timings are retained. Both models/mappers reside in memory, so this is not native-only serving memory or a production cost comparison.
- No production vLLM throughput, HTTP latency or vendor-specific hybrid-model claim is established by this reference HF experiment.

## Compute policy and collection

The user requested parallel preparation and launching the next experiment. Launch command:

```bash
python3 cloud/full_submit.py --submit
```

One `ml.g5.2xlarge` in `us-east-2`; digest-pinned AWS Torch 2.5.1 image. Worker runtime limit 7,200 seconds, process timeout 6,900 seconds; mapper training reserves time for evaluation/export. Public rate observed $1.515/hour, giving approximately $3.03 for two worker hours. A corrective job must include `--prior-compute-usd 0.1515`, so prior compute plus the new runtime-priced estimate stays below the $3.25 compute guard. Storage, transfer, billing overhead and tax are extra; this is not a total invoice ceiling. No endpoint is created. There are no automatic job retries.

S3 root:

```text
s3://sova-finance1-artifacts-use2-169970912263/kv-transfer/sova-kv-full-20261118-131018-22a9/
  input/        Frozen code/data bundle
  checkpoints/  Managed checkpoint snapshots
  live/         Progress, training logs, mapper and evaluation artifacts
  output/       SageMaker final artifacts
```

Read status and collect existing evidence without starting compute:

```bash
python3 cloud/full_collect.py --receipt reports/sova-kv-full-20261118-131018-22a9/submission.json --logs
```

The collector estimates compute using reported billable seconds, not a fabricated invoice. Checkpoint restore currently supports evaluation of a selected mapper; exact optimizer-state training resumption is not implemented. The worker stops after completion or its limit. Results must be inspected before claiming quality retention, speedup or savings.
