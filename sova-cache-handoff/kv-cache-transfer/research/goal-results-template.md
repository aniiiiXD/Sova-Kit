# Memory-recovery candidate: results report template

This is a report specification, **not a result**. No scores, completed updates or memory improvements are asserted here. Populate it only from the final submission receipt, collected artifacts and raw rows; retain failures alongside the final run.

## Run identity and completion

| Field | Authoritative source | Report rule |
|---|---|---|
| Job name, ARN, AWS region, instance, image digest | `submission.json` | Use the actual accepted job, not a plan-only name. |
| Final job status, start/end, failure reason | `status.json` / SageMaker description | A process reaching its final print statement is not evidence that every benchmark item ran. |
| Checkpoint path and artifact hash | Selected `training/best/manifest.json`, weights checksum, S3 receipt | Report whether the selected checkpoint is initialization or a trained update. |
| Models/tokenizer identities | `environment.json`, model identity manifests | Qwen3-0.6B source and Qwen3-1.7B target; record exact revisions/hashes. |
| Numerical path | Worker config and fresh control report | FP32 math attention, query block 256, and actual precision settings; do not describe this as the earlier FP16 path. |
| Training/evaluation prompt caps | Worker config and reference `config.json` | Training 8,192; prospective evaluation 9,000, with the exact frozen 100 QuALITY questions retained. |
| Saved-tensor memory policy | Memory-policy artifact and tested source | Name the actual selective activation offload policy, whether memory is pageable/pinned, and whether frozen-weight copies are skipped. Do not infer it merely from a command-line flag. |

Completion verdict must separately state:

1. Fresh numerical controls passed or failed.
2. Worst-case backward/optimizer-memory preflight passed or failed.
3. Requested versus completed optimizer updates.
4. QuALITY execution completeness for every arm.
5. HellaSwag execution completeness for every arm.
6. Quality-retention and speed/cost acceptance, which remain unestablished unless the corresponding evidence supports them.

## Training and memory record

The next proposed run requests **64 updates** with a checkpoint/validation interval of 64 and an 800-second training allowance. It is a prospectively revised pilot budget, not completion of the historical 400-update or 100-update plans. The full worker limit is proposed as 2,260 seconds, within a 2,560-second SageMaker runtime scope. Replace these proposed values with the accepted config if they change before submission.

| Quantity to report | Exact artifact/field |
|---|---|
| Requested/completed updates, attempted steps, skipped overflows | `training/training-summary.json`: `requested_steps`, `completed_steps`/`completed_updates`, `forward_backward_attempts`, `overflow_attempts`; inspect log if a field is absent |
| Unique documents and questions actually consumed | `training/question-consumption.jsonl` and training summary; do not substitute prepared-record counts |
| Native validation selection and accuracy | `native-control.json`: exact IDs, decision accuracy, predeclared floor and oracle gate |
| Selected mapper update, selection loss | `training-summary.json` and selected manifest: `best_step`, `best_validation_loss`, `selected_at_step` |
| Training/validation article IDs | Actual `training-split.json` and `qa-protocol.json` |
| Initialization reuse | Immutable step-zero artifact hash, its original training article registry and zero-update status; it is initialization reuse, not restored optimizer-state training |
| Worst-case memory probe inputs | Preflight artifact: longest prefix and largest native-query attention footprint, their exact IDs/token counts |
| Preflight optimizer state materialized | Explicit preflight field; a forward-only smoke does not establish backward/Adam memory fit |
| Peak allocated/reserved GPU memory and device total | Preflight/training memory fields, in MiB/GiB with phase named |
| Host-memory peak | Actual recorded RSS/container telemetry if available; otherwise `not measured` |
| Memory or numerical failure | Original traceback/exit status and stage; do not replace it with a later passing test |

**Partition caution:** the actual repair worker shuffles article IDs and chooses the **first 30 for validation**. The earlier all-question tokenizer preflight used a different first-120-training partition for its split totals. Its per-question boundary/length verification remains valid, but its eligible train/validation totals must not be reported as this worker's actual split. Use the actual run artifacts.

The historical `170048` process kill before its first memory probe yielded no Python traceback. Unless host/kernel/container OOM telemetry is collected, label host-memory exhaustion as the diagnosis being tested, rather than a directly observed kernel OOM event.

## QuALITY — unchanged frozen 100 questions

Protocol: custom chat prompt with mapped document prefix and native query suffix, restricted first-token A/B/C/D decision scoring, plus separate unconstrained generation with the predeclared 16-token budget. This is not an exact private-vendor prompt/scorer reproduction.

Fill these rows directly from `quality-summary.json` and `quality/rows.jsonl`:

| Arm | Planned | Executed `ok` | Errors | Over limit | Budget not run | Decision correct / 100 | Strict generation correct / 100 | Valid exact-letter outputs / 100 | Length-capped outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected |
| Native target | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected |
| Mapped target | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected |

Corresponding summary keys are `planned_items`, `completed_items`, `errors`, `over_limit`, `not_run_time_budget`, `decision_correct`, `free_generation_strict_correct`, and `free_generation_format_valid`. The metric values are fractions; obtain integer success counts from raw rows rather than rounding percentages back into invented counts. Truncation is recorded per row as `free_generation_truncated`.

Also include:

- Native-minus/mapper paired decision comparison, with the saved document-cluster interval and sign convention stated explicitly.
- Gold class distribution, predicted class distribution, native/mapper option agreement, and total A/B/C/D probability mass. A high conditional option probability does not establish calibrated confidence.
- At least one actual output example for each failure category, with its ID and no cherry-picked replacement of the headline metric.
- The evaluated context range and actual error/limit outcomes. Tokenizer preflight establishes a maximum of 8,496 tokens on the frozen sample; it does not establish runtime success.

Do not turn label-prefix diagnostics into a replacement strict-generation headline after seeing results. A correct first-token decision and an invalid or truncated free-generation answer are different observations.

## HellaSwag — unchanged frozen 1,000 validation records

Protocol: zero-shot raw completion, no chat/option letters, four complete candidate-ending log-likelihoods, and `acc_norm` divided by Python character length of the processed ending, excluding the delimiter. The frozen 4,000 token-boundary pairs passed offline checks; the longest complete candidate is 137 tokens. This establishes tokenization compatibility, not measured model accuracy.

| Arm | Planned | Executed `ok` | Errors | Over limit | Budget not run | Raw `acc` correct / 1,000 | Character-normalized `acc_norm` correct / 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected |
| Native target | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected |
| Mapped target | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected | Not collected |

Use `hellaswag-summary.json` fields and raw rows. Report both raw and normalized metrics rather than whichever is higher. Preserve all 1,000 planned items in the denominator, with errors and budget-not-run incorrect. Use `source_id` grouping for the paired interval because some questions share a source video/document. It is an out-of-domain mechanism check for this QuALITY-trained mapper, not a FineWeb-calibrated reproduction of a published mapper.

## Timing, memory and cost

For QuALITY, describe `cold_first_token_ms` as synchronized in-process cold model execution through first-token selection. State that it includes source prefill and mapping/import for the mapped arm. Reference `elapsed_ms` also includes generation and diagnostic-scoring work; it is not interchangeable with pure GPU execution or HTTP latency.

For HellaSwag, use full per-item `elapsed_ms` for four-ending evaluation work. The cold prefix-preparation time alone is not complete HellaSwag latency. Report successful paired latency sample counts before any speed ratio. Quality loss must remain visible next to a faster timing result.

If the reference evaluator did not record per-row peak memory, say so and label any quoted memory as training/preflight memory instead. Both base models and the mapper are resident; that is not a native-only production-serving comparison.

Cost reporting must distinguish:

- Prior reported billable compute estimate.
- Accepted job's public-rate/runtime estimate and guard.
- Final SageMaker `BillableTimeInSeconds × observed hourly rate / 3,600` estimate.
- Storage, transfers, taxes and any other unmeasured charges; the estimate is not an invoice.

The proposed current arithmetic is prior estimated compute **$2.1706583335**, plus a 2,560-second runtime-priced estimate **$1.0773333333**, totaling approximately **$3.2479916668** against a **$3.25 compute guard**. These are planning figures, not actual charges or a total-invoice ceiling. Replace them with the final receipt and billable seconds before claiming spent cost.

## Evaluation-time forecast, not a completion promise

The previously estimated combined benchmark cost is roughly **1,050–1,600 worker seconds**: QuALITY about 650–900, HellaSwag about 400–700. Therefore **1,200–1,400 seconds is plausible but tight**. The new 64-update pilot and reuse of a verified untrained step-zero snapshot should remove repeated ridge fitting and leave more evaluation time, but saved-tensor offload can slow backward and has not yet been measured successfully on the GPU.

Already implemented: QuALITY reuses its cold cache for decision plus generation; HellaSwag shares one prefix across the four independently cloned candidates. There is no duplicate decision/generation prefill left to eliminate. Cache-lifetime cleanup avoids holding the preceding variant's full GPU cache during the next prefill without changing numerical operations.

A 900-second QuALITY cap must not be interpreted as a guarantee that HellaSwag also finishes. Preserve phase timings, completion counts and partial rows. If the budget expires, the honest result is partial evaluation with retained evidence; the historical record must not be overwritten or described as a full benchmark.

## Final conclusion fields

The final user-facing conclusion should answer these independently:

1. Did the memory fix pass the real longest-case backward/optimizer preflight and avoid both GPU OOM and host process termination?
2. How many actual updates completed, and which checkpoint was selected using training-domain validation?
3. Did all 100 QuALITY and 1,000 HellaSwag records execute for all three arms? If not, give exact remaining/error counts.
4. What are native and mapped decision/ending scores, their paired difference, and free-generation failure rates?
5. Is there evidence of useful quality-preserving speed or memory improvement, or only a completed negative experiment?

Do not fill missing values with earlier ContractNLI scores, planned item counts, CPU smoke-test outcomes or another model pair's published results.
