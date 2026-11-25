# Completed goal: memory-safe training and measured evaluation

**Completed 19 November 2026 IST.** Memory stress checks and 48 actual finite
updates succeeded; both fixed benchmarks are fully executed across all three
variants, with zero final errors, limits or skipped rows. Independent raw-output
audit passed. Both SageMaker workers are terminal and released.

| Primary metric | Source | Native target | Mapped |
|---|---:|---:|---:|
| QuALITY decisions, 100 items | 47% | 59% | 26% |
| HellaSwag `acc_norm`, 1,000 items | 47.3% | 61.2% | 28.5% |

The memory repair succeeded; mapper quality remains poor. This was a bounded
pilot, not 100 completed optimizer updates or a production-ready result.
Tracked runtime-compute estimate $3.69997, under the $5 cap, before extras.

**Final [results report](reports/sova-kv-eval-resume-20261118-183814-e659/RESULTS.md),
[machine-readable results](reports/sova-kv-eval-resume-20261118-183814-e659/RESULTS.json),
and [independent audit](research/final-results-audit.md).**

The sections below preserve the execution plan and intermediate history.

The user explicitly requested on 18 November 2026: fix the failure, use parallel
agents, execute the work, provide updates about once per minute, and continue
until results exist. Active goal covers the training-memory fix and complete
measured reference evaluation; it does not promise a positive quality result.

Final job: `sova-kv-eval-resume-20261118-183814-e659`, SageMaker `us-east-2`.
Receipt: [submission.json](reports/sova-kv-eval-resume-20261118-183814-e659/submission.json).
Goal monitoring history: `reports/sova-kv-eval-resume-20261118-183814-e659/goal-status-history.jsonl`.

The main repaired worker `173137-4f04` COMPLETED: both real backward stress
cases (8,043 and 7,418 prompt tokens) passed; 48 finite optimizer updates fit its
training window; selected step 48; validation objective 20.88625 → 3.99507 on
the same 30 validation questions. GPU peak across the recorded run 14,289.27 MiB,
host peak RSS 25,447.63 MiB. No memory error recurred. The 100-update target was
not reached; report 48 actual, not 100.

HellaSwag is complete, all 1,000 items per variant, zero errors/limits/timeouts.
Normalized accuracy: source 47.3%, native 61.2%, mapper 28.5%. This is a negative
mapper-quality result despite the successful memory repair.

QuALITY reached its separate 900-second phase budget: 89 complete three-way
questions, one additional source-only result, 10 wholly unexecuted IDs. No
runtime errors or over-limit cases. Current evaluation-only continuation reruns
all three variants for the 11 incomplete IDs, selected by execution status,
never correctness. Completed parent triplets remain unchanged. Same fixed
checkpoint, data, numerical and generation protocol; per-row provenance is
preserved. Merged cross-job latency headlines are disabled. No further training.

Continuation tests: 134 scoped checks passed with immutable hashes. One initial
continuation submit was rejected for an underscore in the AWS job name before
any worker was created; naming was corrected and tested for every mode.
Maximum new compute $0.37875 (900-second job cap); prior tracked recovery-phase
compute $3.54005; cumulative runtime-limit estimate $3.9188 under the $5 cap.
Parent main worker billed 3,254 seconds, estimated $1.36939167, and is released.
The earlier cost/protocol sections below preserve main-run planning history.

The first memory-recovery attempt `170048-8273` failed after 648 billed seconds
($0.2727). SageMaker reported insufficient instance memory, without a Python
traceback or completed preflight report. The abrupt exit supports a host/container
kill; the exact peak was not captured. Generic pinned CPU offload was an additional
memory risk. The current retry uses selective **pageable activation** offload,
keeping frozen parameter storages (including views/transposes) resident on GPU.
Base models freeze before storage classification. It reuses the exact verified
untrained QuALITY step-zero initializer, with a fresh optimizer and repeated
preflight/validation. The failed run's two unsaved updates are not reused.

Current scoped evidence: **123 tests passed**, zero failures/errors/skips:
[test results](reports/smoke-selective-resume-v4/test-results.json). The older
113-test evidence below belongs to the preceding memory-recovery attempt.

## Confirmed failure and repair

The previous job passed 16/16 FP32 native-cache controls and two optimizer
updates, then failed while backward recomputed the complete 28-layer mapper.
Each 7K-token FP32 KV cache is about 1.5 GiB; the old full source preparation
and cross-layer features created a much larger replay peak. This was a Python
CUDA OOM with a traceback, not an unexplained OS crash.

The repaired mapper checkpoints one target layer and 256 source-token positions
at a time, prepares only selected source slices, and uses absolute RoPE offsets.
Output/parameter-gradient parity is tested with nonzero offsets, uneven tails
and an active nonlinear residual. The frozen target's saved backward activations
are additionally placed in pageable CPU memory, excluding frozen weight storage.
Teacher/source no-grad computation does
not save backward activations. Model weights stay frozen on the GPU.

Before any optimizer update, real-GPU preflight performs a complete backward on
the longest eligible prefix and the largest prefix×query workload. It checks
finite gradients, unchanged mapper weights, empty optimizer state and frozen
base-model gradients, then restores RNG and clears gradients. It records GPU
allocated/reserved peaks and host peak RSS. This is the hardware fit test; tiny
CPU tests alone cannot prove A10G fit.

The evaluation runner also releases previous variant caches and decode outputs
before preparing the next variant. Two weak-reference regressions verify that
lifetime boundary without changing scores or timing definitions.

113 scoped offline tests passed, zero failures/errors/skips, with immutable
source hashes: [test evidence](reports/smoke-memory-safe-v1/test-results.json).
Three parallel agents implemented mapper repair, independently reviewed memory
correctness, and fixed evaluation lifetimes/budget planning. Root integrated
and submitted the job. No agent launched additional GPU work.

## Prospective pilot protocol

- Same pinned Qwen3-0.6B → Qwen3-1.7B, FP32 math attention, 256-query attention
  blocks, TF32/reduced-precision reductions disabled, full contexts retained.
- Training cap remains 8,192 tokens, all eligible question pools preserved.
  Target 100 optimizer updates in a 1,100-second training window including
  reused initialization/preflight/selection; actual consumed counts must be reported.
  Checkpoints at step 0 and 100 (or actual budget stop), with 30-second progress
  heartbeats. Validation is for mapper selection, separate from evaluation.
- Fixed QuALITY 100 and HellaSwag 1,000. Evaluation-only context cap 9,000 was
  declared before results so all 100 fixed QuALITY prompts fit; maximum is
  8,496. This does not alter training eligibility or truncate any document.
- QuALITY free-generation cap is prospectively 16 tokens, separate from its
  constrained-choice accuracy. Quality stage gets up to 900 seconds; HellaSwag
  gets remaining budget. HellaSwag scores every ending token with character-
  normalized likelihood. [Budget plan](research/memory-recovery-eval-plan.md).
- Errors/timeouts remain in denominators. A partial benchmark is not complete;
  inspect all planned/executed/error/budget counts before claiming results.

## Compute policy and monitoring

User's explicit fix/run/loop instruction authorizes this bounded goal work.
Local execution is short offline tests, packaging and read-only orchestration;
pretrained model work is cloud-only. One `ml.g5.2xlarge`, 3,600-second job cap,
3,300-second worker cap, S3 artifacts, automatic stop, no endpoint.

For the persistent repair goal, the assistant explicitly notified the user of an
expanded **$5 cumulative runtime-compute cap** before submitting the retry.
The historical launcher retains $3.25; a separately tested explicit recovery-goal
guard handles $5. An initial local submit attempt stopped at the old budget helper
before creating any GPU and incurred no worker compute.

Maximum new compute $1.515; prior recovery-phase compute $2.17065833;
cumulative runtime-limit estimate $3.68565833 under the explicit $5 goal guard.
Storage, transfer, tax and other extras are excluded. Actual billed time must
be reported. The read-only monitoring loop polls every 45 seconds, does not
submit or retry jobs, and exits when the job terminates.

```bash
python3 scripts/watch_goal.py --receipt reports/sova-kv-eval-resume-20261118-183814-e659/submission.json --interval 45 --max-polls 35
```
