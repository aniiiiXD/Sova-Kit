# SOVA cross-model KV-cache research

## Current thesis and plan — 19 November 2026

Memory recovery completed; quality retention failed. The selected mapper scored **26% versus native 59% on QuALITY**, and **28.5% versus native 61.2% on HellaSwag normalized accuracy**. All planned reference rows executed. The workers completed; no endpoint was deployed.

Follow-up diagnostic: [v2 completed24/24 articles](../reports/state-diagnostic-v2/RESULTS.md), with142offline tests and all native-cache/readout controls passing. RMS off improves the untrained initializer's distribution match, but worsens the existing trained mapper on23/24documents by KL. Trained RMS-on/off/native correctness:11/8/13 of24 on this already-used validation panel. Both K/V errors remain; no serving win established. Worker terminated; v2 estimated runtime compute$0.39853, or$0.52015 including the [failed v1 probe](../reports/state-diagnostic-v1/RESULTS.md). No further job submitted.

- [Thesis v1 and staged plan](thesis-v1/THESIS.md): conditional opportunity, evidence, competing explanations, economics and acceptance gates.
- [First experiment card](thesis-v1/first-experiment.md): diagnose key normalization, addressing, value readout and cumulative drift before more training. Proposed, not executed.
- [Current implementation audit](thesis-v1/implementation-audit.md): exact architecture, loss, geometry and falsifiable interventions.
- [Primary-body literature review](thesis-v1/literature-review.md): five methods, timing boundaries, counterevidence and prior art.
- [Experimental design](thesis-v1/experimental-design.md): document-disjoint selection/final evaluation, uncertainty and serving baselines.
- [Final measured results](../reports/sova-kv-eval-resume-20261118-183814-e659/RESULTS.md) and [independent audit](final-results-audit.md).

The existing 100 QuALITY and 1,000 HellaSwag examples have now been observed. Preserve them as development/regression evidence; reserve untouched data for future final-test claims. No new model jobs were launched for this thesis.

## Historical pack below

The following text is an 18 November snapshot and execution history. Pending-evaluation language and older ContractNLI numbers below are superseded by the current evidence above; old prompt-boundary defects are not current diagnoses.

Prepared 18 November 2026 by parallel framework, method-audit and career-positioning reviews. Sources and evidence boundaries are in the individual reports. These are research/draft artifacts; no public post, recruiter message or job application was sent.

## Read these

- [Precision/backend diagnostic](precision-diagnostic.md): completed **16/16 FP32 numerical controls**, unchanged gate, full context, about 9 GiB peak GPU allocation; [results](../reports/sova-kv-precision-20261118-161633-5f6d/RESULTS.md). FP16 paths still fail the strict gate. Repaired-mapper quality remains to be evaluated.
- [Recovery run and completed diagnostic](recovery-run-20261118.md): original 45/48 raw-logit gate failure, probability audit, fresh in-worker safety control, repaired task training and bounded benchmark job.
- [Next experiment and prepared diagnostics](next-experiment.md): current failure, benchmark choice, frozen reference data and the controlled repair sequence.

- [Training guide](training-guide.md): every pipeline stage, trainable/frozen components, data, losses, optimizer, checkpoints, metrics and the live-status command.
- [Serving frameworks](serving-frameworks.md): vLLM, SGLang, LMCache, TensorRT-LLM and Dynamo, concrete integration hooks, runtime compatibility and deployment sequence.
- [Methods and ablations](methods-and-ablations.md): primary-paper comparison, weaknesses in our current candidate and controlled next experiments.
- [Job pitch](job-pitch.md): spoken pitch, CV bullets, LinkedIn draft, GitHub case-study outline, interview questions and official job-posting examples.
- [Completed results](../reports/sova-kv-full-20261118-132341-7daf/RESULTS.md): full quality, timing and compute evidence; the candidate does not pass acceptance.
- [Failure analysis](failure-analysis-v2.md): actual prediction collapse, question coverage and length/hypothesis diagnostics.
- [Architecture recovery plan](architecture-recovery-plan.md): handoff-boundary mismatch, stronger baselines and controlled repairs.
- [Original benchmark protocols](original-benchmark-protocol.md): vendor versus paper campaigns, official QuALITY/HellaSwag reference data and scoring.

## Project positioning

**SOVA — a reproducible cross-model KV-cache transfer experiment.**

Suggested concise pitch:

> I’m building a model-state transfer system that lets a small model read a document and prepares a larger model to continue from its translated cache. I built the PyTorch cache-import path, correctness controls, mapper-training pipeline, and bounded SageMaker jobs with S3 evidence. I’m measuring the quality/latency tradeoff, including failed baselines and numerical failures, before claiming a serving speedup.

This is credible ML systems / inference / research-engineering work. Current evidence does not establish a production API, a novel invention of cross-model transfer, successful quality retention, 100B-model training, or a 10× serving result. The 91-test suite validates software/tiny-model contracts; it does not substitute for pretrained-model evaluation.

## Practical direction from the research

Keep the HF runtime as the inspectable numerical reference. After a quality-passing candidate exists, build one custom vLLM V1 connector in a separate pinned serving container. Treat storage/transport frameworks as infrastructure, not learned translators. Add orchestration or multiple serving engines only when measured needs justify them.

The most important method comparisons are a stronger cross-head/selected-layer ridge reference, a normalization-only control, alternative functional objectives, and document-then-question sampling. The historical ContractNLI trainer retained one formatted question per contract; it did not train on all 1,088 question variants. That version remains a baseline. The new `qa_training` path retains question pools and distills teacher answers after the full native query suffix; its GPU results are pending.

## Current run and evidence

**Goal completed: [full measured results](../reports/sova-kv-eval-resume-20261118-183814-e659/RESULTS.md).**
Memory failure repaired, 48 finite updates, full QuALITY 100 and HellaSwag 1,000
per variant, zero final execution errors/limits/skips, independent raw-output
audit passed, workers released. QuALITY source/native/mapped: 47/59/26%;
HellaSwag normalized: 47.3/61.2/28.5%. Quality retention failed; no serving
acceleration claim. Tracked estimated compute $3.69997 before extras.

**Execution history: [memory-safe training and evaluation](../GOAL_RUN.md).**
Main job `sova-kv-repair-20261118-173137-4f04` used per-layer/token-block
backward replay, selective pageable activation offload that excludes frozen
weight storage, verified untrained step-zero reuse, worst-case backward checks
and 30-second training heartbeats. It completed 48 of 100 requested updates;
the evaluation-only continuation completed remaining QuALITY execution.
123 scoped main-source checks and 134 continuation-source checks passed
(overlapping suites, not additive). The preceding
`170048-8273` job exited abruptly with SageMaker's instance-memory error; its
generic pinned CPU offload was replaced. See the goal log for costs and evidence.

**Previous recovery job: `sova-kv-repair-20261118-162905-150e`** passed 16/16 FP32
controls and two updates, then failed in whole-mapper backward replay with CUDA
OOM. Worker released; estimated compute $0.34802917. The new goal run above
addresses this separate training-memory failure; no repaired-mapper accuracy yet.
[Run details and live-status command](../reports/sova-kv-repair-20261118-162905-150e/RUN.md).

The task-aligned job `sova-kv-repair-20261118-143720-3fef` stopped before training: 13/16 fresh native-cache probability controls passed; all 16 agreed on the first token, but three exceeded the declared distribution tolerances. The worker is released, with estimated runtime compute $0.123725. Its scoped smoke suite passed 101 tests; no repaired-mapper accuracy exists yet. [Final status and evidence](../reports/sova-kv-repair-20261118-143720-3fef/RUN.md). [Recovery protocol](recovery-run-20261118.md) records the changed numerical criterion and preserves the failed historical gate. The later precision diagnostic and active FP32 recovery job are recorded above.

The completed historical job is `sova-kv-full-20261118-132341-7daf`. See [FULL_RUN.md](../FULL_RUN.md) and its dated receipt/status files. Research reports are snapshots; use collected training/evaluation artifacts before updating a CV or publishing a result. A lower validation reconstruction/distillation objective is not a downstream accuracy result.

Historical ContractNLI result:400 updates completed; all1,037 development decisions evaluated. Neural50.82% versus native59.11%, near always-A50.05%; median paired cold contract speedup0.953×. This precedes the completed QuALITY/HellaSwag experiment; it is not the latest result.

The useful portfolio package is one clear architecture diagram, a defensible experiment table, reproducible commands, raw traces, a failure analysis, and a small demo when the quality gate passes. Lead with the engineering judgment and exact measured scope.
