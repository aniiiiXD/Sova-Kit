# Presenting SOVA's cache-transfer work in job applications

Evidence snapshot: **18 November 2026**. Draft language below reflects the saved artifacts, not an assumption that the latest GPU job succeeded. Recheck `FULL_RUN.md` and collected results before publishing an update.

**Verified update:** corrective job `sova-kv-full-20261118-132341-7daf` completed and stopped. The mapper completed **400 optimizer updates**, but **1,037-question decision accuracy was 50.82% versus 59.11% native**, near the 50.05% majority baseline. Contract timing median paired cold speedup was 0.953×; synthetic longer prompts gained approximately 6–7%. No quality-preserving acceleration is established. The failure analysis identified erased question diversity and an eight-token training versus one-token inference boundary mismatch. [Final results](../reports/sova-kv-full-20261118-132341-7daf/RESULTS.md), [training summary](../reports/sova-kv-full-20261118-132341-7daf/collected/training/training-summary.json).

## Position the work

**Project title:** SOVA — Cross-model KV-cache transfer experiments.

**One-line description:** Built a PyTorch experiment stack for smaller-model prefill, learned cache translation, and larger-model decoding, with correctness controls and reproducible GPU evaluation.

The strongest current story is **inference engineering plus empirical research discipline**: a custom runtime integration, a failed initial hypothesis, a numerical debugging loop, and an evaluation system that makes a misleading success difficult to report. It is a research prototype; a deployed production API and a quality-preserving speed improvement have not been established.

Adapt the emphasis to the role:

| Role direction | Lead with | Evidence to show | Gap to address next |
|---|---|---|---|
| ML systems / inference engineer | Cache extraction/import, cold versus warm paths, component timing and GPU memory | `hf_backend.py`, raw pilot traces, identity/zero-cache controls, bounded SageMaker worker | Production serving integration, concurrency and throughput measurements |
| Research engineer | Pair-specific mapper, teacher-logit objective, independent selection data, falsifiable baselines | Mapper implementation, failed ridge results, train/validation split manifest and retained failure logs | Completed larger evaluation, ablations and validation on another workload |
| GPU performance engineer | Attribution of source prefill, mapping and target-tail time; numerical failure analysis | Per-component traces, FP16 failure record and loss-scaling regression checks | Measured profiler traces, kernels, bandwidth/roofline analysis and multi-GPU execution; these are not demonstrated by this project yet |

## A 30-second spoken pitch

“I'm building a cross-model cache-transfer experiment in SOVA: a small Qwen model reads the context, then a learned mapper prepares a larger model's cache. I built the cache-import harness, correctness controls, and reproducible SageMaker jobs. I trained an 11M-parameter mapper and evaluated 1,037 questions. It lost 8.3 accuracy points. My audit found question-coverage problems and a training/inference handoff mismatch; the next experiments test how much those explain.”

## Three CV bullets supported now

- Built a PyTorch/Transformers KV-transfer harness for Qwen3-0.6B → Qwen3-1.7B, covering five cold/warm inference controls, synchronized first-token timing, GPU-memory traces and **91 passing software/integration tests**.
- Prepared **2,397 ContractNLI QA records across 141 documents** with separate mapper-training, mapper-validation and development roles; froze source/artifact hashes and retained errors and over-limit examples in evaluation denominators.
- Trained an approximately **11.13M-parameter** residual cache mapper for **400 optimizer updates** with frozen source/target models, using target-cache reconstruction, target-logit KL, separate validation selection, managed SageMaker/S3 checkpoints and explicit FP16 overflow handling.

These describe implemented work. Do not replace “implemented” with “achieved faster inference at the same accuracy” until the quality and speed gates both pass. The 400-update run and development evaluation completed; its quality/speed acceptance failed. Use first-person language only for work you can explain and defend, including how AI agents assisted implementation and how you reviewed their output. The coverage and boundary mismatches are confirmed; their causal contributions await controlled probes.

Architecture scope matters in a technical interview: the initial ridge map operates per head with fixed depth alignment; it is a deliberately limited baseline rather than a complete reproduction of a stronger published mapper. The nonlinear mapper reads fixed nearby source layers. The current trainer deduplicates to one representative QA per training document and samples token windows, so preparing 1,088 training records does not mean it optimizes separately over every question. These are concrete next ablations, not demonstrated capabilities.

## GitHub case-study outline

1. **Question and scope.** Can a small reader's state reduce a larger reader's prefill work? State the exact pair, dense-attention-only support, hardware, precision and context limits. Credit OneTriangle as inspiration; do not imply access to its private mapper or reproduction of its advertised hybrid pair.
2. **Architecture.** Document → small-model prefix cache → pair-specific mapper → target-cache import → one target tail token → autoregressive decode. Link the existing animated diagram and show which stages are included in cold TTFT.
3. **Controls and data.** Native cold, native warm, source-only, transfer cold, transfer warm; identity-cache parity and zero-cache sensitivity. Show the 64/16/61 document partition and explain why 17 questions about one contract are not 17 independent documents.
4. **Baseline outcome.** Present all pilot attempts and the 12-question sample size. Show 157.74 ms native cold versus 152.39 ms transfer cold, with the paired ratio 1.031×. Immediately show the generation failures and strict-scoring issue. This is a useful negative result, not an optimization headline.
5. **What changed and why.** Replace the simple ridge-only map with the implemented nonlinear residual mapper; describe target-cache reconstruction and full-vocabulary KL. Separate constrained decision accuracy from unconstrained generation compliance prospectively.
6. **Failure analysis.** Torch/SDPA image mismatch, output-format/length problem, and FP16 nonfinite gradients after 84 updates. Include symptoms, evidence, correction, regression test and remaining uncertainty. Lower validation loss does not prove downstream quality.
7. **Reproduction and artifacts.** Pinned checkpoint revisions and container digest, prepared-data manifest, test report, exact submission/collection commands, raw traces, checkpoint policy and reported billable seconds. State that reference-HF, batch-one TTFT is not HTTP serving latency or production throughput.
8. **Next acceptance gates.** Completed prospective quality comparison, document-cluster uncertainty, free-generation failure rates, useful long-context speed improvement and a native-only deployment cost comparison. Keep synthetic timing controls separate from real-task quality.

For a reviewer, prioritize one clean diagram, one honest results table and a runnable bounded experiment over a long list of model names. Do not link nonexistent hosted demos or claim the local branch is already published.

## LinkedIn draft

I'm building SOVA's cross-model KV-cache transfer experiment: let a smaller model read the context, translate its cached state, and let a larger frozen model continue.

The first pair is Qwen3-0.6B → Qwen3-1.7B. I built the PyTorch cache-import path, five cold/warm inference controls, and bounded SageMaker jobs that save their artifacts to S3.

The pilot was humbling. Across 12 contract questions, the paired latency ratio was only 1.03×, and the transferred answers were not useful. Native answers also exposed a mismatch between our exact-letter scoring contract and the generated format. I preserved those results instead of turning the timing difference into a success claim.

The next iteration added a nonlinear cache mapper, target-logit distillation, document-disjoint validation, and separate decision versus free-generation metrics. The trained pipeline passed 91 tests and completed 400 updates. Evaluation on 1,037 questions showed 50.8% mapped accuracy versus 59.1% native. The audit found collapsed predictions, a question-sampling defect and mismatched training/inference boundaries; controlled follow-up probes are being prepared.

What interests me is the boundary between model behavior and serving systems: cache compatibility, numerical stability, fair latency accounting, and evaluation that catches regressions.

I'm looking for ML systems and inference engineering opportunities where I can keep building and testing that boundary.

## Interview talking points

| Question | Concrete answer and artifact |
|---|---|
| What did you build rather than call through an API? | Actual K/V extraction, source RoPE removal, pair-specific mapping, target RoPE restoration, target-cache injection and an instrumented decode path. The standard inference API boundary does not provide the raw state operations this experiment needs. Show the code rather than claiming ownership of a provider's implementation. |
| What was your result? | The larger completed experiment scored 50.82% mapped versus 59.11% native on 1,037 development questions, close to a 50.05% majority baseline. Contract timing median paired cold speedup was 0.953×. This is a negative result. Strict generation also failed, but native label descriptions and mapped repetition are distinct failure categories. See the final results and trace-based audit. |
| Why not just report the faster number? | Only 12 unique questions were measured, despite 120 repeated/variant attempts. Quality failed, and median timing differences do not demonstrate a production win. Cold transfer includes source prefill and mapping; a warm-source handoff answers a different question. |
| Why constrain choices if the task asks for generation? | The prospective decision metric isolates option preference; unconstrained generation is separately evaluated. Conditional A/B/C probabilities can look confident even when total option probability is tiny, so the harness also records full-vocabulary option mass. Neither metric is silently substituted for the old pilot score. |
| Why did training fail? | The first larger run produced nonfinite FP16 gradients before update 85. The guard stopped it and preserved checkpoints/logs. The corrective code retries a window at lower loss scales, counts actual optimizer updates separately, and still stops on persistent overflow. Saved tests are evidence for the guard, not proof that the next training run converges. |
| What would make this production-worthy? | Quality retention on a locked held-out workload, robust formatting/fallback contracts, profiling and kernel/runtime work, concurrency/throughput tests, cache compatibility/versioning, and costs measured against a native-only serving baseline. These remain work to do. |
| Did you invent cache transfer? | No. The work is an independent, scoped implementation and evaluation inspired by existing work. Explain precisely which mapper/runtime choices were implemented and which features—hybrid recurrent state and production serving integration, for example—are absent. |

## Claim ledger

| Safe claim at this snapshot | Evidence | Claim that still needs evidence |
|---|---|---|
| Qwen3-0.6B → Qwen3-1.7B cache translation/import runs on a GPU | [Pilot traces](../reports/sova-kv-pilot-20261118-123416-ea50/collected/evaluation/traces.jsonl) and [controls](../reports/sova-kv-pilot-20261118-123416-ea50/collected/controls.json) | Works on arbitrary pairs, 100B models, or hybrid recurrent-state models |
| 91 software/tiny-model tests passed | [Test report](../reports/smoke-full-v1/test-results.json) | The pretrained mapper preserves task quality or is production-ready |
| Five inference paths and 120 pilot attempts were measured on 12 questions | [Pilot summary](../reports/sova-kv-pilot-20261118-123416-ea50/collected/evaluation/summary.json) | A statistically established speed improvement or thousands of independent evaluated questions |
| 64 train + 16 validation documents are prepared; all 1,037 decisions on 61 development documents completed | [Frozen data manifest](../datasets/prepared-v1/manifest.json), [final results](../reports/sova-kv-full-20261118-132341-7daf/RESULTS.md) | A passing quality/speed result or an untouched final-test claim |
| An 11,125,856-parameter mapper completed 400 actual updates; validation selected checkpoint 200 | [Saved corrective training summary](../reports/sova-kv-full-20261118-132341-7daf/collected/training/training-summary.json); earlier failure logs are retained | Converged training, improved downstream accuracy, or quality-preserving speedup |
| GPU runs have bounded timeouts, S3 artifacts and managed checkpoints | [Full run plan](../FULL_RUN.md), submission receipts and collector | A deployed public API, customer traffic, throughput scale or lower production cost |

## What current employers ask for, and how this project maps

The following are **requirements examples, not an eligibility assessment or a promise of interviews**. Official employer/ATS pages were fetched live on 18 November 2026 and exposed application forms or an Apply link. Recheck before applying; location, experience and authorization constraints still apply.

| Official posting | Requirements visible in the posting | Project signal and remaining gap |
|---|---|---|
| [Together AI — Machine Learning Engineer, Inference](https://job-boards.greenhouse.io/togetherai/jobs/4385540007). San Francisco; application asks about four office days per week. | Python/PyTorch, tested performance tooling and systems fundamentals; 3+ years of production-code experience. Existing serving engines, speculative decoding and CUDA/Triton are preferred. | Cache instrumentation, reproducible tests and cloud artifacts are relevant examples. This single project does not establish years of production service operation or custom-kernel experience. |
| [Anthropic — Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008). San Francisco / New York / Seattle; hybrid. | Performance investigations and profiling, Python, quantitative analysis, written communication and correctness/numerical regression detection. Accelerator and serving-fleet experience are preferred. | The negative-result write-up, cold/warm accounting and numerical guards map well. A fleet-scale latency, throughput or roofline investigation is not yet demonstrated. |
| [Fireworks AI — Member of Technical Staff, Performance Optimization](https://jobs.ashbyhq.com/fireworks/7ad6d17e-950e-4413-be33-9c1489f069d8). San Mateo; hybrid. | GPU profiling, CUDA/ROCm, PyTorch execution, multi-GPU debugging and 5+ years of performance/HPC experience. The role also covers mixed precision and benchmarking. | FP16 failure analysis, model/runtime co-design and explicit benchmark contracts are useful discussion material. Do not claim CUDA-kernel, multi-node or seniority requirements from this project alone. |

My interpretation of these postings: **correctness-aware measurement and the ability to explain a failure are hiring signals in their own right**. This does not establish that the prototype meets every role's experience bar. Lead with concrete engineering artifacts and the judgment they demonstrate; retain the measured limitations next to the measured numbers.

No application, public post, recruiter message or employer contact was sent while preparing this draft.
