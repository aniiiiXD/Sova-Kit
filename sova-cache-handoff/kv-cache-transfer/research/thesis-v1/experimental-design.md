# Cross-model cache translation: prospective experimental design

19 November 2026. **Study plan, not a new result or execution authorization.** This document proposes falsifiable claims, measurements and stopping rules after the completed negative experiment. No model processing, tests, training or cloud jobs were run for this document. Thresholds requiring a product decision are deliberately marked as decisions to freeze, rather than presented as already agreed requirements.

## 1. State which claim is being tested

For document/context `c`, fresh query `q`, source cache `C_S(c)`, receiver `T` and translator `F`, compare:

```text
Native:       target(c + q) → answer
Translation:  source(c) → F(source cache) → target reads q → answer
```

The receiver still performs attention and computation on the native query suffix and generated tokens. This is state translation between frozen models, not removal of attention or a new foundation model.

| Claim | Falsifiable question | Evidence needed |
|---|---|---|
| Mechanism | Does translated state preserve useful receiver behavior on independent contexts/questions? | Correct native-cache controls, strong mapping references, fresh held-out utility and distributional diagnostics |
| Product | In a specified cache-miss workload, is the complete translated request sufficiently accurate, faster and cheaper than available alternatives? | Mechanism result plus optimized end-to-end baselines, memory accounting, concurrency and an explicit service/error budget |
| Faithful public-method reproduction | Does the pinned published method work under its stated model/data/scoring protocol? | Method-matched source-layer/head support, calibration, objective, tokenization and scorer; separately record unavoidable differences |
| Vendor reproduction | Can the same vendor campaign/model pair/item set/service boundary be reproduced? | Exact campaign, model revisions, sample IDs, prompts, mapper/state types and server timing visibility; private unavailable ingredients remain unresolved |

The current small dense pair cannot establish a hybrid 4B→27B vendor claim. Our former same-head ridge and nearest-layer residual are not a faithful full-cross-head, selected-layer paper reproduction. Public-method results and our product candidate must have separate rows even if they share infrastructure. A constrained decision API is one possible product; general reasoning or unrestricted generation is a broader claim and needs its own endpoint.

### Specify a workload where translation could be valuable

The central use case is **a valid source cache already exists, while the desired target cache is absent or evicted**: for example model escalation after a source-model interaction, or a model/version boundary within an explicitly permitted tenant. Record how often this happens in the intended workload. Cache identity must include document tokens, tokenizer, model revision, positions and tenant/access scope. Tenant boundaries are not permission to reuse another tenant's state.

Shared-document repeats alone do not justify translation: a native target prefix-cache hit may be the best solution. Compare native target caching with an equal cache-memory/residency budget, and report target-cache hit rate, eviction policy, translation frequency and residency cost. Different model versions can invalidate either cache; do not assume a free reusable source state. Charge source prefill whenever it has not already been performed for the actual application.

## 2. What the completed experiment already tells us

The [independent final audit](../final-results-audit.md) recomputed the following from frozen labels and stored raw scores. All 3,300 final rows completed successfully; memory recovery and measurement correctness do not imply successful translation.

| Frozen development measurement | Source 0.6B | Native target 1.7B | Mapped target | Mapped − native, document/source-cluster 95% interval |
|---|---:|---:|---:|---|
| QuALITY option-logit decision, 100 article questions | 47% | 59% | 26% | −33 pp; **[−45, −19] pp** |
| HellaSwag character-normalized ending likelihood, 1,000 records | 47.3% | 61.2% | 28.5% | −32.7 pp; **[−36.8211, −28.9866] pp** |

QuALITY strict free-generation correctness was 1/100, 5/100 and 4/100 respectively; exact-letter format validity was 1/100, 9/100 and 18/100. The decision and generation endpoints are therefore materially different. HellaSwag raw unnormalized accuracy was 37.3%, 46.9% and 26.4%.

The selected mapper completed **48 of 100 requested updates**, stopping on a time budget. Its fixed 30-question training-domain validation objective decreased from **20.886248811 to 3.995068248**, with checkpoint 48 selected. That is an objective improvement, not task accuracy or proof of convergence. The experiment strongly rejects useful quality retention for **this candidate, training budget and evaluated workloads**; it does not establish that every translator, the public method or the vendor implementation fails.

The final quality result combines a status-only continuation at the same selected mapper identity. It explicitly does **not** support a merged cross-job latency headline. Keep historical scores and parsers frozen.

**All evaluated QuALITY, HellaSwag and ContractNLI slices are now development data.** Their scores, failures and examples have informed design choices. Other questions from an already inspected article are not a fresh document-level test. Audit every earlier run and artifact before calling any remaining split untouched.

## 3. Data separation and leakage controls

Create a versioned partition registry before fitting the next candidate:

| Pool | Allowed use | Prohibited use |
|---|---|---|
| CAL: generic calibration | Fit source-layer selection and linear/state reconstruction; a method replication follows its published calibration prescription | Final-quality selection or test-document inclusion |
| OPT: functional training | Fit translator using frozen target distributions/continuations; keep the full declared question pool | Test answers or evaluation-generated examples feeding training |
| SELECT: development | Choose architecture, loss weights, checkpoint, length policy and deployment configuration | Calling its eventual result an untouched-test result |
| LOCKED: final evaluation | One declared candidate/configuration and prespecified primary/safety endpoints | Layer selection, prompt adjustment, parser changes, checkpoint choice or repeated uncorrected candidate shopping |

CAL and OPT may share train-only documents if the scientific question requires it, but record that choice. SELECT and LOCKED must be group-disjoint from both and from previously seen sources. Split by the underlying work/source family where possible: article/book, contract/template lineage, company report family or source video. QuALITY has multiple records per article; HellaSwag rows can share a source group. A record-level random split is insufficient.

Pin upstream revisions, licenses/provenance, source/group IDs, exact and normalized document hashes, processed hashes, selection seed and all exclusions. Audit normalized/near duplicates across calibration, optimization, development, historical artifacts and locked data; log the matching rule and disputed cases. This reduces our own leakage, not unknown foundation-model pretraining exposure. Commercial dataset permission remains a separate provenance question.

For future quality claims, prefer independent newly sourced documents with verified answers. Remaining public benchmark records can be a **prospectively locked public holdout** only after prior-access and source-overlap audits; do not automatically call them pristine. Official QuALITY test labels are withheld, so a fully public gold test cannot be promised from that split. Hold locked labels separately where practical and freeze a candidate artifact before access.

Select by source groups before token-length screening. Publish the target length population and context cap in advance. If the product supports only a declared cap, report coverage for that entire population and quality within the eligible population separately. Within the selected supported set, errors, over-limit inputs and budget-not-run items remain visible in the denominator; never silently drop them or truncate evidence away. An unfinished test is incomplete, not a favorable partial result.

## 4. Fix the quality endpoints before new outputs

### Decision product

Primary accuracy is the predeclared option-logit argmax against verified gold, using a frozen prompt, reasoning mode, tokenizer and option representation. Report native agreement, class/macro accuracy, confusion matrices, error/abstention coverage and prespecified length/source strata. Store full-vocabulary option mass as well as option-conditioned probabilities: an apparently decisive A/B/C/D ranking can coexist with little probability on valid answers.

A wrapper can deterministically return a ranked option. That is a legitimate constrained API contract, not evidence that unrestricted text generation is preserved. Do not require beating the source and a tiny target gap as a universal double condition: some strata have no native-target advantage. Report the native-source gap and source alternative per stratum; select the product operating point on quality, cost and latency together. If native itself misses the application's absolute quality floor, native noninferiority alone is not sufficient.

### Generation and standard benchmark protocols

Free generation has its own frozen parser, semantic scoring rule, output budget and stop conditions. Report exact format, semantic correctness, truncation, repetition and invalid output separately. Teacher-forced scores do not replace own-rollout checks. A changed parser is a new prospective protocol; it cannot silently repair old scores.

For HellaSwag use complete conditional ending likelihood, pinned preprocessing/token-boundary handling, raw `acc` and processed **character-length** `acc_norm`. Do not substitute letter generation. For a paper tail-perplexity reproduction, match the declared first-1,024-token cache / next-1,024-token scoring construction rather than calling arbitrary rolling perplexity equivalent. Our custom QuALITY option/generation protocol remains explicitly custom unless exact external prompts and scoring are available.

### Receiver-distribution diagnostics

On SELECT and then the fixed locked diagnostic sample, measure native-to-mapped full-vocabulary KL/TV and teacher-forced continuation cross-entropy on the **same native target trajectory**. Sample actual query-dependent answer positions, plus early/late continuation positions; fixed assistant-template positions must not dominate the objective or report. Separately evaluate free rollout and native/mapped divergence over time. Report both easy and context-dependent examples. Low cache MSE or training loss cannot be a standalone utility acceptance gate.

## 5. Establish plumbing correctness before evaluating learning

Freeze and rerun the same controls for the actual candidate precision, attention backend and context/position range:

1. Native full prompt versus native prefix-cache plus the exact same native suffix. Verify token IDs, split boundary, causal positions, masks, GQA layout and cache lengths; no hidden full-prefill fallback. Repeat fixed inputs across fresh cache allocations and process restarts to identify software nondeterminism.
2. Native identity cache with inverse/restore RoPE, boundary-straddling tokens and the intended query chunking. Include shortest/longest supported lengths and actual problem cases, not only short synthetic fixtures.
3. Zero, shuffled and wrong-document caches on context-dependent/counterfactual fresh-fact examples. Native performance must demonstrably depend on the document; otherwise a preserved answer does not demonstrate preserved context state.

The existing `oracle-probability-v2` software guard uses full-vocabulary TV ≤1e−3, forward and reverse KL ≤1e−4 nats, matching first-token argmax, and only a 1e−12 negative-KL roundoff allowance. These are **numerical native-cache equivalence guards**, not acceptable learned-mapper distortion budgets. If optimized native-native execution cannot meet a chosen guard, characterize its A/A numerical floor on development controls and preregister the appropriate revised guard before candidate testing. Never relax a failed guard after looking at that candidate's quality.

## 6. Diagnose what state is being lost

Use a fixed, small SELECT diagnostic panel, stratified by source, length and evidence position. Capture layer/head measurements with identical token alignment and masks:

| Receiver computation | Question it isolates |
|---|---|
| Native Q, native K, native V | Readout reference |
| Native Q, mapped K, native V | Does translation corrupt addressing? |
| Native Q, native K, mapped V | Does translation corrupt retrieved content? |
| Native Q, mapped K, mapped V | Combined state error without query drift |
| Transferred-run Q with mapped K/V | Additional hidden-state/query drift accumulating through the receiver |

Compare pre-RoPE and correctly positioned post-RoPE K; value/key scale and covariance; relative K/V MSE; attention entropy/distribution divergence; attention-readout cosine and relative error; query drift; and next-token full-vocabulary divergence. Show distributions per target layer/head and prefix-position band, not only a global average. Account explicitly for GQA head sharing. These interventions localize failure; they do not imply that, for example, cosine 0.95 guarantees decision quality.

Use distinct named reference arms: source only; native target; native target prefix-cache hit; former weak ridge; method-matched cross-head/selected-layer linear translator; and a functional translator initialized from that reference. Change one major choice at a time on SELECT: head/layer support, RoPE/normalization, objective, native query boundary, or joint K/V representation. Selective native recomputation is a separate operating point with its recompute fraction and cost reported. It cannot be called pure translation.

## 7. Sample size and uncertainty

An approximately 32-context boundary/length/source panel can be an engineering catastrophe screen; it is chosen for coverage, **not** for a narrow quality claim. Our completed n=100 QuALITY screen was enough to detect a large failure. It is not generally enough to establish a two- or three-point noninferiority margin.

Define paired correctness `D_i = correct_mapped_i − correct_native_i`, difference `Δ = E[D]`, and discordance `q = P(D ≠ 0)`. Then `Var(D) = q − Δ²`. Under a normal planning approximation, a one-sided α=0.025 noninferiority test with 80% power near true Δ=0 needs:

```text
n_effective ≈ (1.96 + 0.84)² × q / δ² = 7.84 × q / δ²
```

For a **3 percentage-point** tolerated loss, hypothetical q=0.10/0.20/0.30 implies approximately **872 / 1,743 / 2,614 effective independent units**, before clustering. This is conditional design arithmetic, not a guarantee of an adequate sample. Estimate discordance and within-source correlation using SELECT; use group-level simulation/bootstrap for the actual design, then freeze the final size before locked outcomes. Multiple questions on the same document do not create independent documents. If enough independent units cannot be obtained, report an exploratory interval rather than a tight preservation claim.

Choose δ from the application: three points means permitting up to 30 additional wrong decisions per 1,000 requests relative to native. That might be tolerable for a noncritical demo and unacceptable for a financial/legal decision. Freeze the absolute native quality floor, allowed additional errors, high-risk-class limits and abstention policy with the intended product owner. There is no universal two-point or three-point margin.

Report paired source/document-cluster uncertainty, including absolute arm scores and mapped-native differences. Resample source groups together across arms; prespecify weighting if group sizes differ. Use enough bootstrap replicates for stable tails and publish seed/method. Timing repeats do not enlarge quality sample size. One frozen candidate and one primary workload avoid uncorrected candidate shopping; multiple locked-test looks/candidates require a prespecified multiplicity/sequential rule or a new holdout. Report all prespecified safety/OOD endpoints rather than choosing whichever passes.

## 8. Measure a real break-even, not mapper-only speed

Separate model/container startup from **prompt-cache** coldness. A loaded engine with no relevant prefix cache is a valid cold-cache measurement; it is not a cold machine. Publish at least these workload states:

| State | Required comparison |
|---|---|
| Neither cache exists | Target cold request versus source prefill + translation + target query/decode |
| Source cache exists, target misses | Handoff plus target query/decode versus target cold request; disclose source sunk work and residency |
| Target cache exists | Native target prefix-cache hit; an essential competitor for repeated documents |
| Both or finite cache capacity | Trace-driven hit/eviction behavior and total cache memory, rather than unlimited free residency |

At a fixed context length `N`, query and output contract:

```text
T_transfer_cold = T_source_prefix + T_map + T_move + T_import
                  + T_target_native_query + T_first_output
T_native_cold   = T_target_full_prompt + T_first_output
```

Warm-source handoff excludes only source work genuinely completed for the application's preceding request. Native prefix hits likewise include real lookup/import/suffix costs. Profile synchronization, bytes moved, memory allocation and receiver suffix work; overlapping stages must be timed end-to-end rather than summed twice. Fit a measured break-even curve over prespecified length buckets on SELECT. Do not turn an extrapolated 1M-token curve into a measured result.

Compare against the **same quality-qualified optimized target baseline**, with equal hardware/resource allowances. An FP32 math-attention correctness oracle is a research control, not a fair production opponent if the native target can use a faster validated backend. Report the oracle track and optimized deployment track separately. If translation requires a slower backend or prevents a useful cache optimization, charge that disadvantage. Standalone native memory must be measured without an unnecessary resident source model.

Use warmups, blocked randomized arm order, synchronized device timing, request timestamps, fixed batch/concurrency points, and enough independent sessions to characterize p50/p95 and run-to-run variation. Request TTFT includes tokenization, transport, scheduling and first-output delivery for an API claim; separately show device-only breakdowns. For a decision endpoint define time-to-decision rather than calling an un-emitted logit a delivered first token. Record generated length so different completion lengths cannot manufacture speedups.

A speed acceptance threshold follows the actual SLO and overhead: for example, reducing a measured 600 ms baseline below a 400 ms SLO would require 1.5× end-to-end improvement. **Those numbers are illustrative, not SOVA measurements.** Freeze the real threshold before final benchmarking; require uncertainty excluding zero benefit and enough benefit to meet the chosen SLO, with declared p95 constraints. Include source/target GPU-seconds, transfer cost, resident-cache opportunity cost, peak memory and capacity at concurrency. Training-job bills are not serving cost per request. A quality-failing mapper does not qualify as a useful cheaper alternative.

## 9. Staged preregistered go/no-go protocol

| Gate | Pass condition, fixed before that stage | If it fails |
|---|---|---|
| G0 — thesis and design | Named claim/workload, reference method differences, source-disjoint pools, hashes, endpoints, error/SLO budgets, candidate-selection rule and bounded compute plan recorded | Resolve the missing design input; no paid run implied by this document |
| G1 — exact native controls | Declared numerical oracle passes actual supported boundaries/lengths/backends; context-negative controls establish task dependence | Repair token/cache/numerical plumbing before interpreting mapper loss |
| G2 — strong state reference | Faithful linear reference evaluated on unseen calibration groups; layer/readout/logit diagnostics locate distortions; candidate improves declared SELECT functional endpoint over its initialization/reference | Stop that candidate at its fixed budget; investigate state/objective assumptions rather than asserting that more parameters must help |
| G3 — bounded functional training | Frozen receiver, intended deployment boundary, actual question diversity and answer-position supervision verified; SELECT curve/checkpoint and resources fully recorded | Describe a time-limited or nonconverged candidate honestly; revise only with a new development protocol |
| G4 — locked utility | One frozen artifact; completed denominator; lower confidence bound of Δ exceeds −δ, absolute task floor and prespecified risk/generation gates met | No quality-preserved product claim; test becomes seen, retain failures, obtain a fresh holdout for redesign |
| G5 — optimized system | Same qualified artifact meets complete-request latency/cost/SLO and cache-capacity gates against cold and warm native competitors | Report quality preservation without a useful systems win; optimize only a measured bottleneck or abandon that workload |
| G6 — bounded API pilot | Engine integration, tenant/cache identity, operational failure/abstention behavior and customer-data evaluations meet the pilot contract | Keep as research evidence; do not imply customer readiness |

G2 is a development selection gate, not a universal claim that every diagnostic must exceed an unexplained cosine/loss threshold. G4 and G5 are the actual product falsification gates, with thresholds derived from declared use. Generic mechanism reproduction and task-specific product training can run as different registered studies; do not mix their most favorable scores and timings.

For every stage, record wall-time/update/sample limits and stop reasons. Changing a length cap, output budget, precision, parser, layer selection or checkpoint rule creates a new protocol version recorded before new measurements. A continuation may complete missing status-defined examples using the exact candidate/configuration, with all arms for a partially completed example rerun atomically and provenance retained. It may not select reruns by correctness or blend incompatible jobs into a latency claim.

## 10. Minimum preregistration record before another experiment

Freeze: claim and product workload; model/tokenizer/source revisions; calibration/optimization/selection/locked group IDs and hashes; mapper and initialization identity; training/selection and stopping rules; transfer boundary/state types/recompute policy; prompt/reasoning/decision/generation contracts; supported lengths and exclusions; primary/safety endpoints and uncertainty; δ/absolute quality floor; timing/cost/cache-residency/SLO boundaries; exact resource and spend cap; and a deviation log.

The immediate scientific deliverable is this filled registration plus a faithful-method comparison and diagnosis plan. It is not another architecture implementation by default. A future positive claim requires three distinct successes: **correct state installation → retained downstream utility → useful end-to-end economics**. The completed negative result establishes why these must be tested separately.

## Local evidence and method references

- [Independent final results audit](../final-results-audit.md): current complete scores, uncertainty and provenance.
- [Original benchmark protocol audit](../original-benchmark-protocol.md): distinctions between public method, vendor campaigns, data/scoring protocols and known access limits. Its proposed-next-run sections are historical; the completed slices are now seen.
- [Methods and ablations](../methods-and-ablations.md): pinned primary-paper references and existing implementation differences. Its initial run-state descriptions precede the completed repair.
- [Architecture recovery plan](../architecture-recovery-plan.md): earlier hypotheses, not independently proven causes of the final negative result.
- [`oracle-probability-v2` implementation](../../kvverify/boundary_probe.py): numerical software guard, separate from quality acceptance.
- [Frozen reference data manifest](../../datasets/reference-benchmarks-v1/manifest.json): existing dataset provenance, hashes and selected IDs.
