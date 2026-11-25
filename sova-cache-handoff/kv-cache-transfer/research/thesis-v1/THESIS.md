# SOVA thesis v1: preserve behavior before claiming cheaper context handoff

19 November 2026 · Research and experimental plan · No new training launched

**20 November update.** The Qwen3 1.7B → 8B run was stopped after 82 updates. At the last scored checkpoint, answer-distribution KL had fallen and gold accuracy was still 14/30. The [loss analysis](answer-loss-analysis.md) explains the objective, checks individual answer changes, and defines the diagnostic that would distinguish an objective problem from a cache representation problem.

The [research council](council-report-20261120-0328.html) recommends keeping training paused. First compare the best achievable accuracy and latency with and without translated cache on the same fresh requests. Test a fixed router only if translation adds value, and test continual updates only once the fixed router works.

The [fresh paired replay](../../reports/paired-replay-longbench-v1/RESULTS.md) found zero incremental speedup at matched accuracy for the archived mapper. In the [six-case component probe](../../reports/component-swap-longbench-v1-flex/RESULTS.md), native keys recovered one answer, native values recovered zero, and restoring both recovered all six. The next experiment can localize the joint attention readout error by layer and question position before changing the loss.

The [native tail repair](../../reports/tail-repair-longbench-v1/RESULTS.md) moved the development frontier. Recomputing 8B's first token and final 31 document tokens recovered one full-prefill case and reduced mean TTFT by 11.4% in a gold-informed replay that preserved every native-correct answer. On the [longer, disjoint panel](../../reports/tail-confirm-longbench-v1/RESULTS.md), the frozen variant recovered none of the three full-prefill cases and saved 0 ms. Focus the next mapper probe on joint K/V readout distortion, using causal layer swaps before selecting a fitting loss.

Follow-up evidence: the [first diagnostic completed24/24 validation articles](../../reports/state-diagnostic-v2/RESULTS.md). RMS-off improves the untrained initializer's first-answer distribution but worsens the existing trained mapper on23/24documents by KL. Both K/V readout errors remain; a posthoc normalization toggle is not the repair. This is diagnostic validation evidence, not a new final-test or serving result.

**Working thesis:** when one model has already processed a document and another model needs to take over, a pair-specific cache translator may preserve enough of the receiver's useful behavior to avoid reading the document again—at lower total cost and latency than the best applicable baseline.

The useful condition is **source cache available, target cache absent**. A repeated document by itself is insufficient: if the target already has its own valid cache, ordinary prefix reuse may be better. Fresh-prompt acceleration is a separate, harder claim because it must pay for the source's initial read.

This is a hypothesis to test, not a result we currently have. The existing candidate fails the quality requirement. The next step is a small set of discriminating experiments, not another blind training extension.

## 1. What our evidence actually establishes

| Measurement | Verified result | Implication |
|---|---|---|
| Native target cache import controls | 16/16 FP32 controls passed | The tested numerical import path can preserve native behavior. This does not validate learned translation. |
| Mapper optimization | 48 finite updates; both base models frozen | Training now runs on the tested card. It did not reach the requested 100 updates or demonstrate convergence. |
| QuALITY, 100 questions | Source 47%, native target 59%, translated 26% | Translation is substantially worse than either model alone. |
| HellaSwag, 1,000 examples, `acc_norm` | Source 47.3%, native target 61.2%, translated 28.5% | Failure extends beyond the training task. |
| Mapper-validation CE+KL | 20.886 → 3.995 | The optimized loss improved; useful downstream behavior did not follow. |
| Final execution coverage | All planned rows completed; zero errors/skips | These poor scores are not explained by unfinished evaluation. |

Mapped-minus-native document/group bootstrap 95% intervals were −45 to −19 percentage points on QuALITY and −36.82 to −28.99 points on HellaSwag. These results reject the current candidate as a quality-preserving replacement. They do not identify the cause or disprove all cache translation.

Authoritative [results and provenance](../../reports/sova-kv-eval-resume-20261118-183814-e659/RESULTS.md), [independent results audit](../final-results-audit.md), and [current implementation audit](implementation-audit.md). Historical prompt-boundary and question-pool defects were repaired before these final results; they must not be reused as current diagnoses.

## 2. The mechanism we are testing

Let `D` be the reusable document prefix and `q` the new question/instructions. For frozen source `S` and target `T`:

```text
Native reference: D ──target prefill──► C_T(D) ──target reads q──► answer

Transfer route:   D ──source prefill──► C_S(D) ──g_phi──► approximate C_T(D)
                                                            │
                                              target reads q ──► answer
```

If `C_S(D)` already exists, its prefill is a sunk cost for incremental handoff timing. It still belongs in whole-workflow resource accounting. The target processes the fresh suffix and generates normally; attention and autoregressive decoding remain present.

The behavioral goal is:

`P_T(answer | C_T(D), q) ≈ P_T(answer | g_phi(C_S(D)), q)`

Cache vectors need not match element by element if they induce the correct receiver behavior. Conversely, low cache MSE does not guarantee correct attention or answers. A deterministic translator cannot recover distinctions absent from its input state. However, a weak source answer alone does not prove its cache lacks the information: the target may decode that representation more effectively.

The first candidate should keep `g_phi` independent of `q`, so one translated prefix can serve several questions. Question-conditioned translation is a separate design with additional computation and less reuse; its cost cannot be hidden.

## 3. What we take from existing work

The following are research precedents, not independently reproduced SOVA results. Paper bodies were inspected; arXiv availability is not evidence of peer review.

| Reference | Lesson for our design |
|---|---|
| [Cross-Model KV Cache Transfer](https://arxiv.org/html/2608.03893v1) | Establish a serious linear reference: the production fit uses all source heads from predictively selected layers. Compare reconstruction with receiver behavior. |
| [CacheLink](https://arxiv.org/abs/2610.15509) | Optimize what the receiver does with the cache; cache reconstruction alone can miss important functional errors. |
| [CacheBridge](https://arxiv.org/abs/2609.00891) | Attention-readout calibration and mapper serving overhead deserve explicit measurement. A wider map is not automatically the better system. |
| [KV-Lingo](https://arxiv.org/html/2609.32610v1) | Functional distillation can improve simple maps. Crucially, its released shape-matched Qwen pairs show competitive head-wise mapping, and mixed CE+KL is not universally worse than KL. Neither wider heads nor changing the loss is a proven fix for us. Its switching latency assumes the source cache already exists. |
| [Hybrid-state handoff](https://arxiv.org/abs/2611.08133) | Hybrid models require more than ordinary K/V transport. Recurrent and convolutional state introduce a separate problem; defer it until the dense reference works. |

[OneTriangle's task-specific service](https://onetriangle.ai/mapper-api) reports favorable held-out results for a different, larger pair and workload. That is vendor evidence of a possible product direction, not a controlled comparison to our implementation. Its [homepage](https://onetriangle.ai/) distinguishes measured campaigns from projections; a projected million-token speedup must not become a measured SOVA claim.

Our potential contribution is an independently auditable quality/latency tradeoff, a clear diagnosis of when handoff fails, and an eventual useful integration. Linear cache translation, distillation, and cache connectors already have prior art. See the [literature review](literature-review.md).

## 4. Separate the three questions

1. **Representation:** does the selected source state contain a recoverable approximation useful to this target?
2. **Behavior:** does the receiver retain task accuracy, evidence use and generation behavior on unseen documents?
3. **Economics:** after mapping, movement, suffix processing, memory and fallback, is this better than re-prefill or available native cache reuse?

A pass on one does not imply a pass on the others. The memory repair established an execution path, not these three claims.

## 5. Why our current pair is a diagnostic, not a speedup promise

Pinned official configurations: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json).

| Geometry | 0.6B source | 1.7B target |
|---|---:|---:|
| Layers | 28 | 28 |
| Query heads | 16 | 16 |
| KV heads | 8 | 8 |
| Head dimension | 128 | 128 |
| Hidden width | 1,024 | 2,048 |
| Intermediate width | 3,072 | 6,144 |

For a dense attention calculation, the quadratic QK/AV work scales as `L × H_query × d_head × N²`. Its coefficient is the same here. Projection and MLP work differ, so this is not a claim of equal runtime. It means parameter ratio alone cannot predict prefill savings, especially as context grows.

Cache payload is `2 × L × H_KV × d_head × N × bytes_per_element`. Both models use **224 KiB per prefix token in FP32**, or **112 KiB in FP16/BF16**. At 8,192 tokens one cache is 1.75 GiB in FP32, excluding allocator overhead, extra buffers and model weights. Translation does not inherently compress that payload; temporarily retaining both caches adds memory.

Before choosing a larger production pair, measure its native source/target prefill curves and realistic cache-transfer cost. Keep this cheap pair for isolating behavior. Do not infer a commercial speed ceiling from model names or parameter counts.

## 6. Competing explanations and experiments that distinguish them

| Hypothesis, currently unproven | Controlled intervention | Result that weakens it |
|---|---|---|
| Forced constant per-head key RMS harms target attention | Same fitted map, same inputs, RMS projection on/off; inspect key norms, attention logits and readouts | Removing it does not improve held-out receiver behavior |
| Source feature support is too restricted | Same-head k1 vs all-head k1; then selected layers at controlled width/data budget | Wider/selected features improve fit but not attention or answers, or head-wise stays competitive |
| Objective misses the decision we score | Identical initializer/data budget: KL-only vs present CE+KL; separately test an explicitly decision-aware auxiliary | Objective-aligned validation and downstream decisions remain unchanged |
| Short, narrow functional training is insufficient | Learning curves in actual updates, unique documents/questions and supervised tokens, with multiple seeds | More exposure improves training loss but held-out behavior plateaus below acceptance |
| Source/target pair loses task-relevant information or is hard to align | Stronger maps, native-query layer probes, length/task strata; later a separately selected pair | Simple controlled changes recover native behavior on this same pair |
| Transfer overhead consumes savings | Warm and cold timing decomposition, plus target-cache-hit baseline | End-to-end speed and cost pass the declared gates under realistic load |

Same-head mapping is a deliberately restricted baseline, but published counterevidence makes it an experiment rather than an assumed root cause. Our constant scalar RMS projection is also not equivalent to applying the target's learned per-channel key normalization. A pre-key-normalization capture variant would be a distinct representation experiment, with the target's actual normalization and RoPE restored in the proper order.

For a fixed teacher distribution `q`, greedy token `y*`, and equally weighted `CE(y*,p) + KL(q||p)`, the unconstrained optimum is `p = (q + one_hot(y*))/2`. The current loss therefore sharpens the teacher instead of exactly preserving it. That algebra does not establish the cause of the observed accuracy loss. Conditional A–D scoring can also be underweighted by full-vocabulary KL when total option mass is low; we should measure both rather than silently change the metric.

## 7. The experimental ladder

| Stage | What we do next, in order | Required evidence before advancing |
|---|---|---|
| A. Protocol | Freeze this failed checkpoint/results; specify workload, exact cache boundary, data partitions, metrics and compute ceiling | Reproduction manifest and untouched final-test IDs reserved without inspecting outcomes |
| B. Feasibility | Compare source/target prefill and mapping/transport cost for candidate deployment conditions | A credible break-even region; otherwise narrow to warm handoff or reject the pair for acceleration |
| C. Diagnose | Native-cache control; swap mapped K/native V and native K/mapped V; native-query readout probes; RMS-only ablation | A localized functional error or a reproducible improvement, not just lower MSE |
| D. Linear reference | Same-head and all-head k1, predictive layer selection only if justified; explicit ridge convention | Select by document-disjoint validation and behavior, with mapper cost recorded |
| E. Functional training | Start from the best controlled reference; compare loss choices at matched exposure; add capacity only for demonstrated underfit | Validation learning curves and downstream retention; more than one seed for a positive candidate |
| F. Sealed quality test | One selected configuration, new unseen documents/groups, all benchmark arms | Prespecified quality bounds with uncertainty; test outcomes do not feed further tuning |
| G. Serving study | Optimized precision/backend checked against the numerical reference; load and fallback tests | Quality still passes; measured end-to-end p50/p95, cost, memory and throughput pass |

Native-query readout probes use target-native queries as a diagnostic oracle. They do not constitute an inference shortcut available without native computation. K-only/V-only substitutions likewise localize errors; they are not deployable quality results.

Do not run a full Cartesian sweep. Start with an unchanged checkpoint control and one-factor normalization test, then choose the next branch from its evidence. Record the hypotheses and selection rule before each bounded run. A better linear reference need not mean materializing all-layer covariance matrices at once; fit memory and accumulation conventions must be planned first.

The 100 QuALITY and 1,000 HellaSwag examples have now been inspected and discussed. Keep them as development/regression evidence; they cannot become a pristine future final test. A file hash prevents mutation, not selection leakage. Split by original document/source group, preserve official evaluation roles, and ensure calibration, teacher traces and layer selection cannot reach the final test. See the [detailed experimental design](experimental-design.md).

## 8. What a win must mean

Before collecting new results, choose a task tolerance. **Proposed engineering gate**, to be finalized in the protocol: the one-sided 95% lower confidence bound on mapped-minus-native accuracy exceeds −2 percentage points for the primary in-domain task, and −5 points on a separately reported OOD stress task. These are SOVA design targets, not established industry standards. If the sample cannot resolve that margin, the outcome is inconclusive, not a pass. Account for document clustering and paired discordance when planning sample size.

Also compare with the source alone and simple task baselines. When the target offers no useful advantage, using the source may dominate translation. Report disagreement with the target separately from accuracy against gold: preserving a wrong target answer is fidelity, not improved correctness.

Use a separate timing gate, for example a prespecified ≥1.3× end-to-end p95 TTFT improvement on the actual intended workload with no cost/request regression. This is a proposed minimum practical benefit, not a forecast. Report errors, fallback rate and valid-request coverage so fast failures cannot pass.

Failure branches are explicit: quality failure blocks serving claims; economic failure blocks acceleration claims even if quality is excellent. If controlled maps plateau, change the pair, workload or approach instead of continuing indefinitely. Distillation, routing, retrieval and quantization remain separate SOVA options.

## 9. The cost equations we must measure

Define `P_S(N)` and `P_T(N)` as native prefix prefill time; `M(N)` as mapping; `X(N)` as transport/layout/import; `U(q,N)` as target suffix plus first-token work. Include synchronization and any model-loading/queueing cost applicable to the deployment.

```text
Native cold:       P_T(N) + U_native(q,N)
Mapped cold:       P_S(N) + M(N) + X(N) + U_mapped(q,N)
Warm handoff:      M(N) + X(N) + U_mapped(q,N)
Native cache hit:  target-cache retrieval/import + U_native(q,N)
```

Cold transfer only wins if the entire second line is cheaper than the first. Warm handoff can win with a pair that does not win cold. If the source cache must be fetched from slow storage, include that fetch. If either cache is kept resident, charge its capacity and eviction opportunity cost. Do not compare a GPU-resident translated cache with a deliberately uncached target and call that a general serving win.

For a routing workflow with escalation probability `r` and no existing target cache, baseline expected incremental work is approximately `P_S + r(P_T + U_native)`, versus `P_S + r(M + X + U_mapped)` for translation. This only models the handoff component; source answer costs, quality-based routing mistakes, fallback, batching, concurrency and cache persistence still need explicit treatment. End-user savings shrink when escalation is rare. Repeated target queries may subsequently use native or translated target caches in both arms; do not repeatedly charge the baseline for a prefill it can reuse.

## 10. Study order and concrete next deliverable

1. **Understand the failed run:** read the results and implementation audit; explain what passed and what did not without mixing historical defects into the current diagnosis.
2. **Read the attention mechanism as a function:** `softmax(QKᵀ / sqrt(d))V`; explain why key scale, value alignment and small repeated layer errors can change answers despite modest MSE.
3. **Compare the methods:** linear fitting, receiver-functional distillation and attention-readout calibration; identify which inputs, losses and timing conditions differ.
4. **Work through the economics:** warm handoff vs cold prompt vs native cache hit; calculate payload and explain why the present pair's parameter ratio is misleading.
5. **Review the [first experiment card](first-experiment.md):** unchanged control, RMS toggle, native K/V substitutions, deterministic diagnostic selection and measurements. Fill its exact hashes, resource estimate and cost/time ceiling before implementation or submission.

**The next scientific step is the bounded diagnostic in that card.** This thesis defines the decision process; no new model training, evaluation, GPU worker, dependency installation or serving deployment was started for it.

Supporting documents: [implementation audit](implementation-audit.md) · [literature review](literature-review.md) · [evaluation protocol](experimental-design.md).
