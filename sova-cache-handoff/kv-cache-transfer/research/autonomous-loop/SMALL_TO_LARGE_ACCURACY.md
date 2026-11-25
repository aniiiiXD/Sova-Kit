# Small-to-large cache transfer accuracy

**Outcome.** The learned 4B-to-8B mapper rescued one otherwise-cold request on each of two panels. A chooser with the correct answers could use those routes to save latency. The tested confidence-based rules failed the accuracy and tail-latency screen. Cross-model transfer remains experimental. Native 8B cache reuse is the verified fast path when target state already exists.

The active model pair is Qwen3-1.7B to Qwen3-8B. The 1.7B model supplies the reusable document cache. The 8B model processes the question and produces the decision. The two checkpoint revisions and tokenizer fingerprints are pinned in the worker receipts.

## 20 November layer-swap diagnostic

The six questions were selected from the earlier LongBench v2 replay because full native 8B answered them correctly and every cheaper original route missed. Every partial swap kept the first 256 tokens native.

| History given to 8B | Correct | Paired mean gold-margin change vs original transfer |
| --- | ---: | ---: |
| Original affine transfer | 0/6 | 0 |
| Both K/V from native in the four layers with the largest readout error | 0/6 | −0.875 |
| Both K/V from native in four comparison layers | 0/6 | +0.667 |
| Both K/V from native in the first 12 layers | 0/6 | −1.292 |
| Both K/V from native in the middle 12 layers | 1/6 | +2.208 |
| Both K/V from native in the last 12 layers | 0/6 | +0.750 |
| Native history keys across all layers | 1/6 | |
| Native history values across all layers | 0/6 | |
| Both native across all layers | 6/6 | |

Job `sova-kv-causal-layers-20261120-030137-a4c7` completed on an L40S for $0.45 estimated SageMaker compute. The locally recomputed scores match the worker. This is an intervention on known failures, so the counts do not estimate population accuracy or serving latency. Large attention-output deviation did not identify a sufficient set of layers. The middle-third effect is a lead, with no evidence that repairing that third alone solves the overall problem.

## Repair experiment

The joint-rank-16 correction uses 64 unique complete QuALITY training articles. For every selected question position, the optimization sees the frozen 8B query and the native suffix. The objective fits both transferred keys and values against the native readout, using normalized readout MSE and a small attention-distribution KL. Both language models stay frozen. The result folds into the same affine weight shape for serving.

The development screen uses the earlier 32 LongBench questions, with a document-disjoint 20-question panel available for replication if a fixed checkpoint clears the gate. The gate requires better recovery of native-correct cases and a paired hindsight TTFT gain. Hindsight results are diagnostic upper bounds. Deployable routing and latency measurements would follow separately. A final pool of nine longer, previously unqueried documents from the same benchmark was frozen before correction results.

The training prompts span 2,508–8,051 tokens. The first evaluation panel begins above 12,000 tokens. If training improves readout fit without recovering accuracy, training-context length, native-vs-mapped query drift, and the information available in a single aligned source layer are concrete remaining explanations.

### First completed panel

| LongBench v2, 32 distinct documents | Correct |
| --- | ---: |
| Native 8B full-history prefill | 13/32 |
| Native 8B full cache hit | 13/32 |
| 1.7B cache hit | 6/32 |
| Native 8B with just the first 256 history tokens | 9/32 |
| Original full affine transfer | 3/32 |
| Original affine transfer with native first 256 | 6/32 |
| Joint readout correction with native first 256 | 10/32 |

The correction recovered six of the 13 native-correct answers versus four for the original hybrid. It solved one of the six cases where all previous cheap routes missed, a different case from the previous 31-token tail result.

On the 17,428-token rescued example, the first timed repeat took 412 ms for the corrected route and 2,305 ms for native cold prefill. The original mapper, short native prefix, and 1.7B-only routes returned different incorrect answers. Native target history already resident answered it in 96 ms on that repeat. These are synchronized same-GPU first-option timings.

A hindsight router preserving every native-correct decision reduced mean compute from 464 to 405 ms (12.6%). It selected the correction on one request. Its p95 was approximately 2.32 seconds on both policies. The paired improvement met the predeclared development gate and started the separate 20-document test with the same folded checkpoint. The projection's 300 MB shape and the online mapping algorithm were unchanged.

## Follow-up controls

- **Ridge scaling.** The original mapper used `XᵀX/N + 0.01 I` for both keys and values. The matched test reconstructs the same 16,384 calibration states from the exact 32 archived articles, verifies that the original fit matches the serialized weights within 0.1% relative norm, and changes only the penalty to `0.01 diag(var(X))`. The GPU records the scale ratio separately for K and V. This tests whether using a single absolute penalty was suppressing low-variance features.
- **Actual mapped trajectory.** A second functional correction uses composed 13,113–19,990-token training prompts. All constituent source documents come from official QuALITY train and are recorded for overlap checks. Fresh question queries and suffix K/V come from the current mapped branch. The fit matches native readouts before and after the frozen attention output projection. A result would test the combined training change. It would not separate length effects from query-trajectory effects.
- **Question-aware excerpts.** The receiver rereads up to four disjoint 224-token source excerpts ranked using the question and the unlabeled choices. The paired run includes a native excerpts-only path and the same excerpts appended after mapped history. The worker charges the measured CPU retrieval and prompt-preparation time in both arms so the score reflects the additional work.

### Replication on longer documents

The unchanged checkpoint scored **6/20** on the 16.98K–26.39K panel, versus **5/20** for the original hybrid and **9/20** for native 8B. It recovered five of the nine native-correct decisions versus four for the original hybrid, but the simple controls already covered those recoveries. It recovered none of the three failures that required full native prefill, so aggregate and p95 latency were unchanged under the native-preservation hindsight policy (579 ms mean and 3,480 ms p95). This rules out promotion of this checkpoint. The separate nine-record panel remains sealed.

Per-question comparisons show four fixes and zero regressions against the old hybrid in development, and one fix and zero regressions in replication. The original hybrid and correction therefore scored 11/52 and 16/52 on the pooled, a posteriori view. Native 8B scored 22/52. The pooled count is descriptive and insufficient to establish quality equivalence.

[Layer swap](../../reports/session-causal-layers-v1/result.json) · [Joint readout result](../../reports/session-joint-readout-v1/result.json) · [Development rows](../../reports/session-joint-readout-v1/development-rows.json) · [Replication rows](../../reports/session-joint-readout-v1/replication-rows.json)

## Ridge penalty audit and recovery

The original fit reproduced with zero measured relative weight and bias error on the 16,384 frozen observations. Median source-feature variance was 0.8184 for keys and 2.9170 for values. The original absolute penalty 0.01 was 1.22% of typical key variance and 0.34% of typical value variance. This rules out disproportionate shrinkage of the typical value feature as the straightforward explanation. The variance-relative candidate tests a different treatment of individual features.

The refit saved the candidate and its audit before PyTorch rejected its inference-only tensors in the fast mapper. The failed run cost $0.47 in estimated SageMaker compute. Mapper construction now returns regular tensors even when moments were accumulated in inference mode. The recovery worker loads the verified stored checkpoint by checksum and evaluates it without fitting again.

[Ridge covariance audit](../../reports/session-ridge-scale-v1/ridge-scale-audit.json) · [Failure stack](../../reports/session-ridge-scale-v1/failure.txt) · [Recovery receipt](../../reports/session-ridge-replay-v1/submission.json)

The resumed ridge screen finished without another fit. The variance-scaled mapper scored 8/32, against 6/32 original hybrid and 13/32 native 8B. It solved none of the six cases missed by the cheaper controls. The hindsight latency policy was identical with or without it, so scaling alone did not qualify for replication. [Resumed ridge result](../../reports/session-ridge-replay-v1/result.json)

The longer-history run and excerpts control were submitted separately. The excerpts control uses a different single-L40S instance size after SageMaker declined a duplicate flexible quota. The combined upper-bound estimate at submission was $27.94 including earlier billable compute and a $2 allowance for extras. The older student bundle was stopped before GPU allocation to remove the same inference-tensor hazard from the per-document folded mapper. The replacement passed 248 local tests and a separate BF16 training-step smoke.

## Longer-history correction and question-aware excerpts

The mapped-trajectory correction finished all 64 updates. On the first 32 questions it scored **9/32** against **6/32** for the original hybrid and **13/32** for native 8B. It recovered the same 17,428-token case as the earlier correction. The selected transfer timing was 411 ms versus 2,337 ms native cold and 96 ms native cache hit on the first timed repeat. The first-panel hindsight mean reduced from 466 to 406 ms.

On the separate 20 documents it scored **7/20** against **5/20** original hybrid and **9/20** native. It solved none of the three cases still requiring full native prefill, so that confirmed no incremental latency benefit under the native-preservation policy. Longer calibration histories and mapped-path queries improved total choices versus the original, with inadequate replication to justify a serving change.

The question-aware excerpts run gave **11/32** with native excerpts and **12/32** with transferred history plus the same excerpts. The 8B native full-document path scored **13/32**. Median model timing plus measured CPU preparation was **230 ms** for excerpts and **722 ms** for excerpts plus transfer, versus **2,178 ms** native cold. Native excerpts created useful cheaper choices in hindsight, reducing mean from 470 to 335 ms while preserving native-correct answers. Adding transfer did not improve that policy and never recovered a native-correct request beyond all cheaper alternatives.

[Long calibration result](../../reports/session-student-readout-v2/result.json) · [Excerpts result](../../reports/session-evidence-replay-fixed-v1/result.json)

## Donor-layer selection

The controlled layer search refit the original 32 sampled documents with either depth-aligned, two-earlier, or two-later source layers. It selected a single donor layer per 8B layer using readout error on 16 excluded QuALITY training articles. It changed 21 of 36 mappings, with a 1.57% improvement in the local proxy and the same one-donor serving format. LongBench accuracy moved from 6/32 to 7/32. Native 8B scored 13/32. No previously unserved native-correct request was rescued, and the hindsight timing policy stayed at 462 ms mean. This does not support a serving change.

The separate Qwen3-4B → Qwen3-8B control uses an equal-depth pair. Both publisher configs list 36 layers and eight KV heads of width 128. A tokenizer-only preflight verified identical vocabularies and special tokens. The exact frozen 32 training documents, original 512 positions per document and original ridge penalty are reused. [4B publisher config](https://huggingface.co/Qwen/Qwen3-4B/raw/1cfa9a7208912126459214e8b04321603b3df60c/config.json) · [8B publisher config](https://huggingface.co/Qwen/Qwen3-8B/raw/b968826d9c46dd6066d109eabc6255188de91218/config.json) · [Layer result](../../reports/session-source-layer-v1/result.json)

An answer-only join across the first-panel results found that the case both readout corrections rescued was also solved by native excerpts. After adding the excerpts control there were four native-correct cases still requiring full prefill. Neither learned readout correction rescued any of those four. The earlier 31-token tail repaired one of them. That tail effect did not replicate on the second panel. This join compares answers only, since the timings were measured on different workers.

## Sequence-logit distillation

A direct residual fit supervises 8B distributions at eight positions spaced across the question, plus a final choice KL. Both base models stay frozen. The first job hit SageMaker's memory limit before an optimizer update (340 billable seconds, about $0.26). The retry recomputes attention intermediates during backward. Tiny output and gradient parity checks passed, and the retry completed its first update on the 19,990-token prompt. The dataset, mapper rank, and objective were retained.

The depth-matched 4B → 8B job completed its 32-question control. Standalone warm 4B scored **12/32** at 85 ms median. Pure transfer scored **3/32** at 368 ms, and the hybrid with native first 256 scored **9/32** at 417 ms. Native 8B scored **13/32**, with 2,178 ms median cold and 97 ms for a native cache hit. The router's hindsight bound preserved native answers at the same 341 ms mean with or without the translation. Thus changing source size and layer geometry alone did not clear the gate. Estimated compute was $1.16. [4B-to-8B result](../../reports/session-source4b-flex-v2/result.json)

The sequence-distillation retry finished all 32 updates before the paired evaluation began. The final nine-document pool remains sealed. If the candidate passes both development panels, the unseen replay scores the learned mapper against native excerpts and the original transfer alternatives on the same worker.

The sequence retry finished all 32 updates with peak allocated GPU memory of 26.4 GiB. Its constrained first-option accuracy was **11/32**, compared with **6/32** for the original hybrid and **13/32** for native 8B. Only five of native 8B's 13 correct cases were also correct under the new mapper. The correction recovered none of the six cases that all original cheap routes missed. It ran at a **408 ms** median from a resident 1.7B cache, versus **2,201 ms** cold native and **98 ms** for an existing native 8B cache. The accuracy-preserving hindsight route stayed at **478 ms mean and 2,359 ms p95**. The development gate failed, and no independent evaluation was spent on this candidate. The successful retry cost $1.10 estimated compute. [Sequence result](../../reports/session-sequence-kl-remat-v2/result.json)

## Trained 4B-to-8B control

The 4B-to-8B correction finished 64 updates using the same longer-history objective and the matching 4B-origin affine initializer. In the first panel it scored **8/32**, versus **9/32** original 4B hybrid, **12/32** warm 4B, and **13/32** native 8B. It recovered seven of the 13 native-correct cases versus four for the original hybrid. One of four native-correct cases missed by all original cheap routes was recovered. It was also missed by native excerpts in the earlier answer-only comparison. On the first timed repeat of that case, corrected transfer took 430 ms versus 2,419 ms native cold and 98 ms for a native 8B cache hit.

The first-panel oracle preserving the native-correct decisions reduced mean **349 to 286 ms** and p95 **2,372 to 2,069 ms**, with the correction chosen once. This triggered the fixed checkpoint's 20-document replication.

On replication the corrected 4B hybrid scored **6/20**, original hybrid **5/20**, warm 4B **7/20**, and native 8B **9/20**. Both hybrids recovered five of the nine native-correct decisions. The correction recovered the one case all original cheap routes missed. This lowered the hindsight mean **259 to 116 ms** and p95 **323 to 181 ms**. The fixed checkpoint therefore reproduced the oracle latency advantage, while its native-correct retention tied the original mapper. It missed the predeclared overall retention condition. The nine-document unseen panel remains sealed. Training and both screens cost an estimated $1.19. [Trained 4B result](../../reports/session-4b-readout-v1/result.json)

I also screened confidence-threshold policies using the winning option's unconditional probability. Rules paid the sum of the latency of all stages they executed. The development screen required preserving every native-correct decision, keeping p95 within 2% of native cold, and rescuing at least one case with transfer. The fastest qualifying control went straight to native 8B. No tested transfer rule qualified even in development. The two successful switched cases establish capacity for a gain when the right request is known. A router still needs a reliable signal for choosing those requests without the gold answer. [Confidence policy screen](../../reports/session-4b-readout-v1/confidence-policy-screen.json)

Native 8B cache reuse matched the cold 8B answer on all 52 questions in the two latest panels. Median model-only first-option times for the existing native cache were 98 ms and 122 ms on the two panels. The independent local graph replay also verified exact-prefix matching, tenant isolation, and branch independence on its tiny Qwen fixture. [Fallback replay](../../reports/session-native-fallback-check-v1/summary.json)

## Final compute reconciliation

SageMaker compute is estimated at **$26.69** for the project including previous runs, of which **$9.04** was incurred in this session. With the $2 allowance for storage and other billing extras, the estimate is **$28.69** against the $30 ceiling. The project ledger shows zero active SageMaker workers. The final local suite passed 250 tests. [Budget ledger](ledger.json)
