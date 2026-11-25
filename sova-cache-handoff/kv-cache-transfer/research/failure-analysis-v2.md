# Why the larger cache-transfer experiment is not preserving the target

Evidence snapshot: 18 November 2026, job **`sova-kv-full-20261118-132341-7daf`**. This report analyzes saved training and quality artifacts only. No new model inference, benchmark, API evaluation or GPU job was run for this analysis. Only JSON/JSONL artifacts were fetched; no mapper weights were downloaded.

**The problem is real cache/behavior distortion, alongside a separate output-format problem.** The trained mapper recovers the ability to start with an option letter, but mostly answers A regardless of the question. Its 50.82% decision accuracy is close to the 50.05% always-A baseline. The target's native 59.11% behavior is not preserved. This is our own dense Qwen3 experiment; it is not the same model pair, data, training procedure or serving implementation as a vendor's benchmark.

## What actually completed

- All **400 optimizer updates** completed on this corrective run; there were zero recorded overflow attempts. Training took **174.80 seconds**. Both Qwen base models remained frozen.
- The approximately **11.13M-parameter** mapper selected at step **200** had validation objective **0.454965**, versus **11.142175** at initialization. These are a composite cache-reconstruction/logit-distillation objective on training-domain windows, not NLI accuracy.
- Quality evaluation completed **1,037 questions across 61 development contracts** for source, native target, ridge and nonlinear mapper: 4,148 decision attempts, with no recorded errors or over-limit cases.
- Separate generation evaluation completed one seeded question per contract: **61 per variant**, 16 generated tokens maximum.
- Identity-cache controls passed: max logit delta **0.033203125**, identical first-token selection, and a zero-cache control changed logits by up to **23.21875**. This supports the basic import/RoPE plumbing on the tested 128-token identity case; it does not validate every learned cross-model cache.

Sources: [training summary](../reports/sova-kv-full-20261118-132341-7daf/collected/training/training-summary.json), [controls](../reports/sova-kv-full-20261118-132341-7daf/collected/controls.json), [quality summary](../reports/sova-kv-full-20261118-132341-7daf/collected/quality/summary.json), [raw decision traces](../reports/sova-kv-full-20261118-132341-7daf/collected/quality/decision.jsonl). The [diagnostic JSON](failure-analysis-v2-data.json) records trace hashes and derived counts without changing the original scores.

## The headline accuracy hides class collapse

A = Entailment; B = Contradiction; C = NotMentioned. Gold counts are A=519, B=95, C=423. Always predicting A scores **519/1,037 = 50.05%**.

| Variant | Correct / 1,037 | Decision accuracy | Predicted A | Predicted B | Predicted C | Full-vocabulary top token is an option |
|---|---:|---:|---:|---:|---:|---:|
| Source 0.6B | 424 | 40.89% | 1 | 0 | 1,036 | 1,037 |
| Native target 1.7B | 613 | 59.11% | 310 | 0 | 727 | 1,037 |
| Ridge | 517 | 49.86% | 1,019 | 18 | 0 | 0 |
| Trained nonlinear mapper | 527 | 50.82% | 929 | 0 | 108 | 1,034 |

The nonlinear mapper predicts **A on 89.59% of questions**. For **42 of 61 contracts**, it gives A for all 17 different hypotheses. The native target changes between A and C within every contract. This is direct evidence that the mapped path is losing much of the question-sensitive behavior the native path has.

The target itself is also weak: it never predicts B, so contradiction recall is zero even without transfer. Matching this target would establish preservation of a limited baseline, not a high-quality legal system.

### Confusion matrices

Each row is a gold class; the three numbers are predictions A/B/C.

| Variant | Gold | Predicted A | Predicted B | Predicted C |
|---|---|---:|---:|---:|
| Source | A | 1 | 0 | 518 |
| Source | B | 0 | 0 | 95 |
| Source | C | 0 | 0 | 423 |
| Native | A | 245 | 0 | 274 |
| Native | B | 10 | 0 | 85 |
| Native | C | 55 | 0 | 368 |
| Ridge | A | 515 | 4 | 0 |
| Ridge | B | 93 | 2 | 0 |
| Ridge | C | 411 | 12 | 0 |
| Neural | A | 475 | 0 | 44 |
| Neural | B | 83 | 0 | 12 |
| Neural | C | 371 | 0 | 52 |

For the neural mapper, A recall rises to 91.52%, but C recall falls to **12.29%**, compared with native C recall **87.00%**. The mapper has not simply retained the target while making a few uncertain mistakes; its class preference has changed substantially.

## Paired comparisons establish a real behavior gap

On the same 1,037 inputs:

- Neural and native choose the same option on only **354/1,037 = 34.14%**.
- Transfer loses **346** native-correct answers and fixes **260** native-wrong answers: net **86 fewer correct**, or **−8.29 percentage points**.
- The saved document-cluster bootstrap interval for the neural-minus-native accuracy difference is approximately **−14.85 to −1.93 percentage points**. This is development evidence, not a final held-out acceptance test.
- Ridge/native agreement is **307/1,037 = 29.60%**; ridge loses 371 native-correct answers and fixes 275.

The mean native-to-neural KL over the **conditional A/B/C distribution** is **0.7599 nats**; ridge is **2.0289 nats**. These diagnostics show that the nonlinear map moved option distributions toward the target relative to ridge, but still does not preserve them. They are **not full-vocabulary KL measurements**: full logits were not saved, only option logits/probabilities and a full-logit checksum.

## Conditioned option scoring can make broken caches look plausible

The primary decision score deliberately asks which of A/B/C has the highest next-token logit. That is a valid declared decision task, but it can hide what the unconstrained decoder actually wants to say.

| Variant | Median total probability assigned to A/B/C | Unconstrained first-token behavior |
|---|---:|---|
| Source | 99.69% | C almost always |
| Native | Approximately 100% | A or C |
| Ridge | **0.0598%** | `The` on all 1,037 questions |
| Neural | **90.57%** | A/C on 1,034 questions; `</think>` on 3 |

For ridge, restricting the decoder to A/B/C discards almost all of the model's probability mass. Its 49.86% score should never be described as ordinary answer-generation accuracy. The neural mapper substantially repairs this instruction-format signal, but its near-majority accuracy and class collapse show that recovering letter-shaped output is not the same as recovering semantic decisions.

This is a diagnostic interpretation of the frozen prospective metric. The benchmark definition and old pilot headlines have not been changed.

## Generation confirms more than a parser problem

Strict exact-letter accuracy is zero on the 61-question generation sample for every variant. That headline alone conflates different failures:

| Variant | Valid exact-letter outputs | Correct prefix diagnostic | Length-capped at 16 tokens | Typical saved output |
|---|---:|---:|---:|---|
| Source | 1/61 | 24/61 | 0/61 | `C. NotMentioned` |
| Native | 0/61 | 33/61 | 9/61 | `A. Entailment`, sometimes followed by an explanation |
| Ridge | 0/61 | 0/61 | 61/61 | `The following is a summary of the key points...` |
| Neural | 0/61 | 35/61 | **60/61** | `A.C\n\nA\n\nC\n\nA...` or repeated A/C lists |

Native formatting violates the requested one-letter contract but often contains an identifiable NLI label. Neural generation repeatedly emits option letters and is almost always cut off; ridge generates summaries. A more permissive parser would fix part of the **native format score**, but it would not repair the transferred model's real decision collapse or corrupted continuation behavior. Increasing the token cap alone has not been shown to fix either problem.

The 35/61 neural prefix result is a separate diagnostic on a different sample. It does not supersede the full 1,037-question result; this generation sample has **33 gold-A questions** and an always-A baseline of **54.10%**.

## Length and hypothesis diagnostics

The real task prompts range from **519 to 6,471 tokens**. Training windows contain at most **1,032 tokens** including the eight-token tail.

| Prompt length | Questions | Native correct | Neural correct | Always-A correct |
|---|---:|---:|---:|---:|
| ≤1,024 | 70 | 50 (71.43%) | 36 (51.43%) | 29 (41.43%) |
| 1,025–2,048 | 389 | 243 (62.47%) | 169 (43.44%) | 170 (43.70%) |
| 2,049–4,096 | 442 | 253 (57.24%) | 230 (52.04%) | 228 (51.58%) |
| 4,097–8,192 | 136 | 67 (49.26%) | 92 (67.65%) | **92 (67.65%)** |

The apparent improvement in the longest bin is explained exactly by **predicting A on all 136 questions**. It is not evidence of better long-context reasoning. Conversely, there is already a sizable neural gap within the training-length range, so length mismatch alone cannot explain the failure.

Examples of hypothesis dependence, each with 61 questions:

| Hypothesis | Native correct | Neural correct | Interpretation |
|---|---:|---:|---|
| nda-1: explicit identification | 44 | 21 | Large loss, despite this being the representative training hypothesis |
| nda-3: verbally conveyed information | 53 | 41 | Target behavior partly lost |
| nda-5: sharing with employees | 6 | 51 | A-default can look like a major improvement when this hypothesis is commonly entailed |
| nda-11: reverse engineering | 60 | 12 | Strong target behavior mostly destroyed |
| nda-17: permissible copying | 43 | 13 | Substantial loss |
| nda-18: no solicitation | 49 | 16 | Substantial loss |
| nda-20: retained information | 39 | 13 | Substantial loss |

All 17 hypothesis counts are retained in the [diagnostic JSON](failure-analysis-v2-data.json). These are descriptive slices with different class distributions, not independently selected acceptance benchmarks.

## Likely contributing causes, and what remains hypothetical

| Finding or hypothesis | Status | Evidence and implication |
|---|---|---|
| The training/selection pipeline reduces each document to one representative question | **Verified implementation gap** | `_unique_documents` keeps the first example. Because prepared records are sorted, all 64 train and 16 validation representatives are **nda-1**. The prepared 1,088/272 QA counts therefore overstate question diversity actually consumed by the current optimizer. Document deduplication is right for leakage control; it should not erase query diversity in query-conditioned distillation. |
| The training objective is not the reported legal decision task | **Verified protocol difference** | The loss is normalized KV reconstruction plus full-vocabulary next-token KL on cropped text windows. Gold labels are not used, and most random windows need not contain the question/answer boundary. A low window objective can coexist with poor full-prompt option behavior. |
| We map almost the entire question/instruction state and let the target read only one token | **Verified boundary; causal effect untested** | `DenseRuntime.answer` translates `ids[:-1]` and target-prefills `ids[-1:]`. The mapped prefix includes the question, choices, answer instruction and chat/assistant formatting. A good map could preserve these; this one demonstrably does not. Letting the target process the question/instructions natively after importing only a document prefix is a plausible repair, not an established result. |
| Short training windows restarting positions generalize poorly to full contracts | **Verified mismatch; causal share unmeasured** | Up to 1,032 training tokens versus real prompts up to 6,471, with window positions restarted at zero. The gap also occurs below 1,024, so this is not a sufficient explanation by itself. |
| The initial mapping family lacks enough cross-model alignment capacity | **Plausible, not established by an ablation** | The ridge is per-head and depth-aligned; neural residuals read fixed nearby layers. The experiment did not test learned layer selection, alternative cross-head maps or a stronger published mapper under a matched protocol. |
| The basic cache import is completely broken | **Not supported as the sole explanation** | The identity control passes and no import-length mismatch/fallback is recorded. Learned mapping quality can still fail despite correct cache plumbing. A 128-token identity control does not prove all long-context numerical behavior. |
| FP16 overflow caused the completed run's poor decision scores | **Not supported by current traces** | This corrective run records 400 completed updates and zero overflow attempts. Other numerical approximation effects remain possible, but the previous failed job's overflow is not evidence that this run's updates were skipped. |

## Next discriminating experiments

These are proposals, not executed fixes:

1. **Document-only transfer, native query suffix.** Import a mapped cache for an explicitly defined document prefix, then let the target process the complete question, choices and assistant framing natively. Count every native suffix token and its latency. This requires a new declared timing/import contract; do not silently compare it as if it still recomputed one token.
2. **Keep document-level isolation while sampling all training hypotheses.** Deduplicate document storage and KV statistics, but draw question-conditioned teacher examples across all 17 hypotheses. Keep mapper selection independent from official dev, and record the actual consumed question distribution.
3. **Add a matched answer-boundary distillation objective.** Compare teacher and mapped distributions at the real full question boundary, alongside cache reconstruction. Retain full-vocabulary option mass and generation checks, not only conditional accuracy. The target's own contradiction weakness must remain visible.
4. **Separate alignment, length and boundary ablations.** Compare the current map, native-query-suffix map and a stronger alignment baseline on a frozen diagnostic subset with fixed precision/context lengths. Test shorter contexts first, then broader length coverage; preserve current results.
5. **Require behavior retention before kernel optimization.** Check native/mapped agreement, per-class recall, paired quality gap and generation degeneration before treating a faster cold path as useful. A 50% score from class collapse is not a satisfactory gate.

The first question for the next run should be **“does the target recover question-sensitive behavior when it reads the query itself?”**, not “can a larger batch or a faster kernel improve the existing collapsed predictions?”

## Additional verified audit: what the old eight-token tail supervised

The pinned-tokenizer [audit](../reports/sova-kv-full-20261118-132341-7daf/collected/tokenizer-audit/audit.json) identifies the old prompt-ending eight tokens as `[198, 151644, 77091, 198, 151667, 271, 151668, 271]`, decoding to `\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.

For an eight-position causal next-token KL on that prompt-ending window, the first seven positions predict further fixed chat-template tokens; only the last position predicts the first answer token. This makes the supervision-position distribution poorly aligned with a claim about answer behavior. It does **not** establish that seven eighths of the measured loss came from formatting: per-position KL was not saved, and KL magnitudes differ by position. The new answer-token-supervision experiment should be labeled a protocol change, with the old result retained.
