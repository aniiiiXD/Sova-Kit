# What the cache mapper learned before we stopped it

20 November 2026 · Qwen3 1.7B → 8B · QuALITY development data

**The updated mapper got closer to the native model’s probability distribution. It still got 14 of 30 answers right.** Probability on the four answer letters rose sharply, choice probabilities became more like the teacher’s, and two correct answers replaced two others. Native 8B got 19/30.

## What we measured

Both checkpoints used the same 30 questions from 15 previously scored articles. I recomputed the winning letter from each arm’s saved option logits and checked it against the frozen gold label. The raw rows contain the four option logits and the worker’s full-vocabulary KL. The option KL can be recomputed from the saved probabilities and log probabilities. [Step 0 rows](../../reports/answer-mapper-loop-v4/stopped-evidence/validation/step-000000.json) · [Step 32 rows](../../reports/answer-mapper-loop-v4/stopped-evidence/validation/step-000032.json) · [Frozen evaluation data](../../datasets/loop-holdout-v1/quality.jsonl)

| Metric | Step 0 | Step 32 |
| --- | ---: | ---: |
| Gold answers correct | 14/30 | 14/30 |
| Same answer as native 8B | 16/30 | 15/30 |
| Mean full-vocabulary forward KL | 5.257 nats | 1.653 nats |
| Mean option-only forward KL | 4.468 nats | 1.510 nats |
| Median probability that the next token is one of the answer letters | 50.2% | 95.8% |
| Median gold logit minus highest wrong-option logit | −0.750 | −0.125 |

Eight predicted letters changed. Two answers went from wrong to right, two went from right to wrong, and four moved between wrong choices. At each checkpoint the mapper missed seven questions that native 8B got right and solved two that native 8B missed. At step 32 it tied the 256-token native-prefix-only control at 14/30. The 1.7B standalone control scored 12/30. The development screen failed, and the differences are descriptive on a small, reused panel. [Checkpoint history](../../reports/answer-mapper-loop-v4/stopped-evidence/validation-history.json)

The latest synced training log ends at update **82**. The median loss was **1.709** across updates 1–32 and **0.829** across updates 51–82. These were different training examples, so these medians are less informative than the comparison on the same 30 validation questions. The stop came before the scheduled step-96 evaluation and export. The available artifacts contain no step-82 accuracy or exported trained mapper. AWS recorded 1,637 billable seconds, about **$2.57** at the logged Ohio rate. No follow-on latency benchmark was submitted. [Training trace](../../reports/answer-mapper-loop-v4/stopped-evidence/training.jsonl) · [AWS terminal state](../../reports/answer-mapper-loop-v4/stopped-evidence/aws-terminal.json) · [Submission and rate](../../reports/answer-mapper-loop-v4/training/submission.json)

## What the loss actually asks for

The **teacher** is native 8B reading the whole document and question. The **student** is the same frozen 8B reading the same question with a hybrid cache. The first 256 document tokens are native and the rest come from translating the 1.7B cache. Only the rank-16 correction to the affine mapper learns. Both Qwen checkpoints stay frozen. Each training question provides one supervised output position, the token where the answer is expected. [Implementation](../../kvverify/answer_mapper.py) · [Frozen worker protocol](../../reports/answer-mapper-loop-v4/stopped-evidence/protocol.json)

Let `p` be the native 8B probabilities at that position, `q` the hybrid-cache 8B probabilities, and `O = {A,B,C,D}` the option tokens. The loss is:

```text
KL(p || q) = sum_i p_i log(p_i / q_i)

L = 0.5 × KL(p || q)
  + 0.5 × KL(p conditioned on O || q conditioned on O)
```

- **Full-vocabulary KL** penalizes the student for putting too little probability on tokens the teacher favors. It cares about the four answer letters and every other possible next token. At fixed teacher probabilities it has the same gradients as cross-entropy with soft targets.
- **Option KL** renormalizes over the four letters. It asks whether the student allocates probability *within* those options the way the teacher does. The option KL is unchanged if all four option logits move up or down together.
- **Gold accuracy** counts the questions on which the highest option logit corresponds to the labeled answer. The gold labels supply no gradient in this run. Matching a teacher that mistakes an answer can also preserve its mistake.

The terms overlap. Full KL already includes a term for how much probability falls on any answer letter, a within-options KL weighted by the teacher’s option mass, and an analogous term for the remaining vocabulary. The second term gives within-option behavior extra weight even when the teacher’s option mass is small. See [Hinton et al. on soft and hard targets](https://arxiv.org/abs/1503.02531).

**Why the loss can fall without a score increase.** Suppose B is correct and the candidate’s conditional probabilities for A/B/C/D move from `0.70/0.20/0.05/0.05` to `0.45/0.40/0.075/0.075`. If the teacher favors B, the second distribution can be much closer to it while A still wins. A question only changes from wrong to right when its gold margin crosses zero. The recorded median margin moved toward zero, but the paired answers show as many regressions as recoveries.

## Which loss would address which failure?

| Objective | Signal it gives the mapper | What we would check |
| --- | --- | --- |
| Cache reconstruction, such as squared error on K and V | Reproduce native cache coordinates. Useful for the ridge initializer. | Whether the reconstructed state produces the right attention output on evidence-bearing tokens. Errors on heavily attended tokens can matter more than many small errors elsewhere. |
| First-answer KL, current run | Match teacher probabilities at the decision position. | Whether gains carry through to gold margins, answer accuracy, and the target’s ability to use the article. |
| Gold option cross-entropy, `−log q(gold given O)` | Push the labeled option up relative to the other three. A hinge on the gold margin could instead focus on rankings that are wrong or close. | Whether gains hold on new documents and with perturbed answer order. This directly targets multiple-choice accuracy and leaves general generation to a separate check. |
| Attention-output loss, comparing `softmax(QKᵀ/√d)V` with native | Teach the mapper to preserve what 8B reads at particular layers and question positions. | Whether fixes survive a full receiver forward, where the receiver’s queries can drift as earlier layers change. |
| Continuation KL over several positions | Match how the receiver consumes the cache beyond the first answer token. | Whether the added supervision improves evidence use and generation enough to justify its training cost. |

CacheBridge supplies a precedent for attention-output calibration. KV-Lingo provides a separate precedent for continuation-level distillation. Both motivate tests on this pair without establishing which objective will improve its accuracy. [CacheBridge §4.2](https://arxiv.org/html/2609.00891#S4.SS2) · [KV-Lingo §3.2](https://arxiv.org/html/2609.32610#S3.SS2)

## Revised thesis and the experiment that can move it forward

**Working hypothesis.** At a handoff where a 1.7B cache already exists and 8B has a cache miss, a useful translator must preserve the 8B computation that retrieves answer-bearing evidence. If that behavior can be recovered with a compact correction, the saved prefill must outweigh translation, cache movement, and any native state we reconstruct.

The completed evidence establishes an **objective–accuracy gap** for this run. It leaves open whether the bottleneck is key addressing, value readout, the source state or mapper capacity, or supervision at a single output position. The speed work measured different weights. [Historical timing report](../../reports/ttft-fast-affine-v2/RESULTS.md)

1. **Localize the failures before more training.** Use the archived, reproducible affine initializer and the frozen 30-question development panel. Keep the native first 256 tokens and the exact question positions fixed. Compare mapped history, native keys only, native values only, and fully native history. Require the fully native branch to match ordinary native inference before interpreting the partial swaps. Inspect the seven native-correct/mapper-wrong questions alongside all 30. This diagnoses the initializer because the stopped run did not export its trained weights.
2. **Measure what the intervention repairs.** For each arm, record the gold margin, which answers flip, within-option KL, and probability assigned to valid answer tokens. Key-only rescue would support an addressing problem. Value-only rescue would support a readout problem. A large joint rescue with weak individual effects would support an interaction. Repeat with short, equal-size history windows around independently annotated evidence and elsewhere as a control. Full native swaps are diagnostic upper bounds that require native prefill.
3. **Choose one objective change after that result.** If a cache intervention recovers the missing answers, test the smallest corresponding key, value, or attention-output correction. If a candidate preserves evidence use and reliably ranks the correct letter below a distractor, compare the current loss with a gold-option term on the same data and compute budget. If short windows cannot recover useful behavior cheaply, reconsider the representation or the model pair before lengthening this run.
4. **Test the actual claim once.** Freeze the candidate, evaluate on independently held-out articles, and time the same candidate against native 8B cold prefill and native cache reuse. Count cache movement and suffix compute. The first question is whether accuracy holds. The second is whether the handoff is faster under conditions where it would be used.

The probe reuses the existing native K/V substitution mechanism. This update analyzed saved outputs and did not run a new GPU experiment. [Component probe design](../autonomous-loop/COMPONENT_DIAGNOSTIC_PLAN.md) · [K/V intervention implementation](../../kvverify/native_prefix.py)
