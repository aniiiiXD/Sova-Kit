# Generated-answer continuation pilot

**Outcome.** The 64-document run completed and exported the mapper. It scored 1/12 on the untouched screen at 611 ms median first-token compute. The archived single-donor control scored 2/12 at 230 ms. The pilot produced no unique rescue and gives no positive signal for scaling this mapper.

The 1.7B and 8B Qwen3 checkpoints stay frozen. Each 8B attention layer receives the original affine prediction plus a trainable rank-16 residual from the three nearest 1.7B layers. The residual starts at zero and its checkpoint is tied to the archived affine weight hash.

## Frozen data

The data uses the authors' QASPER 0.3 train and development splits. The inputs retain the released full paper text and omit questions without a human-annotated extractive answer inside a supporting text paragraph beyond the first 512 document tokens. Each selected paper contributes at most one trajectory. The frozen pool contains 170 training candidates with 221 candidate questions. The 12 development papers have disjoint official IDs, question IDs, and normalized document hashes. They stay out of optimizer and teacher selection. The earlier LongBench holdout files remain sealed. This first screen is specific to extractive research-paper QA. [QASPER release](https://huggingface.co/datasets/allenai/qasper) · [Frozen inputs](../../datasets/qasper-continuation-pilot-v1/manifest.json)

## Training and evaluation

The native 8B teacher generates each answer greedily from the full document and the question. An answer qualifies when it ends, contains at least three emitted answer tokens, and matches a human annotation after case and whitespace normalization. The audit records all attempted questions. Once 64 papers qualify, the worker stores their generated token IDs and full-vocabulary per-step log probabilities. The student receives the same question and the teacher's preceding answer tokens at each step. Forward KL updates only the mapper residual.

The development screen scores free-generation exact match on cold 8B, exact native cache reuse, warm 1.7B, 8B with the first 256 tokens, the original single-donor affine transfer, and trained multi-donor transfer. The main quality count is the number of native-correct questions the candidate rescues after the cheaper controls miss. Timings are synchronized, single-process first-token compute measurements from resident models and measure one replay per route. This pilot combines extra donor inputs and generated-answer training, so a gain would measure their combined effect.

The queued SageMaker worker has a maximum compute reservation of $1.28. Prior project compute and the $2 allowance bring the projected ceiling to $29.97. [Submission receipt](../../reports/answer-continuation-pilot-v1/submission.json)

## QuALITY retry

The QASPER run stopped after 64 candidate papers produced seven verified answers. Its 385 billable seconds cost an estimated $0.30. This observed yield was too low to reach 64 verified papers efficiently under the cap.

The retry uses 150 human-graded QuALITY training documents with up to four candidate questions per article. A teacher trace qualifies only if the native 8B model independently generates the correct letter and the complete corresponding option text. It explicitly trains on the initial decision token as well as the continuation, with half the loss weight on the first token and half on the mean over all generated positions. The snapshot uses 2,431–8,559-token training prompts and a separate 12-document screen with six previously unused LongBench v2 multiple-choice tasks and six unopened QASPER extractive tasks. The existing nine-document LongBench holdout remains untouched. The evaluation therefore measures a small mixed-task pilot. [QuALITY source](https://github.com/nyu-mll/quality) · [Frozen retry data](../../datasets/quality-continuation-pilot-v2/manifest.json) · [Retry submission](../../reports/answer-continuation-quality-v1/submission.json)

The retry qualified 64 documents from 65 attempted articles and 106 questions. It saved the full-vocabulary distributions for 585 emitted positions including end tokens. Verified trajectories contained 6–15 generated tokens including the end token, with a median of 9. The accepted prompt lengths were 2,431–8,314 tokens. The QuALITY annotation verifies the selected answer and the complete option text displayed in the prompt. This selection favors questions the teacher can solve, and the continuation includes option copying.

## Held-out results

| Route | Exact generated answers | Median first-token compute |
| --- | ---: | ---: |
| Cold native 8B | 2/12 | 1,030 ms |
| Native 8B cache hit | 2/12 | 64 ms |
| Warm 1.7B | 2/12 | 33 ms |
| Native 8B with first 256 document tokens | 1/12 | 78 ms |
| Original single-donor transfer | 2/12 | 230 ms |
| Trained three-donor transfer | 1/12 | 611 ms |

Native 8B answered only one question that all cheaper controls missed. Three-donor transfer missed that question too. Each path had one timed trial per held-out document. Cold and native-cache-hit 8B emitted exactly the same answer on 11/12 questions. A diagnostic letter-only rescore of the six multiple-choice questions preserved the accuracy ordering. Native 8B scored 2/6 and trained transfer scored 1/6. Native 8B produced zero exact answers to the six QASPER questions. The small mixed screen and weak native score limit what we can infer about the effect of more training data.

The successful run cost $0.61, with $0.30 for the stopped QASPER run. Estimated project compute is $27.60, or $29.60 including the existing $2 billing reserve. The project has zero active managed workers. The local residual checksum matched the worker's receipt, and the 177,771,792-byte teacher probability archive is in S3. The local suite passed 253 tests. [Verified results](../../reports/answer-continuation-quality-v1/result.json) · [Teacher trajectory index](../../reports/answer-continuation-quality-v1/verified-teacher.json) · [Saved mapper](../../reports/answer-continuation-quality-v1/checkpoint-64/manifest.json) · [Budget ledger](ledger.json)
