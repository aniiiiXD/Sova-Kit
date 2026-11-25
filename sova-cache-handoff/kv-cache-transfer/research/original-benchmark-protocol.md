# Matching the original benchmark: verified protocols and next dataset

Verified on 18 November 2026. Sources below were fetched again for this audit; vendor claims are not treated as independently reproduced results. No model evaluation or cloud job was launched for this report.

## What the completed SOVA run establishes

The [collected quality summary](../reports/sova-kv-full-20261118-132341-7daf/collected/quality-summary.json) contains all 1,037 ContractNLI development questions over 61 contracts, with no errors or over-limit exclusions. Decision accuracy is 40.89% source, 59.11% native target, 49.86% ridge and 50.82% neural. The majority-label baseline is 50.05%. Neural minus native is −8.29 percentage points, with the saved document-cluster interval [−14.85, −1.93] points. Neural macro class accuracy is 34.61%.

The failure is not merely a strict parser:

| Variant | Predicted A | Predicted B | Predicted C | Meaning |
|---|---:|---:|---:|---|
| Source | 1 | 0 | 1,036 | Nearly constant NotMentioned prediction |
| Native | 310 | 0 | 727 | Better overall but never predicts Contradiction |
| Ridge | 1,019 | 18 | 0 | Nearly constant Entailment prediction |
| Neural | 929 | 0 | 108 | Strong Entailment bias persists |

These counts were read from the saved decision JSONL. In free generation, native examples often emit interpretable labels with descriptions, such as `C. NotMentioned`. Neural examples instead repeat sequences such as `A.C` and subsequent letters. All arms score zero under the exact-one-letter contract, while neural exhausts the 16-token cap on 60/61 questions. Relaxing the parser could make native formatting less punitive; it does not fix the neural continuation degeneration.

The [training summary](../reports/sova-kv-full-20261118-132341-7daf/collected/training/training-summary.json) confirms 400 completed updates, frozen base models, 64 optimization documents, 16 mapper-selection documents, 11,125,856 mapper parameters, and best checkpoint at update 200. That proves execution, not preservation of native behavior.

## There are three different “original” comparisons

### A. Vendor generic mapper, task-specific mapper and newer API campaign

The vendor's benchmark page loads its numeric protocol from a public JavaScript data asset. This audit read that asset as text; it did not execute downloaded code. The following are distinct displayed campaigns:

| Vendor campaign | Published pair/protocol | Selected published result |
|---|---|---|
| Generic dense pair | Qwen3 1.7B → 8B; FineWeb-Edu-only adapter; stage one 200 documents × 32,768 tokens, stage two up to 1,280 × 1,024 | QuALITY n=500, reasoning off: 55.5% source, 59.5% transfer, 77.0% native |
| Task-specific dense pair | Separate benchmark-trained adapters; QuALITY listed as 209 training “documents,” n=500, reasoning on | 63.8% source, 72.0% transfer, 83.8% native |
| Generic hybrid headline | Qwen3.5-4B → Qwen3.8-27B; headline n=100 | QuALITY 78% source, 84% transfer, 94% native |

The vendor's separate fine-tuned mapper API page advertises task-trained hybrid QuALITY at 85% source, 90% mapper, 90% native, with 100 held-out questions and document-grouped intervals. It describes teacher-generated task answers and a rank-16 adapter, recommends at least 1,000 questions over 50 documents, and says gold references are for evaluation. Those figures do not replace the generic campaign above.

The vendor's newer downloadable hybrid benchmark bundle, generated 16 November with an expanded-results snapshot dated 13 November, reports ContractNLI 61/100 mapped versus 79/100 native. Its task-matched τ2 slice is 25/51 mapped versus 46/51 native; the source comparison uses an explicitly labeled 9B proxy. These are useful counterexamples to a universal quality-preservation claim. They are not our scores and not the task-trained QuALITY campaign.

The vendor's benchmark analysis explicitly distinguishes generic and task-specific training. Choosing a convenient score from one campaign and timing from another would not constitute a reproduction.

### B. Original NVIDIA paper

[Heo et al., arXiv:2608.03893v1](https://arxiv.org/html/2608.03893v1) uses FineWeb-Edu calibration (500 sequences × 1,024 tokens, stride-four token observations), selected source layers and cross-head production ridge features. HellaSwag is zero-shot and scored with lm-evaluation-harness `acc_norm`, not letter generation. Its WikiText-2 diagnostic uses non-overlapping 2,048-token chunks: score the second 1,024 tokens conditioned on a native or transferred first-1,024-token cache. Ordinary rolling perplexity would be a different metric.

This is the cleaner public mechanism reference, but our same-head k1 ridge, fixed nearest-depth topk3 residual, ContractNLI windows and A/B/C scorer are not that protocol. Failure of our first candidate cannot be described as disproving this paper.

### C. Independent implementation of our exact small pair

[alesha-pro/vllm-cross-model-kv](https://github.com/alesha-pro/vllm-cross-model-kv) independently implements Qwen3-0.6B → 1.7B with a selected-layer k8 mapper and 500-sequence calibration. Its HellaSwag and tail-perplexity protocol is a useful external implementation target. It is not the NVIDIA authors' release or the vendor's implementation, and its reported outcomes have not been reproduced by us.

## What we can and cannot match now

| Requirement | Current SOVA | Needed for the corresponding claim |
|---|---|---|
| Same benchmark family as vendor | Already has ContractNLI; QuALITY is the clearest next long-document task | Freeze the official dataset and a prospective question sample before outputs |
| Same model pair | 0.6B → 1.7B | Vendor dense 1.7B → 8B or hybrid 4B → 27B; the latter needs recurrent/convolution state adapters |
| Same training distribution | Contract-only windows; eight-token KL targets | Generic FineWeb calibration or task-answer distillation, explicitly separated |
| Same query boundary | Maps nearly the whole formatted prompt and recomputes one tail token | State whether only the article is transferred and the receiver natively reads the question/options |
| Same score | Restricted first-token A/B/C ranking, separate strict generation | Exact vendor prompt/scorer for vendor comparison; continuation likelihood for HellaSwag |
| Same sample IDs | Unknown vendor IDs | Obtain vendor sample IDs, hashes, grouping rules and exclusions; otherwise call ours an independent matched-task evaluation |
| Same systems boundary | HF batch-one, both models resident | Match device/precision, source-prefill inclusion, import cost, concurrency and repeated measurements |

One unresolved detail is important: official QuALITY v1.0.1 training has 150 distinct `article_id` values, whereas the vendor dense task-specific tab describes 209 training “documents.” The latter could refer to a different grouping or partition; this audit cannot resolve it. Do not silently equate those numbers.

The vendor's Prefill SDK is a hosted client, not training code or raw cache access. Its 27B interface is blocking, so client wall time is total completion latency unless an independently documented server trace exposes first-token timing. We have not obtained that service's credentials or private mapper internals.

## Recommended next benchmark: official QuALITY train → development

For the user's “same company benchmark” goal, use **QuALITY**, not a second reformulation of ContractNLI. Train/calibrate from official training articles and evaluate a frozen independent development sample. This matches the task family and document-disjoint idea, while remaining honest that it is not the vendor's unpublished exact item set or production model pair.

The [author-maintained QuALITY repository](https://github.com/nyu-mll/quality) at commit `f84977c40dbfef70c9cab48037b7becfc8e45f73` supplies `QuALITY.v1.0.1.htmlstripped.train` and `.dev`. Direct inspection of those bytes gives:

| Split | JSONL records | Unique articles | Questions | Difficult questions | SHA-256 |
|---|---:|---:|---:|---:|---|
| Train | 300 | 150 | 2,523 | 1,251 | `4011e9952d5395beb8ff7637b963481a400630c1bbe2f40dc0d83f5d59f926ed` |
| Development | 230 | 115 | 2,086 | 1,065 | `99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef` |

Two writer records can share an article. Grouping by record or question instead of `article_id` would leak a document across splits. `gold_label` is 1-indexed; options contain four answer strings; report both all-question and difficult-subset accuracy. Official test annotations are withheld and are not part of this preparation.

### Frozen prospective protocol

1. Preserve all 2,523 official training questions and 150 article groups. Separate a train-only article-group validation split for mapper selection before training. Do not deduplicate away all but one question per article in the functional-training index.
2. Freeze **100 questions from 100 distinct official development articles**, seed 19, one seeded question per selected article. Record every article/question ID, hard flag, context hash and full token length. This is a reproducible independent n=100 slice, not “the same 100 as the vendor.”
3. For a task-trained mapper, generate target teacher answers on the training questions with a declared reasoning mode and sufficient output budget; use those traces for receiver-level distillation. Do not assume eight arbitrary next-text positions train the target's answer behavior. Retain a generic FineWeb-Edu calibration-only mapper as a separately named control if budget permits.
4. Transfer the shared article prefix, then let the receiver read the fresh question and four options. Log how many receiver tokens are recomputed; this is distinct from our earlier one-tail-token protocol. All variants get identical article/question text and tokenization.
5. Declare the decision scorer before running: select the maximum A/B/C/D option logit with option probability mass recorded, and report unconstrained generation separately using an anchored label parser plus explicit invalid/truncated categories. A strict-one-letter metric may remain a product-contract diagnostic; it must not be presented as the only semantic accuracy metric.
6. Keep all 100 selected examples in the denominator. A length exceeding the declared cap is an explicit failure, not a deleted row. Use document-level paired intervals, majority/chance baselines and source/native/mapped arms. Increase to all 2,086 development questions only after the small frozen protocol and native control behave sensibly.

At n=100, confidence intervals will be wide. It is adequate for finding a catastrophic transfer problem and for matching the scale of a vendor screen; it is not enough to establish a tight three-point noninferiority claim by default.

## Paper-matched alternative: HellaSwag continuation scoring

Prepare HellaSwag validation as the independent mechanism check, using [author dataset `Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) and pinning its revision. A seeded 1,000-example subset is a development diagnostic; the full official validation split remains a different, larger run.

Use the [pinned harness task](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag/hellaswag.yaml) and [preprocessing](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag/utils.py). Each of four ending strings is teacher-forced after the same context. Sum log probabilities over every ending token; report raw `acc` and normalized `acc_norm`. At this harness revision, normalization divides by the processed ending's Python string length in characters, not its token count; see [scorer implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/api/task.py). Preserve the harness's whitespace/token-boundary handling and zero-shot prompt construction. Do not turn this into “generate A/B/C/D.”

For the calibration reference, pin 500 unique training documents from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), extract 1,024-token prefixes, and use the documented token subsampling for the ridge fit. Do not mix validation examples into selection or calibration. Selected-layer/full-cross-head support is still needed before labeling a mapper a paper reproduction.

This benchmark is useful for quality comparison even though its short prompts are a poor setting for demonstrating a long-context speed advantage. Keep a separate matched long-prefix timing experiment; synthetic timing does not establish long-context task accuracy.

## Budget and execution gate

No new run is authorized by this report itself. The current cumulative cloud limit and compute rules remain controlling. At the previously verified Ohio rate of $1.515/hour, a one-hour worker bound is approximately $1.52 compute and a two-hour bound $3.03, before storage, transfer, setup/billing overhead and tax.

For the next scoped candidate, first profile a small fixed native/mapped subset and estimate the complete n=100 QuALITY or n=1,000 HellaSwag cost from actual forwards and generation lengths. Set a hard worker limit and stop if the native control is degenerate or correctness tests fail. Do not promise that all teacher traces, mapper retraining, full benchmark scoring and kernel work fit the same two-hour budget without measurements.

The first acceptance gate is **a sensible native baseline plus a mapper that avoids class collapse and generation loops**. Only then is it useful to seek an API latency win. Switching dataset names without repairing the method and protocol would obscure the present failure.

## Reference data prepared after this audit

The frozen files now exist under [`datasets/reference-benchmarks-v1`](../datasets/reference-benchmarks-v1/README.md). Their [manifest](../datasets/reference-benchmarks-v1/manifest.json) pins upstream revisions and all source/processed hashes; [selection.json](../datasets/reference-benchmarks-v1/selection.json) fixes the sampled IDs; [validation-evidence.json](../datasets/reference-benchmarks-v1/validation-evidence.json) records 14 successful hash checks, schema parsing and split checks.

- QuALITY training: 2,523 questions / 150 articles; complete development: 2,086 / 115; selected development: 100 questions / 100 distinct articles, with 43 difficult questions. Selected labels are A=21, B=23, C=30, D=26.
- HellaSwag official validation: all 10,042 rows plus 1,000 seed-19 examples, the latter spanning 977 source groups. Original `ind` values are not unique, so immutable pinned-parquet row positions identify examples while preserving original IDs, context and ending strings.
- [`scripts/prepare_reference_benchmarks.py`](../scripts/prepare_reference_benchmarks.py) reproduces preparation using an existing PyArrow installation; no Torch import or dependency installation is involved. Three tiny serialization/sampling tests passed. QuALITY's Unicode line separators are JSON-escaped without changing decoded article text, avoiding accidental record splitting.
- Licensing claims and uncertainty are preserved in the manifest, including per-article QuALITY license information and the HellaSwag card's MIT attribution. No commercial-use guarantee is implied.

No official test file was accessed, and no model was trained or evaluated in preparing these references.
