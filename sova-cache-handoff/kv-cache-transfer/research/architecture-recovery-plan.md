# Architecture audit and recovery plan

18 November 2026. Source inspection and arithmetic over existing saved traces only; no new model evaluation, training or paid job was launched for this audit. Proposed diagnostics below are not completed results.

## Main finding

**Recover an interpretable cross-head, selected-layer ridge baseline before expanding the neural architecture or model size.** In parallel, test the current saved mapper's handoff boundary: training gave the target eight freshly computed suffix tokens, while inference gave it one. We have substantial implementation/protocol differences from the published method and a narrow functional-training distribution. The current result does not establish that the general cache-transfer idea fails.

Three claims must remain separate: reproducing a published method on our smaller pair; reproducing a paper's exact model pair, data and scoring; reproducing the vendor's private hybrid-pair service. Today's work is an independent small-pair experiment, not either exact product reproduction.

## What the actual larger run says

Evidence: [quality summary](../reports/sova-kv-full-20261118-132341-7daf/collected/quality-summary.json), [raw decisions](../reports/sova-kv-full-20261118-132341-7daf/collected/quality/decision.jsonl), [training summary](../reports/sova-kv-full-20261118-132341-7daf/collected/training/training-summary.json), [controls](../reports/sova-kv-full-20261118-132341-7daf/collected/controls.json).

| Variant | Decision accuracy | Macro class accuracy | Predicted labels | Median option probability mass |
|---|---:|---:|---|---:|
| Native target | 59.11% | 44.73% | A: 310; C: 727 | 1.0000 |
| Source | 40.89% | 33.40% | A: 1; C: 1,036 | 0.9969 |
| Ridge | 49.86% | 33.78% | A: 1,019; B: 18 | 0.000598 |
| Neural | 50.82% | 34.61% | A: 929; C: 108 | 0.9057 |

There are 1,037 examples from 61 documents; always choosing A scores **50.05%**. Neural predictions agree with the native target on only **34.14%** of examples. Its paired accuracy difference is **−8.29 percentage points**, with the recorded document-cluster interval **[−14.85, −1.93] pp**. This is genuine decision degradation, not merely strict output-format rejection. Native also never predicts B, so the reference task/model is itself weak on contradiction.

The neural mapper completed 400 optimizer updates; validation selected step 200 with objective 0.454965. That objective improvement did not preserve the downstream task. Unconstrained neural generation commonly repeats fragments such as `A.C` rather than producing a stable answer; all four variants fail the strict one-letter generation metric on the 61-document sample. These are distinct findings.

The 24-example repeated latency sample records median native cold TTFT 183.67 ms versus neural transfer 189.20 ms; median paired speedup 0.953x. Synthetic longer inputs show small gains, but requested length names are not exact token counts and those repeated fillers are not long-context quality benchmarks.

## Confirmed protocol/design problems

### 1. Training and inference use different handoff boundaries

[training_loss](../kvverify/train_mapper.py) maps the prefix and recomputes **eight** target tokens, supervising their next-token distributions. [DenseRuntime.answer](../kvverify/hf_backend.py) maps all but the final prompt token and recomputes **one**. At the training answer position, seven additional recent target K/V positions exist compared with the inference boundary.

The pinned-tokenizer audit confirms the last eight input IDs are `198,151644,77091,198,151667,271,151668,271`, decoding as `\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. Thus the first seven teacher-forced prediction positions supervise fixed template continuations; only the eighth predicts the first answer token. Their per-position losses should be reported separately. This is a confirmed boundary/objective mismatch plus a testable explanation of why average KL can fall without improving first-answer decisions; it is not proof that it is the only cause. [Pinned-tokenizer audit](../reports/sova-kv-full-20261118-132341-7daf/collected/tokenizer-audit/audit.json).

**First diagnostic:** use the same saved mapper on full, identical prompts with target suffix lengths 1, 8 and 32. Map exactly `N−suffix` positions, log both counts, and compare complete first-token distributions against native target. If suffix 8 recovers strongly, align future functional training and deployment boundaries. Any extra target computation belongs in the latency total.

### 2. Deduplication accidentally removes question diversity from functional training

Prepared train data has 1,088 QA records and 17 question types. `_unique_documents` keeps only the first QA for each contract before `runtime.tokenize`. All 64 retained train records and all 16 mapper-validation records contain the **same `nda-1` hypothesis**: explicit identification of confidential information. The other 16 hypotheses never reach the functional KL objective or checkpoint selection.

Deduplicating context for a covariance fit is correct; discarding alternate questions is unnecessary. Represent training as a document plus its question pool, sample documents uniformly and then sample a question. Preserve the same document-level split and hash leakage checks. The native target scores 72.13% on held-out `nda-1` while neural scores 34.43%, so fixing question diversity alone is not enough.

### 3. The functional objective is evaluated on much shorter, cropped contexts

Training uses windows of at most 1,032 tokens and resets their positions to zero. Validation uses the ending window, also cropped. Full development prompts have median **2,223** tokens, range **519–6,471**; **958/1,037** exceed 1,032 tokens.

The crop is legitimate unlabeled distillation because both teacher and student see it. It is nevertheless a different task from preserving the native target's decision given the entire contract. Model selection should include a bounded set of full-length mapper-validation prompts and answer-position KL/native agreement, with no final-evaluation documents used for tuning.

### 4. Our ridge input is severely restricted, and the neural residual is not a replacement for a dense cross-head map

Our ridge accepts one depth-aligned source layer and the correspondingly indexed source head. It cannot learn a permutation or mixture across source heads. The neural variant adds a rank-16 nonlinear residual over three neighboring source layers; its main affine branch remains same-head. A rank-16 correction can be a bottleneck when the missing relation is high-rank cross-head mixing. Neither head indexing nor nearby depth guarantees semantic alignment between independently trained checkpoints.

### 5. Forced constant key RMS adds an untested constraint

Neural mapping divides every predicted content key by its RMS and rescales it with one learned scalar per layer/head. Qwen applies its learned **per-channel** `k_norm.weight` before RoPE. Those operations are not identical: RMS of `gamma ⊙ normalized_key` can vary with key direction, whereas our output magnitude is forced constant within each head. This is an architectural ablation to test, not a confirmed code bug. [Pinned Qwen implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.52.4/src/transformers/models/qwen3/modeling_qwen3.py).

### 6. Ridge regularization units require an explicit check

Our covariance solve is `Cxx/N + alpha I`. The stored calibration count is 63,462 and alpha is 0.01, equivalent to raw-sum regularization `lambda=634.62`. A raw-sum `lambda=0.01` corresponds to normalized alpha about `1.576e−7`. The paper prints a raw-sum equation. Without the author's actual implementation, this is a **mismatch to the printed convention**, not proof that its production code used those exact units or that our penalty caused the failure. Record the objective convention and sweep a small validated range instead of copying an ambiguous numeric constant.

### 7. The chosen pair has a limited long-context compute gap

Both checkpoint configurations have 28 layers, 16 query heads, eight KV heads and head dimension 128. Consequently their dense attention score/value operations have essentially the same quadratic sequence-length arithmetic, even though projection/MLP costs differ. KV payload sizes also match. As attention increasingly dominates long-context prefill, model parameter count alone therefore overstates the potential saving of this particular small reader. This is a geometry-based hypothesis about the measured small speedups, not an established kernel profile. Measure source/native prefill and mapper components before selecting a larger pair. [Source config](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json), [target config](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json).

## Comparison to the published method

The NVIDIA paper fits each target head from **all source heads** in source layers selected by predictive fit, factors out RoPE and calibrates on 500 FineWeb-Edu sequences of 1,024 tokens. Its nonlinear comparison is a separate, much wider MLP trained on cache MSE. It also examines attention-output fidelity because cache fit alone does not determine downstream preservation. Our nearest-three, rank-16 residual, mandatory key-RMS constraint, QA-window KL objective and 0.6B→1.7B pair are new experiment choices. The paper's principal Qwen results use larger pairs and different benchmark scoring. A failure here therefore cannot be described as a faithful negative reproduction. [Paper, sections 3–4 and appendices D–E](https://arxiv.org/html/2608.03893v1).

The primary recovery route is a method-matched cross-head ridge implementation with validation-selected source layers, explicit regularization convention, and no mandatory RMS projection. First prove the implementation on our small pair. Exact reproduction of the published large pair or the private vendor pair is a separate resource/access decision.

## What looks correct, within the tested scope

| Item audited | Finding | Remaining limitation |
|---|---|---|
| RoPE algebra | Our half-rotation sign and inverse correspond to Qwen's implementation; source is unrotated before mapping and target rotation is restored | Existing pretrained identity control is only 128 tokens; repeat at actual lengths and nonzero offsets |
| Cache orientation / ridge solve | Row-vector `X @ W`, covariance `XᵀX`, cross-moment `XᵀY`, bias and tensor head/sequence axes are consistent by inspection | No full per-layer oracle/readout report yet |
| Tokenization | Checkpoint tokenizer hashes match, and each evaluated prompt's source/target token IDs are compared | Chat-vs-plain calibration context distribution still differs |
| Causal token accounting | Teacher tail positions and student tail positions align; no clear off-by-one was found | The eight-token/one-token boundary mismatch remains |
| Frozen model gradients | Tiny tests show functional KL reaches mapper while base models remain frozen | Numerical correctness does not establish quality |
| Native import control | Pretrained target identity control passed; max logit delta 0.0332 and same top token at 128 tokens; zero cache changes logits strongly | Passing one short prompt does not establish all lengths |

These findings reduce the probability of a global RoPE-sign or cache-transpose bug. They do not rule out length-dependent import or attention-readout errors.

## Bounded diagnostic gate before another larger training job

Use a fixed, document-disjoint selection from development **for diagnosis only**, preserve all attempts and hash inputs/configuration. Do not tune a final acceptance result on this selection. No run is authorized by this document itself.

1. **Oracle import parity:** native full forward versus its own exact prefix K/V plus suffix 1/8/32, at short and representative contract lengths. Log all-vocabulary KL, maximum logit error, top-token agreement, positions and counts. Test an inverse/forward RoPE identity path separately from a raw-copy cache. A failing oracle blocks mapper interpretation.
2. **Current-mapper boundary probe:** native, saved ridge and saved neural at matched suffix lengths, with the same tokenizer and full prompt. Record full-vocabulary KL, option probability mass, choice accuracy/confusion, native agreement and complete cold timing. This changes the protocol prospectively; do not pass suffix 8/32 through the old exactly-one-tail validator.
3. **Attention readout isolation:** retain the target's own queries and separately combine native/mapped K and native/mapped V. Measure attention-output cosine and relative error per layer/head, attention entropy and key/value norm distributions. Native-K/mapped-V versus mapped-K/native-V isolates whether value readout or key addressing dominates. Then compare with queries from the fully transferred run to expose accumulated hidden-state drift. Respect GQA head replication and causal masks.
4. **Single-factor ablations:** on the same small calibration/mapper-validation split, compare same-head versus all-head input, depth-nearest versus predictive layer choice, and key RMS on/off. Keep the benchmark slice fixed and report model-selection use. Do not change every component and call the result an explained fix.
5. **Paper-compatible task scoring:** add a frozen small slice of a published benchmark using its continuation-likelihood or generation protocol, rather than treating ContractNLI first-letter logits as the original benchmark. Confirm native performance and label-order behavior first. The official ContractNLI test remains untouched.

After oracle and readout checks pass, scale the method-matched calibration and benchmark coverage. Keep a separate neural research branch; if pursued, train it on the same full-context handoff boundary used at inference, with varied questions, answer-position/rollout distillation and mapper-validation selection based on functional preservation. More optimizer steps alone are not the primary fix suggested by these results.

## Stop and success criteria

Stop if oracle import fails, malformed/nonfinite caches appear, or restricted option scores hide near-zero full-vocabulary option mass. Report the quality drop even when a latency case improves. Advance only when a held-out mechanism check preserves native behavior and a prospectively specified downstream evaluation shows acceptable quality at a useful end-to-end cold speedup. A functional mapper and a production serving adapter remain separate milestones.
