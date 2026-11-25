# Implementation audit for a falsifiable KV-transfer thesis

19 November 2026. Scope: the current dense Qwen3 implementation and completed reference experiment. This is a planning and source/artifact audit, not a new training run, model evaluation, or benchmark reproduction. Only stored JSON/JSONL, source files, and official model configuration/implementation pages were read. No model weights were downloaded and no tests or paid jobs were launched for this report.

## Thesis we can defend today

**We built a numerically checked cache-handoff experiment, but our learned translator does not preserve the target model's useful behavior. We have not yet shown that this pair offers a useful quality–latency tradeoff.**

The next thesis is a question, not a claim of success: can an interpretable, query-faithful cache translator retain native-target decisions while saving more prefill time than its mapping and native-query overhead? Diagnose the current failure before increasing model size, training steps, or architectural complexity.

## 1. Evidence hierarchy and version boundary

The current evidence is the parent job `sova-kv-repair-20261118-173137-4f04` and evaluation continuation `sova-kv-eval-resume-20261118-183814-e659`. The fixed selected mapper checksum is `e2f6390f176a0c4ef262edfaf18a133e5fe88a5a33a898c5415c757269d53c51`.

| Observation | What it establishes | What it does not establish |
|---|---|---|
| [134 tiny-fixture tests passed](../../reports/smoke-eval-resume-v3/test-results.json), zero failures/errors/skips; current `hf_backend.py`, `train_mapper.py`, `qa_training.py`, and `reference_eval.py` hashes match that snapshot | The recorded implementation tests and their tested contracts passed. | Semantic correctness of cross-model alignment or production serving performance. |
| [16/16 pretrained native-cache controls passed](../../reports/sova-kv-repair-20261118-173137-4f04/collected/native-control.json); maximum recorded probability total variation approximately `1.2596e-5` | Native target prefix-cache handoff agrees numerically with its full forward on these long-context controls in the accepted FP32/math-attention path. | An arbitrary learned cache is correct; all shapes/precisions/backends are interchangeable. |
| [48 successful updates](../../reports/sova-kv-repair-20261118-173137-4f04/collected/training/training-summary.json), no overflow attempts; frozen base models | Training executed, selected step48, and exported a mapper. | Completion of the requested100 updates, a400-update run, or convergence. |
| Fixed 30-question mapper-validation objective fell from `20.8862488` to `3.9950682` | The declared teacher-answer CE+KL objective improved on that fixed selection. | A measured improvement in downstream accuracy, native agreement, or speed. Initial downstream benchmark accuracy was not measured under the same final protocol. |
| QuALITY source47%, native59%, mapper26%; HellaSwag `acc_norm` source47.3%, native61.2%, mapper28.5% | This candidate has a large negative quality result on both declared reference protocols. | Impossibility of cross-model cache transfer, exact failure of a published method, or failure of a private vendor's mapper. |

The [final independent audit](../final-results-audit.md) verifies 100 complete QuALITY triplets and1,000 complete HellaSwag triplets:3,300 successful rows, no missing/error/over-limit outcomes. QuALITY mapped-minus-native is−33 percentage points, stored document-cluster95% interval `[−45,−19]`; HellaSwag normalized difference is−32.7 points, stored source-group interval approximately `[−36.82,−28.99]`.

The historical [architecture recovery plan](../architecture-recovery-plan.md), [failure analysis v2](../failure-analysis-v2.md), and [live training audit](../live-training-audit.md) remain useful **dated records**. Their old ContractNLI collapse, cropped-window/template-tail objective, and one-token inference boundary are not descriptions of the final QuALITY experiment. No historical files were rewritten for this audit.

## 2. Exact current computation

### Pair and state dimensions

Both recorded mapper geometries are `[8,128,28]`: eight KV heads,128 dimensions/head,28 layers. Official checkpoint configurations also specify16 query heads for both models. Source hidden size is1,024 and MLP width3,072; target hidden size2,048 and MLP width6,144. Both use standard full attention and standard RoPE in this implementation. [Official source configuration](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json), [official target configuration](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json).

At batch1 and prefix lengthN, each layer's K and V have shape `[1,8,N,128]`. Flattened per-token K or V width is1,024. Equal shapes do not imply that headh or layerl carries the same information across independently trained models.

### Ridge initialization is same-head, fixed-depth, short-context calibration

[`DenseRuntime.fit`](../../kvverify/hf_backend.py) calibrates on plain document text, tokenized without chat special tokens and limited to the first1,024 tokens. The current equal-depth pair gives `source_layers[l]=l`. For every kindK/V, target layerl and headh, it fits

```text
X,Y: [calibration_tokens,128]
Cxx = XᵀX/n − μx μxᵀ
Cxy = XᵀY/n − μx μyᵀ
W[l,h] = (Cxx + αI)⁻¹ Cxy      W: [128,128]
b[l,h] = μy − μx W
predicted_target_head = source_same_head @ W + b
```

Keys are de-RoPE'd before regression and target RoPE is restored after mapping. Values are not rotated. This is not an all-head linear translator: it cannot freely permute/mix source heads in its initial affine path. Normalized covariance uses `alpha=.01`; equivalently a raw-sum normal equation would add `n*alpha` to its diagonal. This is a convention to report and validate, not proof of overregularization.

**Counterevidence to blaming same-head structure:** KV-Lingo AppendixG reports that head-wise translators match head-mixing translators on released shape-preserving Qwen3 pairs, including0.6B/1.7B. Its small-to-large fitted maps have stronger same-index blocks, with mean normalized diagonal0.166–0.216 versus0.125 under the permutation null. Its independently pretrained pair behaves differently. Therefore all-head fitting is a discriminating control, not a presumed necessary repair for this released pair. [Primary paper, AppendixG](https://arxiv.org/html/2609.32610v1#A7).

The new trainer initializes from an exact saved **step-zero neural** artifact; it does not refit ridge in the final job. Its manifest/data-identity checks protect provenance. However, the distribution underlying that initial ridge remains bare-text early-document windows, whereas final mapping sees the exact chat prefix, beginning at its real token positions and extending through the full article.

### Neural map is cross-head only through a rank16 residual

[`NeuralMapper.from_ridge`, `apply`, `_map_layer_token_block`](../../kvverify/train_mapper.py) implement the same function in ordinary and checkpointed forms. For each K/V kind and target layer:

```text
same_head_affine:       8 independent 128→128 maps
three source layers:   fixed nearest depth, not predictive/learned selection
flatten+concatenate:   [N,3×8×128] = [N,3072]
residual:              LN → Linear(3072,16) → SiLU → Linear(16,1024)
reshape:               [N,8,128]
output:                same_head_affine + residual
```

K and V use separate residual weights/features; neither branch directly reads the other kind. The residual up-projection starts at zero. Affine weights and biases remain trainable. The mapper has **11,125,856 parameters**. Rank16 constrains each layer/kind residual to a16-dimensional output subspace before key normalization; the entire affine-plus-nonlinear mapper should not be described as a globally rank16 map.

For keys, predicted vectoru is transformed per token/head to

```text
k̂_content = u / RMS(u) × exp(clamp(g[l,h],−6,6))
k̂_positioned = target_RoPE(position, k̂_content)
```

The learned gain is one scalar per layer/head and is initialized from RMS of the target `k_norm.weight` vector. Therefore the zero-residual neural initializer is generally **not the identical function** as the ridge initializer: compulsory key normalization already changes it.

Pinned Qwen3 applies per-channel learned `k_norm.weight` to its RMS-normalized projected keys before RoPE. A direction-dependent vector gain and our scalar constant-RMS projection are different function classes. Exact native-key reconstruction may require token-dependent magnitudes that this projection removes; functional attention preservation might still be possible through other transformations. Which effect dominates is untested. [Pinned Transformers4.52.4 implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.52.4/src/transformers/models/qwen3/modeling_qwen3.py).

### Full-context question-conditioned training now matches the handoff boundary

[`split_document_prompt`](../../kvverify/qa_training.py) tokenizes the original full chat prompt once and splits at the document-end offset. A straddling token stays in the native suffix. The target freshly processes the original question, choices, instruction and assistant framing after the mapped article cache. The reference QuALITY path uses this same helper in [`reference_eval._prepare`](../../kvverify/reference_eval.py). No cropped gold question is relabeled as valid.

Training inputs:2,032 questions/120 official-train articles; internal validation inputs:491 questions/30 other official-train articles. Complete-prompt8,192-token eligibility excludes83 optimization questions; the optimization pool is1,949 questions/115 articles. The sampler cycles documents and then questions within each document. **Only48 distinct questions from48 distinct documents were consumed** before the time gate, fewer than one115-document cycle. This is limited coverage; it is not proof that more steps alone will solve the problem. Validation selects one fixed seeded eligible question per held-out article,30 total.

[`qa_training_loss`](../../kvverify/qa_training.py) freezes both models. The target generates up to8 actual answer tokens from its complete native prompt. The student consumes the native query suffix plus previous teacher tokens, and matches corresponding answer-position logits. Gold answer labels are not used. Current weights are `CE=1`, `KL=1`, `KV reconstruction=0`.

Memory recovery uses exact per-layer/token-block256 mapper checkpointing, absolute RoPE offsets, query-blocked math attention and selective activation offload. These are engineering fixes that allowed the run to execute; their success does not establish translator fidelity. Do not silently use this FP32 diagnostic implementation as a production-kernel speed benchmark.

## 3. The objective is related to the metric, but is not the metric

At a teacher-forced answer position, letq be the full-vocabulary teacher distribution, y its greedy token, andp the student's distribution. Ignoring constants independent ofp:

```text
L = CE(onehot(y),p) + KL(q||p)
  = −Σv [onehot(y)v + qv] log pv + constant

unconstrained optimum: p* = (onehot(y)+q)/2
```

Thus equal-weight hard-teacher CE plus soft KL intentionally sharpensq. It does not have exact distribution preservation as its unconstrained optimum except whenq is already a point mass. Its argmax remains the teacher argmax in that idealized unconstrained solution; this algebra alone does **not** explain our wrong choices. Under a shared restricted mapper, gradient allocation and approximation errors require measurement.

**Counterevidence to blaming mixed objectives generally:** KV-Lingo AppendixU finds mixed CE+KL broadly comparable to KL, usually within reported standard errors. Its CE uses reference continuations, whereas ours uses greedy target continuations; the protocols are not identical. This paper comparison further argues against treating the sharpening equation as a causal diagnosis. [Primary paper, AppendixU](https://arxiv.org/html/2609.32610v1#A21).

QuALITY's reported decision is argmax over the A/B/C/D first-position logits. Training averages full-vocabulary loss across up to8 generated positions, including formatting/explanatory continuation. Later positions can improve while first-position option relationships remain wrong. Validation currently averages the same composite loss and does not select on native-option agreement or decision accuracy.

Full-vocabulary KL can also weakly constrain conditional option probabilities when the teacher initially prefers formatting such as `Answer:`. For option setO, KL decomposes into an option-mass term, `q(O)*KL(q(.|O)||p(.|O))`, and a non-option term. Lowq(O) reduces the loss weight on the reported conditional decision. This is a mathematical possibility, not evidence that most current teachers have low option mass.

Stored final QuALITY rows give a more precise picture than the old collapse narrative:

| Arm | Median A–D probability mass | Full-vocabulary top token is an option | Predicted A/B/C/D counts |
|---|---:|---:|---|
| Source |0.94122|90/100|30/17/16/37|
| Native |0.99994|95/100|22/25/30/23|
| Mapper |0.63015|82/100|31/9/35/25|

The final mapper does **not** exhibit the older ContractNLI always-A collapse. It loses useful choices and some instruction-format probability. Strict one-letter generation is separately weak: source1/100, native5/100, mapper4/100 correct. Conditional decision scoring and free generation must remain separate.

HellaSwag evaluates full-ending likelihood, not an option-letter API. It is also out of domain for this QuALITY-trained mapper. Its negative result tests broad language-state preservation; it is not a clean test of a FineWeb-calibrated published method. A QuALITY-specific fix need not improve HellaSwag, and any generality claim must explicitly test both.

## 4. Compute and memory: this pair has a limited speed lever

LetL=28, Hq=16, Hkv=8, dh=128. For hidden widthd and gated MLP widthf, dense projection+MLP arithmetic per layer/token, counting multiply and add separately, is approximately

```text
Flinear = 2[d×(Hq dh + 2 Hkv dh) + (Hq dh)×d + 3 d f]
        = 2[6144d + 3df]
```

Across28 layers: source approximately0.880804GFLOP/token; target approximately2.818572GFLOP/token. These omit small normalization/activation costs and the final-token vocabulary projection. The linear ratio is3.2×; total model parameter ratio is not the prefill arithmetic ratio.

Both models have the **same** attention matrix dimensions. A dense full-square QKᵀ plus attention–V calculation costs approximately

```text
Fattention(N) = 4 L Hq dh N² = 229376 N² FLOPs
```

An ideal implementation skipping all causally masked pairs approaches half this leading term. Our diagnostic blocking retains all key positions for each query block; it bounds temporary memory, not the N² arithmetic order. Neither count is a hardware-latency prediction.

| PrefixN | Source linear work | Target linear work | Same full-square attention work, each | Ideal work ratio native/source, ignoring all mapping/suffix overhead |
|---:|---:|---:|---:|---:|
|8,192|7.216TFLOP|23.090TFLOP|15.393TFLOP|1.702×;2.065× under ideal triangular arithmetic|
|32,768|28.862TFLOP|92.359TFLOP|246.291TFLOP|1.231×;1.418× under ideal triangular arithmetic|

These are **optimistic arithmetic ratios under equal effective compute rates**, not measured speedup ceilings across arbitrary different kernels. Mapping, native suffix work, memory traffic and dispatch reduce attainable benefit; unequal kernels can move observed timing in either direction. As quadratic attention dominates, the ratio tends to1 for this equal-attention-geometry pair. No conclusion about a hybrid/linear-attention reader follows from this pair.

Cache bytes are also identical:

```text
Bcache(N,element_bytes) = 2×L×Hkv×dh×N×element_bytes
                       = 57344×N×element_bytes
```

At8,192 tokens, one full prefix cache is1.75GiB in FP32 or896MiB in16-bit. Source-plus-mapped caches temporarily require at least twice this payload before additional copies/activations. Any mapper must at least read its input and write its output:3.5GiB of aggregate cache payload at8,192/FP32 in this same-size case. Actual eager clones, concatenations and repeated layer use add traffic. Cross-device transfer does not compress the target cache simply because the reader has fewer parameters.

Current affine+residual matrix multiplication is approximately22.02MFLOP per prefix token, excluding normalization/RoPE/activation/copy work—small relative to model arithmetic, but this does not imply negligible runtime for many eager, small matrix operations. Target decode uses the ordinary target cache and ordinary target model: this implementation primarily targets **prefill/TTFT**, not faster per-token target decode.

The prospective end-to-end criterion is explicit:

```text
Tsource_prefill(N) + Tmap(N) + Ttarget_native_suffix(Q | N)
    + any actual transport/import/scheduling cost
< Ttarget_native_full(N+Q)
```

Time only functionally acceptable candidates, with identical correct precision/backend, synchronized component timings, batch/concurrency, cache policy and complete inputs. For repeated queries to an unchanged document, compare against **native target prefix-cache reuse**, not a cold target that needlessly recomputes the article. Include any mapper training cost separately. The merged cross-job result does not support a combined latency headline.

## 5. Claims ledger

| Claim | Classification | Reason |
|---|---|---|
| Native cache import is globally broken | Not supported as the explanation of these results |16 accepted long-context oracle cases pass; scope is finite and backend-specific. |
| More question diversity and a native query suffix were implemented | Confirmed |Full-context QA trainer, consumption logs, shared split helper. |
| Those repairs solved useful transfer | Disproven for this candidate/protocol |26% vs59% QuALITY;28.5% vs61.2% HellaSwag normalized. |
| The optimizer silently skipped updates | Contradicted by current saved traces |48 completed updates and zero overflow attempts in this run. |
| Lower validation CE+KL establishes preserved target quality | Disproven as an acceptance argument |Loss improved while the final candidate fails the downstream retention requirement; initial downstream comparison is unavailable. |
| Same-head ridge or fixed nearby layers are inadequate | Plausible hypothesis |No same-data all-head or predictive-layer control has been executed. |
| Constant key RMS is harmful | Plausible hypothesis with a confirmed function-class difference |No paired RMS ablation/readout diagnosis. |
| Source lacks enough information to recover target state | Unresolved |Source end-to-end quality is lower than native, but that does not prove a cache-information lower bound. Mapper quality below source likewise does not identify a unique cause. |
| Rank16 is too small, or48 updates are too few | Unresolved |No capacity/coverage study under a fixed objective; larger runs are not a diagnosis. |
| We reproduced the vendor's architecture or benchmark | Unsupported |Independent dense pair, own prompt/interface/splits; private architecture/IDs are not known. |
| Fixing quality will automatically deliver large speedups | Unsupported; geometry cautions against it |Same quadratic attention geometry and same KV payload; end-to-end break-even unproven. |

## 6. Minimal falsifiable experiments, in order

All rows below are **proposals only**. Use optimization articles for fitting and the existing30 internal mapper-validation articles for diagnosis/selection. The100 official-dev articles already viewed are development evidence now; do not tune on them and later call them an untouched acceptance set. Freeze a genuinely fresh acceptance protocol before making a positive generalization claim. Keep labels for evaluation unless an explicit supervised variant is separately declared.

| Priority / hypothesis | Smallest useful comparison | Fixed measurements | What would falsify the proposed explanation / decision gate |
|---|---|---|---|
|P0: cache error is in addressingK, readoutV, or both | On a predeclared small, length-stratified internal-validation slice, use native target queries at matched suffix positions; compare nativeK/nativeV, mappedK/nativeV, nativeK/mappedV, mappedK/mappedV, at each layer. Include step-zero and step48. Respect GQA expansion, real causal mask and untouched native suffixK/V. |Per-layer/head attention-output relative error/cosine, attention entropy/divergence, K/V norm distribution. Separately run accumulated whole-stack continuation; native-query probes isolate local error, not total drift. |IfK-only/V-only errors are small but whole-stack divergence is large, a simple global norm/readout story is insufficient; study compounding/teacher-query mismatch. If one branch dominates, target that branch before enlarging everything. |
|P1: mandatory RMS projection damages initialization | Compare exact saved ridge, its neural initialization with residual disabled and scalar key-RMS on, and the same weights with scalar key-RMS off. No training. |Same prompts/boundary, local readout fidelity, full first-position KL, option mass and native-option agreement. |No consistent RMS-on/off difference rejects RMS as a dominant initialization explanation. A better result only on one case is not a fix. The final native task metric is secondary to isolating this one factor. |
|P2: same-head alignment is the initial bottleneck | Same calibration tokens, fixed same source layer: same-head128→128 ridge versus all-head1024→128 per target head (equivalently1024→1024 per layer/kind). No residual/RMS change. |Held-out normalized cache error **and** native-query attention-output error; full first-position KL/native agreement on the same internal examples. Explicit normalized-alpha units and a tiny fixed validation grid. |If all-head improves cache MSE only but not readout/function, richer linear feature mixing is insufficient. If neither improves, do not claim head alignment explains the failure. All-head has more parameters/mapping cost; retain that accounting. |
|P3: nearest depth misses predictive source features | Only after a viable all-head baseline: select source layer per target layer/head by a predeclared held-out fit criterion versus fixed aligned depth. Keep feature width/calibration/objective matched; do not simultaneously add a wide MLP. |Fit-selection scores on internal validation, untouched final diagnostic slice, readout fidelity and function. |If selected layers are effectively depth-aligned and gains vanish out of selection data, layer search is not the useful lever. Search itself consumes validation freedom. |
|P4: composite rollout loss underweights the decision position | On the **same chosen mapping family and exact sampled QA sequence**, compare KL-only over first answer position, first-position conditional-option KL plus separately reported mass preservation, and current equal CE+full-vocabulary8-position KL. Give equal example/token budgets and report both. |Fixed internal-validation first-token full KL, option-conditional KL, option mass, native-choice agreement, gold accuracy, and per-answer-position CE/KL. Include untrained initializer. |If loss alignment improves only the selected metric while broad likelihood/generation deteriorates, it is a specialized decision adapter, not general cache communication. If first-position preservation still fails, the objective mismatch is not sufficient. |
|P5: inadequate coverage, rather than wrong inductive bias, is limiting | Only after a small mechanism/function win: freeze architecture/objective and extend a seeded coverage schedule through a full eligible-document cycle, then repeated questions/doc. Save several prespecified checkpoints. |Actual unique QA/doc coverage, fixed validation metrics, confidence intervals, training/validation divergence and per-step cost. |No improvement in functional validation despite decreasing training loss argues against simply buying more steps; stop or revise. A positive short learning curve is not yet an untouched benchmark result. |
|P6: useful cold-prefill break-even exists | For the first quality-acceptable candidate, compare source, native, transferred with identical correct runtime settings at a small fixed length grid; then separately compare document-cache reuse. |Quality/noninferiority margin frozen in advance; component plus end-to-end coldTTFT p50/p95, target decode latency, peak memory, bytes moved, concurrency, and amortization assumptions. |If complete transfer costs exceed native or source alone dominates its quality–cost frontier, this pair is not a product win. A larger or hybrid reader is a separate thesis requiring its own controls. |

Suggested first decision: run **P0 and P1 before neural retraining**, thenP2 if an interpretable feature-alignment deficit remains. These probes distinguish mechanisms; a single combined rewrite followed by a higher score would not explain what was wrong. Accept the possibility that the current pair or translator family is not useful.

## 7. What would make the work credible

A credible positive result needs a frozen pair/runtime, explicit mapper family/calibration protocol, numerical handoff control, independent selection/acceptance data, native-quality retention with uncertainty, and a complete end-to-end speed/memory comparison against strong cold and reuse baselines. A credible negative result needs the same care: preserve raw failures, expose the architecture differences, show where attention readout breaks, and report limits without inventing a universal impossibility claim.

The present artifact already supports an honest systems story: exact token-boundary construction, frozen-model differentiable training, bounded memory/overflow handling, independent benchmark continuation/provenance and a reproducible **negative** semantic result. It does not yet support “target quality at small-model cost,” “10× faster,” general communication between models, or an API someone should purchase on measured performance grounds.

## Source index

- Current code: [`hf_backend.py`](../../kvverify/hf_backend.py), [`train_mapper.py`](../../kvverify/train_mapper.py), [`qa_training.py`](../../kvverify/qa_training.py), [`reference_eval.py`](../../kvverify/reference_eval.py), [`blocked_attention.py`](../../kvverify/blocked_attention.py).
- [Frozen final134-test result](../../reports/smoke-eval-resume-v3/test-results.json); no tests rerun for this report.
- [Final training summary](../../reports/sova-kv-repair-20261118-173137-4f04/collected/training/training-summary.json), [protocol](../../reports/sova-kv-repair-20261118-173137-4f04/collected/training/qa-protocol.json), [step consumption](../../reports/sova-kv-repair-20261118-173137-4f04/collected/training/question-consumption.jsonl), [native oracle controls](../../reports/sova-kv-repair-20261118-173137-4f04/collected/native-control.json).
- Final merged [QuALITY rows](../../reports/sova-kv-eval-resume-20261118-183814-e659/collected/merged/quality/rows.jsonl), [HellaSwag rows](../../reports/sova-kv-eval-resume-20261118-183814-e659/collected/merged/hellaswag/rows.jsonl), and [independent score audit](../final-results-audit.md).
- Official Qwen configuration and pinned Transformers source URLs are cited beside the geometry/normalization claims, checked19November2026. Configuration URLs use `main`, so saved run manifests remain the experiment's identity authority.
