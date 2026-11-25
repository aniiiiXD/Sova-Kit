# SOVA cross-model cache transfer: method audit and experiment plan

Research date: 18 November 2026. This report audits the code and saved evidence, then proposes experiments. It does not announce results from the currently running corrected job. No model code, test gate, or cloud job was changed during this research task.

## Assessment

The research direction is sound, but our present implementation is an economical experimental variant, not a faithful reproduction of the original paper or any vendor's private mapper. The biggest methodological gaps are the weak same-head ridge baseline, unselected source layers, restricted functional-training coverage, and short continuation supervision. A larger parameter count or longer training run alone would not identify which gap matters.

The useful project is a reproducible study of **when a translated cache preserves a receiver's behavior and when the complete request becomes cheaper**. A credible negative finding with a precise failure diagnosis is valuable ML systems work. An unqualified claim of production-ready cross-model memory is not supported by the present evidence.

## Primary research that should guide the work

| Source | Verified method detail | Consequence for SOVA |
|---|---|---|
| [Heo et al., Cross-Model KV Cache Transfer, v1, 4 Aug 2026](https://arxiv.org/html/2608.03893v1) | Production ridge predicts each target head using all source heads from predictively selected source layers. It factors out RoPE and uses 500 FineWeb-Edu sequences of 1,024 tokens, subsampled at stride four. The reported MLP is much wider than our residual. The paper measures attention-output similarity as well as reconstruction. | Our original same-head, depth-aligned ridge is weaker; calling its failure a reproduction failure would be incorrect. Establish a better linear reference and receiver-behavior diagnostics before assigning blame to linear mapping. |
| [CacheBridge, v2, 3 Nov 2026](https://arxiv.org/html/2609.00891v2) | This revision combines compact head support, attention-readout calibration, and a serving system with source-side mapping, pipelining and direct installation. It explicitly studies handoff costs beyond mapper matrix multiplication. | A narrow mapper can be a deliberate systems tradeoff, but needs attention-aware calibration and an end-to-end comparison. Our HF adapter has no paged serving integration or transfer pipeline. |
| [CacheLink, v1, 11 Oct 2026](https://arxiv.org/html/2610.15509v1) | Uses a fixed ridge base and zero-initialized, coupled K/V residual. Gradients pass through a frozen receiver; its objective combines continuation cross-entropy and forward KL without a cache-MSE term. | Receiver-level distillation already has prior art. Our independent K/V branches, trainable affine base, mandatory key normalization, and MSE-plus-KL objective are distinct choices that require ablation, not novelty claims. |
| [RaReCache, v1, 8 Oct 2026](https://arxiv.org/html/2610.11358v1) | Studies selective target recomputation when sender representations do not preserve enough information, using rank disagreement to select positions. | Treat partial recomputation as a separate operating point if pure transfer reaches a quality ceiling. Its recompute fraction and compute must remain explicit; it is not zero-prefill transfer. |
| [Hybrid State Transfer, v1, 5 Nov 2026](https://arxiv.org/html/2611.08133v1) | Hybrid models retain full-attention KV, recurrent memory and convolution history. This paper calibrates state adapters through a frozen target and distinguishes warm handoff-body timing from source prefill and generation. | The present dense-Qwen3 code cannot establish a result for the advertised hybrid production pair. A hybrid implementation is a separate project with additional state and boundary correctness tests. |

These are authors' preprints, not results independently reproduced here. Use the versioned URLs above: CacheBridge's current v2 has a broader systems scope than its original abstract, and summaries of the old revision are not interchangeable with v2.

### Repositories and framework references

- [Susmith4710/kvtransfer](https://github.com/Susmith4710/kvtransfer) is explicitly an independent HF implementation. Its author documents layer selection, receiver diagnostics and multiple benchmarks. It is a reference implementation to inspect and pin, not the original NVIDIA authors' release and not evidence that our code has the same behavior.
- [alesha-pro/vllm-cross-model-kv](https://github.com/alesha-pro/vllm-cross-model-kv) is explicitly an independent vLLM implementation with the same 0.6B → 1.7B pair. It reports a selected-layer k=8, 500-sequence mapper, cold/warm distinctions, artifact identities and visible recompute misses. This is a useful external implementation to compare against under our own harness; we have not rerun its reported results.
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the author-maintained evaluation framework to use for standard task definitions and continuation likelihood scoring. A custom runtime adapter must preserve prompts, tokenization and denominators.
- [vLLM's disaggregated-prefill documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/) is an engine-integration reference. Ordinary same-model cache transport does not itself perform semantic translation between independently trained models; the mapper and correctness gates remain our responsibility.
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) provides trace tooling for identifying actual kernels, allocations and synchronization. A profiler trace should justify optimization work before introducing a custom kernel.

No original-author mapper-training repository for the papers above was located in this bounded audit. That is an access finding, not proof that none exists. Do not label an unrelated repository as official. No private vendor training implementation was available for inspection.

## What our code actually implements

The inspected files are [`hf_backend.py`](../kvverify/hf_backend.py), [`train_mapper.py`](../kvverify/train_mapper.py), [`full_eval.py`](../kvverify/full_eval.py), [`full_worker.py`](../cloud/full_worker.py), and the [prepared-data manifest](../datasets/prepared-v1/manifest.json).

| Component | Actual implementation | Important qualification |
|---|---|---|
| Model pair | Frozen dense Qwen3-0.6B and Qwen3-1.7B; matching tokenizer and KV geometry | Neither foundation model is being trained. This does not evaluate a 27B hybrid model or a 100B foundation model. |
| Ridge initialization | One depth-aligned source layer; separate affine map per corresponding head and per K/V | It cannot directly combine other source heads or distant source layers. |
| Neural mapper | Trainable affine base plus rank-16 nonlinear residual reading all heads of three nearest source layers | Nearby layers are a fixed depth heuristic, not selected for predictive value. The residual's bottleneck is a capacity constraint. |
| Key treatment | Remove source RoPE; map; normalize every predicted key's RMS with a learned gain; restore target RoPE | Step zero already changes the ridge function through normalization. A zero-initialized residual does not make this an exact ridge-start ablation. |
| K/V interaction | Independent K and V feature/residual branches | Receiver loss couples their eventual effect, but the mapper has no explicit joint K/V latent feature fusion. |
| Objective | Normalized KV MSE plus full-vocabulary native-target forward KL at eight causal tail positions | No continuation CE is used. Equal numerical coefficients do not imply equal gradient influence. |
| Optimization plan | 400 actual finite optimizer updates; 1,024-token prefixes; eight-token tails; adaptive CUDA loss scaling | This is a bounded candidate run, not an established convergence prescription. Overflow attempts are separate from completed updates. |
| Optimization corpus | 64 distinct contracts; source file contains 1,088 question records | The trainer deduplicates before tokenization, retaining one formatted question per contract. It does not functionally train on all 1,088 distinct questions. |
| Mapper selection | 16 separate training-domain contracts; token-window KL/MSE selection | Validation also keeps one question per document. It is not downstream quality validation across all 272 supplied questions. |
| Development evaluation | All 1,037 questions from 61 official development contracts; declared token cap | Question count is not independent-document count. Over-limit examples remain visible and count as failures. |
| Scoring | Restricted first-token choice scores, option probability mass, and separately sampled free generation | Good conditional A/B/C ranking does not guarantee a usable unconstrained answer. Exact response quality remains a separate measurement. |
| Runtime | Batch-one HF, synchronized timing, both models resident, one target tail token recomputed on transfer | No real API queue, paged-cache connector, tensor-parallel traffic or native-only memory/cost isolation is measured. |

For the planned 400 updates, at most 409,600 prefix-token presentations and 3,200 tail prediction positions enter this loop. These are presentations with possible repeats, not distinct training examples or independent observations. Uniform document draws would average 6.25 sampled windows per training document; actual coverage should be read from the saved window ledger.

### Saved evidence, not future claims

- The completed pilot [`sova-kv-pilot-20261118-123416-ea50`](../reports/sova-kv-pilot-20261118-123416-ea50/collected/evaluation/summary.json) has 12 unique questions, 120 repeated arm attempts, and approximately 158 ms native versus 152 ms transfer median cold first-token latency. Its strict output-format score was zero for every arm, and transferred outputs often began unrelated prose. This establishes neither preserved quality nor useful business savings.
- The first larger training attempt recorded [84 completed updates before an FP16 gradient failure at attempted update 85](../reports/sova-kv-full-20261118-131018-22a9/collected/training/training-failure.json). Its finite forward loss did not guarantee finite scaled gradients. The code now has bounded adaptive backoff and separate overflow/update accounting. That repair still needs evidence from a successful real GPU execution.
- The corrected retry's final model results are deliberately not asserted here. Read its collected receipt, training summary and evaluation traces before updating any claim.

## Highest-priority design questions

### 1. Are we comparing against a sufficiently capable linear baseline?

Our same-head initialization is a useful low-cost control but cannot identify whether a low-rank nonlinear correction is necessary. The first structural comparison should hold data, token positions, scoring and runtime fixed and change only the source feature support: same-head k1 → all-head k1 → selected-layer support. This is our proposed experiment, informed by the source-feature distinction above.

For this particular 28-layer, 8-head, 128-dimension target, an all-head k1 affine map has approximately 58.7 million non-bias weights, versus approximately 7.34 million in the same-head map. At k8 it would have approximately 469.8 million weights. These are calculations from our geometry, not measured memory allocations. One FP64 covariance at feature width 8,192 needs 512 MiB; keeping K and V covariances for all 28 layers would consume about 28 GiB before cross-statistics and the models. A faithful larger-feature fit therefore needs streaming or per-layer processing, not an unreviewed expansion of the current accumulator.

### 2. Is mandatory key normalization helping or changing the wrong thing?

Compare the original ridge, ridge plus key-RMS normalization alone, and the neural model with normalization disabled. Record actual native/mapped key RMS distributions, attention entropy and receiver output similarity. This separates a scale correction from a learned representation correction. Do not attribute the combined gain or regression to the MLP until this control exists.

### 3. Are we teaching continuation compatibility or only local tensor agreement?

Inspect validation KL, KV reconstruction, attention-output cosine and downstream quality together. Sweep the KV coefficient before adding model capacity. The present reconstruction denominator averages across tens of millions of cache elements, whereas only eight tail positions supply functional supervision; log component gradient norms as well as scalar losses. A decreasing total loss is insufficient evidence of a useful adapter.

A prospective comparison should include KL-only and continuation-CE-plus-KL objectives. Continuation CE can use the observed next text tokens in label-free windows; it does not require inventing answers for cropped contracts. For full QA-specific learning, keep the full evidentiary context or explicitly define a separate evidence-selection task.

### 4. Are document deduplication and question sampling being conflated?

Keep unique documents for KV calibration, but maintain a separate index of every question belonging to each training document. Sample documents first, then a question/window within the selected document for functional training. This preserves document-disjoint splits without discarding most task phrasing. Before changing it, retain the current run as its own versioned baseline.

### 5. Does the target get a genuinely useful new-query boundary?

Our transfer arm maps nearly the entire formatted prompt and recomputes one final token. A product that reuses a contract across questions can instead map the shared document prefix and let the target natively read the new question/instructions. Compare these as separate protocols: the second performs more native target computation and must report its tail length and cost. It could recover target query understanding while still avoiding repeated document prefill.

### 6. Does a speedup survive the complete request and a competent baseline?

Use two distinct ledgers:

`cold request = source prefill + map + move/install + target query + first-token selection`

`warm escalation = map + move/install + target query + first-token selection`

Use the same quality criterion for both. Record source-only, target-only, target warm-prefix, and transfer paths. Isolate a target-only memory measurement rather than charging both resident models to native serving. Ultimately include queue delay, batch/concurrency, cold model loading, cache hit rate and billing. A mapper-only ratio cannot be advertised as end-to-end API acceleration.

## Prospective ablations and stop rules

The following are proposals, not submitted jobs or spending authorization. Reuse frozen data and model revisions. The existing cumulative cloud guard remains authoritative; do not launch a sweep because a table exists. At the observed $1.515/hour rate, a 30-minute worker bound corresponds to roughly $0.76 compute, and two hours to $3.03, excluding storage, transfer, setup/billing overhead and tax. Rate evidence: the [AWS public Ohio training offer](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-2/index.json) saved with the submission receipt.

| Priority | Experiment | What is held fixed / changed | Evidence needed to proceed | Suggested incremental resource limit |
|---|---|---|---|---|
| P0 | Finish the corrected current run and audit it | No method change; 400 real updates, explicit overflow ledger | Successful controls, honest update count, selected checkpoint identity, complete dev/error denominator | Existing job limit only; no additional job |
| P0 | Read saved predictions, loss curves and window coverage | Artifact analysis only | Accuracy versus majority baseline; per-class results; option mass; free-generation failure categories; train/validation divergence | No GPU job |
| P1 | Same-head versus full-cross-head k1 ridge | Identical calibration documents, token samples and held-out diagnostic contexts | Held-out KV error plus native-logit KL/attention-output similarity, not training fit alone | One bounded small-pair job, at most 30 minutes |
| P1 | Key normalization-only intervention | Current ridge and current data; only normalize/not normalize keys | Determines whether normalization alone explains the step-zero change | Bundle with the preceding job |
| P1 | Predictive layer selection versus nearest-depth support | Fixed feature width, token budget and downstream validation split | Selected layers improve held-out receiver behavior enough to justify their fit cost | Up to 30 additional worker minutes, only if the preceding gate passes |
| P2 | Current MSE+KL versus KL-only or CE+KL | Same initialization, data order, number of successful updates, eight-tail protocol | Functional loss gain translates to downstream and free-generation improvement | Two runs within an explicitly approved aggregate cap; do not default to a full grid |
| P2 | Document-only dedup versus document-then-question sampling | Same 64 document IDs and update budget | Broader hypothesis coverage without losing mapper-validation quality | Replace one candidate above rather than multiplying run count |
| P2 | Map-document/read-question versus one-tail-token protocol | Same contract/query pair; explicit native question length | Better quality at a measured additional tail-prefill cost | Short paired validation before any benchmark-wide rerun |
| P3 | Data diversity and continuation horizon | Contract-only versus pinned general-text mix; tail 8 versus 32/64 | Held-out natural-text continuation and a second task improve; no regression hidden by averaging | One factor at a time; price after profiling |
| P3 | Profile and optimize the winning mapper | Exact selected weights and function; compare outputs before/after | Wall-time bottleneck proven by trace; numerical parity; cold and warm improvements | Kernel work only after a quality-passing model exists |
| Later | Selective target recomputation | Explicit replay fraction, separate from pure transfer | Measured quality/latency Pareto improvement rather than a concealed fallback | New scoped experiment and budget |

Do not choose these settings on official test data. Current official dev is already development evidence. Lock the model, scoring, maximum context, generation contract and acceptance thresholds before a final test evaluation. Confidence intervals should resample documents, because 17 hypotheses about one contract share evidence and are not independent documents.

## Minimal credible benchmark package

1. **Mechanism correctness:** same-model identity import, malformed/zero-cache negative control, source/target token identity, imported-position count, RoPE/boundary tests, checksum rejection, no silent recompute, and a real low-precision gradient smoke.
2. **A strong method reference:** retain our cheap ridge, add a genuinely selected/all-head linear reference, and compare the nonlinear candidate on identical inputs. Do not equate higher capacity with a stronger result before measuring it.
3. **Receiver utility:** task accuracy, majority and source-only baselines, per-class performance, unconditional answer-option mass, exact free-generation success, continuation KL/perplexity and attention-output similarity. A single conditional decision score is not the whole product.
4. **Statistical and domain scope:** document-level splits, document-cluster uncertainty, length buckets, one held-out task, and an untouched final test. ContractNLI remains one legal inference task with recurring hypotheses; it cannot establish general finance, coding or reasoning capability. [Official dataset repository](https://github.com/stanfordnlp/contract-nli).
5. **Systems scope:** cold/warm source boundaries, actual transferred bytes and import overhead, peak and resident memory, median/tail latency, standalone native memory, traces, worker teardown and billable seconds. Synthetic long contexts measure timing only until task-bearing long contexts are evaluated.

For calibration diversity, use a small pinned sample from the [official FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), with its provenance/license recorded and no benchmark documents mixed into calibration. Do not download or package the full corpus merely to obtain a few hundred sequences.

## What makes this a credible ML systems portfolio project

The convincing story is the experimental discipline and systems understanding:

- Explain the computation boundary precisely: what is reused, which positions are reconstructed, which positions are recomputed, and which model produces logits.
- Show the failures: the first weak mapper, strict output-contract mismatch, and the FP16 gradient-scaling diagnosis. Preserve their reports rather than rewriting them into successful results.
- Produce an ablation table that explains a behavior change, not only a new architecture diagram. Identify whether layer choice, feature width, objective, normalization or query boundary caused it.
- Provide one reproducible command for preparation and one for the bounded job, pinned code/models/data, machine-readable traces, a cost ledger and automatic termination. Keep credentials and customer documents out of the repository.
- If the method passes quality, port one mapper to a real engine connector, demonstrate numerical/cache-boundary correctness, and show an actual request benchmark under concurrency. If it fails, publish the failure boundary and evidence instead of claiming a serving product.

A truthful current project description is: **“Built a bounded research harness for cross-model KV-cache handoff between frozen Qwen models, including nonlinear mapper training, receiver-distribution distillation, document-disjoint evaluation, low-precision overflow recovery, and SageMaker/S3 artifact collection.”**

Do not yet write “production-ready,” “quality preserved,” “100B post-trained,” “10× faster API,” or “reproduced the vendor.” Add quantitative outcome claims only after they are directly supported by the final saved reports and an explicitly stated measurement boundary.

## Scope of this audit

Only primary papers, their authors' text, official framework/dataset sources, and independent repository authors' own implementation descriptions were used for methodological claims. Third-party paper summaries were discovery aids, not evidence. External repository performance numbers were not imported into SOVA results. Proposed experiments and portfolio recommendations are our inferences, not published findings.
