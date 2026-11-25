# Cross-model state transport: primary-body literature review

Retrieved 19 November 2026. Scope: five specified arXiv preprints, checked through Firecrawl Research body passages and the primary arXiv HTML. This is a literature study, not a model run or an independent reproduction. Publication/peer-review status beyond the retrieved arXiv metadata was not established. All performance figures below are the authors’ measurements under the stated conditions.

The useful hypothesis is **preserving a receiver’s computation across a model switch**, with the reusable state and cost boundary specified. A learned cache translator, functional distillation, or attention-aware correction alone would not establish new research novelty. Our opportunity must be stated as a falsifiable result beyond the methods below.

## 1. Closed-form Cross-Model mapping — Heo et al., August 2026

The production mapper concatenates **all source KV heads** from selected predictive layers, separately fitting each target head. The same-head regressions used to study linear structure are probes, not that production architecture. Keys are de-RoPE’d before fitting and receive target RoPE afterward. Calibration uses 500 FineWeb-Edu sequences, 1,024 tokens each, stride-four subsampling. Equation 4 prints centered, unnormalized normal equations with λ=0.01. Some benchmark results also select the layer-count hyperparameter; Appendix H identifies the selection benchmarks and separate holdouts. Hard Ministral directions remain important counterexamples. Timing uses synthetic inputs on eight H100s, BF16, includes necessary cross-GPU source-cache movement, excludes the receiver LM head, and explicitly omits shipping the mapped cache to a target process. [Method §§3.1–3.3](https://arxiv.org/html/2608.03893#S3), [latency Appendix G](https://arxiv.org/html/2608.03893#A7), [selection Appendix H](https://arxiv.org/html/2608.03893#A8).

**Our algebraic replication check:** if code uses `XᵀX/N + αI`, matching the printed `XᵀX + λI` requires **α=λ/N**, not α=λ. This identifies a convention to test; it does not establish a bug in unpublished author code.

## 2. CacheBridge — Qu et al., September 2026; updated November

Its main mapper deliberately narrows each target head’s inputs to one assigned source head per selected layer, trading unrestricted cross-head mixing for locality and smaller serving cost. Value calibration fixes key maps and fits query-dependent **attention-readout residuals**, then folds corrections into affine value maps. Calibration uses 24 QuALITY documents with four queries each; those documents and associated questions are excluded from QuALITY evaluation. The readout fit weights documents equally and constrains update energy; it is not ordinary cache MSE or arbitrary value rescaling. Serving TTFT starts after a switch with source cache and query available, includes delivery/reconstruction/installation and query processing, and ends at the first target token returned to the host. Main single-server serving uses separate TP4 engines on eight H800s. [Mapper §4](https://arxiv.org/html/2609.00891#S4), [calibration Appendix B](https://arxiv.org/html/2609.00891#A2), [serving boundary §3.1](https://arxiv.org/html/2609.00891#S3.SS1), [setup](https://arxiv.org/html/2609.00891#S5.SS4).

## 3. KV-Lingo — Castin et al., September 2026

Token-wise linear maps use closed-form initialization and frozen-target forward-KL distillation. Generic multi-turn training is reported disjoint from downstream benchmarks; one translator serves each model pair. **Head mixing is not a guaranteed fix:** head-wise translators match mixed-head ones on the released shape-preserving Qwen pairs in Appendix G. Appendix U’s CE uses dataset continuations, unlike our greedy-teacher CE; its mixed objective is comparable to KL. Appendix R’s post-key-norm distillation has modest long-context losses, not universal collapse, and is not our forced scalar-RMS operation. Reverse dense/MoE transfer remains a negative example. Timing starts with an already computed source cache, charging translation plus one decode; H100/vLLM and Apple/MLX are separate settings. [Method §3](https://arxiv.org/html/2609.32610#S3), [training Appendix C](https://arxiv.org/html/2609.32610#A3), [head alignment G](https://arxiv.org/html/2609.32610#A7), [objectives U](https://arxiv.org/html/2609.32610#A21), [normalization R](https://arxiv.org/html/2609.32610#A18), [timing §4.8](https://arxiv.org/html/2609.32610#S4.SS8), [negative direction N](https://arxiv.org/html/2609.32610#A14).

## 4. CacheLink — Qiu et al., October 2026

A fixed ridge backbone is augmented with coupled K/V residuals: separate stems feed a shared fused representation, then produce both corrections. Training differentiates continuation CE plus forward KL through the frozen receiver; the final objective has no cache-reconstruction term. Functional sensitivity to receiver queries does **not** mean the online token-wise adapter receives a future query as input. Four model pairs and seven fixed panels are evaluated; MMLU-Pro and MMLU-Pro-Plus share stems, so they are not independent question populations. Gains remain task-dependent. The latency experiment starts with the reusable sender cache already available. Appendix A specifies checkpoint/training differences across pairs; assuming identical training histories would be inaccurate. [Architecture §5](https://arxiv.org/html/2610.15509#S5), [evaluation §6.1](https://arxiv.org/html/2610.15509#S6.SS1), [latency §6.4](https://arxiv.org/html/2610.15509#S6.SS4), [protocol Appendix A](https://arxiv.org/html/2610.15509#A1).

## 5. Complete hybrid-state transfer — Cai et al., November 2026

Full-attention KV is insufficient for the studied hybrid models: recurrent matrices and ordered convolution histories also cross the boundary. The method combines structured maps, functional calibration, and a state-conditioned recurrent correction for one direction. Low-rank full-attention updates are merged offline; the state-conditioned recurrent correction cannot be absorbed into a constant affine map. A native-state substitution diagnostic isolates affected components. Reported 21.91× timing for 9B→27B is synthetic 16K **body latency**, source cache already present, batch one on two 80GB A800s; it excludes source prefill, vocabulary projection, and answer generation/scoring. Quality and timing use separate inputs. Authors explicitly identify joint quality/efficiency evaluation and other linear-attention architectures as unestablished. [Complete state §4](https://arxiv.org/html/2611.08133#S4), [diagnostics Appendix B](https://arxiv.org/html/2611.08133#A2), [timing Appendix C](https://arxiv.org/html/2611.08133#A3), [limitations §6](https://arxiv.org/html/2611.08133#S6).

## What is established prior art

| Candidate component | Prior-art anchor above | What our experiment could still establish |
|---|---|---|
| Predictive source-layer selection; all-head affine fit; RoPE factoring | Cross-Model | Reproducible behavior on our exact held-out pair and inputs |
| Compact head-local maps; query-readout value correction; transport/install integration | CacheBridge | Whether a specified accuracy loss buys measured resource savings |
| Generic-text, pair-specific linear translator; frozen-target continuation distillation | KV-Lingo | Cross-task robustness and failures with a frozen selection protocol |
| Coupled K/V residual plus CE/KL objective | CacheLink | Incremental benefit over a properly implemented linear baseline |
| REC/Conv plus KV transfer; foldable mapper updates | Hybrid-state paper | Support for an independently tested state layout, without incomplete-state claims |

These are method precedents, not proof that our implementation reproduces them. Vendor SDK capabilities, supported model pairs, service SLAs, and unpublished training recipes require separate primary vendor evidence. A paper’s model pair also does not establish that an SDK accepts that pair.

## Cost boundaries the SOVA thesis must keep separate

The following are proposed accounting definitions for our study, rather than extra claims about any paper:

1. **Existing-cache switch:** existing sender state → translate → move/install → receiver query → first token. Explicitly report cache age, cache placement, and whether the sender already performed useful work.
2. **New raw request:** tokenize → sender prefill → translate → move/install → receiver query → first token. This answers a different economic question and charges sender compute.
3. **Body microbenchmark:** only the named model-body or mapper kernels. State whether vocabulary projection, runtime scheduling, copies, network and synchronization are absent.
4. **Serving request:** admission/queueing → complete handoff → token returned over the API, at specified concurrency and memory residency. Include allocation and rejected/failed requests.

A warm state that must be retained also consumes memory between requests. Our proposed accounting should include the area under retained GPU/host bytes over time, not merely mapper FLOPs or one kernel’s speed. Report source-only, receiver-only, both-model residency, adapter storage, peak temporary memory, and any offload traffic separately. This is an experimental requirement we are imposing, not a result already measured here.

The conditional opportunity must be stated as **source cache available and target-prefix cache miss**. Add the receiver’s own native-prefix cache-hit path as a separate competing baseline. A faster transfer than full re-prefill does not establish an advantage over an already-cached native receiver, nor justify computing/retaining a sender cache solely for transfer. Freeze hit/miss rates or report a sensitivity curve before projecting fleet savings.

## Evidence required before the next thesis claim

Freeze the model pair, tokenizers, state layout, query boundary, calibration documents, independent selection split, and evaluation tasks before optimizing. Preserve all failures in the denominator. Record absolute source/native/transfer scores alongside retention; majority/chance controls remain essential for deciding whether an apparently high ratio is useful.

Our proposed diagnostic ladder is: native cache split/continuation parity → zero/shuffled/wrong-document state controls → faithful linear baseline → attention readouts and final distributions → complete answer/task quality → same-input timing and memory → held-out tasks and a deliberately hard pair. Intervene on keys and values separately before adding more capacity. Freeze gate thresholds prospectively; a posthoc numerical explanation can motivate a new control protocol without rewriting old outcomes.

**Primary research route:** reproduce a precise affine baseline first; then test whether a bounded functional correction improves held-out utility at a stated memory/latency cost. Keep a faithful-replication branch separate from new adapter experiments. Do not label independent K/V residuals a reproduction of coupled fusion, nearest-depth selection predictive-layer search, or single-pass scoring attention-free inference.

## Remaining verification limits

- The inspected body passages do not establish a universally successful model-pair relation or calibration-size guarantee.
- This review did not audit released implementation commits, download author checkpoints, reproduce task tables, or establish complete training-data decontamination independently.
- A large microbenchmark ratio does not establish a quality-preserving same-input service speedup for our workload.
- The proposed wrong-document/random-state control suite above is our requirement; this review does not assert every paper ran it.

## Retrieval and rerun inputs

Workflow: `firecrawl-research-papers`. Date: 2026-11-19. Target count: five. Primary IDs: `2608.03893`, `2609.00891`, `2609.32610`, `2610.15509`, `2611.08133`. Canonical metadata checked with `inspect_paper`; load-bearing method/timing/calibration claims checked with `read_paper` and primary HTML sections linked above. Full tool responses were cached in the research session rather than dumped into chat. Citation-graph expansion from Cross-Model and CacheLink returned no candidates; that empty expansion is not evidence that earlier literature does not exist. Rerun should pin paper versions and released code commits before exact replication.
