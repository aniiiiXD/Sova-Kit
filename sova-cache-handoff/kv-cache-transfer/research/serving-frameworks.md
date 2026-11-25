# Serving SOVA's cross-model cache mapper

Research retrieved **18 November 2026**. Official project documentation and repositories only. This is an implementation recommendation, not evidence that any proposed framework adapter has been built or benchmarked by SOVA. No dependencies, serving processes or cloud jobs were started for this report.

## Recommendation

**Primary serving target: vLLM with a custom V1 KV connector. Fallback: retain the working Hugging Face runtime behind a small streaming HTTP service.** Keep SGLang as a later engine alternative; add LMCache or Dynamo only when storage reuse or multi-worker orchestration is actually needed. TensorRT-LLM is another viable NVIDIA-focused integration, but implementing several engines at once would multiply the compatibility work before we establish a useful mapper.

Our custom component remains the **trained, model-pair-specific transformation**. Transport frameworks can move tensors and schedule cache reuse; that does not make a source model's representations valid for a different target. This distinction follows from the frameworks' documented cache/layout contracts below. It is an inference about their interfaces, not a claim that no research integration could ever implement conversion.

## What SOVA currently has

The scope recorded in [FULL_RUN.md](../FULL_RUN.md) is frozen Qwen3-0.6B and Qwen3-1.7B models, an approximately 11.13M-parameter mapper, target-cache reconstruction plus target-logit KL training, separate mapper-validation documents, and development evaluation on 1,037 QA records from 61 contracts. The initial larger job stopped before update 85 on FP16 gradient overflow. The corrected job named in that handoff is `sova-kv-full-20261118-132341-7daf`, submitted after 91 integrated tests; this research did not poll or reinterpret its live results.

The current adapter uses HF `DynamicCache`, full-attention dense Qwen3, fixed depth-neighbor groups and ordinary RoPE. It does not implement vLLM pages, request scheduling, HTTP latency, continuous batching, arbitrary model pairs, or hybrid recurrent/convolution state conversion. The mapper reads nearby source layers, so a future layer-streaming implementation must respect those cross-layer dependencies.

## Which layer each framework supplies

| Project | Documented capability | What we still must build |
|---|---|---|
| vLLM | Scheduler and worker KV connector hooks; paged-cache loading/saving | Source-cache extraction, learned mapping, target-layout packing and a pair-aware correctness contract |
| SGLang | Separate prefill/decode workers; Mooncake/NIXL transport | A mapper stage and changes at its internal memory-pool/transfer boundary |
| LMCache | Reuse and storage of computed cache chunks across memory/storage tiers | Conversion before publishing an object under a mapped-target identity |
| TensorRT-LLM | Custom cache connector scheduler/worker interfaces and disaggregated serving | A conversion-aware connector, target allocation/layout adapter and numerical parity tests |
| NVIDIA Dynamo | Routing, discovery and orchestration over serving backends and KV transport | Our converter inside a backend or a separately coordinated mapping worker |

Sources: [vLLM connector interface](https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py), [SGLang PD](https://docs.sglang.io/docs/advanced_features/pd_disaggregation), [LMCache integration](https://docs.lmcache.ai/developer_guide/integration.html), [TRT-LLM connector example](https://nvidia.github.io/TensorRT-LLM/examples/llm_kv_cache_connector.html), [Dynamo architecture](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving).

## vLLM: the first production-oriented integration

The V1 connector separates scheduling from worker execution. Scheduler-side hooks include `get_num_new_matched_tokens`, `update_state_after_alloc` and `build_connector_meta`; worker-side hooks include `register_kv_caches`, `start_load_kv`, `wait_for_layer_load`, `save_kv_layer` and completion reporting. Their responsibilities give us explicit places to declare an available mapped prefix, receive allocated destination pages, and finish writes before attention reads them. The interface is explicitly experimental. [Pinned v0.34.0 interface](https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py).

**Proposed first design, not implemented:** a source HF producer computes a prefix; our mapper creates target-format tensors; a custom target connector stages those tensors into target pages. Start with one request, one GPU and a synchronous import. Demonstrate target-native identity-cache parity before substituting learned caches. Then add cancellation, allocation pressure and concurrent requests. Keep the source runtime separate initially so engine porting does not also change the trained mapper's source representations.

Stock NIXL transport is not a learned adapter. Its compatibility handshake checks engine/connector versions, model geometry, attention backend and cache dtype. Supported heterogeneous tensor parallelism or block sizes describe layouts of compatible states; they do not establish semantic compatibility across checkpoints. Do not disable the compatibility check and call the resulting tensor copy cross-model support. [NIXL compatibility contract](https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/).

The target page layout must be read from the selected backend, not assumed to match HF's `[batch, heads, sequence, dim]` tuple. Layout permutation and block padding are representation changes; our learned mapper changes the represented model state. Keep the two transformations independently testable. The scheduler must count only the prefix actually staged successfully, retaining the uncached tail required by the chosen runtime. **The HF experiment's exactly-one-tail-token rule cannot simply be assumed for every paged connector**; measure and log target recomputation after adapting to its block boundaries.

vLLM documents separate prefill/decode tuning and tail-latency control, while warning that disaggregation alone is not a throughput improvement. Therefore compare against native vLLM with ordinary prefix caching and chunked prefill, not merely against our slower HF reference. [Disaggregated prefill guide](https://docs.vllm.ai/en/stable/features/disagg_prefill/).

## Version boundary: do not install the serving stack into training

The vLLM stable documentation retrieved here resolves to **v0.34.0**. Its pinned CUDA requirements specify **Torch 2.14.0**. SOVA's successful reference-training image uses **Torch 2.5.1 and Transformers 4.52.4**. These are different runtime stacks. Export mapper safetensors plus a strict manifest, then load them in a separate, digest-pinned serving container. Run identity, RoPE, dtype and generated-token parity tests again there. This is a compatibility warning, not an instruction to upgrade the active training job. [v0.34.0 CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/requirements/cuda.txt).

Freeze Python, Torch, CUDA/driver, engine tag/commit, attention backend, cache dtype/layout, connector and mapper versions together. A new engine version is a new tested compatibility point, not an automatic upgrade.

## SGLang: useful alternative, with deeper internal coupling

SGLang launches separate workers with `--disaggregation-mode prefill` and `decode`, using Mooncake or `--disaggregation-transfer-backend nixl`; the examples use the same model on both sides. It also documents a staging buffer for heterogeneous tensor parallelism and separate worker profiling. Those solve execution/transport concerns, not source-to-target state learning. [Official PD guide](https://docs.sglang.io/docs/advanced_features/pd_disaggregation).

Its internal transfer definitions expose buffer pointers, sizes, layer identities, cache dtype/layout and distinct state types including Mamba. They provide possible engineering seams, but are internal source interfaces rather than a promise that arbitrary mapped tensors can be submitted through the public generation API. A custom bridge must align memory-pool slots and completion signaling and validate every additional state family. [Transfer definitions](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/disaggregation/base/conn.py).

SGLang exports TTFT and end-to-end latency histograms. Those are useful backend observations; our gateway still needs its own request-start and first-content timestamps spanning the small model and mapper. [Production metrics](https://docs.sglang.io/docs/references/production_metrics).

## LMCache: add reuse after conversion works

Use LMCache when repeated context, cache persistence, eviction or CPU/storage tiers become part of the workload. Its multiprocess vLLM integration has a concrete version trap: at vLLM 0.20.0 and newer, explicitly selecting `lmcache.integration.vllm.lmcache_mp_connector` avoids unintentionally loading the older bundled implementation under the same connector name. Older versions cannot redirect that name in the same way. [Current quickstart](https://docs.lmcache.ai/getting_started/quickstart.html).

The project separately requires runtime ABI compatibility, intended connector loading and validation of the particular model/layout/backend. A matching pair of package version numbers alone is insufficient. [Compatibility guide](https://docs.lmcache.ai/getting_started/compatibility.html).

Our proposed mapped-cache key should include source checkpoint, target checkpoint, tokenizer and prompt-token hashes, mapper version, RoPE/position policy, cache dtype/layout, active adapters, tenant and `state_origin=mapped`. Native-target and approximate mapped states must not share an indistinguishable cache identity. LMCache supplies opaque identity tags and a model-name dimension that could carry parts of this scheme; the complete SOVA contract would be our responsibility. [Cache identity tags](https://docs.lmcache.ai/developer_guide/cache_identity.html).

## TensorRT-LLM: real hooks, not the immediate lowest-risk port

The documented custom connector has scheduler and worker classes, destination-cache registration and load/save hooks. Its persistence example uses `KvCacheConnectorConfig` to name connector classes and demonstrates repeated-prefix reuse; the example explicitly lacks chunked-prefill support and is not production-ready. It is a plausible future conversion seam, not a drop-in cross-model solution. [Connector example and source](https://nvidia.github.io/TensorRT-LLM/examples/llm_kv_cache_connector.html).

Disaggregated serving uses `cache_transceiver_config` on context and generation workers. The documented `NIXL` backend has to be selected explicitly; timeout and staging-buffer ownership affect safe lifetime management. A timeout does not automatically make memory still used by a transfer safe to free. [TRT-LLM disaggregation](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html).

There is documentation skew: LMCache's integration overview labels TRT-LLM as forthcoming, while TRT-LLM publishes an LMCache connector example using `lmcache.integration.tensorrt_llm.tensorrt_adapter`. Treat the exact release combination as unverified until tested. The current example uses the PyTorch backend; do not assume every historical TensorRT engine-building workflow still applies. [TRT-LLM LMCache example](https://nvidia.github.io/TensorRT-LLM/examples/llm_lmcache_connector.html), [LMCache overview](https://docs.lmcache.ai/developer_guide/integration.html).

## Dynamo: orchestration after the single-engine result

Dynamo coordinates prefill selection, backend-specific handoff metadata and decoding, with NIXL moving device data. Its local disaggregated presets use separate GPUs; its Kubernetes guide requires compatible model, dtype, block size and layout and checks shared-memory/RDMA setup. Those operational capabilities become useful after our custom backend produces valid translated states. They do not train a mapper. [Architecture](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving), [local presets](https://docs.dynamo.nvidia.com/dynamo/cli/disaggregated-serving/overview), [deployment prerequisites](https://docs.dynamo.nvidia.com/dynamo/kubernetes/disaggregated-serving/overview).

Do not turn the current single-GPU correctness experiment into a multi-GPU orchestration project simply to use a larger framework name. That recommendation is a project-scoping judgment, not a performance result about Dynamo.

## Memory and latency calculations for our actual pair

For ordinary FP16 dense-attention KV, per-model cache bytes are:

```text
2 × layers × KV-heads × head-dimension × cached-tokens × bytes-per-value
= 2 × 28 × 8 × 128 × N × 2
= 112 KiB per cached token
```

Using our checked model geometry, one cache is approximately **0.875 GiB at 8,192 tokens**, **1.75 GiB at 16,384**, and **3.5 GiB at 32,768**. Keeping complete source and mapped-target caches doubles those totals before weights, page padding, mapper intermediates, temporary copies, CUDA graphs and scheduler reserves. These are arithmetic estimates, not observed allocator peaks. Both models have the same KV geometry: the smaller reader does **not** intrinsically produce a smaller KV payload for this pair.

A cold-request win requires the entire source-prefill + mapper + packing + transfer/import + target-tail/first-token path to beat native target prefill/first-token selection. A fast network cannot compensate automatically for an expensive mapper or lost quality. On one GPU there is no inter-GPU network saving to claim. Layerwise conversion and direct writes into target pages are later optimizations; first establish whole-prefix correctness.

## How to measure the serving result

vLLM exposes TTFT, prefill/decode durations, inter-token latency and end-to-end latency metrics. Use these with gateway traces; backend TTFT alone omits work done before a request reaches the target engine. [Metrics definitions](https://docs.vllm.ai/en/stable/design/metrics/).

Proposed per-request record:

```text
gateway arrival → source queue → source prefill → mapping
→ layout/transport/import → target queue → target first token
→ first non-empty streamed content → final token
```

Record CUDA event timings for GPU components and monotonic wall time for the end-to-end path. Record bytes moved, actual imported/recomputed tokens, allocated and reserved device memory, block occupancy, errors, cancellation and explicit fallback. Distinguish a token-ID event, buffered text emission and a token visible to the client. A text stream can buffer until a word is printable; Hugging Face's streamer implementation makes that distinction concrete. [Pinned streamer source](https://raw.githubusercontent.com/huggingface/transformers/v4.52.4/src/transformers/generation/streamers.py).

Run native cold, native warm-prefix, source-only, cold mapped handoff and warm-source handoff on identical inputs. Add concurrent arrivals only after the single-request path passes. Report p50/p95/p99, quality versus native target, goodput under a declared latency limit, and billed serving capacity under the same workload. Warm handoff must not be mixed into a cold-input headline.

## Concrete sequence from today's worker

1. Finish the current mapper/evaluation run. Preserve its raw errors and quality results. A reduction in training objective is not evidence of downstream task preservation.
2. Freeze one model pair and selected mapper. Establish held-out quality, native-cache identity parity and useful length-dependent cold latency before expanding pairs.
3. Use the existing HF runtime as the **fallback demonstration service**: a bounded request queue, request IDs, cancellation, native/mapped modes and a streaming HTTP response. Its direct cache API keeps the numerical reference inspectable. This API is proposed, not currently implemented. [Pinned HF cache interface](https://raw.githubusercontent.com/huggingface/transformers/v4.52.4/docs/source/en/kv_cache.md).
4. Build the custom vLLM connector in a separate pinned container, first importing a native target cache and then mapped caches. Validate layout, positions, partial final pages and recomputation explicitly.
5. Compare native vLLM against the complete converted path through the same gateway. Quantify source+target residency and concurrency rather than treating the experiment's both-models-resident native arm as a native-only serving memory measurement.
6. Add LMCache for demonstrated repeated-prefix demand; add NIXL/Dynamo only after separate devices or pools are warranted by measurements. Expand to hybrid models only with explicit recurrent/convolution-state conversion and new quality controls.

An honest portfolio description today is: **“Built a reproducible cross-model cache-transfer research harness, trained a frozen-model cache mapper, and instrumented quality, latency, memory and numerical failures on managed GPUs; production connector integration is the next engineering stage.”** Claiming a production API, a universal model adapter, a quality-preserving speedup or the vendor's advertised hybrid pair would exceed the evidence available to this report.
