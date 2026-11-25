# Cross-model cache transfer: verification protocol

For the installable same-model cache handoff and its offline demo, see the [module quickstart](HANDOFF.md).

**Larger experiment completed:** [final results](reports/sova-kv-full-20261118-132341-7daf/RESULTS.md). The corrective job completed 400 mapper updates and all 1,037 development decisions, then stopped. Neural accuracy 50.82% versus native 59.11%; median paired contract cold speedup 0.953×. The candidate does not pass quality/speed acceptance. [Failure analysis](research/failure-analysis-v2.md) identifies question-coverage collapse and a training/inference tail mismatch for controlled follow-up. Earlier failure evidence is retained.

Research audit: 18 November 2026. **All 33 offline smoke tests passed. The corrected SageMaker pilot `sova-kv-pilot-20261118-123416-ea50` completed all 120 measured attempts and stopped. The thesis is not independently verified by SOVA yet.** Native cold median first-token latency was 157.74 ms and transfer cold 152.39 ms (median paired speedup 1.031x). Strict exact-letter accuracy was zero for every arm: native outputs included label descriptions (e.g. `A. Entail`), while transfer often began unrelated prose. The four-token output cap also truncates those descriptions. These results do not establish preserved answer quality or a useful speedup. See the [collected report](reports/sova-kv-pilot-20261118-123416-ea50/collected/evaluation/report.md) and raw traces before interpreting quality. Evaluation formatting/length must be fixed prospectively before a new quality experiment, not silently relaxed after this run.

The initial job ended before model evaluation because its Torch 2.1.0 image failed Transformers' SDPA minimum-version check. Its 204 billed seconds imply about $0.086 compute. The corrected job used an ECR-digest-pinned Torch 2.5.1 SageMaker image and checked CUDA SDPA before downloading checkpoints; its 320 billed seconds imply about $0.135 compute. Total estimated compute across both jobs: $0.221, excluding storage, transfer and tax. No serving endpoint or running worker remains from these jobs.

See [implementation and scoped smoke runbook](IMPLEMENTATION.md), [smoke evidence](reports/smoke-v1/test-results.json), and [GPU submission receipt](reports/sova-kv-pilot-20261118-123416-ea50/submission.json). The real adapter targets dense Qwen3; hybrid state transfer is explicitly unsupported. The pilot uses pinned Qwen3-0.6B → Qwen3-1.7B checkpoints and document-disjoint ContractNLI train/dev data.

Architecture: [interactive motion player](visuals/architecture.html), [animated SVG](visuals/cache-transfer-animated.svg), [static poster](visuals/cache-transfer-poster.png). These illustrate the proposed experiment, not measured performance.

Question: can a smaller reader plus a cache mapper reduce the time and cost of answering with a larger model while preserving enough answer quality for the intended workload?

## What we checked

| Finding | Meaning |
|---|---|
| OneTriangle documents a hosted mapped-prefill service and model-pair-specific mappers. | An ordinary chat request to an unrelated provider does not test this implementation. |
| The documented 27B SDK call returns a blocking result rather than a token stream. | Client wall time measures total response time. A TTFT claim additionally needs trustworthy first-token instrumentation or a streaming target path. |
| No OneTriangle credential/base URL was present in the process environment or four conventional SOVA/home environment files checked. | Access has not been configured in the checked locations. This was not an exhaustive keychain audit. |
| The official SDK wheel URL returned HTTP 403 when fetched for archive inspection. | We did not install or execute it. This does not by itself establish why access was denied. |
| Public independent implementations exist for conventional full-attention Qwen3 pairs. | We can test the general mechanism independently with cache-level control, without claiming to reproduce OneTriangle's private mapper. |
| Qwen3.5/3.8 are hybrid models with recurrent/convolution state as well as ordinary attention KV. | A full-attention-only mapper is insufficient evidence for the advertised hybrid pair. |

The earlier Fireworks account checks and ParseBench results do not verify cross-model cache transfer. They did not export, translate or import any cache tensors.

## Two distinct claims

1. **General mechanism:** translated model state can enable target decoding with less prompt recomputation. Verify with a controllable open-model runtime.
2. **OneTriangle product/headline:** their particular Qwen3.5-4B -> Qwen3.8-27B service delivers its stated speed, quality and cost. Verify that exact service or packaged mapper against a matched native target. Success on a different pair is not verification of this claim.

The current vendor Qwen 4B -> 27B timing table shows 3.9x at 1M as a projection, with measurements through 64K. Do not attribute every advertised peak to this pair or label projected results as measured. Their task-trained mapper and generic mapper quality tables are separate experiments; do not mix them into one comparison.

## Arms on the identical held-out inputs

| Arm | Execution | Purpose |
|---|---|---|
| Source only | Small model prefills and answers | Cheap quality baseline |
| Native target, cold prefix | Large model prefills and answers | Primary cold-request comparator |
| Cold transfer | Small-model prefill + mapping + movement/import + target tail processing + first decoded token | Primary end-to-end transfer test |
| Warm source handoff | Start with an already-existing source cache, then map/import/decode | Separate escalation scenario; never mix into cold-request latency |
| Native target, warm prefix | Ordinary native target prefix reuse, including the uncached tail | Strong reuse baseline and numerical correctness control |

Keep checkpoint hashes, tokenizer/token IDs, prompts, target precision, target hardware, output budget, decoding parameters, concurrency and kernel configuration fixed. Include the question and prompt-template overhead in every context-size check. Measure any provider/network differences rather than silently attributing them to cache transfer.

## Instrumentation required before believing a speedup

- Record source-prefill, map, cache movement/import, target tail recomputation, first-token and full-response durations separately.
- Measure true TTFT at the first generated token, not at HTTP headers, the first empty stream event, or the final response. Label visible-answer latency separately if reasoning tokens precede it.
- Synchronize GPU timing correctly; synchronize both devices for cross-device measurements. Use client monotonic wall time for complete request timing as a second view.
- Log imported token count, target recomputed token count, mapper/checkpoint hashes, fallback status/reason and cache-state layout. A fallback is an operational outcome, not a successful transferred-cache sample.
- Report transferred-only latency alongside all-attempt latency, fallback rate and errors. Do not discard slow failures to improve the headline.
- Time cold transfer from before the small model reads the document. Timing only the receiver or only the mapper answers a different question.
- Include peak allocated/reserved memory, cache bytes transferred, mapper size, GPU-seconds and billed capacity. An idle large-model GPU still costs money while the small model is prefilling.
- Freeze versions and compare against a production native serving configuration as well as any research implementation. Faster than an unoptimized HF forward pass does not establish the same improvement over a tuned serving engine.

For cold transfer:

```text
T_transfer_TTFT = source_prefill + map + move/import + target_tail + first_target_token
speedup = T_native_TTFT / T_transfer_TTFT
```

Mapper calibration/training is an offline cost. Report it separately and amortize it only with an explicit expected request volume. Include both models and any extra devices in the cost comparison.

## Correctness controls

1. Same-model identity transfer must reproduce the native warm-cache path within documented numerical tolerance. Keep any cold/warm BF16 near-tie differences visible.
2. Invalid mapper, checkpoint hash, position metadata or cache length must cause an explicit failure/fallback. Never silently accept an invalid cache as a successful transfer.
3. A controlled zero/shuffled-state ablation should change logits or quality. If it does not, inspect whether the target ignored the cache or recomputed the prompt. Run such ablations only in the isolated evaluation harness.
4. For the hybrid Qwen pair, instrument recurrent and convolution histories as well as attention KV. Preserve token order, positions and any bounded tail-recompute rule.

## Evaluation data and sampling

- **Quality pilot:** up to 200 deterministic, stratified LongBench-v2 examples that fit the native source and target windows, after tokenizing the full prompts. Freeze IDs and hashes before inference. If fewer qualify, report the actual count. The dataset's `train` label is its published evaluation container, not permission to fit on it.
- **Calibration:** use a separate corpus with no evaluation-document overlap. If task-specific mapping is evaluated, split by document before fitting, selecting hyperparameters, or teacher generation.
- **Timing:** 8K, 16K, 32K and 64K prompt lengths where actually supported; add 128K only after the runtime and memory check passes. At least 20 timed repetitions per length after untimed warmup, with model weights loaded but ordinary prefix reuse controlled. Rotate arm order and retain all raw traces.
- **Retrieval controls:** synthetic documents with exact facts at early/middle/late positions can diagnose information loss. These are diagnostic controls, not a substitute for real long-document QA.
- **No silent truncation:** exclude and count over-limit examples. Mark extensions/projections beyond supported native context separately; do not call a 1M extrapolation a successful 1M quality run.
- **Scoring:** use gold multiple-choice labels with strict answer parsing. Unsupported, malformed or missing answers count as wrong. Report source, native target and transfer scores separately.
- **Uncertainty:** paired accuracy difference with 95% document-cluster bootstrap intervals. Hash exact contexts to group reused documents. A small pilot may be inconclusive; do not claim equivalence because a difference is not statistically significant.

## Proposed SOVA acceptance criteria

These are our proposed product thresholds, not reported vendor results or a theorem:

- At least **2x median cold-request TTFT improvement** at both 32K and 64K, measured end to end, with p95 and fallback rates reported. If either length is unsupported, narrow the claim explicitly instead of extrapolating it.
- A lower 95% confidence bound on transfer-minus-native accuracy of at least **-3 percentage points** on the frozen quality set. Increase evaluation size if the pilot cannot resolve that margin.
- Positive measured GPU/capacity cost savings per completed answer at the agreed concurrency, without exceeding the deployment memory budget.
- Identity/import integrity gates pass and the reported transfer subset has no hidden full-prefill fallback.

Outcomes are **pass, fail or inconclusive**, separately for latency, quality and cost. A genuine speedup with unacceptable quality loss is still an important negative product result.

## Practical routes

**Direct product test:** obtain OneTriangle API access or its packaged mapper. Require a native comparator, actual mapped-pair identity, fallback metadata and first-token timing. Without those, a black-box test can assess answer quality and end-to-end response time, but cannot independently establish cache-level causality or true TTFT through the blocking SDK alone.

**Independent mechanism test:** start with a matched-tokenizer, matched-KV-geometry full-attention Qwen3 pair in an instrumented GPU runtime. Run identity/import controls, fit a pair-specific mapper using independent calibration data, and run the arms above. Public implementations are candidate starting points; their reported numbers and tests have not been rerun by us. Exact code revisions must be pinned and reviewed before installation/execution. Then tackle the advertised hybrid pair with a compatible full-state mapper; do not assume the first adapter transfers unchanged.

The local machine has 24 GiB unified memory and no CUDA GPU. The approved pilot runs on one SageMaker `ml.g5.2xlarge` in Ohio, with a 30-minute runtime limit and an observed public compute rate of $1.515/hour. The roughly $0.76 runtime-bound estimate excludes storage, transfer, billing overhead and tax. This is a small mechanism pilot, not a quote for the full protocol or the advertised production pair. No downloadable trained mapper for that exact pair has been verified.

## Compute policy

The user authorized the offline smoke and bounded GPU pilot on 18 November. The local ask-before-compute skill requires showing concrete commands, machine, duration, cost and output before future compute. This job writes artifacts to S3, ends on completion or at its runtime limit, and creates no serving endpoint. `cloud/collect.py` reads status and existing artifacts without starting compute. Failed jobs are not automatically retried.

## Evidence and sources

- [OneTriangle timings: measured versus projected](https://onetriangle.ai/)
- [Prefill SDK and streaming limitations](https://onetriangle.ai/prefill-sdk)
- [Task-trained mapper product and benchmark protocol](https://onetriangle.ai/mapper-api)
- [NVIDIA cross-model KV mapping paper](https://arxiv.org/abs/2608.03893)
- [Hybrid state-transfer paper](https://arxiv.org/abs/2611.08133)
- [Independent HF implementation](https://github.com/Susmith4710/kvtransfer)
- [Independent vLLM implementation](https://github.com/alesha-pro/vllm-cross-model-kv)
- [vLLM correctness gate source](https://raw.githubusercontent.com/alesha-pro/vllm-cross-model-kv/master/examples/transfer_gate.py)
- [LongBench-v2 official evaluation data](https://huggingface.co/datasets/zai-org/LongBench-v2)

The local `source-audit.json` records observations and source-pin attempts. GitHub commit API requests failed during this audit. The branch URLs above identify the sources reviewed.
