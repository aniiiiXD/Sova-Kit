# Completing the frozen reference evaluation after memory recovery

18 November 2026. Planning and source/artifact inspection only. This note does not claim that the repaired GPU training run or its benchmarks have succeeded.

## Fixed experiment and what can be changed prospectively

Keep the exact frozen **100 QuALITY development questions** and **1,000 HellaSwag validation records**, with source, native target and selected mapped target on identical inputs. Keep the existing option-logit decision metric and separately scored unconstrained generation for QuALITY. HellaSwag remains full-ending likelihood with raw `acc` and processed-ending **character-length** `acc_norm`; it is not option-letter generation. The selected mapper must come only from the official-training article partition and its training-domain validation split.

The next declared bounded pilot is **100 requested updates**, validation/checkpoint at initialization and update 100 only, a **1,300-second training budget**, **16-token QuALITY generation**, a **900-second QuALITY phase**, and remaining worker time for HellaSwag. Cloud job limit is 3,200 seconds; process/worker limit is 2,900 seconds. This is a new smaller training pilot, not a claim to have completed the older 400-update plan.

Raise **evaluation-only** prompt capacity to **9,000 tokens** before seeing any benchmark outputs; retain the 8,192-token training cap. Tokenizer-only preflight found three frozen dev questions over 8,192, with maximum **8,496**. The pinned base models support 40,960 positions. A 9,000 evaluation cap therefore permits all 100 selected questions without cropping or changing IDs. This is a declared execution-cap change, not posthoc removal of difficult examples. The same FP32 math-attention/query-block-256 path must remain active for every arm.

## Evidence that constrains the schedule

The failed `sova-kv-repair-20261118-162905-150e` run passed all 16 fresh native-cache controls and reached two actual updates before its mapper-checkpoint backward recomputation exhausted GPU memory.

| Saved observation | Meaning for the next budget |
|---|---|
| First training validation/checkpoint at elapsed **402.24 seconds** | This includes train-side tokenization, ridge fitting and validation initialization. It is not an isolated 30-question validation timer. Budget hundreds of seconds before the first optimizer update. |
| Update 1: **8.84 seconds**, 6,591-token prompt | Long contexts can dominate update cost. |
| Update 2: **2.41 seconds**, 2,879-token prompt | Update cost varies substantially; two observations cannot establish a reliable average. |
| First checkpoint reached at worker elapsed **565.80 seconds** | About 163.56 seconds preceded that train-side 402.24-second initialization, including loading and native controls. |
| OOM: **19.56 GiB** Torch-allocated, **2.17 GiB** reserved/unallocated | The core backward allocation problem must be fixed before expecting a meaningful training run. The failing 30 MiB allocation is not the total memory demand. |

Sources: [training log](../reports/sova-kv-repair-20261118-162905-150e/collected/training/training-log.jsonl), [progress](../reports/sova-kv-repair-20261118-162905-150e/live-progress.json), [native controls](../reports/sova-kv-repair-20261118-162905-150e/collected/native-control.json), and [failure](../reports/sova-kv-repair-20261118-162905-150e/collected/failure.txt).

The earlier 400-update window experiment ran in 175 seconds, but it used a different, much shorter training workload. Its rate is not a credible estimate for this full-document FP32 experiment.

## Existing reuse is already doing the important work

`run_quality_reference` computes one cold prefill per item/variant, scores its first-token decision, and continues generation from **that same cache**. There is no separate duplicate prefill for the decision and generation metrics to remove.

HellaSwag prepares one shared context cache per item/variant. Each of the four candidate endings receives a cloned prefix cache, and every ending token is teacher-forced and scored once. The context does not need four independent prefills. This preserves candidate isolation and the intended conditional likelihood.

Do not shorten inputs, skip variants, replace HellaSwag with letter generation, or silently shrink the frozen samples to fit a deadline. The chosen 16-token QuALITY cap is a prospective reduction from the proposed 32-token recovery cap; it changes only the declared generation budget, not the full-100 first-token decision task. A length-capped generation remains flagged and cannot be called a complete free-form answer.

## Small inference-memory repair, now implemented

Python evaluates an assignment's right-hand side before releasing the previous left-hand object. The previous QuALITY variant's `cache`, `logits`, and final decode `answer` object could therefore remain alive during the next cold prefill; `answer.past_key_values` is a second reference to that cache.

`reference_eval.py` now clears all three tensor owners **before the next attempt** and after serializing Python-only result fields. HellaSwag similarly clears its per-item/variant prefix cache and logits, while continuing to reuse the context within its four candidate branches. No numerical operation, score, token sequence or timed execution boundary was changed.

Two weak-reference regression tests assert that a preceding cache is already gone when the next `_prepare` begins, including a cache held through a decode output. Together with the existing real tiny-Qwen likelihood and document-boundary checks, **15 reference-evaluation tests passed** in about 1.9 seconds on CPU. The first test-double attempt retained arguments through `Mock`; replacing that double with a plain function removed the artificial retention. This was a test-fixture issue, not a second runtime failure.

At 8,192 tokens, one FP32 cache for the matched 28-layer, 8-KV-head, head-dimension-128 geometry is **1.75 GiB**; at 9,000, about **1.92 GiB**. Releasing the previous variant can therefore matter even in inference. These are cache-size calculations, not a measured post-fix GPU-memory result. Inference has no mapper backward graph; its memory demand should be distinguished from the training OOM.

## Budget forecast and honest completion gates

The ranges below are planning estimates, **not measured throughput** for the final repaired runtime:

| Phase | Rough worker time | Risk/control |
|---|---:|---|
| Loading and fresh native controls | 160–220 s | Prior evidence suggests this order of magnitude; setup can vary. |
| Initial fit/validation, up to 100 updates, final validation | Up to the declared 1,300 s | Validate at 0 and 100 only. A 25-update validation interval would add three expensive rounds and consume evaluation time. Record actual completed updates if the time budget ends early. |
| QuALITY 100 × 3 variants, up to 16 generated tokens | Roughly 650–900 s | Estimate from long-prefix native-control costs plus bounded decoding. Evaluate all 100; first-ten-item elapsed time gives an early forecast without changing the sample. |
| HellaSwag 1,000 × 3 × 4 endings | Roughly 400–700 s | All 4,000 ending pairs passed tokenizer preflight; longest complete candidate is only 137 tokens. Kernel-launch and model-weight movement costs may dominate these short calls. |
| Final artifacts/export reserve | At least 100 s | Preserve rows and the selected mapper before the hard stop. |

The high ends do **not all fit simultaneously** inside 2,900 seconds. The proposed 1,300-second training maximum can leave around 1,300 seconds for both reference evaluations, which is plausible near the middle of these ranges but not guaranteed. Neither short CPU tests nor the current logs justify promising that both full benchmarks will finish in one worker.

If time becomes insufficient, keep partial outputs explicitly partial and run an **evaluation-only continuation** with the exact selected mapper, dataset hashes, precision, prompts, generation budget and scoring contract. Do not retrain, select a different checkpoint using development scores, or count unexecuted items as successful measurements. If results are combined across continuations, use a separate provenance ledger with exactly one final result per `(benchmark ID, variant)`, original error rows retained, and a declared merge rule. Existing budget-not-run rows remain part of their original run record.

The current `fully_completed` flag requires every frozen item/variant to return `ok`; the new 9,000-token evaluation cap removes the three known context-limit outcomes, but timeouts and runtime errors can still make the flag false. A worker reaching its final print statement is not proof of completed benchmarks. Report planned, executed, error, over-limit and not-run counts for every arm, then the paired quality statistics.

## Avoid new execution changes until results exist

Do not add per-request CPU offload as the first inference fix. Moving multi-gigabyte weights/caches across PCIe can erase the latency benefit and changes the measured system. Static staging or batching candidate endings may be useful later, but would need its own numerical parity and timing checks. The immediate path is the already verified FP32/math/query-block execution, corrected mapper-training memory, cleaned inference lifetimes, fewer validation rounds, and preserved frozen evaluation.

No GPU job was launched, no pretrained checkpoint was loaded, and no benchmark was rescored while writing this plan. Only the explicitly assigned inference-lifetime change and tiny CPU tests were implemented separately.
