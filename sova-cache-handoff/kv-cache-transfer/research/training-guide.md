# What the training pipeline actually does

This is a guide to the submitted experiment's code and configuration, not a claim that every stage has completed. The authoritative stage is the latest SageMaker status plus S3 `progress.json`; collected logs distinguish the most recent checkpoint from the selected best checkpoint.

Current corrective job: `sova-kv-full-20261118-132341-7daf`.

## Inputs and trainable parameters

| Component | Configuration |
|---|---|
| Source model | Pinned public Qwen3-0.6B, frozen |
| Target model | Pinned public Qwen3-1.7B, frozen |
| Cache geometry | Both: 28 layers, 8 KV heads, 128 dimensions per head |
| Trainable component | Approximately 11.13M-parameter cache mapper |
| Initialization | Same-head, depth-aligned affine ridge baseline |
| Nonlinear branch | Rank-16 residual reading all heads from three nearest-depth source layers |
| Positional handling | Undo source key RoPE, map, normalize key RMS, apply target RoPE |
| Precision | Frozen model/cache FP16; mapper parameters FP32; adaptive CUDA loss scaling |
| Training documents | 64 official-training documents, 1,088 QA records before document deduplication |
| Mapper validation | 16 separate official-training documents, 272 QA records before document deduplication |
| Development evaluation | All 1,037 questions from 61 official-development documents |
| Final test | Excluded from this experiment |

The mapper is not a LoRA fine-tune of Qwen. Neither base model's weights are updated. This is also not RL: the objective is target-model distillation plus cache reconstruction.

## Pipeline stages and completion evidence

| Stage | What happens | Evidence |
|---|---|---|
| Frozen preparation | Verify dataset versions, split isolation and file hashes | `dataset-manifest.json` |
| GPU startup | Allocate A10G, download image, install pinned adapter dependencies | SageMaker stage transitions and CloudWatch logs |
| Model loading | Download pinned public snapshots; hash checkpoint/tokenizer files | `environment.json`, `model-identities.json` |
| Cache controls | Same-model identity transfer must preserve logits/first token within tolerance; zeroed cache must affect logits | `controls.json` |
| Ridge baseline | Fit affine maps from independent training contexts | `ridge/manifest.json`, weights |
| Mapper training | Up to 400 successful optimizer updates | `training/training-log.jsonl` |
| Validation/checkpointing | Evaluate 16 held-out training-domain documents at step 0 and every 50 updates; retain best objective | Checkpoint rows and `training/checkpoints/` |
| Selected-artifact reload | Save and reload the actual validation-selected mapper, checking hashes | `training/best/`, `training-summary.json` |
| Development decisions | Compare source, native target, ridge transfer and neural transfer | `quality/decision.jsonl` |
| Separate generation | Test unconstrained 16-token responses on one seeded question per dev document | `quality/free_generation.jsonl` |
| Matched latency | Five arms, 24 length-stratified development examples, five repetitions | `latency/traces.jsonl` |
| Long-context timing | Synthetic controls near 8K, 16K and 32K if budgets allow | `long-latency/`, exact token counts |
| Export/stop | Save evidence to S3 and end worker | Final job state and reported billable seconds |

## Inside one optimizer update

1. Sample a deduplicated training document, rather than treating its 17 hypotheses as 17 independent contracts.
2. Tokenize its formatted document/question prompt. Select up to 1,032 contiguous tokens: 1,024 prefix tokens plus eight tail tokens. Most steps sample a random window; every fourth step uses the prompt-ending window. Window positions restart at zero.
3. Run the frozen source on the prefix and collect its cache.
4. Run the frozen native target on the same complete window. Retain its prefix cache and next-token distributions at the eight causal tail positions as teacher targets.
5. Translate the source prefix cache using our mapper. Feed the same eight tail tokens to the frozen target with the translated prefix cache.
6. Compute `loss = normalized_KV_MSE + KL(native_target_distribution || mapped_target_distribution)`, averaged over the declared layers/tail positions. No gold-answer validity is assumed after cropping.
7. Scale the loss, backpropagate through the frozen target operations into mapper parameters, unscale the gradients, and clip their global norm to 1. Frozen model parameters receive no updates.
8. Apply AdamW with learning rate `3e-4`, weight decay `0.001`. Count one update only when the optimizer actually changes state using finite gradients.

CUDA loss scale starts at 1,024. On overflow, the optimizer update is skipped, the scale is reduced, and the exact same token window is replayed. At most eight attempts are allowed for one update, with minimum scale 1 and a deadline. Persistent overflow or invalid norms still fail explicitly. Skipped attempts are logged separately from the target of 400 completed updates.

## How to interpret the training metrics

| Metric | Meaning | What it does not establish |
|---|---|---|
| `completed_updates` / `step` | Actual successful optimizer updates | Percentage of whole pipeline completed |
| `loss` | Weighted cache-reconstruction plus prediction-distillation objective on this window | Gold QA accuracy |
| `kv_loss` | Normalized discrepancy between translated and target caches | Exact semantic preservation |
| `kl_loss` | Difference between native-target and translated-cache prediction distributions | Correctness of the native target itself |
| `validation_loss` | Mean objective on 16 separate mapper-selection documents | Unbiased final benchmark performance |
| `best_step` | Checkpoint with lowest mapper-validation objective so far | Necessarily the latest update |
| `grad_norm` | Gradient magnitude before clipping | A comparable score across architectures/tasks |
| `loss_scale_before/after` | FP16 numerical scaling and recovery state | Learning rate or quality score |
| `overflow_attempts` | Skipped attempts, never counted as successful training updates | Successful progress |
| GPU allocated/reserved MiB | Memory occupied/retained by this worker's allocator | Hardware utilization percentage or production native-only memory |

Loss fluctuates because successive windows differ. A lower training loss is useful evidence but cannot replace the held-out decision/generation results.

## Development scores and latency boundaries

Decision accuracy compares A/B/C next-token logits. Conditional option probabilities and their total full-vocabulary probability mass are recorded: a model that strongly prefers prose may still pick an option when forced, so this metric stays separate from generation quality. Per-class/macro scores and the majority-class baseline are included.

The separate free-generation phase records exact one-letter correctness, invalid format, output-length truncation, and a permissive label-prefix diagnostic. A correct diagnostic prefix is not silently promoted to strict correctness. The first pilot's zero strict scores are preserved.

Prompts over 8,192 tokens are retained as over-limit/incorrect in the development denominator rather than truncated or silently removed. Repeated questions on the same document are clustered for paired intervals.

Cold transfer timing begins before source prefill and ends at first target-token selection: source prefill + mapper/move/import + one target tail token. Native warm-prefix and warm-source handoff are separate arms. These are synchronized, batch-one Hugging Face timings, not HTTP latency or production throughput. Synthetic long-context timing does not establish real long-context QA quality.

## Failure history and current limits

The initial larger run completed 84 updates before an FP16 overflow at attempted update 85. Its step-50 validation objective was 0.7494, versus 11.1422 at initialization. That is mapper-objective improvement, not a QA accuracy result. The corrected job restarts from the ridge baseline with the same seed and adaptive recovery; it does not resume optimizer state from the old checkpoint.

Our current ridge is weaker than a full predictive-layer/all-source-head mapper; the nonlinear branch uses fixed depth neighbors and a low-rank bottleneck. These are experimental choices to audit, not a reproduction of a proprietary product. Further work includes broader calibration, layer-selection/full-cross-head baselines, independent final testing, production serving integration, and actual cost/throughput measurements.

## Pull the latest status

From the project root:

```bash
python3 cloud/full_collect.py --receipt reports/sova-kv-full-20261118-132341-7daf/submission.json --status-only --progress
```

This reads job metadata and a small progress artifact; it does not download checkpoint weights or launch compute. For complete artifacts and logs, omit `--status-only --progress` and add `--logs`. Managed S3 checkpoints contain mapper weights/manifests; exact Adam-state training resumption is not implemented.
