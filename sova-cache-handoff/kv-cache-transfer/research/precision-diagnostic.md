# Target-only precision and attention diagnostic

18 November 2026. The user requested this rerun after the fresh QuALITY native-
cache gate stopped training. This diagnostic does not train a mapper.

## Question and fixed controls

Does native full-prompt versus native prefix-cache continuation disagreement
come from low-precision execution / attention backend selection, or does it
remain under a stricter numerical reference?

Use the **exact same 16 questions in the same order** as
`sova-kv-repair-20261118-143720-3fef`, pinned through its hashed
`native-control.json`. All questions belong to the official QuALITY train
partition reserved for mapper validation. No official development results are
used to select examples or adjust thresholds. Maximum control prompt: 7,929
tokens, with unchanged 8,192-token cap and no truncation.

Only pinned Qwen3-1.7B is loaded. The source model and learned mapper are absent.
Full-prompt and cache-continuation calls use the same existing DenseRuntime
forward implementation, native document→question boundary, masks and positions.
The diagnostic explicitly checks prefix and final cache lengths.

## Three predeclared paths

| Arm | Checkpoint execution dtype | SDPA backend |
|---|---|---|
| fp16_auto | float16 | Automatic selection |
| fp16_math | float16 | Forced math |
| fp32_math | float32 | Forced math |

All arms disable TF32 and reduced-precision FP16/BF16 matrix reductions. Math
SDPA reduced-precision reduction is also disabled where the pinned API exposes
that switch. These settings are recorded. **The FP16 automatic arm uses stricter
reduction settings than the historical default, so it is not an exact default-
settings rerun.** Across the new arms, settings other than dtype/backend stay
fixed. Each arm reloads original checkpoint files; casting already-rounded
FP16 weights back to FP32 would be an invalid high-precision reference.

FP16 auto→FP16 math probes dispatch policy; FP16 math→FP32 math probes arithmetic
precision within the same backend family. This does not claim identical device
kernels or reduction ordering across dtypes. PyTorch documents backend-dependent
numerical differences and the math backend's FP32 intermediate handling for
half inputs: [SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
The worker remains pinned to Torch 2.5.1; newer API switches are not assumed.

## Unchanged acceptance and evidence

Keep `oracle-probability-v2`: same first token; full-vocabulary total variation
≤0.001; both KL directions ≤0.0001 nats. Compute unsmoothed probabilities in CPU
float64 outside model execution. The previous 13/16 pass result remains failed.

Store both full-vocabulary vectors per example/arm, input token hash, checkpoint
identities in the submission, context hash, cache lengths, peak GPU memory,
explicit errors / time-budget outcomes and per-arm summaries. No speed claim
is attached to this correctness diagnostic. If FP32 still fails, inspect cache
positions/masks/import behavior before training. If a path passes, only that
verified path can be proposed for the next bounded training experiment.

## Compute policy

The user explicitly requested this rerun. Local work is limited to short offline
smoke tests and AWS orchestration; no pretrained weights are loaded locally.
107 scoped tests passed with immutable source hashes, zero failures/errors/
skips: [smoke evidence](../reports/smoke-precision-v1/test-results.json).

One SageMaker `ml.g5.2xlarge`, `us-east-2`, 1,200-second job cap and 900-second
worker cap, S3 exports, automatic worker release, no endpoint. Maximum new
runtime compute $0.505 at the recorded $1.515/hour rate; prior recovery-phase
runtime compute $1.18590833, maximum cumulative $1.69090833. Storage, transfer,
tax and billing extras are excluded. Report actual billable duration afterward.

## First diagnostic outcome and bounded memory repair

`sova-kv-precision-20261118-160504-c474` completed both FP16 arms: automatic
12/16 pass, math 11/16 pass; both agreed with full-prompt first tokens on all
16. FP32 math passed its first **10/10** controls (maximum TV 0.00002948), then
ran out of memory during full-prompt attention on the next long example.
It requested another 3.75 GiB on a 22.06 GiB usable GPU. This is a normal Python
CUDA OOM with a traceback; it is not an unexplained host death. Worker released,
470 billable seconds, estimated $0.19779167 compute. Do not call this a 16/16
FP32 success.

The retry `sova-kv-precision-20261118-161633-5f6d` runs **only FP32 math** on all
the same 16 controls. `blocked_attention.py` partitions attention's query axis
into blocks of 256, always retaining **all keys, values and context tokens**.
Implicit causal masks use original absolute query indices; explicit cached-
suffix masks are sliced by their query rows. It does not truncate documents,
change model positions, or chunk model-level prefill into a different cache
schedule. The full model forward still receives the complete original prompt.

This is a new memory-bounded implementation within the math backend; its
rounding is not assumed bitwise identical. Tests compare outputs and gradients
against full attention, including implicit causality, cached suffix masks,
additive/broadcast masks and actual tiny Qwen full/native-cache logits. **112
scoped tests passed**, zero failures/errors/skips. Original failed results
remain unchanged. Compare the prior 10 completed FP32 vectors against retry
vectors before interpreting implementation parity; all 16 must pass the same
unmodified numerical gate.

Retry cap: 810 SageMaker seconds / 510 worker seconds, one Ohio A10G, automatic
stop, no training. Maximum new compute $0.340875; cumulative recovery-phase bound
$1.724575 before storage/transfer/tax extras. User's explicit rerun instruction
authorizes this bounded correction; no local pretrained-model execution.

## Completed retry

All **16/16** blocked FP32 controls passed. Maximum TV 0.0000125962, maximum
forward KL 8.61242e-10, peak allocated GPU memory 9,221.38 MiB. Every first token
agreed. The exact first ten original unblocked FP32 cases also pass 20/20
native/cached implementation-parity comparisons against the blocked version.

395 billed seconds; estimated new compute $0.16622917; worker released.
Both precision jobs together used estimated $0.36402083 runtime compute.
[Complete results and limitations](../reports/sova-kv-precision-20261118-161633-5f6d/RESULTS.md).

The repaired trainer can now require the passed proof and use the same FP32
math/query-block path throughout controls, optimization and evaluation. It
repeats its controls before training. This numerical result does not establish
repaired-mapper quality; downstream comparisons are still required.
