# Task-aligned recovery run — 18 November 2026

The prior ContractNLI experiment remains a negative result: 50.82% mapped versus
59.11% native decision accuracy, with 0.953× median paired cold speedup. Nothing
in this recovery work establishes quality-preserving acceleration yet.

## Completed handoff diagnostic

SageMaker job `sova-kv-probe-20261118-142148-ed84` completed, using 299 billed
seconds (estimated $0.12583 worker compute at $1.515/hour; extras excluded).
It sampled 16 document-disjoint ContractNLI development questions before model
outputs. The original mapper and original benchmark results are unchanged.

The original raw-logit oracle gate **failed: 45/48** comparisons satisfied its
elementwise `allclose(rtol=.02, atol=.1)` check. All 48 controls agreed on the
first token. Float64 analysis of their saved full-vocabulary vectors found:

| Maximum across 48 native-cache comparisons | Value |
|---|---:|
| Absolute raw-logit difference | 0.1875 |
| Probability difference | 0.000031944 |
| Total variation | 0.000031944 |
| Forward KL, nats | 0.000001521 |
| Reverse KL, nats | 0.000001474 |

This is consistent with precision-dependent execution differences, rather than
an obvious cache-position fault. It does not prove numerical equivalence for
different questions or distributions. Raw-logit closeness also depends on
irrelevant additive offsets; probability comparison is the relevant behavioral
check. The v1 result stays failed, and its six blocked mapped attempts stay
blocked. We did not backfill them after changing the criterion.

On the **14 common completed questions**, native got 9 correct, mapped suffix-1
got 5, and mapped suffix-8/suffix-32 each got 7. Median native-to-mapped KL was
0.8845, 0.0936 and 0.1148 respectively. This small exploratory diagnostic supports
aligning the handoff, but neither proves causality nor a recovered product.
It has no warmup or repetitions, so its timing is not a speed headline.

Raw evidence: [original summary](../reports/sova-kv-probe-20261118-142148-ed84/collected/probe/summary.json),
[attempt ledger](../reports/sova-kv-probe-20261118-142148-ed84/collected/probe/traces.jsonl),
[separate posthoc numerical audit](../reports/sova-kv-probe-20261118-142148-ed84/collected/probe/numeric-forensics.json).

## Fresh gate before training

The revised `oracle-probability-v2` criterion is declared **after** inspecting
the original diagnostic and **before** the new GPU run. It requires the same
full-vocabulary top token, total variation ≤0.001, and both KL directions
≤0.0001 nats. Computation uses unsmoothed full-vocabulary probabilities in CPU
float64. Meaningful negative controls and additive-offset invariance are tested.

The new worker must confirm this criterion on 16 previously unevaluated QuALITY
mapper-validation questions at the actual document→question handoff. It uses
the target's own prefix cache, with no source or mapper. It stores both logit
vectors and hashes. **If any comparison fails, no mapper is trained.** We will
not keep relaxing this gate; the next diagnostic would examine a higher-
precision execution path. Native constrained accuracy must also reach the
predeclared 35% floor. Over-limit native-control articles are declared before
sampling; official evaluation over-limit items still count as failures.

The original historical gate is explicitly marked failed in submission evidence.
`--fresh-oracle-gate` permits a prospective diagnostic/training job, not a
retroactive pass or a bypass of the fresh in-worker gate.

## What changed in the candidate

- Same frozen dense Qwen3-0.6B → Qwen3-1.7B pair and FP16 inference. No base-model
  fine-tuning, model downloads to the laptop, endpoint, or SSH-managed instance.
- Official QuALITY training articles partitioned 120/30, before length filtering.
  All question pools survive document deduplication. Eligible training:
  1,955 questions / 116 articles; mapper validation: 485 / 29. Exclusions are
  recorded without cropping. Actual consumed questions are logged separately;
  400 updates will not consume all 1,955 questions.
- Fresh ridge initialization on training documents, then a new rank-16 neural
  mapper. Prior neural weights are not resumed. This is still an approximate
  linear initializer, not the paper's all-head predictive-layer reference.
- Map the complete document prefix and natively process the entire fresh
  question/options/chat suffix, identically at training and evaluation.
- Distill actual teacher answer positions using token CE plus forward KL, up to
  eight answer tokens. Gold labels do not enter the optimization objective.
- Mapper activation checkpointing and dynamic loss scaling bound memory and
  preserve finite updates; overflow retries and actual updates are recorded.

## Frozen evaluation and limitations

QuALITY: 100 fixed questions from 100 official development articles, separate
from optimization and checkpoint selection. Three prompts exceed 8,192 tokens;
they remain incorrect/over-limit in the denominator. Report source, native and
mapped constrained decisions and separate free-generation outcomes.

HellaSwag: 1,000 fixed official validation examples. Score every ending token;
report likelihood accuracy and character-length-normalized `acc_norm`. This is
an out-of-domain mechanism check, not a FineWeb-trained paper reproduction.
Errors and budget-expired rows remain in the denominator.

The pair has identical layer count, query/KV-head counts and head dimensions.
Its quadratic attention work and cache payload are therefore similar; parameter
ratio alone cannot promise a large prefill speedup. The first goal is recovered
behavior, followed by profiling and a better model pair if needed.

No exact OneTriangle replication is claimed: our pair, sampled questions and
some prompt/scoring choices differ from its published campaigns.

## Verification and compute policy

**101 tiny offline CPU tests passed**, zero failures/errors/skips, with immutable
scoped source hashes: [test evidence](../reports/smoke-qa-repair-v2/test-results.json).
The laptop runs smoke tests and cloud orchestration only. Heavy model work runs
on one SageMaker `ml.g5.2xlarge` in `us-east-2`, with a 3,600-second job limit,
3,300-second worker limit, S3 progress/checkpoint export and automatic teardown.
Training targets 400 updates within a maximum 1,800-second training window;
benchmark stages share the remaining worker budget, preserving incomplete rows.

The user approved executing this recovery plan. New scope or materially larger
compute requires a concrete command, machine, duration, cost and output review.
Maximum new runtime compute is $1.515, making the current recovery-phase bound
$2.57718 including $1.06218 prior compute. Storage, transfer, tax and billing
extras are excluded. Report actual billed duration after completion.
