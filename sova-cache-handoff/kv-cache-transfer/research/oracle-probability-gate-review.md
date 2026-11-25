# Native-cache oracle gate: independent numerical review

18 November 2026. This review inspects saved evidence and recomputes first-token probability statistics in CPU float64. It does not run a model or change any historical result.

## Diagnosis supported by the evidence

The original `tail-boundary-v1` run remains **failed: 45/48 oracle comparisons passed its declared gate**. Three native-prefix controls failed only raw-logit `allclose`; all 48 retained the same first token. Two documents therefore have six blocked mapped cases, which remain unmeasured rather than being filled in after the fact. [Original summary](../reports/sova-kv-probe-20261118-142148-ed84/collected/probe/summary.json), [original traces](../reports/sova-kv-probe-20261118-142148-ed84/collected/probe/traces.jsonl).

I independently recomputed all 48 native/oracle full-vocabulary probability pairs from their saved logit vectors. Across these pairs:

| Quantity | Maximum |
|---|---:|
| Total variation distance | 0.0000319431 |
| Absolute difference in one token's probability | 0.0000319431 |
| Forward KL, native → oracle | 0.00000152065 nats |
| Reverse KL, oracle → native | 0.00000147390 nats |
| Raw absolute logit difference | 0.1875 |

For the three raw-`allclose` failures, the native probability mass on the offending coordinates is approximately `1.14e−23`, `8.73e−24` and `2.66e−24`. Those coordinates matter to the raw elementwise tolerance but negligibly to these next-token distributions. The independently generated [numeric forensics](../reports/sova-kv-probe-20261118-142148-ed84/collected/probe/numeric-forensics.json) records the probability-level audit without replacing the old gate.

This strongly favors **an overly restrictive raw-logit criterion for FP16 path comparison**, not an identified cache-position or shape fault. It does not prove which kernel produced the differences. Full-prompt and split-prefix/suffix execution have different matrix shapes; PyTorch explicitly does not guarantee bitwise agreement between mathematically equivalent batched/sliced floating-point computations. [Pinned PyTorch 2.5.1 numerical-accuracy notes](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst).

Raw logits are also not identifiable probabilities: adding a constant to every vocabulary logit leaves softmax unchanged. Therefore raw-logit `allclose` is a useful diagnostic but a poor sole behavioral-equivalence gate. The existing token counts, checkpoint identities, cache geometry, causal positions and finite-value checks should remain mandatory.

## Separate, explicit revision

The new utility `probability_equivalence(torch, reference, candidate)` declares **`oracle-probability-v2`**. It does not alter `run_probe`, the v1 gate, its thresholds or its recorded outcome. It evaluates detached complete-vocabulary vectors on CPU in float64, outside GPU/model timers.

All conditions must hold:

- Identical nonempty one-dimensional shapes and finite input logits.
- Identical full-vocabulary first-token argmax.
- Total variation distance **≤0.001**.
- Forward and reverse KL **each ≤0.0001 nats**.

No epsilon smoothing, option-only renormalization or low-probability coordinate deletion is used. Only negative KL summation roundoff in `[−1e−12, 0)` may become zero; materially negative or nonfinite divergences fail explicitly. The output retains raw and clipped KL values, clipping flags and the declared thresholds.

These thresholds are a **post-hoc protocol revision motivated by the numerical audit**. Historical vectors can illustrate the revision; they cannot supply prospective confirmation. The next worker must apply this gate unconditionally to fresh native-cache controls before training, with its new protocol and selection frozen in advance. No previously blocked mapped case becomes observed by changing a JSON summary.

## Required safeguards

The prepared tests check invariance to an additive logit shift, rejection of changed meaningful probabilities even with the same argmax, rejection of an argmax change, malformed/nonfinite inputs, the exact roundoff bound, and input/autograd-state preservation. They also distinguish harmless low-probability raw-logit discrepancies from substantial distribution changes.

Fresh controls should cover the **actual document-prefix/native-question boundary** used by repair training, not only a convenient suffix on the old contracts. Higher-entropy QuALITY answers may expose larger probability shifts than the nearly saturated ContractNLI distributions. If the new gate fails there, retain that failure and inspect a bounded higher-precision or math-attention reference; do not repeatedly loosen thresholds after seeing each new result.

Physical negative controls such as zeroed, shuffled or position-shifted cache should remain separate sanity checks. Passing a one-step probability test on 16 inputs establishes neither every cache tensor's equality nor full generated-sequence equivalence. Mapper quality, free generation, full benchmark scoring and end-to-end latency remain independent acceptance requirements.
