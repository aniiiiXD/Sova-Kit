# First experiment card: locate cache damage before retraining

19 November 2026 · V1 guard failure preserved ·142 v2 smoke tests passed · V2 completed24/24

**Question:** is the first large error caused by the mapper's key-scale projection, key addressing, value readout, or accumulated receiver drift?

This is a diagnostic on existing internal validation data. It cannot establish final task quality or a serving speedup.

## Fixed inputs and identities

- Keep the existing pinned Qwen3-0.6B → Qwen3-1.7B pair, tokenizer, full document/native-query boundary, FP32 math-attention reference and absolute position handling.
- Use the original untrained step-zero artifact and selected step48 artifact. Step48 SHA-256: `e2f6390f176a0c4ef262edfaf18a133e5fe88a5a33a898c5415c757269d53c51`.
- Before execution, record exact step-zero hash, all source/input hashes and experiment configuration in a new receipt. Reuse neither an ambiguous checkpoint name nor an unverified reconstructed initializer.
- Select 24 of the existing 30 mapper-validation document/questions by a seeded length-stratified rule: sort eligible full-prompt lengths, form three ten-document bins, select eight from each with seed19. Record IDs before measuring anything. Do not select by answer correctness or mapper failure severity.
- These internal-validation questions are already used for model selection. Call them diagnostic data. Do not touch newly reserved final-test documents.
- No fitting, optimizer updates, prompt truncation, model-size change, fresh hyperparameter search, or precision optimization in this experiment.

## Comparisons

| Arm | What changes | What stays fixed |
|---|---|---|
| Native control | Target-native cache versus native full forward | Same exact prompt and numerical tolerance as the accepted reference |
| Initializer, RMS on/off | Only the scalar key-RMS projection | Same step-zero affine/residual weights, cache and query |
| Trained mapper, RMS on/off | Only the scalar key-RMS projection | Same step48 weights, cache and query |
| K-only substitution | Mapped prefix keys with native prefix values | Native receiver queries, untouched native suffix keys/values and causal mask |
| V-only substitution | Native prefix keys with mapped prefix values | Same native queries/suffix/mask |
| Both substituted | Mapped prefix keys and values | Same native-query diagnostic reference |

Capture per-layer readouts under native target queries first. Then process the complete native question suffix from the mapped cache and measure its first-answer distribution and layerwise query drift. Label the former **oracle diagnostic** and the latter **actual transfer**. This first card does not evaluate free-generation rollout. They are different experiments, not interchangeable scores.

For the step-zero affine comparison, establish whether an exact ridge artifact exists. If deriving an affine-only control from the neural checkpoint, disable its residual explicitly and call it a reconstructed affine control; do not describe it as a separately verified historical ridge checkpoint.

## Measurements and decision rule

Record per document, layer and head: K/V norm statistics, key attention-logit scale, attention entropy/divergence, attention-output relative error/cosine, full-vocabulary first-answer KL, total option mass and conditional native-choice agreement. Gold accuracy is a secondary diagnostic, with its small sample size stated. Keep all rows, including failures.

Compare arms using paired per-document changes and length strata, not a pooled count of correlated tokens. Any intervals are exploratory diagnostic uncertainty, not final noninferiority evidence. Preserve absolute errors alongside relative changes; near-zero native quantities can make ratios unstable.

- If RMS-off consistently reduces readout and full-stack distribution errors, take a normalization-controlled initializer into the next bounded validation experiment. Do not declare the trained mapper fixed from this slice.
- If only K or V substitutions dominate, prioritize that component's mapping/calibration rather than enlarging both indiscriminately.
- If local native-query errors are small but full-stack continuation diverges, study compounding/query-distribution shift; native-query probe fidelity alone is insufficient.
- If feature alignment remains poor, next compare same-head and all-head k1 linear references on identical calibration data. Report their parameter and fitting-memory costs.
- If neither intervention supports its hypothesis, retain the negative result and revise the explanation. Do not silently add a larger MLP or extend training within this card.

## Execution boundary

Before a future run, finish the diagnostic implementation, appropriate synthetic numerical checks, resource estimate, timeout/cost ceiling and receipt. Stop on native-control failure, nonfinite results or breached limits; do not loosen numerical tolerances after seeing outcomes. The run should export raw measurements, configuration, coverage and a one-page interpretation, and release its worker.

No compute allocation, price quote, elapsed-time promise or submission authorization is implied by this planning document. The present task created the thesis and card only.

## Prepared implementation and compute policy

Implementation: [`state_diagnostic.py`](../../kvverify/state_diagnostic.py), [`worker`](../../cloud/state_diagnostic_worker.py), [`launcher`](../../cloud/state_diagnostic_submit.py), and [synthetic tests](../../tests/test_state_diagnostic.py). Production mapper defaults remain unchanged; an explicit diagnostic argument disables the scalar key-RMS projection. CPU-staged caches and256-token mapping blocks limit GPU temporary residency. Native-query readout must reproduce actual projected target attention within prospective `atol=2e-5, rtol=2e-4`; the unchanged full-vocabulary native-cache oracle also gates each document.

Both artifact manifests were retrieved read-only from S3. Step-zero weights hash: `00494f97de95d0b180dbd5849d08e12f264fd826d5493161eb91815548dcd98b`; step48 hash is above. Worker manifests use canonical sorted compact JSON hashes to avoid whitespace/newline ambiguity. Source/data hashes, exact model revisions, selection rule and raw first-token logits are saved. No official-dev/final-test records are bundled; only the original official-train source is needed.

Pending explicit per-run approval under the local compute skill:

```sh
../sova-sys1/.venv/bin/python scripts/run_state_diagnostic_smoke.py --output reports/smoke-state-diagnostic-v2
```

Machine: laptop CPU only; expected under30seconds, no network/GPU/model downloads. Produces a synthetic-test log and source-hash receipt. V1 passed141tests; the corrected v2 source requires fresh test evidence.

Only if that exact source passes, run:

```sh
../sova-sys1/.venv/bin/python -m cloud.state_diagnostic_submit --smoke-report reports/smoke-state-diagnostic-v2/test-results.json --output reports/state-diagnostic-v2 --max-runtime-seconds 1800 --max-compute-usd 0.80 --submit
```

Machine: one managed AWS SageMaker `ml.g5.2xlarge` in `us-east-2`; expected15–25minutes of worker activity, hard1800-second training-job runtime limit with a shorter internal diagnostic deadline. Current public rate verified19November:$1.515/hour; maximum estimated runtime compute$0.7575, capped by launcher at$0.80. Storage, transfer, provisioning/billing extras and tax are excluded; this is not an all-in billing cap. The laptop only conducts submission/status collection. Source/model downloads and inference occur on AWS, with no optimization or retraining. Managed worker terminates on completion/failure/timeout; no endpoint is deployed. A failed test or changed variant requires a new run approval, not an automatic retry.

[Main thesis](THESIS.md) · [Implementation audit](implementation-audit.md) · [Evaluation protocol](experimental-design.md)

## Execution record

User approved both original v1 commands. Smoke completed141tests in2.63seconds with zero failures/errors/skips. One SageMaker job submitted: `sova-kv-state-diag-20261118-195037-b5e8`, `us-east-2`. [Submission receipt](../../reports/state-diagnostic-v1/submission.json); [smoke evidence](../../reports/smoke-state-diagnostic-v1/test-results.json). That approval did not authorize retries or training.

V1 has now terminated after failing the first native-readout probe guard; [result and diagnosis](../../reports/state-diagnostic-v1/RESULTS.md). Billable289seconds; estimated runtime compute$0.12162. No completed mapper-quality result. The v2 source captures actual SDPA tensors and replays complete native shapes before sampling; tolerances remain unchanged. New failure details include layer, maximum absolute error, relative L2 error and SDPA shapes. The archived v1 receipt/source hashes remain historical evidence.

Pending fresh approval, v2 commands use `reports/smoke-state-diagnostic-v2` and `reports/state-diagnostic-v2`, respectively; all other run limits and costs above remain unchanged. V2 tests have not run and no second GPU job has been submitted.

V2 user approval received. Corrected142-test suite passed in2.47seconds with zero failures/errors/skips. Submitted `sova-kv-state-diag-20261118-201105-0c35` in us-east-2, same1800-second job limit and estimated maximum runtime compute$0.7575. [V2 receipt](../../reports/state-diagnostic-v2/submission.json); [V2 smoke evidence](../../reports/smoke-state-diagnostic-v2/test-results.json). Results pending.

V2 now **Completed**: all24documents and four variants executed; native controls passed. [Final findings and evidence](../../reports/state-diagnostic-v2/RESULTS.md). Trained RMS-on/off/native correctness11/8/13 of24; normalization off improved initializer KL on24/24 but worsened trained KL on23/24. Billable947seconds; estimated runtime compute$0.39853. Worker terminated, no training or next experiment submitted.
