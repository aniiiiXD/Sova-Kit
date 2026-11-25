# Next experiment: repair the comparison before scaling

18 November 2026. This is a prepared diagnostic plan, not a reported improvement or authorization for another paid run.

## Decision

Keep the completed ContractNLI result as a negative development result. Add **QuALITY** as the vendor-style long-document reference and **HellaSwag continuation likelihood** as the paper-style mechanism reference. Test the handoff mismatch on the saved mapper before retraining. Then compare a stronger cross-head ridge baseline and a repaired question-conditioned training protocol.

The vendor already reports ContractNLI alongside QuALITY, so the dataset name does not explain our failure by itself. Its generic, task-specific and newer hybrid campaigns use different model pairs, datasets and scoring settings. Our frozen samples are independent reference slices, not the vendor's unpublished exact items. See the [primary-source protocol audit](original-benchmark-protocol.md).

## Evidence motivating this order

- On 1,037 development questions, neural accuracy is 50.82%, native 59.11% and always-A 50.05%. Neural predicts A 929 times; all 17 questions receive A on 42/61 contracts. This is substantial behavior loss.
- Saved mapper registries contain only `nda-1` across all 64 training and 16 selection documents. Preparing 1,088 training QA records did not mean all those questions were consumed.
- Training recomputed eight target suffix tokens; deployed transfer recomputed one. The pinned-tokenizer audit confirms the last eight input tokens are fixed assistant/thinking-template framing. The first seven ending-window prediction positions supervise more framing; only the final one predicts the first answer token. This is not per-position loss attribution, and it does not establish the mismatch's causal size.
- The ridge is a same-head, depth-aligned approximation; the published reference mixes source heads and selects source layers predictively. The nonlinear rank-16 branch is not a faithful replacement for that reference.
- The completed cold contract timing result is 0.953× median paired speedup; synthetic actual lengths of 7,193, 14,564 and 29,315 tokens gain only about 6–7%. Quality-preserving acceleration is unestablished.

Detailed evidence: [failure analysis](failure-analysis-v2.md), [architecture audit](architecture-recovery-plan.md), [completed results](../reports/sova-kv-full-20261118-132341-7daf/RESULTS.md), [exact tokenizer suffix](../reports/sova-kv-full-20261118-132341-7daf/collected/tokenizer-audit/audit.json).

## Ready now

### Frozen reference data

[Dataset manifest and instructions](../datasets/reference-benchmarks-v1/README.md):

| Reference | Prepared data | Prospective evaluation |
|---|---|---|
| QuALITY v1.0.1 | Official train: 2,523 questions / 150 articles; dev: 2,086 / 115 | Seed 19: 100 questions from 100 distinct dev articles, 43 difficult |
| HellaSwag | Official validation: 10,042 examples | Seed 19: 1,000 examples; preserve complete endings and pinned likelihood scoring |

Source revisions, content hashes, sample IDs, per-article licensing and uncertainty are recorded. Official test files were not read. Fourteen artifact-hash checks and three preparation tests passed. No model scores exist on these newly prepared references.

QuALITY should use a declared question/answer protocol and sufficient output budget; distinguish native/mapped/source and free-generation versus constrained-choice metrics. HellaSwag must score every ending token under the pinned lm-evaluation-harness protocol and report `acc_norm`; generating an option letter is not equivalent.

### Saved-mapper handoff probe

[`kvverify/boundary_probe.py`](../kvverify/boundary_probe.py) is implemented; **nine tiny/offline tests passed**. Its `tail-boundary-v1` protocol freezes up to 16 distinct documents before model outputs and compares the same saved mapper with native suffix lengths 1, 8 and 32. It logs prompt/token/checkpoint hashes, imported/recomputed positions, raw full-vocabulary first logits, target agreement, native-to-mapped KL, option probability mass and complete cold timing. Failures remain in the ledger, with no hidden recomputation.

The probe is an API callable for a separately configured GPU worker; it does not itself download models or launch SageMaker. Its timing has no warmup/repetition protocol and must not be used as a production speed headline. The new files require a new integrated test/hash freeze before any later submission.

## Controlled repair sequence

1. **Check the oracle, then probe the boundary.** At representative lengths, the target's own prefix cache plus suffix must reproduce its full-prompt result. Then compare current learned-cache suffix lengths 1, 8 and 32. A large recovery at 8 supports aligning the training/deployment handoff; no recovery redirects attention to representation fidelity.
2. **Restore question diversity.** Deduplicate document storage and covariance statistics, but keep each document's complete question pool for functional training/selection. Record actual question-type consumption.
3. **Align task supervision.** Use full-context answer-boundary/teacher-answer distillation with the same native suffix policy that inference will use. The vendor publicly describes task-answer distillation; our cropped next-text objective is a different experiment. Mapping only the shared document prefix and allowing the target to read the fresh query is a separate measured alternative.
4. **Establish a stronger linear reference.** Compare same-head versus all-head features, predictive layer choice and explicit regularization units. Isolate key-RMS normalization rather than changing it together with the mapper and loss.
5. **Run the frozen benchmark protocol.** Verify a sensible native baseline first. Evaluate the chosen method on the same frozen items and preserve every error. Keep current ContractNLI results as a separate regression test. Model pair, prompt, reasoning mode and scoring must be stated before comparing to published numbers.
6. **Optimize only a behavior-preserving candidate.** Profile mapper/import/kernel overhead and choose a production connector after quality passes; a faster degenerate answer is not the product target.

This ordering is intended to learn which intervention matters. It does not guarantee a positive result or justify selecting whichever benchmark looks best afterward.
