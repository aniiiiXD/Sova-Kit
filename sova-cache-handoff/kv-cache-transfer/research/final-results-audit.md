# Final independent results audit — PASS

19 November 2026. Local read-only verification of the completed merged reference evaluation. No GPU/model execution, external evaluation or source-code changes were performed. This audit recomputed scores from raw stored predictions/log-likelihoods and the frozen dataset labels; it did not merely copy summary values.

Parent job: `sova-kv-repair-20261118-173137-4f04`.
Evaluation-only continuation: `sova-kv-eval-resume-20261118-183814-e659`.

## Execution completeness and integrity

| Check | Independently verified result |
|---|---|
| Frozen QuALITY IDs | Exactly 100 distinct selected question IDs for each source/native/mapper arm; 300 unique rows total |
| Frozen HellaSwag IDs | Exactly 1,000 distinct selected record IDs for each arm; 3,000 unique rows total |
| Successful execution | All 3,300 merged rows have `status: ok` |
| Errors, over-limit and budget-not-run outcomes in merged result | Zero in every benchmark/arm |
| Frozen data | Current selected-file hashes match both the frozen dataset manifest and continuation input-hash manifest |
| Gold labels | Every QuALITY gold option and HellaSwag gold ending index matches the corresponding frozen record |
| Parent/child QuALITY protocol | Same protocol, seed, model identities, mapper manifest, 9,000-token cap and 16-token generation budget |
| Numerical settings | Parent/child environment fields agree on float32, math attention and query-block size 256 |
| Selected mapper identity | Same checkpoint weight checksum across parent selected manifest, parent benchmark configs, child benchmark config/environment and continuation config |

Fixed mapper checksum:

```text
e2f6390f176a0c4ef262edfaf18a133e5fe88a5a33a898c5415c757269d53c51
```

This is consistency verification across the saved manifests/configuration and execution provenance. The audit did not load the model or download/re-hash unavailable weight bytes.

## Continuation selection and unchanged original results

- The **89 complete parent QuALITY triplets** retain every original row field unchanged. The only extra merged field is provenance metadata.
- The remaining **11 triplets** match the 33 child rows exactly, including their original metrics and run configuration hashes.
- The parent's one source-only result for `quality:dev:61430_R8T5MKW8_4` was replaced atomically together with the entire incomplete triplet. Its old source result was not mixed with child native/mapped results or selected based on correctness.
- All **3,000 HellaSwag rows** retain every original parent row field unchanged; HellaSwag was not re-evaluated or altered by the continuation.
- Canonical hashes of the parent row collections match the merged summaries' recorded parent hashes.

Selection was based on missing/non-`ok` execution status. The merged data contain one final result per frozen ID and variant; repeated source execution in the one incomplete triplet does not inflate the denominator.

## Independently recomputed QuALITY results

Decision predictions were recomputed by argmax over stored A/B/C/D logits and compared with frozen gold labels. Exact-letter generation correctness and format validity were recomputed directly from the saved decoded text.

| Arm | Decision correct / 100 | Decision accuracy | Strict generation correct / 100 | Valid exact-letter outputs / 100 |
|---|---:|---:|---:|---:|
| Source 0.6B | **47** | **47%** | 1 | 1 |
| Native target 1.7B | **59** | **59%** | 5 | 9 |
| Mapped target | **26** | **26%** | 4 | 18 |

The mapped-minus-native decision difference is **−33 percentage points**. An independent 1,000-replicate, seed-19 document-cluster bootstrap over the 100 article groups reproduces the saved interval exactly: **[−45, −19] percentage points**.

This is the declared custom QuALITY decision/generation protocol, not an exact private-vendor prompt or published-score reproduction. Strict generation remains separate from constrained option scoring.

## Independently recomputed HellaSwag results

For each ending, the stored complete token-log-likelihood was ranked directly. Normalized ranking was recomputed by dividing that likelihood by the frozen processed ending's Python character length, excluding the delimiter. Stored prediction indices and accuracy flags match these independent calculations.

| Arm | Raw `acc` correct / 1,000 | Raw `acc` | Character-normalized correct / 1,000 | `acc_norm` |
|---|---:|---:|---:|---:|
| Source 0.6B | **373** | **37.3%** | **473** | **47.3%** |
| Native target 1.7B | **469** | **46.9%** | **612** | **61.2%** |
| Mapped target | **264** | **26.4%** | **285** | **28.5%** |

The mapped-minus-native normalized difference is **−32.7 percentage points**. Independent seed-19, 1,000-replicate bootstrap resampling of the **977 source groups** reproduces the saved interval: **[−36.8211, −28.9866] percentage points**.

This is a frozen 1,000-record zero-shot continuation-likelihood slice and an out-of-domain test of the QuALITY-trained mapper, not a FineWeb-trained paper replication.

## Verdict

**Audit PASS: the complete frozen reference results and their continuation provenance are internally consistent and independently reproducible from the saved artifacts.**

**Model-quality outcome: negative.** The mapped candidate substantially underperforms the native target on both declared decision/likelihood metrics. Completing the memory recovery and benchmark execution does not establish useful quality retention, production readiness or a speed advantage. The continuation explicitly disallows a merged cross-job latency headline.

## Evidence

- [Merged QuALITY rows](../reports/sova-kv-eval-resume-20261118-183814-e659/collected/merged/quality/rows.jsonl) and [summary](../reports/sova-kv-eval-resume-20261118-183814-e659/collected/merged/quality/summary.json)
- [Merged HellaSwag rows](../reports/sova-kv-eval-resume-20261118-183814-e659/collected/merged/hellaswag/rows.jsonl) and [summary](../reports/sova-kv-eval-resume-20261118-183814-e659/collected/merged/hellaswag/summary.json)
- [Original parent QuALITY rows](../reports/sova-kv-repair-20261118-173137-4f04/collected/quality/rows.jsonl)
- [Child QuALITY rows](../reports/sova-kv-eval-resume-20261118-183814-e659/collected/quality/rows.jsonl)
- [Original parent HellaSwag rows](../reports/sova-kv-repair-20261118-173137-4f04/collected/hellaswag/rows.jsonl)
- [Frozen continuation plan](../reports/sova-kv-eval-resume-20261118-183814-e659/collected/resume-plan.json)
