# Training selection and QuALITY execution audit

18 November 2026, job **`sova-kv-repair-20261118-173137-4f04`**. Read-only analysis of saved JSON/JSONL artifacts. No new model inference, GPU job or source change was performed for this audit. This report deliberately does not publish an accuracy headline from an incomplete reference phase.

## Training is finished at its time allowance

| Field | Observed value |
|---|---:|
| Requested updates | 100 |
| Actual completed updates | **48** |
| Training status | `time_budget_reached` |
| Selected mapper step | **48** |
| Initialization validation objective, step 0 | 20.88624881108602 |
| Selected validation objective, step 48 | **3.995068248112996** |
| Fixed mapper-validation questions/documents | 30 / 30 |
| Training elapsed time including final validation | 1,300.484 seconds |
| Recorded overflow attempts | 0 |
| Recorded forward/backward update attempts | 48 |
| Base-model parameters frozen | true |
| Gold labels used in the training objective | false |

Both validation values come from the same fixed training-domain selection; the objective decreased. It is a teacher-distillation selection loss, **not decision accuracy**, and it does not establish preservation of the native model on development benchmarks. Forty-eight updates must not be described as a completed 100-update plan.

## QuALITY phase has closed with budget-limited coverage

The phase retained the exact 100 frozen questions, the prospective 9,000-token cap, all three model paths and the 16-token generation budget. The saved summary reports **`fully_completed: false`**.

| Variant | Planned | Executed `ok` | Runtime errors | Over limit | Not run: phase time budget |
|---|---:|---:|---:|---:|---:|
| Source | 100 | **90** | 0 | 0 | **10** |
| Native target | 100 | **89** | 0 | 0 | **11** |
| Mapped target | 100 | **89** | 0 | 0 | **11** |

There are **300 result rows**: 268 executed results and 32 explicit `not_run_time_budget` outcomes. Eighty-nine questions have all three paths executed. One further question has a source result only; ten have no executed path. The 900-second QuALITY allocation, rather than a recorded memory, context-limit or numerical failure, prevented full execution.

The executed rows passed these independent checks:

- No recorded runtime errors or over-limit outcomes in any arm.
- First-token timing values and conditional option probabilities are finite.
- Decision correctness flags agree with the stored decision and gold option letters.
- Conditional option probabilities sum to one and total option probability mass remains within [0, 1].
- Paired rows use identical prompt/config hashes, gold answers and full token counts.
- Mapped document-prefix tokens plus native query tokens equal the complete prompt length.

These checks establish internal consistency of the recorded execution, not semantic correctness or a speed advantage. No accuracy headline is inferred from them. The original phase scores and its budget-not-run rows must remain unchanged.

## Exact remaining work

[quality-continuation-missing.json](quality-continuation-missing.json) lists the **11 affected question IDs**, their missing variants, original status and the full raw-row checksum. In particular, `quality:dev:61430_R8T5MKW8_4` has source executed but native/mapped missing; the other ten IDs lack all three paths.

A continuation, if authorized and submitted, must reuse the selected step-48 artifact, model/tokenizer hashes, FP32 math/query-block-256 execution, 9,000-token evaluation cap and 16-token generation budget. It must not retrain or select a mapper using these development results. Any aggregate requires an explicit one-result-per-question/variant provenance rule; extra repeated measurements cannot inflate the sample size. This audit has not launched such a continuation.

HellaSwag completion is a separate phase. QuALITY reaching a summary file neither establishes HellaSwag completion nor completes the overall 100-question/1,000-record goal.

## Evidence

- [Training summary](../reports/sova-kv-repair-20261118-173137-4f04/collected/training/training-summary.json)
- [Training log, including both validation checkpoints](../reports/sova-kv-repair-20261118-173137-4f04/collected/training/training-log.jsonl)
- [QuALITY summary](../reports/sova-kv-repair-20261118-173137-4f04/collected/quality/summary.json)
- [All 300 QuALITY result rows](../reports/sova-kv-repair-20261118-173137-4f04/collected/quality/rows.jsonl)
