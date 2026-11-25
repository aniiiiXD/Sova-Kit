# Live training audit: memory recovery is running, quality results pending

Snapshot: **18 November 2026, approximately 18:02 UTC**, job `sova-kv-repair-20261118-173137-4f04`. Read-only analysis of downloaded JSON/JSONL artifacts; no model execution or source changes. The downloaded training log ends at **46 completed updates**. This is an interim snapshot, not the final training/evaluation result.

## What the actual run has done

| Observation | Saved evidence |
|---|---|
| 46 completed optimizer updates | Steps 1–46 appear once each in `training-log.jsonl` and `question-consumption.jsonl`. Their question-ID sets match exactly. |
| 46 distinct questions, documents and question signatures | No repeated question/document in these 46 updates. The document sampler has not yet completed its first 115-document cycle. |
| Real training pool | Actual protocol: 2,032 input training questions; 1,949 eligible questions across 115 eligible training documents. The mapper-validation input has 491 questions. These are the worker's actual partition counts, not the earlier alternate-partition preflight totals. |
| No numerical failure in these updates | Every recorded total loss, CE, KL and gradient norm is finite. All 46 `gradients_finite` flags are true; zero overflow attempts are recorded. |
| No gold-label training | All 46 records have `gold_answer_used: false`. Supervision is the frozen target's generated answer trace and distributions. |
| Accepted current budget | Receipt requests 100 updates, with training-time allowance 1,100 seconds; SageMaker limit 3,600 seconds and worker limit 3,300. Earlier 64-update/2,560-second proposals are not this accepted run. |

At this snapshot, consumption covers **46/115 eligible documents** and **46/1,949 eligible questions**. This demonstrates varied sampling; it does not mean the entire question pool was trained on.

## Teacher answer distribution is not collapsed to one label

Counting whether the **first token itself** is the exact option token A/B/C/D gives:

| First token | Updates |
|---|---:|
| A | 13 |
| B | 8 |
| C | 10 |
| D | 10 |
| Other | 5 |

All five “other” cases begin with the token for `Answer`, followed by text such as `Answer: C` or `Answer: B`. They are a formatting difference, not evidence that the teacher lacked an option label. This is a teacher-output diagnostic, **not teacher accuracy or mapped-model accuracy**.

Teacher traces contain two to eight tokens, median eight. Eighteen traces reached EOS; 28 reached the eight-token training cap. Long teacher explanations can therefore be truncated, and the recorded cap remains part of the declared training objective.

## Objective behavior: finite, descriptively lower, not an accuracy result

| Statistic over completed updates | Value |
|---|---:|
| First 10 updates, median total objective | 15.0236 |
| Last 10 updates, median total objective | 3.6263 |
| Minimum / maximum total objective | 0.5117 / 22.8918 |
| Update 46 total objective | 0.5117 |
| Initialization validation objective | 20.8862 |

Different updates use different documents, questions and teacher traces, so this comparison cannot isolate the effect of learning. It is consistent with a numerically functioning optimization loop, but does not prove convergence, lower held-out loss or improved decision accuracy. The final fixed mapper-validation pass and the actual reference benchmarks must supply those observations separately.

## Memory evidence

The refreshed `memory-preflight.json` reports **passed** for the two selected stress cases:

- Full prompt 8,043 tokens: 7,918-token mapped prefix plus 125-token native query suffix.
- Full prompt 7,418 tokens: 7,084-token mapped prefix plus 334-token native query suffix.

The preflight artifact explicitly says `optimizer_state_materialized: false`; do not describe that isolated probe as a full Adam-state memory test. The subsequent **46 actual optimizer updates** provide the stronger end-to-end evidence that the real optimizer state and selected training cases fit so far.

At the saved heartbeat after update 46:

| Memory field | Reported MiB | Approximate GiB |
|---|---:|---:|
| GPU currently allocated | 9,026.74 | 8.82 |
| GPU peak allocated | 14,289.27 | 13.95 |
| GPU reserved / peak reserved | 15,248.00 | 14.89 |
| GPU total | 22,589.63 | 22.06 |
| Host peak RSS | 25,447.63 | 24.85 |

Host RSS here is a **recorded peak**, not a current-RAM reading. These observations support “the repaired path has survived the stress tests and 46 real updates”; they do not prove that every future input or evaluation phase will fit.

## Immediate timing forecast

At update 46, training elapsed time is **1,076.87 seconds**, against the 1,100-second allowance. Median inter-update time is **17.77 seconds** over the available differences and **15.55 seconds** for the last ten. At that rate, approximately **47–48 total updates** are plausible before the training loop's time gate; it should not be described as a completed 100-update run. A step started before the deadline can finish after it, and the final validation/export adds time after the last update.

The priority after selecting the checkpoint is to finish the unchanged 100-question QuALITY and 1,000-record HellaSwag evaluations, retaining all three arms and all error/budget counts. No reference-benchmark accuracy, quality retention, speedup or completed-goal claim is available from the training records alone.

## Sources

- [Actual submission receipt](../reports/sova-kv-repair-20261118-173137-4f04/submission.json)
- [Training log](../reports/sova-kv-repair-20261118-173137-4f04/collected/training/training-log.jsonl)
- [Question-consumption log](../reports/sova-kv-repair-20261118-173137-4f04/collected/training/question-consumption.jsonl)
- [Actual QA protocol](../reports/sova-kv-repair-20261118-173137-4f04/collected/training/qa-protocol.json)
- [Refreshed memory preflight](../reports/sova-kv-repair-20261118-173137-4f04/collected/training/memory-preflight.json)
- [Saved audit heartbeat](../reports/sova-kv-repair-20261118-173137-4f04/collected/audit-progress.json)

Live files can gain later rows. This audit's diversity, numerical and objective statistics are explicitly restricted to the snapshot's first 46 completed updates.
