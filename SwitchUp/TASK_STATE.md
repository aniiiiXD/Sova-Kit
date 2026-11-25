# SwitchUP trace-to-LoRA task state

**Updated November 25, 2026**

**Current state** · Initial collection complete. Five public traces saved as 81 SwitchIt events. One TradesBench simulator database yielded 255 indexed call outcomes. The two missed appointments are traced to late arrivals after earlier jobs. The next task is to verify a better booking choice from a matched starting state. SageMaker launch paused.

## Sova-Kit release

- [x] Group SwitchIt, SwitchUp, Video Harness, and Cache Handoff in one checkout. Place trace and run data in the sibling data directory and ignore local Aegis state in Git.
- [ ] Verify each module's install and entry point from a clean checkout. The current CI covers the SwitchIt Python and Node packages. Video Harness uses Node 22 or later.
- [ ] Demonstrate one complete trace-to-correction round with a fixed replay suite and verify activation and rollback. Track the LoRA pilot separately through the run plan below.

## Run plan

**Collect** → Import one published trace or capture an authorized workflow run → Join the request, tool calls, tool results, and final state → Verify the outcome against the benchmark’s own checks → Expand to more runs

**Label** → Record the action and observed outcome → Apply the fixed outcome check → Attach a verified correction to the specific failed decision → Retain successful cases as regression controls

**Freeze** → Group related turns by whole run and seed → Prepare training examples → Seal fresh held-out tasks → Pin the checkpoint, prompt, environment, and grader

**Run** → Approve one bounded SageMaker job → Measure the base model → Train LoRA on the training set → Replay the adapter on the same held-out tasks from identical starting states

**Decide** → Compare outcomes case by case → Save the adapter, logs, and checks to S3 → Review the repair and regression evidence before accepting the candidate

## Evidence already verified

| Source | Observed inventory | Role in the plan |
| --- | --- | --- |
| [TradesBench public artifacts](https://huggingface.co/datasets/adityabressl/trades-bench/tree/0c3983bb519c9885a63f3085818ca516e3ec2067/runs/seed5) | Five completed model runs with 1,316 turn logs. The 1,176 calls break down into 694 bookings, 176 customer declines, 265 agent rejections, and 41 acknowledged cancellations. There are 70 opening and 70 closing turns. | Indexed all 255 call outcomes for the GPT-6.1 Sol run and collected four matching traces. The public runs share seed 5, so reserve fresh seeds or distinct tasks for the held-out comparison. |
| [EnterpriseOps-Gym public sample](https://benchmarkrouter.newmeasure.ai/benchmarks/enterpriseops_gym/sample) | One published Teams run with six model turns, ten tool calls, ten tool receipts, and a reported result of eight of eight checks passed. | Importer development. Four of the original SQL checks returned the expected result on receipt data. |
| [EnterpriseOps-Gym public task manifest](https://datasets-server.huggingface.co/size?dataset=ServiceNow-AI%2FEnterpriseOps-Gym) | 649 runnable tasks across eight domains in the `oracle` configuration. | Supplies fresh tasks for paired runs from reset starting states and checks on the final database. |

The two quantities above use different units. A recorded turn belongs to an existing execution. A runnable task produces a new trace each time a model attempts it. [EnterpriseOps-Gym runner](https://github.com/ServiceNow/EnterpriseOps-Gym)

## Collected Teams sample

Match the published request to the versioned task → Collect the displayed turns and tool receipts → Normalize 27 SwitchIt events → Run the original chat and message SQL against rows from the receipts → Save the audit and source hashes. [Audit](../../Sova-Kit-data/traces/enterpriseops-gym/task_20251121_102744_757_7ebc1127_dadb0c94/audit.md) · [Evidence](../../Sova-Kit-data/traces/enterpriseops-gym/task_20251121_102744_757_7ebc1127_dadb0c94/evidence.json) · [Events](../../Sova-Kit-data/traces/enterpriseops-gym/task_20251121_102744_757_7ebc1127_dadb0c94/events.json)

The published grade is 8/8. Two chat creation checks and two message checks reproduced on a SQLite projection of the recorded receipts. The two cancellation checks and two chat membership checks require final database rows. Original tool call IDs are absent, so the audit marks links by published order as inferred. The trace yielded zero exact replay fixtures and is reserved for importer development.

From Sova-Kit, recreate the bundle with `python3 SwitchIt/Traces/collect_enterpriseops.py`. [Collector](../SwitchIt/Traces/collect_enterpriseops.py)

## Collected TradesBench outcomes

Verify the published simulator database hash → Join each call to its simulator invocation → Join booked calls to their jobs and appointments → Verify selected decisions against the original tool receipts → Save the outcome index and four normalized traces. [Audit](../../Sova-Kit-data/traces/tradesbench/seed5/gpt-6.1-sol-xhigh-a2/audit.md) · [Outcome index](../../Sova-Kit-data/traces/tradesbench/seed5/gpt-6.1-sol-xhigh-a2/outcomes.json) · [Events](../../Sova-Kit-data/traces/tradesbench/seed5/gpt-6.1-sol-xhigh-a2/events.json)

The GPT-6.1 Sol run has 151 bookings, 41 customer declines, 53 agent rejections, and 10 acknowledged cancellations. The 151 bookings ended with 130 completed appointments, 10 canceled, two company no-shows, and nine still scheduled. The four captured traces cover both no-shows, one completed booking, and one agent rejection. The simulator verifies the no-shows. A correction requires evidence linking the outcome to a specific decision.

Both missed appointments followed an earlier job on the same technician's schedule. The published booking logs contain the route distances before the agent chose the slots. The final database shows that the next arrival was scheduled 17 and 74 minutes after the cutoff. A booking rule that considers when the previous job is likely to finish and how long the drive will take is ready for a replay check once the pre-booking state is reconstructed. [Dispatch review](../../Sova-Kit-data/traces/tradesbench/seed5/gpt-6.1-sol-xhigh-a2/dispatch-review.md)

From Sova-Kit, recreate the bundle with `python3 SwitchIt/Traces/collect_tradesbench.py --database /path/to/run.db`. The collector verifies the database against its published SHA-256. [Collector](../SwitchIt/Traces/collect_tradesbench.py) · [Published simulator database](https://huggingface.co/datasets/adityabressl/trades-bench/resolve/0c3983bb519c9885a63f3085818ca516e3ec2067/runs/seed5/gpt-6.1-sol-xhigh-a2/run.db)

## State and acceptance checks

- [x] Identify public trace sources and verify counts from manifests and run reports.
- [x] Verify the existing SageMaker profile, Virginia S3 bucket, execution role, and training quota. Prepare and validate the local launch request for the synthetic smoke test.
- [x] Audit the published EnterpriseOps-Gym Teams run. Preserve the source references, task ID, displayed order, sanitized tool arguments and responses, and reported grade. Mark inferred links and missing state. Reproduce four outcome checks on recorded receipts.
- [x] Join selected TradesBench traces to final simulator outcomes. Retain the completed booking as a control candidate and the rejection as an observed disposition. Hold the two no-shows for correction review. Keep the Teams trace for importer development.
- [x] Audit the two bookings that ended in no-shows. Join the original choice and available route distance to the previous job, dispatch, and late arrival in simulator state.
- [ ] **Next work.** Reconstruct one pre-booking state and verify a better booking choice against the original action. Produce reviewed training rows with `trace_id`, the decision event, relevant tool receipts, desired action, observed outcome, and correction evidence. Keep separate verdicts for verified success, verified failure, execution error, and insufficient evidence.
- [ ] Pin a training manifest and an independent held-out manifest. Keep related turns, repeated task variants, and seed-matched trajectories together. Record artifact hashes, starting state, task and tool versions, model revision, prompt, decoder settings, and grader. Use the public sample for importer development and reserve fresh tasks for evaluation.
- [ ] Connect the curated manifests to the LoRA worker and confirm the base and adapter use the same replay contract. Freeze outcome checks before training. Record the exact AWS command, duration, cost estimate, and output location and obtain approval for that run.
- [ ] Capture base and adapter results on the same held-out cases. Review every changed verdict, any missing evidence, and the actual resulting state. Accept the candidate only if it repairs a verified baseline failure and passes the pinned regression checks.
- [ ] Archive the adapter checksum, input manifests, per-case receipts, result summary, and SageMaker billable time. Record the decision and the next task here.

## Prepared synthetic job

The locally prepared job `switchup-qwen15-lora-20261125-a1` uses 192 synthetic order-routing corrections, 24 held-out synthetic cases, and eight training regression cases. It tests the training and replay path. The real-trace corpus and its held-out suite are pending.

- Model · [Qwen2.5-1.5B-Instruct, pinned checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
- Update · Supervised rank-8 LoRA on the corrected actions with the base weights frozen
- [Local bundle](../../Sova-Kit-data/runs/sagemaker/switchup-qwen15-lora-20261125-a1/)
- Launch state · Approval pending. Only local bundle preparation, schema validation, and read-only AWS preflight checks have completed.
- Expected S3 outputs · Adapter weights and configuration, manifest, paired replay experiment, and report

See the [LoRA Route setup](LoRA-Route/README.md), [SwitchUp guide](docs/switchup.md), and [architecture diagram](docs/switchup-architecture.svg).

## Compute policy

The laptop prepares and audits artifacts. SageMaker runs training and model replays using the existing S3 setup. The prepared synthetic job requests one Virginia `ml.g5.xlarge`, a 45-minute runtime limit, and a two-hour capacity-wait limit. SageMaker releases the instance on job termination. Retain the evidence and artifacts in the job-specific S3 prefix. [Job limits](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)

The experiment budget is **$50 total**. The prepared synthetic smoke test is estimated at **$1–$2** based on the verified Virginia rate of $1.408 per instance hour and small storage and logging overhead. Estimate the real-trace job separately once the suite is fixed. Follow the ask-before-compute skill by presenting the exact command, machine, expected duration, estimated cost, and outputs for approval before each launch or retry. [Virginia price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json)
