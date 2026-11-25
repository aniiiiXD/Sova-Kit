# SwitchUp LoRA Route on SageMaker

This pilot trains a rank-8 LoRA on the frozen Qwen2.5-1.5B-Instruct checkpoint. The training examples are 192 synthetic order-routing corrections. The base and adapted models receive identical instructions at replay time. Both run SwitchIt's paired evaluator with 24 held-out cases and eight training regression cases. A synthetic `cancel_order` changes a copied order state, and each case checks the action and resulting state.

The adapter objective is supervised training on the corrected actions. The run records the adapter's SHA-256 after a save and reload. The report counts repaired and regressed cases and records whether the candidate cleared the replay gate. SwitchUP's connected request route currently activates instruction revisions. This experiment stores the weights and replay evidence for a later serving integration.

## Launch bundle

From Sova-Kit, prepare a local bundle with `python3 SwitchUp/LoRA-Route/prepare.py`. Supply an existing Virginia profile, bucket, execution role ARN, and an unused SageMaker job name. `--prefix` defines the path for this pilot inside the bucket. The bundle is written to `Sova-Kit-data/runs/sagemaker/<job-name>/` alongside the kit and includes the [retraining worker](../Retraining/worker.py). Set `SOVA_KIT_DATA_DIR` to use a different data root. The generated `training-job.json` pins the existing AWS image digest, model revision, and job limits. `launch.sh` stages the synthetic dataset and replay code, then submits one training job.

The job writes `adapter_model.safetensors`, `adapter_config.json`, `report.json`, `experiment.json`, and the input manifest into SageMaker's S3 `model.tar.gz` output. Preserve the job description with the report to record billable time.

## Compute policy

The laptop builds the bundle, uploads small input files, and reads job metadata. One Virginia `ml.g5.xlarge` does the training and paired replay. The request limits runtime to 45 minutes and capacity wait to two hours, the minimum accelerated capacity-wait setting accepted by the AWS CLI. SageMaker releases the training instance at termination. The launch bundle uses the existing S3 bucket with a job-specific prefix for input and retained results.

At $1.408 per hour, 45 minutes is $1.056 of instance time. Allow roughly $1–$2 for the first run including small storage, logs, and setup or output overhead. Queue time and time billed by AWS can differ from the runtime setting. One launch and any retry each require approval of the exact command, machine, duration, cost estimate, and output under the user's ask-before-compute skill. [AWS training time limit](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html) · [AWS Virginia pricing](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json) · [Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
