# Implementation and smoke runbook

On 2026-11-18, **all 33 offline smoke tests passed**, including the five tiny random-model adapter checks (4.4 seconds). Evidence: `reports/smoke-v1/test-results.json`. This verifies software correctness checks, not pretrained-model speed or quality. The first bounded SageMaker pilot is tracked by its dated `reports/sova-kv-pilot-*/submission.json` and `status.json`; only collected model traces establish empirical results.

## Components

```text
kvverify/
  data.py          Strict MCQ records, document/context leakage checks, checkpoint hashes
  mathops.py       Tiny affine/RoPE reference math used by the offline smoke suite
  hf_backend.py    Real Qwen3 cache extraction, mapper calibration/import and greedy decode
  evidence.py      Timing/import invariants, paired statistics, durable reports
  runner.py        Five balanced arms, explicit warmup, raw error/attempt ledger
  smoke.py         Offline synthetic integration fixture, explicitly not model evidence
  cli.py           Guarded fit/evaluate, smoke and existing-trace analysis
scripts/smoke.py   Saves unit-test results and the fixture report
tests/
  test_core.py         Math, data isolation, accounting and reporting failures
  test_hf_optional.py  Opt-in random tiny Qwen3 CPU adapter tests
fixtures/              Synthetic calibration/eval examples, not an empirical dataset
```

## Approved-run boundary

The existing Python environment at `../sova-sys1/.venv/bin/python` has Torch 2.15.0, Transformers 4.52.4 and safetensors 0.8.0, confirmed by the successful smoke run. It was reused without dependency changes or checkpoint downloads. The adapter targets Transformers 4.52.4.

Proposed exact smoke command, working directory `Lab/kv-cache-transfer`:

```bash
../sova-sys1/.venv/bin/python scripts/smoke.py --with-torch --output reports/smoke-v1
```

Machine: laptop CPU only. Estimate: 10-30 seconds, no GPU, network, cloud bill, weights download or package installation. A few tiny random Qwen3 models are created in memory, not pretrained models. CUDA is hidden from this test process and HF is put in offline mode.

Output:

```text
reports/smoke-v1/
  tests.txt
  test-results.json
  integration/
    traces.jsonl
    summary.json
    report.md
```

The tiny-model tests exercise real Transformers forward/cache operations: identity-cache next-token logit parity, zero-cache sensitivity, calibration/serialization/checksum rejection and all five timed adapter arms. The dependency-free fixture tests use synthetic timings. Neither establishes speedup or pretrained-model accuracy. Reports explicitly say the thesis remains unverified.

If the command fails, preserve its report. A changed command or rerun needs its own approval under the local compute skill. Do not substitute another environment or install dependencies silently.

## Real model path

The first mechanism reproduction supports **dense, full-attention Qwen3 only**. A Qwen3-0.6B -> Qwen3-1.7B pair has matching 8 KV heads and head dimension 128, according to their official configs. Larger matching dense Qwen3 pairs can follow the same adapter after a memory check. Qwen3.5, Qwen3.8 hybrids, MoE, sliding-window attention and scaled/dynamic RoPE are rejected. A hybrid-state implementation is required to test OneTriangle's exact production pair.

The initial mapper is an explicit **depth-aligned, per-head affine ridge baseline (k=1)**. It undoes source RoPE, fits centered ridge maps for keys/values on independent calibration documents, then restores target RoPE. This is not top-k layer selection, nonlinear mapping, or OneTriangle's task-trained residual mapper. A negative result on this baseline does not disprove all cross-model transfer methods.

Model run commands are guarded by `--execute`, require a CUDA worker and use local safetensors directories. They cannot download models, execute remote model code, start servers or provision GPUs. Both checkpoint weights and tokenizer files are hashed. Source and target tokenizers must match, and every evaluation prompt's token IDs are compared.

Examples below are templates for a later separately scoped/approved worker run. Paths must refer to actual downloaded and reviewed artifacts. Do not execute on the laptop:

```bash
python -m kvverify fit \
  --source models/source-snapshot --target models/target-snapshot \
  --data datasets/calibration.jsonl --prefix-tokens 1024 \
  --output artifacts/ridge-v1 --device cuda:0 --execute

python -m kvverify evaluate \
  --source models/source-snapshot --target models/target-snapshot \
  --mapper artifacts/ridge-v1 --data datasets/heldout.jsonl \
  --output reports/heldout-v1 --device cuda:0 --execute
```

Omitting `--execute` prints a plan without importing Torch or loading a model. Output directories must be new; existing reports/artifacts cannot be overwritten.

The user authorized the first bounded GPU pilot on 2026-11-18. `cloud/submit.py --submit` packages only whitelisted code and existing ContractNLI train/dev files, checks the account and smoke result, fetches the public Ohio training price, and creates one `ml.g5.2xlarge` training job with a 1,800-second runtime limit. The public rate observed was $1.515/hour (about $0.76 for 30 worker minutes, plus storage, transfer, billing overhead and tax). `cloud/bootstrap.sh` preserves the DLC's Torch installation and pins the adapter's other dependencies; it also limits the worker process to 1,500 seconds. No persistent endpoint is created.

The pilot uses pinned Apache-2.0 Qwen3-0.6B and Qwen3-1.7B checkpoints, FP16, 12 training documents (1,024-token calibration prefixes), and 12 distinct development documents with four questions per gold class. Full evaluation prompts up to 8,192 tokens are eligible; excluded candidates are logged, never silently truncated. This deliberately small, label-balanced pilot is not a representative quality benchmark. It runs pretrained identity/zero-cache controls first, then fits and reloads the mapper and measures all five arms twice. Raw traces, mapper, selection manifest and environment metadata go to S3. Both weights remain resident, so reported native-arm memory includes the source model and is not a standalone native-serving cost comparison.

Collect status and existing results without launching more compute:

```bash
python3 cloud/collect.py --receipt reports/ACTUAL-JOB-NAME/submission.json --logs
```

The collector records SageMaker-reported billable seconds when available and estimates compute cost from the public rate. That estimate is not an invoice. Failed jobs are not retried automatically.

## Evidence and current limits

- Five arms: source, native cold, native warm, cold transfer, warm-source transfer. Cold runs construct fresh caches. Warm setup time is recorded and explicitly excluded from warm handoff timing.
- Transfer must import exactly `prompt_tokens - 1` cached positions, then recompute exactly one target tail token. Import/mapping failure becomes an error, not a hidden full-prefill success.
- The mapper stage currently includes any device move and DynamicCache construction. It is labeled as such; `import_ms=0` does not mean that movement/import was free.
- Mapper tensors are placed on the target device before measured evaluation. Warmups are saved separately; order is shuffled deterministically. Both model weights remain resident for all arms.
- In-process TTFT is synchronized model execution through first token selection. It is not HTTP/network latency, streaming TTFT, or time to a visible answer after hidden reasoning.
- The reference HF implementation is not a production vLLM/SGLang benchmark. It has batch size one, greedy decoding and no inter-request native prefix-cache service.
- Peak CUDA allocated/reserved memory is captured. GPU billing, GPU-seconds and dollar savings are **not implemented/measured**; reports cannot declare cost acceptance.
- MCQ scoring is strict: the decoded answer must be exactly a choice letter. Malformed, missing and failed responses are wrong. Repeated timing attempts do not multiply quality sample size.
- Paired document-cluster intervals are exploratory. Reports never automatically establish noninferiority or the thesis from tiny samples or synthetic fixtures.
- Calibration uses explicit prefix chunks. Evaluation never silently truncates; over-limit examples remain counted as such.
- Checkpoint, tokenizer and mapper checksum mismatches fail closed. Calibration document IDs, example IDs and normalized exact context hashes are checked against held-out examples before model loading.
- Exact/whitespace duplicate checks do not catch every near-duplicate or pretraining contamination. Near-duplicate auditing is still required for a substantive quality dataset.

## Compute policy

Per the local ask-before-compute skill, show command, machine, duration, cost and output, and obtain approval before each smoke/model run. No dependency changes, API calls, cloud deployments or GPU use are part of the proposed offline smoke. Before a real GPU run, freeze data/checkpoint hashes, obtain a current quote, agree a runtime budget, collect traces and tear down any billed capacity.

## API references inspected

- [Transformers Qwen3 implementation, runtime version 4.52.4](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen3/modeling_qwen3.py) (inspected alongside 4.57.1; runtime must still pass its tiny adapter smoke).
- [Transformers DynamicCache source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)
- [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Qwen3-1.7B config](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)
