#!/usr/bin/env bash
set -euo pipefail
cd /opt/ml/kvverify
export HF_HOME=/opt/ml/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python3 -m pip install --disable-pip-version-check -q 'numpy==1.26.4' 'transformers==4.52.4' 'huggingface-hub==0.36.2' 'safetensors==0.8.0'
case "${KV_MODE:?}" in probe|repair|precision|eval_resume|state_diagnostic|linear_search|continuation|holdout|prefix_probe|layer_readout|matched_fit|harvey_pilot|ttft|answer_train|paired_replay|readout_train|answer_continuation|mapper_ablation) ;; *) exit 2 ;; esac
exec timeout --signal=TERM --kill-after=20 "${KV_WORKER_TIMEOUT:?}" python3 -u "cloud/${KV_MODE}_worker.py"
