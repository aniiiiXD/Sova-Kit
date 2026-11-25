#!/usr/bin/env bash
set -euo pipefail
cd /opt/ml/kvverify
export HF_HOME=/opt/ml/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# Reuse the established AWS DLC/Torch installation. Do not install Torch.
python3 -m pip install --disable-pip-version-check -q \
  'numpy==1.26.4' 'transformers==4.52.4' \
  'huggingface-hub==0.36.2' 'safetensors==0.8.0'
exec timeout --signal=TERM --kill-after=30 1500 python3 -u cloud/worker.py
