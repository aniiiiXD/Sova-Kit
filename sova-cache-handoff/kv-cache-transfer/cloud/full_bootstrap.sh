#!/usr/bin/env bash
set -euo pipefail
cd /opt/ml/kvverify
export HF_HOME=/opt/ml/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Torch comes from the digest-pinned AWS 2.5.1 CUDA image. Never reinstall it.
python3 -m pip install --disable-pip-version-check -q \
  'numpy==1.26.4' 'transformers==4.52.4' \
  'huggingface-hub==0.36.2' 'safetensors==0.8.0'
: "${KV_WORKER_TIMEOUT:?Missing approved worker timeout}"
case "$KV_WORKER_TIMEOUT" in
  *[!0-9]*|'') echo 'Invalid worker timeout' >&2; exit 2 ;;
esac
if (( KV_WORKER_TIMEOUT < 300 || KV_WORKER_TIMEOUT > 6900 )); then
  echo 'Worker timeout exceeds configured bounds' >&2; exit 2
fi
exec timeout --signal=TERM --kill-after=60 "$KV_WORKER_TIMEOUT" python3 -u cloud/full_worker.py
