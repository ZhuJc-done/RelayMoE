#!/usr/bin/env bash
set -euo pipefail
# Run inside zjc_relaymoe_exp. The base CUDA/PyTorch installation stays intact.
venv=/data07/fy/relaymoe-cache/vllm-venv
python3 -m venv --system-site-packages "$venv"
env -u PIP_CONSTRAINT "$venv/bin/pip" install 'vllm==0.20.0' 'transformers==4.57.6'
"$venv/bin/python" -c 'import torch, vllm, transformers, vllm._C; print(torch.__version__, vllm.__version__, transformers.__version__)'
"$venv/bin/pip" freeze > /data07/fy/relaymoe-cache/vllm-freeze.txt
