#!/usr/bin/env bash
set -euo pipefail
# Run on relay2 host. Attach work to the dedicated experiment container.
if [[ -f /.dockerenv ]]; then
    printf '请在 relay2 宿主机运行；当前已经在容器内。\n' >&2
    exit 1
fi
name="${RELAYMOE_EXPERIMENT_CONTAINER:-zjc_relaymoe_exp}"
if ! docker inspect "$name" >/dev/null 2>&1; then
    mkdir -p /data07/fy/relaymoe-models /data07/fy/relaymoe-cache
    docker run -d --init --name "$name" --gpus all --network host \
      --shm-size=8g --cpus=32 --memory=256g --memory-swap=256g --pids-limit=4096 \
      --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --ulimit nofile=65536:524288 \
      --mount type=bind,src=/data02/fy/relaymoe,dst=/data02/fy/relaymoe \
      --mount type=bind,src=/data07/fy/relaymoe-models,dst=/data07/fy/relaymoe-models \
      --mount type=bind,src=/data07/fy/relaymoe-cache,dst=/data07/fy/relaymoe-cache \
      --workdir /data02/fy/relaymoe \
      -e HF_HOME=/data07/fy/relaymoe-cache/huggingface \
      -e XDG_CACHE_HOME=/data07/fy/relaymoe-cache \
      -e TORCH_EXTENSIONS_DIR=/data07/fy/relaymoe-cache/torch_extensions \
      -e NCCL_IB_DISABLE=1 -e OMP_NUM_THREADS=1 zjc_relaymoe:latest sleep infinity
fi
args=(exec -i -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_NET_DISABLE_INTRA=1
      -e NCCL_NET_PLUGIN=none -e OMP_NUM_THREADS=1 -e MAX_JOBS=2 -e TORCH_CUDA_ARCH_LIST=12.0
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn)
if [[ $# == 0 ]]; then
    set -- bash
    if [[ -t 0 && -t 1 ]]; then args+=(-t); fi
fi
exec docker "${args[@]}" "$name" python3 scripts/with_limits.py "$@"
