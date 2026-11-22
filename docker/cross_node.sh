#!/usr/bin/env bash
set -euo pipefail
if [[ -f /.dockerenv ]]; then
    printf '请在宿主机执行；容器内直接运行开发命令。\n' >&2
    exit 1
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
client=(bash "$script_dir/client.sh")
name="${RELAYMOE_CROSS_CONTAINER:-zjc_relaymoe_cross}"
image="${RELAYMOE_IMAGE:-zjc_relaymoe:latest}"
workspace=/data02/fy/relaymoe
[[ -d "$workspace/relaymoe" && -d /data07/fy/relaymoe-models && -d /data07/fy/relaymoe-cache ]] || {
    printf '代码、模型或缓存目录缺失，请先同步。\n' >&2; exit 1;
}
[[ -d /dev/infiniband ]] || { printf 'RDMA 设备目录缺失。\n' >&2; exit 1; }
if ! "${client[@]}" inspect "$name" >/dev/null 2>&1; then
    "${client[@]}" run -d --init --name "$name" --label relaymoe.role=cross-node \
      --gpus all --network host --device /dev/infiniband \
      --shm-size=8g --cpus=32 --memory=256g --memory-swap=256g --pids-limit=4096 \
      --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --ulimit nofile=65536:524288 \
      --mount "type=bind,src=$workspace,dst=$workspace" \
      --mount type=bind,src=/data07/fy/relaymoe-models,dst=/data07/fy/relaymoe-models,readonly \
      --mount type=bind,src=/data07/fy/relaymoe-cache,dst=/data07/fy/relaymoe-cache \
      --workdir "$workspace" \
      -e HF_HOME=/data07/fy/relaymoe-cache/huggingface \
      -e XDG_CACHE_HOME=/data07/fy/relaymoe-cache \
      -e TORCH_EXTENSIONS_DIR=/data07/fy/relaymoe-cache/torch_extensions \
      -e TRITON_CACHE_DIR=/data07/fy/relaymoe-cache/triton \
      -e NCCL_IB_DISABLE=0 -e OMP_NUM_THREADS=1 -e MAX_JOBS=2 \
      -e TORCH_CUDA_ARCH_LIST=12.0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "$image" sleep infinity
else
    role="$("${client[@]}" inspect -f '{{ index .Config.Labels "relaymoe.role" }}' "$name")"
    [[ "$role" == cross-node ]] || { printf '同名容器不属于本启动器：%s\n' "$name" >&2; exit 1; }
    if [[ "$("${client[@]}" inspect -f '{{.State.Running}}' "$name")" != true ]]; then
        "${client[@]}" start "$name" >/dev/null
    fi
fi
args=(exec -i)
if [[ $# == 0 ]]; then
    set -- bash
    if [[ -t 0 && -t 1 ]]; then args+=(-t); fi
fi
exec "${client[@]}" "${args[@]}" "$name" python3 scripts/with_limits.py "$@"
