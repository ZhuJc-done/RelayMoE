#!/usr/bin/env bash
set -euo pipefail

# This launcher runs on the host, not inside the development container.
if [[ -f /.dockerenv ]]; then
    printf '%s\n' \
        '当前已经在容器内，请直接运行 python3 等开发命令。' \
        'docker/run.sh 用于在宿主机上新建并进入容器；不需要在容器内重复执行。' >&2
    exit 1
fi

docker_cmd="$(type -P docker || true)"
if [[ -z "$docker_cmd" || ! -f "$docker_cmd" || ! -x "$docker_cmd" ]]; then
    printf '未找到 Docker 可执行文件，请在 relay2 宿主机上运行此脚本。\n' >&2
    exit 1
fi

image="${RELAYMOE_IMAGE:-zjc_relaymoe:latest}"
workspace="${RELAYMOE_WORKSPACE:-/data02/fy/relaymoe}"
container_name="${RELAYMOE_CONTAINER:-zjc_relaymoe_dev}"

if [[ ! -d "$workspace/relaymoe" ]]; then
    printf 'RelayMoE checkout not found: %s\n' "$workspace" >&2
    exit 1
fi

# relay2 has no usable docker0 bridge; use host networking by default.
args=(run --rm --init --name "$container_name" --gpus all
    --network "${RELAYMOE_NETWORK:-host}"
    --shm-size=8g --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
    --mount "type=bind,src=$workspace,dst=/data02/fy/relaymoe"
    --workdir /data02/fy/relaymoe)
if [[ -t 0 && -t 1 ]]; then
    args+=(-it)
else
    args+=(-i)
fi

# RDMA is optional for the single-node PCIe development environment.
# Enable this on both servers when validating the cross-node backend.
if [[ "${RELAYMOE_RDMA:-0}" == 1 ]]; then
    if [[ ! -d /dev/infiniband ]]; then
        printf 'RDMA requested but /dev/infiniband is absent\n' >&2
        exit 1
    fi
    args+=(--device /dev/infiniband)
fi

if [[ $# == 0 ]]; then
    set -- /bin/bash
fi
exec "$docker_cmd" "${args[@]}" "$image" "$@"
