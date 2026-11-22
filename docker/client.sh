#!/usr/bin/env bash
set -euo pipefail
# relay1 stores experiment images on data07 because its system Docker disk is full.
# An explicit Docker endpoint/context always takes precedence.
if [[ -z "${DOCKER_HOST:-}" && -z "${DOCKER_CONTEXT:-}" && -S /run/relaymoe-docker/docker.sock ]]; then
    export DOCKER_HOST=unix:///run/relaymoe-docker/docker.sock
fi
exec docker "$@"
