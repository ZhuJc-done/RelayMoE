ARG BASE_IMAGE=maas-dev-cn-beijing.cr.volces.com/ml_maas_inference/vegrid-dev@sha256:2a67f8d93650f41f12c61454e0b1031b2034e6e2759ac20b14ee2d4f6697afc2
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="zjc_relaymoe"
LABEL org.opencontainers.image.description="RelayMoE development environment for relay2 TK300 GPUs"

ENV CUDA_HOME=/usr/local/cuda-13.1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/data02/fy/relaymoe \
    OMP_NUM_THREADS=1 \
    MAX_JOBS=4

WORKDIR /data02/fy/relaymoe
COPY pyproject.toml README.md ./
COPY relaymoe ./relaymoe
COPY examples ./examples
COPY tests ./tests
COPY scripts ./scripts
RUN python3 -m pip install --no-index --no-deps --no-build-isolation -e .

# GPU containers on relay2 receive the host render group (GID 105).
# Resolve its name so interactive Bash's `groups` call succeeds.
RUN if ! getent group 105 >/dev/null; then groupadd --gid 105 render; fi

ENTRYPOINT []
CMD ["/bin/bash"]
