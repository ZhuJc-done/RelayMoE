"""Small runtime, peer-copy, and optional native CUDA compilation checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import tempfile

import torch


def check(*, compile_extension: bool = False) -> None:
    assert torch.cuda.is_available(), "CUDA is unavailable"
    count = torch.cuda.device_count()
    packages = {}
    for name in ("torch", "numpy", "triton", "packaging", "pybind11"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    summary = {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "architectures": torch.cuda.get_arch_list(),
        "gpu_count": count,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)],
        "capabilities": [torch.cuda.get_device_capability(i) for i in range(count)],
    }
    # Tiny allocations on each GPU; this is a correctness check, not a benchmark.
    for i in range(count):
        values = torch.arange(1024, device=f"cuda:{i}", dtype=torch.float32)
        assert values.sum().item() == 523776.0
        matrix = torch.ones((32, 32), device=f"cuda:{i}", dtype=torch.bfloat16)
        assert torch.all(matrix @ matrix == 32).item()
    summary["all_gpu_fp32_bf16"] = "passed"
    summary["peer_access"] = {}
    source = torch.arange(1024, device="cuda:0", dtype=torch.float32)
    for target in (1, 4, 8):
        if target >= count:
            continue
        summary["peer_access"][f"0->{target}"] = torch.cuda.can_device_access_peer(0, target)
        copy = source.to(f"cuda:{target}")
        assert torch.equal(copy.cpu(), source.cpu())
    summary["cross_gpu_copy"] = "passed (copy correctness; not a transport-path measurement)"

    if compile_extension:
        from torch.utils.cpp_extension import load_inline

        cuda_source = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
__global__ void relaymoe_add_one_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.0f;
}
torch::Tensor relaymoe_add_one(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    if (x.numel()) {
        relaymoe_add_one_kernel<<<(x.numel()+255)/256, 256, 0,
            c10::cuda::getCurrentCUDAStream(x.get_device()).stream()>>>(
                x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}
'''
        with tempfile.TemporaryDirectory(prefix="relaymoe-cuda-") as build_dir:
            extension = load_inline(
                name="relaymoe_env_probe",
                cpp_sources="torch::Tensor relaymoe_add_one(torch::Tensor x);",
                cuda_sources=cuda_source,
                functions=["relaymoe_add_one"],
                extra_cuda_cflags=["-O2"],
                build_directory=build_dir,
                verbose=False,
            )
            assert torch.equal(extension.relaymoe_add_one(source), source + 1)
        summary["native_cuda_extension"] = "compiled and executed"
    print(json.dumps(summary, indent=2))
    subprocess.run(["nvcc", "--version"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    check(compile_extension=parser.parse_args().compile)
