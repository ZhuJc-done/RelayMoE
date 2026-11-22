# relay2 开发环境

2026-11-21 在 relay2 构建并验证：

- `zjc_relaymoe:latest`

容器工作目录和宿主机代码目录均为 `/data02/fy/relaymoe`。

## 进入环境

`docker/run.sh` 在 **relay2 宿主机**上新建并进入容器：

```bash
cd /data02/fy/relaymoe
bash docker/run.sh
```

容器已经运行时，在宿主机另一个终端进入它：

```bash
docker exec -it -w /data02/fy/relaymoe zjc_relaymoe_dev bash
```

如果已经在容器内，直接运行开发命令，无需再次运行 `docker/run.sh`。
可执行 `test -f /.dockerenv && echo '当前在容器内'` 确认；宿主机与容器的
提示符可能相同，不能只根据 `root@...` 判断。

启动脚本开放全部 16 张 GPU，将代码目录读写挂载到容器，并使用 8 GiB
共享内存。容器默认名为 `zjc_relaymoe_dev`，首次启动的主 shell 退出后，
容器停止并自动删除；通过 `docker exec` 打开的额外 shell 退出不会停止容器。
代码修改保留在
宿主机。容器内临时安装的软件不会保留，固定依赖应写入 Dockerfile 后重建。

relay2 的默认 Docker bridge 启动失败（docker0 不可用），脚本默认使用
host 网络。可通过 `RELAYMOE_NETWORK` 更改网络模式；纯离线单进程检查可设为
`none`。网络与 RDMA 设备透传分别控制。

relay2 的 GPU 容器进程包含宿主机 `render` 补充组（GID 105）。Dockerfile
在镜像内补齐该组名，避免交互式 Bash 启动时出现
`groups: cannot find name for group ID 105`。该提示本身不会阻止容器启动。

运行独立命令或透传 RDMA 设备：

```bash
bash docker/run.sh python3 -m unittest discover -s tests -v
RELAYMOE_RDMA=1 bash docker/run.sh
```

`RELAYMOE_IMAGE`、`RELAYMOE_WORKSPACE`、`RELAYMOE_CONTAINER` 可覆盖镜像、
宿主机目录和容器名。多容器并行运行时使用不同容器名。

## 实测环境

| 项目 | 版本/配置 |
| --- | --- |
| 宿主系统 | Debian 12，kernel 5.15.152.bsk.11-amd64 |
| 宿主 Python | 3.11.2；默认环境无 PyTorch、无 nvcc |
| GPU | 16 × TK300 72G，每卡 73415 MiB，compute capability 12.0 |
| 宿主 NVIDIA 驱动 | 580.173.02 |
| Docker / NVIDIA Container Toolkit | 29.7.2 / 1.20.0 |
| 容器系统 | Ubuntu 24.04.3 |
| 容器 Python | 3.11.13 |
| PyTorch | 2.11.0+cu130，包含 sm_120 支持 |
| PyTorch CUDA 构建版本 | 13.0 |
| CUDA Toolkit / nvcc | 13.1 / 13.1.115 |
| NCCL | 2.28.9 |
| GCC / CMake / Ninja | 13.3.0 / 3.31.6 / 1.13.0.git.kitware.jobserver-pipe-1 |
| NumPy / Triton | 1.25.0 / 3.6.0 |

PyTorch 使用 CUDA 13.0 构建，镜像中的编译工具链是 CUDA 13.1。
已实际编译并执行一个 `sm_120` CUDA 扩展，确认当前组合可用于这一开发检查；
新增通信后端仍需分别编译和验证其依赖。

GPU 0–3、4–7、8–11、12–15 分别位于四个 PIX 域。
GPU 0–7 属于 NUMA 0，8–15 属于 NUMA 1；跨 PIX 域的连接在同 NUMA 内为
NODE，跨 NUMA 为 SYS。后续实验应区分这些路径，不能将整个节点建模为均匀 PCIe。
宿主机 P2P read 查询显示所有 GPU 对可用；本次复制检查抽查了 0→1、0→4、0→8。

宿主机可见 9 个 ACTIVE Ethernet/RoCE RDMA 端口：`mlx5_0` 为 200 Gb/s，
`mlx5_1`–`mlx5_8` 为 400 Gb/s。镜像提供 libibverbs、libmlx5 与相关诊断工具。
跨节点性能和 GPU Direct RDMA 数据路径尚未验证。

## 验证

以下检查已在新镜像内通过：

- 全部 16 卡的小规模 FP32 求和、BF16 矩阵乘法。
- 0→1、0→4、0→8 跨 GPU 复制结果正确（不代表已测量实际传输路径）。
- 原生 `sm_120` CUDA 扩展编译、加载和执行。
- 16 进程 NCCL all-reduce 与环形 send/recv；测试时禁用 IB。
- 仓库现有 23 项单元测试全部通过。

复现命令：

```bash
cd /data02/fy/relaymoe
bash docker/run.sh python3 scripts/check_cuda.py --compile
bash docker/run.sh env NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 \
  torchrun --standalone --nproc-per-node=16 scripts/check_nccl.py
bash docker/run.sh python3 -m unittest discover -s tests -v
```

这些都是小规模正确性检查，不是 RelayMoE 的实机性能实验。
当前仓库仍提供 planner、simulator 与 backend 接口；实际 CUDA/RDMA 执行后端
需要后续实现。镜像尚未安装 DeepEP、SGLang、vLLM。

## 构建

复用 relay2 已有基础镜像的共享层，固定其 digest：

```text
maas-dev-cn-beijing.cr.volces.com/ml_maas_inference/vegrid-dev@sha256:2a67f8d93650f41f12c61454e0b1031b2034e6e2759ac20b14ee2d4f6697afc2
```

```bash
cd /data02/fy/relaymoe
docker build --pull=false --network=none --progress=plain \
  -f docker/Dockerfile \
  -t zjc_relaymoe:latest .
```

基础镜像需要内部镜像仓库访问权限或本地缓存。`--network=none` 禁止构建 RUN
步骤联网，基础镜像元数据解析仍可能访问仓库。项目使用 editable install，
挂载后可直接导入修改后的本地代码。

检查时 `/data02` 与 Docker 存储所在文件系统均为 98% 使用率，分别约剩
68 GiB 和 80 GiB。当前镜像复用已有层；后续安装大型依赖前需检查可用空间。
