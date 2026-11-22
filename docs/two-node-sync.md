# relay1 / relay2 同步与进入环境

日期：2026-11-22。已在 relay2 创建 `zjc/test_real_moe` 并 push 到 origin，
relay1 已从 origin 拉取。初始实现提交为 `3d8eb46672bc580ec4089cf033192f3e4ad7af5a`；
本文件的后续验收记录随同一分支提交。模型、镜像、缓存和原始实验数据已通过两机内网同步。
Git 不承载权重、镜像层或大体积原始结果；`artifacts/`、`third_party/` 被忽略。
固定第三方版本：DeepEP `95acda3a8d3c8a278fdfca293473a7f1887f593b`。

| 内容 | 两机相同路径 |
| --- | --- |
| Git 工作区与实验结果 | `/data02/fy/relaymoe` |
| Qwen3-30B-A3B | `/data07/fy/relaymoe-models/Qwen3-30B-A3B` |
| vLLM 虚拟环境 | `/data07/fy/relaymoe-cache/vllm-venv` |
| JIT/运行缓存 | `/data07/fy/relaymoe-cache` |
| 同步与校验记录 | `/data07/fy/relaymoe-sync/20261122` |

## Docker 存储

relay1 的系统 `/var/lib/docker` 位于已满的 data00 分区。为避免改动现有容器，
RelayMoE 镜像导入独立 Docker daemon：

- data-root：`/data07/fy/relaymoe-docker/data`
- socket：`unix:///run/relaymoe-docker/docker.sock`
- config/log：`/data07/fy/relaymoe-docker/daemon.json`、`dockerd.log`
- exec-root/PID：`/run/relaymoe-docker/exec`、`dockerd.pid`
- containerd namespaces：`relaymoe`、`plugins.relaymoe`
- bridge、iptables、ip6tables、IP forwarding、masquerade 均禁用；实验容器用 host 网络。

系统 Docker 配置未更改，未重启其 daemon，也未清理其他用户镜像或容器。
独立 daemon 未注册开机自启动；重启机器后可在 relay1 宿主机运行：

```bash
cd /data02/fy/relaymoe
bash docker/start_private_daemon.sh
```

两机统一使用包装入口，relay1 自动选择独立 socket，relay2 使用原 Docker：

```bash
cd /data02/fy/relaymoe
bash docker/client.sh images zjc_relaymoe
bash docker/client.sh ps --filter name=zjc_relaymoe
bash docker/cross_node.sh
```

`cross_node.sh` 创建/进入 `zjc_relaymoe_cross`：同代码目录、模型只读挂载、缓存
读写挂载、RDMA 设备透传、host 网络。容器默认只执行 sleep；进入 shell 并不
启动实验。退出交互 shell 不会停止容器。已经在容器内时直接执行 Python。

原单机 `experiment.sh` 仍保留其关闭 IB 的配置，旧数据复现使用旧入口。
`cross_node.sh` 只是环境入口，32-rank RelayMoE backend 尚需按实验计划实现。

## 后续代码同步

relay2 修改、验证并提交后 push，再由 relay1 拉取：

```bash
# relay2，先查看工作区；仅提交所需代码/文档
git switch zjc/test_real_moe
git push origin zjc/test_real_moe

# relay1，确保没有未处理的工作区修改
git switch zjc/test_real_moe
git pull --ff-only origin zjc/test_real_moe
```

两机镜像内容与 vLLM venv 必须一起保持一致。只拉 Git 不会更新镜像、权重、
编译缓存或 pip 依赖。跨机计量前停止文件传输并再次确认 GPU 与 NIC 使用情况。

## 校验与实验状态

同步验收通过：

- 26 个模型文件合计 61,084,198,898 bytes（56.89 GiB），两机逐文件 SHA256 一致，
  同时匹配原始下载 manifest。
- 76 个 Git 文件、1,360 个单机实验产物、51,970 个缓存文件/符号链接、106 个
  第三方文件的内容/链接目标一致。缓存含 vLLM venv；GPU JIT 后续可能生成新缓存。
- Python 3.11.13、PyTorch 2.11.0+cu130、CUDA runtime 13.0、nvcc 13.1.115、
  NCCL 2.28.9 一致；venv 为 vLLM 0.20.0、Transformers 4.57.6。
  两机成功 import torch、vllm、vllm._C、transformers 和工作区 relaymoe。
- 新容器 `zjc_relaymoe_cross` 已在两机启动，挂载路径、RDMA设备、GPU可见性一致，
  容器仅有 docker-init 和 sleep。23项CPU单元测试通过；本轮没有启动GPU计算基准。

镜像存储方式不同，Docker 显示 ID 的层级不同：

| 标识 | 摘要 |
| --- | --- |
| relay2 OCI index | `sha256:faacddda8151aa8f9c29c90f87172e1c425ea0bd0369563c31b30b24226b59ba` |
| index 中 amd64 manifest | `sha256:86113c515108cefce1f07e1ac53dd09cbadf53da5d5a9e340fd89de44c56fbe4` |
| 该 manifest 的 config / relay1 image ID | `sha256:797faab7f6ea79c6332d352a01850bf8f3e29388fa62d0e233f5ebb18ed679ee` |

已从 relay2 containerd 内容库核对 index→manifest→config 链；两机121个RootFS层、
运行Config、架构与创建时间均一致。不是重新构建出的不同运行环境。

详细同步验收文件位于两机及本地 `artifacts/two-node-32gpu/setup/`。
原始校验清单位于两机 `/data07/fy/relaymoe-sync/20261122`；文件直传未删除源文件。

2026-11-22 11:29 初始快照：relay2 有16个其他 Python GPU 进程；relay1 无可见
compute进程，但15张卡利用率报告100%、显存仅3–19MiB。此记录不是资源预约或
可运行结论，需要启动前复核。11:37终检仍见relay2的16个外部GPU进程，采样显存约
44–56 GiB、利用率99%；relay1依然无可见compute进程而大多数卡利用率报告100%。
此轮只做文件/镜像同步和轻量环境验收，不启动32卡性能实验，也不重置GPU或停止外部进程。
