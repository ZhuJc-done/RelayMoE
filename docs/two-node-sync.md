# relay1 / relay2 同步与进入环境

日期：2026-11-22。用户已授权在 relay2 创建 `zjc/test_real_moe` 并 push 到 origin，
relay1 从 origin 拉取。模型、镜像、缓存和原始实验数据通过两机内网另行同步。
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

同步完成后记录：源码/模型/缓存 SHA256、镜像内容标识、Git HEAD、容器挂载、
Python/PyTorch/CUDA/NCCL/vLLM 版本与 GPU/RDMA 可见性。
详见 `artifacts/two-node-32gpu/setup/` 中的同步验收文件。

2026-11-22 11:29 初始快照：relay2 有16个其他 Python GPU 进程；relay1 无可见
compute进程，但15张卡利用率报告100%、显存仅3–19MiB。此记录不是资源预约或
可运行结论，需要启动前复核。此轮只做文件/镜像同步和轻量环境验收，不启动32卡性能实验。
