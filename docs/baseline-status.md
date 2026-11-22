# 单机对照实现与范围

用户最新指示（2026-11-21）：本阶段只完成单机 16 卡性能；采用仓库中的
CollWay-like 算法即可，无须复现论文完整工程；跨节点由后续新 goal 启动。
完整 logits 对比暂缓，保留已产生的数据，不以 token 一致宣称数值等价。

## 正式对照

| 标签 | 算法 | 真实数据传输 | 范围 |
| --- | --- | --- | --- |
| NCCL all-to-all-v | GPU 目标合并、按 peer 聚合 | `torch.distributed.all_to_all_single` | 真实 NCCL；动态 prepare 计入 |
| CollWay-like | 仓库 `CollWayBaseline`：direct + GPU 收发端点独占的贪心阶段；Combine 反向阶段 | CUDA IPC | 仓库算法的 C++ 实现；不代表原论文完整 chunk/path 工程 |
| Direct IPC | direct 星形发送 | 与 RelayMoE 相同 CUDA IPC 执行器 | 中继消融 |
| RelayMoE | 二项树候选、跨组负载感知、按依赖层批量执行 | CUDA IPC | v9 每层设备同步及 CPU 屏障，CPU 紧凑规划，无整计划缓存 |
| Independent tree | 同候选，每组独立选择 | 同上 | 负载感知消融 |

CollWay-like 8 个 seeds 的阶段成员已逐项核对，与 Python 参考一致；16 卡六种
输入 BF16 专家 GEMM 的 correctness 已通过。后续执行器改动需重新验证。

## DeepEP

本轮不把跨节点转发数据混入单机结论。仓库 `DeepEPBaseline` 对同一节点的
所有目标直接发送，单机逻辑树因此等同 direct；不应把 NUMA 域冒充服务器。

已获取官方 `hybrid-ep` 代码，revision
`95acda3a8d3c8a278fdfca293473a7f1887f593b`，位于服务器 `third_party/DeepEP`。
其 PCIe 文档描述所有通信经 RDMA NIC（包括节点内），不是本轮纯 PCIe P2P
配置。当前实验容器没有 `/dev/infiniband`，未安装/运行其真实通信基准。
relay2 宿主机有 8 个 Active 400Gb/s RDMA 端口及 1 个 200Gb/s 端口；实际
GPU Direct RDMA 可用性未验证。跨节点待下一阶段固定版本、NIC/rank 映射和
公平资源条件后比较，不将 repo 的 DeepEP-like 仿真冒充官方实现。

## 计量

### 当前Qwen3原生模型路径（已检查服务器vLLM 0.20.0源码）

TP16/DP1/EP16，sequence parallel关闭；进入MoE时token hidden states在TP ranks
间已经复制。`fused_moe/config.py:1008`的`use_all2all_kernels`要求DP>1且EP，
本轮条件不成立；`all2all_utils.py:110`选择`NoDPEP` prepare/finalize。
`prepare_finalize/no_dp_ep.py:56`不跨卡dispatch，finalize进行本地专家加权归并；
`runner/moe_runner.py:357`在输出尚未跨rank归约时调用TP AllReduce。
因此配置中的`allgather_reducescatter`不表示本轮原生实际执行AG/RS dispatch。

NCCL/Relay模型适配器主动把源token分片，运行稀疏dispatch/experts/combine，
再allgather恢复TP复制布局。该适配器结果与原生vLLM必须分别标注布局，原生
端到端更快不能等同于原生稀疏dispatch更快。当前关闭custom allreduce；更底层
allreduce具体kernel选择需runtime trace确认，不能只凭配置名断言ring/tree。

NCCL适配器按peer打包调用all_to_all_single；CollWay-like按源/目的集合的flow
逐动作执行cudaMemcpyAsync与event，阶段内每rank至多一个send及一个receive，
不同角色可以同时存在。两者均direct，无中继。CollWay-like与Relay batch对比
包含执行器效率差异，不能把全部差距归因于树算法；Direct/Independent batch
作为相同执行器消融单独报告。用户要求优先优化Relay，baseline保持已验证配置。

- CPU 规划：分组、候选选择/绑定、DAG 与 buffer 布局分别计时。
- 完整 prepare：额外包括元数据 all-gather、GPU→CPU、索引/缓冲构造。
- 100 步样本每次取最慢 rank，方法交替执行。
- CUDA timeline 包含主机提交间隙和必要同步；不称为纯 kernel busy time。
- 模型推算 cut bytes 与真实测量的时间分别标注。

## 核对来源

原文 revision 597 的参考文献给出：
`https://arxiv.org/abs/2609.34340`，
*CollWay: Contention-Aware Scheduling for AlltoAllv on PCIe GPU Systems*。
公开搜索未取得可核验的代码，当前采用用户指定的仓库算法范围。
DeepEP 官方代码及 `docs/README_PCIe.md` 通过 GitHub 直接读取并固定上述 revision。
