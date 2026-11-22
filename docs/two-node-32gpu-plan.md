# 双机 2×16 GPU 实验计划

日期：2026-11-22。当前阶段先同步环境、提交代码与制定计划，尚未启动跨机 GPU 实验。
单机 v9 结果保持冻结；本阶段实现与结果使用单独版本及 `artifacts/two-node-32gpu/`。

## 实验目标与实现起点

在 relay1、relay2 的 32 张 TK300 上测量完整 MoE 通信时延、有效带宽和跨机流量，
判断中继减少源注入/跨机重复传输的收益能否覆盖在线规划、转发及同步开销。
最终以相同输入下的实测结果判断，不预设 RelayMoE、CollWay-like、NCCL 的排序。

当前 v9 CUDA IPC 后端只支持单机：`CompactPlanner` 限制 world≤16，使用
`world*(1<<world)` 的模式数组，executor 使用本机 `/dev/shm` 和 CUDA IPC，
vLLM adapter 也显式要求 EP16。不能仅把 torchrun 的 world size 改为 32。

需先实现：

- 显式 node/local-rank/NUMA/PCIe switch/NIC 映射；本机 IPC 与跨机网络分开执行。
- 32-rank 规划采用稀疏模式表和分层/按需候选，避免枚举 2^32 目的掩码。
  全局 mask 使用明确的无符号类型，扩展固定长度数组并校验 Combine 逆向依赖。
- 跨机初版可用 NCCL P2P 承载分阶段中继边，节点内复用 batch IPC；随后检查
  同步与提交开销。调度算法与 transport 分别标注，不能将这种实现称为原生 RDMA kernel。
- 全局控制元数据交换、GPU→CPU、buffer 构造、远端 ingress 选择和必要同步均计时。
  离线 catalog 构建单列；整计划缓存保持关闭。

## 已确认的硬件与准备

| 项目 | relay1 | relay2 |
| --- | --- | --- |
| GPU | 16×TK300 72G | 16×TK300 72G |
| 驱动 | 580.173.02 | 580.173.02 |
| 管理网 IP | 9.53.62.51 | 9.53.62.52 |
| RDMA | mlx5_1–8，各 400 Gb/s；mlx5_0 为 200 Gb/s | 同左 |
| 代码 | `/data02/fy/relaymoe` | 同左 |
| 模型 | `/data07/fy/relaymoe-models/Qwen3-30B-A3B` | 同左 |
| vLLM 环境/缓存 | `/data07/fy/relaymoe-cache` | 同左 |

端口 Active、驱动模块存在以及 SSH 可达，只是准备条件；尚未证明两机 RoCE/GDR
数据路径可用。实验应使用经过校准的相同 NIC 集合，不将管理网 eth0 混入主数据面。
relay1 的系统 Docker 盘已满，实验镜像使用 `/data07/fy/relaymoe-docker` 的独立
daemon；启动方法见 `two-node-sync.md`。两机新容器均用 `docker/cross_node.sh`。
历史单机 `docker/experiment.sh` 强制关闭 IB，不能用于正式跨机测量。

## 执行顺序

| 顺序 | 内容 | 最小配置和验收 |
| --- | --- | --- |
| 0 | 环境、资源、网络验收 | 镜像实际配置/层摘要、源码、模型 SHA 一致；两机无外部 GPU 作业；32 rank 可连接；GPU↔NIC 拓扑与端口日志归档 |
| 1 | 链路校准 | 每机 1 GPU 起步；4 KiB/64 KiB/1 MiB/16 MiB 单向与双向；单 NIC，再 8 NIC 并发；确认 RDMA/GDR 或明确记录回退 |
| 2 | 真实传输与正确性 | 2×1→2×4→2×16；dispatch 精确匹配；FP32/BF16 Combine 检查；空 rank、全本地、全远端、非均衡、混合路由、连续 buffer 复用 |
| 3 | 通信主扫描 | EP32、256 experts、top8、hidden2048、BF16；T/rank=1/5/21/84/335/670；RelayMoE/NCCL/CollWay-like/DeepEP（安装验证通过后） |
| 4 | 跨机机制与消融 | 中档 T84：远端 expert assignment 比例 0/25/50/75/100%；远端 fan-out=1/2/4/8/16；均匀与热点；完整/DirectBatch/IndependentBatch/固定 ingress |
| 5 | 真实路由回放与全模型 | Qwen3-30B-A3B；32-rank 配置采集第0/24/47层路由；模型 smoke 后 batch1/4、8192输入+1输出 |
| 6 | 重复验证与汇总 | 初扫 seed123；代表性小/中/大及候选最好配置补至少3个独立 seeds/runs；保存全部样本与图表数据 |

阶段 3 的六档 T 与单机相同，是固定每 rank 工作量的扩展测量；总 token 数翻倍。
另在小/中/大点加固定全局 token 数的 16→32 比较，避免混淆两种 scaling 口径。
阶段 4 的比例定义为跨节点 expert assignments 占全部 assignments 的比例，记录
实际 token 跨机比例、目的 GPU 数与字节量。top8 下选到远端的 GPU 数并不等于
此比例。fan-out 控制单独生成，不强求与完整 MoE top8 输入相同。

## 对照方法

| 方法 | 双机实现范围 |
| --- | --- |
| NCCL direct all-to-all-v | 真实 all_to_all_single；按 peer 聚合；相同专家放置、路由、BF16 与同目标 GPU 合并 |
| CollWay-like | 用户指定的仓库端点无冲突分阶段算法；增加网络 transport；明确是算法对照，不是完整 CollWay 系统 |
| RelayMoE | 节点内中继与跨机 ingress 规划、负载感知树选择；完整在线开销计入 |
| DirectBatch | 同一混合执行器关闭中继，隔离执行器与算法收益 |
| IndependentBatch | 保留候选和中继，各 flow 独立选树，隔离跨组负载感知收益 |
| 固定 ingress/层级转发 | 同一执行器实现固定节点入口，比较自适应入口/树选择；若参照仓库 DeepEPBaseline，应命名 DeepEP-like |
| 官方 DeepEP | 已有 checkout revision `95acda3a8d3c8a278fdfca293473a7f1887f593b`；先安装、验证 PCIe 模式和 32 rank；未通过时记录原因，不以仿真替代 |

该 DeepEP checkout 的 `docs/README_PCIe.md` 描述节点内与节点间均经 RDMA NIC，
最多 32 ranks。需核对当前 16GPU/8NIC 的共享关系是否兼容，锁定依赖和 BF16 模式。
所有方法使用相同 NIC 资源预算；保留各自正常并发能力。先保持官方可运行配置，
不开展大规模 baseline 调参。RelayMoE 的主比较使用全部已验证数据 NIC；单 NIC
仅作为瓶颈机制实验，同图所有方法用相同限制。

## 模型布局

优先评估 TP16×DP2、EP32：节点内 TP，跨节点 EP。先验证所固定 vLLM 版本的
启动器、全局 EP group、专家放置、DP 请求协调及适配器接口。不能只改 EP 数。
全局 batch4 在两节点各2请求；batch1 包含空 DP 输入场景，必须能让32 ranks
正确参与同一次调用。若框架不支持该情形，明确暂缺或补充 batch2，不把2个请求
报告成 batch1。TP32×DP1可作兼容备选，但必须单列布局，不能混入TP16×DP2曲线。

主对照使用相同 token 布局、expert GEMM、dtype、融合及缓存设置，比较 RelayMoE
与匹配 NCCL；CollWay-like/官方 DeepEP 在通信层必测，完整模型适配可列第二批。
原生 vLLM 按实际 dispatch/combine 路径单列系统参照。模型加载/JIT 不计入稳态；
必要的 token 布局恢复和其他并行通信计入全模型时间。继续暂缓全模型 logits 对比。

## 采样、指标和产出

- 每配置20次预热+100 measured steps；所有方法使用同批路由并轮换顺序；取每步
  最慢 rank 完整调用时间。沿用尾部3个20-step窗口漂移≤3%、全窗口CV≤5%的标准；
  不稳定统一延至200/300步，仍不稳定则如实标记。
- 动态合成路由每步不同；静态真实路由回放明确标为静态。固定源 seed 和 gate
  weights，保持同目标 GPU 合并与本地专家处理一致。
- 主指标：完整 prepare+dispatch+combine P50/P95/均值、净 speedup、有效带宽；
  另报 dispatch/combine、规划与元数据、额外 buffer/显存、消息数及源注入量。
- 带宽分母使用匹配阶段时间；每 GPU 平均 AlgoBW=该阶段固定语义有效字节/(32×时间)。
  节点 cut payload、NIC port计数器增量和逻辑估计分别报告；NIC计数器含协议/背景
  流量，应记录空载基线并先停文件同步。不能把软件估算当成实测链路带宽。
- 全模型记录请求批次时延、输入 tokens/s、MoE层占比以及跨机通信占比。端到端
  时延取完整调用，不将各段P50相加；不预填预期毫秒数。
- 产出：原始JSON/CSV、环境与源码摘要、稳定性标记；尺寸–时延/带宽、跨机比例–
  净收益、消融、NIC负载和模型batch时延图，导出PNG/PDF/SVG。

首轮以这份有限清单得到初步结果，不做密集扫描。论文可突出经重复验证的最好配置，
同时标明配置和统计口径，并保留完整数据及未获益点供复核。
