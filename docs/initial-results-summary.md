# 单机 16 卡初步实验结果汇总

本文保留收尾前的历史快照。最新覆盖和结果见 `single-node-16gpu-results.md`；
其中 v9 倾斜、top-1、三个静态真实route回放及 batch 4 已补齐，本阶段 GPU 任务已退出。

记录日期：2026-11-21。平台：relay2，16 × TK300 72G，PCIe，8+8 NUMA。
代码目录：`/data02/fy/relaymoe`；当前正式通信实现为 v9。
本次按用户要求停止继续探索性能优化候选，汇总已有结果。已启动的覆盖补测继续执行。
本文为检查时的结果快照，不表示仍在运行的任务已经完成。

## 1. 实验覆盖

| 项目 | 已有数据 | 尚未完成 |
| --- | --- | --- |
| 环境与正确性 | 环境、拓扑、模型权重；2/4/16 卡参考后端；v9 的 18 项 BF16 用例 | 全模型 logits 比较按用户要求暂缓 |
| PCIe P2P / 链路竞争 | 44 个测试点，覆盖 PIX/NODE/SYS、双向及并发 | 本轮已有初步结果 |
| Fan-out | 54 条原始记录，d=1/2/4/8/15、64 KiB/1 MiB、两类目的布局 | 尚未证实普遍的时延优势 |
| R1 消息量扫描 | v7 六个尺寸各 1 seed；v8 中档 T84 有 5 seeds；v9 四个尺寸完成 | v9 大消息补测运行中；多数配置缺其余 seeds |
| R1 倾斜与 top-1 | 已排入运行中的 v9 补测 | 检查时尚无完整配对结果 |
| R1 真实路由回放 | 已采集 Qwen3 路由；模型实验使用自然 gating | 独立的真实 route 通信回放尚未完成 |
| R2 机制消融 | 均衡负载下 DirectBatch、IndependentBatch 与 RelayBatch 的配对结果 | Zipf 0.8 消融尚未完成 |
| R3 完整模型 | Qwen3-30B-A3B，8192+1，batch 1 和 4 均跑通过；v9 有 batch 1 | batch 4 目前只有旧版 v7 数据，缺 v9 复测 |
| R4 在线开销 | prepare、元数据、规划、布局、同步等分项记录 | 可继续汇总，无需再开发优化候选 |
| 数据与绘图 | 原始样本、CSV、PNG/PDF/SVG、源码快照及校验信息已归档 | 完整同版本曲线等待覆盖补测 |

DeepEP 与跨节点 2×16 按用户决定留到下一阶段；不是本轮已完成的对照。
CollWay-like 指仓库算法实现及其执行器，不能等同于完整 CollWay 系统。

## 2. 通信代表结果：v8 中档负载，五个 seeds

配置：EP16、256 experts、top-8、hidden 2048、BF16、每 rank 84 tokens，
远端有效载荷约 2055 KiB/rank。seeds=123/234/345/456/567，每 seed 20 次预热、300 次测量。
时延为各 seed P50 的几何平均；完整通信包含在线准备及 dispatch/combine，不含专家 GEMM。

| 方法 | Dispatch P50 (ms) | Combine P50 (ms) | 完整通信 P50 (ms) |
| --- | ---: | ---: | ---: |
| NCCL 适配器 | 0.8903 | 0.2846 | 5.4218 |
| 仓库 CollWay-like | 16.7996 | 17.7046 | 45.8640 |
| RelayMoE / RelayBatch | 0.5265 | 0.2337 | 3.5806 |
| DirectBatch：关闭中继 | 0.5043 | 0.2745 | 2.4431 |
| IndependentBatch：独立选树 | 0.5308 | 0.2428 | 3.0744 |

RelayMoE 相对 NCCL 的完整通信配对加速比几何平均 **1.514×**，各 seed 为
1.366–1.592×，对应时延降低 33.96%；相对仓库 CollWay-like 为 12.809×。
但它比 DirectBatch 慢 46.56%，比 IndependentBatch 慢 16.46%。
因此这组数据支持相对当前 NCCL 适配器的收益，尚未证明中继与联合选树机制在均衡负载下的净收益。
CollWay-like 的差距包含其逐动作执行器开销，不能全部归因于算法。

有效带宽为同一语义远端字节数除以 P50 阶段时延，再对五 seeds 取几何平均：

| 方法 | Dispatch (GB/s/rank) | Combine (GB/s/rank) |
| --- | ---: | ---: |
| NCCL | 2.364 | 7.393 |
| CollWay-like | 0.125 | 0.119 |
| RelayMoE | 3.997 | 9.005 |

这是有效载荷带宽，不是物理 PCIe 链路带宽或 nccl-tests busbw。
五 seeds 尚未全部通过既定稳定性条件，`verified_best` 仍为空；以上为初步观测。
来源：`artifacts/single-node-16gpu/final-v8/analysis/aggregate.csv`、
`final-v8/analysis/payload-figures/key-data.csv`。

## 3. 当前 v9 的已完成尺寸

检查时四个尺寸完成，均为 seed 123、20 次预热加 300 次测量。
下表为完整通信 P50，单位 ms；不能与 v7/v8 的其他尺寸拼成同版本曲线。

| Tokens/rank | NCCL | CollWay-like | RelayMoE | DirectBatch | IndependentBatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6146 | 1.0874 | 0.6361 | 0.5094 | 0.6270 |
| 5 | 0.8830 | 2.3067 | 0.8656 | 0.6517 | 0.8393 |
| 21 | 1.5564 | 7.4608 | 1.3830 | 0.9873 | 1.2425 |
| 84 | 5.1466 | 42.6791 | 3.3054 | 2.2498 | 2.8042 |

中档 T84 相对 NCCL 约 1.557×；最小消息 RelayMoE 仍慢于 NCCL。
所有列在这些点上均至少一项既定稳定性条件未通过，不能标作稳定最终结论。
T335 正在运行；T670、skew、top-1 尚未完成。不将中途写出的部分样本作为完成结果。
来源：服务器 `artifacts/single-node-16gpu/final-v9/`。

## 4. Qwen3 完整模型

完整 Qwen3-30B-A3B 权重、BF16、TP16/DP1/EP16，eager、关闭 prefix cache。
每次测量是每请求 8192 输入 token 的 prefill 加首个输出 token；20 次预热、300 次测量。

| 版本与批次 | 方法 | P50 (ms) | P95 (ms) |
| --- | --- | ---: | ---: |
| v9 / batch 1 | 原生 vLLM | 244.312 | 245.890 |
| v9 / batch 1 | 匹配布局 NCCL 适配器 | 538.749 | 569.461 |
| v9 / batch 1 | RelayMoE | 510.538 | 592.707 |
| v7 / batch 4 | 原生 vLLM | 970.011 | 988.819 |
| v7 / batch 4 | 匹配布局 NCCL 适配器 | 1987.629 | 2118.336 |
| v7 / batch 4 | RelayMoE | 2141.421 | 3632.618 |

当前 v9 batch 1 相对匹配 NCCL，P50 降低 **5.236%**，P95 高 4.082%。
RelayMoE 尾部窗口漂移 5.278%，超过 3% 阈值；窗口 CV 4.717%，通过 5% 阈值。
旧版 v7 batch 4 的 RelayMoE P50 比 NCCL 高 7.74%；不能据此推定 v9 batch 4。

原生 vLLM 在此布局使用复制 token 输入、本地专家、TP AllReduce；适配器使用
源 token 分块、稀疏 dispatch/combine、AllGather 恢复 TP 布局。因此原生结果作为
系统参照单列，当前也确实更快；不能宣称 RelayMoE 已胜过原生 vLLM。
来源：`qwen3-batch-v9-b1-8192.json`、`qwen3-batch-v7-b4-8192.json`。

## 5. 硬件、机制与在线开销

- P2P 16 MiB 单向测试：PIX 约 48.26 GB/s，NODE 约 46.49 GB/s，SYS 约 46.79 GB/s；
  四对并发 NODE/SYS 聚合约 97.43/98.10 GB/s。来源：服务器 `p2p-intra-affinity-300.json`。
- Fan-out d=15 时，源 GPU 的发送副本数从 15 降至 4，源注入量减少 73.33%。
  这是源端负担减少，不是全局总传输量降至 4，也不意味着墙钟时间必然从 N 降至 log N。
  静态预建计划测试尚未验证普遍的总时延优势。来源：`fanout-v7.json` 及其 summary。
- R4：v8 T84 五 seeds，RelayMoE 的 prepare 均值约 3.274 ms，其中 planner 约 2.412 ms；
  DirectBatch 对应 2.029/1.297 ms，IndependentBatch 对应 2.709/1.852 ms。
  在线准备成本值得在解释消融结果时单列。此处是均值，不能与上表 P50 相减或相加计算占比。
- 长尾诊断中，慢请求主要增加在通信与同步等待；屏障等待也可能由较慢 GPU 引起。
  v11/v12 候选没有一致改善，未采用；正式通信代码保留 v9。

## 6. 后续范围

停止继续探索极致优化。后续围绕冻结的 v9 补齐倾斜、top-1、真实路由独立回放、
倾斜消融与 batch 4，再统一生成表格和图。继续保留所有原始样本、旧版本与未胜出结果。
不把缺测记为完成，不混合版本或仅挑有利 step/seeds。代码继续保留本地及服务器，未 push。
