# RelayMoE 单机 16 卡实验计划（已批准，阶段已收尾）

状态更新：用户要求停止继续调优并结束单机阶段。已完成冻结 v9 的 11 个合成配置、
3 个静态真实路由回放和 Qwen3 batch 1/4；剩余 v9 seeds 与参数加密推迟。
当前结论与范围以 `single-node-16gpu-results.md` 为准，下文保留原始计划以便追溯。

日期：2026-11-21。平台：relay2，16 × TK300 72G，纯 PCIe 节点。
代码与结果目录：`/data02/fy/relaymoe`。镜像：`zjc_relaymoe:latest`。
用户已批准本计划并授权实施；实际进展和偏差见 `experiment-progress.md`。

后续已批准调整（优先于下文初始计划）：本阶段只做单机；CollWay采用仓库的
CollWay-like算法执行实现，不要求完整系统复现；DeepEP/NIC及跨节点留到后续；
暂停logits比较，性能优化集中在RelayMoE；结果归档与最佳收益筛选遵循
`result-recording.md`，全部原始样本保留。下文“待实现/建议确认”等为原始计划
的历史状态，不能用来判断当前完成度，实际状态以进展记录为准。

## 1. 范围与当前状态

本轮完成单机 EP16 的正确性、通信性能、机制消融、在线开销和一个真实 MoE
模型的 prefill 端到端验证。暂不开展双节点、训练、长期稳定性、全模型横向扫描。

已验证：16 卡 CUDA 计算、CUDA 扩展编译运行、NCCL all-reduce 和环形收发、
23 项现有单元测试。这些结果不等于 RelayMoE 的真实 dispatch/combine 已实现。
当前 `relaymoe/runtime.py` 只有 backend 协议；示例 backend 原样返回输入，
仓库中的 NCCL-like、CollWay-like、DeepEP-like 是仿真对照。

实际执行顺序如下，每一阶段通过后才进入下一阶段：

| 顺序 | 工作 | 通过条件 / 产出 |
| --- | --- | --- |
| 0 | 固定环境、资源状态、模型存储与真实 GPU 拓扑 | 环境清单、版本、GPU/NIC/NUMA 映射；无其他 GPU 任务 |
| 1 | 实现真实 NCCL direct 与 RelayMoE 通信 backend | 2→4→16 rank 完成 dispatch→专家算子→combine 数值检查 |
| 2 | P2P、竞争链路和 fan-out 小基准 | 三类 PCIe 路径校准；源注入量、轮数、耗时 |
| 3 | 接入 Qwen3 模型并做小规模 smoke，采集 routing | 一次完整 forward 成功；真实逐层 route 样本 |
| 4 | R1 通信性能扫描，同时采集 R4 在线开销 | 消息量、倾斜、top-1；端到端通信净收益 |
| 5 | R2 两个机制消融 | 完整/直接发送/独立选树，使用相同输入和执行后端 |
| 6 | R3 Qwen3 正式端到端 prefill | 8192 输入 token、输出 1 token；全局 batch 1、4 |
| 7 | 汇总原始样本与报告 | 图、表、数值误差、复现命令、明确未完成项 |

阶段 1 是实际开发工作，不把它当成现有能力。先使用 PyTorch/NCCL 点对点通信
打通功能，再消除逐 action Python 调度、全局逐轮 barrier 等显著开销，接入
原生批量提交/依赖事件；保留不可消除的 CPU 调度成本。功能原型的性能可记录，
但不能冒充优化 CUDA backend 的最终性能。

## 2. 存储与容器

2026-11-21 实测可用容量（GiB，各 dataXX 为独立 NVMe 分区）：

| 目录 | 可用容量 |
| --- | ---: |
| /data00 | 79.0 |
| /data01 | 0.0 |
| /data02 | 68.0 |
| /data03 | 134.7 |
| /data04 | 274.6 |
| /data05 | 240.9 |
| /data06 | 354.2 |
| /data07 | 395.1 |
| /data08 | 338.6 |

`/data00` 与 `/var/lib/docker`、`/var/lib/containerd` 共用一个分区，不能将
它们的可用空间重复相加。

建议批准后在 `/data07/fy/relaymoe-models` 存放模型，缓存放在
`/data07/fy/relaymoe-cache`。先查找已存在的指定模型，匹配版本后只读复用；
缺失时只下载本轮主模型，避免同时保存镜像缓存副本和额外模型副本。
代码和小型结果保留在 `/data02/fy/relaymoe`，日志/trace 初始预算 2 GiB。

当前开发容器仅挂载代码目录，不包含 `/data07`。建议使用同一镜像另起
`zjc_relaymoe_exp` 实验容器，工作目录不变，额外挂载模型和缓存目录；保留
用户当前开发 shell，不改宿主驱动、不修改其他容器。依赖增量须固定版本并
记录到项目环境文件，验证后才更新同一个镜像标签。

运行前重新检查 GPU 进程。容器不是 GPU 资源预约；出现其他 GPU 任务时暂停
本实验并报告。只终止本实验启动的进程。使用 NUMA 就近 CPU 绑定和有限编译
并行度，不改全局时钟、功耗或内核设置。正式单机主实验关闭 IB 数据传输，
检查 NCCL transport 日志区分 P2P、SHM、Socket；仅设置 IB disable 不能证明
所有传输都直接走 GPU P2P。

## 3. 模型选择

官方配置核对结果：

| 模型 | hidden | routed experts / top-k | 用途 |
| --- | ---: | --- | --- |
| Qwen3-30B-A3B | 2048 | 128 / 8 | 本轮唯一必做完整模型，与原 Evaluation 一致 |
| DeepSeek-V2-Lite | 2048 | 64 / 6，另有 2 个 shared experts | 若主模型集成受阻的较小备选；后续跨架构验证 |
| GLM-4.7-Flash | 2048 | 64 / 4，另有 1 个 shared expert | GLM 系列后续扩展候选，不纳入首轮必做 |
| GLM-4.5-Air | 4096 | 128 / 8，另有 1 个 shared expert | 更大模型的后续扩展，不纳入首轮必做 |

Qwen3-30B-A3B 的官方权重索引记录 tensor bytes 为 61,064,245,248，约
56.87 GiB；DeepSeek-V2-Lite 为 31,412,968,448，约 29.26 GiB。
这些是索引声明的张量大小，不是带缓存、加载临时文件的峰值磁盘占用。
`/data07` 容量足够本轮主模型和缓存，下载前再次核对实际文件清单与 dtype。
仅部分专家参与一次计算，不意味着只需下载或驻留这些活跃专家的权重。

默认 BF16、真实完整预训练权重、无训练/反向传播。随机权重可以用于通信和
层功能 smoke，但不作为真实模型端到端结果。只测 MoE 模型；Dense Qwen/GLM
不能检验专家路由通信收益。

端到端优先以固定版本 vLLM 集成通信 backend。Qwen3 先验证 TP16、DP1、EP16，
128 个专家每卡 8 个，固定 expert placement。锁定 attention、expert GEMM、
token 分片/聚合方式；确认 16 rank 组成同一 EP group，避免误测 16 个独立副本。
当前主线源码支持 Qwen3 的 KV head 复制，但选定框架版本仍需验证 SM120、
TP16、BF16 kernel 与自定义通信接口，不能仅凭配置就宣称可运行。
框架本身需要的 all-gather/reduce-scatter 纳入模型耗时，跨方法不改变。

若框架适配暂时受阻，可先完成通信和真实 MoE 层验证；不可把它们标为完整模型
端到端。变更模型、并行布局或计量口径时明确记录，不静默替换对照。

## 4. 正确性与计时基础

实现共享测试输入及元数据，验证：token 不丢失/不重复；expert membership
和 gate 权重正确；同 token 多专家落同卡；源卡本地专家；空流量/空 rank；
不均衡接收量；Combine 恢复原顺序；中继缓冲区无过早复用。

先用整数标识 payload 检查 dispatch 精确一致，再用 FP32 参考专家算子检查
加权 combine，最后用与主实验相同的 BF16 GEMM。dispatch 不做量化时必须
逐元素一致。FP32 小算子初始阈值 `atol=rtol=1e-5`，BF16 单层初始阈值
`atol=rtol=2e-2`，同时记录最大绝对误差与归一化 L2 误差。阈值在 pilot 前
冻结；超出先排查语义/归约精度，不按结果随意放宽。整模型报告 logits 差异，
允许浮点归约顺序导致接近决策边界的 token 差异并单独分析。

两条计时轨道：

1. GPU 数据路径：CUDA events，明确跨 stream 的依赖；每次调用取最慢 rank
   完成时间，再统计分位数。统计汇总放到该次计时区间之外。
2. 完整调用：单调墙钟覆盖 GPU routing 整理、元数据交换、CPU/GPU 规划、
   plan 发布、pack、dispatch、combine、unpack 与必需同步。调度器若获取
   全局流量，相关通信必须计入。先得到每个调用总时间，再取统计量。

不将各段 P50 相加；多 rank/多 stream 耗时可能重叠，分解只使用同次调用的
非重叠墙钟区间。专家 GEMM 单独计时；完整模型包含所有层、attention 和相关
并行通信。离线 catalog 构建、编译/JIT、模型加载只在初始化表中报告。

## 5. 阶段 2：最小硬件与中继机制验证

- P2P 路径：0↔1（同 PIX）、0↔4（同 NUMA、跨 PIX）、0↔8（跨 NUMA）。
  初始 payload 为 4 KiB、64 KiB、1 MiB、16 MiB；单向和双向分别记录。
  核对方向差异，再用各域代表 pair 检查是否可复用校准值。
- 再跑一组同时通信的 pair，检查共享链路竞争，不能用孤立 P2P 带宽直接代替
  16 卡并发下可用带宽。GPU 0–7、8–15 分属两个 NUMA，不能用均匀 PCIe 模型。
- Fan-out：固定源 GPU，远端目的数量 d=1/2/4/8/15，payload 64 KiB 与 1 MiB。
  轮换有代表性的目的集合，比较 direct 与 relay，并记录正向复制/反向归约。
  一对一轮次模型下，理想 relay 轮数为 ceil(log2(d+1))；d=15 时为 4 轮。
  这只描述理想通信轮数，不预设实测时间就是 O(log d)，更不把总字节量称为
  降至 log d。实际启动、复制、PCIe 竞争和中继同步均要测量。
- 真实 NCCL 对照保留其正常并发能力，不为迎合理论轮次人为串行化。

## 6. 阶段 4：R1 通信性能 + R4 在线开销

主合成输入固定 EP16、256 experts、top-8、hidden=2048、BF16。
主输入是每 token 独立的 expert top-k，记录 expert→GPU 映射和随机种子。
不把少量固定 routing motifs 的高重复输入当主结果。

| 扫描 | 配置 |
| --- | --- |
| 大小 | 目标语义有效量 U_dispatch/16 ≈ 32 KiB、128 KiB、512 KiB、2 MiB、8 MiB、16 MiB |
| 倾斜 | 固定中档 token 数，expert 热度 Zipf s=0、0.4、0.6、0.8 |
| 无中继共享控制 | top-1，小消息和大消息各一项 |
| 真实 routing | 从 Qwen3 smoke 中采集的代表性层与 batch 的 route 序列，按其原始配置回放 |

按完整 token 选最接近目标的整数数量，记录实际 T、U，不虚构精确消息量。
语义字节以同 token、同远端目标 GPU 只计一份 payload，排除本地目的；Combine
按其实际通信表示单独定义。AlgoBW=U/(16×T_phase)，分子跨方法固定。
实际逻辑发送字节、源 GPU 注入量、控制元数据字节另外记录，不充当有效字节。
此 AlgoBW 是本实验定义，不直接等同于 nccl-tests 的 busbw。

倾斜扫描保持 token 数、hidden、top-k 一致，专家采样对象和映射固定。
平均目的 GPU 数、局部命中比例、U 会随 routing 改变，逐点如实记录，不能声称
它天然同时满足“相同 token 数”和“精确相同网络字节量”。

必须对照：真实 NCCL direct/all-to-all-v 适配器与 RelayMoE。相同路由、数据、
专家放置、dtype；基础的同目标 GPU 输入合并和同 GPU 专家输出预归约一致。
对框架原生 NCCL allgather/reducescatter backend 单独标注算法，避免与 direct
all-to-all-v 混称同一个 NCCL 实现。

CollWay 只在找到并锁定真实代码后加入；保留其 chunk/path 优化。
DeepEP 需先验证版本和 PCIe 可用性。当前 main 要求 NVLink，且 NCCL 要求高于
当前镜像；其 hybrid-ep PCIe 实现文档描述所有通信经 RDMA NIC，不能混入
“纯 PCIe、不走 NIC”的主对比。若加入，作为允许 NIC 的独立补充配置，使用
相同资源条件重新比较。不能用仓库中的仿真类代替这些系统基线。

每点输出：Dispatch、Combine、完整通信 P50/P95/均值/波动；净 speedup；
AlgoBW；每 rank 显存峰值；额外缓冲峰值；逻辑 bytes/消息数；最繁忙 rank；
token 组数及每组 token 分布；候选数、action 数、目录与计划缓存命中率。

规划主场景：热 catalog、动态 routing、完整 plan cache 禁用。每 seed 准备
100 个独立 step 的 routing，同一批序列供所有方法使用。缓存命中场景只作
补充；若回放相同 trace，明确这是静态通信测量，不能冒充动态规划结果。
R4 直接从小/中/大点取数，不额外扫参数。

## 7. 阶段 5：R2 两个机制消融

复用中等消息量，选均衡和 Zipf 0.8 两种输入，每种比较：

1. 完整 RelayMoE。
2. 关闭中继：同 GPU 局部合并保留，只允许源到各目标的 direct 星形发送。
   不额外强制 direct 必须满足 relay 的对数轮数限制。
3. 关闭跨组负载感知：候选、代价、执行器不变，各流量组独立选树。

公共 pack/GEMM/执行器不变，避免把不同传输 kernel 的影响算成选树收益。
关闭负载感知只作用于选树与绑定，执行优先级仍依据实际总流量一致计算。
比较完整通信成本、源注入量和预先由 direct 确定的瓶颈 cut，记录是否发生
瓶颈转移。物理 cut 字节若来自路径模型，标注“推算”；未有硬件计数时不能
称为实测 PCIe 链路字节/利用率。

## 8. 阶段 6：R3 真实模型端到端

主模型：Qwen3-30B-A3B，完整预训练权重、BF16。固定模型 revision、tokenizer、
推理框架 commit、专家 kernel、EP/TP/DP、rank 映射和缓存/调度开关。

- smoke：先 512/2048 输入 token、batch 1，少量完整 forward 验证集成。
- 正式配置：8192 输入 token/请求，生成 1 token；全局固定 batch=1、4。
  batch=4 是同一次调度的四条请求，不是含义不明的“客户端并发 4”。
- 每 step 是一次新的固定批次 prefill+首 token，不是同一请求连续解码 100 token。
  权重、kernel、catalog 已热，prefix cache 关闭，每次确实执行 prefill。
- 固定非 thinking 的输入模板，输入长度按实际 token IDs 计。固定请求集合和
  方法运行块顺序；真实模型自然 gating，不能以预先固定 route 替换 gating。
- 主指标：服务端 prefill+首 token 的 batch 墙钟时间（P50/P95/均值）、输入
  tokens/s、相对 baseline 的端到端 speedup。HTTP 若启用，另外报告客户端
  TTFT，包括排队/传输，不能和服务端 CUDA 时间混为一谈。
- 从相同运行抽取代表 MoE 层，分解计划/pack、dispatch、expert GEMM、
  combine/unpack；记录通信占完整模型时间比例。
- prefix cache、chunked prefill、CUDA Graph、EPLB 开关在方法间一致。初版
  可统一 eager 做集成，若最终报告 eager 则明确范围；优化态再统一启用支持的
  graph，不能对 baseline 单方面关闭优化制造收益。

本轮先验证 prefill。长输出 decode/ITL 需要独立控制 KV 长度、batch 和时间
边界，列为后续可选实验，不把它混在这次的 100 step 中。

## 9. 短时采样与停止规则

默认 20 warmup + 100 个计时 step/配置/方法；编译/JIT/模型加载必须提前完成。
动态合成配置使用 5 个 workload seeds，方法共享每 seed 的 100-step 序列。
先以小/中/大三点和一个 seed 做 pilot；通过后跑完整清单，无须长时间训练。

预先规定的稳定性规则：将 100 个样本分为 5 个 20-step 窗口，比较后 3 窗的
均值漂移（目标 ≤3%）及重复块均值波动（目标 ≤5%）。动态 routing 自身可有
波动，保留负载统计及配对差异，不把 route 波动简单判成硬件不稳定。
未稳定时按 100 步增量最多扩到 300 步，并排查温度/时钟、其他任务、JIT 和
网络路径；达到上限仍不稳定则标注该点未稳定，不无限运行或挑选有利样本。

微秒级操作使用 CUDA events，必要时在一个计时窗口内重复若干次并除以次数；
注明这种 batch timing 不提供逐调用 P95，也不替代完整动态调用测量。
100 个样本可用于初步中位数/波动验证，不声称有稳定 P99。

跨方法交替运行固定小块，避免固定顺序受到温度或后台负载漂移影响。
保留逐 step 原始样本；先按 seed 做统计，再汇总各 seed 的配对 speedup。
不把 5×100 次执行当成 500 个独立工作负载，不按是否出现加速决定停止。
计时阶段不常驻 profiler，仅挑 3–5 次代表调用采样。

短的是测量阶段；backend 实现、框架集成、编译、模型获取所需时间另计。
结果目录：`artifacts/single-node-16gpu/<run-id>/`，包含 config、环境、原始
JSONL/CSV、正确性报告、汇总表和图片。公开结论区分实测、软件统计与模型推算。

## 10. 本轮建议确认的范围

批准范围：阶段 0–7，主模型只做 Qwen3-30B-A3B；主对照 NCCL 与 RelayMoE；
两个内部消融；20+100 初始采样和最多 300 的固定延长规则；模型放 /data07，
代码和结果留在 /data02。真实 CollWay/DeepEP 接入状况另外报告，不用仿真补位。
已获用户确认，按上述范围执行。

## 核对来源

飞书 Evaluation 第 597 版：
https://bytedance.larkoffice.com/docx/GJhGdy41boqW5Dx6aOdcxcDenJd

2026-11-21 读取以下作者/项目官方配置或文档；执行时进一步锁定不可变 revision：

- https://modelscope.cn/models/Qwen/Qwen3-30B-A3B/resolve/master/config.json
- https://modelscope.cn/models/Qwen/Qwen3-30B-A3B/resolve/master/model.safetensors.index.json
- https://qwenlm.github.io/blog/qwen3/
- https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite/resolve/master/config.json
- https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite/resolve/master/model.safetensors.index.json
- https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V2/main/README.md
- https://modelscope.cn/models/ZhipuAI/GLM-4.7-Flash/resolve/master/config.json
- https://modelscope.cn/models/ZhipuAI/GLM-4.5-Air/resolve/master/config.json
- https://raw.githubusercontent.com/zai-org/GLM-4.5/main/README.md
- https://raw.githubusercontent.com/deepseek-ai/DeepEP/main/README.md
- https://raw.githubusercontent.com/deepseek-ai/DeepEP/hybrid-ep/docs/README_PCIe.md
- https://raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/expert_parallel_deployment.md
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_moe.py

当前 web 查询未返回可用结果，模型与框架信息通过直接读取以上官方内容核对；
Hugging Face 直连失败时读取了作者在 ModelScope 发布的配置与权重索引。
