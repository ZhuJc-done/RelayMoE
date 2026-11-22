# 单机 16 卡阶段收尾结果

按用户最新要求结束本轮单机验证：冻结 v9，补齐有限实验清单，不再继续调优或扩展参数。
模型保留在 `/data07/fy/relaymoe-models/Qwen3-30B-A3B`，代码与原始结果保留在
`/data02/fy/relaymoe`。单机收尾时未 push；2026-11-22 用户另行授权将冻结实现
提交至 `zjc/test_real_moe`，用于双机环境同步，详见 `two-node-sync.md`。

## 完成范围

- 11 个合成配置：六档尺寸、Zipf 0.4/0.6/0.8、两档 top-1；每项五种方法。
- 3 个静态真实路由回放：Qwen3 batch 1 / 512 tokens 的第 0、24、47 层。
- Qwen3 完整模型 v9 batch 1 和 4，8192 输入 token/请求加首个输出 token。
- P2P 44 点、fan-out 54 条原始记录、参考后端 2/4/16 卡以及 v9 BF16 正确性记录。
- 全部原始样本保留。v9 通信只有 seed 123；其余四 seed 延后，v8 五 seed 结果独立保存。
- 采样为 20 warmup +100 steps，不稳定则最多 300；覆盖完成不等于稳定性全部通过。

## v9 完整通信 P50

单位 ms，包含在线 prepare、dispatch、combine，不包含专家 GEMM。
NCCL 是当前 all-to-all-v 适配器；CollWay-like 是仓库算法及执行器，不代表完整论文系统。

| 配置 | NCCL | CollWay-like | RelayMoE | DirectBatch | IndependentBatch | NCCL/Relay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| size-t1-s123 | 0.615 | 1.087 | 0.636 | 0.509 | 0.627 | 0.966× |
| size-t5-s123 | 0.883 | 2.307 | 0.866 | 0.652 | 0.839 | 1.020× |
| size-t21-s123 | 1.556 | 7.461 | 1.383 | 0.987 | 1.242 | 1.125× |
| size-t84-s123 | 5.147 | 42.679 | 3.305 | 2.250 | 2.804 | 1.557× |
| size-t335-s123 | 17.296 | 537.220 | 12.794 | 8.933 | 10.784 | 1.352× |
| size-t670-s123 | 36.502 | 2196.504 | 26.164 | 19.477 | 21.933 | 1.395× |
| skew-0.4-s123 | 5.099 | 59.245 | 3.860 | 2.303 | 3.054 | 1.321× |
| skew-0.6-s123 | 4.870 | 63.066 | 3.837 | 2.306 | 3.076 | 1.269× |
| skew-0.8-s123 | 4.486 | 61.627 | 3.862 | 2.478 | 3.183 | 1.162× |
| top1-t1-s123 | 0.578 | 0.836 | 0.512 | 0.503 | 0.506 | 1.129× |
| top1-t670-s123 | 2.087 | 3.019 | 2.087 | 1.772 | 1.783 | 1.000× |
| trace-layer0 | 1.475 | 7.258 | 1.335 | 0.973 | 1.248 | 1.104× |
| trace-layer24 | 1.721 | 13.825 | 1.664 | 1.103 | 1.367 | 1.034× |
| trace-layer47 | 1.535 | 10.862 | 1.497 | 1.050 | 1.230 | 1.025× |

六档尺寸中，相对 NCCL 的最大已观测完整通信 P50 加速为 1.557×
（size-t84-s123，单 seed 配置级比较，不是最快单步，也不是五 seed 稳定收益）。

静态回放使用真实 expert IDs 和 gate weights，但重复同一份路由；在线 prepare 仍计时，
整计划缓存关闭。这些结果不代表独立动态路由样本或完整模型时延。

## 完整模型

| Batch | 方法 | P50 (ms) | P95 (ms) | 均值 (ms) | 实测 steps | 稳定性通过 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 原生 vLLM（复制 token 布局） | 244.312 | 245.890 | 244.312 | 300 | 是 |
| 1 | 匹配 NCCL 适配器 | 538.749 | 569.461 | 537.979 | 300 | 是 |
| 1 | RelayMoE | 510.538 | 592.707 | 533.559 | 300 | 否 |
| 4 | 原生 vLLM（复制 token 布局） | 968.987 | 983.936 | 970.214 | 300 | 是 |
| 4 | 匹配 NCCL 适配器 | 2070.679 | 2135.983 | 2066.840 | 300 | 是 |
| 4 | RelayMoE | 2009.590 | 3027.357 | 2144.551 | 300 | 否 |

Batch 1：RelayMoE 相对匹配 NCCL 的 P50 时延变化 -5.24%，P95 变化 +4.08%，均值变化 -0.82%（负值表示降低）。

Batch 4：RelayMoE 相对匹配 NCCL 的 P50 时延变化 -2.95%，P95 变化 +41.73%，均值变化 +3.76%（负值表示降低）。

## 硬件、在线成本与机制

16 MiB 传输的聚合 payload 带宽如下；该硬件基准与 MoE 有效载荷带宽分开解释。

| 路径 / 并发 | GB/s |
| --- | ---: |
| PIX_forward | 48.264 |
| PIX_reverse | 46.876 |
| PIX_bidirectional | 49.383 |
| NODE_forward | 46.490 |
| NODE_reverse | 46.670 |
| NODE_bidirectional | 48.248 |
| SYS_forward | 46.788 |
| SYS_reverse | 46.807 |
| SYS_bidirectional | 81.554 |
| NODE_concurrent | 97.431 |
| SYS_concurrent | 98.103 |

在线准备开销采用所有 step 的均值，不能与前面 P50 相加或相减。

| 配置 | 方法 | prepare 均值 (ms) | planner 均值 (ms) | 源注入量相对 Direct 降低 |
| --- | --- | ---: | ---: | ---: |
| size-t84-s123 | relay_batch | 2.991 | 2.180 | 55.85% |
| size-t84-s123 | direct_batch | 1.912 | 1.224 | 0.00% |
| size-t84-s123 | independent_batch | 2.484 | 1.688 | 56.79% |
| skew-0.8-s123 | relay_batch | 3.337 | 2.384 | 52.67% |
| skew-0.8-s123 | direct_batch | 1.973 | 1.147 | 0.00% |
| skew-0.8-s123 | independent_batch | 2.513 | 1.577 | 53.13% |

这两个负载中，DirectBatch 和 IndependentBatch 的完整通信仍快于完整 RelayMoE。
源注入量降低并未抵消所有在线开销，当前数据尚未证实中继与联合选树的净时延收益。
静态 fan-out 的 d=15 源注入副本数为 15→4（减少 73.33%）；不把此结果解释为
全局传输量变为 4 份或实测墙钟时间必然按 log N 缩短。

原生 vLLM 在当前 TP16/DP1/EP16 布局使用复制 token、本地专家和 TP AllReduce；
适配器使用源 token 分片、稀疏 dispatch/combine，再 AllGather 恢复布局。
因此原生系统参照与匹配 NCCL 适配器分列，不能互换。全部方法保持相同 eager、
prefix cache 关闭、BF16 与专家/attention kernel 配置。

## 稳定性与结论边界

通信五 seed 稳定收益尚未完成验证。CSV 保留 P50、P95、尾部窗口漂移和窗口 CV；
稳定阈值为最后三个 20-step 窗口的均值极差/均值 ≤3%，全部完整窗口均值 CV ≤5%。
不裁剪慢样本，不把单 seed 的最大收益改称跨场景普遍收益。
DirectBatch 与 IndependentBatch 是相同执行器的消融；其结果决定机制是否带来净收益。
Fan-out 的源注入量降低与全局传输量、端到端耗时是不同指标。

## 归档与后续

原始输入清单及 SHA256、检查结果见 `artifacts/single-node-16gpu/closeout/audit.json`。
主要表格见该目录 `communication.csv`、`models.csv`、`mechanisms.csv`；逐 step 数据保存在原始 JSON。
机制表的瓶颈 cut 由同配置 DirectBatch 的平均字节数确定，再在相同 cut 上比较各方法；
cut 字节来自软件拓扑模型，不是 PCIe 硬件计数。Torch 显存峰值含同进程常驻后端，
显式 IPC 缓冲区另列，不把两者混称各方法的独立总显存需求。
图分别位于 `final-v9/analysis/`、`trace-v9/analysis/` 和 `model-v9-figures/`。
本阶段不再追加实验。多 seed 复现、模型/拓扑扩展与第二台服务器上的 DeepEP 对比留待新阶段。
测量结束时，实验 runner 已确认所有测量进程退出且 GPU 无计算进程；该状态是结束时快照，
不表示服务器之后一直空闲。专属容器及环境保留，恢复收尾不再启动 GPU 实验。
