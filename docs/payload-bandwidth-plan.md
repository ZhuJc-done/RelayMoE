# Payload 扫描数据与绘图

日期：2026-11-21。目标是生成用户参考图样式的多面板带宽曲线，并保留完整时延、
原始样本和版本记录。当前只展示 relay2 单机 16 GPU 的测量，不填充跨节点或未运行方法。

## 理论预期与实验问题

不预设 `RelayMoE > CollWay-like > NCCL` 的全局排序。在假设每轮每个持有者最多
向一个新节点发送、不同节点的发送可并行且不争抢瓶颈的简化模型中，向 d 个目标
广播相同数据，源端串行需要 d 轮，倍增转发可以用 ceil(log2(d+1)) 轮。
例如 15 个目标为 15 轮对 4 轮。这是轮数推导，不代表实际端到端时间必然按此比例下降。

在线规划、打包、同步、链路共享和中继读写均计入实际耗时；多源并发也会争用链路。
此外 NCCL 的 grouped send/recv 允许不同 peer 的调用并发推进，不能将其等同于
逐条串行发送。NVIDIA 官方依据为 NCCL user guide 的 usage/p2p 页面。
实验需要分别回答传输收益、完整在线通信收益和模型收益。

用户参考图中的随机 AlltoAllv 和均匀 AlltoAll 是两类负载。我们当前是动态 MoE
路由产生的稀疏传输，有 token 多目标共享和 combine 归约语义；借鉴坐标和排版，
不能把当前结果改名为均匀 AlltoAll，或直接比较参考图里的绝对 GB/s。

## 已整理数据

| 数据集 | 尺寸 / rank | seeds | 实测步数 / 方法 / seed | 用途 |
| --- | --- | --- | --- | --- |
| final-v7 | T=1/5/21/84/335/670 | 123 | 200–300 | 完整三方法尺寸预览；不是最新实现 |
| final-v8 | T=84 | 123/234/345/456/567 | 300 | 同一尺寸跨种子重复性 |
| pilot-v9-candidate | T=2/91/728 | 123 | 100–300 | 最新候选；尚缺 CollWay-like |

每轮 warmup=20。三个版本分开导出；所有测量样本保留，不筛除慢 step。
共校验 14 个原始 JSON，201 行 method/seed/phase 数据。

v8 T84 的 P50 跨种子几何平均如下；表中时延越小越好：

| 方法 | Dispatch (ms) | Combine (ms) | 含 prepare 的完整通信 (ms) |
| --- | ---: | ---: | ---: |
| RelayMoE | 0.526 | 0.234 | 3.581 |
| NCCL adapter | 0.890 | 0.285 | 5.422 |
| 仓库 CollWay-like | 16.800 | 17.705 | 45.864 |

此配置 RelayMoE/NCCL 的配对 P50 加速比几何平均为 1.514x。所有 seed 的稳定性
未全部通过，仍标作预览。CollWay-like 使用仓库分阶段 IPC 实现，不是完整 CollWay
系统；该差距包含执行器开销。同 batch 执行器 Direct 与 Independent 的数据另存消融图。

## 指标与输出

每 step 的全局远端有效字节记为 B，P=16：`U = mean_steps(B/P)`。
一个 token 去同一远端 GPU 的多个 expert，隐藏状态只计一次；各方法使用一致语义。
U 排除本地流量与元数据，区别于输入激活字节 `T*2048*2`，也不是 PCIe 硬件计数器。

- 横轴：U/1024，KiB，log2 坐标；动态路由产生的 U 是均值，不假称固定载荷。
- Dispatch、Combine 纵轴：`U / P50(phase_seconds) / 1e9`。
- 完整在线往返纵轴：`2U / P50(full_seconds) / 1e9`，包括 prepare、dispatch、combine。
- 各阶段先取该 step 的跨 rank 最大墙钟耗时，再统计 P50/P95；阶段中位数不可相加。
- 完整通信排除 expert GEMM，不能称为完整模型端到端；模型单独绘图。
- 单 seed 阴影表示 `U/P95` 到 `U/P50`（完整往返为 2U），不是置信区间。
- 多 seed 点为各 seed 带宽的几何平均，误差线为 seed 最小/最大值。

本图使用“平均载荷 / P50 时延”。旧 plot_results.py 的带宽是“逐 step 带宽的中位数”，
动态路由下二者不严格相同，因此新字段明确命名为 bandwidth_payload_over_p50_GBs。

输出在各数据集的 `analysis/payload-figures/`：

- `key-data.csv`：各方法、seed、阶段的字节、P50/P95、带宽、规划均值、稳定性和 SHA256。
- `curve-data.csv`：绘图点和跨种子范围；`paired-speedups.csv`：配对加速比。
- `payload-bandwidth.{png,pdf,svg}`：RelayMoE、仓库 CollWay-like、NCCL 主图。
- `payload-bandwidth-ablations.{png,pdf,svg}`：同执行器消融。
- 独立图注、`manifest.json`：原始文件与代码摘要，可重新生成。

```bash
python scripts/plot_payload_bandwidth.py artifacts/single-node-16gpu/final-v7 --label v7
python scripts/plot_payload_bandwidth.py artifacts/single-node-16gpu/final-v8 --label v8
python scripts/plot_payload_bandwidth.py artifacts/single-node-16gpu/pilot-v9-candidate --label v9-pilot
```

## 下一轮补齐顺序

用户已要求停止进一步性能优化，并对单机 16 卡验证阶段收尾。此要求优先于
原先的加密扫描和多轮优化安排；当前按以下有限范围结束本阶段：

1. 冻结 v9，完成已启动的 seed 123 六档尺寸、三档 Zipf、两档 top-1 配对。
   五种方法包含两项相同执行器消融，因此 Zipf 0.8 消融也在这批内。
2. 回放已采集的 Qwen3 batch 1 / 512 tokens 第 0、24、47 层真实路由，明确标为
   静态 trace，在线 prepare 仍计时，整计划缓存关闭。
3. 补齐 v9 的 Qwen3 batch 4 / 8192+1 配对；与已完成的 batch 1 单列报告。
   继续采用 20 warmup +100 实测，不稳定时按既定规则最多 300，不追加调优。
4. 汇总同版本 CSV 和图片，保留 P50/P95、规划开销、所有慢样本与负收益配置。
   v8 的五 seed 结果独立保留；v9 未完成的多 seed 重复作为后续工作，不伪称五 seed 已完成。
5. 不再加密尺寸、不新增均匀 AlltoAll、不扩展模型与配置。归档后确认本任务 GPU
   进程退出；双机 2×16 和 DeepEP 留给下一阶段。

当前代码与记录均在 main 未提交工作区，未 push。
