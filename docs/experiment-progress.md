# 单机 16 卡执行记录

日期：2026-11-21。用户已批准 `single-node-16gpu-plan.md` 的实验范围。
代码：`/data02/fy/relaymoe`；模型和缓存：`/data07/fy/relaymoe-*`。

## 已验证

- 完整 Qwen3-30B-A3B 下载完成，26 文件、61,084,185,894 字节；逐文件 SHA256
  校验通过，manifest 和完成标记保存在模型目录。
- 分阶段参考 backend：2/4/16 卡通过原始正确性矩阵。
- 事件 host backend：4 卡及 16 卡通过 direct/relay/independent × 六种路由 ×
  FP32/BF16 检查。16 卡结果 `correctness-event-share-16.json`，dispatch 精确一致。
  BF16 最大绝对误差约 0.145（数据幅度约 16）；通过预先冻结的 atol=rtol=0.02。
- vLLM 0.20.0 + transformers 4.57.6 + torch 2.11.0+cu130，SM120 完整模型
  smoke 通过：TP16、DP1、EP16、BF16，8 experts/GPU，FlashAttention2/Triton
  MoE，512 输入 token + 1 输出 token；预热后两次约 30.6/28.5ms，仅 smoke。
  `qwen3-smoke-native.json` 尚不是正式 100 step 或 RelayMoE 端到端对比。

## 发现与处理

1. 默认 NCCL 会因机内路径代价比较将跨 NUMA 流量选入 Socket。仅设置
   `NCCL_IB_DISABLE=1` 或 `NCCL_P2P_LEVEL=SYS` 不够。
   本轮任务进程设 `NCCL_NET_DISABLE_INTRA=1 NCCL_P2P_LEVEL=SYS NCCL_NET_PLUGIN=none`；
   16 rank 日志为 nNodes=1；最新 44 点校准数据路径均为 P2P/CUMEM。
   最新 16MiB 单向约 47GB/s，四对并发聚合约 99GB/s。
   `p2p-intra-calibration.json` 含原始样本；小消息受主机计时/调度影响显著，尚未稳定。
2. 每个有向 GPU pair 独立建 NCCL group 使 /dev/shm 8GiB 耗尽。
   改为 split_share，并按不相交 pair 的 round-robin 批量 split，降低共享内存。
3. torchrun 默认 nofile=1024，通信组 clone 的 TCPStore 连接触顶（现场 FD=1024）。
   `scripts/with_limits.py` 仅对实验 launcher 子进程将 nofile 提升至 65536。
4. direct combine 并发写同一 accumulator 存在潜在竞争；现用独立 reduction
   stream 串行同 buffer 的 add，保持各 transfer 并发。
5. 独立选树消融原先先做完整规划再重做，已修正为仅调用独立调度器。
6. vLLM TP16/DP1 的原生 MoE 路径复制 token，各 rank 计算本地 experts 后
   all-reduce；其标签不能写成 all-to-all。要验证 RelayMoE，需显式切片 token、
   dispatch/combine 后恢复模型要求的布局，并使对照使用相同布局。

## 当前限制与接下来工作

原生 vLLM 8192+1 两组各 20 warmup +100 测量已完成：batch1 P50=243.888ms、
P95=245.901ms、尾窗口漂移0.279%；batch4 P50=984.801ms、P95=991.790ms、
漂移0.378%。这是 eager、TP16/DP1 原生复制 token + all-reduce 基线，当前请求
内容为重复的英语说明句（不是最终多样请求集）。原始文件 `qwen3-native-b{1,4}-8192.json`。

紧凑 C++ 选树/布局 `CompactPlanner` 已对8个随机seed、relay/independent 两策略
逐流量核对，选树与 Python 参考一致（带宽32/24/16的核对配置）。已通过2/4/16
卡 BF16 GEMM 正确性测试，最大绝对误差约0.00456。默认性能带宽暂用47/47/47，
仍需将并发校准/共享cut模型明确化，不能视为已完成拓扑校准。
Qwen3 512+1 紧凑 relay 完整模型热调用0.744/0.708s，输出1246与原生一致；
仍慢于原生，尚未完成 logits误差核对。`qwen3-smoke-compact.json`。
紧凑小消息100step：direct=0.881ms、relay=1.261ms、independent=1.375ms
（完整在线通信中位数），文件 `compact-small.json`。这些是IPC同执行器消融，
不能将此direct称为NCCL。

新增验证：CUDA IPC 原生 host 执行器 `relaymoe/csrc/ipc.cu` 已通过 2/4/16 卡
36 项矩阵；NCCL 批量 direct 已通过 16 卡 12 项。Qwen3 全 48 层已接入 IPC relay，
512+1 完整输出 token=1246 与原生一致，真实路由已保存在
`artifacts/single-node-16gpu/qwen3-smoke-relay/routing-rank*.pt`。
首版全模型调用约 28.7s，移除无用阶段打包/专家行索引后热调用仍 16.5–25.2s。
这是功能结果，不能声称性能收益。规划、对象广播、每 flow 的张量/元数据构建
占主导，需进一步原生化。

第一个小消息动态 100 step（2 tokens/rank，top8，BF16，seed123）保存为
`communication-small.json`：完整通信中位数 NCCL direct=6.48ms，IPC relay=10.00ms，
IPC direct=9.26ms，independent=10.76ms；尚无净加速。这组发生在关闭
`build_expert_rows` 之前，保留作开发诊断，不替代最终同版本正式扫描。
容量复用改成 power-of-two 后 6 次动态调用的 IPC 重映射从 6 次降至 1 次，
但 Python 计划发布/布局仍占主导。`profile-medium*.json/.prof` 为诊断数据。

Python/Gloo 事件 host backend 仍只是功能执行器，控制和规划开销尚未优化。
完整模型仅原生 smoke 成功，还需真实 routing 采集、自定义通信集成、数值对比、
动态 100 step 通信和消融、完整 8192+1 batch1/4 端到端以及报告。
不得用参考 backend 的结果冒充最终原生执行器性能，不得用层测量代替完整模型。

## 最新范围与规划拆分（2026-11-21）

用户已明确：本阶段只取单机性能；CollWay 采用仓库 CollWay-like 算法；
暂不继续 logits 核对；基线合理配置后固定，优化重心是 RelayMoE 净时延/带宽。
双节点与第二台服务器由后续新 goal 提供。

独立 CPU 100 动态路由规划（热 catalog，plan cache 关闭）v2→v3 P50：

| tokens/rank | experts | 完整规划 v2 | 完整规划 v3 | v3选树 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 256 | 0.127ms | 0.127ms | 0.044ms |
| 91 | 256 | 7.741ms | 6.240ms | 2.996ms |
| 728 | 256 | 68.695ms | 49.951ms | 22.393ms |
| 512 | 128 | 48.245ms | 35.261ms | 16.224ms |

v3 消除了排序中反复构造字符串；16 个随机策略/seed 的选树仍与 Python 参考一致。
CollWay-like native 阶段与参考 8 seeds 完全相同，且 16 卡 BF16 检查通过。

v2 完整在线 prepare（含元数据/GPU构造）在 T2/T91 为 0.414/9.706ms；
这是旧版本16 rank测量，不能同上独立CPU计时混称同一口径。
100步完整模型 relay b1 8192+1 P50=2390.55ms，P95=2756.20ms，尾漂移2.166%；
没有端到端加速。`qwen3-relay-b1-8192.json`。

事件 ready/done 队列版正确性通过但性能退化，T91约318ms，已停止测量并恢复
扫描执行器。保留未完成原始文件 `compact-v3-t91.json` 作为回归诊断。
当前在验证 batch IPC：相同选树，按依赖层批量GPU读取与合并归约，计入阶段同步。
不把此 bulk执行器称为原来的物理边独占事件执行器，比较时单独标注。

CPU队列回退后又将 skeleton 的跨switch边计数移到离线预构建，正在核对。

### RelayMoE 批量执行优化

同一中档 T91、20+100 动态样本：
- v4：NCCL 5.498ms，CollWay-like 46.951ms，RelayMoE batch 11.629ms。
  Relay prepare 10.955ms，其中 native planner 6.679ms、materialize 4.044ms。
- v5：NCCL 5.418ms，RelayMoE batch 8.799ms；prepare 8.152ms，
  planner 6.603ms、materialize 1.342ms。
- v6 正在验证：批量执行只构造本rank索引，不再生成16rank完整action对象；
  候选评分同一(edge,count)重复项按flow缓存；无整plan缓存，候选树不变。

batch路径按树依赖层组织peer-read kernel，Combine同一缓冲区的子节点一次
FP32累计再写BF16，减少中间舍入；中间层用任务内共享内存fence同步。
这会改变相较逐边BF16归约的舍入顺序，单层固定容差正确性仍需通过。
对照中的Direct/Independent消融也需使用batch executor以隔离选树差异。

v6 T91：NCCL5.383ms，Relay5.324ms；尾窗口漂移分别3.79%/4.46%，
不能称1%差距为稳定收益。v6三点pilot：T2 NCCL0.667/Relay0.809ms；
T91 NCCL5.628/Relay5.473/DirectBatch3.077ms；
T728 NCCL41.596/Relay41.777/DirectBatch25.134ms。
选树与准备仍占主要成本，当前没有相对同执行器direct的净收益。
v7继续去除选树中的临时map/vector分配，并缓存相同流量量级的指数增量，
改变实现不改变数学分数；需重新核对reference exact trees。

v7 pilot（100 steps）完整通信P50，单位ms：

| T/rank | NCCL | Direct batch | Relay batch | Independent batch |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.674 | 0.572 | 0.798 | 0.799 |
| 91 | 5.407 | 2.894 | 4.738 | 3.861 |
| 728 | 39.737 | 22.754 | 34.447 | 27.978 |

Relay batch完整CPU planner P50：T2 0.083ms、T91 3.150ms、T72825.877ms；
选树分别0.051/2.020/14.769ms。此值来自16rank完整调用内插桩，不能直接与
先前独立CPU所有rank布局构造基准混称同一口径。
相对NCCL中/大消息有约12%/13%时延改善，但同执行器direct更快，小消息无收益。
已启动 `final-v7` 55 configs ×5 methods（5 seeds包含在55中），默认20+100，
窗口漂移>3%按100步延长、最多300。正确性包含8策略/六种路由48cases。

### 结果归档与绘图（执行中）

按用户要求，论文主图突出可复现的最佳配置收益，原始样本全部保留。
新增 `scripts/analyze_results.py`、`scripts/plot_results.py`、
`docs/result-recording.md`，可生成逐seed CSV、跨seed几何平均收益、最佳配置
表与PNG/PDF/SVG。输入文件和源代码SHA256随图数据保存。
汇总的四项检查通过：拒绝缺少seed的正式最大值、拒绝混合源代码版本、
拒绝不配对语义字节、使用P50而不是单个最快step。

首批4个完成配置已复制到本地并完成绘图外观检查。T84/seed123完整通信P50：
Relay4.6239ms、NCCL5.4533ms（15.21%时延下降），Dispatch speedup1.595x。
这是单seed观察值，不是最终五seed结论。所有20-step块的均值变异系数仍有
波动（Relay完整通信21.24%），不能只凭末三窗漂移0.26%称为全程稳定。
正式best筛选同时检查尾漂移≤3%和所有完整20-step块均值CV≤5%。
动态通信v7自动扩步当前只检查前者；本次不修改运行中的程序。

Qwen3后续配对脚本已改用 `relay:compact_batch`，加入完成标记、源代码hash、
窗口统计和100→200→300扩步，仍不采集logits。默认对照为原生vLLM和匹配布局
NCCL；CollWay-like已在通信层覆盖，不把该结果当作完整CollWay系统的模型收益。

正式v7完成seed123的六个size配置后，在检查点停止本任务torchrun，GPU进程
清空后插入fan-out与模型验证；之后续跑其余49个配置。已完成结果不重跑。
停止时进入skew首个warmup/块，未完成记录保留为
`skew-0.4-s123.interrupted.json`。检查发现 `Path.with_suffix` 会把小数skew
文件名截成 `skew-0.json`，现改为直接追加 `.json`，修复配置间覆盖/误跳过；
此改动只影响输出路径，同时补记runner hash，测量backend版本未变。

fan-out v7已完成54方法/配置行，每项20warmup+100初始/最多300实测，原始文件
`fanout-v7.json`。d=15时source injection从15份降为4份（减少73.33%），但
本实现时延没有对应改善。1MiB dispatch：NCCL434.89us、DirectBatch469.71us、
RelayBatch1740.38us；combine384.45/465.98/405.82us。小消息也有host抖动，
保留稳定标志。两种目的分布的PNG/PDF/SVG已生成并本地视觉检查。

batch完整模型512+1短测（4warmup+4测量，只用于集成确认）完成：native P50
27.98ms、matchedNCCL138.87ms、RelayBatch154.34ms。未采集/比较logits。
随后启动8192+1/batch1正式配对20+100，按两项稳定标准最多300步；GPU任务串行。

batch1正式测量已到300步上限，原始文件`qwen3-batch-v7-b1-8192.json`：
native P50/P95=244.614/246.561ms，matched NCCL=510.562/547.234ms，
RelayBatch=557.587/1006.675ms。native通过两项稳定规则，NCCL尾漂移3.21%，
Relay尾漂移29.12%、块CV11.38%，均保留为未稳定。模型层内rank0逐请求48层
加和中位数（诊断口径）：Relay prepare276.4ms，其中metadata101.4ms、
nativeplanner125.2ms、materialize46.9ms；dispatch81.0ms、combine102.0ms、
expert34.4ms、恢复布局allgather37.6ms。分段中位数不与总P50相加。
v7 batch4正式测试已启动，保持相同版本和三种方法。

已准备但尚未编译/采用的v8候选保存在本地`/tmp/relaymoe-ipc-v8.cu`和远端
artifacts目录。候选包括固定小数组代替临时vector、移动flow而非复制、
缓存源/目的集合的canonical拓扑描述、128-bit向量化peer read/write；保持
每次负载评分和选树，无full-plan缓存。当前GPU测量期间不编译或跑候选。
`check_planner_variant.py`将核对树/布局/阶段并做CPU短测，之后还需BF16数值
矩阵和16卡短测；未验证前不声称该候选提升性能。

候选v8另新增“复用TP复制路由”入口，减少模型内切片后再all-gather的重复交换；
配套检查将对齐整齐分片/尾padding/空rank的indices、dispatch和combine。
候选还给模型worker主线程设置与通信基准一致的NUMA邻近CPU亲和性，所有
模型对照共用；v7当前未绑定。候选保存在本地artifacts/candidate-v8与远端
artifacts下，尚未编译/验证。当前batch4在100步时Relay尾漂移3.9%，已扩到200。

batch4在200步时NCCL尾漂移3.61%、Relay尾漂移5.80%，按规则扩至最终300步。
读cgroup v1 CPU统计两次期间nr_throttled保持6，无新增节流，不能归因为CPU
配额耗尽。GPU进程仅本次16个worker，温度45–50°C，显存约38GiB/卡。

### v8 验证与采用（2026-11-21）

用户最新约束：不push。当前本地与relay2均在`main`，HEAD=`5e2f4f0`，
本轮实现均为未提交工作区变更；未创建实验commit，未push。服务器主要新增
文件尚未tracked，数据保留在artifacts，不将大体积实验数据提交到git。

v7 batch4 完成300步：native P50/P95=970.011/988.819ms，matched NCCL=
1987.629/2118.336ms，Relay=2141.421/3632.618ms。native/NCCL通过两项稳定
规则；Relay尾漂移7.943%，块CV4.143%，仍未稳定。原始数据SHA256为
`6c3b92027d9e896083db392cb6e379045b9f75e8b465f6da08cdf7bdb0e74aa7`。
batch1/batch4模型图已输出PNG/PDF/SVG并视觉检查。

发现batch4一次LLM.generate可能被vLLM拆为不同数量的forward pass；层计时
汇总已按每次请求的input_tokens累加完整48层块，验证各rank切分一致，不能
简单假设每次generate恰好48条层记录。原始wall timing不受此汇总修复影响。
batch4关键rank各层/forward加和的均值：Relay prepare969.42ms（metadata
428.95ms、planner397.35ms、materialize143.32ms），dispatch458.59ms、
combine523.92ms。层汇总脚本保留每次generate实际forward次数。

v8 native候选通过192组planner exact checks和16GPU的48个BF16数值用例；
树、容量、indices、dispatch/combine阶段与v7一致，BF16最大绝对误差0.00456。
CPU100步独立诊断中，T84/top8完整规划2727.77→1726.28us，T670/top8
20535.39→13798.79us；保持在线动态选树，不缓存选树结果。
候选源码SHA256 `4b1d0b277450300230c0c92d362429d06949a1b33c4efa51f9c394216ed7fea8`。

v8 native候选16GPU短测（20+100，单seed，非最终结论）完整通信P50/ms：

| T/rank | NCCL | Direct batch | Relay batch | Independent batch |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.662 | 0.515 | 0.667 | 0.659 |
| 91 | 5.061 | 2.385 | 3.520 | 2.938 |
| 728 | 40.428 | 21.617 | 31.033 | 24.433 |

v8比旧v7 pilot Relay分别0.798/4.738/34.447ms改善，但跨轮比较只能作
开发诊断。仍需同轮配对、多seed与稳定性检查，同执行器Direct仍更快。
通过上述检查后已采用native候选；v7源码冻结归档保留。模型侧复用复制路由
和CPU亲和性改动已接入，正在验证复用路径与原allgather完全一致，然后进行
真实模型逐层路由一致性诊断（不比较logits），性能测量将关闭诊断交换。

复用路由已通过16卡10个合成对照用例，以及真实Qwen3/512+1/batch4逐层
检查；后者每层将复制路由与原allgather结果exact比较，确认后性能测试关闭
`RELAYMOE_VALIDATE_REPLICATED_ROUTING`。v8 batch1/8192+1正式配对已启动。
后续通信runner扩步同时使用尾漂移≤3%和全部20步窗口CV≤5%，不修改v7数据。

v8 batch1正式300步已完成：native P50/P95=243.308/245.114ms，matched NCCL=
534.165/567.895ms，Relay=508.925/952.495ms。相对匹配NCCL的P50降低4.73%，
但Relay尾漂移7.76%、块CV9.52%，不作为稳定的端到端胜出结论；native仍更快。
所有GPU工作进程正常退出。模型图已另存`model-v8-figures`，不覆盖v7图。

按wall>=700ms分组仅用于瓶颈诊断，不删除正式统计中的样本：Relay共24个慢请求，
其prepare均值241.75ms，对照276个普通请求231.65ms；dispatch322.93 vs83.76ms，
combine304.37 vs96.02ms，planner103.79 vs103.15ms。长尾主要集中在传输/同步
路径，尚不能进一步归因于PCIe、CPU屏障或其他系统事件。下一步优先细分同步
和kernel耗时；不继续盲目优化CPU选树来解释此长尾。

开始v8正式T84配置的五seed配对（123/234/345/456/567），五种方法包含
NCCL、CollWay-like、DirectBatch、RelayBatch、IndependentBatch，20+100最多300。
该配置用于确认通信收益的重复性；其余规模和skew尚未补齐。

T84五seed均完成300步：完整通信相对NCCL P50加速比几何平均1.514x
（各seed1.366–1.592x），相对仓库CollWay-like12.809x；Relay对Direct batch
为0.682x，对Independent batch为0.859x。五seed稳定性未全部通过，正式
verified_best仍为空。不要将CollWay-like逐动作执行器的开销归为算法收益。
数据/CSV/图保存于`final-v8/analysis`。

v9仅合并batch执行的Python/C++边界：一个C++调用完成所有level，保留每个
level前后原有deviceSync和CPU barrier，GIL在完整transport期间一次释放。
未改变选树、树边、数值归约、IPC kernel或baseline NCCL/CollWay执行器。
候选通过3个batch方法×6路由的18个BF16正确性用例；pilot T2/91/728完整
通信Relay P50=0.653/3.610/26.284ms；T728四方法在100步均稳定，小/中消息
仍有波动。v8源码快照与原始数据保留；当前工作区试用v9进行模型复测。

v9 batch1/8192+1正式300步已完成：native P50/P95=244.312/245.890ms，
匹配NCCL=538.749/569.461ms，Relay=510.538/592.707ms。Relay P50降低
5.236%，但P95比NCCL高4.082%；尾漂移5.278%超过3%，块CV4.717%通过。
相对v8的长尾幅度明显缩小，属于跨轮开发观察；仍不能标作稳定端到端胜出。
原始JSON SHA256=`19679f12d13d998dae871b3e026afc0640a0042e47ccbab0502b1f069de90e7b`。
完整模型运行正常退出，图保存于`model-v9-figures`。

按用户参考图新增`scripts/plot_payload_bandwidth.py`：校验原始SHA和配对step/bytes，
分别导出v7完整尺寸、v8五seed单尺寸、v9 pilot的CSV、PNG/PDF/SVG、图注和manifest。
使用平均远端载荷/P50时延，完整在线往返分子为2U；14个原始JSON、201行阶段数据。
主图与同执行器消融分开；不同版本不拼接。规划见`docs/payload-bandwidth-plan.md`。

新增隔离分段计时诊断。首次v10因容器PYTHONPATH指向主工作区而未加载诊断
模块，输出已由`profile-v10-invalid.json`排除；不得据其源码字段推断执行版本。
修正为显式PYTHONPATH与module入口后，v10b的16个worker均打印并验证诊断
adapter/扩展路径；18个BF16用例通过。后续正式模型runner额外保存各worker
实际模块路径、文件SHA与已加载扩展SHA，并验证源码一致，检查在计时区间外。

v10b单方法20+100步诊断捕获长尾：关键rank在wall<=P50的请求中，dispatch
CPU屏障等待48.72ms、combine55.79ms；wall>=P90请求为195.86/168.71ms。
两项增加260.06ms，占两组总wall差338.80ms的大部分；deviceSync合计增加
65.14ms，planner仅增加7.34ms。某次慢请求rank0–3的dispatch deviceSync
为134–166ms，其他rank约18–25ms，随后全体等待。此归类仅为诊断，不裁剪数据，
CPU屏障等待也可能是慢GPU引起，尚不能认定全由CPU调度造成。
Python/C++边界每请求约0.1ms/阶段，不是主要瓶颈。

v11候选用CUDA IPC event表达每个level的跨rank依赖，CPU发布event record后
由GPU stream wait等待；每个level独立event槽，调用末尾保留deviceSync和全rank
屏障以安全复用事件/缓冲区。不改树、peer kernel、归约顺序或NCCL/CollWay执行器。
18个BF16正确性用例已通过，正在进行小/中/大消息pilot，尚未采用到正式后端。

v11 pilot三尺寸各300步完成：T2/91/728的Relay完整P50为0.787/3.601/27.315ms，
匹配NCCL为0.649/5.230/39.085ms。中消息相对NCCL为1.452x且双方稳定性通过；
小消息增加event开销，大消息仍有波动。与v9跨轮比尚无一致净改善，保留为候选。
补充64次连续事件复用测试，切换token数、hidden16/40/2048、fan-out、local与
空流量，异步输入生成，dispatch与combine均exact通过。全模型v11 batch1配对
正在运行，16worker已验证实际源码SHA与隔离快照一致，未覆盖正式v9通信后端。

v11运行中45秒资源采样：SM时钟2505–2625MHz，温度41–48°C，容器cpu.stat的
nr_throttled保持6。未观察到明显降频或新增CPU限流，不能据此排除全部系统扰动。
宿主有常驻RDMA监控进程占用CPU；本实验没有更改或终止它们。网卡普通计数器
2秒快照仅eth0少量流量，不能用此断言RDMA链路完全无流量。
资源数据保存在`v11-model-resource-samples.jsonl`。

下一个v12候选仅将dispatch的CTA归属从接收GPU改到发送GPU，进行本地读、
远端写；保持原v9的设备同步、相同选树和相同combine归约。用于验证PCIe
传输方向能否改善执行效率；当前仅生成候选源码，须等待v11退出后依次验证。

v11完整模型300步已结束：native P50/P95=244.358/246.609ms，匹配NCCL=
540.013/572.594ms，Relay=518.399/919.260ms。Relay尾漂移17.80%、块CV10.44%，
NCCL尾漂移3.492%也略超阈值。IPC event候选未改善长尾，不采用到正式后端。
原始SHA256=`9f930b8ba3682b0e9de716dc61b0727f52202a83ccb7b446ee1340a591ceb856`。
模型图另存`model-v11-figures`，未与v9混合；v12已启动BF16正确性验证。

v12通过18个BF16用例与64次动态尺寸/空流量连续调用exact验证。后者加入逐token
标识和按rank加权的精确整数贡献，可检查错位、漏传、残留与buffer复用。
native SHA=`1d223bf4c7a91ae2431498e602469c35b8eb7ef30c6e3b81431b068a44e7d7d2`。
三个尺寸pilot正在运行；仍为隔离候选，当前正式后端保持v9。

v12 pilot完成T2/91/728：Relay完整P50=0.677/4.099/26.614ms，匹配NCCL=
0.661/5.353/38.871ms。T728在200步四方法均通过稳定性，小/中消息300步仍有
部分不稳定。相对v9无一致净改善，因此v11/v12均不采用；v9通信源码SHA与
冻结source-v9完全一致。结束本轮候选探索，开始补齐正式同版本覆盖。

`final-v9`第一轮seed123已启动：尺寸T1/5/21/84/335/670，Zipf0.4/0.6/0.8，
top1 T1/670，共11配置，五方法完整配对；20warmup+100实测，必要时延长至300。
这批包含较慢的仓库CollWay-like，单点step限制不变；之后补真实路由和候选
收益配置的其余四seed，不把旧版本数据拼成同一正式曲线。

用户随后明确要求停止极致优化并对单机16卡阶段收尾，范围据此更新：完成当前
seed123的11配置、三个静态真实路由回放和v9 batch4；不再追加参数加密和其余
四seed。多seed覆盖不足保留在结论限制中，v8五seed数据单独归档。收尾期间不改
通信算法，阶段结束后确认任务GPU进程退出，代码不push。

收尾检查：v9 T1/5/21/84/335均完成300步，T670运行中。T335完整通信P50：
NCCL17.296ms、CollWay-like537.220ms、Relay12.794ms、Direct8.933ms、
Independent10.784ms。前五档原始文件已同步本地并逐文件SHA256验证，生成独立
v9预览曲线。真实route的16文件、48层共768个rank/layer payload通过shape、
expert范围、top-k唯一性及权重和检查；服务器保存`routing-manifest.json`。

收尾合成11配置已全部结束：尺寸六档各300步；Zipf0.4/0.6各200步且五方法均
稳定，Zipf0.8与两档top1各300步。T670完整P50：NCCL36.502ms、
CollWay-like2196.504ms、Relay26.164ms、Direct19.477ms、Independent21.933ms。
Zipf0.4/0.6/0.8 Relay相对NCCL完整P50加速约1.321/1.269/1.162x；两个
相同执行器消融仍更快。原始11文件已同步并校验，分析及六档payload图独立保存。

静态真实route三层均完成300步：第0/24/47层Relay完整P50约1.335/1.664/1.497ms，
对NCCL约1.104/1.034/1.025x；仅第24层五方法均满足稳定性。原始文件与图保存于
`trace-v9`。这三份路由来自batch1/512-token真实prefill，每rank32tokens，
不是独立动态路由重复。v9 batch4/8192+1已串行启动，16worker实际加载源码校验通过。

v9 batch4已完成20warmup+300测量：native P50/P95=968.987/983.936ms，
匹配NCCL=2070.679/2135.983ms，Relay=2009.590/3027.357ms。Relay P50降低
2.950%，P95增加41.731%，均值2144.551ms vs NCCL2066.840ms（高3.76%）。
Relay尾漂移8.340%、窗口CV4.515%；前者不通过，因此不称为稳定端到端胜出。
32份逐层原始记录已写出并汇总。所有测量进程正常退出，nvidia-smi计算进程查询为空。

按用户最新收尾范围，11合成配置、3静态trace、v9 batch1/4及图表/复现材料完成；
`summarize_closeout.py`通过221项覆盖与数据核对。剩余四seed、密集尺寸扫描、
跨节点/DeepEP与logits明确延期，不再自动启动单机实验。最终报告为
`docs/single-node-16gpu-results.md`，源码及审计归档在`artifacts/single-node-16gpu/closeout`。
