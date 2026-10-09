# 长期动作价值估计：文献筛选与改进建议

检索日期：2026-10-08。本文是研究建议，不是已实现的算法，也不代表已证明改进有效。没有修改算法、环境、奖励或正在运行的训练。

## 问题与现有证据

建议问题表述：在有限日内时域中，Critic 对动作所引起的剩余日累计成本差估计不准确，可能误导策略更新。准确的总 Q 值、正确的动作排序和正确的局部动作梯度是相关但不同的目标。

此前已验证的动作排序错误来自完成训练的 Dueling 模型。普通 MATD3 仍需独立重复诊断，不能把 Dueling 的错误率直接作为普通 MATD3 的结论。ESS 样本中 158 对日成本差大于 0.1 的动作，Q1 排序错误 68 对；包含即时成本方向与剩余日成本方向相反的案例。该证据支持研究问题，但不唯一识别根因。

本地证据：[验证记录](../day_value_verification_20261008/README.md)。

## 发表出处与分区口径

期刊分区按 JCR/JIF 口径。IEEE 2025-08 官方名单（注明数据基于 2024 JCR、2025-06 发布）将 TPAMI 与 TNNLS 均列为 SCIE、Q1；不是中科院分区，也不声称是最新 2026 分区。

来源：[IEEE 官方名单，第 3 页](https://open.ieee.org/wp-content/uploads/IEEE-Title-List-August-2025.pdf)。会议论文按已核实的正式发表出处列出，不套用 SCI 期刊分区。

## 重点参考文献

| 文献与出处 | 可借鉴内容 | 适用边界与阅读状态 |
|---|---|---|
| Buckman 等，**Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion**，NeurIPS 2018 | 用不同长度的模型 rollout 构造价值目标，并根据集成估计的方差混合目标 | 已读正文第 2–3 节。原方法同时集成动力学、奖励和 Q；方差低不保证偏差低，论文也指出无法可靠估计偏差项。不是直接保证动作排序正确。[正文](https://papers.neurips.cc/paper_files/paper/2018/file/f02208a057804ee16ac72ff4d3cec53b-Paper.pdf) |
| Park、Lee，**Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning**，ICLR 2025 | 在模型轨迹上使用多步 λ-return，降低对一步 bootstrap 的依赖 | 已读摘要、引言与方法前段。原方法针对离线 RL，并包含保守 expectile 回归及 Actor 改动；本项目只能借鉴多步目标思想，不能直接照搬保守项或宣称原算法适用于在线 MATD3。[会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f01e6d46cea7a214578af1d46abaa703-Abstract-Conference.html) |
| Pan 等，**Direct Advantage Estimation**，NeurIPS 2022 | 从轨迹直接学习动作优势，关注动作对回报的作用，而不是先精确拟合总 Q | 已读正文引言、背景与优势讨论。原实验是 on-policy PPO 与离散控制；不是连续动作、off-policy MATD3 的直接替换件。[会议页](https://papers.neurips.cc/paper_files/paper/2022/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html) |
| Duan 等，**Distributional Soft Actor-Critic with Three Refinements**，IEEE TPAMI 47(5), 3935–3946，2025 | 分布式 Critic、期望值替代、双价值分布与方差相关梯度调整，改善估计和训练稳定性 | 已读作者全文主要方法与消融。原方法是最大熵 SAC，采用高斯回报分布；其回报方差不能直接视为模型知识不足。改成 MATD3 分布式 Critic 属于新适配，不能沿用原文保证。[作者全文与期刊信息](https://arxiv.org/abs/2310.05858)，[正文](https://arxiv.org/html/2310.05858v7) |
| Kuznetsov 等，**Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics**，ICML 2020 | TQC：分位数回报分布、截断与集成，细粒度控制高估 | 已读摘要与正文引言。主要解决高估，截断可能加大低估；目前本项目尚未证明高估是主因。[会议页](https://proceedings.mlr.press/v119/kuznetsov20a.html) |
| Bhatt 等，**CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity**，ICLR 2024 | 通过 Critic 归一化和相应更新结构提高样本效率，保持较低更新次数 | 已读正式论文摘要与有关归一化、结论段。原方法还移除目标网络，不能仅插入 BatchNorm 就称为完整 CrossQ；并非专门解决长期回报传递。[会议页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html) |
| Arjona-Medina 等，**RUDDER: Return Decomposition for Delayed Rewards**，NeurIPS 2019 | 学习整段轨迹回报并重新分配奖励，改善延迟奖励的信用分配 | 已读论文摘要与引言、作者说明。本项目小时奖励并不稀疏，而且奖励重分配属于单独的跨算法奖励研究，不作为第一版 Critic 改进。[论文](https://papers.nips.cc/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf) |

补充参考：

- Duan 等，**Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors**，IEEE TNNLS 33(11), 6584–6598，2022；在线发表 2021。DSAC-T 的前身。[作者论文与发表信息](https://arxiv.org/abs/2001.02811)。
- Munos 等，**Safe and Efficient Off-Policy Reinforcement Learning**，NeurIPS 2016。Retrace(λ) 提醒我们，多步轨迹使用旧行为策略时需要处理策略不一致。原重要性权重涉及策略概率，不能直接套在确定性 MATD3 的 Dirac 策略上。[会议页](https://proceedings.neurips.cc/paper/2016/hash/c3992e9a68c5ae12bd18488bc579b30d-Abstract.html)。
- Zerouali、Tweed，**Error bounds and dynamics of bootstrapping in actor-critic reinforcement learning**，TMLR 2023。论文摘要区分价值误差与动作梯度相关的差分误差。此次 OpenReview 页面需要浏览器验证；仅核实搜索索引中的论文摘要和作者代码说明，未据此声称读完全文，也未标注 SCI 分区。[论文](https://openreview.net/forum?id=QCjMJfSnYk)，[作者代码](https://github.com/mathfrak-g-hat/Error_Bounds_Dynamics_Bootstrapping_ACRL)。

## 建议优先级

### 首选：有限时域多步回报校准

以普通 MATD3 为基础，在独立算法包中改变 Critic 的监督目标。借鉴 STEVE 的多时域价值展开和 LEQ 的 λ-return 思想；第一版不同时叠加 Transformer、分布式头、Actor 模仿或奖励重分配。

直观过程：从训练状态出发，执行待评估的联合动作，然后由同一版本的后续策略继续运行。得到 1、3、6、12 小时及剩余日终的回报目标；比较固定多步目标与多时域混合，而不是先凭两个 Critic 一致就认为目标可靠。

示意目标（h 不超过剩余小时数）：

\[
y_t^{(h)}=\sum_{k=0}^{h-1}\gamma^k r_{t+k}
+\mathbf 1_{t+h<T}\gamma^h\bar Q(s_{t+h},\bar\mu(s_{t+h})).
\]

到日终的目标没有 bootstrap 尾项，包含日终环境本身的费用与惩罚。本项目 γ=1、有限 24 小时时域；不能使用 γ<1 的无限时域误差收缩论证来保证这里短 rollout 必然降低误差。

建议先利用已能分支回放的训练环境，减少新增学习动力学模型的误差。该方案属于模型辅助训练；Actor 执行时可保持原接口，但不能把训练方法称为纯 model-free。它估计的是某一后续策略的回报，不是 MILP 最优回报。

实现前必须明确：后续采用当前策略还是目标策略、是否含目标平滑噪声，并保持各监督目标的策略定义一致。直接累加旧回放轨迹的奖励不等于当前策略回报。若标签缓存，需记录策略版本并刷新过期标签。

环境状态恢复、训练数据访问和仿真成本都需要明确计入实验：只使用训练 split；不读取测试日来生成训练标签；所有分支独立复制状态，不影响主训练轨迹；额外仿真步数、标签生成时间与总训练时间单独记录。相同主训练轮数之外，还应比较相同总仿真预算或相同墙钟时间下的结果。

这套目标作用于联合动作和团队回报，ESS、磨机、空调和 EV 使用同一个学习机制。第一阶段是目标校准实验；是否值得加入自适应时域权重，应由固定时域消融决定。

### 第二方向：动作差值/优势监督

DAE 提供相关理论动机，但其原实现不能照搬。可研究在同一状态、同一后续策略与相同外部条件下，两动作的回报差是否能更好地约束 Critic。

本仓库 `matd3_dr_counterfactual` 已经有剩余日 rollout、Critic 回报差监督及 Actor 输出监督，范围主要是 DR。故“加反事实差值损失”不是全新的项目方法，应作为现有对照或统一方案的消融，不能重新包装成首次提出。该方法是否有效仍需结果验证，不能从存在代码推断。

### 第三方向：分布式 Critic

DSAC-T/TQC 可以研究分布式目标、噪声与训练稳定性。但分布预测更丰富不自动意味着动作排序更准确；在固定外部轨迹和确定性策略下，条件回报分布可能很窄。适合在校准目标之后单独评估，避免一次更换结构和目标而无法归因。

## 验证与对照

1. 固定普通 MATD3、数据划分、环境、奖励、动作映射及训练预算。作者实现与本地 Actor 联合动作构造的差异另做对照，不与本次 Critic 改进一起改变。
2. 首先使用相同普通 MLP Actor/Critic，比较一步、固定多步、多时域回报校准；随后才比较与 Dueling 或分布式结构组合。
3. 至少 3 个随机种子作为初步筛选，条件允许使用 5 个；使用相同验证场景选模型，正式测试只报告冻结模型结果。
4. 除日均成本与任务/舒适约束外，报告动作回报差误差、显著动作差上的排序错误率、即时与日内效应方向相反的子集，以及额外仿真和运行时间。
5. 测试诊断状态独立于训练标签；过滤投影后动作相同的候选。既在公共固定状态集比较，也在各模型自身访问状态上比较。有限差分方向还需多方向、多尺度检查，不能用单个候选对等同于完整梯度验证。
6. 只有成本与排序错误共同改善，才有依据认为方案缓解了目标问题；仅 loss 下降、Q1/Q2 更一致、或成本下降均不足以单独完成机制解释。

研究新意须另做相关工作对照。使用多步回报、集成或优势监督本身已有大量研究；本项目的有限日内时域适配、统一机制和机制验证可以作为研究内容，但目前不声称组合已经具有可发表的新颖性。
