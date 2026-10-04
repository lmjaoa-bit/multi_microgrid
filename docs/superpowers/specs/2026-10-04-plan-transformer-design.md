# Plan-Transformer Actor：设计规格

- 日期：2026-10-04
- 算法标识：`matd3_plan_transformer`
- 显示名称：`Plan-Transformer`
- 设计起点：`fe64fa258b962b4048ca2b59f9a8a446a2934d63`
- 状态：聊天中的架构已获用户确认；本文待书面评审，尚未实现。

## 1. 目标和已确认的边界

每个微网的独立 Actor 用 NOW query 对未来和 peer tokens 做两层 cross-attention，同时输出当前动作和 SOC/ESS 计划。计划监督要直接训练生成动作所用的共享注意力表征，帮助网络学习午间盈余、晚峰等跨时段结构。

用户已选择：计划标签来自**当前状态和当时可用预测生成的优化计划**，不是完整真实未来 Oracle，也不是 Replay 后续实际运行轨迹。

外部动作保持 `[ESS intent, DR intent]`，只执行当前动作头。辅助计划不发送给环境，不接管后续动作；每小时重新推理。两个输出头独立，因此本文称为“带辅助计划监督的策略”，不宣称动作天然等于某条计划的第一步。

物理约束、Reward V2、共享观测、场景拆分、清算和 Equal/Shapley 结算保持现有合同。保留 canonical MATD3、MATD3 dispatch 和普通 T；新变体不替换全局默认实验，不自动启动完整训练或结果晋升。

当前奖励按小时计算，并非全程稀疏奖励。辅助目标增加的是跨时段结构监督；时间信用分配继续由 Critic 的 TD 学习承担。学习效果和注意力解释性均需实验验证。

## 2. 代码核查结论

- `configs/cooperative.yaml` 已提供 H24、episode 外预测屏蔽、`peer_energy_summary` 和 Reward V2。
- 普通 T 目前使用 H8 Actor 和 self-attention，不在该包内改成新架构。
- `ObservationBuilder` 已有当前负荷、PV、wind、电价、SOC、peak、DR 状态、ESS 可行边界及 peer summaries。
- 当前 `solve_dispatch` 可传初始 ESS SOC，但缺少完整的中途 DR 库存/温度/EV 状态和历史 peak 接入。日级解还可能使用两维动作接口无法自由选择的交易安排。不能直接把它切片当滚动标签。
- 共享 Replay 接口只接收 transition；共享训练循环先 `env.step`，再 `replay.add`。可靠的教师状态必须在当前动作之前捕获，包含 warmup 阶段。
- HVAC 使用温度、太阳辐射和基荷计算热状态；未来 token 不能只保留 net load 而丢失 base load。
- EV cohort 在到达前隐藏会话实现值，训练教师也不得提前读取这些值。

## 3. Actor 和时间索引

三个 Actor 参数独立，执行时各自仅读取自己的现有局部观测；中心教师和 Critic 的当前联合状态不成为 Actor 的额外输入。

默认 `embed_dim=32`、`num_heads=4`、`num_layers=2`、`hidden_dim=64`、dropout 为 0；各值只写入新算法 YAML。

### 3.1 Token 合同

| Token | 内容 | 数量 |
|---|---|---:|
| NOW query | 当前 base/net load、可再生总功率、电价、碳因子、SOC、ESS 可行边界、peak、完整本地 DR 状态、小时编码和剩余步数 | 1 |
| Future memory | 本地 base/net load、电价、两个 peer 的 energy balance、HVAC 温度及温度存在位、有效位、相对小时编码 | 24 |
| Peer memory | energy balance、deficit、surplus、charge/discharge flex、previous P2P net | 2 |

net load 在还原各源的功率归一化后计算，不能直接相减使用不同尺度的归一化 load/PV/wind。peer 顺序、token 类型和相对时间编码固定并写入 checkpoint 合同。

两层 pre-LN cross-attention：NOW 为 query，24 个未来 token 和 2 个 peer token 为 key/value；每层含残差和 FFN。NOW 编码器允许本地 DR 类型不同造成输入维度不同。

无效 future token 在编码前清零，并使用 key padding mask。peer tokens 始终保留，使日末所有未来位置被屏蔽时仍有有效 memory。无效位置中的 NaN/Inf 不得污染输出；有效输入的 NaN/Inf 必须报错。

### 3.2 输出合同

- 公共 `forward(obs)` 和 `select_actions(...)` 仍只返回两维动作，范围为 `[-1,1]`。
- 算法内部 `forward_with_plan(obs)` 返回动作、`SOC_after[24]`、`ESS_net_kw[24]`，以及按需收集的 attention。动作头和计划头都从相同最终 NOW 表征读取，计划损失不绕过 cross-attention。
- ESS 计划的物理符号固定为正值放电、负值充电；SOC 计划使用步后状态。ESS 功率按设备额定上下界缩放，SOC 输出映射到设备 SOC 区间。边界内的网络输出并不自动保证整条预测计划可行。
- 当前时刻 `t=0..23`，计划槽 `j=0..23` 对应动作时段 `t+j` 和 SOC 边界 `t+j+1`；计划 mask 为 `t+j<24`。
- Future token `k=1..24` 对应外生时段 `t+k`，由现有 forecast validity 屏蔽。计划 mask 与 forecast mask 独立：在 `t=23`，计划第 0 槽有效，所有 future tokens 无效；在 `t=0`，计划有 24 个有效槽，但 `t+24` 的 forecast token 无效。

## 4. 可执行的预测优化教师

### 4.1 方法选择

采用有限预算的 CEM（cross-entropy method）在现有联合原始动作序列空间内优化。候选形状为 `[24-t, 3, 2]`，动作限制为 `[-1,1]`。这是工程方法的具体化，用户将在本文书面评审时确认。

每个候选都在独立的预测场景副本中通过现有 `env.step` 演算，复用现有设备、投影、分配、物理清算和 Reward V2。不得复制设备动力学或另写一套宽松交易约束。教师得到的是预算内的**可行优化计划**，没有全局最优性证书，禁止称为 MILP 最优解、成本下界或动作空间上界。

教师联合优化 ESS 和 DR；辅助标签只提取各微网的实际 ESS 净功率和步后 SOC。完整候选原始动作及 DR 轨迹保留为标签审计证据，不直接加入动作模仿损失。

默认候选 population=32、elite=8、iterations=3、初始标准差=0.5、最小标准差=0.05、均值/方差更新率=0.5，采用逐维高斯采样后裁剪到原始动作范围。初始均值为 0。始终保留已评估的最好候选；同成本时按固定候选序号选择。每次求解额外评估三个固定基准：全部 ESS intent 为 0，全部 DR intent 分别为 0、-1、+1；优化结果不得比其中最好的可行基准更差。随机数来自独立教师 RNG，不消费训练环境或探索 RNG。

默认每个训练日捕获 `t in {0,6,12,18}`，教师缓存最多 2048 个联合计划，标注样本池最多 2048 个上下文。达到上限后停止新求解并复用已标注池；不得在每个梯度更新内重新求解。这些预算是初始工程设置，不是已验证的有效超参数。

### 4.2 允许信息与预测场景

教师可读取训练时的**当前**联合 SOC、DR 状态、协调历史、peak 和当前观测；这是中心化训练监督。执行时 Actor 不读取 peer 私有 DR/SOC。该信息差需要在实验说明中披露，不能称为完全局部教师。

从时刻 t 冻结一次预测快照：当前外生量取当前观测，剩余有效时段取时刻 t 的 `SharedForecastProvider` 当前预测。后续模拟不得滚动读取原训练环境的未来预测行或真实外生数组。

- base load、PV、wind、买卖电价、温度、GHI 使用当时已有的本地/公共预测；记录使用的字段。GHI 等教师字段不会加入 Actor 的新观测输入。
- 未来碳因子没有公开预测，固定使用当前已观测值的 persistence 预测，禁止读真实未来碳序列。
- 日历、占用时段和设备参数使用已知配置。末端约束沿用 Reward V2，无新增 SOC 恢复要求。
- 已 revealed 的 EV cohort 使用当前公开的会话信息和剩余需求；已 departed 的组不再增加未来任务。
- 未 revealed 的 cohort 使用配置分布构造一个确定的名义会话：离散时刻取加权中位数（相等时取较早时刻），需求取范围中点；未来 arrival 还需条件化为严格晚于当前小时。departure/dwell 依配置相应分布确定。名义会话须满足同一会话物理检查；若条件分布为空或名义会话无效，此上下文拒绝标注，不能偷偷采用真实未来会话。
- 只能复制当前已公开的 EV 状态；不能把原环境中未 revealed 的 session、remaining、delivered、departure 缓存带入教师副本。

生成器先把预测场景物化成独立输入，再建立执行副本。原环境的完整未来 frame、forecast arrays、profile caches、设备 frame/base-load 缓存、未公开 EV sessions 和 RNG 均不得成为副本剩余时段的数据来源。所有用于 `env.step` 的入口都要绑定物化预测场景；不能只替换一个 frame 后让其他缓存继续读真实未来。

保留当前小时、SOC、silo、HVAC 双温度、revealed EV 剩余量、已发生 departure 记录、peak 和协调历史；历史服务成本是固定的已发生成本，教师优化从 t 起的增量成本。每个候选从相同状态开始，优化期间原训练环境必须完全不变。

### 4.3 标签、缓存与失败

每条联合标签保存：原始局部观测、全局状态、小时、计划 mask、各微网 SOC/ESS 标签、教师原始动作、预测场景 hash、当前状态 hash、环境/归一化合同 hash、教师版本/预算/seed、基准和最好候选成本、有效状态、拒绝原因及耗时。

缓存 key 包含完整当前状态、冻结预测、已公开 EV 信息、环境/预测规则和教师版本；不按日期或小时单独缓存。过期缓存拒绝使用。标签断开梯度，不能反向传播到教师求解。

无有效候选、预测不足、非有限值或物理复核失败时，本上下文不参与计划监督，RL transition 仍正常保留；记录拒绝原因。标签开关开启时若全程没有生成有效标签，验收失败，不能静默退化为无监督 Actor。

## 5. 共享接口和最小流程接入

公共 `MultiAgentAlgorithm` 和 `replay.add(...)` 参数保持不变，canonical MATD3 无修改。

在 `algorithms/base.py` 增加可选运行时协议 `TrainingContextConsumerProtocol`，方法为 `capture_training_context(env, observations, global_state, metadata)`。共享训练流程只在支持该协议时、每个动作之前调用，metadata 包含 episode index、total steps 和 reset 返回的 day；warmup 同样覆盖。新算法在该方法内核验 train split、选择标注时刻并物化教师输入，随后自己的 `replay.add` 消费对应上下文。严格校验上下文与 transition 的小时和状态 hash 相等，避免错配。

共享流程不读取 Actor、Critic 或计划内部；算法特有分支只存在新包内。其他算法不需要实现此协议，调用次数、动作、RNG 和状态保持相同。

教师只在训练上下文捕获时运行；`select_actions`、验证、学习曲线和正式测试均不运行优化教师。辅助标签使用独立池，不覆盖 RL Replay 的 actions、reward 或 next state。

## 6. 学习目标和梯度边界

保留普通投影 MATD3 的 Critic 类型、TD target、双 Critic、目标噪声、policy delay、gamma=1 和软更新。新包实现自己的 Actor 更新，不能被现有诊断安装器覆盖成不含辅助损失的更新方法。

当前 RL batch 计算 `L_RL=-Q1(s,project(a))`。独立标注池的 batch 计算计划目标，使用当时存储的局部观测；不把标签从同一天另一个状态拼到 RL 样本上。辅助 batch 默认 32，按独立 RNG 无放回采样；池不足 32 但非空时使用全部已有样本，池为空时只做 RL 更新。每次 delayed Actor 更新取一个辅助 batch，三个 Actor 使用同一组联合标签。

```text
L_actor = L_RL + lambda_plan * L_plan
                   + lambda_dyn * L_dyn
                   + lambda_first * L_first
```

- `L_plan`：SOC 和 ESS 功率的 masked SmoothL1，逐样本按有效步数归一化后再平均，三个 Actor 分别计算。SOC 使用实际比例，功率除以本设备最大充/放电额定功率。
- `L_dyn`：以输入当前 SOC 为第 0 边界，约束预测步后 SOC 与预测实际 ESS 功率符合现有容量、时间步、充放电效率；无效槽不参与计算。正功率放电时 `delta_SOC=-P*dt/(eta_dis*E)`，负功率充电时 `delta_SOC=-P*dt*eta_ch/E`。
- `L_first`：计划第 0 槽的 ESS 功率/SOC 与当前联合动作通过现有 `JointActionProjector` 得到的**实际**充放电功率及相应步后 SOC 一致，不能与原始 intent 或 requested power 直接比较。
- 更新 active Actor 时其他 Actor 的动作 detach，Critic 参数冻结；首步损失允许梯度进入 active 动作头和计划头。教师标签始终 detach。
- 标注池不足时只执行 RL 更新。初始权重为 `lambda_plan=0.1`、`lambda_dyn=0.01`、`lambda_first=0.05`，沿用 actor 最大梯度范数 1.0；只允许在 train/validation 调整。
- 无计划监督消融把三个权重均设为 0，并禁用教师捕获/求解；网络结构、初始化 seed 和 RL 超参数保持一致。

记录每个 Actor 的分项损失、标注数量/拒绝率、有效槽数、计划 MAE、首步误差、共享注意力的 RL/辅助梯度范数和裁剪比例。梯度比较用于诊断监督是否实际到达注意力，不预先声称辅助梯度一定改善策略。

## 7. 保存、诊断和结果布局

新 checkpoint 保存 Actor 的动作头/计划头、target Actor、现有 Critic、训练合同和辅助配置；load 在改变模型状态前检查 slug、schema、normalizers、预测/peer/mask 顺序、功率符号、时间索引、reward 和 action semantics。推理加载不依赖教师池或教师 RNG。

普通结果仍写入 `outputs/working_runs/matd3_plan_transformer/runs/<run_id>/`。教师证据使用该 run 内 `auxiliary_plan/`，不可写到共享全局目录。训练若需要续跑，教师 RNG/预算计数/缓存和标注池必须连同 RL Replay 通过现有可恢复算法协议保存；首版不支持续跑时，CLI 要明确拒绝该变体的 resume，不能宣称可精确恢复。

为保存正式测试的计划与注意力，在 `algorithms/base.py` 增加可选 `AlgorithmDiagnosticsProtocol`：`configure_diagnostics(directory, metadata)` 配置本 run 的绝对输出路径和身份；`set_diagnostic_phase(phase, metadata)` 启停记录，`phase` 仅在正式测试为 `formal_test`，其他阶段为 `None`。共享流程在正式测试进入/退出时显式调用，并在退出的 `finally` 中停用； metadata 包含测试 checkpoint 路径和测试阶段标识。算法在原有 `select_actions` 同一次 forward 中保存动作、计划、mask、token 时间和每层每头 attention；不再次推理，不运行教师，不改变返回动作。验证和训练不混入正式测试文件。共享流程只能调用公开可选接口，不能读取模型私有结构。

现有 `evaluation/plotting.py` 保持 saved-result-only 规则。保留标准 `figures/` 中 15 PNG + 15 PDF，额外解释图放 `figures/plan_diagnostics/`，从 `auxiliary_plan/` 和正式测试诊断文件读取；不得在绘图期间调用环境、策略或 MILP。

热图按微网、层和 head 保存，横轴是真实相对/绝对小时；屏蔽格不能显示成低关注的有效小时。显示计划和实际执行轨迹的差别，避免把预测 SOC 当真实 SOC。

## 8. 文件范围

初始包必须由仓库 scaffold 命令创建；本阶段只写设计，不执行 scaffold。

```text
algorithms/matd3_plan_transformer/
  __init__.py
  plugin.py
  algorithm.py
  actor.py
  replay_buffer.py
  plan_teacher.py
  forecast_scenario.py
configs/algorithms/matd3_plan_transformer.yaml
configs/smoke/matd3_plan_transformer.yaml
tests/test_plan_transformer_actor.py
tests/test_plan_teacher.py
tests/test_plan_transformer_pipeline.py
```

`plugin.py` 只暴露一个 `ALGORITHM_SPEC`，使用自己的 slug 和 YAML；按名称加载，不增加全算法中心 import。不从普通 T 实验包继承限制 H8 的构造流程。复用现有通用 Critic、Replay/运行时工具和 projector；所有新覆盖属于新包。

允许的共享改动仅包括：可选训练上下文/测试诊断协议及流程调用、现有 saved-result plotter 的诊断文件读取。禁止修改 `algorithms/matd3/`、其他算法网络、物理规则、环境配置、奖励配置或结算代码。

## 9. 验收和实验解释

### 工程验收

1. Actor 输出 shape/range、H24 和 peer 顺序、时间索引以及 checkpoint round trip 通过。
2. 有效位屏蔽测试覆盖日初、日末、部分预测、全部预测无效，以及无效格中的 NaN/Inf；非法有效输入拒绝。
3. 辅助损失单独反传时两层 cross-attention 均得到有限且非零梯度；动作与辅助开关不改变外部接口。
4. 教师因果测试：保持当前已公开状态和 t 时预测相同，改变真实未来外生值、未来 forecast 行和未 revealed EV 会话，标签/hash/教师候选结果不变；改变 t 时预测应影响教师场景。
5. 教师样本复核：原环境、物理状态、预测缓存和 RNG 前后不变；相同教师 seed 可复现；候选实际 ESS/SOC 满足物理关系、边界、清算平衡和 Reward V2 成本重算；不声称最优证书。
6. warmup 上下文、transition 标签关联、缓存失效/预算/拒绝和无监督消融通过；验证/测试教师调用数为 0。
7. 使用本插件及自己的 smoke YAML 完成 1–3 个训练日和 1–3 个验证/测试日；smoke 的 warmup/batch/教师预算必须确保至少一次真正的 Actor 辅助更新，不能只验证随机动作流水线。
8. 运行 `pytest -q`；确认 formal test 使用 `best_model.pt`，七 coalition MILP、Equal/Shapley、summary、标准 15 PNG + 15 PDF 均完成；检查额外诊断来源为正式测试且绘图只读保存文件。
9. 正常结果在新 slug 的 working run 内；确认 canonical MATD3 和现有算法无代码变化。不自动晋升，不删除现有结果。

### 效果验证

第一项对照是同一个 H24 cross-attention Actor 的辅助监督关闭/开启；普通 T H8 和 MATD3 提供背景基线。使用相同场景和多 seed，分别报告广义系统成本、服务损失、计划误差、物理可行性、教师成本与训练耗时。首版短 smoke 只能证明工程接入，不证明收益或收敛。

关键时段扰动是单独、可标识的诊断研究，不替换正式测试：移动预测晚峰、改变午间盈余或对关键 token 做受控干预，保持当前物理状态不变，检查注意力、计划及当前动作的联合响应。热图只作为观察证据；没有干预对照时不称为因果解释，也不能由漂亮热图推断优化目标已学会。

## 10. 评审结论与下一步

此规格补齐了当前信息边界、计划/forecast 不同时间索引、真实可执行教师、共享流程的最小可选接入和首步物理一致性。CEM 教师、预算、未知 EV 的名义预测规则及可选协议是需要用户审阅的具体工程决定。

书面规格获确认后，再写实施计划并选择执行方式；本文件不授权自动开始完整训练、测试集调参、替换默认算法、提交其他现有工作或晋升 release。
