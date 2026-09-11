# MATD3-ShapleyQ Independent Baseline 修复设计

## 目标

完成当前尚未收尾的 `matd3_shapley_q` 算法，使 Shapley-Q 只分解长期合作增量价值：

```text
Q_coop = Q_grand - Q_ind
sum_i Q_phi_i = Q_coop
Q_grand = Q_ind + sum_i Q_phi_i
```

共享环境继续只输出一个代表大联盟整体表现的 `team_reward`。Shapley value 仍然只用于长期信用分配，不会注入即时的逐智能体 reward。

## 修改范围与不变量

本次修复以现有实验算法包为基础，只增加必要的共享算法生命周期钩子，并保持以下设计不变：

- 保留 3 个去中心化 Actor；
- 保留 3 组 centralized Twin Marginal-Contribution Critic；
- 保留 3-agent exact Shapley predecessor enumeration，以及权重 `1/3, 1/6, 1/6, 1/3`；
- 保留 `single_dispatch` 动作语义；
- 不属于 coalition 的微网仍然进行本地运行；
- 采用 hard terminal SOC，且 `SOC_start = SOC_end = 0.1`；
- 继续使用共享的训练、验证、正式测试、结算和绘图流程。

不会修改 `algorithms/matd3/` 下的 canonical MATD3 实现。

## Independent Baseline

`IndependentBaselineEvaluator` 是固定的反事实评估器。它使用与 Grand Coalition 环境相同的物理配置，唯一差异是关闭所有跨微网 P2P 和 ESS cooperation。每个微网的本地新能源、负荷、本地 ESS 调度和主网交互仍然有效。

由于当前仓库中没有兼容的 no-cooperation MATD3 checkpoint，默认使用确定性的固定启发式策略。该策略没有可学习参数，因此不会接收梯度，也不能被 Shapley Actor 操纵。

继续支持可选的 `frozen_policy` 模式，但必须显式提供兼容 checkpoint。checkpoint 缺失、不兼容或内容不完整时，在算法构造阶段直接报错，不能静默保留随机初始化策略。加载后的所有参数均设置为 `requires_grad=False`，模型保持 evaluation mode，并在 `torch.no_grad()` 中推理。

每个 episode 中，Independent 与 Grand 分支必须使用完全相同的日期、seed、确定性设置、预测噪声设置、load/PV/wind 外生序列、价格上下文和初始 SOC。测试将比较完整场景输入，而不只是某一个微网的 load 数组。

评估器缓存每一步的 independent reward、profit 和用于诊断的 discounted return-to-go reference。未初始化或越界访问 baseline step 时直接报错，不再返回会掩盖问题的零值。

## Replay Buffer 与价值目标

Shapley replay transition 保存以下权威原始量：

- `team_reward`；
- `independent_reward`；
- `cooperation_surplus_reward`；
- observations、state、action、next observations/state 和 terminal flag。

Replay Buffer 强制检查：

```text
cooperation_surplus_reward == team_reward - independent_reward
```

如果 grand-decomposition diagnostic 使用 discounted independent reference，则该 reference 连同相同的 episode/time identity 一起存入 transition。

Shapley Bellman target 定义为：

```text
y_coop = r_coop + gamma * (1 - done) * sum_i min(Q_phi_i_A', Q_phi_i_B')
```

因此，应先对每个智能体的 Twin Q 取最小值，再对三个智能体求和。当前两组 global twin sum 都拟合 `y_coop`，不再拟合 Grand Coalition return。

Actor `i` 继续最大化自己的 `Q_phi_i_A`。Independent Baseline 的全部计算都位于 Actor 和 Critic 的梯度图之外。

## Characteristic Value 语义

Coalition value API 明确区分物理价值 `J(S)` 和归一化特征值：

```text
V(S) = J(S) - J(empty)
```

`J(empty)` 表示三个微网独立运行，不能被强制设为零。纯函数形式的 exact-Shapley 工具会对真实 coalition-value mapping 进行归一化，并根据边际差计算 contribution。由此可执行地验证 `V(empty)=0`、Shapley efficiency、无合作时 contribution 为零，以及存在合作收益时 contribution 总和为正。

`CoalitionAwareProjector` 只 mask cooperation channel。对于所有 coalition mask（包括空联盟），每个微网的 local dispatch channel 都必须保留。

## 诊断指标与输出

训练 episode 记录新增或补齐：

- Grand team return/profit；
- Independent Baseline return/profit；
- Cooperation Surplus return/profit；
- `shapley_q_mg1`、`shapley_q_mg2`、`shapley_q_mg3` 及其总和；
- Cooperation Surplus Q target；
- efficiency residual；
- 当存在真实 independent return-to-go reference 时，记录 grand-decomposition residual。

Validation 继续只依据确定性 Grand Coalition team profit 选择 `best_model.pt`，从而保证 checkpoint selection 始终围绕真实优化目标。

Formal test 使用匹配的日期和 seed，分别评估已选中的 Grand Coalition policy 与同一配置下的 Independent Baseline policy。每日 CSV 和 JSON summary 输出 Grand、Independent 和 Cooperation Surplus 结果。现有 daily coalition MILP evaluation 以及 Equal/Shapley settlement 继续作为独立的后处理流程，不能用于在线 baseline reward。

TensorBoard exporter 将新增字段映射到所要求的 reward 和 Q namespaces。训练终端 dashboard 显示关键的 Grand/Independent/Surplus 与 Shapley efficiency 指标，但不改变保存数据的语义。

只有存在真实 reference 的 residual 才允许输出。不能把恒等式计算结果伪装成 learned grand-decomposition residual。

## Checkpoint 兼容性

新 checkpoint 明确记录 cooperation-surplus objective，并保存 baseline mode/configuration metadata。旧的 `full_enumeration_v1` checkpoint 必须拒绝加载，因为其 Critic 目标可能仍在分解 `Q_grand`，而不是 `Q_grand - Q_ind`。

Checkpoint 加载仍然校验 action semantics、projection signature、state schema 和 observation schemas。

## 错误处理

以下情况必须尽早失败并给出明确错误：

- frozen baseline checkpoint 缺失或不兼容；
- baseline mode 不受支持；
- replay rewards 不满足一致性关系；
- Independent 与 Grand 场景不一致；
- 请求了不存在的 baseline step；
- checkpoint 使用旧版 Shapley-Q 目标语义；
- loss、gradient、reward 或 diagnostic 出现非有限值。

## 测试策略

先新增或强化测试并观察其因缺少目标行为而失败，再修改生产代码使其通过。测试范围包括：

1. 空联盟物理语义：local ESS 和 grid operation 仍然可用，所有 P2P flow 为零。
2. Characteristic normalization：实现基于非零 `J(empty)` 计算出 `V(empty)=0`。
3. Exact Shapley efficiency：contribution 总和等于 `J(grand)-J(empty)`。
4. No-cooperation case：所有物理 coalition value 相同时，surplus 和全部 Shapley contribution 为零。
5. Positive deterministic cooperation case：构造新能源富余卖方与高价缺电买方，使 surplus 及 contribution 总和严格大于零。
6. Frozen-baseline isolation：baseline 所有梯度均为 `None`；在能够激活更新的 batch 上，三个 Actor 均获得有限且非零的梯度。
7. Common scenario：timestamp、物理外生数组、price、forecast、forecast mask/noise、seed 和初始 SOC 完全一致。
8. Replay consistency 以及非法输入拒绝。
9. Cooperation target 的 per-agent Twin-min 顺序。
10. Training/formal-test 输出字段与严格 checkpoint compatibility。

## 验证与验收

在报告完成前执行：

1. 运行 MATD3-ShapleyQ 专属测试。
2. 运行全仓库 `pytest -q`。
3. 运行 1–3 天的 `scripts/run_experiment.py --algorithm matd3_shapley_q` smoke workflow，并确保足够早地启用学习，以实际覆盖 Critic 和 Actor update。
4. 确认选中了 `matd3_shapley_q` plugin 及 `configs/algorithms/matd3_shapley_q.yaml`。
5. 确认 formal test 使用 `best_model.pt`。
6. 确认 test evaluation、Equal/Shapley settlement、summary 和 saved-result-only plotting 全部完成。
7. 确认生成 15 个 PNG 和 15 个 PDF。
8. 确认结果位于 `outputs/matd3_shapley_q/runs/`。
9. 确认 `algorithms/matd3/` 下没有任何文件发生变化。

