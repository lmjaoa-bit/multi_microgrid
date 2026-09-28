# 普通 T 物理功率特征修正实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本文仅为交接方案；本次编写不包含代码实施或训练。

**Goal:** 修正普通 T 将不同尺度归一化的负载、光伏、风电直接相加减的问题，使当前供需差、供需方向和未来净负载具有正确物理语义。

**Architecture:** 保留共享观测及其分字段归一化，在普通 T Actor 内部用观测构造器提供的尺度恢复功率，再以一个统一功率尺度缩放合成特征。当前和预测特征共用同一计算函数；通过普通 T 专属 checkpoint 元数据阻止新旧语义混用。

**Tech Stack:** Python、PyTorch、YAML、pytest，现有共享训练与评估流程。

**Spec:** 本文“设计约定”部分为完整设计依据；仓库 AGENTS.md 为全局约束。

## Global Constraints

- 这是现有普通 T 的特征语义修复，保留 slug `matd3_hierarchical_transformer` 和显示名“普通 T”，无需创建新算法变体。
- 不修改 `algorithms/matd3/`、`algorithms/matd3_dispatch/`、共享观测、环境、奖励、动作投影及结算实现。
- 不同时调整学习率、warmup、探索噪声、网络层数、token 数量或预测时域；普通 T 仍使用 8 小时预测。
- 保留全部旧结果和 checkpoint，不覆盖、删除或重新解释旧运行；修复后从头训练并生成新 run。
- 使用各算法自己的 YAML，通过 `scripts/run_experiment.py` 运行，结果进入 `outputs/working_runs/<slug>/runs/`。
- 测试与绘图沿用共享流程；绘图只读保存文件，不额外运行环境或 MILP。

## Review Focus

- 缺失、非正或非有限归一化尺度必须显式报错，不能默认为 1；见任务 1 的参数校验测试。
- 无光伏/无风电微网和恰好平衡的样本应正确计算，不能因人为分母下限改变符号；见任务 1 的边界测试。
- 预测超出 episode 的有效性 mask 保持现有行为，不能把无效预测计入误差统计；见任务 1 的 mask 回归测试。
- 非训练 split 继续使用训练集统计的尺度，不以测试集重新拟合；见任务 2 的尺度来源测试。
- checkpoint 的版本或尺度不匹配时，必须在任何网络状态加载之前失败；见任务 2 的兼容性与状态不变测试。

## 设计约定

### 已确认的问题

`core/observation.py` 分别输出 `L / S_L`、`PV / S_PV`、`W / S_W`。
`core/profiles.py::normalizers` 的三个分母分别来自负载训练集最大值、光伏装机容量、风电装机容量，并不相同。

`algorithms/matd3_hierarchical_transformer/actor.py::_build_tokens` 当前直接计算：

```python
base_balance = pv + wind - load
balance_sign = torch.sign(base_balance)
f_net_load = f_load - (f_pv + f_wind)
```

这些表达式不是物理净功率。当前 token 还保留了独立的三个输入，但未来 token 将三个功率通道压成了这个错误组合。

### 修正公式和特征范围

唯一尺度来源：`env.obs_builder.norm[mg]`，使用键 `load`、`pv`、`wind`、`power`。不要从 YAML 的平均负载、当前 batch、测试数据或全局状态尺度另行推断。

设输入已分别归一化，统一净负载定义为：

```python
net_load = (load * S_load - pv * S_pv - wind * S_wind) / S_power
```

- 当前供需差：`base_balance = -net_load`，正值表示基础发电盈余。
- 当前方向：`balance_sign = torch.sign(base_balance)`，精确平衡时为 0。
- 未来净负载：同一函数用于每一个预测步，正值表示基础用电缺口。
- 当前 supply token 输入维度仍为 5：原有三个单独归一化通道 + 修正后的供需差 + 方向。
- forecast token 输入维度仍为 6，其他价格、温度、有效性、时间特征保持原样。
- 统一尺度采用已有 `norm['power']`，不另改它的定义，不将结果 clamp 到 [-1, 1]。
- 上述负载是现有观测中的基础负载，不包含动作执行后才确定的可控负载。注释应明确写“基础供需差/基础净负载”，本次不增加 DR 功率估计。
- 不改变 token 顺序、网络权重形状、padding mask、观测维度及动作维度。

### Checkpoint 约定

引入普通 T 专属语义标识 `physical_power_balance_v1`。旧 checkpoint 即使权重形状相同，也不能在修正后的特征下静默加载，否则行为已经改变。

新增普通 T checkpoint 字段：

```text
actor_feature_semantics: "physical_power_balance_v1"
actor_feature_normalizers:
  mg1: {load: ..., pv: ..., wind: ..., power: ...}
  mg2: {load: ..., pv: ..., wind: ..., power: ...}
  mg3: {load: ..., pv: ..., wind: ..., power: ...}
```

新模型在 `load()` 中先检查版本及各 agent 的四个尺度，然后执行已有语义/schema 检查，最后加载权重。尺度按 float64 数值比较，`rtol=1e-12, atol=1e-12`；缺字段、agent 不匹配或数值不匹配均报错。

尺度 tensor 注册为 `persistent=False` buffer，随 `.to(device)` 迁移但不被 checkpoint 权重覆盖；可信尺度来自当前环境，checkpoint 元数据负责兼容性校验。

不在本次修复中增加旧版兼容运行模式。重现旧结果应使用旧代码提交 `950f0dd91f5697716c159843b8273c325d00d406` 和原 checkpoint。

## Task 1：修正 Actor 内部当前与预测净功率

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer/actor.py`
- Modify: `algorithms/matd3_hierarchical_transformer/algorithm.py`（构造参数传递）
- Modify: `tests/test_matd3_hierarchical_transformer.py`（更新直接构造调用）
- Create: `tests/test_plain_t_power_features.py`

**Interfaces:**
- Actor 新增必填 keyword-only 参数 `power_normalizers: Mapping[str, float]`。
- 新增 `_physical_net_load(self, load: torch.Tensor, pv: torch.Tensor, wind: torch.Tensor) -> torch.Tensor`，支持 `[B, 1]` 和 `[B, H, 1]`。
- 在 Actor 初始化时校验四个尺度存在、有限且大于 0，保存有明确键顺序的 float32 buffer。
- Algorithm 创建每个 Actor 时显式传入 `env.obs_builder.norm[mg]`；target Actor 的 deepcopy 自动保留尺度。

- [ ] 先写会在旧实现失败的物理语义测试，使用固定尺度 `load=100, pv=500, wind=1000, power=1000`。
- [ ] 当前反号案例：物理 `L=50, PV=100, W=0`，输入 `(0.5, 0.2, 0)`；断言供需差为 `+0.05`、符号为 `+1`，而非旧值 `-0.3`。
- [ ] 预测案例：每步物理 `L=120, PV=40, W=30`，断言净负载为 `+0.05`；覆盖多 batch、多预测步且不同步数值不同。
- [ ] 重标度不变性：同一物理样本换一组单字段分母并相应重编码输入，保持 `S_power` 不变，断言合成净功率不变。保留独立输入通道的原始归一化方式。
- [ ] 覆盖零发电、只风电、只光伏、零净负载，以及缺键、零/负数、NaN、Inf 的参数拒绝。
- [ ] 用 encoder 的 forward pre-hook 检查真正送入 `supply_encoder` 和 `forecast_encoder` 的 tensor 数值；不能只测辅助函数而遗漏接线错误。
- [ ] 保留已有 shape、动作范围和梯度测试；新增断言无效预测 mask、token 数量及顺序不变。CUDA 可用时验证 buffer 迁移和同输入 CPU/GPU 一致性（允许浮点误差）。
- [ ] 运行定向测试并记录预期失败：`python -m pytest -q tests/test_plain_t_power_features.py`。
- [ ] 按上述接口实现尺度传递和统一公式，将两处物理运算接到同一函数；更新相关物理语义注释。
- [ ] 运行：`python -m pytest -q tests/test_plain_t_power_features.py tests/test_matd3_hierarchical_transformer.py`，全部通过。

## Task 2：版本记录与 checkpoint 兼容性保护

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer/algorithm.py`
- Modify: `configs/algorithms/matd3_hierarchical_transformer.yaml`
- Modify: `tests/test_plain_t_power_features.py`
- Modify: `tests/test_matd3_hierarchical_transformer.py`

**Interfaces:**
- Algorithm 类常量 `ACTOR_FEATURE_SEMANTICS = 'physical_power_balance_v1'`。
- 普通 T YAML 中 `algorithm.network.feature_semantics` 设置为该值；构造时默认该常量，显式传入其他值则报错，不自动选择旧公式。
- `save()` / `load()` 使用前述两个新增 checkpoint 字段；现有 `action_semantics`、`projection_signature` 等字段继续表示原有含义。

- [ ] 写新 checkpoint 往返测试：加载前后相同观测的动作一致，并验证两个新增字段存在。
- [ ] 写缺失版本、错误版本、缺失尺度、尺度不匹配和 agent 不匹配的拒绝测试；失败后模型参数必须与加载前一致。
- [ ] 写尺度来源测试：配置使用同一训练数据，train/val/test 的观测尺度应一致；Actor buffer 与其自身 `env.obs_builder.norm` 对应。
- [ ] 实现版本记录、普通 T YAML 标识、初始化校验与预加载校验。快照已有保存机制自然记录 YAML 标识，不为此修改共享 pipeline。
- [ ] 检查现有恢复训练能力：若普通 T 没有恢复接口，保持现状，不扩展；若有绕过 `load()` 的独立恢复入口，应在普通 T 本地增加等价版本/尺度保护并测试，禁止旧状态跨语义恢复。
- [ ] 重跑上述定向测试，确认新版本可加载、旧版显式拒绝。

## Task 3：完整验收与受控重训交接

**Files:** 仅生成标准工作结果及实施验收记录，不修改历史结果。

- [ ] 运行 `python -m pytest -q`，记录完整结果；若失败，区分本次回归与已有失败，不宣称全部通过。
- [ ] 为每个待 smoke 的算法从其自己的正式 YAML 复制一份到系统临时目录；仅为 smoke 设置 `algorithm.batch_size=8`、`warmup_steps=8`、`replay_capacity=256` 和 `logging.diagnostic_interval=1`。普通 T 临时配置保留新版本标识；不修改正式长训练超参数。
- [ ] 两种算法分别通过下述入口运行 3 个训练日、2 个验证日、2 个测试日；`<临时配置路径>` 替换为对应算法的完整临时 YAML 路径：

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer --algorithm-config <普通T临时配置路径> --episodes 3 --validation-interval 1 --validation-days 2 --test-episodes 2 --no-dashboard
python scripts/run_experiment.py --algorithm matd3 --algorithm-config <MATD3临时配置路径> --episodes 3 --validation-interval 1 --validation-days 2 --test-episodes 2 --no-dashboard
```

- [ ] 不加 `--no-milp` 或 `--no-validation`；确认确实发生网络更新，普通 T 的 checkpoint 包含版本及尺度元数据。
- [ ] 确认 plugin slug、各自 YAML、`best_model.pt` 的实际正式测试加载记录、测试完成、7 联盟 MILP、Equal/Shapley 结算、summary 和绘图均完成。
- [ ] 每个 smoke run 检查 `15 PNG + 15 PDF`，路径为 `outputs/working_runs/<slug>/runs/<run_id>/`；无需测试 release promotion。
- [ ] 在旧测试集基础外生序列上只计算特征，验证修复公式与物理基础净功率一致；符号统计排除绝对物理差值不超过 `1e-5 kW` 的浮点平衡样本。此检查无需重新执行旧策略或 MILP。
- [ ] 检查 diff，确认 canonical MATD3、普通 MATD3 dispatch、共享物理/观测/奖励和历史结果均无修改。
- [ ] 交付修改清单、测试结果、两个 smoke run 路径，以及新旧 checkpoint 的明确兼容性说明。

### 性能验证属于后续实验，不是本次代码正确性的前提

完成上述验收后，若安排长训练：先使用原有正式普通 T 超参数、seed 42、5000 回合和相同 55 个测试日，从头训练一个新 run。只改变此次特征修正，不改变 8 小时预测时域等因素。

重点观察 MG1 充/放电小时覆盖、日均充/放电量、SOC、原始动作正负分布、饱和率和 ESS 分支梯度，同时比较最佳验证收益、正式测试收益、严格可行率、弃电量及合作价值捕获率。

不得将“MG1 必须充电”或“收益必须超过 MATD3”写成软件单元测试。物理特征正确不保证策略一定恢复；若仍退化，再分别做初始化、探索和更新起点的单因素实验。宣称算法普遍提升前补充多个训练种子，并使用相同测试日期进行配对比较。
