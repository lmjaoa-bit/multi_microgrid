# Forecast Context 四组消融实验：设计与实施约束

日期：2026-09-28。文档用途：交由其他开发者实施；本次仅编写方案，没有实施算法或启动训练。

核对代码基线：`cb42e9c04cb9d92360b72e705e4243f751e3680d`。实施前记录实际起始提交；若相关代码已变化，先核对本文中的调用链，不能直接按旧行号套用。

配套任务清单：[实施计划](../plans/2026-09-28-forecast-context-ablation.md)。本设计规定行为与边界；计划规定交付顺序、测试和运行命令。

## 1. 目标及不变条件

目标：判断普通 T 丢弃同伴未来供需、仅使用本地未来 8 小时信息，是否限制合作调度收益；通过可复现的 2×2 实验分别估计两个因素及其交互效果。

实现方式：**一个新插件 `matd3_forecast_context`，四份完整 YAML，四种实验配置；不是四套算法代码。**

硬性约束：

- 不修改 `algorithms/matd3/` 永久基线；不修改现有 `matd3_dispatch`、`matd3_hierarchical_transformer` 的实现和配置。
- 所有新网络与算法覆盖放入 `algorithms/matd3_forecast_context/`；不得将实验分支放进共享训练器、环境或 canonical MATD3。
- 使用 `scripts/create_algorithm_variant.py` 创建初始包；使用 `scripts/run_experiment.py` 完成全部运行。
- 使用现有 `single_dispatch_dr` 动作、`peer_energy_summary` 观测和 `physical_power_balance_v1` 本地物理量计算方式。
- 不改变环境物理、奖励、终端约束、预测生成及其噪声、训练/验证/测试划分、结算、正式绘图流程。
- 不改变 Critic、replay 语义、投影器、Actor/Critic 更新规则、探索策略和检查点选择规则。
- 新运行保存于 `outputs/working_runs/matd3_forecast_context/runs/<run_id>/`；原始运行和 retained releases 不覆盖、不删除。
- 不承诺补信息必然提高收益；正确实现、实验完整性与性能结论分别验收。

本轮不处理库存每日重置、库存补回、剩余电量价值、HVAC 辐射观测、负荷/光伏/风电分量展开、网络加深、奖励修改。它们分别属于后续实验，不能在本轮顺带修复。

## 2. 已核实的现状与证据边界

现有普通 T：

1. `core/observation.py` 已输出本地未来 24 小时预测，并输出每个同伴未来的 `energy_balance`、`deficit`、`surplus`。
2. `algorithms/matd3_hierarchical_transformer/actor.py` 只读取同伴当前 6 项摘要；预测 token 只读取本地前 8 小时。
3. 当前预测 token 的输入为本地基础净负荷、购电价、温度、有效标记、未来时刻 sin、cos，共 6 维。
4. 物理净负荷的尺度修正已经存在；本轮不得重新退回“不同归一化量直接相减”。
5. `plugin.py` 调用 `install_diagnostic_sampling`，会替换类的 `__init__`、`select_actions` 和 `update`，并转换 replay。实际运行行为不能仅靠阅读原类的 `update` 判断。

最新参考运行：

| 对象 | 运行 ID | 训练代码提交 |
|---|---|---|
| MATD3 | `MATD3_2026-09-27_23-54-27_seed42` | `950f0dd91f5697716c159843b8273c325d00d406` |
| 普通 T | `普通T_2026-09-28_09-28-23_seed42` | `19eb8ac879faabffcb888d9b842dab4bde1fe35a` |

最新普通 T 最佳检查点的结构探针：改变 144 个同伴未来字段，三个 Actor 的动作均完全不变；改变本地第 9–24 小时字段也不变；改变本地第 1–8 小时字段会变化。探针使用合成输入，证明输入依赖被裁掉，不证明特定收益损失。

历史结果只作背景。当前训练器已增加学习曲线评估等功能，不能要求新的 A 组训练曲线逐点等于历史运行；A 的严格等价验证面向当前锁定代码、相同权重/输入与受控更新。

## 3. 实验矩阵与配置契约

| 组 | `forecast_horizon` | `include_peer_forecast` | 预测输入维度 | 总 token 数 | 唯一研究因素 |
|---|---:|---|---:|---:|---|
| A | 8 | false | 6 | 14 | 普通 T 等价对照 |
| B | 8 | true | 8 | 14 | 在相同时间范围补同伴未来 |
| C | 24 | false | 6 | 30 | 仅延长本地时间范围 |
| D | 24 | true | 8 | 30 | 同伴信息与长窗口结合 |

token 数 = 4 个本地设备/市场 token + 2 个当前同伴 token + H 个预测 token。**B、D 不增加独立的同伴未来 token**，而是将两位同伴的未来余额拼入对应时刻的预测 token。

固定文件：

- A / 插件默认：`configs/algorithms/matd3_forecast_context.yaml`
- B：`configs/algorithms/matd3_forecast_context/b_peer_h8.yaml`
- C：`configs/algorithms/matd3_forecast_context/c_local_h24.yaml`
- D：`configs/algorithms/matd3_forecast_context/d_peer_h24.yaml`

四份都是完整配置；不使用未经共享加载器支持的 `extends`、自定义 include 或隐式 YAML 合并。

新增配置示例（A）：

```yaml
experiment:
  study_id: forecast_context_2x2
  group: A
  network_family: transformer
algorithm:
  name: matd3_forecast_context
  network:
    feature_semantics: physical_power_balance_v1
    forecast_context_version: forecast_context_v1
    forecast_horizon: 8
    include_peer_forecast: false
    peer_forecast_features: [energy_balance]
    embed_dim: 32
    num_heads: 4
    num_layers: 2
    hidden_dim: 64
```

其余配置从锁定版本的普通 T YAML完整复制，保留现有值：Actor lr=0.0002、Critic lr=0.0003、gamma=0.99、tau=0.005、policy_delay=2、target_policy_noise=0.2、target_noise_clip=0.5、batch_size=128、replay_capacity=200000、warmup_steps=1000、exploration_noise=0.15、actor_max_grad_norm=1.0、hidden_dims=[128,128]。

运行配置固定：CUDA、cpu_threads=4、interop_threads=2；训练 5000 回合、正式测试 55 天；验证每 100 回合一次，覆盖完整验证集；学习曲线每 600 步、10 个固定验证日、seed=424242、包含初始评估；diagnostic_interval=20；保存原有其余 logging 和 constrained_profit 约束。

配置加载时严格校验：

- `algorithm.name` 必须等于插件 slug。
- H 必须为整数 8 或 24，拒绝 bool、非整数、0、负数和其他窗口。
- `include_peer_forecast` 必须为 YAML 布尔量，不接受字符串 `"false"` 的真值转换。
- `peer_forecast_features` 第一版只允许 `[energy_balance]`。
- `experiment.group` 必须与矩阵参数匹配；不允许名为 B 实际跑 A。
- schema 中必须存在需要的本地字段、有效标记和启用时的同伴字段；缺字段立即报错，不能静默补零。
- 非 HVAC 微网没有温度通道是已知设计，允许在编码中置零；HVAC 缺温度字段必须报错。
- 只使用已有顺序 `env.agent_ids`；当前研究要求恰好三微网，若不满足则显式拒绝。

## 4. Actor 精确数据流

### 4.1 保留当前六类 token 结构

保留 ESS、DR、Supply、Trade、两位当前 Peer 的字段和处理方式；保留 DR 异构编码、当前 peer_encoder、skip 分支、两个动作头、Transformer 宽度/层数/激活/dropout、token 类型嵌入和投影方式。

Peer 当前摘要仍按原有六字段读取。其存在不能替代同伴未来信息；本轮同时保留两者。

### 4.2 本地预测六字段

每个 k=1..H，基础输入顺序固定为：

```text
0 local_base_net_load
1 buy_price
2 temperature
3 valid
4 future_hour_sin
5 future_hour_cos
```

本地净负荷：

```text
(load_normalized * S_load
 - pv_normalized * S_pv
 - wind_normalized * S_wind) / S_power
```

正值表示基础电力缺口，负值表示基础电力富余。S 来自训练环境的观测 normalizers；不从 val/test 重新估计。

时间位置沿用现有 sin/cos 角度加法，代表实际未来小时；不新增独立位置编码、不更换相对时间表达。温度沿用观测中的 `/40` 尺度，价格沿用原观测尺度。

### 4.3 同伴未来字段

启用时，在六字段后追加：

```text
6 peer_0_energy_balance
7 peer_1_energy_balance
```

其中 `peer_order = [mg for mg in agent_ids if mg != active_agent]`。读取 schema 中 `<peer>_energy_balance_t+<k>`。

该字段现有含义为 `(PV_peer + WT_peer - base_load_peer) / S_power_peer`：**正值表示同伴富余，与本地净负荷正号含义相反**。第一版保留此语义并在字段说明中显式记录，不能无记录地反号或除以本地尺度。

不要额外拼接 `deficit`、`surplus`，两者可由 balance 推导；不要引入真实未来 DR 行为、同伴隐藏状态、未来实际调度或 MILP 决策。已有预测仍仅表示外生基础供需。

`peer_order` 与各同伴的 `power` 归一化尺度必须进入检查点兼容性信息。

### 4.4 编码维度与掩码

- 关闭同伴预测：`forecast_encoder` 第一层为 `Linear(6,64)`。
- 开启同伴预测：第一层为 `Linear(8,64)`；其余编码层与原实现相同。
- 预测输入 tensor 为 `[batch,H,6 or 8]`；预测输出 `[batch,H,32]`。
- 拼接后的全部 token 为 `[batch,6+H,32]`。
- 预测 key padding mask 为 `valid<0.5`；前六个当前 token 始终有效。
- 同一未来时刻的同伴信息随整枚预测 token 一起屏蔽。不能漏遮挡同伴未来，也不能把越界预测当成有效零值。
- 23 点时未来全部屏蔽，前六枚 token 仍有效，必须得到有限输出。
- 无效字段可能有任意有限数，测试需证明这些字段不影响动作；不要求接受 NaN/Inf 输入。
- 为保持 A 等价，不额外改变原有无效 token 的构造数学。靠已有 attention padding mask 屏蔽它们，不在 A 路径顺带增加新处理。

本轮有意保留“本地分量合并成净负荷”的设计，以避免第三个因素混入。

### 4.5 参数预算与解释限制

启用同伴字段使第一层每个 Actor 多 `2×64=128` 个权重，全系统多 384 个；H 扩展不增加该结构中的可训练位置参数，但会增加 attention 计算量。

因此 B−A 是“加入同伴信息及必要的输入投影”的效果，不能声称完全等参数。如果后续需要排除极小参数增量影响，再单独增加同宽度零输入对照；本轮不扩展为第五组。

## 5. 代码组织与注册

拟新增：

```text
algorithms/matd3_forecast_context/
  __init__.py
  plugin.py
  algorithm.py
  actor.py
configs/algorithms/matd3_forecast_context.yaml
configs/algorithms/matd3_forecast_context/
  b_peer_h8.yaml
  c_local_h24.yaml
  d_peer_h24.yaml
tests/test_forecast_context_actor.py
tests/test_forecast_context_equivalence.py
tests/test_forecast_context_checkpoint.py
tests/test_forecast_context_plugin.py
scripts/summarize_forecast_context.py
tests/test_forecast_context_summary.py
```

先运行脚手架：

```powershell
python scripts/create_algorithm_variant.py --name matd3_forecast_context --base matd3_hierarchical_transformer
```

**脚手架陷阱：**当前 `--base` 用于取得基础配置与说明，生成的类仍继承 canonical MATD3，并不会自动生成普通 T 网络。实施者必须补齐普通 T 的局部适配实现，不能把脚手架输出直接作为完成品。

新 `algorithm.py` 可继续继承稳定的 canonical `MATD3`，在本包内适配普通 T 的构造、投影与诊断接口。新 `actor.py` 保存本地普通 T 实现并添加所述变化。允许复制算法局部适配代码作为冻结起点；严禁复制 trainer、evaluator、settlement、plotter。

新包不能运行时导入 `algorithms.matd3_hierarchical_transformer`；旧普通 T 应能独立移除，且其将来修改不能悄悄改变新实验。测试可读取旧插件作为等价性参考，生产包不依赖它。

插件要求：

- 唯一 `ALGORITHM_SPEC`，slug=`matd3_forecast_context`。
- display_name=`普通 T（预测上下文实验）`；A/B/C/D 是实验组标签，不作为动作版本号。
- default_config_path 指向 A 配置。
- algorithm_class=`Matd3ForecastContext`。
- environment_overrides 与普通 T 一致：`ess_sharing.observation_mode=peer_energy_summary`。
- 在本插件内对**新类**调用 `install_diagnostic_sampling(..., variant="matd3_forecast_context")`，保持 grouped actor 路径，不能使用 `plain_actor=True`。
- 不修改共享诊断模块、不重复包装、不在另一个实验的已包装子类上依赖继承式安装。共享安装函数以 `getattr` 判断已安装，继承标记可能导致错误跳过。
- 必须保留 `branch_modules()`、`last_diagnostics`、`collect_diagnostics` 参数、梯度分支索引及参数缓存；更换 Actor 后重建 target、optimizers 和缓存。
- 不新增中央 Python import；按 slug 惰性加载。

### 5.1 公共选择目录的边界

本轮**不修改** `configs/experiment_profiles.yaml` 和 `experiment_defaults.yaml`。否则新增第二个 transformer profile 可能让现有 `--policy transformer` 从唯一匹配变为歧义。

统一通过 `--algorithm matd3_forecast_context --algorithm-config ...` 调用，不与 `--policy`/`--experiment` 混用。

当前共享 `experiment_contract` 对未进入公共 profile 的插件会记录 `policy: unregistered`。本轮接受这个已知目录分类值，它不等于算法插件未注册；通过 `algorithm_slug`、配置中的 `experiment.network_family: transformer`、检查点及研究 manifest 明确网络类型。汇总脚本不得把该值当作未知网络而漏掉运行。若要改进公共 metadata，应另开兼容性修改，不能以“补登记”为由破坏公共选择器。

## 6. 算法与检查点接口

保留共享协议：`from_env`、`select_actions`、`update`、`save`、`load`、`replay.add`、`len(replay)`；动作仍为每微网两个原始意图，Critic 仍接收现有三个有效物理动作通道。

检查点保留原有网络、target 网络、update_steps、projection_signature、state_schema、observation_schemas、actor_feature_semantics、actor_feature_normalizers，并增加：

```text
algorithm_slug = matd3_forecast_context
forecast_context_version = forecast_context_v1
forecast_context_contract:
  horizon
  include_peer_forecast
  peer_forecast_features
  agent_order
  per_agent:
    peer_order
    ordered_forecast_feature_names
    local_source_schema_names
    peer_source_schema_names
  power_normalizers_by_agent
  mask_semantics = valid_lt_half_padding_v1
  local_balance_sign = positive_deficit
  peer_balance_sign = positive_surplus
  architecture: embed_dim, num_heads, num_layers, hidden_dim
forecast_context_signature = SHA256(canonical JSON of contract)
```

contract 数值来自当前可信环境与解析后的配置，不由 checkpoint 覆盖当前环境；签名比较前也要比较结构及必要数值。浮点 normalizer 使用原有严格近似比较，不能只比较格式化后的小数。

加载前必须拒绝：旧算法 slug、缺少新版元数据、H 不同、peer 开关不同、同伴顺序或 schema 不同、尺度不同、网络尺寸不同、投影配置不同。特别是 A/C 以及 B/D 的权重形状可相同，**只靠 load_state_dict 形状检查不够**。

旧普通 T 权重只允许在测试中直接复制到 A Actor，验证数学等价；正式 `load()` 不能绕过身份检查。训练均从头开始；本轮不承诺中断恢复，也不把权重 load 等同于带 optimizer/replay 的完整恢复。

## 7. 实验规范

### 7.1 固定设计与运行预算

默认训练种子预先固定为 `[42,43,44]`。四组×三个种子共 12 个正式运行，seed42 第一轮若代码与配置未修改可计入 12 个，不必重复。

每个种子下 A/B/C/D 使用同一数据、初始环境、预测噪声生成规则、预算和 runtime；记录所有软件/硬件版本。网络结构不同会消耗不同随机数，不承诺逐步 minibatch 或初始化完全相同；用多种子配对降低偶然性。

现有管线会把训练 seed 用于测试、验证环境种子；所以比较首先在**同一 seed 的 A/B/C/D 内配对**。跨 seed 不声称使用了完全相同的预测噪声实现。本轮不修改共享 seed API；报告注明跨种子方差同时含训练与场景随机性。

所有组使用 5000 回合/120000 环境步、完整验证集、55 个测试日。不得给表现弱的组单独多训、降低噪声、调整 lr 或修改筛选阈值。

完整研究序列：工程测试 → 四组短流程 → 冻结代码配置与种子 → seed42 四组 → seed43/44 四组 → 从已保存结果统一汇总。

若 seed42 发现工程错误，可修复后重跑受影响组并显式标记旧运行不纳入；不能因正式测试收益差而无记录地改参数。若预算不足只能报告单种子探索性结果，不把未运行组写成完成。

### 7.2 验证和测试的角色

- 验证集按现有 constrained_profit 规则选择每组 `best_model.pt`。
- 正式测试仍由共享管线在每次训练结束后执行；不能跳过 MILP/结算以伪装完整实验。
- 四组方案在正式运行前锁定；测试结果用于一次性报告，不用于反复挑配置。
- 当前测试已有共同的 HVAC 轻微越界，不能简单要求每组达到 100% 后再择优隐藏失败；完整报告所有组的严格可行率与违规程度。
- 若要作确认性优越结论，需要另行保留未用于探索的数据或评估轮次；现有 55 天已被分析过，不宣称它们是全新盲测集。

### 7.3 指标与比较方法

主指标：55 个完整测试日的团队日均收益；配对差在 date 一对一连接后计算。其次为相对 Idle/规则参考的增益、负合作增益天数、严格可行率、HVAC 越界度时、未供电量、EV 完成率及下游服务率。

辅助指标：弃电、P2P、电网购售电成本、电池吞吐及退化成本、日末高于初始的电池能量、料仓日初/日末/消耗、磨机产量覆盖、HVAC 用电、动作饱和和安全修正比例、训练耗时、Actor 参数量、后期验证表现。

四个关键差值：

```text
B - A：H=8 时补同伴未来的效果
C - A：无同伴未来时延长窗口的效果
D - C：H=24 时补同伴未来的效果
D - B：有同伴未来时延长窗口的效果
交互项 = (D-C) - (B-A) = D-C-B+A
```

逐 seed 计算上述日配对差，再报告三个 seed 的平均、样本标准差（ddof=1）和三个单独结果。不能把 3×55 个日样本当作 165 次独立训练；三个种子也不足以支持很强的统计显著性结论。展示单 seed 日胜率、均值、中位数、最差日，避免均值掩盖风险。

合作价值捕获率使用 `sum(actual_surplus)/sum(milp_surplus)`，不能平均逐日比值。收益百分比必须标出分母和聚合方式，不能把逐日比例均值解释成总收益提升。

MILP 是完整信息、自由分配的参考上界；其差距不等同于纯学习误差。库存重置仍存在，所有组的结论仅适用于当前单日任务。

判定语言：工程通过 ≠ 性能提升；一致正向的多种子日均收益、服务不恶化和合理成本变化支持“值得继续”；只有消纳提高、仅一个 seed 提升或均值提高但服务恶化，应写为权衡/证据不足。

## 8. 结果追踪与只读汇总

四组共享算法 slug，因此 `latest_run.json` 只有一份，会指向最后一个运行，**不能用它寻找四组全部结果**。

以运行目录中的 `config_snapshot.yaml`（其中 algorithm 节点包含完整算法 YAML）、run_metadata、test_summary 和 test_daily_metrics 识别 group/seed。每个纳入的 group×seed 必须有且仅有一个显式选定的完成运行；存在重跑不能自动挑收益最高者。

另一个 metadata 边界：现有 `run_metadata.forecast_horizon_hours` 记录的是共享观测窗口，因此四组均可能为 24；它不是 Actor 实际读取的 H。研究汇总必须读取配置的 `algorithm.network.forecast_horizon`（在完整 snapshot 中位于 algorithm.algorithm.network）及新检查点 contract，不能用公共 metadata 的 24 将 A/B 错标为 24 小时模型。

新增只读脚本 `scripts/summarize_forecast_context.py`，只读取保存结果，不导入环境、算法执行器或求解器，不重新计算策略/MILP。

CLI：

```text
--study-id <new-id>
--run <absolute-or-relative-run-directory>     # 可重复，显式列出纳入运行
--require-seeds 42 43 44                      # 正式汇总
```

结果写入已有研究报告区域 `reports/forecast_context/<study-id>/`，不是训练运行、也不是 retained release；目录已存在则拒绝覆盖。文件：

- `study_manifest.json`：纳入路径、group、seed、checkpoint episode、完整关键配置、代码提交/dirty、环境与物理 hash、数据文件 SHA256、运行完成状态、排除理由；不使用全局 latest 代替清单。
- `run_summary.csv`：每 group×seed 的主要及辅助指标。
- `paired_daily_deltas.csv`：seed/date/对照对/收益差及服务差。
- `factor_effects.csv`：四个简单效果及交互项，逐 seed 与跨 seed 汇总。
- `README.md`：口径、异常、结论限制、未完成项，链接各源运行。

必须验证日期唯一、55 天完整、每日报表与小时轨迹对账、基准一致、结算守恒、配置差异白名单。自动对比四份配置时仅允许 group、H、peer 开关不同；运行生成时间、路径、计时等不作为配置混杂。

若正式运行某一约束失败或组缺失，汇总应给出明确错误/不完整状态，不以缺失当零，也不静默删掉困难日期。数据文件 hash 只能证明采集时的文件身份；首轮运行前记录一次、研究结束再核对，不能事后凭当前文件声称历史数据必然相同。

正常 15 PNG＋15 PDF 仍由共享绘图器生成；本轮汇总不另外实现训练图表系统。

## 9. 交付与实施纪律

交付物：新插件与四份配置、测试、只读汇总脚本、使用说明、工程验证记录；研究运行完成后再交付 12 个运行的 manifest 和比较结果。工程交付与长训练研究交付分别标明状态。

正式长训练前提交已验证代码并记录 clean Git identity；代码评审与用户既有授权决定是否启动长运行。不要在文档交接阶段执行训练。

禁用或移除新实验时，只处理新包、新配置和必要的实验脚本/测试；原算法和 releases 保留。运行结果删除是单独事项，不因实验失败自动清理。

如需长期保留研究结果，使用 `scripts/manage_results.py promote`，保留 working source、禁止覆盖 release；不得自行使用 `--allow-dirty`。

## 10. 风险清单

| 风险 | 应对与验收 |
|---|---|
| 新组 A 已改变训练行为 | 同权重前向、投影和受控更新等价；确认安装相同共享诊断更新路径 |
| 同伴字段拼接但未参与编码 | 特征构造测试＋受控可微路径测试；不靠 attention 图作证明 |
| H 变化但错误加载旧权重 | contract/signature 明确包含 H，形状相同也拒绝 |
| 缺字段静默零填充 | 除非明确允许非 HVAC 温度，否则构造时 fail fast |
| 同伴尺度/正负号错配 | 固定字段顺序、符号说明和尺度契约，用手算输入验证 |
| 公共命令被新增 profile 破坏 | 不改 profile/default catalog，使用显式 --algorithm |
| 四组结果混用 latest | 显式 run 清单＋group×seed 唯一约束 |
| 分母接近零导致指标夸大 | 主报绝对收益差与 ratio-of-sums，披露百分比定义 |
| 新算法比 A 差 | 完整保留并报告结果，不以性能目标修改测试或隐藏日期 |
