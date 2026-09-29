# Reward V2 + 普通 T 同伴预测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 本仓库本轮建议使用 `superpowers:executing-plans` 串行实施；只有用户选择分工执行时再启用子代理。本文件仅交付执行计划，不启动实现、标定或训练。

**Goal:** 将两个当前算法迁移到可审计的 Reward V2 共享环境，为普通 T 接入 H8 同伴预测，并完成选模、参考优化、结算和保存结果绘图的全链路验收。

**Architecture:** 保留一个共享训练/评价流程和一个 MILP 建模入口。新增小型 V2 配置与成本计算模块，环境只编排物理执行和状态更新；分别保留会计收益、广义成本两种账本。按依赖顺序交付，V2 通过显式配置启用，全部接通后再切换当前正式配置。

**Tech Stack:** Python、PyTorch、NumPy、Pandas、PyYAML、HiGHS/highspy、Matplotlib、pytest；沿用仓库依赖，不引入新训练框架。

**Spec:** [修订设计规范](../specs/2026-09-29-reward-v2-peer-forecast-design.md)，重点读取第 18–20 节；同时遵守根目录 `AGENTS.md`。

## Global Constraints

- canonical `algorithms/matd3/` 与 `configs/algorithms/matd3.yaml` 不改；迁移对象为 `matd3_dispatch` 和 `matd3_hierarchical_transformer`，不新建算法 slug。
- CLI `--algorithm matd3` 仍解析为 `matd3_dispatch`，使用其自身 YAML；用户名称仍为 MATD3、普通 T。
- `action_design=single_dispatch_dr`；两算法使用 `peer_energy_summary`；完整自然日 `delta_t=1.0`、`episode_length=24`。
- `forecast_horizon: 8`，普通 T forecast 每 token 8 维、共 14 token；共享 observation future_horizon=24 保留。
- 禁止 grid export；保留 ESS、P2P 和 MG1 瞬时物理约束；不恢复 terminal SOC/库存/温度目标。
- Reward V2 九项成本之和取负再除以固定 reward_scale；三 Agent 共享一个团队奖励，团队统计不得再相加三次。
- 推荐 V2 当前两算法 `gamma: 1.0`，由短训练检查数值表现；不得默默回到 0.99 或修改 canonical 更新逻辑。
- 七联盟会计值用于 Equal/Shapley；另一次全联盟广义成本求解用于 V2 参考。不得更换 `total_profit`/`v_*` 含义。
- 所有正常运行保存在 `outputs/working_runs/<resolved_slug>/runs/<run_id>/`；历史运行、消融包和 retained releases 不改写。
- 图表只读保存结果，正式验收每个 run 的 `figures/` 有 15 PNG、15 PDF；不把额外目录的图片凑数。
- 不以 test 标定权重；新系数在标定锁定前只允许诊断、测试和短 smoke，不启动 5000 回合训练。
- 全部复用 `scripts/run_experiment.py`；不复制 trainer、evaluator、settlement、plotter。此计划不包含发布或结果 promotion。

## Review Focus

1. 24:00 EV 离站、周末/营业时段切换：先提交设备状态，再计服务损失，最后 done；不遗漏或重复收费（Task 2、4）。
2. 形状相同但 reward/schema/normalizer 不同的模型或恢复文件：加载在更改模型状态前失败，不能续用旧 replay（Task 7）。
3. MILP 超时有 incumbent、某联盟无解：区分近似参考与不可用结果，不能把 incumbent 当最小成本下界（Task 9、10）。
4. 最后一步全部 forecast 无效，填充值甚至含 NaN：清零后再编码，动作有限且与无效字段无关（Task 6）。
5. 缺 objective、未知 selection mode、旧服务增强脚本重写指标：不得默认成零或把软服务重新变成硬筛选（Task 8、11）。

## 文件职责与接口落点

所有下列新路径均为**计划新增**，不表示当前已经存在。

| 文件 | 职责 |
|---|---|
| 新增 `core/reward_v2.py` | V2 配置解析、碳输入解析、纯函数成本计算、可序列化语义合同；不读写 run、不控制设备 |
| 修改 `core/config_validation.py` | 调用 V2 校验；旧语义继续现有路径 |
| 修改 `components/commercial_hvac.py`、`components/ev_charging.py` | 当前物理动作范围和执行后服务事件 |
| 修改 `core/joint_action_projection.py` | 训练有效动作与环境执行一致 |
| 修改 `envs/multi_microgrid_env.py`、`core/cost_accounting.py`、`core/grid_settlement.py` | 物理清算、日峰状态、两套账本编排 |
| 修改 `core/observation.py`、`core/global_state.py`、`core/profiles.py`、`core/data_manager.py` | 峰值/碳公共信息、固定尺度、数据校验 |
| 修改普通 T `actor.py`、`algorithm.py` 和当前 MATD3 `algorithm.py` | 正式 peer forecast 与两插件 checkpoint 合同 |
| 修改 `utils/metrics_collector.py`、`evaluation/service_metrics.py`、`evaluation/evaluate_matd3.py` | 原始分量、日统计、评价输出；中性入口 `evaluate_algorithm.py` 继续转发 |
| 修改 `experiments/checkpoint_selector.py`、`experiments/training_pipeline.py` | 新选模模式、失败处理、恢复/元数据、正式测试 |
| 修改 `optimization/common_constraints.py`、合作/独立入口、`optimization/oracle_replay.py` | 双目标共享 MILP、目标复算和回放差异 |
| 修改 `evaluation/coalition_evaluator.py`、`evaluation/settlement.py` | 七联盟会计结果与广义参考分离、可用性传递 |
| 修改 `evaluation/plotting.py`、`evaluation/service_reliability.py`、`scripts/generate_results.py` | 保存结果消费者的 V2 口径兼容；不新建绘图引擎 |
| 新增 `evaluation/reward_calibration.py`、`scripts/calibrate_reward_v2.py` | 复用现有按日评价入口的标定策略与原始分量报告，不新增训练循环 |
| 新增/扩展 `tests/test_reward_v2_*.py` 与现有相关测试 | 隔离 V2 fixture，保留明确的历史语义 fixture |
| 当前正式/两份 smoke YAML、`docs/ALGORITHM_DEVELOPMENT.md` | 最后切换默认配置、记录使用和复现方式 |

## 执行约定与准备

- [ ] 记录实际起始 commit、工作区修改、数据文件 SHA256、依赖版本；保留本轮已修改的设计稿。使用实际起点，不强制回退设计基线。
- [ ] 执行阶段按 worktree 技能检查是否需要隔离；通过宿主管理工具复用/创建合适 checkout，不在仅写计划阶段创建。
- [ ] 运行一次基线 `pytest -q`，记录已有失败；确认 highspy 可用。不能用跳过 MILP 掩盖验收缺项。
- [ ] 后续每项按“先写行为测试 → 确认失败原因 → 最小实现 → 相关测试通过 → 审查差异并分项提交”推进。提交只包含该项文件，不使用 `git add .` 混入用户工作。
- [ ] 实现中间阶段不切换正式 YAML；测试用深拷贝 V2 fixture 显式启用。历史测试固定自己的旧配置，不通过删除断言让新默认通过。

依赖主线：Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13。部分纯函数工作可独立审阅，但不并行修改共享环境主文件。

## Task 1：冻结 V2 配置、单位和解析入口

**Files:** 新增 `core/reward_v2.py`、`tests/test_reward_v2_config.py`；修改 `core/config_validation.py`。

**Interfaces:**
- `resolve_reward_v2_config(config: Mapping[str, Any]) -> dict[str, Any] | None`：非 V2 返回 None，V2 返回经校验的统一字段，不修改输入。
- `resolve_carbon_factors(config: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray`：优先采用有效数据列；列缺失时仅接受显式固定值；存在坏数据时报错而不以 fallback 掩盖。
- 新配置 `reward.semantics_version=generalized_system_cost_v2`；`reward.calibration_status` 为 `diagnostic` 或 `locked`；共享系数继续归属 `cost`。

- [ ] 写 `test_v2_alias_conflict_rejected`：新 `cost.curtailment_price_per_kwh` 与旧 `cost.renewable_curtailment_cost_per_kwh` 不等时报错；EV 新 `cost.ev_shortfall_price_per_kwh` 与旧设备字段同理，相等允许解析为单一来源。
- [ ] 写 `test_v2_units_and_missing_values`：六个新系数有限且非负，reward_scale>0；缺必填系数、负值、NaN/Inf、非完整自然日配置均失败；参数化验证没有版本时返回 None。
- [ ] 写 `test_carbon_source_is_explicit`：缺列且无固定值失败；显式 `reward.fixed_carbon_factor_kg_per_kwh=0.5` 得到长度与 frame 一致的 0.5；坏列不得被固定值覆盖。
- [ ] 运行 `pytest -q tests/test_reward_v2_config.py`，确认新增行为尚未实现而失败。
- [ ] 实现上述接口；解析输出包含六个新系数、unmet 系数、reward_scale、语义字段。电池退化继续来自各设备配置，不新增团队统一退化率。
- [ ] 定义元数据：设计中的 dispatch/reward/dr/hvac/ev 版本；补充 `observation_semantics_version=reward_v2_public_state_v1`、`reward_discount_semantics=finite_day_undiscounted_v1`、`comfort_metric_semantics=boundary_sampled_degree_hours`、`carbon_information_semantics=current_observed_no_future_v1`、`settlement_value_semantics=accounting_profit_v2`。
- [ ] 运行该测试及 `pytest -q tests/test_experiment_profiles.py`，通过后提交 `feat: define reward v2 configuration contract`。

## Task 2：设备软服务分支与边界事件

**Files:** 修改两设备文件；新增 `tests/test_reward_v2_devices.py`；回归 `tests/test_ev_cohorts.py`、`tests/test_hvac_viability.py`。

**Interfaces:** 保持 `feasible_bounds(t)`、`plan(raw_action,t)`、`commit(plan,delivered_fraction)`、`state(t)` 签名。V2 由 Task 1 解析后的控制语义明确启用；旧分支保留。

- [ ] 写 `test_soft_hvac_never_calls_comfort_projection`：将 viability 求解和 `_one_step_bounds_q` 替换为抛异常函数；soft 模式 `feasible_bounds/plan/state` 仍可执行；范围精确为 `(-cooling_max_kw, heating_max_kw)`。
- [ ] 写 `test_ev_zero_request_is_zero_for_all_groups`：连接且临近离站时 raw=-1，组下界、站端下界、请求和实际充电均为零；raw=0 不断言为零。
- [ ] 写 `test_departure_at_24_is_charged_once`：第 23 步先更新电池能量再生成 shortfall，事件总数为一；后续 state 查询与汇总不产生新事件。
- [ ] 运行 `pytest -q tests/test_reward_v2_devices.py`，确认失败后实现设备分支；EV 总上界 `min(station_limit,sum(group_max))`，组上界含效率和剩余需求，urgency 分配不增加总功率。
- [ ] 补营业时段、周末、24:00 unoccupied 的 HVAC 边界测试；断言温度越界允许发生而所有状态有限，物理功率未被舒适约束改写。
- [ ] 运行 `pytest -q tests/test_reward_v2_devices.py tests/test_ev_cohorts.py tests/test_hvac_viability.py`；通过后提交 `feat: add soft service control semantics`。

## Task 3：训练投影与环境执行一致

**Files:** 修改 `core/joint_action_projection.py`，必要时修改环境传递 bounds 的调用；扩展 `tests/test_joint_action_projection.py`、`tests/test_executed_action_equivalence.py`、`tests/test_ev_mapping_semantics.py`。

**Interfaces:** 原 `JointActionProjector` 接口和有效动作 `[ESS-local,ESS-trade,DR-effective]` 顺序不变；仅改变 V2 的 HVAC/EV 可行范围来源。

- [ ] 写 `test_v2_projector_matches_device_execution`：参数化 HVAC 正/负/零功率、EV 零/满功率、ESS 充/放/不成交；同一 state/action 对比 Torch 有效动作与环境执行值，kW 转换后容差 1e-6。
- [ ] 写 `test_v2_unsaturated_action_has_finite_gradient`：非饱和合法区梯度有限且非零；保留既有无需求 ESS 零梯度测试。
- [ ] 运行上述三个测试文件，确认新增 V2 情景失败后，同步 projector 的物理映射；不得删除 renewable-first 来源约束或整个投影器。
- [ ] 加固定种子随机合法输入回归，对站端拥挤、多组离站顺序、功率饱和逐一核对；通过后提交 `fix: align v2 training and executed actions`。

## Task 4：九项成本与无外送的单步账本

**Files:** 修改 `core/reward_v2.py`、`core/cost_accounting.py`、`core/grid_settlement.py`、环境；新增 `tests/test_reward_v2_accounting.py`。

**Interfaces:** `calculate_reward_v2(*, raw: Mapping[str,float], rates: Mapping[str,float]) -> dict[str,float]`，只计算不改变状态。

`raw` 固定键：`grid_import_cost`、`battery_degradation_cost`、`curtailment_kwh`、`carbon_kg`、`delta_peak_kw`、`mill_unserved_t`、`comfort_degree_hours`、`ev_departure_shortfall_kwh`、`unmet_kwh`。输出九项金额 `cost_grid/battery/curtail/carbon/peak/mill/comfort/ev/unmet`、`generalized_system_cost`、`team_reward`。`rates` 使用 Task 1 的系数名。

- [ ] 写纯函数数值 fixture：raw 按上述顺序为 `[10,2,3,4,5,6,7,8,9]`，对应七个需乘系数的 rate 均为 1、scale=10；断言九项和=54、team_reward=-5.4。这些数值只用于测试，不能写成正式权重。
- [ ] 写 `test_peak_uses_executed_load_and_resets`：负荷 `[100,80,130]` 得增量 `[100,0,30]`；peak_price=2 得成本总和260；reset 后峰值归零。ESS 充电不进入负荷，HVAC 用电用绝对值。
- [ ] 写 `test_ev_and_curtailment_not_double_counted`：会计利润含其各一次，广义账本从 raw 重建也各一次，不通过会计利润二次扣除。
- [ ] 写 `test_v2_surplus_is_curtailed_not_exported`：最终无内部用途的 RES 剩余全部弃电，grid export/revenue=0；保持来源与节点/P2P 守恒断言。
- [ ] 运行 `pytest -q tests/test_reward_v2_accounting.py` 确认失败，然后实现。环境 step 顺序固定为：物理执行 → commit 设备 → 服务事件/峰值 → 成本 → done 与 next state；当前 reward 含最后一步离站事件。
- [ ] 保留 `team_profit=sum(physical_profit)`；新增 `operating_margin_before_penalties`；V2 不调用已关闭的终端惩罚。碳仅进口时计、退化仅实际吞吐计。
- [ ] 运行新增测试及 `pytest -q tests/test_renewable_first_dispatch.py tests/test_cooperative_physics.py`；通过后提交 `feat: implement audited reward v2 accounting`。

## Task 5：公共峰值/碳状态与固定归一化

**Files:** 修改 observation/global_state/profiles/data_manager 和环境；新增 `tests/test_reward_v2_observations.py`。

**Interfaces:** V2 schema 追加 `team_load_peak_so_far`、`grid_carbon_factor`，分别使用固定 `team_power_scale_kw`、`carbon_factor_scale`；观测构造通过环境传入的公共状态读取，禁止 builder 自行重算动作后峰值。

- [ ] 写 `test_same_current_load_different_peak_has_different_state`：当前负荷均150、历史峰值100/200，Actor 观测及全局状态不同；动作前状态不含本步新峰值。
- [ ] 写 `test_normalizers_do_not_depend_on_test_or_episode`：同一训练数据/配置，换测试日不变；尺度正且有限，完整保存到运行/模型合同。
- [ ] 写 `test_carbon_is_current_only`：当前因子进入两种 state，不增加未来实际碳序列；reset 和末步 next state 不越界，日末碳值采用最后一个可用当前值且不 bootstrap。
- [ ] 运行 `pytest -q tests/test_reward_v2_observations.py` 确认失败；实现归一化：team scale 用各 MG 已有训练来源 power normalizer 的和，carbon scale 取训练因子最大值并设置正下限，均保存数值。
- [ ] 同步 schema/value 顺序与 dimension；旧语义的 schema 不变。运行新增测试和 `pytest -q tests/test_constraint_aware_observations.py`；通过后提交 `feat: expose reward v2 public state`。

## Task 6：普通 T 的 H8 peer forecast 与公共 trade 特征

**Files:** 修改 `algorithms/matd3_hierarchical_transformer/actor.py`、`algorithm.py`；新增 `tests/test_reward_v2_peer_forecast.py`；不改历史 forecast-context 包。

**Interfaces:** Actor 构造增加关键字 `include_peer_forecast: bool=False`、`peer_forecast_features: Sequence[str]=("energy_balance",)`；正式 V2 配置设 true。既有六个 forecast 特征后追加按 agent_ids 排序的两个 peer balance；公共峰值/碳加入现有 trade encoder。

- [ ] 写 `test_v2_forecast_contract`：H=8、输入8维、token数14；peer 顺序与 B 组相同。仅对 forecast 特征与 B 组作数值对齐，不比较全模型动作。
- [ ] 写 `test_invalid_forecasts_cannot_change_action`：eval 模式无 dropout 随机性；改无效 peer/local 值（含 NaN）动作不变且有限；全部8个未来无效同样通过。
- [ ] 写 `test_valid_peer_changes_forecast_embedding`：有效 peer 特征改变时编码输入/中间 token 改变，避免依赖随机权重下最终动作必须明显变化的脆弱断言。
- [ ] 写 `test_trade_encoder_consumes_peak_and_carbon`：新增公共状态实际进入 trade encoder；旧 sys_margin 不冒称日峰裕度。
- [ ] 运行 `pytest -q tests/test_reward_v2_peer_forecast.py` 确认失败；回移必要 B 组逻辑到正式包内部，编码前对无效值清零，attention 和 readout 都应用 mask。
- [ ] 运行新增测试和 `pytest -q tests/test_matd3_hierarchical_transformer.py tests/test_plain_t_power_features.py tests/test_forecast_context_equivalence.py`；通过后提交 `feat: add peer forecast to ordinary transformer`。

## Task 7：模型/恢复兼容合同与有限日折扣

**Files:** 修改两个当前插件 `algorithm.py`、`core/reward_v2.py`、pipeline 的恢复校验；新增 `tests/test_reward_v2_checkpoint.py`，扩展 `tests/test_two_algorithm_smoke.py`、`tests/test_training_pipeline_recovery.py`。

**Interfaces:** 新增 `build_reward_v2_contract(config: Mapping[str,Any], observation_schemas: Mapping[str,Sequence[str]], state_schema: Sequence[str], normalizers: Mapping[str,Any]) -> dict[str,Any]`；checkpoint 增加 `reward_v2_contract`，恢复文件含其稳定序列化 hash。两插件保存/加载接口不变。

- [ ] 写参数化 roundtrip：两个插件保存再加载，确定性动作一致；修改 reward 版本、系数、schema次序、尺度、peer次序、mask语义任一项都拒绝，即使 tensor shape 相同。
- [ ] 写 `test_v2_rejects_legacy_before_mutation`：6维旧 checkpoint、缺合同 checkpoint、旧恢复状态均拒绝，模型参数和 replay 在失败前后不变。
- [ ] 写 `test_finite_day_terminal_target`：两个插件测试配置 gamma=1，done=1 的第24步 target 仅等于即时 reward；done=0 保留下一状态值。使用现有算法更新接口，不修改 canonical。
- [ ] 运行 `pytest -q tests/test_reward_v2_checkpoint.py` 确认失败，实现合同校验；普通 T 额外保存 `actor_forecast_context_version=peer_balance_h8_v2`，MATD3 保存 baseline 标记。
- [ ] 记录动作、环境、观测、奖励、normalizer、gamma、算法名称；历史加载分支仍可用于历史配置，但不能加载进 V2 实例。
- [ ] 运行新增测试及上述两份现有测试；通过后提交 `feat: enforce reward v2 checkpoint compatibility`。

## Task 8：日统计与物理合格选模

**Files:** 修改 metrics/service_metrics/evaluate_matd3、selector、pipeline；新增 `tests/test_reward_v2_metrics.py`；扩展 `tests/test_checkpoint_selector.py`、recovery 测试。

**Interfaces:** 保留 `team_reward_mean` 为未折扣日回报均值；新增 `generalized_system_cost_mean`、九项日成本与均值。新模式 `physical_constrained_objective`；selector 新增只读 `requires_eligible_checkpoint: bool`，pipeline 通过该能力判断，不比较旧模式字符串。

物理汇总必需键固定为：`physical_feasible_day_rate`、`power_balance_residual_max_kw`、`source_balance_residual_max_kw`、`p2p_balance_residual_max_kw`、`device_bound_violation_max`、`grid_export_max_kw`、`unmet_energy_mean_kwh`。device violation 为 SOC、库存、额定功率偏差除各自固定尺度后的最大值，不含舒适和 EV 目标短缺。

- [ ] 写 `test_daily_cost_and_shared_reward_conservation`：24步累积九项一致、return=-cost/scale；三个共享 reward 不能乘三；初末库存、温度、生产/消费均保留。
- [ ] 写 `test_physical_selector_ignores_soft_service_rate`：舒适/EV 服务率不足但物理合格可入选；更高 team_reward 胜出。
- [ ] 写 `test_missing_or_nonfinite_objective_never_wins`：缺失、NaN/Inf 均不可入选；未知 mode 构造即报错；无合格 best 时正式测试未被调用且不复制 final。
- [ ] 运行 `pytest -q tests/test_reward_v2_metrics.py tests/test_checkpoint_selector.py` 确认失败，实现原始违约逐步检测，不从已裁剪 SOC/库存掩盖实际违约。
- [ ] V2 selection 固定 min physical_feasible_day_rate=1、其余 max=0，各字段 atol=1e-7；目标 `team_reward_mean`。物理断言失败立即停止，不作为 reward 罚项补偿。
- [ ] 运行上述测试和 `pytest -q tests/test_metrics_recording.py tests/test_service_metric_contract.py tests/test_training_pipeline_recovery.py`；通过后提交 `feat: select physically valid reward v2 checkpoints`。

## Task 9：共享 MILP 双目标与复算

**Files:** 修改 `optimization/common_constraints.py`、`cooperative_dispatch.py`、`independent_dispatch.py`、`oracle_replay.py`；新增 `tests/test_reward_v2_oracle.py`。

**Interfaces:** `solve_dispatch` 保留原参数，追加关键字 `objective_kind: str="accounting_profit"`；仅允许该值或 `generalized_cost_v2`。`DispatchResult` 尾部追加带默认值的 `objective_kind`、`generalized_system_cost: float|None`、`solver_objective: float|None`、`solver_best_bound: float|None`、`solver_gap: float|None`；`total_profit` 不改含义，`total_cost` 保持既有兼容口径，新目标读新增字段。

- [ ] 写 `test_generalized_solver_objective_matches_trajectory`：24小时固定日，按保存轨迹独立复算九项，误差 `<=1e-6*max(1,abs(objective))`；禁止用同一输出字段互相比对冒充复算。
- [ ] 写 `test_oracle_soft_service_and_zero_export`：允许温度 slack/EV shortfall，出口为零，HVAC 加热制冷互斥；peak 仅实际终端负荷；系数和边界采样与 Task 4 一致。
- [ ] 写 `test_solver_incumbent_is_not_lower_bound`：模拟 time limit，有可行 incumbent 保存其目标、bound、gap；无可用解不得生成虚构全零轨迹。
- [ ] 运行 `pytest -q tests/test_reward_v2_oracle.py` 确认失败，再实现共用约束、目标切换和双账本后处理；不复制第二套 MILP。
- [ ] 审查 EV 自由分组分配、新能源优先等可行域差异；回放保存 `solver_trajectory_cost` 与 `replayed_trajectory_cost`，标记不等价原因，不把回放值称最优值。
- [ ] 运行新增测试及 `pytest -q tests/test_renewable_first_oracle.py tests/test_oracle_market_replay.py tests/test_optimization_and_allocation.py`；通过后提交 `feat: add generalized cost dispatch reference`。

## Task 10：七联盟结算与广义参考分开

**Files:** 修改 `evaluation/coalition_evaluator.py`、`settlement.py`、`evaluate_matd3.py`；新增 `tests/test_reward_v2_settlement.py`。

**Interfaces:** 保持 `evaluate_day_coalitions()`、`CoalitionEvaluation.values/grand_result` 为会计口径；新增 `evaluate_generalized_reference(config, *, split: str, day: str, solver_output: bool=False) -> DispatchResult`。共享按日数据可以内部复用，不改变公开 `grand_result` 含义。

- [ ] 写 `test_seven_accounting_calls_plus_one_generalized_call`：记录 objective_kind 和联盟成员；七次会计、一次广义全联盟；`v_*` 只来自 total_profit，`generalized_reference_*` 只来自新增参考。
- [ ] 写 `test_accounting_transfers_conserve_actual_profit`：Equal/Shapley 转移和为零、结算后团队利润等于 RL 实际利润；广义成本不进入这些计算。
- [ ] 写 `test_missing_coalition_and_negative_surplus_are_visible`：缺一联盟不补0、Shapley unavailable；负 surplus 不截零，IR 状态如实输出；近似解传递 approximation/gap。
- [ ] 运行 `pytest -q tests/test_reward_v2_settlement.py` 确认失败，实现评价编排。联盟失败可保留 RL 指标但整日结算标不可用；端到端验收不把不完整结算当成功。
- [ ] 广义参考单独保存轨迹、状态、objective/bound/gap，必要元数据包含日场景/初值指纹；正常路径一天只新增一次求解。
- [ ] 运行新增测试与 allocation 回归，通过后提交 `feat: separate reward reference from accounting settlement`。

## Task 11：保存结果、图表与后处理兼容

**Files:** 修改 `evaluation/plotting.py`、`service_reliability.py`、`scripts/generate_results.py`、pipeline 输出字段；扩展 `tests/test_training_convergence_plot.py`、`test_paper_artifacts.py`；新增 `tests/test_reward_v2_outputs.py`。

**Interfaces:** 保持 `generate_figures()` 和 `augment_run_outputs()` 入口；V2 根据语义版本选择字段，不从旧 run 推断新 reward。

- [ ] 写 `test_saved_v2_figures_do_not_execute_models`：patch 环境 step/策略测试/MILP solve 为抛异常函数，仅凭保存 JSON/CSV 仍可生成图；检查 `figures/` 15个 PNG、15个 PDF。
- [ ] 写 `test_v2_labels_match_data_and_legacy_is_unchanged`：reward 曲线标 generalized return、利润曲线标 accounting profit；V1/V2 不直接合并同一 return 纵轴；旧 fixture 可读且源文件不变。
- [ ] 写 `test_service_augmentation_preserves_reward_and_selection`：运行真实后处理前后，team_reward、generalized cost、选模合同不变；不能把 strict service 指标当 physical feasibility。
- [ ] 运行 `pytest -q tests/test_reward_v2_outputs.py` 确认失败；修改现有15张图的合适面板/标题，不额外生成一套图替代合同。缺必要数据显式标缺失，不伪造零值。
- [ ] 核对 `scripts/generate_results.py` 目前存在可选绘图模块 fallback；以共享 `evaluation.plotting` 的真实产物为验收依据，不能将 `PLOTTING_PENDING.txt` 视为绘图完成，也不为此构建新 plotting 包。
- [ ] 运行新增与上述现有测试，抽查成本分解、服务质量、训练曲线及结算图的 PNG/PDF；通过后提交 `feat: report reward v2 without changing accounting semantics`。

## Task 12：训练/验证标定、锁定权重和切换当前配置

**Files:** 新增 `evaluation/reward_calibration.py`、`scripts/calibrate_reward_v2.py`、`tests/test_reward_v2_calibration.py`；修改 cooperative、两个当前正式 YAML、两份 smoke YAML、pipeline 启动校验及 `scripts/run_experiment.py` 的诊断返回处理。

**Interfaces:** `collect_calibration(config: Mapping[str,Any], *, split: str, days: Sequence[str], policy: Any, seed: int) -> dict[str,Any]` 复用 `evaluate_policy_on_days`；只接受 train/val。CLI 新增 `--config --split --days --seed --output`，不负责训练。`--days` 是日数量，选定日期列表完整写入报告。

- [ ] 写 `test_calibration_rejects_test_split` 和 `test_calibration_records_components_and_scenarios`：拒绝 test；输出日列表、策略、seed、全部 raw 分量、权重候选、推荐理由、scale、配置 hash。
- [ ] 写 `test_idle_actions_are_device_specific`：HVAC signed=0、EV raw=-1；ESS 用现有映射构造真正零吞吐，MG1保留硬库存动作，明确 idle 不是所有设备停机。服务优先规则用固定单步温控和 urgency 充电，不调用未来 viability LP。
- [ ] 运行 `pytest -q tests/test_reward_v2_calibration.py` 确认失败，实现报告与策略适配，不复制评价循环。诊断报告保存到对应当前算法 working run 的 `diagnostics/reward_calibration.json`，run metadata 标记 diagnostic。
- [ ] 从训练/验证日生成 idle、服务优先规则及共享流水线短训练模型的 raw 分量分布；短训练可使用诊断权重，报告明确临时性质，禁止据正式 test 调整。
- [ ] 用实际分布列出 EV 的购电/效率/碳/峰值边际交换和舒适取舍；候选系数固定后记录选择理由，锁定 `reward.calibration_status=locked` 和报告 hash，再确定 reward_scale。不得用“每项数值相等”自动决定权重。
- [ ] 仅在诊断与标定阶段允许 `reward.calibration_status=diagnostic`；启动校验限制诊断训练<=100完整日回合、禁用正式 test，并记录目的；单元测试和下阶段 smoke 不依赖未锁定权重。必要 CLI 诊断开关命名 `--diagnostic-only`，复用 train/validation，完成后显式返回诊断状态而不伪装正常全链完成。
- [ ] 诊断返回 summary 含 `run_purpose=reward_calibration`、`formal_test_completed=false`、`run_directory`；`scripts/run_experiment.py` 据此跳过正式 test 文件所需的 service augmentation/结果生成，并正常记录 Git 身份。`collect_calibration` 只接收已存在的 working run 诊断目录作为 CLI 输出位置，不创建假的完整训练 run；需要新目录时由共享 layout/lifecycle 建立诊断运行记录。
- [ ] 写 `test_unlocked_reward_cannot_start_formal_run`：未锁定且普通运行/5000回合启动失败；诊断模式超过100回合失败；locked 不受该临时预算限制。
- [ ] 短训练比较 Q/TD/梯度是否有限及是否出现数值爆炸，记录 gamma=1 的观测结果；若不稳定，停止正式切换并输出原因，不自行改变设计。短训稳定也不声称收敛。
- [ ] 锁定后切换 cooperative 的V2语义、所有 export_limit=0；两个当前正式/smoke YAML 统一 gamma=1、物理选模；普通 T network H8+peer；两份 smoke 保持 batch8/warmup8、3回合、2测试日。
- [ ] 运行 `pytest -q tests/test_reward_v2_config.py tests/test_reward_v2_calibration.py tests/test_two_algorithm_smoke.py tests/test_experiment_profiles.py`；通过后提交 `feat: calibrate and enable shared reward v2 configuration`。

本任务的标定不是要求执行者凭空猜用户偏好。能依据数据和设计合理确定的候选应给出具体数值与理由；若多个方案对应明显不同服务取舍且无法据现有要求选择，交付完成的候选对比后再询问该项偏好，不以问题替代前面的标定工作。

## Task 13：全量验收、运行证据和交付

**Files:** 扩展 `tests/test_reward_v2_outputs.py` 中产物合同；更新 `docs/ALGORITHM_DEVELOPMENT.md`；新增 `docs/superpowers/reports/2026-09-29-reward-v2-implementation.md`。报告日期执行时如跨日可用实际日期并更新计划链接。

- [ ] 运行 `pytest -q`，保存测试摘要；若基线已有失败，与当前新增失败分开报告，未解决的必要验收失败不得宣告完成。
- [ ] 从仓库根目录依次运行以下两个**现有参数**构成的端到端命令（Task 12 已更新并锁定配置）：

```powershell
python scripts/run_experiment.py --algorithm matd3 --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_dispatch.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_hierarchical_transformer.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
```

- [ ] 不传 `--no-milp`、`--no-validation`、`--output` 或 releases 路径。CPU 为可复现 smoke 默认；正式训练设备另行按用户环境决定。
- [ ] 从各命令返回/metadata 记录确切 run_dir，不在有并发运行时只凭 latest_run 猜测。核对 resolved slug、算法 YAML、环境 snapshot、数据/权重/场景 hash。
- [ ] 检查每个 run：实际梯度更新次数>0；best/final 存在；正式 test 加载的是 best 的文件及 hash；2个完整测试日；每天7个会计联盟+1个广义参考；Equal/Shapley 和 summary 全部可用；figures 恰好15 PNG+15 PDF。
- [ ] 独立从保存 hourly CSV 复算9项成本、日峰和 EV 事件，与 summary 对齐；查零 export、物理残差、软服务指标、初末库存、求解状态/bound/gap。
- [ ] 以这些保存结果单独重跑绘图并确认没有环境/MILP调用；视觉检查关键图和中英文标签可读，无把 generalized return 标 profit。
- [ ] 运行 `git diff --check`，比较起点至终点 canonical、历史消融代码/结果和 releases 的文件清单/hash，确认未改变。只清理本次临时工作且不删除任何需要留存的运行。
- [ ] 在实施报告填起止 commit、变更文件、测试命令/结果、两个 run_dir、标定报告、数值诊断、15+15计数、设计偏离；记录 H8仅指普通 T Actor 窗口、独立单日初末库存、旧测试集参与结构选择等解释限制。
- [ ] 通过后提交文档和必要验收测试 `docs: record reward v2 implementation verification`。不自动启动5000回合，不自动推广到releases。

## 阶段关口与停止条件

| 关口 | 完成任务 | 允许继续的证据 |
|---|---|---|
| A 物理/执行 | 1–3 | 旧语义回归、软服务分支、Torch/环境有效动作一致 |
| B 学习合同 | 4–8 | 成本复算、必要状态、peer mask、模型加载、正确 best 选择 |
| C 对照/输出 | 9–11 | 双目标MILP、结算守恒、保存结果图表口径完整 |
| D 标定 | 12 | train/val 标定与权重锁定，gamma数值诊断明确，无 test 调参 |
| E 落地验收 | 13 | 全量测试与两个完整端到端smoke、所有产物可追溯 |

遇物理不守恒、重复计费、日末事件遗漏、预测泄漏、恢复混版本或求解目标复算不一致，停止后续阶段并修复；不通过调 reward 权重掩盖实现错误。外部求解依赖不可用时明确标记受阻步骤，不宣告全链路通过。

权重有多个合理服务取舍时，提供完成的候选对比再澄清；gamma=1出现不稳定时记录证据并回到设计讨论。其余实现细节按本计划自主推进，不逐文件重复请求确认。

## 设计覆盖与执行交接

| 设计条款 | 负责任务 |
|---|---|
| §3–9 奖励、禁外送、服务和账本 | 1–5、8、12 |
| §10、§18.9 选模 | 8 |
| §11、§18.6–18.7 参考优化和结算 | 9–10 |
| §12、§18.10 预测与兼容 | 5–7 |
| §13、§18.5、§18.8 研究解释/标定 | 8、12–13 |
| §14–17、§20 消费者/元数据/回归/产物 | 7–13 |
| §18.1–18.4 状态、折扣、碳、投影 | 1、3–7、12 |

本计划完成实施后的交付范围是：V2代码、锁定配置、测试、标定证据和两个smoke结果；正式多种子长训为下一轮运行任务。建议由当前执行者串行实现以保持共享接口一致，完成后再独立审查整体差异；若选择子代理方式，各任务仍按上述依赖串行集成。

本文件编写时仅核对了源码、测试路径和CLI参数；复选框均未完成，所列测试与运行命令均为未来执行要求。
