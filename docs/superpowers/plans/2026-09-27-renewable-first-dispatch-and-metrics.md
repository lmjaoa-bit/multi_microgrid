# 联盟风光优先调度与评价可信性修复：实施规范

> For agentic workers: 按 superpowers:executing-plans（如可用）逐任务执行本规范；每个步骤使用复选框记录。用户指定由其他模型落地，本次编写者不执行代码修改，不自动启动子代理或训练。

**Goal:** 实现联盟负荷优先消纳风光、实际成交驱动的学习表示、可信服务评价和可验证的 HVAC 预测可达层。

**Architecture:** DR 投影先产生负荷；共享 Torch 内核完成分阶段能源分配，同时服务环境与两个算法的训练投影。纯指标模块统一训练、验证、正式测试与保存结果重算；HVAC 可达层在构造时刻状态前计算预测可行区间。

**Tech Stack:** Python、PyTorch、NumPy、pandas、PyYAML、highspy（已有依赖）、pytest、现有共享流水线。

**Spec:** [详细设计](../specs/2026-09-27-renewable-first-dispatch-and-metrics-design.md)。必须先读该文档；本计划不能替代公式、字段和禁止项。

## Global Constraints

- 不修改 algorithms/matd3/ 永久参考实现；只改共享模块和两个当前插件的适配/元数据。
- Actor 仍输出两个量：[ESS 有符号充放电请求，DR 动作]。
- 保持独立自然日；ESS、磨机、HVAC 不恢复日初状态。
- 不新增 P2P 成交奖励、舒适奖励权重或“鼓励卖电”奖励。
- 新模式首版为当前统一电价、无损、无限进口和无限 P2P 容量配置；有限历史配置走显式旧模式。
- 真实逐时舒适度不依赖 terminal_mode；无证据不能返回 100%。
- 不用未来真实天气构造 RL 安全层，不把未来成交放进 Actor 观测。
- 不覆盖已有结果、checkpoint、releases；不得把旧轨迹重新标为新调度规则结果。
- 不安装另一套求解器；优先使用已列入 requirements.txt 的 highspy。
- 正常训练/冒烟仅通过 scripts/run_experiment.py；最后两插件各完成完整短流程。

## Review Focus

1. 06→07、18→19、周末和24:00的状态时间不能错位；任务1/7有边界断言。
2. A输出风光同时从电网充电、A接收风光同时卖自己ESS都应合法；任务4/5/8验证来源账本。
3. 不同无效请求不能经历史观测产生不同下一状态；任务6进行全观测逐字段比较。
4. 缺数据、零分母、MILP失败不能被默认值0/1掩盖；任务1/3/9覆盖。
5. HVAC预测快照变化、隐藏真值变动、缓存更新与 replay 的区间必须一致；任务7覆盖且不另抽噪声。

## 0. 执行前置与交付记录

- [ ] 阅读根 AGENTS.md、配套设计和现有测试；不要重写已确认的语义。
- [ ] 保存 git rev-parse HEAD、git status --short、git diff --stat。保留用户现有改动，不清理未知文件。
- [ ] 确认可用 Python：输出 sys.executable 和 numpy/torch/pandas/highspy/pytest 版本，再运行 pytest -q，保存基线日志。
- [ ] 若 NumPy 导入即崩溃，先排查解释器/依赖环境，不能把“未收集测试”写成“算法测试失败”。本次已见默认 Anaconda base 在 NumPy BLAS 初始化处退出；pymgrid_env 曾可导入 Torch/NumPy，但未验证全部依赖。路径只是线索，不是指定部署环境。
- [ ] 建立实施记录（建议 docs/implementation_reports/2026-09-27-renewable-first-dispatch.md），包含开始HEAD、工作区状态、解释器、每任务命令/结果/耗时和偏离事项。报告目录不存在可创建。
- [ ] 查询原问题对应 run，只读取得日期列表、config snapshot、checkpoint路径/hash、旧hourly CSV及求解状态。找不到时用人工固定fixture验证代码，并明示历史数值未复现。

执行顺序：任务1→2→3先修统计；任务4→5→6改清算及学习；任务7加入HVAC预测可达；任务8对齐MILP；任务9完成版本、历史审计、端到端。不要先跑长训练再发现指标仍错。

## 任务1：统一 HVAC 与严格服务指标

**Files**

- 新增 evaluation/service_metrics.py。
- 修改 evaluation/evaluate_matd3.py、evaluation/service_reliability.py、utils/metrics_collector.py。
- 必要的实际状态字段来自 components/commercial_hvac.py、envs/multi_microgrid_env.py。
- 新增 tests/test_service_metrics_v2.py；更新 tests/test_rollout_constraint_metrics.py、tests/test_renewable_service_metrics.py。

**Interfaces**

    summarize_day_services(
        hourly_rows: Sequence[Mapping[str, Any]],
        *, config: Mapping[str, Any], physical_feasible: bool | None,
        schema_version: str
    ) -> dict[str, Any]

    aggregate_service_days(days: Sequence[Mapping[str, Any]]) -> dict[str, Any]

纯函数无 env.step、policy、MILP、磁盘写入。日结果含设计§5/8定义的计数、分母、率、data_complete及缺字段清单。聚合求计数之和，不平均每日百分比；缺样本返回null。

- [ ] 写 H1/H2/H3、X1 固定数据测试：20.593℃对21℃应越界0.407；terminal_mode=none不改变该结果；初步 strict=false。
- [ ] 加 no_occupied_boundaries、missing_temperature、wrong_timestamp、nonfinite_temperature 测试。新schema缺列/非有限值报错，旧审计模式标unknown；不要猜23℃。
- [ ] 运行新测试，记录预期失败位置；如果因导入环境失败，先恢复测试环境。
- [ ] 在逐时记录加入结果时间、实际温度、结果时刻温区/占用标记，区分 step flag 和累计 flag。
- [ ] 实现两个纯函数；将 _extract_rollout_constraint_metrics、_daily_service_metrics、_aggregate_service_metrics 改为使用共同定义。旧兼容适配可保留，不能复制计算逻辑。
- [ ] 终端关闭返回null/applicable=false；继续真实统计逐时舒适。strict 组合实际服务结果，缺数据标unknown。
- [ ] 测试同一轨迹在训练验证、正式评价、文件重算中输出一致；断言“有供电但温度不合格”会失败。
- [ ] 运行 python -m pytest -q tests/test_service_metrics_v2.py tests/test_rollout_constraint_metrics.py tests/test_renewable_service_metrics.py。
- [ ] 审查diff并记录本任务结果；只提交/暂存本任务文件（若实施会话要求Git提交），不得夹带用户修改。

## 任务2：EV 映射诊断与 ESS 削减账本

**Files**

- 修改 core/joint_action_projection.py、components/ev_charging.py、envs/multi_microgrid_env.py、utils/metrics_collector.py、evaluation/evaluate_matd3.py。
- 使用任务1的 evaluation/service_metrics.py 做聚合。
- 更新 tests/test_joint_action_projection.py、tests/test_metrics_recording.py、tests/test_ev_cohorts.py。
- 新增 tests/test_action_diagnostics_v2.py。

**Interfaces**

所有EV字段按设计§6；ESS kW/kWh字段按设计§7。内部JointActionProjector中分别返回 dr_mapping_delta、dr_safety_correction；dr_projected只能来自真正额外修正。EV物理plan与张量projector的请求量/修正量应一致。

- [ ] 写 E1/E2/E3：L=10/U=30/a=0映射20无投影；L=U算forced；下午G3计在场；供电少5单列delivery_shortfall。
- [ ] 写 S1：30=10削减+12成交+8未成交；Δt=.5只在累计电量时乘一次。
- [ ] 写环境producer→collector→daily→summary一致性测试，防止只给一个汇总字典打补丁。
- [ ] 运行新测试观察失败。
- [ ] 修改映射诊断，取消 cohorts 的旧固定小时筛选；available由时段开始sessions产生并保存，不从结果SOC猜。
- [ ] 在当前DR路径显式写 ess_p2p_offer_projection_reduction_kw；区分raw request缩放、offer cap、清算未成交。
- [ ] 新schema必需字段缺失应报错；历史审计只在uncapped/offer同时存在时可复算削减。
- [ ] 对正常EV动作断言 physical_plan.projected == tensor_plan.dr_projected == false；通用MG1/HVAC真实裁剪语义保持。
- [ ] 运行 python -m pytest -q tests/test_action_diagnostics_v2.py tests/test_joint_action_projection.py tests/test_metrics_recording.py tests/test_ev_cohorts.py。
- [ ] 审查通过后记录/提交独立改动。

## 任务3：磨机、总体合作增益与报告别名

**Files**

- 修改 components/industrial_mill.py（实际执行证据）、utils/metrics_collector.py、evaluation/service_metrics.py、evaluation/evaluate_matd3.py。
- 修改 evaluation/plotting.py、evaluation/paper_artifacts.py；核查 analysis/ 与 reports/ 中相关字段消费者。
- 新增 tests/test_cooperative_value_aggregation.py；在任务1测试内加入磨机情况。

**Interfaces**

    aggregate_cooperative_value(
        rows: Sequence[Mapping[str, Any]], *, monetary_tolerance: float
    ) -> dict[str, Any]

放在 evaluation/service_metrics.py（或将纯经济聚合单独放 evaluation/value_metrics.py，但调用名称/返回键保持）。返回设计§9明确列出的主比值、日均值、总分子/分母、有效/排除日期及原因。

- [ ] 写 M1/M2：10.53+16.51−23.40=3.64；coverage=16.51/23.40，服务可以100%；模拟供电不足时不能用计划产量。
- [ ] 写 R1：daily_mean=−49.75、total=4/10.01；添加R2无分母、负分子、MILP失败日期、微小负分母/真实负分母。
- [ ] 写缺字段/NaN不能被含入有效集合的测试；JSON allow_nan=False可写。
- [ ] 运行失败测试，再实现实际磨机产量/库存证据。库存裁剪前值必须保存，不能从已clip库存反推无违约。
- [ ] 去掉 terminal_mode 控制 production_coverage 的逻辑；废弃 max(terminal,production) 作为服务指标。
- [ ] 改 _build_joint_summary 主比值为ratio_of_sums；CSV日比值字段继续保持日级含义，汇总别名和schema明确区分。
- [ ] 修改图例/表格文字：“总体合作价值利用率”“日合作价值利用率分布”“当日产量覆盖率”“下游服务完成率”。库存与独立日假设同时可见。
- [ ] 运行 python -m pytest -q tests/test_cooperative_value_aggregation.py tests/test_service_metrics_v2.py tests/test_paper_artifacts.py。
- [ ] 记录/提交任务结果；确认奖励未改。

## 任务4：实现纯张量风光优先清算内核

**Files**

- 新增 core/renewable_first_dispatch.py。
- 修改 core/config_validation.py、core/joint_action_projection.py（模式分支，不在此步改环境状态提交）。
- 新增 tests/test_renewable_first_dispatch.py。

**Interfaces**

实现设计§4.1的 clear_renewable_first，返回该节完整ledger。新增内部函数 proportional_cross_flow(supply_kw:Tensor[B,N], demand_kw:Tensor[B,N])->Tensor[B,N,N]。前提为同节点供需不重叠；同一batch各样本独立。

- [ ] 写 D1–D8 的数字断言，特别D2按同时比例分配而不是节点先到先得；D4检查cap与unmatched分开。
- [ ] 写 permutation_equivariance、zero_supply、zero_demand、no_self_trade、batch_independence、source_conservation、charge_discharge_exclusion。
- [ ] 写dtype/device测试和连续区间有限差分测试；零分母分支梯度也必须有限，避免torch.where掩盖的0/0反向NaN。
- [ ] 运行失败测试，再实现A至H的流量计算。先完整风光负荷清算，再确定本地ESS，不能复用旧顺序的local值。
- [ ] capped ESS offers与实际成交分开；新能源负荷阶段不受ESS充电请求影响。
- [ ] 实现配置校验：新模式有限容量/非零损耗/异价不支持即报错；旧模式保持现有有限容量测试。
- [ ] 在 JointActionProjector 新模式输出实际三通道，保留旧模式明确分支；本步先以合成状态验证。
- [ ] 运行 python -m pytest -q tests/test_renewable_first_dispatch.py tests/test_joint_action_projection.py tests/test_p2p_capacity.py。
- [ ] 检查无numpy/item/float穿过动作梯度路径，记录/提交任务结果。

## 任务5：接入环境、来源守恒与真实状态提交

**Files**

- 修改 envs/multi_microgrid_env.py、envs/cooperative_energy_sharing.py（只重用价格/结果包装，避免再次清算）。
- 核查 core/grid_settlement.py、core/cost_accounting.py、core/microgrid_core.py。
- 更新 tests/test_renewable_dispatch_environment.py、tests/test_cooperative_physics.py；新增 tests/test_renewable_first_environment.py。

**Interfaces**

新模式 step 消费任务4 ledger；对外保留 info.plans、info.p2p、dr_devices、dr_state_after 等既有包装，并补新字段。只有可微清算外才可以转CPU/float。电池和DR commit 各调用一次。

- [ ] 用真实env fixture复刻D1/D3/D5/D6，并与张量ledger逐字段比较。
- [ ] 写新能源卖方同时电网充电、收到风光后卖自有ESS的合法测试；再写真正grid→P2P、P2P→export、ESS→ESS等非法账本测试。
- [ ] 写unmatched不扣SOC、charge_request不抢占风光负荷、node/system/source守恒、退化按实际吞吐计费。
- [ ] 运行失败测试，再替换旧“先本地ESS再allocate_structured”的DR新模式路径。
- [ ] 替换粗总量互斥assert为来源账本assert；只换assert不够，旧net结算不得重新造流量。
- [ ] 外送/弃电只来自本地剩余风光；网购分成负荷与充电；P2P付款总和为0，团队利润不重复加付款。
- [ ] ESS diagnostics由ledger完整填充；将计划/实际DR同时写入，任务1–3的统计读取实际执行。
- [ ] 运行 python -m pytest -q tests/test_renewable_first_environment.py tests/test_renewable_dispatch_environment.py tests/test_cooperative_physics.py tests/test_metrics_recording.py。
- [ ] 记录/提交任务结果；仍不开始长训练。

## 任务6：两个算法、观测与“无效请求等价”闭环

**Files**

- 修改 core/observation.py、core/global_state.py、algorithms/matd3_dispatch/algorithm.py、algorithms/matd3_hierarchical_transformer/algorithm.py。
- 适配 algorithms/matd3_hierarchical_transformer/actor.py 的字段索引/Trade token；核查 algorithms/diagnostic_sampling.py。
- 新增 tests/test_executed_action_equivalence.py；更新 tests/test_two_algorithm_smoke.py、tests/test_plain_single_dispatch.py、tests/test_matd3_hierarchical_transformer.py、tests/test_diagnostic_sampling.py。

**Interfaces**

两算法 project_actions(state,joint_raw) 返回相同新语义effective与ledger。原始action_dim=2，critic effective_action_dim=3。local/global schema 必须移除设计§4.3列出的无效要约历史字段；保留已有 peer_energy_summary 当前deficit/surplus/energy_balance等足够信息，避免重复添加同义字段。

- [ ] 写同状态A请求10/20/40但成交全0的测试：effective、reward、SOC、下一global state和所有下一observations相同。
- [ ] 原始请求和日志必须不同，以确认测试没有把输入本身抹掉；本质只是策略状态忽略未执行请求。
- [ ] 写positive demand未饱和梯度非零；no demand且离拐点时ESS梯度0；跨agent DR改变peer成交的梯度保留。
- [ ] 写卖方竞争测试：联盟总成交同为10，但A/B成交5/5与8/2时，逐agent Critic输入与下一SOC不同。
- [ ] 运行失败测试，再移除策略观测/state中的无效请求历史依赖；logger仍保留原报价、削减和未成交。
- [ ] 适配Transformer语义索引，不以补零让旧权重“兼容”；共享norm及schema hash升级。
- [ ] 遍历replay regression、target actions、actor loss以及诊断优化路径，统一实际三通道。训练时不重新读env当前状态替代batch保存状态。
- [ ] 新checkpoint同版本往返输出一致；旧v2在新环境拒绝；两个插件的环境hash相同。
- [ ] 运行 python -m pytest -q tests/test_executed_action_equivalence.py tests/test_two_algorithm_smoke.py tests/test_plain_single_dispatch.py tests/test_matd3_hierarchical_transformer.py tests/test_diagnostic_sampling.py。
- [ ] 记录/提交；不得把“Actor请求仍为正”单独判为失败。

## 任务7：HVAC 预测可达区间与未来信息隔离

**Files**

- 新增 components/hvac_viability.py。
- 修改 components/commercial_hvac.py、core/forecast.py、envs/multi_microgrid_env.py。
- 必要时修改 core/observation.py、core/global_state.py 的可行区间准备顺序，避免循环调用。
- 新增 tests/test_hvac_viability.py；更新 tests/test_controllable_loads.py、tests/test_dr_physics_audit.py、tests/test_constraint_aware_observations.py。

**Interfaces**

在新文件定义冻结数据类：

    HVACForecastSnapshot:
        snapshot_id: str
        boundary_timestamps: tuple[str, ...]  # len=H+1
        outside_temperature_c: ndarray[H]
        ghi_w_per_m2: ndarray[H]
        base_load_kw: ndarray[H]
        interval_occupied: ndarray[H]
        comfort_low_c: ndarray[H]             # 对应结果边界1..H
        comfort_high_c: ndarray[H]

    HVACViabilityResult:
        min_signed_power_kw: float
        max_signed_power_kw: float
        feasible: bool
        status: str
        current_min_slack_c: float
        future_min_slack_sum_c: float
        snapshot_id: str

    solve_hvac_viability(
        initial_air_c: float, initial_mass_c: float,
        snapshot: HVACForecastSnapshot,
        *, config: Mapping[str, Any], delta_t: float
    ) -> HVACViabilityResult

使用highspy已有API，求解上下限与必要fallback。环境提供快照和区间给HVAC实例；state()/plan()消费缓存，不各自构造预测。

- [ ] 写2R2C线性矩阵与现有_next_state一致性测试（正/负/零热功率、非1h）；先确保模型完全相同。
- [ ] 写人工提前预热反例：one-step可选动作会使后续07:00不可达；新可行区间排除该动作且仍非空。保存数值fixture而非“多跑几天看看”。
- [ ] 写准确固定预测下逐步递推可达测试；验证min/max首动作均有满足全未来边界的后续轨迹。
- [ ] 写名义模型不可行、solver异常、forecast不足日剩余时段的测试。预测窗口不足新模式明确报错，不用未来真值补齐。
- [ ] 写H6信息隔离：不改当前forecast的隐藏真值变更不影响bounds；snapshot变化必须失效缓存。
- [ ] 运行失败测试，再实现LP：固定初始双温状态、每步设备热功率界、结果时刻温区、无日末恢复约束。
- [ ] 实现设计§5.3的分层松弛fallback；物理不可行与solver failure分开，状态/日志保留slack。
- [ ] SharedForecastProvider公开同缓存的GHI；检查本地GHI列与现有热模型一致。保留已有噪声抽样序列或明确版本变化，不额外调用随机数导致对比漂移。
- [ ] reset及每step结束后为下次观测准备快照/区间；当前动作前不重复求解。t==T无未来LP，返回终止状态约定，不能访问frame[T]。
- [ ] 正常训练状态保存精确准备的HVAC区间；Torch投影用相同区间裁剪，实际plan与投影逐步断言一致。
- [ ] 准确天气的2018-01-29回归若能找回证据则执行；否则报告缺checkpoint，仅交付人工可复现反例，不能写已复现历史日期。
- [ ] 运行 python -m pytest -q tests/test_hvac_viability.py tests/test_controllable_loads.py tests/test_dr_physics_audit.py tests/test_constraint_aware_observations.py tests/test_service_metrics_v2.py。
- [ ] 报告LP每步耗时/调用次数、不可行次数；记录/提交。

## 任务8：MILP 来源优先、合法交易与对照边界

**Files**

- 修改 optimization/common_constraints.py、optimization/cooperative_dispatch.py、optimization/independent_dispatch.py（如调用需适配）、optimization/oracle_replay.py。
- 修改 evaluation/coalition_evaluator.py、evaluation/compare_matd3_milp.py、evaluation/dr_reference.py 的元数据/参考策略适配。
- 新增 tests/test_renewable_first_oracle.py；更新 tests/test_full_information_oracle.py、tests/test_optimization_and_allocation.py、tests/test_dr_reference.py。

**Interfaces**

沿用现有 solve_dispatch/DispatchResult 公共接口，增加metadata说明source_priority_free_allocation及perfect_foresight。validate_device_trajectories继续只证明设备轨迹，不把函数名换掉后宣称证明P2P可执行。

若新增动作重放，接口固定为 compare_oracle_market_replay(oracle_result, replay_hourly_rows, *, atol)->dict，返回设备误差、各来源流量误差、profit_gap、exactly_replayable。不一致是需要解释的对照自由度，不能抹去。

- [ ] 写单成员、二成员、三成员来源优先测试；联盟外风光不能进入；P2P关闭对照为独立。
- [ ] 写D3合法、收到风光后ESS向外卖合法的优化用例；grid转售、ESS→ESS/外网均为0。
- [ ] 写“有可用风光时不以ESS/电网供本可覆盖的负荷”的测试；有剩余风光先满足允许充电再外送/弃电的目标/约束口径须审查。
- [ ] 运行失败测试，再将旧peer_load拆成新能源与ESS来源；去掉总量grid_mode互斥，保留真实grid import/export互斥及来源约束。
- [ ] 增加当前无容量模式的风光负荷优先约束；推导有限big-M并写上界证明注释/测试，不放任意10000。
- [ ] 保留自由分配但明确metadata。拿一个规则环境可行轨迹固定到MILP变量中检查可行性，证明主要约束没有比规则环境更窄。
- [ ] 检查HVAC使用t+1温区、独立日不恢复、EV场景一致。RL有实际舒适违约时，不称其收益为可行算法解。
- [ ] 原有七联盟/Equal/Shapley都运行；求解失败保留None和状态，不用0填价值。
- [ ] 运行 python -m pytest -q tests/test_renewable_first_oracle.py tests/test_full_information_oracle.py tests/test_optimization_and_allocation.py tests/test_dr_reference.py。
- [ ] 记录/提交并列出MILP剩余优势：完美预见、自由分配、已有EV组间自由度；不宣称完全相同动作集合。

## 任务9：版本、配置、历史审计和全流程验收

**Files**

- 修改 configs/cooperative.yaml、configs/action_designs.yaml、experiments/profiles.py，两个当前插件checkpoint签名及需要的自身YAML。
- 修改 experiments/checkpoint_selector.py、experiments/training_pipeline.py（仅真实指标适配，不复制流程）。
- 更新 docs/SYSTEM_MODEL.md、README.md及相关历史交接文档的过时标注。
- 新增 scripts/audit_saved_service_metrics.py；更新 tests/test_checkpoint_selector.py、tests/test_experiment_profiles.py、tests/test_two_algorithm_smoke.py。

**Interfaces**

新增保存结果审计CLI：

    python scripts/audit_saved_service_metrics.py --run <原run目录> --output <独立新审计目录>

必须拒绝output等于run、位于run内、覆盖已存在审计文件、位于outputs/releases内。输出audit_manifest.json（源文件hash/日期/配置/新指标版本/缺字段）、audit_daily_metrics.csv、audit_summary.json。只读保存文件，不调用环境/策略/MILP/训练。

- [ ] 写审计源文件hash前后不变、输出路径拒绝、缺证据null、同轨迹多消费端一致、旧checkpoint拒绝测试。
- [ ] 写checkpoint必需服务指标缺失/不合规不eligible；末温恢复关闭不被误当失败；磨机产消覆盖不足但服务完整不被误判。
- [ ] 运行失败测试，再实现审计CLI和版本元数据；所有checkpoint/API调用方同步。
- [ ] 明确cooperative.yaml采用renewable_load_first及forecast_viability；两当前算法使用相同共享配置，不改网络超参数作为修复的一部分。
- [ ] 同步文档：去掉旧75kW现行描述、日末恢复现行描述、要约即实际动作的说法；补适用版本，不删除历史证据。
- [ ] 全量运行 python -m pytest -q；保存完整日志。不能只挑通过的测试汇报。

### 9.1 两插件短流程命令与条件

默认batch_size=128、warmup_steps=1000，3天72步可能完全没有有效参数更新，不能直接据此宣称训练通过。

分别复制两个算法YAML到操作系统临时目录（使用Python tempfile，不修改正式长期训练配置）。保留算法名/网络参数，仅将algorithm.batch_size=8、warmup_steps=8、replay_capacity=256、logging.diagnostic_interval=1。每个算法一份独立副本；run中保存实际合并配置。不要把MATD3的配置副本给普通T。

~~~powershell
python scripts/run_experiment.py --algorithm matd3 --algorithm-config <MATD3临时YAML> --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --device cpu --no-dashboard
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer --algorithm-config <普通T临时YAML> --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --device cpu --no-dashboard
~~~

尖括号是实施者须替换的已创建临时路径，不可原样执行。命令参数已按当前build_parser核对，执行前若代码变动需再次看--help。不加--no-milp或--no-validation，不把输出指向releases。

- [ ] 每个run至少观察一次critic更新及一次delayed actor更新，loss/梯度有限；普通T不能只验证构造。
- [ ] 正式测试实际load checkpoints/best_model.pt，不只是文件存在；在metadata或日志留可核验路径/hash。
- [ ] 七联盟全部有效，Equal/Shapley预算平衡，summary完成；没有求解器依赖则明示验收阻塞，不跳过后宣称完成。
- [ ] 每run恰有15 PNG和15 PDF；新图复用既有图位，不无意增加/减少数目。
- [ ] 检查config snapshot的插件/YAML、新版本字段、相同physical_config_hash、working_runs/<slug>/runs布局。
- [ ] 从保存CSV独立复算H/E/S/M/R五类指标，与daily/summary/图示一致。
- [ ] 比较实施前baseline的canonical目录diff，必须无本次变更；保留releases和所有历史run。

### 9.2 最终交付报告模板

~~~text
实现基准HEAD / 最终HEAD或diff标识：
解释器和依赖版本：
已实现的语义版本：
全量pytest：命令、通过/失败/跳过数、日志路径：
MATD3短run：路径、插件、配置、best checkpoint、参数更新次数：
普通T短run：同上：
七联盟MILP、Equal、Shapley、图数：
无效报价等价测试：Critic/下一观测/梯度分别是否通过：
HVAC历史日期是否复现、准确预测回归、噪声预测实际违约：
EV映射/强制/修正/未交付计数：
ESS uncapped=削减+成交+未成交 守恒：
磨机产量覆盖与服务率、库存收支：
CVCR总体分子/分母/有效日期/日均值：
旧结果审计目录与未能复算的字段：
MILP自由度和重放差异：
canonical/release保护检查：
未通过、未实现、未知项（不得空泛写“基本完成”）：
~~~

## 10. 给落地模型的可复制指令

> 先阅读仓库AGENTS.md，以及本实施规范和链接的设计文档。按任务1至9逐步实施，先用固定用例暴露问题，再改代码；每任务通过定向检查后再进入下一任务。不要修改永久algorithms/matd3，不要更改容量/电价/奖励掩盖错误，不要碰已有releases。特别落实“实际成交进入Critic、无效要约不进入下一策略状态”的等价性测试，以及HVAC逐时舒适独立于终端恢复。新调度使用同一可微清算内核，不能只改环境或只改某一算法。全部完成后运行全量pytest和两插件短流水线，按§9.2报告证据。若代码现状与文档接口冲突，先记录具体冲突并按目标语义做最小适配；不能擅自删除需求或用假默认值通过验收。历史复算数字仅作旧结果核验，不能硬编码为新结果。
