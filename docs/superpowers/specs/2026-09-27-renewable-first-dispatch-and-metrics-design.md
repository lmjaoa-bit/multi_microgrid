# 联盟风光优先调度与评价可信性修复：设计规范

日期：2026-09-27。用途：交给另一模型实施；本文是目标规范，不是实现完成报告。

配套：[实施步骤与验收清单](../plans/2026-09-27-renewable-first-dispatch-and-metrics.md)。执行者必须同时阅读两份文档及仓库根目录 AGENTS.md。

## 1. 决策、范围与证据状态

### 1.1 用户已经确认的调度顺序

每小时：DR 决定实际负荷 → 本地风光供本地负荷 → 联盟富余风光供其他微网负荷 → 本地 ESS 供本地剩余负荷 → 其他 ESS 供剩余负荷 → 电网兜底。联盟负荷优先于任何 ESS 充电。

Actor 仍输出两个量：[ESS 有符号充放电请求，DR 动作]。不要增加独立 P2P 比例、报价、交易对象或交易方向动作。Actor 决定请求，清算规则决定实际交付。未成交的 ESS 请求不放电、不扣 SOC。

本次交付还必须处理：HVAC 合规虚报与未来占用可达性、EV 投影统计错误、ESS 要约削减漏记、磨机服务与生产口径混淆、合作价值利用率聚合错误。

### 1.2 本文采用的代码核对基准

编写时 HEAD：c45422ab074847f7bfb9397c1c1bb191558cc2a7。实施前重新记录 HEAD、工作区 diff 和运行环境。路径及函数名比旧行号可靠。

已通过源码核对：

| 问题 | 代码位置/机制 | 结论 |
|---|---|---|
| 风光尚未联盟清算就安排本地 ESS | core/single_dispatch.py；envs/multi_microgrid_env.py 的 DR step | 需要改变顺序 |
| HVAC 舒适指标随终端开关一起关闭 | evaluation/evaluate_matd3.py 的 _extract_rollout_constraint_metrics | terminal_mode=none 时错误返回舒适度 1.0 |
| 严格服务合规未含实际温度越界 | evaluation/service_reliability.py 的 _daily_service_metrics | 需要加入逐时服务证据 |
| EV 正常映射被记为投影 | core/joint_action_projection.py；环境 dr_projection_delta | 映射坐标与全区间归一化坐标被直接比较 |
| EV 评价仍用旧固定窗口 | evaluation/evaluate_matd3.py | cohorts 分支也受 available_start_hour/departure_hour 筛选 |
| ESS 削减量生产端缺字段 | envs/multi_microgrid_env.py 的当前 DR 路径；utils/metrics_collector.py | 消费端默认 0 掩盖漏写 |
| 磨机生产率被终端开关置 1 | evaluation/evaluate_matd3.py | 不代表当日产量覆盖消耗 |
| 合作价值利用率取日比值均值 | evaluation/evaluate_matd3.py 的 _build_joint_summary | 不能当成全测试集比值 |

用户提供的历史复算证据（本次文档编写未重新运行对应最佳模型）：

- 2018-01-29，06:00–07:00 后室温 20.593℃，07:00 下限 21℃，制热 35 kW；提前调整 HVAC 动作可达到 21℃。
- 占用边界 479/480 合规，99.79%；全服务日 54/55 合规，98.18%。分母必须追溯到原测试日期及数据，不能硬编码。
- EV 两种旧投影统计 90.68%、68.83%，不能解释为安全层拒绝率。
- ESS 要约削减 MG1/MG2/MG3 日均 38.79、22.96、0.31 kWh，原记录却为 0。
- 磨机日均生产 16.51 t、消耗 23.40 t；初始库存 10.53 t、日末均值约 3.64 t。
- 合作价值利用率：日比值均值 −188.77%，按合作增益之和计算 36.84%。

以上为历史回归线索，不是新规则下的目标数值。找不到原 run/checkpoint 时，必须报告“历史数值未复现”，不得编造 run 路径，也不得为接近旧数值调整物理参数。

### 1.3 禁止扩大或偷换范围

- 不修改 algorithms/matd3/ 永久参考实现；只改共享模块和两个当前插件的适配/元数据。
- 不复制训练、评价、结算或绘图流程；CLI --algorithm matd3 继续映射 matd3_dispatch。
- 不调整电价、容量、效率、退化成本、初始库存、舒适温区、EV 场景生成及原有物理参数来掩盖问题。
- 保持独立自然日；ESS、磨机、HVAC 不恢复日初状态。不改成跨日连续运行，不添加库存末值、SOC 末值或温度末值惩罚。
- 不新增 P2P 成交奖励、舒适奖励权重或“鼓励卖电”奖励；本次采用约束和真实评价修复。
- 不覆盖已有结果、checkpoint、releases；不得把旧轨迹重新标为新调度规则结果。
- 本次是跨算法共享环境研究，不得宣称为某个网络单独的算法改进。

## 2. 运行模式及版本契约

目标模式配置：在共享配置中加入 dispatch.priority_mode: renewable_load_first。旧配置缺该键时保留显式命名的 legacy_local_first 行为；不得让缺字段的旧配置静默获得新物理语义。当前 cooperative.yaml 必须显式选择新模式。

新模式首版支持当前研究设定：零损耗、统一购售电价、无限电网进口、P2P 无额外双边容量、禁止 ESS→ESS、禁止 ESS→外部电网、禁止电网/P2P 中转转售。P2P 关闭时退化为各自独立的同源优先顺序。

有限 P2P 容量、非统一电价或损耗模型保留在旧模式回归路径。新模式收到不支持的组合时启动即明确报错，不得忽略配置或静默切旧规则。无需在本次加入一般网络优化器。

同时保存以下元数据：

- dispatch_semantics: renewable_load_first_v1
- action_semantics: single_dispatch_dr_v3
- metrics_schema_version: service_metrics_v2
- hvac_safety_mode: forecast_viability 或 legacy_one_step
- oracle_allocation_mode: source_priority_free_allocation
- full_information_oracle: true（MILP 对照）

新共享默认启用 forecast_viability；保留旧模式仅用于旧模型审计。旧 checkpoint 对新语义应明确拒绝。动作投影签名、观测 schema、checkpoint 校验、configs/action_designs.yaml、experiment_contract 和两个插件必须一致。不要改用户界面的 MATD3、普通 T 名称。

## 3. 新调度的数学定义

### 3.1 输入、单位与请求

每步 Δt 小时，功率统一 kW，电量 kWh。L_i 是 DR 约束处理后的总电负荷，R_i 是当前本地风光出力。所有 SOC/设备上限使用时段开始状态。

ESS 原始动作 u_i∈[−1,1]：正值请求放电，负值请求充电。继续沿用当前按 SOC 可行最大功率缩放的规则：

    Dreq_i = max(u_i,0) × feasible_discharge_max_i
    Creq_i = max(−u_i,0) × feasible_charge_max_i

两个请求互斥。请求不是实际功率，不能先按请求更新电池。

### 3.2 清算阶段：每阶段的余额只能消费一次

| 阶段 | 来源 | 去向 | 明确规则 |
|---|---|---|---|
| A | 本地风光 | 本地总负荷 | r_local=min(R,L)；S=R−r_local；D=L−r_local |
| B | 所有微网剩余风光 S | 其他微网剩余负荷 D | 清算后得到 S1、D1；充电请求不得预占 S |
| C | 本地 ESS 放电请求 | 本地剩余负荷 D1 | E_local=min(Dreq,D1)；D2=D1−E_local |
| D | 剩余 ESS 放电请求 | 其他微网剩余负荷 D2 | 未限额要约 Oraw=Dreq−E_local；本地还有 D2 的节点不得成为 ESS 卖方 |
| E | 本地剩余风光 S1 | 本地 ESS 充电请求 | 仅请求充电的电池参与；取 min(S1,Creq) |
| F | E 后剩余风光 | 其他 ESS 剩余充电请求 | 不接受 ESS 来源，不得把 P2P 输入再次转卖 |
| G | 电网 | 各节点剩余负荷，然后剩余 ESS 充电请求 | 负荷始终优先；当前进口无限，旧有限模式另行保留 |
| H | 原属本节点的剩余风光 | 外部电网、弃电 | 每节点外送不超过 100 kW；剩余弃电 |
| I | 已实际交付的所有来源 | 状态提交与成本结算 | ESS、DR 仅提交一次；未成交 ESS 留在电池中 |

阶段 E/F 虽列在电网 G 前，但不得消费原本能用于联盟负荷的风光；负荷的 A/B 优先已经完成。无损全连接模式中，如果 B 后仍有联盟剩余负荷，联盟剩余风光应为 0（数值误差范围内）。

阶段 D 的要约上限使用 D2，不使用原始负荷缺口，不含任何 ESS 充电需求。先逐卖方裁剪 O_i=min(Oraw_i,Σ其他D2)，再统一清算；多个 ESS 同时竞争必须同时缩放，不能依照 agent 字典顺序抢占。

### 3.3 当前无容量模式的确定性分配

每个跨网阶段输入 s_i≥0、d_i≥0，且同一节点 s_i×d_i=0。定义 S=Σs、D=Σd。S 或 D 为零时流量全零，否则：

    F_ij = s_i × d_j / max(S,D), i≠j；F_ii=0

因此总成交量=min(S,D)，卖方和买方均按比例分配，不依赖遍历顺序。A/B、C/D、E/F 的局部分配保证买卖集合不相交，必须在测试中检验这一前提；不要把公式直接用于买卖集合重叠的其他市场。

所有阶段保留来源/用途独立矩阵：res_peer_load、ess_peer_load、res_peer_charge。汇总 pair_flows 仅供报告，不能代替来源账本。P2P 价格保持现有中间价和预算平衡；内部付款不额外进入联盟奖励。

### 3.4 两个必须允许的组合

例一：A 本地负荷 30，收到 B 的风光 20，A 请求 ESS 放电 25。A 本地 ESS 供 10，剩余要约 15 可供 C；B→A 是风光，A→C 是 A 自己的 ESS，不是转售。

例二：A 风光 30、负荷 0、ESS 请求充电 30；B 缺 30。先 A→B 风光 30，再电网→A ESS 30。这是用户确认的负荷优先规则。A 同时 P2P 卖电和电网买电并不代表电网转售。

因此必须替换环境和 MILP 中“只要 grid_import>0 且 p2p_sell>0 就非法”的总量条件。禁止行为应由来源守恒证明：grid_load/grid_charge 没有 P2P/外送去向；收到的 P2P 电量只能进入指定终端负荷或充电，不进入出口。

### 3.5 状态与成本守恒

    E_dis_actual = E_local + Σ_j ess_peer_load[i,j]
    E_ch_actual = res_local_charge + Σ_j res_peer_charge[j,i] + grid_charge
    SOC_next = SOC + (η_ch E_ch_actual − E_dis_actual/η_dis) Δt / Capacity

各节点：R + E_dis_actual + grid_import + p2p_import = L_served + E_ch_actual + grid_export + p2p_export + curtailment。

R 的所有去向之和必须等于 R；P2P 总收发相等；电池充放互斥；ESS 实际放电≤请求；充电≤请求。不得对已经执行的新能源/P2P 流量再次使用净功率补算一遍。

## 4. 环境与训练采用同一清算内核

### 4.1 结构决策

新增 core/renewable_first_dispatch.py，提供纯张量清算内核。core/joint_action_projection.py 先处理 DR，再根据模式选择新内核或原 legacy 路径。envs/multi_microgrid_env.py 消费内核返回的完整 ledger，不再重新调用旧 allocate_structured 全阶段清算。

新内核只读状态和动作，不更新 SOC，不写文件，不运行电网结算器，不访问设备对象或 pandas。全部关键流量保持 Torch 张量，允许批次和 device/dtype 传递。

推荐固定接口：

    clear_renewable_first(
        renewable_kw: Tensor[B,N], load_kw: Tensor[B,N],
        charge_request_kw: Tensor[B,N], discharge_request_kw: Tensor[B,N],
        *, p2p_enabled: bool
    ) -> dict[str, Tensor]

字典最少包含：res_local_load_kw、res_peer_load_kw[B,N,N]、load_after_renewable_kw、ess_local_load_kw、ess_offer_uncapped_kw、ess_offer_cap_kw、ess_offer_kw、ess_offer_projection_reduction_kw、ess_peer_load_kw[B,N,N]、ess_unmatched_kw、res_local_charge_kw、res_peer_charge_kw[B,N,N]、grid_load_kw、grid_charge_kw、renewable_surplus_for_export_kw、actual_charge_kw、actual_discharge_kw。矩阵第一节点下标永远是来源，第二下标是去向。

grid_* 在当前无限进口模式代表实际分配；export/curtailment 由现有外部结算逻辑按 100 kW 分拆，但必须与 ledger 一致。物理结算验证不能改写 ledger 的成交量。

### 4.2 Critic 动作表示的最终选择

保留每微网三个 Critic 通道，但新版本含义明确为：

    [(ess_local_actual − actual_charge)/scale,
     ess_peer_actual_sell/scale,
     effective_DR_normalized]

scale=max(max_charge_kw,max_discharge_kw)。第二通道改为实际 ESS 售电，而不是旧要约；因此必须升级语义，拒绝旧 checkpoint。要约和原始动作仍完整记录，不能删除诊断。

这让纯物理上无效的 ESS 要约不再通过第二通道直接放大。但不宣称消除了所有等价动作：饱和区仍有零梯度。必须同时完成下一小节的观测处理，不能让取消的要约通过历史观测继续改变下一步策略。

Replay 继续保存原始二维动作及当时 global_state/next_global_state。critic regression、target action、actor loss 都重新调用同一投影。禁止只改 rollout、只改 replay 或只改某一个算法。

跨微网依赖不可 detach：改变 A 的 DR 会改变 B 的可成交 ESS；Actor 梯度必须穿过整个联合清算。不能用 numpy、item、Python float 或原地覆盖把学习路径截断。验证有限差分应避开 min/max 拐点；拐点只要求有限梯度。

算法诊断优化路径 algorithms/diagnostic_sampling.py 也要检查，不能仍假定第二通道是要约。

### 4.3 用户补充的强制要求：智能体必须区分请求与实际作用

用户特别指出“Critic 看到10/20/20 kW要约，实际成交都是0”的问题。本条是核心验收项，不是未来可选优化。

新语义下，未执行请求只进入审计日志，不进入下一步策略观测或 Critic state。移除动作依赖的 previous_ess_p2p_offer、previous_ess_p2p_unmatched、previous_system_sell_offer、previous_clearing_ratio；previous_system_buy_demand 也不得继续采用未交付请求构造。相同物理执行的两个请求不应通过这些历史字段人为制造不同下一状态。不要为了“维度不变”把旧字段默默换义；明确升级 schema。

保留已有 previous_ess_p2p_accepted、previous_p2p_net、previous_grid_net、previous_system_p2p 等实际执行字段；其他历史字段逐项证明只依赖实际物理执行才可保留。日志中仍保存原始/限额/成交/取消量，便于研究者诊断，但不必把所有日志字段喂给 Actor。

Actor 的 peer_energy_summary 必须保留当前可用的风光余量、负荷缺口及已有预测摘要。若现有字段未完整覆盖，则使用各节点当时可公开的 R 和基础负荷构造 public_base_res_surplus_kw=max(R−base_load,0) 与 public_base_load_deficit_kw=max(base_load−R,0)，按既有 normalizer 归一化并显式命名。它们不是 DR 决策后的精确净需求。禁止把尚未执行的其他 Actor 同时刻动作、未来真实成交或未来真实天气提前输入分散 Actor。

在固定状态下，A的10/20/40 kW无效ESS请求若最终都成交0，必须满足：Critic三个物理通道相同；实际SOC/收益相同；global_state_after与下一步各Actor观测相同（固定随机种子和外生量）；若处于完全无需求的平坦区域，该无效ESS动作对全部Critic输入的局部导数为0。成交区间未饱和时则必须有正确非零梯度。

必须反向检查另一种情况：两个卖方竞争时，A提高请求可能改变A/B实际放电、SOC和未来价值，这不是无效动作，不能用全局总成交量相同就压成同一Critic表示。保留逐微网实际通道，不只输入联盟总成交。

不添加“未成交罚金”、P2P奖励或假梯度来逼交易；不要求Actor原始输出本身为0。验收目标是它学习实际执行的价值，而非把原始动作画得更好看。

## 5. HVAC：真实指标与未来可达性分别解决

### 5.1 时间对齐和真实舒适度（必须先完成）

一条记录明确保存 interval_start_timestamp、state_after_timestamp。动作 t 作用于 [t,t+1)，外生热增益按 t；结果温度 T_air(t+1) 必须使用 t+1 的占用标记及舒适温区。

工作日 06:00→07:00 的结果属于 07:00 占用边界，下限 21℃；18:00→19:00 的结果按 19:00 非占用温区；周末依原工作日规则；24:00 边界沿用当前模型的非占用温区。若以后改成跨日模型需另立研究。

每步记录（以 mg2_ 前缀持久化）：

- hvac_occupied_after，hvac_comfort_low_after_c，hvac_comfort_high_after_c。
- t_air_before_c、t_air_c（实际提交后）、t_mass_before_c、t_mass_c。
- hvac_comfort_violation_c=max(low−T,0)+max(T−high,0)。
- hvac_comfort_violation，hvac_occupied_comfort_violation。
- hvac_step_comfort_infeasible：该时刻模型可行性；hvac_episode_comfort_infeasible：当日累计。不得把现有粘滞布尔 comfort_infeasible 当逐步越界次数。

温度判定容差 1e−6 ℃。必须用实际温度计算，不能用预测温度、有没有投影、设备是否已满功率或有没有缺电代替。供电足够也可能温度违约。

terminal_mode=none 只关闭末温恢复指标/成本：hvac_terminal_joint_success_rate=null，terminal_restoration_applicable=false；不能关闭逐时舒适度。无样本的率返回 null，另存计数与适用性，不返回 100%。

occupied_comfort_rate=合规占用边界数/全部占用边界数。全测试集用计数汇总，不能平均每日百分比。额外报告全部边界合规率、最大越界℃、越界边界数、越界天数。离散边界检查不能宣称整个小时连续温度均合规。

### 5.2 提前预热/预冷：基于可用预测的可达区间

当前 one-step bounds 只能约束下一步。新增 components/hvac_viability.py；不改 2R2C 参数，不扩大舒适带，不新增末温恢复。目标是：在当前可用预测下，保留未来舒适可达的 HVAC 动作区间。

使用热功率 q 而不是同时加热/制冷两个变量：q∈[−COP_cool P_cool_max, COP_heat P_heat_max]。在该变量下，现有 2R2C 离散方程为线性：x[k+1]=A x[k]+B q[k]+b[k]，x=[T_air,T_mass]。A、B、b 必须从现有方程推导并与 _next_state 单步对齐，不能另换积分方法。

每时刻对本自然日剩余区间 t…T−1 建立线性可行问题，对所有结果边界 t+1…T 施加对应舒适上下限。分别最小化、最大化当前 q[t]，得到保留未来可达性的 [q_low,q_high]。凸线性可行域在首个标量上的投影是区间，区间中每个 q[t] 都应有一条预测可行的后续轨迹。

将 q 区间按已有 COP 分段单调转换到有符号电功率区间，仍由 Actor 在区间内提出/被裁剪动作。不要独立同时最小化 T_air 和 T_mass 再拼一个虚假的二维可达箱。

未来外生量只能来自当前 SharedForecastProvider：未来温度、基荷、太阳辐照预测，加已知占用日程。当前时段可用真实当前测量。禁止读取 episode_frame 的未来真实行作为安全层预测，即使文件已经在内存中。MILP 完美预见是单独标注的对照权限。

若预测接口缺热模型需要的 GHI，则向同一个 forecast provider 暴露其当前缓存中的预测 GHI；不得用实际未来 GHI 补洞，也不得将 PV 功率不经验证地当作 GHI。缓存不得额外抽样噪声。

环境每个时刻先准备一次 immutable 预测快照和可达区间，再构造 local/global state，再执行动作。Actor、环境 DR plan、replay 中的可行区间必须来自同一次计算。缓存键至少含日期/t、T_air、T_mass、forecast_snapshot_id、配置语义；预测或状态变化必须失效。

训练时不在 replay batch 内再次求 LP；使用存入 state 的 hvac_feasible_min/max。这些区间由时刻开始状态和预测决定，与本次 ESS/DR 原始动作无关。下一状态也须保存该下一时刻已准备的区间。避免 state()→feasible_bounds()→state() 递归。

### 5.3 不可行和求解器失败必须区分

- 硬可行 LP 得到 optimal：返回首步 min/max；验证上下限有限、顺序正确、设备范围合法。
- proven infeasible：先最小化当前下一边界舒适松弛，再固定该最小值（1e−7 求解容差），最小化后续边界松弛总和，最后求首步允许范围。返回 degraded/infeasible=true 和最小松弛；不得谎称安全保证。
- 求解器异常、超时、非有限解：明确 solver_failure，不得当成物理不可行。训练/正式测试停止并报告；不静默退回满功率或旧逻辑。

新增 hvac_viability_feasible、hvac_viability_current_min_slack_c、hvac_viability_future_min_slack_sum_c、hvac_viability_status、hvac_forecast_snapshot_id 诊断，分别对应实施规范 HVACViabilityResult 的同义字段。舒适率始终从实际温度独立计算。

该方法只保证固定名义预测模型下的可达性。预测误差、预测更新、有限供电可能造成实际违约，必须如实记录。不得宣称任意天气下严格 100% 合规；鲁棒安全层不在本次范围。

历史 2018-01-29 用例：优先恢复相同初始状态、真实天气、旧模型动作序列；冻结完全准确的测试预测时，可达层应在 06:00 前开始必要干预，并使 07:00 达到 21−1e−6℃（前提是整条约束可行）。不允许靠修改初始温度或设备容量通过。

### 5.4 严格服务日指标

physical_feasible_day 保留 SOC、节点功率/P2P 守恒等物理判定。strict_feasible_day 必须同时满足：物理合规、无未供电、无 DR 供给不足、全部 HVAC 边界在适用温区内、磨机下游服务无短缺且库存约束成立、EV 实际离站事件无短缺，以及显式开启的终端条件。

逐时服务和终端恢复分开。缺少必需温度/库存/事件数据时，strict_feasible_day=null，strict_service_data_complete=false；不按 0 缺口自动判真。checkpoint 的必需约束遇缺数据必须不合格。

## 6. EV：把动作映射、强制程度和未交付分开

三群组 EV 的 kW 动作定义保持：z=(clip(a,−1,1)+1)/2；P_mapped=L+z(U−L)。若 L/U 在 state 中为归一化量，先使用相同 scale 还原单位，不能混合 kW 和归一化数值。

字段：

| 字段 | 含义 |
|---|---|
| ev_action_coordinate | z，区间中的位置 |
| ev_feasible_min_kw / ev_feasible_max_kw | 动作执行前的 L/U |
| ev_mapped_request_kw | P_mapped |
| ev_planned_power_kw | 组内分配后的计划总功率 |
| ev_served_power_kw | 实际获得电力的总功率 |
| ev_action_mapping_delta_kw | P_mapped−z×station_max_kw，仅描述坐标变换 |
| ev_safety_correction_kw | P_planned−P_mapped，真正额外安全修正 |
| ev_safety_corrected | abs(correction)>1e−6 kW |
| ev_delivery_shortfall_kw | max(P_planned−P_served,0) |
| ev_available / ev_controllable | 时段开始有在场组；U−L>1e−6 |
| ev_forced_action | 有在场组且 U−L≤1e−6；不等于投影失败 |

正常映射不会增加 dr_projected 或 safety_correction。L=U 不算“被安全层否决”。没有在场组时 L=U=0，所有正常动作映射 0，不产生伪投影。

同时给出安全修正率、强制动作率、平均区间宽度、未交付电量；每个率记录分子/分母。安全修正率分母为所有 ev_available 时段；强制动作率也以 available 为分母；另给可控时段数。无 available 返回 null。

不再用 00:00–07:00 或配置遗留 departure_hour 筛选 cohorts 指标。按真实 sessions 的 arrival≤时段开始<departure 判定在场。离站时只计一次事件，组内分配及 EV 1.5/kWh 离站短缺成本保持原设定。

环境 plan、JointActionProjector、collector、训练验证 summary、正式 test summary、图表必须使用同一语义。旧 mg3_projection_rate 可保留为新安全修正率的显式兼容别名，并附 metrics_schema_version；旧历史字段只标 deprecated，不覆盖旧结果。

## 7. ESS 要约诊断的完整账本

每节点每步必须保存以下 kW 字段，再由 collector 乘 Δt：

    offer_uncapped = max(Dreq−E_local,0)
    offer = min(offer_uncapped, peer_load_cap)
    projection_reduction = max(offer_uncapped−offer,0)
    accepted = Σ ess_peer_load_out
    unmatched = max(offer−accepted,0)

必须满足 offer_uncapped = projection_reduction + accepted + unmatched。projection_reduction 不能混入多个卖家竞争造成的 unmatched；SOC 限制导致的请求缩放也另记，不能混为报价削减。

沿用现有持久化字段 ess_p2p_offer_uncapped_kw、ess_p2p_offer_kw、ess_p2p_offer_projection_reduction_kw、ess_p2p_unmatched_kw 及对应 kWh 聚合。新 schema 的必需字段缺失直接报数据错误；旧 schema 无法复算时为 null，不用 get(...,0) 冒充实测。

如果保留 sharing_offer_ratio，明确命名/说明为派生诊断，不是 Actor 输出，也不一定是“总放电量用于交易的比例”。记录其分母定义；分母为 0 时 null 或明确 not_applicable，禁止用于策略质量评分。

## 8. 磨机：产量覆盖率与下游服务完成率不同

实际产量必须使用交付电力后的 served_duty，不使用 DR plan 中计划产量。记录每步库存裁剪前后状态；不能靠 clip 把库存缺口擦掉。

日级：

    production_t = Σ actual_production_tph × Δt
    consumption_required_t = Σ downstream_consumption_tph × Δt
    production_coverage_ratio = production_t / consumption_required_t
    inventory_drawdown_t = inventory_start_t − inventory_end_t
    balance_residual_t = inventory_start_t + production_t − actual_consumption_t − inventory_end_t

另记录 actual_consumption_t、downstream_unserved_t（物理材料不足）、inventory_min_violation_t、inventory_max_violation_t。未裁剪原始库存低于 silo_min 是操作约束违约，不可自动等同于完全没有材料；只有材料真正不足以供给消耗时才记 downstream_unserved。两者均使严格服务日失败，但不能重复扣减消费量。

在当前无限购电、可行磨机投影路径中应无缺供且库存不越界，actual_consumption=required。仅在实际异常分支诊断不足，不借此次改动重做设备模型。

16.51/23.40≈70.56% 是产量覆盖率；依靠初始库存满足下游需求时，下游服务完成率仍可能为 100%。日末库存不恢复不是故障。终端恢复关闭时 terminal_success=null/applicable=false，不能赋 1 后用 max(terminal_success,production_ratio) 拼服务成功。

旧 mill_production_completion_rate 改为明确的 production_coverage_ratio 别名或弃用；checkpoint 不得把“每日生产≥每日消耗”作为新硬约束，这会违背已确认的独立日设定。

报告列出初末库存、净消耗和独立日重置条件；利润称为“给定每日初始资源条件下的单日运营收益”，不能直接年化或外推连续跨日收益。不新增影子库存成本或修改奖励来美化公平性。

## 9. 合作价值利用率：总体比值与日分布并列

每个可比日期 d：N_d=算法联盟收益−独立 MILP 收益总和；B_d=联盟 MILP 收益−独立 MILP 收益总和。

总体主指标：R_total=ΣN_d/ΣB_d，字段 cooperative_value_capture_ratio_total。日级比值仅在 B_d>货币容差时定义；另给 cooperative_value_capture_ratio_daily_mean、median、有效日数。不得把日均值冠名“总体”。

同一有效日期集合同时计算分子与分母：算法收益、全部所需 MILP 收益有限且求解有效、日期/成本/服务口径一致。失败日期成对排除并列日期、原因、覆盖率；不能分子用 55 天分母用 54 天。近零 B 日期不定义日比值，但 N_d/B_d 原始增益仍进入总体求和；负 N 必须保留。

若 B_d<−tol，应报告 coalition_value_ordering_error 并将该日排除出可比集合，不当作正常小分母。−tol≤B_d<0 可按求解误差归 0，保存原值。ΣB≤tol 时总体 null/status=undefined_no_positive_surplus。tol 沿用结算的货币容差并记录，不借扩大 tol 选择性删日。

输出 numerator_total、denominator_total、included_days、excluded_days、daily_ratio_defined_days 和 aggregation_method=ratio_of_sums。比值内部保存小数，展示乘 100；不裁剪负数或大于 100% 的值，异常值应解释可比性/求解质量。

旧顶层 cooperative_value_capture_ratio 在新 schema 中明确别名到 total；各层 summary、CSV、plotting、paper_artifacts 同步。用户提供的 −188.77% 和 36.84% 是不同口径，不把前者简单称作算术错误。

## 10. 单一指标定义、模型选择与历史审计

新增 evaluation/service_metrics.py 作为无环境副作用的纯计算模块：以逐时实际状态/事件和 run 配置计算指标。训练验证、正式评价、service_reliability 文件后处理、图表均调用它；禁止复制四份稍有不同的舒适率。

必需数据不存在：新 schema 报错；历史审计返回 unknown/null 并列缺字段。无适用事件与确有事件且全部合规必须区分。

当前约束选模配置若引用旧 terminal_success、mill_production_completion_rate、mg3_projection_rate，改用真实服务指标；不新增“每天必须产消平衡”约束。模式保持现有配置（profit_only 不静默改成 constrained_profit）。constrained_profit 下必需指标缺失/未知不得通过；没有合格 checkpoint 时可以保存最小违约候选，但 metadata 明示 eligible=false 和每项违约。

历史报告重算只能读取保存的 config_snapshot、hourly CSV、daily CSV、求解状态和已有收益，写入独立审计目录并记录源文件 hash。不得修改 retained releases；不得为了绘图重新跑环境/模型/MILP。若某字段需要未保存的证据，停止该字段复算并标 unknown。旧模型复现属于另一次明确标记的旧语义运行，不冒充保存结果后处理。

## 11. MILP 与规则环境的可比边界

修改 optimization/common_constraints.py 的来源/用途约束和旧互斥条件，不只改环境。

必须一致：相同日期、初始状态、实际负荷/DR 物理、价表、效率、外送限额、来源禁令、独立日设定、EV sessions、HVAC 结果时刻舒适带。所有非空联盟仅在自己的成员内共享风光，不能引用联盟外微网资源。单成员退化为本地规则。

新增风光对负荷优先约束：本地 res_load=min(R,L)；在当前无容量、统一价格模式，存在可用于联盟负荷的风光余量时，不能同时有 ESS/电网/未供电承担本可由风光满足的负荷。可用非负流量互补条件和二元变量实现；big-M 必须由该日风光、设备、负荷/充电上限推导，不使用 infinity、10000 等任意常数。

充电量仍是MILP优化决策，不能因为存在风光就强迫电池充到上限；但给定已经选择的充电量，先用负荷阶段剩余风光、再用电网，不能同时把可用风光弃掉而以电网完成本可覆盖的充电。无限共享模式下可用总量互补条件表达这一来源顺序，本地风光先满足本地已选充电量，再供应其他已选充电量。

ESS 本地优先在风光共享之后判断：ESS 向外卖电时，不得同时有本地剩余负荷由其他 ESS/电网供给或未满足；但允许先收到他网风光再卖自己的 ESS。旧约束把所有 peer_load 都禁止，必须拆开 res_peer_load 与 ess_peer_load。

删除依 grid_mode 禁止所有 P2P 出口的粗互斥，依 source/use ledger 禁止电网转售。风光供联盟负荷后剩余才可充电、外送、弃电。

不要声称此 MILP 完全复现确定性比例清算：MILP 保留自由分配、全日完美预见和已存在的 EV 组间自由分配优势，模式标 source_priority_free_allocation。将其作为相同来源优先约束下的参考上界候选，不能把收益差全部归因于 RL 学习不足。要声称数学上界，需证明规则策略的可行轨迹能嵌入该 MILP；HVAC 违约策略不在硬舒适 MILP 可行域内，收益比较必须同时展示服务状态。

MILP 的自由分配不要伪装成 Actor 可以实施的动作。optimization/oracle_replay.py 重放后必须对比计划与环境实际流量、利润、服务，记录 allocation_replay_gap；无法一致时明确 oracle_not_exactly_replayable。不要为了让重放收益相等而删除新环境优先规则。

## 12. 数值用例与验收标准

### 12.1 清算固定用例（Δt=1，容量/SOC 足够）

| 编号 | 输入摘要 | 必须得到 |
|---|---|---|
| D1 | A负荷30、B风光20；A请求放25；C无负荷 | B→A风光20；A本地ESS10；A剩余请求15取消 |
| D2 | D1增加C负荷5，风光按比例分配 | B→A风光120/7、B→C风光20/7；A本地ESS90/7；A→C ESS15/7；A总放15；不是先固定B全给A |
| D3 | A风光30/充电请求30/负荷0，B负荷30 | A→B风光30；A电网充电30；合法且不转售 |
| D4 | A/B未限额ESS要约各20、C剩余负荷10 | A/B各限额到10、各削减10、各成交5、各未成交5；不超供 |
| D5 | A负荷0/ESS请求20，B负荷20，C风光20 | C→B风光20；A实际放0；Critic交易通道0 |
| D6 | A风光40/充电请求10，B负荷20/充电请求15 | 风光先供B负荷20，再A充10，再B充10，B电网充5 |
| D7 | 无风光、无P2P对手需求、ESS正动作 | 无额外放电/扣SOC；不转外网 |
| D8 | P2P关闭；A风光20，B缺20 | 不跨网；B电网供20；A余量按本地充电/外送处理 |

D2 特意验证同时比例分配，避免将聊天中的单对单示例误写成按节点顺序撮合。另测零供需、交换 agent 顺序、SOC 边界、批量、float32/float64、非 1h 的能量换算。

### 12.2 服务和统计固定用例

- H1：terminal_mode=none；工作日06→07；实际20.593℃；舒适率不是1，越界0.407℃，strict=false。
- H2：18→19 的20.593℃按非占用温区合规；相同07:00在周末使用周末温区。
- H3：两天占用边界分别1/1、8/9，全体率9/10=90%，不是日均94.44%。
- H4：初始状态可行、固定准确预测、存在提前预热轨迹；任意允许动作保持下一步可达；单步方案反例被新层排除。
- H5：物理不可能场景返回 infeasible+正slack；求解器失败走失败路径；不返回100%服务。
- H6：修改隐藏的未来真实天气而保持当前预测快照不变，可达区间不变；修改预测快照必须使缓存失效。
- E1：L=10,U=30,a=0 → mapped=20；planned=20 时 correction=0，不能记一次投影。
- E2：L=U=10 为强制动作；无在场车L=U=0；G3下午时段必须计入 available。
- E3：planned=20、served=15 → delivery_shortfall=5，不计映射投影；离站事件仅计一次。
- S1：uncapped=30、offer=20、accepted=12 → reduction=10、unmatched=8；Δt=.5时对应5和4 kWh。
- M1：库存10.53、产16.51、耗23.40 → 末库3.64；产量覆盖约70.56%，下游供给可为100%；终端恢复N/A。
- M2：计划生产与实际交付不同，采用实际产量；库存裁剪不能抹掉越界或伪造守恒。
- R1：N=[−1,5]，B=[.01,10] → daily_mean=−49.75，total=4/10.01；均不裁剪。
- R2：一个MILP失败日成对排除；B=0日不定义日比值；总分母0返回null；序列化不含NaN/Infinity。
- X1：缺温度列或缺session证据不能静默成功；验证与正式测试同轨迹得到同指标。
- X2：新 checkpoint 同版本往返成功，旧版本加载新环境明确失败。

### 12.3 交付必须完成

1. 全量 pytest -q 通过；不能只运行定向测试或把失败项 skip 掉。
2. 两个当前插件各跑 1–3 天统一流水线；验证、best_model.pt 选择、正式测试、七联盟 MILP、Equal/Shapley、summary、保存结果绘图都完成。
3. 每 run 为15 PNG+15 PDF，正常结果位于 outputs/working_runs/<slug>/runs/；无 --no-milp、--no-validation 绕过。
4. 检查两算法同 physical_config_hash；自己的 YAML 和选定插件；新语义元数据；正式测试加载 best_model.pt。
5. 从保存 CSV 独立复算 HVAC、EV、ESS削减、磨机、合作价值比值，与所有 summary/图表一致。
6. 记录每阶段运行时间、HVAC LP失败/不可行次数；不把短冒烟收益当学习性能结论。
7. canonical MATD3 diff为空；旧 releases未改。历史复算、旧语义复现、新语义训练分开交付。

## 13. 实施者禁止的捷径

- 不能只把 HVAC 返回1改成99.79，或只补一个最终报表字段。
- 不能靠提高35 kW上限、放宽21℃下限、读未来真值来修舒适度。
- 不能把 EV affine mapping 的差异统一设0，却不补强制程度/供电不足诊断。
- 不能把 ESS uncapped−accepted 全部叫 projection_reduction。
- 不能用“末库存降低”直接判磨机未服务，也不能用 terminal_mode=none直接判生产完成。
- 不能对合作价值利用率做绝对值、截断到0–100%或删除负收益日。
- 不能仅修改 Python 环境执行，却让 target/replay/policy gradient 保持旧动作含义。
- 不能因为新模式改了，就顺手重写永久 MATD3、旧有限容量实验或保留结果。

本文的详细执行顺序、接口与测试归属见配套实施规范。任何偏离上述语义的实现都必须先明确说明差异，不能由落地模型自行用“更合理”替换已经确认的规则。
