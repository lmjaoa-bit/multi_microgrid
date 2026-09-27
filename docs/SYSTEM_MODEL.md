# 多微网协同调度系统建模说明书

> 当前语义：single_dispatch_dr_v3 / service_metrics_v2  
> 适用配置：configs/cooperative.yaml，分支 代号：噜噜。  
> 调度模式：dispatch.priority_mode = renewable_load_first。  
> HVAC 安全层：forecast_viability。  
> MILP 对照：source_priority_free_allocation，全日完美预见。  
>
> 本文只描述当前新训练语义。历史 checkpoint、历史 run 和旧 commit 仍按各自保存时的配置与语义解释；v1/v2 checkpoint 不重新标记为 v3，也不能直接加载到 v3 环境。

## 1. 当前系统

系统包含三个异构微网：

| 微网 | 可再生能源 | ESS | 柔性设备 |
|---|---|---|---|
| MG1 工业 | 520 kW 光伏 | 210 kWh，65/65 kW | 28.5 kW 工业磨机 + 料仓 |
| MG2 商业 | 900 kW 风电 | 290 kWh，80/80 kW | 35 kW 制热 / 35 kW 制冷 HVAC |
| MG3 住宅 | 100 kW 光伏 | 95 kWh，34/34 kW | 780 kWh、93.6 kW EV 车队 |

时间步长为 1 h，每个 episode 是独立自然日，共 24 步。三个 ESS 的当前初始 SOC 均为 0.10，允许范围为 [0.10, 0.90]。当前共享配置不要求日末 SOC 回到初值；每个自然日重新从配置的日初资源状态开始。

当前外部电网进口不设功率上限，单个微网外部售电上限为 100 kW。P2P 为无损全连接能源池，pair_capacity_kw = null，因此当前主场景没有旧版 75 kW 双边瓶颈。有限容量和旧 local-first 行为只属于 legacy 配置。

基础负荷不含柔性设备功率：

L(i,t) = L_base(i,t) + L_flex(i,t).

## 2. ESS 请求与实际执行

ESS 的 SOC 更新只使用实际执行功率：

SOC(t+1) = SOC(t)
          + eta_ch * P_ch_actual * dt / E
          - P_dis_actual * dt / (eta_dis * E).

Actor 的 ESS 原始动作 u 属于 [-1,1]。正值请求放电，负值请求充电：

D_req = max(u,0) * feasible_discharge_max
C_req = max(-u,0) * feasible_charge_max.

请求不等于实际功率。未成交的 ESS 出售请求不会扣 SOC，也不会自动转售到外部电网。

## 3. v3 风光优先清算

当前主环境、Critic 投影、target action 和 Actor loss 共用 core/renewable_first_dispatch.py 的同一纯 Torch 清算内核。

### 3.1 清算顺序

每一步严格按下列来源/用途顺序：

1. 本地风光 -> 本地总负荷；
2. 剩余风光 -> 其他微网剩余负荷；
3. 本地 ESS 请求 -> 本地剩余负荷；
4. 剩余 ESS 请求 -> 其他微网剩余负荷；
5. 负荷阶段后的本地剩余风光 -> 本地 ESS 充电请求；
6. 再剩余风光 -> 其他微网 ESS 充电请求；
7. 外部电网 -> 剩余负荷，再 -> 剩余 ESS 充电请求；
8. 仍剩余的本节点风光 -> 外部电网，超过 100 kW 外送能力的部分弃电；
9. 仅用实际交付流量提交 ESS/DR 状态并结算。

所以当前允许向大电网售电，但外部售电不能抢在本地/联盟负荷之前。它只处理完成联盟风光消纳优先后的本节点剩余新能源。

### 3.2 当前无容量 P2P 比例清算

对任一跨网阶段，节点供给 s_i >= 0、需求 d_i >= 0，且同一节点不会同时作为该阶段供给方和需求方。定义：

S = sum(s_i), D = sum(d_i).

当前无容量模式采用：

F_ij = s_i * d_j / max(S,D), i != j
F_ii = 0.

总成交量为 min(S,D)。多个卖方/买方同时按比例分配，不依赖 agent 遍历顺序。

来源账本始终区分：

- res_peer_load_kw
- ess_peer_load_kw
- res_peer_charge_kw

汇总 pair flow 仅用于报告，不能代替来源守恒。

### 3.3 ESS 对外供电

风光跨网供负荷之后，ESS 先供应本地剩余负荷。剩余放电请求才可能参与 P2P。

未限额要约：

O_raw(i) = D_req(i) - E_local(i).

要约上限只使用其他微网在风光和本地 ESS 后的剩余负荷，不包含 ESS 充电需求：

O(i) = min(O_raw(i), sum_j!=i D2(j)).

多个 ESS 卖方统一比例清算。实际 ESS 放电：

P_dis_actual(i) = E_local(i) + sum_j ESS_peer_load(i,j).

诊断量满足：

uncapped_offer
= projection_reduction + accepted + unmatched.

### 3.4 合法的同时卖电和买电

v3 明确允许：

- A 收到 B 的风光满足本地负荷，同时 A 自己的 ESS 向 C 放电；
- A 的风光先跨网满足 B 的负荷，随后 A 的 ESS 用外部电网充电。

这不属于电网转售。禁止 grid resale 是通过来源账本没有 P2P/外送路径保证，而不是用“grid import > 0 就禁止 P2P sell”的粗互斥。

ESS -> ESS 和 ESS -> 外部电网仍禁止。

## 4. 外部电网、弃电与守恒

当前进口无限，因此正常 v3 主场景不会因为进口功率上限造成普通负荷缺供。完成风光负荷优先和已选择的 ESS 充电后，本节点仍剩余的新能源可在 100 kW 外送上限内出售，超过部分才记弃电。

节点守恒：

R + ESS_dis_actual + grid_import + p2p_import
= load + ESS_ch_actual + grid_export + p2p_export + curtailment.

其中：

ESS_ch_actual
= local_RES_charge + incoming_RES_charge + grid_charge.

P2P 使用实际成交双方外部售/购电价的中间价。内部 P2P 付款预算平衡，只改变个体账面收益分配，不增加团队奖励。

## 5. 三类柔性设备

### 5.1 MG1 工业磨机

额定功率 28.5 kW，额定产量 1.186 t/h，下游需求 0.975 t/h。料仓范围 3.51–15.80 t，日初 10.53 t。

设备每步分别记录：

- planned duty / served duty；
- actual production；
- downstream required / actual consumption / unserved；
- clip 前后的料仓状态；
- inventory min/max violation。

供电不足时，实际产量使用 served duty，不用计划 duty。

产量覆盖率：

ProductionCoverage = sum(actual production) / sum(downstream required).

下游服务完成率：

DownstreamService = sum(actual consumption) / sum(downstream required).

允许利用日初库存供给下游，因此即使当日产量覆盖率约 70%，下游服务仍可为 100%。当前 terminal_mode = none，不要求日末料仓恢复到日初值。

### 5.2 MG2 HVAC

HVAC 保留原 2R2C 热模型，制热/制冷最大电功率各 35 kW，COP 分别为 3.3/3.2。工作日 07:00–18:59 为占用时段，舒适区 [21,25] ℃；其余时段为 [16,30] ℃。

动作 t 作用于 [t,t+1)：

- 外部/内部热增益按 t；
- 结果温度 T_air(t+1) 必须按 t+1 时刻的占用状态和温区判断。

例如工作日 06->07 的结果按 07:00 占用温区判断；18->19 的结果按 19:00 非占用温区判断。24:00 边界按非占用处理。

当前启用 forecast_viability。每一步使用当时冻结的预测快照，对当天剩余时段求 2R2C 线性可达性 LP，得到首步允许功率区间。该安全层支持提前预热/预冷，不读取隐藏未来真实天气。

物理不可行返回 degraded_infeasible 和正 slack；求解器失败直接报错，不能伪装成 100% 舒适。当前 HVAC terminal_mode = none，不要求日末温度恢复。

### 5.3 MG3 EV

MG3 使用多个按日期/seed 可复现的 EV cohort。不同 cohort 有各自到达、离站、需求和充电上限，并非旧版“全部车辆仅 00–07”的单一聚合模型。

cohort 动作采用仿射映射：

P_mapped = L + ((a+1)/2) * (U-L).

这个 mapping 不是 safety projection。L=U 表示动作被当前物理条件强制；无在场车通常为 L=U=0。

诊断明确区分：

- mapping / forced mapping；
- safety correction；
- planned charge；
- served charge；
- delivery shortfall；
- departure event shortfall。

离站 shortfall 只在事件发生时计一次，并按配置的 1.5/kWh 计入经济成本。没有日末 EV 恢复要求。

## 6. service_metrics_v2

新运行使用唯一纯计算模块 evaluation/service_metrics.py。训练验证、正式测试和保存结果后处理共享同一口径。

主要指标：

- strict_feasible_day_rate；
- unmet energy / hour rate；
- HVAC 占用舒适率、全部边界舒适率、degree-hours、最大温差；
- mill production coverage 与 downstream service completion；
- EV departure success / shortfall；
- cooperative value capture ratio。

新 schema 缺必需字段时直接报错；历史旧 run 缺证据时只返回 unknown/null 和 missing_fields，不能用今天的默认配置补历史事实。

终端恢复关闭时，HVAC/磨机 terminal success 为 N/A/null，不能伪造为 100%。

合作价值总体口径：

CVCR_total
= sum(algorithm_profit - independent_profit)
  / sum(coalition_profit - independent_profit).

它不是每日比例的简单平均，不做绝对值或 0–100% 截断。MILP 失败日成对排除；总分母为 0 时返回 null。

## 7. 多智能体学习语义

三个微网各有一个 Actor。当前两个实验算法：

- algorithms/matd3_dispatch：MLP Actor；
- algorithms/matd3_hierarchical_transformer：层级时空 Transformer Actor。

永久参考 algorithms/matd3 不参与 v3 修改。

训练采用 CTDE。每个微网有自己的集中式双 Q Critic，三个 Critic 使用相同团队奖励。

每个 Actor 输出两个 raw 通道：

[a_ESS, a_DR] in [-1,1]^2.

Critic 使用三个执行语义通道：

[
 (ESS_local_actual - actual_charge) / scale,
 ESS_peer_actual_sell / scale,
 effective_DR_normalized
].

第二通道已经从 v2 的 ESS offer 改成 v3 的实际 ESS P2P 售电量。

Replay 仍保存 raw action；critic regression、target action 和 actor loss 都重新通过同一个 JointActionProjector 和共享清算内核。

未执行请求只留在审计日志，不进入下一步策略历史。v3 已移除 previous offer、unmatched、system sell offer、clearing ratio 等请求历史，保留 previous P2P、grid net、ESS accepted 等实际执行量。

## 8. 奖励与独立日

团队奖励：

r_t = sum_i physical_profit(i,t) / 100.

包含外部电网购售电、ESS 退化、未供电、弃电和 EV 离站 shortfall 成本。P2P 内部付款在联盟求和时抵消。

当前配置：

- terminal_soc.mode = none；
- MG1 terminal_mode = none；
- MG2 terminal_mode = none。

因此不存在“为了午夜恢复而提前硬投影”的现行规则，也不存在 ESS/磨机/HVAC 日末恢复惩罚。每个测试日表示给定相同日初资源条件下的独立日运营收益，不能直接解释为连续多日库存轨迹或直接年化收益。

EV 离站 shortfall 是服务事件成本，不属于日末恢复约束。

## 9. MILP 对照边界

全信息 MILP 与 v3 共享：

- 相同日期与设备参数；
- 风光先负荷、选定充电量再风光优先的来源规则；
- 禁止 grid resale、ESS->ESS、ESS->外部电网；
- 相同日初资源和无日末恢复主配置；
- HVAC t+1 舒适约束和 EV session 证据。

但 MILP 保留：

1. 全日完美预见；
2. 联盟内部来源优先条件下的自由分配，而不是环境的确定性比例分配。

因此其 allocation metadata 为 source_priority_free_allocation。它是相同来源优先边界下的参考上界候选，不能把全部收益差简单归因于 RL 学习不足。

compare_oracle_market_replay 会逐来源比较 MILP 与环境实际流量和利润；无法精确重放时报告 oracle_not_exactly_replayable 和 allocation replay gap。

## 10. 历史审计

历史结果只允许只读审计：

python scripts/audit_saved_service_metrics.py --run <原run目录> --output <独立审计目录>

审计工具拒绝：

- 输出覆盖原 run；
- 输出位于原 run 内；
- 覆盖已有审计目录；
- 写入 outputs/releases。

它记录源文件 SHA-256 前后值，只依据保存的 config snapshot、daily/hourly CSV 与 summary 复算。缺失历史证据保持 null，不重新运行环境、策略或 MILP。

## 11. 建模边界

- 本系统不是 AC 潮流、OPF 或电压/无功模型。
- 当前 P2P 无损、无容量瓶颈是研究场景假设，不代表实际线路无限容量。
- 风光模型、ESS 退化、2R2C、料仓和 EV cohort 均是调度研究所需的简化物理模型。
- 外部售电仍允许，但只能处理完成风光优先负荷/充电后的本节点剩余新能源。
- 新旧语义不可混用；旧 checkpoint 在新环境会因 action semantics / state schema / observation schema 不匹配而拒绝加载。
- 正式研究结论必须同时报告经济性与服务/约束表现，不能用利润掩盖 HVAC、EV 或磨机服务失败。

实现入口：configs/cooperative.yaml、core/renewable_first_dispatch.py、core/joint_action_projection.py、envs/multi_microgrid_env.py、components/hvac_viability.py、evaluation/service_metrics.py、optimization/common_constraints.py、optimization/oracle_replay.py。
