# Reward V2 + 普通 T 同伴未来信息正式化：设计规范

日期：2026-09-29  
目标分支：`代号：噜噜`  
设计基线提交：`b1fda726a94c524dd50fcd3206112e2ab72208d9`（`备份before：修改奖励函数`）

本文只冻结设计、兼容边界与验收条件。后续实施必须以该提交为起点核对实际代码，不按旧行号机械修改。

---

## 1. 目标与范围

本次包含两个彼此区分、但在同一版本落地的改动：

1. **共享环境 Reward V2 / 物理语义重构**
   - 禁止向外部大电网售电；
   - 功率平衡继续由物理清算硬保证，不加入普通 reward；
   - 去除会替 Agent 做跨时域决策的强投影；
   - HVAC 改为“物理功率硬约束 + 舒适度软惩罚”；
   - EV 删除 deadline/reachability 最低充电强制；
   - 奖励改为统一的广义系统成本；
   - 不加入 P2P 交易量奖励。

2. **普通 T 的正式算法改进**
   - 将 forecast-context 消融实验 B 组的“同伴未来 energy balance”正式合入普通 T；
   - 保持普通 T 的预测窗口 **8 小时**；
   - 不把 24 小时窗口一并带入；
   - 不依赖 `matd3_forecast_context` 实验插件运行时实现。

这两部分必须在结果解释中分开：Reward V2 是跨算法共享环境变更；同伴未来信息是普通 T 的算法变更。后续新训练结果不能把二者混合后的收益差全部归因于 Transformer 改进。

---

## 2. 最新消融证据与普通 T 的正式选择

最新提交保存的单种子四组消融结果中：

- `B-A`：H=8 下增加同伴未来 balance，日配对团队收益均值约 **+19.04**，55 天胜率约 **69.1%**；
- `C-A`：仅将本地窗口 8h 延长到 24h，均值约 **-20.88**；
- `D-C`：H=24 下增加同伴未来 balance，均值约 **+16.39**；
- `D-B`：已有同伴未来时由 8h 延长到 24h，均值约 **-23.53**。

这些结果只有 seed42，属于探索性证据，不宣称多种子统计结论。但它支持当前工程决策：

```text
普通 T 正式版本 = H=8 + peer future energy_balance
```

不采用 H=24。实验插件及四组历史配置保留为研究证据，不回写、不删除、不改造成正式普通 T 的运行依赖。

---

## 3. Reward V2 总体原则

三个微网仍使用相同团队奖励：

```text
r_mg1,t = r_mg2,t = r_mg3,t = R_team,t
```

定义统一广义系统成本：

```text
C_generalized,t =
    C_grid,t
  + C_battery,t
  + C_curtail,t
  + C_carbon,t
  + C_peak,t
  + C_mill,t
  + C_comfort,t
  + C_ev,t
  + C_unmet,t
```

训练奖励：

```text
R_team,t = - C_generalized,t / reward_scale
```

其中 `reward_scale` 只负责数值尺度，不代表优化偏好。各项偏好通过具有明确物理/经济单位的成本系数表达。

### 明确不加入的奖励

以下内容不得作为 Reward V2 正式项：

- SOC 距离某个目标值的 shaping；
- 功率平衡奖励；
- P2P 成交量奖励；
- DR 动作幅度奖励；
- “鼓励卖电”奖励；
- 未成交 ESS 要约惩罚；
- 为了让曲线更好看而增加的任意常数 bonus。

---

## 4. 功率平衡：硬物理约束，不作为学习目标

功率平衡继续由 renewable-first 清算与外部电网兜底实现。

禁止外送后，每个微网每步应满足：

```text
RES
+ ESS_discharge
+ P2P_import
+ Grid_import
=
Actual_load
+ ESS_charge
+ P2P_export
+ Curtailment
```

并满足系统 P2P 守恒：

```text
sum(P2P_sell) = sum(P2P_buy)
```

正式验收要求：

- `power_balance_residual_kw` 仅允许浮点容差内误差；
- `p2p_balance` 仅允许浮点容差内误差；
- 非零残差应视为环境/账本错误，而不是“扣一点 reward 继续训练”。

`unmet_load` 保留为极端异常/有限供电场景的安全成本与诊断；在当前无限 Grid import 设定下应正常为 0。

---

## 5. 禁止外部大电网售电

正式新环境中：

```text
Grid_export = 0
```

实现要求：

- `configs/cooperative.yaml` 中每个 MG 的 `export_limit_kw` 设为 0；
- `ExternalGrid`、环境结算、MILP、参考策略与测试全部遵循同一约束；
- 既有 `grid_export_kw`、`grid_export_revenue` 输出字段可为兼容保留，但新语义下必须始终为 0；
- 不删除旧结果中的历史外送数据，不重新解释历史运行。

新能源在满足内部用途后，只能：

```text
本地负荷
→ 联盟其他负荷
→ 本地/联盟 ESS 允许的充电
→ Curtailment
```

现有 renewable-first 来源账本继续作为唯一物理事实来源，禁止用净功率二次推导出另一套“新能源去向”。

---

## 6. Reward V2 各分量

### 6.1 电网购电成本

```text
C_grid,t = Σ_m buy_price_m,t * E_grid_import_m,t
```

单位：货币/step。

不再存在售电收入抵扣。

### 6.2 ESS 退化成本

继续使用实际充放电吞吐：

```text
C_battery,t =
Σ_m degradation_price_m
    * (E_charge_m,t + E_discharge_m,t)
```

只按 **实际执行** 吞吐计费，不按 raw action、未成交要约或 SOC 偏离计费。

### 6.3 弃电成本

```text
C_curtail,t = curtailment_price * E_curtailment,t
```

弃电量必须来自禁止外送后的真实最终 surplus。

这是新能源消纳的直接目标项。不得同时再增加“新能源使用 bonus”，避免同一目标正负双重计权。

### 6.4 碳成本

使用已有 `grid_carbon_factor_kg_per_kwh`：

```text
CO2_t = carbon_factor_t * Σ_m E_grid_import_m,t
C_carbon,t = carbon_price_per_kg * CO2_t
```

规则：

- 只在电网进口发生时记碳；
- Grid→ESS 充电的碳在进口时已记账，未来 ESS 放电不得再次重复计算；
- P2P、风光直接供能不额外虚构碳排。

### 6.5 峰值成本

正式优化对象选用 **联盟实际总负荷峰值**，而不是 Grid import 峰值：

```text
P_load_team,t = Σ_m Actual_load_m,t
peak_before_t = max(P_load_team,0 ... P_load_team,t-1)
delta_peak_t = max(0, P_load_team,t - peak_before_t)

C_peak,t = peak_price_per_kw * delta_peak_t
```

一天累计后等价于对当日最大实际负荷收费。

原因：

- 该项直接约束 DR 的削峰行为；
- ESS 不应通过“遮住电网进口峰值”替代真正的负荷移峰；
- Grid import peak 仍作为重要评价指标，但先不进入 Reward V2，避免与购电成本及 ESS 行为重复计权。

日初 `peak_before=0`，不能跨自然日继承。

### 6.6 MG1 下游服务损失

```text
C_mill,t =
mill_unserved_price_per_t
* downstream_unserved_t
```

只惩罚真实未满足的下游物料消费，不惩罚“当小时产量低于消费量”。

允许利用 silo 库存：

```text
Production < Consumption
```

但只要库存补足且：

```text
actual_consumption = required_consumption
```

则该项为 0。

**本次继续保留 MG1 当前时刻的 silo 瞬时可行域。** 当前 `feasible_bounds` 是单步库存物理/操作边界，不是昂贵的未来 LP，不在本轮为了“让 penalty 经常非零”而删除。若未来要把 `silo_min` 从硬操作边界改为软约束，应单独立项，不得偷偷混入 Reward V2。

### 6.7 MG2 HVAC 舒适度损失：采用方案 B

删除 HVAC 对未来舒适可达性的强投影，也删除 one-step comfort clipping。

Actor 只受设备真实功率边界约束：

```text
-cooling_max_kw <= signed_HVAC_power <= heating_max_kw
```

温度由 2R2C 模型真实演化。

每个结果边界时刻计算：

```text
violation_c =
    max(0, comfort_low - T_air)
  + max(0, T_air - comfort_high)

C_comfort,t =
comfort_price_per_degree_hour
* violation_c
* delta_t
```

要求：

- 使用 action t 执行后的 `t+1` 实际空气温度；
- 使用 `t+1` 时刻适用的 occupied/unoccupied comfort band；
- 舒适区内任何温度均为 0 penalty；
- 不写成 `(T-23)^2`；
- 不对“离舒适中点远但仍合规”的温度处罚。

`components/hvac_viability.py` 不得继续位于正式训练调用链。可暂时保留文件作为历史/测试迁移辅助，但普通训练不得再调用 HiGHS viability LP。

### 6.8 MG3 EV 离站短缺

删除 cohort EV 的未来 reachability 最低充电强制：

现有类似：

```text
raw_low =
max(0,
    required_energy
  - max_charge * remaining_future_steps)
```

的强制 lower bound 不再用于正式动作可行域。

正式约束：

- 未到达不能充；
- 已离站不能充；
- 不能超过站端/组额定功率；
- 不能超过剩余需求；
- **当前最低充电功率为 0**。

Actor 继续看到：

- remaining energy；
- hours to departure；
- deadline pressure；
- availability；
- 当前物理最大充电能力。

真实离站事件结算：

```text
C_ev,t =
ev_shortfall_price_per_kwh
* departure_shortfall_kwh
```

组间分配仍可使用 earliest-deadline/urgency 顺序分配 Actor 已选择的总功率；这种分配不能增加 Actor 没有选择的总充电量。

### 6.9 未供电安全成本

保留：

```text
C_unmet,t =
unmet_load_cost_per_kwh
* unmet_load_kwh
```

在当前无限外部购电环境中正常应为 0。它属于异常安全成本，不是主要 reward shaping 项。

---

## 7. 强投影移除边界

本轮删除的是“为了保证未来还有完成可能而替 Agent 决定当前动作”的 projection，而不是所有瞬时物理约束。

### 保留

- ESS SOC 上下限；
- ESS 最大充放电功率；
- ESS 禁止同时充放；
- 当前 P2P/来源守恒；
- MG1 当前 silo 瞬时可行域；
- HVAC 设备制热/制冷额定功率；
- EV 当前连接状态、最大功率、剩余需求上限。

### 删除/退出正式调用链

- HVAC forecast viability LP；
- HVAC one-step comfort clipping；
- EV deadline/reachability minimum charging lower bound；
- 任何 ESS/DR terminal future reachability 强制动作（当前主配置 terminal mode 为 none，不得恢复）。

观察量中的 deadline / pressure 信息可以保留，让 Agent 学习未来风险；环境不能把这些信息重新转成强制当前动作。

---

## 8. Reward 权重与尺度：统一等效成本，不做动态归一化

各项系数必须具有可解释单位：

- `curtailment_price_per_kwh`
- `carbon_price_per_kg`
- `peak_price_per_kw`
- `mill_unserved_price_per_t`
- `comfort_price_per_degree_hour`
- `ev_shortfall_price_per_kwh`

原则：

1. 所有系数固定写入运行 config snapshot；
2. 不根据当前策略在线更新；
3. 不按每个 episode 自适应缩放；
4. 不使用 test set 标定；
5. 所有算法使用同一组环境系数。

### 8.1 标定流程

正式长训练前必须先完成新环境下的 raw component 标定：

记录至少：

- grid import cost；
- battery throughput/cost；
- curtailment kWh；
- carbon kg；
- daily load peak kW；
- mill downstream unserved t；
- comfort violation °C·h；
- EV departure shortfall kWh；
- unmet kWh。

使用 train/validation 场景上的固定 rule/idle/短流程策略生成数量级统计；不得用正式 test 55 天调参数。

标定目标不是让每项“数值完全一样”，而是明确边际交换关系，例如：

```text
减少 1 kWh 弃电值多少等效货币；
减少 1 kg CO2 值多少；
增加 1 kW 日峰值值多少；
1 °C·h 舒适违约值多少；
1 kWh EV 离站短缺值多少。
```

服务损失系数应显著高于普通小幅经济波动，但不能设置成无限大或大到等价于重新恢复硬投影。

`reward_scale` 在系数锁定后再固定，使典型每步 reward 保持适合现有 MATD3 数值范围；它不改变各目标相对权重。

在标定系数未锁定前，只允许单元测试、诊断和短 smoke，不启动正式 5000 回合长训。

---

## 9. 奖励记账与旧指标兼容

不要继续把所有新目标塞进 `physical_profit`，否则“经济收益”和“训练目标”将失去可解释性。

新增明确分层：

```text
economic/accounting metrics
generalized reward metrics
service metrics
```

至少保存：

- `team_profit`：保留历史兼容字段，不作为 Reward V2 唯一来源；
- `generalized_system_cost`；
- `team_reward`；
- 每一个 reward component 的逐步/逐日累计；
- `reward_semantics_version`。

推荐版本：

```text
reward_semantics_version = generalized_system_cost_v2
```

旧 run 不补写、不重算、不覆盖。

训练曲线、summary、CSV 必须能区分：

```text
economic profit
vs
generalized return
```

禁止将新 `team_reward` 标成“profit”。

---

## 10. Checkpoint 选择必须与 Reward V2 对齐

当前普通 T 的 `constrained_profit` 使用：

```text
objective = team_profit_mean
strict service constraints = 100%
```

这与“服务质量改为软 penalty”冲突：如果 checkpoint 仍要求 HVAC/MG1/EV 100% 合规，就等价于在模型选择阶段重新施加硬约束。

Reward V2 后应：

- checkpoint 主目标改为 `team_reward_mean`（越大越好）或完全等价的负 generalized cost；
- 保留真正的物理可行性约束；
- 不再要求 `hvac_comfort_rate=1.0`、`mill service=1.0`、`ev success=1.0` 作为 hard eligibility；
- `strict_feasible_day_rate` 若包含这些服务软目标，不再用于硬筛选；
- 可使用 `physical_feasible_day_rate`、功率/P2P/SOC/设备功率违反率、unmet 等作为真正硬约束。

共享 selector 若需增加通用 `objective_only` 或物理约束模式，必须向后兼容旧配置，不破坏历史算法的解析。

---

## 11. MILP / 参考对照的目标一致性

Reward V2 改变后，不能继续把旧经济 MILP 的 objective 当成新 reward 的数学上界而不说明。

正式对照至少做到：

1. 外部 `grid_export` 上界为 0；
2. 加入相同 grid import、ESS degradation、curtailment、carbon 成本；
3. 加入联盟实际负荷 peak 变量：
   ```text
   P_peak >= sum(actual_load_m,t) for every t
   objective += peak_price_per_kw * P_peak
   ```
   该日目标与环境 incremental peak 累计等价；
4. EV shortfall 使用同一 penalty 系数；
5. HVAC 若作为软舒适目标，则 MILP 不得仍把 comfort 全部硬固定后再称“同目标上界”；应使用上下界 slack 并按 °C·h 同系数计费，或在报告中明确该 MILP 是不同服务域参考；
6. MG1 在本轮保留当前 silo 瞬时硬边界，因此正常 downstream unserved 为 0；MILP保持同一语义。

经济 profit 与 generalized objective 均保存，不能混成一个字段。

MILP 仍有完美预见、自由分配等优势，继续标为 full-information reference，不把差距全解释成 RL 学习不足。

---

## 12. 普通 T：正式加入同伴未来预测信息

### 12.1 保持 8 小时窗口

正式普通 T 继续：

```yaml
forecast_horizon: 8
```

不采用 24h。

### 12.2 每个未来 token 从 6 维变为 8 维

保持现有六维：

```text
0 local_base_net_load
1 buy_price
2 temperature
3 valid
4 future_hour_sin
5 future_hour_cos
```

追加：

```text
6 peer_0_energy_balance
7 peer_1_energy_balance
```

不增加独立 peer future token，总 token 数仍为：

```text
4 local semantic
+ 2 current peer
+ 8 forecast
= 14 tokens
```

只将 `forecast_encoder` 输入维度从 6 改为 8。

### 12.3 同伴字段语义严格沿用 B 组

```text
peer_order = [mg for mg in agent_ids if mg != active_agent]
```

读取：

```text
<peer>_energy_balance_t+k
```

现有语义：

```text
peer balance > 0  => peer renewable surplus
peer balance < 0  => peer deficit
```

本地 `local_base_net_load` 则是：

```text
local net load > 0 => local deficit
```

两者符号语义相反是已知设计，正式版本保持与 B 组完全一致，不在此次“顺便统一符号”，避免引入第三个未验证因素。

不额外加入 peer deficit/surplus 两个冗余通道；它们可由 balance 推导。

### 12.4 Mask

同一未来时刻的本地和 peer future 信息共用 `forecast_valid_t+k`。

无效时刻必须整枚 forecast token 屏蔽；不能让本地字段被 mask 而 peer 未来字段仍参与 attention。

### 12.5 代码组织

正式实现写回：

```text
algorithms/matd3_hierarchical_transformer/
```

但不得让正式普通 T 运行时 import：

```text
algorithms/matd3_forecast_context/
```

可以参考/回移实验 B 的已验证实现逻辑，但正式包必须自包含。

`matd3_forecast_context` 保持历史消融插件，不修改其四组语义，不删除其结果报告。

### 12.6 配置与 checkpoint

普通 T 正式 YAML 增加显式字段：

```yaml
network:
  forecast_horizon: 8
  include_peer_forecast: true
  peer_forecast_features: [energy_balance]
```

检查点必须保存并校验：

- peer forecast 开关；
- peer order；
- feature order；
- power normalizer；
- mask 语义；
- actor context version。

推荐：

```text
actor_forecast_context_version = peer_balance_h8_v2
```

旧普通 T checkpoint 的 forecast encoder 为 `Linear(6,...)`，不得静默部分加载到新 `Linear(8,...)`。旧 checkpoint 需清晰拒绝或只在专门兼容/迁移工具中显式处理。

---

## 13. Reward V2 与普通 T 改进的实验解释边界

后续正式训练时：

- MATD3 与普通 T 必须使用完全相同的 Reward V2、禁止外送、projection 语义和数据；
- 普通 T 比 MATD3 多出的核心算法差异包括 Transformer 架构及本次 peer future context；
- 不能拿旧 Reward V1 下的 MATD3 run 与新 Reward V2 下的普通 T 直接声称算法优越；
- 旧四组消融报告保留为“为什么选择 H8+peer”的工程证据；
- 新 Reward V2 后若要再次确认 peer future 的纯效果，需另开新消融，不把本轮联合改动当成因果证明。

---

## 14. 需要同步检查的非目标副作用

实施时必须逐项审查以下消费者，不能只改环境主函数后结束：

- `core/cost_accounting.py`
- `envs/multi_microgrid_env.py`
- `core/grid_settlement.py`
- `components/commercial_hvac.py`
- `components/ev_charging.py`
- `core/joint_action_projection.py`
- `core/observation.py`
- `core/global_state.py`
- `utils/metrics_collector.py`
- `evaluation/service_metrics.py`
- `evaluation/evaluate_matd3.py`
- `experiments/checkpoint_selector.py`
- `experiments/training_pipeline.py`
- `optimization/common_constraints.py`
- `optimization/cooperative_dispatch.py`
- `optimization/independent_dispatch.py`
- `optimization/oracle_replay.py`
- 普通 T actor/algorithm/config/checkpoint；
- MATD3 与普通 T smoke configs；
- 相关 plot / paper artifact 字段；
- 旧结果审计兼容逻辑。

### 特别防止以下非预期情况

1. 禁止外送后仍因旧 net settlement 路径出现非零 export；
2. curtailment 既进入旧 profit 又在新 reward 重复计两次；
3. EV departure penalty 在 `physical_profit` 和 generalized reward 中重复扣；
4. HVAC comfort 既被 projection 强制又被 penalty 处罚；
5. checkpoint 仍用 100% comfort/EV/mill 约束，变相恢复硬要求；
6. MILP 仍允许 export 或仍以不同 objective 被称为新 reward 上界；
7. 旧 `mg2_projection_rate` / `mg3_projection_rate` 指标因语义改变而被误解；
8. 删除 LP 后 observation/schema 仍宣称是 forecast viability bounds；
9. peer future 已进入普通 T，但 checkpoint metadata 仍标旧 6 维 forecast；
10. Reward V1 与 Reward V2 的 return 曲线被放在同一纵轴上直接比较数值大小。

---

## 15. 版本与结果隔离

新增运行元数据至少保存：

```text
dispatch_semantics = renewable_load_first_v2_no_grid_export
reward_semantics_version = generalized_system_cost_v2
dr_projection_semantics = instantaneous_physical_only_v2
hvac_control_semantics = soft_comfort_penalty_v2
ev_control_semantics = no_deadline_min_projection_v2
actor_forecast_context_version =
    peer_balance_h8_v2   # 普通 T
    none / baseline      # MATD3
```

旧运行、旧 checkpoint、旧 report 均只读。

不允许把旧结果重新标记成新 Reward V2 结果。

---

## 16. 验收测试

### 16.1 物理与 Reward

固定 fixture 必须验证：

- 每节点/系统功率守恒；
- P2P 买卖守恒；
- `grid_export_kw == 0`；
- surplus 无内部去向时全部进入 curtailment；
- ESS degradation 只按实际吞吐；
- carbon 只按 Grid import；
- incremental load peak 一天累计等于 `peak_price * max(team_actual_load)`；
- comfort 区间内 penalty=0，越界按实际 °C·h；
- EV 可以在可用时选择 0 充电，不再被 deadline lower bound 抬高；
- 离站不足才产生 EV shortfall penalty；
- MG1 使用库存满足消费时，即使生产<消费，`downstream_unserved=0`；
- reward components 之和严格等于 generalized cost；
- team reward = `- generalized_cost / reward_scale`。

### 16.2 HVAC

必须有回归测试证明：

- 正式训练路径不调用 `solve_hvac_viability`；
- Actor 请求在设备额定范围内不因 comfort 被改写；
- 允许产生真实 comfort violation，并被 metrics/reward 正确记录；
- 2R2C 状态更新仍有限、守恒逻辑不变。

### 16.3 普通 T peer future

必须验证：

- forecast encoder 输入维度为 8；
- 修改有效的 peer future balance 会影响动作或中间 token；
- 修改被 mask 的未来 peer 字段不影响动作；
- H 仍为 8；
- 当前 peer summary token 保留；
- 不依赖 `matd3_forecast_context` import；
- 旧 6 维 checkpoint 明确拒绝；
- B 组 actor 与正式普通 T 在同配置/同权重映射下的 forecast feature 构造一致。

### 16.4 流水线

完成实现后必须：

1. `pytest -q`；
2. MATD3 Reward V2 smoke；
3. 普通 T Reward V2 + peer future smoke；
4. 确认 best checkpoint 使用 generalized reward 目标与物理约束；
5. 正式 test、MILP、Equal/Shapley、summary、plots 全链路完成；
6. 确认旧 forecast-context 实验结果未被覆盖；
7. 在实施报告中记录起始/结束 commit、配置快照、测试结果与任何偏离。

---

## 17. 本轮明确非目标

- 不加入 P2P 交易量 reward；
- 不增加 24h 普通 T 预测窗口；
- 不新增独立 peer future token；
- 不修改 canonical `algorithms/matd3/` 网络与更新逻辑；
- 不恢复 ESS terminal SOC；
- 不把 MG1 silo_min 改成软约束；
- 不新增 Shapley reward / difference reward；
- 不新增独立 Critic Transformer 改造；
- 不用 test set 调 Reward V2 权重；
- 不覆盖、删除或改写既有训练结果。
