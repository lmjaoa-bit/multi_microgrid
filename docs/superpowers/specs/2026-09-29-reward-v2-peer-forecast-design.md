# Reward V2 + 普通 T 同伴未来信息正式化：设计规范

日期：2026-09-29  
目标分支：`代号：噜噜`  
设计基线提交：`b1fda726a94c524dd50fcd3206112e2ab72208d9`（`备份before：修改奖励函数`）

本文为设计评审修订稿，只定义设计、兼容边界与验收条件，不表示实现或实验效果已经验证。2026-09-29 评审实际核对代码提交为 `9bdb9739ec686340d2a1a4ac100ba2f79e3c4a03`；上述设计基线保留为历史来源。后续实施应记录实际起点并核对两者差异，不回退覆盖后续工作，不按旧行号机械修改。

**评审结论：方向可行，但原稿不足以直接实施。** 必须补齐峰值状态、折扣目标、训练/执行投影一致性、碳信息、结算口径及日末边界。本轮修订保留原有核心选择；第 18–20 节给出问题依据、详细实施顺序和新增验收条件。权重数值仍需按第 8 节标定，不凭设计稿保证效果提升。

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

一天**未折扣**累计后等价于对当日最大实际负荷收费；折扣回报不具有这个等价性，按第 18.2 节统一处理。

原因：

- 该项直接约束 DR 的削峰行为；
- ESS 不应通过“遮住电网进口峰值”替代真正的负荷移峰；
- Grid import peak 仍作为重要评价指标，本轮先不进入 Reward V2，以保持“终端用电负荷削峰”的研究目标。购电电量成本与购电峰值成本并非数学上的重复项；以后若研究变压器需量或电网容量，应另行引入进口峰值目标。

日初 `peak_before=0`，不能跨自然日继承。

`Actual_load` 明确定义为不可控基础负荷 + MG1 实际磨机电功率 / MG2 实际 HVAC 电功率绝对值 / MG3 实际充电站电功率；不含 ESS 充电、P2P 交易量，不扣除新能源或 ESS 放电。必须将动作前 `peak_before` 加入所有 Actor 的公共观测和 Critic 全局状态，动作后更新并进入 next observation；详见第 18.1 节。

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

仅设置 `forecast_viability: false` 不满足要求：当前 `feasible_bounds()` 在该开关关闭后仍执行 `_one_step_bounds_q()`。必须通过显式控制语义进入设备额定功率分支，且 `plan()`、`state()`、联合动作投影和 MILP 全部使用同一分支。

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

服务损失系数应体现接受的服务交换关系，并检查是否足以覆盖省电、碳、削峰带来的综合收益。有限大系数仍是软约束，不等于硬投影；过大主要带来数值尺度和学习难度问题，过小则可能学到主动放弃服务。两者均需标定，不能仅凭“高于电价”确定。

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

对 forecast 分支，将 `forecast_encoder` 输入维度从 6 改为 8；但整个 Reward V2 迁移不能只改这一层：还需把峰值和当前碳强度接入现有 trade token，并更新 schema、版本校验及相关 encoder 输入维度。总 token 数仍为 14。

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
algorithm:
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
- `core/forecast.py`、`core/profiles.py`、`core/data_manager.py`
- `algorithms/matd3_dispatch/algorithm.py`（仅 V2 元数据和兼容性检查，不改 canonical MATD3）
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
8. 两个 smoke 各使用 1–3 个完整自然日，验证正式测试实际加载 `checkpoints/best_model.pt`；每个运行默认输出 15 PNG + 15 PDF。
9. 结果位于 `outputs/working_runs/<resolved_slug>/runs/<run_id>/`，其中 CLI `--algorithm matd3` 解析为 `matd3_dispatch` 并使用其 YAML；不得混用永久参考 `configs/algorithms/matd3.yaml`。
10. 确认 canonical `algorithms/matd3/` 无变更，历史消融结果和 releases 均未改写。若测试 promotion，另验源运行仍存在、manifest、README、registry 完整；未明确接受 dirty 风险不得使用 `--allow-dirty`。

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

---

## 18. 评审发现与必须补齐的设计

### 18.1 峰值成本依赖历史，必须进入状态

当前 `core/observation.py` 和 `core/global_state.py` 没有当日实际负荷历史最大值。普通 T 中名为 `sys_margin` 的量来自上一时刻 P2P 交易量，不是当天负荷峰值，不能复用其语义。

例如，当前负荷同为 150 kW，历史峰值分别为 100 与 200 kW，新增峰值分别为 50 与 0 kW。不给 Critic 这项状态，就会把不同即时成本的情况混在一起；Actor 也无法知道当前是否仍有不增加峰值的调度空间。

具体规则：

1. 环境维护 `daily_load_peak_kw`，reset 置零；在 step 开始保存 `peak_before_kw`，按实际执行负荷计算增量，再更新峰值。
2. 各 Actor 观测与全局状态均加入 `team_load_peak_so_far`。按共享的固定 `team_power_scale_kw` 归一化，尺度由训练数据/设备配置确定，写入 snapshot，不使用本次 episode 的最大值。
3. 普通 T 将该特征加入现有 trade token；平坦 MATD3 通过新 observation schema 接收，无需修改 canonical Actor。
4. 逐步轨迹保存 `team_actual_load_kw`、`peak_before_kw`、`delta_peak_kw`、`peak_after_kw`。不得让动作 t 的输入包含动作执行后才知道的峰值。
5. schema、normalizer、观测语义版本同时写入两种当前插件的 checkpoint；新旧 schema 不混装，也不混用旧 replay。

这只能消除新奖励引入的隐藏历史变量，不应据此宣称现有部分观测预测系统已经完全满足 Markov 假设。

### 18.2 日成本目标与 gamma=0.99 的冲突

两个当前算法 YAML 均使用 `gamma: 0.99`。训练优化折扣回报，而 metrics 汇总的是未折扣日回报：

```text
训练：sum_t gamma^t * R_t
每日评价：sum_t R_t
```

相同的 100 kW 新增峰值若发生在第 0 步与第 23 步，折扣峰值成本分别为 `100*p` 与约 `79.36*p`。离站短缺的远期处罚也会折扣。因此原稿的“每日最大负荷收费等价”仅在未折扣总和层面成立。

**本修订推荐在 V2 的两个当前算法配置中统一使用 `gamma: 1.0`**，因为研究对象是完整 24 小时有限时域日总成本。canonical `configs/algorithms/matd3.yaml` 和更新逻辑不动。小时/剩余时间须可观测，真实第 24 小时终止时禁用 bootstrap；工具中断或训练取消不能伪装成物理终止。

这是一项需要短训练验证的设计选择，不是稳定性保证。记录 Q 值、TD error、梯度和非有限值；若训练不稳定，重新评审这一选择。若最终保留 0.99，则明确声明“折扣训练目标 + 未折扣评价”，保存两种 return，MILP 只能对比未折扣成本，不能再写训练目标完全一致。本轮不使用 `gamma^-t` 放大奖励等补偿技巧。

### 18.3 碳成本的数据来源与可观测性

当前 profiles/data manager 提及 `grid_carbon_factor_kg_per_kwh`，但 Actor/全局状态中未暴露该字段。不能只在奖励里突然增加一个时变且不可见的进口价格因素。

- V2 强制校验该列存在、有限、非负；没有列时只允许显式配置固定碳因子并记录来源，禁止静默以 0 或 1 代替。
- 当前碳因子视为调度前可获得的公共量，加入两种算法观测及全局状态。若实际数据不支持提前可知，只能标为仿真信息假设，不能声称部署条件成立。
- 普通 T 接入现有 trade token，不增加 forecast token 维数；保持未来 token 8 维，不偷偷加入未来实际碳因子。
- 本轮不提供未来碳预测。若碳因子随时间变化，预测不完全属于该方案的信息限制；后续另立预测消融。
- 使用训练集固定归一化尺度，记录 `carbon_information_semantics`。环境、MILP、轨迹复算使用同一原始因子与单位。
- 若因子恒定，碳项等价于给进口电价加固定附加价，可以保留其解释性，但不能据此声称学会了时变碳调度。

### 18.4 环境执行和 Critic 的投影必须一致

当前 `Matd3Dispatch` 明确让 Critic 接收三个有效控制量 `[ESS-local, ESS-trade, DR-effective]`，rollout、target action 和 policy gradient 都依赖 `JointActionProjector`。只改设备 `plan()` 会产生“环境执行一种动作、Critic 训练另一种动作”的错误。

迁移范围必须包括：设备瞬时上下界 → observation/state → 联合投影 → replay/update/target → 执行后 effective action。保留联合投影中的物理映射，只移除 HVAC 舒适与 EV deadline 的行为强制，不删除整个 projector。

EV 组内下界、站端总下界、初始 group allocation 均为零。总上界取站端功率限制与当前各连接组物理上界之和的较小值，组上界包含 `remaining_battery_kwh / (eta * delta_t)`。沿用 urgency 顺序分配，但总分配不得超过 Actor 请求量；`raw=-1` 映射到零充电，`raw=0` 通常是区间中点，不能把二者混同。

对 HVAC 正负功率、EV 零/满功率及 ESS 各分支建立固定场景，比较 NumPy 环境执行和 Torch 投影的有效控制量，含边界和随机合法输入；再验证非饱和位置 Actor 梯度有限且可传播。不能仅检查“LP 不再调用”。

### 18.5 日末服务事件、单位与状态重置

**HVAC：** 继续采用结果边界温度计费。`violation(T[t+1], band[t+1]) * delta_t` 是离散边界采样的度时近似，不是区间内真实温度轨迹积分；reward、metrics、MILP 都应标记 `boundary_sampled_degree_hours` 并同口径。当前 24:00 边界定义为 unoccupied；固定在本轮沿用，测试上班前/下班后边界与周末。若未来改变营业时间跨午夜，需要一并修改此合同。

**EV：** shortfall 是电池侧 kWh，进口电量是站端 kWh，两者通过效率区分。每组在执行最后一个可充电区间后结算一次；例如 departure=24 在第 23 步 commit 之后计入最后一个 reward，再返回 done。不得累计短缺每小时重复扣费，不得在 terminal penalty 再扣一次，也不得在 reset 时漏掉离站事件。当前生成器限制 `0 <= arrival < departure <= 24`；本轮保持 24 步、1 小时的完整自然日，1–3 天 smoke 是完整日的数量，不是把一天截成 1–3 步。

**MG1/ESS/HVAC：** 继续不做日末库存/SOC/温度恢复，但必须承认模型是独立单日问题。MG1 初始库存 10.53 t、下限 3.51 t，单日可净消耗 7.02 t 初始库存而保持消费服务；按当前额定参数，约相当于 168.7 kWh 磨机电耗。这是允许的初始资源消耗，不是等量生产的纯移峰。

每日报告初末 silo、ESS energy、空气/墙体温度，以及磨机生产/消费总量；RL、规则策略、MILP 使用相同初值与无终值要求。不得将该结果直接外推为连续多日运行的可持续收益。如果研究目标改为长期连续调度，应另开跨日状态衔接或终值研究，不在本轮偷偷恢复硬终端约束或加终端 bonus。

### 18.6 广义成本与 Equal/Shapley 结算分离

当前 `evaluate_day_coalitions()` 将 `result.total_profit` 写入七个 `v_*`；`settle_test_day()` 再以这些值和实际 `physical_profit_by_microgrid` 计算转移支付。把 `total_profit` 改成负广义成本会混用口径，甚至使转移支付的解释失效。峰值、碳和舒适偏好也不一定是真实支付。

比较三种可行路线：

| 路线 | 含义 | 本轮选择 |
|---|---|---|
| 保留会计结算，另算广义成本参考 | 七联盟会计值用于结算；独立全联盟求解用于 V2 性能参考 | **采用**，含义清楚，额外一日一次全联盟求解 |
| 所有联盟改为广义效用博弈 | 重新定义各节点广义效用、共同峰值分摊和转移支付 | 暂不采用，扩大研究范围 |
| 将广义最优轨迹的会计收益直接当联盟最优收益 | 节省独立求解但不再是会计最优联盟价值 | 不采用，避免隐含改变博弈定义 |

具体接口要求：

1. `solve_dispatch(..., objective_kind=...)` 显式支持 `accounting_profit` 与 `generalized_cost_v2`，共享约束与后处理，不复制求解器。
2. 七个非空联盟继续求会计目标，使用与 RL 对齐的 V2 物理边界和软服务语义、同一天、同初值、禁止外送；额外动作自由度按第 18.7 节披露，不默认声称可行域完全相同。保留原 `v_*` 作为会计值；固定 `v(empty)=0`。其服务策略可能不同，必须一同输出服务指标，不冒称 V2 reward 上界。
3. 另解一个全联盟广义成本参考，保存 `generalized_reference_*`。七联盟与此参考分开记录目标、轨迹、状态和耗时，不能复用一个 `grand_result` 混淆二者。
4. 为避免改变历史结算含义，V2 的兼容会计利润固定为：售电收入（零）减购电、退化、弃电、安全未供电、EV 离站罚款；terminal mode=none 时不计终端罚款。EV 罚款系数与广义成本采用同一来源。它含已有罚款，**不是纯现金利润**；另报 `operating_margin_before_penalties = export_revenue - import_cost - degradation_cost`，明确其中退化是估算成本，不将舒适/碳/峰值偏好自动算成现金。
5. `team_profit = sum(physical_profit_by_microgrid)`；P2P 转移在团队层面相消。广义成本直接由九个 raw component 相加构建，不通过 `-team_profit + penalties` 拼凑，防止 EV/弃电重复扣除。
6. Equal/Shapley 仍只分配实际会计收益相对会计 disagreement 的 surplus。负 surplus 必须如实报告，不能保证每个成员个体理性；不通过裁剪负收益来伪造改善。保存 `settlement_value_semantics=accounting_profit_v2`。
7. 验证团队收益不变、转移和为零、七联盟值和实际收益口径一致。广义效用分配若将来需要，再单独设计，不能无声替换现有列。

因此第 11 节的目标一致性要求适用于**新增的广义全联盟参考**；七联盟结算参考明确采用另一套会计目标，不称作 Reward V2 上界。

### 18.7 MILP 不能只“增加几个 objective 系数”

当前 `optimization/common_constraints.py` 不仅组装约束，也后处理利润，并用 `-total_profit` 填 objective。引入新成本时要同步模型和 `DispatchResult`，否则求解目标与报告目标会脱节。

- 新增独立 `objective_kind`、`generalized_system_cost`、`accounting_profit` 和逐项 breakdown，保存 solver objective 与根据轨迹重算的 objective，误差须在容差内。
- HVAC 使用正/负舒适 slack 和互斥加热/制冷模式，保证 `abs(signed_power)` 与环境相同；EV 同时保留连接/功率/剩余需求约束与 departure slack。检查软化后没有把原舒适温度变量上下界留成硬限制。
- 只有负荷峰值变量，不把 ESS charge 或 grid import 混入；联盟子集若使用峰值目标，只计该联盟成员。
- 复核新能源负荷优先、ESS 不能相互充电、禁止电网转售、ESS 充放互斥及各设备动作自由度。MILP 的自由 EV 分组分配可比环境固定 urgency 分配更宽，必须说明。
- `optimization/oracle_replay.py` 若把 MILP 控制重新映射到环境原始动作，需报告“求解轨迹”和“可执行回放轨迹”差异。回放后的成本不能继续冒称原 MILP 最优值。
- 只有证明可行域覆盖、目标一致并获得最优证书，才称最小广义成本的下界（负成本/return 的上界）。时间限制下的 incumbent 不是最小成本下界；分别保存 best bound、gap、termination status。
- 任一联盟失败/无可用解，该日 Shapley 标为 unavailable，不能用 0 补齐；有可行但未证最优解时标为近似值，不隐瞒 gap。

### 18.8 权重诱导行为和研究有效性

原稿把各分量列齐，但还缺“策略如何利用奖励”的诊断。

- 峰值下降可能来自少给 EV 充电或让房间不舒适，不一定是成功移峰。结果必须并列给出服务短缺、温度违约、用电量、产量与库存变化。
- 弃电惩罚可能鼓励不必要的 HVAC 用电、磨机囤料或储能耗散。新能源消纳上升不能单独视为收益：同时报告有效服务、ESS 吞吐/损耗、额外用电和末端库存。若异常耗能在当前偏好下确实最优，应调整权重或单独修订目标，不暗加动作奖励掩盖。
- EV 惩罚以电池侧 kWh 标定，比较 `购电价/eta + 碳价*因子/eta` 以及可能增加的峰值成本。例：若某补能必须在峰值时完成，峰值边际代价也必须考虑；仅让 EV penalty 大于最高电价并不够。
- MG1 在瞬时硬库存约束和无限购电下服务损失通常为零，它是诊断/异常项，不应期待它提供常规学习信号。
- 权重与 reward_scale 统一放在共享环境配置，算法 YAML 仅保留算法超参数。旧 `renewable_curtailment_cost_per_kwh`、EV `departure_penalty_per_kwh` 与新名建立单一解析入口；同时出现且冲突时报错，不允许环境与 MILP 读取不同系数。

标定产物为 `reward_calibration.json`（训练/校验日期、策略、原始分量分布、候选权重、边际理由、推荐值、reward_scale、配置指纹）。至少比较零控制、固定服务优先规则与短训练策略；HVAC 零控制是 signed power=0，EV 零控制是 raw=-1，ESS 按实际动作映射单独构造，不使用统一全零 action 冒充 idle。冻结系数后再做正式训练，不用 test 做反复选权重。

第 2 节的旧 55 天结果已经参与 H8+peer 的结构选择，因此该批数据对这一选择并非全新盲测。保留既定数据划分可以做可比评估，但应在报告披露；更强泛化结论需要新的未触碰时间段。本次评审未重新运行四组实验，原数值按已有设计证据保留，正式报告应链接可追溯的运行/配对结果。

### 18.9 Checkpoint 选择必须失败即停

当前 selector 对缺失 objective 使用 `0.0`，新回报通常为负，这可能让缺字段的 checkpoint 胜出。pipeline 还专门判断字符串 `constrained_profit`；只添加新模式名会绕过已有“无合格模型不得测试”的保护。

推荐新增明确的 `physical_constrained_objective` 模式，最大化 `team_reward_mean`，仅使用真实物理约束。selector 与 pipeline 统一通过模式能力判断，不到处写特定字符串。

- 必需 objective/constraint 缺失、NaN、Inf 一律报错或不可入选，不默认零；未知模式名启动即报错。
- 物理约束包含功率/来源/P2P 残差、SOC/设备上下界、零 grid export、当前无限购电下的零 unmet。具体字段须先在 metrics 中实现并核验，不能只在 YAML 发明不存在的名字。
- 对环境断言失败和非有限值立即停止；不把它们当可通过 reward 补偿的软目标。
- 正式运行要求 validation enabled；没有物理合格 best_model 时停止，保留失败诊断，不复制 final_model 冒充 best。
- 两个当前插件统一规则；历史模式仍可读取，旧结果不重算。服务指标继续全量报告，但不参与硬 eligibility。

### 18.10 预测信息和兼容性边界

- peer balance 是同一预测快照的 `PV_hat + wind_hat - base_load_hat`，不是包含未来 DR 动作的实际剩余电量，不得读取未来 action、SOC 或实际灵活负荷。
- 保持共享 observation future_horizon=24 的现有合同，普通 T 仅消费 t+1…t+8；H8 指 Actor forecast 窗口，不代表全环境或集中 Critic 都只看到 8 小时。平坦 MATD3 输入范围应在对比报告列明。
- 测试策略无探索噪声；如保留 `evaluation_forecast_noise=true`，使用固定、与算法无关的日场景种子和预测快照。确定性正式测试不等于完美预知，不能顺手关闭预测误差。
- mask 同时用于 attention key/value 和任何聚合/readout。编码前无效 forecast 字段置零，验证所有未来 token 都无效的最后一步仍有限，不让 NaN 经编码污染结果。
- B 组对齐测试只比较 forecast 特征顺序/数值、归一化和 mask；V2 增加公共峰值/碳状态后不要求整个 Actor 与历史 B 组动作一致。
- 增加 `observation_semantics_version`、`reward_discount_semantics`、`comfort_metric_semantics`、`settlement_value_semantics` 以及环境配置/观测 schema 指纹。模型加载须校验奖励、投影和观测语义，即使张量形状相同也不默认为兼容。
- 旧缺版本配置只允许显式历史回放路径；新正式训练缺关键 V2 元数据应报错，避免训练到一半恢复了旧 replay 或 optimizer 状态。

---

## 19. 详细修改顺序与交付物

以下是可供实施的文件级修改方案，不表示这些代码已经修改。各阶段都沿用一个共享流水线，不新增实验专用 trainer/evaluator。

| 阶段 | 修改位置 | 具体交付与通过条件 |
|---|---|---|
| 1. 冻结合同 | `configs/cooperative.yaml`、配置解析、两个当前算法 YAML | 明确 V2 语义、禁止外送、完整日限制、系数单一来源、gamma=1 推荐配置；对非法/冲突配置失败即停；历史默认不被默默改写 |
| 2. 放开服务动作 | `components/commercial_hvac.py`、`components/ev_charging.py`、`core/joint_action_projection.py` | HVAC 额定功率分支、EV 零下界；环境/Torch 投影一致；EV departure=24 只扣一次；物理边界保持 |
| 3. 建立成本账本 | `core/cost_accounting.py`、`envs/multi_microgrid_env.py`、`core/grid_settlement.py` | 九个原始分量与金额独立计算；峰值 reset/update；会计 profit 和 generalized cost 分层；团队奖励仅写一次且所有 Agent 相同 |
| 4. 补状态与预测 | `core/observation.py`、`core/global_state.py`、数据校验、普通 T actor/algorithm、当前 MATD3 包 | 峰值/当前碳强度公共状态；普通 T 8 维 forecast、14 token；trade token 扩展；checkpoint 严格 schema/语义校验；canonical 不动 |
| 5. 接通评价 | `utils/metrics_collector.py`、`evaluation/service_metrics.py`、`evaluation/evaluate_algorithm.py` 及内部 `evaluate_matd3.py`、selector/pipeline | 日广义成本与回报一致；物理合格 selector；缺字段失败；best checkpoint 正式测试；训练曲线与利润标签分开 |
| 6. 接通参考与结算 | `optimization/common_constraints.py`、合作/独立入口、oracle replay、coalition evaluator、settlement | 七联盟会计求解 + 一次广义全联盟求解；两套 grand result 分离；目标复算一致；预算平衡；求解状态不伪装 |
| 7. 标定和短验证 | 共享 pipeline、校验场景、配置快照、标定报告 | 生成原始分量诊断，固定权重和尺度；两个当前算法跑短训练与 1–3 完整日端到端 smoke；不据此声称收敛 |
| 8. 文档/图表/正式实验 | `evaluation/plotting.py`、输出 summary、现有研究报告入口 | 从保存的 JSON/CSV 作图；经济与广义指标分开；15 PNG+15 PDF；记录算法输入/服务取舍/日末资源限制后再长训 |

兼容边界：历史实验插件与结果保持原样；当前 `matd3_dispatch` 和普通 T 是本次迁移对象，新网络代码放在普通 T 自己包内。这次是明确的跨算法共享环境研究，不是在 canonical MATD3 内塞实验条件。若实施中产生新的算法变体，另按仓库 scaffold 工具建独立 slug，不能以这份设计为由改 canonical。

性能效果验证建议使用至少 3 个训练种子、相同训练预算和日场景，报告每个种子结果与日配对差异；3 个种子仅是起步证据。若要证明 peer 的纯贡献，固定 V2、gamma、公共状态、权重后比较普通 T 的 peer off/on；MATD3 vs 普通 T 的整体比较不能替代这一消融。

---

## 20. 新增验收矩阵与停止条件

除第 16 节外，实施完成必须覆盖下列测试：

| 测试 | 预期结果 |
|---|---|
| 相同当前状态、不同历史峰值 | 峰值观测与 Critic 状态不同，增量成本分别符合定义；next state 更新正确 |
| 峰值序列手工复算与跨日 reset | 未折扣日增量和等于最大负荷；第二天从零开始；不对 kW 再乘 delta_t |
| gamma 与最后一步 | V2 两算法配置相同；真实日末 done=1 无 bootstrap；最后一个 EV 事件包含在 reward |
| HVAC fallback | soft 模式既不调用 viability LP，也不调用 one-step comfort bounds；额定范围内 action 不因温度被重写 |
| EV 多组功率 | 每组和站端总下界均为零，连接/效率/剩余需求上界正确，总分配不超过总请求 |
| 双执行路径 | 环境与 Torch projector 的有效动作一致；非饱和梯度有效，饱和边界可解释 |
| 单步/全日独立复算 | 九项和等于 generalized cost；日 return=-日成本/scale；团队 reward 不因三 Agent 重复相加变三倍 |
| 会计与广义分离 | 一份 EV shortfall 在各自账本各计一次；会计账本作为展示不再被加入 reward；团队 P2P 收支相消 |
| 碳数据/观测 | 缺失或非法输入清晰报错；当前因子可见；没有未经声明的未来实际碳信息 |
| 未来 token | 有效 peer 改变中间 token；无效 token 任意修改不影响动作；全 forecast 无效仍有限；保持 H8 |
| checkpoint 异常 | 旧 schema/旧语义/旧 replay 不兼容时拒绝；objective 缺失或非有限不可入选；无 best 不测 final |
| MILP 与回放 | objective 与轨迹复算一致；记录 bound/gap；可执行回放若不同单独标记；会计 grand 与广义 grand 不混用 |
| 结算失败与负 surplus | 缺联盟不能补零；负 surplus 如实保留；预算平衡和团队会计收益守恒 |
| 结果完整性 | 两个 resolved plugin+自身 YAML、best checkpoint、test、七联盟、广义参考、Equal/Shapley、summary、15 PNG+15 PDF 全部可追溯 |

正式长训启动前必须同时满足：全量 `pytest -q` 通过、两算法端到端 smoke 通过、系数完成标定锁定、无重复计费/遗漏服务事件/状态泄漏、物理残差合格。smoke 仅验证流程和数值可运行，不验证学到的策略质量。

若出现 reward 改善但服务显著恶化、额外耗能消纳、库存消耗解释不清，先出诊断与权衡报告，再决定是否接受该偏好，不以 reward 曲线更高直接宣告方案成功。

本次文档评审只进行了源码核对与设计修订；未修改实现、未运行训练、未宣称上述实现验收通过。
