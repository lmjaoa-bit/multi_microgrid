# 多微网协同调度

本仓库研究三个异构微网的新能源、储能、可控负荷、电网和 P2P 协同调度。当前新训练保留 MATD3 和普通 T 两个学习算法；联盟 MILP、Equal/Shapley 结算与绘图是共享评估流程。

## 当前语义

当前共享配置 configs/cooperative.yaml 使用：

- action_semantics: single_dispatch_dr_v3
- dispatch.priority_mode: renewable_load_first
- metrics_schema_version: service_metrics_v2
- HVAC forecast_viability
- P2P 无损、pair_capacity_kw: null
- 外部电网进口不限、每个微网出口上限 100 kW
- ESS 初始 SOC 0.10，独立自然日，不要求日末 SOC/磨机库存/HVAC 温度恢复
- 当前两个插件和共享配置统一使用 observation mode = peer_energy_summary

风光首先服务本地和联盟剩余负荷，之后才用于已选择的 ESS 充电；最后剩余的本地风光才进入大电网外送/弃电。Critic 的第二个 ESS 通道使用实际 P2P 售电量，不是未成交要约。

当前报表中的 Renewable Non-Curtailment（历史字段名仍可能含 renewable_utilization）定义为“可用新能源减弃电”，因此**包含外送大电网电量**；它不是“联盟内部吸收率”。判断本地/联盟消纳时应同时查看外网售电、P2P、储能充电和弃电来源账本。

详细数学与语义见 docs/SYSTEM_MODEL.md。

## 两个算法

| 显示名 | 运行命令 | 实现与参数 |
|---|---|---|
| MATD3 | python scripts/run_experiment.py --algorithm matd3 | algorithms/matd3_dispatch、configs/algorithms/matd3_dispatch.yaml |
| 普通 T | python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer | algorithms/matd3_hierarchical_transformer、configs/algorithms/matd3_hierarchical_transformer.yaml |

默认算法是 MATD3。两者共享同一个物理环境与双动作设计，只改变 Actor 架构。永久参考 algorithms/matd3 与 configs/algorithms/matd3.yaml 不参与本轮 v3 修改。

历史运行和旧 checkpoint 原样保留。它们按保存时的 config/action semantics 解释，不能重新标记为 v3，也不能用当前服务口径补造当时未保存的证据。

## 使用

PowerShell：

    python scripts/run_experiment.py --algorithm matd3
    python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer
    pytest -q

也可使用 --experiment matd3、--experiment ordinary_t、--policy mlp 或 --policy transformer。算法超参数分别放在自己的 YAML；物理参数位于 configs/cooperative.yaml。

共享流程依次执行训练、验证选择 best_model.pt、正式测试、七个联盟的每日 24 小时 MILP、Equal/Shapley 结算、摘要和保存结果绘图。当前 MATD3 与普通 T 使用相同的 constrained-profit checkpoint selection：先要求 strict feasible、零未供电、HVAC 舒适、磨机下游服务和 EV 离场服务满足约束，再在合规候选中按验证集团队利润选择。关闭的日末恢复约束和磨机产量覆盖率不作为该门槛。

普通训练写入 outputs/working_runs/<algorithm>/runs/<run_id>/。历史 outputs、reports 和 analysis 保留。

历史 run 的服务指标审计必须输出到独立目录：

    python scripts/audit_saved_service_metrics.py --run <原run目录> --output <独立审计目录>

该命令只读取保存文件并记录源文件 hash，不修改原 run，也不允许写入 outputs/releases。

## 代码边界

- algorithms/matd3：永久参考实现，不修改。
- algorithms/matd3_dispatch、algorithms/matd3_hierarchical_transformer：当前两个运行算法。
- core/renewable_first_dispatch.py：v3 共享可微风光优先清算内核。
- components/hvac_viability.py：HVAC 剩余日预测可达性 LP。
- evaluation/service_metrics.py：统一服务指标 v2。
- optimization/common_constraints.py：来源优先、自由分配 MILP 参考。
- optimization/oracle_replay.py：设备轨迹和来源流量 replay gap 审计。
- experiments、evaluation：共享训练、评估、结算与结果生命周期。

新实验算法遵循根目录 AGENTS.md 与 docs/ALGORITHM_DEVELOPMENT.md。
