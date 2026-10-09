# 多微网协同调度

本仓库研究三个异构微网的新能源、储能、可控负荷、电网和 P2P 协同调度。当前主对比为 MATD3-Q 与 Dueling，另保留 DR 反事实实验。所有算法使用共享训练、验证、正式测试、联盟 MILP、Equal/Shapley 结算和绘图流程。

## 当前语义

当前共享配置 configs/cooperative.yaml 使用：

- action_design: single_dispatch_dr
- dispatch.priority_mode: renewable_load_first
- metrics_schema_version: service_metrics_v2
- HVAC soft_comfort，保留当前热模型修改
- P2P 无损、pair_capacity_kw: null
- 外部电网进口不限、出口上限 0 kW
- ESS 初始 SOC 0.10，独立自然日，不要求日末 SOC/磨机库存/HVAC 温度恢复
- 保留插件和共享配置统一使用 observation mode = peer_energy_summary

风光首先服务本地和联盟剩余负荷，之后才用于已选择的 ESS 充电；剩余风光在当前禁止外网售电的配置下弃电。Critic 的第二个 ESS 通道使用实际 P2P 售电量，不是未成交要约。

当前报表中的 Renewable Non-Curtailment（历史字段名仍可能含 renewable_utilization）定义为“可用新能源减弃电”，因此**包含外送大电网电量**；它不是“联盟内部吸收率”。判断本地/联盟消纳时应同时查看外网售电、P2P、储能充电和弃电来源账本。

详细数学与语义见 docs/SYSTEM_MODEL.md。

## 保留的算法

| 显示名 | 运行命令 | 实现与参数 |
|---|---|---|
| MATD3-Q（主对比） | python scripts/run_experiment.py --algorithm matd3_dispatch_q_guided | algorithms/matd3_dispatch_q_guided、configs/algorithms/matd3_dispatch_q_guided.yaml |
| Dueling（主对比） | python scripts/run_experiment.py --algorithm matd3_mlp_dueling_critic_h24_q_guided | algorithms/matd3_mlp_dueling_critic_h24_q_guided、configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided.yaml |
| DR 反事实实验 | python scripts/run_experiment.py --algorithm matd3_dr_counterfactual | algorithms/matd3_dr_counterfactual、configs/algorithms/matd3_dr_counterfactual.yaml |
| MATD3 | python scripts/run_experiment.py --algorithm matd3 | algorithms/matd3_dispatch、configs/algorithms/matd3_dispatch.yaml |
| 普通 T | python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer | algorithms/matd3_hierarchical_transformer、configs/algorithms/matd3_hierarchical_transformer.yaml |

默认算法仍是 MATD3。MATD3 和普通 T 是必要父实现；永久参考 algorithms/matd3 与 configs/algorithms/matd3.yaml 保持原样。DR 反事实的 baseline、actor_only、critic_only、combined 和 smoke 配置全部保留。

保留算法的历史运行和 checkpoint 原样保留在 outputs/working_runs/，另有仓库外备份，仍按保存时的环境配置和 action semantics 解释。被移除算法的结果已备份后清理。算法对比应使用相同的当前环境版本、数据划分、训练预算、种子组和验证选模规则。

## 使用

PowerShell：

    python scripts/run_experiment.py --algorithm matd3_dispatch_q_guided --device cuda --no-dashboard --seed 42
    python scripts/run_experiment.py --algorithm matd3_mlp_dueling_critic_h24_q_guided --device cuda --no-dashboard --seed 42
    python scripts/run_experiment.py --algorithm matd3_dr_counterfactual --device cuda --no-dashboard --seed 42
    pytest -q

也可使用 --experiment matd3、--experiment ordinary_t、--policy mlp 或 --policy transformer。算法超参数分别放在自己的 YAML；物理参数位于 configs/cooperative.yaml。

共享流程依次执行训练、验证选择 best_model.pt、正式测试、七个联盟的每日 24 小时 MILP、Equal/Shapley 结算、摘要和保存结果绘图。三个实验算法采用 physical_constrained_objective 选模：物理可行性约束合规后，按验证集 team_reward_mean 选择；舒适和服务成本进入广义成本奖励。

普通训练写入 outputs/working_runs/<algorithm>/runs/<run_id>/，目录由流水线自动创建。MATD3-Q、Dueling、DR 反事实，以及 MATD3 和普通 T 父实现的已有结果保留；其他算法结果、reports、output 和 analysis 已在仓库外备份后清理。

训练结果现在按两类查看：

```text
<run_id>/
├── figures/                    # 只放图片：00、01，以及春夏秋冬四张调度总览
└── artifacts/                  # 模型、配置、训练记录、测试数据、摘要及其他文件
    ├── checkpoints/
    ├── test_hourly_trajectories/
    ├── config_snapshot.yaml
    ├── training_history.json
    ├── test_summary.json
    └── plot_manifest.json      # 典型日选取依据、源文件 hash、图片清单
```

覆盖四季的正式测试生成 6 张 PNG 和 6 张 PDF；短测试仅生成已有季节。典型日按每季 24 小时外部条件曲线距离选取，所有算法使用相同测试集时选取相同日期。训练中不再执行独立学习曲线评估，也不生成对应的 01b 图片。图片使用通用英文字体标注，便于服务器导出。

只读取保存数据重绘，不重新训练或运行 MILP：

    python scripts/plot_results.py --run-dir <run目录>

重新绘图会清理该 working run 中旧的 PNG/PDF，保留模型和全部原始数据。旧目录结构仍可读取。指定的完整 DR 反事实运行已整理为上述结构；旧 Dueling 总览已从备份恢复到 reports/dispatch_review_20261006/ 作为布局参考。

历史 run 的服务指标审计必须输出到独立目录：

    python scripts/audit_saved_service_metrics.py --run <原run目录> --output <独立审计目录>

该命令只读取保存文件并记录源文件 hash，不修改原 run，也不允许写入 outputs/releases。

## 代码边界

- algorithms/matd3：永久参考实现，不修改。
- algorithms/matd3_dispatch、algorithms/matd3_hierarchical_transformer：保留的父实现。
- algorithms/matd3_dispatch_q_guided、algorithms/matd3_mlp_dueling_critic_h24_q_guided、algorithms/matd3_dr_counterfactual：当前实验算法。
- core/renewable_first_dispatch.py：v3 共享可微风光优先清算内核。
- components/hvac_viability.py：HVAC 剩余日预测可达性 LP。
- evaluation/service_metrics.py：统一服务指标 v2。
- optimization/common_constraints.py：来源优先、自由分配 MILP 参考。
- optimization/oracle_replay.py：设备轨迹和来源流量 replay gap 审计。
- experiments、evaluation：共享训练、评估、结算与结果生命周期。

新实验算法遵循根目录 AGENTS.md 与 docs/ALGORITHM_DEVELOPMENT.md。

## 迁移到服务器

迁移训练代码、保留配置、测试、requirements.txt 和 data/。需要沿用历史结果或恢复训练时，一并迁移对应的 outputs/working_runs/；从头训练的三个主配置不依赖历史 checkpoint。被移除算法的结果、reports/、output/、analysis/ 和开发过程记录已在仓库外备份后清理，本机完整备份不需要迁移。

显式传入 --device cuda，避免 MATD3-Q 与反事实配置默认使用 CPU。服务器上的 PyTorch 安装与线程配置需要按实际 GPU、驱动和测速结果选择；已有测速入口 scripts/benchmark_training_runtime.py 保留。

当前工作区含未提交代码，直接从旧 HEAD 导出会遗漏修改。迁移应包含当前工作区文件，或先由用户审阅后将需要的源文件提交，并保留原 Git 信息以记录运行的源代码身份。

短流程验证分别使用 configs/smoke/matd3_dispatch_q_guided.yaml、configs/smoke/matd3_mlp_dueling_critic_h24_q_guided.yaml 和 configs/algorithms/matd3_dr_counterfactual/smoke.yaml；通过统一入口追加 --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-dashboard --device cpu。
