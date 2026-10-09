# 算法

当前主对比为 MATD3-Q 和 Dueling，另保留 DR 反事实实验：

| 显示名 / 命令名 | 实现目录 | 配置 |
|---|---|---|
| MATD3-Q / `matd3_dispatch_q_guided` | `matd3_dispatch_q_guided` | `configs/algorithms/matd3_dispatch_q_guided.yaml` |
| Dueling / `matd3_mlp_dueling_critic_h24_q_guided` | `matd3_mlp_dueling_critic_h24_q_guided` | `configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided.yaml` |
| DR 反事实 / `matd3_dr_counterfactual` | `matd3_dr_counterfactual` | `configs/algorithms/matd3_dr_counterfactual.yaml` |
| MATD3 / `matd3` | `matd3_dispatch` | `configs/algorithms/matd3_dispatch.yaml` |
| 普通 T / `matd3_hierarchical_transformer` | `matd3_hierarchical_transformer` | `configs/algorithms/matd3_hierarchical_transformer.yaml` |

MATD3 和普通 T 是保留的父实现。所有插件使用 `peer_energy_summary` 观测和 `single_dispatch_dr` 双动作设计；MATD3 仍是全局默认。`matd3` 命令别名解析为 `matd3_dispatch`。`algorithms/matd3/` 是永久保留的参考源码。DR 反事实的消融和 smoke 配置全部保留。

保留算法的历史运行结果仍在 `outputs/working_runs/`，另有仓库外备份，保留原有配置与标签。环境、训练、评估、结算和绘图使用共享实现。详见根目录 `AGENTS.md` 与 `docs/ALGORITHM_DEVELOPMENT.md`。
