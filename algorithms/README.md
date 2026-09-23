# 算法

当前新训练只提供两个算法：

| 显示名 / 命令名 | 实现目录 | 配置 |
|---|---|---|
| MATD3 / `matd3` | `matd3_dispatch` | `configs/algorithms/matd3_dispatch.yaml` |
| 普通 T / `matd3_hierarchical_transformer` | `matd3_hierarchical_transformer` | `configs/algorithms/matd3_hierarchical_transformer.yaml` |

两个插件都使用 `peer_energy_summary` 观测和 `single_dispatch_dr` 双动作设计；MATD3 是全局默认。`matd3` 命令别名解析为 `matd3_dispatch`。`algorithms/matd3/` 是永久保留的参考源码，不作为第三个可选方案。

历史运行结果保留原有配置与标签，不因当前默认值改变。环境、训练、评估、结算和绘图使用共享实现。详见 `docs/EXPERIMENT_PROFILES.md`。
