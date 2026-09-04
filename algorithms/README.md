# 算法

当前运行方案只有两个：

| 显示名 / 命令名 | 独立实现目录 | 配置 |
|---|---|---|
| MATD3 / matd3 | matd3_dispatch | configs/algorithms/matd3_dispatch.yaml |
| CrossGate / crossgate | crossgate | configs/algorithms/crossgate.yaml |

两个方案共用单调度动作，普通 MATD3 为全局默认。`matd3` 命令别名解析为 `matd3_dispatch`。
`algorithms/matd3/` 是永久参考源代码，供保留算法复用 Actor、Critic、Replay 等基础实现，不作为第三个可选方案。
环境、训练、评估、结算、绘图仍使用共享实现。详见 `docs/EXPERIMENT_PROFILES.md`。
