# Algorithms

`algorithms/matd3` 是永久保留的标准基线：独立 Actor、集中式 Twin Critic、Replay Buffer、Target Networks、目标策略平滑、延迟策略更新和软目标更新。

每个新增算法使用独立子目录和 `plugin.py`，不要直接修改 `algorithms/matd3`。插件由名称按需加载，因此删除某个失败算法目录不会影响其他算法。完整规则见 `docs/ALGORITHM_DEVELOPMENT.md`。

当前 ESS 跨时空共享对照组：

| 算法名 | 动作 | P2P/ESS 共享机制 | Actor 信息 |
|---|---:|---|---|
| `matd3` | 1D | 既有剩余功率自动协调 | 原本地观测 |
| `matd3_ess_sharing` | 2D | 学习 ESS 报价比例 | 跨微网状态、预测与市场反馈 |
| `matd3_renewable_p2p_only` | 2D | 仅可再生能源 P2P，ESS 报价物理屏蔽 | 与 ESS Sharing 相同的跨微网信息 |
| `milp_ess_sharing_mpc` | 2D | 4 小时滚动 MILP 决定电池功率和 ESS 报价比例 | 与 ESS Sharing 相同的当前状态和预测窗口 |
| `milp_renewable_p2p_only` | 2D | 同窗口滚动 MILP，仅允许可再生能源跨微网交易 | 与 ESS Sharing 相同的当前状态和预测窗口 |

五个保留方法均通过统一入口运行，例如：

```powershell
python scripts/run_experiment.py --algorithm matd3
python scripts/run_experiment.py --algorithm matd3_ess_sharing
python scripts/run_experiment.py --algorithm matd3_renewable_p2p_only
python scripts/run_experiment.py --algorithm milp_ess_sharing_mpc
python scripts/run_experiment.py --algorithm milp_renewable_p2p_only
```
