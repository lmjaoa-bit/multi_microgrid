# Algorithms

`algorithms/matd3` 是永久保留的标准基线：独立 Actor、集中式 Twin Critic、Replay Buffer、Target Networks、目标策略平滑、延迟策略更新和软目标更新。

每个新增算法使用独立子目录和 `plugin.py`，不要直接修改 `algorithms/matd3`。插件由名称按需加载，因此删除某个失败算法目录不会影响其他算法。完整规则见 `docs/ALGORITHM_DEVELOPMENT.md`。

当前 ESS 跨时空共享对照组：

| 算法名 | 动作 | P2P/ESS 共享机制 | Actor 信息 |
|---|---:|---|---|
| `matd3` | 1D | 既有剩余功率自动协调 | 原本地观测 |
| `matd3_independent` | 1D | 关闭 P2P | 原本地观测 |
| `matd3_ess_full_offer` | 1D | 可共享的 ESS 放电全部报价 | 原本地观测 |
| `matd3_ess_local_obs` | 2D | 学习 ESS 报价比例 | 本地状态与历史市场反馈 |
| `matd3_ess_sharing` | 2D | 学习 ESS 报价比例 | 跨微网状态、预测与市场反馈 |

五个方法均通过统一入口分别训练，例如：

```powershell
python scripts/run_experiment.py --algorithm matd3
python scripts/run_experiment.py --algorithm matd3_independent
python scripts/run_experiment.py --algorithm matd3_ess_full_offer
python scripts/run_experiment.py --algorithm matd3_ess_local_obs
python scripts/run_experiment.py --algorithm matd3_ess_sharing
```
