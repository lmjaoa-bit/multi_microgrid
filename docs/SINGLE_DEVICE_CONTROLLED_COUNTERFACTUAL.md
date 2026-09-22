# 单设备受控反事实：原始数据版（诊断，不是环境可兑现收益）

本分支基于 `codex/clairvoyant-gap-replay-audit-20260922`。它不修改 MATD3、Transformer、环境清算、现有全知 MILP 或 Shapley 评估。新增独立入口 `scripts/run_single_device_counterfactual.py`。

## 输入与数据安全门槛

必须提供研究交接压缩包，其中包含 `code/perfect_foresight_dr.py`、`data/revised_55_counterfactual_daily.csv` 和两算法逐小时 RL 轨迹。程序从**仓库原始** `data/spatial_market_2018_train.csv` / `data/spatial_market_2018_test.csv` 加载天气和异构微网负荷，不使用轨迹反推天气。程序检查 RL 真实基础负荷/可再生与原始数据一致，并且逐日重新求解 `free`，要求其成本在默认 ±0.02 内复现交接包**修订版**历史日成本。若失败，会保存显式失败状态，不输出该日设备收益。旧 555.342194/天参考模型与修订模型不能混算。

独立源模型通过精确锚点注入“按微网锁定”选择器，继续采用交接包修订版的 source-aware 限制、24h 设备动态和费用。未改变交接模型的 HVAC 舒适硬约束及限供处理，因此不能悄悄把原先 `2018-01-29` 和 `2018-02-26` 的不可行记录删掉，或将本模型称作环境严格理论上限。

## 七组对照实验

| 场景 | MG1 ESS | MG2 ESS | MG3 ESS | 工业负载 | HVAC | EV | 交易及主网 |
|---|---|---|---|---|---|---|---|
| fixed_all | 固定 | 固定 | 固定 | 固定 | 固定 | 固定 | 重优化 |
| release_mg1_ess | **释放** | 固定 | 固定 | 固定 | 固定 | 固定 | 重优化 |
| release_mg2_ess | 固定 | **释放** | 固定 | 固定 | 固定 | 固定 | 重优化 |
| release_mg3_ess | 固定 | 固定 | **释放** | 固定 | 固定 | 固定 | 重优化 |
| release_mg1_mill | 固定 | 固定 | 固定 | **释放** | 固定 | 固定 | 重优化 |
| release_mg2_hvac | 固定 | 固定 | 固定 | 固定 | **释放** | 固定 | 重优化 |
| release_mg3_ev | 固定 | 固定 | 固定 | 固定 | 固定 | **释放** | 重优化 |

`controlled_cost_reduction = C(fixed_all) - C(release_one)`。每次只开放一个设备的整日动作轨迹，其余五个设备保持该 RL 策略实际**执行**的每小时充放电/柔性电负荷，交易与主网随供需改变重新优化。该差值是同一放宽 MILP 内的**受控成本下降**；六项通常不可相加，也不能直接当作某个 Actor 在真实环境的可兑现收益。

## PowerShell

在项目根目录：

```powershell
git fetch origin
git restore --source origin/codex/clairvoyant-single-device-counterfactual-20260922 -- scripts/run_single_device_counterfactual.py docs/SINGLE_DEVICE_CONTROLLED_COUNTERFACTUAL.md

# 第一步：只核对一个已有完整历史结果的测试日。
python scripts/run_single_device_counterfactual.py --handoff-zip "C:\实际路径\全知MILP差距分解_交接数据包.zip" --day 2018-12-31

# 单日所有七组通过后，去掉 --day 求解完整 55 天 × 2 算法。
python scripts/run_single_device_counterfactual.py --handoff-zip "C:\实际路径\全知MILP差距分解_交接数据包.zip"
```

需替换 `--handoff-zip` 的实际路径。可设置 `--output`、`--time-limit`、`--mip-gap`。程序在 `outputs/single_device_cf/` 增量保存：
- `baseline_reproduction.csv`：55 天原始数据与修订版历史自由优化成本复现情况；
- `single_device_scenarios.csv`：每一天每种算法七组方案的求解状态、原成本、受控下降、分项成本、MIP gap、约束残差、失败理由；
- `single_device_summary.json`：两算法**完全相同且七组都有效**的日期集合、六设备日均受控成本下降，以及全部失败条目。

若任何日期的任一算法某场景失败，脚本会保留失败信息，并在末尾返回退出码 2。此退出码并不表示结果文件丢失。预计历史硬 HVAC/限供不一致可能仍令两天无法完整配对，但必须以这次使用仓库原始 CSV 的实际执行结果为准。

## 最终研究判读

报告先列相同有效样本数，再比较六个 `controlled_cost_reduction` 的均值和逐日分布。单日诊断或少数有效日期不能替代完整 55 天。即使 53 天重现了原修订模型的总 ESS/DR 趋势，也**不能**以六设备下降额相加表示总可优化空间；与原环境的 local-first P2P、`delivered_fraction`、软舒适违约、原动作回放逐小时等价尚未认证。
