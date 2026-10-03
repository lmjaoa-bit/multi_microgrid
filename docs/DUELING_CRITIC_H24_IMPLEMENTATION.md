# Dueling Critic T-Q H24 实施与验收记录

日期：2026-10-03。实施依据：[设计与实施规范](design/matd3_mlp_dueling_critic_h24_q_guided.md)。

## 当前状态

新算法代码、独立配置、T1–T12 专项验证和完整流水线冒烟已落地。用户随后明确授权修复三项既有共享测试失败并整理提交当前研究代码；最终全量复测 **638 passed，4 warnings，169.93 秒**，包括全部 50 项新算法专项测试和 18 项正式运行/报告工具测试。

用户最新要求：**正式训练由用户运行**。本会话没有启动正式训练。已交付单次训练命令和可选的九次顺序对照工具，研究判定仍待完整正式结果。

## 交付文件

- `algorithms/matd3_mlp_dueling_critic_h24_q_guided/{__init__,plugin,algorithm,actor,critic}.py`
- `configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided.yaml`
- `configs/smoke/matd3_mlp_dueling_critic_h24_q_guided.yaml`：仅用于覆盖实际更新与诊断落盘。
- `tests/test_matd3_mlp_dueling_critic_h24_q_guided.py`
- `scripts/run_dueling_h24_study.py`：严格调用现有 shared pipeline 的顺序对照入口。
- `diagnostics/dueling_study_report.py`：只读保存结果的配对分析与 Markdown/JSON 报告。
- `tests/test_dueling_h24_study.py`：配置、配对统计、干净提交、中断、锁恢复与日志保留验证。
- 两个既有插件发现测试文件：仅在三处预期插件列表中加入新 slug，保留原有未提交改动。
- [实施计划](superpowers/plans/2026-10-03-dueling-critic-h24.md)。

## 结构与隔离

每个 Q 独立计算 `Q(s,a) = V(s) + A(s,a) - A(s,0)`。V 使用状态当前 token、24 个预测 token、相同的预测 mask 与 pooling；A 使用状态专属 LayerNorm、默认 detach 的 V 上下文和直接输入的有效动作，并在第二层再次注入动作。

Twin Critic 的 V、A 均不共享参数。A 的输出权重在 ±0.003 范围均匀初始化，bias 为零。Q1、Q2 的共享接口与 H24 相同；`decompose` 仅用于诊断。

Actor 文件逐字复制 H24 原文件主体，头部记录来源 checkout HEAD 和源文件 SHA256。来源 H24 文件在任务开始时尚未被 Git 跟踪，因此该历史 HEAD 不能单独重建原 Actor；文件哈希明确记录了实际使用的来源。正式实验应使用交付后的完整干净提交，其中原始 H24 和新增算法均已跟踪。

运行时依赖只使用自身包、正式 Transformer 父类和现有 core。新插件在禁止导入 H24 对照包的独立进程中仍可加载。没有对子类重复安装共享诊断 wrapper。

正式 YAML 与 H24 在解析后仅 `algorithm.name`、`algorithm.critic_network` 不同。Actor、Q-guidance 和所有其他训练超参数保持一致。已通过 AST 核对：Q-guidance 原方法主体、`select_actions`、`update`、`save`、`load`、`_checkpoint_metadata` 均保持一致；仅把 guidance 主体移到本地 helper，以覆盖其所有提前返回后的诊断记录。

## 诊断实现与规范中的冲突

共享 `DiagnosticSchedule.due()` 会推进 `next_boundary`。它已在共享 update 的 Actor 阶段之前调用；设计 §4.4 所述再调用一次会返回 false，丢失全部新诊断。

本地 hook 观察共享 cursor 的推进，缓存当前 update step，供三个 agent 使用同一次调度结果。它不重复调用或修改共享 schedule，不改变共享更新代码。测试覆盖 interval 1、2、3、5 和稀疏/密集诊断下的训练轨迹。

七个指标均通过现有 auxiliary metrics 返回。统计在 `no_grad` 下计算，dQ/da 仅对 detach 的有效动作副本求导；不调用参数 backward，也不改变已有参数梯度。单样本方差使用 population std 防止 NaN；正常 batch 的 adv share 使用规范中的 sample std。诊断关闭和开启时 Actor、Critic、两类目标网络的参数逐元素完全一致。

## 专项测试

专项测试覆盖 T1–T12，并补充 schema 异常、三个 A 分支消融开关、无 H24 包依赖，以及正式尺寸网络 CPU checkpoint → CUDA 实际更新。

最终 50 项新算法专项测试全部通过，包括 T1–T12 与 CUDA 更新；全仓库精确计数见下方。

## 完整流水线冒烟

两次运行均为 3 个训练日、2 个测试日、seed 42，结果全部位于规定的 working-run 目录。

| 运行 | 配置 | 实际更新 | 图表 |
|---|---|---|---|
| `DUELINGCRITICT-QH24_2026-10-03_22-20-56_seed42` | 本算法默认正式 YAML | 未达到 learning_starts=1000 | 15 组 canonical PNG/PDF，加 1 组学习曲线，共 16+16 |
| `DUELINGCRITICT-QH24_2026-10-03_22-23-49_seed42` | 独立 smoke YAML | 是，learning_starts=8 | 恰好 15 PNG + 15 PDF |

完整路径：

```text
outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/DUELINGCRITICT-QH24_2026-10-03_22-20-56_seed42/
outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/DUELINGCRITICT-QH24_2026-10-03_22-23-49_seed42/
```

默认运行严格使用设计 §7.1 命令。由于 72 步少于正式配置的 1000 步学习起点，不能用它验证更新或诊断落盘。补充的独立 smoke 只改变测试规模、学习起点、batch/capacity、诊断频率和 CPU 运行设置；不改变正式 YAML。

补充运行关闭额外学习曲线评估，保留 validation 与 MILP。best validation episode 为 1；正式测试 `checkpoint_used` 与 summary 的 `best_checkpoint` 均指向 `checkpoints/best_model.pt`，而非 final model。

两天的七种 coalition MILP、广义成本参考、Equal 和 Shapley 结算全部完成。`milp_failure_count=0`，`shapley_valid_day_count=2`，Shapley 最大预算残差约 `2.56e-13`。诊断 CSV 有 9 行（3 episodes × 3 agents），七种新增指标的 mean/p95/max/count/nonfinite_count 列完整，所有 nonfinite_count 均为零。

两次 run 的环境哈希均为：

```text
745ca262da0cebcdf5fc4f1f1b3bf3b7f60d5136161a4b2f858e813b6c988bf2
```

图表由共享 plotter 从保存的 JSON/CSV 生成。未改变绘图实现，也没有重新运行环境或 MILP来生成图表。没有 promote，也没有修改任何 release。

## 参数量与同机速度

设备：NVIDIA GeForce RTX 2060 SUPER；PyTorch 2.11.0+cu128；batch 128；CPU threads 4，interop threads 2。两算法均使用其正式 YAML，在已满学习起点的 replay 上运行真实环境与完整 update；每次 24 小时，先预热 2 episodes，再测 6 episodes，每段测量前后同步 CUDA。该测量覆盖 env、策略、Critic/Actor 更新和正式频率的诊断，不包括 validation、MILP 或绘图。

| 指标 | H24 | Dueling |
|---|---:|---:|
| 单 Q 参数 | 57,505 | 101,824 |
| 单 Q 的 V 分支 | — | 57,073 |
| 单 Q 的 A 分支（含状态 LayerNorm） | — | 44,751 |
| 每个 agent 的 Twin Critic | 115,010 | 203,648 |
| MG1 Actor | 64,568 | 64,568 |
| MG2 Actor | 64,906 | 64,906 |
| MG3 Actor | 66,593 | 66,593 |
| 平均秒/episode | 2.4875 | 2.5978 |

耗时比 **1.0444**，低于设计的 1.3 上限。V 比 H24 单 Q 少 432 个参数，恰好对应移除当前 token 输入中的 9 维动作（9 × 48）。这是短程速度验收数据，不是 5000-episode 收敛或利润证据。

用户授权共享修复后，使用相同正式网络尺寸、replay 预填充、2 次预热和 6 次测量再次验证：H24 为 2.6354 秒/episode，Dueling 为 2.7657 秒/episode，耗时比 **1.049473**，仍通过 1.3 上限；各网络参数量保持一致。原始两批计时证据分别保留。

## 改动范围核对

第一阶段对设计禁止修改的目录及文件保存 SHA256，排除缓存后共 69 个源文件。当时与初始工作区比较变化为 **0**。随后授权续作仅改动其中两处：共享物理投影的零需求梯度，以及正式 Transformer Actor 的 exact GELU 调用方式；另改审计脚本的路径格式。canonical MATD3、H24 对照包、共享 update、共享流水线、评估和物理 YAML 保持任务开始时的内容。

任务开始时全局 `git diff --stat` 包含此前已存在的 `experiments/training_pipeline.py` 修改；本次未改动其初始内容，也未恢复它以制造干净工作区。用户授权整理提交后，该原有修改随既有研究代码一起保留。本次实际新增到两个既有插件发现测试文件的差异仅为三行插件名单。

验收证据保存于更新型冒烟目录下的 `verification/`：`h24_diff.patch`、两个测试文件的增量 patch、初始 Git 状态、保护文件哈希、默认/更新冒烟日志、最终测试日志，以及速度原始 JSON/日志/复现脚本。

授权续作证据另存于 `DUELINGCRITICT-QH24_2026-10-03_22-53-33_seed42/verification/`：最终全量测试日志、三个算法烟测日志、forward 精确比较记录与脚本、第二次计时脚本/JSON/日志，以及独立只读审查记录。

独立只读代码审查未发现 actionable implementation findings；性能和冒烟由本会话实际验证。审查未评价的正式研究结果仍为待完成；不涉及的 inherited checkpoint/resume 行为保持 H24 原实现。

## 全量验证记录

第一阶段运行 `python -m pytest -q`：**608 passed，3 failed，4 warnings**，耗时 155.36 秒。三处插件名单断言已补入新算法并通过；其余三处失败如下。全部 50 项新算法专项测试通过。

以下三项在不加载新算法的单独测试运行中同样失败，且对应共享源文件哈希未变化：

1. `tests/test_audit_saved_service_metrics.py::test_audit_is_read_only_and_marks_missing_historical_evidence`：Windows source_files 路径与测试要求的斜线格式不一致。
2. `tests/test_executed_action_equivalence.py::test_no_demand_region_has_zero_local_gradient`：无负荷需求时，返回的 discharge 梯度为 `[0, 0.5, 0.5]`，测试要求全零。
3. `tests/test_plain_t_power_features.py::test_power_normalizer_buffer_is_nonpersistent_and_moves_with_actor`：普通 Transformer 的 CPU/CUDA 输出差超过现有严格容差。

第一阶段保留三项失败，向用户说明其来源与修改范围。用户回复“允许”后，在保留原有断言和 CPU/CUDA 容差的情况下完成下面的独立共享修复。

1. 审计 source_files 的相对路径使用 `as_posix()`，兼容 Windows manifest 规范。
2. `clear_renewable_first` 在局部剩余负荷为零时显式返回零局部储能放电，去掉 `minimum(0,0)` 对无效请求的虚假 0.5 梯度。正需求和 peer 放电梯度保留。float32/float64、P2P 开/关、随机合法请求及边界组合的 **1,142,916 个输出标量**与原实现精确一致。
3. 正式 Transformer Actor 使用原数学定义的 functional GELU callable，避开 fully-masked CUDA 推理的原生融合路径精度误差。架构、参数量和 state_dict 不变；训练输出和各参数梯度与标准 exact GELU 精确一致，五个 seed 的 CUDA 输出均通过原容差与 float64 参照比较。

三处定向测试 62 passed；第一次修复后全量 620 passed，4 warnings，159.69 秒。新增对照工具 18 项测试通过，并独立审查修复了中断后遗留子进程、过期锁恢复和早期失败日志覆盖问题。最终 `python -m pytest -q`：**638 passed，4 warnings，169.93 秒**。审查复核没有剩余 P1/P2 问题。

共享梯度修复会影响训练，故正式对照改为三个算法各重新训练 seeds 42/43/44，**共九次**；历史 seed-42 结果完整保留，但不复用为本批比较证据。

### 修复后再次核验三个算法的完整流程

均为 3 个训练日、2 个测试日、实际更新已启用，各使用自己的 smoke YAML；formal YAML 不变。

| 算法 | 运行目录名 | 最佳模型测试 | MILP 失败 | Shapley 有效日 | PNG/PDF |
|---|---|---|---:|---:|---|
| Dueling | `DUELINGCRITICT-QH24_2026-10-03_22-53-33_seed42` | 已核验路径和 SHA256 | 0 | 2 | 15 / 15 |
| H24 | `CRITICT-QH24_2026-10-03_22-53-58_seed42` | 已核验路径和 SHA256 | 0 | 2 | 15 / 15 |
| MATD3Q | `MATD3Q-GUIDEDDISPATCH_2026-10-03_22-54-22_seed42` | 已核验路径和 SHA256 | 0 | 2 | 15 / 15 |

目录均为 `outputs/working_runs/<slug>/runs/<上表目录名>/`。三者环境哈希相同；Equal/Shapley、summary 和 saved-result plotting 完成，PNG/PDF 文件名一一对应。只读报告解析已实际读取这三个保存目录，缺少正式训练后段时保持“待完成”。这些工程冒烟记录的 Git 状态为 dirty，明确不当作正式研究结果。

本次验证进程设置 `MKL_THREADING_LAYER=SEQUENTIAL`、`OMP_NUM_THREADS=1`，避免系统 Anaconda 的 NumPy/MKL 与 PyTorch 自带 OpenMP 重复初始化；没有使用 `KMP_DUPLICATE_LIB_OK`，没有修改用户系统安装。pipeline 和速度测量仍使用其配置的线程设置。

## 正式实验与研究报告（设计 §7.2–§8.2）

以下运行尚未启动，不能将冒烟数据填入正式结果表：

| 算法 | seed | 正式运行目录 |
|---|---|---|
| Dueling | 42 | 待完成 |
| Dueling | 43 | 待完成 |
| Dueling | 44 | 待完成 |
| H24 | 42 | 待用户运行 |
| H24 | 43 | 待完成 |
| H24 | 44 | 待完成 |
| MATD3Q | 42 | 待用户运行 |
| MATD3Q | 43 | 待完成 |
| MATD3Q | 44 | 待完成 |

已核实设计指定的两个历史 seed-42 路径存在，各有 55 天保存结果，环境哈希与新算法相同。不过两者的 `git_dirty` 也均为 true，不能把“环境哈希相同”解释为其代码状态已完全可复现。

- H24：`outputs/working_runs/matd3_mlp_transformer_critic_h24_q_guided/runs/CRITICT-QH24_2026-10-03_08-43-59_seed42/`
- MATD3Q：`outputs/working_runs/matd3_dispatch_q_guided/runs/MATD3Q-GUIDEDDISPATCH_2026-09-29_17-35-36_seed42/`

§7.3 正式报告的每算法/seed 利润、idle 增益、严格可行率、广义成本、3-seed 均值±标准差、按日期配对的 bootstrap CI/胜率、三个训练窗口的过程诊断，以及 grid/peak/curtail/comfort/EV 成本分解，均须等待正式数据完整后填写。bootstrap 必须先按 seed 和日期对齐 55 × 3 对样本，并清楚说明重采样单位。

### 用户运行命令

源码保留在本地 `codex/dueling-h24-study` 分支；按用户授权将原有未提交研究源码和本次实施一起固定提交，提交号可用 `git rev-parse HEAD` 查看。交付时工作区干净，没有 push、merge 或 promote。

在仓库根目录运行单次新算法正式实验（默认正式 YAML 已设 5000 episodes、55 test days）：

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '1'
python scripts/run_experiment.py --algorithm matd3_mlp_dueling_critic_h24_q_guided --seed 42 --device cuda --no-dashboard
```

seed 43、44 仅替换 `--seed`。所有训练、validation、最佳 checkpoint 选择、正式测试、MILP、Equal/Shapley、summary 和绘图仍由同一共享入口执行。

若要一次顺序完成九次同提交对照并自动生成 §7.3 报告，可由用户运行：

```powershell
python scripts/run_dueling_h24_study.py --dry-run
python scripts/run_dueling_h24_study.py --study-dir outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/studies/h24_dueling_main
```

第一条只列出计划，不训练。第二条按 seed 42/43/44，每个 seed 依次运行 Dueling、H24、MATD3Q，强制干净且固定的 Git 提交、CUDA、各自正式 YAML、相同环境哈希和结果验收。请为新研究使用新的 study 目录；既有目录不能覆盖。

`study_manifest.json` 保存进度和每次 attempt 的独立日志。每次运行完成并核验后更新 `report.md` 和 `report.json`，覆盖每 seed 结果、样本均值/标准差、55×3 日期配对的分层 bootstrap CI/胜率、三个窗口诊断、成本、速度和参数量。完整九次之前不会输出研究判定。bootstrap 固定 5000 次，先重采样 seed，再重采样该 seed 内日期，成对差值始终保持配对。

中断或失败后可用同一目录加 `--resume`。已核验完成的任务跳过；未完成任务从新 working run 重训，原目录和日志保留，**这是队列重试，不是接续训练 checkpoint**。Ctrl+C 会停止该队列自己启动的训练进程，退出确认后才解锁；断电后恢复仅在旧协调程序和训练进程均不活跃时进行，不会启动重复训练。

```powershell
python scripts/run_dueling_h24_study.py --study-dir outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/studies/h24_dueling_main --resume
python scripts/run_dueling_h24_study.py --study-dir outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/studies/h24_dueling_main --report-only
```

`--report-only` 只读取已有结果，不重新执行训练、模型、环境或 MILP。所有研究结果保持 working 状态，未 promote。短程速度验收 **通过**；研究判定 **待用户完成正式训练**。没有声称“机制成立”“性能成立”或“算法不成立”。
