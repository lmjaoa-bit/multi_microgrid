# 储能日内规划 Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现纯神经网络推理的储能日内规划 Actor，使当前动作由计划第一步产生，并验证专家监督的成本与接入正确性。

**Architecture:** 新建 `matd3_ess_plan_transformer` 插件。保留当前 Dueling H24 的 Critic、DR MLP 路径、TD3 参数与 ESS Q 引导，仅增加 ESS 的 Transformer 计划路径及训练监督。训练专家用紧凑储能优化模型提出候选，再用冻结预测场景的真实物理执行核验；推理与策略评估不调用专家。

**Tech Stack:** 当前项目 Python、PyTorch、NumPy、pandas、PyYAML、highspy、pytest；不安装新增运行依赖。

**Spec:** `docs/design/ess_intraday_plan_proposal_20261008.md`，包括耗时预算补充。

## Global Constraints

- `algorithms/matd3/` 与所有既有算法、正式 YAML、共享物理和奖励不得修改；只在新包、配置、测试和文档中实施。
- 新插件独立注册，YAML `algorithm.name=matd3_ess_plan_transformer`，正常结果在 `outputs/working_runs/matd3_ess_plan_transformer/runs/`。
- Actor 仅接收当前本地观测、公开 peer 摘要与 H24 预测，不添加 peer 私有 SOC；训练专家可使用 CTDE 当前状态及当前可用预测，不能读取真实未来。
- 计划槽 0 是当前动作；预测槽 k 是 t+k；只规划当天剩余时段，终端 SOC 维持 `mode=none`。
- 当前动作来自计划第一步，无独立 ESS 动作头；SOC 与申请功率通过效率及容量确定性关联，实际 SOC 仅由共享环境提交。
- 当前 DR 的输入、MLP 结构、动作语义与更新方式保留；ESS 的新策略可能通过奖励间接影响 DR 学习，不承诺 DR 动作数值不变。
- 关闭规划监督时，网络结构相同且专家调用数为零；标准测试后的联盟 MILP 仍是共享后处理。
- 专家监督的总增量耗时目标为同结构无监督版本的 20% 以内，这是验收目标，不是已有实测结果。新增网络相对旧 MLP 的开销另报。
- 本次交付含本地小样本计时与 1–3 天完整流水线冒烟，不含服务器操作、5000 轮训练或发布保留结果。

## Review Focus

1. 23 点仅有当前槽有效、未来全掩码：输出有限、无越界、不读取次日信息（任务 1）。
2. SOC 边界或放电被清算拒绝：第一步按瞬时范围映射，下一步从实际 SOC 重算（任务 1、3）。
3. 未揭示 EV 会话、未来天气和碳因子：修改实际未来样本不能改变专家输入与标签，训练 RNG 不受专家影响（任务 2）。
4. 优化器失败、超时或候选实际成本恶化：拒收并记录原因，训练继续，不能将不可执行候选称为专家最优（任务 2、3）。
5. 检查点契约不匹配、评估阶段或预算耗尽：拒绝不兼容加载，评估不触发专家，预算暂停不冒充成功监督（任务 3、4）。

## 工作区与文件边界

工作区已有大量修改和删除；执行前记录当前源文件哈希及 Git 状态，不 reset、不 stash、不恢复已删除算法。
建议由本会话直接在当前工作区的独立新包实施；若用户选独立 worktree，必须以当前工作区内容为基准，
不能只从 HEAD 建立而丢失已使用的 Dueling 和共享流水线修订。创建前遵循 using-git-worktrees 技能。
不要提交用户原有变更。设计与计划先作为工作文件保留。

新包文件职责：

| 文件 | 职责 |
|---|---|
| `__init__.py`, `plugin.py` | 导出算法和一个 ALGORITHM_SPEC |
| `algorithm.py` | 初始化、已有 RL 行为、规划监督钩子、检查点契约 |
| `actor.py`, `plan_projection.py` | DR MLP、ESS Transformer、名义电池动力学和第一步动作 |
| `critic.py`, `features.py` | 当前对照实现的本地版本，避免依赖可删除的实验包 |
| `forecast_scenario.py` | 当前信息快照、隔离预测场景、冻结 DR 基线 |
| `plan_teacher.py` | 储能候选优化、物理核验、标签接收 |
| `plan_buffer.py`, `plan_supervision.py` | 独立标签池、轨迹损失、频率与预算 |
| `diagnostics.py` | 保存标签来源、计时、覆盖、拒收和正式测试调用证据 |

## Task 1: 独立插件与由计划产生动作的 Actor

**Files:** Create `algorithms/matd3_ess_plan_transformer/{__init__,plugin,algorithm,actor,features,critic,plan_projection}.py`；Create `configs/algorithms/matd3_ess_plan_transformer.yaml`；Test `tests/test_ess_plan_actor.py`。

**Interfaces:**
- `EssPlanOutput(actions, intents, request_kw, nominal_soc_after, mask)`：actions `[B,2]`，其余 `[B,24]`；正功率放电，负功率充电。
- `decode_nominal_plan(intents, current_soc, mask, battery, delta_t) -> tuple[Tensor, Tensor]`，按名义 SOC 逐槽计算瞬时上限和申请功率；无效槽功率为零、SOC 持续。
- `EssPlanActor.forward_with_plan(observations, collect_diagnostics=False) -> EssPlanOutput`；`forward` 与 `action_logits` 保持现有调用签名，ESS logit 使用计划槽 0 的 logit，DR logit 来自原 MLP 路径。
- `Matd3EssPlanTransformer` 保持当前算法接口；本地复制当前 Dueling 初始化、Critic 与 Q 引导，使用共享诊断更新钩子，不能把新算法条件加到共享代码。

- [x] 写失败测试并运行 `pytest -q tests/test_ess_plan_actor.py`：初始 SOC 0.1 时零放电、SOC 0.9 时零充电；槽 0 intent 等于返回动作 ESS 通道；效率动力学误差 <1e-6；23 点后 23 槽掩码；所有未来掩码或掩码槽含 NaN 仍有限；有效槽 NaN 明确报错；当前 ESS 动作反传能到达预测编码和计划解码，DR 路径不接收规划损失梯度。
- [x] 用规定脚本建脚手架：`python scripts/create_algorithm_variant.py --name matd3_ess_plan_transformer --base matd3_mlp_dueling_critic_h24_q_guided`，然后只在新包替换默认骨架。
- [x] 本地复制现有 Critic/固定特征并记录源哈希；配置从当前对照复制、只增加规划设置和新 slug。ESS 编码 embed=32、heads=4、layers=2、hidden=64、dropout=0；当前语义状态作为 query、有效未来槽及两个 peer 摘要作为 memory；24 个相对时间槽解码 signed intents。
- [x] 运行上述测试；另用 import blocker 验证新包不依赖已删除 Plan-Transformer 或其他可删除 H24 对照包，两个既有插件仍可独立加载。

## Task 2: 因果预测场景与低预算储能专家

**Files:** Create `forecast_scenario.py`, `plan_teacher.py`；Test `tests/test_ess_plan_teacher.py`。

**Interfaces:**
- `capture_forecast_context(env, observations, global_state, metadata) -> ForecastContext`，包含当前状态、观测、可用预测、mask、配置及来源哈希、隔离场景模板。
- `rollout_frozen_baseline(context, actors) -> BaselineTrajectory`：只在隔离预测场景中运行冻结权重，保存 DR raw/实际功率、ESS 基线、成本与初始状态。
- `solve_ess_candidate(context, baseline, time_limit_seconds) -> CandidateResult`：highspy 储能模型，不优化 DR。
- `verify_candidate(context, baseline, candidate) -> PlanLabel`：记录有效性、拒收原因、requested intents、申请/实际功率、名义/实际 SOC、有效 mask、预测来源、基线/候选实际成本、求解状态与总耗时。

- [x] 写失败测试并运行 `pytest -q tests/test_ess_plan_teacher.py`：改真实未来行/未揭示会话后 context hash 相同；当前预测变化后 hash 不同；原环境状态与 RNG 未变；EV 只用已揭示状态和条件先验；未来碳因子沿用当前值；val/test context 明确拒绝。
- [x] 实现物化场景：可参考 Git 中旧 `forecast_scenario.py` 的隔离方式，但放在新包并重新测试，不能自动恢复旧算法。只读当前实际行，未来外生量使用当前预测；清除数据集、未来天气、设备 frame、未揭示 EV 实例等间接信息通路。
- [x] 实现紧凑源流优化：储能容量/效率/SOC/充放电互斥；网电供负荷或充电，RES 供负荷/充电或弃电，ESS 仅供负荷，禁止网电转售、ESS 对 ESS 及出口；固定预测 DR、优先满足负荷的 RES 总量。目标含购电、当前碳因子对应成本、弃电、电池损耗，固定 DR 舒适与峰值成本在比较中保留。
- [x] 说明该优化是候选生成器：自由流分配近似比例清算。不得称为物理最优标签；把候选按实际 SOC 逐步反算 raw intent，沿现有物理规则核验完整预测轨迹。通常一条基线和一条候选，不做 99 次 CEM 搜索。
- [x] 接收规则：约束残差 ≤1e-6、实际成本不高于基线 1e-5、全部有效槽有限；不符合则拒收，不放宽共享物理。保留申请、实际轨迹及误差。监督的名义 SOC 从标签 intent 按任务 1 的模型重算，实际 SOC 用于诊断，不能把申请功率与实际 SOC 当作同一条动力学轨迹。
- [x] 测试两时段低价充高价放的可行改进、零负荷拒绝无用放电、满电/空电边界、网电禁止转售，以及 mock 求解失败/超时/成本恶化；确认只生成训练标签且没有修改共享优化器。

## Task 3: 标签复用、RL 集成及耗时控制

**Files:** Create `plan_buffer.py`, `plan_supervision.py`, `diagnostics.py`；Modify 新包 `algorithm.py`、新 YAML；Test `tests/test_ess_plan_training.py`。

**Interfaces:**
- `PlanBuffer.add(label)`, `sample(batch_size, device) -> PlanBatch`：观测、intent、名义功率/SOC、mask、哈希；独立 RNG，不改变 `replay.add`。
- `plan_supervision_loss(output, batch) -> tuple[Tensor, dict]`：有效槽 intent 与归一化申请功率/SOC 的 SmoothL1，先每条轨迹按有效槽数归一化再平均；全部无效时返回有梯度的零。
- 新算法实现现有 `capture_training_context`、`actor_auxiliary_loss`、`configure_diagnostics`、`set_diagnostic_phase`；已有 Q 引导先执行，再按计划监督频率添加损失，Critic 更新原样保留。
- `TeacherBudget` 记录物化场景、基线、求解、核验、写标签的实际总时间；解算器 time limit 不能代替总标签计时。

- [x] 写失败测试并运行 `pytest -q tests/test_ess_plan_training.py`：监督 loss 对最后有效槽有梯度；仅监督 ESS；关闭监督无捕获、求解、辅助 loss；标签复用不重新求解；预算耗尽跳过新生成且计数准确。
- [x] 默认设置：`lambda_plan=0.1`、label capacity/max_labels=512、aux batch=16、每 10 次 Actor 更新一次监督；第 50 个 episode 起，每 10 个 episode 选一个时刻，按 0/6/12/18 循环覆盖；不能把标签全部用在训练最初几天。
- [x] 专家生成预算暂分配为可比非专家训练时间的 15%，为辅助训练留余量；每标签最大求解时间 1 秒，实际总成本完整计入。按剩余预算和实测单标签成本决定 admission；预算不足延后，不读未来、不改精度换取虚假可行性。
- [x] 保存累计教师耗时、监督次数、覆盖、拒收率、跳过原因和 cache 来源；20% 总目标仍由任务 4 的配对完整计时验证，不能把 15% 生成限制当作总目标已达成。
- [x] 配置短 smoke 从 episode 0、每 episode 轮换采样，其他训练规模缩小；不绕过物理、标签验证或使用测试样本。若预算内零有效监督，smoke 不通过，先定位。
- [x] 检查点绑定结构、输入 schema、物理/奖励合同、规划和专家版本；加载前完整验证，再一次性写入网络。第一版仅支持推理检查点，显式拒绝训练恢复，标签池和专家 RNG 未恢复不能假称 resume。
- [x] 测试不兼容/非有限检查点不改变内存权重；调用 select_actions/val/test 的专家计数为零；实际功率削减后下一次规划从新观测 SOC 开始；超时/拒收不污染 replay 或训练环境。

## Task 4: 配对运行时计量与完整共享流水线验收

**Files:** Create `scripts/benchmark_ess_plan_training.py`, `configs/smoke/matd3_ess_plan_transformer.yaml`；Test `tests/test_ess_plan_pipeline.py`；Modify 新包 `diagnostics.py`、新算法使用文档。

**Interfaces:**
- `benchmark_case(config, seed, episodes, output_directory) -> dict`：只用 train split；相同网络/种子/线程/批量，监督 on/off 交替运行，包含生成、辅助反传和写标签总耗时；保存 JSON 原始计时与配置哈希。
- Benchmark 输出 `on_seconds`, `off_seconds`, `extra_fraction`, `teacher_seconds`, `label_count`, `supervision_updates`, `reject_count`，以及旧 MLP 对照的独立计时。CUDA 计时须同步，不能混用不同硬件结果。

- [x] 测试计时账目、专家训练/测试阶段计数、算法 slug/config 与正式测试 best checkpoint 契约。
- [x] 运行计时试验：固定输入及更新次数、先预热，至少 3 组配对；让样本实际触发标签与辅助更新，单独报告冷启动和稳态。先检查专家可行/收益、有效覆盖，再看耗时；若总增量 >20%，调整采样频率/标签复用并重测，不能把目标写成已实测。计时只证明本地样本，不保证完整 5000 轮。
- [x] 运行 `pytest -q`。实施前记录已有失败；新增失败必须定位，不能修改既有测试来隐藏。
- [x] 运行共享 smoke：`python scripts/run_experiment.py --algorithm matd3_ess_plan_transformer --algorithm-config configs/smoke/matd3_ess_plan_transformer.yaml --validation-days 2 --device cpu`；新 smoke 设置 3 训练日、2 测试日、每轮 validation、update 开启；仅生成存在季节的图。
- [x] 核对 `best_model.pt` 参与正式测试、Equal/Shapley、日报/摘要/plot_manifest 完成；标准 PNG/PDF 00、01、01b 与可用季节齐全；全部非图证据位于 artifacts、专家调用在策略测试为零。
- [x] 保存验收报告：精确命令、源哈希、运行路径、性能量级/未验证项、教师监督覆盖；检查 canonical MATD3 和所有既有算法/config/共享流程相对实施前哈希不变。

## 计划自检与后续边界

本计划覆盖动作硬连接、整段训练信号、真实与名义 SOC 区分、预测因果性、标签物理核验、预算和可比计时。
专家模型是近似候选生成器，物理回放后的改进才支持接收标签；算法收益需要后续同预算多种子实验，
不能用 smoke、计划误差或 52.83/天的全信息审查结果代替。
实施过程中若紧凑专家在 20% 预算内不能提供足够标签，报告失败并重新审查路线，不启用旧 CEM。

**建议执行方式：** 本会话直接执行任务 1–4，先 Actor 与专家单元验证，再训练接入、计时和共享 smoke。
实施前需要用户审查这份计划并选择执行方式；本文件没有启动实现、依赖安装或训练。
