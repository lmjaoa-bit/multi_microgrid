# Plan-Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增独立的 MATD3 Plan-Transformer，用当前预测生成可执行的 SOC/ESS 计划标签训练共享 cross-attention，同时保持现有两维动作和物理实验合同。

**Architecture:** 三个独立 Actor 以 NOW query 对 H24 和 peer memory 做两层 cross-attention，共享表征连接动作头与辅助计划头。中心化教师在隔离的预测场景副本中用 CEM 优化联合原始动作，复用现有 `env.step`，将可执行轨迹存入独立标注池。共享流程仅增加可选公开协议调用；正式测试不求解教师，解释图只读取保存的结果。

**Tech Stack:** Python、PyTorch、NumPy、pandas、PyYAML、pytest、matplotlib，以及现有 HiGHS coalition 评估；无新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-04-plan-transformer-design.md`（用户已确认）。

## Global Constraints

- slug=`matd3_plan_transformer`，display name=`Plan-Transformer`；自己的 plugin 和 YAML，按名称加载。
- canonical `algorithms/matd3/`、现有算法网络、`configs/cooperative.yaml`、reward 和 settlement 不修改；`matd3` 默认仍选择 `matd3_dispatch`。
- 对外动作 `[ESS intent, DR intent]`，范围 `[-1,1]`；ESS 计划正值放电、负值充电，SOC 为步后状态。
- Actor H24、embed=32、heads=4、layers=2、hidden=64、dropout=0；gamma=1，actor 最大梯度范数=1.0。
- 教师仅 train，当前联合状态可用，但未知未来数据/EV 会话不可用；未来 carbon 使用当前值 persistence。
- CEM population=32、elite=8、iterations=3、initial_std=0.5、min_std=0.05、update_rate=0.5；capture hours=[0,6,12,18]，cache/pool capacity=2048。
- lambda_plan=0.1、lambda_dyn=0.01、lambda_first=0.05，auxiliary batch=32；权重调节只使用 train/validation。
- 首版沿用现有不支持该变体 resume 时明确拒绝的流程，不实现可恢复协议，不宣称训练状态可精确恢复。
- 标准输出为 `outputs/working_runs/<slug>/runs/<run_id>/`；标准 figures 保留 15 PNG + 15 PDF，额外图放 `figures/plan_diagnostics/`。
- 本计划执行只包括开发、单元验证和 1–3 日 smoke；不自动执行多 seed 完整训练，不晋升 release，不删除结果或提交其他未跟踪工作。

## Review Focus

1. 日末仍有一条有效计划，但所有未来 token 都无效：动作有限，当前 SOC 边界正确，两个 mask 不错位（Task 3）。
2. 教师副本中的设备 frame、profile cache、future forecast 行或隐藏 EV session 残留：改变真实未来不得改变冻结场景、标签或成本（Tasks 1–2）。
3. warmup 不调用 `select_actions`，而验证重新 reset 环境：context 必须在 warmup 动作前捕获，不能串到其他 transition 或 phase（Tasks 2、5）。
4. 非兼容 checkpoint 或不可恢复 resume：必须在参数/输出被改写前拒绝，并保持已存在模型状态和研究结果（Tasks 4–5）。
5. 正式测试中途失败或绘图重复运行：诊断 phase 在 finally 中关闭，文件来自唯一测试 forward，重绘不触发模型/环境/教师（Task 6）。

## File Map

| 文件 | 职责 |
|---|---|
| `algorithms/matd3_plan_transformer/forecast_scenario.py` | 明确的因果 PlanningContext、EV 名义预测、独立场景构造和副本恢复 |
| `algorithms/matd3_plan_transformer/plan_teacher.py` | CEM、轨迹物理审计、标签及有界缓存 |
| `algorithms/matd3_plan_transformer/replay_buffer.py` | 原接口 RL Replay、pending context 关联、独立辅助池及采样 |
| `algorithms/matd3_plan_transformer/actor.py` | NOW cross-attention、两维动作、SOC/ESS 辅助输出与 attention |
| `algorithms/matd3_plan_transformer/plan_losses.py` | masked 监督、SOC 动力学和实际首步一致性 |
| `algorithms/matd3_plan_transformer/algorithm.py` | 控制器、更新、配置/合同、checkpoint、公开诊断访问 |
| `algorithms/matd3_plan_transformer/diagnostics.py` | 教师证据和同一次正式测试 forward 的诊断文件 |
| `algorithms/matd3_plan_transformer/__init__.py`, `plugin.py` | 包和独立插件 |
| `configs/algorithms/matd3_plan_transformer.yaml`, `configs/smoke/matd3_plan_transformer.yaml` | 正式默认参数和确实更新的短 smoke |
| `algorithms/base.py` | 两个可选公开协议，不改变基本算法接口 |
| `experiments/training_pipeline.py` | 训练前 context 调用、run 内诊断路径配置 |
| `evaluation/evaluate_matd3.py` | 当前中立评估实现中的正式测试诊断启停；`evaluate_algorithm.py` 保持中立入口 |
| `evaluation/plotting.py` | 保存文件驱动的额外计划与 attention 图 |
| `diagnostics/plan_transformer_probe.py`, `scripts/probe_plan_transformer.py` | 独立预测扰动诊断入口，非正式测试，不使用环境 rollout |
| `tests/test_plan_teacher.py`, `tests/test_plan_transformer_actor.py`, `tests/test_plan_transformer_pipeline.py`, `tests/test_plan_transformer_diagnostics.py` | 按职责定义局部 fixtures 与验收测试 |
| `docs/ALGORITHM_DEVELOPMENT.md` | 新变体运行方式、教师口径和 smoke 限制 |

`plan_losses.py` 和 `diagnostics.py` 是新包内部的职责拆分；不移动共享物理/评估代码。当前评估中立入口实际转发到 `evaluate_matd3.py`，因此只在现有实现里增加算法中立协议调用，不复制 evaluator。

## Execution Preflight

这些步骤在用户确认计划/执行方式后、Task 1 前执行，不是 Task 8 才执行。

- [ ] 使用 `using-git-worktrees` 检测是否已经隔离，遵守用户的工作区选择。若创建 App 工作区，指定包含本规格和计划的当前分支/提交，不能默认从远端 main 开始；只复用已确认适用的 attachment。
- [ ] 记录实际 base commit、working directory、当前差异及四项既有未跟踪文件；保存受保护目录的 Git diff/文件 hash。计划中的 Git 提交只包含自己的指定路径。
- [ ] 运行一次基线 `pytest -q` 并记录实际通过/失败；既有失败与新增失败分开。依赖可用时不重复安装，只按需要补 requirements 中已有依赖。

## Task 1: 构造不含真实未来的预测场景

**Files:** Create scaffold、新包 `forecast_scenario.py`；Test `tests/test_plan_teacher.py`。

**Interfaces:**
- `capture_planning_context(env, observations, global_state, metadata) -> PlanningContext`：当前 t、agent order、局部 observations、state、timestamp、设备当前可见状态、history/peak、冻结预测、环境/归一化 hash、provenance。
- `PlanningContext.context_hash: str`，`PlanningContext.forecast_hash: str`，`PlanningContext.plan_mask: np.ndarray[bool,24]`；数组复制后禁止外部修改。
- `build_forecast_environment(context: PlanningContext) -> MultiMicrogridEnv`：每次返回独立、从当前 t 开始的预测场景副本，只复用现有物理类；`PlanningContext` 不保存原 env 引用、完整未来 frame 或未知会话。
- `PlanningContextRejected(ValueError)`：无效预测、无可构造 EV 条件预测等可审计拒绝；程序错误不吞掉成标签拒绝。

- [ ] **Step 1:** 为 capture purity、冻结场景、日末索引、revealed/hidden EV、场景全部缓存入口定义失败测试。局部 `case` fixture 创建 train env 并以确定 seed reset，推进到指定 t。关键断言：
  ```python
  assert context.plan_mask.sum() == 24 - env.t
  assert context.context_hash == capture_after_hidden_future_corruption.context_hash
  assert teacher_env.t == env.t
  assert teacher_env.daily_load_peak_kw == env.daily_load_peak_kw
  assert original_environment_fingerprint_after == original_environment_fingerprint_before
  ```
- [ ] **Step 2:** `pytest -q tests/test_plan_teacher.py -k 'context or forecast_environment'`，确认失败来自缺少模块/接口。
- [ ] **Step 3:** 执行 `python scripts/create_algorithm_variant.py --name matd3_plan_transformer`。只在新包实现上述 context 和副本接口。设备 inventory 从当前状态显式白名单构造，未来外生量来自 t 时 forecast；不 deepcopy 原环境后保留其隐藏对象。父模型临时 scaffold 不作运行完成声明。
- [ ] **Step 4:** 绑定副本的 exogenous arrays、episode frame、forecast provider、HVAC/EV frame 与 base-load 缓存、profile cache 到冻结预测输入；未来 carbon persistence、EV 加权中位数/条件化规则按规格落实。revealed EV 当前已交付量/剩余量和已发生事件保持，hidden group 只使用名义会话。
- [ ] **Step 5:** 运行 Step 2 测试，并逐项检查温度、GHI、异构 tariff、silo、HVAC 双温度、EV 剩余需求、历史 peak 和协调历史恢复正确。有效 forecast 内 NaN/Inf 拒绝；物理合同不重写。
- [ ] **Step 6:** 提交仅包含本 Task 的新包 scaffold、context 和测试，message=`feat: add causal forecast planning contexts`。

## Task 2: CEM 教师和辅助标签池

**Files:** Create `plan_teacher.py`, `replay_buffer.py`；Extend `tests/test_plan_teacher.py`。

**Interfaces:**
- `PlanTeacher(settings: Mapping, seed: int)`；`solve(context: PlanningContext) -> PlanLabel`、`solve_calls: int`、`cache_size: int`。
- `PlanLabel`：valid/rejection_reason、agent order、context/forecast/contract hash、原始 obs/state/t/mask、`soc_after` 和 `ess_net_kw`（逐 agent `[24]`）、原始联合 actions（`[24,3,2]`）、DR 轨迹、基准/最优可行成本、seed、预算、耗时。无效槽存 0，由 mask 决定有效性。
- `PlanReplayBuffer(capacity, auxiliary_capacity, seed, auxiliary_seed)`：基本 `add/sample/__len__` 沿用 `FastReplayBuffer`，`bind_context(context: PlanningContext, label: PlanLabel | None) -> None`，`sample_auxiliary(batch_size: int, agent_ids: list[str], device: torch.device) -> AuxiliaryBatch | None`。
- `AuxiliaryBatch`：observations、global_state、mask、soc_after、ess_net_kw 的 tensors 和 context hashes；空池返回 None，非空按 `min(requested,size)` 无放回采样。

- [ ] **Step 1:** 写可复现 CEM、真实未来污染无效、基准成本上限、预算/cache invalidation、物理轨迹、transition 关联和原 Replay 不变测试。population/elite/iterations/capacity 必须为正整数、elite 不大于 population、std 有限正数、update_rate 在 `(0,1]`，无效配置初始化时拒绝。无噪声 ESS=0 轨迹用于独立核对 SOC；CEM 更好的候选用于核对 best retention。
  ```python
  assert label.valid
  assert label.best_cost <= min(label.baseline_costs) + 1e-7
  np.testing.assert_allclose(label.soc_after['mg1'][label.mask], replayed_soc)
  assert same_seed_label.context_hash == label.context_hash
  assert teacher.solve_calls == calls_before_cache_hit
  ```
- [ ] **Step 2:** `pytest -q tests/test_plan_teacher.py -k 'teacher or auxiliary or association'`，确认新增行为先失败。
- [ ] **Step 3:** 实现 CEM 的 clipped diagonal Gaussian、固定三个 DR 基准、排序 tie-break、独立 RNG 与最好候选保留；成本取从 t 起的 Reward V2 增量。每候选重新恢复同一个预测 context；硬物理不可行/非有限候选拒绝，不把软服务成本当硬约束。
- [ ] **Step 4:** 实现按完整 context+教师合同 hash 的缓存及预算；标签逐时保存实际 ESS/SOC/DR。辅助池与 RL Replay 分离，binding 只消费一次；obs/state hash 不匹配直接报错，不能给错误 transition 标注。辅助开关全关时不求解，不消费辅助 RNG。
- [ ] **Step 5:** `pytest -q tests/test_plan_teacher.py`，检查原 env/RNG 完全不变，hidden future 修改后同 seed 标签和成本一致；teacher 无候选可返回可审计拒绝且 RL transition 仍可加入。
- [ ] **Step 6:** 提交本 Task，message=`feat: add feasible CEM plan labels and auxiliary replay`。

## Task 3: H24 Actor 和三项辅助损失

**Files:** Create `actor.py`, `plan_losses.py`；Test `tests/test_plan_transformer_actor.py`。

**Interfaces:**
- `PlanTransformerActor(schema, agent_ids, agent_id, mg_config, power_normalizers, network_config)`；`forward(obs: Tensor) -> Tensor[B,2]`、`forward_with_plan(obs: Tensor, collect_attention: bool=False) -> PlanOutput`。
- `PlanOutput`：`actions[B,2]`、`soc_after[B,24]`、`ess_net_kw[B,24]`、`plan_mask[B,24]`、`attention[B,2,4,26] | None`（layer/head/key，不平均 heads），`forecast_mask[B,24]`。
- `plan_loss_terms(output: PlanOutput, batch: AuxiliaryBatch, agent_id: str, projected_plan: Mapping[str,Tensor], battery_config: Mapping, delta_t: float) -> LossTerms`；`LossTerms.plan/dynamics/first` 为 scalar tensors，另有 detached error metrics。

- [ ] **Step 1:** 写 token 顺序/功率归一化、base-load 不丢失、t=0/23 mask、全部 future 无效、无效格 NaN/Inf、attention 梯度和首步实际投影测试。
  ```python
  assert actor(obs).shape == (2, 2)
  assert result.soc_after.shape == (2, 24)
  assert result.plan_mask[0].sum() == 24
  assert result.plan_mask[1].sum() == 1
  assert result.forecast_mask[1].sum() == 0
  assert torch.isfinite(result.actions).all()
  assert all(layer_grad_norm > 0 for layer_grad_norm in auxiliary_attention_grad_norms)
  ```
- [ ] **Step 2:** `pytest -q tests/test_plan_transformer_actor.py`，确认失败来自新架构/损失缺失。
- [ ] **Step 3:** 按规格实现 NOW/local DR encoder、24 个 future memory、2 个 peer memory、固定 peer/type/time 编码和两层 pre-LN cross-attention。先屏蔽无效输入后编码；验证有效输入和 schema 必需字段。所有输出头共用最终 NOW，默认尺寸严格取 Global Constraints。
- [ ] **Step 4:** 实现 masked SmoothL1、正负 ESS 功率的不同效率关系和实际首步一致性，逐样本有效长度归一化；t=23 的 first loss 仍生效。损失读取 projector 的 `actual_charge_kw`/`actual_discharge_kw`，不用 requested power。构造一个 requested 与 actual 不等的共享供需样本固定这个区别。
- [ ] **Step 5:** 重跑 Step 2，验证有效预测改变可影响 plan/action；每个 cross-attention layer 的辅助梯度非零有限；无效 padding 值改变不影响 action。
- [ ] **Step 6:** 提交本 Task，message=`feat: add cross-attention plan actor and auxiliary losses`。

## Task 4: 算法控制器、独立配置和 checkpoint

**Files:** Modify 新包 `algorithm.py`, `plugin.py`, `__init__.py`, 正式 YAML；Create smoke YAML；Test `tests/test_plan_transformer_pipeline.py`。

**Interfaces:**
- `Matd3PlanTransformer.from_env(env, config, device='cpu', seed=None)`，共享 `select_actions/update/save/load/replay` 原合同。
- `capture_training_context(env, observations, global_state, metadata) -> None`：hours/budget/split/开关判断、capture/solve/bind；可审计拒绝记录，程序错误传播。
- `inspect_policy(observations) -> dict[str, PlanOutput]`：无教师、无诊断写入、无噪声的公开检查入口，只用于单独诊断。
- 后续 Task 6 实现 `configure_diagnostics(directory: Path, metadata: Mapping) -> None` 与 `set_diagnostic_phase(phase: str | None, metadata: Mapping) -> None`。

- [ ] **Step 1:** 写注册/环境一致、shared API、delayed 辅助更新、全关消融、正常 checkpoint round trip 和 mismatch 原子拒绝测试。测试先保存参数 hash，再改变 checkpoint 的 slug、schema、normalizers、mask/index/符号或 reward/action 合同，断言异常后参数 hash 不变。
- [ ] **Step 2:** `pytest -q tests/test_plan_transformer_pipeline.py -k 'registry or update or checkpoint or ablation'`，确认新控制器行为先失败。
- [ ] **Step 3:** 在新包实现投影 MATD3 控制器，复用通用 Critic/projector/runtime tools，保留 TD 语义；不要继承普通 T 限制 H8 的构造方法。构造独立 Actor/target、optimizers 和辅助池；不安装会覆盖 update/select_actions 的现有诊断 monkeypatch。
- [ ] **Step 4:** delayed Actor 更新只优化 active Actor；其他 Actor detach、Critic freeze/finally restore。RL batch 与独立 auxiliary batch 不混淆；计算三损失并按权重合并，记录每项及 RL/辅助共享注意力梯度诊断，诊断额外梯度评估不能改变实际 optimizer 更新。
- [ ] **Step 5:** 实现完整新 checkpoint 合同预验证，包括 state_dict keys/shapes，之后一次性 load；推理 load 不要求存在标签池。正式 YAML 保留训练/验证设置和本 slug；smoke 为 batch=8、warmup=8、3 个训练日、2 个验证/测试日、teacher population=4/elite=2/iterations=1/max_labels=12/aux batch=4，仍保留两层 Actor。
- [ ] **Step 6:** 重跑 Step 2；在测试中至少有一次 `auxiliary_label_count>0` 和 `actor_auxiliary_update_count>0`，loss/梯度均 finite。确认关辅助后三权重=0，教师调用为 0，Actor 参数结构和动作接口一致。
- [ ] **Step 7:** 提交本 Task，message=`feat: integrate isolated MATD3 Plan-Transformer controller`。

## Task 5: 共享流程的可选 hooks

**Files:** Modify `algorithms/base.py`, `experiments/training_pipeline.py`, `evaluation/evaluate_matd3.py`；Extend `tests/test_plan_transformer_pipeline.py`。

**Interfaces:**
- `TrainingContextConsumerProtocol.capture_training_context(env: Any, observations: Mapping, global_state: np.ndarray, metadata: Mapping[str,Any]) -> None`，runtime-checkable。
- `AlgorithmDiagnosticsProtocol.configure_diagnostics(directory: Path, metadata: Mapping[str,Any]) -> None` / `set_diagnostic_phase(phase: str | None, metadata: Mapping[str,Any]) -> None`，runtime-checkable。
- 不给 `MultiAgentAlgorithm` 增加必需方法，不修改基本 `replay.add` 参数。

- [ ] **Step 1:** 用无 hook 的 fake algorithm、记录 hook 的 fake algorithm 和新 controller，测试 warmup 之前的调用顺序、上下文消费、validation/learning curve 无 teacher，以及 evaluator 退出异常 finally 停止 phase。固定 before hook 观察 t、state，after `replay.add` 关联同一 t；resume 使用已有拒绝错误。
- [ ] **Step 2:** `pytest -q tests/test_plan_transformer_pipeline.py -k 'hook or warmup or phase or resume' tests/test_training_pipeline_recovery.py`，新增 hook 测试先失败。
- [ ] **Step 3:** 在训练算法创建且有 run directory 后对实现诊断协议者配置 `<run>/auxiliary_plan`；每次动作选择前用协议调用 context，metadata 带 episode index、total steps、reset day。不得读取 private actor/critic；没有协议的算法保持原路径。
- [ ] **Step 4:** evaluator 现有 `spec.create/load` 后配置该次正式测试输出和 checkpoint；仅包围算法的正式测试 policy rollout 启动 phase，metadata 含按执行顺序的 days/seeds。idle/其他 baseline、MILP、validation 不进入 phase；finally 停用。`evaluate_algorithm.py` 中立转发入口不复制实现。
- [ ] **Step 5:** 重跑 Step 2 和 `pytest -q tests/test_two_algorithm_smoke.py tests/test_algorithm_registry_and_layout.py`；确认 MATD3/普通 T 基本接口正常。resume 首版明确拒绝本变体，不新增伪恢复状态。
- [ ] **Step 6:** 提交本 Task，message=`feat: add optional training-context and evaluation-diagnostics hooks`。

## Task 6: 保存教师和测试证据，绘制解释图

**Files:** Create 新包 `diagnostics.py`；Modify 新包 `algorithm.py`, `evaluation/plotting.py`；Test `tests/test_plan_transformer_diagnostics.py`。

**Interfaces:**
- `PlanDiagnosticsRecorder(directory: Path, metadata: Mapping)`；`record_label(label: PlanLabel) -> None`、`set_phase(phase: str | None, metadata: Mapping) -> None`、`record_policy(outputs: Mapping[str,PlanOutput], observations: Mapping) -> None`。
- 版本化 `auxiliary_plan/manifest.json`、`teacher_labels.jsonl`、`formal_test_plan_attention.jsonl`；manifest 含 observation schemas/normalizers 和诊断版本，每条测试记录含 checkpoint/run、day/seed/t、agent、原始局部 observation、当前动作、计划、两个 mask、peer order、真实 token 时间、layer/head attention。保存原始 observation 供 Task 7 使用，不能靠反推图中净负荷重建输入。
- `generate_plan_diagnostic_figures(run_dir: str | Path, formats=('png','pdf')) -> list[Path]` 加入现有 `evaluation/plotting.py`，由 `generate_figures` 在保存诊断存在时调用；输出在子目录，不改变标准 15 图。

- [ ] **Step 1:** 写每个测试 step 仅一次 Actor forward、开启记录前后动作相等、teacher 调用为 0、phase 异常关闭、文件身份和日末 token 时间测试。绘图测试禁止构造 env/algorithm，使用独立保存文件 fixture；第二次重绘结果数量和来源一致。
  ```python
  assert forward_calls_per_agent == 24 * test_days
  assert test_teacher_calls == 0
  assert rows_count == 24 * test_days * 3
  assert len(list((run / 'figures').glob('*.png'))) == 15
  assert len(list((run / 'figures').glob('*.pdf'))) == 15
  ```
- [ ] **Step 2:** `pytest -q tests/test_plan_transformer_diagnostics.py`，确认诊断文件/图行为先失败。
- [ ] **Step 3:** 实现 recorder 及控制器公开协议方法；select_actions 一次 `forward_with_plan` 取动作并按 phase 可选写入。recording RNG 与动作探索无关，validation 无记录；重复 phase/run 文件身份冲突明确拒绝。标签拒绝原因同样保存；无有效标签时 summary 标出且验收失败。
- [ ] **Step 4:** 保存文件 plotter 对每微网绘制各层/各 head attention（屏蔽格留白）以及实际轨迹与辅助 SOC/ESS 对照；横轴使用 t 和 token 相对/绝对小时，不平均掉不同 head。校验 checkpoint 和 test days 一致；缺少额外诊断的旧结果维持标准绘图。
- [ ] **Step 5:** 重跑 Step 2 和 `pytest -q tests/test_training_convergence_plot.py`。日末图的唯一有效 plan 槽能显示；formal test 实际 SOC 与预测 SOC 清楚区分；绘图完全只读 saved input。
- [ ] **Step 6:** 提交本 Task，message=`feat: save and plot plan and attention diagnostics`。

## Task 7: 可复现的关键时段预测扰动诊断

**Files:** Create `diagnostics/plan_transformer_probe.py`, `scripts/probe_plan_transformer.py`；Extend `tests/test_plan_transformer_diagnostics.py`；Modify `docs/ALGORITHM_DEVELOPMENT.md`。

**Interfaces:**
- `probe_forecast_sensitivity(model, observations: Mapping[str,np.ndarray], cases: Sequence[Mapping]) -> list[dict]`：通过公开 `inspect_policy` 比较 base/perturbed；纯 inference，无 teacher/env rollout/写入源 run。
- CLI `python scripts/probe_plan_transformer.py --run <completed_run> --output <new_study_directory>`：从 snapshot/best checkpoint/已保存正式测试观测构造输入，输出新的诊断研究；只接受本 slug 和独立、未存在输出目录。

- [ ] **Step 1:** 测试只改变有效 future token：相邻小时交换本地+peer 负荷/PV/wind/温度内容保持 mask/time编码、电价独立交换，和午间 surplus 增量（增 PV 10% 额定容量、保持基荷/电价不变）。assert 当前 state、当前动作输入、源 run hash、无效 slots 保持原样；不要求未训练模型有固定方向收益。
- [ ] **Step 2:** `pytest -q tests/test_plan_transformer_diagnostics.py -k 'probe or perturbation'`，新增行为先失败。
- [ ] **Step 3:** 实现只读 checkpoint 和观测的诊断 CLI，输出动作/计划/attention 差值和干预清单；报告标注 diagnostic-only、非因果证明和非真实 dispatch 收益，不混进 test_summary。
- [ ] **Step 4:** 更新开发文档：命令、因果教师合同、CEM 可行解口径、推理不依赖教师、resume 首版限制、无辅助消融和多 seed 效果研究要求；无辅助配置通过临时 YAML 生成并保证同结构/环境，不新增另一个算法 slug。
- [ ] **Step 5:** 重跑 Step 2；使用 Task 8 smoke 完成后的小输入验证 CLI 生成单独 study 且源 run 不变，不启动完整效果研究。
- [ ] **Step 6:** 提交本 Task，message=`feat: add saved-observation plan sensitivity probe`。

## Task 8: 全套验证和短流程验收

**Files:** Test 新增所有测试；Review 变更范围；完成文档中的验收记录，不新增镜像实现的测试。

**Interfaces:** 前述 plugin、正式/smoke YAML、共享单一 `run_experiment.py`、saved-result plotter 和 probe。

- [ ] **Step 1:** 复核 Execution Preflight 中记录的 base commit/工作区和基线测试；确认 Task 1–7 的验收与提交记录均存在，已知失败与新增失败分开，不重复安装依赖或运行基线测试。
- [ ] **Step 2:** 所有变更完成后运行 `pytest -q`，必须实际确认输出。失败先用 systematic-debugging 定位；修复后只重跑关联检查，再完成全套必要验证。
- [ ] **Step 3:** 运行完整 CPU smoke：
  ```powershell
  python scripts/run_experiment.py --algorithm matd3_plan_transformer --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_plan_transformer.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
  ```
  必须成功结束；不使用 `--no-milp`/`--no-validation`。确认实际辅助更新大于 0、标签有效大于 0、loss/梯度 finite、验证/测试教师为 0，并保存耗时；不把 smoke 描述为性能提升。
- [ ] **Step 4:** 检查 `latest_run.json` 和 snapshot：选中本 slug/本 smoke YAML/相同环境，结果在正确 working run；`best_model.pt` 是正式测试 checkpoint。检查两个测试日完整 24 小时、七 coalition 值、Equal/Shapley conservation、summary 和 plotting success。
- [ ] **Step 5:** 按 standard figures 根目录确认 15 PNG + 15 PDF，额外图在指定子目录；诊断行数 `2*24*3`，checkpoint/day/seed 对齐。单独重绘保存结果一次，断言既有 checkpoint/JSON/CSV hash 不变、没有环境/策略/优化调用。
- [ ] **Step 6:** 执行 Task 7 的小型保存观测 probe，并检查输入 run hash 不变。无辅助开关用单元/短更新验证 teacher=0；不自动再跑一个完整 smoke，除非出现相关未决失败。
- [ ] **Step 7:** 对起点比较 `git diff --name-only <base> -- algorithms/matd3 configs/cooperative.yaml configs/algorithms/matd3.yaml algorithms/matd3_dispatch algorithms/matd3_hierarchical_transformer` 为空；用户原有未跟踪工作仍保留。按 requesting-code-review 流程做最终审查，修复实质问题后执行相关验证；使用 verification-before-completion，按 finishing-a-development-branch 和用户授权处理集成。
- [ ] **Step 8:** 提交最终验收文档（只含自己的改动），报告测试数量、smoke 路径、模型/结算/图和辅助监督证据、教师耗时及未做完整效果研究的范围；不得晋升 dirty/未知 release 或删除旧结果。

## Execution Handoff

文档已按规格自检；本计划需要用户评审并选择执行方式之后实施。推荐在本会话由主代理逐项执行，因为教师上下文、辅助 batch、实际投影和诊断身份共用接口，串行实现更便于保持一致。独立工作区或当前目录由用户选择；当前目录有四项既有未跟踪文件，执行者不覆盖/提交它们。

Native 执行使用 `superpowers:executing-plans`，Subagent-driven 执行使用 `superpowers:subagent-driven-development`。选定后沿用该方式，不按 Task 反复询问。
