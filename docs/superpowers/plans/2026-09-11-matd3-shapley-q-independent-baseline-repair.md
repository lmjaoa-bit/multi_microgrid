# MATD3-ShapleyQ Independent Baseline 修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐任务实施本计划。所有步骤均使用 checkbox（`- [ ]`）跟踪。

**目标：** 在不改动 canonical MATD3 的前提下，完成 `matd3_shapley_q` 的 Independent Baseline、Cooperation Surplus Bellman target、真实验收测试、日志及正式测试输出。

**架构：** 保留现有 3 Actor、3 组 Twin Marginal-Contribution Critic 与 exact Shapley 结构。固定 Independent Baseline 在同场景的无合作影子环境中运行，Replay 保存 Grand/Independent/Surplus 三类 reward，Critic 只学习 Cooperation Surplus Q；共享流水线通过公开的可选 episode hook 收集算法专属信息。

**技术栈：** Python 3.13、PyTorch、NumPy、PyYAML、pytest、现有 `MultiMicrogridEnv`、JSON/CSV、TensorBoard。

**规格：** `docs/superpowers/specs/2026-09-11-matd3-shapley-q-independent-baseline-repair-design.md`

## 全局约束

- 不修改 `algorithms/matd3/` 下 Actor、Critic、Replay Buffer、MATD3 update、checkpoint 或公共行为。
- `matd3_shapley_q` 必须继续是独立 plugin，配置文件为 `configs/algorithms/matd3_shapley_q.yaml`。
- 环境继续输出统一 `team_reward`，不得新增即时 Shapley reward shaping。
- 保留 3 个 Actor、3 组 centralized Twin Marginal-Contribution Critic 和 exact Shapley 权重 `1/3, 1/6, 1/6, 1/3`。
- `Q_coop = Q_grand - Q_ind`，`sum_i Q_phi_i = Q_coop`。
- `SOC_start = SOC_end = 0.1`，终端 SOC 使用 hard constraint，不加入 terminal SOC reward、bonus 或 economic value。
- Validation 继续仅按 Grand Coalition team profit 选择 `best_model.pt`。
- 保留当前工作区中 Gemini 的未提交修改，不回滚、不覆盖无关用户改动。
- 每个生产修改必须先有会因目标行为缺失而失败的测试。

---

## 文件职责图

- `algorithms/matd3_shapley_q/coalition.py`：纯数学的 coalition normalization、exact Shapley 与 predecessor metadata。
- `algorithms/matd3_shapley_q/coalition_projector.py`：只屏蔽跨微网合作通道，保留本地调度。
- `algorithms/matd3_shapley_q/independent_baseline.py`：固定 baseline 策略、同场景影子环境、baseline trajectory 与 return-to-go。
- `algorithms/matd3_shapley_q/replay_buffer.py`：保存并校验 Grand/Independent/Surplus transition。
- `algorithms/matd3_shapley_q/algorithm.py`：Shapley-Q target、Actor/Critic update、diagnostics 与 checkpoint。
- `envs/multi_microgrid_env.py`：公开最近一次 reset 的场景参数，不改变环境 reward/physics。
- `experiments/training_pipeline.py`：调用公开 algorithm hook，保存 episode 与 summary 指标。
- `evaluation/evaluate_matd3.py`：正式测试中对同一日期/seed 运行固定 baseline 并输出 paired fields。
- `utils/training_dashboard.py`：显示 Grand/Independent/Surplus 与 Shapley efficiency。
- `scripts/export_runs_to_tensorboard.py`：导出要求的 reward/Q tags。
- `configs/cooperative.yaml`：hard terminal SOC 的权威环境配置及正确注释。
- `configs/algorithms/matd3_shapley_q.yaml`：baseline mode 与 Shapley-Q diagnostics。
- `tests/test_matd3_shapley_q.py`：算法数学、physics、gradient、scenario、target 和 checkpoint 测试。
- `tests/test_metrics_recording.py`：训练/正式测试输出字段测试。
- `tests/test_daily_soc_and_future_prices.py`：hard/none terminal SOC 行为边界。

### Task 1：实现可执行的 Characteristic Value 与 exact Shapley 语义

**Files:**

- Modify: `algorithms/matd3_shapley_q/coalition.py`
- Modify: `tests/test_matd3_shapley_q.py`

**Interfaces:**

- Consumes: coalition key `frozenset[str]` 与完整的 `J(S)` mapping。
- Produces: `normalize_characteristic_values(physical_values, agents) -> dict[frozenset[str], float]`；`exact_shapley_values(physical_values, agents) -> dict[str, float]`。

- [ ] **Step 1: 用真实非零 `J(empty)` 替换常数自减测试**

```python
def test_characteristic_normalization_uses_nonzero_independent_value():
    agents = ["mg1", "mg2", "mg3"]
    physical = {
        frozenset(): 80.0,
        frozenset({"mg1"}): 84.0,
        frozenset({"mg2"}): 83.0,
        frozenset({"mg3"}): 82.0,
        frozenset({"mg1", "mg2"}): 91.0,
        frozenset({"mg1", "mg3"}): 89.0,
        frozenset({"mg2", "mg3"}): 88.0,
        frozenset(agents): 100.0,
    }
    normalized = normalize_characteristic_values(physical, agents)
    assert physical[frozenset()] != 0.0
    assert normalized[frozenset()] == pytest.approx(0.0)
    assert normalized[frozenset(agents)] == pytest.approx(20.0)
```

- [ ] **Step 2: 运行测试并确认因函数不存在而失败**

Run: `pytest -q tests/test_matd3_shapley_q.py::test_characteristic_normalization_uses_nonzero_independent_value`

Expected: FAIL，提示不能导入或找不到 `normalize_characteristic_values`。

- [ ] **Step 3: 实现完整 coalition key 校验与归一化**

```python
Coalition = frozenset[str]

def all_coalitions(agents: Sequence[str]) -> list[Coalition]:
    ids = tuple(agents)
    return [frozenset(c) for size in range(len(ids) + 1)
            for c in itertools.combinations(ids, size)]

def normalize_characteristic_values(
    physical_values: Mapping[Coalition, float],
    agents: Sequence[str],
) -> dict[Coalition, float]:
    expected = set(all_coalitions(agents))
    if set(physical_values) != expected:
        raise ValueError("physical_values must contain every coalition exactly once")
    baseline = float(physical_values[frozenset()])
    return {coalition: float(value) - baseline
            for coalition, value in physical_values.items()}
```

- [ ] **Step 4: 新增 efficiency、no-cooperation 与 positive-cooperation 数学测试**

```python
AGENTS = ("mg1", "mg2", "mg3")

def positive_cooperation_values():
    return {
        frozenset(): 80.0,
        frozenset({"mg1"}): 84.0,
        frozenset({"mg2"}): 83.0,
        frozenset({"mg3"}): 82.0,
        frozenset({"mg1", "mg2"}): 91.0,
        frozenset({"mg1", "mg3"}): 89.0,
        frozenset({"mg2", "mg3"}): 88.0,
        frozenset(AGENTS): 100.0,
    }

def test_exact_shapley_efficiency_targets_cooperation_surplus():
    values = positive_cooperation_values()
    phi = exact_shapley_values(values, AGENTS)
    assert sum(phi.values()) == pytest.approx(
        values[frozenset(AGENTS)] - values[frozenset()]
    )

def test_exact_shapley_is_zero_when_cooperation_changes_no_value():
    values = {coalition: 80.0 for coalition in all_coalitions(AGENTS)}
    phi = exact_shapley_values(values, AGENTS)
    assert phi == pytest.approx({mg: 0.0 for mg in AGENTS})

def test_exact_shapley_sum_is_positive_for_positive_cooperation():
    values = positive_cooperation_values()
    phi = exact_shapley_values(values, AGENTS)
    assert sum(phi.values()) > 0.0
```

- [ ] **Step 5: 确认三个测试先因 `exact_shapley_values` 缺失而失败**

Run: `pytest -q tests/test_matd3_shapley_q.py -k "exact_shapley or characteristic_normalization"`

Expected: FAIL，且失败来自目标函数缺失，不是 fixture 或拼写错误。

- [ ] **Step 6: 用标准 predecessor marginal formula 实现 exact Shapley**

```python
def exact_shapley_values(
    physical_values: Mapping[Coalition, float],
    agents: Sequence[str],
) -> dict[str, float]:
    normalized = normalize_characteristic_values(physical_values, agents)
    result: dict[str, float] = {}
    for active in agents:
        total = 0.0
        for predecessors in enumerate_predecessor_coalitions(active, agents):
            coalition = frozenset(predecessors)
            joined = coalition | {active}
            total += shapley_weight(len(coalition), len(agents)) * (
                normalized[joined] - normalized[coalition]
            )
        result[active] = total
    return result
```

- [ ] **Step 7: 运行 Task 1 测试并提交**

Run: `pytest -q tests/test_matd3_shapley_q.py -k "coalition or shapley or characteristic"`

Expected: PASS。

```powershell
git add algorithms/matd3_shapley_q/coalition.py tests/test_matd3_shapley_q.py
git commit -m "feat: normalize Shapley values against independent operation"
```

### Task 2：严格固定 Independent Baseline 并同步 Common Scenario

**Files:**

- Modify: `envs/multi_microgrid_env.py`
- Modify: `algorithms/matd3_shapley_q/independent_baseline.py`
- Modify: `configs/algorithms/matd3_shapley_q.yaml`
- Modify: `tests/test_matd3_shapley_q.py`

**Interfaces:**

- Consumes: `MultiMicrogridEnv.last_reset_options`，以及 `IndependentBaselineEvaluator.reset_episode(env)`。
- Produces: `get_step_data(t) -> dict[str, float]`，字段为 `reward`、`profit`、`return_to_go`、`next_return_to_go`；`episode_metrics() -> dict[str, float]`；`scenario_signature(env) -> dict[str, Any]`。

- [ ] **Step 1: 新增 unsupported/missing frozen checkpoint 失败测试**

```python
def algorithm_config_with_baseline(mode, checkpoint=None):
    cfg = yaml.safe_load(
        (ROOT / "configs/algorithms/matd3_shapley_q.yaml").read_text(encoding="utf-8")
    )
    cfg["algorithm"]["shapley_q"]["baseline"] = {
        "mode": mode,
        "checkpoint": checkpoint,
        "disable_p2p": True,
        "freeze_parameters": True,
    }
    return cfg

def test_baseline_rejects_unsupported_mode():
    cfg = algorithm_config_with_baseline(mode="trainable")
    with pytest.raises(ValueError, match="baseline mode"):
        IndependentBaselineEvaluator(env, cfg)

def test_frozen_baseline_requires_checkpoint():
    cfg = algorithm_config_with_baseline(mode="frozen_policy", checkpoint=None)
    with pytest.raises(ValueError, match="checkpoint"):
        IndependentBaselineEvaluator(env, cfg)
```

- [ ] **Step 2: 确认当前代码未按预期报错**

Run: `pytest -q tests/test_matd3_shapley_q.py -k "baseline_rejects or frozen_baseline_requires"`

Expected: FAIL。

- [ ] **Step 3: 在环境 reset 时保存有效场景参数**

```python
self.last_reset_options = {
    "day": str(self.episode_df.iloc[0]["timestamp"]),
    "seed": None if seed is None else int(seed),
    "deterministic": bool(deterministic),
    "forecast_noise": forecast_noise,
}
```

该字段只记录 reset 输入，不改变 episode sampling、forecast 或 reward。

- [ ] **Step 4: 实现 baseline mode/checkpoint 的 fail-fast 校验**

```python
allowed_modes = {"heuristic", "frozen_policy"}
if self.mode not in allowed_modes:
    raise ValueError(f"Unsupported independent baseline mode: {self.mode}")
if self.mode == "frozen_policy" and not self.checkpoint_path:
    raise ValueError("frozen_policy baseline requires an explicit checkpoint")
```

相对 checkpoint 使用 `env.config["_project_root"]` 解析；检查文件存在、`algorithm_slug == "matd3_dispatch"`、action semantics、observation schemas 和 actor state dict，任何不匹配都报错。

- [ ] **Step 5: 新增并运行完整 Common Scenario 失败测试**

```python
def test_common_scenario_matches_all_physical_and_forecast_inputs():
    env.reset(seed=73, deterministic=False, forecast_noise=True)
    evaluator.reset_episode(env)
    ind = evaluator.ind_env
    pd.testing.assert_frame_equal(env.episode_context_df, ind.episode_context_df)
    for mg in env.agent_ids:
        for field in ("load_kw", "pv_kw", "wind_kw"):
            np.testing.assert_allclose(
                env._episode_exogenous[mg][field], ind._episode_exogenous[mg][field]
            )
        for field in env.forecasts._local_arrays[mg]:
            np.testing.assert_allclose(
                env.forecasts._local_arrays[mg][field],
                ind.forecasts._local_arrays[mg][field],
            )
    for field in env.forecasts._public_arrays:
        np.testing.assert_allclose(
            env.forecasts._public_arrays[field], ind.forecasts._public_arrays[field]
        )
    np.testing.assert_allclose(env.forecasts._valid_array, ind.forecasts._valid_array)
```

Run: `pytest -q tests/test_matd3_shapley_q.py::test_common_scenario_matches_all_physical_and_forecast_inputs`

Expected: FAIL，因为当前 baseline 使用不同的 reset flags。

- [ ] **Step 6: 使用 Grand reset 参数启动影子环境并断言场景一致**

```python
options = dict(env.last_reset_options)
self.ind_env.reset(**options)
self._assert_common_scenario(env)
```

`_assert_common_scenario` 比较 context、外生数组、forecast arrays、forecast validity mask 和每个 battery 的 initial SOC；只允许 P2P/ESS cooperation 配置不同。

- [ ] **Step 7: 缓存 discounted baseline return-to-go 并严格访问 step**

```python
running = 0.0
self.discounted_returns = [0.0] * len(self.step_rewards)
for index in range(len(self.step_rewards) - 1, -1, -1):
    running = self.step_rewards[index] + self.gamma * running
    self.discounted_returns[index] = running

def get_step_data(self, t: int) -> dict[str, float]:
    if not 0 <= t < len(self.step_rewards):
        raise IndexError(f"Independent baseline step {t} is unavailable")
    return {
        "reward": self.step_rewards[t],
        "profit": self.step_profits[t],
        "return_to_go": self.discounted_returns[t],
        "next_return_to_go": (
            self.discounted_returns[t + 1] if t + 1 < len(self.discounted_returns) else 0.0
        ),
    }

def episode_metrics(self) -> dict[str, float]:
    return {
        "independent_baseline_return": self.total_return,
        "independent_baseline_profit": self.total_profit,
    }
```

`reset_episode` 同时清空 `self.step_infos`，并在每个 `ind_env.step` 后执行 `self.step_infos.append(info)`，供物理语义测试审计。

- [ ] **Step 8: 将 baseline physics 测试改为检查真实能量流**

```python
def test_independent_baseline_keeps_local_ess_and_grid_but_disables_p2p():
    evaluator.reset_episode(env)
    rows = evaluator.step_infos
    assert all(row["p2p"]["total_p2p_kwh"] == pytest.approx(0.0) for row in rows)
    assert any(sum(v["throughput_kwh"] for v in row["devices"].values()) > 0
               for row in rows)
    assert any(sum(v["grid_import_kw"] + v["grid_export_kw"]
                   for v in row["settlements"].values()) > 0 for row in rows)
```

- [ ] **Step 9: 创建临时合法 checkpoint，验证 frozen parameters 与推理图隔离**

```python
def test_frozen_policy_is_loaded_and_never_receives_gradients(tmp_path):
    independent_env_config = copy.deepcopy(env.config)
    independent_env_config["p2p"]["enabled"] = False
    independent_env_config["ess_sharing"]["ess_p2p_enabled"] = False
    independent_env = MultiMicrogridEnv(independent_env_config, split=env.split)
    source = Matd3Dispatch.from_env(
        independent_env,
        yaml.safe_load((ROOT / "configs/algorithms/matd3_dispatch.yaml").read_text()),
        seed=42,
    )
    checkpoint = tmp_path / "independent.pt"
    source.save(checkpoint)
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("frozen_policy", str(checkpoint))
    )
    env.reset(seed=42)
    evaluator.reset_episode(env)
    assert evaluator.frozen_actors is not None
    assert evaluator.frozen_actors.training is False
    assert all(not parameter.requires_grad for parameter in evaluator.frozen_actors.parameters())
    assert all(parameter.grad is None for parameter in evaluator.frozen_actors.parameters())
```

- [ ] **Step 10: 运行 Task 2 测试并提交**

Run: `pytest -q tests/test_matd3_shapley_q.py -k "baseline or common_scenario or independent"`

Expected: PASS。

```powershell
git add envs/multi_microgrid_env.py algorithms/matd3_shapley_q/independent_baseline.py configs/algorithms/matd3_shapley_q.yaml tests/test_matd3_shapley_q.py
git commit -m "feat: add fixed common-scenario independent baseline"
```

### Task 3：修正 Replay 与 per-agent Twin-min Cooperation Surplus target

**Files:**

- Modify: `algorithms/matd3_shapley_q/replay_buffer.py`
- Modify: `algorithms/matd3_shapley_q/algorithm.py`
- Modify: `tests/test_matd3_shapley_q.py`

**Interfaces:**

- Consumes: `IndependentBaselineEvaluator.get_step_data(t)` 中的两个 return-to-go 字段。
- Produces: Replay batch keys `independent_return_to_go`、`next_independent_return_to_go`；`Matd3ShapleyQ._conservative_shapley_sum(q1, q2)`。

- [ ] **Step 1: 扩展 Replay consistency 测试**

```python
buf.add(
    obs, state, actions, team_r, obs, state, False,
    independent_reward=ind_r,
    cooperation_surplus_reward=team_r - ind_r,
    independent_return_to_go=12.0,
    next_independent_return_to_go=7.0,
)
batch = buf.sample(1, AGENTS, torch.device("cpu"))
assert batch["independent_return_to_go"].item() == pytest.approx(12.0)
assert batch["next_independent_return_to_go"].item() == pytest.approx(7.0)
```

- [ ] **Step 2: 确认 Replay 测试因参数缺失而失败**

Run: `pytest -q tests/test_matd3_shapley_q.py::test_replay_surplus_consistency`

Expected: FAIL，提示 unexpected keyword argument。

- [ ] **Step 3: 扩展 `ShapleyTransition.add/sample` 并检查所有 reward/reference 为 finite**

```python
@dataclass(frozen=True)
class ShapleyTransition:
    # 保留现有字段
    independent_return_to_go: float
    next_independent_return_to_go: float

values = {
    "team_reward": r_grand,
    "independent_reward": r_ind,
    "cooperation_surplus_reward": r_coop,
    "independent_return_to_go": float(independent_return_to_go),
    "next_independent_return_to_go": float(next_independent_return_to_go),
}
bad = [name for name, value in values.items() if not np.isfinite(value)]
if bad:
    raise ValueError(f"Non-finite Shapley replay values: {bad}")
```

`sample` 将两个 reference 以 `[batch, 1]` float tensor 返回。

- [ ] **Step 4: 新增 Twin-min 顺序测试**

```python
def test_conservative_target_takes_twin_min_per_agent_before_sum():
    q1 = {"mg1": torch.tensor([[10.0]]), "mg2": torch.tensor([[1.0]]),
          "mg3": torch.tensor([[10.0]])}
    q2 = {"mg1": torch.tensor([[1.0]]), "mg2": torch.tensor([[10.0]]),
          "mg3": torch.tensor([[1.0]])}
    result = Matd3ShapleyQ._conservative_shapley_sum(q1, q2, AGENTS)
    assert result.item() == pytest.approx(3.0)
    assert result.item() != pytest.approx(21.0)
```

- [ ] **Step 5: 确认测试因 helper 缺失而失败**

Run: `pytest -q tests/test_matd3_shapley_q.py::test_conservative_target_takes_twin_min_per_agent_before_sum`

Expected: FAIL。

- [ ] **Step 6: 实现 per-agent min 并替换错误的 `min(global sums)`**

```python
@staticmethod
def _conservative_shapley_sum(q1, q2, agent_ids):
    return sum(torch.minimum(q1[mg], q2[mg]) for mg in agent_ids)

tgt_qcoop_min = self._conservative_shapley_sum(
    tgt_qphi1, tgt_qphi2, self.agent_ids
)
y_coop = r_coop + (1.0 - batch["done"]) * self.gamma * tgt_qcoop_min
```

保留 `loss_q1 = mse(sum(cur_qphi1.values()), y_coop)` 与 `loss_q2 = mse(sum(cur_qphi2.values()), y_coop)`。

- [ ] **Step 7: 计算真实 reference 支持的 grand-decomposition residual**

```python
qcoop_current = self._conservative_shapley_sum(cur_qphi1, cur_qphi2, self.agent_ids)
grand_target = batch["team_reward"] + (1.0 - batch["done"]) * self.gamma * (
    batch["next_independent_return_to_go"] + tgt_qcoop_min
)
grand_decomposition_residual = (
    batch["independent_return_to_go"] + qcoop_current - grand_target
)
```

将其均值写入顶层 metrics 与 `algorithm_diagnostics`。这不是恒等式：它比较当前分解估计与 Grand Bellman target。

- [ ] **Step 8: 强化 Actor gradient 测试**

```python
alg.policy_delay = 1
obs, reset = env.reset(seed=42)
alg.on_episode_start(env, seed=42)
state = reset["global_state"]
for t in range(8):
    actions = {mg: np.array([-0.5], dtype=np.float32) for mg in AGENTS}
    next_obs, rewards, done, info = env.step(actions)
    baseline = alg.get_independent_step_data(t)
    alg.replay.add(
        obs, state, actions, rewards["mg1"], next_obs, info["global_state_after"], done,
        independent_reward=baseline["reward"],
        cooperation_surplus_reward=rewards["mg1"] - baseline["reward"],
        independent_return_to_go=baseline["return_to_go"],
        next_independent_return_to_go=baseline["next_return_to_go"],
    )
    obs, state = next_obs, info["global_state_after"]
metrics = alg.update(batch_size=8)
assert all(np.isfinite(metrics["actor_grad_norms"][mg])
           and metrics["actor_grad_norms"][mg] > 0 for mg in AGENTS)
if alg.baseline_evaluator.frozen_actors is not None:
    assert all(parameter.grad is None
               for parameter in alg.baseline_evaluator.frozen_actors.parameters())
```

- [ ] **Step 9: 严格 checkpoint 版本测试**

保存新 checkpoint 后检查：

```python
assert checkpoint["shapley_q_version"] == "surplus_q_v2"
assert checkpoint["value_definition"] == "cooperation_surplus"
assert checkpoint["baseline"]["mode"] == "heuristic"
```

把版本改成 `full_enumeration_v1` 后，`load` 必须抛出 `ValueError`，不得继续兼容旧目标。

- [ ] **Step 10: 运行 Task 3 测试并提交**

Run: `pytest -q tests/test_matd3_shapley_q.py -k "replay or conservative or gradient or checkpoint"`

Expected: PASS。

```powershell
git add algorithms/matd3_shapley_q/replay_buffer.py algorithms/matd3_shapley_q/algorithm.py tests/test_matd3_shapley_q.py
git commit -m "fix: train Shapley critics on cooperation surplus"
```

### Task 4：补齐训练 hook、episode diagnostics、summary 与 TensorBoard

**Files:**

- Modify: `experiments/training_pipeline.py`
- Modify: `utils/training_dashboard.py`
- Modify: `scripts/export_runs_to_tensorboard.py`
- Modify: `tests/test_metrics_recording.py`
- Modify: `tests/test_matd3_shapley_q.py`

**Interfaces:**

- Consumes: `algorithm.on_episode_start(env)`、`algorithm.get_independent_step_data(t)` 与 update metrics。
- Produces: `_algorithm_replay_extras(algorithm, step_idx, team_reward) -> dict[str, float]`、training history/summary fields 和 TensorBoard tags `reward/*`、`q/*`。

- [ ] **Step 1: 新增 transition hook 与 episode 字段测试**

```python
class FakeAlgorithm:
    def get_independent_step_data(self, t):
        assert t == 3
        return {"reward": 2.0, "profit": 200.0,
                "return_to_go": 8.0, "next_return_to_go": 6.0}

def test_algorithm_replay_extras_uses_fixed_baseline_step():
    extras = _algorithm_replay_extras(FakeAlgorithm(), 3, team_reward=5.0)
    assert extras == pytest.approx({
        "independent_reward": 2.0,
        "cooperation_surplus_reward": 3.0,
        "independent_return_to_go": 8.0,
        "next_independent_return_to_go": 6.0,
    })
```

- [ ] **Step 2: 运行测试并确认字段或 hook 缺失**

Run: `pytest -q tests/test_matd3_shapley_q.py -k "episode_record or transition_hook"`

Expected: FAIL。

- [ ] **Step 3: 将 pipeline 的 baseline step 调用统一到一个公开方法**

```python
def _algorithm_replay_extras(algorithm, step_idx, team_reward):
    if not hasattr(algorithm, "get_independent_step_data"):
        return {}
    baseline = algorithm.get_independent_step_data(step_idx)
    return {
        "independent_reward": baseline["reward"],
        "cooperation_surplus_reward": team_reward - baseline["reward"],
        "independent_return_to_go": baseline["return_to_go"],
        "next_independent_return_to_go": baseline["next_return_to_go"],
    }

extras = _algorithm_replay_extras(algorithm, step_idx, team_reward)
algorithm.replay.add(
    observations, state, actions, team_reward,
    next_observations, info["global_state_after"], done,
    **extras,
)
```

非 Shapley 算法继续走现有 `replay.add`，不改变接口。

- [ ] **Step 4: 为 smoke verification 增加隐藏的 CLI 超参数覆盖**

```python
parser.add_argument("--warmup-steps", type=int, help=argparse.SUPPRESS)
parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)

warmup = int(args.warmup_steps if args.warmup_steps is not None
             else settings.get("warmup_steps", 1000))
batch_size = int(args.batch_size if args.batch_size is not None
                 else settings.get("batch_size", 128))
```

新增 parser/resolution 测试，确保覆盖值只影响当前 run，不写回算法 YAML。

- [ ] **Step 5: 补齐 `_diagnostic_means` 和 `training_summary`**

```python
scalar_names = (
    # 保留现有字段
    "grand_decomposition_residual",
)

def final_mean(field: str, window: int = 100):
    return _mean([record.get(field) for record in history[-window:]])

training_summary.update({
    "final_100_grand_return_mean": final_mean("grand_team_return"),
    "final_100_independent_return_mean": final_mean("independent_baseline_return"),
    "final_100_cooperation_surplus_return_mean": final_mean("cooperation_surplus_return"),
    "final_100_shapley_q_sum_mean": final_mean("shapley_q_sum"),
    "final_100_efficiency_residual_mean": final_mean("efficiency_residual"),
    "final_100_grand_decomposition_residual_mean": final_mean(
        "grand_decomposition_residual"
    ),
})
```

没有 update 的短 episode 使用 `None`，不得伪造 0。

- [ ] **Step 6: 在 dashboard 添加最小必要展示**

```python
def metric(name):
    value = self.last_metrics.get(name)
    return "N/A" if value is None else f"{float(value):.4f}"

profit_table.add_row(
    f"Grand Return: {metric('grand_team_return')}",
    f"Independent: {metric('independent_baseline_return')}",
)
profit_table.add_row(
    f"Surplus: {metric('cooperation_surplus_return')}",
    f"Efficiency residual: {metric('efficiency_residual')}",
)
profit_table.add_row(
    f"Shapley sum: {metric('shapley_q_sum')}",
    f"Grand decomp residual: {metric('grand_decomposition_residual')}",
)
```

字段缺失时显示 `N/A`，以保持其他算法兼容。

- [ ] **Step 7: 新增 TensorBoard exporter 测试**

使用临时 run directory 和 fake `SummaryWriter`，检查以下 tags 被写入：

```python
expected_tags = {
    "reward/team", "reward/independent", "reward/cooperation_surplus",
    "q/shapley_mg1", "q/shapley_mg2", "q/shapley_mg3", "q/shapley_sum",
    "q/efficiency_residual", "q/grand_decomposition_residual",
}
```

- [ ] **Step 8: 确认 exporter 测试先失败，再添加准确 mapping**

```python
tag_map = {
    "team_reward": "reward/team",
    "independent_baseline_return": "reward/independent",
    "cooperation_surplus_return": "reward/cooperation_surplus",
    "shapley_q_mg1": "q/shapley_mg1",
    "shapley_q_mg2": "q/shapley_mg2",
    "shapley_q_mg3": "q/shapley_mg3",
    "shapley_q_sum": "q/shapley_sum",
    "efficiency_residual": "q/efficiency_residual",
    "grand_decomposition_residual": "q/grand_decomposition_residual",
}
```

- [ ] **Step 9: 运行 Task 4 测试并提交**

Run: `pytest -q tests/test_metrics_recording.py tests/test_matd3_shapley_q.py -k "tensorboard or episode or diagnostic or summary"`

Expected: PASS。

```powershell
git add experiments/training_pipeline.py utils/training_dashboard.py scripts/export_runs_to_tensorboard.py tests/test_metrics_recording.py tests/test_matd3_shapley_q.py
git commit -m "feat: report independent and Shapley-Q training diagnostics"
```

### Task 5：在 Formal Test 中输出同策略 Independent Baseline 结果

**Files:**

- Modify: `evaluation/evaluate_matd3.py`
- Modify: `tests/test_matd3_shapley_q.py`
- Modify: `tests/test_metrics_recording.py`

**Interfaces:**

- Consumes: `evaluate_policy_on_days(..., episode_start_fn=None, episode_finalize_fn=None)`。
- Produces: 每日字段 `grand_team_return`、`independent_baseline_return`、`cooperation_surplus_return` 及 profit 对应字段；summary 的 matched-baseline statistics。

- [ ] **Step 1: 为 evaluator lifecycle callback 写失败测试**

```python
summary, daily = evaluate_policy_on_days(
    env, [day], policy,
    episode_start_fn=lambda current_env, episode_seed: algorithm.on_episode_start(current_env),
    episode_finalize_fn=lambda: {
        "independent_baseline_return": algorithm.baseline_evaluator.get_episode_return(),
        "independent_baseline_profit": algorithm.baseline_evaluator.get_episode_profit(),
    },
)
assert "independent_baseline_return" in daily[0]
```

- [ ] **Step 2: 确认测试因 callback 参数不受支持而失败**

Run: `pytest -q tests/test_metrics_recording.py -k "evaluation_baseline_lifecycle"`

Expected: FAIL，提示 unexpected keyword argument。

- [ ] **Step 3: 实现向后兼容的 evaluation lifecycle callbacks**

```python
def evaluate_policy_on_days(
    env,
    days,
    policy_fn,
    *,
    seed=42,
    deterministic_environment=True,
    forecast_noise=None,
    capture_hourly=False,
    episode_start_fn=None,
    episode_finalize_fn=None,
):
    episode_seed = int(seed) + index
    observations, _ = env.reset(
        day=day,
        seed=episode_seed,
        deterministic=bool(deterministic_environment),
        forecast_noise=forecast_noise,
    )
    if episode_start_fn is not None:
        episode_start_fn(env, episode_seed)
    # 保留现有 rollout 与 collector.record_step 调用。
    metrics = collector.finalize_episode()
    if episode_finalize_fn is not None:
        metrics.update(dict(episode_finalize_fn()))
        metrics["grand_team_return"] = float(metrics["team_reward"])
        metrics["cooperation_surplus_return"] = (
            metrics["grand_team_return"] - metrics["independent_baseline_return"]
        )
        metrics["grand_team_profit"] = float(metrics["team_profit"])
        metrics["cooperation_surplus_profit"] = (
            metrics["grand_team_profit"] - metrics["independent_baseline_profit"]
        )
```

Callback 默认 `None`，因此 MATD3、CrossGate、PNN 等现有算法不受影响。

- [ ] **Step 4: 在 `evaluate()` 中只为有 baseline evaluator 的算法启用 callbacks**

```python
episode_start = None
episode_finalize = None
if hasattr(algorithm, "baseline_evaluator"):
    episode_start = lambda current_env, episode_seed: algorithm.on_episode_start(
        current_env, episode_seed
    )
    episode_finalize = algorithm.baseline_evaluator.episode_metrics
```

正式测试的 Independent Baseline 必须是在线训练使用的固定 policy；MILP independent value 继续使用不同字段并只用于 settlement/benchmark。

- [ ] **Step 5: 扩展 daily CSV row 与 JSON summary 测试**

```python
assert row["cooperation_surplus_return"] == pytest.approx(
    row["grand_team_return"] - row["independent_baseline_return"]
)
assert row["cooperation_surplus_profit"] == pytest.approx(
    row["algorithm_team_profit"] - row["independent_baseline_profit"]
)
assert result["independent_baseline_profit_statistics"]["mean"] is not None
```

- [ ] **Step 6: 运行 Formal Test 相关测试并提交**

Run: `pytest -q tests/test_metrics_recording.py tests/test_matd3_shapley_q.py -k "evaluation or formal or baseline"`

Expected: PASS。

```powershell
git add evaluation/evaluate_matd3.py tests/test_metrics_recording.py tests/test_matd3_shapley_q.py
git commit -m "feat: evaluate Shapley-Q against its fixed baseline"
```

### Task 6：验证 hard terminal SOC、配置隔离与真实物理场景

**Files:**

- Modify: `configs/cooperative.yaml`
- Modify: `tests/test_daily_soc_and_future_prices.py`
- Modify: `tests/test_matd3_shapley_q.py`

**Interfaces:**

- Consumes: `terminal_soc.mode in {none, penalty, hard}` 与现有环境 action API。
- Produces: hard mode 无 shaping/penalty 且终端回到初始 SOC 的可执行保证。

- [ ] **Step 1: 修正配置注释并保留数值**

```yaml
terminal_soc:
  # Daily dispatch uses a hard terminal target with no reward shaping.
  mode: hard
  tolerance: 1.0e-6
  penalty_per_kwh: 1.5
```

`penalty_per_kwh` 只供显式 `mode: penalty` 的独立测试使用，hard mode 不读取它形成 reward。

- [ ] **Step 2: 加强 terminal SOC 测试**

```python
assert step_info["shaping_reward"] == pytest.approx(0.0)
assert all(v == pytest.approx(0.0)
           for v in step_info["terminal_soc_penalty"].values())
assert all(core.battery.soc == pytest.approx(core.battery.initial_soc, abs=1.1e-5)
           for core in env.cores.values())
```

保留显式把测试配置设为 `mode: none` 的旧行为测试，避免全局默认改变其语义。

- [ ] **Step 3: 构造确定性正合作物理场景**

用临时 24 小时 CSV 和复制后的环境配置创建两个环境。测试 helper 显式固定同一外生场景：

```python
def make_deterministic_config(tmp_path):
    timestamps = pd.date_range("2018-01-01", periods=24, freq="h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "load_kw": np.ones(24),
        "pv_kw": np.zeros(24),
        "wind_kw": np.zeros(24),
        "buy_price_per_kwh": np.full(24, 1.20),
        "grid_carbon_factor_kg_per_kwh": np.full(24, 0.5),
        "temperature_c": np.full(24, 20.0),
        "ghi_w_per_m2": np.full(24, 1000.0),
        "wind_speed_mps": np.zeros(24),
        "mg1_ghi_w_per_m2": np.full(24, 1000.0),
        "mg1_wind_speed_mps": np.zeros(24),
        "mg2_ghi_w_per_m2": np.zeros(24),
        "mg2_wind_speed_mps": np.zeros(24),
        "mg3_ghi_w_per_m2": np.zeros(24),
        "mg3_wind_speed_mps": np.zeros(24),
    })
    csv_path = tmp_path / "deterministic.csv"
    frame.to_csv(csv_path, index=False)
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    for key in ("train_path", "val_path", "test_path"):
        cfg["data"][key] = str(csv_path)
    cfg["observation"]["noisy_forecast"] = False
    return cfg

def build_deterministic_pair(base_config):
    grand_cfg = copy.deepcopy(base_config)
    independent_cfg = copy.deepcopy(base_config)
    grand_cfg["p2p"]["enabled"] = True
    grand_cfg["ess_sharing"]["ess_p2p_enabled"] = True
    independent_cfg["p2p"]["enabled"] = False
    independent_cfg["ess_sharing"]["ess_p2p_enabled"] = False
    return MultiMicrogridEnv(grand_cfg), MultiMicrogridEnv(independent_cfg)

deterministic_config = make_deterministic_config(tmp_path)
grand, independent = build_deterministic_pair(deterministic_config)
grand_obs, _ = grand.reset(day=TEST_DAY, seed=17, deterministic=True)
ind_obs, _ = independent.reset(day=TEST_DAY, seed=17, deterministic=True)
zero_actions = {mg: np.zeros(1, dtype=np.float32) for mg in grand.agent_ids}
grand_profit = independent_profit = grand_p2p_kwh = 0.0
for _ in range(24):
    grand_obs, grand_rewards, grand_done, grand_info = grand.step(zero_actions)
    ind_obs, ind_rewards, ind_done, ind_info = independent.step(zero_actions)
    grand_profit += grand_info["team_profit"]
    independent_profit += ind_info["team_profit"]
    grand_p2p_kwh += grand_info["p2p"]["total_p2p_kwh"]
grand_return = grand_profit / grand.accounting.reward_scale
independent_return = independent_profit / independent.accounting.reward_scale
```

该 CSV 令 MG1 具有明显 renewable surplus、MG2 具有高价 deficit。随后断言：

```python
assert grand_p2p_kwh > 0.0
assert grand_profit > independent_profit
assert grand_return - independent_return > 0.0
```

- [ ] **Step 4: 构造确定性 no-cooperation 物理场景**

```python
def rollout_fixed_actions(env, day, seed):
    observations, _ = env.reset(day=day, seed=seed, deterministic=True)
    rewards, profit = [], 0.0
    done = False
    while not done:
        actions = {mg: np.zeros(1, dtype=np.float32) for mg in env.agent_ids}
        observations, shared_rewards, done, info = env.step(actions)
        rewards.append(float(shared_rewards[env.agent_ids[0]]))
        profit += float(info["team_profit"])
    return {"rewards": rewards, "profit": profit}

no_coop_config = copy.deepcopy(deterministic_config)
no_coop_config["p2p"]["enabled"] = False
no_coop_config["ess_sharing"]["ess_p2p_enabled"] = False
grand_values = rollout_fixed_actions(
    MultiMicrogridEnv(no_coop_config), TEST_DAY, seed=23
)
independent_values = rollout_fixed_actions(
    MultiMicrogridEnv(copy.deepcopy(no_coop_config)), TEST_DAY, seed=23
)
assert grand_values["rewards"] == pytest.approx(independent_values["rewards"])
assert grand_values["profit"] == pytest.approx(independent_values["profit"])

j_values = {
    coalition: independent_values["profit"]
    for coalition in all_coalitions(AGENTS)
}
phi = exact_shapley_values(j_values, AGENTS)
assert phi == pytest.approx({mg: 0.0 for mg in AGENTS})
```

- [ ] **Step 5: 验证空联盟 projector 不清零本地动作**

```python
empty = torch.zeros((1, 3))
effective, plans = alg.coalition_projector(state, raw, empty)
assert torch.count_nonzero(effective[..., 0]).item() > 0
assert torch.count_nonzero(effective[..., 1]).item() == 0
assert all(not plan["in_coalition"].any() for plan in plans)
```

- [ ] **Step 6: 运行 physics/SOC 测试并提交**

Run: `pytest -q tests/test_daily_soc_and_future_prices.py tests/test_matd3_shapley_q.py -k "terminal or cooperation or empty_coalition or physics"`

Expected: PASS。

```powershell
git add configs/cooperative.yaml tests/test_daily_soc_and_future_prices.py tests/test_matd3_shapley_q.py
git commit -m "test: verify Shapley-Q physics and hard terminal SOC"
```

### Task 7：全量验证与共享流水线 Smoke Test

**Files:**

- Verify only: `algorithms/matd3/`
- Generated: `outputs/matd3_shapley_q/runs/<run_id>/...`

**Interfaces:**

- Consumes: 前六个任务的完整实现。
- Produces: 可审计的测试输出和标准 run directory。

- [ ] **Step 1: 运行专属测试**

Run: `pytest -q tests/test_matd3_shapley_q.py`

Expected: 全部 PASS，无 skipped 的核心验收测试。

- [ ] **Step 2: 运行全仓库测试**

Run: `pytest -q`

Expected: 0 failed。

- [ ] **Step 3: 运行带实际 update 的短流水线**

Run:

```powershell
python scripts/run_experiment.py --algorithm matd3_shapley_q --episodes 2 --test-episodes 2 --validation-interval 1 --validation-days 1 --warmup-steps 0 --batch-size 8 --no-dashboard
```

隐藏 CLI override 只作用于本次 smoke run，确保至少发生一次 Critic update 和一次 delayed Actor update；不得把这些 smoke 数值写回正式 YAML 默认值。

Expected: training → validation → best checkpoint → formal test → coalition MILP → Equal/Shapley settlement → summary → plotting 全部完成。

- [ ] **Step 4: 自动核对 run artifacts**

```powershell
$latest = Get-Content outputs/matd3_shapley_q/latest_run.json | ConvertFrom-Json
$run = $latest.run_directory
Test-Path "$run/checkpoints/best_model.pt"
Test-Path "$run/checkpoints/final_model.pt"
(Get-ChildItem "$run/figures" -Filter *.png).Count
(Get-ChildItem "$run/figures" -Filter *.pdf).Count
```

Expected: 两个 checkpoint 都存在，PNG=15，PDF=15；`test_summary.json` 的 `checkpoint_used` 指向 `best_model.pt`。

- [ ] **Step 5: 核对 plugin/config/output identity**

检查 `run_metadata.json`、`config_snapshot.yaml` 和 `summary.json`：

```text
algorithm_slug = matd3_shapley_q
algorithm config path = configs/algorithms/matd3_shapley_q.yaml
run directory contains outputs/matd3_shapley_q/runs/
checkpoint selection metric = validation_team_profit_mean
```

- [ ] **Step 6: 核对指标恒等式与非有限值**

逐条检查 training history 与 formal-test daily CSV：

```text
cooperation_surplus_reward = team_reward - independent_reward
cooperation_surplus_return = grand_team_return - independent_baseline_return
cooperation_surplus_profit = grand_team_profit - independent_baseline_profit
```

所有 reward、Q、residual 与 return 字段必须 finite；不得用 `null` 掩盖已经发生 update 的 episode。

- [ ] **Step 7: 确认 canonical MATD3 未改变**

Run: `git diff -- algorithms/matd3/`

Expected: 无输出。

- [ ] **Step 8: 检查最终 diff 与提交**

Run: `git diff --check`，Expected: 无 error。

```powershell
git add algorithms/matd3_shapley_q configs/algorithms/matd3_shapley_q.yaml envs/multi_microgrid_env.py experiments/training_pipeline.py evaluation/evaluate_matd3.py utils/training_dashboard.py scripts/export_runs_to_tensorboard.py configs/cooperative.yaml tests/test_matd3_shapley_q.py tests/test_metrics_recording.py tests/test_daily_soc_and_future_prices.py
git commit -m "feat: complete independent-baseline Shapley-Q training"
```
