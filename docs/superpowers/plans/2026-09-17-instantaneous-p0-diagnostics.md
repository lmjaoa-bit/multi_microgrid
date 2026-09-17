# Instantaneous P0 Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace strong future-reachability projection with one canonical instantaneous environment, add P0 terminal-pressure observations for every current algorithm, and sample expensive training diagnostics every 20 updates without breaking MATD3 policy delay.

**Architecture:** Current-step device and ESS limits remain in the component models and joint differentiable projector; terminal targets move entirely to observations and end-of-episode penalties. A small constraint-guidance helper supplies normalized P0 values to both Actor observations and centralized Critic state. A new diagnostic-sampling module installs one of two shared projected-update paths on the four experiment plugins while leaving `algorithms/matd3/` untouched.

**Tech Stack:** Python 3, NumPy, pandas, PyYAML, PyTorch, pytest, existing plugin registry and shared experiment pipeline.

**Spec:** `docs/superpowers/specs/2026-09-17-instantaneous-p0-diagnostics-design.md`

## Global Constraints

- Do not modify any file under `algorithms/matd3/`.
- Preserve the user's modified `data/spatial_market_2018_{train,val,test}.csv` and metadata file.
- `MultiMicrogridEnv` is the only formal environment; no strong/instantaneous selector remains.
- Keep current-step physical constraints, terminal targets, and terminal penalties.
- Keep `network.forecast_horizon: 4` in both Transformer YAML files.
- Use `python scripts/run_experiment.py --algorithm <slug>` for every pipeline smoke run.
- Normal run artifacts stay under `outputs/working_runs/<slug>/runs/`.

## File Structure

- `components/battery.py`: instantaneous ESS action bounds only; no terminal-reachability API.
- `components/industrial_mill.py`: next-silo-state feasibility and raw P0 mill state.
- `components/commercial_hvac.py`: next-comfort-state feasibility and raw P0 thermal state.
- `components/ev_charging.py`: availability/headroom feasibility and existing deadline state.
- `core/microgrid_core.py`: consume the simplified instantaneous battery API.
- `core/constraint_observations.py`: pure ESS/mill/HVAC pressure calculations shared by builders.
- `core/observation.py`: public per-agent P0 schema and normalized values.
- `core/global_state.py`: matching centralized P0 fields and instantaneous feasible ranges.
- `envs/multi_microgrid_env.py`: canonical environment construction, terminal penalty breakdown, and metadata.
- `core/config_validation.py`, `configs/cooperative.yaml`, `experiments/profiles.py`: one environment contract with penalty-mode terminal targets.
- `algorithms/diagnostic_sampling.py`: diagnostic scheduler, fast projected updates, and installer.
- `algorithms/runtime_helpers.py`: replay packing and action-selection helpers only.
- Four algorithm plugins and YAML files: install fast paths and set `logging.diagnostic_interval: 20`.
- Focused tests: instantaneous physics, P0 schemas/formulas, scheduling, four-plugin training/checkpoint behavior.
- Remove the obsolete instantaneous-ablation environment, config, runner, test, and CI workflow.

---

### Task 1: Remove Cross-Time Projection From Physical Components

**Files:**
- Modify: `components/battery.py`
- Modify: `components/industrial_mill.py`
- Modify: `components/commercial_hvac.py`
- Modify: `components/ev_charging.py`
- Modify: `core/microgrid_core.py`
- Modify: `algorithms/milp_mpc/algorithm.py`
- Modify: `tests/test_controllable_loads.py`
- Modify: `tests/test_cooperative_physics.py`
- Create: `tests/test_instantaneous_constraints.py`

**Interfaces:**
- Produces: `BatteryModel.map_action(action: float) -> dict[str, float]` using SOC/power limits only.
- Produces: each controllable load's `feasible_bounds(t)` using only the next physical state.
- Preserves: `plan(raw_action, t)`, `commit(plan, delivered_fraction)`, and `state(t)` public interfaces.

- [ ] **Step 1: Write failing instantaneous-constraint tests**

Add tests that distinguish instantaneous feasibility from terminal reachability:

```python
def test_battery_last_hour_does_not_force_terminal_soc(env):
    env.reset(seed=3, deterministic=True)
    battery = env.cores["mg1"].battery
    battery.soc = battery.soc_min
    bounds = env._battery_flex(23)["mg1"]
    assert bounds["feasible_charge_min_ratio"] == pytest.approx(0.0)
    assert bounds["feasible_discharge_max_ratio"] == pytest.approx(0.0)


def test_mill_last_hour_only_protects_next_silo_state(mill):
    mill.silo_t = mill.initial_silo_t
    plan = mill.plan(-1.0, 23)
    assert plan["effective_duty"] == pytest.approx(0.0)
    assert mill.silo_min_t <= plan["silo_after_t"] <= mill.silo_max_t
    assert abs(plan["silo_after_t"] - mill.initial_silo_t) > mill.terminal_tolerance_t


def test_ev_minimum_charge_is_zero_before_departure(ev):
    low, high = ev.feasible_bounds(6)
    assert low == pytest.approx(0.0)
    assert high > 0.0


def test_hvac_bounds_never_call_terminal_optimizer(hvac, monkeypatch):
    monkeypatch.setattr(
        hvac,
        "_solve_terminal_q",
        lambda *args, **kwargs: pytest.fail("terminal optimization called"),
        raising=False,
    )
    low, high = hvac.feasible_bounds(0)
    assert -hvac.cool_max_kw <= low <= high <= hvac.heat_max_kw
```

- [ ] **Step 2: Run the focused tests and verify the expected red state**

Run:

```powershell
pytest -q tests/test_instantaneous_constraints.py tests/test_controllable_loads.py tests/test_cooperative_physics.py
```

Expected: failures show the last-hour ESS/mill and EV lower bounds are still raised by terminal reachability; the HVAC test reports a terminal optimizer call.

- [ ] **Step 3: Simplify the component implementations**

Implement these exact semantics:

```python
def feasible_bounds(self, t: int) -> tuple[float, float]:  # mill
    production = self.production_tph * self.delta_t
    consumption = self.downstream_tph * self.delta_t
    low = max(0.0, (self.silo_min_t - self.silo_t + consumption) / production)
    high = min(1.0, (self.silo_max_t - self.silo_t + consumption) / production)
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def feasible_bounds(self, t: int) -> tuple[float, float]:  # EV
    if self._remaining_steps(t) <= 0:
        return 0.0, 0.0
    headroom_kwh = max(0.0, (self.max_soc - self.soc) * self.capacity_kwh)
    maximum = min(self.max_charge_kw, headroom_kwh / (self.eta * self.delta_t))
    return 0.0, float(maximum)
```

For HVAC, retain `_one_step_bounds_q(t)` and remove `_solve_terminal_q` plus the terminal optimization merge. For ESS, remove `reachable_soc_envelope`, remove the reachability arguments and forced minimum branches from `map_action`, and compute maxima from `soc_min`, `soc_max`, efficiencies, and power ratings. Change `MicrogridCore.plan_local` to call only `battery.map_action(float(action["battery"]))`. Update MILP-MPC's neutral bound query to the same instantaneous call so the repository has no remaining caller of the removed reachability API.

- [ ] **Step 4: Run focused tests to verify green**

Run:

```powershell
pytest -q tests/test_instantaneous_constraints.py tests/test_controllable_loads.py tests/test_cooperative_physics.py tests/test_milp_mpc.py
```

Expected: all selected tests pass and `rg -n "reachable_soc_envelope|_solve_terminal_q" components core envs algorithms` returns no implementation references.

- [ ] **Step 5: Commit the component change**

```powershell
git add components/battery.py components/industrial_mill.py components/commercial_hvac.py components/ev_charging.py core/microgrid_core.py algorithms/milp_mpc/algorithm.py tests/test_instantaneous_constraints.py tests/test_controllable_loads.py tests/test_cooperative_physics.py tests/test_milp_mpc.py
git commit -m "refactor: make device constraints instantaneous"
```

### Task 2: Make the Instantaneous Environment Canonical

**Files:**
- Modify: `envs/multi_microgrid_env.py`
- Modify: `core/config_validation.py`
- Modify: `configs/cooperative.yaml`
- Modify: `experiments/profiles.py`
- Modify: `.github/workflows/matd3-dr-short-validation.yml`
- Delete: `envs/instantaneous_projection_env.py`
- Delete: `configs/cooperative_instantaneous_projection.yaml`
- Delete: `scripts/run_projection_ablation.py`
- Delete: `tests/test_projection_ablation.py`
- Delete: `.github/workflows/matd3-projection-ablation-short-validation.yml`
- Test: `tests/test_instantaneous_constraints.py`
- Test: `tests/test_experiment_profiles.py`
- Test: `tests/test_metrics_recording.py`

**Interfaces:**
- Produces: `MultiMicrogridEnv._terminal_constraint_breakdown(mg: str) -> dict[str, float]`.
- Produces: terminal `info["terminal_constraint_penalty"]` with ESS, mill, HVAC, EV, total-equivalent-energy, rate, and total penalty.
- Removes: `constraint_projection.mode` and the projection-ablation entry points.

- [ ] **Step 1: Write failing canonical-environment and penalty tests**

Add the following assertions to `tests/test_instantaneous_constraints.py`:

```python
def test_canonical_config_has_no_projection_mode():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert "constraint_projection" not in config
    assert config["terminal_soc"]["mode"] == "penalty"


def test_terminal_penalty_reports_every_device(env):
    env.reset(seed=7, deterministic=True)
    done = False
    while not done:
        actions = {mg: np.asarray([0.0, -1.0], dtype=np.float32) for mg in env.agent_ids}
        _, rewards, done, info = env.step(actions)
    breakdown = info["terminal_constraint_penalty"]
    assert set(breakdown) == set(env.agent_ids)
    assert set(breakdown["mg1"]) >= {
        "ess_equivalent_kwh", "mill_equivalent_kwh", "hvac_equivalent_kwh",
        "ev_equivalent_kwh", "total_equivalent_kwh", "total_penalty",
    }
    assert info["terminal_constraint_penalty_total"] == pytest.approx(
        sum(item["total_penalty"] for item in breakdown.values())
    )
    assert rewards["mg1"] == pytest.approx(info["team_profit"] / env.accounting.reward_scale)
```

Update profile-contract tests to assert that no selectable projection mode or physical-settings entry remains.

- [ ] **Step 2: Run the new environment tests and verify red**

Run:

```powershell
pytest -q tests/test_instantaneous_constraints.py tests/test_experiment_profiles.py tests/test_metrics_recording.py
```

Expected: the canonical config still contains `mode: strong`, and the base DR environment does not yet apply all-device terminal penalties.

- [ ] **Step 3: Move terminal penalties into `MultiMicrogridEnv` and remove the selector**

Move the equivalent-energy formulas from the obsolete subclass into `_terminal_constraint_breakdown`. Apply the returned penalty once, after committing final device states and before computing team profit:

```python
breakdown = {mg: self._terminal_constraint_breakdown(mg) for mg in self.agent_ids}
for mg, item in breakdown.items():
    penalty = item["total_penalty"] if terminal else 0.0
    costs[mg]["terminal_constraint_penalty"] = penalty
    costs[mg]["terminal_soc_penalty"] = penalty
    costs[mg]["physical_profit"] -= penalty
    costs[mg]["individual_accounting_profit"] -= penalty
```

Make both `_step_dr` and `_step_legacy` call the same helper. Set `terminal_soc.mode: penalty`, delete `constraint_projection` validation and contract fields, remove the separate config/environment/runner/test/workflow, and replace the CI strong-mode assertion with an assertion that terminal mode is `penalty` and no projection-mode key exists.

- [ ] **Step 4: Run canonical-environment tests to verify green**

Run:

```powershell
pytest -q tests/test_instantaneous_constraints.py tests/test_experiment_profiles.py tests/test_metrics_recording.py tests/test_dr_environment.py
```

Expected: all selected tests pass, and this search returns no matches:

```powershell
rg -n "constraint_projection|InstantaneousProjectionEnv|run_projection_ablation" configs core envs experiments scripts tests .github
```

- [ ] **Step 5: Commit the canonical environment**

```powershell
git add envs/multi_microgrid_env.py core/config_validation.py configs/cooperative.yaml experiments/profiles.py .github/workflows/matd3-dr-short-validation.yml tests/test_instantaneous_constraints.py tests/test_experiment_profiles.py tests/test_metrics_recording.py
git add -u envs configs scripts tests .github
git commit -m "feat: make instantaneous constraints canonical"
```

### Task 3: Add Shared P0 Constraint-Aware Observations

**Files:**
- Create: `core/constraint_observations.py`
- Modify: `components/industrial_mill.py`
- Modify: `components/commercial_hvac.py`
- Modify: `core/observation.py`
- Modify: `core/global_state.py`
- Modify: `envs/multi_microgrid_env.py`
- Create: `tests/test_constraint_aware_observations.py`
- Modify: `tests/test_daily_soc_and_future_prices.py`
- Modify: `tests/test_dr_algorithm_adapters.py`
- Modify: `tests/test_pnn_transformer_runtime_backport.py`

**Interfaces:**
- Produces: `ess_terminal_guidance(battery, remaining_hours) -> dict[str, float]` with physical kW plus normalized signed requirement and pressure.
- Produces: raw component fields `required_avg_production_tph`, `production_pressure`, `terminal_air_gap_c`, `terminal_mass_gap_c`, `time_remaining_h`, and `thermal_recovery_pressure`.
- Produces: stable Actor fields and matching `<mg>_...` Critic fields described in the design spec.

- [ ] **Step 1: Write failing formula and schema tests**

Create tests that name every required public field:

```python
ESS_FIELDS = {"terminal_soc_gap", "time_to_terminal", "required_avg_power_to_target", "terminal_pressure"}
MILL_FIELDS = {"silo_target_gap", "time_remaining", "required_avg_production", "production_pressure"}
HVAC_FIELDS = {"terminal_air_gap", "terminal_mass_gap", "time_remaining", "thermal_recovery_pressure"}
EV_FIELDS = {"remaining_energy", "hours_to_departure", "deadline_pressure"}


def test_local_and_global_schemas_expose_p0(env):
    for mg, fields in (("mg1", MILL_FIELDS), ("mg2", HVAC_FIELDS), ("mg3", EV_FIELDS)):
        assert ESS_FIELDS <= set(env.obs_builder.schema(mg))
        assert fields <= set(env.obs_builder.schema(mg))
        global_names = set(env.global_state_builder.schema())
        assert {f"{mg}_{name}" for name in ESS_FIELDS} <= global_names
        assert {f"{mg}_{name}" for name in fields} <= global_names


def test_ess_guidance_matches_action_sign(env):
    env.reset(seed=11, deterministic=True)
    battery = env.cores["mg1"].battery
    battery.soc = battery.initial_soc - 0.1
    guidance = ess_terminal_guidance(battery, remaining_hours=4.0)
    assert guidance["required_avg_power_to_target"] < 0.0
    assert guidance["terminal_pressure"] == pytest.approx(
        abs(guidance["required_avg_power_to_target"])
    )


def test_terminal_observation_is_finite_and_keeps_gap(env):
    env.reset(seed=13, deterministic=True)
    env.cores["mg1"].battery.soc += 0.1
    flex = env._battery_flex(env.episode_length)["mg1"]
    assert flex["required_avg_power_to_target"] == pytest.approx(0.0)
    assert flex["terminal_pressure"] == pytest.approx(0.0)
    assert flex["terminal_soc_gap"] == pytest.approx(0.1)
```

Add exact mill and HVAC formula cases with a four-hour remainder, plus shape assertions comparing each built array to its schema length.

- [ ] **Step 2: Run the P0 tests and verify red**

Run:

```powershell
pytest -q tests/test_constraint_aware_observations.py tests/test_daily_soc_and_future_prices.py tests/test_dr_algorithm_adapters.py
```

Expected: schema assertions fail because ESS urgency, industrial production pressure, and HVAC terminal recovery fields are absent.

- [ ] **Step 3: Implement the pure guidance formulas**

Implement `ess_terminal_guidance` with action-sign convention and efficiencies:

```python
def ess_terminal_guidance(battery, remaining_hours: float) -> dict[str, float]:
    gap_soc = float(battery.soc) - float(battery.initial_soc)
    hours = max(float(remaining_hours), 0.0)
    if hours == 0.0:
        required_kw = pressure = 0.0
    elif gap_soc < 0.0:
        required_kw = gap_soc * battery.capacity_kwh / (battery.eta_charge * hours)
        pressure = abs(required_kw) / max(battery.max_charge_kw, 1e-12)
    else:
        required_kw = gap_soc * battery.capacity_kwh * battery.eta_discharge / hours
        pressure = abs(required_kw) / max(battery.max_discharge_kw, 1e-12)
    scale = battery.max_charge_kw if required_kw < 0.0 else battery.max_discharge_kw
    return {
        "terminal_soc_gap": gap_soc,
        "time_to_terminal": hours,
        "required_avg_power_kw": required_kw,
        "required_avg_power_to_target": required_kw / max(scale, 1e-12),
        "terminal_pressure": pressure,
    }
```

For the mill use `(initial_silo - silo) / hours + downstream_tph`, divided by `rated_production_tph`. For HVAC use `initial-current` gaps, heat capacities, remaining hours, direction-appropriate COP, and heating/cooling electric limits; expose signed gaps and an absolute un-clipped pressure. Return zeros for rate/pressure when no time remains.

- [ ] **Step 4: Wire matching local and global schemas**

Add ESS P0 fields next to `terminal_soc_gap` in every local schema and add prefixed ESS P0 fields per microgrid in global state. Replace EV public schema names `ev_remaining_energy` and `ev_hours_to_departure` with `remaining_energy` and `hours_to_departure`, while retaining raw component-state keys for metrics compatibility. Extend mill and HVAC schema/value lists in the same order in both builders. Ensure Transformer adapter feature expansion consumes the new dynamic schemas without changing its forecast slice.

- [ ] **Step 5: Run P0 and actor-adapter tests to verify green**

Run:

```powershell
pytest -q tests/test_constraint_aware_observations.py tests/test_daily_soc_and_future_prices.py tests/test_dr_algorithm_adapters.py tests/test_matd3_transformer.py tests/test_pnn_transformer.py tests/test_pnn_transformer_runtime_backport.py
```

Expected: all tests pass; every built observation/state is finite and has exactly the declared dimension.

- [ ] **Step 6: Commit P0 observations**

```powershell
git add core/constraint_observations.py components/industrial_mill.py components/commercial_hvac.py core/observation.py core/global_state.py envs/multi_microgrid_env.py tests/test_constraint_aware_observations.py tests/test_daily_soc_and_future_prices.py tests/test_dr_algorithm_adapters.py tests/test_pnn_transformer_runtime_backport.py
git commit -m "feat: add P0 constraint-aware observations"
```

### Task 4: Implement Policy-Delay-Safe Sparse Diagnostics

**Files:**
- Create: `algorithms/diagnostic_sampling.py`
- Modify: `algorithms/runtime_helpers.py`
- Create: `tests/test_diagnostic_sampling.py`
- Modify: `tests/test_dr_algorithm_adapters.py`

**Interfaces:**
- Produces: `DiagnosticSchedule(interval: int, policy_delay: int)` with `due(update_steps: int) -> bool`.
- Produces: `fast_plain_matd3_update(self, batch_size: int) -> dict[str, object]`.
- Produces: `fast_projected_update(self, batch_size: int) -> dict[str, object]`.
- Produces: `install_diagnostic_sampling(cls, *, variant: str, plain_actor: bool = False)`.
- Consumes: packed replay and action-selection helpers retained in `algorithms/runtime_helpers.py`.

- [ ] **Step 1: Write failing scheduler and hot-path tests**

Create tests for interval/policy-delay behavior and absence of ordinary-step diagnostics:

```python
@pytest.mark.parametrize(
    ("interval", "policy_delay", "expected"),
    [(20, 2, [20, 40]), (21, 2, [22, 42]), (5, 3, [6, 12])],
)
def test_schedule_only_fires_on_actor_updates(interval, policy_delay, expected):
    schedule = DiagnosticSchedule(interval, policy_delay)
    actual = [step for step in range(1, 45) if schedule.due(step)]
    assert actual[: len(expected)] == expected
    assert all(step % policy_delay == 0 for step in actual)


def test_ordinary_update_returns_no_metrics_and_skips_gradient_diagnostics(algorithm, monkeypatch):
    algorithm.diagnostic_schedule = DiagnosticSchedule(20, algorithm.policy_delay)
    monkeypatch.setattr(
        algorithm,
        "_actor_gradient_diagnostics",
        lambda *_: pytest.fail("gradient diagnostics called on ordinary update"),
    )
    metrics = algorithm.update(batch_size=2)
    assert metrics == {}


def test_diagnostic_update_contains_actor_critic_q_and_td_metrics(algorithm):
    algorithm.diagnostic_schedule = DiagnosticSchedule(2, algorithm.policy_delay)
    algorithm.update_steps = 1
    metrics = algorithm.update(batch_size=2)
    assert set(metrics) >= {
        "critic_loss", "actor_loss", "critic_grad_norm", "actor_grad_norm",
        "q_stats", "td_error_mean", "td_error_std", "td_error_rmse",
    }
    assert all({"mean", "std", "min", "max"} <= set(row) for row in metrics["q_stats"].values())
```

Add an AST/source test asserting `algorithms/diagnostic_sampling.py` contains no `bool(torch.isfinite(...).all())` or equivalent Python truth conversion.

- [ ] **Step 2: Run diagnostic tests and verify red**

Run:

```powershell
pytest -q tests/test_diagnostic_sampling.py tests/test_dr_algorithm_adapters.py
```

Expected: import failure for `algorithms.diagnostic_sampling` and existing algorithms emit metrics/gradient diagnostics on ordinary steps.

- [ ] **Step 3: Implement the schedule**

Use a stateful next-boundary scheduler so non-divisible intervals move to the next Actor step:

```python
class DiagnosticSchedule:
    def __init__(self, interval: int, policy_delay: int) -> None:
        if isinstance(interval, bool) or interval < 0:
            raise ValueError("logging.diagnostic_interval must be a non-negative integer")
        self.interval = int(interval)
        self.policy_delay = int(policy_delay)
        self.next_boundary = self.interval

    def due(self, update_steps: int) -> bool:
        if self.interval == 0 or update_steps % self.policy_delay != 0:
            return False
        if update_steps < self.next_boundary:
            return False
        while self.next_boundary <= update_steps:
            self.next_boundary += self.interval
        return True
```

- [ ] **Step 4: Implement fast ordinary and diagnostic paths**

Both update functions must:

1. Return `{}` before replay warmup.
2. Run Critic backward/optimizer every update.
3. Increment `update_steps` exactly once.
4. Run Actor and soft target updates only when `update_steps % policy_delay == 0`.
5. On ordinary updates, avoid `torch.isfinite`, `_gradient_norm`, Q/TD reporting reductions, parameter clones, and `.cpu()`/`float(tensor)` conversions.
6. On a due update, collect loss, Q `mean/std/min/max`, TD `mean/std/RMSE`, and gradient tensors, stack them, and perform one host conversion for Critic rows and one for Actor rows.
7. Preserve existing per-agent branch/layer and action/projection diagnostics only inside the due branch.

`fast_plain_matd3_update` calls flat actors without a `collect_diagnostics` keyword. `fast_projected_update` passes `collect_diagnostics=due` to structured actors. Both use the existing joint differentiable projection and the same TD3 target-noise and policy-loss equations as their current implementations.

- [ ] **Step 5: Run the diagnostic tests to verify green**

Run:

```powershell
pytest -q tests/test_diagnostic_sampling.py tests/test_dr_algorithm_adapters.py tests/test_single_dispatch.py
```

Expected: all selected tests pass, including the interval 21/policy delay 2 case and the empty ordinary-update payload.

- [ ] **Step 6: Commit sparse diagnostics**

```powershell
git add algorithms/diagnostic_sampling.py algorithms/runtime_helpers.py tests/test_diagnostic_sampling.py tests/test_dr_algorithm_adapters.py tests/test_single_dispatch.py
git commit -m "perf: sample training diagnostics sparsely"
```

### Task 5: Install Fast Updates for All Four Algorithms

**Files:**
- Modify: `algorithms/matd3_dispatch/plugin.py`
- Modify: `algorithms/pnn/plugin.py`
- Modify: `algorithms/matd3_transformer/plugin.py`
- Modify: `algorithms/pnn_transformer/plugin.py`
- Modify: `configs/algorithms/matd3_dispatch.yaml`
- Modify: `configs/algorithms/pnn.yaml`
- Modify: `configs/algorithms/matd3_transformer.yaml`
- Modify: `configs/algorithms/pnn_transformer.yaml`
- Modify: `configs/algorithms/matd3_dispatch_short_validation.yaml`
- Modify: `.github/workflows/dr-experimental-algorithms-smoke.yml`
- Create: `tests/test_four_algorithm_smoke.py`
- Modify: `tests/test_matd3_transformer.py`
- Modify: `tests/test_pnn_transformer.py`

**Interfaces:**
- Consumes: `install_diagnostic_sampling` from Task 4.
- Produces: all four plugin algorithm classes with `diagnostics_interval == 20` by default and a `DiagnosticSchedule` bound to their actual `policy_delay`.

- [ ] **Step 1: Write failing four-plugin integration tests**

Parameterize over the four slugs:

```python
@pytest.mark.parametrize("slug", ["matd3_dispatch", "pnn", "matd3_transformer", "pnn_transformer"])
def test_algorithm_uses_shared_diagnostic_sampling(slug, tmp_path):
    spec = get_algorithm_spec(slug)
    env_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(spec.build_environment_config(env_config))
    algorithm_config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    algorithm = spec.algorithm_class.from_env(env, algorithm_config, device="cpu", seed=5)
    assert algorithm.diagnostics_interval == 20
    assert algorithm.diagnostic_schedule.policy_delay == algorithm.policy_delay
    assert algorithm.action_dim == 2
    assert getattr(type(algorithm), "_diagnostic_sampling_installed", False)
```

Add a real replay/update/save/load loop with batch size two, assert a diagnostic payload appears at the configured boundary, and assert checkpoint action and observation schemas round-trip.

- [ ] **Step 2: Run the integration tests and verify red**

Run:

```powershell
pytest -q tests/test_four_algorithm_smoke.py
```

Expected: MATD3, PNN, and MATD3-Transformer lack the shared installer and most YAML files lack `logging.diagnostic_interval`.

- [ ] **Step 3: Install the correct update path and update configuration**

In each plugin, install the DR adapter before diagnostic sampling. Use:

```python
install_diagnostic_sampling(Matd3Dispatch, variant="matd3_dispatch", plain_actor=True)
install_diagnostic_sampling(PNN, variant="pnn")
install_diagnostic_sampling(Matd3Transformer, variant="matd3_transformer")
install_diagnostic_sampling(PnnTransformer, variant="pnn_transformer")
```

Move any old `algorithm.diagnostics_interval` setting to:

```yaml
logging:
  diagnostic_interval: 20
```

The installer reads the full config mapping's `logging` section, caches Actor parameter counts at initialization, converts replay to the existing packed `FastReplayBuffer`, and binds the flat or structured update function. Update the CI temporary-config builder to set `cfg["logging"]["diagnostic_interval"] = 1` rather than an algorithm key.

- [ ] **Step 4: Lock both Transformer horizons at four**

Add explicit assertions:

```python
assert config["algorithm"]["network"]["forecast_horizon"] == 4
assert algorithm.actors["mg1"].forecast_horizon == 4
```

Do not change `observation.future_horizon`; only the network prediction horizon is in scope.

- [ ] **Step 5: Run four-plugin and Transformer tests to verify green**

Run:

```powershell
pytest -q tests/test_four_algorithm_smoke.py tests/test_matd3_transformer.py tests/test_pnn_actor.py tests/test_pnn_transformer.py tests/test_pnn_transformer_runtime_backport.py
```

Expected: all four algorithms train from replay, emit diagnostics only at scheduled Actor updates, preserve raw action dimension two, and save/load checkpoints; both Transformer horizons equal four.

- [ ] **Step 6: Commit plugin integration**

```powershell
git add algorithms/matd3_dispatch/plugin.py algorithms/pnn/plugin.py algorithms/matd3_transformer/plugin.py algorithms/pnn_transformer/plugin.py configs/algorithms/matd3_dispatch.yaml configs/algorithms/pnn.yaml configs/algorithms/matd3_transformer.yaml configs/algorithms/pnn_transformer.yaml configs/algorithms/matd3_dispatch_short_validation.yaml .github/workflows/dr-experimental-algorithms-smoke.yml tests/test_four_algorithm_smoke.py tests/test_matd3_transformer.py tests/test_pnn_transformer.py
git commit -m "perf: enable sparse diagnostics for all algorithms"
```

### Task 6: Repository and Four-Algorithm Pipeline Verification

**Files:**
- Verify only; do not edit generated run artifacts.

**Interfaces:**
- Consumes: the four registered algorithms, canonical environment, and shared pipeline.
- Produces: fresh evidence for compile, unit, checkpoint, settlement, plotting, and layout claims.

- [ ] **Step 1: Verify no canonical MATD3 or user data file was changed by implementation**

Run:

```powershell
git diff 4302520 -- algorithms/matd3
git status --short -- data
```

Expected: the first command is empty; the second shows only the same four pre-existing user-modified data files.

- [ ] **Step 2: Compile the requested packages**

Run:

```powershell
python -m compileall algorithms envs
```

Expected: exit code 0.

- [ ] **Step 3: Run the complete test suite**

Run:

```powershell
pytest -q
```

Expected: exit code 0 with no failed tests.

- [ ] **Step 4: Build temporary ten-episode smoke configs outside the repository**

Create the directory `$smokeConfigRoot = Join-Path $env:TEMP 'instantaneous-p0-smoke-configs'`.
For each slug, copy its YAML into that directory and set:

```yaml
runtime:
  device: cpu
  cpu_threads: 1
  interop_threads: 1
training:
  episodes: 10
  test_episodes: 1
validation:
  enabled: true
  interval: 5
  max_days: 1
logging:
  diagnostic_interval: 20
  dashboard: false
```

Also set `algorithm.batch_size: 16`, `algorithm.warmup_steps: 0`, and `algorithm.replay_capacity: 10000` without changing network or horizon settings.

- [ ] **Step 5: Run all four shared pipelines**

Run once per slug:

```powershell
$smokeConfigRoot = Join-Path $env:TEMP 'instantaneous-p0-smoke-configs'
python scripts/run_experiment.py --algorithm matd3_dispatch --algorithm-config (Join-Path $smokeConfigRoot 'matd3_dispatch.yaml') --device cpu --seed 42 --episodes 10 --test-episodes 1 --validation-interval 5 --validation-days 1 --no-dashboard
python scripts/run_experiment.py --algorithm pnn --algorithm-config (Join-Path $smokeConfigRoot 'pnn.yaml') --device cpu --seed 42 --episodes 10 --test-episodes 1 --validation-interval 5 --validation-days 1 --no-dashboard
python scripts/run_experiment.py --algorithm matd3_transformer --algorithm-config (Join-Path $smokeConfigRoot 'matd3_transformer.yaml') --device cpu --seed 42 --episodes 10 --test-episodes 1 --validation-interval 5 --validation-days 1 --no-dashboard
python scripts/run_experiment.py --algorithm pnn_transformer --algorithm-config (Join-Path $smokeConfigRoot 'pnn_transformer.yaml') --device cpu --seed 42 --episodes 10 --test-episodes 1 --validation-interval 5 --validation-days 1 --no-dashboard
```

Expected: every command exits 0 and writes under `outputs/working_runs/<slug>/runs/`.

- [ ] **Step 6: Verify saved artifacts and formal-test checkpoint provenance**

For each `latest_run.json`, assert:

```python
assert len(training_history) == 10
assert run_metadata["algorithm_slug"] == slug
assert summary["test_days_count"] == 1
assert (run_dir / "checkpoints/best_model.pt").is_file()
assert (run_dir / "checkpoints/final_model.pt").is_file()
assert test_summary["checkpoint_used"].endswith("best_model.pt")
assert test_summary["checkpoint_selection_metric"] == "validation_team_profit_mean"
assert (run_dir / "test_daily_metrics.csv").is_file()
assert test_summary["coalition_milp_enabled"] is True
assert "equal_all_ir_satisfaction_rate" in test_summary
assert "shapley_all_ir_satisfaction_rate" in test_summary
assert "shapley_valid_day_count" in test_summary
assert len(list((run_dir / "figures").glob("*.png"))) == 15
assert len(list((run_dir / "figures").glob("*.pdf"))) == 15
```

- [ ] **Step 7: Review the final diff and commit any verification-only test corrections**

Run:

```powershell
git diff --check
git status --short
git diff --stat 4302520..HEAD
```

Expected: no whitespace errors, no generated run artifacts staged, no canonical MATD3 changes, and no user data files included in implementation commits.
