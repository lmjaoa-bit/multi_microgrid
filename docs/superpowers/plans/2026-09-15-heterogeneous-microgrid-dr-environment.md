# Heterogeneous Multi-Microgrid DR Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Follow TDD and commit after each coherent task.

**Goal:** Implement the approved heterogeneous industrial-commercial-residential demand-response environment, validate it with plain MATD3 first, and preserve the current pre-DR environment for algorithms that have not yet been migrated.

**Architecture:** Keep `algorithms/matd3/` untouched. Add three independent controllable-load components, make the environment execute DR before ESS/P2P/grid settlement, and add a differentiable joint action projector for the `matd3_dispatch` adapter so replay-time/actor-time projection matches rollout. Isolate the old environment/config/action contract so CrossGate, Transformer, PNN, and other advanced algorithms are not silently migrated in this phase.

**Tech Stack:** Python 3, NumPy, pandas, PyYAML, PyTorch, HiGHS/highspy, pytest, existing experiment/plugin infrastructure.

**Spec:** `docs/superpowers/specs/2026-09-15-heterogeneous-microgrid-dr-environment-design.md`

## Global Constraints

- Implementation branch: `codex/controllable-devices`.
- Baseline reference: `reproduce/pnn-transformer-8.99` at `ffb5e835858cb5b3963dac30fd53e48df5d66156`.
- Do not change canonical `algorithms/matd3/` actor, critic, replay buffer, update equations, or checkpoint behavior.
- Plain MATD3 (`matd3_dispatch`) is the only learning algorithm migrated to the new DR action contract in this phase.
- CrossGate and other advanced algorithms remain on a preserved pre-DR environment until separately migrated and retrained.
- New raw action per agent is exactly `[ess_dispatch_intent, dr_intent]`.
- New Critic effective action per agent is exactly `[ess_local_effective, ess_trade_effective, dr_effective]`.
- Shared team reward remains physical coalition profit/cost only; no DR-specific reward shaping is added.
- P2P internal payments remain excluded from team reward.
- Hard device constraints and terminal anti-reset constraints are enforced by projection/reachability, not routine reward penalties.
- All new daily episodes are natural 24-hour days.
- New P2P pair capacity is 75 kW.
- All ESS units initialize at 50% SOC and finish within ±0.03 SOC of 50%.
- Final verification must include `pytest -q`, a deterministic 24-hour environment rollout, a short MATD3 pipeline smoke run, and representative-day MILP checks.

## Implementation Refinements Discovered During Planning

1. `matd3_dispatch` currently hard-codes a one-dimensional raw Actor even though its Critic consumes two projected ESS controls. Therefore the new environment cannot be implemented by changing only `env.action_dim`; `matd3_dispatch` must migrate to a two-dimensional raw Actor and a three-dimensional effective Critic control per agent.
2. NASA POWER cannot reliably distinguish weather at local P2P distances: its documented meteorological resolution is roughly 0.5×0.625 degrees and solar resolution roughly 1×1 degree. The new data builder must not pretend that several 10-km points are independent NASA observations. Use a provider abstraction and a high-resolution regional historical provider for the new dataset; keep the old NASA-generated files untouched for legacy experiments.
3. `algorithms/milp_mpc` is a legacy one-action controller. Do not silently adapt it in this phase. The full-day DR MILP oracle is updated for evaluation/validation; rolling MILP-MPC remains attached to the legacy environment until a later algorithm-migration task.

---

## File Map

### New files
- `configs/cooperative_legacy_pre_dr.yaml` — exact pre-DR environment snapshot for unmigrated algorithms.
- `components/controllable_load.py` — thin DR protocol/shared helpers.
- `components/industrial_mill.py` — MG1 mill/silo model.
- `components/commercial_hvac.py` — MG2 2R2C HVAC model and terminal reachability.
- `components/ev_charging.py` — MG3 aggregate EV model.
- `core/joint_action_projection.py` — differentiable raw `[ESS, DR]` to effective `[ESS local, ESS trade, DR]` projection.
- `tests/test_controllable_loads.py` — device-model unit tests.
- `tests/test_joint_action_projection.py` — differentiable projector tests.
- `tests/test_dr_environment.py` — integrated environment physics/terminal tests.
- `tests/test_regional_weather_data.py` — data-builder/provider tests.

### Modified files
- `configs/cooperative.yaml`
- `configs/action_designs.yaml`
- `configs/environment_profiles.yaml`
- `configs/experiment_profiles.yaml`
- `core/action_codec.py`
- `core/types.py`
- `core/config_validation.py`
- `core/data_manager.py`
- `core/profiles.py`
- `core/forecast.py`
- `core/observation.py`
- `core/global_state.py`
- `core/microgrid_core.py`
- `envs/multi_microgrid_env.py`
- `scripts/build_spatial_market_data.py`
- `utils/metrics_collector.py`
- `algorithms/matd3_dispatch/algorithm.py`
- `experiments/profiles.py`
- `optimization/common_constraints.py`
- `evaluation/coalition_evaluator.py` only if data/result fields must be extended without changing settlement semantics.
- Existing tests that hard-code raw action dimension 1 or terminal mode `none`.

---

### Task 1: Isolate the legacy environment and define the new DR experiment contract

**Files:**
- Create: `configs/cooperative_legacy_pre_dr.yaml`
- Modify: `configs/cooperative.yaml`
- Modify: `configs/action_designs.yaml`
- Modify: `configs/environment_profiles.yaml`
- Modify: `configs/experiment_profiles.yaml`
- Modify: `experiments/profiles.py`
- Modify: `tests/test_experiment_profiles.py`

**Contract:**

```yaml
single_dispatch:
  raw_actions: [signed_dispatch_intent]

single_dispatch_dr:
  raw_actions: [signed_dispatch_intent, dr_intent]
```

`matd3` selects `single_dispatch_dr` + the new `cooperative_standard`; `crossgate` selects legacy `single_dispatch` + `cooperative_legacy_pre_dr`.

- [ ] **Step 1: Preserve the old physical config before editing**

Copy the current `configs/cooperative.yaml` byte-for-byte to `configs/cooperative_legacy_pre_dr.yaml`.

- [ ] **Step 2: Write failing profile-isolation tests**

Add tests equivalent to:

```python
def test_matd3_and_crossgate_are_intentionally_on_different_environment_generations():
    matd3 = profile("matd3")
    crossgate = profile("crossgate")
    assert matd3["action_design"] == "single_dispatch_dr"
    assert crossgate["action_design"] == "single_dispatch"
    assert matd3["environment"] == "cooperative_standard"
    assert crossgate["environment"] == "cooperative_legacy_pre_dr"


def test_dr_action_design_has_two_raw_controls():
    designs = catalog("action_designs")
    assert designs["single_dispatch_dr"]["raw_actions"] == [
        "signed_dispatch_intent", "dr_intent"
    ]
```

- [ ] **Step 3: Decouple action-design ID from ESS control mode**

`ess_sharing.control_mode` remains `single_dispatch`; add a top-level config field:

```yaml
action_design: single_dispatch_dr
```

Legacy config uses:

```yaml
action_design: single_dispatch
```

Update `experiment_contract()` so action design comes from `env_config["action_design"]`, while `ess_sharing.control_mode` continues to describe ESS physics only.

- [ ] **Step 4: Run targeted tests**

```powershell
pytest tests/test_experiment_profiles.py -q
```

Expected: PASS with different environment hashes/contracts by design.

- [ ] **Step 5: Commit**

```powershell
git add configs/cooperative_legacy_pre_dr.yaml configs/action_designs.yaml configs/environment_profiles.yaml configs/experiment_profiles.yaml experiments/profiles.py tests/test_experiment_profiles.py
git commit -m "feat: isolate legacy and DR environment contracts"
```

---

### Task 2: Implement MG1, MG2, and MG3 controllable-load models independently

**Files:**
- Create: `components/controllable_load.py`
- Create: `components/industrial_mill.py`
- Create: `components/commercial_hvac.py`
- Create: `components/ev_charging.py`
- Create: `tests/test_controllable_loads.py`

**Thin common interface:**

```python
class ControllableLoad(Protocol):
    def reset(self, episode_frame: pd.DataFrame, base_load_kw: np.ndarray) -> None: ...
    def state(self, t: int) -> dict[str, float]: ...
    def feasible_bounds(self, t: int) -> tuple[float, float]: ...
    def plan(self, raw_action: float, t: int) -> dict[str, Any]: ...
    def commit(self, plan: Mapping[str, Any]) -> dict[str, float]: ...
```

The device classes may expose additional typed helpers; do not force their internal physics into one generic state machine.

- [ ] **Step 1: Write MG1 failing tests**

Cover reset, physical bounds, terminal reachability, and exact dynamics:

```python
def test_mill_inventory_dynamics_and_terminal_reachability():
    mill = IndustrialMillDR.from_config(MG1_CONFIG, delta_t=1.0)
    mill.reset(day_frame, base_load)
    assert mill.silo_t == pytest.approx(10.53)
    lo, hi = mill.feasible_bounds(0)
    assert 0.0 <= lo <= hi <= 1.0
    plan = mill.plan(1.0, 0)
    mill.commit(plan)
    assert mill.silo_t == pytest.approx(10.53 + 1.186 - 0.975)
```

At `t=23`, assert the feasible duty interval guarantees `|S24-10.53| <= 0.20` whenever the terminal set is physically reachable.

- [ ] **Step 2: Implement `IndustrialMillDR`**

Use analytic current-step bounds from intersection of:

```text
physical silo next-state range
terminal reachable range after remaining hours
0 <= duty <= 1
```

Return requested/effective duty, mill kW, production, downstream consumption, next silo, projection delta, and terminal-feasibility flags.

- [ ] **Step 3: Write MG3 failing tests**

```python
def test_ev_minimum_charge_becomes_forced_near_deadline():
    ev = EVChargingDR.from_config(MG3_CONFIG, delta_t=1.0)
    ev.reset(day_frame, base_load)
    lo0, hi0 = ev.feasible_bounds(0)
    assert lo0 < hi0
    for t in range(6):
        ev.commit(ev.plan(-1.0, t))
    lo6, hi6 = ev.feasible_bounds(6)
    assert lo6 == pytest.approx(hi6)
    assert lo6 > 0.0
```

Also assert no V2G, no charging after 07:00, SOC never exceeds 80%, and `deadline_pressure == 0` after departure.

- [ ] **Step 4: Implement `EVChargingDR`**

Use:

```python
minimum_kw = max(
    0.0,
    (remaining_battery_kwh - eta * max_kw * max(remaining_steps - 1, 0) * dt)
    / (eta * dt),
)
maximum_kw = min(max_kw, remaining_battery_kwh / (eta * dt))
```

Map raw `[-1, 1]` to proposed charging `[0, max_kw]`, then project into `[minimum_kw, maximum_kw]`.

- [ ] **Step 5: Write MG2 failing tests**

Cover weather-dependent reset, signed HVAC, comfort projection, terminal thermal feasibility, and physical cap:

```python
def test_hvac_initial_temperature_uses_midnight_outdoor_temperature():
    hvac = CommercialHVACDR.from_config(MG2_CONFIG, delta_t=1.0)
    hvac.reset(day_frame_with_midnight_tout_5c, base_load)
    assert hvac.t_air == pytest.approx(19.4)
    assert hvac.t_mass == pytest.approx(21.2)


def test_hvac_signed_action_and_terminal_band():
    hvac.reset(day_frame, base_load)
    for t in range(24):
        lo, hi = hvac.feasible_bounds(t)
        assert -35.0 <= lo <= hi <= 35.0
        plan = hvac.plan(0.0, t)
        hvac.commit(plan)
    assert abs(hvac.t_air - hvac.initial_t_air) <= 1.0 + 1e-6
    assert abs(hvac.t_mass - hvac.initial_t_mass) <= 0.3 + 1e-6
```

- [ ] **Step 6: Implement `CommercialHVACDR` 2R2C dynamics**

Use explicit one-hour discrete dynamics with thermal HVAC power:

```python
q_hvac_kw = p_electric_kw * cop_heat if p_electric_kw >= 0 else p_electric_kw * cop_cool
```

Solar gain:

```python
q_solar_kw = 0.025 * ghi_w_per_m2
q_solar_air = 0.20 * q_solar_kw
q_solar_mass = 0.80 * q_solar_kw
```

Internal gains are 0.50 or 0.20 times non-HVAC base load according to occupancy.

- [ ] **Step 7: Implement exact terminal thermal reachability with a small HiGHS LP**

For each current hour, solve two continuous LPs over remaining `q_hvac` and thermal states to obtain minimum/maximum feasible current thermal control while enforcing:

```text
current physical HVAC thermal bounds
future hourly comfort bounds
terminal T_air band ±1.0 C around initial
terminal T_mass band ±0.3 C around initial
```

Map the resulting thermal bounds back to signed electric kW via the corresponding COP. If the LP is infeasible, use the closest one-step physical bound and set `terminal_thermal_infeasible=True`; never exceed 35 kW.

- [ ] **Step 8: Run unit tests**

```powershell
pytest tests/test_controllable_loads.py -q
```

- [ ] **Step 9: Commit**

```powershell
git add components/controllable_load.py components/industrial_mill.py components/commercial_hvac.py components/ev_charging.py tests/test_controllable_loads.py
git commit -m "feat: add heterogeneous controllable load models"
```

---

### Task 3: Add the two-dimensional raw action codec and differentiable joint projector

**Files:**
- Modify: `core/action_codec.py`
- Modify: `core/types.py`
- Create: `core/joint_action_projection.py`
- Create: `tests/test_joint_action_projection.py`
- Modify: `algorithms/matd3_dispatch/algorithm.py`
- Modify: `tests/test_plain_single_dispatch.py`

**Raw contract:**

```text
raw per agent = [ESS intent, DR intent]
```

**Effective Critic contract:**

```text
effective per agent = [ESS local, ESS trade, DR effective]
```

- [ ] **Step 1: Write codec tests for both generations**

```python
def test_legacy_codec_stays_one_dimensional():
    codec = ActionCodec("single_dispatch")
    assert codec.action_dim == 1
    assert codec.decode([0.25]) == {"battery": 0.25}


def test_dr_codec_decodes_two_raw_controls():
    codec = ActionCodec("single_dispatch_dr")
    assert codec.action_dim == 2
    assert codec.decode([0.25, -0.5]) == {"battery": 0.25, "dr": -0.5}
```

- [ ] **Step 2: Implement codec/type changes without changing legacy behavior**

`DecodedAction` must make `dr` optional for legacy calls but required by the new action design.

- [ ] **Step 3: Write differentiable projector tests**

Build a synthetic global state containing DR safe bounds. Assert:

```python
raw = torch.tensor([[[0.4, -1.0], [-0.2, 0.8], [0.1, -0.7]]], requires_grad=True)
effective, plans = projector(state, raw)
assert effective.shape == (1, 3, 3)
effective.sum().backward()
assert raw.grad is not None
assert torch.isfinite(raw.grad).all()
```

Also assert MG1/MG3 raw DR map `[-1,1] -> [0,1]`, MG2 remains signed `[-1,1]`, and all are clipped to state-stored safe feasible bounds.

- [ ] **Step 4: Implement `JointActionProjector`**

Internally call existing `SingleDispatchProjector` with `raw[..., 0:1]`. Build the DR channel from `raw[..., 1]` and global-state fields:

```text
mg1_dr_min, mg1_dr_max
mg2_dr_min, mg2_dr_max
mg3_dr_min, mg3_dr_max
```

All stored DR bounds are normalized to the same canonical scale consumed by the Critic:

```text
MG1 duty fraction: 0..1
MG2 signed electric power / 35 kW: -1..1
MG3 charging power / 93.6 kW: 0..1
```

- [ ] **Step 5: Migrate `matd3_dispatch` only**

Change the adapter to:

```text
raw actor action_dim = 2
critic physical action_dim = 3 per agent
projector = JointActionProjector
checkpoint action_semantics = single_dispatch_dr_v1
```

Do not modify canonical `algorithms/matd3/`.

Update actor diagnostics to report both raw channels, including `dispatch_action_grad_rms`, `dr_action_grad_rms`, and separate saturation rates.

- [ ] **Step 6: Run tests**

```powershell
pytest tests/test_joint_action_projection.py tests/test_plain_single_dispatch.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add core/action_codec.py core/types.py core/joint_action_projection.py algorithms/matd3_dispatch/algorithm.py tests/test_joint_action_projection.py tests/test_plain_single_dispatch.py
git commit -m "feat: add joint ESS and DR action projection"
```

---

### Task 4: Build geographically coherent regional weather data with local temperature

**Files:**
- Modify: `scripts/build_spatial_market_data.py`
- Modify: `core/data_manager.py`
- Create: `tests/test_regional_weather_data.py`
- Generate: `data/regional_market_2018_train.csv`
- Generate: `data/regional_market_2018_val.csv`
- Generate: `data/regional_market_2018_test.csv`
- Generate/update: `data/regional_market_2018_metadata.json`

**Provider decision:** Use a high-resolution historical source for the new regional files instead of pretending nearby NASA POWER points are independent. The script keeps provider metadata explicit. Use Open-Meteo Historical Weather API with hourly `temperature_2m`, `shortwave_radiation`, and `wind_speed_100m`, local timezone `Asia/Shanghai`, and 2018 data. The provider currently documents 9-km historical ECMWF IFS coverage from 2017 onward; keep the exact returned model/grid metadata in the JSON sidecar.

Use three nearby synthetic-node coordinates within one Yancheng-area regional scenario, separated enough to sample distinct high-resolution grid cells but close enough to remain a regional distribution community:

```python
LOCATIONS = {
    "mg1": {"latitude": 33.30, "longitude": 120.12},
    "mg2": {"latitude": 33.38, "longitude": 120.20},
    "mg3": {"latitude": 33.31, "longitude": 120.28},
}
```

These are scenario nodes, not claims about named physical facilities.

- [ ] **Step 1: Write parser/provider tests using mocked HTTP payloads**

Assert the builder produces:

```text
mg1_ghi_w_per_m2
mg1_wind_speed_mps
mg1_temperature_c
mg2_ghi_w_per_m2
mg2_wind_speed_mps
mg2_temperature_c
mg3_ghi_w_per_m2
mg3_wind_speed_mps
mg3_temperature_c
```

and preserves the existing market/load columns and 2018 timestamp alignment.

- [ ] **Step 2: Refactor the builder into fetch + normalize + join helpers**

No third-party HTTP dependency is required; keep `urllib.request`.

- [ ] **Step 3: Extend `DataManager` validation**

When a microgrid config declares a local temperature column, require it and allow physically valid negative Celsius values. Do not reuse the current nonnegative validation rule for temperature.

- [ ] **Step 4: Generate all three regional split files**

```powershell
python scripts/build_spatial_market_data.py
```

- [ ] **Step 5: Verify data quality**

Add/run checks that timestamps are contiguous within valid days, no required values are NaN/inf, and the three local weather series are not byte-identical across the full year.

- [ ] **Step 6: Commit**

```powershell
git add scripts/build_spatial_market_data.py core/data_manager.py tests/test_regional_weather_data.py data/regional_market_2018_train.csv data/regional_market_2018_val.csv data/regional_market_2018_test.csv data/regional_market_2018_metadata.json
git commit -m "feat: add regional high resolution weather dataset"
```

---

### Task 5: Update configuration and base-load/forecast semantics

**Files:**
- Modify: `configs/cooperative.yaml`
- Modify: `core/config_validation.py`
- Modify: `core/profiles.py`
- Modify: `core/forecast.py`
- Modify tests: `tests/test_daily_soc_and_future_prices.py`
- Add focused configuration assertions to `tests/test_dr_environment.py`

**New base-load targets:**

```text
MG1 base mean = 48.6 kW
MG2 non-HVAC mean = 75.0 kW initially; final calibration must validate total 79-81 kW
MG3 non-EV mean = 57.52 kW
```

**Fixed config changes:**

```text
pair_capacity_kw = 75
terminal_soc.mode = hard
battery.initial_soc = 0.50 for all three MGs
```

- [ ] **Step 1: Write failing config validation tests**

Require exactly one DR type per new microgrid:

```text
mg1: industrial_mill
mg2: commercial_hvac
mg3: ev_charging
```

Validate all approved numeric ranges, terminal bands, EV deadline, COPs, silo parameters, and local temperature column.

- [ ] **Step 2: Update `cooperative.yaml` with the approved parameters**

Do not change tariffs, RES capacities, ESS capacities/powers, grid limits, reward scale, unmet cost, or curtailment cost except the already approved values.

- [ ] **Step 3: Make `HeterogeneousProfileModel` explicitly produce base load**

The exogenous series for the new environment represents base load. Keep returned compatibility key `load_kw`, but also return `base_load_kw` and make them equal before DR is applied.

- [ ] **Step 4: Add temperature forecasting**

`SharedForecastProvider` adds local outdoor temperature for MG2. First implementation uses zero temperature forecast noise:

```yaml
forecast_sigma_1h:
  temperature: 0.0
forecast_error_max:
  temperature: 0.0
```

Do not change the 24-hour forecast horizon in this phase.

- [ ] **Step 5: Update forecast tests**

At the natural-day boundary, future `base_load`, PV, wind, price, temperature, and validity mask must all mask consistently beyond the episode.

- [ ] **Step 6: Commit**

```powershell
git add configs/cooperative.yaml core/config_validation.py core/profiles.py core/forecast.py tests/test_daily_soc_and_future_prices.py tests/test_dr_environment.py
git commit -m "feat: configure heterogeneous DR physical system"
```

---

### Task 6: Integrate DR into the environment execution flow and state commits

**Files:**
- Modify: `core/microgrid_core.py`
- Modify: `envs/multi_microgrid_env.py`
- Create/modify: `tests/test_dr_environment.py`
- Modify: `tests/test_cooperative_physics.py`

- [ ] **Step 1: Write failing execution-order tests**

Construct deterministic actions and assert:

```python
assert plan["actual_load_kw"] == pytest.approx(
    plan["base_load_kw"] + plan["flexible_load_kw"]
)
assert plan["load_kw"] == pytest.approx(plan["actual_load_kw"])
```

Create a case where DR increases load and verify ESS local-first dispatch uses the increased actual load, not the old base load.

- [ ] **Step 2: Add device construction/reset**

`MultiMicrogridEnv` creates one DR component per microgrid from config and calls device `reset()` after the natural day is chosen. Legacy config creates no DR components and keeps one-dimensional actions unchanged.

- [ ] **Step 3: Reorder `step()`**

Implement exactly:

```text
decode raw -> DR plan/project -> actual load -> ESS projection -> P2P -> grid -> commit
```

Pass an exogenous dictionary containing the new actual `load_kw` into `MicrogridCore.plan_local()` so existing ESS/P2P logic works on the correct load.

- [ ] **Step 4: Preserve load service priority over optional ESS charging**

When grid/P2P limits create a shortage, reduce optional ESS charging before allowing `unmet_load_kw`.

If true unmet load still remains after all supply limits, allocate service to base load first and record `dr_supply_shortfall_kw`. Only this physical-infeasibility path may reduce a committed DR energy state; price alone must never curtail a projected DR action.

- [ ] **Step 5: Commit device states after settlement**

Commit ESS, silo, thermal state, and EV SOC exactly once per hour. Verify no state changes during observation building or Critic projection.

- [ ] **Step 6: Add terminal tests**

Run intentionally adversarial actions for 24 hours and assert:

```text
all ESS terminal SOC within 0.50 ± 0.03
MG1 silo within 10.53 ± 0.20 t
MG2 terminal T_air within initial ±1.0 C unless explicit infeasible flag
MG2 terminal T_mass within initial ±0.3 C unless explicit infeasible flag
MG3 SOC at 07:00 = 0.80 within numerical tolerance
```

- [ ] **Step 7: Run tests**

```powershell
pytest tests/test_dr_environment.py tests/test_cooperative_physics.py -q
```

- [ ] **Step 8: Commit**

```powershell
git add core/microgrid_core.py envs/multi_microgrid_env.py tests/test_dr_environment.py tests/test_cooperative_physics.py
git commit -m "feat: integrate DR before ESS and P2P settlement"
```

---

### Task 7: Extend local observations and centralized Critic state

**Files:**
- Modify: `core/observation.py`
- Modify: `core/global_state.py`
- Modify: `tests/test_dr_environment.py`
- Modify: `tests/test_plain_single_dispatch.py`

- [ ] **Step 1: Write schema tests**

MG1 local observation must include silo ratio/gap, mill feasible bounds, and time remaining.
MG2 must include T_air, T_mass, T_out, comfort margins, occupancy/weekend, and HVAC feasible bounds.
MG3 must include EV SOC/gap, remaining energy, hours to departure, availability, charge bounds, and deadline pressure.

Peer Actor information remains energy summaries only; assert MG1 local observation does not contain peer `T_air`, peer silo, or peer EV SOC.

- [ ] **Step 2: Add normalized DR states/bounds to `GlobalStateBuilder`**

The centralized state contains all three devices' internal states and the normalized terminal-safe DR bounds required by `JointActionProjector`.

Name the projector-critical fields consistently:

```text
mg1_dr_min, mg1_dr_max
mg2_dr_min, mg2_dr_max
mg3_dr_min, mg3_dr_max
```

- [ ] **Step 3: Add MG2 future temperature to local and centralized forecasts**

Actor sees its own future outdoor temperature; Critic sees the same forecast realization used by the Actor. Do not leak future DR actions.

- [ ] **Step 4: Assert observation/state dimensions are dynamically consumed**

`matd3_dispatch` Actor final layer output is 2; Critic joint effective input is 9 (=3 controls × 3 agents). No fixed observation dimension should be introduced.

- [ ] **Step 5: Run tests**

```powershell
pytest tests/test_dr_environment.py tests/test_plain_single_dispatch.py tests/test_daily_soc_and_future_prices.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add core/observation.py core/global_state.py tests/test_dr_environment.py tests/test_plain_single_dispatch.py tests/test_daily_soc_and_future_prices.py
git commit -m "feat: expose DR state to actors and centralized critics"
```

---

### Task 8: Expand diagnostics without changing the team objective

**Files:**
- Modify: `utils/metrics_collector.py`
- Modify: `tests/test_metrics_recording.py`
- Modify: `envs/multi_microgrid_env.py` only for explicit info payload fields.

- [ ] **Step 1: Replace obsolete second-action semantics**

Remove `sharing_offer_ratio_raw` interpretation for new DR runs. Record:

```text
raw_ess_action
raw_dr_action
effective_ess_dispatch
effective_dr_action
ess_projection_delta
dr_projection_delta
ess_projected
dr_projected
```

Legacy runs may retain legacy fields only when action design is `single_dispatch`.

- [ ] **Step 2: Add common electrical diagnostics**

Per MG/hour record:

```text
base_load_kw, flexible_load_kw, actual_load_kw
pv_kw, wind_kw, renewable_kw
ess_charge_kw, ess_discharge_kw, ess_to_local_kw
ess_p2p_offer_kw, ess_p2p_accepted_kw, ess_p2p_unmatched_kw
p2p_buy_kw, p2p_sell_kw
grid_import_kw, grid_export_kw
unmet_load_kw, curtailment_kw, power_balance_residual_kw
```

- [ ] **Step 3: Add device-specific diagnostics**

MG1: requested/effective duty, production, silo level/ratio/bounds, feasible duty, terminal gap.
MG2: signed HVAC request/effective power, heating/cooling electrical kW, T_air/T_mass/T_out, comfort margins, occupancy, feasible bounds, comfort and terminal infeasibility flags.
MG3: availability, requested/effective charging, SOC, target/gap, remaining energy/time, min/max, deadline pressure, departure target met.

- [ ] **Step 4: Add P2P capacity diagnostics**

Per day compute pair saturation rate, saturated pair-hours, and blocked/unserved P2P demand attributable to pair capacity where deterministically measurable.

Do not invent source attribution such as `grid_to_flexible_load_kwh` unless an explicit deterministic accounting rule is implemented and documented. Physical power balance diagnostics take priority over cosmetic source labels.

- [ ] **Step 5: Run metrics tests**

```powershell
pytest tests/test_metrics_recording.py tests/test_dr_environment.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add utils/metrics_collector.py envs/multi_microgrid_env.py tests/test_metrics_recording.py tests/test_dr_environment.py
git commit -m "feat: add heterogeneous DR diagnostics"
```

---

### Task 9: Extend the full-day MILP oracle to the new DR physics

**Files:**
- Modify: `optimization/common_constraints.py`
- Modify: `evaluation/coalition_evaluator.py` only as needed for extended result fields.
- Modify: `tests/test_optimization_and_allocation.py`
- Add MILP assertions to: `tests/test_dr_environment.py`

**Scope:** Update the full-day oracle used for physical validation and coalition evaluation. Do not migrate `algorithms/milp_mpc` yet.

- [ ] **Step 1: Extend `DispatchData`**

Carry base load, renewable generation, local temperature/GHI needed by HVAC, and timestamps/calendar state. DR is endogenous to the MILP.

- [ ] **Step 2: Add MG1 linear constraints**

Add mill duty, flexible load, silo dynamics/bounds, and terminal silo band.

- [ ] **Step 3: Add MG3 linear constraints**

Add EV charge, SOC/energy dynamics, availability, charger cap, and 07:00 target.

- [ ] **Step 4: Add MG2 thermal constraints**

Use signed thermal HVAC variables with bounded heating/cooling conversion. If separate heating and cooling nonnegative variables are used, add a binary mode to prevent simultaneous heating/cooling. Add 2R2C dynamics, hourly comfort bands, weather-based initial state, and terminal thermal bands.

- [ ] **Step 5: Retain existing ESS/grid/P2P constraints**

Use new 75-kW pair capacity and hard 50% ESS cyclic target. Keep P2P internal payments out of the physical coalition objective.

- [ ] **Step 6: Write representative-day oracle tests**

For at least one winter, spring, and summer day assert:

```text
solver optimal/feasible
unmet load approximately zero
all device terminal constraints satisfied
pair flows <= 75 kW
grid import/export <= 200 kW
nodal power balance residual approximately zero
```

- [ ] **Step 7: Re-run P2P sensitivity helper/check**

Evaluate 50/75/100 kW on the same representative days and save/print the comparison for manual review. The formal config remains 75 kW unless results materially contradict the prior screening.

- [ ] **Step 8: Commit**

```powershell
git add optimization/common_constraints.py evaluation/coalition_evaluator.py tests/test_optimization_and_allocation.py tests/test_dr_environment.py
git commit -m "feat: extend MILP oracle to heterogeneous DR"
```

---

### Task 10: Calibrate MG2 baseline load and perform end-to-end verification

**Files:**
- Modify `configs/cooperative.yaml` only if MG2 `non_hvac_target_mean_kw` requires calibration.
- Add a small reproducible calibration/diagnostic entry point under `scripts/` if no existing diagnostics script cleanly supports it.
- Update tests only for final calibrated expected range, not a magic exact annual kWh.

- [ ] **Step 1: Run baseline thermostat calibration on the regenerated train year**

Use the final local MG2 temperature/GHI data, weather-dependent daily initial state, and terminal thermal rule. Calculate annual-average non-HVAC, HVAC, and total actual load.

Acceptance:

```text
79 kW <= annual average MG2 total load <= 81 kW
comfort infeasible hours near zero
35-kW cap hit rate low and explicitly reported
```

If total average is outside 79–81 kW, change only MG2 non-HVAC target and rerun. Do not change 2R2C parameters, COP, comfort bands, or HVAC power solely to force the mean.

- [ ] **Step 2: Run full unit/integration suite**

```powershell
pytest -q
```

All tests must pass before claiming implementation complete.

- [ ] **Step 3: Run deterministic 24-hour physical rollout**

Use zero/neutral or scripted actions and verify no NaN/inf, no power-balance residual, valid terminal states, and action/observation dimensions.

- [ ] **Step 4: Run a short plain-MATD3 pipeline smoke test**

Use a temporary short-test override/config that exercises training, validation, formal test, coalition MILP evaluation, Equal/Shapley settlement, summary generation, and plotting without changing the long-run research hyperparameters.

Primary command remains:

```powershell
python scripts/run_experiment.py --algorithm matd3_dispatch --device cuda
```

For the smoke run, use the repository's existing short-run CLI/config override mechanism rather than editing the production episode count in place.

Verify:

```text
selected algorithm = matd3_dispatch
environment action_dim = 2
Actor output dimension = 2
Critic effective joint action dimension = 9
best_model.pt is used for formal test
results land under outputs/working_runs/matd3_dispatch/runs/
coalition evaluation completes on the new full-day DR MILP
```

- [ ] **Step 5: Inspect physical acceptance metrics**

At minimum report:

```text
team cost/profit
per-MG cost/profit
renewable curtailment and utilization
unmet load
P2P energy and 75-kW saturation rate
ESS terminal SOC errors
MG1 silo terminal error/projection rate
MG2 comfort violations, terminal thermal errors, HVAC energy/cap-hit rate
MG3 departure SOC/target success and forced-charge rate
power-balance residual maximum
```

- [ ] **Step 6: Confirm legacy isolation**

Instantiate CrossGate on `cooperative_legacy_pre_dr` and verify it still sees action_dim=1 without modifying its network. Do not compare performance with the new MATD3 environment; this is compatibility only.

- [ ] **Step 7: Final commit**

```powershell
git add configs/cooperative.yaml scripts/ tests/
git commit -m "test: validate heterogeneous DR environment end to end"
```

---

## Completion Criteria

Implementation is complete only when all of the following are true:

1. `configs/cooperative.yaml` is the new approved DR environment and `configs/cooperative_legacy_pre_dr.yaml` preserves the prior environment.
2. Plain MATD3 emits two raw controls per agent and trains through the same raw-to-effective projection used by rollout.
3. The centralized Critic receives 3 effective controls per agent; no rollout/training action semantic mismatch remains.
4. MG1 silo, MG2 thermal, MG3 EV, and all ESS states obey their hard terminal rules or expose an explicit physical-infeasibility diagnostic.
5. No DR constraint is duplicated as routine reward shaping.
6. P2P pair capacity is 75 kW and all P2P/grid/power-balance invariants hold.
7. MG2 final annual-average actual load is 79–81 kW after calibration on the final regional dataset.
8. Local temperature is present in the final dataset and future temperature is available to MG2 observations/centralized state.
9. Metrics record raw vs projected actions and device-specific physics sufficient to diagnose irrational policies.
10. Full-day DR MILP is feasible on representative winter/spring/summer days and obeys the same physical constraints as the environment.
11. `pytest -q` passes and the short MATD3 shared pipeline completes end-to-end.
12. Advanced algorithms are not silently declared compatible; migration/retraining is a separate follow-up phase.
