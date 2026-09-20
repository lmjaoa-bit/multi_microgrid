# Feasible DR Transformer V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated V2 hierarchical Transformer variant that preserves V1's zero-projection/EV-reliability behavior while adding an adaptive HVAC terminal-recovery anchor, an economic EV charging reference, and diagnostics that separate learned preference from rule-based feasibility support.

**Architecture:** A local `AdaptiveDRTransform` extends V1's hard feasible interval and EV latest-feasible floor with two independent reference mechanisms. MG2 blends its V1 feasible action toward a smooth terminal-recovery anchor; MG3 maps latent residuals around a price- and system-net-load-aware reference while retaining the V1 floor. A V2-local actor, training update, checkpoint identity, configurations, and saved-result gate checker keep the experiment isolated from canonical MATD3, the original Transformer, V1, shared physics, and shared rewards.

**Tech Stack:** Python 3.13, PyTorch, NumPy, PyYAML, pytest, standard-library JSON/CSV, existing shared MATD3 experiment pipeline.

**Spec:** `docs/superpowers/specs/2026-09-20-feasible-dr-v2-design.md`

## Global Constraints

- The algorithm slug is exactly `matd3_hierarchical_transformer_feasible_dr_v2`.
- The display name is exactly `MATD3 Hierarchical Transformer Feasible DR V2`.
- The action semantics string is exactly `single_dispatch_dr_adaptive_reference_v2`.
- Do not modify `algorithms/matd3/`, `algorithms/matd3_hierarchical_transformer/`, or `algorithms/matd3_hierarchical_transformer_feasible_dr/`.
- Do not modify canonical, parent-Transformer, or V1 YAML files.
- Do not modify environment physics, constraints, rewards, P2P clearing, settlement, action dimensions, observation dimensions, field order, or data splits.
- MG1 architecture and action mapping remain identical to V1.
- MG2/MG3 rollout exploration and target-policy noise remain in latent space before adaptive mapping.
- Replay stores the two-dimensional environment-compatible action, never the latent preference.
- Default V2 network and learning hyperparameters remain identical to V1.
- Normal outputs live under `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/runs/`.
- No full seed-42 experiment begins until focused tests, the full suite, and the shared-pipeline smoke test pass.
- Full experiments run in the order V2-HVAC, V2-EV, V2-Combined; a failed focused gate stops later full runs.
- Do not promote a run automatically.
- Use TDD for every production behavior: write the test, run it and observe the intended failure, add minimal code, run it green, then commit.
- Do not use subagents if the repository owner repeats the earlier instruction to execute natively; otherwise the external implementer chooses its own execution workflow.

## Review Focus

1. **Incomplete clean base:** a new worktree must contain the currently untracked parent Transformer/evaluation dependencies; the preflight test blocks instead of silently implementing on an incomplete commit.
2. **Forecast-slot masking:** EV weighting must ignore forecast-invalid and post-departure slots, including the one-slot-at-departure edge case; Task 4 pins this.
3. **Normalization reconstruction:** local load/PV/wind and peer net-load forecasts use different normalizers; Task 4 reconstructs physical kW and tests a known synthetic trajectory.
4. **Reliability precedence:** no price or net-load reference may lower the V1 latest-feasible EV floor, even under extreme latent/target noise; Tasks 4 and 6 pin this.
5. **Thermal direction and gate limits:** positive terminal energy gap must request heating, negative gap cooling, and a zero/full recovery gate must reproduce V1/select the anchor exactly; Task 3 pins this.

---

### Task 0: Verify a reproducible implementation base

**Files:**
- Inspect: `AGENTS.md`
- Inspect: `algorithms/matd3_hierarchical_transformer/actor.py`
- Inspect: `algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py`
- Inspect: `algorithms/matd3_hierarchical_transformer_feasible_dr/feasible_dr.py`
- Inspect: `algorithms/matd3_hierarchical_transformer_feasible_dr/training.py`
- Inspect: `configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml`
- Inspect: `evaluation/paper_artifacts.py`
- Inspect: `tests/test_matd3_hierarchical_transformer_feasible_dr.py`

**Interfaces:**
- Consumes: the reviewed V1 implementation and the design at `docs/superpowers/specs/2026-09-20-feasible-dr-v2-design.md`.
- Produces: a known-complete base checkout with a green baseline; this task creates no source change and no commit.

- [ ] **Step 1: Read repository and V2 design instructions completely**

Run:

```powershell
Get-Content AGENTS.md -Raw
Get-Content docs/superpowers/specs/2026-09-20-feasible-dr-v2-design.md -Raw
```

Expected: both files are readable and the implementer can restate the isolation and experiment-order constraints before proceeding.

- [ ] **Step 2: Verify the base contains tracked V1 dependencies**

Run:

```powershell
git ls-files --error-unmatch algorithms/matd3_hierarchical_transformer/actor.py
git ls-files --error-unmatch algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py
git ls-files --error-unmatch evaluation/paper_artifacts.py
git status --short --branch
```

Expected: all three `git ls-files` commands succeed. If any fails, stop and ask the repository owner for a clean baseline commit containing the reviewed parent Transformer, V1, and evaluation artifacts. Do not silently add or commit pre-existing untracked files as part of V2.

- [ ] **Step 3: Create or enter an isolated V2 worktree only after Step 2 passes**

Use the `superpowers:using-git-worktrees` skill. The branch name should be:

```text
codex/feasible-dr-v2
```

Expected: the isolated checkout contains the files verified in Step 2. If the owner explicitly requires the current checkout, record that decision and continue without moving unrelated dirty files.

- [ ] **Step 4: Run the baseline tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer.py tests/test_matd3_hierarchical_transformer_feasible_dr.py
pytest -q
```

Expected: the focused tests and full suite pass before V2 changes. Record the exact passed-test count and warnings. If red, stop and report the named baseline failures instead of changing unrelated code.

- [ ] **Step 5: Record the immutable implementation base**

Run before Task 1 creates any commit:

```powershell
$implementationBase = git rev-parse HEAD
git config --local codex.feasibleDrV2Base $implementationBase
git config --get codex.feasibleDrV2Base
```

Expected: the printed value is the same 40-character commit SHA. Tasks 9 and 10 use this recorded base to prove protected files were untouched.

### Task 1: Scaffold the isolated V2 plugin and three reproducible configurations

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/__init__.py`
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/plugin.py`
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py`
- Create: `configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml`
- Create: `configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2_hvac.yaml`
- Create: `configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2_ev.yaml`
- Create: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

**Interfaces:**
- Consumes: `scripts/create_algorithm_variant.py`, `algorithms.registry.get_algorithm_spec(slug)`, and V1's default YAML.
- Produces: plugin slug `matd3_hierarchical_transformer_feasible_dr_v2`, class `Matd3HierarchicalTransformerFeasibleDRV2`, default combined config, and two `--algorithm-config` ablation configs.

- [ ] **Step 1: Write the failing registry/configuration test**

Add to `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`:

```python
from pathlib import Path

import yaml

from algorithms.registry import get_algorithm_spec


SLUG = "matd3_hierarchical_transformer_feasible_dr_v2"
CONFIG_ROOT = Path("configs/algorithms")


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_v2_plugin_and_ablation_configs_are_isolated():
    spec = get_algorithm_spec(SLUG)
    combined = _load(f"{SLUG}.yaml")
    hvac = _load(f"{SLUG}_hvac.yaml")
    ev = _load(f"{SLUG}_ev.yaml")

    assert spec.slug == SLUG
    assert spec.algorithm_class.__name__ == "Matd3HierarchicalTransformerFeasibleDRV2"
    assert Path(spec.default_config_path).name == f"{SLUG}.yaml"
    for config in (combined, hvac, ev):
        assert config["algorithm"]["name"] == SLUG

    assert combined["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is True
    assert combined["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is True
    assert hvac["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is True
    assert hvac["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is False
    assert ev["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is False
    assert ev["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is True
```

- [ ] **Step 2: Run the test and verify the plugin is missing**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py::test_v2_plugin_and_ablation_configs_are_isolated
```

Expected: FAIL because the V2 package/configurations do not exist.

- [ ] **Step 3: Scaffold the package from V1**

Run:

```powershell
python scripts/create_algorithm_variant.py `
  --name matd3_hierarchical_transformer_feasible_dr_v2 `
  --base matd3_hierarchical_transformer_feasible_dr
```

Replace the generated class/import with:

```python
from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)


class Matd3HierarchicalTransformerFeasibleDRV2(Matd3HierarchicalTransformerFeasibleDR):
    """V2 adaptive-reference feasible DR Transformer."""

    action_semantics = "single_dispatch_dr_adaptive_reference_v2"
```

Set `__init__.py` to export exactly this class. Set `plugin.py` to expose exactly one `ALGORITHM_SPEC` with the slug/display name from the spec and default config `matd3_hierarchical_transformer_feasible_dr_v2.yaml`.

- [ ] **Step 4: Create the combined adaptive-DR configuration**

Copy V1's complete YAML, preserve every network/training/runtime/validation/logging value, change only `algorithm.name`, replace `algorithm.feasible_dr` with this section, and keep `diagnostics` under `algorithm`:

```yaml
  adaptive_dr:
    enabled: true
    interval_epsilon: 1.0e-6
    exploration_in_latent_space: true
    target_smoothing_in_latent_space: true
    agents: {mg1: false, mg2: true, mg3: true}
    hvac:
      interval_mapping_enabled: true
      recovery_anchor_enabled: true
      recovery_pressure_start: 0.35
      recovery_pressure_full: 0.85
      recovery_start_hours: 6.0
      recovery_full_hours: 2.0
    ev:
      catchup_floor_enabled: true
      reliability_margin: 0.0
      economic_reference_enabled: true
      price_weight: 2.0
      net_load_weight: 1.0
      softmax_temperature: 1.0
  diagnostics:
    log_feasible_interval: true
    log_latent_action: true
    log_mandatory_floor: true
    log_adaptive_reference: true
    mask_unavailable_ev_saturation: true
```

- [ ] **Step 5: Create exact HVAC-only and EV-only configurations**

Copy the complete combined YAML twice. In `_hvac.yaml`, set only:

```yaml
recovery_anchor_enabled: true
economic_reference_enabled: false
```

In `_ev.yaml`, set only:

```yaml
recovery_anchor_enabled: false
economic_reference_enabled: true
```

All three YAML files must retain `algorithm.name: matd3_hierarchical_transformer_feasible_dr_v2` so `--algorithm-config` passes shared-pipeline validation.

- [ ] **Step 6: Add the reusable V2 environment fixture**

Add to the same test module:

```python
import pytest

from envs.multi_microgrid_env import MultiMicrogridEnv


@pytest.fixture
def v2_env_and_config():
    env_config = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    algorithm_config = _load(f"{SLUG}.yaml")
    env = MultiMicrogridEnv(env_config, split="train")
    return env, algorithm_config
```

Every later test that mutates the configuration receives a newly loaded dictionary from this fixture.

- [ ] **Step 7: Run the registry/configuration test**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py::test_v2_plugin_and_ablation_configs_are_isolated
```

Expected: PASS.

- [ ] **Step 8: Commit the isolated scaffold**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2 `
  configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml `
  configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2_hvac.yaml `
  configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2_ev.yaml `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: scaffold adaptive feasible DR v2"
```

### Task 2: Implement the common residual map and strict configuration validation

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/adaptive_dr.py`
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

**Interfaces:**
- Consumes: PyTorch tensors with broadcast-compatible shapes and the YAML `algorithm.adaptive_dr` mapping.
- Produces: `smoothstep(value, edge0, edge1)`, `residual_map(latent, lower, reference, upper, interval_epsilon)`, immutable `AdaptiveDRMap`, and `validate_adaptive_dr_config(config, forecast_horizon, ev_max_steps, episode_horizon_hours)`.

- [ ] **Step 1: Write failing residual-map endpoint and property tests**

Add:

```python
import torch

from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    residual_map,
    smoothstep,
)


def test_residual_map_endpoints_and_reference():
    lower = torch.tensor([[-0.4], [0.1], [-0.8]])
    reference = torch.tensor([[0.2], [0.3], [-0.2]])
    upper = torch.tensor([[0.9], [0.7], [0.5]])
    latent = torch.tensor([[-1.0], [0.0], [1.0]])
    actual = residual_map(latent, lower, reference, upper, interval_epsilon=1e-6)
    assert torch.allclose(actual, torch.tensor([[-0.4], [0.3], [0.5]]))


def test_residual_map_keeps_ten_thousand_random_actions_in_interval():
    generator = torch.Generator().manual_seed(42)
    lower = torch.rand(10_000, 1, generator=generator) * 2 - 1
    upper = lower + torch.rand(10_000, 1, generator=generator) * (1 - lower)
    reference = lower + torch.rand(10_000, 1, generator=generator) * (upper - lower)
    latent = torch.rand(10_000, 1, generator=generator) * 2 - 1
    actual = residual_map(latent, lower, reference, upper, interval_epsilon=1e-6)
    assert torch.all(actual >= lower - 1e-7)
    assert torch.all(actual <= upper + 1e-7)
```

- [ ] **Step 2: Write failing smoothstep and gradient tests**

Add:

```python
def test_smoothstep_limits_and_midpoint():
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    actual = smoothstep(values, 0.0, 1.0)
    assert torch.allclose(actual, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_residual_map_has_finite_gradients_on_both_sides():
    latent = torch.tensor([[-0.25], [0.25]], requires_grad=True)
    actual = residual_map(
        latent,
        torch.tensor([[-0.5], [-0.5]]),
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([[0.75], [0.75]]),
        interval_epsilon=1e-6,
    )
    actual.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert torch.all(latent.grad.abs() > 0)
```

- [ ] **Step 3: Run the residual tests and verify missing symbols fail**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "residual_map or smoothstep"
```

Expected: FAIL because `adaptive_dr.py` and its functions do not exist.

- [ ] **Step 4: Implement the common types and functions**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdaptiveDRMap:
    action: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    reference: torch.Tensor
    available: torch.Tensor
    mandatory_floor: torch.Tensor
    recovery_anchor: torch.Tensor
    recovery_gate: torch.Tensor


def smoothstep(value: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    if not edge1 > edge0:
        raise ValueError("smoothstep requires edge1 > edge0")
    unit = ((value - float(edge0)) / (float(edge1) - float(edge0))).clamp(0.0, 1.0)
    return unit.square() * (3.0 - 2.0 * unit)


def residual_map(
    latent: torch.Tensor,
    lower: torch.Tensor,
    reference: torch.Tensor,
    upper: torch.Tensor,
    *,
    interval_epsilon: float,
) -> torch.Tensor:
    low = torch.minimum(lower, upper)
    high = torch.maximum(lower, upper)
    ref = torch.maximum(low, torch.minimum(reference, high))
    z = latent.clamp(-1.0, 1.0)
    mapped = ref + torch.relu(z) * (high - ref) - torch.relu(-z) * (ref - low)
    degenerate = (high - low) <= float(interval_epsilon)
    return torch.where(degenerate, low + 0.0 * latent, mapped)
```

- [ ] **Step 5: Run the common-map tests green**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "residual_map or smoothstep"
```

Expected: PASS.

- [ ] **Step 6: Write failing configuration-validation tests**

Add parametrized cases that mutate a fresh combined config and assert construction raises `ValueError`:

```python
@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("hvac", "recovery_pressure_start", 0.9, "recovery_pressure"),
        ("hvac", "recovery_start_hours", 1.0, "recovery.*hours"),
        ("hvac", "recovery_start_hours", 25.0, "episode horizon"),
        ("ev", "price_weight", -0.1, "price_weight"),
        ("ev", "net_load_weight", -0.1, "net_load_weight"),
        ("ev", "softmax_temperature", 0.0, "softmax_temperature"),
    ],
)
def test_invalid_adaptive_config_is_rejected(v2_env_and_config, section, field, value, message):
    env, config = v2_env_and_config
    config["algorithm"]["adaptive_dr"][section][field] = value
    with pytest.raises(ValueError, match=message):
        Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)
```

For the pressure case, set `recovery_pressure_start=0.9` while `recovery_pressure_full=0.85`. For the hours case, set `recovery_start_hours=1.0` while `recovery_full_hours=2.0`.

- [ ] **Step 7: Run validation tests and verify the stub accepts invalid values**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py::test_invalid_adaptive_config_is_rejected
```

Expected: FAIL because V2 validation is not implemented.

- [ ] **Step 8: Implement exact validation**

Add `validate_adaptive_dr_config(config, *, forecast_horizon, ev_max_steps, episode_horizon_hours)` and call it before actor construction. Validate:

```python
if not config.get("exploration_in_latent_space", True):
    raise ValueError("V2 requires exploration_in_latent_space=true")
if not config.get("target_smoothing_in_latent_space", True):
    raise ValueError("V2 requires target_smoothing_in_latent_space=true")
if float(config.get("interval_epsilon", 1e-6)) <= 0:
    raise ValueError("interval_epsilon must be positive")

hvac = config.get("hvac", {})
if not 0 <= float(hvac["recovery_pressure_start"]) < float(hvac["recovery_pressure_full"]):
    raise ValueError("recovery_pressure_start must be below recovery_pressure_full")
if not 0 <= float(hvac["recovery_full_hours"]) < float(hvac["recovery_start_hours"]):
    raise ValueError("recovery_full_hours must be below recovery_start_hours")
if float(hvac["recovery_start_hours"]) > float(episode_horizon_hours):
    raise ValueError("recovery_start_hours exceeds episode horizon")

ev = config.get("ev", {})
for field in ("price_weight", "net_load_weight"):
    if float(ev[field]) < 0:
        raise ValueError(f"{field} must be nonnegative")
if float(ev["softmax_temperature"]) <= 0:
    raise ValueError("softmax_temperature must be positive")
if bool(ev.get("economic_reference_enabled", True)) and forecast_horizon < ev_max_steps:
    raise ValueError("forecast horizon is shorter than EV availability window")
```

- [ ] **Step 9: Run validation and common tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "invalid_adaptive or residual_map or smoothstep"
```

Expected: PASS.

- [ ] **Step 10: Commit common mapping and validation**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2/adaptive_dr.py `
  algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: add adaptive DR mapping primitives"
```

### Task 3: Implement the MG2 adaptive terminal-recovery anchor

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/adaptive_dr.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

**Interfaces:**
- Consumes: normalized MG2 observation fields, HVAC physical config, `smoothstep`, and the V1 feasible interval.
- Produces: `hvac_recovery_anchor(...)`, `hvac_recovery_gate(...)`, and `AdaptiveDRTransform.map_with_details(...)` for `commercial_hvac`.

- [ ] **Step 1: Write failing thermal-direction and magnitude tests**

Add direct helper tests:

```python
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    hvac_recovery_anchor,
)


HVAC_PHYSICS = {
    "c_air_kwh_per_c": 15.0,
    "c_mass_kwh_per_c": 250.0,
    "cop_heat": 3.3,
    "cop_cool": 3.2,
    "heating_max_kw": 35.0,
    "cooling_max_kw": 35.0,
    "delta_t": 1.0,
}


def test_hvac_recovery_anchor_uses_gap_energy_and_direction():
    anchor = hvac_recovery_anchor(
        air_gap_c=torch.tensor([[1.0], [-1.0], [0.0]]),
        mass_gap_c=torch.tensor([[0.2], [-0.2], [0.0]]),
        remaining_hours=torch.tensor([[5.0], [5.0], [5.0]]),
        lower=torch.full((3, 1), -1.0),
        upper=torch.full((3, 1), 1.0),
        **HVAC_PHYSICS,
    )
    heating_kw = (1.0 * 15.0 + 0.2 * 250.0) / 5.0 / 3.3
    cooling_kw = (-1.0 * 15.0 - 0.2 * 250.0) / 5.0 / 3.2
    assert anchor[0, 0].item() == pytest.approx(heating_kw / 35.0)
    assert anchor[1, 0].item() == pytest.approx(cooling_kw / 35.0)
    assert anchor[2, 0].item() == pytest.approx(0.0)
```

- [ ] **Step 2: Write failing gate-limit and monotonicity tests**

Add:

```python
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    hvac_recovery_gate,
)


def test_hvac_recovery_gate_limits_and_monotonicity():
    pressure = torch.tensor([[0.1], [0.5], [1.0]])
    hours = torch.tensor([[8.0], [4.0], [1.0]])
    gate = hvac_recovery_gate(
        pressure,
        hours,
        pressure_start=0.35,
        pressure_full=0.85,
        start_hours=6.0,
        full_hours=2.0,
    )
    assert gate[0, 0].item() == pytest.approx(0.0)
    assert 0.0 < gate[1, 0].item() < 1.0
    assert gate[2, 0].item() == pytest.approx(1.0)
    assert torch.all(gate[1:] >= gate[:-1])
```

- [ ] **Step 3: Run the new HVAC tests and verify missing helpers fail**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "hvac_recovery"
```

Expected: FAIL because the recovery helpers do not exist.

- [ ] **Step 4: Implement recovery anchor and gate**

Add exact helper signatures:

```python
def hvac_recovery_anchor(
    *,
    air_gap_c: torch.Tensor,
    mass_gap_c: torch.Tensor,
    remaining_hours: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    c_air_kwh_per_c: float,
    c_mass_kwh_per_c: float,
    cop_heat: float,
    cop_cool: float,
    heating_max_kw: float,
    cooling_max_kw: float,
    delta_t: float,
) -> torch.Tensor:
    hours = remaining_hours.clamp_min(float(delta_t))
    thermal_kw = (
        air_gap_c * float(c_air_kwh_per_c)
        + mass_gap_c * float(c_mass_kwh_per_c)
    ) / hours
    electric_kw = torch.where(
        thermal_kw >= 0,
        thermal_kw / float(cop_heat),
        thermal_kw / float(cop_cool),
    )
    normalized = torch.where(
        electric_kw >= 0,
        electric_kw / float(heating_max_kw),
        electric_kw / float(cooling_max_kw),
    )
    return torch.maximum(lower, torch.minimum(normalized, upper))


def hvac_recovery_gate(
    pressure: torch.Tensor,
    remaining_hours: torch.Tensor,
    *,
    pressure_start: float,
    pressure_full: float,
    start_hours: float,
    full_hours: float,
) -> torch.Tensor:
    pressure_gate = smoothstep(pressure, pressure_start, pressure_full)
    time_gate = 1.0 - smoothstep(remaining_hours, full_hours, start_hours)
    return pressure_gate * time_gate
```

- [ ] **Step 5: Run helper tests green**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "hvac_recovery"
```

Expected: PASS.

- [ ] **Step 6: Write failing V1-parity/full-anchor/in-interval tests**

Build a test transform with schema:

```python
HVAC_SCHEMA = (
    "terminal_air_gap", "terminal_mass_gap", "time_remaining",
    "thermal_recovery_pressure", "hvac_feasible_min", "hvac_feasible_max",
)
```

Add three assertions:

```python
def test_hvac_zero_gate_matches_v1_and_full_gate_selects_anchor():
    observation = torch.tensor([
        [1.0 / 40.0, 0.2 / 40.0, 8.0 / 24.0, 0.1, -0.5, 0.8],
        [1.0 / 40.0, 0.2 / 40.0, 1.0 / 24.0, 1.0, -0.5, 0.8],
    ])
    latent = torch.tensor([[0.2], [0.2]])
    transform = make_hvac_v2_transform()
    details = transform.map_with_details(observation, latent)

    v1_free = -0.5 + ((0.2 + 1.0) / 2.0) * (0.8 - -0.5)
    assert details.action[0, 0].item() == pytest.approx(v1_free)
    assert details.recovery_gate[0, 0].item() == pytest.approx(0.0)
    assert details.recovery_gate[1, 0].item() == pytest.approx(1.0)
    assert details.action[1, 0].item() == pytest.approx(details.recovery_anchor[1, 0].item())
```

Add a 10,000-row random test asserting the mapped action lies between `hvac_feasible_min` and `hvac_feasible_max` and remains finite.

- [ ] **Step 7: Run integration tests and verify `map_with_details` lacks HVAC behavior**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "zero_gate or full_gate or random_hvac"
```

Expected: FAIL because `AdaptiveDRTransform`/HVAC mapping is incomplete.

- [ ] **Step 8: Implement `AdaptiveDRTransform` HVAC construction and mapping**

Subclass V1's `FeasibleDRTransform`. Cache Python integer indices for every HVAC field. De-normalize gaps/hours as specified. For a disabled recovery anchor, return the V1 feasible action and a zero gate. For an enabled anchor, return:

```python
action = (1.0 - gate) * free_action + gate * anchor
```

Populate `AdaptiveDRMap` with zero tensors for EV-only fields so diagnostics never need type-specific missing values.

- [ ] **Step 9: Run all HVAC and common-map tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "hvac or residual_map or smoothstep"
```

Expected: PASS.

- [ ] **Step 10: Commit the HVAC anchor**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2/adaptive_dr.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: add adaptive HVAC recovery anchor"
```

### Task 4: Implement the MG3 economic charging reference

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/adaptive_dr.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

**Interfaces:**
- Consumes: remaining EV battery energy, V1 latest-feasible floor, current/future price, reconstructed system net-load kW, validity/departure masks, and normalization metadata.
- Produces: `economic_slot_weights(...)`, `ev_economic_reference(...)`, and `AdaptiveDRTransform.map_with_details(...)` for `ev_charging`.

- [ ] **Step 1: Write failing economic-weight tests**

Add:

```python
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    economic_slot_weights,
)


def test_equal_economic_signals_produce_uniform_weights():
    weights = economic_slot_weights(
        prices=torch.tensor([[0.4, 0.4, 0.4]]),
        system_net_load_kw=torch.tensor([[100.0, 100.0, 100.0]]),
        valid=torch.tensor([[True, True, True]]),
        price_weight=2.0,
        net_load_weight=1.0,
        temperature=1.0,
    )
    assert torch.allclose(weights, torch.full((1, 3), 1.0 / 3.0))


def test_lower_price_and_net_load_receive_more_weight():
    weights = economic_slot_weights(
        prices=torch.tensor([[0.4, 0.8, 0.8]]),
        system_net_load_kw=torch.tensor([[50.0, 100.0, 150.0]]),
        valid=torch.tensor([[True, True, True]]),
        price_weight=2.0,
        net_load_weight=1.0,
        temperature=1.0,
    )
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
```

- [ ] **Step 2: Write failing mask and all-equal-range tests**

Add:

```python
def test_economic_weights_mask_invalid_and_post_departure_slots():
    weights = economic_slot_weights(
        prices=torch.tensor([[0.8, 0.4, 0.1, 0.1]]),
        system_net_load_kw=torch.tensor([[100.0, 100.0, 0.0, 0.0]]),
        valid=torch.tensor([[True, True, False, False]]),
        price_weight=2.0,
        net_load_weight=1.0,
        temperature=1.0,
    )
    assert weights[0, 2:].sum().item() == pytest.approx(0.0)
    assert weights[0, :2].sum().item() == pytest.approx(1.0)
```

- [ ] **Step 3: Run weight tests and verify missing helper failure**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "economic_weights or economic_signals"
```

Expected: FAIL because `economic_slot_weights` is missing.

- [ ] **Step 4: Implement stable masked weights**

Use this signature and behavior:

```python
def economic_slot_weights(
    *,
    prices: torch.Tensor,
    system_net_load_kw: torch.Tensor,
    valid: torch.Tensor,
    price_weight: float,
    net_load_weight: float,
    temperature: float,
) -> torch.Tensor:
    mask = valid.bool()
    large = torch.finfo(prices.dtype).max
    small = torch.finfo(prices.dtype).min

    p_min = torch.where(mask, prices, large).min(dim=-1, keepdim=True).values
    p_max = torch.where(mask, prices, small).max(dim=-1, keepdim=True).values
    n_min = torch.where(mask, system_net_load_kw, large).min(dim=-1, keepdim=True).values
    n_max = torch.where(mask, system_net_load_kw, small).max(dim=-1, keepdim=True).values

    p_score = (prices - p_min) / (p_max - p_min).clamp_min(1e-6)
    n_score = (system_net_load_kw - n_min) / (n_max - n_min).clamp_min(1e-6)
    logits = (-float(price_weight) * p_score - float(net_load_weight) * n_score) / float(temperature)
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights))
```

Validate at least one valid slot per row before calling this helper; a row with no valid slot represents an unavailable EV and bypasses weighting.

- [ ] **Step 5: Run weight tests green**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "economic_weights or economic_signals"
```

Expected: PASS.

- [ ] **Step 6: Write failing reference/floor/unavailable tests**

Add:

```python
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    ev_economic_reference,
)


def test_ev_reference_respects_floor_upper_and_one_slot_case():
    reference = ev_economic_reference(
        remaining_battery_kwh=torch.tensor([[30.0]]),
        efficiency=1.0,
        delta_t=1.0,
        max_charge_kw=20.0,
        current_weight=torch.tensor([[1.0]]),
        lower=torch.tensor([[0.5]]),
        upper=torch.tensor([[1.0]]),
    )
    assert torch.allclose(reference, torch.tensor([[1.0]]))


def test_ev_reference_never_weakens_latest_feasible_floor():
    reference = ev_economic_reference(
        remaining_battery_kwh=torch.tensor([[10.0]]),
        efficiency=1.0,
        delta_t=1.0,
        max_charge_kw=20.0,
        current_weight=torch.tensor([[0.01]]),
        lower=torch.tensor([[0.4]]),
        upper=torch.tensor([[0.8]]),
    )
    assert reference.item() == pytest.approx(0.4)
```

Add an integration test with `ev_available=0` asserting `action=-1`, `reference=0`, and `available=False`.

- [ ] **Step 7: Run reference tests and verify failure**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "ev_reference or unavailable"
```

Expected: FAIL because the reference helper/EV integration is missing.

- [ ] **Step 8: Implement current-slot reference and residual mapping**

Add:

```python
def ev_economic_reference(
    *,
    remaining_battery_kwh: torch.Tensor,
    efficiency: float,
    delta_t: float,
    max_charge_kw: float,
    current_weight: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    remaining_grid_kwh = remaining_battery_kwh / float(efficiency)
    reference_kw = remaining_grid_kwh * current_weight / float(delta_t)
    reference_unit = reference_kw / float(max_charge_kw)
    return torch.maximum(lower, torch.minimum(reference_unit, upper))
```

In the EV branch:

1. reconstruct current/future physical prices and system net-load kW with per-field normalizers;
2. build a valid mask from `ev_available`, `hours_to_departure`, and `forecast_valid_t+k`;
3. compute economic weights only when the economic reference is enabled;
4. compute the V1 mandatory floor and `lower=max(feasible_min, floor)`;
5. use the V1 midpoint as reference when the economic reference is disabled;
6. use `residual_map` in unit charge space;
7. convert to raw `2*unit-1`;
8. return raw `-1` for unavailable rows.

- [ ] **Step 9: Write and run the normalization-reconstruction test**

Construct a synthetic schema where local values and peers use deliberately different scales:

```python
normalizers = {
    "mg1": {"load": 100.0, "pv": 200.0, "wind": 1.0, "power": 250.0, "price": 2.0},
    "mg2": {"load": 80.0, "pv": 1.0, "wind": 300.0, "power": 320.0, "price": 2.0},
    "mg3": {"load": 60.0, "pv": 120.0, "wind": 1.0, "power": 150.0, "price": 1.5},
}
```

Encode a known two-slot physical trajectory and assert the transform's extracted price and system-net-load tensors match the hand-calculated kW values. This test must fail if local normalized values are summed without de-normalization.

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "normalization_reconstruction or ev_reference or economic"
```

Expected after implementation: PASS.

- [ ] **Step 10: Run all adaptive-transform tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "residual or smoothstep or hvac or ev or economic or normalization"
```

Expected: PASS.

- [ ] **Step 11: Commit the EV economic reference**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2/adaptive_dr.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: add economic EV charging reference"
```

### Task 5: Integrate the adaptive transform into an isolated actor and algorithm

**Files:**

- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/actor.py`
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

- [ ] **Step 1: Reuse the isolated V2 environment/config fixture**

Use the `v2_env_and_config` fixture created in Task 1. Keep the environment episode length at 24 and pass seed 42 explicitly to algorithm constructors unless an individual test needs a smaller synthetic tensor.

- [ ] **Step 2: Write the shared-interface and action-shape tests**

Instantiate the algorithm through the registry and assert it provides:

```python
required = ("select_actions", "update", "save", "load", "replay")
assert all(hasattr(algorithm, name) for name in required)
actions = algorithm.select_actions(observations, noise_std=0.0)
assert set(actions) == set(env.agent_ids)
assert all(action.shape == env.action_space(agent).shape for agent, action in actions.items())
```

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "shared_interface or action_shape"
```

Expected: FAIL because the generated algorithm still creates the inherited V1 actors.

- [ ] **Step 3: Write the exact MG1 parity test**

Build V1 and V2 actors from the same schema, copy the V1 MG1 state dictionary into the V2 MG1 actor, and evaluate the same observation tensor with `noise_std=0.0`. Assert bitwise equality:

```python
assert torch.equal(v1_action, v2_action)
```

The test must also assert that MG1 does not expose HVAC or EV adaptive diagnostic fields.

- [ ] **Step 4: Implement `AdaptiveDRActor`**

Use this public constructor and mapping entry point:

```python
class AdaptiveDRActor(FeasibleDRActor):
    def __init__(
        self,
        schema,
        agent_ids,
        active_agent,
        mg_config=None,
        *,
        adaptive_config,
        normalizers_by_microgrid,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        forecast_horizon=8,
        hidden_dim=64,
        delta_t=1.0,
        episode_length=24,
    ):
        ...

    def map_latent_actions(
        self,
        observations: torch.Tensor,
        latent_actions: torch.Tensor,
        *,
        collect_diagnostics: bool = False,
    ) -> torch.Tensor:
        ...
```

Implementation rules:

1. Reuse the V1 Transformer trunk and non-DR action heads without alteration.
2. For MG1, call the inherited V1 forward path directly; do not construct an adaptive transform.
3. For MG2, replace only the HVAC latent-to-feasible mapping with the V2 recovery-anchor residual mapping.
4. For MG3, replace only the EV latent-to-feasible mapping with the V2 economic-reference residual mapping.
5. Never clamp a mapped action outside the physical interval as a normal execution path; assert the transform already returns an in-interval result, allowing only a numerical tolerance of `1e-6`.
6. Preserve V1's tensor return type. When `collect_diagnostics=True`, update `self.last_diagnostics` with detached scalar mapping diagnostics; when false, do not compute or mutate the V2 mapping-diagnostic fields.

- [ ] **Step 5: Build normalization metadata and its signature once**

In `algorithm.py`, extract immutable physical scales for every microgrid:

```python
source_normalizers = env.global_state_builder.norm
normalizers_by_microgrid = {
    microgrid_id: {
        field: float(source_normalizers[microgrid_id][field])
        for field in ("load", "pv", "wind", "power", "price")
    }
    for microgrid_id in env.agent_ids
}
```

Reject a missing, non-finite, or non-positive scale with the microgrid and field named in the error. Serialize the resulting mapping with sorted keys and compact separators, then compute its SHA-256 hex digest. Store both the mapping and digest on the V2 algorithm.

- [ ] **Step 6: Replace only the V1 actor construction**

The V2 class may subclass the V1 feasible-DR algorithm to reuse replay, critics, update scheduling, and optimizer conventions. Convert the V2 config into a private V1-compatible config only for the inherited constructor, then immediately replace:

- online actors;
- target actors;
- actor optimizers;
- actor parameter caches used by delayed updates.

Create every replacement as `AdaptiveDRActor`, copy online weights to its target, and keep the original V2 YAML/config object for checkpoint identity and run snapshots. Do not mutate `config["algorithm"]["name"]` in the caller-owned object.

- [ ] **Step 7: Test deterministic feasible actions for MG2 and MG3**

For at least 256 synthetic boundary-heavy observations, assert:

```python
mapped = actor.map_latent_actions(observations, latent)
assert torch.all(mapped >= feasible_min - 1e-6)
assert torch.all(mapped <= feasible_max + 1e-6)
```

Also run a deterministic environment episode and assert the recorded MG2/MG3 projection counts are zero.

- [ ] **Step 8: Run the actor/algorithm integration tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "shared_interface or action_shape or mg1_parity or actor or deterministic_feasible"
```

Expected: PASS.

- [ ] **Step 9: Commit the actor integration**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2/actor.py `
  algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: integrate adaptive feasible DR actors"
```

### Task 6: Preserve latent-space training and add actionable diagnostics

**Files:**

- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/training.py`
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

- [ ] **Step 1: Write the extreme-noise feasibility test**

Use latent values at `-1`, `0`, and `1`, add exploration/target noise with standard deviation larger than the production setting, clamp the noisy latent preference to `[-1, 1]`, then map it. Assert that every mapped HVAC/EV action remains feasible and finite, including zero-width intervals and unavailable EV rows.

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "extreme_noise"
```

Expected: FAIL until the V2 training helpers are connected.

- [ ] **Step 2: Implement the three training entry points**

Add:

```python
def map_noisy_adaptive_latent(
    actor,
    observations,
    latent_actions,
    *,
    noise_std,
    noise_clip,
    collect_diagnostics=False,
):
    ...

def select_adaptive_actions(algorithm, observations, noise_std):
    ...

def update_adaptive_dr(algorithm, batch_size):
    ...
```

Rules:

1. Add exploration and target-policy smoothing noise before feasible mapping.
2. Clamp only the latent preference to `[-1, 1]`; let the adaptive transform enforce the physical interval.
3. Preserve the V1 critic target, actor loss, delayed-update cadence, soft target update, replay sampling, gradient clipping, and optimizer order exactly.
4. The only permitted behavioral substitutions are the adaptive actor mapping and scheduled diagnostic collection.
5. Keep target actors in evaluation mode and ensure diagnostic collection does not attach tensors to a retained autograd graph.
6. In the delayed actor-update loop call `map_latent_actions(obs, latent, collect_diagnostics=diagnostic_due)`, pass the actor through the existing `_actor_metric_payload`, and add `dr_projection_delta_mean` beside its existing `dr_projection_rate` from the same projected-action plan.

- [ ] **Step 3: Write the diagnostic schedule test**

Configure a schedule that collects on update 2 but not update 1. Assert update 1 omits the expensive DR payload and update 2 contains these exact keys:

```python
COMMON_KEYS = {
    "actor_dr_latent_mean",
    "actor_dr_latent_std",
    "actor_dr_mapped_mean",
    "actor_dr_feasible_min_mean",
    "actor_dr_feasible_max_mean",
    "actor_dr_interval_width_mean",
    "actor_dr_lower_boundary_rate",
    "actor_dr_upper_boundary_rate",
    "actor_dr_effective_boundary_rate",
    "dr_projection_rate",
    "dr_projection_delta_mean",
}
MG2_KEYS = {
    "actor_hvac_recovery_anchor_mean",
    "actor_hvac_recovery_gate_mean",
    "actor_hvac_recovery_gate_active_rate",
    "actor_hvac_recovery_gate_full_rate",
    "actor_hvac_terminal_air_gap_mean_c",
    "actor_hvac_terminal_mass_gap_mean_c",
    "actor_hvac_terminal_recovery_pressure_mean",
}
MG3_KEYS = {
    "actor_ev_mandatory_floor_mean",
    "actor_ev_mandatory_floor_active_rate",
    "actor_ev_economic_reference_mean",
    "actor_ev_reference_above_floor_mean",
    "actor_ev_latent_reference_deviation_mean",
    "actor_ev_deadline_pressure_mean",
    "actor_ev_available_rate",
    "actor_ev_available_boundary_rate",
}
```

These are the keys inside each `actor_diagnostics[mg]` payload after `_actor_metric_payload` applies its established `actor_` prefix to `actor.last_diagnostics`; projection fields remain unprefixed because they are payload-level metrics. The test must check key names and numeric values, not merely the number of keys.

- [ ] **Step 4: Mask unavailable EV rows from saturation statistics**

Define the available-boundary statistic over rows where EV availability is true. If no row is available, emit `0.0`, never `NaN`. Add tests for an all-unavailable batch and a mixed batch where only the available row is at its lower boundary.

- [ ] **Step 5: Run training and diagnostics tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "noise or training or update or diagnostic or unavailable"
```

Expected: PASS.

- [ ] **Step 6: Commit training integration**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2/training.py `
  algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: train and diagnose adaptive DR references"
```

### Task 7: Enforce strict V2 checkpoint identity

**Files:**

- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py`

- [ ] **Step 1: Write a complete round-trip test**

Create an algorithm, perform one update from a seeded replay batch, save it, load it into a fresh instance, and assert equality of:

- online and target actor parameters;
- critic and target critic parameters;
- optimizer states;
- update counters;
- algorithm slug;
- action semantics;
- projection signature;
- adaptive DR config;
- observation schemas;
- global state schema;
- normalization mapping and signature.

- [ ] **Step 2: Write one rejection test per identity field**

Parametrize over these mutations:

```python
identity_mutations = {
    "algorithm_slug": "matd3_hierarchical_transformer_feasible_dr",
    "action_semantics": "single_dispatch_feasible_dr_v1",
    "projection_signature": "0" * 64,
    "adaptive_dr_config": {"hvac": {"enabled": False}},
    "observation_schemas": {"mg1": ("wrong_field",)},
    "state_schema": ("wrong_field",),
    "normalizers_by_microgrid": {"mg1": {"load": 1.0}},
    "normalization_signature": "0" * 64,
}
```

Each mutated checkpoint must raise `ValueError` with the mismatched field name in the message. Also reject a checkpoint that omits any identity field; do not silently supply defaults for legacy checkpoints.

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "checkpoint"
```

Expected: FAIL before save/load is overridden.

- [ ] **Step 3: Save the full V2 identity envelope**

The checkpoint payload must include at least:

```python
{
    "algorithm_slug": "matd3_hierarchical_transformer_feasible_dr_v2",
    "action_semantics": "single_dispatch_dr_adaptive_reference_v2",
    "projection_signature": self.projection_signature,
    "adaptive_dr_config": canonical_adaptive_config,
    "observation_schemas": self.observation_schemas,
    "state_schema": self.state_schema,
    "normalizers_by_microgrid": normalizers_by_microgrid,
    "normalization_signature": normalization_signature,
    "actors": self.actors.state_dict(),
    "target_actors": self.target_actors.state_dict(),
    "critics": self.critics.state_dict(),
    "target_critics": self.target_critics.state_dict(),
    "actor_optimizers": {mg: opt.state_dict() for mg, opt in self.actor_optimizers.items()},
    "critic_optimizers": {mg: opt.state_dict() for mg, opt in self.critic_optimizers.items()},
    "update_steps": self.update_steps,
}
```

Canonicalize the adaptive config recursively before comparison so YAML dictionary order cannot create a false mismatch. Preserve the repository's existing device-safe tensor loading behavior.

- [ ] **Step 4: Validate identity before mutating live state**

On load, validate every identity field and tensor-key structure before calling any `load_state_dict`. This guarantees a rejected checkpoint cannot leave a partially changed algorithm instance.

- [ ] **Step 5: Run checkpoint and adjacent V1 isolation tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py -k "checkpoint"
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "checkpoint"
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint isolation**

Run:

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr_v2/algorithm.py `
  tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
git commit -m "feat: isolate adaptive DR v2 checkpoints"
```

### Task 8: Add a saved-result-only V2 acceptance gate

**Files:**

- Create: `scripts/check_feasible_dr_v2_gate.py`
- Create: `tests/test_feasible_dr_v2_gate.py`

- [ ] **Step 1: Write pure metric-calculation tests**

Create compact saved-result fixtures for a passing and failing run. Cover at least:

- mean grid import during hours 16–21 computed from hourly trajectory rows;
- projection rate denominator and zero-action edge cases;
- terminal joint success extraction;
- EV valley-share extraction;
- missing file, missing column, and non-finite metric rejection.

For the peak calculation, use hand-checkable rows:

```python
rows = [
    {"episode": 0, "hour": 16, "grid_import_kwh": 100.0},
    {"episode": 0, "hour": 17, "grid_import_kwh": 200.0},
    {"episode": 1, "hour": 16, "grid_import_kwh": 300.0},
    {"episode": 1, "hour": 17, "grid_import_kwh": 400.0},
]
assert mean_peak_import(rows, start_hour=16, end_hour=21) == 500.0
```

The helper must first sum all 16:00–21:00 grid-import energy within each episode and then average those daily sums. If the saved rows are per-microgrid rather than already system-aggregated, sum microgrids within `(episode, hour)` before forming the daily total.

Run:

```powershell
pytest -q tests/test_feasible_dr_v2_gate.py
```

Expected: FAIL because the checker does not exist.

- [ ] **Step 2: Implement strict saved-artifact readers**

Read only persisted artifacts from the supplied run directory:

- `paper_metrics.json`;
- `test_summary.json`;
- `dr_shift_metrics.csv`;
- `test_hourly_trajectories/*.csv`;
- `run_metadata.json`;
- `training_summary.json`.

Do not import or run the environment, evaluator, settlement, or plotting code. Validate that metadata names the V2 slug and that the formal test used `checkpoints/best_model.pt`. Return a structured error for every missing or malformed input.

- [ ] **Step 3: Implement focused and combined gates**

Expose:

```python
def evaluate_v2_gate(
    run_dir: Path,
    baseline_dir: Path,
    mode: Literal["hvac", "ev", "combined"],
) -> dict[str, object]:
    ...
```

Every mode must apply the preservation gates:

```python
PRESERVATION_LIMITS = {
    "mg2_projection_rate_max": 0.01,
    "mg3_projection_rate_max": 0.01,
    "ev_success_rate_min": 1.00,
    "ev_shortfall_kwh_per_day_max": 0.05,
    "hvac_comfort_rate_min": 0.995,
    "strict_feasible_rate_min": 1.00,
    "unmet_dr_energy_kwh_per_day_max": 0.0,
    "mg1_production_rate_min": 0.995,
}
```

Apply these mode-specific thresholds from the approved design:

```python
HVAC_LIMITS = {
    "terminal_joint_success_rate_min": 0.4545,
    "terminal_joint_success_rate_target": 0.55,
    "terminal_task_penalty_max": 35.25,
}
EV_LIMITS = {
    "ev_valley_energy_share_min": 0.95,
    "mean_peak_load_reduction_pct_min": -4.07,
}
COMBINED_LIMITS = {
    "total_system_cost_max": 874.63,
    "nonterminal_system_cost_max": 839.38,
    "terminal_task_penalty_max": 35.25,
    "ev_valley_energy_share_min": 0.95,
    "mean_peak_grid_import_kwh_max": 283.02,
    "mean_peak_load_reduction_pct_min": -4.07,
    "mean_peak_load_reduction_pct_target": 0.0,
    "p2p_energy_kwh_per_day_min": 1049.78,
    "p2p_offer_acceptance_rate_min": 0.3149,
    "ess_equivalent_cycles_per_day_max": 1.1665,
    "renewable_utilization_rate_min": 0.9661,
}
```

The `target` thresholds are informational stretch goals and must be reported separately from required pass/fail limits.

For combined mode, report MG1 terminal silo success relative to V1's `41.82%` as a regression warning. A value below `35%` must set `eligible_for_multiseed` to false even if every combined hard limit passes.

- [ ] **Step 4: Add a safe command-line interface**

Support:

```powershell
$v2Run = (Get-Content outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json -Raw | ConvertFrom-Json).run_directory
python scripts/check_feasible_dr_v2_gate.py `
  --run $v2Run `
  --baseline outputs/working_runs/matd3_hierarchical_transformer/runs/MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42 `
  --mode combined
```

The checker must write `v2_gate_report.json` directly inside the directory supplied by `--run`. Refuse to overwrite an existing report unless `--overwrite` is passed. Exit codes:

- `0`: every required gate passed;
- `1`: inputs were valid but one or more required gates failed;
- `2`: artifacts or arguments were invalid.

The JSON report must include source paths, source-file SHA-256 hashes, thresholds, actual values, per-gate status, stretch-goal status, and overall status.

Also add a mutually exclusive multi-seed mode with exactly three repeated `--multiseed-run` paths, exactly three repeated `--multiseed-baseline` paths, and `--multiseed-output`. Match runs by the seed recorded in metadata, reject duplicates/missing seeds, and report every seed plus mean, median, population standard deviation, and interquartile range. Its required conditions are:

- each seed has MG2/MG3 projection at or below 1%;
- each seed has 100% MG3 departure success;
- each seed has at least 99.5% MG2 occupied comfort;
- median MG2 terminal thermal success is at least 45.45%, with no seed below 40%;
- median EV valley charging share is at least 95%;
- median total operating cost does not exceed the matching original-Transformer median;
- median 16:00–21:00 grid import is no more than 5% above the matching original-Transformer median.

Apply the same overwrite protection and exit-code meanings to the multi-seed report.

- [ ] **Step 5: Test CLI output and overwrite protection**

Run the single-run and multi-seed CLIs against test fixtures through `subprocess`, assert all three exit codes, inspect both JSON schemas, verify seed matching and the median/IQR calculations, and confirm a second invocation without `--overwrite` leaves the original report byte-for-byte unchanged.

- [ ] **Step 6: Run checker tests**

Run:

```powershell
pytest -q tests/test_feasible_dr_v2_gate.py
```

Expected: PASS.

- [ ] **Step 7: Commit the gate checker**

Run:

```powershell
git add -- scripts/check_feasible_dr_v2_gate.py tests/test_feasible_dr_v2_gate.py
git commit -m "feat: add adaptive DR v2 result gates"
```

### Task 9: Run repository regression and a shared-pipeline smoke experiment

**Files:**

- Verify: `algorithms/matd3/`
- Verify: `algorithms/matd3_hierarchical_transformer/`
- Verify: `algorithms/matd3_hierarchical_transformer_feasible_dr/`
- Verify: `configs/algorithms/matd3.yaml`
- Verify: `configs/algorithms/matd3_hierarchical_transformer.yaml`
- Verify: `configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml`
- Produce: a pipeline-generated child directory under `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/runs/`

- [ ] **Step 1: Run all focused V2 tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py tests/test_feasible_dr_v2_gate.py
```

Expected: PASS with no skipped V2 behavior test.

- [ ] **Step 2: Run parent and V1 regression tests**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer.py tests/test_matd3_hierarchical_transformer_feasible_dr.py
```

Expected: PASS.

- [ ] **Step 3: Run the full repository suite**

Run:

```powershell
pytest -q
```

Expected: PASS. Record the exact passed-test count, warnings, and duration in the implementation handoff.

- [ ] **Step 4: Prove protected implementations and configs are unchanged**

Run:

```powershell
$implementationBase = git config --get codex.feasibleDrV2Base
if (-not $implementationBase) { throw "Missing recorded V2 implementation base" }
$protectedPaths = @(
  "algorithms/matd3",
  "algorithms/matd3_hierarchical_transformer",
  "algorithms/matd3_hierarchical_transformer_feasible_dr",
  "configs/algorithms/matd3.yaml",
  "configs/algorithms/matd3_hierarchical_transformer.yaml",
  "configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml"
)
git diff --exit-code $implementationBase -- $protectedPaths
```

Expected: exit code 0 and no diff. Any output is a release blocker; restore V2 isolation without discarding unrelated owner changes.

- [ ] **Step 5: Run the three-episode combined smoke experiment**

Run:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_feasible_dr_v2 `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml `
  --episodes 3 `
  --test-episodes 3 `
  --validation-interval 1 `
  --validation-days 1 `
  --device cpu `
  --seed 42 `
  --no-dashboard
if ($LASTEXITCODE -ne 0) { throw "V2 smoke experiment failed" }
```

Expected: training, validation, best-checkpoint selection, deterministic formal test, seven-coalition daily MILP evaluation, Equal/Shapley settlement, summaries, paper artifacts, and plotting all complete.

- [ ] **Step 6: Verify smoke artifacts and identity**

Run:

```powershell
$latest = Get-Content `
  outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json `
  -Raw | ConvertFrom-Json
$smokeRun = [System.IO.Path]::GetFullPath($latest.run_directory)
$requiredFiles = @(
  "checkpoints/best_model.pt",
  "checkpoints/final_model.pt",
  "config_snapshot.yaml",
  "run_metadata.json",
  "training_history.json",
  "validation_history.json",
  "training_summary.json",
  "test_daily_metrics.csv",
  "test_summary.json",
  "summary.json",
  "paper_metrics.json",
  "paper_metrics.csv",
  "dr_shift_metrics.csv"
)
$requiredFiles | ForEach-Object {
  if (-not (Test-Path (Join-Path $smokeRun $_))) { throw "Missing smoke artifact: $_" }
}
$metadata = Get-Content (Join-Path $smokeRun "run_metadata.json") -Raw | ConvertFrom-Json
$summary = Get-Content (Join-Path $smokeRun "test_summary.json") -Raw | ConvertFrom-Json
if ($metadata.algorithm_slug -ne "matd3_hierarchical_transformer_feasible_dr_v2") { throw "Wrong smoke plugin" }
if ($summary.checkpoint_used -notmatch "best_model\.pt$") { throw "Formal test did not use best_model.pt" }
if ((Get-ChildItem (Join-Path $smokeRun "test_hourly_trajectories") -Filter *.csv).Count -ne 3) { throw "Wrong smoke test-day count" }
if ((Get-ChildItem (Join-Path $smokeRun "figures") -Filter *.png).Count -ne 24) { throw "Wrong PNG count" }
if ((Get-ChildItem (Join-Path $smokeRun "figures") -Filter *.pdf).Count -ne 24) { throw "Wrong PDF count" }
```

Open `config_snapshot.yaml` and confirm the combined adaptive config is present. Inspect saved metrics and confirm MG2/MG3 projection is zero, EV departure success is 100%, and strict feasible-day rate is 100%. These smoke checks validate wiring, not statistical performance.

- [ ] **Step 7: Fix only failures attributable to V2 and rerun the complete gate**

Use `superpowers:systematic-debugging` for any failure. Add a regression test before each fix, commit the smallest V2-local change, then repeat Steps 1–6. Do not weaken tests, shared rewards, physics, or thresholds to make the smoke pass.

### Task 10: Run staged seed-42 ablations, then conditional multi-seed confirmation

**Files:**

- Read baseline: `outputs/working_runs/matd3_hierarchical_transformer/runs/MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42/`
- Read V1 reference: `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/runs/MATD3HIERARCHICALTRANSFORMERFEASIBLEDR_2026-09-20_19-30-09_seed42/`
- Produce HVAC runs under: `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/runs/`
- Produce EV runs under: `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/runs/`
- Produce combined runs under: `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/runs/`
- Produce per-run: `v2_gate_report.json`
- Produce after three passing V2 seeds: `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/multiseed_gate_report.json`

- [ ] **Step 1: Lock the seed-42 comparison baseline**

Run:

```powershell
$baseline42 = [System.IO.Path]::GetFullPath(
  "outputs/working_runs/matd3_hierarchical_transformer/runs/MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42"
)
$v1Seed42 = [System.IO.Path]::GetFullPath(
  "outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/runs/MATD3HIERARCHICALTRANSFORMERFEASIBLEDR_2026-09-20_19-30-09_seed42"
)
if (-not (Test-Path (Join-Path $baseline42 "test_summary.json"))) { throw "Missing approved seed-42 baseline" }
if (-not (Test-Path (Join-Path $v1Seed42 "test_summary.json"))) { throw "Missing approved V1 seed-42 reference" }
Get-FileHash (Join-Path $baseline42 "test_summary.json") -Algorithm SHA256
Get-FileHash (Join-Path $baseline42 "paper_metrics.json") -Algorithm SHA256
Get-FileHash (Join-Path $baseline42 "dr_shift_metrics.csv") -Algorithm SHA256
Get-FileHash (Join-Path $v1Seed42 "test_summary.json") -Algorithm SHA256
Get-FileHash (Join-Path $v1Seed42 "paper_metrics.json") -Algorithm SHA256
Get-FileHash (Join-Path $v1Seed42 "dr_shift_metrics.csv") -Algorithm SHA256
```

Record these hashes in the experiment handoff. Do not edit or regenerate the approved baseline.

- [ ] **Step 2: Run V2-HVAC seed 42 and enforce its focused gate**

Run:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_feasible_dr_v2 `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2_hvac.yaml `
  --device cuda `
  --seed 42 `
  --no-dashboard
if ($LASTEXITCODE -ne 0) { throw "V2-HVAC experiment failed" }
$hvacRun = (Get-Content outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json -Raw | ConvertFrom-Json).run_directory
python scripts/check_feasible_dr_v2_gate.py --run $hvacRun --baseline $baseline42 --mode hvac
if ($LASTEXITCODE -ne 0) { throw "V2-HVAC gate failed; stop before V2-EV" }
```

If the gate fails, stop. Diagnose recovery-anchor sign, denormalized air/mass gaps, gate activation, interval clipping, and available recovery power from saved diagnostics before changing V2-local parameters.

- [ ] **Step 3: Run V2-EV seed 42 and enforce its focused gate**

Run only after Step 2 passes:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_feasible_dr_v2 `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2_ev.yaml `
  --device cuda `
  --seed 42 `
  --no-dashboard
if ($LASTEXITCODE -ne 0) { throw "V2-EV experiment failed" }
$evRun = (Get-Content outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json -Raw | ConvertFrom-Json).run_directory
python scripts/check_feasible_dr_v2_gate.py --run $evRun --baseline $baseline42 --mode ev
if ($LASTEXITCODE -ne 0) { throw "V2-EV gate failed; stop before V2-Combined" }
```

If reliability fails, treat it as an implementation defect. If valley share or peak-load reduction fails, inspect slot masks, physical-unit reconstruction, forecast alignment, temperature, and floor override rate; tune only the V2 EV configuration.

- [ ] **Step 4: Run V2-Combined seed 42 and enforce the full gate**

Run only after both focused variants pass:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_feasible_dr_v2 `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml `
  --device cuda `
  --seed 42 `
  --no-dashboard
if ($LASTEXITCODE -ne 0) { throw "V2-Combined seed-42 experiment failed" }
$combined42 = (Get-Content outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json -Raw | ConvertFrom-Json).run_directory
python scripts/check_feasible_dr_v2_gate.py --run $combined42 --baseline $baseline42 --mode combined
if ($LASTEXITCODE -ne 0) { throw "V2-Combined seed-42 gate failed; do not start multi-seed runs" }
```

Even when the hard gate passes, do not proceed if `v2_gate_report.json` sets `eligible_for_multiseed` to false because MG1 terminal silo success is below 35%.

- [ ] **Step 5: Produce the seed-42 comparison table**

From the three immutable `v2_gate_report.json` files plus the saved artifacts under `$baseline42` and `$v1Seed42`, report V1, original Transformer, V2-HVAC, V2-EV, and V2-Combined for:

- total and non-terminal operating cost;
- terminal task penalty by microgrid;
- MG2/MG3 projection rate;
- HVAC occupied comfort and terminal thermal success;
- EV departure success, shortfall, and valley share;
- strict feasible-day rate and unmet energy;
- 16:00–21:00 system grid import and peak-load reduction;
- P2P energy and offer acceptance;
- ESS equivalent cycles;
- renewable utilization;
- MG1 production and terminal silo success.

Label every hard failure, preservation pass, and stretch target separately. Do not infer causality from the combined run alone; use the two focused ablations to attribute changes.

- [ ] **Step 6: Ensure matching original-Transformer seeds 43 and 44 exist**

Only after Step 4 passes, locate completed original-Transformer runs for seeds 43 and 44 with the same environment hash, physical hash, train/test split, episode count, and algorithm config as the approved seed-42 baseline. If either is absent, generate it without changing parent code:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer.yaml `
  --device cuda `
  --seed 43 `
  --no-dashboard
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer.yaml `
  --device cuda `
  --seed 44 `
  --no-dashboard
```

Capture their run directories as `$baseline43` and `$baseline44` only after verifying metadata compatibility. A mismatched baseline blocks the multi-seed claim.

- [ ] **Step 7: Run V2-Combined seeds 43 and 44**

Run:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_feasible_dr_v2 `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml `
  --device cuda `
  --seed 43 `
  --no-dashboard
if ($LASTEXITCODE -ne 0) { throw "V2-Combined seed-43 experiment failed" }
$combined43 = (Get-Content outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json -Raw | ConvertFrom-Json).run_directory

python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_feasible_dr_v2 `
  --algorithm-config configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml `
  --device cuda `
  --seed 44 `
  --no-dashboard
if ($LASTEXITCODE -ne 0) { throw "V2-Combined seed-44 experiment failed" }
$combined44 = (Get-Content outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/latest_run.json -Raw | ConvertFrom-Json).run_directory
```

- [ ] **Step 8: Evaluate and report the three-seed confirmation**

Run:

```powershell
python scripts/check_feasible_dr_v2_gate.py `
  --multiseed-run $combined42 `
  --multiseed-run $combined43 `
  --multiseed-run $combined44 `
  --multiseed-baseline $baseline42 `
  --multiseed-baseline $baseline43 `
  --multiseed-baseline $baseline44 `
  --multiseed-output outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/multiseed_gate_report.json
if ($LASTEXITCODE -ne 0) { throw "V2 multi-seed gate failed" }
```

Report each seed plus mean, median, population standard deviation, and interquartile range. Include the exact V2 and baseline run IDs, Git commits, config hashes, environment hashes, and source-artifact hashes.

- [ ] **Step 9: Recheck isolation and hand off without promotion**

Repeat Task 9 Step 4. Confirm all V2 results remain under the V2 working-run directory and every formal test used `best_model.pt`. Do not run `scripts/manage_results.py promote`; promotion requires explicit owner acceptance after reviewing the seed-42 and multi-seed reports.
