# Feasible DR Transformer V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated hierarchical Transformer variant whose MG2 HVAC and MG3 EV actions are generated inside their current feasible intervals, with an EV latest-feasible charging floor and latent-space policy noise.

**Architecture:** A local `FeasibleDRTransform` maps each actor's learned DR preference into an environment-compatible raw action using feasible bounds already present in the observation. A local actor subclass exposes latent and mapped forward paths, while a local algorithm subclass owns rollout noise, target-policy smoothing, sampled diagnostics, and checkpoint identity so neither canonical MATD3 nor the existing Transformer experiment changes.

**Tech Stack:** Python 3, PyTorch, NumPy, PyYAML, pytest, existing shared MATD3 experiment pipeline.

**Spec:** `docs/superpowers/specs/2026-09-20-feasible-dr-transformer-design.md`

## Global Constraints

- Do not modify `algorithms/matd3/` or its public behavior.
- Do not modify `algorithms/matd3_hierarchical_transformer/`; the existing experiment remains a frozen comparison point.
- Do not modify environment physics, rewards, P2P clearing, settlement, action dimensionality, observation dimensionality, or data splits.
- The algorithm slug is exactly `matd3_hierarchical_transformer_feasible_dr`.
- The action semantics string is exactly `single_dispatch_dr_feasible_v1`.
- MG1 uses the existing Transformer action path unchanged.
- MG2 and MG3 policy noise is applied to latent DR preferences before feasible mapping.
- Replay stores two-dimensional environment-compatible actions, not latent preferences.
- Initial `ev.reliability_margin` is `0.0`; initial `hvac.recovery_anchor_enabled` is `false`.
- Normal outputs live only under `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/runs/`.
- Implementation is native in the current task; do not use subagents.

## Review Focus

1. EV exactly outside its availability window must map to zero charging even when the latent preference and exploration noise are maximal; Task 2 pins this with an unavailable-EV test.
2. Floating-point intervals whose width is at or below `interval_epsilon` must return the fixed bound without NaN, division, or unstable gradients; Task 2 pins this with a degenerate-interval test.
3. Heterogeneous observation shapes must remain supported by action selection; Task 4 pins this with a per-agent observation test.
4. Target-policy noise must not bypass the feasible transform during critic target construction; Task 4 exercises the same `map_noisy_latent` helper used by critic targets with forced large noise.
5. Checkpoint loading must reject both the old Transformer slug and a mismatched feasible-DR configuration; Task 5 pins both incompatibility cases.

---

### Task 1: Scaffold the isolated plugin and configuration

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr/__init__.py`
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr/plugin.py`
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py`
- Create: `configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml`
- Create: `tests/test_matd3_hierarchical_transformer_feasible_dr.py`

**Interfaces:**
- Consumes: `algorithms.registry.get_algorithm_spec(slug)` and `Matd3HierarchicalTransformer.from_env(env, config, device, seed)`.
- Produces: plugin slug `matd3_hierarchical_transformer_feasible_dr`, class `Matd3HierarchicalTransformerFeasibleDR`, and a dedicated YAML configuration.

- [ ] **Step 1: Write the failing registry/config test**

```python
from pathlib import Path

import yaml

from algorithms.registry import get_algorithm_spec


def test_feasible_dr_plugin_uses_its_own_config():
    spec = get_algorithm_spec("matd3_hierarchical_transformer_feasible_dr")
    config = yaml.safe_load(Path(spec.default_config_path).read_text(encoding="utf-8"))

    assert spec.slug == "matd3_hierarchical_transformer_feasible_dr"
    assert spec.algorithm_class.__name__ == "Matd3HierarchicalTransformerFeasibleDR"
    assert config["algorithm"]["name"] == spec.slug
    assert config["algorithm"]["feasible_dr"]["agents"] == {
        "mg1": False,
        "mg2": True,
        "mg3": True,
    }
```

- [ ] **Step 2: Run the test and verify the plugin does not exist**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py::test_feasible_dr_plugin_uses_its_own_config
```

Expected: failure reporting that `algorithms.matd3_hierarchical_transformer_feasible_dr` is unknown.

- [ ] **Step 3: Create the required isolated scaffold**

Run:

```powershell
python scripts/create_algorithm_variant.py --name matd3_hierarchical_transformer_feasible_dr --base matd3_hierarchical_transformer
```

Replace the generated class with an initial subclass:

```python
from __future__ import annotations

from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer


class Matd3HierarchicalTransformerFeasibleDR(Matd3HierarchicalTransformer):
    """Hierarchical Transformer with feasible MG2/MG3 DR actions."""

    action_semantics = "single_dispatch_dr_feasible_v1"
```

Set `plugin.py` to expose exactly one specification:

```python
from __future__ import annotations

from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_feasible_dr",
    display_name="MATD3 Hierarchical Transformer Feasible DR",
    algorithm_class=Matd3HierarchicalTransformerFeasibleDR,
    default_config_path=(
        PROJECT_ROOT
        / "configs"
        / "algorithms"
        / "matd3_hierarchical_transformer_feasible_dr.yaml"
    ),
    description="Hierarchical Transformer with feasible HVAC and deadline-safe EV DR actions.",
)
```

Copy the existing Transformer configuration and add this exact block under `algorithm`:

```yaml
  feasible_dr:
    enabled: true
    agents:
      mg1: false
      mg2: true
      mg3: true
    interval_epsilon: 1.0e-6
    exploration_in_latent_space: true
    target_smoothing_in_latent_space: true
    ev:
      catchup_floor_enabled: true
      reliability_margin: 0.0
    hvac:
      interval_mapping_enabled: true
      recovery_anchor_enabled: false
```

- [ ] **Step 4: Run the registry/config test**

Run:

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py::test_feasible_dr_plugin_uses_its_own_config
```

Expected: PASS.

- [ ] **Step 5: Commit the isolated scaffold**

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml tests/test_matd3_hierarchical_transformer_feasible_dr.py
git commit -m "feat: scaffold feasible DR transformer variant"
```

### Task 2: Implement and property-test the feasible DR transform

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr/feasible_dr.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr.py`

**Interfaces:**
- Consumes: observation tensors and schema names from the existing Transformer actor.
- Produces: `FeasibleDRTransform.forward(observation, latent_dr) -> raw_dr`, `FeasibleDRTransform.scalar_diagnostics(observation, latent_dr) -> dict[str, float]`, and `latest_feasible_ev_floor(remaining_energy, hours_to_departure, max_charge_kw, efficiency, delta_t, upper, reliability_margin) -> Tensor`.

- [ ] **Step 1: Write failing interval, deadline, edge-case, and gradient tests**

```python
import torch

from algorithms.matd3_hierarchical_transformer_feasible_dr.feasible_dr import (
    FeasibleDRTransform,
    latest_feasible_ev_floor,
)


def test_latest_feasible_ev_floor_preserves_future_schedule():
    floor = latest_feasible_ev_floor(
        remaining_energy=torch.tensor([[30.0]]),
        hours_to_departure=torch.tensor([[2.0]]),
        max_charge_kw=20.0,
        efficiency=1.0,
        delta_t=1.0,
        upper=torch.tensor([[1.0]]),
    )
    assert torch.allclose(floor, torch.tensor([[0.5]]))


def test_hvac_transform_maps_random_preferences_inside_bounds():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
    )
    generator = torch.Generator().manual_seed(42)
    lower = torch.rand(10_000, 1, generator=generator) * 2.0 - 1.0
    width = torch.rand(10_000, 1, generator=generator) * (1.0 - lower)
    upper = lower + width
    observation = torch.cat([lower, upper], dim=1)
    latent = torch.rand(10_000, 1, generator=generator) * 2.0 - 1.0

    raw = transform(observation, latent)

    assert torch.all(raw >= lower - 1.0e-7)
    assert torch.all(raw <= upper + 1.0e-7)


def test_ev_unavailable_maps_to_zero_charge():
    transform = FeasibleDRTransform.for_test(
        dr_type="ev_charging",
        schema=(
            "remaining_energy",
            "hours_to_departure",
            "ev_available",
            "ev_feasible_min",
            "ev_feasible_max",
        ),
        max_charge_kw=100.0,
        efficiency=0.95,
        episode_length=24,
    )
    observation = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0]])
    raw = transform(observation, torch.tensor([[1.0]]))
    assert torch.allclose(raw, torch.tensor([[-1.0]]))


def test_degenerate_interval_is_finite_and_exact():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
        interval_epsilon=1.0e-6,
    )
    latent = torch.tensor([[0.25]], requires_grad=True)
    raw = transform(torch.tensor([[0.3, 0.3]]), latent)
    raw.sum().backward()
    assert torch.allclose(raw, torch.tensor([[0.3]]))
    assert torch.isfinite(raw).all()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_non_degenerate_transform_has_nonzero_finite_gradient():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
    )
    latent = torch.tensor([[0.0]], requires_grad=True)
    raw = transform(torch.tensor([[-0.5, 0.75]]), latent)
    raw.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().item() > 0.0
```

- [ ] **Step 2: Run the transform tests and verify missing symbols**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "floor or transform or unavailable or degenerate"
```

Expected: import failure for `feasible_dr`.

- [ ] **Step 3: Implement the transform with explicit schema indices**

Implement:

```python
def latest_feasible_ev_floor(
    remaining_energy: torch.Tensor,
    hours_to_departure: torch.Tensor,
    *,
    max_charge_kw: float,
    efficiency: float,
    delta_t: float,
    upper: torch.Tensor,
    reliability_margin: float = 0.0,
) -> torch.Tensor:
    steps = torch.clamp(torch.ceil(hours_to_departure / delta_t), min=0.0)
    future_steps = torch.clamp(steps - 1.0, min=0.0)
    future_energy = efficiency * max_charge_kw * future_steps * delta_t
    current_kw = torch.clamp(
        (remaining_energy - future_energy) / (efficiency * delta_t),
        min=0.0,
    )
    normalized = current_kw / max_charge_kw + reliability_margin
    return torch.minimum(torch.clamp(normalized, min=0.0), upper)
```

`FeasibleDRTransform` is an `nn.Module` that stores field indices as registered buffers, converts normalized `hours_to_departure` and `remaining_energy` back to physical units from the actor configuration, and maps:

```python
preference = (latent_dr.clamp(-1.0, 1.0) + 1.0) * 0.5
effective = lower + preference * (upper - lower)
```

For EV, replace `lower` with `maximum(lower, mandatory_floor)` and return `effective * 2.0 - 1.0`. For HVAC, return signed `effective`. For a disabled transform, return `latent_dr.clamp(-1.0, 1.0)`.

Provide `for_test` with the explicit scalar inputs used above.

- [ ] **Step 4: Run the focused transform tests**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "floor or transform or unavailable or degenerate"
```

Expected: PASS.

- [ ] **Step 5: Commit the transform**

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr/feasible_dr.py tests/test_matd3_hierarchical_transformer_feasible_dr.py
git commit -m "feat: map DR preferences into feasible intervals"
```

### Task 3: Add the feasible actor without changing MG1

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr/actor.py`
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr.py`

**Interfaces:**
- Consumes: `HierarchicalTransformerActor.forward(observation, collect_diagnostics=False) -> Tensor[B, 2]` and `FeasibleDRTransform`.
- Produces: `FeasibleDRActor.forward_latent`, `FeasibleDRActor.map_latent_actions`, and mapped `FeasibleDRActor.forward`.

- [ ] **Step 1: Write failing actor behavior tests**

```python
def test_mg1_feasible_actor_matches_parent_actor(env_and_config):
    env, parent_config, feasible_config = env_and_config
    parent = Matd3HierarchicalTransformer.from_env(env, parent_config, device="cpu", seed=42)
    feasible = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, feasible_config, device="cpu", seed=42
    )
    feasible.actors["mg1"].load_state_dict(parent.actors["mg1"].state_dict(), strict=False)
    observation, _ = env.reset(seed=42)
    tensor = torch.as_tensor(observation["mg1"], dtype=torch.float32).unsqueeze(0)
    expected = parent.actors["mg1"](tensor)
    actual = feasible.actors["mg1"](tensor)
    assert torch.allclose(actual, expected, atol=1.0e-7)


def test_mg2_and_mg3_actor_requests_are_feasible(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    for mg in ("mg2", "mg3"):
        tensor = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
        action = algorithm.actors[mg](tensor)
        latent = algorithm.actors[mg].forward_latent(tensor)
        remapped = algorithm.actors[mg].map_latent_actions(tensor, latent)
        assert torch.allclose(action, remapped)
        assert action.shape == (1, 2)
```

- [ ] **Step 2: Run the actor tests and verify the actor class is missing**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "mg1_feasible_actor or actor_requests"
```

Expected: failure because the new algorithm still constructs parent actors.

- [ ] **Step 3: Implement the actor wrapper and replace actors in the new algorithm constructor**

Implement:

```python
class FeasibleDRActor(HierarchicalTransformerActor):
    def forward_latent(self, observation, collect_diagnostics=False):
        return super().forward(observation, collect_diagnostics=collect_diagnostics)

    def map_latent_actions(self, observation, latent_actions):
        ess = latent_actions[:, 0:1].clamp(-1.0, 1.0)
        dr = self.feasible_dr(observation, latent_actions[:, 1:2])
        return torch.cat([ess, dr], dim=-1)

    def forward(self, observation, collect_diagnostics=False):
        latent = self.forward_latent(observation, collect_diagnostics=collect_diagnostics)
        action = self.map_latent_actions(observation, latent)
        if collect_diagnostics:
            self.last_diagnostics.update(
                self.feasible_dr.scalar_diagnostics(observation, latent[:, 1:2])
            )
        return action
```

The algorithm constructor replaces actors and target actors with `FeasibleDRActor` instances using the same network settings, then recreates actor optimizers and diagnostic parameter groups. Critics, replay, projector, and their state shapes do not change.

- [ ] **Step 4: Run actor behavior and gradient tests**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "actor or gradient"
```

Expected: PASS, including non-zero DR-head gradients for MG2 and MG3 and exact MG1 parity.

- [ ] **Step 5: Commit the feasible actor**

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr/actor.py algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py tests/test_matd3_hierarchical_transformer_feasible_dr.py
git commit -m "feat: add feasible DR transformer actors"
```

### Task 4: Implement latent-space rollout and training noise

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_feasible_dr/training.py`
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr.py`

**Interfaces:**
- Consumes: feasible actor latent/mapping methods, replay batches, projector, critics, optimizers, and diagnostic schedule.
- Produces: `select_feasible_actions(self, observations, noise_std)` and `update_feasible_dr(self, batch_size)`.

- [ ] **Step 1: Write failing rollout-noise, heterogeneous-observation, target-noise, and replay tests**

```python
def test_noisy_rollout_actions_remain_feasible(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    for _ in range(24):
        actions = algorithm.select_actions(observations, noise_std=2.0)
        observations, _, done, info = env.step(actions)
        for mg in ("mg2", "mg3"):
            assert not info["by_microgrid"][mg]["dr_projected"]
        if done:
            break


def test_select_actions_accepts_heterogeneous_observation_shapes(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    actions = algorithm.select_actions(observations, noise_std=0.2)
    assert set(actions) == set(env.agent_ids)
    assert all(action.shape == (2,) for action in actions.values())


def test_replay_stores_environment_actions_not_latent_preferences(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    state = env.global_state()
    actions = algorithm.select_actions(observations, noise_std=0.5)
    next_observations, rewards, done, _ = env.step(actions)
    algorithm.replay.add(
        observations,
        state,
        actions,
        sum(rewards.values()),
        next_observations,
        env.global_state(),
        done,
    )
    batch = algorithm.replay.sample(1, env.agent_ids, torch.device("cpu"))
    for mg in env.agent_ids:
        assert torch.allclose(batch["actions"][mg][0], torch.as_tensor(actions[mg]))
```

Test the exact helper used for target-policy smoothing:

```python
def test_target_noise_is_mapped_before_projection(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    state = torch.as_tensor(env.global_state(), dtype=torch.float32).unsqueeze(0)
    raw_actions = []
    for mg in env.agent_ids:
        observation = torch.as_tensor(
            observations[mg], dtype=torch.float32
        ).unsqueeze(0)
        actor = algorithm.target_actors[mg]
        latent = actor.forward_latent(observation)
        extreme_noise = torch.full_like(latent, 10.0)
        raw_actions.append(
            map_noisy_latent(
                actor,
                observation,
                latent,
                extreme_noise,
                noise_clip=algorithm.target_noise_clip,
            )
        )
    _, plans = algorithm.project_actions(state, torch.stack(raw_actions, dim=1))
    for index, mg in enumerate(env.agent_ids):
        if mg in ("mg2", "mg3"):
            assert not plans[index]["dr_projected"].any()
```

- [ ] **Step 2: Run the focused tests and verify inherited post-mapping noise fails**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "noisy_rollout or heterogeneous or replay_stores or target_noise"
```

Expected: at least noisy-rollout or target-noise fails with inherited behavior.

- [ ] **Step 3: Implement algorithm-local rollout and sampled update**

The shared helper used by rollout and target construction is:

```python
def map_noisy_latent(actor, observation, latent, noise, *, noise_clip):
    clipped_noise = noise.clamp(-noise_clip, noise_clip)
    noisy_latent = (latent + clipped_noise).clamp(-1.0, 1.0)
    return actor.map_latent_actions(observation, noisy_latent)
```

Rollout ordering:

```python
latent = actor.forward_latent(observation_tensor)
noise = self.rng.normal(0.0, noise_std, size=tuple(latent.shape))
noise_tensor = torch.as_tensor(noise, dtype=latent.dtype, device=latent.device)
noisy_latent = torch.clamp(latent + noise_tensor, -1.0, 1.0)
mapped = actor.map_latent_actions(observation_tensor, noisy_latent)
```

Target ordering:

```python
target_latent = target_actor.forward_latent(next_observation)
target_noise = torch.randn_like(target_latent).mul(self.target_policy_noise).clamp(
    -self.target_noise_clip,
    self.target_noise_clip,
)
target_raw = target_actor.map_latent_actions(
    next_observation,
    (target_latent + target_noise).clamp(-1.0, 1.0),
)
```

Actor optimization uses mapped current actions with gradients enabled:

```python
latent = actor.forward_latent(observation, collect_diagnostics=diagnostic_due)
active_action = actor.map_latent_actions(observation, latent)
projected, plans = self.project_actions(state, torch.stack(joint, dim=1))
loss = -critic.q1(state, projected.flatten(1)).mean()
```

Initialize `DiagnosticSchedule` and convert replay to `FastReplayBuffer` in the new constructor. Do not call `install_diagnostic_sampling`, because it installs post-mapping rollout and target noise.

- [ ] **Step 4: Run rollout, update, and diagnostic tests**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py -k "noisy_rollout or heterogeneous or replay_stores or target_noise or update or diagnostic"
```

Expected: PASS. Diagnostics include interval width, latent action statistics, projection rate, and MG3 mandatory-floor activation.

- [ ] **Step 5: Commit latent-space training behavior**

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr/training.py algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py tests/test_matd3_hierarchical_transformer_feasible_dr.py
git commit -m "feat: apply policy noise before feasible DR mapping"
```

### Task 5: Add checkpoint isolation and end-to-end plugin tests

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr/algorithm.py`
- Modify: `algorithms/matd3_hierarchical_transformer_feasible_dr/plugin.py`
- Modify: `tests/test_matd3_hierarchical_transformer_feasible_dr.py`

**Interfaces:**
- Consumes: configured actors and the existing Transformer checkpoint tensor payload.
- Produces: strict `save(path)` and `load(path)` for the new identity and feasible-DR configuration.

- [ ] **Step 1: Write failing checkpoint compatibility tests**

```python
def test_checkpoint_round_trip_and_incompatibility(env_and_config, tmp_path):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    before = algorithm.select_actions(observations, noise_std=0.0)
    path = tmp_path / "model.pt"
    algorithm.save(path)
    restored = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    restored.load(path)
    after = restored.select_actions(observations, noise_std=0.0)
    for mg in env.agent_ids:
        assert torch.allclose(torch.as_tensor(before[mg]), torch.as_tensor(after[mg]))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["algorithm_slug"] = "matd3_hierarchical_transformer"
    wrong_slug = tmp_path / "wrong_slug.pt"
    torch.save(payload, wrong_slug)
    with pytest.raises(ValueError, match="Incompatible algorithm checkpoint"):
        restored.load(wrong_slug)

    payload["algorithm_slug"] = "matd3_hierarchical_transformer_feasible_dr"
    payload["feasible_dr_config"] = {"enabled": False}
    wrong_config = tmp_path / "wrong_config.pt"
    torch.save(payload, wrong_config)
    with pytest.raises(ValueError, match="feasible DR configuration"):
        restored.load(wrong_config)
```

- [ ] **Step 2: Run the checkpoint test and verify inherited identity fails**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py::test_checkpoint_round_trip_and_incompatibility
```

Expected: FAIL because inherited save/load uses the old slug and omits `feasible_dr_config`.

- [ ] **Step 3: Implement strict checkpoint metadata**

Save these identity fields with actors, target actors, critics, target critics, and `update_steps`:

```python
{
    "algorithm_slug": "matd3_hierarchical_transformer_feasible_dr",
    "action_semantics": "single_dispatch_dr_feasible_v1",
    "projection_signature": self.projection_signature,
    "feasible_dr_config": self.feasible_dr_config,
    "observation_schemas": self.observation_schemas,
    "state_schema": self.state_schema,
}
```

`load` compares every identity field before loading tensors. Normalize configuration to plain JSON-compatible data so equality is deterministic.

- [ ] **Step 4: Run the full new-variant test file**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer_feasible_dr.py
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint isolation**

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr tests/test_matd3_hierarchical_transformer_feasible_dr.py
git commit -m "feat: isolate feasible DR checkpoints"
```

### Task 6: Verify the repository and shared experiment workflow

**Files:**
- Modify only if a test exposes an implementation defect: files created in Tasks 1-5.
- Inspect: the run directory named by `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/latest_run.json`.

**Interfaces:**
- Consumes: registered algorithm and shared `scripts/run_experiment.py` pipeline.
- Produces: passing repository tests and a short working-run artifact tree.

- [ ] **Step 1: Run focused baseline and new-variant tests**

```powershell
pytest -q tests/test_matd3_hierarchical_transformer.py tests/test_matd3_hierarchical_transformer_feasible_dr.py
```

Expected: PASS, proving existing Transformer behavior remains intact.

- [ ] **Step 2: Run the full test suite**

```powershell
pytest -q
```

Expected: PASS. Record unrelated pre-existing failures separately; do not edit unrelated code to conceal them.

- [ ] **Step 3: Run a short shared-pipeline smoke experiment**

Use the new plugin configuration with command-line smoke overrides:

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_feasible_dr --episodes 3 --test-episodes 3 --validation-interval 1 --validation-days 1 --device cpu --no-dashboard
```

Expected sequence:

```text
training
validation
best_model.pt selection
deterministic formal test
daily coalition MILP evaluation
Equal and Shapley settlement
summary generation
saved-result plotting
```

- [ ] **Step 4: Verify smoke artifacts and invariants**

Confirm:

```text
run_metadata.json names matd3_hierarchical_transformer_feasible_dr
config_snapshot.yaml contains the new algorithm.name
checkpoints/best_model.pt exists and is the formal-test checkpoint
test_summary.json and test_daily_metrics.csv exist
Equal and Shapley outputs exist
figures contains 15 PNG and 15 PDF files unless smoke mode explicitly suppresses PDFs
the run is under outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/runs/
```

Compare `git diff -- algorithms/matd3 algorithms/matd3_hierarchical_transformer` with the pre-implementation state and confirm neither reference package changed.

- [ ] **Step 5: Commit focused verification fixes only if needed**

```powershell
git add -- algorithms/matd3_hierarchical_transformer_feasible_dr configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml tests/test_matd3_hierarchical_transformer_feasible_dr.py
git commit -m "fix: complete feasible DR pipeline verification"
```

Do not create an empty commit when no verification fix is required.

- [ ] **Step 6: Record implementation evidence**

Report focused and full test results, smoke-run ID and absolute path, selected plugin/configuration, formal-test checkpoint, figure counts, reference-package integrity, and any smoke metrics explicitly labeled as smoke evidence rather than final 5,000-episode evidence.

### Task 7: Run the controlled acceptance experiments

**Files:**
- Inspect: the accepted run directories referenced by `outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/latest_run.json` after each seed.
- Do not modify source files unless a reproducible implementation defect is first demonstrated by a focused test.

**Interfaces:**
- Consumes: verified plugin, unchanged full training configuration, and the same date splits used by the current hierarchical Transformer result.
- Produces: seed-42 gate evidence followed, only if it passes, by seeds 43 and 44 confirmation evidence.

- [ ] **Step 1: Run the full seed-42 experiment**

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_feasible_dr --seed 42 --no-dashboard
```

Expected: a complete 5,000-episode working run using the plugin's default configuration and the shared validation, formal-test, coalition-MILP, settlement, summary, and plotting stages.

- [ ] **Step 2: Apply the seed-42 gate**

Read saved histories and formal-test outputs only. Confirm:

```text
MG3 formal-test DR projection rate <= 1%
MG3 late-training DR projection rate <= 5%
MG3 departure success = 55/55 days
MG3 departure shortfall <= 0.05 kWh/day
MG2 formal-test DR projection rate <= 1%
MG2 occupied comfort >= 99.5%
MG2 terminal thermal success >= 45.45%
MG1 production completion >= 99.5%
non-terminal operating cost degradation <= 2%
battery equivalent-cycle increase <= 5%
peak-period grid-import degradation <= 5%
P2P offer-acceptance degradation <= 5 percentage points
```

If a hard gate fails, stop the multi-seed rollout and report the failing evidence. Diagnose only within the failure routes defined by the specification.

- [ ] **Step 3: Run seeds 43 and 44 only after seed 42 passes**

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_feasible_dr --seed 43 --no-dashboard
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_feasible_dr --seed 44 --no-dashboard
```

- [ ] **Step 4: Apply the multi-seed confirmation gate**

Report each seed plus mean, median, standard deviation, and interquartile range. Confirm:

```text
each seed MG2/MG3 formal-test projection rate <= 1%
median MG3 departure success = 100%
no seed MG3 departure success < 98%
median non-terminal operating-cost degradation <= 1%
```

Do not promote any run automatically. Promotion remains an explicit user decision after reviewing the evidence.
