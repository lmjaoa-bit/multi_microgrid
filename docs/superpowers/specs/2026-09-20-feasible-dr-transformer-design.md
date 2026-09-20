# MATD3 Hierarchical Transformer Feasible DR V1 Design

## 1. Purpose

This design defines the first isolated improvement to the hierarchical Transformer experiment. It addresses the mismatch between the DR action requested by an actor and the action that the environment can physically execute, with emphasis on:

- MG3 EV actions that are frequently projected and sometimes miss the departure target.
- MG2 HVAC actions that are frequently projected.
- Actor gradients being learned through actions whose executed meaning differs from the original request.

The first version optimizes action feasibility and deadline reliability only. It does not add peak-shaving objectives, P2P objectives, new observation fields, or new reward terms.

## 2. Experimental isolation

Create a separate algorithm variant:

```text
slug: matd3_hierarchical_transformer_feasible_dr
display_name: MATD3 Hierarchical Transformer Feasible DR
action_semantics: single_dispatch_dr_feasible_v1
```

Required layout:

```text
algorithms/matd3_hierarchical_transformer_feasible_dr/
├── __init__.py
├── plugin.py
├── algorithm.py
├── actor.py
└── feasible_dr.py

configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml
```

The implementation may reuse the existing hierarchical Transformer experiment through imports or subclassing, but it must not modify:

- `algorithms/matd3/`
- `algorithms/matd3_hierarchical_transformer/`
- environment physics or constraints
- reward definitions
- the P2P market or settlement implementation
- the external two-dimensional action interface
- training, validation, or test splits

Working results must be written under:

```text
outputs/working_runs/matd3_hierarchical_transformer_feasible_dr/runs/
```

## 3. Selected approach

V1 uses feasible-interval reparameterization for MG2 and MG3 plus a latest-feasible EV charging floor. It does not use a projection penalty, constraint critic, Lagrangian multiplier, or terminal auxiliary loss.

This approach is selected because it removes the known actor/environment action mismatch with few new hyperparameters while retaining the existing environment, reward, observation schema, and critic semantics.

MG1 remains unchanged in V1 because its formal-test DR projection rate is already zero and its production completion is already satisfactory. MG1 renewable-alignment work belongs to a later experiment.

## 4. Existing information used by V1

No observation dimensions are added. The current observation already exposes the required signals.

MG2 fields include:

```text
terminal_air_gap
terminal_mass_gap
time_remaining
thermal_recovery_pressure
hvac_feasible_min
hvac_feasible_max
```

MG3 fields include:

```text
remaining_energy
hours_to_departure
ev_available
ev_feasible_min
ev_feasible_max
deadline_pressure
```

The Transformer continues to encode these fields in its DR token. The change occurs between the learned DR preference and the environment-compatible raw action.

## 5. Action data flow

```text
existing observation
→ Transformer encoder
→ latent DR preference z_dr in [-1, 1]
→ exploration or target-smoothing noise in latent space
→ FeasibleDRTransform
→ environment-compatible raw DR action in [-1, 1]
→ existing environment and differentiable joint-action projector
→ replay and centralized critics
```

The ESS channel remains unchanged.

The environment continues to receive:

```text
[a_ess, a_dr]
```

## 6. Feasible DR transform

Let the Transformer produce a latent preference `z_dr` and define:

```text
p = (z_dr + 1) / 2
```

`p` is the actor's relative preference within the current feasible interval.

### 6.1 MG2 HVAC

Let `lower` and `upper` be the current normalized signed HVAC bounds obtained from the existing observation.

```text
d_hvac = lower + p * (upper - lower)
a_dr = d_hvac
```

Requirements:

- `a_dr` must remain inside `[lower, upper]`.
- If the interval is degenerate, output the fixed bound.
- The mapping must be differentiable over non-degenerate intervals.
- Terminal thermal features remain inputs to the learned actor; V1 does not add a hand-written preheating or precooling rule.

### 6.2 MG3 EV

V1 calculates the minimum current charging power needed to preserve the existence of a future schedule that can still reach the departure target.

For remaining battery-energy requirement `E_remaining`, charging efficiency `eta`, maximum charging power `P_max`, time step `delta_t`, and `N` available charging steps including the current step:

```text
future_max_energy = eta * P_max * (N - 1) * delta_t

mandatory_power = max(
    0,
    (E_remaining - future_max_energy) / (eta * delta_t)
)
```

Normalize and constrain the floor:

```text
mandatory_ratio = clip(
    mandatory_power / P_max,
    feasible_min,
    feasible_max
)
```

The final effective EV charging request is:

```text
d_ev = mandatory_ratio + p * (feasible_max - mandatory_ratio)
a_dr = 2 * d_ev - 1
```

When departure is distant, the mandatory floor is normally zero and the actor retains flexibility to use low-price periods. If charging has been deferred too far, the floor rises so the actor cannot select an action that makes the remaining schedule mathematically impossible under full delivery.

This is not an unconditional departure guarantee. External supply shortfalls or delivered fractions below one can still cause an energy shortfall.

### 6.3 MG1 industrial mill

MG1 retains the current hierarchical Transformer DR output path in V1. No feasible transform or terminal rule is added for MG1.

## 7. Exploration and target-policy noise

Noise must be added before feasible mapping. Adding noise to the final environment action would reintroduce infeasible requests.

Rollout exploration:

```text
z_noisy = clip(z_dr + exploration_noise, -1, 1)
a_dr = FeasibleDRTransform(z_noisy, observation)
```

Target policy smoothing:

```text
target_z_noisy = clip(target_z + clipped_target_noise, -1, 1)
target_a_dr = FeasibleDRTransform(target_z_noisy, next_observation)
```

The ESS channel retains its existing noise behavior.

## 8. Replay and critic semantics

Replay continues to store the two-dimensional environment-compatible action:

```text
[a_ess, a_dr]
```

The shared replay interface is unchanged.

Critics continue to consume the existing differentiably projected joint effective action:

```text
[ESS_local, ESS_trade, DR_effective]
```

For deterministic MG2 and MG3 actions, the requested and effective DR values should differ only by floating-point tolerance.

Actor gradients must pass through `FeasibleDRTransform` and the existing joint-action projector.

## 9. Configuration

The initial configuration must include:

```yaml
algorithm:
  name: matd3_hierarchical_transformer_feasible_dr

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

  diagnostics:
    log_feasible_interval: true
    log_latent_action: true
    log_mandatory_floor: true
```

For the first controlled experiment:

- `reliability_margin` must remain zero.
- `recovery_anchor_enabled` must remain false.
- No auxiliary loss is introduced.
- Existing learning rates, Transformer layers, attention heads, hidden dimensions, reward settings, and environment settings remain unchanged.

## 10. Diagnostics

Record at least the following per microgrid:

```text
dr_latent_mean
dr_latent_std
dr_feasible_min_mean
dr_feasible_max_mean
dr_interval_width_mean
dr_requested_mean
dr_effective_mean
dr_projection_rate
dr_projection_delta_mean
dr_lower_boundary_rate
dr_upper_boundary_rate
```

MG3 additionally records:

```text
ev_mandatory_floor_mean
ev_mandatory_floor_active_rate
ev_deadline_pressure_mean
ev_departure_success_rate
ev_departure_shortfall_kwh
```

MG2 additionally records:

```text
hvac_terminal_recovery_pressure
occupied_comfort_rate
terminal_thermal_success_rate
terminal_air_gap
terminal_mass_gap
```

Diagnostics may be added to algorithm training history and saved test summaries, but they must not affect reward or physical execution.

## 11. Functional requirements

1. Each actor still supplies exactly two environment actions.
2. Deterministic MG2 and MG3 DR requests remain within the current feasible interval.
3. Rollout exploration and target-policy smoothing operate in latent space before feasible mapping.
4. Non-degenerate mappings preserve finite, useful gradients to the Transformer actor.
5. Degenerate intervals produce the exact fixed bound without NaN or Inf.
6. Replay stores environment-compatible actions, not latent preferences.
7. MG1 action behavior remains unchanged.
8. The algorithm uses a distinct checkpoint identity and cannot silently load an incompatible hierarchical Transformer checkpoint.

Each checkpoint must include:

```text
algorithm_slug
action_semantics
projection_signature
feasible_dr_config
observation_schemas
state_schema
```

## 12. Verification requirements

### 12.1 Unit tests

Tests must cover:

1. Random HVAC bounds always produce an in-interval action.
2. Random EV bounds always produce an in-interval action.
3. The EV latest-feasible charging floor is calculated correctly.
4. An unavailable EV produces zero charging.
5. Degenerate intervals are stable.
6. A latent preference of `-1` selects the lower admissible value.
7. A latent preference of `1` selects the upper admissible value.
8. Exploration noise cannot produce an infeasible final request.
9. Target-policy smoothing cannot produce an infeasible target request.
10. The transform has finite and non-zero latent gradients over non-degenerate intervals.
11. MG1 output matches the existing Transformer behavior.
12. Save/load preserves deterministic actions.

A vectorized property test must generate at least 10,000 valid random intervals and verify every mapped action lies inside its interval.

### 12.2 Pipeline verification

Before completion, verification must include:

- `pytest -q`
- a shared-pipeline smoke run of an appropriate short duration
- confirmation that the new plugin and its own YAML were selected
- confirmation that `best_model.pt` was used for formal testing
- confirmation that formal test, Equal/Shapley settlement, summaries, and saved-result plotting completed
- confirmation of 15 PNG and 15 PDF figures unless an explicitly approved PNG-only run is used
- confirmation that outputs use the new working-run directory
- confirmation that canonical MATD3 and the existing Transformer package were not changed

## 13. Single-seed experiment gate

Run the first ablation with seed 42 and the same training, validation, and formal-test dates as the current hierarchical Transformer result.

| Metric | Current Transformer | V1 requirement |
|---|---:|---:|
| MG3 formal-test DR projection rate | 42.08% | at most 1% |
| MG3 late-training DR projection rate | 83.42% | at most 5% |
| MG3 departure success | 52/55 days | 55/55 days |
| MG3 departure shortfall | 1.30 kWh/day | at most 0.05 kWh/day |
| MG2 formal-test DR projection rate | 26.06% | at most 1% |
| MG2 occupied comfort | 99.79% | at least 99.5% |
| MG2 terminal thermal success | 45.45% | must not decrease |
| MG1 production completion | 100.39% | at least 99.5% |

Economic and operational guardrails:

- Non-terminal operating cost must not worsen by more than 2%.
- Battery equivalent cycles must not increase by more than 5%.
- Peak-period grid import must not worsen by more than 5%.
- P2P offer acceptance must not decrease by more than five percentage points.

V1 is primarily a DR reliability experiment. A significant profit improvement is not required for acceptance.

## 14. Multi-seed confirmation gate

After the seed-42 gate passes, run at least seeds 42, 43, and 44.

Requirements:

- Every seed has deterministic MG2 and MG3 formal-test projection rates at or below 1%.
- Median MG3 departure success is 100%.
- No seed has MG3 departure success below 98%.
- Median non-terminal operating cost does not worsen by more than 1% relative to the corresponding existing Transformer runs.
- Report per-seed results plus mean, median, standard deviation, and interquartile range.

## 15. Failure interpretation

### Projection remains high

Treat this as an implementation defect. Check action-codec conversion, the location of exploration noise, replay semantics, and whether target actors bypass the feasible transform.

### Projection is eliminated but EV still misses departure

Check delivered energy and supply shortfalls. The next isolated experiment may add either a small EV reliability margin or a terminal-feasibility auxiliary loss, but not P2P or peak-shaving changes at the same time.

### MG2 projection is eliminated but terminal temperature does not improve

Proceed to a separate V1.1 experiment with either a recovery anchor or terminal-feasibility auxiliary loss.

### Reliability improves but cost degrades beyond the guardrail

Inspect the EV floor formula and remaining-step calculation first. Do not weaken the departure requirement to hide an overly conservative implementation.

## 16. Explicitly deferred work

V1 does not include:

- a separate P2P offer action
- a P2P acceptance or value auxiliary loss
- a peak-demand reward or demand charge
- a peak-risk critic
- a new 24-hour forecast input
- Transformer depth, head-count, or hidden-width changes
- broad hyperparameter retuning
- MG1 photovoltaic-alignment changes

These items require separate ablations after Feasible DR V1 is evaluated.
