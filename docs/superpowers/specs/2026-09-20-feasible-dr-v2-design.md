# MATD3 Hierarchical Transformer Feasible DR V2 Design

## 1. Purpose

This design defines the second isolated improvement to the hierarchical Transformer feasible-DR experiment.

V1 successfully removed the mismatch between the MG2/MG3 DR action requested by the actor and the action physically executed by the environment. On the 55-day seed-42 formal test, V1 achieved:

- MG2 formal-test DR projection rate: `0%`.
- MG3 formal-test DR projection rate: `0%`.
- MG3 departure target success: `100%`.
- MG3 mean departure shortfall: effectively `0 kWh/day`.
- MG2 occupied comfort compliance: `100%`.
- Strict feasible-day rate: `100%`.
- Unmet energy: `0 kWh`.

V1 did not pass the complete seed-42 acceptance gate because:

- MG2 joint terminal thermal success fell from `45.45%` to `25.45%`.
- MG3 valley charging share fell from `97.40%` to `90.94%`.
- Mean terminal task penalty increased from `35.25` to `59.31` currency/day.
- Mean total operating cost increased from `874.63` to `895.08` currency/day.
- Mean 16:00–21:00 grid import increased from `269.54` to `347.68 kWh/day`.
- Mean peak-load reduction relative to the fixed DR reference worsened from `-4.07%` to `-7.61%`.

At the same time, V1's non-terminal operating cost improved from `839.38` to `835.77` currency/day. This indicates that the main failure is terminal recovery and timing, not ordinary dispatch cost.

V2 must preserve the hard feasibility gains of V1 while improving terminal recovery and economically timed DR behavior. It must remain an algorithm-only study: no environment physics, reward, action-space, settlement, market-clearing, or data-split change is permitted.

## 2. Experimental isolation

Create a separate algorithm variant:

```text
slug: matd3_hierarchical_transformer_feasible_dr_v2
display_name: MATD3 Hierarchical Transformer Feasible DR V2
action_semantics: single_dispatch_dr_adaptive_reference_v2
```

Required layout:

```text
algorithms/matd3_hierarchical_transformer_feasible_dr_v2/
├── __init__.py
├── plugin.py
├── algorithm.py
├── actor.py
├── adaptive_dr.py
└── training.py

configs/algorithms/matd3_hierarchical_transformer_feasible_dr_v2.yaml
tests/test_matd3_hierarchical_transformer_feasible_dr_v2.py
```

V2 may subclass and reuse V1, but must not modify:

- `algorithms/matd3/`
- `algorithms/matd3_hierarchical_transformer/`
- `algorithms/matd3_hierarchical_transformer_feasible_dr/`
- `configs/algorithms/matd3.yaml`
- `configs/algorithms/matd3_hierarchical_transformer.yaml`
- `configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml`
- environment physics or constraints
- shared reward definitions
- the P2P market, settlement, or Shapley implementation
- the external two-dimensional action interface
- observation dimensions or field order
- train, validation, or formal-test splits

Working results must be written under:

```text
outputs/working_runs/matd3_hierarchical_transformer_feasible_dr_v2/runs/
```

V1 results remain the direct feasibility baseline. The original hierarchical Transformer remains the economic comparison baseline.

## 3. Goals

### 3.1 Required V1 invariants

V2 must retain:

1. MG2 and MG3 actions are generated within their current physical feasible intervals.
2. Exploration noise and target-policy smoothing are applied in latent space before final mapping.
3. MG3 latest-feasible charging floor remains a hard reliability lower bound.
4. Replay stores environment-compatible actions, not latent preferences.
5. Centralized critics consume the existing projected joint effective action.
6. MG1 architecture and action mapping remain identical to V1.
7. Checkpoints are isolated by slug, action semantics, observation schema, projection signature, and complete V2 configuration.

### 3.2 V2 improvements

V2 adds two independently configurable mechanisms:

1. **MG2 adaptive recovery anchor**: gradually biases the feasible HVAC action toward a terminal-recovery power when recovery pressure becomes material.
2. **MG3 economic charging reference**: changes the neutral latent preference from the feasible interval midpoint to a price- and system-net-load-aware charging reference while preserving the V1 catch-up floor.

V2 also repairs the diagnostic path so training records whether behavior came from the latent actor, a hard feasibility floor, or an adaptive reference.

## 4. Non-goals

V2 does not include:

- a new reward term
- a demand charge or peak penalty
- a P2P acceptance reward
- an auxiliary actor loss
- a Lagrangian multiplier or constraint critic
- a separate P2P offer action
- a new action dimension
- a new observation field
- a new forecast provider
- a longer observation horizon
- Transformer depth, head-count, embedding-width, or hidden-width changes
- broad hyperparameter tuning
- MG1-specific policy changes
- environment or reward changes presented as an algorithm-only improvement

Peak shaving, P2P acceptance, and battery behavior are V2 acceptance guardrails. They are not implemented by silently changing shared physics or rewards.

## 5. Considered approaches

### 5.1 Selected: adaptive reference plus learned residual

The feasible interval remains the hard action domain. V2 computes a physically or economically meaningful reference inside that interval. The actor's latent output chooses a residual position below or above the reference.

Advantages:

- preserves exact V1 feasibility
- preserves a meaningful actor gradient
- keeps the full feasible interval reachable
- separates reliability floors from learned preference
- adds no reward-weight tuning
- supports clean HVAC-only, EV-only, and combined ablations

### 5.2 Rejected for V2: auxiliary actor losses

Terminal, valley-charge, peak-load, and P2P auxiliary losses would directly optimize the observed metrics, but would add multiple scale-sensitive weights and make reward-versus-regularizer attribution difficult. This remains a later option if the reference-action design fails.

### 5.3 Rejected for V2: shared reward modification

Changing the shared reward would affect every algorithm and become a cross-algorithm environment study. It is outside V2's algorithm-only scope.

## 6. Common action data flow

```text
existing observation
→ hierarchical Transformer
→ latent ESS and DR actions
→ latent exploration or target-smoothing noise
→ V1 hard feasible bounds and reliability floor
→ V2 adaptive reference
→ learned residual mapping around the reference
→ environment-compatible raw action
→ existing differentiable joint-action projector
→ replay and centralized critics
```

The external action remains:

```text
[a_ess, a_dr]
```

The ESS channel is unchanged by the V2 mapping. It may learn a different policy during retraining because DR behavior changes, so ESS, peak-import, and P2P metrics remain mandatory guardrails.

## 7. Common residual mapping

Let:

```text
z = clip(z_dr, -1, 1)
lower <= reference <= upper
```

Define the V2 residual map:

```text
positive = max(z, 0)
negative = max(-z, 0)

mapped = reference
       + positive * (upper - reference)
       - negative * (reference - lower)
```

Properties:

- `z = -1` selects `lower`.
- `z = 0` selects `reference`.
- `z = 1` selects `upper`.
- Every `z` maps inside `[lower, upper]`.
- The map is continuous everywhere and piecewise differentiable.
- Both sides preserve non-zero latent gradients when their corresponding interval width is non-zero.
- Degenerate intervals return the fixed bound without NaN or Inf.

This replaces V1's fixed midpoint interpretation while retaining the whole feasible action set.

## 8. MG2 adaptive terminal-recovery anchor

### 8.1 Inputs

Use only existing MG2 observation fields:

```text
terminal_air_gap
terminal_mass_gap
time_remaining
thermal_recovery_pressure
hvac_feasible_min
hvac_feasible_max
```

Use the existing HVAC physical configuration already passed to the actor:

```text
c_air_kwh_per_c
c_mass_kwh_per_c
cop_heat
cop_cool
heating_max_kw
cooling_max_kw
```

Observation de-normalization is fixed by the existing observation contract:

```text
air_gap_c = terminal_air_gap * 40
mass_gap_c = terminal_mass_gap * 40
remaining_hours = time_remaining * episode_length * delta_t
```

### 8.2 Recovery power

Calculate the average thermal power needed to restore the initial thermal state over the remaining horizon:

```text
required_thermal_kw =
    (air_gap_c * c_air_kwh_per_c
     + mass_gap_c * c_mass_kwh_per_c)
    / max(remaining_hours, delta_t)
```

Convert to signed electrical power:

```text
if required_thermal_kw >= 0:
    recovery_electric_kw = required_thermal_kw / cop_heat
else:
    recovery_electric_kw = required_thermal_kw / cop_cool
```

Normalize using the same signed-action convention as the environment:

```text
if recovery_electric_kw >= 0:
    recovery_raw = recovery_electric_kw / heating_max_kw
else:
    recovery_raw = recovery_electric_kw / cooling_max_kw
```

The physical anchor is:

```text
recovery_anchor = clip(recovery_raw, feasible_lower, feasible_upper)
```

### 8.3 Smooth activation gate

Use a cubic smoothstep:

```text
smoothstep(x; edge0, edge1):
    u = clip((x - edge0) / (edge1 - edge0), 0, 1)
    return u*u*(3 - 2*u)
```

Pressure gate:

```text
pressure_gate = smoothstep(
    thermal_recovery_pressure,
    recovery_pressure_start,
    recovery_pressure_full,
)
```

Time gate:

```text
time_gate = 1 - smoothstep(
    remaining_hours,
    recovery_full_hours,
    recovery_start_hours,
)
```

Required ordering:

```text
0 <= recovery_full_hours < recovery_start_hours <= episode horizon
0 <= recovery_pressure_start < recovery_pressure_full
```

Final gate:

```text
recovery_gate = pressure_gate * time_gate
```

Default first-experiment settings:

```yaml
recovery_pressure_start: 0.35
recovery_pressure_full: 0.85
recovery_start_hours: 6.0
recovery_full_hours: 2.0
```

These values must remain configuration fields and must be tested at zero, partial, and full activation.

### 8.4 HVAC reference and action

The actor retains its V1 free feasible action:

```text
preference = (z + 1) / 2
free_action = lower + preference * (upper - lower)
```

Blend toward the recovery anchor:

```text
adaptive_reference =
    (1 - recovery_gate) * free_action
    + recovery_gate * recovery_anchor
```

For MG2, `adaptive_reference` is the final action. This blend is intentional: MG2 already uses the complete latent preference over the instantaneous feasible interval; V2 only adds terminal recovery pressure rather than changing the semantic meaning of latent zero.

Requirements:

- zero gate exactly reproduces V1 MG2 output
- full gate selects the clipped recovery anchor
- partial gate is a convex combination inside the feasible interval
- action remains finite and differentiable in latent preference when gate is below one
- action does not violate instantaneous comfort bounds
- the implementation does not claim a mathematical terminal guarantee

## 9. MG3 economic charging reference

### 9.1 Inputs

Use existing MG3 fields:

```text
remaining_energy
hours_to_departure
ev_available
ev_feasible_min
ev_feasible_max
deadline_pressure
grid_buy_price
price_t+1 ... price_t+H
load, pv, wind
load_t+1, pv_t+1, wind_t+1 ...
peer net-load fields and peer net-load forecasts
forecast validity masks
```

The configured Transformer horizon is eight hours. The EV availability window is seven hours, so the current observation provides enough forecast coverage for every pre-departure decision. V2 construction must reject a configuration whose usable forecast horizon is shorter than the maximum EV remaining-step count.

### 9.2 Physical reconstruction

The V2 actor receives read-only normalization metadata from the environment when it is constructed:

```text
normalizers_by_microgrid
price_norm_by_microgrid
```

This metadata reconstructs current and forecast physical quantities without adding observation fields.

For each candidate slot from the current step through departure:

```text
local_net_load_kw = load_kw - pv_kw - wind_kw
system_net_load_kw = local_net_load_kw + sum(peer_net_load_kw)
price = local_buy_price
```

Forecast-invalid slots and slots at or after departure are masked out.

### 9.3 Stable economic weights

Normalize price and system net load within the remaining valid slot set:

```text
price_score_t =
    (price_t - min(price)) / max(max(price) - min(price), epsilon)

net_score_t =
    (system_net_load_t - min(system_net_load))
    / max(max(system_net_load) - min(system_net_load), epsilon)
```

Compute stable softmax weights:

```text
logit_t = -price_weight * price_score_t
          -net_load_weight * net_score_t

weight_t = softmax(logit over valid remaining slots)
```

Default first-experiment settings:

```yaml
price_weight: 2.0
net_load_weight: 1.0
softmax_temperature: 1.0
```

The implementation applies temperature as:

```text
softmax(logit / softmax_temperature)
```

Required validation:

```text
price_weight >= 0
net_load_weight >= 0
softmax_temperature > 0
```

When all prices and net loads are equal, weights are uniform.

### 9.4 Current reference power

Convert remaining battery energy into remaining grid-side charging energy:

```text
remaining_grid_energy_kwh = remaining_battery_energy_kwh / eta_charge
```

The provisional current reference is:

```text
reference_kw = remaining_grid_energy_kwh
               * current_weight
               / delta_t
```

V1's latest-feasible floor remains the hard lower bound:

```text
lower = max(environment_feasible_min, latest_feasible_floor)
upper = environment_feasible_max
reference = clip(reference_kw / max_charge_kw, lower, upper)
```

If the EV is unavailable:

```text
lower = upper = reference = 0
raw environment action = -1
```

If only one valid charging slot remains, the reference equals the feasible amount needed now, clipped by the physical upper bound.

### 9.5 Learned residual around reference

Use the common residual map in unit charging space:

```text
unit_action = residual_map(z, lower, reference, upper)
raw_action = 2 * unit_action - 1
```

The actor can still select any action in the reliable feasible interval. The reference changes only the semantic meaning of latent zero.

The economic reference does not replace the catch-up floor. Reliability always has priority over price or system-load timing.

## 10. Exploration and target-policy smoothing

V2 retains V1 ordering:

```text
latent action
→ add exploration or clipped target noise
→ clamp latent to [-1, 1]
→ compute hard feasible bounds and references
→ map noisy latent around the reference
```

No noise may be applied after the V2 mapping.

Forced large-noise tests must prove that rollout and target actions remain within the final hard interval.

## 11. Training behavior

V2 uses the existing MATD3 critic targets, delayed actor updates, replay semantics, gradient clipping, optimizer settings, and target-network updates.

V2 needs a local `training.py` because V1's update path does not call mapping diagnostics when it separately invokes `forward_latent` and `map_latent_actions`. The V2 local training module must remain behaviorally identical to V1 except for:

1. invoking the V2 actor and transform
2. collecting V2 mapping diagnostics on the configured diagnostic schedule
3. preserving latent-space exploration and target smoothing

No shared trainer, evaluator, settlement, or plotter is copied into the algorithm package.

Replay continues to store:

```text
[environment-compatible ESS raw action,
 environment-compatible DR raw action]
```

## 12. Diagnostics

### 12.1 Common fields

Record on scheduled actor-update batches:

```text
dr_latent_mean
dr_latent_std
dr_mapped_mean
dr_feasible_min_mean
dr_feasible_max_mean
dr_interval_width_mean
dr_lower_boundary_rate
dr_upper_boundary_rate
dr_effective_boundary_rate
dr_projection_rate
dr_projection_delta_mean
```

`dr_effective_boundary_rate` is computed on mapped actions, not latent preferences.

### 12.2 MG2 fields

```text
hvac_recovery_anchor_mean
hvac_recovery_gate_mean
hvac_recovery_gate_active_rate
hvac_recovery_gate_full_rate
hvac_terminal_air_gap_mean_c
hvac_terminal_mass_gap_mean_c
hvac_terminal_recovery_pressure_mean
```

### 12.3 MG3 fields

```text
ev_mandatory_floor_mean
ev_mandatory_floor_active_rate
ev_economic_reference_mean
ev_reference_above_floor_mean
ev_latent_reference_deviation_mean
ev_deadline_pressure_mean
ev_available_rate
ev_available_boundary_rate
```

MG3 boundary metrics used for policy interpretation must exclude unavailable hours. Unavailable hours still map to raw `-1`, but they must not be counted as learned saturation.

### 12.4 Formal-test outputs

The shared formal-test pipeline already saves hourly trajectories, DR shift metrics, paper metrics, and comparison figures. V2 implementation must not change shared evaluator behavior merely to improve reported numbers.

If V2-specific diagnostics are needed in saved formal-test artifacts, add them only through algorithm-neutral optional fields and matching regression tests. Do not make the evaluator inspect private actor internals.

## 13. Configuration

Initial V2 configuration:

```yaml
algorithm:
  name: matd3_hierarchical_transformer_feasible_dr_v2

  adaptive_dr:
    enabled: true
    interval_epsilon: 1.0e-6
    exploration_in_latent_space: true
    target_smoothing_in_latent_space: true

    agents:
      mg1: false
      mg2: true
      mg3: true

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

All unrelated learning and network hyperparameters remain identical to V1 for the first controlled experiment.

## 14. Ablation contract

The mechanisms must be independently configurable without code changes.

### 14.1 V2-HVAC

```yaml
hvac.recovery_anchor_enabled: true
ev.economic_reference_enabled: false
```

Purpose: isolate HVAC terminal-recovery improvement while retaining V1 EV behavior.

### 14.2 V2-EV

```yaml
hvac.recovery_anchor_enabled: false
ev.economic_reference_enabled: true
```

Purpose: isolate EV valley scheduling and rebound behavior while retaining V1 HVAC behavior.

### 14.3 V2-Combined

```yaml
hvac.recovery_anchor_enabled: true
ev.economic_reference_enabled: true
```

Purpose: measure interaction after both isolated mechanisms satisfy their focused gates.

The implementation plan must define how these configurations are materialized reproducibly. Checkpoints from one ablation must be rejected by a differently configured instance.

## 15. Checkpoint identity

Each V2 checkpoint must include:

```text
algorithm_slug
action_semantics
projection_signature
adaptive_dr_config
observation_schemas
state_schema
normalization_signature
update_steps
```

Loading must reject:

- original Transformer checkpoints
- V1 checkpoints
- a different V2 adaptive-DR configuration
- a different observation schema
- a different projection signature
- incompatible normalization metadata

## 16. Functional requirements

1. Every actor still supplies exactly two environment actions.
2. MG1 deterministic output matches V1 exactly when weights and observations match.
3. Zero MG2 recovery gate reproduces V1 MG2 output exactly.
4. Full MG2 recovery gate returns the clipped recovery anchor.
5. Partial MG2 gate returns a convex in-interval blend.
6. MG3 latent `-1`, `0`, and `1` select the reliable lower bound, economic reference, and upper bound respectively.
7. Equal remaining price and net-load signals produce a uniform EV reference schedule.
8. A cheaper/lower-net-load current slot receives at least as much reference charging as a more expensive/higher-net-load slot, subject to hard bounds.
9. EV unavailability produces zero charging and raw action `-1`.
10. EV catch-up floor cannot be weakened by the economic reference.
11. Exploration and target noise cannot create infeasible actions.
12. Non-degenerate action maps produce finite, useful latent gradients.
13. Degenerate intervals remain stable.
14. Replay stores mapped environment actions.
15. V2 diagnostics distinguish actor preference, hard floor, and adaptive reference.

## 17. Unit and property-test requirements

Tests must cover at least:

### 17.1 Common residual mapping

1. `z=-1`, `z=0`, and `z=1` select lower, reference, and upper.
2. Ten thousand random valid `(lower, reference, upper, z)` tuples remain in interval.
3. Degenerate intervals are exact and finite.
4. Gradients are finite on both sides of the reference.

### 17.2 HVAC recovery

5. Zero terminal gap produces a zero recovery anchor.
6. Positive gap requests heating and negative gap requests cooling.
7. Recovery power uses the configured capacitances and COPs.
8. Pressure below start produces a zero pressure gate.
9. Pressure above full and time below full produce a full gate.
10. Gate monotonicity holds as pressure rises or remaining time falls.
11. Zero gate exactly matches V1 output.
12. Full gate selects the clipped recovery anchor.
13. Random HVAC inputs remain inside instantaneous feasible bounds.

### 17.3 EV economic reference

14. Equal signals produce uniform weights.
15. Lower price increases the current-slot weight.
16. Lower system net load increases the current-slot weight.
17. Invalid and post-departure forecast slots receive zero weight.
18. One remaining slot allocates the required feasible amount now.
19. Reference never falls below the latest-feasible floor.
20. Reference never exceeds the physical upper bound.
21. Unavailable EV maps to zero charge/raw `-1`.
22. Large exploration and target noise remain feasible.

### 17.4 Diagnostics and identity

23. Mapping diagnostics appear during a scheduled training update.
24. MG3 unavailable hours do not count toward available boundary rate.
25. No diagnostic field is NaN or Inf.
26. Save/load preserves deterministic actions.
27. V2 rejects V1 and mismatched-ablation checkpoints.
28. MG1 output remains exactly equal to V1.

## 18. Pipeline verification

Before implementation completion:

1. Run the focused V2 tests.
2. Run V1 and original Transformer regression tests.
3. Run `pytest -q`.
4. Run a 3-episode shared-pipeline smoke experiment using the V2 combined configuration.
5. Confirm the selected plugin and V2 YAML.
6. Confirm formal test uses `best_model.pt`.
7. Confirm formal test, coalition MILP, Equal/Shapley settlement, summaries, paper artifacts, and plotting finish.
8. Confirm 24 PNG and 24 PDF figures under the current plotting contract.
9. Confirm outputs use the V2 working-run directory.
10. Confirm canonical MATD3, original Transformer, and V1 packages have zero implementation diff attributable to V2.

## 19. Seed-42 experimental sequence

Do not run all full experiments at once.

### 19.1 Run V2-HVAC first

Hard focused gate:

```text
MG2 formal-test projection rate <= 1%
MG2 occupied comfort >= 99.5%
MG2 terminal thermal success >= 45.45%
MG3 departure success = 100%
MG3 shortfall <= 0.05 kWh/day
```

If HVAC terminal success does not recover, stop before V2-EV and diagnose anchor activation, recovery direction, terminal gaps, and available recovery power.

### 19.2 Run V2-EV second

Hard focused gate:

```text
MG3 formal-test projection rate <= 1%
MG3 departure success = 100%
MG3 shortfall <= 0.05 kWh/day
MG3 valley charging share >= 95%
mean peak-load reduction >= -4.07%
```

If reliability falls, treat it as an implementation defect: the economic reference may not bypass or weaken the latest-feasible floor.

### 19.3 Run V2-Combined third

Run only after both focused variants satisfy their gates.

## 20. Combined seed-42 acceptance gate

### 20.1 Reliability and feasibility

```text
MG2 formal-test projection rate <= 1%
MG3 formal-test projection rate <= 1%
MG2 occupied comfort >= 99.5%
MG2 terminal thermal success >= 45.45%; target >= 55%
MG3 departure success = 100%
MG3 shortfall <= 0.05 kWh/day
strict feasible-day rate = 100%
unmet energy = 0 kWh
MG1 production completion >= 99.5%
```

MG1 terminal silo success is not directly controlled by V2, but must be reported. A result below V1's `41.82%` is a regression warning; a result below `35%` blocks multi-seed continuation until the cause is understood.

### 20.2 Economics and operation

```text
total operating cost <= 874.63 currency/day
non-terminal operating cost <= 839.38 currency/day
terminal task penalty <= 35.25 currency/day
EV valley charging share >= 95%
16:00-21:00 grid import <= 283.02 kWh/day
mean peak-load reduction >= -4.07%; target >= 0%
P2P energy must not fall more than 5% below 1105.03 kWh/day
P2P offer acceptance must not fall more than 5 percentage points below 36.49%
ESS equivalent cycles must not exceed 1.1665 cycles/day
renewable utilization must not fall more than 0.5 percentage points below 97.11%
```

The 16:00–21:00 grid-import metric must be calculated from saved hourly trajectories as the daily sum of all three microgrid grid-import powers over those six hours. The definition must be recorded in the result analysis.

## 21. Multi-seed confirmation

Run seeds 43 and 44 only after the V2-Combined seed-42 gate passes.

Requirements:

- every seed has MG2/MG3 projection at or below 1%
- every seed has MG3 departure success at 100%
- every seed has MG2 occupied comfort at or above 99.5%
- median MG2 terminal thermal success is at least 45.45%
- no seed has MG2 terminal thermal success below 40%
- median EV valley charging share is at least 95%
- median total operating cost does not exceed the matching original Transformer median
- median 16:00–21:00 grid import does not exceed the matching original Transformer by more than 5%
- report each seed plus mean, median, standard deviation, and interquartile range

Do not promote any run automatically.

## 22. Failure interpretation

### 22.1 Projection reappears

Treat as an implementation defect. Check mapping order, latent noise position, unavailable-EV encoding, and target-actor mapping.

### 22.2 HVAC terminal success remains low

Inspect:

- recovery-anchor sign
- denormalized gap values
- recovery gate activation rate
- anchor clipping by the instantaneous feasible interval
- whether remaining physical capacity is sufficient
- air versus mass terminal success separately

Do not increase gate strength blindly before verifying these diagnostics.

### 22.3 EV reliability falls

Treat as an implementation defect. The economic reference must not reduce the latest-feasible lower bound.

### 22.4 EV valley share remains low

Inspect slot masks, physical price reconstruction, forecast alignment, softmax temperature, and whether the reference is frequently overridden by the floor. Tune only the V2 EV reference parameters, not the environment reward.

### 22.5 Peak import remains high while DR metrics improve

Separate DR load timing from ESS dispatch using saved hourly trajectories. Because peak-period grid import is a joint ESS/DR/P2P result, do not claim the DR reference alone caused the failure. If ESS behavior dominates, define a later isolated joint-dispatch experiment rather than expanding V2 mid-run.

### 22.6 Total cost remains high but non-terminal cost passes

Decompose terminal penalties by microgrid and terminal state. Do not hide terminal failure by weakening penalties or acceptance thresholds.

## 23. Reproducibility and result lifecycle

Every run must record:

- V2 plugin slug and display name
- complete adaptive-DR configuration
- ablation identity: HVAC, EV, or Combined
- Git commit and dirty status
- environment and physical configuration hashes
- selected best checkpoint episode
- validation metric and score
- formal-test dates and seed
- all gate metrics and their exact definitions

Promotion is allowed only after:

1. the working tree identity is reviewable,
2. the seed-42 combined gate passes,
3. required multi-seed confirmation is complete, and
4. the user explicitly accepts the result.

## 24. Expected implementation boundaries

The implementation plan should organize work into independently reviewable tasks:

1. isolated V2 scaffold and configuration
2. common residual-map and configuration validation
3. MG2 recovery anchor
4. MG3 economic reference
5. diagnostic collection and EV availability masking
6. V2 actor, training integration, and strict checkpoint isolation
7. repository regression and shared-pipeline smoke verification
8. staged seed-42 ablation runs and gate analysis
9. multi-seed confirmation only after seed-42 acceptance

Each production behavior must be developed test-first. No task may modify canonical MATD3, the original Transformer, V1, shared environment physics, or shared rewards.
