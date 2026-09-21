# Predictive HVAC DR Transformer Design

**Date:** 2026-09-21

**Status:** Proposed design, pending user review

**Target algorithm slug:** `matd3_hierarchical_transformer_predictive_dr`

**Initial experiment mode:** HVAC-only, seed 42

## 1. Purpose

This document specifies an isolated successor to
`matd3_hierarchical_transformer_feasible_dr_v2`. The successor replaces V2's
static HVAC recovery anchor and forceful recovery gate with a finite-horizon,
2R2C-aware physical reference while preserving Transformer control authority.

The design addresses the observed V2 failure pattern:

- lowering the recovery-pressure thresholds increased HVAC intervention;
- terminal mass-temperature success improved, but terminal air-temperature and
  joint success became worse;
- MG2's own HVAC terminal penalty increased even though the aggregate terminal
  penalty fell because MG1 and MG3 happened to improve;
- the modified run introduced three MG3 supply-infeasible days;
- validation-profit-only checkpoint selection did not protect terminal recovery
  or strict feasibility.

The primary goal is not to make the recovery controller more aggressive. It is
to make its requested action physically meaningful and aligned with the exact
terminal success criteria.

## 2. Design decisions

### 2.1 Selected approach

Use a receding-horizon, deterministic physical reference that:

1. predicts air and thermal-mass evolution using the existing 2R2C model;
2. treats air and thermal mass as separate states with separate time scales;
3. limits mass recovery through the actual air-to-mass heat-transfer path;
4. makes air recovery dominant in the last two hours;
5. moves the semantic center of the Transformer's feasible action map toward the
   physical reference without overriding the Transformer's final decision;
6. uses constraint-aware validation checkpoint selection.

### 2.2 Rejected alternatives

#### More threshold tuning

Further tuning of `recovery_pressure_start`, `recovery_pressure_full`, or an
arbitrary mass coefficient is rejected as the primary solution. The two V2 runs
already show that stronger activation trades air success for mass success. The
static anchor and its activation pressure use the same mass-dominated energy
approximation, so changing thresholds cannot repair the underlying mismatch.

#### Full online MPC/QP safety layer

A per-action constrained optimization layer could provide stronger guarantees,
but it would add solver dependencies, substantially increase training cost, and
complicate batched GPU execution. It is deferred unless the proposed predictive
reference fails after the root hypothesis is tested.

### 2.3 Isolation requirements

The implementation shall create a new algorithm package. It shall not modify:

- `algorithms/matd3/`;
- `algorithms/matd3_hierarchical_transformer/`;
- `algorithms/matd3_hierarchical_transformer_feasible_dr/`;
- `algorithms/matd3_hierarchical_transformer_feasible_dr_v2/`;
- existing V1 or V2 algorithm configuration files;
- shared environment physics or rewards.

The new implementation may reuse existing classes through imports, but all new
behavioral overrides must live in the new package. Generic checkpoint-selection
support may be added to the shared training pipeline only if its default remains
exactly the current profit-only behavior for every existing algorithm.

### 2.4 Frozen parent contracts

Behavioral preservation uses the V2 HVAC-only implementation at commit
`f8ad10c` as the parent contract. For identical weights, observations, and
latent actions, V3 must preserve that parent's MG1, MG3, critic, replay, noise,
and update behavior. Only MG2's HVAC action transform and diagnostics may
change.

Economic comparison uses the saved parent-Transformer run:

```text
outputs/working_runs/matd3_hierarchical_transformer/runs/
MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42
```

The gate checker must load the reference metrics from this saved run and record
its run identity and artifact hash. It must not silently switch to a newer
`latest_run.json` target.

## 3. Scope

### 3.1 Included in the first implementation

- an isolated algorithm package and HVAC-only configuration;
- an MG2 predictive HVAC reference;
- a reference-centered residual action map;
- preservation tests for MG1 and MG3;
- predictive-HVAC diagnostics;
- generic, opt-in constrained checkpoint selection;
- a V3 gate report and HVAC-only formal experiment protocol.

### 3.2 Explicitly excluded

- changing the environment's 2R2C equations;
- changing the shared reward function or terminal penalties;
- adding new observation dimensions;
- changing the Transformer network architecture;
- changing MG1 scheduling behavior;
- adding the MG3 economic charging reference;
- implementing a supply-aware EV/ESS joint safety controller;
- running EV-only or Combined experiments before HVAC-only passes;
- claiming a mathematical terminal guarantee.

The three MG3 failure dates remain mandatory regression cases, but the first
implementation protects them through validation and acceptance gates rather
than changing the MG3 policy in the same experiment.

## 4. Package and configuration layout

Create the initial package with the repository scaffold command, then add only
the necessary local overrides:

```text
algorithms/matd3_hierarchical_transformer_predictive_dr/
├── __init__.py
├── plugin.py
├── algorithm.py
├── actor.py
├── predictive_hvac.py
└── training.py

configs/algorithms/
└── matd3_hierarchical_transformer_predictive_dr_hvac.yaml
```

Do not create EV-only or Combined configurations until the HVAC-only gate has
passed.

The plugin shall expose exactly one `ALGORITHM_SPEC`:

```text
slug: matd3_hierarchical_transformer_predictive_dr
display_name: MATD3 Hierarchical Transformer Predictive DR
default_config_path:
  configs/algorithms/matd3_hierarchical_transformer_predictive_dr_hvac.yaml
```

The algorithm must satisfy the shared interface in `algorithms/base.py`:

```text
from_env(env, config, device, seed)
select_actions(observations, noise_std)
update(batch_size)
save(path)
load(path)
replay.add(...)
len(replay)
```

## 5. Responsibilities by module

### 5.1 `predictive_hvac.py`

This module owns pure, batch-vectorized physical calculations:

- observation de-normalization for HVAC state and forecasts;
- recovery-phase weights;
- mass controllability and assist limits;
- 2R2C rollout;
- reference air-temperature trajectory;
- signed HVAC reference power;
- finite-value validation and air-only fallback;
- reference-center transition.

It must not depend on the replay buffer, critic, optimizer, environment object,
or file I/O. It must operate on tensors already placed on the actor's device and
must not call `.item()`, `.cpu()`, NumPy, or Python scalar conversion in the
forward path.

### 5.2 `actor.py`

The local actor shall reuse the existing hierarchical Transformer and replace
only MG2's DR action mapping.

Responsibilities:

- obtain the Transformer's latent DR output;
- call the predictive HVAC reference module for MG2;
- apply the reference-centered residual mapping;
- leave MG1 and MG3 behavior identical to the selected HVAC-only parent;
- expose diagnostic tensors without changing action semantics;
- preserve latent-space exploration and target-policy smoothing.

### 5.3 `training.py`

The local training classes shall reuse V2's replay and update behavior. Noise
must remain in latent space before physical action mapping for both exploration
and target smoothing.

No trainer, evaluator, settlement logic, or plotter may be copied into this
package.

### 5.4 `algorithm.py`

The algorithm wrapper shall:

- construct the local actors;
- reuse compatible critics and replay behavior;
- expose the shared algorithm protocol;
- isolate checkpoints with the new algorithm slug;
- save a normalized snapshot of `predictive_hvac` configuration;
- reject checkpoints from MATD3, the parent Transformer, V1, or V2;
- reject checkpoints whose predictive-HVAC contract is incompatible.

## 6. Observation contract

The first implementation must use the existing observation fields only.

Required MG2 fields:

```text
t_air
t_mass
t_out
comfort_lower_margin
comfort_upper_margin
occupied
is_weekend
terminal_air_gap
terminal_mass_gap
time_remaining
hvac_feasible_min
hvac_feasible_max
load
load_t+1 ... load_t+H
temperature_t+1 ... temperature_t+H
forecast_valid_t+1 ... forecast_valid_t+H, when enabled
hour_sin
hour_cos
```

Reconstruction uses the current repository normalization contract:

```text
T_air_c = t_air * 40
T_mass_c = t_mass * 40
T_out_c = t_out * 40
air_gap_c = terminal_air_gap * 40
mass_gap_c = terminal_mass_gap * 40
remaining_hours = time_remaining * episode_length * delta_t
```

The initial states are reconstructed as:

```text
T_air_initial = T_air + air_gap
T_mass_initial = T_mass + mass_gap
```

Physical load and temperature forecasts must be reconstructed using the same
normalization metadata already provided to V2 actors. The implementation must
validate every required schema index at construction time.

The current observation contract does not expose future GHI. Because recovery
begins only in the last six hours of the day, the first implementation sets
future solar gains to zero and relies on hourly receding-horizon correction.
This approximation must be recorded in checkpoint metadata and the run config.
It must not be hidden or described as exact solar forecasting.

## 7. Physical model

Use the environment's exact state convention and signs.

Let signed electrical HVAC power be `P_k`, positive for heating and negative
for cooling. Convert it to thermal power as:

```text
Q_hvac,k = P_k * cop_heat,  when P_k >= 0
Q_hvac,k = P_k * cop_cool,  when P_k < 0
```

Air evolution:

```text
T_air,k+1 = T_air,k
  + delta_t / C_air * (
      (T_out,k - T_air,k) / R_out_air
      + (T_mass,k - T_air,k) / R_air_mass
      + Q_internal,k
      + Q_solar_air,k
      + Q_hvac,k
    )
```

Thermal-mass evolution:

```text
T_mass,k+1 = T_mass,k
  + delta_t / C_mass * (
      (T_air,k - T_mass,k) / R_air_mass
      + Q_solar_mass,k
    )
```

The current environment uses explicit Euler integration. Consequently, action
`P_k` changes `T_air,k+1` but cannot change `T_mass,k+1` directly. Its first
effect on thermal mass occurs through the next interval's air-to-mass transfer.
The planner must preserve this one-step delay.

Internal gains use the existing environment convention:

```text
Q_internal,k = base_load_kw,k * occupied_internal_fraction
```

or the unoccupied fraction outside occupied hours. Future occupied state is
derived from the current hour and the configured occupied-hours set.

## 8. Recovery horizon and phases

Define:

```text
H = min(
  ceil(remaining_hours / delta_t),
  recovery_horizon_steps,
  usable_forecast_steps,
)
```

Default recovery horizon is six hours. The configured forecast horizon must be
at least six steps.

### 8.1 More than six hours remaining

The physical reference is diagnostic only. MG2 exactly reproduces the parent's
free feasible action mapping.

### 8.2 Two to six hours remaining

The planner gradually corrects both states. Thermal mass may bias the desired
air trajectory, but only through an achievable air-to-mass transfer and within
the configured assist band.

### 8.3 Last two hours

Air recovery has strict priority:

- air weight rises to its terminal value;
- mass weight falls to its terminal value;
- the assist band contracts to the terminal air tolerance;
- any mass request that moves air farther from its initial value is vetoed;
- when one step remains, mass weight is exactly zero.

This is a control priority, not a fixed electrical-power cap. Required power
still includes weather leakage, internal gains, mass exchange, COP, device
limits, and instantaneous comfort feasibility.

## 9. Reference trajectory

### 9.1 Smooth air recovery

For a simulated state with `h` steps remaining:

```text
T_air_next_air = T_air + (T_air_initial - T_air) / max(h, 1)
```

This creates a linear-in-remaining-time return rather than a one-step jump.

### 9.2 Recovery phase factor

```text
r = clip((h - final_air_priority_steps)
         / (recovery_horizon_steps - final_air_priority_steps), 0, 1)
```

With six-hour recovery and two-hour final air priority:

```text
r = clip((h - 2) / 4, 0, 1)
```

Weights interpolate as:

```text
w_air = final_air_weight
        + r * (early_air_weight - final_air_weight)

w_mass = final_mass_weight
         + r * (early_mass_weight - final_mass_weight)
```

When `h == 1`, set `w_mass = 0` exactly regardless of configuration.

Default weights:

```text
early_air_weight = 1.0
early_mass_weight = 1.0
final_air_weight = 10.0
final_mass_weight = 0.1
```

### 9.3 Air assist band

The maximum additional air displacement allowed for mass recovery is:

```text
assist_margin_c = terminal_air_tolerance_c
                  + r * mass_assist_extra_c
```

Default `mass_assist_extra_c` is `1.0`, producing approximately:

```text
six hours remaining: T_air_initial +/- 2 C
two hours remaining: T_air_initial +/- 1 C
```

The assist interval is intersected with the next state's comfort interval.

### 9.4 Mass controllability

If fewer than two steps remain, the current action has no useful terminal mass
control authority. Set the mass-required air target equal to the air-only target
and set the mass weight to zero.

Otherwise, distribute the remaining desired mass correction across the
remaining controllable transfer intervals:

```text
mass_control_steps = max(h - 1, 1)

mass_rate_requested =
  (T_mass_initial - T_mass)
  / (mass_control_steps * delta_t)
```

Calculate the largest directional mass rate available while air remains inside
the assist interval. For mass warming, use the upper assist bound; for mass
cooling, use the lower assist bound:

```text
T_air_assist =
  upper_assist, when T_mass_initial - T_mass >= 0
  lower_assist, otherwise

mass_rate_limit =
  ((T_air_assist - T_mass) / R_air_mass + Q_solar_mass)
  / C_mass
```

Use the following sign-safe clipping rule so cooling and heating cannot be
implemented differently by accident:

```text
mass_direction = sign(T_mass_initial - T_mass)

directed_rate_available = max(
  mass_direction * mass_rate_limit,
  0,
)

mass_rate_clipped = mass_direction * min(
  abs(mass_rate_requested),
  directed_rate_available,
)
```

When the mass gap is exactly zero, define `mass_direction = 0` and
`mass_rate_clipped = 0`.

The air temperature required to support that clipped mass rate is:

```text
T_air_required_mass = T_mass
  + R_air_mass * (
      C_mass * mass_rate_clipped - Q_solar_mass
    )
```

Finally clip `T_air_required_mass` into the assist interval.

This replaces V2's invalid assumption that all mass stored energy can be
delivered or removed directly by HVAC during the remaining time.

### 9.5 Combined next-air target

Combine the two candidate air targets:

```text
T_air_next_ref =
  (w_air * T_air_next_air + w_mass * T_air_required_mass)
  / max(w_air + w_mass, epsilon)
```

Then apply, in order:

1. final-air-priority veto;
2. assist-interval clipping;
3. next-state comfort clipping.

The final-air-priority veto applies when `h <= final_air_priority_steps`. If the
mass-weighted target increases absolute air error relative to the air-only
target, replace it with the air-only target. The exact test is:

```text
abs(T_air_initial - T_air_next_ref)
  > abs(T_air_initial - T_air_next_air) + epsilon
```

### 9.6 Reference HVAC power

Invert the air equation to reach the selected next-air target:

```text
Q_hvac_ref =
    C_air * (T_air_next_ref - T_air) / delta_t
  - (T_out - T_air) / R_out_air
  - (T_mass - T_air) / R_air_mass
  - Q_internal
  - Q_solar_air
```

Convert thermal power to signed electrical power:

```text
P_ref = Q_hvac_ref / cop_heat, when Q_hvac_ref >= 0
P_ref = Q_hvac_ref / cop_cool, when Q_hvac_ref < 0
```

Normalize with the existing signed action convention, then clip to the current
instantaneous feasible interval.

The planner rolls the model forward for up to `H` steps using each simulated
reference action. The first reference action is returned to the actor; predicted
terminal states are retained for diagnostics.

## 10. Reference-center engagement

The predictive reference must not cause a discontinuity when the recovery
window begins.

Define:

```text
free_center = (lower + upper) / 2

engagement = 1 - smoothstep(
  remaining_hours,
  reference_transition_full_hours,
  recovery_horizon_hours,
)

center = (1 - engagement) * free_center
         + engagement * predictive_reference
```

Defaults:

```text
recovery_horizon_hours = 6.0
reference_transition_full_hours = 4.0
```

This engagement changes only the semantic center of the latent map. It does not
blend or override the final actor action.

Outside the recovery horizon, `engagement` is exactly zero and the parent action
mapping must be reproduced exactly.

## 11. Reference-centered residual action mapping

Let the Transformer's latent DR output be `z` in `[-1, 1]` and let `center` be
inside `[lower, upper]`.

```text
action = center + z * (center - lower), when z < 0
action = center + z * (upper - center), when z >= 0
```

Required exact properties:

```text
z = -1 -> lower
z =  0 -> center
z =  1 -> upper
```

The final action must be clipped once more to `[lower, upper]` to protect against
floating-point drift.

The physical reference and predicted trajectories are detached from the actor
gradient graph. The derivative of final action with respect to latent action
must remain nonzero everywhere except:

- a degenerate feasible interval;
- an endpoint forced by clipping;
- an exact zero-width side around the center.

Exploration noise and target-policy smoothing remain applied to latent `z`
before this mapping.

## 12. Configuration contract

The HVAC-only configuration shall contain:

```yaml
algorithm:
  name: matd3_hierarchical_transformer_predictive_dr

  predictive_hvac:
    enabled: true
    recovery_horizon_hours: 6.0
    reference_transition_full_hours: 4.0
    final_air_priority_hours: 2.0

    early_air_weight: 1.0
    early_mass_weight: 1.0
    final_air_weight: 10.0
    final_mass_weight: 0.1

    mass_assist_extra_c: 1.0
    solar_forecast_mode: zero
    invalid_forecast_fallback: air_only

    residual_reference_mapping: true
    detach_physical_reference: true
```

Construction must reject:

- nonpositive recovery horizon;
- recovery horizon longer than the usable forecast horizon;
- `reference_transition_full_hours` outside `[0, recovery_horizon_hours)`;
- `final_air_priority_hours` outside `(0, recovery_horizon_hours)`;
- negative weights;
- `final_air_weight <= final_mass_weight`;
- negative `mass_assist_extra_c`;
- unsupported forecast or fallback modes;
- disabling residual mapping or physical-reference detachment in the first
  experiment contract.

The first HVAC-only configuration shall preserve the parent V2 HVAC experiment's
network, optimizer, replay, noise, training-length, and MG3 catch-up-floor
settings. MG3's economic reference remains disabled.

## 13. Failure handling

### 13.1 Degenerate feasible interval

If `upper - lower <= interval_epsilon`, return the unique lower-bound action and
keep a zero-gradient connection to latent action using the same stable pattern
as V1/V2.

### 13.2 Invalid forecast slots

Use only consecutive valid forecast slots. When a future slot is invalid, fill
temperature and load with the last valid value for the remaining rollout and
record a forecast-fallback diagnostic.

### 13.3 Missing required schema

Fail construction with a descriptive error naming every missing field. Do not
silently disable predictive control when the configuration says it is enabled.

### 13.4 Nonfinite calculation

If any predictive state, target, or reference becomes nonfinite, use an
air-only reference for the affected batch row:

```text
T_air_next_ref = T_air + (T_air_initial - T_air) / max(H, 1)
```

Invert the air equation, clip to the feasible interval, and increment the
nonfinite-fallback diagnostic. A nonfinite final action is a hard runtime error.

### 13.5 Physically unreachable target

The planner must not increase requested power merely because mass remains
outside tolerance when available air-to-mass transfer is insufficient. It shall
return the best bounded reference, expose predicted residuals, and allow the
formal gate to fail. No code path may claim a terminal guarantee.

## 14. Diagnostics

Extend actor diagnostics for MG2 with at least:

```text
hvac_predictive_reference_mean
hvac_reference_engagement_mean
hvac_air_target_next_mean_c
hvac_mass_required_air_mean_c
hvac_air_weight_mean
hvac_mass_weight_mean
hvac_mass_control_steps_mean
hvac_mass_rate_requested_mean_cph
hvac_mass_rate_clipped_mean_cph
hvac_predicted_terminal_air_gap_mean_c
hvac_predicted_terminal_mass_gap_mean_c
hvac_final_air_priority_veto_rate
hvac_forecast_fallback_rate
hvac_nonfinite_fallback_rate
```

Diagnostics must remain observational. They must not change actions, gradients,
replay contents, or environment behavior.

## 15. Constrained checkpoint selection

### 15.0 Validation metric production

`evaluate_policy_on_days` must produce the constraint metrics from the
validation rollout already in memory. Definitions must match formal reporting:

- strict feasible-day rate and unmet energy use `evaluation/service_reliability.py`;
- HVAC joint terminal success uses final-hour absolute air and mass gaps with
  the configured terminal tolerances;
- HVAC comfort rate uses the same hourly comfort flags as formal evaluation;
- mill success uses the configured terminal silo tolerance;
- EV departure success uses the configured departure-energy requirement;
- projection rates use the saved per-device projection flags.

Metric extraction must be algorithm-neutral, deterministic, and side-effect
free. Validation must not invoke settlement, plotting, formal-test artifact
writers, or paper-table generation. Existing algorithms may receive additional
summary fields, but their default checkpoint ranking must remain unchanged.

### 15.1 Generic opt-in behavior

Add a generic checkpoint selector outside the algorithm package, preferably a
focused helper module used by `experiments/training_pipeline.py`.

The default for existing configurations remains:

```text
mode = profit_only
```

The new configuration opts into:

```yaml
validation:
  selection:
    mode: constrained_profit
    objective: team_profit_mean
    constraints:
      strict_feasible_day_rate: {min: 1.0}
      unmet_energy_mean_kwh: {max: 0.0, atol: 1.0e-9}
      hvac_terminal_joint_success_rate: {min: 0.4545454545}
      hvac_comfort_rate: {min: 0.995}
      mill_terminal_success_rate: {min: 0.995}
      ev_departure_success_rate: {min: 1.0}
      mg2_projection_rate: {max: 0.0, atol: 1.0e-9}
      mg3_projection_rate: {max: 0.0, atol: 1.0e-9}
```

Metric names may be represented as dotted paths if the validation summary is
nested, but the names and their source must be stable and documented.

### 15.2 Ranking rule

Each validation checkpoint receives:

```text
eligible: boolean
normalized_constraint_violation: nonnegative scalar
objective: team_profit_mean
```

Candidate ordering is lexicographic:

1. eligible candidates outrank ineligible candidates;
2. among eligible candidates, higher objective wins;
3. among ineligible candidates, lower normalized violation wins;
4. exact ties use higher objective, then later episode number.

Normalized violation for a minimum constraint is:

```text
max((minimum - value) / max(abs(minimum), epsilon), 0)
```

For a maximum constraint:

```text
max((value - maximum - atol) / max(abs(maximum), 1, epsilon), 0)
```

Total violation is the sum of individual normalized violations. Missing or
nonfinite required metrics make the candidate ineligible and assign infinite
violation.

### 15.3 No eligible validation checkpoint

If no validation checkpoint is eligible:

- save the least-violating candidate as `best_model.pt` for diagnosis;
- write `constraint_eligible: false` in checkpoint-selection metadata;
- list every failed validation constraint;
- continue formal testing with the diagnostic best checkpoint;
- force the final experiment gate to fail;
- reject release promotion.

Formal-test data must never be used to choose the checkpoint.

### 15.4 Validation history

Each validation record must include:

```text
selection_mode
constraint_eligible
constraint_violation_total
constraint_values
constraint_failures
is_best
```

Existing fields and legacy behavior remain unchanged for profit-only runs.

## 16. Checkpoint isolation

Every checkpoint must contain:

```text
algorithm_slug
predictive_hvac_config
predictive_hvac_contract_version
observation_schema_by_agent
```

Loading must reject:

- a different slug;
- a missing predictive contract;
- an unsupported contract version;
- a different normalized predictive configuration;
- an incompatible observation schema.

Error messages must name the mismatched field and both values.

## 17. Tests

### 17.1 Pure physical tests

Create focused tests for `predictive_hvac.py`:

1. one-step zero-solar rollout matches `CommercialHVACDR` exactly;
2. signed heating and cooling COP conversion matches the component;
3. action at the final step cannot change terminal mass in the explicit-Euler
   convention;
4. mass correction direction is correct for warming and cooling;
5. mass control authority decreases as controllable steps disappear;
6. mass-request clipping never reverses direction;
7. the assist interval contracts smoothly from six to two hours;
8. air weight increases and mass weight decreases monotonically;
9. mass weight is exactly zero with one step remaining;
10. a huge mass gap cannot create a heat-capacity-sized terminal pulse;
11. final-air-priority veto rejects a mass target that worsens air error;
12. invalid forecasts use the documented fallback;
13. all outputs remain finite for extreme but valid states.

### 17.2 Property tests

Use at least 10,000 randomized valid batches covering:

- heating and cooling;
- positive and negative air/mass gaps;
- one through six hours remaining;
- narrow, wide, and degenerate feasible intervals;
- valid and partially invalid forecasts;
- all latent values near `-1`, `0`, and `1`.

Required properties:

- final action lies in `[lower, upper]`;
- no NaN or infinity is returned;
- `z=-1`, `z=0`, and `z=1` satisfy exact mapping semantics;
- the mapping is monotonic in latent action;
- nondegenerate mappings have nonzero latent gradient;
- predicted states remain finite.

### 17.3 Parent preservation tests

For identical weights and observations:

- MG1 output must match the parent HVAC-only variant exactly;
- MG3 output must match the parent HVAC-only variant exactly;
- MG2 output outside the recovery horizon must match exactly;
- critic, replay, update cadence, target noise, and exploration noise behavior
  must remain unchanged.

### 17.4 Checkpoint tests

- round-trip save/load succeeds for an identical contract;
- V1 and V2 checkpoints are rejected;
- changed predictive configuration is rejected;
- changed schema is rejected;
- error messages identify the mismatch.

### 17.5 Checkpoint-selection tests

- an eligible lower-profit checkpoint outranks an ineligible higher-profit one;
- among eligible checkpoints, higher profit wins;
- among ineligible checkpoints, lower violation wins;
- missing and nonfinite metrics are rejected;
- exact ties use the documented deterministic rule;
- profit-only selection remains bit-for-bit compatible with existing behavior;
- formal-test metrics cannot enter the selector.

### 17.6 Closed-loop tests

Add deterministic 24-hour HVAC rollouts for:

- a cold day;
- a hot day;
- a mild day;
- air and mass gaps in the same direction;
- air and mass gaps in opposite directions;
- a terminal mass target that is physically unreachable;
- a last-two-hour large mass gap.

Assertions shall cover:

- instantaneous comfort;
- terminal air tolerance;
- absence of terminal extreme-power pulses;
- finite actions and states;
- correct diagnostic flags when mass remains unreachable.

The MG3 regression set must include:

```text
2018-02-26
2018-04-27
2018-04-30
```

The first implementation does not change MG3 behavior, but no candidate may be
accepted if these or other validation days produce unmet energy.

## 18. Experiment protocol

### 18.1 Verification before full training

Run:

1. focused predictive-HVAC and selector tests;
2. full `pytest -q`;
3. a one-to-three-episode shared-pipeline smoke test;
4. a deterministic closed-loop HVAC test set.

The smoke test must confirm:

- the selected plugin and HVAC-only configuration;
- independent checkpoint metadata;
- `best_model.pt` formal-test selection;
- deterministic formal test;
- coalition MILP evaluation;
- Equal and Shapley settlement;
- summaries and saved-result-only figures;
- output under the new algorithm's working-run directory;
- zero changes in canonical MATD3, parent Transformer, V1, and V2.

### 18.2 First full experiment

Run exactly one complete HVAC-only experiment with seed 42 and the same training
length, validation cadence, formal-test days, environment hash, and physical
configuration used by the two V2 runs.

Do not change multiple predictive-HVAC parameters after seeing formal-test
results. If the run fails, diagnose the first failed hypothesis with saved
diagnostics and change one conceptual element at a time.

### 18.3 Multi-seed progression

Do not start multi-seed training until seed 42 passes every hard gate. After a
pass, use pre-registered seeds `42`, `43`, and `44` without per-seed tuning.

EV-only and Combined work require separate reviewed designs after the HVAC stage.

## 19. Acceptance criteria

### 19.1 Hard gates for seed 42

All must pass:

| Area | Metric | Requirement |
|---|---|---:|
| HVAC | Joint terminal thermal success | >= 45.45% |
| HVAC | Terminal task penalty | <= 35.25 |
| HVAC | Comfort rate | >= 99.5% |
| Reliability | Strict feasible-day rate | 100% |
| Reliability | Mean unmet energy | 0 kWh/day |
| MG1 | Mill terminal success | >= 99.5% |
| MG3 | EV departure success | 100% |
| Action | MG2 projection rate | 0 |
| Action | MG3 projection rate | 0 |

### 19.2 Economic guardrails

Use the approved parent Transformer seed-42 formal result as the frozen
reference:

| Metric | Requirement |
|---|---:|
| Mean total operating cost | <= parent * 1.02 |
| Peak grid import | <= parent * 1.05 |
| Renewable utilization | no decrease greater than 0.5 percentage points |
| P2P acceptance rate | no decrease greater than 5 percentage points |

Hard-gate success does not excuse a material economic regression.

### 19.3 Progression rule

Any hard-gate failure produces an overall `FAIL`, regardless of average reward
or total cost. A failed run remains a working result and must not be promoted to
`outputs/releases/`.

## 20. Output and reproducibility requirements

Normal runs must use:

```text
outputs/working_runs/matd3_hierarchical_transformer_predictive_dr/runs/<run_id>/
```

Required artifacts include the repository-standard checkpoints, histories,
configuration snapshot, metadata, daily metrics, hourly trajectories,
settlements, summaries, and figures.

The full experiment must start from a clean committed code state. Run metadata
must record:

- Git commit;
- clean/dirty state;
- environment and physical hashes;
- algorithm config hash;
- predictive-HVAC contract version;
- checkpoint-selection mode and eligibility;
- seed and exact test dates.

The two dirty V2 runs remain valid diagnostic evidence but are not acceptable as
the reproducibility standard for V3.

## 21. Gate report

The V3 gate checker shall report:

- each hard gate's value, threshold, and pass/fail status;
- each economic guardrail;
- checkpoint validation eligibility;
- parent run identity and hash used for comparison;
- overall status;
- eligibility for multi-seed training;
- eligibility for release promotion.

The checker must reject:

- a run from another slug;
- an incomplete run;
- a run whose formal test did not use `best_model.pt`;
- missing or nonfinite required metrics;
- a run selected from formal-test data;
- a dirty run when release eligibility is requested.

## 22. Expected interpretation of outcomes

### 22.1 Expected successful signature

- late gate-like forceful intervention disappears;
- predicted reference power remains physically bounded;
- air success rises without mass success collapsing;
- joint terminal success reaches or exceeds the parent target;
- MG2 terminal penalty decreases rather than aggregate improvement coming from
  unrelated agents;
- strict feasibility remains 100%;
- economics remain within guardrails.

### 22.2 Evidence that the hypothesis is wrong

The design hypothesis is not supported if:

- the predictive reference accurately forecasts terminal state but joint success
  remains low because the Transformer systematically chooses away from it;
- terminal air succeeds but mass remains unreachable even when recovery begins
  six hours early;
- future-solar approximation dominates the prediction error;
- the physical reference itself produces large comfort-bound saturation;
- seed-42 success cannot reproduce across seeds without parameter changes.

Such evidence should trigger a new architectural discussion. It must not be
answered by repeatedly lowering thresholds or increasing fixed weights.

## 23. Definition of done for implementation

Implementation is complete only when:

1. the isolated package and HVAC-only config exist;
2. every specified unit, property, preservation, checkpoint, selector, and
   closed-loop test passes;
3. full `pytest -q` passes;
4. the short shared-pipeline smoke test completes through plotting and
   settlement;
5. checkpoint isolation and constrained selection are verified;
6. existing MATD3, parent Transformer, V1, and V2 sources remain unchanged;
7. a clean committed seed-42 HVAC-only run is completed;
8. the gate report is generated from saved artifacts;
9. every hard gate and economic guardrail passes before multi-seed work begins.

Passing tests without a passing formal gate is not experimental success. A
passing formal result from a dirty or unidentifiable code state is not a
reproducible result.
