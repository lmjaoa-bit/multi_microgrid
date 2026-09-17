# Instantaneous Constraints, P0 Observations, and Sparse Diagnostics

**Date:** 2026-09-17  
**Base branch:** `codex/controllable-devices`

## Objective

Make instantaneous physical projection the only supported environment semantics for MATD3, PNN,
MATD3-Transformer, and PNN-Transformer. Remove all environment logic that selects future actions on
an agent's behalf, expose terminal pressure through P0 observations instead, and reduce training
overhead by collecting expensive diagnostics only at a configured interval. Transformer prediction
horizons remain four hours.

## Scope and invariants

- The canonical implementation under `algorithms/matd3/` remains unchanged.
- All four current experiment algorithms use the same environment, P0 observation contract, and
  terminal-penalty semantics.
- Current-step physical constraints remain hard constraints.
- Terminal targets remain observable and terminal violations remain penalized.
- Training, validation, formal testing, settlement, plotting, checkpoint selection, and output
  layout continue to use the shared experiment pipeline.
- `network.forecast_horizon` remains `4` for both Transformer variants.
- Existing user changes in the spatial-market data files are outside scope and must be preserved.

## Canonical environment

`MultiMicrogridEnv` becomes the sole formal environment. The strong/instantaneous mode switch,
the instantaneous ablation subclass, its separate configuration, and its ablation runner are
removed.

The environment continues to use its joint differentiable projector, but the feasible bounds fed
to that projector contain only current-step safety limits:

- ESS: SOC bounds, charge/discharge power limits, efficiency, and grid/dispatch limits.
- Industrial mill: the next silo level must remain inside the physical silo band.
- HVAC: the next thermal state must remain inside the active comfort band and electrical power
  limits.
- EV: availability, battery headroom, charging power, efficiency, and maximum SOC.

The following future-reachability mechanisms are removed:

- ESS terminal-SOC reachability projection.
- Industrial-mill backward terminal-silo reachability intervals.
- HVAC terminal-state optimization.
- EV forced minimum charging for the departure target.

The ordinary configuration uses terminal penalty semantics rather than hard terminal projection.
At episode end, the environment computes an equivalent-energy penalty for ESS, mill, HVAC, and EV
terminal deviations. The penalty changes reward/accounting only; it never changes the executed
action. Existing detailed cost and terminal-penalty records remain available to summaries.

## P0 observation contract

P0 fields are added to both decentralized Actor observations and the centralized Critic state. Raw
component state may retain explicit physical-unit suffixes; public observation schema names remain
short and stable. Time values are normalized by episode length and physical quantities are
normalized by the corresponding capacity or rated power before entering a network.

### ESS

The existing `soc` and `terminal_soc_gap` remain. The target is the configured initial cyclic SOC.

- `time_to_terminal`: remaining episode hours divided by episode length.
- `required_avg_power_to_target`: signed grid-side average power needed over the remaining hours,
  normalized by the applicable charge/discharge power rating. Negative denotes charging and
  positive denotes discharging, matching the ESS action sign convention. Charging and discharging
  efficiency are included.
- `terminal_pressure`: absolute required average power divided by the applicable power rating. It
  is not clipped, so a value above one exposes an infeasible terminal recovery requirement.

At the terminal state, required average power and pressure are zero to avoid division by zero;
`terminal_soc_gap` still exposes the achieved deviation.

### Industrial mill

- `silo_target_gap`: current silo inventory minus the initial daily target, normalized by silo
  capacity.
- `time_remaining`: remaining episode time divided by episode length.
- `required_avg_production`: average production rate needed to offset future downstream
  consumption and return to the target, normalized by rated production.
- `production_pressure`: the un-clipped required production rate divided by rated production. A
  value outside `[0, 1]` explicitly identifies an infeasible recovery requirement.

### HVAC

- `terminal_air_gap`: initial air temperature minus current air temperature, normalized by a fixed
  temperature scale.
- `terminal_mass_gap`: initial thermal-mass temperature minus current thermal-mass temperature,
  normalized by the same scale.
- `time_remaining`: remaining episode time divided by episode length.
- `thermal_recovery_pressure`: the magnitude of the average equivalent electrical power required
  to recover both temperature states, using their heat capacities and the direction-appropriate
  COP, divided by the matching heating or cooling power limit. The value is not clipped.

The signed gap fields supply direction; pressure supplies urgency.

### EV

The existing physical values and deadline-pressure calculation remain. Actor schema names are
standardized to:

- `remaining_energy`
- `hours_to_departure`
- `deadline_pressure`

The feasible minimum is always zero while the EV is available. Failure to reach the departure
target is reported and penalized, not corrected by forced charging.

## Sparse training diagnostics

A shared `algorithms/diagnostic_sampling.py` module owns the fast projected-update paths for the
four experiment algorithms. Plugins install the appropriate fast path without changing canonical
MATD3.

Configuration is read from:

```yaml
logging:
  diagnostic_interval: 20
```

Every training update still performs the required critic update, delayed actor update, and target
soft updates. Non-diagnostic updates do not perform Python truth conversion of CUDA tensors,
finite checks, Q/TD reductions for reporting, gradient-norm reporting, parameter snapshots, or
device-to-host metric transfers.

A diagnostic sample is collected only on an Actor update. The first eligible Actor update at or
after each interval boundary is selected, so diagnostics cannot land on a critic-only step even if
`diagnostic_interval` is not divisible by `policy_delay`. A diagnostic sample includes:

- Critic and Actor losses, including per-agent values.
- Q mean, standard deviation, minimum, and maximum.
- TD-error mean, standard deviation, and RMSE.
- Critic and Actor gradient norms plus existing branch/layer diagnostics.
- Existing action/projection and optimizer-step diagnostics where supported by the Actor.

Diagnostic tensors are batched and copied to the host once per diagnostic event where practical.
The shared training pipeline accepts empty update metrics on ordinary steps and aggregates only the
sampled records.

## Compatibility and checkpoints

Observation and centralized-state schemas change, so old checkpoints are intentionally
incompatible with the new environment contract. New checkpoints retain explicit schema and
projection signatures and must reject mismatched historical checkpoints. Raw action dimensions
remain two per agent and effective Critic action dimensions remain three per agent.

## Tests and verification

Implementation follows test-first development. Focused tests must demonstrate:

1. Only instantaneous bounds are used for ESS, mill, HVAC, and EV.
2. Terminal violations change penalties but do not retroactively change actions.
3. Every P0 schema field exists in both local observations and global state, has the documented
   sign/normalization, and produces finite values at the terminal boundary.
4. Observation dimensions and Transformer feature adaptation remain consistent for all agents.
5. Transformer network horizons remain four.
6. Ordinary updates omit diagnostic synchronization and diagnostic payloads.
7. Diagnostic events include critic and Actor statistics and always coincide with an Actor update,
   including a non-divisible interval/policy-delay case.
8. All four plugins use their own YAML, initialize successfully, train from replay, preserve action
   dimensions, and save/load checkpoints.

Final verification consists of:

- `python -m compileall algorithms envs`
- `pytest -q`
- Four shared-pipeline smoke runs of ten training episodes, with a short formal test and the normal
  result lifecycle under `outputs/working_runs/<slug>/runs/`.
- Confirmation that `best_model.pt` drives formal testing, Equal/Shapley settlement completes,
  and the expected PNG/PDF figures are produced.
- A diff check confirming that no file under canonical `algorithms/matd3/` changed.

