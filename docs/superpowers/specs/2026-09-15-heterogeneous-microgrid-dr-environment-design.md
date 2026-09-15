# Heterogeneous Multi-Microgrid DR Environment Design

Date: 2026-09-15
Branch: `codex/controllable-devices`
Baseline: `reproduce/pnn-transformer-8.99` at `ffb5e835858cb5b3963dac30fd53e48df5d66156`

## 1. Goal

Build a physically coherent 24-hour cooperative multi-microgrid environment for validating plain MATD3 before adapting Transformer, PNN, ShapleyQ, or other advanced algorithms.

The environment models one regional low-carbon energy community containing three autonomous but interconnected heterogeneous microgrids:

- MG1: industrial microgrid with PV, ESS, and controllable cement raw-meal grinding;
- MG2: commercial/public-service building microgrid with wind generation, ESS, and controllable HVAC thermal flexibility;
- MG3: residential community microgrid with PV, ESS, and aggregate EV charging flexibility.

The research objective is heterogeneous spatiotemporal coordinated dispatch:

- ESS transfers electrical energy across time;
- DR transfers load across time;
- P2P transfers electrical energy across microgrids.

The grand coalition minimizes total physical operating cost. Individual microgrid accounting remains available for analysis, participation evaluation, and later Shapley-based contribution/allocation work, but the baseline MATD3 uses one shared team reward.

## 2. Scope and Non-Goals

This design changes the physical environment and action interface. It does not yet redesign the Transformer, PNN, or Shapley algorithms.

In scope:

- heterogeneous load decomposition;
- three independent DR device models;
- actor action expansion from one raw control to two raw controls;
- DR-aware local observations and centralized state;
- revised hourly execution order;
- ESS cyclic terminal constraint;
- HVAC thermal-state initialization and terminal anti-reset treatment;
- P2P capacity recalibration;
- expanded diagnostics;
- data regeneration needed for local temperature and geographically coherent weather.

Out of scope for this phase:

- AC or DC power-flow equations;
- voltage/reactive-power control;
- coalition formation or endogenous membership;
- Shapley reward shaping;
- Transformer architecture changes;
- PNN architecture changes;
- endogenous electricity-market bidding;
- V2G;
- multi-day continuous RL episodes.

## 3. Physical Background and Topology

The three microgrids belong to one regional distribution-level energy community in a Jiangsu coastal demonstration region. They are separate controllable nodes connected through the same regional distribution network and a local cooperative energy-sharing platform.

The current Nanjing-Yancheng-Suzhou interpretation must not be used for final local-P2P experiments because those locations are too geographically separated for the simplified no-network-flow model.

The regenerated spatial weather dataset shall use three nearby but distinct sites inside one regional distribution area. The data sites should be separated enough to retain correlated-but-nonidentical weather, while remaining credible as nodes of one local distribution-level energy community. Exact coordinates are data-generation configuration rather than an RL interface, but all three must belong to the same declared regional scenario.

Topology:

```text
                   External Grid
                   ↑     ↑     ↑
                   │     │     │
                  MG1   MG2   MG3
                   ↖     ↕     ↗
                Cooperative P2P Pool
```

All three microgrids retain independent grid import/export interfaces. Any pair may exchange energy through the cooperative pool subject to pair capacity.

No AC/DC network-flow model is introduced. Network physics are represented only by P2P pair-capacity constraints and accounting consistency.

## 4. Fixed Energy-System Capacities

The following existing generation and ESS capacities are retained:

| Item | MG1 | MG2 | MG3 |
|---|---:|---:|---:|
| Nominal average total load target | 72 kW | 80 kW | 68 kW |
| PV capacity | 520 kW | 0 | 100 kW |
| Wind capacity | 0 | 900 kW | 0 |
| ESS energy | 210 kWh | 290 kWh | 95 kWh |
| ESS charge/discharge limit | 65 kW | 80 kW | 34 kW |
| Grid import limit | 200 kW | 200 kW | 200 kW |
| Grid export limit | 200 kW | 200 kW | 200 kW |

The P2P pair-capacity baseline is changed from 50 kW to:

```text
pair_capacity_kw = 75.0
```

Rationale: representative-day optimization showed 50 kW frequently saturates and blocks material cooperation value, while 75 kW captures most of the benefit available at 100 kW without making the network effectively unconstrained.

P2P loss remains zero in the first version, and same-hour grid resale remains prohibited.

## 5. Action Interface

### 5.1 Raw actor output

Each actor shall output exactly two normalized raw controls:

```text
[ess_dispatch_intent, dr_intent]
```

Thus the per-agent raw action dimension changes from 1 to 2.

The first component preserves the current single-dispatch ESS semantics. The actor does not directly choose independent `ESS->local` and `ESS->P2P` powers.

The second component has agent-specific DR semantics:

- MG1: raw-meal mill duty intent;
- MG2: signed HVAC intent;
- MG3: EV charging intent.

### 5.2 Effective ESS execution

The existing local-first single-dispatch principle is retained.

The ESS projector derives effective quantities such as:

- battery charge;
- battery discharge;
- discharge serving local load;
- ESS P2P offer;
- accepted ESS P2P export;
- unmatched ESS offer.

Local actual electrical load has priority over optional ESS charging.

Local-first means local real load is protected before ESS export. It does not mean all local storage charging must precede P2P export. The policy must be free to choose between storing renewable surplus locally and making surplus available to peers.

### 5.3 Raw and effective action logging

Both proposed and projected actions must be recorded. At minimum:

- `raw_ess_action`;
- `raw_dr_action`;
- `effective_ess_dispatch`;
- `effective_dr_action`;
- `ess_projection_delta`;
- `dr_projection_delta`;
- `ess_projected`;
- `dr_projected`.

Existing metrics logic that interprets a second raw action as a sharing-offer ratio must be removed or renamed because the second raw action becomes DR intent.

## 6. Hourly Execution Order

For each hour `t`, all actors make decisions simultaneously from the state at the beginning of the hour. Execution order is then:

```text
1. read exogenous data and current device states
2. obtain all three raw actor actions
3. project and execute DR actions
4. compute actual flexible loads
5. compute actual total loads
6. project local-first ESS dispatch against actual loads
7. form P2P supply/demand
8. clear P2P subject to 75 kW/pair capacity
9. settle residual demand/surplus with the external grid
10. commit ESS, silo, thermal, and EV states
11. calculate reward and diagnostics
12. build the next observation/state
```

The key invariant is:

```text
DR projection -> actual load -> ESS local-first projection -> P2P -> grid
```

ESS must never be projected against an obsolete pre-DR load.

## 7. Load Semantics

The environment must explicitly distinguish:

```text
base_load_kw
flexible_load_kw
actual_load_kw
```

with:

```text
actual_load_kw = base_load_kw + flexible_load_kw
```

For compatibility, `load_kw` may remain an alias for `actual_load_kw`, but internal logic and outputs must use the explicit fields.

Future load forecasts refer to exogenous base/non-flexible load, not future policy-determined flexible load.

## 8. MG1 Industrial DR: Raw-Meal Grinding and Silo

### 8.1 Physical interpretation

MG1 represents a cement raw-meal grinding subsystem feeding a continuous downstream process through an intermediate raw-meal silo.

The controllable device is the raw mill. The actor controls a continuous one-hour equivalent duty cycle, not motor speed.

### 8.2 Load decomposition

MG1 nominal average total load remains approximately 72 kW.

Flexible raw-mill share is fixed at approximately 32.5%:

```text
average raw-mill electrical power = 23.4 kW
non-flexible average base load = 48.6 kW
```

Rated raw-mill power:

```text
P_mill_rated = 28.5 kW
```

Specific energy:

```text
24 kWh/t
```

Rated production:

```text
q_prod_rated ≈ 1.186 t/h
```

Continuous downstream consumption:

```text
q_consume = 0.975 t/h
```

### 8.3 Silo state

Silo parameters:

```text
capacity = 17.55 t
minimum = 3.51 t
maximum = 15.80 t
initial = 10.53 t
```

Dynamics for 1 h timestep:

```text
S[t+1] = S[t] + 1.186 * u_mill[t] - 0.975
0 <= u_mill[t] <= 1
```

The actor proposal is projected to the instantaneous feasible duty interval and to a terminal-reachability interval.

Terminal anti-reset condition:

```text
|S[24] - 10.53| <= 0.20 t
```

This is a hard feasibility requirement, not a reward penalty.

### 8.4 MG1 DR observation fields

At minimum:

- normalized silo level;
- silo deviation from daily target;
- feasible minimum mill duty;
- feasible maximum mill duty;
- time remaining to episode end.

The fixed downstream consumption rate does not need a 24-step forecast token sequence.

## 9. MG2 Commercial HVAC DR: 2R2C Thermal Model

### 9.1 Thermal model

MG2 uses a reversible heat pump with a 2R2C building model:

```text
T_out <-> T_air <-> T_mass
```

Dynamics:

```text
C_air * dT_air/dt = (T_out - T_air)/R_out_air
                    + (T_mass - T_air)/R_air_mass
                    + Q_internal + Q_solar_air + Q_HVAC

C_mass * dT_mass/dt = (T_air - T_mass)/R_air_mass
                      + Q_solar_mass
```

Fixed parameters:

```text
R_out_air = 0.22 degC/kW
R_air_mass = 0.08 degC/kW
C_air = 15 kWh/degC
C_mass = 250 kWh/degC
heating electric limit = 35 kW
cooling electric limit = 35 kW
heating COP = 3.3
cooling COP = 3.2
```

Signed HVAC convention:

```text
negative -> cooling
zero     -> off
positive -> heating
```

Electrical HVAC consumption remains nonnegative; only thermal effect is signed.

### 9.2 Comfort constraints

Weekday occupied period:

```text
07:00-19:00
21 <= T_air <= 25 degC
```

Unoccupied period, including all weekend hours:

```text
16 <= T_air <= 30 degC
```

The actor HVAC proposal is projected to the one-step feasible electrical-power range required by the thermal model and comfort band.

If the 35 kW physical limit cannot satisfy comfort, apply the physical maximum and record `comfort_infeasible`; do not crash and do not silently violate the power rating.

### 9.3 Internal and solar gains

Solar gain:

```text
Q_solar = 0.025 * GHI_kW_equivalent
```

Using the agreed implementation convention based on GHI in W/m2, with:

```text
20% -> air node
80% -> mass node
```

Internal gains:

```text
occupied:   Q_internal = 0.50 * P_nonHVAC
unoccupied: Q_internal = 0.20 * P_nonHVAC
```

### 9.4 Load calibration

The MG2 non-HVAC base profile is calibrated to approximately:

```text
non_hvac_target_mean_kw = 75.0
weekend_multiplier = 0.57
```

The calibrated thermostat reference should produce approximately 5 kW annual-average HVAC electrical consumption so that total mean load remains near 80 kW.

The baseline calibration thermostat uses tighter bands:

```text
weekday occupied: 22-24 degC
otherwise:         18-28 degC
```

### 9.5 Daily thermal-state initialization

The building must not initialize by directly setting indoor temperature equal to outdoor temperature.

For each natural-day episode, read the local MG2 outdoor temperature at 00:00:

```text
T_out_0
```

Then initialize:

```text
T_air_0 = clip(23 + 0.20 * (T_out_0 - 23), 18, 28)
T_mass_0 = clip(23 + 0.10 * (T_out_0 - 23), 18, 28)
```

This creates weather-dependent but thermally buffered initial indoor states.

### 9.6 Terminal thermal anti-reset condition

To prevent the policy from exploiting the daily reset by leaving the building heavily overcooled or underheated at 24:00, apply hard terminal bands relative to the episode's initialized thermal state:

```text
|T_air_24  - T_air_0|  <= 1.0 degC
|T_mass_24 - T_mass_0| <= 0.3 degC
```

The mass-state bound is intentionally tighter because `T_mass` represents the dominant stored thermal energy.

The implementation should enforce terminal reachability rather than add a routine terminal-temperature reward penalty.

### 9.7 MG2 DR observation fields

At minimum:

- `T_air`;
- `T_mass`;
- `T_out`;
- lower comfort margin;
- upper comfort margin;
- `occupied`;
- `is_weekend`;
- feasible signed HVAC minimum;
- feasible signed HVAC maximum.

Future outdoor temperature is part of the forecast vector.

## 10. MG3 Residential DR: Aggregate EV Charging

### 10.1 Physical model

MG3 represents aggregate residential EV charging only. V2G is disabled.

Fixed fleet parameters:

```text
equivalent EV count = 13
battery per EV = 60 kWh
aggregate battery capacity = 780 kWh
initial SOC at 00:00 = 50%
target and maximum SOC = 80%
charger power per EV = 7.2 kW
aggregate charging limit = 93.6 kW
charging efficiency = 0.93
available window = 00:00-07:00
departure deadline = 07:00
```

Required battery-side energy:

```text
234 kWh/day
```

Required grid-side energy:

```text
approximately 251.61 kWh/day
```

The 80% target is also the first-version maximum SOC so the EV fleet cannot act as an unlimited renewable sink.

### 10.2 Residential load decomposition

The old 68 kW nominal average total load is preserved in energy scale.

After subtracting expected EV charging energy, the residential non-EV base-load target is approximately:

```text
57.52 kW average
```

Total load is:

```text
P_load_MG3 = P_base_residential + P_EV
```

### 10.3 EV state and reachability

Dynamics:

```text
SOC_EV[t+1] = SOC_EV[t] + eta * P_EV[t] * dt / E_fleet
```

The deadline is a hard requirement:

```text
SOC_EV[07:00] = 80%
```

The feasible minimum charge power is computed from remaining energy, remaining time, efficiency, and maximum charger power so the target remains reachable.

The feasible maximum is limited by charger power and remaining energy to the 80% cap.

### 10.4 MG3 DR observation fields

At minimum:

- EV SOC;
- SOC gap to target;
- remaining required energy;
- hours to departure;
- EV availability;
- feasible minimum charging power;
- feasible maximum charging power;
- deadline pressure.

Deadline pressure is defined as:

```text
remaining_energy / (eta * P_EV_max * remaining_time)
```

## 11. ESS Daily Boundary Condition

All ESS units initialize at:

```text
SOC_0 = 50%
```

Physical bounds remain:

```text
10% <= SOC <= 90%
```

Every 24 h episode must satisfy a hard cyclic terminal condition:

```text
|SOC_24 - 50%| <= 3 percentage points
```

This replaces the previous unconstrained terminal mode and prevents the daily reset from creating free electrical energy.

The terminal condition should be enforced through reachable-envelope projection rather than a routine reward penalty.

## 12. P2P and Grid Settlement

After DR and local ESS projection, each microgrid exposes residual local deficit/surplus and any ESS sharing offer.

P2P clearing must obey:

- no pair flow above 75 kW;
- no same-hour grid import resold through P2P;
- P2P energy conservation at the system level;
- zero first-version P2P loss;
- actual load is served before optional ESS charging when external supply is constrained.

P2P may supply peer load and may also support peer ESS charging when physically feasible.

The external grid is the final balancing resource after P2P clearing.

## 13. Reward and Optimization Objective

Baseline MATD3 is a cooperative grand-coalition problem.

The team objective is expected discounted total physical operating cost minimization:

```text
min E[sum_t gamma^t * C_team_t]
```

with shared reward:

```text
r_1 = r_2 = r_3 = -C_team / reward_scale
```

Physical team cost contains at least:

- grid import cost;
- grid export revenue;
- ESS degradation cost;
- unmet-load cost;
- renewable-curtailment cost.

P2P payments are internal transfers and must not change team physical profit/cost. They may affect per-microgrid accounting ledgers used for analysis.

Hard constraints already enforced by the environment are not duplicated as routine DR reward penalties. In particular, do not add normal penalties for:

- silo bounds;
- HVAC comfort bounds;
- EV deadline;
- ESS SOC bounds;
- terminal cyclic constraints;
- P2P capacity.

If an exogenous/physical condition makes a hard target impossible even at the device limit, record explicit infeasibility diagnostics. Such abnormal infeasibility may be considered separately later but is not silently converted into ordinary shaping.

## 14. Local Observations

Every actor receives a common energy-state core including:

- hour sine/cosine;
- base load;
- PV;
- wind;
- buy/sell prices;
- ESS SOC;
- ESS feasible charge/discharge information;
- previous P2P/grid exchange summaries;
- current system P2P clearing summary where already part of the observation design.

The actor also receives its own detailed DR state as specified above.

Under `cross_microgrid`, peer information is limited to compact energy summaries, primarily:

- peer net base load;
- peer ESS SOC;
- existing peer future net-load summaries where retained.

An actor does not receive another microgrid's private device state such as peer building temperature, peer EV SOC, or peer silo inventory.

This preserves decentralized execution while allowing limited cooperative information sharing.

## 15. Centralized Critic State

The centralized critic may see the full joint physical state during training.

In addition to existing energy variables, the global state must include sufficient DR state to remain Markov:

MG1:

- silo level;
- terminal target/gap;
- feasible mill-duty bounds.

MG2:

- `T_air`;
- `T_mass`;
- `T_out`;
- occupancy/weekend state;
- comfort margins;
- feasible HVAC bounds.

MG3:

- EV SOC;
- remaining energy;
- time to departure;
- availability;
- deadline pressure;
- feasible EV charging bounds.

Critic input remains joint state plus joint action. The three actors continue to update separately against the shared team objective.

## 16. Forecast Semantics

The environment-level future horizon remains 24 h for the first MATD3 validation phase.

Forecasts represent only exogenous quantities:

- base/non-flexible load;
- PV-driving irradiance;
- wind speed / wind generation inputs;
- price;
- outdoor temperature.

Flexible load is endogenous and must not be forecast as though it were exogenous.

MG2 requires future local outdoor temperature support in the forecast provider.

MG1's fixed downstream material-consumption rule and MG3's deterministic departure deadline do not require repeated 24-step target tokens.

Later Transformer experiments may consume a shorter adaptive maximum horizon such as 8 h, but that is an algorithm-level encoding choice and does not change the environment's ability to expose a 24 h exogenous forecast window.

## 17. Weather and Data Changes

The spatial data builder must support local temperature in addition to irradiance and wind speed. NASA POWER input should include `T2M` for the regenerated regional data.

For consistency, the regenerated dataset should preferably contain local temperature for all three microgrids even though MG2 is the first version that requires temperature dynamics.

Required conceptual local columns include:

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

MG2 HVAC dynamics and forecasts use `mg2_temperature_c`, not the current shared legacy `temperature_c` field.

## 18. Episode and Split Semantics

One episode is one complete natural day:

```text
00:00-24:00
24 hourly steps
```

Training samples complete natural days from the training split.

Validation uses a fixed set of validation dates.

Testing uses a fixed set of unseen test dates shared across algorithms.

Workday/weekend state is derived from the actual timestamp, not sampled independently.

The first environment version remains a set of independent 24 h representative-day episodes. Cross-day physical consistency is enforced through device-specific terminal constraints rather than by changing RL to multi-day continuous episodes.

## 19. Diagnostics and Research Outputs

### 19.1 Electrical balance fields

Per microgrid and step, record at least:

- `base_load_kw`;
- `flexible_load_kw`;
- `actual_load_kw`;
- `pv_kw`;
- `wind_kw`;
- `renewable_kw`;
- `ess_charge_kw`;
- `ess_discharge_kw`;
- `ess_to_local_kw`;
- `ess_p2p_offer_kw`;
- `ess_p2p_accepted_kw`;
- `ess_p2p_unmatched_kw`;
- `p2p_buy_kw`;
- `p2p_sell_kw`;
- `grid_import_kw`;
- `grid_export_kw`;
- `unmet_load_kw`;
- `curtailment_kw`;
- `power_balance_residual_kw`.

System diagnostics must verify nodal power balance and aggregate P2P energy balance.

### 19.2 MG1 device diagnostics

Record at least:

- requested/effective mill duty;
- mill power;
- raw-meal production;
- downstream consumption;
- silo level and ratio;
- feasible silo/duty bounds;
- terminal target and terminal gap.

Episode summaries include shifted mill energy, terminal silo gap, and mill projection rate.

### 19.3 MG2 device diagnostics

Record at least:

- signed/effective HVAC action;
- HVAC electrical power;
- heating/cooling electrical energy;
- indoor air temperature;
- thermal mass temperature;
- outdoor temperature;
- comfort limits and margins;
- occupancy/weekend state;
- feasible HVAC bounds;
- comfort violation;
- comfort infeasibility.

Episode summaries include heating/cooling energy, comfort violation hours, HVAC projection rate, terminal thermal gap, and boundary-hit rates.

### 19.4 MG3 device diagnostics

Record at least:

- EV availability;
- requested/effective EV power;
- EV SOC;
- target SOC and SOC gap;
- remaining energy;
- hours to departure;
- feasible EV power bounds;
- deadline pressure;
- departure SOC;
- target-met flag.

Episode summaries include EV energy, departure SOC, target success, forced-charging rate, and EV projection rate.

### 19.5 P2P capacity diagnostics

Because 75 kW is a calibrated but still finite pair capacity, report:

- pair-capacity saturation rate;
- blocked P2P energy;
- P2P demand unserved specifically because of pair capacity.

These diagnostics decide whether later sensitivity testing at 50/75/100 kW is scientifically useful.

### 19.6 Energy-attribution diagnostics

The environment may report source-to-flexible-load metrics such as renewable-to-flexible-load or P2P-to-flexible-load only after a deterministic energy-source accounting rule is explicitly implemented. These quantities must not be inferred loosely from aggregate net flows.

## 20. Configuration and Component Architecture

The three DR devices should be separate components behind a thin common integration contract rather than implemented as a large conditional block inside the environment.

Suggested conceptual components:

```text
IndustrialMillDR
CommercialHVACDR
EVChargingDR
```

Each component should expose clear operations equivalent to:

```text
reset(...)
feasible_bounds(...)
project_action(...)
plan/step(...)
commit(...)
state/diagnostics(...)
```

The exact Python method names may follow repository conventions, but the separation of responsibilities is part of the design.

The environment coordinates these components and owns electrical settlement order; the device classes own their internal physical state and feasibility calculations.

Configuration validation must understand each DR type and reject missing or physically inconsistent parameters before training begins.

## 21. Expected Existing-Code Impact

The implementation plan is expected to touch, at minimum, the following areas:

- `configs/cooperative.yaml` and related experiment configs;
- `configs/action_designs.yaml`;
- `core/action_codec.py`;
- `core/single_dispatch.py` only as needed to accept the first component of a 2D raw action while preserving ESS semantics;
- `core/profiles.py` for base-load semantics;
- `core/forecast.py` for outdoor-temperature forecasts;
- `core/global_state.py`;
- `core/config_validation.py`;
- `core/microgrid_core.py` and/or environment orchestration for DR-before-ESS sequencing;
- `envs/multi_microgrid_env.py`;
- metrics collection and episode summaries;
- spatial-data generation scripts;
- tests covering physics, action projection, terminal reachability, and balances.

The core MATD3 network implementation should not require algorithmic restructuring because actor output dimension and replay/action shapes are already driven by `env.action_dim`. The implementation plan must still verify all assumptions with tests.

## 22. MILP-Based Parameter Screening Conclusions

A representative-day optimization screen was used as a design check, not as a replacement for final environment validation.

The screening supports:

- retaining 520/900/100 kW RES capacities;
- retaining 210/290/95 kWh ESS capacities;
- retaining 200 kW grid limits;
- increasing P2P pair capacity from 50 to 75 kW;
- enforcing ESS terminal cycling after changing initial SOC to 50%;
- explicitly treating HVAC terminal thermal state to avoid reset exploitation.

The screening also showed that 100 kW P2P produced only small additional benefit over 75 kW in representative conditions, while 50 kW was frequently binding.

Formal post-implementation MILP comparison must be rerun against the final environment equations and regenerated local-temperature data before publication-quality conclusions are claimed.

## 23. Acceptance Criteria

The environment design is considered successfully implemented only if all of the following hold:

1. Every actor raw action is exactly 2D: ESS intent plus its own DR intent.
2. DR affects actual load before ESS projection.
3. Local-first ESS behavior remains intact.
4. All three DR state equations satisfy their hard bounds under normal feasible scenarios.
5. MG1 terminal silo condition is met without free reset energy/material.
6. ESS terminal SOC condition is met without terminal reward shaping.
7. MG2 thermal initialization follows the 00:00 outdoor-temperature formula.
8. MG2 terminal air/mass conditions prevent free thermal reset exploitation.
9. MG3 reaches 80% SOC by 07:00 without V2G or charging above the target.
10. P2P pair flows never exceed 75 kW.
11. Same-hour grid resale remains blocked.
12. Nodal and system power-balance residuals are numerically negligible.
13. Shared team reward equals the physical grand-coalition objective and excludes internal P2P transfers.
14. Local actor observations contain only own detailed device state plus allowed peer summaries.
15. Centralized state contains enough joint DR state to be Markov.
16. Forecast flexible load is not treated as an exogenous series.
17. MG2 receives local outdoor temperature and future temperature forecasts.
18. Train/validation/test remain natural-day separated and reproducible.
19. Output diagnostics are sufficient to identify projection abuse, boundary saturation, P2P congestion, unmet load, curtailment, and terminal reset exploitation.
20. Plain MATD3 can train and evaluate on the new environment before Transformer/PNN/Shapley variants are migrated.

## 24. Validation Strategy Before Long Training

Implementation validation proceeds in this order:

1. unit tests for each DR device state equation and feasible-bound calculation;
2. unit tests for ESS and DR raw-to-effective action projection;
3. deterministic 24 h hand-crafted scenarios for each microgrid;
4. power-balance and P2P-conservation tests;
5. terminal reachability tests for ESS, silo, thermal state, and EV deadline;
6. random-action environment rollouts over multiple natural days;
7. short MATD3 smoke training;
8. fixed-date deterministic evaluation;
9. final MILP benchmark using the same finalized data/constraints;
10. only after these pass, begin full MATD3 training and later migrate Transformer/PNN/Shapley variants.

## 25. Research Interpretation

The baseline study is a cooperative grand-coalition dispatch problem, not coalition-formation optimization.

The central scientific structure is:

```text
temporal flexibility:
ESS + industrial inventory + building thermal inertia + EV deadline flexibility

spatial flexibility:
P2P energy sharing
```

Plain MATD3 establishes whether distributed actors under CTDE can minimize grand-coalition operating cost in this heterogeneous physical environment.

Per-microgrid costs/profits remain observable so later work can test whether total cooperation gains are unevenly distributed. That later analysis motivates Shapley contribution/allocation and, separately, Shapley-informed credit assignment without changing the physical environment definition established here.
