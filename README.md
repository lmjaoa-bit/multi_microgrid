# Standalone Multi-Microgrid P2P + Conditional MATD3

This repository is an independent multi-microgrid research implementation. It is not the old single-microgrid project wrapped three times.

## Current scope

Implemented:

- three structurally heterogeneous grid-connected microgrids;
- commercial / industrial / residential load profiles rather than simple load scaling;
- shared regional GHI, wind-speed and grid-price forecasts;
- microgrid-specific PV/wind power obtained from shared weather and installed capacities;
- PV, wind, battery, gas turbine and load in every MG;
- batteries charge only from local PV/wind surplus;
- unified sealed-bid P2P double auction with midpoint settlement;
- buy bids high-to-low, sell asks low-to-high, proportional allocation at equal prices;
- P2P before main-grid settlement;
- 15 kW main-grid export limit per MG;
- decentralized actor observations and a dedicated CTDE critic state;
- four-hour forecasts available to both actors and centralized critics, without future-realization leakage;
- linearly scaled reward `reward = -total_cost / 100` with no clipping or P2P bonus;
- Conditional MATD3 baseline with three independent actors and per-agent centralized twin critics;
- `Q_ON - Q_OFF` soft-target learning for generator commitment;
- ON-counterfactual generator-setpoint training;
- OFF/ON expected-Q learning for battery and P2P bid;
- hard commitment execution with epsilon-flip exploration;
- joint replay buffer storing time-aligned hard actions;
- synchronous actor update by default, with a sequential mode interface retained.

Not included in this version:

- CrossGate;
- a formal proof of Nash equilibrium (the included unilateral best-response check is empirical);
- IEEE-33 or network power flow;
- generator ramp / minimum-up / minimum-down constraints;
- terminal SOC penalty;
- long research training runs or tuned final hyperparameters.

## Heterogeneous V1 scenario

- **MG1:** commercial daytime load, 300 kW PV, 40 kW wind, 120 kWh / 40 kW BESS, 60 kW MT.
- **MG2:** industrial flatter load, 70 kW PV, 300 kW wind, 200 kWh / 50 kW BESS, 70 kW MT.
- **MG3:** residential morning/evening load, 60 kW PV, 40 kW wind, 80 kWh / 40 kW BESS, 90 kW MT.

All three MGs see the same regional weather forecast. Their renewable-power forecasts differ because installed capacities differ. Local load forecasts remain participant-specific.

On the training split the constructed mean/peak loads are approximately:

- MG1: 60.0 / 120.8 kW;
- MG2: 78.0 / 114.7 kW;
- MG3: 67.0 / 131.3 kW.

The corresponding renewable-energy/load ratios are approximately 0.688, 0.357 and 0.153, providing seller-dominant, flexible-prosumer and buyer-dominant tendencies without hard-coding market roles.

## Action definition

The environment action representation is always:

```text
[battery, hard_generator_commitment, generator_setpoint, p2p_bid]
```

All values are normalized to `[-1, 1]`. Conditional MATD3 executes commitment only as:

```text
OFF = -1
ON  = +1
```

When OFF, the stored/executed generator setpoint is fixed to `-1`. When ON, the setpoint maps continuously to `[P_min, P_max]`.

## Conditional MATD3 logic

Each MG has its own actor and two centralized critics:

```text
Actor_i(o_i)
Q_i^1(s, a_1, a_2, a_3)
Q_i^2(s, a_1, a_2, a_3)
```

For the focal MG, the algorithm evaluates two executable counterfactual branches:

```text
OFF: [battery, -1, -1, bid]
ON : [battery, +1, generator_setpoint, bid]
```

Commitment preference uses the mean of the twin critics for each branch and forms a soft target from `DeltaQ = Q_ON - Q_OFF`. Bellman targets retain TD3's conservative `min(Q1, Q2)`. Target commitment is hard and receives no Gaussian smoothing.

## Observation / CTDE state

Each actor observation is currently 48 dimensions. It contains local current state, four-hour local load/PV/wind forecasts, shared four-hour grid-price/GHI/wind-speed forecasts, and previous public P2P information.

The centralized critic state is currently 84 dimensions. It contains:

- all three MG true current physical states;
- public current grid/carbon/time information once;
- previous market information;
- current-time available four-hour shared weather/grid-price forecasts;
- all three MG four-hour local load/PV/wind forecasts.

It does **not** contain future realized values.

Use `env.ctde_spec` and `env.global_state_schema`; do not hard-code dimensions.

## Reward

Economic and penalty accounting remains in physical monetary units in `info["costs"]`. Learning uses only a positive linear scaling:

```text
reward_i = - total_cost_i / 100
```

There is no reward clipping, P2P-volume bonus, renewable-consumption bonus, or terminal SOC term.

## Project structure

```text
algorithms/conditional_matd3/
    config.py
    config_validation.py
    networks.py
    replay_buffer.py
    conditional_matd3.py
    trainer.py
components/
core/
    forecast.py
    profiles.py
    ...
envs/
market/
configs/
    algorithms/
        conditional_matd3.yaml
    multi_microgrid.yaml
data/
scripts/
tests/
```

## Training

Environment/physical parameters and algorithm parameters are intentionally
separate:

- `configs/multi_microgrid.yaml`: shared microgrid, market, data and physical settings;
- `configs/algorithms/conditional_matd3.yaml`: episodes, seed, validation cadence,
  network sizes, learning rates, replay/warm-up, exploration noise, target smoothing
  and Nash-verification defaults.

Edit the algorithm YAML for persistent training changes. CLI flags remain available
as one-run overrides. The default configuration can therefore be trained with:

```bash
python scripts/train_conditional_matd3.py
```

The formal 300-episode Conditional MATD3 training entry point is:

```bash
python scripts/train_conditional_matd3.py --episodes 300 --warmup-steps 1000 --validation-interval 20 --validation-episodes 10
```

For a longer run, override both the episode count and warm-up explicitly or edit
the same fields in the algorithm YAML:

```bash
python scripts/train_conditional_matd3.py --episodes 5000 --warmup-steps 5000 --validation-interval 20 --validation-episodes 10
```

Alternative algorithm configurations can be selected without touching the
environment file:

```bash
python scripts/train_conditional_matd3.py --algo-config configs/algorithms/conditional_matd3.yaml
```

`--device auto` selects CUDA when available and otherwise uses CPU. By default, a Rich terminal dashboard refreshes once per episode and shows progress, ETA, warmup/update state, current and MA20 system/MG metrics, and available Conditional MATD3 diagnostics. Use `--rolling-window` to change the display-only averaging window. For redirected output or servers without ANSI support, disable the dashboard:

When `--output` is omitted, each run is saved under `outputs/conditional_matd3_YYYY-MM-DD_HH-MM-SS/`. The algorithm name identifies the trainer and the date-time suffix prevents runs started on the same day from overwriting one another.

```bash
python scripts/train_conditional_matd3.py --episodes 300 --warmup-steps 1000 --no-live
```

Each output directory contains:

- `configs/environment.yaml`, `configs/algorithm.yaml` and
  `configs/effective_runtime.yaml`: immutable inputs copied for reproducibility;

- `training_log.csv`: fixed per-episode system, per-MG, critic and actor diagnostic columns;
- `validation_log.csv`: deterministic metrics on the fixed validation-day set;
- `policy_stability_log.csv`: fixed-probe action change at each validation checkpoint;
- `policy_action_change_vs_episode.png`: per-MG policy-stability trace;
- `run_config.json`: CLI arguments, runtime metadata, checkpoint scope and the resolved environment configuration;
- `best_model.pt`: lightweight online-actor policy selected by the lowest mean validation system cost;
- `last_checkpoint.pt`: complete network/optimizer state, overwritten periodically and at training completion;
- `crash_checkpoint.pt`: best-effort state written if training fails after initialization.

Validation always uses a separate `split="val"` environment, the same fixed dates recorded in `run_config.json`, and deterministic actions without exploration or training-state mutation. Periodic `checkpoint_epXXXX.pt` history is disabled by default; enable debug snapshots explicitly with `--keep-periodic-checkpoints`.

Approximate continuation is available with `--resume`:

```bash
python scripts/train_conditional_matd3.py --episodes 500 --resume outputs/run_300ep/last_checkpoint.pt --output outputs/run_300ep
```

Resume restores actor/critic/target networks, optimizers, `q_scale_ema`, update count, episode and environment-step counters. Replay-buffer contents and RNG states are not restored, so it is not an exact training-trajectory continuation. Full replay/trainer/RNG recovery remains recommended before formal long training.

## Empirical epsilon-Nash verification

After training, verify a saved `best_model.pt` by freezing two agents and retraining only the focal agent as a unilateral best response:

```bash
python scripts/evaluate_nash.py --checkpoint outputs/run_300ep/best_model.pt --br-episodes 1000 --br-restarts 5 --validation-interval 50 --validation-days 10 --test-days 30
```

Best-response training uses only the train split. Validation cost selects the best checkpoint and restart; the test split is evaluated once after selection. The default mixed initialization starts one restart from the learned focal actor and the remaining restarts from random focal actors. Opponent actors, critics and targets stay bitwise fixed, opponent actions are deterministic, and only the focal agent receives exploration and target-policy smoothing.

The `nash/` output directory contains machine-readable summary CSV/JSON files, per-day baseline and best-response test metrics, best-response training diagnostics, focal-agent policy checkpoints and comparison plots. A unilateral gain below the configured epsilon threshold is evidence of an empirical epsilon-Nash candidate under the tested optimizer, seeds, restart count and compute budget; it is not a mathematical proof that no better response exists.

## Validation commands

```bash
pytest -q
python scripts/smoke_test.py
python scripts/algorithm_smoke_test.py
python scripts/inspect_environment.py
```

`algorithm_smoke_test.py` is only a gradient/interface smoke test. It is not a performance experiment.
