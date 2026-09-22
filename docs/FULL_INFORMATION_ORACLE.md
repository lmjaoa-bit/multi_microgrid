# Complete-device full-information MILP: diagnostic protocol

This **experimental, isolated** entry point is based on the environment in
`codex/matd3-cross-microgrid-comparable-20260922`; it does not replace the
historical seven-coalition ESS-only Shapley oracle or alter MATD3 training.

## Scope / what is actually optimized

Each 24h optimization sees **realized** base load, renewable availability,
electricity prices, outside temperature, solar gains and equipment parameters.
It jointly optimizes ESS charge/discharge with instantaneous limits and soft
terminal SOC penalty; industrial mill duty with linear silo inventory and
the soft terminal inventory penalty; reversible HVAC heating/cooling with the
environment's two-state 2R2C equations, occupancy-dependent temperature
bands, and both soft terminal thermal penalties; and aggregate EV charging
between arrival and departure with charge efficiency, SOC bounds and the
soft end-of-day EV energy-shortfall penalty. Grid and P2P energy balances,
per-pair bidirectional capacity, grid limits, non-simultaneous charge/discharge,
grid import/export exclusivity and cost parameters come from the active
environment configuration.

The model keeps the existing ESS-only oracle unchanged unless
`include_dr=True` is passed. The new entry point always sets this flag.

## Important limitations: NOT yet a proven exact environment upper bound

The environment uses **local-first sequential** renewable/ESS offers, P2P
clearing, then grid balancing. The MILP currently uses a joint bidirectional
P2P flow optimization with renewable/ESS export-source limits; the set of
physically balanced trades can differ from the actual pool's allocations.
HVAC temperature bands are enforced strictly, whereas the environment can
flag an unavoidable one-step violation. MILP unmet-energy slack does not
replicate the environment's proportional delivered-fraction feedback into
DR device state. Therefore label the result **full-information relaxed
planning reference**, not verified environment-equivalent optimum, until
a forward execution validator proves that each optimized hourly decision
is implementable with the actual action codec, projector, local-first pool,
and device state updates.

If a day cannot be proven `kOptimal` or the HVAC constraints are
infeasible, the runner records that day as a failure and refuses to publish
a 55-day mean; it does not silently substitute the old ESS-only MILP.

The cost comparison is descriptive and benefits from oracle access to future
realizations; the gap is not automatically an achievable improvement for
an 8h forecast Actor.

## Running the isolated benchmark

Install the optional `highspy` dependency if it is not already installed,
then from the repository root:

```powershell
python scripts/run_full_information_oracle.py --day 2018-04-27
```

For all complete test days:

```powershell
python scripts/run_full_information_oracle.py --all-test-days
```

For a **paired** comparison, supply the actual `test_daily_metrics.csv`
files from the aligned MATD3 and ordinary Transformer runs; never use
older runs with different observation / physical environment contracts:

```powershell
python scripts/run_full_information_oracle.py --all-test-days `
  --compare-matd3 "C:\\path\\to\\MATD3_2026-09-22_17-35-13_seed42\\test_daily_metrics.csv" `
  --compare-transformer "C:\\path\\to\\MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42\\test_daily_metrics.csv"
```

Outputs are under `outputs/working_runs/oracle_full_information/runs/`:
`audit.json`, `daily_comparison.csv` and `hourly/<date>.csv`.
The comparison CSV reports signed **RL cost minus oracle cost** by day
(positive means the RL policy costs more). A negative gap or failed day is
a **model equivalence warning**, not proof of exceptional RL performance.
Existing MATD3/Transformer checkpoints and training outputs are untouched.

## Next required equivalence gate before publishing an upper-bound claim

Replay the suggested schedules via actual `MultiMicrogridEnv.step()` or
an action-level inverse projector; compare per-hour actual vs optimized
flexible power, ESS dispatch, P2P settlements, unmet load, SOC, inventory,
temperatures, terminal penalties and team profit, with explicit numerical
tolerances. If the oracle schedule cannot be faithfully replayed, keep its
relaxed-reference label and do not advertise its cost gap as a theoretical
upper bound. Once replay equivalence is achieved on all paired 55 test days,
only then decompose the gap into fixed-DR / fixed-ESS / fully joint
counterfactuals and consider structural Actor changes.
