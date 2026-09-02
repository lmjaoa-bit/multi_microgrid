# Algorithm development structure

Repository-wide agent requirements are defined in `AGENTS.md`. This document expands those rules for
human maintainers; if the two differ, follow `AGENTS.md`.

## Separation rules

The canonical baseline lives only in `algorithms/matd3/`. Do not edit it when testing a new idea.
Every experimental algorithm owns three isolated areas:

```text
algorithms/<algorithm>/          implementation and plugin.py
configs/algorithms/<algorithm>.yaml
outputs/<algorithm>/runs/<run_id>/
```

The shared workflow owns everything that should remain identical across algorithms:

```text
scripts/run_experiment.py        command-line entry
experiments/training_pipeline.py shared training and validation implementation
scripts/train_matd3.py           thin legacy-compatible entry
evaluation/evaluate_algorithm.py test and saved-result generation
evaluation/settlement.py         Equal and Shapley settlement
evaluation/plotting.py           saved-result-only figures
experiments/layout.py            output directory layout
```

## Create a variant

```powershell
python scripts/create_algorithm_variant.py --name matd3_attention
python scripts/run_experiment.py --algorithm matd3_attention --episodes 10 --test-episodes 3
```

The scaffold initially inherits MATD3 behavior. Put overrides and new network modules only inside
`algorithms/matd3_attention/`; tune only `configs/algorithms/matd3_attention.yaml`.

Each plugin must expose an `AlgorithmSpec`, and its algorithm instance must provide:

- `from_env(...)`
- `select_actions(...)`
- `update(batch_size)`
- `save(path)` and `load(path)`
- `replay.add(...)` and `len(replay)`

## Remove a failed experiment

Stop its process, then remove only these isolated paths:

```text
algorithms/<failed_algorithm>/
configs/algorithms/<failed_algorithm>.yaml
outputs/<failed_algorithm>/       optional; retain it when results are still useful
```

No central registry file needs editing. Other algorithms are loaded only when explicitly selected.

## Standard run contents

```text
outputs/<algorithm>/
├── latest_run.json
├── legacy/                       migrated historical loose artifacts only
└── runs/
    └── <ALGORITHM>_<timestamp>_seed<seed>/
        ├── checkpoints/
        │   ├── best_model.pt
        │   └── final_model.pt
        ├── figures/
        ├── test_hourly_trajectories/
        ├── config_snapshot.yaml
        ├── run_metadata.json
        ├── training_history.json
        ├── validation_history.json
        ├── training_summary.json
        ├── test_daily_metrics.csv
        ├── test_summary.json
        └── summary.json
```

Every registered algorithm follows the same training, best-checkpoint validation, formal test,
MILP coalition evaluation, Equal/Shapley settlement, summary generation, and plotting sequence.

## ESS spatiotemporal sharing study

The current comparison methods are independent plugins, even when they reuse the canonical MATD3
network/update implementation:

| Slug | Action dimension | Environment control | Observation scope |
|---|---:|---|---|
| `matd3` | 1 | `residual_auto` | local baseline |
| `matd3_independent` | 1 | P2P disabled | local baseline |
| `matd3_ess_full_offer` | 1 | `fixed_full_offer` | local baseline |
| `matd3_ess_local_obs` | 2 | `explicit_offer` | local plus market feedback |
| `matd3_ess_sharing` | 2 | `explicit_offer` | cross-microgrid forecasts and feedback |

`sharing_offer_ratio` is a feasible offer cap, not an executed-trade ratio. Physical clearing may
accept less because peer demand or pair capacity is insufficient. This distinction is persisted in
hourly and daily result files through offer, accepted, and unmatched ESS energy fields.
