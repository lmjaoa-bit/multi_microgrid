# Algorithm development structure

Repository-wide agent requirements are defined in `AGENTS.md`. This document expands those rules for
human maintainers; if the two differ, follow `AGENTS.md`.

## Separation rules

The canonical baseline lives only in `algorithms/matd3/`. Do not edit it when testing a new idea.
Every experimental algorithm owns three isolated areas:

```text
algorithms/<algorithm>/          implementation and plugin.py
configs/algorithms/<algorithm>.yaml
outputs/working_runs/<algorithm>/runs/<run_id>/
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
experiments/result_lifecycle.py  working/release result lifecycle
scripts/manage_results.py        explicit release promotion CLI
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

## Working runs and retained releases

Normal training always produces a working result first:

```text
code/config change
→ python scripts/run_experiment.py ...
→ outputs/working_runs/<algorithm>/runs/<run_id>/
→ inspect/debug/compare
```

Working runs are temporary research material. They may be deleted after they are no longer useful.
A working run does **not** become a permanent paper result just because it completed successfully.

When a result is accepted for long-term retention, first make sure the corresponding code is committed
and the recorded Git state is clean. Then promote the run explicitly:

```powershell
python scripts/manage_results.py promote `
  --run outputs/working_runs/matd3_hierarchical_transformer/runs/<run_id> `
  --release flexible_load_v1
```

Promotion copies the complete run into:

```text
outputs/releases/<release_name>/<run_id>/
```

Each retained run contains `release_manifest.json` and `README.md`; the global index is
`outputs/releases/registry.csv`. Promotion never deletes the working source and never overwrites an
existing retained run.

A run recorded with dirty or unknown Git state is rejected by default. `--allow-dirty` is an explicit
escape hatch only; the generated release README/manifest records the warning so the reproducibility
risk cannot be hidden.

Typical lifecycle:

```text
modify code/config
→ short/medium runs in working_runs
→ reject or refine
→ commit accepted code
→ run/re-run the experiment from that committed state
→ inspect final result
→ promote to a named release
→ use releases/registry.csv for later comparison and paper writing
```

Use new release names (`flexible_load_v1`, `flexible_load_v2`, etc.) instead of replacing old release
artifacts. Treat retained release files as immutable source evidence; regenerated figures or new
analysis should live in separate report/analysis directories.

## Remove a failed experiment

Stop its process, then remove only these isolated paths:

```text
algorithms/<failed_algorithm>/
configs/algorithms/<failed_algorithm>.yaml
outputs/working_runs/<failed_algorithm>/       optional; retain it when results are still useful
```

No central algorithm registry file needs editing. Other algorithms are loaded only when explicitly
selected. Do not delete a retained `outputs/releases/...` result during ordinary cleanup.

## Standard working-run contents

```text
outputs/working_runs/<algorithm>/
├── latest_run.json
└── runs/
    └── <ALGORITHM>_<timestamp>_seed<seed>/
        ├── checkpoints/
        │   ├── best_model.pt
        │   └── final_model.pt
        ├── figures/
        ├── dr_reference_hourly_trajectories/
        ├── test_hourly_trajectories/
        ├── config_snapshot.yaml
        ├── run_metadata.json
        ├── training_history.json
        ├── validation_history.json
        ├── training_summary.json
        ├── dr_reference_daily_metrics.csv
        ├── dr_reference_summary.json
        ├── dr_shift_metrics.csv
        ├── test_daily_metrics.csv
        ├── test_summary.json
        └── summary.json
```

Every registered algorithm follows the same training, best-checkpoint validation, formal test,
deterministic rule-reference test, MILP coalition evaluation, Equal/Shapley settlement, summary
generation, and saved-result-only plotting sequence. The rule reference uses neutral ESS dispatch,
constant material-balanced MG1 production, minimum-intervention MG2 comfort control, and immediate
MG3 EV charging to its departure target.

## Current methods

Only MATD3 (`matd3_dispatch`, command alias `matd3`) and 普通 T
(`matd3_hierarchical_transformer`) are active. Both use `peer_energy_summary` for new runs.
Both use `single_dispatch`; ordinary MATD3 is the global default.
The canonical `algorithms/matd3/` source and its reference YAML remain unchanged and are not a third selectable method.
Historical outputs retain their original labels. See [EXPERIMENT_PROFILES.md](EXPERIMENT_PROFILES.md).
