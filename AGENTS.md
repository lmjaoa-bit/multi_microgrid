# Repository instructions for algorithm development

These instructions apply to the entire repository. Read this file before changing an algorithm,
training workflow, evaluation, settlement, plotting, configuration, or output layout.

## 1. Preserve the canonical MATD3 baseline

`algorithms/matd3/` is the permanent reference implementation.

- Do not modify its actor, critic, replay buffer, MATD3 update logic, checkpoint format, or public
  behavior when implementing an experimental algorithm.
- A change to canonical MATD3 is allowed only when the user explicitly requests a baseline fix or
  baseline change.
- Never implement a new idea by editing MATD3 in place. Create a separate algorithm package.
- `algorithms/matd3/plugin.py` is registration metadata, not an experimental implementation file.

## 2. Required structure for every new algorithm

Use a lowercase snake_case slug, for example `matd3_attention`.

```text
algorithms/<algorithm_slug>/
├── __init__.py
├── plugin.py
├── algorithm.py
└── actor.py / critic.py / replay_buffer.py   # only when the variant needs local replacements

configs/algorithms/<algorithm_slug>.yaml
outputs/working_runs/<algorithm_slug>/runs/<run_id>/
```

Create the initial isolated scaffold with:

```powershell
python scripts/create_algorithm_variant.py --name <algorithm_slug>
```

The scaffold may inherit canonical MATD3 initially. Put every override and new network module in the
new algorithm package. Do not place experimental conditionals inside `algorithms/matd3/`.

Each `plugin.py` must expose exactly one `ALGORITHM_SPEC: AlgorithmSpec` with:

- `slug`
- `display_name`
- `algorithm_class`
- `default_config_path`
- a concise description

Plugins are loaded by requested name. Do not add a central import for every algorithm; a broken or
deleted experimental package must not prevent MATD3 or another algorithm from running.

## 3. Shared algorithm interface

Every algorithm used by the shared pipeline must provide:

```text
classmethod from_env(env, config, device, seed)
select_actions(observations, noise_std)
update(batch_size)
save(path)
load(path)
replay.add(...)
len(replay)
```

The authoritative protocol and runtime validation live in:

- `algorithms/base.py`
- `algorithms/registry.py`

Do not make the shared pipeline inspect private actor/critic internals. Algorithm-specific training
behavior belongs behind this interface.

## 4. One shared experiment workflow

All algorithms must use:

```powershell
python scripts/run_experiment.py --algorithm <algorithm_slug>
```

The current ordinary MATD3 dispatch controller runs with:

```powershell
python scripts/run_experiment.py --algorithm matd3
```

Do not copy the trainer, evaluator, settlement code, or plotter into an algorithm package. The single
shared implementation is:

- `experiments/training_pipeline.py` — training, validation, best-checkpoint selection, formal test
- `experiments/layout.py` — result directory layout
- `experiments/result_lifecycle.py` — working-run Git identity and retained-release promotion
- `scripts/manage_results.py` — explicit promotion of accepted runs into releases
- `evaluation/evaluate_algorithm.py` — algorithm-neutral evaluation entry
- `evaluation/coalition_evaluator.py` — seven daily coalition MILP values
- `evaluation/settlement.py` and `evaluation/shapley_allocator.py` — Equal/Shapley settlement
- `evaluation/plotting.py` — saved-result-only PNG/PDF figures

The required sequence is:

```text
training
→ validation
→ select best_model.pt
→ deterministic formal test
→ daily 24-hour coalition MILP evaluation
→ Equal and Shapley settlement
→ test summary generation
→ plotting from saved JSON/CSV files
```

Settlement and plotting are post-processing. They must not change reward, actions, replay data,
physical dispatch, actor/critic updates, or environment constraints.

## 5. Configuration isolation

- Environment and physical-system configuration: `configs/cooperative.yaml`.
- Per-algorithm hyperparameters: `configs/algorithms/<algorithm_slug>.yaml`.
- The YAML `algorithm.name` must equal the selected plugin slug.
- Tune an experimental algorithm only in its own YAML file.
- Current runs use `configs/algorithms/matd3_dispatch.yaml` or
  `configs/algorithms/matd3_hierarchical_transformer.yaml`.
- `algorithms/matd3/` and `configs/algorithms/matd3.yaml` are permanent source references only.
- CLI `--algorithm matd3` selects `matd3_dispatch`; do not restore the retired action modes.
- The current action design is `single_dispatch_dr`; the global default experiment is `matd3`.
- Both current plugins use `peer_energy_summary` observations.
- User-facing names are MATD3 and 普通 T, without action version numbers.

If an experiment requires an environment or reward change, treat it as a separate cross-algorithm
study. Do not silently change shared physics or rewards while presenting the result as only an
algorithm improvement.

## 6. Standard result lifecycle and layout

Every normal training run is a working result by default:

```text
outputs/working_runs/<algorithm_slug>/
├── latest_run.json
└── runs/
    └── <ALGORITHM>_<timestamp>_seed<seed>/
        ├── figures/                 # PNG/PDF images only
        └── artifacts/               # all non-image research evidence
            ├── checkpoints/
            │   ├── best_model.pt
            │   └── final_model.pt
            ├── test_hourly_trajectories/
            ├── config_snapshot.yaml
            ├── run_metadata.json
            ├── training_history.json
            ├── validation_history.json
            ├── training_summary.json
            ├── test_daily_metrics.csv
            ├── test_summary.json
            ├── summary.json
            └── plot_manifest.json
```

Do not write new checkpoints, histories, summaries, or figures directly into `outputs/` or
`outputs/releases/` during training. Do not mix runs from different algorithms.

When a run has been reviewed and accepted for long-term retention, promote it explicitly:

```powershell
python scripts/manage_results.py promote `
  --run outputs/working_runs/<algorithm_slug>/runs/<run_id> `
  --release <release_name>
```

Retained results live under:

```text
outputs/releases/
├── registry.csv
└── <release_name>/
    └── <run_id>/
        ├── release_manifest.json
        ├── README.md
        └── complete copied run artifacts
```

Release promotion copies and verifies the run; it must never delete the working source. Existing
release runs must never be overwritten. A dirty or unknown Git state is rejected by default; use
`--allow-dirty` only when the user explicitly accepts that reproducibility risk. Retained release
artifacts are treated as immutable research evidence.

Figures must read saved result files. Plotting must never rerun the environment, policy test, or MILP.

## 7. Removing a failed algorithm

A failed experiment must remain removable without editing another algorithm:

```text
algorithms/<failed_algorithm>/
configs/algorithms/<failed_algorithm>.yaml
outputs/working_runs/<failed_algorithm>/     # optional; delete only if its results are no longer needed
```

Never delete canonical MATD3 or shared pipeline files while removing an experiment. Result deletion
is separate from code deletion and must respect the user's retention request. Never delete a retained
`outputs/releases/...` result as part of routine working-run cleanup.

## 8. Required verification for algorithm or pipeline changes

Before reporting completion:

1. Run `pytest -q`.
2. Run a short pipeline smoke test with 1–3 training/test days as appropriate.
3. Confirm the selected plugin and its own YAML were used.
4. Confirm `best_model.pt` is the formal-test checkpoint.
5. Confirm test, Equal/Shapley settlement, summary, and plotting completed.
6. Confirm figures include 6 PNG and 6 PDF files for a four-season formal test: 00, 01 and four seasonal dispatch overviews. Independent learning-curve evaluation and its 01b figure are retired. Short smoke tests include only the available seasons. Never fabricate missing seasonal or learning-curve data. Seasonal selection uses saved exogenous 24-hour profiles only, with source hashes and selected dates in artifacts/plot_manifest.json.
7. Confirm normal results were written under `outputs/working_runs/<slug>/runs/`.
8. If testing promotion, confirm the source working run remains and the retained run contains a manifest, README, and registry entry.
9. Confirm no canonical MATD3 file changed for an experimental algorithm.

For detailed human-oriented guidance, see `docs/ALGORITHM_DEVELOPMENT.md`.

## 9. Server work and handoff

Before connecting to the training server, changing its code, starting a run, or downloading
results, read `docs/SERVER_WORKFLOW.md` for connection details, verified environment,
current task status, and synchronization conventions. Recheck live status before acting.
Connection information is context, not blanket authorization to start training or publish code.
