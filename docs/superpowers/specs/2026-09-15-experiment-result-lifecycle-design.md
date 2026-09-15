# Experiment Result Lifecycle Design

Date: 2026-09-15
Branch: `codex/controllable-devices`
Baseline: `reproduce/pnn-transformer-8.99` at `ffb5e835858cb5b3963dac30fd53e48df5d66156`

## 1. Goal

Introduce a two-stage experiment-result lifecycle that supports frequent debugging and iteration without mixing temporary runs with research results that must be retained for later analysis and paper writing.

The system must keep the current shared training pipeline and algorithm isolation intact. It must not change MATD3/PNN/Transformer learning behavior, environment physics, reward logic, checkpoint format, evaluation logic, settlement logic, or plotting semantics.

## 2. Lifecycle

Every new training run is temporary by default.

```text
code/config change
    ↓
training
    ↓
outputs/working_runs/<algorithm>/runs/<run_id>
    ↓
inspect and compare
    ├── reject → keep temporarily or delete later
    └── accept → explicit promote command
                    ↓
              outputs/releases/<release_name>/<run_id>
```

A run becomes a retained research result only through an explicit promotion operation after training has completed and the user has decided that the run is worth preserving.

Training must never write directly into `outputs/releases/`.

## 3. Directory Layout

Default layout:

```text
outputs/
├── working_runs/
│   ├── matd3_dispatch/
│   │   ├── latest_run.json
│   │   └── runs/
│   ├── matd3_transformer/
│   ├── pnn/
│   └── pnn_transformer/
│
└── releases/
    ├── registry.csv
    ├── <release_name>/
    │   ├── release_manifest.json
    │   ├── README.md
    │   └── <run_id>/
    │       └── complete run artifacts
    └── ...
```

The internal structure of a working run remains compatible with the current pipeline:

```text
<run_id>/
├── checkpoints/
│   ├── best_model.pt
│   └── final_model.pt
├── figures/
├── test_hourly_trajectories/
├── config_snapshot.yaml
├── run_metadata.json
├── training_history.json
├── training_history.jsonl        # only when enabled by logging config
├── validation_history.json
├── training_summary.json
├── test_daily_metrics.csv
├── test_evaluation.json
├── test_summary.json
└── summary.json
```

This change is intentionally limited to the experiment/output-management layer. Algorithm packages continue to use the shared pipeline.

## 4. Default Training Behavior

`python scripts/run_experiment.py ...` continues to be the single training entry point.

Unless an explicit compatibility output override is supplied, the default output root becomes:

```text
outputs/working_runs
```

Therefore a command such as:

```powershell
python scripts/run_experiment.py --algorithm pnn_transformer --device cuda
```

writes to:

```text
outputs/working_runs/pnn_transformer/runs/<run_id>
```

No `--stage release` mode is added. Release status is deliberately assigned only after a completed run is reviewed.

## 5. Git Identity Recorded for Every Run

The current pipeline already records `git_commit`. Extend the run metadata to record:

- `git_branch`
- `git_commit`
- `git_dirty`

`git_dirty` means the working tree contained tracked or untracked changes at run start.

Git metadata is informational during normal training. Training is allowed from a dirty tree because short debugging runs are expected during development.

## 6. Promotion Command

Add one result-management entry point, for example:

```powershell
python scripts/manage_results.py promote \
  --run outputs/working_runs/pnn_transformer/runs/<run_id> \
  --release flexible_load_v1
```

The command must:

1. Resolve and validate the source run path.
2. Confirm the source is a completed working run, not an arbitrary directory.
3. Validate required artifacts.
4. Read run metadata and summary data.
5. Reject dirty-code runs by default.
6. Reject destination collisions instead of overwriting an existing retained result.
7. Copy the complete run into the release directory using a temporary destination first.
8. Verify the copied release before finalizing it.
9. Generate `release_manifest.json` and a concise `README.md`.
10. Append one row to `outputs/releases/registry.csv` only after successful finalization.

Promotion uses copy-then-verify rather than move. The working run remains in place until the user later chooses to clean working results. This avoids losing the only copy if a filesystem or cloud-sync operation is interrupted.

## 7. Promotion Validation

Required artifacts for a completed retained run:

- `config_snapshot.yaml`
- `run_metadata.json`
- `training_summary.json`
- `summary.json`
- `validation_history.json` when validation was enabled
- `test_daily_metrics.csv`
- `test_summary.json` or `test_evaluation.json`
- `checkpoints/best_model.pt`

`checkpoints/final_model.pt`, figures, hourly trajectories, diagnostics, and full training history are retained when present but do not independently determine promotion eligibility.

The promotion check must verify that:

- metadata and summary refer to the same `run_id`;
- algorithm identity is consistent;
- the selected best checkpoint exists;
- formal testing completed;
- the recorded source Git commit exists in metadata;
- the release name is safe as a filesystem component;
- the destination does not already exist.

If `git_dirty` is true, promotion fails unless the user explicitly passes `--allow-dirty`. The release manifest must record that override when used.

## 8. Release Immutability

A release is treated as immutable research evidence.

The promotion command must never overwrite an existing release run. If the same scientific idea changes, create a new release name such as:

```text
flexible_load_v1
flexible_load_v2
```

If figures or derived analyses need to be regenerated later, raw release artifacts remain unchanged. New analysis outputs should live in a separate analysis/report directory rather than replacing source result files.

## 9. Release Manifest

Each release group receives a machine-readable manifest containing at least:

```text
release_name
promoted_at
run_id
algorithm_slug
algorithm_display_name
seed
train_episodes
test_episodes
experiment_profile
config_hash
git_branch
git_commit
git_dirty
allow_dirty_override
source_run_directory
release_run_directory
best_checkpoint
best_validation_episode
best_validation_score
```

The manifest is the authoritative bridge between a retained result and the exact code/configuration that produced it.

## 10. Human-Readable README

Each release group also contains `README.md` summarizing:

- release purpose;
- algorithm and run ID;
- seed and training/test episode counts;
- Git branch/commit/dirty status;
- config hash;
- best-validation checkpoint information;
- promotion timestamp;
- original working-run path;
- any dirty-code override warning.

The README is generated from saved metadata rather than manually reconstructed later.

## 11. Global Registry

`outputs/releases/registry.csv` contains one row per promoted run. Minimum columns:

```text
release_name
run_id
algorithm_slug
seed
train_episodes
test_episodes
git_branch
git_commit
git_dirty
config_hash
best_validation_episode
best_validation_score
promoted_at
release_run_directory
```

This registry is append-only under normal operation. A promotion is not considered successful until its registry row has been written.

The registry makes later comparison and paper-result lookup possible without inspecting timestamped directories manually.

## 12. Failure Safety

Promotion must be conservative because Google Drive or another sync client may be watching `outputs/`.

Rules:

- Never delete the source working run during promotion.
- Copy into a temporary release directory first.
- Validate the copied files before renaming/finalizing the release directory.
- Never overwrite an existing release directory.
- Do not write a registry row until promotion is complete.
- A failed promotion leaves the working run untouched and must not create a valid-looking partial release.

Temporary promotion directories use a recognizable suffix such as `.promoting` so they can be identified and removed safely after an interrupted operation.

## 13. Existing Result Utilities

`experiments/layout.py` remains the authority for run-directory creation, but gains support for the working-root layout.

The existing `scripts/organize_results.py` remains a legacy migration utility. It should not be repurposed as the release-promotion command because its current responsibility is moving historical loose artifacts into `runs/`/`legacy/` and repairing embedded paths.

A dedicated `scripts/manage_results.py` keeps promotion semantics separate from legacy migration.

## 14. Documentation Contract

Repository documentation that currently states `outputs/<algorithm>/runs/` as the only standard path must be updated to distinguish:

- active/temporary runs: `outputs/working_runs/<algorithm>/runs/`;
- retained research results: `outputs/releases/<release_name>/<run_id>/`.

The canonical MATD3 preservation rule and the shared train → validation → best checkpoint → formal test → settlement → plotting workflow remain unchanged.

## 15. Tests

Add focused tests covering at least:

1. `ExperimentLayout` defaults to `outputs/working_runs/<algorithm>/runs/`.
2. Run IDs remain unique and algorithm-isolated.
3. Git metadata fields are present in generated metadata.
4. A complete clean run can be promoted.
5. A dirty run is rejected by default.
6. `--allow-dirty` permits promotion and records the override.
7. Missing required artifacts reject promotion without modifying the source.
8. An existing destination rejects promotion rather than overwriting it.
9. A successful promotion writes manifest, README, and exactly one registry row.
10. Promotion preserves the source working run.
11. Existing shared training/evaluation behavior remains unchanged.

Before completion, run the repository's required test suite and a short shared-pipeline smoke test.

## 16. Non-Goals

This change does not:

- delete old results automatically;
- automatically decide whether a run is scientifically good;
- upload results to Google Drive directly;
- alter model architecture or learning hyperparameters;
- alter environment physics, reward, P2P settlement, or Shapley allocation;
- change checkpoint serialization;
- introduce automatic retention policies based only on episode count.

Deletion of working runs remains an explicit user action.

## 17. Initial Use After Implementation

The intended workflow for the upcoming flexible-load work is:

```text
1. Develop on codex/controllable-devices.
2. Run all experiments normally → outputs/working_runs/...
3. Inspect debug/candidate results.
4. Delete failed working runs when no longer useful.
5. When a version is accepted, commit the code and ensure the tree is clean.
6. Run/re-run the retained experiment from that clean commit.
7. Promote the accepted run into a named release.
8. Keep the release immutable and use its manifest/registry for later paper analysis.
```

This makes the code repository the source of truth for code and Google Drive/synced `outputs` the source of truth for experiment artifacts, with explicit metadata linking the two.
