# Experiment Result Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all normal training runs land under `outputs/working_runs/`, add explicit promotion into immutable named releases, and bind every retained result to Git branch/commit/dirty state plus a global release registry.

**Architecture:** Keep the existing shared training pipeline and algorithm plugins unchanged in behavior. Extend the experiment/output-management layer only: the training parser/layout writes working runs; a new `experiments/result_lifecycle.py` owns validation/promotion/manifest/registry logic; a thin `scripts/manage_results.py` exposes the CLI. Promotion copies to a temporary destination, validates, atomically finalizes, and never deletes the source run.

**Tech Stack:** Python 3, `pathlib`, `argparse`, `json`, `csv`, `shutil`, `subprocess`, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-experiment-result-lifecycle-design.md`

## Global Constraints

- Baseline branch: `reproduce/pnn-transformer-8.99` at `ffb5e835858cb5b3963dac30fd53e48df5d66156`.
- Implementation branch: `codex/controllable-devices`.
- Do not alter model architecture, learning updates, environment physics, reward logic, settlement logic, plotting semantics, or checkpoint serialization.
- Training must never write directly to `outputs/releases/`.
- Promotion must never overwrite an existing retained run.
- Promotion must preserve the source working run.
- Dirty-code runs are rejected by default and require explicit `--allow-dirty`.
- Release artifacts are immutable source evidence; derived analyses belong elsewhere.
- Existing explicit `--output-root` / compatibility output overrides must keep working.

---

## File Map

- Modify `experiments/layout.py` — default working-run root semantics and path helpers only.
- Modify `experiments/training_pipeline.py` — default CLI output root and Git branch/dirty metadata capture.
- Create `experiments/result_lifecycle.py` — pure result validation and promotion implementation.
- Create `scripts/manage_results.py` — CLI wrapper for `promote`.
- Modify `tests/test_algorithm_registry_and_layout.py` — working-run layout coverage.
- Create `tests/test_result_lifecycle.py` — promotion, collision, dirty-state, registry, source-preservation tests.
- Modify `tests/test_experiment_profiles.py` or add focused metadata tests if that file is cleaner for parser/metadata behavior.
- Modify `AGENTS.md` — authoritative output-layout contract.
- Modify `docs/ALGORITHM_DEVELOPMENT.md` — human workflow documentation.
- Optionally modify `README.md` only if it currently documents the old result path directly.

---

### Task 1: Make working runs the default output location

**Files:**
- Modify: `experiments/layout.py`
- Modify: `experiments/training_pipeline.py`
- Modify: `tests/test_algorithm_registry_and_layout.py`

**Interfaces:**
- `ExperimentLayout.create(output_root, algorithm_slug, display_name, seed, *, algorithm_root_override=None)` remains backward compatible.
- New constant: `DEFAULT_WORKING_OUTPUT_ROOT = Path("outputs") / "working_runs"` in `experiments/layout.py`.
- `build_parser()` uses `DEFAULT_WORKING_OUTPUT_ROOT` as `--output-root` default.

- [ ] **Step 1: Write failing layout/parser tests**

Add assertions equivalent to:

```python
from experiments.layout import DEFAULT_WORKING_OUTPUT_ROOT, ExperimentLayout
from experiments.training_pipeline import build_parser


def test_default_working_output_root_is_explicit():
    assert DEFAULT_WORKING_OUTPUT_ROOT == Path("outputs") / "working_runs"


def test_training_parser_defaults_to_working_runs():
    args = build_parser().parse_args([])
    assert args.output_root == Path("outputs") / "working_runs"


def test_experiment_layout_under_working_root(tmp_path: Path):
    layout = ExperimentLayout.create(tmp_path / "working_runs", "pnn_transformer", "PNN-Transformer", 42)
    assert layout.run_dir.parent == (tmp_path / "working_runs" / "pnn_transformer" / "runs").resolve()
```

- [ ] **Step 2: Run targeted tests and verify failure**

Run:

```bash
pytest tests/test_algorithm_registry_and_layout.py -q
```

Expected: failures because `DEFAULT_WORKING_OUTPUT_ROOT` does not yet exist and parser still defaults to `outputs`.

- [ ] **Step 3: Implement the minimal default-root change**

In `experiments/layout.py` add:

```python
DEFAULT_WORKING_OUTPUT_ROOT = Path("outputs") / "working_runs"
```

Keep `ExperimentLayout.create()` generic; do not hard-code release semantics into it.

In `experiments/training_pipeline.py` import the constant and change:

```python
parser.add_argument("--output-root", type=Path, default=DEFAULT_WORKING_OUTPUT_ROOT,
                    help="Working runs are isolated below OUTPUT_ROOT/<algorithm>/runs/.")
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
pytest tests/test_algorithm_registry_and_layout.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/layout.py experiments/training_pipeline.py tests/test_algorithm_registry_and_layout.py
git commit -m "feat: route training runs to working output area"
```

---

### Task 2: Record complete Git identity for every run

**Files:**
- Modify: `experiments/training_pipeline.py`
- Create or modify test: `tests/test_experiment_profiles.py` or a focused metadata test file

**Interfaces:**
- Replace the single `_git_commit()` helper with:

```python
def _git_identity() -> dict[str, Any]:
    return {
        "git_branch": str | None,
        "git_commit": str | None,
        "git_dirty": bool | None,
    }
```

- `git_dirty=None` means Git identity could not be determined, not “clean”.

- [ ] **Step 1: Write failing unit tests for Git identity**

Use `monkeypatch` on `subprocess.run` so tests do not depend on the developer machine. Cover clean, dirty, detached-HEAD, and Git-unavailable cases. Example expectation:

```python
assert identity == {
    "git_branch": "codex/controllable-devices",
    "git_commit": "abc123",
    "git_dirty": False,
}
```

For detached HEAD, branch should be `None` while commit is retained.

- [ ] **Step 2: Run the focused tests and verify failure**

Run the exact test module with `pytest -q`.

Expected: FAIL because `_git_identity()` does not exist.

- [ ] **Step 3: Implement `_git_identity()`**

Use:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --porcelain --untracked-files=normal
```

Rules:

```python
branch = None if raw_branch == "HEAD" else raw_branch
commit = raw_commit or None
dirty = bool(status_stdout.strip())
```

If Git commands fail, return `None` for unknown fields without blocking training.

- [ ] **Step 4: Merge Git identity into `run_metadata.json`**

Replace the current `"git_commit": _git_commit()` entry with:

```python
**_git_identity(),
```

Do not change any training/evaluation behavior.

- [ ] **Step 5: Run tests**

Run the focused Git-identity tests plus:

```bash
pytest tests/test_algorithm_registry_and_layout.py tests/test_experiment_profiles.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/training_pipeline.py tests/
git commit -m "feat: record git branch commit and dirty state"
```

---

### Task 3: Implement release validation and promotion core

**Files:**
- Create: `experiments/result_lifecycle.py`
- Create: `tests/test_result_lifecycle.py`

**Interfaces:**

```python
class PromotionError(RuntimeError):
    pass


def validate_release_name(name: str) -> str: ...


def inspect_run(run_dir: Path) -> dict[str, Any]: ...


def promote_run(
    run_dir: Path,
    releases_root: Path,
    release_name: str,
    *,
    allow_dirty: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]: ...
```

`promote_run()` returns the finalized manifest dict.

Required artifacts:

```python
REQUIRED_ARTIFACTS = (
    "config_snapshot.yaml",
    "run_metadata.json",
    "training_summary.json",
    "summary.json",
    "test_daily_metrics.csv",
    "checkpoints/best_model.pt",
)
```

At least one of `test_summary.json` or `test_evaluation.json` must exist. `validation_history.json` is required only when metadata/config shows validation was enabled; otherwise it is optional.

- [ ] **Step 1: Write failing release-name tests**

Accept names such as `baseline_v1`, `flexible-load-v2`, and `chapter4_baseline`.
Reject empty names, `.`, `..`, path separators, absolute paths, and names containing traversal components.

- [ ] **Step 2: Write failing run-inspection tests**

Build a minimal fake completed run in `tmp_path` with internally consistent `run_id`, algorithm identity, metadata, summary, and `best_model.pt`. Test missing artifact, inconsistent `run_id`, inconsistent algorithm slug, and missing Git commit.

- [ ] **Step 3: Run tests and confirm failure**

```bash
pytest tests/test_result_lifecycle.py -q
```

Expected: FAIL because module/functions do not exist.

- [ ] **Step 4: Implement `validate_release_name()` and `inspect_run()`**

`inspect_run()` must parse JSON defensively and raise `PromotionError` with a precise path/reason. It returns normalized data including:

```python
{
    "run_id": ...,
    "algorithm_slug": ...,
    "algorithm_display_name": ...,
    "seed": ...,
    "train_episodes": ...,
    "test_episodes": ...,
    "git_branch": ...,
    "git_commit": ...,
    "git_dirty": ...,
    "config_hash": ...,
    "best_validation_episode": ...,
    "best_validation_score": ...,
}
```

Treat `git_dirty is None` as unsafe for release unless `allow_dirty=True`; a release must never silently interpret unknown Git state as clean.

- [ ] **Step 5: Write failing promotion safety tests**

Cover:

```text
clean complete run -> promoted
dirty run -> rejected
unknown dirty state -> rejected
allow_dirty=True -> promoted and warning recorded
existing destination -> rejected
missing artifact -> rejected
source directory remains after success
source directory remains after failure
```

- [ ] **Step 6: Implement copy-verify-finalize promotion**

Destination shape:

```text
<releases_root>/<release_name>/<run_id>/
    release_manifest.json
    README.md
    ...copied run artifacts...
```

Temporary path:

```text
<releases_root>/<release_name>/.<run_id>.promoting
```

Algorithm:

```python
1. inspect source
2. enforce dirty policy
3. reject final/temp collisions
4. copytree(source, temp)
5. inspect copied temp
6. write per-run release_manifest.json into temp
7. write per-run README.md into temp
8. rename temp -> final with Path.replace/rename only when final is absent
9. append registry row
10. return manifest
```

If any step before final rename fails, remove only the temporary `.promoting` directory and leave source untouched.

If registry append fails after finalization, raise `PromotionError` stating that the finalized run exists but registry repair is required; do not delete a valid finalized release automatically.

- [ ] **Step 7: Generate README content from metadata**

README must include release name, run ID, algorithm, seed, train/test episode counts, branch, commit, dirty status, config hash, best-validation episode/score, promotion timestamp, original source path, and an explicit warning when `allow_dirty` was used.

- [ ] **Step 8: Run result-lifecycle tests**

```bash
pytest tests/test_result_lifecycle.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add experiments/result_lifecycle.py tests/test_result_lifecycle.py
git commit -m "feat: add safe experiment release promotion"
```

---

### Task 4: Add append-only global release registry

**Files:**
- Modify: `experiments/result_lifecycle.py`
- Modify: `tests/test_result_lifecycle.py`

**Interfaces:**

```python
REGISTRY_FIELDS = (
    "release_name", "run_id", "algorithm_slug", "seed",
    "train_episodes", "test_episodes", "git_branch", "git_commit",
    "git_dirty", "config_hash", "best_validation_episode",
    "best_validation_score", "promoted_at", "release_run_directory",
)


def append_registry_row(registry_path: Path, manifest: Mapping[str, Any]) -> None: ...
```

- [ ] **Step 1: Add failing registry tests**

Verify first promotion creates a header plus one row; second distinct run appends a second row; attempting to append the same `(release_name, run_id)` raises `PromotionError` instead of duplicating it.

- [ ] **Step 2: Run targeted tests and verify failure**

```bash
pytest tests/test_result_lifecycle.py -q
```

- [ ] **Step 3: Implement registry append with duplicate protection**

Read existing CSV when present, check `(release_name, run_id)` uniqueness, then append exactly one row with `csv.DictWriter`.

Do not rewrite historical rows during normal promotion.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_result_lifecycle.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/result_lifecycle.py tests/test_result_lifecycle.py
git commit -m "feat: add release experiment registry"
```

---

### Task 5: Expose the explicit promotion CLI

**Files:**
- Create: `scripts/manage_results.py`
- Modify: `tests/test_result_lifecycle.py` or create `tests/test_manage_results_cli.py`

**Interfaces:**

CLI:

```powershell
python scripts/manage_results.py promote \
  --run <working-run-dir> \
  --release <release-name> \
  [--releases-root outputs/releases] \
  [--allow-dirty]
```

- [ ] **Step 1: Write failing parser/CLI tests**

Test that `promote` requires `--run` and `--release`, defaults releases root to `outputs/releases`, and forwards `allow_dirty` correctly.

- [ ] **Step 2: Run tests and verify failure**

Run the new CLI test module.

- [ ] **Step 3: Implement thin CLI wrapper**

The CLI should only parse arguments, call `promote_run()`, print a compact JSON summary, and return nonzero with a clear message on `PromotionError`. Do not duplicate validation logic in the script.

- [ ] **Step 4: Run CLI tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/manage_results.py tests/
git commit -m "feat: add experiment release management cli"
```

---

### Task 6: Update repository contracts and human documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/ALGORITHM_DEVELOPMENT.md`
- Modify: `README.md` only if it contains the old path contract

**Interfaces:**
- Documentation must state active runs and retained runs separately.

- [ ] **Step 1: Update authoritative output layout in `AGENTS.md`**

Replace the old single-path contract with:

```text
outputs/
├── working_runs/<algorithm_slug>/runs/<run_id>/...
└── releases/<release_name>/<run_id>/...
```

State that all shared training commands write to `working_runs`; only `scripts/manage_results.py promote` may create retained release runs.

- [ ] **Step 2: Document the research workflow**

In `docs/ALGORITHM_DEVELOPMENT.md`, add:

```text
modify code/config
→ run normally into working_runs
→ inspect/debug
→ commit accepted code and ensure clean tree
→ rerun retained experiment if needed
→ promote accepted run
→ use releases/registry.csv for later comparison
```

Document `--allow-dirty` as an exceptional escape hatch that leaves a warning in the release manifest.

- [ ] **Step 3: Search for stale path references**

Run:

```bash
grep -R "outputs/<algorithm_slug>/runs\|outputs/<slug>/runs\|outputs/<algorithm>/runs" -n AGENTS.md README.md docs scripts experiments tests
```

Update only references that describe the default current workflow; preserve historical/migration documentation where explicitly labeled legacy.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md docs/ALGORITHM_DEVELOPMENT.md README.md
git commit -m "docs: document working and release result lifecycle"
```

---

### Task 7: Regression verification and smoke test

**Files:**
- No intended production-code changes unless verification reveals a defect.

**Interfaces:**
- Existing algorithm behavior must remain unchanged.

- [ ] **Step 1: Run the focused result-management tests**

```bash
pytest tests/test_result_lifecycle.py tests/test_algorithm_registry_and_layout.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full repository test suite**

```bash
pytest -q
```

Expected: PASS with no new failures.

- [ ] **Step 3: Run a short shared-pipeline smoke test**

Use the accelerated PNN-Transformer branch behavior and minimal explicit episode counts, for example:

```powershell
python scripts/run_experiment.py --algorithm pnn_transformer --device cpu --episodes 1 --test-episodes 1 --validation-days 1 --no-dashboard
```

If CUDA is available in the execution environment, `--device cuda` may be used instead; device choice must not alter result-management semantics.

Verify the run appears under:

```text
outputs/working_runs/pnn_transformer/runs/<run_id>/
```

and that `run_metadata.json` contains `git_branch`, `git_commit`, and `git_dirty`.

- [ ] **Step 4: Promote the smoke run into a temporary release root**

Do not use the user's real retained-results folder for verification. Example:

```powershell
python scripts/manage_results.py promote \
  --run outputs/working_runs/pnn_transformer/runs/<run_id> \
  --release lifecycle_smoke \
  --releases-root outputs/_release_smoke
```

Verify:

```text
outputs/_release_smoke/lifecycle_smoke/<run_id>/release_manifest.json
outputs/_release_smoke/lifecycle_smoke/<run_id>/README.md
outputs/_release_smoke/registry.csv
```

If the smoke run is dirty by construction, explicitly use `--allow-dirty` and verify the manifest records the override.

- [ ] **Step 5: Verify canonical algorithm files were not changed**

Run:

```bash
git diff reproduce/pnn-transformer-8.99 -- algorithms/matd3 algorithms/pnn algorithms/pnn_transformer core envs
```

Expected: no result-lifecycle-driven changes in model/environment logic.

- [ ] **Step 6: Review final branch diff**

Run:

```bash
git diff reproduce/pnn-transformer-8.99...HEAD --stat
git status --short
```

Expected: only planned experiment-management, tests, and documentation changes; clean working tree after commits.

- [ ] **Step 7: Final verification commit only if fixes were required**

If verification required corrective edits, commit those edits separately with a message that names the verified defect. Otherwise do not create an empty commit.
