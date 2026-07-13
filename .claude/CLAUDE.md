# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A 2022 educational ECG (MIT-BIH Arrhythmia Database) machine-learning project being incrementally
modernized into a reproducible, auditable data-engineering portfolio case study — not into a
production or clinical system. It is explicitly **not medical software** and must never be
described as diagnostic, clinical, or treatment-related. See the "Project positioning" section of
`AGENTS.md` for the full framing rules, and follow them.

**This repo already has an `AGENTS.md`** with detailed operating rules (Fish-shell-only command
syntax, Git workflow, PR/issue metadata requirements, GitHub Project #5 field conventions,
modernization/documentation conventions). Read it — those rules apply to you too. This file adds
commands and architecture context that `AGENTS.md` doesn't cover.

## Shell

The user's interactive shell is **Fish**, on macOS. Always give Fish-syntax commands, not
Bash/Zsh (e.g. `set -gx NAME value` not `export NAME=value`; `source .venv/bin/activate.fish` not
`source .venv/bin/activate`).

## Commands

Environment setup (uses [uv](https://docs.astral.sh/uv/), Python pinned via `.python-version`):

```fish
uv sync --locked --dev
uv run pre-commit install --install-hooks
```

Tests:

```fish
uv run pytest                          # full suite (unit + integration + scripts)
uv run pytest tests/unit/test_windows.py                    # one file
uv run pytest tests/unit/test_windows.py::test_specific_case  # one test
uv run pytest -k "split"               # by keyword
```

Type checking, linting, formatting:

```fish
uv run pyright                         # Basic mode, src/ and tests/
uv run pre-commit run --all-files      # ruff-check --fix, ruff-format, markdownlint, gitleaks, zizmor, pyright, etc.
```

Run the full local pipeline (downloads MIT-BIH into the ignored `data/raw/` zone on first run):

```fish
uv run ecg-data run-pipeline \
  --repository-root . \
  --dataset-config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --split-config configs/splitting-v2.toml \
  --training-config configs/training-baseline-v1.toml \
  --evaluation-config configs/evaluation-baseline-v1.toml
```

Individual pipeline stages are also exposed as their own `ecg-data` subcommands — `acquire`,
`inventory`, `verify`, `validate-record`, `map-annotations`, `extract-windows`, `split-windows`,
`index-dataset`, `create-run-manifest`, `validate-benchmark-policy`, `list-runs`, `purge-run`. Run
`uv run ecg-data <subcommand> --help` for arguments; each maps to one module under
`src/ecg_anomaly_detection/`.

Local artifact cleanup:

```fish
uv run ecg-data list-runs --repository-root .
uv run ecg-data purge-run --repository-root . --run-id <run-id> --dry-run
```

Notebook checks (optional `notebooks` dependency group; never executes cells against the real
dataset — synthetic data only):

```fish
uv run --group notebooks ecg-data check-local-notebooks
```

Build package artifacts without publishing (used as a CI assurance check only):

```fish
uv build
```

Dependency changes go through `uv add` / `uv add --dev` / `uv add --group notebooks` / `uv add --group experiments`
so `pyproject.toml` and `uv.lock` stay in sync — never edit dependency versions by hand.

## Architecture

### Pipeline shape

The package implements one linear, local, sequential pipeline over the MIT-BIH Arrhythmia
Database:

```text
acquire → inventory → validate → map annotations → extract windows
                                                    ↓
                                subject/record-aware split → dataset index
                                                    ↓
                         training → validation-only evaluation → run manifest
```

`src/ecg_anomaly_detection/pipeline.py` (`run_pipeline`) is the orchestrator: it loads all six
stage configs up front, generates a UUID `run_id`, and calls each stage module in order, writing
outputs under `data/{raw,interim,processed}/` and `artifacts/runs/<run-id>/`. Every run is
isolated by its UUID so runs never collide or share mutable state. The CLI (`cli.py`) is a thin
argparse wrapper — one subcommand per stage function, plus the `run-pipeline` composite — and
should never contain business logic itself; that belongs in the stage modules.

### Module responsibilities (`src/ecg_anomaly_detection/`)

- `config.py` — versioned TOML dataset config loading (`configs/*.toml`), shared by all stages.
- `acquisition.py` — fail-safe HTTPS retrieval of the dataset from PhysioNet into `data/raw/`;
  idempotent (later runs verify and reuse rather than re-download).
- `inventory.py` — expected-file inventory and SHA-256 integrity baseline/verification.
- `records.py` — typed WFDB signal/annotation ingestion and structural validation.
- `labels.py` — versioned, closed-world annotation → binary-target mapping with audit counts.
- `windows.py` — boundary-safe beat-centered window extraction to NPZ, preserving row-level
  record/beat lineage (channel identity is resolved by name, `MLII`, not position — see
  `docs/window-extraction.md`).
- `splitting.py` — deterministic **record/subject-grouped** train/val/test partitioning (the
  modernization's core fix vs. the 2022 random-window split — see "Leakage" below), plus split
  quality diagnostics/enforcement.
- `dataset_index.py` — indexes grouped record shards for lazy loading without concatenating
  arrays; the training/evaluation boundary reads only from this index.
- `training.py` — deterministic baseline estimator fit **only from indexed training shards**.
- `evaluation.py` — evaluates the frozen model **only on indexed validation shards**; the indexed
  test partition is never opened by supported code (see "Held-out test" below).
- `run_manifest.py` — ties Git revision, environment, config digests, dataset/split lineage, and
  artifact digests together into one auditable manifest per run.
- `reproducibility.py` — environment/runtime/resource evidence capture.
- `benchmark_policy.py` — loads/validates the protected-test evaluation policy (governance, not
  execution — see `docs/benchmark-governance.md`).
- `local_execution.py` — `list-runs` / `purge-run`: reclaim disk from a specific run's output by
  exact run ID, without touching the shared acquisition baseline or other runs.
- `experiment_tracking.py` — `ExperimentTracker`, used by ignored local notebooks for
  interruption-tolerant hyperparameter search (checkpoints one candidate at a time).
- `notebook_quality.py` — static checks (format/strip-outputs/lint) for the ignored
  `notebooks/local/` sandbox; never executes cells.
- `progress.py` — per-stage progress banners with elapsed time, flushed per line so subprocess
  consumers (including the Step 0 notebook) can stream it live.

### Two non-obvious invariants that shape most changes

1. **Leakage-aware splitting is load-bearing.** The 2022 archived notebook split *individual
   windows* randomly after window creation, so windows from the same record/subject could land in
   both train and test — its 95.3% accuracy is explicitly flagged as likely inflated
   (`docs/historical-results.md`). The modern pipeline instead splits at the *record/subject*
   level with assertions that subjects/records never cross partitions (`splitting.py`,
   `docs/record-grouped-splitting.md`). Any change touching windows/splitting must preserve this
   guarantee and its tests.
2. **The indexed test partition is never opened by supported code.** `evaluation.py` only ever
   reads validation shards. Full test-set evaluation is a deliberately deferred, separately
   governed milestone (`docs/evaluation-policy.md`, `docs/benchmark-governance.md`) — don't add
   code paths that read the test partition without checking those docs first.

### Directory boundaries (see `docs/architecture.md` for the full contract)

- `src/ecg_anomaly_detection/` — the only place reusable/tested behavior belongs. Notebooks and
  `scripts/` call into it; they don't reimplement it.
- `configs/` — versioned, non-secret TOML pipeline configuration.
- `data/{raw,external,interim,processed}/` and `artifacts/`, `reports/figures/` — all generated
  and **gitignored**; the only tracked files are `.gitkeep` placeholders, `data/README.md`, and
  the deliberately allowlisted `reports/figures/modern-pipeline-lineage.svg` (`.gitignore:144`).
  Never commit source or derived ECG data.
- `archive/original_2022/` — preserved, unsupported historical notebooks/images; excluded from
  lint/format hooks (`ruff.toml`, `.pre-commit-config.yaml`) and must not be rewritten except via
  a dedicated, explicitly reviewed archival repair.
- `notebooks/` (curated, 00→01→02) call supported package APIs and are execution-tested in CI
  against synthetic data; `notebooks/local/` is an ignored, disposable experimentation sandbox
  (not supported workflow input or benchmark evidence).
- `tests/{unit,integration,scripts}/` plus `tests/fixtures/` — only small synthetic or
  explicitly-redistributable fixtures; no real patient-derived ECG data.
- `scripts/` — thin operational entry points (label sync, drift detection, notebook/project
  metadata validation), not pipeline logic.

### Testing conventions

Introduce supported behavior and its tests together. `tests/unit/` isolates individual
transformations/validation; `tests/integration/` exercises CLI/pipeline component boundaries and
one synthetic end-to-end pipeline run; `tests/scripts/` tests `scripts/` tooling. Pytest is
configured with `--strict-config --strict-markers` (`pyproject.toml`) — undeclared markers/config
fail the run rather than warn.

## Git and PR workflow notes specific to this repo

- Commits directly to `main` are blocked by a local pre-commit hook; always branch first.
- `AGENTS.md` documents mandatory PR/issue metadata (labels from `.github/labels.json`,
  milestone, GitHub Project #5 field population with specific Status transitions). Do not skip
  this when opening PRs or issues here — see also this user's own standing memory on GitHub
  Project #5 conventions.
- `docs/governance/label-taxonomy.md` is authoritative on label spelling; the historical
  pre-taxonomy migration there is a record, not a menu — never mint a legacy-style label.
