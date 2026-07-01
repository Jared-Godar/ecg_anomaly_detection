# Contributing

This repository is a historical educational ECG machine-learning project under incremental
modernization. Contributions should improve its value as a reproducible data-engineering case
study without implying that it is medical software or clinically validated.

## Before starting

- Read the [documentation index](docs/README.md) and
  [modernization roadmap](docs/modernization-roadmap.md).
- Keep the preserved `archive/original_2022/` bundle unchanged unless a dedicated archival repair
  is proposed and reviewed.
- Do not commit source ECG records, derived patient-level data, credentials, generated models, or
  unreviewed third-party images.
- Introduce supported behavior and its tests together.

## Local workflow

Create the locked environment and install the commit hooks:

```fish
uv sync --locked --dev
uv run pre-commit install --install-hooks
```

Create a focused branch from current `main`:

```fish
git switch main
git pull --ff-only origin main
git switch -c feature/short-description
```

Before opening a pull request, run:

```fish
uv run pytest
uv run pre-commit run --all-files
git diff --check
```

## Pull request expectations

A pull request should:

- explain what changed, why it changed, and how it was validated;
- remain small enough to review as one coherent modernization step;
- distinguish implemented behavior from proposed architecture;
- document new configuration, generated artifacts, and data contracts;
- preserve dataset attribution and the non-clinical use limitation; and
- update the roadmap or status documentation when a capability materially changes.

Any new data workflow should record source version, checksums, configuration, code revision,
schema version, row counts, and record-level split membership. Tests must use synthetic fixtures or
inputs with explicit redistribution permission.
