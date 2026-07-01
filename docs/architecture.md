# Repository architecture

## Purpose

This document defines the target repository boundaries for the incremental modernization. The directories are scaffolded before implementation so data, source code, notebooks, tests, and generated outputs do not become mixed again.

The original notebooks, `wrangle.py`, and `images/` remain at the repository root for now. Moving them belongs to dedicated, history-preserving notebook and pipeline refactor changes.

## Directory map

```text
.
├── artifacts/                 # ignored generated models and run outputs
├── configs/                   # versioned pipeline and experiment configuration
├── data/
│   ├── raw/                   # immutable upstream files; ignored
│   ├── external/              # other externally sourced files; ignored
│   ├── interim/               # resumable transformations; ignored
│   └── processed/             # model-ready outputs; ignored
├── docs/                      # architecture, provenance, limitations, roadmap
├── notebooks/
│   └── archive/original_2022/ # eventual home for preserved legacy notebooks
├── reports/
│   └── figures/               # reproducible generated figures; ignored
├── scripts/                   # thin operational entry points
├── src/ecg_anomaly_detection/ # installable Python package
└── tests/
    ├── fixtures/              # small synthetic, redistributable fixtures
    ├── integration/           # component-boundary and workflow tests
    └── unit/                  # isolated transformation and validation tests
```

## Boundary rules

### Source code

Reusable behavior belongs under `src/ecg_anomaly_detection/`. Notebooks and scripts should call package functions rather than carry duplicate implementations. Empty modules are not added until their behavior and tests are introduced together.

### Configuration

Versioned, non-secret configuration belongs in `configs/`. Machine-specific paths, credentials, and tokens must come from ignored environment files or an external secret manager. An `.env.example` may be committed later if environment variables become part of the supported interface.

### Data

The four data stages have distinct meanings:

1. `raw`: immutable files retrieved from the authoritative source;
2. `external`: additional third-party reference inputs;
3. `interim`: reproducible but non-final transformations; and
4. `processed`: model-ready tables or arrays produced by the pipeline.

Only each directory's `.gitkeep` contract is tracked. Source and derived ECG data remain ignored. A future pipeline run manifest should record upstream version, checksums, configuration, code revision, row counts, and record-level split membership.

### Notebooks

Curated notebooks will eventually live in `notebooks/` and consume package APIs. Original 2022 notebooks will be preserved under `notebooks/archive/original_2022/` only after links and documentation are updated in the same change.

### Tests

Tests may commit only small synthetic fixtures or fixtures with explicit redistribution permission. No source ECG recordings or patient-derived extracts should be added for convenience.

### Generated outputs

Models, serialized objects, run outputs, and generated figures are ignored by default. Reproducible metadata and selected portfolio figures may be committed later through an explicit review rather than by weakening the ignore policy globally.

## Planned migration sequence

1. add the reproducible Python environment and package metadata;
2. move reusable wrangling behavior into the source package with tests;
3. preserve original notebooks in the archive and create curated notebooks;
4. introduce configuration-driven pipeline entry points;
5. add generated run manifests, validation reports, and CI checks.

This scaffold represents intended ownership and data flow. It does not imply that the legacy workflow has already been refactored or made reproducible.
