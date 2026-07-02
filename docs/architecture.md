# Repository architecture

## Purpose

This document defines the implemented repository boundaries and target component ownership for the
incremental modernization. The separation prevents source data, reusable code, notebooks, tests,
and generated outputs from becoming mixed again.

The original notebooks, `wrangle.py`, and presentation images are preserved together under
`archive/original_2022/`. The modern package and locked development environment are implemented;
pipeline modules remain intentionally absent until supported behavior and tests arrive together.

## Directory map

```text
.
├── archive/original_2022/     # preserved, unsupported original project bundle
├── artifacts/                 # ignored generated models and run outputs
├── configs/                   # versioned pipeline and experiment configuration
├── data/
│   ├── raw/                   # immutable upstream files; ignored
│   ├── external/              # other externally sourced files; ignored
│   ├── interim/               # resumable transformations; ignored
│   └── processed/             # model-ready outputs; ignored
├── docs/                      # architecture, provenance, limitations, roadmap
├── notebooks/                 # curated notebooks using supported package APIs
├── reports/
│   └── figures/               # reproducible generated figures; ignored
├── scripts/                   # thin operational entry points
├── src/ecg_anomaly_detection/ # installable Python package
├── pyproject.toml             # package metadata and tool configuration
├── uv.lock                    # exact cross-platform dependency resolution
└── tests/
    ├── fixtures/              # small synthetic, redistributable fixtures
    ├── integration/           # component-boundary and workflow tests
    └── unit/                  # isolated transformation and validation tests
```

## Boundary rules

### Source code

Reusable behavior belongs under `src/ecg_anomaly_detection/`. The package currently implements
versioned dataset configuration, repository paths, local file inventory, SHA-256 verification, and
their CLI boundary. Notebooks and scripts should call package functions rather than carry duplicate
implementations. New modules are added only with supported behavior and tests.

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

Curated notebooks will live in `notebooks/` and consume package APIs. Original 2022 notebooks are preserved in `archive/original_2022/`, separate from supported notebooks and without rewriting their historical outputs.

### Tests

Tests may commit only small synthetic fixtures or fixtures with explicit redistribution permission. No source ECG recordings or patient-derived extracts should be added for convenience.

### Generated outputs

Models, serialized objects, run outputs, and generated figures are ignored by default. Reproducible metadata and selected portfolio figures may be committed later through an explicit review rather than by weakening the ignore policy globally.

## Planned migration sequence

1. **Completed:** add the reproducible Python environment, package metadata, and CI smoke test.
2. **Completed:** define the expected MIT-BIH file inventory and local integrity checks.
3. **Next:** load and validate WFDB signals and annotations with synthetic tests.
4. Add record-grouped splitting, run manifests, and machine-readable metrics.
5. Create curated notebooks that call supported package APIs.

See the [proposed pipeline design](pipeline-design.md) for stage contracts and lineage metadata. The
directory structure defines ownership; it does not imply that the archived workflow has already
been refactored or reproduced.
