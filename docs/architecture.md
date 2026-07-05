# Repository architecture

## Purpose

This document defines the implemented repository boundaries and target component ownership for the
incremental modernization. The separation prevents source data, reusable code, notebooks, tests,
and generated outputs from becoming mixed again.

The original notebooks, `wrangle.py`, and presentation images are preserved together under
`archive/original_2022/`. The modern package and locked development environment are implemented;
pipeline modules are added only when supported behavior and tests arrive together.

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

`pyproject.toml` separates core runtime, repository engineering, notebook infrastructure, and
optional model experiments. `uv.lock` resolves all groups while each documented sync selects only
the workflow it needs. See [local environment reproducibility](environment-reproducibility.md).

## Boundary rules

### Source code

Reusable behavior belongs under `src/ecg_anomaly_detection/`. The package currently implements
versioned dataset configuration, repository paths, local file inventory, SHA-256 verification,
fail-safe HTTPS acquisition, typed WFDB ingestion, record validation reports, versioned annotation
mapping, and their CLI boundaries. Boundary-safe window extraction produces ignored NPZ artifacts
with row-level lineage.
Deterministic record-grouped splitting consumes that lineage and produces an ignored JSON membership
manifest with record and target counts. Auditable run manifests connect those stage outputs to Git,
the locked environment, configuration digests, and generated artifact digests.
The local orchestrator invokes these package boundaries sequentially and isolates every run under a
UUID without moving transformation logic into the CLI.
The processed dataset index references validated per-record shards rather than duplicating their
arrays, retaining record-grouped membership and content digests for future lazy-loading consumers.
The baseline is fit only from indexed training shards. A separate evaluator verifies persisted
digests, loads only indexed validation shards, and writes deterministic machine-readable metrics.
Notebooks and scripts should call package functions rather than carry duplicate implementations.
New modules are added only with supported behavior and tests.

### Configuration

Versioned, non-secret configuration belongs in `configs/`. Machine-specific paths, credentials, and tokens must come from ignored environment files or an external secret manager. An `.env.example` may be committed later if environment variables become part of the supported interface.

### Data

The four data stages have distinct meanings:

1. `raw`: immutable files retrieved from the authoritative source;
2. `external`: additional third-party reference inputs;
3. `interim`: reproducible but non-final transformations; and
4. `processed`: model-ready tables or arrays produced by the pipeline.

Only each directory's `.gitkeep` contract is tracked. Source and derived ECG data remain ignored.
Split and run manifests are generated locally and remain ignored. Run manifests connect upstream
version and checksums, configuration, code revision, environment versions, artifact digests, row
counts, and record-level split membership without embedding ECG data.

### Notebooks

Curated notebooks will live in `notebooks/` and consume package APIs. Original 2022 notebooks are preserved in `archive/original_2022/`, separate from supported notebooks and without rewriting their historical outputs.

### Tests

Tests may commit only small synthetic fixtures or fixtures with explicit redistribution permission. No source ECG recordings or patient-derived extracts should be added for convenience.

### Generated outputs

Models, serialized objects, run outputs, and generated figures are ignored by default. Reproducible metadata and selected portfolio figures may be committed later through an explicit review rather than by weakening the ignore policy globally.

## Planned migration sequence

1. **Completed:** add the reproducible Python environment, package metadata, and CI smoke test.
2. **Completed:** define the expected MIT-BIH file inventory and local integrity checks.
3. **Completed:** load and validate WFDB signals and annotations with synthetic tests.
4. **Completed:** add a versioned, closed-world annotation mapping with audit counts.
5. **Completed:** add boundary-safe window extraction while retaining source identity.
6. **Completed:** assign complete records to deterministic grouped dataset partitions.
7. **Completed:** add run manifests that connect inputs, configuration, code, and generated outputs.
8. **Completed:** add repeatable retrieval from the authoritative dataset source.
9. **Completed:** orchestrate current stages through one tested local workflow.
10. **Completed:** index grouped model-ready record shards without concatenating arrays.
11. **Completed:** add deterministic baseline training and validation-only metric contracts.
12. **Next:** define protected test evaluation and model-card policy.
13. Create curated notebooks that call supported package APIs.

See the [proposed pipeline design](pipeline-design.md) for stage contracts and lineage metadata. The
directory structure defines ownership; it does not imply that the archived workflow has already
been refactored or reproduced.
