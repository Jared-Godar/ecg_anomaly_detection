# Local pipeline orchestration

## Scope

The supported orchestration command connects the implemented data stages through one
configuration-driven workflow. It acquires and inventories the configured dataset, validates every
record, maps annotations, extracts windows, assigns complete records to grouped partitions, and
writes an auditable run manifest.

The workflow stops before model-ready partition materialization, training, or evaluation. Completing
a run does not validate a model or establish clinical suitability.

## Run the supported workflow

From the repository root:

```fish
uv run ecg-data run-pipeline \
  --repository-root . \
  --dataset-config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --split-config configs/splitting-v1.toml
```

The first run retrieves the configured source files. Later runs verify and reuse the acquisition
baseline instead of redownloading unchanged files.

## Output layout

Each invocation receives a UUID and creates isolated ignored output directories:

```text
artifacts/
├── datasets/<dataset>/<version>/acquisition.json
└── runs/<run-id>/
    ├── inventory.json
    ├── validation/<record-id>.json
    ├── mapping/<record-id>.json
    ├── windows/<record-id>.json
    ├── split.json
    └── run-manifest.json

data/interim/runs/<run-id>/
└── windows/<record-id>.npz
```

The final run manifest hashes every configuration, report, and window artifact and records the Git
and environment identity. The split manifest retains record membership; individual NPZ artifacts
retain row-level record and annotation lineage.

## Execution and failure behavior

Records are processed sequentially in configured order. This favors deterministic evidence and
bounded memory use over local parallelism. Each record's signal arrays are released before the next
record is loaded.

Run directories are never reused or overwritten. If a later stage fails, its partial UUID directory
remains available for diagnosis and the command returns a nonzero status. A retry creates a new run
ID while reusing only the verified dataset acquisition baseline. Automatic deletion is intentionally
avoided because it would remove useful failure evidence.

All configuration must be committed under `configs/`. Generated raw, interim, and artifact files
remain ignored. The automated end-to-end test writes three tiny synthetic WFDB records and replaces
the network transport; CI never downloads MIT-BIH data.

## Current limitations

- The workflow is local and sequential; no cloud orchestrator is implemented.
- Window artifacts remain separated by record and are not yet materialized into model-ready
  partition datasets.
- The v1 split balances record counts, not target distributions.
- Training, metrics, model selection, and model-card generation remain unimplemented.
