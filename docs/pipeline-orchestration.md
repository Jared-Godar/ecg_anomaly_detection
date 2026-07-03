# Local pipeline orchestration

## Scope

The supported orchestration command connects the implemented data stages through one
configuration-driven workflow. It acquires and inventories the configured dataset, validates every
record, maps annotations, extracts windows, assigns complete records to grouped partitions, and
writes an auditable run manifest, and fits a deterministic training-only baseline.

The workflow creates a model-ready grouped shard index and stops after fitting, before held-out
evaluation. Completing a run does not report model quality or establish clinical suitability.

## Run the supported workflow

From the repository root:

```fish
uv run ecg-data run-pipeline \
  --repository-root . \
  --dataset-config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --split-config configs/splitting-v1.toml \
  --training-config configs/training-baseline-v1.toml
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
    ├── training/
    │   ├── model.json
    │   └── training-metadata.json
    └── run-manifest.json

data/interim/runs/<run-id>/
└── windows/<record-id>.npz

data/processed/runs/<run-id>/
└── dataset-index.json
```

The final run manifest hashes every configuration, report, window artifact, processed dataset
index, fitted model, and training metadata. The split manifest retains record membership;
individual NPZ artifacts retain row-level record and annotation lineage. Only training shards are
opened by the fitting stage.

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
- The model-ready index references record shards rather than concatenating arrays.
- The v1 split balances record counts, not target distributions.
- Held-out evaluation, metrics, model selection, and model-card generation remain unimplemented.
