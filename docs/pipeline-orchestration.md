# Local pipeline orchestration

## Scope

The supported orchestration command connects the implemented data stages through one
configuration-driven workflow. It acquires and inventories the configured dataset, validates every
record, maps annotations, extracts windows, assigns complete records to grouped partitions, and
writes an auditable run manifest, fits a deterministic training-only baseline, and evaluates the
frozen model on validation shards only.

The workflow does not open or report the indexed test partition. Validation metrics are exploratory
pipeline evidence and do not establish final model quality or clinical suitability.

## Run the supported workflow

From the repository root:

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
    ├── split_quality_summary.json
    ├── environment_summary.json
    ├── runtime_summary.json
    ├── resource_summary.json
    ├── evidence_manifest.json
    ├── training/
    │   ├── model.json
    │   └── training-metadata.json
    ├── evaluation/
    │   └── validation-metrics.json
    └── run-manifest.json

data/interim/runs/<run-id>/
└── windows/<record-id>.npz

data/processed/runs/<run-id>/
└── dataset-index.json
```

The evidence summaries capture the host and interpreter, Git and lockfile identity, stage timings,
best-effort host resources, and artifact digests. The final run manifest hashes every configuration,
evidence summary, report, window artifact, processed dataset
index, fitted model, training metadata, and validation metrics. The split manifest retains record
membership;
individual NPZ artifacts retain row-level record and annotation lineage. Only training shards are
opened by fitting, and only validation shards are opened by evaluation. Test descriptors are not
resolved, opened, scored, summarized, or reported.

This operational evidence supports reproducibility review. It does not prove generalization,
clinical validity, or medical utility. Validation metrics remain validation-only, and held-out
benchmark evaluation remains intentionally protected. Runtime and resource observations may vary
by host environment and system load.

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
- Test-partition evaluation, model selection, and model-card generation remain unimplemented.

## Channel contract during artifact generation

The governed pipeline depends on window artifacts whose channel identity is explicit and consistent. During artifact generation, the public window configuration selects `MLII` by channel name instead of relying on positional index `0`.

This matters because some datasets may not expose the same signal identity at the same positional index for every record. The pipeline therefore treats mixed resolved channel identities as a data-contract failure. That failure is expected to stop model-ready artifact generation rather than producing a dataset index over inconsistent shards.

A successful pipeline run must produce model-ready artifacts only after the window extraction and shard identity contracts are satisfied, including:

- `data/processed/runs/<run-id>/dataset-index.json`;
- `artifacts/runs/<run-id>/split.json`;
- `artifacts/runs/<run-id>/split_quality_summary.json`; and
- `artifacts/runs/<run-id>/run-manifest.json`.

A clean failure message is useful diagnostics, but it is not equivalent to successful artifact generation. Notebook workflows that depend on these artifacts remain blocked until the governed pipeline produces them.
