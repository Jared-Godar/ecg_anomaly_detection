# Auditable run manifests

## Scope

The supported run-manifest stage links a pipeline run to its source revision, Python environment,
dataset inventory, configuration, grouped split, evidence reports, and generated artifacts. Every
referenced file is represented by a repository-relative path, byte size, and SHA-256 digest.

The manifest is operational lineage for a reproducible data workflow. It is not a clinical
validation record and does not establish model quality, safety, or suitability for patient care.

## Create a manifest

Run the command from the repository root after the referenced stages have completed:

```fish
uv run ecg-data create-run-manifest \
  --repository-root . \
  --inventory-manifest artifacts/mitdb-v1.0.0-inventory.json \
  --split-manifest artifacts/split-manifest.json \
  --config configs/mitdb-v1.0.0.toml \
  --config configs/annotation-map-v1.toml \
  --config configs/windowing-v1.toml \
  --config configs/splitting-v2.toml \
  --evidence artifacts/record-100-window-report.json \
  --artifact data/interim/record-100-windows.npz \
  --output artifacts/run-manifest.json
```

Repeat `--evidence` and `--artifact` for additional files. All inputs and the output must stay
within the repository boundary. Symbolic links, duplicate paths, malformed inventory or split
manifests, and output paths outside the repository are rejected.

## Schema contents

Schema version 1 records:

- a UUID run identifier and timezone-aware UTC creation time;
- the full Git commit and whether tracked or non-ignored untracked changes were present;
- Python implementation, version, platform, machine architecture, and installed package versions;
- the repository-relative `uv.lock` path and digest;
- dataset identity plus every inventoried source filename, byte size, and checksum;
- split policy, mapping and window versions, seed, record memberships, and target counts;
- the hashed split-quality summary with disjointness, distributions, ratios, and acceptance results;
- configuration, evidence, and artifact paths, sizes, and checksums.

For orchestrated runs, artifact evidence includes the frozen model, training metadata, and
`evaluation/validation-metrics.json`. The metrics document separately records the verified dataset
index, model, and validation-shard digests used for scoring.

Orchestrated runs also include the versioned reproducibility summaries and `evidence_manifest.json`
described in [Reproducibility evidence](reproducibility-evidence.md). The run manifest hashes these
documents; the evidence manifest connects the split identity to configuration, operational reports,
derived artifacts, and validation-only evaluation outputs.

Absolute local paths and file contents are not serialized. The output belongs under `artifacts/`
and remains ignored because split membership and derived artifact metadata are record-level data.

## Reproducibility interpretation

The manifest makes the inputs to a run inspectable and detects later file changes. A dirty Git state
is recorded rather than silently ignored, but a clean committed revision is preferred for portfolio
evidence. Installed-package versions describe the executing environment; the `uv.lock` digest links
them to the committed dependency resolution.

The standalone manifest command only links supplied outputs; the `run-pipeline` command orchestrates
the stages that create them. Neither command uploads evidence to external storage.

Reproducibility evidence does not prove generalization, clinical validity, or medical utility.
Held-out benchmark evaluation remains intentionally protected, and host runtime/resource values are
expected to vary.
