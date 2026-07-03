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
  --config configs/splitting-v1.toml \
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
- configuration, evidence, and artifact paths, sizes, and checksums.

Absolute local paths and file contents are not serialized. The output belongs under `artifacts/`
and remains ignored because split membership and derived artifact metadata are record-level data.

## Reproducibility interpretation

The manifest makes the inputs to a run inspectable and detects later file changes. A dirty Git state
is recorded rather than silently ignored, but a clean committed revision is preferred for portfolio
evidence. Installed-package versions describe the executing environment; the `uv.lock` digest links
them to the committed dependency resolution.

The current command links existing stage outputs. It does not orchestrate those stages, retrieve the
dataset, train a model, calculate evaluation metrics, or upload evidence to external storage.
