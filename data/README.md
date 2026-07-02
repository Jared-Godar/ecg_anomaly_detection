# Data directory

This directory defines local data stages; it does not contain source or derived ECG data in Git.

| Directory | Contract |
|---|---|
| `raw/` | Immutable files retrieved from the authoritative upstream dataset. |
| `external/` | Other third-party inputs used for reference or enrichment. |
| `interim/` | Reproducible intermediate transformations that can be rebuilt. |
| `processed/` | Model-ready outputs produced from versioned pipeline configuration. |

The contents of all four directories are ignored except for `.gitkeep` placeholders. Do not place credentials, identifying information, or manually edited source files here.

Pipeline stages must never overwrite raw inputs. Derived outputs should be replaceable from source
files, versioned configuration, and a recorded code revision. When acquisition is implemented, raw
files must be accompanied by a local manifest containing the upstream version, retrieval timestamp,
file inventory, and checksums.

See [data provenance](../docs/data-provenance.md) for the authoritative source, licensing, attribution, and privacy constraints.
See the [proposed pipeline design](../docs/pipeline-design.md) for lineage and validation contracts.
See [data integrity](../docs/data-integrity.md) for the implemented inventory and SHA-256 commands.
