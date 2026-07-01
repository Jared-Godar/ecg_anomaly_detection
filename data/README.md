# Data directory

This directory defines local data stages; it does not contain source or derived ECG data in Git.

| Directory | Contract |
|---|---|
| `raw/` | Immutable files retrieved from the authoritative upstream dataset. |
| `external/` | Other third-party inputs used for reference or enrichment. |
| `interim/` | Reproducible intermediate transformations that can be rebuilt. |
| `processed/` | Model-ready outputs produced from versioned pipeline configuration. |

The contents of all four directories are ignored except for `.gitkeep` placeholders. Do not place credentials, identifying information, or manually edited source files here.

See [data provenance](../docs/data-provenance.md) for the authoritative source, licensing, attribution, and privacy constraints.
