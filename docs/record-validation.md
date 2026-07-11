# WFDB record ingestion and validation

## Scope

The supported package loads one local WFDB signal and its `.atr` reference annotations, preserves
source metadata, applies data-contract checks, and writes a machine-readable validation report.
It does not transform labels, extract beat windows, split records, train a model, or make clinical
claims.

The command operates only on ignored local files. CI creates a small synthetic WFDB record at test
time and does not download MIT-BIH data.

## Validate one record

After acquisition or inventory has checked the files against the committed expected metadata, use
the [local integrity workflow](data-integrity.md) and validate a configured record:

```fish
uv run ecg-data validate-record \
  --config configs/mitdb-v1.0.0.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --record-id 100 \
  --output artifacts/record-100-validation.json
```

The command returns a nonzero status when required files cannot be loaded or a validation rule
fails. Validation reports belong under `artifacts/` and remain excluded from Git.

`validate-record` prints a `[1/1] validate-record: starting` banner, its existing completion
message, and a `[1/1] validate-record: complete in MM:SS` banner (or `failed after MM:SS`) to
stdout, matching [`run-pipeline`'s progress output](pipeline-orchestration.md#progress-output).
This is purely observational and never changes the command's exit code or the written report.

## Implemented checks

- record ID is present in the versioned dataset configuration;
- required `.hea`, `.dat`, and `.atr` paths are regular files rather than symlinks;
- sample rate matches the configured 360 Hz contract;
- physical signals form a non-empty samples-by-channels array;
- every physical signal value is finite;
- channel names and units are present and match the signal width;
- channel names are unique;
- annotation locations and symbols have equal lengths;
- annotation sample indices are nonnegative, ordered, and inside the signal boundary; and
- signal and annotation objects retain the same record ID.

## Typed outputs

`SignalRecord` retains the record ID, sample rate, immutable physical-signal array, channel names,
and units. `AnnotationSet` retains the record ID, immutable sample-index array, and original WFDB
symbols. No project-specific label mapping occurs during ingestion.

The JSON report records dataset and record identity, sample and channel counts, channel metadata,
annotation counts by original symbol, and the names of completed checks. It contains operational
metadata only; it is not a clinical validation report.

## Current boundaries

Validation confirms structural expectations needed by the next pipeline stage. Source-file byte
integrity is enforced by acquisition/inventory, not by this record command. Validation does not
assess signal quality, resolve channel preference, authenticate the upstream publisher, or establish
model fitness. Those concerns require separate configuration, evidence, and tests.

The next supported stage applies the [versioned annotation mapping](annotation-mapping.md) without
altering the original symbols stored by ingestion.
