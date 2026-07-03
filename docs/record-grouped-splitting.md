# Deterministic record-grouped splitting

## Scope

The supported split stage assigns complete record IDs to train, validation, and test partitions.
Every window from a record receives the same membership, preventing overlapping or adjacent windows
from that record from crossing partition boundaries.

This is a data-engineering integrity control, not evidence of clinical validity or model
generalization. Record grouping is not necessarily subject grouping: if multiple records represent
the same person, additional subject identity would be required to enforce subject-level separation.

## Versioned policy

`configs/splitting-v1.toml` defines:

| Setting | Value |
|---|---:|
| Strategy | `seeded-record-shuffle` |
| Seed | `2022` |
| Train ratio | `0.70` |
| Validation ratio | `0.15` |
| Test ratio | `0.15` |

The algorithm sorts unique record IDs, applies a local seeded shuffle, apportions record counts by
largest remainder, and guarantees three non-empty partitions. At least three records are required.
The implementation does not mutate global random state.

## Create a split manifest

Pass `--input` once for each generated window artifact:

```fish
uv run ecg-data split-windows \
  --split-config configs/splitting-v1.toml \
  --input data/interim/record-100-windows.npz \
  --input data/interim/record-101-windows.npz \
  --input data/interim/record-102-windows.npz \
  --output artifacts/split-manifest.json
```

Input NPZ files are opened with `allow_pickle=False`. Each must contain the row-level `record_ids`
and `target_values` written by the window stage plus matching mapping and window-configuration
identities. A record appearing in multiple input artifacts is rejected to prevent accidental
duplicate inputs.

## Manifest contract

The JSON manifest records:

- split policy name, version, strategy, and seed;
- mapping and window-configuration identities;
- source artifact paths;
- total record and window counts;
- record membership for each partition; and
- record, window, and target-value counts by partition.

The manifest is deterministic for identical configuration, metadata, and input ordering. Generated
manifests and window artifacts remain ignored by Git because they describe derived record-level
data.

## Limitations

The v1 policy balances record counts approximately according to configured ratios; it does not
stratify by target distribution. Target counts are reported so imbalance remains visible. Any
future stratified or subject-grouped policy requires a new versioned strategy and regression tests.
