# Deterministic subject-aware splitting

## Scope

Split schema v2 assigns complete subject IDs to train, validation, and test partitions. Every record
and every window belonging to a subject follows that assignment. Record grouping alone prevented one
record from crossing partitions, but it could still separate records 201 and 202, which the upstream
database directory identifies as originating from the same source tape.

This is a reproducibility and leakage-control boundary for an educational project. It is not evidence
of clinical validity or generalization to new populations.

## Versioned policy and metadata

`configs/splitting-v2.toml` uses schema version 2 and `seeded-subject-shuffle`. Its
`[record_subjects]` table explicitly maps every configured record to an opaque subject ID. Records
201 and 202 share one ID; other MIT-BIH records have distinct IDs. The splitter requires the mapping
to cover the input records exactly.

The algorithm sorts unique subject IDs, applies a local seeded shuffle, apportions subject counts by
largest remainder, and guarantees three non-empty partitions. It does not mutate global random
state. Identical configuration and window metadata produce identical membership.

```fish
uv run ecg-data split-windows \
  --split-config configs/splitting-v2.toml \
  --input data/interim/record-100-windows.npz \
  --input data/interim/record-101-windows.npz \
  --input data/interim/record-102-windows.npz \
  --output artifacts/split-manifest.json \
  --quality-output artifacts/split_quality_summary.json
```

Input NPZ files retain row-level record IDs, target values, mapping identity, and window
configuration identity. A record appearing in multiple artifacts is rejected.

## Manifest and validation contract

Schema v2 records the split policy, source artifacts, total subject/record/window counts, and these
fields per partition:

- `subject_ids` and `subject_count`;
- `record_ids`, `record_subjects`, and `record_count`; and
- `window_count` and `target_value_counts`.

Validation rejects subject overlap, record overlap, incomplete coverage, inconsistent counts, and
record-to-subject mappings that disagree with partition membership. Per-record shards preserve
window and annotation lineage; the model-ready index adds `subject_id` to every shard descriptor.

Subject separation is necessary but insufficient: a leakage-free split may still have too few
subjects, records, windows, positive examples, or represented classes to support a meaningful
evaluation. The separate `split_quality_summary.json` reports those conditions and applies the
configured acceptance policy. See [Split quality reporting](split-quality-reporting.md).

## Migration from v1

Replace `configs/splitting-v1.toml` with `configs/splitting-v2.toml`. A custom dataset must add an
exhaustive `[record_subjects]` table, change the strategy from `seeded-record-shuffle` to
`seeded-subject-shuffle`, and bump `schema_version` to 2. Schema v1 remains readable for historical
manifests, but new supported runs use v2.

## Evaluation limitations

The policy approximately balances subject counts, not targets or windows. Validation metrics remain
exploratory. This change intentionally does not open, score, or publish results for test shards; a
held-out benchmark requires a separate, explicitly reviewed evaluation stage.
