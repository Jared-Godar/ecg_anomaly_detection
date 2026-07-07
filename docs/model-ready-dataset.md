# Model-ready grouped dataset index

## Scope

The supported indexing stage converts the grouped split manifest and per-record window artifacts
into a model-ready dataset contract. It does not concatenate all windows into one in-memory array.
Instead, each train, validation, and test partition contains an ordered list of immutable
record-level NPZ shards with counts, repository-relative paths, sizes, and SHA-256 digests.

This design preserves subject-aware membership and record-level lineage, supports lazy one-record-
at-a-time loading, and avoids duplicating a potentially multi-gigabyte window matrix. Each shard
descriptor includes both `subject_id` and `record_id`; its NPZ arrays retain window-level record,
annotation, and center-sample lineage.

## Pipeline output

The orchestrated workflow writes:

```text
data/processed/runs/<run-id>/dataset-index.json
```

The index references the corresponding ignored shards under:

```text
data/interim/runs/<run-id>/windows/<record-id>.npz
```

The final run manifest hashes both the dataset index and every referenced shard.

## Standalone command

The stage can also be run independently by repeating `--input` for every record shard:

```fish
set -l run_id YOUR_RUN_UUID
set -l shard_arguments

for shard in data/interim/runs/$run_id/windows/*.npz
  set --append shard_arguments --input $shard
end

uv run ecg-data index-dataset \
  --repository-root . \
  --split-manifest artifacts/runs/$run_id/split.json \
  $shard_arguments \
  --output data/processed/runs/$run_id/dataset-index.json
```

The loop adds one `--input` argument for each shard. The orchestration command builds that list
automatically.

## Validation contract

Before writing the index, the stage verifies:

- the split manifest schema, three partition names, counts, target keys, and subject/record disjointness;
- exact agreement between split membership and supplied record shards;
- one and only one record per shard;
- non-pickle NPZ loading and required lineage fields;
- finite, two-dimensional floating-point window arrays;
- equal row counts across windows, targets, source symbols, and center indices;
- consistent sample rate, channel, window width, mapping version, and window configuration;
- partition record, window, and target counts against the split manifest; and
- repository boundaries for split, interim shard, and processed index paths.

The output is created once and is not overwritten.

## Training boundary

The baseline training stage opens only train shards. The separate evaluation stage opens only
validation shards. Test shard descriptors remain reserved and are not resolved or summarized.

The index itself does not train a model, calculate metrics, establish broader population
generalization, or support clinical use. Its subject-independence guarantee is limited to the
explicit record-to-subject metadata supplied to split schema v2.

Indexing runs only after configured split-quality checks pass. The quality summary remains separate
run evidence; it is not a source of model features and does not expose test metrics.

## Shard channel identity validation

The model-ready dataset index validates that all referenced window shards share one geometry and configuration identity. Channel identity is part of that contract.

The index rejects mixed resolved channel names across shards. For example, shards extracted from `MLII` and `V5` are not considered interchangeable, even if they were produced with the same window length, mapping configuration, and split manifest.

When mixed channel identities are detected, the index reports:

- configured channel selector, when available;
- observed channel names and record counts;
- affected records; and
- guidance that cleanup is not the usual fix.

The expected fix is to use a configuration that satisfies the channel contract, such as name-based selection with:

```toml
channel_name = "MLII"
```

Alternatively, records that cannot satisfy the configured channel contract must be explicitly excluded or handled by a documented fallback strategy. The repository does not silently coerce or merge channel identities.
