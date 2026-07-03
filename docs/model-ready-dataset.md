# Model-ready grouped dataset index

## Scope

The supported indexing stage converts the grouped split manifest and per-record window artifacts
into a model-ready dataset contract. It does not concatenate all windows into one in-memory array.
Instead, each train, validation, and test partition contains an ordered list of immutable
record-level NPZ shards with counts, repository-relative paths, sizes, and SHA-256 digests.

This design preserves the record boundary established by splitting, supports lazy one-record-at-a-
time loading, and avoids duplicating a potentially multi-gigabyte window matrix.

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

- the split manifest schema, three partition names, counts, target keys, and record disjointness;
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

The baseline training stage consumes partition membership from this index and opens only train
shards. Validation and test remain reserved for a future, separate evaluation stage.

The index itself does not train a model, establish subject-level independence, calculate metrics,
or support clinical use. Record grouping remains distinct from patient grouping when subject identity
is unavailable.
