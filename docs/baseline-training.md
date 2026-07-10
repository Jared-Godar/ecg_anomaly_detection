# Deterministic baseline training

## Scope and boundary

The baseline stage fits one seeded random-projection nearest-centroid classifier. It consumes only
the shards listed in the dataset index's `train` partition. Validation and test shard paths are not
passed to the loader, opened, transformed, or used during fitting. The projection is scaled by
`1 / sqrt(projection_components)`, the standard Johnson-Lindenstrauss normalization that keeps it
approximately distance-preserving; see the
[Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) for
the mathematical basis.

This stage persists fitted parameters and training lineage. It intentionally does not calculate or
report held-out metrics, confusion matrices, benchmark comparisons, or claims about performance.
The output is for educational pipeline development, not clinical or diagnostic use.

## Configuration and outputs

The committed `configs/training-baseline-v1.toml` fixes the estimator, seed, and projection width.
An orchestrated run writes ignored artifacts to:

```text
artifacts/runs/<run-id>/training/
├── model.json
└── training-metadata.json
```

The metadata identifies the training records and shards, their SHA-256 digests, the dataset index,
configuration identity, seed, class counts, and model digest. The final run manifest independently
hashes the model, metadata, and training configuration.

## Failure behavior

Training rejects malformed indexes, missing or modified training shards, invalid array shapes,
non-finite values, lineage or count mismatches, fewer than two training classes, symbolic links,
and paths outside their managed repository zones. Model and metadata files are create-only. If
metadata persistence fails after model persistence, the incomplete model is removed.

## Runtime and resources

Fitting is local, single-process, and CPU-only. Training shards are read sequentially and then
concatenated in memory. Peak memory is therefore approximately the complete training window matrix
plus its projected representation and model parameters. With the current MIT-BIH window geometry,
plan for several hundred megabytes of available memory; exact use depends on the emitted window
count and NumPy representation.

Runtime scales approximately with `training_windows × window_samples × projection_components`.
No GPU, network service, or multiprocessing runtime is required. The JSON model is expected to be
well below one megabyte at the configured 32 projection components; source and derived data remain
the dominant disk usage.
