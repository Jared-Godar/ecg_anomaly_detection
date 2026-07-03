# Deterministic baseline validation evaluation

## Scope

The supported evaluator loads the persisted deterministic baseline without refitting it and scores
only the `validation` partition named by the dataset index. It does not resolve, open, score,
summarize, or report indexed test shards. Validation results are exploratory pipeline evidence, not
a final benchmark and not evidence of clinical or diagnostic performance.

The committed `configs/evaluation-baseline-v1.toml` fixes the evaluator, partition, configuration
version, and zero-division value. Version 1 defines undefined precision, recall, or F1 as `0.0`.

## Output

Each orchestrated run writes:

```text
artifacts/runs/<run-id>/evaluation/validation-metrics.json
```

The deterministic JSON contains class order, confusion matrix, accuracy, per-class precision,
recall, F1 and support, macro averages, record and window counts, and SHA-256 evidence for the
dataset index, frozen model, and validation shards. The run manifest also records the metrics
artifact's path, size, and digest.

## Validation and failure behavior

Before opening a shard, evaluation verifies that training metadata exactly identifies the current
dataset index and frozen model. It then validates the model schema, feature width, class IDs, and
parameter shapes. Only descriptors under `partitions.validation.shards` are resolved. Every selected
shard must match its indexed size and digest and provide finite compatible windows, integer known
labels, correct record lineage, and aggregate counts matching the index.

All verification, prediction, and metric calculation completes in memory before the metrics output
is created. Digest mismatches, unknown labels, malformed models, incompatible datasets, invalid
shards, and metric failures therefore leave no validation metrics artifact.
