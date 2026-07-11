# Validation-only threshold sweep analysis

## Scope

The supported baseline evaluator (`docs/baseline-evaluation.md`) always predicts a hard class per
window using `np.argmin` over centroid distances. The threshold sweep is a separate, optional
reporting stage that scores the same frozen baseline against the same `validation` partition, but
also reports how coverage and accuracy trade off as a per-window confidence margin is raised. It
does not change the frozen baseline, its training, or the existing evaluator's metrics contract.

Like the rest of validation-only evaluation, this is exploratory pipeline evidence, not a
benchmark and not evidence of clinical or diagnostic performance. It is never run against the
protected `test` partition; `evaluation.py`'s `SUPPORTED_PARTITION` constant governs both this
stage and the existing evaluator, and `threshold_sweep.partition` is rejected unless it is
`"validation"`.

## What the margin is — and is not

For each window, the evaluator already computes the squared distance from the projected window to
every class centroid before collapsing it to a hard label. The margin is the gap between the
smallest and second-smallest of those distances: a larger margin means the two nearest centroids
were more clearly separated for that window.

The margin is a **raw squared distance in the model's projected feature space**. It is:

- **not** a probability or confidence score,
- **not** calibrated or bounded,
- **not** comparable across different projection widths or centroid geometries,
- **not** a basis for ROC, AUC, or calibration analysis (see `MODEL_CARD.md`'s "Known
  limitations": the baseline exposes no ranked score or predicted probability to score against;
  the margin does not change that).

A higher margin only ever means "further from the runner-up class in this specific model's fixed
projection," nothing more.

## Configuration and CLI

`configs/threshold-sweep-v1.toml` fixes `schema_version = 1`, the sweep name and version, the
`validation` partition, the zero-division convention, and a non-empty, strictly increasing
`thresholds` array. This is not part of the orchestrated `run-pipeline` sequence — it is a
standalone, optional stage, following the same optional-subcommand precedent as
`record-benchmark-approval`:

```fish
uv run ecg-data evaluate-threshold-sweep \
  --repository-root . \
  --dataset-index data/processed/runs/<run-id>/dataset-index.json \
  --model artifacts/runs/<run-id>/training/model.json \
  --training-metadata artifacts/runs/<run-id>/training/training-metadata.json \
  --config configs/threshold-sweep-v1.toml \
  --output artifacts/runs/<run-id>/evaluation/threshold-sweep-metrics.json
```

The command reuses the existing evaluator's isolation contract in full: training metadata must
exactly identify the current dataset index and frozen model, every resolved validation shard must
match its indexed digest, and the deterministic JSON output is only ever written after all
verification, prediction, and metric calculation succeeds in memory.

`evaluate-threshold-sweep` prints a `[1/1] evaluate-threshold-sweep: starting` banner, its existing
completion message, and a `[1/1] evaluate-threshold-sweep: complete in MM:SS` banner (or
`failed after MM:SS`) to stdout, matching
[`run-pipeline`'s progress output](pipeline-orchestration.md#progress-output).

## Output

For each configured threshold, `threshold-sweep-metrics.json` reports the covered-window count
(windows whose margin is at or above that threshold) and macro-averaged precision, recall, and F1
computed only over those covered windows — the same undefined-metric convention
(`zero_division`) as the existing evaluator. Record lineage and SHA-256 evidence for the dataset
index, frozen model, and validation shards are included, matching `validation-metrics.json`'s
evidence style. This is a separate output artifact; it does not add fields to or otherwise change
`ValidationMetrics`'s existing schema or consumers.

No plotting dependency or figure-generation step is included: the JSON sweep output is sufficient
input for a notebook or downstream script to plot a coverage/accuracy curve, and adding a plotting
dependency to the core package for an optional artifact was judged unnecessary scope.
