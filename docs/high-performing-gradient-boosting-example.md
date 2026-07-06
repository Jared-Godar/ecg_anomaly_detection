# High-performing gradient boosting example notebook

## Purpose

The public example notebook at `notebooks/examples/high-performing-gradient-boosting-validation.ipynb` demonstrates how the governed pipeline artifacts can support a higher-performing local classical ML experiment without weakening the repository's evaluation boundaries.

The notebook is positioned as a reproducible local validation example. It is not a benchmark, not clinical evidence, not diagnostic software, not production ML, and not a protected-test evaluation.

## Relationship to other notebooks

| Notebook area | Role |
|---|---|
| `notebooks/narrative-walkthrough.ipynb` | Canonical package-backed explanation of the supported modernization workflow and evidence boundaries |
| `notebooks/examples/high-performing-gradient-boosting-validation.ipynb` | Public example that trains a fixed tuned classical ML model on generated train artifacts and evaluates validation artifacts only |
| `notebooks/local/` | Ignored local experimentation sandbox for disposable notebooks, checkpoints, predictions, and result exports |

The example notebook is intentionally not a general-purpose tuning workbench. It promotes a fixed configuration derived from local experimentation and leaves broader search, checkpointing, and optional model-family comparison in the ignored local sandbox.

## Prerequisites

Install the locked notebook and experiment dependency groups from the repository root:

```fish
uv sync --locked --group notebooks --group experiments
```

Generate the required local pipeline artifacts first:

```fish
uv run ecg-data run-pipeline \
  --repository-root . \
  --dataset-config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --split-config configs/splitting-v2.toml \
  --training-config configs/training-baseline-v1.toml \
  --evaluation-config configs/evaluation-baseline-v1.toml
```

The notebook expects these ignored generated artifacts to exist:

```text
data/processed/runs/<run-id>/dataset-index.json
data/interim/runs/<run-id>/windows/<record-id>.npz
artifacts/runs/<run-id>/evaluation/validation-metrics.json  # optional baseline comparison
```

## Model configuration

The notebook uses raw waveform windows only and trains this fixed estimator:

```python
HistGradientBoostingClassifier(
    learning_rate=0.015,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=0.1,
    max_iter=450,
    random_state=0,
)
```

Balanced sample weights are applied during fitting. The notebook does not use FFT augmentation, engineered feature expansion, LightGBM, XGBoost, serialized local checkpoints, or a hyperparameter search grid.

LightGBM remains an optional future local optimization candidate because adding it as the default public notebook path would complicate environment and dependency governance.

## Evaluation boundary

The notebook:

- resolves only `train` and `validation` descriptors from the model-ready dataset index;
- verifies indexed shard sizes and SHA-256 digests before loading waveform arrays;
- trains only on train shards;
- evaluates only on validation shards;
- shows metrics and a confusion matrix for the grouped validation partition;
- compares against repository baseline validation metrics when they exist for the same run; and
- leaves the protected `test` partition unopened, unscored, unsummarized, and unreported.

Observed values from this notebook should be described as local validation-only experiment results. They must not be described as benchmark evidence, clinical evidence, diagnostic evidence, deployment fitness, or final model performance.

## Artifact policy

The notebook does not write generated data, model artifacts, predictions, plots, or checkpoints. Generated local data and run evidence remain ignored under `data/`, `artifacts/`, and `reports/` unless a future reviewed change explicitly allowlists a small deterministic documentation asset.
