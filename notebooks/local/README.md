# Local notebook sandbox

This directory is an ignored workspace for local investigation, visualization, troubleshooting,
and model-development experiments. This README is the only supported, tracked file in the
directory.

## Boundary

Everything created here is local and disposable, including:

- notebooks and checkpoints;
- machine-local configuration;
- fitted models and serialized predictions;
- intermediate and final result exports; and
- experiment logs and run metadata.

These artifacts are not part of the supported package, canonical narrative notebook, run
manifests, reproducibility evidence, protected-test evidence, release artifacts, or portfolio
benchmarks. Do not commit source data, derived record-level data, credentials, absolute local
paths, or outputs containing record-level observations.

The repository's research-and-education-only limitation applies to all work performed here. Local
results must not be presented as clinical, diagnostic, monitoring, or treatment evidence.

## Environment

From the repository root, install the locked notebook kernel and optional experimentation tools:

```fish
uv sync --locked --group notebooks --group experiments
```

Verify the kernel, then select the environment's Python interpreter as the notebook kernel in an
editor:

```fish
uv run --group notebooks python -m ipykernel --version
```

Register the discoverable project kernel and verify it points to the current checkout using the
[environment reproducibility guide](../../docs/environment-reproducibility.md). Use the notebook
group without `experiments` when only the supported narrative notebook infrastructure is needed.

The optional groups are deliberately separate from the default development environment. Their
presence does not make local experiments part of the supported pipeline.

## Static quality checks

Validate every local `.ipynb` file from the repository root:

```fish
uv run --group notebooks ecg-data check-local-notebooks
```

The command parses notebook JSON, validates the nbformat schema, and reports cell counts, saved
outputs and their serialized size, execution counts, trust metadata, stale kernel names, and
absolute machine-local paths. Add `--json` for deterministic machine-readable output. Default
discovery recursively includes `.ipynb` files in this ignored directory while excluding Jupyter
checkpoint directories.

Formatting and output removal are explicit local mutations:

```fish
uv run --group notebooks ecg-data check-local-notebooks --format
uv run --group notebooks ecg-data check-local-notebooks --strip-outputs
```

`--format` applies nbformat's canonical JSON serialization. `--strip-outputs` clears code-cell
outputs and execution counts and also canonicalizes the file. Both operations are deterministic;
review the affected notebooks locally because ignored files do not appear in the ordinary Git
diff. Use `--include-narrative` only when intentionally checking the tracked walkthrough as well.

This workflow never executes cells, imports notebook code, trains models, runs pipeline stages,
evaluates data, or creates reproducibility or benchmark evidence. It does not make local notebooks
supported repository artifacts and is not a notebook execution workflow for CI or pytest.

If validation reports `invalid-notebook`, open the file in a notebook editor and repair or restore
its JSON before using formatting or stripping. Hygiene warnings do not fail validation; structural
or repository-boundary errors do.

## Cleaning up local pipeline runs

Iterating against `run-pipeline` output from a notebook accumulates `artifacts/runs/<run-id>/` and
its companion `data/interim/` and `data/processed/` directories. List and reclaim them explicitly
rather than editing generated files by hand:

```fish
uv run ecg-data list-runs --repository-root .
uv run ecg-data purge-run --repository-root . --run-id <run-id> --dry-run
```

See [local artifact lifecycle helpers](../../docs/pipeline-orchestration.md#local-artifact-lifecycle-helpers)
for the full command reference and the guarantee that these commands never touch the dataset
acquisition baseline. For a full local reset instead of removing a single run, use
`clean-local-pipeline-state.fish` if a Step 0 preflight check writes one.

## Checkpoint and resume long-running experiments

`ecg_anomaly_detection.experiment_tracking.ExperimentTracker` replaces the hand-rolled
`atomic_write_csv`/`atomic_write_pickle`/`load_pickle` checkpoint helpers that local
hyperparameter-search notebooks have previously had to define inline. It checkpoints one candidate
at a time, immediately after it completes, so an interruption loses at most the in-progress
candidate:

```python
from pathlib import Path
from ecg_anomaly_detection.experiment_tracking import ExperimentTracker

checkpoint_dir = Path("notebooks/local/model-experiments/outputs/gradient-boosting-v4")
candidate_ids = [spec["experiment_id"] for spec in model_specs]
tracker = ExperimentTracker(checkpoint_dir, candidate_ids)

for spec in model_specs:
    if tracker.is_completed(spec["experiment_id"]):
        continue  # already checkpointed in a prior, interrupted run
    with tracker.track(spec["experiment_id"]) as recorder:
        model = fit_candidate(spec)
        predictions = model.predict(x_validation_model)
        recorder.set_metrics(compute_metrics(y_validation_model, predictions))
        recorder.set_predictions(predictions)
    print(tracker.progress())  # completed/remaining counts, elapsed time, estimated remaining

final_results_path = tracker.finalize("macro_f1")  # raises if any candidate is still incomplete
```

Re-running the same cell with the same `checkpoint_dir` and candidate list resumes automatically:
already-checkpointed candidates are skipped via `is_completed()`, and `tracker.load_result()` /
`tracker.load_predictions()` recover a completed candidate's metrics and predictions without
refitting. `finalize()` writes a `final-results.json` sorted by the metric you name, and refuses to
run while any candidate remains incomplete rather than silently reporting a partial leaderboard as
final.

This is unrelated to `ecg_anomaly_detection.local_execution`'s `list-runs`/`purge-run` above, which
manage governed `run_pipeline()` output, not ad hoc experiment loops. Checkpoint directories here
are local, disposable, and excluded from Git like everything else in this directory — never run
manifests, benchmark artifacts, model card evidence, or reproducibility evidence bundles.

## Working practices

- Use repository-relative paths and package APIs where practical.
- Keep the indexed test partition unopened unless separately authorized under evaluation policy.
- Save resumable local checkpoints after each completed candidate for long-running experiments.
- Treat local outputs as replaceable; preserve durable conclusions in reviewed documentation or
  tested package code.
- Before proposing a supported artifact, remove saved outputs and hidden state, verify provenance,
  document evaluation limitations, and move reusable logic into tested package modules.
