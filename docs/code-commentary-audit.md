# Code commentary audit

This audit records the repository-wide internal-documentation pass completed for issue #149. Its
purpose is to make the supported Python implementation readable to reviewers who may not already
know the pipeline, its governance constraints, or the reasons behind individual control-flow
decisions.

## Standard

Every supported Python module, class, function, method, and nested callable must have a docstring.
Docstrings describe the callable's responsibility, inputs, outputs, failure behavior, and important
repository constraints where those details apply. Meaningful control-flow and resource-management
blocks also receive nearby `#` comments that explain both the operation and the reason for it.

The standard deliberately favors explicit guideposts over terse implementation. Tests are included
because they are executable specifications: their fixtures, fakes, setup phases, branch selection,
and assertions should be as reviewable as production code.

`scripts/check_code_commentary.py` enforces the structural floor. It parses Python with the standard
library AST and tokenizer, then rejects:

- modules without a module docstring;
- classes, functions, methods, asynchronous callables, or nested callables without docstrings;
- `if`, loop, `try`, context-manager, and pattern-matching blocks without an immediately preceding
  standalone comment; and
- module-level assignments without a leading explanatory comment.

The checker cannot judge whether prose is accurate or useful. Human review remains responsible for
semantic quality, especially around data lineage, partition isolation, deterministic behavior,
failure handling, and research-use limitations.

## Scope and baseline

The pre-audit baseline recorded during issue intake was 66 supported Python files, 15,081 lines, and
771 definitions. The completed audit contains 68 files, 22,920 lines, 789 definitions, 910 audited
control/resource blocks, and 2,011 standalone comment lines. The two additional files are the
checker itself and its tests.

The audit covers every `.py` file below `src/`, `scripts/`, and `tests/`. The preserved
`archive/original_2022/` tree is intentionally excluded: it is historical evidence and must not be
silently rewritten to meet current package standards.

## Audited inventory

### Package modules

- `src/ecg_anomaly_detection/__init__.py`
- `src/ecg_anomaly_detection/acquisition.py`
- `src/ecg_anomaly_detection/benchmark_approval.py`
- `src/ecg_anomaly_detection/benchmark_policy.py`
- `src/ecg_anomaly_detection/cli.py`
- `src/ecg_anomaly_detection/config.py`
- `src/ecg_anomaly_detection/dataset_index.py`
- `src/ecg_anomaly_detection/evaluation.py`
- `src/ecg_anomaly_detection/experiment_tracking.py`
- `src/ecg_anomaly_detection/held_out_config.py`
- `src/ecg_anomaly_detection/inventory.py`
- `src/ecg_anomaly_detection/labels.py`
- `src/ecg_anomaly_detection/local_execution.py`
- `src/ecg_anomaly_detection/notebook_quality.py`
- `src/ecg_anomaly_detection/pipeline.py`
- `src/ecg_anomaly_detection/progress.py`
- `src/ecg_anomaly_detection/records.py`
- `src/ecg_anomaly_detection/reproducibility.py`
- `src/ecg_anomaly_detection/run_manifest.py`
- `src/ecg_anomaly_detection/splitting.py`
- `src/ecg_anomaly_detection/training.py`
- `src/ecg_anomaly_detection/windows.py`

### Repository scripts

- `scripts/check_code_commentary.py`
- `scripts/check_held_out_trigger_safety.py`
- `scripts/detect_label_drift.py`
- `scripts/github/set_merged_project_status.py`
- `scripts/github/validate_project_metadata.py`
- `scripts/sync_github_labels.py`
- `scripts/validate_curated_notebooks.py`

### Integration tests

- `tests/integration/test_acquisition_cli.py`
- `tests/integration/test_benchmark_approval_cli.py`
- `tests/integration/test_cli.py`
- `tests/integration/test_dataset_index_cli.py`
- `tests/integration/test_local_execution_cli.py`
- `tests/integration/test_pipeline.py`
- `tests/integration/test_records.py`
- `tests/integration/test_run_manifest_cli.py`
- `tests/integration/test_splitting_cli.py`
- `tests/integration/test_threshold_sweep_cli.py`

### Script tests

- `tests/scripts/test_check_code_commentary.py`
- `tests/scripts/test_check_held_out_trigger_safety.py`
- `tests/scripts/test_detect_label_drift.py`
- `tests/scripts/test_set_merged_project_status.py`
- `tests/scripts/test_validate_curated_notebooks.py`
- `tests/scripts/test_validate_project_metadata.py`

### Unit tests

- `tests/unit/test_acquisition.py`
- `tests/unit/test_benchmark_approval.py`
- `tests/unit/test_benchmark_policy.py`
- `tests/unit/test_cli.py`
- `tests/unit/test_config.py`
- `tests/unit/test_dataset_index.py`
- `tests/unit/test_evaluation.py`
- `tests/unit/test_experiment_tracking.py`
- `tests/unit/test_held_out_config.py`
- `tests/unit/test_inventory.py`
- `tests/unit/test_labels.py`
- `tests/unit/test_local_execution.py`
- `tests/unit/test_notebook_quality.py`
- `tests/unit/test_package.py`
- `tests/unit/test_pipeline.py`
- `tests/unit/test_progress.py`
- `tests/unit/test_records.py`
- `tests/unit/test_reproducibility.py`
- `tests/unit/test_run_manifest.py`
- `tests/unit/test_splitting.py`
- `tests/unit/test_threshold_analysis.py`
- `tests/unit/test_training.py`
- `tests/unit/test_windows.py`

## Ongoing enforcement

The local `code-commentary` pre-commit hook runs the checker against the complete supported Python
tree on every invocation, even when a commit does not directly stage a Python file. Run it directly
with:

```fish
uv run python scripts/check_code_commentary.py
```

Passing the checker establishes structural coverage, not permission to add low-value boilerplate.
New code should explain domain meaning and design intent in the same terms used by the surrounding
module, and reviewers should reject comments that merely restate syntax or drift from behavior.
