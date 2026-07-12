# Benchmark governance

## Purpose and boundary

This policy governs any future use of the protected `test` partition. Test evaluation is disabled
by default and is not implemented by the supported evaluator. A future benchmark requires an
explicit, separately reviewed opt-in; this policy does not authorize execution by itself.

Governance provides controls for evaluation discipline and disclosure. It does not establish model
quality, generalization, clinical validity, or medical utility. This repository is a historical,
educational machine-learning project being modernized as a data-engineering case study. It is not
medical software.

Issue #73 completed one governed execution under this policy. Its aggregate result, required
disclosures, rerun record, and limitations are documented in
[Held-out evaluation](held-out-evaluation.md); generated evidence remains outside Git.

The machine-readable policy is `configs/benchmark-policy-v1.toml`. Validate it without reading data,
loading a model, or generating metrics:

```fish
uv run ecg-data validate-benchmark-policy --policy configs/benchmark-policy-v1.toml
```

`validate-benchmark-policy` prints a `[1/1] validate-benchmark-policy: starting` banner, its
existing completion message, and a `[1/1] validate-benchmark-policy: complete in MM:SS` banner (or
`failed after MM:SS`) to stdout, matching
[`run-pipeline`'s progress output](pipeline-orchestration.md#progress-output).

## Eligibility

A candidate is eligible only after training and model selection are complete. The candidate model,
dataset configuration, grouped split, training configuration, and evaluation configuration must be
immutable. Split integrity and configured quality checks must have passed. Reproducibility evidence
and an auditable run manifest must exist. A named benchmark owner must approve the specific purpose
before any protected-test shard is opened.

Validation results may guide development. Protected-test results may not be used for model,
threshold, feature, preprocessing, or configuration selection. No benchmark may be run merely to
inspect progress.

## Execution procedure

Any future implementation must fail closed and perform these steps in order:

1. Record approval, owner, candidate identity, purpose, and whether a prior attempt exists.
2. Verify the repository commit, dataset and split identity, configuration hashes,
   reproducibility-evidence reference, and run-manifest reference.
3. Use a separately reviewed command that is explicitly enabled for that approved execution.
4. Execute once against the protected partition without changing the frozen candidate.
5. Archive the result and all required evidence before publication or further development.

The current `evaluate_validation_from_index(...)` path remains validation-only. Changing
`SUPPORTED_PARTITION`, reusing the validation command for test data, or manually opening test shards
is not an approved benchmark procedure.

### Implemented: approval and lineage verification (steps 1-2) and held-out execution (steps 3-5)

`ecg-data record-benchmark-approval` implements steps 1 and 2 above. Given a human-authored
approval-input record, a validated `BenchmarkPolicy`, and an existing `RunManifest`, it verifies
that the policy is still disabled by default, that the approval's candidate identity matches the
run manifest, and that every entry in `policy.required_lineage_references` resolves to a concrete,
present value on that manifest. It fails closed, listing exactly which references are missing, and
writes an `ApprovalRecord` as audit evidence that the gate was checked — not a benchmark result and
not authorization to execute anything. It never opens, reads, or scores a `test`-partition shard.

```fish
uv run ecg-data record-benchmark-approval \
  --repository-root . \
  --policy configs/benchmark-policy-v1.toml \
  --run-manifest artifacts/runs/<run-id>/run-manifest.json \
  --approval <path-to-approval-input>.toml \
  --output artifacts/runs/<run-id>/benchmark_approval.json
```

`record-benchmark-approval` prints a `[1/1] record-benchmark-approval: starting` banner, its
existing completion message, and a `[1/1] record-benchmark-approval: complete in MM:SS` banner (or
`failed after MM:SS`) to stdout, matching
[`run-pipeline`'s progress output](pipeline-orchestration.md#progress-output).

Steps 3-5 — a separately reviewed execution command, the single run against the protected
partition, and archival before publication — are implemented by `ecg-data evaluate-held-out`
(see below).

### Implemented: held-out execution, archival, and disclosure (steps 3-5)

`ecg-data evaluate-held-out` implements steps 3-5 above. It loads a verified `ApprovalRecord`
(written by `record-benchmark-approval`), validates the `HeldOutExecutionConfig` and
`BenchmarkPolicy`, confirms the approval record's `candidate_run_id` matches the model artifact's
run path, opens only the `test`-partition shards from the dataset index, scores the frozen model
exactly once, and writes two artifacts atomically:

- `artifacts/runs/<run-id>/held-out-evaluation/held-out-metrics.json` — confusion matrix,
  accuracy, per-class precision/recall/F1, and complete artifact lineage (index, shard, and model
  digests bound to the approval record reference).
- `artifacts/runs/<run-id>/held-out-evaluation/held-out-disclosure.json` — required disclosure
  evidence per the policy's `required_disclosures`, `required_limitations`, `prohibited_claims`,
  and `required_archival_records` fields.

It fails closed at every step: a config without `execution_enabled = true`, a policy without
`test_evaluation_enabled = true`, a missing or mismatched approval record, any digest mismatch,
or any lineage failure aborts before any test shard is opened. If the disclosure write fails after
the metrics file is written, the metrics file is removed rather than leaving an unapproved result
on disk.

The CI workflow (`.github/workflows/held-out-evaluation.yml`) triggers only on `workflow_dispatch`,
never on `push` or `pull_request`, satisfying `scripts/check_held_out_trigger_safety.py`.

```fish
uv run ecg-data evaluate-held-out \
  --repository-root . \
  --dataset-index artifacts/datasets/mitdb/1.0.0/dataset-index.json \
  --model artifacts/runs/<run-id>/training/model.json \
  --training-metadata artifacts/runs/<run-id>/training/training-metadata.json \
  --held-out-config configs/evaluation-heldout.toml \
  --policy configs/benchmark-policy-v1.toml \
  --approval-record artifacts/runs/<run-id>/benchmark_approval.json \
  --output artifacts/runs/<run-id>/held-out-evaluation/held-out-metrics.json \
  --disclosure artifacts/runs/<run-id>/held-out-evaluation/held-out-disclosure.json
```

`evaluate-held-out` prints a `[1/1] evaluate-held-out: starting` banner, its completion message,
and a `[1/1] evaluate-held-out: complete in MM:SS` banner (or `failed after MM:SS`) to stdout.

### Disabled-by-default execution controls

Two additional controls support `ecg-data evaluate-held-out`. Neither opens the `test` partition
independently nor changes
`evaluate_validation_from_index(...)`'s validation-only behavior:

- `configs/evaluation-heldout.toml`, loaded by `load_held_out_config(...)` in
  `held_out_config.py`, is a disabled-by-default configuration schema for a future held-out
  execution. Routine loading rejects `execution.execution_enabled = true`; only the separately
  reviewed execution command opts into loading an enabled copy, and it requires
  `execution.requires_recorded_approval = true`.
- `scripts/check_held_out_trigger_safety.py` is a read-only governance-as-code check, run as a
  job in `.github/workflows/repository-hygiene.yml`, that parses every workflow file and fails if
  any workflow whose name matches a held-out/benchmark-execution naming convention could trigger
  on a routine `push` or `pull_request` event instead of only `workflow_dispatch` and/or a `push`
  restricted to `release-*` tags. The manual-only held-out workflow is covered by this guard.

These controls are preconditions for steps 3-5, not authorization by themselves. Execution still
requires an explicitly enabled reviewed configuration and policy copy plus a named benchmark
owner's recorded approval.

## Reruns

A rerun requires new recorded approval and is limited to a documented infrastructure failure before
result inspection or a verified implementation defect that invalidates the complete prior result.
The original attempt and reason must remain archived. A disappointing or inconclusive result, a
changed model, or post-result model selection is not a rerun; it is a new benchmark candidate and
requires a newly reviewed protocol. Repeated access must never become an informal validation loop.

## Publication and disclosure

Every publication, README claim, portfolio statement, report, presentation, or model card using a
future protected-test result must disclose:

- repository commit hash;
- dataset configuration hash;
- split identity;
- training configuration hash;
- evaluation configuration hash;
- reproducibility evidence reference;
- hardware summary;
- runtime summary;
- assumptions; and
- limitations.

Limitations must explicitly cover the dataset, annotations, class imbalance, binary mapping, split
methodology, historical age and context of the dataset, lack of clinical validation, and lack of
medical utility. Results must be described as bounded evidence for one frozen candidate and one
grouped split, not as proof of broader performance.

Prohibited claims include assertions that governance or a benchmark establishes model quality,
generalization, clinical validity, medical utility, diagnostic usefulness, healthcare-AI readiness,
medical-device suitability, or production healthcare readiness.

## Archival

Archive the approval record, immutable lineage references, benchmark results, publication
disclosures, hardware and runtime evidence, and complete rerun history together. History is
append-only. Invalidated or superseded results remain retained with their status and explanation;
they are not overwritten or silently removed. Generated benchmark artifacts must remain outside Git
unless a separate review approves a safe, non-dataset disclosure artifact.

## Known evidence limitations

The MIT-BIH Arrhythmia Database is a small historical dataset with constraints in cohort coverage,
recording context, and contemporary representativeness. Its annotations inherit expert-policy and
source limitations. The binary mapping collapses heterogeneous rhythms and excludes some symbols;
class imbalance can make aggregate metrics misleading. Subject-grouped splitting reduces direct
subject leakage but does not prove performance on new populations, sites, devices, or clinical
settings. No pipeline or model in this repository has clinical validation or demonstrated medical
utility.
