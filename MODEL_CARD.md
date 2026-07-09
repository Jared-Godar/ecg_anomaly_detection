# Model card: deterministic ECG baseline

![Technical banner for the model card, showing validation-only context and responsible ML governance motifs.](docs/assets/ecg-model-card-banner.png)

## Card scope and status

This model card documents the supported deterministic baseline and the pipeline that produces it.
It is a transparency and reproducibility record for a historical educational project under
modernization, not a performance claim. It describes repository state at the checked-out revision;
generated models, datasets, and metrics are deliberately not committed. For the tagged repository
version this corresponds to, see the [changelog](CHANGELOG.md) and [GitHub
releases](https://github.com/Jared-Godar/ecg_anomaly_detection/releases).

The repository is a modernization, reproducibility, data-engineering, operational-maturity, and
responsible machine-learning documentation case study. It is not medical or clinical software,
healthcare AI, a diagnostic or monitoring system, a production machine-learning service, or
regulatory evidence. Neither this card nor the documented controls establish clinical validity,
medical effectiveness, deployment readiness, diagnostic suitability, or regulatory suitability.

## Model summary

| Property | Supported baseline |
|---|---|
| Task | Project-specific binary classification of beat-centered ECG windows |
| Estimator | Seeded random-projection nearest-centroid classifier |
| Inputs | Six-second, single-channel windows with 2,160 samples at 360 Hz |
| Outputs | Integer class `0` (`reference_normal`) or `1` (`selected_other`) |
| Training data | Only shards indexed in the `train` partition |
| Evaluation data | Only shards indexed in the `validation` partition |
| Test status | Indexed and protected; not opened or scored by the supported evaluator |
| Configuration | `configs/training-baseline-v1.toml` and `configs/evaluation-baseline-v1.toml` |
| Runtime | Local, CPU-only, single-process |

The baseline is intentionally simple. Its role is to exercise deterministic training, artifact
lineage, partition isolation, metric calculation, and reproducibility evidence. It is not presented
as an optimized model or as the best method for ECG analysis.

## Repository objectives

- Replace notebook-bound steps with versioned configuration, tested package boundaries, explicit
  lineage, and auditable run evidence.
- Demonstrate acquisition integrity, schema validation, subject-aware preparation, deterministic
  execution, evaluation discipline, and artifact governance.
- Make supported behavior reproducible from a locked environment while keeping source data,
  patient-level derivatives, trained models, and generated evidence outside Git.
- Preserve the original 2022 work as historical evidence while disclosing its evaluation defects.
- Provide a portfolio case study in responsible engineering and documentation, without implying
  healthcare or production readiness.

## Dataset and labels

The pipeline uses version 1.0.0 of the public
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), distributed by PhysioNet
under the Open Data Commons Attribution License v1.0. The database contains 48 half-hour,
two-channel ambulatory ECG excerpts from 47 subjects, sampled at 360 Hz, with reference beat
annotations. Cite the dataset using DOI [10.13026/C2F305](https://doi.org/10.13026/C2F305) and cite
PhysioNet as specified in [data provenance](docs/data-provenance.md).

Repository-reviewed expected file sizes and SHA-256 digests in `configs/mitdb-v1.0.0.toml` provide
a local source-integrity control. They do not replace the upstream landing page, DOI, license, or
citations as provenance, and they do not establish the suitability of the data for another purpose.

The versioned mapping in `configs/annotation-map-v1.toml` preserves the historical project's binary
target:

| Value | Project label | Upstream symbols | Interpretation constraint |
|---:|---|---|---|
| 0 | `reference_normal` | `N` | An upstream reference annotation, not a finding about a person |
| 1 | `selected_other` | `L R V / A f F j a E J e S` | A heterogeneous project grouping, not a disease or diagnosis |

Twenty-four other configured symbols are excluded and counted. Unknown symbols fail closed. This
binary mapping discards annotation detail, combines unlike beat types, and inherits limitations in
the historical annotations and their expert policy. It does not represent a complete taxonomy or
support clinical interpretation. See [annotation mapping](docs/annotation-mapping.md).

### Dataset limitations

- The dataset is small, historical, and not representative of contemporary populations, sites,
  devices, acquisition practices, or deployment conditions.
- Public availability does not make the records project-owned or appropriate for unrestricted
  redistribution. Re-identification attempts or linkage with identifying data are prohibited.
- The pipeline uses a single named channel (`MLII`, resolved per record by name — see
  [window extraction](docs/window-extraction.md#channel-identity-contract)) without a
  comparative channel-selection or signal-quality study.
- Adjacent beat-centered windows can overlap; grouping prevents them from crossing partitions only
  because every record follows its subject assignment.
- Class prevalence and coverage can vary by subject partition. Quality diagnostics expose selected
  conditions but cannot establish representativeness or generalization.

## Supported pipeline

```text
acquisition
→ validation
→ annotation mapping
→ window extraction
→ subject-aware splitting
→ split diagnostics
→ baseline training
→ validation evaluation
→ reproducibility evidence
```

Acquisition is pinned to the configured source inventory. Validation checks record structure.
Mapping and extraction retain record, annotation, and sample lineage. Split schema v2 assigns
complete subjects to train, validation, and test with a seeded 70/15/15 subject shuffle; records
201 and 202 share one subject identity. Split diagnostics check disjointness, minimum counts, class
coverage, and ratio deviation according to `configs/splitting-v2.toml`.

Training opens only indexed train shards. Evaluation loads the frozen model without refitting and
opens only indexed validation shards. Artifact digests connect configurations, source and derived
inputs, the dataset index, fitted model, validation metrics, and run evidence. The supported local
workflow is sequential; it is not a deployed service or cloud implementation.

## Evaluation methodology and reporting

The current evaluator reports a confusion matrix, accuracy, per-class precision, recall, F1 and
support, and macro averages. Undefined precision, recall, or F1 is set to `0.0` by the versioned
evaluation configuration. Metrics are descriptive evidence for one baseline, one validation
partition, and one run lineage. Class support and the confusion matrix are required context because
class imbalance can make aggregate metrics misleading.

No metric value is asserted in this card. Validation values are generated locally under
`artifacts/runs/<run-id>/evaluation/validation-metrics.json` and must be interpreted with their run
manifest and reproducibility evidence. They are not a final benchmark, evidence of test-set or
population generalization, or evidence of clinical or medical utility.

The indexed test partition remains protected and unopened by the supported evaluator. A future
held-out evaluation requires a separate implementation and explicit approval under the
[evaluation policy](docs/evaluation-policy.md) and
[benchmark governance](docs/benchmark-governance.md). Governance alone does not authorize access.

The saved 2022 notebook results are separate historical outputs. Their random beat-window split
could place windows from the same record, including overlapping windows, on both sides of the split.
Those results may be inflated and do not measure generalization to unseen subjects. They must be
cited only with the defects documented in [historical results](docs/historical-results.md), never as
the supported baseline's benchmark.

## Assumptions

The documented baseline assumes that:

- upstream files match the configured MIT-BIH v1.0.0 inventory and repository-reviewed digests;
- configured record-to-subject identities are complete and correct for the included records;
- the selected annotations, `MLII` channel, six-second geometry, and exclusion policy are suitable
  only for reproducing this educational project definition;
- each generated shard and artifact matches its recorded size, digest, schema, counts, and lineage;
- train, validation, and test remain subject- and record-disjoint under split schema v2;
- validation is used for bounded development evidence and the protected test partition is not used
  for model, threshold, feature, preprocessing, or configuration selection; and
- identical configuration and inputs support deterministic membership and outputs, while elapsed
  time and resource observations vary by host and system load.

A violated assumption invalidates the affected interpretation. Passing automated checks reduces
specific integrity and leakage risks; it does not prove data quality, model quality, broader
generalization, or fitness for use.

## Intended use

Supported uses are limited to:

- education and research about reproducible data and machine-learning pipelines;
- demonstration and review of configuration, provenance, validation, lineage, grouped splitting,
  artifact integrity, evaluation isolation, and release governance;
- regression testing of the repository's transformation, training, metric, and evidence contracts;
- local reproduction of validation-only pipeline evidence; and
- responsible portfolio discussion of modernizing a historical project and correcting its claims.

## Prohibited use

Do not use this repository, its model, labels, outputs, metrics, or documentation for:

- diagnosis, screening, triage, prognosis, monitoring, treatment, alarms, or patient-care decisions;
- clinical workflows, healthcare deployment, medical devices, or safety-critical decisions;
- claims of medical effectiveness, clinical utility, diagnostic performance, regulatory
  suitability, healthcare readiness, or production readiness;
- identifying subjects or linking records with identifying information;
- presenting validation metrics as a held-out benchmark or population-generalization evidence;
- presenting the archived 2022 metrics as supported modern results; or
- bypassing protected-test controls, repeatedly inspecting test results, or selecting a model from
  test evidence.

## Reproducibility and lineage

Create the locked environment and run the supported pipeline from the repository root:

```fish
uv sync --locked --dev

uv run ecg-data run-pipeline \
  --repository-root . \
  --dataset-config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --split-config configs/splitting-v2.toml \
  --training-config configs/training-baseline-v1.toml \
  --evaluation-config configs/evaluation-baseline-v1.toml
```

Review the generated run manifest, environment summary, runtime summary, resource summary, evidence
manifest, split-quality summary, training metadata, and validation metrics together. The evidence
records Git state, lockfile and configuration identities, input and artifact digests, split
membership, model identity, host context, and elapsed stages. Generated source data, derived data,
models, metrics, and evidence remain ignored and are not portable merely because their schemas are
documented.

Detailed contracts are in [pipeline orchestration](docs/pipeline-orchestration.md),
[run manifests](docs/run-manifests.md), [reproducibility evidence](docs/reproducibility-evidence.md),
[model-ready dataset](docs/model-ready-dataset.md), [baseline training](docs/baseline-training.md),
and [baseline evaluation](docs/baseline-evaluation.md).

## Known limitations and residual risks

- One grouped split provides a stronger leakage boundary than the historical window split but does
  not measure variation across multiple splits or establish generalization beyond this dataset.
- Subject grouping depends on explicit metadata and prevents known subject crossover; it cannot
  detect unknown relationships, duplicate physiology, or other latent dependencies.
- The split algorithm balances subject counts rather than class or window distributions. Some
  sparse-class checks are warnings, so an accepted run can still have weak class coverage.
- The nearest-centroid baseline, random projection, fixed binary labels, and default class decision
  are engineering fixtures rather than evidence-backed modeling choices for another context.
- ROC/AUC and calibration analysis are not supported or claimed, and are not planned: the
  supported estimator predicts a hard class by nearest-centroid assignment
  (`np.argmin` over centroid distances) and exposes no ranked score or predicted
  probability for either to score against. Adding one would be a new modeling
  choice, not an evaluation-reporting addition, and is out of scope for this
  baseline's intentionally fixture-grade role (see above).
- Threshold-based decision analysis over the existing per-window centroid-distance
  margin, and figures generated from validation-only evaluation output, are
  candidate follow-up work, not yet scoped or implemented. Any such work must
  stay within the `validation` partition per [evaluation policy](docs/evaluation-policy.md).
- No uncertainty estimate, robustness study, subgroup analysis, external validation,
  device/site shift analysis, or held-out test result is supported or claimed.
- Determinism depends on the recorded software, configuration, inputs, and execution contract.
  Runtime and resource values are host observations, not performance benchmarks.
- The original 2022 environment was not captured, and its results are not exactly reproducible.
- Historical archive image attribution and `wrangle.py` tutorial-code adaptation extent are both
  audited (see [NOTICE.md](NOTICE.md), [`ATTRIBUTION.md`](archive/original_2022/ATTRIBUTION.md),
  and [`PROVENANCE.md`](archive/original_2022/PROVENANCE.md)). Neither is reused in this card.

## Maintenance and change control

Changes to source inventory, annotation mapping, window geometry, subject metadata, splitting,
estimator, or evaluation semantics change the evidence boundary and require versioned configuration,
regenerated artifacts, tests, and updated disclosure. Protected-test evaluation requires separate
review and must not be added implicitly to the validation path. Repository releases and any future
result publication follow [release governance](docs/governance/releases.md) and benchmark governance.

Report software vulnerabilities through [SECURITY.md](SECURITY.md). Other corrections should follow
[CONTRIBUTING.md](CONTRIBUTING.md). This card should be reviewed whenever a supported model, dataset
contract, evaluation policy, or material limitation changes.
