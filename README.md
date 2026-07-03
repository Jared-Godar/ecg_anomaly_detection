# ECG Beat-Window Classification

> Historical machine-learning project under incremental modernization as a reproducible data pipeline case study.

This repository preserves a 2022 educational project that used the public MIT-BIH Arrhythmia Database to classify six-second ECG windows as normal or abnormal. The modernization focuses on reproducible data workflows, environment management, validation, testing, and responsible communication of model results.

## Important use limitation

This project is for research, education, and software-engineering demonstration only. It is not medical software, has not been clinically validated, and must not be used for diagnosis, monitoring, treatment, or patient-care decisions.

The labels are simplified reference-annotation classes from a historical dataset. A model output from this project does not establish whether a person has a medical condition.

## Project status

The repository preserves the original notebook-oriented experiment and now provides a modern
package scaffold, locked Python environment, automated quality gates, and explicit data-directory
contracts. Supported stages now inventory and validate local records, map annotations, extract
boundary-safe windows, and create deterministic record-grouped split manifests. Auditable run
manifests connect those outputs to code, environment, configuration, and artifact digests. Dataset
retrieval from the versioned PhysioNet file source is implemented with local integrity evidence.
A local orchestration command connects all currently supported data stages. Model-ready partition
indexing is implemented without concatenating large arrays. Model training and model evaluation are
not yet implemented.

## What this case study demonstrates

- incremental modernization without erasing the historical record;
- reproducible Python environment and package management with `uv`;
- separation of source data, derived data, code, notebooks, and generated artifacts;
- automated linting, security scanning, dependency maintenance, and tests;
- explicit provenance, licensing, lineage, and responsible-use documentation; and
- an evaluation plan designed around record-grouped splits rather than random beat-window leakage.

| Area | Current state | Modernization target |
|---|---|---|
| Data access | Manual download and local paths | Versioned HTTPS acquisition and integrity checks implemented |
| Environment | Python 3.12 project with a committed `uv` lockfile | Add pipeline dependencies as supported modules land |
| Transformation | Archived notebook and `wrangle.py` logic | Tested package modules and command-line workflow |
| Evaluation | Random beat-window split | Record-grouped split implemented; evaluation pending |
| Quality controls | Assertions only | Schema, signal, window, and split-integrity tests |
| Automation | Linting, type checks, security scans, and tests in CI | Add data-independent notebook checks |

See the [repository architecture](docs/architecture.md) for directory boundaries and the [modernization roadmap](docs/modernization-roadmap.md) for the planned phases.

Start with the [documentation guide](docs/README.md). Contributor setup and automated quality
checks are documented in the [development workflow](docs/development.md).

## Quick start

Install [uv](https://docs.astral.sh/uv/), then create the exact development environment and run the
current smoke test:

```fish
uv sync --locked --dev
uv run pytest
```

The supported environment installs the modern package scaffold only. It does not make the archived
2022 notebooks reproducible or download any ECG data.

## Implemented data check

The package can retrieve the configured MIT-BIH files from PhysioNet into the ignored canonical raw
zone using fail-safe, idempotent behavior. See [dataset acquisition](docs/dataset-acquisition.md).

The package provides a metadata-driven command that validates the 144 required MIT-BIH record
files and records a local SHA-256 integrity baseline. The inventory command itself does not download
or redistribute data. Commands and trust limitations are documented in
[data integrity](docs/data-integrity.md).

It also loads individual local WFDB records, preserves physical signals and original annotations,
validates structural contracts, and writes machine-readable reports. See
[record validation](docs/record-validation.md).

Original annotation symbols can be mapped through a versioned, closed-world policy that reports
every inclusion and exclusion and fails on unknown symbols. See
[annotation mapping](docs/annotation-mapping.md).

Mapped annotations can be converted into boundary-safe six-second windows with row-level lineage,
versioned geometry, and overlap reporting. See [window extraction](docs/window-extraction.md).

One or more window artifacts can be assigned to deterministic train, validation, and test
partitions without allowing a record to cross boundaries. See
[record-grouped splitting](docs/record-grouped-splitting.md).

Generated stage evidence can be linked to the Git revision, installed environment, dependency lock,
configuration, dataset inventory, split membership, and artifact checksums. See
[run manifests](docs/run-manifests.md).

All supported data stages can be executed through one sequential, configuration-driven command. See
[pipeline orchestration](docs/pipeline-orchestration.md).

Grouped partitions are exposed as a model-ready index over validated, immutable record shards. See
[model-ready dataset](docs/model-ready-dataset.md).

## Historical workflow

The original experiment:

1. downloaded MIT-BIH waveform and annotation files;
2. read the data with the WFDB Python package;
3. selected the first signal channel;
4. extracted six-second windows centered on annotated beats;
5. mapped selected annotations into normal and abnormal classes;
6. randomly divided beat windows into training, validation, and test sets;
7. trained and tuned random-forest classifiers; and
8. presented exploration and results in [`report.ipynb`](archive/original_2022/report.ipynb).

The original code depends on absolute local paths and an unrecorded environment. The modern package
environment is reproducible, but the archived workflow is not supported and is not expected to run
unchanged from a clean checkout.

## Historical result and its limitation

The 2022 notebook reports 95.3% test accuracy, 85.5% recall for the combined abnormal class, and a 0.2% false-positive rate. These values describe the saved output of the original experiment; they are not validated portfolio benchmarks.

Most importantly, the original split assigns individual beat windows randomly. Windows from the same record—and potentially overlapping windows—can therefore occur in training and test data. This can inflate performance and does not measure generalization to unseen patients. The modernized pipeline now creates record-grouped memberships, but no new benchmark will be published before a supported training and evaluation workflow exists.

One historical notebook cell also displays a validation accuracy of 1.00 because it mistakenly scores predictions against themselves. The separately calculated validation accuracy for that model is 0.845. Details are recorded in [historical results](docs/historical-results.md).

## Dataset

The project uses the [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/) hosted by PhysioNet. It contains 48 half-hour, two-channel ambulatory ECG recordings from 47 subjects, sampled at 360 Hz, with reference beat annotations.

- Dataset DOI: [10.13026/C2F305](https://doi.org/10.13026/C2F305)
- Upstream data license: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/)
- Access: open, subject to the upstream license and attribution requirements
- Repository policy: source and derived data are not committed

Required citations and data-handling notes are in [data provenance](docs/data-provenance.md). The repository's MIT license applies to project code and documentation; it does not replace the dataset's license.

## Repository map

| Path | Purpose | Status |
|---|---|---|
| `src/ecg_anomaly_detection/` | Modern installable Python package | Inventory through grouped splitting plus run evidence implemented |
| `configs/` | Versioned, non-secret pipeline configuration | Dataset, mapping, windowing, and splitting implemented |
| `data/` | Ignored raw, external, interim, and processed data stages | Scaffolded with documented contracts |
| `notebooks/` | Future curated notebooks | Scaffolded; curated notebooks pending |
| `tests/` | Unit, integration, and synthetic-fixture boundaries | Current supported stages tested without ECG data |
| `scripts/` | Thin operational entry points | Scaffolded |
| `artifacts/` | Ignored generated models and run outputs | Scaffolded |
| `reports/figures/` | Ignored reproducible figure output | Scaffolded |
| `archive/original_2022/` | Original notebooks, wrangling code, and presentation images | Preserved; unsupported historical reference |
| `docs/` | Architecture, modernization, provenance, and result documentation | Active |
| `pyproject.toml`, `uv.lock` | Package metadata and exact dependency resolution | Implemented |

No source data, generated feature tables, or trained model artifacts are tracked in Git.

## Known limitations

- This is a small historical dataset and is not representative of modern deployment populations.
- The binary target collapses heterogeneous annotations and excludes others.
- The first signal channel is used without a channel-selection analysis.
- The original split is not patient/record-grouped or stratified.
- Adjacent windows may overlap.
- The original environment and exact dependency versions were not captured.
- Record grouping prevents record crossover but does not establish subject-level independence when
  multiple records represent the same person.
- Some exploratory notebooks contain saved errors and duplicated code.
- Third-party image and tutorial attribution is being audited; see [NOTICE.md](NOTICE.md).

## Modernization principles

- Preserve the original work and label it clearly.
- Separate immutable raw data, derived data, and generated artifacts.
- Make lineage and configuration explicit.
- Validate data contracts and split boundaries.
- Prefer patient/record-level evaluation.
- Report limitations alongside metrics.
- Avoid clinical or diagnostic claims.
- Add cloud-oriented design only where it is implemented or clearly presented as a future extension.

## Current next step

The next implementation slice is deterministic baseline training that fits only on the indexed train
partition and emits machine-readable held-out metrics. Any new benchmark will be reported with its
record-level limitations. The target contracts are documented in the
[proposed pipeline design](docs/pipeline-design.md).

## License

Project code and original documentation are available under the [MIT License](LICENSE). Dataset files and third-party materials retain their own licenses and attribution requirements.

## Citation

If referencing this repository, use the metadata in [CITATION.cff](CITATION.cff). If using the dataset, cite both MIT-BIH and PhysioNet as described in [data provenance](docs/data-provenance.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the Fish-compatible local workflow, data-safety rules,
validation commands, and pull request expectations.
