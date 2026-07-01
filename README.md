# ECG Beat-Window Classification

> Historical machine-learning project under incremental modernization as a reproducible data pipeline case study.

This repository preserves a 2022 educational project that used the public MIT-BIH Arrhythmia Database to classify six-second ECG windows as normal or abnormal. The modernization focuses on reproducible data workflows, environment management, validation, testing, and responsible communication of model results.

## Important use limitation

This project is for research, education, and software-engineering demonstration only. It is not medical software, has not been clinically validated, and must not be used for diagnosis, monitoring, treatment, or patient-care decisions.

The labels are simplified reference-annotation classes from a historical dataset. A model output from this project does not establish whether a person has a medical condition.

## Project status

The repository currently contains the original notebook-oriented experiment. Its code, outputs, and reported metrics are retained for historical context while the workflow is modernized incrementally.

| Area | Current state | Modernization target |
|---|---|---|
| Data access | Manual download and local paths | Configured, versioned acquisition with integrity checks |
| Environment | Python 3.12 project with a committed `uv` lockfile | Add pipeline dependencies as supported modules land |
| Transformation | Archived notebook and `wrangle.py` logic | Tested package modules and command-line workflow |
| Evaluation | Random beat-window split | Patient/record-grouped validation |
| Quality controls | Assertions only | Schema, signal, split, and metric tests |
| Automation | Linting, security scans, and package smoke tests in CI | Add data-independent pipeline and notebook checks |

See the [repository architecture](docs/architecture.md) for directory boundaries and the [modernization roadmap](docs/modernization-roadmap.md) for the planned phases.

Contributor setup and automated quality checks are documented in the [development workflow](docs/development.md).

## Quick start

Install [uv](https://docs.astral.sh/uv/), then create the exact development environment and run the
current smoke test:

```fish
uv sync --locked --dev
uv run pytest
```

The supported environment installs the modern package scaffold only. It does not make the archived
2022 notebooks reproducible or download any ECG data.

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

The original code depends on absolute local paths and an unrecorded environment, so a clean checkout is not yet reproducible. Reproducible commands will be added in the environment phase rather than suggesting that the legacy workflow currently runs unchanged.

## Historical result and its limitation

The 2022 notebook reports 95.3% test accuracy, 85.5% recall for the combined abnormal class, and a 0.2% false-positive rate. These values describe the saved output of the original experiment; they are not validated portfolio benchmarks.

Most importantly, the original split assigns individual beat windows randomly. Windows from the same record—and potentially overlapping windows—can therefore occur in training and test data. This can inflate performance and does not measure generalization to unseen patients. The modernized pipeline will use record-grouped evaluation before publishing a new benchmark.

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
| `src/ecg_anomaly_detection/` | Modern installable Python package | Scaffolded; implementation pending |
| `configs/` | Versioned, non-secret pipeline configuration | Scaffolded |
| `data/` | Ignored raw, external, interim, and processed data stages | Scaffolded with documented contracts |
| `notebooks/` | Future curated notebooks | Scaffolded; curated notebooks pending |
| `tests/` | Unit, integration, and synthetic-fixture boundaries | Scaffolded; tests pending |
| `scripts/` | Thin operational entry points | Scaffolded |
| `artifacts/` | Ignored generated models and run outputs | Scaffolded |
| `reports/figures/` | Ignored reproducible figure output | Scaffolded |
| `archive/original_2022/` | Original notebooks, wrangling code, and presentation images | Preserved; unsupported historical reference |
| `docs/` | Architecture, modernization, provenance, and result documentation | Active |

No source data, generated feature tables, or trained model artifacts are tracked in Git.

## Known limitations

- This is a small historical dataset and is not representative of modern deployment populations.
- The binary target collapses heterogeneous annotations and excludes others.
- The first signal channel is used without a channel-selection analysis.
- The original split is not patient/record-grouped or stratified.
- Adjacent windows may overlap.
- The original environment and exact dependency versions were not captured.
- Automated coverage currently validates only the package and environment scaffold; pipeline tests
  will be added with supported transformation behavior.
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

## License

Project code and original documentation are available under the [MIT License](LICENSE). Dataset files and third-party materials retain their own licenses and attribution requirements.

## Citation

If referencing this repository, use the metadata in [CITATION.cff](CITATION.cff). If using the dataset, cite both MIT-BIH and PhysioNet as described in [data provenance](docs/data-provenance.md).
