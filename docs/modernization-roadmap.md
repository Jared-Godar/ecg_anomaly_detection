# Modernization roadmap

The work is intentionally incremental so that the original project remains inspectable while each replacement gains tests and documentation.

## Phase 1 — MVP documentation

- Reframe the repository as a historical project under modernization.
- Add research-only and non-clinical use limitations.
- Separate historical outputs from validated claims.
- Document dataset provenance, license, and citations.
- Record known leakage, metric, attribution, and reproducibility risks.

## Phase 2 — Reproducible environment

- Select a supported Python version and package workflow.
- Add `pyproject.toml`, development dependencies, and a lock file.
- Replace absolute paths with configuration.
- Add repeatable dataset retrieval and integrity checks.
- Define raw, interim, processed, and artifact locations.

Exit criterion: a contributor can create the environment and run a lightweight data-access smoke test from documented commands.

## Phase 3 — Notebook cleanup

- Maintain the original notebooks in the dated `archive/original_2022/` bundle.
- Identify one canonical narrative notebook.
- Remove duplicated pipeline implementation from curated notebooks.
- Clear stale errors and excessive outputs.
- Replace or attribute third-party imagery.

Exit criterion: curated notebooks run against package functions and have a clearly documented order and purpose.

## Phase 4 — Pipeline refactor

- Create an installable `src` package.
- Separate acquisition, validation, windowing, splitting, training, and evaluation.
- Retain record identifiers throughout the pipeline.
- introduce a configuration-driven command-line entry point.
- Write run manifests and machine-readable metrics.

Exit criterion: the pipeline can be run without editing source files and produces traceable outputs.

## Phase 5 — Testing and validation

- Add synthetic fixtures that do not redistribute source ECG data.
- Test sample-rate, shape, boundary-window, and label-mapping behavior.
- Assert that records never cross split boundaries.
- Test confusion-matrix and metric calculations.
- Add a small end-to-end pipeline test.

Exit criterion: core transformations, split integrity, and metrics have automated regression coverage.

## Phase 6 — CI/CD

- Run formatting, linting, type checks, and tests in CI.
- Validate notebooks without downloading the complete dataset.
- Add dependency and secret scanning.
- Build documentation or package artifacts without automatic external publication.

Exit criterion: every proposed change receives automated, data-independent quality checks.

## Phase 7 — Portfolio polish

- Add architecture and data-lineage diagrams.
- Publish a model card and newly evaluated record-grouped benchmark.
- Document runtime, resource use, and operational tradeoffs.
- Add an explicitly future-state cloud architecture where useful.
- Review every README claim against generated evidence.

Exit criterion: the repository demonstrates senior-level data engineering judgment without implying production or clinical readiness.
