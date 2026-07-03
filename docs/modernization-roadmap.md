# Modernization roadmap

The work is intentionally incremental so that the original project remains inspectable while each
replacement gains tests and documentation. Checkboxes report repository state, not aspirations.

## Phase 1 — MVP documentation

- [x] Reframe the repository as a historical project under modernization.
- [x] Add research-only and non-clinical use limitations.
- [x] Separate historical outputs from validated claims.
- [x] Document dataset provenance, license, and citations.
- [x] Record known leakage, metric, attribution, and reproducibility risks.

## Phase 2 — Reproducible environment

- [x] Select Python 3.12 and `uv` as the supported environment workflow.
- [x] Add `pyproject.toml`, development dependencies, and a lock file.
- [ ] Replace absolute paths with configuration.
- [ ] Add repeatable dataset retrieval from the authoritative source.
- [x] Add expected-file inventory and local SHA-256 integrity checks.
- [x] Define raw, external, interim, processed, report, and artifact locations.

Exit criterion: a contributor can create the environment and run a lightweight data-access smoke
test from documented commands. Environment creation is complete; data access remains open.

## Phase 3 — Notebook cleanup

- [x] Maintain the original notebooks in the dated `archive/original_2022/` bundle.
- [ ] Identify one canonical narrative notebook.
- [ ] Remove duplicated pipeline implementation from curated notebooks.
- [ ] Clear stale errors and excessive outputs.
- [ ] Replace or attribute third-party imagery.

Exit criterion: curated notebooks run against package functions and have a clearly documented order and purpose.

## Phase 4 — Pipeline refactor

- [x] Create an installable `src` package boundary.
- [ ] Separate acquisition, validation, windowing, splitting, training, and evaluation.
  Inventory, record validation, mapping, window extraction, and grouped splitting are implemented;
  acquisition, training, and evaluation remain open.
- [x] Retain record identifiers through current inventory, ingestion, mapping, and window stages.
- [ ] Introduce a configuration-driven command-line entry point.
- [x] Write auditable run manifests for current data-stage evidence.
- [ ] Write machine-readable metrics when supported evaluation is implemented.

Exit criterion: the pipeline can be run without editing source files and produces traceable outputs.

## Phase 5 — Testing and validation

- [ ] Add synthetic fixtures that do not redistribute source ECG data.
- [x] Test sample-rate, shape, boundary-window, and label-mapping behavior.
- [x] Assert that records never cross split boundaries.
- [ ] Test confusion-matrix and metric calculations.
- [ ] Add a small end-to-end pipeline test.

Exit criterion: core transformations, split integrity, and metrics have automated regression coverage.

## Phase 6 — CI/CD

- [x] Run formatting, linting, security checks, and tests in CI.
- [x] Add Pyright Basic static type checking for source and tests.
- [ ] Validate curated notebooks without downloading the complete dataset.
- [x] Add dependency updates and secret scanning.
- [ ] Build documentation or package artifacts without automatic external publication.

Exit criterion: every proposed change receives automated, data-independent quality checks.

## Phase 7 — Portfolio polish

- [x] Document repository architecture and proposed data lineage.
- [ ] Publish a model card and newly evaluated record-grouped benchmark.
- [ ] Document runtime, resource use, and operational tradeoffs.
- [x] Describe future-state cloud concerns without claiming implementation.
- [ ] Review every model and pipeline claim against generated evidence.

Exit criterion: the repository demonstrates senior-level data engineering judgment without implying production or clinical readiness.
