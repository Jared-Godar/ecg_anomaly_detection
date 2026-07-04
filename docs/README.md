# Documentation guide

The maintained documentation is organized by the question a reader is trying to answer.

| Document | Purpose |
|---|---|
| [Project README](../README.md) | Portfolio overview, current status, quick start, and limitations |
| [Narrative walkthrough](../notebooks/narrative-walkthrough.ipynb) | Canonical package-backed explanation of the supported workflow and evidence boundaries |
| [Notebook guidance](../notebooks/README.md) | Supported notebook, execution, business-logic, and generated-figure policy |
| [Model card](../MODEL_CARD.md) | Baseline scope, assumptions, evaluation, intended use, and prohibited use |
| [Architecture](architecture.md) | Implemented repository boundaries and target component ownership |
| [Pipeline design](pipeline-design.md) | Proposed lineage, validation controls, run outputs, and cloud mapping |
| [Dataset acquisition](dataset-acquisition.md) | Versioned HTTPS retrieval, idempotency, recovery, and trust boundary |
| [Data integrity](data-integrity.md) | Implemented local file inventory, SHA-256 baseline, and trust boundary |
| [Record validation](record-validation.md) | Implemented WFDB ingestion, validation rules, and report schema |
| [Annotation mapping](annotation-mapping.md) | Versioned binary target policy, exclusions, and audit report |
| [Window extraction](window-extraction.md) | Boundary-safe geometry, lineage-preserving NPZ output, and overlap audit |
| [Record-grouped splitting](record-grouped-splitting.md) | Deterministic partition membership, leakage controls, and split manifest |
| [Split quality reporting](split-quality-reporting.md) | Partition diagnostics, acceptance thresholds, and failure behavior |
| [Run manifests](run-manifests.md) | Git, environment, configuration, dataset, split, and artifact evidence |
| [Reproducibility evidence](reproducibility-evidence.md) | Versioned environment, runtime, resource, and digest evidence |
| [Pipeline orchestration](pipeline-orchestration.md) | One-command local workflow, outputs, failure behavior, and limits |
| [Model-ready dataset](model-ready-dataset.md) | Grouped shard index, lazy-loading contract, and training boundary |
| [Baseline evaluation](baseline-evaluation.md) | Frozen-model validation metrics, isolation, and digest checks |
| [Evaluation policy](evaluation-policy.md) | Development evaluation and protected-test boundaries |
| [Benchmark governance](benchmark-governance.md) | Future benchmark eligibility, execution, disclosure, rerun, and archival rules |
| [Data provenance](data-provenance.md) | Dataset source, license, attribution, privacy, and label provenance |
| [Historical results](historical-results.md) | Saved 2022 metrics and their known evaluation defects |
| [Development workflow](development.md) | Locked environment, tests, hooks, and CI behavior |
| [Modernization roadmap](modernization-roadmap.md) | Completed, active, and planned modernization phases |
| [Governance guide](governance/index.md) | Repository, issue, security, versioning, and release governance |
| [Issue workflow](governance/issue-workflow.md) | Issue intake, triage, status transitions, and project tracking |
| [Label taxonomy](governance/label-taxonomy.md) | Label dimensions, assignment rules, and deterministic bootstrap |
| [Repository governance](governance/repository-governance.md) | Ownership, pull-request workflow, and enforcement boundaries |
| [Security governance](governance/security-policy.md) | Private reporting, supported versions, dependency stewardship, and limitations |
| [Release governance](governance/releases.md) | Release boundaries, evidence, artifact hygiene, and explicit limitations |
| [Versioning policy](governance/versioning.md) | Semantic version rules and pre-1.0 change disclosure |
| [Release checklist](governance/release-checklist.md) | Review steps for a future, explicitly authorized release |
| [Changelog](../CHANGELOG.md) | Unreleased changes and version history |
| [Contributing](../CONTRIBUTING.md) | Change scope, data safety, validation, and pull request expectations |
| [Third-party notices](../NOTICE.md) | Dataset, dependency, tutorial, and historical asset attribution status |

## Documentation rules

- Describe only tested behavior as implemented.
- Label unbuilt components and cloud services as proposed.
- Present the 2022 model output only as historical evidence with the record-leakage caveat.
- Keep source-dataset attribution and the research/educational-use limitation visible.
- Prefer links to generated evidence once the modern pipeline produces manifests and reports.
