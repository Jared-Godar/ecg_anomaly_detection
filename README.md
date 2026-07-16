# ECG Anomaly Detection Pipeline

[![Quality gates](https://github.com/Jared-Godar/ecg_anomaly_detection/actions/workflows/quality.yml/badge.svg)](https://github.com/Jared-Godar/ecg_anomaly_detection/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](pyproject.toml)

![ECG Anomaly Detection — Modernization Case Study: governance, reproducibility, engineering discipline, validation-only, responsible ML](docs/assets/ecg-readme-project-overview-banner.png)

This project started as a 2022 educational ECG classification experiment on the
public MIT-BIH Arrhythmia Database. It has since been rebuilt from scratch under
strict governance, engineering best practices, and documented workflows for
model-agnostic agentic coding — a showcase of how disciplined software
engineering transforms a notebook prototype into an auditable, reproducible data
pipeline.

## Important use limitation

This project is for research, education, and software-engineering demonstration
only. It is **not** medical software, has not been clinically validated, and must
not be used for diagnosis, monitoring, treatment, or patient-care decisions.

## Why this project

Most of my client-facing consulting work has been in heavily regulated public- and
private-sector environments, where security concerns have historically limited the use of
cloud-based LLMs. This project is an exercise in working as a project manager and software
engineer with a cloud LLM — demonstrating that rigorous governance, reproducibility, and
operational maturity are achievable in a modern ML pipeline. The deliberate
overengineering of governance and compliance began as an unrelated goal, but over the
course of the work it became instrumental in enabling an agent to work effectively and
safely in a cloud environment. It is also the first project where I established governance
and stewardship first, rather than bolting them on after the fact.

Automated metadata tagging on every issue and pull request, merge gates that block any PR
lacking complete metadata or a changelog entry, and scheduled drift detection have
lightened the manual load of important-but-tedious tasks — letting the
[GitHub project board](https://github.com/users/Jared-Godar/projects/5) serve as a true
project-management tool rather than a simple issue tracker. The result is a project that
is auditable, reproducible, and safe to run in a cloud environment, and a showcase of
working with LLMs in a disciplined, responsible way.

- **Project board (live):** [ECG Pipeline Modernization — GitHub Project](https://github.com/users/Jared-Godar/projects/5)
- **Milestones:** [Milestone dashboard](https://github.com/Jared-Godar/ecg_anomaly_detection/milestones)

## Agentic Engineering

Development is governed by a tracked operating contract
([`AGENTS.md`](AGENTS.md)) that binds every coding-agent session — including
cold-start and cloud sessions with no inherited memory — to hard, non-negotiable
commitments:

- **Self-recording promises** — every standing rule is captured in a tracked
  file, never only in agent memory; a promise that lives only in conversation is
  not considered made
- **Session-handoff continuity** — on wind-down, the agent produces a Markdown
  handoff with state snapshot, copy-pasteable next steps, and open risks so work
  continues without an agent present
- **Metadata gates as merge-tier checks** — a CI workflow
  ([`validate_project_metadata.py`](scripts/github/validate_project_metadata.py))
  enforces nine required Project fields, assignee, labels, milestone, and closing
  references on every pull request before merge
- **Standing authorization with four gated actions** — agents run the full
  workflow unprompted (issue, branch, implement, gate, document, disclose) and
  pause only at push, open-PR, merge, and release-tag
- **Model-agnostic** — the contract works identically across Claude Code, Codex, AmazonQ, CoPIilor, and any successor.; no vendor lock-in in the workflow

*Deep dive: [Agentic Engineering — the full readout](docs/portfolio/agentic-engineering.md)*

## Governance

The repository enforces process discipline through CI and project-board
automation rather than trust:

- **Nine-field Project tracking** — every issue and PR carries Status,
  Workstream, Issue Type, Priority, Risk, Size, Repository Area, Portfolio
  Signal, and Target Release; validated by a GraphQL-backed CI gate with quota
  stewardship ([docs](docs/governance/github-project.md))
- **Label taxonomy** — namespaced `type:`, `area:`, `priority:`, `portfolio:`
  labels with a completed legacy-migration
  ([taxonomy](docs/governance/label-taxonomy.md))
- **Per-PR changelog enforcement** — a dedicated CI job rejects any PR that
  touches substantive code without updating `CHANGELOG.md`
  ([workflow](.github/workflows/metadata-governance.yml))
- **Protected-test benchmark governance** — held-out evaluation requires explicit
  approval and is bounded to one frozen candidate per governed execution
  ([policy](docs/benchmark-governance.md))
- **Bot-author exemption class** — Dependabot PRs receive compensating checks
  (own Project membership, nine-field completeness) instead of the
  closing-reference requirement

*Deep dive: [Governance — the full readout](docs/portfolio/governance.md)*

## Reproducibility

Every pipeline run produces machine-readable lineage tying code state to
artifacts:

- **Run manifests** — UUID-isolated runs record git revision + dirty flag,
  full `EnvironmentSnapshot` (Python version, platform, every installed package),
  `uv.lock` digest, dataset evidence, split evidence, and SHA-256 digests for
  every configuration/evidence/artifact file
  ([`run_manifest.py`](src/ecg_anomaly_detection/run_manifest.py))
- **Locked environment** — `uv sync --locked` with committed lockfile; separate
  dependency groups for dev, notebooks, and experiments; Python
  `>=3.12,<3.14` pinned in `pyproject.toml`
- **Versioned TOML configs** — ten configuration files with `schema_version`
  fields covering dataset identity, annotation mapping, windowing, splitting,
  training, and evaluation ([`configs/`](configs/))
- **Deterministic seeds** — `seed = 2022` in split and training configs; repeated
  runs reproduce the same subject partitions and fitted models

*Deep dive: [Reproducibility — the full readout](docs/portfolio/reproducibility.md)*

## Testing Rigor

The test suite is structured as `unit/`, `integration/`, `scripts/`, and
`fixtures/` under strict pytest configuration
(`--strict-config --strict-markers --import-mode=importlib`):

- **Synthetic WFDB fixtures** — tests run against generated records, never the
  real dataset; no download required for CI or local development
- **Notebook execution in CI** — all three curated public notebooks execute
  end-to-end on every PR against synthetic data
  ([workflow](.github/workflows/notebook-validation.yml))
- **Type checking** — `pyright` in basic mode covers `src/` and `tests/` on
  every commit
- **Full pre-commit surface** — Ruff formatting and linting, markdownlint,
  gitleaks secret scanning, trailing-whitespace and EOF enforcement
- **Package build assurance** — CI builds wheel and sdist, verifies artifacts
  exist and are not tracked in git
- **Code commentary gate** — a script enforces docstrings and comments on all
  supported Python, treating missing commentary as a merge-blocking failure

*Deep dive: [Testing Rigor — the full readout](docs/portfolio/testing-rigor.md)*

## Data Engineering

The pipeline's load-bearing contribution is eliminating the data-leakage defect
in the original 2022 experiment:

- **Subject-grouped splitting** — a seeded-subject-shuffle strategy assigns
  entire subjects (not individual beat windows) to train/validation/test
  partitions; records 201 and 202 share one subject ID because they originate
  from the same source tape
  ([`splitting.py`](src/ecg_anomaly_detection/splitting.py),
  [design doc](docs/record-grouped-splitting.md))
- **Pairwise-disjoint enforcement** — code-level guards reject any split where a
  subject or record appears in more than one partition
- **Typed WFDB ingestion** — `wfdb`-based record loading with per-record
  structural validation and channel-identity resolution
- **Closed-world label mapping** — a versioned annotation map explicitly handles
  every AAMI symbol class; unmapped annotations fail rather than silently
  dropping
- **Closed file inventory** — 144 expected source files (48 records x 3
  extensions), each with committed SHA-256 and size; any drift is a hard failure
- **Split quality acceptance** — configurable thresholds for minimum
  subjects/records/windows per partition, required class coverage, and maximum
  ratio deviation

*Deep dive: [Data Engineering — the full readout](docs/portfolio/data-engineering.md)*

## Operational Maturity

External calls follow a defensive contract: retry transient failures, fail fast
on permanent ones, and exit gracefully with bounded messaging:

- **Bounded transient retries** — HTTPS downloads retry timeouts, connection
  resets, `IncompleteRead`, and HTTP 429/500/502/503/504 up to three attempts
  with exponential backoff (2 s, 4 s); permanent errors (404, digest mismatch)
  fail on attempt one
  ([`acquisition.py`](src/ecg_anomaly_detection/acquisition.py))
- **Graceful exhaustion** — on failure, the message names what failed, states
  plainly that it is an external connectivity condition rather than a code
  defect, and gives re-run remediation
- **Atomic file install** — acquired files are staged in a temporary directory,
  digest-verified, then hard-linked into place; a crash mid-acquisition leaves no
  partial state
- **Run lifecycle** — `list-runs` inventories local artifacts by UUID;
  `purge-run` reclaims disk space for a specific run without touching the shared
  acquisition baseline
- **Per-stage progress reporting** — `run-pipeline` emits record-level
  acquisition progress with qualified timing and remaining-duration estimates,
  stage banners, and a model-fit heartbeat — all flushed without per-iteration
  noise

*Deep dive: [Operational Maturity — the full readout](docs/portfolio/operational-maturity.md)*

## Quick start

```fish
uv sync --locked --dev
uv run pytest
```

Run the full pipeline:

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

See [environment reproducibility](docs/environment-reproducibility.md) for
dependency groups, interpreter requirements, and notebook kernel setup.

## Reference

| Resource | Link |
|---|---|
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Pipeline orchestration | [docs/pipeline-orchestration.md](docs/pipeline-orchestration.md) |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |
| Data provenance and licensing | [docs/data-provenance.md](docs/data-provenance.md) |
| Governance index | [docs/governance/](docs/governance/index.md) |
| Modernization roadmap | [docs/modernization-roadmap.md](docs/modernization-roadmap.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Release notes — v1.0.0 (retrospective) | [docs/releases/v1.0.0.md](docs/releases/v1.0.0.md) |
| Release notes — v1.1.0 | [docs/releases/v1.1.0.md](docs/releases/v1.1.0.md) |
| Public notebooks | [notebooks/README.md](notebooks/README.md) |
| Historical results audit | [docs/historical-results.md](docs/historical-results.md) |
| Known limitations | [MODEL_CARD.md#known-limitations-and-residual-risks](MODEL_CARD.md#known-limitations-and-residual-risks) |

The project uses the [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
(DOI: [10.13026/C2F305](https://doi.org/10.13026/C2F305)) under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
Repository code is [MIT-licensed](LICENSE); dataset and third-party terms are
recorded in [NOTICE.md](NOTICE.md) and [data provenance](docs/data-provenance.md).

---

Jared Godar — open to data and ML engineering roles.
[GitHub](https://github.com/Jared-Godar)
