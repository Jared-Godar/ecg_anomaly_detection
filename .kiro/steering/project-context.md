# Project context

This steering file provides baseline facts about the repository so that every
Kiro session starts with an accurate mental model, avoiding stale assumptions
from training data.

## Governing contract

AGENTS.md is a binding, non-negotiable contract — not a suggestion, not
context, not optional reading. Every standing commitment applies to every
session. The rules were always there. Not following them is a defect in the
agent, not a gap in the documentation.

Before doing anything in this repository:

1. Read AGENTS.md on disk (not from memory)
2. Read the governance docs it references (on disk, not from memory)
3. Follow them as written — not approximately, not "in spirit," as written
4. Verify your own actions against them before claiming anything is done

## Repository identity

- **Name:** ecg_anomaly_detection
- **Owner:** Jared-Godar (GitHub)
- **License:** MIT (repository code); PhysioNet datasets carry their own terms
- **Python:** >=3.12, <3.14 (managed by `uv`, lockfile committed)
- **Package manager:** uv (replaces pip/poetry)
- **Build backend:** hatchling

## Project positioning

This is a historical educational ECG machine-learning project being modernized
into a data-engineering portfolio case study. It is NOT medical software and must
never be described as clinical, diagnostic, monitoring, or treatment software.

Original notebook results are presented as historical results with known
evaluation limitations (random beat-window split, not patient-grouped). The
modernized pipeline uses record/patient-grouped evaluation.

## Local environment

- macOS (darwin)
- Interactive shell: Fish
- All user-facing commands must use Fish syntax
- `gh` CLI authenticates via macOS Keychain (no tokens in env vars)
- Virtual environment: `.venv/` (activate with `source .venv/bin/activate.fish`)

## Key directories

| Path | Purpose |
|------|---------|
| `src/ecg_anomaly_detection/` | Installable package source |
| `notebooks/` | Curated analysis notebooks |
| `scripts/` | Utility and CI scripts |
| `tests/` | pytest test suite |
| `docs/` | Governance, architecture, and reference docs |
| `configs/` | Pipeline configuration YAML |
| `data/` | Data directory contracts (contents gitignored) |
| `artifacts/` | Generated outputs (gitignored except `.gitkeep`) |
| `archive/original_2022/` | Historical notebooks preserved as-is |

## Dataset

PhysioNet Computing in Cardiology Challenge 2017 (AF classification).
DOI and citation requirements must remain visible in documentation.
Raw data is never committed to Git.

## Engineering discipline

- Defensively code every external call: retry transient failures (timeouts,
  connection resets, 5xx) with bounded backoff; fail fast on permanent errors
  (404, auth, digest mismatch); on exhaustion exit gracefully with a clear
  message naming what failed and giving remediation steps — never a raw
  traceback on a user-facing surface.
- Diagnose before suppressing. Prove the root cause of a warning or failure
  before silencing it. Do not trade a real protection for cosmetic quiet.
- Governance docs are negotiable through explicit proposal, not silent
  workaround. When real friction surfaces, propose an update — don't route
  around it.

## Modernization approach

- Work incrementally; preserve original material where it remains useful.
- Propose destructive deletion or irreversible migration before performing it.
- Separate raw data, derived data, and generated artifacts.
- Keep source datasets and derived patient-level data out of Git unless
  redistribution has been explicitly reviewed.
- Make data provenance, configuration, schema validation, split integrity,
  and run metadata explicit.
- Add environment reproducibility before claiming a clean checkout reproduces
  results.
- Add tests around transformations, label mapping, boundary windows, grouped
  splitting, and metrics.

## Documentation and attribution

- Keep the dataset DOI, upstream license, and required citations visible.
- Distinguish the repository's MIT-licensed work from third-party datasets,
  images, tutorial material, and package licenses.
- Do not reuse historical images in new portfolio material until their source
  and reuse terms are verified.
- Label future-state cloud architecture as proposed unless actually implemented
  and tested.

## Portfolio presentation

- Present the project as a responsible modernization case study.
- Emphasize reproducibility, data pipeline hygiene, grouped evaluation,
  testability, and documentation.
- Avoid overclaiming model quality or medical applicability.
- Prefer clear architecture notes, runbooks, limitations, and decision records
  over flashy claims.

## Evaluation boundaries

- The implemented evaluator scores only the `validation` partition.
- Validation evidence supports pipeline verification and bounded model
  development. It is not a final benchmark and must not be presented as
  clinical evidence.
- The protected `test` partition is disabled by default and governed by
  `docs/benchmark-governance.md`.
- Do not present the original random beat-window split as evidence of
  generalization to unseen patients.
- Claims of diagnostic usefulness, healthcare-AI readiness, medical-device
  suitability, or production healthcare readiness are prohibited.
