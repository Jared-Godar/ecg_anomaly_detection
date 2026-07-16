# Testing Rigor

The README's Testing Rigor signal summarizes a layered verification strategy: a tiered pytest
suite that never touches the real dataset, curated notebooks executed end-to-end in CI, static
type checking, a broad pre-commit surface, package build assurance, and a merge-blocking
internal-documentation gate. This page tells the fuller story behind each bullet — what was
built, why it is shaped that way, and where it lives. The suite protects a research, education,
and software-engineering demonstration project (see the
[use limitation](../../README.md#important-use-limitation)); nothing here constitutes clinical
validation.

## A tiered suite under strict pytest configuration

Tests are organized by what they verify, not just where code lives: `tests/unit/` isolates
individual transformations and validation logic, `tests/integration/` exercises CLI and pipeline
component boundaries, `tests/scripts/` covers the operational and governance tooling under
`scripts/` (metadata gates, changelog validation, label sync, notebook validation), and
`tests/fixtures/` is the designated home for committed fixtures. The suite currently collects
more than 600 tests.

Pytest runs with `--strict-config --strict-markers --import-mode=importlib` plus `-ra`
([`pyproject.toml`](../../pyproject.toml)). Strict mode turns configuration drift into failures
rather than warnings: an undeclared marker or a misspelled ini option fails the run instead of
silently doing nothing. Supported behavior and its tests are introduced together — every module
in `src/ecg_anomaly_detection/` has a corresponding unit-test file, and the same commentary
standard applied to production code applies to tests, because tests are executable
specifications.

The capstone is `tests/integration/test_pipeline.py`: it seeds three synthetic WFDB records,
runs the entire acquisition-through-evaluation pipeline via the real CLI with only the HTTPS
fetcher faked, then re-runs the identical command and asserts zero fetch calls — proving the
verify-and-reuse resume path end to end while still producing a fresh, independent run manifest.

## Synthetic WFDB fixtures — no real dataset required

No test downloads or reads the MIT-BIH Arrhythmia Database. Tests that need ECG-shaped input
write synthetic records at runtime through the real `wfdb` package (`wfdb.wrsamp` /
`wfdb.wrann`), so the ingestion, validation, and windowing code paths are exercised against
genuine WFDB file formats without any patient-derived data entering the repository or CI.
[`docs/architecture.md`](../architecture.md) makes this a standing contract: tests may commit
only small synthetic fixtures or fixtures with explicit redistribution permission — never source
ECG recordings. In practice the generation-at-runtime approach has kept `tests/fixtures/` down
to a placeholder. The payoff is a suite that runs identically on a fresh clone, in CI, and on
an offline machine.

## Notebooks executed end-to-end in CI

The three curated public notebooks are not just linted — they are executed cell by cell on every
pull request and push to `main`
([`notebook-validation.yml`](../../.github/workflows/notebook-validation.yml)).
[`scripts/validate_curated_notebooks.py`](../../scripts/validate_curated_notebooks.py) runs the
real, unmodified notebooks inside an isolated `git worktree` copy of the repository, seeded with
a small synthetic WFDB record set and a matching acquisition manifest so `acquire_dataset` takes
its verify-and-reuse path and the real dataset is never fetched. This is deliberately distinct
from the static `ecg-data check-local-notebooks` command, which validates notebook JSON and
metadata without executing any cell. The design decision: documentation that executes is the
only documentation guaranteed not to rot, so the executable walkthroughs are held to the same
CI bar as code. See [notebook validation](../../notebooks/README.md#validation).

## Type checking with pyright

Pyright runs in Basic mode over `src/` and `tests/`, configured in
[`pyproject.toml`](../../pyproject.toml) against Python 3.12. It runs twice per change: as an
always-run local pre-commit hook for fast feedback, and as a CI step in the quality-gates
workflow — the shared enforcement boundary, since local hooks can be skipped or missing on
another machine. Basic mode is documented as an enforced baseline rather than the final target;
Strict mode is deferred until third-party WFDB boundaries can pass without broad ignores
([development workflow](../development.md#type-checking)).

## The pre-commit surface

[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) layers file hygiene
(trailing whitespace, final newlines, line endings, oversized additions, merge markers, case
conflicts, broken symlinks), syntax validation (JSON, TOML, YAML), Ruff linting and formatting
scoped to `src/`, `scripts/`, and `tests/`, markdownlint, gitleaks secret scanning, zizmor
GitHub Actions security auditing, private-key detection, a local block on committing directly
to `main`, and the always-run commentary hooks described below. The preserved
`archive/original_2022/` tree is excluded so current tooling never rewrites the historical
record.

CI runs the same configuration on every pull request
([`quality.yml`](../../.github/workflows/quality.yml)), skipping only hooks that have stronger
CI-native replacements: gitleaks is superseded by a dedicated job that scans the complete Git
history, pyright runs as its own step, and `no-commit-to-branch` is a local-only guard with no
protective value in CI. Nothing on the local surface is weaker in CI than at the keyboard.

## Package build assurance

A dedicated `package-build` CI job runs `uv build` on every pull request, verifies that both a
wheel and a source distribution were actually produced under `dist/`, and confirms via
`git ls-files` that no built artifact is tracked in the repository. It is a build-only assurance
check — it never uploads or publishes; publication remains a separate, explicitly reviewed
decision under [release governance](../governance/releases.md). The sdist contents are scoped in
[`pyproject.toml`](../../pyproject.toml) to `src/` plus top-level metadata files, so a source
distribution never carries the historical archive, tests, or CI configuration — and never any
dataset or model artifacts.

## The code-commentary gate

[`scripts/check_code_commentary.py`](../../scripts/check_code_commentary.py) enforces an
intentionally exhaustive internal-documentation policy using the standard-library AST and
tokenizer. It rejects modules without docstrings; classes, functions, methods, and nested
callables without docstrings; `if`/loop/`try`/`with`/`match` blocks without an immediately
preceding explanatory comment; and module-level assignments without a leading comment. The hook
is `always_run` locally and executes inside the CI pre-commit job, making missing commentary a
merge-blocking failure rather than a review nit. The completed audit behind it covered 68
supported Python files, 22,275 lines, and 789 definitions
([code commentary audit](../code-commentary-audit.md)).

A companion checker, `scripts/check_notebook_commentary.py`, extends the identical standard to
the curated notebooks' Python code cells by reusing the same audit engine on `nbformat`-extracted
cell source — statically, never executing a cell, and skipping non-Python cell magics. Both
checkers enforce a structural floor only; the audit doc is explicit that human review remains
responsible for semantic quality.

## Where to go deeper

- [Development workflow](../development.md) — environment setup, hook installation, and the full
  list of local checks
- [Code commentary audit](../code-commentary-audit.md) — the standard, the audited inventory,
  and the limits of automated enforcement
- [Notebook validation](../../notebooks/README.md#validation) — how the curated notebooks are
  executed in CI without the real dataset
- [Architecture](../architecture.md) — the directory contract, including the fixtures policy
- [Release governance](../governance/releases.md) — why package builds are assurance-only
