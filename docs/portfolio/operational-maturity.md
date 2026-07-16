# Operational Maturity

The README's Operational Maturity signal summarizes how this project treats the unglamorous parts
of a data pipeline — network failures, partial writes, disk accumulation, and long silent waits —
as designed behavior rather than afterthoughts. This page expands that summary: what each control
does, why it exists, and where it lives. As everywhere in this repository, the pipeline is a
research/education/software-engineering demonstration, not medical software.

The controls below share one philosophy: an operator should never see a raw traceback for an
external condition, never inherit partial state from a crash, and never wonder whether a
long-running command is still alive.

## A defensive contract for every external call

The repository's operating contract ([`AGENTS.md`](../../AGENTS.md), "Engineering discipline")
codifies a standing rule: every operation that leaves the process for a network or external
service must retry *transient* failures a bounded number of times with backoff, fail fast on
*permanent* errors, and on exhaustion exit gracefully with a bounded message — never a raw
traceback on a user-facing surface. The rule originated from a real incident: a brief PhysioNet
timeout surfaced an unhandled traceback on the first-run notebook surface during a fresh-clone
walkthrough, and the retrofit (issue #201) turned the lesson into tracked policy.

The reference implementation is dataset acquisition
([`acquisition.py`](../../src/ecg_anomaly_detection/acquisition.py)). Its design separates the
*classification* question from the *retry* question:

- A dedicated `TransientAcquisitionError` subtype marks failures a short wait can plausibly fix —
  timeouts, dropped/reset connections, a response body truncated mid-transfer
  (`http.client.IncompleteRead`), name-resolution blips, and HTTP 429/500/502/503/504. Because it
  subclasses the module's ordinary `AcquisitionError`, non-retrying callers behave exactly as
  before.
- A single retry wrapper (`_fetch_with_transient_retries`) sits between acquisition and every
  transport call, so the policy is uniform and testable without network access: up to three total
  attempts with exponential backoff (2 s, then 4 s), the partially staged file removed between
  attempts so each retry re-enters the same clean staging contract.
- Permanent failures — HTTP 404/403, rejected redirects, size-cap violations, and every digest or
  size mismatch — never carry the transient type, so they fail fast on attempt one. Retrying them
  could not change the outcome and would only delay the clean failure.
- On exhaustion, the raised message names the URL and attempt count, states plainly that the cause
  is an external connectivity or service condition rather than a repository or setup defect, and
  gives re-run remediation — because acquisition is atomic, a re-run restarts cleanly.

Retries never weaken integrity: whichever attempt succeeds still passes the same staged size and
SHA-256 checks against the expectations committed in the dataset config.

## Atomic staged-and-verified file install

Acquisition is built so that a crash at any point leaves no partial state the next run could
mistake for valid data. Files are streamed into a temporary staging directory created alongside
the destination (same filesystem, so the later hard-link install is atomic), opened with
exclusive-create mode so staging can never silently overwrite anything, and verified against the
committed size and SHA-256 expectations before install. The acquisition manifest is written first
— to a temporary file that is fsynced and then hard-linked into place, a pattern that makes
"write exactly once, never overwrite" an enforced guarantee rather than a convention — and only
then are the verified files hard-linked into the destination, so files can never exist on disk
that the manifest does not account for.

The same discipline covers re-runs: an existing file is re-hashed and checked against both the
committed config and the recorded manifest digest (never trusted merely because it exists), a
missing file is re-downloaded through the same retry and verification path, and required files
present *without* a manifest fail loudly rather than being adopted. There is deliberately no
`--force` option. Full behavior and the trust boundary are documented in
[dataset acquisition](../dataset-acquisition.md) and [data integrity](../data-integrity.md).

## Run lifecycle: list-runs and purge-run

Every `run-pipeline` invocation gets a UUID and writes into isolated directories, and run
directories are never reused or overwritten — a failed run's partial output stays on disk as
debugging evidence. That design accumulates disk usage by construction, so the lifecycle helpers
in [`local_execution.py`](../../src/ecg_anomaly_detection/local_execution.py) give the operator an
explicit, bounded way to reclaim it:

```fish
uv run ecg-data list-runs --repository-root .
uv run ecg-data purge-run --repository-root . --run-id <run-id> --dry-run
uv run ecg-data purge-run --repository-root . --run-id <run-id>
```

`list-runs` reports each run's ID, total size, and whether its `run-manifest.json` exists, newest
first. `purge-run` removes exactly the three companion directories one run created
(`artifacts/runs/<run-id>/`, `data/interim/runs/<run-id>/`, `data/processed/runs/<run-id>/`) — a
run is either fully present or fully removed. The scope guards are code, not convention: the run
ID must be a canonical lowercase UUID matching what the pipeline itself generates, symlinked run
paths are refused rather than deleted through, a run ID with no matching directories fails with a
nonzero exit instead of silently succeeding, and `data/raw/` and the shared acquisition baseline
under `artifacts/datasets/` are never in scope regardless of the requested ID. `--dry-run`
previews the directories and bytes involved before anything is deleted. Cleanup is deliberately
manual and explicit — automatic deletion would erase useful failure evidence.

## Qualified per-stage progress reporting

Long-running local commands report progress through one shared, observational-only module
([`progress.py`](../../src/ecg_anomaly_detection/progress.py)): output never influences control
flow, artifact contents, or evidence schemas, and a reporter with no stream attached is a silent
no-op. `run-pipeline` prints start/completion banners for each of its seven stages with measured
elapsed time; acquisition adds one line per completed record — not one per file, so a 48-record
first download stays visibly active without 144 low-value lines.

Each acquisition line carries a qualified timing suffix built on `UnitTimingEstimator`: the
record's own measured duration, the phase's measured elapsed time, and an `approx. remaining`
projection. The estimator is deliberately honest about what it knows — until three records have
completed it reports an explicit `estimating...` warm-up state instead of an unstable one-sample
number, the projection is always labeled `approx.` and never presented as a deadline, and the
final record reports a factual `00:00`. Every line is flushed as written, because Python fully
block-buffers non-TTY stdout: the setup notebook streams this command through a subprocess pipe,
and without per-line flushing all output would arrive in one batch at process exit.

Restraint is part of the design. A documented timing-enrichment audit in
[pipeline orchestration](../pipeline-orchestration.md#timing-enrichment-audit-issue-199)
inventories every progress site and records why most of them deliberately do *not* carry
remaining-time projections — fast stages keep plain elapsed banners, the long model fit gets a
minute-scale heartbeat rather than a fabricated projection, and `list-runs`/`purge-run` print no
banners at all because they are near-instantaneous.

## Where to go deeper

- [Dataset acquisition](../dataset-acquisition.md) — retry policy, staging, safety and recovery
  behavior, and the acquisition trust boundary.
- [Data integrity](../data-integrity.md) — the closed 144-file inventory and the committed
  size/SHA-256 expectations every install is verified against.
- [Pipeline orchestration](../pipeline-orchestration.md) — progress output, the timing-enrichment
  audit, run output layout, and the lifecycle helpers in context.
- [Run manifests](../run-manifests.md) — the per-run evidence these operational controls protect.
- [`AGENTS.md`](../../AGENTS.md) — the tracked operating contract where the defensive
  external-call rule is codified.
