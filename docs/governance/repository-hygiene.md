# Repository hygiene automation

Lightweight, deterministic automation for repository maintenance, scoped to what
is actually justified by this repository's real activity rather than every
capability a generic "repo hygiene" checklist might suggest.

## Label drift detection

`scripts/detect_label_drift.py` compares the labels applied to open issues and
pull requests against the canonical set in
[`.github/labels.json`](../../.github/labels.json), and reports any applied
label that is not in that manifest. It is read-only: it never creates, renames,
deletes, or applies a label, and never guesses a corrected name. Remediation
(migrating a legacy label with `gh label edit`, or adding a genuinely-missing
label to the manifest) remains the maintainer decision described in
[label taxonomy § Completed legacy-label migration](label-taxonomy.md#completed-legacy-label-migration-105-113).

```fish
uv run python scripts/detect_label_drift.py
```

Checks open issues and pull requests by default; pass `--include-closed` to
audit the full historical set instead. `.github/workflows/repository-hygiene.yml`
runs this weekly (Monday 06:00 UTC) and on manual dispatch, using the default
`GITHUB_TOKEN` — reading issue, PR, and label data needs no elevated
permissions, unlike the Project V2 field checks in
[GitHub metadata automation](github-metadata-automation.md).

### Exit codes and quota reporting

The exit status is the triage contract for a failed run, whether manual or
from the weekly `repository-hygiene.yml` schedule:

| Exit code | Meaning |
| --------- | ------- |
| 0 | No label drift detected. |
| 1 | Label drift detected on at least one checked item; every drifted item and its non-canonical labels are listed on stderr. |
| 2 | Execution failure (`error:` prefix on stderr) — manifest load, authentication, or `gh` CLI error. |
| 3 | GraphQL quota condition (`quota:` prefix on stderr) — transient shared-pool infrastructure, **not** label drift or a data problem; rerun after the hourly quota reset. |

The `gh issue list` / `gh pr list` reads behind this check are GraphQL-backed,
so every run draws on the shared 5000-points/hour GraphQL pool. Each run
preflights that pool and prints a consumption report to stderr on success and
failure alike, in the form
`GraphQL quota: N before, N after, N consumed (limit 5000, resets at …)`.
The preflight threshold, `--min-graphql-quota`, defaults to `0` — observe-only,
so a manual hygiene run is never blocked by a merely busy pool; raising it
makes the preflight itself stop the run with exit code 3 before any listing is
fetched. Even at the default, an actually exhausted pool encountered mid-run
still exits 3 rather than 2. If the consumption report itself cannot be
fetched, the run prints a `warning:` line and keeps its real exit code — the
best-effort report never masks the run's actual outcome. See
[GitHub metadata automation § GraphQL quota stewardship](github-metadata-automation.md#graphql-quota-stewardship)
for the shared pool's full transport and quota-spend inventory.

A label can match the canonical *formatting* (a space after the colon) while
still not being in the manifest — the check compares against the actual
manifest set, not just a style pattern. Running this for the first time
against this repository's real state surfaced 40 labels in use that were not
declared in `.github/labels.json`. Issue #67 reconciled the open/current-work
portion of that drift: the seven new-format labels in active current use
(`area: cli`, `area: pipeline`, `area: ci-cd`, `area: validation`, `area: data`,
`portfolio: operational-maturity`, `risk: low`) are declared in the manifest.
Issues #105 and #113 then reconciled the remaining historical residue: every
pre-taxonomy legacy label was relabeled to its canonical successor and
deleted, and the zero-usage unused GitHub default labels were retired (see
[label taxonomy § Completed legacy-label migration](label-taxonomy.md#completed-legacy-label-migration-105-113)
for the full record). `--include-closed` now reports zero drift as well.

## Board drift detection

`scripts/detect_board_drift.py` is the scheduled verification backstop behind the
creation-time board automation (issue #233; see
[GitHub metadata automation § Creation-time board population](github-metadata-automation.md#creation-time-board-population-issue-233)).
For every open issue and pull request (governed-bot items excluded) it flags missing
Project #5 membership, an unset Status, and any field left unset despite a label that
derives it via the shared mapping table (`scripts/github/project_label_mapping.py`). A
populated field whose value differs from its label derivation is deliberately **not**
flagged — curated values win — and full nine-field completeness stays the PR-time
metadata gate's job. Like its label sibling, it is read-only: remediation is a manual
`populate_project_item.py` run or the documented `gh` fallback commands.

```fish
uv run python scripts/detect_board_drift.py
```

`.github/workflows/repository-hygiene.yml` runs this weekly alongside the other hygiene
checks, in its own `board-drift` job using the Projects-scoped classic
`PROJECT_METADATA_TOKEN` — the board snapshot is a Projects V2 GraphQL read the default
`GITHUB_TOKEN` cannot perform for a user-owned project. It shares the label-drift
check's exit-code contract (0 clean, 1 drift, 2 execution failure, 3 quota), but its
`--min-graphql-quota` preflight defaults to `250` rather than observe-only `0`: each run
pays for one full board snapshot (~203 points measured live), so starting one on a pool
that cannot serve it would fail mid-run anyway.

## Held-out execution trigger safety

`scripts/check_held_out_trigger_safety.py` is a read-only governance-as-code check that parses
every `.github/workflows/*.yml` file and fails if any workflow whose filename or declared `name:`
matches a held-out/benchmark-execution naming convention could trigger on a routine `push` or
`pull_request` event. Such a workflow is only considered safe if its triggers are limited to
`workflow_dispatch` and/or a `push` restricted to `release-*` tags. It never runs, dispatches, or
inspects a workflow's job contents, and never touches the protected `test` partition — see
[Benchmark governance](../benchmark-governance.md).

```fish
uv run python scripts/check_held_out_trigger_safety.py
```

No held-out/benchmark-execution workflow exists yet, so this currently passes trivially. Its
purpose is to guard against one being added later — when a future execution command (issue #73)
actually gets a workflow file — with an unsafe trigger that could run against a routine push or
pull request. `.github/workflows/repository-hygiene.yml` runs this weekly (Monday 06:00 UTC)
and on manual dispatch, alongside label drift detection.

## Stale issue and pull request handling: declined

The originating issue for this automation framed it as conditional: add
automation "if justified by repository activity." Stale issue/PR bots
(auto-labeling or auto-closing items inactive for N days) are not implemented,
because this repository's actual observed activity does not justify one:

- Single maintainer, no external-contributor backlog to triage.
- Issues and pull requests in this repository's real history are opened and
  closed same-day or within a few days; there is no accumulation of
  long-dormant open items for a stale bot to act on.
- An intentionally-open item can be a deliberate state, not neglect — for
  example an issue held open pending a documented maintainer decision. A
  stale bot cannot distinguish "forgotten" from "deliberately pending" and
  would risk auto-closing the latter.

If repository activity changes (multiple contributors, a growing backlog of
genuinely dormant items), this is worth revisiting — but adding it now would
be automation solving a problem this repository does not currently have.
