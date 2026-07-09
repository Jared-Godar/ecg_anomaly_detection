# Label taxonomy

The machine-readable source of truth is [`.github/labels.json`](../../.github/labels.json). Labels use
lowercase names, a colon plus one space between namespace and value, and stable descriptions. The bootstrap
script creates missing labels and updates declared colors and descriptions; it never deletes undeclared labels.

## Required assignment

Every open issue should have exactly one label from each of these dimensions:

| Dimension | Values | Use |
|---|---|---|
| `type:*` | `bug`, `documentation`, `governance`, `maintenance`, `modernization`, `technical-debt` | Kind of work |
| `priority:*` | `p0`, `p1`, `p2`, `p3` | Scheduling urgency |
| `status:*` | `triage`, `ready`, `in-progress`, `blocked`, `needs-decision` | Current workflow state |

Priority reflects repository impact, not issue size. Reserve `priority: p0` for an active security or integrity
failure requiring immediate action. Use `p1` for the next high-impact work, `p2` for normal accepted work, and
`p3` for low-urgency or opportunistic improvements.

Every issue should also have at least one `area:*` label: `ci-cd`, `cli`, `data`, `documentation`,
`evaluation`, `modeling`, `pipeline`, `quality`, `repository`, or `validation`. Multiple areas
are acceptable when the implementation genuinely spans them.

`area: pipeline` is the sole pipeline-area label; `area: data-pipeline` was its predecessor and was retired in
the #105/#113 migration pass (see below) once every carrying issue/PR was relabeled to `area: pipeline`.
`area: validation` covers schema or data-contract validation specifically, distinct from `area: quality`'s
general tests/typing/static-analysis scope.

## Contextual assignment

Add these only when they communicate useful routing or review context:

- `modernization:*` identifies historical preservation, reproducibility, split integrity, testability, or UX
  (developer-facing workflow ergonomics).
- `portfolio:*` identifies case-study presentation, portfolio-release, or operational-maturity work.
- `risk:*` flags data-integrity, evaluation, security review, or explicitly low-risk concerns.
- `size:*` estimates review surface from `xs` through `l`; it does not encode priority.
- `dependency:*` records an external, repository-setting, or maintainer-decision dependency.

GitHub automation labels `bug` and `question` may remain when useful; they supplement this taxonomy and are
not declared by the manifest. `duplicate`, `good first issue`, `help wanted`, `invalid`, and `wontfix` were
retired in the #105/#113 migration pass (zero usage, not applicable to a single-maintainer repository).

## Completed legacy-label migration (#105, #113)

The table below is a historical record of the one-time migration executed against the live repository. All
legacy pre-taxonomy label spellings have been normalized; none remain. New issues and pull requests must use
only the current-taxonomy spellings declared in `.github/labels.json` — see AGENTS.md's "Pull request
metadata" section.

Renaming a legacy label directly with `gh label edit --name` only works when no label with the target name
already exists. In this repository the canonical name was already a separate, declared label for every legacy
spelling except `modernization:ux`, so migration meant: add the canonical label to each issue/PR still carrying
the legacy one, remove the legacy label from it, then delete the now-unused legacy label once no issue
referenced it. `modernization:ux` had no existing canonical counterpart, so it was renamed directly to
`modernization: ux` (a new declared value — recurring, real developer-experience work across DX-002/003/004/005).

| Legacy label | Migrated to | Basis for the call |
|---|---|---|
| `historical-preservation` | `modernization: historical-preservation` | Direct match |
| `modernization:preservation` | `modernization: historical-preservation` | Direct match |
| `maintenance` | `type: maintenance` | Direct match |
| `dependencies` | `type: maintenance` | Per-item review: all five carrying items (issue #47, PRs #6/#44/#48/#54) were internal `uv` dependency-group upkeep or Dependabot version bumps — routine maintenance, not a dependency on an external service |
| `type:governance` | `type: governance` | Direct match |
| `type:modernization` | `type: modernization` | Direct match |
| `type:enhancement` / `enhancement` | `type: modernization` | This repository's "enhancement" history is modernization-era capability work |
| `priority:p1` / `p2` / `p3` | `priority: p1` / `p2` / `p3` | Direct match |
| `priority:p4` | `priority: p3` | The taxonomy's ladder intentionally stops at `p3`; no `p4` rung exists |
| `size:l` / `m` / `s` | `size: l` / `m` / `s` | Direct match |
| `risk:low` | `risk: low` | Direct match |
| `area:cli` | `area: cli` | Direct match |
| `area:documentation` | `area: documentation` | Direct match |
| `modernization:ux` | `modernization: ux` (new declared value) | Renamed directly — no conflicting canonical label existed |
| `area:portfolio` | `area: repository` + `portfolio: operational-maturity` | Issue #43 (GOV-007) is cross-repository governance/label alignment work — operational rigor, not narrative or release-gate |
| `area:archive` | `area: repository` | Closest existing fit; no dedicated archive value exists |
| `area:artifacts` | `area: pipeline` | Pipeline-output lifecycle |
| `area:automation` | `area: ci-cd` | Direct match |
| `area:local-experimentation` | `area: cli` | Tracks local dev/CLI tooling, not general repository upkeep |
| `area:notebooks` | `area: documentation` | Issue #37 (DX-002) establishes notebook-workspace policy/governance, not model-training work |
| `documentation` (bare GitHub default) | `type: documentation` | Only present on closed/merged pre-taxonomy `[codex]` PRs |
| `area: data-pipeline` | `area: pipeline` | `area: pipeline` was already the de facto active successor |

Zero-usage GitHub default labels were resolved per-label: `bug` and `question` were kept (useful, no
replacement needed); `duplicate`, `good first issue`, `help wanted`, `invalid`, and `wontfix` were deleted
(zero usage, not applicable to a single-maintainer repository with no external contributors).

The bootstrap script intentionally does not rename or delete labels because those operations can rewrite or
remove metadata on existing issues; this migration was executed as an explicit, maintainer-authorized one-time
pass rather than through the script.

## Bootstrap and validation

Preview deterministic commands from the repository root:

```fish
python3 scripts/sync_github_labels.py --dry-run
```

Create or update the declared labels in the current GitHub repository:

```fish
python3 scripts/sync_github_labels.py
```

Target a specific repository when the current directory is not connected to the intended remote:

```fish
python3 scripts/sync_github_labels.py --repo Jared-Godar/ecg_anomaly_detection
```

The script requires an authenticated GitHub CLI. It validates the complete manifest before making changes and
uses `gh label create --force`, so repeated runs converge on the declared names, colors, and descriptions.

This script converges the *set of labels the repository offers* on the manifest. It does not check which
labels are actually *applied* to a given issue or pull request — see
[repository hygiene automation](repository-hygiene.md#label-drift-detection) for the separate, read-only
check that does.
