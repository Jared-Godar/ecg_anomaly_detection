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

Every issue should also have at least one `area:*` label: `ci-cd`, `cli`, `data`, `data-pipeline`,
`documentation`, `evaluation`, `modeling`, `pipeline`, `quality`, `repository`, or `validation`. Multiple areas
are acceptable when the implementation genuinely spans them.

`area: pipeline` is the active successor to `area: data-pipeline` in current use (DX-00x and PIPE-00x work);
`area: data-pipeline` is retained as declared because deprecating it is a maintainer-directed migration (see
below), not a manifest change. `area: validation` covers schema or data-contract validation specifically,
distinct from `area: quality`'s general tests/typing/static-analysis scope.

## Contextual assignment

Add these only when they communicate useful routing or review context:

- `modernization:*` identifies historical preservation, reproducibility, split integrity, or testability.
- `portfolio:*` identifies case-study presentation, portfolio-release, or operational-maturity work.
- `risk:*` flags data-integrity, evaluation, security review, or explicitly low-risk concerns.
- `size:*` estimates review surface from `xs` through `l`; it does not encode priority.
- `dependency:*` records an external, repository-setting, or maintainer-decision dependency.

GitHub automation labels such as `bug`, `good first issue`, `help wanted`, `duplicate`, `invalid`, `question`,
and `wontfix` may remain when useful. They supplement this taxonomy and are not declared by the manifest.

## Existing-label normalization

Normalize legacy labels when applying this taxonomy. Renaming a legacy label directly with `gh label edit` only
works when no label with the target name already exists; where the canonical name is already a separate,
declared label (the common case below, per the #67 reconciliation), migration instead means: add the canonical
label to each issue/PR still carrying the legacy one, remove the legacy label from it, then delete the
now-unused legacy label once no issue references it. That per-issue relabeling is a deliberate maintainer
action, not something the bootstrap or drift-detection scripts perform.

| Existing label | Normalized label | Confidence |
|---|---|---|
| `historical-preservation` | `modernization: historical-preservation` | Confident |
| `modernization:preservation` | `modernization: historical-preservation` | Confident |
| `maintenance` | `type: maintenance` | Confident |
| `dependencies` | `dependency: external` or `type: maintenance`, according to meaning | Confident (per-item judgment) |
| `type:governance` | `type: governance` | Confident |
| `type:modernization` | `type: modernization` | Confident |
| `type:enhancement` / `enhancement` | `type: modernization` (this repository's "enhancement" history is modernization-era capability work) | Confident |
| `priority:p1` / `p2` / `p3` | `priority: p1` / `p2` / `p3` | Confident |
| `priority:p4` | `priority: p3` (the taxonomy's ladder intentionally stops at `p3`; no `p4` rung exists) | Needs maintainer confirmation |
| `size:l` / `m` / `s` | `size: l` / `m` / `s` | Confident |
| `risk:low` | `risk: low` | Confident |
| `area:cli` | `area: cli` | Confident |
| `area:documentation` | `area: documentation` | Confident |
| `modernization:ux` | No existing dimension value fits developer-experience work; either declare `modernization: ux` as a new value (recurring, real use across DX-002/003/004/005) or force-fit into `modernization: testability` | Needs maintainer decision |
| `area:portfolio` | `area: repository`, and consider an additional `portfolio:*` contextual label depending on the specific work | Needs maintainer decision |
| `area:archive` | `area: repository` (closest existing fit; no dedicated archive value exists) | Needs maintainer decision |
| `area:artifacts` | `area: pipeline` (pipeline-output lifecycle) | Needs maintainer decision |
| `area:automation` | `area: ci-cd` | Needs maintainer decision |
| `area:local-experimentation` | `area: cli` or `area: repository` | Needs maintainer decision |
| `area:notebooks` | `area: modeling` or `area: documentation`, according to whether the work is model-facing or narrative-facing | Needs maintainer decision |
| `documentation` (bare GitHub default) | `type: documentation` | Confident, but only present on closed/merged pre-taxonomy `[codex]` PRs |
| `area: data-pipeline` | `area: pipeline` (see note above) | Needs maintainer decision |

The bootstrap script intentionally does not rename or delete labels because those operations can rewrite or
remove metadata on existing issues. After confirming no conflicting labels exist, the maintainer may migrate a
legacy label with `gh label edit`, then rerun the bootstrap script. Where a conflict does exist (the common case
above), the per-issue relabel-then-delete procedure applies instead, and deletion of the emptied legacy label
is always a separate, explicit maintainer action.

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
