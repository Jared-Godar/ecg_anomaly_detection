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

Every issue should also have at least one `area:*` label: `data-pipeline`, `documentation`, `evaluation`,
`modeling`, `quality`, or `repository`. Multiple areas are acceptable when the implementation genuinely spans
them.

## Contextual assignment

Add these only when they communicate useful routing or review context:

- `modernization:*` identifies historical preservation, reproducibility, split integrity, or testability.
- `portfolio:*` identifies case-study presentation or portfolio-release work.
- `risk:*` flags data-integrity, evaluation, or security review concerns.
- `size:*` estimates review surface from `xs` through `l`; it does not encode priority.
- `dependency:*` records an external, repository-setting, or maintainer-decision dependency.

GitHub automation labels such as `good first issue`, `help wanted`, `duplicate`, `invalid`, and `wontfix` may
remain when useful. They supplement this taxonomy and are not declared by the manifest.

## Existing-label normalization

Normalize legacy labels when applying this taxonomy:

| Existing label | Normalized label |
|---|---|
| `historical-preservation` | `modernization: historical-preservation` |
| `dependencies` | `dependency: external` or `type: maintenance`, according to meaning |
| `maintenance` | `type: maintenance` |
| `area: data-pipeline` | unchanged |
| `area: repository` | unchanged |
| `area: quality` | unchanged |

The bootstrap script intentionally does not rename or delete labels because those operations can rewrite or
remove metadata on existing issues. After confirming no conflicting labels exist, the maintainer may migrate a
legacy label with `gh label edit`, then rerun the bootstrap script.

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
