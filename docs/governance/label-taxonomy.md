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
- `portfolio:*` identifies case-study presentation, portfolio-release, operational-maturity,
  testing-rigor, agentic-engineering, or governance work (see the
  [portfolio-signal extension](#portfolio-signal-extension-210) for the 2026 additions and the
  [alignment audit](#label-to-board-field-alignment-237) for `portfolio: governance`).
- `risk:*` flags data-integrity, evaluation, security review, or explicitly low-risk concerns.
- `size:*` estimates review surface from `xs` through `l`; it does not encode priority.
- `dependency:*` records an external, repository-setting, or maintainer-decision dependency.

Since issue #233, taxonomy labels are also the machine source for Project #5 board fields: the
`project-item-autofill` workflow mirrors `type:`/`priority:`/`risk:`/`size:`/`area:`/`portfolio:`
labels onto the corresponding board fields via the explicit table in
`scripts/github/project_label_mapping.py`, filling only unset fields — see
[Creation-time board population](github-metadata-automation.md#creation-time-board-population-issue-233).
Assigning these labels is therefore the reviewable judgment step; the board transcription is
mechanical.

GitHub automation labels `bug` and `question` may remain when useful; they supplement this taxonomy and are
not declared by the manifest. `duplicate`, `good first issue`, `help wanted`, `invalid`, and `wontfix` were
retired in the #105/#113 migration pass (zero usage, not applicable to a single-maintainer repository).

## Portfolio-signal extension (#210)

The 2026-07-14 portfolio-signal audit minted two additional contextual `portfolio:*` labels,
each paired with a matching Portfolio Signal board option (see the
[Portfolio Signal boundaries](github-project.md#portfolio-signal-boundaries-210) for the
board-side option IDs and the full boundary definitions):

| Label | Applies when |
|---|---|
| `portfolio: testing-rigor` | The test suite, coverage, or verification design is the item's subject; CI pipeline plumbing stays `portfolio: operational-maturity` |
| `portfolio: agentic-engineering` | Agent contracts, instruction files, or agent-workflow enforcement are the subject; agent-*authored* work on other subjects does not qualify |

Both remain contextual per this section's rule — never required, never auto-assigned.

A third audit candidate, CI supply-chain hardening / meta-CI (SHA-pinned action refs, zizmor,
tested workflow-safety scripts), was deliberately **not** minted: the existing near-orphan
options (Systems Thinking, Ownership) showed that a signal with too few carrying items dilutes
the taxonomy rather than sharpening it. That work stays mapped to
`portfolio: operational-maturity`. Revisit trigger: reconsider a dedicated signal once a
hardening workstream accumulates roughly eight or more items whose primary subject it is.

## Notebook-surface label mapping (#207)

There is deliberately no `area: notebooks` label, even though the ECG Pipeline Modernization
board's Repository Area single-select field does offer a `notebooks` option (see
[GitHub Project governance](github-project.md#required-fields)). The two surfaces map like this:

| Surface | Value for notebook-focused work |
|---|---|
| Issue/PR labels (this taxonomy) | `area: documentation` |
| Project board Repository Area field | `notebooks` |

The label side treats the supported public notebooks as part of the repository's documentation
surface — consistent with the #105/#113 migration, which folded the legacy `area:notebooks`
spelling into `area: documentation` (see the migration table below) rather than minting a
current-taxonomy notebooks label. The board field keeps the finer-grained `notebooks` value
because Repository Area is a single-select routing field, not a label, and costs nothing to keep
specific. Precedent: PRs #202 and #205 both carry `area: documentation` labels with `notebooks`
as their board Repository Area; an attempted `area: notebooks` label on PR #205 failed PR
creation because the label does not exist.

Do not mint an `area: notebooks` label ad hoc when a PR hits that failure — apply
`area: documentation` and set the board field to `notebooks`. If the asymmetry ever becomes a
real friction, propose adding the label to `.github/labels.json` through the normal label-sync
workflow instead of creating it directly on an issue or PR.

## Label-to-board-field alignment (#237)

The taxonomy's `area:*`/`portfolio:*` labels and the board's Repository Area / Portfolio Signal
option sets are deliberately not isomorphic. The 2026-07-15 alignment audit (#237) reviewed every
gap in both directions and recorded a per-label decision, so no remaining asymmetry is an
oversight. For these two namespaces the creation-time automation (#233) derives only the exact
same-named pairs listed in `scripts/github/project_label_mapping.py` (other namespaces carry
documented non-identity mappings — see the
[metadata-automation summary](github-metadata-automation.md#creation-time-board-population-issue-233));
everything below stays human-set on the board.

Five area labels have no unambiguous Repository Area translation and are **permanently
human-set** on the board side:

| Label | Why no option maps |
|---|---|
| `area: cli` | CLI work spans the `developer-experience` option and the stage option of whichever subcommand is touched; no single option captures the CLI surface |
| `area: data` | The board carves data handling finer: `acquisition`, `provenance`, `manifests`, and `splitting` are separate options the umbrella label cannot pick among |
| `area: pipeline` | Pipeline work spans every stage option; the board deliberately offers no umbrella option |
| `area: quality` | No counterpart at all: the board's `validation` option means data/schema validation (a pipeline stage), not this label's tests/typing/static-analysis scope |
| `area: repository` | Splits between `governance` and `developer-experience` depending on the item (live examples: issues #236 and #237 both carry it with a human-set Repository Area of `governance`) |

The label-less Repository Area options (`acquisition`, `provenance`, `manifests`, `splitting`,
`reproducibility`, `governance`, `notebooks`, `developer-experience`) remain board-only routing
values — the single-select field costs nothing to keep specific, while a label for each would
dilute the taxonomy (the #210 minting discipline). `notebooks` keeps its dedicated mapping rule
(see the [notebook-surface label mapping](#notebook-surface-label-mapping-207) above).

On the portfolio side, the audit minted one label: **`portfolio: governance`**, mapped to the
board's pre-existing Governance option. Demand evidence, per the #210 minting bar: the signal was
carried by 57 board items at audit time (second only to Operational Maturity's 59), and the
filing of #237 itself bounced on the missing label. A label addition reshapes nothing — the
option and its ID are unchanged. `portfolio: case-study` and `portfolio: release` stay unmapped:
they mark narrative/lifecycle context, not one of the board's signal dimensions.

Four other label-less Portfolio Signal options also exceed the #210 item-count bar
(Documentation at 38 items, Developer Experience at 19, Reproducibility at 17, Data Engineering
at 13) but have produced no filing friction, so the audit did not mint labels for them; the
remaining two, Systems Thinking (4 items) and Ownership (5), sit below the bar as the near-orphan
options the #210 section above already documents. As with the #210 additions and the governance
label itself, the revisit trigger is demonstrated friction: when a filing actually wants one of
these signals as a label, propose it through the normal label-sync workflow (#241 is the
standing record).

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
