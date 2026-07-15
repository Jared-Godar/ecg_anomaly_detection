# GitHub Project governance

The public [ECG Pipeline Modernization project](https://github.com/users/Jared-Godar/projects/5)
is the planning and tracking surface for repository modernization, governance, developer
experience, release readiness, and long-term stewardship.

GitHub issues remain the source of truth for work scope and acceptance criteria. Milestones group
work by repository maturity. Project fields carry execution and portfolio metadata without
overloading repository labels.

## Required fields

Every issue and pull request added to the project must populate these fields:

| Field | Values |
|---|---|
| Status | Backlog, Ready, In Progress, Blocked, Review, Validation, Merged, Closed, Not Planned |
| Workstream | Acquisition, Pipeline, Evaluation, Governance, Developer Experience, Documentation, Release, Stewardship |
| Issue Type | Feature, Enhancement, Governance, Documentation, Technical Debt, Bug, Investigation, Research |
| Priority | Critical, High, Medium, Low |
| Risk | Low, Medium, High |
| Size | XS, S, M, L, XL |
| Repository Area | acquisition, provenance, manifests, validation, splitting, evaluation, modeling, reproducibility, documentation, governance, notebooks, ci-cd, developer-experience |
| Portfolio Signal | Reproducibility, Operational Maturity, Governance, Data Engineering, Documentation, Developer Experience, Systems Thinking, Ownership, Testing Rigor, Agentic Engineering |
| Target Release | Modernization Foundation, Portfolio Release, Stewardship, Future |

Use the most specific defensible value. Leave uncertain metadata blank and request maintainer
review instead of manufacturing precision.

The Repository Area option `notebooks` has no matching `area: notebooks` label — notebook-surface
work carries `area: documentation` on the label side; see the
[notebook-surface label mapping](label-taxonomy.md#notebook-surface-label-mapping-207) in the
label taxonomy.

The label taxonomy and the board's option sets are deliberately not isomorphic. The 2026-07-15
alignment audit (#237) reviewed every gap in both directions and recorded a per-label decision:
five area labels (`cli`, `data`, `pipeline`, `quality`, `repository`) are permanently human-set
on the board side, the label-less Repository Area options remain finer-grained board-only routing
values, and `portfolio: governance` was minted for the pre-existing Governance signal option. See
the [label-to-board-field alignment](label-taxonomy.md#label-to-board-field-alignment-237)
section of the label taxonomy for the full decision table.

### Portfolio Signal boundaries (#210)

The 2026-07-14 portfolio-signal audit (#210) added `Testing Rigor` and `Agentic Engineering` to
the Portfolio Signal single-select. Their boundaries against the pre-existing options:

- **Testing Rigor** applies when the test suite, coverage, or verification design **is the
  item's subject**. CI pipeline plumbing remains Operational Maturity.
- **Agentic Engineering** applies when agent contracts, instruction files, or agent-workflow
  enforcement are the subject. Agent-**authored** work on other subjects does not qualify.

Both options were added through the project web UI (never via `updateProjectV2Field` — see the
option-set rewrite warning in [Setting Status via the CLI](#setting-status-via-the-cli)) and
their IDs re-verified afterwards with `gh project field-list 5 --owner Jared-Godar`; the
pre-existing option IDs were confirmed unchanged by the same read. The field's full option-ID
table (field ID `PVTSSF_lAHOAQEwMM4BcY39zhXJtrU`):

| Portfolio Signal | Option ID |
|---|---|
| Reproducibility | `130e719a` |
| Operational Maturity | `cb3db70f` |
| Governance | `afd2bfcf` |
| Data Engineering | `7f1675f7` |
| Documentation | `1fe3d55b` |
| Developer Experience | `3eb76ac0` |
| Systems Thinking | `f70713a0` |
| Ownership | `27243f96` |
| Testing Rigor | `d52a7c36` |
| Agentic Engineering | `5486739f` |

The matching contextual labels are `portfolio: testing-rigor` and `portfolio: agentic-engineering`
(see the [label taxonomy](label-taxonomy.md#portfolio-signal-extension-210), which also records
the deliberately deferred third candidate, CI supply-chain hardening / meta-CI). The 2026-07-15
alignment audit (#237) later minted `portfolio: governance` for the pre-existing Governance
option — a label addition only, requiring no board change (see the
[label-to-board-field alignment](label-taxonomy.md#label-to-board-field-alignment-237) section).

### Bundling pull requests

A pull request commonly closes more than one issue (e.g. PR #160 closing #158 and #159). When its
closing issues disagree on a field's value, apply this convention rather than deciding it fresh
per PR:

- **Milestone** tracks the pull request's own shipping vehicle, not any individual closing issue's
  categorization. A milestone is a delivery commitment, not a topic
  (`docs/governance/issue-workflow.md`) -- an issue bucketed under a broad `Target Release` like
  Stewardship is still milestoned once it concretely ships as part of milestoned work. Confirmed
  live: #158, #159, #161, and #162 all carry milestone `v1.1.0 -- Second Portfolio Release` because
  all four ship through the same PR cluster, regardless of how each issue was individually
  triaged.
- **Target Release** (the Project field) likewise tracks the pull request's actual shipping
  vehicle rather than being copied from a closing issue's value. Precedent: PR #160 set Target
  Release to `Portfolio Release`, matching its own `v1.1.0` milestone, rather than inheriting
  #158's own `Stewardship` value -- the PR's release timing is concretely known at merge time in a
  way an individual issue's own bucket isn't.
- **Workstream**, **Issue Type**, and **Portfolio Signal** are single-select fields, so a bundling
  PR can only carry one value each -- there is no structural way to union them across disagreeing
  closing issues the way a multi-valued label could. Use the closing issue that represents the
  PR's primary or largest piece of work, judged first by relative Size (the larger-Size issue
  wins), falling back to which issue the PR's own title and summary substantively center on when
  Size is equal or one issue isn't yet sized.
  - Not "first-listed closing issue wins": the order `Closes #N` references appear in a PR body is
    a typing artifact, not a substantive signal, and using it would make the outcome depend on
    which issue the author happened to reference first rather than which one the PR is actually
    mostly about.
  - Not "majority of closing issues wins": bundling PRs in this repo often pair several small,
    narrowly-scoped issues with one larger piece of work, and counting occurrences would let three
    XS issues outvote the one L issue that represents the bulk of the change.

This convention governs future bundling PRs; it does not retroactively correct the already-merged
field values of PR #155 or PR #160 (see #162).

## Status lifecycle

Use this progression for new work:

![State diagram of seven statuses in order: Backlog, Ready, In Progress, Validation, Review, Merged, Closed. A Blocked state branches from In Progress and returns to it. Arrow labels name the trigger for each transition, such as a linked pull request moving work to In Progress.](../diagrams/exports/governance-status-lifecycle.svg)

*The ECG Pipeline Modernization board's status lifecycle: the progression every
work item follows, with Blocked as a temporary excursion from In Progress.* —
generated from
[`governance-status-lifecycle.dot`](../diagrams/src/governance-status-lifecycle.dot)
via Graphviz (see [`docs/diagrams/design-spec.md`](../diagrams/design-spec.md)).

- New project items begin in Backlog.
- A linked implementation pull request moves work to In Progress.
- Validation covers tests, documentation, metadata, and final evidence checks.
- Review means the deliverable is ready for maintainer review.
- Merged describes a merged pull request or an issue awaiting final closure.
- Closed describes a completed, closed issue.
- Not Planned (#208) describes an issue closed with GitHub's native "not planned" state reason —
  withdrawn, superseded, or deliberately declined work. It is used **only** for issues carrying
  that state reason; `Closed` stays reserved for completed issues, and pull request lanes are
  unaffected. The native state reason carries the distinction on the issue itself, but not in
  board views or field-based filters — this lane makes withdrawn work visibly distinct from
  delivered work on the board. It is a terminal excursion available from any lane at withdrawal
  time, not a step in the delivery progression, which is why the lifecycle diagram above
  deliberately continues to show only the delivery path.

## Historical record

The initial backfill populated all 34 merged modernization pull requests and nine open backlog
issues. Historical pull requests use their original milestones, labels, bodies, and native
Created/Closed timestamps to retain the delivery sequence. Their project Status is Merged; open
issues begin in Backlog.

Historical metadata is a conservative reconstruction. Explicit labels and milestones take
precedence over title/body interpretation. Do not overwrite a manually curated project value
unless it is blank or demonstrably inconsistent with repository state.

## Views

Project views should support both retrospective review and future execution:

| View | Layout | Configuration |
|---|---|---|
| Active Work Board | Board | Group by Status; exclude Merged and Closed |
| Modernization Roadmap | Roadmap | Group by Milestone; sort Priority descending |
| Governance Dashboard | Table | Filter Workstream = Governance |
| Developer Experience | Table | Filter Workstream = Developer Experience |
| Portfolio Signals | Table | Group by Portfolio Signal |
| Active Backlog | Table | Exclude Merged and Closed; sort Priority descending |
| Stewardship Dashboard | Table | Filter Target Release = Stewardship |
| Validation and Release | Table | Filter Evaluation, Release, or Reproducibility work |
| Delivery History | Table | Filter Status = Merged; group by Milestone; sort Closed ascending |

GitHub's public Project V2 API does not currently expose supported mutations for saved views or
workflow actions. Configure those elements in the web interface and verify them after changes.

## Automation

The project is linked to this repository. Built-in workflows should provide these transitions:

| Trigger | Status |
|---|---|
| Item added | Backlog |
| Pull request linked to issue | In Progress |
| Pull request merged | Merged |
| Issue closed | Closed |

In practice, a merged pull request's item has consistently landed at `Closed` rather than `Merged`
-- merging also closes the pull request, so both the `Pull request merged` and `Item closed`
workflows fire on the same event, and `Closed` appears to win. Confirmed live via the project's
Workflows panel that both are enabled; the public API does not expose each workflow's configured
target or execution order. Reordering or disabling one of the two built-in workflows is also not
exposed as a supported mutation.

Rather than rely on the built-in workflows to resolve that race, `.github/workflows/project-status-sync.yml`
listens for `pull_request: closed` and, when `github.event.pull_request.merged == true`, calls
[`scripts/github/set_merged_project_status.py`](../../scripts/github/set_merged_project_status.py)
to explicitly set the item's Status to `Merged` after the built-in workflows have already run,
winning the race deterministically instead of leaving `Closed` as the final value. Originally
tracked in #100, superseded by #117.

The built-in `Issue closed` workflow does not distinguish closure state reasons, so an issue
closed as "not planned" initially lands in `Closed` like any other closed issue; move it to
`Not Planned` manually afterward with a read-back check (see the CLI section below).
`project-status-sync.yml` is unaffected by the `Not Planned` lane — it targets pull-request
merge events only, never issue closures.

Since issue #233, `.github/workflows/project-item-autofill.yml` also automates the
creation-time leg: on `issues`/`pull_request` `opened`/`labeled` events it adds the item to
the board when absent, defaults Status to Backlog when (and only when) Status is unset, and
fills unset fields derivable from the item's labels, converging as labels land — see
[Creation-time board population](github-metadata-automation.md#creation-time-board-population-issue-233)
for the mapping table, precedence rules, bot/fork exclusions, and the manual fallback. It never
overwrites a populated field and never regresses a Status lane.

Priority, Risk, Size, Repository Area, Issue Type, and Portfolio Signal are automated only as
label *mirrors*: the reviewable engineering judgment lives in assigning the taxonomy labels
(docs/governance/label-taxonomy.md), and the automation transcribes exactly that decision onto
the board, leaving ambiguous or unlabeled cases blank. Workstream and Target Release have no
label source and are never inferred — heuristic assignment remains intentionally avoided for
exactly the original reason: a confidently wrong value reads as deliberate triage.

## Setting Status via the CLI

The Status field is writable through `gh project item-edit` (confirmed by round-trip test: moved
an item to a different option, verified via a fresh read, reverted). `project-status-sync.yml`
automates this for the merged-pull-request case; every other transition (Backlog, Ready, In
Progress, Blocked, Review, Validation, and Closed on issue closure) is still set manually, and
this is also the fallback if that workflow ever fails. Field and option IDs are
internal GraphQL identifiers specific to this project; re-verify them with
`gh project field-list 5 --owner Jared-Godar` if this table stops matching the live schema.

```fish
set -l project_id PVT_kwHOAQEwMM4BcY39
set -l status_field_id PVTSSF_lAHOAQEwMM4BcY39zhXCFmM

# Find an item's ID. This full item-list read is the ONE board-wide discovery
# snapshot a logical batch of work is allowed (see GraphQL quota stewardship in
# github-metadata-automation.md): take it once, cache every item ID and current
# value you need from it, and never repeat it per item or per mutation.
gh project item-list 5 --owner Jared-Godar --format json --limit 500 \
  | jq '.items[] | select(.content.number == <ISSUE_OR_PR_NUMBER>) | .id'

# Targeted read-back: read exactly the mutated item's field via node(id:) --
# one GraphQL point regardless of board size, where a repeated item-list scan
# pays full-board pagination on every verification (the issue #173 fix).
# Prints the option name, "unset" when the field has no value, and nothing at
# all when the item itself no longer exists.
function project_status_read_back --argument-names item_id
    gh api graphql \
        -f query='query($item: ID!) { node(id: $item) { ... on ProjectV2Item {
            fieldValueByName(name: "Status") {
                ... on ProjectV2ItemFieldSingleSelectValue { name } } } } }' \
        -f item=$item_id \
        --jq 'if .data.node == null then empty
              else (.data.node.fieldValueByName.name // "unset") end'
end

# Specific to this document's Project #5: item-edit takes the project's node ID,
# but item-list takes its number and owner, so all three identifiers must be
# changed together if this pattern is ever adapted to another project.
function set_project_status_verified \
        --argument-names project_id item_id field_id option_id expected_value
    for attempt in 1 2
        gh project item-edit --id $item_id --field-id $field_id --project-id $project_id \
            --single-select-option-id $option_id

        # A fresh targeted read is authoritative even when gh reports "no changes to make".
        set -l observed (project_status_read_back $item_id)
        # An item absent from the read-back entirely (empty output) is not the
        # same as an unset field: the requested state is unknowable, so stop.
        if test -z "$observed"
            echo "Item $item_id not found in the Project #5 read-back" >&2
            return 1
        end
        if test "$observed" = "$expected_value"
            return 0
        end

        if test $attempt -eq 2
            echo "Project field verification failed: expected '$expected_value', got '$observed'" >&2
            return 1
        end
        echo "Field read back as '$observed', not '$expected_value'; retrying the mutation once" >&2
    end
end

# Apply fields sequentially; call this once per field before starting the next.
set_project_status_verified $project_id <ITEM_ID> $status_field_id <OPTION_ID> <EXPECTED_STATUS>
```

Run Project field mutations sequentially, never as one back-to-back batch. After each mutation,
read the mutated item again -- via the targeted `node(id:)` read-back above, not another full
`item-list` -- and compare the observed value with the exact requested value before starting the
next field. The function above demonstrates the Status field; use the same two-attempt bound and
the corresponding `fieldValueByName` name for every other single-select field.
In particular, `error: no changes to make` is not proof that a value was already present: it is
an inconclusive mutation result until a fresh read-back confirms the requested value. A
conclusive `gh` failure (an authentication error, a wrong field or option ID) still prints its
own stderr between the retries -- stop and fix that instead of re-running; only
`no changes to make` is the inconclusive case (`set_merged_project_status.py` enforces this
distinction explicitly by failing fast on every other error class).

The read-back-verified mutation *rule* is unchanged by issue #173's quota hardening; only the
read's *scope* narrowed (one item, one field) so that verification cost no longer scales with
board size. Quota thresholds, consumption reporting, and the shared-pool recovery procedure are
documented in [GraphQL quota
stewardship](github-metadata-automation.md#graphql-quota-stewardship).

| Status | Option ID |
|---|---|
| Backlog | `2bb9dbb0` |
| Ready | `010387a6` |
| In Progress | `ac9c56d8` |
| Blocked | `61c602c6` |
| Review | `94bf0d2e` |
| Validation | `eed5658a` |
| Merged | `d9e041e5` |
| Closed | `2f81c115` |
| Not Planned | `d4be655b` |

These IDs are one generation newer than the pre-#208 set: adding the `Not Planned` option through
the `updateProjectV2Field` GraphQL mutation regenerated every option ID in the field, so none of
the previously documented values survive (see the warning below and the incident record on
PR #209).

> **Warning — never add or edit options via `updateProjectV2Field`.** The mutation's
> `singleSelectOptions` argument *replaces* the field's entire option set, and GitHub regenerates
> the ID of **every** option — including options resent unchanged, by their exact current names.
> Because items store their single-select values by option ID, this orphans the stored Status
> value of every item on the board in one call (observed live during issue #208's board leg:
> all 208 items read back `unset`). Add or rename options through the project's web UI instead,
> which preserves existing IDs. If a rewrite ever happens anyway: take no further field-level
> action, and restore from the most recent pre-mutation `item-list` snapshot using the per-item
> read-back-verified mutation loop above — item-level `gh project item-edit` writes are safe and
> were used for the verified 209/209 restore on 2026-07-14. Name-resolving automation
> (`set_merged_project_status.py`, `project-status-sync.yml`) is unaffected by ID regeneration;
> only ID-carrying documentation and cached scripts go stale.

Completeness of the required fields above is checked automatically on every pull request; see
[Automated pull-request metadata gate](github-metadata-automation.md#automated-pull-request-metadata-gate).
