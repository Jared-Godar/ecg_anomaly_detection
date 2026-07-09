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
| Status | Backlog, Ready, In Progress, Blocked, Review, Validation, Merged, Closed |
| Workstream | Acquisition, Pipeline, Evaluation, Governance, Developer Experience, Documentation, Release, Stewardship |
| Issue Type | Feature, Enhancement, Governance, Documentation, Technical Debt, Bug, Investigation, Research |
| Priority | Critical, High, Medium, Low |
| Risk | Low, Medium, High |
| Size | XS, S, M, L, XL |
| Repository Area | acquisition, provenance, manifests, validation, splitting, evaluation, modeling, reproducibility, documentation, governance, notebooks, ci-cd, developer-experience |
| Portfolio Signal | Reproducibility, Operational Maturity, Governance, Data Engineering, Documentation, Developer Experience, Systems Thinking, Ownership |
| Target Release | Modernization Foundation, Portfolio Release, Stewardship, Future |

Use the most specific defensible value. Leave uncertain metadata blank and request maintainer
review instead of manufacturing precision.

## Status lifecycle

Use this progression for new work:

```text
Backlog -> Ready -> In Progress -> Validation -> Review -> Merged -> Closed
                         |
                         +-> Blocked
```

- New project items begin in Backlog.
- A linked implementation pull request moves work to In Progress.
- Validation covers tests, documentation, metadata, and final evidence checks.
- Review means the deliverable is ready for maintainer review.
- Merged describes a merged pull request or an issue awaiting final closure.
- Closed describes a completed, closed issue.

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

Automatic priority, risk, size, and portfolio assignments are intentionally avoided. Those fields
require reviewable engineering judgment.

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

# Find an item's ID:
gh project item-list 5 --owner Jared-Godar --format json --limit 200 \
  | jq '.items[] | select(.content.number == <ISSUE_OR_PR_NUMBER>) | .id'

gh project item-edit --id <ITEM_ID> --field-id $status_field_id --project-id $project_id \
  --single-select-option-id <OPTION_ID>
```

| Status | Option ID |
|---|---|
| Backlog | `f75ad846` |
| Ready | `1e06bed4` |
| In Progress | `47fc9ee4` |
| Blocked | `044cc57b` |
| Review | `34b40d2e` |
| Validation | `54cd2334` |
| Merged | `03b2e725` |
| Closed | `98236657` |

Completeness of the required fields above is checked automatically on every pull request; see
[Automated pull-request metadata gate](github-metadata-automation.md#automated-pull-request-metadata-gate).
