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

Automatic priority, risk, size, and portfolio assignments are intentionally avoided. Those fields
require reviewable engineering judgment.

Completeness of the required fields above is checked automatically on every pull request; see
[Automated pull-request metadata gate](github-metadata-automation.md#automated-pull-request-metadata-gate).
