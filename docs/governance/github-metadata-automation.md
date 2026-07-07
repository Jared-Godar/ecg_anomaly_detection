# GitHub metadata automation record

This document records the programmatic repository and Project V2 metadata work used to establish
the modernization backlog. It provides representative, auditable examples without duplicating
complete issue bodies or treating generated command output as maintained documentation.

## Scope

The bootstrap process:

- verified the authenticated GitHub account and repository remote;
- created missing labels and verified existing milestones;
- created nine issues from a reviewed intake catalog without exact-title duplication;
- preserved the complete issue-body structure during creation;
- expanded Project #5 with structured planning fields and options;
- added every repository issue to the project;
- backfilled nine open issues and 34 historical pull requests; and
- verified every item had complete required project metadata.

The issue catalog was intentionally not retained. GitHub issues are now authoritative, and keeping
a second copy of every issue body would create an immediate drift risk.

## Idempotent issue creation

Automation checks all issue states by exact title before creation. A representative flow is:

```text
parse reviewed catalog
-> validate title, labels, milestone, and body sections
-> list open and closed issues
-> skip an exact-title match
-> create only missing labels and milestones
-> create the issue
-> read it back and compare title, body, labels, milestone, number, and URL
```

Exact label names matter. For example, `type:modernization` and `type: modernization` are distinct
GitHub labels. Validation compares normalized label sets rather than assuming visually similar
names are equivalent.

## Project V2 field update

Single-select fields are discovered by name and options by exact value. Existing item values are
preserved unless blank or clearly inconsistent with GitHub state. Updates use field and option
node IDs returned by Project V2 queries.

A representative mutation shape is:

```graphql
mutation UpdatePlanningField(
  $project: ID!
  $item: ID!
  $field: ID!
  $option: String!
) {
  updateProjectV2ItemFieldValue(
    input: {
      projectId: $project
      itemId: $item
      fieldId: $field
      value: {singleSelectOptionId: $option}
    }
  ) {
    projectV2Item {
      id
    }
  }
}
```

Authentication tokens, raw API responses, local paths, and temporary operational scripts are not
committed.

## Deterministic mapping precedence

Historical mappings use this evidence order:

1. GitHub item state and merge state
2. Explicit repository labels
3. Assigned milestone
4. Issue or pull-request identifier and title
5. Body language and acceptance criteria

Examples:

| Evidence | Project value |
|---|---|
| Closed issue | Status = Closed |
| Merged pull request | Status = Merged |
| Open issue without linked implementation | Status = Backlog |
| `risk: evaluation` | Risk = High |
| `size: m` | Size = M |
| M5 developer-experience issue | Workstream = Developer Experience |
| Reproducibility evidence deliverable | Portfolio Signal = Reproducibility |

When multiple values remain plausible, automation leaves the field blank for maintainer review.

## Validation

After mutation, validation reads the project back and checks:

- expected fields and exact options exist;
- every current issue is a project item;
- item URLs and GitHub numbers resolve correctly;
- open issues and merged pull requests have coherent statuses;
- every item contains all required planning fields; and
- repository source files remain unchanged by metadata-only operations.

The initial result contained 43 project items and 387 populated required field values.

## API limitations

The supported `gh project` commands and public Project V2 GraphQL mutations can manage fields,
options, items, and item values. They do not currently provide supported creation or editing of
saved project views and built-in workflow actions. Those settings require web-interface review.

This limitation is why view definitions and expected workflow transitions are maintained in
[GitHub Project governance](github-project.md).

## Automated pull-request metadata gate

`.github/workflows/metadata-governance.yml` runs `scripts/github/validate_project_metadata.py`
on every pull request event, converting the field requirements above from documentation-only
guidance into an enforced check. The script and workflow are deliberately separate: the script
takes no GitHub Actions-specific input (only `--pr-number`, `--repo`, `--owner`, and
`--project-number`), so it runs identically from a terminal for local debugging.

```fish
uv run python scripts/github/validate_project_metadata.py --pr-number 65
```

The check validates two layers:

- **Pull request level**: an assignee, a milestone, at least one `type:*` label, at least one
  `area:*` label, and a body closing reference (`Closes #N`, `Fixes #N`, `Resolves #N`, and their
  keyword variants) to an issue.
- **Linked issue level**: for every issue number extracted from a closing reference, that the
  issue is a member of the tracked Project and has every field in
  [Required fields](github-project.md#required-fields) populated.

### Token requirement and rollout

Reading Project V2 field values requires a token with the `project` scope. The default
repository-scoped `GITHUB_TOKEN` a workflow receives does not have that scope for a user-owned
project (the same limitation recorded above for the historical bootstrap). Configure a repository
secret named `PROJECT_METADATA_TOKEN` — a fine-grained personal access token or GitHub App
installation token with `project` (read) and `contents`/`pull-requests` (read) access — to enable
the linked-issue-level checks.

Until that secret exists, the workflow falls back to `GITHUB_TOKEN`, the Project read fails, and
the script prints a warning and skips the linked-issue-level checks rather than failing the PR —
the pull-request-level checks above still enforce immediately regardless. Pass
`--strict-project-checks` (not set in the current workflow) once the secret is configured and
verified working, to make an unreadable Project a hard failure instead of an advisory warning.
This lets PR-level enforcement start immediately while Project-field enforcement is opted into
deliberately, rather than landing as an unexpectedly broken required check on every future PR the
moment this workflow merges.

Marking this workflow as a required status check in branch protection is a separate, manual
repository-settings decision and is not configured by the workflow itself.

### Why issue creation is not blocked

GitHub provides no rejection mechanism for issue creation comparable to a required pull-request
status check. Issue-only metadata gaps (an issue with no open pull request yet) remain a
manual-review concern; enforcement here starts at the pull-request stage, where a required check
can actually block a merge.

### Tests

`tests/scripts/test_validate_project_metadata.py` covers the pure validation logic (closing-
reference extraction, pull-request-level checks, linked-issue field-completeness checks) directly,
and the `gh`-subprocess boundary with mocked process output. It does not call the network; running
it requires no GitHub token.
