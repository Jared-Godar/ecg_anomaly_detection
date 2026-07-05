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
