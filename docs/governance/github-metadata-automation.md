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

- **Pull request level**: an assignee, at least one `type:*` label, at least one `area:*` label,
  and a body closing reference (`Closes #N`, `Fixes #N`, `Resolves #N`, and their keyword
  variants) to an issue. A milestone is also required, unless every issue the pull request closes
  is itself deliberately unmilestoned (per [issue workflow](issue-workflow.md)'s rule that a
  milestone is a delivery commitment, not a mandatory tag) — the check reads each closing issue's
  own milestone field and inherits that decision rather than forcing an unrelated milestone onto
  the pull request.
- **Linked issue level**: for every issue number extracted from a closing reference, that the
  issue is a member of the tracked Project and has every field in
  [Required fields](github-project.md#required-fields) populated.

### Token requirement and rollout

Reading Project V2 field values requires a token with the `project` scope. The default
repository-scoped `GITHUB_TOKEN` a workflow receives does not have that scope for a user-owned
project (the same limitation recorded above for the historical bootstrap). A repository secret
named `PROJECT_METADATA_TOKEN` supplies it.

That token must be a **classic** personal access token, not a fine-grained one. GitHub's
fine-grained tokens do not currently expose a Projects permission for a project owned by a user
account at all (only for organization-owned projects) — this is a documented platform limitation,
not a configuration mistake; see GitHub's own [personal access tokens
documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
and [Projects API guide](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-api-to-manage-projects).
Project #5 here is user-owned (`Jared-Godar`), so a classic token is the only supported option.
Scope it to `project` (read and write — needed by
[`project-status-sync.yml`](../../.github/workflows/project-status-sync.yml), which reuses this
same secret to explicitly set a merged pull request's Status field, see [GitHub Project
governance](github-project.md#automation)) plus `public_repo` (this repository is public; grants
the read access to pull requests and issues this gate itself needs) plus `read:org`. The last one
is easy to miss: every `gh project` subcommand resolves `--owner` by querying both the user and
organization GraphQL types, and a token without `read:org` fails that resolution with a
misleading `unknown owner type` error — indistinguishable at a glance from an actual ownership
problem, even though Project #5's owner (`Jared-Godar`, a user account) is correct (confirmed
live: see [cli/cli#7985](https://github.com/cli/cli/issues/7985) and
[cli/cli#8885](https://github.com/cli/cli/issues/8885)). Prefer these three narrow scopes over
the broader `repo` scope.

The workflow passes `--strict-project-checks`: an unreadable Project (missing or misconfigured
token) is a hard failure, not an advisory warning. The `GITHUB_TOKEN` fallback in the workflow's
`env` block exists only so a removed or rotated-out secret degrades to a clear authentication error
in the job log rather than the workflow silently not running; it is not a soft-enforcement mode.
Pull-request-level checks (assignee, milestone, labels, closing reference) and linked-issue-level
checks (Project membership and field completeness) both enforce immediately once
`PROJECT_METADATA_TOKEN` is configured.

### Local credential handling

`PROJECT_METADATA_TOKEN` is a GitHub Actions secret and is consumed only in CI. Never
materialize its value to a file on a local machine, even inside a gitignored directory such as
`secrets/` — a gitignored file is still plaintext on disk, readable by any local process, script,
backup, or tool with filesystem access, not just Git. Nothing in `scripts/` or `docs/` reads a
local token file, so there is never a reason to create one.

Running `scripts/github/validate_project_metadata.py`,
`scripts/github/set_merged_project_status.py`, or any ad hoc `gh project`/`gh pr`/`gh issue`
command locally instead needs only an interactive `gh auth login` session with the `project`
scope (`gh auth status` reports the active scopes; add it with `gh auth refresh -s project` if
it's missing). That session-based token is managed by the `gh` CLI's own credential storage, not
a file this repository's tooling ever touches.

Marking this workflow as a required status check in branch protection is a separate, manual
repository-settings decision, not configured by the workflow itself. As of #91, `Validate PR and
linked-issue metadata` is one of the required status checks on `main` — see [repository
governance](repository-governance.md#current-branch-protection-on-main) for the current
configuration.

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
