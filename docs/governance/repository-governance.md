# Repository governance

## Scope and ownership

This repository is maintained by `@Jared-Godar`. It is a modernization,
reproducibility, and repository operations case study based on a historical
educational ECG machine-learning project. It is not production ML, medical or
clinical software, or healthcare AI.

The committed `CODEOWNERS` file assigns default ownership of all repository
content to the maintainer. More-specific ownership rules can be added later
without changing the default. GitHub uses this file to identify responsible
reviewers; whether code-owner review is required is controlled separately by
repository settings.

## Branch and pull-request workflow

`main` is the protected, authoritative branch. Changes should be developed on
short-lived topic branches and merged through pull requests. Direct pushes to
`main` are not part of the normal workflow.

Each pull request should:

- describe its purpose, scope, and validation;
- preserve unrelated historical material and user changes;
- pass all required CI checks;
- be updated with `main` before merge when required by branch protection; and
- have all review conversations resolved.

Accepted work begins with the applicable issue form and follows the documented
[issue workflow](issue-workflow.md). The [label taxonomy](label-taxonomy.md) provides consistent type,
priority, status, area, modernization, portfolio, risk, size, and dependency metadata.

The repository uses squash merges only. Each pull request therefore becomes one
coherent commit on `main`, providing linear history while allowing iterative
work on its topic branch. Merge commits and rebase merges should remain disabled,
and merged topic branches should be deleted automatically.

## CI and review model

Required CI checks provide the repeatable evidence needed before merge. A pull
request must not merge while a required check is pending or failing. The exact
required checks are selected in the `main` branch protection or ruleset settings
and should match the checks emitted by the committed workflows.

This is a single-maintainer repository. The maintainer is accountable for
self-reviewing each pull request's diff, validation evidence, and documentation
before merge. CODEOWNERS makes that accountability explicit, but it does not
create an independent reviewer. Where GitHub permits the pull-request author to
merge without a separate approval, passing CI and documented self-review are the
minimum evidence. External review can be requested when risk or scope warrants it.

New commits should dismiss stale approvals so review evidence applies to the
current revision. All review conversations should be resolved before merge,
including comments raised during self-review or by automated tooling.

## Enforcement boundary

Committed files document ownership and the expected workflow. GitHub branch
protection, allowed merge methods, automatic branch deletion, and
required-review behavior are configured in repository settings, not enforced
by `CODEOWNERS` or this document alone. The current configuration is recorded
below; re-verify it periodically against `gh api repos/Jared-Godar/
ecg_anomaly_detection/branches/main/protection` rather than trusting this
document to stay current on its own.

## Current branch protection on `main`

Applied 2026-07-08 (#91), verified live against the API above:

- **Required status checks** (strict — the branch must be up to date before
  merge): `Locked environment and tests`, `Pre-commit checks`, `Secret scan`,
  `Build package artifacts`, `Execute curated notebooks without the full
  dataset`, `Validate PR and linked-issue metadata`.
- **`Detect label drift` is intentionally not a required check.** Its
  workflow triggers only on `schedule` and `workflow_dispatch`, never
  `pull_request`, so it structurally cannot post a status to a pull
  request's head commit — marking it required would make every pull request
  permanently unmergeable. It also fails by design until #67 is resolved.
  This is a deliberate exclusion, not an oversight.
- **Pull requests are required before merge; 0 approvals are required.**
  This repository has one maintainer (`@Jared-Godar`), who is both author and
  would-be reviewer on every pull request, and GitHub does not allow a pull
  request author to approve their own pull request — requiring any approval
  would make every pull request permanently unmergeable. Requiring the
  pull-request mechanism itself (branch, required checks, review surface,
  conversation resolution) without a second-approver requirement is the
  enforceable ceiling for a single-maintainer repository; self-review per
  the CI and review model above remains the actual review discipline.
- **Enforced for administrators**: the repository owner is bound by the same
  rules as any other contributor; there is no merge-button bypass. A narrow,
  documented, audited exception exists for a required check failure
  independently proven to be pure infrastructure rather than a defect the
  pull request introduced — see [GitHub metadata
  automation](github-metadata-automation.md#maintainer-override-for-confirmed-infrastructure-failures)
  (#157). `enforce_admins` returns to `true` immediately after each use; this
  is not a standing bypass.
- **Linear history is required.** Merge commits cannot be merged into
  `main`. In practice every merge to date has also been a squash merge
  (verified: `git log --merges` shows no two-parent commits on `main`), but
  see the known residual gap below.
- **Force pushes and deletion of `main` are both disabled.**
- **Conversation resolution is required** before merge.
- **`.github/CODEOWNERS` assigns `@Jared-Godar`** as default owner of all
  repository content; confirmed present and correctly scoped.

### Known residual gap

Resolved 2026-07-09 (#98): the repository-level merge-method settings
(`allow_merge_commit`, `allow_rebase_merge`) were both still enabled even
though squash-only was the documented and, so far, actual practice. Branch
protection's linear-history requirement blocked true two-parent merge
commits, but did not by itself disable the rebase-merge option in the GitHub
merge-button UI. Applied via `gh api -X PATCH repos/Jared-Godar/
ecg_anomaly_detection -f allow_merge_commit=false -f allow_rebase_merge=false`
and confirmed live: `allow_merge_commit` and `allow_rebase_merge` are now
both `false`; `allow_squash_merge` remains `true`. Squash is now the only
merge method GitHub's UI and API will accept for this repository.
