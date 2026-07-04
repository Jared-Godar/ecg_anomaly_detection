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
protection or rulesets, allowed merge methods, automatic branch deletion, and
required-review behavior are configured in repository settings. They are not
enforced by `CODEOWNERS` or this document alone. Repository settings must be
configured and periodically checked against the validation checklist below.

## PR-ready validation checklist

Before merging this governance change, confirm:

- [ ] `main` branch protection or an equivalent ruleset is enabled.
- [ ] Pull requests are required before merge.
- [ ] Required CI status checks are enabled.
- [ ] Branches are required to be up to date before merge.
- [ ] Stale approvals are dismissed when new commits are pushed.
- [ ] Review conversations must be resolved before merge.
- [ ] Force pushes to `main` are blocked.
- [ ] Deletion of `main` is blocked.
- [ ] Linear history is required.
- [ ] Merge commits are disabled.
- [ ] Rebase merges are disabled.
- [ ] Squash merging is enabled and retained as the only merge method.
- [ ] Automatic deletion of head branches after merge is enabled.
- [ ] GitHub recognizes `.github/CODEOWNERS` and assigns `@Jared-Godar`.
- [ ] CI passes on the pull request.
