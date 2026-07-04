# Issue workflow

## Purpose

Issues define accepted repository work for this historical educational ECG project and its modernization
case study. They are not requests for medical, clinical, diagnostic, monitoring, or treatment functionality.
Never attach source ECG data, derived patient-level data, credentials, or generated model artifacts.

## Create and triage

Choose the narrowest issue form: bug report, documentation update, governance change, modernization task,
or technical debt. Blank issues are disabled so each proposal captures reproducible evidence, scope, and
acceptance criteria.

New forms receive one `type:*` label and `status: triage`. During triage, the maintainer:

1. confirms the issue is not a duplicate and is consistent with the project framing;
2. makes the outcome, non-goals, acceptance criteria, and data boundary explicit;
3. assigns one `priority:*`, one `status:*`, at least one `area:*`, and relevant thematic labels;
4. assigns a milestone only when the work is required for that milestone; and
5. adds accepted work to the `ECG Pipeline Modernization` project when project-level tracking is useful.

Use `M3 — Baseline modeling and evaluation` for accepted baseline or grouped-evaluation work and
`M4 — Portfolio release` for release-readiness work. A milestone is a delivery commitment, not a topic.

## Status transitions

The normal flow is `status: triage` → `status: ready` → `status: in-progress` → closed. Use
`status: needs-decision` when a recorded maintainer choice is required. Use `status: blocked` only with a
comment naming the blocker and the condition for resuming work. Closed issues do not retain a status label.

An issue becomes `status: ready` only when its acceptance criteria and boundaries are implementable. Assign
`status: in-progress` when a branch or pull request is active, and link the pull request. Pull requests should
close issues with a GitHub closing keyword when all acceptance criteria are satisfied.

## Scope and review

Keep issues small enough to produce one coherent pull request. Split unrelated outcomes into linked issues.
Large modernization work may use a tracking issue, but each child issue must remain independently testable.

Validation evidence must be proportional to risk. Pipeline and evaluation changes should cover provenance,
schema validation, boundary behavior, record or patient grouping, metrics, and run metadata as applicable.
Governance-only changes must state that pipeline, model, dataset, evaluation, CI, and repository-setting
behavior remains unchanged, or identify any exception explicitly.

See [Label taxonomy](label-taxonomy.md) for assignment rules and
[Repository governance](repository-governance.md) for pull-request and merge expectations.
