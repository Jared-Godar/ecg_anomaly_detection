# ADR 0001 — Automated Dependabot PR closure

- **Status:** Accepted (2026-07-31)
- **Deciders:** Jared Godar (maintainer)
- **Tracking:** #278
- **Related:** #193 (governed-bot exemption), #184 (metadata gate), `docs/governance/github-metadata-automation.md`

## Context

Dependabot PRs accumulate stale — sometimes a week or more — even when the update is
trivially safe, and even when it is a **security patch**. Two independent causes:

1. **Mechanical.** The metadata-governance gate requires every PR (bots included) to
   carry an assignee and `type:*`/`area:*` labels. Those are supplied by each ecosystem's
   `labels`/`assignees` block in `.github/dependabot.yml`. An ecosystem with **no block**
   — including one that only ever appears via Dependabot *security* updates — produces PRs
   with no labels and no assignee, which fail the gate and go red. PR #274 (jupyterlab
   4.6.1→4.6.2, a 5-CVE security patch in the `uv` ecosystem, which had no block) is the
   canonical instance: it sat red for a week purely on missing metadata.
2. **Effort.** Even a green Dependabot PR waits on a human to judge repo-wide/doc impact
   and click merge. The maintainer typically discovers these only by accident, long after
   they land.

The result the maintainer wants to eliminate: *stumbling onto a week-old security patch
that could have been closed automatically.*

## Decision

Introduce an end-to-end automated response to Dependabot PRs, and — for a bounded, explicit
tier — **take the human out of the merge loop**:

1. **Metadata at the source.** Every PR-opening ecosystem in `.github/dependabot.yml`
   carries `labels` + `assignees`, so PRs arrive gate-compliant. (Phase 1, this ADR's PR:
   adds the missing `uv` block.)
2. **Audit.** An isolated Claude agent reviews each Dependabot PR for repo-wide impact and
   any documentation surface the bump touches.
3. **Comment** the audit findings on the PR.
4. **Commit** fixes for any gap the audit finds, then re-wait for CI green on that commit.
5. **Squash-merge** when the guardrails below all hold.
6. **Notify** via SNS + GitHub: "here is what was fixed automatically."

### Merge guardrails (all five required for autonomous merge)

1. author is `dependabot[bot]`
2. bump is patch or minor (major → held for the maintainer)
3. changed files ⊆ {lockfiles, `.pre-commit-config.yaml`, `.github/workflows/*` action
   pins, `CHANGELOG.md`}
4. no tracked doc pins the bumped version
5. no security-audit hook (zizmor / ruff) reports a **new** finding

Any miss → the agent comments what is needed and holds for the maintainer. Every autonomous
merge emits a notification. This is a **bounded waiver** of the standing "maintainer merges
via the GUI on a green light" rule (AGENTS.md work-item workflow step 9), scoped to this
tier only; it does not extend to human-authored PRs or to the automation's own
infrastructure PRs.

## Security rationale — why the audit runs OUTSIDE privileged CI

`.github/workflows/dependabot-autofill.yml` carries a hard SECURITY INVARIANT: it runs on
`pull_request_target` (write-capable secrets in scope) and is safe **only because it never
checks out or executes PR-head content**. The audit must *read* the PR's changed files —
which, if done in that privileged context, would reintroduce remote-code-execution with a
write PAT in scope. Therefore the Claude audit runs as an **isolated agent** (a scheduled
cloud routine) with its own scoped credentials, never inside the privileged CI job. The
mechanical, no-head-execution tagging stays in on-event CI. This preserves the invariant
intact rather than punching an exception through it.

## Consequences

- Safe Dependabot updates — including security patches — land within a cycle, not a week.
- The metadata gate stays fully strict for human PRs; nothing is weakened, the config is
  completed.
- Committing to a Dependabot branch stops Dependabot's own auto-rebase — acceptable because
  merge follows immediately.
- Every autonomous merge spends agent tokens on the audit; acceptable and capped.

## Alternatives considered

- **Claude in-CI (event-driven, instant).** Possible via a split-privilege dance
  (unprivileged read → artifact → privileged act). Rejected as default: materially more
  complex and it re-litigates the deliberately-drawn `pull_request_target` invariant. The
  isolated-routine latency (hours) still eliminates the week-old-stall problem.
- **Weaken the metadata gate for bots.** Rejected: the gate is correct; the `dependabot.yml`
  config was incomplete. Completing the config preserves 100% of the gate's intent.
- **Keep the human in the merge loop.** Rejected by the maintainer: the whole point is to
  close a visible security gap without depending on the maintainer noticing.

## Rollout

Piloted in `ecg_anomaly_detection` (highest Dependabot traffic + the metadata gate), then
ported to the other portfolio repos, which share this AGENTS.md lineage; phases are tracked
in issue #278.
