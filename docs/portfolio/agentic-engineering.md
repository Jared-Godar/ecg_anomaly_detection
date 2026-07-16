# Agentic Engineering

Much of this repository's modernization was implemented by coding agents — and that is precisely
why its development process is itself engineered and enforced like the pipeline code. Rather than
trusting any one tool's session memory, the repository binds every coding-agent session to a
tracked, model-agnostic operating contract: [`AGENTS.md`](../../AGENTS.md). This page expands the
README's Agentic Engineering signal into the fuller story — what each mechanism is, why it exists,
and where it lives in the repository.

The premise throughout: an agent's promise, plan, or claim is worthless unless it is written to a
tracked file, enforced by a gate, or verified with evidence in the current session. Nearly every
mechanism below exists because a failure of the softer alternative was observed first.

## One contract, every model

[`AGENTS.md`](../../AGENTS.md) is addressed, in its own title, to "coding agents (Claude Code,
Codex, and any successor)." It is a tracked repository file — not vendor configuration — so a
cold-start executor or cloud session that inherits no local memory is still bound by it the moment
it reads the checkout. The contract opens with a block of standing commitments it describes as
"hard, non-negotiable contracts," where "violating one is a defect, not a style choice," and goes
on to codify the canonical work-item workflow, engineering discipline (defensive external calls,
diagnose-before-suppressing), Fish-shell command conventions, PR metadata requirements, and the
project-positioning rules that keep this repository honestly framed as an educational case study.

Tool-specific files exist only as thin adapters over that single source of truth:
[`.claude/CLAUDE.md`](../../.claude/CLAUDE.md) adds Claude Code mechanics (commands, architecture
context, worktree-specific handoff handling) and explicitly defers to `AGENTS.md` for the
operating rules, while the tracked [`.kiro/steering/`](../../.kiro/steering/workflow-rules.md)
files do the same for Kiro sessions. `workflow-rules.md` states the design plainly: it is "a
redundant enforcement surface" and "if this file ever conflicts with AGENTS.md, AGENTS.md wins."

## Self-recording promises

The contract's first standing commitment is a meta-rule about itself: whenever an agent agrees to
a new standing rule, it must record that commitment in `AGENTS.md` in the same turn it is made —
"a promise that lives only in conversation or in agent memory is not considered made." Several
commitments in the file carry their origin stories inline: the per-PR CHANGELOG gate was codified
after a release audit found 25 milestone PRs merged with zero changelog entries
(issues #179, #184), and
the issue-first rule notes it was violated on the very PR that introduced it.

Companion rules keep agent reporting honest. "Done means done" requires that nothing be reported
complete unless executed and verified in the current session, with every status stated plainly as
done, relayed, queued, owed, or not done. "Calibrated claims" forbids presenting inferred or
memory-sourced statements with the tone of verified fact. The same rules are restated for Kiro in
[`.kiro/steering/agent-conduct.md`](../../.kiro/steering/agent-conduct.md).

## Standing authorization with four gated actions

The contract resolves the usual agent-autonomy tension in both directions at once. Once a work
item is agreed, the agent runs the entire canonical workflow unprompted — log the tracking issue
first, sync and branch, implement with every commit gated (`pytest`, `pyright`, pre-commit, `git diff --check`, a
code-commentary check, and the CHANGELOG entry), run an exhaustive documentation pass, disclose
scope decisions in the PR body, file issues for every gap found, and execute the full post-merge
closure pass. Pausing to ask permission for any of these is itself classified as a defect.

Exactly four actions require an explicit per-instance go-ahead: pushing a branch, opening a pull
request, the merge click (the maintainer merges via GUI), and tagging or publishing a release.
The boundary keeps irreversible, externally visible actions under human control while removing
the "yes, do the thing you already committed to" ritual from everything else. A closing "floor,
not ceiling" rule blocks the inverse failure mode: declining an obviously correct action because
"it wasn't in the contract" is treated as the same class of defect as skipping a listed duty.

## Metadata gates enforced in CI, not trusted to memory

Process rules that stay documentation-only decay; this repository converts them into required
merge gates. [`metadata-governance.yml`](../../.github/workflows/metadata-governance.yml) runs
[`validate_project_metadata.py`](../../scripts/github/validate_project_metadata.py) on every pull
request: the PR must carry an assignee, `type:*` and `area:*` labels, an inherited-or-waived
milestone, and a closing reference to an issue — and that issue must be a member of GitHub
Project #5 with all nine planning fields populated (Status, Workstream, Issue Type, Priority,
Risk, Size, Repository Area, Portfolio Signal, Target Release). A second job in the same workflow
mechanically enforces the per-PR CHANGELOG contract. Branch protection runs with `enforce_admins`
enabled, so the gates block the maintainer's own merges too; the one documented override path
requires independently proven infrastructure failure, recorded in a PR comment before use.

The gate itself is engineered, not scripted: distinct exit codes separate
metadata violations from unreadable data and from shared-quota exhaustion, GraphQL consumption is
budgeted and reported per run (the board snapshot was measured live at 203 points), and
bot-authored dependency PRs get a compensated path — their own board membership and nine-field
completeness — instead of a silent exemption. The board itself carries an `Agentic Engineering`
Portfolio Signal option, with a documented boundary: it applies when agent contracts and
enforcement are the subject of the work, not merely because an agent authored it. Full detail
lives in [GitHub metadata automation](../governance/github-metadata-automation.md) and
[GitHub Project governance](../governance/github-project.md).

## Session-handoff continuity

Agent sessions end — quotas run out, context compacts, machines sleep — and the contract treats
that as an operational event with a required artifact. On any wind-down signal, the agent must
write a Markdown handoff walkthrough to the gitignored `artifacts/session-handoffs/` directory
before the session ends, containing a state snapshot (branch, commits, PR and issue numbers, gate
results, a done/queued/owed accounting), numbered next steps where every action is a
copy-pasteable Fish code block with its verification command, relevant links, and open risks. The
stated goal is that the maintainer "can continue the in-flight work locally with no agent at all."
The rule extends to worktree-isolated sessions (write inside the worktree, copy out before the
worktree is pruned) and to sessions with no writable checkout (fall back to a fenced in-chat
block). Like the CHANGELOG gate, the handoff is owed in the moment, not reconstructed later.

## Durable context for a multi-agent future

Everything an agent needs to operate correctly here is tracked in Git: `AGENTS.md`,
`.claude/CLAUDE.md`, and the `.kiro/steering/` set —
[`project-context.md`](../../.kiro/steering/project-context.md) (baseline repository facts, so
sessions do not start from stale training-data assumptions),
[`agent-conduct.md`](../../.kiro/steering/agent-conduct.md) (behavioral rules and scope
boundaries), [`workflow-rules.md`](../../.kiro/steering/workflow-rules.md) (the eleven-step
canonical workflow and gate commands), plus manual-inclusion references for Project #5 field IDs
and a [dispatch seed-prompt template](../../.kiro/steering/seed-prompt-template.md) for handing
work to a cold-start session. Tracked `.kiro/hooks/` definitions re-inject the contract
mechanically — a session-start bootstrap that requires reading the governance docs on disk before
any action, a standing-contract reminder re-issued on every user prompt, and a pre-commit
reminder that names the four gate commands.

The steering files also encode hard scope boundaries: agents never touch `AGENTS.md`,
`docs/governance/*`, or `.github/*` without explicit authorization, and governance friction is
resolved by proposing a documented update — never by silently routing around the rule.

## Where to go deeper

- [`AGENTS.md`](../../AGENTS.md) — the operating contract itself: standing commitments, canonical
  workflow, gated actions
- [GitHub metadata automation](../governance/github-metadata-automation.md) — the CI gates,
  bot-PR policy, creation-time board autofill, and GraphQL quota stewardship
- [GitHub Project governance](../governance/github-project.md) — the nine required fields, status
  lifecycle, and the Agentic Engineering signal boundary
- [Governance index](../governance/index.md) — the full governance documentation set
- [`.kiro/steering/workflow-rules.md`](../../.kiro/steering/workflow-rules.md) — the redundant
  enforcement surface pattern in full
