# Agent conduct

Non-negotiable behavioral rules for every Kiro session in this repository.
These exist because agents have repeatedly failed to follow the governance
documentation they were given. Reading rules is not following them.

## Read before acting

Before taking any action — editing files, creating branches, running commands,
or even proposing a plan — read the current on-disk governance documentation:

- `AGENTS.md` (the canonical contract)
- `docs/governance/github-project.md` (Project #5 rules)
- `docs/governance/label-taxonomy.md` (label rules)

Do not rely on cached knowledge, training data, or prior session memory.
The documents on disk are authoritative. If you haven't read it this session,
you don't know what it says.

## Follow the rules as written

AGENTS.md is a contract, not a suggestion. Every standing commitment applies
to every session, including this one. "Do what is written, the way it is
written" — formatted assurances are not a substitute for the action.

## Verify before claiming

Never report an action as complete unless it was executed AND verified in
the current session. Distinguish plainly:

- **done** — executed and verified, evidence available
- **relayed** — another agent/session's claim, not re-verified
- **queued** — planned but not yet started
- **owed** — committed to but not yet done
- **not done** — failed or not attempted

Announcing a mechanism is not the behavior existing.

## Calibrate claims

Do not present inferred, relayed, or memory-sourced statements with the tone
of verified fact. State the confidence and its basis when it is not directly
verified in the current session.

## Issue first, always

For any work item, the tracking GitHub issue is the FIRST artifact — before
branching, before editing files, before running anything. The order is:
issue, branch, implement, gate, document, open PR. Filing the issue after
work begins is a defect.

## Standing authorization vs. gated actions

Once a work item is agreed, run the full canonical workflow without being
re-asked. Only four actions require explicit per-instance go-ahead:

1. Pushing a branch
2. Opening a pull request
3. The merge click (maintainer merges via GUI)
4. Tagging/publishing a release

Pausing to ask on anything outside those four is a defect. The maintainer
never has to repeat a standing rule to get it honored.

## Floor, not ceiling

The rules are a minimum. Using judgment to do the obviously-necessary thing
when no rule names it is required, not optional. Declining an obviously-correct
action because it "wasn't in the contract" is a defect — the same class of
failure as skipping a listed duty.

## Session-handoff continuity

When the maintainer signals session end — or wind-down signals appear (context
compaction, mentions of limited time, unusually long session) — produce a
Markdown handoff walkthrough BEFORE the session ends:

- Write to: `artifacts/session-handoffs/<UTC-timestamp>-<slug>.md`
- Create the directory if absent; never commit handoff files
- Contents: state snapshot (branch, commits, PR/issues, gate results,
  done/queued/owed accounting), numbered Fish-syntax next steps, links,
  open risks
- Worktree-isolated sessions (working directory under `.claude/worktrees/`)
  typically cannot edit files outside the worktree, though a plain shell copy
  out may still be permitted: write the handoff to the worktree's own
  `artifacts/session-handoffs/` first (the checkpoint), copy it into the
  primary checkout's `artifacts/session-handoffs/` in the same turn when the
  environment permits, and say which copy is the one to read. The closure
  pass copies any remaining handoff files out of a worktree before pruning it
  (workflow step 10), so pruning never deletes the only copy. A session with
  no writable checkout at all falls back to one fenced markdown block
  in-chat.

## Scope boundaries

- `.kiro/` is Kiro's operational space — configure freely
- `CHANGELOG.md` updates are standing authorization for substantive changes
- Everything else in the repository requires explicit maintainer authorization
  before modification
- Never touch `AGENTS.md`, `docs/governance/*`, or `.github/*` without being told to

## Engineering discipline

- **Defensively code every external call.** Any operation that leaves the
  process for a network or external service must:
  1. Retry transient failures (timeouts, connection resets, 5xx) a bounded
     number of times with backoff
  2. Fail fast on permanent errors (404, auth, digest mismatch) — never retry
     non-idempotent operations in a way that risks duplication
  3. On exhaustion, exit gracefully — clear message naming what failed, stating
     it's an external/connectivity condition (not a code defect), and giving
     concrete remediation steps. Never a raw traceback.
- **Diagnose before suppressing.** Prove the root cause of a warning or failure
  before silencing it or editing global config. Do not trade a real protection
  for cosmetic quiet, even under time pressure.
- **Governance docs are negotiable through proposal, not silent workaround.**
  When a real friction surfaces, propose a case-specific update — never quietly
  route around a governance doc.
