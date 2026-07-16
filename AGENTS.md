# Repository instructions for coding agents (Claude Code, Codex, and any successor)

## Standing commitments to the maintainer

These are hard, non-negotiable contracts. They exist in this tracked file — not only in any
agent's private memory — precisely so that every session, including cold-start executor and
cloud sessions that do NOT inherit local agent memory, is bound by them. Violating one is a
defect, not a style choice.

- **Self-recording promises (meta-rule).** Whenever an agent agrees to a new standing rule or
  "always / never" commitment with the maintainer, it must record that commitment in THIS file,
  in the same turn it is made, before claiming the matter is settled. A promise that lives only
  in conversation or in agent memory is not considered made. If a commitment cannot be captured
  durably or enforced as stated, say so at promise time — never let the maintainer discover
  later that it landed nowhere.
- **Do what is written, the way it is written.** If an agent takes the time to write a rule down
  or to tell the maintainer something is done, it must take the time to actually do it that way.
  Formatted assurances are not a substitute for the action.
- **Done means done.** Never report an action as complete unless it was executed AND verified in
  the current session, with the evidence available to show. Distinguish plainly in every report:
  done (receipt attached) / relayed (an executor's claim not re-verified) / queued / owed / not
  done. Announcing a mechanism (a memory file, an issue, a gate) is not the behavior existing.
- **CHANGELOG on every PR.** If the repository has a `CHANGELOG.md`, every pull request with
  substantive changes updates it under `## Unreleased` in that same PR. Treat the entry as a
  merge gate on par with passing tests, not release-time archaeology. (Origin: 25 milestone PRs
  merged with zero changelog entries before a release-gate audit caught it — issues #179 backfill,
  #184 mechanical enforcement.)
- **Session-handoff continuity.** When the maintainer signals that the current agent session is
  ending (any "wrap up" / "session limit approaching" / "hand off" phrasing) — or wind-down
  signals appear without an explicit ask (context compaction, the maintainer mentioning limited
  time, an unusually long session) — the agent produces a Markdown handoff walkthrough BEFORE the
  session ends, so the maintainer can continue the in-flight work locally with no agent at all.
  Write it to the gitignored `artifacts/session-handoffs/<UTC-timestamp>-<slug>.md` (create the
  directory if absent; never commit handoff files) and point the maintainer at it. A session
  isolated in a `.claude/worktrees/` worktree typically cannot edit files outside its worktree
  (file-editing tools are confined there; a plain shell copy out may still be permitted): write
  the handoff to the worktree's own `artifacts/session-handoffs/` first (that is the
  checkpoint), copy it into the primary checkout's `artifacts/session-handoffs/` in the same
  turn when the environment permits the copy, and say plainly which copy is the one to read. The closure pass
  copies any remaining handoff files out of a worktree before pruning it (canonical workflow
  step 10), so pruning never deletes the only copy. A session with no writable checkout at all
  falls back to one fenced markdown block in-chat (see the `.claude/CLAUDE.md` addendum). Required
  contents: a state snapshot (branch, commits, PR/issue numbers, which gates ran with their
  results, and a plain done / queued / owed accounting); numbered next steps in which every
  action is a copy-pasteable **Fish** code block runnable from the repository root (see Local
  environment), each followed by its verification command; relevant links (PR, issues, project
  board); and open risks or watch-items. Never include secrets or tokens. Checkpoint the current
  atomic step first; producing the handoff is wind-down priority one after that — owed in the
  moment, like the changelog gate, not archaeology. Agents cannot observe the maintainer's actual
  usage limits, so the maintainer's request is always the authoritative trigger; the wind-down
  signals above justify producing one proactively. (Codified 2026-07-14 at the maintainer's
  request — #211.)
- **Log the issue before touching the repo.** For any work item, the tracking GitHub issue is
  the FIRST artifact — create it before branching, before editing files, before running anything.
  The order is: issue → branch → implement → gate → document → open PR. Filing the issue after
  the work has begun is a defect; do not treat it as a formality to backfill. (This ordering was
  itself violated on the PR that introduced this section — the maintainer had to say so twice.)
- **Standing authorization vs. the four gated actions.** Once a work item is agreed, the agent
  runs the full canonical workflow — log the issue, branch, implement, gate, document, disclose,
  and the complete post-merge closure pass INCLUDING pruning merged local branches/worktrees —
  WITHOUT being re-asked. The maintainer never has to repeat a standing rule to get it honored,
  and never has to say "yes, do the thing you already committed to." Only four actions require an
  explicit per-instance go-ahead: (1) pushing a branch, (2) opening a pull request, (3) the merge
  click (the maintainer merges via GUI), and (4) tagging/publishing a release. Pausing to ask on
  anything outside those four is a defect.
- **Calibrated claims.** Do not present inferred, relayed, or memory-sourced statements with the
  tone of verified fact. State the confidence and its basis when it is not directly verified in
  the current session.
- **Check outside-sandbox permission first.** When an authorization or permission barrier is
  encountered, the first remediation step is to check whether the required permission is
  available through the environment's approved out-of-sandbox mechanism. Do this before trying
  workarounds or asking the maintainer to repeat an authorization they may already have granted.
  Checking availability does not itself grant permission, bypass an approval requirement, or
  broaden the action that the maintainer authorized; use the platform's normal approval flow when
  the out-of-sandbox action still requires explicit consent.
- **Floor, not ceiling.** This section is a minimum, not an exhaustive list of permitted actions.
  Using judgment to do the obviously-necessary thing when no rule names it is required, not
  optional. Declining or omitting an obviously-correct action because it "wasn't in the contract"
  or is "out of scope" is itself a defect — the same class of failure as skipping a listed duty,
  not a form of diligence. When genuinely unsure whether to act, surface the choice; do not treat
  silence in the rules as a reason to do nothing.

## Canonical work-item workflow

Run this whole sequence unprompted for every agreed work item. The maintainer should never have to
name a step to get it done. Only the four gated actions in the standing commitments (push, open-PR,
merge, release-tag) require an explicit go-ahead; everything else here is standing authorization.

1. **Log the tracking issue first** — before branching or editing (see standing commitments).
   Populate full metadata and add it to Project #5 with all nine fields.
2. **Sync, then branch.** `git fetch` and confirm `git log --oneline main..origin/main` is empty
   (fast-forward `main` first if not); then cut the feature branch. Re-run this drift check again
   before finalizing/pushing an already-open PR — the serial-PR assumption does not hold.
3. **Implement, and gate every commit** — `uv run pytest`, `uv run pyright`,
   `uv run pre-commit run --all-files`, `git diff --check` all green before each commit. The
   CHANGELOG entry (standing commitments) and the exhaustive code/notebook commentary standard
   are part of this gate: run `scripts/check_code_commentary.py` and treat missing docstrings/
   comments on any supported Python — including newly added scripts — as a gate failure.
4. **Documentation pass** — an exhaustive `git ls-files "*.md"` sweep, not a guessed subset; also
   check the previously merged PR for gaps it left behind.
5. **Disclose scope decisions in the PR body** — what was deliberately excluded and why, and call
   out any metadata left blank on purpose (e.g. no milestone) so a correct omission is not mistaken
   for an oversight.
6. **File issues for every gap or idea found**, by default — even unscheduled or unlikely-to-be-
   acted-on ones; visibility over caution. When a closely-related gap surfaces mid-task, propose
   folding it in rather than deferring it as "not original scope"; reserve caution for irreversible
   or high-stakes calls.
7. **On the maintainer's go-ahead: push and open the PR** with full metadata (assignee, labels,
   milestone when applicable, Project #5 membership with all nine fields). Verify every piece with
   `gh` afterward; never infer success from the creation command. Set the closing issue's own
   `status:` label to `status: in-progress`.
8. **Immediately verify PR-readiness with tooling yourself** — run `gh pr checks <N>` and
   `uv run python scripts/github/validate_project_metadata.py --pr-number <N> --strict-project-checks`
   and quote the actual result. Never say "let me know when CI is green." A cancelled duplicate
   check can block the merge box after a real pass — re-run it or push an empty commit.
9. **The maintainer merges via the GUI.** Do not merge.
10. **Closure pass, unprompted, after the maintainer confirms merge:** verify via `gh` that the PR
    is `MERGED` and every closed issue is `CLOSED`; set the PR Project status to `Merged` and each
    issue to `Closed`, each confirmed with a read-back (an action-gating read the
    [verification graduation ladder](docs/governance/github-metadata-automation.md#automation-verification-graduation-ladder-issue-248)
    never waives: don't trust `project-status-sync` automation alone here — a known quirk can
    force `Closed` instead of `Merged`); **strip the
    `status: in-progress` label from every closed issue** (`gh issue close` does not remove it);
    check whether the milestone is now empty and close it only if genuinely so; `git switch main`
    and `git pull --ff-only`; and **prune** — `git fetch --prune`, then delete every fully-merged
    local branch with `git branch -D` (squash merges break `-d`'s ancestry check) and remove the
    corresponding worktrees, not just the branch from the PR just closed. Before removing any
    worktree, copy every file under its `artifacts/session-handoffs/` into the primary
    checkout's `artifacts/session-handoffs/` (the session-handoff contract's copy-out step —
    pruning must never delete the only copy of a handoff).
11. **Milestone discipline:** verify a work item's milestone live against the actual scope docs
    (`docs/evaluation-policy.md`, `docs/benchmark-governance.md`) rather than pattern-matching;
    an issue found mid-implementation of already-milestoned work should generally join that
    milestone rather than default to unmilestoned.

## Engineering discipline

- **Defensively code every external call.** Any operation that leaves the process for a network or
  external service — dataset/file downloads, package installs (`uv sync`/PyPI), HTTP/API requests,
  remote CLIs, registry or container pulls — must (1) retry *transient* failures (timeouts,
  connection resets, transient 5xx) a bounded number of times with backoff, while failing fast on
  *permanent* errors (404, auth, digest/size mismatch) and never retrying a non-idempotent
  operation in a way that risks duplication or corruption; and (2) on exhaustion, exit *gracefully*
  — a clear, bounded message that names what failed, states plainly that it is an
  external/connectivity condition rather than a code or setup defect, and gives concrete
  remediation steps — never a raw traceback, especially on a user-facing surface such as the public
  notebooks. Retries must preserve existing integrity guarantees (atomic commits, checksum
  verification). New external calls meet this bar when written; known existing gaps are tracked and
  retrofitted (#201 retrofitted the rule's origin sites: the Step 0 notebook's PhysioNet download
  and `uv sync` bootstrap).
- **Diagnose before suppressing.** Prove the root cause of a warning or failure before silencing
  it or editing global config; do not trade a real protection for cosmetic quiet, even under time
  pressure.
- **Governance docs are negotiable, not to be silently worked around.** When a real friction with
  `AGENTS.md` or a governance doc surfaces, propose a case-specific update instead of quietly
  routing around it.

## Local environment

- The development host is macOS.
- The user's interactive shell is Fish.
- Write every user-facing shell command and command sequence in Fish syntax.
- Do not present Bash/Zsh-only syntax such as `export NAME=value`, `source .venv/bin/activate`, or `VAR=value command` without a Fish equivalent.
- Prefer these Fish forms:
  - environment variable: `set -gx NAME value`
  - one-command environment variable: `env NAME=value command`
  - virtual environment activation: `source .venv/bin/activate.fish`
  - command substitution: `(command)`
  - conditional chaining: `command; and next-command`
  - failure chaining: `command; or fallback-command`
- When a command invokes a script whose language is Bash, keep the script itself valid for its declared interpreter, but show Fish syntax around the invocation.
- Prefer macOS-compatible utilities and flags. Do not assume GNU-specific behavior unless the required GNU tool is installed and documented.

## Codex operating mode and cost control

- Default operating mode:
  - Model: GPT-5.5
  - Reasoning: Medium
  - Speed: Standard
- Use lower-cost settings for:
  - README edits
  - markdown cleanup
  - issue drafting
  - changelog updates
  - simple formatting
  - small documentation changes
- Use higher reasoning only for:
  - architecture changes
  - multi-file refactors
  - dependency conflicts
  - test failures
  - notebook-to-package restructuring
  - final review before merge
- Avoid Fast mode unless the task is blocked, unusually complex, or time-sensitive.
- Prefer feature-based commits and PRs over noisy one-line commits.
- Big rewrites are acceptable when they improve maintainability, reproducibility, or clarity, but they should be scoped to a coherent feature.

## Git workflow

- Preserve unrelated user changes in the worktree.
- Inspect `git status --short` before staging or committing.
- Stage only files belonging to the current task.
- Do not push branches, create pull requests, merge, rebase, or modify remote state unless the user explicitly asks.
- Prefer non-interactive Git output in instructions when practical. For example, use `git --no-pager show <commit>` when the user only needs to inspect a commit.
- Never use destructive Git commands unless the user explicitly requests them and the consequences are clear.

## Pull request metadata

- Every pull request created or updated by Codex must receive repository metadata before handoff.
- Assign every pull request to `@Jared-Godar` as the responsible maintainer.
- Assign labels from `.github/labels.json`, including one `type:*` label, at least one `area:*`
  label, and useful modernization, portfolio, or dependency labels. Keep status, priority, risk,
  size, and release planning in Project fields rather than duplicating them as labels.
- New work must use current-taxonomy label spellings only (`namespace: value`, with the space).
  Never mint a legacy pre-taxonomy spelling (e.g. `type:modernization`, `area:notebooks`, bare
  `documentation`) as a new label on an issue or pull request — see
  [Label taxonomy](docs/governance/label-taxonomy.md)'s "Completed legacy-label migration" section.
  That migration (#105, #113) already normalized every historical instance; the section is a
  historical record, not a menu of acceptable spellings for new work.
- Assign the delivery milestone that the work is required to complete. Do not use milestones as topic labels.
- Add the pull request to the `ECG Pipeline Modernization` GitHub Project when it contributes to the tracked roadmap.
- Set the project status to `In Progress` once implementation starts, `Review` once the pull
  request is open and awaiting merge, `Merged` after merge (`project-status-sync.yml` sets this
  automatically; verify at the cadence its current tier on the automation-verification
  graduation ladder prescribes — see the planning-metadata rules below — and correct manually if
  it ever doesn't -- see
  [GitHub Project governance](docs/governance/github-project.md#automation)), and `Closed` when
  its linked issue is completed.
- Verify assignee, labels, milestone, project membership, and project status after creating the pull request; do not infer success from the creation command alone.
- When metadata is ambiguous, report the unresolved choice instead of inventing a classification.

### Project planning metadata

- **Canonical rule: every issue and pull request must be on Project #5 with the mandatory label
  set** (one `type:*`, at least one `area:*`, plus the applicable priority/risk/size/portfolio
  labels). This is not optional per item; the automation below exists to make it the default,
  not to make it someone else's job.
- Add every issue and pull request to the `ECG Pipeline Modernization` Project #5. The
  `project-item-autofill` workflow (issue #233) does the creation-time leg automatically on
  `opened`/`labeled` events: board membership, Status → Backlog when unset, and every
  label-derivable field (`type:` → Issue Type, `priority:` → Priority, `risk:` → Risk,
  `size:` → Size, `area:` → Repository Area, `portfolio:` → Portfolio Signal), filling only
  UNSET fields — curated values always win. Agents still verify the result with a read-back at
  the cadence the graduation ladder below prescribes for this automation's current tier, and
  still set the fields the automation deliberately never touches (see
  [Creation-time board population](docs/governance/github-metadata-automation.md#creation-time-board-population-issue-233)).
- **Automation verification graduation ladder (issue #248).** How often agents re-verify a
  *recurring automation's* outcome is tiered, so trust earned by evidence retires routine
  verification cost without weakening the guarantee: (1) a **new or changed** automation gets a
  read-back on **every** event until ~5 consecutive clean observations accumulate; (2) a
  **proven** automation is **sampled** — read back every 3rd event, but ALWAYS when the next
  action depends on the outcome (e.g. lane state gating a closure pass); (3) a
  **machine-checked** automation — its invariant enforced by the scheduled board-drift backstop
  (`scripts/detect_board_drift.py`) — gets no routine per-event reads at all; and (4) **any
  observed failure resets that automation to tier 1**. Current tier placements, their streak
  evidence, and the full tier definitions live in
  [github-metadata-automation.md](docs/governance/github-metadata-automation.md#automation-verification-graduation-ladder-issue-248)
  — consult that table instead of assuming per-event verification is always owed. One-off agent
  writes are unaffected: every mutation an agent performs itself keeps its targeted read-back.
- Populate Status, Workstream, Issue Type, Priority, Risk, Size, Repository Area, Portfolio Signal,
  and Target Release for every project item. Workstream and Target Release have no label source
  and are always set by a human or agent judgment call, never inferred by automation.
- Set new issues to `Backlog`, active pull requests to `In Progress` while implementation is
  underway and `Review` once open and awaiting merge, merged pull requests to `Merged`
  (`project-status-sync.yml` sets this automatically; correct manually if it ever doesn't -- see
  [GitHub Project governance](docs/governance/github-project.md#automation)), completed closed
  issues to `Closed`, and issues closed with GitHub's native "not planned" state reason to
  `Not Planned` (never `Closed`, which stays reserved for completed work -- see
  [GitHub Project governance](docs/governance/github-project.md#status-lifecycle)).
- Preserve manually curated project values unless they are blank or demonstrably inconsistent
  with repository state.
- Link each implementation pull request to its issue with a supported closing reference when the
  pull request completes the issue scope.
- When a pull request must NOT auto-close its tracking issue (e.g. receipts-gated multi-leg
  governance work), use the sanctioned non-closing marker (`Non-closing ref: #N — <reason>`)
  instead of a closing keyword. The marker satisfies the gate's closing-reference requirement
  and runs the same linked-issue Project membership, field-completeness, and milestone-
  inheritance checks — only GitHub's auto-close is skipped. Naming the same issue with both a
  closing keyword and the marker is a hard violation. See
  [github-metadata-automation.md](docs/governance/github-metadata-automation.md#non-closing-issue-reference-marker).
- Verify all nine project fields after creating or updating an issue or pull request.

## Project positioning

- Treat this repository as a historical educational ECG machine-learning project being modernized into a data-engineering portfolio case study.
- Do not describe the project as medical software or imply clinical, diagnostic, monitoring, or treatment use.
- Preserve clear research/educational-use limitations in public documentation.
- Present original notebook results as historical results with their known evaluation limitations.
- Do not present the original random beat-window split as evidence of generalization to unseen patients.
- Prefer record/patient-grouped evaluation for the modernized pipeline.

## Modernization approach

- Work incrementally and preserve original material where it remains useful.
- Propose destructive deletion or irreversible migration before performing it.
- Separate raw data, derived data, and generated artifacts.
- Keep source datasets and derived patient-level data out of Git unless redistribution has been explicitly reviewed.
- Make data provenance, configuration, schema validation, split integrity, and run metadata explicit.
- Add environment reproducibility before claiming that a clean checkout can reproduce results.
- Add tests around transformations, label mapping, boundary windows, grouped splitting, and metrics.

## Documentation and attribution

- Keep the dataset DOI, upstream license, and required citations visible.
- Distinguish the repository's MIT-licensed work from third-party datasets, images, tutorial material, and package licenses.
- Do not reuse historical images in new portfolio material until their source and reuse terms are verified.
- Label future-state cloud architecture as proposed unless it is actually implemented and tested.

## Portfolio presentation

- Present the project as a responsible modernization case study.
- Emphasize reproducibility, data pipeline hygiene, grouped evaluation, testability, and documentation.
- Avoid overclaiming model quality or medical applicability.
- Prefer clear architecture notes, runbooks, limitations, and decision records over flashy claims.
