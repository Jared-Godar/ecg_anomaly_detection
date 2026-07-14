# Workflow rules

Operational rules that every Kiro session must follow. These are a redundant
enforcement surface for the AGENTS.md standing commitments and canonical
workflow. AGENTS.md is the source of truth — if this file ever conflicts
with AGENTS.md, AGENTS.md wins. This file exists because agents have
demonstrated they will ignore rules they've been given unless those rules
are repeated in their face at every opportunity.

## The canonical work-item workflow (all 11 steps)

Run this whole sequence unprompted for every agreed work item. The maintainer
should never have to name a step to get it done.

1. **Log the tracking issue first** — before branching or editing. Populate
   full metadata and add it to Project #5 with all nine fields.
2. **Sync, then branch.** `git fetch` and confirm
   `git log --oneline main..origin/main` is empty (fast-forward `main` first
   if not); then cut the feature branch. Re-run this drift check again before
   finalizing/pushing an already-open PR.
3. **Implement, and gate every commit** — all gate commands green before each
   commit (see below). CHANGELOG entry and code commentary standard are part
   of this gate.
4. **Documentation pass** — exhaustive `git ls-files "*.md"` sweep, not a
   guessed subset; also check the previously merged PR for gaps.
5. **Disclose scope decisions in the PR body** — what was excluded and why,
   metadata left blank on purpose.
6. **File issues for every gap or idea found** — visibility over caution.
   Propose folding closely-related gaps in rather than deferring as
   "not original scope."
7. **On maintainer go-ahead: push and open PR** with full metadata (assignee,
   labels, milestone, Project #5 with all nine fields). Verify with `gh`.
   Set the closing issue's status label to `status: in-progress`.
8. **Verify PR-readiness yourself** — run `gh pr checks <N>` and
   `uv run python scripts/github/validate_project_metadata.py --pr-number <N> --strict-project-checks`
   and quote the result. Never say "let me know when CI is green."
9. **The maintainer merges via GUI.** Do not merge.
10. **Closure pass (unprompted after merge confirmation):** verify PR is
    `MERGED` and issues are `CLOSED` via `gh`; set Project status to `Merged`
    and issues to `Closed` (read-back verified); strip `status: in-progress`
    label from closed issues; check milestone emptiness; `git switch main`,
    `git pull --ff-only`; prune merged local branches with `git branch -D`
    and remove corresponding worktrees.
11. **Milestone discipline:** verify a work item's milestone live against
    `docs/evaluation-policy.md` and `docs/benchmark-governance.md` rather
    than pattern-matching.

## Gate commands (run before every commit)

```fish
uv run pytest
uv run pyright
uv run pre-commit run --all-files
git diff --check
```

All four must pass. A commit with any gate red is a defect.

Additionally, run `scripts/check_code_commentary.py` and treat missing
docstrings or comments on any supported Python file as a gate failure.

## CHANGELOG discipline

Every pull request with substantive changes updates `CHANGELOG.md` under
`## Unreleased` in the same PR. This is a merge gate on par with passing
tests, not release-time archaeology.

## Commit and PR hygiene

- Stage only files belonging to the current task (`git add` by name, never `-A`)
- Inspect `git status --short` before staging
- Prefer feature-based commits over noisy one-line commits
- PR titles: under 70 characters; details go in the body
- PR body must include a `Closes #N` reference when the PR completes an issue

## Gated actions (require explicit maintainer go-ahead)

1. Pushing a branch
2. Opening a pull request
3. The merge click (maintainer merges via GUI)
4. Tagging / publishing a release

Everything else in the canonical workflow is standing authorization — do it
without being re-asked. Pausing to ask on anything outside those four is a
defect.

## PR metadata checklist

Every PR must have:

- Assignee: `@Jared-Godar`
- Labels: one `type:` label, at least one `area:` label, plus relevant others
- Milestone (when applicable)
- Project #5 membership with all nine fields populated
- Closing reference in body (`Closes #N`)

Verify all metadata with `gh` after creation — never infer success from the
creation command alone.

## Label taxonomy

Use current-taxonomy spellings only (`namespace: value` with the space).
Never mint legacy pre-taxonomy spellings (e.g. `type:modernization`,
`area:notebooks`). See `docs/governance/label-taxonomy.md`.

## gh CLI authentication

The `gh` CLI authenticates via macOS Keychain. Never put tokens in environment
variables or shell history. If a permission barrier is encountered, check
whether the required scope is already available before asking the maintainer.
