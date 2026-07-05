# Repository instructions for Codex

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
- Assign labels from `.github/labels.json`, including one `type:*` label, at least one `area:*` label, and any useful modernization, portfolio, risk, size, or dependency labels.
- Assign the delivery milestone that the work is required to complete. Do not use milestones as topic labels.
- Add the pull request to the `ECG Pipeline Modernization` GitHub Project when it contributes to the tracked roadmap.
- Set the project status to `In Progress` for an open pull request and `Done` after it is merged or closed as completed.
- Verify assignee, labels, milestone, project membership, and project status after creating the pull request; do not infer success from the creation command alone.
- When metadata is ambiguous, report the unresolved choice instead of inventing a classification.

### Project planning metadata

- Add every issue and pull request to the `ECG Pipeline Modernization` Project #5.
- Populate Status, Workstream, Issue Type, Priority, Risk, Size, Repository Area, Portfolio Signal,
  and Target Release for every project item.
- Set new issues to `Backlog`, active pull requests to `In Progress`, merged pull requests to
  `Merged`, and completed closed issues to `Closed`.
- Preserve manually curated project values unless they are blank or demonstrably inconsistent
  with repository state.
- Link each implementation pull request to its issue with a supported closing reference when the
  pull request completes the issue scope.
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
