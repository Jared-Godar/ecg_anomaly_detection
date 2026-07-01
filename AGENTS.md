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

## Git workflow

- Preserve unrelated user changes in the worktree.
- Inspect `git status --short` before staging or committing.
- Stage only files belonging to the current task.
- Do not push branches, create pull requests, merge, rebase, or modify remote state unless the user explicitly asks.
- Prefer non-interactive Git output in instructions when practical. For example, use `git --no-pager show <commit>` when the user only needs to inspect a commit.
- Never use destructive Git commands unless the user explicitly requests them and the consequences are clear.

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
