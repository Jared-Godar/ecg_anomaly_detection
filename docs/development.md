# Development workflow

## Prerequisites

- macOS or Linux
- Git
- Python
- [uv](https://docs.astral.sh/uv/)

The project environment will be defined in a later modernization phase. The quality tooling is
isolated from the project runtime and can be installed now.

## Install commit hooks

From the repository root, install the pinned pre-commit version and its hooks:

```fish
uv tool install pre-commit==4.6.0
pre-commit install --install-hooks
```

The hook runs automatically before each commit. Commits to `main` are blocked locally so changes
go through a review branch and pull request.

## Run checks manually

Run the same file-level checks used in CI:

```fish
pre-commit run --all-files
```

The checks cover:

- trailing whitespace, final newlines, and consistent line endings;
- merge markers, case conflicts, broken symlinks, and oversized additions;
- JSON, TOML, and YAML syntax;
- private keys and secret patterns;
- Python linting, import ordering, and formatting with Ruff;
- Markdown style;
- GitHub Actions security with zizmor; and
- staged secret scanning with Gitleaks.

The preserved `archive/original_2022/` bundle is excluded from style and formatting hooks so the
historical record is not rewritten. A separate CI job scans the complete Git history for secrets.

## Pull request checks

GitHub Actions runs all repository checks for every pull request and every push to `main`. Third-
party Actions are pinned to immutable commit SHAs, and Dependabot proposes weekly updates for
Actions and pre-commit hooks.

The local hook is a fast feedback mechanism, not a replacement for CI. Git hooks can be skipped or
missing on another machine; the pull request checks are the shared enforcement boundary.
