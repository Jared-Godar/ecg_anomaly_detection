# Development workflow

## Prerequisites

- macOS or Linux
- Git
- [uv](https://docs.astral.sh/uv/)

The repository pins Python 3.12.13 for development in `.python-version`. `uv` installs the requested
Python version when it is not already available and resolves the exact environment from `uv.lock`.

## Create the project environment

From the repository root, create or update the locked environment:

```fish
uv sync --locked --dev
```

Run commands through `uv` so they consistently use the project environment:

```fish
uv run python -c "import ecg_anomaly_detection; print(ecg_anomaly_detection.__doc__)"
uv run pytest
uv run pyright
```

Direct activation is optional. When an interactive environment is useful in Fish:

```fish
source .venv/bin/activate.fish
```

## Install commit hooks

After syncing the project environment, install its Git hooks:

```fish
uv run pre-commit install --install-hooks
```

The hook runs automatically before each commit. Commits to `main` are blocked locally so changes
go through a review branch and pull request.

## Run checks manually

Run the same file-level checks used in CI:

```fish
uv run pre-commit run --all-files
```

Build the package artifacts without publishing them:

```fish
uv build
```

The checks cover:

- trailing whitespace, final newlines, and consistent line endings;
- merge markers, case conflicts, broken symlinks, and oversized additions;
- JSON, TOML, and YAML syntax;
- private keys and secret patterns;
- Python linting, import ordering, and formatting with Ruff;
- Python type checking in Basic mode with Pyright;
- Markdown style;
- GitHub Actions security with zizmor; and
- staged secret scanning with Gitleaks.

The preserved `archive/original_2022/` bundle is excluded from style and formatting hooks so the
historical record is not rewritten. A separate CI job scans the complete Git history for secrets.

## Pull request checks

GitHub Actions recreates the locked environment, runs the test suite, and runs all repository
type checks and repository checks for every pull request and every push to `main`. Third-party
Actions are pinned to immutable commit SHAs, and Dependabot proposes weekly updates for Actions and
pre-commit hooks.

## Type checking

Pyright runs in Basic mode across `src/` and `tests/`. This matches the current Pylance editor mode
without committing personal VS Code settings. Run it directly with:

```fish
uv run pyright
```

Basic mode is an enforced baseline, not the final target. Strict mode should be considered only
after third-party WFDB boundaries and future pipeline interfaces can pass without broad ignores.

The local hook is a fast feedback mechanism, not a replacement for CI. Git hooks can be skipped or
missing on another machine; the pull request checks are the shared enforcement boundary.

## Dependency changes

Add or update dependencies through `uv` so `pyproject.toml` and `uv.lock` remain synchronized. For
example, a supported runtime dependency would be added with:

```fish
uv add package-name
```

Development-only tools belong in the development dependency group:

```fish
uv add --dev package-name
```

After any dependency change, run the tests and all hooks. Commit `uv.lock` with the corresponding
metadata change. Do not add historical notebook dependencies to the supported package unless a
modern package module requires them and receives tests in the same change.

## Generated local files

The `.venv/`, Python caches, coverage output, build artifacts, local data, generated models, and
report outputs are ignored. A clean pull request should contain only intentional source,
configuration, tests, documentation, and lockfile changes.
