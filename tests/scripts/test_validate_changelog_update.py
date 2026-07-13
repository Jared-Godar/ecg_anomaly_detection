"""Tests for the per-PR changelog gate script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test that exercises GitHub access
mocks subprocess at the gh boundary -- no live API calls, per issue #184's
acceptance criteria.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "validate_changelog_update.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("validate_changelog_update", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. vcu.evaluate_changelog_gate).
vcu = importlib.util.module_from_spec(_SPEC)
# dataclasses and postponed type hints resolve via sys.modules[cls.__module__], so the
# module is registered there before exec_module runs the dynamically loaded file.
sys.modules[_SPEC.name] = vcu
_SPEC.loader.exec_module(vcu)

# The shared API layer the script imports by directory adjacency; its error
# classes are what the exit-code tests below provoke and assert against.
_gha = vcu.github_api


def _completed(payload: object) -> subprocess.CompletedProcess:
    """Build a successful gh subprocess result carrying a JSON payload.

    Args:
        payload: The object to serialize as the process's stdout.

    Returns:
        A CompletedProcess shaped like a successful `gh` invocation.
    """

    return subprocess.CompletedProcess(args=["gh"], returncode=0, stdout=json.dumps(payload))


def _pr_payload(body: str) -> dict[str, object]:
    """Build a REST pull-request payload carrying the one field the gate reads.

    Args:
        body: The pull request body text.

    Returns:
        A dict shaped like `gh api repos/.../pulls/N` output.
    """

    return {"number": 65, "body": body}


def _files_payload(*filenames: str) -> list[list[dict[str, object]]]:
    """Build a slurped (`--paginate --slurp`) changed-files payload of one page.

    Args:
        filenames: The changed paths to report, one file entry each.

    Returns:
        A list of pages, each a list of file entries, matching gh's slurp shape.
    """

    return [[{"filename": name} for name in filenames]]


# --- substantive_paths -----------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/ecg_anomaly_detection/splitting.py",
        "scripts/github/validate_changelog_update.py",
        "docs/governance/releases.md",
        "configs/windowing-v1.toml",
        ".github/workflows/metadata-governance.yml",
    ],
)
def test_substantive_paths_matches_every_governed_prefix(path: str) -> None:
    """Each of the five governed path prefixes registers as substantive.

    Args:
        path: One changed path under a governed prefix.
    """

    assert vcu.substantive_paths([path]) == (path,)


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "CHANGELOG.md",
        "tests/unit/test_windows.py",
        "uv.lock",
        ".github/labels.json",
        "notebooks/00-overview.ipynb",
    ],
)
def test_substantive_paths_ignores_ungoverned_paths(path: str) -> None:
    """Paths outside the governed prefixes (including the changelog itself, tests,
    lockfiles, and non-workflow .github files) never demand an entry by themselves.

    Args:
        path: One changed path outside every governed prefix.
    """

    assert vcu.substantive_paths([path]) == ()


def test_substantive_paths_preserves_input_order() -> None:
    """Substantive paths come back in diff order so failure output reads naturally."""

    paths = ["docs/a.md", "README.md", "src/b.py", "configs/c.toml"]
    assert vcu.substantive_paths(paths) == ("docs/a.md", "src/b.py", "configs/c.toml")


# --- touches_changelog -----------------------------------------------------------


def test_touches_changelog_requires_the_root_changelog() -> None:
    """Only the repository-root CHANGELOG.md counts; a nested file of the same name
    is substantive documentation, not the changelog this gate enforces."""

    assert vcu.touches_changelog(["CHANGELOG.md", "src/x.py"]) is True
    assert vcu.touches_changelog(["docs/CHANGELOG.md", "src/x.py"]) is False


# --- has_exemption_marker --------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "changelog: not-needed",
        "changelog: not-needed -- board-only reconciliation, no repository diff",
        "  changelog: not-needed",
        "CHANGELOG: NOT-NEEDED",
        "Summary line.\nchangelog: not-needed\nMore text.",
    ],
)
def test_has_exemption_marker_accepts_a_real_marker_line(body: str) -> None:
    """The marker is honored at line start, any case, with or without a trailing reason.

    Args:
        body: A PR body containing a genuine marker line.
    """

    assert vcu.has_exemption_marker(body) is True


@pytest.mark.parametrize(
    "body",
    [
        "",
        "no marker here",
        # Mid-sentence prose mentioning the token is not a declaration; the
        # marker must start its own line to be deliberate and visible.
        "we decided changelog: not-needed applies here",
        # Quoted as inline code -- documentation about the marker, not a use of it.
        "add the `changelog: not-needed` line to exempt a PR",
        # Quoted inside a fenced block, e.g. a PR documenting the mechanism itself.
        "Usage:\n```text\nchangelog: not-needed -- reason\n```\ndone.",
    ],
)
def test_has_exemption_marker_rejects_absent_or_quoted_markers(body: str) -> None:
    """No marker, a mid-sentence mention, or a code-quoted example never exempts a PR.

    The code-quoted cases are the self-referential trap: the pull request that
    documents this mechanism quotes the marker text, and must not thereby
    exempt itself.

    Args:
        body: A PR body that must not register as exempted.
    """

    assert vcu.has_exemption_marker(body) is False


def test_has_exemption_marker_tolerates_a_missing_body() -> None:
    """A PR with no body at all (None from the API) reads as not exempted, not an error."""

    assert vcu.has_exemption_marker(None) is False


# --- evaluate_changelog_gate -----------------------------------------------------


def test_gate_passes_when_no_substantive_path_changed() -> None:
    """A diff touching only ungoverned paths needs no entry and no marker."""

    violations, notices = vcu.evaluate_changelog_gate(["README.md", "uv.lock"], "")
    assert violations == ()
    assert any("no substantive paths" in notice for notice in notices)


def test_gate_passes_when_changelog_updated_alongside_substantive_paths() -> None:
    """The contract's happy path: substantive change plus a CHANGELOG.md update."""

    violations, notices = vcu.evaluate_changelog_gate(
        ["src/ecg_anomaly_detection/windows.py", "CHANGELOG.md"], ""
    )
    assert violations == ()
    assert any("CHANGELOG.md is updated" in notice for notice in notices)


def test_gate_passes_on_explicit_exemption_and_says_so() -> None:
    """The exemption path: a substantive diff with a real marker passes, and the
    output names the exemption loudly instead of passing silently."""

    violations, notices = vcu.evaluate_changelog_gate(
        ["docs/governance/releases.md"],
        "changelog: not-needed -- typo fix with no behavioral effect",
    )
    assert violations == ()
    assert any("exemption declared" in notice for notice in notices)


def test_gate_fails_on_substantive_diff_without_entry_or_marker() -> None:
    """The failure path this gate exists for: substantive paths, no changelog,
    no marker. The violation names offending paths and both remedies."""

    violations, notices = vcu.evaluate_changelog_gate(
        ["src/ecg_anomaly_detection/splitting.py", "configs/splitting-v2.toml"], "unrelated body"
    )
    assert notices == ()
    assert len(violations) == 1
    assert "src/ecg_anomaly_detection/splitting.py" in violations[0]
    assert "## Unreleased" in violations[0]
    assert "changelog: not-needed" in violations[0]


def test_gate_failure_truncates_a_long_substantive_path_sample() -> None:
    """More than five substantive paths are summarized with a +N annotation so the
    violation stays one readable line."""

    paths = [f"src/module_{i}.py" for i in range(8)]
    violations, _ = vcu.evaluate_changelog_gate(paths, "")
    assert "(+3 more)" in violations[0]


def test_gate_ignores_a_code_quoted_marker_on_the_failure_path() -> None:
    """A marker that only appears inside a code span still fails the gate -- quoting
    the mechanism is not invoking it."""

    violations, _ = vcu.evaluate_changelog_gate(
        ["scripts/github/x.py"], "see `changelog: not-needed` for the escape hatch"
    )
    assert len(violations) == 1


# --- fetch_pull_request_body / fetch_changed_paths -------------------------------


def test_fetch_pull_request_body_reads_via_rest() -> None:
    """The body fetch is REST (gh api repos/...), with gh's own placeholders left
    for it to resolve when no explicit repo is given."""

    # Mock the single gh call at the subprocess boundary and capture its argv.
    with patch.object(subprocess, "run", return_value=_completed(_pr_payload("hello"))) as run:
        body = vcu.fetch_pull_request_body(65, repo=None)
    assert body == "hello"
    assert run.call_args_list[0].args[0] == ["gh", "api", "repos/{owner}/{repo}/pulls/65"]


def test_fetch_pull_request_body_normalizes_a_null_body() -> None:
    """REST reports a JSON null for an empty PR description; the gate sees ''."""

    # A null body in the REST payload is the no-description case.
    with patch.object(subprocess, "run", return_value=_completed({"number": 65, "body": None})):
        assert vcu.fetch_pull_request_body(65, repo=None) == ""


def test_fetch_changed_paths_paginates_and_includes_rename_sources() -> None:
    """The file listing is fetched with --paginate --slurp, flattens every page, and
    includes a rename's previous path so a file moved out of a governed tree still
    registers as substantive change."""

    pages = [
        [{"filename": "src/new_name.py", "previous_filename": "src/old_name.py"}],
        [{"filename": "README.md"}],
    ]
    # Serve the two-page slurped listing from one mocked gh call and capture argv.
    with patch.object(subprocess, "run", return_value=_completed(pages)) as run:
        paths = vcu.fetch_changed_paths(65, repo="Jared-Godar/ecg_anomaly_detection")
    assert paths == ("src/new_name.py", "src/old_name.py", "README.md")
    assert run.call_args_list[0].args[0] == [
        "gh",
        "api",
        "repos/Jared-Godar/ecg_anomaly_detection/pulls/65/files",
        "--paginate",
        "--slurp",
    ]


def test_fetch_changed_paths_deduplicates_preserving_order() -> None:
    """A path listed more than once across pages is reported exactly once."""

    pages = [[{"filename": "docs/a.md"}], [{"filename": "docs/a.md"}, {"filename": "docs/b.md"}]]
    # The duplicate path spans two pages, exercising cross-page deduplication.
    with patch.object(subprocess, "run", return_value=_completed(pages)):
        assert vcu.fetch_changed_paths(65, repo=None) == ("docs/a.md", "docs/b.md")


# --- main ------------------------------------------------------------------------


def _main_with_responses(body: str, *filenames: str) -> int:
    """Run main() for PR #65 against mocked body and changed-files responses.

    Args:
        body: The PR body the mocked body fetch returns.
        filenames: The changed paths the mocked file listing returns.

    Returns:
        main()'s exit code.
    """

    # main() makes exactly two gh calls, in a fixed order: the PR body read,
    # then the changed-files listing; side_effect serves them in sequence.
    responses = [_completed(_pr_payload(body)), _completed(_files_payload(*filenames))]
    # side_effect pops one response per call, matching main()'s fixed call order.
    with patch.object(subprocess, "run", side_effect=responses):
        return vcu.main(["--pr-number", "65"])


def test_main_fails_a_substantive_pr_without_changelog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end failure path: substantive diff, no entry, no marker exits 1 and
    reports the violation on stderr."""

    exit_code = _main_with_responses("just a description", "src/x.py", "docs/y.md")
    assert exit_code == 1
    assert "Changelog gate failed" in capsys.readouterr().err


def test_main_passes_a_substantive_pr_with_changelog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end happy path: the CHANGELOG.md update satisfies the gate with exit 0."""

    exit_code = _main_with_responses("description", "src/x.py", "CHANGELOG.md")
    assert exit_code == 0
    assert "Changelog gate passed" in capsys.readouterr().out


def test_main_passes_an_exempted_pr_and_prints_the_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end exemption path: the marker passes the gate and the notice makes the
    exemption visible in the check's own output."""

    exit_code = _main_with_responses(
        "changelog: not-needed -- board-only reconciliation", "docs/governance/x.md"
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "exemption declared" in output
    assert "Changelog gate passed" in output


def test_main_returns_2_when_the_pull_request_cannot_be_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unreadable PR (bad number, auth failure) exits 2 -- required data missing,
    distinct from a genuine gate failure."""

    # gh exits non-zero when the PR number doesn't exist.
    failure = subprocess.CalledProcessError(1, ["gh"], stderr="pull request not found")
    # Every gh call fails identically, so the first fetch already surfaces the error.
    with patch.object(subprocess, "run", side_effect=failure):
        exit_code = vcu.main(["--pr-number", "999999"])
    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_main_returns_3_on_a_primary_rate_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An exhausted primary rate limit exits 3 -- transient shared infrastructure,
    never presented as a changelog defect in the PR being validated."""

    # gh's primary rate-limit wording contains "rate limit" without "secondary",
    # which the shared layer classifies as PrimaryRateLimitError.
    failure = subprocess.CalledProcessError(1, ["gh"], stderr="API rate limit exceeded for user")
    # The shared layer classifies this wording before any retry, failing fast.
    with patch.object(subprocess, "run", side_effect=failure):
        exit_code = vcu.main(["--pr-number", "65"])
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err
