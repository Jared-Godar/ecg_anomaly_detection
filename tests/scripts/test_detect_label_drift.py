"""Tests for the label-drift detection script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test mocks the subprocess
boundary; none performs a live GitHub call.

The shared GitHub access layer (`scripts/github/github_api.py`) the script
migrated onto in issue #175 has its own test module; the tests here cover
this script's orchestration on top of it -- in particular the observe-only
quota default that must never block a manual hygiene run, and the distinct
quota exit code.
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
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "detect_label_drift.py"
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("detect_label_drift", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. dld.find_drifted_labels).
dld = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern.
sys.modules[_SPEC.name] = dld
_SPEC.loader.exec_module(dld)


# --- load_canonical_label_names -----------------------------------------------------


def test_load_canonical_label_names_reads_the_real_manifest() -> None:
    """The actual committed .github/labels.json parses into a frozenset containing known canonical labels."""

    manifest_path = Path(__file__).resolve().parents[2] / ".github" / "labels.json"
    names = dld.load_canonical_label_names(manifest_path)
    assert "type: modernization" in names
    assert "area: repository" in names
    assert isinstance(names, frozenset)


def test_load_canonical_label_names_rejects_a_missing_schema_version(tmp_path: Path) -> None:
    """A labels.json with no top-level schema_version key is rejected rather than silently accepted.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"labels": []}), encoding="utf-8")
    # This fixture's manifest has no "schema_version" key at all.
    with pytest.raises(dld.LabelDriftError, match="schema_version"):
        dld.load_canonical_label_names(manifest)


def test_load_canonical_label_names_rejects_an_unnamed_label(tmp_path: Path) -> None:
    """A label entry with no "name" field (only a color) is rejected.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    manifest = tmp_path / "labels.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "labels": [{"color": "ffffff"}]}), encoding="utf-8"
    )
    # This fixture's one label entry has a "color" but no "name".
    with pytest.raises(dld.LabelDriftError, match="non-empty name"):
        dld.load_canonical_label_names(manifest)


# --- find_drifted_labels / find_drifted_items -----------------------------------------


def test_find_drifted_labels_returns_only_non_canonical_entries() -> None:
    """Of three labels, only the two that aren't exact members of the canonical set are returned."""

    canonical = frozenset({"type: modernization", "area: repository"})
    result = dld.find_drifted_labels(
        ["type: modernization", "area:repository", "priority:p4"], canonical
    )
    assert result == ("area:repository", "priority:p4")


def test_find_drifted_labels_with_no_drift_returns_empty_tuple() -> None:
    """When every label on an item is already canonical, find_drifted_labels returns nothing."""

    canonical = frozenset({"type: modernization", "area: repository"})
    result = dld.find_drifted_labels(["type: modernization", "area: repository"], canonical)
    assert result == ()


def test_find_drifted_items_flags_only_items_with_drift() -> None:
    """Of two items, only the one carrying a non-canonical label is included in the drifted result."""

    canonical = frozenset({"type: modernization"})
    items = [
        {"kind": "issue", "number": 61, "title": "clean", "labels": ["type: modernization"]},
        {"kind": "issue", "number": 42, "title": "GOV-006", "labels": ["type:governance"]},
    ]
    drifted = dld.find_drifted_items(items, canonical)
    assert len(drifted) == 1
    assert drifted[0].number == 42
    assert drifted[0].kind == "issue"
    assert drifted[0].drifted_labels == ("type:governance",)


def test_find_drifted_items_with_a_fully_clean_set_returns_empty_tuple() -> None:
    """A batch of items with only canonical labels produces an empty drifted-items tuple."""

    canonical = frozenset({"type: modernization"})
    items = [{"kind": "issue", "number": 1, "title": "x", "labels": ["type: modernization"]}]
    assert dld.find_drifted_items(items, canonical) == ()


def test_a_label_present_in_the_manifest_style_but_not_the_set_still_drifts() -> None:
    """A label matching canonical *formatting* (space after the colon) but absent from the set still drifts.

    Guards the real finding this tool surfaced: "risk: low" looks like a
    well-formed canonical label but was never actually added to
    .github/labels.json, so exact-set membership -- not just formatting --
    must be what determines drift.
    """

    canonical = frozenset({"risk: data-integrity", "risk: evaluation", "risk: security"})
    assert dld.find_drifted_labels(["risk: low"], canonical) == ("risk: low",)


# --- I/O boundary (mocked subprocess) -----------------------------------------------


def test_fetch_items_combines_issues_and_pull_requests() -> None:
    """fetch_items merges the separate `gh issue list` and `gh pr list` calls into one tagged sequence.

    Confirms each item is tagged with its originating "kind" ("issue" vs
    "pull request") and that both label lists are flattened from gh's
    nested {"name": ...} objects to plain strings.
    """

    issue_stdout = json.dumps([{"number": 1, "title": "an issue", "labels": [{"name": "a"}]}])
    pr_stdout = json.dumps([{"number": 2, "title": "a pr", "labels": [{"name": "b"}]}])
    responses = iter([issue_stdout, pr_stdout])

    def fake_run(cmd, **kwargs):
        """Return the next queued fake gh response, ignoring which command was actually run.

        Args:
            cmd: The subprocess command list (unused; only call order matters).
            kwargs: Ignored subprocess.run keyword arguments.

        Returns:
            A successful CompletedProcess with the next fixture response as stdout.
        """

        return subprocess.CompletedProcess(cmd, 0, stdout=next(responses), stderr="")

    # The first call is expected to be `gh issue list`, the second `gh pr list`.
    with patch.object(subprocess, "run", side_effect=fake_run):
        items = dld.fetch_items(repo=None, include_closed=False)

    assert len(items) == 2
    assert items[0] == {"kind": "issue", "number": 1, "title": "an issue", "labels": ["a"]}
    assert items[1] == {"kind": "pull request", "number": 2, "title": "a pr", "labels": ["b"]}


def test_fetch_items_raises_the_shared_api_error_on_gh_failure() -> None:
    """A failing `gh` invocation (e.g. an unauthenticated CLI) surfaces as the shared layer's error.

    Before issue #175 this script's private _run_gh translated the failure
    into LabelDriftError; the shared run_gh raises GitHubApiError instead,
    which main() catches identically (LabelDriftError subclasses it).
    """

    # gh exits non-zero when the local CLI has no valid authentication.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="not authenticated"),
        ),
        pytest.raises(dld.github_api.GitHubApiError, match="not authenticated"),
    ):
        dld.fetch_items(repo=None, include_closed=False)


# --- main (quota stewardship, mocked subprocess) --------------------------------------


def _completed(stdout: str) -> subprocess.CompletedProcess:
    """Build a fake successful subprocess.CompletedProcess with the given stdout.

    Args:
        stdout: The text `gh` would have printed to stdout.

    Returns:
        A CompletedProcess with returncode 0 and empty stderr.
    """

    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _quota(remaining: int) -> subprocess.CompletedProcess:
    """Build a fake `gh api rate_limit` response with the given GraphQL points remaining.

    Args:
        remaining: The GraphQL points remaining to report.

    Returns:
        A successful CompletedProcess carrying the rate_limit payload.
    """

    payload = {
        "resources": {
            "graphql": {
                "limit": 5000,
                "used": 5000 - remaining,
                "remaining": remaining,
                "reset": 1770000000,
            }
        }
    }
    return _completed(json.dumps(payload))


# A clean one-issue listing page, labeled only with a name the committed
# manifest actually declares, for happy-path main() tests.
_CLEAN_ISSUES = _completed(
    json.dumps([{"number": 1, "title": "an issue", "labels": [{"name": "type: governance"}]}])
)
# The matching clean pull-request listing page; unlabeled, so it can never drift.
_CLEAN_PRS = _completed(json.dumps([{"number": 2, "title": "a pr", "labels": []}]))


def test_main_happy_path_reports_quota_and_detects_no_drift(
    capsys: pytest.CaptureFixture,
) -> None:
    """A clean repository exits 0 and prints the before/after/consumed quota line.

    The stderr report must carry the before, after, and consumed values --
    the accountability line issue #173 requires of every quota-consuming run,
    extended to this hygiene script by issue #175.
    """

    # Full sequence: preflight, issue listing, PR listing, report.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(4990), _CLEAN_ISSUES, _CLEAN_PRS, _quota(4988)],
    ):
        exit_code = dld.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No label drift detected." in captured.out
    # The consumption report names all three accounting values.
    assert "4990 before" in captured.err
    assert "4988 after" in captured.err
    assert "2 consumed" in captured.err


def test_main_still_reports_quota_when_drift_is_found(capsys: pytest.CaptureFixture) -> None:
    """A drift finding keeps its exit code 1, with the quota report alongside it.

    The quota line is accounting, not a verdict: it must accompany the drift
    report without altering the drift exit code CI gates on.
    """

    drifted_issues = _completed(
        json.dumps([{"number": 42, "title": "old", "labels": [{"name": "type:governance"}]}])
    )
    # Preflight, drifted issue listing, clean PR listing, report.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(4990), drifted_issues, _CLEAN_PRS, _quota(4988)],
    ):
        exit_code = dld.main([])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Label drift detected on 1 item(s):" in captured.err
    assert "type:governance" in captured.err
    assert "4990 before" in captured.err


def test_main_default_threshold_never_blocks_even_on_a_drained_pool(
    capsys: pytest.CaptureFixture,
) -> None:
    """With the observe-only default, a fully drained pool still runs the check.

    This is issue #175's explicit non-goal made executable: the default
    threshold of 0 must record and report consumption without ever blocking
    a manual hygiene run.
    """

    # The pool reads 0 remaining at preflight and at report time alike.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(0), _CLEAN_ISSUES, _CLEAN_PRS, _quota(0)],
    ):
        exit_code = dld.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No label drift detected." in captured.out
    assert "0 before" in captured.err


def test_main_stops_with_exit_three_when_an_explicit_threshold_is_undercut(
    capsys: pytest.CaptureFixture,
) -> None:
    """An operator-supplied positive threshold stops the run before any listing.

    The stop must happen before either GraphQL-backed listing is fetched, and
    its exit code must differ from both the drift code (1) and the failure
    code (2) so hygiene output can never conflate a drained pool with drift.
    """

    # Both rate_limit reads (preflight and report) see a nearly drained pool.
    with patch.object(subprocess, "run", side_effect=[_quota(12), _quota(12)]) as mock_run:
        exit_code = dld.main(["--min-graphql-quota", "50"])
    assert exit_code == 3
    captured = capsys.readouterr()
    assert "quota:" in captured.err
    assert "only 12 of 5000" in captured.err
    # The only gh traffic allowed is the free REST rate_limit accounting --
    # neither the issue listing nor the PR listing may have been fetched.
    for call in mock_run.call_args_list:
        assert "rate_limit" in call.args[0]
        assert "list" not in call.args[0]


def test_main_maps_a_primary_rate_limit_to_exit_three(capsys: pytest.CaptureFixture) -> None:
    """Primary rate-limit exhaustion mid-run exits 3, distinct from failures and drift.

    The drained shared pool is infrastructure, not evidence about the
    repository's labels; the exit code and the "quota:" prefix keep the two
    unmistakably apart.
    """

    exhausted = subprocess.CalledProcessError(
        1, ["gh"], stderr="GraphQL: API rate limit already exceeded for user ID 16855088."
    )
    # Preflight passes (observe-only default), the first listing then hits the
    # exhausted pool, and the report still runs.
    with patch.object(subprocess, "run", side_effect=[_quota(60), exhausted, _quota(0)]):
        exit_code = dld.main([])
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err


def test_main_returns_two_on_an_ordinary_gh_failure(capsys: pytest.CaptureFixture) -> None:
    """A gh CLI failure (e.g. an unauthenticated CLI) keeps the pre-migration exit code 2."""

    auth_error = subprocess.CalledProcessError(1, ["gh"], stderr="not authenticated")
    # Preflight passes, then the issue listing fails; the report still runs.
    with patch.object(subprocess, "run", side_effect=[_quota(4990), auth_error, _quota(4990)]):
        exit_code = dld.main([])
    assert exit_code == 2
    assert "not authenticated" in capsys.readouterr().err
