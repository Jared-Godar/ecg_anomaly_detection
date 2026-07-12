"""Tests for the post-merge Project #5 status-sync script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test mocks the subprocess
boundary; none performs a live GitHub call.

The shared GitHub access layer (`scripts/github/github_api.py`) has its own
test module; the tests here cover this script's orchestration on top of it --
in particular that mutation verification is a fresh *targeted* read-back and
that no code path ever performs a full board `item-list` scan (issue #173).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Locate the script relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "set_merged_project_status.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("set_merged_project_status", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. smps.set_status_merged).
smps = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern so relative imports inside the script (if any)
# would resolve correctly.
sys.modules[_SPEC.name] = smps
_SPEC.loader.exec_module(smps)

# The shared access layer the script imports; referenced directly for its
# dataclasses and error types.
gha = smps.github_api


# A fixed fake OWNER/REPO string reused across every test that needs one.
_REPO = "Jared-Godar/ecg_anomaly_detection"


def _completed(stdout: str) -> subprocess.CompletedProcess:
    """Build a fake successful subprocess.CompletedProcess with the given stdout.

    Args:
        stdout: The text `gh` would have printed to stdout.

    Returns:
        A CompletedProcess with returncode 0 and empty stderr.
    """

    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _read_back(status: str | None) -> subprocess.CompletedProcess:
    """Build a fake targeted node(id:) read-back response for the Status field.

    Args:
        status: The Status option name to report, or None for an unset field
            (GitHub returns JSON null for fieldValueByName in that case).

    Returns:
        A successful CompletedProcess carrying the GraphQL response envelope.
    """

    value = {"name": status} if status is not None else None
    return _completed(json.dumps({"data": {"node": {"fieldValueByName": value}}}))


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


def _pr_items(nodes: list[dict[str, object]]) -> subprocess.CompletedProcess:
    """Build a fake targeted projectItems lookup response for one pull request.

    Args:
        nodes: The PR's project-item memberships to report.

    Returns:
        A successful CompletedProcess carrying the GraphQL response envelope.
    """

    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "projectItems": {"pageInfo": {"hasNextPage": False}, "nodes": nodes}
                }
            }
        }
    }
    return _completed(json.dumps(payload))


# The one membership node used by every main() test that models a tracked PR.
_TRACKED_NODE: dict[str, object] = {
    "id": "PVTI_target",
    "project": {"id": "PVT_project", "number": 5, "owner": {"login": "Jared-Godar"}},
}

# The Status field/option pair used by every set_status_merged test.
_FIELD = gha.SingleSelectOption(field_id="PVTSSF_status", option_id="merged")


def _client() -> object:
    """Build a fresh ProjectClient bound to the fixture project for one test.

    Returns:
        A ProjectClient for Jared-Godar's Project #5.
    """

    return gha.ProjectClient("Jared-Godar", 5)


# --- set_status_merged -----------------------------------------------------------------


def test_set_status_merged_skips_the_mutation_when_already_merged() -> None:
    """A Status that already reads Merged costs one targeted read and no mutation.

    This is the idempotent-skip regression from issue #173: a rerun (or a
    built-in workflow that already produced the desired value) must not spend
    a mutation or a second read.
    """

    # The very first (pre-check) read already observes the desired value.
    with patch.object(subprocess, "run", side_effect=[_read_back("Merged")]) as mock_run:
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    assert mock_run.call_count == 1
    # That one call must be a read, never the item-edit mutation.
    assert "item-edit" not in mock_run.call_args_list[0].args[0]


def test_set_status_merged_invokes_item_edit_with_expected_args() -> None:
    """set_status_merged mutates with every required flag and verifies the result."""

    # Pre-check observes Closed, the mutation runs, the fresh read-back confirms.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_read_back("Closed"), _completed(""), _read_back("Merged")],
    ) as mock_run:
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    # The exact argument list pins each flag to its own value; membership checks
    # alone would let the two opaque node ids swap across --id and --project-id.
    assert mock_run.call_args_list[1].args[0] == [
        "gh",
        "project",
        "item-edit",
        "--id",
        "PVTI_target",
        "--project-id",
        "PVT_project",
        "--field-id",
        "PVTSSF_status",
        "--single-select-option-id",
        "merged",
    ]
    # The verification read is the targeted GraphQL query, not a board scan.
    assert mock_run.call_args_list[2].args[0][:3] == ["gh", "api", "graphql"]


def test_set_status_merged_never_performs_a_full_board_scan() -> None:
    """No code path in the mutation loop issues `gh project item-list`.

    This is issue #173's central regression: the read-back-verified mutation
    rule is preserved, but its verification reads are targeted -- a full board
    snapshot inside a per-item mutation loop must never come back.
    """

    # A full mutate-verify-retry cycle: pre-check, two mutations, two read-backs.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _read_back("Closed"),
            _completed(""),
            _read_back("Closed"),
            _completed(""),
            _read_back("Merged"),
        ],
    ) as mock_run:
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    # Even across the retry, not one invocation may be a board-wide item-list.
    for call in mock_run.call_args_list:
        assert "item-list" not in call.args[0]


def test_set_status_merged_verifies_no_changes_error_before_accepting_it() -> None:
    """A no-change error succeeds only when a fresh read confirms Merged."""

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    # Despite gh's error, the independent targeted read proves no retry is needed.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_read_back("Closed"), no_change, _read_back("Merged")],
    ) as mock_run:
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    assert mock_run.call_count == 3


def test_set_status_merged_retries_when_read_back_is_stale() -> None:
    """A first read-back still showing the old value triggers exactly one bounded retry.

    This is the stale-read-back regression from issue #173: targeted reads can
    briefly observe pre-mutation state, so one retry (with its own fresh
    read-back) must remain.
    """

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    # First read remains Closed; the single retry changes the second read to Merged.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _read_back("Closed"),
            no_change,
            _read_back("Closed"),
            _completed(""),
            _read_back("Merged"),
        ],
    ) as mock_run:
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    assert mock_run.call_count == 5


def test_set_status_merged_retries_when_clean_mutation_reads_back_wrong() -> None:
    """A mutation that exits cleanly but reads back wrong still earns the retry.

    This is the Closed-beats-Merged race itself: item-edit succeeds, a built-in
    workflow overwrites the value, and only the read-back can reveal it. The
    retry must not be conditional on gh's 'no changes to make' error.
    """

    # Both mutations exit zero; only the second read-back shows Merged.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _read_back("Closed"),
            _completed(""),
            _read_back("Closed"),
            _completed(""),
            _read_back("Merged"),
        ],
    ) as mock_run:
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    assert mock_run.call_count == 5
    # The fourth call must be the re-run mutation itself, not another read.
    assert mock_run.call_args_list[3].args[0][:3] == ["gh", "project", "item-edit"]


def test_set_status_merged_fails_after_bounded_retry() -> None:
    """Two unverified attempts fail clearly, reporting the exact observed value."""

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    # Both mutation/read pairs leave the value Closed, exhausting the fixed bound;
    # the message must name the observed value so operators see what actually won.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=[
                _read_back("Closed"),
                no_change,
                _read_back("Closed"),
                no_change,
                _read_back("Closed"),
            ],
        ) as mock_run,
        pytest.raises(smps.ProjectStatusSyncError, match="read back as 'Closed' after 2 attempts"),
    ):
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    assert mock_run.call_count == 5


def test_set_status_merged_reports_unset_when_status_never_appears() -> None:
    """A Status that never reads back at all is reported as 'unset', not None.

    GitHub omits unset fields entirely (fieldValueByName is JSON null), so the
    read-back yields None throughout; the failure message must translate that
    into 'unset' rather than crashing or printing a Python None.
    """

    # Both mutations exit zero, but the item's Status stays absent throughout.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=[
                _read_back(None),
                _completed(""),
                _read_back(None),
                _completed(""),
                _read_back(None),
            ],
        ),
        pytest.raises(smps.ProjectStatusSyncError, match="read back as 'unset' after 2 attempts"),
    ):
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)


def test_set_status_merged_propagates_other_gh_errors_without_read_back() -> None:
    """A gh failure other than 'no changes to make' keeps its fail-fast path.

    Only the observed false no-change response is eligible for read-back
    recovery; masking an authentication or schema fault behind a retry would
    hide the actual defect from CI output.
    """

    auth_error = subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope")
    # The unrelated CLI failure must propagate immediately: no read-back, no retry.
    with (
        patch.object(subprocess, "run", side_effect=[_read_back("Closed"), auth_error]) as mock_run,
        pytest.raises(gha.GitHubApiError, match="insufficient scope"),
    ):
        smps.set_status_merged(_client(), "PVTI_target", "PVT_project", _FIELD)
    # The pre-check read plus the one failed mutation: nothing after it.
    assert mock_run.call_count == 2


# --- main ------------------------------------------------------------------------------


def test_main_returns_zero_and_warns_when_pr_not_tracked(capsys: pytest.CaptureFixture) -> None:
    """A PR that isn't a Project item is a non-fatal warning (exit 0), not a failure.

    Not every merged PR is necessarily tracked on the project board, so this must
    succeed with a warning rather than treating it as an error.
    """

    # Preflight passes, the targeted lookup finds no membership, the report runs.
    with patch.object(
        subprocess, "run", side_effect=[_quota(4988), _pr_items([]), _quota(4988)]
    ) as mock_run:
        exit_code = smps.main(["--pr-number", "999999", "--repo", _REPO])
    assert exit_code == 0
    assert "not a Project" in capsys.readouterr().err
    # No mutation may have been attempted for an untracked PR.
    for call in mock_run.call_args_list:
        assert "item-edit" not in call.args[0]


def test_main_returns_two_on_gh_failure(capsys: pytest.CaptureFixture) -> None:
    """A gh CLI failure (e.g. insufficient token scope) exits with code 2."""

    auth_error = subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope")
    # Preflight passes, then the targeted lookup fails; the report still runs.
    with patch.object(subprocess, "run", side_effect=[_quota(4988), auth_error, _quota(4988)]):
        exit_code = smps.main(["--pr-number", "116", "--repo", _REPO])
    assert exit_code == 2
    assert "insufficient scope" in capsys.readouterr().err


def test_main_stops_with_exit_three_before_any_mutation_when_quota_is_low(
    capsys: pytest.CaptureFixture,
) -> None:
    """A preflight below the threshold exits 3 without a single GraphQL call.

    This is the quota-exhaustion-before-mutation regression from issue #173:
    the stop must happen before the lookup and before item-edit, and its exit
    code must differ from the metadata-failure code so CI output can never
    conflate the two.
    """

    # Both rate_limit reads (preflight and report) see a nearly drained pool.
    with patch.object(subprocess, "run", side_effect=[_quota(12), _quota(12)]) as mock_run:
        exit_code = smps.main(["--pr-number", "116", "--repo", _REPO, "--min-graphql-quota", "50"])
    assert exit_code == 3
    captured = capsys.readouterr()
    assert "only 12 of 5000" in captured.err
    # The only gh traffic allowed is the free REST rate_limit accounting --
    # no GraphQL lookup, no field-list, and above all no mutation.
    for call in mock_run.call_args_list:
        assert "rate_limit" in call.args[0]
        assert "item-edit" not in call.args[0]


def test_main_maps_a_primary_rate_limit_to_exit_three(capsys: pytest.CaptureFixture) -> None:
    """Primary rate-limit exhaustion mid-run exits 3, distinct from metadata failures.

    The drained shared pool is infrastructure, not a defect in the PR being
    synced; the exit code and the "quota:" prefix keep the two unmistakably apart.
    """

    exhausted = subprocess.CalledProcessError(
        1, ["gh"], stderr="GraphQL: API rate limit already exceeded for user ID 16855088."
    )
    # Preflight passes (the pool drains between the two calls), the lookup
    # then hits the exhausted pool, and the report still runs.
    with patch.object(subprocess, "run", side_effect=[_quota(60), exhausted, _quota(0)]):
        exit_code = smps.main(["--pr-number", "116", "--repo", _REPO])
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err


def test_main_happy_path_mutates_verifies_and_reports_quota(
    capsys: pytest.CaptureFixture,
) -> None:
    """A tracked PR is mutated, verified via targeted read-back, and quota is reported.

    The stderr report must carry the before, after, and consumed values --
    the accountability line issue #173 requires of every quota-consuming run.
    """

    # Full sequence: preflight, targeted lookup, field-list, pre-check read,
    # mutation, verification read, report.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _quota(4988),
            _pr_items([_TRACKED_NODE]),
            _completed(
                json.dumps(
                    {
                        "fields": [
                            {
                                "id": "PVTSSF_status",
                                "name": "Status",
                                "options": [{"id": "merged", "name": "Merged"}],
                            }
                        ]
                    }
                )
            ),
            _read_back("Closed"),
            _completed(""),
            _read_back("Merged"),
            _quota(4983),
        ],
    ) as mock_run:
        exit_code = smps.main(["--pr-number", "116", "--repo", _REPO])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "status to Merged for pull request #116" in captured.out
    # The consumption report names all three accounting values.
    assert "4988 before" in captured.err
    assert "4983 after" in captured.err
    assert "5 consumed" in captured.err
    # End-to-end, the run must never fall back to a board-wide item-list.
    for call in mock_run.call_args_list:
        assert "item-list" not in call.args[0]
