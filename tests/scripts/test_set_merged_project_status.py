"""Tests for the post-merge Project #5 status-sync script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`.
"""

from __future__ import annotations

import importlib.util
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
# The module object every test in this file calls into (e.g. smps.fetch_project_id).
smps = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern so relative imports inside the script (if any)
# would resolve correctly.
sys.modules[_SPEC.name] = smps
_SPEC.loader.exec_module(smps)


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


# --- fetch_project_id ----------------------------------------------------------------


def test_fetch_project_id_parses_gh_output() -> None:
    """fetch_project_id extracts the "id" field from `gh project view`'s JSON output."""

    # gh's real `project view --format json` output includes an "id" field.
    with patch.object(subprocess, "run", return_value=_completed('{"id": "PVT_kwHOAQEwMM4BcY39"}')):
        assert smps.fetch_project_id("Jared-Godar", 5) == "PVT_kwHOAQEwMM4BcY39"


def test_fetch_project_id_raises_on_gh_failure() -> None:
    """A failing `gh` invocation is translated into ProjectStatusSyncError with gh's stderr."""

    # gh exits non-zero when the project can't be found (e.g. permissions, typo).
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="project not found"),
        ),
        pytest.raises(smps.ProjectStatusSyncError, match="project not found"),
    ):
        smps.fetch_project_id("Jared-Godar", 5)


# --- fetch_status_field ---------------------------------------------------------------


def _status_field_payload(options: list[dict[str, str]]) -> str:
    """Build a fake `gh project field-list` JSON payload with one "Status" field.

    Args:
        options: The Status field's single-select options to include.

    Returns:
        JSON text matching gh's field-list output shape.
    """

    import json

    return json.dumps({"fields": [{"id": "PVTSSF_status", "name": "Status", "options": options}]})


def test_fetch_status_field_returns_field_and_merged_option_ids() -> None:
    """fetch_status_field extracts both the Status field's id and its "Merged" option's id."""

    payload = _status_field_payload(
        [{"id": "backlog", "name": "Backlog"}, {"id": "merged", "name": "Merged"}]
    )
    # Two options are present; only "Merged" is what fetch_status_field must find.
    with patch.object(subprocess, "run", return_value=_completed(payload)):
        field = smps.fetch_status_field("Jared-Godar", 5)
    assert field.field_id == "PVTSSF_status"
    assert field.merged_option_id == "merged"


def test_fetch_status_field_raises_when_status_field_missing() -> None:
    """A project with no field named "Status" at all raises a specific, actionable error."""

    import json

    payload = json.dumps({"fields": [{"id": "x", "name": "Priority", "options": []}]})
    # The only field present is "Priority", not "Status".
    with (
        patch.object(subprocess, "run", return_value=_completed(payload)),
        pytest.raises(smps.ProjectStatusSyncError, match="no 'Status' field"),
    ):
        smps.fetch_status_field("Jared-Godar", 5)


def test_fetch_status_field_raises_when_merged_option_missing() -> None:
    """A Status field that exists but lacks a "Merged" option raises a specific error."""

    payload = _status_field_payload([{"id": "backlog", "name": "Backlog"}])
    # "Status" exists but its only option is "Backlog", not "Merged".
    with (
        patch.object(subprocess, "run", return_value=_completed(payload)),
        pytest.raises(smps.ProjectStatusSyncError, match="no 'Merged' option"),
    ):
        smps.fetch_status_field("Jared-Godar", 5)


# --- find_pull_request_item_id ---------------------------------------------------------


def _item_list_payload(items: list[dict[str, object]]) -> str:
    """Build a fake `gh project item-list` JSON payload from a list of item objects.

    Args:
        items: The project items to include, each shaped like gh's own item objects.

    Returns:
        JSON text matching gh's item-list output shape.
    """

    import json

    return json.dumps({"items": items})


def test_find_pull_request_item_id_matches_type_number_and_repo() -> None:
    """The matching item is found only when type, number, AND repository all agree.

    Three decoy items are included, each failing exactly one of the three match
    criteria (wrong type, wrong repo), to confirm the lookup requires all three to
    agree rather than matching on any single field.
    """

    items = [
        {"content": {"type": "Issue", "number": 116, "repository": _REPO}, "id": "wrong-type"},
        {
            "content": {"type": "PullRequest", "number": 999, "repository": "other/repo"},
            "id": "wrong-repo",
        },
        {
            "content": {"type": "PullRequest", "number": 116, "repository": _REPO},
            "id": "PVTI_target",
        },
    ]
    # Only the third item matches all three criteria (type, number, repository).
    with patch.object(subprocess, "run", return_value=_completed(_item_list_payload(items))):
        item_id = smps.find_pull_request_item_id("Jared-Godar", 5, _REPO, 116)
    assert item_id == "PVTI_target"


def test_find_pull_request_item_id_returns_none_when_not_tracked() -> None:
    """A PR that isn't a Project item at all returns None, not an error."""

    # An empty item list means nothing in the project matches any PR.
    with patch.object(subprocess, "run", return_value=_completed(_item_list_payload([]))):
        assert smps.find_pull_request_item_id("Jared-Godar", 5, _REPO, 999999) is None


# --- fetch_item_status -------------------------------------------------------------------


def test_fetch_item_status_returns_current_status_value() -> None:
    """The requested item's own status property is returned, not another item's."""

    payload = _item_list_payload(
        [
            {"id": "PVTI_other", "status": "Backlog"},
            {"id": "PVTI_target", "status": "Review"},
        ]
    )
    # Two items are listed; the lookup must select by item id, not list position.
    with patch.object(subprocess, "run", return_value=_completed(payload)):
        assert smps.fetch_item_status("Jared-Godar", 5, "PVTI_target") == "Review"


def test_fetch_item_status_returns_none_when_status_unset() -> None:
    """An item present without any status property reads as None, not an error."""

    payload = _item_list_payload([{"id": "PVTI_target"}])
    # Freshly added project items can legitimately have no Status value at all.
    with patch.object(subprocess, "run", return_value=_completed(payload)):
        assert smps.fetch_item_status("Jared-Godar", 5, "PVTI_target") is None


def test_fetch_item_status_raises_when_item_vanishes() -> None:
    """An item missing from the read-back is an explicit failure, not an unset value.

    Losing the item between mutation and verification makes the requested state
    unknowable, so this must raise rather than burn the retry on a phantom 'unset'.
    """

    payload = _item_list_payload([{"id": "PVTI_other", "status": "Merged"}])
    # The item list no longer contains the mutated item at all.
    with (
        patch.object(subprocess, "run", return_value=_completed(payload)),
        pytest.raises(smps.ProjectStatusSyncError, match="no longer contains item"),
    ):
        smps.fetch_item_status("Jared-Godar", 5, "PVTI_target")


# --- set_status_merged -----------------------------------------------------------------


def test_set_status_merged_invokes_item_edit_with_expected_args() -> None:
    """set_status_merged mutates with every required flag and verifies the result."""

    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    read_back = _item_list_payload([{"id": "PVTI_target", "status": "Merged"}])
    # The mutation succeeds, then the item list confirms the desired value.
    with patch.object(
        subprocess, "run", side_effect=[_completed(""), _completed(read_back)]
    ) as mock_run:
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)
    # The exact argument list pins each flag to its own value; membership checks
    # alone would let the two opaque node ids swap across --id and --project-id.
    assert mock_run.call_args_list[0].args[0] == [
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
    assert mock_run.call_args_list[1].args[0][:3] == ["gh", "project", "item-list"]


def test_set_status_merged_verifies_no_changes_error_before_accepting_it() -> None:
    """A no-change error succeeds only when a fresh read confirms Merged."""

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    read_back = _item_list_payload([{"id": "PVTI_target", "status": "Merged"}])
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    # Despite gh's error, the independent item read proves no retry is needed.
    with patch.object(
        subprocess, "run", side_effect=[no_change, _completed(read_back)]
    ) as mock_run:
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)
    assert mock_run.call_count == 2


def test_set_status_merged_retries_when_read_back_is_not_merged() -> None:
    """A missing desired value triggers one retry, followed by another read-back."""

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    closed = _item_list_payload([{"id": "PVTI_target", "status": "Closed"}])
    merged = _item_list_payload([{"id": "PVTI_target", "status": "Merged"}])
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    # First read remains Closed; the single retry changes the second read to Merged.
    with patch.object(
        subprocess,
        "run",
        side_effect=[no_change, _completed(closed), _completed(""), _completed(merged)],
    ) as mock_run:
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)
    assert mock_run.call_count == 4


def test_set_status_merged_retries_when_clean_mutation_reads_back_wrong() -> None:
    """A mutation that exits cleanly but reads back wrong still earns the retry.

    This is the Closed-beats-Merged race itself: item-edit succeeds, a built-in
    workflow overwrites the value, and only the read-back can reveal it. The
    retry must not be conditional on gh's 'no changes to make' error.
    """

    closed = _item_list_payload([{"id": "PVTI_target", "status": "Closed"}])
    merged = _item_list_payload([{"id": "PVTI_target", "status": "Merged"}])
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    # Both mutations exit zero; only the second read-back shows Merged.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_completed(""), _completed(closed), _completed(""), _completed(merged)],
    ) as mock_run:
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)
    assert mock_run.call_count == 4
    # The third call must be the re-run mutation itself, not another read.
    assert mock_run.call_args_list[2].args[0][:3] == ["gh", "project", "item-edit"]


def test_set_status_merged_fails_after_bounded_retry() -> None:
    """Two unverified attempts fail clearly, reporting the exact observed value."""

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    closed = _item_list_payload([{"id": "PVTI_target", "status": "Closed"}])
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    # Both mutation/read pairs leave the value Closed, exhausting the fixed bound;
    # the message must name the observed value so operators see what actually won.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=[no_change, _completed(closed), no_change, _completed(closed)],
        ) as mock_run,
        pytest.raises(smps.ProjectStatusSyncError, match="read back as 'Closed' after 2 attempts"),
    ):
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)
    assert mock_run.call_count == 4


def test_set_status_merged_reports_unset_when_status_never_appears() -> None:
    """A Status that never reads back at all is reported as 'unset', not None."""

    no_status = _item_list_payload([{"id": "PVTI_target"}])
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    # Both mutations exit zero, but the item's Status stays absent throughout, so
    # the failure message must translate the None read-back into 'unset'.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=[
                _completed(""),
                _completed(no_status),
                _completed(""),
                _completed(no_status),
            ],
        ),
        pytest.raises(smps.ProjectStatusSyncError, match="read back as 'unset' after 2 attempts"),
    ):
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)


def test_set_status_merged_propagates_other_gh_errors_without_read_back() -> None:
    """A gh failure other than 'no changes to make' keeps its fail-fast path.

    Only the observed false no-change response is eligible for read-back
    recovery; masking an authentication or schema fault behind a retry would
    hide the actual defect from CI output.
    """

    auth_error = subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope")
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    # The unrelated CLI failure must propagate immediately: no read-back, no retry.
    with (
        patch.object(subprocess, "run", side_effect=auth_error) as mock_run,
        pytest.raises(smps.ProjectStatusSyncError, match="insufficient scope"),
    ):
        smps.set_status_merged("PVTI_target", "PVT_project", field, "Jared-Godar", 5)
    assert mock_run.call_count == 1


# --- _run_gh rate-limit classification and retry ------------------------------------


def test_run_gh_fails_fast_on_primary_rate_limit_without_retrying() -> None:
    """A primary (hours-long) rate limit is fatal immediately, with no retry attempt.

    Retrying inside a single CI job's lifetime cannot help this failure mode
    (it takes up to an hour to clear), so a retry would only waste time and
    delay the same unavoidable failure.
    """

    # GitHub's real wording for this case, observed live against PR #155.
    error = subprocess.CalledProcessError(
        1, ["gh"], stderr="GraphQL: API rate limit already exceeded for user ID 16855088."
    )
    # subprocess.run always raises this same error, so a retry would just hit
    # it again -- the assertions below confirm _run_gh doesn't bother trying.
    with (
        patch.object(subprocess, "run", side_effect=error) as mock_run,
        patch.object(smps.time, "sleep") as mock_sleep,
        pytest.raises(smps.ProjectStatusSyncError, match="rate limit exhausted"),
    ):
        smps._run_gh(["pr", "view", "155"])
    # Exactly one attempt: no retry loop should have run for this error class.
    assert mock_run.call_count == 1
    mock_sleep.assert_not_called()


def test_run_gh_retries_then_succeeds_on_secondary_rate_limit() -> None:
    """A secondary (short-lived, abuse-detection) rate limit is retried and can recover."""

    secondary_error = subprocess.CalledProcessError(
        1, ["gh"], stderr="You have exceeded a secondary rate limit. Please wait."
    )
    success = subprocess.CompletedProcess([], 0, stdout='{"ok": true}', stderr="")
    # Fails twice, then succeeds on the third attempt.
    with (
        patch.object(
            subprocess, "run", side_effect=[secondary_error, secondary_error, success]
        ) as mock_run,
        patch.object(smps.time, "sleep") as mock_sleep,
    ):
        result = smps._run_gh(["pr", "view", "155"])
    assert result == '{"ok": true}'
    assert mock_run.call_count == 3
    # First attempt has no delay; the two retries use the first two entries of
    # the fixed backoff schedule.
    assert [call.args[0] for call in mock_sleep.call_args_list] == [2, 5]


def test_run_gh_raises_after_exhausting_secondary_rate_limit_retries() -> None:
    """A secondary rate limit that never clears still fails, after using every retry slot."""

    secondary_error = subprocess.CalledProcessError(
        1, ["gh"], stderr="You have exceeded a secondary rate limit. Please wait."
    )
    # Every attempt fails the same way, so the retry schedule must exhaust
    # completely before this raises.
    with (
        patch.object(subprocess, "run", side_effect=secondary_error) as mock_run,
        patch.object(smps.time, "sleep"),
        pytest.raises(smps.ProjectStatusSyncError, match="secondary rate limit"),
    ):
        smps._run_gh(["pr", "view", "155"])
    # One initial attempt plus one retry per entry in the backoff schedule.
    assert mock_run.call_count == 1 + len(smps._SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS)


# --- main ------------------------------------------------------------------------------


def test_main_returns_zero_and_warns_when_pr_not_tracked(capsys: pytest.CaptureFixture) -> None:
    """A PR that isn't a Project item is a non-fatal warning (exit 0), not a failure.

    Not every merged PR is necessarily tracked on the project board, so this must
    succeed with a warning rather than treating it as an error.
    """

    # An empty item list means the PR was never added to Project #5.
    with patch.object(subprocess, "run", return_value=_completed(_item_list_payload([]))):
        exit_code = smps.main(["--pr-number", "999999", "--repo", _REPO])
    assert exit_code == 0
    assert "not a Project" in capsys.readouterr().err


def test_main_returns_two_on_gh_failure() -> None:
    """A gh CLI failure (e.g. insufficient token scope) exits with code 2."""

    # gh exits non-zero with a scope-related error message.
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope"),
    ):
        exit_code = smps.main(["--pr-number", "116", "--repo", _REPO])
    assert exit_code == 2
