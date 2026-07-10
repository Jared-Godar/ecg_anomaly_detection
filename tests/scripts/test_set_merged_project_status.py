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

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "set_merged_project_status.py"
)
_SPEC = importlib.util.spec_from_file_location("set_merged_project_status", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smps = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = smps
_SPEC.loader.exec_module(smps)


_REPO = "Jared-Godar/ecg_anomaly_detection"


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


# --- fetch_project_id ----------------------------------------------------------------


def test_fetch_project_id_parses_gh_output() -> None:
    with patch.object(subprocess, "run", return_value=_completed('{"id": "PVT_kwHOAQEwMM4BcY39"}')):
        assert smps.fetch_project_id("Jared-Godar", 5) == "PVT_kwHOAQEwMM4BcY39"


def test_fetch_project_id_raises_on_gh_failure() -> None:
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
    import json

    return json.dumps({"fields": [{"id": "PVTSSF_status", "name": "Status", "options": options}]})


def test_fetch_status_field_returns_field_and_merged_option_ids() -> None:
    payload = _status_field_payload(
        [{"id": "backlog", "name": "Backlog"}, {"id": "merged", "name": "Merged"}]
    )
    with patch.object(subprocess, "run", return_value=_completed(payload)):
        field = smps.fetch_status_field("Jared-Godar", 5)
    assert field.field_id == "PVTSSF_status"
    assert field.merged_option_id == "merged"


def test_fetch_status_field_raises_when_status_field_missing() -> None:
    import json

    payload = json.dumps({"fields": [{"id": "x", "name": "Priority", "options": []}]})
    with (
        patch.object(subprocess, "run", return_value=_completed(payload)),
        pytest.raises(smps.ProjectStatusSyncError, match="no 'Status' field"),
    ):
        smps.fetch_status_field("Jared-Godar", 5)


def test_fetch_status_field_raises_when_merged_option_missing() -> None:
    payload = _status_field_payload([{"id": "backlog", "name": "Backlog"}])
    with (
        patch.object(subprocess, "run", return_value=_completed(payload)),
        pytest.raises(smps.ProjectStatusSyncError, match="no 'Merged' option"),
    ):
        smps.fetch_status_field("Jared-Godar", 5)


# --- find_pull_request_item_id ---------------------------------------------------------


def _item_list_payload(items: list[dict[str, object]]) -> str:
    import json

    return json.dumps({"items": items})


def test_find_pull_request_item_id_matches_type_number_and_repo() -> None:
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
    with patch.object(subprocess, "run", return_value=_completed(_item_list_payload(items))):
        item_id = smps.find_pull_request_item_id("Jared-Godar", 5, _REPO, 116)
    assert item_id == "PVTI_target"


def test_find_pull_request_item_id_returns_none_when_not_tracked() -> None:
    with patch.object(subprocess, "run", return_value=_completed(_item_list_payload([]))):
        assert smps.find_pull_request_item_id("Jared-Godar", 5, _REPO, 999999) is None


# --- set_status_merged -----------------------------------------------------------------


def test_set_status_merged_invokes_item_edit_with_expected_args() -> None:
    field = smps.StatusField(field_id="PVTSSF_status", merged_option_id="merged")
    with patch.object(subprocess, "run", return_value=_completed("")) as mock_run:
        smps.set_status_merged("PVTI_target", "PVT_project", field)
    called_args = mock_run.call_args.args[0]
    assert called_args[:3] == ["gh", "project", "item-edit"]
    assert "--id" in called_args and "PVTI_target" in called_args
    assert "--project-id" in called_args and "PVT_project" in called_args
    assert "--field-id" in called_args and "PVTSSF_status" in called_args
    assert "--single-select-option-id" in called_args and "merged" in called_args


# --- main ------------------------------------------------------------------------------


def test_main_returns_zero_and_warns_when_pr_not_tracked(capsys: pytest.CaptureFixture) -> None:
    with patch.object(subprocess, "run", return_value=_completed(_item_list_payload([]))):
        exit_code = smps.main(["--pr-number", "999999", "--repo", _REPO])
    assert exit_code == 0
    assert "not a Project" in capsys.readouterr().err


def test_main_returns_two_on_gh_failure() -> None:
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope"),
    ):
        exit_code = smps.main(["--pr-number", "116", "--repo", _REPO])
    assert exit_code == 2
