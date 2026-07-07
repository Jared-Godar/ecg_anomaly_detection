"""Tests for the PR/Project metadata validation script.

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
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "validate_project_metadata.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_project_metadata", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
vpm = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves postponed (`from __future__ import annotations`) type hints via
# sys.modules[cls.__module__], so the module must be registered there before exec_module
# runs a dynamically loaded file's @dataclass decorators.
sys.modules[_SPEC.name] = vpm
_SPEC.loader.exec_module(vpm)


# --- extract_closing_issue_numbers -----------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Closes #38.", (38,)),
        ("Closes #36.\nCloses #60.\nRelated to #61 (not addressed by this PR).", (36, 60)),
        ("no closing reference here", ()),
        ("Fixes #12 and resolves #34", (12, 34)),
        ("closed #5", (5,)),
        ("FIXES #7", (7,)),
        ("", ()),
    ],
)
def test_extract_closing_issue_numbers(body: str, expected: tuple[int, ...]) -> None:
    assert vpm.extract_closing_issue_numbers(body) == expected


def test_extract_closing_issue_numbers_deduplicates_preserving_order() -> None:
    assert vpm.extract_closing_issue_numbers("Closes #5. Also closes #5. Fixes #3.") == (5, 3)


def test_extract_closing_issue_numbers_does_not_match_a_bare_reference() -> None:
    assert vpm.extract_closing_issue_numbers("See #38 for context.") == ()


# --- validate_pull_request ---------------------------------------------------------


def test_complete_pull_request_has_no_violations() -> None:
    pr = vpm.PullRequestMetadata(
        number=1,
        assignees=("Jared-Godar",),
        milestone="M5",
        labels=("type: modernization", "area: cli"),
        body="Closes #38",
    )
    assert vpm.validate_pull_request(pr) == ()


def test_pull_request_missing_everything_reports_all_violations() -> None:
    pr = vpm.PullRequestMetadata(
        number=2, assignees=(), milestone=None, labels=(), body="no reference"
    )
    violations = vpm.validate_pull_request(pr)
    assert len(violations) == 5
    assert "pull request has no assignee" in violations
    assert "pull request has no milestone" in violations
    assert "pull request is missing a type:* label" in violations
    assert "pull request is missing an area:* label" in violations
    assert any("closing issue reference" in v for v in violations)


def test_old_style_labels_without_a_space_are_recognized() -> None:
    pr = vpm.PullRequestMetadata(
        number=3,
        assignees=("x",),
        milestone="M5",
        labels=("type:modernization", "area:cli"),
        body="Closes #1",
    )
    assert vpm.validate_pull_request(pr) == ()


def test_a_thematic_label_alone_does_not_satisfy_the_type_or_area_requirement() -> None:
    pr = vpm.PullRequestMetadata(
        number=4,
        assignees=("x",),
        milestone="M5",
        labels=("portfolio: case-study",),
        body="Closes #1",
    )
    violations = vpm.validate_pull_request(pr)
    assert "pull request is missing a type:* label" in violations
    assert "pull request is missing an area:* label" in violations


# --- build_project_field_report / validate_project_membership ----------------------


def _project_item(number: int, **fields: str) -> dict[str, object]:
    item: dict[str, object] = {"content": {"type": "Issue", "number": number}}
    item.update(fields)
    return item


def test_issue_not_present_in_project_is_reported_as_a_non_member() -> None:
    report = vpm.build_project_field_report(99, [])
    assert report.is_project_member is False
    assert report.missing_fields == ()
    assert vpm.validate_project_membership(report) == (
        "issue #99 is not a member of the tracked Project",
    )


def test_issue_with_all_required_fields_has_no_violations() -> None:
    complete = _project_item(
        38,
        status="Closed",
        workstream="Developer Experience",
        **{"issue Type": "Enhancement"},
        priority="Low",
        risk="Low",
        size="M",
        **{"repository Area": "developer-experience"},
        **{"portfolio Signal": "Developer Experience"},
        **{"target Release": "Future"},
    )
    report = vpm.build_project_field_report(38, [complete])
    assert report.is_project_member is True
    assert report.missing_fields == ()
    assert vpm.validate_project_membership(report) == ()


def test_issue_missing_some_fields_is_reported_by_name() -> None:
    partial = _project_item(38, status="In Progress", workstream="Developer Experience")
    report = vpm.build_project_field_report(38, [partial])
    assert report.is_project_member is True
    assert "Issue Type" in report.missing_fields
    assert "Target Release" in report.missing_fields
    assert "Status" not in report.missing_fields
    violations = vpm.validate_project_membership(report)
    assert len(violations) == 1
    assert "issue #38 is missing Project fields:" in violations[0]


def test_a_matching_pull_request_item_is_not_mistaken_for_the_issue() -> None:
    pr_item = {"content": {"type": "PullRequest", "number": 38}}
    report = vpm.build_project_field_report(38, [pr_item])
    assert report.is_project_member is False


# --- I/O boundary (mocked subprocess) -----------------------------------------------


def test_fetch_pull_request_parses_gh_output() -> None:
    fake_stdout = (
        '{"number": 65, "assignees": [{"login": "Jared-Godar"}], '
        '"milestone": {"title": "M5"}, "labels": [{"name": "type: modernization"}], '
        '"body": "Closes #38"}'
    )
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout=fake_stdout, stderr=""),
    ):
        pr = vpm.fetch_pull_request(65, repo=None)
    assert pr.number == 65
    assert pr.assignees == ("Jared-Godar",)
    assert pr.milestone == "M5"
    assert pr.labels == ("type: modernization",)


def test_fetch_pull_request_raises_on_gh_failure() -> None:
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="pull request not found"),
    ):
        with pytest.raises(vpm.MetadataValidationError, match="pull request not found"):
            vpm.fetch_pull_request(999999, repo=None)


def test_fetch_project_items_raises_metadata_validation_error_on_failure() -> None:
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope"),
    ):
        with pytest.raises(vpm.MetadataValidationError, match="insufficient scope"):
            vpm.fetch_project_items("Jared-Godar", 5)
