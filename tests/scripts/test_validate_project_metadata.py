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

# Centralize _SCRIPT_PATH so every caller shares the same documented invariant.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "validate_project_metadata.py"
)
# Centralize _SPEC so every caller shares the same documented invariant.
_SPEC = importlib.util.spec_from_file_location("validate_project_metadata", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# Construct vpm once so the module exposes one stable shared definition.
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
    """Verify that extract closing issue numbers.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        body: The body value supplied by the caller or surrounding test fixture.
        expected: The expected value supplied by the caller or surrounding test fixture.
    """

    assert vpm.extract_closing_issue_numbers(body) == expected


def test_extract_closing_issue_numbers_deduplicates_preserving_order() -> None:
    """Verify that extract closing issue numbers deduplicates preserving order.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert vpm.extract_closing_issue_numbers("Closes #5. Also closes #5. Fixes #3.") == (5, 3)


def test_extract_closing_issue_numbers_does_not_match_a_bare_reference() -> None:
    """Verify that extract closing issue numbers does not match a bare reference.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert vpm.extract_closing_issue_numbers("See #38 for context.") == ()


# --- validate_pull_request ---------------------------------------------------------


def test_complete_pull_request_has_no_violations() -> None:
    """Verify that complete pull request has no violations.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    pr = vpm.PullRequestMetadata(
        number=1,
        assignees=("Jared-Godar",),
        milestone="M5",
        labels=("type: modernization", "area: cli"),
        body="Closes #38",
    )
    assert vpm.validate_pull_request(pr) == ()


def test_pull_request_missing_everything_reports_all_violations() -> None:
    """Verify that pull request missing everything reports all violations.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

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
    """Verify that old style labels without a space are recognized.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    pr = vpm.PullRequestMetadata(
        number=3,
        assignees=("x",),
        milestone="M5",
        labels=("type:modernization", "area:cli"),
        body="Closes #1",
    )
    assert vpm.validate_pull_request(pr) == ()


def test_a_thematic_label_alone_does_not_satisfy_the_type_or_area_requirement() -> None:
    """Verify that a thematic label alone does not satisfy the type or area requirement.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

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


def test_missing_milestone_passes_when_not_required() -> None:
    """Verify that missing milestone passes when not required.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    pr = vpm.PullRequestMetadata(
        number=5,
        assignees=("x",),
        milestone=None,
        labels=("type: documentation", "area: documentation"),
        body="Closes #78",
    )
    violations = vpm.validate_pull_request(pr, require_milestone=False)
    assert "pull request has no milestone" not in violations


def test_missing_milestone_still_fails_when_required() -> None:
    """Verify that missing milestone still fails when required.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    pr = vpm.PullRequestMetadata(
        number=6,
        assignees=("x",),
        milestone=None,
        labels=("type: documentation", "area: documentation"),
        body="Closes #78",
    )
    violations = vpm.validate_pull_request(pr, require_milestone=True)
    assert "pull request has no milestone" in violations


# --- closing_issue_milestones_require_pr_milestone ----------------------------------


def test_milestone_required_when_no_closing_issues_are_known() -> None:
    """Verify that milestone required when no closing issues are known.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert vpm.closing_issue_milestones_require_pr_milestone(()) is True


def test_milestone_not_required_when_every_closing_issue_is_unmilestoned() -> None:
    """Verify that milestone not required when every closing issue is unmilestoned.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert vpm.closing_issue_milestones_require_pr_milestone((None, None)) is False


def test_milestone_required_when_any_closing_issue_has_one() -> None:
    """Verify that milestone required when any closing issue has one.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert vpm.closing_issue_milestones_require_pr_milestone((None, "M8")) is True


def test_milestone_required_when_every_closing_issue_has_one() -> None:
    """Verify that milestone required when every closing issue has one.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    assert vpm.closing_issue_milestones_require_pr_milestone(("M5", "M5")) is True


# --- build_project_field_report / validate_project_membership ----------------------


def _project_item(number: int, **fields: str) -> dict[str, object]:
    """Compute and return project item for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        number: The number value supplied by the caller or surrounding test fixture.
        fields: The fields value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    item: dict[str, object] = {"content": {"type": "Issue", "number": number}}
    item.update(fields)
    return item


def test_issue_not_present_in_project_is_reported_as_a_non_member() -> None:
    """Verify that issue not present in project is reported as a non member.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    report = vpm.build_project_field_report(99, [])
    assert report.is_project_member is False
    assert report.missing_fields == ()
    assert vpm.validate_project_membership(report) == (
        "issue #99 is not a member of the tracked Project",
    )


def test_issue_with_all_required_fields_has_no_violations() -> None:
    """Verify that issue with all required fields has no violations.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

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
    """Verify that issue missing some fields is reported by name.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

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
    """Verify that a matching pull request item is not mistaken for the issue.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    pr_item = {"content": {"type": "PullRequest", "number": 38}}
    report = vpm.build_project_field_report(38, [pr_item])
    assert report.is_project_member is False


# --- I/O boundary (mocked subprocess) -----------------------------------------------


def test_fetch_pull_request_parses_gh_output() -> None:
    """Verify that fetch pull request parses gh output.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    fake_stdout = (
        '{"number": 65, "assignees": [{"login": "Jared-Godar"}], '
        '"milestone": {"title": "M5"}, "labels": [{"name": "type: modernization"}], '
        '"body": "Closes #38"}'
    )
    # Scope `patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
    # stdout=fake_stdout, stderr=''))` here so the expected failure and fixture cleanup stay scoped
    # to this assertion.
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
    """Verify that fetch pull request raises on gh failure.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    # Scope `patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['gh'],
    # stderr='pull request not found')), pytest.raises(vpm.MetadataValidationError, match='pull
    # request not found')` here so the expected failure and fixture cleanup stay scoped to this
    # assertion.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="pull request not found"),
        ),
        pytest.raises(vpm.MetadataValidationError, match="pull request not found"),
    ):
        vpm.fetch_pull_request(999999, repo=None)


def test_fetch_issue_milestone_parses_gh_output_with_a_milestone() -> None:
    """Verify that fetch issue milestone parses gh output with a milestone.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    # Scope `patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
    # stdout='{"milestone": {"title": "M8"}...` here so the expected failure and fixture cleanup
    # stay scoped to this assertion.
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout='{"milestone": {"title": "M8"}}', stderr=""
        ),
    ):
        assert vpm.fetch_issue_milestone(69, repo=None) == "M8"


def test_fetch_issue_milestone_returns_none_when_absent() -> None:
    """Verify that fetch issue milestone returns none when absent.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    # Scope `patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
    # stdout='{"milestone": null}', stderr=...` here so the expected failure and fixture cleanup
    # stay scoped to this assertion.
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout='{"milestone": null}', stderr=""),
    ):
        assert vpm.fetch_issue_milestone(67, repo=None) is None


def test_fetch_issue_milestone_raises_on_gh_failure() -> None:
    """Verify that fetch issue milestone raises on gh failure.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    # Scope `patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['gh'],
    # stderr='issue not found')), pytest.raises(vpm.MetadataValidationError, match='issue not
    # found')` here so the expected failure and fixture cleanup stay scoped to this assertion.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="issue not found"),
        ),
        pytest.raises(vpm.MetadataValidationError, match="issue not found"),
    ):
        vpm.fetch_issue_milestone(999999, repo=None)


def test_fetch_project_items_raises_metadata_validation_error_on_failure() -> None:
    """Verify that fetch project items raises metadata validation error on failure.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    # Scope `patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['gh'],
    # stderr='insufficient scope')), pytest.raises(vpm.MetadataValidationError, match='insufficient
    # scope')` here so the expected failure and fixture cleanup stay scoped to this assertion.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope"),
        ),
        pytest.raises(vpm.MetadataValidationError, match="insufficient scope"),
    ):
        vpm.fetch_project_items("Jared-Godar", 5)
