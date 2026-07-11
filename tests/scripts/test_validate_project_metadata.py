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

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "validate_project_metadata.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("validate_project_metadata", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. vpm.validate_pull_request).
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
    """GitHub's closing-keyword vocabulary (closes/fixes/resolves, any case) is matched,
    "related to" is not, and an empty body yields no issue numbers at all.

    Args:
        body: A PR body text to scan.
        expected: The issue numbers extract_closing_issue_numbers must return, in
            the order they appear in body.
    """

    assert vpm.extract_closing_issue_numbers(body) == expected


def test_extract_closing_issue_numbers_deduplicates_preserving_order() -> None:
    """The same issue number closed twice in one body appears only once, at its first position."""

    assert vpm.extract_closing_issue_numbers("Closes #5. Also closes #5. Fixes #3.") == (5, 3)


def test_extract_closing_issue_numbers_does_not_match_a_bare_reference() -> None:
    """A bare "#38" with no closing keyword in front of it is not treated as a closing reference."""

    assert vpm.extract_closing_issue_numbers("See #38 for context.") == ()


def test_extract_closing_issue_numbers_ignores_a_backtick_quoted_keyword() -> None:
    """A closing keyword quoted as inline-code prose (e.g. describing another PR's body) is not
    treated as this PR's own closing directive.

    Reproduces the live false positive from issue #161: PR #160's body quoted
    `` `Closes #154` `` as prose describing PR #155's content, and the unguarded regex
    matched it as a real directive for PR #160 itself.
    """

    body = "This PR's body quotes `Closes #154` as an example of PR #155's own wording."
    assert vpm.extract_closing_issue_numbers(body) == ()


def test_extract_closing_issue_numbers_ignores_a_fenced_code_block() -> None:
    """A closing keyword inside a fenced Markdown code block is not treated as a closing
    directive, matching the inline-code-span case but for a multi-line quoted example."""

    body = "Real text.\n```text\nCloses #154\n```\nMore real text.\nCloses #38."
    assert vpm.extract_closing_issue_numbers(body) == (38,)


# --- validate_pull_request ---------------------------------------------------------


def test_complete_pull_request_has_no_violations() -> None:
    """A PR with an assignee, milestone, type:/area: labels, and a closing reference passes cleanly."""

    pr = vpm.PullRequestMetadata(
        number=1,
        assignees=("Jared-Godar",),
        milestone="M5",
        labels=("type: modernization", "area: cli"),
        body="Closes #38",
        state="OPEN",
    )
    assert vpm.validate_pull_request(pr) == ()


def test_pull_request_missing_everything_reports_all_violations() -> None:
    """Every one of the five required PR fields being absent produces five distinct violations.

    Confirms validate_pull_request checks each requirement independently
    rather than short-circuiting on the first failure it finds.
    """

    pr = vpm.PullRequestMetadata(
        number=2, assignees=(), milestone=None, labels=(), body="no reference", state="OPEN"
    )
    violations = vpm.validate_pull_request(pr)
    assert len(violations) == 5
    assert "pull request has no assignee" in violations
    assert "pull request has no milestone" in violations
    assert "pull request is missing a type:* label" in violations
    assert "pull request is missing an area:* label" in violations
    assert any("closing issue reference" in v for v in violations)


def test_old_style_labels_without_a_space_are_recognized() -> None:
    """ "type:modernization" (no space after the colon) still satisfies the type:* requirement.

    The label taxonomy's canonical spelling has a space ("type: modernization"),
    but older labels created before that convention was adopted must still be
    recognized so this check doesn't regress on pre-existing PRs.
    """

    pr = vpm.PullRequestMetadata(
        number=3,
        assignees=("x",),
        milestone="M5",
        labels=("type:modernization", "area:cli"),
        body="Closes #1",
        state="OPEN",
    )
    assert vpm.validate_pull_request(pr) == ()


def test_a_thematic_label_alone_does_not_satisfy_the_type_or_area_requirement() -> None:
    """A "portfolio: case-study" label does not stand in for a required type:* or area:* label.

    Thematic labels and type/area labels are orthogonal categories; having one
    must not mask the absence of the other.
    """

    pr = vpm.PullRequestMetadata(
        number=4,
        assignees=("x",),
        milestone="M5",
        labels=("portfolio: case-study",),
        body="Closes #1",
        state="OPEN",
    )
    violations = vpm.validate_pull_request(pr)
    assert "pull request is missing a type:* label" in violations
    assert "pull request is missing an area:* label" in violations


def test_missing_milestone_passes_when_not_required() -> None:
    """With require_milestone=False, a PR with no milestone reports no milestone violation.

    Milestone is only mandatory once a PR's closing issues determine it is
    (see closing_issue_milestones_require_pr_milestone below); the flag lets
    validate_pull_request be called either way.
    """

    pr = vpm.PullRequestMetadata(
        number=5,
        assignees=("x",),
        milestone=None,
        labels=("type: documentation", "area: documentation"),
        body="Closes #78",
        state="OPEN",
    )
    violations = vpm.validate_pull_request(pr, require_milestone=False)
    assert "pull request has no milestone" not in violations


def test_missing_milestone_still_fails_when_required() -> None:
    """With require_milestone=True, a PR with no milestone reports the milestone violation."""

    pr = vpm.PullRequestMetadata(
        number=6,
        assignees=("x",),
        milestone=None,
        labels=("type: documentation", "area: documentation"),
        body="Closes #78",
        state="OPEN",
    )
    violations = vpm.validate_pull_request(pr, require_milestone=True)
    assert "pull request has no milestone" in violations


# --- closing_issue_milestones_require_pr_milestone ----------------------------------


def test_milestone_required_when_no_closing_issues_are_known() -> None:
    """An empty tuple of closing-issue milestones defaults to requiring a PR milestone.

    With no closing issues to consult, there's no information suggesting the
    milestone requirement can be relaxed, so the conservative default applies.
    """

    assert vpm.closing_issue_milestones_require_pr_milestone(()) is True


def test_milestone_not_required_when_every_closing_issue_is_unmilestoned() -> None:
    """If every issue the PR closes is itself unmilestoned, the PR need not have one either."""

    assert vpm.closing_issue_milestones_require_pr_milestone((None, None)) is False


def test_milestone_required_when_any_closing_issue_has_one() -> None:
    """A mix of one milestoned and one unmilestoned closing issue still requires a PR milestone.

    Any milestoned closing issue is enough to trigger the requirement, even
    when it's not unanimous.
    """

    assert vpm.closing_issue_milestones_require_pr_milestone((None, "M8")) is True


def test_milestone_required_when_every_closing_issue_has_one() -> None:
    """When every closing issue shares the same milestone, the PR is still required to have one."""

    assert vpm.closing_issue_milestones_require_pr_milestone(("M5", "M5")) is True


# --- build_project_field_report / validate_project_membership ----------------------


def _project_item(number: int, **fields: str) -> dict[str, object]:
    """Build a fake Project #5 item dict shaped like `gh project item-list`'s JSON output.

    Args:
        number: The GitHub issue number this item represents.
        fields: Project field name/value pairs to attach (e.g. status="Closed").

    Returns:
        A dict with a nested "content" object of type "Issue" plus the given fields.
    """

    item: dict[str, object] = {"content": {"type": "Issue", "number": number}}
    item.update(fields)
    return item


def test_issue_not_present_in_project_is_reported_as_a_non_member() -> None:
    """An issue absent from the Project #5 item list is reported as a non-member, not as missing fields."""

    report = vpm.build_project_field_report(99, [])
    assert report.is_project_member is False
    assert report.missing_fields == ()
    assert vpm.validate_project_membership(report) == (
        "issue #99 is not a member of the tracked Project",
    )


def test_issue_with_all_required_fields_has_no_violations() -> None:
    """An issue carrying every one of the nine required Project #5 fields passes with no violations."""

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
    """Only the specific fields absent from the item (e.g. "Issue Type") are named as missing.

    "status" and "workstream" are present on the fixture item, so they must not
    appear in missing_fields even though most other required fields are absent.
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
    """A Project item with the same number but content.type "PullRequest" is not counted as the issue.

    Issue and PR numbers share the same numbering space in a GitHub repo, so
    matching must check content.type, not just the number, to avoid a false
    membership match.
    """

    pr_item = {"content": {"type": "PullRequest", "number": 38}}
    report = vpm.build_project_field_report(38, [pr_item])
    assert report.is_project_member is False


# --- find_prematurely_closed_issues (issue #158) ------------------------------------


def test_find_prematurely_closed_issues_flags_a_non_merge_closure() -> None:
    """An issue closed by a non-merge event produces a warning naming that issue."""

    state = vpm.IssueClosureState(154, is_closed=True, closed_via_commit=False)
    warnings = vpm.find_prematurely_closed_issues((state,))
    assert len(warnings) == 1
    assert "issue #154" in warnings[0]
    assert "non-merge event" in warnings[0]


def test_find_prematurely_closed_issues_does_not_flag_a_merge_linked_closure() -> None:
    """An issue closed via a merged commit/PR reference is not flagged; that's the expected path."""

    state = vpm.IssueClosureState(38, is_closed=True, closed_via_commit=True)
    assert vpm.find_prematurely_closed_issues((state,)) == ()


def test_find_prematurely_closed_issues_does_not_flag_an_open_issue() -> None:
    """An issue that is still open is never flagged, regardless of closed_via_commit's value.

    closed_via_commit is meaningless for an open issue (fetch_issue_closure_state
    always returns False for it); is_closed is the gating condition.
    """

    state = vpm.IssueClosureState(38, is_closed=False, closed_via_commit=False)
    assert vpm.find_prematurely_closed_issues((state,)) == ()


def test_find_prematurely_closed_issues_reports_only_the_flagged_ones() -> None:
    """Given a mix of closure states, only the non-merge-closed issue is named in the output."""

    states = (
        vpm.IssueClosureState(1, is_closed=False, closed_via_commit=False),
        vpm.IssueClosureState(2, is_closed=True, closed_via_commit=True),
        vpm.IssueClosureState(3, is_closed=True, closed_via_commit=False),
    )
    warnings = vpm.find_prematurely_closed_issues(states)
    assert len(warnings) == 1
    assert "issue #3" in warnings[0]


# --- I/O boundary (mocked subprocess) -----------------------------------------------


def test_fetch_pull_request_parses_gh_output() -> None:
    """fetch_pull_request converts `gh pr view`'s JSON shape into a flat PullRequestMetadata.

    Confirms the nested assignee/milestone/label objects in gh's JSON are
    each reduced to the flat strings validate_pull_request expects.
    """

    fake_stdout = (
        '{"number": 65, "assignees": [{"login": "Jared-Godar"}], '
        '"milestone": {"title": "M5"}, "labels": [{"name": "type: modernization"}], '
        '"body": "Closes #38", "state": "OPEN"}'
    )
    # gh's real `pr view --json` output nests assignees/milestone/labels as objects.
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
    assert pr.state == "OPEN"


def test_fetch_pull_request_raises_on_gh_failure() -> None:
    """A failing `gh pr view` invocation is translated into MetadataValidationError with gh's stderr."""

    # gh exits non-zero when the PR number doesn't exist.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="pull request not found"),
        ),
        pytest.raises(vpm.MetadataValidationError, match="pull request not found"),
    ):
        vpm.fetch_pull_request(999999, repo=None)


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
        patch.object(vpm.time, "sleep") as mock_sleep,
        pytest.raises(vpm.MetadataValidationError, match="rate limit exhausted"),
    ):
        vpm._run_gh(["pr", "view", "155"])
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
        patch.object(vpm.time, "sleep") as mock_sleep,
    ):
        result = vpm._run_gh(["pr", "view", "155"])
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
        patch.object(vpm.time, "sleep"),
        pytest.raises(vpm.MetadataValidationError, match="secondary rate limit"),
    ):
        vpm._run_gh(["pr", "view", "155"])
    # One initial attempt plus one retry per entry in the backoff schedule.
    assert mock_run.call_count == 1 + len(vpm._SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS)


def test_fetch_issue_milestone_parses_gh_output_with_a_milestone() -> None:
    """fetch_issue_milestone extracts the milestone title from `gh issue view`'s JSON output."""

    # gh's real `issue view --json milestone` output nests the title under "milestone".
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout='{"milestone": {"title": "M8"}}', stderr=""
        ),
    ):
        assert vpm.fetch_issue_milestone(69, repo=None) == "M8"


def test_fetch_issue_milestone_returns_none_when_absent() -> None:
    """An issue with no milestone assigned returns None, not an error or an empty string."""

    # gh reports a JSON null when the issue has no milestone set.
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout='{"milestone": null}', stderr=""),
    ):
        assert vpm.fetch_issue_milestone(67, repo=None) is None


def test_fetch_issue_milestone_raises_on_gh_failure() -> None:
    """A failing `gh issue view` invocation is translated into MetadataValidationError with gh's stderr."""

    # gh exits non-zero when the issue number doesn't exist.
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
    """A failing `gh project item-list` invocation is translated into MetadataValidationError.

    "insufficient scope" is a realistic failure here: reading Project #5
    items requires a `gh` token with the `project` scope granted.
    """

    # gh exits non-zero when the token lacks the project scope.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope"),
        ),
        pytest.raises(vpm.MetadataValidationError, match="insufficient scope"),
    ):
        vpm.fetch_project_items("Jared-Godar", 5)


# --- fetch_issue_closure_state (issue #158) -----------------------------------------


def _timeline_pages(events: list[dict[str, object]]) -> str:
    """Build a fake `gh api ... --paginate --slurp` JSON payload of one timeline page.

    Args:
        events: The timeline event objects to include on that single page.

    Returns:
        JSON text matching gh's --paginate --slurp output shape: an array of pages,
        each page itself an array of event objects.
    """

    import json

    return json.dumps([events])


def test_fetch_issue_closure_state_returns_not_closed_for_an_open_issue() -> None:
    """An open issue is reported as not closed, with no timeline lookup performed.

    Only one gh invocation (the state check) should happen; fetching the
    timeline for an issue that isn't even closed would be wasted work.
    """

    # gh reports state "OPEN" for an issue that has never been closed.
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout='{"state": "OPEN"}', stderr=""),
    ) as mock_run:
        state = vpm.fetch_issue_closure_state(38, repo=None)
    assert state == vpm.IssueClosureState(38, is_closed=False, closed_via_commit=False)
    assert mock_run.call_count == 1


def test_fetch_issue_closure_state_detects_a_merge_linked_closure() -> None:
    """A closed timeline event carrying a commit_id is recognized as a merge-linked closure."""

    state_response = subprocess.CompletedProcess([], 0, stdout='{"state": "CLOSED"}', stderr="")
    timeline_response = subprocess.CompletedProcess(
        [], 0, stdout=_timeline_pages([{"event": "closed", "commit_id": "abc123"}]), stderr=""
    )
    # First gh call checks state; second fetches the timeline once state is CLOSED.
    with patch.object(subprocess, "run", side_effect=[state_response, timeline_response]):
        state = vpm.fetch_issue_closure_state(38, repo=None)
    assert state == vpm.IssueClosureState(38, is_closed=True, closed_via_commit=True)


def test_fetch_issue_closure_state_detects_a_non_merge_closure() -> None:
    """A closed timeline event with commit_id null is recognized as a direct, manual closure.

    Mirrors the real, live evidence behind issue #158: issue #154's timeline
    showed exactly this shape (a closed event with no commit_id) while its
    fixing PR #155 was still open.
    """

    state_response = subprocess.CompletedProcess([], 0, stdout='{"state": "CLOSED"}', stderr="")
    timeline_response = subprocess.CompletedProcess(
        [], 0, stdout=_timeline_pages([{"event": "closed", "commit_id": None}]), stderr=""
    )
    # First gh call checks state; second fetches the timeline once state is CLOSED.
    with patch.object(subprocess, "run", side_effect=[state_response, timeline_response]):
        state = vpm.fetch_issue_closure_state(154, repo=None)
    assert state == vpm.IssueClosureState(154, is_closed=True, closed_via_commit=False)


def test_fetch_issue_closure_state_uses_the_most_recent_closed_event() -> None:
    """When an issue was closed, reopened, and closed again, only the latest closure counts.

    The first closure here is merge-linked but the issue was later reopened and
    then closed a second time manually; the manual (most recent) closure must be
    the one that determines closed_via_commit, not the earlier merge-linked one.
    """

    state_response = subprocess.CompletedProcess([], 0, stdout='{"state": "CLOSED"}', stderr="")
    timeline_response = subprocess.CompletedProcess(
        [],
        0,
        stdout=_timeline_pages(
            [
                {"event": "closed", "commit_id": "abc123"},
                {"event": "reopened"},
                {"event": "closed", "commit_id": None},
            ]
        ),
        stderr="",
    )
    # Same two-call sequence as the other fetch_issue_closure_state tests: state
    # check first, then the timeline once state is confirmed CLOSED.
    with patch.object(subprocess, "run", side_effect=[state_response, timeline_response]):
        state = vpm.fetch_issue_closure_state(38, repo=None)
    assert state.closed_via_commit is False


def test_fetch_issue_closure_state_passes_explicit_repo_directly_in_the_endpoint() -> None:
    """With an explicit repo, the timeline endpoint embeds it directly (gh api has no --repo flag)."""

    state_response = subprocess.CompletedProcess([], 0, stdout='{"state": "CLOSED"}', stderr="")
    timeline_response = subprocess.CompletedProcess(
        [], 0, stdout=_timeline_pages([{"event": "closed", "commit_id": "abc123"}]), stderr=""
    )
    # Capture the actual gh invocations to inspect the timeline call's endpoint below.
    with patch.object(
        subprocess, "run", side_effect=[state_response, timeline_response]
    ) as mock_run:
        vpm.fetch_issue_closure_state(38, repo="Jared-Godar/ecg_anomaly_detection")
    timeline_call_args = mock_run.call_args_list[1].args[0]
    assert "repos/Jared-Godar/ecg_anomaly_detection/issues/38/timeline" in timeline_call_args
    assert "--paginate" in timeline_call_args
    assert "--slurp" in timeline_call_args


def test_fetch_issue_closure_state_raises_on_gh_failure() -> None:
    """A failing `gh issue view` state check is translated into MetadataValidationError."""

    # gh exits non-zero when the issue number doesn't exist.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="issue not found"),
        ),
        pytest.raises(vpm.MetadataValidationError, match="issue not found"),
    ):
        vpm.fetch_issue_closure_state(999999, repo=None)
