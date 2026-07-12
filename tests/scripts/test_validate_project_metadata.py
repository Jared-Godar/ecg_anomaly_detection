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

# The shared access layer the script imports; referenced directly for its error types.
_gha = vpm.github_api


def _completed(payload: object) -> subprocess.CompletedProcess:
    """Build a fake successful subprocess.CompletedProcess whose stdout is JSON.

    Args:
        payload: The object `gh` would have printed as JSON.

    Returns:
        A CompletedProcess with returncode 0 and empty stderr.
    """

    import json

    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")


def _pr_payload(body: str = "Closes #38", state: str = "open") -> dict[str, object]:
    """Build a complete REST pull-request payload that passes every PR-level check.

    Args:
        body: The PR body text (controls closing references).
        state: REST's lowercase state value ("open"/"closed").

    Returns:
        A dict shaped like `gh api repos/.../pulls/N` output.
    """

    return {
        "number": 65,
        "assignees": [{"login": "Jared-Godar"}],
        "milestone": {"title": "M5"},
        "labels": [{"name": "type: modernization"}, {"name": "area: ci-cd"}],
        "body": body,
        "state": state,
    }


def _issue_payload(milestone: str | None = "M5", state: str = "open") -> dict[str, object]:
    """Build a REST issue payload carrying the two fields fetch_issue_overview reads.

    Args:
        milestone: The issue's milestone title, or None for unmilestoned.
        state: REST's lowercase state value ("open"/"closed").

    Returns:
        A dict shaped like `gh api repos/.../issues/N` output.
    """

    return {
        "milestone": {"title": milestone} if milestone else None,
        "state": state,
    }


def _quota(remaining: int) -> subprocess.CompletedProcess:
    """Build a fake `gh api rate_limit` response with the given GraphQL points remaining.

    Args:
        remaining: The GraphQL points remaining to report.

    Returns:
        A successful CompletedProcess carrying the rate_limit payload.
    """

    return _completed(
        {
            "resources": {
                "graphql": {
                    "limit": 5000,
                    "used": 5000 - remaining,
                    "remaining": remaining,
                    "reset": 1770000000,
                }
            }
        }
    )


def _complete_item(number: int) -> dict[str, object]:
    """Build a Project item for one issue with every required field populated.

    Args:
        number: The GitHub issue number the item represents.

    Returns:
        A dict shaped like one entry of `gh project item-list`'s items array.
    """

    return {
        "content": {"type": "Issue", "number": number},
        "status": "In Progress",
        "workstream": "Governance",
        "issue Type": "Governance",
        "priority": "High",
        "risk": "Medium",
        "size": "M",
        "repository Area": "ci-cd",
        "portfolio Signal": "Operational Maturity",
        "target Release": "Future",
    }


def test_fetch_pull_request_parses_rest_output() -> None:
    """fetch_pull_request converts REST's JSON shape into a flat PullRequestMetadata.

    Confirms the nested assignee/milestone/label objects are reduced to the flat
    strings validate_pull_request expects, and that REST's lowercase state is
    normalized to the uppercase form the rest of the module compares against.
    """

    # REST reports state lowercase; everything else nests objects like GraphQL did.
    with patch.object(subprocess, "run", return_value=_completed(_pr_payload())) as mock_run:
        pr = vpm.fetch_pull_request(65, repo=None)
    assert pr.number == 65
    assert pr.assignees == ("Jared-Godar",)
    assert pr.milestone == "M5"
    assert "type: modernization" in pr.labels
    assert pr.state == "OPEN"
    # The read must be REST (gh api repos/...), with gh's own {owner}/{repo}
    # placeholders left for it to resolve when no explicit repo is given.
    assert mock_run.call_args_list[0].args[0] == ["gh", "api", "repos/{owner}/{repo}/pulls/65"]


def test_fetch_pull_request_embeds_an_explicit_repo_in_the_endpoint() -> None:
    """With an explicit repo, the REST endpoint embeds it directly (gh api has no --repo)."""

    # The explicit slug replaces the placeholders in the endpoint path.
    with patch.object(subprocess, "run", return_value=_completed(_pr_payload())) as mock_run:
        vpm.fetch_pull_request(65, repo="Jared-Godar/ecg_anomaly_detection")
    assert mock_run.call_args_list[0].args[0] == [
        "gh",
        "api",
        "repos/Jared-Godar/ecg_anomaly_detection/pulls/65",
    ]


def test_fetch_pull_request_raises_on_gh_failure() -> None:
    """A failing REST read is translated into the shared GitHubApiError with gh's stderr."""

    # gh exits non-zero when the PR number doesn't exist.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="pull request not found"),
        ),
        pytest.raises(_gha.GitHubApiError, match="pull request not found"),
    ):
        vpm.fetch_pull_request(999999, repo=None)


# --- fetch_issue_overview -------------------------------------------------------------


def test_fetch_issue_overview_reads_milestone_and_state_in_one_call() -> None:
    """One REST read yields both the milestone and the open/closed state.

    This single read is what both the milestone-inheritance and premature-
    closure stages reuse -- the request-deduplication rule from issue #173.
    """

    # A milestoned, closed issue exercises both extracted fields at once.
    with patch.object(
        subprocess, "run", return_value=_completed(_issue_payload("M8", state="closed"))
    ) as mock_run:
        overview = vpm.fetch_issue_overview(69, repo=None)
    assert overview.number == 69
    assert overview.milestone == "M8"
    assert overview.is_closed is True
    assert mock_run.call_count == 1


def test_fetch_issue_overview_returns_none_milestone_when_absent() -> None:
    """An issue with no milestone assigned reads as None, not an error or empty string."""

    # REST reports a JSON null when the issue has no milestone set.
    with patch.object(
        subprocess, "run", return_value=_completed(_issue_payload(None, state="open"))
    ):
        overview = vpm.fetch_issue_overview(67, repo=None)
    assert overview.milestone is None
    assert overview.is_closed is False


def test_fetch_issue_overview_raises_on_gh_failure() -> None:
    """A failing REST read is translated into the shared GitHubApiError with gh's stderr."""

    # gh exits non-zero when the issue number doesn't exist.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="issue not found"),
        ),
        pytest.raises(_gha.GitHubApiError, match="issue not found"),
    ):
        vpm.fetch_issue_overview(999999, repo=None)


# --- fetch_issue_closure_state (issue #158) -----------------------------------------


def _timeline_pages(events: list[dict[str, object]]) -> subprocess.CompletedProcess:
    """Build a fake `gh api ... --paginate --slurp` response of one timeline page.

    Args:
        events: The timeline event objects to include on that single page.

    Returns:
        A successful CompletedProcess whose stdout matches gh's --paginate --slurp
        output shape: an array of pages, each page itself an array of event objects.
    """

    return _completed([events])


def test_fetch_issue_closure_state_skips_the_timeline_for_an_open_issue() -> None:
    """An open issue is reported as not closed, with no gh call at all.

    The open/closed state arrives on the already-fetched overview, so an open
    issue must cost zero additional API reads here.
    """

    overview = vpm.IssueOverview(38, milestone=None, is_closed=False)
    # No subprocess mock response is provided: any call would fail the test.
    with patch.object(subprocess, "run") as mock_run:
        state = vpm.fetch_issue_closure_state(overview, repo=None)
    assert state == vpm.IssueClosureState(38, is_closed=False, closed_via_commit=False)
    mock_run.assert_not_called()


def test_fetch_issue_closure_state_detects_a_merge_linked_closure() -> None:
    """A closed timeline event carrying a commit_id is recognized as a merge-linked closure."""

    overview = vpm.IssueOverview(38, milestone="M5", is_closed=True)
    # The one gh call fetches the timeline; state came from the overview.
    with patch.object(
        subprocess,
        "run",
        return_value=_timeline_pages([{"event": "closed", "commit_id": "abc123"}]),
    ) as mock_run:
        state = vpm.fetch_issue_closure_state(overview, repo=None)
    assert state == vpm.IssueClosureState(38, is_closed=True, closed_via_commit=True)
    assert mock_run.call_count == 1


def test_fetch_issue_closure_state_detects_a_non_merge_closure() -> None:
    """A closed timeline event with commit_id null is recognized as a direct, manual closure.

    Mirrors the real, live evidence behind issue #158: issue #154's timeline
    showed exactly this shape (a closed event with no commit_id) while its
    fixing PR #155 was still open.
    """

    overview = vpm.IssueOverview(154, milestone="M5", is_closed=True)
    # The timeline's most recent closed event carries no commit_id.
    with patch.object(
        subprocess,
        "run",
        return_value=_timeline_pages([{"event": "closed", "commit_id": None}]),
    ):
        state = vpm.fetch_issue_closure_state(overview, repo=None)
    assert state == vpm.IssueClosureState(154, is_closed=True, closed_via_commit=False)


def test_fetch_issue_closure_state_uses_the_most_recent_closed_event() -> None:
    """When an issue was closed, reopened, and closed again, only the latest closure counts.

    The first closure here is merge-linked but the issue was later reopened and
    then closed a second time manually; the manual (most recent) closure must be
    the one that determines closed_via_commit, not the earlier merge-linked one.
    """

    overview = vpm.IssueOverview(38, milestone="M5", is_closed=True)
    # Three timeline events model the close/reopen/close-manually history.
    with patch.object(
        subprocess,
        "run",
        return_value=_timeline_pages(
            [
                {"event": "closed", "commit_id": "abc123"},
                {"event": "reopened"},
                {"event": "closed", "commit_id": None},
            ]
        ),
    ):
        state = vpm.fetch_issue_closure_state(overview, repo=None)
    assert state.closed_via_commit is False


def test_fetch_issue_closure_state_passes_explicit_repo_directly_in_the_endpoint() -> None:
    """With an explicit repo, the timeline endpoint embeds it directly (gh api has no --repo flag)."""

    overview = vpm.IssueOverview(38, milestone="M5", is_closed=True)
    # Capture the actual gh invocation to inspect the timeline call's endpoint below.
    with patch.object(
        subprocess,
        "run",
        return_value=_timeline_pages([{"event": "closed", "commit_id": "abc123"}]),
    ) as mock_run:
        vpm.fetch_issue_closure_state(overview, repo="Jared-Godar/ecg_anomaly_detection")
    timeline_call_args = mock_run.call_args_list[0].args[0]
    assert "repos/Jared-Godar/ecg_anomaly_detection/issues/38/timeline" in timeline_call_args
    assert "--paginate" in timeline_call_args
    assert "--slurp" in timeline_call_args


# --- fetch_project_items --------------------------------------------------------------


def test_fetch_project_items_raises_the_shared_error_on_failure() -> None:
    """A failing `gh project item-list` invocation is translated into GitHubApiError.

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
        pytest.raises(_gha.GitHubApiError, match="insufficient scope"),
    ):
        vpm.fetch_project_items("Jared-Godar", 5)


# --- main: quota safeguards, deduplication, and snapshot reuse (issue #173) -----------


def test_main_fetches_each_closing_issue_once_across_stages(
    capsys: pytest.CaptureFixture,
) -> None:
    """One closing issue costs exactly one native REST read across both consuming stages.

    The milestone-inheritance stage and the premature-closure stage previously
    each fetched the same issue; the deduplicated overview must satisfy both
    from a single read (the request-deduplication regression from issue #173).
    """

    # Full sequence: PR read, ONE issue read (open, so no timeline), quota
    # preflight, the single Project snapshot, quota report.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _completed(_issue_payload()),
            _quota(4988),
            _completed({"items": [_complete_item(38)]}),
            _quota(4985),
        ],
    ) as mock_run:
        exit_code = vpm.main(["--pr-number", "65", "--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 0
    # Exactly one invocation may touch the issue's native endpoint.
    issue_reads = [
        call for call in mock_run.call_args_list if any("issues/38" in arg for arg in call.args[0])
    ]
    assert len(issue_reads) == 1
    # The run's consumption report is printed for the GraphQL phase.
    assert "3 consumed" in capsys.readouterr().err


def test_main_fetches_the_project_snapshot_once_for_multiple_closing_issues() -> None:
    """Two closing issues share one Project snapshot -- never one item-list per issue.

    This is the one-snapshot-per-phase regression from issue #173: membership
    and field checks for every closing issue must reuse the same fetched list.
    """

    # Both issues are open (no timeline reads); the snapshot carries them both.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload(body="Closes #38. Closes #40.")),
            _completed(_issue_payload()),
            _completed(_issue_payload()),
            _quota(4988),
            _completed({"items": [_complete_item(38), _complete_item(40)]}),
            _quota(4985),
        ],
    ) as mock_run:
        exit_code = vpm.main(["--pr-number", "65", "--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 0
    # Exactly one board-wide item-list across the whole run.
    snapshot_reads = [call for call in mock_run.call_args_list if "item-list" in call.args[0]]
    assert len(snapshot_reads) == 1


def test_main_stops_with_exit_three_before_the_snapshot_when_quota_is_low(
    capsys: pytest.CaptureFixture,
) -> None:
    """A preflight below the threshold exits 3 without fetching the Project snapshot.

    The quota condition must not masquerade as a metadata violation (exit 1)
    or an unreadable-project failure (exit 2/strict violation): it is shared-
    pool infrastructure with its own exit code and "quota:" prefix.
    """

    # The preflight (and the report after it) see a nearly drained pool.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _completed(_issue_payload()),
            _quota(12),
            _quota(12),
        ],
    ) as mock_run:
        exit_code = vpm.main(
            [
                "--pr-number",
                "65",
                "--repo",
                "Jared-Godar/ecg_anomaly_detection",
                "--strict-project-checks",
                "--min-graphql-quota",
                "50",
            ]
        )
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err
    # The expensive snapshot must never have been requested.
    for call in mock_run.call_args_list:
        assert "item-list" not in call.args[0]


def test_main_maps_a_primary_rate_limit_to_exit_three(capsys: pytest.CaptureFixture) -> None:
    """Primary rate-limit exhaustion exits 3, distinct from ordinary fetch failures (exit 2)."""

    exhausted = subprocess.CalledProcessError(
        1, ["gh"], stderr="GraphQL: API rate limit already exceeded for user ID 16855088."
    )
    # The very first read hits the drained pool; no preflight ever ran, so no
    # consumption report is expected either.
    with patch.object(subprocess, "run", side_effect=exhausted):
        exit_code = vpm.main(["--pr-number", "65", "--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err


def test_main_warns_and_skips_project_checks_when_not_strict(
    capsys: pytest.CaptureFixture,
) -> None:
    """Without --strict-project-checks, an unreadable Project degrades to a warning.

    The PR-level checks still enforce; only the Project field checks are
    skipped -- preserving the token-rollout behavior documented in the module
    docstring across the issue #173 refactor.
    """

    scope_error = subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope")
    # The snapshot read fails after a passing preflight; the report still runs.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _completed(_issue_payload()),
            _quota(4988),
            scope_error,
            _quota(4988),
        ],
    ):
        exit_code = vpm.main(["--pr-number", "65", "--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 0
    assert "skipping Project field checks" in capsys.readouterr().err


def test_main_flags_a_manually_closed_issue_from_the_shared_overview(
    capsys: pytest.CaptureFixture,
) -> None:
    """A closing issue closed by a non-merge event is warned about, reusing the overview.

    The closure stage must draw the issue's closed state from the overview
    (one native read total) and spend its own gh call only on the timeline.
    """

    # The issue is closed, so the sequence gains exactly one timeline read.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _completed(_issue_payload(state="closed")),
            _timeline_pages([{"event": "closed", "commit_id": None}]),
            _quota(4988),
            _completed({"items": [_complete_item(38)]}),
            _quota(4985),
        ],
    ) as mock_run:
        exit_code = vpm.main(["--pr-number", "65", "--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 0
    assert "non-merge event" in capsys.readouterr().err
    # Still exactly one native read of the issue itself (the timeline endpoint
    # is a different resource path and is allowed its own single read).
    issue_reads = [
        call
        for call in mock_run.call_args_list
        if any(arg.endswith("issues/38") for arg in call.args[0])
    ]
    assert len(issue_reads) == 1
