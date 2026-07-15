"""Tests for the scheduled Project-board drift backstop (issue #233).

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. The pure cross-check logic
(find_board_drift) is tested directly; the main() entry point is tested with
the subprocess boundary mocked -- no test performs a live GitHub call.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Locate the script relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "detect_board_drift.py"
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("detect_board_drift", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. dbd.main).
dbd = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern so the script's sibling imports resolve
# consistently when several governance test modules share one process.
sys.modules[_SPEC.name] = dbd
_SPEC.loader.exec_module(dbd)


def _open_item(
    number: int,
    kind: str = "issue",
    labels: list[str] | None = None,
    author: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one open-item row in the shape fetch_open_items produces.

    Args:
        number: The issue or PR number.
        kind: "issue" or "pull request".
        labels: The item's label names; defaults to none.
        author: gh's author object; defaults to a human maintainer.

    Returns:
        The open-item dict find_board_drift consumes.
    """

    return {
        "kind": kind,
        "number": number,
        "title": f"Item {number}",
        "labels": labels or [],
        "author": author if author is not None else {"login": "Jared-Godar", "is_bot": False},
    }


def _board_item(
    number: int,
    content_type: str = "Issue",
    fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one board-snapshot row in gh's item-list JSON shape.

    Args:
        number: The content number the item tracks.
        content_type: gh's content type, "Issue" or "PullRequest".
        fields: Field JSON key -> value pairs to set on the item (gh flattens
            populated fields into top-level keys like "status" and
            "issue Type").

    Returns:
        The board-item dict find_board_drift consumes.
    """

    item: dict[str, Any] = {"content": {"type": content_type, "number": number}}
    item.update(fields or {})
    return item


# --- pure cross-check logic -----------------------------------------------------------


def test_missing_membership_is_flagged_alone() -> None:
    """An open item absent from the board reports exactly the membership gap."""

    findings = dbd.find_board_drift([_open_item(240, labels=["size: s"])], [])
    assert len(findings) == 1
    # Membership is the only problem reported -- field checks are moot for a
    # non-member, so no unset-field noise accompanies it.
    assert findings[0].problems == ("not a member of the tracked Project",)


def test_fully_converged_item_reports_nothing() -> None:
    """A member with Status and every label-derived field populated is clean."""

    board = [
        _board_item(
            240,
            fields={"status": "Backlog", "size": "S", "issue Type": "Governance"},
        )
    ]
    open_items = [_open_item(240, labels=["size: s", "type: governance"])]
    assert dbd.find_board_drift(open_items, board) == ()


def test_unset_status_and_derivable_fields_are_flagged_together() -> None:
    """A member missing Status and a label-derived field reports both problems."""

    # The board item exists but has neither Status nor Size populated.
    findings = dbd.find_board_drift([_open_item(240, labels=["size: m"])], [_board_item(240)])
    assert len(findings) == 1
    assert "Status is unset" in findings[0].problems
    # The message names the derivation so the reviewer knows what to set.
    assert any("Size is unset" in problem and "'M'" in problem for problem in findings[0].problems)


def test_curated_divergence_is_not_flagged() -> None:
    """A populated field differing from its label derivation is NOT drift.

    Curated values win (house rule from AGENTS.md): the backstop only flags
    unset fields, never a maintainer's deliberate override.
    """

    # Label says p3 -> Low, but the board holds a curated High; that is fine.
    board = [_board_item(240, fields={"status": "Ready", "priority": "High"})]
    assert dbd.find_board_drift([_open_item(240, labels=["priority: p3"])], board) == ()


def test_pull_requests_match_on_the_pull_request_content_type() -> None:
    """A PR is matched against PullRequest board content, not a same-numbered Issue.

    Issues and PRs share one numbering space, so the cross-check must match
    on content type as well as number or a PR could borrow an issue's row.
    """

    # The board tracks issue #240 only; the OPEN PR #240 has no board item.
    findings = dbd.find_board_drift(
        [_open_item(240, kind="pull request")],
        [_board_item(240, content_type="Issue", fields={"status": "Backlog"})],
    )
    assert len(findings) == 1
    assert findings[0].kind == "pull request"


def test_governed_bot_items_are_excluded() -> None:
    """Dependabot-authored items are the autofill path's territory, never flagged here."""

    bot_author = {"login": "app/dependabot", "is_bot": True}
    findings = dbd.find_board_drift([_open_item(240, author=bot_author)], [])
    assert findings == ()


def test_bot_lookalike_login_without_is_bot_is_still_checked() -> None:
    """A human account with a bot-like login does not earn the governed-bot exclusion."""

    lookalike = {"login": "dependabot", "is_bot": False}
    findings = dbd.find_board_drift([_open_item(240, author=lookalike)], [])
    assert len(findings) == 1


def test_draft_board_items_are_ignored_by_the_index() -> None:
    """Draft items (no repository content) never match and never crash the check."""

    draft = {"content": {"type": "DraftIssue", "title": "note"}}
    findings = dbd.find_board_drift([_open_item(240)], [draft])
    # The open issue is still (correctly) reported as a non-member.
    assert len(findings) == 1


# --- main() orchestration and exit codes ------------------------------------------------


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


def _listing(rows: list[dict[str, Any]]) -> subprocess.CompletedProcess:
    """Build a fake `gh issue list`/`gh pr list` JSON response.

    Args:
        rows: The listing rows, each with number/title/labels/author.

    Returns:
        A successful CompletedProcess carrying the listing payload.
    """

    return _completed(json.dumps(rows))


def _snapshot(items: list[dict[str, Any]]) -> subprocess.CompletedProcess:
    """Build a fake `gh project item-list` snapshot response.

    Args:
        items: The board items in gh's item-list JSON shape.

    Returns:
        A successful CompletedProcess carrying the snapshot payload.
    """

    return _completed(json.dumps({"items": items}))


def test_main_exits_zero_when_everything_is_converged(capsys) -> None:
    """A repository with no gaps prints the all-clear and exits 0."""

    issue_row = {
        "number": 240,
        "title": "Item 240",
        "labels": [{"name": "size: s"}],
        "author": {"login": "Jared-Godar", "is_bot": False},
    }
    responses = [
        _quota(4000),
        _listing([issue_row]),
        _listing([]),
        _snapshot([_board_item(240, fields={"status": "Backlog", "size": "S"})]),
        _quota(3790),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses):
        exit_code = dbd.main(["--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 0
    assert "No board drift detected." in capsys.readouterr().out


def test_main_exits_one_and_reports_every_finding(capsys) -> None:
    """Detected drift lists each item's problems and exits 1 for the scheduled run."""

    issue_row = {
        "number": 240,
        "title": "Item 240",
        "labels": [],
        "author": {"login": "Jared-Godar", "is_bot": False},
    }
    responses = [
        _quota(4000),
        _listing([issue_row]),
        _listing([]),
        _snapshot([]),
        _quota(3790),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses):
        exit_code = dbd.main(["--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Board drift detected on 1 item(s):" in err
    assert "#240" in err


def test_main_preflight_stop_maps_to_the_quota_exit_code() -> None:
    """A pool below the snapshot-sized threshold stops the run with exit code 3."""

    # 100 remaining is below the 250 default sized to the ~203-point snapshot;
    # only the free preflight/report rate_limit reads may occur on this path.
    with patch.object(subprocess, "run", side_effect=[_quota(100), _quota(100)]) as mock_run:
        exit_code = dbd.main(["--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 3
    # Nothing beyond the two free rate_limit reads was attempted.
    assert len(mock_run.call_args_list) == 2


def test_main_maps_a_gh_failure_to_the_data_exit_code() -> None:
    """An ordinary gh failure (e.g. auth) exits 2, distinct from drift and quota."""

    failure = subprocess.CalledProcessError(1, ["gh"], output="", stderr="gh: authentication")
    # The listing call fails with a non-transient gh error mid-run.
    with patch.object(subprocess, "run", side_effect=[_quota(4000), failure, _quota(4000)]):
        exit_code = dbd.main(["--repo", "Jared-Godar/ecg_anomaly_detection"])
    assert exit_code == 2
