"""Tests for the creation-time Project #5 board-population script (issue #233).

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test mocks the subprocess
boundary; none performs a live GitHub call.

The shared GitHub access layer and the mapping table have their own test
modules; the tests here cover this script's orchestration on top of them:
the server-side governed-bot skip (exit 0 with zero mutations), the declared
content-type cross-check, the add-then-verify board-membership path, the
Status-Backlog-only-when-unset default, curated-value preservation, the
conflict-withholding behavior, and the house exit-code mapping (issue #173
discipline: targeted lookups only, never a full board `item-list` scan).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# Locate the script relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "populate_project_item.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("populate_project_item", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. ppi.main).
ppi = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern so the script's own sibling imports resolve
# consistently when several governance test modules share one process.
sys.modules[_SPEC.name] = ppi
_SPEC.loader.exec_module(ppi)

# The shared access layer the script imports; referenced directly for its
# error types.
gha = ppi.github_api
# The shared mapping table the script imports; referenced for the canonical
# derivable-field order.
plm = ppi.project_label_mapping


# A fixed fake OWNER/REPO string reused across every test that needs one.
_REPO = "Jared-Godar/ecg_anomaly_detection"

# The fixed base arguments used by every issue-flavored main() test (repo
# supplied explicitly, so no test on this path spends a call on gh
# repository resolution).
_ISSUE_ARGS = ["--content-type", "issue", "--number", "240", "--repo", _REPO]


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


def _content(
    labels: list[str],
    *,
    login: str = "Jared-Godar",
    account_type: str = "User",
    is_pull_request: bool = False,
) -> subprocess.CompletedProcess:
    """Build a fake REST `gh api repos/.../issues/N` response.

    Args:
        labels: The item's label names.
        login: The author's login.
        account_type: The author's account type, e.g. "User" or "Bot".
        is_pull_request: Whether to include the `pull_request` marker key the
            REST issues endpoint uses to distinguish PRs from issues.

    Returns:
        A successful CompletedProcess carrying the content payload.
    """

    payload: dict[str, object] = {
        "number": 240,
        "user": {"login": login, "type": account_type},
        "labels": [{"name": name} for name in labels],
    }
    # The REST issues endpoint marks pull requests with a pull_request key;
    # tests set it to model a mis-declared invocation.
    if is_pull_request:
        payload["pull_request"] = {"url": "https://example.invalid"}
    return _completed(json.dumps(payload))


def _issue_items(nodes: list[dict[str, object]]) -> subprocess.CompletedProcess:
    """Build a fake targeted projectItems lookup response for one issue.

    Args:
        nodes: The issue's project-item memberships to report.

    Returns:
        A successful CompletedProcess carrying the GraphQL response envelope.
    """

    payload = {
        "data": {
            "repository": {
                "issue": {"projectItems": {"pageInfo": {"hasNextPage": False}, "nodes": nodes}}
            }
        }
    }
    return _completed(json.dumps(payload))


def _read_back(value: str | None) -> subprocess.CompletedProcess:
    """Build a fake targeted node(id:) read-back response for a single-select field.

    Args:
        value: The option name to report, or None for an unset field (GitHub
            returns JSON null for fieldValueByName in that case).

    Returns:
        A successful CompletedProcess carrying the GraphQL response envelope.
    """

    payload = {"name": value} if value is not None else None
    return _completed(json.dumps({"data": {"node": {"fieldValueByName": payload}}}))


def _fields_list() -> subprocess.CompletedProcess:
    """Build a fake `gh project field-list` response covering Status and derivable fields.

    Each field carries every option any test in this module asks the client
    to resolve, so one payload serves every scenario.

    Returns:
        A successful CompletedProcess carrying the field-list payload.
    """

    fields = [
        {
            "id": "PVTSSF_status",
            "name": "Status",
            "options": [{"id": "backlog", "name": "Backlog"}],
        },
        {
            "id": "PVTSSF_type",
            "name": "Issue Type",
            "options": [{"id": "gov", "name": "Governance"}, {"id": "bug", "name": "Bug"}],
        },
        {
            "id": "PVTSSF_priority",
            "name": "Priority",
            "options": [{"id": "med", "name": "Medium"}, {"id": "high", "name": "High"}],
        },
        {
            "id": "PVTSSF_risk",
            "name": "Risk",
            "options": [{"id": "low", "name": "Low"}, {"id": "hi", "name": "High"}],
        },
        {"id": "PVTSSF_size", "name": "Size", "options": [{"id": "s", "name": "S"}]},
        {
            "id": "PVTSSF_area",
            "name": "Repository Area",
            "options": [{"id": "cicd", "name": "ci-cd"}],
        },
        {
            "id": "PVTSSF_signal",
            "name": "Portfolio Signal",
            "options": [{"id": "opmat", "name": "Operational Maturity"}],
        },
    ]
    return _completed(json.dumps({"fields": fields}))


# The one membership node used by every main() test that models a tracked item.
_TRACKED_NODE: dict[str, object] = {
    "id": "PVTI_item",
    "project": {"id": "PVT_project", "number": 5, "owner": {"login": "Jared-Godar"}},
}


def _gh_arg_lists(mock_run) -> list[list[str]]:
    """Extract every gh argument list the mocked subprocess.run received.

    Args:
        mock_run: The patched subprocess.run mock.

    Returns:
        One argument list (including the leading "gh") per call, in order.
    """

    return [call.args[0] for call in mock_run.call_args_list]


# --- governed-bot skip and content-type cross-check -----------------------------------


def test_governed_bot_item_is_skipped_with_zero_mutations() -> None:
    """A Dependabot-authored item exits 0 without any board lookup or mutation.

    The Dependabot autofill path owns bot items' board metadata; this script
    must stand down cleanly so the workflow stays green on bot events.
    """

    responses = [
        _quota(4000),
        _content(["dependency: external"], login="dependabot[bot]", account_type="Bot"),
        _quota(4000),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 0
    # Exactly preflight, the REST content read, and the closing quota report:
    # no GraphQL lookup, no item-add, no item-edit.
    invoked = _gh_arg_lists(mock_run)
    assert len(invoked) == 3
    assert all("item-add" not in args and "item-edit" not in args for args in invoked)


def test_bot_login_without_bot_account_type_is_not_skipped() -> None:
    """A lookalike user login does not trigger the governed-bot skip.

    The account type is GitHub-assigned and cannot be self-selected, so it is
    the impersonation-proof half of the guard -- a User account named like
    the bot still gets normal population (here: tracked item, no labels, and
    a curated Status, so the run performs reads only).
    """

    responses = [
        _quota(4000),
        _content([], login="dependabot[bot]", account_type="User"),
        _issue_items([_TRACKED_NODE]),
        _read_back("Ready"),
        _quota(4000),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses):
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 0


def test_mis_declared_content_type_is_a_hard_stop() -> None:
    """Declaring an issue that is really a pull request exits 2 before any mutation."""

    responses = [
        _quota(4000),
        # The payload carries the pull_request marker, contradicting the
        # declared --content-type issue.
        _content([], is_pull_request=True),
        _quota(4000),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 2
    # The run stopped at the cross-check: no lookup, no add, no edit.
    invoked = _gh_arg_lists(mock_run)
    assert all("item-add" not in args and "item-edit" not in args for args in invoked)


# --- membership and population happy paths ---------------------------------------------


def test_new_issue_is_added_and_populated_from_labels() -> None:
    """An untracked issue is added (verified) and its derivable fields filled.

    Models issue #233's own acceptance scenario: a fresh issue with taxonomy
    labels ends with board membership, Status=Backlog, and every
    label-derivable field populated, each write verified by read-back.
    """

    responses = [
        _quota(4000),
        _content(["type: governance", "priority: p2"]),
        # Targeted lookup finds nothing; the script adds and re-looks-up.
        _issue_items([]),
        _completed("{}"),
        _issue_items([_TRACKED_NODE]),
        # Status: unset -> schema read -> edit -> verified Backlog.
        _read_back(None),
        _fields_list(),
        _completed(""),
        _read_back("Backlog"),
        # Issue Type: unset -> edit -> verified Governance (schema cached).
        _read_back(None),
        _completed(""),
        _read_back("Governance"),
        # Priority: unset -> edit -> verified Medium.
        _read_back(None),
        _completed(""),
        _read_back("Medium"),
        _quota(3985),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 0
    invoked = _gh_arg_lists(mock_run)
    # The add targeted the issue URL on the right board.
    add_call = next(args for args in invoked if "item-add" in args)
    assert f"https://github.com/{_REPO}/issues/240" in add_call
    # Quota discipline: the run never took a board-wide snapshot.
    assert all("item-list" not in args for args in invoked)


def test_curated_values_are_preserved_untouched() -> None:
    """Fields that already hold any value are read but never mutated.

    Curated values win (house rule from AGENTS.md): a tracked item whose
    Status and Priority are already set costs one read per field and zero
    item-edit mutations, which is also what makes converge-on-labeled reruns
    cheap and safe.
    """

    responses = [
        _quota(4000),
        _content(["priority: p1"]),
        _issue_items([_TRACKED_NODE]),
        # Status already curated to a lane the automation must not regress.
        _read_back("In Progress"),
        # Priority already curated; the p1 derivation is NOT applied.
        _read_back("High"),
        _quota(3998),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 0
    # Zero mutations happened anywhere in the run.
    assert all("item-edit" not in args for args in _gh_arg_lists(mock_run))


def test_conflicting_labels_withhold_the_field_but_fill_the_rest(capsys) -> None:
    """A label conflict is warned about while unambiguous fields still populate."""

    responses = [
        _quota(4000),
        # risk: low vs risk: security conflict; size: s derives cleanly.
        _content(["risk: low", "risk: security", "size: s"]),
        _issue_items([_TRACKED_NODE]),
        # Status: unset -> schema read -> edit -> verified.
        _read_back(None),
        _fields_list(),
        _completed(""),
        _read_back("Backlog"),
        # Size: unset -> edit -> verified (Risk is never touched).
        _read_back(None),
        _completed(""),
        _read_back("S"),
        _quota(3990),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 0
    # The conflict is surfaced for the maintainer in the run's own output.
    assert "conflicting label-derived options" in capsys.readouterr().err
    # No mutation ever targeted the Risk field's id.
    edits = [args for args in _gh_arg_lists(mock_run) if "item-edit" in args]
    assert all("PVTSSF_risk" not in args for args in edits)


def test_pull_request_flavor_uses_the_pr_lookup_and_url() -> None:
    """--content-type pull-request drives the PR-side lookup and the /pull/ add URL."""

    pr_lookup_empty = _completed(
        json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "projectItems": {"pageInfo": {"hasNextPage": False}, "nodes": []}
                        }
                    }
                }
            }
        )
    )
    pr_lookup_found = _completed(
        json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "projectItems": {
                                "pageInfo": {"hasNextPage": False},
                                "nodes": [_TRACKED_NODE],
                            }
                        }
                    }
                }
            }
        )
    )
    responses = [
        _quota(4000),
        # The REST payload carries the pull_request marker, matching the
        # declared kind; no labels, so only the Status default is planned.
        _content([], is_pull_request=True),
        pr_lookup_empty,
        _completed("{}"),
        pr_lookup_found,
        _read_back(None),
        _fields_list(),
        _completed(""),
        _read_back("Backlog"),
        _quota(3992),
    ]
    args = ["--content-type", "pull-request", "--number", "240", "--repo", _REPO]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = ppi.main(args)
    assert exit_code == 0
    add_call = next(a for a in _gh_arg_lists(mock_run) if "item-add" in a)
    assert f"https://github.com/{_REPO}/pull/240" in add_call


# --- exit-code mapping -----------------------------------------------------------------


def test_preflight_stop_maps_to_the_quota_exit_code() -> None:
    """A drained shared pool stops the run before any work, with exit code 3."""

    # Only the free preflight/report rate_limit reads may occur on this path.
    with patch.object(subprocess, "run", side_effect=[_quota(10), _quota(10)]) as mock_run:
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 3
    # Nothing beyond the two free rate_limit reads was attempted.
    assert len(_gh_arg_lists(mock_run)) == 2


def test_unconfirmed_read_back_maps_to_the_policy_exit_code() -> None:
    """A mutation whose read-back never confirms exits 1, not 2.

    The house convention separates an unverified write (policy, exit 1) from
    unreadable data (exit 2), so operators can tell a lost race from a
    broken token at a glance.
    """

    responses = [
        _quota(4000),
        _content([]),
        _issue_items([_TRACKED_NODE]),
        # Status unset; both mutation attempts read back still-unset.
        _read_back(None),
        _fields_list(),
        _completed(""),
        _read_back(None),
        _completed(""),
        _read_back(None),
        _quota(3990),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses):
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 1


def test_unverifiable_item_add_maps_to_the_data_exit_code() -> None:
    """An item-add whose verifying re-lookup finds nothing exits 2."""

    responses = [
        _quota(4000),
        _content([]),
        # Lookup, add, and the verifying re-lookup still empty.
        _issue_items([]),
        _completed("{}"),
        _issue_items([]),
        _quota(3996),
    ]
    # Drive main() against the scripted gh exchange; no real call leaves the process.
    with patch.object(subprocess, "run", side_effect=responses):
        exit_code = ppi.main(_ISSUE_ARGS)
    assert exit_code == 2
