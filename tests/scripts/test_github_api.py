"""Tests for the shared GitHub API access layer used by the governance scripts.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test mocks the subprocess
boundary; none performs a live GitHub call.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Locate the module relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "github" / "github_api.py"
# Load the module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("github_api", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. gha.run_gh).
gha = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, both to match
# the standard importlib.util pattern and so the governance scripts' own
# `import github_api` statements resolve to this same module object when the
# script test files load them into the same pytest process.
sys.modules[_SPEC.name] = gha
_SPEC.loader.exec_module(gha)


def _completed(stdout: str) -> subprocess.CompletedProcess:
    """Build a fake successful subprocess.CompletedProcess with the given stdout.

    Args:
        stdout: The text `gh` would have printed to stdout.

    Returns:
        A CompletedProcess with returncode 0 and empty stderr.
    """

    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _quota_payload(remaining: int, *, limit: int = 5000, reset: int = 1770000000) -> str:
    """Build a fake `gh api rate_limit` JSON payload for the GraphQL resource.

    Args:
        remaining: The GraphQL points remaining to report.
        limit: The GraphQL hourly point budget to report.
        reset: The epoch second at which the window resets.

    Returns:
        JSON text matching GitHub's rate_limit endpoint shape.
    """

    return json.dumps(
        {
            "resources": {
                "graphql": {
                    "limit": limit,
                    "used": limit - remaining,
                    "remaining": remaining,
                    "reset": reset,
                }
            }
        }
    )


# --- run_gh rate-limit classification and retry --------------------------------------


def test_run_gh_returns_stdout_on_success() -> None:
    """A clean gh invocation returns its captured stdout unchanged."""

    # The simplest possible success: one attempt, no retries, stdout through.
    with patch.object(subprocess, "run", return_value=_completed('{"ok": true}')) as mock_run:
        assert gha.run_gh(["pr", "view", "155"]) == '{"ok": true}'
    assert mock_run.call_count == 1


def test_run_gh_fails_fast_on_primary_rate_limit_without_retrying() -> None:
    """A primary (hours-long) rate limit raises PrimaryRateLimitError immediately.

    Retrying inside a single CI job's lifetime cannot help this failure mode
    (it takes up to an hour to clear), so a retry would only waste time and
    delay the same unavoidable failure. The distinct exception type is what
    lets callers exit with the quota-specific exit code.
    """

    # GitHub's real wording for this case, observed live against PR #155.
    error = subprocess.CalledProcessError(
        1, ["gh"], stderr="GraphQL: API rate limit already exceeded for user ID 16855088."
    )
    # subprocess.run always raises this same error, so a retry would just hit
    # it again -- the assertions below confirm run_gh doesn't bother trying.
    with (
        patch.object(subprocess, "run", side_effect=error) as mock_run,
        patch.object(gha.time, "sleep") as mock_sleep,
        pytest.raises(gha.PrimaryRateLimitError, match="rate limit exhausted"),
    ):
        gha.run_gh(["pr", "view", "155"])
    # Exactly one attempt: no retry loop should have run for this error class.
    assert mock_run.call_count == 1
    mock_sleep.assert_not_called()


def test_run_gh_retries_then_succeeds_on_secondary_rate_limit() -> None:
    """A secondary (short-lived, abuse-detection) rate limit is retried and can recover."""

    secondary_error = subprocess.CalledProcessError(
        1, ["gh"], stderr="You have exceeded a secondary rate limit. Please wait."
    )
    # Fails twice, then succeeds on the third attempt.
    with (
        patch.object(
            subprocess, "run", side_effect=[secondary_error, secondary_error, _completed("ok")]
        ) as mock_run,
        patch.object(gha.time, "sleep") as mock_sleep,
    ):
        result = gha.run_gh(["pr", "view", "155"])
    assert result == "ok"
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
    # completely before this raises -- and as the base error class, not the
    # primary-rate-limit one, since the secondary limit is a different condition.
    with (
        patch.object(subprocess, "run", side_effect=secondary_error) as mock_run,
        patch.object(gha.time, "sleep"),
        pytest.raises(gha.GitHubApiError, match="secondary rate limit"),
    ):
        gha.run_gh(["pr", "view", "155"])
    # One initial attempt plus one retry per entry in the backoff schedule.
    assert mock_run.call_count == 1 + len(gha.SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS)


def test_run_gh_translates_a_missing_gh_binary_into_the_module_error() -> None:
    """A missing gh executable raises GitHubApiError, not a bare FileNotFoundError."""

    # subprocess.run raises FileNotFoundError when the binary isn't on PATH.
    with (
        patch.object(subprocess, "run", side_effect=FileNotFoundError()),
        pytest.raises(gha.GitHubApiError, match="not installed"),
    ):
        gha.run_gh(["pr", "view", "155"])


def test_run_gh_propagates_an_ordinary_failure_without_retry() -> None:
    """A non-rate-limit gh failure (e.g. auth) raises immediately as the base error class."""

    auth_error = subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope")
    # An ordinary failure is neither quota condition, so it must not be
    # retried and must not arrive as a rate-limit exception type.
    with (
        patch.object(subprocess, "run", side_effect=auth_error) as mock_run,
        pytest.raises(gha.GitHubApiError, match="insufficient scope"),
    ):
        gha.run_gh(["pr", "view", "155"])
    assert mock_run.call_count == 1


# --- fetch_graphql_quota / QuotaMonitor ------------------------------------------------


def test_fetch_graphql_quota_parses_the_graphql_resource() -> None:
    """fetch_graphql_quota extracts the GraphQL resource fields from the REST payload."""

    # The rate_limit endpoint reports several resources; only graphql matters here.
    with patch.object(subprocess, "run", return_value=_completed(_quota_payload(4988))):
        quota = gha.fetch_graphql_quota()
    assert quota.limit == 5000
    assert quota.remaining == 4988
    assert quota.used == 12
    assert quota.reset_epoch == 1770000000


def test_quota_monitor_preflight_passes_above_threshold() -> None:
    """A preflight with remaining quota above the minimum records a baseline and proceeds."""

    monitor = gha.QuotaMonitor(minimum_remaining=50)
    # 4988 remaining is comfortably above the 50-point threshold.
    with patch.object(subprocess, "run", return_value=_completed(_quota_payload(4988))):
        before = monitor.preflight()
    assert before.remaining == 4988
    assert monitor.preflighted is True


def test_quota_monitor_preflight_stops_below_threshold() -> None:
    """A preflight below the minimum raises the insufficient-quota error, naming the numbers.

    This is the stop-before-mutation safeguard: the error message must carry
    the observed remaining value, the configured threshold, and the reset time
    so an operator can decide when to rerun.
    """

    monitor = gha.QuotaMonitor(minimum_remaining=50)
    # Only 12 points remain, below the 50-point threshold.
    with (
        patch.object(subprocess, "run", return_value=_completed(_quota_payload(12))),
        pytest.raises(gha.GraphQLQuotaInsufficientError, match="only 12 of 5000"),
    ):
        monitor.preflight()
    # The baseline is still recorded so a consumption report remains possible.
    assert monitor.preflighted is True


def test_quota_monitor_zero_threshold_observes_without_blocking() -> None:
    """A non-positive threshold disables the stop but still records the baseline."""

    monitor = gha.QuotaMonitor(minimum_remaining=0)
    # Zero points remain, yet the disabled threshold must not block the run.
    with patch.object(subprocess, "run", return_value=_completed(_quota_payload(0))):
        before = monitor.preflight()
    assert before.remaining == 0


def test_quota_monitor_report_computes_consumption() -> None:
    """report() subtracts the after snapshot from the before snapshot into one line."""

    monitor = gha.QuotaMonitor(minimum_remaining=50)
    # 4988 before, 4976 after: 12 points consumed by the run in between.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_completed(_quota_payload(4988)), _completed(_quota_payload(4976))],
    ):
        monitor.preflight()
        line = monitor.report()
    assert "4988 before" in line
    assert "4976 after" in line
    assert "12 consumed" in line


def test_quota_monitor_report_annotates_a_window_reset() -> None:
    """A remaining value that rose between snapshots is annotated as a window reset."""

    monitor = gha.QuotaMonitor(minimum_remaining=50)
    # remaining rises from 100 to 5000: the hourly window reset mid-run.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_completed(_quota_payload(100)), _completed(_quota_payload(5000))],
    ):
        monitor.preflight()
        line = monitor.report()
    assert "reset during the run" in line


def test_quota_monitor_report_before_preflight_is_a_sequencing_error() -> None:
    """report() without a recorded baseline raises rather than inventing numbers."""

    monitor = gha.QuotaMonitor(minimum_remaining=50)
    # No preflight has run, so there is no baseline to subtract from.
    with pytest.raises(gha.GitHubApiError, match="before preflight"):
        monitor.report()


# --- ProjectClient schema caching -----------------------------------------------------


def _field_list_payload() -> str:
    """Build a fake `gh project field-list` JSON payload with two single-select fields.

    Returns:
        JSON text matching gh's field-list output shape.
    """

    return json.dumps(
        {
            "fields": [
                {
                    "id": "PVTSSF_status",
                    "name": "Status",
                    "options": [
                        {"id": "backlog", "name": "Backlog"},
                        {"id": "merged", "name": "Merged"},
                    ],
                },
                {
                    "id": "PVTSSF_risk",
                    "name": "Risk",
                    "options": [{"id": "low", "name": "Low"}],
                },
            ]
        }
    )


def test_single_select_option_resolves_field_and_option_ids() -> None:
    """single_select_option returns the requested field's id and named option's id."""

    client = gha.ProjectClient("Jared-Godar", 5)
    # One field-list read backs the lookup.
    with patch.object(subprocess, "run", return_value=_completed(_field_list_payload())):
        resolved = client.single_select_option("Status", "Merged")
    assert resolved.field_id == "PVTSSF_status"
    assert resolved.option_id == "merged"


def test_single_select_option_caches_the_field_list_across_lookups() -> None:
    """Schema and option lookups are cached: two different lookups cost one field-list read.

    This is the schema-caching regression required by issue #173 -- resolving
    a second field/option pair on the same client must not re-fetch the schema.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    # Both lookups (different fields entirely) share the one cached payload.
    with patch.object(
        subprocess, "run", return_value=_completed(_field_list_payload())
    ) as mock_run:
        first = client.single_select_option("Status", "Merged")
        second = client.single_select_option("Risk", "Low")
        repeated = client.single_select_option("Status", "Merged")
    assert mock_run.call_count == 1
    assert first.option_id == "merged"
    assert second.option_id == "low"
    assert repeated is first


def test_single_select_option_raises_when_field_missing() -> None:
    """A project without the requested field raises a specific, actionable error."""

    client = gha.ProjectClient("Jared-Godar", 5)
    # The fixture schema has Status and Risk, but no field named "Nonexistent".
    with (
        patch.object(subprocess, "run", return_value=_completed(_field_list_payload())),
        pytest.raises(gha.GitHubApiError, match="no 'Nonexistent' field"),
    ):
        client.single_select_option("Nonexistent", "Merged")


def test_single_select_option_raises_when_option_missing() -> None:
    """A field that exists but lacks the requested option raises a specific error."""

    client = gha.ProjectClient("Jared-Godar", 5)
    # "Status" exists but has no option named "Nonexistent".
    with (
        patch.object(subprocess, "run", return_value=_completed(_field_list_payload())),
        pytest.raises(gha.GitHubApiError, match="no 'Nonexistent' option"),
    ):
        client.single_select_option("Status", "Nonexistent")


def test_items_fetches_the_snapshot_once_and_reuses_it() -> None:
    """The full board snapshot is fetched at most once per client instance.

    This is the one-snapshot-per-phase regression: repeated items() calls in
    the same logical phase must serve the cached list, never re-paginate the board.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    payload = json.dumps({"items": [{"id": "PVTI_a"}, {"id": "PVTI_b"}]})
    # Three lookups, one underlying item-list read.
    with patch.object(subprocess, "run", return_value=_completed(payload)) as mock_run:
        first = client.items()
        second = client.items()
        third = client.items()
    assert mock_run.call_count == 1
    assert first is second is third
    assert [item["id"] for item in first] == ["PVTI_a", "PVTI_b"]


# --- ProjectClient targeted pull-request item lookup -----------------------------------


def _pr_items_payload(nodes: list[dict[str, object]], *, has_next_page: bool = False) -> str:
    """Build a fake targeted projectItems GraphQL response for one pull request.

    Args:
        nodes: The PR's project-item memberships, each shaped like the query's nodes.
        has_next_page: Whether the response claims more memberships exist.

    Returns:
        JSON text matching `gh api graphql`'s response envelope.
    """

    return json.dumps(
        {
            "data": {
                "repository": {
                    "pullRequest": {
                        "projectItems": {
                            "pageInfo": {"hasNextPage": has_next_page},
                            "nodes": nodes,
                        }
                    }
                }
            }
        }
    )


def test_fetch_pull_request_item_matches_project_number_and_owner() -> None:
    """The PR's membership list is matched on both project number and owner login.

    Two decoy memberships are included -- same owner but a different project
    number, and the same number but a different owner -- to confirm both
    criteria must agree.
    """

    nodes = [
        {
            "id": "wrong-number",
            "project": {"id": "PVT_x", "number": 7, "owner": {"login": "Jared-Godar"}},
        },
        {
            "id": "wrong-owner",
            "project": {"id": "PVT_y", "number": 5, "owner": {"login": "someone-else"}},
        },
        {
            "id": "PVTI_target",
            "project": {"id": "PVT_project", "number": 5, "owner": {"login": "Jared-Godar"}},
        },
    ]
    client = gha.ProjectClient("Jared-Godar", 5)
    # Only the third membership matches both the number and the owner.
    with patch.object(
        subprocess, "run", return_value=_completed(_pr_items_payload(nodes))
    ) as mock_run:
        item = client.fetch_pull_request_item("Jared-Godar/ecg_anomaly_detection", 155)
    assert item is not None
    assert item.item_id == "PVTI_target"
    assert item.project_id == "PVT_project"
    # The lookup must be the targeted GraphQL query, never a board-wide item-list.
    invoked = mock_run.call_args_list[0].args[0]
    assert invoked[:3] == ["gh", "api", "graphql"]
    assert "item-list" not in invoked


def test_fetch_pull_request_item_returns_none_when_not_tracked() -> None:
    """A PR with no membership on this project returns None, not an error."""

    client = gha.ProjectClient("Jared-Godar", 5)
    # An empty membership list means the PR was never added to the project.
    with patch.object(subprocess, "run", return_value=_completed(_pr_items_payload([]))):
        assert client.fetch_pull_request_item("Jared-Godar/ecg_anomaly_detection", 155) is None


def test_fetch_pull_request_item_fails_loudly_on_a_truncated_page() -> None:
    """hasNextPage on the membership list is an explicit failure, not a silent 'not tracked'.

    A truncated page makes absence unprovable -- the sought project could be on
    the next page -- so returning None here would silently skip a tracked PR.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    # No matching node on this page, but the response says more pages exist.
    with (
        patch.object(
            subprocess,
            "run",
            return_value=_completed(_pr_items_payload([], has_next_page=True)),
        ),
        pytest.raises(gha.GitHubApiError, match="inconclusive"),
    ):
        client.fetch_pull_request_item("Jared-Godar/ecg_anomaly_detection", 155)


def test_fetch_pull_request_item_rejects_a_malformed_repo_slug() -> None:
    """A repo argument without the OWNER/REPO shape fails before any gh call."""

    client = gha.ProjectClient("Jared-Godar", 5)
    # No subprocess mock: the malformed slug must be rejected without any call.
    with (
        patch.object(subprocess, "run") as mock_run,
        pytest.raises(gha.GitHubApiError, match="malformed repository slug"),
    ):
        client.fetch_pull_request_item("not-a-slug", 155)
    mock_run.assert_not_called()


# --- ProjectClient targeted single-select read-back ------------------------------------


def _read_back_payload(node: object) -> str:
    """Build a fake targeted node(id:) GraphQL response.

    Args:
        node: The node object to embed (None models a vanished item).

    Returns:
        JSON text matching `gh api graphql`'s response envelope.
    """

    return json.dumps({"data": {"node": node}})


def test_fetch_item_single_select_returns_the_current_value() -> None:
    """A set single-select value comes back as its option name."""

    client = gha.ProjectClient("Jared-Godar", 5)
    payload = _read_back_payload({"fieldValueByName": {"name": "Merged"}})
    # One targeted query reads exactly this item's Status.
    with patch.object(subprocess, "run", return_value=_completed(payload)) as mock_run:
        assert client.fetch_item_single_select("PVTI_target", "Status") == "Merged"
    # The read-back must be the targeted GraphQL query, never a board-wide item-list.
    invoked = mock_run.call_args_list[0].args[0]
    assert invoked[:3] == ["gh", "api", "graphql"]
    assert "item-list" not in invoked


def test_fetch_item_single_select_returns_none_for_an_unset_field() -> None:
    """GitHub omits unset fields (fieldValueByName is JSON null); that reads as None.

    This is the omitted-unset-field regression required by issue #173: a
    freshly added item legitimately has no Status value at all, and that must
    be reported as unset, not as an error and not as a phantom value.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    payload = _read_back_payload({"fieldValueByName": None})
    # The item exists; the field simply has no value on it.
    with patch.object(subprocess, "run", return_value=_completed(payload)):
        assert client.fetch_item_single_select("PVTI_target", "Status") is None


def test_fetch_item_single_select_raises_when_the_item_vanishes() -> None:
    """A null node is an explicit failure, not an unset value.

    Losing the item between mutation and verification makes the requested state
    unknowable, so this must raise rather than report a phantom 'unset'.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    # node null means the id no longer resolves to anything.
    with (
        patch.object(subprocess, "run", return_value=_completed(_read_back_payload(None))),
        pytest.raises(gha.GitHubApiError, match="no longer exists"),
    ):
        client.fetch_item_single_select("PVTI_target", "Status")


def test_fetch_item_single_select_raises_when_the_node_is_not_a_project_item() -> None:
    """A node of another GraphQL type (fragment mismatch) is an explicit failure."""

    client = gha.ProjectClient("Jared-Godar", 5)
    # An empty node object means the ProjectV2Item fragment matched nothing.
    with (
        patch.object(subprocess, "run", return_value=_completed(_read_back_payload({}))),
        pytest.raises(gha.GitHubApiError, match="not a Project V2 item"),
    ):
        client.fetch_item_single_select("PVTI_target", "Status")


def test_fetch_item_single_select_raises_on_a_non_single_select_union_type() -> None:
    """A set value of a different union type is an explicit failure, never 'unset'.

    Project field values span multiple GraphQL union types (issue #173's Risks
    section); a text/number/date value matches none of the query's fragments
    and arrives as an empty object, which must be surfaced, not misread.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    payload = _read_back_payload({"fieldValueByName": {}})
    # The field has a value, but not a single-select one.
    with (
        patch.object(subprocess, "run", return_value=_completed(payload)),
        pytest.raises(gha.GitHubApiError, match="not a single-select"),
    ):
        client.fetch_item_single_select("PVTI_target", "Status")


def test_fetch_item_single_select_is_never_cached() -> None:
    """Every read-back performs a fresh query -- the verification read must never be stale.

    Caching here would silently defeat the read-back-verified mutation rule
    (#164/#170): the second call must observe a value that changed between calls.
    """

    client = gha.ProjectClient("Jared-Godar", 5)
    first = _completed(_read_back_payload({"fieldValueByName": {"name": "Closed"}}))
    second = _completed(_read_back_payload({"fieldValueByName": {"name": "Merged"}}))
    # The underlying value changes between the two calls; both must be observed.
    with patch.object(subprocess, "run", side_effect=[first, second]) as mock_run:
        assert client.fetch_item_single_select("PVTI_target", "Status") == "Closed"
        assert client.fetch_item_single_select("PVTI_target", "Status") == "Merged"
    assert mock_run.call_count == 2
