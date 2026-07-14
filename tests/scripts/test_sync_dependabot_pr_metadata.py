"""Tests for the Dependabot Project #5 board-autofill script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test mocks the subprocess
boundary; none performs a live GitHub call.

The shared GitHub access layer (`scripts/github/github_api.py`) has its own
test module; the tests here cover this script's orchestration on top of it:
the server-side Dependabot-author guard (exit 1 with zero mutations), the
add-then-verify board-membership path, curated-value preservation, the
read-back-verified field mutations with their bounded retry, and the quota
exit-code mapping (issue #173 discipline: targeted lookups only, never a
full board `item-list` scan).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Locate the script relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "sync_dependabot_pr_metadata.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("sync_dependabot_pr_metadata", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. sdpm.main).
sdpm = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern so relative imports inside the script (if any)
# would resolve correctly.
sys.modules[_SPEC.name] = sdpm
_SPEC.loader.exec_module(sdpm)

# The shared access layer the script imports; referenced directly for its
# dataclasses and error types.
gha = sdpm.github_api


# A fixed fake OWNER/REPO string reused across every test that needs one.
_REPO = "Jared-Godar/ecg_anomaly_detection"

# The fixed base arguments every main() test passes (repo supplied explicitly,
# so no test on this path spends a call on gh repository resolution).
_ARGS = ["--pr-number", "117", "--repo", _REPO]


def _completed(stdout: str) -> subprocess.CompletedProcess:
    """Build a fake successful subprocess.CompletedProcess with the given stdout.

    Args:
        stdout: The text `gh` would have printed to stdout.

    Returns:
        A CompletedProcess with returncode 0 and empty stderr.
    """

    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


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


def _pr_author(login: str | None, account_type: str | None) -> subprocess.CompletedProcess:
    """Build a fake REST `gh api repos/.../pulls/N` response with the given author.

    Args:
        login: The author's login, or None to model a deleted account
            (GitHub serializes the PR's `user` as JSON null in that case).
        account_type: The author's account type, e.g. "Bot" or "User".

    Returns:
        A successful CompletedProcess carrying the pull-request payload.
    """

    user = {"login": login, "type": account_type} if login is not None else None
    return _completed(json.dumps({"user": user, "number": 117}))


def _fields_list() -> subprocess.CompletedProcess:
    """Build a fake `gh project field-list` response covering all nine synced fields.

    Field and option node ids are derived deterministically from each field's
    position in the script's own defaults table, so tests can pin an edit to
    a specific field by its synthetic id.

    Returns:
        A successful CompletedProcess carrying the field-list payload.
    """

    fields = []
    # One field entry per synced field, each holding exactly its default option.
    for index, (field_name, option_name) in enumerate(sdpm._BOT_PR_FIELD_DEFAULTS):
        fields.append(
            {
                "id": _field_id(index),
                "name": field_name,
                "options": [{"id": _option_id(index), "name": option_name}],
            }
        )
    return _completed(json.dumps({"fields": fields}))


def _field_id(index: int) -> str:
    """Return the synthetic field node id for the defaults-table entry at `index`.

    Args:
        index: The field's position in the script's defaults table.

    Returns:
        The deterministic fake field id used by `_fields_list`.
    """

    return f"PVTSSF_field_{index}"


def _option_id(index: int) -> str:
    """Return the synthetic option node id for the defaults-table entry at `index`.

    Args:
        index: The field's position in the script's defaults table.

    Returns:
        The deterministic fake option id used by `_fields_list`.
    """

    return f"opt_{index}"


def _field_sync_sequence(
    curated: dict[str, str] | None = None,
) -> list[subprocess.CompletedProcess]:
    """Build the per-field subprocess responses for one full nine-field sync pass.

    Mirrors the script's fixed call pattern: each field starts with a targeted
    current-value read; a curated (non-blank) field stops there, while a blank
    field is followed by the one-time field-list schema read (first blank field
    only, since the client caches it), the item-edit mutation, and the
    verifying read-back reporting the default as set.

    Args:
        curated: Field name -> existing curated value for fields that must be
            reported non-blank (and therefore preserved); every other field
            reads blank and gets the full mutate-and-verify exchange.

    Returns:
        The ordered fake responses for the whole nine-field pass.
    """

    curated = curated or {}
    responses: list[subprocess.CompletedProcess] = []
    # The field-list schema read happens exactly once, on the first blank field.
    schema_fetched = False
    # Walk the script's own defaults table so the fake sequence can never
    # drift from the real sync order.
    for field_name, option_name in sdpm._BOT_PR_FIELD_DEFAULTS:
        # A curated field costs exactly one read: the script preserves it.
        if field_name in curated:
            responses.append(_read_back(curated[field_name]))
            continue
        responses.append(_read_back(None))
        # Only the first blank field pays for the schema read; the client
        # caches the payload for every later option resolution.
        if not schema_fetched:
            responses.append(_fields_list())
            schema_fetched = True
        responses.append(_completed(""))
        responses.append(_read_back(option_name))
    return responses


# The one membership node used by every main() test that models a tracked PR.
_TRACKED_NODE: dict[str, object] = {
    "id": "PVTI_target",
    "project": {"id": "PVT_project", "number": 5, "owner": {"login": "Jared-Godar"}},
}

# The REST author payload for a genuine Dependabot pull request.
_DEPENDABOT_AUTHOR = _pr_author("dependabot[bot]", "Bot")


def _edit_calls(mock_run: MagicMock) -> list[list[str]]:
    """Extract the argument lists of every `gh project item-edit` invocation.

    Args:
        mock_run: The patched subprocess.run mock whose calls are inspected.

    Returns:
        The full argv list of each item-edit call, in invocation order.
    """

    return [call.args[0] for call in mock_run.call_args_list if "item-edit" in call.args[0]]


# --- resolve_repository ------------------------------------------------------------------


def test_resolve_repository_returns_explicit_slug_without_any_gh_call() -> None:
    """An explicit OWNER/REPO slug is returned as-is, with zero subprocess traffic."""

    # No side effects are provided: any gh invocation would raise StopIteration.
    with patch.object(subprocess, "run", side_effect=[]) as mock_run:
        assert sdpm.resolve_repository(_REPO) == _REPO
    assert mock_run.call_count == 0


def test_resolve_repository_asks_gh_when_slug_is_omitted() -> None:
    """A None slug is resolved via `gh repo view` from the surrounding checkout."""

    # gh reports the checkout's canonical nameWithOwner in JSON.
    with patch.object(
        subprocess, "run", side_effect=[_completed(json.dumps({"nameWithOwner": _REPO}))]
    ) as mock_run:
        assert sdpm.resolve_repository(None) == _REPO
    # The one call must be the JSON-format repo view, nothing else.
    assert mock_run.call_args_list[0].args[0] == ["gh", "repo", "view", "--json", "nameWithOwner"]


# --- main: field syncing ----------------------------------------------------------------


def test_main_sets_all_nine_fields_with_read_back_verification(
    capsys: pytest.CaptureFixture,
) -> None:
    """A tracked Dependabot PR with a fully blank board row gets all nine defaults.

    Every one of the nine documented bot-PR defaults must be written by its
    own item-edit, and every edit must be immediately verified by a fresh
    targeted GraphQL read-back -- the mutation's exit status is never proof.
    """

    # Full sequence: preflight, author guard, targeted lookup, nine blank
    # field syncs (read, one-time schema fetch, edit, read-back), report.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([_TRACKED_NODE]),
        *_field_sync_sequence(),
        _quota(4970),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 0
    edits = _edit_calls(mock_run)
    # One verified edit per field in the defaults table, no more, no fewer.
    assert len(edits) == len(sdpm._BOT_PR_FIELD_DEFAULTS)
    # Each edit must carry its own field/option pair, pinned by the synthetic
    # ids, so two fields can never silently swap their targets.
    for index, edit in enumerate(edits):
        assert edit[edit.index("--field-id") + 1] == _field_id(index)
        assert edit[edit.index("--single-select-option-id") + 1] == _option_id(index)
        assert edit[edit.index("--id") + 1] == "PVTI_target"
        assert edit[edit.index("--project-id") + 1] == "PVT_project"
    # Every mutation is followed immediately by a targeted GraphQL read-back.
    for position, call in enumerate(mock_run.call_args_list):
        # Only item-edit calls carry the follow-up read-back requirement.
        if "item-edit" in call.args[0]:
            assert mock_run.call_args_list[position + 1].args[0][:3] == ["gh", "api", "graphql"]
    captured = capsys.readouterr()
    assert "9 field(s) set, 0 preserved" in captured.out
    # The consumption report names all three accounting values (issue #173).
    assert "4990 before" in captured.err
    assert "4970 after" in captured.err
    assert "20 consumed" in captured.err


def test_main_preserves_curated_values_and_edits_only_blank_fields(
    capsys: pytest.CaptureFixture,
) -> None:
    """A field a human already set is preserved verbatim; only blank fields are edited.

    Curated values win (house rule from AGENTS.md): the bot defaults must
    never overwrite a maintainer's manual triage, so a non-blank Priority
    gets no item-edit at all -- not even one writing the same value back.
    """

    # Priority already holds a curated High; the other eight fields are blank.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([_TRACKED_NODE]),
        *_field_sync_sequence(curated={"Priority": "High"}),
        _quota(4973),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 0
    edits = _edit_calls(mock_run)
    # Eight edits for the eight blank fields; the curated one is untouched.
    assert len(edits) == len(sdpm._BOT_PR_FIELD_DEFAULTS) - 1
    # Priority sits at index 3 of the defaults table; no edit may target it.
    priority_index = [name for name, _ in sdpm._BOT_PR_FIELD_DEFAULTS].index("Priority")
    # Not one issued edit may carry the curated field's id anywhere in its argv.
    for edit in edits:
        assert _field_id(priority_index) not in edit
    captured = capsys.readouterr()
    assert "'Priority' preserved (already 'High')" in captured.out
    assert "8 field(s) set, 1 preserved" in captured.out


def test_main_never_performs_a_full_board_scan() -> None:
    """No code path issues `gh project item-list` (issue #173's central regression).

    Both the membership lookup and every verification read must be targeted;
    a full board snapshot inside this per-PR automation must never appear.
    """

    # A representative full run: tracked PR, all nine fields blank.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([_TRACKED_NODE]),
        *_field_sync_sequence(),
        _quota(4970),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        assert sdpm.main(_ARGS) == 0
    # End-to-end, not one invocation may be a board-wide item-list.
    for call in mock_run.call_args_list:
        assert "item-list" not in call.args[0]


# --- main: board membership -------------------------------------------------------------


def test_main_adds_an_untracked_pr_to_the_board_and_verifies_by_re_lookup(
    capsys: pytest.CaptureFixture,
) -> None:
    """An untracked PR is added via item-add, verified by a fresh targeted re-lookup.

    The add's own output is never trusted as proof of membership: the item id
    used for every later mutation must come from the re-lookup. All nine
    fields read curated here so the membership path stays isolated.
    """

    # Lookup finds nothing, item-add runs, the re-lookup finds the new item.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([]),
        _completed(json.dumps({"id": "PVTI_target"})),
        _pr_items([_TRACKED_NODE]),
        *_field_sync_sequence(curated=dict(sdpm._BOT_PR_FIELD_DEFAULTS)),
        _quota(4980),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 0
    # Exactly one item-add, with the PR's canonical URL and project identity.
    add_calls = [call.args[0] for call in mock_run.call_args_list if "item-add" in call.args[0]]
    assert len(add_calls) == 1
    assert f"https://github.com/{_REPO}/pull/117" in add_calls[0]
    assert add_calls[0][:4] == ["gh", "project", "item-add", "5"]
    # The call immediately after the add must be the verifying re-lookup.
    add_position = next(
        position
        for position, call in enumerate(mock_run.call_args_list)
        if "item-add" in call.args[0]
    )
    assert mock_run.call_args_list[add_position + 1].args[0][:3] == ["gh", "api", "graphql"]
    assert "0 field(s) set, 9 preserved" in capsys.readouterr().out


def test_main_exits_two_when_the_re_lookup_cannot_verify_the_add(
    capsys: pytest.CaptureFixture,
) -> None:
    """An item-add whose verifying re-lookup still finds nothing is a hard exit 2.

    Without a re-lookup-confirmed item id there is nothing safe to mutate, so
    the run must stop as "required data unreadable" rather than assume the add
    worked -- and no field edit may have been attempted.
    """

    # The add reports success, but both targeted lookups come back empty.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([]),
        _completed(json.dumps({"id": "PVTI_target"})),
        _pr_items([]),
        _quota(4986),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 2
    assert "re-lookup still cannot find" in capsys.readouterr().err
    # The unverifiable membership must have blocked every field mutation.
    assert _edit_calls(mock_run) == []


# --- main: the Dependabot author guard ---------------------------------------------------


@pytest.mark.parametrize(
    ("login", "account_type"),
    [
        # An ordinary human PR must never receive bot defaults.
        ("Jared-Godar", "User"),
        # A user account registered with a Dependabot-lookalike login is not a
        # bot: the account type is the server-side, non-self-assignable signal.
        ("dependabot[bot]", "User"),
        # A deleted author (GitHub serializes `user` as null) is not Dependabot.
        (None, None),
    ],
)
def test_main_refuses_non_dependabot_authors_without_mutating(
    login: str | None, account_type: str | None, capsys: pytest.CaptureFixture
) -> None:
    """A PR not authored by dependabot[bot] (type Bot) exits 1 with zero mutations.

    The server-side REST guard is what makes this script unusable as a
    general-purpose board-stamper: the bot defaults are only defensible for
    routine dependency PRs, so the refusal must precede the item-add and
    every field edit.
    """

    responses = [_quota(4990), _pr_author(login, account_type), _quota(4990)]
    # Preflight passes, the REST guard sees the non-Dependabot author, report runs.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 1
    assert "refusing to apply bot-PR board defaults" in capsys.readouterr().err
    # Zero mutations of any kind: no board add, no field edit.
    for call in mock_run.call_args_list:
        assert "item-add" not in call.args[0]
        assert "item-edit" not in call.args[0]


# --- main: read-back verification -------------------------------------------------------


def test_main_retries_an_unconfirmed_mutation_once_then_exits_one(
    capsys: pytest.CaptureFixture,
) -> None:
    """A field whose read-back never confirms the default fails after one bounded retry.

    Each attempted write earns its own fresh verification read; when both
    attempts read back blank, the run must fail as a policy error (exit 1)
    naming the observed value, never report a false success.
    """

    # Status reads blank, both edits exit cleanly, both read-backs stay blank.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([_TRACKED_NODE]),
        _read_back(None),
        _fields_list(),
        _completed(""),
        _read_back(None),
        _completed(""),
        _read_back(None),
        _quota(4983),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 1
    # Exactly two attempts: the original mutation plus its one bounded retry.
    assert len(_edit_calls(mock_run)) == 2
    captured = capsys.readouterr()
    assert "'Status' read back as 'unset' after 2 attempts" in captured.err
    # The failed run still prints its quota accountability line (issue #173).
    assert "consumed" in captured.err


def test_main_accepts_a_no_change_error_when_the_read_back_confirms() -> None:
    """gh's 'no changes to make' error succeeds only via an independent read-back.

    The CLI's no-change response is inconclusive on its own; the fresh
    targeted read observing the requested value is what proves the field
    already holds the default (e.g. a concurrent run landed it first).
    """

    no_change = subprocess.CalledProcessError(1, ["gh"], stderr="error: no changes to make")
    # Status reads blank, the edit reports no-change, the read-back confirms
    # Review anyway; the remaining eight fields read curated to end the run.
    responses = [
        _quota(4990),
        _DEPENDABOT_AUTHOR,
        _pr_items([_TRACKED_NODE]),
        _read_back(None),
        _fields_list(),
        no_change,
        _read_back("Review"),
        *[_read_back(option) for _, option in sdpm._BOT_PR_FIELD_DEFAULTS[1:]],
        _quota(4984),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses):
        assert sdpm.main(_ARGS) == 0


# --- main: quota discipline --------------------------------------------------------------


def test_main_stops_with_exit_three_before_any_work_when_quota_is_low(
    capsys: pytest.CaptureFixture,
) -> None:
    """A preflight below the threshold exits 3 without a single GraphQL call.

    The stop must happen before the author guard, the lookup, the item-add,
    and every field edit, and its exit code must differ from both failure
    codes so CI output can never conflate shared-pool exhaustion with a
    defect in the pull request being stamped.
    """

    # Both rate_limit reads (preflight and report) see a nearly drained pool;
    # the script's default threshold of 50 must reject 12 remaining points.
    with patch.object(subprocess, "run", side_effect=[_quota(12), _quota(12)]) as mock_run:
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 3
    assert "only 12 of 5000" in capsys.readouterr().err
    # The only gh traffic allowed is the free REST rate_limit accounting --
    # no author fetch, no GraphQL lookup, and above all no mutation.
    for call in mock_run.call_args_list:
        assert "rate_limit" in call.args[0]
        assert "item-add" not in call.args[0]
        assert "item-edit" not in call.args[0]


def test_main_maps_a_primary_rate_limit_during_mutation_to_exit_three(
    capsys: pytest.CaptureFixture,
) -> None:
    """Primary rate-limit exhaustion mid-run exits 3, distinct from both failure codes.

    A pool that drains between the preflight and a field mutation is
    infrastructure, not a policy defect or unreadable data; the "quota:"
    prefix and exit code 3 keep the three outcomes unmistakably apart.
    """

    exhausted = subprocess.CalledProcessError(
        1, ["gh"], stderr="GraphQL: API rate limit already exceeded for user ID 16855088."
    )
    # Preflight passes, the run reaches the first field edit, and that edit
    # hits the exhausted pool; the report still runs afterwards.
    responses = [
        _quota(60),
        _DEPENDABOT_AUTHOR,
        _pr_items([_TRACKED_NODE]),
        _read_back(None),
        _fields_list(),
        exhausted,
        _quota(0),
    ]
    # The whole run happens behind the mocked subprocess boundary.
    with patch.object(subprocess, "run", side_effect=responses):
        exit_code = sdpm.main(_ARGS)
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err
