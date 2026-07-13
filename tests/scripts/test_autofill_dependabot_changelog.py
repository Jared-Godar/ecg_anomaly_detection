"""Tests for the Dependabot changelog-autofill script (issue #193).

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. All GitHub access is exercised through
mocked subprocess.run calls -- no network is ever touched.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "autofill_dependabot_changelog.py"
)
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("autofill_dependabot_changelog", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. adc.main).
adc = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves postponed (`from __future__ import annotations`) type hints via
# sys.modules[cls.__module__], so the module must be registered there before exec_module
# runs a dynamically loaded file's @dataclass decorators.
sys.modules[_SPEC.name] = adc
_SPEC.loader.exec_module(adc)

# The shared access layer the script imports; referenced directly for its error types.
_gha = adc.github_api

# The repository slug every test run claims to operate on.
_REPO = "Jared-Godar/ecg_anomaly_detection"
# The pull request number under test; its "(#192)" token keys every bullet.
_PR_NUMBER = 192
# The Dependabot-shaped head branch the workflow passes through --head-ref.
_HEAD_REF = "dependabot/uv/ruff-1.1.0"
# The head sha the workflow run was queued for, passed through --head-sha.
_HEAD_SHA = "abc123def4567890"
# The full argv every main() invocation uses unless a test overrides a value.
_ARGS = [
    "--pr-number",
    str(_PR_NUMBER),
    "--repo",
    _REPO,
    "--head-ref",
    _HEAD_REF,
    "--head-sha",
    _HEAD_SHA,
]

# The bullet the default single-dependency payload must render for PR #192.
_DESIRED_DEFAULT = "- Bump `ruff` from 1.0.0 to 1.1.0 (uv) via Dependabot (#192)."

# A changelog whose Unreleased section already carries an (empty) Dependencies
# subsection -- the ordinary steady-state shape of this repository's file.
_CHANGELOG_WITH_HEADING = """# Changelog

## Unreleased

### Added

### Dependencies

### Fixed

## 1.0.0 - 2026-01-01

### Added

- Old release entry.
"""

# The exact full text the happy-path PUT must carry: the default bullet
# inserted into the empty Dependencies subsection, blank-line separated.
_EXPECTED_HAPPY = """# Changelog

## Unreleased

### Added

### Dependencies

- Bump `ruff` from 1.0.0 to 1.1.0 (uv) via Dependabot (#192).

### Fixed

## 1.0.0 - 2026-01-01

### Added

- Old release entry.
"""

# A changelog whose Unreleased section has no Dependencies heading at all
# (e.g. after a release freeze reset removed the empty subsection).
_CHANGELOG_WITHOUT_HEADING = """# Changelog

## Unreleased

### Added

- Something else (#100).

## 1.0.0 - 2026-01-01
"""

# The heading-creation path's exact expected output: the subsection appended
# at the end of the Unreleased section, before the next '## ' heading, with a
# blank line before and after.
_EXPECTED_HEADING_CREATED = """# Changelog

## Unreleased

### Added

- Something else (#100).

### Dependencies

- Bump `ruff` from 1.0.0 to 1.1.0 (uv) via Dependabot (#192).

## 1.0.0 - 2026-01-01
"""

# A changelog already carrying a stale (#192)-keyed bullet (an earlier version
# range) next to another PR's bullet that must survive untouched.
_CHANGELOG_WITH_STALE_BULLET = """# Changelog

## Unreleased

### Dependencies

- Bump `other` from 1 to 2 (uv) via Dependabot (#188).
- Bump `ruff` from 0.9.0 to 1.0.0 (uv) via Dependabot (#192).

## 1.0.0 - 2026-01-01
"""

# The replace-in-place path's exact expected output: the stale (#192) bullet
# replaced at its original position, the (#188) neighbor untouched.
_EXPECTED_REPLACED = """# Changelog

## Unreleased

### Dependencies

- Bump `other` from 1 to 2 (uv) via Dependabot (#188).
- Bump `ruff` from 1.0.0 to 1.1.0 (uv) via Dependabot (#192).

## 1.0.0 - 2026-01-01
"""

# A changelog whose (#192) bullet already equals the desired rendering exactly
# -- the idempotent no-op case that terminates the self-trigger loop.
_CHANGELOG_ALREADY_CURRENT = """# Changelog

## Unreleased

### Dependencies

- Bump `ruff` from 1.0.0 to 1.1.0 (uv) via Dependabot (#192).

## 1.0.0 - 2026-01-01
"""


def _completed(payload: object) -> subprocess.CompletedProcess:
    """Build a fake successful subprocess.CompletedProcess whose stdout is JSON.

    Args:
        payload: The object `gh` would have printed as JSON.

    Returns:
        A CompletedProcess with returncode 0 and empty stderr.
    """

    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")


def _dep(
    name: str = "ruff",
    prev: str = "1.0.0",
    new: str = "1.1.0",
    ecosystem: str = "uv",
) -> dict[str, str]:
    """Build one fetch-metadata updated-dependencies element.

    Args:
        name: The dependencyName field.
        prev: The prevVersion field (empty string models a brand-new dependency).
        new: The newVersion field.
        ecosystem: The packageEcosystem field.

    Returns:
        A dict shaped like one element of fetch-metadata's
        updated-dependencies-json output.
    """

    return {
        "dependencyName": name,
        "prevVersion": prev,
        "newVersion": new,
        "packageEcosystem": ecosystem,
    }


def _pr_payload(
    login: str = "dependabot[bot]",
    user_type: str = "Bot",
    state: str = "open",
    head_ref: str = _HEAD_REF,
    head_sha: str = _HEAD_SHA,
    head_full_name: str = _REPO,
) -> dict[str, object]:
    """Build a live REST pull-request payload that passes every policy check by default.

    Args:
        login: The PR author's login.
        user_type: The PR author's server-attested account type.
        state: The PR's open/closed state.
        head_ref: The head branch name.
        head_sha: The live head commit sha.
        head_full_name: The head repository's OWNER/REPO slug.

    Returns:
        A dict shaped like `gh api repos/.../pulls/N` output.
    """

    return {
        "number": _PR_NUMBER,
        "user": {"login": login, "type": user_type},
        "state": state,
        "head": {
            "ref": head_ref,
            "sha": head_sha,
            "repo": {"full_name": head_full_name},
        },
    }


def _blob(text: str, sha: str = "blob-sha-1") -> subprocess.CompletedProcess:
    """Build a fake REST contents-API response for CHANGELOG.md.

    Args:
        text: The file text to base64-encode into the payload.
        sha: The Git blob sha the PUT must precondition on.

    Returns:
        A successful CompletedProcess carrying the contents payload.
    """

    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return _completed({"content": encoded, "encoding": "base64", "sha": sha})


def _commit(
    sha: str = "c0ffee1234567890",
    login: str = "dependabot[bot]",
    verified: bool = True,
) -> dict[str, object]:
    """Build one commit object as `gh api repos/.../pulls/N/commits` returns it.

    Args:
        sha: The commit sha.
        login: The commit author's GitHub login (None models an unlinked author).
        verified: GitHub's server-side signature-verification verdict.

    Returns:
        A dict carrying the sha, author, and verification fields the
        authorship proof reads.
    """

    return {
        "sha": sha,
        "author": {"login": login},
        "commit": {"verification": {"verified": verified}},
    }


def _commits(*commits: dict[str, object]) -> subprocess.CompletedProcess:
    """Build a fake `gh api ... --paginate --slurp` commits response of one page.

    Args:
        commits: The commit objects to include on that single page.

    Returns:
        A successful CompletedProcess whose stdout matches --slurp's shape:
        an array of pages, each page an array of commit objects.
    """

    return _completed([list(commits)])


# gh's stderr wording for a contents-PUT optimistic-locking conflict, observed
# when the file's blob sha moved between the GET and the PUT.
_CONFLICT_ERROR = subprocess.CalledProcessError(
    1, ["gh"], stderr="gh: CHANGELOG.md does not match blob-sha-1 (HTTP 409)"
)


def _put_calls(mock_run: object) -> list:
    """Return every mocked subprocess call that was a contents PUT.

    Args:
        mock_run: The patched subprocess.run mock.

    Returns:
        The calls whose gh argv included the "PUT" method flag value.
    """

    return [call for call in mock_run.call_args_list if "PUT" in call.args[0]]  # type: ignore[attr-defined]


def _decoded_put_content(call: object) -> str:
    """Decode the CHANGELOG.md text a mocked contents PUT carried.

    Args:
        call: One mocked subprocess.run call (from _put_calls).

    Returns:
        The base64-decoded file text from the call's content= field.
    """

    argv = call.args[0]  # type: ignore[attr-defined]
    # The content= field is one element of the fixed gh argv; exactly one exists per PUT.
    content_fields = [arg for arg in argv if isinstance(arg, str) and arg.startswith("content=")]
    assert len(content_fields) == 1
    return base64.b64decode(content_fields[0].removeprefix("content=")).decode("utf-8")


@pytest.fixture(autouse=True)
def _default_dependencies_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a valid single-dependency UPDATED_DEPENDENCIES_JSON to every test.

    Individual tests override or delete it (via their own monkeypatch calls)
    to exercise the missing/hostile-input paths.

    Args:
        monkeypatch: pytest's environment-patching fixture.
    """

    monkeypatch.setenv(adc.UPDATED_DEPENDENCIES_ENV, json.dumps([_dep()]))


# --- rendering and metadata validation ----------------------------------------------


def test_render_bullets_omits_the_from_clause_when_prev_version_is_empty() -> None:
    """An empty prevVersion renders the shorter 'Bump `x` to N' form, never 'from  to'."""

    deps = adc.parse_dependencies([_dep(prev="")])
    bullets = adc.render_bullets(deps, _PR_NUMBER)
    assert bullets == ("- Bump `ruff` to 1.1.0 (uv) via Dependabot (#192).",)


def test_parse_dependencies_accepts_a_pre_commit_url_shaped_name() -> None:
    """pre-commit ecosystem dependency names are repository URLs and must validate.

    The colon and slashes are required members of the name allowlist precisely
    for this case.
    """

    element = _dep(
        name="https://github.com/astral-sh/ruff-pre-commit",
        prev="v0.1.0",
        new="v0.2.0",
        ecosystem="pre-commit",
    )
    deps = adc.parse_dependencies([element])
    assert deps[0].name == "https://github.com/astral-sh/ruff-pre-commit"
    # The URL survives into the rendered bullet verbatim, inside the code span.
    bullets = adc.render_bullets(deps, _PR_NUMBER)
    assert "`https://github.com/astral-sh/ruff-pre-commit`" in bullets[0]


def test_parse_dependencies_rejects_an_empty_array() -> None:
    """A Dependabot PR that updates nothing is nonsensical; an empty array fails closed."""

    # The empty array must raise the policy-violation (exit 1) error class.
    with pytest.raises(adc.PolicyViolationError):
        adc.parse_dependencies([])


# --- main: happy path and text manipulation ------------------------------------------


def test_happy_path_inserts_the_bullet_and_puts_the_exact_expected_text() -> None:
    """The full pipeline writes exactly the expected file text to the head branch.

    Covers the ordered call sequence (live PR fetch, contents GET, commits
    proof, contents PUT) and asserts the PUT payload byte-for-byte: decoded
    content, blob-sha precondition, target branch, and commit message.
    """

    # Full sequence: live PR, changelog GET, authorship proof, contents PUT.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit()),
            _completed({}),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    puts = _put_calls(mock_run)
    assert len(puts) == 1
    # The decoded content is the whole point of the run; compare exactly.
    assert _decoded_put_content(puts[0]) == _EXPECTED_HAPPY
    argv = puts[0].args[0]
    assert f"repos/{_REPO}/contents/CHANGELOG.md" in argv
    assert "sha=blob-sha-1" in argv
    assert f"branch={_HEAD_REF}" in argv
    assert f"message=chore(deps): record Dependabot update in CHANGELOG (#{_PR_NUMBER})" in argv


def test_missing_dependencies_heading_is_created_at_the_end_of_unreleased() -> None:
    """With no '### Dependencies' heading, one is created before the next '## ' heading.

    The created subsection is blank-line separated on both sides, per the
    release-freeze-reset recovery path in the script contract.
    """

    # Same sequence as the happy path, against the heading-less fixture.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITHOUT_HEADING),
            _commits(_commit()),
            _completed({}),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert _decoded_put_content(_put_calls(mock_run)[0]) == _EXPECTED_HEADING_CREATED


def test_stale_keyed_bullet_is_replaced_in_place_and_neighbors_survive() -> None:
    """An existing (#192)-keyed bullet with different content is replaced at its position.

    The (#188) neighbor bullet must survive untouched -- replace-or-insert is
    keyed strictly on this PR's own token.
    """

    # The fixture carries a stale (#192) bullet from a superseded version range.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_STALE_BULLET),
            _commits(_commit()),
            _completed({}),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert _decoded_put_content(_put_calls(mock_run)[0]) == _EXPECTED_REPLACED


def test_exact_match_is_a_no_op_with_no_put_and_no_authorship_fetch() -> None:
    """When the desired entry is already present, the run exits 0 without writing.

    The idempotency check runs before the authorship proof (the self-trigger
    loop terminator), so neither the commits endpoint nor the PUT may be
    called at all: exactly the PR fetch and the contents GET happen.
    """

    # Only two reads may occur; any further call would exhaust the side_effect list.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_ALREADY_CURRENT),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert mock_run.call_count == 2
    assert _put_calls(mock_run) == []


def test_grouped_multi_dependency_bullets_render_sorted_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grouped update renders one bullet per dependency, sorted by dependencyName.

    Args:
        monkeypatch: pytest's environment-patching fixture, used to supply a
            two-dependency payload in deliberately reverse-sorted order.
    """

    monkeypatch.setenv(
        adc.UPDATED_DEPENDENCIES_ENV,
        json.dumps([_dep(name="zebra-pkg", prev="2.0", new="2.1"), _dep(name="alpha-pkg")]),
    )
    # Standard happy-path sequence; only the rendered block differs.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit()),
            _completed({}),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    content = _decoded_put_content(_put_calls(mock_run)[0])
    alpha = content.index("- Bump `alpha-pkg` from 1.0.0 to 1.1.0 (uv) via Dependabot (#192).")
    zebra = content.index("- Bump `zebra-pkg` from 2.0 to 2.1 (uv) via Dependabot (#192).")
    # alpha-pkg must precede zebra-pkg despite arriving second in the payload.
    assert alpha < zebra


# --- main: fail-closed metadata validation -------------------------------------------


@pytest.mark.parametrize(
    "hostile_name",
    [
        "bad`name",
        "bad$(name)",
        "bad\nname",
        "bad name; rm -rf /",
    ],
)
def test_hostile_dependency_names_are_rejected_with_exit_one(
    hostile_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shell metacharacters, backticks, spaces, and newlines in a name fail closed.

    Validation runs after the live PR re-fetch (phase order), so exactly one
    gh call happens and nothing is ever written.

    Args:
        hostile_name: The attacker-shaped dependencyName to reject.
        monkeypatch: pytest's environment-patching fixture.
    """

    monkeypatch.setenv(adc.UPDATED_DEPENDENCIES_ENV, json.dumps([_dep(name=hostile_name)]))
    # Only the live PR fetch may run before validation rejects the payload.
    with patch.object(subprocess, "run", side_effect=[_completed(_pr_payload())]) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 1
    assert mock_run.call_count == 1


def test_oversized_rendered_block_is_rejected_with_exit_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A rendered block above the 10KB cap fails closed before any write.

    Args:
        monkeypatch: pytest's environment-patching fixture.
        capsys: pytest's output-capture fixture.
    """

    # 60 individually valid deps with near-maximum-length names render well
    # past the 10KB cap while every element passes per-field validation.
    huge = [_dep(name=f"pkg{i:03d}" + "x" * 190) for i in range(60)]
    monkeypatch.setenv(adc.UPDATED_DEPENDENCIES_ENV, json.dumps(huge))
    # Only the live PR fetch may run before the size cap rejects the render.
    with patch.object(subprocess, "run", side_effect=[_completed(_pr_payload())]) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 1
    assert mock_run.call_count == 1
    assert "byte cap" in capsys.readouterr().err


# --- main: live PR re-validation ------------------------------------------------------


def test_live_pr_author_mismatch_fails_closed_with_exit_one(
    capsys: pytest.CaptureFixture,
) -> None:
    """A live PR authored by anyone but dependabot[bot] is rejected before any other call.

    Args:
        capsys: pytest's output-capture fixture.
    """

    # The single mocked call is the live re-fetch; rejection stops everything after it.
    with patch.object(
        subprocess, "run", side_effect=[_completed(_pr_payload(login="Jared-Godar"))]
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 1
    assert mock_run.call_count == 1
    assert "policy violation" in capsys.readouterr().err


def test_live_pr_author_type_not_bot_fails_closed_with_exit_one() -> None:
    """A user account masquerading with a bot-like login still fails the type check.

    user.type is server-attested: a renamed human account reports 'User', and
    the login check alone must not be the only line of defense.
    """

    # The login matches but the server-attested account type does not.
    with patch.object(
        subprocess, "run", side_effect=[_completed(_pr_payload(user_type="User"))]
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 1
    assert mock_run.call_count == 1


def test_live_head_sha_mismatch_is_a_superseded_run_with_exit_zero(
    capsys: pytest.CaptureFixture,
) -> None:
    """A moved live head sha means a newer synchronize run supersedes this one: exit 0, no write.

    Args:
        capsys: pytest's output-capture fixture.
    """

    # The live head moved past the sha this run was queued for.
    with patch.object(
        subprocess, "run", side_effect=[_completed(_pr_payload(head_sha="newer-sha"))]
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert mock_run.call_count == 1
    assert "supersedes" in capsys.readouterr().out


# --- main: commit authorship proof ----------------------------------------------------


def test_a_human_commit_on_the_branch_vetoes_the_write_with_exit_one(
    capsys: pytest.CaptureFixture,
) -> None:
    """One non-Dependabot commit among the PR's commits blocks the write entirely.

    A human commit means a human owns the changelog; the gate stays red for
    human attention rather than autofilling over their work.

    Args:
        capsys: pytest's output-capture fixture.
    """

    # The commits list mixes a legitimate Dependabot commit with a human one.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit(), _commit(sha="dead1234567890ab", login="Jared-Godar")),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 1
    assert _put_calls(mock_run) == []
    assert "authorship proof failed" in capsys.readouterr().err


def test_an_unverified_dependabot_commit_vetoes_the_write_with_exit_one() -> None:
    """A dependabot[bot]-authored commit without GitHub's verified signature still fails.

    fetch-metadata parses forgeable commit-message YAML; only GitHub's
    server-side signature verification makes the authorship claim trustworthy.
    """

    # The single commit claims the right author but lacks the verified signature.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit(verified=False)),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 1
    assert _put_calls(mock_run) == []


def test_all_verified_dependabot_commits_authorize_the_write() -> None:
    """Multiple commits that are all dependabot[bot]-authored and verified proceed to the PUT."""

    # Two verified Dependabot commits (e.g. after a rebase) pass the proof.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit(), _commit(sha="beef1234567890ab")),
            _completed({}),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert len(_put_calls(mock_run)) == 1


# --- main: optimistic-locking conflict handling ---------------------------------------


def test_a_conflicting_put_retries_once_from_the_get_and_succeeds() -> None:
    """A 409 on the first PUT triggers exactly one re-GET/re-PUT cycle that can succeed.

    The retry deliberately skips the authorship proof (the mover's own
    synchronize run re-proves it from scratch), so the sequence gains one
    contents GET and one PUT, never a second commits fetch.
    """

    # First PUT conflicts; the retry re-GETs and the second PUT lands.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit()),
            _CONFLICT_ERROR,
            _blob(_CHANGELOG_WITH_HEADING),
            _completed({}),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert len(_put_calls(mock_run)) == 2
    # Exactly one commits fetch: the retry must not re-run the authorship proof.
    commits_calls = [
        call for call in mock_run.call_args_list if any("commits" in a for a in call.args[0])
    ]
    assert len(commits_calls) == 1


def test_a_second_conflict_defers_to_the_newer_run_with_exit_zero(
    capsys: pytest.CaptureFixture,
) -> None:
    """Two consecutive PUT conflicts exit 0: the mover's own synchronize run supersedes this one.

    Args:
        capsys: pytest's output-capture fixture.
    """

    # Both PUT attempts conflict -- something keeps moving the head branch.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob(_CHANGELOG_WITH_HEADING),
            _commits(_commit()),
            _CONFLICT_ERROR,
            _blob(_CHANGELOG_WITH_HEADING),
            _CONFLICT_ERROR,
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 0
    assert len(_put_calls(mock_run)) == 2
    assert "deferring" in capsys.readouterr().out


# --- main: unreadable required data ----------------------------------------------------


def test_missing_updated_dependencies_env_exits_two_without_any_gh_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A missing UPDATED_DEPENDENCIES_JSON is unreadable required data: exit 2, zero gh calls.

    Args:
        monkeypatch: pytest's environment-patching fixture.
        capsys: pytest's output-capture fixture.
    """

    monkeypatch.delenv(adc.UPDATED_DEPENDENCIES_ENV)
    # Any subprocess call at all would be a defect; none may happen.
    with patch.object(subprocess, "run") as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 2
    mock_run.assert_not_called()
    assert adc.UPDATED_DEPENDENCIES_ENV in capsys.readouterr().err


def test_unparseable_updated_dependencies_json_exits_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undecodable JSON in the environment variable exits 2 before any gh call.

    Args:
        monkeypatch: pytest's environment-patching fixture.
    """

    monkeypatch.setenv(adc.UPDATED_DEPENDENCIES_ENV, "{this is not json")
    # No gh call may run against an input contract that cannot even be decoded.
    with patch.object(subprocess, "run") as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 2
    mock_run.assert_not_called()


def test_a_changelog_without_an_unreleased_section_exits_two_without_writing(
    capsys: pytest.CaptureFixture,
) -> None:
    """A changelog missing '## Unreleased' is structurally unreadable: exit 2, no PUT.

    Args:
        capsys: pytest's output-capture fixture.
    """

    # The fetched file has headings, but none of them is the Unreleased section.
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            _completed(_pr_payload()),
            _blob("# Changelog\n\n## 1.0.0 - 2026-01-01\n\n- Old entry.\n"),
        ],
    ) as mock_run:
        exit_code = adc.main(_ARGS)
    assert exit_code == 2
    assert _put_calls(mock_run) == []
    assert "Unreleased" in capsys.readouterr().err


# --- main: rate-limit classification ---------------------------------------------------


def test_a_primary_rate_limit_maps_to_exit_three(capsys: pytest.CaptureFixture) -> None:
    """Primary rate-limit exhaustion exits 3, distinct from policy (1) and data (2) failures.

    Args:
        capsys: pytest's output-capture fixture.
    """

    exhausted = subprocess.CalledProcessError(
        1, ["gh"], stderr="API rate limit already exceeded for user ID 16855088."
    )
    # The very first gh call hits the drained pool and fails fast.
    with patch.object(subprocess, "run", side_effect=exhausted):
        exit_code = adc.main(_ARGS)
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err
