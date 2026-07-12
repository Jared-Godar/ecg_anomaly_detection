"""Tests for the label-sync script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. Every test mocks the subprocess
boundary; none performs a live GitHub call.

The shared GitHub access layer (`scripts/github/github_api.py`) the script
migrated onto in issue #175 has its own test module; the tests here cover
this script's orchestration on top of it -- in particular that --dry-run
still performs no gh calls at all, that the repository-root working
directory reaches gh unchanged, and that the observe-only quota default
never blocks a manual hygiene run.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sync_github_labels.py"
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("sync_github_labels", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. sgl.load_labels).
sgl = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern.
sys.modules[_SPEC.name] = sgl
_SPEC.loader.exec_module(sgl)


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


def _write_manifest(tmp_path: Path, labels: list[dict[str, str]]) -> Path:
    """Write a schema-version-1 label manifest fixture and return its path.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        labels: The label entries to declare in the fixture manifest.

    Returns:
        The path of the written manifest file.
    """

    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"schema_version": 1, "labels": labels}), encoding="utf-8")
    return manifest


# A minimal two-label manifest reused by every main() test; two entries prove
# the sync loops over the whole manifest rather than stopping after one.
_TWO_LABELS: list[dict[str, str]] = [
    {"name": "type: governance", "color": "5319E7", "description": "Policy"},
    {"name": "risk: low", "color": "0e8a16", "description": "Low risk"},
]


# --- load_labels -----------------------------------------------------------------------


def test_load_labels_reads_the_real_manifest() -> None:
    """The actual committed .github/labels.json parses, with colors normalized to lowercase."""

    manifest_path = Path(__file__).resolve().parents[2] / ".github" / "labels.json"
    labels = sgl.load_labels(manifest_path)
    assert labels, "the committed manifest must declare at least one label"
    # Every entry carries the three required fields, with the color lowercased.
    for label in labels:
        assert set(label) == {"name", "color", "description"}
        assert label["color"] == label["color"].lower()


def test_load_labels_rejects_a_missing_schema_version(tmp_path: Path) -> None:
    """A labels.json with no top-level schema_version key is rejected rather than silently accepted.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    manifest = tmp_path / "labels.json"
    # This fixture's manifest has no "schema_version" key at all.
    manifest.write_text(json.dumps({"labels": []}), encoding="utf-8")
    # The loader must name the missing key rather than accept the manifest.
    with pytest.raises(ValueError, match="schema_version"):
        sgl.load_labels(manifest)


def test_load_labels_rejects_a_label_missing_a_field(tmp_path: Path) -> None:
    """A label entry without all three of name, color, and description is rejected.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # This fixture's one label entry has no "description".
    manifest = _write_manifest(tmp_path, [{"name": "x", "color": "ffffff"}])
    # The loader must name all three required fields in its rejection.
    with pytest.raises(ValueError, match="non-empty name, color, and description"):
        sgl.load_labels(manifest)


def test_load_labels_rejects_a_duplicate_name(tmp_path: Path) -> None:
    """Two manifest entries declaring the same label name are rejected.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # Both fixture entries compete to define the label "x".
    duplicate = [
        {"name": "x", "color": "ffffff", "description": "one"},
        {"name": "x", "color": "000000", "description": "two"},
    ]
    # The loader must name the duplicated label in its rejection.
    with pytest.raises(ValueError, match="duplicate label: x"):
        sgl.load_labels(_write_manifest(tmp_path, duplicate))


def test_load_labels_rejects_an_invalid_color(tmp_path: Path) -> None:
    """A color that is not exactly six bare hex digits is rejected.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    # The leading '#' makes this color invalid for `gh label create`.
    bad_color = [{"name": "x", "color": "#fffff", "description": "d"}]
    # The loader must reject the malformed color before it ever reaches gh.
    with pytest.raises(ValueError, match="invalid six-digit color"):
        sgl.load_labels(_write_manifest(tmp_path, bad_color))


# --- main: --dry-run (no subprocess at all) --------------------------------------------


def test_dry_run_prints_commands_without_any_subprocess_call(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """--dry-run prints one pasteable gh command per label and never spawns a process.

    Pre-migration, a dry run made zero subprocess calls -- not even quota
    accounting -- and issue #175's zero-behavior-change non-goal keeps it
    that way, so it must work offline and unauthenticated.
    """

    manifest = _write_manifest(tmp_path, _TWO_LABELS)
    # Any subprocess call at all would fail the test immediately.
    with patch.object(
        subprocess, "run", side_effect=AssertionError("dry-run must not spawn processes")
    ):
        exit_code = sgl.main(["--manifest", str(manifest), "--dry-run"])
    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    # One JSON-encoded command per manifest label, in manifest order, each
    # keeping its leading "gh" so a reviewer can paste it into a shell.
    assert [json.loads(line) for line in lines] == [
        [
            "gh",
            "label",
            "create",
            "type: governance",
            "--color",
            "5319e7",
            "--description",
            "Policy",
            "--force",
        ],
        [
            "gh",
            "label",
            "create",
            "risk: low",
            "--color",
            "0e8a16",
            "--description",
            "Low risk",
            "--force",
        ],
    ]


# --- main: live sync (mocked subprocess) ------------------------------------------------


def test_main_syncs_every_label_with_expected_args_and_reports_quota(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A live sync creates every manifest label via gh and prints the quota report.

    The exact argument list pins each flag to its own value, and the
    repository-root working directory must reach subprocess.run unchanged --
    that is what keeps gh's repository inference identical to the
    pre-migration subprocess.run(cwd=ROOT) call.
    """

    manifest = _write_manifest(tmp_path, _TWO_LABELS)
    # Full sequence: preflight, one create per label, report.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(4990), _completed(""), _completed(""), _quota(4990)],
    ) as mock_run:
        exit_code = sgl.main(["--manifest", str(manifest)])
    assert exit_code == 0
    # The second call is the first label's mutation (the first is the preflight).
    assert mock_run.call_args_list[1].args[0] == [
        "gh",
        "label",
        "create",
        "type: governance",
        "--color",
        "5319e7",
        "--description",
        "Policy",
        "--force",
    ]
    # Both mutations run from the repository root; the free rate_limit reads
    # have no cwd requirement and pass None like any plain run_gh call.
    assert mock_run.call_args_list[1].kwargs["cwd"] == sgl.ROOT
    assert mock_run.call_args_list[2].kwargs["cwd"] == sgl.ROOT
    captured = capsys.readouterr()
    # The consumption report names all three accounting values (REST mutations
    # spend no GraphQL points, so a quiet pool shows zero consumed).
    assert "4990 before" in captured.err
    assert "4990 after" in captured.err
    assert "0 consumed" in captured.err


def test_main_passes_an_explicit_repo_through_to_gh(tmp_path: Path) -> None:
    """--repo OWNER/REPO is appended to every label mutation gh runs."""

    manifest = _write_manifest(tmp_path, _TWO_LABELS[:1])
    # Preflight, the single label's mutation, report.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(4990), _completed(""), _quota(4990)],
    ) as mock_run:
        exit_code = sgl.main(
            ["--manifest", str(manifest), "--repo", "Jared-Godar/ecg_anomaly_detection"]
        )
    assert exit_code == 0
    # The mutation's trailing arguments carry the explicit repo selection.
    assert mock_run.call_args_list[1].args[0][-2:] == [
        "--repo",
        "Jared-Godar/ecg_anomaly_detection",
    ]


def test_main_forwards_gh_confirmation_output(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Any stdout gh produces for a mutation is forwarded verbatim to the operator.

    run_gh captures gh's output instead of letting it reach the terminal
    directly, so the script re-emits whatever gh said (usually nothing when
    captured) rather than silently discarding it.
    """

    manifest = _write_manifest(tmp_path, _TWO_LABELS[:1])
    # This fixture's gh emits a confirmation line despite being captured.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(4990), _completed("label created\n"), _quota(4990)],
    ):
        exit_code = sgl.main(["--manifest", str(manifest)])
    assert exit_code == 0
    assert "label created" in capsys.readouterr().out


def test_main_returns_two_on_a_gh_failure(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A gh CLI failure (e.g. insufficient token scope) exits 2 with a clean message.

    Pre-migration this surfaced as an uncaught CalledProcessError traceback;
    the shared layer translates it into GitHubApiError, which main() maps to
    the conventional failure exit code.
    """

    manifest = _write_manifest(tmp_path, _TWO_LABELS[:1])
    auth_error = subprocess.CalledProcessError(1, ["gh"], stderr="insufficient scope")
    # Preflight passes, the mutation fails, the report still runs.
    with patch.object(subprocess, "run", side_effect=[_quota(4990), auth_error, _quota(4990)]):
        exit_code = sgl.main(["--manifest", str(manifest)])
    assert exit_code == 2
    assert "insufficient scope" in capsys.readouterr().err


def test_main_default_threshold_never_blocks_even_on_a_drained_pool(tmp_path: Path) -> None:
    """With the observe-only default, a fully drained GraphQL pool still syncs labels.

    This is issue #175's explicit non-goal made executable: the sync's
    mutations are REST, so even a zero-point GraphQL pool must not stop a
    manual hygiene run under the default threshold.
    """

    manifest = _write_manifest(tmp_path, _TWO_LABELS[:1])
    # The pool reads 0 remaining at preflight and at report time alike.
    with patch.object(
        subprocess,
        "run",
        side_effect=[_quota(0), _completed(""), _quota(0)],
    ) as mock_run:
        exit_code = sgl.main(["--manifest", str(manifest)])
    assert exit_code == 0
    # The mutation must actually have run despite the drained pool.
    assert "create" in mock_run.call_args_list[1].args[0]


def test_main_stops_with_exit_three_when_an_explicit_threshold_is_undercut(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """An operator-supplied positive threshold stops the run before any mutation.

    The stop must happen before the first label mutation, and its exit code
    must differ from the failure code (2) so hygiene output can never
    conflate a drained pool with a label defect.
    """

    manifest = _write_manifest(tmp_path, _TWO_LABELS)
    # Both rate_limit reads (preflight and report) see a nearly drained pool.
    with patch.object(subprocess, "run", side_effect=[_quota(12), _quota(12)]) as mock_run:
        exit_code = sgl.main(["--manifest", str(manifest), "--min-graphql-quota", "50"])
    assert exit_code == 3
    captured = capsys.readouterr()
    assert "quota:" in captured.err
    assert "only 12 of 5000" in captured.err
    # The only gh traffic allowed is the free REST rate_limit accounting --
    # not one label mutation may have been attempted.
    for call in mock_run.call_args_list:
        assert "rate_limit" in call.args[0]
        assert "create" not in call.args[0]


def test_main_maps_a_primary_rate_limit_to_exit_three(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Primary rate-limit exhaustion mid-run exits 3, distinct from ordinary failures.

    The drained shared pool is infrastructure, not a defect in the label
    manifest; the exit code and the "quota:" prefix keep the two
    unmistakably apart.
    """

    exhausted = subprocess.CalledProcessError(
        1, ["gh"], stderr="API rate limit already exceeded for user ID 16855088."
    )
    manifest = _write_manifest(tmp_path, _TWO_LABELS[:1])
    # Preflight passes (observe-only default), the mutation then hits the
    # exhausted pool, and the report still runs.
    with patch.object(subprocess, "run", side_effect=[_quota(60), exhausted, _quota(0)]):
        exit_code = sgl.main(["--manifest", str(manifest)])
    assert exit_code == 3
    assert "quota:" in capsys.readouterr().err
