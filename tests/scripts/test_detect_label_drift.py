"""Tests for the label-drift detection script.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`.
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
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "detect_label_drift.py"
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("detect_label_drift", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. dld.find_drifted_labels).
dld = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern.
sys.modules[_SPEC.name] = dld
_SPEC.loader.exec_module(dld)


# --- load_canonical_label_names -----------------------------------------------------


def test_load_canonical_label_names_reads_the_real_manifest() -> None:
    """The actual committed .github/labels.json parses into a frozenset containing known canonical labels."""

    manifest_path = Path(__file__).resolve().parents[2] / ".github" / "labels.json"
    names = dld.load_canonical_label_names(manifest_path)
    assert "type: modernization" in names
    assert "area: repository" in names
    assert isinstance(names, frozenset)


def test_load_canonical_label_names_rejects_a_missing_schema_version(tmp_path: Path) -> None:
    """A labels.json with no top-level schema_version key is rejected rather than silently accepted.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"labels": []}), encoding="utf-8")
    # This fixture's manifest has no "schema_version" key at all.
    with pytest.raises(dld.LabelDriftError, match="schema_version"):
        dld.load_canonical_label_names(manifest)


def test_load_canonical_label_names_rejects_an_unnamed_label(tmp_path: Path) -> None:
    """A label entry with no "name" field (only a color) is rejected.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    manifest = tmp_path / "labels.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "labels": [{"color": "ffffff"}]}), encoding="utf-8"
    )
    # This fixture's one label entry has a "color" but no "name".
    with pytest.raises(dld.LabelDriftError, match="non-empty name"):
        dld.load_canonical_label_names(manifest)


# --- find_drifted_labels / find_drifted_items -----------------------------------------


def test_find_drifted_labels_returns_only_non_canonical_entries() -> None:
    """Of three labels, only the two that aren't exact members of the canonical set are returned."""

    canonical = frozenset({"type: modernization", "area: repository"})
    result = dld.find_drifted_labels(
        ["type: modernization", "area:repository", "priority:p4"], canonical
    )
    assert result == ("area:repository", "priority:p4")


def test_find_drifted_labels_with_no_drift_returns_empty_tuple() -> None:
    """When every label on an item is already canonical, find_drifted_labels returns nothing."""

    canonical = frozenset({"type: modernization", "area: repository"})
    result = dld.find_drifted_labels(["type: modernization", "area: repository"], canonical)
    assert result == ()


def test_find_drifted_items_flags_only_items_with_drift() -> None:
    """Of two items, only the one carrying a non-canonical label is included in the drifted result."""

    canonical = frozenset({"type: modernization"})
    items = [
        {"kind": "issue", "number": 61, "title": "clean", "labels": ["type: modernization"]},
        {"kind": "issue", "number": 42, "title": "GOV-006", "labels": ["type:governance"]},
    ]
    drifted = dld.find_drifted_items(items, canonical)
    assert len(drifted) == 1
    assert drifted[0].number == 42
    assert drifted[0].kind == "issue"
    assert drifted[0].drifted_labels == ("type:governance",)


def test_find_drifted_items_with_a_fully_clean_set_returns_empty_tuple() -> None:
    """A batch of items with only canonical labels produces an empty drifted-items tuple."""

    canonical = frozenset({"type: modernization"})
    items = [{"kind": "issue", "number": 1, "title": "x", "labels": ["type: modernization"]}]
    assert dld.find_drifted_items(items, canonical) == ()


def test_a_label_present_in_the_manifest_style_but_not_the_set_still_drifts() -> None:
    """A label matching canonical *formatting* (space after the colon) but absent from the set still drifts.

    Guards the real finding this tool surfaced: "risk: low" looks like a
    well-formed canonical label but was never actually added to
    .github/labels.json, so exact-set membership -- not just formatting --
    must be what determines drift.
    """

    canonical = frozenset({"risk: data-integrity", "risk: evaluation", "risk: security"})
    assert dld.find_drifted_labels(["risk: low"], canonical) == ("risk: low",)


# --- I/O boundary (mocked subprocess) -----------------------------------------------


def test_fetch_items_combines_issues_and_pull_requests() -> None:
    """fetch_items merges the separate `gh issue list` and `gh pr list` calls into one tagged sequence.

    Confirms each item is tagged with its originating "kind" ("issue" vs
    "pull request") and that both label lists are flattened from gh's
    nested {"name": ...} objects to plain strings.
    """

    issue_stdout = json.dumps([{"number": 1, "title": "an issue", "labels": [{"name": "a"}]}])
    pr_stdout = json.dumps([{"number": 2, "title": "a pr", "labels": [{"name": "b"}]}])
    responses = iter([issue_stdout, pr_stdout])

    def fake_run(cmd, **kwargs):
        """Return the next queued fake gh response, ignoring which command was actually run.

        Args:
            cmd: The subprocess command list (unused; only call order matters).
            kwargs: Ignored subprocess.run keyword arguments.

        Returns:
            A successful CompletedProcess with the next fixture response as stdout.
        """

        return subprocess.CompletedProcess(cmd, 0, stdout=next(responses), stderr="")

    # The first call is expected to be `gh issue list`, the second `gh pr list`.
    with patch.object(subprocess, "run", side_effect=fake_run):
        items = dld.fetch_items(repo=None, include_closed=False)

    assert len(items) == 2
    assert items[0] == {"kind": "issue", "number": 1, "title": "an issue", "labels": ["a"]}
    assert items[1] == {"kind": "pull request", "number": 2, "title": "a pr", "labels": ["b"]}


def test_fetch_items_raises_label_drift_error_on_gh_failure() -> None:
    """A failing `gh` invocation (e.g. an unauthenticated CLI) is translated into LabelDriftError."""

    # gh exits non-zero when the local CLI has no valid authentication.
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="not authenticated"),
        ),
        pytest.raises(dld.LabelDriftError, match="not authenticated"),
    ):
        dld.fetch_items(repo=None, include_closed=False)
