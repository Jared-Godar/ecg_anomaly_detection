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

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "detect_label_drift.py"
_SPEC = importlib.util.spec_from_file_location("detect_label_drift", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
dld = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dld
_SPEC.loader.exec_module(dld)


# --- load_canonical_label_names -----------------------------------------------------


def test_load_canonical_label_names_reads_the_real_manifest() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / ".github" / "labels.json"
    names = dld.load_canonical_label_names(manifest_path)
    assert "type: modernization" in names
    assert "area: repository" in names
    assert isinstance(names, frozenset)


def test_load_canonical_label_names_rejects_a_missing_schema_version(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"labels": []}), encoding="utf-8")
    with pytest.raises(dld.LabelDriftError, match="schema_version"):
        dld.load_canonical_label_names(manifest)


def test_load_canonical_label_names_rejects_an_unnamed_label(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "labels": [{"color": "ffffff"}]}), encoding="utf-8"
    )
    with pytest.raises(dld.LabelDriftError, match="non-empty name"):
        dld.load_canonical_label_names(manifest)


# --- find_drifted_labels / find_drifted_items -----------------------------------------


def test_find_drifted_labels_returns_only_non_canonical_entries() -> None:
    canonical = frozenset({"type: modernization", "area: repository"})
    result = dld.find_drifted_labels(
        ["type: modernization", "area:repository", "priority:p4"], canonical
    )
    assert result == ("area:repository", "priority:p4")


def test_find_drifted_labels_with_no_drift_returns_empty_tuple() -> None:
    canonical = frozenset({"type: modernization", "area: repository"})
    result = dld.find_drifted_labels(["type: modernization", "area: repository"], canonical)
    assert result == ()


def test_find_drifted_items_flags_only_items_with_drift() -> None:
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
    canonical = frozenset({"type: modernization"})
    items = [{"kind": "issue", "number": 1, "title": "x", "labels": ["type: modernization"]}]
    assert dld.find_drifted_items(items, canonical) == ()


def test_a_label_present_in_the_manifest_style_but_not_the_set_still_drifts() -> None:
    # Guards the real finding this tool surfaced: a label can match canonical
    # *formatting* (space after colon) while still not being in the manifest
    # (e.g. "risk: low" is real-world drift even though it looks canonical-style).
    canonical = frozenset({"risk: data-integrity", "risk: evaluation", "risk: security"})
    assert dld.find_drifted_labels(["risk: low"], canonical) == ("risk: low",)


# --- I/O boundary (mocked subprocess) -----------------------------------------------


def test_fetch_items_combines_issues_and_pull_requests() -> None:
    issue_stdout = json.dumps([{"number": 1, "title": "an issue", "labels": [{"name": "a"}]}])
    pr_stdout = json.dumps([{"number": 2, "title": "a pr", "labels": [{"name": "b"}]}])
    responses = iter([issue_stdout, pr_stdout])

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=next(responses), stderr="")

    with patch.object(subprocess, "run", side_effect=fake_run):
        items = dld.fetch_items(repo=None, include_closed=False)

    assert len(items) == 2
    assert items[0] == {"kind": "issue", "number": 1, "title": "an issue", "labels": ["a"]}
    assert items[1] == {"kind": "pull request", "number": 2, "title": "a pr", "labels": ["b"]}


def test_fetch_items_raises_label_drift_error_on_gh_failure() -> None:
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, ["gh"], stderr="not authenticated"),
    ):
        with pytest.raises(dld.LabelDriftError, match="not authenticated"):
            dld.fetch_items(repo=None, include_closed=False)
