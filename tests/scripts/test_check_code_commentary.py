"""Test the exhaustive internal-commentary policy checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Standalone repository scripts are not installed as package modules, so load the
# checker through its concrete path exactly as production-adjacent script tests do.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_code_commentary.py"
# Preserve the import specification so the loader and module registration use one identity.
_SPEC = importlib.util.spec_from_file_location("check_code_commentary", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# Register the dynamically created module before execution because dataclasses resolve their
# defining module through `sys.modules` while class decorators run.
commentary = importlib.util.module_from_spec(_SPEC)
# Expose the module under the specification name before executing its class decorators.
sys.modules[_SPEC.name] = commentary
_SPEC.loader.exec_module(commentary)

# Bind the public script functions locally so individual tests read like ordinary unit tests.
audit_file = commentary.audit_file
# Expose discovery separately because its deterministic path handling has an isolated contract.
discover_python_files = commentary.discover_python_files
# Expose the command entry point so tests can verify process-status behavior without a subprocess.
main = commentary.main


def _write(tmp_path: Path, source: str) -> Path:
    """Create one temporary Python source file for an isolated audit scenario."""

    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_audit_reports_missing_module_and_definition_docstrings(tmp_path: Path) -> None:
    """Verify missing module and function narration produces precise diagnostics."""

    path = _write(tmp_path, "def undocumented():\n    return 1\n")

    violations = audit_file(path)

    # Assert both policy layers so one cannot regress while the other still fails.
    assert [item.message for item in violations] == [
        "module is missing a docstring",
        "function 'undocumented' is missing a docstring",
    ]


def test_audit_requires_comments_before_control_flow_and_module_objects(tmp_path: Path) -> None:
    """Verify assignments and control-flow blocks require preceding standalone comments."""

    path = _write(
        tmp_path,
        '"""Fixture module."""\n\nVALUE = 1\n\ndef choose(flag: bool) -> int:\n'
        '    """Choose a fixture value."""\n    if flag:\n        return VALUE\n    return 0\n',
    )

    violations = audit_file(path)

    # Keep diagnostics tied to the exact missing guideposts under test.
    assert [item.message for item in violations] == [
        "module-level assignment is missing a leading explanation comment",
        "If block is missing a leading explanation comment",
    ]


def test_audit_accepts_fully_narrated_source_and_elif_continuations(tmp_path: Path) -> None:
    """Verify compliant narration passes without demanding impossible comments before elif."""

    path = _write(
        tmp_path,
        '"""Fixture module."""\n\n# Keep the fixture threshold explicit for the branch test.\nVALUE = 1\n\n'
        'def choose(flag: int) -> int:\n    """Choose a fixture value for the supplied branch."""\n\n'
        "    # Separate positive and zero values so the behavior is explicit.\n"
        "    if flag > 0:\n        return VALUE\n    elif flag == 0:\n        return 0\n    return -1\n",
    )

    # A fully narrated file should produce no policy diagnostics.
    assert audit_file(path) == ()


def test_audit_reaches_nested_callables(tmp_path: Path) -> None:
    """Verify an inner helper cannot evade the definition-docstring requirement."""

    path = _write(
        tmp_path,
        '"""Fixture module."""\n\ndef outer() -> int:\n'
        '    """Return the value produced by an inner helper."""\n\n'
        "    def inner() -> int:\n        return 1\n\n    return inner()\n",
    )

    violations = audit_file(path)

    # Pin the diagnostic to the inner definition so a shallow tree walk cannot pass this test.
    assert [item.message for item in violations] == ["function 'inner' is missing a docstring"]


def test_discovery_is_recursive_unique_and_sorted(tmp_path: Path) -> None:
    """Verify source discovery is deterministic across files and nested directories."""

    nested = tmp_path / "nested"
    nested.mkdir()
    first = _write(tmp_path, '"""First."""\n')
    second = nested / "second.py"
    second.write_text('"""Second."""\n', encoding="utf-8")

    files = discover_python_files((nested, first, tmp_path))

    # Deduplication prevents overlapping roots from auditing the same file twice.
    assert files == tuple(sorted((first, second)))


def test_main_fails_when_no_python_files_are_discovered(tmp_path: Path) -> None:
    """Verify an empty audit target cannot masquerade as successful coverage."""

    # Exit status two distinguishes configuration failure from policy violations.
    assert main([str(tmp_path)]) == 2
