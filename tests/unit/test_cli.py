"""Tests for CLI-boundary helpers that are not tied to one subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecg_anomaly_detection.cli import ArtifactDiscoveryError, _resolve_input_paths


def test_file_arguments_pass_through_unchanged(tmp_path: Path) -> None:
    """Verify that file arguments pass through unchanged.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    artifact = tmp_path / "100.npz"
    artifact.write_bytes(b"data")

    assert _resolve_input_paths([artifact]) == (artifact,)


def test_directory_expands_to_sorted_npz_children(tmp_path: Path) -> None:
    """Verify that directory expands to sorted npz children.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    directory = tmp_path / "windows"
    directory.mkdir()
    (directory / "102.npz").write_bytes(b"data")
    (directory / "100.npz").write_bytes(b"data")
    (directory / "101.npz").write_bytes(b"data")
    (directory / "notes.txt").write_bytes(b"ignored")

    result = _resolve_input_paths([directory])

    assert [path.name for path in result] == ["100.npz", "101.npz", "102.npz"]


def test_directory_is_not_searched_recursively(tmp_path: Path) -> None:
    """Verify that directory is not searched recursively.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    directory = tmp_path / "windows"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "100.npz").write_bytes(b"data")
    (nested / "101.npz").write_bytes(b"data")

    result = _resolve_input_paths([directory])

    assert [path.name for path in result] == ["100.npz"]


def test_mixed_file_and_directory_arguments_combine(tmp_path: Path) -> None:
    """Verify that mixed file and directory arguments combine.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    directory = tmp_path / "windows"
    directory.mkdir()
    (directory / "101.npz").write_bytes(b"data")
    standalone = tmp_path / "100.npz"
    standalone.write_bytes(b"data")

    result = _resolve_input_paths([standalone, directory])

    assert [path.name for path in result] == ["100.npz", "101.npz"]


def test_empty_directory_raises_a_clear_error(tmp_path: Path) -> None:
    """Verify that empty directory raises a clear error.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    directory = tmp_path / "empty"
    directory.mkdir()

    # Scope `pytest.raises(ArtifactDiscoveryError, match='no \\*.npz artifact files found')` here so
    # the expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(ArtifactDiscoveryError, match="no \\*.npz artifact files found"):
        _resolve_input_paths([directory])


def test_nonexistent_path_raises_a_clear_error(tmp_path: Path) -> None:
    """Verify that nonexistent path raises a clear error.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    # Scope `pytest.raises(ArtifactDiscoveryError, match='does not exist')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(ArtifactDiscoveryError, match="does not exist"):
        _resolve_input_paths([tmp_path / "missing"])


def test_symlinked_directory_is_rejected(tmp_path: Path) -> None:
    """Verify that symlinked directory is rejected.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    real_target = tmp_path / "real"
    real_target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_target, target_is_directory=True)

    # Scope `pytest.raises(ArtifactDiscoveryError, match='symbolic link')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(ArtifactDiscoveryError, match="symbolic link"):
        _resolve_input_paths([link])


def test_symlinked_file_is_rejected(tmp_path: Path) -> None:
    """Verify that symlinked file is rejected.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    real_target = tmp_path / "real.npz"
    real_target.write_bytes(b"data")
    link = tmp_path / "link.npz"
    link.symlink_to(real_target)

    # Scope `pytest.raises(ArtifactDiscoveryError, match='symbolic link')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(ArtifactDiscoveryError, match="symbolic link"):
        _resolve_input_paths([link])


def test_directory_expansion_overlapping_an_explicit_file_is_rejected(tmp_path: Path) -> None:
    """Verify that directory expansion overlapping an explicit file is rejected.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    directory = tmp_path / "windows"
    directory.mkdir()
    shard = directory / "100.npz"
    shard.write_bytes(b"data")

    # Scope `pytest.raises(ArtifactDiscoveryError, match='duplicate input artifact')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(ArtifactDiscoveryError, match="duplicate input artifact"):
        _resolve_input_paths([directory, shard])
