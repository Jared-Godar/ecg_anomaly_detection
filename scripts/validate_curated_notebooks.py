#!/usr/bin/env python3
"""Execute the three curated public notebooks end-to-end without the real dataset.

Runs `notebooks/00-environment-setup-and-artifact-generation.ipynb`, then
`01-narrative-walkthrough.ipynb`, then
`02-high-performing-gradient-boosting-validation.ipynb` in an isolated `git
worktree` copy of the repository, seeded with a small synthetic WFDB record
set and a matching acquisition manifest so `acquire_dataset` takes its
existing verify-and-reuse path and never calls the network fetcher (see
`ecg_anomaly_detection.acquisition.acquire_dataset`). This is genuine
cell-by-cell execution of the real, unmodified curated notebooks, not a
structural/syntax-only check -- DX-007 (`notebook_quality.py`) already
covers that and explicitly does not execute notebooks.

Two repository-tracked config files are overridden inside the isolated copy
only (never the real checked-out files): `configs/mitdb-v1.0.0.toml` (trimmed
`record_ids`/`expected_source_files` to the synthetic set) and
`configs/splitting-v2.toml` (synthetic `record_subjects`, and quality
thresholds sized for a six-record fixture instead of the real dataset's
production thresholds). Every other config, and all three notebooks
themselves, run completely unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# The three curated notebooks, in required execution order (00 generates artifacts
# the later two read); this is the default `notebooks` argument to
# validate_curated_notebooks, overridable in tests to exercise a subset.
NOTEBOOKS: tuple[str, ...] = (
    "notebooks/00-environment-setup-and-artifact-generation.ipynb",
    "notebooks/01-narrative-walkthrough.ipynb",
    "notebooks/02-high-performing-gradient-boosting-validation.ipynb",
)

# Six record IDs (900-905), chosen outside the real MIT-BIH numbering range so a
# synthetic record can never be mistaken for (or collide with) a real acquired one.
SYNTHETIC_RECORD_IDS: tuple[str, ...] = ("900", "901", "902", "903", "904", "905")
# Matches the real MIT-BIH dataset's sample rate, since the dataset config asserts
# this value and the curated notebooks assume it when reasoning about durations.
SAMPLE_RATE_HZ = 360
# Short enough that six synthetic records generate and execute quickly, while still
# long enough to fit every beat with its full pre/post window margin (see
# BEAT_MARGIN_SAMPLES) without any beat falling outside the record.
RECORD_SECONDS = 15
# 3s pre/post window margin at 360 Hz is 1080 samples each side; keep beats
# well clear of both record edges so none are excluded by boundary_policy.
BEAT_MARGIN_SAMPLES = 1200
# One "N" (normal) and one "V" (ventricular ectopic) pair per record on average,
# giving every synthetic record both target classes so the pipeline's binary
# classification and split-quality class-coverage checks have real data to exercise.
BEAT_SYMBOLS: tuple[str, ...] = ("N", "N", "V", "N", "V", "N")


class NotebookValidationError(RuntimeError):
    """Raised when curated-notebook execution cannot be set up or does not pass."""


@dataclass(frozen=True, slots=True)
class SyntheticFile:
    """One generated WFDB file's identity, for building config/manifest content."""

    name: str
    size_bytes: int
    sha256: str


def build_dataset_config_toml(files: Sequence[SyntheticFile], record_ids: Sequence[str]) -> str:
    """Return a trimmed configs/mitdb-v1.0.0.toml restricted to the synthetic records.

    Keeps schema, slug, version, and both URLs identical to the real config --
    they are format-validated but never dereferenced, since the pre-seeded
    acquisition manifest below means the network fetcher is never called.
    """
    expected_source_files = "\n".join(
        f'  {{ path = "{item.name}", size_bytes = {item.size_bytes}, sha256 = "{item.sha256}" }},'
        for item in sorted(files, key=lambda item: item.name)
    )
    record_ids_toml = ", ".join(f'"{record_id}"' for record_id in record_ids)
    return (
        "schema_version = 1\n\n"
        "[dataset]\n"
        'name = "MIT-BIH Arrhythmia Database"\n'
        'slug = "mitdb"\n'
        'version = "1.0.0"\n'
        'source_url = "https://physionet.org/content/mitdb/1.0.0/"\n'
        'download_url = "https://physionet.org/files/mitdb/1.0.0/"\n'
        "sample_rate_hz = 360\n"
        'annotation_extension = "atr"\n'
        'required_extensions = ["atr", "dat", "hea"]\n'
        "expected_source_files = [\n"
        f"{expected_source_files}\n"
        "]\n"
        f"record_ids = [{record_ids_toml}]\n"
    )


def build_split_config_toml(record_ids: Sequence[str]) -> str:
    """Return a trimmed configs/splitting-v2.toml sized for the synthetic fixture.

    Keeps the real strategy, seed, and ratios. Quality thresholds are relaxed
    from the real config's production minimums (5 subjects/records per
    partition) to fit a six-record synthetic set -- this check validates that
    the pipeline mechanically runs end to end, not production-grade split
    quality, which the real config already enforces for real runs.
    """
    record_subjects = "\n".join(f'{record_id} = "subject-{record_id}"' for record_id in record_ids)
    return (
        "schema_version = 2\n\n"
        "[split]\n"
        'name = "subject-aware-holdout"\n'
        'version = "2.0.0"\n'
        'strategy = "seeded-subject-shuffle"\n'
        "seed = 2022\n\n"
        "[split.ratios]\n"
        "train = 0.70\n"
        "validation = 0.15\n"
        "test = 0.15\n\n"
        "[split.quality]\n"
        "min_subjects_per_partition = 1\n"
        "min_records_per_partition = 1\n"
        "min_windows_per_partition = 1\n"
        "min_positive_examples_per_partition = 0\n"
        "required_class_coverage = []\n"
        "required_classes = [0, 1]\n"
        "max_partition_ratio_deviation = 1.0\n"
        'default_severity = "warning"\n'
        'warning_checks = ["minimum_positive_examples", "required_class_coverage"]\n\n'
        "[record_subjects]\n"
        f"{record_subjects}\n"
    )


def build_acquisition_manifest(files: Sequence[SyntheticFile], record_ids: Sequence[str]) -> str:
    """Return a pre-seeded acquisition.json matching the trimmed dataset config.

    `acquire_dataset` validates every field here against the dataset config
    it is given; matching them exactly here is what makes it take the
    verify-and-reuse path (destination files already exist and validate)
    instead of calling the network fetcher.
    """
    by_name = {item.name: item for item in files}
    download_url = "https://physionet.org/files/mitdb/1.0.0/"
    manifest_files = [
        {
            "path": f"{record_id}.{extension}",
            "url": download_url + f"{record_id}.{extension}",
            "size_bytes": by_name[f"{record_id}.{extension}"].size_bytes,
            "sha256": by_name[f"{record_id}.{extension}"].sha256,
        }
        for record_id in record_ids
        for extension in ("atr", "dat", "hea")
    ]
    manifest = {
        "schema_version": 1,
        "dataset_slug": "mitdb",
        "dataset_version": "1.0.0",
        "source_url": "https://physionet.org/content/mitdb/1.0.0/",
        "download_url": download_url,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "files": manifest_files,
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _sha256_of(path: Path) -> SyntheticFile:
    """Calculate the SHA-256 digest of one local file.

    Used to build both the trimmed dataset config's expected_source_files and the
    pre-seeded acquisition manifest, so both must agree with the actual bytes written
    to disk for acquire_dataset's verify-and-reuse path to succeed.

    Args:
        path: Path to the synthetic WFDB companion file to hash.

    Returns:
        The file's name, size, and SHA-256 digest.
    """

    data = path.read_bytes()
    return SyntheticFile(path.name, len(data), hashlib.sha256(data).hexdigest())


def write_synthetic_record(directory: Path, record_id: str, *, seed: int) -> None:
    """Write one synthetic WFDB record with well-spaced normal/abnormal beats."""
    import numpy as np
    import wfdb

    n_samples = SAMPLE_RATE_HZ * RECORD_SECONDS
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, RECORD_SECONDS, n_samples, endpoint=False)
    base = 0.05 * np.sin(2 * np.pi * 1.2 * axis)
    signal = base + rng.normal(0.0, 0.01, n_samples)
    signals = np.column_stack((signal, signal * 0.6))
    wfdb.wrsamp(
        record_id,
        fs=SAMPLE_RATE_HZ,
        units=["mV", "mV"],
        sig_name=["MLII", "V5"],
        p_signal=signals,
        write_dir=str(directory),
        fmt=["16", "16"],
    )
    beat_positions = np.linspace(
        BEAT_MARGIN_SAMPLES, n_samples - BEAT_MARGIN_SAMPLES, len(BEAT_SYMBOLS), dtype=int
    )
    wfdb.wrann(
        record_id,
        "atr",
        sample=beat_positions,
        symbol=list(BEAT_SYMBOLS),
        write_dir=str(directory),
    )


def generate_synthetic_records(
    directory: Path, record_ids: Sequence[str]
) -> tuple[SyntheticFile, ...]:
    """Write every synthetic record into `directory` and return their digests."""
    # Vary the seed per record (100 + index) so records don't all generate identical
    # noise, while keeping generation fully deterministic across runs.
    for index, record_id in enumerate(record_ids):
        write_synthetic_record(directory, record_id, seed=100 + index)
    return tuple(
        _sha256_of(path)
        for path in sorted(directory.iterdir())
        if path.suffix in {".atr", ".dat", ".hea"}
    )


def seed_worktree(worktree: Path, record_ids: Sequence[str] = SYNTHETIC_RECORD_IDS) -> None:
    """Populate an isolated worktree copy with synthetic data and trimmed configs."""
    # Generate into a scratch staging directory first, then copy into the worktree's
    # real data/raw/ layout below, so a failure during generation never leaves a
    # partially populated raw-data directory inside the worktree itself.
    with tempfile.TemporaryDirectory(prefix="ecg-synthetic-records-") as staging_name:
        staging = Path(staging_name)
        files = generate_synthetic_records(staging, record_ids)

        raw_dir = worktree / "data" / "raw" / "mitdb" / "1.0.0"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Copy every generated WFDB companion file into the worktree's real raw-data
        # location, matching where acquire_dataset expects to find them.
        for item in files:
            shutil.copy2(staging / item.name, raw_dir / item.name)

    (worktree / "configs" / "mitdb-v1.0.0.toml").write_text(
        build_dataset_config_toml(files, record_ids), encoding="utf-8"
    )
    (worktree / "configs" / "splitting-v2.toml").write_text(
        build_split_config_toml(record_ids), encoding="utf-8"
    )

    manifest_dir = worktree / "artifacts" / "datasets" / "mitdb" / "1.0.0"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "acquisition.json").write_text(
        build_acquisition_manifest(files, record_ids), encoding="utf-8"
    )


def _run(
    command: Sequence[str], *, cwd: Path, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run one fixed subprocess command and translate execution failures.

    check=False is deliberate: callers inspect result.returncode themselves and
    raise NotebookValidationError with the captured stdout/stderr attached, giving a
    more actionable error than a bare CalledProcessError would.

    Args:
        command: The argv to execute.
        cwd: Working directory for the subprocess.
        timeout_seconds: Maximum time to wait before killing the subprocess.

    Returns:
        The completed process, including captured stdout/stderr and return code.
    """

    # A hung command (e.g. a notebook cell that never returns) would otherwise block
    # this script indefinitely; translate the timeout into the same
    # NotebookValidationError type every other failure mode in this module raises.
    try:
        # every call site below passes a fixed literal command list, not
        # runtime/user-constructed input.
        return subprocess.run(  # noqa: S603
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise NotebookValidationError(
            f"command timed out after {timeout_seconds}s: {' '.join(command)}"
        ) from error


def create_worktree(repository_root: Path, worktree: Path) -> None:
    """Create a detached-HEAD Git worktree, isolated from the real checkout.

    --detach avoids creating or checking out a branch, since this worktree is
    disposable scratch space for one validation run, not a location anyone should
    commit to.

    Args:
        repository_root: The real repository to create the worktree from.
        worktree: Path where the new worktree should be created.
    """

    result = _run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=repository_root,
        timeout_seconds=120,
    )
    # A failed worktree creation means every later step (seeding, notebook
    # execution) has nowhere to run, so it must fail immediately here.
    if result.returncode != 0:
        raise NotebookValidationError(f"could not create worktree: {result.stderr}")


def remove_worktree(repository_root: Path, worktree: Path) -> None:
    """Remove the isolated validation worktree and unregister its Git metadata.

    check=False and no return-code check is deliberate: this runs from a `finally`
    block (see validate_curated_notebooks) and must not raise over a cleanup failure
    while an earlier, more meaningful exception may already be propagating.

    Args:
        repository_root: The real repository the worktree was created from.
        worktree: Path to the worktree to remove.
    """

    # command is a fixed literal ("git", "worktree", "remove", "--force", <path>),
    # not runtime/user-constructed input.
    subprocess.run(  # noqa: S603
        ["git", "worktree", "remove", "--force", str(worktree)],  # noqa: S607
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )


def sync_notebook_environment(worktree: Path, *, timeout_seconds: float) -> None:
    """Install the notebooks and dev dependency groups inside the isolated worktree.

    The worktree is a separate checkout with its own .venv; it must run `uv sync`
    independently rather than reusing the real checkout's environment, since the
    curated notebooks import this package and its notebook-only dependencies.

    Args:
        worktree: The isolated worktree to sync dependencies into.
        timeout_seconds: Maximum time to wait for `uv sync` to complete.
    """

    result = _run(
        ["uv", "sync", "--group", "notebooks", "--group", "dev"],
        cwd=worktree,
        timeout_seconds=timeout_seconds,
    )
    # A failed sync means the worktree's environment isn't usable, so notebook
    # execution would fail anyway (likely with a confusing import error); fail here instead.
    if result.returncode != 0:
        raise NotebookValidationError(
            f"uv sync failed in isolated worktree:\n{result.stdout}\n{result.stderr}"
        )


# jupyter nbconvert's CLI sets the kernel's working directory to the notebook
# file's own directory (notebooks/), not the repository root the curated
# notebooks assume via Path.cwd(). Driving execution through nbclient
# directly, with resources["metadata"]["path"] set explicitly, is the
# documented way to control that; there is no equivalent nbconvert CLI flag.
_NOTEBOOK_RUNNER = """\
import sys
import nbformat
from nbclient import NotebookClient

notebook_path, working_directory, timeout_seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=timeout_seconds,
    resources={"metadata": {"path": working_directory}},
)
client.execute()
nbformat.write(notebook, notebook_path)
"""


def execute_notebook(
    worktree: Path, notebook_relative_path: str, *, timeout_seconds: float
) -> None:
    """Execute one curated notebook in place, inside the isolated worktree."""
    notebook_path = worktree / notebook_relative_path
    # A missing notebook means the caller passed a wrong relative path, or the
    # worktree wasn't seeded correctly; either way, fail with a specific path rather
    # than letting nbclient raise a less clear file-not-found error.
    if not notebook_path.is_file():
        raise NotebookValidationError(f"curated notebook not found: {notebook_path}")
    result = _run(
        [
            "uv",
            "run",
            "--group",
            "notebooks",
            "python3",
            "-c",
            _NOTEBOOK_RUNNER,
            str(notebook_path),
            str(worktree),
            str(int(timeout_seconds)),
        ],
        cwd=worktree,
        timeout_seconds=timeout_seconds + 60,
    )
    # The subprocess's own timeout is notebook_timeout_seconds (passed to
    # _NOTEBOOK_RUNNER as its nbclient execute() timeout); this outer _run timeout is
    # padded by 60s so the *process* has time to report a clean nbclient timeout
    # error before this script's own subprocess-level timeout would kill it instead.
    if result.returncode != 0:
        raise NotebookValidationError(
            f"{notebook_relative_path} failed to execute:\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def validate_curated_notebooks(
    repository_root: Path,
    *,
    notebooks: Sequence[str] = NOTEBOOKS,
    keep_worktree: bool = False,
    sync_timeout_seconds: float = 600,
    notebook_timeout_seconds: float = 600,
) -> Path:
    """Run every curated notebook end-to-end in an isolated, synthetic-seeded copy.

    Returns the worktree path (only meaningful when `keep_worktree` is True,
    for local debugging; the caller is responsible for removing it).
    """
    # The temp parent directory holds only the worktree path itself; keeping it
    # separate from the worktree's own removal (handled by `git worktree remove` in
    # the `finally` block below) avoids the two cleanup mechanisms conflicting.
    with tempfile.TemporaryDirectory(prefix="ecg-notebook-validation-") as parent_name:
        worktree = Path(parent_name) / "worktree"
        create_worktree(repository_root, worktree)
        # The worktree must always be removed (via remove_worktree in `finally`)
        # whether seeding, sync, or notebook execution succeeds or raises, so Git
        # never accumulates orphaned worktree registrations from failed runs.
        try:
            seed_worktree(worktree)
            sync_notebook_environment(worktree, timeout_seconds=sync_timeout_seconds)
            # Execute notebooks strictly in order (see NOTEBOOKS' own comment): 00
            # generates artifacts 01 and 02 depend on.
            for notebook_relative_path in notebooks:
                print(f"[validate-notebooks] executing {notebook_relative_path}...")
                execute_notebook(
                    worktree, notebook_relative_path, timeout_seconds=notebook_timeout_seconds
                )
                print(f"[validate-notebooks] {notebook_relative_path} passed.")
        finally:
            # --keep-worktree is an opt-in debugging aid: copy the worktree's final
            # state somewhere durable before it's removed below, so a failure can be
            # inspected after this script exits.
            if keep_worktree:
                kept = repository_root / ".notebook-validation-worktree"
                # Remove any previous kept copy first, since shutil.copytree refuses
                # to write into an already-existing destination directory.
                if kept.exists():
                    shutil.rmtree(kept)
                shutil.copytree(worktree, kept)
                print(f"[validate-notebooks] kept worktree for inspection: {kept}")
            remove_worktree(repository_root, worktree)
    return worktree


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the curated-notebook validation entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: repository root and keep-worktree flag.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to create the isolated worktree from (default: current directory)",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Copy the isolated worktree to .notebook-validation-worktree/ before removing it, "
        "for local debugging of a failure.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation straightforward
    to audit.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The value produced by the documented operation.
    """

    args = parse_args(argv)
    # Every failure mode across worktree creation, seeding, sync, and notebook
    # execution is collapsed into NotebookValidationError; catch it once here for
    # uniform error reporting.
    try:
        validate_curated_notebooks(args.repository_root.resolve(), keep_worktree=args.keep_worktree)
    except NotebookValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("[validate-notebooks] all curated notebooks executed successfully.")
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
