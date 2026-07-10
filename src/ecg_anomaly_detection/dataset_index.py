"""Model-ready partition index over immutable per-record window shards."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.splitting import PartitionSummary, SplitManifest, read_split_manifest

# Centralize BUFFER_SIZE so every caller shares the same documented invariant.
BUFFER_SIZE = 1024 * 1024


class DatasetIndexError(ValueError):
    """Raised when model-ready shard evidence violates its contract."""


@dataclass(frozen=True, slots=True)
class IndexedFile:
    """Repository-relative file identity and digest."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """One immutable record shard assigned to exactly one partition."""

    record_id: str
    subject_id: str
    window_count: int
    target_value_counts: dict[str, int]
    file: IndexedFile


@dataclass(frozen=True, slots=True)
class PartitionIndex:
    """Ordered shard membership and aggregate counts for one partition."""

    subject_ids: tuple[str, ...]
    subject_count: int
    record_count: int
    window_count: int
    target_value_counts: dict[str, int]
    shards: tuple[ShardIndex, ...]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    """Machine-readable model input contract without duplicating window arrays."""

    schema_version: int
    split_name: str
    split_version: str
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    sample_rate_hz: float
    channel_index: int
    channel_name: str
    window_samples: int
    total_subject_count: int
    total_record_count: int
    total_window_count: int
    split_manifest: IndexedFile
    partitions: dict[str, PartitionIndex]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _InspectedShard:
    """Hold validated identity, counts, and digest evidence for one immutable shard.

    This internal record separates potentially unsafe file inspection from deterministic index
    assembly after every shard has passed its schema and lineage checks.
    """

    index: ShardIndex
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    sample_rate_hz: float
    channel_index: int
    channel_name: str
    channel_selector: str | None
    configured_channel_index: int | None
    configured_channel_name: str | None
    window_samples: int


def create_dataset_index(
    repository_root: Path,
    split_manifest_path: Path,
    window_artifact_paths: Sequence[Path],
) -> DatasetIndex:
    """Validate record shards and index them according to grouped split membership."""
    root = repository_root.resolve()
    # A pyproject.toml at the root is the cheapest available signal that this is really
    # the repository root and not an arbitrary directory, before any path-boundary checks
    # below trust it as the containment root for shard and manifest paths.
    if not (root / "pyproject.toml").is_file():
        raise DatasetIndexError(f"repository root does not contain pyproject.toml: {root}")
    split_path = _resolve_file(root, split_manifest_path, ("artifacts",), "split manifest")
    # read_split_manifest already validates leakage-freedom and count consistency; only
    # translate its ValueError into this module's own exception type so callers only need
    # to catch DatasetIndexError for every failure mode in this stage.
    try:
        split_manifest = read_split_manifest(split_path)
    except ValueError as error:
        raise DatasetIndexError(f"invalid split manifest: {error}") from error
    # An index built from zero shards would otherwise silently produce empty partitions.
    if not window_artifact_paths:
        raise DatasetIndexError("at least one window artifact is required")

    inspected = [_inspect_shard(root, path) for path in window_artifact_paths]
    by_record: dict[str, _InspectedShard] = {}
    # Index every inspected shard by its record ID, checking uniqueness as they're added.
    for shard in inspected:
        # Each shard is expected to be exactly one record's windows; two shards claiming
        # the same record would make partition assignment for that record ambiguous.
        if shard.index.record_id in by_record:
            raise DatasetIndexError(
                f"record occurs in multiple model-ready shards: {shard.index.record_id}"
            )
        by_record[shard.index.record_id] = shard

    expected_records = {
        record_id
        for partition in split_manifest.partitions.values()
        for record_id in partition.record_ids
    }
    # The shard set on disk and the split manifest's record membership must agree
    # exactly; a mismatch usually means shards were built from a different windowing
    # run than the split, which would silently misassign records to partitions.
    if set(by_record) != expected_records:
        raise DatasetIndexError(
            "window shard records do not match split membership; "
            f"missing={sorted(expected_records - set(by_record))}, "
            f"extra={sorted(set(by_record) - expected_records)}"
        )
    identity = _validate_shard_identity(inspected, split_manifest)
    partitions = {
        name: _build_partition_index(summary, by_record)
        for name, summary in split_manifest.partitions.items()
    }
    # Cross-check every built partition against the manifest it was built from.
    for name, partition in partitions.items():
        expected = split_manifest.partitions[name]
        # Recompute every count from the assembled shards and compare against the
        # manifest's own numbers -- this is a self-consistency check that the index
        # faithfully reflects the split it claims to index, not a fresh business rule.
        if (
            partition.subject_count != expected.subject_count
            or partition.subject_ids != expected.subject_ids
            or partition.record_count != expected.record_count
            or partition.window_count != expected.window_count
            or partition.target_value_counts != expected.target_value_counts
        ):
            raise DatasetIndexError(f"indexed partition counts do not match split manifest: {name}")
    return DatasetIndex(
        schema_version=2,
        split_name=split_manifest.split_name,
        split_version=split_manifest.split_version,
        mapping_name=identity.mapping_name,
        mapping_version=identity.mapping_version,
        window_config_name=identity.window_config_name,
        window_config_version=identity.window_config_version,
        sample_rate_hz=identity.sample_rate_hz,
        channel_index=identity.channel_index,
        channel_name=identity.channel_name,
        window_samples=identity.window_samples,
        total_subject_count=split_manifest.total_subject_count,
        total_record_count=split_manifest.total_record_count,
        total_window_count=split_manifest.total_window_count,
        split_manifest=_indexed_file(root, split_path),
        partitions=partitions,
    )


def write_dataset_index(index: DatasetIndex, repository_root: Path, output_path: Path) -> None:
    """Write a model-ready dataset index under the ignored processed data zone."""
    root = repository_root.resolve()
    candidate = output_path if output_path.is_absolute() else root / output_path
    # Reject a symlink before resolving it: resolving would silently follow the link and
    # validate/write to wherever it points, defeating the repository-root containment
    # check just below.
    if candidate.is_symlink():
        raise DatasetIndexError("dataset index output must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root (e.g. via `..` segments);
    # translate that into this module's own exception so the failure mode is uniform.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise DatasetIndexError("dataset index output must stay within repository root") from error
    # Pin the output location to data/processed/ specifically, matching this repo's
    # documented directory contract for generated, gitignored pipeline artifacts.
    if len(relative.parts) < 2 or relative.parts[:2] != ("data", "processed"):
        raise DatasetIndexError("dataset index output must be written under data/processed/")
    # Enforce the extension so downstream tooling that globs for *.json indexes can rely
    # on finding this file without a separate content sniff.
    if resolved.suffix != ".json":
        raise DatasetIndexError("dataset index output must use the .json extension")
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from the open() call below.
    if not resolved.parent.is_dir():
        raise DatasetIndexError(f"dataset index parent directory does not exist: {resolved.parent}")
    # Open with mode "x" (exclusive create) rather than "w" so an index is never silently
    # overwritten -- each run_id's artifacts are meant to be immutable once written.
    try:
        # The `with` block ensures the file handle closes even if to_json() or the write
        # itself raises partway through.
        with resolved.open("x", encoding="utf-8") as output:
            output.write(index.to_json())
    except FileExistsError as error:
        raise DatasetIndexError(f"dataset index already exists: {resolved}") from error


def _inspect_shard(repository_root: Path, path: Path) -> _InspectedShard:
    """Load, validate, and summarize one window-artifact NPZ file as a candidate shard.

    This is the per-file half of dataset indexing: it trusts nothing about the artifact's
    contents until every required field has been type- and shape-checked, since these
    files may originate from an untrusted or externally shared pipeline run. Cross-shard
    checks (record uniqueness, identity agreement) happen afterward in create_dataset_index.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        path: Path to the window-artifact NPZ file to inspect.

    Returns:
        A validated _InspectedShard combining its index entry with identity metadata.
    """

    resolved = _resolve_file(
        repository_root,
        path,
        ("data", "interim"),
        "window artifact",
    )
    # allow_pickle=False is a security boundary against arbitrary code execution from an
    # untrusted NPZ file; collapse every load/parse failure mode into DatasetIndexError so
    # callers don't need to know numpy's or the filesystem's own exception types.
    try:
        # The `with` block ensures the lazy NpzFile handle closes even if a required field
        # is missing or malformed partway through parsing.
        with np.load(resolved, allow_pickle=False) as artifact:
            required = {
                "schema_version",
                "windows",
                "record_ids",
                "center_sample_indices",
                "source_symbols",
                "target_values",
                "sample_rate_hz",
                "channel_index",
                "channel_name",
                "mapping_name",
                "mapping_version",
                "window_config_name",
                "window_config_version",
            }
            missing = required - set(artifact.files)
            # Check the whole required-field set up front so one missing field reports
            # clearly, instead of a bare KeyError from whichever access happens first.
            if missing:
                raise DatasetIndexError(f"window artifact {resolved} is missing: {sorted(missing)}")
            # schema_version pins the NPZ layout produced by window extraction; a mismatch
            # means this artifact predates or postdates what this indexer understands.
            if _integer_scalar(artifact["schema_version"], "schema_version") != 1:
                raise DatasetIndexError("window artifact must use schema_version 1")
            windows = np.asarray(artifact["windows"])
            record_ids = _string_vector(artifact["record_ids"], "record_ids")
            center_indices = _integer_vector(
                artifact["center_sample_indices"], "center_sample_indices"
            )
            source_symbols = _string_vector(artifact["source_symbols"], "source_symbols")
            target_values = _integer_vector(artifact["target_values"], "target_values")
            sample_rate_hz = _positive_float_scalar(artifact["sample_rate_hz"], "sample_rate_hz")
            channel_index = _nonnegative_integer_scalar(artifact["channel_index"], "channel_index")
            channel_name = _string_scalar(artifact["channel_name"], "channel_name")
            channel_selector = _optional_artifact_string_scalar(artifact, "channel_selector")
            # channel_selector records how the channel was chosen during extraction
            # (by index or by name); any other value means the artifact wasn't produced
            # by the documented window-extraction stage.
            if channel_selector is not None and channel_selector not in {
                "channel_index",
                "channel_name",
            }:
                raise DatasetIndexError("channel_selector must be channel_index or channel_name")
            configured_channel_index = _optional_artifact_integer_scalar(
                artifact, "configured_channel_index"
            )
            # -1 is the writer's sentinel for "not configured by index" (NPZ has no native
            # optional-int type); normalize it to None so downstream code has one absent
            # representation instead of two.
            if configured_channel_index == -1:
                configured_channel_index = None
            # A present, non-sentinel value must still be a valid channel index.
            if configured_channel_index is not None and configured_channel_index < 0:
                raise DatasetIndexError(
                    "configured_channel_index must be nonnegative or -1 when absent"
                )
            configured_channel_name = _optional_artifact_string_scalar(
                artifact, "configured_channel_name", allow_empty=True
            )
            # Same normalization as configured_channel_index above, but using "" as the
            # writer's sentinel since this field is a string rather than an int.
            if configured_channel_name == "":
                configured_channel_name = None
            mapping_name = _string_scalar(artifact["mapping_name"], "mapping_name")
            mapping_version = _string_scalar(artifact["mapping_version"], "mapping_version")
            window_config_name = _string_scalar(
                artifact["window_config_name"], "window_config_name"
            )
            window_config_version = _string_scalar(
                artifact["window_config_version"], "window_config_version"
            )
    except DatasetIndexError:
        # Re-raise our own field-level diagnostics untouched so they aren't swallowed by
        # the broader numpy/OSError handler below.
        raise
    except (BadZipFile, OSError, ValueError) as error:
        raise DatasetIndexError(f"could not inspect window artifact {resolved}: {error}") from error

    row_count = len(record_ids)
    # A shard is defined as one record's windows; zero rows or more than one distinct
    # record ID both violate that contract before any other field is trusted.
    if row_count == 0 or len(set(record_ids)) != 1:
        raise DatasetIndexError("each model-ready shard must contain one non-empty record")
    # The window matrix must be a finite, non-degenerate 2-D float array with one row per
    # window; a non-finite value (NaN/inf) would silently corrupt any model trained on it.
    if (
        windows.ndim != 2
        or windows.shape[0] != row_count
        or windows.shape[1] == 0
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise DatasetIndexError("window matrix must be finite floating-point rows")
    # These three lineage/label arrays are parallel to the windows matrix (one entry per
    # row); an unequal length means the artifact was corrupted or half-written.
    if not (len(center_indices) == len(source_symbols) == len(target_values) == row_count):
        raise DatasetIndexError("window shard lineage arrays must have equal row counts")
    # Center indices are sample offsets into the source record and are expected to be
    # monotonically non-decreasing, matching the order windows were extracted in --
    # a negative or out-of-order index would indicate a corrupted or hand-edited artifact.
    if np.any(center_indices < 0) or np.any(np.diff(center_indices) < 0):
        raise DatasetIndexError("window center indices must be nonnegative and ordered")
    counts = Counter(int(value) for value in target_values)
    return _InspectedShard(
        index=ShardIndex(
            record_id=record_ids[0],
            subject_id="",  # populated from the validated split manifest
            window_count=row_count,
            target_value_counts={str(value): counts[value] for value in sorted(counts)},
            file=_indexed_file(repository_root, resolved),
        ),
        mapping_name=mapping_name,
        mapping_version=mapping_version,
        window_config_name=window_config_name,
        window_config_version=window_config_version,
        sample_rate_hz=sample_rate_hz,
        channel_index=channel_index,
        channel_name=channel_name,
        channel_selector=channel_selector,
        configured_channel_index=configured_channel_index,
        configured_channel_name=configured_channel_name,
        window_samples=windows.shape[1],
    )


def _optional_artifact_string_scalar(
    artifact: Any, name: str, *, allow_empty: bool = False
) -> str | None:
    """Read one optional scalar string field from an NPZ artifact, or None if absent.

    Some identity fields (channel_selector, configured_channel_name) were added after
    the original window-extraction schema and may be missing from older artifacts;
    treating absence as None (rather than requiring the field) keeps this indexer
    compatible with both old and new artifacts.

    Args:
        artifact: The open NpzFile to read from.
        name: The field name to look up.
        allow_empty: Whether an empty string is an acceptable present value.

    Returns:
        The field's string value, or None if the field is absent from this artifact.
    """

    # Absence is a valid, expected state for these backward-compatibility fields.
    if name not in artifact.files:
        return None
    value = artifact[name]
    # A present-but-non-scalar value means the field exists but wasn't written with the
    # expected 0-D shape, which points at a schema mismatch rather than absence.
    if np.asarray(value).shape != ():
        raise DatasetIndexError(f"{name} must be a scalar string")
    scalar = value.item()
    # numpy's .item() can return a non-str Python object for some dtypes; confirm the
    # unwrapped value is actually a string before returning it as one.
    if not isinstance(scalar, str):
        raise DatasetIndexError(f"{name} must be a scalar string")
    # allow_empty distinguishes fields where "" is a legitimate configured value
    # (configured_channel_name, when selection was by index instead) from fields where
    # emptiness would itself be invalid.
    if not allow_empty and not scalar:
        raise DatasetIndexError(f"{name} must be a non-empty scalar string")
    return scalar


def _optional_artifact_integer_scalar(artifact: Any, name: str) -> int | None:
    """Read one optional scalar integer field from an NPZ artifact, or None if absent.

    Mirrors _optional_artifact_string_scalar for integer-typed optional fields
    (currently just configured_channel_index).

    Args:
        artifact: The open NpzFile to read from.
        name: The field name to look up.

    Returns:
        The field's integer value, or None if the field is absent from this artifact.
    """

    # Absence is a valid, expected state for this backward-compatibility field.
    if name not in artifact.files:
        return None
    return _integer_scalar(artifact[name], name)


def _validate_shard_identity(
    shards: Sequence[_InspectedShard], split_manifest: SplitManifest
) -> _InspectedShard:
    """Confirm every shard shares one geometry/configuration identity, matching the split.

    A DatasetIndex reports one mapping/window-config/channel identity for the whole
    dataset (see DatasetIndex's fields), which is only valid if every shard actually
    agrees. This also cross-checks that identity against the split manifest's own
    mapping/window-config fields, since a mismatch there would mean the shards and the
    split were produced by different, incompatible upstream runs.

    Args:
        shards: Every inspected shard that will be indexed.
        split_manifest: The split manifest these shards are being indexed against.

    Returns:
        The first shard, used as the representative identity for the whole dataset.
    """

    first = shards[0]
    identity = (
        first.mapping_name,
        first.mapping_version,
        first.window_config_name,
        first.window_config_version,
        first.sample_rate_hz,
        first.channel_name,
        first.window_samples,
    )
    # Compare every other shard's identity tuple against the first shard's; a channel
    # mismatch in particular (e.g. extraction fell back to a different lead for some
    # records) would silently mix incompatible signal geometries into one dataset.
    if any(
        (
            shard.mapping_name,
            shard.mapping_version,
            shard.window_config_name,
            shard.window_config_version,
            shard.sample_rate_hz,
            shard.channel_name,
            shard.window_samples,
        )
        != identity
        for shard in shards[1:]
    ):
        raise DatasetIndexError(_format_shard_identity_mismatch(shards))
    # The shards' agreed-upon mapping/window-config identity must also match what the
    # split manifest itself declares, confirming the split and the shards trace back to
    # the same annotation-mapping and window-extraction run.
    if identity[:4] != (
        split_manifest.mapping_name,
        split_manifest.mapping_version,
        split_manifest.window_config_name,
        split_manifest.window_config_version,
    ):
        raise DatasetIndexError("window shard identity does not match split manifest")
    return first


def _format_shard_identity_mismatch(shards: Sequence[_InspectedShard]) -> str:
    """Build a diagnostic message explaining which records have inconsistent channels.

    Channel mismatches are the most common real-world cause of shard-identity failures
    (some MIT-BIH records don't carry the same lead), so this renders a report naming the
    specific affected records and any configured channel selection, rather than the bare
    "identity mismatch" a generic comparison would produce -- letting a reviewer act on
    it without re-deriving which records are the problem by hand.

    Args:
        shards: Every inspected shard whose identities disagreed.

    Returns:
        A multi-line human-readable diagnostic for the identity mismatch.
    """

    channel_counts = Counter(shard.channel_name for shard in shards)
    # If every shard actually shares one channel name, the mismatch must be in some
    # other identity field (mapping/window-config/sample-rate), which this
    # channel-focused report can't usefully explain further.
    if len(channel_counts) <= 1:
        return "window shards do not share one geometry and configuration identity"

    lines = ["Window extraction produced inconsistent channel identities."]
    configured_selection = _format_configured_channel_selection(shards)
    # Only append the configuration line when shards agree on one, since disagreement
    # means there's no single configured value worth reporting.
    if configured_selection is not None:
        lines.append(configured_selection)

    lines.append("Observed channel identities:")
    # List every distinct channel name observed and how many records used it.
    for channel_name, count in sorted(channel_counts.items()):
        record_word = "record" if count == 1 else "records"
        lines.append(f"  - {channel_name}: {count} {record_word}")

    first_channel = shards[0].channel_name
    affected = [
        shard
        for shard in sorted(shards, key=lambda item: item.index.record_id)
        if shard.channel_name != first_channel
    ]
    # Only list records whose channel differs from the first shard's, so the report
    # focuses a reviewer on the minority case rather than repeating the majority.
    if affected:
        lines.append("Affected records:")
        # Name each specific record and its channel so a reviewer can act without
        # re-deriving the affected set themselves.
        for shard in affected:
            lines.append(f"  - {shard.index.record_id}: {shard.channel_name}")

    lines.append(
        "This is not usually fixed by cleaning local artifacts. Configure extraction "
        "by channel name or explicitly exclude/fallback records that cannot satisfy "
        "the channel contract."
    )
    return "\n".join(lines)


def _format_configured_channel_selection(shards: Sequence[_InspectedShard]) -> str | None:
    """Describe the shared channel-selection configuration, if every shard agrees on one.

    Surfaces *how* extraction was told to pick a channel (by index or by name) as part of
    the mismatch report in _format_shard_identity_mismatch, so a reviewer can tell whether
    the configuration itself was ambiguous versus the source records genuinely differing.

    Args:
        shards: Every inspected shard to check for a shared selection configuration.

    Returns:
        A one-line description of the shared configuration, or None if shards disagree
        on their selection method/value (in which case there's nothing single to report).
    """

    selections = {
        (
            shard.channel_selector,
            shard.configured_channel_index,
            shard.configured_channel_name,
        )
        for shard in shards
    }
    # More than one distinct (selector, index, name) combination means the shards were
    # extracted under different configurations -- no single summary line applies.
    if len(selections) != 1:
        return None

    selector, configured_index, configured_name = next(iter(selections))
    # Only report a value for the selector method that was actually used.
    if selector == "channel_index" and configured_index is not None:
        return f"Configured channel selection: channel_index = {configured_index}"
    # Same guard as above, for the by-name selector instead of by-index.
    if selector == "channel_name" and configured_name is not None:
        return f'Configured channel selection: channel_name = "{configured_name}"'
    return None


def _build_partition_index(
    summary: PartitionSummary,
    by_record: dict[str, _InspectedShard],
) -> PartitionIndex:
    """Assemble one partition's ShardIndex entries from the split's record membership.

    This is where subject/record membership (from the split manifest) is joined against
    each record's actual inspected shard, producing the lazily-loadable index entries
    that training/evaluation later read instead of concatenating full window arrays.

    Args:
        summary: This partition's record/subject membership from the split manifest.
        by_record: Every inspected shard, keyed by record ID.

    Returns:
        The assembled PartitionIndex for this partition.
    """

    shards = tuple(
        ShardIndex(
            record_id=by_record[record_id].index.record_id,
            subject_id=summary.record_subjects[record_id],
            window_count=by_record[record_id].index.window_count,
            target_value_counts=by_record[record_id].index.target_value_counts,
            file=by_record[record_id].index.file,
        )
        for record_id in summary.record_ids
    )
    target_counts = {
        target: sum(shard.target_value_counts.get(target, 0) for shard in shards)
        for target in summary.target_value_counts
    }
    return PartitionIndex(
        subject_ids=summary.subject_ids,
        subject_count=summary.subject_count,
        record_count=len(shards),
        window_count=sum(shard.window_count for shard in shards),
        target_value_counts=target_counts,
        shards=shards,
    )


def _resolve_file(
    repository_root: Path,
    path: Path,
    required_prefix: tuple[str, ...],
    description: str,
) -> Path:
    """Resolve a path and enforce it stays within a required subdirectory of the repo.

    Shared by every stage in this module that reads a repository-relative input (split
    manifests under artifacts/, window artifacts under data/interim/): resolving through
    symlinks and checking containment prevents a maliciously or accidentally crafted path
    from reading files outside the repository the pipeline is meant to operate on.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        path: The candidate path, absolute or relative to repository_root.
        required_prefix: The path segments the resolved file must be nested under
            (e.g. ("data", "interim")).
        description: Human-readable label for this file, used in error messages.

    Returns:
        The resolved, validated absolute path.
    """

    candidate = path if path.is_absolute() else repository_root / path
    # Reject a symlink before resolving it, so a link that points outside the required
    # prefix can't be validated against a resolved target it doesn't actually name.
    if candidate.is_symlink():
        raise DatasetIndexError(f"{description} must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes repository_root (e.g. via `..`
    # segments); translate that into this module's own exception type.
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise DatasetIndexError(f"{description} must stay within repository root") from error
    # Confirm the path is nested under the expected subtree (artifacts/, data/interim/,
    # etc.) so this loader can't be pointed at an arbitrary repository file.
    if relative.parts[: len(required_prefix)] != required_prefix:
        required = "/".join(required_prefix)
        raise DatasetIndexError(f"{description} must be stored under {required}/")
    # A directory or special file at this path would fail np.load/read_text downstream
    # with a less specific error; check it's a regular file up front.
    if not resolved.is_file():
        raise DatasetIndexError(f"{description} must be a regular file: {resolved}")
    return resolved


def _indexed_file(repository_root: Path, path: Path) -> IndexedFile:
    """Hash and size one file, recording it as a repository-relative IndexedFile.

    The dataset index stores a digest and size for every shard/manifest file it
    references rather than embedding their contents, so the index stays small while
    still letting a consumer verify a shard on disk hasn't changed since indexing.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        path: The already-resolved, validated file path to hash.

    Returns:
        The file's repository-relative path, size, and SHA-256 digest.
    """

    digest = hashlib.sha256()
    size_bytes = 0
    # Read in fixed-size chunks rather than the whole file at once, since window
    # artifacts can be large and this keeps peak memory bounded during hashing.
    with path.open("rb") as source:
        # The walrus operator lets both the read and the loop's termination condition
        # (an empty final chunk) live in one line without a separate `break`.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return IndexedFile(
        path=path.relative_to(repository_root).as_posix(),
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


def _string_vector(value: np.ndarray[Any, Any], field: str) -> tuple[str, ...]:
    """Validate and convert one 1-D string field from a window NPZ artifact.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's values as a tuple of Python strings.
    """

    array = np.asarray(value)
    # dtype.kind 'U'/'S' cover numpy's Unicode and byte-string dtypes; a 1-D shape is
    # required since this always represents one value per window row.
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise DatasetIndexError(f"{field} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    # record_ids in particular must never contain an empty string, since it would be
    # indistinguishable from a missing lineage entry downstream.
    if any(not item for item in result):
        raise DatasetIndexError(f"{field} contains an empty string")
    return result


def _integer_vector(value: np.ndarray[Any, Any], field: str) -> np.ndarray[Any, Any]:
    """Validate one 1-D integer field from a window NPZ artifact.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's values as a numpy array (dtype preserved, unlike _integer_vector
        in splitting.py, since callers here need the original int width for arithmetic
        like np.diff on center_sample_indices).
    """

    array = np.asarray(value)
    # dtype.kind 'i'/'u' cover numpy's signed and unsigned integer dtypes.
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise DatasetIndexError(f"{field} must be an integer vector")
    return array


def _string_scalar(value: np.ndarray[Any, Any], field: str) -> str:
    """Validate and convert one 0-D (scalar) string field from a window NPZ artifact.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's value as a Python string.
    """

    array = np.asarray(value)
    # 0-D shape distinguishes a scalar identity field from a per-row string vector.
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise DatasetIndexError(f"{field} must be a string scalar")
    result = str(array.item())
    # An empty identity field (e.g. mapping_name="") would make the shard-identity
    # agreement check in _validate_shard_identity meaningless.
    if not result:
        raise DatasetIndexError(f"{field} must not be empty")
    return result


def _integer_scalar(value: np.ndarray[Any, Any], field: str) -> int:
    """Validate and convert one 0-D (scalar) integer field from a window NPZ artifact.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's value as a Python int.
    """

    array = np.asarray(value)
    # 0-D shape distinguishes a scalar identity field from a per-row integer vector.
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise DatasetIndexError(f"{field} must be an integer scalar")
    return int(array.item())


def _nonnegative_integer_scalar(value: np.ndarray[Any, Any], field: str) -> int:
    """Validate one scalar integer field from a window NPZ artifact as nonnegative.

    Used for channel_index specifically, where a negative value would mean the writer
    encoded "no channel selected" using a sentinel this indexer doesn't recognize.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's value as a Python int.
    """

    result = _integer_scalar(value, field)
    # channel_index is an array position and can never be negative.
    if result < 0:
        raise DatasetIndexError(f"{field} must be nonnegative")
    return result


def _positive_float_scalar(value: np.ndarray[Any, Any], field: str) -> float:
    """Validate one scalar numeric field from a window NPZ artifact as finite and positive.

    Used for sample_rate_hz; accepts int or unsigned dtypes in addition to float since a
    sample rate stored as a whole-number NPZ field is still a legitimate positive value.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's value as a Python float.
    """

    array = np.asarray(value)
    # dtype.kind 'f'/'i'/'u' cover float, signed-int, and unsigned-int dtypes.
    if array.ndim != 0 or array.dtype.kind not in {"f", "i", "u"}:
        raise DatasetIndexError(f"{field} must be a numeric scalar")
    result = float(array.item())
    # A sample rate of zero, negative, NaN, or infinite would make every downstream
    # time-based calculation (window durations, center-index-to-seconds) meaningless.
    if not np.isfinite(result) or result <= 0:
        raise DatasetIndexError(f"{field} must be finite and positive")
    return result
