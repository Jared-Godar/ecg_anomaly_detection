# Changelog

This changelog records notable repository changes using a structure inspired by
Keep a Changelog. It does not claim formal compliance with that specification.

## Unreleased

### Added

- `ecg-data run-pipeline` now prints per-stage progress banners (start, completion,
  elapsed time, and key counts or artifact paths) for acquisition, inventory, record
  processing, split, split diagnostics, training, and validation evaluation, plus
  total elapsed run time, so long-running local runs no longer appear frozen.
- `ecg-data list-runs` and `ecg-data purge-run` list and reclaim disk space from
  local `run-pipeline` output by exact run ID, with a `--dry-run` preview, without
  touching the dataset acquisition baseline or any other run's directories.
- `ecg_anomaly_detection.experiment_tracking.ExperimentTracker` checkpoints
  long-running local experiment loops one candidate at a time, so an
  interruption loses at most the in-progress candidate. Supports resume via
  `is_completed()`, progress/ETA reporting, and a sorted `finalize()` summary
  once every candidate has completed.
- `ecg-data split-windows` and `ecg-data index-dataset` accept a directory as
  `--input`, expanded to its immediate `*.npz` files (sorted, non-recursive).
  Directory and file arguments can be mixed and repeated; an empty directory,
  missing path, symlink, or duplicate resolved file fails with a clear
  diagnostic instead of a confusing downstream error.

### Changed

### Fixed

- Pipeline progress output is now flushed per line so subprocess consumers
  (including the Step 0 notebook) receive it live instead of buffered until
  process exit, which is the default for non-TTY stdout in Python.

### Removed

### Governance

- Defined release, versioning, and release-review policies for this engineering
  portfolio repository.
- Added an automated pull-request metadata gate (`.github/workflows/metadata-governance.yml`,
  `scripts/github/validate_project_metadata.py`) that validates PR assignee, milestone,
  `type:*`/`area:*` labels, and closing issue reference, plus the linked issue's Project #5
  membership and required-field completeness, enforced via a repository-scoped
  `PROJECT_METADATA_TOKEN` secret with read-only Projects access.

### Documentation

- Added the initial changelog and release governance documentation.
- Completed the historical archive attribution and provenance audit for
  `archive/original_2022/images/`, adding `archive/original_2022/ATTRIBUTION.md`
  and `PROVENANCE.md` (retroactive entry for #59, missed when that PR merged).

### Security

## 0.1.0

`0.1.0` is the current initial package version. No Git tag or GitHub release is
associated with this version, and no historical release date is asserted.
