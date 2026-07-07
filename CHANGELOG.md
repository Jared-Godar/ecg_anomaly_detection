# Changelog

This changelog records notable repository changes using a structure inspired by
Keep a Changelog. It does not claim formal compliance with that specification.

## Unreleased

### Added

- `ecg-data run-pipeline` now prints per-stage progress banners (start, completion,
  elapsed time, and key counts or artifact paths) for acquisition, inventory, record
  processing, split, split diagnostics, training, and validation evaluation, plus
  total elapsed run time, so long-running local runs no longer appear frozen.

### Changed

### Fixed

- Pipeline progress output is now flushed per line so subprocess consumers
  (including the Step 0 notebook) receive it live instead of buffered until
  process exit, which is the default for non-TTY stdout in Python.

### Removed

### Governance

- Defined release, versioning, and release-review policies for this engineering
  portfolio repository.

### Documentation

- Added the initial changelog and release governance documentation.

### Security

## 0.1.0

`0.1.0` is the current initial package version. No Git tag or GitHub release is
associated with this version, and no historical release date is asserted.
