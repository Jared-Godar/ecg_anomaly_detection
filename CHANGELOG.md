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
- `.github/workflows/quality.yml` adds a `package-build` job that runs
  `uv build` on every pull request as a build-only assurance check (wheel and
  source distribution), confirms the result is not committed to Git, and
  never uploads or publishes anywhere. `pyproject.toml` scopes the source
  distribution to `src/`, `README.md`, `CHANGELOG.md`, `NOTICE.md`, and
  `LICENSE`; hatchling's unconfigured default previously bundled the entire
  git-tracked tree, including `archive/original_2022/`'s historical images
  and notebooks (2.6 MB, 170 files, down to 72 KB, 29 files).

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
- Added `scripts/detect_label_drift.py` and a weekly `.github/workflows/repository-hygiene.yml`
  run that reports labels applied to open issues/PRs that are not declared in
  `.github/labels.json`. Read-only; never relabels anything. Stale issue/PR bot automation
  was considered and explicitly declined as not justified by this repository's actual
  activity — see `docs/governance/repository-hygiene.md`.
- Fixed the pull-request metadata gate's milestone check
  (`scripts/github/validate_project_metadata.py`), which previously required every pull
  request to carry a milestone unconditionally. It now inherits the requirement from the
  issue(s) the pull request closes, exempting a pull request only when every closing issue
  is itself deliberately unmilestoned, per the existing "milestone is a delivery commitment,
  not a mandatory tag" policy.

### Documentation

- Added the initial changelog and release governance documentation.
- Completed the historical archive attribution and provenance audit for
  `archive/original_2022/images/`, adding `archive/original_2022/ATTRIBUTION.md`
  and `PROVENANCE.md` (retroactive entry for #59, missed when that PR merged).
- Fixed `README.md`'s "Current status" table, which listed subject-grouped
  guarantees across paired records (e.g. 201/202, sharing one source tape)
  as not yet implemented; split schema v2 has enforced this since its
  introduction (see `docs/record-grouped-splitting.md`).
- Completed the `archive/original_2022/wrangle.py` tutorial-code adaptation
  audit (`archive/original_2022/PROVENANCE.md`'s new "Code provenance
  evidence" section), retrieving the cited article's linked source
  repository and comparing it against `wrangle.py` line by line. Its
  `load_ecg`, `make_dataset`, and `build_XY` functions and parameter lists
  are directly adapted from that source; its `split_my_data` function is
  not — the source splits by patient identity, while `split_my_data` uses
  an ordinary beat-level split, independent of the cited approach. No
  archived file was modified.
- Resolved `README.md`'s ambiguous "Not yet implemented" listing for cloud
  deployment/orchestration and runtime/resource benchmarks: both are
  permanent, by-design scope exclusions for this local portfolio case
  study, not pending work, per `docs/pipeline-design.md`'s existing
  "Proposed cloud mapping" framing and `docs/reproducibility-evidence.md`'s
  existing host-variance disclosure. Removed both from the "pending" table
  column and added a short explanatory note instead. Also removed the
  stale "Historical tutorial code adaptation-extent audit" row, resolved
  by the prior entry above but never removed from this table at the time.
- Scoped `MODEL_CARD.md`'s bundled "no threshold analysis, ROC/AUC,
  calibration analysis" limitation into two distinct dispositions.
  ROC/AUC and calibration analysis are confirmed permanently out of
  scope: the supported estimator predicts a hard class by nearest-centroid
  assignment and exposes no ranked score or predicted probability for
  either to evaluate against, so adding one would be a new modeling
  choice, not an evaluation-reporting addition. Threshold-based decision
  analysis over the existing per-window centroid-distance margin, and
  generated figures, are identified as a candidate follow-up (not created
  as an issue without further review) since that margin is already
  computed internally and reporting a sweep over it doesn't require any
  new modeling choice. Also fixed two more copies of the already-stale
  "tutorial code adaptation extent... unaudited"/"remains under review"
  claim (`MODEL_CARD.md`, `README.md`'s limitations list) missed by the
  #74 doc sweep, which only caught the "Current status" table's copy.

### Security

## 0.1.0

`0.1.0` is the current initial package version. No Git tag or GitHub release is
associated with this version, and no historical release date is asserted.
