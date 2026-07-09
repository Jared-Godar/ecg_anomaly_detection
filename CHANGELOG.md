# Changelog

![Technical banner for the changelog, showing version history and governance motifs.](docs/assets/ecg-changelog-banner.png)

This changelog records notable repository changes using a structure inspired by
Keep a Changelog. It does not claim formal compliance with that specification.

## Unreleased

### Added

- Added CI status, license, and Python-version badges to `README.md` (`Quality gates` workflow
  badge, MIT license, Python 3.12/3.13), confirmed green on `main` before badging (#94).
- Added `src/ecg_anomaly_detection/benchmark_approval.py` (`ApprovalInput`, `ApprovalRecord`,
  `BenchmarkApprovalError`, `record_benchmark_approval()`) and the `ecg-data
  record-benchmark-approval` CLI subcommand, implementing `docs/benchmark-governance.md`'s
  eligibility, approval-recording, and lineage-verification steps as a fail-closed gate that
  reuses `load_benchmark_policy()` and a `RunManifest` for lineage identity. Also adds
  `run_manifest.read_run_manifest()` as a JSON-to-dataclass counterpart to the existing
  `write_run_manifest()`. Covers governance steps 1-2 (approval recording and lineage
  verification) only; never opens, reads, or scores the protected `test` partition anywhere in
  code or tests, and `evaluation.py`'s `SUPPORTED_PARTITION` is unchanged. The separately
  reviewed execution step remains tracked by #73; a previously scoped disabled-by-default
  execution-gating config and CI stretch item was cut to keep this change reviewable and filed
  as #126 (#72).
- Added `configs/evaluation-heldout.toml` and `src/ecg_anomaly_detection/held_out_config.py`
  (`HeldOutExecutionConfig`, `HeldOutConfigError`, `load_held_out_config()`), a disabled-by-default
  configuration schema for a future held-out execution that fails closed the same way
  `BenchmarkPolicy` does: the loader rejects any config where `execution.execution_enabled` is not
  `false` or `execution.requires_recorded_approval` is not `true`. Also adds
  `scripts/check_held_out_trigger_safety.py`, a read-only governance-as-code check — wired into
  `.github/workflows/repository-hygiene.yml` alongside label drift detection — that fails if any
  workflow whose name matches a held-out/benchmark-execution naming convention could trigger on a
  routine `push` or `pull_request` event instead of only `workflow_dispatch` and/or a `push`
  restricted to `release-*` tags. Neither piece implements execution, opens the `test` partition,
  or changes `evaluation.py`'s `SUPPORTED_PARTITION`; both exist only so the execution command
  tracked by #73 has a validated, fail-closed place to plug into. Closes #126 (#130).

### Changed

### Fixed

- Fixed `create_split_quality_summary()` computing incorrect `shard_count` and `actual_ratios["shards"]` when `len(metadata.source_artifacts) > 1`: replaced the broken set-comprehension fallback (which used `record_id` as a shard path) with a direct `record_shards` lookup, and moved the total-unique-shards denominator outside the partition loop (#131).

### Removed

### Governance

- Disabled `allow_merge_commit` and `allow_rebase_merge` on the repository via `gh api -X PATCH`,
  leaving `allow_squash_merge` as the only enabled merge method: closes the gap where GitHub's
  merge-button UI still offered merge-commit and rebase-merge options despite squash-only being
  the documented and actual practice. Confirmed live via a follow-up `gh api` read-back. The
  `docs/governance/repository-governance.md` "Known residual gap" note (added under #97) is
  updated to record the fix instead of the gap (#98).
- Changed `modernization: ux`'s color from `d4c5f9` to `e99695` in `.github/labels.json` and
  synced live via `scripts/sync_github_labels.py`: `d4c5f9` was still shared with
  `portfolio: case-study`, and `e99695` is not used by any other declared label. Confirmed live
  after sync that only `modernization: ux` changed color and `portfolio: case-study` was
  untouched (#115).
- Documented local credential handling for `PROJECT_METADATA_TOKEN`-adjacent tooling:
  `docs/governance/github-metadata-automation.md` now states explicitly that the secret is a
  CI-only credential and that local `gh project`/`gh pr`/`gh issue` commands should authenticate
  via an interactive `gh auth login` session (with the `project` scope), never a token file --
  even a gitignored one is still plaintext on disk. Prompted by finding an undocumented local
  token file left over from earlier manual bootstrap work; the token itself was separately
  revoked and rotated (#121).
- Corrected `PROJECT_METADATA_TOKEN`'s documented token type from "fine-grained personal access
  token" to classic: fine-grained tokens do not expose a Projects permission for a user-owned
  project at all (a documented GitHub platform limitation, confirmed against current GitHub
  docs), and Project #5 is user-owned. Scope guidance updated to `project` (read/write) +
  `public_repo` + `read:org` on a classic token -- the last one live-tested and confirmed
  necessary: without it, every `gh project` subcommand fails its `--owner` resolution with a
  misleading `unknown owner type` error rather than an auth error (#121, #123).
- Added `project-status-sync.yml`, a GitHub Action that listens for `pull_request: closed` and,
  when the pull request merged, explicitly sets its Project #5 item's Status to `Merged` via
  `scripts/github/set_merged_project_status.py` -- resolving the #100 race where the built-in
  `Pull request merged` and `Item closed` workflows fire on the same event and `Closed` wins.
  `docs/governance/github-project.md` and `AGENTS.md` updated to describe the fix instead of the
  manual-correction workaround (#117).
- Executed the #105-documented migrate/retire pass against every remaining legacy label: 26 legacy
  spellings relabeled onto their canonical successor across 9 issues and 26 pull requests, then
  deleted once empty; `modernization:ux` renamed directly to a newly declared `modernization: ux`
  (no conflicting canonical existed); 5 zero-usage GitHub default labels (`duplicate`,
  `good first issue`, `help wanted`, `invalid`, `wontfix`) deleted, `bug` and `question` kept.
  `.github/labels.json` and `docs/governance/label-taxonomy.md` reconciled to the final label set;
  `area: data-pipeline` removed from both now that every carrier moved to `area: pipeline`. Two
  additional stale `status: in-progress` labels found by #113's full-history audit removed from
  issue #89 and PR #34 (#105, #113).
- Closed/merged items no longer carry a stale `status:*` label: removed leftover
  `status: in-progress` from issues #67, #107, #109, and PR #110 (#111).
- PR #108's Project #5 Status corrected from `Closed` to `Merged` (the #100 race, now applied to a
  PR item directly), its previously-empty custom fields backfilled, and its conflicting label set
  resolved to one `type:*` label, canonical `area: documentation`, and an added `priority:*` label.
  PR #110 added to Project #5 (it was absent entirely) and fully populated. #107/#109's Issue Type
  and Target Release fields reconciled to consistent values (#112).
- Added an explicit guard against minting legacy-format labels on new work: `AGENTS.md` and
  `docs/governance/label-taxonomy.md` now state that the existing-label normalization table records
  historical drift for a future migration pass, not acceptable spellings for new issues/PRs. Also
  documented (but did not execute) a full migrate/retire classification for the remaining legacy
  labels, posted to #105, ready for a future explicitly-authorized destructive pass (#105).
- `.github/labels.json` gains seven labels that were already in active current use but never
  backported into the manifest: `area: ci-cd`, `area: cli`, `area: data`, `area: pipeline`,
  `area: validation`, `portfolio: operational-maturity`, and `risk: low` (#67). This resolves the
  label drift `repository-hygiene.yml` reports against currently-open issues and pull requests;
  `scripts/detect_label_drift.py` (no `--include-closed`) now finds zero drift. A 31-item residual
  of pre-taxonomy legacy labels and unused GitHub defaults remains on closed/merged history --
  documented in `docs/governance/label-taxonomy.md`'s expanded normalization table, left for a
  separate, deliberate maintainer decision (per-issue relabeling and label deletion are both
  higher-consequence than a manifest edit, and are not performed by this change).

### Documentation

- `docs/governance/label-taxonomy.md`'s `area:*` dimension list and existing-label normalization
  table expanded to cover all 40 labels found undeclared by #67's live drift scan, marking each as
  a confident migration target or an open judgment call for the maintainer. `docs/governance/
  repository-hygiene.md` updated to describe the reconciled state rather than the original
  drift finding.

- `docs/governance/repository-governance.md` now describes the branch protection actually
  applied to `main` (#91) -- required status checks, 0-approval pull-request requirement,
  admin enforcement, linear history, and force-push/deletion restrictions -- verified against
  live settings, replacing the prior aspirational validation checklist. Discloses a known
  residual gap (repository-level merge-method settings still permit rebase merges; tracked in
  #98). `docs/governance/github-metadata-automation.md` updated to note the metadata gate is
  now a required status check.
- `docs/governance/github-project.md` gains a CLI reference for setting Project #5's Status field
  via `gh project item-edit`, with the live field/option IDs, confirmed writable by a round-trip
  test (moved an item, verified, reverted). Discloses that the `Pull request merged` workflow has
  consistently landed merged items at `Closed` rather than the documented `Merged` (tracked in
  #100). `AGENTS.md`'s pull-request-metadata guidance corrected to use `Review` (not `In Progress`)
  for an open pull request awaiting merge, matching this document's own lifecycle description.

- Added three AI-generated banner images (`docs/assets/`) to the root `README.md`, `notebooks/
  README.md`, and the Step 0 environment-setup notebook, giving the repository's highest-visibility
  entry points a consistent visual identity. `NOTICE.md` records their AI-generated-original
  provenance, distinct from the unresolved-provenance historical imagery covered by the MOD-008
  audit (#107).

### Security

## 1.0.0 - 2026-07-08

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
- `scripts/validate_curated_notebooks.py` and
  `.github/workflows/notebook-validation.yml` execute the three curated
  public notebooks end to end on every pull request, in an isolated `git
  worktree` copy seeded with a small synthetic WFDB record set and a
  matching acquisition manifest, so `acquire_dataset` takes its existing
  verify-and-reuse path and the real MIT-BIH dataset is never downloaded.
  This is genuine cell-by-cell execution of the real, unmodified curated
  notebooks, distinct from `scripts/notebook_quality.py`'s structural/
  hygiene-only check, which never executes a cell. See
  [notebook validation](notebooks/README.md#validation).

### Changed

### Fixed

- Pipeline progress output is now flushed per line so subprocess consumers
  (including the Step 0 notebook) receive it live instead of buffered until
  process exit, which is the default for non-TTY stdout in Python.
- Notebook 02's Step 2 readiness check and `resolve_indexed_file()` read
  `shard["path"]`/`shard["relative_path"]`, but the dataset index nests a
  shard's path and hash under `shard["file"]["path"]`/`shard["file"]["sha256"]`
  (`ShardIndex.file: IndexedFile`). This made Step 2 fail with "Missing train
  shard files: None None..." for any real Step 0 run, not just the new
  synthetic execution check that caught it.

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
- Completed a systematic audit of every model and pipeline claim in
  `README.md`, `MODEL_CARD.md`, and `docs/*` against generated evidence,
  actual configuration files, and code behavior (#71). Verified accurate
  and left unchanged: clean-checkout reproducibility across the `dev`,
  `notebooks`, and `experiments` locked sync groups (tested via a real
  `git clone` into a throwaway location, not inherited from CI); the
  historical confusion matrix and metric values in `docs/historical-
  results.md` against `archive/original_2022/report.ipynb`'s actual saved
  cell outputs, digit for digit; the annotation-mapping table and its
  24-symbol exclusion count against `configs/annotation-map-v1.toml`;
  the 70/15/15 split ratios and `configs/training-baseline-v1.toml`'s
  32 projection components; and the 144-source-file count (48 records
  x 3 extensions) in `docs/data-provenance.md`. Found and fixed:
  - `README.md`, `MODEL_CARD.md` (two places), and
    `docs/window-extraction.md` all claimed "the first signal channel"
    or "channel index 0" is used without a channel-selection analysis.
    PIPE-006 (#56) already replaced positional channel selection with
    name-based `MLII` resolution specifically because channel `0` isn't
    consistently `MLII` across records -- confirmed against the real
    `configs/windowing-v1.toml` and `windows.py`. Also confirmed, by
    fetching the actual MIT-BIH `.hea` headers for records 102, 104, and
    114, that `configs/windowing-v1.toml`'s `exclude_record_ids =
    ["102", "104"]` is exactly correct: those two records have no
    `MLII` channel at all (`V5`/`V2` only), while `114` (which shares
    the same historical `channel_index = 0` instability) does have an
    `MLII` channel and needs no exclusion under name-based selection.
  - `docs/architecture.md`'s "Planned migration sequence" claimed
    "**Next:** define protected test evaluation and model-card policy"
    and listed creating curated notebooks as a future, unnumbered step.
    Both are long complete (`MODEL_CARD.md`, `docs/benchmark-
    governance.md`, and `notebooks/00`-`02` all exist and are
    execution-validated); marked both items `Completed` and added a
    pointer to `docs/modernization-roadmap.md` as the authoritative,
    currently-maintained source, to keep this superseded list from
    drifting the same way again. Also updated the directory map, which
    was missing `.github/`, `notebooks/local/`, and `tests/scripts/`.
  - `docs/README.md`'s documentation index was missing an entry for
    `docs/baseline-training.md`, a real, linked-from-elsewhere file.
  - `docs/environment-reproducibility.md` implied `scikit-learn` is only
    installed with `--group experiments`; it is directly declared in
    the `notebooks` group and already available with `--group notebooks`
    alone. Fixed the workflow table and both import-verification
    commands, confirmed against a real clean-checkout sync.

### Security

## 0.1.0

`0.1.0` is the current initial package version. No Git tag or GitHub release is
associated with this version, and no historical release date is asserted.
