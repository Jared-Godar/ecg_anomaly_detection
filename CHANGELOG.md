# Changelog

![Technical banner for the changelog, showing version history and governance motifs.](docs/assets/ecg-changelog-banner.png)

This changelog records notable repository changes using a structure inspired by
Keep a Changelog. It does not claim formal compliance with that specification.

## Unreleased

### Added

### Changed

### Fixed

### Removed

### Dependencies

### Governance

- Codified the **proactive continuity walkthrough** and **merge green-light** contracts in
  `AGENTS.md` (#251). Every implementation session now writes a fill-in-the-rails walkthrough —
  numbered mechanical steps as copy-pasteable Fish blocks, each followed by its verification
  command — immediately after branching (not on wind-down request), refreshed at the PR-open and
  awaiting-merge checkpoints, to the gitignored `artifacts/walkthroughs/<UTC-timestamp>-issue-<n>-<slug>.md`;
  worktree sessions follow the same write-locally-then-copy-out flow as session handoffs, and the
  closure pass's pre-prune copy-out now covers walkthroughs alongside handoffs. A new tracked
  template, `docs/governance/templates/continuity-walkthrough-template.md`, generalizes the #246
  exemplar (executed end-to-end by hand as PR #249), keeping its load-bearing traits
  (verification after every action step, Fish-only syntax with `printf` instead of heredocs, the
  no-changes-needed early exit, the #100 issue-lane quirk and cancelled-duplicate-check
  watch-items, the gh `--jq` `[0:7]` slice) while pointing at the canonical board-ID table in
  `docs/governance/github-project.md` instead of freezing IDs (the #213 option-ID regeneration
  lesson) and reflecting the verification-graduation-ladder tier placements in its board blocks.
  The canonical PR workflow additionally requires an explicit merge signal: a HOLD from the first
  push naming what is still running, then an unprompted "GREEN LIGHT: clear to squash-merge
  PR #N via the GUI" the moment post-open verification completes — GUI check status and
  merge-button eligibility are never the authoritative merge signal, and merge-independent work
  never delays the green light (origin: the #217→#219 mid-run merge preemption and the PR #250
  duplicated parallel closure pass). The session-handoff contract now links the walkthrough for
  command rails instead of repeating them.

### Documentation

### Security

## 1.1.0 - 2026-07-16

### Added

- Maintainer editorial pass for #246: added a first-person "Why this project" section to
  `README.md` (regulated-sector motivation and governance-first observations), linked the
  public GitHub project board and milestone dashboard from the README, and reworded the
  Agentic Engineering model-agnostic claim to mirror `AGENTS.md`; the #245 documentation
  pages were reviewed end-to-end and held with no alterations.
- Portfolio-signal deep-dive pages and in-repo release notes for #245: six curated pages under
  `docs/portfolio/` (`agentic-engineering`, `governance`, `reproducibility`, `testing-rigor`,
  `data-engineering`, `operational-maturity`), each expanding its README portfolio-signal
  section into a fuller readout that links into the existing deep docs, with every claim
  traceable to repository code, configuration, documentation, or recorded evidence;
  `docs/releases/v1.0.0.md`, a retrospective portfolio-signal reframing of the published
  v1.0.0 release (which identifies itself as such — the published GitHub release body is
  deliberately untouched as the honest historical record); and `docs/releases/v1.1.0.md`, the
  canonical v1.1.0 release notes seeded from the approved draft v2 on #180 with the tag
  session's ⟨slots⟩ intact (only relative link paths were adapted to the page's location).
  Every README portfolio-signal section now ends with a deep-dive link to its
  `docs/portfolio/` page, and the README Reference table links both release-notes pages.
- Automated Project #5 membership and label-derived field population at item-creation time for
  #233. A new `project-item-autofill` workflow fires on `issues`/`pull_request`
  `opened`/`labeled` events and runs `scripts/github/populate_project_item.py`, which
  idempotently adds the item to the board (verified by targeted re-lookup), defaults Status to
  Backlog only when unset, and fills every field derivable from the item's labels via the new
  shared mapping table `scripts/github/project_label_mapping.py` (`type:` → Issue Type,
  `priority:` → Priority, `risk:` → Risk, `size:` → Size, `area:` → Repository Area,
  `portfolio:` → Portfolio Signal) — converging as labels land, only ever filling unset fields
  (curated values win), resolving option IDs by name at runtime, and skipping governed-bot
  (Dependabot) items whose stamping the Dependabot autofill path owns. Workstream and Target
  Release remain deliberately human-set. A scheduled hygiene backstop
  (`scripts/detect_board_drift.py`, new `board-drift` job in `repository-hygiene.yml`) flags
  open items with missing board membership or unset label-derivable fields; the shared access
  layer gained the issue-side targeted lookup, `ensure_item`, and `set_single_select_if_unset`
  primitives (extracted from the Dependabot sync, which now delegates to them). Documented in
  `docs/governance/github-metadata-automation.md`, `docs/governance/github-project.md`, and
  `AGENTS.md`; unit tests cover the mapping table's manifest-completeness invariant,
  precedence, idempotency, conflict withholding, and the house exit-code mapping.
- Added a sanctioned non-closing issue reference marker to the automated PR metadata gate for
  #216: a pull request body line of the form `Non-closing ref: #N — <reason>` (case-insensitive,
  em-dash or hyphen, reason mandatory) now satisfies the closing-reference requirement without
  asking GitHub to auto-close the referenced issue on merge. The marker runs the same
  linked-issue checks as a closing keyword (Project #5 membership, nine-field completeness,
  milestone inheritance) — only the auto-close side effect is skipped. Naming the same issue via
  both a closing keyword and the marker is detected as an ambiguity violation. Fenced code blocks
  and inline code spans are stripped before matching (consistent with closing-keyword parsing).
  Documented in `docs/governance/github-metadata-automation.md` and the `AGENTS.md`
  project-planning-metadata rules. Unit tests cover marker parsing, code-block/inline-code
  inertness, the ambiguity case, and `_run_validation` integration with the Project snapshot.
- Added Kiro IDE workspace configuration for #220: tracked steering files
  (`project-context.md`, `workflow-rules.md`, `github-project.md`, `seed-prompt-template.md`)
  and agent hooks (`gate-reminder.json`) provide durable multi-agent workflow context, while
  `.kiro/settings/` (machine-specific permissions and MCP config) stays gitignored. The
  `.gitignore` uses a deny-then-whitelist pattern (`.kiro/` ignored, `!.kiro/steering/*.md` and
  `!.kiro/hooks/*.json` tracked).
- Extended the portfolio-signal taxonomy for #210 with two paired label + board-option signals:
  `portfolio: testing-rigor` (layered test design and coverage discipline as the item's subject)
  and `portfolio: agentic-engineering` (agent contracts, instruction files, or agent-workflow
  enforcement as the subject). `.github/labels.json` declares the labels,
  `docs/governance/label-taxonomy.md` records their applicability boundaries plus the
  deliberately deferred third candidate (CI supply-chain hardening / meta-CI stays
  `portfolio: operational-maturity` until roughly eight or more primary-subject items
  accumulate), and `docs/governance/github-project.md` documents the two new Portfolio Signal
  board options — added via the project web UI, never `updateProjectV2Field` — with the field's
  full read-back-verified option-ID table. The labels' declared colors intentionally diverge from
  the issue's proposal, which had duplicated `portfolio: case-study` and `portfolio: release`
  exactly (maintainer-confirmed oversight).
- Added a `Not Planned` Status lane to the ECG Pipeline Modernization project board for #208 and
  documented its semantics in `docs/governance/github-project.md` and the `AGENTS.md`
  project-planning bullet: the lane is used only for issues closed with GitHub's native
  "not planned" state reason, keeping withdrawn work visibly distinct from delivered work in
  board views and field-based filters. `Closed` stays reserved for completed issues,
  pull-request lanes and the `project-status-sync` automation are unaffected (it targets PR merge
  events, not issue closures), and the built-in issue-closed workflow's inability to distinguish
  state reasons is recorded: items land in `Closed` first and are moved to `Not Planned` manually
  with a read-back check. The two withdrawn hosted-notebook issues (#198, #200) are backfilled
  into the new lane.
- Documented the notebook-surface label mapping for #207 in `docs/governance/label-taxonomy.md`,
  cross-referenced from `docs/governance/github-project.md`: there is deliberately no
  `area: notebooks` label — notebook-focused issues and pull requests carry `area: documentation`
  on the label side (precedent: PRs #202 and #205, consistent with the #105/#113 migration of the
  legacy `area:notebooks` spelling) while the board's Repository Area field records the
  finer-grained `notebooks` value — so future PR authors stop rediscovering the asymmetry through
  failed label assignment.
- Added bounded transient-failure retries to dataset acquisition for #201, implementing the
  `AGENTS.md` defensive-external-calls rule at its origin site: each per-file HTTPS transfer now
  retries plausibly transient connectivity failures (timeouts, dropped/reset/refused connections,
  name-resolution failures, and HTTP 429/500/502/503/504) up to three total attempts with
  exponential backoff (2s, then 4s), removing the partially staged file between attempts, so a
  single brief PhysioNet blip no longer aborts a whole 48-record acquisition. Permanent failures
  (HTTP 404/403, size-cap violations, digest or size mismatches, rejected redirects) still fail
  fast on the first attempt, and every existing integrity guarantee — atomic commit-on-full-success,
  SHA-256 digest verification against committed expectations, final-URL validation, and size caps —
  applies unchanged to whichever attempt succeeds. When retries are exhausted, the failure message
  names the URL and attempt count, states plainly that the cause is an external connectivity or
  service condition rather than a repository or setup defect, and gives re-run remediation.
  Deterministic tests inject the backoff sleep and cover transient-then-success, exhaustion,
  permanent-not-retried, integrity-mismatch-not-retried, and resume-path retry behavior.
- Added qualified timing and remaining-duration estimates to long-running acquisition progress for
  #199: each existing per-record completion line (in `run-pipeline` and the standalone `acquire`
  command) now also reports that record's measured duration, the acquisition phase's measured
  elapsed time, and an `approx. remaining` projection derived only from the current run (mean
  completed-record duration times records outstanding), with an explicit
  `approx. remaining estimating...` warm-up state until three records have completed and a factual
  `00:00` on the final record. The shared, clock-injected estimator and suffix formatter live in
  `progress.py` (`UnitTimingEstimator`, `format_unit_timing_suffix`) with deterministic tests, and
  update frequency is unchanged: still exactly one concise line per configured record. A documented
  audit in the pipeline orchestration guide inventories every other intermediate progress site
  across the governed CLI and public notebooks 00–02 with its apply/elapsed-only/no-change decision,
  so timing decoration is not added indiscriminately. Observational only: no change to acquisition
  ordering, retries, manifests, artifacts, models, metrics, or evaluation boundaries.

- Established Graphviz as the repository's diagram toolchain: version-controlled `.dot` sources
  under `docs/diagrams/src/`, two Python composition helpers (`docs/diagrams/compose_inset.py`,
  `docs/diagrams/pad_svg.py`), and a design spec (`docs/diagrams/design-spec.md`) documenting the
  approved visual language, deterministic generation commands, and per-diagram rationale.
  Regeneration is reproducible given the same Graphviz version; Graphviz itself is a one-time
  authoring tool, not a declared package dependency (#104, #167).
- Implemented and executed, once, the separately governed held-out evaluation tracked by #73:
  `src/ecg_anomaly_detection/held_out_evaluation.py` and its `ecg-data` CLI entry point implement
  an explicitly enabled, approval-gated evaluator that verifies frozen-candidate, grouped-split,
  model, training, dataset-index, shard, and approval lineage before any protected access, with a
  manual-only (`workflow_dispatch`) `.github/workflows/held-out-evaluation.yml` and no routine
  trigger. The one governed execution evaluated the frozen candidate against 16,545 protected
  `test` windows across seven subject-grouped records, publishing aggregate-only results,
  artifact identities, runtime and hardware context, append-only rerun evidence (including a
  preserved pre-result output-parent failure and the policy-permitted retry), and interpretation
  limitations in `docs/held-out-evaluation.md`; source ECG data, patient-level derived data,
  models, predictions, and generated run artifacts remain outside Git. The result is bounded
  evidence for one frozen candidate and one grouped split, not a claim of broader
  generalization, clinical validity, medical utility, or production-healthcare readiness (#73).
- Added a comprehensive internal-documentation standard across every supported Python file under
  `src/`, `scripts/`, and `tests/`: all modules, classes, functions, methods, and nested callables
  now carry docstrings, while control flow, resource management, setup, assertions, and other
  meaningful blocks include intentionally explicit comments describing both behavior and design
  intent. Added `scripts/check_code_commentary.py`, its focused test suite, an always-run
  pre-commit hook, and `docs/code-commentary-audit.md` to preserve the complete audited inventory
  and prevent structural coverage from regressing. The historical `archive/original_2022/` tree is
  intentionally unchanged (#149).
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
- Added a validation-only centroid-distance margin threshold sweep: `configs/threshold-sweep-v1.toml`,
  `evaluate_threshold_sweep_from_index()` and `load_threshold_sweep_config()` in `evaluation.py`,
  and the standalone `ecg-data evaluate-threshold-sweep` CLI subcommand (not part of the
  orchestrated `run-pipeline` sequence, following the `record-benchmark-approval` precedent). For
  each configured threshold, reports covered-window count and macro-averaged precision/recall/F1
  over only windows whose per-window nearest/second-nearest centroid distance gap is at or above
  that threshold, against the `validation` partition only. The margin is a raw squared distance in
  projected feature space, not a probability, and this introduces no ROC, AUC, or calibration
  claim; the protected `test` partition is never opened. Persists a new `ThresholdSweepMetrics`
  artifact (`threshold-sweep-metrics.json`) separate from the existing evaluator's
  `ValidationMetrics` schema. See `docs/threshold-sweep-analysis.md`. Closes #84.
- Added direct unit tests for the threshold-sweep evaluator's core arithmetic, which previously
  had only end-to-end coverage: exact nearest-to-second-nearest centroid-distance margins
  (including tied distances), threshold filtering, macro precision/recall/F1 with explicit
  zero-division behavior, and table-driven configuration-loader validation cases (missing
  fields, protected-`test` selection, unsorted and non-finite thresholds). Test-only; no
  production behavior change (#147, #148).

### Changed

- Reframed `README.md` as a recruiter-facing engineering showcase for #217: the modernization
  narrative moves from headline to origin-story context, replaced by six evidence-backed
  portfolio-signal sections (Agentic Engineering, Governance, Reproducibility, Testing Rigor,
  Data Engineering, Operational Maturity) with concrete file/feature citations. The large
  capability table, current-status table, architecture table, historical-experiment section,
  and detailed known-limitations section are trimmed from the README and preserved in their
  existing linked `docs/` pages — no information lost, only relocated for a skim-friendly
  recruiter audience. Quick start, use-limitation notice, dataset attribution, and reference
  links are retained in compact form.
- Hardened the public Step 0 notebook's external-connectivity touch-points for #201 so a transient
  network failure no longer surfaces as a raw Python traceback: a new shared helper cell owns a
  bounded connectivity-signature classifier (including the acquisition retry layer's graceful
  exhaustion wording), a `[connectivity]` guidance panel (what happened, that it is an external
  network/service condition rather than a code or setup bug, and what to do), and a deliberate
  `SystemExit` halt that stops Run All without a multi-frame traceback while still blocking Steps
  1 and 2 on missing artifacts. The environment bootstrap now streams `uv sync --group notebooks`
  output live while capturing it for classification, the governed pipeline runner's failure
  classifier gains a distinct `EXTERNAL_CONNECTIVITY` classification ahead of its generic fallback,
  and the opt-in Bash launcher's `curl` installer and dependency-sync steps print equivalent
  bounded guidance. Non-connectivity failures keep their existing strict classifications and
  `RuntimeError` behavior, the recovery note now describes the implemented automatic retries and
  graceful halt, and static regression tests assert the shared classifier, both classified
  touch-points, and the absence of a bare re-raise on the connectivity path (notebook v2.54).

- Improved the three supported public notebooks' presentation and navigation for #194: notebooks
  01 and 02 now use approved banners consistent with notebook 00; cross-notebook and relevant
  policy references are repository-relative links; important prerequisites, protected-test
  boundaries, destructive-action warnings, and interpretation limits use a shared accessible
  callout treatment; and regression coverage validates local targets, banner dimensions/PNG
  structure/alt text, role-consistent callout styling, and lineage palette/claim semantics. The
  notebook guide now documents a non-executing Jupyter HTML render
  check for banners, the lineage diagram, links, and panels. The Step 1 lineage diagram has also
  been redesigned in the approved dark Graphviz visual system with a tracked source, composited
  legend, reproducible export workflow, explicit protected-test boundary, and visible source
  attribution. All three public notebooks now give quiet, immediately flushed execution feedback:
  Step 0 adds qualified bootstrap and first-run timing guidance around its existing streamed
  pipeline stages, now including one downloaded/reused integrity update per configured acquisition
  record and a short invocation cell that keeps live VS Code/Jupyter output in view. Step 1 bounds
  optional run-evidence discovery with one start/completion pair and exposes configured channel and
  record-exclusion context. Step 2 reports waveform loading, fixed-model fitting, and validation
  scoring, makes successful repository path resolution visible, and emits a qualified minute-scale
  elapsed heartbeat during an otherwise silent fit. All three notebooks now open with a concise,
  conversational task overview, qualified first-run/rerun timing, and compact jump links; detailed
  version history is preserved in bottom appendices. The supported workflow now stays within one
  local VS Code/Jupyter checkout using its locked `.venv` kernel. Notebook 00 verifies and retains
  generated state in that checkout, while notebooks 01 and 02 begin with visible local continuity
  confirmations before independently checking their required artifacts; no selector, cross-runtime
  copy, upload, external-storage handoff, or programmatic kernel restart remains. Optional
  web-runtime integration is deferred to #200 instead of being represented as supported by #194.
  Expectations remain deliberately approximate and measured times remain observational,
  with no changes to model inputs, parameters, metrics, evaluation boundaries, saved outputs, or
  artifact policy.
- Documented the public notebooks' fixed random seeds and their purpose for #194: notebook 00
  explains that the versioned split and training configs both pin `seed = 2022`, notebook 01
  explains the split and training seeds it already surfaces, and notebook 02 explains its fixed
  `random_state=0` — in every case so repeated local runs reproduce the same subject partitions,
  the same fitted models, and the same validation metrics. Seed values themselves are unchanged.
- Added a recovery note to the Step 0 notebook for the rare transient PhysioNet download timeout,
  explaining that the failure is a network hiccup rather than a broken notebook and that a re-run
  restarts the atomic download cleanly. The deliberate defensive handling (graceful messaging and
  optional connectivity retries) is tracked separately in #201.
- Codified a **defensive-external-calls** engineering-discipline rule in `AGENTS.md`: external
  calls (downloads, package installs, HTTP/API, remote CLIs) must retry transient failures with
  backoff and, on exhaustion, exit gracefully with clear remediation instead of a raw traceback.
  Surfaced by this PR's walkthrough; existing gaps are retrofitted under #201.

- Prepared the repository for the v1.1.0 release: bumped the package version from `1.0.0` to
  `1.1.0` (`pyproject.toml`, `uv.lock` via `uv lock`, and `tests/unit/test_package.py`'s version
  assertion), backfilled this changelog's entries for all v1.1.0-milestone work, moved the
  accumulated `Unreleased` content under this dated `1.1.0` heading with a fresh empty
  `Unreleased` scaffold above it, and updated the three current-version statements (`README.md`,
  `docs/governance/versioning.md`, `docs/governance/releases.md`) to describe the 1.1.0 state.
  Creates no tag, GitHub release, or package publication — tagging remains a separate, deliberate
  maintainer act tracked by #180 (#179).
- Extended per-stage progress banners from `run-pipeline` to the 13 standalone `ecg-data`
  subcommands that do real I/O (`acquire`, `inventory`, `verify`, `validate-record`,
  `map-annotations`, `extract-windows`, `split-windows`, `index-dataset`, `create-run-manifest`,
  `validate-benchmark-policy`, `record-benchmark-approval`, `evaluate-threshold-sweep`, and
  `check-local-notebooks`), each wrapping its existing body in one `ProgressReporter` stage block
  with its original completion message preserved. `list-runs`/`purge-run` are deliberately
  excluded as near-instantaneous local operations, and `check-local-notebooks` routes banners to
  stderr so `--json` stdout stays machine-parseable. Covered by `capsys` banner assertions across
  8 integration-test files plus a first CLI-level test for `validate-benchmark-policy`;
  `docs/pipeline-orchestration.md` and the 11 other affected subcommand docs updated (#61, #165).
- Changed `ruff.toml`'s `lint.select` to add `UP` (pyupgrade), `S` (flake8-bandit), and `SIM`
  (flake8-simplify) for ongoing regression protection, plus a `tests/**` per-file-ignore for
  `S101` (`assert` is the test mechanism itself under pytest, not a stripped-under-`-O` production
  guard). Of the 627 violations the expanded rule set surfaced, 577 were that `S101`-in-tests
  case; the remainder were resolved with zero behavior change: mechanical auto-fixes
  (`typing.Callable`/`Sequence`/`Iterator`/`Mapping` → `collections.abc`, Yoda-condition
  rewrites, import re-sorting), hand-merged nested `with` statements where ruff couldn't
  auto-fix due to line length, and targeted `# noqa` suppressions with one-line justifications
  on `subprocess` calls that only ever run a fixed, hardcoded command list (`S603`/`S607`), one
  already-validated `urllib.request.Request` call (`S310`), and one deterministic seeded shuffle
  used for reproducible splitting rather than cryptographic purposes (`S311`). Closes #136.
- Changed `run_manifest._capture_git_state()` to degrade gracefully instead of raising
  `RunManifestError` when Git is unavailable or exits non-zero, matching the existing
  `reproducibility.capture_git_metadata()` pattern: it now returns a sentinel `GitState(revision=
  "unknown", dirty=None)` rather than failing the whole `create-run-manifest` run. `GitState.dirty`
  is widened to `bool | None` to carry the sentinel; the existing hard-error path (Git succeeds but
  returns a malformed, non-40-hex revision) is unchanged. `UNKNOWN_GIT_REVISION` is documented on
  the `GitState` dataclass and in `docs/run-manifests.md`, and is provably distinguishable from a
  real commit hash since it is not 40 hex characters (#133). Also adds a
  `test_benchmark_approval.py` test covering `_missing_lineage_references()`'s previously-untested
  fail-closed else-branch for a required lineage reference beyond the known 7 (#135). Bundled
  together because a new test proves the two compose safely: a manifest whose Git state degraded
  under #133 still fails closed on `repository_commit_hash` in `record_benchmark_approval()`,
  so the graceful-degradation change cannot let an unverifiable-provenance run pass benchmark
  approval (#133, #135).
- Changed `pyproject.toml`'s `Development Status` classifier from `3 - Alpha` to `4 - Beta`,
  reflecting the pipeline's actual maturity now that the modernization has a tagged 1.0.0 release.
  No dependency, version, or build-backend change; `uv build` and the CI package-build job remain
  green (#137).

### Fixed

- Corrected three v1.1.0 release-gate documentation and metadata defects surfaced by the #218
  release-readiness audit, all landing before the deliberate tag/publish step (#180). (1) The
  tracked Kiro steering file `.kiro/steering/project-context.md` named the wrong dataset in its
  `## Dataset` section — "PhysioNet Computing in Cardiology Challenge 2017 (AF classification)" —
  which would seed every Kiro session with a false premise; it now names the actual dataset, the
  MIT-BIH Arrhythmia Database v1.0.0 (`mitdb`, PhysioNet, DOI 10.13026/C2F305), matching
  `configs/mitdb-v1.0.0.toml`, `docs/data-provenance.md`, and `README.md` (#223). A repo-wide
  sweep confirmed this was the only 2017/CinC/AF residue. (2) `docs/governance/releases.md`
  asserted the package was already "tagged as the `v1.1.0` GitHub release," contradicting
  `versioning.md`'s metadata-is-not-a-tag rule and the same file's own following sentences; the
  paragraph now states accurately that the metadata is `1.1.0` while no `v1.1.0` tag or release
  exists yet, and records that the separately authorized tag step updates it to released state at
  that time (#224). (3) `CITATION.cff` lacked a `version` field and carried a stale
  `date-released: 2022-01-18`, so it could not satisfy `versioning.md`'s requirement that package
  metadata, changelog heading, tag, and citation identify the same version; it now declares
  `version: 1.1.0` (matching `pyproject.toml`) and intentionally omits `date-released` — with a
  documented in-file comment — until the tag is cut, deferring the release-date stamp to #180
  rather than asserting a date for a release that has not occurred. Validated with `cffconvert`
  (schema 1.2.0). Documentation and metadata only; no supported pipeline code, tests, or
  notebooks changed.
- Folded `http.client.IncompleteRead` into acquisition's single-exception contract and transient
  retry classification for #206: a server closing a known-`Content-Length` response mid-body makes
  `response.read()` raise `IncompleteRead` — an `http.client.HTTPException`, outside the previously
  caught `(OSError, TimeoutError, URLError)` set — so it escaped `acquire_dataset` as a raw
  traceback instead of entering #201's bounded retry path. The production HTTPS transport now
  collapses every `http.client.HTTPException` into `AcquisitionError` and classifies
  `IncompleteRead` specifically as transient (a mid-body connection drop, retried up to three
  total attempts with 2s/4s backoff and staged-file cleanup between attempts), while other
  protocol-level exchange failures stay permanent and fail fast on the first attempt. Every #201
  integrity and exhaustion guarantee is unchanged: digest/size mismatches are never retried,
  atomic commit-on-full-success holds, and exhausted retries still exit with the same graceful
  bounded external-connectivity message. Deterministic tests pin the classification boundary
  (`IncompleteRead` transient, `BadStatusLine` permanent), the collapse at the transport seam,
  and a full end-to-end retry of a truncated body through the production transport via the
  injectable sleep seam.
- Corrected the #192 Dependabot entry under `### Dependencies` below, which recorded the
  superseded `setup-uv` version 8.3.0 instead of the actually merged 8.3.1 (#204). A mid-flight
  `@dependabot recreate` retargeted the group bump from 8.3.0 to 8.3.1, but the final autofill
  run's `dependabot/fetch-metadata` structured outputs still carried the stale version, which the
  autofill script faithfully rendered per its documented no-head-content security contract — an
  upstream metadata staleness, not an automation defect. Version string only; no code change.
- Fixed the Step 1 and Step 2 notebooks' local-checkout kernel guard, which rejected the correct
  uv-created `.venv` kernel: it resolved the interpreter symlink out to the base Python and then
  failed its `.venv` membership check. The guard now compares the interpreter path lexically
  (matching Step 0), so a correctly selected uv venv kernel is accepted while genuinely wrong
  kernels are still refused. Added a regression test asserting the symlink-safe comparison.
  Surfaced during the #194 walkthrough.
- Hardened the shared GitHub CLI layer (`scripts/github/github_api.py`) against transient
  GitHub 5xx server errors (#190): `run_gh` now retries them on the same bounded 2s/5s/10s
  schedule it already used for the secondary rate limit, so a freshly opened pull request's
  momentarily uncomputed diff (observed as `Server Error ... (HTTP 500)` on the changelog
  gate's first CI run) no longer fails a governance check on its first attempt. Primary
  rate-limit classification stays fail-fast and 4xx caller errors are never retried; every
  shared-layer consumer inherits the hardening. Also corrected `run_gh`'s docstring, which
  named a nonexistent `GraphQLQuotaExhaustedError` instead of `PrimaryRateLimitError`.

- Stopped `extract_closing_issue_numbers` in `scripts/github/validate_project_metadata.py` from
  matching closing keywords quoted as prose: the raw-text `_CLOSING_KEYWORD_PATTERN` scan treated
  a backtick-quoted example like `` `Closes #154` `` inside a sentence as a real closing
  directive (reproduced live by PR #160's own body, which produced a spurious "closed by a
  non-merge event" warning about an issue the PR never closed). Fenced code blocks and inline
  code spans are now stripped before matching — chosen over anchoring matches to dedicated
  closing-reference lines, which would diverge from GitHub's own full-prose auto-close detection
  and risk false negatives (#161, #163).
- Taught `_run_gh()` in `scripts/github/validate_project_metadata.py` to distinguish GitHub's two
  rate-limit failure modes instead of collapsing every failing `gh` call into the same
  `MetadataValidationError`: primary (hours-long) rate-limit exhaustion now fails fast with a
  message clearly labeled as infrastructure rather than a metadata defect, while the secondary
  (short-lived) rate limit gets a small bounded retry with backoff. Found live when this PR's own
  metadata check failed on a GraphQL budget exhaustion caused by same-session
  `gh project item-edit` usage; covered by three new tests in
  `tests/scripts/test_validate_project_metadata.py`. Closes #156 (#155).
- Restored Pyright's built-in default excludes (`**/node_modules`, `**/__pycache__`, `**/.*`)
  alongside the repository's custom entries (`archive`, `.venv`, `build`, `dist`) in
  `pyproject.toml`'s `[tool.pyright]` `exclude` array: setting `exclude` explicitly replaces
  rather than extends the defaults, which had left runtime `__pycache__` and hidden directories
  under `src/`/`tests/` inside Pyright's analysis scope — the exact gap Pylance's
  `missingDefaultExcludes` diagnostic warns about. Closes #154 (#155).
- Corrected `notebook_quality.py`'s `NARRATIVE_NOTEBOOK` constant from the stale
  `notebooks/narrative-walkthrough.ipynb` to the actual `notebooks/01-narrative-walkthrough.ipynb`:
  the stale path made `discover_local_notebooks(..., include_narrative=True)` silently skip the
  narrative notebook because its intentional missing-file tolerance masked the mismatch. Fixed
  the regression test that reused the same stale synthetic filename and added
  `test_narrative_notebook_constant_matches_real_repository_file`, backed by the real repository
  file, so a future rename cannot drift silently again (#152, #153).
- Fixed `create_split_quality_summary()` computing incorrect `shard_count` and `actual_ratios["shards"]` when `len(metadata.source_artifacts) > 1`: replaced the broken set-comprehension fallback (which used `record_id` as a shard path) with a direct `record_shards` lookup, and moved the total-unique-shards denominator outside the partition loop (#131).
- Fixed `_install_without_overwrite()` in `acquisition.py` raising a confusing generic `AcquisitionError` when `os.link()` hits `EXDEV` (staging and destination on different filesystems, e.g. Docker volumes or network mounts). It now falls back to copying the source into a temporary file alongside the destination (guaranteed same filesystem) and hard-linking from there, preserving the atomic no-overwrite guarantee that a plain copy-and-replace would have lost (#132).
- Stopped `quality.yml`'s `pre-commit` job failing on every push to `main`: the
  `no-commit-to-branch` hook always fails on the `push`-to-`main` CI trigger, whose checkout
  legitimately is `main`. The CI job's existing `SKIP` list now includes `no-commit-to-branch`
  (CI invocation only); `.pre-commit-config.yaml` is unchanged, so the hook still protects local
  commits against landing directly on `main` (#90, #95).

### Removed

### Dependencies

- Bump `astral-sh/setup-uv` from 8.2.0 to 8.3.1 (github_actions) via Dependabot (#192).

### Governance

- Executed the v1.1.0 pre-tag release deliverables for #180 in this final pull request:
  promoted every accumulated `## Unreleased` entry into this `## 1.1.0` section and re-dated
  its heading to the tag date (2026-07-16), stamped `date-released: 2026-07-16` into
  `CITATION.cff` (removing the documented tag-time deferral comment), flipped
  `docs/governance/releases.md` from its pending-state paragraph to the released state, and
  filled the tag-session slots in `docs/releases/v1.1.0.md` (tag date, changelog anchor, the
  #236/#237 and #245/#246 greatest-hits lines, and a fresh clean-checkout verification
  receipt). Release-facing metadata and notes only; no supported pipeline code, tests, or
  notebooks changed. The annotated `v1.1.0` tag lands on this pull request's merge commit so
  the tagged tree and its release-facing metadata agree (`docs/governance/versioning.md`).
- Codified the **automation verification graduation ladder** (#248) in `AGENTS.md`'s
  project-planning rules and `docs/governance/github-metadata-automation.md` (the detailed
  policy home, with a pointer from `docs/governance/github-project.md`'s Automation section):
  how often agent sessions re-verify a recurring automation's outcome now migrates down a
  four-tier cost ladder as trust accumulates — per-event read-backs for new or changed
  automation (until ~5 consecutive clean observations, recorded as documented streak evidence),
  sampling every 3rd event for proven automation (action-gating reads are never skipped at any
  tier), and scheduled drift-backstop checks with no routine per-event reads once the invariant
  is machine-checked — with an explicit regression rule: any observed failure resets that
  automation to tier 1. Current placements were recorded with live-verified evidence
  (2026-07-16): `project-status-sync` at tier 2 on a clean streak ≥ 4 (PRs #242/#243/#247/#249,
  run conclusions and board lanes both read back), creation-time autofill at the tier 1→2
  boundary (probe #238 plus live observations, including issue #248's own automatic
  `portfolio: governance` → Portfolio Signal derivation; underivable-label gaps keep per-event
  reads valuable), and issue-lane-on-close recorded as a manual step outside the ladder (no
  automation exists to verify). #240 is noted as the first tier-3 migration instance. One-off
  agent writes are explicitly unaffected — every agent-performed mutation keeps its targeted
  read-back.
- Amended the **session-handoff continuity** contract for worktree-isolated sessions (#236): a
  session confined to a `.claude/worktrees/` worktree often cannot write to the primary
  checkout's `artifacts/session-handoffs/` (its file-editing tools are confined to the
  worktree), so its handoff previously landed in the worktree's own ignored
  artifacts zone — exactly the directory deleted when the closure pass prunes the worktree (hit
  live during #233). The contract's three surfaces (`AGENTS.md` standing commitments plus
  canonical-workflow step 10, the `.claude/CLAUDE.md` Claude addendum, and
  `.kiro/steering/agent-conduct.md` with `workflow-rules.md` step 10) now codify the fallback
  chain: write the handoff worktree-local first as the checkpoint, copy it into the primary
  checkout's `artifacts/session-handoffs/` in the same turn when that checkout is writable, and
  fall back to one fenced in-chat markdown block when no checkout is writable at all; the
  closure pass copies any remaining handoff files out of a worktree before removing it, so
  pruning never deletes the only copy. Project memory records the primary-checkout path, which
  survives pruning, never the worktree path.
- Codified a **session-handoff continuity** contract in `AGENTS.md`'s standing commitments, with
  a Claude-specific addendum in `.claude/CLAUDE.md` (#211): when a working session approaches the
  maintainer's usage limit — signaled explicitly ("wrap up", "session limit approaching") or via
  wind-down signs such as context compaction — the agent writes a Markdown handoff walkthrough to
  the gitignored `artifacts/session-handoffs/<UTC-timestamp>-<slug>.md` before the session ends: a
  state snapshot (branch/commit/PR/issue state, gates run with results, a plain done/queued/owed
  accounting), numbered next steps in which every action is a copy-pasteable Fish code block
  runnable from the repository root with a per-step verification command, relevant links, and
  open risks — so in-flight work continues locally without any agent. Handoff files are never
  committed and never contain secrets; the maintainer's request is the authoritative trigger
  because agents cannot observe actual plan quotas. `docs/architecture.md`'s directory map now
  names session handoffs among the ignored `artifacts/` contents.
- Integrated bot-authored (Dependabot) pull requests into the changelog and metadata gates
  (#193): a new `Dependabot PR autofill` workflow (`.github/workflows/dependabot-autofill.yml`,
  `pull_request_target`, triple-gated on the immutable PR author `dependabot[bot]`, a same-repo
  head, and a `dependabot/**` branch) auto-commits a `### Dependencies` changelog entry to each
  Dependabot PR branch — derived exclusively from SHA-pinned `dependabot/fetch-metadata`
  structured outputs (regex-allowlisted, fail-closed, never attacker-influenceable free text)
  and written through a contents-API PUT with the classic `PROJECT_METADATA_TOKEN` PAT so the
  required checks re-run on the amended PR — and adds the PR to Project #5 with nine documented
  bot-default fields, read-back verified (`scripts/github/autofill_dependabot_changelog.py`,
  `scripts/github/sync_dependabot_pr_metadata.py`). The workflow never checks out or executes
  PR-head content; a per-commit server-side authorship proof (every commit authored by
  `dependabot[bot]` with a verified signature) gates the write, and an idempotency-first
  `(#N)`-keyed replace-or-insert terminates the self-trigger loop and self-heals Dependabot
  force-pushes. `scripts/github/validate_project_metadata.py` now treats governed bot authors
  as a documented exempt class: linked-issue and milestone requirements are waived,
  compensated by enforcing the PR's own Project #5 membership and field completeness from the
  same snapshot (no silent bypass, per #184's design principle); human PRs are unchanged.
  `.github/dependabot.yml` now applies taxonomy-valid labels (`dependency: external`,
  `type: maintenance`, and a per-ecosystem `area:*` label) plus the maintainer as assignee —
  its previous `dependencies`/`automation` label names no longer exist in the taxonomy, which
  is why PR #192 arrived label-less. A new `scripts/check_privileged_workflow_safety.py` guard
  (run as a repository-hygiene job) mechanically enforces the privileged workflow's structural
  security invariants — no PR-head checkout, no `${{ }}` interpolation in run bodies,
  immutable-author gating (never `github.actor`), no persisted credentials, and no
  ambient `contents: write` token — so a future edit cannot silently reintroduce the classic
  `pull_request_target` injection vector. The empty `### Dependencies` heading above is now
  part of the Unreleased template, and stale workflow comments describing
  `PROJECT_METADATA_TOKEN` as a fine-grained PAT were corrected to match the authoritative
  governance record (a classic PAT: `project` + `public_repo` + `read:org`). Auto-merge for
  low-risk bumps was considered and deliberately deferred; every bot PR still requires the
  maintainer's merge click.

- Required coding agents to check the environment's approved out-of-sandbox permission path as
  the first remediation step after encountering an authorization barrier (#195), before trying
  workarounds or asking the maintainer to repeat an existing authorization. The new standing
  commitment explicitly preserves normal approval requirements and does not treat permission
  availability as authorization or broaden the approved action.
- Added a "Standing commitments to the maintainer" section to `AGENTS.md` capturing hard
  cross-session contracts — self-recording of new promises, done-means-done reporting, a
  CHANGELOG entry on every substantive PR, the standing-authorization-vs-four-gated-actions
  rule (only push, open-PR, merge, and release-tag require an explicit go-ahead), and calibrated
  claim reporting — so cold-start executor and cloud sessions that do not inherit local agent
  memory are bound by them, a "log the issue before touching the repo" ordering rule (issue →
  branch → implement → gate → document → PR), and a "floor, not ceiling" clause making clear the
  section is a minimum and that declining an obviously-correct action because it is unlisted is
  itself a defect. Also added "Canonical work-item workflow" and "Engineering discipline" sections
  consolidating the previously-uncodified standing commitments — sync-before-branch, gate-every-
  commit (incl. CHANGELOG and the commentary standard), exhaustive documentation sweep, scope and
  deliberate-exemption disclosure, default-to-logging-issues, self-verification of PR readiness
  with tooling, the full closure pass (read-back-verified lane transitions, `status:` label
  stripping, milestone checks, and branch/worktree pruning), milestone-verification discipline,
  diagnose-before-suppressing, and treating governance docs as negotiable rather than silently
  bypassed. Generalized the file's title from Codex-specific to any coding agent.
- Mechanically enforced the per-PR changelog contract (#184): a new
  `scripts/github/validate_changelog_update.py` gate, run as an `Enforce per-PR changelog
  updates` job in `.github/workflows/metadata-governance.yml`, fails any pull request that
  touches substantive paths (`src/`, `scripts/`, `docs/`, `configs/`, `.github/workflows/`)
  without updating `CHANGELOG.md`; a genuinely entry-free pull request declares a visible
  `changelog: not-needed -- <reason>` line in its body instead (markers quoted in code spans or
  fenced blocks are ignored). REST-only, consuming no shared GraphQL quota; tests cover the
  failure, exemption, and quoted-marker paths with mocked gh calls. `docs/governance/releases.md`
  and `CONTRIBUTING.md` now document the continuous-changelog contract and the exemption
  mechanism. Alongside this bundle (#182, handled as board edits with no repository diff),
  divergent Project #5 Target Release values on delivered issue↔PR pairs were reconciled to
  their curated pair-partner values with read-back verification.
- Aligned the `area:*`/`portfolio:*` label taxonomy with the Project #5 board's Repository Area
  and Portfolio Signal option sets (#237): each of the five area labels with no same-named board
  option (`cli`, `data`, `pipeline`, `quality`, `repository`) now carries a recorded per-label
  decision — all five are permanently human-set, with the rationale enumerated in
  `scripts/github/project_label_mapping.py`'s `UNMAPPED_LABELS` and pinned by a regression test.
  The portfolio-side sweep minted one label, `portfolio: governance`, mapped to the board's
  pre-existing Governance option (57 carrying items at audit time; filing #237 itself bounced on
  the missing label) — a label addition only, no board option was created or modified.
  `portfolio: case-study` and `portfolio: release` stay human-set as lifecycle markers. A new
  "Label-to-board-field alignment (#237)" section in `docs/governance/label-taxonomy.md` records
  the full decision table, the board-only option rationale, and the friction-based revisit
  trigger for the remaining label-less Portfolio Signal options;
  `docs/governance/github-project.md` and `github-metadata-automation.md` cross-reference it.

- Migrated both label-hygiene scripts — `scripts/detect_label_drift.py` and
  `scripts/sync_github_labels.py` — onto the shared GitHub API layer
  (`scripts/github/github_api.py`), deleting their private `_run_gh` and raw `subprocess.run`
  plumbing so their GraphQL-backed listings and `gh label create --force` mutations gain the
  layer's bounded secondary-rate-limit retries and fail-fast primary-limit classification. Both
  scripts now run a `QuotaMonitor` preflight, print a before/after/consumed GraphQL quota report
  to stderr on success and failure alike, and exit with the distinct quota code 3 — while
  defaulting to observe-only (`--min-graphql-quota` 0) so a drained quota pool never blocks a
  manual hygiene run. `run_gh` gained an optional `cwd` passthrough preserving the sync script's
  repository-inference pinning to the checkout root; `LabelDriftError` now subclasses
  `GitHubApiError`. Behavior parity was verified (byte-identical `--dry-run` output, unchanged
  exit codes 0/1/2 pinned by tests), with one deliberate error-path change: a `gh` failure in the
  sync script now exits 2 with a clean `error:` message instead of an uncaught
  `CalledProcessError` traceback. The transport inventory table in
  `docs/governance/github-metadata-automation.md#graphql-quota-stewardship` was updated for both
  rows, correcting the sync script's over-listed command shape from the #173 inventory.
  Closes #175 (#176).
- Hardened the governance scripts against GraphQL quota exhaustion by introducing
  `scripts/github/github_api.py`, a shared operation-scoped GitHub API access layer — `run_gh`
  with primary/secondary rate-limit classification (`PrimaryRateLimitError` distinct from
  ordinary failures), a `QuotaMonitor` REST-based quota preflight with a configurable
  `--min-graphql-quota` threshold and before/after/consumed reporting, and a `ProjectClient` with
  cached Project schema/identity, at most one cached board snapshot per instance, and
  never-cached targeted read-backs — and rebuilding `set_merged_project_status.py` and
  `validate_project_metadata.py` on it. Merge-time status sync now uses a bounded targeted
  `repository.pullRequest.projectItems` lookup and `node(id:)` field read-backs instead of full
  `gh project item-list --limit 500` board scans (measured live: ~400+ GraphQL points per merged
  PR down to ~5), the metadata validator keeps exactly one Project snapshot per run and moves
  native PR/issue reads to REST, quota conditions exit with a distinct code 3, and the
  read-back-verified mutation rule from #164/#170 is preserved untouched — only the scope of the
  verification read narrowed. Covered by 45 deterministic mocked-subprocess tests including the
  new `tests/scripts/test_github_api.py`; `docs/governance/github-metadata-automation.md` gained
  a GraphQL quota stewardship section (shared-pool model, measured transport inventory of
  repository-owned call sites, recovery runbook) and `docs/governance/github-project.md`'s manual
  runbook now uses the targeted read-back (#173, #174).
- Reworked `scripts/github/set_merged_project_status.py`'s Status mutation from fire-and-forget
  to read-back-verified: a new `fetch_item_status` helper reads the item's live Status via
  `gh project item-list` (a vanished item fails explicitly instead of reading as unset), and
  `set_status_merged` now treats gh's `no changes to make` error as inconclusive — success is
  only ever concluded from a fresh read-back showing `Merged`, with one bounded retry on any
  non-Merged read-back and the exact observed value reported on the second failure.
  `docs/governance/github-project.md` replaces its bare `gh project item-edit` example with a
  matching Fish `set_project_status_verified` function (mutation, read-back, compare, one bounded
  retry) plus the operating rule to run Project field mutations sequentially, never batched, and
  never accept `no changes to make` as proof a value was already set. Motivated by a live
  reproduction (gh 2.96.0) where nine back-to-back `gh project item-edit` calls all falsely
  reported `no changes to make` on genuinely new field values, each succeeding when re-run
  individually (#164, #170).
- Documented a convention in `docs/governance/github-project.md` for setting a bundling PR's own
  Milestone, Target Release, Workstream, Issue Type, and Portfolio Signal when its closing issues
  disagree on those Project #5 fields, grounded in the PR #155 and PR #160 precedents where such
  values had been set by undocumented ad hoc judgment (#162, #163).
- Added an observational, non-blocking check to `scripts/github/validate_project_metadata.py`
  that warns when a PR's `Closes #N` issue was already closed by a non-merge event while the PR
  itself remains open: `fetch_issue_closure_state` reads the issue's GitHub timeline
  (`gh api .../timeline`) and treats a `closed` event with a null `commit_id` as a manual close.
  Warnings are reported on a separate stderr channel and never join `validate_pull_request`'s
  violations tuple, so the new check cannot fail a merge gate. Motivated by issue #154, which was
  closed manually while its fixing PR #155 was still open, leaving its Project #5 Status stuck at
  `In Progress` after merge (#158, #160).
- Ported #156's primary/secondary GitHub rate-limit classification and bounded retry-with-backoff
  from `validate_project_metadata.py` into the sibling
  `scripts/github/set_merged_project_status.py`'s `_run_gh`, so a transient rate-limit exhaustion
  during the post-merge Project status sync surfaces as a clearly labeled infrastructure failure
  instead of a raw, unclassified `ProjectStatusSyncError`; all other failure modes and the
  script's Status-mutation logic are unchanged, with both new code paths covered by
  mocked-`subprocess.run` tests (#159, #160).
- Documented a narrow, audited maintainer override for required-check failures independently
  proven to be pure infrastructure (not the pull request's own doing): record the proof (e.g.
  `gh api rate_limit` output) in a PR comment, temporarily disable `enforce_admins` on `main`,
  merge, and immediately re-enable it — an accountable escape hatch, not a standing bypass. Added
  as the new "Maintainer override for confirmed infrastructure failures" section in
  `docs/governance/github-metadata-automation.md`, with the "Enforced for administrators" bullet
  in `docs/governance/repository-governance.md` updated to reference it. Closes #157 (#155).
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

- Refreshed the Project #5 Status option-ID table in `docs/governance/github-project.md` (#212)
  to the post-#208 generation — all eight pre-existing IDs plus the new `Not Planned` row, re-verified
  against the live field schema at fix time — and documented the `updateProjectV2Field` footgun
  discovered during that work: the mutation's `singleSelectOptions` argument replaces the whole
  option set and regenerates every option ID (even for options resent unchanged by name),
  orphaning every item's stored Status value board-wide; options must be added through the
  project web UI, with the per-item read-back-verified `gh project item-edit` loop as the
  restore path (the route used for the verified 209/209 board restore recorded on PR #209).
  Completes the follow-up commit PR #209's body promised but did not land before merge.
- Added a release-time changelog-coverage backstop to
  `docs/governance/release-checklist.md` (#186): enumerate the full
  `git log <last-release-tag>..main` commit range and confirm every commit's issue or pull
  request is represented in the changelog section being frozen, mapping PR-citing commit
  subjects to issue-citing entries before flagging gaps (the naive number match produced false
  positives), complementing #184's per-PR gate by also catching direct commits.
- Corrected `.claude/CLAUDE.md`'s directory-boundaries bullet (#183): the generated data,
  artifact, and report zones track not only `.gitkeep` placeholders but also `data/README.md`
  and the deliberately allowlisted `reports/figures/modern-pipeline-lineage.svg`, verified
  against `git ls-files data artifacts reports`.

- Corrected `README.md`'s "Current status" table, which still listed threshold analysis as not
  yet implemented: the validation-only centroid-distance margin threshold sweep shipped under #84
  (`configs/threshold-sweep-v1.toml`, `ecg-data evaluate-threshold-sweep`,
  `docs/threshold-sweep-analysis.md`) and `MODEL_CARD.md` already describes it as implemented.
  The table now claims only generated evaluation figures as the remaining candidate follow-up and
  lists the sweep as implemented with its existing caveats (a raw squared-distance margin, not a
  probability; no ROC/AUC or calibration claim; `validation` partition only) (#181).
- Documented the `scripts/detect_label_drift.py` exit-code contract and GraphQL quota reporting
  in `docs/governance/repository-hygiene.md`, the tool's primary operating doc for its weekly
  `repository-hygiene.yml` schedule: a new "Exit codes and quota reporting" subsection tabulates
  exit codes 0/1/2/3 with their stderr prefixes (`error:`, `quota:`), notes the observe-only
  `--min-graphql-quota` default of 0 and the two routes to exit 3, describes the
  `GraphQL quota: N before, N after, N consumed` stderr report printed on success and failure
  alike (a failed quota report downgrades to a `warning:` line without changing the exit code),
  and links to `github-metadata-automation.md#graphql-quota-stewardship` for the full quota
  inventory — so an operator triaging a failed weekly run from this doc alone can distinguish a
  drained shared GraphQL pool (exit 3, rerun after the hourly reset) from genuine drift (exit 1)
  or an execution failure (exit 2). Documentation only; no script or workflow behavior changed
  (#177, #178).
- Replaced the ASCII process-flow diagrams in `README.md`, `docs/pipeline-design.md`,
  `docs/governance/github-project.md`, and `docs/governance/github-metadata-automation.md` with
  four publication-quality visuals — implemented-pipeline-overview, local-flow-artifact-zones,
  governance-status-lifecycle, and governance-automation-overlay — committed as SVG with PNG
  fallbacks under `docs/diagrams/exports/`, each embedded with alt text, a caption, and a source
  attribution line while preserving the semantics of the ASCII originals (#103, #167).
- Extended issue #149's exhaustive docstring/comment standard to the three curated, public-facing
  notebooks' Python code cells (330 → 652 standalone comment lines across
  `00-environment-setup-and-artifact-generation.ipynb`, `01-narrative-walkthrough.ipynb`, and
  `02-high-performing-gradient-boosting-validation.ipynb`), with a `tokenize`-stripped diff
  confirming no logic changes. Refactored `scripts/check_code_commentary.py`'s `audit_file` into
  a reusable `audit_source` operating on in-memory source text, added
  `scripts/check_notebook_commentary.py` as a thin `nbformat`-based wrapper that statically
  audits each code cell without executing it, wired it into `.pre-commit-config.yaml` as a
  `notebook-commentary` hook, and recorded the completed inventory in
  `docs/code-commentary-audit.md` (#151, #153).
- Documented the selective `.claude/` tracking policy: project-level `CLAUDE.md` instructions and
  `settings.json` are retained in Git, while arbitrary tool-local state remains ignored. Markdown
  linting now explicitly includes `.claude/CLAUDE.md` despite excluding other local Claude files
  from its filesystem-based scan (#169).
- Signaled the shipped v1.0.0 release from `README.md`: every pre-existing `v1.0.0` mention in
  `README.md`/`MODEL_CARD.md` referred to the MIT-BIH dataset version, so a release marker
  linking the actual `v1.0.0` GitHub release and the changelog was added ahead of any
  dataset-version text. `docs/modernization-roadmap.md`'s Phase 7 note moved from prospective to
  retrospective phrasing, and `MODEL_CARD.md` gained a version-agnostic changelog/releases
  pointer so the card needs no edit on future releases (#92, #96).
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

- Documented the random-projection normalization in `training.py`'s `_fit()` and
  `docs/baseline-training.md`'s "Scope and boundary" section: dividing by
  `sqrt(projection_components)` is the standard Johnson-Lindenstrauss scaling that keeps the
  projection approximately distance-preserving, not an arbitrary constant. No change to the
  normalization value, training behavior, or dependencies (#140).

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
