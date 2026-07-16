# Reproducibility

The README's Reproducibility signal makes one claim: every pipeline run produces
machine-readable lineage tying code state to artifacts. This page tells the fuller story behind
that claim — how runs are isolated, what the run manifest actually records, how the environment
is locked, and where determinism comes from. As everywhere in this repository, the evidence is
operational lineage for a research/education/software-engineering demonstration, not clinical
validation — the closing section spells out what it does and does not claim.

## UUID-isolated runs

Every `run-pipeline` invocation generates a fresh run identifier and validates that it is a
canonical lowercase UUID — non-canonical forms (uppercase, missing hyphens) are rejected so that
every directory path built from the ID is byte-for-byte predictable
([`pipeline.py`](../../src/ecg_anomaly_detection/pipeline.py)). The run ID keys three parallel
directory trees: `artifacts/runs/<run-id>/`, `data/interim/runs/<run-id>/`, and
`data/processed/runs/<run-id>/`. Because each run owns its own subdirectory in all three trees,
concurrent or repeated runs never share mutable state or overwrite each other's output. The only
cross-run shared state is the acquisition baseline — the raw dataset under `data/raw/` plus its
acquisition evidence under `artifacts/datasets/` — which is verified and reused rather than
re-downloaded; the `purge-run` command reclaims a single run's disk space by exact ID without
touching that baseline or any other run.

## Run manifests: one document ties everything together

The manifest stage ([`run_manifest.py`](../../src/ecg_anomaly_detection/run_manifest.py))
produces a schema-versioned JSON document linking a run to everything that influenced it:

- **Git state** — the full commit hash plus a dirty flag covering tracked and non-ignored
  untracked changes. When Git state cannot be captured, the manifest records the sentinel
  `revision = "unknown"` with `dirty = null` rather than guessing; the sentinel is provably not a
  40-character hex commit, so downstream lineage checks that require a real commit hash still fail
  closed.
- **Environment snapshot** — Python version and implementation, platform, machine architecture,
  and the version of every installed package, linked back to the committed dependency resolution
  through the `uv.lock` path and SHA-256 digest.
- **Dataset and split evidence** — dataset identity with every inventoried source file's name,
  size, and checksum, plus the grouped split's strategy, seed, versions, per-partition
  subject/record memberships, and quality summary.
- **File evidence for everything else** — repository-relative path, byte size, and SHA-256 digest
  for each configuration, evidence, and artifact file, so any later change to an input or output
  is detectable.

Two design decisions shape the format. Serialization is deterministic — sorted keys, fixed
indentation, trailing newline — so manifests diff cleanly. And absolute local paths and file
contents are never serialized; the manifest itself stays under the ignored `artifacts/` zone
because split membership and derived-artifact metadata are record-level data that the repository
deliberately does not distribute.

## Recorded reproducibility evidence

Beyond the manifest, each orchestrated run writes four evidence documents under its run directory
([`reproducibility.py`](../../src/ecg_anomaly_detection/reproducibility.py),
[design doc](../reproducibility-evidence.md)): an environment summary (OS, architecture, Python
runtime, optional `uv` version, `uv.lock` identity, best-effort Git commit/branch/dirty state), a
runtime summary with per-stage elapsed seconds, a resource summary (CPU model, core count,
memory, filesystem capacity), and an evidence manifest connecting the split identity to SHA-256
digests for configurations, operational reports, derived artifacts, the fitted baseline, and
validation metrics. The evidence manifest does not hash itself — that would be a circular digest —
so the run manifest hashes it instead, closing the chain.

The failure semantics are deliberate: optional platform values that a host cannot expose are
recorded as `null` without failing the run, while genuine contract violations — an output outside
`artifacts/`, a symbolic link, a duplicate input, a missing lockfile — fail evidence generation
outright. Runtime and resource figures are documented as host-dependent operational observations,
never as performance benchmarks.

## Locked environment with explicit dependency groups

The supported environments are declared in [`pyproject.toml`](../../pyproject.toml) and resolved
exactly by the committed `uv.lock`; all supported commands run through `uv sync --locked` and
`uv run`. Python is pinned to 3.12.13 via `.python-version`, with `requires-python >=3.12,<3.14`
bounding the supported range. Dependencies are split into a minimal core (`numpy`, `wfdb`) plus
three named groups — `dev` for repository engineering, `notebooks` for the supported Jupyter
stack, `experiments` for optional modeling libraries — each syncable independently.

The group structure is a repaired contract, not an aesthetic choice: earlier failures happened
because `uv sync` *succeeded* — it correctly removed packages absent from the declared graph and
exposed that the environment contract was incomplete. The documented rule now is that every
package lives in the narrowest owning group, added with `uv add` so declaration and lockfile
never drift ([environment reproducibility](../environment-reproducibility.md)).

## Versioned configs and deterministic seeds

All ten pipeline configuration files in [`configs/`](../../configs/) are TOML documents whose
first line declares a `schema_version`, covering dataset identity, annotation mapping, windowing,
splitting (v1 and v2), training, validation evaluation, held-out evaluation, threshold sweeps,
and the benchmark policy. Loaders validate the declared schema version, so a config cannot be
silently reinterpreted by newer code.

Determinism is seeded, not incidental. The active split config
([`splitting-v2.toml`](../../configs/splitting-v2.toml)) declares `seed = 2022` for its
`seeded-subject-shuffle` strategy, so repeated runs reproduce the same subject-disjoint
partitions ([record-grouped splitting](../record-grouped-splitting.md)). The baseline training
config ([`training-baseline-v1.toml`](../../configs/training-baseline-v1.toml)) declares the same
seed for its seeded-random-projection estimator, whose determinism comes entirely from that
config value: the same seed and features yield the same fitted model.

## What the evidence does and does not claim

Reproducibility evidence makes a run's inputs inspectable and later file changes detectable; it
does not prove generalization, clinical validity, or medical utility. Validation metrics remain
validation-only pipeline evidence, and the archived 2022 numbers are historical evidence flagged
with a leaky-split caveat, not a benchmark ([historical results](../historical-results.md)). The
held-out test partition sits behind a separately governed protocol — defined in advance, its
protected data opened and scored exactly once under explicit approval (an earlier invocation
failed before any protected shard was opened; the policy-permitted retry is recorded in
append-only rerun history), with the run manifest required lineage but not sufficient
authorization ([held-out evaluation record](../held-out-evaluation.md),
[benchmark governance](../benchmark-governance.md)). Neither the repository nor its releases
distribute the dataset, derived patient data, or trained models; every generated artifact stays
in ignored zones.

## Where to go deeper

- [Auditable run manifests](../run-manifests.md) — schema contents, CLI usage, interpretation
- [Reproducibility evidence](../reproducibility-evidence.md) — the four per-run evidence files
- [Local environment reproducibility](../environment-reproducibility.md) — dependency groups,
  interpreter validation, notebook kernels
- [Pipeline orchestration](../pipeline-orchestration.md) — how `run-pipeline` sequences the stages
- [Evaluation policy](../evaluation-policy.md) and
  [benchmark governance](../benchmark-governance.md) — why the test partition stays closed
