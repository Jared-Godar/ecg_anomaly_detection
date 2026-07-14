# Local pipeline orchestration

![Technical banner for pipeline orchestration, showing repeatable execution flow motifs.](assets/ecg-pipeline-orchestration-banner.png)

## Scope

The supported orchestration command connects the implemented data stages through one
configuration-driven workflow. It acquires and inventories the configured dataset, validates every
record, maps annotations, extracts windows, assigns complete records to grouped partitions, and
writes an auditable run manifest, fits a deterministic training-only baseline, and evaluates the
frozen model on validation shards only.

The workflow does not open or report the indexed test partition. Validation metrics are exploratory
pipeline evidence and do not establish final model quality or clinical suitability.

## Run the supported workflow

From the repository root:

```fish
uv run ecg-data run-pipeline \
  --repository-root . \
  --dataset-config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --split-config configs/splitting-v2.toml \
  --training-config configs/training-baseline-v1.toml \
  --evaluation-config configs/evaluation-baseline-v1.toml
```

The first run retrieves the configured source files. Later runs verify and reuse the acquisition
baseline instead of redownloading unchanged files.

## Progress output

`run-pipeline` prints a start and completion banner for each of its seven reported stages
(acquisition, inventory, record processing, split, split diagnostics, training, validation
evaluation), with elapsed time and key counts or artifact paths on completion. The record-processing
stage additionally prints one indented line per record so long record loops do not appear frozen.
Acquisition uses the same record-level cadence after each record's required companion files pass
their size and digest checks. It distinguishes newly downloaded files, verified existing files, and
partial resume without expanding to noisy per-file output. Each acquisition record line also carries
a qualified timing suffix — the record's own measured duration, the acquisition phase's measured
elapsed time, and an `approx. remaining` projection — because a first 48-record network download is
the pipeline's one long wait whose remaining work is countable (see the audit below).
Representative output:

```text
run 32888ee3-7781-4171-b43e-e076a73b363c starting
[1/7] acquisition: starting (3 records, 9 files expected)
    record 1/3 (100): downloaded and verified 3 files | record 00:01 | elapsed 00:01 | approx. remaining estimating...
    record 2/3 (101): downloaded and verified 3 files | record 00:01 | elapsed 00:02 | approx. remaining estimating...
    record 3/3 (102): downloaded and verified 3 files | record 00:02 | elapsed 00:04 | approx. remaining 00:00
[1/7] acquisition: complete in 00:04 (manifest written to artifacts/datasets/synthetic/1.0.0/acquisition.json)
[2/7] inventory: starting
[2/7] inventory: complete in 00:01 (9 files verified)
[3/7] record_processing: starting (3 records)
    record 1/3 (100): 3 windows
    record 2/3 (101): 3 windows
    record 3/3 (102): 3 windows
[3/7] record_processing: complete in 00:02 (9 windows across 3 records)
...
completed run 32888ee3-7781-4171-b43e-e076a73b363c in 00:09: 3 records, 9 windows, manifest artifacts/runs/32888ee3-7781-4171-b43e-e076a73b363c/run-manifest.json
```

The projection method is deliberately simple and current-run-only: mean completed-record duration
(phase elapsed divided by records completed) multiplied by the records still outstanding. Until
three records have completed, the line reports an explicit `approx. remaining estimating...`
warm-up state instead of an unstable one-sample number; the final record reports `00:00`. The
record and elapsed values are measured observations, while the remaining value is inference and is
always labeled `approx.` — it is never a deadline, precise finish time, or guarantee. Verified
reruns reuse files in milliseconds, so their projections simply shrink toward zero; a single slow
outlier record shifts the mean rather than dominating the estimate.

A stage that raises prints `failed after MM:SS` instead of `complete in MM:SS`; the underlying
error still propagates and the command still returns a nonzero status. This output is purely
observational: it never affects run identity, evidence contents, or artifact schemas, and the
existing `runtime_summary.json` per-stage timings (see below) remain the authoritative timing
evidence.

### Timing-enrichment audit (issue #199)

Enriched timing is deliberately not applied everywhere. The following inventory covers every
intermediate progress/status site across the governed CLI pipeline and the three public notebooks,
with the decision and its basis, so future contributors extend timing only where a demonstrated
long wait has a defensible timing basis rather than decorating fast phases with timestamps:

| Progress site | Decision | Basis |
| --- | --- | --- |
| `run-pipeline` / `acquire` per-record acquisition lines | **Enriched timing** | The one long-running phase (minutes on a first 48-record network download) with countable, comparable remaining units; observed live in the #194 walkthrough. |
| Seven `run-pipeline` stage banners and the single-stage subcommand banners | No change | Completion banners already report measured elapsed time; the stages other than acquisition normally finish in seconds locally, below the threshold where projections help. |
| `record_processing` per-record window lines | No change | The whole phase completes in roughly a minute locally and each unit takes fractions of a second; three timing fields on 48 fast lines would be decoration, and the stage banner already reports total elapsed. |
| `list-runs` / `purge-run` | No change | Near-instantaneous local filesystem operations; they print no progress banners at all by design. |
| Notebook 00 bootstrap (`uv sync`) | Elapsed-only (existing) | Long-running on a cold cache but without countable remaining units (dependency count, wheel sizes, and cache state vary); it keeps its single qualified expectation and measured completion line, and uv streams its own per-package output. |
| Notebook 00 governed pipeline relay | Inherits enrichment | The relay streams the CLI's lines unchanged, so the acquisition timing suffix arrives automatically; the runner keeps its broad qualified first-run expectation and measured process-exit duration. |
| Notebook 01 run-evidence discovery scan | No change | One bounded start/completion pair around a directory scan that varies with accumulated local runs; already qualified, normally sub-second. |
| Notebook 02 partition load / fit / score stages | Elapsed-only (existing) | Long fit without defensible remaining units (no iteration-count observation surface worth coupling to); it keeps its three bounded stage pairs and the single restrained minute-scale heartbeat rather than a fabricated remaining projection. |

The shared abstraction behind the enriched acquisition lines is
`ecg_anomaly_detection.progress.UnitTimingEstimator` plus `format_unit_timing_suffix`, covered by
deterministic clock-injected tests; operations keep their own wording via the unit-label parameter.

Every other standalone subcommand except `list-runs` and `purge-run` prints the same shape of
banner around its own single-stage body: a `[1/1] <command>: starting` line, its existing
completion message, then a `[1/1] <command>: complete in MM:SS` (or `failed after MM:SS`) line.
See each stage's own documentation page for its exact output. `list-runs` and `purge-run` are
near-instantaneous local filesystem operations and do not print progress banners.

Every line is flushed as it is written. Python fully block-buffers stdout when it is not a
terminal, which is exactly the case for `notebooks/00-environment-setup-and-artifact-generation.ipynb`'s
Step 0 cell: it runs this command through `subprocess.Popen` specifically so a reviewer can watch
progress live. Without an explicit flush per line, every banner above would arrive in one batch at
process exit instead of live. The notebook streams the CLI's combined stdout/stderr line by line
from an intentionally short invocation cell, keeping live output close to the visible call in VS
Code and Jupyter while the preceding setup cell retains failure classification and verification
logic. Before starting, Step 0 also gives a broad, qualified first-run expectation and names
download speed, cache state, record count, CPU, and disk as variable factors; its measured
completion line is runtime feedback, not benchmark evidence.

## Output layout

Each invocation receives a UUID and creates isolated ignored output directories:

```text
artifacts/
├── datasets/<dataset>/<version>/acquisition.json
└── runs/<run-id>/
    ├── inventory.json
    ├── validation/<record-id>.json
    ├── mapping/<record-id>.json
    ├── windows/<record-id>.json
    ├── split.json
    ├── split_quality_summary.json
    ├── environment_summary.json
    ├── runtime_summary.json
    ├── resource_summary.json
    ├── evidence_manifest.json
    ├── training/
    │   ├── model.json
    │   └── training-metadata.json
    ├── evaluation/
    │   └── validation-metrics.json
    └── run-manifest.json

data/interim/runs/<run-id>/
└── windows/<record-id>.npz

data/processed/runs/<run-id>/
└── dataset-index.json
```

The evidence summaries capture the host and interpreter, Git and lockfile identity, stage timings,
best-effort host resources, and artifact digests. The final run manifest hashes every configuration,
evidence summary, report, window artifact, processed dataset
index, fitted model, training metadata, and validation metrics. The split manifest retains record
membership;
individual NPZ artifacts retain row-level record and annotation lineage. Only training shards are
opened by fitting, and only validation shards are opened by evaluation. Test descriptors are not
resolved, opened, scored, summarized, or reported.

This operational evidence supports reproducibility review. It does not prove generalization,
clinical validity, or medical utility. Validation metrics remain validation-only, and held-out
benchmark evaluation remains intentionally protected. Runtime and resource observations may vary
by host environment and system load.

## Local artifact lifecycle helpers

Iterative local runs accumulate `artifacts/runs/<run-id>/`, `data/interim/runs/<run-id>/`, and
`data/processed/runs/<run-id>/` directories, since [run directories are never reused or
overwritten](#execution-and-failure-behavior). `list-runs` and `purge-run` give an operator an
explicit way to inspect and reclaim that local disk space without touching governed, create-only
artifacts:

```fish
uv run ecg-data list-runs --repository-root .
uv run ecg-data purge-run --repository-root . --run-id <run-id> --dry-run
uv run ecg-data purge-run --repository-root . --run-id <run-id>
```

`list-runs` reports each run's ID, total size, directory count, and whether `run-manifest.json`
exists, newest first. `purge-run` removes exactly the three companion directories for one named
run ID — never `data/raw/`, never `artifacts/datasets/<dataset>/<version>/acquisition.json` (the
shared dataset acquisition baseline, which is not run-scoped), and never another run's
directories. `--dry-run` reports what would be removed and the bytes that would be freed without
deleting anything. A run ID must be a canonical lowercase UUID and must resolve to at least one
existing directory, or the command fails with a nonzero exit rather than silently doing nothing.

This is deliberately manual and explicit, not automatic cleanup: nothing here changes the
`run-pipeline` retry behavior described below, and it does not touch the acquisition baseline that
[dataset retrieval](dataset-acquisition.md) verifies and reuses across runs.

## Execution and failure behavior

Records are processed sequentially in configured order. This favors deterministic evidence and
bounded memory use over local parallelism. Each record's signal arrays are released before the next
record is loaded.

Run directories are never reused or overwritten. If a later stage fails, its partial UUID directory
remains available for diagnosis and the command returns a nonzero status. A retry creates a new run
ID while reusing only the verified dataset acquisition baseline. Automatic deletion is intentionally
avoided because it would remove useful failure evidence.

All configuration must be committed under `configs/`. Generated raw, interim, and artifact files
remain ignored. The automated end-to-end test writes three tiny synthetic WFDB records and replaces
the network transport; CI never downloads MIT-BIH data.

## Current limitations

- The workflow is local and sequential; no cloud orchestrator is implemented.
- The model-ready index references record shards rather than concatenating arrays.
- The v1 split balances record counts, not target distributions.
- Test-partition evaluation, model selection, and model-card generation remain unimplemented.

## Channel contract during artifact generation

The governed pipeline depends on window artifacts whose channel identity is explicit and consistent. During artifact generation, the public window configuration selects `MLII` by channel name instead of relying on positional index `0`.

This matters because some datasets may not expose the same signal identity at the same positional index for every record. The pipeline therefore treats mixed resolved channel identities as a data-contract failure. That failure is expected to stop model-ready artifact generation rather than producing a dataset index over inconsistent shards.

A successful pipeline run must produce model-ready artifacts only after the window extraction and shard identity contracts are satisfied, including:

- `data/processed/runs/<run-id>/dataset-index.json`;
- `artifacts/runs/<run-id>/split.json`;
- `artifacts/runs/<run-id>/split_quality_summary.json`; and
- `artifacts/runs/<run-id>/run-manifest.json`.

A clean failure message is useful diagnostics, but it is not equivalent to successful artifact generation. Notebook workflows that depend on these artifacts remain blocked until the governed pipeline produces them.
