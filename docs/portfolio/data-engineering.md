# Data Engineering

This page expands the [README's Data Engineering signal](../../README.md#data-engineering) into the
fuller story behind each bullet. The load-bearing contribution of the modernization is not a model
or a metric — it is the data-handling contract that any trustworthy future metric depends on. The 2022
experiment split individual beat windows randomly after they were created, so windows from the same
subject could land in both training and test sets; its saved numbers are flagged as likely inflated
for exactly this reason (see [historical results](../historical-results.md)). Everything below is the
engineering that closes that gap.

This is a research, education, and software-engineering demonstration. It is not medical software,
has not been clinically validated, and must not be used for diagnosis, monitoring, treatment, or
patient-care decisions.

## Subject-grouped splitting: the load-bearing fix

The split stage assigns whole subjects — never individual beat windows — to the train, validation,
and test partitions. Every record and every window belonging to a subject follows that single
assignment, which is what makes a leaked window structurally impossible rather than merely unlikely.

Record grouping alone would already keep one record out of two partitions, but it is insufficient:
the MIT-BIH directory identifies records 201 and 202 as originating from the same source tape, so a
record-level split could still separate two recordings of one person across the train/test boundary.
Split schema v2 handles this by requiring an explicit `[record_subjects]` table in
[`configs/splitting-v2.toml`](../../configs/splitting-v2.toml) that maps every configured record to
an opaque subject ID; records 201 and 202 share one ID, and the splitter requires the mapping to
cover the input records exactly. Subject identity is declared, not inferred — inferring it from
record IDs is precisely the kind of guess that produced the original leakage.

The algorithm (`create_split_manifest` in
[`splitting.py`](../../src/ecg_anomaly_detection/splitting.py)) sorts the unique subject IDs, applies
a local seeded shuffle that does not mutate global random state, apportions subject counts by the
largest-remainder method, and guarantees three non-empty partitions. Identical configuration and
window metadata reproduce identical membership. The `seed` lives in the config, so the same subject
partitions come back on every run. See the [design doc](../record-grouped-splitting.md) for the full
schema-v2 contract and its migration from the legacy per-record schema v1.

## Pairwise-disjoint enforcement

Grouping is only trustworthy if it is checked. `_validate_partitions` runs immediately after the
shuffle, before any manifest is written: it pairwise-intersects every partition's subject set against
every other's and raises `SplitError` on any overlap, then repeats the check at the record level, and
finally confirms the union of all partitions covers every input subject and record exactly. A bug in
the boundary arithmetic is caught here rather than silently propagating into a leaky split.

The same invariant is re-verified defensively at every trust boundary. When a manifest is read back
from disk (`SplitManifest.from_json`), `_validate_serialized_manifest` re-checks subject and record
disjointness, per-partition count consistency, and record-to-subject agreement — because a manifest
on disk may have been hand-edited or written by an incompatible tool. Window loading enforces the
upstream half of the guarantee: `load_window_metadata` rejects any record ID that appears in more
than one source NPZ artifact, since a record spanning two shards could not be assigned to a single
partition. Leakage-freedom is asserted at construction, at deserialization, and at ingestion.

## Typed WFDB ingestion with structural validation

Before any splitting, records enter through typed WFDB ingestion
([`records.py`](../../src/ecg_anomaly_detection/records.py),
[record validation](../record-validation.md)). Loading produces immutable `SignalRecord` and
`AnnotationSet` objects carrying the record ID, sample rate, physical-signal array, channel names,
and units — with no label mapping applied yet. A data-contract check runs on every record: the
sample rate must match the configured 360 Hz, physical signals must form a non-empty
samples-by-channels array of finite values, channel names and units must be present, unique, and
match the signal width, and annotation indices must be nonnegative, ordered, and inside the signal
boundary. A validation report names the completed checks; a failed rule returns a nonzero status.

### Channel identity resolved by name, not position

Window extraction selects the MIT-BIH `MLII` channel by name, resolved per record: for each record,
extraction inspects the available signal names, finds the configured channel, and uses that record's
local channel index for slicing ([window extraction](../window-extraction.md)). This replaced the
archived project's implicit assumption that the first loaded channel is always `MLII`. The
difference is concrete: records `102` and `104` have no `MLII` channel at all and are excluded, while
record `114` — which shared the historical `channel_index = 0` instability — does have an `MLII`
channel and needs no exclusion under name-based selection. Mixed channel identities such as `MLII`
and `V5` are treated as a reproducibility and lineage defect, never as equivalent signals; a missing
named channel fails as a data-contract error that reports the record and its available channel names.

## Closed-world label mapping

Annotation mapping ([`labels.py`](../../src/ecg_anomaly_detection/labels.py),
[annotation mapping](../annotation-mapping.md)) turns the original WFDB symbols into a versioned
binary target defined in [`configs/annotation-map-v1.toml`](../../configs/annotation-map-v1.toml):
`N` maps to `reference_normal` (0), and a selected group of beat symbols maps to `selected_other`
(1). The second target is deliberately named `selected_other`, not a diagnostic label — it exists to
reproduce the historical binary target, not to assert a clinical category.

The mapping is closed-world. It explicitly lists 24 excluded symbols, and any symbol that appears in
neither the target rules nor the exclusion list raises an error rather than being silently dropped or
assigned. A new upstream annotation therefore stops the pipeline for a human decision instead of
quietly corrupting the label set. Exclusions are counted in the audit report, and changing any symbol
assignment requires a new mapping version and regenerated downstream data.

## The closed 144-file SHA-256 inventory

The dataset contract is a closed set, enforced by hash. `configs/mitdb-v1.0.0.toml` defines 48 record
IDs and three required file types per record — `.hea` header, `.dat` signal, `.atr` annotations — for
an expected inventory of exactly 144 files ([data integrity](../data-integrity.md)). The committed
config pins each file's relative path, exact byte size, and SHA-256 digest. Inventory and verify
([`inventory.py`](../../src/ecg_anomaly_detection/inventory.py)) fail if any required file is missing,
unexpected, not a regular file, or differs from the pinned size or digest — any drift is a hard
failure, not a warning. All 144 SHA-256 values match the `SHA256SUMS.txt` that PhysioNet distributes
inside its versioned v1.0.0 ZIP. Because that checksum file is unsigned, the inventory pins expected
content and detects later change but is explicitly not cryptographic proof of publisher identity, and
no ECG files are ever committed to the repository.

## Split-quality acceptance thresholds

A leakage-free split can still be a bad split: it may hold too few subjects, records, windows, or
positive examples to support a meaningful evaluation. The split stage writes a separate
`split_quality_summary.json` with per-partition subject, record, shard, window, and represented-class
counts, explicit subject- and record-disjointness results, and configured-versus-actual ratios
([split quality reporting](../split-quality-reporting.md)). The `[split.quality]` table configures
minimum subjects, records, windows, and positive examples per partition, required classes and the
partitions that must cover them, and a maximum deviation between configured and actual subject
ratios.

Each check carries a severity. `default_severity` is `warning` or `failure`, and check names listed
in `warning_checks` are downgraded to warnings — the supported check names are `minimum_subjects`,
`minimum_records`, `minimum_windows`, `minimum_positive_examples`, `required_class_coverage`,
`partition_ratio_deviation`, `subject_disjointness`, and `record_disjointness`. Failure-level
violations are written to the summary and then stop the command before any indexing, training, or
evaluation runs. This workflow balances subject counts, not class or window stratification; it
exposes sparse partitions but does not prove population generalization, and it never scores the test
partition. Evaluation remains validation-only.

## Where to go deeper

- [Record-grouped splitting](../record-grouped-splitting.md) — the schema-v2 subject-split contract
  and migration from v1.
- [Split quality reporting](../split-quality-reporting.md) — the acceptance policy and diagnostics
  document.
- [Boundary-safe window extraction](../window-extraction.md) — window geometry and the channel
  identity contract.
- [Versioned annotation mapping](../annotation-mapping.md) — the closed-world target policy.
- [WFDB record ingestion and validation](../record-validation.md) — structural data-contract checks.
- [Local dataset inventory and integrity](../data-integrity.md) — the 144-file SHA-256 contract.
- [Historical results](../historical-results.md) — the 2022 leaky-split outputs, presented only as
  caveated historical evidence.
