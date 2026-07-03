# Proposed pipeline and data lineage

## Status

This document is a target design. The repository currently implements the package boundary,
locked environment, directory contracts, expected MIT-BIH file inventory, local SHA-256 integrity
verification, typed WFDB ingestion, structural signal and annotation validation, tests, and CI
quality gates. A versioned annotation mapping with closed-world symbol validation and audit counts
and boundary-safe single-channel window extraction with row-level lineage are also implemented.
Deterministic record-grouped splitting and its machine-readable membership manifest are implemented.
Auditable run manifests now connect repository-relative evidence to code and environment identity.
Versioned, fail-safe HTTPS acquisition is also implemented. A local sequential orchestrator connects
all currently supported data stages. A processed dataset index validates and references grouped
record shards for lazy loading. Training and evaluation are not yet implemented in the supported
package.

## Target local flow

```text
PhysioNet MIT-BIH v1.0.0
          |
          v
acquire + verify checksums  ---> data/raw/ (immutable, ignored)
          |
          v
validate records + annotations ---> validation report
          |
          v
create labeled beat windows ---> data/interim/ (rebuildable, ignored)
          |
          v
group by record, then split ---> data/processed/ (model-ready, ignored)
          |
          +--------------------> split manifest
          v
train + evaluate ------------> artifacts/ (ignored)
                                  |
                                  +--> run manifest
                                  +--> machine-readable metrics
                                  +--> generated figures
```

Every transformation should accept versioned configuration and produce deterministic metadata.
Notebooks should consume package APIs and generated outputs rather than implement parallel data
pipelines.

## Stage contracts

| Stage | Required inputs | Validation boundary | Expected outputs |
|---|---|---|---|
| Acquire | Dataset URL, version, expected files | HTTP result, file inventory, checksum | Immutable raw files and acquisition manifest |
| Validate | Raw signals and annotations | Record ID, channel, sample rate, duration, annotation symbols | Validation report and accepted-record inventory |
| Window | Accepted records, label map, window configuration | Bounds, shape, finite values, excluded-label counts | Windows retaining record and annotation identity |
| Split | Window metadata and split configuration | No record crosses partitions; class/record counts reported | Train, validation, and test membership manifest |
| Train | Training partition and model configuration | Fit only on training data; deterministic seed recorded | Model artifact and training metadata |
| Evaluate | Frozen model and held-out partition | Tested metric definitions; no split mutation | Metrics, confusion matrix, and evaluation report |

## Run manifest minimum

The supported run manifest records:

- run identifier and UTC timestamp;
- Git commit and dirty-worktree status;
- Python and locked dependency versions;
- source dataset name, version, file inventory, and checksums;
- configuration content or digest;
- schema and label-map versions;
- record IDs and row/window counts by split;
- split random seeds; and
- artifact paths and checksums.

Model parameters and machine-readable metrics will be added when supported training and evaluation
stages exist.

The manifest is operational evidence, not a clinical validation record.

## Proposed cloud mapping

No cloud infrastructure is currently implemented. A future enterprise deployment could map the
same contracts to object storage for immutable data zones, an orchestrator for idempotent stages, a
container registry for pinned runtime images, a metadata catalog for lineage, and centralized logs
and metrics for operations. Cloud services should be selected only after the local contracts are
implemented and tested; the design should not require cloud resources to reproduce the portfolio
workflow locally.

Security controls for any future cloud implementation should include least-privilege identities,
encryption, private networking where appropriate, immutable audit logs, secret-manager integration,
retention rules, and explicit controls preventing source or derived ECG data from entering Git or
public build artifacts.
