# Data provenance

## Source

This project uses the MIT-BIH Arrhythmia Database, version 1.0.0, distributed by PhysioNet.

- Landing page: <https://physionet.org/content/mitdb/1.0.0/>
- DOI: <https://doi.org/10.13026/C2F305>
- Published version date: February 24, 2005
- Access policy: open access under the specified upstream license
- Data license: Open Data Commons Attribution License v1.0

PhysioNet describes the dataset as 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects. Signals were digitized at 360 samples per second, and the database provides reference beat annotations.

## Required attribution

Work using the data should cite both the database publication and PhysioNet:

1. Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45-50.
2. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215-e220.

Refer to the dataset landing page for authoritative citation formats and current access terms.

## Repository handling policy

Raw waveform and annotation files are intentionally excluded from Git. Derived tables, cached downloads, and trained artifacts should also remain excluded unless a future artifact has an explicit redistribution review and a documented reason to be versioned.

The repository layout distinguishes:

- `data/raw/`: immutable upstream files;
- `data/interim/`: resumable intermediate transformations;
- `data/processed/`: model-ready, reproducible outputs; and
- `artifacts/`: generated models, metrics, and run manifests; and
- `reports/figures/`: reproducible generated figures selected for reporting.

Each generated dataset should carry a manifest containing at least the source dataset name and version, retrieval timestamp, checksums, transformation configuration, code revision, schema version, row counts, and record identifiers included in each split.

The required lineage fields and stage boundaries are expanded in the
[proposed pipeline design](pipeline-design.md). Those contracts are design requirements until the
corresponding package modules and tests are implemented.

## Privacy and responsible use

The upstream database is publicly distributed under its stated access policy. Public availability does not make the signals project-owned or suitable for unrestricted repackaging. This repository links to the authoritative source rather than redistributing recordings.

No attempt should be made to identify subjects or combine the records with identifying information. Any future dataset must receive a separate privacy, license, and access review rather than inheriting the assumptions documented here.

## Label provenance

The original project maps `N` annotations to normal, maps a selected list of annotation symbols to abnormal, and discards annotations outside those lists. This binary target is a project-specific simplification, not a diagnosis and not the full upstream annotation taxonomy.

The modernization must make that mapping a versioned configuration, validate every encountered annotation, and report excluded counts. Changes to the mapping must create a new derived-dataset version.
