# Third-party notices and attribution status

This repository contains project code, public-dataset references, and historical presentation materials. The root MIT license applies only to material owned by this repository's author.

## MIT-BIH Arrhythmia Database

The source dataset is distributed separately by PhysioNet under the Open Data Commons Attribution License v1.0. It is not included in this repository. See [data provenance](docs/data-provenance.md) for source, license, DOI, and citations.

## WFDB and Python dependencies

The project uses third-party Python packages under their respective licenses. A dependency inventory and lock file will be added during the reproducible-environment phase.

## Historical tutorial influence

`archive/original_2022/wrangle.py` identifies the Towards Data Science article “Detecting Heart Arrhythmias with Deep Learning in Keras” as sample code used while developing the original dataset workflow. The source file retains the original article URL.

The precise extent of adaptation has not yet been audited. That audit should occur before the pipeline refactor, with attribution retained where required.

## Images and notebook research material

The `archive/original_2022/images/` directory and historical notebooks contain diagrams, photographs, and background material assembled for the original educational presentation. Their authorship and reuse terms are not consistently recorded.

Until the attribution audit is complete:

- do not assume those assets are covered by the repository's MIT license;
- do not reuse them in new publications or generated documentation;
- retain them only as historical project material; and
- prefer replacement figures generated directly from appropriately licensed data.

The modernization will either document a source and compatible license for each retained asset, replace it with an original asset, or propose its removal in a dedicated reviewable commit.
