# Third-party notices and attribution status

This repository contains project code, public-dataset references, and historical presentation materials. The root MIT license applies only to material owned by this repository's author.

## MIT-BIH Arrhythmia Database

The source dataset is distributed separately by PhysioNet under the Open Data Commons Attribution License v1.0. It is not included in this repository. See [data provenance](docs/data-provenance.md) for source, license, DOI, and citations.

## WFDB and Python dependencies

The supported package and development dependencies are declared in `pyproject.toml` and resolved in
`uv.lock`. They remain subject to their respective licenses; the repository's MIT license does not
relicense them. Dependencies used only by the archived 2022 notebooks are historical references and
are not part of the supported modern environment.

The supported runtime currently uses NumPy for typed array contracts and WFDB for local signal and
annotation file access. Their inclusion does not transfer ownership of the source dataset or imply
endorsement of this project for clinical use.

## Historical tutorial influence

`archive/original_2022/wrangle.py` identifies the Towards Data Science article “Detecting Heart Arrhythmias with Deep Learning in Keras” as sample code used while developing the original dataset workflow. The source file retains the original article URL.

The precise extent of adaptation has not yet been audited. That audit should occur before the pipeline refactor, with attribution retained where required.

## Images and notebook research material

The `archive/original_2022/images/` directory and historical notebooks contain diagrams, photographs, and background material assembled for the original educational presentation. Their authorship and reuse terms are not consistently recorded.

An attribution audit is complete; see [`archive/original_2022/ATTRIBUTION.md`](archive/original_2022/ATTRIBUTION.md) for the per-file inventory and [`archive/original_2022/PROVENANCE.md`](archive/original_2022/PROVENANCE.md) for the audit method and evidence. Of the 16 images audited, 7 are attributed to a specific external source (Christopher Olah's "Understanding LSTM Networks"), 2 are assessed as author-original, and 7 remain of unresolved provenance — disclosed explicitly rather than guessed at.

Regardless of attribution status:

- do not assume those assets are covered by the repository's MIT license;
- do not reuse them in new publications or generated documentation;
- retain them only as historical project material; and
- prefer replacement figures generated directly from appropriately licensed data.

The archive's [preservation policy](archive/original_2022/README.md#preservation-policy) excludes replacing or removing archived images. Attribution documentation, not replacement, is how this repository closes the gap for imagery that predates the modernization.
