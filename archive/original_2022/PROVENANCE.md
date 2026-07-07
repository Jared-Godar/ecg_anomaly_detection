# Historical archive provenance

This document records how the attribution audit in [`ATTRIBUTION.md`](ATTRIBUTION.md) was
performed, and the provenance evidence that exists for the archive's non-image material. It is a
documentation-only closure of the gap tracked by MOD-008; it does not modify any archived file.

## Audit method

The audit searched every archived notebook (`explore.ipynb`, `explore_model2.ipynb`,
`report.ipynb`, `eeg_explore.ipynb`, `model.ipynb`, `wrangle.ipynb`) for:

1. every `![...](images/...)` embed, to build the complete list of referenced image files;
2. the markdown and code cells immediately surrounding each embed, to capture any adjacent
   citation, URL, or caption; and
3. any URL or "sample code" acknowledgment in the notebooks or `wrangle.py`, to capture provenance
   of non-image material.

The resulting per-file status is recorded in `ATTRIBUTION.md`. Nothing in this process altered a
notebook, image, or script; the archive's [preservation policy](README.md#preservation-policy)
still applies unchanged.

## Notebook and dataset provenance

- **Primary dataset.** `explore.ipynb`, `explore_model2.ipynb`, and `report.ipynb` all cite the
  MIT-BIH Arrhythmia Database at <https://physionet.org/content/mitdb/1.0.0/> as the modeling
  dataset. This matches the source documented for the modern pipeline in
  [`docs/data-provenance.md`](../../docs/data-provenance.md); the archive's use of it predates and
  is independent of the modern acquisition path.
- **Alternate dataset.** `eeg_explore.ipynb` separately explores the PhysioNet Challenge 2017
  dataset (<https://physionet.org/content/challenge-2017/1.0.0/>) and references an external
  example repository, <https://github.com/darkbiologist/ECG-Anomaly-Detection-Using-Deep-Learning>,
  as inspiration for that exploration. This dataset and notebook were not carried forward into the
  supported modern pipeline.
- **Tutorial influence on `wrangle.py`.** Already documented in
  [`NOTICE.md`](../../NOTICE.md#historical-tutorial-influence): `wrangle.py` cites the Towards Data
  Science article "Detecting Heart Arrhythmias with Deep Learning in Keras" as the source of sample
  code used while developing the original data-loading and windowing logic. `NOTICE.md` notes that
  the precise extent of adaptation has not been separately re-audited beyond that self-reported
  citation; this document does not change that status.
- **Educational references.** Several markdown cells in `explore.ipynb` and `explore_model2.ipynb`
  cite external explanatory material by URL (for example, a YouTube anomaly-detection overview and
  scikit-learn's outlier-detection documentation). These are reading references, not embedded
  assets, and require no attribution action beyond the citations already present in the notebook
  text.

## Image provenance evidence

The one substantive, reusable citation found in the archived notebooks is Christopher Olah's blog
post "Understanding LSTM Networks" (<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>),
linked directly beneath the RNN diagrams in both `explore.ipynb` and `explore_model2.ipynb`. The
seven images attributed to that source in `ATTRIBUTION.md` (`RNN1.png`–`RNN4.png`, `LTSM1.png`,
`LSTM2.png`, `gate.png`) all appear within the same contiguous RNN/LSTM section as that citation
and depict the diagrams that post is known for (unrolled RNNs, short/long dependency gaps, and the
four-gate LSTM cell). No other embedded image has an adjacent citation, caption, or recognizable
external origin.

`example_waveforms-01.png` (and its uncommitted-reference vector counterpart,
`example_waveforms.svg`) is treated as author-original because it appears in `eeg_explore.ipynb`
immediately after cells that load and plot the author's own Challenge 2017 exploration — there is
no external image being embedded, only a plot the notebook itself produces.

## Unresolved provenance

Seven files — `ecg.jpeg`, `clinic.jpeg`, `watch.jpeg`, `labelec_ecg.png`, `thanks.png`, `RNN5.png`,
and `fft.png` — have no citation, caption, or naming evidence tying them to a specific source. This
audit does not guess at their origin. They remain historical artifacts of unknown provenance,
disclosed as such in `ATTRIBUTION.md` and governed by the same non-reuse guidance in `NOTICE.md`
that already applied to all archive imagery before this audit.

If provenance for any of these files is discovered later (for example, from the original author's
own records), update `ATTRIBUTION.md` and this file rather than the archived notebooks themselves.
