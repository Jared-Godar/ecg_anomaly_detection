# Historical archive provenance

This document records how the attribution audit in [`ATTRIBUTION.md`](ATTRIBUTION.md) was
performed, the provenance evidence that exists for the archive's non-image material, and the
`wrangle.py` code-adaptation audit (tracked by MOD-008, #35, and #74). It is a
documentation-only closure of those gaps; it does not modify any archived file.

## Audit method

The image and notebook-citation audit searched every archived notebook (`explore.ipynb`,
`explore_model2.ipynb`, `report.ipynb`, `eeg_explore.ipynb`, `model.ipynb`, `wrangle.ipynb`) for:

1. every `![...](images/...)` embed, to build the complete list of referenced image files;
2. the markdown and code cells immediately surrounding each embed, to capture any adjacent
   citation, URL, or caption; and
3. any URL or "sample code" acknowledgment in the notebooks or `wrangle.py`, to capture provenance
   of non-image material.

The separate `wrangle.py` code-adaptation audit (#74) retrieved the cited article and the GitHub
repository and notebook it links to, then compared `wrangle.py`'s functions, variable names, and
parameter values against that source line by line. See
[Code provenance evidence](#code-provenance-evidence) for the result.

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
- **Tutorial influence on `wrangle.py`.** `wrangle.py` cites the Towards Data Science article
  "Detecting Heart Arrhythmias with Deep Learning in Keras" (Andrew Long) as the source of sample
  code used while developing the original data-loading and windowing logic. The adaptation extent
  is now audited; see [Code provenance evidence](#code-provenance-evidence) below.
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

## Code provenance evidence

`wrangle.py`'s header comment cites the Towards Data Science article "Detecting Heart Arrhythmias
with Deep Learning in Keras with Dense, CNN, and LSTM" by Andrew Long. The live article URL
returns 404; the article was located via its canonical Medium mirror
(<https://medium.com/towards-data-science/detecting-heart-arrhythmias-with-deep-learning-in-keras-with-dense-cnn-and-lstm-add337d9e41f>),
which links to the author's own source repository,
<https://github.com/andrewwlong/deep_arrhythmias> (`Arrythmia Project.ipynb`). That notebook, not
just the article's prose, is the actual comparison source.

**Directly adapted.** `wrangle.py`'s `load_ecg`, `make_dataset`, and `build_XY` functions match the
source notebook's same-named functions line for line: identical signatures, identical variable
names (`p_signal`, `atr_sym`, `atr_sample`, `df_ann`, `num_cols`, `max_rows`, `max_row`), identical
control flow, and identical assert messages (`'sample freq is not 360'`,
`'number of X, max_rows rows messed up'`, `'number of X, Y rows messed up'`,
`'number of Y, sym rows messed up'`). The only differences are cosmetic: the source's inline `#`
comments become a docstring in `wrangle.py`, and spacing after commas differs (PEP 8 style versus
the source's compact style). The `pts` patient-ID list and the `nonbeat`/`abnormal` annotation-symbol
lists are also identical in content, order, and line-wrapping to the source notebook's own lists.
The `num_sec = 3` and `fs = 360` parameters match the source exactly.

**Not adapted from this source.** `wrangle.py`'s `split_my_data` function is original to the 2022
project, not the cited article. The source notebook splits by patient identity
(`pts_train = random.sample(pts, 36)`, evaluating on the disjoint remainder) specifically to avoid
one patient's beats appearing in both training and validation data. `split_my_data` instead performs
an ordinary beat-level `sklearn.model_selection.train_test_split` across `X_all`/`Y_all`/`sym_all`
with no patient grouping — the leakage risk already disclosed in
[historical results](../../docs/historical-results.md). The cited source's own split methodology
was not carried over, even though its data-loading and windowing code was.

**Attribution status.** The Medium article and its linked source code are the author's original,
copyrighted work; this repository has no license to redistribute or relicense them. `wrangle.py`'s
existing header citation is retained as-is (the archive is immutable — see the preservation policy
below), and this document supplies the fuller attribution the header comment alone did not: author
name, the source repository URL, and the specific functions and values adapted. `wrangle.py` is
historical archive material only; the supported modern pipeline does not import or reuse it.

## Unresolved provenance and disposition

Seven files — `ecg.jpeg`, `clinic.jpeg`, `watch.jpeg`, `labelec_ecg.png`, `thanks.png`, `RNN5.png`,
and `fft.png` — originally had no citation, caption, or naming evidence tying them to a specific
source, and the initial audit did not guess at their origin. A dedicated provenance research pass
on 2026-07-19 (PR #258) then resolved two of them to the inventory's confirmed-match standard —
`RNN5.png` (exact SHA-256 match to Christopher Olah's "Understanding LSTM Networks") and
`watch.jpeg` (visual match to a 2021 How-To Geek article) — and recorded an honest failed attempt
for the other five.

The maintainer's 2026-07-19 disposition decision (recorded in the portfolio program's control
workspace, a private repo: <https://github.com/Jared-Godar/github-portfolio-modernization/issues/35>)
then settled those five, bringing the inventory to its final state of
9 attributed / 2 author-original / 3 retained-by-decision / 2 removed:

- **Retained by decision (3):** `labelec_ecg.png`, `thanks.png`, `fft.png` — low-risk residuals,
  kept as historical artifacts of unknown provenance: disclosed in `ATTRIBUTION.md`, excluded
  from the MIT license per `NOTICE.md`, research attempts documented per file.
- **Removed from HEAD (2):** `ecg.jpeg`, `clinic.jpeg` — unidentifiable stock-style photos whose
  licensing exposure could not be resolved by attribution. Both intentionally remain in git
  history (no history rewrite); their `ATTRIBUTION.md` rows are the removal record, and the
  archived notebooks' references to them now dangle as part of the historical record.

If provenance for any retained or removed file is discovered later (for example, from the original
author's own records), update `ATTRIBUTION.md` and this file rather than the archived notebooks
themselves.
