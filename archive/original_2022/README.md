# Original 2022 project archive

This directory preserves the original educational ECG anomaly-classification project as it existed before the portfolio modernization began.

## Status

The files here are historical reference material. They are not part of the supported modern workflow, are not guaranteed to run in a current Python environment, and must not be interpreted as medical or clinical software.

Known limitations include:

- absolute paths tied to the original author's machine;
- an undocumented Python 3.8-era environment;
- random beat-window splitting that can place records and overlapping windows across train, validation, and test sets;
- a saved validation cell that compares predictions with themselves;
- duplicated exploratory code and saved notebook errors; and
- unresolved provenance for 7 of the 16 presentation images, disclosed explicitly in
  [`ATTRIBUTION.md`](ATTRIBUTION.md) rather than concealed.

See [historical results](../../docs/historical-results.md) for the evaluation audit and [data provenance](../../docs/data-provenance.md) for dataset licensing and attribution.

## Archived contents

| Path | Historical role |
|---|---|
| `report.ipynb` | Original canonical project narrative and saved final results. |
| `explore.ipynb` | Initial exploration, research notes, and modeling work. |
| `explore_model2.ipynb` | Extended model experiments with saved execution errors. |
| `model.ipynb` | Focused random-forest modeling scratchpad. |
| `eeg_explore.ipynb` | Alternate-dataset experiment using Challenge 2017 files. |
| `wrangle.ipynb` | Incomplete wrangling scratchpad. |
| `wrangle.py` | Original data loading, window generation, and random split code. |
| `images/` | Historical presentation images and diagrams; attribution audit complete. See [`ATTRIBUTION.md`](ATTRIBUTION.md) and [`PROVENANCE.md`](PROVENANCE.md). |
| `ATTRIBUTION.md` | Per-file attribution status inventory for `images/`. |
| `PROVENANCE.md` | Attribution audit method and provenance evidence for archived notebooks, `wrangle.py`, and `images/`. |

## Preservation policy

- Preserve these files rather than modernizing them in place.
- Record known defects instead of silently changing historical outputs.
- Keep the bundle together so its existing relative `images/` references continue to resolve.
- Do not add downloaded ECG recordings, derived CSV files, environments, or model artifacts.
- Build all supported replacement behavior under `src/`, `notebooks/`, and `tests/` at the repository root.

One original notebook reference expects `images/LSTM1.png`, while the tracked asset is named `images/LTSM1.png`. This mismatch is intentionally retained as part of the historical record.
