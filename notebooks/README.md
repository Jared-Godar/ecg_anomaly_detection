# Notebooks

![Notebook Workflow — a guided path from setup to validation: environment and artifact generation, narrative walkthrough, validation-only model example](../docs/assets/ecg-notebook-workflow-banner.png)

This directory uses a small, ordered public notebook workflow.

## Recommended order

| Step | Notebook | Purpose |
|---:|---|---|
| 0 | [`00-environment-setup-and-artifact-generation.ipynb`](./00-environment-setup-and-artifact-generation.ipynb) | Fresh-clone setup, environment verification, local artifact preflight, governed pipeline execution, and remediation guidance. |
| 1 | [`01-narrative-walkthrough.ipynb`](./01-narrative-walkthrough.ipynb) | Supported modernization narrative: architecture, data lifecycle, governance, reproducibility, and repository positioning. |
| 2 | [`02-high-performing-gradient-boosting-validation.ipynb`](./02-high-performing-gradient-boosting-validation.ipynb) | Streamlined validation-only gradient boosting example using generated Step 0 artifacts. |

## Supported local execution

Run the complete 00 → 01 → 02 walkthrough from one local repository checkout in VS Code or
JupyterLab. Sync the locked notebook dependencies, select that checkout's `.venv` kernel in every
notebook, and keep the same checkout open across all three steps:

```fish
uv sync --locked --group notebooks
uv run --group notebooks jupyter lab notebooks/00-environment-setup-and-artifact-generation.ipynb
```

Notebook 00 ends with a visible local continuity confirmation. Notebooks 01 and 02 begin with a
matching confirmation before independently verifying the Step 0 files they need. Generated state
is reused in place; the notebooks do not copy it, upload it, or open the protected test partition.
See [environment reproducibility](../docs/environment-reproducibility.md#run-the-supported-local-notebook-workflow)
for kernel registration and troubleshooting.

Optional web-runtime integration is not a supported path in these notebooks. It remains a future
enhancement tracked by [Issue #200](https://github.com/Jared-Godar/ecg_anomaly_detection/issues/200).

## Boundary

Generated datasets, processed indexes, run manifests, metrics, and trained models are local artifacts and should remain ignored by Git unless a future issue explicitly changes that policy.

These notebooks do not establish clinical, diagnostic, production, deployment, or benchmark evidence.

## Validation

`.github/workflows/notebook-validation.yml` runs `scripts/validate_curated_notebooks.py` on every
pull request, executing all three curated notebooks above end to end in an isolated `git worktree`
copy of the repository. The copy is seeded with a small synthetic WFDB record set and a matching
acquisition manifest, so `acquire_dataset` takes its existing verify-and-reuse path and the real
MIT-BIH dataset is never downloaded. This is genuine cell-by-cell execution of the real, unmodified
notebooks — distinct from `scripts/notebook_quality.py`'s structural/hygiene-only check
(`ecg-data check-local-notebooks`), which validates notebook JSON and metadata without executing
any cell.

```fish
uv run python scripts/validate_curated_notebooks.py
```

Pass `--keep-worktree` to copy the isolated worktree to `.notebook-validation-worktree/` (gitignored)
before it is removed, for inspecting a local failure.

## Runtime feedback

Long or locally variable phases use concise, immediately flushed progress feedback. Step 0 reports
qualified expectations for dependency bootstrap and first-run pipeline generation while retaining
the CLI's per-stage stream and one integrity-verified acquisition line per configured record. Its
actual pipeline invocation is kept in a short cell so live output remains visible beneath the call.
Step 1 emits one start/completion pair only while scanning optional local run evidence. Step 2
reports bounded load, fit, and validation-score phases, plus one qualified elapsed-time heartbeat
per minute during the otherwise silent fit. Every estimate is deliberately approximate and names
the local factors that can change it; measured completion times are observational only and do not
alter artifacts, model settings, or evidence.

The committed public notebooks keep execution counts and outputs empty. Progress lines, metrics,
plots, and generated run artifacts remain local to an explicit execution and are not saved into the
tracked notebook files.

Each notebook opens with a conversational purpose statement, qualified first-run/rerun timing, and
one compact line of in-notebook jump links for returning users. Detailed version history is
preserved in a Markdown appendix at the bottom instead of competing with the first-run workflow.

### Presentation rendering

The public notebooks use repository-relative Markdown links and images so navigation, banners, the
lineage diagram, and accessible HTML callout panels render from the notebook's repository location.
To inspect the same Markdown through the supported Jupyter stack without executing cells, render
temporary HTML copies beside the source notebooks. Keeping the temporary HTML in `notebooks/`
during review is intentional: nbconvert preserves repository-relative Markdown image URLs, so the
HTML must retain the source notebook's directory context while it is inspected.

```fish
uv run --group notebooks jupyter nbconvert --to html --output-dir notebooks \
  notebooks/00-environment-setup-and-artifact-generation.ipynb \
  notebooks/01-narrative-walkthrough.ipynb \
  notebooks/02-high-performing-gradient-boosting-validation.ipynb
```

Open the temporary HTML files locally and confirm that each banner, the Step 1 lineage diagram,
repository navigation links, and the callout headings are visible. These rendered files are review
artifacts, not tracked documentation or execution evidence; remove them after review or move them
under the ignored `notebooks/local/` sandbox.
