# Notebooks

This directory contains the single supported [narrative walkthrough](narrative-walkthrough.ipynb).
It explains the modern workflow by loading validated package configuration, describing the
supported CLI, and optionally reading generated JSON evidence. It does not download data during
routine execution or duplicate acquisition, transformation, training, or evaluation logic.

The original notebooks are preserved in
[`archive/original_2022/`](../archive/original_2022/README.md). They remain available as historical
reference material but are not part of the supported modern workflow.

Local investigation and model-development notebooks belong in the ignored
[`notebooks/local/`](local/README.md) sandbox. That workspace is explicitly unsupported: its
notebooks, configurations, checkpoints, predictions, and result exports are not repository
evidence and must not be committed.

## Supported notebook contract

Supported notebooks must:

- have a numbered execution order and a single stated purpose;
- use repository-relative configuration;
- avoid embedding source data or model artifacts;
- contain no saved execution errors;
- delegate business logic to tested package modules or CLI commands;
- read generated evidence only when it exists and degrade cleanly when ignored local artifacts are
  absent;
- avoid routine full-dataset downloads during notebook validation; and
- repeat the project's research-only, non-clinical, non-production use limitation where results
  are presented.

The canonical notebook is expected to execute top-to-bottom without hidden state. The repository
does not yet install or run a notebook execution checker in CI; current validation covers notebook
JSON structure and the importable Python cell bodies. Data-independent automated notebook
execution remains follow-up work.

## Generated figures

Modern notebook figures belong in `reports/figures/`, must be deterministic and clearly identified
as supported modern assets, and must not reproduce pipeline logic. Generated local plots remain
ignored unless a specific reviewed figure is intentionally allowlisted. Historical images under
`archive/original_2022/images/` must not be reused or altered without verified source and reuse
terms.
