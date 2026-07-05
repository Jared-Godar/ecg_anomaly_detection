# Local notebook sandbox

This directory is an ignored workspace for local investigation, visualization, troubleshooting,
and model-development experiments. This README is the only supported, tracked file in the
directory.

## Boundary

Everything created here is local and disposable, including:

- notebooks and checkpoints;
- machine-local configuration;
- fitted models and serialized predictions;
- intermediate and final result exports; and
- experiment logs and run metadata.

These artifacts are not part of the supported package, canonical narrative notebook, run
manifests, reproducibility evidence, protected-test evidence, release artifacts, or portfolio
benchmarks. Do not commit source data, derived record-level data, credentials, absolute local
paths, or outputs containing record-level observations.

The repository's research-and-education-only limitation applies to all work performed here. Local
results must not be presented as clinical, diagnostic, monitoring, or treatment evidence.

## Environment

From the repository root, install the locked notebook kernel and optional experimentation tools:

```fish
uv sync --locked --group notebooks --group experiments
```

Verify the kernel, then select the environment's Python interpreter as the notebook kernel in an
editor:

```fish
uv run --group notebooks python -m ipykernel --version
```

The optional groups are deliberately separate from the default development environment. Their
presence does not make local experiments part of the supported pipeline.

## Working practices

- Use repository-relative paths and package APIs where practical.
- Keep the indexed test partition unopened unless separately authorized under evaluation policy.
- Save resumable local checkpoints after each completed candidate for long-running experiments.
- Treat local outputs as replaceable; preserve durable conclusions in reviewed documentation or
  tested package code.
- Before proposing a supported artifact, remove saved outputs and hidden state, verify provenance,
  document evaluation limitations, and move reusable logic into tested package modules.
