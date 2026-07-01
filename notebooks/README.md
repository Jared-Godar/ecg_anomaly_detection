# Notebooks

This directory is reserved for curated notebooks that call the supported Python package rather than duplicate pipeline behavior.

No curated notebook is implemented yet. The directory contract is documented now so future
notebooks remain presentation and analysis layers rather than becoming a second pipeline.

The original notebooks are preserved in [`archive/original_2022/`](../archive/original_2022/README.md). They remain available as historical reference material but are not part of the supported modern workflow.

Curated notebooks should:

- have a numbered execution order and a single stated purpose;
- use repository-relative configuration;
- avoid embedding source data or model artifacts;
- contain no saved execution errors;
- call tested package functions for acquisition, transformation, and evaluation; and
- repeat the project's research-only, non-clinical use limitation where results are presented.
