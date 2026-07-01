# Documentation guide

The maintained documentation is organized by the question a reader is trying to answer.

| Document | Purpose |
|---|---|
| [Project README](../README.md) | Portfolio overview, current status, quick start, and limitations |
| [Architecture](architecture.md) | Implemented repository boundaries and target component ownership |
| [Pipeline design](pipeline-design.md) | Proposed lineage, validation controls, run outputs, and cloud mapping |
| [Data provenance](data-provenance.md) | Dataset source, license, attribution, privacy, and label provenance |
| [Historical results](historical-results.md) | Saved 2022 metrics and their known evaluation defects |
| [Development workflow](development.md) | Locked environment, tests, hooks, and CI behavior |
| [Modernization roadmap](modernization-roadmap.md) | Completed, active, and planned modernization phases |
| [Contributing](../CONTRIBUTING.md) | Change scope, data safety, validation, and pull request expectations |
| [Third-party notices](../NOTICE.md) | Dataset, dependency, tutorial, and historical asset attribution status |

## Documentation rules

- Describe only tested behavior as implemented.
- Label unbuilt components and cloud services as proposed.
- Present the 2022 model output only as historical evidence with the record-leakage caveat.
- Keep source-dataset attribution and the research/educational-use limitation visible.
- Prefer links to generated evidence once the modern pipeline produces manifests and reports.
