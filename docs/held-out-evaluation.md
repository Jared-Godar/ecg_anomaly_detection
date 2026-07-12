# Held-out evaluation record

## Scope and status

Issue #73 completed the separately reviewed protected-test protocol for frozen candidate
`4c07d501-4dee-4543-98ee-1bc7aa99ffef`, generated from clean commit
`e3a767786a6a1aa27af9fcab4579acc03bc41e80`. `Jared-Godar` approved this specific execution before
protected data was opened.

The first invocation failed during output-parent validation, before any protected shard was opened
or result inspected. The failure remains in append-only rerun history. A new approval acknowledged
the prior attempt, and the policy-permitted retry completed without changing the candidate.

## Aggregate result

The frozen baseline scored 16,545 windows from seven records in the subject-grouped test partition.
Accuracy was `0.4427319432`; macro precision, recall, and F1 were `0.5290139607`, `0.6493400910`, and
`0.3617623923`. Class support was 15,730 for class `0` and 815 for class `1`. Class `0`
precision/recall/F1 was `0.9852415027`/`0.4201525747`/`0.5890899367`; class `1` was
`0.0727864186`/`0.8785276074`/`0.1344348479`.

These are bounded descriptive results for one frozen candidate and one grouped split. They do not
establish model quality, population generalization, clinical validity, medical utility, diagnostic
usefulness, or production-healthcare readiness.

## Reproducibility and disclosures

| Evidence | Identity or value |
|---|---|
| Repository commit | `e3a767786a6a1aa27af9fcab4579acc03bc41e80` (clean) |
| Dataset configuration | `configs/mitdb-v1.0.0.toml`; SHA-256 `4565da32eca2958ccd6747664c6e275a939da5b85fafca28360f88d8f2503ecd` |
| Split identity | `subject-aware-holdout` v2.0.0; seeded subject shuffle, seed 2022; record- and subject-disjoint |
| Training configuration | `configs/training-baseline-v1.toml`; SHA-256 `83438c339b93cefb7bda7fc1c38d0298ba39a14c2148465baf875222f43b6a96` |
| Evaluation configuration | `configs/evaluation-baseline-v1.toml`; SHA-256 `7ebd0fb5251dc504506615a8f3b4a309b50b5a41259e94e7166d17ea2985124d` |
| Reproducibility evidence | Evidence-manifest SHA-256 `47e7e2d9db4e1645f6bb7bfb44e56418e7884a31e742a76b7aa78aeab58c4242` |
| Runtime | Candidate pipeline `24.569549583` seconds; held-out command completed in under one second |
| Hardware | Apple M1, arm64, 8 logical cores, 16 GiB memory; Darwin 25.5.0; CPython 3.12.13 |
| Initial/retry approvals | SHA-256 `9cfbcbf5aeb6a57c6b8a8d35cb6e15806fdee7ad3e307a7e3d603ce318c59476` / `6544de9425ea92a311d8df7a5eb007896e5df7878d1555d338c8bdc2b0a6c102` |
| Rerun history | SHA-256 `d2f529be45c746f63bc69a56563e71800aa428a11914740fd74bc29707fbf34c` |
| Metrics/disclosure | SHA-256 `46b4e5b7a50d2285c7bb287e8dac3b56b6a4de8d5a95c7ab967eebe36382b433` / `b1a58e8d5e42cc519c87bd81584f6e6ab63b47ef670b10ba8690fb9bf66e5c77` |

Generated approval, metrics, disclosure, runtime/resource, and rerun-history files remain together
under the ignored run directory. Source ECG data, patient-level derived data, models, record
membership, and predictions are not committed.

## Assumptions and limitations

- MIT-BIH is small and historical, with limited cohort, device, site, and recording-context
  coverage; its annotations inherit source and expert-policy limitations.
- The binary mapping collapses heterogeneous rhythms and excludes some symbols.
- Severe class imbalance makes accuracy and aggregate metrics incomplete without per-class results.
- Subject grouping reduces direct leakage, but one split cannot establish performance across new
  populations, sites, devices, or clinical settings.
- Beat-centered windows may overlap within a partition; channel and window geometry remain
  repository-specific assumptions.
- Neither this result nor its governance establishes clinical validation or medical utility.
