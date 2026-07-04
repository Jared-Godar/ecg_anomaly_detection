# Release governance

## Purpose and current state

Releases are versioned engineering portfolio artifacts: they provide a stable,
reviewable snapshot of the repository's source, configuration, documentation,
and reproducibility contracts. They are not a distribution channel for data,
models, benchmark results, or medical functionality.

The package metadata currently starts at `0.1.0`. The repository has no tags or
GitHub releases. This policy documents how a future release should be reviewed;
it does not authorize or automate a tag, GitHub release, or package publication.

A release does not imply production readiness, clinical suitability, medical
utility, or regulatory compliance. This project remains a research, education,
and software-engineering demonstration and must not be used for diagnosis,
monitoring, treatment, or patient-care decisions.

## Release boundary

A future release source archive should include:

- source code and tests;
- versioned configuration;
- dependency and environment lockfiles;
- governance and technical documentation;
- the changelog and release notes; and
- references to the repository's reproducibility policies and evidence model.

It should exclude:

- raw or generated datasets and patient-level data;
- trained models and benchmark outputs;
- local run manifests or other machine-local evidence;
- caches and temporary files; and
- built package artifacts unless publishing those artifacts is an intentional,
  separately reviewed part of that release.

Repository source archives and Python distributions are distinct artifacts.
Creating a GitHub release does not by itself approve publishing a wheel or source
distribution to a package index. Any future package publication requires an
explicitly scoped governance and implementation change.

## Release decision and evidence

The maintainer decides whether a repository state is release-ready after the
[release checklist](release-checklist.md) is complete. The proposed version and
scope must follow the [versioning policy](versioning.md). Release notes should
summarize user-visible changes, compatibility effects, known limitations, and
reproducibility implications without presenting historical or validation-only
metrics as evidence of generalization.

The changelog is prepared in the pull request that makes the repository ready
for release. Tagging and publication remain separate, deliberate operations and
must not occur as an incidental effect of merging a governance change.

## Artifact hygiene

The root `.gitignore` excludes `dist/`, so locally built wheels and source
distributions remain outside version control. A release review must use
`git ls-files dist` to confirm that built artifacts have not been committed. If
they are tracked in the future, remove them from Git in a reviewed change and
retain an ignore policy; do not delete local artifacts blindly.

Raw data, derived patient-level data, trained models, generated reports, and
local manifests remain governed by the repository's existing data and artifact
boundaries regardless of release status.
