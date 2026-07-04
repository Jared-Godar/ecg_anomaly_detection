# Versioning policy

## Version source

The project uses semantic versioning for the package and repository contract.
The current initial version is `0.1.0`, declared in `pyproject.toml`. A version
number in package metadata is not evidence that a Git tag, GitHub release, or
published Python distribution exists.

For a release, the package metadata, changelog heading, tag, and release notes
must identify the same version. Until a release is intentionally created, the
`Unreleased` changelog section describes work after the current metadata state.

## Increment rules

- **MAJOR**: breaking repository contracts, CLI interfaces, artifact schemas,
  configuration schemas, or reproducibility guarantees.
- **MINOR**: new, non-breaking pipeline, governance, documentation, or CLI
  capabilities.
- **PATCH**: fixes, clarifications, documentation corrections, and other
  non-breaking maintenance.

Dependency changes use the increment required by their observable contract
effect, not merely by the dependency's own version number.

## Pre-1.0 changes

Pre-1.0 releases may introduce contract changes as the modernization matures.
Such changes must still be explicit: release notes and the changelog must name
the affected interface or schema, describe migration or reproducibility impact,
and identify any compatibility break. Pre-1.0 status is not permission to make
silent breaking changes.

The maintainer should prefer the smallest increment that accurately communicates
the change. Version decisions must not imply production, clinical, medical, or
regulatory maturity.
