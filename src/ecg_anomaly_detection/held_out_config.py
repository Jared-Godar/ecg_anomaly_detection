"""Disabled-by-default configuration schema for a future held-out (`test`-partition) execution.

This module loads and validates `configs/evaluation-heldout.toml`. It never opens, reads, lists,
or scores a file under the `test` partition, never touches `evaluation.py`'s `SUPPORTED_PARTITION`
or evaluator behavior, and implements no execution command. It exists only so that a future,
separately reviewed execution command (tracked by a dependent issue) has a validated, fail-closed
place to plug into -- see docs/benchmark-governance.md.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecg_anomaly_detection.training import SUPPORTED_ESTIMATOR

# Centralize SUPPORTED_PARTITION so every caller shares the same documented invariant.
SUPPORTED_PARTITION = "test"


class HeldOutConfigError(ValueError):
    """Raised when the held-out execution config fails closed."""


@dataclass(frozen=True, slots=True)
class HeldOutExecutionConfig:
    """Validated, inert configuration for a not-yet-implemented held-out execution."""

    schema_version: int
    name: str
    version: str
    evaluator: str
    partition: str
    execution_enabled: bool
    requires_recorded_approval: bool


def load_held_out_config(path: Path) -> HeldOutExecutionConfig:
    """Load and validate the held-out execution config without accessing data or model artifacts."""
    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise HeldOutConfigError(f"could not load held-out config {path}: {error}") from error
    values = document.get("execution")
    # Evaluate `document.get('schema_version') != 1 or not isinstance(values, dict)` explicitly so
    # invalid or alternate states follow the documented contract.
    if document.get("schema_version") != 1 or not isinstance(values, dict):
        raise HeldOutConfigError(
            "held-out config must use schema_version = 1 and an [execution] table"
        )
    execution_enabled = _boolean(values, "execution_enabled")
    requires_recorded_approval = _boolean(values, "requires_recorded_approval")
    config = HeldOutExecutionConfig(
        schema_version=1,
        name=_string(values, "name"),
        version=_string(values, "version"),
        evaluator=_string(values, "evaluator"),
        partition=_string(values, "partition"),
        execution_enabled=execution_enabled,
        requires_recorded_approval=requires_recorded_approval,
    )
    # Evaluate `config.evaluator != SUPPORTED_ESTIMATOR` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.evaluator != SUPPORTED_ESTIMATOR:
        raise HeldOutConfigError(f"execution.evaluator must be {SUPPORTED_ESTIMATOR!r}")
    # Evaluate `config.partition != SUPPORTED_PARTITION` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.partition != SUPPORTED_PARTITION:
        raise HeldOutConfigError(f"execution.partition must be {SUPPORTED_PARTITION!r}")
    # Evaluate `config.execution_enabled` explicitly so invalid or alternate states follow the
    # documented contract.
    if config.execution_enabled:
        raise HeldOutConfigError("execution.execution_enabled must be false")
    # Evaluate `not config.requires_recorded_approval` explicitly so invalid or alternate states
    # follow the documented contract.
    if not config.requires_recorded_approval:
        raise HeldOutConfigError("execution.requires_recorded_approval must be true")
    return config


def _string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested structured field.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise HeldOutConfigError(f"execution.{key} must be a non-empty string")
    return value.strip()


def _boolean(values: dict[str, Any], key: str) -> bool:
    """Require and return a strict boolean from the requested structured field.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, bool)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(value, bool):
        raise HeldOutConfigError(f"execution.{key} must be a boolean")
    return value
