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

# The one partition this future, not-yet-implemented execution config is scoped to.
# Deliberately named separately from evaluation.py's own SUPPORTED_PARTITION ("validation")
# constant -- the two are unrelated by design, since this module governs a different,
# still-disabled partition and must never be confused with the one evaluation.py
# actually reads from.
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


def load_held_out_config(path: Path, *, allow_execution: bool = False) -> HeldOutExecutionConfig:
    """Load and validate the held-out execution config without accessing data or model artifacts."""
    # Translate a missing, unreadable, or malformed-TOML file into HeldOutConfigError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise HeldOutConfigError(f"could not load held-out config {path}: {error}") from error
    values = document.get("execution")
    # schema_version pins this loader's understanding of the [execution] table's shape.
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
    # This module only describes the same frozen-baseline evaluator training.py
    # implements; a config naming any other estimator would be describing execution
    # behavior that doesn't exist anywhere in this package.
    if config.evaluator != SUPPORTED_ESTIMATOR:
        raise HeldOutConfigError(f"execution.evaluator must be {SUPPORTED_ESTIMATOR!r}")
    # This module governs exactly one partition, named "test" (this module's own
    # SUPPORTED_PARTITION); a config naming any other value describes a different,
    # unsupported gate.
    if config.partition != SUPPORTED_PARTITION:
        raise HeldOutConfigError(f"execution.partition must be {SUPPORTED_PARTITION!r}")
    # Routine callers retain the disabled-by-default boundary; the separately reviewed
    # evaluator is the sole caller allowed to load an explicitly enabled copy.
    if config.execution_enabled and not allow_execution:
        raise HeldOutConfigError("execution.execution_enabled must be false")
    # Symmetric to the check above: any future execution must require a recorded
    # approval (see benchmark_approval.py) rather than running unconditionally.
    if not config.requires_recorded_approval:
        raise HeldOutConfigError("execution.requires_recorded_approval must be true")
    return config


def _string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested `[execution]` field.

    Args:
        values: The parsed `[execution]` table to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise HeldOutConfigError(f"execution.{key} must be a non-empty string")
    return value.strip()


def _boolean(values: dict[str, Any], key: str) -> bool:
    """Require and return a strict boolean from the requested `[execution]` field.

    "Strict" here means TOML's actual boolean type, not a truthy value like the
    string "true" or the integer 1 -- these are governance flags this module fails
    closed on, so they shouldn't be ambiguous about what counts as enabled.

    Args:
        values: The parsed `[execution]` table to read from.
        key: The field name to extract.

    Returns:
        The field's boolean value.
    """

    value = values.get(key)
    # bool is checked with isinstance, not truthiness, so "true"/1/etc. are rejected.
    if not isinstance(value, bool):
        raise HeldOutConfigError(f"execution.{key} must be a boolean")
    return value
