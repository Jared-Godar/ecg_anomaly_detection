"""Tests for benchmark governance policy validation."""

from pathlib import Path
from typing import Any

import pytest

from ecg_anomaly_detection.benchmark_policy import (
    REQUIRED_DISCLOSURES,
    REQUIRED_PROHIBITED_CLAIMS,
    BenchmarkPolicyError,
    load_benchmark_policy,
)


def _policy_text() -> str:
    """Read the repository's real, committed configs/benchmark-policy-v1.toml as text.

    Tests mutate this text (via string replacement) to construct otherwise-valid
    policies with exactly one field violating the benchmark governance invariants.

    Returns:
        The committed benchmark policy file's full text.
    """

    root = Path(__file__).parents[2]
    return (root / "configs" / "benchmark-policy-v1.toml").read_text(encoding="utf-8")


def test_committed_benchmark_policy_parses_successfully() -> None:
    """The actual committed configs/benchmark-policy-v1.toml parses and disables test evaluation.

    Confirms the loaded policy's required-disclosure and prohibited-claim
    sets are supersets of the code's own REQUIRED_DISCLOSURES/
    REQUIRED_PROHIBITED_CLAIMS constants, not just that parsing succeeds.
    """

    root = Path(__file__).parents[2]
    policy = load_benchmark_policy(root / "configs" / "benchmark-policy-v1.toml")

    assert policy.policy_id == "benchmark-governance-v1"
    assert policy.protected_partition == "test"
    assert policy.test_evaluation_enabled is False
    assert policy.explicit_future_opt_in_required is True
    assert policy.required_disclosures >= REQUIRED_DISCLOSURES
    assert policy.prohibited_claims >= REQUIRED_PROHIBITED_CLAIMS


def test_policy_rejects_missing_required_disclosure(tmp_path: Path) -> None:
    """A policy missing the "runtime_summary" disclosure entry is rejected at load time.

    Every disclosure in REQUIRED_DISCLOSURES is a floor the on-disk policy
    must meet; a policy that drops one silently would weaken governance
    without anyone editing the code.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "policy.toml"
    path.write_text(
        _policy_text().replace('  "runtime_summary",\n', ""),
        encoding="utf-8",
    )

    # The "runtime_summary" disclosure line was stripped out above.
    with pytest.raises(BenchmarkPolicyError, match="runtime_summary"):
        load_benchmark_policy(path)


def test_policy_enforces_test_evaluation_disabled_by_default(tmp_path: Path) -> None:
    """A policy that flips test_evaluation_enabled to true is rejected outright.

    This is the primary governance gate mirroring held_out_config's
    execution_enabled check: no on-disk policy is permitted to enable
    test-partition evaluation.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "policy.toml"
    path.write_text(
        _policy_text().replace("test_evaluation_enabled = false", "test_evaluation_enabled = true"),
        encoding="utf-8",
    )

    # test_evaluation_enabled was rewritten to true above, which is never permitted.
    with pytest.raises(BenchmarkPolicyError, match="must be false"):
        load_benchmark_policy(path)


def test_policy_rejects_missing_prohibited_claim_category(tmp_path: Path) -> None:
    """A policy missing the "clinical_validity_established" prohibited-claim category is rejected.

    This category is the direct enforcement point of the "never described as
    diagnostic, clinical, or treatment-related" framing rule this repository
    is bound to; dropping it from the policy would remove that guardrail.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    path = tmp_path / "policy.toml"
    path.write_text(
        _policy_text().replace('  "clinical_validity_established",\n', ""),
        encoding="utf-8",
    )

    # The "clinical_validity_established" prohibited-claim line was stripped out above.
    with pytest.raises(BenchmarkPolicyError, match="clinical_validity_established"):
        load_benchmark_policy(path)


def test_policy_loader_reads_only_the_supplied_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_benchmark_policy opens exactly the path it was given, not some other discovered file.

    Wraps Path.open to record every path opened during the call, so a
    regression that fell back to a default/cached policy location instead
    of the caller's explicit path would be caught.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        monkeypatch: Used to substitute a tracking wrapper for Path.open.
    """

    path = tmp_path / "policy.toml"
    path.write_text(_policy_text(), encoding="utf-8")
    opened: list[Path] = []
    original_open = Path.open

    def tracked_open(candidate: Path, *args: Any, **kwargs: Any) -> Any:
        """Record candidate in opened, then delegate to the real Path.open.

        Args:
            candidate: The path being opened.
            args: Positional arguments to forward to the real Path.open.
            kwargs: Keyword arguments to forward to the real Path.open.

        Returns:
            Whatever the real Path.open returns for this call.
        """

        opened.append(candidate)
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    load_benchmark_policy(path)

    assert opened == [path]
