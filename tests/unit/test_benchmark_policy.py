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
    root = Path(__file__).parents[2]
    return (root / "configs" / "benchmark-policy-v1.toml").read_text(encoding="utf-8")


def test_committed_benchmark_policy_parses_successfully() -> None:
    root = Path(__file__).parents[2]
    policy = load_benchmark_policy(root / "configs" / "benchmark-policy-v1.toml")

    assert policy.policy_id == "benchmark-governance-v1"
    assert policy.protected_partition == "test"
    assert policy.test_evaluation_enabled is False
    assert policy.explicit_future_opt_in_required is True
    assert REQUIRED_DISCLOSURES <= policy.required_disclosures
    assert REQUIRED_PROHIBITED_CLAIMS <= policy.prohibited_claims


def test_policy_rejects_missing_required_disclosure(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(
        _policy_text().replace('  "runtime_summary",\n', ""),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkPolicyError, match="runtime_summary"):
        load_benchmark_policy(path)


def test_policy_enforces_test_evaluation_disabled_by_default(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(
        _policy_text().replace("test_evaluation_enabled = false", "test_evaluation_enabled = true"),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkPolicyError, match="must be false"):
        load_benchmark_policy(path)


def test_policy_rejects_missing_prohibited_claim_category(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(
        _policy_text().replace('  "clinical_validity_established",\n', ""),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkPolicyError, match="clinical_validity_established"):
        load_benchmark_policy(path)


def test_policy_loader_reads_only_the_supplied_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(_policy_text(), encoding="utf-8")
    opened: list[Path] = []
    original_open = Path.open

    def tracked_open(candidate: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(candidate)
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    load_benchmark_policy(path)

    assert opened == [path]
