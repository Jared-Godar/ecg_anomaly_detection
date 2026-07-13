"""Smoke tests for the installable package boundary."""

from importlib.metadata import version

import ecg_anomaly_detection


def test_package_is_installed() -> None:
    """The locked development environment installs the local package."""
    assert version("ecg-anomaly-detection") == "1.1.0"
    assert ecg_anomaly_detection.__doc__ is not None
    package_documentation = " ".join(ecg_anomaly_detection.__doc__.split())
    assert "not intended for clinical" in package_documentation
