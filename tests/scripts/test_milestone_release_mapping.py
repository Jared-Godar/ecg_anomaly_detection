"""Tests for the milestone -> Target Release coherence table (issue #240).

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. The module is pure data plus pure
functions, so no test here mocks anything or touches the network.

The load-bearing tests are the pinning invariants: every table row's value
set must be a non-empty subset of the pinned Target Release option set, and
the set of enumerated milestone titles is pinned exactly -- so minting a
milestone (or renaming a board option) forces a conscious table decision
(and a reviewable diff) instead of silent drift. Unlike the label table's
completeness test, there is no tracked manifest file to compare against
(milestones live only on GitHub), so the pin is an explicit expected list
kept here in the test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Locate the module relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "milestone_release_mapping.py"
)
# Load the module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("milestone_release_mapping", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into.
mrm = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, so the
# governance scripts' own `import milestone_release_mapping` statements
# resolve to this same module object when their test files load them into
# one pytest process.
sys.modules[_SPEC.name] = mrm
_SPEC.loader.exec_module(mrm)

# The complete milestone-title pin: every milestone that exists on the
# repository at table-authoring time (verified live via GraphQL 2026-07-16).
# A new milestone must be added BOTH to the table and here -- the mirrored
# duplication is the forcing function, standing in for the tracked manifest
# the label-table completeness test compares against.
_EXPECTED_MILESTONE_TITLES: frozenset[str] = frozenset(
    {
        "M1 — Repository foundation",
        "M2 — Reproducible data pipeline",
        "M3 — Baseline modeling and evaluation",
        "M4 — Portfolio release",
        "M5 — Developer Experience and Execution Ergonomics",
        "M6 — Historical Stewardship and Provenance Closure",
        "M7 — Governance Completion",
        "M8 — Phase 6/7 Roadmap Completion",
        "M9 — Held-out Evaluation",
        "M10 — Post-1.0 CI and Portfolio Integrity",
        "M11 — Post-1.0 Release Optional Polish",
        "M12 — Optional Future Enhancements",
        "v1.0.0 — First Portfolio Release",
        "v1.1.0 — Second Portfolio Release",
    }
)


# --- table integrity ---------------------------------------------------------------


def test_table_enumerates_exactly_the_known_milestones() -> None:
    """The table's keys are precisely the pinned milestone-title set.

    A missing row would make the drift check report a live milestone as
    unknown; an extra row would be dead data masking a rename. Either way
    the fix is a conscious, reviewed edit to both the table and this pin.
    """

    assert frozenset(mrm.MILESTONE_TO_TARGET_RELEASES) == _EXPECTED_MILESTONE_TITLES


def test_every_row_is_a_nonempty_subset_of_the_option_pin() -> None:
    """Each milestone maps to at least one option, all drawn from the pinned set.

    An empty set would make every pairing incoherent (a contradiction, not a
    check), and a value outside the option pin would mean a board-side option
    rename orphaned the row.
    """

    # Walk every table row so a single bad entry names its own milestone.
    for milestone, coherent in mrm.MILESTONE_TO_TARGET_RELEASES.items():
        assert coherent, f"{milestone!r} maps to an empty coherence set"
        assert coherent <= mrm.TARGET_RELEASE_OPTIONS, (
            f"{milestone!r} maps outside the pinned Target Release options"
        )


def test_unmilestoned_set_is_a_nonempty_proper_subset_of_the_options() -> None:
    """The unmilestoned envelope is non-empty and excludes at least one option.

    If it ever grew to the full option set, the unmilestoned direction of the
    check would be vacuous -- every value would pass -- which is a design
    change that must be made deliberately, not by accretion.
    """

    assert mrm.UNMILESTONED_TARGET_RELEASES
    assert mrm.UNMILESTONED_TARGET_RELEASES < mrm.TARGET_RELEASE_OPTIONS


# --- coherence_problems behavior --------------------------------------------------


def test_neither_side_set_is_not_this_checks_concern() -> None:
    """An unmilestoned item with no Target Release produces no problem.

    Field completeness is the PR-time metadata gate's job; this check only
    judges pairs.
    """

    assert mrm.coherence_problems(None, None) == ()


@pytest.mark.parametrize("bucket", ["Stewardship", "Future"])
def test_unmilestoned_long_horizon_buckets_are_coherent(bucket: str) -> None:
    """Stewardship and Future are the two values allowed without a milestone."""

    assert mrm.coherence_problems(None, bucket) == ()


def test_unmilestoned_release_vehicle_bucket_is_flagged() -> None:
    """A release-arc Target Release without any milestone is incoherent.

    This is the reverse drift direction from issue #240's problem statement:
    the field claims a delivery vehicle that does not exist.
    """

    problems = mrm.coherence_problems(None, "Portfolio Release")
    assert len(problems) == 1
    # The message names the offending value and the allowed alternatives.
    assert "'Portfolio Release'" in problems[0]
    assert "no milestone" in problems[0]


def test_milestoned_with_unset_target_release_is_flagged() -> None:
    """A milestoned item whose Target Release is unset is an explicit finding."""

    problems = mrm.coherence_problems("v1.1.0 — Second Portfolio Release", None)
    assert len(problems) == 1
    assert "Target Release is unset" in problems[0]


def test_pair_outside_the_milestones_envelope_is_flagged() -> None:
    """A populated pair outside the enumerated envelope names both sides."""

    # v1.1.0's curated envelope is {Portfolio Release, Stewardship}; Future
    # (the direction issue #240's specimens drifted) is outside it.
    problems = mrm.coherence_problems("v1.1.0 — Second Portfolio Release", "Future")
    assert len(problems) == 1
    assert "'Future'" in problems[0]
    assert "v1.1.0" in problems[0]


def test_unknown_milestone_is_itself_the_finding() -> None:
    """A milestone with no table row is reported as needing a row, not skipped."""

    problems = mrm.coherence_problems("M99 — Not A Real Milestone", "Future")
    assert len(problems) == 1
    # The message points at this module so the fix lands in the right file.
    assert "milestone_release_mapping.py" in problems[0]


def test_every_enumerated_pair_is_coherent() -> None:
    """Every (milestone, option) pair the table blesses passes the check.

    Exhaustive over the table, so a row edit can never accidentally make the
    check reject a value the table itself declares coherent.
    """

    # Every row, every blessed option: the full cross-product must pass.
    for milestone, coherent in mrm.MILESTONE_TO_TARGET_RELEASES.items():
        # Each option in the row's set is a pairing the table declares fine.
        for option in coherent:
            assert mrm.coherence_problems(milestone, option) == ()
