"""Tests for the shared label -> Project #5 field/option mapping (issue #233).

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`. The module is pure data plus pure
functions, so no test here mocks anything or touches the network.

The load-bearing test is the manifest-completeness invariant: every label in
`.github/labels.json` must be explicitly mapped, explicitly listed as
deliberately unmapped, or belong to a namespace declared as having no board
counterpart -- so growing the taxonomy forces a conscious mapping decision
(and a reviewable diff) instead of silent drift.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Locate the module relative to this test file (not the current working
# directory), so the test suite works regardless of where pytest is invoked from.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "github" / "project_label_mapping.py"
)
# Load the module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("project_label_mapping", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into.
plm = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, so the
# governance scripts' own `import project_label_mapping` statements resolve
# to this same module object when their test files load them into one
# pytest process.
sys.modules[_SPEC.name] = plm
_SPEC.loader.exec_module(plm)

# The canonical label manifest the mapping must stay complete against.
_MANIFEST_PATH = Path(__file__).resolve().parents[2] / ".github" / "labels.json"


def _manifest_label_names() -> list[str]:
    """Load every canonical label name from the repository's label manifest.

    Returns:
        The label names declared in .github/labels.json, in manifest order.
    """

    data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return [label["name"] for label in data["labels"]]


# --- table integrity ---------------------------------------------------------------


def test_every_manifest_label_has_an_explicit_mapping_decision() -> None:
    """Every canonical label is mapped, listed as unmapped, or in a non-field namespace.

    This is the anti-drift invariant: adding a label to .github/labels.json
    without deciding its board translation must fail here, forcing the
    decision into the mapping module where it is reviewable.
    """

    undecided = []
    # Every manifest label must fall into one of the three decision buckets.
    for name in _manifest_label_names():
        # First bucket: the label is explicitly mapped to a field option.
        if name in plm.LABEL_TO_FIELD_OPTION:
            continue
        # Second bucket: the label is explicitly, deliberately unmapped.
        if name in plm.UNMAPPED_LABELS:
            continue
        # Third bucket: the whole namespace has no board-field counterpart.
        if any(name.startswith(namespace) for namespace in plm.NON_FIELD_NAMESPACES):
            continue
        undecided.append(name)
    assert undecided == []


def test_mapped_and_unmapped_labels_are_disjoint_and_canonical() -> None:
    """No label is both mapped and unmapped, and every entry names a manifest label.

    A mapping or exclusion for a label that is not (or no longer) in the
    manifest is stale table content that would silently never fire.
    """

    manifest = set(_manifest_label_names())
    mapped = set(plm.LABEL_TO_FIELD_OPTION)
    # A label cannot simultaneously be mapped and deliberately unmapped.
    assert mapped.isdisjoint(plm.UNMAPPED_LABELS)
    # Every table entry (either side) must refer to a live canonical label.
    assert mapped <= manifest
    assert manifest >= plm.UNMAPPED_LABELS


def test_mapping_only_targets_the_declared_derivable_fields() -> None:
    """Every mapping targets a declared derivable field -- never Workstream/Target Release.

    Workstream and Target Release are an explicit automation non-goal
    (issue #233): they have no label source, so any mapping entry targeting
    them would be exactly the heuristic inference the design rules out.
    """

    targeted = {field for field, _option in plm.LABEL_TO_FIELD_OPTION.values()}
    assert targeted <= set(plm.DERIVABLE_FIELDS)
    # The non-goals stated by name, so this test reads as the governance rule.
    assert "Workstream" not in targeted
    assert "Target Release" not in targeted
    assert "Status" not in targeted


def test_area_and_portfolio_exclusions_record_the_237_alignment_decision() -> None:
    """The #237 alignment audit's permanent exclusions are all present in UNMAPPED_LABELS.

    Issue #237 audited every area label without a same-named Repository Area
    option and both lifecycle portfolio labels, recording each as permanently
    human-set. This test pins that decision: removing any of these from
    UNMAPPED_LABELS (without mapping it) must be a deliberate, reviewed
    reversal, not an incidental edit.
    """

    decided_unmapped = {
        # The five area labels with no unambiguous Repository Area translation.
        "area: cli",
        "area: data",
        "area: pipeline",
        "area: quality",
        "area: repository",
        # The two lifecycle/artifact-type portfolio labels with no same-named
        # Portfolio Signal option.
        "portfolio: case-study",
        "portfolio: release",
    }
    assert decided_unmapped <= plm.UNMAPPED_LABELS


def test_mapping_table_is_immutable() -> None:
    """The mapping is a read-only view: consumers cannot mutate the shared table."""

    # A MappingProxyType rejects item assignment with TypeError; anything else
    # means the shared table is silently mutable.
    try:
        plm.LABEL_TO_FIELD_OPTION["type: bug"] = ("Issue Type", "Feature")  # type: ignore[index]
    except TypeError:
        return
    raise AssertionError("LABEL_TO_FIELD_OPTION accepted a mutation; it must be read-only")


# --- derive_field_options ------------------------------------------------------------


def test_derive_maps_a_full_taxonomy_label_set() -> None:
    """A realistic full label set derives one option per derivable field.

    Uses issue #233's own labels (verified live against its board item), so
    the test doubles as a record of the expected end-to-end derivation.
    """

    derived, conflicts = plm.derive_field_options(
        [
            "type: governance",
            "priority: p2",
            "size: s",
            "area: ci-cd",
            "portfolio: operational-maturity",
            "risk: low",
            # Status labels carry no field information for the board.
            "status: in-progress",
        ]
    )
    assert conflicts == ()
    assert derived == {
        "Issue Type": "Governance",
        "Priority": "Medium",
        "Risk": "Low",
        "Size": "S",
        "Repository Area": "ci-cd",
        "Portfolio Signal": "Operational Maturity",
    }


def test_derive_maps_the_governance_portfolio_label_minted_by_237() -> None:
    """`portfolio: governance` derives the board's pre-existing Governance signal.

    The label was minted by the #237 alignment audit for the option that
    already existed on the board (57 carrying items at audit time), so the
    derivation is a direct name match like the other portfolio mappings.
    """

    derived, conflicts = plm.derive_field_options(["portfolio: governance"])
    assert conflicts == ()
    assert derived == {"Portfolio Signal": "Governance"}


def test_derive_ignores_unmapped_and_unknown_labels() -> None:
    """Deliberately unmapped and entirely unknown labels derive nothing, without error."""

    derived, conflicts = plm.derive_field_options(
        ["type: modernization", "area: pipeline", "portfolio: release", "not-a-real-label"]
    )
    assert derived == {}
    assert conflicts == ()


def test_derive_reports_a_conflict_and_withholds_the_field() -> None:
    """Labels deriving different options for one field yield a conflict, not a winner.

    The automation must leave a genuinely ambiguous field for maintainer
    review (docs/governance/github-project.md), so the conflicted field is
    absent from the derived values while unrelated fields still derive.
    """

    derived, conflicts = plm.derive_field_options(["risk: low", "risk: security", "size: m"])
    # The unambiguous field still derives; the conflicted one is withheld.
    assert derived == {"Size": "M"}
    assert len(conflicts) == 1
    # The conflict message names the field and both contributing labels, so
    # a log reader can resolve it without re-deriving anything.
    assert "Risk" in conflicts[0]
    assert "risk: low" in conflicts[0]
    assert "risk: security" in conflicts[0]


def test_derive_treats_agreeing_labels_as_one_clean_derivation() -> None:
    """Two labels deriving the SAME option for one field agree -- no conflict."""

    # Both domain risk labels translate to High, so the derivation is clean.
    derived, conflicts = plm.derive_field_options(["risk: evaluation", "risk: security"])
    assert derived == {"Risk": "High"}
    assert conflicts == ()


def test_derive_emits_fields_in_canonical_order() -> None:
    """Derived fields iterate in DERIVABLE_FIELDS order regardless of label order."""

    derived, _conflicts = plm.derive_field_options(
        ["portfolio: testing-rigor", "size: l", "type: bug"]
    )
    # Insertion order of the result dict must follow the canonical field
    # order, so every consumer logs and mutates deterministically.
    assert list(derived) == ["Issue Type", "Size", "Portfolio Signal"]
