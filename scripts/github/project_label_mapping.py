#!/usr/bin/env python3
"""Deterministic repository-label -> Project #5 field/option mapping (issue #233).

One shared, pure-data source of truth for which board fields are *derivable*
from the canonical label taxonomy (`.github/labels.json`), consumed by both
sides of the creation-time board automation:

- `scripts/github/populate_project_item.py` (the `project-item-autofill`
  workflow's writer) uses it to decide which unset fields it may fill; and
- `scripts/detect_board_drift.py` (the scheduled hygiene backstop) uses it to
  flag items whose mapped label is present but whose derived field is unset.

Design rules, decided with the maintainer (issue #233) and consistent with
docs/governance/github-project.md's "use the most specific defensible value,
leave uncertain metadata blank" instruction:

- The mapping is an explicit enumerated table, never a string transform: the
  label taxonomy and the board's option sets are not isomorphic
  (`priority: p2` maps to the option "Medium"; `area: pipeline` has no
  matching Repository Area option at all), so anything other than a
  hand-audited table would manufacture precision.
- Labels with no unambiguous single option are deliberately absent from the
  table -- their fields stay human-set. The exclusions are enumerated in
  `UNMAPPED_LABELS` (with the rationale in comments below) so a future label
  addition fails the completeness test in
  tests/scripts/test_project_label_mapping.py instead of silently drifting.
- Workstream and Target Release have NO label source at all and are an
  explicit automation non-goal (issue #233): they carry judgment the label
  taxonomy does not encode, so no entry here may ever target them.
- Option *names* only, never option IDs: Project V2 option IDs are unstable
  (the 2026-07-14 board-wide regeneration documented in
  docs/governance/github-project.md rewrote every one), so consumers must
  resolve IDs by name at runtime via `github_api.ProjectClient`.

This module is pure data plus pure functions -- no `gh` calls, no network --
so its behavior is exhaustively unit-testable without any GitHub token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

# The six Project #5 single-select fields with at least one label-derivable
# option, in the canonical field order used by docs/governance/github-project.md.
# Status is deliberately not in this table: its automation default (Backlog on
# item creation) is event-driven, not label-driven, and lives with the writer.
DERIVABLE_FIELDS: tuple[str, ...] = (
    "Issue Type",
    "Priority",
    "Risk",
    "Size",
    "Repository Area",
    "Portfolio Signal",
)

# The exact label -> (field, option) table. Every key is a current-taxonomy
# label spelling from .github/labels.json; every value names a live option on
# the corresponding Project #5 single-select field
# (docs/governance/github-project.md#required-fields).
LABEL_TO_FIELD_OPTION: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        # --- type:* -> Issue Type -------------------------------------------------
        # Direct name matches between the taxonomy and the option set.
        "type: bug": ("Issue Type", "Bug"),
        "type: documentation": ("Issue Type", "Documentation"),
        "type: governance": ("Issue Type", "Governance"),
        "type: technical-debt": ("Issue Type", "Technical Debt"),
        # `type: maintenance` follows the documented Dependabot-autofill
        # precedent (docs/governance/github-metadata-automation.md's bot-PR
        # field-default table): routine maintenance is technical-debt
        # servicing on the board.
        "type: maintenance": ("Issue Type", "Technical Debt"),
        # `type: modernization` is deliberately unmapped: modernization work
        # lands as either "Feature" or "Enhancement" depending on scope, a
        # judgment call the label does not encode.
        # --- priority:* -> Priority -----------------------------------------------
        # The taxonomy ladder (docs/governance/label-taxonomy.md: p0 is an
        # active security/integrity failure requiring immediate action, p1 the
        # next high-impact work, p2 normal accepted work, p3 low-urgency)
        # aligns rung-for-rung with the board's Critical/High/Medium/Low
        # options; verified live against board items (e.g. issue #233:
        # `priority: p2` with board Priority "Medium", read back 2026-07-14).
        "priority: p0": ("Priority", "Critical"),
        "priority: p1": ("Priority", "High"),
        "priority: p2": ("Priority", "Medium"),
        "priority: p3": ("Priority", "Low"),
        # --- risk:* -> Risk ---------------------------------------------------------
        # `risk: low` is a direct level match. The three domain labels flag an
        # elevated-stakes domain rather than a level; the historical mapping
        # record (docs/governance/github-metadata-automation.md, "Deterministic
        # mapping precedence": `risk: evaluation` -> Risk = High) established
        # High as their board translation, and data-integrity/security carry
        # the same elevated stakes. Risk "Medium" has no label source and
        # stays human-set.
        "risk: low": ("Risk", "Low"),
        "risk: data-integrity": ("Risk", "High"),
        "risk: evaluation": ("Risk", "High"),
        "risk: security": ("Risk", "High"),
        # --- size:* -> Size ---------------------------------------------------------
        # Direct ladder match. The label taxonomy tops out at `size: l`; the
        # board's XL option has no label source and stays human-set
        # (docs/governance/label-taxonomy.md).
        "size: xs": ("Size", "XS"),
        "size: s": ("Size", "S"),
        "size: m": ("Size", "M"),
        "size: l": ("Size", "L"),
        # --- area:* -> Repository Area ----------------------------------------------
        # Only exact same-named options are mapped; the taxonomy's other area
        # labels (cli, data, pipeline, quality, repository) have no
        # same-named Repository Area option, and picking a "closest" option
        # would be exactly the heuristic inference issue #233 rules out.
        "area: ci-cd": ("Repository Area", "ci-cd"),
        "area: documentation": ("Repository Area", "documentation"),
        "area: evaluation": ("Repository Area", "evaluation"),
        "area: modeling": ("Repository Area", "modeling"),
        "area: validation": ("Repository Area", "validation"),
        # --- portfolio:* -> Portfolio Signal ------------------------------------------
        # Direct name matches (including the two options added by #210).
        # `portfolio: case-study` and `portfolio: release` have no same-named
        # option and stay human-set.
        "portfolio: agentic-engineering": ("Portfolio Signal", "Agentic Engineering"),
        "portfolio: operational-maturity": ("Portfolio Signal", "Operational Maturity"),
        "portfolio: testing-rigor": ("Portfolio Signal", "Testing Rigor"),
    }
)

# Canonical-taxonomy labels that are DELIBERATELY not mapped, with the reason
# recorded in the table comments above. Enumerated so the completeness test
# can assert that every label in .github/labels.json is either mapped here,
# listed here, or belongs to a namespace with no board-field counterpart --
# forcing a conscious decision (and a diff in this file) whenever the
# taxonomy grows.
UNMAPPED_LABELS: frozenset[str] = frozenset(
    {
        # Ambiguous between the Issue Type options "Feature" and "Enhancement".
        "type: modernization",
        # No same-named Repository Area option exists for any of these.
        "area: cli",
        "area: data",
        "area: pipeline",
        "area: quality",
        "area: repository",
        # No same-named Portfolio Signal option exists for either of these.
        "portfolio: case-study",
        "portfolio: release",
    }
)

# Label namespaces that intentionally have no Project-field counterpart at
# all: status labels track the repository-side workflow (the board's Status
# lane is event-driven, not label-driven), and the modernization/dependency
# namespaces are thematic labels with no matching single-select field.
NON_FIELD_NAMESPACES: tuple[str, ...] = ("status:", "modernization:", "dependency:")


def derive_field_options(
    labels: Sequence[str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Derive the board field values implied by an item's current labels.

    Args:
        labels: The item's current label names, in any order.

    Returns:
        A two-tuple of:
        - the derived ``{field name: option name}`` values, containing only
          fields whose mapped labels agree on exactly one option; and
        - human-readable conflict descriptions for every field whose mapped
          labels disagree (e.g. ``risk: low`` and ``risk: security`` on one
          item). Conflicted fields are excluded from the derived values --
          the automation must leave a genuinely ambiguous field for
          maintainer review rather than pick a winner.
    """

    # Group every mapped label's target option by field, preserving which
    # labels contributed so conflict messages can name them exactly.
    options_by_field: dict[str, dict[str, list[str]]] = {}
    # One pass over the item's labels collects every mapped derivation.
    for label in labels:
        mapped = LABEL_TO_FIELD_OPTION.get(label)
        # Unmapped labels (including every non-taxonomy label) simply carry no
        # field information; they are not an error here -- label canonicality
        # is detect_label_drift.py's separate concern.
        if mapped is None:
            continue
        field_name, option_name = mapped
        options_by_field.setdefault(field_name, {}).setdefault(option_name, []).append(label)

    derived: dict[str, str] = {}
    conflicts: list[str] = []
    # Emit fields in the canonical DERIVABLE_FIELDS order so every consumer
    # (mutation loops, drift reports, tests) sees one deterministic ordering.
    for field_name in DERIVABLE_FIELDS:
        candidates = options_by_field.get(field_name)
        # No mapped label touched this field: nothing to derive, no conflict.
        if not candidates:
            continue
        # Exactly one distinct option (possibly voted for by several labels,
        # e.g. two domain risk labels both implying High) is a clean derivation.
        if len(candidates) == 1:
            derived[field_name] = next(iter(candidates))
            continue
        # Two or more distinct options for one field is a real ambiguity the
        # automation must not resolve; describe it precisely for the logs.
        described = ", ".join(
            f"{option!r} (from {', '.join(sorted(labels_for))})"
            for option, labels_for in sorted(candidates.items())
        )
        conflicts.append(f"field {field_name!r} has conflicting label-derived options: {described}")
    return derived, tuple(conflicts)
