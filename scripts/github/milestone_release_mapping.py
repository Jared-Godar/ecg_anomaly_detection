#!/usr/bin/env python3
"""Deterministic milestone -> Target Release coherence table (issue #240).

One shared, pure-data source of truth for which Project #5 `Target Release`
values are *coherent* with each repository milestone, consumed by
`scripts/detect_board_drift.py`'s read-only milestone <-> Target Release
coherence check -- the first tier-3 migration under the automation
verification graduation ladder (issue #248, documented in
docs/governance/github-metadata-automation.md).

Design rules, following the sibling label table
(`scripts/github/project_label_mapping.py`) and decided on issue #240:

- The mapping is an explicit enumerated table, never a string heuristic:
  milestone titles and Target Release options are not isomorphic (three
  different milestone waves share the `Portfolio Release` option; nothing in
  a title like "M8 -- Phase 6/7 Roadmap Completion" mechanically encodes
  `Future`), so anything other than a hand-audited table would manufacture
  precision.
- Each milestone maps to a SET of coherent options, not a single value. The
  bundling convention in docs/governance/github-project.md deliberately lets
  an item keep a broad triage-time bucket (e.g. Target Release `Stewardship`)
  after it is milestoned into a concrete release vehicle, so the
  post-reconciliation curated record (issue #182's pass) is legitimately
  heterogeneous for several milestones. A single-valued table would declare
  that curated history incoherent; the sets below are the narrowest envelopes
  the live record supports (every distribution verified via GraphQL,
  2026-07-16, all 185 milestoned board items).
- A milestone title absent from this table is itself reported as drift by the
  consumer, so minting a new milestone forces a conscious row addition (and a
  reviewable diff) here -- the same forcing function the label table gets
  from its manifest-completeness test.
- Option *names* only, never option IDs: Project V2 option IDs are unstable
  (the 2026-07-14 board-wide regeneration documented in
  docs/governance/github-project.md rewrote every one), so consumers must
  resolve IDs by name at runtime when they need them.

This module is pure data plus pure functions -- no `gh` calls, no network --
so its behavior is exhaustively unit-testable without any GitHub token.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# The Project #5 field this table governs, by display name (consumers derive
# gh's JSON key from it with their existing field-key helpers).
TARGET_RELEASE_FIELD: str = "Target Release"

# The field's complete option set (docs/governance/github-project.md's
# required-fields table, re-verified live via GraphQL 2026-07-16). The
# completeness tests pin every mapped value to this set, so a board-side
# option rename forces a reviewed update here instead of silently orphaning
# table rows.
TARGET_RELEASE_OPTIONS: frozenset[str] = frozenset(
    {
        "Modernization Foundation",
        "Portfolio Release",
        "Stewardship",
        "Future",
    }
)

# The exact milestone-title -> coherent-Target-Release-set table. Every key is
# a live milestone title (em dash and all, exactly as the GitHub API reports
# it); every value is the narrowest option set the maintainer's curated,
# post-#182-reconciliation record supports, with the observed distribution
# (verified live 2026-07-16) recorded per row. Unanimous rows are
# single-option; mixed rows enumerate exactly what the curated record uses,
# because the bundling convention (docs/governance/github-project.md) makes
# that breadth deliberate, not drift.
MILESTONE_TO_TARGET_RELEASES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # --- v1.0.0 wave: the foundational modernization arc ---------------------
        # M1/M2 delivered the repository + pipeline foundation; unanimous
        # Modernization Foundation (13/13 and 11/11 items respectively).
        "M1 — Repository foundation": frozenset({"Modernization Foundation"}),
        "M2 — Reproducible data pipeline": frozenset({"Modernization Foundation"}),
        # M3/M4 delivered the modeling baseline and the first portfolio
        # presentation pass; unanimous Portfolio Release (5/5 and 6/6).
        "M3 — Baseline modeling and evaluation": frozenset({"Portfolio Release"}),
        "M4 — Portfolio release": frozenset({"Portfolio Release"}),
        # M5 is the one genuinely broad historical milestone: its 20 items span
        # all four options (9 Modernization Foundation, 6 Stewardship,
        # 3 Portfolio Release, 2 Future) because developer-experience work was
        # triaged into every bucket. The full set records that reality; the row
        # still catches an unset Target Release on a milestoned item.
        "M5 — Developer Experience and Execution Ergonomics": frozenset(
            {
                "Modernization Foundation",
                "Portfolio Release",
                "Stewardship",
                "Future",
            }
        ),
        # M6/M7 are stewardship-arc milestones by their own charters; unanimous
        # Stewardship (2/2 and 7/7).
        "M6 — Historical Stewardship and Provenance Closure": frozenset({"Stewardship"}),
        "M7 — Governance Completion": frozenset({"Stewardship"}),
        # M8 closed out roadmap phases whose remaining scope was deferred
        # long-horizon work; unanimous Future (3/3).
        "M8 — Phase 6/7 Roadmap Completion": frozenset({"Future"}),
        # The v1.0.0 release vehicle itself; unanimous Portfolio Release (4/4).
        "v1.0.0 — First Portfolio Release": frozenset({"Portfolio Release"}),
        # --- v1.1.0 wave: post-1.0 delivery ---------------------------------------
        # M9's held-out evaluation items were all triaged as deferred
        # long-horizon work; unanimous Future (6/6).
        "M9 — Held-out Evaluation": frozenset({"Future"}),
        # M10 mixes stewardship fixes with one release-bound and one deferred
        # item (4 Stewardship, 1 Portfolio Release, 1 Future in the curated
        # record); Modernization Foundation is the one option it never uses.
        "M10 — Post-1.0 CI and Portfolio Integrity": frozenset(
            {"Portfolio Release", "Stewardship", "Future"}
        ),
        # M11's optional-polish charter spans stewardship and deferred work
        # (11 Stewardship, 15 Future); never release-bound.
        "M11 — Post-1.0 Release Optional Polish": frozenset({"Stewardship", "Future"}),
        # The v1.1.0 release vehicle: predominantly Portfolio Release (53/67),
        # with Stewardship-triaged items shipped through it under the bundling
        # convention (14/67). Modernization Foundation or Future here would be
        # genuine drift -- exactly the direction issue #240's specimens took.
        "v1.1.0 — Second Portfolio Release": frozenset({"Portfolio Release", "Stewardship"}),
        # --- open long-horizon work -----------------------------------------------
        # M12 collects deliberately deferred enhancements; unanimous Future
        # (9/9, the only milestone with open items at table-authoring time).
        "M12 — Optional Future Enhancements": frozenset({"Future"}),
    }
)

# Target Release values coherent for an item with NO milestone: the two
# long-horizon buckets. Modernization Foundation and Portfolio Release each
# name a concrete delivery arc, so carrying one without any milestone claims
# a release vehicle that does not exist -- the reverse drift direction issue
# #240's problem statement calls out. An unmilestoned item with Target
# Release unset is deliberately NOT this table's concern: full nine-field
# completeness is the PR-time metadata gate's job.
UNMILESTONED_TARGET_RELEASES: frozenset[str] = frozenset({"Stewardship", "Future"})


def coherence_problems(
    milestone_title: str | None,
    target_release: str | None,
) -> tuple[str, ...]:
    """Describe every milestone <-> Target Release incoherence for one item.

    Pure function over the enumerated tables above; the caller supplies both
    sides (the item's milestone title from the repository and its Target
    Release option name from the board) and receives human-readable problem
    strings in the same shape the drift report already uses.

    Args:
        milestone_title: The item's milestone title exactly as the GitHub API
            reports it, or None when the item is unmilestoned.
        target_release: The item's populated Target Release option name, or
            None when the field is unset.

    Returns:
        A tuple of problem descriptions; empty when the pair is coherent.
        At most one problem is ever produced per item, but the tuple shape
        matches the drift report's problems container for direct extension.
    """

    # Neither side present: nothing to cross-check. Whether the fields SHOULD
    # be populated is the PR-time metadata gate's completeness question, not
    # this coherence check's.
    if milestone_title is None and target_release is None:
        return ()

    # Unmilestoned but bucketed: only the long-horizon buckets are coherent
    # without a delivery vehicle (the issue #240 reverse-drift direction).
    if milestone_title is None:
        # A release-arc bucket (or an unrecognized option name) without any
        # milestone claims a delivery vehicle that does not exist.
        if target_release not in UNMILESTONED_TARGET_RELEASES:
            return (
                f"Target Release {target_release!r} is set but the item has no "
                f"milestone; unmilestoned items may only carry "
                f"{' or '.join(sorted(UNMILESTONED_TARGET_RELEASES))}",
            )
        return ()

    expected = MILESTONE_TO_TARGET_RELEASES.get(milestone_title)
    # A milestone this table has never heard of cannot be checked -- and that
    # is itself the finding: minting a milestone must come with a table row,
    # or the coherence guarantee silently rots.
    if expected is None:
        return (
            f"milestone {milestone_title!r} has no row in "
            f"scripts/github/milestone_release_mapping.py; add one so its "
            f"Target Release coherence can be checked",
        )

    # Milestoned but unbucketed: the issue #240 problem statement flags this
    # explicitly -- a delivery vehicle without its board-side release bucket.
    if target_release is None:
        return (f"milestoned ({milestone_title!r}) but Target Release is unset",)

    # Both sides present: coherent exactly when the pair is in the envelope.
    if target_release not in expected:
        return (
            f"Target Release {target_release!r} is incoherent with milestone "
            f"{milestone_title!r} (coherent values: "
            f"{', '.join(sorted(expected))})",
        )
    return ()
