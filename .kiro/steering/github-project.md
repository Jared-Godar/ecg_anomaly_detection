---
inclusion: manual
---

# GitHub Project #5 — ECG Pipeline Modernization

Reference data for populating Project #5 fields. Include this steering file
(via `#github-project`) when working on issue/PR metadata or project board
operations.

## Project identifiers

- **Project number:** 5
- **Project node ID:** `PVT_kwHOAQEwMM4BcY39`
- **Owner:** Jared-Godar

## Status field

- **Field ID:** `PVTSSF_lAHOAQEwMM4BcY39zhXCFmM`

| Status | Option ID |
|--------|-----------|
| Backlog | `2bb9dbb0` |
| Ready | `010387a6` |
| In Progress | `ac9c56d8` |
| Blocked | `61c602c6` |
| Review | `94bf0d2e` |
| Validation | `eed5658a` |
| Merged | `d9e041e5` |
| Closed | `2f81c115` |
| Not Planned | `d4be655b` |

## Portfolio Signal field

- **Field ID:** `PVTSSF_lAHOAQEwMM4BcY39zhXJtrU`

| Portfolio Signal | Option ID |
|-----------------|-----------|
| Reproducibility | `130e719a` |
| Operational Maturity | `cb3db70f` |
| Governance | `afd2bfcf` |
| Data Engineering | `7f1675f7` |
| Documentation | `1fe3d55b` |
| Developer Experience | `3eb76ac0` |
| Systems Thinking | `f70713a0` |
| Ownership | `27243f96` |
| Testing Rigor | `d52a7c36` |
| Agentic Engineering | `5486739f` |

## Required nine fields

Every issue and PR added to the project must populate:

1. Status
2. Workstream
3. Issue Type
4. Priority
5. Risk
6. Size
7. Repository Area
8. Portfolio Signal
9. Target Release

## Status lifecycle

- New items: Backlog (set automatically by `project-item-autofill.yml` when unset)
- Implementation PR linked: In Progress
- PR open and awaiting merge: Review
- PR merged: Merged (set by `project-status-sync.yml`; verify at the cadence of
  its current tier on the automation verification graduation ladder — the
  closure pass's lane read is action-gating and always happens)
- Issue completed and closed: Closed
- Issue closed as "not planned": Not Planned

## Creation-time autofill (issue #233)

`project-item-autofill.yml` adds every human-authored issue/PR to the board on
`opened`/`labeled` events and mirrors label-derivable fields (`type:` → Issue
Type, `priority:` → Priority, `risk:` → Risk, `size:` → Size, `area:` →
Repository Area, `portfolio:` → Portfolio Signal), filling only UNSET fields —
curated values win. Workstream and Target Release stay human-set. Verify at the
cadence of the automation's current tier on the automation verification
graduation ladder (issue #248; placements and streak evidence in
`docs/governance/github-metadata-automation.md`) — automation is a default, not
a substitute for verification, and any observed failure resets it to per-event
read-backs.

## CLI patterns (Fish)

```fish
# Discover an item's project-internal ID
gh project item-list 5 --owner Jared-Godar --format json --limit 500 \
  | jq '.items[] | select(.content.number == <N>) | .id'

# Set a field (always read-back-verify afterward)
gh project item-edit --id <ITEM_ID> --field-id <FIELD_ID> \
  --project-id PVT_kwHOAQEwMM4BcY39 \
  --single-select-option-id <OPTION_ID>
```

## Warning

Never use `updateProjectV2Field` to add or edit options — it replaces the
entire option set and regenerates ALL option IDs, orphaning every stored value
on the board. Add/rename options through the web UI only.
