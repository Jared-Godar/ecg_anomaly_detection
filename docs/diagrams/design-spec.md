# Diagram design specification

This document is the authoritative design and content specification for the four
documentation diagrams tracked by #103 (asset replacement) and #104 (generation
workflow). It exists so that any contributor or agent session can produce or revise
the diagrams without access to the original visual references, and so the visual
system stays consistent across all four assets.

## Positioning constraints (non-negotiable)

- This repository is an educational data-engineering portfolio case study, not a
  clinical, diagnostic, or production system. Diagram text must never introduce
  clinical, benchmark, or production-readiness claims.
- The diagrams mimic the *visual language* of AWS-documentation-style architecture
  art (rounded cards, dashed stage containers, numbered step badges, restrained
  palette) but must not use literal AWS service iconography (S3, CodePipeline,
  Lambda, etc.). No cloud infrastructure exists in this project
  (`docs/pipeline-design.md`: "It does not implement final test evaluation, cloud
  infrastructure, or distributed processing").
- Diagram wording must match the host document's current terminology exactly.
  Each diagram mirrors *its own* host section, not a shared vocabulary: the README
  overview says `acquire`; the pipeline-design flow says
  `acquire + verify checksums`. Preserve those differences.

## Visual system

Direction confirmed by the maintainer (2026-07-11): **blues and grays, editorial
restraint** — slate ink and gray connectors from the editorial spec drafted in the
issue #103 comments, applied to the AWS-documentation layout language (dashed stage
containers, numbered circular badges, gray pipeline lane) from the visual
references. No purple/lavender. Warm amber appears in exactly one place: the
`Blocked` state and other warning-semantic elements.

### Palette

| Role | Fill | Stroke | Notes |
|---|---|---|---|
| Ink (titles, card text) | — | — | `#0F172A` font color everywhere |
| Muted text (annotations, captions) | — | — | `#475569` |
| Standard step card | `#EAF2FB` | `#7096C4` | rounded corners (radius 8), stroke width 2 |
| Emphasis/terminal card (outputs, manifests) | `#FFFFFF` | `#7096C4` | white card variant, as in the reference action cards |
| Stage / zone container | `#F4F8FD` | `#8FAFD4` dashed | bold title, dash pattern ~3 |
| Pipeline lane (super-container) | `#F1F5F9` | `#CBD5E1` | radius 12, the "gray cylinder" analogue |
| Primary flow connector | — | `#3E6DB5` | stroke width 2, triangle arrowhead |
| Secondary/output connector | — | `#64748B` | side outputs, feedback edges |
| Numbered step badge | `#DBE8F8` | `#3E6DB5` | small circle, bold `#0F172A` numeral, sits *on* the flow line (chain the edge through the circle) |
| Blocked / warning | `#FEF3C7` | `#B45309` | the only warm accent in the system |
| Merged (lifecycle) | `#C7DBF5` | `#3E6DB5` | deeper blue than standard cards |
| Closed (lifecycle) | `#E2E8F0` | `#64748B` | neutral gray terminal state |

### Type and iconography

- D2's default embedded font (Source Sans Pro subset) — clean sans-serif matching
  the references. No custom font loading.
- Small neutral glyph icons (single-color line style, `#3E6DB5` or `#475569`,
  24×24 viewBox, stroke width 2, round caps) hand-authored as SVGs in
  `docs/diagrams/icons/`: database cylinder (dataset source), document/checklist
  (manifests, reports), folder (data zones), ECG waveform (window extraction —
  the one domain-flavored glyph), branch/split, table grid (index), bar chart
  (metrics/evaluation), shield-check (validation/gates), tag (labels/annotations),
  kanban board (Project #5), person (maintainer).
- Icons are generic primitives. Nothing resembling a specific vendor's service
  icon set.

### Layout language

- Main flows read left-to-right; long flows wrap into rows (use transparent
  un-labeled containers per row so the layout engine stacks them) rather than
  producing an unreadably wide single strip. Target rendered aspect ratios that
  stay legible at GitHub's ~830 px content width.
- Stage groupings and storage zones are dashed-border containers with bold titles,
  exactly like the reference "Source Stage" / "Prod Stage" panels.
- Numbered circular badges mark the ordered steps of a flow. Implementation trick:
  break the connector at a small circle node (`a -- badge; badge -> b`) so the
  badge sits on the line, as in the reference architecture diagram.
- Freestanding annotations (e.g. the test-partition footnote) are plain gray
  italic text, not boxed callouts.
- Automated versus manual distinction (diagram 4 only): solid connectors for
  automated transitions, dashed connectors for maintainer-judgment steps, plus a
  small legend.

## Toolchain

- **D2 v0.7.1 for all four diagrams** with the bundled `elk` layout engine
  (`--layout elk`). Graphviz was evaluated and not selected: reproducing this
  card/container/badge system in DOT requires HTML-like labels and manual
  spacing tuning, and D2's SVG output has cleaner typography. A single tool also
  keeps the regeneration workflow one-step.
- SVG is the primary committed asset. PNG fallback rendered from the SVG via
  `rsvg-convert` at 2× scale.
- Sources live in `docs/diagrams/src/*.d2`, shared styles in
  `docs/diagrams/src/_theme.d2` (spread-imported), icons in
  `docs/diagrams/icons/`, rendered assets in `docs/diagrams/exports/`.
- Rendered assets are deliberately **not** placed in `reports/figures/`: that
  directory is the gitignored output zone for pipeline-generated figures, and
  diagram exports are documentation assets, not run outputs.

Render commands (Fish syntax, matching repository convention):

```fish
d2 --layout elk docs/diagrams/src/implemented-pipeline-overview.d2 docs/diagrams/exports/implemented-pipeline-overview.svg
rsvg-convert --zoom 2 --format png --output docs/diagrams/exports/implemented-pipeline-overview.png docs/diagrams/exports/implemented-pipeline-overview.svg
```

## Diagram 1 — Implemented Pipeline Overview

- **Host**: `README.md`, `## Implemented pipeline` section (replaces the ASCII
  block there).
- **Semantics** (must match the ASCII exactly): source node
  `PhysioNet MIT-BIH v1.0.0` feeds a sequential local pipeline:
  `acquire -> inventory -> validate -> map annotations -> extract windows`, then
  `subject-aware split -> dataset index`, then
  `training -> validation-only evaluation`, terminating in
  `auditable run manifest`.
- **Layout**: source card (database glyph, white card) above or left of a gray
  pipeline lane titled to reflect the README prose ("local and sequential"
  supported workflow). Inside the lane, three wrapped rows matching the ASCII's
  three tiers; numbered badges 1–9 on the flow. `auditable run manifest` as a
  white terminal card with document glyph.
- **Footnote annotation** (gray italic, from README prose): the indexed test
  partition remains unopened and unreported in the supported workflow.
- **Caption**: "The implemented local pipeline: sequential stages from PhysioNet
  acquisition through validation-only evaluation, ending in an auditable run
  manifest."
- **Alt text**: "Flow diagram of the implemented pipeline. PhysioNet MIT-BIH
  v1.0.0 feeds nine sequential stages: acquire, inventory, validate, map
  annotations, extract windows, subject-aware split, dataset index, training,
  and validation-only evaluation, which produces an auditable run manifest. A
  note states the indexed test partition remains unopened."

## Diagram 2 — Local Flow and Artifact Zones

- **Host**: `docs/pipeline-design.md`, `## Target local flow` section.
- **Semantics** (match the ASCII): `PhysioNet MIT-BIH v1.0.0` →
  `acquire + verify checksums` → `data/raw/ (immutable, ignored)`;
  → `validate records + annotations` → side output `validation report`;
  → `create labeled beat windows` → `data/interim/ (rebuildable, ignored)`;
  → `map records to subjects, then split` → `data/processed/ (model-ready,
  ignored)` and side output `split manifest`;
  → `train + evaluate` → `artifacts/ (ignored)` with three sub-outputs:
  `run manifest`, `machine-readable metrics`, `generated figures`.
- **Layout**: vertical main flow (like the reference vertical pipeline), process
  cards in the center; the four storage zones (`data/raw/`, `data/interim/`,
  `data/processed/`, `artifacts/`) as dashed zone containers or folder-glyph
  cards on the right; side-output documents (validation report, split manifest)
  branching right with secondary gray connectors. Preserve each zone's
  parenthetical qualifier — the immutable/rebuildable/model-ready/ignored
  wording is a real directory contract.
- **Caption**: "Target local flow: each transformation writes into a gitignored
  data zone, with validation and split evidence emitted alongside the artifacts."
- **Alt text**: "Vertical flow diagram. PhysioNet MIT-BIH v1.0.0 flows through
  acquire and verify checksums into data/raw, then validate records and
  annotations producing a validation report, then create labeled beat windows
  into data/interim, then map records to subjects and split into data/processed
  with a split manifest, then train and evaluate into artifacts, which contains
  a run manifest, machine-readable metrics, and generated figures. All zones are
  gitignored."

## Diagram 3 — Governance Status Lifecycle

- **Host**: `docs/governance/github-project.md`, `## Status lifecycle` section.
- **Semantics** (match the ASCII): linear progression
  `Backlog -> Ready -> In Progress -> Validation -> Review -> Merged -> Closed`,
  with `Blocked` branching off `In Progress` (and returning — draw the return
  edge, the board treats Blocked as a temporary lane).
- **Transition labels** (from the section's bullets, abbreviated on the arrows):
  linked implementation PR (Ready → In Progress); tests, documentation,
  metadata, and evidence checks (In Progress → Validation); ready for maintainer
  review (Validation → Review); PR merged (Review → Merged); issue closed
  (Merged → Closed).
- **Layout**: horizontal pill-shaped state cards (high border-radius), standard
  blue cards for active states, `Merged` in the deeper blue, `Closed` in
  terminal gray, `Blocked` in the amber warning style hanging below
  `In Progress` with a two-way pair of gray connectors.
- **Caption**: "Project #5 status lifecycle: the progression every work item
  follows, with Blocked as a temporary excursion from In Progress."
- **Alt text**: "State diagram of seven statuses in order: Backlog, Ready, In
  Progress, Validation, Review, Merged, Closed. A Blocked state branches from In
  Progress and returns to it. Arrow labels name the trigger for each transition,
  such as a linked pull request moving work to In Progress."

## Diagram 4 — Governance Automation Overlay (net-new)

- **Host**: `docs/governance/github-metadata-automation.md` (no ASCII precursor;
  place near the top as an orientation figure).
- **Semantics**, synthesized from three sections of that document:
  1. **Idempotent issue creation** (`## Idempotent issue creation`): parse
     reviewed catalog → validate title, labels, milestone, body → list open and
     closed issues → skip exact-title match → create missing labels/milestones →
     create issue → read back and compare.
  2. **Project V2 field update** (`## Project V2 field update` +
     `## Validation`): fields discovered by name, options by exact value;
     existing values preserved unless blank or inconsistent; GraphQL mutation by
     node ID; read-back validation of fields, membership, and statuses.
  3. **PR metadata gate** (`## Automated pull-request metadata gate`):
     `.github/workflows/metadata-governance.yml` runs
     `scripts/github/validate_project_metadata.py` on every PR event; validates
     the PR level (assignee, `type:*` and `area:*` labels, closing reference,
     milestone-or-inherited-exemption) and the linked-issue level (Project
     membership, all required fields populated).
  4. Include the post-merge status sync (`project-status-sync.yml` force-sets
     Status to Merged after merge) as the fourth automated touchpoint.
- **Layout**: reference-architecture style. A large thin-bordered container for
  GitHub, holding two inner containers: the repository (issues, PRs, labels,
  milestones) and Project #5 (board + planning fields). Automation components as
  cards outside/below with numbered badges on their flow lines into the
  containers. Solid connectors = automated actions; dashed connectors = manual
  maintainer steps (catalog review, web-UI view configuration, maintainer
  review); include the small legend.
- **Caption**: "Where automation touches governance: idempotent issue creation,
  Project V2 field updates with read-back validation, the per-PR metadata gate,
  and post-merge status sync — with maintainer judgment left in the loop."
- **Alt text**: "Architecture-style diagram of governance automation. Four
  automated flows touch a GitHub container holding the repository and Project
  number 5: issue creation that skips exact-title matches, field updates that
  preserve curated values, a pull-request metadata gate validating labels,
  milestone, closing references, and project fields, and a post-merge job that
  sets status to Merged. Dashed lines mark manual maintainer steps such as
  catalog review and view configuration."

## Review workflow

Drafts are iterated on this branch with rendered previews shared for maintainer
review. **No pull request is opened until the maintainer has explicitly approved
the image quality of the complete four-diagram set.** Doc integration (swapping
the ASCII blocks for image references with captions and alt text) happens in this
same branch once approval is given.
