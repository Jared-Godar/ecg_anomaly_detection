# Diagram design specification

This document is the authoritative design and content specification for the four
documentation diagrams tracked by #103 (asset replacement) and #104 (generation
workflow). It exists so that any contributor or agent session can produce or revise
the diagrams without access to the original visual references, and so the visual
system stays consistent across all four assets.

**Diagrams 2 (Local Flow and Artifact Zones) and 3 (Governance Status Lifecycle)
are the maintainer-approved reference implementations** (both signed off
2026-07-11): diagram 3 for the card/connector/title visual system, diagram 2 for
the legend/caption/compose architecture, and diagram 3 again for retrofitting
that architecture onto an already-approved layout without disturbing it. They
were built through several corrected rounds each (see "History" at the end of
this document for what was tried and rejected, and why). Diagrams 1 and 4 must
follow their pattern exactly unless a specific reason requires deviating --
state the reason if so.

## Positioning constraints (non-negotiable)

- This repository is an educational data-engineering portfolio case study, not a
  clinical, diagnostic, or production system. Diagram text must never introduce
  clinical, benchmark, or production-readiness claims.
- No cloud infrastructure exists in this project (`docs/pipeline-design.md`: "It
  does not implement final test evaluation, cloud infrastructure, or distributed
  processing"). Do not use literal AWS or other vendor service iconography
  anywhere -- icons are generic primitives only.
- Diagram wording must match the host document's current terminology exactly.
  Each diagram mirrors *its own* host section, not a shared vocabulary: the README
  overview says `acquire`; the pipeline-design flow says
  `acquire + verify checksums`. Preserve those differences.
- **Never use an internal numeric ID as if it were a meaningful name.** "Project
  #5" (the raw GitHub Project board number) is exactly the kind of placeholder
  this forbids -- confirmed live (2026-07-11) when the maintainer caught it in
  diagram 3's subtitle: "Always use meaningful descriptive names not bullshit
  numeric placeholders with no value." Use the board's actual name, **ECG
  Pipeline Modernization**, or omit the reference and describe the thing
  generically ("the project board") if a proper noun would be clunkier than
  useful. This applies to every diagram, not just diagram 3 -- diagram 4's own
  semantics section below already needed the same fix.

## Title and explanatory framing (required, durable standard)

Every diagram, wherever it is embedded or viewed -- in its host doc, in a review
gallery, or opened as a bare image file with no surrounding context -- must tell
the reader what they are looking at on its own, without relying on the viewer
already having the host document open.

**How this is satisfied: the title and a one-line subtitle are rendered as part of
the diagram image itself**, at the top, using the graph's own title/label
mechanism (see Toolchain below) -- not markdown text floating separately, and not
a separate un-attached shape positioned by coordinate guessing. This was reached
only after two rejected approaches, in order:

1. **Markdown-only framing** (heading + paragraph in the host doc, nothing in the
   image) was the first standard written here and was **wrong** -- it was
   silently substituted for the maintainer's actual request the first time an
   in-image title proved hard to render correctly, and the maintainer explicitly
   rejected the substitution: "you spent a bunch of tokens/time on a workaround
   that changes WHERE the explanation would eventually live, but you never
   actually delivered ANY visible title/explanation in what I'm looking at right
   now." The lesson: when the technically-easy path stops matching the request,
   say so and ask, don't quietly redefine the deliverable.
2. **In-image title via D2** was then attempted and also failed, for genuine
   technical reasons documented in Toolchain below (not a style choice) -- this
   is why the diagrams are Graphviz now, not D2.

The markdown-level heading and caption in the host doc (see the embed pattern
below) still matter, but as a *second*, complementary layer for page navigation
and accessibility -- not a substitute for the in-image title.

Host-doc embed pattern for all four diagrams:

```markdown
### <Diagram title>

<One to two sentences of context: what this shows and why it matters here.>

![<alt text from this spec>](../diagrams/exports/<file>.svg)

*<caption from this spec>* — generated from
[`<file>.dot`](../diagrams/src/<file>.dot) via Graphviz (see
[`docs/diagrams/design-spec.md`](design-spec.md)).
```

That trailing source link is deliberate, not decorative: the maintainer explicitly
asked (2026-07-11) that the programmatic, reproducible generation of these
diagrams be a **visible portfolio signal** -- proof that even the documentation
art is code, versioned, and regenerable, consistent with this project's
reproducibility framing -- even though that's not the usual call for a project
about the pipeline rather than the portfolio. Don't bury the source reference;
surface it in every embed.

## Visual system

**Direction confirmed by the maintainer (2026-07-11), replacing an earlier,
never-actually-validated "blues and grays" direction**: dark navy ground with a
glowing cyan accent, extracted directly from this repository's own existing
documentation banners (`docs/assets/ecg-*-banner.png`) via pixel sampling --
*not* eyeballed from a screenshot. Those banners were the maintainer's actual
style reference the whole time; an earlier round of this spec was written and
"user-confirmed" against a different, lighter palette that turned out never to
have been checked against the real reference material. Lesson generalized in
`feedback_deconfliction`-adjacent territory: if a *different* Claude Code session
saw style-defining reference material, that context does not propagate here on
its own -- it has to land in this repo (this file, an issue, a memory) or get
re-supplied. Don't assume a "confirmed" direction is actually confirmed against
the right thing without evidence.

### Palette

Extracted from `docs/assets/ecg-development-workflow-banner.png` and
`ecg-contributing-banner.png` by sampling actual pixel values, then adapted for
diagram cards (see `governance-status-lifecycle.dot` for the reference
implementation):

| Role | Fill | Stroke | Notes |
|---|---|---|---|
| Canvas background | `#000C1E` | — | near-black navy, sampled directly from the banners |
| Title text | — | `#FFFFFF` | bold, ~18pt |
| Subtitle text | — | `#00D4E8` | ~13pt, one line, no internal IDs (see above) |
| Standard state/step card | `#0A1B33` | `#00D4E8` | rounded box, stroke width 2 |
| Merged / success-terminal card | `#00506B` | `#00E5FF` | brighter fill and stroke than standard -- deliberately the most saturated card in the system |
| Closed / neutral-terminal card | `#122238` | `#4A6584` | muted text (`#B9C7DA`), reads as "done, no longer active" |
| Blocked / warning card | `#2A1F0A` | `#F5A623` | the only warm accent in the system |
| Primary flow connector | — | `#00D4E8` | stroke width 2 |
| Secondary/branch connector (e.g. Blocked) | — | `#F5A623` | dashed, matches the Blocked card's warm accent |
| Edge label text | — | `#8FE3F0` | ~11pt, **must** sit on an HTML `<TABLE>` cell with `BGCOLOR="#000C1E"` behind it -- see Toolchain; a plain label lets the connector line visibly cut through the text |

The bright pure-cyan (`#00FFFF`) seen at the brightest points in the source
banners is a glow/bloom highlight, not the base accent color -- use `#00D4E8` (or
`#00E5FF` for the one deliberately-more-saturated Merged card) as the working
accent, not the highlight extreme.

**Glow approximation**: neither Graphviz nor D2 render a true CSS-style colored
glow/blur. A bright, moderately thick cyan stroke is the approved approximation
(maintainer, 2026-07-11: "glow approximation workaround seems fine") -- do not
spend effort trying to fake a real blur filter in SVG for this.

### Type and iconography

- Graphviz's default `Helvetica,Arial,sans-serif` font stack -- no custom font
  loading, matches the banners' clean sans-serif.
- **Edge/transition labels are Title Case** ("Linked PR", "Tests, Docs, Metadata,
  Evidence"), confirmed explicitly by the maintainer (2026-07-11), not sentence
  case or lowercase.
- Icons, where a diagram calls for them (diagram 3 needed none -- a pure state
  machine reads fine as labeled cards alone): simple line-style glyphs in
  `#00D4E8` or `#FFFFFF`, 24×24, stroke width 2, round caps, generic primitives
  only (database cylinder, document, folder, branch/split, shield-check, tag,
  kanban board, person). Hand-author as SVGs in `docs/diagrams/icons/` if a
  diagram's content genuinely needs one -- don't add icons just because the
  original plan listed them if the actual content reads fine without.

### Layout language

- **Aspect ratio rule: not too wide for markdown embedding is the standard --
  literal squareness is not the goal.** Confirmed explicitly by the maintainer
  (2026-07-11) after two overcorrections in the other direction: a first draft
  came out ~7.8:1 (unreadable without horizontal scrolling), a "fix" then
  overcorrected to ~0.5:1 (too tall, sparse). The approved diagram 3 render lands
  at roughly 2.3:1 (985×424pt before padding) -- treat that as the working target
  band, not an exact number.
- Wrap long flows into multiple rows using Graphviz `{ rank=same; A; B; C }`
  groups (see Toolchain) rather than a single wide strip.
- Pill/rounded-box cards throughout; no dashed zone containers or numbered
  circular badges were needed for diagram 3's content, but diagram 1 (nine
  sequential stages) and diagram 4 (multi-source automation overlay) may
  legitimately need them -- badges and containers are still permitted where the
  content's own structure calls for them, they're just not mandatory decoration.
- Give the diagram generous background padding on all four sides via
  `pad_svg.py` (see Toolchain) -- confirmed explicitly required by the
  maintainer after an early render clipped card borders right at the canvas
  edge.

## Legend and caption (required, durable standard)

Finalized 2026-07-11 with the maintainer's approval of diagrams 2 and 3; every
diagram carries both of these, rendered as part of the image itself.

**Legend.** Every color or line-style distinction used in a diagram exists for a
reason, and that reason must be stated explicitly in a legend -- not implied.
The legend is a single HTML `<TABLE>` node in its **own standalone `.dot` file**
(`<name>-legend.dot`), never a node inside the main graph: sharing the main
graph's rank system makes the whole rank band (and the `ranksep` gaps around it)
grow to the legend's height, distorting the flow's approved spacing. Structure
(see `local-flow-artifact-zones-legend.dot` and
`governance-status-lifecycle-legend.dot`): bordered `#0A1B33` panel on
`#4A6584`, bold 11pt white "Legend" header, 9pt Title Case entries, 22x12 color
swatch cells, arrow-glyph cells for connector styles. Where a swatch's fill is
too dark to read against the panel (`#0A1B33`, `#122238`), give the swatch a
1px border in the card style's stroke color, as diagram 3's legend does.

**Caption.** A plain-language figure caption in **non-technical language**,
separate from the legend: white box (`BGCOLOR="#FFFFFF"`, `#8A8F98` border),
black 11pt sans-serif text, bold lead-in phrase naming the diagram,
left-justified lines, sized to its own content rather than spanning the image
width (maintainer, verbatim: "I am okay with the white inset box over the
previous clarification for it spanning the entire image"). Two attachment
variants, both approved:

- **Baked-in node** (diagram 2): a `shape=none` HTML-table node inside the main
  graph, positioned by `style=invis` edges from the last rank. Use when this
  does not disturb the layout or canvas.
- **Separate composited graph** (diagram 3, `<name>-caption.dot`): required
  when the caption-as-node damages the approved layout -- hanging it off a
  right-heavy bottom rank widened diagram 3's canvas by ~126pt, and adding more
  invisible positioning edges fed the crossing-minimization pass, which
  **mirrored the entire graph**. When that happens, keep the main `.dot`
  untouched and splice the caption in with the compositor.

**Placement is a per-diagram visual-harmony judgment, not canon** (maintainer,
verbatim: "Where exactly the legend goes is not canonical, but depends on the
overall visual layout on a per workflow basis to have the best overall
harmony"). The constants: never extend the canvas into a mostly-empty new
column ("dont orphan the legend and make it unnecessarily wide - use your space
efficiently"); prefer the diagram's own existing empty space; center the piece
in a band bounded by real neighboring content edges; and **measure those bounds
from actual SVG coordinates -- never eyeball them**. The approved placements and
their exact inset values are recorded in each diagram's section below so every
render is reproducible.

## Toolchain

**Graphviz (`dot`), not D2.** D2 with the bundled `elk` layout engine was the
original toolchain decision and is what diagram 3's first several rounds used --
it was abandoned after `elk` proved unable to support this diagram's actual
requirements, confirmed by direct, reproducible failures, not a style preference:

- `elk` **does not support locked/explicit node positions at all** -- a direct
  compile error ("layout engine \"elk\" does not support locked positions") the
  moment `top`/`left` were set on any shape.
- Four different D2 positioning mechanisms were tried, in order, for the
  in-image title and the multi-row wrap, and each hit a real, reproducible
  limitation: markdown-text shapes render via SVG `foreignObject`, which
  `rsvg-convert` silently fails to render into the PNG; a `near`-positioned
  plain-text container produced a stray auto-generated label and overlapping
  text; an unconnected top-level container's own `direction:` setting was
  silently overridden once it became graph-connected to anything else (the
  entire flow collapsed from horizontal to vertical); and `grid-columns`/
  `grid-rows` fills **column-major**, not row-major, which is easy to get wrong
  and produced a confusing zigzag layout with overlapping edge labels on the
  first attempt.
- Graphviz's `{ rank=same; ... }` grouping is well-documented, predictable, and
  solved the multi-row wrap correctly on the first real attempt.

**Rendering pipeline (Fish syntax):** render each graph with `dot`, splice the
legend (and, for the separate-caption variant, the caption) into the main SVG
with `compose_inset.py`, then pad and convert. The compose step's inset values
are per-diagram measured constants recorded in each diagram's section below.

```fish
dot -Tsvg docs/diagrams/src/<name>.dot -o /tmp/<name>.svg
dot -Tsvg docs/diagrams/src/<name>-legend.dot -o /tmp/<name>-legend.svg
python3 docs/diagrams/compose_inset.py /tmp/<name>.svg /tmp/<name>-legend.svg \
    docs/diagrams/exports/<name>.svg --<left|right>-inset <pt> --top-inset <pt>
python3 docs/diagrams/pad_svg.py docs/diagrams/exports/<name>.svg
rsvg-convert --zoom 2 --format png --output docs/diagrams/exports/<name>.png docs/diagrams/exports/<name>.svg
```

- **`compose_inset.py` (`docs/diagrams/compose_inset.py`)** splices a
  standalone-rendered graph (legend, caption box) into a base SVG as a plain
  transformed `<g>` group in the base's own coordinate system. It exists
  because the two obvious alternatives both fail, confirmed empirically:
  nesting `<svg>` viewports trips an rsvg-convert bug that visibly clips the
  inset's text in the combined render even though the inset renders perfectly
  alone, and putting the inset in the main graph's rank system distorts the
  approved layout (see the Legend/caption section above). Insets are placed by
  measured offsets from the base background polygon's edges; an inset placed
  beyond an edge (diagram 3's caption, below the graph) expands the canvas,
  background, and viewBox just enough to cover it. Base and output may be the
  same path, so several insets can be stacked in sequence.

- **`pad_svg.py` (`docs/diagrams/pad_svg.py`) is a required step, not optional
  polish.** Graphviz's own `margin` graph attribute stops reliably expanding the
  background polygon once HTML-like (`<TABLE>`-based) edge labels are present in
  the graph -- confirmed by inspecting the actual rendered background polygon's
  coordinates, which collapsed to ~4pt of margin regardless of a 0.75in `margin`
  attribute the moment edge labels were switched to HTML tables. Rather than
  keep chasing Graphviz's internal bounding-box computation, this script pads
  the already-rendered SVG directly and deterministically, independent of
  whatever content changes triggered the regression. Run it on every diagram,
  every render, after `dot` and before `rsvg-convert`.
- SVG is the primary committed asset; PNG is the fallback for viewers without
  SVG support, rendered from the *padded* SVG.
- **In-image title/subtitle**: set via the graph's own `label`, `labelloc=t`,
  `labeljust=c` attributes (title bold, subtitle a `<FONT COLOR="#00D4E8"
  POINT-SIZE="13">` run within the same label) -- this renders as native SVG
  text, not `foreignObject`, so it survives the PNG conversion correctly.
- **Edge labels needing breathing room from the connector line**: wrap the label
  text in an HTML-like label,
  `label=<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4" BGCOLOR="#000C1E"><TR><TD><FONT COLOR="#8FE3F0" POINT-SIZE="11">Label Text</FONT></TD></TR></TABLE>>`
  -- the matching-background table cell visually "erases" the connector line
  where it would otherwise run straight through the text.
- Sources live in `docs/diagrams/src/*.dot` (main graph plus its `-legend.dot`
  and, where used, `-caption.dot` companions), the compose and padding scripts
  at `docs/diagrams/compose_inset.py` and `docs/diagrams/pad_svg.py`, icons
  (where used) in `docs/diagrams/icons/`, rendered assets in
  `docs/diagrams/exports/`.
- Rendered assets are deliberately **not** placed in `reports/figures/`: that
  directory is the gitignored output zone for pipeline-generated figures, and
  diagram exports are documentation assets, not run outputs.
- Both sources and the padding script are committed and intended to be
  publicly discoverable -- see the reproducibility note under "Title and
  explanatory framing" above.

## Diagram 1 — Implemented Pipeline Overview (approved 2026-07-11)

- **Host**: `README.md`, `## Implemented pipeline` section (replaces the ASCII
  block there).
- **Files**: `src/implemented-pipeline-overview.dot` +
  `-legend.dot` + `-caption.dot` + `-footnote.dot` (the test-partition note,
  spliced like the others).
- **Approved compose placement** (three splices): caption centered under the
  heading below everything (`--left-inset 371.07 --top-inset 481`); footnote
  right-aligned under the terminal row it annotates
  (`--right-inset 4 --top-inset 405`); legend upper-right, right edge aligned
  with the terminal card, vertically centered between the subtitle's bottom
  and that card's top (`--right-inset 4 --top-inset 150.18`).
- **Semantics** (must match the ASCII exactly): source node
  `PhysioNet MIT-BIH v1.0.0` feeds a sequential local pipeline:
  `acquire -> inventory -> validate -> map annotations -> extract windows`, then
  `subject-aware split -> dataset index`, then
  `training -> validation-only evaluation`, terminating in
  `auditable run manifest`.
- **Layout**: follow diagram 3's pattern -- title/subtitle baked in, wrap into
  `rank=same` rows to stay within the ~2:1-3:1 aspect band, not a single wide
  strip of nine cards.
- **Footnote annotation** (from README prose): the indexed test partition
  remains unopened and unreported in the supported workflow. Render as plain
  muted (`#8FE3F0` or dimmer) text, not a boxed callout.
- **Caption**: "The implemented local pipeline: sequential stages from PhysioNet
  acquisition through validation-only evaluation, ending in an auditable run
  manifest."
- **Alt text**: "Flow diagram of the implemented pipeline. PhysioNet MIT-BIH
  v1.0.0 feeds nine sequential stages: acquire, inventory, validate, map
  annotations, extract windows, subject-aware split, dataset index, training,
  and validation-only evaluation, which produces an auditable run manifest. A
  note states the indexed test partition remains unopened."

## Diagram 2 — Local Flow and Artifact Zones (approved 2026-07-11)

- **Host**: `docs/pipeline-design.md`, `## Target local flow` section.
- **Files**: `src/local-flow-artifact-zones.dot` (flow, with baked-in caption
  node) + `src/local-flow-artifact-zones-legend.dot`.
- **Approved compose placement**: legend inset upper-right, vertically centered
  between the subtitle's bottom edge and the top of the `data/processed/` block
  (maintainer-directed): `--right-inset 4 --top-inset 174.27`.
- **Semantics** (match the ASCII): `PhysioNet MIT-BIH v1.0.0` →
  `acquire + verify checksums` → `data/raw/ (immutable, ignored)`;
  → `validate records + annotations` → side output `validation report`;
  → `create labeled beat windows` → `data/interim/ (rebuildable, ignored)`;
  → `map records to subjects, then split` → `data/processed/ (model-ready,
  ignored)` and side output `split manifest`;
  → `train + evaluate` → `artifacts/ (ignored)` with three sub-outputs:
  `run manifest`, `machine-readable metrics`, `generated figures`.
- **Layout**: follow diagram 3's pattern. Preserve each zone's parenthetical
  qualifier — the immutable/rebuildable/model-ready/ignored wording is a real
  directory contract, not decoration.
- **Caption**: "Target local flow: each transformation writes into a gitignored
  data zone, with validation and split evidence emitted alongside the artifacts."
- **Alt text**: "Vertical flow diagram. PhysioNet MIT-BIH v1.0.0 flows through
  acquire and verify checksums into data/raw, then validate records and
  annotations producing a validation report, then create labeled beat windows
  into data/interim, then map records to subjects and split into data/processed
  with a split manifest, then train and evaluate into artifacts, which contains
  a run manifest, machine-readable metrics, and generated figures. All zones are
  gitignored."

## Diagram 3 — Governance Status Lifecycle (reference implementation, approved; legend/caption retrofit approved 2026-07-11)

- **Host**: `docs/governance/github-project.md`, `## Status lifecycle` section.
- **Files**: `src/governance-status-lifecycle.dot` (graph only -- kept exactly
  as originally approved, no caption node; see the Legend/caption section for
  why) + `src/governance-status-lifecycle-legend.dot` +
  `src/governance-status-lifecycle-caption.dot`.
- **Approved compose placement** (two splices, caption first):
  caption centered under the heading, one ranksep below the `Merged`/`Closed`
  row: `--left-inset 206.98 --top-inset 417.9`; legend centered in the
  rectangle bounded by the `Backlog`/`Ready` row's bottom, the caption's top,
  the padded image's left edge, and the `Blocked` box's left edge
  (maintainer-directed): `--left-inset 64.63 --top-inset 203.6`.
- **Semantics**: linear progression
  `Backlog -> Ready -> In Progress -> Validation -> Review -> Merged -> Closed`,
  with `Blocked` branching off `In Progress` and returning (bidirectional dashed
  amber edge).
- **Transition labels** (Title Case, boxed against the connector line): "Linked
  PR" (Ready → In Progress); "Tests, Docs, Metadata, Evidence" (In Progress →
  Validation); "Ready for Review" (Validation → Review); "PR Merged" (Review →
  Merged); "Issue Closed" (Merged → Closed).
- **Subtitle** (verbatim, maintainer-specified 2026-07-11): "Modernization work
  item progression".
- **Layout**: three Graphviz `rank=same` rows -- `{Backlog, Ready, In Progress}`,
  `{Blocked, Validation, Review}`, `{Merged, Closed}` -- approved final aspect
  ratio ~2.3:1 before padding. Source: `docs/diagrams/src/governance-status-lifecycle.dot`.
- **Caption**: "The Project #5 status lifecycle" ~~-- no, do not use this~~.
  Use: "The ECG Pipeline Modernization board's status lifecycle: the progression
  every work item follows, with Blocked as a temporary excursion from In
  Progress."
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
- **Layout**: reference-architecture style, following diagram 3's palette/type
  rules. A large thin-bordered container for GitHub, holding two inner
  containers: the repository (issues, PRs, labels, milestones) and the **ECG
  Pipeline Modernization board** (board + planning fields) -- not "Project #5."
  Automation components as cards outside/below with flow lines into the
  containers. Solid connectors = automated actions; dashed connectors = manual
  maintainer steps (catalog review, web-UI view configuration, maintainer
  review); include a small legend.
- **Caption**: "Where automation touches governance: idempotent issue creation,
  project board field updates with read-back validation, the per-PR metadata
  gate, and post-merge status sync — with maintainer judgment left in the loop."
- **Alt text**: "Architecture-style diagram of governance automation. Four
  automated flows touch a GitHub container holding the repository and the ECG
  Pipeline Modernization project board: issue creation that skips exact-title
  matches, field updates that preserve curated values, a pull-request metadata
  gate validating labels, milestone, closing references, and project fields, and
  a post-merge job that sets status to Merged. Dashed lines mark manual
  maintainer steps such as catalog review and view configuration."

## Review workflow

Drafts are iterated on this branch with rendered previews shared for maintainer
review. **No pull request is opened until the maintainer has explicitly approved
the image quality of the complete four-diagram set.** Diagrams 1, 2, and 3
(including diagram 3's legend/caption retrofit) have that approval as of
2026-07-11; diagram 4 does not yet. Doc integration (swapping the ASCII
blocks for image references with captions and alt text) happens in this same
branch once full-set approval is given.

## History: what was tried and rejected on diagram 3, and why

Kept here so a future session doesn't re-attempt any of these:

1. Markdown-only title framing (no in-image title) -- rejected by the
   maintainer as not actually meeting the "tell the user what they're looking
   at" requirement when the diagram is viewed standalone.
2. D2 markdown-text title shape -- silently fails to render in the
   `rsvg-convert` PNG fallback (`foreignObject` unsupported).
3. D2 plain-text title in a `near`-positioned container -- stray auto-label,
   overlapping heading/subheading text.
4. D2 plain-text title in an unconnected `direction: down` top-level container
   next to the flow -- same overlap bug, plus the flow's own `direction: right`
   got silently overridden once graph-connected, collapsing to one vertical
   column.
5. D2 explicit `top`/`left` positioning -- `elk` layout engine does not support
   locked positions at all; hard compile error.
6. D2 `grid-columns`/`grid-rows` -- fills column-major, not row-major; produced
   a zigzag reading order and overlapping edge labels.
7. **Switched to Graphviz.** `{ rank=same }` rows worked correctly on the first
   real attempt. In-image title via `label`/`labelloc`/`labeljust` worked
   immediately (native SVG text, not `foreignObject`).
8. Graphviz `margin` attribute for canvas padding -- worked initially, then
   silently stopped working (collapsed to ~4pt) once edge labels were switched
   to HTML `<TABLE>` labels for line-breathing-room. Replaced with the
   deterministic `pad_svg.py` post-process, which is now the permanent,
   required step regardless of what else changes.

And on the legend/caption compose architecture (diagrams 2 and 3, 2026-07-11):

1. Legend as a node in the main graph's rank system -- an entire rank band and
   its `ranksep` gaps grew to the legend's height, distorting the flow's
   approved vertical spacing. Split into a standalone legend graph.
2. Nested-`<svg>`-in-`<svg>` composition of the two renders -- rsvg-convert
    visibly clips the inset's text in the combined render even though the
    legend renders perfectly on its own. Replaced by splicing the inset's root
    `<g>` as a plain transformed group (`compose_inset.py`).
3. Side-by-side composition into a new full-height column -- rejected by the
    maintainer ("dont orphan the legend and make it unnecessarily wide - use
    your space efficiently"): the legend filled ~20% of a column that cost
    ~30% extra canvas width. Insets go inside the diagram's own empty space.
4. Diagram 3's caption as an in-graph node -- invisible edges from
    `Merged`/`Closed` hung it bottom-right and widened the canvas ~126pt;
    adding more invisible edges (from `Backlog`/`Blocked`) to recenter it fed
    the crossing-minimization pass and **mirrored the entire approved
    layout**. The graph `.dot` was reverted to its approved form and the
    caption became a separate composited graph.
