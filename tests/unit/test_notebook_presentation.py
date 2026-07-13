"""Protect repository-relative links, visual assets, and callouts in public notebooks."""

from __future__ import annotations

import html
import re
import struct
from collections import defaultdict
from pathlib import Path

import nbformat
import pytest

# Resolve from this test module so the assertions exercise the active checkout rather
# than relying on a caller's working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Keep the supported public sequence explicit: every test below must cover all three
# ordered notebooks, including Step 0's established visual treatment.
PUBLIC_NOTEBOOKS = (
    REPOSITORY_ROOT / "notebooks/00-environment-setup-and-artifact-generation.ipynb",
    REPOSITORY_ROOT / "notebooks/01-narrative-walkthrough.ipynb",
    REPOSITORY_ROOT / "notebooks/02-high-performing-gradient-boosting-validation.ipynb",
)
# Keep execution-UX assertions anchored to the ordered public workflow without
# duplicating repository-relative paths outside PUBLIC_NOTEBOOKS.
STEP0_NOTEBOOK, STEP1_NOTEBOOK, STEP2_NOTEBOOK = PUBLIC_NOTEBOOKS
# The approved Step 1/Step 2 banners must retain notebook 00's exact raster
# contract, so future replacements cannot silently drift in size or PNG mode.
BANNER_PATHS = (
    REPOSITORY_ROOT / "docs/assets/ecg-first-time-environment-setup-banner.png",
    REPOSITORY_ROOT / "docs/assets/ecg-notebook-narrative-walkthrough-banner.png",
    REPOSITORY_ROOT / "docs/assets/ecg-gradient-boosting-validation-banner.png",
)
# The primary DOT file is the editable factual pipeline source.
LINEAGE_SOURCE = REPOSITORY_ROOT / "docs/diagrams/src/modern-pipeline-lineage.dot"
# The separate legend source is composited through the documented workflow.
LINEAGE_LEGEND_SOURCE = REPOSITORY_ROOT / "docs/diagrams/src/modern-pipeline-lineage-legend.dot"
# This stable allowlisted path is the SVG notebook 01 actually renders.
LINEAGE_EXPORT = REPOSITORY_ROOT / "reports/figures/modern-pipeline-lineage.svg"
# The approved local-flow export defines the visual palette Issue 194 reuses.
REFERENCE_DIAGRAM = REPOSITORY_ROOT / "docs/diagrams/exports/local-flow-artifact-zones.svg"
# PNG files always begin with this eight-byte signature before their IHDR chunk.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Markdown and inline HTML are both supported in Jupyter Markdown cells. Extract both
# forms so a visually correct link cannot silently point outside the repository tree.
MARKDOWN_TARGET = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+['\"].*?['\"])?\)")
# Capture image alt text independently from general link extraction so banner
# accessibility cannot regress while the target path itself still resolves.
MARKDOWN_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)")
# Inline callout links use raw HTML, so inspect href/src attributes in addition to the
# standard Markdown form above.
HTML_TARGET = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
# Capture the accessibility role, accessible name, shared inline style, and content
# separately so failures identify which part of the visual contract drifted.
CALLOUT = re.compile(
    r'<div role="(?P<role>alert|note)" aria-label="(?P<label>[^"]+)" '
    r'style="(?P<style>[^"]+)">(?P<body>.*?)</div>',
    re.DOTALL,
)
# Hex colors in exported SVGs are normalized to lowercase by Graphviz. Comparing
# sets makes palette inheritance explicit without depending on element ordering.
HEX_COLOR = re.compile(r"#[0-9a-f]{6}")


def _markdown(notebook_path: Path) -> str:
    """Return all Markdown-cell source from one public notebook in display order."""
    notebook = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")


def _code_cells(notebook_path: Path) -> tuple[nbformat.NotebookNode, ...]:
    """Return code cells from one public notebook in execution order."""
    notebook = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    return tuple(cell for cell in notebook.cells if cell.cell_type == "code")


def _png_ihdr(path: Path) -> tuple[int, int, int, int, int, int, int]:
    """Return width, height, and five format fields from one PNG's IHDR chunk."""
    header = path.read_bytes()[:29]
    # A short file, wrong signature, or missing leading IHDR chunk is not the
    # approved PNG asset contract and should fail with a path-specific message.
    if len(header) != 29 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG with a leading IHDR chunk: {path}")
    return struct.unpack(">IIBBBBB", header[16:29])


def _local_targets(markdown: str) -> set[str]:
    """Return repository-local link and image targets found in Markdown or HTML."""
    targets = set(MARKDOWN_TARGET.findall(markdown)) | set(HTML_TARGET.findall(markdown))
    # Fragments and external schemes do not resolve against the notebook directory.
    return {
        target.split("#", maxsplit=1)[0]
        for target in targets
        if target and not target.startswith(("#", "data:", "http://", "https://", "mailto:"))
    }


@pytest.mark.parametrize("notebook_path", PUBLIC_NOTEBOOKS, ids=lambda path: path.stem)
def test_public_notebook_local_targets_resolve(notebook_path: Path) -> None:
    """Every local Markdown/HTML link and image must resolve from its notebook directory."""
    unresolved = [
        target
        for target in sorted(_local_targets(_markdown(notebook_path)))
        if not (notebook_path.parent / target).resolve().exists()
    ]

    assert unresolved == []


def test_public_notebook_visual_assets_are_linked() -> None:
    """The three banners and Step 1 lineage diagram remain repository-owned linked assets."""
    combined_markdown = "\n".join(_markdown(path) for path in PUBLIC_NOTEBOOKS)

    assert "../docs/assets/ecg-first-time-environment-setup-banner.png" in combined_markdown
    assert "../docs/assets/ecg-notebook-narrative-walkthrough-banner.png" in combined_markdown
    assert "../docs/assets/ecg-gradient-boosting-validation-banner.png" in combined_markdown
    assert "../reports/figures/modern-pipeline-lineage.svg" in combined_markdown
    assert "../docs/diagrams/src/modern-pipeline-lineage.dot" in combined_markdown
    assert "../docs/diagrams/design-spec.md" in combined_markdown


def test_public_notebook_banners_preserve_raster_and_accessibility_contract() -> None:
    """All approved banners remain 1280×640, 8-bit RGB PNGs with meaningful alt text."""
    # IHDR color type 2 is truecolor RGB without alpha; compression, filtering,
    # and interlace method 0 are the baseline notebook 00 asset conventions.
    expected_ihdr = (1280, 640, 8, 2, 0, 0, 0)
    assert {_png_ihdr(path) for path in BANNER_PATHS} == {expected_ihdr}

    combined_markdown = "\n".join(_markdown(path) for path in PUBLIC_NOTEBOOKS)
    images = {
        match.group("target"): match.group("alt").strip()
        for match in MARKDOWN_IMAGE.finditer(combined_markdown)
    }
    # Match by repository filename because each notebook references its banner
    # relatively, while BANNER_PATHS above are absolute test-fixture paths.
    for banner_path in BANNER_PATHS:
        matching_alt_text = [
            alt for target, alt in images.items() if target.endswith(banner_path.name)
        ]
        assert len(matching_alt_text) == 1
        assert len(matching_alt_text[0]) >= 40


@pytest.mark.parametrize("notebook_path", PUBLIC_NOTEBOOKS, ids=lambda path: path.stem)
def test_public_notebook_callouts_share_accessible_structure(notebook_path: Path) -> None:
    """Important panels expose semantic roles, labels, contrast colors, and visible headings."""
    callouts = tuple(CALLOUT.finditer(_markdown(notebook_path)))

    assert callouts
    assert {match.group("role") for match in callouts} == {"alert", "note"}
    # Validate every panel, rather than only the first match, because each notebook has
    # multiple severity levels and each panel must remain independently accessible.
    for match in callouts:
        style = match.group("style")
        body = match.group("body")
        assert match.group("label").strip()
        assert "border-left-width: 8px" in style
        assert "background:" in style
        assert "color:" in style
        assert "line-height: 1.5" in style
        assert '<strong style="display: block;' in body


def test_public_notebook_callout_styles_are_consistent_by_semantic_role() -> None:
    """Every alert shares one style and every note shares one style across the workflow."""
    styles_by_role: defaultdict[str, set[str]] = defaultdict(set)
    # Aggregate all panels before asserting so a one-off drift in any notebook is
    # compared against the same role's treatment everywhere else.
    for notebook_path in PUBLIC_NOTEBOOKS:
        # Record each panel independently so multiple panels in one notebook cannot
        # hide a local style drift behind another notebook's matching panel.
        for match in CALLOUT.finditer(_markdown(notebook_path)):
            styles_by_role[match.group("role")].add(match.group("style"))

    assert set(styles_by_role) == {"alert", "note"}
    assert all(len(styles) == 1 for styles in styles_by_role.values())


def test_lineage_source_and_export_preserve_reference_style_and_boundaries() -> None:
    """The tracked lineage system retains the reference palette and factual claim limits."""
    reference_svg = REFERENCE_DIAGRAM.read_text(encoding="utf-8")
    lineage_svg = LINEAGE_EXPORT.read_text(encoding="utf-8")
    # Graphviz encodes punctuation such as hyphens as numeric XML entities; decode
    # them before comparing rendered human-readable semantics with DOT labels.
    readable_lineage_svg = html.unescape(lineage_svg)
    lineage_source = LINEAGE_SOURCE.read_text(encoding="utf-8")
    legend_source = LINEAGE_LEGEND_SOURCE.read_text(encoding="utf-8")

    # The modern lineage export inherits every color in the approved local-flow
    # diagram, with amber additions reserved for the protected boundary.
    assert set(HEX_COLOR.findall(reference_svg)) <= set(HEX_COLOR.findall(lineage_svg))

    # Check both editable source and public export: source-only wording could fail
    # to render, while export-only wording could become impossible to regenerate.
    for semantic_text in (
        "Subject-Aware Split",
        "Training Only",
        "Validation Only",
        "Protected Test Partition",
        "Unopened, Unreported",
        "educational pipeline evidence",
        "not clinical",
        "benchmark evidence",
    ):
        assert semantic_text.lower() in lineage_source.lower()
        assert semantic_text.lower() in readable_lineage_svg.lower()
    assert "Protected Boundary" in legend_source
    assert 'id="modern-pipeline-lineage-legend-inset"' in lineage_svg


def test_step2_long_phases_use_concise_qualified_progress_feedback() -> None:
    """Step 2 reports three bounded stages with estimates explicitly qualified by local factors."""
    code_cells = _code_cells(STEP2_NOTEBOOK)
    combined_code = "\n".join(cell.source for cell in code_cells)

    # Reuse the package's tested, immediately flushed reporter rather than growing a
    # second notebook-only timing implementation with subtly different behavior.
    assert "from ecg_anomaly_detection.progress import ProgressReporter" in combined_code
    assert "NOTEBOOK_PROGRESS = ProgressReporter(stream=sys.stdout)" in combined_code

    expected_phases = (
        ("def load_partition", "artifact size and disk speed vary"),
        ("model.fit(", "data size and hardware vary"),
        ("model.predict(", "validation size and hardware vary"),
    )
    # Each long phase gets exactly one context-managed start/completion pair and one
    # broad expectation. This prevents per-shard/per-iteration output from creeping in.
    for phase_marker, qualifying_text in expected_phases:
        matching_cells = [cell.source for cell in code_cells if phase_marker in cell.source]
        assert len(matching_cells) == 1
        assert matching_cells[0].count("NOTEBOOK_PROGRESS.stage(") == 1
        assert matching_cells[0].count("_stage.detail(") == 1
        assert qualifying_text in matching_cells[0]

    # No interim reporter notes are needed: three start lines and three measured
    # completion lines are the complete runtime-feedback surface.
    assert combined_code.count("NOTEBOOK_PROGRESS.stage(") == 3
    assert "NOTEBOOK_PROGRESS.note(" not in combined_code

    # Progress must remain observational: lock the fixed estimator configuration
    # and the single fit/predict path that produce the notebook's material results.
    for configuration_line in (
        "learning_rate=0.015",
        "max_leaf_nodes=31",
        "min_samples_leaf=30",
        "l2_regularization=0.1",
        "max_iter=450",
        "random_state=0",
    ):
        assert configuration_line in combined_code
    assert combined_code.count("model.fit(") == 1
    assert combined_code.count("model.predict(") == 1


def test_step0_preserves_streaming_with_qualified_runtime_guidance() -> None:
    """Step 0 qualifies bootstrap/pipeline timing and flushes existing streamed progress."""
    code_cells = _code_cells(STEP0_NOTEBOOK)
    combined_code = "\n".join(cell.source for cell in code_cells)

    # Bootstrap keeps one expectation and one measured completion instead of
    # printing dependency-level estimates that would quickly become inaccurate.
    assert "first installs vary with network and cache" in combined_code
    assert "Environment bootstrap complete after %s" in combined_code
    assert "bootstrap_elapsed" in combined_code

    pipeline_cells = [cell.source for cell in code_cells if "def stream_pipeline" in cell.source]
    assert len(pipeline_cells) == 1
    pipeline_source = pipeline_cells[0]
    # The first-run range is broad and paired with explicit local variables; it is
    # planning guidance, while the existing CLI stream remains the live detail.
    assert "several minutes to tens of minutes on a first local run" in pipeline_source
    assert "Download speed, cache state, record count, CPU, and disk vary" in pipeline_source
    assert 'print(line, end="", flush=True)' in pipeline_source
    assert "Pipeline process ended after" in pipeline_source


def test_step1_limits_progress_to_variable_local_evidence_discovery() -> None:
    """Step 1 reports only its optional run-evidence scan and leaves quick narrative cells quiet."""
    code_cells = _code_cells(STEP1_NOTEBOOK)
    combined_code = "\n".join(cell.source for cell in code_cells)

    # Step 1 reuses the same tested reporter as Step 2, but only the one directory
    # scan can vary meaningfully with a reviewer's accumulated ignored local runs.
    assert "from ecg_anomaly_detection.progress import ProgressReporter" in combined_code
    assert "NARRATIVE_PROGRESS = ProgressReporter(stream=sys.stdout)" in combined_code
    assert combined_code.count("NARRATIVE_PROGRESS.stage(") == 1
    assert "number of local runs and disk speed vary" in combined_code
    assert "NARRATIVE_PROGRESS.note(" not in combined_code


@pytest.mark.parametrize("notebook_path", PUBLIC_NOTEBOOKS, ids=lambda path: path.stem)
def test_public_notebooks_are_committed_without_runtime_state(notebook_path: Path) -> None:
    """Public notebooks retain source only; local progress, results, and plots stay uncommitted."""
    code_cells = _code_cells(notebook_path)

    # Execution counts and outputs are checked independently so a partially cleaned
    # notebook cannot pass merely because one of the two runtime-state forms is empty.
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)
