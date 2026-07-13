"""Protect repository-relative links, visual assets, and callouts in public notebooks."""

from __future__ import annotations

import re
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
# Markdown and inline HTML are both supported in Jupyter Markdown cells. Extract both
# forms so a visually correct link cannot silently point outside the repository tree.
MARKDOWN_TARGET = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+['\"].*?['\"])?\)")
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


def _markdown(notebook_path: Path) -> str:
    """Return all Markdown-cell source from one public notebook in display order."""
    notebook = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")


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
