#!/usr/bin/env python3
"""Inset one Graphviz-rendered SVG (a legend, a caption box) into another.

Exists because the two obvious alternatives both fail:

- Putting a legend or caption in the same Graphviz rank system as the main
  graph distorts the approved layout: a tall sidebar node makes its whole
  rank band (and the ranksep gaps around it) grow to match, and invisible
  positioning edges feed the crossing-minimization pass, which can reorder
  or mirror the real nodes.
- Compositing with nested `<svg>` elements trips an rsvg-convert bug that
  visibly clips the inset's text in the combined render even though it
  renders perfectly on its own.

So instead: render each piece as its own independent graph, then splice the
inset's root `<g>` into the base SVG as a plain transformed group -- native
SVG content in the base's own coordinate system, no nested viewports --
placed by measurement (background-polygon geometry plus CLI-supplied insets),
never by eyeballing. An inset placed inside the base's existing empty space
costs zero extra canvas; an inset placed beyond an edge (e.g. a caption box
below the graph) expands the canvas, background, and viewBox just enough to
cover it.

Usage:
    python3 compose_inset.py <base_svg> <inset_svg> <output_svg> \
        --top-inset POINTS (--left-inset POINTS | --right-inset POINTS)

Insets are measured from the base background polygon's edges to the same
edge of the inset's own background bounding box. Base and output may be the
same path to stack several insets. Run before pad_svg.py: this operates on
raw `dot` output, and the composed file then goes through the standard
pad + rsvg-convert steps.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

BG_POLYGON = re.compile(r'<polygon fill="#000c1e" stroke="none" points="([^"]+)"/>')
ROOT_G = re.compile(r'<g id="graph0" class="graph" transform="([^"]+)">')
SVG_TAG = re.compile(
    r'<svg width="([\d.]+)pt" height="([\d.]+)pt"([^>]*)'
    r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.]+) ([\d.]+)"'
)


def _bounds(svg: str, path: Path) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) of the first background polygon,
    in the root <g>'s coordinate system."""
    match = BG_POLYGON.search(svg)
    if match is None:
        raise ValueError(f"{path}: no #000c1e background polygon found")
    points = [tuple(map(float, p.split(","))) for p in match.group(1).split()]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _root_g_content(svg: str, path: Path) -> tuple[str, str]:
    """Return (transform, inner_content) of the Graphviz root <g> element."""
    match = ROOT_G.search(svg)
    if match is None:
        raise ValueError(f"{path}: no Graphviz root <g id=\"graph0\"> found")
    start = match.end()
    end = svg.rfind("</g>")
    return match.group(1), svg[start:end]


def _expand_canvas(svg: str, base_path: Path, x0: float, y0: float, x1: float, y1: float) -> str:
    """Grow the SVG tag, viewBox, and background polygon so the rectangle
    (x0, y0)-(x1, y1) in root-<g> coordinates is inside the canvas."""
    bg_x0, bg_y0, bg_x1, bg_y1 = _bounds(svg, base_path)
    left = max(0.0, bg_x0 - x0)
    top = max(0.0, bg_y0 - y0)
    right = max(0.0, x1 - bg_x1)
    bottom = max(0.0, y1 - bg_y1)
    if not any((left, top, right, bottom)):
        return svg

    tag = SVG_TAG.search(svg)
    if tag is None:
        raise ValueError(f"{base_path}: could not find a Graphviz-style <svg> tag")
    width, height, mid, view_x, view_y, view_w, view_h = tag.groups()
    # Graphviz emits integer pt dimensions and pad_svg.py's parser relies on
    # that -- round the grown canvas up to whole points to stay compatible.
    new_tag = (
        f'<svg width="{math.ceil(float(width) + left + right)}pt"'
        f' height="{math.ceil(float(height) + top + bottom)}pt"{mid}'
        f'viewBox="{float(view_x) - left:.2f} {float(view_y) - top:.2f}'
        f' {float(view_w) + left + right:.2f} {float(view_h) + top + bottom:.2f}"'
    )
    svg = svg.replace(tag.group(0), new_tag, 1)

    nx0, ny0, nx1, ny1 = bg_x0 - left, bg_y0 - top, bg_x1 + right, bg_y1 + bottom
    new_bg = (
        f'<polygon fill="#000c1e" stroke="none"'
        f' points="{nx0},{ny1} {nx0},{ny0} {nx1},{ny0} {nx1},{ny1} {nx0},{ny1}"/>'
    )
    return svg.replace(BG_POLYGON.search(svg).group(0), new_bg, 1)


def compose(
    base_path: Path,
    inset_path: Path,
    output_path: Path,
    top_inset: float,
    left_inset: float | None,
    right_inset: float | None,
) -> None:
    """Write `output_path`: `base_path` with `inset_path` spliced in."""
    base = base_path.read_text()
    inset = inset_path.read_text()

    base_x_min, base_y_min, base_x_max, base_y_max = _bounds(base, base_path)
    inset_x_min, inset_y_min, inset_x_max, inset_y_max = _bounds(inset, inset_path)
    inset_w = inset_x_max - inset_x_min
    inset_h = inset_y_max - inset_y_min

    inset_transform, inset_content = _root_g_content(inset, inset_path)
    # Avoid duplicate element ids once the two documents share one file.
    stem = inset_path.stem
    inset_content = inset_content.replace('id="', f'id="{stem}-')

    # The inset's background-polygon coordinates sit *inside* its root <g>'s
    # own translate, which stays in the transform chain below -- fold it into
    # the bounds so `tx`/`ty` place the visible box, not the pre-translate one.
    translate = re.search(r"translate\(([\d.\-]+)[ ,]([\d.\-]+)\)", inset_transform)
    if translate is None:
        raise ValueError(f"{inset_path}: no translate() in the root <g> transform")
    inset_x_min += float(translate.group(1))
    inset_y_min += float(translate.group(2))

    # Place the inset's background bounding box relative to the base canvas's
    # edges: its top edge `top_inset` points below the canvas top, and its
    # left (or right) edge `left_inset` (`right_inset`) points inside the
    # matching canvas edge.
    if left_inset is not None:
        tx = (base_x_min + left_inset) - inset_x_min
    else:
        tx = (base_x_max - right_inset - inset_w) - inset_x_min
    ty = (base_y_min + top_inset) - inset_y_min

    x0, y0 = tx + inset_x_min, ty + inset_y_min
    base = _expand_canvas(base, base_path, x0, y0, x0 + inset_w, y0 + inset_h)

    inset_group = (
        f'<g id="{stem}-inset" transform="translate({tx:.2f} {ty:.2f}) {inset_transform}">'
        f"{inset_content}</g>"
    )

    end = base.rfind("</g>")
    output_path.write_text(base[:end] + inset_group + "\n" + base[end:])
    print(
        f"inset {inset_path.name} into {base_path.name} at "
        f"x [{x0:.2f}, {x0 + inset_w:.2f}] y [{y0:.2f}, {y0 + inset_h:.2f}] -> {output_path}"
    )


def main(argv: list[str] | None = None) -> int:
    """Compose base + inset per CLI arguments; return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_svg", type=Path)
    parser.add_argument("inset_svg", type=Path)
    parser.add_argument("output_svg", type=Path)
    parser.add_argument(
        "--top-inset",
        type=float,
        required=True,
        help="points between the base canvas's top edge and the inset's top edge",
    )
    anchor = parser.add_mutually_exclusive_group(required=True)
    anchor.add_argument(
        "--left-inset",
        type=float,
        help="points between the base canvas's left edge and the inset's left edge",
    )
    anchor.add_argument(
        "--right-inset",
        type=float,
        help="points between the base canvas's right edge and the inset's right edge",
    )
    args = parser.parse_args(argv)

    compose(
        args.base_svg,
        args.inset_svg,
        args.output_svg,
        args.top_inset,
        args.left_inset,
        args.right_inset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
