#!/usr/bin/env python3
"""Pad a Graphviz-generated SVG's canvas and background with solid color on all sides.

Graphviz's own `margin` graph attribute is unreliable once HTML-like edge labels are
present in the graph -- it silently stops expanding the background polygon to match
the requested margin (confirmed empirically: the background polygon collapsed to a
~4pt margin regardless of a 0.75in `margin` attribute once TABLE-based edge labels
were added). This script pads the already-rendered SVG directly instead, independent
of Graphviz's internal bounding-box computation, so diagram content changes can never
silently shrink the intended margin again.

Usage:
    python3 pad_svg.py <svg_path> [--pad POINTS] [--bg '#000C1E']
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def pad_svg(svg_path: Path, pad: float, bg: str) -> None:
    """Rewrite one Graphviz SVG in place with `pad` points of `bg`-colored margin."""
    svg = svg_path.read_text()

    svg_tag_pattern = re.compile(
        r'<svg width="(\d+)pt" height="(\d+)pt"([^>]*)'
        r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.]+) ([\d.]+)"'
    )
    match = svg_tag_pattern.search(svg)
    if match is None:
        raise ValueError(f"{svg_path}: could not find a Graphviz-style <svg> tag to pad")
    width, height, mid, view_x, view_y, view_w, view_h = match.groups()
    width, height = int(width), int(height)
    view_x, view_y, view_w, view_h = (float(view_x), float(view_y), float(view_w), float(view_h))

    new_width, new_height = width + 2 * pad, height + 2 * pad
    new_view_x, new_view_y = view_x - pad, view_y - pad
    new_view_w, new_view_h = view_w + 2 * pad, view_h + 2 * pad
    new_svg_tag = (
        f'<svg width="{new_width}pt" height="{new_height}pt"{mid}'
        f'viewBox="{new_view_x:.2f} {new_view_y:.2f} {new_view_w:.2f} {new_view_h:.2f}"'
    )
    svg = svg.replace(match.group(0), new_svg_tag, 1)

    # The first "<polygon fill="{bg}" stroke="none" ...>" is Graphviz's own background
    # rect, drawn in the *inner* (post-transform) coordinate system used by the root
    # <g> element -- expand it by `pad` in that same coordinate system, and insert a
    # fresh copy just before it so the enlarged rect paints first (bottom of z-order).
    bg_pattern = re.compile(rf'<polygon fill="{re.escape(bg.lower())}" stroke="none" points="([^"]+)"/>')
    bg_match = bg_pattern.search(svg)
    if bg_match is None:
        raise ValueError(f"{svg_path}: could not find the background polygon (fill={bg})")
    points = [tuple(map(float, p.split(","))) for p in bg_match.group(1).split()]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    new_rect = f'<polygon fill="{bg.lower()}" stroke="none" points="{x0},{y1} {x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
    svg = svg.replace(bg_match.group(0), new_rect + bg_match.group(0), 1)

    svg_path.write_text(svg)


def main(argv: list[str] | None = None) -> int:
    """Pad the given SVG file in place; return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg_path", type=Path)
    parser.add_argument("--pad", type=float, default=40.0, help="padding in points (default: 40)")
    parser.add_argument("--bg", default="#000C1E", help="background color (default: #000C1E)")
    args = parser.parse_args(argv)

    pad_svg(args.svg_path, args.pad, args.bg)
    print(f"padded {args.svg_path} by {args.pad}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
