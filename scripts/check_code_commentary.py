#!/usr/bin/env python3
"""Enforce the repository's intentionally exhaustive internal-commentary policy.

The portfolio uses comments as reviewer-facing navigation, not merely as occasional
notes for unusually difficult algorithms. This checker therefore applies a stricter
standard than a conventional linter: every supported Python definition needs a
docstring, every meaningful control-flow or resource-management block needs an
immediately preceding explanatory comment, and every module-level assignment needs
an adjacent comment describing the object or constant it creates.

The immutable historical archive and generated or third-party code are outside this
policy. The caller supplies explicit roots, which default to the supported ``src``,
``scripts``, and ``tests`` trees.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

# These definition nodes can all carry Python docstrings and therefore share one
# fail-closed rule. Including nested definitions prevents documentation coverage from
# looking complete while small inner helpers remain opaque.
DOCUMENTED_DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

# These statements start implementation blocks whose intent and failure boundary are
# easier to review when a nearby comment explains both the operation and its reason.
COMMENTED_BLOCKS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)

# Repository roots are explicit rather than discovered from Git so the check remains
# deterministic in source archives, isolated notebook worktrees, and CI checkouts.
DEFAULT_ROOTS = (Path("src"), Path("scripts"), Path("tests"))


@dataclass(frozen=True, slots=True)
class CommentaryViolation:
    """Describe one missing internal-documentation requirement at a source location."""

    path: Path
    line: int
    message: str

    def render(self) -> str:
        """Return a stable, compiler-style diagnostic for terminal and CI output."""

        return f"{self.path.as_posix()}:{self.line}: {self.message}"


def _comment_only_lines(path: Path) -> set[int]:
    """Return line numbers containing standalone comments in one Python source file.

    Inline comments are intentionally excluded because a trailing note cannot orient a
    reviewer before the implementation block begins. Tokenization is used instead of
    string matching so ``#`` characters inside strings are never misclassified.
    """

    comment_lines: set[int] = set()
    # Bound the binary source handle to tokenization so it closes on every path.
    with path.open("rb") as source:
        # Inspect lexical tokens so comments remain distinguishable from string data.
        for token in tokenize.tokenize(source.readline):
            # Require the comment to be the first non-whitespace content on its line.
            if token.type == tokenize.COMMENT and token.line[: token.start[1]].strip() == "":
                comment_lines.add(token.start[0])
    return comment_lines


def _previous_content_line(lines: list[str], line_number: int) -> int | None:
    """Locate the nearest preceding non-blank source line, if one exists."""

    candidate = line_number - 1
    # Skip visual whitespace because formatters may separate related narrative blocks.
    while candidate > 0 and not lines[candidate - 1].strip():
        candidate -= 1
    return candidate or None


def _has_leading_comment(lines: list[str], comment_lines: set[int], line_number: int) -> bool:
    """Return whether a statement is introduced by a standalone explanatory comment."""

    previous = _previous_content_line(lines, line_number)
    return previous in comment_lines if previous is not None else False


def _is_elif(lines: list[str], node: ast.If) -> bool:
    """Return whether an ``If`` node represents an ``elif`` continuation.

    Python's AST models ``elif`` as a nested ``If`` node. Treating it as a fresh block
    would demand a comment between ``if`` branches where Python syntax allows none.
    """

    return lines[node.lineno - 1].lstrip().startswith("elif ")


def audit_file(path: Path) -> tuple[CommentaryViolation, ...]:
    """Audit one supported Python file for exhaustive docstrings and guide comments."""

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    # Convert syntax failures into policy diagnostics instead of uncaught tracebacks.
    try:
        # Parse first so malformed Python produces one clear policy diagnostic.
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return (
            CommentaryViolation(
                path,
                error.lineno or 1,
                f"cannot audit invalid Python: {error.msg}",
            ),
        )

    violations: list[CommentaryViolation] = []
    # A module docstring gives every file an immediate purpose and boundary statement.
    if ast.get_docstring(tree, clean=False) is None:
        violations.append(CommentaryViolation(path, 1, "module is missing a docstring"))

    comment_lines = _comment_only_lines(path)
    # Walk nested definitions and blocks so no private or inner implementation escapes.
    for node in ast.walk(tree):
        # Require substantive placement first; qualitative review remains file-by-file.
        if (
            isinstance(node, DOCUMENTED_DEFINITIONS)
            and ast.get_docstring(node, clean=False) is None
        ):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            violations.append(
                CommentaryViolation(
                    path, node.lineno, f"{kind} {node.name!r} is missing a docstring"
                )
            )

        # Report a block only when it is independently commentable and the lexical scan found no
        # preceding standalone comment; parent narration does not document nested decisions.
        if (
            isinstance(node, COMMENTED_BLOCKS)
            and not (isinstance(node, ast.If) and _is_elif(lines, node))
            and not _has_leading_comment(lines, comment_lines, node.lineno)
        ):
            violations.append(
                CommentaryViolation(
                    path,
                    node.lineno,
                    f"{type(node).__name__} block is missing a leading explanation comment",
                )
            )

    # Module-level objects cannot carry normal docstrings, so require adjacent narration.
    for statement in tree.body:
        # Restrict this requirement to assignment forms that create module-level objects. Imports
        # and definitions already communicate their names, and definitions carry docstrings.
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and not _has_leading_comment(
            lines, comment_lines, statement.lineno
        ):
            violations.append(
                CommentaryViolation(
                    path,
                    statement.lineno,
                    "module-level assignment is missing a leading explanation comment",
                )
            )

    # Stable ordering makes local output and CI failures easy to compare. When a
    # definition begins on line one, report the enclosing module contract first.
    return tuple(
        sorted(
            violations,
            key=lambda item: (
                item.line,
                0 if item.message == "module is missing a docstring" else 1,
                item.message,
            ),
        )
    )


def discover_python_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return every supported Python file beneath the configured repository roots."""

    files: set[Path] = set()
    # Resolve each explicit root independently so a missing tree fails visibly below.
    for root in roots:
        # A direct file supports focused tests and one-off local investigation.
        # Branch explicitly so files and recursive directory roots cannot overlap.
        if root.is_file() and root.suffix == ".py":
            files.add(root)
        # A directory contributes all Python descendants in deterministic path order.
        elif root.is_dir():
            files.update(root.rglob("*.py"))
    return tuple(sorted(files))


def audit_roots(roots: tuple[Path, ...]) -> tuple[CommentaryViolation, ...]:
    """Audit all discovered Python files and return one stable violation sequence."""

    violations: list[CommentaryViolation] = []
    # Keep file audits isolated so every problem is reported in one invocation.
    for path in discover_python_files(roots):
        violations.extend(audit_file(path))
    return tuple(violations)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse optional source roots for focused or repository-wide commentary checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help="Python files or directories to audit (default: src scripts tests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the commentary audit and return a conventional process exit status."""

    args = parse_args(argv)
    roots = tuple(args.roots)
    files = discover_python_files(roots)
    # Treat an empty discovery set as configuration failure rather than false success.
    if not files:
        print("error: commentary audit found no Python files", file=sys.stderr)
        return 2

    violations = audit_roots(roots)
    # Emit every violation so one remediation pass can address the complete inventory.
    if violations:
        print(
            f"Code commentary audit failed: {len(violations)} violation(s) "
            f"across {len(files)} file(s).",
            file=sys.stderr,
        )
        # Preserve deterministic diagnostics for review and automated parsing.
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        return 1

    print(f"Code commentary audit passed for {len(files)} Python file(s).")
    return 0


# Keep import behavior side-effect free while providing a normal executable script boundary.
if __name__ == "__main__":
    raise SystemExit(main())
