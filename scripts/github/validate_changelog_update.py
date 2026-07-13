#!/usr/bin/env python3
"""Fail a pull request that changes substantive paths without updating the changelog.

Mechanically enforces the standing CHANGELOG contract (AGENTS.md "Standing
commitments", docs/governance/releases.md): every pull request whose diff
touches a substantive path -- source, scripts, documentation, versioned
configuration, or CI workflows -- must update `CHANGELOG.md` in that same pull
request. The contract previously existed only as reviewer/agent habit and
decayed silently across 25 merged pull requests before the v1.1.0 release-gate
audit caught it (issues #179 backfill, #184 this enforcement).

A pull request that genuinely needs no entry declares that explicitly and
visibly with a marker line in its body:

    changelog: not-needed -- <short reason>

Only the `changelog: not-needed` token is machine-checked (at the start of a
line, any case); the reason is strongly recommended so the exemption is
self-explaining in review and in the merge record. The marker is deliberately
a PR-body line rather than a label: a new `changelog:*` label would join the
strict label taxonomy (`.github/labels.json`, sync and drift tooling) and
duplicate state the PR body can carry directly, where it is versioned with the
description and visible in review. Marker text quoted inside fenced code
blocks or inline code spans is ignored, so a pull request *documenting* the
marker (like the one introducing this script) cannot accidentally exempt
itself.

Entry *content* is out of scope by design (issue #184's non-goals): this gate
checks that `CHANGELOG.md` was touched, not that the entry is good -- style
and accuracy stay human/agent judgment.

Quota stewardship (issue #173 conventions): this script is REST-only -- the
pull request body and changed-file list both come from `gh api repos/...` --
so it consumes zero points from the shared GraphQL pool. The primary REST
rate limit is still classified distinctly (exit code 3) so an exhausted pool
is never misread as a changelog defect in the pull request being validated.

Exit codes: 0 the gate passes (entry present, exemption declared, or no
substantive paths touched), 1 the gate fails (substantive diff with neither a
changelog update nor an exemption marker), 2 required data could not be read
(authentication, missing PR), 3 a rate-limit condition -- transient shared
infrastructure, never a defect in the pull request being validated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

# scripts/github/ is operational tooling, not an installed package, so the
# shared github_api helper that lives alongside this script is imported by
# putting this script's own directory on sys.path first. That is already true
# when the script runs directly (sys.path[0] is the script's directory) but
# not when the test suite loads this file from its path, so the insertion is
# explicit and idempotent.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Guard against duplicate insertion when several governance scripts are loaded
# into one process (e.g. the test suite imports each of them).
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import github_api  # noqa: E402  (needs the sys.path insertion above)

# Path prefixes whose changes are substantive enough to require a changelog
# entry, per issue #184: package source, operational scripts, documentation,
# versioned pipeline configuration, and CI workflows. Everything else (tests,
# lockfiles, editor/agent configuration, the changelog itself) can still be
# mentioned in an entry, but does not by itself demand one.
SUBSTANTIVE_PATH_PREFIXES: tuple[str, ...] = (
    "src/",
    "scripts/",
    "docs/",
    "configs/",
    ".github/workflows/",
)

# The single changelog file this gate requires substantive pull requests to
# touch; repository-root CHANGELOG.md is the only changelog this project keeps.
CHANGELOG_PATH: str = "CHANGELOG.md"

# Matches the explicit exemption marker at the start of a line, any case, with
# or without surrounding whitespace. Only the token itself is matched; any
# reason text after it on the same line is for human readers, not the machine.
_EXEMPTION_MARKER_PATTERN = re.compile(
    r"^\s*changelog:\s*not-needed\b",
    re.IGNORECASE | re.MULTILINE,
)

# Matches a fenced Markdown code block (```...```), including the language-tag
# line, so a marker quoted inside one is stripped before marker matching runs.
# Matched before inline spans so a fenced block's own triple backticks aren't
# mistaken for three single-backtick inline spans. These two patterns
# deliberately duplicate ~10 lines of validate_project_metadata.py rather than
# refactoring that merged, tested module's private helpers into a shared home
# in the same change that introduces this gate; both gates stay independently
# runnable and the duplication is small, fixed, and documented here.
_FENCED_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)

# Matches a Markdown inline code span (`...`), confined to a single line to
# mirror GitHub's own inline-code rules, so a marker quoted as prose is
# stripped before marker matching runs.
_INLINE_CODE_SPAN_PATTERN = re.compile(r"`[^`\n]*`")


def _strip_markdown_code(text: str) -> str:
    """Blank out fenced code blocks and inline code spans in Markdown text.

    Replaces each match with a single space rather than deleting it, so text
    immediately before and after a stripped span cannot be spliced into a new,
    unintended line-start match.

    Args:
        text: Raw Markdown text (e.g. a PR body).

    Returns:
        text with fenced code blocks and inline code spans replaced by spaces.
    """

    text = _FENCED_CODE_BLOCK_PATTERN.sub(" ", text)
    return _INLINE_CODE_SPAN_PATTERN.sub(" ", text)


def has_exemption_marker(body: str | None) -> bool:
    """Return whether a PR body declares the explicit changelog exemption marker.

    Fenced code blocks and inline code spans are stripped first, so a marker
    quoted as documentation or example text is never honored as a real
    exemption -- only a marker written as an actual line of the body counts.

    Args:
        body: The pull request body; None (an absent body) reads as no marker.

    Returns:
        True when a real (non-quoted) `changelog: not-needed` line is present.
    """

    return _EXEMPTION_MARKER_PATTERN.search(_strip_markdown_code(body or "")) is not None


def substantive_paths(paths: Sequence[str]) -> tuple[str, ...]:
    """Return the subset of changed paths that require a changelog entry.

    Args:
        paths: Every path the pull request changes (new paths, plus previous
            paths for renames, so a file moved out of a substantive tree still
            counts as substantive change).

    Returns:
        The paths matching SUBSTANTIVE_PATH_PREFIXES, in input order.
    """

    return tuple(path for path in paths if path.startswith(SUBSTANTIVE_PATH_PREFIXES))


def touches_changelog(paths: Sequence[str]) -> bool:
    """Return whether the changed-path list includes the repository changelog."""

    return CHANGELOG_PATH in paths


def evaluate_changelog_gate(
    paths: Sequence[str], body: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply the gate's decision rule to one PR's changed paths and body.

    Pure so the decision table is directly unit-testable without any GitHub
    access: violations fail the gate (exit 1), notices are informational lines
    the caller prints either way.

    Args:
        paths: Every path the pull request changes.
        body: The pull request body, scanned for the exemption marker.

    Returns:
        A (violations, notices) pair of message tuples; empty violations means
        the gate passes.
    """

    substantive = substantive_paths(paths)
    # No substantive path in the diff means there is nothing to demand an
    # entry for; the gate passes without needing a changelog or a marker.
    if not substantive:
        return (), ("no substantive paths changed; no changelog entry required",)
    # A changelog update in the same PR satisfies the contract directly --
    # the gate never inspects the entry's content (issue #184's non-goal).
    if touches_changelog(paths):
        return (), (f"{CHANGELOG_PATH} is updated alongside the substantive changes",)
    # The explicit exemption marker passes the gate while stating loudly, in
    # the check's own output, that an exemption (not an entry) was used.
    if has_exemption_marker(body):
        return (), (
            "changelog exemption declared via 'changelog: not-needed' marker in the "
            "PR body; no entry enforced",
        )
    # Substantive diff, no changelog, no marker: the exact silent-decay failure
    # mode this gate exists to stop. Name a bounded sample of the substantive
    # paths so the contributor sees *why* the diff is considered substantive.
    sample = ", ".join(substantive[:5])
    # Only annotate truncation when there genuinely are more paths than shown.
    more = f" (+{len(substantive) - 5} more)" if len(substantive) > 5 else ""
    return (
        (
            f"substantive paths changed ({sample}{more}) without a {CHANGELOG_PATH} "
            "update; add an entry under '## Unreleased' or declare the explicit "
            "exemption line 'changelog: not-needed -- <reason>' in the PR body"
        ),
    ), ()


def _rest_endpoint(resource_path: str, repo: str | None) -> str:
    """Build a `gh api` REST endpoint for this repository, with or without an explicit repo.

    gh api has no --repo flag; an explicit repo is embedded directly in the
    endpoint path, while the literal "{owner}/{repo}" placeholders (kept
    unexpanded) let gh resolve them itself from the current directory's Git
    remote when repo is None.

    Args:
        resource_path: The path below the repository, e.g. "pulls/155".
        repo: Optional GitHub OWNER/REPO; None defers to gh's own resolution.

    Returns:
        The endpoint string to pass to `gh api`.
    """

    # The placeholder form is a literal gh feature, not an f-string gap: gh
    # substitutes {owner}/{repo} from the current Git remote at run time.
    prefix = f"repos/{repo}" if repo else "repos/{owner}/{repo}"
    return f"{prefix}/{resource_path}"


def fetch_pull_request_body(pr_number: int, repo: str | None) -> str:
    """Fetch one pull request's body via REST.

    REST rather than `gh pr view` (which is GraphQL-backed) keeps this gate
    entirely off the shared GraphQL point pool -- see the module docstring.

    Args:
        pr_number: The pull request to fetch.
        repo: Optional GitHub OWNER/REPO; None lets gh infer it from the
            current directory's Git remote.

    Returns:
        The PR body text; an absent body comes back as an empty string.
    """

    payload = json.loads(github_api.run_gh(["api", _rest_endpoint(f"pulls/{pr_number}", repo)]))
    return payload.get("body") or ""


def fetch_changed_paths(pr_number: int, repo: str | None) -> tuple[str, ...]:
    """Fetch every path a pull request changes via the REST changed-files listing.

    `--paginate --slurp` wraps every page of the listing into one JSON array
    of pages, so pull requests larger than one page (30 files by default) are
    fully covered rather than silently truncated. For renamed files both the
    new and the previous path are included, so a file renamed out of a
    substantive tree still registers as substantive change.

    Args:
        pr_number: The pull request whose changed files to list.
        repo: Optional GitHub OWNER/REPO; None lets gh infer it from the
            current directory's Git remote.

    Returns:
        Every changed path (new paths plus rename sources), deduplicated,
        in listing order.
    """

    endpoint = _rest_endpoint(f"pulls/{pr_number}/files", repo)
    pages = json.loads(github_api.run_gh(["api", endpoint, "--paginate", "--slurp"]))
    # dict.setdefault preserves first-occurrence order while deduplicating, so
    # a path appearing as both a rename source and a separate change is
    # reported exactly once.
    seen: dict[str, None] = {}
    # Flatten the page array and collect both path fields per file entry.
    for page in pages:
        # Each page is itself a list of file entries; walk them individually.
        for entry in page:
            seen.setdefault(entry["filename"], None)
            # Renames also carry the pre-rename path; a None (non-rename) is
            # simply skipped rather than recorded.
            if entry.get("previous_filename"):
                seen.setdefault(entry["previous_filename"], None)
    return tuple(seen)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the changelog-gate entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: the pull request number and optional repository slug.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation
    straightforward to audit, mirroring the sibling metadata gate.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The exit code documented in the module docstring.
    """

    args = parse_args(argv)

    # A pull request whose body or file list can't be fetched can't be gated
    # at all; fail with the distinct "required data unreadable" code rather
    # than guessing. A primary rate-limit condition gets its own exit code so
    # a drained shared pool is never misread as a changelog defect.
    try:
        body = fetch_pull_request_body(args.pr_number, args.repo)
        paths = fetch_changed_paths(args.pr_number, args.repo)
    except github_api.PrimaryRateLimitError as error:
        print(f"quota: {error}", file=sys.stderr)
        return 3
    except github_api.GitHubApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    violations, notices = evaluate_changelog_gate(paths, body)

    # Notices are informational context (why the gate passed) printed on the
    # success path; they are never mixed into the failure report below.
    for notice in notices:
        print(f"note: {notice}")

    # A non-empty violations tuple means the contract was not met; report and
    # exit non-zero so this can gate a required PR status check.
    if violations:
        print("Changelog gate failed:", file=sys.stderr)
        # List every violation explicitly, mirroring the sibling gate's output
        # shape, so the contributor sees the full remedy in one pass.
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print(f"Changelog gate passed for pull request #{args.pr_number}.")
    return 0


# Standard script entry-point guard: only run main() when executed directly, not
# when imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
