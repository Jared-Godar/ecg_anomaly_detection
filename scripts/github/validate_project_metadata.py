#!/usr/bin/env python3
"""Validate pull request and linked-issue metadata against Project #5 governance.

Enforces the PR-level and linked-issue-level metadata requirements described in
docs/governance/github-project.md and docs/governance/github-metadata-automation.md:
a pull request must have an assignee, a milestone, a type:* label, an area:* label,
and a closing reference to an issue; that issue must be a member of the tracked
GitHub Project and have every required Project field populated.

This intentionally does not attempt to validate issues at creation time. GitHub
provides no clean rejection mechanism for issue creation comparable to a required
pull-request status check, so issue-only metadata gaps remain a manual-review
concern (see docs/governance/github-metadata-automation.md).

Reading Project V2 field values requires a token with the `project` scope, which
the default GITHUB_TOKEN in a repository-scoped Actions run does not have for a
user-owned project. When that data cannot be read, this script prints a warning
and skips the Project field checks rather than crashing -- unless
--strict-project-checks is passed, in which case an unreadable Project is a hard
failure. This lets the PR-level checks (assignee, milestone, labels, closing
reference) enforce immediately while Project field enforcement is opted into
once a suitably scoped token is configured.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Sequence

REQUIRED_PROJECT_FIELDS: tuple[str, ...] = (
    "Status",
    "Workstream",
    "Issue Type",
    "Priority",
    "Risk",
    "Size",
    "Repository Area",
    "Portfolio Signal",
    "Target Release",
)

# gh CLI normalizes each field's JSON key to a lowercase-first-word version of its
# display name (observed directly: "Issue Type" -> "issue Type", "Repository Area"
# -> "repository Area"). This maps the human-readable names above to those keys.
_FIELD_JSON_KEYS: dict[str, str] = {
    name: name[0].lower() + name[1:] for name in REQUIRED_PROJECT_FIELDS
}

_CLOSING_KEYWORD_PATTERN = re.compile(
    r"\b(?:clos(?:e|es|ed)|fix(?:es|ed)?|resolv(?:e|es|ed))\s*:?\s*#(\d+)",
    re.IGNORECASE,
)


class MetadataValidationError(RuntimeError):
    """Raised when required PR or Project data cannot be read from GitHub."""


@dataclass(frozen=True, slots=True)
class PullRequestMetadata:
    """The subset of PR data this check cares about."""

    number: int
    assignees: tuple[str, ...]
    milestone: str | None
    labels: tuple[str, ...]
    body: str


@dataclass(frozen=True, slots=True)
class ProjectFieldReport:
    """Whether an issue is tracked in the Project and which fields are missing."""

    issue_number: int
    is_project_member: bool
    missing_fields: tuple[str, ...]


def extract_closing_issue_numbers(body: str) -> tuple[int, ...]:
    """Return issue numbers referenced by a GitHub closing keyword, in order, deduplicated."""
    seen: dict[int, None] = {}
    for match in _CLOSING_KEYWORD_PATTERN.finditer(body or ""):
        seen.setdefault(int(match.group(1)), None)
    return tuple(seen)


def _has_label_with_prefix(labels: Sequence[str], prefix: str) -> bool:
    normalized_prefix = prefix.replace(" ", "").lower()
    return any(label.replace(" ", "").lower().startswith(normalized_prefix) for label in labels)


def validate_pull_request(pr: PullRequestMetadata) -> tuple[str, ...]:
    """Return PR-level violations; an empty tuple means the PR-level checks pass."""
    violations: list[str] = []
    if not pr.assignees:
        violations.append("pull request has no assignee")
    if not pr.milestone:
        violations.append("pull request has no milestone")
    if not _has_label_with_prefix(pr.labels, "type:"):
        violations.append("pull request is missing a type:* label")
    if not _has_label_with_prefix(pr.labels, "area:"):
        violations.append("pull request is missing an area:* label")
    if not extract_closing_issue_numbers(pr.body):
        violations.append("pull request body has no closing issue reference (e.g. 'Closes #123')")
    return tuple(violations)


def build_project_field_report(
    issue_number: int, project_items: Sequence[dict[str, Any]]
) -> ProjectFieldReport:
    """Build a field-completeness report for one issue from an already-fetched item list."""
    matching = next(
        (
            item
            for item in project_items
            if item.get("content", {}).get("type") == "Issue"
            and item.get("content", {}).get("number") == issue_number
        ),
        None,
    )
    if matching is None:
        return ProjectFieldReport(issue_number, is_project_member=False, missing_fields=())
    missing = tuple(
        name for name in REQUIRED_PROJECT_FIELDS if not matching.get(_FIELD_JSON_KEYS[name])
    )
    return ProjectFieldReport(issue_number, is_project_member=True, missing_fields=missing)


def validate_project_membership(report: ProjectFieldReport) -> tuple[str, ...]:
    """Return violations for one issue's Project membership and field completeness."""
    if not report.is_project_member:
        return (f"issue #{report.issue_number} is not a member of the tracked Project",)
    if report.missing_fields:
        return (
            f"issue #{report.issue_number} is missing Project fields: "
            + ", ".join(report.missing_fields),
        )
    return ()


def _run_gh(args: list[str]) -> str:
    try:
        result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise MetadataValidationError("gh CLI is not installed or not on PATH") from error
    except subprocess.CalledProcessError as error:
        raise MetadataValidationError(
            f"gh {' '.join(args)} failed: {error.stderr.strip() or error.stdout.strip()}"
        ) from error
    return result.stdout


def fetch_pull_request(pr_number: int, repo: str | None) -> PullRequestMetadata:
    """Fetch a pull request's metadata via the gh CLI."""
    args = ["pr", "view", str(pr_number), "--json", "number,assignees,milestone,labels,body"]
    if repo:
        args.extend(["--repo", repo])
    payload = json.loads(_run_gh(args))
    milestone = payload.get("milestone")
    return PullRequestMetadata(
        number=payload["number"],
        assignees=tuple(assignee["login"] for assignee in payload.get("assignees", [])),
        milestone=milestone.get("title") if milestone else None,
        labels=tuple(label["name"] for label in payload.get("labels", [])),
        body=payload.get("body") or "",
    )


def fetch_project_items(owner: str, project_number: int) -> list[dict[str, Any]]:
    """Fetch every item in the tracked Project, once, for reuse across issue lookups.

    Requires a token with the `project` scope. Raises MetadataValidationError
    (rather than letting a CalledProcessError propagate) so callers can choose
    whether an unreadable Project is a hard failure or an advisory warning.
    """
    args = [
        "project",
        "item-list",
        str(project_number),
        "--owner",
        owner,
        "--format",
        "json",
        "--limit",
        "500",
    ]
    payload = json.loads(_run_gh(args))
    return payload["items"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to the current repository")
    parser.add_argument("--owner", default="Jared-Godar", help="Project owner login")
    parser.add_argument("--project-number", type=int, default=5)
    parser.add_argument(
        "--strict-project-checks",
        action="store_true",
        help=(
            "Treat an unreadable Project (missing or insufficiently scoped token) as a "
            "hard failure instead of a skipped-with-warning advisory check."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        pr = fetch_pull_request(args.pr_number, args.repo)
    except MetadataValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    violations = list(validate_pull_request(pr))
    closing_issues = extract_closing_issue_numbers(pr.body)

    if closing_issues:
        try:
            items = fetch_project_items(args.owner, args.project_number)
        except MetadataValidationError as error:
            message = f"could not read Project #{args.project_number} data: {error}"
            if args.strict_project_checks:
                violations.append(message)
            else:
                print(
                    f"warning: {message}; skipping Project field checks "
                    "(pass --strict-project-checks to make this a hard failure once a "
                    "Project-scoped token is configured)",
                    file=sys.stderr,
                )
        else:
            for issue_number in closing_issues:
                report = build_project_field_report(issue_number, items)
                violations.extend(validate_project_membership(report))

    if violations:
        print("Metadata validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print(f"Metadata validation passed for pull request #{pr.number}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
