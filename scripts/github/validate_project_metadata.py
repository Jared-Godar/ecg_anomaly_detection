#!/usr/bin/env python3
"""Validate pull request and linked-issue metadata against Project #5 governance.

Enforces the PR-level and linked-issue-level metadata requirements described in
docs/governance/github-project.md and docs/governance/github-metadata-automation.md:
a pull request must have an assignee, a type:* label, an area:* label, and a
closing reference to an issue; that issue must be a member of the tracked
GitHub Project and have every required Project field populated.

A pull request must also have a milestone -- unless every issue it closes is
itself deliberately unmilestoned, per docs/governance/issue-workflow.md's rule
that a milestone is a delivery commitment assigned only when work requires
one. The milestone requirement is inherited from the closing issue(s) rather
than forced onto the PR, so an issue's own milestone decision is the single
source of truth and cannot drift out of sync the way a separate opt-out label
could.

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# The complete, fixed set of Project #5 fields every tracked issue must have
# populated (see docs/governance/github-project.md); a PR's closing issue(s) must
# satisfy all of these before the PR itself is considered metadata-complete.
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

# Matches GitHub's recognized closing keywords (close/closes/closed, fix/fixes/fixed,
# resolve/resolves/resolved), case-insensitively, followed by an issue reference --
# the same keyword set GitHub itself uses to auto-close an issue when a PR merges,
# so this script only recognizes references that would actually functionally close something.
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
    # dict.setdefault preserves first-occurrence order while deduplicating, so a PR
    # body referencing the same issue with two different closing keywords still
    # yields that issue number exactly once.
    for match in _CLOSING_KEYWORD_PATTERN.finditer(body or ""):
        seen.setdefault(int(match.group(1)), None)
    return tuple(seen)


def _has_label_with_prefix(labels: Sequence[str], prefix: str) -> bool:
    """Return whether any label starts with a given prefix, ignoring spaces and case.

    Normalizing away spaces and case tolerates minor formatting differences between
    how a label is typed in different places (e.g. "Type: Bug" vs "type:bug") while
    still requiring the same semantic prefix.

    Args:
        labels: The PR's label names to search.
        prefix: The required prefix (e.g. "type:" or "area:").

    Returns:
        True if at least one label matches the normalized prefix.
    """

    normalized_prefix = prefix.replace(" ", "").lower()
    return any(label.replace(" ", "").lower().startswith(normalized_prefix) for label in labels)


def validate_pull_request(
    pr: PullRequestMetadata, *, require_milestone: bool = True
) -> tuple[str, ...]:
    """Return PR-level violations; an empty tuple means the PR-level checks pass.

    require_milestone defaults to True so every existing caller keeps the original,
    conservative behavior unless it explicitly opts out based on closing-issue state
    (see closing_issue_milestones_require_pr_milestone).
    """
    violations: list[str] = []
    # An unassigned PR has no clear owner accountable for addressing review feedback.
    if not pr.assignees:
        violations.append("pull request has no assignee")
    # require_milestone is inherited from closing-issue state by the caller (see this
    # function's own docstring); only enforced when that inheritance says it should be.
    if require_milestone and not pr.milestone:
        violations.append("pull request has no milestone")
    # type:* and area:* are both required per docs/governance/github-project.md's
    # PR-level metadata rules, checked independently so both gaps are ever reported.
    if not _has_label_with_prefix(pr.labels, "type:"):
        violations.append("pull request is missing a type:* label")
    # Same reasoning as the type:* check above, for area:*.
    if not _has_label_with_prefix(pr.labels, "area:"):
        violations.append("pull request is missing an area:* label")
    # No recognized closing keyword means this PR won't auto-close any issue on
    # merge, breaking the issue-tracking chain this governance model depends on.
    if not extract_closing_issue_numbers(pr.body):
        violations.append("pull request body has no closing issue reference (e.g. 'Closes #123')")
    return tuple(violations)


def closing_issue_milestones_require_pr_milestone(milestones: Sequence[str | None]) -> bool:
    """Return whether the PR must carry a milestone, inherited from its closing issues.

    Exempt only when every closing issue is itself deliberately unmilestoned. No
    closing issues (or a milestone that could not be determined) fails closed and
    still requires one, matching this check's original, more conservative behavior.
    """
    # No closing issues means there's no issue-level milestone decision to inherit
    # from, so this falls back to the conservative default of requiring one.
    if not milestones:
        return True
    return any(milestone is not None for milestone in milestones)


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
    # An issue absent from the Project's item list isn't tracked at all, so there's
    # no point checking its individual field values -- membership itself is the violation.
    if matching is None:
        return ProjectFieldReport(issue_number, is_project_member=False, missing_fields=())
    missing = tuple(
        name for name in REQUIRED_PROJECT_FIELDS if not matching.get(_FIELD_JSON_KEYS[name])
    )
    return ProjectFieldReport(issue_number, is_project_member=True, missing_fields=missing)


def validate_project_membership(report: ProjectFieldReport) -> tuple[str, ...]:
    """Return violations for one issue's Project membership and field completeness."""
    # Non-membership and missing fields are reported as distinct, mutually exclusive
    # violations -- a non-member issue has no field values to check yet.
    if not report.is_project_member:
        return (f"issue #{report.issue_number} is not a member of the tracked Project",)
    # Only reached for a confirmed Project member; report every missing field together.
    if report.missing_fields:
        return (
            f"issue #{report.issue_number} is missing Project fields: "
            + ", ".join(report.missing_fields),
        )
    return ()


def _run_gh(args: list[str]) -> str:
    """Run one fixed GitHub CLI command and return its captured output.

    Args:
        args: The `gh` subcommand and its arguments (without the leading "gh" itself).

    Returns:
        The command's captured stdout.
    """

    # Collapse "gh not installed" (FileNotFoundError) and "gh exited non-zero"
    # (CalledProcessError, since check=True) into one MetadataValidationError, so
    # callers only need to catch this module's own exception type.
    try:
        # command is a fixed literal ("gh", *args) built from this module's own
        # subcommand arguments, not runtime/user-constructed input.
        result = subprocess.run(  # noqa: S603
            ["gh", *args],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
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
    # repo is optional; omitting --repo lets gh infer it from the current
    # directory's Git remote, matching gh's own default behavior.
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


def fetch_issue_milestone(issue_number: int, repo: str | None) -> str | None:
    """Fetch one issue's own milestone via the gh CLI.

    Reads only the issue's native milestone field, not Project #5 data, so this
    needs no elevated Project-scope token -- the default GITHUB_TOKEN suffices.
    """
    args = ["issue", "view", str(issue_number), "--json", "milestone"]
    # repo is optional; omitting --repo lets gh infer it from the current
    # directory's Git remote, matching gh's own default behavior.
    if repo:
        args.extend(["--repo", repo])
    payload = json.loads(_run_gh(args))
    milestone = payload.get("milestone")
    return milestone.get("title") if milestone else None


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
    """Parse command-line arguments for the metadata validation entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: PR number, repo, project owner/number, and strictness flag.
    """

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
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation straightforward
    to audit.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The value produced by the documented operation.
    """

    args = parse_args(argv)

    # A pull request that can't be fetched at all makes every other check moot;
    # fail immediately rather than attempting partial validation against no data.
    try:
        pr = fetch_pull_request(args.pr_number, args.repo)
    except MetadataValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    closing_issues = extract_closing_issue_numbers(pr.body)

    require_milestone = True
    # Only fetch closing issues' milestones (and only then reconsider the PR
    # milestone requirement) when there are closing issues to check in the first
    # place; with none, require_milestone keeps its conservative True default.
    if closing_issues:
        # Same "can't fetch, can't validate" reasoning as the PR fetch above.
        try:
            closing_milestones = tuple(
                fetch_issue_milestone(number, args.repo) for number in closing_issues
            )
        except MetadataValidationError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        require_milestone = closing_issue_milestones_require_pr_milestone(closing_milestones)

    violations = list(validate_pull_request(pr, require_milestone=require_milestone))

    # Project field checks only make sense when there's at least one closing issue
    # to check them against; a PR with none already failed validate_pull_request's
    # own closing-reference check above.
    if closing_issues:
        # An unreadable Project (see fetch_project_items' own docstring) is handled
        # specially below rather than propagating like other MetadataValidationErrors,
        # since the default GITHUB_TOKEN often lacks the scope to read it.
        try:
            items = fetch_project_items(args.owner, args.project_number)
        except MetadataValidationError as error:
            message = f"could not read Project #{args.project_number} data: {error}"
            # --strict-project-checks opts into treating this as a hard failure;
            # otherwise it's an advisory warning so PR-level checks can still enforce
            # immediately without requiring a Project-scoped token everywhere.
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
            # Check every closing issue's Project membership/fields against the one
            # already-fetched item list, rather than re-fetching per issue.
            for issue_number in closing_issues:
                report = build_project_field_report(issue_number, items)
                violations.extend(validate_project_membership(report))

    # A non-empty violations list means at least one PR-level or issue-level
    # requirement failed; report every one explicitly and exit non-zero so this can
    # gate a required PR status check.
    if violations:
        print("Metadata validation failed:", file=sys.stderr)
        # List every violation, so a contributor can fix them all in one pass
        # instead of re-running this check repeatedly to discover them one at a time.
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print(f"Metadata validation passed for pull request #{pr.number}.")
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
