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

While the PR itself is open, this script also observationally checks whether any
issue it closes was already closed by a non-merge event (a direct manual close,
not a merged commit/PR referencing it) -- see issue #158. This never fails the
run; it is printed as a separate, non-blocking warning, since an issue closed
independently of its fixing PR is an anomaly worth a human's attention, not
proof that this particular PR's own metadata is incomplete.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
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
    state: str


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


# Short, fixed backoff schedule for GitHub's *secondary* rate limit (its
# short-lived abuse-detection throttle, distinct from the hours-long primary
# point-budget limit below). These delays are deliberately small: the secondary
# limit is documented to clear within seconds, so there is no benefit to a long
# or exponential schedule, only to CI wall-clock time spent waiting.
_SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS: tuple[int, ...] = (2, 5, 10)


def _is_primary_rate_limit_error(message: str) -> bool:
    """True when gh's error text is GitHub's primary (points/hour) rate limit.

    Distinguished from the secondary/abuse-detection limit below because this
    one takes up to an hour to clear -- no retry within a single CI job's
    lifetime can help, so callers should fail fast instead of waiting.
    """

    # GitHub's own wording for this case always includes "rate limit" without
    # the word "secondary"; checking for the absence of "secondary" is what
    # separates this from _is_secondary_rate_limit_error below, since both
    # messages otherwise share the substring "rate limit".
    lowered = message.lower()
    return "rate limit" in lowered and "secondary" not in lowered


def _is_secondary_rate_limit_error(message: str) -> bool:
    """True when gh's error text is GitHub's transient secondary/abuse-detection throttle."""

    return "secondary rate limit" in message.lower()


def _run_gh(args: list[str]) -> str:
    """Run one fixed GitHub CLI command and return its captured output.

    Retries a bounded number of times, with short fixed delays, when gh
    reports GitHub's transient secondary rate limit -- but never for the
    primary (hours-long) rate limit, where retrying inside one CI job cannot
    help and would only waste its runtime.

    Args:
        args: The `gh` subcommand and its arguments (without the leading "gh" itself).

    Returns:
        The command's captured stdout.
    """

    # The first attempt has no delay; each retry after a secondary-rate-limit
    # failure waits progressively longer per _SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS.
    delays = (0, *_SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS)
    # Walk the fixed attempt schedule rather than recursing, so the bound on
    # total attempts is visible directly from `delays` with no separate counter.
    for attempt_index, delay in enumerate(delays):
        # Only a retry (attempt_index > 0) has a delay; the first attempt runs
        # immediately.
        if delay:
            time.sleep(delay)
        # Collapse "gh not installed" (FileNotFoundError) and "gh exited
        # non-zero" (CalledProcessError, since check=True) into one
        # MetadataValidationError, so callers only need to catch this module's
        # own exception type.
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
            message = error.stderr.strip() or error.stdout.strip()
            is_last_attempt = attempt_index == len(delays) - 1
            # A genuine metadata defect and an hours-long token-budget
            # exhaustion must never look the same in CI output, or a human
            # will waste time "fixing" a PR that was never the problem -- so
            # this case fails immediately, without spending any retry.
            if _is_primary_rate_limit_error(message):
                raise MetadataValidationError(
                    "GitHub API rate limit exhausted -- this is a transient "
                    f"infrastructure condition, not a metadata defect: gh {' '.join(args)} "
                    f"failed: {message}"
                ) from error
            # Transient and documented to clear within seconds; loop around to
            # the next (longer) delay instead of failing, unless this was
            # already the last scheduled attempt.
            if _is_secondary_rate_limit_error(message) and not is_last_attempt:
                continue
            raise MetadataValidationError(f"gh {' '.join(args)} failed: {message}") from error
        else:
            return result.stdout
    # Unreachable: `delays` is a fixed non-empty literal, so every iteration of
    # the loop above either returns on success or raises on failure. Kept only
    # so a static checker can see every code path produces or raises a value.
    raise AssertionError("unreachable: _run_gh's retry loop always returns or raises")


def fetch_pull_request(pr_number: int, repo: str | None) -> PullRequestMetadata:
    """Fetch a pull request's metadata via the gh CLI."""
    args = [
        "pr",
        "view",
        str(pr_number),
        "--json",
        "number,assignees,milestone,labels,body,state",
    ]
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
        state=payload["state"],
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


@dataclass(frozen=True, slots=True)
class IssueClosureState:
    """Whether a closing issue is closed, and if so, whether via a merge-linked commit.

    See find_prematurely_closed_issues and issue #158 for why this distinction
    matters: an issue closed independently of the PR that is supposed to fix it
    can be left stuck (e.g. its Project #5 Status field never advances past
    "In Progress", since the merge-triggered status-sync automation never runs
    for a closure it didn't cause) -- confirmed live for issue #154 against PR #155.
    """

    issue_number: int
    is_closed: bool
    closed_via_commit: bool


def fetch_issue_closure_state(issue_number: int, repo: str | None) -> IssueClosureState:
    """Fetch whether an issue is closed, and if closed, whether that closure is commit-linked.

    A GitHub issue timeline's `closed` event carries a `commit_id` only when the
    closure was caused by a merged commit/PR referencing the issue (e.g. "Closes
    #N"); a direct manual close (via the UI, API, or `gh issue close`) always
    leaves `commit_id` null -- the exact signal observed live for issue #154's
    2026-07-11T03:41:41Z manual closure while PR #155 (its `Closes #154` PR) was
    still open. The issue's own state is checked first so an issue that's still
    open never needs the additional timeline lookup.

    Args:
        issue_number: The issue to check.
        repo: Optional GitHub OWNER/REPO; None lets gh infer it from the current
            directory's Git remote, matching this module's other fetch functions.

    Returns:
        The issue's closure state.
    """

    state_args = ["issue", "view", str(issue_number), "--json", "state"]
    # repo is optional; omitting --repo lets gh infer it from the current
    # directory's Git remote, matching this module's other fetch functions.
    if repo:
        state_args.extend(["--repo", repo])
    state_payload = json.loads(_run_gh(state_args))
    # An open issue was never closed by anything, merge or otherwise, so there's
    # no timeline event to inspect -- skip the extra gh call entirely.
    if state_payload.get("state") != "CLOSED":
        return IssueClosureState(issue_number, is_closed=False, closed_via_commit=False)

    # gh api has no --repo flag; an explicit repo is embedded directly in the
    # endpoint path, while the literal "{owner}/{repo}" placeholders (kept
    # unexpanded via f-string brace-escaping) let gh resolve them itself from
    # the current directory's Git remote when repo is None.
    endpoint = (
        f"repos/{repo}/issues/{issue_number}/timeline"
        if repo
        else f"repos/{{owner}}/{{repo}}/issues/{issue_number}/timeline"
    )
    # --paginate --slurp wraps every page of timeline events into one JSON
    # array of pages, so a long-lived issue's full history is scanned rather
    # than only its first page of events.
    pages = json.loads(_run_gh(["api", endpoint, "--paginate", "--slurp"]))
    events = [event for page in pages for event in page]
    # An issue can be closed and reopened more than once; the most recent
    # "closed" event describes how it reached its current closed state.
    closed_events = [event for event in events if event.get("event") == "closed"]
    closed_via_commit = bool(closed_events) and closed_events[-1].get("commit_id") is not None
    return IssueClosureState(issue_number, is_closed=True, closed_via_commit=closed_via_commit)


def find_prematurely_closed_issues(
    closure_states: Sequence[IssueClosureState],
) -> tuple[str, ...]:
    """Return warnings for closing issues closed by a non-merge event while the PR stays open.

    Purely observational (see issue #158 and this module's own docstring): an
    issue closed independently of its fixing PR does not mean the PR itself is
    metadata-incomplete, so these are reported as warnings only, never folded
    into validate_pull_request's violations tuple.
    """

    return tuple(
        f"issue #{state.issue_number} was closed by a non-merge event while this "
        "pull request remains open"
        for state in closure_states
        if state.is_closed and not state.closed_via_commit
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

    warnings: list[str] = []
    # The premature-closure check (issue #158) only makes sense while this PR is
    # still open: once merged, GitHub's own automation is expected to have
    # closed the issue anyway, and once the PR itself is closed/abandoned there
    # is no longer an open fix in flight for the check to protect.
    if closing_issues and pr.state == "OPEN":
        # A failure here must never fail the whole run (see this check's own
        # non-blocking, observational design in issue #158) -- print a warning
        # and continue rather than propagating like the hard-failing fetches above.
        try:
            closure_states = tuple(
                fetch_issue_closure_state(number, args.repo) for number in closing_issues
            )
        except MetadataValidationError as error:
            print(
                f"warning: could not check closing-issue closure state: {error}",
                file=sys.stderr,
            )
        else:
            warnings.extend(find_prematurely_closed_issues(closure_states))

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

    # Observational warnings (issue #158) are printed unconditionally, separately
    # from violations, and never affect the exit code -- they exist to surface an
    # anomaly for human review, not to gate this or any other PR's merge.
    if warnings:
        print("Observational warnings (non-blocking):", file=sys.stderr)
        # List every warning, mirroring the violations list below, so a reviewer
        # sees every flagged anomaly in one pass rather than one at a time.
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)

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
