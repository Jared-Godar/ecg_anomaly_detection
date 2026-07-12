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

Quota stewardship (issue #173): native pull-request and issue metadata is read
via REST (`gh api repos/...`), keeping this script's GraphQL consumption to
exactly one Project #5 snapshot per run -- fetched once and reused across every
closing issue's membership and field checks. Each closing issue's native
metadata (milestone and state) is likewise fetched once and reused by both the
milestone-inheritance and premature-closure stages. The GraphQL phase is
preflighted against a configurable minimum-remaining threshold, and the run
prints a quota before/after/consumed report so every consumer of the shared
5000-points/hour pool is accountable in its own logs.

Exit codes: 0 all checks passed, 1 metadata violations found, 2 required data
could not be read (a genuine failure: authentication, missing PR), 3 a GraphQL
quota condition -- transient shared-pool infrastructure, never a defect in the
pull request being validated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# scripts/github/ is operational tooling, not an installed package, so the
# shared github_api helper that lives alongside this script is imported by
# putting this script's own directory on sys.path first. That is already true
# when the script runs directly (sys.path[0] is the script's directory) but
# not when the test suite loads this file from its path, so the insertion is
# explicit and idempotent.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Guard against duplicate insertion when both governance scripts are loaded
# into one process (e.g. the test suite imports each of them).
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import github_api  # noqa: E402  (needs the sys.path insertion above)

# The two error classes that mean "the shared API quota, not this pull
# request, is the problem": grouped once so every handler that must let them
# escalate to main()'s dedicated exit path re-raises the same tuple.
_QUOTA_ERRORS: tuple[type[github_api.GitHubApiError], ...] = (
    github_api.PrimaryRateLimitError,
    github_api.GraphQLQuotaInsufficientError,
)

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

# Matches a fenced Markdown code block (```...```), including the language-tag line, so a closing
# keyword quoted inside one -- e.g. a PR body demonstrating another PR's output -- is stripped
# before keyword matching runs. Matched first, before inline spans, so a fenced block's own triple
# backticks aren't mistaken for three single-backtick inline spans.
_FENCED_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)

# Matches a Markdown inline code span (`...`), so a closing keyword quoted as prose -- e.g.
# `` `Closes #154` `` describing a *different* PR's body -- is stripped before keyword matching
# runs. Confined to a single line: GitHub's own inline-code-span rules don't let a span cross a
# newline either.
_INLINE_CODE_SPAN_PATTERN = re.compile(r"`[^`\n]*`")


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
class IssueOverview:
    """One closing issue's native metadata, fetched once and reused across stages.

    Both the milestone-inheritance check and the premature-closure check need
    the same issue's native data; fetching it once per issue (rather than once
    per stage) is the request-deduplication rule from issue #173.
    """

    number: int
    milestone: str | None
    is_closed: bool


@dataclass(frozen=True, slots=True)
class ProjectFieldReport:
    """Whether an issue is tracked in the Project and which fields are missing."""

    issue_number: int
    is_project_member: bool
    missing_fields: tuple[str, ...]


def _strip_markdown_code(text: str) -> str:
    """Blank out fenced code blocks and inline code spans so quoted text within them can't
    be mistaken for a real Markdown directive.

    Replaces each match with a single space rather than deleting it, so a keyword
    immediately before and a reference immediately after a stripped span don't get
    spliced into a new, unintended match.

    Args:
        text: Raw Markdown text (e.g. a PR body).

    Returns:
        text with fenced code blocks and inline code spans replaced by spaces.
    """

    text = _FENCED_CODE_BLOCK_PATTERN.sub(" ", text)
    return _INLINE_CODE_SPAN_PATTERN.sub(" ", text)


def extract_closing_issue_numbers(body: str) -> tuple[int, ...]:
    """Return issue numbers referenced by a GitHub closing keyword, in order, deduplicated.

    Fenced code blocks and inline code spans are stripped before matching, so a closing
    keyword quoted as prose or example text (e.g. `` `Closes #154` `` describing a
    *different* PR) isn't mistaken for a real closing directive in this PR's own body.
    """
    seen: dict[int, None] = {}
    # dict.setdefault preserves first-occurrence order while deduplicating, so a PR
    # body referencing the same issue with two different closing keywords still
    # yields that issue number exactly once.
    for match in _CLOSING_KEYWORD_PATTERN.finditer(_strip_markdown_code(body or "")):
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


def fetch_pull_request(pr_number: int, repo: str | None) -> PullRequestMetadata:
    """Fetch a pull request's native metadata via REST.

    REST rather than `gh pr view` (which is GraphQL-backed) keeps native
    pull-request reads off the shared GraphQL point pool entirely -- see this
    module's quota-stewardship docstring and issue #173. REST reports state in
    lowercase ("open"/"closed"), so it is normalized to the uppercase form the
    rest of this module compares against.
    """
    payload = json.loads(github_api.run_gh(["api", _rest_endpoint(f"pulls/{pr_number}", repo)]))
    milestone = payload.get("milestone")
    return PullRequestMetadata(
        number=payload["number"],
        assignees=tuple(assignee["login"] for assignee in payload.get("assignees", [])),
        milestone=milestone.get("title") if milestone else None,
        labels=tuple(label["name"] for label in payload.get("labels", [])),
        body=payload.get("body") or "",
        state=payload["state"].upper(),
    )


def fetch_issue_overview(issue_number: int, repo: str | None) -> IssueOverview:
    """Fetch one issue's native milestone and open/closed state via REST, in one call.

    One REST read serves both downstream consumers (milestone inheritance and
    the premature-closure check), replacing the two separate GraphQL-backed
    `gh issue view` calls the stages previously made for the same issue --
    the request-deduplication rule from issue #173.

    Args:
        issue_number: The issue to fetch.
        repo: Optional GitHub OWNER/REPO; None lets gh infer it from the current
            directory's Git remote.

    Returns:
        The issue's number, milestone title (None when unmilestoned), and
        whether it is currently closed.
    """

    payload = json.loads(github_api.run_gh(["api", _rest_endpoint(f"issues/{issue_number}", repo)]))
    milestone = payload.get("milestone")
    return IssueOverview(
        number=issue_number,
        milestone=milestone.get("title") if milestone else None,
        # REST reports state as lowercase "open"/"closed".
        is_closed=payload.get("state") == "closed",
    )


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


def fetch_issue_closure_state(overview: IssueOverview, repo: str | None) -> IssueClosureState:
    """Resolve whether an already-fetched issue's closure, if any, is commit-linked.

    A GitHub issue timeline's `closed` event carries a `commit_id` only when the
    closure was caused by a merged commit/PR referencing the issue (e.g. "Closes
    #N"); a direct manual close (via the UI, API, or `gh issue close`) always
    leaves `commit_id` null -- the exact signal observed live for issue #154's
    2026-07-11T03:41:41Z manual closure while PR #155 (its `Closes #154` PR) was
    still open. The issue's own state arrives on the already-fetched overview
    (no additional read), so an issue that's still open never costs the
    timeline lookup either.

    Args:
        overview: The issue's already-fetched native metadata.
        repo: Optional GitHub OWNER/REPO; None lets gh infer it from the current
            directory's Git remote, matching this module's other fetch functions.

    Returns:
        The issue's closure state.
    """

    # An open issue was never closed by anything, merge or otherwise, so there's
    # no timeline event to inspect -- skip the gh call entirely.
    if not overview.is_closed:
        return IssueClosureState(overview.number, is_closed=False, closed_via_commit=False)

    # --paginate --slurp wraps every page of timeline events into one JSON
    # array of pages, so a long-lived issue's full history is scanned rather
    # than only its first page of events. The timeline endpoint is REST.
    endpoint = _rest_endpoint(f"issues/{overview.number}/timeline", repo)
    pages = json.loads(github_api.run_gh(["api", endpoint, "--paginate", "--slurp"]))
    events = [event for page in pages for event in page]
    # An issue can be closed and reopened more than once; the most recent
    # "closed" event describes how it reached its current closed state.
    closed_events = [event for event in events if event.get("event") == "closed"]
    closed_via_commit = bool(closed_events) and closed_events[-1].get("commit_id") is not None
    return IssueClosureState(overview.number, is_closed=True, closed_via_commit=closed_via_commit)


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

    This is the run's single full Project snapshot -- the one legitimately
    board-wide read (issue #173's one-snapshot-per-phase rule), reused across
    every closing issue's membership and field checks rather than re-fetched
    per issue. It is also this script's only GraphQL consumption. Requires a
    token with the `project` scope. Raises GitHubApiError (rather than letting
    a CalledProcessError propagate) so callers can choose whether an
    unreadable Project is a hard failure or an advisory warning.
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
    payload = json.loads(github_api.run_gh(args))
    return payload["items"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the metadata validation entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: PR number, repo, project owner/number, strictness flag,
        and the minimum-remaining GraphQL quota threshold.
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
    parser.add_argument(
        "--min-graphql-quota",
        type=int,
        default=github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA,
        help=(
            "Minimum remaining GraphQL points required by the preflight check before "
            "the Project snapshot is fetched; 0 or below disables the stop (the "
            "before/after report is still printed). Default: "
            f"{github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA}."
        ),
    )
    return parser.parse_args(argv)


def _run_validation(args: argparse.Namespace, monitor: github_api.QuotaMonitor) -> int:
    """Run every validation stage and return the process exit status.

    Split from main() so the quota-condition exit path (exit code 3) can be
    handled in exactly one place there: any stage that hits the primary rate
    limit or the preflight threshold re-raises to main() instead of folding
    the condition into its own stage-local error handling.

    Args:
        args: The parsed command-line arguments.
        monitor: The run's quota monitor; preflighted here, reported by main().

    Returns:
        0 when validation passes, 1 for metadata violations, 2 when required
        data cannot be read.
    """

    # A pull request that can't be fetched at all makes every other check moot;
    # fail immediately rather than attempting partial validation against no data.
    # Quota conditions re-raise to main()'s dedicated exit path first.
    try:
        pr = fetch_pull_request(args.pr_number, args.repo)
    except _QUOTA_ERRORS:
        raise
    except github_api.GitHubApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    closing_issues = extract_closing_issue_numbers(pr.body)

    # Each closing issue's native metadata is fetched exactly once and reused
    # by both the milestone-inheritance and premature-closure stages below --
    # the request-deduplication rule from issue #173.
    overviews: dict[int, IssueOverview] = {}
    require_milestone = True
    # Only fetch closing issues' native metadata (and only then reconsider the
    # PR milestone requirement) when there are closing issues to check in the
    # first place; with none, require_milestone keeps its conservative True default.
    if closing_issues:
        # Same "can't fetch, can't validate" reasoning as the PR fetch above.
        try:
            overviews = {
                number: fetch_issue_overview(number, args.repo) for number in closing_issues
            }
        except _QUOTA_ERRORS:
            raise
        except github_api.GitHubApiError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        require_milestone = closing_issue_milestones_require_pr_milestone(
            tuple(overview.milestone for overview in overviews.values())
        )

    warnings: list[str] = []
    # The premature-closure check (issue #158) only makes sense while this PR is
    # still open: once merged, GitHub's own automation is expected to have
    # closed the issue anyway, and once the PR itself is closed/abandoned there
    # is no longer an open fix in flight for the check to protect.
    if closing_issues and pr.state == "OPEN":
        # A failure here must never fail the whole run (see this check's own
        # non-blocking, observational design in issue #158) -- print a warning
        # and continue rather than propagating like the hard-failing fetches
        # above. Quota conditions still re-raise: a drained shared pool would
        # fail every later stage anyway, and deserves its distinct exit code.
        try:
            closure_states = tuple(
                fetch_issue_closure_state(overview, args.repo) for overview in overviews.values()
            )
        except _QUOTA_ERRORS:
            raise
        except github_api.GitHubApiError as error:
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
        # specially below rather than propagating like other GitHubApiErrors,
        # since the default GITHUB_TOKEN often lacks the scope to read it. The
        # preflight guards the snapshot -- this run's only GraphQL spend -- and
        # its quota conditions re-raise to main()'s dedicated exit path, never
        # masquerading as a metadata or token problem.
        try:
            monitor.preflight()
            items = fetch_project_items(args.owner, args.project_number)
        except _QUOTA_ERRORS:
            raise
        except github_api.GitHubApiError as error:
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

    # One monitor per run: it preflights the GraphQL phase inside
    # _run_validation and reports consumption below -- both reads are REST and
    # free, so accounting can never worsen the quota.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)

    # Quota conditions from any stage exit through this single handler with
    # their own exit code (3) and wording: transient shared-pool
    # infrastructure, resumable by rerunning after the reset, never a
    # metadata defect in the pull request being validated.
    try:
        exit_code = _run_validation(args, monitor)
    except _QUOTA_ERRORS as error:
        print(f"quota: {error}", file=sys.stderr)
        exit_code = 3

    # The consumption report prints on success and failure alike -- a failed
    # run's consumption is exactly the evidence needed when diagnosing a
    # drained pool. A run that never reached its GraphQL phase (no closing
    # issues, or an earlier hard failure) has no baseline and nothing to
    # report; otherwise reporting is best-effort, because a report failure
    # must never mask the run's real outcome.
    if monitor.preflighted:
        # Best-effort only: see the comment above for why a report failure is
        # downgraded to a warning instead of changing the exit code.
        try:
            print(monitor.report(), file=sys.stderr)
        except github_api.GitHubApiError as report_error:
            print(
                f"warning: could not report quota consumption: {report_error}",
                file=sys.stderr,
            )
    return exit_code


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
