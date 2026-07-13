#!/usr/bin/env python3
"""Shared, operation-scoped GitHub API access layer for the governance scripts.

Owns the GitHub CLI plumbing that `set_merged_project_status.py` and
`validate_project_metadata.py` previously each duplicated, plus the GraphQL
quota safeguards added by issue #173:

- one `run_gh` wrapper with bounded retries for GitHub's transient conditions
  (the secondary rate limit and, since issue #190, transient 5xx server
  errors) and fail-fast primary-rate-limit classification (raised as the
  distinct `PrimaryRateLimitError` so callers can separate an hours-long
  quota drain from a genuine metadata defect or an ordinary command failure);
- `QuotaMonitor`, a REST-based (`gh api rate_limit` -- free, the endpoint is
  documented not to count against any limit) GraphQL quota preflight with a
  configurable minimum-remaining threshold and before/after/consumed
  reporting;
- `ProjectClient`, an operation-scoped Project V2 accessor that caches
  schema lookups (field and option ids) and at most one full item snapshot
  per instance, performs a bounded *targeted* pull-request -> project-item
  lookup instead of scanning a full board read, and reads one item's
  single-select field back via a targeted GraphQL `node(id:)` query.

Consumption model this module enforces (docs/governance/github-project.md):
schema/identity lookups and the optional full snapshot are cached for the
lifetime of one client (one logical phase); mutation *read-backs* are
deliberately never cached, because a stale cached value would defeat the
read-back-verified mutation rule from #164/#170 entirely.

This file lives in `scripts/github/` (not `src/`) because it is operational
governance tooling, not part of the installed `ecg_anomaly_detection`
package; the scripts alongside it import it by file-system adjacency.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class GitHubApiError(RuntimeError):
    """Base error for every failure this module (or its script callers) raises.

    The governance scripts subclass this for their own script-level defects, so
    one `except GitHubApiError` in a script's `main` catches both layers.
    """


class PrimaryRateLimitError(GitHubApiError):
    """GitHub's primary (points-per-hour) rate limit is already exhausted.

    Raised instead of the base error so callers can map it to a distinct exit
    path: this is transient shared-pool infrastructure, not a metadata defect,
    and no retry within one CI job's lifetime can clear it. gh's wording is
    the same for the GraphQL and REST pools, so this class covers whichever
    pool the failing command was drawing from (for these governance scripts,
    in practice the shared GraphQL pool).
    """


class GraphQLQuotaInsufficientError(GitHubApiError):
    """Remaining GraphQL quota is below the caller's configured safety threshold.

    Raised by `QuotaMonitor.preflight` *before* any expensive or mutating work
    starts, so a run stops safely rather than half-completing a mutation phase
    on a nearly drained pool.
    """


# Default minimum GraphQL points that must remain before a governance script
# starts its GraphQL phase. Sized against measured reality (2026-07-12, live):
# one full Project #5 snapshot (`gh project item-list --limit 500`) cost 203
# points -- GraphQL pricing scales with requested node counts, not calls --
# while the targeted read-backs cost ~1 point each. The threshold must cover
# the most expensive phase it guards (the validator's single snapshot), so 250
# clears that with margin while staying at 5% of the 5000/hour shared pool,
# never blocking runs on a merely busy pool.
DEFAULT_MINIMUM_GRAPHQL_QUOTA: int = 250

# Short, fixed backoff schedule for GitHub's *secondary* rate limit (its
# short-lived abuse-detection throttle, distinct from the hours-long primary
# point-budget limit). These delays are deliberately small: the secondary
# limit is documented to clear within seconds, so there is no benefit to a
# long or exponential schedule, only to CI wall-clock time spent waiting.
# Transient 5xx server errors (issue #190) share this same schedule: the
# observed failure mode -- a freshly opened PR's diff momentarily uncomputed,
# clearing within ~12 seconds -- has the same seconds-scale recovery profile,
# so a separate schedule would add configuration without adding behavior.
SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS: tuple[int, ...] = (2, 5, 10)


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


def _is_transient_server_error(message: str) -> bool:
    """True when gh's error text reports a transient GitHub 5xx server error.

    Grounded in the live failure that motivated this classifier (issue #190,
    the changelog gate's first CI run): `gh: Server Error: Sorry, this diff is
    temporarily unavailable due to heavy server load. (HTTP 500)` -- a freshly
    opened pull request's diff was momentarily uncomputed, and a manual rerun
    passed 12 seconds later. gh surfaces 5xx responses either with the words
    "server error" or with the status code itself ("HTTP 500", "HTTP 502",
    ...), so both signals are checked. 4xx responses match neither substring
    and keep failing fast: a genuine caller error (bad path, missing scope,
    not found) must never be retried, only re-raised immediately.
    """

    lowered = message.lower()
    # "http 5" matches every 5xx status code gh prints ("(HTTP 500)",
    # "HTTP 502", ...) while matching no 4xx code, which keeps the
    # retry strictly scoped to server-side failures.
    return "server error" in lowered or "http 5" in lowered


def run_gh(args: list[str], cwd: Path | None = None) -> str:
    """Run one fixed GitHub CLI command and return its captured output.

    Retries a bounded number of times, with short fixed delays, when gh
    reports one of GitHub's transient conditions -- the secondary rate limit
    or a 5xx server error (issue #190) -- but never for the primary
    (hours-long) rate limit, which raises the distinct PrimaryRateLimitError
    immediately so callers can report it as infrastructure rather than a
    defect in the work being validated. Ordinary failures (4xx, auth, bad
    arguments) also fail fast: retrying a genuine caller error would only
    delay the inevitable and mask the real defect.

    Args:
        args: The `gh` subcommand and its arguments (without the leading "gh" itself).
        cwd: Optional working directory for the gh process. A caller that
            omits `--repo` and relies on gh inferring the repository from the
            surrounding Git checkout can pin that inference to a known
            directory here (`sync_github_labels.py` pins it to the repository
            root); None keeps the calling process's own working directory,
            exactly like plain subprocess.run.

    Returns:
        The command's captured stdout.
    """

    # The first attempt has no delay; each retry after a transient failure
    # (secondary rate limit or 5xx server error) waits progressively longer
    # per SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS.
    delays = (0, *SECONDARY_RATE_LIMIT_RETRY_DELAYS_SECONDS)
    # Walk the fixed attempt schedule rather than recursing, so the bound on
    # total attempts is visible directly from `delays` with no separate counter.
    for attempt_index, delay in enumerate(delays):
        # Only a retry (attempt_index > 0) has a delay; the first attempt runs
        # immediately.
        if delay:
            time.sleep(delay)
        # Collapse "gh not installed" (FileNotFoundError) and "gh exited
        # non-zero" (CalledProcessError, since check=True) into this module's
        # own exception hierarchy, so callers only need to catch GitHubApiError.
        try:
            # command is a fixed literal ("gh", *args) built from the calling
            # script's own subcommand arguments, not runtime/user-constructed input.
            result = subprocess.run(  # noqa: S603
                ["gh", *args],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
        except FileNotFoundError as error:
            raise GitHubApiError("gh CLI is not installed or not on PATH") from error
        except subprocess.CalledProcessError as error:
            message = error.stderr.strip() or error.stdout.strip()
            is_last_attempt = attempt_index == len(delays) - 1
            # A genuine metadata defect and an hours-long token-budget
            # exhaustion must never look the same in CI output, or a human
            # will waste time "fixing" a PR that was never the problem -- so
            # this case fails immediately, without spending any retry, and as
            # its own exception type so callers can exit distinctly.
            if _is_primary_rate_limit_error(message):
                raise PrimaryRateLimitError(
                    "GitHub API rate limit exhausted -- this is a transient "
                    f"infrastructure condition, not a metadata defect: gh {' '.join(args)} "
                    f"failed: {message}"
                ) from error
            # Both transient conditions -- the secondary rate limit and a 5xx
            # server error -- are documented/observed to clear within seconds;
            # loop around to the next (longer) delay instead of failing,
            # unless this was already the last scheduled attempt.
            if (
                _is_secondary_rate_limit_error(message) or _is_transient_server_error(message)
            ) and not is_last_attempt:
                continue
            raise GitHubApiError(f"gh {' '.join(args)} failed: {message}") from error
        else:
            return result.stdout
    # Unreachable: `delays` is a fixed non-empty literal, so every iteration of
    # the loop above either returns on success or raises on failure. Kept only
    # so a static checker can see every code path produces or raises a value.
    raise AssertionError("unreachable: run_gh's retry loop always returns or raises")


@dataclass(frozen=True, slots=True)
class GraphQLQuota:
    """One point-in-time reading of the authenticated token's GraphQL rate limit."""

    limit: int
    used: int
    remaining: int
    reset_epoch: int

    def reset_time_utc(self) -> str:
        """Render the reset instant as a human-readable UTC timestamp for log output."""

        return datetime.fromtimestamp(self.reset_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_graphql_quota() -> GraphQLQuota:
    """Read the GraphQL resource of GitHub's rate-limit endpoint via REST.

    `gh api rate_limit` is a REST call, and GitHub documents the rate_limit
    endpoint itself as not counting against any quota -- so checking and
    reporting quota is always free and can never worsen the condition it is
    measuring.

    Returns:
        The current GraphQL quota snapshot for the authenticated token.
    """

    payload = json.loads(run_gh(["api", "rate_limit"]))
    resource = payload["resources"]["graphql"]
    return GraphQLQuota(
        limit=resource["limit"],
        used=resource["used"],
        remaining=resource["remaining"],
        reset_epoch=resource["reset"],
    )


class QuotaMonitor:
    """GraphQL quota preflight and before/after/consumed accounting for one run.

    One monitor instance covers one script invocation: `preflight()` records
    the before snapshot and enforces the configured minimum threshold;
    `report()` takes the after snapshot and formats the consumption line the
    governance docs require every quota-consuming run to print.
    """

    def __init__(self, minimum_remaining: int = DEFAULT_MINIMUM_GRAPHQL_QUOTA) -> None:
        """Configure the safety threshold; a value of 0 or below disables the stop.

        Args:
            minimum_remaining: Minimum GraphQL points that must remain for
                preflight to allow the run to proceed. Non-positive values keep
                the before/after reporting but never block, so operators can
                observe consumption without enforcement.
        """

        # The threshold is stored as given (including non-positive "disabled"
        # values); preflight interprets it rather than normalizing it here so
        # the reported configuration always matches what the caller passed.
        self.minimum_remaining = minimum_remaining
        # Populated by preflight(); report() needs it to compute consumption,
        # so it stays None until a preflight has actually happened.
        self._before: GraphQLQuota | None = None

    @property
    def preflighted(self) -> bool:
        """Whether preflight() has recorded a baseline, so report() can succeed.

        Lets callers print the consumption report only on runs that actually
        entered their quota-consuming phase, without provoking (and then
        swallowing) the sequencing error report() raises otherwise.
        """

        return self._before is not None

    def preflight(self) -> GraphQLQuota:
        """Record the before snapshot and stop the run if quota is below threshold.

        Returns:
            The before snapshot, for callers that want to log it immediately.

        Raises:
            GraphQLQuotaInsufficientError: Remaining quota is below the
                configured (positive) minimum, so the caller must stop before
                starting any expensive or mutating GraphQL work.
        """

        self._before = fetch_graphql_quota()
        # A non-positive threshold means "observe but never block": the before
        # snapshot above is still recorded so report() works either way.
        if self.minimum_remaining > 0 and self._before.remaining < self.minimum_remaining:
            raise GraphQLQuotaInsufficientError(
                f"GraphQL quota preflight: only {self._before.remaining} of "
                f"{self._before.limit} points remain, below the configured minimum of "
                f"{self.minimum_remaining}; stopping before any mutation. The quota "
                f"resets at {self._before.reset_time_utc()}."
            )
        return self._before

    def report(self) -> str:
        """Take the after snapshot and format the before/after/consumed line.

        Returns:
            One human-readable line, e.g.
            "GraphQL quota: 4830 before, 4818 after, 12 consumed (limit 5000, resets ...)".
        """

        # report() without a preflight would have no baseline to subtract from;
        # that is a caller sequencing bug, surfaced explicitly rather than as a
        # confusing None arithmetic error.
        if self._before is None:
            raise GitHubApiError("QuotaMonitor.report() called before preflight()")
        after = fetch_graphql_quota()
        consumed = self._before.remaining - after.remaining
        # A window reset between the two snapshots makes remaining rise instead
        # of fall; the negative difference is real evidence of that, so it is
        # annotated rather than clamped away.
        reset_note = " (quota window reset during the run)" if consumed < 0 else ""
        return (
            f"GraphQL quota: {self._before.remaining} before, {after.remaining} after, "
            f"{consumed} consumed{reset_note} "
            f"(limit {after.limit}, resets at {after.reset_time_utc()})."
        )


@dataclass(frozen=True, slots=True)
class SingleSelectOption:
    """One Project V2 single-select field's id paired with one named option's id."""

    field_id: str
    option_id: str


@dataclass(frozen=True, slots=True)
class ProjectItemRef:
    """A Project V2 item id paired with its owning project's node id.

    Both come back from the same targeted lookup, so callers that mutate the
    item (which requires the project id) never need a separate `project view`.
    """

    item_id: str
    project_id: str


# Upper bound on how many project memberships one pull request's targeted
# lookup requests. A PR on this repository belongs to exactly one project
# (Project #5), so 50 is far beyond plausible reality -- and the query still
# checks hasNextPage and fails loudly (never silently returns "not tracked")
# if the bound were somehow exceeded.
_PULL_REQUEST_PROJECT_ITEMS_BOUND: int = 50

# Targeted lookup: resolve one pull request's Project V2 item memberships
# directly from the PR side, instead of listing every item on the board and
# scanning for the PR. Costs one GraphQL point regardless of board size,
# where a full `gh project item-list --limit 500` paginates the whole board.
_PULL_REQUEST_PROJECT_ITEMS_QUERY: str = """
query($owner: String!, $name: String!, $number: Int!, $first: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      projectItems(first: $first) {
        pageInfo { hasNextPage }
        nodes {
          id
          project {
            id
            number
            owner {
              ... on User { login }
              ... on Organization { login }
            }
          }
        }
      }
    }
  }
}
"""

# Targeted read-back: read exactly one field's current value on exactly one
# already-known Project item. This is the per-mutation verification read that
# replaces the full-board `item-list` scan (issue #173): the read-back-verified
# mutation rule from #164/#170 is unchanged, only its scope shrank. Only the
# single-select fragment is requested because every governed read-back target
# (Status and the other lifecycle fields) is single-select; a value of any
# other union type comes back as an empty object and is treated as an explicit
# error below, never mistaken for "unset".
_ITEM_SINGLE_SELECT_QUERY: str = """
query($item: ID!, $field: String!) {
  node(id: $item) {
    ... on ProjectV2Item {
      fieldValueByName(name: $field) {
        ... on ProjectV2ItemFieldSingleSelectValue { name }
      }
    }
  }
}
"""


class ProjectClient:
    """Operation-scoped Project V2 access with cached identity and one snapshot.

    One instance corresponds to one logical phase of work (one script run, one
    mutation-plus-verification pass). Schema lookups (`single_select_option`)
    and the optional full snapshot (`items`) are fetched once and cached for
    the instance's lifetime; the targeted read-back (`fetch_item_single_select`)
    is deliberately never cached -- see that method's docstring.
    """

    def __init__(self, owner: str, project_number: int) -> None:
        """Bind the client to one project identity for its whole lifetime.

        Args:
            owner: The project owner's login (a user account for Project #5).
            project_number: The project's user-visible number (5 here).
        """

        # Project identity is fixed per client so every cached lookup below is
        # unambiguous about which project it describes.
        self._owner = owner
        self._project_number = project_number
        # Cache of (field name, option name) -> resolved ids, filled from one
        # field-list read the first time any option is requested.
        self._single_select_cache: dict[tuple[str, str], SingleSelectOption] = {}
        # The raw field-list payload backing the cache above; None until the
        # first schema lookup so a client that never needs schema never pays
        # for the read.
        self._fields_payload: list[dict[str, Any]] | None = None
        # At most one full board snapshot per client (per logical phase); None
        # until a caller actually asks for it.
        self._items: list[dict[str, Any]] | None = None

    def single_select_option(self, field_name: str, option_name: str) -> SingleSelectOption:
        """Resolve a single-select field's id and one named option's id, cached.

        The underlying `gh project field-list` read happens at most once per
        client instance, no matter how many field/option pairs are resolved.

        Args:
            field_name: The field's display name, e.g. "Status".
            option_name: The option's display name, e.g. "Merged".

        Returns:
            The resolved field and option node ids.

        Raises:
            GitHubApiError: The field or the option does not exist on the project.
        """

        cache_key = (field_name, option_name)
        # Serve repeated lookups from the cache so schema resolution costs one
        # GraphQL-backed read per phase, not one per lookup.
        if cache_key in self._single_select_cache:
            return self._single_select_cache[cache_key]
        # First schema lookup on this client: fetch and retain the whole
        # field-list payload so later lookups for other fields are also free.
        if self._fields_payload is None:
            args = [
                "project",
                "field-list",
                str(self._project_number),
                "--owner",
                self._owner,
                "--format",
                "json",
            ]
            self._fields_payload = json.loads(run_gh(args))["fields"]
        field = next((f for f in self._fields_payload if f.get("name") == field_name), None)
        # A project without the requested field can't be mutated toward it;
        # name the field so the operator sees exactly what is missing.
        if field is None:
            raise GitHubApiError(f"Project #{self._project_number} has no {field_name!r} field")
        option = next((o for o in field.get("options", []) if o.get("name") == option_name), None)
        # The field exists but lacks the specific option, so there is no option
        # id to write -- again named explicitly for the operator.
        if option is None:
            raise GitHubApiError(
                f"Project #{self._project_number}'s {field_name!r} field has no "
                f"{option_name!r} option"
            )
        resolved = SingleSelectOption(field_id=field["id"], option_id=option["id"])
        # Populate the cache so this exact pair is never re-resolved.
        self._single_select_cache[cache_key] = resolved
        return resolved

    def items(self) -> list[dict[str, Any]]:
        """Fetch the full board snapshot at most once and reuse it thereafter.

        This is the one-snapshot-per-phase discovery read: board-wide
        validation legitimately needs every item, but must pay for the
        pagination exactly once per client instance, never inside a per-item
        loop. Requires a token with the `project` scope.

        Returns:
            Every item on the project, in gh's item-list JSON shape.
        """

        # Reuse the already-fetched snapshot for every later lookup in this
        # phase -- the whole point of the one-snapshot rule.
        if self._items is None:
            args = [
                "project",
                "item-list",
                str(self._project_number),
                "--owner",
                self._owner,
                "--format",
                "json",
                "--limit",
                "500",
            ]
            self._items = json.loads(run_gh(args))["items"]
        return self._items

    def fetch_pull_request_item(self, repo: str, pr_number: int) -> ProjectItemRef | None:
        """Find one pull request's item on this project via a bounded targeted lookup.

        Resolves from the pull-request side (`repository.pullRequest.projectItems`)
        instead of scanning a full board snapshot, so the cost is one GraphQL
        point regardless of how many items the board holds.

        Args:
            repo: The pull request's "OWNER/REPO" slug.
            pr_number: The pull request's number in that repository.

        Returns:
            The item and project node ids, or None when the pull request is
            not tracked on this client's project at all.

        Raises:
            GitHubApiError: The repo slug is malformed, or the pull request
                belongs to more project items than the bounded page requested
                (making "not tracked" unknowable from this page alone).
        """

        # The GraphQL repository() field takes owner and name separately, so
        # the conventional OWNER/REPO slug is split here.
        repo_owner, separator, repo_name = repo.partition("/")
        # A malformed slug fails loudly here rather than sending a broken query.
        if not separator or not repo_owner or not repo_name:
            raise GitHubApiError(f"malformed repository slug {repo!r}; expected OWNER/REPO")
        args = [
            "api",
            "graphql",
            "-f",
            f"query={_PULL_REQUEST_PROJECT_ITEMS_QUERY}",
            "-f",
            f"owner={repo_owner}",
            "-f",
            f"name={repo_name}",
            "-F",
            f"number={pr_number}",
            "-F",
            f"first={_PULL_REQUEST_PROJECT_ITEMS_BOUND}",
        ]
        payload = json.loads(run_gh(args))
        memberships = payload["data"]["repository"]["pullRequest"]["projectItems"]
        # A truncated page makes absence unprovable: the sought project could
        # be on the next page. Fail loudly instead of silently reporting the
        # PR as untracked -- the bound is a page size, not a coverage cap.
        if memberships["pageInfo"]["hasNextPage"]:
            raise GitHubApiError(
                f"pull request #{pr_number} belongs to more than "
                f"{_PULL_REQUEST_PROJECT_ITEMS_BOUND} project items; bounded targeted "
                "lookup is inconclusive"
            )
        # Scan the PR's own (small) membership list for this client's project,
        # matching both number and owner login since numbers are only unique
        # per owner.
        for node in memberships["nodes"]:
            project = node["project"]
            # casefold comparison tolerates login-case differences between the
            # configured owner and GitHub's canonical spelling.
            if (
                project["number"] == self._project_number
                and project["owner"].get("login", "").casefold() == self._owner.casefold()
            ):
                return ProjectItemRef(item_id=node["id"], project_id=project["id"])
        return None

    def fetch_item_single_select(self, item_id: str, field_name: str) -> str | None:
        """Read one item's current single-select value via a targeted node(id:) query.

        Deliberately performs a fresh query on every call and caches nothing:
        this is the mutation-verification read, and the entire value of the
        read-back rule (#164/#170) is that it observes the item's *current*
        state, including a built-in workflow having just overwritten it.

        Args:
            item_id: The Project V2 item's node id (PVTI_...).
            field_name: The single-select field to read, e.g. "Status".

        Returns:
            The option name currently set, or None when the field is unset --
            GitHub omits unset fields entirely, returning a JSON null for
            fieldValueByName rather than any empty-value object.

        Raises:
            GitHubApiError: The item no longer exists, the id is not a Project
                item, or the field's value is of a non-single-select union
                type (a schema surprise that must never be misread as unset).
        """

        args = [
            "api",
            "graphql",
            "-f",
            f"query={_ITEM_SINGLE_SELECT_QUERY}",
            "-f",
            f"item={item_id}",
            "-f",
            f"field={field_name}",
        ]
        payload = json.loads(run_gh(args))
        node = payload["data"]["node"]
        # Losing the item between mutation and verification makes the requested
        # state unknowable, so fail instead of treating a missing node as unset.
        if node is None:
            raise GitHubApiError(
                f"Project item {item_id} no longer exists during {field_name!r} read-back"
            )
        # A node of any other GraphQL type does not match the ProjectV2Item
        # inline fragment, leaving the fieldValueByName key absent entirely --
        # which means the id was never a Project item to begin with.
        if "fieldValueByName" not in node:
            raise GitHubApiError(f"node {item_id} is not a Project V2 item")
        value = node["fieldValueByName"]
        # GitHub returns JSON null for an unset field (the field has no value
        # at all on this item); that is the one legitimate None case.
        if value is None:
            return None
        # A set value of a non-single-select type matches none of the query's
        # fragments and arrives as an empty object. Project field values span
        # multiple GraphQL union types (issue #173's Risks section), so this is
        # surfaced as an explicit error rather than misread as unset.
        if "name" not in value:
            raise GitHubApiError(
                f"Project item {item_id}'s {field_name!r} value is not a single-select "
                "option; refusing to interpret it"
            )
        return value["name"]
