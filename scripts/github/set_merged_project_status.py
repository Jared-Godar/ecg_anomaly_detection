#!/usr/bin/env python3
"""Force a merged pull request's Project #5 item to Status = Merged.

Addresses the status-drift race documented in
docs/governance/github-project.md#automation: merging a pull request also
closes it, so GitHub Project's built-in `Pull request merged` and `Item
closed` workflows both fire on the same event, and `Closed` has consistently
won over `Merged`. The public Project V2 API exposes no supported way to
inspect or reorder built-in workflow execution, so this script wins the race
explicitly, after the fact, via a direct field mutation.

Run only from the `project-status-sync` workflow, after
`github.event.pull_request.merged == true` on a `pull_request: closed`
event. Idempotent: setting an item's Status to Merged when it is already
Merged is a harmless no-op mutation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass


class ProjectStatusSyncError(RuntimeError):
    """Raised when the merged-status sync cannot complete."""


@dataclass(frozen=True, slots=True)
class StatusField:
    """The Status field's id and the option id for its 'Merged' value."""

    field_id: str
    merged_option_id: str


# Short, fixed backoff schedule for GitHub's *secondary* rate limit (its
# short-lived abuse-detection throttle, distinct from the hours-long primary
# point-budget limit below). These delays are deliberately small: the secondary
# limit is documented to clear within seconds, so there is no benefit to a long
# or exponential schedule, only to workflow wall-clock time spent waiting.
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
        # ProjectStatusSyncError, so callers only need to catch this module's
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
            raise ProjectStatusSyncError("gh CLI is not installed or not on PATH") from error
        except subprocess.CalledProcessError as error:
            message = error.stderr.strip() or error.stdout.strip()
            is_last_attempt = attempt_index == len(delays) - 1
            # A genuine metadata defect and an hours-long token-budget
            # exhaustion must never look the same in CI output, or a human
            # will waste time "fixing" a PR that was never the problem -- so
            # this case fails immediately, without spending any retry.
            if _is_primary_rate_limit_error(message):
                raise ProjectStatusSyncError(
                    "GitHub API rate limit exhausted -- this is a transient "
                    f"infrastructure condition, not a metadata defect: gh {' '.join(args)} "
                    f"failed: {message}"
                ) from error
            # Transient and documented to clear within seconds; loop around to
            # the next (longer) delay instead of failing, unless this was
            # already the last scheduled attempt.
            if _is_secondary_rate_limit_error(message) and not is_last_attempt:
                continue
            raise ProjectStatusSyncError(f"gh {' '.join(args)} failed: {message}") from error
        else:
            return result.stdout
    # Unreachable: `delays` is a fixed non-empty literal, so every iteration of
    # the loop above either returns on success or raises on failure. Kept only
    # so a static checker can see every code path produces or raises a value.
    raise AssertionError("unreachable: _run_gh's retry loop always returns or raises")


def fetch_project_id(owner: str, project_number: int) -> str:
    """Fetch a user-owned Project V2's node id."""
    args = ["project", "view", str(project_number), "--owner", owner, "--format", "json"]
    payload = json.loads(_run_gh(args))
    return payload["id"]


def fetch_status_field(owner: str, project_number: int) -> StatusField:
    """Fetch the Status field's id and its 'Merged' option id."""
    args = ["project", "field-list", str(project_number), "--owner", owner, "--format", "json"]
    payload = json.loads(_run_gh(args))
    field = next((f for f in payload["fields"] if f.get("name") == "Status"), None)
    # A project without a Status field at all can't be the field-id mutation target
    # this script is designed to write to.
    if field is None:
        raise ProjectStatusSyncError(f"Project #{project_number} has no 'Status' field")
    option = next((o for o in field.get("options", []) if o.get("name") == "Merged"), None)
    # Without a "Merged" option's node id, set_status_merged has no value to write --
    # the Status field exists but doesn't have the specific option this script needs.
    if option is None:
        raise ProjectStatusSyncError(
            f"Project #{project_number}'s Status field has no 'Merged' option"
        )
    return StatusField(field_id=field["id"], merged_option_id=option["id"])


def find_pull_request_item_id(
    owner: str, project_number: int, repo: str, pr_number: int
) -> str | None:
    """Return the Project item id for a pull request, or None if it isn't tracked."""
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
    # Scan every project item for the one whose content matches this exact pull
    # request (type + number + repo, since a project can track items across
    # multiple repositories with overlapping PR numbers).
    for item in payload["items"]:
        content = item.get("content", {})
        # All three fields must match: type distinguishes PRs from issues, number and
        # repository together uniquely identify this exact pull request.
        if (
            content.get("type") == "PullRequest"
            and content.get("number") == pr_number
            and content.get("repository") == repo
        ):
            return item["id"]
    return None


def set_status_merged(item_id: str, project_id: str, field: StatusField) -> None:
    """Set one Project item's Status field to Merged."""
    _run_gh(
        [
            "project",
            "item-edit",
            "--id",
            item_id,
            "--project-id",
            project_id,
            "--field-id",
            field.field_id,
            "--single-select-option-id",
            field.merged_option_id,
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the merged-status sync entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: PR number, repo, project owner, and project number.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument(
        "--repo", required=True, help="GitHub OWNER/REPO of the merged pull request"
    )
    parser.add_argument("--owner", default="Jared-Godar", help="Project owner login")
    parser.add_argument("--project-number", type=int, default=5)
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

    # Every gh CLI failure mode this script can hit is collapsed into
    # ProjectStatusSyncError by _run_gh; catch it once here for uniform error reporting.
    try:
        item_id = find_pull_request_item_id(
            args.owner, args.project_number, args.repo, args.pr_number
        )
        # A PR that was never added to Project #5 has nothing to sync; this is a
        # legitimate, non-fatal state (warn and exit 0), not a script failure, since
        # not every merged PR is necessarily tracked on this project board.
        if item_id is None:
            print(
                f"warning: pull request #{args.pr_number} is not a Project "
                f"#{args.project_number} item; nothing to sync",
                file=sys.stderr,
            )
            return 0
        project_id = fetch_project_id(args.owner, args.project_number)
        field = fetch_status_field(args.owner, args.project_number)
        set_status_merged(item_id, project_id, field)
    except ProjectStatusSyncError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"Set Project #{args.project_number} status to Merged for pull request #{args.pr_number}."
    )
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
