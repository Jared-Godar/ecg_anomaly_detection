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
event. Idempotent: every mutation is verified by a fresh targeted read of the
mutated item (a GraphQL `node(id:)` query for exactly the Status field --
never a full board scan; see issue #173 and
docs/governance/github-project.md). GitHub CLI's ``no changes to make`` error
is inconclusive until that read-back confirms the requested value; a
read-back that is not yet ``Merged`` -- whether absent or a concretely
different value -- earns one bounded retry.

Quota stewardship (issue #173): the run preflights the shared GraphQL point
budget before touching it, stops with exit code 3 when the remaining quota is
below a configurable threshold (or when GitHub reports the primary limit
already exhausted mid-run), and prints a before/after/consumed report so
every consumer of the shared 5000-points/hour pool is accountable in its own
logs.

Exit codes: 0 success (including "PR not tracked on the board"), 2 a genuine
failure (metadata defect, authentication, schema), 3 a GraphQL quota
condition -- transient shared-pool infrastructure, never a defect in the
pull request being synced.
"""

from __future__ import annotations

import argparse
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
# Guard against duplicate insertion when both governance scripts are loaded
# into one process (e.g. the test suite imports each of them).
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import github_api  # noqa: E402  (needs the sys.path insertion above)

# This script's own preflight threshold: a full run costs ~5 GraphQL points
# (targeted lookup, field-list, mutation, read-backs -- no board snapshots),
# so 25 is a 5x margin. Deliberately far below the shared
# github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA (sized for a ~203-point board
# snapshot this script never takes): requiring that much headroom here would
# block a cheap merge sync that a moderately drained pool could easily serve.
_MIN_GRAPHQL_QUOTA_DEFAULT: int = 25


class ProjectStatusSyncError(github_api.GitHubApiError):
    """Raised when the merged-status sync cannot complete.

    Subclasses the shared GitHubApiError so main()'s single except clause
    catches script-level defects and access-layer failures uniformly.
    """


def set_status_merged(
    client: github_api.ProjectClient,
    item_id: str,
    project_id: str,
    field: github_api.SingleSelectOption,
) -> None:
    """Set one Project item's Status to Merged and verify it, with one bounded retry.

    Starts with an idempotent pre-check: a Status that already reads Merged
    skips the mutation entirely (and, since the pre-check is a fresh targeted
    read, the observed value is current, not cached). Otherwise, verification
    is a fresh targeted read of exactly this item's Status field
    after every attempted write -- the mutation's exit status is never treated
    as proof of success (the read-back-verified mutation rule from #164/#170;
    issue #173 narrowed the read's scope from a full board scan to one item,
    not the requirement).

    Args:
        client: The Project access layer bound to the target project.
        item_id: The Project item to mutate (PVTI_...).
        project_id: The owning project's node id (PVT_..., required by item-edit).
        field: The Status field's id and the "Merged" option's id.
    """

    # Idempotent skip (issue #173): when a rerun -- or a built-in workflow that
    # happened to produce the desired value -- finds Status already Merged,
    # one targeted read is the run's entire cost and no mutation is attempted.
    if client.fetch_item_single_select(item_id, "Status") == "Merged":
        return

    mutation_args = [
        "project",
        "item-edit",
        "--id",
        item_id,
        "--project-id",
        project_id,
        "--field-id",
        field.field_id,
        "--single-select-option-id",
        field.option_id,
    ]
    # Two attempts bound recovery from gh's misleading no-change response while
    # ensuring every attempted write receives its own authoritative read-back.
    for attempt in range(2):
        # Only the observed false no-change response is eligible for read-back
        # recovery; unrelated CLI failures retain their existing fail-fast path.
        try:
            github_api.run_gh(mutation_args)
        except github_api.GitHubApiError as error:
            # A different error has no evidence that the mutation was accepted,
            # so preserve it rather than masking authentication or schema faults.
            if "no changes to make" not in str(error).lower():
                raise

        # Fresh targeted read of exactly this item's Status -- never cached,
        # never a full board scan (github_api.fetch_item_single_select's own
        # docstring records both guarantees).
        observed_status = client.fetch_item_single_select(item_id, "Status")
        # The fresh item value, not gh's exit status or message, proves success.
        if observed_status == "Merged":
            return
        # Any non-Merged read-back (absent or a different value) earns one retry;
        # the second fails with the exact observed value so operators never
        # receive a false success message.
        if attempt == 1:
            observed = observed_status if observed_status is not None else "unset"
            raise ProjectStatusSyncError(
                f"Project item {item_id} Status read back as "
                f"{observed!r} after 2 attempts; expected 'Merged'"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the merged-status sync entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: PR number, repo, project owner, project number, and
        the minimum-remaining GraphQL quota threshold.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument(
        "--repo", required=True, help="GitHub OWNER/REPO of the merged pull request"
    )
    parser.add_argument("--owner", default="Jared-Godar", help="Project owner login")
    parser.add_argument("--project-number", type=int, default=5)
    parser.add_argument(
        "--min-graphql-quota",
        type=int,
        default=_MIN_GRAPHQL_QUOTA_DEFAULT,
        help=(
            "Minimum remaining GraphQL points required by the preflight check before "
            "any mutation is attempted; 0 or below disables the stop (the "
            "before/after report is still printed). Default: "
            f"{_MIN_GRAPHQL_QUOTA_DEFAULT}."
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

    # One monitor per run: preflight before any GraphQL work, report after --
    # both reads are REST and free, so accounting can never worsen the quota.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)
    # One client per run = one logical phase: schema lookups are cached on it,
    # and no code path in this script ever requests a full board snapshot.
    client = github_api.ProjectClient(args.owner, args.project_number)

    # Every failure mode is collapsed into the GitHubApiError hierarchy by the
    # access layer; catch it once here for uniform error reporting, with the
    # two quota conditions mapped to their own exit code so CI output can
    # never conflate shared-pool exhaustion with a defect in this PR.
    try:
        # Stop before any mutation when the shared pool is already too low --
        # a half-completed mutation phase on a drained pool is strictly worse
        # than a clean, resumable early exit.
        monitor.preflight()
        item = client.fetch_pull_request_item(args.repo, args.pr_number)
        # A PR that was never added to Project #5 has nothing to sync; this is a
        # legitimate, non-fatal state (warn and exit 0), not a script failure, since
        # not every merged PR is necessarily tracked on this project board.
        if item is None:
            print(
                f"warning: pull request #{args.pr_number} is not a Project "
                f"#{args.project_number} item; nothing to sync",
                file=sys.stderr,
            )
            return 0
        field = client.single_select_option("Status", "Merged")
        set_status_merged(client, item.item_id, item.project_id, field)
    except (
        github_api.GraphQLQuotaInsufficientError,
        github_api.PrimaryRateLimitError,
    ) as error:
        # Quota conditions get their own exit code (3) and wording: transient
        # shared-pool infrastructure, resumable by rerunning after the reset,
        # never a metadata defect in the pull request being synced.
        print(f"quota: {error}", file=sys.stderr)
        return 3
    except github_api.GitHubApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        # The consumption report prints on success and failure alike -- a
        # failed run's consumption is exactly the evidence needed when
        # diagnosing a drained pool. A run that failed before preflight ever
        # recorded a baseline has nothing to report; otherwise reporting is
        # best-effort, because a report failure must never mask the run's
        # real outcome.
        if monitor.preflighted:
            # Best-effort only: see the comment above for why a report failure
            # is downgraded to a warning instead of changing the exit code.
            try:
                print(monitor.report(), file=sys.stderr)
            except github_api.GitHubApiError as report_error:
                print(
                    f"warning: could not report quota consumption: {report_error}",
                    file=sys.stderr,
                )

    print(
        f"Set Project #{args.project_number} status to Merged for pull request #{args.pr_number}."
    )
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
