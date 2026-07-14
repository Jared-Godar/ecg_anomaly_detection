#!/usr/bin/env python3
"""Add a Dependabot pull request to Project #5 and populate its required fields.

Dependabot PRs arrive with no Project #5 membership and none of the nine
single-select fields the metadata-governance gate requires, so every routine
dependency bump would fail strict board validation until a human stamped the
board by hand. This script performs that stamping automatically from the
`dependabot-autofill` workflow, with documented bot-PR defaults:

    Status=Review, Workstream=Stewardship, Issue Type=Technical Debt,
    Priority=Low, Risk=Low, Size=XS, Repository Area=ci-cd,
    Portfolio Signal=Operational Maturity, Target Release=Stewardship.

Rationale for the defaults: a Dependabot PR is routine, low-risk, ungrouped-
scope dependency maintenance. It is awaiting merge from the moment it opens
(Status=Review), it is stewardship rather than roadmap work (Workstream and
Target Release=Stewardship), keeping dependencies current is technical-debt
servicing (Issue Type), the blast radius of a lockfile bump is small
(Priority=Low, Risk=Low, Size=XS), the automation surface it exercises is
CI/CD (Repository Area=ci-cd), and the signal it sends a portfolio reviewer
is operational maturity (Portfolio Signal).

Those defaults are only defensible for routine dependency PRs, so the script
verifies server-side (via REST, before any mutation) that the PR's author is
`dependabot[bot]` with account type `Bot`, and exits 1 otherwise -- it must
be unusable as a general-purpose board-stamper for human PRs. Curated values
always win: a field that already holds any value is preserved untouched
(house rule from AGENTS.md), so a maintainer's manual triage is never
overwritten by the bot defaults.

Quota stewardship (issue #173): one GraphQL quota preflight before any work,
targeted pull-request -> project-item lookup (never a full board snapshot),
a fresh targeted read-back after every actual mutation, and a
before/after/consumed report printed on success and failure alike.

Exit codes follow the house convention: 0 success, 1 policy/validation
failure (non-Dependabot author, or a mutation whose read-back never
confirmed), 2 required data unreadable (gh/API failures, an item-add that
the verifying re-lookup could not find), 3 a GraphQL quota condition
(transient shared-pool infrastructure, never a defect in the PR).
"""

from __future__ import annotations

import argparse
import json
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

# This script's own preflight threshold: a full run costs roughly 15-25
# GraphQL points (one targeted PR-item lookup, an optional item-add with its
# verifying re-lookup, one field-list schema read, and up to nine
# edit-plus-read-back pairs at ~1 point per read), so 50 gives a 2x-3x margin.
# Deliberately far below the shared github_api.DEFAULT_MINIMUM_GRAPHQL_QUOTA
# (sized for a ~203-point board snapshot this script never takes): requiring
# that much headroom would block a cheap autofill that a moderately drained
# pool could easily serve.
_MIN_GRAPHQL_QUOTA_DEFAULT: int = 50

# The exact GitHub login of Dependabot's PR-authoring bot. Checked together
# with the account type below: login alone could be shadowed by a user
# account named to resemble the bot's display form.
_DEPENDABOT_LOGIN: str = "dependabot[bot]"
# The required account type for that login. GitHub assigns `Bot` server-side
# and it cannot be self-selected, so it is the impersonation-proof half of
# the author guard.
_DEPENDABOT_ACCOUNT_TYPE: str = "Bot"

# The nine required Project #5 single-select fields and their documented
# bot-PR default options, in the fixed order they are synced (deterministic
# ordering keeps logs and tests stable). The module docstring records the
# governance rationale behind each default.
_BOT_PR_FIELD_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("Status", "Review"),
    ("Workstream", "Stewardship"),
    ("Issue Type", "Technical Debt"),
    ("Priority", "Low"),
    ("Risk", "Low"),
    ("Size", "XS"),
    ("Repository Area", "ci-cd"),
    ("Portfolio Signal", "Operational Maturity"),
    ("Target Release", "Stewardship"),
)


class DependabotSyncPolicyError(github_api.GitHubApiError):
    """Raised for policy/validation failures that must map to exit code 1.

    Covers the two conditions the house exit-code convention classifies as
    policy rather than infrastructure: a PR whose author is not Dependabot
    (the bot defaults are indefensible for human PRs), and a field mutation
    whose targeted read-back never confirmed the requested value within the
    bounded retry. Subclasses the shared GitHubApiError so main()'s handlers
    stay a simple most-specific-first chain.
    """


def resolve_repository(repo: str | None) -> str:
    """Return the OWNER/REPO slug, asking gh to resolve it when not supplied.

    The targeted project-item lookup, the REST author guard, and the
    item-add URL all need a concrete slug, but local operators inside the
    checkout should not have to spell it out -- gh already knows it from the
    surrounding Git checkout.

    Args:
        repo: The caller-supplied "OWNER/REPO" slug, or None to resolve it.

    Returns:
        The concrete "OWNER/REPO" slug.
    """

    # An explicit slug (the workflow always passes one) skips the extra call.
    if repo is not None:
        return repo
    # gh resolves the repository from the current checkout's Git remotes; the
    # JSON format keeps parsing exact rather than scraping human output.
    payload = json.loads(github_api.run_gh(["repo", "view", "--json", "nameWithOwner"]))
    return payload["nameWithOwner"]


def require_dependabot_author(repo: str, pr_number: int) -> None:
    """Verify server-side that the PR was authored by Dependabot, or refuse to run.

    Fetches the pull request over REST (free with respect to the GraphQL
    pool) and requires the author login to be `dependabot[bot]` with account
    type `Bot`. Workflow-side `if:` conditions already gate on the event
    payload, but this guard is what makes the *script* unusable as a
    general-purpose board-stamper for human PRs regardless of how it is
    invoked (governance: the bot defaults are only defensible for routine
    dependency PRs).

    Args:
        repo: The pull request's "OWNER/REPO" slug.
        pr_number: The pull request's number in that repository.

    Raises:
        DependabotSyncPolicyError: The author is not Dependabot; the caller
            must exit 1 without having mutated anything.
    """

    # REST, not GraphQL: the guard must not draw from the shared GraphQL pool
    # this script's preflight budgets for.
    payload = json.loads(github_api.run_gh(["api", f"repos/{repo}/pulls/{pr_number}"]))
    # A deleted author account leaves `user` null; that is not Dependabot.
    author = payload.get("user") or {}
    login = author.get("login")
    account_type = author.get("type")
    # Both checks are required: the `Bot` account type cannot be
    # self-assigned, so it defends against a user account registered with a
    # Dependabot-lookalike login.
    if login != _DEPENDABOT_LOGIN or account_type != _DEPENDABOT_ACCOUNT_TYPE:
        raise DependabotSyncPolicyError(
            f"pull request #{pr_number} was authored by {login!r} (type {account_type!r}), "
            f"not {_DEPENDABOT_LOGIN!r} (type {_DEPENDABOT_ACCOUNT_TYPE!r}); refusing to "
            "apply bot-PR board defaults to a non-Dependabot pull request"
        )


def ensure_project_item(
    client: github_api.ProjectClient,
    repo: str,
    pr_number: int,
    owner: str,
    project_number: int,
) -> github_api.ProjectItemRef:
    """Return the PR's project item, adding it to the board first when absent.

    The add is never assumed to have worked: it is verified by a fresh
    targeted re-lookup (the read-back-verified mutation rule from #164/#170),
    and a re-lookup that still cannot find the item is a hard failure --
    required data is unreadable, so the field sync cannot proceed.

    Args:
        client: The Project access layer bound to the target project.
        repo: The pull request's "OWNER/REPO" slug.
        pr_number: The pull request's number in that repository.
        owner: The project owner's login, for the item-add invocation.
        project_number: The project's user-visible number, for item-add.

    Returns:
        The item and project node ids for the (possibly just-added) item.

    Raises:
        github_api.GitHubApiError: The item-add reported success but the
            verifying re-lookup still cannot find the item (exit code 2).
    """

    # Targeted lookup first: most re-runs (synchronize events) find the item
    # already on the board, costing one GraphQL point and no mutation.
    item = client.fetch_pull_request_item(repo, pr_number)
    # Absent from the board: add it, then verify the add by re-looking it up
    # rather than trusting item-add's exit status or output.
    if item is None:
        github_api.run_gh(
            [
                "project",
                "item-add",
                str(project_number),
                "--owner",
                owner,
                "--url",
                f"https://github.com/{repo}/pull/{pr_number}",
                "--format",
                "json",
            ]
        )
        item = client.fetch_pull_request_item(repo, pr_number)
        # The re-lookup is the only accepted evidence the add worked; without
        # it there is no item id to mutate, so the run cannot continue.
        if item is None:
            raise github_api.GitHubApiError(
                f"added pull request #{pr_number} to Project #{project_number} but the "
                "verifying re-lookup still cannot find its item; refusing to proceed"
            )
    return item


def set_field_default(
    client: github_api.ProjectClient,
    item: github_api.ProjectItemRef,
    field_name: str,
    option_name: str,
    project_number: int,
) -> bool:
    """Set one field to its bot-PR default unless a curated value already exists.

    Reads the field's current value first: any existing value is preserved
    untouched (curated values win -- house rule from AGENTS.md), because a
    maintainer's manual triage must never be overwritten by bot defaults.
    A blank field is mutated and then verified by a fresh targeted read-back,
    with one bounded retry; the mutation's exit status is never treated as
    proof of success (#164/#170).

    Args:
        client: The Project access layer bound to the target project.
        item: The project item to mutate, with its owning project's node id.
        field_name: The single-select field's display name, e.g. "Priority".
        option_name: The bot-PR default option to set, e.g. "Low".
        project_number: The project's user-visible number, for log messages.

    Returns:
        True when the field was actually mutated (and verified); False when
        an existing curated value was preserved.

    Raises:
        DependabotSyncPolicyError: The read-back never confirmed the
            requested value within the bounded retry (exit code 1).
    """

    # Fresh targeted read of the current value; this doubles as the
    # idempotency pre-check, so a rerun over an already-stamped PR costs one
    # read per field and zero mutations.
    current = client.fetch_item_single_select(item.item_id, field_name)
    # Non-blank means curated (a human, a built-in workflow, or a previous
    # run already chose a value): preserve it and say so in the log.
    if current is not None:
        print(f"Project #{project_number} field {field_name!r} preserved (already {current!r}).")
        return False
    # Only a blank field earns the default; resolve the ids now (the
    # field-list read behind this is cached on the client, so nine fields
    # cost one schema read total).
    field = client.single_select_option(field_name, option_name)
    mutation_args = [
        "project",
        "item-edit",
        "--id",
        item.item_id,
        "--project-id",
        item.project_id,
        "--field-id",
        field.field_id,
        "--single-select-option-id",
        field.option_id,
    ]
    # Two attempts bound recovery from gh's misleading no-change response and
    # from a stale first read-back, while ensuring every attempted write
    # receives its own authoritative verification read.
    for attempt in range(2):
        # Only the observed false no-change response is eligible for
        # read-back recovery; unrelated CLI failures (auth, schema, the
        # primary rate limit) retain their fail-fast propagation.
        try:
            github_api.run_gh(mutation_args)
        except github_api.GitHubApiError as error:
            # A different error carries no evidence the mutation was
            # accepted, so preserve it rather than masking the real fault.
            if "no changes to make" not in str(error).lower():
                raise

        # Fresh targeted read of exactly this item's field -- never cached,
        # never a board scan; only this read, not gh's exit status, proves
        # the mutation landed.
        observed = client.fetch_item_single_select(item.item_id, field_name)
        # The requested option coming back verbatim is the success condition.
        if observed == option_name:
            print(
                f"Project #{project_number} field {field_name!r} set to {option_name!r} "
                "(verified by read-back)."
            )
            return True
        # Any other read-back (absent or a different value) earns one retry;
        # the second failure reports the exact observed value so operators
        # see what actually won.
        if attempt == 1:
            observed_text = observed if observed is not None else "unset"
            raise DependabotSyncPolicyError(
                f"Project item {item.item_id} field {field_name!r} read back as "
                f"{observed_text!r} after 2 attempts; expected {option_name!r}"
            )
    # Unreachable: the loop above always returns on success or raises on the
    # final attempt. Kept only so a static checker sees every path resolve.
    raise AssertionError("unreachable: set_field_default's retry loop returns or raises")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Dependabot board-autofill entry point.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: PR number, optional repo slug, project owner,
        project number, and the minimum-remaining GraphQL quota threshold.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "GitHub OWNER/REPO of the Dependabot pull request; omitted, gh resolves it "
            "from the surrounding Git checkout"
        ),
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

    Keeping orchestration here makes terminal behavior and error translation
    straightforward to audit: preflight, author guard, board membership,
    then the nine field syncs, with quota accounting reported either way.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        The house-convention exit code: 0 success, 1 policy/validation
        failure, 2 required data unreadable, 3 a GraphQL quota condition.
    """

    args = parse_args(argv)

    # One monitor per run: preflight before any GraphQL work, report after --
    # both reads are REST and free, so accounting can never worsen the quota.
    monitor = github_api.QuotaMonitor(minimum_remaining=args.min_graphql_quota)
    # One client per run = one logical phase: schema lookups are cached on it,
    # and no code path in this script ever requests a full board snapshot.
    client = github_api.ProjectClient(args.owner, args.project_number)

    # Counters for the closing summary line, so a log reader sees at a glance
    # how many fields the run actually touched versus left curated.
    fields_set = 0
    fields_preserved = 0

    # Every failure mode is collapsed into the GitHubApiError hierarchy by
    # the access layer; the handlers below map it onto the house exit-code
    # convention, most specific first, so shared-pool exhaustion, a policy
    # refusal, and unreadable data can never be conflated in CI output.
    try:
        # Stop before any mutation when the shared pool is already too low --
        # a half-stamped board on a drained pool is strictly worse than a
        # clean, resumable early exit.
        monitor.preflight()
        repo = resolve_repository(args.repo)
        # The author guard runs before the board is even looked up, so a
        # non-Dependabot PR exits 1 having issued zero mutations (including
        # the item-add, which is itself a mutation).
        require_dependabot_author(repo, args.pr_number)
        item = ensure_project_item(client, repo, args.pr_number, args.owner, args.project_number)
        # Sync the nine fields in their fixed documented order; each is
        # independently preserved-or-set with its own verification read.
        for field_name, option_name in _BOT_PR_FIELD_DEFAULTS:
            # Tally each outcome for the closing summary line.
            if set_field_default(client, item, field_name, option_name, args.project_number):
                fields_set += 1
            else:
                fields_preserved += 1
    except (
        github_api.GraphQLQuotaInsufficientError,
        github_api.PrimaryRateLimitError,
    ) as error:
        # Quota conditions get their own exit code (3) and wording: transient
        # shared-pool infrastructure, resumable by rerunning after the reset,
        # never a defect in the pull request being stamped.
        print(f"quota: {error}", file=sys.stderr)
        return 3
    except DependabotSyncPolicyError as error:
        # Policy/validation failures (non-Dependabot author, unconfirmed
        # mutation) are exit 1 per the house convention.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except github_api.GitHubApiError as error:
        # Everything else in the hierarchy is required data being unreadable
        # (gh/API failures, an unverifiable item-add): exit 2.
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
            # Best-effort only: see the comment above for why a report
            # failure is downgraded to a warning instead of changing the
            # exit code.
            try:
                print(monitor.report(), file=sys.stderr)
            except github_api.GitHubApiError as report_error:
                print(
                    f"warning: could not report quota consumption: {report_error}",
                    file=sys.stderr,
                )

    print(
        f"Synced Project #{args.project_number} metadata for Dependabot pull request "
        f"#{args.pr_number}: {fields_set} field(s) set, {fields_preserved} preserved."
    )
    return 0


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
