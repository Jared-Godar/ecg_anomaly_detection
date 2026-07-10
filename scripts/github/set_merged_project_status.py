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
from collections.abc import Sequence
from dataclasses import dataclass


class ProjectStatusSyncError(RuntimeError):
    """Raised when the merged-status sync cannot complete."""


@dataclass(frozen=True, slots=True)
class StatusField:
    """The Status field's id and the option id for its 'Merged' value."""

    field_id: str
    merged_option_id: str


def _run_gh(args: list[str]) -> str:
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
        raise ProjectStatusSyncError(
            f"gh {' '.join(args)} failed: {error.stderr.strip() or error.stdout.strip()}"
        ) from error
    return result.stdout


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
    if field is None:
        raise ProjectStatusSyncError(f"Project #{project_number} has no 'Status' field")
    option = next((o for o in field.get("options", []) if o.get("name") == "Merged"), None)
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
    for item in payload["items"]:
        content = item.get("content", {})
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument(
        "--repo", required=True, help="GitHub OWNER/REPO of the merged pull request"
    )
    parser.add_argument("--owner", default="Jared-Godar", help="Project owner login")
    parser.add_argument("--project-number", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        item_id = find_pull_request_item_id(
            args.owner, args.project_number, args.repo, args.pr_number
        )
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


if __name__ == "__main__":
    raise SystemExit(main())
