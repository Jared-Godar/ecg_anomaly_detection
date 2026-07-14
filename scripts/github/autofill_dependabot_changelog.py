#!/usr/bin/env python3
"""Write a Dependabot pull request's CHANGELOG entry onto its own head branch (issue #193).

Runs from the `pull_request_target` Dependabot-autofill workflow, in the BASE
repository's context, holding the write-capable classic PAT
(`PROJECT_METADATA_TOKEN`). It renders one deterministic `### Dependencies`
bullet per updated dependency and PUTs the updated `CHANGELOG.md` back to the
pull request's head branch over REST, so the per-PR changelog gate (#184) can
pass on dependency bumps without a human writing the entry by hand.

Security model (every element is load-bearing; see the workflow's own
SECURITY INVARIANT block):

- No PR-head code is ever checked out or executed. The head's `CHANGELOG.md`
  is read as inert bytes over the REST contents API, and the entry content
  comes exclusively from dependabot/fetch-metadata's structured outputs --
  validated here against anchored allowlist regexes and a total size cap
  before any byte of it is written anywhere.
- The event payload is never trusted alone: the pull request is re-fetched
  live and must still be an open PR authored by the server-attested
  `dependabot[bot]` account (login AND `user.type == "Bot"`), from a same-repo
  `dependabot/`-prefixed head branch. Any violation fails closed (exit 1).
- Before writing, EVERY commit on the PR must be authored by `dependabot[bot]`
  and carry GitHub's server-side `verification.verified == true` signature
  (verified live against PR #192). fetch-metadata parses commit-message YAML,
  which a human with push access could forge -- but they cannot forge GitHub's
  dependabot[bot] signature, so a single human/unverified commit means a human
  owns the changelog and the gate stays red for human attention.
- The idempotency check runs BEFORE the authorship proof, deliberately: the
  PAT-authored changelog commit re-triggers this workflow via `synchronize`,
  and on that self-triggered re-run the branch legitimately contains our own
  non-Dependabot commit. Finding the desired entry already present terminates
  the loop (exit 0, no write) before authorship would reject our own commit.

Exit codes: 0 success or benign no-op (already written, superseded run, or a
conflicting writer whose own push re-triggers this workflow), 1 fail-closed
policy violation (untrusted author, invalid metadata, unverified commit),
2 required data unreadable (missing/oversized/unstructured changelog, missing
or unparseable UPDATED_DEPENDENCIES_JSON, API read failure), 3 a rate-limit
quota condition -- transient shared-pool infrastructure, never a defect in
the pull request being processed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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

# The environment variable carrying dependabot/fetch-metadata's
# updated-dependencies-json output. It arrives via the environment rather than
# argv because it is a large multi-line JSON blob set by the workflow's env
# indirection (no ${{ }} inside run: bodies).
UPDATED_DEPENDENCIES_ENV = "UPDATED_DEPENDENCIES_JSON"

# GitHub's server-attested login for the Dependabot app's bot account; the
# only author this script will ever write a changelog entry on behalf of.
DEPENDABOT_LOGIN = "dependabot[bot]"

# Head branches Dependabot creates always live under this prefix; anything
# else claiming to be a Dependabot PR fails the live re-validation.
DEPENDABOT_BRANCH_PREFIX = "dependabot/"

# Allowlist for dependency names. pre-commit ecosystem dependency names are
# repository URLs (e.g. https://github.com/astral-sh/ruff-pre-commit), so the
# colon and slashes are required members of the class; shell metacharacters,
# whitespace, backticks, and newlines all remain excluded because the name is
# embedded verbatim inside a Markdown code span in the file we write.
DEPENDENCY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@/.:_-]{0,199}$")

# Allowlist for version strings (semver, PEP 440, git tags with +/_ metadata).
# Empty versions are tolerated separately by the caller; a non-empty version
# must match this shape exactly.
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,99}$")

# Allowlist for fetch-metadata's package-ecosystem identifiers (e.g. "uv",
# "github_actions", "pre-commit").
ECOSYSTEM_PATTERN = re.compile(r"^[A-Za-z0-9/_.-]{1,100}$")

# Hard cap on the total rendered bullet block. A legitimate grouped Dependabot
# PR renders a few hundred bytes; anything past 10KB is not a changelog entry,
# it is an injection attempt or a malfunction, and fails closed.
MAX_RENDERED_BLOCK_BYTES = 10 * 1024

# The changelog section this script is allowed to touch; its absence means the
# file is not structured the way this repository's changelog contract
# requires, which is unreadable-data territory (exit 2), never a write.
UNRELEASED_HEADING = "## Unreleased"

# The subsection inside Unreleased that dependency bumps land in; created on
# demand when a release freeze reset removed it.
DEPENDENCIES_HEADING = "### Dependencies"


class PolicyViolationError(github_api.GitHubApiError):
    """A fail-closed policy violation: untrusted input or authorship (exit 1).

    Subclasses the shared GitHubApiError (per its own docstring's convention
    for script-level defects) so main() can order its handlers from most to
    least specific and still catch both layers.
    """


@dataclass(frozen=True, slots=True)
class UpdatedDependency:
    """One validated dependency update from fetch-metadata's structured output."""

    name: str
    prev_version: str
    new_version: str
    ecosystem: str


@dataclass(frozen=True, slots=True)
class ChangelogBlob:
    """The head branch's CHANGELOG.md content paired with its Git blob sha.

    The blob sha is what the contents PUT presents as its optimistic-locking
    precondition, so both values must come from the same GET.
    """

    text: str
    blob_sha: str


def pr_reference_token(pr_number: int) -> str:
    """Return the idempotency key embedded in every bullet this script writes.

    The parenthesized form is self-delimiting: "(#19)" can never be found
    inside "(#192)" because the closing parenthesis pins the number's end.

    Args:
        pr_number: The pull request number the token refers to.

    Returns:
        The literal token, e.g. "(#192)".
    """

    return f"(#{pr_number})"


def parse_dependencies(payload: object) -> tuple[UpdatedDependency, ...]:
    """Validate fetch-metadata's parsed updated-dependencies array, fail-closed.

    Every field that will be rendered into the changelog must match its
    anchored allowlist regex; anything else -- wrong container type, missing
    field, non-string value, disallowed character -- raises PolicyViolationError
    (exit 1) rather than being sanitized, because sanitizing attacker-shaped
    input still writes attacker-chosen bytes.

    Args:
        payload: The already-json-decoded UPDATED_DEPENDENCIES_JSON value.

    Returns:
        The validated dependency updates, in input order.

    Raises:
        PolicyViolationError: The payload is not a non-empty array of objects
            whose fields all pass validation.
    """

    # fetch-metadata always emits a JSON array; a non-array (or an empty one,
    # which would mean "a Dependabot PR that updates nothing" -- nonsensical
    # and suspicious) fails closed.
    if not isinstance(payload, list) or not payload:
        raise PolicyViolationError(
            f"{UPDATED_DEPENDENCIES_ENV} must be a non-empty JSON array of dependency objects"
        )
    dependencies: list[UpdatedDependency] = []
    # Validate every element independently so the first bad element names
    # itself; one bad element rejects the whole payload (never a partial write).
    for index, element in enumerate(payload):
        # Each element must be a JSON object before any field can be read.
        if not isinstance(element, dict):
            raise PolicyViolationError(f"dependency element {index} is not a JSON object")
        name = element.get("dependencyName")
        # The name is embedded inside a Markdown code span in the file we
        # write, so it is held to the strictest allowlist.
        if not isinstance(name, str) or not DEPENDENCY_NAME_PATTERN.fullmatch(name):
            raise PolicyViolationError(
                f"dependency element {index} has a missing or disallowed dependencyName"
            )
        ecosystem = element.get("packageEcosystem")
        # The ecosystem is rendered in parentheses; same fail-closed rule.
        if not isinstance(ecosystem, str) or not ECOSYSTEM_PATTERN.fullmatch(ecosystem):
            raise PolicyViolationError(
                f"dependency element {index} has a missing or disallowed packageEcosystem"
            )
        versions: dict[str, str] = {}
        # prevVersion and newVersion share one rule: absent/null reads as
        # empty (fetch-metadata omits prevVersion for brand-new dependencies),
        # and a non-empty value must match the version allowlist exactly.
        for field in ("prevVersion", "newVersion"):
            value = element.get(field)
            # Normalize JSON null / absent to the empty string the renderer
            # understands, without letting falsy non-strings slip through.
            if value is None:
                value = ""
            # A non-string (number, object) or a disallowed non-empty string
            # both fail closed.
            if not isinstance(value, str) or (value and not VERSION_PATTERN.fullmatch(value)):
                raise PolicyViolationError(f"dependency element {index} has a disallowed {field}")
            versions[field] = value
        dependencies.append(
            UpdatedDependency(
                name=name,
                prev_version=versions["prevVersion"],
                new_version=versions["newVersion"],
                ecosystem=ecosystem,
            )
        )
    return tuple(dependencies)


def render_bullets(dependencies: Sequence[UpdatedDependency], pr_number: int) -> tuple[str, ...]:
    """Render the desired changelog bullets: one per dependency, sorted, size-capped.

    Every bullet is a single line by construction (validation already rejected
    newlines in every field) and ends with the pr_reference_token idempotency
    key, so later runs can find and converge on exactly these lines.

    Args:
        dependencies: The validated dependency updates to render.
        pr_number: The pull request number embedded in each bullet's token.

    Returns:
        The bullet lines, sorted by dependency name and deduplicated.

    Raises:
        PolicyViolationError: The rendered block exceeds MAX_RENDERED_BLOCK_BYTES.
    """

    rendered: list[tuple[str, str]] = []
    # Pair each line with its dependency name so the sort key is the name
    # itself (the contract's ordering), not merely the line text.
    for dep in dependencies:
        # An empty prevVersion means Dependabot introduced the dependency (or
        # could not determine the old version), so the "from X" clause is
        # omitted rather than rendered empty.
        if dep.prev_version:
            line = (
                f"- Bump `{dep.name}` from {dep.prev_version} to {dep.new_version} "
                f"({dep.ecosystem}) via Dependabot (#{pr_number})."
            )
        else:
            line = (
                f"- Bump `{dep.name}` to {dep.new_version} "
                f"({dep.ecosystem}) via Dependabot (#{pr_number})."
            )
        rendered.append((dep.name, line))
    ordered: list[str] = []
    # Sort by (name, line) for a deterministic order, then deduplicate exact
    # repeats (a grouped PR can list one dependency twice across directories).
    for _, line in sorted(rendered):
        # Only the first occurrence of an identical line survives.
        if line not in ordered:
            ordered.append(line)
    block_bytes = len("\n".join(ordered).encode("utf-8"))
    # The cap bounds what this script will ever write in one commit; exceeding
    # it is treated as hostile or malfunctioning input, never truncated.
    if block_bytes > MAX_RENDERED_BLOCK_BYTES:
        raise PolicyViolationError(
            f"rendered changelog block is {block_bytes} bytes, above the "
            f"{MAX_RENDERED_BLOCK_BYTES}-byte cap"
        )
    return tuple(ordered)


def live_pull_request_violations(
    payload: dict[str, Any], repo: str, head_ref: str
) -> tuple[str, ...]:
    """Re-validate the live pull request against the Dependabot-only policy.

    Server-side re-validation of everything the workflow's job-level `if`
    already asserted from the (immutable but potentially stale) event payload:
    the PR must still be open, authored by the real dependabot[bot] Bot
    account, and from a same-repository dependabot/ head branch.

    Args:
        payload: The live `gh api repos/{repo}/pulls/{n}` response.
        repo: The OWNER/REPO slug this run is authorized to write to.
        head_ref: The head branch name the workflow was invoked for.

    Returns:
        Every violation found; an empty tuple means the PR is trusted.
    """

    violations: list[str] = []
    user = payload.get("user") or {}
    # The login is the primary identity check; GitHub reserves the
    # "[bot]" suffix, but the type check below still backs it up.
    if user.get("login") != DEPENDABOT_LOGIN:
        violations.append(f"live PR author is {user.get('login')!r}, not {DEPENDABOT_LOGIN!r}")
    # user.type is server-attested: a human account renaming itself to look
    # bot-like still reports type "User".
    if user.get("type") != "Bot":
        violations.append(f"live PR author type is {user.get('type')!r}, not 'Bot'")
    # A closed/merged PR must never receive new autofill commits.
    if payload.get("state") != "open":
        violations.append(f"live PR state is {payload.get('state')!r}, not 'open'")
    head = payload.get("head") or {}
    head_repo = head.get("repo") or {}
    # A fork head would mean pushing the PAT-authored commit into someone
    # else's repository; only same-repo branches are ever written to.
    if head_repo.get("full_name") != repo:
        violations.append(
            f"live PR head repository is {head_repo.get('full_name')!r}, not {repo!r}"
        )
    live_ref = head.get("ref")
    # The branch this run writes to must be exactly the branch the workflow
    # event named, or the write target was swapped mid-flight.
    if live_ref != head_ref:
        violations.append(f"live PR head ref is {live_ref!r}, not {head_ref!r}")
    # Dependabot's own branches always live under dependabot/; anything else
    # is a masquerade regardless of the author checks above.
    if not str(live_ref or "").startswith(DEPENDABOT_BRANCH_PREFIX):
        violations.append(
            f"live PR head ref {live_ref!r} is not under {DEPENDABOT_BRANCH_PREFIX!r}"
        )
    return tuple(violations)


def authorship_violations(commits: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """Require every PR commit to be dependabot[bot]-authored and signature-verified.

    This is what defeats forged Dependabot-style metadata: fetch-metadata
    parses commit-message YAML, which a human with push access could forge,
    but they cannot forge GitHub's server-side dependabot[bot] authorship and
    `verification.verified` signature (verified live against PR #192: author
    'dependabot[bot]', verified true, committer 'web-flow'). A human commit on
    the branch means a human owns the changelog, so the gate stays red for
    human attention rather than autofilling over their work.

    Args:
        commits: The `gh api repos/{repo}/pulls/{n}/commits` objects.

    Returns:
        Every violation found; an empty tuple authorizes the write.
    """

    # A PR with no visible commits cannot be attested; fail closed rather
    # than treating the vacuous case as proven.
    if not commits:
        return ("pull request reports no commits; authorship cannot be proven",)
    violations: list[str] = []
    # Every commit must pass both checks; a single exception vetoes the write.
    for commit in commits:
        sha = str(commit.get("sha") or "unknown")[:12]
        author = (commit.get("author") or {}).get("login")
        # Authorship is the GitHub-account linkage, not the forgeable
        # commit-header name/email.
        if author != DEPENDABOT_LOGIN:
            violations.append(f"commit {sha} author is {author!r}, not {DEPENDABOT_LOGIN!r}")
        verified = ((commit.get("commit") or {}).get("verification") or {}).get("verified")
        # Only GitHub's own attested signature verification counts; anything
        # else (false, missing, or a truthy non-True) fails closed.
        if verified is not True:
            violations.append(f"commit {sha} is not signature-verified by GitHub")
    return tuple(violations)


def _unreleased_bounds(lines: Sequence[str]) -> tuple[int, int]:
    """Locate the '## Unreleased' section: (heading index, end index exclusive).

    The section ends at the next '## ' heading or end-of-file.

    Args:
        lines: The changelog text split into lines.

    Returns:
        The heading's line index and the exclusive end index of its section.

    Raises:
        github_api.GitHubApiError: No Unreleased section exists -- the file is
            not structured per the changelog contract, so nothing is writable.
    """

    start: int | None = None
    # Scan for the heading itself; strip() tolerates trailing whitespace while
    # still requiring the exact heading text.
    for index, line in enumerate(lines):
        # Only an exact heading match counts; a prose mention of the phrase
        # inside a longer line must not.
        if line.strip() == UNRELEASED_HEADING:
            start = index
            break
    # Absence is unreadable-structure territory (exit 2), never grounds to
    # invent a section in a file this script doesn't understand.
    if start is None:
        raise github_api.GitHubApiError(
            f"CHANGELOG.md has no {UNRELEASED_HEADING!r} section; refusing to write"
        )
    end = len(lines)
    # The section runs until the next same-level heading (the previous
    # release's '## X.Y.Z' line) or the end of the file.
    for index in range(start + 1, len(lines)):
        # Any '## ' heading (never '###') terminates the Unreleased section.
        if lines[index].startswith("## "):
            end = index
            break
    return start, end


def _dependencies_bounds(
    lines: Sequence[str], section_start: int, section_end: int
) -> tuple[int, int] | None:
    """Locate '### Dependencies' inside Unreleased: (heading index, end index exclusive).

    The subsection ends at the next '### ' heading or the Unreleased section's
    own end (which is already bounded by the next '## ' heading).

    Args:
        lines: The changelog text split into lines.
        section_start: The Unreleased heading's line index.
        section_end: The Unreleased section's exclusive end index.

    Returns:
        The subsection bounds, or None when the heading does not exist yet.
    """

    heading: int | None = None
    # Search only inside the Unreleased section; a Dependencies subsection in
    # an already-released section must never be touched.
    for index in range(section_start + 1, section_end):
        # Exact heading match, same reasoning as _unreleased_bounds.
        if lines[index].strip() == DEPENDENCIES_HEADING:
            heading = index
            break
    # A missing heading is a legitimate state (e.g. after a release freeze
    # reset); the caller creates it.
    if heading is None:
        return None
    sub_end = section_end
    # The subsection ends at the next sibling '### ' heading, if any exists
    # before the Unreleased section itself ends.
    for index in range(heading + 1, section_end):
        # Only a same-level '### ' heading terminates the subsection early.
        if lines[index].startswith("### "):
            sub_end = index
            break
    return heading, sub_end


def collect_pr_bullets(text: str, pr_number: int) -> tuple[str, ...]:
    """Collect the existing Dependencies bullets keyed to this PR's token.

    This feeds the idempotency check that terminates the self-trigger loop:
    when the collected set already equals the desired set, there is nothing
    left to write.

    Args:
        text: The head branch's current CHANGELOG.md text.
        pr_number: The pull request whose token keys the bullets.

    Returns:
        The exact bullet lines (verbatim) carrying this PR's token.

    Raises:
        github_api.GitHubApiError: The file has no Unreleased section.
    """

    lines = text.split("\n")
    section_start, section_end = _unreleased_bounds(lines)
    bounds = _dependencies_bounds(lines, section_start, section_end)
    # No Dependencies subsection means no bullets of ours can exist yet.
    if bounds is None:
        return ()
    token = pr_reference_token(pr_number)
    return tuple(
        line
        for line in lines[bounds[0] + 1 : bounds[1]]
        if line.lstrip().startswith("- ") and token in line
    )


def apply_pr_bullets(text: str, pr_number: int, desired: Sequence[str]) -> str:
    """Return the changelog text with this PR's bullets replaced or inserted.

    Replace-or-insert is keyed on the immutable pr_reference_token, so the
    operation converges across Dependabot force-push regenerations and
    group-membership changes without accumulating stale lines: every existing
    bullet carrying the token is removed and the desired set takes the first
    removed bullet's position (or is appended when none existed). A missing
    '### Dependencies' heading is created at the end of the Unreleased
    section, blank-line-separated on both sides.

    Args:
        text: The head branch's current CHANGELOG.md text.
        pr_number: The pull request whose token keys the bullets.
        desired: The rendered bullet lines to converge the file onto.

    Returns:
        The complete updated file text.

    Raises:
        github_api.GitHubApiError: The file has no Unreleased section.
    """

    lines = text.split("\n")
    section_start, section_end = _unreleased_bounds(lines)
    bounds = _dependencies_bounds(lines, section_start, section_end)
    # With no Dependencies heading at all, create the whole subsection at the
    # end of the Unreleased section, before the next '## ' heading.
    if bounds is None:
        insert_at = section_end
        # Swallow the section's trailing blank lines so exactly one blank line
        # separates the last existing content from the new heading.
        while insert_at > section_start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        block = ["", DEPENDENCIES_HEADING, "", *desired, ""]
        return "\n".join([*lines[:insert_at], *block, *lines[section_end:]])
    heading, sub_end = bounds
    subsection = list(lines[heading + 1 : sub_end])
    token = pr_reference_token(pr_number)
    keyed = {
        index
        for index, line in enumerate(subsection)
        if line.lstrip().startswith("- ") and token in line
    }
    # With existing keyed bullets, the desired set takes the first one's
    # position and every other keyed line is dropped -- replace-in-place that
    # preserves neighbors (other PRs' bullets) exactly where they were.
    if keyed:
        first = min(keyed)
        new_subsection: list[str] = []
        # Rebuild the subsection line by line so unrelated content keeps its
        # original relative order.
        for index, line in enumerate(subsection):
            # The first keyed bullet's slot receives the whole desired set.
            if index == first:
                new_subsection.extend(desired)
            # Every other keyed bullet is stale and is dropped.
            elif index in keyed:
                continue
            else:
                new_subsection.append(line)
    else:
        insert = len(subsection)
        # Append after the last non-blank content line so the subsection's
        # trailing blank separation from the next heading is preserved.
        while insert > 0 and not subsection[insert - 1].strip():
            insert -= 1
        # An empty subsection (heading directly followed by blank lines) needs
        # one blank line between the heading and the first bullet.
        separator = [""] if insert == 0 else []
        new_subsection = [*subsection[:insert], *separator, *desired, *subsection[insert:]]
        # Guarantee a blank line after the bullets when the subsection carried
        # no trailing blank of its own (e.g. heading was flush against the
        # next heading), keeping the output Markdown-lint-clean.
        if not new_subsection or new_subsection[-1].strip():
            new_subsection.append("")
    return "\n".join([*lines[: heading + 1], *new_subsection, *lines[sub_end:]])


def fetch_live_pull_request(pr_number: int, repo: str) -> dict[str, Any]:
    """Fetch the pull request's live state via REST for server-side re-validation.

    Args:
        pr_number: The pull request to fetch.
        repo: The OWNER/REPO slug (always explicit in this workflow).

    Returns:
        The decoded REST payload.
    """

    return json.loads(github_api.run_gh(["api", f"repos/{repo}/pulls/{pr_number}"]))


def fetch_changelog(repo: str, ref: str) -> ChangelogBlob:
    """Read the head branch's CHANGELOG.md as inert bytes via the REST contents API.

    Args:
        repo: The OWNER/REPO slug.
        ref: The commit sha to read the file at (the validated head sha, so
            the content and the blob sha the PUT preconditions on agree).

    Returns:
        The decoded file text and its Git blob sha.

    Raises:
        github_api.GitHubApiError: The file is missing (gh's 404 surfaces
            here) or its inline content was omitted (the contents API omits
            content for files over 1MB) -- unreadable data, never a write.
    """

    payload = json.loads(
        github_api.run_gh(["api", f"repos/{repo}/contents/CHANGELOG.md?ref={ref}"])
    )
    # The contents API omits inline content (encoding "none") for files over
    # 1MB; a changelog that large is outside this script's contract.
    if payload.get("encoding") != "base64" or not payload.get("content"):
        raise github_api.GitHubApiError(
            "CHANGELOG.md content was not returned inline (file over 1MB?); refusing to write"
        )
    text = base64.b64decode(payload["content"]).decode("utf-8")
    return ChangelogBlob(text=text, blob_sha=payload["sha"])


def fetch_pr_commits(pr_number: int, repo: str) -> list[dict[str, Any]]:
    """Fetch every commit on the pull request, across all pages.

    --paginate --slurp wraps each page in one JSON array of pages, so a PR
    with more than one page of commits is still fully checked -- the
    authorship proof must see EVERY commit.

    Args:
        pr_number: The pull request whose commits to list.
        repo: The OWNER/REPO slug.

    Returns:
        The flattened commit objects.
    """

    pages = json.loads(
        github_api.run_gh(
            ["api", f"repos/{repo}/pulls/{pr_number}/commits", "--paginate", "--slurp"]
        )
    )
    return [commit for page in pages for commit in page]


def put_changelog(repo: str, branch: str, blob_sha: str, text: str, pr_number: int) -> None:
    """PUT the updated CHANGELOG.md back to the PR head branch via REST.

    The PAT-authored commit fires a pull_request `synchronize` event, so the
    required checks (including the changelog gate this entry satisfies) re-run
    with full secrets -- a GITHUB_TOKEN push would not re-trigger them.

    Args:
        repo: The OWNER/REPO slug.
        branch: The head branch name to commit to.
        blob_sha: The blob sha from the paired GET; GitHub rejects the PUT
            with a conflict when the file has moved since, which is exactly
            the optimistic-locking behavior the retry logic relies on.
        text: The complete new file text.
        pr_number: The pull request number, embedded in the commit message.
    """

    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    github_api.run_gh(
        [
            "api",
            "-X",
            "PUT",
            f"repos/{repo}/contents/CHANGELOG.md",
            "-f",
            f"message=chore(deps): record Dependabot update in CHANGELOG (#{pr_number})",
            "-f",
            f"content={encoded}",
            "-f",
            f"sha={blob_sha}",
            "-f",
            f"branch={branch}",
        ]
    )


def _is_write_conflict(error: github_api.GitHubApiError) -> bool:
    """True when a failed contents PUT was an optimistic-locking conflict.

    GitHub reports a moved blob as HTTP 409 ("... does not match ...") and,
    for some precondition forms, HTTP 412; both mean "someone else moved the
    head", never a defect in this run's own request.

    Args:
        error: The GitHubApiError raised by the failed PUT.

    Returns:
        Whether the failure is the retryable conflict case.
    """

    lowered = str(error).lower()
    return "http 409" in lowered or "http 412" in lowered


def _sync_changelog(args: argparse.Namespace, desired: Sequence[str]) -> int:
    """Run the GET -> idempotency -> authorship -> compute -> PUT sequence.

    On an optimistic-locking conflict the sequence retries exactly once from
    the GET (re-GET, re-check idempotency, re-PUT -- the authorship proof is
    not re-run: whatever moved the head fired its own `synchronize`, whose
    fresh run re-proves authorship from scratch). A second conflict exits 0:
    that newer synchronize-triggered run supersedes this one.

    Args:
        args: The parsed command-line arguments.
        desired: The rendered bullet lines to converge the changelog onto.

    Returns:
        0 on success or benign no-op.

    Raises:
        PolicyViolationError: The authorship proof failed (exit 1).
        github_api.GitHubApiError: Required data was unreadable or the write
            failed for a non-conflict reason (exit 2).
    """

    # Exactly two attempts: the original pass and one conflict retry.
    for attempt in range(2):
        blob = fetch_changelog(args.repo, args.head_sha)
        existing = collect_pr_bullets(blob.text, args.pr_number)
        # IDEMPOTENCY FIRST -- the self-trigger loop terminator. This runs
        # deliberately BEFORE the authorship proof: on our own self-triggered
        # re-run the branch legitimately contains the PAT-authored changelog
        # commit, which the authorship proof would (correctly) reject.
        # Comparison is exact string equality, order-insensitive.
        if set(existing) == set(desired):
            print(
                f"CHANGELOG.md already carries the desired entry for pull request "
                f"#{args.pr_number}; nothing to write."
            )
            return 0
        # AUTHORSHIP PROOF gates the write, on the first attempt only (the
        # conflict retry deliberately skips it -- see this function's
        # docstring for why that is safe).
        if attempt == 0:
            commits = fetch_pr_commits(args.pr_number, args.repo)
            violations = authorship_violations(commits)
            # Any non-Dependabot or unverified commit vetoes the write and
            # leaves the changelog gate red for human attention.
            if violations:
                raise PolicyViolationError(
                    "refusing to write; commit authorship proof failed: " + "; ".join(violations)
                )
        new_text = apply_pr_bullets(blob.text, args.pr_number, desired)
        # The PUT preconditions on the GET's blob sha; a conflict means the
        # head moved between the two calls and is handled below, while every
        # other failure propagates to main()'s exit-2 handler.
        try:
            put_changelog(args.repo, args.head_ref, blob.blob_sha, new_text, args.pr_number)
        except github_api.PrimaryRateLimitError:
            raise
        except github_api.GitHubApiError as error:
            # Only the optimistic-locking conflict is retryable; anything
            # else is a real failure and must never exit green.
            if not _is_write_conflict(error):
                raise
            # First conflict: loop back to the GET for one retry.
            if attempt == 0:
                continue
            # Second conflict: whatever moved the head fired its own
            # synchronize event, whose run supersedes this one -- benign.
            print(
                "note: CHANGELOG.md moved twice during this run; deferring to the "
                "newer synchronize-triggered run."
            )
            return 0
        print(f"Wrote CHANGELOG.md entry for pull request #{args.pr_number} to {args.head_ref}.")
        return 0
    # Unreachable: every second-attempt path above returns or raises. Kept so
    # a static checker sees every code path produces or raises a value.
    raise AssertionError("unreachable: _sync_changelog always returns or raises")


def _run_autofill(args: argparse.Namespace) -> int:
    """Run every phase in order and return the process exit status.

    Split from main() so the rate-limit exit path (exit code 3) can be handled
    in exactly one place there, mirroring validate_project_metadata.py.

    Args:
        args: The parsed command-line arguments.

    Returns:
        0 on success or benign no-op.

    Raises:
        PolicyViolationError: A fail-closed policy violation (exit 1).
        github_api.GitHubApiError: Required data unreadable (exit 2), or a
            rate-limit condition (exit 3, via the PrimaryRateLimitError
            subclass main() handles first).
    """

    raw = os.environ.get(UPDATED_DEPENDENCIES_ENV)
    # The metadata is this script's entire input contract; without it there is
    # nothing trustworthy to render (exit 2 via the base error class).
    if raw is None or not raw.strip():
        raise github_api.GitHubApiError(
            f"required environment variable {UPDATED_DEPENDENCIES_ENV} is missing or empty"
        )
    # Undecodable JSON is unreadable required data (exit 2), distinct from
    # well-formed-but-disallowed content (exit 1, in parse_dependencies).
    try:
        dependencies_payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise github_api.GitHubApiError(
            f"{UPDATED_DEPENDENCIES_ENV} is not valid JSON: {error}"
        ) from error

    # LIVE PR RE-FETCH: server-side re-validation, never trusting the event
    # payload alone (the workflow's job-level `if` already screened it, but
    # this script independently proves the same facts against the live API).
    pr_payload = fetch_live_pull_request(args.pr_number, args.repo)
    violations = live_pull_request_violations(pr_payload, args.repo, args.head_ref)
    # Any identity/state violation fails closed before a single byte of the
    # metadata is even validated.
    if violations:
        raise PolicyViolationError("; ".join(violations))
    live_head_sha = (pr_payload.get("head") or {}).get("sha")
    # A moved head means a newer synchronize event already queued a fresh run
    # against the new sha; this superseded run bows out benignly.
    if live_head_sha != args.head_sha:
        print(
            f"note: live head sha {live_head_sha!r} differs from this run's "
            f"{args.head_sha!r}; a newer synchronize run supersedes this one."
        )
        return 0

    # VALIDATE + RENDER: allowlist validation and deterministic rendering of
    # the bullets this run will converge the changelog onto.
    dependencies = parse_dependencies(dependencies_payload)
    desired = render_bullets(dependencies, args.pr_number)

    return _sync_changelog(args, desired)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the changelog-autofill entry point.

    The updated-dependencies JSON deliberately arrives via the
    UPDATED_DEPENDENCIES_JSON environment variable rather than argv, because
    it is a large multi-line JSON blob set by the workflow's env indirection.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        Parsed arguments: PR number, repo slug, head ref, and head sha.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", required=True, help="GitHub OWNER/REPO")
    parser.add_argument("--head-ref", required=True, help="The PR head branch name")
    parser.add_argument("--head-sha", required=True, help="The PR head sha this run was queued for")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return its process exit status.

    Keeping orchestration here makes terminal behavior and error translation
    straightforward to audit: each exception layer maps to exactly one
    documented exit code, ordered from most to least specific.

    Args:
        argv: Optional command-line arguments; defaults to the process arguments.

    Returns:
        0 success/benign no-op, 1 policy violation, 2 unreadable required
        data, 3 rate-limit quota condition.
    """

    args = parse_args(argv)
    # Every phase raises rather than printing-and-returning, so the exit-code
    # mapping lives in exactly one place. Handler order matters: both specific
    # classes subclass GitHubApiError and must be caught before it.
    try:
        return _run_autofill(args)
    except github_api.PrimaryRateLimitError as error:
        print(f"quota: {error}", file=sys.stderr)
        return 3
    except PolicyViolationError as error:
        print(f"policy violation: {error}", file=sys.stderr)
        return 1
    except github_api.GitHubApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


# Standard script entry-point guard: only run main() when executed directly, not when
# imported (e.g. by this script's own test module).
if __name__ == "__main__":
    raise SystemExit(main())
