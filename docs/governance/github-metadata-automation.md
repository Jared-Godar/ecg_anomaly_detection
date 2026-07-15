# GitHub metadata automation record

This document records the programmatic repository and Project V2 metadata work used to establish
the modernization backlog. It provides representative, auditable examples without duplicating
complete issue bodies or treating generated command output as maintained documentation.

![Architecture-style diagram of governance automation. Four automated flows touch a GitHub container holding the repository and the ECG Pipeline Modernization project board: issue creation that skips exact-title matches, field updates that preserve curated values, a pull-request metadata gate validating labels, milestone, closing references, and project fields, and a post-merge job that sets status to Merged. Dashed lines mark manual maintainer steps such as catalog review and view configuration.](../diagrams/exports/governance-automation-overlay.svg)

*Where automation touches governance: idempotent issue creation, project board
field updates with read-back validation, the per-PR metadata gate, and
post-merge status sync — with maintainer judgment left in the loop.* —
generated from
[`governance-automation-overlay.dot`](../diagrams/src/governance-automation-overlay.dot)
via Graphviz (see [`docs/diagrams/design-spec.md`](../diagrams/design-spec.md)).

## Scope

The bootstrap process:

- verified the authenticated GitHub account and repository remote;
- created missing labels and verified existing milestones;
- created nine issues from a reviewed intake catalog without exact-title duplication;
- preserved the complete issue-body structure during creation;
- expanded Project #5 with structured planning fields and options;
- added every repository issue to the project;
- backfilled nine open issues and 34 historical pull requests; and
- verified every item had complete required project metadata.

The issue catalog was intentionally not retained. GitHub issues are now authoritative, and keeping
a second copy of every issue body would create an immediate drift risk.

## Idempotent issue creation

Automation checks all issue states by exact title before creation. A representative flow is:

```text
parse reviewed catalog
-> validate title, labels, milestone, and body sections
-> list open and closed issues
-> skip an exact-title match
-> create only missing labels and milestones
-> create the issue
-> read it back and compare title, body, labels, milestone, number, and URL
```

Exact label names matter. For example, `type:modernization` and `type: modernization` are distinct
GitHub labels. Validation compares normalized label sets rather than assuming visually similar
names are equivalent.

## Project V2 field update

Single-select fields are discovered by name and options by exact value. Existing item values are
preserved unless blank or clearly inconsistent with GitHub state. Updates use field and option
node IDs returned by Project V2 queries.

A representative mutation shape is:

```graphql
mutation UpdatePlanningField(
  $project: ID!
  $item: ID!
  $field: ID!
  $option: String!
) {
  updateProjectV2ItemFieldValue(
    input: {
      projectId: $project
      itemId: $item
      fieldId: $field
      value: {singleSelectOptionId: $option}
    }
  ) {
    projectV2Item {
      id
    }
  }
}
```

Authentication tokens, raw API responses, local paths, and temporary operational scripts are not
committed.

## Deterministic mapping precedence

Historical mappings use this evidence order:

1. GitHub item state and merge state
2. Explicit repository labels
3. Assigned milestone
4. Issue or pull-request identifier and title
5. Body language and acceptance criteria

Examples:

| Evidence | Project value |
|---|---|
| Closed issue | Status = Closed |
| Merged pull request | Status = Merged |
| Open issue without linked implementation | Status = Backlog |
| `risk: evaluation` | Risk = High |
| `size: m` | Size = M |
| M5 developer-experience issue | Workstream = Developer Experience |
| Reproducibility evidence deliverable | Portfolio Signal = Reproducibility |

When multiple values remain plausible, automation leaves the field blank for maintainer review.

## Validation

After mutation, validation reads the project back and checks:

- expected fields and exact options exist;
- every current issue is a project item;
- item URLs and GitHub numbers resolve correctly;
- open issues and merged pull requests have coherent statuses;
- every item contains all required planning fields; and
- repository source files remain unchanged by metadata-only operations.

The initial result contained 43 project items and 387 populated required field values.

## API limitations

The supported `gh project` commands and public Project V2 GraphQL mutations can manage fields,
options, items, and item values. They do not currently provide supported creation or editing of
saved project views and built-in workflow actions. Those settings require web-interface review.

This limitation is why view definitions and expected workflow transitions are maintained in
[GitHub Project governance](github-project.md).

## Automated pull-request metadata gate

`.github/workflows/metadata-governance.yml` runs `scripts/github/validate_project_metadata.py`
on every pull request event, converting the field requirements above from documentation-only
guidance into an enforced check. The script and workflow are deliberately separate: the script
takes no GitHub Actions-specific input (only `--pr-number`, `--repo`, `--owner`,
`--project-number`, `--strict-project-checks`, and `--min-graphql-quota`), so it runs
identically from a terminal for local debugging. Exit codes distinguish outcomes: 0 passed,
1 metadata violations, 2 required data unreadable, 3 a GraphQL quota condition (see
[GraphQL quota stewardship](#graphql-quota-stewardship)).

```fish
uv run python scripts/github/validate_project_metadata.py --pr-number 65
```

The check validates two layers:

- **Pull request level**: an assignee, at least one `type:*` label, at least one `area:*` label,
  and a body closing reference (`Closes #N`, `Fixes #N`, `Resolves #N`, and their keyword
  variants) to an issue. A milestone is also required, unless every issue the pull request closes
  is itself deliberately unmilestoned (per [issue workflow](issue-workflow.md)'s rule that a
  milestone is a delivery commitment, not a mandatory tag) — the check reads each closing issue's
  own milestone field and inherits that decision rather than forcing an unrelated milestone onto
  the pull request.
- **Linked issue level**: for every issue number extracted from a closing reference or a
  non-closing marker (see [Non-closing issue reference marker](#non-closing-issue-reference-marker)
  below), that the issue is a member of the tracked Project and has every field in
  [Required fields](github-project.md#required-fields) populated.

### Per-PR changelog gate

The same workflow runs a second, independent job (`Enforce per-PR changelog updates`) executing
`scripts/github/validate_changelog_update.py`, which mechanically enforces the standing
CHANGELOG contract (issue #184; see [release governance](releases.md)): a pull request whose
diff touches a substantive path (`src/`, `scripts/`, `docs/`, `configs/`,
`.github/workflows/`) must also touch `CHANGELOG.md`, or declare a visible
`changelog: not-needed -- <reason>` exemption line in its body. Marker text quoted inside
fenced code blocks or inline code spans is ignored, so a pull request documenting the
mechanism cannot exempt itself. The gate is REST-only (the pull request body and changed-file
listing), so it needs no Project-scoped token and consumes no shared GraphQL quota. Exit codes
mirror the metadata gate: 0 passed, 1 gate failed, 2 required data unreadable, 3 a rate-limit
condition.

```fish
uv run python scripts/github/validate_changelog_update.py --pr-number 65
```

### Non-closing issue reference marker

A receipts-gated pull request — one whose tracking issue must stay open past merge until
post-merge receipts land — cannot use a closing keyword without forcing a noisy
close-then-reopen cycle on its tracking issue. Issue #216 introduced a sanctioned alternative:
an explicit marker line in the pull request body that binds the PR to its tracking issue for
governance purposes (Project membership, field completeness, milestone inheritance) without
asking GitHub to auto-close the issue on merge.

**Syntax:**

```text
Non-closing ref: #N — <reason>
```

The marker is case-insensitive. The separator accepts either an em-dash (`—`) or a plain ASCII
hyphen (`-`). The reason after the separator is **mandatory** — it is the audit trail
justifying why the issue is deliberately kept open past this PR's merge. A marker with no
reason (or whitespace-only after the separator) is deliberately not recognized.

**Behavior:**

- Satisfies the closing-reference requirement in `validate_pull_request` (the PR passes
  without a closing keyword when at least one non-closing marker is present).
- Runs the **same** downstream checks as a closing reference: the referenced issue must be a
  tracked Project member with every required field populated, and its milestone is inherited
  for the PR's own milestone requirement.
- The **only** thing skipped is GitHub's auto-close — the issue stays open when the PR merges.
- The premature-closure observational check (issue #158) does not apply to non-closing refs,
  since those issues are *supposed* to remain open past merge.
- Fenced code blocks and inline code spans are stripped before matching, so a marker quoted as
  prose or example text is not mistaken for a real directive.

**Ambiguity is a hard violation:** naming the same issue number via both a closing keyword
(`Closes #N`) and a non-closing marker (`Non-closing ref: #N — reason`) in the same PR body
is reported as a metadata failure. Auto-closing and deliberately keeping open are contradictory
requests; the PR must choose one.

**When to use:**

- Multi-leg governance work whose tracking issue must accumulate post-merge receipts (e.g.
  closure-pass label changes, Project field updates) before it can be closed.
- A PR that delivers one leg of a broader tracking issue that will be closed by a later PR.

**When NOT to use:**

- When the PR fully completes the issue scope and the issue should close on merge — use a
  standard closing keyword (`Closes #N`, `Fixes #N`, `Resolves #N`).
- As an escape hatch to avoid linking work to a tracking issue — the marker enforces the same
  metadata bar as a closing keyword, just without the auto-close side effect.

### Token requirement and rollout

Reading Project V2 field values requires a token with the `project` scope. The default
repository-scoped `GITHUB_TOKEN` a workflow receives does not have that scope for a user-owned
project (the same limitation recorded above for the historical bootstrap). A repository secret
named `PROJECT_METADATA_TOKEN` supplies it.

That token must be a **classic** personal access token, not a fine-grained one. GitHub's
fine-grained tokens do not currently expose a Projects permission for a project owned by a user
account at all (only for organization-owned projects) — this is a documented platform limitation,
not a configuration mistake; see GitHub's own [personal access tokens
documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
and [Projects API guide](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-api-to-manage-projects).
Project #5 here is user-owned (`Jared-Godar`), so a classic token is the only supported option.
Scope it to `project` (read and write — needed by
[`project-status-sync.yml`](../../.github/workflows/project-status-sync.yml), which reuses this
same secret to explicitly set a merged pull request's Status field, see [GitHub Project
governance](github-project.md#automation), and by
`scripts/github/sync_dependabot_pr_metadata.py`, which populates a bot pull request's own board
fields) plus `public_repo` (this repository is public; grants
the read access to pull requests and issues this gate itself needs, and — since the bot-PR
automation below — also authorizes the contents-API `PUT` that
[`dependabot-autofill.yml`](../../.github/workflows/dependabot-autofill.yml) uses to write the
auto-generated changelog entry to a Dependabot branch, see [Bot-authored (Dependabot) pull
requests](#bot-authored-dependabot-pull-requests)) plus `read:org`. The last one
is easy to miss: every `gh project` subcommand resolves `--owner` by querying both the user and
organization GraphQL types, and a token without `read:org` fails that resolution with a
misleading `unknown owner type` error — indistinguishable at a glance from an actual ownership
problem, even though Project #5's owner (`Jared-Godar`, a user account) is correct (confirmed
live: see [cli/cli#7985](https://github.com/cli/cli/issues/7985) and
[cli/cli#8885](https://github.com/cli/cli/issues/8885)). Prefer these three narrow scopes over
the broader `repo` scope.

The workflow passes `--strict-project-checks`: an unreadable Project (missing or misconfigured
token) is a hard failure, not an advisory warning. The `GITHUB_TOKEN` fallback in the workflow's
`env` block exists only so a removed or rotated-out secret degrades to a clear authentication error
in the job log rather than the workflow silently not running; it is not a soft-enforcement mode.
Pull-request-level checks (assignee, milestone, labels, closing reference) and linked-issue-level
checks (Project membership and field completeness) both enforce immediately once
`PROJECT_METADATA_TOKEN` is configured.

### Local credential handling

`PROJECT_METADATA_TOKEN` is a GitHub Actions secret and is consumed only in CI. Never
materialize its value to a file on a local machine, even inside a gitignored directory such as
`secrets/` — a gitignored file is still plaintext on disk, readable by any local process, script,
backup, or tool with filesystem access, not just Git. Nothing in `scripts/` or `docs/` reads a
local token file, so there is never a reason to create one.

Running `scripts/github/validate_project_metadata.py`,
`scripts/github/set_merged_project_status.py`, or any ad hoc `gh project`/`gh pr`/`gh issue`
command locally instead needs only an interactive `gh auth login` session with the `project`
scope (`gh auth status` reports the active scopes; add it with `gh auth refresh -s project` if
it's missing). That session-based token is managed by the `gh` CLI's own credential storage, not
a file this repository's tooling ever touches.

Marking this workflow as a required status check in branch protection is a separate, manual
repository-settings decision, not configured by the workflow itself. As of #91, `Validate PR and
linked-issue metadata` is one of the required status checks on `main` — see [repository
governance](repository-governance.md#current-branch-protection-on-main) for the current
configuration.

### Maintainer override for confirmed infrastructure failures

`main`'s branch protection has `enforce_admins` enabled — see [repository
governance](repository-governance.md#current-branch-protection-on-main) — so a required check
blocks merge unconditionally, including for the repository owner. That is deliberate: it is the
strongest available signal that the rules bind the maintainer too. It has one narrow, documented
exception (#157), added after a live case where PR #155's `Validate PR and linked-issue metadata`
check failed on `GraphQL: API rate limit already exceeded`, independently confirmed via
`gh api rate_limit` to be a full point-budget exhaustion from unrelated same-session API usage —
not a defect in that pull request.

The exception applies only when a required check's failure is independently proven to be pure
infrastructure, never merely suspected or convenient. When it applies:

1. Record the proof (e.g. `gh api rate_limit` output, or equivalent independent evidence for a
   different failure mode) in a pull request comment before doing anything else.
2. Temporarily disable `enforce_admins`:

   ```fish
   gh api --method DELETE repos/Jared-Godar/ecg_anomaly_detection/branches/main/protection/enforce_admins
   ```

3. Merge the pull request.
4. Immediately re-enable `enforce_admins`:

   ```fish
   gh api --method POST repos/Jared-Godar/ecg_anomaly_detection/branches/main/protection/enforce_admins
   ```

`enforce_admins` must return to `true` immediately after the merge — this is a one-time, audited
exception per use, not a standing bypass. Softening the metadata gate itself (e.g. to a warning)
is explicitly rejected as an alternative: the same gate caught three real metadata gaps on PR #155
in the same session this exception was proposed, and a warning would blunt that protection for
every future pull request, not just a rare confirmed-infrastructure case.

### Why issue creation is not blocked

GitHub provides no rejection mechanism for issue creation comparable to a required pull-request
status check. Issue-only metadata gaps (an issue with no open pull request yet) remain a
manual-review concern; enforcement here starts at the pull-request stage, where a required check
can actually block a merge.

### Tests

`tests/scripts/test_validate_project_metadata.py` covers the pure validation logic (closing-
reference extraction, pull-request-level checks, linked-issue field-completeness checks) directly,
and the `gh`-subprocess boundary with mocked process output -- including the quota safeguards:
one native read per closing issue reused across stages, one Project snapshot per run, the
preflight stop, and the distinct quota exit code. `tests/scripts/test_github_api.py` covers the
shared access layer (rate-limit classification, quota accounting, schema caching, the targeted
lookup and read-back, and the union-type and unset-field edge cases).
`tests/scripts/test_detect_label_drift.py` and `tests/scripts/test_sync_github_labels.py` cover
the two label-hygiene scripts that migrated onto the same layer in issue #175, including their
observe-only preflight defaults (a drained pool must never block a manual hygiene run) and the
distinct quota exit code. `tests/scripts/test_validate_changelog_update.py` covers the
changelog gate (issue #184): the substantive-path classification, the exemption-marker rules
(including that a marker quoted in code spans or fenced blocks is never honored), the
failure/exemption/happy decision paths end to end, and the unreadable-data and rate-limit exit
codes. `tests/scripts/test_project_label_mapping.py`,
`test_populate_project_item.py`, and `test_detect_board_drift.py` cover the creation-time
board automation (issue #233): the mapping table's manifest-completeness invariant, the
governed-bot skip, add-then-verify membership, the fill-only-unset precedence rule, conflict
withholding, the drift cross-check, and their exit-code mappings. None of these calls the
network; running them requires no GitHub token.

## Bot-authored (Dependabot) pull requests

### Problem and chosen policy

Both required merge gates were structurally unable to pass on a Dependabot pull request, and
PR #192 demonstrated it live: the [metadata gate](#automated-pull-request-metadata-gate)
requires an assignee, `type:*` and `area:*` labels, a milestone or an inherited waiver, a
closing reference, and complete Project #5 fields — none of which Dependabot supplies — and the
[changelog gate](#per-pr-changelog-gate) requires a `CHANGELOG.md` update that Dependabot never
writes. Issue #193 evaluated the options and chose **automation, not exemption**: a governed
workflow auto-generates the changelog entry on the bot's own branch, and companion automation
populates the pull request's board metadata.

Two deliberate non-exemptions bound this policy:

- The changelog contract is **not** waived for bot authors. The entry is generated, but it must
  exist in the pull request's diff for the gate to pass, exactly as for a human author
  (issue #184's no-silent-bypass principle).
- The metadata gate waives only the linked-issue and milestone checks for governed bot authors
  (a dependency bump closes no issue and is deliberately unmilestoned). That waiver is
  compensated, not free: the validator instead enforces the pull request's **own** Project #5
  membership and all nine planning fields (see [the field-default table
  below](#bot-pull-request-field-defaults)), so a bot pull request faces the same
  complete-metadata bar, sourced differently.

### Native metadata via `.github/dependabot.yml`

Everything Dependabot can attach natively is configured declaratively rather than scripted:

- **Labels** (per ecosystem, because the gate requires one `type:*` **and** one `area:*`
  label): the `github-actions` ecosystem gets `dependency: external`, `type: maintenance`, and
  `area: ci-cd`; the `pre-commit` ecosystem gets `dependency: external`, `type: maintenance`,
  and `area: quality`.
- **Assignee**: `Jared-Godar`, satisfying the gate's assignee requirement.
- **Milestone**: deliberately none. Bot dependency pull requests are unmilestoned by design — a
  milestone is a delivery commitment ([issue workflow](issue-workflow.md)), and routine
  dependency currency is standing stewardship, not scheduled delivery. The validator's
  governed-bot path waives milestone inheritance accordingly instead of forcing one.

### Autofill workflow security model

`.github/workflows/dependabot-autofill.yml` is the repository's only `pull_request_target`
workflow, which makes its security model the load-bearing part of this policy. Under
`pull_request_target` the workflow runs in the **base** repository context with normal Actions
secrets available — a configuration that is famously dangerous when combined with untrusted
pull-request content. It is safe here because of, and only because of, the following invariants,
each auditable in the workflow file:

- **(a) Triple identity gate on the immutable event payload.** The job runs only when
  `github.event.pull_request.user.login == 'dependabot[bot]'`, the head repository is this
  repository (no forks), and the head ref starts with `dependabot/`. The conditions read the
  event payload, never `github.actor` — the actor is whoever caused the event (for example a
  human pushing to the bot's branch or re-running the workflow) and is not evidence of
  authorship.
- **(b) Base-ref-only checkout.** The single `actions/checkout` step checks out the base branch
  (the `pull_request_target` default) with `persist-credentials: false`. No step ever checks
  out, downloads, or executes pull-request-head content; the head's `CHANGELOG.md` is read as
  inert bytes over the REST contents API, never written to the runner's working tree.
- **(c) Structured, allowlisted entry content.** The changelog entry text derives exclusively
  from the SHA-pinned `dependabot/fetch-metadata` action's structured outputs, validated
  against fail-closed regex allowlists before use, and every dynamic value crosses into `run:`
  bodies via `env` — there is no `${{ }}` expression interpolation inside any `run:` body, so
  no value can be re-parsed as shell.
- **(d) Per-commit server-side authorship proof.** Before any write, the entry writer
  (`scripts/github/autofill_dependabot_changelog.py`) verifies via the API that **every** commit on the pull request is authored by `dependabot[bot]` with
  GitHub's `verification.verified` flag true. This defeats the residual insider vector the
  identity gate alone cannot: a human with push access appending a commit (with forged
  metadata) to a genuine `dependabot/**` branch.
- **(e) One confined write path.** The only write is a contents-API `PUT` of `CHANGELOG.md` to
  the head branch, authenticated by the classic `PROJECT_METADATA_TOKEN` PAT scoped into that
  single trusted step's `env`. The workflow's `GITHUB_TOKEN` stays read-only throughout
  (`contents: read`, `pull-requests: read`), and `fetch-metadata` visibly receives
  `secrets.GITHUB_TOKEN`, never the PAT.
- **(f) Idempotency terminates the loop.** The PAT-authored push re-triggers the workflow
  (`synchronize`); the entry writer is a `(#N)`-keyed replace-or-insert, so the second run
  finds its own entry current and makes no write, terminating the self-trigger loop. The same
  property self-heals Dependabot's own force-pushed rebases, which discard the entry commit.
- **(g) Mechanical enforcement against drift.** `scripts/check_privileged_workflow_safety.py`
  re-verifies the structural invariants of every `pull_request_target` workflow, so a future
  edit cannot quietly weaken them. Its five rules: (1) no `actions/checkout` step may set
  `with.ref` to the pull request's head; (2) no `${{ }}` expression interpolation inside any
  `run:` body — dynamic values must cross as `env`; (3) every job's `if:` must pin
  `github.event.pull_request.user.login` and may never consult `github.actor`; (4)
  `persist-credentials: false` on every checkout step; (5) a top-level `permissions:` block
  must exist and must not grant `contents: write`, keeping the ambient `GITHUB_TOKEN`
  read-only.

Adding any pull-request-head checkout, or executing head-derived files, would reintroduce
remote code execution with a write-capable PAT in scope. Do not do it; the safety checker
exists to make that mistake loud.

### Operational notes

- **Why the PAT writes the entry.** A push authenticated by the workflow's own `GITHUB_TOKEN`
  does not trigger new workflow runs, so the required gates would never re-evaluate the pull
  request after the entry lands. The PAT-authored commit fires a `pull_request: synchronize`
  event, re-running the required checks against the now-complete diff.
- **First-run redness of the metadata gate is expected.** The metadata-governance workflow runs
  on plain `pull_request`; when Dependabot triggers it, GitHub supplies only Dependabot's
  secrets, so `PROJECT_METADATA_TOKEN` resolves empty and the strict Project read fails. The
  autofill workflow itself is unaffected — `pull_request_target` receives normal secrets even
  when Dependabot triggers it — and its changelog push re-runs the gates under a human-adjacent
  actor with full secrets. The initial red check is self-correcting, not flakiness.
- **Quota profile.** `scripts/github/sync_dependabot_pr_metadata.py` follows the
  [consumption rules](#consumption-rules): one targeted PR-item lookup, an optional item-add
  with a verifying re-lookup, one field-list schema read, and at most nine
  `item-edit`/read-back pairs (skipping fields already curated) — roughly 15–25 points, guarded
  by a preflight default of 50; it never takes a full-board snapshot.
- **Never a required check.** `dependabot-autofill.yml` must not be added to branch
  protection's required status checks: it runs only for Dependabot-authored pull requests, so
  as a required check it would deadlock every human pull request, and its job is to feed the
  existing required gates, not to be one.

### Bot pull-request field defaults

When the metadata sync runs, it adds the pull request to Project #5 (if absent) and populates
exactly these nine fields. A field that already holds a value is preserved untouched — curation
by the maintainer always wins over the defaults.

| Field | Default | Rationale |
|---|---|---|
| Status | Review | A bot pull request arrives ready for maintainer review, never in Backlog |
| Workstream | Stewardship | Dependency currency is standing repository stewardship |
| Issue Type | Technical Debt | Version drift is debt paid down routinely |
| Priority | Low | Routine grouped bumps; a security advisory escalates manually |
| Risk | Low | SHA-pinned, gate-validated, human-merged changes |
| Size | XS | A grouped version bump is the smallest reviewable unit |
| Repository Area | ci-cd | Both managed ecosystems are CI/quality tooling |
| Portfolio Signal | Operational Maturity | Automated dependency governance is itself the demonstrated signal |
| Target Release | Stewardship | Unmilestoned by design; tracked in the standing stewardship lane |

### Auto-merge: considered and deferred

Issue #193's third scope item — enabling auto-merge for green bot pull requests — was
considered and **deferred**. No auto-merge is configured: every bot pull request still requires
the maintainer's explicit merge click, keeping a human decision on every change that reaches
`main`. Revisiting that decision is tracked separately from this policy.

## Creation-time board population (issue #233)

**Canonical rule:** every issue and pull request must be a Project #5 member carrying the
mandatory label set. Membership and field population were entirely manual at item-creation time
until issue #233, and the gap bit repeatedly — #210 was never added to the board at filing, and
the ten #218-audit findings issues (#223–#232) landed with zero board presence, costing a
~100-operation manual back-fill that was interrupted mid-batch. The
[`project-item-autofill`](../../.github/workflows/project-item-autofill.yml) workflow makes the
rule the default rather than a per-item ritual.

### Behavior

The workflow fires on `issues: [opened, labeled]` and `pull_request: [opened, labeled]` and runs
`scripts/github/populate_project_item.py`, which:

1. **Adds the item to Project #5 when absent** — idempotent, verified by a targeted re-lookup
   (`gh project item-add`'s exit status is never accepted as evidence).
2. **Defaults Status to Backlog only when Status is unset** — the automation never regresses a
   lane a human, an agent, or a built-in workflow already advanced. The ensure-Backlog check
   runs on every event (not just `opened`), so a missed `opened` run is back-filled by the next
   `labeled` event.
3. **Fills every label-derivable field that is currently unset**, converging as labels land:

   | Label namespace | Project field | Mapping notes |
   |---|---|---|
   | `type:*` | Issue Type | Direct name matches plus `type: maintenance` → Technical Debt (the Dependabot-defaults precedent); `type: modernization` deliberately unmapped (Feature vs. Enhancement is judgment) |
   | `priority:*` | Priority | `p0` → Critical, `p1` → High, `p2` → Medium, `p3` → Low (the taxonomy ladder rung-for-rung; verified live against board items) |
   | `risk:*` | Risk | `risk: low` → Low; the domain labels (`data-integrity`, `evaluation`, `security`) → High per the historical mapping precedent above; Medium has no label source |
   | `size:*` | Size | Direct ladder match `xs`/`s`/`m`/`l` → XS/S/M/L; the board's XL has no label source |
   | `area:*` | Repository Area | Exact same-named options only (`ci-cd`, `documentation`, `evaluation`, `modeling`, `validation`); `cli`/`data`/`pipeline`/`quality`/`repository` were audited per-label by #237 and recorded as **permanently human-set** (rationale in the mapping module and the [label taxonomy](label-taxonomy.md#label-to-board-field-alignment-237)) |
   | `portfolio:*` | Portfolio Signal | Direct name matches (`operational-maturity`, `testing-rigor`, `agentic-engineering`, plus `governance`, minted by #237 for the board's pre-existing Governance option); `case-study`/`release` are lifecycle markers with no same-named option and stay human-set (#237) |

   The authoritative table is code, not this summary:
   `scripts/github/project_label_mapping.py`, whose unit tests enforce that every label in
   `.github/labels.json` is either mapped, explicitly listed as deliberately unmapped, or in a
   namespace with no board counterpart — growing the taxonomy without deciding its board
   translation fails the suite.

Boundaries, all deliberate:

- **Curated values win.** Only UNSET fields are ever written; a populated field is preserved
  untouched regardless of what the labels derive. Every write is verified by a targeted
  read-back with one bounded retry (#164/#170).
- **Workstream and Target Release are never inferred.** They have no label source; heuristic
  inference (title prefixes, defaults) is an explicit non-goal, because a confidently-wrong
  value reads as deliberate triage. The PR-time metadata gate remains the enforcement point for
  full nine-field completeness.
- **Conflicting derivations are withheld.** Two labels deriving different options for one field
  (e.g. `risk: low` plus `risk: security`) produce a logged warning and no write — ambiguity is
  for maintainer review.
- **Governed-bot items are skipped** (workflow condition plus a server-side REST author check),
  because [the Dependabot autofill path](#bot-authored-dependabot-pull-requests) owns bot PRs'
  board stamping with different defaults. Fork pull requests are also skipped at the workflow
  level: plain `pull_request` events from forks receive no secrets, so the run could only fail;
  the scheduled backstop below catches them instead.
- **Option IDs are resolved by name at runtime** (one cached `field-list` read per run), never
  hardcoded — the 2026-07-14 board-wide option-ID regeneration
  ([github-project.md](github-project.md#setting-status-via-the-cli)) proved stored IDs go
  stale wholesale.

### Testing and change caveat

`issues:`-triggered workflows execute the workflow definition from the **default branch**, so
changes to `project-item-autofill.yml` (or the script behind it) take effect only after merge
and cannot be exercised from a feature branch. The mapping/idempotency/precedence logic is
therefore unit-tested exhaustively (`tests/scripts/test_populate_project_item.py`,
`test_project_label_mapping.py`, `test_github_api.py`), and behavior changes should be verified
post-merge with one disposable probe issue: file it with taxonomy labels, read back membership
and fields, add one more label to confirm convergence, then close it as not-planned with lane
`Not Planned`.

### Scheduled backstop

The weekly [`repository-hygiene`](../../.github/workflows/repository-hygiene.yml) run gained a
`board-drift` job executing `scripts/detect_board_drift.py`: for every OPEN issue and pull
request (governed-bot items excluded) it flags missing board membership, an unset Status, and
any unset field whose deriving label is present. It never flags a populated field that differs
from its derivation (curated values win) and never mutates anything — remediation is a manual,
read-back-verified `populate_project_item.py` run or the fallback commands below. It exits 1 on
drift, using the house exit-code convention otherwise (0 clean, 2 unreadable, 3 quota).

### Manual fallback

When the automation is down, or for an item it cannot reach (e.g. a fork PR), the same
convergence runs from any checkout with a `project`-scoped `gh` session:

```fish
# Converge one item (issue or pull request) exactly as the workflow would.
uv run python scripts/github/populate_project_item.py --content-type issue --number <N>
uv run python scripts/github/populate_project_item.py --content-type pull-request --number <N>

# Check the whole repository for membership/field gaps without mutating anything.
uv run python scripts/detect_board_drift.py
```

Fields the automation deliberately leaves blank (Workstream, Target Release, any conflicted or
unmapped derivation) are set manually with the read-back-verified `gh project item-edit` loop in
[GitHub Project governance](github-project.md#setting-status-via-the-cli).

## GraphQL quota stewardship

Issue #173 hardened the repository's governance automation against avoidable GraphQL
consumption, after issue #171 and PR #170 demonstrated the failure mode live: the GraphQL
point budget is **one shared 5000-points/hour pool per user account**, drawn on simultaneously
by interactive `gh` sessions, coding-agent sessions and their subagents, and the
`PROJECT_METADATA_TOKEN`-backed workflows (a classic PAT belongs to the same user, so it shares
the same pool). PR #170's merge-time `project-status-sync` run failed on a pool drained by
unrelated same-session work and was recovered with `gh run rerun --failed` after the hourly
reset.

### Which repository automation consumes GraphQL

Projects V2 has **no REST API**: every `gh project` subcommand (`view`, `field-list`,
`item-list`, `item-edit`, `item-add`) and every `gh api graphql` call is GraphQL. Several other
`gh` abstractions are GraphQL-backed even though nothing in their syntax says so; the table
below records the transport of every repository-owned call site (the issue #173 inventory,
updated by issue #175 when the two label-hygiene scripts migrated onto the shared layer).
`gh`'s underlying transport for a given command can change across CLI versions -- re-verify
this table when upgrading `gh` majors.

| Call site | Command shape | Transport |
|---|---|---|
| `scripts/github/set_merged_project_status.py` | `gh api graphql` targeted PR-item lookup and Status read-back; `gh project field-list` / `item-edit` | GraphQL (~5 points/run; no full-board reads) |
| `scripts/github/validate_project_metadata.py` | one `gh project item-list --limit 500` snapshot | GraphQL (measured live 2026-07-12: **203 points** for one snapshot -- GraphQL pricing scales with requested node counts, so this dominates the script's spend) |
| `scripts/github/validate_project_metadata.py` | `gh api repos/.../pulls/N`, `.../issues/N`, `.../issues/N/timeline` | REST (moved off `gh pr view` / `gh issue view`, which are GraphQL-backed) |
| `scripts/github/validate_changelog_update.py` | `gh api repos/.../pulls/N` and `.../pulls/N/files --paginate` (issue #184) | REST only -- zero GraphQL spend, so it runs no quota preflight; the primary REST rate limit still maps to exit code 3 |
| the four `QuotaMonitor`-carrying scripts (`set_merged_project_status.py`, `validate_project_metadata.py`, `detect_label_drift.py`, `sync_github_labels.py`) | `gh api rate_limit` preflight/report | REST; the endpoint is documented as not counting against any quota |
| `scripts/detect_label_drift.py` | `gh issue list` / `gh pr list --json` via the shared `run_gh` (issue #175) | GraphQL-backed listing; low frequency (manual/scheduled hygiene); observe-only preflight default (0) with before/after/consumed reporting and quota exit code 3 |
| `scripts/sync_github_labels.py` | `gh label create --force` per manifest label via the shared `run_gh` (issue #175; the pre-migration inventory listed `label list`/`edit`, which the script never actually called) | REST mutations -- no GraphQL spend of its own; low frequency (manual); observe-only preflight default (0) with before/after/consumed reporting and quota exit code 3; `--dry-run` makes no `gh` calls at all |
| `scripts/github/sync_dependabot_pr_metadata.py` | `gh api graphql` targeted PR-item lookup, optional item-add with verifying re-lookup, `gh project field-list`, up to nine `item-edit`/read-back pairs (issue #193) | GraphQL (~15-25 points/run; preflight default 50; no full-board reads) |
| `scripts/github/autofill_dependabot_changelog.py` | `gh api repos/...` pull-request, commit, and contents reads plus the changelog contents `PUT` (issue #193) | REST only -- zero GraphQL spend |
| `scripts/github/populate_project_item.py` | `gh api graphql` targeted issue/PR-item lookup, optional item-add with verifying re-lookup, `gh project field-list`, up to seven `item-edit`/read-back pairs; the content/labels/author read is REST (issue #233) | GraphQL (~10-25 points/run; preflight default 50; no full-board reads) |
| `scripts/detect_board_drift.py` | `gh issue list` / `gh pr list --json` plus one `gh project item-list` snapshot (issue #233) | GraphQL-backed listings plus the ~203-point snapshot; low frequency (weekly/manual hygiene); preflight default 250 (snapshot-sized, like the validator) with before/after/consumed reporting and quota exit code 3 |
| `.github/workflows/*.yml` | no direct `gh` calls | -- (workflows only invoke the scripts above) |

### Consumption rules

The shared access layer (`scripts/github/github_api.py`) owns these rules; new governance
tooling should build on it rather than shelling out to `gh` directly:

- **One full Project snapshot per logical phase, maximum.** Board-wide discovery or validation
  may read the whole item list once; the result is cached and reused across every per-item
  check in that phase. A full `item-list` inside a per-item loop is a defect (the exact pattern
  issue #173 removed from `set_merged_project_status.py`, whose verification reads previously
  paid full-board pagination twice per run).
- **Targeted read-back after every actual mutation.** Verification reads exactly the mutated
  item's field via GraphQL `node(id:)` + `fieldValueByName` -- one point regardless of board
  size. The read-back is never cached and never skipped: mutation exit status alone is not
  proof of success (`error: no changes to make` is inconclusive until a fresh read confirms
  the value), and a stale first read-back earns one bounded retry. Unset fields return JSON
  `null` and are reported as unset; a value of an unexpected union type (Project field values
  span several GraphQL union types) is an explicit error, never misread as unset.
- **Cache identity and schema per operation.** Project ID, field IDs, and option IDs are
  resolved at most once per run and reused; the targeted PR-item lookup returns the project ID
  alongside the item ID, so no separate `gh project view` is needed at all.
- **REST for native metadata.** Labels, assignees, milestones, comments, issue/PR state, and
  issue timelines are read via `gh api repos/...` REST endpoints, keeping native reads off the
  shared GraphQL pool entirely.
- **Preflight, threshold, and stop.** Before its GraphQL phase, each script reads
  `gh api rate_limit` (free) and stops -- before any mutation -- when the remaining GraphQL
  quota is below `--min-graphql-quota` (`0` disables the stop but keeps the reporting). The
  defaults are sized per script to the phase they guard: 250 points for the validator
  (covers its measured 203-point snapshot with margin) and 25 for the merge sync (~5 points
  needed -- demanding snapshot-sized headroom there would block a cheap sync a moderately
  drained pool could easily serve). Both scripts are idempotent and resumable, so a stopped
  run is simply rerun after the reset; `set_merged_project_status.py` additionally skips its
  mutation outright when a fresh pre-check already reads `Merged`.
- **Report before/after/consumed.** Every run prints a
  `GraphQL quota: N before, M after, K consumed` line on success and failure alike, so any
  future pool drain can be attributed from workflow logs. The delta is a pool-level
  measurement: concurrent consumers of the same shared pool can inflate a run's apparent
  consumption, so treat the line as evidence, not as an exact per-run invoice.
- **Distinguish quota from defects.** Primary (hours-long) rate-limit exhaustion and a failed
  preflight exit with code **3** and a `quota:` prefix -- never the metadata-violation (1) or
  data-read-failure (2) codes -- so CI output can never make a drained pool look like a broken
  pull request. The transient secondary/abuse-detection throttle and transient GitHub 5xx
  server errors (issue #190: a freshly opened pull request's diff can be momentarily
  uncomputed, surfacing as `Server Error ... (HTTP 500)`) share the same short bounded
  retries (2s/5s/10s); the primary limit is never retried in-job, and 4xx caller errors
  always fail fast.

### Recovery and local artifacts

When a workflow run fails with exit code 3 (or a confirmed `rate limit already exceeded`
message), the pull request is not the problem: wait for the hourly window
(`gh api rate_limit` shows the reset time) and `gh run rerun --failed`. The
maintainer-override procedure for a required check blocked by confirmed infrastructure failure
is documented [above](#maintainer-override-for-confirmed-infrastructure-failures).

Ad hoc board snapshots taken during governance work (e.g. the one discovery snapshot of a
manual phase) are working evidence, not documentation: keep them in temporary directories or
gitignored locations, never in tracked repository content -- the same boundary this document
already applies to tokens and raw API responses.
