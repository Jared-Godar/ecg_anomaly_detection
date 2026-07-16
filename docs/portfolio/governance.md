# Governance

The README's [Governance](../../README.md#governance) section makes one claim: process discipline
here is enforced through CI and project-board automation rather than trust. This page tells the
fuller story behind each bullet — what was built, why it took that shape, and where each mechanism
lives in the repository. The same boundary applies throughout: governance disciplines *process*;
it does not establish model quality or medical utility. This project is a research, education, and
software-engineering demonstration, not medical software.

The unifying design decision is that every rule worth having gets a mechanical check, and the
checks bind the maintainer too: `main`'s branch protection runs with `enforce_admins` enabled, so
a failing required check blocks the repository owner's own merge button. The single documented
exception — a required-check failure independently proven to be pure infrastructure — is itself a
recorded, per-use, immediately-reverted procedure
([repository governance](../governance/repository-governance.md#current-branch-protection-on-main)).

## Nine-field Project tracking with a GraphQL-backed gate

Every issue and pull request on the public ECG Pipeline Modernization board carries nine required
fields — Status, Workstream, Issue Type, Priority, Risk, Size, Repository Area, Portfolio Signal,
and Target Release ([field definitions](../governance/github-project.md#required-fields)). Fields
live on the board rather than as labels so that execution and portfolio metadata never overload
the repository's label surface, while GitHub issues stay the source of truth for scope.

Field completeness is not an honor system.
[`metadata-governance.yml`](../../.github/workflows/metadata-governance.yml) runs
[`validate_project_metadata.py`](../../scripts/github/validate_project_metadata.py) on every
substantive pull request event as a required status check: the PR must have an assignee, `type:*` and `area:*`
labels, a milestone (or an inherited, deliberate waiver), and a closing reference to an issue —
and every referenced issue must be a tracked Project member with all nine fields populated. The
script is deliberately decoupled from GitHub Actions: it takes only plain CLI arguments, so it
runs identically from a local terminal for debugging, and its exit codes separate metadata
violations (1) from unreadable data (2) and quota conditions (3), so CI output can never make a
drained API pool look like a broken pull request.

Because Projects V2 has no REST API, the gate's board reads are GraphQL — drawn from one shared
5000-points/hour pool spanning interactive sessions, agents, and CI tokens. The tooling therefore
practices explicit quota stewardship: at most one full board snapshot per logical phase, targeted
single-item read-backs after every mutation, a free preflight quota check before any GraphQL
phase, and a `before/after/consumed` report on every run
([quota stewardship](../governance/github-metadata-automation.md#graphql-quota-stewardship)).

Population is automated where judgment is not required.
[`project-item-autofill.yml`](../../.github/workflows/project-item-autofill.yml) adds new items to
the board and mirrors taxonomy labels onto their matching fields via an explicit mapping table
([`project_label_mapping.py`](../../scripts/github/project_label_mapping.py)) — filling only unset
fields, never regressing a Status lane, and withholding conflicting derivations for maintainer
review. Workstream and Target Release are never inferred at all, for a stated reason: a
confidently wrong value reads as deliberate triage. A weekly read-only board-drift check backstops
the creation-time automation and cross-checks each item's milestone against its Target Release
via an enumerated coherence table
([repository hygiene](../governance/repository-hygiene.md#board-drift-detection)).

## Namespaced label taxonomy and the completed legacy migration

Labels follow a namespaced taxonomy — `type:`, `priority:`, `status:`, `area:`, plus contextual
`modernization:`, `portfolio:`, `risk:`, `size:`, and `dependency:` dimensions — with a
machine-readable source of truth in [`.github/labels.json`](../../.github/labels.json) (47
declared labels) and assignment rules in the
[label taxonomy](../governance/label-taxonomy.md). Since the creation-time board automation,
assigning these labels *is* the reviewable judgment step; the board transcription is mechanical.

The taxonomy was retrofitted onto a live repository, and that migration is a governance artifact
in its own right: every pre-taxonomy spelling (`type:governance`, `area:notebooks`,
`priority:p4`, and the rest) was reviewed per-label, relabeled to its canonical successor on every
carrying item, and deleted (one label, `modernization:ux`, was renamed in place), with the full
decision table preserved as a historical record — a
record, not a menu; legacy spellings must never be re-minted
([completed migration](../governance/label-taxonomy.md#completed-legacy-label-migration-105-113)).
A weekly read-only drift check compares applied labels against the manifest so the taxonomy cannot
silently decay again ([label drift detection](../governance/repository-hygiene.md#label-drift-detection)).
Where the label set and the board's option sets deliberately diverge, a two-way alignment audit
recorded a per-label decision for every gap
([alignment audit](../governance/label-taxonomy.md#label-to-board-field-alignment-237)).

## Per-PR changelog enforcement

The changelog is prepared continuously, not reconstructed at release time: every pull request that
touches a substantive path (`src/`, `scripts/`, `docs/`, `configs/`, `.github/workflows/`) must
update [`CHANGELOG.md`](../../CHANGELOG.md) in that same pull request. A second, independent job
in the same workflow runs
[`validate_changelog_update.py`](../../scripts/github/validate_changelog_update.py) to enforce it
mechanically. A PR that genuinely needs no entry must say so visibly with a
`changelog: not-needed -- <reason>` line in its body — an audited exemption, not a silent bypass —
and marker text quoted inside code blocks is ignored, so a pull request documenting the mechanism
cannot exempt itself. The gate is REST-only by design: it needs no Project-scoped token and spends
nothing from the shared GraphQL pool
([per-PR changelog gate](../governance/github-metadata-automation.md#per-pr-changelog-gate)).

## Protected-test benchmark governance

The indexed `test` partition is never opened by supported code paths: the supported evaluator
reads validation shards only, and any protected-test use is governed by a machine-readable policy
(`configs/benchmark-policy-v1.toml`) plus [benchmark governance](../benchmark-governance.md).
Execution requires a frozen, immutable candidate; a named benchmark owner's recorded approval,
verified against the run manifest's lineage by `record-benchmark-approval` (which fails closed and
never opens a test shard); and exactly one scoring run per governed execution, with reruns limited
to documented infrastructure failures or verified implementation defects — never a retry of a
disappointing result. One governed execution has been completed under this policy; its bounded
result and required disclosures are documented as posture in
[held-out evaluation](../held-out-evaluation.md), not quoted as a benchmark. Historical numbers
from the 2022 notebook are likewise flagged, not celebrated: the archived split could place
windows from the same subject in training, validation, and test alike, and
[historical results](../historical-results.md) records why those metrics may be inflated.

The guard is also governance-as-code: a weekly check
([`check_held_out_trigger_safety.py`](../../scripts/check_held_out_trigger_safety.py)) parses
every workflow file and fails if anything matching a held-out/benchmark naming convention could
trigger on a routine `push` or `pull_request` — the checker permits nothing beyond manual dispatch
or `release-*` tag pushes, and the actual held-out workflow triggers on `workflow_dispatch` alone.

## The bot-author exemption class: automation, not exemption

Dependabot pull requests structurally cannot satisfy the human-oriented gates — they close no
issue, carry no milestone, and never write a changelog entry. The chosen policy is automation
rather than waiver
([bot-authored pull requests](../governance/github-metadata-automation.md#bot-authored-dependabot-pull-requests)).
A governed workflow ([`dependabot-autofill.yml`](../../.github/workflows/dependabot-autofill.yml))
auto-generates the changelog entry on the bot's own branch, so the entry still exists in the diff
the gate checks, and companion automation populates the bot PR's own board metadata. The metadata
gate waives only the closing-reference and milestone checks for governed bot authors, and that
waiver is compensated: the validator instead enforces the pull request's **own** Project
membership and all nine planning fields, so a bot PR faces the same complete-metadata bar, sourced
differently. Because this is the repository's only `pull_request_target` workflow, its security
model is load-bearing — identity gates on the immutable event payload, base-ref-only checkout,
allowlisted entry content, per-commit server-side authorship verification, and a single confined
write path — and a checker script
([`check_privileged_workflow_safety.py`](../../scripts/check_privileged_workflow_safety.py))
re-verifies those structural invariants so a future edit cannot quietly weaken them. Auto-merge
was considered and deferred: every bot PR still requires the maintainer's explicit merge click.

## Where to go deeper

- [Governance documentation index](../governance/index.md) — the full policy set
- [GitHub Project governance](../governance/github-project.md) — fields, lifecycle, views, automation
- [GitHub metadata automation](../governance/github-metadata-automation.md) — the gates, quota stewardship, and bot policy
- [Label taxonomy](../governance/label-taxonomy.md) — dimensions, alignment audit, migration record
- [Repository governance](../governance/repository-governance.md) — branch protection and review model
- [Repository hygiene automation](../governance/repository-hygiene.md) — drift detection and declined automation
- [Release governance](../governance/releases.md) and [versioning policy](../governance/versioning.md)
- [Benchmark governance](../benchmark-governance.md) and [held-out evaluation](../held-out-evaluation.md)
- [AGENTS.md](../../AGENTS.md) — the agent operating contract these gates enforce
