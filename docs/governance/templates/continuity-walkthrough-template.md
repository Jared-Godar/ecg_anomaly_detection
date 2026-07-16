# Continuity-walkthrough template

This is the copy-and-fill source for the **proactive continuity walkthrough** required by the
`AGENTS.md` standing commitments (issue #251): a fill-in-the-rails document every
implementation session writes **immediately after branching**, so the maintainer can finish
the canonical PR workflow by hand if the session dies at any point. It generalizes the #246
exemplar walkthrough that the maintainer executed end-to-end, solo, as PR #249.

How to use it:

- Copy everything below the cutter line into
  `artifacts/walkthroughs/<UTC-timestamp>-issue-<n>-<slug>.md` (gitignored; create the
  directory if absent). A worktree-isolated session writes to the worktree's own
  `artifacts/walkthroughs/` first, then copies the file into the primary checkout's
  `artifacts/walkthroughs/` in the same turn — the same copy-out flow as session handoffs.
- Replace every `⟨SLOT⟩` you can at write time; leave genuinely-unknown values (PR number,
  merge SHA) as slots and fill them at the two refresh checkpoints: **PR opened** and
  **awaiting merge**.
- Resolve the board/field IDs from the canonical table in
  [`docs/governance/github-project.md`](../github-project.md) ("Setting Status via the CLI")
  at fill time — IDs are deliberately not frozen here because a field-option regeneration
  incident (#213) invalidated every previously recorded ID at once.
- Rails only: numbered mechanical steps, each action followed by its verification command.
  Work-in-progress narrative belongs to the wind-down session handoff, which links the
  walkthrough instead of repeating commands.
- Every command block is **Fish** and must run from the repository root. Fish has no
  heredocs — write multi-line files with `printf '%s\n' …`. Never include secrets or live
  tokens.

---

<!-- Everything below the cutter is the copy-and-fill portion: a second, standalone document
     whose own top-level heading is intentional (it becomes the filled walkthrough's title).
     Only markdownlint's single-h1 rule is waived, for that one line. -->
<!-- markdownlint-disable-next-line MD025 -->
# Walkthrough: #⟨ISSUE-NUMBER⟩ — ⟨ISSUE TITLE⟩

A start-to-finish, do-it-yourself guide through the canonical PR workflow for issue
[#⟨ISSUE-NUMBER⟩](https://github.com/Jared-Godar/ecg_anomaly_detection/issues/⟨ISSUE-NUMBER⟩).
Every command block is Fish, copy-pasteable, and runs from the repository root
(`~/Developer/portfolio/ecg_anomaly_detection`). Each action step is followed by its
verification.

**Written:** ⟨UTC-TIMESTAMP⟩ at the ⟨post-branch / PR-open / awaiting-merge⟩ checkpoint.
**Preconditions:** ⟨e.g. "#⟨N⟩ must be merged first" — or "none"⟩

Board/ID quick reference (copy the live values from the canonical table in
`docs/governance/github-project.md` — re-verify there if any command errors):

```text
project_id            ⟨PROJECT-ID⟩
Status field          ⟨STATUS-FIELD-ID⟩
  In Progress ⟨ID⟩ · Review ⟨ID⟩ · Merged ⟨ID⟩ · Closed ⟨ID⟩
Workstream field      ⟨WORKSTREAM-FIELD-ID⟩   (⟨CHOSEN-OPTION⟩ ⟨ID⟩)
Target Release field  ⟨TARGET-RELEASE-FIELD-ID⟩   (⟨CHOSEN-OPTION⟩ ⟨ID⟩)
Portfolio Signal fld  ⟨SIGNAL-FIELD-ID⟩   (⟨CHOSEN-OPTION⟩ ⟨ID⟩ — omit if label-derivable)
issue #⟨N⟩ item id     ⟨ITEM-ID⟩
```

## Step 0 — confirm preconditions and mark the issue started

⟨Precondition check command(s), or delete this block if none:⟩

```fish
gh issue view ⟨PRECONDITION-ISSUE⟩ --json state,title -q '"\(.state)  \(.title)"'
```

Label swap + board lane (skip whichever already happened):

```fish
gh issue edit ⟨ISSUE-NUMBER⟩ --remove-label "⟨status: triage / status: ready / status: blocked⟩" --add-label "status: in-progress"
gh project item-edit --id ⟨ITEM-ID⟩ \
  --field-id ⟨STATUS-FIELD-ID⟩ \
  --project-id ⟨PROJECT-ID⟩ --single-select-option-id ⟨IN-PROGRESS-OPTION-ID⟩
```

Verify:

```fish
gh issue view ⟨ISSUE-NUMBER⟩ --json labels -q '[.labels[].name] | join(", ")'
```

Expected: contains `status: in-progress`, no other `status:*` label.

## Step 1 — sync main (never skip; parallel merges happen)

```fish
git checkout main
git fetch origin
git pull --ff-only origin main
```

Verify — clean tree, local main matches origin:

```fish
git status --short --branch
git log -1 --oneline
```

Expected: `## main...origin/main` with nothing listed below it.

## Step 2 — branch first (commits to main are hook-blocked anyway)

```fish
git switch -c ⟨BRANCH-NAME⟩
```

Verify:

```fish
git branch --show-current
```

Expected: `⟨BRANCH-NAME⟩`.

## Step 3 — the actual work

⟨Fill in the work items as a checklist: files to create/edit, decisions already made, the
acceptance criteria from the issue body. Keep it mechanical — what to change, not why.⟩

-

**If you end up changing nothing** (review-only work items): skip Steps 4–11 entirely,
record the outcome, and close out —

```fish
gh issue close ⟨ISSUE-NUMBER⟩ --comment "⟨No-changes outcome, dated⟩"
gh issue edit ⟨ISSUE-NUMBER⟩ --remove-label "status: in-progress"
gh project item-edit --id ⟨ITEM-ID⟩ \
  --field-id ⟨STATUS-FIELD-ID⟩ \
  --project-id ⟨PROJECT-ID⟩ --single-select-option-id ⟨CLOSED-OPTION-ID⟩
git checkout main; and git branch -d ⟨BRANCH-NAME⟩
```

Then jump to Step 12.

## Step 4 — CHANGELOG entry (gate-tier, every substantive PR)

Add a bullet under `## Unreleased` → `### ⟨SUBSECTION⟩` (create the subsection under
Unreleased if it's absent):

```fish
code CHANGELOG.md
```

Suggested text (adjust to what you actually changed):

```text
- ⟨One entry describing the change, ending with (#⟨ISSUE-NUMBER⟩).⟩
```

## Step 5 — run the gates locally before committing

```fish
uv run pre-commit run --all-files
uv run pytest
uv run pyright
```

Expected: all hooks Passed/Skipped; `⟨CURRENT-COUNT⟩ passed`; `0 errors`.

## Step 6 — commit

```fish
git add -A
git status --short
git commit -m "⟨type(scope): subject (#ISSUE-NUMBER)⟩"
```

Verify (pre-commit hooks run automatically at commit; a hook edit means re-add and retry):

```fish
git log -1 --stat
```

## Step 7 — push (gated: maintainer go-ahead) — 🔴 HOLD begins here

```fish
git push -u origin ⟨BRANCH-NAME⟩
```

Verify: the output shows the new remote branch and a `Create a pull request` URL.

From this push until Step 9's green light, the branch is under an explicit **HOLD**: the
GitHub GUI will show "all checks passed" and an enabled squash-merge button while
verification is still in flight — GUI eligibility is never the authoritative merge signal.

## Step 8 — open the PR (assignee at open — the strict gate fails without it)

Write the PR body to a temp file (Fish has no heredocs; `printf` one line per argument):

```fish
printf '%s\n' \
  '## Summary' \
  '' \
  '⟨One-paragraph summary.⟩' \
  '' \
  '⟨Closes #ISSUE-NUMBER — or the sanctioned non-closing marker: Non-closing ref: #N — reason⟩' \
  > /tmp/pr-body-⟨ISSUE-NUMBER⟩.md
gh pr create \
  --title "⟨type(scope): subject (#ISSUE-NUMBER)⟩" \
  --body-file /tmp/pr-body-⟨ISSUE-NUMBER⟩.md \
  --assignee Jared-Godar \
  ⟨--label "…" for each label, mirroring the issue's taxonomy labels⟩ \
  ⟨--milestone "…" — or omit and disclose "deliberately unmilestoned" in the PR body⟩
```

Record the PR number: **PR #⟨PR-NUMBER⟩** (fill at the PR-open refresh).

## Step 9 — verify the PR yourself, then announce the green light

CI checks (metadata gate included):

```fish
gh pr checks ⟨PR-NUMBER⟩ --watch
```

Expected: all green. A cancelled duplicate metadata-governance run can wedge the merge box
even after a real pass — if that happens, re-run it:

```fish
gh run rerun --failed
```

Board: the creation-time autofill workflow populates the PR's label-derived fields on open
(currently at the tier-1→2 boundary of the
[verification graduation ladder](../github-metadata-automation.md#automation-verification-graduation-ladder-issue-248),
so read the derivable fields back rather than assuming). Grab the PR's item id, then set the
human-owned fields — Status→Review, Workstream, Target Release, plus Portfolio Signal /
Repository Area whenever the PR's labels are among the permanently human-set ones (#237):

```fish
set ITEM_ID (gh api graphql -f query='{ repository(owner:"Jared-Godar", name:"ecg_anomaly_detection") { pullRequest(number:⟨PR-NUMBER⟩) { projectItems(first:5) { nodes { id project { number } } } } } }' --jq '.data.repository.pullRequest.projectItems.nodes[] | select(.project.number == 5) | .id')
echo "PR item: $ITEM_ID"
gh project item-edit --id $ITEM_ID --field-id ⟨STATUS-FIELD-ID⟩ \
  --project-id ⟨PROJECT-ID⟩ --single-select-option-id ⟨REVIEW-OPTION-ID⟩
gh project item-edit --id $ITEM_ID --field-id ⟨WORKSTREAM-FIELD-ID⟩ \
  --project-id ⟨PROJECT-ID⟩ --single-select-option-id ⟨WORKSTREAM-OPTION-ID⟩
gh project item-edit --id $ITEM_ID --field-id ⟨TARGET-RELEASE-FIELD-ID⟩ \
  --project-id ⟨PROJECT-ID⟩ --single-select-option-id ⟨TARGET-OPTION-ID⟩
```

Read back all nine fields (never trust a silent success):

```fish
gh api graphql -f query='query { node(id: "'$ITEM_ID'") { ... on ProjectV2Item {
  fieldValues(first: 20) { nodes { ... on ProjectV2ItemFieldSingleSelectValue {
    field { ... on ProjectV2SingleSelectField { name } } name } } } } } }' \
  --jq '[.data.node.fieldValues.nodes[] | select(.name != null) | {(.field.name): .name}] | add'
```

Expected: nine fields, Status=Review. If any `gh project item-edit` printed
`error: no changes to make` on a genuinely unset field, retry that single field once — the
known intermittent CLI quirk resolves on individual retry.

Run the metadata gate the same way CI does:

```fish
uv run python scripts/github/validate_project_metadata.py --pr-number ⟨PR-NUMBER⟩ --strict-project-checks
```

**When — and only when — everything above is green:** announce, unprompted,
**🟢 GREEN LIGHT: clear to squash-merge PR #⟨PR-NUMBER⟩ via the GUI.** Until that
announcement the merge is under HOLD, whatever the GUI shows. Merge-independent work
(walkthrough refresh, extract drafting) must never delay the green light — note that it
continues after the merge instead.

## Step 10 — merge (maintainer, GUI, squash — on the green light)

Merge via the GitHub web UI — squash merge (the only enabled strategy). The remote branch
auto-deletes.

Verify from the terminal:

```fish
gh pr view ⟨PR-NUMBER⟩ --json state,mergeCommit --jq '"\(.state)  \(.mergeCommit.oid[0:7])"'
```

Expected: `MERGED  ⟨short SHA⟩` (fill at the awaiting-merge refresh: **merge SHA ⟨SHA⟩**).

## Step 11 — closure pass

Pull main and confirm the squash commit arrived:

```fish
git checkout main
git pull --ff-only origin main
git log -2 --oneline
```

A `Closes #N` reference auto-closed the issue on merge, but two things never happen
automatically — strip the status label, and set the **issue's** board lane to Closed (the
PR lane goes to Merged on its own via `project-status-sync`, ladder tier 2 — spot-check its
run conclusion; the issue lane does not move itself — the #100 quirk, manual at every tier):

```fish
gh run list --workflow project-status-sync.yml --limit 1
gh issue edit ⟨ISSUE-NUMBER⟩ --remove-label "status: in-progress"
gh project item-edit --id ⟨ITEM-ID⟩ \
  --field-id ⟨STATUS-FIELD-ID⟩ \
  --project-id ⟨PROJECT-ID⟩ --single-select-option-id ⟨CLOSED-OPTION-ID⟩
```

Read back:

```fish
gh issue view ⟨ISSUE-NUMBER⟩ --json state,labels -q '"\(.state)  \([.labels[].name] | join(", "))"'
```

Expected: `CLOSED` with no `status:*` label. Delete the local branch (`-D`: squash merges
break `-d`'s ancestry check) and any worktree — after copying its `artifacts/` handoffs and
walkthroughs into the primary checkout:

```fish
git branch -D ⟨BRANCH-NAME⟩
```

## Step 12 — final read-back and report

```fish
gh issue list --milestone "⟨MILESTONE⟩" --state open --json number,title -q '.[] | "#\(.number) \(.title)"'
```

⟨Or, for unmilestoned work, whatever end-state read proves the item is fully closed.⟩
Paste the output plus the merge SHA into the PM thread — that is the extract's core.

## Watch-items

- Anything that fails in `gh pr checks`: read the failing job's log before retrying —
  first CI failures on this repo have historically been metadata (missing assignee), not
  content.
- GraphQL quota is a shared 5k/hr pool (you + any agent sessions + the status-sync PAT).
  Prefer the targeted read-back query above over repeated full `gh project item-list`
  discovery scans.
- ⟨Task-specific risks: date-sensitive stamps, parallel sessions on the same surface, …⟩
- No secrets appear anywhere above; nothing here needs elevated tokens beyond the existing
  `gh` auth.
