# Repository hygiene automation

Lightweight, deterministic automation for repository maintenance, scoped to what
is actually justified by this repository's real activity rather than every
capability a generic "repo hygiene" checklist might suggest.

## Label drift detection

`scripts/detect_label_drift.py` compares the labels applied to open issues and
pull requests against the canonical set in
[`.github/labels.json`](../../.github/labels.json), and reports any applied
label that is not in that manifest. It is read-only: it never creates, renames,
deletes, or applies a label, and never guesses a corrected name. Remediation
(migrating a legacy label with `gh label edit`, or adding a genuinely-missing
label to the manifest) remains the maintainer decision described in
[label taxonomy § Existing-label normalization](label-taxonomy.md#existing-label-normalization).

```fish
uv run python scripts/detect_label_drift.py
```

Checks open issues and pull requests by default; pass `--include-closed` to
audit the full historical set instead. `.github/workflows/repository-hygiene.yml`
runs this weekly (Monday 06:00 UTC) and on manual dispatch, using the default
`GITHUB_TOKEN` — reading issue, PR, and label data needs no elevated
permissions, unlike the Project V2 field checks in
[GitHub metadata automation](github-metadata-automation.md).

A label can match the canonical *formatting* (a space after the colon) while
still not being in the manifest — the check compares against the actual
manifest set, not just a style pattern. Running this for the first time
against this repository's real state surfaced 40 labels in use that are not
declared in `.github/labels.json`, including several new-format labels in
active current use (`area: cli`, `area: pipeline`, `area: ci-cd`,
`area: validation`, `area: data`, `portfolio: operational-maturity`,
`risk: low`) alongside pre-taxonomy legacy labels and unused GitHub default
labels. That is a manifest-completeness question distinct from this tool's
job of flagging drift against whatever the manifest currently says, and is
tracked separately rather than resolved here.

## Stale issue and pull request handling: declined

The originating issue for this automation framed it as conditional: add
automation "if justified by repository activity." Stale issue/PR bots
(auto-labeling or auto-closing items inactive for N days) are not implemented,
because this repository's actual observed activity does not justify one:

- Single maintainer, no external-contributor backlog to triage.
- Issues and pull requests in this repository's real history are opened and
  closed same-day or within a few days; there is no accumulation of
  long-dormant open items for a stale bot to act on.
- An intentionally-open item can be a deliberate state, not neglect — for
  example an issue held open pending a documented maintainer decision. A
  stale bot cannot distinguish "forgotten" from "deliberately pending" and
  would risk auto-closing the latter.

If repository activity changes (multiple contributors, a growing backlog of
genuinely dormant items), this is worth revisiting — but adding it now would
be automation solving a problem this repository does not currently have.
