---
inclusion: manual
---

# Dispatch seed-prompt template

Use this template (via `#seed-prompt-template`) when generating a dispatch
prompt to hand off a work item to a new Kiro session. The prompt provides
enough context that the receiving session can execute without re-reading the
full conversation history.

## Template

```markdown
# Dispatch: <one-line summary>

## Context

<2-3 sentences: what branch exists, what state it's in, link to handoff if any>

## Issue

<GitHub issue URL>

## What's done on disk

<bulleted list of files already created/modified, if any>

## Remaining work

<numbered steps — the canonical workflow from AGENTS.md, starting from wherever
the previous session left off>

## Constraints

- Fish shell on macOS
- `gh` authenticates via keychain (no tokens in env)
- <any task-specific constraints: files to avoid staging, etc.>
- Follow AGENTS.md canonical workflow
```

## Usage notes

- Include only facts verified in the session that generated the prompt
- Mark anything unverified with "(unverified)" or "(relayed)"
- The dispatch should be self-contained: a cold-start session with no memory
  must be able to execute from it
- Reference the handoff file path if one was written
