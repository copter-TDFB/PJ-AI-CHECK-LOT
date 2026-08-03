# Agent Instructions

Before doing any work in this repository, read `CLAUDE.md` first

Use `CLAUDE.md` as the main project onboarding document.

After reading it, inspect only the files relevant to the user's current request. Do not revert existing user changes unless the user explicitly asks for that.

<!-- pm-bridge contract: begin contract-version=2 -->
## pm-bridge contract

This repo is managed via a Jira board (project KAN) by a PM agent. When you are
dispatched with a task packet (a `# KAN-<n>:` prompt), the packet is your work
order — follow it exactly. Whether dispatched or run manually:

**Git rules**
- **Do not run git write commands.** No `add`, `commit`, `stash`, `rebase`,
  `checkout`, `branch`, `merge`, `push`, or `tag`. Leave your work as uncommitted
  changes in the working tree — the PM commits and pushes it (on a manual run,
  the human does). Read-only git (`status`, `diff`, `log`, `show`) is fine and
  encouraged. Sandboxed executors cannot write to `.git` at all, so an attempted
  commit only wastes a turn and can leave a stale `index.lock` behind.
- Never work on `main`/`master`. You are already placed on the correct branch
  before you start (`ai/KAN-<n>-<slug>` when dispatched) — do not switch it.
- Never touch gitignored secrets (`gcp-key.json`, `client_secret*`, `.env*`) or `.pm-bridge/`.
  Your packet lives in `.pm-bridge/`, including its verification commands — editing it is
  not a shortcut to passing, because the PM holds the authoritative copy outside the
  worktree, compares the two, and checks every changed path against the packet's
  Boundaries after each round.

**Working rules (always apply)**
1. State assumptions explicitly in your final output. Ambiguity with no
   user-visible behavior or data impact: choose the conservative
   interpretation and flag it. Ambiguity that would change user-visible
   behavior, UI, or data: do NOT implement — stop and make
   `BLOCKED: <question>` the very LAST line of your report, with nothing
   after it. Only that final line is read as your question.
2. Minimum code that solves the task — no speculative features or abstractions.
3. Surgical changes only: every changed line traces to the task; match existing
   style; don't "improve" unrelated code.
4. Define success as the task's verification commands passing — run them and
   loop until green before reporting done.

**Definition of done**
- All acceptance criteria met and verification commands pass (paste output).
- Report: what changed · assumptions · files touched · anything noticed but left alone.
<!-- pm-bridge contract: end -->
