# AI Review Bot — Design

> **The middle tier.** A thorough, deterministic-prompt reviewer that runs on
> every PR, checks it against project conventions, cross-platform rules, scope
> discipline, regression risk, and **doc-sync**, then posts a structured comment.
> It does not block merge on its own verdict — the deterministic gates and humans
> remain the authority — but it raises the floor so humans review *less, better*.

---

## 1. Reuse what already works

The repo already runs a GitHub Models workflow, `pr-desc.yml`, that:

- triggers on `pull_request_target` (labeled `ai-describe`),
- builds context with `.github/models/pr-desc/build_context.js` from a diff +
  per-module specs in `.github/models/pr-desc/context/modules/*.md`,
- composes a prompt from `prompt.md` + `rules.json` (module weighting),
- calls the model with `permissions: models: read`,
- sanitizes the output and posts it as a PR comment (never edits the PR body).

**The AI review bot is the same machine with a review-oriented prompt and a
verdict.** We mirror the structure under `.github/models/ai-review/` so the two
share maintenance patterns and the module knowledge base.

---

## 2. Proposed layout

```
.github/
  workflows/ai-review.yml                 # mirrors pr-desc.yml; trigger + post comment
  models/ai-review/
    prompt.md                             # the review prompt template (below)
    build_context.js                      # reuse/extend pr-desc's builder
    rules.json                            # module weighting + arch-path patterns
    context/                              # SHARED knowledge base (see §5)
      overview.md  →  links to docs/architecture/
      modules/*.md →  kept consistent with docs/architecture/02-components/
```

---

## 3. What the bot checks (its rubric)

Derived from `AGENTS.md`, `architecture-and-ci.md`, the `review-for-nntr` skill,
and this architecture tree. The prompt instructs the model to report findings in
these dimensions, each tied to citable invariants/rules:

1. **Scope discipline** — diff limited to the stated feature/fix; no drive-by
   refactors, formatting churn, renames, dead-code removal (`AGENTS.md` Patch Scope).
2. **Cross-platform correctness** — integer width/signedness, pointer size, file
   handles, paths, line endings, compiler extensions, host-only commands; narrow
   `#if` only ([cross-platform](../03-crosscutting/cross-platform.md)).
3. **Architecture-invariant conformance** — dispatch backbone rules
   (INV-BCK/TEN/CONT): default-throw not pure-virtual, single init path,
   ContextData stamping, `supports_*()` fallback, no `#ifdef` at call sites.
4. **Test presence (TDD)** — new/changed behavior ships with a unit test at the
   right layer ([testing-strategy](testing-strategy.md) §3).
5. **Doc-sync** — if architecture-significant paths changed, did
   `docs/architecture/` change too? ([doc-maintenance-contract](doc-maintenance-contract.md)).
6. **CI hygiene** — clang-format (changed lines), doxygen tags on new public
   files/APIs, cppcheck themes, spelling/prohibited words, exec bits, newlines.
7. **Commit hygiene** — signed-off, body ≥ 16 words, meaningful subject+body,
   author = the user, **no AI/model attribution** anywhere.
8. **Regression risk** — call out hot spots (tensor/backend/graph) and the most
   likely break the diff could introduce on a platform we can't run locally.

---

## 4. Output format (proposed `prompt.md` `[OUTPUT FORMAT]`)

Mirror the `pr-desc` discipline (fixed sections, backticks sanitized to
`<code>`), English, no model names in the body:

```
### Verdict
<approve | comment | change-required>  — one-line rationale.

### Findings
- [dimension] severity(blocker|major|minor|nit): file:line — issue → suggested fix.
  (cite the invariant/rule, e.g. INV-BCK-1, AGENTS:scope)

### Cross-platform notes
<gates this likely affects that CANNOT be run locally; what to watch>

### Doc-sync
<arch paths touched? matching docs/architecture/ updated? if not, which doc>

### Tests
<is new behavior covered? which test file; gaps>
```

The verdict is **advisory** — posted as a comment, not a merge block. Authority to
block stays with the deterministic required checks (§ci-topology 3.1) and humans.

---

## 5. The shared knowledge base (why this tree matters to the bot)

The bot is only as good as its context. The `context/modules/*.md` specs are
exactly the L3 component view in different words. The plan:

- **Single source of truth:** `docs/architecture/02-components/*.md` and
  `03-crosscutting/*.md` become the canonical descriptions.
- **Generate or sync** the `.github/models/*/context/` specs from them (a small
  script, or a CI check that they don't drift). This means **updating an
  architecture doc automatically improves both human and AI review**, closing the
  loop the user asked for.
- `rules.json` gains an `arch_paths` section listing the architecture-significant
  patterns the doc-sync check and bot use.

---

## 6. Guardrails

- **`pull_request_target` runs with secrets** — it must check out the PR head
  (`pr-desc.yml` already does) and **never execute PR code**; it only reads the
  diff + repo context. Keep it build-free.
- **`permissions: models: read` + `pull-requests: write`** only; least privilege.
- **No auto-merge / no auto-approve.** The bot comments; it does not gate.
- **Deterministic prompt, pinned model** for reproducible reviews; version the
  prompt so changes are reviewable.
- **No model/vendor name** in any posted comment (project rule).

---

## 7. Rollout

Advisory comment first (low risk, learn from false positives) → tune prompt &
rules → optionally make *specific deterministic sub-checks* (doc-sync, commit
hygiene) hard gates in CI, while the model's holistic verdict stays advisory.
