# Human Review Playbook

> **The top tier.** As change volume and PR size grow, human reviewers stop
> reading every line and instead review the **architecture delta**: did the shape
> of the system change, and is that change correct? Low-level correctness is
> already covered by TDD + CI + the AI bot.

---

## 1. What a human reviewer is responsible for

- **Architecture deltas.** New subsystems, new dependency edges (especially
  upward — INV-CONT-3), a new dispatch tier or backend, changed invariants,
  changed public surface (INV-CTX-3).
- **Intent vs implementation.** Does the change actually solve the stated problem,
  and is this the right place/shape to solve it?
- **Invariant changes.** Any PR that modifies a numbered `INV-*` is a design
  decision needing judgment + an [ADR](../adr/).
- **Tradeoffs the tests can't express.** API ergonomics, on-device budget
  implications, maintainability, future-proofing.

## 2. What a human reviewer does NOT need to do

- Re-derive low-level correctness already proven by unit tests → trust the suite
  (and check a test exists — the AI bot flags missing ones).
- Hand-check formatting, doxygen, spelling, commit hygiene → deterministic gates.
- Mentally simulate cross-platform builds → that's the CI matrix's job; read the
  bot's cross-platform notes instead.
- Read every line of a large mechanical diff → spot-check + trust the gates.

This is the point of the pyramid: **human attention is the scarcest resource, so
spend it where only humans add value.**

---

## 3. The review flow

1. **Read the architecture-doc diff first.** A well-formed PR updates
   `docs/architecture/` when it changes the system's shape. That diff is the
   high-level summary of intent. If architecture paths changed but no doc did →
   ask for the doc (the contract); the doc-sync gate should have caught it.
2. **Read the PR description** (the `pr-desc` bot's "What & Why" + pointers).
3. **Read the AI review bot's findings.** Triage: confirm/dismiss each, focusing
   your own attention on blockers/majors and anything the bot was unsure about.
4. **Check the gate status.** Required checks green? Coverage delta sane?
5. **Judge the delta.** Is the new shape right? New edges justified? Invariants
   intact or properly changed (with ADR)? Public surface stable or intentionally
   versioned?
6. **Decide:** approve / request changes on *architecture/intent* grounds. Defer
   low-level nits to the bot and gates unless they're genuinely important.

---

## 4. Escalation triggers (stop and look closely)

- L1/L2 changes (context, subsystem map, dispatch backbone) — always.
- Any `INV-*` modified or removed.
- Public C/C++/CAPI/JNI signature change.
- A new external dependency or build option.
- A new upward dependency edge between subsystems.
- The AI bot flagged a blocker the author dismissed without a convincing reason.

For any escalation trigger, expect an [ADR](../adr/) and treat it as a design
review, not a code review.

---

## 5. Receiving and giving feedback

- Reviewers: be specific, cite the invariant/doc, distinguish blocker vs nit.
- Authors: verify feedback technically before implementing (don't blindly comply,
  don't dismiss) — the `superpowers:receiving-code-review` discipline applies.
- Disagreements about an invariant are resolved by an ADR, not by debate in
  comments.
