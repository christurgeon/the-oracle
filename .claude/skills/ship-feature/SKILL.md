---
name: ship-feature
description: End-to-end feature development orchestrator that drives a single feature from vague intent through goal definition, concept exploration, spec hardening, implementation planning, subagent-driven execution, and final holistic review. Use whenever the user wants to "build a feature", "add a feature", "ship X", "implement X end-to-end", starts a request with `/ship-feature`, says "let's start a new feature", or otherwise indicates a structured multi-phase workflow rather than a one-off change. Use even when the request sounds simple — if it involves designing something new (not just patching existing behavior), this enforces clarifying intent before coding. Delegates to Superpowers skills (brainstorming, write-plan, execute-plan, test-driven-development, requesting-code-review, finishing-a-development-branch) where they fit, and adds the cross-phase gating, artifact chain, and goal-anchored final review they don't individually provide.
---

# Ship Feature

A lifecycle orchestrator for taking a feature from "vague idea" to "merged in main" without losing intent along the way.

**Prerequisite:** the Superpowers plugin for Claude Code. Each phase below delegates to one of its skills. Install the plugin before you use this skill.

## Philosophy

The default failure mode of AI-assisted feature work is jumping to implementation before the goal is understood, then drifting from the goal as tasks accumulate. This skill exists to prevent that. It enforces a strict phase gate: each phase produces a canonical artifact that the next phase consumes, and you do not advance until the current artifact is locked.

You are the orchestrator. You delegate execution work to subagents. You never lose sight of the original user intent. When in doubt, ask — premature confidence is more expensive than a few extra questions.

## Uncertainty and escalation

Two failure modes accumulate silently across a feature: hidden assumptions and unflagged ambiguity. Both compound. The skill addresses them in three ways.

**Surface assumptions in the artifacts that already exist.** Every implementation depends on things the spec didn't verify — API rate limits, timezone behavior, the assumption that existing middleware will compose cleanly with new code. The `spec.md` template includes an Assumptions section for this reason. When a subagent discovers a new assumption mid-execution, it appends it to the spec and the orchestrator confirms it before continuing. Do not maintain a separate `assumptions.md` — standalone uncertainty logs rot fast.

**Tag reversibility per task.** For each task in `plan.md`, mark reversibility as:
- `easy` — refactor later, no external commitment, no data implication
- `costly` — data migration, internal API change consumers depend on, infrastructure change
- `one-way` — data loss possible, external contract published, irreversible deployment

One-way tasks require explicit user confirmation before the implementation subagent is spawned. They are not appropriate places to guess.

**Pause and escalate to the human when any of these trigger:**

- A task as planned contradicts another task or spec requirement
- A subagent flags it is guessing on a critical-path decision
- An implementation introduces a new auth boundary, secret, or external trust relationship
- A schema migration could lose data or break a deployed contract
- The work crosses a one-way reversibility door not flagged in the plan
- Multiple acceptable approaches exist and the tradeoff is value-laden, not technical (UX, business logic, naming)
- Two consecutive task pre-flight reviews surface the same architectural concern (deeper issue surfacing)
- An assumption discovered mid-execution invalidates work already completed

When triggered: stop. State the trigger, summarize the options, name the question, and wait. Do not produce a best guess and continue. Fake certainty is the most expensive failure mode in this lifecycle — drift you can detect; bad commitments shipped under false confidence you usually cannot.

**Confidence is not a Likert scale.** Do not ask subagents to rate confidence 1–5; the signal degrades to noise within a handful of tasks. Instead, require explicit flagging: a subagent must call out anything it is guessing about and anywhere it chose between non-obvious alternatives without an obvious winner. Silence implies the work was straightforward. Unexpected silence on complex work is itself a warning sign worth investigating.

## Complexity check

Features accrete abstractions, dependencies, and operational surface area faster than they accrete value. The skill counters this with a qualitative check at two gates — not a metrics dashboard, which is gameable and produces noise without a reference point.

**Gate 1: end of PLAN, before any code is written.** Read `goal.md` and `plan.md` side by side and answer three questions:

1. Is the planned scope still the smallest version that meets `goal.md`'s success criteria? If the plan has grown past what the goal requires, cut it back.
2. Does any task introduce an abstraction (generic interface, configurable strategy, plugin point) that has only one concrete consumer? If yes, inline it. Abstractions earn their keep at two consumers, not one.
3. Does any dependency, service, or piece of operational surface area exist in the plan that isn't required to meet a specific spec line? If yes, justify it explicitly or remove it.

**Gate 2: during EXECUTE, any time `plan.md` grows by more than ~25% from its size at end-of-PLAN.** Task additions during execution are a scope creep signal. When the threshold trips, pause and surface to the user: "The plan has grown from N tasks to M tasks. Here's what was added and why. Should we continue, simplify, or split the remainder into a follow-up feature?"

The check is qualitative against `goal.md`, not metrical. The discipline is asking the questions at predictable gates. Continuing with known bloat is the failure mode this check exists to prevent — once detected, the response is one of: simplify in place, defer scope to a follow-up feature, or revisit the spec.

## Lifecycle phases

```
GOAL → CONCEPT → SPEC → PLAN → EXECUTE (per task) → HOLISTIC REVIEW
```

Each phase has an entry condition, a single canonical artifact, and an exit gate. The artifact chain lives under `.feature/<slug>/` in the repo root, where `<slug>` is a short kebab-case name derived from the goal (confirm the slug with the user before creating the directory).

State is tracked in `.feature/<slug>/state.json`:

```json
{
  "slug": "kalshi-weather-bot",
  "phase": "spec",
  "completed_tasks": ["task-001", "task-002"],
  "current_task": "task-003",
  "started_at": "2026-05-15T10:00:00Z"
}
```

When invoked, first read `.feature/<slug>/state.json` if it exists and resume from the recorded phase. Otherwise start at GOAL.

---

### Phase 1: GOAL

**Purpose:** turn a vague request into a written goal that anyone could read and understand what's being built and why.

**Entry condition:** user has expressed intent to build something. The intent may be a single sentence ("I want to build a Kalshi weather bot") or a paragraph. Either way, treat it as incomplete.

**What you do:** ask clarifying questions, one focused batch at a time. Aim for 3–6 questions per round, no more than 2 rounds. Cover:

- What problem does this solve, and for whom?
- What does "done" look like? What's the smallest version that's still useful?
- What's explicitly out of scope?
- What are the hard constraints (latency, cost, compliance, deadlines)?
- What are the known unknowns?

Do not propose architecture or implementation in this phase. If the user volunteers implementation ideas, capture them as "user preferences" but defer decisions.

**Artifact:** `.feature/<slug>/goal.md`

```markdown
# Goal: <one-line summary>

## Problem
<what's broken / what's missing>

## Success criteria
<bulleted, observable, testable where possible>

## Non-goals
<explicit list of things we're not doing>

## Constraints
<hard limits: latency, cost, compliance, deadlines, dependencies>

## Open questions
<things we still don't know>

## User preferences captured
<implementation hints from user, to revisit in CONCEPT>
```

**Exit gate:** user explicitly confirms `goal.md`. Do not advance silently. Say "Here's the goal as I understand it. Anything to change before we move to concept exploration?"

---

### Phase 2: CONCEPT

**Purpose:** explore what the feature looks like at a system level before committing to specifics.

**Entry condition:** `goal.md` is confirmed.

**What you do:** invoke the Superpowers `brainstorming` skill. It will run a Socratic exchange to refine the concept. Feed it the contents of `goal.md` as the seed. Stay engaged throughout — the brainstorming skill drives the exchange, but you ensure the conversation stays anchored to the goal.

During this phase, explore at least two distinct approaches and articulate the tradeoffs. If only one approach surfaces, you have not brainstormed enough — push back and force an alternative, even if it's a strawman.

Cover: user-facing surface (CLI/API/UI), data model at a high level, key external dependencies, deployment shape, observability story, failure modes.

**Artifact:** `.feature/<slug>/concept.md`

```markdown
# Concept: <slug>

## Narrative
<2–4 paragraphs: how the feature works from the user's perspective, end to end>

## Approaches considered
### Approach A: <name>
- Sketch:
- Pros:
- Cons:

### Approach B: <name>
- Sketch:
- Pros:
- Cons:

## Recommended approach and why

## Major risks
<technical, product, operational>

## Dependencies
<systems, services, libraries, data sources this depends on>
```

**Exit gate:** user confirms the recommended approach. If they want to mix approaches or invent a new one, loop back through brainstorming — do not paper over disagreement.

---

### Phase 3: SPEC

**Purpose:** convert the concept into enforceable specifications that subagents cannot legitimately violate during implementation.

**Entry condition:** `concept.md` is confirmed.

**What you do:** write the spec yourself (do not delegate this — spec authorship is orchestrator work). Reference `goal.md` and `concept.md` continuously. The spec is the source of truth that every subsequent task is graded against.

The spec must be concrete enough that two competent engineers reading it independently would produce compatible implementations. If it isn't, it isn't done.

**Artifact:** `.feature/<slug>/spec.md`

```markdown
# Spec: <slug>

## Functional requirements
<numbered list of what the feature must do, each independently verifiable>

## Non-functional requirements
- Performance:
- Reliability:
- Security:
- Observability:
- Cost ceiling (if applicable):

## Interfaces and contracts
<API shapes, CLI surface, message schemas, DB schema changes — be concrete>

## Data model
<entities, fields, types, relationships, migrations needed>

## Edge cases
<numbered list — what happens when X is missing, malformed, concurrent, etc.>

## Failure modes and recovery
<for each: detection, response, rollback>

## Testing requirements
<what kinds of tests, coverage expectations, integration vs unit>

## Assumptions
<numbered list of things this spec depends on but has not verified. Each must be falsifiable. Examples: "Kalshi's REST API rate limits remain at current published values", "existing auth middleware composes with new endpoints without modification", "user's local timezone is acceptable for scheduling display". Subagents append to this list when they discover new assumptions during implementation.>

## Out of scope (carried from goal.md non-goals)
```

**Exit gate:** user reviews and confirms `spec.md`. Specifically ask: "Is there anything in this spec that would block you from accepting the feature if it were implemented exactly as written?" If yes, fix it now, not later.

---

### Phase 4: PLAN

**Purpose:** decompose the spec into a sequence of bounded, independently verifiable tasks.

**Entry condition:** `spec.md` is confirmed.

**What you do:** invoke the Superpowers `write-plan` skill, passing it `spec.md` as input. Stay engaged — review the generated plan against the spec. Look for:

- Tasks that are too large (any task that touches >5 files or >300 lines of net new code should probably split)
- Missing dependency ordering
- Tasks that don't tie back to a specific spec requirement
- Hidden coupling that should be broken into separate tasks

Reorder and split tasks until the plan looks like something a junior engineer could execute one task at a time without losing their place.

**Artifact:** `.feature/<slug>/plan.md`

Each task in the plan must include:
- Task ID and title
- Which spec requirement(s) it implements
- Files expected to be modified
- Test strategy
- Verification command(s)
- Definition of done
- **Reversibility: `easy` / `costly` / `one-way`** (see Uncertainty and escalation section). One-way tasks require explicit user confirmation before the implementation subagent is spawned.

**Exit gate:** the plan reads as a coherent sequence with no obvious gaps. Confirm with the user, then update `state.json` to phase `execute` with `current_task` set to task-001.

---

### Phase 5: EXECUTE (per task)

This phase runs in a loop, one task at a time, until all tasks in `plan.md` are complete.

**For each task:**

1. **Pre-flight review.** Spawn a subagent with the task description, `spec.md`, and the relevant section of `plan.md`. Its job is to review the task plan and flag anything that looks wrong before code is written. The subagent answers: does this task as written advance the spec? Are the verification commands sufficient? Are there edge cases the task ignores? Does this task hit any of the escalation triggers listed in the Uncertainty and escalation section?

   If the subagent flags concerns, address them before proceeding. Update `plan.md` if needed. If any escalation trigger fires — especially a `one-way` reversibility tag without prior user confirmation — stop and surface to the user before spawning the implementation subagent.

2. **Implementation.** Spawn the implementation subagent with:
   - The task description
   - The relevant spec excerpts (not the full spec — context-minimize)
   - The relevant files only (do not give it the whole repo)
   - The Superpowers `test-driven-development` skill is in scope — implementation must follow red-green-refactor
   - Instruction: flag explicitly anything you are guessing about, and anywhere you chose between non-obvious alternatives. If you encounter contradiction with the spec, a new auth boundary, or anything matching an escalation trigger, stop and report rather than guessing.

3. **Post-implementation review.** When the implementation subagent returns, invoke the Superpowers `requesting-code-review` skill. Pass it:
   - The diff
   - The task description from `plan.md`
   - The relevant spec requirements

   The review must explicitly check spec adherence, not just code quality. If the review flags critical issues, send them back to the implementation subagent and re-review. Do not advance to the next task with critical issues open.

   Additionally: if the implementation surfaced new assumptions (the subagent flagged guesses, or the diff implicitly depends on something not in the spec), append them to the Assumptions section of `spec.md` and confirm with the user before continuing.

4. **Goal anchor check.** Before marking the task complete, ask yourself: does this change still advance `goal.md`? If you can't trace the change back to a goal success criterion, something has drifted. Stop and investigate.

5. **Update state.** Append the task ID to `completed_tasks` in `state.json`, advance `current_task`, and append a short note to `.feature/<slug>/reviews/<task-id>.md` summarizing what changed and any deferred items.

**Critical: do not let subagents see more context than they need.** Subagents working on task-005 should not have task-001's implementation diff in context, only the current state of the relevant files. Context pollution is the leading cause of inconsistent abstractions across a multi-task feature.

---

### Phase 6: HOLISTIC REVIEW

**Purpose:** verify the feature as a whole, against the original goal, not just task-by-task.

**Entry condition:** all tasks in `plan.md` are marked complete.

**What you do:**

1. **Re-read `goal.md` first.** Not the spec, not the plan — the goal. The goal is what the user actually wanted. Everything else is means to that end.

2. **Spawn a holistic reviewer subagent** with `goal.md`, `spec.md`, and the full diff of all changes across the feature branch. Its job is to answer:
   - Does the implemented feature actually solve the problem stated in `goal.md`?
   - Are all success criteria from `goal.md` demonstrably met?
   - Are there architectural inconsistencies across tasks (different naming conventions, duplicated abstractions, contradictory error handling)?
   - Is there hidden coupling between modules that the task-level reviews missed?
   - Are there gaps in test coverage at the integration boundary?
   - Is observability sufficient to debug this feature in production?
   - Are there non-goals from `goal.md` that the implementation accidentally violated (scope creep)?
   - Review the Assumptions section of `spec.md` — are any of them now demonstrably wrong, or known to be wrong because of how the implementation evolved? Invalidated assumptions are critical findings.

3. **Address findings.** Critical findings spawn new tasks appended to `plan.md` and execute through Phase 5. Minor findings get logged in `.feature/<slug>/reviews/holistic.md` for follow-up.

4. **Hand off to `finishing-a-development-branch`.** Once the holistic review is clean, invoke the Superpowers `finishing-a-development-branch` skill to run final tests, present merge options, and clean up.

5. **Archive the feature.** Move `.feature/<slug>/` to `.feature/_archive/<slug>/` so the next feature starts clean.

---

## Subagent delegation patterns

You drive the lifecycle. Subagents do bounded work. The split:

| Work | Who |
|---|---|
| Phase gate decisions | Orchestrator (you) |
| Artifact authorship (goal, spec) | Orchestrator |
| Brainstorming dialogue | Delegated via `brainstorming` skill |
| Plan generation | Delegated via `write-plan` skill |
| Task pre-flight review | Subagent |
| Implementation | Subagent (with `test-driven-development`) |
| Per-task code review | Delegated via `requesting-code-review` skill |
| Holistic end review | Subagent |
| Branch finishing | Delegated via `finishing-a-development-branch` skill |

When spawning a subagent, give it: the task, the minimum context it needs, the exit criteria, and the skill(s) it should follow. Do not dump the full feature directory on it.

### Profile composition

Subagent role (what they do) is independent from profile (how they reason). The same role takes different profiles depending on the task. Profile is delivered as a paragraph of behavioral guidance in the spawn prompt — there is no separate agent type, and there is no fixed taxonomy to pick from. Compose to match what the task demands.

Defaults that have proven useful:

- **Brainstorming during CONCEPT** — exploratory profile. Generate at least one approach you don't initially like. Resist anchoring on the first plausible option.
- **Pre-flight review of a `one-way` task** — paranoid profile. Assume the plan is wrong until proven otherwise. Attack the proposed approach for failure modes; do not perform a polite review.
- **Pre-flight review of an `easy` task** — balanced profile. Sanity-check for obvious issues without theatrical caution.
- **Implementation of performance-sensitive code** — optimization-biased profile. Spec performance bars are non-negotiable. Profile-driven decisions get explicit justification in comments.
- **Implementation at integration boundaries** — reliability-biased profile. Favor explicit error handling, retries, and observability over terseness. The cost of a missing log line at 2am is higher than the cost of three extra lines now.
- **Holistic end-of-feature review** — integration-focused profile. Look for inconsistencies *across* tasks (naming, error handling, abstraction shape), not within them — the per-task reviews already covered within-task quality.

When two profiles seem to fit equally, the task is probably scoped wrong — split it.

---

## Anti-patterns

Things that look productive but aren't:

- **Skipping the goal phase because the user "already said what they want".** They didn't. They said one version of it. The goal artifact forces the missing detail to surface.
- **Writing the spec during brainstorming.** Brainstorming explores options; spec commits. Mixing them produces specs anchored to the first idea that sounded plausible.
- **Letting subagents see the full plan.** They start optimizing for the next task instead of the current one, and abstractions get pre-baked for hypothetical needs.
- **Treating passing tests as proof of correctness.** Tests prove the implementation matches the test. The holistic review proves the feature matches the goal. Both are required.
- **Skipping the goal anchor check during execution.** Drift accumulates one task at a time. The check is cheap. Run it every task.
- **Running this skill on a 30-minute change.** The overhead isn't worth it for small fixes. Use it for features — net new capability the user described as a feature.
- **Producing a "best guess" when an escalation trigger fires.** The triggers exist because past-you decided in advance these were not autonomous decisions. Respect that. The cost of a five-minute pause is always lower than the cost of a wrong commitment shipped under fake confidence.
- **Adding tasks to `plan.md` during EXECUTE without re-evaluating against `goal.md`.** Tasks accumulate because they "seem necessary now," and a feature designed for the goal becomes a feature designed for the in-flight implementation. The Gate 2 complexity check exists to make this visible — do not work around it.

---

## Resuming an in-progress feature

If `.feature/<slug>/state.json` exists when this skill is invoked:

1. Read it. Identify the current phase.
2. Read the most recent artifact (`goal.md`, `concept.md`, `spec.md`, or `plan.md` depending on phase).
3. Summarize the current state to the user in 3–5 sentences: what's been done, what's next, what (if anything) is blocked.
4. Confirm with the user before proceeding: "Resuming at phase X, next action is Y — sound right?"

Resumption is not optional. Features take days to weeks. The user will absolutely come back mid-feature and expect you to pick up where you left off.

---

## When NOT to use this skill

- The user asked for a one-line bug fix.
- The user wants to refactor existing code with no new behavior.
- The user wants a quick prototype to validate an idea (use brainstorming + write-plan directly instead).
- The user is asking a question, not requesting a build.

If unsure, ask: "This sounds like a feature build — should I run the ship-feature lifecycle, or do you want something lighter-weight?"
