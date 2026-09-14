# ZeroProof Simulations

The covering-grid simulator. Register the agent — tools, policy, optional
traces, optional `execute=` — and we spin the situations, sample pairwise,
walk hard. That is the expensive part labs still do by hand. Their judge
writes `r`. We do not steal the grade. This document is how it thinks, not
a tour of every option.

## The problem it solves

An agent's failures are specific. It hands off too early on one kind of
request, invents an order number under a particular kind of pressure,
loses its manners on the fourth turn. Fixing that with a system prompt
works until it does not. Fixing it in the weights needs examples of the
situation done right, and enough of them, with enough variety, that the
model learns the behavior rather than the example. Writing those by hand
is the expensive part of every post-training pipeline. The simulator
replaces the writing. Their judge keeps the judgment.

## Two ways in

**Describe the behavior.** One sentence is enough to start: "a personal
finance assistant that confirms before it moves money." The SDK drafts the
tools such an agent would have, builds a world around them, writes the
people who would talk to it, and runs the conversations. This is the
cold-start path for an agent that does not exist yet, or a behavior you
want to add to one that does.

**Point at the agent's traces.** Production already says where it is weak.
`load_traces` → `trace_report` → `simulate(traces=...)`. `steering_weight`
aims the budget at failures. `evaluate(...).failed_traces()` is the next
step on the same cards.

`agent=` is their policy on our cards — on-policy play. Hosted Qwen is our
walker when they want coverage first. Same grid either way. The writer can
still be Qwen; who plays is what makes the row on-policy.

Both paths use the same engine. The first one is what produced the
training set behind our first fine-tune, from nothing but the agent's
tool list and policy.

## How the simulator thinks

**Situations are coordinates, not prompts.** Asking a model for a thousand
user requests gives you a thousand variations of the same polite,
well-specified ask. The SDK instead declares covering axes from this
agent: which tool (their names, plus unrelated and multi_tool), which
policy clause, world_state when the tools have referent keys, tool
condition, stance, history. Length, vagueness, tone, and texture are
writer-only — they color the ask, they do not define the cell. The planned
grid is a pairwise covering array: every pair of axis values appears
together in at least one planned cell, which is the coverage strength the
testing literature settled on because most real failures come from two
things interacting. On a cold start the engine then flips most fault cells
to success (one cell per fault type stays), so the tool-condition axis is
sampled, not covered, unless you raise `fault_rate` or pass
`prefer_success=False`. What the run actually touched is a number:
`data.coverage["pairwise"]` is planned pairs, covered pairs, and the
fraction.

**People are sampled, not described.** A coordinate says the customer is
in a hurry and their order was already cancelled. A second layer decides
how that person writes: lowercase, clipped, run-on, with a typo, polite,
sarcastic. The writer never sees those labels; it sees an aside in prose,
because a model told to be terse writes an essay about being terse. The
same person shows up on turn five that showed up on turn one.

**The world answers honestly.** Tool calls go to a simulated world that is
deterministic for a given seed, returns records shaped like the tool's own
schema, remembers what it created, and says no. An unknown identifier is
not found. An argument that echoes the schema instead of the person's
details ("first name", user@example.com) is refused with a hint. A world
that never says no teaches an agent that never expects it; we learned
that the expensive way and built it in.

**Grade after is the product.** Authority stays with them. Default
`grade=False`: rows stay ungraded. Ungraded is honest, not a fake 0. Their
judge writes `r` via `data.grade(judge=)`, `zps.run_judge`, or `zps.grade`.
A broken judge stays ungraded. `evaluate()` is the same contract with
lineage `source=eval` (grade writes `source=grade`), so evals stay distinct
from training rewards. Structural flags (`grade=True`) are display, not
the score. A second judge — `audit_grades` in
`zeroproof.simulations.score.grade_llm` — can recommend rubric and eval
holes later. Advice, never `r`. It does not run on `simulate`.

**Failure is loud.** If the writer, the world, or a judge cannot do its
job, the run says so. Template fallbacks are never quietly substituted for
model-written situations, because a dataset that looks real and is not is
worse than no dataset.

## What you get

A JSONL file of conversations in chat format, with tool schemas, each
row carrying its situation (which axes, which world state, which faults
were scheduled), its persona tags, and, once their judge writes `r`, the
reward and the reason. From there the handoff: `select_for_sft`,
`select_for_rl`, `export_training`, `export_preference`, `pass_at`
(pass@1 / pass^k / pass@k) — SFT / preference / GRPO-shaped for their
trainer (Prime, TRL, their cluster). Training math lives there; we supply
the on-policy (or covering) trajectories and the judge contract. The live
loop is the same cards, stepped: they act, the world answers, their judge
scores, they update. The default leakage check is lexical: word and
bigram overlap with numbers collapsed, so it drops near-copies and copies
that differ only in an id, and it does not catch a paraphrase. Pass a
semantic `embedder=` to the leakage functions when that matters.

## What it is not

It is not ground truth. Every row is a simulation, kept by a grader, and
should be reviewed the way you would review a contractor's work. The world
is not your database. The people are drawn from a persona distribution,
not from your customers. The value is coverage, variety, and honesty
about all three.
