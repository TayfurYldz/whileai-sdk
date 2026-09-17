---
name: strengthen-your-evals
description: >
  How to build an agent eval that survives scrutiny, and how to tell a
  real gain from a measurement artifact. Load before designing a held-out
  eval, before reporting a base-vs-trained delta, or when a result
  straddles zero and you are deciding whether to buy more data. Covers
  eval sizing, the four ways an eval silently lies, and what to put on a
  card.
metadata:
  version: "1.0.0"
---

# Strengthen your evals

Every rule here was paid for. On 2026-09-17 seven lanes trained models against
simulated data; the first two "wins" were both withdrawn, and the causes were
measurement, not modelling.

## Simulate the training data. Evaluate for real.

These are two different jobs and conflating them is the most expensive mistake
available.

**Simulation is how you get training data:** diverse, adversarial, cheap, shaped
at whatever the model is bad at.

**The eval should be something you did not build**: an external benchmark, a fixed
task set with gold answers, a held-out set of real traffic. Anything where you did
not choose both the question and what counts as a good answer.

Why it matters, measured on one adapter graded both ways:

| eval | base scores |
|---|---|
| our own simulated holdout | **73.2%** |
| tau2-bench, external | **18.0%** |

Same base model, same agent, 55 points apart. A simulated eval is built from the
same assumptions as the simulated training data, so it inherits every one of them.
Ours were easier than reality in at least five separate ways: a mocked world that
echoed the agent's own claims back as facts, fault injection that silently never
fired, a judge that dropped the long rows, conversations that ended before the hard
turn, and criteria that nothing in the set ever failed.

Every one of those inflates the base and hides the gain. The result that survived
the day was the one graded on a benchmark nobody here wrote.

**The claim worth making is "trained on simulated data, better on a real eval."**
Not "trained on simulated data, better on our simulation."

If you have no external benchmark, the next best things, in order: a fixed task set
with gold answers you can check with a program; a held-out slice of real traffic;
your own simulation with the user scripted rather than generated. Say which one you
used, every time.

## 0. First: did the eval actually run?

Before any statistics, check the agent was exercised at all. A hollow run does not
look broken. It looks like a perfect score.

Measured: a tester's first hosted run scored **pass@1 = 1.00** because the situation
writer invented order ids that did not exist, so the refund tool was never called
once. Nothing failed, so nothing looked wrong.

Check, in this order:
- **Did any rollout call a tool?** If not, you measured the model talking about the
  task.
- **Was every declared tool touched by something?** A tool no rollout ever calls is
  either unreachable (no dispatch branch) or irrelevant to your situations. Both are
  bugs and they look identical from outside.
- **Did every marker or criterion fire on at least one row?** A criterion that never
  fires reports a clean 1.000 and has taught nothing.
- **Do the ids in your situations exist in your world?** Invented entities are the
  usual cause of a hollow run. Real ids belong in `seeds=` or in the tool
  descriptions, not left to the writer to guess.

The SDK surfaces these: `simulate()` carries `no_tool_calls` in `degraded`, and
`run_judge` / `evaluate` / `data.grade` attach `coverage_warnings` to
`ScoredData.warnings`. Read them on the first run, not the tenth.

**A number from a hollow run is worse than no number, because it is high.**

## 1. Size the eval before you run it

An eval too small to see your effect will report a null no matter how good the
model is. Work out the resolvable effect FIRST.

Per-prompt paired spread is remarkably stable for agent rubrics: **sd ~= 0.376**,
measured across five independent lanes (range 0.336-0.453). At that spread:

**Use `holdout_size(effect, base=, k=, rows=)` from the SDK.** It models the test
`delta_report` actually runs. Pass `rows=` to read base and k off your own data.

| true effect | 50% power (interval just excludes zero) | **80% power (what you want)** |
|---|---|---|
| +0.03 | ~600 | **~1020** |
| +0.05 | ~220 | **~370** |
| +0.07 | ~110 | **~190** |
| +0.10 | ~55 | **~95** |

**Design for 80% power, not 50%.** At 50% power a real effect of exactly that size
fails to clear zero half the time. You are coin-flipping on whether your own true
result reads as a null. The 50%-power column was first circulated here as if it were
the answer; it is roughly half the prompts actually needed.

**A 50-prompt eval only reliably detects a 13-point gain.** That is a very large gain
to demand of a LoRA on a few hundred rows.

**`holdout_size` assumes the two arms are independent**, so on a paired eval it overstates the
tasks you need. The size of the overstatement is set by how much your tasks differ in
difficulty, and it does NOT shrink as you raise k (k cancels out of the ratio). Model
sd divided by true sd, simulated at 6000 tasks per cell:

| spread of per-task base rates | k=1 | k=4 | k=16 |
|---|---|---|---|
| 0.02 | 1.00 | 1.00 | 1.01 |
| 0.20 | 1.08 | 1.08 | 1.10 |
| 0.40 | 1.26 | 1.26 | 1.27 |

So a hard, spread-out holdout is told to buy ~30% more tasks than it needs, at any k.
The diagnostic is free from rows you already have: the **sd of per-task base pass
rates**, `statistics.pstdev(pass_at(rows).per_task.values())`. Near 0.4, treat it as an upper
bound. A measured external run at that spread needed 82 tasks where the model asked
for 147.

Averaging ratios across lanes hides this: a set spanning 0.72 to 1.16 has a mean near
1.0 and is not evidence of calibration. An understatement below 1.0 cannot come from
the independence assumption, which can only overstate. Look for another cause.

**Prompts or rollouts? Measure, do not assume.** "Raising k never narrows a
bootstrap over prompts, always spend on prompts" was asserted and then refuted by
counter-measurement on another lane. It depends on how often your arms actually
disagree on the same prompt:

- **Mixed-verdict rate near 0%.** Every rollout of a prompt agrees. Extra k re-measures
  a settled prompt. Spend on prompts.
- **Mixed-verdict rate high (~40%).** Per-prompt rates are genuinely uncertain, and k
  buys real precision on each one.

Compute it on the eval arms before choosing: the share of prompts whose k rollouts do
not all agree. Beware the comparison that looks decisive and is not: "103 prompts at
k=1 beat 51 prompts at k=2" varies prompt count and k at once and isolates neither.

## 2. The four ways an eval silently lies

**The simulated user runs on the model under test.** If your eval generates user
turns live, and the person is voiced by whichever weights you are evaluating, the
two arms face different environments. Pin the user to one fixed model on BOTH arms.
Symptom: arms show different mean user-turn counts. Caution: after pinning, some
asymmetry is legitimate, because a better agent resolves things in fewer exchanges.

**Rows vanish from the denominator.** If the grader can fail on a row, the row must
still be COUNTED. Long trajectories are the ones that fail to grade, and long
correlates with failing, so the drop is not random. One lane's base pass rate moved
**0.717 -> 0.603** when the dropped rows came back.
> A random drop widens an interval. A drop correlated with failing moves the
> estimate, and always in the flattering direction.
Always report **graded-count per arm**. Unequal denominators mean the number is not
paired, whatever else is right.

**A fixed task set pins less than you think.** Pinning tasks often fixes only the
OPENING prompt. Everything after it is still generated. Check turn counts.

**The world confirms whatever the agent claims.** A mocked world that echoes call
arguments back as record fields will confirm any assertion the agent makes, and a
grounding rubric then scores the fabrication as grounded. This is a reward hack
living in the world rather than the reward, so hack-scanning the reward will not
find it.

## 3. Prefer designs that cannot be confounded

Ranked by how little can go wrong:

1. **Fixed prompts + greedy decoding + a program grader.** Nothing is generated at
   eval time, so nothing can drift between arms. Identity-style trait evals and
   execution-match SQL evals both work this way, and both produced defensible
   numbers on a day when nothing else did.
2. **External benchmark with an external, pinned user simulator.** Code-graded, one
   process serving both arms.
3. **Scripted multi-turn.** You can keep multi-turn and still be fixed: write the
   user's lines into the task file rather than generating them.
4. **Live simulation.** Only when the others cannot express the task, and then pin
   the user model explicitly.

**Use a program grader wherever a program can decide it.** "Does this SQL return the
gold result set", "does the answer name both fields", "did an irreversible call
follow a user yes" are all code. Reach for a judge only for things no program can
see, such as tone or register, and then use a different model family from the
policy, because a judge prefers its own family's writing.

## 4. Reading a straddling result

**"Straddles zero" means the eval cannot tell, not that the model did not improve.**
Say which.

Distinguish two different moves, because they are not the same thing:
- **Removing a bias** (fixing a confound) genuinely changes the estimate.
- **Adding prompts** narrows the interval; the point estimate moving is noise.

One lane read +0.071 -> +0.059 -> +0.041 as "the effect erodes under scrutiny". The
first step was a real bias removal. The second was **0.53 SE**, which is noise. Treating
both as erosion teaches you to expect every effect to vanish, when small evals are
simply noisy in both directions.

**When the estimate keeps sitting below what your eval can resolve, stop buying
sample size.** "The effect is smaller than +0.045 on this task" is a finding.

**On binary rewards, check how many tasks actually moved.** One result of
+0.140 [+0.020, +0.260] rested on 11 discordant tasks out of 50; the exact sign test
gave p = 0.065. The bootstrap excluded zero, the sign test did not. Report both.

## 5. Before you train

- **Base pass rate decides the optimizer, measured ON THE EVAL YOU WILL RUN, not on
  the training distribution.** `select_for_sft` keeps only passing rows, so above
  roughly 0.6 base there is little left to imitate. Lanes at 0.73 and 0.90 nulled for
  exactly this.
  One lane chose an arm because base scored **40%**, comfortably in the trainable
  band, but that was measured on the training world (one numeric-boundary bug). The
  holdout was a different repo with a boolean-logic bug and two failing tests, where
  base scored **7%**. They sized the round on one distribution and graded it on a much
  harder one. Combined with 54 prompts, which only detects ~13 points, they were asking
  a LoRA on 122 demonstrations for a 13-point swing off a 7% floor. **The null was
  determined before the run started.**
  Run your base against the actual holdout first. If base is near the floor there, the
  eval cannot show a gain no matter how good the training data is.
- **Per-criterion FAILURE counts, not row counts.** A criterion nothing fails cannot
  be learned however many rows attack it.
- **For a trait, gate twice:** does the base LACK it (judge base replies with no
  persona in the prompt), and does your DATA CARRY it (judge trait rows against
  control rows). One lane measured base 0.150 and separation 0.983 before spending
  anything, and ruled out a second trait at base 0.93-0.96.
- **If you fix your generator, RE-MEASURE THE BASE before sizing the round.** A
  generation fix can move the base more than training moves the trained arm. One lane
  measured base 0.527, fixed a defect that was cutting conversations off before the
  confirmation step, and the base rose to 0.771 on the repaired set, because the base
  was good at confirming once the conversation let it. Their own SDK fix removed most
  of their own headroom, and the round was then too small to show anything.
- **Control for length.** Passing trajectories are usually shorter than failing ones,
  so preference pairs carry a brevity signal. Report trained-vs-base reply length; a
  delta carried by length is not a delta in the trait.

## 6. What goes on the card

The numbers, the eval size, and the resolvable effect at that size. That is the
useful line for a reader: it tells them how to read the number in front of them.

Not the correction history. Not what an earlier version said. Those belong in your
own log.
