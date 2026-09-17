---
name: strengthen-your-evals
description: >
  Design an agent eval that survives scrutiny, and tell a real gain from a
  measurement artifact. Use when sizing a held-out eval, reporting a
  base-versus-trained delta, deciding whether a straddling result needs more
  data, or checking whether a dataset can train the behaviour at all. Covers
  eval sizing, the four ways an eval silently lies, designs that cannot be
  confounded, and what belongs on a model card.
metadata:
  version: "1.0.0"
---

# Strengthen your evals

Most reported gains on simulated agent data are measurement, not modelling.
The failures below were measured, not theorised: across seven training lanes
on one day, the first two wins were both withdrawn, and neither cause was the
model.

## 1. Size the eval before you run it

An eval too small to see your effect reports a null however good the model is.
Work out the resolvable effect first.

Per-prompt paired spread is stable for agent rubrics at roughly `sd = 0.376`,
measured across five independent lanes with a range of 0.336 to 0.453.

| true effect | prompts needed to exclude zero |
|---|---|
| +0.03 | ~600 |
| +0.05 | ~220 |
| +0.07 | ~110 |
| +0.10 | ~55 |

Measure your own spread rather than assuming this one:
`sd = half_width / 1.96 * sqrt(n)` from any `pass_at` interval you already have.
A sixth lane, code-scored on a tool-using helpdesk agent, measured `sd = 0.233`
over 219 prompts, well below the range above. It resolved 4.4 points at 80%
power where the table predicts 6.5, so the table cost that lane a third of its
sensitivity in planning. The spread is a property of your rubric, not a constant.

A 50-prompt eval only detects a 10-point gain, which is a very large demand of
a LoRA trained on a few hundred rows.

**Effective sample is PROMPTS, not rollouts.** Raising `repeats` sharpens each
prompt's estimate; it does not narrow a bootstrap taken over prompts. One lane's
single-rollout eval on 103 prompts had a tighter interval than its two-rollout
eval on 51. Spend the budget on prompts.

## 2. The four ways an eval silently lies

**The simulated user runs on the model under test.** If user turns are generated
live and the person is voiced by whichever weights you are evaluating, the two
arms face different environments. Pin the user to one fixed model on both arms.
Symptom: the arms show different mean user-turn counts. After pinning, some
asymmetry is legitimate, because a better agent resolves things in fewer turns.

**Rows vanish from the denominator.** If the grader can fail on a row, that row
must still be counted. Long trajectories are the ones that fail to grade, and
long correlates with failing, so the loss is not random. One base pass rate moved
from 0.717 to 0.603 when the dropped rows were recovered. A random drop widens an
interval; a drop correlated with failing moves the estimate, always in the
flattering direction. Always report graded count per arm. Unequal denominators
mean the number is not paired, whatever else is right.

**A fixed task set pins less than you think.** Pinning tasks often fixes only the
opening prompt, and everything after it is still generated. Check turn counts per
arm.

**The world confirms whatever the agent claims.** A mocked world that echoes call
arguments back as record fields will confirm any assertion the agent makes, and a
grounding rubric then scores the fabrication as grounded. This reward hack lives
in the world rather than the reward, so scanning the reward will not find it.

## 3. Prefer designs that cannot be confounded

Ranked by how little can go wrong:

1. **Fixed prompts, greedy decoding, a program grader.** Nothing is generated at
   eval time, so nothing can drift between arms.
2. **External benchmark with an external, pinned user simulator,** code-graded,
   one serving process for both arms.
3. **Scripted multi-turn.** Keep multi-turn and stay fixed by writing the user's
   lines into the task file instead of generating them.
4. **Live simulation.** Only when the others cannot express the task, and then
   pin the user model explicitly.

Use a program grader wherever a program can decide it. Whether a query returns
the gold result set, whether an answer names both fields, whether an irreversible
call followed a user's yes, are all code. Reach for a judge only for what no
program can see, such as tone or register, and then use a different model family
from the policy, because a judge prefers its own family's writing.

## 4. Reading a result that straddles zero

Straddling zero means the eval cannot tell, not that the model did not improve.
Say which.

Distinguish two moves that are not the same thing. Removing a bias genuinely
changes the estimate. Adding prompts narrows the interval, and the point estimate
moving is noise. One lane read +0.071, then +0.059, then +0.041 as an effect
eroding under scrutiny; the first step was a real bias removal and the second was
0.53 standard errors, which is noise.

When the estimate keeps sitting below what the eval can resolve, stop buying
sample size. "The effect is smaller than +0.045 on this task" is a finding.

On binary rewards, check how many tasks actually moved. One result of
+0.140 [+0.020, +0.260] rested on 11 discordant tasks out of 50, where an exact
sign test gave p = 0.065. The bootstrap excluded zero and the sign test did not.
Report both.

## 5. Before you train

- **Base pass rate decides the optimizer.** `select_for_sft` keeps only rows the
  grader passed, so above roughly 0.6 base there is little left to imitate. Lanes
  at 0.73 and 0.90 returned nulls for exactly this reason.
- **Push the right rows.** SFT wants selected rows. Pushing a raw graded pool
  trains the model on its own failures, and the loss curve of such a run looks
  better than the correct one, not worse.
- **Per-criterion failure counts, not row counts.** A criterion nothing fails
  cannot be learned however many rows attack it.
- **For a trait, gate twice.** Does the base lack it, judged on base replies with
  no persona in the prompt, and does your data carry it, judged against control
  rows. One lane measured base 0.150 and separation 0.983 before spending
  anything, and ruled out a second trait at base 0.93 to 0.96.
- **Control for length.** Passing trajectories are usually shorter than failing
  ones, so preference pairs carry a brevity signal. Report trained-versus-base
  reply length; a delta carried by length is not a delta in the trait.

## 6. What goes on the card

The numbers, the eval size, and the resolvable effect at that size. Those three
tell a reader how to read the number in front of them. Report graded count per
arm alongside them.

Not the correction history, and not what an earlier version of the number said.
Those belong in your own log.
