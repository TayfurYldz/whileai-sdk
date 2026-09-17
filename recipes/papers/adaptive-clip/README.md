# Adaptive clip: the upper bound follows how rare a correct answer was

**Paper:** Group Adaptive Clipping Policy Optimization, Sheng Jia et al., arXiv:2609.00444, August 2026. https://arxiv.org/abs/2609.00444
**Book:** rlhfbook.com ch. 6 Policy gradients: the clipped surrogate objective, and what the width of the clip range does to an update the group thinks is worth making.
**Claim:** GRPO clips every rollout against the same upper bound, so the one correct answer in a hard group and the seventh correct answer in an easy group are held back equally; letting the bound widen as correct answers get rarer gives the informative rollouts more room and raises pass@1 and pass@k on math and code.
**The change:** the upper clip bound is computed per group from how many of its rollouts were right, instead of being one fixed number.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, so there is no overlap to check for).
2. Reward, both arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. The paper changes the clip, not the reward, so nothing here is shaped.
3. Baseline arm: GRPO with `epsilon` 0.20 and `epsilon_high` 0.28, fixed for every rollout. That pair is DAPO's clip-higher and it is the paper's own token-level default.
4. Recipe arm: same 0.20 floor and the same 0.28 ceiling, but the upper bound slides per group, `eps_hi(c) = eps_lo + (eps_hi_max - eps_lo) * (k - c) / (k - 1)` with `c` correct out of `k = 8` rollouts. One right out of eight keeps the full 0.28; seven right gets 0.2114.
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task. The untrained base is evaluated three times first, and that spread is the noise floor a delta has to clear; the train set is decontaminated against the holdout before any training. Paired delta with a 95% interval (`wai.pass_at`, `wai.delta_report`).

Both arms share the floor and the ceiling, so the comparison isolates the sliding and not the width. There is no KL term (`beta` 0), which leaves the clip as the only trust region in the run — the thing the paper is about. The recipe runs the paper's token-level importance sampling, not its sequence-level GSPO variant, so its Seq-IS epsilons (3e-3 / 5e-3) do not apply here.

## Run

```bash
python recipe.py --selftest   # the clip schedule and the clamp, offline, no GPU and no key
python recipe.py              # both arms, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --eps-high-max 0.36
```

## Result

Run today, both arms, on one L40S. Round 2 is the headline: round 1 ran the
clip where it cannot bind, so it could not have tested the paper. See Climb.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.34 | [0.28, 0.41] | 0.58 | 0 | 0 |
| Baseline (fixed upper bound 0.28) | 0.57 | [0.50, 0.64] | 0.80 | 40 | 20.5 |
| Recipe (bound slides with the group) | 0.51 | [0.44, 0.58] | 0.77 | 40 | 12.1 |

Recipe vs baseline: **-0.065 [-0.117, -0.013]** over 120 paired tasks.
Verdict: **flat**. The interval excludes zero, but on the wrong side of it —
`delta_report` calls this `moved_the_wrong_way`, and this repo reserves "moved"
for a gain. Read it as: at this size, sliding the bound did not help, and the
run that did it came back 6.5 points lower.

Do not read that 6.5 as the paper being wrong. The clip fired on roughly 0.03%
of tokens even in round 2 (`clip_ratio/high_mean` between 0 and 0.0003), and
two arms at one training seed cannot separate a 6.5-point gap from ordinary
training variance — the run_std in the Checks table is the *eval's* noise, not
the trainer's. What this recipe can say is that GRPO itself worked: both arms
took the base from 0.34 to above 0.50.

The selftest also turned up one thing worth knowing before you read the paper's
equation 11 literally: at `c = 0` it returns 0.2914, above the 0.28 ceiling it
is supposed to stop at. The equation is written for a group that splits,
`1 <= c <= k`. All-wrong groups have a zero advantage and contribute no
gradient, so the recipe clamps the count into `[1, k]` and they land on the
ceiling instead of over it.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`. These are the round 2 numbers.

| Check | Book | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | ch. 16 | **run_std 0.0032**, so a delta under 0.0064 is noise. This measures re-running the eval, not re-running the training |
| Holdout is clean: `decontaminate(train, against=holdout)` | ch. 16 | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits — measured, not assumed |
| Reward is a program, not a judge | ch. 7, 13 | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `delta_report(proxy=)` | ch. 14 | `proxy=None`: the training reward *is* the target metric, the same binary check, so there is no proxy to over-optimize |
| Length: mean completion length before -> after, per arm | ch. 14 | **715 chars base -> 589 baseline, 608 recipe.** Both arms got shorter and more right, so the gain is not length gaming |
| Hack scan on the last training batch: `hack_scan` | ch. 14 | top feature `contains:days AND contains:\\` — GSM8K word-problem vocabulary, not a reward surface. Nothing is endorsed, so this is the scan reporting it found nothing |
| Pinned: seed, torch, transformers, trl, peft | app. C | seed 17 in the trainer, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

The two arms share the seed, the data, the holdout, the reward and every
trainer knob except the clip bound, so the delta has one cause available to it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: eps 0.20 / 0.28, k = 8, 40 steps, lr 1e-4, LoRA r=32, 1 policy update per batch | baseline 0.61, recipe 0.59 | -0.013 [-0.054, +0.031], flat |
| 2 | same, but 2 policy updates per batch (`--num-iterations 2`) so the clip can bind at all | baseline 0.57, recipe 0.51 | -0.065 [-0.117, -0.013], flat (wrong way) |

**Round 1 could not have tested the paper, and the trainer's own logs say so.**
TRL takes one policy update per batch of rollouts by default. On that update
the sampling policy and the trained policy are the same, so the importance
ratio is exactly 1 — rlhfbook ch. 6: "the policy ratio starts at 1 for the
first gradient step for that batch". A ratio of 1 never reaches a bound of
1.20 or 1.28, so `epsilon_high` is dead weight and a *per-group*
`epsilon_high` is dead weight per group. `clip_ratio/high_mean` was 0.0 in all
80 logged steps of round 1. The -0.013 it reported was generation
nondeterminism between two arms running the same arithmetic.

Round 2 takes the chapter's "1-4 gradient steps per batch" at 2, which is the
smallest change that lets the second update be off-policy. The clip does then
fire — and barely: 0.0003 of tokens at its highest. A bound that touches 0.03%
of tokens is not a place where a per-group schedule can show itself, which is
the honest summary of this recipe at this budget. Round 3 is more updates per
batch (4) or a bigger batch, not a different `eps_hi_max`.

## Learned

- The bound can be made per-group without touching TRL's loss body. `epsilon_high` is read inside `_compute_loss`, and `torch.clamp` takes tensor bounds, so handing it a (batch, 1) tensor broadcasts over the (batch, tokens) ratio and the `clip_ratio` metric TRL logs stays correct.
- Reading the group's correct count off the sign of the advantage only works because this reward is binary: with rewards in {0, 1} the advantage is `r - c/k`, positive for exactly the correct rollouts. A shaped reward would break that and need the counts carried separately.
- Holding the ceiling equal across the arms is the honest comparison but it is also the conservative one: it makes the recipe a strictly tighter clip than the baseline. The paper compares against a fixed bound too, on a much bigger batch (256 prompts against 6 here), so a flat result could mean the batch rather than the idea.
- **Check that your change is reachable before you spend a GPU hour on it.** The per-group bound was implemented correctly, tested on the CPU against real TRL, and verified to arrive at the loss intact — and still could not move a gradient, because nothing in an on-policy run ever asks what the upper bound is. One line of the trainer's own logging (`clip_ratio/high_mean`) would have said so before the run. It is now in the Checks a reader can see.
- **GRPO was the intervention that mattered here.** Both arms moved the base from 0.34 to 0.51-0.57 on GSM8K in 40 steps, which dwarfs everything the paper's change could have done at this scale. Two-thirds of groups were flat (`frac_reward_zero_std` around 0.5-0.67), so most of that came from a third of the rollouts.

Verified 2026-09-17, whileai 0.53, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. 32.6 GPU minutes, $1.09 on one L40S (round 1: 43.1 minutes, $1.44). Run page: https://www.zeroproofai.com/platform/training/run_b546f2c31dcbcf4e
