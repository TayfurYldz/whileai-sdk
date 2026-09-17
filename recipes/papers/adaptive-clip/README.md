# Adaptive clip: the upper bound follows how rare a correct answer was

**Paper:** Group Adaptive Clipping Policy Optimization, Sheng Jia et al., arXiv:2609.00444, August 2026. https://arxiv.org/abs/2609.00444
**Claim:** GRPO clips every rollout against the same upper bound, so the one correct answer in a hard group and the seventh correct answer in an easy group are held back equally; letting the bound widen as correct answers get rarer gives the informative rollouts more room and raises pass@1 and pass@k on math and code.
**The change:** the upper clip bound is computed per group from how many of its rollouts were right, instead of being one fixed number.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, so there is no overlap to check for).
2. Reward, both arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. The paper changes the clip, not the reward, so nothing here is shaped.
3. Baseline arm: GRPO with `epsilon` 0.20 and `epsilon_high` 0.28, fixed for every rollout. That pair is DAPO's clip-higher and it is the paper's own token-level default.
4. Recipe arm: same 0.20 floor and the same 0.28 ceiling, but the upper bound slides per group, `eps_hi(c) = eps_lo + (eps_hi_max - eps_lo) * (k - c) / (k - 1)` with `c` correct out of `k = 8` rollouts. One right out of eight keeps the full 0.28; seven right gets 0.2114.
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task, before and after each arm, paired delta with a 95% interval (`zps.pass_at`, `zps.delta_report`).

Both arms share the floor and the ceiling, so the comparison isolates the sliding and not the width. There is no KL term (`beta` 0), which leaves the clip as the only trust region in the run — the thing the paper is about. The recipe runs the paper's token-level importance sampling, not its sequence-level GSPO variant, so its Seq-IS epsilons (3e-3 / 5e-3) do not apply here.

## Run

```bash
python recipe.py --selftest   # the clip schedule and the clamp, offline, no GPU and no key
python recipe.py              # both arms, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --eps-high-max 0.36
```

## Result

Not run yet. This recipe was written in a session with no Modal credentials, so there is no GPU number to report and `results.json` carries `"verified": "1970-01-01"`, which this repo uses to mean never run. Every cell below fills in on the first run; none of them are estimates.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | — | — | — | 0 | 0 |
| Baseline (fixed upper bound 0.28) | — | — | — | 40 | — |
| Recipe (bound slides with the group) | — | — | — | 40 | — |

Recipe vs baseline: not measured. Verdict: flat, because a verdict with no run behind it is not a result.

What did get checked today, without a GPU. `python recipe.py --selftest` prints the schedule and shows the bound reaching a real loss. Separately, the trainer subclass was run end to end on the CPU against TRL 0.19.1 with a tiny Qwen2 model and a reward rigged to give each group a known number of correct rollouts. Each group got the bound equation 11 says it should, every value that arrived at the loss was one the equation can produce, and the bounds stayed glued to their own rollouts through TRL's batch shuffle and its gradient-accumulation split — the alignment that this whole change rests on. The baseline arm came through carrying no per-group bound at all. What none of that says is whether the change moves pass@1; that needs the GPU run.

The selftest also turned up one thing worth knowing before you read the paper's equation 11 literally: at `c = 0` it returns 0.2914, above the 0.28 ceiling it is supposed to stop at. The equation is written for a group that splits, `1 <= c <= k`. All-wrong groups have a zero advantage and contribute no gradient, so the recipe clamps the count into `[1, k]` and they land on the ceiling instead of over it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: eps 0.20 / 0.28, k = 8, 40 steps, lr 1e-4, LoRA r=32 | not run | — |

The paper's own knob is `eps_hi_max`, the ceiling the bound slides down from. With the ceiling at the baseline's 0.28 the recipe can only ever clip *more* than the baseline, so a flat round 1 may just mean the sliding had no room to help. If round 1 comes back flat, round 2 raises `--eps-high-max` to 0.36 before anything else moves, which lets a rare correct rollout go further than the baseline ever could.

## Learned

- The bound can be made per-group without touching TRL's loss body. `epsilon_high` is read inside `_compute_loss`, and `torch.clamp` takes tensor bounds, so handing it a (batch, 1) tensor broadcasts over the (batch, tokens) ratio and the `clip_ratio` metric TRL logs stays correct.
- Reading the group's correct count off the sign of the advantage only works because this reward is binary: with rewards in {0, 1} the advantage is `r - c/k`, positive for exactly the correct rollouts. A shaped reward would break that and need the counts carried separately.
- Holding the ceiling equal across the arms is the honest comparison but it is also the conservative one: it makes the recipe a strictly tighter clip than the baseline. The paper compares against a fixed bound too, on a much bigger batch (256 prompts against 6 here), so a flat round 1 could mean the batch rather than the idea.

Verified 1970-01-01 (never run), zeroproof 0.49, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. Run page: none yet.
