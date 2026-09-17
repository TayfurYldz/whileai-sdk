# Filter metric: phantom advantages under a shaped reward

**Paper:** The Filter Metric is Safety-Critical: Phantom Advantages in Group-Relative RL under Shaped Rewards, Juntao Yu, arXiv:2609.13866, September 2026. https://arxiv.org/abs/2609.13866
**Claim:** When the reward is a 0/1 outcome plus a shaping term, dropping "flat" rollout groups by the shaped score instead of by the outcome collapses GRPO training; the paper reports 0.040 exact match against 0.754 on GSM8K with Qwen2.5-1.5B, same trainer, same everything else.
**The change:** the group-is-flat test reads the binary outcome instead of the shaped score.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, so no overlap to check for).
2. Reward, both arms: `outcome - 0.30 * min(len/512, 1)`. The outcome is `MathEqual` against the GSM8K gold number, so the reward is a program. The length term is the shaping the paper attacks: among wrong answers, the shortest one scores best.
3. Baseline arm: drop a group when its **shaped scores** are all equal (`--filter-metric score`). A group of four wrong answers of different lengths is not all-equal, so it survives, and GRPO's divide-by-group-std turns those length crumbs into full-size advantages. Those are the phantom advantages — gradient that looks like signal but only encodes "be shorter".
4. Recipe arm: drop a group when its **binary outcomes** are all equal (`--filter-metric outcome`). All-wrong is now flat, so the group is dropped and teaches nothing. This is what DAPO's dynamic sampling means.
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task, before and after each arm, paired delta with a 95% interval (`zps.pass_at`, `zps.delta_report`).

Dropping is done by masking: a group whose rewards are all equal gets a zero advantage, so it contributes nothing. The paper's DAPO arm deletes the group and refills the batch with fresh prompts. Masking reproduces the advantage-level effect on a fixed batch; it does not reproduce the refill, so this recipe cannot say anything about the paper's refill-rate numbers.

## Run

```bash
python recipe.py --selftest   # the filter on hand-written groups, offline, no GPU and no key
python recipe.py              # both arms, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --steps 80
```

## Result

Not run yet. This recipe was written in a session with no Modal credentials, so there is no GPU number to report and `results.json` carries `"verified": "1970-01-01"`, which this repo uses to mean never run. Every cell below fills in on the first run; none of them are estimates.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | — | — | — | 0 | 0 |
| Baseline (filter on shaped score) | — | — | — | 40 | — |
| Recipe (filter on binary outcome) | — | — | — | 40 | — |

Recipe vs baseline: not measured. Verdict: flat, because a verdict with no run behind it is not a result.

What did get checked today, without a GPU: `python recipe.py --selftest` shows the one change doing what the paper describes. On a group of four wrong answers that differ only in length, the score filter leaves rewards `[-0.001, -0.212, -0.002, -0.212]` (kept, and the shortest wrong answer wins) while the outcome filter flattens them to `[-0.106, -0.106, -0.106, -0.106]` (advantage zero, nothing learned). On a group that really splits, one right and three wrong, both filters keep it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: lambda 0.30, 40 steps, lr 1e-4, LoRA r=32, 5 rollouts per prompt | not run | — |

The paper's own knob is lambda, the weight on the length penalty. It sweeps 0.1, 0.3 and 0.5 and reports the collapse at 0.30, which is the default here. If round 1 comes back flat, round 2 raises lambda to 0.50 before anything else moves.

## Learned

- The filter metric is a separate choice from the reward, and a trainer will let you set it to the shaped score without complaining. Nothing in the loss curve says which one you picked.
- Shaping and group-std normalization interact: a shaping term too small to matter on its own becomes a full-size advantage once every rollout in the group is wrong and the std collapses to the shaping noise.
- This recipe's sizes are cut down from the paper's 32 prompts per step to 8 to fit one GPU hour, so a flat round 1 could mean the effect needs the bigger batch rather than that the effect is not there.

Verified 1970-01-01 (never run), zeroproof 0.48, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. Run page: none yet.
