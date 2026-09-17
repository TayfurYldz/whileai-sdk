# Endpoint SFT: keep the two ends of the reasoning trace, drop the middle

**Paper:** Revisiting Complete Reasoning Traces for Post-Training, Jaehui Hwang et al., arXiv:2609.07103, September 2026. https://arxiv.org/abs/2609.07103
**Book:** rlhfbook.com ch. 4 Instruction tuning: the prompt is masked and "the completions are what the model actually learns from", so editing the completion text is editing the whole training signal — this recipe changes nothing else.
**Claim:** the middle of a machine-written reasoning trace is weakly attended and carries little that the model cannot re-derive; training on the first and last steps only (about 20% of the tokens removed) matches full-trace SFT and sometimes beats it.
**The change:** the assistant's trace is cut to its first `n` steps plus its last `n` steps before training. Same problems, same model, same learning rate, same optimizer steps, same prompt mask.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Train data: 600 R1 traces from `open-r1/OpenR1-Math-220k`, each one the dataset's own math-verified generation, filtered to 800–3500 tokens and at least 24 steps. Holdout: 64 problems from `HuggingFaceH4/MATH-500`, a different corpus.
2. Baseline arm: TRL `SFTTrainer` + LoRA (r=32), one epoch, lr 1e-4, 8 sequences per optimizer step, 4096-token context, loss on the completion only. The window in step 1 keeps every trace inside that context, so the baseline is never truncated from the right — "full trace" must not quietly mean "the trace minus its answer".
3. Recipe arm: identical, except each trace is cut to its first `n` and last `n` blank-line-separated steps. `n` is one number for the dataset, picked so about 20% of trace tokens go (the paper's §C.3 rule: n=100 on s1K-1.1, n=200 on OpenThoughts3). On this data that is **n = 21**, dropping **19.2%** of trace tokens.
4. Eval: pass@1 on the same 64 problems, 4 samples each, temperature 0.6, graded by `MathEqual` against the public gold answer. The untrained base is evaluated three times first and that spread is the noise floor; the train set is decontaminated against the holdout before anything trains.
5. Read the Result table with its interval, not its middle number. 64 problems is a small holdout — see Learned.

## Run

```bash
python recipe.py --selftest   # the truncation rule, offline: no GPU, no key, no network
python recipe.py --plan       # the same rule on the real data, CPU only: picks n, prints what it drops
python recipe.py              # both arms, budgeted for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --n-holdout 200
```

Needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Modal workspace `zeroproofai`). `WHILEAI_API_KEY` is optional: with it the run appears on zeroproofai.com/platform, without it the recipe trains and prints the same numbers.

## Result

**Not run.** This recipe was written in a session with no Modal credentials, so there is no GPU number to report and `results.json` carries `"verified": "1970-01-01"`, which this repo uses to mean never run. Every cell below fills in on the first run. None of them are estimates.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | — | — | — | 0 | 0 |
| Baseline (full trace) | — | — | — | — | — |
| Recipe (endpoints only) | — | — | — | — | — |

Recipe vs baseline: not measured. Verdict: **flat** — a verdict with no run behind it is not a result.

What did get measured today, on the CPU, through the shipped code path:

| Measured without a GPU | Number |
|---|---|
| Train traces after filtering | 600, median 50 steps, median 2464 tokens |
| `n` chosen by the paper's 20% rule | 21 steps at each end |
| Trace tokens the cut removes | 19.2% (1,464,426 → 1,181,008 target tokens) |
| Traces with no middle to remove (≤ 2n steps) | 230 of 600 |
| `decontaminate(train, against=holdout)` | 0 of 600 dropped, and 0 against all 500 MATH-500 problems |

`python recipe.py --selftest` checks the truncation rule itself: that the ends keep `<think>`, `</think>` and the boxed answer, that a short trace comes back untouched, that the drop falls as `n` rises, and that `n` is chosen by token mass rather than by trace count. What none of it says is whether the change moves pass@1. That needs the GPU run.

## Checks

Nothing here is ticked by hand: every cell is written by `recipe.py` into `results.json`. The rows that need a trained model are empty because the run has not happened; the rows that do not are filled in and say so.

| Check | Book | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | ch. 16 | not run — a delta under 2 x run_std will be called noise |
| Holdout is clean: `decontaminate(train, against=holdout)` | ch. 16 | **run today, CPU: 0 of 600 train rows dropped** (also 0 against all 500 MATH-500 problems). OpenR1-Math-220k comes from NuminaMath and MATH-500 is a slice of the MATH test set, so this was worth measuring rather than assuming |
| Reward is a program, not a judge | ch. 7, 13 | `MathEqual` against the public MATH-500 gold answer. No judge, no model in the loop |
| Proxy vs target: `delta_report(proxy=)` | ch. 14 | `proxy="marker:trace_form"`: SFT optimizes the *shape* of the trace (a closed `<think>` block ending in `\boxed{}`) whether or not the answer is right. If that rises and pass@1 does not, the report says over-optimized and the verdict cannot be "moved" |
| Length: mean completion length before -> after, per arm | ch. 14 | not run. Worth watching here: the recipe arm is trained on shorter targets, so a length drop is expected and is not by itself a win |
| Hack scan on the last training batch: `hack_scan` | ch. 14 | not run. SFT has no per-rollout training reward to scan, so this runs on the arm's graded holdout rollouts instead — what separates a right answer from a wrong one. Nothing is endorsed |
| Pinned: seed, torch, transformers, trl, peft | app. C | seed 17 in the trainer, `--seed 0` for the holdout draw; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

Both arms share the seed, the problems, the holdout, the grader and every trainer knob. The target text is the only difference, so the delta has one cause available to it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: n chosen for a ~20% token drop (n=21 here), 600 traces, 1 epoch, lr 1e-4 | not run | — |

The knob the paper says matters is the cutoff itself: Figure B reports that performance is flat across a broad band of retained-step counts but falls off under heavy truncation. Round 2, if round 1 is flat and budget is left, lowers the target drop so the cut reaches the 230 traces it currently leaves alone — the direction the paper's own curve says has room before it hurts.

## Learned

- The paper's dataset-wide single `n` leaves a lot of the data alone: at n=21, 230 of 600 traces here have 42 steps or fewer, so they are byte-identical between the arms. The change is real for 62% of the rows and absent for the rest, which caps how large a delta this design can produce before anything is trained.
- The effect size this recipe can see is far bigger than the effect the paper reports. On Qwen3-4B the paper's E-SFT gain averages +1.07 points, and it is *negative* on MATH itself (91.60 → 91.13). A 64-problem holdout at k=4 cannot resolve a point. So what this recipe actually tests is the paper's weaker, more useful claim: that dropping ~20% of the trace tokens does not cost pass@1. A flat verdict is the expected outcome and is worth shipping.
- Gradient checkpointing is on here, unlike the GRPO recipes next door. Their rule is about trainers that generate while they train, where checkpointing corrupts Qwen generation on these pins. `SFTTrainer` is teacher-forced and never generates, so the rule does not reach it, and 4096-token sequences want the memory back.

Verified 1970-01-01 (never run), whileai 0.51, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. Run page: none yet.
