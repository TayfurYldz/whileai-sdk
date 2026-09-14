# GRPO on Modal, with the dashboard watching

Group-relative RL on one testable rule, end to end: prompts from the
simulator, a reward that is a function rather than a judge, TRL's
`GRPOTrainer` with a LoRA adapter, reward and KL on the training page as it
runs, and a before/after on a holdout when it is done.

## Run it

```bash
pip install zeroproof modal
modal profile activate <your workspace>
export ZEROPROOF_API_KEY=...          # for the dashboard; optional
modal run examples/grpo/train_modal.py
modal run examples/grpo/train_modal.py --steps 80 --gpu H100 --run-name refund-grpo-v2
```

Default: 200 prompts, 20% held out by scenario, Qwen2.5-1.5B-Instruct, 40
steps of 8 generations, one A10G, under fifteen minutes. The run's URL is
printed at the start. No key means the same run with the numbers printed at
the end only.

## The environment

`reward.py` is the whole environment. The agent is the refund assistant with
one rule in its policy: look an order up before refunding it, never invent an
order id, ask when none is given. The reward reads the policy's first reply:

| prompt | right move | reward |
|---|---|---|
| names an order (`ORD-4017`) | `lookup_order` with that exact id | 1.0 |
| names an order | refund first, another tool, or ask for the id again | 0.0 to 0.3 |
| about an order, no id | ask for the id, no tool call | 1.0 |
| about an order, no id | any tool call (the id is invented) | 0.0 |
| off topic | a short reply, no tool call | 1.0 |

A well-formed `<tool_call>` block adds 0.2, capped at 1.0, so the policy
learns the wire format before the rule. `pass@1` counts a reply as a pass at
1.0 only; the raw score rides along as a marker, and so does `well_formed`,
which the delta report guards.

Prompts come from `zps.simulate(simulator=False, ...)`: the template writer
needs no model and no key, and every prompt carries its `case` (the order
id it names, whether it is about orders at all) so the reward has ground
truth.

## What you see

- **During:** `reward`, `reward_std`, `kl`, `completion_length` and the
  progress bar at zeroproofai.com/platform/training, from
  `zps.TrainerCallback`.
- **After:** pass@1 before and after on the same holdout prompts, four
  samples each, with intervals; `run.delta` puts the paired comparison on
  the run page and names `well_formed` if it regressed. The adapter and both
  holdout row files land on the `zeroproof-grpo-runs` volume under the run
  name.

## Reading it as RL

This is the reasoning-model recipe at toy scale: verifiable reward,
group-relative advantage, no value model, small KL to the reference
(`beta=0.04`), LoRA so a 1.5B model trains on one GPU. Two things to try
before believing a number: `--steps 10` to check the reward curve moves at
all, and a second seed on the prompts, since 40 holdout prompts is a wide
interval. The dashboard shows both runs side by side.
