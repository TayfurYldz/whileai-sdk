# One entry point, or your before/after measures the SDK

**Seat:** a post-training engineer at a startup that ships one production agent, trying to
find out whether a small open model can take over the boring half of it.

**Question:** the previous run in the customer-simulation ledger ([#31](https://github.com/whilehq/whileai-sdk/issues/31))
trained an adapter and got `no_change_detected (-0.025)`, but could not tell whether that
meant "the fine-tune did nothing" or "the two arms went through different code paths". Its
baseline ran through `wai.hosted_model`, which sends `chat_template_kwargs={"enable_thinking": false}`;
its trained model ran through `wai.local_model`, which has no way to send it
([#264](https://github.com/whilehq/whileai-sdk/issues/264)). The confound sat exactly on the
before/after axis. **Does the delta survive putting both arms through the same entry point?**

What you will learn: how to pin a task set across a model swap, why the noise floor has to come
first, and the one thing that turned out to matter more than the fine-tune — that grading a
reply which still contains its own reasoning moves this eval's headline number by **26 points**,
which is an order of magnitude more than the adapter moved it.

You need `WHILEAI_API_KEY`. **No training run and no `wai.serve` call**: both models were already
hosted on the account. `--dry-run` needs no key and no GPU.

## Run it

```bash
pip install whileai            # 0.53
cd recipes/community/same-entrypoint-before-after
python run.py                  # tasks -> 3 base passes -> 1 adapter pass -> report
python run.py --dry-run        # offline, no key, no GPU
python run.py report           # re-print from saved rows
```

| flag | default | what it does |
|---|---|---|
| `step` | `all` | `tasks`, `eval`, `report` |
| `--budget` | 400 | rollouts the offline template writer draws the task grid from |
| `--repeats` | 4 | rollouts per task, so `pass^k` and `pass@k` mean something |
| `--dry-run` | off | offline end to end |

## What I actually ran

- **Base arm:** served model `qwen3-4b-think` (`adapterRunId: None`, i.e. the bare `Qwen/Qwen3-4B`)
- **Trained arm:** served model `billing-boring-half` (`adapterRunId: run_327b614f3682cae5`,
  the previous seat's SFT run: 87 rows, LoRA r=16, 2 epochs, held-out loss 5.4585 → 1.7336)
- Both through `wai.local_model(ENDPOINT, name, tools=..., api_key=..., temperature=0.8)`
- 37 pinned held-out tasks × 4 repeats ≈ 151 rows per pass, `concurrency=8`
- Three base passes (seeds 101/202/303) for the noise floor, one adapter pass (seed 101)

## Results

**RESULTS_TABLE_PLACEHOLDER**

## What did not work, and what to copy

**Copy this: grade a thinking model's reply with the reasoning stripped, and say which you did.**
Every one of 151 rows came back with `<think>` still in `final_text`, and 34 of them were cut off
before `</think>` — so `final_text` was reasoning and nothing else. Markers that read prose then
score the model's *hypotheticals* ("if the invoice were $500…") as claims it made. Stripping the
reasoning moved `no_invented_amount` from 0.310 to 0.577 and pass@1 from 0.485 to 0.750. The
noise band from three re-runs of the same model on the same pinned tasks is 0.055, so the
artefact is about **4.8× the noise band** — much larger than the thing the SDK correctly tells
you to worry about, and invisible unless you go looking.

This is the whole reason the previous run's `-0.025` was uninterpretable: with one arm through
`hosted_model` and one through `local_model`, that 26-point artefact is applied to **one side only**.

**Do not trust a marker pinned at exactly 1.000.** `escalated_over_200` and `used_a_tool` both read
1.000 with a zero-width interval. That is not a pass, it is a marker that never had a chance to
fail: across 151 rows the model made **2** `issue_credit` calls and 63 `escalate_to_human` calls,
so the branch the marker guards was barely exercised. The ledger has warned about this shape twice
and I still built two of them. A marker is only evidence if both outcomes occur in the corpus.

**`split_pseudo_production` does not give you a task-disjoint holdout**
([#268](https://github.com/whilehq/whileai-sdk/issues/268)). It is prompt-disjoint, which is what
its docstring promises, but 16–17 of ~28–37 held-out `scenario_id`s also appear in train, and
`decontaminate` reports the pair clean because it compares prompts. Since the engine's whole job is
writing rephrasings of one situation, prompt-disjoint and situation-disjoint are very different
sets. The script prints the task overlap so you cannot miss it.

**Things that cost me time and are worth knowing before you start:**

- The row handed to `grader=` has `steps`, not `messages`.
- A callable agent's own step dict uses `args`; the simulated model's steps use **`arguments`**.
  My `>$200` check read `args` only, so it silently never fired — and a marker that never fires
  reads as a perfect 1.000.
- The declared `returns` shape is not what comes back. I declared `amount_usd`; the simulator
  returned `{"status": "ok", "amount": 455.0}`. Match on values, not key names.
- `marker_summary` entries carry the rate under **`mean`**. There is no `rate` key; asking for one
  gets you `None` next to a populated `ci95`.
- `wai.local_model` has a **zero-character docstring**
  ([#269](https://github.com/whilehq/whileai-sdk/issues/269)), and it is the only way to evaluate a
  model you serve. I read the signature.
- Thinking mode costs about **8× wall clock**: 151 rows in 492 s here, against ~124 rows in 38–57 s
  for the previous seat's `hosted_model` pass at the same concurrency.

## Cost

No training run was started and nothing new was served — both models were already hosted, so this
is inference only: **four passes × ~151 rows ≈ 33 minutes of an already-warm A10G, well under $1**.
The SDK still reports no cost anywhere (`run`, `wai.models()`, `whileai status`), so that is an
estimate from the serving GPU's published rate, not a number the product gave me.

## Next

1. Re-run with the reasoning turned off at the server, not stripped in the grader, once
   [#264](https://github.com/whilehq/whileai-sdk/issues/264) has a `thinking=False`. Stripping is a
   workaround; the 34 truncated rows are *lost*, not recoverable, because the model spent its whole
   budget reasoning and never answered.
2. Build the eval on situations that actually exercise the `>$200` branch. Neither
   `scenario_dimensions` nor the prompt carries the invoice amount — the simulator invents it — so
   the only way I can see to force the branch is `result_shapes=` / `fault_plans=` on `local_model`.
   Untested.
3. Someone should run `method="grpo"` on the hosted path. `list_runs()` still shows only `sft` and
   one failed `dpo` on this account.
