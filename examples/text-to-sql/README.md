# Text-to-SQL: hill-climb a model on your schema with a verifier as the reward

A question about a database in, one SQL query out, and a reward that is a
program: run the query, compare the result set to the gold query's result.
No judge. This example builds the task set for a schema, benchmarks any
model on it, trains a 4B model with GRPO against that reward on Modal, and
measures each round on the same held-out tasks, so the curve is paired,
has an interval, and cannot be gamed by a wordier answer.

The schema shipped here is a small online store (8 tables, 300 orders,
seeded from `gen_seed.py`). Swap in yours: [Bring your own schema](#bring-your-own-schema).

## What you get

| File | Role |
|---|---|
| `schema.sql`, `seed.sql`, `gen_seed.py` | the database (Postgres 16). `seed.sql` is canonical: the tasks were checked against it. `gen_seed.py` is how it was made (its reviews block was not deterministic when the shipped file was generated, so a regeneration differs there; regenerate only together with re-authoring tasks) |
| `schema_prompt.py`, `prompt.txt` | the policy's system prompt: DDL + notes on what the data means + the one-query rule |
| `tasks.jsonl` | 417 tasks: `question`, gold `sql`, `archetype`, `difficulty`. 81 are held out by a hash of the id, the same split in every script |
| `author.py` | writes tasks for a schema with Claude Sonnet 5, executing every gold query twice before keeping it |
| `sql_verifier.py` | `SQLExec`, the verifier (a `zeroproof.simulations.verify.Verifier`): execution match, Spider-style. Also the task/split/row helpers and the in-container Postgres for the trainer |
| `rollout.py` | `zps.simulate(tasks=...)`: k samples per task on the account's hosted Qwen3-4B, a model you served with `zps.serve` (`--hosted`), Claude (callable agent), or any SDK agent spec (`--agent openai:...`) |
| `build.py` | grades every rollout file, pass@1 / pass^k / pass@k, `optimize(mode="rl")`, `hack_scan`, pushes train / holdout / eval sets to your account |
| `train_grpo_modal.py` | TRL `GRPOTrainer` + LoRA on Modal with Postgres inside the container (reward = `sql_verifier.shaped_reward`); `--from-run` chains rounds |
| `train.py` | the hosted SFT alternative: `zps.train(method="sft")` on the gold demonstrations, then `zps.serve` |
| `delta.py` | `delta_report` before vs after by difficulty and archetype, attached to the run page |

## Numbers so far (81 held-out tasks, 4 samples each, temperature 0.7)

| Model | pass@1 | 95% CI | pass@4 |
|---|---|---|---|
| Claude Sonnet 5 | 0.88 | 0.81..0.94 | 0.90 |
| Claude Haiku 4.5 | 0.72 | 0.61..0.81 | 0.75 |
| Qwen3-4B, thinking on | 0.69 | 0.60..0.77 | 0.81 |
| Qwen3-4B, thinking off | 0.58 | 0.48..0.68 | 0.64 |

Two things to read off that table before training anything. pass@4 minus
pass@1 is the headroom a grouped update can amplify: 0.06 with thinking
off, 0.12 with it on. And where the model loses: multi-table joins,
self-joins, date buckets, derived metrics (the archetype table `build.py`
prints).

Training the thinking-off model went nowhere, as the headroom predicted:
hosted SFT on the 336 gold demonstrations 0.58 -> 0.63 and GRPO with the
execution reward 0.58 -> 0.59, both intervals covering zero. The hill
climb is in thinking mode; rounds and their numbers are at the bottom.

## Run it

Needs: Python 3.11+, `pip install "zeroproof>=0.47" "psycopg[binary]" openai anthropic`,
a Postgres you can create a database on, `ZEROPROOF_API_KEY` from
[zeroproofai.com/platform](https://zeroproofai.com/platform) (the hosted
Qwen3-4B endpoint, the datasets page and the training page), and a Modal
account for the RL step.

**1. The database.** Any Postgres 16 works; the scripts read `T2S_PG_DSN`
(default `postgresql://postgres@127.0.0.1:5499/shop`).

```bash
docker run -d --name t2s -e POSTGRES_HOST_AUTH_METHOD=trust -p 5499:5432 postgres:16
psql -h 127.0.0.1 -p 5499 -U postgres -c "CREATE DATABASE shop"
psql -h 127.0.0.1 -p 5499 -U postgres -d shop -f schema.sql -f seed.sql
```

**2. Benchmark the base model.** Four samples per held-out task; rows
land in `raw/<model>.jsonl` and the run resumes if interrupted.

```bash
python rollout.py --model qwen3-4b --split holdout --k 4      # hosted Qwen3-4B, thinking off (the SDK's default for it)
python rollout.py --model sonnet-5 --split holdout --k 4      # ANTHROPIC_API_KEY, or AWS creds for Bedrock
python build.py                                              # grades, prints the tables, writes out/benchmark.md
python build.py --push                                       # also pushes the sets to your account
```

Rollouts go through `zps.simulate(agent, system_prompt=..., tasks=..., repeats=k)`:
the SDK replays the task prompts on the agent and the gold SQL is attached
to the rows afterwards. Thinking on for the base model: serve it under a
name and sample that (`zps.serve("qwen3-4b-think", base_model="Qwen/Qwen3-4B")`,
then `--hosted qwen3-4b-think`). Any other model: `--agent openai:<model>`
with `OPENAI_BASE_URL`, or add a callable to `MODELS`.

**3. Train, round one.** GRPO on Qwen3-4B, LoRA rank 16, 8 samples per
prompt, the execution reward (1.0 on a result match, 0.1 when the query
runs but is wrong, 0 otherwise). Postgres is installed in the image and
seeded in the container, so the reward needs nothing from your machine.

```bash
PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --run-name t2s-r1 \
  --thinking --skip-eval --steps 100 --gpu H100 --accum 4 --max-completion-length 1536
```

About 65 s a step in thinking mode on an H100 (completions average 900
tokens), so 100 steps is under two hours and roughly $8. `--skip-eval`
skips the slow in-container before/after sampling; the measurement comes
from the served adapter in the next step, through vLLM, in minutes. `--spawn`
submits the call and returns, so nothing depends on your laptop staying
connected (a network drop cancelled a 3 h run at step 49 without it). The run
shows on your training page as it goes (reward, KL, completion length); the
adapter and `summary.json` land on the `zeroproof-train-runs` volume under
the run id.

**4. Serve and measure.** The adapter is saved on the `zeroproof-train-runs`
volume under the run id, which is what `zps.serve` hosts.

```python
import zeroproof.simulations as zps

zps.serve("t2s-r1", "run_...")  # the run id printed by step 3
```

```bash
python rollout.py --hosted t2s-r1 --split holdout --k 4
python build.py
python delta.py --before hosted-qwen3-4b-think --after hosted-t2s-r1
```

`delta.py` prints the paired before/after with a 95% interval, by
difficulty and by archetype, and attaches it to the run page.

**5. Round two.** Start from round one's adapter, measure the same way.

```bash
PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --run-name t2s-r2 --from-run run_... \
  --thinking --skip-eval --steps 100 --gpu H100 --accum 4 --max-completion-length 1536
```

## Runs in parallel

Every run is its own Modal container with its own Postgres and its own
directory on the volume, keyed by run id, so launch as many as you like at
once and compare on the same holdout:

```bash
for lr in 1e-5 2e-5 5e-5; do
  PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --run-name t2s-lr$lr --learning-rate $lr \
    --thinking --skip-eval --steps 100 --gpu H100 --accum 4 --max-completion-length 1536
done
```

Then serve each, `rollout.py --hosted <name>`, and one `build.py`
prints them side by side. Knobs: `--learning-rate`, `--beta`, `--steps`,
`--num-generations`, `--loss-type` (bnpo, grpo, dr_grpo),
`--max-completion-length`, `--lora-rank`.

## Why the tasks are authored, not simulated

`zps.simulate` writes situations, rollouts and world state; it never writes
an answer key, so a verifiable task set is rows you bring that carry
`privileged.reference` (the SDK README says the same under *Verifiers*).
For SQL the reference has to be a query that is exactly right on the data,
and the question has to be unambiguous about which rows count and what to
return, or an exact-match verifier punishes valid readings. `author.py`
therefore has a teacher write the question and the gold together, executes
the gold twice, and keeps it only when it returns 1-50 stable rows. Once
the tasks exist, everything downstream is the SDK: `simulate(tasks=...)`
for rollouts, `data.grade(judge=SQLExec())`, `pass_at`, `optimize`,
`push_rows`, `training_run`, `serve`, `delta_report`.

## Bring your own schema

1. Replace `schema.sql` (DDL, comments welcome: the model reads them) and
   `seed.sql` (data; a few hundred rows is enough, the point is that the
   gold queries have answers).
2. Rewrite `NOTES` in `schema_prompt.py`: the formulas and NULL meanings a
   new analyst would need. Run `python schema_prompt.py` to refresh
   `prompt.txt`.
3. Load the database (step 1 above) and write tasks:
   `python author.py --rounds 2` (Claude Sonnet 5; `ANTHROPIC_API_KEY`, or
   AWS credentials for Bedrock; ~15 min and a few dollars for ~450 tasks).
   Every gold query is executed twice and must return 1-50 rows; near
   duplicates are dropped. Read a sample of `tasks.jsonl`: the questions
   must be unambiguous about which rows count and what to return, because
   the verifier is exact.
4. Steps 2-5 above, unchanged.

The verifier (`sql_verifier.py: SQLExec`, the same module the trainer
mounts) compares result sets as multisets, floats rounded to 2 places,
text case-folded, columns in any order, and in order only when the gold
query has ORDER BY. It reads the gold from `privileged.reference`, which
the training export never projects, so the answer key cannot leak into a
training file.

## What we learned building it

- **Headroom first.** pass@4 - pass@1 says whether RL has anything to
  amplify. At 0.06 nothing moved in 300 steps; at 0.12 it does.
- **Thinking off is the wrong regime for this model.** 0.58 vs 0.69 for
  the same weights.
- **`gradient_checkpointing=True` broke generation.** With TRL 0.19.1 and
  transformers 4.54 on Qwen3-4B, checkpointing makes TRL generate without a
  KV cache and every training completion was random tokens with reward 0,
  while a plain `model.generate` was fine. It is off here; memory is handled
  by micro-batches (`--accum 4` = 8 samples in four batches of 2) and an
  H100. An L40S OOMs at this prompt length.
- **Sampling at 0.7 gives half-duplicate groups.** `optimize(mode="rl")`
  drops them; the RL set from 8 samples on 336 prompts was 144 rows in 38
  groups, `pool_exhausted` in the scan. Train on the prompts, not the set.
- **Code-fenced replies read as truncated to `looks_finished` before 0.46**
  (zeroproof-sdk#212, fixed in 0.46); `build.py` carries the same patch so
  it also runs on 0.44.
- **Thinking models need a reply budget.** `simulate(agent_max_tokens=4096,
  timeout=300)` (zeroproof >= 0.47); on the default 2048-token cap and
  60 s timeout the base lost 8% of replies mid-thought and 4 of 81 tasks.

## The hill climb (thinking on, GRPO, execution reward)

Measured through `rollout.py --hosted <name>` (the SDK path, `agent_max_tokens=4096`,
`timeout=300`), 81 held-out tasks, 4 samples each, paired by task.

| Round | From | Steps | lr / beta | pass@1 (95% CI) | delta vs previous |
|---|---|---|---|---|---|
| base | Qwen/Qwen3-4B, thinking on | - | - | 0.61 (0.53..0.69), pass@4 0.84 | - |
| r1 | base | 100 | 2e-5 / 0.04 | 0.60 (0.52..0.69) | -0.003 (-0.062..+0.056), flat |
| r2 | r1 | 200 | 5e-5 / 0.01 | 0.62 (0.53..0.70) | +0.009 (-0.052..+0.071) vs base, flat |

Neither round moved the holdout, while the training reward did climb
(round 1 first-25-step mean 0.49 to last-25 0.63; round 2 up to 0.60-0.75
with KL 0.08), and thinking length fell from ~1,090 to ~800 tokens. That
combination means the policy got better at the prompts it was shown and no
better at held-out ones: 2,400 samples over 336 prompts, LoRA rank 16, is
too small a dose for a 4B model to generalize SQL reasoning from, and the
first thing GRPO learns is the cheap thing (shorter thinking, fewer
failures to emit a query: `has_sql` 0.87 -> 0.90). What the numbers say to
do next, in order: generate with vLLM inside the trainer (`use_vllm`,
colocate) so a round costs minutes instead of 65 s a step, then run
5-10 epochs over the prompts with 16 samples each; only then judge the
method. The table above is the product either way: every round is a
paired number with an interval on the same holdout, so "it got better" is
a claim the customer can check, and "it did not" is caught before anyone
ships it.
