# zeroproof

The ZeroProof Python SDK. One package, two importable modules:

- `zeroproof`: the platform client. OTLP trace ingest and trace-dataset listing against the token gate.
- `zeroproof.simulations`: the covering-grid simulator. (Was the separate top-level package `zeroproof_simulations`; that name still imports for two releases with a deprecation warning.) Register the agent — tools, policy, optional traces, optional `execute=` — and we spin the situations, sample pairwise, walk hard. That is the expensive part labs still do by hand. Their judge writes `r`. SFT / preference / GRPO-shaped export is the handoff to their trainer.

This repo absorbed the `zeroproof-simulations` package; `zeroproof-simulations` on PyPI is deprecated in favor of `zeroproof`.

Releases of `zeroproof` before 0.3 were an unrelated encrypted agent-to-agent messaging client. That code was removed in 0.04; pin `zeroproof<0.3` if you still depend on it.

Register the agent and we cover it. Tools and system prompt become the grid: this agent's tools, rules, stance, world state, tool condition, history. Traces, when you have them (`load_traces` → `trace_report` → `simulate(traces=...)`), steer the same grid at what already failed; `steering_weight` aims the budget; `evaluate(...).failed_traces()` is the next round on the same cards. `agent=` is their policy playing our cards — on-policy. Omit it and hosted Qwen walks the same grid, coverage first. The writer can still be Qwen. Every row is a full conversation: user turns, agent turns, tool calls, tool results, scheduled faults. Rows come back ungraded: honest, not a fake 0. Default `explore`: one unique situation per row. How it thinks: [docs/simulations.md](docs/simulations.md).

## How a row gets made

![How a row gets made: the draw, the coverage grid, the search arms, the rollout, the split](docs/how-a-row-gets-made.svg)

A situation is drawn across this agent's covering axes and a writer layer for how the person sounds. It fills a cell in the pairwise grid, nudges the five search arms, and the player walks it against a world that breaks on schedule. The row that comes out splits into `Task`, `Rollout`, `Judgment`, and `Marker`; `Judgment` lands when their judge writes `r`. Every training target is a projection of some of those four. The interactive version, running on real rows, is at [zeroproofai.com/docs/engine](https://zeroproofai.com/docs/engine).

## Overview

Register. Simulate. Grade after. Hand off.

1. **Register the agent.** Tools and system prompt. Optional traces. Optional `execute=`. That is the spec of the world.
2. **Build a world from those tools.** Objects, plausible results, and faults (timeout, deny, junk).
3. **Write users.** A separate writer (same hosted model, different prompt, no agent policy) samples this agent's covering axes: tool (their names, plus unrelated and multi_tool), rule, stance, world_state when the tools have referent keys, tool_condition, history.
4. **Pick the diverse ones.** Embeddings plus a bit of noise so the batch is not 200 copies of the same prompt.
5. **Walk hard.** `agent=` is their policy on our cards — on-policy play. Hosted Qwen is our walker when they want coverage first. Same grid. User text, agent text, tool calls, tool results, `final_text`. Default `grade=False`: rows stay ungraded. Ungraded is honest, not a fake 0.
6. **Grade after.** Authority stays with them. Their judge writes `r` with `data.grade(judge=...)`, `zps.run_judge(...)`, or `zps.grade(...)`. `evaluate(...)` is the same contract with lineage `source=eval` (vs `source=grade`), so evals stay distinct from training rewards. A broken judge stays ungraded. Structural flags (`grade=True`) are display, not the score.
7. **Hand off.** `select_for_sft`, `select_for_rl`, `export_training`, `export_preference`, `pass_at` — SFT / preference / GRPO-shaped for their trainer (Prime, TRL, their cluster). Training math lives there; we supply the on-policy (or covering) trajectories and the judge contract. The live loop is the same cards, stepped: they act, the world answers, their judge scores, they update.

Stop when the row cap or the clock hits.

## How to use

```bash
pip install zeroproof   # or: uv add zeroproof
```

`agent=` is their policy on our cards. Any OpenAI-compatible chat endpoint
that returns tool calls works. That play is on-policy for that student.
The situation writer still defaults to hosted Qwen unless you point
`simulator=` at your endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # only for a non-OpenAI endpoint
```

```python
import zeroproof.simulations as zps

data = zps.simulate(
    agent="openai:gpt-4.1-mini",
    tools=my_tools,
    system_prompt=my_system_prompt,
    output="rollout.jsonl",
)
```

Omit `agent=` and hosted Qwen is our walker — same grid, coverage first.
Ask us for a `VLLM_API_KEY`; the endpoint is shared and rate limited.

```bash
export VLLM_API_KEY=...
```

No key at all: the situation writer also defaults to hosted Qwen, even when
`agent=` is your own function. Pass `simulator=False` to use the built-in
template writer instead. It needs no model and runs in seconds; the
situations are less varied than a model writes, so it is for wiring up your
agent and grader, not for a training set.

```python
data = zps.simulate(
    my_agent, tools=my_tools, system_prompt=my_system_prompt, simulator=False, budget=40
)
```

Working in this repo: `uv sync`, then `uv run pytest` after `uv sync --extra dev`.

One runtime dependency (`requests`), Python 3.10+. Installing from PyPI rather than a
path or a git URL matters if you build a Prime Intellect environment on this:
the Environments Hub installs a pushed env with plain pip, so a `[tool.uv.sources]`
git pin resolves locally and then fails on their runtime with a
`ModuleNotFoundError`.

```python
import zeroproof.simulations as zps

data = zps.simulate(tools=my_tools, system_prompt=my_system_prompt, output="rollout.jsonl")
data = zps.simulate(agent=my_agent)
```

Traces first when you have them. `steering_weight` aims at failures; the next
round is the same cards, stepped — `simulate(traces=evald.failed_traces())`:

```python
traces = zps.load_traces(trace_source)
print(zps.trace_report(traces, tools=my_tools, policy=my_system_prompt))
data = zps.simulate(
    agent=my_agent, tools=my_tools, system_prompt=my_system_prompt, traces=traces
)
scored = data.grade(judge=my_judge)           # lineage source=grade
evald = zps.evaluate(holdout, judge=my_judge)  # same contract, source=eval
sft, _ = scored.select_for_sft()
zps.export_training(sft, output="train.jsonl", system_prompt=my_system_prompt, tools=my_tools)
```

A second judge — `audit_grades` in `zeroproof.simulations.score.grade_llm` —
can recommend rubric and eval holes later. Advice, never `r`. It does not run
on `simulate`.

Pass `spec=` if you have a local tools-and-system-prompt folder. The generated datasets are on [Hugging Face](https://huggingface.co/datasets/zero-proof-ai/agent-simulations), organized by agent type instead of stored in this repo.

| Knob | Default | |
|---|---|---|
| `agent` / `spec` | hosted Qwen | Callable, URL, or tools + system prompt |
| `budget` / `time_budget` | `1000` / `None` | Stop when either hits. The clock is off unless you set it; `0` or `None` keeps it off |
| `requests_per_situation` | from mode | Phrasings: ways to ask one situation. Alias `phrasings=` |
| `rollouts_per_request` | from mode | Repeats: reruns of one phrasing. Alias `repeats=` |
| `fault_rate` | `0.5` | Broken tools. `0` off. Applied by the mock world, so a callable `agent=` that answers its own tool calls never sees one |
| `simulator` | hosted Qwen | Situation writer. `False` uses the built-in template writer (no model, less variety); an `openai:`/`vllm:` spec runs it on your endpoint |
| `reproducible` | `False` | Same seed, same concurrency, same agent: same rows. Runs batch by batch, so uneven latency costs throughput. Needs the clock off |
| `grade` | `False` | Ungraded until their judge writes `r` (`data.grade(judge=)`, `zps.run_judge`, `zps.grade`). `grade=True` is structural flags, not the score |
| `llm_grade` | `False` | Extra LLM pass on the row. Their judge still writes `r` |
| `output` | | JSONL path |

## What to run

Depends on the use case. How each scenario is built is in [The recipe](#the-recipe).

| You want | Mode | What happens |
|---|---|---|
| Many distinct situations | `explore` (default) | New situation every row |
| Same situation, different wording | `sft` | Multiple phrasings: tone, intent, personality |
| Same request, different agent behavior | `rl` | Multiple repeats of one phrasing |
| A mix, until coverage plateaus | `adaptive` | New situations, phrasings, and repeats. Best with `until="saturation"` |

```python
zps.simulate(tools=my_tools, system_prompt=my_system_prompt)  # explore
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="rl")
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="adaptive", until="saturation")
```

## Examples

| Example | What it does |
|---|---|
| [`examples/agent-behavior`](examples/agent-behavior) | Start here if the platform is new to you. Runs a coding agent with bad habits against real tests, streams every turn to Zero Proof as OTLP spans plus a judge verdict, and fills a dashboard with behaviour worth looking at. No dependencies. |
| [`examples/prime-intellect-rl`](examples/prime-intellect-rl) | Generates a GRPO-ready dataset with `simulate(mode="rl")` and checks it carries gradient before you spend GPU time on it. |
| [`examples/schema`](examples/schema) | One row file in, six training targets out: eval, SFT, preference, GRPO prompts, OPSD hints, OPD. Migrates any legacy file first. Offline, no key. |
| [`examples/identity`](examples/identity) | Builds a leak-free SFT set that teaches a model a new name and maker, with Modal scripts to train a LoRA and evaluate it. No model calls to generate. |

## Sign in

```bash
zeroproof login
```

Prints a link and a short code. Open the link, sign in or sign up, press
Approve. The key is saved to `~/.zeroproof/credentials.json` and every
platform call below reads it from there. Interrupted before you approved?
Run it again; it resumes the same code. This is the path for coding
agents too: tell yours to run `zeroproof login` and click the link it
shows you. `zeroproof status` shows which key is in use, `zeroproof
logout` removes it.

No account yet, or no browser? One command creates the account and the
key. Open the dashboard later by signing in with an email code.

```bash
zeroproof signup --email you@example.com
```

## Store datasets on Zero Proof Labs

Push a run to your Zero Proof Labs account to store and version it.
Credentials resolve in this order: `api_key=` argument,
`ZEROPROOF_DELEGATED_CREDENTIAL` (a short-lived `zp_dc_...` issued from a
Clerk session), `ZEROPROOF_API_KEY`, then the key saved by `zeroproof
login`.

```python
# Runtime path with a delegated credential
# export ZEROPROOF_DELEGATED_CREDENTIAL="zp_dc_..."

# If you need to mint one from a Clerk session token:
# credential = zps.issue_delegated_credential(clerk_token, ttl_seconds=3600)
# export ZEROPROOF_DELEGATED_CREDENTIAL=credential["credential"]

data = zps.simulate(spec="specs/github")
v1 = data.push("github-explore-v1")  # -> {"datasetId": "ds_...", ...}

# iterate, then push the next version with lineage
v2 = data.push("github-explore-v2", parent=v1["datasetId"])

zps.datasets()  # list yours + storage used
rows = zps.pull(v1["datasetId"])  # rows, or pass path= for a file
zps.push_file("rollout.jsonl")  # upload an existing JSONL
zps.delete_dataset(v1["datasetId"])  # permanent
```

Storage is private per account, 5 GB free. `parent=` records dataset
lineage so iterations show as a family on the platform.

## Speed

Two-minute airline runs using ZeroProof-hosted Qwen. Results were measured on the
hosted GPU with warm replicas and burst under load.

| Mode | Rows | Rate | Unique openers |
|---|---|---|---|
| `explore` | 240 | 120/min | 240 |
| `sft` | 278 | 139/min | 278 |
| `rl` | 625 | 296/min | 209 |

## Parameter reference

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | | Local tools and system prompt path |
| `tools`, `system_prompt` | from spec or agent | Tool list and agent system prompt |
| `situations` | | Distinct situations (N) |
| `requests_per_situation` | from mode | Phrasings per situation (n). Alias `phrasings=` / `n=` |
| `rollouts_per_request` | from mode | Repeats per phrasing (k). Alias `repeats=` |
| `unique_situations` | on in `explore` | Unique situations only |
| `mode` | `"explore"` | `explore`, `sft`, `rl`, `adaptive` |
| `reproducible` | `False` | Round-synchronous scheduling; see the knob table |
| `budget` | `1000` | Row cap |
| `time_budget` | `None` | Seconds. Off by default; `None` or `0` disables |
| `until` | `"compute"` | `"saturation"` also stops when coverage plateaus |
| `grade` | `False` | Ungraded until their judge writes `r`. `grade=True` is structural flags, not the score |
| `llm_grade` | `False` | Extra LLM pass on the row |
| `output` | | JSONL path |
| `advanced` | | Keys below |

| `advanced` key | Default | |
|---|---|---|
| `concurrency` | `32` | Parallel rollouts |
| `stop_grace` | `5` | Seconds to wait for running rollouts and writer waves after a stop; queued ones are cancelled, still-running ones are reported as `rollouts_abandoned` / `writer_waves_abandoned` |
| `embedder` | `"hash"` | Prompt selection |
| `seed` | `0` | Reproducible draws. Bit-for-bit at `concurrency: 1` or with `reproducible=True`; otherwise which rows land before the cap depends on thread timing |
| `avg_turns` | `4` | Target conversation length |

Aliases: `phrasings=` / `n=` → `requests_per_situation`; `repeats=` → `rollouts_per_request`; `unique=` → `unique_situations`; `policy=` → `system_prompt`; `risk=` → `fault_rate`.

## Output

Each row, in `data.trajectories` and on disk: `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. Optional `world_state`, `faults`, `reward`, `reason`. Ungraded means no `reward` — honest, not a fake 0. `llm_grade=True` adds `llm_reward`. `zps.rank(path)` adds `quality` without changing `reward`. After their judge, hand off with `select_for_sft`, `select_for_rl`, `export_training`, `export_preference`; `pass_at` measures the groups their trainer will see.

After grading, `data.pass_at` (also on the `ScoredData` from `judge=` and `evaluate`) gives pass@1, pass^k and pass@k off the same groups, one job each: pass@1 is the measurement headline (the agent runs once in production), pass^k is the reliability line (all k repeats pass), and pass@k minus pass@1 (`.headroom`) is what a grouped RL update has to learn from, the same asks `group_signal` counts as mixed. k is the smallest group of repeats; below `repeats=4` the k-way numbers are `None` with a note rather than a noisy figure. `.per_task` is the raw per-prompt pass-rate vector. With an LLM judge, pass@k inflates on false positives and pass^k on false negatives, so pass@1 stays the headline.

```python
scored = data.grade(judge=my_judge)
print(scored.pass_at)  # pass@1 0.61 | pass^8 0.32 | pass@8 0.88 | headroom 0.27 (200 groups, k=8)
```

Every row carries `schema_version` (`"1"`). A row is a projection of four objects in `zeroproof.simulations.schema`: `Task` (the situation), `Rollout` (one episode), `Judgment` (a scorer's verdict), `Marker` (a behavior measurement). `zps.from_row(row)` splits a row into them and `zps.to_row(...)` flattens them back. The wire contract is `zeroproof/simulations/schemas/row-v1.json`. Rows written before the stamp are version 0 and load by shape, so older files still work.

## The recipe

Each scenario is a draw across covering axes from **this** agent.

**Covering** (pairwise; `data.coverage["pairwise"]`)

- tool: this agent's names, plus unrelated and multi_tool
- rule: clauses from the system prompt
- stance: ordinary, retry, adversarial, and the rest of that axis
- world_state: exists / missing / already acted on — only when the tools have referent keys
- tool_condition: success, timeout, deny, stale, malformed
- history: fresh, prior_failure, and the rest of that axis

**Writer-only** (not covering): length, vagueness, tone, texture.

Ordinary asks first, then the edges. On top of that, we embed the openers and add a bit of random noise so the batch stays spread out, not a cluster of near-copies. Spend the row cap and the clock on diversity, not copies. Modes: `explore` / `sft` (phrasings) / `rl` (repeats) / `adaptive` until saturation.

## Package layout

The public surface is the package itself: `import zeroproof.simulations as zps`.
Internals are grouped by stage and may move between releases.

| folder | what lives there |
|---|---|
| `generate/` | situation grid, writer, diversity selection, agent runners and adapters |
| `score/` | judge contract, structural flags, quality ranking, SFT / preference / GRPO-shaped handoff |
| `ingest/` | trace loading, OpenTelemetry rows, platform push and pull |
| `world/` | the mock tool environment |
| `run/` | the engine behind `simulate()`: knob resolution (`config.py`), spec loading (`spec.py`), row helpers (`rows.py`), and the scheduler itself (`engine.py`: inputs, build, loop, finish) |
| `simulation.py`, `data.py`, `export.py` | the `simulate()` entry point, its result object, and training export |

## Development

```bash
uv sync --extra dev
uv run pytest           # about two minutes, no network
uv run ruff check .     # lint; `--fix` for the mechanical ones
uv run mypy             # type check
pre-commit install      # optional: ruff and whitespace hooks on commit
```

CI runs the suite on Python 3.10 through 3.13, ruff, mypy, and a plain-pip
install of the built wheel into a clean venv.

## License

Apache-2.0
