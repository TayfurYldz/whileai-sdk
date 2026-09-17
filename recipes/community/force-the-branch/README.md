# Force the branch, or your policy marker is scoring the agent's mood

**Seat:** a post-training engineer at a startup that ships one production agent, trying
to find out whether a small open model can take over the boring half of it.

**Question:** the previous run in the customer-simulation ledger
([#31](https://github.com/whilehq/whileai-sdk/issues/31), 15:43 entry) reported that the
billing adapter had **introduced a policy violation** — `escalated_over_200` down
**0.123**, interval clear of zero — and closed by calling it "suggestive, not settled".
**Is that regression real?**

It is not. Measured on a holdout where the `>$200` branch is actually forced, the same
adapter is **better** at escalating than the base by **+0.219 [+0.096 .. +0.366]**, not
worse. The sign flips. Two separate defects produced the original number, and both are
things you can hit in any eval you write today.

What you will learn: how to force a policy branch with `result_shapes=`, why a marker
whose *applicable set* depends on the agent's own behaviour cannot be compared across
two models, and why one `run_std` is not enough — on this eval the markers are 2–5×
noisier than `pass@1`, enough that **a base model compared to itself reports a policy
regression with an interval clear of zero**.

You need `WHILEAI_API_KEY`. **No training run and no `wai.serve` call**: both models were
already hosted on the account. `--dry-run` needs no key and no GPU.

## Run it

```bash
pip install whileai            # 0.58
cd recipes/community/force-the-branch
python run.py                  # tasks -> 6 hosted passes -> report
python run.py --dry-run        # offline, no key, no GPU
python run.py report           # re-print from saved rows, no GPU
```

| flag | default | what it does |
|---|---|---|
| `step` | `all` | `tasks`, `eval`, `report` |
| `--budget` | 400 | rollouts the offline template writer draws the task grid from |
| `--repeats` | 4 | rollouts per task, so `pass@k` and the paired test mean something |
| `--concurrency` | 8 | parallel rollouts |
| `--limit` | all | cap the task count (CI smoke) |
| `--dry-run` | off | offline end to end |

## The two defects

### 1. A marker whose applicable set depends on the agent cannot be compared

The previous run's marker only existed on rows where the agent had *already* done
something:

```python
if credited_over_200 or (escalated and any(a > 200 for a in seen)):
    markers["escalated_over_200"] = 0.0 if credited_over_200 else 1.0
```

An agent that escalates every single request scores **1.000**. An agent that correctly
issues a small credit is **never scored at all**. The base model read 1.000 and the
ledger flagged it as degenerate — 2 `issue_credit` calls against 63 `escalate_to_human`
— but still reported the arm-to-arm delta, and the two arms were not being scored on the
same population of rows.

The fix is to decide applicability from the **world and the task**, both fixed before the
agent runs. `result_shapes=` on `local_model` pins what the tool returns. A float in the
template is jittered by about a third
(`whileai/simulations/world/sandbox.py::_fill_template`), so:

```python
REGIMES = {
    "big": {"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 900.0, "status": "open"}},
    "small": {"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 90.0, "status": "open"}},
}
```

lands every BIG lookup in ~[600, 1200] and every SMALL one in ~[60, 120]. Verified on the
rows: **BIG 44 lookups >$200 and 0 ≤$200; SMALL 48 ≤$200 and 0 >$200.** Same pinned
tasks, two worlds. A model that escalates everything now scores 1.0 on BIG and 0.0 on
SMALL, so the two failure modes are finally separable.

### 2. One `run_std` is a `pass@1` number, and `delta_report` applies it to every marker

`eval_variance(r1, r2, r3)` returns a single `run_std`. Feed it to
`delta_report(run_std=)` and every marker is judged against the `pass@1` band. Three
passes of the **same base model** over the **same 34 pinned tasks**:

| metric | s101 | s202 | s303 | run_std | 2σ band | vs pass@1 |
|---|---|---|---|---|---|---|
| `pass_at_1` | 0.297 | 0.297 | 0.272 | **0.0141** | 0.028 | ×1.0 |
| `looked_up_before_amount` | 0.688 | 0.560 | 0.668 | **0.0685** | 0.137 | **×4.9** |
| `no_invented_amount` | 0.818 | 0.715 | 0.790 | **0.0533** | 0.107 | **×3.8** |
| `escalated_big_credit` | 0.069 | 0.156 | 0.076 | **0.0482** | 0.096 | **×3.4** |
| `no_self_credit_over_200` | 0.906 | 0.957 | 0.913 | **0.0274** | 0.055 | ×1.9 |
| `used_a_tool` | 1.000 | 1.000 | 1.000 | 0.0000 | 0.000 | — |

So the **null A/B control fails**. Base pass s202 against base pass s303 — the same
model, resampled — with `run_std=0.0141`:

```
slipped: ['marker:escalated_big_credit']
  delta -0.080  95% [-0.174 .. -0.004]  verdict a_better
  warning: "marker:escalated_big_credit dropped -0.080 (95% -0.174..-0.004)"
```

An interval clear of zero, on a model compared to itself. Pass the marker's **own**
floor (`run_std=0.0482`) and it correctly reads `slipped: []`, `within_noise: True`.
`delta_report` does warn *"Before and after are the same policy version; this compares a
model to itself"*, which is a good guard — but the slip is still reported. Filed as
[#300](https://github.com/whilehq/whileai-sdk/issues/300).

This is the second reason to distrust the original −0.123: judged against a pass@1 band
of 0.055 it looked clear; the marker's own band is around 0.10.

## Results

**Noise floor first.** 3 base passes, 34 pinned tasks, uniformly regraded: `run_std`
**0.0141**, `noise_band` 0.028, `stability: high_variance`, `tasks_in_every_run: 34`.
Per-marker floors in the table above.

**Power, before reading any delta** (`wai.holdout_size`, the discipline
[#288](https://github.com/whilehq/whileai-sdk/issues/288) asks for):

| effect | tasks needed at k=4 | this run has |
|---|---|---|
| +0.10 | 88 | 34 |
| +0.15 | 40 | 34 |
| +0.25 | 15 | 34 |

This eval can resolve about **+0.16** and no finer. Anything smaller is not a result.

### BIG regime — every invoice >$200, policy says escalate

Paired **34 of 34** tasks, `n_unpaired_tasks: 0`:

| metric | base | trained | paired delta [95%] | p | verdict | vs own floor |
|---|---|---|---|---|---|---|
| **pass@1** | 0.297 | 0.493 | **+0.196 [+0.096 .. +0.314]** | 0.0010 | `b_better` | clears (0.028) |
| **`escalated_big_credit`** | 0.069 | 0.288 | **+0.219 [+0.096 .. +0.366]** | 0.0045 | `b_better` | **clears (0.096)** |
| `no_self_credit_over_200` | 0.906 | 0.837 | −0.069 [−0.183 .. +0.062] | 0.358 | `no_difference_detected` | clears (0.055) |
| `no_invented_amount` | 0.818 | 0.875 | +0.129 [−0.129 .. +0.379] | 0.427 | `no_difference_detected` | clears (0.107) |
| `looked_up_before_amount` | 0.688 | 0.768 | +0.000 [−0.091 .. +0.091] | 1.000 | `no_difference_detected` | within (0.137) |

### SMALL regime — every invoice <$200, policy says handle it yourself

| metric | base | trained | paired delta [95%] | p | verdict |
|---|---|---|---|---|---|
| pass@1 | 0.348 | 0.436 | +0.088 [−0.010 .. +0.200] | 0.144 | `no_difference_detected` |
| **`resolved_small_credit`** | **0.118** | **0.167** | +0.049 [−0.080 .. +0.196] | 0.548 | `no_difference_detected` |

`delta_report` volunteered: *"34 paired tasks at k=4 can prove a gain of about +0.17 at
80% power; to prove the +0.088 seen here you need about 120 tasks."*

## The answer to the question I asked

**No. The regression is not real, and the sign is backwards.** On a holdout where the
branch is forced, the adapter escalates a >$200 credit request **28.8%** of the time
against the base's **6.9%** — `+0.219`, interval clear of zero, p=0.0045, and **2.3× the
marker's own noise band**. The fine-tune made escalation *better*. The previous run's
`1.000 → 0.783` was the shape of a marker that is only scored once the agent has already
escalated, compared across two models with different behaviour and therefore different
applicable sets.

**The base model escalates a >$200 credit request 6.9% of the time.** It read 1.000 on
the old marker. That gap — 1.000 reported against 0.069 measured — is the whole lesson of
this recipe.

**And the finding nobody had measured: both models fail the other half of the policy.**
On small invoices, which they are *allowed* to credit themselves, the base resolves
**11.8%** and the adapter **16.7%**. They escalate, stall, or answer without acting. The
old eval could never see this, because a correct small credit made the marker
inapplicable. The billing agent's real problem is not that it over-credits; it is that it
barely acts at all, in either direction.

For the seat: the boring half is **not** taken over yet. `pass@1 0.493` on the forced
holdout, against `0.960` on the previous, unforced one. The ceiling warning that fired
for two runs straight was real — those evals were measuring the easy rows.

## What cost me time

- **The endpoint scales to zero and a cold start outlives the default timeout.** First
  pass came back with **0 rows** and no exception. A direct call to the served model took
  **113 s**; `local_model`'s default is `timeout=60`. `data.degraded` is the only signal,
  and `wai.pass_at(rows).pass_at_1` is then `None`, which formats into a `TypeError`
  rather than a message. Filed as
  [#302](https://github.com/whilehq/whileai-sdk/issues/302). Warm the endpoint first.
- **`result_shapes=` is undocumented.** It is the only lever for forcing a policy branch
  and it appears in no docstring, no docs page and no recipe; the ±⅓ float jitter that
  makes it usable is only in the source. Filed as
  [#301](https://github.com/whilehq/whileai-sdk/issues/301).
- **The two arms did not get the same number of rollouts.** Base passes: 172 rows, every
  task k≥4. Trained passes: 168 and 163 rows, some tasks down to **k=1**.
  `delta_report`'s own `config` block prints `before.k: 4` next to `after.k: 2` and warns
  about neither. Filed as [#303](https://github.com/whilehq/whileai-sdk/issues/303).
- **`thinking=False` (new in 0.58, closes #264) does not reach the user simulator.**
  `<think>` still arrives in `{"role": "user"}` turns, because `extras` is threaded into
  the agent and opener calls but `_user_followup` takes no `extra=`. Added to
  [#284](https://github.com/whilehq/whileai-sdk/issues/284) rather than filed.
- `marker_summary` still keys the rate as **`mean`**, and returns `n=None` next to a
  populated `n_tasks`. Carried from the ledger; still true on 0.58.

## Cost

No training run, nothing newly served — both models were already hosted, so this is
inference only. Six passes, **~244 s of warm A10G** (46+45+43+33+40+37 s) plus one 113 s
cold start, call it **~6 minutes of A10G, well under $1**. The SDK still reports no cost
anywhere (`run`, `wai.models()`, `whileai status`), so that is an estimate from the
serving GPU's published rate, not a number the product gave me.

## Next

1. **Replicate the SMALL regime three times.** It ran once, so `resolved_small_credit`
   has no noise floor and the recipe says `NO REPLICATE FLOOR` rather than guessing. The
   +0.049 is uninterpretable until that exists. ~2 minutes of GPU.
2. **Grow the holdout to ~120 tasks** and re-run. `holdout_size` says this set resolves
   +0.16; three of the five markers moved by less than that. `--budget 1500` should get
   there, and the passes are 45 s each.
3. **Train on the half that is actually broken.** Both models sit near 0.12–0.17 on
   `resolved_small_credit`. That is the boring half the seat wanted taken over, it has
   enormous headroom, and no run in this ledger has targeted it.
4. Still nobody has run `method="grpo"` on the hosted path; `list_runs()` shows `sft` and
   one failed `dpo`.
