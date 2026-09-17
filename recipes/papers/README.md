# Papers

One directory per paper. A recipe here is a recent post-training paper's idea
cut down to a run that fits in under an hour on one GPU, with the number it
moved and the number it did not. The building blocks are the step recipes next
door (`../04-train/grpo`, `../04-train/dpo`, `../04-train/text-to-sql`,
`../04-train/hosted-loop`); a paper recipe copies one of them and changes one
thing.

Every recipe answers the same five questions in the same order: which paper,
what it claims, the steps, one command, what happened.

<!-- table:start -->
| Recipe | Paper | Base | Metric | Baseline -> Recipe | Verified |
|---|---|---|---|---|---|
| [adaptive-clip](adaptive-clip) | [2609.00444](https://arxiv.org/abs/2609.00444) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.00 -> 0.00 (+0.00 [+0.00, +0.00], flat) | never run |
<!-- table:end -->

The table is generated: `python recipes/papers/check.py --write` reads every
`results.json`. Do not edit it by hand.

## Run one

```bash
pip install zeroproof modal
export ZEROPROOF_API_KEY=...        # run page + datasets at zeroproofai.com/platform
modal token set --token-id ... --token-secret ...
cd recipes/papers/<slug>
python recipe.py                    # both arms, writes results.json
```

## The contract

- `README.md` in the shape of [`_template/README.md`](_template/README.md): Paper, Claim, The change, numbered steps, one command, the Result table, the Climb table, three Learned bullets, the Verified line.
- `recipe.py`: one file. Data, then train, then eval, then `results.json`. Two arms on the same holdout: the baseline and the paper's change. Paired delta with a 95% interval (`zps.delta_report`).
- `results.json`: the numbers the table above reads. Shape in [`_template/results.json`](_template/results.json).
- Default run: under 60 GPU minutes, under $10. Bigger runs behind a flag.
- Public data or a seeded environment that lives in the recipe directory. No customer data.
- A flat result is a result. Say so in the table.
- `python recipes/papers/check.py --write` passes (`tests/recipes/test_papers.py` runs it in CI).

## Maintenance

A daily agent re-runs the recipe with the oldest verified date, refreshes its
numbers, fixes what broke, and adds one new recipe from research published in
the last 60 days. Everything arrives as a pull request. One comment per run on
the issue titled "Recipe log". Several agents can work at once: each recipe is
its own directory and the table is generated, so two new recipes never touch
the same line.
