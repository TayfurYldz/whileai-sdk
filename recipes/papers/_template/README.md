# <Recipe name>

**Paper:** <title>, <first author et al.>, <arXiv id or venue>, <month year>. <link>
**Claim:** <one sentence: what the paper says happens, and why>
**The change:** <one line: what the recipe arm does that the baseline arm does not>

## Recipe

1. Base: `<model>`. Data: <dataset>, <n> train tasks, <n> held out by <rule>.
2. Baseline: <trainer>, <steps> steps, <batch> x <generations>, lr <x>, <key knobs>.
3. Recipe: baseline plus <the one change>.
4. Eval: <metric> on the same holdout, <k> samples per task, paired delta with a 95% interval.
5. <one more line if something matters, otherwise delete>

## Run

```bash
python recipe.py                         # both arms, ~<n> min on one <GPU>, ~$<x>
python recipe.py --arm recipe --steps 200  # one arm, longer
```

## Result

| Arm | <metric> | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | | | | 0 | 0 |
| Baseline | | | | | |
| Recipe | | | | | |

Recipe vs baseline: <+0.00 [lo, hi]>. Verdict: <moved / flat>.

## Climb

| Round | What changed | <metric> | vs previous |
|---|---|---|---|
| 1 | as the paper | | |

## Learned

- <what moved>
- <what did not>
- <what to try next>

Verified <YYYY-MM-DD>, zeroproof <version>, <trainer and version>. Run page: <url>
