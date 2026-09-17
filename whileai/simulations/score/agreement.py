"""Does the judge agree with labels you trust?

An LLM judge is a reward model, and a reward model is only as good as
its accuracy on a held-out set you labeled yourself (rlhf-book ch. 5
"Suggested Experiments": 50 to 200 pairs is enough to tune on). Without
that number a training run optimizes the judge's habits, not the
behavior. ``judge_agreement`` is that number, plus the two directions of
disagreement, which are not symmetric: a judge that passes a failure
teaches the failure (a reward hack in RL, a bad demonstration in SFT); a
judge that fails a pass only wastes a row.

The same function measures self-consistency: judge the same rows twice
and pass the second run as ``gold``. Agreement below what two humans
would reach is the ceiling on any drift alarm built on this judge.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

# Below this the accuracy estimate has a +/-0.1 error bar and the book's
# own guidance (50-200 held-out pairs) is not met.
MIN_GOLD = 50
# A judge that passes one in ten gold failures leaks that many bad rows
# into a training set at the pass rate of the run.
LEAK_THRESHOLD = 0.1


def _label(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value in (0, 1):
        return int(value)
    return None


def row_key(row: dict) -> str:
    """Stable identity for matching a judged row to its gold twin."""
    if row.get("rollout_id"):
        return str(row["rollout_id"])
    if row.get("scenario_id") is not None and row.get("rollout_index") is not None:
        return f"{row['scenario_id']}#{row['rollout_index']}"
    text = f"{row.get('prompt', '')}\x1f{row.get('final_text', '')}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    judge_pass = (tp + fp) / n
    gold_pass = (tp + fn) / n
    pe = judge_pass * gold_pass + (1 - judge_pass) * (1 - gold_pass)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def judge_agreement(
    rows: Sequence[dict],
    gold: str | Sequence[dict] = "gold_reward",
    *,
    reward: str = "reward",
) -> dict[str, Any]:
    """Agreement between the judge's ``reward`` and a trusted label.

    ``gold`` is either a key on the same rows (default ``gold_reward``,
    the field to fill when you hand-label a sample) or a second row list
    from another scoring pass, matched by rollout id, scenario id plus
    rollout index, or prompt plus final text. Only exact 0/1 labels on
    both sides count; partial scores and unjudged rows are reported as
    skipped, not guessed.

    Returns ``n``, ``agreement``, ``kappa`` (Cohen, chance-corrected), the
    confusion counts, ``pass_when_gold_fail`` (the leak rate: gold
    failures the judge passed) and ``fail_when_gold_pass``, both pass
    rates, and ``warnings``.
    """
    judged = [r for r in rows if isinstance(r, dict)]
    pairs: list[tuple[int, int]] = []
    skipped = 0
    unmatched = 0
    if isinstance(gold, str):
        for row in judged:
            j, g = _label(row.get(reward)), _label(row.get(gold))
            if j is None or g is None:
                skipped += 1
                continue
            pairs.append((j, g))
    else:
        by_key: dict[str, dict] = {}
        for row in gold:
            if isinstance(row, dict):
                by_key.setdefault(row_key(row), row)
        for row in judged:
            twin = by_key.get(row_key(row))
            if twin is None:
                unmatched += 1
                continue
            j, g = _label(row.get(reward)), _label(twin.get(reward))
            if j is None or g is None:
                skipped += 1
                continue
            pairs.append((j, g))

    tp = sum(1 for j, g in pairs if j == 1 and g == 1)
    fp = sum(1 for j, g in pairs if j == 1 and g == 0)
    fn = sum(1 for j, g in pairs if j == 0 and g == 1)
    tn = sum(1 for j, g in pairs if j == 0 and g == 0)
    n = len(pairs)
    leak = fp / (fp + tn) if (fp + tn) else None
    miss = fn / (fn + tp) if (fn + tp) else None
    warnings: list[str] = []
    if n == 0:
        warnings.append(
            f"no rows carry both {reward!r} and a gold label; hand-label a sample into "
            f"{gold!r} first"
            if isinstance(gold, str)
            else "no judged row matched a gold row by rollout id, scenario id, or prompt"
        )
    elif n < MIN_GOLD:
        warnings.append(
            f"{n} gold rows; the accuracy estimate is coarse below {MIN_GOLD} "
            "(rlhf-book ch. 5 suggests 50-200)"
        )
    if leak is not None and leak >= LEAK_THRESHOLD and fp:
        warnings.append(
            f"judge passed {fp} of {fp + tn} gold failures ({leak:.0%}); those rows train "
            "the failure, not the behavior"
        )
    return {
        "n": n,
        "n_skipped": skipped,
        "n_unmatched": unmatched,
        "agreement": round((tp + tn) / n, 4) if n else None,
        "kappa": _kappa(tp, fp, fn, tn),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pass_when_gold_fail": round(leak, 4) if leak is not None else None,
        "fail_when_gold_pass": round(miss, 4) if miss is not None else None,
        "judge_pass_rate": round((tp + fp) / n, 4) if n else None,
        "gold_pass_rate": round((tp + fn) / n, 4) if n else None,
        "warnings": warnings,
    }


__all__ = ["LEAK_THRESHOLD", "MIN_GOLD", "judge_agreement", "row_key"]
