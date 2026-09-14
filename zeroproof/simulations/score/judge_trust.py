"""Can the judge be trusted? The reward is only as good as the judge.

``judge_agreement`` (score/agreement.py) is the accuracy number against
labels you trust. This module is the rest of the trust report around it
(rlhf-book ch. 5: "do not let length influence your evaluation",
temperature 0 for stable ratings; ch. 12: review judge disagreements
against human labels, a second model agreeing is not proof; ch. 14: a
train/test split of the preference signal shows where optimization stops
transferring):

* **Agreement** with the ``gold_reward`` labels (0/1), from
  ``judge_agreement``, plus a Wilson interval on it.
* **Held-out halves**: the labeled rows split by task hash into two
  halves, agreement on each. Tune the rubric on one half and read the
  other; if they diverge the rubric is fit to its examples.
* **Length sensitivity** without re-judging: among rows humans called
  correct, judge pass rate on short versus long replies (split at the
  median), and the same among rows humans called wrong. A gap is a length
  bias the human labels rule out as real.
* **Perturbation** (needs the judge callable): re-judge a sample as-is
  for consistency, and once with neutral filler appended to the reply.
  Flips on the filler run mean the judge pays for length.

Rows the judge and the humans disagree on come back as a review queue.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from typing import Any

from .agreement import judge_agreement
from .hygiene import reply_length
from .stats import wilson_interval

GOLD_KEY = "gold_reward"
FILLER = " Let me know if there is anything else I can help with."
LENGTH_GAP_FLAG = 0.15
FLIP_FLAG = 0.10


def _label(row: dict, key: str) -> int | None:
    v = row.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f in (0.0, 1.0) else None


def _task(row: dict) -> str:
    return str(row.get("task_id") or row.get("scenario_id") or row.get("prompt") or "")


def _agreement(rows: Sequence[dict], gold: str) -> dict[str, Any]:
    out = judge_agreement(rows, gold)
    c = out["confusion"]
    out["ci95"] = wilson_interval(c["tp"] + c["tn"], out["n"])
    return out


def _half(task: str) -> int:
    return int(hashlib.sha256(task.encode("utf-8")).hexdigest()[:8], 16) % 2


def length_sensitivity(labeled: Sequence[dict], *, gold: str = GOLD_KEY) -> dict[str, Any]:
    """Judge pass rate on short vs long replies, within each gold label."""
    out: dict[str, Any] = {}
    max_gap = 0.0
    for label in (1, 0):
        rows = [r for r in labeled if _label(r, gold) == label]
        if len(rows) < 6:
            out[f"gold_{label}"] = {"n": len(rows), "note": "too few rows"}
            continue
        lengths = sorted(reply_length(r) for r in rows)
        median = lengths[len(lengths) // 2]
        short = [r for r in rows if reply_length(r) < median]
        long = [r for r in rows if reply_length(r) >= median]
        if not short or not long:
            out[f"gold_{label}"] = {"n": len(rows), "note": "no length spread"}
            continue
        rate_s = sum(_label(r, "reward") or 0 for r in short) / len(short)
        rate_l = sum(_label(r, "reward") or 0 for r in long) / len(long)
        gap = rate_l - rate_s
        max_gap = max(max_gap, abs(gap))
        out[f"gold_{label}"] = {
            "n": len(rows),
            "median_chars": median,
            "judge_pass_short": rate_s,
            "judge_pass_long": rate_l,
            "gap_long_minus_short": gap,
        }
    out["max_gap"] = max_gap
    out["flagged"] = max_gap >= LENGTH_GAP_FLAG
    return out


def _padded(row: dict) -> dict:
    out = dict(row)
    out["final_text"] = str(row.get("final_text") or "") + FILLER * 3
    messages = [dict(m) for m in (row.get("messages") or []) if isinstance(m, dict)]
    for m in reversed(messages):
        if str(m.get("role") or "") == "assistant":
            m["content"] = str(m.get("content") or "") + FILLER * 3
            break
    if messages:
        out["messages"] = messages
    return out


def perturbation(
    rows: Sequence[dict],
    judge: Callable[[dict], Any],
    *,
    sample: int = 40,
    seed: int = 0,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Re-judge a sample as-is (consistency) and with filler (length)."""
    from .judging import run_judge

    graded = [r for r in rows if isinstance(r, dict) and _label(r, "reward") is not None]
    rng = random.Random(seed)
    picked = graded if len(graded) <= sample else rng.sample(graded, sample)
    if not picked:
        return {"n": 0, "note": "no graded rows to re-judge"}
    again = run_judge(picked, judge, source="judge_trust", concurrency=concurrency)
    padded = run_judge(
        [_padded(r) for r in picked], judge, source="judge_trust", concurrency=concurrency
    )

    def flips(scored) -> tuple[int, int]:
        n = flipped = 0
        for original, rescored in zip(picked, scored.rows):
            a, b = _label(original, "reward"), _label(rescored, "reward")
            if a is None or b is None:
                continue
            n += 1
            flipped += a != b
        return flipped, n

    f_same, n_same = flips(again)
    f_pad, n_pad = flips(padded)
    consistency = (f_same / n_same) if n_same else None
    length = (f_pad / n_pad) if n_pad else None
    pad_up = sum(
        1
        for o, p in zip(picked, padded.rows)
        if _label(o, "reward") == 0 and _label(p, "reward") == 1
    )
    pad_down = sum(
        1
        for o, p in zip(picked, padded.rows)
        if _label(o, "reward") == 1 and _label(p, "reward") == 0
    )
    return {
        "n": len(picked),
        "consistency_flip_rate": consistency,
        "filler_flip_rate": length,
        "filler_flips_up": pad_up,
        "filler_flips_down": pad_down,
        "flagged_consistency": consistency is not None and consistency >= FLIP_FLAG,
        "flagged_length": length is not None and length >= FLIP_FLAG,
        "errors": sum(1 for r in again.rows + padded.rows if r.get("judge_status") != "ok"),
    }


def judge_trust(
    rows: Sequence[dict],
    judge: Callable[[dict], Any] | None = None,
    *,
    gold: str = GOLD_KEY,
    sample: int = 40,
    seed: int = 0,
    concurrency: int = 8,
) -> dict[str, Any]:
    """The judge-trust report. See the module docstring.

    ``rows`` carry the judge's ``reward``; rows that also carry ``gold``
    (0/1, default ``gold_reward``) feed the agreement, held-out, and
    length checks. Pass ``judge`` to add the perturbation checks, which
    call it on up to ``sample`` rows twice more.
    """
    rows = [r for r in rows if isinstance(r, dict)]
    labeled = [r for r in rows if _label(r, gold) is not None and _label(r, "reward") is not None]
    agree = _agreement(labeled, gold)
    halves = {
        "a": _agreement([r for r in labeled if _half(_task(r)) == 0], gold),
        "b": _agreement([r for r in labeled if _half(_task(r)) == 1], gold),
    }
    length = length_sensitivity(labeled, gold=gold)
    queue = [
        {
            "task": _task(r),
            "gold": _label(r, gold),
            "judge": _label(r, "reward"),
            "reason": str(r.get("reason") or "")[:200],
            "final_text": str(r.get("final_text") or "")[:200],
        }
        for r in labeled
        if _label(r, gold) != _label(r, "reward")
    ]
    perturb = (
        perturbation(rows, judge, sample=sample, seed=seed, concurrency=concurrency)
        if judge
        else None
    )

    warnings: list[str] = list(agree.get("warnings") or [])
    if agree["kappa"] is not None and agree["n"] and agree["kappa"] < 0.4:
        warnings.append(f"kappa {agree['kappa']:.2f}: judge and humans barely agree beyond chance")
    if halves["a"]["agreement"] is not None and halves["b"]["agreement"] is not None:
        gap = abs(halves["a"]["agreement"] - halves["b"]["agreement"])
        if gap >= 0.15 and min(halves["a"]["n"], halves["b"]["n"]) >= 10:
            warnings.append(
                f"agreement differs by {gap:.0%} between task halves; the rubric may be fit to "
                "the examples it was tuned on"
            )
    if length.get("flagged"):
        warnings.append(
            f"judge pass rate differs by {length['max_gap']:.0%} between short and long replies "
            "with the same gold label: length bias"
        )
    if perturb:
        if perturb.get("flagged_consistency"):
            warnings.append(
                f"{perturb['consistency_flip_rate']:.0%} of verdicts flip on an identical re-judge; "
                "set temperature 0 or add a second sample"
            )
        if perturb.get("flagged_length"):
            warnings.append(
                f"{perturb['filler_flip_rate']:.0%} of verdicts flip when neutral filler is appended "
                f"({perturb['filler_flips_up']} up, {perturb['filler_flips_down']} down): the judge "
                "reads length"
            )
    ok = not any(
        w.startswith(("kappa", "judge pass rate differs", "judge passed")) or "flip" in w
        for w in warnings
    )
    return {
        "ok": ok,
        "n_rows": len(rows),
        "n_labeled": len(labeled),
        "agreement": agree,
        "held_out_halves": halves,
        "length_sensitivity": length,
        "perturbation": perturb,
        "disagreements": queue,
        "warnings": warnings,
    }


def format_judge_trust(report: dict[str, Any]) -> str:
    a = report["agreement"]
    lines = ["PASS" if report["ok"] else "FAIL"]
    if a["n"]:
        ci = a["ci95"]
        kappa = f", kappa {a['kappa']:.2f}" if a["kappa"] is not None else ""
        lines.append(
            f"agreement {a['agreement']:.0%} (95% {ci[0]:.0%}..{ci[1]:.0%}, n={a['n']}){kappa}"
        )
        c = a["confusion"]
        lines.append(f"  confusion tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
        for name in ("a", "b"):
            h = report["held_out_halves"][name]
            if h["n"]:
                lines.append(f"  half {name}: {h['agreement']:.0%} (n={h['n']})")
    ls = report["length_sensitivity"]
    if ls.get("max_gap") is not None:
        lines.append(f"length gap {ls['max_gap']:.0%}" + ("  FLAG" if ls.get("flagged") else ""))
    p = report.get("perturbation")
    if p and p.get("n"):
        lines.append(
            f"re-judge flips {p['consistency_flip_rate']:.0%}, filler flips "
            f"{p['filler_flip_rate']:.0%} (n={p['n']})"
        )
    lines.append(f"disagreements to review: {len(report['disagreements'])}")
    for w in report["warnings"]:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = [
    "FILLER",
    "GOLD_KEY",
    "format_judge_trust",
    "judge_trust",
    "length_sensitivity",
    "perturbation",
]
