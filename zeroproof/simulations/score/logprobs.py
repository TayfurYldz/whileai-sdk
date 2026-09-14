"""What the policy's own log-probabilities buy you.

``simulate(logprobs=True)`` asks the rollout model for the log-probability
of every token it generated and stamps each agent turn with the sum and
the token count (``step["logprob"]``, ``step["n_tokens"]``), and the row
with the totals. Two things need them (rlhf-book ch. 6 on off-policy
correction, ch. 15 on the KL penalty):

* **Importance ratios.** A trainer that updates on rollouts sampled from
  an older policy corrects with ``exp(new_logprob - logprob)``; without the
  sampling-time logprob the ratio cannot be formed and the update is
  silently off-policy.
* **KL to a reference.** The sampled estimate of ``KL(pi || pi_ref)`` is
  the mean over generated tokens of ``log pi - log pi_ref``. Score the
  same rows under the reference (``ref_logprob``) and ``mean_kl`` gives the
  per-task number ``Calibration.mean_kl`` was declared for.

``logprob_report`` is the pre-flight: how much was captured, how confident
the policy was, and whether confidence predicts reward, which on a judge
that pays for fluency it should not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .agreement import row_key
from .hygiene import pearson


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _tokens(row: dict) -> int:
    n = row.get("n_tokens")
    return int(n) if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0


def _task(row: dict) -> str:
    return str(row.get("task_id") or row.get("prompt") or "")


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[pos], 4)


def logprob_report(rows: Sequence[dict]) -> dict[str, Any]:
    """Coverage and shape of the captured logprobs.

    ``mean_token_logprob`` is total logprob over total tokens. The
    per-row quantiles are of each row's own mean, so one long rollout
    does not dominate. ``corr_reward_confidence`` is Pearson between the
    0/1 reward and the per-row mean over graded rows: a strong positive
    value says the judge rewards fluency, not behavior.
    """
    judged = [r for r in rows if isinstance(r, dict)]
    with_lp = [r for r in judged if _num(r.get("logprob")) is not None and _tokens(r) > 0]
    total_lp = sum(float(r["logprob"]) for r in with_lp)
    total_tok = sum(_tokens(r) for r in with_lp)
    means = [float(r["logprob"]) / _tokens(r) for r in with_lp]
    graded = [
        (float(r["reward"]), float(r["logprob"]) / _tokens(r))
        for r in with_lp
        if r.get("reward") in (0, 1) and not isinstance(r.get("reward"), bool)
    ]
    corr = pearson([g[1] for g in graded], [g[0] for g in graded]) if len(graded) >= 3 else None
    truncated = sum(
        1
        for r in with_lp
        if any(isinstance(s, dict) and s.get("truncated") for s in r.get("steps") or [])
    )
    warnings: list[str] = []
    if judged and not with_lp:
        warnings.append(
            "no row carries logprob/n_tokens; run simulate(logprobs=True) with a model "
            "backend (a callable agent= must stamp its own)"
        )
    if corr is not None and corr >= 0.3:
        warnings.append(
            f"reward tracks the policy's confidence (r={corr:.2f}); a judge that pays for "
            "fluency is a reward hack (rlhf-book ch. 14)"
        )
    if with_lp and truncated:
        warnings.append(f"{truncated} rollout(s) hit the token cap; their logprob is partial")
    return {
        "n_rows": len(judged),
        "n_with_logprobs": len(with_lp),
        "n_tokens": total_tok,
        "mean_token_logprob": round(total_lp / total_tok, 4) if total_tok else None,
        "row_mean_p10": _quantile(means, 0.1),
        "row_mean_p50": _quantile(means, 0.5),
        "row_mean_p90": _quantile(means, 0.9),
        "corr_reward_confidence": round(corr, 4) if corr is not None else None,
        "n_truncated": truncated,
        "warnings": warnings,
    }


def mean_kl(rows: Sequence[dict], ref: str | Sequence[dict] = "ref_logprob") -> dict[str, Any]:
    """Sampled ``KL(pi || pi_ref)`` per generated token, overall and per task.

    ``ref`` is either a key on the same rows holding the reference model's
    summed logprob over the same tokens (default ``ref_logprob``), or a
    second row list scored under the reference, matched by rollout id,
    scenario id plus rollout index, or prompt plus final text, carrying
    ``logprob``. Rows missing either side are skipped and counted. Per
    task the estimate pools tokens across that task's rollouts, which is
    what a per-task difficulty record wants.
    """
    judged = [r for r in rows if isinstance(r, dict)]
    ref_of: dict[int, float] = {}
    if isinstance(ref, str):
        for i, row in enumerate(judged):
            value = _num(row.get(ref))
            if value is not None:
                ref_of[i] = value
    else:
        by_key: dict[str, float] = {}
        for row in ref:
            if isinstance(row, dict) and _num(row.get("logprob")) is not None:
                by_key.setdefault(row_key(row), float(row["logprob"]))
        for i, row in enumerate(judged):
            value = by_key.get(row_key(row))
            if value is not None:
                ref_of[i] = value
    diff_total = 0.0
    tok_total = 0
    per_task_diff: dict[str, float] = {}
    per_task_tok: dict[str, int] = {}
    used = 0
    for i, row in enumerate(judged):
        lp = _num(row.get("logprob"))
        n = _tokens(row)
        if lp is None or n == 0 or i not in ref_of:
            continue
        used += 1
        diff = lp - ref_of[i]
        diff_total += diff
        tok_total += n
        task = _task(row)
        per_task_diff[task] = per_task_diff.get(task, 0.0) + diff
        per_task_tok[task] = per_task_tok.get(task, 0) + n
    per_task = {t: round(per_task_diff[t] / per_task_tok[t], 6) for t in per_task_diff}
    return {
        "mean_kl": round(diff_total / tok_total, 6) if tok_total else None,
        "per_task": per_task,
        "n_rows": used,
        "n_skipped": len(judged) - used,
        "n_tokens": tok_total,
    }


__all__ = ["logprob_report", "mean_kl"]
