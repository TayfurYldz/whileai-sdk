"""Did training move the behavior, and did anything else slip?

One call over two graded, marker-scored row sets: the rollouts before a
training run and the rollouts after it, on the same tasks. Every metric
the two sets share (pass@1 and each marker) is compared as paired task
differences with a bootstrap interval (``stats.compare_runs``), so the
answer is "moved by X, interval Y" and not a pair of means.

Markers are read as higher-is-better. A metric named in
``must_not_regress`` whose interval sits entirely below zero is a
regression and fails the report; any other metric that drops
significantly is a warning (rlhf-book ch. 15: post-training on one thing
forgets others, and on-policy data forgets less, which is only visible
if you measure the others).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .stats import DEFAULT_BOOT, compare_runs, marker_names


def delta_report(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = None,
    must_not_regress: Sequence[str] = (),
    markers: Sequence[str] | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare ``after`` to ``before`` on pass@1 and every shared marker.

    ``target`` names the metric the run was meant to move (``"pass_at_1"``
    or ``"marker:<name>"``); the verdict on it is the headline.
    ``must_not_regress`` lists metrics whose significant drop fails the
    report. Metric names for markers are the marker names; pass@1 is
    ``"pass_at_1"``.
    """
    names = (
        list(markers)
        if markers is not None
        else sorted(set(marker_names(before)) & set(marker_names(after)))
    )
    metrics = ["pass_at_1", *[f"marker:{m}" for m in names]]
    results: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(metrics):
        results[metric] = compare_runs(before, after, metric=metric, n_boot=n_boot, seed=seed + i)

    def _key(name: str) -> str:
        return name if name == "pass_at_1" or name.startswith("marker:") else f"marker:{name}"

    guarded = {_key(m) for m in must_not_regress}
    regressions = [m for m in metrics if m in guarded and results[m]["verdict"] == "a_better"]
    slipped = [m for m in metrics if m not in guarded and results[m]["verdict"] == "a_better"]
    improved = [m for m in metrics if results[m]["verdict"] == "b_better"]
    target_key = _key(target) if target else None
    target_result = results.get(target_key) if target_key else None
    if target_result is None and target_key:
        target_verdict = "target_not_measured"
    elif target_result is None:
        target_verdict = None
    else:
        target_verdict = {
            "b_better": "moved",
            "a_better": "moved_the_wrong_way",
            "no_difference_detected": "no_change_detected",
            "insufficient_data": "insufficient_data",
        }[target_result["verdict"]]
    ok = not regressions and target_verdict not in {"moved_the_wrong_way"}
    warnings: list[str] = []
    for m in regressions:
        r = results[m]
        warnings.append(
            f"REGRESSION {m}: {r['delta']:+.3f} (95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}), "
            "named in must_not_regress"
        )
    for m in slipped:
        r = results[m]
        warnings.append(
            f"{m} dropped {r['delta']:+.3f} (95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f})"
        )
    if target_result and target_result.get("note"):
        warnings.append(f"{target_key}: {target_result['note']}")
    if target_verdict == "target_not_measured":
        warnings.append(f"target {target!r} is not on both row sets")
    return {
        "ok": ok,
        "target": target_key,
        "target_verdict": target_verdict,
        "target_delta": target_result["delta"] if target_result else None,
        "target_ci95": target_result["ci95"] if target_result else None,
        "n_paired_tasks": results["pass_at_1"]["n_paired"],
        "improved": improved,
        "regressions": regressions,
        "slipped": slipped,
        "metrics": results,
        "warnings": warnings,
    }


def format_delta_report(report: dict[str, Any]) -> str:
    """The block a person reads: headline, then one line per metric."""
    lines: list[str] = []
    if report.get("target"):
        r = report["metrics"].get(report["target"])
        if r and r.get("delta") is not None and r.get("ci95"):
            lines.append(
                f"{report['target']}: {report['target_verdict']} "
                f"({r['delta']:+.3f}, 95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, "
                f"{r['n_paired']} paired tasks)"
            )
        else:
            lines.append(f"{report['target']}: {report['target_verdict']}")
    lines.append("PASS" if report["ok"] else "FAIL")
    for name, r in report["metrics"].items():
        if r.get("delta") is None:
            lines.append(f"  {name:<28} insufficient data")
            continue
        ci = r.get("ci95")
        span = f"{ci[0]:+.3f}..{ci[1]:+.3f}" if ci else "n/a"
        tag = {
            "b_better": "up",
            "a_better": "DOWN",
            "no_difference_detected": "flat",
            "insufficient_data": "n/a",
        }[r["verdict"]]
        pair = "paired" if r["paired"] else "unpaired"
        lines.append(
            f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
            f"[{span}]  {tag}  ({r['n_used']} {pair})"
        )
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = ["delta_report", "format_delta_report"]
