"""Tester reports #268 (holdout shares situations) and #270 (degenerate markers)."""

from __future__ import annotations

import whileai.simulations as wai
from whileai.simulations.score.delta import delta_report
from whileai.simulations.score.stats import metric_summary

# ------------------------------------------------------------ #268


def _situation(sid, phrasings, k=3):
    rows = []
    for p, text in enumerate(phrasings):
        for i in range(k):
            rows.append(
                {
                    "scenario_id": sid,
                    "prompt": text,
                    "rollout_index": p * k + i,
                    "final_text": f"reply {i}",
                    "steps": [{"tool": "lookup_invoice", "args": {}, "result": {"ok": True}}],
                    "reward": 1,
                }
            )
    return rows


def test_split_is_disjoint_in_the_unit_reports_count_in():
    rows = []
    for s in range(12):
        rows += _situation(f"s{s}", [f"ask {s} plainly", f"ask {s} again, reworded"])
    held, train = wai.split_pseudo_production(rows, fraction=0.25, seed=7)
    assert held and train and len(held) + len(train) == len(rows)
    hk = {wai.task_key(r) for r in held}
    tk = {wai.task_key(r) for r in train}
    assert not (hk & tk), hk & tk
    # every phrasing and repeat of a held-out situation is held out with it
    for sid in hk:
        assert sum(1 for r in held if r["scenario_id"] == sid) == 6
    assert not ({r["prompt"] for r in held} & {r["prompt"] for r in train})


def test_split_still_works_on_prompt_only_rows():
    rows = [
        {"prompt": f"p{i}", "rollout_index": j, "final_text": "x", "steps": [], "reward": 1}
        for i in range(8)
        for j in range(2)
    ]
    held, train = wai.split_pseudo_production(rows, fraction=0.25, seed=1)
    assert held and train
    assert not ({r["prompt"] for r in held} & {r["prompt"] for r in train})
    # rows with no key at all are their own tasks and do not clump
    bare = [{"final_text": "x", "steps": [], "reward": 1} for _ in range(8)]
    held, train = wai.split_pseudo_production(bare, fraction=0.25, seed=1)
    assert held and train


# ------------------------------------------------------------ #270


def _marked(values, name="escalated"):
    return [
        {
            "scenario_id": f"t{i // 3}",
            "prompt": f"p{i // 3}",
            "rollout_index": i % 3,
            "final_text": "x",
            "reward": 1 if i % 2 else 0,
            "markers": {name: v},
        }
        for i, v in enumerate(values)
    ]


def test_a_marker_that_never_varied_is_flagged_not_certified():
    rows = _marked([1.0] * 12)
    out = metric_summary(rows, "marker:escalated")
    assert out["degenerate"] is True and out["ci95"] is None and out["mean"] == 1.0
    assert out["n_rows_at_1"] == 12 and out["n_rows_at_0"] == 0
    assert "has not been shown to be able to come out any other way" in out["warning"]
    assert "must_not_regress" in out["warning"]
    # the same table entry for a marker pinned at 0 (a string-vs-float bug reads like this)
    zero = metric_summary(_marked([0.0] * 12), "marker:escalated")
    assert zero["degenerate"] and zero["ci95"] is None and zero["n_rows_at_0"] == 12
    # a marker with contrast keeps its interval and no warning
    live = metric_summary(_marked([1.0, 0.0] * 6), "marker:escalated")
    assert live["degenerate"] is False and live["ci95"] is not None and "warning" not in live
    assert live["n_rows_at_1"] == 6 and live["n_rows_at_0"] == 6
    # marker_summary carries it through
    assert wai.marker_summary(rows)["escalated"]["degenerate"] is True


def test_delta_report_names_a_guard_that_cannot_fail():
    before = _marked([1.0] * 12)
    after = _marked([1.0] * 12)
    report = delta_report(before, after, must_not_regress=["escalated"], n_boot=100)
    assert report["degenerate_guards"] == ["marker:escalated"]
    assert any("this guard cannot fail" in w for w in report["warnings"])
    live_before = _marked([1.0, 0.0] * 6)
    live_after = _marked([1.0, 1.0, 0.0] * 4)
    report = delta_report(live_before, live_after, must_not_regress=["escalated"], n_boot=100)
    assert report["degenerate_guards"] == []
