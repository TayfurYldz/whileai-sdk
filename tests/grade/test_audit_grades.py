"""#255: how often does the verifier fail a right answer?"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.score.audit import _reason_key, audit_grades, audit_warning
from whileai.simulations.score.judging import run_judge
from whileai.simulations.verify import ExactMatch


def _rows():
    """Twelve tasks graded by ExactMatch against a numeric gold. Four
    replies are right but spelled differently (4.0, 04, 4.00, "four"),
    six are wrong, two are exact."""
    cases = [
        ("4", "4"),
        ("4", "4"),
        ("4", "4.0"),
        ("4", "04"),
        ("4", "4.00"),
        ("4", "four"),
        ("4", "5"),
        ("4", "3"),
        ("4", "44"),
        ("4", ""),
        ("4", "5.0"),
        ("4", "nope"),
    ]
    rows = []
    for i, (gold, reply) in enumerate(cases):
        rows.append(
            {
                "prompt": f"what is 2+2 ({i})",
                "scenario_id": f"t{i}",
                "rollout_index": 0,
                "final_text": reply,
                "privileged": {"reference": gold},
                "steps": [],
            }
        )
    return run_judge(rows, ExactMatch()).rows


def _numeric_judge(row):
    """A judge that knows 4.0 is 4, and that the reference is on the row."""
    ref = str(row["privileged"]["reference"]).strip()
    text = str(row.get("final_text") or "").strip().lower()
    words = {"four": "4", "five": "5", "three": "3"}
    text = words.get(text, text)
    try:
        ok = float(text) == float(ref)
    except ValueError:
        ok = False
    assert row["audit"]["verifier"] == "ExactMatch" and row["audit"]["verifier_reward"] == 0
    assert "correct" in row["privileged"]["rubric"]
    return {"reward": 1 if ok else 0, "reason": "same number" if ok else "different"}


def test_audit_estimates_the_false_negative_rate_with_an_interval():
    rows = _rows()
    report = audit_grades(rows, judge=_numeric_judge, sample=100, seed=0)
    assert report["n_failed"] == 10 and report["n_checked"] == 10
    assert report["false_negatives"] == 4 and report["fn_rate"] == 0.4
    lo, hi = report["fn_ci95"]
    assert 0.15 < lo < 0.4 < hi < 0.75
    assert report["estimated_wrong_fails"] == 4
    assert report["verifier"] == "ExactMatch" and report["judge"] == "_numeric_judge"
    assert any(w.startswith("VERIFIER:") for w in report["warnings"])
    assert any("second opinion" in w for w in report["warnings"])
    assert len(report["examples"]) == 4
    assert "overturned 4 of 10" in wai.format_audit(report)
    # the rule's failure kinds, with how many the judge overturned
    assert sum(s["n"] for s in report["reasons"].values()) == 10
    assert sum(s["fn"] for s in report["reasons"].values()) == 4


def test_audit_samples_and_is_seeded():
    rows = _rows()
    a = audit_grades(rows, judge=_numeric_judge, sample=5, seed=3)
    b = audit_grades(rows, judge=_numeric_judge, sample=5, seed=3)
    assert a["n_sampled"] == 5 and a["fn_rate"] == b["fn_rate"]
    assert any("small sample" in w for w in a["warnings"])
    with pytest.raises(ValueError, match="sample"):
        audit_grades(rows, judge=_numeric_judge, sample=0)


def test_audit_checks_passes_too_and_survives_a_broken_judge():
    rows = _rows()

    def strict_then_broken(row):
        if row["audit"]["verifier_reward"] == 1:
            return {"reward": 0, "reason": "the judge disagrees with every pass"}
        raise RuntimeError("boom")

    report = audit_grades(rows, judge=strict_then_broken, sample=10, passes=2, seed=0)
    assert report["fn_rate"] is None and report["judge_errors"] == 10
    assert report["fp_rate"] == 1.0 and report["n_passes_checked"] == 2
    assert any("nothing to say about the verifier" in w for w in report["warnings"])


def test_audit_warning_reaches_the_rl_selection():
    rows = _rows()
    report = audit_grades(rows, judge=_numeric_judge, sample=100)
    # the selection itself needs mixed groups; the warning is what we test
    _picked, sel = wai.select_for_rl(rows, target=10, enforce_band=False, audit=report)
    assert sel["audit"]["fn_rate"] == 0.4
    assert any("verifier false-negative rate 40%" in w for w in sel["hygiene_warnings"])
    quiet = {"fn_rate": 0.05, "fn_ci95": (0.0, 0.2), "n_checked": 40}
    assert audit_warning(quiet) is None and audit_warning(None) is None


def test_reason_key_drops_numbers_and_the_verifier_prefix():
    assert _reason_key("sql_exec: result differs: got 3 rows x 2 cols", "sql_exec") == (
        "result differs"
    )
    assert _reason_key("sql error: OperationalError: no such column", None) == "sql error"
    assert _reason_key("ExactMatch: fail", "ExactMatch") == "fail"
    assert _reason_key("", None) == "(no reason)"
