"""The GRPO example's reward and prompt builder, offline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import zeroproof.simulations as zps

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "grpo"


def _load():
    spec = importlib.util.spec_from_file_location("grpo_example_reward", EXAMPLE / "reward.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("grpo_example_reward", module)
    spec.loader.exec_module(module)
    return module


CALL = '<tool_call>\n{"name": "lookup_order", "arguments": {"order_id": "ORD-4017"}}\n</tool_call>'
REFUND = '<tool_call>{"name": "create_refund", "arguments": {"order_id": "ORD-4017", "amount": 20}}</tool_call>'


def test_case_reads_the_order_id_and_domain():
    r = _load()
    assert r.case_for("please check ORD-4017 for me") == {"order_id": "ORD-4017", "in_domain": True}
    assert r.case_for("look at ord_991 please")["order_id"] == "ORD-991"
    assert r.case_for("I need help with a refund") == {"order_id": None, "in_domain": True}
    assert r.case_for("How tall is Kilimanjaro?") == {"order_id": None, "in_domain": False}


def test_score_rewards_the_rule_and_the_format():
    r = _load()
    with_id = r.case_for("please check ORD-4017 for me")
    assert r.score(CALL, with_id) == 1.0
    assert r.score(REFUND, with_id) == pytest.approx(0.2)  # well formed, wrong move
    wrong_id = CALL.replace("ORD-4017", "ORD-9999")
    assert r.score(wrong_id, with_id) == pytest.approx(0.3)
    assert r.score("Sure, what is your order number?", with_id) == pytest.approx(0.3)
    assert r.score("<tool_call>{not json</tool_call>", with_id) == 0.0

    no_id = r.case_for("I need help with a refund")
    assert r.score("Of course. What is the order id?", no_id) == 1.0
    assert r.score(CALL, no_id) == pytest.approx(0.2)  # invented id, format bonus only
    assert r.score("Refunded.", no_id) == pytest.approx(0.4)

    off = r.case_for("How tall is Kilimanjaro?")
    assert r.score("I can only help with orders and refunds.", off) == 1.0
    assert r.score(CALL, off) == pytest.approx(0.2)
    assert r.score("", off) == 0.0


def test_reward_rows_feed_pass_at_and_delta():
    r = _load()
    prompts = [
        {
            "prompt": "check ORD-100 please",
            "case": r.case_for("check ORD-100 please"),
            "scenario_id": "s1",
        },
        {"prompt": "refund help", "case": r.case_for("refund help"), "scenario_id": "s2"},
    ]
    before = r.reward_rows(prompts, [[REFUND.replace("4017", "100"), "hm"], ["Refunded.", "ok"]])
    after = r.reward_rows(
        prompts,
        [
            [CALL.replace("4017", "100"), CALL.replace("4017", "100")],
            ["What is the order id?", "Order id?"],
        ],
    )
    assert zps.pass_at(before).pass_at_1 == 0.0 and zps.pass_at(after).pass_at_1 == 1.0
    assert before[0]["markers"]["tool_rule"] == pytest.approx(0.2)
    rep = zps.delta_report(before, after, target="pass_at_1", must_not_regress=["well_formed"])
    assert rep["target_delta"] == pytest.approx(1.0)


def test_build_prompts_offline_and_split():
    r = _load()
    items = r.build_prompts(24, seed=1)
    assert 8 <= len(items) <= 24 and all(set(i) == {"prompt", "case", "scenario_id"} for i in items)
    assert any(i["case"]["order_id"] for i in items) and any(
        not i["case"]["order_id"] for i in items
    )
    train, held = r.split_holdout(items, 0.25)
    assert len(train) + len(held) == len(items)
    assert r.split_holdout(items, 0.25) == (train, held)  # deterministic
