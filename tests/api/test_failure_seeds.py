"""Failing asks in traces seed the run: capability failures have no axis signal."""

from __future__ import annotations

from tests.helpers import POLICY, TOOLS, simulate_offline

FAILS = [
    {
        "prompt": "what share of revenue came from software last quarter",
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "1"}, "result": {"status": "ok"}}
        ],
        "final_text": "0.0%",
        "reward": 0,
    },
    {
        "prompt": "which month had the most orders in 2026",
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "2"}, "result": {"status": "ok"}}
        ],
        "final_text": "April",
        "reward": 0,
    },
    {
        "prompt": "how many orders are there",
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "3"}, "result": {"status": "ok"}}
        ],
        "final_text": "800",
        "reward": 1,
    },
]


def test_failing_asks_from_traces_become_seeds_and_are_disclosed():
    data = simulate_offline(tools=TOOLS, policy=POLICY, traces=FAILS, budget=8, concurrency=1)
    mining = data.search["trace_mining"]
    assert mining["failure_seeds"] == 2
    prompts = {r["prompt"] for r in data.trajectories}
    # the originals are sources: they aim the run and never enter it verbatim
    assert not ({f["prompt"] for f in FAILS} & prompts)
