"""A stale fault answers with an out-of-date record, never a bare hash."""

from __future__ import annotations

import json

from tests.helpers import TOOLS
from whileai.simulations.world.sandbox import MockEnvironment


def test_stale_result_is_a_record_marked_stale():
    tool = TOOLS[0]["function"]["name"] if "function" in TOOLS[0] else TOOLS[0]["name"]
    params = (TOOLS[0].get("function", TOOLS[0]))["parameters"]["properties"]
    args = {k: "4412" for k in list(params)[:1]}
    env = MockEnvironment(TOOLS, seed=1, faults={tool: {"mode": "stale", "rate": 1.0}})
    out = env.call(tool, args)
    assert out["status"] == "ok" and out["stale"] is True and out["as_of"]
    data = out["data"]
    assert isinstance(data, dict) and "result" not in data
    # the same record the fresh call would give, so a stale read is out of
    # date, not meaningless
    fresh = MockEnvironment(TOOLS, seed=1).call(tool, args)
    assert json.dumps(data, sort_keys=True) == json.dumps(fresh["data"], sort_keys=True)
