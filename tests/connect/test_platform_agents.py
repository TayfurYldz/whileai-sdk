"""The agent registry from the SDK: list, register, and a push attaching the spec."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations import data as data_mod
from zeroproof.simulations.ingest import platform


def test_agents_and_register_agent(monkeypatch):
    calls = []

    def fake_call(method, path, api_key, body=None, **kw):
        calls.append((method, path, body))
        if path == "/agents" and method == "GET":
            return {"agents": [{"slug": "airline-support", "traces": 3}]}
        return {"slug": "airline-support", **(body or {})}

    monkeypatch.setattr(platform, "_call", fake_call)
    assert zps.agents() == [{"slug": "airline-support", "traces": 3}]
    out = zps.register_agent(
        "Airline Support",
        description="refunds",
        tools=[{"name": "lookup"}],
        system_prompt="Be kind.",
    )
    assert out["slug"] == "airline-support"
    assert calls[-1] == (
        "POST",
        "/agents",
        {
            "name": "Airline Support",
            "description": "refunds",
            "tools": [{"name": "lookup"}],
            "system_prompt": "Be kind.",
        },
    )


def test_push_with_agent_attaches_the_spec(monkeypatch):
    from tests.helpers import simulate_offline

    data = simulate_offline(budget=2, seed=0)
    registered = []
    monkeypatch.setattr(data_mod, "push_rows", lambda rows, name, **kw: {"datasetId": "ds_1"})
    monkeypatch.setattr(
        platform, "register_agent", lambda name, **kw: registered.append((name, kw)) or {}
    )
    data.push("run", agent="github", gate=False)
    assert registered and registered[0][0] == "github"
    kw = registered[0][1]
    assert kw["tools"] and kw["system_prompt"]

    registered.clear()
    data.push("run", gate=False)
    assert registered == [], "no agent named, nothing to attach"
