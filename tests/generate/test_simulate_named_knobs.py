"""The knobs most runs touch are named parameters of ``simulate()``: an
editor shows them, they travel the same road as before, and a misspelled
one is still a ``TypeError``."""

import inspect

import pytest

import whileai.simulations as wai
from tests.helpers import simulate_offline

TOOLS = [{"name": "lookup", "description": "x", "parameters": {"type": "object", "properties": {}}}]


def test_common_knobs_are_in_the_signature():
    params = inspect.signature(wai.simulate).parameters
    for name in (
        "repeats",
        "phrasings",
        "repeat_policy",
        "concurrency",
        "simulator",
        "user_model",
        "backend",
        "seed",
        "sampling",
        "max_turns",
        "avg_turns",
        "fault_rate",
        "temperature",
        "timeout",
        "logprobs",
        "seeds",
    ):
        assert name in params, name
        assert params[name].default is None, name


def test_named_repeats_and_policy_reach_the_run():
    data = simulate_offline(
        policy="Refund desk.",
        tools=TOOLS,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        situations=2,
        budget=4,
    )
    assert len(data.trajectories) == 4
    assert max(int(r.get("rollout_index") or 0) for r in data.trajectories) == 1
    assert data.repeat_policy == "fixed"


def test_misspelled_knob_is_a_type_error():
    with pytest.raises(TypeError, match="repeatz"):
        simulate_offline(policy="Refund desk.", tools=TOOLS, budget=2, repeatz=3)
