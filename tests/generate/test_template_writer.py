"""The test-only template writer: what the suite runs offline instead of a model."""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS, scripted_agent
from tests.template_writer import _PROBE_FAMILIES, open_ended_probes, template_writer


def test_probe_families_stay_intact():
    names = [name for name, _ in _PROBE_FAMILIES]
    assert {"out_of_domain_factual", "creative", "garbage_input", "prompt_injection"} <= set(names)
    probes = open_ended_probes(TOOLS, POLICY, per_round=16, seed=0)
    blob = " ".join(probes).lower()
    assert "mongolia" in blob or "haiku" in blob or "asdf" in blob
    assert "ignore" in blob


def test_package_refuses_to_simulate_without_a_model():
    with pytest.raises(ValueError, match="no offline template writer"):
        zps.simulate(scripted_agent, tools=TOOLS, system_prompt=POLICY, simulator=False, budget=2)
    assert not hasattr(zps, "open_ended_probes")


def test_writer_factory_is_called_with_the_resolved_agent():
    data = zps.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=template_writer,
        budget=4,
        seed=0,
    )
    assert len(data.trajectories) == 4
    assert data.stopped_because == "budget"
