"""export_environment: a simulation becomes an installable RL environment.

Offline. The verifiers-backed tests skip when the package is not
installed (``uv sync --extra rl``); the export itself needs only the SDK.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.environment import _ref_of, build_tasks, resolve_ref
from zeroproof.simulations.score.grading import conduct_grade

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "Refund an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = "Look up an order before refunding it."


def _rows() -> list[dict]:
    """Four prompts, k=4: one always solved, one never, two mixed, one ungraded."""
    plan = {
        "refund ORD-1 please": [1, 1, 1, 1],
        "refund ORD-2 please": [0, 0, 0, 0],
        "where is ORD-3": [1, 0, 1, 0],
        "cancel ORD-4 today": [1, 1, 0, 1],
        "hi, need help with ORD-5": [None],
    }
    rows = []
    for i, (prompt, labels) in enumerate(plan.items()):
        for k, label in enumerate(labels):
            rows.append(
                {
                    "prompt": prompt,
                    "scenario_id": f"s{i}",
                    "rollout_index": k,
                    "reward": label,
                    "seed": 7,
                    "world_state": "entity exists" if i % 2 else "",
                    "faults": {"lookup_order": {"mode": "timeout", "rate": 1.0}} if i == 2 else {},
                    "steps": [],
                    "final_text": "done",
                    "judge_status": "ok" if label is not None else "missing_reward",
                }
            )
    return rows


def outcome_reward(row: dict) -> dict:
    """A module-level judge: reward 1 when the agent looked the order up."""
    called = any(s.get("tool") == "lookup_order" for s in row.get("steps") or [])
    return {"reward": 1 if called else 0, "reason": "looked up" if called else "no lookup"}


def test_build_tasks_applies_band_split_and_decontamination():
    train, held, report = build_tasks(_rows(), holdout=0.5, band=(0.2, 0.8))
    tasks = train + held
    prompts = {t["prompt"] for t in tasks}
    assert "refund ORD-1 please" not in prompts and "refund ORD-2 please" not in prompts
    assert "hi, need help with ORD-5" in prompts  # ungraded prompts are kept
    assert report["band_dropped"] == 2 and report["tasks"] == 3
    mixed = next(t for t in tasks if t["prompt"] == "where is ORD-3")
    assert mixed["info"]["calibration"] == {"pass_rate": 0.5, "n": 4}
    assert mixed["info"]["faults"]["lookup_order"]["mode"] == "timeout"
    assert all(t["info"]["split"] in ("train", "holdout") for t in tasks)
    assert {t["example_id"] for t in tasks} == {t["info"]["task_id"] for t in tasks}


def test_build_tasks_explicit_holdout_and_no_band():
    train, held, _ = build_tasks(_rows(), holdout=["where is ORD-3"], band=None)
    assert [t["prompt"] for t in held] == ["where is ORD-3"]
    assert len(train) == 4  # no band: unanimous prompts stay


def test_refs_round_trip_and_reject_locals():
    ref = _ref_of(conduct_grade)
    assert ref == "zeroproof.simulations.score.grading:conduct_grade"
    assert resolve_ref(ref) is conduct_grade
    assert resolve_ref(_ref_of(outcome_reward)) is outcome_reward
    with pytest.raises(ValueError, match="importable"):
        _ref_of(lambda row: 1)
    with pytest.raises(ValueError, match="module:attr"):
        _ref_of("conduct_grade")


def test_export_environment_writes_an_installable_package(tmp_path):
    out = tmp_path / "refund-agent"
    report = zps.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, reward=outcome_reward
    )
    assert report["name"] == "refund_agent" and report["warnings"] == []
    assert report["reward"].endswith(":outcome_reward")
    assert resolve_ref(report["reward"]) is outcome_reward
    pkg = out / "refund_agent"
    spec = json.loads((pkg / "spec.json").read_text())
    assert spec["system_prompt"] == POLICY
    assert [t["name"] for t in spec["tools"]] == ["lookup_order", "create_refund"]
    assert spec["tools"][0]["parameters"]["required"] == ["order_id"]
    assert spec["max_turns"] >= 1 and spec["execute"] is None
    train = [json.loads(line) for line in (pkg / "data" / "train.jsonl").read_text().splitlines()]
    held = [json.loads(line) for line in (pkg / "data" / "holdout.jsonl").read_text().splitlines()]
    assert len(train) + len(held) == report["tasks"] == 3
    assert set(train[0]) == {"prompt", "info", "example_id"}
    assert (
        "prompt" in (pkg / "__init__.py").read_text()
        or "load_environment" in (pkg / "__init__.py").read_text()
    )
    pyproject = (out / "pyproject.toml").read_text()
    assert 'name = "refund-agent"' in pyproject and "verifiers>=0.3" in pyproject
    assert 'build-backend = "hatchling.build"' in pyproject
    assert 'include = ["refund_agent/**", "pyproject.toml", "README.md"]' in pyproject
    assert "[tool.verifiers.eval]" in pyproject
    readme = (out / "README.md").read_text()
    assert ":outcome_reward" in readme and "prime eval run refund-agent" in readme
    assert "2 train / 1 holdout" in readme or "1 train / 2 holdout" in readme
    vendored = (pkg / "_zp_env.py").read_text()
    assert "from zeroproof.simulations.export import _resolve" in vendored
    assert "from ." not in vendored.replace("from ._", "")


def test_export_without_reward_warns_about_process_reward(tmp_path):
    report = zps.export_environment(_rows(), tmp_path / "env", tools=TOOLS, system_prompt=POLICY)
    assert report["reward"].endswith(":conduct_grade")
    assert any("process reward" in w for w in report["warnings"])
    assert "process reward" in (tmp_path / "env" / "README.md").read_text()


def test_export_needs_tools(tmp_path):
    with pytest.raises(ValueError, match="tools"):
        zps.export_environment(_rows(), tmp_path / "env", system_prompt=POLICY)


def test_task_file_is_not_a_training_file():
    """The answer key rides in ``info`` for the server; nothing in the
    student-visible ``prompt`` field carries it."""
    rows = _rows()
    for r in rows:
        r["privileged"] = {"reference": "zzteachersecretzz"}
    train, held, _ = build_tasks(rows, holdout=0.5, band=None)
    for t in train + held:
        assert "zzteachersecretzz" not in t["prompt"]
        assert t["info"]["privileged"]["reference"] == "zzteachersecretzz"


vf = pytest.importorskip("verifiers", reason="verifiers not installed (uv sync --extra rl)")


def _fake_state(task: dict, spec: dict) -> dict:
    return {
        "info": task["info"],
        "prompt": [
            {"role": "system", "content": spec["system_prompt"]},
            {"role": "user", "content": task["prompt"]},
        ],
        "completion": [],
    }


def test_load_environment_drives_a_rollout_through_the_mock_world(tmp_path):
    out = tmp_path / "refund-agent"
    zps.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, reward=outcome_reward
    )
    env = zps.load_environment(out / "refund_agent" / "spec.json")
    assert [t.name for t in env.tool_defs] == ["lookup_order", "create_refund"]
    assert len(env.dataset) >= 1
    spec = json.loads((out / "refund_agent" / "spec.json").read_text())
    task = json.loads((out / "refund_agent" / "data" / "train.jsonl").read_text().splitlines()[0])
    state = asyncio.run(env.setup_state(_fake_state(task, spec)))
    assert state["zp_steps"] == [] and state["zp_world"] is not None

    args = env.update_tool_args("lookup_order", {"order_id": "ORD-9"}, [], state)
    msg = asyncio.run(env.call_tool("lookup_order", args, "call_1"))
    assert msg.role == "tool" and msg.tool_call_id == "call_1"
    result = json.loads(msg.content)
    assert "status" in result
    assert state["zp_steps"][0]["tool"] == "lookup_order"
    assert state["zp_steps"][0]["arguments"] == {"order_id": "ORD-9"}

    state["completion"] = [{"role": "assistant", "content": "Looked it up; refunded."}]
    assert env.reward_func(state) == 1.0
    assert env.n_calls(state) == 1.0 and env.judge_ok(state) == 1.0
    rubrics = getattr(env.rubric, "rubrics", None) or [env.rubric]
    rubric_funcs = {f.__name__ for r in rubrics for f in getattr(r, "funcs", [])}
    assert {"reward_func", "n_calls", "judge_ok", "lookup_order_calls"} <= rubric_funcs


def test_load_environment_world_is_seeded_per_task(tmp_path):
    out = tmp_path / "env"
    zps.export_environment(_rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5)
    env = zps.load_environment(out / "env" / "spec.json")
    spec = json.loads((out / "env" / "spec.json").read_text())
    faulty = next(
        json.loads(line)
        for line in (out / "env" / "data" / "train.jsonl").read_text().splitlines()
        + (out / "env" / "data" / "holdout.jsonl").read_text().splitlines()
        if json.loads(line)["info"]["faults"]
    )
    s1 = asyncio.run(env.setup_state(_fake_state(faulty, spec)))
    s2 = asyncio.run(env.setup_state(_fake_state(faulty, spec)))
    a1 = env.update_tool_args("lookup_order", {"order_id": "ORD-3"}, [], s1)
    a2 = env.update_tool_args("lookup_order", {"order_id": "ORD-3"}, [], s2)
    r1 = json.loads(asyncio.run(env.call_tool("lookup_order", a1, "c")).content)
    r2 = json.loads(asyncio.run(env.call_tool("lookup_order", a2, "c")).content)
    assert r1 == r2 == {"status": "timeout", "error": "request timed out"}


def test_load_environment_with_a_live_world(tmp_path):
    calls: list[tuple[str, dict]] = []

    def world(tool: str, arguments: dict) -> dict:
        calls.append((tool, arguments))
        return {"status": "ok", "order": arguments.get("order_id")}

    out = tmp_path / "env"
    zps.export_environment(_rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5)
    env = zps.load_environment(out / "env" / "spec.json", execute=world, reward=outcome_reward)
    spec = json.loads((out / "env" / "spec.json").read_text())
    task = json.loads((out / "env" / "data" / "train.jsonl").read_text().splitlines()[0])
    state = asyncio.run(env.setup_state(_fake_state(task, spec)))
    assert "zp_world" not in state
    args = env.update_tool_args("lookup_order", {"order_id": "ORD-1"}, [], state)
    msg = asyncio.run(env.call_tool("lookup_order", args, "c1"))
    assert json.loads(msg.content) == {"status": "ok", "order": "ORD-1"}
    assert calls == [("lookup_order", {"order_id": "ORD-1"})]


def test_vendored_module_loads_standalone(tmp_path):
    """The copy in the package works when the installed SDK predates the
    environment module: import it from its file, not from the package."""
    import importlib.util

    out = tmp_path / "env"
    zps.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, reward=outcome_reward
    )
    path = out / "env" / "_zp_env.py"
    spec = importlib.util.spec_from_file_location("zp_env_vendored", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.load_environment(out / "env" / "spec.json")
    assert [t.name for t in env.tool_defs] == ["lookup_order", "create_refund"]
