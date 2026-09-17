"""Purpose (train / holdout / eval / raw), mode, and the holdout split on push."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations import data as data_mod
from whileai.simulations.ingest import platform


class Recorder:
    def __init__(self):
        self.calls = []
        self.n = 0

    def __call__(self, method, path, api_key, body=None, *, raw_url=None, public=False, **kw):
        self.calls.append((method, path, body))
        if raw_url:
            return b""
        if path == "/datasets" and method == "POST":
            self.n += 1
            return {"datasetId": f"ds_{self.n}", "uploadUrl": "https://s3/x"}
        if path.endswith("/finalize"):
            return {"datasetId": path.split("/")[2], "status": "ready"}
        return {"datasetId": "ds_x", **(body or {})}


def test_push_rows_sends_purpose_mode_agent_description(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(platform, "_call", rec)
    platform.push_rows(
        [{"prompt": "p", "reward": 1}],
        "n",
        purpose="holdout",
        mode="rl",
        agent="a",
        description="d",
    )
    created = next(b for m, p, b in rec.calls if p == "/datasets" and m == "POST")
    assert created == {
        "name": "n",
        "purpose": "holdout",
        "mode": "rl",
        "agent": "a",
        "description": "d",
    }
    with pytest.raises(ValueError, match="purpose"):
        platform.push_rows([{"prompt": "p"}], "n", purpose="test")


def test_update_dataset_and_preview(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(platform, "_call", rec)
    out = wai.update_dataset("ds_9", purpose="eval", description="held")
    assert out["purpose"] == "eval"
    assert rec.calls[-1] == (
        "POST",
        "/datasets/ds_9/meta",
        {"purpose": "eval", "description": "held"},
    )
    with pytest.raises(ValueError):
        wai.update_dataset("ds_9")
    with pytest.raises(ValueError, match="mode"):
        wai.update_dataset("ds_9", mode="fast")
    wai.preview("ds_9")
    assert rec.calls[-1][:2] == ("GET", "/datasets/ds_9/preview")


def test_profile_unwraps_and_can_force(monkeypatch):
    calls = []

    def fake_call(method, path, api_key, body=None, **kw):
        calls.append(path)
        return {"datasetId": "ds_9", "profile": {"rows": 3, "pass_rate": 0.5}}

    monkeypatch.setattr(platform, "_call", fake_call)
    assert wai.profile("ds_9") == {"rows": 3, "pass_rate": 0.5}
    wai.profile("ds_9", force=True)
    assert calls == ["/datasets/ds_9/profile", "/datasets/ds_9/profile?force=1"]


def test_split_holdout_is_by_task_and_deterministic():
    rows = [{"scenario_id": f"s{i % 10}", "prompt": f"p{i}"} for i in range(100)]
    train, held = data_mod._split_holdout(rows, 0.3)
    assert len(train) + len(held) == 100
    held_tasks = {r["scenario_id"] for r in held}
    train_tasks = {r["scenario_id"] for r in train}
    assert not (held_tasks & train_tasks), "a task is wholly on one side"
    assert 0 < len(held_tasks) < 10
    assert data_mod._split_holdout(rows, 0.3) == (train, held)
    assert data_mod._split_holdout(rows, None) == (rows, [])
    with pytest.raises(ValueError):
        data_mod._split_holdout(rows, 1.5)


def test_push_with_holdout_makes_two_linked_datasets(monkeypatch):
    from tests.helpers import simulate_offline

    data = simulate_offline(budget=6, seed=0)
    pushes = []

    def fake_push_rows(
        rows,
        name,
        *,
        api_key=None,
        parent=None,
        purpose=None,
        mode=None,
        agent=None,
        description=None,
    ):
        pushes.append(
            {"name": name, "n": len(rows), "parent": parent, "purpose": purpose, "mode": mode}
        )
        return {"datasetId": f"ds_{len(pushes)}"}

    monkeypatch.setattr(data_mod, "push_rows", fake_push_rows)
    monkeypatch.setattr(
        data_mod, "_split_holdout", lambda rows, f: (rows[:-1], rows[-1:]) if f else (rows, [])
    )
    out = data.push("run", holdout=0.2, gate=False)
    assert out["datasetId"] == "ds_1"
    assert out["holdout"]["datasetId"] == "ds_2"
    assert pushes[0]["purpose"] == "train" and pushes[0]["parent"] is None
    assert pushes[1] == {
        "name": "run-holdout",
        "n": 1,
        "parent": "ds_1",
        "purpose": "holdout",
        "mode": pushes[0]["mode"],
    }

    pushes.clear()
    data.push("evalset", purpose="eval", gate=False)
    assert pushes == [
        {
            "name": "evalset",
            "n": len(data.rows()),
            "parent": None,
            "purpose": "eval",
            "mode": pushes[0]["mode"],
        }
    ]
