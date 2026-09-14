"""push_file gate, typed Calibration round trip, and the training loss mask."""

from __future__ import annotations

import json

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations import schema
from zeroproof.simulations.score.publish_gate import PublishGateError


def _rows(spec: dict[str, list[int]]) -> list[dict]:
    return [
        {
            "prompt": prompt,
            "reward": label,
            "final_text": f"Issue {i} is open.",
            "steps": [{"tool": "get_issue", "arguments": {"number": i}, "result": {"ok": 1}}],
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "get_issue", "arguments": {"number": i}}],
                },
                {"role": "tool", "name": "get_issue", "content": '{"ok": 1}'},
                {"role": "assistant", "content": f"Issue {i} is open."},
            ],
        }
        for prompt, labels in spec.items()
        for i, label in enumerate(labels)
    ]


def test_push_file_gates_and_uploads_stamped_rows(monkeypatch, tmp_path):
    from zeroproof.simulations.ingest import platform

    uploads: list[bytes] = []

    def fake_call(method, path, api_key=None, body=None, **kw):
        if kw.get("raw_url"):
            uploads.append(kw["data"])
            return {}
        if path == "/datasets":
            return {"datasetId": "ds_f", "uploadUrl": "https://u"}
        return {"datasetId": "ds_f"}

    monkeypatch.setattr(platform, "_call", fake_call)
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps(r) + "\n" for r in _rows({"a": [1, 0, 1, 0]})))
    out = zps.push_file(str(good), api_key="k")
    assert out["gate"]["ok"] and out["datasetId"] == "ds_f"
    sent = [json.loads(line) for line in uploads[-1].decode().splitlines()]
    assert all(r["calibration"]["pass_rate"] == 0.5 for r in sent)

    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in _rows({"a": [1, 1, 1, 1]})))
    with pytest.raises(PublishGateError, match="no_mixed_groups"):
        zps.push_file(str(bad), api_key="k")
    raw = zps.push_file(str(bad), api_key="k", gate=False)
    assert "gate" not in raw and uploads[-1] == bad.read_bytes()


def test_calibration_round_trips_typed():
    rows = _rows({"a": [1, 0, 1, 1]})
    zps.calibrate(rows, policy="Look up first.", model="qwen")
    row = zps.to_row(*zps.from_row(schema.stamp(dict(rows[0]))))
    assert row["calibration"]["pass_rate"] == 0.75 and row["calibration"]["n"] == 4
    cal = zps.calibration_of(row)
    assert isinstance(cal, schema.Calibration)
    assert cal.pass_rate == 0.75 and cal.n == 4 and cal.student.model == "qwen"
    assert cal.student.prompt_hash and cal.task_id == "a"
    _task, rollout, _j, _m = zps.from_row(row)
    assert isinstance(rollout.extra["calibration"], schema.Calibration)
    assert "calibration" not in (rollout.extra.get("passthrough") or {})

    assert zps.calibration_of({"prompt": "x"}) is None
    assert zps.validate(schema.stamp({"prompt": "x", "calibration": {"n": 0, "pass_rate": 2}})) == [
        "calibration_invalid"
    ]
    assert (
        zps.validate(schema.stamp({"prompt": "x", "calibration": {"n": 3, "pass_rate": 0.5}})) == []
    )


def test_training_rows_carry_a_loss_mask_on_assistant_turns_only():
    out = zps.training_rows(_rows({"a": [1]}), system_prompt="Be careful.", tools=[{"x": 1}])
    entry = out[0]
    roles = [m["role"] for m in entry["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert entry["loss_mask"] == [0, 0, 1, 0, 1]
    assert zps.validate(entry, "training") == []
    broken = dict(entry, loss_mask=[1, 0])
    assert zps.validate(broken, "training") == ["loss_mask_invalid"]
    schema_doc = zps.schema.load_json_schema()
    assert "loss_mask" in schema_doc["$defs"]["training_row"]["properties"]
    assert "calibration" in schema_doc["$defs"]["row"]["properties"]
