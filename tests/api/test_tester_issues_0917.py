"""Five tester reports from 2026-09-17: #260 #261 #262 #263 #264."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations import training
from whileai.simulations.generate import agents
from whileai.simulations.generate.agents import complete as _real_complete
from whileai.simulations.score.judging import _first_turn, build_preference_pairs

# ------------------------------------------------------------ #261 mine_traces


def _trace(result, reward=1):
    return {
        "prompt": "What is invoice INV-1?",
        "steps": [
            {"tool": "lookup_invoice", "arguments": {"invoice_id": "INV-1"}, "result": result}
        ],
        "final_text": "Invoice INV-1 is $10.00.",
        "reward": reward,
    }


@pytest.mark.parametrize(
    "result",
    [
        {"invoice_id": "INV-1", "amount_usd": 10.0, "status": "paid"},
        {"invoice_id": "INV-1", "status": "open"},
        {"invoice_id": "INV-1", "status": "completed"},
        {"invoice_id": "INV-1", "status": "ok"},
        {"invoice_id": "INV-1"},
    ],
)
def test_a_status_key_with_the_tools_own_vocabulary_is_not_a_fault(result):
    rows = [_trace(result) for _ in range(5)]
    mined = wai.mine_traces(wai.load_traces(rows))
    assert mined["faults"] == {}
    assert mined["tools"]["lookup_invoice"] == {"n": 5, "fault_n": 0}
    assert mined["flaw_rows"] == []


@pytest.mark.parametrize(
    "status, fault",
    [("error", "error"), ("timeout", "timeout"), ("not_found", "not_found"), ("denied", "deny")],
)
def test_a_failing_status_still_counts_as_a_fault(status, fault):
    rows = [_trace({"invoice_id": "INV-1", "status": status}) for _ in range(3)]
    mined = wai.mine_traces(wai.load_traces(rows))
    assert mined["faults"] == {fault: 3}
    assert mined["tools"]["lookup_invoice"]["fault_n"] == 3
    assert len(mined["flaw_rows"]) == 3


def test_a_zero_label_is_still_a_flaw_row_without_any_fault():
    rows = [_trace({"status": "paid"}, reward=0) for _ in range(2)]
    mined = wai.mine_traces(wai.load_traces(rows))
    assert mined["faults"] == {} and len(mined["flaw_rows"]) == 2


# ------------------------------------------------------------ #262 serve(name, get_run(id))


class _Gate:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, api_key=None, body=None, **kw):
        self.calls.append((method, path, body))
        if method == "GET" and path.startswith("/runs/"):
            return {
                "runId": "run_h1",
                "status": "done",
                "baseModel": "Qwen/Qwen3-4B",
                "adapter": "volume zeroproof-train-runs:/run_h1/adapter",
            }
        if method == "POST" and path == "/models":
            return {**body, "version": 1, "endpoint": "https://serve.example/v1"}
        if method == "DELETE" and path.startswith("/models/"):
            return {"name": path.rsplit("/", 1)[-1], "deleted": True}
        raise AssertionError(f"unexpected {method} {path}")


def test_serve_accepts_the_record_get_run_returns():
    gate = _Gate()
    record = gate("GET", "/runs/run_h1")
    row = training.serve("my-model", record, transport=gate)
    assert row["adapter"] == record["adapter"] and row["baseModel"] == "Qwen/Qwen3-4B"
    # the record carried everything; no second GET was needed
    assert [c[0] for c in gate.calls] == ["GET", "POST"]


def test_serve_names_the_accepted_types_instead_of_dying_in_urllib():
    with pytest.raises(TypeError, match=r"TrainingRun, the record wai.get_run"):
        training.serve("my-model", 42, transport=_Gate())
    with pytest.raises(TypeError, match="no runId"):
        training.serve("my-model", {"series": []}, transport=_Gate())


def test_training_run_exposes_its_id():
    run = training.TrainingRun("run_h1", name="x", transport=_Gate())
    assert run.id == "run_h1" == run.run_id


# ------------------------------------------------------------ #263 unserve


def test_unserve_deletes_the_model_row_and_is_exported_both_ways():
    gate = _Gate()
    assert wai.unserve("My-Model", transport=gate) == {"name": "my-model", "deleted": True}
    assert gate.calls == [("DELETE", "/models/my-model", None)]
    assert wai.delete_model is wai.unserve
    with pytest.raises(ValueError, match="name"):
        wai.unserve("  ", transport=gate)


# ------------------------------------------------------------ #264 thinking


def test_spoken_text_never_carries_think_markup():
    assert agents._spoken_text({"content": "<think>\nhmm\n</think>The answer is 4."}) == (
        "The answer is 4."
    )
    assert agents._spoken_text({"content": "<think>"}) == ""
    assert agents._spoken_text({"content": "<think>\n</think>"}) == ""
    assert agents._spoken_text({"content": "plain"}) == "plain"


def test_local_model_thinking_knob_reaches_the_request(monkeypatch):
    seen = []

    def fake_complete(_url, _model, messages, **kwargs):
        seen.append(kwargs.get("extra"))
        return {"content": "done.", "_finish_reason": "stop"}

    monkeypatch.setattr(agents, "complete", fake_complete)
    tools = [{"name": "noop", "description": "n", "parameters": {"type": "object"}}]
    for thinking, want in (
        (False, {"chat_template_kwargs": {"enable_thinking": False}}),
        (None, None),
    ):
        seen.clear()
        agent = agents.local_model(
            "http://127.0.0.1:9/v1", "served-adapter", tools=tools, thinking=thinking, max_turns=1
        )
        agent("hi")
        assert seen and all(x == want for x in seen)


def test_complete_extra_lands_in_the_payload(monkeypatch):
    import json as _json

    payloads = []

    class Resp:
        status = 200

        def read(self):
            return b'{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}'

        def getheader(self, _name):
            return None

    class Conn:
        def request(self, _method, _path, body=None, headers=None):
            payloads.append(_json.loads(body))

        def getresponse(self):
            return Resp()

    monkeypatch.setattr(agents, "_thread_connection", lambda *_a, **_k: Conn())
    _real_complete(
        "http://127.0.0.1:9/v1",
        "m",
        [{"role": "user", "content": "hi"}],
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


# ------------------------------------------------------------ #260 pairs the trainer can use


def _rollout(prompt, i, reward, final):
    return {
        "prompt": prompt,
        "scenario_id": prompt,
        "rollout_index": i,
        "steps": [
            {"tool": "open_source", "arguments": {"url": "u"}, "result": {"status": "timeout"}}
        ],
        "final_text": final,
        "reward": reward,
        "judge_status": "ok",
    }


def test_pairs_say_when_the_first_turns_read_the_same():
    rows = []
    for p in range(10):
        rows.append(_rollout(f"p{p}", 0, 1, "The source timed out; no finding."))
        rows.append(_rollout(f"p{p}", 1, 0, "The source says revenue grew 40%."))
    pairs, report = build_preference_pairs(rows)
    assert report["pairs"] == 10
    assert report["first_turn_identical"] == 10 and report["trainer_pairs"] == 0
    assert all(p["first_turn_differs"] is False for p in pairs)
    assert any("hosted DPO trainer compares first turns only" in w for w in report["warnings"])
    exported = wai.export_preference(pairs, validate=False)
    assert exported["first_turn_identical"] == 10
    assert any("first turns only" in w for w in exported["warnings"])


def test_pairs_whose_first_turns_differ_carry_no_note():
    rows = []
    for p in range(3):
        good = _rollout(f"p{p}", 0, 1, "ok")
        bad = _rollout(f"p{p}", 1, 0, "bad")
        bad["steps"][0]["arguments"] = {"url": "other"}
        rows += [good, bad]
    pairs, report = build_preference_pairs(rows)
    assert report["first_turn_identical"] == 0 and report["trainer_pairs"] == 3
    assert all(p["first_turn_differs"] for p in pairs)
    assert not any("first turns" in w for w in report["warnings"])


def test_first_turn_reads_the_tool_call_then_text_then_final():
    assert _first_turn({"steps": [{"text": "hi"}, {"tool": "t", "arguments": {"a": 1}}]}) == (
        '{"arguments": {"a": 1}, "name": "t"}'
    )
    assert _first_turn({"steps": [{"text": "hi"}], "final_text": "bye"}) == "hi"
    assert _first_turn({"messages": [{"role": "assistant", "content": "m"}]}) == "m"
    assert _first_turn({"final_text": "bye"}) == "bye"
