"""Invented tool execution must not look like a real pass."""

import json

from zeroproof.simulations.world.sandbox import MockEnvironment


def _shell_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the project test suite.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Read one order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        },
    ]


def _blob(out: dict) -> str:
    return json.dumps(out).lower()


def _shell_payload(out: dict) -> dict:
    data = out.get("data")
    return data if isinstance(data, dict) else out


def test_invented_run_tests_never_reports_suite_passed():
    for seed in range(32):
        env = MockEnvironment(_shell_tools(), seed=seed)
        env._exists = lambda value: True
        out = env.call("run_tests", {"command": "pytest"})
        blob = _blob(out)
        payload = _shell_payload(out)
        assert out["status"] == "error"
        assert out["reason"] == "invented_execution"
        assert out.get("invented") is True
        assert payload.get("invented") is True
        assert payload.get("exit_code") == 1
        assert " passed" not in blob
        assert "suite passed" not in blob
        assert payload.get("exit_code") != 0


def test_unknown_command_does_not_exit_zero():
    env = MockEnvironment(_shell_tools())
    env._exists = lambda value: True
    out = env.call("run_command", {"command": "not-a-real-binary --wat"})
    payload = _shell_payload(out)
    assert out["status"] == "error"
    assert out["reason"] == "invented_execution"
    assert payload.get("exit_code") == 1
    assert payload.get("exit_code") != 0


def test_writer_shape_cannot_invent_a_green_suite():
    env = MockEnvironment(
        _shell_tools(),
        result_shapes={
            "run_tests": {
                "command": "pytest",
                "exit_code": 0,
                "stdout": "============================== 12 passed in 0.10s "
                "===============================",
                "stderr": "",
            }
        },
    )
    env._exists = lambda value: True
    out = env.call("run_tests", {"command": "pytest"})
    blob = _blob(out)
    payload = _shell_payload(out)
    assert out["status"] == "error"
    assert payload.get("exit_code") == 1
    assert "12 passed" not in blob
    assert " passed" not in blob


def test_empty_trace_file_read_is_labeled_invented():
    env = MockEnvironment(_shell_tools())
    env._exists = lambda value: True
    out = env.call("read_file", {"path": "src/app.py"})
    data = out.get("data") or {}
    assert out["status"] == "ok"
    assert data.get("invented") is True
    assert isinstance(data.get("content"), str)


def test_invented_write_does_not_claim_success():
    env = MockEnvironment(_shell_tools())
    out = env.call("write_file", {"path": "src/app.py", "content": "x = 1\n"})
    assert out["status"] == "error"
    assert out["reason"] == "invented_execution"
    assert out.get("invented") is True
    assert "bytes_written" not in out


def test_record_tools_still_answer_without_execute():
    env = MockEnvironment(_shell_tools(), world_state="entity exists")
    out = env.call("lookup_order", {"order_id": "ORD-9"})
    assert out["status"] in {"ok", "created"}
    assert out.get("reason") != "invented_execution"
    data = out.get("data") or out
    assert data.get("invented") is not True
