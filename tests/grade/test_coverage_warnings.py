"""A hollow run says so: no tool calls, an untouched declared tool, a marker
that fired on no row. ``run_judge`` attaches the notes; ``evaluate(data, ...)``
reads the declared tools off the run."""

import whileai.simulations as wai
from tests.helpers import simulate_offline

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    },
    {"name": "issue_refund", "description": "Refund an order.", "parameters": {"type": "object"}},
]


def _row(steps, markers=None, prompt="Refund A1001"):
    row = {"prompt": prompt, "steps": steps, "final_text": "ok", "reward": 1.0}
    if markers is not None:
        row["markers"] = markers
    return row


def test_no_tool_calls_is_named_when_tools_are_known():
    rows = [_row([]) for _ in range(4)]
    notes = wai.coverage_warnings(rows, tools=TOOLS)
    assert len(notes) == 1
    assert notes[0].startswith("0 of 4 rows called a tool")
    assert "seeds=" in notes[0]


def test_no_tools_declared_means_no_tool_note():
    # A question-and-answer set graded by a verifier is not hollow for
    # calling no tool: nothing was declared.
    rows = [{"prompt": "2+2?", "final_text": "4", "reward": 1.0} for _ in range(4)]
    assert wai.coverage_warnings(rows) == []
    assert wai.coverage_warnings([_row([]) for _ in range(4)]) == []


def test_untouched_declared_tool_is_named():
    lookup = {"tool": "lookup_order", "arguments": {"order_id": "A1001"}, "result": {}}
    rows = [_row([lookup]) for _ in range(4)]
    notes = wai.coverage_warnings(rows, tools=TOOLS)
    assert len(notes) == 1
    assert "1 of 2 declared tools were never called (issue_refund)" in notes[0]
    # Names alone work too (what a SimulationData's declared_tools holds).
    assert wai.coverage_warnings(rows, tools=["lookup_order", "issue_refund"]) == notes


def test_marker_on_zero_rows_is_named():
    lookup = {"tool": "lookup_order", "arguments": {}, "result": {}}
    refund = {"tool": "issue_refund", "arguments": {}, "result": {}}
    rows = [
        _row([lookup, refund], markers={"looked_up_first": 1.0, "escalates_over_limit": None})
        for _ in range(3)
    ]
    notes = wai.coverage_warnings(rows, tools=TOOLS)
    assert len(notes) == 1
    assert "marker 'escalates_over_limit' fired on 0 of 3 rows" in notes[0]


def test_run_judge_attaches_warnings_and_evaluate_reads_tools_off_the_run():
    def silent(message):
        return {"steps": [], "final_text": "Hello."}

    data = simulate_offline(silent, policy="Refund desk.", tools=TOOLS, budget=4)
    assert "no_tool_calls" in data.degraded
    assert any("0 of" in w and "called a tool" in w for w in data.warnings)

    scored = wai.evaluate(data, lambda row: {"reward": 1.0, "reason": "ok"})
    assert scored.warnings and scored.warnings[0].startswith("0 of")
    # A plain row list does not know the declared tools; nothing is claimed.
    scored_rows = wai.run_judge(data.trajectories, lambda row: {"reward": 1.0, "reason": "ok"})
    assert scored_rows.warnings == []
    # ...unless they are passed.
    scored_tools = wai.run_judge(
        data.trajectories, lambda row: {"reward": 1.0, "reason": "ok"}, tools=TOOLS
    )
    assert scored_tools.warnings == scored.warnings


def test_a_full_run_is_quiet():
    def careful(message):
        return {
            "steps": [
                {"tool": "lookup_order", "arguments": {"order_id": "A1001"}, "result": {}},
                {"tool": "issue_refund", "arguments": {"order_id": "A1001"}, "result": {}},
            ],
            "final_text": "Refunded.",
        }

    data = simulate_offline(careful, policy="Refund desk.", tools=TOOLS, budget=4)
    assert "no_tool_calls" not in data.degraded
    scored = wai.evaluate(data, lambda row: {"reward": 1.0, "reason": "ok", "markers": {"m": 1.0}})
    assert scored.warnings == []
