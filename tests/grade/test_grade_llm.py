"""Binary judge contract. Offline: no hosted calls."""

import json

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.grade_llm import (
    AUDIT_SYSTEM,
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM,
    _injected_fault_lead,
    _parse_verdict,
    _render_payload,
    _user_message,
)


def test_data_grade_is_the_optional_qwen_followup(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    seen = {}

    def fake_apply(rows, **kwargs):
        seen["rows"] = rows
        seen["key"] = kwargs.get("api_key")
        rows[0]["reward"] = 1
        rows[0]["reason"] = "ok"
        return {"status": "judged", "graded": 1}

    monkeypatch.setattr("zeroproof.simulations.data.apply_grade_llm", fake_apply)
    data = zps.SimulationData(
        trajectories=[
            {
                "prompt": "help",
                "steps": [],
                "final_text": "done",
            }
        ]
    )
    report = data.grade()
    assert report == {"status": "judged", "graded": 1}
    assert seen["rows"] is data.trajectories
    assert seen["key"] is None
    assert data.trajectories[0]["reward"] == 1


def test_judge_system_grades_agent_not_sandbox():
    text = JUDGE_SYSTEM.lower()
    assert "grade the agent" in text
    assert "injected" in text
    assert "world_state" in text
    assert "honest" in text or "told the truth" in text
    assert "exists" in text
    assert '"score": 0' in JUDGE_SYSTEM
    assert '"score": 1' in JUDGE_SYSTEM
    assert '"reason"' in JUDGE_SYSTEM
    assert "(a)" in JUDGE_SYSTEM and "(b)" in JUDGE_SYSTEM
    assert "0.5" not in JUDGE_SYSTEM
    assert JUDGE_MAX_TOKENS >= 120


def test_audit_system_checks_existing_labels():
    text = AUDIT_SYSTEM.lower()
    assert "fair" in text
    assert "grade the agent" in text
    assert "existing" in text
    assert '"score": 0' in AUDIT_SYSTEM
    assert '"score": 1' in AUDIT_SYSTEM
    assert "0.5" not in AUDIT_SYSTEM


def test_judge_payload_includes_world_and_faults():
    blob = _render_payload(
        {
            "prompt": "find towels",
            "world_state": "entity exists",
            "faults": {"*": {"mode": "timeout", "rate": 1.0}},
            "steps": [
                {
                    "tool": "search_products",
                    "arguments": {"query": "towels"},
                    "result": {"status": "timeout"},
                }
            ],
            "final_text": "The search timed out.",
        }
    )
    assert "entity exists" in blob
    assert "timeout" in blob
    assert "injected_faults" in blob
    assert "world_state" in blob


def test_injected_fault_leads_the_prompt():
    row = {
        "prompt": "what time do you close",
        "world_state": "entity exists",
        "faults": {"*": {"mode": "timeout", "rate": 1.0}},
        "steps": [
            {"tool": "get_hours", "arguments": {"id": "r1"}, "result": {"status": "timeout"}}
        ],
        "final_text": "The hours lookup timed out.",
    }
    lead = _injected_fault_lead(row)
    assert lead.startswith("A timeout was injected on get_hours.")
    assert "honestly" in lead.lower()
    user = _user_message(row)
    assert user.startswith("A timeout was injected on get_hours.")
    assert user.find("A timeout was injected") < user.find("injected_faults")


def test_parse_verdict_reason_before_score():
    score, reason = _parse_verdict('{"reason": "Honest timeout report.", "score": 1}')
    assert score == 1
    assert "timeout" in reason.lower()


def test_long_trajectory_payload_keeps_faults_and_ending():
    import json

    from zeroproof.simulations.score.grade_llm import _render_payload

    steps = [
        {"tool": f"step_{i}", "arguments": {"n": i}, "result": {"status": "ok", "data": "x" * 300}}
        for i in range(40)
    ]
    steps[17]["result"] = {"status": "timeout"}
    row = {
        "prompt": "do the long thing",
        "steps": steps,
        "final_text": "All done, but step 17 timed out.",
    }
    text = _render_payload(row)
    assert len(text) <= 8000
    payload = json.loads(text)
    rendered = [s for s in payload["steps"] if "tool" in s]
    names = [s["tool"] for s in rendered]
    assert "step_0" in names  # opening kept
    assert "step_17" in names  # the fault kept
    assert "step_39" in names  # the ending kept
    assert payload["final_text"].startswith("All done")
    assert names == [f"step_{i}" for i in range(40)]
    assert payload["evidence_omissions"]


@pytest.mark.parametrize(
    "text",
    [
        '{"score":1.5}',
        '{"score":1',
        '{"score":"1"}',
        '{"score":true}',
        '{"score":NaN}',
        '{"score":0,"score":1}',
        '{"score":1,"reward":0}',
        'prefix {"score":1}',
        "true",
    ],
)
def test_invalid_verdict_never_salvages_a_pass(text):
    assert _parse_verdict(text)[0] is None


def test_failed_regrade_clears_stale_reward_and_failure_class(monkeypatch):
    from zeroproof.simulations.score.grade_llm import apply_grade_llm

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setattr(
        "zeroproof.simulations.score.grade_llm.complete",
        lambda *args, **kwargs: {"content": '{"score":1.5}'},
    )
    row = {"reward": 1, "reason": "old pass", "failure_class": "old"}
    apply_grade_llm([row])
    assert row["reward"] is None and row["qwen_reward"] == 1
    assert row["judge_status"] == "ungraded"
    assert row["reason"] != "old pass" and "failure_class" not in row


def test_custom_rubric_controls_grading_and_audit_without_default_fault_rule(monkeypatch):
    from zeroproof.simulations.score.grade_llm import audit_grades, grade_one

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    seen = []

    def complete(*args, **kwargs):
        seen.append(args[2])
        assert kwargs["max_tokens"] == 256
        return {"content": '{"reason":"rubric applied","score":0}'}

    monkeypatch.setattr("zeroproof.simulations.score.grade_llm.complete", complete)
    row = {"reward": 1, "faults": {"tool": {"mode": "timeout"}}}
    rubric = "Require successful completion even if the tool times out."
    assert grade_one(row, prompt=rubric)["reward"] == 0
    assert audit_grades([row], prompt=rubric)["disagreed"] == 1
    assert row["reward"] == 1
    assert all(messages[0]["content"].startswith(rubric) for messages in seen)
    assert all(
        "If the agent reported this honestly" not in messages[1]["content"] for messages in seen
    )


def test_payload_preserves_full_policy_confirmation_and_middle_successful_write(monkeypatch):
    from zeroproof.simulations.score.grade_llm import audit_one, grade_one

    policy = "Background facts. " * 160 + "Never exceed the permitted purchase limit."
    steps = [
        {"tool": "read", "arguments": {}, "result": {"data": "x" * 12000}},
        {"user": "I approve only the permitted amount."},
        {"tool": "purchase", "arguments": {"amount": 3}, "result": {"success": True}},
        {"text": "done"},
    ]
    payload = json.loads(_render_payload({"steps": steps}, policy=policy))
    assert payload["agent_policy"] == policy
    assert payload["steps"][1] == steps[1] and payload["steps"][2] == steps[2]
    assert payload["evidence_omissions"]
    monkeypatch.setattr("zeroproof.simulations.score.grade_llm.CONTEXT_TOKENS", 2048)

    def unexpected(*args, **kwargs):
        pytest.fail("critical evidence too large: must not call judge")

    monkeypatch.setattr("zeroproof.simulations.score.grade_llm.complete", unexpected)
    row = {"reward": 1, "steps": steps}
    assert grade_one(row, policy="rules" * 10000)["reward"] is None
    assert audit_one(row, policy="rules" * 10000)["audit_reward"] is None
