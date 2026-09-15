"""zps.send_score: grading a run that already ran, which is what cut() filters on."""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.ingest.platform import PlatformError


@pytest.fixture
def calls(monkeypatch):
    """Record what the client sends, and answer as the gate does."""
    seen: list[dict] = []

    def fake_call(method, path, api_key=None, body=None, **kw):
        seen.append({"method": method, "path": path, "body": body, "kw": kw})
        return {"applied": [{"traceId": "t1", "names": ["score"]}]}

    from zeroproof.simulations.ingest import platform

    monkeypatch.setattr(platform, "_call", fake_call)
    return seen


def answer(monkeypatch, reply):
    from zeroproof.simulations.ingest import platform

    monkeypatch.setattr(platform, "_call", lambda *a, **k: reply)


def test_one_run_passed(calls):
    zps.send_score("4bf92f3577b34da6", 1.0)

    assert calls[0]["method"] == "POST"
    assert calls[0]["path"] == "/v1/scores"
    assert calls[0]["body"] == {"traceId": "4bf92f3577b34da6", "name": "score", "value": 1.0}
    # /v1/scores authenticates on X-Api-Key, never a bearer token.
    assert calls[0]["kw"]["require_api_key"] is True


def test_true_and_false_are_the_pass_rule(calls):
    zps.send_score("t1", True)
    zps.send_score("t1", False)

    # 1.0, not "truthy": the cut's pass rule is reward >= 1.0.
    assert calls[0]["body"]["value"] == 1.0
    assert calls[1]["body"]["value"] == 0.0


def test_a_named_measurement_sits_beside_the_verdict(calls):
    zps.send_score("t1", 0.82, name="helpfulness", description="judge, 0-1")

    assert calls[0]["body"] == {
        "traceId": "t1",
        "name": "helpfulness",
        "value": 0.82,
        "description": "judge, 0-1",
    }


def test_a_word_needs_no_number(calls):
    zps.send_score("t1", name="verdict", label="refused")

    assert calls[0]["body"] == {"traceId": "t1", "name": "verdict", "label": "refused"}


def test_a_batch_across_runs(calls):
    zps.send_score(scores=[{"traceId": " t1 ", "value": 1}, {"traceId": "t2", "value": 0}])

    assert calls[0]["body"] == {
        "scores": [
            {"traceId": "t1", "value": 1, "name": "score"},
            {"traceId": "t2", "value": 0, "name": "score"},
        ]
    }


def test_several_measurements_about_one_run_share_its_id(calls):
    zps.send_score(
        "t1", scores=[{"name": "helpfulness", "value": 0.8}, {"value": 1.0}], pass_at=1.0
    )

    assert calls[0]["body"] == {
        "traceId": "t1",
        "scores": [
            {"name": "helpfulness", "value": 0.8, "pass_at": 1.0},
            {"value": 1.0, "name": "score", "pass_at": 1.0},
        ],
    }


def test_no_trace_id_is_caught_before_the_round_trip(calls):
    with pytest.raises(PlatformError, match="trace id"):
        zps.send_score(value=1.0)
    with pytest.raises(PlatformError, match="traceId"):
        zps.send_score(scores=[{"value": 1.0}])

    assert calls == []


def test_a_measurement_that_says_nothing_is_caught(calls):
    with pytest.raises(PlatformError, match="value"):
        zps.send_score("t1")

    assert calls == []


def test_a_trace_id_nobody_sent_raises_instead_of_looking_fine(monkeypatch):
    answer(monkeypatch, {"applied": [], "unknown": ["t9"]})

    with pytest.raises(PlatformError, match="No run on this account"):
        zps.send_score("t9", 1.0)


def test_a_partly_good_batch_comes_back_whole(monkeypatch):
    answer(
        monkeypatch,
        {"applied": [{"traceId": "t1", "names": ["score"]}], "unknown": ["t9"]},
    )

    out = zps.send_score(scores=[{"traceId": "t1", "value": 1}, {"traceId": "t9", "value": 1}])

    assert out["unknown"] == ["t9"]


def test_the_gates_own_complaint_is_what_the_caller_reads(monkeypatch):
    answer(
        monkeypatch,
        {
            "applied": [],
            "rejected": [{"error": 'measurement "x": direction must be higher or lower'}],
        },
    )

    with pytest.raises(PlatformError, match="direction must be higher or lower"):
        zps.send_score("t1", 1.0, direction="sideways")
