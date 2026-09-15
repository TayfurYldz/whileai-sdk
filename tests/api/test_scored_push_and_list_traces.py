"""``ScoredData.push`` and ``list_traces()`` resolve the key like every other call."""

from __future__ import annotations

import pytest

import zeroproof
from tests.helpers import simulate_offline
from zeroproof.simulations.score import judging


def test_scored_data_push_is_push_rows_on_the_graded_copies(monkeypatch):
    data = simulate_offline(budget=4, concurrency=1)
    scored = data.grade(judge=lambda row: {"reward": 1})
    seen = {}

    def fake_push_rows(rows, name, **kwargs):
        seen["rows"], seen["name"], seen["kwargs"] = rows, name, kwargs
        return {"datasetId": "ds_test"}

    monkeypatch.setattr("zeroproof.simulations.ingest.platform.push_rows", fake_push_rows)
    out = scored.push("demo-rl", gate=True, mode="rl", agent="demo")
    assert out == {"datasetId": "ds_test"}
    assert seen["rows"] is scored.rows
    assert seen["name"] == "demo-rl"
    assert seen["kwargs"] == {"gate": True, "mode": "rl", "agent": "demo"}
    assert isinstance(scored, judging.ScoredData)


def test_list_traces_reads_the_saved_key(monkeypatch):
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.setattr("zeroproof.auth.stored_api_key", lambda: "zp_saved")
    seen = {}

    class _Res:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"traces": []}

    def fake_get(url, headers=None, timeout=None):
        seen["url"], seen["headers"] = url, headers
        return _Res()

    monkeypatch.setattr("zeroproof.ingest.requests.get", fake_get)
    assert zeroproof.list_traces() == {"traces": []}
    assert seen["headers"] == {"X-Api-Key": "zp_saved"}
    assert seen["url"].endswith("/traces")


def test_list_traces_without_any_key_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.setattr("zeroproof.auth.stored_api_key", lambda: None)
    with pytest.raises(zeroproof.ZeroProofIngestError, match="zeroproof login"):
        zeroproof.list_traces()
