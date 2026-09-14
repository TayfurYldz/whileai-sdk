"""Publishing a dataset as a public card, and pulling public sets with no key."""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.ingest import platform


class Recorder:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, method, path, api_key, body=None, *, raw_url=None, public=False, **kw):
        self.calls.append((method, path, body, public, raw_url))
        if raw_url:
            return b'{"prompt": "p"}\n'
        reply = self.replies.get((method, path))
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_publish_posts_agent_and_description(monkeypatch):
    rec = Recorder(
        {("POST", "/datasets/ds_1/publish"): {"datasetId": "ds_1", "agent": "airline-support"}}
    )
    monkeypatch.setattr(platform, "_call", rec)
    card = zps.publish("ds_1", "airline-support", "Graded refunds.")
    assert card["agent"] == "airline-support"
    assert rec.calls == [
        (
            "POST",
            "/datasets/ds_1/publish",
            {"agent": "airline-support", "description": "Graded refunds."},
            False,
            None,
        )
    ]
    zps.unpublish("ds_1")
    assert rec.calls[-1][:2] == ("POST", "/datasets/ds_1/unpublish")


def test_catalog_needs_no_key(monkeypatch):
    rec = Recorder({("GET", "/catalog"): {"datasets": [], "agents": []}})
    monkeypatch.setattr(platform, "_call", rec)
    assert zps.catalog() == {"datasets": [], "agents": []}
    assert rec.calls[0][3] is True, "public call, no X-Api-Key"


def test_pull_uses_the_catalog_when_there_is_no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path))
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_DELEGATED_CREDENTIAL", raising=False)
    rec = Recorder({("GET", "/catalog/ds_pub/download"): {"parts": ["https://s3/x"]}})
    monkeypatch.setattr(platform, "_call", rec)
    rows = zps.pull("ds_pub")
    assert rows == [{"prompt": "p"}]
    assert rec.calls[0][:2] == ("GET", "/catalog/ds_pub/download") and rec.calls[0][3] is True


def test_pull_prefers_your_own_copy_and_falls_back_to_the_catalog(monkeypatch):
    monkeypatch.setenv("ZEROPROOF_API_KEY", "zp_x")
    rec = Recorder(
        {
            ("GET", "/datasets/ds_pub/download"): platform.PlatformError(
                "GET /datasets/ds_pub/download -> 404: No such dataset on your account"
            ),
            ("GET", "/catalog/ds_pub/download"): {"parts": ["https://s3/x"]},
        }
    )
    monkeypatch.setattr(platform, "_call", rec)
    assert zps.pull("ds_pub") == [{"prompt": "p"}]
    assert [c[1] for c in rec.calls[:2]] == [
        "/datasets/ds_pub/download",
        "/catalog/ds_pub/download",
    ]

    rec = Recorder(
        {
            ("GET", "/datasets/ds_pub/download"): platform.PlatformError(
                "GET ... -> 401: Invalid API key"
            )
        }
    )
    monkeypatch.setattr(platform, "_call", rec)
    with pytest.raises(platform.PlatformError, match="401"):
        zps.pull("ds_pub")


def test_push_with_publish_needs_an_agent_and_returns_the_card(monkeypatch):
    from tests.helpers import simulate_offline

    data = simulate_offline(budget=2, seed=0)
    calls = []

    def fake_push_rows(rows, name, *, api_key=None, parent=None, **meta):
        calls.append(("push", name))
        return {"datasetId": "ds_new"}

    def fake_publish(dataset_id, agent, description=None, *, api_key=None):
        calls.append(("publish", dataset_id, agent, description))
        return {"datasetId": dataset_id, "agent": agent}

    monkeypatch.setattr("zeroproof.simulations.data.push_rows", fake_push_rows)
    monkeypatch.setattr(platform, "publish", fake_publish)
    with pytest.raises(ValueError, match="agent"):
        data.push("run", publish=True)
    out = data.push("run", agent="github", publish=True, description="d")
    assert out["datasetId"] == "ds_new" and out["card"]["agent"] == "github"
    assert calls == [("push", "run"), ("publish", "ds_new", "github", "d")]
