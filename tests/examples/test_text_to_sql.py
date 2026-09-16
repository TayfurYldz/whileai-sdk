"""examples/text-to-sql: the verifier's matching rule, the shipped task set, and
the prompt file the Modal trainer mounts. No database needed."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "text-to-sql"


@pytest.fixture(scope="module")
def sqlreward():  # the verifier module
    sys.path.insert(0, str(EXAMPLE))
    try:
        yield importlib.import_module("sql_verifier")
    finally:
        sys.modules.pop("sql_verifier", None)
        sys.modules.pop("schema_prompt", None)
        sys.path.remove(str(EXAMPLE))


def test_extract_sql_takes_the_last_fenced_query(sqlreward):
    text = "<think>plan</think>\n```sql\nSELECT 1\n```\nand then\n```sql\nSELECT 2;\n```"
    assert sqlreward.extract_sql(text) == "SELECT 2"
    assert sqlreward.extract_sql("no query here") is None
    assert sqlreward.extract_sql("```sql\nDELETE FROM orders\n```") is None
    assert (
        sqlreward.extract_sql("Sure: select count(*) from orders") == "select count(*) from orders"
    )


def test_equivalent_is_a_multiset_match_with_column_permutation(sqlreward):
    gold = sqlreward.norm_rows([("US", 40), ("CA", 15)])
    assert sqlreward.equivalent(sqlreward.norm_rows([(15, "ca"), (40, "US")]), gold, ordered=False)
    assert not sqlreward.equivalent(sqlreward.norm_rows([("US", 40)]), gold, ordered=False)
    assert not sqlreward.equivalent(
        sqlreward.norm_rows([("US", 41), ("CA", 15)]), gold, ordered=False
    )
    # ordered when the gold has ORDER BY: same rows, wrong order, fails
    assert not sqlreward.equivalent(
        sqlreward.norm_rows([("CA", 15), ("US", 40)]), gold, ordered=True
    )
    assert sqlreward.equivalent(
        sqlreward.norm_rows([("US", 40.004), ("CA", 15)]), gold, ordered=True
    )


def test_norm_rounds_money_and_lowercases_text(sqlreward):
    from decimal import Decimal

    assert sqlreward._norm(Decimal("12.345")) == 12.35
    assert sqlreward._norm("  Chase ") == "chase"
    assert sqlreward._norm("1,234.50") == 1234.5
    assert sqlreward._norm(-0.001) == 0.0


def test_shipped_tasks_are_well_formed():
    rows = [json.loads(line) for line in (EXAMPLE / "tasks.jsonl").open(encoding="utf-8")]
    assert len(rows) >= 400
    ids = set()
    for r in rows:
        assert {"id", "question", "sql", "archetype", "difficulty"} <= set(r)
        assert r["sql"].lstrip().lower().startswith(("select", "with"))
        assert r["id"] not in ids
        ids.add(r["id"])
    # the holdout split is a hash of the task id, so it is the same in every script
    hold = [
        r
        for r in rows
        if int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.2
    ]
    assert 60 <= len(hold) <= 110


def test_prompt_file_matches_the_schema_prompt():
    sys.path.insert(0, str(EXAMPLE))
    try:
        sp = importlib.import_module("schema_prompt")
        assert (EXAMPLE / "prompt.txt").read_text(encoding="utf-8") == sp.system_prompt()
        assert "CREATE TABLE orders" in sp.system_prompt()
    finally:
        sys.modules.pop("schema_prompt", None)
        sys.path.remove(str(EXAMPLE))


def test_author_imports_without_the_anthropic_client(monkeypatch):
    """author.py is the bring-your-own-schema entry point; it must at least import."""
    import types

    stub = types.ModuleType("anthropic")
    # author.py builds a client at import; it must construct, nothing more
    stub.Anthropic = stub.AnthropicBedrock = type(
        "Client", (), {"__init__": lambda self, **kw: None}
    )
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    for name in ("author", "sql_verifier", "schema_prompt"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(EXAMPLE))
    try:
        author = importlib.import_module("author")
        assert author.ARCHETYPES and author.DIFFICULTIES and author.STYLES
    finally:
        for name in ("author", "sql_verifier", "schema_prompt"):
            sys.modules.pop(name, None)
        sys.path.remove(str(EXAMPLE))
