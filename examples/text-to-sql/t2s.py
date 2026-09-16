"""Shared pieces: the schema prompt, the SQL execution verifier, row shape, split.

Schema = a small online-store Postgres database (schema.sql + seed.sql, generated
by gen_seed.py, seed 42). The verifier runs candidate SQL on it read-only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import threading
from collections import Counter
from datetime import date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path
from typing import Any

from schema_prompt import NOTES, ddl, system_prompt  # noqa: F401  (re-exported)

from zeroproof.simulations.verify.base import Verifier

HERE = Path(__file__).resolve().parent
DSN = os.environ.get("T2S_PG_DSN") or "postgresql://postgres@127.0.0.1:5499/shop"
TASKS = HERE / "tasks.jsonl"
RAW = HERE / "raw"
OUT = HERE / "out"
AGENT = "text-to-sql-shop"
HOLDOUT = 0.2
MAX_ROWS = 200
STATEMENT_TIMEOUT_MS = 5000

TABLES = [
    "categories",
    "products",
    "customers",
    "employees",
    "orders",
    "order_items",
    "payments",
    "reviews",
]

ARCHETYPES = [
    "single-table aggregation (COUNT/SUM/AVG/MIN/MAX with WHERE filters)",
    "top-N / ranking with ORDER BY and LIMIT",
    "two-table JOIN with a filter or aggregate",
    "multi-table JOIN (three or more tables)",
    "GROUP BY breakdown with HAVING or a per-group aggregate",
    "NULL semantics (shipped_at, sales_rep_id, discount_pct, referred_by, review title)",
    "self-join / hierarchy (category parent, employee manager, customer referrer)",
    "date and time (DATE_TRUNC, EXTRACT, month or year buckets, intervals, first/last event)",
    "anti-join / existence (customers with no orders, products never reviewed, orders without a captured payment, NOT EXISTS / NOT IN / LEFT JOIN IS NULL)",
    "derived metric (order total with discount and shipping, margin, average rating, share of total, days to ship)",
]
DIFFICULTIES = ["easy", "medium", "hard"]
STYLES = [
    "casual business user asking a quick question",
    "formal reporting request from a manager",
    "terse power-user shorthand",
    "precise analyst specification that may name columns",
]

# ----------------------------------------------------------------- SQL run

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_SELECT_START = re.compile(r"\b(select|with)\b", re.I)
_tls = threading.local()


def extract_sql(text: str) -> str | None:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = _SQL_BLOCK.findall(text)
    sql = blocks[-1] if blocks else None
    if sql is None:
        m = _SELECT_START.search(text)
        if not m:
            return None
        sql = text[m.start() :]
    sql = sql.strip().rstrip(";").strip()
    if not _SELECT_START.match(sql):
        return None
    return sql


def _conn():
    import psycopg

    conn = getattr(_tls, "conn", None)
    if conn is None or conn.closed:
        conn = psycopg.connect(DSN, autocommit=True)
        conn.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        conn.execute("SET default_transaction_read_only = on")
        _tls.conn = conn
    return conn


def run_sql(sql: str, limit: int = MAX_ROWS + 1) -> list[tuple]:
    """Execute a read-only SELECT on the store database. Raises on error."""
    if not _SELECT_START.match(sql.strip()):
        raise ValueError("not a SELECT")
    if ";" in sql.rstrip(";"):
        raise ValueError("one statement only")
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)  # type: ignore[arg-type]
            if cur.description is None:
                return []
            return cur.fetchmany(limit)
    except Exception:
        # autocommit: nothing to roll back, but a broken connection must be dropped
        if conn.closed or getattr(conn, "broken", False):
            _tls.conn = None
        raise


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, (int, float)):
        r = round(float(v), 2)
        return 0.0 if r == 0 else r
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, dtime):
        return v.isoformat()
    if isinstance(v, timedelta):
        return round(v.total_seconds() / 86400, 2)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            f = float(s.replace(",", "").replace("$", ""))
            return _norm(f)
        except ValueError:
            return s
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return str(v)


def norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, re.I) is not None


def _key(t: tuple) -> str:
    return json.dumps(t, default=str)


def equivalent(cand: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Spider-style execution match: same multiset of rows (same sequence
    when the gold orders), column names ignored, columns may be permuted."""
    if len(cand) != len(gold):
        return False
    if not gold:
        return True
    ncols = len(gold[0])
    if any(len(r) != ncols for r in cand):
        return False
    perms = [tuple(range(ncols))]
    if ncols <= 5:
        perms += [p for p in itertools.permutations(range(ncols)) if p != perms[0]]
    for perm in perms:
        c = [tuple(r[i] for i in perm) for r in cand]
        if ordered:
            if c == gold:
                return True
        elif Counter(map(_key, c)) == Counter(map(_key, gold)):
            return True
    return False


_gold_cache: dict[str, list[tuple]] = {}
_gold_lock = threading.Lock()


def gold_rows(sql: str) -> list[tuple]:
    h = hashlib.sha256(sql.encode()).hexdigest()
    with _gold_lock:
        if h in _gold_cache:
            return _gold_cache[h]
    rows = norm_rows(run_sql(sql))
    with _gold_lock:
        _gold_cache[h] = rows
    return rows


class SQLExec(Verifier):
    """Reward 1 when the candidate SQL's result matches the gold SQL's result."""

    kind = "rule"

    def __init__(self):
        super().__init__(name="sql_exec")

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        gold_sql = reference if isinstance(reference, str) else (reference or {}).get("sql")
        if not gold_sql:
            return None, "no gold sql"
        sql = extract_sql(candidate)
        if not sql:
            return 0, "no sql query in reply"
        try:
            got = norm_rows(run_sql(sql))
        except Exception as exc:  # postgres errors are the signal here
            msg = str(exc).splitlines()[0][:160]
            return 0, f"sql error: {type(exc).__name__}: {msg}"
        if len(got) > MAX_ROWS:
            return 0, f"result too large (>{MAX_ROWS} rows)"
        want = gold_rows(gold_sql)
        ok = equivalent(got, want, ordered=has_order_by(gold_sql))
        if ok:
            return 1, f"result matches ({len(want)} rows)"
        return (
            0,
            f"result differs: got {len(got)} rows x {len(got[0]) if got else 0} cols, want {len(want)} x {len(want[0]) if want else 0}",
        )


# ----------------------------------------------------------------- tasks


def task_id(question: str) -> str:
    return "t2s_" + hashlib.sha256(question.strip().lower().encode()).hexdigest()[:12]


def load_tasks(path: Path = TASKS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def bucket(scenario_id: str) -> float:
    h = hashlib.sha256(str(scenario_id).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_of(scenario_id: str) -> str:
    return "holdout" if bucket(scenario_id) < HOLDOUT else "train"


# ----------------------------------------------------------------- rows


def make_row(
    task: dict,
    rollout_index: int,
    model: str,
    reply: str,
    *,
    finish_reason: str | None = None,
    usage: dict | None = None,
    latency_s: float | None = None,
) -> dict:
    sys_p = system_prompt()
    return {
        "scenario_id": task["id"],
        "rollout_index": rollout_index,
        "prompt": task["question"],
        "system_prompt": sys_p,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": task["question"]},
            {"role": "assistant", "content": reply},
        ],
        "final_text": reply,
        "steps": [],
        "privileged": {"reference": task["sql"]},
        "model_version": model,
        "category": task["archetype"],
        "difficulty": task["difficulty"],
        "style": task.get("style"),
        "split": split_of(task["id"]),
        "agent": AGENT,
        "truncated": finish_reason == "length",
        "finish_reason": finish_reason,
        "usage": usage,
        "latency_s": latency_s,
    }


def teacher_row(task: dict) -> dict:
    """The gold SQL as a demonstration (a Rollout by the teacher, keyed by task)."""
    reply = f"```sql\n{task['sql']}\n```"
    row = make_row(task, 0, "teacher:gold", reply, finish_reason="stop")
    row["reward"] = 1
    row["reason"] = "gold sql"
    row["judge_status"] = "ok"
    row["judge_name"] = "sql_exec"
    return row


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
