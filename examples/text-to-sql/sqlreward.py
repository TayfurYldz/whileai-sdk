"""Execution-match reward for text-to-SQL, self-contained for the Modal container.

Same rule as common.SQLExec: run the candidate on the seeded store database,
compare the result to the gold query's result (multiset of rows, floats to 2dp,
column permutation allowed, ordered only when the gold has ORDER BY).
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from typing import Any

DSN = os.environ.get("T2S_PG_DSN") or "postgresql://postgres@127.0.0.1:5499/shop"
MAX_ROWS = 200
STATEMENT_TIMEOUT_MS = 5000
_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_SELECT_START = re.compile(r"\b(select|with)\b", re.I)
_tls = threading.local()


# ----------------------------------------------------------------- postgres in the container


def start_postgres(
    schema_sql: str, seed_sql: str, data_dir: str = "/tmp/pgdata", port: int = 5499
) -> None:
    """initdb + start a trust-auth cluster as the postgres user, load the store."""
    bins = sorted(glob.glob("/usr/lib/postgresql/*/bin"))
    if not bins:
        raise RuntimeError("no postgresql binaries in the image")
    pgbin = bins[-1]
    subprocess.run(["chown", "-R", "postgres:postgres", os.path.dirname(data_dir)], check=False)
    if not os.path.exists(os.path.join(data_dir, "PG_VERSION")):
        subprocess.run(
            [
                "su",
                "postgres",
                "-c",
                f"{pgbin}/initdb -D {data_dir} -A trust -E UTF8 >/tmp/initdb.log 2>&1",
            ],
            check=True,
        )
    subprocess.run(
        [
            "su",
            "postgres",
            "-c",
            f"{pgbin}/pg_ctl -D {data_dir} -o '-p {port} -c listen_addresses=127.0.0.1' -l /tmp/pg.log -w start",
        ],
        check=True,
    )
    for _ in range(30):
        if (
            subprocess.run(
                ["su", "postgres", "-c", f"{pgbin}/pg_isready -h 127.0.0.1 -p {port}"],
                capture_output=True,
            ).returncode
            == 0
        ):
            break
        time.sleep(1)
    subprocess.run(
        ["su", "postgres", "-c", f"{pgbin}/createdb -h 127.0.0.1 -p {port} shop"], check=False
    )
    with open("/tmp/schema.sql", "w") as f:
        f.write(schema_sql)
    with open("/tmp/seed.sql", "w") as f:
        f.write(seed_sql)
    subprocess.run(
        [
            "su",
            "postgres",
            "-c",
            f"{pgbin}/psql -q -v ON_ERROR_STOP=1 -h 127.0.0.1 -p {port} -d shop -f /tmp/schema.sql -f /tmp/seed.sql",
        ],
        check=True,
    )
    n = run_sql("select count(*) from orders")[0][0]
    print(f"postgres up on {port}: {n} orders")


# ----------------------------------------------------------------- execution


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
            return _norm(float(s.replace(",", "").replace("$", "")))
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


def verdict(text: str, gold_sql: str) -> tuple[int, int, str]:
    """(correct 0/1, executes 0/1, reason)."""
    sql = extract_sql(text)
    if not sql:
        return 0, 0, "no sql query in reply"
    try:
        got = norm_rows(run_sql(sql))
    except Exception as exc:
        return 0, 0, f"sql error: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
    if len(got) > MAX_ROWS:
        return 0, 1, "result too large"
    want = gold_rows(gold_sql)
    ok = equivalent(got, want, ordered=has_order_by(gold_sql))
    return (
        (1 if ok else 0),
        1,
        ("result matches" if ok else f"result differs: got {len(got)} rows, want {len(want)}"),
    )


def shaped_reward(text: str, gold_sql: str) -> float:
    """Training reward: 1.0 correct, 0.1 runs but wrong, 0 no query / error."""
    correct, executes, _ = verdict(text, gold_sql)
    if correct:
        return 1.0
    return 0.1 if executes else 0.0


def reward_rows(
    tasks: list[dict], replies: list[list[str]], model: str, system_prompt: str
) -> list[dict]:
    """SDK rows (one per reply) graded with the binary verdict, for pass_at and delta."""
    rows = []
    for task, group in zip(tasks, replies):
        for i, text in enumerate(group):
            correct, executes, reason = verdict(text, task["sql"])
            rows.append(
                {
                    "scenario_id": task["id"],
                    "rollout_index": i,
                    "prompt": task["question"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task["question"]},
                        {"role": "assistant", "content": text},
                    ],
                    "final_text": text,
                    "steps": [],
                    "privileged": {"reference": task["sql"]},
                    "model_version": model,
                    "category": task["archetype"],
                    "difficulty": task["difficulty"],
                    "reward": correct,
                    "reason": reason,
                    "judge_status": "ok",
                    "judge_name": "sql_exec",
                    "markers": {
                        "executes": executes,
                        "has_sql": int(extract_sql(text) is not None),
                    },
                }
            )
    return rows
