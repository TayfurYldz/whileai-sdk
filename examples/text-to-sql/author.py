"""Author (question, gold SQL) tasks with Claude Sonnet 5 on Bedrock.

Every gold query is executed on the snapshot before it is kept: it must be a
SELECT that returns 1..50 rows, not all NULL, stable across two runs, and not a
duplicate of an existing task (by normalized SQL or question overlap).

Usage: python author.py [--rounds 2] [--per-batch 8] [--concurrency 6]
Appends to tasks.jsonl; resumable (batches already done are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2s import (
    ARCHETYPES,
    DIFFICULTIES,
    HERE,
    NOTES,
    STYLES,
    TABLES,
    TASKS,
    ddl,
    load_tasks,
    norm_rows,
    run_sql,
    task_id,
)

# ANTHROPIC_API_KEY -> the Claude API; otherwise AWS credentials -> Bedrock.
if os.environ.get("ANTHROPIC_API_KEY"):
    MODEL = os.environ.get("T2S_AUTHOR_MODEL") or "claude-sonnet-5"
    client: anthropic.Anthropic | anthropic.AnthropicBedrock = anthropic.Anthropic(
        max_retries=3, timeout=180.0
    )
else:
    MODEL = os.environ.get("T2S_AUTHOR_MODEL") or "global.anthropic.claude-sonnet-5"
    client = anthropic.AnthropicBedrock(
        aws_region=os.environ.get("AWS_REGION") or "us-west-2", max_retries=3, timeout=180.0
    )
lock = threading.Lock()
DONE = HERE / "author.done.txt"


def inventory() -> str:
    """What the data looks like, so questions can be specific and answerable."""
    q = run_sql
    out = []
    out.append(
        "Row counts: " + ", ".join(f"{t}={q(f'select count(*) from {t}')[0][0]}" for t in TABLES)
    )
    out.append(
        "categories (id, name, parent_id): "
        + str(q("select category_id,name,parent_id from categories order by 1"))
    )
    out.append(
        "products (id, name, category_id, unit_price, unit_cost, stock_qty, discontinued): "
        + str(
            q(
                "select product_id,name,category_id,unit_price,unit_cost,stock_qty,discontinued from products order by 1"
            )
        )
    )
    out.append(
        "customers by country: "
        + str(q("select country,count(*) from customers group by 1 order by 2 desc"))
    )
    out.append(
        "customer cities: "
        + str(q("select country,city,count(*) from customers group by 1,2 order by 1,3 desc"))
    )
    out.append(
        "customers signup_date range: "
        + str(
            q(
                "select min(signup_date),max(signup_date), count(*) filter (where referred_by is not null) as referred, count(*) filter (where marketing_opt_in) as opted_in from customers"
            )
        )
    )
    out.append(
        "sample customer names: "
        + str(
            q(
                "select customer_id, first_name, last_name, email from customers order by customer_id limit 12"
            )
        )
    )
    out.append(
        "employees: "
        + str(
            q(
                "select employee_id,full_name,department,hire_date,salary,manager_id from employees order by 1"
            )
        )
    )
    out.append(
        "orders by status: "
        + str(q("select status,count(*) from orders group by 1 order by 2 desc"))
    )
    out.append(
        "orders date range + nulls: "
        + str(
            q(
                "select min(order_date),max(order_date), count(*) filter (where shipped_at is null) as unshipped, count(*) filter (where sales_rep_id is null) as self_service, count(*) filter (where discount_pct is null) as no_discount from orders"
            )
        )
    )
    out.append(
        "orders per year-month (top 8): "
        + str(
            q(
                "select to_char(order_date,'YYYY-MM') m,count(*) from orders group by 1 order by 2 desc limit 8"
            )
        )
    )
    out.append(
        "orders shipping_country: "
        + str(q("select shipping_country,count(*) from orders group by 1 order by 2 desc"))
    )
    out.append(
        "orders discount values: "
        + str(q("select discount_pct,count(*) from orders group by 1 order by 1"))
    )
    out.append(
        "orders shipping_cost values: "
        + str(q("select shipping_cost,count(*) from orders group by 1 order by 1"))
    )
    out.append(
        "orders per sales rep: "
        + str(q("select sales_rep_id,count(*) from orders group by 1 order by 1"))
    )
    out.append(
        "order_items: "
        + str(
            q(
                "select count(*), min(quantity), max(quantity), count(distinct order_id), count(*) filter (where oi.unit_price <> p.unit_price) as price_differs from order_items oi join products p using (product_id)"
            )
        )
    )
    out.append(
        "payments (method, status, n): "
        + str(q("select method,status,count(*) from payments group by 1,2 order by 1,2"))
    )
    out.append(
        "reviews by rating: "
        + str(
            q(
                "select rating,count(*), count(*) filter (where title is null) as untitled from reviews group by 1 order by 1"
            )
        )
    )
    out.append(
        "products never reviewed: "
        + str(
            q(
                "select count(*) from products p where not exists (select 1 from reviews r where r.product_id=p.product_id)"
            )
        )
    )
    out.append(
        "customers with no orders: "
        + str(
            q(
                "select count(*) from customers c where not exists (select 1 from orders o where o.customer_id=c.customer_id)"
            )
        )
    )
    out.append(
        "top products by units sold: "
        + str(
            q(
                "select p.name, sum(quantity) from order_items oi join products p using (product_id) group by 1 order by 2 desc limit 8"
            )
        )
    )
    return "\n".join(out)


SYSTEM = """You write text-to-SQL benchmark tasks for a small online-store database (PostgreSQL 16). Each task is a natural-language question plus the ONE gold SQL query that answers it. The tasks train and evaluate a model that must translate the question into SQL, so the question must be answerable unambiguously from the schema and notes below, and the gold SQL must be exactly right on this data."""


def brief(inv: str) -> str:
    return f"Schema:\n\n{ddl()}\n\n{NOTES}\n\nData inventory (real values in the snapshot):\n{inv}"


def batch_prompt(archetype: str, difficulty: str, style: str, n: int, avoid: list[str]) -> str:
    avoid_block = "\n".join(f"- {a}" for a in avoid[-40:]) or "- (none yet)"
    return f"""Write {n} new tasks.

Archetype: {archetype}
Difficulty: {difficulty}  (easy = one table, one or two filters; medium = grouping, derived columns, or a join; hard = join + aggregation + a trap or NULL handling, or nested logic)
Question style: {style}

Rules for the question:
- Written by someone at the store asking about its data, in the style above. The casual, formal and terse styles must NOT name tables or column identifiers (say "sales rep", not sales_rep_id); the analyst style may.
- Unambiguous: say exactly which rows count (e.g. "only delivered orders", "ignoring cancelled orders", "counting a failed payment attempt as no payment", "customers who signed up in 2024") and exactly what to return (which columns, in which order, or one number). Say "rounded to 2 decimals" when the gold rounds. When a question involves an order's total, say whether discount and shipping are included.
- Do not restate the notes or explain the schema inside the question. Keep it 1-3 sentences.
- Different from every question in the avoid list (different rows, columns or metric, not a rewording).

Rules for the gold SQL:
- PostgreSQL 16 dialect. One SELECT or WITH...SELECT. No SELECT *. Returns 1 to 50 rows, never an empty result and never only NULLs.
- Return only what the question asks for, in that order. One number -> one row, one column.
- Every ORDER BY ends with a unique tie-breaker (an id column) so the order is deterministic.
- Use ROUND(x::numeric, 2) when the question says rounded. Averages of integers should be cast or rounded as the question says.
- Order totals follow the notes: SUM(quantity * order_items.unit_price) * (1 - COALESCE(discount_pct, 0) / 100) + shipping_cost.
- Use the inventory above so filters actually match rows (real names, countries, statuses, dates, spellings).

Avoid (existing questions for this archetype):
{avoid_block}

Output: a JSON array only, no prose, each item {{"question": str, "sql": str}}."""


_JSON = re.compile(r"\[.*\]", re.S)


def parse(text: str) -> list[dict]:
    start = text.find("[")
    if start < 0:
        return []
    body = text[start:]
    end = body.rfind("]")
    items = None
    if end > 0:
        try:
            items = json.loads(body[: end + 1])
        except json.JSONDecodeError:
            items = None
    if items is None:
        # cut off mid-array: keep every complete object before the cut
        cut = body.rfind("}")
        while cut > 0:
            try:
                items = json.loads(body[: cut + 1] + "]")
                break
            except json.JSONDecodeError:
                cut = body.rfind("}", 0, cut)
        if items is None:
            return []
    return [i for i in items if isinstance(i, dict) and i.get("question") and i.get("sql")]


def norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def tokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", q.lower()))


def validate(item: dict, seen_sql: set[str], seen_q: list[set[str]]) -> tuple[bool, str]:
    sql = item["sql"].strip().rstrip(";")
    if re.search(r"\bselect\s+\*", sql, re.I):
        return False, "select *"
    if len(sql) > 4000:
        return False, "sql too long"
    try:
        rows = run_sql(sql, limit=51)
        rows2 = run_sql(sql, limit=51)
    except Exception as exc:
        return False, f"sql error: {str(exc).splitlines()[0][:100]}"
    if not rows:
        return False, "empty result"
    if len(rows) > 50:
        return False, "more than 50 rows"
    if all(v is None for r in rows for v in r):
        return False, "all null"
    if norm_rows(rows) != norm_rows(rows2):
        return False, "nondeterministic"
    if norm_sql(sql) in seen_sql:
        return False, "duplicate sql"
    qt = tokens(item["question"])
    for other in seen_q:
        inter = len(qt & other)
        if inter / max(1, len(qt | other)) > 0.75:
            return False, "near-duplicate question"
    if len(item["question"]) < 20 or len(item["question"]) > 700:
        return False, "question length"
    return True, ""


def ask(prompt: str, brief_text: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=9000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=[
            {"type": "text", "text": SYSTEM},
            {"type": "text", "text": brief_text, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--per-batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--only", default="", help="substring filter on archetype")
    args = ap.parse_args()

    inv = inventory()
    brief_text = brief(inv)
    tasks = load_tasks()
    seen_sql = {norm_sql(t["sql"]) for t in tasks}
    seen_q = [tokens(t["question"]) for t in tasks]
    by_arch: dict[str, list[str]] = {}
    for t in tasks:
        by_arch.setdefault(t["archetype"], []).append(t["question"])
    done = set(DONE.read_text(encoding="utf-8").split("\n")) if DONE.exists() else set()

    jobs = []
    for r in range(args.rounds):
        for ai, arch in enumerate(ARCHETYPES):
            if args.only and args.only not in arch:
                continue
            for di, diff in enumerate(DIFFICULTIES):
                style = STYLES[(ai + di + r) % len(STYLES)]
                key = f"{r}|{arch}|{diff}|{style}"
                if key in done:
                    continue
                jobs.append((key, arch, diff, style))
    print(f"{len(tasks)} tasks on disk; {len(jobs)} batches to run", flush=True)

    stats = {"kept": 0, "rejected": {}}
    t0 = time.time()

    def run(job):
        key, arch, diff, style = job
        with lock:
            avoid = list(by_arch.get(arch, []))
        text = ask(batch_prompt(arch, diff, style, args.per_batch, avoid), brief_text)
        items = parse(text)
        kept = []
        with lock:
            for it in items:
                ok, why = validate(it, seen_sql, seen_q)
                if not ok:
                    stats["rejected"][why.split(":")[0]] = (
                        stats["rejected"].get(why.split(":")[0], 0) + 1
                    )
                    continue
                task = {
                    "id": task_id(it["question"]),
                    "question": it["question"].strip(),
                    "sql": it["sql"].strip().rstrip(";"),
                    "archetype": arch,
                    "difficulty": diff,
                    "style": style,
                    "author": MODEL,
                }
                seen_sql.add(norm_sql(task["sql"]))
                seen_q.append(tokens(task["question"]))
                by_arch.setdefault(arch, []).append(task["question"])
                kept.append(task)
                stats["kept"] += 1
            with TASKS.open("a", encoding="utf-8") as f:
                for task in kept:
                    f.write(json.dumps(task) + "\n")
            with DONE.open("a", encoding="utf-8") as f:
                f.write(key + "\n")
        return key, len(items), len(kept)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                key, n, k = fut.result()
                print(
                    f"[{i}/{len(jobs)}] {k}/{n} kept  {key[:60]}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{i}/{len(jobs)}] ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    print(json.dumps(stats, indent=1))
    print(f"total tasks: {len(load_tasks())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
