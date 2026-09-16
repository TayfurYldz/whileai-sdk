"""The policy's system prompt: the schema (schema.sql) plus the notes that say
what the data means. No dependencies, so the Modal launcher can build it.

Bring your own schema: replace schema.sql and seed.sql, rewrite NOTES for
your data, run `python schema_prompt.py` to refresh prompt.txt, then
`python author.py` to write tasks for it.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_SQL = HERE / "schema.sql"
PROMPT_TXT = HERE / "prompt.txt"

NOTES = """\
Notes (PostgreSQL 16):
- An order's item subtotal is SUM(order_items.quantity * order_items.unit_price) over its items (use the order_items price, not products.unit_price). Order total = subtotal * (1 - COALESCE(orders.discount_pct, 0) / 100) + orders.shipping_cost.
- orders.status is one of pending, paid, shipped, delivered, cancelled, refunded. Cancelled and pending orders have no captured payment. A refunded order's payment row has status 'refunded'. Some orders have an extra payment row with status 'failed' before the successful one.
- shipped_at is NULL until an order ships; sales_rep_id is NULL for self-service web orders; discount_pct is NULL when no discount applied; customers.referred_by is NULL when not referred; reviews.title is NULL for rating-only reviews; categories.parent_id is NULL for top-level categories; employees.manager_id is NULL for the head of the company.
- Text comparisons are case-sensitive; country codes are upper-case ISO-2 (US, CA, GB, DE, FR, AU). Use ROUND(x::numeric, 2) for rounding, DATE_TRUNC / EXTRACT for calendar buckets, and shipped_at - order_date for durations."""

RULES = """\
Rules:
- Reply with exactly one SELECT (or WITH ... SELECT) query inside a ```sql fenced block and nothing else. No explanation.
- Return only the columns the question asks for, in the order asked. A question that asks for one number gets one row with one column.
- Round only when the question asks for rounding."""


def ddl() -> str:
    text = SCHEMA_SQL.read_text(encoding="utf-8")
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("DROP TABLE")).strip()


def system_prompt() -> str:
    return (
        "You translate questions about an online store's database into ONE PostgreSQL query.\n\n"
        "Schema:\n\n"
        f"{ddl()}\n\n"
        f"{NOTES}\n\n"
        f"{RULES}"
    )


if __name__ == "__main__":
    PROMPT_TXT.write_text(system_prompt(), encoding="utf-8", newline="\n")
    print(f"wrote {PROMPT_TXT} ({len(system_prompt())} chars)")
