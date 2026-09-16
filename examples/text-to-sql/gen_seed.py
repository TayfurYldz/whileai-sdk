"""Generate schema.sql + seed.sql for a small online-store database (Postgres).

Deterministic (seed 42). Eight tables, a few hundred rows, and the usual
real-world traps: NULLs that mean something (unshipped orders, no sales rep,
no discount), cancelled orders without payments, failed payments, discontinued
products, self-referencing keys (category parent, employee manager, customer
referrer), customers with no orders and products with no reviews.

Usage: python gen_seed.py   -> writes schema.sql and seed.sql next to this file
Load:  psql -h 127.0.0.1 -p 5499 -U postgres -d shop -f schema.sql -f seed.sql
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = random.Random(42)

SCHEMA = """\
-- Online store: customers place orders for products; orders are paid, shipped, reviewed.
DROP TABLE IF EXISTS reviews, payments, order_items, orders, employees, customers, products, categories CASCADE;

CREATE TABLE categories (
  category_id   INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,
  parent_id     INTEGER REFERENCES categories(category_id)   -- NULL for a top-level category
);

CREATE TABLE products (
  product_id    INTEGER PRIMARY KEY,
  sku           TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  category_id   INTEGER NOT NULL REFERENCES categories(category_id),
  unit_price    NUMERIC(10,2) NOT NULL,      -- current list price
  unit_cost     NUMERIC(10,2) NOT NULL,      -- what the store pays
  stock_qty     INTEGER NOT NULL,
  discontinued  BOOLEAN NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMP NOT NULL
);

CREATE TABLE customers (
  customer_id       INTEGER PRIMARY KEY,
  first_name        TEXT NOT NULL,
  last_name         TEXT NOT NULL,
  email             TEXT NOT NULL UNIQUE,
  country           TEXT NOT NULL,           -- ISO-2 code: US, CA, GB, DE, FR, AU
  city              TEXT NOT NULL,
  signup_date       DATE NOT NULL,
  marketing_opt_in  BOOLEAN NOT NULL,
  referred_by       INTEGER REFERENCES customers(customer_id)   -- NULL when not referred
);

CREATE TABLE employees (
  employee_id   INTEGER PRIMARY KEY,
  full_name     TEXT NOT NULL,
  department    TEXT NOT NULL,               -- Sales, Support, Warehouse, Management
  hire_date     DATE NOT NULL,
  salary        NUMERIC(10,2) NOT NULL,
  manager_id    INTEGER REFERENCES employees(employee_id)     -- NULL for the head of the company
);

CREATE TABLE orders (
  order_id          INTEGER PRIMARY KEY,
  customer_id       INTEGER NOT NULL REFERENCES customers(customer_id),
  sales_rep_id      INTEGER REFERENCES employees(employee_id),  -- NULL for self-service web orders
  order_date        TIMESTAMP NOT NULL,
  status            TEXT NOT NULL,           -- pending, paid, shipped, delivered, cancelled, refunded
  shipped_at        TIMESTAMP,               -- NULL until the order ships
  shipping_country  TEXT NOT NULL,
  shipping_cost     NUMERIC(10,2) NOT NULL,
  discount_pct      NUMERIC(5,2)             -- percent off the item subtotal; NULL means no discount
);

CREATE TABLE order_items (
  order_id      INTEGER NOT NULL REFERENCES orders(order_id),
  product_id    INTEGER NOT NULL REFERENCES products(product_id),
  quantity      INTEGER NOT NULL,
  unit_price    NUMERIC(10,2) NOT NULL,      -- price charged at order time (may differ from products.unit_price)
  PRIMARY KEY (order_id, product_id)
);

CREATE TABLE payments (
  payment_id    INTEGER PRIMARY KEY,
  order_id      INTEGER NOT NULL REFERENCES orders(order_id),
  amount        NUMERIC(10,2) NOT NULL,
  method        TEXT NOT NULL,               -- card, paypal, bank_transfer, gift_card
  status        TEXT NOT NULL,               -- captured, failed, refunded
  paid_at       TIMESTAMP NOT NULL
);

CREATE TABLE reviews (
  review_id     INTEGER PRIMARY KEY,
  product_id    INTEGER NOT NULL REFERENCES products(product_id),
  customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
  rating        INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  title         TEXT,                        -- NULL when the reviewer left only a rating
  created_at    TIMESTAMP NOT NULL
);
"""

CATEGORIES = [
    (1, "Electronics", None),
    (2, "Home & Kitchen", None),
    (3, "Sports & Outdoors", None),
    (4, "Audio", 1),
    (5, "Computers", 1),
    (6, "Cookware", 2),
    (7, "Furniture", 2),
    (8, "Cycling", 3),
    (9, "Camping", 3),
]

PRODUCT_NAMES = {
    4: [
        "Noise-Cancelling Headphones",
        "Bluetooth Speaker",
        "Studio Microphone",
        "Wireless Earbuds",
        "Turntable",
    ],
    5: [
        "Mechanical Keyboard",
        "27in 4K Monitor",
        "USB-C Docking Station",
        "Wireless Mouse",
        "Laptop Stand",
        "1TB NVMe SSD",
        "Webcam 1080p",
    ],
    6: [
        "Cast Iron Skillet",
        "Chef Knife 8in",
        "Stainless Stock Pot",
        "Nonstick Fry Pan",
        "Dutch Oven 6qt",
        "Bamboo Cutting Board",
    ],
    7: [
        "Standing Desk",
        "Ergonomic Office Chair",
        "Bookshelf 5-Tier",
        "Bedside Table",
        "Floor Lamp",
    ],
    8: [
        "Road Bike Helmet",
        "Bike Lock U-Style",
        "Cycling Gloves",
        "Bike Pump Floor",
        "LED Bike Light Set",
        "Water Bottle Cage",
    ],
    9: [
        "2-Person Tent",
        "Sleeping Bag 20F",
        "Camp Stove",
        "Headlamp 400lm",
        "Trekking Poles",
        "Insulated Water Bottle",
    ],
}
PRICE_RANGE = {4: (39, 349), 5: (19, 599), 6: (14, 129), 7: (49, 499), 8: (9, 89), 9: (15, 249)}

FIRST = [
    "Olivia",
    "Liam",
    "Emma",
    "Noah",
    "Ava",
    "Oliver",
    "Sophia",
    "Elijah",
    "Isabella",
    "Lucas",
    "Mia",
    "Mason",
    "Amelia",
    "Logan",
    "Harper",
    "Ethan",
    "Evelyn",
    "James",
    "Abigail",
    "Aiden",
    "Ella",
    "Jackson",
    "Scarlett",
    "Sebastian",
    "Grace",
    "Mateo",
    "Chloe",
    "Jack",
    "Zoe",
    "Owen",
]
LAST = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Wilson",
    "Anderson",
    "Taylor",
    "Thomas",
    "Moore",
    "Martin",
    "Lee",
    "Walker",
    "Hall",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Green",
    "Baker",
    "Adams",
    "Nelson",
    "Hill",
    "Campbell",
    "Mitchell",
]
CITIES = {
    "US": ["New York", "Austin", "Denver", "Seattle", "Chicago"],
    "CA": ["Toronto", "Vancouver", "Montreal"],
    "GB": ["London", "Manchester", "Bristol"],
    "DE": ["Berlin", "Munich", "Hamburg"],
    "FR": ["Paris", "Lyon"],
    "AU": ["Sydney", "Melbourne"],
}
COUNTRY_W = [("US", 40), ("CA", 15), ("GB", 15), ("DE", 12), ("FR", 8), ("AU", 10)]

EMPLOYEES = [
    (1, "Dana Whitfield", "Management", date(2018, 3, 1), 145000, None),
    (2, "Marcus Chen", "Sales", date(2019, 6, 17), 82000, 1),
    (3, "Priya Natarajan", "Sales", date(2020, 1, 13), 78000, 2),
    (4, "Tomás Alvarez", "Sales", date(2021, 9, 6), 71000, 2),
    (5, "Hannah Kowalski", "Support", date(2019, 11, 4), 58000, 1),
    (6, "Jamal Okafor", "Support", date(2022, 4, 25), 52000, 5),
    (7, "Ingrid Solberg", "Warehouse", date(2020, 8, 10), 61000, 1),
    (8, "Kenji Watanabe", "Warehouse", date(2023, 2, 20), 47000, 7),
    (9, "Leah Goldberg", "Sales", date(2023, 10, 2), 64000, 2),
]
SALES_REPS = [2, 3, 4, 9]


def q(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, datetime):
        return "'" + v.strftime("%Y-%m-%d %H:%M:%S") + "'"
    if isinstance(v, date):
        return "'" + v.isoformat() + "'"
    return "'" + str(v).replace("'", "''") + "'"


def insert(table: str, cols: list[str], rows: list[tuple]) -> str:
    out = [f"INSERT INTO {table} ({', '.join(cols)}) VALUES"]
    out.append(",\n".join("  (" + ", ".join(q(v) for v in r) + ")" for r in rows) + ";")
    return "\n".join(out)


def pick_country() -> str:
    return R.choices([c for c, _ in COUNTRY_W], weights=[w for _, w in COUNTRY_W])[0]


def main() -> int:
    (HERE / "schema.sql").write_text(SCHEMA, encoding="utf-8")
    parts = []

    # products
    products = []
    pid = 1
    for cat, names in PRODUCT_NAMES.items():
        lo, hi = PRICE_RANGE[cat]
        for n in names:
            price = round(R.uniform(lo, hi) - 0.01, 2)
            cost = round(price * R.uniform(0.45, 0.75), 2)
            created = datetime(2022, 1, 1) + timedelta(
                days=R.randint(0, 900), hours=R.randint(8, 18)
            )
            discontinued = R.random() < 0.12
            stock = 0 if discontinued and R.random() < 0.6 else R.randint(0, 240)
            products.append(
                (pid, f"SKU-{cat:02d}{pid:03d}", n, cat, price, cost, stock, discontinued, created)
            )
            pid += 1

    # customers
    customers = []
    emails = set()
    for cid in range(1, 61):
        f, last = R.choice(FIRST), R.choice(LAST)
        email = f"{f}.{last}{cid}@example.com".lower()
        assert email not in emails
        emails.add(email)
        country = pick_country()
        city = R.choice(CITIES[country])
        signup = date(2022, 1, 1) + timedelta(days=R.randint(0, 1000))
        referred = R.randint(1, cid - 1) if cid > 5 and R.random() < 0.25 else None
        customers.append((cid, f, last, email, country, city, signup, R.random() < 0.55, referred))
    signup_of = {c[0]: c[6] for c in customers}
    country_of = {c[0]: c[4] for c in customers}

    # orders: ~300; ~8 customers never order
    silent = set(R.sample(range(1, 61), 8))
    buyers = [c for c in range(1, 61) if c not in silent]
    orders, items, payments = [], [], []
    oid, payid = 1, 1
    price_of = {p[0]: p[4] for p in products}
    for _ in range(300):
        cid = R.choice(buyers)
        earliest = max(signup_of[cid], date(2023, 1, 1))
        odate = datetime.combine(earliest, datetime.min.time()) + timedelta(
            days=R.randint(1, (date(2025, 6, 30) - earliest).days),
            hours=R.randint(6, 22),
            minutes=R.randint(0, 59),
        )
        status = R.choices(
            ["pending", "paid", "shipped", "delivered", "cancelled", "refunded"],
            weights=[6, 10, 14, 55, 9, 6],
        )[0]
        rep = R.choice(SALES_REPS) if R.random() < 0.35 else None
        shipped = (
            odate + timedelta(days=R.randint(1, 6), hours=R.randint(0, 12))
            if status in ("shipped", "delivered", "refunded")
            else None
        )
        ship_country = country_of[cid] if R.random() < 0.9 else pick_country()
        ship_cost = R.choice([0, 4.99, 7.99, 12.5, 19.99])
        discount = R.choice([5, 10, 15, 20, 25]) if R.random() < 0.3 else None
        orders.append((oid, cid, rep, odate, status, shipped, ship_country, ship_cost, discount))
        chosen = R.sample(products, R.choices([1, 2, 3, 4, 5], weights=[35, 30, 20, 10, 5])[0])
        subtotal = 0.0
        for p in chosen:
            qty = R.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]
            unit = (
                price_of[p[0]]
                if R.random() < 0.8
                else round(price_of[p[0]] * R.uniform(0.85, 1.1), 2)
            )
            items.append((oid, p[0], qty, unit))
            subtotal += qty * unit
        total = round(subtotal * (1 - (discount or 0) / 100) + ship_cost, 2)
        if status != "pending" and status != "cancelled":
            if R.random() < 0.15:  # a failed attempt before the real one
                payments.append(
                    (
                        payid,
                        oid,
                        total,
                        R.choice(["card", "paypal"]),
                        "failed",
                        odate + timedelta(minutes=R.randint(1, 30)),
                    )
                )
                payid += 1
            method = R.choices(
                ["card", "paypal", "bank_transfer", "gift_card"], weights=[60, 25, 10, 5]
            )[0]
            payments.append(
                (
                    payid,
                    oid,
                    total,
                    method,
                    "refunded" if status == "refunded" else "captured",
                    odate + timedelta(minutes=R.randint(2, 120)),
                )
            )
            payid += 1
        elif status == "cancelled" and R.random() < 0.3:
            payments.append(
                (payid, oid, total, "card", "failed", odate + timedelta(minutes=R.randint(1, 30)))
            )
            payid += 1
        oid += 1

    # reviews: 120, only from customers who ordered the product (delivered), some products never reviewed
    delivered = {}
    for o in orders:
        if o[4] in ("delivered", "refunded"):
            for it in items:
                if it[0] == o[0]:
                    delivered.setdefault(it[1], set()).add((o[1], o[3]))
    unreviewed = set(R.sample(sorted(delivered), 6))
    reviews = []
    rid = 1
    seen = set()
    # sorted: set iteration order is not stable across processes
    pool = [
        (p, c, d) for p, cs in delivered.items() if p not in unreviewed for (c, d) in sorted(cs)
    ]
    R.shuffle(pool)
    titles = [
        "Great value",
        "Not what I expected",
        "Solid build",
        "Arrived late but works",
        "Would buy again",
        "Stopped working",
        "Perfect",
        "Okay for the price",
        "Excellent quality",
        "Disappointed",
    ]
    for p, c, d in pool:
        if len(reviews) >= 120:
            break
        if (p, c) in seen:
            continue
        seen.add((p, c))
        rating = R.choices([1, 2, 3, 4, 5], weights=[6, 8, 16, 35, 35])[0]
        title = R.choice(titles) if R.random() < 0.7 else None
        reviews.append(
            (rid, p, c, rating, title, d + timedelta(days=R.randint(3, 40), hours=R.randint(0, 23)))
        )
        rid += 1

    parts.append(insert("categories", ["category_id", "name", "parent_id"], CATEGORIES))
    parts.append(
        insert(
            "products",
            [
                "product_id",
                "sku",
                "name",
                "category_id",
                "unit_price",
                "unit_cost",
                "stock_qty",
                "discontinued",
                "created_at",
            ],
            products,
        )
    )
    parts.append(
        insert(
            "customers",
            [
                "customer_id",
                "first_name",
                "last_name",
                "email",
                "country",
                "city",
                "signup_date",
                "marketing_opt_in",
                "referred_by",
            ],
            customers,
        )
    )
    parts.append(
        insert(
            "employees",
            ["employee_id", "full_name", "department", "hire_date", "salary", "manager_id"],
            EMPLOYEES,
        )
    )
    parts.append(
        insert(
            "orders",
            [
                "order_id",
                "customer_id",
                "sales_rep_id",
                "order_date",
                "status",
                "shipped_at",
                "shipping_country",
                "shipping_cost",
                "discount_pct",
            ],
            orders,
        )
    )
    parts.append(insert("order_items", ["order_id", "product_id", "quantity", "unit_price"], items))
    parts.append(
        insert(
            "payments",
            ["payment_id", "order_id", "amount", "method", "status", "paid_at"],
            payments,
        )
    )
    parts.append(
        insert(
            "reviews",
            ["review_id", "product_id", "customer_id", "rating", "title", "created_at"],
            reviews,
        )
    )
    (HERE / "seed.sql").write_text(
        "BEGIN;\n" + "\n\n".join(parts) + "\nCOMMIT;\n", encoding="utf-8"
    )
    print(
        f"products {len(products)} customers {len(customers)} orders {len(orders)} items {len(items)} payments {len(payments)} reviews {len(reviews)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
