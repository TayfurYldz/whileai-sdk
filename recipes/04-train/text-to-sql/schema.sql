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
