# Data dictionary

_Generated from `src/analytics_agent/schema_registry.py` by `scripts/build_docs.py`. Do not edit by hand._

## `customers`

**Purpose:** One row per registered customer account. Source of customer region and segment.

**Grain:** One row per customer_id.

**Primary key:** `customer_id`

**Other uniqueness constraints:** (email)

**Partitioning:** none &nbsp; **Clustering:** region, customer_segment

**Expected rows:** ~2,000 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `customer_id` | STRING | REQUIRED | Surrogate customer identifier, format CUS000001. |  | CUS000042 |
| `email` | STRING | REQUIRED | Synthetic contact email (always @example.com). Treated as PII: never send to the LLM or logs. **(PII)** |  | customer000042@example.com |
| `region` | STRING | REQUIRED | Customer home region; used for regional sales and tax. | `Northeast`, `Southeast`, `Midwest`, `Southwest`, `West` | Midwest |
| `country_code` | STRING | REQUIRED | ISO 3166-1 alpha-2 country code. | `US` | US |
| `customer_segment` | STRING | REQUIRED | Commercial segment. | `consumer`, `small_business`, `corporate` | consumer |
| `signup_date` | DATE | REQUIRED | Date the account was created (UTC). Must be on or before every order date for this customer. |  | 2024-03-17 |
| `created_at` | TIMESTAMP | REQUIRED | Record creation timestamp (UTC). |  | 2024-03-17 09:12:44 UTC |

**Validation rules:**

- customer_id unique and not null
- email unique, not null
- region and customer_segment in allowed values
- signup_date <= first order_date

## `products`

**Purpose:** Product catalog with list price and category.

**Grain:** One row per product_id.

**Primary key:** `product_id`

**Other uniqueness constraints:** (sku)

**Partitioning:** none &nbsp; **Clustering:** category

**Expected rows:** ~250 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `product_id` | STRING | REQUIRED | Surrogate product identifier, format PRD00001. |  | PRD00017 |
| `sku` | STRING | REQUIRED | Stock keeping unit; business key, unique across catalog. |  | SKU-ELE-00017 |
| `product_name` | STRING | REQUIRED | Display name. |  | Compact Wireless Speaker |
| `category` | STRING | REQUIRED | Merchandising category. | `Electronics`, `Home & Kitchen`, `Apparel`, `Beauty`, `Sports & Outdoors`, `Books`, `Toys` | Electronics |
| `list_price` | NUMERIC | REQUIRED | Current list price in USD (> 0). In this version prices do not change over time. |  | 129.99 |
| `unit_cost` | NUMERIC | REQUIRED | Unit cost in USD (> 0 and < list_price). |  | 71.50 |
| `is_active` | BOOL | REQUIRED | Whether the product is currently sellable. |  | true |
| `created_at` | TIMESTAMP | REQUIRED | Record creation timestamp (UTC). |  | 2024-01-05 00:00:00 UTC |

**Validation rules:**

- product_id unique; sku unique
- category in allowed values
- list_price > 0; 0 < unit_cost < list_price

## `orders`

**Purpose:** Order header: who ordered, when, lifecycle status and order-level money totals.

**Grain:** One row per order_id.

**Primary key:** `order_id`

**Foreign keys:** `customer_id` → `customers.customer_id`

**Partitioning:** DAY on order_date &nbsp; **Clustering:** order_status, customer_id

**Expected rows:** ~15,000 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `order_id` | STRING | REQUIRED | Surrogate order identifier, format ORD0000001. |  | ORD0001234 |
| `customer_id` | STRING | REQUIRED | FK to customers.customer_id. |  | CUS000042 |
| `order_timestamp` | TIMESTAMP | REQUIRED | When the order was placed (UTC). |  | 2026-02-14 18:03:11 UTC |
| `order_date` | DATE | REQUIRED | DATE(order_timestamp) in UTC. Partition column and the date used to attribute sales to a period. |  | 2026-02-14 |
| `order_status` | STRING | REQUIRED | Lifecycle status as of the data snapshot. Refund state is NOT stored here; see refunds. | `pending_payment`, `processing`, `shipped`, `delivered`, `cancelled` | delivered |
| `sales_channel` | STRING | REQUIRED | Channel the order was placed through. | `web`, `mobile_app`, `marketplace` | web |
| `currency_code` | STRING | REQUIRED | Currency of all money columns. | `USD` | USD |
| `items_subtotal` | NUMERIC | REQUIRED | SUM(order_items.gross_amount) for the order (before discounts). |  | 149.98 |
| `discount_amount` | NUMERIC | REQUIRED | SUM(order_items.discount_amount) for the order. |  | 15.00 |
| `shipping_amount` | NUMERIC | REQUIRED | Shipping charged to the customer (0.00 when merchandise net >= 75.00, else 6.99). |  | 0.00 |
| `tax_amount` | NUMERIC | REQUIRED | Sales tax = ROUND((items_subtotal - discount_amount) * regional rate, 2). |  | 9.11 |
| `order_total` | NUMERIC | REQUIRED | items_subtotal - discount_amount + shipping_amount + tax_amount. Amount the customer is asked to pay. |  | 144.09 |
| `cancelled_timestamp` | TIMESTAMP | NULLABLE | Set if and only if order_status = 'cancelled'. |  |  |

**Validation rules:**

- order_id unique
- customer_id exists in customers
- order_status in allowed values
- order_total = items_subtotal - discount_amount + shipping_amount + tax_amount (±0.01)
- items_subtotal = SUM(order_items.gross_amount) (±0.01)
- cancelled_timestamp not null iff status = cancelled
- order_timestamp <= snapshot as_of timestamp

> **Join fan-out warning:** Joining orders to order_items, payments, or refunds multiplies order rows. Never SUM order-level columns (order_total, shipping_amount, tax_amount) after such a join; aggregate the child table first, or use COUNT(DISTINCT order_id).

## `order_items`

**Purpose:** Order lines: one product and quantity per line with line-level money.

**Grain:** One row per order_item_id (one product line within an order).

**Primary key:** `order_item_id`

**Other uniqueness constraints:** (order_id, line_number); (order_id, product_id)

**Foreign keys:** `order_id` → `orders.order_id`; `product_id` → `products.product_id`

**Partitioning:** none &nbsp; **Clustering:** order_id, product_id

**Expected rows:** ~33,000 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `order_item_id` | STRING | REQUIRED | Surrogate order line identifier, format ITM00000001. |  | ITM00003391 |
| `order_id` | STRING | REQUIRED | FK to orders.order_id. |  | ORD0001234 |
| `line_number` | INT64 | REQUIRED | 1-based line number within the order; (order_id, line_number) is unique. |  | 2 |
| `product_id` | STRING | REQUIRED | FK to products.product_id. A product appears at most once per order. |  | PRD00017 |
| `quantity` | INT64 | REQUIRED | Units ordered, integer between 1 and 10. |  | 1 |
| `unit_price` | NUMERIC | REQUIRED | Price per unit charged on this order (equals product list_price in this version). |  | 129.99 |
| `gross_amount` | NUMERIC | REQUIRED | quantity * unit_price. |  | 129.99 |
| `discount_amount` | NUMERIC | REQUIRED | Line discount, 0 <= discount_amount < gross_amount. |  | 13.00 |
| `net_amount` | NUMERIC | REQUIRED | gross_amount - discount_amount. Merchandise value after discount, excludes tax and shipping. |  | 116.99 |

**Validation rules:**

- order_item_id unique; (order_id, line_number) unique
- order_id and product_id exist in parents
- quantity between 1 and 10
- gross_amount = quantity * unit_price; net_amount = gross_amount - discount_amount

## `payments`

**Purpose:** Payment attempts against orders, including failed and pending attempts.

**Grain:** One row per payment attempt (payment_id). An order can have several attempts.

**Primary key:** `payment_id`

**Other uniqueness constraints:** (provider_transaction_id)

**Foreign keys:** `order_id` → `orders.order_id`

**Partitioning:** none (small table; DATE(payment_timestamp) is the candidate at scale) &nbsp; **Clustering:** order_id, payment_status

**Expected rows:** ~15,500 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `payment_id` | STRING | REQUIRED | Surrogate payment attempt identifier, format PAY0000001. |  | PAY0001301 |
| `order_id` | STRING | REQUIRED | FK to orders.order_id. |  | ORD0001234 |
| `provider_transaction_id` | STRING | REQUIRED | Payment provider's transaction reference. Business key: must be unique; duplicates indicate double-recorded payments. |  | txn_9f1c2a7be04d3311 |
| `payment_timestamp` | TIMESTAMP | REQUIRED | When the attempt was processed (UTC); must be >= order_timestamp. |  | 2026-02-14 18:04:02 UTC |
| `payment_method` | STRING | REQUIRED | Instrument used. | `credit_card`, `debit_card`, `paypal`, `gift_card` | credit_card |
| `payment_status` | STRING | REQUIRED | captured = money taken; failed = attempt declined; pending = awaiting confirmation. | `captured`, `failed`, `pending` | captured |
| `amount` | NUMERIC | REQUIRED | Attempted amount in USD (> 0). For captured payments equals orders.order_total. |  | 144.09 |
| `failure_reason` | STRING | NULLABLE | Set if and only if payment_status = 'failed'. | `insufficient_funds`, `card_declined`, `fraud_suspected`, `expired_card` |  |

**Validation rules:**

- payment_id unique; provider_transaction_id unique
- order_id exists in orders
- payment_timestamp >= order_timestamp
- booked orders have exactly one captured payment equal to order_total
- failure_reason not null iff status = failed

> **Join fan-out warning:** Orders can have a failed attempt before the captured one. Filter payment_status = 'captured' and aggregate before joining to orders.

## `shipments`

**Purpose:** Physical shipment of an order and its delivery outcome.

**Grain:** One row per shipment_id. In this version an order has at most one shipment; models must not assume exactly one.

**Primary key:** `shipment_id`

**Other uniqueness constraints:** (order_id)

**Foreign keys:** `order_id` → `orders.order_id`

**Partitioning:** none &nbsp; **Clustering:** carrier, shipment_status

**Expected rows:** ~13,800 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `shipment_id` | STRING | REQUIRED | Surrogate shipment identifier, format SHP0000001. |  | SHP0000877 |
| `order_id` | STRING | REQUIRED | FK to orders.order_id. |  | ORD0001234 |
| `carrier` | STRING | REQUIRED | Carrier (fictional). | `swiftship`, `parcelpro`, `metroexpress` | parcelpro |
| `shipment_status` | STRING | REQUIRED | in_transit or delivered as of the snapshot. | `in_transit`, `delivered` | delivered |
| `shipped_timestamp` | TIMESTAMP | REQUIRED | When the parcel left the warehouse (UTC); >= order_timestamp. |  | 2026-02-15 10:20:00 UTC |
| `promised_delivery_date` | DATE | REQUIRED | Delivery date promised at checkout = order_date + 5 days. |  | 2026-02-19 |
| `delivered_timestamp` | TIMESTAMP | NULLABLE | Set if and only if shipment_status = 'delivered'; must be >= shipped_timestamp. |  | 2026-02-18 14:41:09 UTC |

**Validation rules:**

- shipment_id unique; at most one shipment per order (this version)
- delivered_timestamp not null iff status = delivered
- delivered_timestamp >= shipped_timestamp
- shipment status consistent with order status

## `refunds`

**Purpose:** Refund transactions: item-level returns on delivered orders and full refunds of cancelled-but-paid orders.

**Grain:** One row per refund transaction (refund_id).

**Primary key:** `refund_id`

**Foreign keys:** `order_id` → `orders.order_id`; `order_item_id` → `order_items.order_item_id` (nullable); `payment_id` → `payments.payment_id`

**Partitioning:** none &nbsp; **Clustering:** order_id, refund_status

**Expected rows:** ~1,700 (clean)

| Column | Type | Mode | Description | Allowed values | Example |
|---|---|---|---|---|---|
| `refund_id` | STRING | REQUIRED | Surrogate refund identifier, format REF000001. |  | REF000118 |
| `order_id` | STRING | REQUIRED | FK to orders.order_id. |  | ORD0001234 |
| `order_item_id` | STRING | NULLABLE | FK to order_items. NULL means an order-level refund (used for cancellations, refunding the full captured amount). |  | ITM00003391 |
| `payment_id` | STRING | REQUIRED | FK to the captured payments.payment_id being refunded. |  | PAY0001301 |
| `refund_timestamp` | TIMESTAMP | REQUIRED | When the refund was initiated (UTC). |  | 2026-02-27 08:00:12 UTC |
| `refund_amount` | NUMERIC | REQUIRED | Refunded amount in USD (> 0). Item refunds cover merchandise only (no tax/shipping) and may be partial. |  | 58.50 |
| `refund_reason` | STRING | REQUIRED | Reason code. | `damaged`, `wrong_item`, `not_as_described`, `late_delivery`, `changed_mind`, `order_cancelled` | damaged |
| `refund_status` | STRING | REQUIRED | completed = money returned; pending = initiated, not yet settled. Only completed refunds reduce net sales. | `completed`, `pending` | completed |

**Validation rules:**

- refund_id unique; FKs exist
- SUM(refund_amount) per order_item <= order_items.net_amount
- order-level refund <= captured amount
- order_item_id is NULL only for refund_reason = 'order_cancelled'

