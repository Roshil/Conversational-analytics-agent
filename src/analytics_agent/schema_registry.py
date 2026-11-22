"""Schema registry: the single source of truth for the Analytics Agent data model.

Everything that needs to know about tables reads from here:
  * the synthetic data generator (column order, types),
  * the Parquet/CSV exporters (Arrow types),
  * the BigQuery schema JSON files used for explicit-schema loads,
  * the rendered data dictionary (docs/data_dictionary.md),
  * later: the SQL validator allowlist and the agent's schema context.

Keeping one definition prevents the classic drift where the docs say one thing,
the warehouse another and the LLM prompt a third.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Controlled vocabularies (allowed values). Referenced by generator and checks.
# ---------------------------------------------------------------------------
REGIONS = ("Northeast", "Southeast", "Midwest", "Southwest", "West")
SEGMENTS = ("consumer", "small_business", "corporate")
CATEGORIES = (
    "Electronics",
    "Home & Kitchen",
    "Apparel",
    "Beauty",
    "Sports & Outdoors",
    "Books",
    "Toys",
)
ORDER_STATUSES = (
    "pending_payment",
    "processing",
    "shipped",
    "delivered",
    "cancelled",
)
# Orders whose payment has been captured and that are not cancelled. These are
# the only orders that count toward sales metrics.
BOOKED_ORDER_STATUSES = ("processing", "shipped", "delivered")
SALES_CHANNELS = ("web", "mobile_app", "marketplace")
PAYMENT_METHODS = ("credit_card", "debit_card", "paypal", "gift_card")
PAYMENT_STATUSES = ("captured", "failed", "pending")
FAILURE_REASONS = (
    "insufficient_funds",
    "card_declined",
    "fraud_suspected",
    "expired_card",
)
CARRIERS = ("swiftship", "parcelpro", "metroexpress")
SHIPMENT_STATUSES = ("in_transit", "delivered")
REFUND_REASONS = (
    "damaged",
    "wrong_item",
    "not_as_described",
    "late_delivery",
    "changed_mind",
    "order_cancelled",
)
REFUND_STATUSES = ("completed", "pending")
CURRENCIES = ("USD",)
COUNTRIES = ("US",)


@dataclass(frozen=True)
class Column:
    """One column in a registered table.

    Attributes:
        name: Column name as it appears in BigQuery / Parquet.
        bq_type: Canonical BigQuery type (``STRING``, ``INT64``, ``NUMERIC``,
            ``BOOL``, ``DATE``, ``TIMESTAMP``).
        mode: ``REQUIRED`` or ``NULLABLE``.
        description: Human-readable description used in the data dictionary
            and LLM schema prompts.
        allowed_values: Optional controlled vocabulary for the column.
        pii: When ``True`` the column holds personal data and must not be
            sent to the LLM or logs.
        example: Illustrative value shown in the data dictionary.
    """

    name: str
    bq_type: str  # STRING, INT64, NUMERIC, BOOL, DATE, TIMESTAMP
    mode: str  # REQUIRED or NULLABLE
    description: str
    allowed_values: Optional[tuple] = None
    pii: bool = False
    example: str = ""

    def to_bigquery_json(self) -> dict:
        """Serialise this column to the BigQuery JSON schema format.

        Maps canonical type names to the legacy names BigQuery's JSON schema
        still expects (``INT64`` → ``INTEGER``, ``BOOL`` → ``BOOLEAN``).
        """
        type_map = {"INT64": "INTEGER", "BOOL": "BOOLEAN", "FLOAT64": "FLOAT"}
        return {
            "name": self.name,
            "type": type_map.get(self.bq_type, self.bq_type),
            "mode": self.mode,
            "description": self.description[:1024],
        }


@dataclass(frozen=True)
class ForeignKey:
    """A foreign-key relationship from one column to a parent table.

    Attributes:
        column: Local column name.
        ref_table: Parent table name.
        ref_column: Parent column name.
        nullable: When ``True`` the local column may be NULL (optional FK).
        note: Optional free-text clarification.
    """

    column: str
    ref_table: str
    ref_column: str
    nullable: bool = False
    note: str = ""


@dataclass(frozen=True)
class Table:
    """One registered analytics table with its full schema metadata.

    Attributes:
        name: Table name (also the key in ``TABLES``).
        purpose: Short business description of the table.
        grain: Natural grain (e.g. ``"One row per order_id."``).
        primary_key: Column names that form the primary key.
        columns: Ordered tuple of ``Column`` definitions.
        business_keys: Additional uniqueness constraints (each an ordered
            tuple of column names).
        foreign_keys: Tuple of ``ForeignKey`` relationships.
        partitioning: Partitioning scheme description (or ``"none"``).
        clustering: Clustering column names.
        expected_rows: Human-readable expected row-count note.
        validation_rules: Documented integrity rules for this table.
        fanout_warning: Optional warning about join fan-out hazards.
    """

    name: str
    purpose: str
    grain: str
    primary_key: tuple
    columns: tuple
    business_keys: tuple = ()  # other uniqueness constraints
    foreign_keys: tuple = ()
    partitioning: str = "none"
    clustering: tuple = ()
    expected_rows: str = ""
    validation_rules: tuple = ()
    fanout_warning: str = ""

    @property
    def column_names(self) -> list[str]:
        """Return the ordered list of column names."""
        return [c.name for c in self.columns]

    @property
    def required_columns(self) -> list[str]:
        """Return the names of columns whose mode is ``REQUIRED``."""
        return [c.name for c in self.columns if c.mode == "REQUIRED"]

    def column(self, name: str) -> Column:
        """Look up a column by name.

        Args:
            name: Column name.

        Returns:
            The matching ``Column``.

        Raises:
            KeyError: If *name* is not a column of this table.
        """
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}.{name} not in schema")

    def to_bigquery_json(self) -> list[dict]:
        """Serialise every column to the BigQuery JSON schema format."""
        return [c.to_bigquery_json() for c in self.columns]


R, N = "REQUIRED", "NULLABLE"

CUSTOMERS = Table(
    name="customers",
    purpose="One row per registered customer account. Source of customer region and segment.",
    grain="One row per customer_id.",
    primary_key=("customer_id",),
    columns=(
        Column(
            "customer_id",
            "STRING",
            R,
            "Surrogate customer identifier, format CUS000001.",
            example="CUS000042",
        ),
        Column(
            "email",
            "STRING",
            R,
            "Synthetic contact email (always @example.com). Treated as PII: never send to the LLM or logs.",
            pii=True,
            example="customer000042@example.com",
        ),
        Column(
            "region",
            "STRING",
            R,
            "Customer home region; used for regional sales and tax.",
            REGIONS,
            example="Midwest",
        ),
        Column(
            "country_code",
            "STRING",
            R,
            "ISO 3166-1 alpha-2 country code.",
            COUNTRIES,
            example="US",
        ),
        Column(
            "customer_segment",
            "STRING",
            R,
            "Commercial segment.",
            SEGMENTS,
            example="consumer",
        ),
        Column(
            "signup_date",
            "DATE",
            R,
            "Date the account was created (UTC). Must be on or before every order date for this customer.",
            example="2024-03-17",
        ),
        Column(
            "created_at",
            "TIMESTAMP",
            R,
            "Record creation timestamp (UTC).",
            example="2024-03-17 09:12:44 UTC",
        ),
    ),
    business_keys=(("email",),),
    clustering=("region", "customer_segment"),
    expected_rows="~2,000 (clean)",
    validation_rules=(
        "customer_id unique and not null",
        "email unique, not null",
        "region and customer_segment in allowed values",
        "signup_date <= first order_date",
    ),
)

PRODUCTS = Table(
    name="products",
    purpose="Product catalog with list price and category.",
    grain="One row per product_id.",
    primary_key=("product_id",),
    columns=(
        Column(
            "product_id",
            "STRING",
            R,
            "Surrogate product identifier, format PRD00001.",
            example="PRD00017",
        ),
        Column(
            "sku",
            "STRING",
            R,
            "Stock keeping unit; business key, unique across catalog.",
            example="SKU-ELE-00017",
        ),
        Column(
            "product_name",
            "STRING",
            R,
            "Display name.",
            example="Compact Wireless Speaker",
        ),
        Column(
            "category",
            "STRING",
            R,
            "Merchandising category.",
            CATEGORIES,
            example="Electronics",
        ),
        Column(
            "list_price",
            "NUMERIC",
            R,
            "Current list price in USD (> 0). In this version prices do not change over time.",
            example="129.99",
        ),
        Column(
            "unit_cost",
            "NUMERIC",
            R,
            "Unit cost in USD (> 0 and < list_price).",
            example="71.50",
        ),
        Column(
            "is_active",
            "BOOL",
            R,
            "Whether the product is currently sellable.",
            example="true",
        ),
        Column(
            "created_at",
            "TIMESTAMP",
            R,
            "Record creation timestamp (UTC).",
            example="2024-01-05 00:00:00 UTC",
        ),
    ),
    business_keys=(("sku",),),
    clustering=("category",),
    expected_rows="~250 (clean)",
    validation_rules=(
        "product_id unique; sku unique",
        "category in allowed values",
        "list_price > 0; 0 < unit_cost < list_price",
    ),
)

ORDERS = Table(
    name="orders",
    purpose="Order header: who ordered, when, lifecycle status and order-level money totals.",
    grain="One row per order_id.",
    primary_key=("order_id",),
    columns=(
        Column(
            "order_id",
            "STRING",
            R,
            "Surrogate order identifier, format ORD0000001.",
            example="ORD0001234",
        ),
        Column(
            "customer_id",
            "STRING",
            R,
            "FK to customers.customer_id.",
            example="CUS000042",
        ),
        Column(
            "order_timestamp",
            "TIMESTAMP",
            R,
            "When the order was placed (UTC).",
            example="2026-02-14 18:03:11 UTC",
        ),
        Column(
            "order_date",
            "DATE",
            R,
            "DATE(order_timestamp) in UTC. Partition column and the date used to attribute sales to a period.",
            example="2026-02-14",
        ),
        Column(
            "order_status",
            "STRING",
            R,
            "Lifecycle status as of the data snapshot. Refund state is NOT stored here; see refunds.",
            ORDER_STATUSES,
            example="delivered",
        ),
        Column(
            "sales_channel",
            "STRING",
            R,
            "Channel the order was placed through.",
            SALES_CHANNELS,
            example="web",
        ),
        Column(
            "currency_code",
            "STRING",
            R,
            "Currency of all money columns.",
            CURRENCIES,
            example="USD",
        ),
        Column(
            "items_subtotal",
            "NUMERIC",
            R,
            "SUM(order_items.gross_amount) for the order (before discounts).",
            example="149.98",
        ),
        Column(
            "discount_amount",
            "NUMERIC",
            R,
            "SUM(order_items.discount_amount) for the order.",
            example="15.00",
        ),
        Column(
            "shipping_amount",
            "NUMERIC",
            R,
            "Shipping charged to the customer (0.00 when merchandise net >= 75.00, else 6.99).",
            example="0.00",
        ),
        Column(
            "tax_amount",
            "NUMERIC",
            R,
            "Sales tax = ROUND((items_subtotal - discount_amount) * regional rate, 2).",
            example="9.11",
        ),
        Column(
            "order_total",
            "NUMERIC",
            R,
            "items_subtotal - discount_amount + shipping_amount + tax_amount. Amount the customer is asked to pay.",
            example="144.09",
        ),
        Column(
            "cancelled_timestamp",
            "TIMESTAMP",
            N,
            "Set if and only if order_status = 'cancelled'.",
            example="",
        ),
    ),
    foreign_keys=(ForeignKey("customer_id", "customers", "customer_id"),),
    partitioning="DAY on order_date",
    clustering=("order_status", "customer_id"),
    expected_rows="~15,000 (clean)",
    validation_rules=(
        "order_id unique",
        "customer_id exists in customers",
        "order_status in allowed values",
        "order_total = items_subtotal - discount_amount + shipping_amount + tax_amount (±0.01)",
        "items_subtotal = SUM(order_items.gross_amount) (±0.01)",
        "cancelled_timestamp not null iff status = cancelled",
        "order_timestamp <= snapshot as_of timestamp",
    ),
    fanout_warning=(
        "Joining orders to order_items, payments, or refunds multiplies order rows. "
        "Never SUM order-level columns (order_total, shipping_amount, tax_amount) after such a join; "
        "aggregate the child table first, or use COUNT(DISTINCT order_id)."
    ),
)

ORDER_ITEMS = Table(
    name="order_items",
    purpose="Order lines: one product and quantity per line with line-level money.",
    grain="One row per order_item_id (one product line within an order).",
    primary_key=("order_item_id",),
    columns=(
        Column(
            "order_item_id",
            "STRING",
            R,
            "Surrogate order line identifier, format ITM00000001.",
            example="ITM00003391",
        ),
        Column(
            "order_id",
            "STRING",
            R,
            "FK to orders.order_id.",
            example="ORD0001234",
        ),
        Column(
            "line_number",
            "INT64",
            R,
            "1-based line number within the order; (order_id, line_number) is unique.",
            example="2",
        ),
        Column(
            "product_id",
            "STRING",
            R,
            "FK to products.product_id. A product appears at most once per order.",
            example="PRD00017",
        ),
        Column(
            "quantity",
            "INT64",
            R,
            "Units ordered, integer between 1 and 10.",
            example="1",
        ),
        Column(
            "unit_price",
            "NUMERIC",
            R,
            "Price per unit charged on this order (equals product list_price in this version).",
            example="129.99",
        ),
        Column(
            "gross_amount",
            "NUMERIC",
            R,
            "quantity * unit_price.",
            example="129.99",
        ),
        Column(
            "discount_amount",
            "NUMERIC",
            R,
            "Line discount, 0 <= discount_amount < gross_amount.",
            example="13.00",
        ),
        Column(
            "net_amount",
            "NUMERIC",
            R,
            "gross_amount - discount_amount. Merchandise value after discount, excludes tax and shipping.",
            example="116.99",
        ),
    ),
    business_keys=(("order_id", "line_number"), ("order_id", "product_id")),
    foreign_keys=(
        ForeignKey("order_id", "orders", "order_id"),
        ForeignKey("product_id", "products", "product_id"),
    ),
    clustering=("order_id", "product_id"),
    expected_rows="~33,000 (clean)",
    validation_rules=(
        "order_item_id unique; (order_id, line_number) unique",
        "order_id and product_id exist in parents",
        "quantity between 1 and 10",
        "gross_amount = quantity * unit_price; net_amount = gross_amount - discount_amount",
    ),
)

PAYMENTS = Table(
    name="payments",
    purpose="Payment attempts against orders, including failed and pending attempts.",
    grain="One row per payment attempt (payment_id). An order can have several attempts.",
    primary_key=("payment_id",),
    columns=(
        Column(
            "payment_id",
            "STRING",
            R,
            "Surrogate payment attempt identifier, format PAY0000001.",
            example="PAY0001301",
        ),
        Column(
            "order_id",
            "STRING",
            R,
            "FK to orders.order_id.",
            example="ORD0001234",
        ),
        Column(
            "provider_transaction_id",
            "STRING",
            R,
            "Payment provider's transaction reference. Business key: must be unique; duplicates indicate double-recorded payments.",
            example="txn_9f1c2a7be04d3311",
        ),
        Column(
            "payment_timestamp",
            "TIMESTAMP",
            R,
            "When the attempt was processed (UTC); must be >= order_timestamp.",
            example="2026-02-14 18:04:02 UTC",
        ),
        Column(
            "payment_method",
            "STRING",
            R,
            "Instrument used.",
            PAYMENT_METHODS,
            example="credit_card",
        ),
        Column(
            "payment_status",
            "STRING",
            R,
            "captured = money taken; failed = attempt declined; pending = awaiting confirmation.",
            PAYMENT_STATUSES,
            example="captured",
        ),
        Column(
            "amount",
            "NUMERIC",
            R,
            "Attempted amount in USD (> 0). For captured payments equals orders.order_total.",
            example="144.09",
        ),
        Column(
            "failure_reason",
            "STRING",
            N,
            "Set if and only if payment_status = 'failed'.",
            FAILURE_REASONS,
            example="",
        ),
    ),
    business_keys=(("provider_transaction_id",),),
    foreign_keys=(ForeignKey("order_id", "orders", "order_id"),),
    partitioning="none (small table; DATE(payment_timestamp) is the candidate at scale)",
    clustering=("order_id", "payment_status"),
    expected_rows="~15,500 (clean)",
    validation_rules=(
        "payment_id unique; provider_transaction_id unique",
        "order_id exists in orders",
        "payment_timestamp >= order_timestamp",
        "booked orders have exactly one captured payment equal to order_total",
        "failure_reason not null iff status = failed",
    ),
    fanout_warning=(
        "Orders can have a failed attempt before the captured one. "
        "Filter payment_status = 'captured' and aggregate before joining to orders."
    ),
)

SHIPMENTS = Table(
    name="shipments",
    purpose="Physical shipment of an order and its delivery outcome.",
    grain="One row per shipment_id. In this version an order has at most one shipment; models must not assume exactly one.",
    primary_key=("shipment_id",),
    columns=(
        Column(
            "shipment_id",
            "STRING",
            R,
            "Surrogate shipment identifier, format SHP0000001.",
            example="SHP0000877",
        ),
        Column(
            "order_id",
            "STRING",
            R,
            "FK to orders.order_id.",
            example="ORD0001234",
        ),
        Column(
            "carrier",
            "STRING",
            R,
            "Carrier (fictional).",
            CARRIERS,
            example="parcelpro",
        ),
        Column(
            "shipment_status",
            "STRING",
            R,
            "in_transit or delivered as of the snapshot.",
            SHIPMENT_STATUSES,
            example="delivered",
        ),
        Column(
            "shipped_timestamp",
            "TIMESTAMP",
            R,
            "When the parcel left the warehouse (UTC); >= order_timestamp.",
            example="2026-02-15 10:20:00 UTC",
        ),
        Column(
            "promised_delivery_date",
            "DATE",
            R,
            "Delivery date promised at checkout = order_date + 5 days.",
            example="2026-02-19",
        ),
        Column(
            "delivered_timestamp",
            "TIMESTAMP",
            N,
            "Set if and only if shipment_status = 'delivered'; must be >= shipped_timestamp.",
            example="2026-02-18 14:41:09 UTC",
        ),
    ),
    business_keys=(("order_id",),),
    foreign_keys=(ForeignKey("order_id", "orders", "order_id"),),
    clustering=("carrier", "shipment_status"),
    expected_rows="~13,800 (clean)",
    validation_rules=(
        "shipment_id unique; at most one shipment per order (this version)",
        "delivered_timestamp not null iff status = delivered",
        "delivered_timestamp >= shipped_timestamp",
        "shipment status consistent with order status",
    ),
)

REFUNDS = Table(
    name="refunds",
    purpose="Refund transactions: item-level returns on delivered orders and full refunds of cancelled-but-paid orders.",
    grain="One row per refund transaction (refund_id).",
    primary_key=("refund_id",),
    columns=(
        Column(
            "refund_id",
            "STRING",
            R,
            "Surrogate refund identifier, format REF000001.",
            example="REF000118",
        ),
        Column(
            "order_id",
            "STRING",
            R,
            "FK to orders.order_id.",
            example="ORD0001234",
        ),
        Column(
            "order_item_id",
            "STRING",
            N,
            "FK to order_items. NULL means an order-level refund (used for cancellations, refunding the full captured amount).",
            example="ITM00003391",
        ),
        Column(
            "payment_id",
            "STRING",
            R,
            "FK to the captured payments.payment_id being refunded.",
            example="PAY0001301",
        ),
        Column(
            "refund_timestamp",
            "TIMESTAMP",
            R,
            "When the refund was initiated (UTC).",
            example="2026-02-27 08:00:12 UTC",
        ),
        Column(
            "refund_amount",
            "NUMERIC",
            R,
            "Refunded amount in USD (> 0). Item refunds cover merchandise only (no tax/shipping) and may be partial.",
            example="58.50",
        ),
        Column(
            "refund_reason",
            "STRING",
            R,
            "Reason code.",
            REFUND_REASONS,
            example="damaged",
        ),
        Column(
            "refund_status",
            "STRING",
            R,
            "completed = money returned; pending = initiated, not yet settled. Only completed refunds reduce net sales.",
            REFUND_STATUSES,
            example="completed",
        ),
    ),
    foreign_keys=(
        ForeignKey("order_id", "orders", "order_id"),
        ForeignKey(
            "order_item_id", "order_items", "order_item_id", nullable=True
        ),
        ForeignKey("payment_id", "payments", "payment_id"),
    ),
    clustering=("order_id", "refund_status"),
    expected_rows="~1,700 (clean)",
    validation_rules=(
        "refund_id unique; FKs exist",
        "SUM(refund_amount) per order_item <= order_items.net_amount",
        "order-level refund <= captured amount",
        "order_item_id is NULL only for refund_reason = 'order_cancelled'",
    ),
)

TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        CUSTOMERS,
        PRODUCTS,
        ORDERS,
        ORDER_ITEMS,
        PAYMENTS,
        SHIPMENTS,
        REFUNDS,
    )
}
TABLE_LOAD_ORDER = (
    "customers",
    "products",
    "orders",
    "order_items",
    "payments",
    "shipments",
    "refunds",
)


def get_table(name: str) -> Table:
    """Look up a registered table by name.

    Args:
        name: Table name (one of the keys in ``TABLES``).

    Returns:
        The corresponding ``Table``.

    Raises:
        KeyError: If *name* is not an approved table, with a message listing
            the approved names.
    """
    try:
        return TABLES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown table {name!r}. Approved tables: {sorted(TABLES)}"
        ) from exc


def render_data_dictionary_markdown() -> str:
    """Render the data dictionary from the registry so docs never drift from code.

    Produces a Markdown document covering every table in ``TABLE_LOAD_ORDER``
    with purpose, grain, keys, partitioning, columns, validation rules and
    fan-out warnings.

    Returns:
        A complete Markdown string suitable for writing to
        ``docs/data_dictionary.md``.
    """
    lines = [
        "# Data dictionary",
        "",
        "_Generated from `src/analytics_agent/schema_registry.py` by "
        "`scripts/build_docs.py`. Do not edit by hand._",
        "",
    ]
    for name in TABLE_LOAD_ORDER:
        t = TABLES[name]
        lines += [
            f"## `{t.name}`",
            "",
            f"**Purpose:** {t.purpose}",
            "",
            f"**Grain:** {t.grain}",
            "",
            f"**Primary key:** `{', '.join(t.primary_key)}`",
            "",
        ]
        if t.business_keys:
            bk = "; ".join(
                "(" + ", ".join(k) + ")" for k in t.business_keys
            )
            lines += [f"**Other uniqueness constraints:** {bk}", ""]
        if t.foreign_keys:
            fks = "; ".join(
                f"`{fk.column}` → `{fk.ref_table}.{fk.ref_column}`"
                + (" (nullable)" if fk.nullable else "")
                for fk in t.foreign_keys
            )
            lines += [f"**Foreign keys:** {fks}", ""]
        lines += [
            f"**Partitioning:** {t.partitioning} &nbsp; "
            f"**Clustering:** {', '.join(t.clustering) or 'none'}",
            "",
            f"**Expected rows:** {t.expected_rows}",
            "",
        ]
        lines += [
            "| Column | Type | Mode | Description | Allowed values | Example |",
            "|---|---|---|---|---|---|",
        ]
        for c in t.columns:
            allowed = (
                ", ".join(f"`{v}`" for v in c.allowed_values)
                if c.allowed_values
                else ""
            )
            desc = c.description + (" **(PII)**" if c.pii else "")
            lines.append(
                f"| `{c.name}` | {c.bq_type} | {c.mode} | {desc} | "
                f"{allowed} | {c.example} |"
            )
        lines += (
            ["", "**Validation rules:**", ""]
            + [f"- {r}" for r in t.validation_rules]
            + [""]
        )
        if t.fanout_warning:
            lines += [
                f"> **Join fan-out warning:** {t.fanout_warning}",
                "",
            ]
    return "\n".join(lines)