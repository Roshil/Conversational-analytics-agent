"""Order-item and order lookups: deterministic, parameterised, no LLM.

* "What product is in order item ITM00000001?"  -> one line of the order_items table (with the product's name and category)
* "Which items are in order ORD0000002?"         -> every line of that order, plus order totals computed from the lines
* "Was there a discount on it?"                  -> follow-up: uses the item or order from the conversation, only when
                                                    exactly one was discussed; otherwise the agent asks which one

Each lookup is a fixed query with one bound parameter (@order_item_id or @order_id). Outcomes are distinguished:
found, not_found, conflicting (the same item ID has records that disagree) and error (the query failed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from analytics_agent.bq import SqlParam
from analytics_agent.customer_lookup import LookupClarification

ITEM_ID = re.compile(r"\bitm\d{8}\b", re.IGNORECASE)
ORDER_ID = re.compile(r"\bord\d{7}\b", re.IGNORECASE)
MAX_IDS = 5

ITEM_FIELDS = {
    "product": ("product", "what was bought", "what did they buy", "sku", "product name", "category"),
    "quantity": ("quantity", "how many units", "units", "how many were", "qty"),
    "unit_price": ("unit price", "price", "cost each", "per unit"),
    "gross_amount": ("gross", "before discount", "full price"),
    "discount_amount": ("discount", "discounted", "reduction"),
    "net_amount": ("net", "paid", "amount", "total", "value", "cost", "how much"),
    "line_number": ("line number", "line", "position"),
}
REFERENCE_WORDS = ("it", "its", "that item", "this item", "the item", "that line", "that order", "this order",
                   "the order", "same order", "them", "they", "those items", "these items")
# Information that is NOT stored for an order line (tax and shipping are handled separately: the answer explains that
# order_items holds merchandise only).
UNAVAILABLE_ITEM_INFO = re.compile(
    r"\b(profit|margin|markup|unit cost|cost price|cogs|supplier|vendor|warehouse|stock level|inventory|"
    r"return reason|tracking number|customer name|e-?mail)\b", re.IGNORECASE)
# Without an ID these words mean an aggregate question (metric pipeline), not a follow-up about one item or order.
AGGREGATE_WORDS = re.compile(r"\b(by|per|each|every|all|average|how many orders|percentage|percent|share|most|top|"
                             r"best|category|categories|region|month|quarter|year|week|last|trend)\b", re.IGNORECASE)


@dataclass
class OrderItemRequest:
    item_ids: list
    order_ids: list
    fields: list
    existence_only: bool = False
    source: str = "question"                     # "question" or "conversation"

    @property
    def record_ids(self) -> list:
        return list(self.item_ids) + list(self.order_ids)


@dataclass
class ItemRecord:
    record_id: str
    kind: str                                    # "item" or "order"
    status: str                                  # found | not_found | conflicting | error
    lines: list = field(default_factory=list)    # one dict per order line
    conflicting_fields: list = field(default_factory=list)
    error: str = ""

    @property
    def totals(self) -> dict:
        def total(column):
            return round(sum(Decimal(str(line[column] or 0)) for line in self.lines), 2)
        return {"lines": len(self.lines), "units": sum(int(line["quantity"] or 0) for line in self.lines),
                "gross_amount": float(total("gross_amount")), "discount_amount": float(total("discount_amount")),
                "net_amount": float(total("net_amount"))}


@dataclass
class OrderItemResult:
    request: OrderItemRequest
    records: list
    sql: str

    @property
    def status(self) -> str:
        statuses = {r.status for r in self.records}
        if "error" in statuses:
            return "error"
        if statuses == {"not_found"}:
            return "no_data"
        return "answered"


# ----------------------------------------------------------------------------- parsing

def _mentions(text: str, phrases) -> bool:
    return any(re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", text) for p in phrases)


def requested_fields(text: str) -> list:
    low = text.lower()
    found = [name for name, words in ITEM_FIELDS.items() if _mentions(low, words)]
    return found or list(ITEM_FIELDS)


def parse_order_item_lookup(question: str, recent_ids: list | None = None):
    """Return an OrderItemRequest, a LookupClarification, or None when the question is not an item/order lookup."""
    text = (question or "").strip()
    low = text.lower()
    item_ids = list(dict.fromkeys(m.upper() for m in ITEM_ID.findall(text)))
    order_ids = list(dict.fromkeys(m.upper() for m in ORDER_ID.findall(text)))
    if len(item_ids) + len(order_ids) > MAX_IDS:
        return LookupClarification(f"Please ask about at most {MAX_IDS} order items or orders at a time.")
    existence = _mentions(low, ("exist", "exists", "valid"))
    unavailable = UNAVAILABLE_ITEM_INFO.search(low)
    if (item_ids or order_ids) and unavailable:
        return LookupClarification(
            f"I don't have \"{unavailable.group(0)}\" for order items: an order line stores only the product, quantity, "
            "unit price, gross amount, discount and net amount (plus the product's name and category). "
            "Profit and margin are not defined metrics, so they are not calculated.")
    if item_ids or order_ids:
        return OrderItemRequest(item_ids, order_ids, requested_fields(low), existence_only=existence)

    if AGGREGATE_WORDS.search(low) or not _mentions(low, REFERENCE_WORDS):
        return None
    recent = list(dict.fromkeys(recent_ids or []))
    recent_items = [r for r in recent if r.startswith("ITM")]
    recent_orders = [r for r in recent if r.startswith("ORD")]
    asks_about_order = _mentions(low, ("that order", "this order", "the order", "same order"))
    if asks_about_order and len(recent_orders) == 1:
        return OrderItemRequest([], recent_orders, requested_fields(low), source="conversation")
    if not asks_about_order and len(recent_items) == 1 and not recent_orders:
        return OrderItemRequest(recent_items, [], requested_fields(low), source="conversation")
    if not asks_about_order and len(recent_orders) == 1 and not recent_items:
        return OrderItemRequest([], recent_orders, requested_fields(low), source="conversation")
    if recent_items or recent_orders:
        return LookupClarification(f"Which one do you mean: {', '.join(recent)}? Please give the order item or order ID.")
    return None                                  # no item/order in the conversation: not an item lookup


# ----------------------------------------------------------------------------- execution

def _sql(refs: dict, by: str) -> str:
    where = "oi.order_item_id = @order_item_id" if by == "item" else "oi.order_id = @order_id"
    return (f"SELECT oi.order_item_id, oi.order_id, oi.line_number, oi.product_id, p.product_name, p.category,\n"
            f"       oi.quantity, oi.unit_price, oi.gross_amount, oi.discount_amount, oi.net_amount\n"
            f"FROM {refs['order_items']} AS oi\n"
            f"LEFT JOIN {refs['products']} AS p ON p.product_id = oi.product_id\n"
            f"WHERE {where}\n"
            f"ORDER BY oi.line_number\n"
            f"LIMIT 50")


def _clean(value):
    if isinstance(value, Decimal):
        return float(round(value, 2))
    if isinstance(value, float):
        return round(value, 2)
    return value


def lookup_order_items(runner, settings, dataset: str, request: OrderItemRequest) -> OrderItemResult:
    refs = settings.refs(dataset)
    records = []
    for kind, ids, param in (("item", request.item_ids, "order_item_id"), ("order", request.order_ids, "order_id")):
        for record_id in ids:
            res = runner.run(_sql(refs, kind), [SqlParam(param, "STRING", record_id)], max_rows=50,
                             allowed_tables=settings.allowed_tables(dataset), dry_run_first=False)
            if not res.ok:
                records.append(ItemRecord(record_id, kind, "error", error=f"{res.error_kind}: {res.error_message}"))
                continue
            lines = [{k: _clean(v) for k, v in row.items()} for row in res.rows]
            if not lines:
                records.append(ItemRecord(record_id, kind, "not_found"))
                continue
            conflicts = []
            if kind == "item" and len(lines) > 1:
                conflicts = [c for c in lines[0] if len({str(line[c]) for line in lines}) > 1]
            records.append(ItemRecord(record_id, kind, "conflicting" if conflicts else "found", lines, conflicts))
    sql = _sql(refs, "item" if request.item_ids else "order")
    return OrderItemResult(request, records, sql)


# ----------------------------------------------------------------------------- answer

def _money(value) -> str:
    return f"{float(value):,.2f}"


def _describe_line(line: dict, fields: list) -> str:
    product = f"{line['product_id']}" + (f" ({line['product_name']}, {line['category']})" if line.get("product_name") else "")
    parts = []
    if "product" in fields:
        parts.append(f"product {product}")
    if "quantity" in fields:
        parts.append(f"quantity {line['quantity']}")
    if "unit_price" in fields:
        parts.append(f"unit price {_money(line['unit_price'])}")
    if "gross_amount" in fields:
        parts.append(f"gross amount {_money(line['gross_amount'])}")
    if "discount_amount" in fields:
        parts.append(f"discount {_money(line['discount_amount'])}")
    if "net_amount" in fields:
        parts.append(f"net amount {_money(line['net_amount'])}")
    if "line_number" in fields:
        parts.append(f"line {line['line_number']} of order {line['order_id']}")
    return "; ".join(parts)


def format_order_item_answer(result: OrderItemResult) -> str:
    out = []
    fields = result.request.fields
    for record in result.records:
        noun = "Order item" if record.kind == "item" else "Order"
        if record.status == "error":
            out.append(f"I could not look up {record.record_id}: the query failed ({record.error.split(':')[0]}). "
                       "This is not the same as it not existing.")
        elif record.status == "not_found":
            out.append(f"{noun} {record.record_id} does not exist in the order_items table.")
        elif result.request.existence_only:
            out.append(f"Yes, {noun.lower()} {record.record_id} exists.")
        elif record.kind == "item":
            line = record.lines[0]
            out.append(f"Order item {record.record_id} (line {line['line_number']} of order {line['order_id']}): "
                       f"{_describe_line(line, fields)}.")
            if record.status == "conflicting":
                out.append(f"Data-quality warning: {len(record.lines)} records exist for {record.record_id} and they "
                           f"disagree on {', '.join(record.conflicting_fields)}; the first record is shown.")
            elif len(record.lines) > 1:
                out.append(f"Note: {len(record.lines)} identical records exist for {record.record_id} (duplicate ID).")
        else:
            totals = record.totals
            out.append(f"Order {record.record_id} has {totals['lines']} line item(s), {totals['units']} unit(s) in total:")
            line_fields = [f for f in fields if f != "line_number"]          # the line number is already shown
            for line in record.lines:
                out.append(f"  - {line['order_item_id']} (line {line['line_number']}): {_describe_line(line, line_fields)}")
            out.append(f"Order totals from its lines: gross {_money(totals['gross_amount'])}, discount "
                       f"{_money(totals['discount_amount'])}, net {_money(totals['net_amount'])} "
                       "(merchandise only: tax and shipping are not part of order_items).")
    if result.request.source == "conversation":
        out.append(f"(Taken from the conversation: {', '.join(result.request.record_ids)}.)")
    return "\n".join(out)