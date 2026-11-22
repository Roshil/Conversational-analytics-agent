# 03 — Defect catalog (corrupted dataset)

_Generated from `data/manifests/defect_catalog.json`. Each order is touched by at most one defect; expected findings include genuine cascades. A perfect validator reports exactly these findings and nothing else._

| ID | Table | Columns | Type | Severity | Affected | Detection rule | Expected findings |
|---|---|---|---|---|---|---|---|
| D01 | `customers` | customer_id, email, region | duplicate_primary_key | critical | 5 | `CUS_PK_UNIQUE` | `CUS_PK_UNIQUE`: 5; `CUS_EMAIL_UNIQUE`: 5 |
| D02 | `customers` | region | missing_required_value | high | 10 | `CUS_REQUIRED_NOT_NULL` | `CUS_REQUIRED_NOT_NULL`: 10 |
| D03 | `products` | category | invalid_category | medium | 3 | `PRD_CATEGORY_ALLOWED` | `PRD_CATEGORY_ALLOWED`: 3 |
| D04 | `products` | sku | duplicate_business_key | high | 4 | `PRD_SKU_UNIQUE` | `PRD_SKU_UNIQUE`: 2 |
| D05 | `orders` | customer_id | orphan_foreign_key | high | 8 | `ORD_CUSTOMER_FK` | `ORD_CUSTOMER_FK`: 8 |
| D06 | `orders` | order_status | invalid_status | medium | 6 | `ORD_STATUS_ALLOWED` | `ORD_STATUS_ALLOWED`: 6 |
| D07 | `orders` | tax_amount, order_total | incorrect_monetary_total | high | 10 | `ORD_TOTAL_CONSISTENT` | `ORD_TOTAL_CONSISTENT`: 10 |
| D08 | `order_items` | quantity | negative_quantity | high | 5 | `ITM_QUANTITY_RANGE` | `ITM_QUANTITY_RANGE`: 5; `ITM_AMOUNTS_CONSISTENT`: 5 |
| D09 | `order_items` | order_id | orphan_foreign_key | high | 5 | `ITM_ORDER_FK` | `ITM_ORDER_FK`: 5 |
| D10 | `order_items` | * | duplicate_rows | critical | 12 | `ITM_PK_UNIQUE` | `ITM_PK_UNIQUE`: 12; `ITM_LINE_UNIQUE`: 12; `ORD_SUBTOTAL_MATCHES_ITEMS`: 12 |
| D11 | `payments` | provider_transaction_id | duplicate_transaction | critical | 15 | `PAY_TXN_UNIQUE` | `PAY_TXN_UNIQUE`: 15; `PAY_CAPTURED_MATCHES_ORDER`: 15 |
| D12 | `payments` | payment_timestamp | invalid_timestamp_sequence | medium | 5 | `PAY_AFTER_ORDER` | `PAY_AFTER_ORDER`: 5 |
| D13 | `refunds` | refund_amount | refund_exceeds_refundable | critical | 6 | `REF_WITHIN_REFUNDABLE` | `REF_WITHIN_REFUNDABLE`: 6 |
| D14 | `shipments` | delivered_timestamp | invalid_timestamp_sequence | high | 7 | `SHP_DELIVERY_AFTER_SHIP` | `SHP_DELIVERY_AFTER_SHIP`: 7 |
| D15 | `shipments` | shipment_status, delivered_timestamp | inconsistent_state | high | 6 | `SHP_DELIVERED_TS_CONSISTENT` | `SHP_DELIVERED_TS_CONSISTENT`: 6 |

### D01 — duplicate_primary_key

**What:** 5 customers re-inserted with the same customer_id and email but a different region (e.g. a botched CRM sync).

**Business impact:** Joining orders to customers doubles these customers' orders, inflating regional sales and AOV; the customer has two conflicting regions.

**Expected outcome:** Fail CUS_PK_UNIQUE with 5 keys; also CUS_EMAIL_UNIQUE (same emails). Analytics joining customers must be flagged as unreliable for these customers.

**Example affected ids:** CUS000192, CUS000705, CUS000766, CUS001219, CUS001813

### D02 — missing_required_value

**What:** region set to NULL for 10 customers.

**Business impact:** Their orders fall into a NULL region bucket in regional reports.

**Expected outcome:** Fail CUS_REQUIRED_NOT_NULL with 10 customers. Regional answers must disclose an 'unknown region' share.

**Example affected ids:** CUS000102, CUS000391, CUS000626, CUS000672, CUS000692

### D03 — invalid_category

**What:** 3 products get non-canonical categories: a typo, a case variant and a spelling variant.

**Business impact:** Category totals split into phantom categories; canonical categories are under-reported.

**Expected outcome:** Fail PRD_CATEGORY_ALLOWED with 3 products.

**Example affected ids:** PRD00106, PRD00132, PRD00175

### D04 — duplicate_business_key

**What:** 2 pairs of different products share a SKU.

**Business impact:** Inventory/SKU-level reporting merges unrelated products.

**Expected outcome:** Fail PRD_SKU_UNIQUE with 2 SKU values (4 products).

**Example affected ids:** PRD00012, PRD00101, PRD00150, PRD00233

### D05 — orphan_foreign_key

**What:** 8 booked orders reference customer_ids that do not exist.

**Business impact:** INNER JOIN to customers silently drops these orders' sales; LEFT JOIN yields NULL region.

**Expected outcome:** Fail ORD_CUSTOMER_FK with 8 orders.

**Example affected ids:** ORD0001550, ORD0003289, ORD0003772, ORD0004014, ORD0004845

### D06 — invalid_status

**What:** 6 delivered orders carry the non-canonical status 'complete'.

**Business impact:** Filters on booked statuses exclude these orders, understating sales.

**Expected outcome:** Fail ORD_STATUS_ALLOWED with 6 orders. Consistency rules skip records already failing the status domain.

**Example affected ids:** ORD0002530, ORD0002992, ORD0003690, ORD0008736, ORD0011021

### D07 — incorrect_monetary_total

**What:** tax_amount zeroed on 10 booked orders while order_total (and the captured payment) still include tax.

**Business impact:** Tax reporting understated; order_total no longer equals its components.

**Expected outcome:** Fail ORD_TOTAL_CONSISTENT with 10 orders. Payments still reconcile to order_total, so PAY_CAPTURED_MATCHES_ORDER passes.

**Example affected ids:** ORD0000653, ORD0000683, ORD0002003, ORD0002359, ORD0002452

### D08 — negative_quantity

**What:** quantity negated on 5 lines; money columns left unchanged.

**Business impact:** Unit totals understated; quantity * unit_price no longer equals gross_amount.

**Expected outcome:** Fail ITM_QUANTITY_RANGE and ITM_AMOUNTS_CONSISTENT with the same 5 lines.

**Example affected ids:** ITM00003673, ITM00005733, ITM00007483, ITM00015213, ITM00022701

### D09 — orphan_foreign_key

**What:** 5 order lines inserted that reference non-existent orders.

**Business impact:** Item-level sales that do not join to an order header; item-only queries over-report.

**Expected outcome:** Fail ITM_ORDER_FK with 5 lines.

**Example affected ids:** ITM90000000, ITM90000001, ITM90000002, ITM90000003, ITM90000004

### D10 — duplicate_rows

**What:** 12 order lines loaded twice (exact duplicate rows).

**Business impact:** Item-based sales metrics are inflated by the duplicated lines' net_amount (fan-out).

**Expected outcome:** Fail ITM_PK_UNIQUE and ITM_LINE_UNIQUE with 12 lines; cascade: ORD_SUBTOTAL_MATCHES_ITEMS fails for their 12 orders.

**Example affected ids:** ITM00002344, ITM00009253, ITM00009631, ITM00014255, ITM00023525

### D11 — duplicate_transaction

**What:** 15 captured payments recorded twice under new payment_ids with the same provider_transaction_id.

**Business impact:** Total payments received is overstated by the duplicated amounts; payments no longer reconcile to orders.

**Expected outcome:** Fail PAY_TXN_UNIQUE with 15 transaction ids; cascade: PAY_CAPTURED_MATCHES_ORDER fails for the 15 orders.

**Example affected ids:** PAY9000000, PAY9000001, PAY9000002, PAY9000003, PAY9000004

### D12 — invalid_timestamp_sequence

**What:** 5 captured payments timestamped 1-48 hours before their order was placed.

**Business impact:** Payment-date reports shift cash into earlier periods; indicates clock or ETL timezone problems.

**Expected outcome:** Fail PAY_AFTER_ORDER with 5 payments.

**Example affected ids:** PAY0000404, PAY0002340, PAY0002623, PAY0006684, PAY0012214

### D13 — refund_exceeds_refundable

**What:** 6 completed item refunds set to 3x the line's net amount.

**Business impact:** Net sales understated; potential financial loss or fraud signal.

**Expected outcome:** Fail REF_WITHIN_REFUNDABLE with 6 refunds. Legitimate partial refunds must still pass.

**Example affected ids:** REF000186, REF000199, REF000208, REF000228, REF000587

### D14 — invalid_timestamp_sequence

**What:** 7 delivered shipments have delivered_timestamp earlier than shipped_timestamp.

**Business impact:** Delivery-time metrics become negative/flattering; on-time rate overstated.

**Expected outcome:** Fail SHP_DELIVERY_AFTER_SHIP with 7 shipments.

**Example affected ids:** SHP0003754, SHP0005112, SHP0005278, SHP0006270, SHP0009584

### D15 — inconsistent_state

**What:** 6 shipments marked delivered with no delivered_timestamp.

**Business impact:** These shipments silently drop out of on-time/late calculations.

**Expected outcome:** Fail SHP_DELIVERED_TS_CONSISTENT with 6 shipments.

**Example affected ids:** SHP0001329, SHP0002271, SHP0002470, SHP0003201, SHP0005228

## Legitimate cases that must NOT be flagged

| Case | Count (clean) | Why it is legitimate |
|---|---|---|
| pending_payment_orders | 22 | Recent orders awaiting payment confirmation; excluded from sales, not an error. |
| failed_then_captured_payments | 994 | A declined attempt followed by a successful one; only the captured row counts. |
| cancelled_orders_with_captured_payment_and_full_refund | 232 | Paid then cancelled; an order-level refund returns the full amount. |
| cancelled_orders_without_payment | 190 | Cancelled before any payment attempt. |
| partial_item_refunds | 528 | Refund below the line's net amount (e.g. goodwill credit). |
| pending_refunds | 28 | Initiated within the last days of the snapshot; not yet settled. |
| in_transit_shipments | 54 | No delivered_timestamp yet; excluded from on-time denominators. |
| processing_orders_without_shipment | 3 | Paid but not yet shipped. |
| customers_without_orders | 98 | Registered but never purchased. |
| products_never_sold | 0 | Catalog items with no sales in the window. |
