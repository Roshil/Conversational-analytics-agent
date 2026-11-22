import json

from analytics_agent import schema_registry as sr

VALID_TYPES = {"STRING", "INT64", "NUMERIC", "BOOL", "DATE", "TIMESTAMP"}


def test_seven_tables_in_load_order():
    assert list(sr.TABLE_LOAD_ORDER) == list(sr.TABLES)
    assert len(sr.TABLES) == 7


def test_columns_unique_and_typed():
    for t in sr.TABLES.values():
        assert len(t.column_names) == len(set(t.column_names)), t.name
        for c in t.columns:
            assert c.bq_type in VALID_TYPES, (t.name, c.name)
            assert c.mode in {"REQUIRED", "NULLABLE"}
            assert c.description


def test_keys_reference_real_columns():
    for t in sr.TABLES.values():
        for col in t.primary_key:
            assert t.column(col).mode == "REQUIRED"
        for key in t.business_keys:
            for col in key:
                t.column(col)
        for fk in t.foreign_keys:
            t.column(fk.column)
            sr.get_table(fk.ref_table).column(fk.ref_column)


def test_parents_load_before_children():
    pos = {n: i for i, n in enumerate(sr.TABLE_LOAD_ORDER)}
    for t in sr.TABLES.values():
        for fk in t.foreign_keys:
            assert pos[fk.ref_table] < pos[t.name]


def test_booked_statuses_subset_of_allowed():
    assert set(sr.BOOKED_ORDER_STATUSES) <= set(sr.ORDER_STATUSES)
    assert "cancelled" not in sr.BOOKED_ORDER_STATUSES


def test_bigquery_json_uses_legacy_type_names():
    js = sr.get_table("order_items").to_bigquery_json()
    types = {f["name"]: f["type"] for f in js}
    assert types["quantity"] == "INTEGER"
    assert json.dumps(js)


def test_pii_flagged():
    assert sr.CUSTOMERS.column("email").pii


def test_data_dictionary_mentions_every_column():
    md = sr.render_data_dictionary_markdown()
    for t in sr.TABLES.values():
        assert f"## `{t.name}`" in md
        for c in t.columns:
            assert f"`{c.name}`" in md
