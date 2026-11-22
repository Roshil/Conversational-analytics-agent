"""Customer record lookup: deterministic, parameterised, privacy-aware.

Questions such as "What is the email address of customer CUS000001?" or "Does CUS999999 exist?" do not need an LLM to
write SQL. They are answered by a fixed query template with a bound @customer_id parameter, so user input is never
concatenated into SQL and no model can widen the query.

Behaviour:
* customer IDs are recognised in any casing ("cus000001"); up to MAX_IDS per question;
* requested fields are matched on synonyms and tolerant of small spelling mistakes ("emial", "segement");
  no specific field, or "all details", means all permitted fields;
* "that customer" / "their" / "this customer" uses the conversation ONLY when exactly one customer was discussed;
  otherwise the agent asks which customer is meant;
* outcomes are distinguished: found, not_found, conflicting (the same ID has differing records), error (query failed);
* email is a sensitive field: it is masked (c***@example.com) unless revealing is explicitly enabled.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from analytics_agent.bq import SqlParam

CUSTOMER_ID = re.compile(r"\bcus\d{6}\b", re.IGNORECASE)
# Things that look like a customer ID but are not valid: wrong number of digits or a separator (CUS00001, CUS-000001)
MALFORMED_ID = re.compile(r"\bcus(?:[-_ ]?\d{1,5}|[-_ ]?\d{7,}|[-_ ]\d{6})\b", re.IGNORECASE)
MAX_IDS = 5

PERMITTED_FIELDS = ("customer_id", "email", "region", "country_code", "customer_segment", "signup_date", "created_at")
SENSITIVE_FIELDS = frozenset({"email"})

FIELD_SYNONYMS = {
    "email": ("email", "e-mail", "mail", "email address"),
    "region": ("region", "area", "location"),
    "country_code": ("country", "country code", "nation"),
    "customer_segment": ("segment", "customer segment", "category of customer", "customer type", "type of customer"),
    "signup_date": ("signup", "sign up", "sign-up", "signed up", "joined", "join date", "registration", "registered"),
    "created_at": ("created", "creation", "created at", "record created"),
}
ALL_FIELDS_WORDS = ("all details", "all available", "everything", "all fields", "full record", "profile", "all info",
                    "all information", "all the details", "details for", "details of", "about")
EXISTENCE_WORDS = ("exist", "exists", "is there a customer", "is there any customer", "valid customer")
REFERENCE_WORDS = ("that customer", "this customer", "the same customer", "same customer", "their", "them", "his", "her",
                   "that one", "this one", "the customer", "it", "its", "they")
# A question mentioning these is an analytics question (e.g. "net sales of CUS000001"), not a record lookup.
ANALYTICS_WORDS = re.compile(r"\b(sales|revenue|orders?|refunds?|payments?|spend|spent|aov|average|total value|shipments?|"
                             r"deliver(y|ies|ed)|purchases?|purchased|bought|buy|buys)\b", re.IGNORECASE)
# Without a customer ID, these words mean the question is about many customers (an aggregate), not a follow-up lookup.
AGGREGATE_WORDS = re.compile(r"\b(by|per|each|every|all|total|average|how many|count|percentage|percent|share|"
                             r"lifetime|retention|churn|most|top|best|valuable)\b", re.IGNORECASE)
# Customer attributes that are NOT stored in the customers table ("email address" is the email field, not an address).
UNAVAILABLE_ATTRIBUTES = re.compile(
    r"(?<!email )(?<!e-mail )\b(home |postal |street |mailing |billing |shipping )?address(es)?\b|\bphone|\btelephone|\bmobile\b|"
    r"\b(first|last|full|customer'?s?) name\b|\bnames?\b|\bbirthday|date of birth|\bage\b|\bgender\b|\bincome\b|\bsalary\b|"
    r"\bpassword|credit card|\bssn\b|social security", re.IGNORECASE)


@dataclass
class LookupRequest:
    customer_ids: list[str]
    fields: list[str]
    existence_only: bool = False
    source: str = "question"                 # "question" or "conversation"


@dataclass
class LookupClarification:
    question: str


@dataclass
class CustomerRecord:
    customer_id: str
    status: str                              # found | not_found | conflicting | error
    values: dict = field(default_factory=dict)
    record_count: int = 0
    conflicting_fields: list = field(default_factory=list)
    error: str = ""


@dataclass
class LookupResult:
    request: LookupRequest
    records: list[CustomerRecord]
    sql: str
    email_revealed: bool

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


def _fuzzy_field(token: str) -> Optional[str]:
    """Map a possibly misspelt word to a field name ('emial' -> email, 'segement' -> customer_segment)."""
    choices = {syn: name for name, syns in FIELD_SYNONYMS.items() for syn in syns if " " not in syn and len(syn) >= 5}
    match = difflib.get_close_matches(token, list(choices), n=1, cutoff=0.8)
    return choices[match[0]] if match else None


EXCLUSION = re.compile(r"\b(without|except|excluding|exclude|leave out|leaving out|but not|apart from|other than|"
                       r"skip|omit|hide)\b(?P<rest>[^.?!;]*)", re.IGNORECASE)


def excluded_fields(text: str) -> set:
    """Fields the user explicitly asked to leave out ("... but leave out the email")."""
    excluded = set()
    for match in EXCLUSION.finditer(text.lower()):
        rest = match.group("rest")
        excluded |= {name for name, syns in FIELD_SYNONYMS.items() if _mentions(rest, syns)}
    return excluded


def requested_fields(text: str) -> list[str]:
    low = text.lower()
    excluded = excluded_fields(low)
    if excluded:
        # A field mentioned only to exclude it is not requested: drop the exclusion clause before matching fields.
        kept_text = EXCLUSION.sub(" ", low)
        named = [name for name, syns in FIELD_SYNONYMS.items() if _mentions(kept_text, syns) and name not in excluded]
        return named or [f for f in PERMITTED_FIELDS if f not in excluded]
    if _mentions(low, ALL_FIELDS_WORDS):
        return list(PERMITTED_FIELDS)
    found = [name for name, syns in FIELD_SYNONYMS.items() if _mentions(low, syns)]
    if not found:
        for token in re.findall(r"[a-z\-]{5,}", low):
            name = _fuzzy_field(token)
            if name and name not in found:
                found.append(name)
    return found or list(PERMITTED_FIELDS)


def parse_customer_lookup(question: str, recent_customer_ids: list[str] | None = None):
    """Return a LookupRequest, a LookupClarification, or None when the question is not a record lookup."""
    text = (question or "").strip()
    low = text.lower()
    ids = list(dict.fromkeys(m.upper() for m in CUSTOMER_ID.findall(text)))
    malformed = [m.group(0) for m in MALFORMED_ID.finditer(text)]
    if malformed:
        return LookupClarification(
            f"{', '.join(malformed)} is not a valid customer ID: IDs are 'CUS' followed by exactly 6 digits "
            "(for example CUS000001). Please check the ID.")
    unavailable = UNAVAILABLE_ATTRIBUTES.search(low)
    if ids and unavailable:
        return LookupClarification(
            f"I don't have \"{unavailable.group(0).strip()}\" for customers: the customers table stores only email (masked), "
            "region, country, segment, signup date and record creation time. Purchases are in the orders data: ask a "
            "separate question about this customer's orders or sales.")
    if ids and ANALYTICS_WORDS.search(low):
        return None                          # analytics about a customer goes through the metric pipeline
    if len(ids) > MAX_IDS:
        return LookupClarification(f"Please ask about at most {MAX_IDS} customer IDs at a time.")
    existence = _mentions(low, EXISTENCE_WORDS)
    if ids:
        return LookupRequest(ids, requested_fields(low), existence_only=existence and not _named_field(low))

    if AGGREGATE_WORDS.search(low):
        return None                          # "lifetime value by segment", "all customers": not about one customer
    refers_back = _mentions(low, REFERENCE_WORDS)
    looks_like_lookup = refers_back and (_named_field(low) or _mentions(low, ("look up", "lookup", "details", "record")))
    if not looks_like_lookup:
        return None
    recent = list(dict.fromkeys(recent_customer_ids or []))
    if len(recent) == 1:
        return LookupRequest(recent, requested_fields(low), source="conversation")
    if not recent:
        return LookupClarification("Which customer do you mean? Please give the customer ID (for example CUS000001).")
    return LookupClarification(f"Which customer do you mean: {', '.join(recent)}? Please give the customer ID.")


def _named_field(low: str) -> bool:
    return any(_mentions(low, syns) for syns in FIELD_SYNONYMS.values())


# ----------------------------------------------------------------------------- execution

def lookup_sql(customers_ref: str, fields: list[str]) -> str:
    """Fixed template: only permitted columns, one bound parameter, no user text in the SQL."""
    columns = ["customer_id"] + [f for f in fields if f in PERMITTED_FIELDS and f != "customer_id"]
    return (f"SELECT {', '.join(columns)}\n"
            f"FROM {customers_ref}\n"
            f"WHERE customer_id = @customer_id\n"
            f"LIMIT 10")


def mask_email(email) -> str:
    if not email or "@" not in str(email):
        return "(no valid email on record)"
    local, domain = str(email).split("@", 1)
    return f"{local[:1]}***@{domain}"


def lookup_customers(runner, settings, dataset: str, request: LookupRequest, reveal_email: bool = False) -> LookupResult:
    fields = [] if request.existence_only else request.fields
    sql = lookup_sql(settings.refs(dataset)["customers"], fields)
    records = []
    for customer_id in request.customer_ids:
        res = runner.run(sql, [SqlParam("customer_id", "STRING", customer_id)], max_rows=10,
                         allowed_tables=settings.allowed_tables(dataset), dry_run_first=False)
        if not res.ok:
            records.append(CustomerRecord(customer_id, "error", error=f"{res.error_kind}: {res.error_message}"))
            continue
        if not res.rows:
            records.append(CustomerRecord(customer_id, "not_found"))
            continue
        values = {}
        conflicts = []
        for column in res.rows[0]:
            distinct = {str(row[column]) for row in res.rows}
            if len(distinct) > 1:
                conflicts.append(column)
            value = res.rows[0][column]
            if column in SENSITIVE_FIELDS and not reveal_email:
                value = mask_email(value)
            values[column] = str(value) if value is not None else None
        status = "conflicting" if conflicts else "found"
        records.append(CustomerRecord(customer_id, status, values, len(res.rows), conflicts))
    return LookupResult(request, records, sql, reveal_email)


# ----------------------------------------------------------------------------- answer

FIELD_LABELS = {"customer_id": "Customer ID", "email": "Email", "region": "Region", "country_code": "Country",
                "customer_segment": "Segment", "signup_date": "Signup date", "created_at": "Record created"}


def format_lookup_answer(result: LookupResult) -> str:
    lines = []
    for record in result.records:
        cid = record.customer_id
        if record.status == "error":
            lines.append(f"I could not look up {cid}: the query failed ({record.error.split(':')[0]}). "
                         "This is not the same as the customer not existing.")
        elif record.status == "not_found":
            lines.append(f"Customer {cid} does not exist in the customers table.")
        elif result.request.existence_only:
            lines.append(f"Yes, customer {cid} exists.")
        else:
            details = [f"{FIELD_LABELS.get(k, k)}: {v if v is not None else '(missing)'}"
                       for k, v in record.values.items() if k != "customer_id"]
            lines.append(f"Customer {cid} — " + "; ".join(details) + ".")
            if record.status == "conflicting":
                lines.append(f"Data-quality warning: {record.record_count} records exist for {cid} and they disagree on "
                             f"{', '.join(record.conflicting_fields)}; the first record is shown. This is a data-quality "
                             "issue (duplicate customer ID).")
            elif record.record_count > 1:
                lines.append(f"Note: {record.record_count} records exist for {cid} (duplicate customer ID); they agree on "
                             "the fields shown, but may differ in others. This is a data-quality issue.")
    if any("email" in r.values for r in result.records) and not result.email_revealed:
        lines.append("Email addresses are masked by policy. An authorised user can enable full emails with "
                     "CUSTOMER_LOOKUP_REVEAL_EMAIL=true.")
    if result.request.source == "conversation":
        lines.append(f"(Customer taken from the conversation: {', '.join(result.request.customer_ids)}.)")
    return "\n".join(lines)