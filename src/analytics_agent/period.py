"""Deterministic resolution of time periods in questions.

Relative periods are resolved against the data snapshot, never by the model alone:
the last complete month is the month before the snapshot (as_of 2026-07-03 -> June 2026).

    resolve_period("monthly net sales over the last six months")
    -> Period(start=2026-01-01, end=2026-06-30, label="last 6 complete months")

Supported: explicit ISO dates/ranges, "last/past/previous N months|quarters|weeks|days",
"last month/quarter/year", "this year", "YTD", "Q1 2026", "first quarter of 2026",
"H1 2026", "first/second half of 2025", "June 2026", "in 2025".
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

AS_OF = date(2026, 7, 3)
WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "a": 1,
    "an": 1,
}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {
    m.lower(): i for i, m in enumerate(calendar.month_abbr) if m
}
ORDINAL_Q = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
}


@dataclass(frozen=True)
class Period:
    """Inclusive calendar date range resolved from a natural-language question.

    Attributes:
        start: Inclusive start date.
        end: Inclusive end date.
        label: Human-readable description of the period (e.g.
            ``"last 6 complete months"``, ``"Q2 2026"``).
    """

    start: date
    end: date
    label: str

    @property
    def start_iso(self) -> str:
        """Return the start date as an ISO-8601 string (``YYYY-MM-DD``)."""
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        """Return the end date as an ISO-8601 string (``YYYY-MM-DD``)."""
        return self.end.isoformat()


def _month_start(y: int, m: int) -> date:
    """Return the first day of month *m* in year *y*."""
    return date(y, m, 1)


def _month_end(y: int, m: int) -> date:
    """Return the last day of month *m* in year *y*."""
    return date(y, m, calendar.monthrange(y, m)[1])


def _shift_month(y: int, m: int, delta: int) -> tuple[int, int]:
    """Shift a year/month pair by *delta* months.

    Args:
        y: Starting year.
        m: Starting month (1–12).
        delta: Number of months to add (may be negative).

    Returns:
        A ``(year, month)`` pair after the shift.
    """
    idx = y * 12 + (m - 1) + delta
    return idx // 12, idx % 12 + 1


def _num(token: str) -> Optional[int]:
    """Parse a digit string or English word-number into an integer.

    Args:
        token: A digit string (``"6"``) or word (``"six"``, ``"a"``).

    Returns:
        The corresponding integer, or ``None`` when the token is not recognised.
    """
    token = token.lower()
    return int(token) if token.isdigit() else WORD_NUMBERS.get(token)


def resolve_period(text: str, as_of: date = AS_OF) -> Optional[Period]:
    """Resolve the time period a question refers to.

    Relative expressions (``"last six months"``, ``"YTD"``, ``"last quarter"``,
    …) are computed against *as_of* so the model never invents the window.
    Explicit ISO dates, named months, quarters and halves are also recognised.

    Args:
        text: Natural-language question (or fragment) that may name a period.
        as_of: Snapshot date used as the reference for relative periods.
            Defaults to the project's data snapshot (``2026-07-03``).

    Returns:
        A ``Period`` when a period is detected, otherwise ``None``.
    """
    if not text:
        return None
    t = " " + text.lower().replace(",", " ") + " "
    last_m_y, last_m = _shift_month(as_of.year, as_of.month, -1)  # last complete month
    q_now = (as_of.month - 1) // 3 + 1
    last_q_y, last_q = (
        (as_of.year, q_now - 1) if q_now > 1 else (as_of.year - 1, 4)
    )

    # explicit ISO dates
    iso = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if len(iso) >= 2:
        a, b = sorted(date.fromisoformat(x) for x in iso[:2])
        return Period(a, b, f"{a} to {b}")
    if len(iso) == 1 and re.search(r"\b(on|for)\s+\d{4}-\d{2}-\d{2}", t):
        d = date.fromisoformat(iso[0])
        return Period(d, d, str(d))

    n_word = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    m = re.search(
        rf"\b(?:last|past|previous|prior|trailing)\s+{n_word}\s+"
        rf"(months?|quarters?|weeks?|days?)\b",
        t,
    )
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        if unit.startswith("month"):
            sy, sm = _shift_month(last_m_y, last_m, -(n - 1))
            return Period(
                _month_start(sy, sm),
                _month_end(last_m_y, last_m),
                f"last {n} complete months",
            )
        if unit.startswith("quarter"):
            ey, eq = last_q_y, last_q
            sy, sm = _shift_month(ey, (eq - 1) * 3 + 1, -3 * (n - 1))
            return Period(
                _month_start(sy, sm),
                _month_end(ey, eq * 3),
                f"last {n} complete quarters",
            )
        days = n * 7 if unit.startswith("week") else n
        end = as_of - timedelta(days=1)
        return Period(
            end - timedelta(days=days - 1),
            end,
            f"last {days} days before {as_of}",
        )

    if re.search(r"\b(last|previous|prior)\s+month\b", t):
        return Period(
            _month_start(last_m_y, last_m),
            _month_end(last_m_y, last_m),
            f"last complete month ({calendar.month_name[last_m]} {last_m_y})",
        )
    if re.search(r"\b(last|previous|prior)\s+quarter\b", t):
        return Period(
            _month_start(last_q_y, (last_q - 1) * 3 + 1),
            _month_end(last_q_y, last_q * 3),
            f"last complete quarter (Q{last_q} {last_q_y})",
        )
    if re.search(r"\b(this year|year to date|year-to-date|ytd)\b", t):
        return Period(
            date(as_of.year, 1, 1),
            _month_end(last_m_y, last_m),
            f"year to date {as_of.year} (complete months)",
        )
    if re.search(r"\b(last|previous|prior)\s+year\b", t):
        y = as_of.year - 1
        return Period(date(y, 1, 1), date(y, 12, 31), f"calendar year {y}")

    m = re.search(r"\bq([1-4])\s*(?:of\s+)?(\d{4})\b", t) or re.search(
        r"\b(\d{4})\s*q([1-4])\b", t
    )
    if m:
        a, b = m.groups()
        q, y = (int(a), int(b)) if len(a) == 1 else (int(b), int(a))
        return Period(
            _month_start(y, (q - 1) * 3 + 1),
            _month_end(y, q * 3),
            f"Q{q} {y}",
        )
    m = re.search(
        r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+"
        r"(?:of\s+)?(\d{4})\b",
        t,
    )
    if m:
        q, y = ORDINAL_Q[m.group(1)], int(m.group(2))
        return Period(
            _month_start(y, (q - 1) * 3 + 1),
            _month_end(y, q * 3),
            f"Q{q} {y}",
        )
    m = re.search(r"\bh([12])\s*(?:of\s+)?(\d{4})\b", t) or re.search(
        r"\b(first|second)\s+half\s+(?:of\s+)?(\d{4})\b", t
    )
    if m:
        h = 1 if m.group(1) in ("1", "first") else 2
        y = int(m.group(2))
        return Period(
            date(y, 1 if h == 1 else 7, 1),
            date(y, 6, 30) if h == 1 else date(y, 12, 31),
            f"H{h} {y}",
        )
    # several months mentioned ("May 2026 and June 2026", "May and June 2026",
    # "between March and June 2026"): cover all of them; a month without a year
    # takes the next year mentioned after it.
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    mentions = list(
        re.finditer(rf"\b({month_names})\b\.?(?:\s+(\d{{4}}))?", t)
    )
    if len(mentions) >= 2:
        resolved, pending = [], []
        for mm in mentions:
            pending.append(MONTHS[mm.group(1)])
            if mm.group(2):
                resolved += [(int(mm.group(2)), mo) for mo in pending]
                pending = []
        if len(resolved) >= 2 and not pending:
            (y1, m1), (y2, m2) = min(resolved), max(resolved)
            return Period(
                _month_start(y1, m1),
                _month_end(y2, m2),
                f"{calendar.month_name[m1]} {y1} to "
                f"{calendar.month_name[m2]} {y2}",
            )
    m = re.search(
        r"\b("
        + "|".join(sorted(MONTHS, key=len, reverse=True))
        + r")\.?\s+(\d{4})\b",
        t,
    )
    if m:
        mo, y = MONTHS[m.group(1)], int(m.group(2))
        return Period(
            _month_start(y, mo),
            _month_end(y, mo),
            f"{calendar.month_name[mo]} {y}",
        )
    m = re.search(r"\b(?:in|for|during|of)\s+(20\d{2})\b", t)
    if m:
        y = int(m.group(1))
        return Period(date(y, 1, 1), date(y, 12, 31), f"calendar year {y}")
    return None


def parse_period_label(period: Optional[str]) -> Optional[Period]:
    """Parse a stored period label of the form ``YYYY-MM-DD..YYYY-MM-DD``.

    Used to rehydrate a period previously written into conversation history.

    Args:
        period: Label string, or ``None``.

    Returns:
        A ``Period`` when the label is well-formed, otherwise ``None``.
    """
    if not period or ".." not in period:
        return None
    a, b = period.split("..", 1)
    try:
        return Period(
            date.fromisoformat(a), date.fromisoformat(b), f"{a} to {b}"
        )
    except ValueError:
        return None