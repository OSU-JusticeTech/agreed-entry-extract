"""
Post-OCR text cleaning.

Applies conservative fixes to OCR output:

1. Fix common OCR character substitutions in dollar amounts (O→0, l→1, etc.)
2. Validate dates and flag impossible ones (e.g. November 31)
3. Re-order payment schedule lines chronologically
4. Normalize whitespace and page markers

All changes are logged and returned so they can be audited.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _is_valid_date(month: int, day: int, year: Optional[int] = None) -> bool:
    if month < 1 or month > 12:
        return False
    max_days = DAYS_IN_MONTH[month]
    if month == 2 and year and _is_leap_year(year):
        max_days = 29
    return 1 <= day <= max_days


def _validate_dates(text: str) -> tuple[str, list[str]]:
    """Find and flag invalid dates in *text*."""
    changes: list[str] = []

    # Pattern: "Month DD, YYYY"
    def _check_text_date(match: re.Match) -> str:
        month_name = match.group(1).lower()
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        month_num = MONTH_NAMES.get(month_name)
        if month_num and not _is_valid_date(month_num, day, year):
            changes.append(f"Invalid date: {match.group(0)}")
            return f"{match.group(0)} [INVALID DATE]"
        return match.group(0)

    text = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b",
        _check_text_date,
        text,
        flags=re.IGNORECASE,
    )

    # Pattern: "MM/DD/YYYY" or "MM-DD-YYYY"
    def _check_numeric_date(match: re.Match) -> str:
        month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        if not _is_valid_date(month, day, year):
            changes.append(f"Invalid date: {match.group(0)}")
            return f"{match.group(0)} [INVALID DATE]"
        return match.group(0)

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", _check_numeric_date, text)

    return text, changes


# ---------------------------------------------------------------------------
# OCR number fixes
# ---------------------------------------------------------------------------

_OCR_DIGIT_MAP = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2", "g": "9", "q": "9"}


def _fix_ocr_numbers(text: str) -> tuple[str, list[str]]:
    """Fix common OCR character errors in money amounts."""
    changes: list[str] = []

    def _fix_money(match: re.Match) -> str:
        original = match.group(0)
        numeric = original[1:]  # after $
        fixed = numeric
        for wrong, right in _OCR_DIGIT_MAP.items():
            fixed = fixed.replace(wrong, right)
        if fixed != numeric:
            changes.append(f"Fixed money: {original} -> ${fixed}")
            return "$" + fixed
        return original

    text = re.sub(r"\$[\d,.\w]+", _fix_money, text)

    def _fix_decimals(match: re.Match) -> str:
        original = match.group(0)
        fixed = original
        for wrong in ("O", "o", "l", "I"):
            fixed = fixed.replace(wrong, _OCR_DIGIT_MAP[wrong])
        if fixed != original:
            changes.append(f"Fixed number: {original} -> {fixed}")
        return fixed

    text = re.sub(r"\b[\d]{1,3}[,.]?[\dOoIl]{2,3}\.[\dOoIl]{2}\b", _fix_decimals, text)

    return text, changes


# ---------------------------------------------------------------------------
# Payment plan ordering
# ---------------------------------------------------------------------------

def _order_payment_plans(text: str) -> tuple[str, list[str]]:
    """Ensure payment schedule lines are in chronological order."""
    changes: list[str] = []

    def _parse_date(snippet: str) -> Optional[datetime]:
        m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", snippet)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 2000 if year < 50 else 1900
            try:
                return datetime(year, month, day)
            except ValueError:
                return None

        m = re.search(
            r"(January|February|March|April|May|June|July|August|September"
            r"|October|November|December)\s+(\d{1,2}),?\s*(\d{4})",
            snippet,
            re.IGNORECASE,
        )
        if m:
            month_num = MONTH_NAMES.get(m.group(1).lower())
            if month_num:
                try:
                    return datetime(int(m.group(3)), month_num, int(m.group(2)))
                except ValueError:
                    return None
        return None

    schedule_patterns = [
        r"(Payment\s+Schedule|Payment\s+Plan|Installment\s+Schedule)[:\s]*\n((?:.*\n)*?)(?=\n\n|\Z)",
    ]

    for pattern in schedule_patterns:
        def _sort_block(match: re.Match) -> str:
            header = match.group(1)
            content = match.group(2)
            lines = [l for l in content.strip().split("\n") if l.strip()]
            dated = [(d, l) for l in lines if (d := _parse_date(l)) is not None]
            undated = [l for l in lines if _parse_date(l) is None]
            if dated:
                dates_only = [d for d, _ in dated]
                if dates_only != sorted(dates_only):
                    changes.append("Reordered payment schedule chronologically")
                    dated.sort(key=lambda x: x[0])
            sorted_lines = [l for _, l in dated] + undated
            return header + "\n" + "\n".join(sorted_lines)

        text = re.sub(pattern, _sort_block, text, flags=re.IGNORECASE)

    return text, changes


# ---------------------------------------------------------------------------
# General cleanup
# ---------------------------------------------------------------------------

def _general_cleanup(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\[--- PAGE (\d+) START ---\]", r"\n\n=== PAGE \1 ===\n\n", text)
    text = re.sub(r"\[--- PAGE (\d+) END ---\]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> tuple[str, list[str]]:
    """
    Apply all cleaning steps to OCR output.

    Parameters
    ----------
    text : str
        Raw OCR text (possibly multi-page with page markers).

    Returns
    -------
    tuple[str, list[str]]
        Cleaned text and a list of human-readable change descriptions.
    """
    all_changes: list[str] = []

    text, changes = _fix_ocr_numbers(text)
    all_changes.extend(changes)

    text, changes = _validate_dates(text)
    all_changes.extend(changes)

    text, changes = _order_payment_plans(text)
    all_changes.extend(changes)

    text = _general_cleanup(text)

    return text, all_changes
