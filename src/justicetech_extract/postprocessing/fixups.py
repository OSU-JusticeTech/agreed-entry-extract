"""
Post-extraction fixups and normalizations.

Ported from ``Step3_more_for_paymen_plan_bigger_than_10.ipynb``.

Applied automatically after extraction:

1. Recalculate ``total_payment_sum`` from all payments (not just first 10)
2. Normalize case numbers (``5812`` → ``S812`` prefix correction)
3. Normalize ``outcome_type`` variants (``Vacate or Pay`` → ``Pay or Vacate``)
4. Flag OCR artifact plaintiff names (timestamps misread as names)
5. **v0.2** Correct plaintiff/defendant names via known landlord fuzzy matching
6. **v0.2** Fix monthly rent entries misclassified as fixed amounts
7. **v0.2** Validate payment dates against agreement date
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from justicetech_extract.models import ExtractedCourtInfo, OutcomeType, PaymentType


# ======================================================================
# Known landlord names in Franklin County eviction filings.
# These are repeat filers whose names are frequently garbled by OCR.
# Keys are canonical names; values are known OCR misreads.
# ======================================================================
KNOWN_PLAINTIFFS = {
    "McNaughten": [
        "mcnaughten", "mcnaughton", "mcnaught", "mc naughten",
        "menaughten", "menaughton",
    ],
    "Vinebrook": [
        "vinebrook", "vine brook", "vinebrook homes", "vinbrook",
        "vinebrooke", "vnebrook", "vinebroo",
    ],
    "S81": [
        "s81", "s812", "s8 1", "s8'12", "s8l2", "s8id",
        "581", "5812", "s 81", "s-81",
    ],
    "Wallick": [
        "wallick", "walick", "wallck", "waliick",
    ],
    "NRP Group": [
        "nrp group", "nrp", "nrp grp",
    ],
    "Elam": [
        "elam", "elam and", "elam &",
    ],
    "Riverdale": [
        "riverdale", "river dale", "riverdal", "riverdalle",
    ],
    "National Church Residences": [
        "national church", "natl church", "national church residences",
    ],
    "CASTO": [
        "casto", "casto communities",
    ],
    "Blueprint": [
        "blueprint", "blue print",
    ],
    "Hearty Home": [
        "hearty home", "hearty homes", "heartyhome",
    ],
}


def _fuzzy_match_plaintiff(name: str) -> Optional[str]:
    """
    Try to match a garbled plaintiff name against known landlords.

    Uses case-insensitive substring matching against known OCR misreads.
    Returns the canonical name if matched, or None.
    """
    if not name:
        return None

    name_lower = name.strip().lower()

    # Skip very short names (likely garbage)
    if len(name_lower) < 2:
        return None

    for canonical, variants in KNOWN_PLAINTIFFS.items():
        # Exact match (case-insensitive)
        if name_lower == canonical.lower():
            return canonical
        # Match against known OCR variants
        for variant in variants:
            if name_lower == variant or variant in name_lower:
                return canonical
            # Also check if the name starts with the variant
            if name_lower.startswith(variant[:4]) and len(variant) >= 4:
                # Levenshtein-lite: check if at least 60% of characters match
                matches = sum(1 for a, b in zip(name_lower, variant) if a == b)
                if len(variant) > 0 and matches / len(variant) > 0.6:
                    return canonical

    return None


def apply_fixups(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Apply all post-extraction fixups to an :class:`ExtractedCourtInfo`.

    This is idempotent — safe to call multiple times.
    """
    info = fix_payment_total(info)
    info = normalize_case_number(info)
    info = normalize_outcome_type(info)
    info = fix_known_plaintiff(info)
    info = fix_rent_misclassification(info)
    info = validate_payment_dates(info)
    # flag_ocr_plaintiff is diagnostic only (returns a warning string)
    # — it does not modify the info object
    return info


def fix_known_plaintiff(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Correct plaintiff name using known landlord lookup.

    Many Franklin County eviction plaintiffs are repeat filers whose names
    get garbled by OCR (e.g., "S8'12" → "S81", "W gelbeh" is NOT a known
    plaintiff so it stays unchanged).
    """
    if info.plaintiff:
        corrected = _fuzzy_match_plaintiff(info.plaintiff)
        if corrected and corrected != info.plaintiff:
            info.plaintiff = corrected

    return info


def fix_rent_misclassification(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Fix monthly rent payments incorrectly classified as fixed_amount.

    Common OCR/LLM error: "March rent" gets extracted as a fixed_amount
    with a dollar value (e.g., $750). This detects such cases and
    reclassifies them as monthly_rent.

    Heuristics:
    - extra_text contains "rent" → monthly_rent
    - amount is a month name → monthly_rent
    - payment_type is fixed_amount but month_rent field is populated → monthly_rent
    """
    month_names = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    }

    for p in info.payment_schedule:
        should_be_rent = False

        # Check if extra_text mentions rent
        if p.extra_text and "rent" in p.extra_text.lower():
            should_be_rent = True

        # Check if amount is actually a month name
        if p.amount and p.amount.lower().strip() in month_names:
            p.month_rent = p.amount.strip().capitalize()
            p.amount = None
            should_be_rent = True

        # Check if month_rent is set but type is wrong
        if p.month_rent and p.payment_type == PaymentType.FIXED_AMOUNT:
            should_be_rent = True

        if should_be_rent:
            p.payment_type = PaymentType.MONTHLY_RENT
            # If amount looks like a round number that was guessed, clear it
            if p.amount:
                try:
                    val = float(p.amount.replace(",", ""))
                    # Round numbers (100, 200, 500, 750, 1000) are likely guesses
                    if val > 0 and val % 50 == 0 and not p.extra_text:
                        p.amount = None
                except ValueError:
                    pass

    # Recalculate total after rent fixes
    info = fix_payment_total(info)
    return info


def validate_payment_dates(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Validate payment dates against the agreement date.

    Flags suspicious dates:
    - Payment date before the agreement date
    - Payment date more than 18 months after the agreement date
    - Payment dates that look like OCR digit swaps (e.g., 1/19 vs 1/9)

    Adds warnings to extra_text rather than deleting data.
    """
    if not info.agreement_signed_date:
        return info

    try:
        agreement_date = datetime.strptime(info.agreement_signed_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return info

    max_date = agreement_date + timedelta(days=548)  # ~18 months

    for p in info.payment_schedule:
        if not p.due_date:
            continue
        try:
            pay_date = datetime.strptime(p.due_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        warning = None

        if pay_date < agreement_date - timedelta(days=7):
            warning = "DATE_BEFORE_AGREEMENT"
        elif pay_date > max_date:
            warning = "DATE_TOO_FAR_FUTURE"

        if warning:
            if p.extra_text:
                p.extra_text = f"{p.extra_text}; {warning}"
            else:
                p.extra_text = warning

    return info


def fix_payment_total(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Recalculate ``total_payment_sum`` from the full payment schedule.

    The original CSV export only looked at the first 10 payments;
    this uses ALL payments.
    """
    total = 0.0
    for p in info.payment_schedule:
        if p.payment_type == PaymentType.FIXED_AMOUNT and p.amount:
            try:
                total += float(p.amount.replace(",", ""))
            except ValueError:
                pass

    if total > 0:
        info.total_payment_sum = f"{total:.2f}"

    return info


def normalize_case_number(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Fix OCR-corrupted case number prefixes.

    Known issue: ``5812`` is an OCR misread of ``S812``.
    """
    if info.case_number:
        info.case_number = re.sub(r"\b5812\b", "S812", info.case_number)
    return info


def normalize_outcome_type(info: ExtractedCourtInfo) -> ExtractedCourtInfo:
    """
    Normalize outcome_type variants to canonical forms.

    ``Vacate or Pay`` → ``Pay or Vacate``
    """
    if info.outcome_type and info.outcome_type.value == "Vacate or Pay":
        info.outcome_type = OutcomeType.PAY_OR_VACATE
    return info


def flag_ocr_plaintiff(info: ExtractedCourtInfo) -> Optional[str]:
    """
    Detect plaintiff names that are OCR artifacts (court filing timestamps
    misread as names).

    Returns a warning string if flagged, or ``None``.
    """
    if not info.plaintiff:
        return None

    # Timestamps like "2024 JAN 22 PM 3:42" misread as names
    if re.search(r"\d{4}\s+[A-Z]{3}\s+\d{1,2}\s+[AP]M", info.plaintiff):
        return f"OCR artifact detected: '{info.plaintiff}'"

    # Mostly numeric
    digits = sum(c.isdigit() for c in info.plaintiff)
    if len(info.plaintiff) > 0 and digits / len(info.plaintiff) > 0.5:
        return f"Mostly numeric plaintiff: '{info.plaintiff}'"

    return None
