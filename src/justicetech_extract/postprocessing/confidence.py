"""
Per-row confidence scoring for extracted court documents.

Computes a 0.00-1.00 confidence score based on six signals:

1. Method agreement (25%) -- Do LLM and regex agree on outcome_type?
2. Field completeness (15%) -- Are all expected fields populated?
3. Filename consistency (5%) -- Do extracted IDs match the filename?
4. Payment validity (20%) -- Are payment entries well-formed?
5. OCR quality (15%) -- How clean was the OCR text?
6. Name plausibility (20%) -- Do plaintiff/defendant names look correct?

The combined score is mapped to a label: HIGH (>=0.85), MEDIUM (>=0.60), LOW (<0.60).


"""

from __future__ import annotations

import logging
import re
from typing import Optional

from justicetech_extract.models import (
    ExtractedCourtInfo,
    OutcomeType,
    PaymentType,
    PipelineResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
W_METHOD_AGREEMENT = 0.25
W_FIELD_COMPLETENESS = 0.15
W_FILENAME_CONSISTENCY = 0.05
W_PAYMENT_VALIDITY = 0.20
W_OCR_QUALITY = 0.15
W_NAME_PLAUSIBILITY = 0.20


# ---------------------------------------------------------------------------
# Entity indicators for plaintiff name analysis
# ---------------------------------------------------------------------------
# Words that suggest the plaintiff is a business/institutional entity.
# Used in swap detection: a single-word plaintiff WITHOUT any of these
# combined with an empty defendant is a strong swap signal.
ENTITY_INDICATORS = frozenset({
    "llc", "inc", "corp", "ltd", "lp", "lc",
    "group", "properties", "property", "prop",
    "realty", "real estate", "real",
    "rentals", "rental",
    "apartments", "apts", "apt",
    "housing", "homes", "home",
    "management", "mgmt",
    "investments", "investment",
    "associates", "company", "co.",
    "partners", "capital",
    "solutions", "services",
    "trust", "foundation",
    "church", "residences", "communities",
    "dba", "d/b/a",
    "estate", "estates",
})


def _has_entity_indicator(name):
    """Check if a name contains business/entity keywords."""
    name_lower = name.lower()
    return any(indicator in name_lower for indicator in ENTITY_INDICATORS)


# ---------------------------------------------------------------------------
# Signal 1: Extraction Method Agreement
# ---------------------------------------------------------------------------

def _score_method_agreement(result):
    """
    Score based on whether LLM and regex extraction agreed.

    Returns (score, reason).
    """
    method = result.info.extraction_method or ""

    if "LLM+Regex" in method:
        # Both ran -- check for override warnings in errors
        had_override = any("overriding" in e.lower() for e in result.errors)
        if had_override:
            return 0.6, "LLM+Regex ran but outcome was overridden (disagreement resolved)"
        return 1.0, "LLM and Regex both ran and agreed"
    elif "Regex-only" in method:
        return 0.4, "Regex-only (no LLM cross-validation)"
    else:
        return 0.3, "Single extraction method: " + method


# ---------------------------------------------------------------------------
# Signal 2: Field Completeness
# ---------------------------------------------------------------------------

def _score_field_completeness(info):
    """
    Score based on how many expected fields are populated.

    Core fields: case_number, plaintiff, defendant, outcome_type.
    Conditional: payments (for Pay-and-Stay), vacate_date (for Vacate-Only).
    """
    missing = []

    if not info.case_number:
        missing.append("case_number")
    if not info.plaintiff:
        missing.append("plaintiff")
    if not info.defendant:
        missing.append("defendant")
    if not info.outcome_type:
        missing.append("outcome_type")

    outcome = (info.outcome_type.value if info.outcome_type else "").lower()

    if "pay" in outcome:
        if len(info.payment_schedule) == 0:
            missing.append("payment_schedule (expected for Pay outcome)")
    elif "vacate" in outcome:
        if not info.mandatory_vacate_date:
            missing.append("mandatory_vacate_date (expected for Vacate outcome)")

    total_checks = 4 + 1  # 4 core + 1 conditional
    populated = total_checks - len(missing)
    score = max(0.1, populated / total_checks)

    if missing:
        reason = "Missing: " + ", ".join(missing)
    else:
        reason = "All expected fields populated"

    return round(score, 2), reason


# ---------------------------------------------------------------------------
# Signal 3: Filename Consistency
# ---------------------------------------------------------------------------

def _score_filename_consistency(info):
    """
    Score based on whether filename-derived fields match extraction.

    The filename contains ground-truth case number and date.
    """
    if not info.filename:
        return 0.5, "No filename available for validation"

    checks_passed = 0
    checks_total = 0
    issues = []

    if info.case_number:
        checks_total += 1
        cn_digits = re.sub(r"[^0-9]", "", info.case_number)
        fn_digits = re.sub(r"[^0-9]", "", info.filename)
        if cn_digits and cn_digits in fn_digits:
            checks_passed += 1
        else:
            issues.append("case_number not found in filename")

    if info.agreement_signed_date:
        checks_total += 1
        date_digits = re.sub(r"[^0-9]", "", info.agreement_signed_date)
        fn_clean = info.filename.replace("_", " ").replace("-", " ")
        if date_digits[:4] in fn_clean.replace(" ", "") or date_digits in fn_clean.replace(" ", ""):
            checks_passed += 1
        else:
            issues.append("agreement_date not matched in filename")

    if checks_total == 0:
        return 0.5, "No fields to validate against filename"

    score = checks_passed / checks_total
    reason = "Filename matches" if not issues else "; ".join(issues)
    return round(score, 2), reason


# ---------------------------------------------------------------------------
# Signal 4: Payment Schedule Validity
# ---------------------------------------------------------------------------

def _score_payment_validity(info):
    """
    Score based on payment schedule quality.

    Checks: completeness of individual entries, consistency of types,
    whether a pay-type outcome actually has payments, and whether any
    payment dates were flagged as anomalous by the fixups module.
    """
    outcome = (info.outcome_type.value if info.outcome_type else "").lower()
    payments = info.payment_schedule

    # Vacate-only with no payments is expected and correct
    if "vacate" in outcome and "pay" not in outcome and len(payments) == 0:
        return 1.0, "Vacate-only with no payments (expected)"

    # Pay-type with zero payments is suspicious
    if "pay" in outcome and len(payments) == 0:
        return 0.2, "Pay-type outcome but no payments extracted"

    if len(payments) == 0:
        return 0.7, "No payments and no pay-type outcome"

    # Score individual payment quality
    well_formed = 0
    partial = 0
    date_warnings = 0
    issues = []

    for p in payments:
        has_amount = p.amount is not None or p.month_rent is not None
        has_date = p.due_date is not None

        # Check for date anomaly flags inserted by validate_payment_dates()
        extra = p.extra_text or ""
        has_date_warning = (
            "DATE_BEFORE_AGREEMENT" in extra
            or "DATE_TOO_FAR_FUTURE" in extra
        )

        if has_date_warning:
            # Flagged payment dates count as partial regardless of completeness
            partial += 1
            date_warnings += 1
        elif has_amount and has_date:
            well_formed += 1
        elif has_amount or has_date:
            partial += 1

    total = len(payments)
    if total == 0:
        return 0.5, "Payment schedule present but empty after filtering"

    quality = (well_formed + 0.5 * partial) / total

    if date_warnings > 0:
        issues.append(str(date_warnings) + " payment(s) with date anomaly")
    if partial - date_warnings > 0:
        issues.append(str(partial - date_warnings) + " payment(s) missing date or amount")
    if well_formed == total:
        issues.append("All payments have amount and date")

    return round(quality, 2), "; ".join(issues) if issues else "Payments present"


# ---------------------------------------------------------------------------
# Signal 5: OCR Text Quality
# ---------------------------------------------------------------------------

def _score_ocr_quality(result):
    """
    Score based on OCR text quality indicators.

    Checks: number of cleaning changes, presence of [?] markers,
    text length, and ratio of non-ASCII / garbled content.
    """
    text = result.cleaned_text or result.ocr_text or ""
    changes = result.ocr_changes or []
    issues = []

    if not text or len(text) < 50:
        return 0.1, "OCR text is empty or extremely short"

    score = 1.0

    change_count = len(changes)
    if change_count > 20:
        score -= 0.3
        issues.append(str(change_count) + " OCR corrections applied")
    elif change_count > 10:
        score -= 0.15
        issues.append(str(change_count) + " OCR corrections applied")

    uncertain_count = text.count("[?]")
    if uncertain_count > 5:
        score -= 0.3
        issues.append(str(uncertain_count) + " uncertain [?] markers")
    elif uncertain_count > 2:
        score -= 0.15
        issues.append(str(uncertain_count) + " uncertain [?] markers")

    if len(text) < 200:
        score -= 0.2
        issues.append("Short OCR text (" + str(len(text)) + " chars)")

    artifact_warnings = [e for e in result.errors if "ocr" in e.lower() or "artifact" in e.lower()]
    if artifact_warnings:
        score -= 0.2
        issues.append("OCR artifact warning flagged")

    score = max(0.1, score)
    reason = "; ".join(issues) if issues else "Clean OCR text"
    return round(score, 2), reason


# ---------------------------------------------------------------------------
# Signal 6: Name Plausibility
# ---------------------------------------------------------------------------

def _score_name_plausibility(info):
    """
    Score based on whether plaintiff and defendant names look plausible.

    Checks for:
    - Label-word detection: If plaintiff or defendant field contains a form
      label like 'Plaintiff' or 'Defendant(s)' instead of an actual name,
      the extraction grabbed the wrong text from the document layout.
    - Swap detection: Single-word non-entity plaintiff + empty defendant
      is a strong signal that plaintiff/defendant names were swapped during
      extraction (the defendant last name ended up in the plaintiff field).
    - Empty names: Missing plaintiff or defendant.
    - Very short names: 3 or fewer characters suggests OCR garbling.
    - Single-word defendant: Common in handwritten forms (only last name
      written), but reduces verifiability.
    """
    score = 1.0
    issues = []

    p = (info.plaintiff or "").strip()
    d = (info.defendant or "").strip()

    # --- Label-word detection (highest priority) ---
    # If the plaintiff or defendant field contains a form label instead of
    # an actual name, the extraction grabbed the wrong text.  Common on
    # handwritten fill-in-the-blank forms where the regex picks up
    # "Plaintiff," or "Defendant(s)" as the name itself.
    _LABEL_WORDS = {
        "plaintiff", "plaintiffs", "plaintiff(s)", "plaintiff,",
        "defendant", "defendants", "defendant(s)", "defendant,",
        "v.", "vs.", "vs",
    }
    p_lower = p.lower().rstrip(".,:()")
    d_lower = d.lower().rstrip(".,:()")

    label_issues = []
    if p_lower in _LABEL_WORDS:
        label_issues.append("plaintiff is a form label ('%s'), not an actual name" % p)
    if d_lower in _LABEL_WORDS:
        label_issues.append("defendant is a form label ('%s'), not an actual name" % d)
    if label_issues:
        return 0.1, "; ".join(label_issues)

    # --- Swap detection ---
    # A single-word plaintiff with no entity indicators AND an empty defendant
    # is a strong signal of extraction swap: the defendant last name was
    # placed in the plaintiff field, and the actual plaintiff was missed.
    if (
        p
        and not d
        and " " not in p
        and not _has_entity_indicator(p)
    ):
        return 0.2, "Likely plaintiff/defendant swap: single-word non-entity plaintiff with empty defendant"

    # --- Plaintiff checks ---
    if not p:
        score -= 0.4
        issues.append("plaintiff is empty")
    elif len(p) <= 3:
        score -= 0.15
        issues.append("plaintiff very short (" + str(len(p)) + " chars)")

    # --- Defendant checks ---
    if not d:
        score -= 0.4
        issues.append("defendant is empty")
    elif len(d) <= 3:
        score -= 0.15
        issues.append("defendant very short (" + str(len(d)) + " chars)")
    elif " " not in d and "," not in d:
        # Single-word defendant: common in handwritten forms where only the
        # last name is written. Correct extraction, but incomplete info.
        score -= 0.1
        issues.append("defendant is single word (possibly incomplete)")

    score = max(0.1, round(score, 2))
    reason = "; ".join(issues) if issues else "Names look plausible"
    return score, reason


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

def compute_confidence(result):
    """
    Compute a per-row confidence score for an extraction result.

    Parameters
    ----------
    result : PipelineResult
        The full pipeline result with extraction output and diagnostics.

    Returns
    -------
    dict
        Keys: confidence_score (float 0.00-1.00),
        confidence_label (str: HIGH/MEDIUM/LOW),
        confidence_details (str: breakdown of signal scores).
    """
    s1, r1 = _score_method_agreement(result)
    s2, r2 = _score_field_completeness(result.info)
    s3, r3 = _score_filename_consistency(result.info)
    s4, r4 = _score_payment_validity(result.info)
    s5, r5 = _score_ocr_quality(result)
    s6, r6 = _score_name_plausibility(result.info)

    combined = (
        W_METHOD_AGREEMENT * s1
        + W_FIELD_COMPLETENESS * s2
        + W_FILENAME_CONSISTENCY * s3
        + W_PAYMENT_VALIDITY * s4
        + W_OCR_QUALITY * s5
        + W_NAME_PLAUSIBILITY * s6
    )
    combined = round(combined, 2)

    if combined >= 0.85:
        label = "HIGH"
    elif combined >= 0.60:
        label = "MEDIUM"
    else:
        label = "LOW"

    details = (
        "method_agreement=%.2f (%s); "
        "field_completeness=%.2f (%s); "
        "filename_consistency=%.2f (%s); "
        "payment_validity=%.2f (%s); "
        "ocr_quality=%.2f (%s); "
        "name_plausibility=%.2f (%s)"
    ) % (s1, r1, s2, r2, s3, r3, s4, r4, s5, r5, s6, r6)

    logger.info(
        "Confidence: %.2f [%s] -- %s",
        combined, label, details,
    )

    return {
        "confidence_score": combined,
        "confidence_label": label,
        "confidence_details": details,
    }
