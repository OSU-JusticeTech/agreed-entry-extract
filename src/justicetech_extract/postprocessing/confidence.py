"""
Per-row confidence scoring for extracted court documents.

Computes a 0.00–1.00 confidence score based on five signals:

1. **Method agreement** (30%) — Do LLM and regex agree on outcome_type?
2. **Field completeness** (20%) — Are all expected fields populated?
3. **Filename consistency** (15%) — Do extracted IDs match the filename?
4. **Payment validity** (20%) — Are payment entries well-formed?
5. **OCR quality** (15%) — How clean was the OCR text?

The combined score is mapped to a label: HIGH (≥0.85), MEDIUM (≥0.60), LOW (<0.60).
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
W_METHOD_AGREEMENT = 0.30
W_FIELD_COMPLETENESS = 0.20
W_FILENAME_CONSISTENCY = 0.15
W_PAYMENT_VALIDITY = 0.20
W_OCR_QUALITY = 0.15


# ---------------------------------------------------------------------------
# Signal 1: Extraction Method Agreement
# ---------------------------------------------------------------------------

def _score_method_agreement(result: PipelineResult) -> tuple[float, str]:
    """
    Score based on whether LLM and regex extraction agreed.

    Returns (score, reason).
    """
    method = result.info.extraction_method or ""

    if "LLM+Regex" in method:
        # Both ran — check for override warnings in errors
        had_override = any("overriding" in e.lower() for e in result.errors)
        if had_override:
            return 0.6, "LLM+Regex ran but outcome was overridden (disagreement resolved)"
        return 1.0, "LLM and Regex both ran and agreed"
    elif "Regex-only" in method:
        return 0.4, "Regex-only (no LLM cross-validation)"
    else:
        # Unknown or single method
        return 0.3, f"Single extraction method: {method}"


# ---------------------------------------------------------------------------
# Signal 2: Field Completeness
# ---------------------------------------------------------------------------

def _score_field_completeness(info: ExtractedCourtInfo) -> tuple[float, str]:
    """
    Score based on how many expected fields are populated.

    Core fields: case_number, plaintiff, defendant, outcome_type.
    Conditional: payments (for Pay-and-Stay), vacate_date (for Vacate-Only).
    """
    missing = []

    # Core fields that should always be present
    if not info.case_number:
        missing.append("case_number")
    if not info.plaintiff:
        missing.append("plaintiff")
    if not info.defendant:
        missing.append("defendant")
    if not info.outcome_type:
        missing.append("outcome_type")

    # Conditional fields based on outcome
    outcome = (info.outcome_type.value if info.outcome_type else "").lower()

    if "pay" in outcome:
        # Pay-type outcomes should have payments
        if len(info.payment_schedule) == 0:
            missing.append("payment_schedule (expected for Pay outcome)")
    elif "vacate" in outcome:
        # Vacate outcomes should have a vacate date
        if not info.mandatory_vacate_date:
            missing.append("mandatory_vacate_date (expected for Vacate outcome)")

    total_checks = 4 + 1  # 4 core + 1 conditional
    populated = total_checks - len(missing)
    score = max(0.1, populated / total_checks)

    if missing:
        reason = f"Missing: {', '.join(missing)}"
    else:
        reason = "All expected fields populated"

    return round(score, 2), reason


# ---------------------------------------------------------------------------
# Signal 3: Filename Consistency
# ---------------------------------------------------------------------------

def _score_filename_consistency(info: ExtractedCourtInfo) -> tuple[float, str]:
    """
    Score based on whether filename-derived fields match extraction.

    The filename contains ground-truth case number and date.
    """
    if not info.filename:
        return 0.5, "No filename available for validation"

    checks_passed = 0
    checks_total = 0
    issues = []

    # Check case number appears in filename
    if info.case_number:
        checks_total += 1
        # Normalize both for comparison
        cn_digits = re.sub(r"[^0-9]", "", info.case_number)
        fn_digits = re.sub(r"[^0-9]", "", info.filename)
        if cn_digits and cn_digits in fn_digits:
            checks_passed += 1
        else:
            issues.append("case_number not found in filename")

    # Check date appears in filename
    if info.agreement_signed_date:
        checks_total += 1
        date_digits = re.sub(r"[^0-9]", "", info.agreement_signed_date)
        fn_clean = info.filename.replace("_", " ").replace("-", " ")
        # Try to find the date components in filename
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

def _score_payment_validity(info: ExtractedCourtInfo) -> tuple[float, str]:
    """
    Score based on payment schedule quality.

    Checks: completeness of individual entries, consistency of types,
    and whether a pay-type outcome actually has payments.
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
    issues = []

    for p in payments:
        has_amount = p.amount is not None or p.month_rent is not None
        has_date = p.due_date is not None

        if has_amount and has_date:
            well_formed += 1
        elif has_amount or has_date:
            partial += 1
        # else: empty — already filtered by pipeline

    total = len(payments)
    if total == 0:
        return 0.5, "Payment schedule present but empty after filtering"

    quality = (well_formed + 0.5 * partial) / total

    if partial > 0:
        issues.append(f"{partial} payment(s) missing date or amount")
    if well_formed == total:
        issues.append("All payments have amount and date")

    return round(quality, 2), "; ".join(issues) if issues else "Payments present"


# ---------------------------------------------------------------------------
# Signal 5: OCR Text Quality
# ---------------------------------------------------------------------------

def _score_ocr_quality(result: PipelineResult) -> tuple[float, str]:
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

    # Penalize for many OCR cleaning changes
    change_count = len(changes)
    if change_count > 20:
        score -= 0.3
        issues.append(f"{change_count} OCR corrections applied")
    elif change_count > 10:
        score -= 0.15
        issues.append(f"{change_count} OCR corrections applied")

    # Penalize for [?] uncertainty markers from Gemini
    uncertain_count = text.count("[?]")
    if uncertain_count > 5:
        score -= 0.3
        issues.append(f"{uncertain_count} uncertain [?] markers")
    elif uncertain_count > 2:
        score -= 0.15
        issues.append(f"{uncertain_count} uncertain [?] markers")

    # Penalize for very short text (possible scan failure)
    if len(text) < 200:
        score -= 0.2
        issues.append(f"Short OCR text ({len(text)} chars)")

    # Penalize for OCR artifact warnings
    artifact_warnings = [e for e in result.errors if "ocr" in e.lower() or "artifact" in e.lower()]
    if artifact_warnings:
        score -= 0.2
        issues.append("OCR artifact warning flagged")

    score = max(0.1, score)
    reason = "; ".join(issues) if issues else "Clean OCR text"
    return round(score, 2), reason


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

def compute_confidence(result: PipelineResult) -> dict:
    """
    Compute a per-row confidence score for an extraction result.

    Parameters
    ----------
    result : PipelineResult
        The full pipeline result with extraction output and diagnostics.

    Returns
    -------
    dict
        Keys: ``confidence_score`` (float 0.00–1.00),
        ``confidence_label`` (str: HIGH/MEDIUM/LOW),
        ``confidence_details`` (str: breakdown of signal scores).
    """
    s1, r1 = _score_method_agreement(result)
    s2, r2 = _score_field_completeness(result.info)
    s3, r3 = _score_filename_consistency(result.info)
    s4, r4 = _score_payment_validity(result.info)
    s5, r5 = _score_ocr_quality(result)

    combined = (
        W_METHOD_AGREEMENT * s1
        + W_FIELD_COMPLETENESS * s2
        + W_FILENAME_CONSISTENCY * s3
        + W_PAYMENT_VALIDITY * s4
        + W_OCR_QUALITY * s5
    )
    combined = round(combined, 2)

    if combined >= 0.85:
        label = "HIGH"
    elif combined >= 0.60:
        label = "MEDIUM"
    else:
        label = "LOW"

    details = (
        f"method_agreement={s1:.2f} ({r1}); "
        f"field_completeness={s2:.2f} ({r2}); "
        f"filename_consistency={s3:.2f} ({r3}); "
        f"payment_validity={s4:.2f} ({r4}); "
        f"ocr_quality={s5:.2f} ({r5})"
    )

    logger.info(
        "Confidence: %.2f [%s] — %s",
        combined, label, details,
    )

    return {
        "confidence_score": combined,
        "confidence_label": label,
        "confidence_details": details,
    }
