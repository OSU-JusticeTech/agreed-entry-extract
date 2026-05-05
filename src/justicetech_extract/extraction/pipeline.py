"""
Combined extraction pipeline: LLM + regex cross-validation.

This is the primary entry point for extracting structured data from a
single court document's OCR text.  It mirrors the v3.8/v3.9 strategy:

1. Try LLM extraction first (if configured).
2. Run regex extraction.
3. Cross-validate: fill LLM gaps with regex, override LLM outcome_type
   when regex detects that all vacate language is inside breach clauses.
4. Always prefer filename-derived case_number and agreement_date.
"""

from __future__ import annotations

import logging
from typing import Optional

from justicetech_extract.config import Settings
from justicetech_extract.extraction.llm_extractor import LLMExtractor
from justicetech_extract.extraction.regex_extractor import RegexExtractor
from justicetech_extract.models import ExtractedCourtInfo, OutcomeType

logger = logging.getLogger(__name__)


def extract(
    text: str,
    filename: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> ExtractedCourtInfo:
    """
    Extract structured information from a court document.

    This is the **main public function** of the package.

    Parameters
    ----------
    text : str
        OCR'd (and optionally cleaned) document text.
    filename : str, optional
        Original filename — used to extract case_number and agreement_date
        from structured naming conventions.
    settings : Settings, optional
        Pipeline configuration.  If *None*, loads from environment variables.

    Returns
    -------
    ExtractedCourtInfo
        Populated Pydantic model with all extracted fields.

    Examples
    --------
    Minimal usage (regex only)::

        from justicetech_extract import extract

        with open("document.txt") as f:
            text = f.read()

        info = extract(text, filename="2024_CVG_056254_..._cleaned.txt")
        print(info.case_number)      # "2024 CVG 056254"
        print(info.outcome_type)     # OutcomeType.PAY_AND_STAY
        print(info.to_flat_dict())   # dict suitable for CSV/database

    With LLM enhancement::

        from justicetech_extract import extract
        from justicetech_extract.config import Settings

        settings = Settings(
            llm_api_key="sk-...",
            llm_base_url="https://litellmproxy.osu-ai.org",
            llm_model="GPT-4o",
        )

        info = extract(text, filename="doc.txt", settings=settings)
    """
    if settings is None:
        settings = Settings()

    # --- Step 1: Regex extraction (always runs) ---
    regex_result = RegexExtractor.extract_all(text, filename)
    logger.info("Regex extraction complete")

    # --- Step 2: LLM extraction (if configured) ---
    llm_result: Optional[ExtractedCourtInfo] = None
    if settings.use_llm and settings.llm_api_key:
        try:
            llm_ext = LLMExtractor(settings)
            if llm_ext.available:
                llm_result = llm_ext.extract_to_model(text, filename)
                if llm_result:
                    logger.info("LLM extraction complete")
        except Exception as e:
            logger.warning("LLM extraction failed, using regex only: %s", e)

    # --- Step 3: Merge results ---
    if llm_result is not None:
        merged = _merge_results(llm_result, regex_result, text)
        merged.extraction_method = "LLM+Regex"
    else:
        merged = regex_result
        merged.extraction_method = "Regex-only"

    # --- Step 4: Filename overrides (always win) ---
    fn_case = RegexExtractor.extract_case_number("", filename)
    fn_date = RegexExtractor.extract_agreement_date("", filename)

    if fn_case:
        merged.case_number = fn_case
    if fn_date:
        merged.agreement_signed_date = fn_date

    merged.filename = filename
    return merged


def _merge_results(
    llm: ExtractedCourtInfo,
    regex: ExtractedCourtInfo,
    text: str,
) -> ExtractedCourtInfo:
    """
    Cross-validate LLM and regex results.

    Strategy (matching v3.8+ logic):
    - Start with LLM result.
    - Fill any ``None`` fields from regex.
    - If LLM says "vacate" but regex says "Pay and Stay" AND all vacate
      language in the text is inside breach/notice clauses, override to
      regex outcome.
    """
    # Fill gaps: use regex values where LLM returned None
    merged_data = llm.model_dump()
    regex_data = regex.model_dump()

    for key, regex_val in regex_data.items():
        if key in ("raw_text", "extraction_method", "filename"):
            continue
        if merged_data.get(key) is None and regex_val is not None:
            merged_data[key] = regex_val
            logger.debug("Enhanced %s with regex", key)

    # Cross-validate outcome_type
    llm_outcome = (llm.outcome_type.value if llm.outcome_type else "").lower().replace(" ", "")
    regex_outcome = (regex.outcome_type.value if regex.outcome_type else "").lower().replace(" ", "")

    if llm_outcome != regex_outcome and regex.outcome_type:
        if "vacate" in llm_outcome and regex_outcome == "payandstay":
            # Check if all vacate language is in breach context
            all_vacate_is_breach = RegexExtractor.detect_breach_clause_vacate(text)

            if all_vacate_is_breach:
                logger.warning(
                    "Overriding LLM outcome '%s' → '%s' "
                    "(all vacate language is in breach/notice context)",
                    llm.outcome_type,
                    regex.outcome_type,
                )
                merged_data["outcome_type"] = regex.outcome_type
                merged_data["outcome_details"] = regex.outcome_details
                merged_data["mandatory_vacate_date"] = None
            else:
                logger.info(
                    "LLM says '%s', regex says '%s' — trusting LLM "
                    "(vacate language found outside breach context)",
                    llm.outcome_type,
                    regex.outcome_type,
                )
                if regex.mandatory_vacate_date:
                    merged_data["mandatory_vacate_date"] = regex.mandatory_vacate_date

    # Filter out blank/template payments (no amount AND no month_rent)
    # These come from empty table rows that the LLM may count as payments.
    if merged_data.get("payment_schedule"):
        original_count = len(merged_data["payment_schedule"])
        merged_data["payment_schedule"] = [
            p for p in merged_data["payment_schedule"]
            if (p.get("amount") if isinstance(p, dict) else getattr(p, "amount", None)) is not None
            or (p.get("month_rent") if isinstance(p, dict) else getattr(p, "month_rent", None)) is not None
        ]
        filtered_count = len(merged_data["payment_schedule"])
        if filtered_count < original_count:
            logger.info(
                "Removed %d blank payment(s) from schedule",
                original_count - filtered_count,
            )
            # Renumber payments
            for i, p in enumerate(merged_data["payment_schedule"]):
                if isinstance(p, dict):
                    p["payment_number"] = i + 1
                else:
                    p.payment_number = i + 1

    # Reconstruct model
    result = ExtractedCourtInfo(**merged_data)
    return result
