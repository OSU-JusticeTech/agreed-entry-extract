"""
Outcome reclassification for vague or generic outcome types.

When the initial extraction yields ``Unknown``, ``Agreed Entry``,
``Agreed Judgment``, or ``Settlement``, this module re-classifies by
sending the full document text back to the LLM with a focused
classification prompt.

Ported from ``Step2_final_outcome_replace_uncertainty.ipynb``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from justicetech_extract.config import Settings
from justicetech_extract.models import ExtractedCourtInfo, OutcomeType

logger = logging.getLogger(__name__)

# Outcome types that trigger reclassification
LABELS_TO_RECLASSIFY = {"Unknown", "Agreed Entry", "Agreed Judgment", "Settlement"}

VALID_OUTCOMES = {"Pay and Stay", "Pay and Vacate", "Pay or Vacate", "Vacate Only"}

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a legal document classifier specializing in Franklin County \
Municipal Court eviction cases.

Classify the outcome into EXACTLY ONE category:

1. **Pay and Stay** — Tenant pays and STAYS. No unconditional vacate.
   "acknowledges receipt of Notice to Vacate" is PROCEDURAL, not vacate.
   "if breached... agrees to vacate" is a PENALTY CLAUSE — tenant stays if they pay.

2. **Pay and Vacate** — Tenant MUST pay AND vacate. Both required unconditionally.

3. **Pay or Vacate** — Tenant CHOOSES: pay OR vacate (alternative options).

4. **Vacate Only** — Tenant must vacate, no payment arrangement.

Return ONLY valid JSON:
{"outcome_type": "...", "confidence": "high/medium/low", "reasoning": "..."}
"""


def needs_reclassification(info: ExtractedCourtInfo) -> bool:
    """Check whether this result's outcome needs reclassification."""
    if info.outcome_type is None:
        return True
    return info.outcome_type.value in LABELS_TO_RECLASSIFY


def reclassify_outcome(
    info: ExtractedCourtInfo,
    document_text: str,
    settings: Optional[Settings] = None,
    max_retries: int = 3,
) -> ExtractedCourtInfo:
    """
    Re-classify a vague outcome type using the LLM.

    Parameters
    ----------
    info : ExtractedCourtInfo
        The initial extraction result.
    document_text : str
        Full OCR text for LLM context.
    settings : Settings, optional
        Pipeline configuration.
    max_retries : int
        Number of retry attempts on API failure.

    Returns
    -------
    ExtractedCourtInfo
        Updated model with reclassified outcome_type (if successful).
    """
    if settings is None:
        settings = Settings()

    if not needs_reclassification(info):
        return info

    if not settings.llm_api_key:
        logger.warning("No LLM API key — cannot reclassify outcome")
        return info

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed — cannot reclassify outcome")
        return info

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    truncated = document_text[:8000]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Case: {info.case_number or 'unknown'}\n\n"
                            f"Document text:\n{truncated}"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=300,
            )

            result_text = response.choices[0].message.content.strip()
            result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)
            parsed = json.loads(result_text)

            new_outcome = parsed.get("outcome_type")
            if new_outcome not in VALID_OUTCOMES:
                logger.warning("Invalid reclassification '%s', retrying...", new_outcome)
                time.sleep(1)
                continue

            # Map to enum
            for member in OutcomeType:
                if member.value == new_outcome:
                    info.outcome_type = member
                    info.outcome_details = parsed.get("reasoning", info.outcome_details)
                    info.extraction_method = f"{info.extraction_method}+Reclassified"
                    logger.info(
                        "Reclassified %s → %s (confidence: %s)",
                        info.case_number,
                        new_outcome,
                        parsed.get("confidence"),
                    )
                    return info

        except json.JSONDecodeError:
            logger.warning("JSON parse error on attempt %d", attempt + 1)
            time.sleep(1)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait = 10 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
            else:
                logger.error("Reclassification error: %s", e)
                return info

    logger.warning("Reclassification exhausted retries for %s", info.case_number)
    return info
