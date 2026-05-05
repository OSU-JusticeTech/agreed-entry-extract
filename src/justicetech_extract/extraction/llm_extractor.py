"""
LLM-based extraction using any OpenAI-compatible API.

This module sends the OCR text to an LLM (e.g. GPT-4o via OSU LiteLLM
proxy) with a structured prompt and parses the JSON response into
:class:`~justicetech_extract.models.ExtractedCourtInfo`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from justicetech_extract.config import Settings
from justicetech_extract.models import (
    ExtractedCourtInfo,
    OutcomeType,
    PaymentScheduleItem,
    PaymentType,
)

logger = logging.getLogger(__name__)

# Maximum chars of document text to send to the LLM
MAX_TEXT_LENGTH = 15_000

# The system + user prompt that has been validated on the 9-document test suite
SYSTEM_PROMPT = (
    "You extract structured data from legal documents. "
    "Return valid JSON. Be careful to distinguish procedural language "
    "and breach-conditional language from actual unconditional obligations."
)

EXTRACTION_PROMPT = """\
You are analyzing Franklin County Municipal Court eviction case documents. \
Extract ALL information accurately.

⚠️ CRITICAL: AVOID LEGALESE FALSE POSITIVES ⚠️

Many documents contain the phrase "Defendant(s) hereby acknowledges receipt of the Notice to Vacate served..."
This is PROCEDURAL LANGUAGE acknowledging that a notice was served in the past.
It is NOT an agreement to vacate.

Similarly, many documents contain breach clauses like:
"if any of the terms or conditions are breached... Defendant agrees to vacate the premises immediately"
This is a CONDITIONAL CONSEQUENCE of breach, NOT a mandatory vacate obligation.
The defendant is agreeing to a penalty IF they fail, not agreeing to vacate as part of the deal.

When a payment agreement follows this language, it typically means the defendant is paying to STAY, not paying to leave.

HOWEVER, some documents contain BOTH a payment schedule AND an explicit, unconditional vacate obligation
outside of any breach clause, such as:
- "* D Shall vacate by 2/29/24."
- "Defendant shall vacate the premises by March 1, 2024."
These indicate "Pay and Vacate" — the defendant must pay AND leave.

REQUIRED FIELDS:

1. **Case Number**: Format like "2024 CVG 056254" or "24 CVG 55650"

2. **Plaintiff**: The landlord/property owner (appears before "v." or "Plaintiff,")

3. **Defendant**: The tenant (appears after "v." or "Defendant(s)")

4. **Payment Schedule**: Extract EVERY payment with:
   - due_date: The date payment is due (format: YYYY-MM-DD)
   - payment_type: Either "fixed_amount" or "monthly_rent"
   - amount: Dollar amount (numeric only, no $) - only for fixed_amount type
   - month_rent: Month name (e.g., "February") - only for monthly_rent type
   - time: If specified (e.g., "6:00 PM", "11:59 PM")
   - extra_text: Additional info (e.g., "third party assistance", "water + late fees")

   IMPORTANT: Tables may have 2, 3, or 4 columns. Check ALL columns for data.
   Amount descriptions may be like "$Feb rent + late fee" or "$2538.52"

   CRITICAL RULE FOR MONTHLY RENT ENTRIES:
   When a payment line says "March rent", "Feb rent", "April rent", or similar:
   - Set payment_type to "monthly_rent"
   - Set month_rent to the month name (e.g., "March")
   - Do NOT guess or invent a dollar amount. Leave amount as null.
   - Only set amount if a specific dollar figure is explicitly written next to the rent label
   For example: "$March rent" → payment_type: "monthly_rent", month_rent: "March", amount: null
   But: "$750 March rent" → payment_type: "monthly_rent", month_rent: "March", amount: "750"

5. **Mandatory Vacate Date**: ONLY extract if there is an ACTUAL UNCONDITIONAL OBLIGATION to vacate
   ✓ CORRECT: "Defendant shall vacate the premises by January 15, 2025"
   ✓ CORRECT: "Defendant agrees to vacate by 2/1/2025"
   ✓ CORRECT: "* D Shall vacate by 2/29/24" (abbreviated Defendant)
   ✓ CORRECT: "agree to voluntarily vacate the premise and turn in keys on or before 6-30-24"
   ✗ WRONG: "acknowledges receipt of the Notice to Vacate served on..."
   ✗ WRONG: "Notice to Vacate was served in accordance with..."
   ✗ WRONG: "if terms are breached... agrees to vacate immediately" (CONDITIONAL)
   ✗ WRONG: "Defendant agrees to not object to Plaintiff's judgment and/or file any motion or pleading to stop the setout." (BREACH CONSEQUENCE)

   If you only see acknowledgment of a Notice to Vacate being served, set mandatory_vacate_date to null.
   If the vacate language appears inside a breach/default clause, set mandatory_vacate_date to null.

6. **Outcome Type**: MUST be one of these three categories:
   - "Vacate Only" - Defendant agrees to vacate before a certain date (no payment)
   - "Pay and Stay" - Defendant agrees to pay and remain in the home
   - "Pay and Vacate" - Defendant agrees to both pay AND vacate

   DECISION LOGIC:
   - Has payment schedule + NO actual unconditional vacate obligation = "Pay and Stay"
   - Has payment schedule + actual unconditional vacate date = "Pay and Vacate"
   - NO payment + has vacate date = "Vacate Only"

   "acknowledges receipt of Notice to Vacate" ≠ actual vacate obligation!
   "if breached... agrees to vacate" ≠ actual vacate obligation! (it's a penalty clause)

7. **Outcome Details**: Additional specifics

8. **Additional Agreement Terms**: Key conditions (semicolon-separated string)

9. **Other fields**: sealing_reference_stipulation, enforcement_period, assistance_deadline

Return JSON with snake_case keys. Use null for missing values.

Document Text:
{document_text}"""


class LLMExtractor:
    """
    Extract court document information using an OpenAI-compatible LLM.

    Parameters
    ----------
    settings : Settings
        Pipeline configuration (API key, model, URL, etc.).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        """Lazy-initialise the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai library is required for LLM extraction. "
                    "Install with: pip install openai"
                ) from exc

            if not self.settings.llm_api_key:
                raise ValueError(
                    "LLM_API_KEY is required for LLM extraction. "
                    "Set the LLM_API_KEY environment variable or pass it in Settings."
                )

            self._client = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._client

    @property
    def available(self) -> bool:
        """Whether the LLM backend is configured and ready."""
        try:
            _ = self.client
            return True
        except (ImportError, ValueError):
            return False

    def extract(self, text: str) -> dict[str, Any]:
        """
        Send document text to the LLM and return the parsed JSON response.

        Returns an empty dict on failure (logged, not raised).
        """
        try:
            prompt = EXTRACTION_PROMPT.format(document_text=text[:MAX_TEXT_LENGTH])

            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.llm_temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            logger.info("LLM extraction successful")

            # Normalize list fields
            if isinstance(result.get("additional_agreement_terms"), list):
                result["additional_agreement_terms"] = "; ".join(
                    result["additional_agreement_terms"]
                )

            return result

        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            return {}

    def extract_to_model(self, text: str, filename: Optional[str] = None) -> Optional[ExtractedCourtInfo]:
        """
        Extract and return as a Pydantic model, or ``None`` on failure.
        """
        raw = self.extract(text)
        if not raw:
            return None

        # Convert payment_schedule dicts to models
        payments = []
        for i, p in enumerate(raw.get("payment_schedule", []), start=1):
            ptype = None
            if p.get("payment_type") == "fixed_amount":
                ptype = PaymentType.FIXED_AMOUNT
            elif p.get("payment_type") == "monthly_rent":
                ptype = PaymentType.MONTHLY_RENT

            # Coerce amount to string — GPT sometimes returns numbers instead of strings
            amount = p.get("amount")
            if amount is not None and not isinstance(amount, str):
                amount = str(amount)

            payments.append(
                PaymentScheduleItem(
                    payment_number=i,
                    due_date=p.get("due_date"),
                    payment_type=ptype,
                    amount=amount,
                    month_rent=p.get("month_rent"),
                    time=p.get("time"),
                    extra_text=p.get("extra_text"),
                )
            )

        # Map outcome_type string to enum
        outcome_enum = None
        ot = raw.get("outcome_type")
        if ot:
            for member in OutcomeType:
                if member.value.lower() == ot.lower():
                    outcome_enum = member
                    break

        # Coerce numeric fields that GPT may return as numbers
        total_sum = raw.get("total_payment_sum")
        if total_sum is not None and not isinstance(total_sum, str):
            total_sum = str(total_sum)

        return ExtractedCourtInfo(
            case_number=raw.get("case_number"),
            plaintiff=raw.get("plaintiff"),
            defendant=raw.get("defendant"),
            payment_schedule=payments,
            total_payment_sum=total_sum,
            outcome_type=outcome_enum,
            outcome_details=raw.get("outcome_details"),
            mandatory_vacate_date=raw.get("mandatory_vacate_date"),
            third_party_acceptance=raw.get("third_party_acceptance"),
            additional_agreement_terms=raw.get("additional_agreement_terms"),
            extraction_method="LLM",
            filename=filename,
            raw_text=text,
        )
