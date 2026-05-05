"""
Pydantic data models for JusticeTech court document extraction.

These models define the schema for extracted court information,
payment schedules, and pipeline results.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PaymentType(str, Enum):
    """Type of payment obligation."""

    FIXED_AMOUNT = "fixed_amount"
    MONTHLY_RENT = "monthly_rent"


class OutcomeType(str, Enum):
    """Outcome classification for eviction agreements."""

    PAY_AND_STAY = "Pay and Stay"
    PAY_AND_VACATE = "Pay and Vacate"
    PAY_OR_VACATE = "Pay or Vacate"
    VACATE_ONLY = "Vacate Only"
    AGREED_ENTRY = "Agreed Entry"
    AGREED_JUDGMENT = "Agreed Judgment"
    SETTLEMENT = "Settlement"
    UNKNOWN = "Unknown"


class PaymentScheduleItem(BaseModel):
    """A single payment in a court-ordered payment schedule."""

    payment_number: int = Field(..., ge=1, description="1-indexed payment number")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    payment_type: Optional[PaymentType] = None
    amount: Optional[str] = Field(None, description="Dollar amount (numeric string, no $)")
    month_rent: Optional[str] = Field(None, description="Month name for rent payments")
    time: Optional[str] = Field(None, description="Time of day if specified (e.g. '5:00 PM')")
    extra_text: Optional[str] = Field(None, description="Additional info (e.g. 'water + late fees')")
    raw_text: Optional[str] = Field(None, description="Original text this was extracted from")


class ExtractedCourtInfo(BaseModel):
    """
    Complete extracted information from a single Franklin County
    Municipal Court eviction document.

    This is the primary output of the extraction pipeline.
    """

    # --- Identifiers ---
    case_number: Optional[str] = Field(None, description="e.g. '2024 CVG 056254'")
    agreement_signed_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    filename: Optional[str] = Field(None, description="Source document filename")

    # --- Parties ---
    plaintiff: Optional[str] = Field(None, description="Landlord / property owner name")
    defendant: Optional[str] = Field(None, description="Tenant name")

    # --- Financial terms ---
    payment_schedule: list[PaymentScheduleItem] = Field(default_factory=list)
    total_payment_sum: Optional[str] = Field(None, description="Sum of fixed payments")

    # --- Outcome ---
    outcome_type: Optional[OutcomeType] = None
    outcome_details: Optional[str] = None
    mandatory_vacate_date: Optional[str] = Field(None, description="YYYY-MM-DD if unconditional vacate")

    # --- Additional terms ---
    third_party_acceptance: Optional[bool] = None
    assistance_deadline: Optional[str] = None
    additional_agreement_terms: Optional[str] = None
    sealing_reference_stipulation: Optional[str] = None
    enforcement_period: Optional[str] = None

    # --- Metadata ---
    extraction_method: Optional[str] = Field(
        None, description="How extraction was performed (e.g. 'LLM+Regex', 'Regex-only')"
    )
    raw_text: Optional[str] = Field(None, description="Full OCR text (excluded from serialization by default)")

    # --- Confidence ---
    confidence_score: Optional[float] = Field(
        None, description="Combined confidence score (0.00–1.00)"
    )
    confidence_label: Optional[str] = Field(
        None, description="Confidence tier: HIGH (≥0.85), MEDIUM (≥0.60), LOW (<0.60)"
    )
    confidence_details: Optional[str] = Field(
        None, description="Breakdown of individual signal scores"
    )

    def to_flat_dict(self, max_payments: int = 20) -> dict:
        """
        Flatten into a single-level dict suitable for CSV / database row.

        Payment schedule items are expanded into ``payment_1_date``,
        ``payment_1_amount``, ... up to *max_payments*.
        """
        row: dict = {
            "filename": self.filename,
            "case_number": self.case_number,
            "agreement_signed_date": self.agreement_signed_date,
            "plaintiff": self.plaintiff,
            "defendant": self.defendant,
            "outcome_type": self.outcome_type.value if self.outcome_type else None,
            "outcome_details": self.outcome_details,
            "payment_count": len(self.payment_schedule),
            "total_payment_sum": self.total_payment_sum,
            "has_more_than_10_payments": len(self.payment_schedule) > 10,
        }

        for i in range(max_payments):
            if i < len(self.payment_schedule):
                p = self.payment_schedule[i]
                row[f"payment_{i+1}_date"] = p.due_date

                if p.payment_type == PaymentType.FIXED_AMOUNT:
                    row[f"payment_{i+1}_amount"] = p.amount
                elif p.payment_type == PaymentType.MONTHLY_RENT:
                    parts = []
                    if p.month_rent:
                        parts.append(p.month_rent)
                    if p.amount:
                        parts.append(f"${p.amount}")
                    row[f"payment_{i+1}_amount"] = " ".join(parts) if parts else "Monthly rent"
                else:
                    row[f"payment_{i+1}_amount"] = p.amount

                row[f"payment_{i+1}_type"] = p.payment_type.value if p.payment_type else None
                row[f"payment_{i+1}_extra"] = p.extra_text
            else:
                row[f"payment_{i+1}_date"] = None
                row[f"payment_{i+1}_amount"] = None
                row[f"payment_{i+1}_type"] = None
                row[f"payment_{i+1}_extra"] = None

        row.update(
            {
                "mandatory_vacate_date": self.mandatory_vacate_date,
                "third_party_acceptance": self.third_party_acceptance,
                "additional_agreement_terms": self.additional_agreement_terms,
                "assistance_deadline": self.assistance_deadline,
                "extraction_method": self.extraction_method,
                "confidence_score": self.confidence_score,
                "confidence_label": self.confidence_label,
                "confidence_details": self.confidence_details,
            }
        )
        return row


class PipelineResult(BaseModel):
    """Full pipeline result wrapping extraction output with diagnostics."""

    info: ExtractedCourtInfo
    ocr_text: Optional[str] = Field(None, description="Raw OCR output before cleaning")
    cleaned_text: Optional[str] = Field(None, description="Cleaned OCR text fed to extractor")
    ocr_backend: Optional[str] = Field(None, description="Which OCR backend was used")
    ocr_changes: list[str] = Field(default_factory=list, description="Cleaning changes applied")
    errors: list[str] = Field(default_factory=list, description="Non-fatal warnings during processing")
