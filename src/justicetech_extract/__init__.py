"""
justicetech-extract
===================

Extract structured data from Franklin County Municipal Court eviction documents.

Quick start
-----------

Process a single PDF (Gemini 3 Flash OCR via LiteLLM — recommended)::

    from justicetech_extract import process_pdf
    from justicetech_extract.config import Settings

    settings = Settings(
        ocr_backend="gemini",
        llm_api_key="sk-...",  # same key for both OCR and extraction
    )
    result = process_pdf("path/to/document.pdf", settings=settings)
    print(result.info.case_number)
    print(result.info.outcome_type)

Or just set env vars and it works with zero config::

    # In .env:
    #   OCR_BACKEND=gemini
    #   LLM_API_KEY=sk-...
    result = process_pdf("path/to/document.pdf")

Process pre-OCR'd text::

    from justicetech_extract import extract

    info = extract(ocr_text, filename="2024_CVG_056254_..._cleaned.txt")

With custom settings::

    from justicetech_extract import extract
    from justicetech_extract.config import Settings

    settings = Settings(
        llm_api_key="sk-...",
        llm_model="GPT-4o",
        ocr_backend="external",
    )
    info = extract(text, settings=settings)
"""

from __future__ import annotations

__version__ = "0.3.1"

import logging
from pathlib import Path
from typing import Optional, Union

from justicetech_extract.config import OCRBackend, Settings
from justicetech_extract.extraction.pipeline import extract
from justicetech_extract.models import (
    ExtractedCourtInfo,
    OutcomeType,
    PaymentScheduleItem,
    PaymentType,
    PipelineResult,
)
from justicetech_extract.ocr.clean import clean_ocr_text
from justicetech_extract.postprocessing.fixups import apply_fixups
from justicetech_extract.postprocessing.reclassify import (
    needs_reclassification,
    reclassify_outcome,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Main entry points
    "extract",
    "extract_from_text",
    "process_pdf",
    # Models
    "ExtractedCourtInfo",
    "PaymentScheduleItem",
    "PaymentType",
    "OutcomeType",
    "PipelineResult",
    # Config
    "Settings",
    "OCRBackend",
]


def extract_from_text(
    text: str,
    filename: Optional[str] = None,
    settings: Optional[Settings] = None,
    clean: bool = True,
    reclassify: bool = True,
) -> PipelineResult:
    """
    Full extraction pipeline from OCR text.

    This is the **recommended entry point** for processing a single document
    when you already have the OCR text.

    Parameters
    ----------
    text : str
        OCR'd document text.
    filename : str, optional
        Source filename (used for case_number / date extraction).
    settings : Settings, optional
        Pipeline configuration.  Loads from env vars if *None*.
    clean : bool
        Whether to apply OCR text cleaning before extraction.
    reclassify : bool
        Whether to reclassify vague outcome types via LLM.

    Returns
    -------
    PipelineResult
        Full result with extraction output, OCR diagnostics, and warnings.
    """
    if settings is None:
        settings = Settings()

    errors: list[str] = []
    ocr_changes: list[str] = []
    raw_text = text

    # Step 1: Clean OCR text
    if clean:
        text, ocr_changes = clean_ocr_text(text)

    # Step 2: Extract
    info = extract(text, filename=filename, settings=settings)

    # Step 3: Post-processing fixups
    info = apply_fixups(info)

    # Step 4: Reclassify vague outcomes
    if reclassify and needs_reclassification(info) and settings.llm_api_key:
        try:
            info = reclassify_outcome(info, text, settings=settings)
        except Exception as e:
            errors.append(f"Reclassification failed: {e}")

    # Step 5: Flag OCR artifacts
    from justicetech_extract.postprocessing.fixups import flag_ocr_plaintiff

    flag = flag_ocr_plaintiff(info)
    if flag:
        errors.append(flag)

    result = PipelineResult(
        info=info,
        ocr_text=raw_text,
        cleaned_text=text if clean else None,
        ocr_backend="external",
        ocr_changes=ocr_changes,
        errors=errors,
    )

    # Step 6: Compute per-row confidence score
    from justicetech_extract.postprocessing.confidence import compute_confidence

    conf = compute_confidence(result)
    result.info.confidence_score = conf["confidence_score"]
    result.info.confidence_label = conf["confidence_label"]
    result.info.confidence_details = conf["confidence_details"]

    return result


def process_pdf(
    pdf_path: Union[str, Path],
    filename: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> PipelineResult:
    """
    End-to-end pipeline: PDF → images → OCR → clean → extract → post-process.

    This is the **recommended entry point** for processing a single PDF
    when you want the package to handle everything.

    Parameters
    ----------
    pdf_path : str or Path
        Path to a ``.pdf`` file.
    filename : str, optional
        Override filename for metadata extraction.  Defaults to the
        PDF's stem.
    settings : Settings, optional
        Pipeline configuration.

    Returns
    -------
    PipelineResult
        Full result including OCR text, extraction, and diagnostics.

    Raises
    ------
    ImportError
        If required OCR dependencies are not installed.
    """
    if settings is None:
        settings = Settings()

    pdf_path = Path(pdf_path)
    if filename is None:
        filename = pdf_path.stem

    # Step 1: PDF → images
    from justicetech_extract.ocr.pdf_convert import pdf_to_images

    images = pdf_to_images(pdf_path, dpi=settings.pdf_dpi)

    # Step 2: OCR
    ocr_backend_name = settings.ocr_backend.value
    if settings.ocr_backend == OCRBackend.GEMINI:
        from justicetech_extract.ocr.gemini import GeminiOCR

        backend = GeminiOCR(
            model=settings.gemini_model,
            api_key=settings.effective_gemini_api_key,
            base_url=settings.effective_gemini_base_url,
        )
    elif settings.ocr_backend == OCRBackend.NANONETS:
        from justicetech_extract.ocr.nanonets import NanonetsOCR

        backend = NanonetsOCR(
            model_path=settings.ocr_model_path,
            device=settings.ocr_device,
            max_dimension=settings.ocr_max_dimension,
        )
    else:
        raise ValueError(
            f"OCR backend '{settings.ocr_backend}' cannot process PDFs. "
            "Use 'gemini' or 'nanonets', or provide pre-OCR'd text via extract_from_text()."
        )

    raw_ocr = backend.process_document_images(images)

    # Step 3-5: Clean → Extract → Post-process
    result = extract_from_text(
        raw_ocr,
        filename=filename,
        settings=settings,
        clean=True,
        reclassify=True,
    )
    result.ocr_text = raw_ocr
    result.ocr_backend = ocr_backend_name

    return result
