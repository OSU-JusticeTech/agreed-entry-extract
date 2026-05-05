"""
OCR backends for converting document images to text.

Supported backends:

- **gemini**: Cloud OCR via Google Gemini 3 Flash (recommended)
- **nanonets**: Local inference with ``nanonets/Nanonets-OCR2-3B`` (legacy)
- **external**: Bypass OCR — caller provides pre-existing text

Each backend implements :class:`OCRBackendBase` so they can be swapped
without changing downstream code.
"""

from justicetech_extract.ocr.base import OCRBackendBase
from justicetech_extract.ocr.clean import clean_ocr_text
from justicetech_extract.ocr.pdf_convert import pdf_to_images

__all__ = ["OCRBackendBase", "clean_ocr_text", "pdf_to_images"]
