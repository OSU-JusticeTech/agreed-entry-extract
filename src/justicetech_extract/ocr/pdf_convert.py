"""
Convert a PDF file to a list of PIL images (one per page).

Requires ``pip install justicetech-extract[pdf]`` which brings in
``pdf2image`` (and a system install of ``poppler-utils``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[Image.Image]:
    """
    Convert every page of *pdf_path* to an RGB PIL Image.

    Parameters
    ----------
    pdf_path : str or Path
        Path to a ``.pdf`` file.
    dpi : int
        Resolution for rasterisation (default 300).

    Returns
    -------
    list[PIL.Image.Image]
        One image per page, in page order.

    Raises
    ------
    ImportError
        If ``pdf2image`` is not installed.
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for PDF conversion. "
            "Install with: pip install justicetech-extract[pdf]"
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Converting %s at %d DPI ...", pdf_path.name, dpi)
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info("  → %d page(s)", len(images))
    return [img.convert("RGB") for img in images]
