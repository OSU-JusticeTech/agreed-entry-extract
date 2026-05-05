"""Abstract base class for OCR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from PIL import Image


class OCRBackendBase(ABC):
    """
    Interface that all OCR backends must implement.

    Subclasses handle model loading, image preprocessing, and text generation.
    The pipeline calls :meth:`process_image` for each page, then concatenates
    the results with page markers.
    """

    @abstractmethod
    def process_image(self, image: Image.Image) -> str:
        """
        Run OCR on a single PIL Image and return the extracted text.

        Parameters
        ----------
        image : PIL.Image.Image
            An RGB image of a single document page.

        Returns
        -------
        str
            The OCR'd text for that page.
        """

    def process_document_images(self, images: list[Image.Image]) -> str:
        """
        Process a multi-page document (list of page images).

        Concatenates per-page results with page markers for downstream parsing.
        """
        if len(images) == 1:
            return self.process_image(images[0])

        parts: list[str] = []
        for page_num, img in enumerate(images, start=1):
            page_text = self.process_image(img)
            parts.append(f"\n\n\n=== PAGE {page_num} ===\n\n\n")
            parts.append(page_text)

        return "".join(parts)
