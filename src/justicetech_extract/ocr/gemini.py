"""
Gemini 3 Flash OCR backend (via OSU LiteLLM proxy).

Uses Gemini 3 Flash's native vision capabilities for document OCR,
routed through the same OSU LiteLLM proxy used for GPT-4o extraction.

No additional dependencies required — uses the ``openai`` SDK already
in the base package requirements.

Environment variables
---------------------
GEMINI_API_KEY       LiteLLM proxy API key.  Falls back to ``LLM_API_KEY``.
GEMINI_BASE_URL      LiteLLM proxy URL.  Falls back to ``LLM_BASE_URL``.
GEMINI_MODEL         Model name registered in LiteLLM.
                     Default: ``gemini-3-flash-preview``
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

from PIL import Image

from justicetech_extract.ocr.base import OCRBackendBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default OCR prompt — tuned for court documents
# ---------------------------------------------------------------------------
DEFAULT_OCR_PROMPT = (
    "Extract all text from this document image exactly as written. "
    "Preserve the original layout, paragraph breaks, and formatting. "
    "Return tables in plain text with aligned columns. "
    "If there are checkboxes, use ☐ for unchecked and ☑ for checked. "
    "If there is handwritten text, transcribe it as accurately as possible "
    "and note any uncertainty with [?]. "
    "Do NOT add any commentary, explanation, or markdown formatting — "
    "return only the extracted document text."
)


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class GeminiOCR(OCRBackendBase):
    """
    OCR using Gemini 3 Flash via OSU's LiteLLM proxy.

    This backend calls the same LiteLLM proxy endpoint used for GPT-4o
    extraction, just targeting a different model.  No local GPU, no
    Google credentials, no extra dependencies required.

    Parameters
    ----------
    model : str
        Model name as registered in LiteLLM.
        Default: ``gemini-3-flash-preview``.
    api_key : str
        LiteLLM proxy API key (same key used for GPT-4o).
    base_url : str
        LiteLLM proxy base URL.
        Default: ``https://litellmproxy.osu-ai.org``.
    max_output_tokens : int
        Maximum tokens the model may generate per page.
    max_retries : int
        Number of retries on transient API errors.
    ocr_prompt : str, optional
        Custom OCR prompt.  Uses :data:`DEFAULT_OCR_PROMPT` if *None*.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        base_url: str = "https://litellmproxy.osu-ai.org",
        max_output_tokens: int = 8192,
        max_retries: int = 3,
        ocr_prompt: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.ocr_prompt = ocr_prompt or DEFAULT_OCR_PROMPT

        self._client = None

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-initialise the OpenAI client pointed at LiteLLM proxy."""
        if self._client is not None:
            return self._client

        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info(
            "Initialised Gemini OCR client (model=%s, proxy=%s)",
            self.model,
            self.base_url,
        )
        return self._client

    # ------------------------------------------------------------------
    # Core OCR
    # ------------------------------------------------------------------

    def process_image(self, image: Image.Image) -> str:  # noqa: D401
        """OCR a single page image via Gemini 3 Flash and return text."""
        client = self._get_client()
        image = image.convert("RGB")

        # Convert PIL image → base64 data URL for the vision API
        b64 = _pil_to_base64(image, "PNG")
        data_url = f"data:image/png;base64,{b64}"

        # Build the vision message (OpenAI-compatible format)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": self.ocr_prompt,
                    },
                ],
            },
        ]

        # Retry loop for transient errors
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    temperature=1.0,  # Gemini 3 default; do not lower
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # Retry on rate-limit or transient server errors
                if any(kw in err_str for kw in ("429", "503", "500", "rate", "quota")):
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini OCR error (attempt %d/%d): %s — retrying in %ds",
                        attempt,
                        self.max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"Gemini OCR failed after {self.max_retries} retries: {last_err}"
        )

    def process_document_images(self, images: list[Image.Image]) -> str:
        """
        Process a multi-page document.

        Overrides the base implementation to provide progress logging,
        since each page is a remote API call.
        """
        if len(images) == 1:
            return self.process_image(images[0])

        parts: list[str] = []
        for page_num, img in enumerate(images, start=1):
            logger.info("  OCR page %d/%d ...", page_num, len(images))
            page_text = self.process_image(img)
            parts.append(f"\n\n\n=== PAGE {page_num} ===\n\n\n")
            parts.append(page_text)

        return "".join(parts)
