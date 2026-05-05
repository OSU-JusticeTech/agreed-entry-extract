"""
Nanonets OCR backend (``nanonets/Nanonets-OCR2-3B``).

Requires ``pip install justicetech-extract[nanonets]`` for torch + transformers.

This backend loads the model fresh for each image to avoid GPU memory leaks
on long-running batch jobs.  For single-PDF processing this is acceptable;
for high-throughput batch work consider keeping the model loaded.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

from PIL import Image

from justicetech_extract.ocr.base import OCRBackendBase

logger = logging.getLogger(__name__)


def _resize_if_needed(image: Image.Image, max_dimension: int) -> Image.Image:
    """Down-scale image so neither side exceeds *max_dimension*."""
    w, h = image.size
    if w <= max_dimension and h <= max_dimension:
        return image
    scale = max_dimension / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


class NanonetsOCR(OCRBackendBase):
    """
    OCR using the Nanonets-OCR2-3B vision-language model.

    Parameters
    ----------
    model_path : str
        HuggingFace model identifier or local path.
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_dimension : int
        Resize images to this max edge length before inference.
    max_new_tokens : int
        Maximum tokens the model may generate per page.
    keep_loaded : bool
        If *True*, keep model in GPU memory between calls (faster but uses
        more memory).  If *False* (default), load/unload per call.
    """

    # Default prompt matching the production pipeline
    DEFAULT_PROMPT = (
        "Extract the text from the above document as if you were reading it naturally. "
        "Return the tables in html format. Return the equations in LaTeX representation. "
        "If there is an image in the document and image caption is not present, add a small "
        "description of the image inside the <img></img> tag; otherwise, add the image caption "
        "inside <img></img>. Watermarks should be wrapped in brackets. "
        "Ex: <watermark>OFFICIAL COPY</watermark>. "
        "Page numbers should be wrapped in brackets. "
        "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
        "Prefer using ☐ and ☑ for check boxes."
    )

    def __init__(
        self,
        model_path: str = "nanonets/Nanonets-OCR2-3B",
        device: str = "cuda",
        max_dimension: int = 1024,
        max_new_tokens: int = 10_000,
        keep_loaded: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_dimension = max_dimension
        self.max_new_tokens = max_new_tokens
        self.keep_loaded = keep_loaded
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load model and processor."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, PretrainedConfig

        logger.info("Loading Nanonets model from %s ...", self.model_path)

        # Workaround for transformers bug where Qwen2.5-VL sub-configs
        # are dicts instead of PretrainedConfig objects, causing
        # "'dict' object has no attribute 'to_dict'" errors.
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(config, "text_config") and isinstance(config.text_config, dict):
            config.text_config = PretrainedConfig.from_dict(config.text_config)
        if hasattr(config, "vision_config") and isinstance(config.vision_config, dict):
            config.vision_config = PretrainedConfig.from_dict(config.vision_config)

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def _unload_model(self):
        """Free GPU memory."""
        import torch

        del self._model, self._processor
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image: Image.Image) -> str:  # noqa: D401
        """OCR a single page image and return text."""
        import torch

        self._load_model()

        image = image.convert("RGB")
        image = _resize_if_needed(image, self.max_dimension)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                ],
            },
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)

        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = [
                out[len(inp) :]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            result = self._processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        finally:
            del inputs, output_ids
            if not self.keep_loaded:
                self._unload_model()

        return result
