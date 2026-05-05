"""
Configuration for JusticeTech extraction pipeline.

All external service credentials and tuning knobs are controlled here.
Settings are loaded from environment variables (optionally via a ``.env`` file).

Environment variables
---------------------
LLM_API_KEY          API key for the LLM endpoint (required for LLM extraction)
LLM_BASE_URL         Base URL of an OpenAI-compatible API
                     Default: ``https://litellmproxy.osu-ai.org``
LLM_MODEL            Model name to request
                     Default: ``GPT-4o``
LLM_TEMPERATURE      Sampling temperature (0 = deterministic)
                     Default: ``0.1``

OCR_BACKEND          Which OCR backend to use: ``gemini`` | ``nanonets`` | ``external``
                     Default: ``external`` (expects pre-OCR'd text)
OCR_MODEL_PATH       HuggingFace model path for Nanonets backend
                     Default: ``nanonets/Nanonets-OCR2-3B``
OCR_MAX_DIMENSION    Resize images to this max dimension before OCR
                     Default: ``1024``
OCR_DEVICE           Torch device for local OCR models (``cuda`` | ``cpu``)
                     Default: ``cuda``

GEMINI_API_KEY       API key for Gemini OCR via LiteLLM proxy.
                     Falls back to ``LLM_API_KEY`` if not set.
GEMINI_BASE_URL      LiteLLM proxy URL for Gemini OCR.
                     Falls back to ``LLM_BASE_URL`` if not set.
GEMINI_MODEL         Gemini model identifier in LiteLLM
                     Default: ``gemini-3-flash-preview``

PDF_DPI              DPI for PDF-to-image conversion
                     Default: ``300``
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRBackend(str, Enum):
    """Supported OCR backends."""

    GEMINI = "gemini"      # Gemini 3 Flash via LiteLLM proxy (recommended)
    NANONETS = "nanonets"  # Local Nanonets-OCR2-3B (legacy, requires GPU)
    EXTERNAL = "external"  # pre-OCR'd text provided by caller


class Settings(BaseSettings):
    """
    Pipeline configuration.  All fields can be set via environment variables
    (prefix-free) or via a ``.env`` file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----- LLM (GPT-4o / OpenAI-compatible) -----
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias="LLM_API_KEY",
        description="API key for the LLM endpoint",
    )
    llm_base_url: str = Field(
        default="https://litellmproxy.osu-ai.org",
        validation_alias="LLM_BASE_URL",
    )
    llm_model: str = Field(
        default="GPT-4o",
        validation_alias="LLM_MODEL",
    )
    llm_temperature: float = Field(
        default=0.1,
        validation_alias="LLM_TEMPERATURE",
    )

    # ----- OCR -----
    ocr_backend: OCRBackend = Field(
        default=OCRBackend.EXTERNAL,
        validation_alias="OCR_BACKEND",
    )
    ocr_model_path: str = Field(
        default="nanonets/Nanonets-OCR2-3B",
        validation_alias="OCR_MODEL_PATH",
    )
    ocr_max_dimension: int = Field(
        default=1024,
        validation_alias="OCR_MAX_DIMENSION",
    )
    ocr_device: str = Field(
        default="cuda",
        validation_alias="OCR_DEVICE",
    )

    # ----- Gemini OCR (recommended backend, via LiteLLM proxy) -----
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        description="LiteLLM proxy API key.  Falls back to LLM_API_KEY if not set.",
    )
    gemini_base_url: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_BASE_URL",
        description="LiteLLM proxy URL.  Falls back to LLM_BASE_URL if not set.",
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        validation_alias="GEMINI_MODEL",
        description="Model name as registered in LiteLLM proxy.",
    )

    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Gemini API key, falling back to the LLM key."""
        return self.gemini_api_key or self.llm_api_key

    @property
    def effective_gemini_base_url(self) -> str:
        """Gemini base URL, falling back to the LLM proxy URL."""
        return self.gemini_base_url or self.llm_base_url

    # ----- PDF conversion -----
    pdf_dpi: int = Field(
        default=300,
        validation_alias="PDF_DPI",
    )

    # ----- Extraction tuning -----
    use_llm: bool = Field(
        default=True,
        description="If False, use regex-only extraction (no API calls)",
    )
    use_regex_fallback: bool = Field(
        default=True,
        description="Enhance LLM results with regex cross-validation",
    )

    # Legacy aliases for backwards compatibility with existing scripts
    @property
    def openai_api_key(self) -> Optional[str]:
        return self.llm_api_key

    @property
    def base_url(self) -> str:
        return self.llm_base_url

    @property
    def model(self) -> str:
        return self.llm_model
