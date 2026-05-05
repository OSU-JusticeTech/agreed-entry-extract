"""
Court document extraction — regex and LLM-based.

The extraction module contains two complementary strategies:

- :mod:`~justicetech_extract.extraction.regex_extractor` — deterministic
  pattern matching optimized for Franklin County Municipal Court forms.
- :mod:`~justicetech_extract.extraction.llm_extractor` — LLM-based
  extraction via any OpenAI-compatible API.

The :func:`extract` convenience function runs both and cross-validates.
"""

from justicetech_extract.extraction.pipeline import extract

__all__ = ["extract"]
