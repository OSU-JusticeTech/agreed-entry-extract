"""
Post-processing steps applied after initial extraction.

- :mod:`~justicetech_extract.postprocessing.reclassify` — re-classify vague
  outcome types (Unknown, Agreed Entry) via LLM
- :mod:`~justicetech_extract.postprocessing.fixups` — normalize case numbers,
  fix payment totals, split vacate dates, flag OCR artifacts
- :mod:`~justicetech_extract.postprocessing.confidence` — compute per-row
  confidence scores based on extraction quality signals
"""
