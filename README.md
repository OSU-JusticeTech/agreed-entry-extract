# justicetech-extract

Extract structured data from Franklin County Municipal Court eviction documents.

This package consolidates the JusticeTech extraction pipeline into an installable Python library that can process PDFs autonomously as they arrive — no Jupyter notebooks or manual intervention required.

## Installation

```bash
# Core package (regex extraction + LLM via OpenAI API)
pip install .

# With Nanonets OCR backend
pip install ".[nanonets]"

# With PDF-to-image support (requires system poppler-utils)
pip install ".[pdf]"

# Everything
pip install ".[all]"

# Development (includes pytest, ruff, mypy)
pip install ".[dev]"
```

### System dependencies

If using the PDF pipeline, install poppler:

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# On OSC, poppler is typically available via: module load poppler
```

## Quick Start

### Process pre-OCR'd text (most common)

```python
from justicetech_extract import extract_from_text

with open("cleaned_document.txt") as f:
    text = f.read()

result = extract_from_text(
    text,
    filename="2024_CVG_056254_..._cleaned.txt"
)

# Access structured fields
print(result.info.case_number)       # "2024 CVG 056254"
print(result.info.plaintiff)         # "Sunrise Properties LLC"
print(result.info.outcome_type)      # OutcomeType.PAY_AND_STAY
print(result.info.payment_schedule)  # [PaymentScheduleItem(...), ...]

# Get a flat dict for CSV/database
row = result.info.to_flat_dict()
```

### Process a PDF end-to-end

```python
from justicetech_extract import process_pdf

result = process_pdf("document.pdf")
print(result.info.to_flat_dict())
```

### Regex-only (no API calls)

```python
from justicetech_extract import extract_from_text
from justicetech_extract.config import Settings

settings = Settings(use_llm=False)
result = extract_from_text(text, filename="doc.txt", settings=settings)
```

### CLI

```bash
# Extract from a text file
justicetech extract cleaned_doc.txt -o result.json

# Regex only (no API calls)
justicetech extract cleaned_doc.txt --regex-only

# Process a PDF
justicetech process document.pdf -o result.json

# Batch process a directory
justicetech extract-dir ./post_ocr/ -o ./results/
```

## Configuration

All settings are controlled via environment variables or a `.env` file:

```bash
cp .env.example .env
# Edit .env with your values
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | API key for OpenAI-compatible endpoint |
| `LLM_BASE_URL` | `https://litellmproxy.osu-ai.org` | LLM endpoint URL |
| `LLM_MODEL` | `GPT-4o` | Model name (case-sensitive!) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `OCR_BACKEND` | `external` | `nanonets` or `external` |
| `OCR_MODEL_PATH` | `nanonets/Nanonets-OCR2-3B` | HuggingFace model path |
| `OCR_DEVICE` | `cuda` | `cuda` or `cpu` |
| `PDF_DPI` | `300` | DPI for PDF rasterization |

## Architecture

```
PDF ──→ [pdf_to_images] ──→ Images ──→ [OCR Backend] ──→ Raw Text
                                                              │
                                                    [clean_ocr_text]
                                                              │
                                                        Cleaned Text
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │                               │
                                      [RegexExtractor]              [LLMExtractor]
                                              │                               │
                                              └───────────┬───────────────────┘
                                                          │
                                                  [cross-validate]
                                                          │
                                                  [apply_fixups]
                                                          │
                                                  ExtractedCourtInfo
```

### Package structure

```
src/justicetech_extract/
├── __init__.py          # Public API: extract_from_text(), process_pdf()
├── config.py            # Settings via pydantic-settings (.env support)
├── models.py            # Pydantic data models
├── cli.py               # Command-line interface
├── ocr/
│   ├── base.py          # Abstract OCR backend interface
│   ├── nanonets.py      # Nanonets-OCR2-3B backend
│   ├── pdf_convert.py   # PDF → images via pdf2image
│   └── clean.py         # Post-OCR text cleaning
├── extraction/
│   ├── regex_extractor.py   # Deterministic pattern matching (v3.9)
│   ├── llm_extractor.py     # LLM-based extraction
│   └── pipeline.py          # Combined extraction + cross-validation
└── postprocessing/
    └── fixups.py        # Payment totals, case number normalization
```

## Testing

```bash
# Install dev dependencies
pip install ".[dev]"

# Run all tests
pytest

# Run fast tests only (no API calls)
pytest -m "not slow"

# Run with coverage
pytest --cov=justicetech_extract

# Run ground truth accuracy tests
pytest tests/test_ground_truth.py -v
```

### Ground truth tests

The test suite includes accuracy benchmarks against manually verified
extractions. To populate:

1. Copy validated text files into `tests/fixtures/`
2. Add expected results to `tests/ground_truth/cases.json`
3. Run `pytest tests/test_ground_truth.py -v`

The tests report per-field accuracy and fail if it drops below configured
thresholds (default 85% overall, 95% for case numbers). This lets you
quickly verify after LLM updates or provider switches.

## Migration from existing scripts

This package consolidates:

| Original file | Package module |
|----------------|----------------|
| `pdf_to_images.py` | `justicetech_extract.ocr.pdf_convert` |
| `nanonets_ocr.py` | `justicetech_extract.ocr.nanonets` |
| `clean_ocr.py` | `justicetech_extract.ocr.clean` |
| `court_document_extractor.py` (ImprovedExtractor) | `justicetech_extract.extraction.regex_extractor` |
| `court_document_extractor.py` (EnhancedCourtDocumentExtractor) | `justicetech_extract.extraction.pipeline` + `llm_extractor` |
| `court_extractor_osc.py` | `justicetech_extract.cli` (batch mode) |
| `Step3_*.ipynb` (payment fixups) | `justicetech_extract.postprocessing.fixups` |
| SLURM scripts | Not included — batch orchestration is external |

### What's NOT in the package

- **SLURM job scripts** — batch orchestration stays external
- **File transfer** (Globus) — handled by the processing pipeline
- **Dashboard** (Streamlit) — separate project, consumes this package's output

## Switching OCR providers

The OCR backend is abstracted behind `OCRBackendBase`. To switch from
Nanonets to a local/OSC API endpoint:

```python
from justicetech_extract.config import Settings

# Use a different model
settings = Settings(
    ocr_backend="nanonets",
    ocr_model_path="/path/to/local/model",
)
```

To add a completely new backend, subclass `OCRBackendBase` and implement
`process_image()`.

## License

MIT
