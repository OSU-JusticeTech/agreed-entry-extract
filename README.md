# justicetech-extract

> Structured data extraction from Franklin County Municipal Court eviction documents.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![OCR: Gemini 3 Flash](https://img.shields.io/badge/OCR-Gemini%203%20Flash%20Preview-orange?logo=google&logoColor=white)
![Extraction: GPT--4o](https://img.shields.io/badge/extraction-GPT--4o-teal?logo=openai&logoColor=white)

**justicetech-extract** is an installable Python package that processes eviction case PDFs autonomously — no Jupyter notebooks or manual intervention required. It uses [Google Gemini 3 Flash Preview](https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview) for vision-based OCR, then combines deterministic regex parsing with [GPT-4o](https://platform.openai.com/docs/models/gpt-4o) for structured field extraction, cross-validating results to maximize accuracy across dozens of document layouts.

Built as part of the JusticeTech initiative at [The Ohio State University](https://www.osu.edu/), this tool supports access-to-justice research by transforming unstructured court filings into analysis-ready structured data.

## Highlights

- **Vision-based OCR** — Gemini 3 Flash Preview processes document images directly, replacing traditional OCR pipelines with a multimodal LLM approach.
- **Hybrid extraction** — regex patterns handle predictable fields; GPT-4o handles ambiguous or variable-format content. Cross-validation catches errors from either approach.
- **End-to-end PDF pipeline** — PDF rasterization → Gemini OCR → text cleaning → extraction → post-processing, all in a single function call.
- **Pluggable OCR backends** — abstract interface makes it straightforward to swap Gemini for any alternative provider.
- **CLI included** — single-file and batch processing from the command line.

## Installation

```bash
# Core package (regex + LLM extraction)
pip install .

# With PDF-to-image conversion (requires system poppler-utils)
pip install ".[pdf]"

# Full installation
pip install ".[all]"

# Development (pytest, ruff, mypy)
pip install ".[dev]"
```

### System dependencies

PDF rasterization requires [poppler](https://poppler.freedesktop.org/):

```bash
# Ubuntu / Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Ohio Supercomputer Center (OSC)
module load poppler
```

## Quick Start

### Process a PDF end-to-end

```python
from justicetech_extract import process_pdf

result = process_pdf("document.pdf")
print(result.info.to_flat_dict())
```

### Regex-only mode (no API calls)

```python
from justicetech_extract import extract_from_text
from justicetech_extract.config import Settings

settings = Settings(use_llm=False)
result = extract_from_text(text, filename="doc.txt", settings=settings)
```

### Command-line interface

```bash
# Single file
justicetech extract cleaned_doc.txt -o result.json

# Regex only
justicetech extract cleaned_doc.txt --regex-only

# PDF end-to-end
justicetech process document.pdf -o result.json

# Batch directory
justicetech process-dir ./sample/ -o ./results/
```

## Configuration

All settings are controlled via environment variables or a `.env` file:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OCR_BACKEND` | `gemini` | Vision model used for OCR |
| `LLM_API_KEY` | — | API key for the extraction LLM endpoint |
| `LLM_BASE_URL` | `https://litellmproxy.osu-ai.org` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `GPT-4o` | Model used for structured extraction |

## Architecture

```
PDF ──→ pdf_to_images ──→ Images ──→ Gemini 3 Flash (OCR) ──→ Raw Text
                                                                  │
                                                           clean_ocr_text
                                                                  │
                                                            Cleaned Text
                                                                  │
                                                  ┌───────────────┴───────────────┐
                                                  │                               │
                                           RegexExtractor                  LLMExtractor
                                           (deterministic)                   (GPT-4o)
                                                  │                               │
                                                  └───────────┬───────────────────┘
                                                              │
                                                       cross-validate
                                                              │
                                                        apply_fixups
                                                              │
                                                     ExtractedCourtInfo
```

**OCR stage.** PDFs are rasterized to images, then processed by Gemini 3 Flash Preview using its native vision capabilities. This approach handles poor scans, stamps, and handwritten annotations more robustly than traditional OCR engines.

**Extraction stage.** The **RegexExtractor** uses ~60 compiled patterns to extract well-structured fields (case numbers, dates, addresses). The **LLMExtractor** sends cleaned text to GPT-4o for free-text reasoning — outcome classification, payment schedule parsing, and judgment interpretation. Cross-validation reconciles disagreements, preferring regex for fields where patterns are reliable and GPT-4o output for fields requiring contextual understanding.

### Out of scope

The following remain external to the package by design:

- **SLURM job scripts** — batch orchestration is deployment-specific.
- **File transfer** (Globus) — handled upstream in the processing pipeline.
- **Dashboard** (Streamlit) — separate project that consumes this package's output.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

```bash
git clone [https://github.com/OSU-JusticeTech/justicetech-extract.git](https://github.com/OSU-JusticeTech/agreed-entry-extract)
cd justicetech-extract
pip install ".[dev]"

# Lint
ruff check src/
mypy src/

# Test
pytest
```

## Acknowledgments

This project is developed at the[Translational Data Analytics Institute (TDAI)](https://tdai.osu.edu/) at The Ohio State University as part of the JusticeTech initiative, in collaboration with the Moritz College of Law. The pipeline processes eviction filings from the Franklin County Municipal Court to support research on housing stability and access to justice.

## License

[MIT](LICENSE)
