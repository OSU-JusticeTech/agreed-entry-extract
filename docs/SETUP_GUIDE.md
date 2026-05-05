# JusticeTech Extract — Setup & Usage Guide

## 1. Overview

`justicetech-extract` is a Python package that processes Franklin County Municipal Court eviction document PDFs and extracts structured data into CSV files. The extracted fields include case number, plaintiff, defendant, outcome type, payment schedule, vacate dates, and additional agreement terms.

**Pipeline:**

PDF → Images → OCR (Gemini-3-flash-preview
) → Text Cleaning → Regex Extraction + LLM Extraction (GPT-4o) → Cross-Validation → Post-Processing Fixups → CSV/JSON Output

### Key Features

- **One-command batch processing:** PDFs in, CSV out
- **Hybrid extraction:** deterministic regex patterns cross-validated with LLM-based extraction for maximum accuracy
- **Checkpoint/resume:** if a job is interrupted, resubmit and it skips already-processed files
- **Incremental CSV:** results are written after each file, so partial results are preserved even if the job crashes
- **Known plaintiff correction:** automatically fixes common OCR misreads of repeat landlord names
- **Date validation:** flags suspicious payment dates that may be OCR errors
- **Configurable OCR backend:** Nanonets-OCR2-3B, optimized for court documents

---

## 2. Prerequisites

| Requirement | Details |
|-------------|---------|
| Package File | `justicetech-extract.tar.gz` (provided by project lead) |
| LLM API Key | Access key for the OSU LiteLLM Proxy |

No prior experience with Python packaging or OCR is required. This guide walks through every step.

---

## 3. Installation on OSC

### Step 1: Upload the Package

Open OSC OnDemand in your browser (https://ondemand.osc.edu). Navigate to **Files → Home Directory**. Click **Upload** and select the `justicetech-extract.tar.gz` file.

### Step 2: Open a Terminal

In OnDemand, go to **Clusters → Shell Access** (e.g., Ascend Shell Access).

### Step 3: Load Python and Create a Virtual Environment

> ⚠️ **IMPORTANT:** You must load Python 3.10 BEFORE creating the virtual environment. Otherwise the venv will use the system Python 3.9, which is not compatible with this package.

This is a one-time setup step. A virtual environment isolates the project dependencies from the system Python.

```bash
module load python/3.10
python --version  # Verify: should say Python 3.10.x
python -m venv ~/jt-env
source ~/jt-env/bin/activate
```

### Step 4: Extract the Package

```bash
cd ~
tar xzf justicetech-extract.tar.gz
```

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install './justicetech-extract[all]'
pip install transformers==4.49.0
```

> ⚠️ Note the quotes around `'./justicetech-extract[all]'` — they are required. Without quotes, the shell misinterprets the square brackets.

The `[all]` flag installs all required dependencies: PyTorch, Transformers, torchvision, PDF-to-image conversion, accelerate, and OCR model utilities. The second command pins the Transformers library to a version compatible with the Nanonets OCR model.

If you reconnect to OSC later, you only need these two commands before using the tool:

```bash
module load python/3.10
source ~/jt-env/bin/activate
```

---

## 4. Configuration

All configuration is controlled via a `.env` file. This file must be placed in the working directory where you run the pipeline (the directory you `cd` into before running the command).

> ⚠️ The file must be named `.env` (with a leading dot). A file named just `env` will not be detected. On OSC, files starting with a dot are hidden — use `ls -a` to see them.

### Create the `.env` File

Navigate to your pipeline directory and create the file:

```bash
cd /fs/scratch/PAS3267/pipeline
cat > .env << 'EOF'
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://litellmproxy.osu-ai.org
LLM_MODEL=GPT-4o
OCR_BACKEND=nanonets
OCR_DEVICE=cuda
EOF
```

Replace `your_api_key_here` with your actual API key for the OSU LiteLLM Proxy. Contact the project lead for access.

Verify the file was created correctly:

```bash
cat .env
```

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *(required)* | API key for the LLM endpoint |
| `LLM_BASE_URL` | `https://litellmproxy.osu-ai.org` | LLM endpoint URL |
| `LLM_MODEL` | `GPT-4o` | Model name (case-sensitive) |
| `OCR_BACKEND` | `external` | `nanonets` or `external` |
| `OCR_DEVICE` | `cuda` | `cuda` (GPU) or `cpu` |
| `PDF_DPI` | `300` | Resolution for PDF rasterization |

---

## 5. Running the Pipeline

### Step 1: Create a SLURM Job Script

Create a file named `run_pipeline.sh` in your pipeline directory:

```bash
#!/bin/bash
#SBATCH --job-name=justicetech
#SBATCH --account=PAS3267
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/jt_%j.log

module load python/3.10
source ~/jt-env/bin/activate

cd /fs/scratch/PAS3267/pipeline

justicetech process-dir ./pdfs -o ./results
```

Edit the paths to match your directory structure. Set `--time` based on the number of PDFs (approximately 30–60 seconds per document). For 1,000 PDFs, request at least 12 hours.

> ⚠️ The `cd` line is critical — it must point to the directory containing your `.env` file. The pipeline reads `.env` from the current working directory.

### Step 2: Prepare Directories

```bash
mkdir -p /fs/scratch/PAS3267/pipeline/logs
mkdir -p /fs/scratch/PAS3267/pipeline/results
```

Place your PDF files in the input directory (e.g., `/fs/scratch/PAS3267/pipeline/pdfs`).

### Step 3: Submit the Job

```bash
cd /fs/scratch/PAS3267/pipeline
sbatch run_pipeline.sh
```

### Alternative: Interactive GPU Session

For quick tests with a few files, you can request an interactive GPU session:

```bash
sinteractive -A PAS3267 -p nextgen -g 1 -t 01:00:00
```

Once you get a node:

```bash
module load python/3.10
source ~/jt-env/bin/activate
cd /fs/scratch/PAS3267/pipeline
justicetech process-dir ./pdfs -o ./results
```

---

## 6. Monitoring & Output

### Check Job Status

```bash
squeue -u your_username
```

`ST` column meanings: `PD` = Pending (waiting for GPU), `R` = Running, `CG` = Completing.

### Watch Progress in Real Time

```bash
tail -f /fs/scratch/PAS3267/pipeline/logs/jt_*.log
```

The log shows progress with estimated time remaining:

```
[1/100] document_001.pdf
  OK (34.2s) — case: 2024 CVG 000001
[2/100] document_002.pdf — ETA: 55min
  OK (31.8s) — case: 2024 CVG 000002
```

### Output Files

| File | Description |
|------|-------------|
| `extracted_court_info.csv` | All extracted data (written incrementally) |
| `extracted_court_info.json` | Same data in JSON format (written at end) |
| `errors.json` | Any files that failed processing |

### Checkpoint / Resume

If a job is interrupted (timeout, crash, etc.), simply resubmit with the same command. The pipeline reads the existing CSV and skips files that were already processed. No results are lost.

### Re-running from Scratch

If you want to reprocess all files (e.g., after a code update), delete the previous results first:

```bash
rm /fs/scratch/PAS3267/pipeline/results/*
sbatch run_pipeline.sh
```

---

## 7. Command Reference

| Command | Description |
|---------|-------------|
| `justicetech process-dir ./pdfs -o ./out` | Batch PDFs → CSV (requires GPU) |
| `justicetech extract-dir ./txts -o ./out` | Batch text files → CSV (no GPU) |
| `justicetech process doc.pdf -o out.json` | Single PDF → JSON |
| `justicetech extract doc.txt -o out.json` | Single text file → JSON |
| `justicetech extract doc.txt --regex-only` | Regex only, no LLM API calls |

---

## 8. Troubleshooting

### "module poppler not found"

This warning is harmless and can be safely ignored. The `pdf2image` package includes its own poppler support via pip.

### "requires a different Python: 3.9.x not in '>=3.10'"

Your virtual environment was created with the wrong Python version. You must load Python 3.10 before creating the venv:

```bash
deactivate
rm -rf ~/jt-env
module load python/3.10
python -m venv ~/jt-env
source ~/jt-env/bin/activate
```

### "cannot import AutoModelForVision2Seq"

Your `transformers` version is incompatible. Run:

```bash
pip install transformers==4.49.0
```

### "'dict' object has no attribute 'to_dict'"

This is a known bug in Transformers with Qwen2.5-VL models. Make sure you have the latest version of the package installed (v0.2.0+), which includes the workaround.

### "OCR backend 'external' cannot process PDFs"

Your `.env` file is either missing, in the wrong directory, or does not contain `OCR_BACKEND=nanonets`. Check:

```bash
cat .env       # View the file contents
ls -a          # Verify .env exists (not 'env')
```

The `.env` file must be in the directory you `cd` into in your SLURM script.

### "Resuming: N files already processed, 0 remaining"

The checkpoint/resume feature detected a previous run. To reprocess, delete the old results:

```bash
rm /fs/scratch/PAS3267/pipeline/results/*
```

### Job stuck in PD (Pending)

The GPU queue is busy. Check queue size with:

```bash
squeue -p nextgen | wc -l
```

Options: wait for the queue, try a different partition, or request an interactive session.

### CUDA Out of Memory

The Nanonets-OCR2-3B model requires approximately 8 GB of GPU memory. Ensure your SLURM job requests a GPU with sufficient memory (32 GB recommended via `--mem=32G`).

### LLM Extraction Errors

Verify that your `LLM_API_KEY` is valid and the LiteLLM proxy is accessible:

```bash
curl -H "Authorization: Bearer YOUR_KEY" https://litellmproxy.osu-ai.org/models
```

---

## 9. Architecture & Package Structure

```
justicetech-extract/
├── pyproject.toml              # Package metadata & dependencies
├── README.md
├── docs/                       # Documentation
│   └── SETUP_GUIDE.md
└── src/justicetech_extract/
    ├── __init__.py             # Public API
    ├── cli.py                  # Command-line interface
    ├── config.py               # Settings (.env support)
    ├── models.py               # Pydantic data models
    ├── ocr/
    │   ├── __init__.py
    │   ├── base.py             # Base OCR class
    │   ├── clean.py            # Post-OCR text cleaning
    │   ├── nanonets.py         # Nanonets-OCR2-3B backend
    │   └── pdf_convert.py      # PDF → images
    ├── extraction/
    │   ├── __init__.py
    │   ├── llm_extractor.py
    │   ├── pipeline.py
    │   └── regex_extractor.py
    └── postprocessing/
        ├── __init__.py
        ├── fixups.py
        └── reclassify.py
```

---

## 10. Contact & Support

- **Yuehua (Zoe) Duan** — duan.425@osu.edu

For bug reports or feature requests, contact Zoe via email.
