"""
Command-line interface for justicetech-extract.

Usage examples::

    # Process a single PDF (full pipeline)
    justicetech process document.pdf --output result.json

    # Process a pre-OCR'd text file
    justicetech extract cleaned_text.txt --output result.json

    # Process a directory of text files
    justicetech extract-dir ./post_ocr/ --output ./results/

    # Regex-only (no API calls)
    justicetech extract cleaned_text.txt --regex-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_extract(args):
    """Extract from a single text file."""
    from justicetech_extract import extract_from_text
    from justicetech_extract.config import Settings

    settings = Settings(use_llm=not args.regex_only)

    text = Path(args.input).read_text(encoding="utf-8")
    filename = Path(args.input).stem

    result = extract_from_text(
        text,
        filename=filename,
        settings=settings,
        clean=not args.no_clean,
        reclassify=not args.regex_only,
    )

    output = result.info.model_dump(exclude={"raw_text"})

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))


def cmd_process(args):
    """Process a PDF end-to-end."""
    from justicetech_extract import process_pdf
    from justicetech_extract.config import Settings

    settings = Settings()

    result = process_pdf(args.input, settings=settings)

    output = result.info.model_dump(exclude={"raw_text"})

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))


def cmd_process_dir(args):
    """Process a directory of PDFs end-to-end → CSV + JSON."""
    import csv
    import time

    from justicetech_extract import extract_from_text
    from justicetech_extract.config import OCRBackend, Settings
    from justicetech_extract.ocr.pdf_convert import pdf_to_images

    settings = Settings()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF files in {input_dir}")

    if not pdfs:
        print("No PDFs found. Exiting.")
        return

    # --- Checkpoint / resume: skip already-processed files ---
    csv_path = output_dir / "extracted_court_info.csv"
    processed_files = set()
    fieldnames = None

    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if "source_file" in row:
                    processed_files.add(row["source_file"])
        if processed_files:
            print(f"Resuming: {len(processed_files)} files already processed, "
                  f"{len(pdfs) - len(processed_files)} remaining")

    # --- Create OCR backend ONCE, reuse for all files ---
    backend = None
    if settings.ocr_backend == OCRBackend.GEMINI:
        from justicetech_extract.ocr.gemini import GeminiOCR

        backend = GeminiOCR(
            model=settings.gemini_model,
            api_key=settings.effective_gemini_api_key,
            base_url=settings.effective_gemini_base_url,
        )
    elif settings.ocr_backend == OCRBackend.NANONETS:
        from justicetech_extract.ocr.nanonets import NanonetsOCR

        backend = NanonetsOCR(
            model_path=settings.ocr_model_path,
            device=settings.ocr_device,
            max_dimension=settings.ocr_max_dimension,
            keep_loaded=True,
        )
    else:
        print(f"Error: OCR backend '{settings.ocr_backend}' cannot process PDFs. "
              "Use 'gemini' or 'nanonets'.")
        return

    results = []
    errors = []
    skipped = 0
    total_time = 0.0

    for i, pdf in enumerate(pdfs, 1):
        # Skip already-processed files (checkpoint/resume)
        if pdf.name in processed_files:
            skipped += 1
            continue

        # Progress with ETA
        done = len(results) + len(errors)
        eta = ""
        if done > 0:
            avg = total_time / done
            remaining = len(pdfs) - i + 1 - skipped
            eta_sec = avg * remaining
            eta = f" — ETA: {eta_sec / 60:.0f}min" if eta_sec > 60 else f" — ETA: {eta_sec:.0f}s"

        print(f"\n[{i}/{len(pdfs)}] {pdf.name}{eta}")
        start = time.time()
        try:
            images = pdf_to_images(pdf, dpi=settings.pdf_dpi)
            raw_ocr = backend.process_document_images(images)

            result = extract_from_text(
                raw_ocr,
                filename=pdf.stem,
                settings=settings,
                clean=True,
                reclassify=True,
            )
            result.ocr_text = raw_ocr
            result.ocr_backend = settings.ocr_backend.value

            row = result.info.to_flat_dict()
            row["source_file"] = pdf.name
            results.append(row)

            elapsed = time.time() - start
            total_time += elapsed
            print(f"  OK ({elapsed:.1f}s) — case: {result.info.case_number}")

            # --- Incremental CSV: write each result immediately ---
            if fieldnames is None:
                fieldnames = list(row.keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
            else:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  FAILED ({elapsed:.1f}s) — {e}")
            errors.append({"file": pdf.name, "error": str(e)})

    # Save JSON (all results)
    if results:
        json_path = output_dir / "extracted_court_info.json"
        json_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved JSON to {json_path}")

    # Save errors
    if errors:
        err_path = output_dir / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"{len(errors)} failures saved to {err_path}")

    total = len(results) + len(errors) + skipped
    print(f"\nDone: {len(results)} OK, {len(errors)} failed, "
          f"{skipped} skipped (resumed), {total} total")


def cmd_extract_dir(args):
    """Process a directory of text files."""
    from justicetech_extract import extract_from_text
    from justicetech_extract.config import Settings

    settings = Settings(use_llm=not args.regex_only)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in {input_dir}")

    results = []
    for i, txt_file in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        result = extract_from_text(
            text,
            filename=txt_file.stem,
            settings=settings,
            clean=not args.no_clean,
            reclassify=not args.regex_only,
        )
        results.append(result.info.to_flat_dict())

    if not results:
        print("No results to save.")
        return

    # Save as JSON
    json_path = output_dir / "extracted_court_info.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {len(results)} results to {json_path}")

    # Save as CSV
    import csv

    csv_path = output_dir / "extracted_court_info.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="justicetech",
        description="Extract structured data from Franklin County court documents",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract from a text file")
    p_extract.add_argument("input", help="Path to OCR'd text file")
    p_extract.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    p_extract.add_argument("--regex-only", action="store_true", help="Skip LLM, regex only")
    p_extract.add_argument("--no-clean", action="store_true", help="Skip OCR text cleaning")
    p_extract.set_defaults(func=cmd_extract)

    # --- process ---
    p_process = subparsers.add_parser("process", help="Process a PDF end-to-end")
    p_process.add_argument("input", help="Path to PDF file")
    p_process.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    p_process.set_defaults(func=cmd_process)

    # --- process-dir ---
    p_pdir = subparsers.add_parser("process-dir", help="Process a directory of PDFs → CSV")
    p_pdir.add_argument("input", help="Directory with .pdf files")
    p_pdir.add_argument("-o", "--output", required=True, help="Output directory")
    p_pdir.set_defaults(func=cmd_process_dir)

    # --- extract-dir ---
    p_dir = subparsers.add_parser("extract-dir", help="Process a directory of text files")
    p_dir.add_argument("input", help="Directory with .txt files")
    p_dir.add_argument("-o", "--output", required=True, help="Output directory")
    p_dir.add_argument("--regex-only", action="store_true")
    p_dir.add_argument("--no-clean", action="store_true")
    p_dir.set_defaults(func=cmd_extract_dir)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception as e:
        logging.error("Fatal: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
