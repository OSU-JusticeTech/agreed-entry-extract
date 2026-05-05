"""
Ground truth accuracy tests.

These tests compare extraction results against manually verified
ground truth records.  They allow a configurable error tolerance
so you can quickly verify after LLM updates or provider switches.

Ground truth format
-------------------
``tests/ground_truth/cases.json`` is a JSON array of objects::

    [
      {
        "filename": "2024_CVG_056254_..._cleaned.txt",
        "text_file": "fixtures/2024_CVG_056254_cleaned.txt",
        "expected": {
          "case_number": "2024 CVG 056254",
          "plaintiff": "Sunrise Properties LLC",
          "defendant": "Jane Smith",
          "outcome_type": "Pay and Stay",
          "payment_count": 3,
          "total_payment_sum": "3938.52",
          "mandatory_vacate_date": null
        }
      }
    ]

Populating ground truth
-----------------------
1. Copy a representative set of cleaned text files into ``tests/fixtures/``.
2. Run extraction manually and verify results.
3. Add verified results to ``tests/ground_truth/cases.json``.

The test suite will report per-field accuracy and fail if the overall
accuracy drops below the configured threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from justicetech_extract import extract_from_text
from justicetech_extract.config import Settings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum accuracy per field (0.0–1.0) for the test to pass.
# Set conservatively — tighten as the pipeline matures.
FIELD_ACCURACY_THRESHOLDS = {
    "case_number": 0.95,
    "outcome_type": 0.85,
    "plaintiff": 0.80,
    "defendant": 0.80,
    "payment_count": 0.85,
    "mandatory_vacate_date": 0.90,
}

# Overall accuracy across all fields
OVERALL_ACCURACY_THRESHOLD = 0.85

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"


def _load_ground_truth() -> list[dict]:
    gt_file = GROUND_TRUTH_DIR / "cases.json"
    if not gt_file.exists():
        return []
    return json.loads(gt_file.read_text())


def _field_matches(expected, actual) -> bool:
    """Compare expected vs actual with some tolerance."""
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False

    # String comparison (case-insensitive, whitespace-normalized)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()

    # Numeric comparison (allow small float differences)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 0.01

    return expected == actual


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroundTruthAccuracy:
    """
    Run extraction against ground truth and report accuracy.

    Skipped automatically if no ground truth file exists.
    """

    @pytest.fixture(autouse=True)
    def _load_cases(self):
        self.cases = _load_ground_truth()
        if not self.cases:
            pytest.skip(
                "No ground truth file found at tests/ground_truth/cases.json. "
                "See test docstring for format."
            )

    def test_per_field_accuracy(self):
        """Check that each field meets its accuracy threshold."""
        settings = Settings(use_llm=False)  # Regex-only for deterministic tests

        field_correct: dict[str, int] = {}
        field_total: dict[str, int] = {}

        for case in self.cases:
            text_file = FIXTURES_DIR / case.get("text_file", case["filename"] + ".txt")
            if not text_file.exists():
                continue

            text = text_file.read_text(encoding="utf-8")
            result = extract_from_text(
                text,
                filename=case["filename"],
                settings=settings,
                clean=True,
                reclassify=False,
            )
            info = result.info
            expected = case["expected"]

            for field, expected_val in expected.items():
                field_total.setdefault(field, 0)
                field_correct.setdefault(field, 0)
                field_total[field] += 1

                # Get actual value
                if field == "payment_count":
                    actual_val = len(info.payment_schedule)
                elif field == "outcome_type":
                    actual_val = info.outcome_type.value if info.outcome_type else None
                else:
                    actual_val = getattr(info, field, None)

                if _field_matches(expected_val, actual_val):
                    field_correct[field] += 1

        # Report and check thresholds
        print("\n=== Ground Truth Accuracy Report ===")
        all_correct = 0
        all_total = 0

        for field in sorted(field_total.keys()):
            total = field_total[field]
            correct = field_correct.get(field, 0)
            accuracy = correct / total if total > 0 else 0
            threshold = FIELD_ACCURACY_THRESHOLDS.get(field, 0.0)
            status = "PASS" if accuracy >= threshold else "FAIL"

            print(f"  {field:30s}  {correct}/{total}  ({accuracy:.0%})  [{status}]")

            all_correct += correct
            all_total += total

            if field in FIELD_ACCURACY_THRESHOLDS:
                assert accuracy >= threshold, (
                    f"Field '{field}' accuracy {accuracy:.0%} "
                    f"below threshold {threshold:.0%}"
                )

        overall = all_correct / all_total if all_total > 0 else 0
        print(f"\n  {'OVERALL':30s}  {all_correct}/{all_total}  ({overall:.0%})")

        if all_total == 0:
            pytest.skip("No ground truth text files found in tests/fixtures/")

        assert overall >= OVERALL_ACCURACY_THRESHOLD, (
            f"Overall accuracy {overall:.0%} below threshold "
            f"{OVERALL_ACCURACY_THRESHOLD:.0%}"
        )

    def test_no_regressions_on_case_number(self):
        """Case number extraction should be near-perfect (from filenames)."""
        settings = Settings(use_llm=False)

        for case in self.cases:
            text_file = FIXTURES_DIR / case.get("text_file", case["filename"] + ".txt")
            if not text_file.exists():
                continue

            expected_case = case["expected"].get("case_number")
            if expected_case is None:
                continue

            text = text_file.read_text(encoding="utf-8")
            result = extract_from_text(text, filename=case["filename"], settings=settings)

            assert result.info.case_number == expected_case, (
                f"Case number mismatch for {case['filename']}: "
                f"expected '{expected_case}', got '{result.info.case_number}'"
            )
