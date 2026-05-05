"""Tests for OCR text cleaning."""

from justicetech_extract.ocr.clean import clean_ocr_text


class TestOCRCleaning:
    def test_fixes_money_ocr_errors(self):
        text = "Defendant shall pay $1,O00.00 by 2/15/2024"
        cleaned, changes = clean_ocr_text(text)
        assert "$1,000.00" in cleaned
        assert len(changes) > 0

    def test_flags_invalid_dates(self):
        text = "Payment due November 31, 2024"
        cleaned, changes = clean_ocr_text(text)
        assert "[INVALID DATE]" in cleaned
        assert any("Invalid date" in c for c in changes)

    def test_valid_dates_untouched(self):
        text = "Payment due November 30, 2024"
        cleaned, changes = clean_ocr_text(text)
        assert "[INVALID DATE]" not in cleaned

    def test_normalizes_page_markers(self):
        text = "[--- PAGE 1 START ---]\nSome text\n[--- PAGE 1 END ---]"
        cleaned, _ = clean_ocr_text(text)
        assert "=== PAGE 1 ===" in cleaned
        assert "[--- PAGE" not in cleaned

    def test_collapses_whitespace(self):
        text = "Line 1\n\n\n\n\n\n\nLine 2"
        cleaned, _ = clean_ocr_text(text)
        assert "\n\n\n\n" not in cleaned

    def test_empty_input(self):
        cleaned, changes = clean_ocr_text("")
        assert cleaned == ""
        assert changes == []

    def test_no_false_positives_on_clean_text(self):
        text = "Defendant shall pay $2,538.52 on or before 2/15/2024 by 5:00 PM"
        cleaned, changes = clean_ocr_text(text)
        assert cleaned.strip() == text.strip()
        assert len(changes) == 0
