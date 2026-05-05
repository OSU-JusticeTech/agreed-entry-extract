"""Tests for data models and post-processing fixups."""

import pytest

from justicetech_extract.models import (
    ExtractedCourtInfo,
    OutcomeType,
    PaymentScheduleItem,
    PaymentType,
    PipelineResult,
)
from justicetech_extract.postprocessing.fixups import (
    apply_fixups,
    fix_payment_total,
    flag_ocr_plaintiff,
    normalize_case_number,
)


class TestExtractedCourtInfo:
    def test_default_construction(self):
        info = ExtractedCourtInfo()
        assert info.case_number is None
        assert info.payment_schedule == []

    def test_to_flat_dict(self):
        info = ExtractedCourtInfo(
            case_number="2024 CVG 056254",
            outcome_type=OutcomeType.PAY_AND_STAY,
            payment_schedule=[
                PaymentScheduleItem(
                    payment_number=1,
                    due_date="2024-02-15",
                    payment_type=PaymentType.FIXED_AMOUNT,
                    amount="2538.52",
                ),
            ],
        )
        flat = info.to_flat_dict(max_payments=3)
        assert flat["case_number"] == "2024 CVG 056254"
        assert flat["outcome_type"] == "Pay and Stay"
        assert flat["payment_1_date"] == "2024-02-15"
        assert flat["payment_1_amount"] == "2538.52"
        assert flat["payment_2_date"] is None
        assert flat["payment_count"] == 1

    def test_to_flat_dict_monthly_rent(self):
        info = ExtractedCourtInfo(
            payment_schedule=[
                PaymentScheduleItem(
                    payment_number=1,
                    due_date="2024-03-01",
                    payment_type=PaymentType.MONTHLY_RENT,
                    amount="700",
                    month_rent="March",
                ),
            ],
        )
        flat = info.to_flat_dict()
        assert flat["payment_1_amount"] == "March $700"


class TestFixups:
    def test_fix_payment_total(self):
        info = ExtractedCourtInfo(
            payment_schedule=[
                PaymentScheduleItem(payment_number=1, payment_type=PaymentType.FIXED_AMOUNT, amount="100"),
                PaymentScheduleItem(payment_number=2, payment_type=PaymentType.FIXED_AMOUNT, amount="200.50"),
            ],
            total_payment_sum="0",
        )
        info = fix_payment_total(info)
        assert info.total_payment_sum == "300.50"

    def test_normalize_case_number_5812(self):
        info = ExtractedCourtInfo(case_number="5812 CVG 001234")
        info = normalize_case_number(info)
        assert info.case_number == "S812 CVG 001234"

    def test_normalize_case_number_normal(self):
        info = ExtractedCourtInfo(case_number="2024 CVG 056254")
        info = normalize_case_number(info)
        assert info.case_number == "2024 CVG 056254"

    def test_flag_ocr_plaintiff_timestamp(self):
        info = ExtractedCourtInfo(plaintiff="2024 JAN 22 PM 3:42")
        flag = flag_ocr_plaintiff(info)
        assert flag is not None
        assert "OCR artifact" in flag

    def test_flag_ocr_plaintiff_normal(self):
        info = ExtractedCourtInfo(plaintiff="Sunrise Properties LLC")
        flag = flag_ocr_plaintiff(info)
        assert flag is None

    def test_apply_all_fixups(self):
        info = ExtractedCourtInfo(
            case_number="5812 CVG 001234",
            payment_schedule=[
                PaymentScheduleItem(payment_number=1, payment_type=PaymentType.FIXED_AMOUNT, amount="500"),
            ],
        )
        info = apply_fixups(info)
        assert info.case_number == "S812 CVG 001234"
        assert info.total_payment_sum == "500.00"


class TestPipelineResult:
    def test_construction(self):
        result = PipelineResult(
            info=ExtractedCourtInfo(case_number="2024 CVG 056254"),
            ocr_backend="nanonets",
            ocr_changes=["Fixed money: $1,O00 -> $1,000"],
            errors=[],
        )
        assert result.info.case_number == "2024 CVG 056254"
        assert result.ocr_backend == "nanonets"
        assert len(result.ocr_changes) == 1
