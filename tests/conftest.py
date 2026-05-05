"""
Pytest configuration and shared fixtures for justicetech-extract tests.
"""

import json
from pathlib import Path

import pytest

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"


@pytest.fixture
def sample_text_pay_and_stay():
    """A synthetic document that should extract as 'Pay and Stay'."""
    return """\
FRANKLIN COUNTY MUNICIPAL COURT
COLUMBUS, OHIO

Case No. 2024 CVG 056254

Sunrise Properties LLC
Plaintiff,

v.

Jane Smith
Defendant(s)

AGREED ENTRY

The parties agree as follows:

1. Defendant shall pay $2,538.52 on or before 2/15/2024 by 5:00 PM
2. Defendant shall pay $700.00 on or before 3/1/2024
3. Defendant shall pay $700.00 on or before 4/1/2024

Defendant(s) hereby acknowledges receipt of the Notice to Vacate served
on or about January 5, 2024.

If any of the terms or conditions are breached by the Defendant(s),
Plaintiff may move directly for judgment and Defendant agrees to vacate
the premises immediately.

Case to be dismissed if all terms are met.

JUDGE: Magistrate Johnson
Date: 1/22/2024
"""


@pytest.fixture
def sample_text_pay_and_vacate():
    """A synthetic document that should extract as 'Pay and Vacate'."""
    return """\
FRANKLIN COUNTY MUNICIPAL COURT

Case No. 2024 CVG 012345

ABC Realty
Plaintiff,

v.

John Doe
Defendant

AGREED ENTRY

1. Defendant shall pay $1,500.00 by 3/15/2024.
2. Defendant shall vacate the premises on or before April 1, 2024.

Date: 2/10/2024
"""


@pytest.fixture
def sample_text_vacate_only():
    """A synthetic document that should extract as 'Vacate Only'."""
    return """\
FRANKLIN COUNTY MUNICIPAL COURT

Case No. 2024 CVG 099999

Metro Housing Corp
Plaintiff,

v.

Alex Rivera
Defendant

AGREED ENTRY

Defendant agrees to vacate the premises on or before March 31, 2024.
Defendant shall leave the premises in broom swept clean condition
and return all keys to the landlord.

Date: 3/1/2024
"""


@pytest.fixture
def sample_filename():
    """A structured filename for case_number and date extraction."""
    return "2024_CVG_056254_abc123_2024_CVG_056254_-_1_22_2024_-_DAGREED_-_CV_Docket_-_1_23_2024_cleaned.txt"


# ---------------------------------------------------------------------------
# Ground truth support
# ---------------------------------------------------------------------------

def load_ground_truth() -> list[dict]:
    """
    Load ground truth records from ``tests/ground_truth/cases.json``.

    Each record has:
    - ``filename``: name of the text file in ``tests/fixtures/``
    - ``expected``: dict of expected field values

    Returns an empty list if the file doesn't exist yet (allows tests
    to be written before ground truth is populated).
    """
    gt_file = GROUND_TRUTH_DIR / "cases.json"
    if not gt_file.exists():
        return []
    return json.loads(gt_file.read_text())


@pytest.fixture
def ground_truth_cases():
    """Load all ground truth test cases."""
    return load_ground_truth()
