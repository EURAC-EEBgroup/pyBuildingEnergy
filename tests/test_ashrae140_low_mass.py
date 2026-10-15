"""Input/regression coverage for the complete ASHRAE 140 low-mass suite."""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import LOW_MASS_ACCEPTANCE, build_bestest_case  # noqa: E402


@pytest.mark.parametrize("case_number", tuple(LOW_MASS_ACCEPTANCE))
def test_low_mass_case_has_complete_input(case_number):
    bui = build_bestest_case(case_number)
    assert bui["building"]["name"] == f"ASHRAE140_Case{case_number}"
    assert len(bui["building_surface"]) >= 7
    windows = [s for s in bui["building_surface"] if s["type"] == "transparent"]
    assert windows
    assert all(float(w["area"]) > 0.0 for w in windows)
    assert all("orientation" in w for w in windows)


def test_low_mass_acceptance_matrix_contains_all_600_series_cases():
    assert tuple(LOW_MASS_ACCEPTANCE) == (600, 610, 620, 630, 640, 650, 660, 670, 680, 685, 695)
