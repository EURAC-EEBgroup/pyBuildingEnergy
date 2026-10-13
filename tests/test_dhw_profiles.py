import numpy as np
import pytest

import pybuildingenergy as pybui


def test_dhw_annex_b_profiles_are_public_and_normalized():
    raw = pybui.load_dhw_annex_b_table_b2_profiles()
    assert set(raw) == {
        "single_family_dwelling",
        "apartment_dwelling",
        "residential_home_elderly",
        "student_residence",
        "hospital",
    }

    fractions = pybui.dhw_annex_b_table_b2_hourly_fractions("hospital")
    assert list(fractions.columns) == ["Workday", "Weekend", "Holiday"]
    assert fractions.shape == (24, 3)
    assert np.allclose(fractions.sum(axis=0), 1.0)


def test_dhw_annex_b_profile_unknown_key_reports_available_profiles():
    with pytest.raises(ValueError, match="single_family_dwelling"):
        pybui.dhw_annex_b_table_b2_hourly_fractions("office")
