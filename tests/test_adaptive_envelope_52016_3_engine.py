"""EN ISO 52016-3 adaptive window in the single-zone EN ISO 52016-1 engine.

ASHRAE 140 Case 600 (Denver TMY3, south window 12 m2), the south window is made
an external roller blind with automated control in an office with heating and
cooling.  Annual energy needs [kWh] with the result of this implementation:

    =========================================  =========  =========
    case                                         Q_H        Q_C
    =========================================  =========  =========
    fixed window (no adaptive element)           4417.2     6251.0
    adaptive, extended state = retracted state   4417.1     6251.1   (must equal the fixed window)
    adaptive, roller blind extended g=0.10       4999.0     1681.8   (cooling -73 %, heating +13 %)
    =========================================  =========  =========

The third row has no external reference: it documents the current result and
the direction of the effect.  The agreement of the control logic with the EPB
Center case study is checked in ``test_adaptive_envelope_52016_3_epb_case_study``.
"""

import copy
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import build_case, WEATHER_FILE  # noqa: E402
from pybuildingenergy.source.utils import ISO52016  # noqa: E402

RETRACTED = {"g_value": 0.769, "u_value": 2.42, "tau_v": 0.8}
EXTENDED = {"g_value": 0.10, "u_value": 1.5, "tau_v": 0.02}

EXPECTED_KWH = {
    "fixed": (4417.2, 6251.0),
    "neutral": (4417.1, 6251.1),
    "adaptive": (4999.0, 1681.8),
}


def _adaptive(extended):
    return {
        "type": "roller_blind", "position": "external", "control_type": "automated",
        "space_category": "office", "system": "heating_cooling",
        "states": {"retracted": dict(RETRACTED), "extended": dict(extended)},
    }


def _run(adaptive=None):
    bui = build_case("light")
    for surface in bui["building_surface"]:
        if surface["name"] == "South window" and adaptive is not None:
            surface["adaptive_envelope"] = copy.deepcopy(adaptive)
    out = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
        c_int_per_A_us=0.0, hvac_control_variable="air",
    )
    return out[0]


@pytest.fixture(scope="module")
def results():
    return {
        "fixed": _run(),
        "neutral": _run(_adaptive(RETRACTED)),
        "adaptive": _run(_adaptive(EXTENDED)),
    }


def _kwh(df):
    return df["Q_H"].sum() / 1000.0, df["Q_C"].sum() / 1000.0


@pytest.mark.parametrize("case", ["fixed", "neutral", "adaptive"])
def test_annual_energy_needs(results, case):
    q_h, q_c = _kwh(results[case])
    exp_h, exp_c = EXPECTED_KWH[case]
    assert q_h == pytest.approx(exp_h, rel=0.005)
    assert q_c == pytest.approx(exp_c, rel=0.005)


def test_retracted_equals_extended_gives_the_fixed_window(results):
    fixed_h, fixed_c = _kwh(results["fixed"])
    neutral_h, neutral_c = _kwh(results["neutral"])
    assert neutral_h == pytest.approx(fixed_h, abs=0.5)
    assert neutral_c == pytest.approx(fixed_c, abs=0.5)


def test_adaptive_blind_cuts_cooling_and_raises_heating(results):
    fixed_h, fixed_c = _kwh(results["fixed"])
    adaptive_h, adaptive_c = _kwh(results["adaptive"])
    assert adaptive_c < 0.5 * fixed_c
    assert adaptive_h > fixed_h


def test_hourly_state_columns_are_returned(results):
    df = results["adaptive"]
    for suffix in ("modes", "state", "fraction", "ext_state", "g", "u", "tau_v"):
        assert f"adaptive_South window_{suffix}" in df.columns
    assert df["adaptive_South window_state"].between(1, 23).all()
    assert df["adaptive_South window_g"].between(0.10, 0.769).all()
    assert df["adaptive_South window_state"].nunique() > 5  # the state really changes
    assert not any(c.startswith("adaptive_") for c in results["fixed"].columns)


def test_adaptive_vs_reference_workflow(results):
    """6.10 / clause 8 workflow: KPIs, monthly report, statistics and conformity report."""
    from pybuildingenergy.source.adaptive_envelope_52016_3 import format_conformity_report
    from pybuildingenergy.source.adaptive_envelope_workflow_52016_3 import (
        compare_adaptive_with_reference,
    )

    bui = build_case("light")
    for surface in bui["building_surface"]:
        if surface["name"] == "South window":
            surface["adaptive_envelope"] = _adaptive(EXTENDED)
    area = float(bui["building"]["net_floor_area"])

    cmp = compare_adaptive_with_reference(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
        c_int_per_A_us=0.0, hvac_control_variable="air",
    )
    fixed_h, fixed_c = EXPECTED_KWH["fixed"]
    adapt_h, adapt_c = EXPECTED_KWH["adaptive"]
    # annual indicators of Table 6, adaptive - reference [kWh/m2]
    assert cmp.kpis["dQ_H_nd_kWh_m2"] == pytest.approx((adapt_h - fixed_h) / area, rel=0.02)
    assert cmp.kpis["dQ_C_nd_kWh_m2"] == pytest.approx((adapt_c - fixed_c) / area, rel=0.02)
    assert cmp.kpis["dQ_C_nd_pct"] == pytest.approx(100 * (adapt_c - fixed_c) / fixed_c, abs=1.0)
    assert cmp.kpis["dT_high_Kh"] < 0  # the blind reduces the overheating
    # monthly rows and weighted mean properties of the window
    assert list(cmp.monthly.index) == list(range(1, 13)) + ["year"]
    name = "South window"
    assert 0.10 <= cmp.monthly.loc["year", f"{name}:g_gl_mn"] <= 0.769
    assert 1.5 <= cmp.monthly.loc["year", f"{name}:U_w_mn"] <= 2.42
    # statistics and report
    assert sum(cmp.statistics[name]["occurrences"].values()) == len(cmp.adaptive_hourly)
    assert cmp.report["elements"][name]["annex_c_table"] == "C.7"
    assert "Not implemented" in format_conformity_report(cmp.report)
    assert not any(c.startswith("adaptive_") for c in cmp.reference_hourly.columns)
