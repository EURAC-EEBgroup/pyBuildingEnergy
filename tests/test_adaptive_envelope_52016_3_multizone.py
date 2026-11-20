"""EN ISO 52016-3 adaptive windows in the multizone engine: one control context per zone."""

import copy
import json
import warnings
from pathlib import Path

import pytest

from pybuildingenergy.source.utils import ISO52016

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"
pytestmark = pytest.mark.skipif(not (CONFIG.exists() and WEATHER.exists()), reason="inputs missing")

WIN = {"ground": "Transparent south surface_Z_ground", "first": "Transparent south surface_Z_first"}
RETRACTED = {"g_value": 0.65, "u_value": 0.7, "tau_v": 0.8}       # = the window itself
EXTENDED = {"g_value": 0.10, "u_value": 0.6, "tau_v": 0.02}


def _adaptive(extended, space="office", control="automated"):
    return {"type": "roller_blind", "position": "external", "control_type": control,
            "space_category": space, "system": "heating_cooling",
            "states": {"retracted": dict(RETRACTED), "extended": dict(extended)}}


def _run(adaptive_by_window=None):
    bui = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
    for s in bui["building_surface"]:
        if adaptive_by_window and s["name"] in adaptive_by_window:
            s["adaptive_envelope"] = adaptive_by_window[s["name"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw", include_total=False)


def _q(annual, zone, col):
    return float(annual.set_index("zone").loc[zone, col])


@pytest.fixture(scope="module")
def runs():
    return {
        "fixed": _run(),
        "neutral": _run({WIN["ground"]: _adaptive(RETRACTED), WIN["first"]: _adaptive(RETRACTED)}),
        "first_only": _run({WIN["first"]: _adaptive(EXTENDED)}),
        "two_controls": _run({
            WIN["ground"]: _adaptive(EXTENDED, space="residential", control="manual"),
            WIN["first"]: _adaptive(EXTENDED, space="office", control="automated"),
        }),
    }


def test_no_adaptive_window_means_no_adaptive_context(runs):
    hourly, _ = runs["fixed"]
    assert not any(c.startswith("adaptive_") for c in hourly.columns)


def test_neutral_adaptive_windows_reproduce_the_fixed_windows(runs):
    _, a0 = runs["fixed"]
    _, a1 = runs["neutral"]
    for zone in ("Z_ground", "Z_first"):
        for col in ("Q_H_annual_kWh", "Q_C_annual_kWh"):
            assert _q(a1, zone, col) == pytest.approx(_q(a0, zone, col), rel=1e-3, abs=0.5)


def test_context_only_for_the_zone_with_adaptive_windows(runs):
    hourly, annual = runs["first_only"]
    assert f"adaptive_{WIN['first']}_state" in hourly.columns
    assert f"adaptive_{WIN['ground']}_state" not in hourly.columns
    _, a0 = runs["fixed"]
    d_first = _q(a0, "Z_first", "Q_C_annual_kWh") - _q(annual, "Z_first", "Q_C_annual_kWh")
    d_ground = _q(a0, "Z_ground", "Q_C_annual_kWh") - _q(annual, "Z_ground", "Q_C_annual_kWh")
    assert d_first > 0.0                      # the blind of Z_first cuts its cooling need
    assert abs(d_ground) < 0.5 * d_first      # Z_ground only feels it through the partition


def test_each_zone_follows_its_own_control_scenario(runs):
    hourly, _ = runs["two_controls"]
    s_ground = hourly[f"adaptive_{WIN['ground']}_state"]
    s_first = hourly[f"adaptive_{WIN['first']}_state"]
    assert s_ground.notna().all() and s_first.notna().all()
    assert (s_ground != s_first).mean() > 0.05      # different scenarios -> different states


def test_duplicate_adaptive_window_name_across_zones_is_rejected():
    bui = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])
    for s in bui["building_surface"]:
        if s["name"] in WIN.values():
            s["name"] = "Same window"
            s["adaptive_envelope"] = _adaptive(EXTENDED)
    with pytest.raises(ValueError, match="more than one zone"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw")


# ------------------------------------------- shading factor of each window (Formula 69)


def test_each_window_element_gets_its_own_shading_factor():
    """Minimal case: two windows with factors 0 and 1 must not both receive 0.5."""
    import numpy as np
    import pandas as pd

    sim = pd.DataFrame({"W_A": [0.0], "W_B": [1.0]})
    groups = {("Z", "SV"): [("A", 1.0), ("B", 1.0)]}
    f = ISO52016._surface_shading_factor_from_timeseries
    assert f(sim, 0, {"name": "A", "zone": "Z", "ISO52016_orientation_string": "SV"}, groups) == 0.0
    assert f(sim, 0, {"name": "B", "zone": "Z", "ISO52016_orientation_string": "SV"}, groups) == 1.0
    # an aggregated element averages only its own windows, weighted by g A (1 - F_fr)
    agg = {"name": "A + C", "window_components": [{"name": "A", "weight": 3.0}, {"name": "C", "weight": 1.0}]}
    assert f(sim, 0, agg, groups) == pytest.approx(0.25)


def _split_south_window(adaptive_b):
    """Z_first south window split in A (fixed, deep overhang) and B (unshaded, g = 0.3)."""
    bui = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
    out = []
    for s in bui["building_surface"]:
        if s["name"] != WIN["first"]:
            out.append(s)
            continue
        a = copy.deepcopy(s)
        a.update(name="South A shaded", area=3.75, width=2.5, height=1.5, shading_type="horizontal_overhang",
                 width_or_distance_of_shading_elements=0.0,
                 overhang_properties={"width_of_horizontal_overhangs": 3.0})
        b = copy.deepcopy(s)
        b.update(name="South B unshaded", area=3.75, width=2.5, height=1.5, g_value=0.3)
        for k in ("shading_type", "overhang_properties", "width_or_distance_of_shading_elements"):
            b.pop(k, None)                       # B has no shading element at all
        if adaptive_b:
            same = {"g_value": 0.3, "u_value": s["u_value"], "tau_v": 0.5}
            b["adaptive_envelope"] = {"type": "roller_blind", "position": "external",
                                      "control_type": "automated", "space_category": "office",
                                      "system": "heating_cooling",
                                      "states": {"retracted": dict(same), "extended": dict(same)}}
        out += [a, b]
    bui["building_surface"] = out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw", include_total=False)


def test_adaptive_and_fixed_windows_with_different_shading_multizone():
    """The adaptive window B is kept as its own element; the fixed window A is shaded.

    With both states equal to the fixed properties of B, the zone must behave exactly as with two
    fixed windows (where the shading of each window enters the aggregated element with its own
    weight).  With the zone/orientation average the unshaded B received half of A's shadow.
    """
    hourly_ad, annual_ad = _split_south_window(adaptive_b=True)
    hourly_fx, annual_fx = _split_south_window(adaptive_b=False)
    assert "adaptive_South B unshaded_state" in hourly_ad.columns
    # each element carries its own factor: the shaded fixed window is shaded, the unshaded
    # adaptive window is never shaded (with the zone/orientation average it got half of A's shadow)
    col_a = [c for c in hourly_ad.columns if c.startswith("F_sh_South A shaded")]
    assert len(col_a) == 1 and hourly_ad[col_a[0]].min() < 0.9   # A (with any fixed SV window)
    assert "F_sh_South B unshaded" not in hourly_ad.columns
    for zone in ("Z_ground", "Z_first"):
        for col in ("Q_H_annual_kWh", "Q_C_annual_kWh"):
            assert _q(annual_ad, zone, col) == pytest.approx(_q(annual_fx, zone, col), rel=2e-3, abs=0.5)


def test_multizone_adaptive_vs_reference_indicators_per_zone():
    """EN ISO 52016-3 6.10 / clause 8 on the multizone engine: same building and climate,
    adaptive window only in Z_first; indicators per zone and statistics per window."""
    import math

    from pybuildingenergy.source.adaptive_envelope_workflow_52016_3 import (
        compare_adaptive_with_reference_multizone,
    )

    bui = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
    for s in bui["building_surface"]:
        if s["name"] == WIN["first"]:
            s["adaptive_envelope"] = _adaptive(EXTENDED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare_adaptive_with_reference_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw")
    assert set(cmp.kpis) == {"Z_ground", "Z_first"}
    # the blind lowers the cooling need of its own zone ...
    assert cmp.kpis["Z_first"]["dQ_C_nd_kWh_m2"] < -1.0
    # ... and much less (through the floor) in the other zone
    assert abs(cmp.kpis["Z_ground"]["dQ_C_nd_kWh_m2"]) < abs(cmp.kpis["Z_first"]["dQ_C_nd_kWh_m2"])
    assert set(cmp.statistics) == {WIN["first"]}
    assert WIN["first"] in cmp.report["elements"]
    assert f"{WIN['first']}:g_mn" in cmp.monthly["Z_first"].columns or any(
        c.startswith(WIN["first"] + ":") for c in cmp.monthly["Z_first"].columns)
    assert not any(c.startswith(WIN["first"] + ":") for c in cmp.monthly["Z_ground"].columns)
    # comfort limits default to the zone set-points; no percentage of numerical noise
    zg = next(z for z in bui["zones"] if z["name"] == "Z_ground")
    assert cmp.comfort_limits["Z_ground"][0] == float(zg["heating_setpoint"])
    for k in cmp.kpis.values():
        assert math.isnan(k["dT_low_pct"]) or abs(k["dT_low_Kh"]) > 0.0 or k["dT_low_pct"] == 0.0
