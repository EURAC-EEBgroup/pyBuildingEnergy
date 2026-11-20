"""Window aggregation and per-window shading weights (ISO 52016-1 Formula 69)."""

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.utils import ISO52016, _surface_frame_area_fraction


def _win(name, area, g, ffr, **extra):
    s = {
        "name": name, "type": "transparent", "area": area, "u_value": 1.4,
        "g_value": g, "frame_area_fraction": ffr, "width": 1.0, "height": area,
        "ISO52016_type_string": "W", "ISO52016_orientation_string": "SV",
        "sky_view_factor": 0.5, "orientation": {"azimuth": 180, "tilt": 90},
    }
    s.update(extra)
    return s


def test_aggregated_fixed_windows_keep_sum_of_g_a_one_minus_ffr():
    wins = [_win("A", 2.0, 0.7, 0.1), _win("B", 6.0, 0.3, 0.4)]
    out = ISO52016._aggregate_surfaces_by_direction({"building_surface": wins})
    (agg,) = [s for s in out["building_surface"] if s["type"] == "transparent"]
    expected = sum(w["g_value"] * w["area"] * (1 - w["frame_area_fraction"]) for w in wins)
    got = agg["g_value"] * agg["area"] * (1 - _surface_frame_area_fraction(agg))
    assert got == pytest.approx(expected, rel=1e-12)
    # the old product of the two area-weighted means was different
    g_mean = (0.7 * 2 + 0.3 * 6) / 8
    ffr_mean = (0.1 * 2 + 0.4 * 6) / 8
    assert g_mean * 8 * (1 - ffr_mean) != pytest.approx(expected, rel=1e-3)


def test_shading_factor_with_zero_base_g_is_not_forced_to_one():
    df = pd.DataFrame({"W_ad": [0.3, 0.3]})
    surface = {"name": "ad", "window_components": [
        {"name": "ad", "weight": 0.0, "area_weight": 2.0}]}
    f = ISO52016._surface_shading_factor_from_timeseries(df, 0, surface, {})
    assert f == pytest.approx(0.3)


def test_shading_factor_zero_g_group_uses_area_weights():
    df = pd.DataFrame({"W_a": [0.0], "W_b": [1.0]})
    surface = {"name": "a + b", "window_components": [
        {"name": "a", "weight": 0.0, "area_weight": 3.0},
        {"name": "b", "weight": 0.0, "area_weight": 1.0}]}
    f = ISO52016._surface_shading_factor_from_timeseries(df, 0, surface, {})
    assert f == pytest.approx(0.25)
