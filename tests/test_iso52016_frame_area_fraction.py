"""The single-zone ISO 52016 path honours a window's ``frame_area_fraction``."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.utils import ISO52016
from tests.test_ventilation_boundary import (
    _legacy_building,
    _minimal_profile_df,
    _patched_profiles,
)


@contextmanager
def _weather_with_sun_on_south(n=24, t_out=10.0):
    """Minimal weather in which only the south facade receives irradiance."""
    idx = pd.date_range("2023-06-01", periods=n, freq="h")
    sim_df = pd.DataFrame(index=idx)
    sim_df["T2m"] = t_out
    sim_df["WS10m"] = 0.0
    for ori in ("HOR", "NV", "EV", "SV", "WV"):
        lit = ori == "SV"
        sim_df[f"I_sol_dif_{ori}"] = 100.0 if lit else 0.0
        sim_df[f"I_sol_dir_w_{ori}"] = 200.0 if lit else 0.0
        sim_df[f"I_sol_tot_{ori}"] = 300.0 if lit else 0.0
    mock_wb = MagicMock()
    mock_wb.simulation_df = sim_df
    mock_ci = MagicMock()
    mock_ci.sim_df = sim_df
    with patch.object(ISO52016, "Weather_data_bui", return_value=mock_wb), \
         patch("pybuildingenergy.source.utils.Calculation_ISO_52010", return_value=mock_ci):
        yield sim_df


def _south_window(name="window", area=4.0, **extra):
    surf = {
        "name": name,
        "type": "transparent",
        "area": area,
        "u_value": 1.2,
        "g_value": 0.6,
        "sky_view_factor": 0.5,
        "width": area / 2.0,
        "height": 2.0,
        "parapet": 1.0,
        "orientation": {"azimuth": 180.0, "tilt": 90.0},
        "name_adj_zone": None,
    }
    surf.update(extra)
    return surf


def _solar_gains(windows, control_mode="standard", n=24):
    bld = _legacy_building([{
        "name": "inf",
        "ventilation_type": "prescribed",
        "heat_transfer_coefficient_w_k": 50.0,
        "source_temperature_c": 10.0,
    }])
    bld["building_surface"].extend(windows)
    with _weather_with_sun_on_south(n=n), _patched_profiles(_minimal_profile_df(n)):
        result = ISO52016._single_zone_52016_engine(
            bld,
            control_mode=control_mode,
            weather_source="epw",
            path_weather_file=None,
            warmup_hours=0,
        )
    out = result[0] if isinstance(result, tuple) else result
    return out["Q_solar_gains"].to_numpy()


@pytest.mark.parametrize("control_mode", ["standard", "ahu_causal"])
@pytest.mark.parametrize("ffr", [0.0, 0.1, 0.4])
def test_window_solar_gain_scales_with_frame_area_fraction(ffr, control_mode):
    default = _solar_gains([_south_window()], control_mode=control_mode)
    given = _solar_gains([_south_window(frame_area_fraction=ffr)], control_mode=control_mode)

    assert np.all(default > 0.0)
    np.testing.assert_allclose(given, default * (1.0 - ffr) / (1.0 - 0.25), rtol=1e-9)


def test_aggregated_windows_keep_their_glazed_area():
    """Two windows merged into one keep sum((1 - F_fr,i) * A_i)."""
    split = _solar_gains([
        _south_window("w1", area=2.0, frame_area_fraction=0.1),
        _south_window("w2", area=6.0, frame_area_fraction=0.3),
    ])
    merged = _solar_gains([_south_window("w", area=8.0, frame_area_fraction=0.25)])
    np.testing.assert_allclose(split, merged, rtol=1e-9)


def test_aggregation_area_weights_frame_area_fraction():
    building_object = {
        "building_surface": [
            dict(_south_window("w1", area=2.0, frame_area_fraction=0.1),
                 ISO52016_type_string="W", ISO52016_orientation_string="SV"),
            dict(_south_window("w2", area=6.0, frame_area_fraction=0.3),
                 ISO52016_type_string="W", ISO52016_orientation_string="SV"),
        ]
    }
    window = ISO52016._aggregate_surfaces_by_direction(building_object)["building_surface"][0]
    assert window["frame_area_fraction"] == pytest.approx(0.25)


def test_aggregation_without_frame_area_fraction_keeps_default():
    building_object = {
        "building_surface": [
            dict(_south_window("w1"), ISO52016_type_string="W", ISO52016_orientation_string="SV"),
            dict(_south_window("w2"), ISO52016_type_string="W", ISO52016_orientation_string="SV"),
        ]
    }
    window = ISO52016._aggregate_surfaces_by_direction(building_object)["building_surface"][0]
    assert "frame_area_fraction" not in window
