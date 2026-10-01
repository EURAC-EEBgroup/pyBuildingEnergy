"""Non-cardinal azimuths in the single-zone ISO 52016 path snap to the nearest cardinal direction."""

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

ORIENTATIONS = ("HOR", "NV", "EV", "SV", "WV")


@contextmanager
def _weather_with_sun_on(label, n=24, t_out=10.0):
    """Minimal weather in which only the `label` orientation receives irradiance."""
    idx = pd.date_range("2023-06-01", periods=n, freq="h")
    sim_df = pd.DataFrame(index=idx)
    sim_df["T2m"] = t_out
    sim_df["WS10m"] = 0.0
    for ori in ORIENTATIONS:
        lit = ori == label
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


def _building_with_window(azimuth, tilt=90.0):
    bld = _legacy_building([{
        "name": "inf",
        "ventilation_type": "prescribed",
        "heat_transfer_coefficient_w_k": 50.0,
        "source_temperature_c": 10.0,
    }])
    bld["building_surface"].append({
        "name": "window",
        "type": "transparent",
        "area": 4.0,
        "u_value": 1.2,
        "g_value": 0.6,
        "sky_view_factor": 0.5,
        "width": 2.0,
        "height": 2.0,
        "parapet": 1.0,
        "orientation": {"azimuth": azimuth, "tilt": tilt},
        "name_adj_zone": None,
    })
    return bld


def _solar_gains(bld, label, control_mode="standard", n=24):
    with _weather_with_sun_on(label, n=n), _patched_profiles(_minimal_profile_df(n)):
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
@pytest.mark.parametrize(
    "azimuth, expected",
    [
        (179.0, "SV"),
        (181.0, "SV"),
        (269.0, "WV"),
        (350.0, "NV"),
        (10.0, "NV"),
        (46.0, "EV"),
        (-100.0, "WV"),
    ],
)
def test_window_gets_irradiance_of_nearest_cardinal_direction(azimuth, expected, control_mode):
    bld = _building_with_window(azimuth)
    q_sol = _solar_gains(bld, expected, control_mode=control_mode)

    assert bld["building_surface"][-1]["ISO52016_orientation_string"] == expected
    assert np.all(q_sol > 0.0)
