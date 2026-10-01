"""Tilted surfaces (tilt not 0 or 90 deg) keep their azimuth when mapped to ISO 52016 orientations."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.utils import ISO52016
from tests.test_ventilation_boundary import (
    _legacy_building,
    _minimal_profile_df,
    _one_zone_building,
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


def _window(azimuth, tilt, **extra):
    surf = {
        "name": "roof_window",
        "type": "transparent",
        "area": 4.0,
        "u_value": 1.2,
        "g_value": 0.6,
        "sky_view_factor": 0.75,
        "width": 2.0,
        "height": 2.0,
        "parapet": 1.0,
        "orientation": {"azimuth": azimuth, "tilt": tilt},
    }
    surf.update(extra)
    return surf


TILTED_CASES = [
    (180.0, 60.0, "SV"),
    (0.0, 60.0, "NV"),
    (90.0, 75.0, "EV"),
    (275.0, 50.0, "WV"),
    (180.0, 30.0, "HOR"),
]


@pytest.mark.parametrize("control_mode", ["standard", "ahu_causal"])
@pytest.mark.parametrize("azimuth, tilt, expected", TILTED_CASES)
def test_single_zone_tilted_window_uses_irradiance_of_its_direction(azimuth, tilt, expected, control_mode):
    bld = _legacy_building([{
        "name": "inf",
        "ventilation_type": "prescribed",
        "heat_transfer_coefficient_w_k": 50.0,
        "source_temperature_c": 10.0,
    }])
    bld["building_surface"].append(_window(azimuth, tilt, name_adj_zone=None))
    n = 24
    with _weather_with_sun_on(expected, n=n), _patched_profiles(_minimal_profile_df(n)):
        result = ISO52016._single_zone_52016_engine(
            bld,
            control_mode=control_mode,
            weather_source="epw",
            path_weather_file=None,
            warmup_hours=0,
        )
    out = result[0] if isinstance(result, tuple) else result

    assert bld["building_surface"][-1]["ISO52016_orientation_string"] == expected
    assert np.all(out["Q_solar_gains"].to_numpy() > 0.0)


@pytest.mark.parametrize("azimuth, tilt, expected", TILTED_CASES)
def test_multizone_tilted_surface_orientation(azimuth, tilt, expected):
    bld = _one_zone_building()
    bld["building_surface"].append(_window(azimuth, tilt, boundary="OUTDOORS", zone="zone1"))
    with _weather_with_sun_on(expected, n=24):
        ISO52016.simulate_envelope_multizone_free_floating(
            building_object=bld,
            use_profiles=False,
            warmup_hours=0,
        )

    assert bld["building_surface"][-1]["ISO52016_orientation_string"] == expected
