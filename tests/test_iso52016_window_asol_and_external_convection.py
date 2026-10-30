"""Coverage for two single-zone ISO 52016-1 engine fixes.

1. ``Solar_absorption_of_element``: ISO 52016-1 6.5.7.3 sets a_sol = 0 for
   transparent elements (their solar gain enters only through g_gl). The
   engine used to fall back to a_sol = g_value for a window without
   ``solar_absorptance`` (and to use any value given), double-counting solar.

2. ``external_convection_model`` ("doe2", "mowitt", "blast",
   "simplecombined") was accepted by the single-zone engine but never used:
   only the multizone path evaluated it, so the option was a silent no-op.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import build_case, WEATHER_FILE  # noqa: E402
from pybuildingenergy.source.utils import ISO52016  # noqa: E402


def _surfaces():
    return {
        "building_surface": [
            {"name": "wall", "type": "opaque", "ISO52016_type_string": "OP", "solar_absorptance": 0.6},
            {"name": "win_no_asol", "type": "transparent", "ISO52016_type_string": "W", "g_value": 0.7},
            {"name": "win_with_asol", "type": "transparent", "ISO52016_type_string": "W",
             "g_value": 0.7, "solar_absorptance": 0.5},
            {"name": "adj", "type": "opaque", "ISO52016_type_string": "ADJ", "solar_absorptance": 0.6},
        ]
    }


def test_window_solar_absorption_is_zero_per_iso_6_5_7_3():
    a = ISO52016.Solar_absorption_of_element(_surfaces()).a_sol_pli_eli
    np.testing.assert_allclose(a[0], [0.6, 0.0, 0.0, 0.0])
    assert not a[1:].any()  # only the external node absorbs


def _run(**kwargs):
    bui = build_case(kwargs.pop("mass", "light"))
    for key, value in kwargs.pop("building_parameters", {}).items():
        bui["building_parameters"][key] = value
    out = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
        c_int_per_A_us=0.0, hvac_control_variable="air", **kwargs,
    )
    df = out[0] if isinstance(out, tuple) else out
    return df["Q_H"].sum(), df["Q_C"].sum()


@pytest.fixture(scope="module")
def table_result():
    return _run()


def test_explicit_table_model_matches_default(table_result):
    assert _run(external_convection_model="table") == pytest.approx(table_result, rel=1e-12)


@pytest.mark.parametrize("model", ["doe2", "mowitt"])
def test_external_convection_model_changes_single_zone_result(table_result, model):
    heating, cooling = _run(external_convection_model=model)
    assert np.isfinite(heating) and np.isfinite(cooling)
    assert heating > 0 and cooling > 0
    rel_h = abs(heating - table_result[0]) / table_result[0]
    rel_c = abs(cooling - table_result[1]) / table_result[1]
    assert max(rel_h, rel_c) > 1e-3, f"{model}: external_convection_model had no effect"


def test_external_convection_model_read_from_building_parameters():
    via_kwarg = _run(external_convection_model="doe2")
    via_bui = _run(building_parameters={"external_convection_model": "doe2"})
    assert via_bui == pytest.approx(via_kwarg, rel=1e-12)
