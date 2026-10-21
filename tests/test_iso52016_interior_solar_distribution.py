"""Optional ``interior_solar_distribution`` of the single-zone engine.

The input gives the fractions of the transmitted solar radiation absorbed by
the floor, ceiling, walls and lost back through the windows (the form of
ASHRAE 140 Tables 7-13, 7-43 ... 7-45).  Without it the engine keeps the
ISO 52016-1 area-weighted distribution.
"""

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import WEATHER_FILE, build_bestest_case  # noqa: E402
from pybuildingenergy.source.utils import (  # noqa: E402
    ISO52016,
    _interior_solar_distribution_factors,
    _solar_distribution_category,
)

AREAS = {"floor": 48.0, "ceiling": 48.0, "east": 16.2, "west": 16.2, "north": 21.6, "south": 21.6}


def _run(distribution=None):
    bui = build_bestest_case(600)
    if distribution is not None:
        bui["building_parameters"]["interior_solar_distribution"] = distribution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
            c_int_per_A_us=0.0, hvac_control_variable="air",
        )
    return out[0] if isinstance(out, tuple) else out


@pytest.fixture(scope="module")
def baseline():
    return _run()


def test_area_equivalent_fractions_reproduce_the_default(baseline):
    total = sum(AREAS.values())
    df = _run({**{k: v / total for k, v in AREAS.items()}, "lost": 0.0})
    for col in ("Q_H", "Q_C", "T_air", "T_op"):
        assert np.allclose(df[col].to_numpy(), baseline[col].to_numpy(), rtol=1e-9, atol=1e-6), col


def test_floor_heavy_distribution_changes_the_loads(baseline):
    df = _run({"floor": 0.9, "ceiling": 0.04, "east": 0.01, "west": 0.01, "north": 0.02,
               "south": 0.01, "lost": 0.01})
    assert df["Q_C"].sum() != pytest.approx(baseline["Q_C"].sum(), rel=0.01)
    assert df["Q_H"].sum() != pytest.approx(baseline["Q_H"].sum(), rel=0.01)


def test_lost_fraction_reduces_the_solar_gain_and_keeps_the_balance(baseline):
    lost = 0.25
    total = sum(AREAS.values())
    # Area-proportional shares of the radiation that is not lost: 1 - lost in total.
    df = _run({**{k: (1 - lost) * v / total for k, v in AREAS.items()}, "lost": lost})
    assert df["Q_solar_gains"].sum() == pytest.approx((1 - lost) * baseline["Q_solar_gains"].sum(), rel=1e-9)
    assert df["Q_C"].sum() < baseline["Q_C"].sum()


@pytest.mark.parametrize("cfg", [
    {"floor": 0.5, "lost": 0.1},                                  # does not add up to 1
    {"floor": 1.0, "bogus": 0.0},                                 # unknown key
    {"floor": 1.2, "ceiling": -0.2},                              # negative fraction
    {"floor": 0.5, "ceiling": 0.5, "lost": 1.0},                  # lost out of range
])
def test_invalid_distribution_is_rejected(cfg):
    surfaces = build_bestest_case(600)["building_surface"]
    bui = {"building_surface": surfaces, "building_parameters": {"interior_solar_distribution": cfg}}
    with pytest.raises(ValueError):
        _interior_solar_distribution_factors(bui, np.array([s["area"] for s in surfaces]),
                                             np.ones(len(surfaces), dtype=bool))


def test_categories_of_the_ashrae_cell():
    cats = {s["name"]: _solar_distribution_category(s) for s in build_bestest_case(600)["building_surface"]}
    assert cats == {"North wall": "north", "South wall": "south", "East wall": "east", "West wall": "west",
                    "Roof": "ceiling", "Floor": "floor", "South window": "south"}


def test_factors_integrate_to_one_over_the_interior_surfaces():
    bui = copy.deepcopy(build_bestest_case(600))
    surfaces = bui["building_surface"]
    areas = np.array([s["area"] for s in surfaces])
    bui["building_parameters"]["interior_solar_distribution"] = {
        "floor": 0.642, "ceiling": 0.167, "east": 0.038, "west": 0.038, "north": 0.053, "south": 0.027,
        "lost": 0.035}
    factors, lost = _interior_solar_distribution_factors(bui, areas, np.ones(len(surfaces), dtype=bool))
    assert lost == pytest.approx(0.035)
    assert float(np.sum(factors * areas)) == pytest.approx(1.0)
    south = [i for i, s in enumerate(surfaces) if s["name"] in ("South wall", "South window")]
    assert np.allclose(factors[south], factors[south[0]])  # same flux on a wall and its window
