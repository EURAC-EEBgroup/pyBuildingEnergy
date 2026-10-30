"""ASHRAE 140-2023 Section 7.2.3 in-depth cases: inputs as built, and the sign of
the differences the cases are designed to isolate.

Section 7.2.3 has no acceptance ranges, so the simulation checks only assert the
direction of physically unambiguous differences (more infiltration raises the
heating need, more exterior absorptance raises the cooling need, ...).
"""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import ALTITUDE_FACTOR, SOLAR_DISTRIBUTION_TABLES  # noqa: E402
from in_depth_cases import (  # noqa: E402
    CASE_SPECS, DIFFERENCES, HC_AREA_M2, IN_DEPTH_CASES, build_in_depth_case,
)


def _surface(bui, name):
    return next(s for s in bui["building_surface"] if s["name"] == name)


def test_the_25_cases_of_section_7_2_3_are_defined():
    assert set(IN_DEPTH_CASES) == set(CASE_SPECS)
    assert len(IN_DEPTH_CASES) == 25
    for case in IN_DEPTH_CASES:
        build_in_depth_case(case)


def test_unsupported_case_and_mode_are_rejected():
    with pytest.raises(ValueError):
        build_in_depth_case(600)
    with pytest.raises(ValueError):
        build_in_depth_case(220, solar_distribution="other")


def test_differences_refer_to_known_cases():
    known = set(IN_DEPTH_CASES) | {600, 900}
    assert all(c in known and b in known for c, b, _ in DIFFERENCES)


@pytest.mark.parametrize("case", [c for c in IN_DEPTH_CASES if CASE_SPECS[c]["fill"] == "hc"])
def test_high_conductance_elements_replace_the_windows(case):
    bui = build_in_depth_case(case)
    assert not any(s["type"] == "transparent" for s in bui["building_surface"])
    hc = _surface(bui, "High-conductance wall elements")
    south = _surface(bui, "South wall")
    assert hc["area"] == HC_AREA_M2 and hc["type"] == "opaque"
    assert hc["construction_class"] != south["construction_class"]  # not aggregated with the wall
    assert hc["area"] + south["area"] == pytest.approx(21.6)
    # Table 7-36: R = 2 x 0.00305 + 0.192 m2K/W surface to surface.
    r_c = 1 / hc["u_value"] - 1 / (hc["convective_heat_transfer_coefficient_internal"]
                                    + hc["radiative_heat_transfer_coefficient_internal"]) \
        - 1 / (hc["convective_heat_transfer_coefficient_external"]
               + hc["radiative_heat_transfer_coefficient_external"])
    assert r_c == pytest.approx(0.198, abs=1e-3)


@pytest.mark.parametrize("case", [195, 395])
def test_solid_conduction_cases_have_a_full_lightweight_south_wall(case):
    bui = build_in_depth_case(case)
    assert _surface(bui, "South wall")["area"] == pytest.approx(21.6)
    assert all(s["name"] != "High-conductance wall elements" for s in bui["building_surface"])
    assert not any(s["type"] == "transparent" for s in bui["building_surface"])


@pytest.mark.parametrize("case,ach,gains", [
    (220, 0.0, 0.0), (230, 1.0, 0.0), (240, 0.0, 200.0), (400, 0.0, 0.0),
    (410, 0.5, 0.0), (420, 0.5, 200.0), (430, 0.5, 200.0), (440, 0.5, 200.0),
])
def test_infiltration_and_internal_gains(case, ach, gains):
    bui = build_in_depth_case(case)
    params = bui["building_parameters"]
    assert params["ventilation"]["components"][0]["air_changes_per_hour"] == \
        pytest.approx(ach * ALTITUDE_FACTOR)
    assert params["internal_gains"][1]["full_load"] * 48.0 == pytest.approx(gains)


def test_thermostat_20_20_in_the_200_series_and_20_27_in_the_400_series():
    for case, cooling in ((220, 20.0), (270, 20.0), (320, 27.0), (400, 27.0), (810, 27.0)):
        sp = build_in_depth_case(case)["building_parameters"]["temperature_setpoints"]
        assert sp["heating_setpoint"] == 20.0 and sp["cooling_setpoint"] == cooling


def test_exterior_absorptance_and_raised_floor():
    for case, expected in ((220, 0.1), (250, 0.9), (430, 0.6), (800, 0.6)):
        bui = build_in_depth_case(case)
        for name in ("North wall", "Roof"):
            assert _surface(bui, name)["solar_absorptance"] == expected
        assert _surface(bui, "Floor")["solar_absorptance"] == 0.0  # receives no solar (7.2.1.5.1)


def test_infrared_emittance_scales_the_radiative_coefficients():
    base = _surface(build_in_depth_case(220), "North wall")
    interior = _surface(build_in_depth_case(210), "North wall")
    exterior = _surface(build_in_depth_case(215), "North wall")
    assert interior["radiative_heat_transfer_coefficient_internal"] == pytest.approx(
        base["radiative_heat_transfer_coefficient_internal"] / 9.0)
    assert interior["radiative_heat_transfer_coefficient_external"] == \
        base["radiative_heat_transfer_coefficient_external"]
    assert exterior["radiative_heat_transfer_coefficient_external"] == pytest.approx(
        base["radiative_heat_transfer_coefficient_external"] / 9.0)
    # Tables 7-40 / 7-41 convective coefficients of walls.
    assert interior["convective_heat_transfer_coefficient_internal"] == 1.9
    assert exterior["convective_heat_transfer_coefficient_external"] == 11.8


def test_combined_coefficients_of_table_7_46():
    for case, interior, exterior in ((450, True, True), (460, True, False), (470, False, True)):
        wall = _surface(build_in_depth_case(case), "North wall")
        window = _surface(build_in_depth_case(case), "South window")
        if interior:
            assert wall["convective_heat_transfer_coefficient_internal"] == 1.8
            assert wall["radiative_heat_transfer_coefficient_internal"] == 0.0
            assert window["convective_heat_transfer_coefficient_internal"] == 4.5
        else:
            assert wall["radiative_heat_transfer_coefficient_internal"] > 0.0
        if exterior:
            assert wall["convective_heat_transfer_coefficient_external"] == 21.6
            assert wall["radiative_heat_transfer_coefficient_external"] == 0.0
            assert window["convective_heat_transfer_coefficient_external"] == 17.8
        else:
            assert wall["radiative_heat_transfer_coefficient_external"] > 0.0


@pytest.mark.parametrize("case,table", [
    (270, "270"), (280, "280"), (290, "270"), (300, "300"), (310, "300"), (320, "270"),
    (440, "280"), (810, "280"),
])
def test_interior_solar_distribution_tables(case, table):
    params = build_in_depth_case(case)["building_parameters"]
    assert params["interior_solar_distribution"] == SOLAR_DISTRIBUTION_TABLES[table]
    assert "interior_solar_distribution" not in \
        build_in_depth_case(case, solar_distribution="area")["building_parameters"]


def test_cases_without_windows_have_no_solar_distribution():
    for case in (195, 220, 250, 400, 800):
        assert "interior_solar_distribution" not in build_in_depth_case(case)["building_parameters"]


def test_window_geometry_and_shading():
    assert _surface(build_in_depth_case(270), "South window")["area"] == 12.0
    assert not _surface(build_in_depth_case(270), "South window").get("shading")
    assert _surface(build_in_depth_case(290), "South window")["shading"] is True
    east = _surface(build_in_depth_case(300), "East window")
    assert east["area"] == 6.0 and not east.get("shading")
    shaded = _surface(build_in_depth_case(310), "West window")
    assert shaded["shading"] is True and "side_fins" in shaded


def test_high_mass_cases_use_the_table_7_27_walls():
    light = _surface(build_in_depth_case(430), "North wall")["thermal_capacity"]
    heavy = _surface(build_in_depth_case(800), "North wall")["thermal_capacity"]
    assert heavy > 3 * light
    assert _surface(build_in_depth_case(800), "Roof")["thermal_capacity"] == \
        _surface(build_in_depth_case(430), "Roof")["thermal_capacity"]  # roof does not change


# --- simulation: direction of the differences the cases isolate ---------------------

@pytest.fixture(scope="module")
def loads():
    from run_in_depth import run_in_depth_case
    return {c: run_in_depth_case(c) for c in (220, 230, 240, 250, 270, 280, 400, 410)}


def test_zero_infiltration_case_runs(loads):
    assert loads[220]["annual_heating_MWh"] > 0.0


def test_more_infiltration_raises_the_heating_need(loads):
    assert loads[230]["annual_heating_MWh"] > loads[220]["annual_heating_MWh"]
    assert loads[410]["annual_heating_MWh"] > loads[400]["annual_heating_MWh"]


def test_internal_gains_lower_the_heating_need(loads):
    assert loads[240]["annual_heating_MWh"] < loads[220]["annual_heating_MWh"]


def test_higher_exterior_absorptance_raises_the_cooling_need(loads):
    assert loads[250]["annual_cooling_MWh"] > loads[220]["annual_cooling_MWh"]


def test_south_windows_raise_the_cooling_need(loads):
    assert loads[270]["annual_cooling_MWh"] > loads[220]["annual_cooling_MWh"]


def test_lower_interior_absorptance_lowers_the_cooling_need(loads):
    # More of the transmitted solar is lost through the windows (Table 7-44).
    assert loads[280]["annual_cooling_MWh"] < loads[270]["annual_cooling_MWh"]
