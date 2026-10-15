"""Input conformity of the ASHRAE 140 harness with Standard 140-2023.

These are fast checks (no simulation) that each input the harness builds
round-trips to the value the standard specifies, through the same
conventions the single-zone ISO 52016-1 engine applies to it.
"""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

import case_definitions as cd  # noqa: E402
from pybuildingenergy.source.utils import _single_zone_opaque_surface_iso_type  # noqa: E402

COEFFICIENT_SETS = tuple(cd.SURFACE_COEFFICIENTS)


def _surface(bui, name):
    return next(s for s in bui["building_surface"] if s["name"] == name)


def _construction_resistance(s):
    """R_c as Conductance_node_of_element recovers it from u_value."""
    h_i = s["convective_heat_transfer_coefficient_internal"] + s["radiative_heat_transfer_coefficient_internal"]
    h_e = s["convective_heat_transfer_coefficient_external"] + s["radiative_heat_transfer_coefficient_external"]
    return 1.0 / s["u_value"] - 1.0 / h_i - 1.0 / h_e


# Surface-to-surface resistances, Tables 7-3 / 7-28 (window: Table 7-11). The
# Case 900 floor uses the sum of its Table 7-27 layers (0.071 + 25.175): Table
# 7-28 prints 25.256, which does not match those layers.
@pytest.mark.parametrize("coefficients", COEFFICIENT_SETS)
@pytest.mark.parametrize("mass,name,r_expected", [
    ("light", "North wall", 1.789), ("light", "Roof", 2.992), ("light", "Floor", 25.254),
    ("heavy", "North wall", 1.797), ("heavy", "Floor", 25.246),
    ("light", "South window", 1.0 / 5.208 + 2 * 0.00305),
])
def test_construction_resistance_round_trips(mass, name, r_expected, coefficients):
    s = _surface(cd.build_case(mass, surface_coefficients=coefficients), name)
    assert _construction_resistance(s) == pytest.approx(r_expected, abs=2e-3)


# Total air-to-air U with the combined coefficients, Tables 7-3 and 7-11.
@pytest.mark.parametrize("name,u_expected", [
    ("North wall", 0.418), ("Roof", 0.276), ("Floor", 0.039), ("South window", 2.10),
])
def test_combined_coefficients_reproduce_table_7_3_u_values(name, u_expected):
    s = _surface(cd.build_case("light", surface_coefficients="ashrae_combined"), name)
    assert s["u_value"] == pytest.approx(u_expected, abs=2e-3)


def test_heat_capacities_match_table_7_3_and_7_28():
    # Table 7-3 / 7-28 "Total" heat capacity [kJ/K]: wall, floor, roof.
    light, heavy = cd.build_case("light"), cd.build_case("heavy")
    walls = ("North wall", "South wall", "East wall", "West wall")
    def total_kj(bui, names):
        return sum(_surface(bui, n)["thermal_capacity"] * _surface(bui, n)["area"] for n in names) / 1000
    assert total_kj(light, walls) == pytest.approx(924, rel=0.01)
    assert total_kj(light, ("Floor",)) == pytest.approx(936, rel=0.01)
    assert total_kj(light, ("Roof",)) == pytest.approx(872, rel=0.01)
    assert total_kj(heavy, walls) == pytest.approx(9232, rel=0.01)
    assert total_kj(heavy, ("Floor",)) == pytest.approx(5376, rel=0.01)


def test_raised_floor_is_outdoor_air_without_solar():
    floor = _surface(cd.build_case("heavy"), "Floor")
    assert _single_zone_opaque_surface_iso_type(floor) == "OP"  # not ground contact
    assert floor["solar_absorptance"] == 0.0
    assert floor["sky_view_factor"] < 1e-3


def test_infiltration_and_vent_fan_give_site_mass_flow():
    # Table 7-3 footnote f: UA_inf = 18.555 W/K at 1650 m.
    ua = cd.INFILTRATION_ACH * cd.VOLUME_M3 / 3600 * cd.RHO_CP_ENGINE
    assert ua == pytest.approx(18.555, rel=2e-3)
    # Table 7-16: 1700 m3/h at site density.
    vent = cd.build_bestest_case(650)["building_parameters"]["ventilation"]["components"][-1]
    q_site = vent["air_changes_per_hour"] * cd.VOLUME_M3 * cd.RHO_CP_ENGINE / cd.RHO_CP_SITE
    assert q_site == pytest.approx(1700.0, rel=1e-6)


def test_window_uses_2023_shgc_and_angle_dependent_fw():
    bui = cd.build_case("light")
    win = _surface(bui, "South window")
    assert win["g_value"] == 0.769  # Table 7-11
    fw_south = win["solar_transmittance_correction_factor"]
    # Below 1 (normal incidence) and above SHGC(70 deg)/SHGC_n: beam on a south
    # wall arrives at grazing angles in summer, so F_w can fall below the
    # hemispherical/normal ratio 0.670/0.769 = 0.871.
    assert 0.509 / 0.769 < fw_south < 1.0
    for azimuth in (90.0, 270.0):
        assert 0.509 / 0.769 < cd.effective_window_fw(cd.WINDOW_600, azimuth) < 1.0


def test_site_matches_table_a1_1():
    b = cd.build_case("light")["building"]
    assert (b["latitude"], b["longitude"]) == (39.833, -104.650)


def test_unknown_coefficient_set_is_rejected():
    with pytest.raises(ValueError):
        cd.build_case("light", surface_coefficients="nope")


def _windows(bui):
    return [s for s in bui["building_surface"] if s["type"] == "transparent"]


@pytest.mark.parametrize("case_number", [610, 910])
def test_south_overhang_geometry_figure_7_3(case_number):
    (win,) = _windows(cd.build_bestest_case(case_number))
    assert win["overhang_properties"]["width_of_horizontal_overhangs"] == 1.0  # depth
    assert win["width_or_distance_of_shading_elements"] == 0.5  # 2.7 - (0.2 + 2.0)
    assert "side_fins" not in win


@pytest.mark.parametrize("case_number", [630, 930])
def test_east_west_overhang_and_fins_figures_7_5_7_6(case_number):
    wins = _windows(cd.build_bestest_case(case_number))
    assert {w["orientation"]["azimuth"] for w in wins} == {90.0, 270.0}
    for w in wins:
        assert (w["width"], w["height"]) == (3.0, 2.0)
        assert w["overhang_properties"]["width_of_horizontal_overhangs"] == 1.0
        assert w["width_or_distance_of_shading_elements"] == 0.5
        assert w["side_fins"] == {
            "left": {"depth": 1.0, "distance": 0.0},
            "right": {"depth": 1.0, "distance": 0.0},
        }


def test_case_980_and_995_use_table_7_31_constructions():
    for case_number in (980, 995):
        bui = cd.build_bestest_case(case_number)
        assert _construction_resistance(_surface(bui, "North wall")) == pytest.approx(6.390, abs=2e-3)
        assert _construction_resistance(_surface(bui, "Roof")) == pytest.approx(10.198, abs=2e-3)
        assert _construction_resistance(_surface(bui, "Floor")) == pytest.approx(25.246, abs=2e-3)
    walls = ("North wall", "South wall", "East wall", "West wall")
    bui = cd.build_bestest_case(980)
    total = sum(_surface(bui, n)["thermal_capacity"] * _surface(bui, n)["area"] for n in walls) / 1000
    assert total == pytest.approx(9395, rel=0.01)  # Table 7-32


@pytest.mark.parametrize("case_number,cooling_setpoint", [(980, 27.0), (985, 20.0), (995, 20.0)])
def test_case_98x_thermostats(case_number, cooling_setpoint):
    sp = cd.build_bestest_case(case_number)["building_parameters"]["temperature_setpoints"]
    assert (sp["heating_setpoint"], sp["cooling_setpoint"]) == (20.0, cooling_setpoint)


def test_case_960_two_zone_geometry():
    bui = cd.build_case_960()
    by_zone = {}
    for s in bui["building_surface"]:
        by_zone.setdefault(s["zone"], []).append(s)
    back, sun = by_zone["back"], by_zone["sun"]
    common = _surface(bui, "Common wall")
    assert common["boundary"] == "INTERNAL" and common["adjacent_zone"] == "sun"
    assert common["area"] == pytest.approx(21.6)
    # Table 7-29: R = 0.392 m2K/W surface to surface, 0.2 m concrete.
    assert _construction_resistance(common) == pytest.approx(0.392, abs=2e-3)
    assert common["thermal_capacity"] == pytest.approx(0.20 * 1400 * 1000)
    assert not any(s["type"] == "transparent" for s in back)
    assert sum(s["area"] for s in sun if s["type"] == "transparent") == pytest.approx(12.0)
    assert _surface(bui, "Sun east wall")["area"] == pytest.approx(5.4)
    zones = {z["name"]: z for z in bui["zones"]}
    assert zones["sun"]["volume"] == pytest.approx(43.2)
    assert zones["sun"]["heating_capacity"] == zones["sun"]["cooling_capacity"] == 0.0
    assert zones["sun"]["internal_gains"][1]["full_load"] == 0.0
    assert zones["back"]["internal_gains"][1]["full_load"] * 48.0 == pytest.approx(200.0)
