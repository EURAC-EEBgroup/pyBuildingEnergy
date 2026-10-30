"""Regression coverage for the ASHRAE 140-2023 Case 600/900 harness.

Case 600/900 are run with ``hvac_control_variable="air"``: Section
7.2.1.13 requires the thermostat to sense only zone-air temperature, while
pybuildingenergy's own ISO 52016-1 *default* controls operative
temperature.

The only acceptance criteria used are the normative annual-load ranges of
Annex A3 (Tables A3-1 to A3-4). Case 600 heating/cooling and Case 900
heating fall inside them and are asserted as such, so a regression that
pushes them out is caught. Case 900 annual cooling is still above its
narrow Table A3-4 range; that open gap is asserted only against a
ceiling calibrated to the current result, so the test does not claim
compliance it does not have.
"""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import (  # noqa: E402
    ANNEX_A3_AVAILABLE, HIGH_MASS_ACCEPTANCE, LOW_MASS_ACCEPTANCE, WEATHER_FILE, build_case,
)

# The Annex A3 range values are copyright ASHRAE and not distributed; these
# checks run only where they are installed locally (see case_definitions.py).
needs_annex_a3 = pytest.mark.skipif(
    not ANNEX_A3_AVAILABLE,
    reason="ASHRAE 140-2023 Annex A3 ranges not installed (Standards/local_reference_data/)",
)
from pybuildingenergy.source.utils import ISO52016  # noqa: E402


def _run(mass: str, **kwargs):
    bui = build_case(mass)
    out = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
        c_int_per_A_us=0.0, **kwargs,
    )
    return out[0] if isinstance(out, tuple) else out


@pytest.fixture(scope="module")
def case600_result():
    return _run("light", hvac_control_variable="air")


@pytest.fixture(scope="module")
def case900_result():
    return _run("heavy", hvac_control_variable="air")


def test_weather_fixture_exists():
    assert WEATHER_FILE.is_file(), f"Denver TMY3 EPW fixture missing: {WEATHER_FILE}"


def test_case600_runs_and_has_both_heating_and_cooling(case600_result):
    sim_df = case600_result
    annual_heating_kWh = sim_df["Q_H"].sum() / 1000.0
    annual_cooling_kWh = sim_df["Q_C"].sum() / 1000.0
    assert annual_heating_kWh > 0.0
    assert annual_cooling_kWh > 0.0
    # Sanity bound: a single-zone 48 m2 test cell cannot need more than a
    # few hundred MWh/year; this just catches a gross unit/scale error.
    assert annual_heating_kWh < 50_000.0
    assert annual_cooling_kWh < 50_000.0


def test_case900_has_less_annual_load_than_case600(case600_result, case900_result):
    """Adding thermal mass at essentially the same U-value (Case 900 vs 600)
    must reduce both annual heating and cooling -- this is the core physical
    effect ASHRAE 140's Case 600/900 pair is designed to exercise."""
    heating_600 = case600_result["Q_H"].sum()
    heating_900 = case900_result["Q_H"].sum()
    cooling_600 = case600_result["Q_C"].sum()
    cooling_900 = case900_result["Q_C"].sum()

    assert heating_900 < 0.7 * heating_600
    assert cooling_900 < 0.7 * cooling_600


def test_case600_south_window_dominates_solar_gain(case600_result):
    sim_df = case600_result
    assert "Q_solar_gains" in sim_df.columns
    annual_solar_kWh = sim_df["Q_solar_gains"].sum() / 1000.0
    # 12 m2 of unshaded south glazing in a sunny climate (Denver): this
    # should be a large fraction of the annual cooling load, not negligible.
    annual_cooling_kWh = sim_df["Q_C"].sum() / 1000.0
    assert annual_solar_kWh > 0.3 * annual_cooling_kWh


A3_RANGES = {600: LOW_MASS_ACCEPTANCE[600], 900: HIGH_MASS_ACCEPTANCE[900]}


@needs_annex_a3
@pytest.mark.parametrize(
    "case_number,result_fixture,metric,column",
    [
        (600, "case600_result", "heating", "Q_H"),
        (600, "case600_result", "cooling", "Q_C"),
        (900, "case900_result", "heating", "Q_H"),
    ],
)
def test_annual_load_within_annex_a3_range(case_number, result_fixture, metric, column, request):
    sim_df = request.getfixturevalue(result_fixture)
    low, high = A3_RANGES[case_number][metric]
    assert low <= sim_df[column].sum() / 1e6 <= high


@needs_annex_a3
def test_case900_cooling_open_gap_does_not_grow(case900_result):
    """Case 900 cooling is above its Table A3-4 range (about 2.94 MWh today).
    Fail if it moves further away, without claiming compliance."""
    low, high = HIGH_MASS_ACCEPTANCE[900]["cooling"]
    assert low <= case900_result["Q_C"].sum() / 1e6 < 1.2 * high


def test_hvac_control_variable_defaults_to_operative_and_rejects_bad_value():
    bui = build_case("light")
    out_default = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE), c_int_per_A_us=0.0,
    )
    out_explicit = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE), c_int_per_A_us=0.0,
        hvac_control_variable="operative",
    )
    sim_default = out_default[0] if isinstance(out_default, tuple) else out_default
    sim_explicit = out_explicit[0] if isinstance(out_explicit, tuple) else out_explicit
    assert sim_default["Q_H"].sum() == pytest.approx(sim_explicit["Q_H"].sum())
    assert sim_default["Q_C"].sum() == pytest.approx(sim_explicit["Q_C"].sum())

    with pytest.raises(ValueError):
        ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
            hvac_control_variable="not-a-real-mode",
        )


def test_hvac_control_variable_air_reduces_case600_load_vs_operative(case600_result):
    """The core finding of this harness: EnergyPlus-style air-temperature
    control (what ASHRAE 140's reference programs use) draws substantially
    less heating/cooling than pybuildingenergy's operative-temperature
    default for this highly-glazed, low-mass test cell, because a hot (from
    direct solar) or cold (from the poorly-insulated window) interior
    surface temperature no longer has to be averaged into the setpoint."""
    sim_air = case600_result
    sim_operative = _run("light")

    assert sim_air["Q_H"].sum() < 0.9 * sim_operative["Q_H"].sum()
    assert sim_air["Q_C"].sum() < 0.9 * sim_operative["Q_C"].sum()


@needs_annex_a3
def test_case960_sunspace_back_zone_within_annex_a3(case900_result):
    """Two-zone Case 960 (multizone engine): back-zone loads minus Case 900
    within Tables A3-3/A3-4; the sun zone free-floats."""
    # Load by path: examples/iso52016_bestest/ has a module with the same name.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ashrae140_run_comparison", EXAMPLES_DIR / "run_comparison.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    run_case_960 = runner.run_case_960

    r960 = run_case_960()
    for metric, column in (("heating", "Q_H"), ("cooling", "Q_C")):
        low, high = HIGH_MASS_ACCEPTANCE[960][metric]
        delta = r960[f"annual_{metric}_MWh"] - case900_result[column].sum() / 1e6
        assert low <= delta <= high, (metric, delta)
    assert r960["sun_zone_T_min_C"] < 20.0 < 27.0 < r960["sun_zone_T_max_C"]


@needs_annex_a3
def test_case630_overhang_and_fins_within_annex_a3():
    """Figure 7-5/7-6 geometry (ISO 52016-1 F.4-F.6, F.22): 630 - 620 in Table A3-1/A3-2."""
    from case_definitions import build_bestest_case

    def annual(case_number):
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            build_bestest_case(case_number), weather_source="epw",
            path_weather_file=str(WEATHER_FILE), c_int_per_A_us=0.0, hvac_control_variable="air",
        )
        df = out[0] if isinstance(out, tuple) else out
        return df["Q_H"].sum() / 1e6, df["Q_C"].sum() / 1e6

    (h620, c620), (h630, c630) = annual(620), annual(630)
    lo_h, hi_h = LOW_MASS_ACCEPTANCE[630]["heating"]
    lo_c, hi_c = LOW_MASS_ACCEPTANCE[630]["cooling"]
    assert lo_h <= h630 - h620 <= hi_h
    assert lo_c <= c630 - c620 <= hi_c
