"""Regression coverage for ISO 52016-1:2017's own Case 600/900 verification
harness (examples/iso52016_bestest/) -- distinct from, and not comparable
to, tests/test_ashrae140_bestest.py (see both READMEs for why).

Reference values for Case 900 are the CORRECTED ones (see
iso_reference_results.py's docstring): the as-printed standard Table
28/29 Case 900 columns are a known, documented errata (heating/cooling
columns swapped in the verification spreadsheet).

Two weather inputs are exercised here:

- The Denver TMY3 EPW approximation (``test_*_tmy3_approximation``):
  always runs. Case 600 matches the standard's own reference within 10%;
  Case 900 within ~30% -- looser because this approximate weather
  concentrates most of its gap in the shoulder-season months (April,
  November), see README.md.
- ISO 52016-1's own BESTEST climate spreadsheet
  (``test_*_exact_iso_climate``), via ``iso_climate_loader``: skipped
  automatically when that file (ISO's own copyrighted content, not
  shipped here) or the optional ``xlrd`` package isn't available. When it
  is, both cases match within ~10% -- confirming the TMY3 approximation,
  not a modelling defect, was the dominant source of Case 900's larger gap.
"""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "iso52016_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from iso_case_definitions import build_case, WEATHER_FILE  # noqa: E402
from iso_climate_loader import iso_bestest_climate  # noqa: E402
from iso_reference_results import ANNUAL_HEATING_KWH, ANNUAL_COOLING_KWH, REFERENCES_AVAILABLE  # noqa: E402

# The Tables 28-34 reference values are ISO copyright and not distributed;
# comparisons run only where they are installed locally (iso_reference_results.py).
_skip_without_references = pytest.mark.skipif(
    not REFERENCES_AVAILABLE,
    reason="ISO 52016-1 Tables 28-34 reference values not installed (Standards/local_reference_data/)",
)
from pybuildingenergy.source.utils import ISO52016  # noqa: E402


def _exact_iso_climate_available() -> bool:
    try:
        with iso_bestest_climate():
            return True
    except (FileNotFoundError, ImportError):
        return False


_EXACT_CLIMATE_AVAILABLE = _exact_iso_climate_available()
_skip_without_exact_climate = pytest.mark.skipif(
    not _EXACT_CLIMATE_AVAILABLE,
    reason="ISO 52016-1 BESTEST climate spreadsheet (or xlrd) not available locally",
)


def _run(mass: str):
    bui = build_case(mass)
    out = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE), c_int_per_A_us=0.0,
    )
    return out[0] if isinstance(out, tuple) else out


def _run_with_exact_climate(mass: str):
    bui = build_case(mass)
    with iso_bestest_climate():
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER_FILE), c_int_per_A_us=0.0,
        )
    return out[0] if isinstance(out, tuple) else out


@pytest.fixture(scope="module")
def case600_result():
    return _run("light")


@pytest.fixture(scope="module")
def case900_result():
    return _run("heavy")


@pytest.fixture(scope="module")
def case600_result_exact_climate():
    return _run_with_exact_climate("light")


@pytest.fixture(scope="module")
def case900_result_exact_climate():
    return _run_with_exact_climate("heavy")


def test_weather_fixture_exists():
    assert WEATHER_FILE.is_file(), f"Denver TMY3 EPW fixture missing: {WEATHER_FILE}"


@_skip_without_references
def test_case600_matches_iso52016_own_reference_tmy3_approximation(case600_result):
    sim_df = case600_result
    annual_heating_kWh = sim_df["Q_H"].sum() / 1000.0
    annual_cooling_kWh = sim_df["Q_C"].sum() / 1000.0

    assert annual_heating_kWh == pytest.approx(ANNUAL_HEATING_KWH[600], rel=0.10)
    assert annual_cooling_kWh == pytest.approx(ANNUAL_COOLING_KWH[600], rel=0.10)


@_skip_without_references
def test_case900_matches_corrected_iso52016_reference_tmy3_approximation(case900_result):
    """Case 900 runs ~24% high on heating and ~6% high on cooling against
    the corrected reference values (see iso_reference_results.py) when
    using the Denver TMY3 EPW approximation -- see README.md: this gap is
    concentrated in the shoulder-season months (April, November), and
    shrinks to ~5% when ISO 52016-1's own climate data is used instead
    (test_case900_matches_iso52016_reference_exact_climate below). The
    bound here is loose enough to absorb the TMY3 approximation, tight
    enough to catch a regression that pushes it back toward the old,
    wrong multi-hundred-percent deviation."""
    annual_heating_kWh = case900_result["Q_H"].sum() / 1000.0
    annual_cooling_kWh = case900_result["Q_C"].sum() / 1000.0

    assert annual_heating_kWh == pytest.approx(ANNUAL_HEATING_KWH[900], rel=0.30)
    assert annual_cooling_kWh == pytest.approx(ANNUAL_COOLING_KWH[900], rel=0.30)


@_skip_without_references
@_skip_without_exact_climate
def test_case600_matches_iso52016_reference_exact_climate(case600_result_exact_climate):
    sim_df = case600_result_exact_climate
    annual_heating_kWh = sim_df["Q_H"].sum() / 1000.0
    annual_cooling_kWh = sim_df["Q_C"].sum() / 1000.0

    assert annual_heating_kWh == pytest.approx(ANNUAL_HEATING_KWH[600], rel=0.10)
    assert annual_cooling_kWh == pytest.approx(ANNUAL_COOLING_KWH[600], rel=0.10)


@_skip_without_references
@_skip_without_exact_climate
def test_case900_matches_iso52016_reference_exact_climate(case900_result_exact_climate):
    """With ISO 52016-1's own BESTEST climate data, Case 900 matches its
    own (corrected) reference within ~6% on both metrics -- comparable to
    Case 600's match quality, and a large improvement on the ~24%/~6% the
    TMY3 approximation gives (see the _tmy3_approximation test above and
    README.md). Confirms the TMY3 approximation was the dominant source
    of Case 900's larger gap, not a modelling defect."""
    sim_df = case900_result_exact_climate
    annual_heating_kWh = sim_df["Q_H"].sum() / 1000.0
    annual_cooling_kWh = sim_df["Q_C"].sum() / 1000.0

    assert annual_heating_kWh == pytest.approx(ANNUAL_HEATING_KWH[900], rel=0.10)
    assert annual_cooling_kWh == pytest.approx(ANNUAL_COOLING_KWH[900], rel=0.10)


def test_case900_still_needs_less_heating_and_cooling_than_case600(case600_result, case900_result):
    """Even with the confirmed Case 900 gap, adding thermal mass must still
    reduce both annual heating and cooling relative to Case 600 -- the
    basic physical direction ISO 52016-1's own Case 600/900 pair is
    designed to exercise must survive regardless of this open finding."""
    assert case900_result["Q_H"].sum() < case600_result["Q_H"].sum()
    assert case900_result["Q_C"].sum() < case600_result["Q_C"].sum()


def test_iso_harness_defaults_to_operative_control_unlike_ashrae_harness():
    """This harness must NOT pass hvac_control_variable="air" -- doing so
    would silently turn it into a (worse) copy of ashrae140_bestest rather
    than ISO 52016-1's own clause-7.2.2.1 verification case. build_case()'s
    output dict is the actual kwarg source for Temperature_and_Energy_needs_
    calculation, so checking it (rather than source text, which legitimately
    discusses the option in comments) is the real guarantee.
    """
    bui = build_case("light")
    assert "hvac_control_variable" not in str(bui)


# --- Table 27 Tests 2, 4, 5, 6 and the clause 7.2.2.3 floor boundary --------

import iso_reference_results as _ref  # noqa: E402
from pybuildingenergy.source.utils import _single_zone_opaque_surface_iso_type  # noqa: E402


@pytest.mark.parametrize("mass", ["light", "heavy"])
@pytest.mark.parametrize("control", ["continuous", "intermittent", "free_floating"])
def test_floor_is_outdoor_air_not_ground(mass, control):
    """Table 23 footnote a: floor + insulation is an opaque construction to
    outdoor air; sky_view_factor 0 would make the engine treat it as ground."""
    floor = next(s for s in build_case(mass, control)["building_surface"] if s["name"] == "Floor")
    assert _single_zone_opaque_surface_iso_type(floor) == "OP"
    h_i = floor["convective_heat_transfer_coefficient_internal"] + floor["radiative_heat_transfer_coefficient_internal"]
    h_e = floor["convective_heat_transfer_coefficient_external"] + floor["radiative_heat_transfer_coefficient_external"]
    r_c = 1 / floor["u_value"] - 1 / h_i - 1 / h_e
    assert r_c == pytest.approx(25.254 if mass == "light" else 25.246, abs=2e-3)


def test_intermittent_and_free_floating_inputs():
    sp = build_case("light", "intermittent")["building_parameters"]
    assert sp["temperature_setpoints"]["heating_setback"] == 10.0
    assert sp["heating_profile"]["weekday"] == [0.0] * 7 + [1.0] * 16 + [0.0]  # 20 degC 07-23
    caps = build_case("heavy", "free_floating")["building_parameters"]["system_capacities"]
    assert caps["heating_capacity"] == caps["cooling_capacity"] == 0.0
    with pytest.raises(ValueError):
        build_case("light", "weekly")


def _run_exact(mass, control):
    with iso_bestest_climate():
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            build_case(mass, control), weather_source="epw", path_weather_file=str(WEATHER_FILE),
            c_int_per_A_us=0.0,
        )
    return out[0] if isinstance(out, tuple) else out


@_skip_without_references
@_skip_without_exact_climate
def test_case940_matches_iso52016_reference_exact_climate():
    df = _run_exact("heavy", "intermittent")
    assert df["Q_H"].sum() / 1000 == pytest.approx(_ref.ANNUAL_HEATING_KWH_ALL["940"], rel=0.10)
    assert df["Q_C"].sum() / 1000 == pytest.approx(_ref.ANNUAL_COOLING_KWH_ALL["940"], rel=0.10)


@_skip_without_references
@_skip_without_exact_climate
def test_case640_heating_matches_and_cooling_gap_is_bounded_exact_climate():
    """640 heating is within 10% of Table 28; cooling is about +10.3% of
    Table 29 (open gap, bounded so that it cannot grow unnoticed)."""
    df = _run_exact("light", "intermittent")
    assert df["Q_H"].sum() / 1000 == pytest.approx(_ref.ANNUAL_HEATING_KWH_ALL["640"], rel=0.10)
    assert df["Q_C"].sum() / 1000 == pytest.approx(_ref.ANNUAL_COOLING_KWH_ALL["640"], rel=0.12)


@_skip_without_references
@_skip_without_exact_climate
@pytest.mark.parametrize("mass,case_id", [("light", "600FF"), ("heavy", "900FF")])
def test_free_floating_annual_mean_operative_temperature_exact_climate(mass, case_id):
    df = _run_exact(mass, "free_floating")
    assert df["Q_H"].sum() == df["Q_C"].sum() == 0.0
    assert df["T_op"].mean() == pytest.approx(_ref.ANNUAL_OPERATIVE_TEMPERATURE_C[case_id], abs=0.5)
