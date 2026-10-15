"""Run ASHRAE 140-2023 Section 7 cases through pybuildingenergy.

Annual loads are checked only against the normative ranges in Annex A3 of
ASHRAE Standard 140-2023 (Tables A3-1 to A3-4), read from a local file that
is not distributed (see case_definitions.py); the range values are never
printed. Peak loads are reported for diagnostics because
Annex A3 does not define individual peak-load acceptance ranges for Cases
600 and 900.

Usage: python examples/ashrae140_bestest/run_comparison.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import (  # noqa: E402
    ANNEX_A3_AVAILABLE,
    HIGH_MASS_ACCEPTANCE,
    annex_a3_table,
    LOW_MASS_ACCEPTANCE,
    WEATHER_FILE,
    build_bestest_case,
    build_case,
    build_case_960,
)
from pybuildingenergy.source.utils import ISO52016  # noqa: E402


def run_case_960(surface_coefficients: str = "ashrae_convective") -> dict:
    """Case 960 through the multizone engine: back-zone loads, sun-zone temperatures."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.simulate_envelope_multizone_free_floating(
            build_case_960(surface_coefficients), path_weather_file=str(WEATHER_FILE),
            weather_source="epw", c_int_per_A_us=0.0, hvac_control_variable="air",
        )
    sim_df = out[0] if isinstance(out, tuple) else out
    q = sim_df["Q_HVAC_back"]  # +heating, -cooling [W]
    t_sun = sim_df["T_air_sun"]
    return {
        "annual_heating_MWh": q.clip(lower=0.0).sum() / 1e6,
        "annual_cooling_MWh": -q.clip(upper=0.0).sum() / 1e6,
        "peak_heating_kW": q.max() / 1000.0,
        "peak_cooling_kW": -q.min() / 1000.0,
        "sun_zone_T_mean_C": t_sun.mean(),
        "sun_zone_T_min_C": t_sun.min(),
        "sun_zone_T_max_C": t_sun.max(),
    }


def run_case(mass: str = "light", case_number: int | None = None,
             surface_coefficients: str = "ashrae_convective") -> dict:
    if case_number == 960:
        return run_case_960(surface_coefficients)
    bui = (build_case(mass, surface_coefficients=surface_coefficients) if case_number is None
           else build_bestest_case(case_number, surface_coefficients=surface_coefficients))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
            # Case 600/900 are an empty test cell; no furniture/air lumped mass.
            c_int_per_A_us=0.0,
            # Section 7.2.1.13 requires the thermostat to sense only zone-air
            # temperature.
            hvac_control_variable="air",
        )
    sim_df = out[0] if isinstance(out, tuple) else out
    return {
        "annual_heating_MWh": sim_df["Q_H"].sum() / 1e6,
        "annual_cooling_MWh": sim_df["Q_C"].sum() / 1e6,
        "peak_heating_kW": sim_df["Q_H"].max() / 1000.0,
        "peak_cooling_kW": sim_df["Q_C"].max() / 1000.0,
    }


FREE_FLOAT_CASES = (600, 650, 680, 900, 950, 980)  # Section 7.2.2.3


def run_free_float_case(case_number: int = 900,
                        surface_coefficients: str = "ashrae_convective") -> dict:
    """Run a free-floating Section 7 case (600FF, 650FF, 680FF, 900FF, 950FF,
    980FF) without mechanical HVAC.

    Standard 140-2023 Section 7.2.2.3: each case is identical to its base
    case except that there is no heating or cooling; 650FF/950FF keep the
    night vent-fan schedule. The returned values are zone-air temperatures
    only (Sections 7.3.6-7.3.7).
    """
    if case_number not in FREE_FLOAT_CASES:
        raise ValueError(f"Free-float cases are {FREE_FLOAT_CASES}")
    bui = build_bestest_case(case_number, surface_coefficients=surface_coefficients)
    bui["building_parameters"]["system_capacities"].update({
        "heating_capacity": 0.0,
        "cooling_capacity": 0.0,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
            c_int_per_A_us=0.0, hvac_control_variable="air",
        )
    sim_df = out[0] if isinstance(out, tuple) else out
    temperatures = sim_df["T_air"]
    minimum_time = temperatures.idxmin()
    maximum_time = temperatures.idxmax()
    february_first = temperatures[
        (temperatures.index.month == 2) & (temperatures.index.day == 1)
    ]
    bins = np.arange(-50.0, 99.0, 1.0)
    frequencies, _ = np.histogram(temperatures.to_numpy(), bins=bins)
    return {
        "annual_mean_T_air_C": float(temperatures.mean()),
        "annual_min_T_air_C": float(temperatures.min()),
        "annual_min_time": str(minimum_time),
        "annual_max_T_air_C": float(temperatures.max()),
        "annual_max_time": str(maximum_time),
        "february_1_hourly_T_air_C": [float(value) for value in february_first],
        "temperature_bin_frequencies": {
            int(lower): int(count)
            for lower, count in zip(bins[:-1], frequencies)
            if count
        },
        "annual_heating_Wh": float(sim_df["Q_H"].sum()),
        "annual_cooling_Wh": float(sim_df["Q_C"].sum()),
    }


A3_SOURCE = "2020 edition, Addendum b"


def range_status(value: float, limits) -> str:
    """PASS/FAIL against an Annex A3 range; N/A where the standard gives no
    range; UNAVAILABLE when the range values are not installed locally."""
    if limits is None:
        return "N/A"
    if limits == "unavailable":
        return "UNAVAILABLE"
    return "PASS" if limits[0] <= value <= limits[1] else "FAIL"


def verdict_text(value: float, status: str, label: str, table: str) -> str:
    """Result phrase without the range values (they are copyright ASHRAE)."""
    where = f"ANSI/ASHRAE 140-2023, Annex A3, {table}"
    if status == "N/A":
        return f"no ASHRAE 140 acceptance range for this result ({label}; {where})"
    if status == "UNAVAILABLE":
        return (f"{value:+.3f} — acceptance range not available locally ({label}; values in {where}; "
                f"see annex_a3_ranges.template.json)")
    phrase = ("within the ASHRAE 140 acceptance range" if status == "PASS"
              else "outside the ASHRAE 140 acceptance range")
    return f"{value:+.3f} — {status} ({phrase}, {label}; {A3_SOURCE} — {where})"


def main(surface_coefficients: str = "ashrae_convective"):
    for mass, case_number, criteria in (
        ("light", 600, LOW_MASS_ACCEPTANCE[600]),
        ("heavy", 900, HIGH_MASS_ACCEPTANCE[900]),
    ):
        print(f"\n=== ASHRAE 140 Case {case_number} ({'lightweight' if mass == 'light' else 'heavyweight'}) ===")
        result = run_case(mass, surface_coefficients=surface_coefficients)
        for metric in ("heating", "cooling"):
            value = result[f"annual_{metric}_MWh"]
            status = range_status(value, criteria[metric])
            name = {"heating": "Annual heating", "cooling": "Annual cooling"}[metric]
            print(f"  {name:<15} {verdict_text(value, status, f'Case {case_number}', annex_a3_table(case_number, metric))}")
        print(f"  {'Peak heating':<15} {result['peak_heating_kW']:7.3f} kW  diagnostic only")
        print(f"  {'Peak cooling':<15} {result['peak_cooling_kW']:7.3f} kW  diagnostic only")


def _suite(acceptance: dict, surface_coefficients: str, title: str) -> list[dict]:
    """Run all cases of one Annex A3 test group and grade every range case."""
    results = {n: run_case(case_number=n, surface_coefficients=surface_coefficients) for n in acceptance}
    rows = []
    for case_number, criteria in acceptance.items():
        parent = criteria["parent"]
        for metric in ("heating", "cooling"):
            value = results[case_number][f"annual_{metric}_MWh"]
            label = f"Case {case_number}"
            if parent is not None:
                value -= results[parent][f"annual_{metric}_MWh"]
                label = f"Case {case_number} − Case {parent}"
                if criteria.get("reverse"):
                    value, label = -value, f"Case {parent} − Case {case_number}"
            status = range_status(value, criteria[metric])
            rows.append({"case": case_number, "parent": parent or "-", "metric": metric,
                         "value": value, "label": label, "status": status,
                         "table": annex_a3_table(case_number, metric)})

    print(f"\n=== ASHRAE 140 {title}: Annex A3 range cases ===")
    for row in rows:
        print(f"{row['case']:>4} {row['metric']:<8} "
              f"{verdict_text(row['value'], row['status'], row['label'], row['table'])}")
    print("\nAbsolute annual loads [MWh/year]")
    print("case heating cooling peak_heating_kW peak_cooling_kW")
    for case_number, result in results.items():
        print(f"{case_number:>4} {result['annual_heating_MWh']:>7.3f} "
              f"{result['annual_cooling_MWh']:>7.3f} {result['peak_heating_kW']:>16.3f} "
              f"{result['peak_cooling_kW']:>15.3f}")
    sun = results.get(960)
    if sun:
        print(f"\nCase 960 sun zone (free-floating, diagnostic): mean {sun['sun_zone_T_mean_C']:.2f} C, "
              f"min {sun['sun_zone_T_min_C']:.2f} C, max {sun['sun_zone_T_max_C']:.2f} C")
    return rows


def run_low_mass_suite(surface_coefficients: str = "ashrae_convective") -> list[dict]:
    """Cases 600-695 against Tables A3-1/A3-2."""
    return _suite(LOW_MASS_ACCEPTANCE, surface_coefficients, "low-mass suite")


def run_high_mass_suite(surface_coefficients: str = "ashrae_convective") -> list[dict]:
    """Cases 900-995 against Tables A3-3/A3-4."""
    return _suite(HIGH_MASS_ACCEPTANCE, surface_coefficients, "high-mass suite")


def summarize_pass_counts(low_rows: list[dict], high_rows: list[dict]) -> None:
    """Count results within range against Table A3-14's pass criteria."""
    if not ANNEX_A3_AVAILABLE:
        print("Annex A3 range values not installed locally: group counts not available "
              "(ANSI/ASHRAE 140-2023, Annex A3, Tables A3-1 to A3-4 and A3-14).")
        return
    for label, rows, total, needed in (("Thermal Fabric Low Mass", low_rows, 21, 18),
                                       ("Thermal Fabric High Mass", high_rows, 19, 17)):
        graded = [r for r in rows if r["status"] in ("PASS", "FAIL")]
        passed = sum(r["status"] == "PASS" for r in graded)
        print(f"{label}: {passed}/{len(graded)} range cases within range "
              f"(Table A3-14: {needed} of {total} required)")


if __name__ == "__main__":
    coefficients = sys.argv[1] if len(sys.argv) > 1 else "ashrae_convective"
    print(f"surface_coefficients = {coefficients!r}")
    main(coefficients)
    summarize_pass_counts(run_low_mass_suite(coefficients), run_high_mass_suite(coefficients))
