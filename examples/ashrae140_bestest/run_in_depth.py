"""Run the ASHRAE 140-2023 Section 7.2.3 in-depth cases through pybuildingenergy.

The standard prints no acceptance range for these cases (they are diagnostic), so
the harness reports the annual loads and the *differences* between a case and its
base case (Informative Table 7-33): each difference isolates one modelling feature.
Reference results of other programs are not reproduced.

Usage: python examples/ashrae140_bestest/run_in_depth.py [--solar-distribution area|table]
"""

import argparse
import sys
import warnings
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import WEATHER_FILE  # noqa: E402
from in_depth_cases import DIFFERENCES, IN_DEPTH_CASES, build_in_depth_case  # noqa: E402
from run_comparison import run_case  # noqa: E402
from pybuildingenergy.source.utils import ISO52016  # noqa: E402


def simulate_in_depth(case_number: int, solar_distribution: str = "table"):
    """Return the hourly DataFrame of one in-depth case."""
    bui = build_in_depth_case(case_number, solar_distribution=solar_distribution)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
            c_int_per_A_us=0.0, hvac_control_variable="air",
        )
    return out[0] if isinstance(out, tuple) else out


def run_in_depth_case(case_number: int, solar_distribution: str = "table") -> dict:
    sim_df = simulate_in_depth(case_number, solar_distribution)
    return {
        "annual_heating_MWh": sim_df["Q_H"].sum() / 1e6,
        "annual_cooling_MWh": sim_df["Q_C"].sum() / 1e6,
        "peak_heating_kW": sim_df["Q_H"].max() / 1000.0,
        "peak_cooling_kW": sim_df["Q_C"].max() / 1000.0,
    }


def run_all(solar_distribution: str = "table") -> dict:
    """Results of the 25 in-depth cases plus the main-suite bases 600 and 900."""
    results = {n: run_in_depth_case(n, solar_distribution) for n in IN_DEPTH_CASES}
    for base in (600, 900):
        results[base] = run_case(case_number=base)
    return results


def differences(results: dict) -> list[dict]:
    """Heating and cooling difference (MWh) of every pair of ``DIFFERENCES``."""
    rows = []
    for case, base, feature in DIFFERENCES:
        rows.append({
            "case": case, "base": base, "feature": feature,
            "d_heating_MWh": results[case]["annual_heating_MWh"] - results[base]["annual_heating_MWh"],
            "d_cooling_MWh": results[case]["annual_cooling_MWh"] - results[base]["annual_cooling_MWh"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solar-distribution", choices=("area", "table"), default="table")
    args = parser.parse_args()
    results = run_all(args.solar_distribution)
    print(f"ASHRAE 140-2023 Section 7.2.3 in-depth cases (solar distribution: {args.solar_distribution})")
    print(f"{'Case':>5} {'Heating MWh':>12} {'Cooling MWh':>12} {'Peak H kW':>10} {'Peak C kW':>10}")
    for case in sorted(results):
        r = results[case]
        print(f"{case:>5} {r['annual_heating_MWh']:12.3f} {r['annual_cooling_MWh']:12.3f} "
              f"{r['peak_heating_kW']:10.3f} {r['peak_cooling_kW']:10.3f}")
    print("\nDifferences (case - base):")
    for row in differences(results):
        print(f"  {row['case']} - {row['base']}: dHeating {row['d_heating_MWh']:+8.3f} MWh, "
              f"dCooling {row['d_cooling_MWh']:+8.3f} MWh   ({row['feature']})")


if __name__ == "__main__":
    main()
