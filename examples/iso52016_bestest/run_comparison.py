"""Run ISO 52016-1:2017's own verification cases (clause 7.2.3, Table 27:
600, 640, 900, 940, 600FF, 900FF) through pybuildingenergy and report the
clause 7.2.4 outputs against the standard's own reference values (Tables
28-34). See iso_case_definitions.py and README.md for why this is a
*different* case from examples/ashrae140_bestest/.

Reference values are read from a local file that is not distributed (ISO
copyright, see iso_reference_results.py); only deviations are printed.

ISO 52016-1 gives no pass/fail tolerance: clause 7.2 asks for the results
and their deviations from the reference to be reported. This harness flags
annual heating/cooling within +/-10% of the reference (the tolerance used
by tests/test_iso52016_bestest.py) and reports the free-floating cases
as operative-temperature deviations only.

Uses ISO 52016-1's own BESTEST climate spreadsheet
(iso_climate_loader.ISO_CLIMATE_XLS_PATH) when available -- see that
module's docstring for why it is not shipped in this repository -- and
falls back to the Denver TMY3 EPW approximation otherwise.

Usage: python examples/iso52016_bestest/run_comparison.py
"""

import sys
import warnings
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EXAMPLES_DIR))

from iso_case_definitions import CASE_IDS, build_case, WEATHER_FILE  # noqa: E402
from iso_climate_loader import iso_bestest_climate  # noqa: E402
import iso_reference_results as ref  # noqa: E402
from pybuildingenergy.source.utils import ISO52016  # noqa: E402

ANNUAL_TOLERANCE = 0.10
CASE_INPUTS = {case_id: key for key, case_id in CASE_IDS.items()}  # "640" -> ("light", "intermittent")


def simulate(case_id: str):
    """Return (hourly DataFrame, using_exact_climate) for one Table 27 case."""
    mass, control = CASE_INPUTS[case_id]
    bui = build_case(mass, control)
    calc_kwargs = dict(
        weather_source="epw", path_weather_file=str(WEATHER_FILE), c_int_per_A_us=0.0,
        # Deliberately NOT passing hvac_control_variable: this case wants
        # the library's default ("operative"), per ISO 52016-1 clause 7.2.2.1.
    )
    try:
        with iso_bestest_climate() as _, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ISO52016.Temperature_and_Energy_needs_calculation(bui, **calc_kwargs)
        using_exact_climate = True
    except (FileNotFoundError, ImportError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ISO52016.Temperature_and_Energy_needs_calculation(bui, **calc_kwargs)
        using_exact_climate = False
    sim_df = out[0] if isinstance(out, tuple) else out
    return sim_df, using_exact_climate


def run_case(mass: str, control: str = "continuous") -> dict:
    sim_df, using_exact_climate = simulate(CASE_IDS[(mass, control)])
    return {
        "annual_heating_kWh": sim_df["Q_H"].sum() / 1000.0,
        "annual_cooling_kWh": sim_df["Q_C"].sum() / 1000.0,
        "annual_mean_T_op_C": float(sim_df["T_op"].mean()),
        "using_exact_climate": using_exact_climate,
    }


def _fmt_row(label, value, reference, table):
    """The reference value itself is not printed (ISO copyright): only the deviation."""
    if reference is None:
        print(f"  {label:<22} pybuildingenergy={value:9.1f} kWh  "
              f"(reference: EN ISO 52016-1:2017 {table}, not installed locally)")
        return
    dev_pct = 100.0 * (value - reference) / reference
    status = "within +/-10%" if abs(dev_pct) <= 100 * ANNUAL_TOLERANCE else "OUTSIDE +/-10%"
    print(f"  {label:<22} pybuildingenergy={value:9.1f} kWh  "
          f"{dev_pct:+.1f}% vs EN ISO 52016-1:2017 {table} ({status})")


def main():
    for (mass, control), case_id in CASE_IDS.items():
        labels = {"light": "lightweight", "heavy": "heavyweight", "continuous": "continuous",
                  "intermittent": "intermittent", "free_floating": "free floating"}
        print(f"\n=== ISO 52016-1 Case {case_id} ({labels[mass]}, {labels[control]}) ===")
        result = run_case(mass, control)
        climate_label = ("ISO 52016-1's own BESTEST climate data" if result["using_exact_climate"]
                         else "Denver TMY3 EPW approximation (see README.md)")
        print(f"  weather: {climate_label}")
        if not case_id.endswith("FF"):
            _fmt_row("Annual heating", result["annual_heating_kWh"],
                     ref.ANNUAL_HEATING_KWH_ALL.get(case_id), "Table 28")
            _fmt_row("Annual cooling", result["annual_cooling_kWh"],
                     ref.ANNUAL_COOLING_KWH_ALL.get(case_id), "Table 29")
        t_ref = ref.ANNUAL_OPERATIVE_TEMPERATURE_C.get(case_id)
        dev = "" if t_ref is None else f"{result['annual_mean_T_op_C'] - t_ref:+.2f} K vs "
        print(f"  {'Annual mean T_op':<22} pybuildingenergy={result['annual_mean_T_op_C']:9.2f} degC  "
              f"{dev}EN ISO 52016-1:2017 Table 30")


if __name__ == "__main__":
    main()
