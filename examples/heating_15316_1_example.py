"""Run a heating-system example using `iso_15316_1.py`.

This example mirrors the style of `heat_pump_15316_4_2_example.py`, but it is
focused on the heating chain only:

1. Create a simple hourly heating-demand profile.
2. Feed the load table to `HeatingSystemCalculator`.
3. Optionally use analytical distribution and boiler generation modules.
4. Save the hourly results to CSV.

The script is intentionally compact so it can be used as a starting point for
custom EN 15316-1 / EN 15316-3 / EN 15316-4-1 workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pybuildingenergy as pybui  # noqa: E402


def default_output_dir() -> Path:
    return REPO_ROOT / "examples" / "outputs" / "heating_15316_1_example"


def example_loads() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=24, freq="h")
    q_h = np.array(
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.2,
            0.5, 0.8, 1.2, 1.5, 1.8, 2.0,
            2.2, 2.1, 1.9, 1.7, 1.4, 1.1,
            0.9, 0.6, 0.4, 0.2, 0.1, 0.0,
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "Q_H_kWh": q_h,
            "T_op": np.full(24, 20.0),
            "T_ext": np.array(
                [
                    -2.0, -2.5, -3.0, -3.5, -4.0, -3.0,
                    -1.5, 0.0, 1.5, 3.0, 4.0, 5.0,
                    5.5, 5.0, 4.0, 2.5, 1.0, 0.0,
                    -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
                ],
                dtype=float,
            ),
        },
        index=index,
    )


def heating_config(distribution_mode: str, generation_mode: str) -> dict:
    base = {
        "emitter_type": "Floor heating",
        "nominal_power": 8.0,
        "emission_efficiency": 90.0,
        "flow_temp_control_type": "Type 2 - Based on outdoor temperature",
        "selected_emm_cont_circuit": 0,
        "mixing_valve": True,
        "mixing_valve_delta": 2.0,
        "heat_losses_recovered": True,
        "distribution_loss_recovery": 90.0,
        "simplified_approach": 80.0,
        "distribution_aux_recovery": 80.0,
        "distribution_aux_power": 30.0,
        "distribution_loss_coeff": 48.0,
        "distribution_operation_time": 1.0,
        "full_load_power": 24.0,
        "max_monthly_load_factor": 100.0,
        "tH_gen_i_ON": 1.0,
        "auxiliary_power_generator": 0.0,
        "fraction_of_auxiliary_power_generator": 40.0,
        "generator_circuit": "independent",
        "gen_flow_temp_control_type": "Type A - Based on outdoor temperature",
        "gen_outdoor_temp_data": pd.DataFrame(
            {
                "θext_min_gen": [-7],
                "θext_max_gen": [15],
                "θflw_gen_max": [60],
                "θflw_gen_min": [35],
            },
            index=["Generator curve"],
        ),
        "speed_control_generator_pump": "variable",
        "generator_nominal_deltaT": 20.0,
        "efficiency_model": "simple",
        "calc_when_QH_positive_only": False,
        "off_compute_mode": "full",
        "distribution_calculation_mode": distribution_mode,
        "generation_calculation_mode": generation_mode,
    }

    if distribution_mode == "analytical":
        base["distribution_15316_3_config"] = {
            "time_step_hours": 1.0,
            "demand_unit": "kWh",
            "heating": {
                "operation_mode": "demand",
                "nominal_power_kW": 8.0,
                "design_flow_m3_h": 0.5,
                "design_deltaT_K": 10.0,
                "pipe_sections": [
                    {
                        "length_m": 12.0,
                        "equivalent_length_m": 2.0,
                        "linear_thermal_transmittance_W_mK": 0.45,
                        "ambient_temperature_C": 20.0,
                        "recoverable": True,
                    }
                ],
                "pump_control_code": 4,
                "eei": 0.23,
                "hydraulic_correction_factor": 1.0,
                "recoverable_aux_fraction": 0.25,
                "pressure_loss_per_m_kPa": 0.10,
                "additional_pressure_kPa": 0.0,
                "resistance_ratio": 0.30,
                "pump_selection_factor": 1.0,
                "pump_label_power_kW": 0.0,
                "part_load_mode": "load",
            },
        }

    if generation_mode == "boiler_15316_4_1":
        base["boiler_generation_config"] = {
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "rated_power_kW": 24.0,
            "intermediate_load_fraction": 0.30,
            "eta_Pn_test_pct": 98.0,
            "eta_Pint_test_pct": 106.0,
            "theta_test_Pn_C": 60.0,
            "theta_test_Pint_C": 40.0,
            "f_corr_pct_per_K": 0.04,
            "P_gen_ls_P0_W": 100.0,
            "P_aux_on_W": 80.0,
            "P_aux_off_W": 5.0,
            "f_jacket": 0.40,
            "f_location": 1.0,
            "f_aux_recoverable": 0.75,
            "dew_point_C": 55.0,
            "condensing_gain_pct": 11.0,
            "efficiency_table": {
                "condensing": {
                    "eta_Pn_test_pct": 98.0,
                    "eta_Pint_test_pct": 106.0,
                    "theta_test_Pn_C": 60.0,
                    "theta_test_Pint_C": 40.0,
                }
            },
            "loss_table": {
                "condensing": {
                    "P_gen_ls_P0_W": 100.0,
                }
            },
            "boiler_location": "inside_heated",
        }

    return base


def run_example(distribution_mode: str, generation_mode: str, output_dir: Path) -> pd.DataFrame:
    loads = example_loads()
    calc = pybui.HeatingSystemCalculator(heating_config(distribution_mode, generation_mode))
    results = calc.run_timeseries(loads)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "heating_15316_1_hourly_results.csv")
    pd.DataFrame([results.sum(numeric_only=True)]).to_csv(
        output_dir / "heating_15316_1_summary.csv",
        index=False,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EN 15316-1 heating example.")
    parser.add_argument(
        "--distribution-mode",
        choices=["simplified", "analytical"],
        default="simplified",
        help="Distribution calculation mode.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["legacy", "boiler_15316_4_1"],
        default="legacy",
        help="Generation calculation mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory where CSV outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_example(
        distribution_mode=args.distribution_mode,
        generation_mode=args.generation_mode,
        output_dir=Path(args.output_dir),
    )
    print(results[["QH_dis_i_in(kWh)", "QH_gen_out(kWh)", "EHW_gen_in(kWh)"]].head())


if __name__ == "__main__":
    main()
