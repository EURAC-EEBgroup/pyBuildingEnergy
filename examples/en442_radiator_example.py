"""Minimal EN 442 radiator-to-EN 15316 heating-chain example.

The catalogue values below are illustrative.  Replace them with the declared
values of a certified product before using the calculation in a project.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pybuildingenergy as pybui  # noqa: E402


RADIATOR_SYSTEM = {
    "emitter_type": "Panel radiator",
    "emitter_calculation_method": "en442",
    "emitter_rating": {
        "phi_50_kW": 8.0,
        "exponent_n": 1.30,
        "phi_30_kW": 8.0 * (30.0 / 50.0) ** 1.30,
        "maximum_operating_temperature_C": 110.0,
        "product_reference": "Illustrative EN 442 panel radiator array",
        # Optional EN 442/manufacturer hydraulic declaration.
        "standard_water_flow_kg_s": 0.10,
        "pressure_drop_at_standard_flow_kPa": 5.0,
        "pressure_drop_exponent": 2.0,
    },
    # EN 15316 circuit design, deliberately separate from EN 442 rating data.
    "circuit_design": {"design_water_deltaT_K": 20.0},
    "emission_efficiency": 100.0,
    "emission_operation_time": 1.0,
    "selected_emm_cont_circuit": 0,
    "flow_temp_control_type": "Type 1 - Based on demand",
    "mixing_valve": False,
    "distribution_calculation_mode": "analytical",
    "distribution_loss_coeff": 0.0,
    "distribution_length_m": 20.0,
    "distribution_pressure_loss_per_m_kPa": 0.10,
    "full_load_power": 24.0,
    "calc_when_QH_positive_only": False,
    "off_compute_mode": "full",
}


def run_example() -> pd.DataFrame:
    checked = pybui.check_heating_system_inputs(RADIATOR_SYSTEM)["config"]
    calculator = pybui.HeatingSystemCalculator(checked)
    loads = pd.DataFrame(
        {
            "Q_H_kWh": [2.0, 4.0, 8.0, 16.0],
            "T_op": [20.0, 20.0, 20.0, 20.0],
            "T_ext": [10.0, 5.0, 0.0, -5.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )
    return calculator.run_timeseries(loads)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_example()
    print(
        results[
            [
                "ΦH_em_requested(kW)",
                "ΦH_em_available(kW)",
                "ΦH_em_eff(kW)",
                "QH_em_unmet(kWh)",
                "θH_em_flow(°C)",
                "θH_em_ret(°C)",
                "QH_gen_out(kWh)",
            ]
        ].rename(
            columns={
                "ΦH_em_requested(kW)": "requested_kW",
                "ΦH_em_available(kW)": "available_kW",
                "ΦH_em_eff(kW)": "delivered_kW",
                "QH_em_unmet(kWh)": "unmet_kWh",
                "θH_em_flow(°C)": "emitter_supply_C",
                "θH_em_ret(°C)": "emitter_return_C",
                "QH_gen_out(kWh)": "generator_output_kWh",
            }
        ).to_string()
    )
