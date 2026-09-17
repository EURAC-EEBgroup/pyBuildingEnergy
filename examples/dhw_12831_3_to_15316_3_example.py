"""Run DHW annual demand from EN 12831-3 and distribution losses from EN 15316-3.

This example builds an hourly annual domestic hot water demand profile using
the EN 12831-3 implementation in ``pybuildingenergy.source.DHW`` and then
passes that profile to the EN 15316-3 distribution calculator to estimate:

- pipe thermal losses;
- recoverable thermal losses;
- circulation pump auxiliary electricity;
- recovered auxiliary heat;
- generator-side DHW input after distribution.

The script is intended as a compact, inspectable bridge between the DHW need
calculation and the distribution-loss calculation.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import pandas as pd


EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (SRC_DIR, PROJECT_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import pybuildingenergy as pybui  # noqa: E402
from pybuildingenergy.source.DHW import Volume_and_energy_DHW_calculation  # noqa: E402


DHW_DAY_TYPE_COLUMNS = ("Workday", "Weekend", "Holiday")
DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT = {
    "single_family_dwelling": [
        1.8, 1.0, 0.6, 0.3, 0.4, 0.6, 2.4, 4.7, 6.8, 5.7, 6.1, 6.1,
        6.3, 6.4, 5.1, 4.4, 4.3, 4.7, 5.7, 6.5, 6.6, 5.8, 4.5, 3.1,
    ],
}


def default_output_dir() -> Path:
    return EXAMPLES_DIR / "outputs" / "dhw_12831_3_to_15316_3"


def build_annual_dhw_profile() -> pd.DataFrame:
    """Create an hourly annual DHW load series from the EN 12831-3 model."""

    building_area_m2 = 120.0
    country = "IT"

    fractions = dhw_annex_b_table_b2_hourly_fractions("single_family_dwelling")
    sum_fractions = pd.DataFrame(fractions.sum(), columns=["fractions"])

    year = 2026
    calendar = pybui.generate_calendar(country, year)
    n_workdays = int((calendar["values"] == "Working").sum())
    n_weekends = int((calendar["values"] == "Non-Working").sum())
    n_holidays = int((calendar["values"] == "Holiday").sum())
    total_days = int(calendar["values"].count())

    dhw_result = Volume_and_energy_DHW_calculation(
        n_workdays,
        n_weekends,
        n_holidays,
        sum_fractions,
        total_days,
        fractions,
        42.0,
        13.5,
        60.0,
        11.2,
        mode_calc="number_of_units",
        building_type_B3="Residential",
        building_area=building_area_m2,
        unit_count=4,
        building_type_B5="Dwelling",
        residential_typology="residential_building - simple housing - AVG",
        calculation_method="table",
        year=year,
        country_calendar=calendar,
    )

    hourly_energy = pd.Series(dhw_result[7], name="Q_W_kWh")
    hourly_index = pd.date_range(f"{year}-01-01 00:00:00", periods=len(hourly_energy), freq="h")
    hourly_energy.index = hourly_index

    profile = pd.DataFrame(
        {
            "Q_W_kWh": hourly_energy.astype(float),
            "T_ext": pd.NA,
            "T_op": 20.0,
        }
    )
    profile.index.name = "timestamp"
    return profile


def dhw_annex_b_table_b2_hourly_fractions(profile_key: str = "single_family_dwelling") -> pd.DataFrame:
    if profile_key not in DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT:
        available = ", ".join(sorted(DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT))
        raise ValueError(f"Unknown DHW profile '{profile_key}'. Available: {available}")

    percent = pd.Series(
        DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT[profile_key],
        index=range(24),
        dtype=float,
    )
    total = float(percent.sum())
    if total <= 0.0:
        raise ValueError(f"DHW profile '{profile_key}' must have a positive sum.")
    fractions = percent / total
    return pd.DataFrame({column: fractions.values for column in DHW_DAY_TYPE_COLUMNS})


def distribution_config() -> dict:
    return {
        "time_step_hours": 1.0,
        "demand_unit": "kWh",
        "dhw": {
            "operation_mode": "demand",
            "nominal_power_kW": 8.0,
            "design_flow_m3_h": 0.5,
            "design_deltaT_K": 10.0,
            "dhw_temperature_C": 55.0,
            "dhw_return_deltaT_K": 5.0,
            "max_length_m": 25.0,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 0.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 4,
            "pump_selection_factor": 1.0,
            "pump_label_power_kW": 0.0,
            "eei": 0.23,
            "hydraulic_correction_factor": 1.0,
            "part_load_mode": "load",
            "recoverable_aux_fraction": 0.25,
            "pipe_sections": [
                {
                    "length_m": 12.0,
                    "equivalent_length_m": 2.0,
                    "linear_thermal_transmittance_W_mK": 0.45,
                    "ambient_temperature_C": 20.0,
                    "recoverable": True,
                },
                {
                    "length_m": 8.0,
                    "equivalent_length_m": 1.0,
                    "linear_thermal_transmittance_W_mK": 0.35,
                    "ambient_temperature_C": 18.0,
                    "recoverable": False,
                },
            ],
        },
    }


def storage_config(enabled: bool) -> dict:
    if not enabled:
        return {}

    return {
        "time_step_hours": 1.0,
        "demand_unit": "kWh",
        "dhw": {
            "enabled": True,
            "storage_volume_l": 180.0,
            "storage_setpoint_C": 55.0,
            "output_temperature_C": 55.0,
            "ambient_temperature_C": 20.0,
            "standby_loss_kWh_per_day_ref": 0.90,
            "standby_set_temperature_ref_C": 55.0,
            "standby_ambient_temperature_ref_C": 20.0,
            "standby_loss_adaptation_factor": 1.0,
            "connection_loss_factor": 1.0,
            "thermal_loss_room_fraction": 0.75,
            "auxiliary_to_medium_fraction": 0.25,
            "input_pump_power_kW": 0.0,
            "input_pump_flow_m3_h": 0.0,
            "input_pump_deltaT_K": 10.0,
            "output_pump_power_kW": 0.0,
            "output_pump_flow_m3_h": 0.0,
            "output_pump_deltaT_K": 10.0,
            "operation_mode": "demand",
        },
    }


def run_example(
    output_dir: Path,
    use_storage: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pybui.DistributionSimulationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dhw_profile = build_annual_dhw_profile()
    dhw_profile.to_csv(output_dir / "dhw_12831_3_hourly_profile.csv")

    distribution_input = dhw_profile.copy()
    dist_calc = pybui.DistributionSystemCalculator(distribution_config())
    dist_result = dist_calc.run_timeseries(distribution_input)

    dist_result.timeseries.to_csv(output_dir / "distribution_15316_3_dhw_hourly_results.csv")
    pd.DataFrame([dist_result.summary]).to_csv(
        output_dir / "distribution_15316_3_dhw_summary.csv",
        index=False,
    )

    storage_result = None
    if use_storage:
        storage_input = pd.DataFrame(
            {"Q_W_kWh": dist_result.timeseries["QW_dis_in_kWh"].astype(float)},
            index=dhw_profile.index,
        )
        storage_calc = pybui.StorageSystemCalculator(storage_config(True))
        storage_result = storage_calc.run_timeseries(storage_input)
        storage_result.timeseries.to_csv(output_dir / "storage_15316_5_hourly_results.csv")
        pd.DataFrame([storage_result.summary]).to_csv(
            output_dir / "storage_15316_5_summary.csv",
            index=False,
        )

    return dhw_profile, dist_result.timeseries, storage_result.timeseries if storage_result is not None else None, dist_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EN 12831-3 DHW need, optional EN 15316-5 storage, and EN 15316-3 distribution."
    )
    parser.add_argument(
        "--storage",
        action="store_true",
        help="Enable optional DHW storage calculation according to EN 15316-5.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory where CSV outputs are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dhw_profile, distribution_input, storage_result, dist_result = run_example(
        output_dir,
        use_storage=bool(args.storage),
    )

    summary = dist_result.summary
    print("DHW annual profile and distribution losses calculated.")
    print(f"Hours: {len(dhw_profile)}")
    print(f"Annual DHW demand [kWh]: {dhw_profile['Q_W_kWh'].sum():.2f}")
    if storage_result is not None:
        print(
            f"Annual storage losses [kWh]: {storage_result.summary.get('QW_sto_ls_kWh', 0.0):.2f}"
        )
        print(
            f"Annual storage auxiliaries [kWh]: {storage_result.summary.get('WW_sto_aux_kWh', 0.0):.2f}"
        )
        print(
            f"Storage-side DHW input [kWh]: {storage_result.summary.get('QW_sto_in_kWh', 0.0):.2f}"
        )
        print(
            f"DHW demand passed to distribution [kWh]: {distribution_input['Q_W_kWh'].sum():.2f}"
        )
    print(f"Annual distribution losses [kWh]: {summary.get('QW_dis_ls_kWh', 0.0):.2f}")
    print(f"Annual DHW pump auxiliaries [kWh]: {summary.get('WW_dis_aux_kWh', 0.0):.2f}")
    print(f"Generator-side DHW input [kWh]: {summary.get('QW_dis_in_kWh', 0.0):.2f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
