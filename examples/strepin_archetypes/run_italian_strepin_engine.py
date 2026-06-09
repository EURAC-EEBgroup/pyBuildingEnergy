"""Run pyBuildingEnergy simulations for Italian STREPIN archetype packages.

The workbook defines the archetype geometry and package levels. This script
builds pre/post-retrofit BUI cases, runs ISO52016, and writes an engine-derived
summary that can be compared with the workbook's estimate columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pybuildingenergy as pybui  # noqa: E402


DEFAULT_WORKBOOK = (
    REPO_ROOT
    / "examples"
    / "strepin_archetypes"
    / "Italian_STREPIN_archetype_renovation_cost_models_visuals_reworked_fixed.xlsx"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "outputs" / "italian_strepin_engine"
DEFAULT_ZONE_B_EPW = REPO_ROOT / "examples" / "Palermo_PVGIS_TMY.epw"
DEFAULT_ZONE_E_EPW = REPO_ROOT / "examples" / "2020_Milan.epw"


def _split_selection(raw: str, allowed: list[str]) -> list[str]:
    value = str(raw).strip()
    if value.lower() == "all":
        return allowed
    wanted = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown selection(s): {', '.join(unknown)}. "
            f"Available: {', '.join(allowed)}"
        )
    return wanted


def _weather_kwargs(args: argparse.Namespace, climate_zone: str) -> dict:
    kwargs = {"weather_source": args.weather_source}
    if args.weather_source == "epw":
        epw = args.zone_b_epw if climate_zone == "B" else args.zone_e_epw
        if epw is None:
            raise ValueError(f"Missing EPW path for climate zone {climate_zone}.")
        epw_path = Path(epw).resolve()
        if not epw_path.exists():
            raise FileNotFoundError(epw_path)
        kwargs["path_weather_file"] = str(epw_path)
    return kwargs


def _run_iso52016(case: pybui.StrepinCase, args: argparse.Namespace):
    checked, issues = pybui.sanitize_and_validate_BUI(case.bui, fix=True)
    errors = [issue for issue in issues if issue["level"] == "ERROR"]
    if errors:
        details = "\n".join(f"- {item['path']}: {item['msg']}" for item in errors)
        raise RuntimeError(f"{case.archetype_id}/{case.package_id} is invalid:\n{details}")

    climate_zone = str(case.archetype["Climate_Zone"])
    result = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        checked,
        **_weather_kwargs(args, climate_zone),
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("ISO52016 returned an unexpected result.")
    return result[0], result[1]


def _prepare_system_loads(case: pybui.StrepinCase, hourly: pd.DataFrame) -> pd.DataFrame:
    loads = pd.DataFrame(index=hourly.index)
    loads["T_ext"] = hourly["T_ext"].astype(float)
    if "Q_H" in hourly:
        loads["Q_H_kWh"] = hourly["Q_H"].astype(float).clip(lower=0.0) / 1000.0
    else:
        loads["Q_H_kWh"] = hourly["Q_HC"].astype(float).clip(lower=0.0) / 1000.0
    if "Q_C" in hourly:
        loads["Q_C_kWh"] = hourly["Q_C"].astype(float).clip(lower=0.0) / 1000.0
    else:
        loads["Q_C_kWh"] = (-hourly["Q_HC"].astype(float).clip(upper=0.0)) / 1000.0

    area_m2 = float(case.archetype["Af_m2"])
    dhw_kWh_a = float(case.archetype.get("Baseline_DHW_Useful", 0.0) or 0.0) * area_m2
    loads["Q_W_kWh"] = dhw_kWh_a / max(len(loads), 1)
    return loads


def _strepin_hp_map(scop: float, capacity_kW: float) -> pd.DataFrame:
    rows = []
    for source in [-7.0, 2.0, 7.0, 12.0]:
        for sink in [35.0, 45.0, 55.0]:
            cop = max(1.2, float(scop) + 0.055 * (source - 7.0) - 0.030 * (sink - 45.0))
            rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(float(capacity_kW), 1.0),
                    "cop": cop,
                }
            )
    return pd.DataFrame(rows)


def _strepin_cooling_result(
    loads: pd.DataFrame,
    *,
    cooling_eer: float,
    nominal_capacity_kW: float,
) -> pybui.CoolingGenerationSimulationResult:
    return pybui.CoolingGenerationSystemCalculator(
        {
            "demand_unit": "kWh",
            "nominal_capacity_kW": max(float(nominal_capacity_kW), 1.0),
            "nominal_eer": max(float(cooling_eer), 1.0),
            "control_power_kW": 0.005,
            "part_load_performance_method": "simple",
        }
    ).run_timeseries(loads)


def _run_strepin_standard_chain(
    case: pybui.StrepinCase,
    hourly: pd.DataFrame,
    tables: pybui.ItalianStrepinTables,
    args: argparse.Namespace,
) -> dict[str, float | str | None]:
    loads = _prepare_system_loads(case, hourly)
    area_m2 = float(case.archetype["Af_m2"])
    heating_system = case.bui.get("strepin", {}).get("heating_system", {})
    carrier = str(heating_system.get("carrier", "gas"))
    system_type = str(heating_system.get("system_type", "gas_boiler"))
    efficiency = float(
        heating_system.get(
            "seasonal_efficiency_or_scop",
            case.archetype.get("Baseline_Boiler_Eff", 0.85),
        )
        or 0.85
    )
    nominal_capacity_kW = max(float(case.archetype.get("Heating_Capacity_kW", 0.0) or 0.0), 1.0)

    if carrier == "electricity":
        hp_map = _strepin_hp_map(efficiency, nominal_capacity_kW)
        heating_result = pybui.HeatPumpSystemCalculator(
            {
                "demand_unit": "kWh",
                "heating_performance_map": hp_map,
                "dhw_performance_map": hp_map,
                "cooling_enabled": False,
                "source_type": "air",
                "bin_width_C": 1.0,
                "design_outdoor_temperature_C": -3.0
                if str(case.archetype["Climate_Zone"]) == "B"
                else -7.0,
                "heating_cutoff_temperature_C": 16.0,
                "heating_sink_temp_at_design_C": 45.0,
                "heating_sink_temp_at_cutoff_C": 35.0,
                "dhw_target_temperature_C": 60.0,
                "dhw_sink_temperature_C": 55.0,
                "hp_operating_limit_C": 55.0,
                "backup_mode": "parallel",
                "heating_backup_efficiency": 1.0,
                "dhw_backup_efficiency": 1.0,
                "external_auxiliary_power_W": 40.0,
                "standby_power_W": 4.0,
                "part_load_performance_method": "simple",
            }
        ).run_timeseries(loads.assign(Q_C_kWh=0.0))
        heating_summary = heating_result.summary
        gas_final_kWh = 0.0
        hp_electricity_kWh = (
            heating_summary.get("EHW_gen_in_kWh", 0.0)
            + heating_summary.get("WHW_gen_aux_kWh", 0.0)
        )
        heating_standard = "EN 15316-4-2"
    else:
        heating_result = pybui.CombustionBoilerSystemCalculator(
            {
                "demand_unit": "kWh",
                "fuel": "natural_gas",
                "nominal_power_kW": nominal_capacity_kW,
                "full_load_efficiency": min(max(efficiency, 0.50), 1.05),
                "part_load_efficiency": min(max(efficiency + 0.03, 0.50), 1.08),
                "condensing": "condensing" in system_type,
                "auxiliary_power_kW": 0.020,
                "standby_auxiliary_power_kW": 0.002,
                "standby_loss_kWh_per_h": 0.006,
            }
        ).run_timeseries(loads)
        heating_summary = heating_result.summary
        gas_final_kWh = heating_summary.get("EHW_fuel_in_kWh", 0.0)
        hp_electricity_kWh = heating_summary.get("WHW_gen_aux_kWh", 0.0)
        heating_standard = "EN 15316-4-1"

    cooling_result = _strepin_cooling_result(
        loads,
        cooling_eer=args.cooling_eer,
        nominal_capacity_kW=nominal_capacity_kW * 0.6,
    )
    cooling_summary = cooling_result.summary
    cooling_electricity_kWh = cooling_summary.get("EC_total_kWh", 0.0)
    aux_electricity_kWh = float(case.archetype.get("Baseline_Aux_Elec", 0.0) or 0.0) * area_m2
    gross_electricity_kWh = hp_electricity_kWh + cooling_electricity_kWh + aux_electricity_kWh

    pv = case.bui.get("strepin", {}).get("pv_system", {})
    pv_kWp = float(pv.get("pv_kWp", 0.0) or 0.0)
    climate_zone = str(case.archetype["Climate_Zone"])
    pv_yield = float(tables.assumption_value(f"PV_Yield_Zone_{climate_zone}", 0.0) or 0.0)
    pv_loads = loads.assign(E_site_el_load_kWh=gross_electricity_kWh / max(len(loads), 1))
    pv_result = pybui.RenewableEnergySystemCalculator(
        {
            "demand_unit": "kWh",
            "solar_thermal": {"enabled": False, "area_m2": 0.0},
            "pv": {
                "enabled": pv_kWp > 0.0,
                "capacity_kWp": pv_kWp,
                "annual_yield_kWh_kWp": pv_yield,
                "performance_ratio": 0.80,
                "self_consumption_fraction": float(
                    tables.assumption_value("PV_SelfConsumption_Factor", 0.65) or 0.65
                ),
            },
        }
    ).run_timeseries(pv_loads)
    pv_summary = pv_result.summary
    grid_electricity_kWh = pv_summary.get("E_grid_after_PV_kWh", gross_electricity_kWh)
    pv_generation_kWh = pv_summary.get("E_PV_gen_kWh", 0.0)
    pv_self_consumed_kWh = pv_summary.get("E_PV_self_consumed_kWh", 0.0)
    pv_export_kWh = pv_summary.get("E_PV_export_kWh", 0.0)

    gas_pef = float(tables.assumption_value("Gas_Primary_Energy_Factor", 1.05) or 1.05)
    electricity_pef = float(
        tables.assumption_value("Electricity_Primary_Energy_Factor", 2.17) or 2.17
    )
    final_energy_kWh = gas_final_kWh + grid_electricity_kWh
    primary_nonren_kWh = gas_final_kWh * gas_pef + grid_electricity_kWh * electricity_pef
    reference = case.workbook_reference or {}

    return {
        "Archetype_ID": case.archetype_id,
        "Package_ID": case.package_id,
        "Building_Type": case.archetype.get("Building_Type"),
        "Type_Code": case.archetype.get("Type_Code"),
        "Climate_Zone": climate_zone,
        "Period_Code": case.archetype.get("Period_Code"),
        "Af_m2": area_m2,
        "System_chain": "en_standards",
        "Heating_generation_standard": heating_standard,
        "Cooling_generation_standard": "EN 16798-13",
        "PV_standard": "EN 15316-4-6",
        "Q_H_engine_kWh_a": loads["Q_H_kWh"].sum(),
        "Q_C_engine_kWh_a": loads["Q_C_kWh"].sum(),
        "Q_H_engine_kWh_m2a": loads["Q_H_kWh"].sum() / area_m2 if area_m2 > 0 else 0.0,
        "Q_C_engine_kWh_m2a": loads["Q_C_kWh"].sum() / area_m2 if area_m2 > 0 else 0.0,
        "DHW_useful_kWh_a": loads["Q_W_kWh"].sum(),
        "QHW_gen_out_kWh_a": heating_summary.get("QHW_gen_out_kWh", 0.0),
        "EHW_gen_in_kWh_a": heating_summary.get("EHW_gen_in_kWh", 0.0),
        "SPF_or_eta_HW_gen": heating_summary.get(
            "SPF_HW_gen", heating_summary.get("eta_HW_gen", None)
        ),
        "Gas_final_kWh_a": gas_final_kWh,
        "HeatPump_electricity_kWh_a": hp_electricity_kWh if carrier == "electricity" else 0.0,
        "Heating_aux_electricity_kWh_a": heating_summary.get("WHW_gen_aux_kWh", 0.0),
        "Cooling_electricity_kWh_a": cooling_electricity_kWh,
        "Cooling_SEER": cooling_summary.get("SEER_C_gen", None),
        "Aux_electricity_kWh_a": aux_electricity_kWh,
        "Gross_electricity_kWh_a": gross_electricity_kWh,
        "PV_kWp": pv_kWp,
        "PV_generation_kWh_a": pv_generation_kWh,
        "PV_self_consumed_kWh_a": pv_self_consumed_kWh,
        "PV_export_kWh_a": pv_export_kWh,
        "Grid_electricity_kWh_a": grid_electricity_kWh,
        "Final_energy_kWh_a": final_energy_kWh,
        "Final_energy_kWh_m2a": final_energy_kWh / area_m2 if area_m2 > 0 else 0.0,
        "Primary_nonren_kWh_a": primary_nonren_kWh,
        "Primary_nonren_kWh_m2a": primary_nonren_kWh / area_m2 if area_m2 > 0 else 0.0,
        "Heating_system_carrier": carrier,
        "Heating_efficiency_or_SCOP": efficiency,
        "Reference_workbook_final_kWh_m2a": reference.get("Final_Energy_Intensity_kWh_m2a"),
        "Reference_workbook_primary_kWh_m2a": reference.get(
            "Primary_Energy_Intensity_kWhEPnren_m2a"
        ),
        "Reference_workbook_package_name": reference.get("Package_Name"),
    }


def run(args: argparse.Namespace) -> Path:
    tables = pybui.load_italian_strepin_tables(args.workbook)
    archetype_ids = _split_selection(args.archetypes, tables.archetype_ids)
    package_ids = _split_selection(args.packages, tables.package_ids)

    cases = [
        tables.make_case(
            archetype_id,
            package_id,
            ideal_hvac_capacity=not args.use_workbook_capacity,
            air_change_rate_h=args.air_change_rate,
        )
        for archetype_id in archetype_ids
        for package_id in package_ids
    ]
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_rows = []
        for case in cases:
            dry_rows.append(
                {
                    "Archetype_ID": case.archetype_id,
                    "Package_ID": case.package_id,
                    "Climate_Zone": case.archetype["Climate_Zone"],
                    "Af_m2": case.archetype["Af_m2"],
                    "surface_count": len(case.bui["building_surface"]),
                    "heating_system": case.bui.get("strepin", {})
                    .get("heating_system", {})
                    .get("system_type"),
                    "pv_kWp": case.bui.get("strepin", {})
                    .get("pv_system", {})
                    .get("pv_kWp", 0.0),
                }
            )
        out = output_dir / "italian_strepin_engine_case_index.csv"
        pd.DataFrame(dry_rows).to_csv(out, index=False)
        print(f"Prepared {len(cases)} cases. Dry-run index: {out}")
        return out

    summary_rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.archetype_id} {case.package_id}")
        hourly, annual = _run_iso52016(case, args)
        if args.system_chain == "en-standards":
            summary_rows.append(
                _run_strepin_standard_chain(case, hourly, tables, args)
            )
        else:
            summary_rows.append(
                pybui.summarize_engine_performance(
                    case,
                    annual,
                    tables,
                    include_cooling=not args.no_cooling_postprocess,
                    cooling_eer=args.cooling_eer,
                )
            )
        if args.write_hourly:
            case_dir = output_dir / "hourly"
            case_dir.mkdir(exist_ok=True)
            hourly.to_csv(case_dir / f"{case.archetype_id}_{case.package_id}.csv")

    summary = pd.DataFrame(summary_rows)
    out = output_dir / "italian_strepin_engine_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Completed {len(summary)} simulations. Summary: {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument(
        "--archetypes",
        default="all",
        help="Comma-separated archetype IDs or 'all'.",
    )
    parser.add_argument(
        "--packages",
        default="P00",
        help="Comma-separated package IDs or 'all'. Use P00 for pre-retrofit.",
    )
    parser.add_argument(
        "--weather-source",
        choices=["epw", "pvgis", "climatedata"],
        default="epw",
        help="Weather source passed to ISO52016. Default: epw.",
    )
    parser.add_argument(
        "--zone-b-epw",
        default=str(DEFAULT_ZONE_B_EPW),
        help="EPW file used for STREPIN climate zone B when --weather-source epw.",
    )
    parser.add_argument(
        "--zone-e-epw",
        default=str(DEFAULT_ZONE_E_EPW),
        help="EPW file used for STREPIN climate zone E when --weather-source epw.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--system-chain",
        choices=["en-standards", "simple-postprocess"],
        default="en-standards",
        help=(
            "Post-ISO system calculation. 'en-standards' runs the EN generation "
            "modules for heating/DHW, EN 16798-13 for cooling and EN 15316-4-6 "
            "for PV. 'simple-postprocess' keeps the earlier annual efficiency "
            "summary. Default: en-standards."
        ),
    )
    parser.add_argument(
        "--air-change-rate",
        type=float,
        default=0.30,
        help="Base residential air change rate [1/h] used to build custom Hve.",
    )
    parser.add_argument(
        "--cooling-eer",
        type=float,
        default=3.0,
        help="Annual EER used only in the summary post-processing layer.",
    )
    parser.add_argument(
        "--no-cooling-postprocess",
        action="store_true",
        help="Exclude cooling electricity from the annual summary post-processing.",
    )
    parser.add_argument(
        "--use-workbook-capacity",
        action="store_true",
        help=(
            "Use workbook heating capacity in ISO52016. By default the runner uses "
            "large ideal capacity to calculate unconstrained energy need."
        ),
    )
    parser.add_argument(
        "--write-hourly",
        action="store_true",
        help="Write one hourly ISO52016 CSV per simulated case.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build cases and write an index without running ISO52016.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cases, useful for smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
