"""Run pyBuildingEnergy simulations for Italian STREPIN archetype packages.

The workbook defines the archetype geometry and package levels. This script
builds pre/post-retrofit BUI cases, runs ISO52016, and writes an engine-derived
summary that can be compared with the workbook's estimate columns.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
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


WORKBOOK_MEASURE_CODES = {"wall", "roof_floor", "windows", "heating", "pv"}


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


def _parse_measure_levels(raw: str | None) -> dict[str, int] | None:
    if raw is None or not str(raw).strip():
        return None
    levels: dict[str, int] = {}
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "--measure-levels entries must be code=level, e.g. "
                "wall=2,roof_floor=3,windows=2,heating=3,pv=2."
            )
        key, value = token.split("=", 1)
        key = key.strip()
        if key not in WORKBOOK_MEASURE_CODES:
            raise ValueError(
                f"Unknown workbook measure {key!r}. "
                f"Available: {', '.join(sorted(WORKBOOK_MEASURE_CODES))}."
            )
        levels[key] = int(value)
    return levels


def _parse_extra_measure_specs(
    raw: str | None,
    raw_json: str | None = None,
) -> list[dict]:
    specs: list[dict] = []
    if raw:
        for item in str(raw).split(","):
            token = item.strip()
            if not token:
                continue
            specs.append(_extra_measure_spec_from_token(token))
    if raw_json:
        loaded = json.loads(raw_json)
        if isinstance(loaded, dict):
            loaded = [loaded]
        for item in loaded:
            if not isinstance(item, dict):
                raise ValueError("--extra-measure-json must contain objects.")
            specs.append(item)
    return specs


def _extra_measure_spec_from_token(token: str) -> dict:
    normalized = token.strip().lower().replace("_", "-")
    if normalized in {"lighting", "lighting-led", "led-lighting"}:
        return {
            "measure_code": "lighting",
            "parameters": {"installed_power_density_W_m2": 4.0},
        }
    if normalized in {"lighting-controls", "lighting-led-controls"}:
        return {
            "measure_code": "lighting",
            "parameters": {
                "installed_power_density_W_m2": 4.0,
                "daylight_dependency_factor": 0.80,
                "occupancy_dependency_factor": 0.85,
                "daylight_control_fraction": 0.45,
                "occupancy_control_fraction": 0.35,
            },
        }
    if normalized in {"ventilation", "ventilation-hrv", "hrv"}:
        return {
            "measure_code": "ventilation",
            "parameters": {
                "heat_recovery_efficiency": 0.75,
                "mechanical_fraction": 1.0,
                "specific_fan_power_W_s_m3": 900.0,
            },
        }
    if normalized.startswith("bacs-"):
        return {
            "measure_code": "bacs",
            "parameters": {"bacs_class": normalized.split("-", 1)[1].upper()},
        }
    if normalized in {"bacs"}:
        return {"measure_code": "bacs", "parameters": {"bacs_class": "B"}}
    if normalized in {"biomass", "biomass-boiler", "pellet-boiler"}:
        return {"measure_code": "biomass", "parameters": {}}
    if normalized in {"district", "district-heat", "district-heating"}:
        return {"measure_code": "district_heat", "parameters": {}}
    if normalized in {"chp", "cogeneration"}:
        return {"measure_code": "chp", "parameters": {}}
    if normalized in {"solar-thermal", "solarthermal"}:
        return {"measure_code": "solar_thermal", "parameters": {}}
    if normalized == "battery":
        return {"measure_code": "battery", "parameters": {"capacity_kWh": 5.0}}
    if normalized.startswith("battery-"):
        capacity = float(normalized.split("-", 1)[1].replace("kwh", ""))
        return {"measure_code": "battery", "parameters": {"capacity_kWh": capacity}}
    if normalized in {"pv-shading", "pv-shading-10"}:
        return {
            "measure_code": "pv_shading",
            "parameters": {"shading_loss_fraction": 0.10},
        }
    if normalized.startswith("pv-shading-"):
        pct = float(normalized.rsplit("-", 1)[1])
        return {
            "measure_code": "pv_shading",
            "parameters": {"shading_loss_fraction": pct / 100.0},
        }
    raise ValueError(
        f"Unknown extra measure token {token!r}. Use tokens such as "
        "lighting-led, ventilation-hrv, bacs-a, biomass-boiler, "
        "solar-thermal, battery-5, pv-shading-10, district-heat or chp."
    )


def _case_with_extra_measures(
    case: pybui.StrepinCase,
    extra_specs: list[dict],
) -> pybui.StrepinCase:
    if not extra_specs:
        return case
    bui = pybui.apply_extra_measure_specs_to_bui(case.bui, extra_specs)
    labels = [
        str(spec.get("measure_code", spec.get("code", ""))).replace("_", "-")
        for spec in extra_specs
    ]
    package = dict(case.package)
    package["Extra_Measures"] = ",".join(labels)
    package["Package_Type"] = f"{package.get('Package_Type', '')} + engine overlays"
    return replace(case, bui=bui, package=package)


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
    for source, target in [
        ("GHI", "GHI"),
        ("ghi", "GHI"),
        ("I_sol_global_W_m2", "GHI"),
        ("global_horizontal_irradiance_W_m2", "GHI"),
        ("lighting_profile", "lighting_profile"),
        ("ventilation_profile", "ventilation_profile"),
        ("occupancy_profile", "occupancy_profile"),
    ]:
        if source in hourly and target not in loads:
            loads[target] = hourly[source]
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


def _building_area(case: pybui.StrepinCase) -> float:
    return float(case.archetype.get("Af_m2", 0.0) or 0.0)


def _building_volume(case: pybui.StrepinCase) -> float:
    building = case.bui.get("building", {})
    volume = float(building.get("volume", 0.0) or 0.0)
    if volume > 0.0:
        return volume
    return _building_area(case) * float(building.get("floor_height", 3.0) or 3.0)


def _run_bacs_on_loads(
    loads: pd.DataFrame,
    bacs_config: dict,
) -> tuple[pd.DataFrame, pybui.BACSSimulationResult | None]:
    if not bacs_config:
        return loads, None
    result = pybui.BACSControlFactorCalculator(bacs_config).run_timeseries(
        loads[["Q_H_kWh", "Q_C_kWh", "Q_W_kWh"]]
    )
    adjusted = loads.copy()
    for col in ["Q_H_kWh", "Q_C_kWh", "Q_W_kWh"]:
        bacs_col = f"{col}_bacs"
        if bacs_col in result.timeseries:
            adjusted[col] = result.timeseries[bacs_col]
    return adjusted, result


def _service_factor(bacs_result: pybui.BACSSimulationResult | None, service: str) -> float:
    if bacs_result is None:
        return 1.0
    value = bacs_result.summary.get(f"{service}_factor", 1.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _run_lighting(
    case: pybui.StrepinCase,
    loads: pd.DataFrame,
    bacs_result: pybui.BACSSimulationResult | None,
) -> tuple[float, pybui.LightingSimulationResult | None]:
    lighting_config = case.bui.get("strepin", {}).get("lighting_system", {})
    if not lighting_config:
        return 0.0, None
    config = {"area_m2": _building_area(case), **lighting_config}
    result = pybui.LightingSystemCalculator(config).run_timeseries(loads)
    electricity = result.summary.get("E_L_total_kWh", 0.0) * _service_factor(
        bacs_result, "lighting"
    )
    return float(electricity), result


def _run_ventilation(
    case: pybui.StrepinCase,
    loads: pd.DataFrame,
    bacs_result: pybui.BACSSimulationResult | None,
) -> tuple[float, pybui.VentilationSystemSimulationResult | None]:
    ventilation_config = case.bui.get("strepin", {}).get("ventilation_system", {})
    if not ventilation_config:
        return 0.0, None
    config = {
        "volume_m3": _building_volume(case),
        **ventilation_config,
    }
    vloads = loads.assign(T_zone=20.0)
    result = pybui.VentilationSystemCalculator(config).run_timeseries(vloads)
    electricity = result.summary.get("E_vent_fan_kWh", 0.0) * _service_factor(
        bacs_result, "ventilation"
    )
    return float(electricity), result


def _apply_solar_thermal(
    case: pybui.StrepinCase,
    loads: pd.DataFrame,
) -> tuple[pd.DataFrame, pybui.RenewableEnergySimulationResult | None]:
    solar = case.bui.get("strepin", {}).get("solar_thermal", {})
    if not solar or not bool(solar.get("enabled", True)):
        return loads, None
    result = pybui.RenewableEnergySystemCalculator(
        {
            "demand_unit": "kWh",
            "solar_thermal": solar,
            "pv": {"enabled": False, "capacity_kWp": 0.0},
        }
    ).run_timeseries(loads)
    adjusted = loads.copy()
    adjusted["Q_H_kWh"] = result.timeseries["Q_H_after_solar_kWh"]
    adjusted["Q_W_kWh"] = result.timeseries["Q_W_after_solar_kWh"]
    return adjusted, result


def _primary_factor_config(
    tables: pybui.ItalianStrepinTables,
    *,
    export_credit_method: str = "none",
) -> dict:
    gas_pef = float(tables.assumption_value("Gas_Primary_Energy_Factor", 1.05) or 1.05)
    electricity_pef = float(
        tables.assumption_value("Electricity_Primary_Energy_Factor", 2.17) or 2.17
    )
    return {
        "primary_energy_factors": {
            "natural_gas": {"nonrenewable": gas_pef, "renewable": 0.0},
            "gas": {"nonrenewable": gas_pef, "renewable": 0.0},
            "electricity": {"nonrenewable": electricity_pef, "renewable": 0.0},
            "biomass": {"nonrenewable": 0.20, "renewable": 1.00},
            "district_heat": {"nonrenewable": 0.80, "renewable": 0.20},
        },
        "export_primary_energy_factors": {
            "electricity": {"nonrenewable": electricity_pef, "renewable": 0.0}
        },
        "carriers": ["natural_gas", "electricity", "biomass", "district_heat"],
        "export_credit_method": export_credit_method,
    }


def _economic_components(case: pybui.StrepinCase) -> list[dict]:
    components: list[dict] = []
    for measure in case.measures:
        components.append(
            {
                "name": f"{measure.get('Measure_Code')} L{measure.get('Level')}",
                "investment_EUR": float(measure.get("Investment_EUR") or 0.0),
                "annual_maintenance_EUR": float(
                    measure.get("Annual_Maintenance_EUR") or 0.0
                ),
                "lifetime_years": int(float(measure.get("Lifetime_years") or 30)),
                "replacement_cost_EUR": float(
                    measure.get("Replacement_Cost_PV_EUR") or 0.0
                ),
                "residual_value_EUR": float(measure.get("Residual_Value_PV_EUR") or 0.0),
            }
        )
    for measure in case.bui.get("strepin", {}).get("extra_measures", []):
        components.append(
            {
                "name": str(measure.get("measure_code", "extra")),
                "investment_EUR": float(measure.get("investment_EUR") or 0.0),
                "annual_maintenance_EUR": float(
                    measure.get("annual_maintenance_EUR") or 0.0
                ),
                "lifetime_years": int(float(measure.get("lifetime_years") or 30)),
                "replacement_cost_EUR": float(
                    measure.get("replacement_cost_EUR") or measure.get("investment_EUR") or 0.0
                ),
                "residual_value_EUR": float(measure.get("residual_value_EUR") or 0.0),
            }
        )
    return components


def _cost_result(
    case: pybui.StrepinCase,
    tables: pybui.ItalianStrepinTables,
    annual_energy: dict[str, float],
    annual_exports: dict[str, float],
) -> pybui.EconomicSimulationResult:
    return pybui.CostOptimalityCalculator(
        {
            "calculation_period_years": int(
                float(tables.assumption_value("Calculation_Period", 30) or 30)
            ),
            "real_discount_rate": float(
                tables.assumption_value("Real_Discount_Rate", 0.03) or 0.03
            ),
            "floor_area_m2": _building_area(case),
            "energy_prices": {
                "natural_gas": float(tables.assumption_value("Gas_Price", 0.12) or 0.12),
                "gas": float(tables.assumption_value("Gas_Price", 0.12) or 0.12),
                "electricity": float(
                    tables.assumption_value("Electricity_Price", 0.30) or 0.30
                ),
                "biomass": float(tables.assumption_value("Biomass_Price", 0.08) or 0.08),
                "district_heat": float(
                    tables.assumption_value("District_Heat_Price", 0.11) or 0.11
                ),
            },
            "export_prices": {
                "electricity": float(tables.assumption_value("PV_Export_Value", 0.06) or 0.06)
            },
        }
    ).calculate(
        components=_economic_components(case),
        annual_energy=annual_energy,
        annual_exports=annual_exports,
    )


def _run_strepin_standard_chain(
    case: pybui.StrepinCase,
    hourly: pd.DataFrame,
    tables: pybui.ItalianStrepinTables,
    args: argparse.Namespace,
) -> dict[str, float | str | None]:
    raw_loads = _prepare_system_loads(case, hourly)
    area_m2 = float(case.archetype["Af_m2"])
    metadata = case.bui.get("strepin", {})
    heating_system = metadata.get("heating_system", {})
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
    climate_zone = str(case.archetype["Climate_Zone"])

    loads_after_bacs, bacs_result = _run_bacs_on_loads(
        raw_loads,
        metadata.get("bacs", {}),
    )
    lighting_electricity_kWh, lighting_result = _run_lighting(
        case,
        loads_after_bacs,
        bacs_result,
    )
    ventilation_electricity_kWh, ventilation_result = _run_ventilation(
        case,
        loads_after_bacs,
        bacs_result,
    )
    solar_loads, solar_result = _apply_solar_thermal(case, loads_after_bacs)

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
        ).run_timeseries(solar_loads.assign(Q_C_kWh=0.0))
        heating_summary = heating_result.summary
        gas_final_kWh = 0.0
        biomass_final_kWh = 0.0
        district_heat_final_kWh = 0.0
        chp_el_generation_kWh = 0.0
        chp_el_self_consumed_kWh = 0.0
        chp_el_export_kWh = 0.0
        hp_electricity_kWh = (
            heating_summary.get("EHW_gen_in_kWh", 0.0)
            + heating_summary.get("WHW_gen_aux_kWh", 0.0)
        )
        heating_standard = "EN 15316-4-2"
    elif carrier == "biomass" or "biomass" in system_type:
        heating_result = pybui.BiomassBoilerSystemCalculator(
            {
                "demand_unit": "kWh",
                "nominal_power_kW": nominal_capacity_kW,
                "full_load_efficiency": min(max(efficiency, 0.50), 0.95),
                "part_load_efficiency": min(max(efficiency - 0.04, 0.45), 0.90),
            }
        ).run_timeseries(solar_loads)
        heating_summary = heating_result.summary
        gas_final_kWh = 0.0
        biomass_final_kWh = heating_summary.get("E_biomass_delivered_kWh", 0.0)
        district_heat_final_kWh = 0.0
        chp_el_generation_kWh = 0.0
        chp_el_self_consumed_kWh = 0.0
        chp_el_export_kWh = 0.0
        hp_electricity_kWh = heating_summary.get("WHW_gen_aux_kWh", 0.0)
        heating_standard = "EN 15316-4 biomass/solid-fuel generation"
    elif carrier == "district_heat" or "district" in system_type:
        heating_result = pybui.DistrictEnergySystemCalculator(
            {
                "demand_unit": "kWh",
                "cooling_enabled": False,
                "heating_substation_efficiency": min(max(efficiency, 0.70), 1.0),
                "heating_distribution_loss_fraction": 0.03,
                "auxiliary_power_kW": 0.010,
            }
        ).run_timeseries(solar_loads)
        heating_summary = heating_result.summary
        gas_final_kWh = 0.0
        biomass_final_kWh = 0.0
        district_heat_final_kWh = heating_summary.get("EHW_district_in_kWh", 0.0)
        chp_el_generation_kWh = 0.0
        chp_el_self_consumed_kWh = 0.0
        chp_el_export_kWh = 0.0
        hp_electricity_kWh = heating_summary.get("WHW_gen_aux_kWh", 0.0)
        heating_standard = "EN 15316-4-5"
    elif "cogeneration" in system_type or system_type == "chp":
        heating_result = pybui.CogenerationSystemCalculator(
            {
                "demand_unit": "kWh",
                "fuel": "natural_gas",
                "nominal_thermal_power_kW": nominal_capacity_kW,
                "thermal_efficiency": min(max(efficiency, 0.35), 0.70),
                "electrical_efficiency": float(
                    heating_system.get("electrical_efficiency", 0.30) or 0.30
                ),
                "backup_efficiency": 0.92,
            }
        ).run_timeseries(solar_loads)
        heating_summary = heating_result.summary
        gas_final_kWh = heating_summary.get("E_chp_fuel_in_kWh", 0.0)
        biomass_final_kWh = 0.0
        district_heat_final_kWh = 0.0
        chp_el_generation_kWh = heating_summary.get("E_chp_el_generated_kWh", 0.0)
        chp_el_self_consumed_kWh = 0.0
        chp_el_export_kWh = 0.0
        hp_electricity_kWh = heating_summary.get("WHW_gen_aux_kWh", 0.0)
        heating_standard = "EN 15316-4-4"
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
        ).run_timeseries(solar_loads)
        heating_summary = heating_result.summary
        gas_final_kWh = heating_summary.get("EHW_fuel_in_kWh", 0.0)
        biomass_final_kWh = 0.0
        district_heat_final_kWh = 0.0
        chp_el_generation_kWh = 0.0
        chp_el_self_consumed_kWh = 0.0
        chp_el_export_kWh = 0.0
        hp_electricity_kWh = heating_summary.get("WHW_gen_aux_kWh", 0.0)
        heating_standard = "EN 15316-4-1"

    cooling_result = _strepin_cooling_result(
        loads_after_bacs,
        cooling_eer=args.cooling_eer,
        nominal_capacity_kW=nominal_capacity_kW * 0.6,
    )
    cooling_summary = cooling_result.summary
    cooling_electricity_kWh = cooling_summary.get("EC_total_kWh", 0.0)
    aux_electricity_kWh = (
        float(case.archetype.get("Baseline_Aux_Elec", 0.0) or 0.0)
        * area_m2
        * _service_factor(bacs_result, "auxiliary")
    )
    gross_electricity_kWh = (
        hp_electricity_kWh
        + cooling_electricity_kWh
        + aux_electricity_kWh
        + lighting_electricity_kWh
        + ventilation_electricity_kWh
    )
    if chp_el_generation_kWh > 0.0:
        chp_el_self_consumed_kWh = min(gross_electricity_kWh, chp_el_generation_kWh)
        chp_el_export_kWh = max(chp_el_generation_kWh - chp_el_self_consumed_kWh, 0.0)
    electric_load_after_chp_kWh = max(gross_electricity_kWh - chp_el_self_consumed_kWh, 0.0)

    pv = metadata.get("pv_system", {})
    pv_kWp = float(pv.get("pv_kWp", 0.0) or 0.0)
    pv_yield = float(tables.assumption_value(f"PV_Yield_Zone_{climate_zone}", 0.0) or 0.0)
    battery = metadata.get("battery", {})
    pv_self_consumption_mode = "dispatch" if battery else "fixed_fraction"
    pv_loads = solar_loads.assign(
        E_site_el_load_kWh=electric_load_after_chp_kWh / max(len(solar_loads), 1)
    )
    pv_result = pybui.RenewableEnergySystemCalculator(
        {
            "demand_unit": "kWh",
            "solar_thermal": {"enabled": False, "area_m2": 0.0},
            "pv": {
                "enabled": pv_kWp > 0.0,
                "capacity_kWp": pv_kWp,
                "annual_yield_kWh_kWp": pv_yield,
                "performance_ratio": 0.80,
                "shading_loss_fraction": float(pv.get("shading_loss_fraction", 0.0) or 0.0),
                "orientation_loss_fraction": float(
                    pv.get("orientation_loss_fraction", 0.0) or 0.0
                ),
                "self_consumption_fraction": float(
                    tables.assumption_value("PV_SelfConsumption_Factor", 0.65) or 0.65
                ),
                "self_consumption_mode": pv_self_consumption_mode,
                "battery": battery,
            },
        }
    ).run_timeseries(pv_loads)
    pv_summary = pv_result.summary
    grid_electricity_kWh = pv_summary.get("E_grid_after_PV_kWh", electric_load_after_chp_kWh)
    pv_generation_kWh = pv_summary.get("E_PV_gen_kWh", 0.0)
    pv_self_consumed_kWh = pv_summary.get("E_PV_self_consumed_kWh", 0.0)
    pv_export_kWh = pv_summary.get("E_PV_export_kWh", 0.0)
    exported_electricity_kWh = pv_export_kWh + chp_el_export_kWh

    final_energy_kWh = (
        gas_final_kWh
        + biomass_final_kWh
        + district_heat_final_kWh
        + grid_electricity_kWh
    )
    primary_result = pybui.PrimaryEnergyAccountingCalculator(
        _primary_factor_config(
            tables,
            export_credit_method=args.export_credit_method,
        )
    ).run_annual(
        {
            "E_delivered_natural_gas_kWh": gas_final_kWh,
            "E_delivered_biomass_kWh": biomass_final_kWh,
            "E_delivered_district_heat_kWh": district_heat_final_kWh,
            "E_delivered_electricity_kWh": grid_electricity_kWh,
            "E_exported_electricity_kWh": exported_electricity_kWh,
        }
    )
    primary_summary = primary_result.summary
    annual_energy_by_carrier = {
        "natural_gas": gas_final_kWh,
        "biomass": biomass_final_kWh,
        "district_heat": district_heat_final_kWh,
        "electricity": grid_electricity_kWh,
    }
    cost_result = _cost_result(
        case,
        tables,
        annual_energy_by_carrier,
        {"electricity": exported_electricity_kWh},
    )
    cost_summary = cost_result.summary
    reference = case.workbook_reference or {}
    solar_summary = solar_result.summary if solar_result is not None else {}
    lighting_summary = lighting_result.summary if lighting_result is not None else {}
    ventilation_summary = ventilation_result.summary if ventilation_result is not None else {}
    bacs_summary = bacs_result.summary if bacs_result is not None else {}

    return {
        "Archetype_ID": case.archetype_id,
        "Package_ID": case.package_id,
        "Extra_Measures": case.package.get("Extra_Measures", ""),
        "Building_Type": case.archetype.get("Building_Type"),
        "Type_Code": case.archetype.get("Type_Code"),
        "Climate_Zone": climate_zone,
        "Period_Code": case.archetype.get("Period_Code"),
        "Af_m2": area_m2,
        "System_chain": "en_standards",
        "Heating_generation_standard": heating_standard,
        "Cooling_generation_standard": "EN 16798-13",
        "PV_standard": "EN 15316-4-6",
        "Solar_thermal_standard": "EN 15316-4-3" if solar_result is not None else None,
        "Lighting_standard": "EN 15193-1" if lighting_result is not None else None,
        "Ventilation_standard": "EN 16798-5/7" if ventilation_result is not None else None,
        "BACS_standard": "EN 15232-1 / EN ISO 52120-1" if bacs_result is not None else None,
        "Primary_energy_standard": "EN ISO 52000-1",
        "Economic_standard": "EN 15459-1",
        "Q_H_engine_kWh_a": raw_loads["Q_H_kWh"].sum(),
        "Q_C_engine_kWh_a": raw_loads["Q_C_kWh"].sum(),
        "Q_H_after_BACS_solar_kWh_a": solar_loads["Q_H_kWh"].sum(),
        "Q_C_after_BACS_kWh_a": loads_after_bacs["Q_C_kWh"].sum(),
        "Q_H_engine_kWh_m2a": raw_loads["Q_H_kWh"].sum() / area_m2 if area_m2 > 0 else 0.0,
        "Q_C_engine_kWh_m2a": raw_loads["Q_C_kWh"].sum() / area_m2 if area_m2 > 0 else 0.0,
        "DHW_useful_kWh_a": raw_loads["Q_W_kWh"].sum(),
        "DHW_after_BACS_solar_kWh_a": solar_loads["Q_W_kWh"].sum(),
        "Solar_thermal_used_kWh_a": solar_summary.get("Q_solar_thermal_used_kWh", 0.0),
        "QHW_gen_out_kWh_a": heating_summary.get("QHW_gen_out_kWh", 0.0),
        "EHW_gen_in_kWh_a": heating_summary.get("EHW_gen_in_kWh", 0.0),
        "SPF_or_eta_HW_gen": heating_summary.get(
            "SPF_HW_gen", heating_summary.get("eta_HW_gen", None)
        ),
        "Gas_final_kWh_a": gas_final_kWh,
        "Biomass_final_kWh_a": biomass_final_kWh,
        "District_heat_final_kWh_a": district_heat_final_kWh,
        "HeatPump_electricity_kWh_a": hp_electricity_kWh if carrier == "electricity" else 0.0,
        "Heating_aux_electricity_kWh_a": heating_summary.get("WHW_gen_aux_kWh", 0.0),
        "CHP_electricity_generated_kWh_a": chp_el_generation_kWh,
        "CHP_self_consumed_kWh_a": chp_el_self_consumed_kWh,
        "CHP_export_kWh_a": chp_el_export_kWh,
        "Cooling_electricity_kWh_a": cooling_electricity_kWh,
        "Cooling_SEER": cooling_summary.get("SEER_C_gen", None),
        "Aux_electricity_kWh_a": aux_electricity_kWh,
        "Lighting_electricity_kWh_a": lighting_electricity_kWh,
        "Lighting_LENI_kWh_m2a": lighting_summary.get("LENI_kWh_m2a"),
        "Ventilation_fan_electricity_kWh_a": ventilation_electricity_kWh,
        "Ventilation_heat_recovered_kWh_a": ventilation_summary.get(
            "Q_vent_heat_recovered_kWh", 0.0
        ),
        "BACS_class": bacs_summary.get("BACS_class"),
        "Gross_electricity_kWh_a": gross_electricity_kWh,
        "Electric_load_after_CHP_kWh_a": electric_load_after_chp_kWh,
        "PV_kWp": pv_kWp,
        "PV_generation_kWh_a": pv_generation_kWh,
        "PV_self_consumed_kWh_a": pv_self_consumed_kWh,
        "PV_export_kWh_a": pv_export_kWh,
        "Battery_capacity_kWh": float(battery.get("capacity_kWh", 0.0) or 0.0),
        "Battery_discharge_to_load_kWh_a": pv_summary.get(
            "E_battery_discharge_to_load_kWh", 0.0
        ),
        "Grid_electricity_kWh_a": grid_electricity_kWh,
        "Exported_electricity_kWh_a": exported_electricity_kWh,
        "Final_energy_kWh_a": final_energy_kWh,
        "Final_energy_kWh_m2a": final_energy_kWh / area_m2 if area_m2 > 0 else 0.0,
        "Primary_nonren_kWh_a": primary_summary.get("PE_net_nonren_kWh", 0.0),
        "Primary_nonren_kWh_m2a": (
            primary_summary.get("PE_net_nonren_kWh", 0.0) / area_m2 if area_m2 > 0 else 0.0
        ),
        "Primary_delivered_nonren_kWh_a": primary_summary.get(
            "PE_delivered_nonren_kWh", 0.0
        ),
        "Primary_delivered_nonren_kWh_m2a": (
            primary_summary.get("PE_delivered_nonren_kWh", 0.0) / area_m2
            if area_m2 > 0
            else 0.0
        ),
        "Primary_total_kWh_a": primary_summary.get("PE_net_total_kWh", 0.0),
        "Primary_export_credit_nonren_kWh_a": primary_summary.get(
            "PE_export_credit_nonren_kWh", 0.0
        ),
        "Export_credit_method": args.export_credit_method,
        "Investment_EUR": cost_summary.get("Investment_EUR", 0.0),
        "Annual_energy_cost_EUR": cost_summary.get("Annual_energy_cost_EUR", 0.0),
        "Annual_export_value_EUR": cost_summary.get("Annual_export_value_EUR", 0.0),
        "Annual_running_cost_EUR": cost_summary.get("Annual_running_cost_EUR", 0.0),
        "Global_cost_EUR": cost_summary.get("Global_cost_EUR", 0.0),
        "Global_cost_EUR_m2": cost_summary.get("Global_cost_EUR_m2", 0.0),
        "Annualized_global_cost_EUR_m2a": cost_summary.get(
            "Annualized_global_cost_EUR_m2a", 0.0
        ),
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
    measure_levels = _parse_measure_levels(args.measure_levels)
    extra_specs = _parse_extra_measure_specs(args.extra_measures, args.extra_measure_json)

    if measure_levels:
        label = "CUSTOM_" + "_".join(
            f"{code}{level}" for code, level in sorted(measure_levels.items())
        )
        cases = [
            tables.make_measure_case(
                archetype_id,
                measure_levels,
                package_id=label,
                ideal_hvac_capacity=not args.use_workbook_capacity,
                air_change_rate_h=args.air_change_rate,
            )
            for archetype_id in archetype_ids
        ]
    else:
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
    cases = [_case_with_extra_measures(case, extra_specs) for case in cases]
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
                    "extra_measures": case.package.get("Extra_Measures", ""),
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
        "--measure-levels",
        default=None,
        help=(
            "Build arbitrary workbook measure combinations instead of package IDs, "
            "for example wall=2,roof_floor=3,windows=2,heating=3,pv=2. "
            "Missing levels default to 1."
        ),
    )
    parser.add_argument(
        "--extra-measures",
        default=None,
        help=(
            "Comma-separated engine-side measures to apply on top of the package "
            "or custom levels. Supported tokens include lighting-led, "
            "lighting-controls, ventilation-hrv, bacs-a, bacs-b, biomass-boiler, "
            "district-heat, chp, solar-thermal, battery-5 and pv-shading-10."
        ),
    )
    parser.add_argument(
        "--extra-measure-json",
        default=None,
        help=(
            "JSON object or list of objects with measure_code and parameters, "
            "used for custom values not covered by --extra-measures."
        ),
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
        "--export-credit-method",
        choices=["none", "symmetric"],
        default="none",
        help=(
            "EN ISO 52000-1 exported-electricity primary-energy credit. "
            "'none' reports exports but does not credit them against the headline "
            "non-renewable primary indicator, matching the previous STREPIN "
            "comparison logic; 'symmetric' credits exported electricity with the "
            "configured export primary factor."
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
