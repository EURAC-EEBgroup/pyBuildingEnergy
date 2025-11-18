# -*- coding: utf-8 -*-
import os
import json
import argparse
import sys
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


from pybuildingenergy.source.check_input import sanitize_and_validate_BUI, check_heating_system_inputs
from pybuildingenergy.source.utils import ISO52016
from pybuildingenergy.source.graphs import Graphs_and_report
from pybuildingenergy.source.iso_15316_1 import HeatingSystemCalculator



# =========================
#   DEFAULT INPUTS (BUI)
# =========================
BUI = {
    "building": {
        "name": "test-cy",
        "azimuth_relative_to_true_north": 41.8,
        "latitude": 46.49018685497359,
        "longitude": 11.327028776009655,
        "exposed_perimeter": 40,
        "height": 3,
        "wall_thickness": 0.3,
        "n_floors": 1,
        "building_type_class": "Residential_apartment",
        "adj_zones_present": False,
        "number_adj_zone": 2,
        "net_floor_area": 100,
        "construction_class": "class_i",
    },
    "adjacent_zones": [],
    "building_surface": [
        {"name": "Roof surface", "type": "opaque", "area": 130, "sky_view_factor": 1.0, "u_value": 2.2,
         "solar_absorptance": 0.4, "thermal_capacity": 741500.0, "orientation": {"azimuth": 0, "tilt": 0},
         "name_adj_zone": None},
        {"name": "Opaque north surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5, "u_value": 1.4,
         "solar_absorptance": 0.4, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 0, "tilt": 90},
         "name_adj_zone": None},
        {"name": "Opaque south surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5, "u_value": 1.4,
         "solar_absorptance": 0.4, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 180, "tilt": 90},
         "name_adj_zone": None},
        {"name": "Opaque east surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5, "u_value": 1.2,
         "solar_absorptance": 0.6, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 90, "tilt": 90},
         "name_adj_zone": None},
        {"name": "Opaque west surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5, "u_value": 1.2,
         "solar_absorptance": 0.7, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 270, "tilt": 90},
         "name_adj_zone": None},
        {"name": "Slab to ground", "type": "opaque", "area": 100, "sky_view_factor": 0.0, "u_value": 1.6,
         "solar_absorptance": 0.6, "thermal_capacity": 405801, "orientation": {"azimuth": 0, "tilt": 0},
         "name_adj_zone": None},
        {"name": "Transparent east surface", "type": "transparent", "area": 4, "sky_view_factor": 0.5, "u_value": 5,
         "g_value": 0.726, "height": 2, "width": 1, "parapet": 1.1,
         "orientation": {"azimuth": 90, "tilt": 90}, "shading": False,
         "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5,
         "overhang_proprieties": {"width_of_horizontal_overhangs": 1}, "name_adj_zone": None},
        {"name": "Transparent west surface", "type": "transparent", "area": 4, "sky_view_factor": 0.5, "u_value": 5,
         "g_value": 0.726, "height": 2, "width": 1, "parapet": 1.1,
         "orientation": {"azimuth": 270, "tilt": 90}, "shading": False,
         "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5,
         "overhang_proprieties": {"width_of_horizontal_overhangs": 1}, "name_adj_zone": None},
    ],
    "units": {
        "area": "m²",
        "u_value": "W/m²K",
        "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
        "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m²",
        "internal_gain_profile": "Normalized to 0-1",
        "HVAC_profile": "0: off, 1: on",
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0,
            "heating_setback": 17.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "units": "°C",
        },
        "system_capacities": {"heating_capacity": 10_000_000.0, "cooling_capacity": 12_000_000.0, "units": "W"},
        "airflow_rates": {"infiltration_rate": 1.0, "units": "ACH (air changes per hour)"},
        "internal_gains": [],
        "construction": {"wall_thickness": 0.3, "thermal_bridges": 2, "units": "m / W/mK"},
        "climate_parameters": {"coldest_month": 1, "units": "1-12"},
        "heating_profile": {"weekday": [0.0]*5 + [1.0]*17 + [0.0]*2, "weekend": [0.0]*5 + [1.0]*17 + [0.0]*2},
        "cooling_profile": {"weekday": [0.0]*5 + [1.0]*17 + [0.0]*2, "weekend": [0.0]*7 + [1.0]*14 + [0.0]*3},
        "ventilation_profile": {"weekday": [0.0]*5 + [1.0]*17 + [0.0]*2, "weekend": [0.0]*7 + [1.0]*14 + [0.0]*3},
    },
}

# =========================
#  DEFAULT HVAC INPUTS
# =========================
INPUT_SYSTEM_HVAC = {
    "emitter_type": "Floor heating 1",
    "nominal_power": 8,
    "emission_efficiency": 90,
    "flow_temp_control_type": "Type 2 - Based on outdoor temperature",
    "selected_emm_cont_circuit": 0,
    "mixing_valve": True,
    "mixing_valve_delta": 2,
    "heat_losses_recovered": True,
    "distribution_loss_recovery": 90,
    "simplified_approach": 80,
    "distribution_aux_recovery": 80,
    "distribution_aux_power": 30,
    "distribution_loss_coeff": 48,
    "distribution_operation_time": 1,
    "full_load_power": 27,  # kW
    "max_monthly_load_factor": 100,
    "tH_gen_i_ON": 1,  # h
    "auxiliary_power_generator": 0,
    "fraction_of_auxiliary_power_generator": 40,
    "generator_circuit": "independent",
    "gen_flow_temp_control_type": "Type A - Based on outdoor temperature",
    "gen_outdoor_temp_data": pd.DataFrame(
        {"θext_min_gen": [-7], "θext_max_gen": [15], "θflw_gen_max": [60], "θflw_gen_min": [35]},
        index=["Generator curve"],
    ),
    "speed_control_generator_pump": "variable",
    "generator_nominal_deltaT": 20,  # °C
    "efficiency_model": "simple",
    "calc_when_QH_positive_only": False,
    "off_compute_mode": "full",
}


# =========================
#  HELPER FUNCTIONS I/O
# =========================
def _load_json_if_path(maybe_path: str) -> Dict[str, Any]:
    """Se maybe_path esiste come file, carica e ritorna il JSON; altrimenti alza errore."""
    with open(maybe_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_output_dir(base_dir: str) -> str:
    out = os.path.abspath(base_dir)
    os.makedirs(out, exist_ok=True)
    return out


# =========================
#  MAIN PIPELINE 
# =========================
def run_pipeline(
    bui: Dict[str, Any],
    hvac_cfg: Dict[str, Any],
    weather_source: str,
    path_weather_file: str,
    out_dir: str,
    report_name: str,
) -> Tuple[str, str, str]:
    """
    Esegue: QC BUI -> ISO52016 -> ISO15316-1 -> grafici
    Ritorna: (path_hourly_iso52016, path_annual_iso52016, path_hourly_iso15316)
    """
    # 1) HVAC QC
    hvac_checked = check_heating_system_inputs(hvac_cfg)
    print("Emitter scelto:", hvac_checked["emitter_type"])
    for m in hvac_checked["messages"]:
        print(" -", m)
    hvac_cfg = hvac_checked["config"]

    # 2) BUI QC
    bui_fixed, report = sanitize_and_validate_BUI(bui, fix=True)
    for r in report:
        lvl = r["level"]
        print(f"[{lvl}] {r['path']}: {r['msg']}" + (" (fix applied)" if r["fix_applied"] else ""))

    bui_checked, issues = sanitize_and_validate_BUI(bui_fixed, fix=False)
    errors = [e for e in issues if e["level"] == "ERROR"]
    if errors:
        print("\n❌ Errors in BUI input data — simulation interrupted:")
        for e in errors:
            print(f" - {e['path']}: {e['msg']}")
        raise ValueError("Invalid BUI input")

    print("\n✅ BUI valid — starting ISO 52016 simulation...")
    os.makedirs(out_dir, exist_ok=True)

    # 3) ISO 52016
    iso52016 = ISO52016()
    hourly_sim, annual_results_df = iso52016.Temperature_and_Energy_needs_calculation(
        bui_checked, weather_source=weather_source, path_weather_file=path_weather_file
    )
    hourly_path = os.path.join(out_dir, "hourly_sim__ISO52016.csv")
    annual_path = os.path.join(out_dir, "annual_sim__ISO52016.csv")
    hourly_sim.to_csv(hourly_path, index=False)
    annual_results_df.to_csv(annual_path, index=False)

    # 4) ISO 15316-1 (systems)
    calc = HeatingSystemCalculator(hvac_cfg)
    df_in = calc.load_csv_data(hourly_sim)  # se accetta DF
    df_out = calc.run_timeseries()
    iso15316_path = os.path.join(out_dir, "hourly_heating_system__ISO15316.csv")
    df_out.to_csv(iso15316_path, index=False)

    # 5) Graphs and report
    Graphs_and_report(
        df=hourly_sim,
        season="heating_cooling",
        building_area=bui_checked["building"]["net_floor_area"],
    ).bui_analysis_page(folder_directory=out_dir, name_file=report_name)

    print("\n✅ Done.")
    print(f" - ISO52016 hourly : {hourly_path}")
    print(f" - ISO52016 annual : {annual_path}")
    print(f" - ISO15316 hourly : {iso15316_path}")
    return hourly_path, annual_path, iso15316_path


# =========================
#        __main__
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ISO 52016 (fabric) + ISO 15316-1 (heating system) on a BUI JSON."
    )
    p.add_argument("--bui", type=str, default=None,
                   help="Path a un file JSON BUI (se omesso, usa quello interno).")
    p.add_argument("--hvac", type=str, default=None,
                   help="Path a un file JSON HVAC (se omesso, usa quello interno).")
    p.add_argument("--weather-source", type=str, default="pvgis", choices=["pvgis", "epw"],
                   help="Fonte meteo: pvgis oppure epw.")
    p.add_argument("--epw", type=str, default=None,
                   help="Percorso file EPW (richiesto se --weather-source epw).")
    p.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "Result"),
                   help="Cartella risultati (default: ./Result).")
    p.add_argument("--report-name", type=str, default="main_report",
                   help="Nome report HTML/PDF generato.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Upload BUI/HVAC if given, otherwise use the default data
    bui = BUI
    hvac_cfg = INPUT_SYSTEM_HVAC
    if args.bui:
        try:
            bui = _load_json_if_path(args.bui)
        except Exception as e:
            print(f"Errore nel caricare BUI JSON: {e}")
            sys.exit(1)
    if args.hvac:
        try:
            hvac_cfg = _load_json_if_path(args.hvac)
        except Exception as e:
            print(f"Errore nel caricare HVAC JSON: {e}")
            sys.exit(1)

    # Evaluation of meteo coerhency 
    weather_source = args.weather_source
    epw_path = args.epw if weather_source == "epw" else None
    if weather_source == "epw" and not epw_path:
        print("Devi specificare --epw PATH quando --weather-source=epw.")
        sys.exit(2)

    out_dir = _prepare_output_dir(args.outdir)

    try:
        run_pipeline(
            bui=bui,
            hvac_cfg=hvac_cfg,
            weather_source=weather_source,
            path_weather_file=epw_path,
            out_dir=out_dir,
            report_name=args.report_name,
        )
    except Exception as exc:
        print(f"\n❌ Simulazione fallita: {exc}")
        sys.exit(3)
