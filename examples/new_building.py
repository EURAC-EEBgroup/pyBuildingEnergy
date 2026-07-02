import sys
from pathlib import Path

# Ensure local package import works when running the script directly
# (package sources live in ../src).
EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (SRC_DIR, PROJECT_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import pybuildingenergy as pybui
import numpy as np
import pandas as pd 
import plotly.express as px 
from pybuildingenergy.source.utils import *
from pybuildingenergy.source.check_input import sanitize_and_validate_BUI, check_heating_system_inputs
from pybuildingenergy.source.graphs import Graphs_and_report
from pybuildingenergy.source.graphs import *
from pybuildingenergy.source.iso_15316_1 import HeatingSystemCalculator
from pybuildingenergy.source.example_inputs import (
    get_example_cooling_input,
    get_example_hvac_input,
)
from pybuildingenergy.source.hvac_html_reports import _build_temperature_report
from pybuildingenergy.source.hvac_quality_checks import validate_hvac_input_coherence
from pybuildingenergy.source.hvac_sankey_reports import (
    _build_cooling_sankey_report,
    _build_sankey_consumption_report,
)
from pybuildingenergy.source.generate_profile import HourlyProfileGenerator, get_country_code_from_latlon
from pybuildingenergy.source.DHW import *
from pybuildingenergy.source.ventilation import *
from pybuildingenergy.source.table_iso_16798_1 import *

WEATHER_CANDIDATES = [
    EXAMPLES_DIR / "2050_Athens.epw",
    EXAMPLES_DIR / "2020_Milan.epw",
]
# WEATHER_FILE = next((p for p in WEATHER_CANDIDATES if p.exists()), None)
WEATHER_FILE = None
WEATHER_SOURCE = "epw" if WEATHER_FILE is not None else "pvgis"
print(WEATHER_FILE)

GENERATE_EXTRA_REPORTS = False

def _run_iso52016(building_obj):
    kwargs = {
        "weather_source": WEATHER_SOURCE,
        "latent_indoor_rh_pct": INPUT_SYSTEM_HVAC.get("latent_indoor_rh_pct", 50.0),
    }
    if WEATHER_SOURCE == "epw":
        kwargs["path_weather_file"] = str(WEATHER_FILE)

    out = ISO52016.Temperature_and_Energy_needs_calculation(building_obj, **kwargs)
    if isinstance(out, tuple) and len(out) == 3:
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1], {}
    raise RuntimeError("Unexpected output format from Temperature_and_Energy_needs_calculation")


def _export_hvac_flow_results(hourly_sim: pd.DataFrame, hvac_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Export a compact stage-by-stage HVAC results table.

    The output combines:
    - building energy needs from ISO52016;
    - emission, distribution and generation outputs from ISO 15316-1 / EN 15316 blocks.
    """

    if not isinstance(hourly_sim, pd.DataFrame) or not isinstance(hvac_df, pd.DataFrame):
        raise TypeError("hourly_sim and hvac_df must be pandas DataFrames.")

    stage_df = pd.DataFrame(index=hourly_sim.index)

    building_cols = [
        "Q_HC",
        "Q_H",
        "Q_C",
        "T_op0",
        "T_air",
        "T_op",
        "T_ext",
    ]
    for col in building_cols:
        if col in hourly_sim.columns:
            stage_df[f"building_{col}"] = hourly_sim[col]

    if "Q_latent_W" in hourly_sim.columns:
        stage_df["building_Q_C_sensible"] = pd.to_numeric(hourly_sim.get("Q_C", 0.0), errors="coerce").fillna(0.0)
        stage_df["building_Q_C_latent"] = pd.to_numeric(hourly_sim["Q_latent_W"], errors="coerce").fillna(0.0)
        stage_df["building_Q_C_total"] = (
            stage_df["building_Q_C_sensible"].astype(float) + stage_df["building_Q_C_latent"].astype(float)
        )

    hvac_cols = [
        "Q_h(kWh)",
        "QH_em_i_in(kWh)",
        "QH_dis_i_req(kWh)",
        "QH_dis_i_in(kWh)",
        "QH_gen_out(kWh)",
        "EHW_gen_in(kWh)",
        "EHW_gen_aux(kWh)",
        "QW_gen_i_ls_rbl_H(kWh)",
        "Q_w_dis_i_ls(kWh)",
        "Q_w_dis_i_aux(kWh)",
        "Q_w_dis_i_ls_rbl_H(kWh)",
        "ΦH_em_eff(kW)",
        "θH_em_flow(°C)",
        "θH_em_ret(°C)",
        "θH_dis_flw(°C)",
        "θH_dis_ret(°C)",
        "θX_gen_cr_flw(°C)",
        "θX_gen_cr_ret(°C)",
        "V_H_em_eff(m3/h)",
        "V_H_dis(m3/h)",
        "V_H_gen(m3/h)",
        "efficiency_gen(%)",
        "emission_calculation_mode",
        "generation_type",
        "HP_EH_hp_in(kWh)",
        "HP_EH_backup_in(kWh)",
        "HP_QH_environment_in(kWh)",
        "HP_QH_unmet(kWh)",
        "HP_SPF_HW_gen(-)",
    ]
    for col in hvac_cols:
        if col in hvac_df.columns:
            stage_df[f"hvac_{col}"] = hvac_df[col]

    out_path = Path(output_dir) / "hvac_stage_results.csv"
    stage_df.to_csv(out_path)
    print(f"[info] HVAC stage results written to {out_path}")
    return stage_df


def _default_heat_pump_15316_4_2_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact default performance maps for an air-to-water heat pump."""

    heating_rows = []
    for source in [-15, -7, 2, 7, 12, 20]:
        for sink in [35, 45, 55]:
            capacity = 12.0 + 0.12 * source - 0.055 * (sink - 35.0)
            cop = 4.2 + 0.055 * source - 0.045 * (sink - 35.0)
            heating_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 3.0),
                    "cop": max(cop, 1.6),
                }
            )

    cooling_rows = []
    for source in [20, 25, 30, 35, 40]:
        for sink in [7, 12, 18]:
            capacity = 10.0 - 0.08 * (source - 25.0) + 0.035 * (sink - 7.0)
            eer = 3.7 - 0.055 * (source - 25.0) + 0.045 * (sink - 7.0)
            cooling_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 3.0),
                    "eer": max(eer, 1.7),
                }
            )

    return pd.DataFrame(heating_rows), pd.DataFrame(cooling_rows)


def _apply_heating_heat_pump_generation(
    hourly_sim: pd.DataFrame,
    hvac_df: pd.DataFrame,
    input_hvac: dict,
    output_dir: str,
) -> tuple[pd.DataFrame, pybui.HeatPumpSimulationResult]:
    """Replace the heating generator stage with EN 15316-4-2 heat-pump results."""

    if not isinstance(hourly_sim, pd.DataFrame) or not isinstance(hvac_df, pd.DataFrame):
        raise TypeError("hourly_sim and hvac_df must be pandas DataFrames.")
    if "QH_gen_out(kWh)" not in hvac_df.columns:
        raise KeyError("Heat-pump generation requires hvac_df['QH_gen_out(kWh)'].")

    hp_cfg = dict(input_hvac.get("heating_heat_pump_15316_4_2_config", {}))
    heating_map = hp_cfg.get("heating_performance_map")
    cooling_map = hp_cfg.get("cooling_performance_map")
    if heating_map is None or cooling_map is None:
        default_heating_map, default_cooling_map = _default_heat_pump_15316_4_2_maps()
        heating_map = default_heating_map if heating_map is None else heating_map
        cooling_map = default_cooling_map if cooling_map is None else cooling_map

    hp_cfg.update(
        {
            "heating_performance_map": heating_map,
            "cooling_performance_map": cooling_map,
            "dhw_performance_map": hp_cfg.get("dhw_performance_map", heating_map),
            "demand_unit": "kWh",
            "heating_enabled": True,
            "cooling_enabled": False,
            "dhw_enabled": False,
        }
    )

    loads = pd.DataFrame(index=hvac_df.index)
    loads["T_ext"] = (
        pd.to_numeric(hvac_df["T_ext(°C)"], errors="coerce")
        if "T_ext(°C)" in hvac_df.columns
        else pd.to_numeric(hourly_sim["T_ext"], errors="coerce").reset_index(drop=True)
    )
    loads["Q_H_kWh"] = pd.to_numeric(hvac_df["QH_gen_out(kWh)"], errors="coerce").fillna(0.0).clip(lower=0.0)
    loads["Q_C_kWh"] = 0.0
    loads["Q_W_kWh"] = 0.0
    if "θX_gen_cr_flw(°C)" in hvac_df.columns:
        loads["T_H_sink_C"] = pd.to_numeric(hvac_df["θX_gen_cr_flw(°C)"], errors="coerce")
    elif "θH_dis_flw(°C)" in hvac_df.columns:
        loads["T_H_sink_C"] = pd.to_numeric(hvac_df["θH_dis_flw(°C)"], errors="coerce")
    if "θX_gen_cr_ret(°C)" in hvac_df.columns:
        loads["T_H_return_C"] = pd.to_numeric(hvac_df["θX_gen_cr_ret(°C)"], errors="coerce")
    elif "θH_dis_ret(°C)" in hvac_df.columns:
        loads["T_H_return_C"] = pd.to_numeric(hvac_df["θH_dis_ret(°C)"], errors="coerce")

    hp_calc = pybui.HeatPumpSystemCalculator(hp_cfg)
    hp_result = hp_calc.run_timeseries(loads)
    hp_summary = dict(hp_result.summary)

    q_h = float(hp_summary.get("QH_gen_out_kWh", loads["Q_H_kWh"].sum()))
    e_hp = float(hp_summary.get("EH_hp_in_kWh", 0.0))
    e_backup = float(hp_summary.get("EHW_backup_in_kWh", 0.0))
    e_gen = float(hp_summary.get("EHW_gen_in_kWh", e_hp + e_backup))
    w_aux = float(hp_summary.get("WHW_gen_aux_kWh", 0.0))
    spf = float(hp_summary.get("SPF_HW_gen", q_h / (e_gen + w_aux) if (e_gen + w_aux) > 0 else 0.0))

    out = hvac_df.copy()
    if q_h > 0:
        weights = pd.to_numeric(out["QH_gen_out(kWh)"], errors="coerce").fillna(0.0).clip(lower=0.0) / q_h
    else:
        weights = pd.Series(0.0, index=out.index)
    out["generation_type"] = "heat_pump_15316_4_2"
    out["EHW_gen_in(kWh)"] = weights * e_gen
    out["EH_gen_in(kWh)"] = weights * e_gen
    out["EWH_gen_in(kWh)"] = 0.0
    out["EHW_gen_aux(kWh)"] = weights * w_aux
    out["QW_gen_i_ls_rbl_H(kWh)"] = weights * float(hp_summary.get("QHW_gen_ls_rbl_tot_kWh", 0.0))
    out["efficiency_gen(%)"] = spf * 100.0
    out["HP_EH_hp_in(kWh)"] = weights * e_hp
    out["HP_EH_backup_in(kWh)"] = weights * e_backup
    out["HP_QH_environment_in(kWh)"] = weights * float(hp_summary.get("QHW_environment_in_kWh", 0.0))
    out["HP_QH_unmet(kWh)"] = weights * float(hp_summary.get("QHW_unmet_kWh", 0.0))
    out["HP_SPF_HW_gen(-)"] = spf

    out_dir = Path(output_dir)
    hp_result.bins.to_csv(out_dir / "heating_heat_pump_15316_4_2_bin_results.csv", index=False)
    pd.DataFrame([hp_summary]).to_csv(out_dir / "heating_heat_pump_15316_4_2_summary.csv", index=False)
    loads.to_csv(out_dir / "heating_heat_pump_15316_4_2_loads.csv")
    print(
        "[info] Heating generator: EN 15316-4-2 heat pump "
        f"QH={q_h:.1f} kWh, final electricity={e_gen + w_aux:.1f} kWh, SPF={spf:.2f}"
    )
    return out, hp_result


# BUI = {
#     "building": {
#         "name": "test-cy",
#         "azimuth_relative_to_true_north": 41.8,
#         "latitude": 37.98880066730187,
#         "longitude": 23.733531819066098,
#         "exposed_perimeter": 40,
#         "height": 3,
#         "wall_thickness": 0.3,
#         "n_floors": 1,
#         "building_type_class": "Residential_apartment",
#         "adj_zones_present": False,
#         "number_adj_zone":2,
#         "net_floor_area": 100,
#         "construction_class": "class_i",
#     },
#     "adjacent_zones": [
#         {
#             "name":"adj_1",
#             "orientation_zone": {
#                 "azimuth": 0,
#             },
#             "area_facade_elements": np.array([20,60,30,30,50,50], dtype=object),
#             "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
#             "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
#             "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
#             'volume': 300, 
#             'building_type_class':'Residential_apartment',
#             'a_use':50 
#         },
#         {
#             "name":"adj_2",
#             "orientation_zone": {
#                 "azimuth": 180,
#             },
#             "area_facade_elements": np.array([20,60,30,30,50,50], dtype=object),
#             "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
#             "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
#             "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
#             'volume': 300, 
#             'building_type_class':'Residential_apartment',
#             'a_use':50 
#         }
#     ],
#     "building_surface": [
#         {
#             "name": "Roof surface",
#             "type": "opaque",
#             "area": 130,
#             "sky_view_factor": 1.0,
#             "u_value": 2.2,
#             "solar_absorptance": 0.4,
#             "thermal_capacity": 741500.0,
#             "orientation": {
#                 "azimuth": 0,
#                 "tilt": 0
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Opaque north surface",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.4,
#             "solar_absorptance": 0.4,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 0,
#                 "tilt": 90
#             },
#             "name_adj_zone": "adj_1"
#         },
#         {
#             "name": "Opaque south surface",
#             "type": "opaque",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.4,
#             "solar_absorptance": 0.4,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 180,
#                 "tilt": 90
#             },
#             "name_adj_zone": "adj_2"
#         },
#         {
#             "name": "Opaque east surface",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.2,
#             "solar_absorptance": 0.6,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 90,
#                 "tilt": 90
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Opaque west surface",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.2,
#             "solar_absorptance": 0.7,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 270,
#                 "tilt": 90
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Slab to ground",
#             "type": "opaque",
#             "area": 100,
#             "sky_view_factor": 0.0,
#             "u_value": 1.6,
#             "solar_absorptance": 0.6,
#             "thermal_capacity": 405801,
#             "orientation": {
#                 "azimuth": 0,
#                 "tilt": 0
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Transparent east surface",
#             "type": "transparent",
#             "area": 4,
#             "sky_view_factor": 0.5,
#             "u_value": 5,
#             "g_value": 0.726,
#             "height": 2,
#             "width": 1,
#             "parapet": 1.1,
#             "orientation": {
#                 "azimuth": 90,
#                 "tilt": 90
#             },
#             "shading": False,
#             "shading_type": "horizontal_overhang",
#             "width_or_distance_of_shading_elements": 0.5,
#             "overhang_proprieties": {
#                 "width_of_horizontal_overhangs":1
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Transparent west surface",
#             "type": "transparent",
#             "area": 4,
#             "sky_view_factor": 0.5,
#             "u_value": 5,
#             "g_value": 0.726,
#             "height": 2,
#             "width": 1,
#             "parapet": 1.1,
#             "orientation": {
#                 "azimuth": 270,
#                 "tilt": 90
#             },
#             "shading": False,
#             "shading_type": "horizontal_overhang",
#             "width_or_distance_of_shading_elements": 0.5,
#             "overhang_proprieties": {
#                 "width_of_horizontal_overhangs":1
#             },
#             "name_adj_zone": None
#         }
#     ],
#     "units": {
#         "area": "m²",
#         "u_value": "W/m²K",
#         "thermal_capacity": "J/kgK",
#         "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
#         "tilt": "degrees (0=horizontal, 90=vertical)",
#         "internal_gain": "W/m²",
#         "internal_gain_profile": "Normalized to 0-1",
#         "HVAC_profile": "0: off, 1: on"
#     },
#     "building_parameters": {
#         "temperature_setpoints": {
#             "heating_setpoint": 20.0,
#             "heating_setback": 17.0,
#             "cooling_setpoint": 26.0,
#             "cooling_setback": 30.0,
#             "units": "°C"
#         },
#         "system_capacities": {
#             "heating_capacity": 10000000.0,
#             "cooling_capacity": 12000000.0,
#             "units": "W"
#         },
#         "airflow_rates": {
#             "infiltration_rate": 1.0,
#             "ventilation_rate_extra": 1.0,
#             "units": "ACH (air changes per hour)"
#         },
#         "internal_gains": [
#             {
#                 "name": "occupants",
#                 "full_load": 4.2,
#                 "weekday": [1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1.0,1.0],
#                 "weekend": [1.0,1.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1.0,1.0]
#             },
#             {
#                 "name": "appliances",
#                 "full_load": 3,
#                 "weekday": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
#                 "weekend": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
#             },
#             {
#                 "name": "lighting",
#                 "full_load": 3,
#                 "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
#                 "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
#             }
#         ],
#         "construction": {
#             "wall_thickness": 0.3,
#             "thermal_bridges": 2,
#             "units": "m (for thickness), W/mK (for thermal bridges)"
#         },
#         "climate_parameters": {
#             "coldest_month": 1,
#             "units": "1-12 (January-December)"
#         },
#         "heating_profile": {
#             "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#             "weekend": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#         },
#         "cooling_profile": {
#             "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#             "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
#         },
#         "ventilation_profile": {
#             "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#             "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
#         }
#     }
# }
BUI = {
    "building": {
        "name": "Archetype_ITA_SFH_2010",
        "azimuth_relative_to_true_north": 0,
        "latitude": 41.9,
        "longitude": 12.5,
        "exposed_perimeter": 40,
        "height": 6,
        "wall_thickness": 0.35,
        "n_floors": 2,
        "building_type_class": "Residential_apartment",
        "adj_zones_present": False,
        "number_adj_zone": 0,
        "net_floor_area": 120,
        "construction_class": "class_i",
        "construction_year": "2010-today",
        "country": "Italy"
    },
    "adjacent_zones": [
        {
            "name": "adj_1",
            "orientation_zone": {
                "azimuth": 0.0
            },
            "area_facade_elements": [
                20,
                60,
                30,
                30,
                50,
                50
            ],
            "typology_elements": [
                "OP",
                "OP",
                "OP",
                "OP",
                "GR",
                "OP"
            ],
            "transmittance_U_elements": [
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.5156683855612851,
                1.162633192818565
            ],
            "orientation_elements": [
                "NV",
                "SV",
                "EV",
                "WV",
                "HOR",
                "HOR"
            ],
            "volume": 300.0,
            "building_type_class": "Residential_apartment",
            "a_use": 50.0
        },
        {
            "name": "adj_2",
            "orientation_zone": {
                "azimuth": 180.0
            },
            "area_facade_elements": [
                20,
                60,
                30,
                30,
                50,
                50
            ],
            "typology_elements": [
                "OP",
                "OP",
                "OP",
                "OP",
                "GR",
                "OP"
            ],
            "transmittance_U_elements": [
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.5156683855612851,
                1.162633192818565
            ],
            "orientation_elements": [
                "NV",
                "SV",
                "EV",
                "WV",
                "HOR",
                "HOR"
            ],
            "volume": 300.0,
            "building_type_class": "Residential_apartment",
            "a_use": 50.0
        }
    ],
    "building_surface": [
        {
            "name": "Roof surface",
            "type": "opaque",
            "area": 130.0,
            "sky_view_factor": 1.0,
            "u_value": 2.2,
            "solar_absorptance": 0.4,
            "thermal_capacity": 741500.0,
            "orientation": {
                "azimuth": 0.0,
                "tilt": 0.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 13.0
        },
        {
            "name": "Opaque north surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 0.0,
                "tilt": 90.0
            },
            "name_adj_zone": "adj_1",
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Opaque south surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 180.0,
                "tilt": 90.0
            },
            "name_adj_zone": "adj_2",
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Opaque east surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.2,
            "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 90.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Opaque west surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.2,
            "solar_absorptance": 0.7,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 270.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Slab to ground",
            "type": "opaque",
            "area": 100.0,
            "sky_view_factor": 0.5,
            "u_value": 1.6,
            "solar_absorptance": 0.6,
            "thermal_capacity": 405801.0,
            "orientation": {
                "azimuth": 0.0,
                "tilt": 0.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 10.0
        },
        {
            "name": "Transparent east surface",
            "type": "transparent",
            "area": 3.0,
            "sky_view_factor": 0.5,
            "u_value": 5.0,
            "solar_absorptance": 0.5,
            "thermal_capacity": 0.0,
            "orientation": {
                "azimuth": 90.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 2.0,
            # "length": 1.0,
            "g_value": 0.726,
            "width": 1.0,
            "parapet": 1.1,
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs": 1.0
            }
        },
        {
            "name": "Transparent west surface",
            "type": "transparent",
            "area": 5.0,
            "sky_view_factor": 0.5,
            "u_value": 5.0,
            "solar_absorptance": 0.5,
            "thermal_capacity": 0.0,
            "orientation": {
                "azimuth": 270.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 2.0,
            # "length": 1.0,
            "g_value": 0.726,
            "width": 1.0,
            "parapet": 1.1,
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs": 1.0
            }
        }
    ],
    "units": {
        "area": "m\u00b2",
        "u_value": "W/m\u00b2K",
        "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
        "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m\u00b2",
        "internal_gain_profile": "Normalized to 0-1",
        "HVAC_profile": "0: off, 1: on"
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0,
            "heating_setback": 17.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "units": "\u00b0C"
        },
        "system_capacities": {
            "heating_capacity": 10000000.0,
            "cooling_capacity": 12000000.0,
            "units": "W"
        },
        "ventilation": {
            "ventilation_type": "occupancy",
            "flow_rate_per_person": 0.3,
            "units": "l/(s m2)",
            # Keep a numeric fallback because ventilation.py always casts this field to float().
            "custom_heat_transfer_coefficient_ventilation": 0.0,
            "info": "ventilation type could be: 1) Occupancy 2) occupancy 3)custom. If custum the value of custom_heat_transfer_coefficient_ventilation is used"
        },
        "internal_gains": [
            {
                "name": "occupants",
                "full_load": 4.2,
                "weekday": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.2,
                    0.5,
                    0.5,
                    0.5,
                    0.8,
                    0.8,
                    0.8,
                    1.0,
                    1.0
                ],
                "weekend": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.2,
                    0.5,
                    0.5,
                    0.5,
                    0.8,
                    0.8,
                    0.8,
                    1.0,
                    1.0
                ]
            },
            {
                "name": "appliances",
                "full_load": 3.0,
                "weekday": [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.5,
                    0.5,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.8,
                    0.8,
                    0.8,
                    0.6,
                    0.6
                ],
                "weekend": [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.5,
                    0.5,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.8,
                    0.8,
                    0.8,
                    0.6,
                    0.6
                ]
            },
            {
                "name": "lighting",
                "full_load": 3.0,
                "weekday": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.15,
                    0.15
                ],
                "weekend": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.15,
                    0.15
                ]
            }
        ],
        "construction": {
            "wall_thickness": 0.35,
            "thermal_bridges": 2.0,
            "units": "m (for thickness), W/mK (for thermal bridges)"
        },
        "climate_parameters": {
            "coldest_month": 1,
            "units": "1-12 (January-December)"
        },
        "heating_profile": {
            "weekday": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ],
            "weekend": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ]
        },
        "cooling_profile": {
            "weekday": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ],
            "weekend": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0
            ]
        },
        "ventilation_profile": {
            "weekday": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ],
            "weekend": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ]
        }
    }
}


INPUT_SYSTEM_HVAC = get_example_hvac_input()

# ============================================================
#           SYSTEM GENERATION PROFILE SELECTION
# ============================================================
# Pick one of the predefined generation system layouts below and let the
# script activate only the relevant input blocks.
#
# 1. heating + DHW: condensing boiler
# 2. heating + DHW: condensing boiler, cooling: air-air heat pump
# 3. heating + cooling + DHW: reversible air-to-water heat pump
# 4. heating: air-to-water heat pump, DHW: air-to-water heat-pump water heater,
#    cooling: air-air heat pump
SYSTEM_GENERATION_PROFILE = 2


def _reshape_inputs_for_generation_profile(cfg: dict, profile: int) -> dict:
    cfg = dict(cfg)

    def _ensure_cooling_enabled(enabled: bool) -> None:
        cfg.setdefault("cooling_storage_enabled", bool(enabled))
        cfg.setdefault("cooling_enabled", bool(enabled))

    if profile == 1:
        cfg["heating_generator_type"] = "boiler_15316_4_1"
        cfg["generation_calculation_mode"] = "boiler_15316_4_1"
        cfg["same_generator_for_heating_and_dhw"] = True
        cfg["dhw_storage_enabled"] = True
        cfg["boiler_generation_config"]["boiler_type"] = "condensing"
        cfg["boiler_generation_config"]["fuel_type"] = "natural_gas"
        cfg["dhw_generator_config"]["boiler_type"] = "condensing"
        cfg["dhw_generator_config"]["fuel_type"] = "natural_gas"
        INPUT_SYSTEM_COOLING["enabled"] = False
        INPUT_SYSTEM_COOLING["cooling_storage_enabled"] = False
    elif profile == 2:
        cfg["heating_generator_type"] = "boiler_15316_4_1"
        cfg["generation_calculation_mode"] = "boiler_15316_4_1"
        cfg["same_generator_for_heating_and_dhw"] = True
        cfg["dhw_storage_enabled"] = True
        cfg["boiler_generation_config"]["boiler_type"] = "condensing"
        cfg["boiler_generation_config"]["fuel_type"] = "natural_gas"
        cfg["dhw_generator_config"]["boiler_type"] = "condensing"
        cfg["dhw_generator_config"]["fuel_type"] = "natural_gas"
        INPUT_SYSTEM_COOLING["enabled"] = True
        INPUT_SYSTEM_COOLING["cooling_storage_enabled"] = False
        INPUT_SYSTEM_COOLING["cooling_system_16798_9_config"]["system_type"] = "air"
        INPUT_SYSTEM_COOLING["cooling_system_16798_9_config"]["generator_temperature_control"] = "VARIABLE"
        INPUT_SYSTEM_COOLING["cooling_generation_16798_13_config"]["generation_type"] = "COMP"
    elif profile == 3:
        cfg["heating_generator_type"] = "heat_pump_15316_4_2"
        cfg["generation_calculation_mode"] = "heat_pump_15316_4_2"
        cfg["same_generator_for_heating_and_dhw"] = True
        cfg["dhw_storage_enabled"] = True
        INPUT_SYSTEM_COOLING["enabled"] = False
        INPUT_SYSTEM_COOLING["cooling_storage_enabled"] = False
    elif profile == 4:
        cfg["heating_generator_type"] = "heat_pump_15316_4_2"
        cfg["generation_calculation_mode"] = "heat_pump_15316_4_2"
        cfg["same_generator_for_heating_and_dhw"] = True
        cfg["dhw_storage_enabled"] = True
        cfg["dhw_generator_config"]["nominal_power_kW"] = cfg["dhw_generator_config"].get("nominal_power_kW", 8.0)
        INPUT_SYSTEM_COOLING["enabled"] = True
        INPUT_SYSTEM_COOLING["cooling_storage_enabled"] = False
        INPUT_SYSTEM_COOLING["cooling_system_16798_9_config"]["system_type"] = "air"
        INPUT_SYSTEM_COOLING["cooling_generation_16798_13_config"]["generation_type"] = "COMP"
    else:
        raise ValueError("SYSTEM_GENERATION_PROFILE must be 1, 2, 3 or 4.")

    return cfg

# ============================================================
#           COOLING SYSTEM CONFIGURATION
#           EN 16798-9 (operating conditions) / EN 16798-15
#           (chilled-water storage) / EN 16798-13 (compression
#           generation)
# ============================================================
INPUT_SYSTEM_COOLING = get_example_cooling_input()

INPUT_SYSTEM_HVAC = _reshape_inputs_for_generation_profile(
    INPUT_SYSTEM_HVAC,
    SYSTEM_GENERATION_PROFILE,
)

# ============================================================
#           HVAC INPUT CONSISTENCY CHECK
# ============================================================
_coherence_summary = validate_hvac_input_coherence(INPUT_SYSTEM_HVAC)
_same_generator_for_heating_and_dhw = bool(_coherence_summary["same_generator_for_heating_and_dhw"])
_heating_generator_type = _coherence_summary["heating_generator_type"]
_dhw_storage_enabled = bool(INPUT_SYSTEM_HVAC.get("dhw_storage_enabled", False))
_dhw_nominal = float(_coherence_summary["dhw_nominal"])
_distribution_flow = float(_coherence_summary["distribution_flow_m3_h"])
_expected_flow = float(_coherence_summary["expected_flow_m3_h"])

# ============================================================
#           QUALITY CHECK SYSTEM INPUT HVAC
# ============================================================

# res = pybui.check_heating_system_inputs(INPUT_SYSTEM_HVAC)
res = check_heating_system_inputs(INPUT_SYSTEM_HVAC)


print("Selected Emitter:", res["emitter_type"])
print("Messages:")
for m in res["messages"]:
    print("-", m)
INPUT_SYSTEM_HVAC = res["config"]

# ============================================================
#           CALCULATION HVAC
# ============================================================

calc = HeatingSystemCalculator(INPUT_SYSTEM_HVAC)



bui_fixed, report = sanitize_and_validate_BUI(BUI, fix=True)


# print issues
for r in report:
    lvl = r["level"]
    print(f"[{lvl}] {r['path']}: {r['msg']}" + (" (fix applied)" if r["fix_applied"] else ""))

# validate BUI
bui_checked, issues = sanitize_and_validate_BUI(BUI, fix=False)
bui_checked['building_surface']
# extract only errors (level "ERROR")
errors = [e for e in issues if e["level"] == "ERROR"]



# def process_building(building_archetype, output_dir="result_test"):
#     """Process a single building archetype and save results"""
#     try:

#         # Process the building
#         (
#             hourly_sim,
#             annual_results_df,
#             _,
#         ) = _run_iso52016(building_archetype)

#         # Generate unique filenames for each building
#         building_name = building_archetype["building"].get("name", "unknown")
#         hourly_file = os.path.join(output_dir, f"hourly_sim_{building_name}.csv")
#         annual_file = os.path.join(output_dir, f"annual_results_{building_name}.csv")

#         # Save results with unique filenames
#         hourly_sim.to_csv(hourly_file)
#         annual_results_df.to_csv(annual_file, index=False)

#         # Calculate metrics
#         heating_kWh = hourly_sim[hourly_sim["Q_HC"] > 0]["Q_HC"].sum() / 1000
#         cooling_kWh = -hourly_sim[hourly_sim["Q_HC"] < 0]["Q_HC"].sum() / 1000
#         treated_floor_area = building_archetype["building"]["treated_floor_area"]
#         heating_kWh_per_sqm = heating_kWh / treated_floor_area
#         cooling_kWh_per_sqm = cooling_kWh / treated_floor_area

#         return {
#             "building_name": building_name,
#             "heating_kWh": heating_kWh,
#             "cooling_kWh": cooling_kWh,
#             "heating_kWh_per_sqm": heating_kWh_per_sqm,
#             "cooling_kWh_per_sqm": cooling_kWh_per_sqm,
#             "status": "success",
#         }

#     except Exception as e:
#         return {
#             "building_name": building_archetype["building"].get("name", "unknown"),
#             "error": str(e),
#             "status": "failed",
#         }


if errors:
    print("❌ Errors in BUI input data — simulation interrupted:\n")
    for e in errors:
        print(f" - {e['path']}: {e['msg']}")
    raise ValueError("Invalid BUI input: correct the data and retry.")
else:
    print("✅ BUI valid — starting ISO52016 simulation...\n")
    if WEATHER_SOURCE == "epw":
        print(f"[info] Weather source: epw ({WEATHER_FILE})")
    else:
        print("[info] Weather source: pvgis (no local EPW found)")
    file_dir = "/Users/dantonucci/Documents/GitHub/pybuildingenergy/result_test"
    # hourly_sim,annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(bui_checked, weather_source="epw", path_weather_file=str(WEATHER_FILE))
    hourly_sim, annual_results_df, sankey_data = _run_iso52016(bui_checked)

    # ISO 15316-1 calculation
    df_in = calc.load_csv_data(hourly_sim)  # columns: Q_H, T_op, T_ext (or aliases)
    df_out = calc.run_timeseries()
    heating_heat_pump_result = None
    if _heating_generator_type == 'heat_pump_15316_4_2':
        df_out, heating_heat_pump_result = _apply_heating_heat_pump_generation(
            hourly_sim,
            df_out,
            INPUT_SYSTEM_HVAC,
            file_dir,
        )
    else:
        df_out = df_out.copy()
        df_out["generation_type"] = "boiler_15316_4_1"
    _export_hvac_flow_results(hourly_sim, df_out, file_dir)
    _build_sankey_consumption_report(
        hourly_sim,
        df_out,
        file_dir,
        same_generator_for_heating_and_dhw=bool(INPUT_SYSTEM_HVAC.get("same_generator_for_heating_and_dhw", True)),
    )
    if GENERATE_EXTRA_REPORTS:
        _build_temperature_report(df_out, file_dir)
        Graphs_and_report(df=hourly_sim, season='heating_cooling', building_area=BUI['building']['net_floor_area']).bui_analysis_page(
            folder_directory=file_dir,
            name_file="main_report",
        )
    print("[info] Column legend: examples/hvac_stage_results_legend_it.md")

    # ============================================================
    #           COOLING CHAIN: EN 16798-9 / EN 16798-15 / EN 16798-13
    # ============================================================
    # Cooling-side operating conditions (EN 16798-9), optional chilled-water
    # storage (EN 16798-15) and compression cooling generation (EN 16798-13).
    # Follows the same chaining pattern as heat_pump_15316_4_2_example.py:
    # each stage reads/updates a shared "cooling_loads" frame via Q_C_kWh.
    cooling_loads = pd.DataFrame(index=hourly_sim.index)
    cooling_loads["T_ext"] = hourly_sim["T_ext"].astype(float)
    cooling_loads["T_op"] = hourly_sim["T_op"].astype(float)
    if "Q_C" in hourly_sim.columns:
        cooling_loads["Q_C_kWh"] = hourly_sim["Q_C"].astype(float) / 1000.0
    else:
        cooling_loads["Q_C_kWh"] = (-hourly_sim["Q_HC"].clip(upper=0)).astype(float) / 1000.0

    cooling_system_result = None
    cooling_storage_result = None
    cooling_generation_result = None

    if INPUT_SYSTEM_COOLING.get("enabled", True) and cooling_loads["Q_C_kWh"].sum() > 0:
        # EN 16798-9: cooling-side operating conditions (chilled-water temperatures,
        # distribution volume flow and generation-side cooling request).
        cooling_system_calc = pybui.CoolingSystemCalculator(
            INPUT_SYSTEM_COOLING["cooling_system_16798_9_config"]
        )
        cooling_system_result = cooling_system_calc.run_timeseries(cooling_loads)
        cooling_loads["Q_C_16798_9_out_kWh"] = cooling_system_result.timeseries["Q_C_dis_out_tot_req_kWh"]
        cooling_loads["Q_C_gen_in_req_16798_9_kWh"] = cooling_system_result.timeseries["Q_C_gen_in_req_kWh"]
        cooling_loads["theta_C_dis_supply_C"] = cooling_system_result.timeseries["theta_C_dis_supply_C"]
        cooling_loads["theta_C_dis_return_C"] = cooling_system_result.timeseries["theta_C_dis_return_C"]
        cooling_loads["theta_C_gen_out_req_C"] = cooling_system_result.timeseries["theta_C_gen_out_req_C"]
        cooling_loads["T_C_sink_C"] = cooling_system_result.timeseries["T_C_sink_C"]
        cooling_loads["Q_C_kWh"] = cooling_system_result.timeseries["Q_C_gen_in_req_kWh"]

        # EN 16798-15: optional chilled-water buffer storage heat gains and pump auxiliaries.
        if INPUT_SYSTEM_COOLING.get("cooling_storage_enabled", True):
            cooling_storage_calc = pybui.CoolingStorageSystemCalculator(
                INPUT_SYSTEM_COOLING["cooling_storage_16798_15_config"]
            )
            cooling_storage_result = cooling_storage_calc.run_timeseries(cooling_loads)
            cooling_loads["Q_C_storage_out_kWh"] = cooling_storage_result.timeseries["Q_C_sto_out_kWh"]
            cooling_loads["Q_C_sto_loss_kWh"] = cooling_storage_result.timeseries["Q_C_sto_ls_tot_kWh"]
            cooling_loads["W_C_sto_aux_kWh"] = cooling_storage_result.timeseries["W_C_sto_aux_kWh"]
            cooling_loads["Q_C_kWh"] = cooling_storage_result.timeseries["Q_C_sto_in_kWh"]
            cooling_loads["T_C_sink_C"] = cooling_storage_result.timeseries["T_C_sink_C"]
            cooling_loads["theta_C_gen_out_req_C"] = cooling_storage_result.timeseries["theta_C_sto_in_req_C"]

        # EN 16798-13: compression cooling generation (electricity input, EER, rejected heat).
        cooling_generation_calc = pybui.CoolingGenerationSystemCalculator(
            INPUT_SYSTEM_COOLING["cooling_generation_16798_13_config"]
        )
        cooling_generation_result = cooling_generation_calc.run_timeseries(cooling_loads)

        cooling_loads.to_csv(Path(file_dir) / "cooling_loads_16798.csv")
        cooling_system_result.timeseries.to_csv(Path(file_dir) / "cooling_16798_9_hourly_results.csv")
        pd.DataFrame([cooling_system_result.summary]).to_csv(
            Path(file_dir) / "cooling_16798_9_summary.csv", index=False
        )
        if cooling_storage_result is not None:
            cooling_storage_result.timeseries.to_csv(
                Path(file_dir) / "cooling_storage_16798_15_hourly_results.csv"
            )
            pd.DataFrame([cooling_storage_result.summary]).to_csv(
                Path(file_dir) / "cooling_storage_16798_15_summary.csv", index=False
            )
        cooling_generation_result.timeseries.to_csv(
            Path(file_dir) / "cooling_generation_16798_13_hourly_results.csv"
        )
        pd.DataFrame([cooling_generation_result.summary]).to_csv(
            Path(file_dir) / "cooling_generation_16798_13_summary.csv", index=False
        )

        print(f"EN 16798-9 cooling request: {cooling_system_result.summary['QC_dis_out_tot_req_kWh']:,.1f} kWh")
        print(f"EN 16798-9 mean cooling flow: {cooling_system_result.summary['q_V_C_dis_mean_m3_h']:.3f} m3/h")
        if cooling_storage_result is not None:
            print(f"EN 16798-15 cooling storage heat gains: {cooling_storage_result.summary['QC_sto_ls_tot_kWh']:,.1f} kWh")
            print(f"EN 16798-15 cooling storage auxiliaries: {cooling_storage_result.summary['WC_sto_aux_kWh']:,.1f} kWh")
        print(f"EN 16798-13 cooling electricity: {cooling_generation_result.summary['EC_total_kWh']:,.1f} kWh")
        print(f"EN 16798-13 cooling rejected heat: {cooling_generation_result.summary['QC_gen_out_kWh']:,.1f} kWh")
        print(f"EN 16798-13 SEER_C_gen: {cooling_generation_result.summary['SEER_C_gen']:.2f}")
        _build_cooling_sankey_report(
            hourly_sim,
            file_dir,
            cooling_system_result=cooling_system_result,
            cooling_storage_result=cooling_storage_result,
            cooling_generation_result=cooling_generation_result,
        )
    else:
        print("[info] Cooling chain skipped: no cooling need in hourly_sim or INPUT_SYSTEM_COOLING.enabled=False.")

    # ============================================================
    #           DHW CHAIN: optional storage + distribution
    # ============================================================
    year_for_dhw = int(pd.DatetimeIndex(hourly_sim.index).year.min()) if len(hourly_sim.index) else 2023
    italy_calendar = generate_calendar("Italy", year_for_dhw)
    n_workdays = int((italy_calendar["values"] == "Working").sum())
    n_weekends = int((italy_calendar["values"] == "Non-Working").sum())
    n_holidays = int((italy_calendar["values"] == "Holiday").sum())
    total_days = int(italy_calendar["values"].count())

    hourly_fractions_examples = pd.DataFrame({
        "Workday": [0,0,0,0,0,0,0,0,5,10,10,10,20,10,10,10,10,5,0,0,0,0,0,0],
        "Weekend": [0,0,0,0,0,0,0,0,5,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0],
        "Holiday": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    })
    sum_fractions = pd.DataFrame(hourly_fractions_examples.sum())
    sum_fractions.columns = ["fractions"]
    dhw_calc = Volume_and_energy_DHW_calculation(
        n_workdays,
        n_weekends,
        n_holidays,
        sum_fractions,
        total_days,
        hourly_fractions_examples,
        42,
        13.5,
        60,
        11.2,
        mode_calc='number_of_units',
        building_type_B3='Residential',
        building_area=120,
        unit_count=10,
        building_type_B5='Dwelling',
        residential_typology='residential_building - simple housing - AVG',
        calculation_method='table',
        year=year_for_dhw,
        country_calendar=italy_calendar
    )

    dhw_hourly = pd.DataFrame(
        {"Q_W_kWh": pd.Series(dhw_calc[7], index=hourly_sim.index, dtype=float)},
        index=hourly_sim.index,
    )
    dhw_distribution_calc = pybui.DistributionSystemCalculator(INPUT_SYSTEM_HVAC.get('distribution_15316_3_config', {}))
    dhw_distribution_result = dhw_distribution_calc.run_timeseries(dhw_hourly)
    dhw_distribution_result.timeseries.to_csv(f"{file_dir}/dhw_distribution_15316_3_hourly_results.csv")
    pd.DataFrame([dhw_distribution_result.summary]).to_csv(
        f"{file_dir}/dhw_distribution_15316_3_summary.csv",
        index=False,
    )

    dhw_storage_result = None
    if _dhw_storage_enabled:
        dhw_storage_input = pd.DataFrame(
            {"Q_W_kWh": dhw_distribution_result.timeseries["Q_W_dis_in_kWh"].astype(float)},
            index=hourly_sim.index,
        )
        dhw_storage_calc = pybui.StorageSystemCalculator(INPUT_SYSTEM_HVAC.get('dhw_storage_config', {}))
        dhw_storage_result = dhw_storage_calc.run_timeseries(dhw_storage_input)
        dhw_storage_result.timeseries.to_csv(f"{file_dir}/dhw_storage_15316_5_hourly_results.csv")
        pd.DataFrame([dhw_storage_result.summary]).to_csv(
            f"{file_dir}/dhw_storage_15316_5_summary.csv",
            index=False,
        )

    hourly_sim = hourly_sim.copy()
    if "Q_C" in hourly_sim.columns:
        hourly_sim["Q_C_sensible"] = pd.to_numeric(hourly_sim["Q_C"], errors="coerce").fillna(0.0)
        hourly_sim["Q_C_latent"] = (
            pd.to_numeric(hourly_sim["Q_latent_W"], errors="coerce").fillna(0.0)
            if "Q_latent_W" in hourly_sim.columns
            else 0.0
        )
        hourly_sim["Q_C_total"] = hourly_sim["Q_C_sensible"] + hourly_sim["Q_C_latent"]
    combined_parts = [hourly_sim.reset_index(drop=True)]
    if isinstance(df_out, pd.DataFrame):
        combined_parts.append(df_out.add_prefix("hvac_").reset_index(drop=True))
    combined_parts.append(dhw_hourly.add_prefix("dhw_").reset_index(drop=True))
    if hasattr(dhw_distribution_result, "timeseries"):
        combined_parts.append(dhw_distribution_result.timeseries.add_prefix("dhw_dis_").reset_index(drop=True))
    if dhw_storage_result is not None:
        combined_parts.append(dhw_storage_result.timeseries.add_prefix("dhw_sto_").reset_index(drop=True))
    combined_hourly = pd.concat(combined_parts, axis=1)
    combined_hourly.to_csv(f"{file_dir}/hvac_dhw_hourly_results.csv")

    heating_generator = float(df_out.get("QH_gen_out(kWh)", pd.Series(0.0, index=df_out.index)).sum())
    if dhw_storage_result is not None:
        dhw_generator = float(
            pd.to_numeric(
                dhw_storage_result.timeseries["Q_W_sto_in_kWh"],
                errors="coerce",
            ).fillna(0.0).sum()
        )
    else:
        dhw_generator = float(
            pd.to_numeric(
                dhw_distribution_result.timeseries["Q_W_dis_in_kWh"],
                errors="coerce",
            ).fillna(0.0).sum()
        )

    if _same_generator_for_heating_and_dhw:
        print(
            f"[info] Same generator for heating and DHW: heating={heating_generator:.2f} kWh, "
            f"DHW after distribution/storage={dhw_generator:.2f} kWh, combined={heating_generator + dhw_generator:.2f} kWh"
        )
    else:
        dhw_gen_cfg = INPUT_SYSTEM_HVAC.get('dhw_generator_config', {})
        dhw_gen_power = float(dhw_gen_cfg.get('rated_power_kW', dhw_gen_cfg.get('nominal_power_kW', 0.0)))
        if dhw_gen_power <= 0:
            raise ValueError(
                "HVAC input error: dhw_generator_config must define rated_power_kW or nominal_power_kW when DHW is separate."
            )
        print(
            "[info] Separate DHW generator selected. "
            f"Use dhw_generator_config for generator sizing; DHW demand after distribution/storage is {dhw_generator:.2f} kWh."
        )

    # ============================================================
    #           FINAL ENERGY SUMMARY
    # ============================================================
    heating_final_energy = float(
        pd.to_numeric(df_out.get("EHW_gen_in(kWh)", pd.Series(0.0, index=df_out.index)), errors="coerce")
        .fillna(0.0)
        .sum()
    )
    cooling_final_energy = (
        float(cooling_generation_result.summary["EC_total_kWh"])
        if cooling_generation_result is not None
        else 0.0
    )
    dhw_final_energy = dhw_generator
    total_final_energy = heating_final_energy + cooling_final_energy + dhw_final_energy

    summary_df = pd.DataFrame(
        [
            {
                "heating_final_energy_kWh": heating_final_energy,
                "cooling_final_energy_kWh": cooling_final_energy,
                "dhw_final_energy_kWh": dhw_final_energy,
                "total_final_energy_kWh": total_final_energy,
            }
        ]
    )
    summary_df.to_csv(Path(file_dir) / "final_energy_summary.csv", index=False)
    print(
        "[summary] Final energy: "
        f"heating={heating_final_energy:,.1f} kWh, "
        f"cooling={cooling_final_energy:,.1f} kWh, "
        f"DHW={dhw_final_energy:,.1f} kWh, "
        f"total={total_final_energy:,.1f} kWh"
    )

    _build_sankey_consumption_report(
        hourly_sim,
        df_out,
        file_dir,
        same_generator_for_heating_and_dhw=_same_generator_for_heating_and_dhw,
        dhw_useful_df=dhw_hourly,
        dhw_distribution_df=dhw_distribution_result.timeseries,
        dhw_storage_df=(dhw_storage_result.timeseries if dhw_storage_result is not None else None),
    )

    raise SystemExit(0)