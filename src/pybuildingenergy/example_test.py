import numpy as np
from source.chek_input import sanitize_and_validate_BUI


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
        "number_adj_zone":0,
        "net_floor_area": 100,
    },
    "adjacent_zones": [
        {
            "name":"adj_1",
            "orientation_zone": {
                "azimuth": 0,
            },
            "area_facade_elements": np.array([60,60,30,30,50,50], dtype=object),
            "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
            "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
            "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
            'volume': 300, 
            'building_type_class':'Residential_apartment',
            'a_use':50 
        }
    ],
    "building_surface": [
        {
            "name": "Roof surface",
            "type": "opaque",
            "area": 130,
            "sky_view_factor": 1.0,
            "u_value": 2.2,
            "solar_absorptance": 0.4,
            "thermal_capacity": 741500.0,
            "orientation": {
                "azimuth": 0,
                "tilt": 0
            }
        },
        {
            "name": "Opaque north surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 0,
                "tilt": 90
            }
        },
        {
            "name": "Opaque north surface",
            "type": "opaque",
            "area": 6,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 0,
                "tilt": 90
            }
        },
        {
            "name": "Opaque south surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 180,
                "tilt": 90
            }
        },
        {
            "name": "Opaque east surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 1.2,
            "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 90,
                "tilt": 90
            }
        },
        {
            "name": "Opaque west surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 1.2,
            "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 270,
                "tilt": 90
            }
        },
        {
            "name": "Slab to ground",
            "type": "opaque",
            "area": 100,
            "sky_view_factor": 0.0,
            "u_value": 1.6,
            "solar_absorptance": 0.6,
            "thermal_capacity": 405801,
            "orientation": {
                "azimuth": 0,
                "tilt": 0
            }
        },
        {
            "name": "Transparent north surface",
            "type": "transparent",
            "area": 6,
            "sky_view_factor": 0.5,
            "u_value": 5,
            "g_value": 0.726,
            "height": 2,
            "width": 1,
            "parapet": 1.1,
            "orientation": {
                "azimuth": 0,
                "tilt": 90
            },
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs":1
            }
        },
        {
            "name": "Transparent east surface",
            "type": "transparent",
            "area": 4,
            "sky_view_factor": 0.5,
            "u_value": 5,
            "g_value": 0.726,
            "height": 2,
            "width": 1,
            "parapet": 1.1,
            "orientation": {
                "azimuth": 90,
                "tilt": 90
            },
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs":1
            }
        },
        {
            "name": "Transparent west surface",
            "type": "transparent",
            "area": 4,
            "sky_view_factor": 0.5,
            "u_value": 5,
            "g_value": 0.726,
            "height": 2,
            "width": 1,
            "parapet": 1.1,
            "orientation": {
                "azimuth": 270,
                "tilt": 90
            },
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs":1
            }
        }
    ],
    "units": {
        "area": "m²",
        "u_value": "W/m²K",
        "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
        "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m²",
        "internal_gain_profile": "Normalized to 0-1",
        "HVAC_profile": "0: off, 1: on"
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0,
            "heating_setback": 17.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "units": "°C"
        },
        "system_capacities": {
            "heating_capacity": 10000000.0,
            "cooling_capacity": 12000000.0,
            "units": "W"
        },
        "airflow_rates": {
            "infiltration_rate": 1.0,
            "ventilation_rate_extra": 1.0,
            "units": "ACH (air changes per hour)"
        },
        "internal_gains": [
            {
                "name": "occupants",
                "full_load": 4.2,
                "weekday": [1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1.0,1.0],
                "weekend": [1.0,1.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1.0,1.0]
            },
            {
                "name": "appliances",
                "full_load": 3,
                "weekday": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
                "weekend": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
            },
            {
                "name": "lighting",
                "full_load": 3,
                "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
                "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
            }
        ],
        "construction": {
            "wall_thickness": 0.3,
            "thermal_bridges": 2,
            "units": "m (for thickness), W/mK (for thermal bridges)"
        },
        "climate_parameters": {
            "coldest_month": 1,
            "units": "1-12 (January-December)"
        },
        "heating_profile": {
            "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            "weekend": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
        },
        "cooling_profile": {
            "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
        },
        "ventilation_profile": {
            "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
        }
    }
}




from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
from tqdm import tqdm
import os
import json
from functools import partial
import numpy as np
from source.utils import ISO52016
from source.graphs import Graphs_and_report
from data.building_archetype import Selected_bui_archetype
from global_inputs import main_directory_
import os
from source.iso_15316_1 import HeatingSystemCalculator

calc = HeatingSystemCalculator({
    'emitter_type': 'Floor heating',
    'nominal_power': 8,
    'emission_efficiency': 90,
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
    'selected_emm_cont_circuit': 0,
    'mixing_valve': True,

    # --- generator ---
    'full_load_power': 27,                  # kW
    'max_monthly_load_factor': 100,         # %
    'tH_gen_i_ON': 1,                       # h
    'auxiliary_power_generator': 0,         # %
    'fraction_of_auxiliary_power_generator': 40,   # %
    'generator_circuit': 'independent',     # 'direct' | 'independent'

    # Primary: independent climatic curve
    'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',
    'gen_outdoor_temp_data': pd.DataFrame({
        "θext_min_gen": [-7],
        "θext_max_gen": [15],
        "θflw_gen_max": [60],
        "θflw_gen_min": [35],
    }, index=["Generator curve"]),

    'speed_control_generator_pump': 'variable',
    'generator_nominal_deltaT': 20,         # °C

    # Efficiency model
    'efficiency_model': 'simple',

    # Calculation options
    'calc_when_QH_positive_only': False,
    'off_compute_mode': 'full',

    # Optional explicit generator setpoints (commented by default)
    # 'θHW_gen_flw_set': 50,
    # 'θHW_gen_ret_set': 40,
})

def process_building(building_archetype, output_dir="results"):
    """Process a single building archetype and save results"""
    try:

        # Process the building
        (
            hourly_sim,
            annual_results_df,
        ) = ISO52016.Temperature_and_Energy_needs_calculation(
            building_archetype,
            weather_source="pvgis",
        )

        # Generate unique filenames for each building
        building_name = building_archetype["building"].get("name", "unknown")
        hourly_file = os.path.join(output_dir, f"hourly_sim_{building_name}.csv")
        annual_file = os.path.join(output_dir, f"annual_results_{building_name}.csv")

        # Save results with unique filenames
        hourly_sim.to_csv(hourly_file)
        annual_results_df.to_csv(annual_file, index=False)

        # Calculate metrics
        heating_kWh = hourly_sim[hourly_sim["Q_HC"] > 0]["Q_HC"].sum() / 1000
        cooling_kWh = -hourly_sim[hourly_sim["Q_HC"] < 0]["Q_HC"].sum() / 1000
        treated_floor_area = building_archetype["building"]["treated_floor_area"]
        heating_kWh_per_sqm = heating_kWh / treated_floor_area
        cooling_kWh_per_sqm = cooling_kWh / treated_floor_area

        return {
            "building_name": building_name,
            "heating_kWh": heating_kWh,
            "cooling_kWh": cooling_kWh,
            "heating_kWh_per_sqm": heating_kWh_per_sqm,
            "cooling_kWh_per_sqm": cooling_kWh_per_sqm,
            "status": "success",
        }

    except Exception as e:
        return {
            "building_name": building_archetype["building"].get("name", "unknown"),
            "error": str(e),
            "status": "failed",
        }


bui_fixed, report = sanitize_and_validate_BUI(BUI, fix=True)

# stampa problemi
for r in report:
    lvl = r["level"]
    print(f"[{lvl}] {r['path']}: {r['msg']}" + (" (fix applied)" if r["fix_applied"] else ""))

# valida BUI
bui_checked, issues = sanitize_and_validate_BUI(BUI, fix=False)

# estrai solo gli errori (livello "ERROR")
errors = [e for e in issues if e["level"] == "ERROR"]

if errors:
    print("❌ Errore nei dati di input BUI — simulazione interrotta:\n")
    for e in errors:
        print(f" - {e['path']}: {e['msg']}")
    raise ValueError("Input BUI non valido: correggere i dati e riprovare.")
else:
    print("✅ BUI valido — avvio simulazione ISO52016...\n")
    file_dir = os.path.dirname(os.path.realpath(__file__))
    hourly_sim,annual_results_df = ISO52016.Temperature_and_Energy_needs_calculation(BUI,weather_source="pvgis")
    path_hourly_sim_result = file_dir + "/Result/hourly_sim__arch.csv"
    path_annual_sim_result = file_dir + "/Result/annual_sim__arch.csv"
    hourly_sim.to_csv(path_hourly_sim_result)
    annual_results_df.to_csv(path_annual_sim_result)
        
    # ISO 15316-1 calculation
    df_in = calc.load_csv_data(hourly_sim)  # colonne: Q_H, T_op, T_ext (o alias)
    df_out = calc.run_timeseries()
    df_out.to_csv(file_dir + "/Result/hourly_heating_system.csv")

    # Generate Graphs
    dir_chart_folder = file_dir+ "/Result"
    name_report = "main_report"
    Graphs_and_report(df = hourly_sim,season ='heating_cooling',building_area=BUI['building']['net_floor_area'] ).bui_analysis_page(
    folder_directory=dir_chart_folder,
    name_file=name_report)

    # Optional:
    # process_building(BUI)

