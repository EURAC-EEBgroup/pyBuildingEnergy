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

def _run_iso52016(building_obj):
    kwargs = {"weather_source": WEATHER_SOURCE}
    if WEATHER_SOURCE == "epw":
        kwargs["path_weather_file"] = str(WEATHER_FILE)

    out = ISO52016.Temperature_and_Energy_needs_calculation(building_obj, **kwargs)
    if isinstance(out, tuple) and len(out) == 3:
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1], {}
    raise RuntimeError("Unexpected output format from Temperature_and_Energy_needs_calculation")

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
#             "overhang_properties": {
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
#             "overhang_properties": {
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
            "overhang_properties": {
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
            "overhang_properties": {
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
            "custom_heat_transfer_coefficient_ventilation": None,
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


INPUT_SYSTEM_HVAC = {
    # ---- emitter ----
    'emitter_type': 'Floor heating',
    'nominal_power': 8,
    'emission_efficiency': 90,
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
    'selected_emm_cont_circuit': 0,
    'mixing_valve': True,
    # 'TB14': custom_TB14, #  <- Uncomment and upload your emittere table, oterwhise the default stored in gloabl_inputs.py is used
    # 'heat_emission_data' : pd.DataFrame({ <- Uncomment and upload your emittere table, oterwhise the default stored in gloabl_inputs.py is used
    #         "θH_em_flw_max_sahz_i": [45],
    #         "ΔθH_em_w_max_sahz_i": [8],
    #         "θH_em_ret_req_sahz_i": [20],
    #         "βH_em_req_sahz_i": [80],
    #         "θH_em_flw_min_tz_i": [28],
    #     }, index=[
    #         "Max flow temperature HZ1",
    #         "Max Δθ flow / return HZ1",
    #         "Desired return temperature HZ1",
    #         "Desired load factor with ON-OFF for HZ1",
    #         "Minimum flow temperature for HZ1"
    #     ]),
    'mixing_valve_delta':2,
    # 'constant_flow_temp':42,

    # --- distribution ---
    'heat_losses_recovered': True,
    'distribution_loss_recovery': 90,
    'simplified_approach': 80,
    'distribution_aux_recovery': 80,
    'distribution_aux_power': 30,
    'distribution_loss_coeff': 48,
    'distribution_operation_time': 1,
    
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
    'mixing_valve_delta':2,

    # Optional explicit generator setpoints (commented by default)
    # 'θHW_gen_flw_set': 50,
    # 'θHW_gen_ret_set': 40,

    # Efficiency model
    'efficiency_model': 'simple',

    # Calculation options
    'calc_when_QH_positive_only': False,
    'off_compute_mode': 'full',




}

# ============================================================
#           QUALITY CHECK SYSTEM INPUT HVAC
# ============================================================

# res = pybui.check_heating_system_inputs(INPUT_SYSTEM_HVAC)
res = check_heating_system_inputs(INPUT_SYSTEM_HVAC)


print("Selected Emitter:", res["emitter_type"])
print("Messaggi:")
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



def process_building(building_archetype, output_dir="result_test"):
    """Process a single building archetype and save results"""
    try:

        # Process the building
        (
            hourly_sim,
            annual_results_df,
            _,
        ) = _run_iso52016(building_archetype)

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
    path_hourly_sim_result = file_dir + "/hourly_sim__arch.csv"
    path_annual_sim_result = file_dir + "/annual_sim__arch.csv"
    hourly_sim.to_csv(path_hourly_sim_result)
    annual_results_df.to_csv(path_annual_sim_result)
        
    # ISO 15316-1 calculation
    df_in = calc.load_csv_data(hourly_sim)  # colonne: Q_H, T_op, T_ext (o alias)
    df_out = calc.run_timeseries()
    df_out.to_csv(file_dir + "/hourly_heating_system.csv")

    # Generate Graphs
    dir_chart_folder = file_dir
    name_report = "main_report"
    Graphs_and_report(df = hourly_sim,season ='heating_cooling',building_area=BUI['building']['net_floor_area'] ).bui_analysis_page(
    folder_directory=dir_chart_folder,
    name_file=name_report)


# ================================================================================================================
#                                   DHW NEEDS
# ================================================================================================================

# Water temperature of the mixed (cold and hot) water drawn at the tap
teta_W_draw = 42 
# Cold water temperature
teta_W_cold = 11.2 
# hot water delivery_temeprature 60°C
teta_w_h_ref = 60
# cold water supply temperature 13.5°X´C
teta_w_c_ref = 13.5
# Physical constant
# Building inputs
building_area = 1000
building_type = 'Dwellings'
# Use Profiles
hourly_fractions_examples = pd.DataFrame({
    "Workday" : [0,0,0,0,0,0,0,0,5,10,10,10,20,10,10,10,10,5,0,0,0,0,0,0],
    "Weekend" : [0,0,0,0,0,0,0,0,5,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0],
    "Holiday" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
)
sum_fractions = pd.DataFrame(hourly_fractions_examples.sum())
sum_fractions.columns= ["fractions"]
# National calendar
hourly_fractions = hourly_fractions_examples
calendar_nation = "Italy"
Italy_calendar = generate_calendar(calendar_nation, 2023)
n_workdays = sum(Italy_calendar['values'] == 'Working')
n_weekends = sum(Italy_calendar['values'] == 'Non-Working')
n_holidays = sum(Italy_calendar['values'] == 'Holiday')
Italy_calendar['values'].unique()
total_days = Italy_calendar.count().values[0]

# DHW needs
DHW_calc = Volume_and_energy_DHW_calculation(
    n_workdays, n_weekends, n_holidays,sum_fractions, total_days, hourly_fractions,
    teta_W_draw, 
    teta_w_c_ref,
    teta_w_h_ref,
    teta_W_cold,
    mode_calc= 'number_of_units', 
    building_type_B3= 'Residential', 
    building_area= 142, 
    unit_count= 10, 
    building_type_B5= 'Dwelling',
    residential_typology= 'residential_building - simple housing - AVG', # table B_4
    calculation_method= 'table',
    year= 2015,
    country_calendar=Italy_calendar
    )

# Plot data
t_start=24
t_end = 48
df = pd.DataFrame(dict(
    x = list(range(0,len(DHW_calc[6])))[t_start:t_end],
    y = DHW_calc[6][t_start:t_end],
))
fig = px.line(df, x="x", y="y", title = "DHW profile") 
fig.show()

df['z'] = df['y'].cumsum()
fig_1 = px.line(df, x="x", y="z", title = "DHW profile cumulative") 
fig_1.show()



# PRIMARY ENERGY CALCULATION 
