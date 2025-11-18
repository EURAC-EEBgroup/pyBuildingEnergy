# # __author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
# # __credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
# # __license__ = "MIT"
# # __version__ = "0.1"
# # __maintainer__ = "Daniele Antonucci"

# """

# # Acutal limitation
# Vedere test area edificio e area del solaio controterra differenti 
# # Italy:
# - tasso di ventilazione fissato a 0.3  h-1
# - considerato solo edifici tipo zona climatica media (E)

# """

# import numpy as np
# # from pybuildingenergy.source.functions import Perimeter_from_area, Area_roof, Check_area
# from source.functions import Perimeter_from_area, Area_roof, Check_area
# import pickle

# # ================================================================================================
# #                           COMPONENTS ARCHETYPE
# # ================================================================================================

# # @Italy
# # WALL
# code_wall = ["wall01", "wall02", "wall03"]
# description_wall = [
#     "Masonry with lists of stones and bricks (40cm)",
#     "solid brick masonry",
#     "hollow brick masonry",
# ]
# thickness_wall = [0.40, 0.38, 0.40]
# heat_capacity_wall = [665658, 523248, 319500]
# U_wall = [1.61, 1.48, 1.26]
# R_wall = [1 / value for value in U_wall]

# # ROOF
# code_roof = ["roof01", "roof02"]
# description_roof = [
#     "Pitched roof with wood structure and planking",
#     "Pitched roof with brick-concrete slab",
# ]
# thickness_roof = [0.34, 0.34]
# heat_capacity_roof = [278856, 390606]
# U_roof = [1.8, 2.2]
# R_roof = [1 / value for value in U_roof]

# # FLOOR
# code_floor = ["floor01", "floor02"]
# description_floor = [
#     "Concrete floor on soil",
#     "floor with reinforced brick-concreate slab, low insulation",
# ]
# thickness_floor = [0.27, 0.34]
# heat_capacity_floor = [463800, 448050]
# U_floor = [2.0, 0.98]
# R_floor = [1 / value for value in U_floor]

# # WINDOW
# code_window = ["window01", "window02"]
# description_window = [
#     "Single glass, methal frame without thermal break",
#     "single glasss wood frame",
# ]
# U_window = [5.7, 4.9]
# R_window = [1 / value for value in U_window]
# g_window = [0.85, 0.85]

# # ========================================================================================================================
# #                                       INPUTS: SINGLE FAMILY HOUSE
# # ========================================================================================================================

# periods = [
#     "before 1900",
#     "1901-1920",
#     "1921-1945",
#     "1946-1960",
#     "1961-1875",
#     "1976-1990",
#     "1991-2005",
#     "2006-today",
# ]
# bui_types = ["single_family_house"] * len(periods)
# area = [139, 115, 116, 162, 156, 199, 172, 174]
# window_area = [17.4, 14.4, 14.5, 20.3, 19.5, 24.9, 21.5, 21.8]  # 1/8 della superificie
# volume = [533, 448, 455, 583, 679, 725, 605, 607]
# coldest_month = [1] * len(periods)
# S_V = [0.77, 0.82, 0.81, 0.75, 0.73, 0.72, 0.73, 0.72]
# S_envelope = [S * volume for S, volume in zip(S_V, volume)]
# number_of_floor = [2, 2, 2, 2, 2, 2, 2, 2]
# height = [
#     round(volume_i / (area_i / number_of_floor_i), 2)
#     for volume_i, area_i, number_of_floor_i in zip(volume, area, number_of_floor)
# ]
# bui_height = [x / (y / z) for x, y, z in zip(volume, area, number_of_floor)]
# base = [
#     (value / number_of_floor) / 10
#     for value, number_of_floor in zip(area, number_of_floor)
# ]
# perimeter = [Perimeter_from_area(value, 10 / 2) for value in area]
# area_north = [round(10 * heights, 2) for heights in bui_height]
# area_south = area_north
# area_west = [round(bases * heights, 2) for bases, heights in zip(base, bui_height)]
# area_east = area_west
# area_roof = [round(Area_roof(10, leng_roof) / 2, 2) for leng_roof in base]
# thermal_bridge_heat = [10] * len(periods)
# w_code = [
#     "wall01",
#     "wall01",
#     "wall02",
#     "wall02",
#     "wall03",
#     "wall01",
#     "wall01",
#     "wall01",
# ]
# r_code = [
#     "roof01",
#     "roof01",
#     "roof01",
#     "roof02",
#     "roof02",
#     "roof01",
#     "roof01",
#     "roof01",
# ]
# win_code = [
#     "window01",
#     "window01",
#     "window02",
#     "window02",
#     "window02",
#     "window02",
#     "window02",
#     "window02",
# ]
# f_code = [
#     "floor01",
#     "floor01",
#     "floor01",
#     "floor01",
#     "floor01",
#     "floor01",
#     "floor01",
#     "floor01",
# ]
# building_category_const = ["old", "old", "old", "old", "old", "old", "medium", "medium"]
# air_change_rate_base_value = [0.08, 0.14, 0.14, 0.1, 0.1, 0.1, 0.1, 0.1]
# # GLOBAL INPUTS
# typology_elements = np.array(
#     ["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"], dtype=object
# )
# orientation_elements = np.array(
#     ["NV", "SV", "EV", "WV", "HOR", "HOR", "NV", "SV", "EV", "WV"], dtype=object
# )
# solar_abs_elements = np.array(
#     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.6, 0.6, 0.6, 0.6], dtype=object
# )
# heat_convective_elements_internal = np.array(
#     [2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object
# )
# heat_radiative_elements_internal = np.array(
#     [5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13], dtype=object
# )
# heat_convective_elements_external = np.array(
#     [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=object
# )
# heat_radiative_elements_external = np.array(
#     [4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14], dtype=object
# )
# sky_factor_elements = np.array(
#     [0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object
# )
# baseline_hci = np.array(
#     [2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object
# )
# baseline_hce = np.array(
#     [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=object
# )

# profile_residential_1 = {
#     "code": "profile01",
#     "type": "residential",
#     "profile_workdays_internal_gains": np.array(
#         [
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#         ]
#     ),
#     "profile_weekend_internal_gains": np.array(
#         [
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#         ]
#     ),
#     "profile_workdays_ventilation": np.array(
#         [
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#         ]
#     ),
#     "profile_weekend_ventilation": np.array(
#         [
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#         ]
#     ),
# }

# heating_installed = [True] * len(periods)
# cooling_installed = [False] * len(periods)

# # ========================================================================================================================
# #                                       GENERATE BUILDING ARCHETYPE
# # ========================================================================================================================


# class Buildings(object):
#     def __init__(
#         self,
#         building_type: str,
#         periods: str,
#         latitude: float,
#         longitude: float,
#         exposed_perimeter: float,
#         area: float,
#         number_of_floor: int,
#         height: float,
#         volume: float,
#         slab_on_ground: float = None,
#         wall_thickness=float,
#         coldest_month=int,
#         surface_envelope=float,
#         surface_envelope_model=float,
#         side=float,
#         heating_mode=bool,  # True or False if heating syste is
#         cooling_mode=bool,
#         heating_setpoint=float,  # Heating set-point in °C
#         cooling_setpoint=float,  # Cooling set-point in °
#         heating_setback=float,  # Heating set-back in °C
#         cooling_setback=float,  # Cooling set-back in °C
#         power_heating_max=float,  # Max Power of the heating system
#         power_cooling_max=float,  # Max Power of the cooling system
#         air_change_rate_base_value=float,  # air change rate
#         air_change_rate_extra=float,
#         internal_gains_base_value=float,
#         internal_gains_extra=float,
#         thermal_bridge_heat=float,  # value of thermal bridges
#         thermal_resistance_floor=float,
#         area_elements=list,  # Area of each facade elements
#         transmittance_U_elments=list,
#         thermal_resistance_R_elements=list,
#         thermal_capacity_elements=list,
#         g_factor_windows=list,
#         occ_level_wd=np.array,
#         occ_level_we=np.array,
#         comf_level_wd=np.array,
#         comf_level_we=np.array,
#         construction_class="class_i",
#         **kwargs,
#     ):

#         self.building_type = building_type
#         self.periods = periods
#         self.latitude = latitude
#         self.longitude = longitude
#         self.annual_mean_internal_temperature = kwargs.get(
#             "annual_mean_internal_temperature"
#         )
#         self.annual_mean_external_temperature = kwargs.get(
#             "annual_mean_external_temperature"
#         )
#         self.amplitude_of_internal_temperature_variations = kwargs.get(
#             "amplitude_of_internal_temperature_variations"
#         )
#         self.a_use = area
#         self.exposed_perimeter = exposed_perimeter
#         self.height = height
#         self.number_of_floor = number_of_floor
#         self.volume = volume  # Initialize attribute to None
#         self.slab_on_ground = slab_on_ground
#         self.wall_thickness = wall_thickness
#         self.coldest_month = coldest_month
#         self.surface_envelope = (
#             surface_envelope  # calculated from sum of surfaces + floor  + roof
#         )
#         self.surface_envelope_model = (
#             surface_envelope_model  # calculated from S/V coefficient
#         )
#         self.base = side  # side of rectangular shape of building, a side of 10 meters is hypothesized
#         self.heating_mode = heating_mode
#         self.cooling_mode = cooling_mode
#         self.heating_setpoint = heating_setpoint
#         self.heating_setback = heating_setback
#         self.cooling_setpoint = cooling_setpoint
#         self.cooling_setback = cooling_setback
#         self.power_cooling_max = -power_cooling_max
#         self.power_heating_max = power_heating_max
#         self.air_change_rate_base_value = air_change_rate_base_value
#         self.air_change_rate_extra = air_change_rate_extra
#         self.internal_gains_base_value = internal_gains_base_value
#         self.internal_gains_extra = internal_gains_extra
#         self.thermal_bridge_heat = thermal_bridge_heat
#         self.thermal_resistance_floor = thermal_resistance_floor
#         self.typology_elements = typology_elements
#         self.orientation_elements = orientation_elements
#         self.solar_abs_elements = solar_abs_elements
#         self.area_elements = area_elements
#         self.transmittance_U_elments = transmittance_U_elments
#         self.thermal_resistance_R_elements = thermal_resistance_R_elements
#         self.thermal_capacity_elements = thermal_capacity_elements
#         self.g_factor_windows = g_factor_windows
#         self.heat_convective_elements_internal = heat_convective_elements_internal
#         self.heat_radiative_elements_internal = heat_radiative_elements_internal
#         self.heat_convective_elements_external = heat_convective_elements_external
#         self.heat_radiative_elements_external = heat_radiative_elements_external
#         self.sky_factor_elements = sky_factor_elements
#         self.occ_level_wd = occ_level_wd
#         self.occ_level_we = occ_level_we
#         self.comf_level_wd = comf_level_wd
#         self.comf_level_we = comf_level_we
#         self.baseline_hci = baseline_hci
#         self.baseline_hce = baseline_hce
#         self.construction_class = construction_class
#         self.weather_source = kwargs.get("weather_source")
#         # self.tmy_filename = kwargs.get("tmy_filename")
#         # self.location = kwargs.get("location")

#     @property
#     def slab_on_ground(self):
#         return self._slab_on_ground

#     @slab_on_ground.setter
#     def slab_on_ground(self, value):
#         """
#         Calculation of the slab on ground area
#         Aslab = Area/number_of_floor [m2]
#         :param value: slab on ground area [m2]
#         :return Area of slab on ground
#         """
#         if value is None:
#             self._slab_on_ground = self.a_use / self.number_of_floor
#         else:
#             self._slab_on_ground = value

#     @property
#     def volume(self):
#         return self._volume

#     @volume.setter
#     def volume(self, value):
#         """
#         Calucalte Volume, if not provided
#         :param value: building volume
#         :retrun Volume
#         """
#         if value is None:
#             self._volume = (self.a_use / self.number_of_floor) * self.height
#         else:
#             self._volume = value

#     # Area of building elements can not be 0. Se
#     @property
#     def area_elements(self):
#         return self._area_elements

#     @area_elements.setter
#     def area_elements(self, values):
#         """
#         Quality check of area value
#         :param values:List of areas for each individual facade element
#         """
#         self._area_elements = Check_area(values)

#     def update_values(self, new_values: dict) -> object:
#         """
#         Update characteristic of a building providing a dictionary with new values

#         :param new_values: new values of a building feature
#         :return building object updated

#         .. note::
#             new_inputs = {
#                 'latitude':46.66345144066082,
#                 'longitude':9.71636944229362,
#                 'Eln':10, #
#                 'a_use': 100,
#                 "slab_on_ground":100,#
#                 'heating_setpoint':22,
#                 'cooling_setpoint':24,
#                 'power_heating_max':40000,
#                 'power_cooling_max':-10000
#             }
#         """
#         for key, value in new_values.items():
#             if not hasattr(self, key):
#                 pass
#             if isinstance(value, list):
#                 if isinstance(getattr(self, key), list):
#                     if len(value) != len(getattr(self, key)):
#                         raise ValueError(
#                             f"The length of '{key}' must match the original length"
#                         )
#                     setattr(self, key, value)
#                 else:
#                     raise ValueError(
#                         f"The length of '{key}' must match the original length"
#                     )
#             else:
#                 if isinstance(getattr(self, key), list):
#                     raise ValueError(
#                         f"The length of '{key}' must match the original length"
#                     )
#                 setattr(self, key, value)

#     def inputs_validation(self):
#         """
#         Validate inputs according to define rules and provide list of possible errors
#         Rules:

#             * Perimeter should be lower than area of building. Limitation building higher than 16m2
#             * Transmittance values too hight or too low
#             * Area of the floor slab on gorund should be lower than the area of the roof
#         """
#         # Check Volume
#         if self.volume is None:
#             self.volume = (self.a_use / self.number_of_floor) * self.height

#         #
#         quality_check_errors = []
#         # 1. Check perimeter and area
#         if self.a_use >= 16:
#             if self.exposed_perimeter > self.a_use:
#                 quality_check_errors.append(
#                     "Possible error. Check the value of perimeter and area if they are correct."
#                 )

#         # 2. Check value of envelope transmittance
#         for i, u_value in enumerate(self.transmittance_U_elments):
#             element = self.typology_elements[i]
#             if element == "OP":
#                 nameElement = "Opaque Element"
#             elif element == "W":
#                 nameElement = "Transaprent Element"
#             elif element == "HOR":
#                 nameElement = "Floor or Roof"

#             orient_elment = self.orientation_elements[i]

#             if u_value > 8 or u_value <= 0.1:
#                 quality_check_errors.append(
#                     f"Possible error. Transmittance of the element {nameElement} oriented to {orient_elment} too low or too hight"
#                 )

#         # 3. Check area roof and floor slab on ground
#         area_roof = self.area_elements[5]
#         area_floor = self.area_elements[6]
#         if area_floor == area_roof:
#             quality_check_errors.append(
#                 f"Warning!. The area of the floor slab on ground is higher than the area of the roof"
#             )

#         print(quality_check_errors)
#         return quality_check_errors


# # ===================================================================================================
# #                           GET INPUTS FROM SPECIFIC ARCHETYPE
# # ===================================================================================================
# class Buildings_from_dictionary(object):
#     def __init__(self, data):
#         for key, value in data.items():
#             setattr(self, key, value)

#     def update_values(self, new_values: dict) -> object:
#         """
#         Update characteristic of a building providing a dictionary with new values

#         :param new_values: dictioanry with new values of a building feature
#         :return building object updated

#         .. note::
#             Example
#             new_inputs = {
#                 'latitude':46.66345144066082,
#                 'longitude':9.71636944229362,
#                 'Eln':10, 
#                 'a_use': 100,
#                 "slab_on_ground":100,
#                 'heating_setpoint':22,
#                 'cooling_setpoint':24,
#                 'power_heating_max':40000,
#                 'power_cooling_max':-10000
#             }
#         """
#         for key, value in new_values.items():
#             if not hasattr(self, key):
#                 pass
#             if isinstance(value, list):
#                 if isinstance(getattr(self, key), list):
#                     if len(value) != len(getattr(self, key)):
#                         raise ValueError(
#                             f"The length of '{key}' must match the original length"
#                         )
#                     setattr(self, key, value)
#                 else:
#                     raise ValueError(
#                         f"The length of '{key}' must match the original length"
#                     )
#             else:
#                 if isinstance(getattr(self, key), list):
#                     raise ValueError(
#                         f"The length of '{key}' must match the original length"
#                     )
#                 setattr(self, key, value)

#     def update_facade_elements(self, new_values):
#         """
#         Update facade elements

#         :param new_values: new_values from a dictionary

#         """
#         for key, value in new_values.items():
#             if not hasattr(self, key):
#                 pass
#             setattr(self, key, value)

#     def inputs_validation(self):
#         """
#         QUALITY CHECK

#         Validate inputs according to the following rules and provide list of possible errors

#         Rules:

#             * Perimeter should be lower than area of building. Limitation building higher than 16m2
#             * Transmittance values too hight or too low
#             * Check area of the wall should be higher than the area of the window for the same orientation
#             * Area of the floor slab on gorund should be lower than the area of the roof

#         """
#         quality_check_errors = []
#         # 1. Check perimeter and area
#         if self.a_use >= 16:
#             if self.exposed_perimeter > self.a_use:
#                 quality_check_errors.append(
#                     "Possible error. Check the value of perimeter and area if they are correct."
#                 )

#         # 2. Check value of envelope transmittance
#         for i, u_value in enumerate(self.transmittance_U_elments):
#             element = self.typology_elements[i]
#             if element == "OP":
#                 nameElement = "Opaque Element"
#             elif element == "W":
#                 nameElement = "Transaprent Element"
#             elif element == "HOR":
#                 nameElement = "Floor or Roof"

#             orient_elment = self.orientation_elements[i]

#             if u_value > 8 or u_value <= 0.1:
#                 quality_check_errors.append(
#                     f"Possible error. Transmittance of the element {nameElement} oriented to {orient_elment} too low or too hight"
#                 )

#         # 3. Check area roof and floor slab on ground
#         area_roof = self.area_elements[5]
#         area_floor = self.area_elements[6]
#         if area_floor == area_roof:
#             quality_check_errors.append(
#                 f"Warning!. The area of the floor slab on ground is higher than the area of the roof"
#             )

#         print(quality_check_errors)
#         return quality_check_errors


# class Selected_bui_archetype:
#     def __init__(self, bui_archetype, period, latitude, longitude):
#         self.bui_archetype = bui_archetype
#         self.period = period
#         self.latitude = float(latitude)
#         self.longitude = float(longitude)
#         print(bui_archetype, period, latitude, longitude)

#         if bui_archetype in bui_types:
#             self.bui_archetype = bui_archetype
#         else:
#             raise ValueError(
#                 f"Invalid choice for archetype. Possible choices are: {', '.join(bui_types)}"
#             )

#         if period in periods:
#             self.built_year = period
#         else:
#             raise ValueError(
#                 f"Invalid choice for possible periods. Possible choices are: {', '.join(periods)}"
#             )

#         if isinstance(latitude, float):
#             self.latitude = latitude
#         else:
#             raise ValueError("latitude should be a float")

#         if isinstance(longitude, float):
#             self.longitude = longitude
#         else:
#             raise ValueError("longitude should be a float")

#     def get_archetype(self, pickle_file_path):
#         """
#         Get archetype from a list of possible archetypes defined in the ``archetypes.pickle``
#         :param pickle_file_path: path of the archetypes pickle file
#         :return building archetype (*type*: obj)
#         """
#         # Read data from the pickle file
#         with open(pickle_file_path, "rb") as f:
#             archetypes = pickle.load(f)
#         # Filter according to inputs
#         selected_archetype = [
#             bui
#             for bui in archetypes
#             if bui["building_type"] == self.bui_archetype
#             and bui["periods"] == self.period
#         ][0]
#         selected_archetype["latitude"] = self.latitude
#         selected_archetype["longitude"] = self.longitude

#         return Buildings_from_dictionary(selected_archetype)

# -*- coding: utf-8 -*-
# __author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
# __credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
# __license__ = "MIT"
# __version__ = "0.2"
# __maintainer__ = "Daniele Antonucci"

"""
Refactor:
- Inputs now taken from BUI schema (see dict BUI below).
- Added INPUT_SYSTEM_HVAC handling and HVACSystem class.
- Kept previous Buildings and Buildings_from_dictionary APIs for backward compatibility.
- Fixed minor issues (missing pandas import; stray return; safer defaults).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from functions import Perimeter_from_area, Area_roof, Check_area
# from source.functions import Perimeter_from_area, Area_roof, Check_area
import pickle

# ================================================================================================
#                           COMPONENTS ARCHETYPE (legacy, kept for reference)
# ================================================================================================

# @Italy
# WALL (legacy)
code_wall = ["wall01", "wall02", "wall03"]
description_wall = [
    "Masonry with lists of stones and bricks (40cm)",
    "solid brick masonry",
    "hollow brick masonry",
]
thickness_wall = [0.40, 0.38, 0.40]
heat_capacity_wall = [665658, 523248, 319500]
U_wall = [1.61, 1.48, 1.26]
R_wall = [1 / value for value in U_wall]

# ROOF (legacy)
code_roof = ["roof01", "roof02"]
description_roof = [
    "Pitched roof with wood structure and planking",
    "Pitched roof with brick-concrete slab",
]
thickness_roof = [0.34, 0.34]
heat_capacity_roof = [278856, 390606]
U_roof = [1.8, 2.2]
R_roof = [1 / value for value in U_roof]

# FLOOR (legacy)
code_floor = ["floor01", "floor02"]
description_floor = [
    "Concrete floor on soil",
    "floor with reinforced brick-concreate slab, low insulation",
]
thickness_floor = [0.27, 0.34]
heat_capacity_floor = [463800, 448050]
U_floor = [2.0, 0.98]
R_floor = [1 / value for value in U_floor]

# WINDOW (legacy)
code_window = ["window01", "window02"]
description_window = [
    "Single glass, methal frame without thermal break",
    "single glasss wood frame",
]
U_window = [5.7, 4.9]
R_window = [1 / value for value in U_window]
g_window = [0.85, 0.85]

# ================================================================================================
#                                       GLOBAL DEFAULTS
# ================================================================================================
# Order used by the thermal model (10 elements total)
# 0..3 = vertical opaque N,S,E,W; 4 = roof (HOR); 5 = floor (HOR); 6..9 = windows N,S,E,W
typology_elements = np.array(["OP", "OP", "OP", "OP", "HOR", "HOR", "W", "W", "W", "W"], dtype=object)
orientation_elements = np.array(["NV", "SV", "EV", "WV", "HOR", "HOR", "NV", "SV", "EV", "WV"], dtype=object)

# Radiative/convective coefficients & factors (defaults, kept as before)
solar_abs_elements = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.6, 0.6, 0.6, 0.6], dtype=object)
heat_convective_elements_internal = np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object)
heat_radiative_elements_internal  = np.array([5.13]*10, dtype=object)
heat_convective_elements_external = np.array([20.0]*10, dtype=object)
heat_radiative_elements_external  = np.array([4.14]*10, dtype=object)
sky_factor_elements               = np.array([0.50, 0.50, 0.50, 0.50, 1.00, 0.00, 0.50, 0.50, 0.50, 0.50], dtype=object)
baseline_hci = heat_convective_elements_internal.copy()
baseline_hce = heat_convective_elements_external.copy()

# Simple residential profiles (legacy)
profile_residential_1 = {
    "code": "profile01",
    "type": "residential",
    "profile_workdays_internal_gains": np.array(
        [1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1]
    ),
    "profile_weekend_internal_gains": np.array(
        [1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1]
    ),
    "profile_workdays_ventilation": np.array(
        [1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1]
    ),
    "profile_weekend_ventilation": np.array(
        [1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1]
    ),
}

heating_installed = [True]
cooling_installed = [False]

# ================================================================================================
#                                       CLASSES
# ================================================================================================

class Buildings(object):
    def __init__(
        self,
        building_type: str,
        periods: str,
        latitude: float,
        longitude: float,
        exposed_perimeter: float,
        area: float,
        number_of_floor: int,
        height: float,
        volume: float,
        slab_on_ground: float = None,
        wall_thickness=float,
        coldest_month=int,
        surface_envelope=float,
        surface_envelope_model=float,
        side=float,
        heating_mode=bool,  # True or False
        cooling_mode=bool,
        heating_setpoint=float,  # °C
        cooling_setpoint=float,  # °C
        heating_setback=float,   # °C
        cooling_setback=float,   # °C
        power_heating_max=float,  # W
        power_cooling_max=float,  # W (positive input; stored as negative internally)
        air_change_rate_base_value=float,  # ach
        air_change_rate_extra=float,
        internal_gains_base_value=float,
        internal_gains_extra=float,
        thermal_bridge_heat=float,  # W/K
        thermal_resistance_floor=float,
        area_elements=list,  # len=10 as per global order
        transmittance_U_elments=list,
        thermal_resistance_R_elements=list,
        thermal_capacity_elements=list,
        g_factor_windows=list,
        occ_level_wd=np.array,
        occ_level_we=np.array,
        comf_level_wd=np.array,
        comf_level_we=np.array,
        construction_class="class_i",
        **kwargs,
    ):
        self.building_type = building_type
        self.periods = periods
        self.latitude = latitude
        self.longitude = longitude
        self.annual_mean_internal_temperature = kwargs.get("annual_mean_internal_temperature")
        self.annual_mean_external_temperature = kwargs.get("annual_mean_external_temperature")
        self.amplitude_of_internal_temperature_variations = kwargs.get("amplitude_of_internal_temperature_variations")
        self.a_use = area
        self.exposed_perimeter = exposed_perimeter
        self.height = height
        self.number_of_floor = number_of_floor
        self.volume = volume
        self.slab_on_ground = slab_on_ground
        self.wall_thickness = wall_thickness
        self.coldest_month = coldest_month
        self.surface_envelope = surface_envelope
        self.surface_envelope_model = surface_envelope_model
        self.base = side
        self.heating_mode = heating_mode
        self.cooling_mode = cooling_mode
        self.heating_setpoint = heating_setpoint
        self.heating_setback = heating_setback
        self.cooling_setpoint = cooling_setpoint
        self.cooling_setback = cooling_setback
        self.power_cooling_max = -abs(power_cooling_max)
        self.power_heating_max = abs(power_heating_max)
        self.air_change_rate_base_value = air_change_rate_base_value
        self.air_change_rate_extra = air_change_rate_extra
        self.internal_gains_base_value = internal_gains_base_value
        self.internal_gains_extra = internal_gains_extra
        self.thermal_bridge_heat = thermal_bridge_heat
        self.thermal_resistance_floor = thermal_resistance_floor

        # per-element properties
        self.typology_elements = typology_elements
        self.orientation_elements = orientation_elements
        self.solar_abs_elements = solar_abs_elements
        self.area_elements = area_elements
        self.transmittance_U_elments = transmittance_U_elments
        self.thermal_resistance_R_elements = thermal_resistance_R_elements
        self.thermal_capacity_elements = thermal_capacity_elements
        self.g_factor_windows = g_factor_windows
        self.heat_convective_elements_internal = heat_convective_elements_internal
        self.heat_radiative_elements_internal = heat_radiative_elements_internal
        self.heat_convective_elements_external = heat_convective_elements_external
        self.heat_radiative_elements_external = heat_radiative_elements_external
        self.sky_factor_elements = sky_factor_elements

        # profiles
        self.occ_level_wd = occ_level_wd
        self.occ_level_we = occ_level_we
        self.comf_level_wd = comf_level_wd
        self.comf_level_we = comf_level_we
        self.baseline_hci = baseline_hci
        self.baseline_hce = baseline_hce
        self.construction_class = construction_class
        self.weather_source = kwargs.get("weather_source")

        # HVAC placeholder (will be attached by builder)
        self.hvac: Optional["HVACSystem"] = None

    @property
    def slab_on_ground(self):
        return self._slab_on_ground

    @slab_on_ground.setter
    def slab_on_ground(self, value):
        if value is None:
            self._slab_on_ground = self.a_use / self.number_of_floor
        else:
            self._slab_on_ground = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        if value is None:
            self._volume = (self.a_use / self.number_of_floor) * self.height
        else:
            self._volume = value

    @property
    def area_elements(self):
        return self._area_elements

    @area_elements.setter
    def area_elements(self, values):
        self._area_elements = Check_area(values)

    def update_values(self, new_values: dict) -> object:
        for key, value in new_values.items():
            if not hasattr(self, key):
                continue
            if isinstance(value, list):
                if isinstance(getattr(self, key), list):
                    if len(value) != len(getattr(self, key)):
                        raise ValueError(f"The length of '{key}' must match the original length")
                    setattr(self, key, value)
                else:
                    raise ValueError(f"The length of '{key}' must match the original length")
            else:
                if isinstance(getattr(self, key), list):
                    raise ValueError(f"The length of '{key}' must match the original length")
                setattr(self, key, value)

    def inputs_validation(self):
        if self.volume is None:
            self.volume = (self.a_use / self.number_of_floor) * self.height

        quality_check_errors = []
        if self.a_use >= 16 and self.exposed_perimeter > self.a_use:
            quality_check_errors.append("Possible error. Perimeter larger than area (check units/values).")

        for i, u_value in enumerate(self.transmittance_U_elments):
            element = self.typology_elements[i]
            nameElement = "Opaque Element" if element == "OP" else ("Transparent Element" if element == "W" else "Floor/Roof")
            orient_elment = self.orientation_elements[i]
            if u_value > 8 or u_value <= 0.1:
                quality_check_errors.append(
                    f"Possible error. U-value of {nameElement} ({orient_elment}) too low/high: {u_value}"
                )

        area_roof = self.area_elements[4]
        area_floor = self.area_elements[5]
        if area_floor > area_roof:
            quality_check_errors.append("Warning: slab-on-ground area is higher than roof area")

        return quality_check_errors


class Buildings_from_dictionary(object):
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def update_values(self, new_values: dict) -> object:
        for key, value in new_values.items():
            if not hasattr(self, key):
                continue
            if isinstance(value, list):
                if isinstance(getattr(self, key), list):
                    if len(value) != len(getattr(self, key)):
                        raise ValueError(f"The length of '{key}' must match the original length")
                    setattr(self, key, value)
                else:
                    raise ValueError(f"The length of '{key}' must match the original length")
            else:
                if isinstance(getattr(self, key), list):
                    raise ValueError(f"The length of '{key}' must match the original length")
                setattr(self, key, value)

    def update_facade_elements(self, new_values):
        for key, value in new_values.items():
            if not hasattr(self, key):
                continue
            setattr(self, key, value)

    def inputs_validation(self):
        quality_check_errors = []
        if self.a_use >= 16 and self.exposed_perimeter > self.a_use:
            quality_check_errors.append("Possible error. Perimeter larger than area (check units/values).")

        for i, u_value in enumerate(self.transmittance_U_elments):
            element = self.typology_elements[i]
            nameElement = "Opaque Element" if element == "OP" else ("Transparent Element" if element == "W" else "Floor/Roof")
            orient_elment = self.orientation_elements[i]
            if u_value > 8 or u_value <= 0.1:
                quality_check_errors.append(
                    f"Possible error. U-value of {nameElement} ({orient_elment}) too low/high: {u_value}"
                )

        area_roof = self.area_elements[4]
        area_floor = self.area_elements[5]
        if area_floor > area_roof:
            quality_check_errors.append("Warning: slab-on-ground area is higher than roof area")

        return quality_check_errors


class Selected_bui_archetype:
    def __init__(self, bui_archetype, period, latitude, longitude):
        self.bui_archetype = bui_archetype
        self.period = period
        self.latitude = float(latitude)
        self.longitude = float(longitude)

    def get_archetype(self, pickle_file_path):
        with open(pickle_file_path, "rb") as f:
            archetypes = pickle.load(f)
        selected_archetype = [
            bui for bui in archetypes
            if bui["building_type"] == self.bui_archetype and bui["periods"] == self.period
        ][0]
        selected_archetype["latitude"] = self.latitude
        selected_archetype["longitude"] = self.longitude
        return Buildings_from_dictionary(selected_archetype)

# ================================================================================================
#                              HVAC System definition (NEW)
# ================================================================================================

@dataclass
class HVACSystem:
    config: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    @property
    def emitter_type(self) -> str:
        return self.config.get('emitter_type', 'Unknown')

    # Add convenience accessors as needed...
    # Example:
    @property
    def nominal_power(self) -> float:
        return float(self.config.get('nominal_power', 0.0))

# ================================================================================================
#                   Utilities: map BUI -> internal arrays expected by Buildings
# ================================================================================================

def _orient_key(azimuth: float, tilt: float, typ: str) -> Tuple[str, str]:
    """Return (typology, orientation) key for mapping."""
    if typ == "opaque":
        t = "OP"
    elif typ == "transparent":
        t = "W"
    else:
        t = "OP"

    if tilt == 0:
        o = "HOR"
    else:
        # Map azimuth to NV/SV/EV/WV
        if   315 <= azimuth <= 360 or 0 <= azimuth < 45:  o = "NV"
        elif 45 <= azimuth < 135:                         o = "EV"
        elif 135 <= azimuth < 225:                        o = "SV"
        else:                                             o = "WV"
    return t, o

def bui_to_internal_arrays(BUI: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Build arrays (len=10) matching the global order:
      [OP NV, OP SV, OP EV, OP WV, HOR roof, HOR floor, W NV, W SV, W EV, W WV]
    Missing items are set to zero area and reasonable defaults.
    """
    # Initialize defaults
    area      = np.zeros(10, dtype=float)
    U         = np.ones(10, dtype=float) * 1.5      # default U
    R         = 1.0 / U
    C_th      = np.zeros(10, dtype=float)           # thermal capacity
    g_win     = np.zeros(10, dtype=float)           # only for windows
    s_abs     = solar_abs_elements.astype(float).copy()
    sky_fact  = sky_factor_elements.astype(float).copy()

    # Helper to index by (typ, orient)
    index_map = {
        ("OP","NV"):0, ("OP","SV"):1, ("OP","EV"):2, ("OP","WV"):3,
        ("HOR","HOR_roof"):4, ("HOR","HOR_floor"):5,
        ("W","NV"):6, ("W","SV"):7, ("W","EV"):8, ("W","WV"):9
    }

    # Parse building surfaces
    for surf in BUI.get("building_surface", []):
        typ  = surf.get("type", "opaque").lower()
        az   = float(surf.get("orientation", {}).get("azimuth", 0))
        tilt = float(surf.get("orientation", {}).get("tilt", 90))
        a    = float(surf.get("area", 0))
        u    = float(surf.get("u_value", 1.5))
        cap  = float(surf.get("thermal_capacity", 0.0))
        svf  = float(surf.get("sky_view_factor", 0.5))
        sabs = float(surf.get("solar_absorptance", 0.6))

        if typ == "opaque" and tilt == 0:
            # Distinguish roof vs floor by name heuristic
            key = ("HOR", "HOR_roof") if "roof" in surf.get("name","").lower() else ("HOR","HOR_floor")
            idx = index_map[key]
        else:
            t, o = _orient_key(az, tilt, typ)
            idx = index_map[(t, o)]

        area[idx] = a
        U[idx] = u
        R[idx] = 1.0 / max(u, 1e-6)
        C_th[idx] = cap
        sky_fact[idx] = svf
        s_abs[idx] = sabs

        if typ == "transparent":
            g = float(surf.get("g_value", 0.6))
            g_win[idx] = g

    # Ensure windows present for each orientation: if missing, keep area 0
    # For windows with area 0, g can stay 0.

    return {
        "area_elements": area,
        "transmittance_U_elments": U,
        "thermal_resistance_R_elements": R,
        "thermal_capacity_elements": C_th,
        "g_factor_windows": g_win,
        "solar_abs_elements": s_abs,
        "sky_factor_elements": sky_fact,
    }

# ================================================================================================
#                           Builder: create Buildings from BUI + HVAC
# ================================================================================================

def build_building_from_BUI(BUI: Dict[str, Any], INPUT_SYSTEM_HVAC: Optional[Dict[str, Any]] = None) -> Buildings:
    b = BUI["building"]
    bp = BUI["building_parameters"]

    # Envelope arrays
    arr = bui_to_internal_arrays(BUI)

    # Global shape params
    a_use = float(b.get("net_floor_area", 0.0))
    n_floors = int(b.get("n_floors", 1))
    height = float(b.get("height", 3.0))
    exposed_perimeter = float(b.get("exposed_perimeter", Perimeter_from_area(a_use, 10/2)))
    base_side = 10.0  # same hypothesis as before

    # Envelope totals
    area_roof = arr["area_elements"][4]
    area_floor = arr["area_elements"][5]
    surface_envelope = float(np.sum(arr["area_elements"][:6]))  # opaque + roof + floor (no windows added here)
    S_V = surface_envelope / max((a_use / n_floors) * height, 1e-6)

    # Setpoints & capacities
    tset = bp["temperature_setpoints"]
    caps = bp["system_capacities"]
    climate = bp.get("climate_parameters", {})
    coldest_month = int(climate.get("coldest_month", 1))

    # Air exchange
    air = bp.get("airflow_rates", {})
    ach = float(air.get("infiltration_rate", 0.5))  # ACH
    ach_extra = float(air.get("ventilation_rate_extra", 0.0))

    # Internal gains (W/m²) → take base as sum of full_load (as indicative)
    int_gains = bp.get("internal_gains", [])
    internal_gains_base = sum(float(x.get("full_load", 0)) for x in int_gains)
    internal_gains_extra = 0.0

    # Occupancy / ventilation profiles (24 values each)
    def _arr_24(dct, key): return np.array(dct.get(key, [0]*24), dtype=float)

    heating_prof = bp.get("heating_profile", {})
    cooling_prof = bp.get("cooling_profile", {})
    ventilation_prof = bp.get("ventilation_profile", {})

    occ_wd = _arr_24(heating_prof, "weekday")  # generic slot; replace with your own occupancy if available
    occ_we = _arr_24(heating_prof, "weekend")
    comf_wd = _arr_24(cooling_prof, "weekday")
    comf_we = _arr_24(cooling_prof, "weekend")

    building = Buildings(
        building_type=b.get("building_type_class", "Residential"),
        periods="BUI_input",
        latitude=float(b.get("latitude")),
        longitude=float(b.get("longitude")),
        exposed_perimeter=exposed_perimeter,
        area=a_use,
        number_of_floor=n_floors,
        height=height,
        volume=b.get("volume"),  # if None, computed from a_use/n_floors*height
        slab_on_ground=b.get("slab_on_ground"),  # if None, computed from area/n_floors
        wall_thickness=float(b.get("wall_thickness", 0.3)),
        coldest_month=coldest_month,
        surface_envelope=surface_envelope,
        surface_envelope_model=S_V,
        side=base_side,
        heating_mode=True,
        cooling_mode=True,
        heating_setpoint=float(tset.get("heating_setpoint", 20.0)),
        cooling_setpoint=float(tset.get("cooling_setpoint", 26.0)),
        heating_setback=float(tset.get("heating_setback", 17.0)),
        cooling_setback=float(tset.get("cooling_setback", 30.0)),
        power_heating_max=float(caps.get("heating_capacity", 0.0)),
        power_cooling_max=float(caps.get("cooling_capacity", 0.0)),
        air_change_rate_base_value=ach,
        air_change_rate_extra=ach_extra,
        internal_gains_base_value=internal_gains_base,
        internal_gains_extra=internal_gains_extra,
        thermal_bridge_heat=float(bp.get("construction", {}).get("thermal_bridges", 0.0)),
        thermal_resistance_floor=1.0 / max(arr["transmittance_U_elments"][5], 1e-6),
        area_elements=arr["area_elements"].tolist(),
        transmittance_U_elments=arr["transmittance_U_elments"].tolist(),
        thermal_resistance_R_elements=arr["thermal_resistance_R_elements"].tolist(),
        thermal_capacity_elements=arr["thermal_capacity_elements"].tolist(),
        g_factor_windows=arr["g_factor_windows"].tolist(),
        occ_level_wd=occ_wd,
        occ_level_we=occ_we,
        comf_level_wd=comf_wd,
        comf_level_we=comf_we,
        construction_class=b.get("construction_class", "class_i"),
    )

    # Attach HVAC if provided
    if INPUT_SYSTEM_HVAC:
        building.hvac = HVACSystem(INPUT_SYSTEM_HVAC)

    return building

# ================================================================================================
#                                      EXAMPLE INPUTS
# ================================================================================================

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
        "construction_class": "class_i"
    },
    "adjacent_zones": [
        {
            "name":"adj_1",
            "orientation_zone": {"azimuth": 0},
            "area_facade_elements": np.array([20,60,30,30,50,50], dtype=object),
            "typology_elements": np.array(['OP','OP','OP','OP','GR','OP'], dtype=object),
            "transmittance_U_elements": np.array([0.82,0.82,0.82,0.82,0.52,1.16], dtype=object),
            "orientation_elements": np.array(['NV','SV','EV','WV','HOR','HOR'], dtype=object),
            'volume': 300, 'building_type_class':'Residential_apartment', 'a_use':50
        },
        {
            "name":"adj_2",
            "orientation_zone": {"azimuth": 180},
            "area_facade_elements": np.array([20,60,30,30,50,50], dtype=object),
            "typology_elements": np.array(['OP','OP','OP','OP','GR','OP'], dtype=object),
            "transmittance_U_elements": np.array([0.82,0.82,0.82,0.82,0.52,1.16], dtype=object),
            "orientation_elements": np.array(['NV','SV','EV','WV','HOR','HOR'], dtype=object),
            'volume': 300, 'building_type_class':'Residential_apartment', 'a_use':50
        }
    ],
    "building_surface": [
        {
            "name": "Roof surface", "type": "opaque", "area": 130,
            "sky_view_factor": 1.0, "u_value": 2.2, "solar_absorptance": 0.4,
            "thermal_capacity": 741500.0, "orientation": {"azimuth": 0, "tilt": 0},
            "name_adj_zone": None
        },
        {
            "name": "Opaque north surface", "type": "opaque", "area": 30,
            "sky_view_factor": 0.5, "u_value": 1.4, "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0, "orientation": {"azimuth": 0, "tilt": 90},
            "name_adj_zone": "adj_1"
        },
        {
            "name": "Opaque south surface", "type": "opaque", "area": 30,
            "sky_view_factor": 0.5, "u_value": 1.4, "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0, "orientation": {"azimuth": 180, "tilt": 90},
            "name_adj_zone": "adj_2"
        },
        {
            "name": "Opaque east surface", "type": "opaque", "area": 30,
            "sky_view_factor": 0.5, "u_value": 1.2, "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0, "orientation": {"azimuth": 90, "tilt": 90},
            "name_adj_zone": None
        },
        {
            "name": "Opaque west surface", "type": "opaque", "area": 30,
            "sky_view_factor": 0.5, "u_value": 1.2, "solar_absorptance": 0.7,
            "thermal_capacity": 1416240.0, "orientation": {"azimuth": 270, "tilt": 90},
            "name_adj_zone": None
        },
        {
            "name": "Slab to ground", "type": "opaque", "area": 100,
            "sky_view_factor": 0.0, "u_value": 1.6, "solar_absorptance": 0.6,
            "thermal_capacity": 405801, "orientation": {"azimuth": 0, "tilt": 0},
            "name_adj_zone": None
        },
        {
            "name": "Transparent east surface", "type": "transparent", "area": 4,
            "sky_view_factor": 0.5, "u_value": 5, "g_value": 0.726,
            "height": 2, "width": 1, "parapet": 1.1,
            "orientation": {"azimuth": 90, "tilt": 90}, "shading": False,
            "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
            "name_adj_zone": None
        },
        {
            "name": "Transparent west surface", "type": "transparent", "area": 4,
            "sky_view_factor": 0.5, "u_value": 5, "g_value": 0.726,
            "height": 2, "width": 1, "parapet": 1.1,
            "orientation": {"azimuth": 270, "tilt": 90}, "shading": False,
            "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
            "name_adj_zone": None
        }
    ],
    "units": {
        "area": "m²", "u_value": "W/m²K", "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)", "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m²", "internal_gain_profile": "Normalized to 0-1", "HVAC_profile": "0: off, 1: on"
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0, "heating_setback": 17.0,
            "cooling_setpoint": 26.0, "cooling_setback": 30.0, "units": "°C"
        },
        "system_capacities": {"heating_capacity": 1.0e7, "cooling_capacity": 1.2e7, "units": "W"},
        "airflow_rates": {"infiltration_rate": 1.0, "units": "ACH"},
        "internal_gains": [
            {"name":"occupants","full_load":4.2,"weekday":[1]*24,"weekend":[1]*24},
            {"name":"appliances","full_load":3.0,"weekday":[0.6]*24,"weekend":[0.6]*24},
            {"name":"lighting","full_load":3.0,"weekday":[0.2]*24,"weekend":[0.2]*24}
        ],
        "construction": {"wall_thickness": 0.3, "thermal_bridges": 2, "units": "m / W/mK"},
        "climate_parameters": {"coldest_month": 1, "units": "1-12"},
        "heating_profile": {
            "weekday": [0]*5 + [1]*17 + [0]*2, "weekend": [0]*5 + [1]*17 + [0]*2
        },
        "cooling_profile": {
            "weekday": [0]*5 + [1]*17 + [0]*2, "weekend": [0]*7 + [1]*14 + [0]*3
        },
        "ventilation_profile": {
            "weekday": [0]*5 + [1]*17 + [0]*2, "weekend": [0]*7 + [1]*14 + [0]*3
        }
    }
}

INPUT_SYSTEM_HVAC = {
    # ---- emitter ----
    'emitter_type': 'Floor heating 1',
    'nominal_power': 8,
    'emission_efficiency': 90,
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
    'selected_emm_cont_circuit': 0,
    'mixing_valve': True,
    # 'TB14': custom_TB14,
    # 'heat_emission_data': pd.DataFrame({...}),
    'mixing_valve_delta': 2,
    # 'constant_flow_temp': 42,

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
    'mixing_valve_delta': 2,

    # Optional explicit generator setpoints
    # 'θHW_gen_flw_set': 50,
    # 'θHW_gen_ret_set': 40,

    # Efficiency model
    'efficiency_model': 'simple',

    # Calculation options
    'calc_when_QH_positive_only': False,
    'off_compute_mode': 'full',
}

# ================================================================================================
#                                  Example usage (build & validate)
# ================================================================================================

if __name__ == "__main__":
    building = build_building_from_BUI(BUI, INPUT_SYSTEM_HVAC)
    errs = building.inputs_validation()
    if errs:
        print("[QUALITY CHECK]", *errs, sep="\n - ")
    else:
        print("Inputs look OK.")
    if building.hvac:
        print("HVAC attached:", building.hvac.emitter_type, "Nominal power:", building.hvac.nominal_power, "kW")
