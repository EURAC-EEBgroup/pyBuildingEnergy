# __author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
# __credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
# __license__ = "MIT"
# __version__ = "0.1"
# __maintainer__ = "Daniele Antonucci"

"""

# Acutal limitation
Vedere test area edificio e area del solaio controterra differenti 
# Italy:
- tasso di ventilazione fissato a 0.3  h-1
- considerato solo edifici tipo zona climatica media (E)

"""

import numpy as np
from src.pybuildingenergy.source.functions import (
    Perimeter_from_area,
    Area_roof,
    Check_area,
)
import pickle

# ================================================================================================
#                           COMPONENTS ARCHETYPE
# ================================================================================================

# @Italy
# WALL
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

# ROOF
code_roof = ["roof01", "roof02"]
description_roof = [
    "Pitched roof with wood structure and planking",
    "Pitched roof with brick-concrete slab",
]
thickness_roof = [0.34, 0.34]
heat_capacity_roof = [278856, 390606]
U_roof = [1.8, 2.2]
R_roof = [1 / value for value in U_roof]

# FLOOR
code_floor = ["floor01", "floor02"]
description_floor = [
    "Concrete floor on soil",
    "floor with reinforced brick-concreate slab, low insulation",
]
thickness_floor = [0.27, 0.34]
heat_capacity_floor = [463800, 448050]
U_floor = [2.0, 0.98]
R_floor = [1 / value for value in U_floor]

# WINDOW
code_window = ["window01", "window02"]
description_window = [
    "Single glass, methal frame without thermal break",
    "single glasss wood frame",
]
U_window = [5.7, 4.9]
R_window = [1 / value for value in U_window]
g_window = [0.85, 0.85]

# ========================================================================================================================
#                                       INPUTS: SINGLE FAMILY HOUSE
# ========================================================================================================================

periods = [
    "before 1900",
    "1901-1920",
    "1921-1945",
    "1946-1960",
    "1961-1875",
    "1976-1990",
    "1991-2005",
    "2006-today",
]
bui_types = ["single_family_house"] * len(periods)
area = [139, 115, 116, 162, 156, 199, 172, 174]
window_area = [17.4, 14.4, 14.5, 20.3, 19.5, 24.9, 21.5, 21.8]  # 1/8 della superificie
volume = [533, 448, 455, 583, 679, 725, 605, 607]
coldest_month = [1] * len(periods)
S_V = [0.77, 0.82, 0.81, 0.75, 0.73, 0.72, 0.73, 0.72]
S_envelope = [S * volume for S, volume in zip(S_V, volume)]
number_of_floor = [2, 2, 2, 2, 2, 2, 2, 2]
height = [
    round(volume_i / (area_i / number_of_floor_i), 2)
    for volume_i, area_i, number_of_floor_i in zip(volume, area, number_of_floor)
]
bui_height = [x / (y / z) for x, y, z in zip(volume, area, number_of_floor)]
base = [
    (value / number_of_floor) / 10
    for value, number_of_floor in zip(area, number_of_floor)
]
perimeter = [Perimeter_from_area(value, 10 / 2) for value in area]
area_north = [round(10 * heights, 2) for heights in bui_height]
area_south = area_north
area_west = [round(bases * heights, 2) for bases, heights in zip(base, bui_height)]
area_east = area_west
area_roof = [round(Area_roof(10, leng_roof) / 2, 2) for leng_roof in base]
thermal_bridge_heat = [10] * len(periods)
w_code = [
    "wall01",
    "wall01",
    "wall02",
    "wall02",
    "wall03",
    "wall01",
    "wall01",
    "wall01",
]
r_code = [
    "roof01",
    "roof01",
    "roof01",
    "roof02",
    "roof02",
    "roof01",
    "roof01",
    "roof01",
]
win_code = [
    "window01",
    "window01",
    "window02",
    "window02",
    "window02",
    "window02",
    "window02",
    "window02",
]
f_code = [
    "floor01",
    "floor01",
    "floor01",
    "floor01",
    "floor01",
    "floor01",
    "floor01",
    "floor01",
]
building_category_const = ["old", "old", "old", "old", "old", "old", "medium", "medium"]
air_change_rate_base_value = [0.08, 0.14, 0.14, 0.1, 0.1, 0.1, 0.1, 0.1]
# GLOBAL INPUTS
typology_elements = np.array(
    ["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"], dtype=object
)
orientation_elements = np.array(
    ["NV", "SV", "EV", "WV", "HOR", "HOR", "NV", "SV", "EV", "WV"], dtype=object
)
solar_abs_elements = np.array(
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.6, 0.6, 0.6, 0.6], dtype=object
)
heat_convective_elements_internal = np.array(
    [2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object
)
heat_radiative_elements_internal = np.array(
    [5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13], dtype=object
)
heat_convective_elements_external = np.array(
    [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=object
)
heat_radiative_elements_external = np.array(
    [4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14], dtype=object
)
sky_factor_elements = np.array(
    [0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object
)
baseline_hci = np.array(
    [2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object
)
baseline_hce = np.array(
    [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=object
)

profile_residential_1 = {
    "code": "profile01",
    "type": "residential",
    "profile_workdays_internal_gains": np.array(
        [
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
            0.0,
            0.0,
            1.0,
            1.0,
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
        ]
    ),
    "profile_weekend_internal_gains": np.array(
        [
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
            0.0,
            0.0,
            1.0,
            1.0,
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
        ]
    ),
    "profile_workdays_ventilation": np.array(
        [
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
            0.0,
            0.0,
            1.0,
            1.0,
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
        ]
    ),
    "profile_weekend_ventilation": np.array(
        [
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
            0.0,
            0.0,
            1.0,
            1.0,
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
        ]
    ),
}

heating_installed = [True] * len(periods)
cooling_installed = [False] * len(periods)

# ========================================================================================================================
#                                       GENERATE BUILDING ARCHETYPE
# ========================================================================================================================


class Buildings(object):
    def __init__(
        self,
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
        heating_mode=bool,  # True or False if heating syste is
        cooling_mode=bool,
        heating_setpoint=float,  # Heating set-point in °C
        cooling_setpoint=float,  # Cooling set-point in °
        heating_setback=float,  # Heating set-back in °C
        cooling_setback=float,  # Cooling set-back in °C
        power_heating_max=float,  # Max Power of the heating system
        power_cooling_max=float,  # Max Power of the cooling system
        air_change_rate_base_value=float,  # air change rate
        air_change_rate_extra=float,
        internal_gains_wd=np.array,
        internal_gains_we=np.array,
        thermal_bridge_heat=float,  # value of thermal bridges
        thermal_resistance_floor=float,
        area_elements=list,  # Area of each facade elements
        transmittance_U_elements=list,
        thermal_resistance_R_elements=list,
        thermal_capacity_elements=list,
        g_factor_windows=list,
        heating_profile_wd=np.array,
        heating_profile_we=np.array,
        cooling_profile_wd=np.array,
        cooling_profile_we=np.array,
        ventilation_profile_wd=np.array,
        ventilation_profile_we=np.array,
        azimuth_relative_to_true_north=float,  # Azimuth angle between true north and the main building surface (long side or entrance), measured clockwise in degrees. N = 0, E = 90, S = 180, W = 270
        construction_class="class_i",
        building_type="",
        periods="",
        **kwargs,
    ):

        self.building_type = building_type
        self.periods = periods
        self.latitude = latitude
        self.longitude = longitude
        self.annual_mean_internal_temperature = kwargs.get(
            "annual_mean_internal_temperature"
        )
        self.annual_mean_external_temperature = kwargs.get(
            "annual_mean_external_temperature"
        )
        self.amplitude_of_internal_temperature_variations = kwargs.get(
            "amplitude_of_internal_temperature_variations"
        )
        self.a_use = area
        self.exposed_perimeter = exposed_perimeter
        self.height = height
        self.number_of_floor = number_of_floor
        self.volume = volume  # Initialize attribute to None
        self.slab_on_ground = slab_on_ground
        self.wall_thickness = wall_thickness
        self.coldest_month = coldest_month
        self.surface_envelope = (
            surface_envelope  # calculated from sum of surfaces + floor  + roof
        )
        self.surface_envelope_model = (
            surface_envelope_model  # calculated from S/V coefficient
        )
        self.base = side  # side of rectangular shape of building, a side of 10 meters is hypothesized
        self.heating_mode = heating_mode
        self.cooling_mode = cooling_mode
        self.heating_setpoint = heating_setpoint
        self.heating_setback = heating_setback
        self.cooling_setpoint = cooling_setpoint
        self.cooling_setback = cooling_setback
        self.power_cooling_max = -power_cooling_max
        self.power_heating_max = power_heating_max
        self.air_change_rate_base_value = air_change_rate_base_value
        self.air_change_rate_extra = air_change_rate_extra
        self.internal_gains_wd = internal_gains_wd
        self.internal_gains_we = internal_gains_we
        self.thermal_bridge_heat = thermal_bridge_heat
        self.thermal_resistance_floor = thermal_resistance_floor
        self.typology_elements = typology_elements
        self.orientation_elements = orientation_elements
        self.solar_abs_elements = solar_abs_elements
        self.area_elements = area_elements
        self.transmittance_U_elements = transmittance_U_elements
        self.thermal_resistance_R_elements = thermal_resistance_R_elements
        self.thermal_capacity_elements = thermal_capacity_elements
        self.g_factor_windows = g_factor_windows
        self.heat_convective_elements_internal = heat_convective_elements_internal
        self.heat_radiative_elements_internal = heat_radiative_elements_internal
        self.heat_convective_elements_external = heat_convective_elements_external
        self.heat_radiative_elements_external = heat_radiative_elements_external
        self.sky_factor_elements = sky_factor_elements
        self.heating_profile_wd = heating_profile_wd
        self.heating_profile_we = heating_profile_we
        self.cooling_profile_wd = cooling_profile_wd
        self.cooling_profile_we = cooling_profile_we
        self.ventilation_profile_wd = ventilation_profile_wd
        self.ventilation_profile_we = ventilation_profile_we
        self.azimuth_relative_to_true_north = azimuth_relative_to_true_north
        self.baseline_hci = baseline_hci
        self.baseline_hce = baseline_hce
        self.construction_class = construction_class
        self.weather_source = kwargs.get("weather_source")
        # self.tmy_filename = kwargs.get("tmy_filename")
        # self.location = kwargs.get("location")

    @property
    def slab_on_ground(self):
        return self._slab_on_ground

    @slab_on_ground.setter
    def slab_on_ground(self, value):
        """
        Calculation of the slab on ground area
        Aslab = Area/number_of_floor [m2]
        :param value: slab on ground area [m2]
        :return Area of slab on ground
        """
        if value is None:
            self._slab_on_ground = self.a_use / self.number_of_floor
        else:
            self._slab_on_ground = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        """
        Calucalte Volume, if not provided
        :param value: building volume
        :retrun Volume
        """
        if value is None:
            self._volume = (self.a_use / self.number_of_floor) * self.height
        else:
            self._volume = value

    # Area of building elements can not be 0. Se
    @property
    def area_elements(self):
        return self._area_elements

    @area_elements.setter
    def area_elements(self, values):
        """
        Quality check of area value
        :param values:List of areas for each individual facade element
        """
        self._area_elements = Check_area(values)

    def update_values(self, new_values: dict) -> object:
        """
        Update characteristic of a building providing a dictionary with new values

        :param new_values: new values of a building feature
        :return building object updated

        .. note::
            new_inputs = {
                'latitude':46.66345144066082,
                'longitude':9.71636944229362,
                'Eln':10, #
                'a_use': 100,
                "slab_on_ground":100,#
                'heating_setpoint':22,
                'cooling_setpoint':24,
                'power_heating_max':40000,
                'power_cooling_max':-10000
            }
        """
        for key, value in new_values.items():
            if not hasattr(self, key):
                pass
            if isinstance(value, list):
                if isinstance(getattr(self, key), list):
                    if len(value) != len(getattr(self, key)):
                        raise ValueError(
                            f"The length of '{key}' must match the original length"
                        )
                    setattr(self, key, value)
                else:
                    raise ValueError(
                        f"The length of '{key}' must match the original length"
                    )
            else:
                if isinstance(getattr(self, key), list):
                    raise ValueError(
                        f"The length of '{key}' must match the original length"
                    )
                setattr(self, key, value)

    def inputs_validation(self):
        """
        Validate inputs according to define rules and provide list of possible errors
        Rules:

            * Perimeter should be lower than area of building. Limitation building higher than 16m2
            * Transmittance values too hight or too low
            * Area of the floor slab on gorund should be lower than the area of the roof
        """
        # Check Volume
        if self.volume is None:
            self.volume = (self.a_use / self.number_of_floor) * self.height

        #
        quality_check_errors = []
        # 1. Check perimeter and area
        if self.a_use >= 16:
            if self.exposed_perimeter > self.a_use:
                quality_check_errors.append(
                    "Possible error. Check the value of perimeter and area if they are correct."
                )

        # 2. Check value of envelope transmittance
        for i, u_value in enumerate(self.transmittance_U_elements):
            element = self.typology_elements[i]
            if element == "OP":
                nameElement = "Opaque Element"
            elif element == "W":
                nameElement = "Transaprent Element"
            elif element == "HOR":
                nameElement = "Floor or Roof"

            orient_elment = self.orientation_elements[i]

            if u_value > 8 or u_value <= 0.1:
                quality_check_errors.append(
                    f"Possible error. Transmittance of the element {nameElement} oriented to {orient_elment} too low or too hight"
                )

        # 3. Check area roof and floor slab on ground
        area_roof = self.area_elements[5]
        area_floor = self.area_elements[6]
        if area_floor == area_roof:
            quality_check_errors.append(
                f"Warning!. The area of the floor slab on ground is higher than the area of the roof"
            )

        print(quality_check_errors)
        return quality_check_errors


# ===================================================================================================
#                           GET INPUTS FROM SPECIFIC ARCHETYPE
# ===================================================================================================
class Buildings_from_dictionary(object):
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def update_values(self, new_values: dict) -> object:
        """
        Update characteristic of a building providing a dictionary with new values

        :param new_values: dictioanry with new values of a building feature
        :return building object updated

        .. note::
            Example
            new_inputs = {
                'latitude':46.66345144066082,
                'longitude':9.71636944229362,
                'Eln':10,
                'a_use': 100,
                "slab_on_ground":100,
                'heating_setpoint':22,
                'cooling_setpoint':24,
                'power_heating_max':40000,
                'power_cooling_max':-10000
            }
        """
        for key, value in new_values.items():
            if not hasattr(self, key):
                pass
            if isinstance(value, list):
                if isinstance(getattr(self, key), list):
                    if len(value) != len(getattr(self, key)):
                        raise ValueError(
                            f"The length of '{key}' must match the original length"
                        )
                    setattr(self, key, value)
                else:
                    raise ValueError(
                        f"The length of '{key}' must match the original length"
                    )
            else:
                if isinstance(getattr(self, key), list):
                    raise ValueError(
                        f"The length of '{key}' must match the original length"
                    )
                setattr(self, key, value)

    def update_facade_elements(self, new_values):
        """
        Update facade elements

        :param new_values: new_values from a dictionary

        """
        for key, value in new_values.items():
            if not hasattr(self, key):
                pass
            setattr(self, key, value)

    def inputs_validation(self):
        """
        QUALITY CHECK

        Validate inputs according to the following rules and provide list of possible errors

        Rules:

            * Perimeter should be lower than area of building. Limitation building higher than 16m2
            * Transmittance values too hight or too low
            * Check area of the wall should be higher than the area of the window for the same orientation
            * Area of the floor slab on gorund should be lower than the area of the roof

        """
        quality_check_errors = []
        # 1. Check perimeter and area
        if self.a_use >= 16:
            if self.exposed_perimeter > self.a_use:
                quality_check_errors.append(
                    "Possible error. Check the value of perimeter and area if they are correct."
                )

        # 2. Check value of envelope transmittance
        for i, u_value in enumerate(self.transmittance_U_elements):
            element = self.typology_elements[i]
            if element == "OP":
                nameElement = "Opaque Element"
            elif element == "W":
                nameElement = "Transaprent Element"
            elif element == "HOR":
                nameElement = "Floor or Roof"

            orient_elment = self.orientation_elements[i]

            if u_value > 8 or u_value <= 0.1:
                quality_check_errors.append(
                    f"Possible error. Transmittance of the element {nameElement} oriented to {orient_elment} too low or too hight"
                )

        # 3. Check area roof and floor slab on ground
        area_roof = self.area_elements[5]
        area_floor = self.area_elements[6]
        if area_floor == area_roof:
            quality_check_errors.append(
                f"Warning!. The area of the floor slab on ground is higher than the area of the roof"
            )

        print(quality_check_errors)
        return quality_check_errors


class Selected_bui_archetype:
    def __init__(self, bui_archetype, period, latitude, longitude):
        self.bui_archetype = bui_archetype
        self.period = period
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        print(bui_archetype, period, latitude, longitude)

        if bui_archetype in bui_types:
            self.bui_archetype = bui_archetype
        else:
            raise ValueError(
                f"Invalid choice for archetype. Possible choices are: {', '.join(bui_types)}"
            )

        if period in periods:
            self.built_year = period
        else:
            raise ValueError(
                f"Invalid choice for possible periods. Possible choices are: {', '.join(periods)}"
            )

        if isinstance(latitude, float):
            self.latitude = latitude
        else:
            raise ValueError("latitude should be a float")

        if isinstance(longitude, float):
            self.longitude = longitude
        else:
            raise ValueError("longitude should be a float")

    def get_archetype(self, pickle_file_path):
        """
        Get archetype from a list of possible archetypes defined in the ``archetypes.pickle``
        :param pickle_file_path: path of the archetypes pickle file
        :return building archetype (*type*: obj)
        """
        # Read data from the pickle file
        with open(pickle_file_path, "rb") as f:
            archetypes = pickle.load(f)
        # Filter according to inputs
        selected_archetype = [
            bui
            for bui in archetypes
            if bui["building_type"] == self.bui_archetype
            and bui["periods"] == self.period
        ][0]
        selected_archetype["latitude"] = self.latitude
        selected_archetype["longitude"] = self.longitude

        return Buildings_from_dictionary(selected_archetype)
