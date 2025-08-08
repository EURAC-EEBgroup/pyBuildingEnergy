# __author__ = "Daniele Antonucci, Ulrich Filippi Oberegger, Olga Somova"
# __credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberegger", "Olga Somova"]
# __license__ = "MIT"
# __version__ = "0.1"
# __maintainer__ = "Daniele Antonucci"

import requests
import pandas as pd
import datetime as dt
from timezonefinder import TimezoneFinder
from pytz import timezone
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from pvlib.iotools import epw
from src.pybuildingenergy.source.functions import (
    Equation_of_time,
    Hour_angle_calc,
    Air_mass_calc,
    Get_positions,
    Filter_list_by_indices,
)


@dataclass
class WeatherDataResult:
    elevation: float
    weather_data: pd.DataFrame
    utc_offset: int
    latitude: float
    longitude: float


@dataclass
class Solar_irradiance:
    """
    Hourly solar irradiance
    Output of `Solar_irradiance_calculation`
    """

    solar_irradiance: pd.DataFrame


@dataclass
class simdf_52010:
    sim_df: pd.DataFrame


@dataclass
class numb_nodes_facade_elements:
    Rn: int
    Pln: np.array
    PlnSum: np.array


@dataclass
class conduttance_elements:
    h_pli_eli: np.array


@dataclass
class solar_absorption_elements:
    a_sol_pli_eli: np.array


@dataclass
class aeral_heat_capacity:
    kappa_pli_eli: np.array


@dataclass
class simulation_df:
    simulation_df: pd.DataFrame


@dataclass
class temp_ground:
    R_gr_ve: float
    Theta_gr_ve: np.array
    thermal_bridge_heat: float


@dataclass
class h_vent_and_int_gains:
    H_ve: pd.Series
    Phi_int: pd.Series
    sim_df_update: pd.DataFrame


# ===============================================================================================
#                                       MODULES SIMULATIONS
# ===============================================================================================

#                                       ISO 52010
# ===============================================================================================


class ISO52010:
    solar_constant = 1370  # [W/m2]
    K_eps = 1.104  # [rad^-3]
    Perez_coefficients_matrix = np.array(
        [
            [1.065, -0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
            [1.230, 0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
            [1.500, 0.330, 0.487, -0.221, 0.055, -0.064, -0.026],
            [1.950, 0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
            [2.800, 0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
            [4.500, 1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
            [6.200, 1.060, -1.600, -0.359, 0.264, -1.127, 0.131],
            [99999, 0.678, -0.327, -0.250, 0.156, -1.377, 0.251],
        ]
    )  # # Values for clearness index and brightness coefficients as function of clearness parameter

    def __init__(self):
        pass

    # GET WEATHER DATA FROM .epw FILE
    @classmethod
    def get_tmy_data_epw(cls, path_weather_file):
        """
        Get Weather data from epw file

        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

        :return:
            * *elevation*: altitude of specifici location (type: **float**)
            * *weather_data*: dataframe with weather parameters (e.g. outdoor temperature, outdoor relative humidity, etc.) (type: **pd.DataFrame**)
            * *utc_offset*: refers to the difference in time between Coordinated Universal Time (UTC) and the local time of a specific location (type: **int**)
            * *latitude*: latitude of the building place (type: **float**)
            * *longitude*: longitude of the building place (type: **float**)
        """

        # Read EPW file
        weather_data = epw.read_epw(path_weather_file)

        # Weather data filter in a format to be used by ISO52016 in a csv
        df_weather_time_series = weather_data[0]
        tmy_weather_data = df_weather_time_series.loc[
            :,
            [
                "temp_air",
                "relative_humidity",
                "ghi",
                "dni",
                "dhi",
                "ghi_infrared",
                "wind_speed",
                "wind_direction",
                "atmospheric_pressure",
            ],
        ]
        tmy_weather_data.index.name = "time(UTC)"

        # Convert DatetimeIndex to the desired format
        tmy_weather_data.index = df_weather_time_series.index.tz_convert(
            None
        )  # Remove timezone information
        tmy_weather_data.index = df_weather_time_series.index.strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Format datetime index
        tmy_weather_data.index = pd.DatetimeIndex(tmy_weather_data.index)
        #
        tmy_weather_data.columns = [
            "T2m",
            "RH",
            "G(h)",
            "Gb(n)",
            "Gd(h)",
            "IR(h)",
            "WS10m",
            "WD10m",
            "SP",
        ]

        #
        location_info = weather_data[1]
        elevation = location_info["altitude"]
        utcoffset_in_hours = int(location_info["TZ"])
        latitude_ = location_info["latitude"]
        longitude_ = location_info["longitude"]

        return WeatherDataResult(
            elevation=elevation,
            weather_data=tmy_weather_data,
            utc_offset=utcoffset_in_hours,
            latitude=latitude_,
            longitude=longitude_,
        )

    # GET DATA FROM PVGIS
    @classmethod
    def get_tmy_data_pvgis(cls, building_object) -> WeatherDataResult:
        """
        Get Weather data from pvgis API

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        :return:
            * *elevation*: altitude of specifici location (type: **float**)
            * *weather_data*: dataframe with wetaher parameters (e.g. outdoor temperature, outdoor relative humidity, etc.) (type: **pd.DataFrame**)
            * *utc_offset*: refers to the difference in time between Coordinated Universal Time (UTC) and the local time of a specific location (type: **int**)
            * *latitude*: latitude of the building place (type: **float**)
            * *longitude*: longitude of the building place (type: **float**)

        .. note::
            In case only weather data is desired, the ``building_object`` can only have the **latitude** and **longitude** parameters.

        """

        # Connection to PVGIS API to get weather data
        if isinstance(building_object, dict):
            latitude = building_object["building"]["latitude"]
            longitude = building_object["building"]["longitude"]
        else:
            latitude = building_object.__getattribute__("latitude")
            longitude = building_object.__getattribute__("longitude")
        url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={latitude}&lon={longitude}&outputformat=json&browser=1"
        response = requests.request("GET", url, allow_redirects=True)
        data = response.json()
        df_weather = pd.DataFrame(data["outputs"]["tmy_hourly"])

        # Time data into UTC
        df_weather["time(UTC)"] = [
            dt.datetime.strptime(x, "%Y%m%d:%H%M") for x in df_weather["time(UTC)"]
        ]

        # Change year to 2019 before sorting by date, because the months in the tmy file are stitched together from different years
        df_weather["time(UTC)"] = df_weather["time(UTC)"].apply(
            lambda x: x.replace(year=2019)
        )

        # Order data in date ascending order
        df_weather = df_weather.sort_values(by="time(UTC)")
        df_weather.index = df_weather["time(UTC)"]
        del df_weather["time(UTC)"]

        # Elevation is not needed for the energy demand calculation, only for the PV optimization
        loc_elevation = data["inputs"]["location"]["elevation"]
        latitude_ = data["inputs"]["location"]["latitude"]
        longitude_ = data["inputs"]["location"]["longitude"]
        # TIMEZONE FINDER
        tf = TimezoneFinder()
        utcoffset_in_hours = int(
            timezone(tf.timezone_at(lng=longitude, lat=latitude))
            .localize(df_weather.index[0])
            .utcoffset()
            .total_seconds()
            / 3600.0
        )

        return WeatherDataResult(
            elevation=loc_elevation,
            weather_data=df_weather,
            utc_offset=utcoffset_in_hours,
            latitude=latitude_,
            longitude=longitude_,
        )

    @classmethod
    def Solar_irradiance_calculation(
        cls,
        latitude_deg,
        longitude_deg,
        timezone,
        beta_ic_deg,
        gamma_ic_deg,
        DHI,
        DNI,
        ground_solar_reflectivity,
        calendar,
        n_timesteps,
        n_days,
    ):
        """
        ISO 52010-1:2017 specifies a calculation procedure for the conversion of climatic data for energy calculations.
        The main element in ISO 52010-1:2017 is the calculation of solar irradiance on a surface with arbitrary orientation and tilt.
        A simple method for conversion of solar irradiance to illuminance is also provided.
        The solar irradiance and illuminance on an arbitrary surface are applicable as input for energy and daylighting calculations, for building elements
        (such as roofs, facades and windows) and for components of technical building systems (such as thermal solar collectors, PV panels)

        :param timezone_utc: The UTC offset (or time offset) is an amount of time subtracted from or added to Coordinated Universal Time (UTC) time to specify the local solar time. (type: **int**)
        :param beta_ic_deg: Tilt angle of inclined surface from horizontal is measured upwards facing (hor.facing upw=0, vert=90, hor.facing down=180). (type: **int**)
        :param gamma_ic_deg: Orientation angle of the inclined (ic) surface, expressed as the geographical azimuth angle of the horizontal projection of the inclined (S=0, E=pos., W=neg.) surface normal. (type: **int**)
        :param DHI: Diffuse horizontal irradiance - [W/m2] (type: **float**)
        :param DNI: Direct (beam) irradiance - [W/m2] (type: **float**)
        :param ground_solar_reflectivity: solar reflectivity of the ground (type: **float**)
        :param calendar: dataframe with days of the year (from 1 to 365) and hour of day (1 to 24) (type: **int**)
        :param n_timesteps: number of hour in a year = 8760 (type: **int**)

        :return: **solar_irradiance**: hourly solar irradiance (type: **pd.DataFrame**)
        """

        beta_ic = np.radians(beta_ic_deg)
        gamma_ic = np.radians(gamma_ic_deg)
        DayWeekJan1 = 1  # doy of week of 1 January; 1=Monday, 7=Sunday
        latitude = np.radians(latitude_deg)
        earth_orbit_deviation_deg = 360 / n_days * calendar["day of year"]  # [deg]
        earth_orbit_deviation = np.radians(earth_orbit_deviation_deg)  # [rad]
        declination_deg = (
            0.33281
            - 22.984 * np.cos(earth_orbit_deviation)
            - 0.3499 * np.cos(2 * earth_orbit_deviation)
            - 0.1398 * np.cos(3 * earth_orbit_deviation)
            + 3.7872 * np.sin(earth_orbit_deviation)
            + 0.03205 * np.sin(2 * earth_orbit_deviation)
            + 0.07187 * np.sin(3 * earth_orbit_deviation)
        )  # [deg]
        declination = np.radians(declination_deg)
        t_eq = Equation_of_time(calendar["day of year"])  # [min]
        time_shift = timezone - (longitude_deg / 15)  # [h]
        solar_time = calendar["hour of day"] - t_eq / 60 - time_shift  # [h]
        hour_angle_deg = Hour_angle_calc(solar_time)  # [deg]
        hour_angle = np.radians(hour_angle_deg)
        solar_altitude_angle_sin = np.sin(declination) * np.sin(latitude) + np.cos(
            declination
        ) * np.cos(latitude) * np.cos(hour_angle)
        solar_altitude_angle = np.arcsin(
            np.sin(declination) * np.sin(latitude)
            + np.cos(declination) * np.cos(latitude) * np.cos(hour_angle)
        )
        solar_altitude_angle[solar_altitude_angle < 1e-4] = 0
        solar_incidence_angle_ic_cos = (
            np.sin(declination) * np.sin(latitude) * np.cos(beta_ic)
            - np.sin(declination)
            * np.cos(latitude)
            * np.sin(beta_ic)
            * np.cos(gamma_ic)
            + np.cos(declination)
            * np.cos(latitude)
            * np.cos(beta_ic)
            * np.cos(hour_angle)
            + np.cos(declination)
            * np.sin(latitude)
            * np.sin(beta_ic)
            * np.cos(gamma_ic)
            * np.cos(hour_angle)
            + np.cos(declination)
            * np.sin(beta_ic)
            * np.sin(gamma_ic)
            * np.sin(hour_angle)
        )  # ic=inclined surface
        solar_incidence_angle_ic = np.arccos(solar_incidence_angle_ic_cos)
        air_mass = Air_mass_calc(solar_altitude_angle)  # [-]
        solar_constant = 1370  # [W/m2]
        I_ext = solar_constant * (
            1 + 0.033 * np.cos(earth_orbit_deviation)
        )  # extra-terrestrial radiation [W/m2]
        solar_zenith_angle = np.pi / 2 - solar_altitude_angle
        solar_azimuth_angle_aux_1_sin = (
            np.cos(declination) * np.sin(np.pi - hour_angle)
        ) / np.cos(np.arcsin(solar_altitude_angle_sin))
        solar_azimuth_angle_aux_1_cos = (
            np.cos(latitude) * np.sin(declination)
            + np.sin(latitude) * np.cos(declination) * np.cos(np.pi - hour_angle)
        ) / np.cos(np.arcsin(solar_altitude_angle_sin))
        solar_azimuth_angle_aux_2 = np.arcsin(
            np.cos(declination)
            * np.sin(np.pi - hour_angle)
            / np.cos(np.arcsin(solar_altitude_angle_sin))
        )
        solar_azimuth_angle = -(np.pi + solar_azimuth_angle_aux_2)
        mask = (solar_azimuth_angle_aux_1_cos > 0) & (
            solar_azimuth_angle_aux_1_sin >= 0
        )
        solar_azimuth_angle[mask] = np.pi - solar_azimuth_angle_aux_2[mask]
        mask = solar_azimuth_angle_aux_1_cos < 0
        solar_azimuth_angle[mask] = solar_azimuth_angle_aux_2[mask]
        a_perez = pd.Series(np.zeros(n_timesteps), index=calendar.index)
        mask = solar_incidence_angle_ic_cos > 0
        a_perez[mask] = solar_incidence_angle_ic_cos[mask]
        b_perez = np.maximum(
            np.cos(np.radians(85)) * np.ones(n_timesteps), np.cos(solar_zenith_angle)
        )
        clearness = pd.Series(999 * np.ones(n_timesteps), index=calendar.index)  # [-]
        K_eps = 1.104  # [rad^-3]
        mask = (
            DHI > 0
        )  # DHI=diffuse horizontal irradiance; DNI=direct (beam) normal irradiance; GHI=global horizontal irradiance
        clearness[mask] = (
            (DHI[mask] + DNI[mask]) / DHI[mask]
            + K_eps * np.float_power(solar_altitude_angle[mask], 3)
        ) / (1 + K_eps * np.float_power(solar_altitude_angle[mask], 3))
        sky_brightness = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [-]
        sky_brightness[mask] = air_mass[mask] * DHI[mask] / I_ext[mask]
        Perez_coefficients_matrix = np.array(
            [
                [1.065, -0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
                [1.230, 0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
                [1.500, 0.330, 0.487, -0.221, 0.055, -0.064, -0.026],
                [1.950, 0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
                [2.800, 0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
                [4.500, 1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
                [6.200, 1.060, -1.600, -0.359, 0.264, -1.127, 0.131],
                [99999, 0.678, -0.327, -0.250, 0.156, -1.377, 0.251],
            ]
        )  # see Excel implementation of ISO 52010-1; columns in this order: epsilon, f11, f12, f13, f21, f22, f23
        PerezF = np.zeros(
            (n_timesteps, 6)
        )  # columns in the order: F11, F12, F13, F21, F22, F23

        mask_DHI = DHI > 0
        for i in range(6):
            mask_clearness = clearness < Perez_coefficients_matrix[0, 0]
            mask = mask_DHI & mask_clearness
            PerezF[mask, i] = Perez_coefficients_matrix[0, i + 1]
            for j in range(6):
                mask_clearness = (clearness >= Perez_coefficients_matrix[j, 0]) & (
                    clearness < Perez_coefficients_matrix[j + 1, 0]
                )
                mask = mask_DHI & mask_clearness
                PerezF[mask, i] = Perez_coefficients_matrix[j + 1, i + 1]
            mask_clearness = clearness >= Perez_coefficients_matrix[6, 0]
            mask = mask_DHI & mask_clearness
            PerezF[mask, i] = Perez_coefficients_matrix[7, i + 1]

        PerezF1 = np.maximum(
            0,
            PerezF[:, 0]
            + PerezF[:, 1] * sky_brightness
            + PerezF[:, 2] * solar_zenith_angle,
        )
        PerezF2 = (
            PerezF[:, 3]
            + PerezF[:, 4] * sky_brightness
            + PerezF[:, 5] * solar_zenith_angle
        )
        I_dir = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [W/m2]
        mask = solar_incidence_angle_ic_cos > 0
        I_dir[mask] = DNI[mask] * solar_incidence_angle_ic_cos[mask]
        I_dif = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [W/m2]
        mask = DHI > 0
        I_dif[mask] = DHI[mask] * (
            (1 - PerezF1[mask]) * (1 + np.cos(beta_ic)) / 2
            + PerezF1[mask] * a_perez[mask] / b_perez[mask]
            + PerezF2[mask] * np.sin(beta_ic)
        )
        I_dif_ground = (
            (DHI + DNI * np.sin(solar_altitude_angle))
            * ground_solar_reflectivity
            * (1 - np.cos(beta_ic))
            / 2
        )  # [W/m2]
        I_circum = DHI * PerezF1 * a_perez / b_perez  # [W/m2]
        I_dif_tot = I_dif - I_circum + I_dif_ground
        I_dir_tot = I_dir + I_circum
        I_tot = I_dif_tot + I_dir_tot

        return Solar_irradiance(solar_irradiance=I_tot)


def Calculation_ISO_52010(
    building_object, path_weather_file, weather_source="pvgis"
) -> simdf_52010:
    """
    Calculation procedure for the conversion of climatic data for energy calculation.
    The main element in ISO 52010-1:2017 is the calculation of solar irradiance on a surface with arbitrary orientation and tilt


    :param building_object:  Building object create according to the method ``Building``or ``Buildings_from_dictionary``
    :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

    :return: **sim_df**: climatic data for energy simulation (type: **pd.DataFrame**)
    """

    # Get weather dataframe
    if weather_source == "pvgis":
        weatherData = ISO52010.get_tmy_data_pvgis(building_object)
    elif weather_source == "epw":
        weatherData = ISO52010.get_tmy_data_epw(path_weather_file)
    else:
        raise ValueError("select the right weather source: 'epw' or 'pvgis'")

    sim_df = weatherData.weather_data
    timezoneW = weatherData.utc_offset

    # Change time index
    if len(sim_df) > 8760:  # In the case of a leap year, (ITA: anno bisestile)
        pass
    else:
        sim_df.index = pd.to_datetime(
            {
                "year": 2009,
                "month": sim_df.index.month,
                "day": sim_df.index.day,
                "hour": sim_df.index.hour,
            }
        )

    for column in sim_df:
        sim_df[column] = np.roll(sim_df[column], timezoneW)
    sim_df.rename_axis(index={"time(UTC)": "time(local)"}, inplace=True)
    sim_df["day of year"] = sim_df.index.dayofyear
    sim_df["hour of day"] = sim_df.index.hour + 1  # 1 to 24
    or_tilt_azim_dic = {
        "HOR": (0, 0),
        "SV": (90, 0),
        "EV": (90, 90),
        "NV": (90, 180),
        "WV": (90, -90),
    }  # dictionary mapping orientation in orientation_elements with (beta_ic_deg=elevation/tilt, gamma_ic_deg=azimuth), see util.util.ISO52010_calc()

    if len(sim_df) > 8760:
        n_tsteps = 8784
        n_days_year = 366
    else:
        n_tsteps = 8760
        n_days_year = 365

    if isinstance(building_object, dict):
        orientation_elements = ["EV", "HOR", "SV", "NV", "WV"]
    else:
        orientation_elements = building_object.__getattribute__("orientation_elements")

    for orientation in set(orientation_elements):

        if isinstance(building_object, dict):
            azimuth_relative_to_true_north = building_object["building"][
                "azimuth_relative_to_true_north"
            ]
        else:
            azimuth_relative_to_true_north = (
                building_object.azimuth_relative_to_true_north
            )

        gamma_ic_deg = (  # Transformation from azimuth relative to true north (convention for building orientation) to angle for solar irradiance calculation (convention for ISO 52010)
            or_tilt_azim_dic[orientation][1] - azimuth_relative_to_true_north
        )

        sim_df[orientation] = ISO52010.Solar_irradiance_calculation(
            n_timesteps=n_tsteps,
            n_days=n_days_year,
            latitude_deg=weatherData.latitude,
            longitude_deg=weatherData.longitude,
            timezone=timezoneW,
            beta_ic_deg=or_tilt_azim_dic[orientation][0],
            gamma_ic_deg=gamma_ic_deg,
            DHI=sim_df["Gd(h)"],
            DNI=sim_df["Gb(n)"],
            ground_solar_reflectivity=0.2,
            calendar=sim_df[["day of year", "hour of day"]],
        ).solar_irradiance

    sim_df = pd.concat(
        [sim_df[sim_df.index.month == 12], sim_df]
    )  # weather_data augmented by warmup period consisting of December month copied at the beginning

    return simdf_52010(sim_df=sim_df)


#                                       ISO 52016
# ===============================================================================================


class ISO52016:

    or_tilt_azim_dic = {
        "HOR": (0, 0),
        "SV": (90, 0),
        "EV": (90, 90),
        "NV": (90, 180),
        "WV": (90, -90),
    }  # dictionary mapping orientation in orientation_elements with (beta_ic_deg=elevation/tilt, gamma_ic_deg=azimuth), see util.util.ISO52010_calc()

    # __slots__=("inputs","sim_df")

    def __init__(self):
        pass

        """
        Simulate the Final Energy consumption of the building using the ISO52016
        """

    @classmethod
    def Number_of_nodes_element(cls, building_object) -> numb_nodes_facade_elements:
        """
        Calculation of the number of nodes for each element.
        If OPACQUE, or ADIABATIC -> n_nodes = 5
        If TRANSPARENT-> n_nodes = 2

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``
        :return:
            * *Rn*: value of last node to be used in the definition of the element vector
            * *Pln*: inizial number of nodes according to the type of element (5 - opaque element, 2 - transparent element)
            * *PlnSum*: sequential number of nodes based on the list of opaque and transparent elements
        """
        # Number of envelop building elements
        if isinstance(building_object, dict):
            el_list = len(building_object["building_surface"])
        else:
            el_list = len(building_object.__getattribute__("typology_elements"))
        # Initialize Pln with all elements as 5
        Pln = np.full(el_list, 5)
        # Replace elements with value 2 where type is "W"
        if isinstance(building_object, dict):
            for i, surf in enumerate(building_object["building_surface"]):
                if surf["type"] == "transparent":
                    Pln[i] = 2
        else:
            Pln[building_object.__getattribute__("typology_elements") == "W"] = 2
        # Calculation fo number of nodes for each building element (wall, roof, window)
        PlnSum = np.array([0] * el_list)
        for Eli in range(1, el_list):
            PlnSum[Eli] = (
                PlnSum[Eli - 1] + Pln[Eli - 1]
            )  # Index of matrix , each row is a node

        Rn = (
            PlnSum[-1] + Pln[-1] + 1
        )  # value of last node to be used in the definition of the vector

        return numb_nodes_facade_elements(Rn, Pln, PlnSum)

    @classmethod
    def Conduttance_node_of_element(
        cls, building_object, lambda_gr=2.0
    ) -> conduttance_elements:
        """
        Calculation of the conduttance between node "pli" adn node "pli-1", as determined per type of construction
        element in 6.5.7 in W/m2K

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param lambda_gr: hermal conductivity of ground [W/(m K)]. Default=2

        .. note:: Required parameters of building_object:

            * type: type of building element OPAQUE -'OP', TRANSPARENT - 'W', GROUND -'GR'
            * res: theraml ressistance of opaque building element
            * kappa_m: heat_capacity of the element in Table B.14
            * solar_absorption_coeff: solar absorption coefficient of element provided by user or using values of Table A.15 and B.15 of ISO 52016
            * area: area of each element [m2]
            * ori_tilt: orientation and tilt values
            * g_value: onyl for window

        :return: h_pli_eli: conduttance coefficient between nodes (W/m2K). *type*: np.array
        """
        R_gr = 0.5 / lambda_gr  # thermal resistance of 0.5 m of ground [m2 K/W]
        # Number of envelop building elements
        el_type = building_object.__getattribute__("typology_elements")
        # Initialization of conduttance coefficient calcualation
        h_pli_eli = np.zeros((4, len(el_type)))

        U_eli = building_object.__getattribute__("transmittance_U_elements")
        R_c_eli = building_object.__getattribute__("thermal_resistance_R_elements")
        h_ci_eli = building_object.__getattribute__("heat_convective_elements_internal")
        h_ri_eli = building_object.__getattribute__("heat_radiative_elements_internal")
        h_ce_eli = building_object.__getattribute__("heat_convective_elements_external")
        h_re_eli = building_object.__getattribute__("heat_radiative_elements_external")

        for i in range(0, len(el_type)):
            if R_c_eli[i] == 0.0:
                R_c_eli[i] = (
                    1 / U_eli[i]
                    - 1 / (h_ci_eli[i] + h_ri_eli[i])
                    - 1 / (h_ce_eli[i] + h_re_eli[i])
                )

        # layer = 1
        layer_no = 0
        for i in range(len(el_type)):
            if (
                building_object.__getattribute__("thermal_resistance_R_elements")[i]
                != 0
            ):
                if el_type[i] == "OP":
                    # h_pli_eli[0, i] = 6 / BUI.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[0, i] = 6 / R_c_eli[i]
                elif el_type[i] == "W":
                    # h_pli_eli[0, i] = 1 / BUI.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[0, i] = 1 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[0, i] = 2 / R_gr

        # layer = 2
        layer_no = 1
        for i in range(len(el_type)):
            if (
                building_object.__getattribute__("thermal_resistance_R_elements")[i]
                != 0
            ):
                if el_type[i] == "OP":
                    # h_pli_eli[layer_no, i] = 3 / building_object.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[layer_no, i] = 3 / R_c_eli[i]
                elif el_type[i] == "GR":
                    # h_pli_eli[layer_no, i] = 1 / (building_object.__getattribute__('thermal_resistance_R_elements')[i] / 4 + R_gr / 2)
                    h_pli_eli[layer_no, i] = 1 / (R_c_eli[i] / 4 + R_gr / 2)

        # layer = 3
        layer_no = 2
        for i in range(len(el_type)):
            if (
                building_object.__getattribute__("thermal_resistance_R_elements")[i]
                != 0
            ):
                if el_type[i] == "OP":
                    # h_pli_eli[layer_no, i] = 3 / building_object.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[layer_no, i] = 3 / R_c_eli[i]
                elif el_type[i] == "GR":
                    # h_pli_eli[layer_no, i] = 2 / building_object.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[layer_no, i] = 2 / R_c_eli[i]

        # layer = 4
        layer_no = 3
        for i in range(len(el_type)):
            if (
                building_object.__getattribute__("thermal_resistance_R_elements")[i]
                != 0
            ):
                if el_type[i] == "OP":
                    # h_pli_eli[layer_no, i] = 6 / building_object.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[layer_no, i] = 6 / R_c_eli[i]
                elif el_type[i] == "GR":
                    # h_pli_eli[layer_no, i] = 4 / building_object.__getattribute__('thermal_resistance_R_elements')[i]
                    h_pli_eli[layer_no, i] = 4 / R_c_eli[i]

        return conduttance_elements(h_pli_eli=h_pli_eli)

    @classmethod
    def Solar_absorption_of_element(cls, building_object) -> solar_absorption_elements:
        """
        Calculation of solar absorption for each single elements

        :param type: list of elements type.
        :param a_sol: coefficients of solar absorption for each elements

        :return: a_sol_pli_eli: solar absorption of each single nodes (type: *np.array*)

        .. note::
            EXAMPLE:

            inputs = {
                "type": ["GR", "OP", "OP", "OP", "OP", "OP", "W", "W", "W", "W"],
                "a_sol": [0, 0.6, 0.6, 0.6, 0.6, 0.6, 0, 0, 0, 0],
            }

            a_sol_pli_eli = array(
                [
                    [0. , 0.6, 0.6, 0.6, 0.6, 0.6, 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]
                ]
            )

        """
        # Number of envelop building elements
        el_list = len(building_object.__getattribute__("typology_elements"))
        # Coefficient list of elements
        solar_abs_elements = building_object.__getattribute__("solar_abs_elements")

        # Initialization of solar_abs_coeff
        a_sol_pli_eli = np.zeros((5, el_list))
        a_sol_pli_eli[0, :] = solar_abs_elements

        return solar_absorption_elements(a_sol_pli_eli=a_sol_pli_eli)

    @classmethod
    def Areal_heat_capacity_of_element(cls, building_object) -> aeral_heat_capacity:
        """
        Calculation of the areal heat capacity of the node "pli" and node "pli-1" as
        determined per type of construction element [J/m2K] - 6.5.7 ISO 52016

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        .. note:: Required parameters of building_object:

            * type: type of building element OPAQUE -'OP', TRANSPARENT - 'W', GROUND -'GR'
            * kappa_m: heat_capacity of the element in Table B.14
            * construction_class: lass of construction with respect to the distribution of the mass in the construction
                        Table B.13. Possible choice: class_i, class_e, class_ie, class_d, class_m

        :return: aeral_heat_capacity: aeral heat capacity of each facade element (type: *np.array*)
        """

        # Number of envelop building elements
        el_type = building_object.__getattribute__("typology_elements")
        # List of heat capacyit of building envelope elements
        list_kappa_el = building_object.__getattribute__("thermal_capacity_elements")
        # Initialization of heat capacity of nodes
        kappa_pli_eli_ = np.zeros((5, len(el_type)))

        #
        if (
            building_object.__getattribute__("construction_class") == "class_i"
        ):  # Mass concetrated at internal side
            # OPAQUE: kpl5 = km_eli ; kpl1=kpl2=kpl3=kpl4=0
            # GROUND: kpl5 = km_eli ; kpl3=kpl4=0
            node = 1
            for i in range(len(el_type)):
                if building_object.__getattribute__("typology_elements")[i] == "GR":
                    kappa_pli_eli_[node, i] = 1e6  # heat capacity of the ground

            node = 4
            for i in range(len(el_type)):
                if building_object.__getattribute__("typology_elements")[i] == "OP":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                elif building_object.__getattribute__("typology_elements")[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        elif (
            building_object.__getattribute__("construction_class") == "class_e"
        ):  # mass concentrated at external side
            # OPAQUE: kpl1 = km_eli ; kpl2=kpl3=kpl4=kpl5=0
            # GROUND: kpl3 = km_eli ; kpl4=kpl5=0
            node = 0
            for i in range(len(el_type)):
                if building_object.__getattribute__("typology_elements")[i] == "OP":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                elif building_object.__getattribute__("typology_elements")[i] == "GR":
                    node = 2
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        elif (
            building_object.__getattribute__("construction_class") == "class_ie"
        ):  # mass divided over internal and external side)
            # OPAQUE: kpl1 = kpl5 = km_eli/2 ; kpl2=kpl3=kpl4=0
            # GROUND: kpl1 = kp5 =km_eli/2; kpl4=0
            node = 0
            for i in range(len(el_type)):
                if building_object.__getattribute__("typology_elements")[i] == "OP":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
                elif building_object.__getattribute__("typology_elements")[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
            node = 4
            for i in range(len(el_type)):
                if building_object.__getattribute__("typology_elements")[i] == "OP":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
                elif building_object.__getattribute__("typology_elements")[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2

        elif (
            building_object.__getattribute__("construction_class") == "class_d"
        ):  # (mass equally distributed)
            # OPAQUE: kpl2=kpl3=kpl4=km_eli/4
            # GROUND: kpl3=km_eli/4; kpl4=km_eli/2
            node_list_1 = [1, 2, 3]
            for node in node_list_1:
                for i in range(len(el_type)):
                    if building_object.__getattribute__("typology_elements")[i] == "OP":
                        kappa_pli_eli_[node, i] = list_kappa_el[i] / 4
                    if building_object.__getattribute__("typology_elements")[i] == "GR":
                        if node == 2:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 4
                        if node == 3:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 2

            # OPAQUE kpl1=kpl5= km_eli/8
            # GROUND:kpl5=km_eli/4
            node_list_2 = [0, 4]
            for node in node_list_2:
                for i in range(len(el_type)):
                    if building_object.__getattribute__("typology_elements")[i] == "OP":
                        kappa_pli_eli_[node, i] = list_kappa_el[i] / 8
                    if building_object.__getattribute__("typology_elements")[i] == "GR":
                        if node == 4:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 4

        elif (
            building_object.__getattribute__("construction_class") == "class_m"
        ):  # mass concentrated inside
            # OPAQUE: kpl1=kpl2=kpl4=kpl5=0; kpl3= km_eli
            # GROUND: kpl4=km_eli; kpl3=kpl5=0
            node = 2
            for i in range(len(el_type)):
                if building_object.__getattribute__("typology_elements")[i] == "OP":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                if building_object.__getattribute__("typology_elements")[i] == "GR":
                    node = 3
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        return aeral_heat_capacity(kappa_pli_eli=kappa_pli_eli_)

    @classmethod
    def Temp_calculation_of_ground(
        cls, building_object, lambda_gr=2.0, R_si=0.17, R_se=0.04, psi_k=0.05, **kwargs
    ) -> temp_ground:
        """
        Virtual ground temperature calculation of ground according to ISO 13370-1:2017
        for salb-on-ground (sog) floor

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param lambda_gr: hermal conductivity of ground [W/(m K)]. Default=2
        :param R_se: external surface resistance (for ground floor calculation). Default 0.04
        :param R_si: internal surface resistance (for ground floor calculation). Default 0.17
        :param psi_k: linear thermal transmittance associated with wall/floor junction [W/(m K)]. Default 0.05

        .. note:: Required parameters of building_object:

            * heating: TRUE or FALSE. Is there a heating system?
            * cooling: TRUE or FALSE. Is there a cooling system?
            * heating_setpoint: setpoint for the heating system (default 20°C)
            * cooling_setpoint: setpoint for cooling system (default 26°C)
            * latitude_deg: latitude of location in degrees
            * slab_on_ground_area: area of the building in contact with the ground
            * perimeter: perimeter of the building [m]
            * wall_thickness: thickness of the wall [m]
            * thermal_resistance_floor: resitance of the floor [m2K/W]
            * thermal_bridge_heat: Thermal bridges heat transfer coefficient - sum of thermal bridges (clause 6.6.5.3)
            * coldest_month: coldest month, if not provided automatically selected according to the hemisphere

        :return:
            * **R_gr_ve**: Thermal resistance of virtual layer (floor_slab)
            * **thermal_bridge_heat**: Heat transfer coefficient of overall thermal briges
            * **Theta_gr_ve**: Internal Temperature of the ground

        .. caution::
            The calculation is only for buildings with a ground floor slab. That is, in direct contact with the ground and not for unheated rooms.
            To be integrated: code related to different types of contacts based on the presence of an unheated room or other.

        .. note::
            Calculation of annual_mean_internal_temperature and its amplitude variations
            if heating and colling are selected:

                * the annual mean internal temperature is the average between Heating and Cooling setpoints
                * the amplitude variations is the mean of the difference between Heating and Cooling setpoints

            if not heating and cooling the value should be provided by the user:

                * if the user doesn't provide any value, the following values are used:
                    * annual_mean_internal_temperature = 23 <- ((26 (standard C set point) + 20 (standard H setpoint))/2)
                    * amplitude_of_internal_temperature_variations = 3 <- (26-20)/2

        .. note::
            Defintion of the coldest month accoriding to the position.
            If the user doesn't provide a value between 1 (January) and 12 (Decemebr)
            the default values: 1 for northern hemisphere or 7 in southern hemisphere are used

        """

        # ============================
        #
        R_gr = 0.5 / lambda_gr  # thermal resistance of 0.5 m of ground [m2 K/W]

        # ============================
        # GET MIN, MAX AND MEAN of External temperature values at monthly(M) resolution
        path_weather_file_ = kwargs.get("path_weather_file")
        sim_df = Calculation_ISO_52010(building_object, path_weather_file_).sim_df

        try:
            external_temperature_monthly_averages = sim_df["T2m"].resample("ME").mean()
            external_temperature_monthly_minima = sim_df["T2m"].resample("ME").min()
            external_temperature_monthly_maxima = sim_df["T2m"].resample("ME").max()
        except:
            external_temperature_monthly_averages = sim_df["T2m"].resample("M").mean()
            external_temperature_monthly_minima = sim_df["T2m"].resample("M").min()
            external_temperature_monthly_maxima = sim_df["T2m"].resample("M").max()

        # amplitude of external temperature variations
        amplitude_of_external_temperature_variations = (
            external_temperature_monthly_maxima - external_temperature_monthly_minima
        ).mean() / 2
        # annual mean of external temperature
        if not isinstance(building_object, dict):
            if (
                hasattr(building_object, "annual_mean_external_temperature")
                and building_object.__getattribute__("annual_mean_external_temperature")
                is not None
            ):
                # Use provided annual mean external temperature if available
                annual_mean_external_temperature = building_object.__getattribute__(
                    "annual_mean_external_temperature"
                )
        else:
            # Use monthly average external temperature as fallback
            annual_mean_external_temperature = (
                external_temperature_monthly_averages.mean()
            )

        # ============================

        # ============================
        if building_object.__getattribute__(
            "heating_mode"
        ) and building_object.__getattribute__("cooling_mode"):
            if (
                building_object.__getattribute__("heating_setpoint") is not None
                and building_object.__getattribute__("cooling_setpoint") is not None
            ):
                # Calculate annual mean internal temperature and amplitude of internal temperature variations
                annual_mean_internal_temperature = (
                    building_object.__getattribute__("heating_setpoint")
                    + building_object.__getattribute__("cooling_setpoint")
                ) / 2  # [°C]
                amplitude_of_internal_temperature_variations = (
                    building_object.__getattribute__("cooling_setpoint")
                    - building_object.__getattribute__("heating_setpoint")
                ) / 2  # [K]
        else:
            if (
                hasattr(building_object, "annual_mean_internal_temperature")
                and building_object.__getattribute__("annual_mean_internal_temperature")
                is not None
            ):
                # Use provided annual mean internal temperature if available
                annual_mean_internal_temperature = building_object.__getattribute__(
                    "annual_mean_internal_temperature"
                )
                # User can provide amplitude_of_internal_temperature_variations
                if (
                    hasattr(
                        building_object, "amplitude_of_internal_temperature_variations"
                    )
                    and building_object.__getattribute__(
                        "amplitude_of_internal_temperature_variations"
                    )
                    is not None
                ):
                    amplitude_of_internal_temperature_variations = (
                        building_object.__getattribute__(
                            "amplitude_of_internal_temperature_variations"
                        )
                    )
                else:
                    amplitude_of_internal_temperature_variations = 3
            else:
                # Use default or expert input if user-provided data is not available
                annual_mean_internal_temperature = (
                    23  # Default estimate or expert input, user input
                )
                amplitude_of_internal_temperature_variations = 3
        # ============================

        # ============================
        if not building_object.__getattribute__("coldest_month"):
            if building_object.__getattribute__("latitude") >= 0:
                building_object.__setattr__("coldest_month", 1)
                # building_object.coldest_month = 1  # 1..12;
            else:
                building_object.__setattr__("coldest_month", 7)
                # building_object.coldest_month= 7

        internal_temperature_by_month = np.zeros(12)
        for month in range(12):
            internal_temperature_by_month[month] = (
                annual_mean_internal_temperature
                - amplitude_of_internal_temperature_variations
                * np.cos(
                    2
                    * np.pi
                    * (month + 1 - building_object.__getattribute__("coldest_month"))
                    / 12
                )
            )  # estimate
        # ============================

        # ============================
        """
        Area in contact with the ground. 
        If the value is nor provided by the user 
        """
        sog_area = building_object.__getattribute__("slab_on_ground")
        if sog_area == -999:
            sog_area = sum(
                Filter_list_by_indices(
                    building_object.__getattribute__("area"),
                    Get_positions(
                        building_object.__getattribute__("typology_elements"), "GR"
                    ),
                )
            )
        # ============================

        # ============================
        """
        Calcualtion of the perimeter.
        If the value is not provided by the user a rectangluar shape of the building is considered.
        The perimeter is calcuated according to the area of the south and east facade
        """
        if building_object.__getattribute__("exposed_perimeter") == None:
            # SOUTH FACADE
            south_facade_area = sum(
                Filter_list_by_indices(
                    building_object.__getattribute__("area"),
                    Get_positions(
                        building_object.__getattribute__("orientation_elements"), "SV"
                    ),
                )
            )
            # EAST FACADE
            east_facade_area = sum(
                Filter_list_by_indices(
                    building_object.__getattribute__("area"),
                    Get_positions(
                        building_object.__getattribute__("orientation_elements"), "EV"
                    ),
                )
            )
            #
            facade_height = np.sqrt(east_facade_area * south_facade_area / sog_area)
            sog_width = south_facade_area / facade_height
            sog_length = sog_area / sog_width
            exposed_perimeter = 2 * (sog_length + sog_width)
        else:
            exposed_perimeter = building_object.__getattribute__("exposed_perimeter")
        characteristic_floor_dimension = sog_area / (0.5 * exposed_perimeter)
        # ============================

        # ============================
        """
        Calculation of temperature of the ground using:
            1. the thermal Resistance (R) and Transmittance (U) of the floor
            2. External Temperature [°C]
        """
        if not building_object.__getattribute__("wall_thickness"):
            # building_object.wall_thickness = 0.35  # [m]
            building_object.__setattr__("wall_thickness", 0.35)

        if not building_object.thermal_resistance_floor:
            building_object.__setattr__("thermal_resistance_floor", 5.3)
            # building_object.thermal_resistance_floor = 5.3  # Floor construction thermal resistance (excluding effect of ground) [m2 K/W]

        # The thermal transmittance depends on the characteristic dimension of the floor, B' [see 8.1 and Equation (2)], and the total equivalent thickness, dt (see 8.2), defined by Equation (3):
        equivalent_ground_thickness = building_object.__getattribute__(
            "wall_thickness"
        ) + lambda_gr * (
            building_object.thermal_resistance_floor + R_se
        )  # [m]

        if (
            equivalent_ground_thickness < characteristic_floor_dimension
        ):  # uninsulated and moderately insulated floors
            U_sog = (
                2
                * lambda_gr
                / (np.pi * characteristic_floor_dimension + equivalent_ground_thickness)
                * np.log(
                    np.pi * characteristic_floor_dimension / equivalent_ground_thickness
                    + 1
                )
            )  # thermal transmittance of slab on ground including effect of ground [W/(m2 K)]
        else:  # well-insulated floors
            U_sog = lambda_gr / (
                0.457 * characteristic_floor_dimension + equivalent_ground_thickness
            )

        # calcualtion of thermal resistance of virtual layer
        R_gr_ve = (
            1 / U_sog
            - R_si
            - building_object.__getattribute__("thermal_resistance_floor")
            - R_gr
        )

        # Adding thermal bridges
        if not building_object.__getattribute__("thermal_bridge_heat"):
            # building_object.thermal_bridge_heat = exposed_perimeter * psi_k
            building_object.__setattr__(
                "thermal_bridge_heat", exposed_perimeter * psi_k
            )
        else:
            thermal_bridge = building_object.__getattribute__("thermal_bridge_heat")
            # building_object.thermal_bridge_heat += exposed_perimeter * psi_k
            building_object.__setattr__(
                "thermal_bridge_heat", thermal_bridge + (exposed_perimeter * psi_k)
            )

        # Calculation of steady-state  ground  heat  transfer  coefficients  are  related  to  the  ratio  of  equivalent  thickness
        # to  characteristic floor dimension, and the periodic heat transfer coefficients are related to the ratio
        # of equivalent thickness to periodic penetration depth
        steady_state_heat_transfer_coefficient = (
            sog_area * U_sog + exposed_perimeter * psi_k
        )  # [W/K]
        periodic_penetration_depth = 3.2  # [m]
        H_pi = (
            sog_area
            * lambda_gr
            / equivalent_ground_thickness
            * np.sqrt(
                2
                / (
                    np.float_power(
                        1 + periodic_penetration_depth / equivalent_ground_thickness, 2
                    )
                    + 1
                )
            )
        )  # periodic heat transfer coefficient related to internal temperature variations [W/K]
        H_pe = (
            0.37
            * exposed_perimeter
            * lambda_gr
            * np.log(periodic_penetration_depth / equivalent_ground_thickness + 1)
        )  # periodic heat transfer coefficient related to external temperature variations [W/K]
        annual_average_heat_flow_rate = steady_state_heat_transfer_coefficient * (
            annual_mean_internal_temperature - annual_mean_external_temperature
        )  # [W]
        periodic_heat_flow_due_to_internal_temperature_variation = np.zeros(12)
        a_tl = 0  # time lead of the heat flow cycle compared with that of the internal temperature [months]
        b_tl = 1  # time lag of the heat flow cycle compared with that of the external temperature [months]
        for month in range(12):
            periodic_heat_flow_due_to_internal_temperature_variation[month] = (
                -H_pi
                * amplitude_of_internal_temperature_variations
                * np.cos(
                    2
                    * np.pi
                    * (
                        month
                        + 1
                        - building_object.__getattribute__("coldest_month")
                        + a_tl
                    )
                    / 12
                )
            )
        periodic_heat_flow_due_to_external_temperature_variation = np.zeros(12)
        for month in range(12):
            periodic_heat_flow_due_to_external_temperature_variation[month] = (
                H_pe
                * amplitude_of_external_temperature_variations
                * np.cos(
                    2
                    * np.pi
                    * (
                        month
                        + 1
                        - building_object.__getattribute__("coldest_month")
                        - b_tl
                    )
                    / 12
                )
            )
        average_heat_flow_rate = (
            annual_average_heat_flow_rate
            + periodic_heat_flow_due_to_internal_temperature_variation
            + periodic_heat_flow_due_to_external_temperature_variation
        )
        Theta_gr_ve = internal_temperature_by_month - (
            average_heat_flow_rate
            - exposed_perimeter
            * psi_k
            * (annual_mean_internal_temperature - annual_mean_external_temperature)
        ) / (sog_area * U_sog)

        return temp_ground(
            R_gr_ve=R_gr_ve,
            Theta_gr_ve=Theta_gr_ve,
            thermal_bridge_heat=building_object.__getattribute__("thermal_bridge_heat"),
        )

    @classmethod
    def Occupancy_profile(
        cls, building_object, path_weather_file, weather_source="pvgis"
    ) -> simulation_df:
        """
        Definition of occupancy profile for:

            #. Internal gains
            #. temperature control and ventilation

        The data is divided in weekend and workday

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

        .. note:: Required parameters of building_object:

            * occ_level_wd: occupancy profile of workday for modification of internal gains
            * occ_level_we: occupancy profile of weekend for modification of internal gains
            * comf_level_wd: occupancy profile of workday for modification of ventilation
            * comf_level_we: occupancy profile of weekend for modification of ventilation
            * heating_setpoint: value of heating setpoint. e.g 20
            * heating_setback: value of heating setback. eg.10
            * cooling_setpoint: value of cooling setpoint. e.g 26
            * cooling_setback: value of cooling setback. eg.20

        :retrun: sim_df: dataframe with inputs for simulation having information of weather, occupancy, heating and cooling setpoint and setback
        """
        # WEATHER DATA
        if weather_source == "pvgis":
            sim_df = pd.DataFrame(
                Calculation_ISO_52010(
                    building_object, path_weather_file, weather_source=weather_source
                ).sim_df
            )
        elif weather_source == "epw":
            sim_df = pd.DataFrame(
                Calculation_ISO_52010(
                    building_object, path_weather_file, weather_source=weather_source
                ).sim_df
            )
        sim_df.index = pd.DatetimeIndex(sim_df.index)
        # number of days of simulation (13 months)
        number_of_days_with_warmup_period = len(sim_df) // 24
        # Inizailization occupancy for Internal Gain
        sim_df["occupancy level"] = (
            np.nan
        )  # 9504 are the numbers of hours in one year + December month for warmup period
        # Inizialization occupancy for Indoor Temperature and Ventilation control
        sim_df["comfort level"] = np.nan

        # Internal gains
        if isinstance(building_object, dict):
            internal_gains_wd = building_object["building_parameters"][
                "internal_gains_total"
            ]["weekday"]
            internal_gains_we = building_object["building_parameters"][
                "internal_gains_total"
            ]["weekend"]
            heating_profile_wd = building_object["building_parameters"][
                "heating_profile"
            ]["weekday"]
            heating_profile_we = building_object["building_parameters"][
                "heating_profile"
            ]["weekend"]
            cooling_profile_wd = building_object["building_parameters"][
                "cooling_profile"
            ]["weekday"]
            cooling_profile_we = building_object["building_parameters"][
                "cooling_profile"
            ]["weekend"]
            ventilation_profile_wd = building_object["building_parameters"][
                "ventilation_profile"
            ]["weekday"]
            ventilation_profile_we = building_object["building_parameters"][
                "ventilation_profile"
            ]["weekend"]
        else:
            internal_gains_wd = building_object.__getattribute__("internal_gains_wd")
            internal_gains_we = building_object.__getattribute__("internal_gains_we")
            # Heating control
            heating_profile_wd = building_object.__getattribute__("heating_profile_wd")
            heating_profile_we = building_object.__getattribute__("heating_profile_we")
            # Cooling control
            cooling_profile_wd = building_object.__getattribute__("cooling_profile_wd")
            cooling_profile_we = building_object.__getattribute__("cooling_profile_we")
            # Ventilation control
            ventilation_profile_wd = building_object.__getattribute__(
                "ventilation_profile_wd"
            )
            ventilation_profile_we = building_object.__getattribute__(
                "ventilation_profile_we"
            )

        """ WORKDAY """
        # Number of workdays during the entire simulation period
        wd_mask = sim_df.index.weekday < 5
        # number of workdays for the entire period of simulation (year + warmup: 13 months)
        number_of_weekdays_with_warmup_period = sum(wd_mask) // 24
        # Associate the occupancy profile to simulation hourly time of workdays
        sim_df.loc[wd_mask, "internal gains"] = np.tile(
            internal_gains_wd, number_of_weekdays_with_warmup_period
        )
        sim_df.loc[wd_mask, "heating control"] = np.tile(
            heating_profile_wd, number_of_weekdays_with_warmup_period
        )
        sim_df.loc[wd_mask, "cooling control"] = np.tile(
            cooling_profile_wd, number_of_weekdays_with_warmup_period
        )
        sim_df.loc[wd_mask, "ventilation control"] = np.tile(
            ventilation_profile_wd, number_of_weekdays_with_warmup_period
        )

        """ WEEKEND """
        # number of weekend days for the entire period of simulation (year + warmup: 13 months)
        number_of_weekend_days_with_warmup_period = (
            number_of_days_with_warmup_period - number_of_weekdays_with_warmup_period
        )
        # Number of workdays during the entire simulation period
        we_mask = sim_df.index.weekday >= 5
        # Associate the occupancy profile to simulation hourly time of weekends
        sim_df.loc[we_mask, "internal gains"] = np.tile(
            internal_gains_we, number_of_weekend_days_with_warmup_period
        )
        sim_df.loc[we_mask, "heating control"] = np.tile(
            heating_profile_we, number_of_weekend_days_with_warmup_period
        )
        sim_df.loc[we_mask, "cooling control"] = np.tile(
            cooling_profile_we, number_of_weekend_days_with_warmup_period
        )
        sim_df.loc[we_mask, "ventilation control"] = np.tile(
            ventilation_profile_we, number_of_weekend_days_with_warmup_period
        )

        """ HEATING AND COOLING """
        """ HEATING """
        # Associate setback and setpoint of heating to occupancy profile for comfort
        if isinstance(building_object, dict):
            sim_df["Heating"] = building_object["building_parameters"][
                "temperature_setpoints"
            ]["heating_setback"]
            sim_df.loc[sim_df["heating control"] > 0, "Heating"] = building_object[
                "building_parameters"
            ]["temperature_setpoints"]["heating_setpoint"]
        else:
            # Associate setback and setpoint of heating to occupancy profile for comfort
            sim_df["Heating"] = building_object.__getattribute__("heating_setback")
            sim_df.loc[sim_df["heating control"] > 0, "Heating"] = (
                building_object.__getattribute__("heating_setpoint")
            )

        """ COOLING """
        # Associate setback and setpoint of cooling to occupancy profile for comfort
        if isinstance(building_object, dict):
            sim_df["Cooling"] = building_object["building_parameters"][
                "temperature_setpoints"
            ]["cooling_setback"]
            sim_df.loc[sim_df["cooling control"] > 0, "Cooling"] = building_object[
                "building_parameters"
            ]["temperature_setpoints"]["cooling_setpoint"]
        else:
            sim_df["Cooling"] = building_object.__getattribute__("cooling_setback")
            sim_df.loc[sim_df["cooling control"] > 0, "Cooling"] = (
                building_object.__getattribute__("cooling_setpoint")
            )

        return simulation_df(simulation_df=sim_df)

    @classmethod
    def Vent_heat_transf_coef_and_Int_gains(
        cls,
        building_object,
        path_weather_file,
        c_air=1006,
        rho_air=1.204,
        weather_source="pvgis",
    ) -> h_vent_and_int_gains:
        """
        Calculation of heat transfer coefficient (section 8 - ISO 13789:2017 and 6.6.6 ISO 52016:2017 ) and internal gains

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))
        :param c_air: specific heat of air at constant pressure [J/(kg K)]. Default: 1006
        :param rho_air: ir density at 20 °C [kg/m3]. Default: 1.204

        .. note:: Required parameters of building_object:

            * air_change_rate_base_value: ventilation air change rate [m3/h]
            * air_change_rate_extra: extra iar change in case of comfort values [m3/h]
            * a_use: useful area of the building [m2]
            * internal_gains_base_value: value for internal gains [W/m2]
            * internal_gains_extra: eventual extra internal gains during occupied hours [W/m2]

        :return:
            * H_ve: heat transfer coefficient for ventilation [W/K]
            * Phi_int: internal gains [W]

        """
        # VENTILATION (CONDUTTANCE)
        sim_df = ISO52016.Occupancy_profile(
            building_object, path_weather_file, weather_source=weather_source
        ).simulation_df
        if isinstance(building_object, dict):
            for surf in building_object["building_surface"]:
                if (
                    surf["type"] == "opaque" and surf["sky_view_factor"] == 0
                ):  # Slab to ground
                    building_volume = (
                        surf["area"] * building_object["building"]["height"]
                    )
                    treated_floor_area = (
                        surf["area"] * building_object["building"]["n_floors"]
                    )
                    building_object["building"]["volume"] = building_volume
                    building_object["building"][
                        "treated_floor_area"
                    ] = treated_floor_area
                    break
            sim_df["air flow rate"] = (
                building_object["building_parameters"]["airflow_rates"][
                    "infiltration_rate"
                ]
                * building_volume
            )  # [m3/h]
            sim_df.loc[sim_df["ventilation control"] > 0, "air flow rate"] += (
                building_object["building_parameters"]["airflow_rates"][
                    "ventilation_rate_extra"
                ]
                * building_volume
            )
        else:
            sim_df["air flow rate"] = building_object.__getattribute__(
                "air_change_rate_base_value"
            ) * building_object.__getattribute__(
                "a_use"
            )  # [m3/h]
            sim_df.loc[
                sim_df["ventilation control"] > 0, "air flow rate"
            ] += building_object.__getattribute__(
                "air_change_rate_extra"
            ) * building_object.__getattribute__(
                "a_use"
            )

        air_flow_rate = sim_df["air flow rate"]
        H_ve = c_air * rho_air / 3600 * air_flow_rate  # [W/K]

        # INTERNAL GAINS
        if isinstance(building_object, dict):
            sim_df["internal gains"] *= treated_floor_area  # [W]
        else:
            sim_df["internal gains"] *= building_object.__getattribute__("a_use")  # [W]
        Phi_int = sim_df["internal gains"]

        return h_vent_and_int_gains(H_ve=H_ve, Phi_int=Phi_int, sim_df_update=sim_df)

    @classmethod
    def Temperature_and_Energy_needs_calculation(
        cls,
        building_object,
        nrHCmodes=2,
        c_int_per_A_us=10000,
        f_int_c=0.4,
        f_sol_c=0.1,
        f_H_c=1,
        f_C_c=1,
        delta_Theta_er=11,
        **kwargs,
    ):
        """
        Calcualation fo energy needs according to the equation (37) of ISO 52016:2017. Page 60.

        [Matrix A] x [Node temperature vector X] = [State vector B]

        where:
        Theta_int_air: internal air temperature [°C]
        Theta_op_act: Actual operative temperature [°C]

        :param building_object: Building object create according to the method ``Building`` or ``Buildings_from_dictionary``.
        :param nrHCmodes:  inizailization of system mode: 0 for Heating, 1 for Cooling, 2 for Heating and Cooling. Default: 2
        :param c_int_per_A_us: areal thermal capacity of air and furniture per thermally conditioned zone. Default: 10000
        :param f_int_c: convective fraction of the internal gains into the zone. Default: 0.4
        :param f_sol_c: convective fraction of the solar radiation into the zone. Default: 0.1
        :param f_H_c: convective fraction of the heating system per thermally conditioned zone (if system specific). Deafult: 1
        :param f_C_c: convective fraction of the cooling system per thermally conditioned zone (if system specific). Default: 1
        :param delta_Theta_er: Average difference between external air temperature and sky temperature. Default: 11

        .. note::
            INPUT:
            **sim_df*: dataframe with:

                * index: time of simulation on hourly resolution and timeindex typology (13 months on hourly resolution)
                * T2m: Exteranl temperarture [°C]
                * RH: External humidity [%]
                * G(h):
                * Gb(n):
                * Gd(h):
                * IR(h):
                * WS10m:
                * WD10m:
                * SP:
                * day of year:
                * hour of day:
                * HOR:
                * NV:
                * WV:
                * EV:
                * SV:
                * occupancy_level:
                * comfort_level:
                * Heating:
                * Cooling:
                * air_flow_rate:
                * internal_gains

            * **power_heating_max**: max power of the heating system (provided by the user) in W
            * **power_cooling_max**: max power of the cooling system (provided by the user) in W
            * **Rn**: ... result of function ``Number_of_nodes_element``
            * **Htb**: Heat transmission coefficient for Thermal bridges (provided by the user)
            * **H_ve**: ... result of function ``Ventilation_heat_transfer_coefficient``
            * **Phi_int**: ... result of function ``Internal_heat_gains``
            * **a_use**: building area [m2]
            * **Pln**: ... result of function ``Number_of_nodes_element``
            * **PlnSum**: ... result of function ``Number_of_nodes_element``
            * **a_sol_pli_eli**: ... result of function ``Solar_absorption_of_elment``
            * **kappa_pli_eli**: ... result of function  ``Areal_heat_capacity_of_element``
            * **heat_convective_elements_internal**: internal convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_convective_elements_external**: external convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_radiative_elements_external**: external radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_radiative_elements_internal**: internal radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **sky_factor_elements**: View factor between element and sky
            * **R_gr_ve**: ... result of function ``Temp_calculation_of_ground`` (Thermal Resitance of virtual layer)
            * **Theta_gr_ve**: ... result of function ``Temp_calculation_of_ground``
            * **h_pli_eli**: ... result of function ``Conduttance_node_of_element``

        """
        i = 1
        with tqdm(total=15) as pbar:

            pbar.set_postfix({"Info": f"Inizailization {i}"})

            # INIZIALIZATION
            if kwargs["weather_source"] == "pvgis":
                path_weather_file_ = None
            elif kwargs["weather_source"] == "epw":
                path_weather_file_ = (
                    kwargs["path_weather_file"]
                    if "path_weather_file" in kwargs
                    else None
                )
            int_gains_vent = ISO52016.Vent_heat_transf_coef_and_Int_gains(
                building_object,
                path_weather_file=path_weather_file_,
                weather_source=kwargs["weather_source"],
            )
            sim_df = int_gains_vent.sim_df_update
            Tstepn = len(sim_df)  # number of hours to perform the simulation

            # HEating and cooling Load
            Phi_HC_nd_calc = np.zeros(
                3
            )  # Load of Heating or Cooling needed to heat/cool the zone - calculated
            Phi_HC_nd_act = np.zeros(
                Tstepn
            )  # Load of Heating or Cooling needed to heat/cool the zone - actual

            # Temperature (indoor and operative)
            Theta_int_air = np.zeros((Tstepn, 3))
            Theta_int_r_mn = np.zeros((Tstepn, 3))  # <---
            Theta_int_op = np.zeros((Tstepn, 3))
            Theta_op_act = np.zeros(Tstepn)
            #
            pbar.update(1)

            # Time
            Dtime = 3600.0 * np.ones(Tstepn)
            #
            pbar.update(1)

            # Mode
            # nrHCmodes = 2 # inizailization of system modality (0 - heating, 1 - cooling or 2- both)
            colB_act = 0  # the vector B has 3 columns (1st column actual value, 2nd: maximum value reachable in heating, 3rd: maximum value reachbale in cooling)

            #
            pbar.update(1)
            # Number of building element
            if isinstance(building_object, dict):
                bui_eln = len(building_object["building_surface"])
            else:
                bui_eln = len(building_object.__getattribute__("typology_elements"))

            #
            pbar.update(1)
            # Element types and orientations
            if isinstance(building_object, dict):
                typology_elements = np.array(bui_eln * ["EXT"], dtype="object")
                for i, surf in enumerate(building_object["building_surface"]):
                    if surf["type"] == "opaque":
                        if surf["sky_view_factor"] == 0:
                            typology_elements[i] = "GR"
                        else:
                            if surf["adiabatic"]:
                                typology_elements[i] = "AD"
                            else:
                                typology_elements[i] = "OP"
                    elif surf["type"] == "transparent":
                        typology_elements[i] = "W"
            else:
                typology_elements = np.array(
                    building_object.__getattribute__("typology_elements")
                )
            Type_eli = bui_eln * ["EXT"]
            Type_eli[np.where(typology_elements == "GR")[0][0]] = "GR"
            #
            pbar.update(1)
            if isinstance(building_object, dict):
                g_values = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    if surf["type"] == "transparent":
                        g_values[i] = surf["g_value"]
                tau_sol_eli = g_values
            else:
                tau_sol_eli = np.array(
                    building_object.__getattribute__("g_factor_windows")
                )

            # Building area of elements
            if isinstance(building_object, dict):
                area_elements = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    area_elements[i] = surf["area"]
            else:
                area_elements = np.array(
                    building_object.__getattribute__("area_elements")
                )
            area_elements_tot = np.sum(area_elements)  # Sum of all areas
            pbar.update(1)

            # Orientation and tilt
            if isinstance(building_object, dict):
                orientation_elements = np.zeros(bui_eln, dtype=object)
                for i, surf in enumerate(building_object["building_surface"]):
                    azimuth = surf["orientation"]["azimuth"]
                    tilt = surf["orientation"]["tilt"]
                    if tilt == 0:
                        orientation_elements[i] = "HOR"
                    elif tilt == 90:
                        match azimuth:
                            case 0:
                                orientation_elements[i] = "NV"
                            case 90:
                                orientation_elements[i] = "EV"
                            case 180:
                                orientation_elements[i] = "SV"
                            case 270:
                                orientation_elements[i] = "WV"
            else:
                orientation_elements = np.array(
                    building_object.__getattribute__("orientation_elements")
                )
            pbar.update(1)

            # External temperature ... (to be checked)
            theta_sup = sim_df["T2m"]
            # Internal capacity
            if isinstance(building_object, dict):
                C_int = (
                    c_int_per_A_us * building_object["building"]["treated_floor_area"]
                )
            else:
                C_int = c_int_per_A_us * building_object.__getattribute__("a_use")
            pbar.update(1)

            # Heat transfer coefficient for each element area
            # Values according to ISO 13789
            if isinstance(building_object, dict):
                hci_facade = 2.5
                hci_ground = 0.7
                hci_roof = 5.0
                hci = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    if surf["sky_view_factor"] == 0:
                        hci[i] = hci_ground
                    elif surf["sky_view_factor"] == 1:
                        hci[i] = hci_roof
                    else:
                        hci[i] = hci_facade
                Ah_ci = np.dot(area_elements, hci)
            else:
                Ah_ci = np.dot(
                    area_elements,
                    building_object.__getattribute__(
                        "heat_convective_elements_internal"
                    ),
                )
            pbar.update(1)

            # mean internal radiative transfer coefficient
            if isinstance(building_object, dict):
                heat_radiative_elements_internal_mn = (
                    np.dot(area_elements, np.array(5.13 * np.ones(bui_eln)))
                    / area_elements_tot
                )
            else:
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        building_object.__getattribute__(
                            "heat_radiative_elements_internal"
                        ),
                    )
                    / area_elements_tot
                )
            pbar.update(1)

            # Initialization vectorB and temperature
            nodes = ISO52016.Number_of_nodes_element(building_object)
            Theta_old = 20 * np.ones(nodes.Rn)
            VecB = 20 * np.ones((nodes.Rn, 3))
            pbar.update(1)

            # Temperature ground and thermal bridges
            t_Th = ISO52016.Temp_calculation_of_ground(
                building_object, path_weather_file=path_weather_file_
            )
            #
            pbar.set_postfix({"Info": f"Calculating ground temperature"})
            pbar.update(1)
            h_pli_eli = ISO52016.Conduttance_node_of_element(building_object).h_pli_eli

            pbar.set_postfix({"Info": f"Calculating conduttance of elements"})
            pbar.update(1)
            kappa_pli_eli = (
                ISO52016().Areal_heat_capacity_of_element(building_object).kappa_pli_eli
            )

            pbar.set_postfix({"Info": f"Calculating aeral heat capacity of elements"})
            pbar.update(1)
            a_sol_pli_eli = (
                ISO52016().Solar_absorption_of_element(building_object).a_sol_pli_eli
            )

            pbar.set_postfix({"Info": f"Calculating solar absorption of element"})
            pbar.update(1)

        """
        CALCULATION OF SENSIBLE HEATING AND COOLING LOAD (following the procedure of poin 6.5.5.2 of UNI ISO 52016)
        For each hour and each zone the actual internal operative temperature θ and the actual int;ac;op;zt;t 6.5.5.2 Sensible heating and cooling load
        heating or cooling load, ΦHC;ld;ztc;t, is calculated using the following step-wise procedure: 
        """

        with tqdm(total=Tstepn) as pbar:
            for Tstepi in range(Tstepn):

                Theta_H_set = sim_df.iloc[Tstepi]["Heating"]
                Theta_C_set = sim_df.iloc[Tstepi]["Cooling"]
                Theta_old = VecB[:, colB_act]

                # firs step:
                # HEATING:
                # if there is no set point for heating (heating system not installed) -> heating power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_H_set < -995:  #
                    power_heating_max_act = 0
                else:
                    power_heating_max_act = building_object.__getattribute__(
                        "power_heating_max"
                    )  #

                # COOLING:
                # if there is no set point for heating (cooling system not installed) -> cooling power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_C_set > 995:
                    power_cooling_max_act = 0
                else:
                    power_cooling_max_act = building_object.__getattribute__(
                        "power_cooling_max"
                    )

                Phi_HC_nd_calc[0] = (
                    0  # the load has three values:  0 no heating e no cooling, 1  heating, 2 cooling
                )
                if power_heating_max_act == 0 and power_cooling_max_act == 0:  #
                    nrHCmodes = 1
                elif power_cooling_max_act == 0:
                    colB_H = 1
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_H] = power_heating_max_act
                elif power_heating_max_act == 0:
                    colB_C = 1
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_C] = power_cooling_max_act
                else:
                    nrHCmodes = 3
                    colB_H = 1
                    colB_C = 2
                    Phi_HC_nd_calc[colB_H] = power_heating_max_act
                    Phi_HC_nd_calc[colB_C] = power_cooling_max_act

                iterate = True
                while iterate:

                    iterate = False

                    VecB = np.zeros((nodes.Rn, 3))
                    MatA = np.zeros((nodes.Rn, nodes.Rn))

                    Phi_sol_zi = 0

                    for Eli in range(bui_eln):
                        if Type_eli[Eli] == "EXT":
                            # Solar gains for each elements, the sim_df['SV' or 'EV', etc.] is calculated based on the
                            # UNI 52010:
                            # Phi_sol_zi: solar gain [W]
                            # tu_sol_ei: g-value of windows
                            # sim_df[orientation_elements[Eli]].iloc[Tstepi]: UNI52010
                            Phi_sol_zi += (
                                tau_sol_eli[Eli]
                                * area_elements[Eli]
                                * sim_df[orientation_elements[Eli]].iloc[Tstepi]
                            )

                    ri = 0
                    # Energy balacne on zone level. Eq. (38) UNI 52016
                    # XTemp = Thermal capacity at specific time (t) and for  a specific degree °C [W] +
                    # + Ventilation loss (at time t)[W] + Transmission loss (at time t)[W] + intrnal gain[W] + solar gain [W]. Missed the
                    # the convective fraction of the heating/cooling system
                    XTemp = (
                        t_Th.thermal_bridge_heat * sim_df.iloc[Tstepi]["T2m"]
                        + int_gains_vent.H_ve.iloc[Tstepi] * theta_sup.iloc[Tstepi]
                        + f_int_c * int_gains_vent.Phi_int.iloc[Tstepi]
                        + f_sol_c * Phi_sol_zi
                        + (C_int / Dtime[Tstepi]) * Theta_old[ri]
                    )

                    # adding the convective fraction of the heating/cooling system according to the type of system available (heating, cooling and heating and cooling)
                    for cBi in range(nrHCmodes):
                        if Phi_HC_nd_calc[cBi] > 0:
                            f_HC_c = f_H_c
                        else:
                            f_HC_c = f_C_c
                        VecB[ri, cBi] += XTemp + f_HC_c * Phi_HC_nd_calc[cBi]

                    ci = 0

                    # First part of the equation on the square bracket(38)
                    MatA[ri, ci] += (
                        (C_int / Dtime[Tstepi])
                        + Ah_ci
                        + t_Th.thermal_bridge_heat
                        + int_gains_vent.H_ve.iloc[Tstepi]
                    )

                    for Eli in range(bui_eln):
                        Pli = nodes.Pln[Eli]
                        ci = nodes.PlnSum[Eli] + Pli
                        MatA[ri, ci] -= (
                            area_elements[Eli]
                            * building_object.__getattribute__(
                                "heat_convective_elements_internal"
                            )[Eli]
                        )

                    for Eli in range(bui_eln):
                        for Pli in range(nodes.Pln[Eli]):
                            ri += 1
                            XTemp = (
                                a_sol_pli_eli[Pli, Eli]
                                * sim_df[orientation_elements[Eli]].iloc[Tstepi]
                                + (kappa_pli_eli[Pli, Eli] / Dtime[Tstepi])
                                * Theta_old[ri]
                            )
                            for cBi in range(nrHCmodes):
                                VecB[ri, cBi] += XTemp
                            if Pli == (nodes.Pln[Eli] - 1):
                                XTemp = (1 - f_int_c) * int_gains_vent.Phi_int.iloc[
                                    Tstepi
                                ] + (1 - f_sol_c) * Phi_sol_zi
                                for cBi in range(nrHCmodes):
                                    if Phi_HC_nd_calc[cBi] > 0:
                                        f_HC_c = f_H_c
                                    else:
                                        f_HC_c = f_C_c
                                    VecB[ri, cBi] += (
                                        XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]
                                    ) / area_elements_tot
                            elif Pli == 0:
                                if Type_eli[Eli] == "EXT":
                                    XTemp = (
                                        building_object.__getattribute__(
                                            "heat_convective_elements_external"
                                        )[Eli]
                                        + building_object.__getattribute__(
                                            "heat_radiative_elements_external"
                                        )[Eli]
                                    ) * sim_df["T2m"].iloc[
                                        Tstepi
                                    ] - building_object.__getattribute__(
                                        "sky_factor_elements"
                                    )[
                                        Eli
                                    ] * building_object.__getattribute__(
                                        "heat_radiative_elements_external"
                                    )[
                                        Eli
                                    ] * delta_Theta_er
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp
                                elif Type_eli[Eli] == "GR":
                                    XTemp = (1 / t_Th.R_gr_ve) * t_Th.Theta_gr_ve[
                                        sim_df.index.month[Tstepi] - 1
                                    ]
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp

                            ci = 1 + nodes.PlnSum[Eli] + Pli
                            MatA[ri, ci] += kappa_pli_eli[Pli, Eli] / Dtime[Tstepi]
                            if Pli == (nodes.Pln[Eli] - 1):
                                MatA[ri, ci] += (
                                    building_object.__getattribute__(
                                        "heat_convective_elements_internal"
                                    )[Eli]
                                    + heat_radiative_elements_internal_mn
                                )
                                MatA[ri, 0] -= building_object.__getattribute__(
                                    "heat_convective_elements_internal"
                                )[Eli]
                                for Elk in range(bui_eln):
                                    Plk = nodes.Pln[Elk] - 1
                                    ck = 1 + nodes.PlnSum[Elk] + Plk
                                    MatA[ri, ck] -= (
                                        area_elements[Elk] / area_elements_tot
                                    ) * building_object.__getattribute__(
                                        "heat_radiative_elements_internal"
                                    )[
                                        Elk
                                    ]
                            elif Pli == 0:
                                if Type_eli[Eli] == "EXT":
                                    MatA[ri, ci] += (
                                        building_object.__getattribute__(
                                            "heat_convective_elements_external"
                                        )[Eli]
                                        + building_object.__getattribute__(
                                            "heat_radiative_elements_external"
                                        )[Eli]
                                    )
                                elif Type_eli[Eli] == "GR":
                                    MatA[ri, ci] += 1 / t_Th.R_gr_ve
                            if Pli > 0:
                                MatA[ri, ci] += h_pli_eli[Pli - 1, Eli]
                                MatA[ri, ci - 1] -= h_pli_eli[Pli - 1, Eli]
                            if Pli < nodes.Pln[Eli] - 1:
                                MatA[ri, ci] += h_pli_eli[Pli, Eli]
                                MatA[ri, ci + 1] -= h_pli_eli[Pli, Eli]

                    theta = np.linalg.solve(MatA, VecB)
                    VecB = theta

                    Theta_int_air[Tstepi, :] = VecB[0, :]
                    Theta_int_r_mn[Tstepi, :] = 0
                    for Eli in range(bui_eln):
                        ri = nodes.PlnSum[Eli] + nodes.Pln[Eli]
                        Theta_int_r_mn[Tstepi, :] += area_elements[Eli] * VecB[ri, :]
                    Theta_int_r_mn[Tstepi, :] /= area_elements_tot
                    Theta_int_op[Tstepi, :] = 0.5 * (
                        Theta_int_air[Tstepi, :] + Theta_int_r_mn[Tstepi, :]
                    )

                    if nrHCmodes > 1:  # se
                        if Theta_int_op[Tstepi, 0] < Theta_H_set:
                            Theta_op_set = Theta_H_set
                            Phi_HC_nd_act[Tstepi] = (
                                building_object.__getattribute__("power_heating_max")
                                * (Theta_op_set - Theta_int_op[Tstepi, 0])
                                / (
                                    Theta_int_op[Tstepi, colB_H]
                                    - Theta_int_op[Tstepi, 0]
                                )
                            )
                            if Phi_HC_nd_act[Tstepi] > building_object.__getattribute__(
                                "power_heating_max"
                            ):
                                Phi_HC_nd_act[Tstepi] = (
                                    building_object.__getattribute__(
                                        "power_heating_max"
                                    )
                                )
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_H]
                                colB_act = colB_H
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True
                        elif Theta_int_op[Tstepi, 0] > Theta_C_set:
                            Theta_op_set = Theta_C_set
                            Phi_HC_nd_act[Tstepi] = (
                                building_object.__getattribute__("power_cooling_max")
                                * (Theta_op_set - Theta_int_op[Tstepi, 0])
                                / (
                                    Theta_int_op[Tstepi, colB_C]
                                    - Theta_int_op[Tstepi, 0]
                                )
                            )
                            if Phi_HC_nd_act[Tstepi] < building_object.__getattribute__(
                                "power_cooling_max"
                            ):
                                Phi_HC_nd_act[Tstepi] = (
                                    building_object.__getattribute__(
                                        "power_cooling_max"
                                    )
                                )
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_C]
                                colB_act = colB_C
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True
                        else:
                            Phi_HC_nd_act[Tstepi] = 0
                            Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                            colB_act = 0
                    else:
                        Phi_HC_nd_act[Tstepi] = Phi_HC_nd_calc[0]
                        Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                        colB_act = 0
                pbar.update(1)

        # post-processing
        Tstep_first_act = 744  # 744 = 24 * 31; actual first time step (1 January) after warmup month of December
        hourly_results = pd.DataFrame(
            data=np.vstack(
                (
                    Phi_HC_nd_act[Tstep_first_act:],
                    Theta_op_act[Tstep_first_act:],
                    sim_df["T2m"][Tstep_first_act:],
                )
            ).T,
            index=sim_df[Tstep_first_act:].index,
            columns=["Q_HC", "T_op", "T_ext"],
        )

        hourly_results["Q_H"] = 0
        mask = hourly_results["Q_HC"] > 0
        hourly_results.loc[mask, "Q_H"] = hourly_results.loc[mask, "Q_HC"].astype(
            "int64"
        )
        hourly_results["Q_C"] = 0
        mask = hourly_results["Q_HC"] < 0
        hourly_results.loc[mask, "Q_C"] = -hourly_results.loc[mask, "Q_HC"].astype(
            "int64"
        )

        Q_H_annual = hourly_results["Q_H"].sum()
        Q_C_annual = hourly_results["Q_C"].sum()
        Q_H_annual_per_sqm = Q_H_annual / building_object.__getattribute__("a_use")
        Q_C_annual_per_sqm = Q_C_annual / building_object.__getattribute__("a_use")

        annual_results_dic = {
            "Q_H_annual": Q_H_annual,
            "Q_C_annual": Q_C_annual,
            "Q_H_annual_per_sqm": Q_H_annual_per_sqm,
            "Q_C_annual_per_sqm": Q_C_annual_per_sqm,
        }
        annual_results_df = pd.DataFrame([annual_results_dic])

        return hourly_results, annual_results_df
