# __author__ = "Daniele Antonucci, Ulrich Filippi Oberegger, Olga Somova"
# __credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberegger", "Olga Somova"]
# __license__ = "MIT"
# __version__ = "0.1"
# __maintainer__ = "Daniele Antonucci"

from operator import pos
import requests
import pandas as pd
import datetime as dt
from timezonefinder import TimezoneFinder
from pytz import timezone
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from pvlib.iotools import epw
from source.ventilation import VentilationInternalGains   
from source.generate_profile import HourlyProfileGenerator
from source.functions import *
from source.generate_profile import get_country_code_from_latlon
from source.table_iso_16798_1 import * 


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
class Shading_Reduction_factor_window :
    """
    Hourly shading reduction factor calculated as Annex F
    """

    shading_reduction_factor_window: pd.DataFrame

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
    # Phi_int: pd.Series
    sim_df_update: pd.DataFrame


@dataclass
class h_natural_vent:
    H_ve_nat: np.array

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

    # GET WEATHER DATA FROM .epw FILE
    @classmethod
    def get_tmy_data_epw(cls, path_weather_file):
        """
        Get Wetaher data from epw file

        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

        :return:
            * *elevation*: altitude of specifici location (type: **float**)
            * *weather_data*: dataframe with wetaher parameters (e.g. outdoor temperature, outdoor relative humidity, etc.) (type: **pd.DataFrame**)
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

    @classmethod
    def Shading_reduction_factor_window(
        cls,
        solar_altitude_angle,
        solar_azimuth_angle,
        I_dir_tot,
        I_dif_tot,
        calendar,
        n_timesteps,
        orientation,
        building_object
    ):
        """
        calculate t
        Calcola il fattore di riduzione per ombreggiamento per ciascuna finestra.
        Ritorna un oggetto Shading_Reduction_factor_window oppure None se non applicabile.
        """

        # 1) Casi banali / invalidi
        if orientation is None:
            # Nessuna orientazione specificata: o ritorna None, oppure potresti decidere di iterare su tutte.
            return None

        if orientation == "HOR":
            # Orizzontale: nessuna finestra verticale da calcolare
            return None

        # 2) Mappatura coerente (come nella 1ª funzione)
        orientation_lookup = {
            "NV": 0.0,    # North-facing azimuth
            "SV": 180.0,  # South
            "EV": 90.0,   # East
            "WV": 270.0,  # West
        }
        if orientation not in orientation_lookup:
            raise ValueError(f"Unknown orientation '{orientation}' passed to shading calculation.")

        orientation_angle = float(orientation_lookup[orientation]) % 360.0

        # 3) Filtro finestre trasparenti con confronto robusto dell’azimut
        def _matches_orientation(surface):
            az = surface.get("orientation", {}).get("azimuth")
            if az is None:
                return False
            try:
                az_f = float(az) % 360.0
            except (TypeError, ValueError):
                return False
            return np.isclose(az_f, orientation_angle, atol=1e-6)

        filtered_windows = [
            s for s in building_object.get("building_surface", [])
            if s.get("type") == "transparent" and _matches_orientation(s)
        ]
        if not filtered_windows:
            return None

        # 4) Calcolo fattori per ciascuna finestra
        F_sh_dir_df = pd.DataFrame(index=calendar.index)

        for window in filtered_windows:
            F_sh_dir_k_ts_hour = pd.Series(np.zeros(n_timesteps), index=calendar.index, dtype=float)
            h_k_sun_t_hour     = pd.Series(np.zeros(n_timesteps), index=calendar.index, dtype=float)

            for i in range(n_timesteps):
                # Proteggi da NaN nei valori angolari
                alpha_deg = float(np.degrees(solar_altitude_angle.iloc[i])) if pd.notna(solar_altitude_angle.iloc[i]) else 0.0
                phi_deg   = float(np.degrees(solar_azimuth_angle.iloc[i]))  if pd.notna(solar_azimuth_angle.iloc[i])  else 0.0

                F_sh_dir_k_t = shading_reduction_factor(
                    alpha_sol_t=alpha_deg,
                    phi_sol_t=phi_deg,
                    beta_k_t=90,
                    gamma_k_t=orientation_angle,
                    D_k_ovh_q=window.get("overhang_proprieties", {}).get("width_of_horizontal_overhangs"),
                    L_k_ovh_q=window.get("width_or_distance_of_shading_elements"),
                    elements_shading_type=window.get("shading_type"),
                    H_k=window.get("height"),
                    W_k=window.get("width"),
                )

                # Attenzione: assumo F_sh_dir_k_t = (fattore_dir, altezza_sole)
                fatt_dir = F_sh_dir_k_t[0] if F_sh_dir_k_t is not None else 0.0
                if fatt_dir > 0:
                    I_dir = float(I_dir_tot.iloc[i]) if pd.notna(I_dir_tot.iloc[i]) else 0.0
                    I_dif = float(I_dif_tot.iloc[i]) if pd.notna(I_dif_tot.iloc[i]) else 0.0
                    denom = I_dir + I_dif
                    F_sh_obst_k_t = 0.0 if denom == 0 else (fatt_dir * I_dir + I_dif) / denom
                else:
                    F_sh_obst_k_t = 0.0

                F_sh_dir_k_ts_hour.iat[i] = F_sh_obst_k_t
                h_k_sun_t_hour.iat[i]     = F_sh_dir_k_t[1] if F_sh_dir_k_t is not None else 0.0

            name = window.get('name', 'unknown')
            F_sh_dir_df[f"W_{name}"]     = F_sh_dir_k_ts_hour
            F_sh_dir_df[f"H_sun_{name}"] = h_k_sun_t_hour

        return Shading_Reduction_factor_window(shading_reduction_factor_window=F_sh_dir_df)



    # @classmethod
    # def Shading_reduction_factor_window(
    #     cls,
    #     solar_altitude_angle,
    #     solar_azimuth_angle,
    #     I_dir_tot,
    #     I_dif_tot,
    #     calendar,
    #     n_timesteps,
    #     orientation,
    #     building_object
    # ) -> pd.DataFrame:
    #     """
    #     Calculate the shading reduction factor for each window.
    #     :param solar_altitude_angle: Solar altitude angle
    #     :param solar_azimuth_angle: Solar azimuth angle
    #     :param I_dir_tot: Direct solar radiation
    #     :param I_dif_tot: Diffuse solar radiation
    #     :param calendar: Calendar
    #     :param n_timesteps: Number of timesteps
    #     :param orientation: Orientation of the window
    #     :param building_object: Building object

    #     :return: Shading reduction factor for each window
    #     """
    #     orientation_df  =pd.DataFrame({
    #         "name": ['NV', 'SV', 'EV', 'WV'],
    #         "angle": [0,180,90,270]
    #     })
    #     try:
    #         # Extract data from trasparent surfaces 
    #         orientation_angle = orientation_df.loc[orientation_df['name'] == orientation,'angle'].values[0]
    #         filtered_windows = [
    #             s for s in building_object.get("building_surface", [])
    #             if s.get("type") == "transparent"
    #             and s.get("orientation", {}).get("azimuth") == orientation_angle
    #         ]
    #         F_sh_dir_df = pd.DataFrame(index=calendar.index)
    #         for window in filtered_windows:
    #             # orientation_angle = orientation_df.loc[orientation_df['name'] == orientation_w,'angle'].values[0]
    #             F_sh_dir_k_ts_hour = pd.Series(np.zeros(n_timesteps), index=calendar.index)
    #             h_k_sun_t_hour = pd.Series(np.zeros(n_timesteps), index=calendar.index)                    
    #             for i in range(n_timesteps):
    #                 F_sh_dir_k_t = shading_reduction_factor(
    #                     alpha_sol_t=np.degrees(solar_altitude_angle.iloc[i]),
    #                     phi_sol_t=np.degrees(solar_azimuth_angle.iloc[i]),
    #                     beta_k_t=90,
    #                     gamma_k_t=orientation_angle,
    #                     D_k_ovh_q=window.get("overhang_proprieties", {}).get("width_of_horizontal_overhangs"),
    #                     L_k_ovh_q=window.get("width_or_distance_of_shading_elements"),
    #                     elements_shading_type=window.get("shading_type"),
    #                     H_k=window.get("height"),
    #                     W_k=window.get("width")
    #                 )
    #                 if F_sh_dir_k_t[0] > 0:
    #                     if (I_dir_tot.iloc[i] + I_dif_tot.iloc[i]) == 0:
    #                         F_sh_obst_k_t = 0
    #                     else:       
    #                         F_sh_obst_k_t = (F_sh_dir_k_t[0] * I_dir_tot.iloc[i] + I_dif_tot.iloc[i]) / (I_dir_tot.iloc[i] + I_dif_tot.iloc[i])
    #                         # F_sh_obst_k_t = (F_sh_dir_k_t[0] * I_dir_tot[i] + I_dif_tot[i]) / (I_dir_tot[i] + I_dif_tot[i])
    #                 else:
    #                     F_sh_obst_k_t = 0
    #                 F_sh_dir_k_ts_hour.iloc[i] = F_sh_obst_k_t
    #                 h_k_sun_t_hour.iloc[i] = F_sh_dir_k_t[1]
    #             F_sh_dir_df[f"W_{window.get('name')}"] = F_sh_dir_k_ts_hour
    #             F_sh_dir_df[f"H_sun_{window.get('name')}"] = h_k_sun_t_hour
    #         return Shading_Reduction_factor_window(shading_reduction_factor_window=F_sh_dir_df)
    #     except:
    #         pass

 

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
        building_object
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
        solar_altitude_angle_sin = np.sin(declination) * np.sin(latitude) + np.cos(declination) * np.cos(latitude) * np.cos(hour_angle)
        solar_altitude_angle = np.arcsin(np.sin(declination) * np.sin(latitude)+ np.cos(declination) * np.cos(latitude) * np.cos(hour_angle))
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
        #
        I_tot = pd.DataFrame({"I_sol_tot":I_dif_tot + I_dir_tot})
        I_tot['I_sol_dif'] = I_dif_tot
        I_tot['I_sol_dir'] = I_dir_tot
        
        return Solar_irradiance(solar_irradiance=I_tot), solar_altitude_angle, solar_azimuth_angle, I_dir_tot, I_dif_tot


def Calculation_ISO_52010(building_object, path_weather_file, weather_source="pvgis") -> simdf_52010:
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
    # Convert the NumPy array to a tuple
    if isinstance(building_object, dict):
        orientation_elements = ["EV", "HOR", "SV", "NV", "WV"]
    else:
        orientation_elements = building_object.__getattribute__("orientation_elements")

    for orientation in set(orientation_elements):

        Solar_irradiance, alt, az, I_dir_tot, I_dif_tot = ISO52010.Solar_irradiance_calculation(
            n_timesteps=n_tsteps,
            n_days=n_days_year,
            latitude_deg=weatherData.latitude,
            longitude_deg=weatherData.longitude,
            timezone=timezoneW,
            beta_ic_deg=or_tilt_azim_dic[orientation][0],
            gamma_ic_deg=or_tilt_azim_dic[orientation][1],
            DHI=sim_df["Gd(h)"],
            DNI=sim_df["Gb(n)"],
            ground_solar_reflectivity=0.2,
            calendar=sim_df[["day of year", "hour of day"]],
            building_object=building_object
        )
        Solar_irradiance.solar_irradiance.columns = [f'I_sol_tot_{orientation}',f'I_sol_dif_{orientation}',f'I_sol_dir_w_{orientation}']
        sim_df = pd.concat([sim_df, Solar_irradiance.solar_irradiance], axis=1)

        Shading_factor = ISO52010.Shading_reduction_factor_window(
            solar_altitude_angle=alt,
            solar_azimuth_angle=az,
            I_dir_tot=I_dir_tot,
            I_dif_tot=I_dif_tot,
            calendar=sim_df[["day of year", "hour of day"]],
            n_timesteps=n_tsteps,
            building_object=building_object,
            orientation=orientation
        )
        if Shading_factor != None:
            sim_df = pd.concat([sim_df, Shading_factor.shading_reduction_factor_window], axis=1)

    sim_df = pd.concat([sim_df[sim_df.index.month == 12], sim_df])  # weather_data augmented by warmup period consisting of December month copied at the beginning

    # sim_df.to_csv('sim_df_with_shadingf.csv')
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
        el_list = len(building_object["building_surface"])
        # Initialize Pln with all elements as 5
        Pln = np.full(el_list, 5)
        # Replace elements with value 2 where type is "W",  0 if "adiabatic"
        for i, surf in enumerate(building_object["building_surface"]):
            if surf["type"] == "transparent":
                Pln[i] = 2
            elif surf["type"] == "adiabatic":
                Pln[i] = 0
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

            * type: type of building element opaque, transparent or adiabatic
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
        el_type = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        # Initialization of conduttance coefficient calcualation
        h_pli_eli = np.zeros((4, len(el_type)))
        U_eli = [surf["u_value"] for surf in building_object["building_surface"]]
        R_c_eli = [0.0] * len(el_type)
        h_ci_eli = [
            surf["convective_heat_transfer_coefficient_internal"]
            for surf in building_object["building_surface"]
        ]
        h_ri_eli = [
            surf["radiative_heat_transfer_coefficient_internal"]
            for surf in building_object["building_surface"]
        ]
        convective_heat_transfer_coefficient_external = 20.0  # See ISO 13789
        h_ce_eli = [convective_heat_transfer_coefficient_external] * len(el_type)
        for i, surf in enumerate(building_object["building_surface"]):
            if surf["ISO52016_type_string"] == "AD":
                h_ce_eli[i] = 0.0
                surf["convective_heat_transfer_coefficient_external"] = 0.0
            else:
                surf["convective_heat_transfer_coefficient_external"] = (
                    convective_heat_transfer_coefficient_external
                )
        radiative_heat_transfer_coefficient_external = 4.14  # See ISO 13789
        h_re_eli = [radiative_heat_transfer_coefficient_external] * len(el_type)
        for i, surf in enumerate(building_object["building_surface"]):
            if surf["ISO52016_type_string"] == "AD":
                h_re_eli[i] = 0.0
                surf["radiative_heat_transfer_coefficient_external"] = 0.0
            else:
                surf["radiative_heat_transfer_coefficient_external"] = (
                    radiative_heat_transfer_coefficient_external
                )

        for i in range(0, len(el_type)):
            if el_type[i] == "AD":
                R_c_eli[i] = float("inf")
            if R_c_eli[i] == 0.0:
                R_c_eli[i] = (
                    1 / U_eli[i]
                    - 1 / (h_ci_eli[i] + h_ri_eli[i])
                    - 1 / (h_ce_eli[i] + h_re_eli[i])
                )

        # layer = 1
        layer_no = 0
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    h_pli_eli[0, i] = 6 / R_c_eli[i]
                elif el_type[i] == "W":
                    h_pli_eli[0, i] = 1 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[0, i] = 2 / R_gr

        # layer = 2
        layer_no = 1
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":                    
                    h_pli_eli[layer_no, i] = 3 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[layer_no, i] = 1 / (R_c_eli[i] / 4 + R_gr / 2)

        # layer = 3
        layer_no = 2
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    h_pli_eli[layer_no, i] = 3 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[layer_no, i] = 2 / R_c_eli[i]

        # layer = 4
        layer_no = 3
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    h_pli_eli[layer_no, i] = 6 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[layer_no, i] = 4 / R_c_eli[i]

        return conduttance_elements(h_pli_eli=h_pli_eli)

    @classmethod
    def Solar_absorption_of_element(cls, building_object) -> solar_absorption_elements:
        """
        Calculation of solar absorption for each single elements

        :param building_object: building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        :return: a_sol_pli_eli: solar absorption of each single nodes (type: *np.array*)

        .. note:: 
            EXAMPLE:

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
        el_list = len(building_object["building_surface"])

        # Coefficient list of elements
        solar_abs_elements = [0.0] * el_list
        for i, surf in enumerate(building_object["building_surface"]):
            if "solar_absorptance" in surf:  # Opaque element
                if surf["ISO52016_type_string"] == "AD" or surf["ISO52016_type_string"] == "ADJ":
                    solar_abs_elements[i] = 0.0
                else:
                    solar_abs_elements[i] = surf["solar_absorptance"]
            else:  # Transparent element
                solar_abs_elements[i] = surf["g_value"]

        # Initialization of solar_abs_coeff
        a_sol_pli_eli = np.zeros((5, el_list))
        a_sol_pli_eli[0, :] = solar_abs_elements

        return solar_absorption_elements(a_sol_pli_eli=a_sol_pli_eli)

    @classmethod
    def Areal_heat_capacity_of_element(cls, building_object) -> aeral_heat_capacity:
        """
        Calculation of the aeral heat capacity of the node "pli" and node "pli-1" as
        determined per type of construction element [W/m2K] - 6.5.7 ISO 52016

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        .. note:: Required parameters of building_object:

            * type: type of building element OPAQUE -'OP', TRANSPARENT - 'W', GROUND -'GR'
            * kappa_m: heat_capacity of the element in Table B.14
            * construction_class: lass of construction with respect to the distribution of the mass in the construction
                        Table B.13. Possible choice: class_i, class_e, class_ie, class_d, class_m

        :return: aeral_heat_capacity: aeral heat capacity of each facade element (type: *np.array*)
        """
        # Number of envelop building elements
        el_type = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        list_kappa_el = [0] * len(el_type)
        for i, surf in enumerate(building_object["building_surface"]):
            if "thermal_capacity" in surf:
                list_kappa_el[i] = surf["thermal_capacity"]

        # Initialization of heat capacity of nodes
        kappa_pli_eli_ = np.zeros((5, len(el_type)))

        if building_object['building']['construction_class'] == "class_i":
            # Mass concentrated at internal side
            # OPAQUE: kpl5 = km_eli ; kpl1=kpl2=kpl3=kpl4=0
            # GROUND: kpl5 = km_eli ; kpl3=kpl4=0
            node = 1
            for i in range(len(el_type)):
                if el_type[i] == "GR":
                    kappa_pli_eli_[node, i] = 1e6  # heat capacity of the ground
            node = 4
            for i in range(len(el_type)):
                if el_type[i] != "W":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        elif (
            building_object['building']['construction_class'] == "class_e"
        ):  # mass concentrated at external side
            # OPAQUE: kpl1 = km_eli ; kpl2=kpl3=kpl4=kpl5=0
            # GROUND: kpl3 = km_eli ; kpl4=kpl5=0
            node = 0
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                elif el_type[i] == "GR":
                    node = 2
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        elif (
            building_object['building']['construction_class'] == "class_ie"
        ):  # mass divided over internal and external side)
            # OPAQUE: kpl1 = kpl5 = km_eli/2 ; kpl2=kpl3=kpl4=0
            # GROUND: kpl1 = kp5 =km_eli/2; kpl4=0
            node = 0
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
                elif el_type[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
            node = 4
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
                elif el_type[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2

        elif (
            building_object['building']["construction_class"] == "class_d"
        ):  # (mass equally distributed)
            # OPAQUE: kpl2=kpl3=kpl4=km_eli/4
            # GROUND: kpl3=km_eli/4; kpl4=km_eli/2
            node_list_1 = [1, 2, 3]
            for node in node_list_1:
                for i in range(len(el_type)):
                    if el_type[i] == "OP" or el_type[i] == "ADJ":
                        kappa_pli_eli_[node, i] = list_kappa_el[i] / 4
                    if el_type[i] == "GR":
                        if node == 2:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 4
                        if node == 3:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 2

            # OPAQUE kpl1=kpl5= km_eli/8
            # GROUND:kpl5=km_eli/4
            node_list_2 = [0, 4]
            for node in node_list_2:
                for i in range(len(el_type)):
                    if el_type[i] == "OP" or el_type[i] == "ADJ":
                        kappa_pli_eli_[node, i] = list_kappa_el[i] / 8
                    if el_type[i] == "GR":
                        if node == 4:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 4

        elif (
            building_object['building']["construction_class"] == "class_m"
        ):  # mass concentrated inside
            # OPAQUE: kpl1=kpl2=kpl4=kpl5=0; kpl3= km_eli
            # GROUND: kpl4=km_eli; kpl3=kpl5=0
            node = 2
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                if el_type[i] == "GR":
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
        # Use monthly average external temperature as fallback
        annual_mean_external_temperature = (external_temperature_monthly_averages.mean())

        # ============================

        # ============================
        # For the Grins Beat project, we assume active heating and cooling setpoints
        annual_mean_internal_temperature = (
            building_object["building_parameters"]["temperature_setpoints"][
                "heating_setpoint"
            ]
            + building_object["building_parameters"]["temperature_setpoints"][
                "cooling_setpoint"
            ]
        ) / 2
        amplitude_of_internal_temperature_variations = (
            building_object["building_parameters"]["temperature_setpoints"][
                "cooling_setpoint"
            ]
            - building_object["building_parameters"]["temperature_setpoints"][
                "heating_setpoint"
            ]
        ) / 2
        
        # ============================

        # ============================

        coldest_month = 1
        building_object["building_parameters"]["coldest_month"] = coldest_month

        internal_temperature_by_month = np.zeros(12)
        for month in range(12):
            internal_temperature_by_month[month] = (
                annual_mean_internal_temperature
                - amplitude_of_internal_temperature_variations
                * np.cos(2 * np.pi * (month + 1 - coldest_month) / 12)
            )  # estimate
        # ============================

        # ============================
        """
        Area in contact with the ground. 
        If the value is nor provided by the user 
        """
        for surf in building_object["building_surface"]:
            if surf["sky_view_factor"] == 0:
                sog_area = surf["area"]
        # ============================

        # ============================
        """
        Calculation of the perimeter.
        If the value is not provided by the user a rectangluar shape of the building is considered.
        The perimeter is calcuated according to the area of the south and east facade
        """
        
        exposed_perimeter = building_object["building"]["exposed_perimeter"]
        characteristic_floor_dimension = sog_area / (0.5 * exposed_perimeter)
        # ============================

        # ============================
        """
        Calculation of temperature of the ground using:
            1. the thermal Resistance (R) and Transmittance (U) of the floor
            2. External Temperature [°C]
        """
        wall_thickness = building_object["building"]["wall_thickness"]
        thermal_resistance_floor = 5.3
        # building_object.thermal_resistance_floor = 5.3  # Floor construction thermal resistance (excluding effect of ground) [m2 K/W]

        # The thermal transmittance depends on the characteristic dimension of the floor, B' [see 8.1 and Equation (2)], and the total equivalent thickness, dt (see 8.2), defined by Equation (3):
        equivalent_ground_thickness = wall_thickness + lambda_gr * (thermal_resistance_floor + R_se)  # [m]

        if (
            equivalent_ground_thickness < characteristic_floor_dimension
        ):  # uninsulated and moderately insulated floors
            U_sog = (2 * lambda_gr/ (np.pi * characteristic_floor_dimension + equivalent_ground_thickness)
                * np.log(
                    np.pi * characteristic_floor_dimension / equivalent_ground_thickness
                    + 1
                )
            )  # thermal transmittance of slab on ground including effect of ground [W/(m2 K)]
        else:  # well-insulated floors
            U_sog = lambda_gr / (0.457 * characteristic_floor_dimension + equivalent_ground_thickness)

        # calcualtion of thermal resistance of virtual layer
        R_gr_ve = 1 / U_sog - R_si - thermal_resistance_floor - R_gr

        # Adding thermal bridges
        thermal_bridge_heat = exposed_perimeter * psi_k

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
                * np.cos(2 * np.pi * (month + 1 - coldest_month + a_tl) / 12)
            )
        periodic_heat_flow_due_to_external_temperature_variation = np.zeros(12)
        for month in range(12):
            periodic_heat_flow_due_to_external_temperature_variation[month] = (
                H_pe
                * amplitude_of_external_temperature_variations
                * np.cos(2 * np.pi * (month + 1 - coldest_month - b_tl) / 12)
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
            thermal_bridge_heat=thermal_bridge_heat,
        )

    @classmethod
    def Weather_data_bui(cls, building_object, path_weather_file, weather_source="pvgis") -> simulation_df:
        """
        Get weather data for the building object.

        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

        :retrun: sim_df: dataframe with inputs for simulation having information of weather, occupancy, heating and cooling setpoint and setback
        """
        # WEATHER DATA
        if weather_source == "pvgis":
            sim_df = pd.DataFrame(Calculation_ISO_52010(building_object, path_weather_file, weather_source=weather_source).sim_df)
        
        elif weather_source == "epw":
            sim_df = pd.DataFrame(Calculation_ISO_52010(building_object, path_weather_file, weather_source=weather_source).sim_df)
        
        sim_df.index = pd.DatetimeIndex(sim_df.index)
        
        return simulation_df(simulation_df=sim_df)
    
    @classmethod
    def transmission_heat_transfer_coefficient_ISO13789(cls,adj_zone, n_ue=0.5, qui=0):
        '''
        Calculation of heat transfer coefficient, Htr calculated as
        Htr = Hd + Hg + Hu + Ha
        where:
        Hd: direct transmission heat transfer coefficient between the heated and cooled space and exterior trough the building envelope in W/K
        Hg: transmission trasnfer coefficient through the ground in W/K
        Hu: transmission heat transfer coefficent through unconditioned space
        Ha: transmision heat transfer coefficient to adjacent buildings
        '''

        '''
        1. From the adj_zone dictionary, get the orientation, U and area of the facade elements
        '''
        orientation_df  =pd.DataFrame({
            "name": ['NV', 'SV', 'EV', 'WV'],
            "angle": [180,0,90,270]
        })

        # --- Step 1: Get selected orientation name from azimuth
        azimuth = adj_zone["orientation_zone"]["azimuth"]
        orientation_name = orientation_df.loc[orientation_df["angle"] == azimuth, "name"].values[0]

        # --- Step 2: Extract arrays
        areas = adj_zone["area_facade_elements"].astype(float)
        U_values = adj_zone["transmittance_U_elements"].astype(float)
        orientations = np.asarray(adj_zone["orientation_elements"], dtype=object)

        # --- Step 3: Boolean masks
        # mask_selected = orientation_name == orientation_name
        # mask_others = orientation_name != orientation_name
        mask_selected = orientations == orientation_name
        mask_others = orientations != orientation_name

        # --- Step 4: Calculate sums
        '''
        Transmission losses of walls attached to the unconditioned zone
        '''
        Hd_zt_ztu = np.sum(areas[mask_selected] * U_values[mask_selected])

        '''   
        Transmission losses of external walls of unconditioned zone
        '''
        Hd_ztu_ext = np.sum(areas[mask_others] * U_values[mask_others])

        '''
        4. Calculate ventilation losses Hve,iu and Hve,ue
        
        1) Hve,iu = rho*cp*qiu
        2) Hve,ue = rho*cp*que
        
        rho: density of air
        cp: specific heat capacity of air
        qiu: air flow rate in m3/h between conditioned and unconditioned zone   
        que: air flow rate in m3/h between unconditioned zone and external environment
        Note:
        rho_cp = 0.33 Wh/(m3K) if qiu in m3/h

        # air change rate of unconditioned spaces:
        in order to not understimate the transmission heat transfer, the air flow rate between a conditioned space and an unconditioned space 
        shall be assumed to be zero
        qiu= 0
        que = Vu * n_ue
        Vu: volume of the air in the unconditioned space
        n_ue: air change rate between the unconditioned space and the external environment in h-1. Can be taken from the table df_n_ue that is table 7  of ISO 13789

        default n_ue = 0.5 that is for building having All joints between components well-sealed, no ventilation opening provided

        '''    
        volume_zone = adj_zone["volume"]
        que = volume_zone * n_ue
        Hve_ue =  0.33 * que
        Hve_iu = 0.33 * qui
        # 
        Hue = Hd_ztu_ext + Hve_ue
        Hiu = Hd_zt_ztu + Hve_iu
        H_ztu_tot = float(round(Hiu +Hue,3))
        
        '''
        Calculation of the adjustment factor for the thermally uncononditioned afjacent zone ztu for in month m
        b = Hztu_e_m/Hztu_tot_m

        where:
        H_ztu_tot_m = sum(j=1 to n)(H_ztc_j_ztu) + Hztu_e_m
        where:
        H_ztc_j_ztu: heat transfer coefficient between the thermally unconditioned zone and the adjacent thermally conditioned zone j  = Hiu (ISO 13789)
        Hztu_e_m: heat transfer coefficient between the thermally unconditioned zone and the external environment for month m =  Hue (ISO 13789)
        '''
        b_ztu_m = float(round(Hue/(Hue + Hiu),3))

        '''
        Calculation of the distribution factor for the heat transfer between thermally conditioned zone i and the adjacent thermally unconditioned zone ztu, for month m
    
        if multiple thermally conditioned zones:
            F_ztc_ztu_m = H_ztc_i_ztu_m / sum(j=1 to n)(H_ztc_j_ztu)
        where:
            H_ztc_i_ztu_m: heat transfer coefficient between the single thermally conditioned zone and the adjacent thermally unconditioned zone for month m
        if only 1 thermally conditioned zones:
            F_ztc_ztu_m = 1

        '''
        F_ztc_ztu_m = 1 #<<<< write the code formultiple thermally consitioned zones
        
        return H_ztu_tot, b_ztu_m, F_ztc_ztu_m


    def _aggregate_surfaces_by_direction(cls, building_object):
        """
        Collapse multiple surfaces that share the same direction into a single
        equivalent surface so transmission & solar terms are computed on aggregates.

        Key = (ISO52016_type_string, ISO52016_orientation_string, type)

        - Additive: area, thermal_capacity
        - Area-weighted: u_value, g_value, sky_view_factor,
                        h_conv/rad (int & ext), solar_absorptance
        - Non-aggregated fields: keep first reasonable 'name' with a suffix
        """
        if not isinstance(building_object, dict):
            # Only implemented for dict-style building_object here
            return building_object

        # If the tags are not present yet, do nothing
        for s in building_object.get("building_surface", []):
            if "ISO52016_type_string" not in s or "ISO52016_orientation_string" not in s:
                return building_object

        from collections import defaultdict

        buckets = defaultdict(lambda: {
            "name": [],
            "type": None,
            "area": 0.0,
            "uA": 0.0,            # accumulator for U*A
            "gA": 0.0,            # accumulator for g*A (windows only)
            "svfA": 0.0,          # sky_view_factor * A
            "hciA": 0.0,          # convective int * A
            "hriA": 0.0,          # radiative  int * A
            "hceA": 0.0,          # convective ext * A
            "hreA": 0.0,          # radiative  ext * A
            "a_sol_A": 0.0,       # solar_absorptance * A
            "thermal_capacity": 0.0,
            "ISO52016_type_string": None,
            "ISO52016_orientation_string": None
        })

        for s in building_object["building_surface"]:
            tstr = s["ISO52016_type_string"]
            ostr = s["ISO52016_orientation_string"]
            key = (tstr, ostr, s["type"])

            A = float(s.get("area", 0.0))
            U = float(s.get("u_value", 0.0))
            g = float(s.get("g_value", 0.0)) if s.get("type") == "transparent" else 0.0
            svf = float(s.get("sky_view_factor", 0.0))
            hci = float(s.get("convective_heat_transfer_coefficient_internal", 0.0))
            hri = float(s.get("radiative_heat_transfer_coefficient_internal", 0.0))
            hce = float(s.get("convective_heat_transfer_coefficient_external", 0.0))
            hre = float(s.get("radiative_heat_transfer_coefficient_external", 0.0))
            a_sol = float(s.get("solar_absorptance", 0.0))
            Cth = float(s.get("thermal_capacity", 0.0))

            b = buckets[key]
            b["name"].append(s.get("name", "surface"))
            b["type"] = s["type"]
            b["ISO52016_type_string"] = tstr
            b["ISO52016_orientation_string"] = ostr

            b["area"] += A
            b["uA"]   += U * A
            b["gA"]   += g * A
            b["svfA"] += svf * A
            b["hciA"] += hci * A
            b["hriA"] += hri * A
            b["hceA"] += hce * A
            b["hreA"] += hre * A
            b["a_sol_A"] += a_sol * A
            b["thermal_capacity"] += Cth

        # Build new surfaces list
        new_surfaces = []
        for key, b in buckets.items():
            A = b["area"] if b["area"] > 0 else 1.0  # safety
            agg = {
                "name": " + ".join(b["name"])[:120],  # trimmed
                "type": b["type"],
                "area": b["area"],
                "u_value": b["uA"] / A,
                "sky_view_factor": b["svfA"] / A,
                "solar_absorptance": b["a_sol_A"] / A,
                "thermal_capacity": b["thermal_capacity"],
                "ISO52016_type_string": b["ISO52016_type_string"],
                "ISO52016_orientation_string": b["ISO52016_orientation_string"],
                # carry typical window fields if present (area-weighted g)
                "g_value": b["gA"] / A if b["type"] == "transparent" else 0.0,
                # re-attach heat transfer coefficients (area-weighted)
                "convective_heat_transfer_coefficient_internal": b["hciA"] / A,
                "radiative_heat_transfer_coefficient_internal": b["hriA"] / A,
                "convective_heat_transfer_coefficient_external": b["hceA"] / A,
                "radiative_heat_transfer_coefficient_external": b["hreA"] / A,
            }

            # Orientation block: keep the numeric azimuth/tilt from the *first* source name if you like;
            # here we synthesize from the tag only. If you need exact azimuth/tilt, you can store one exemplar.
            # For now, keep a neutral placeholder — your code already uses the *_orientation_string* for logic.
            agg["orientation"] = {"azimuth": 0, "tilt": 90}
            if b["ISO52016_orientation_string"] == "HOR":
                agg["orientation"] = {"azimuth": 0, "tilt": 0}

            new_surfaces.append(agg)

        # Return a shallow-copied object with compacted surfaces
        new_bui = dict(building_object)
        new_bui["building_surface"] = new_surfaces
        return new_bui

    
    
    @classmethod
    def generate_category_profile(
            cls,
            building_object,
            occupants_schedule_workdays,
            occupants_schedule_weekend,
            appliances_schedule_workdays,
            appliances_schedule_weekend,
            lighting_schedule_workdays,
            lighting_schedule_weekend,
        ):
        """
        Generate category_profiles using the profiles defined in the BUI if present.
        If there are NO ventilation/heating/cooling profiles, use the default
        OCCUPANCY profiles (based on building_type_class).
        :param building_object: building object
        :param occupants_schedule_workdays: occupants schedule workdays
        :param occupants_schedule_weekend: occupants schedule weekend
        :param appliances_schedule_workdays: appliances schedule workdays
        :param appliances_schedule_weekend: appliances schedule weekend
        :param lighting_schedule_workdays: lighting schedule workdays
        :param lighting_schedule_weekend: lighting schedule weekend

        :return: dict category_profiles
        """

        import numpy as np

        bt = building_object["building"]["building_type_class"]

        # ---------------- helpers locali ----------------
        def _get_schedule_pair_for_bt(bt_key, workdays_map, weekend_map, name):
            """Give (weekday, weekend) from default dictionaries for the type bt_key."""
            if bt_key not in workdays_map or bt_key not in weekend_map:
                raise KeyError(f"{name}: '{bt_key}' not present in default profiles.")
            wd = np.asarray(workdays_map[bt_key], dtype=float)
            hd = np.asarray(weekend_map[bt_key], dtype=float)
            if wd.shape != (24,) or hd.shape != (24,):
                raise ValueError(f"{name}: default profiles must have 24 values.")
            return wd, hd

        def _pair_from_bui(profile_dict, name):
            """
            If profile_dict is present and valid, returns (weekday, weekend) as np.array(24).
            Altrimenti ritorna None.
            """
            if not profile_dict:
                return None
            try:
                wd = np.asarray(profile_dict["weekday"], dtype=float)
                hd = np.asarray(profile_dict["weekend"], dtype=float)
            except Exception:
                return None
            if wd.shape != (24,) or hd.shape != (24,):
                return None
            return wd, hd

        # ---------- OCCUPANCY ----------
        # 1) try to read from BUI (internal_gains -> 'occupants'); 2) otherwise use default for bt
        occ_entry = None
        for g in building_object["building_parameters"].get("internal_gains", []):
            if g.get("name") == "occupants":
                occ_entry = g
                break

        if occ_entry is not None:
            occ_wd = np.asarray(occ_entry["weekday"], dtype=float)
            occ_hd = np.asarray(occ_entry["weekend"], dtype=float)
            if occ_wd.shape != (24,) or occ_hd.shape != (24,):
                raise ValueError("occupancy: profilo BUI deve avere 24 valori.")
        else:
            occ_wd, occ_hd = _get_schedule_pair_for_bt(
                bt, occupants_schedule_workdays, occupants_schedule_weekend, "occupancy"
            )

        # ---------- APPLIANCES ----------
        app_entry = None
        for g in building_object["building_parameters"].get("internal_gains", []):
            if g.get("name") == "appliances":
                app_entry = g
                break

        if app_entry is not None:
            app_wd = np.asarray(app_entry["weekday"], dtype=float)
            app_hd = np.asarray(app_entry["weekend"], dtype=float)
            if app_wd.shape != (24,) or app_hd.shape != (24,):
                raise ValueError("appliances: profilo BUI deve avere 24 valori.")
        else:
            app_wd, app_hd = _get_schedule_pair_for_bt(
                bt, appliances_schedule_workdays, appliances_schedule_weekend, "appliances"
            )

        # ---------- LIGHTING ----------
        lig_entry = None
        for g in building_object["building_parameters"].get("internal_gains", []):
            if g.get("name") == "lighting":
                lig_entry = g
                break

        if lig_entry is not None:
            lig_wd = np.asarray(lig_entry["weekday"], dtype=float)
            lig_hd = np.asarray(lig_entry["weekend"], dtype=float)
            if lig_wd.shape != (24,) or lig_hd.shape != (24,):
                raise ValueError("lighting: profilo BUI deve avere 24 valori.")
        else:
            lig_wd, lig_hd = _get_schedule_pair_for_bt(
                bt, lighting_schedule_workdays, lighting_schedule_weekend, "lighting"
            )

        # ---------- VENTILATION / HEATING / COOLING ----------
        # Rule: if no profiles in the BUI → use the OCCUPANCY default profiles (occ_wd/occ_hd)
        bp = building_object["building_parameters"]

        # ventilation
        pair = _pair_from_bui(bp.get("ventilation_profile"), "ventilation_profile")
        if pair is None:
            vent_wd, vent_hd = occ_wd.copy(), occ_hd.copy()   # <-- fallback a OCCUPANCY default
        else:
            vent_wd, vent_hd = pair

        # heating
        pair = _pair_from_bui(bp.get("heating_profile"), "heating_profile")
        if pair is None:
            heat_wd, heat_hd = occ_wd.copy(), occ_hd.copy()   # <-- fallback a OCCUPANCY default
        else:
            heat_wd, heat_hd = pair

        # cooling
        pair = _pair_from_bui(bp.get("cooling_profile"), "cooling_profile")
        if pair is None:
            cool_wd, cool_hd = occ_wd.copy(), occ_hd.copy()   # <-- fallback a OCCUPANCY default
        else:
            cool_wd, cool_hd = pair

        # ---------- build category_profiles ----------
        category_profiles = {
            "ventilation": {"weekday": vent_wd, "holiday": vent_hd},
            "heating":     {"weekday": heat_wd, "holiday": heat_hd},
            "cooling":     {"weekday": cool_wd, "holiday": cool_hd},
            "occupancy":   {"weekday": occ_wd,  "holiday": occ_hd},
            "lighting":    {"weekday": lig_wd,  "holiday": lig_hd},
            "appliances":  {"weekday": app_wd,  "holiday": app_hd},
        }

        return category_profiles


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
        :param k_m_int_a_zt: areal thermal capacity of air and furniture per thermally conditioned zone. Default: 10000 J/m2K
        :param f_int_c: convective fraction of the internal gains into the zone. Default: 0.4
        :param f_sol_c: convective fraction of the solar radiation into the zone. Default: 0.1
        :param f_H_c: convective fraction of the heating system per thermally conditioned zone (if system specific). Deafult: 1
        :param f_C_c: convective fraction of the cooling system per thermally conditioned zone (if system specific). Default: 1
        :param delta_Theta_er: Average difference between external air temperature and sky temperature. Default: 11 fro intermediate zones, 13 Tropics and 9 Sub polar areas

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
        with tqdm(total=13) as pbar:

            pbar.set_postfix({"Info": f"Inizailization {i}"})

            # INIZIALIZATION
            if kwargs["weather_source"] == "pvgis":
                path_weather_file_ = None
            
            elif kwargs["weather_source"] == "epw":
                path_weather_file_ = (kwargs["path_weather_file"] if "path_weather_file" in kwargs else None)

            sim_df = ISO52016().Weather_data_bui(building_object, path_weather_file_, weather_source=kwargs["weather_source"]).simulation_df
            Tstepn = len(sim_df)  # number of hours to perform the simulation

            # Heating and cooling Load
            Phi_HC_nd_calc = np.zeros(3)  # Load of Heating or Cooling needed to heat/cool the zone - calculated
            Phi_HC_nd_act = np.zeros(Tstepn)  # Load of Heating or Cooling needed to heat/cool the zone - actual

            # Name adjacent zones
            name_adjacent_zones = [surface["name_adj_zone"] for surface in building_object["building_surface"]]

            # Temperature (indoor and operative)
            '''
            Inizialize vector temperature
            Theta_int_air: internal air temperature
            Theta_int_r_mn: mean radiant temperature
                caluclated as:
                Theta_int_r_min = sum(eli=1 to n)(A_eli * teta_pli=oln,eli,t)/sum(eli=1 to n)(A_eli)
                where:
                A_eli: area of element eli
                teta_pli=oln,eli,t: is the temperature at node pli=pln of the building element eli

            Theta_int_op: operative temperature
            '''
            Theta_int_air = np.zeros((Tstepn, 3))
            Theta_int_r_mn = np.zeros((Tstepn, 3))  # <---
            Theta_int_op = np.zeros((Tstepn, 3))
            Theta_op_act = np.zeros(Tstepn)
            pbar.update(1)

            # Time
            Dtime = 3600.0 * np.ones(Tstepn)
            pbar.update(1)

            # Mode
            colB_act = 0  # the vector B has 3 columns (1st column actual value, 2nd: maximum value reachable in heating, 3rd: maximum value reachbale in cooling)
            pbar.update(1)
            
            # # Number of building element
            bui_eln = len(building_object["building_surface"])

            # Element types and orientations
            typology_elements = np.array(bui_eln * ["EXT"], dtype="object")
            for i, surf in enumerate(building_object["building_surface"]):
                if surf["type"] == "opaque":
                    if surf["sky_view_factor"] == 0:
                        typology_elements[i] = "GR"
                    else:
                        typology_elements[i] = "OP"
                elif surf["type"] == "adiabatic":
                    typology_elements[i] = "AD"
                elif surf["type"] == "transparent":
                    typology_elements[i] = "W"
                elif surf["type"] == "adjacent":
                    typology_elements[i] = "ADJ"
                surf["ISO52016_type_string"] = typology_elements[i]

            Type_eli = bui_eln * ["EXT"]
            for i, t in enumerate(typology_elements):
                if t == "GR":
                    Type_eli[i] = "GR"
                elif t == "ADJ":
                    Type_eli[i] = "ADJ"
                elif t == "AD":
                    Type_eli[i] = "AD"
                else:
                    Type_eli[i] = "EXT"
            
            # --- HYDRATION: set the coefficients for the surfaces if missing ---
            if isinstance(building_object, dict):
                # Typical values (ok for default robust)
                hci_facade = 2.5   # convective internal walls
                hci_ground = 0.7   # convective internal towards ground
                hci_roof   = 5.0   # convective internal roofs
                hce_facade = 20.0  # convective external walls
                hce_roof   = 25.0  # convective external roofs
                hce_ground = 4.0   # convective external ground/adiabatic ≈ negligible
                hre_int    = 5.13  # radiative internal
                hre_ext    = 5.13  # radiative external

                for i, surf in enumerate(building_object["building_surface"]):
                    svf = surf.get("sky_view_factor", 1)
                    tstr = surf.get("ISO52016_type_string", "EXT")

                    # --- convective internal ---
                    if tstr == "AD":         # adiabatic: no exchange with the zone
                        hci = 0.0
                    elif svf == 0:           # towards ground
                        hci = hci_ground
                    elif svf == 1:           # roof
                        hci = hci_roof
                    else:                    # facade
                        hci = hci_facade
                    surf.setdefault("convective_heat_transfer_coefficient_internal", hci)

                    # --- radiative internal ---
                    surf.setdefault("radiative_heat_transfer_coefficient_internal", hre_int)

                    # --- convective external ---
                    if tstr == "AD":
                        hce = 0.0            # external side does not exist for AD: we cancel it
                    elif Type_eli[i] == "GR":
                        hce = hce_ground
                    elif svf == 1:
                        hce = hce_roof
                    else:
                        hce = hce_facade
                    surf.setdefault("convective_heat_transfer_coefficient_external", hce)

                    # --- radiative external ---
                    surf.setdefault("radiative_heat_transfer_coefficient_external", hre_ext)

            #
            pbar.update(1)
            if isinstance(building_object, dict):
                g_values = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    if surf["type"] == "transparent":
                        g_values[i] = surf["g_value"]
                g_gl_wi_t = g_values
            else:
                g_gl_wi_t = np.array(building_object.g_factor_windows)
            
            # Building Area of elements
            if isinstance(building_object, dict):
                area_elements = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    area_elements[i] = surf["area"]
            else:
                area_elements = np.array(building_object.area_elements)
            
            area_elements_tot = np.sum(area_elements)  # Sum of all areas
            pbar.update(1)

            # Orientation and tilt
        
            # 1) Assign the orientation string to all surfaces (without aggregating here)
            orientation_elements = np.empty(bui_eln, dtype=object)
            for i, surf in enumerate(building_object["building_surface"]):
                azimuth = float(surf["orientation"]["azimuth"])
                tilt = float(surf["orientation"]["tilt"])

                # Tolerances for robustness
                def is_close(x, target, tol=1e-6):
                    return abs(x - target) <= tol

                if is_close(tilt, 0.0):
                    orientation_elements[i] = "HOR"
                elif is_close(tilt, 90.0):
                    # normalizza azimuth in [0, 360)
                    az = azimuth % 360.0
                    if is_close(az, 0.0) or is_close(az, 360.0):
                        orientation_elements[i] = "NV"
                    elif is_close(az, 90.0):
                        orientation_elements[i] = "EV"
                    elif is_close(az, 180.0):
                        orientation_elements[i] = "SV"
                    elif is_close(az, 270.0):
                        orientation_elements[i] = "WV"
                    else:
                        # fallback: choose the closest cardinal point
                        # (NV=0, EV=90, SV=180, WV=270)
                        candidates = np.array([0.0, 90.0, 180.0, 270.0])
                        labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
                        orientation_elements[i] = labels[np.argmin(np.abs((az - candidates) % 360.0))]
                else:
                    # if tilt is not exactly 0 or 90, decide the logic (here we map for threshold)
                    orientation_elements[i] = "HOR" if tilt < 45.0 else "NV"

                surf["ISO52016_orientation_string"] = orientation_elements[i]

            # 2) Aggregate (once only) and recalculate helpers
            building_object = ISO52016()._aggregate_surfaces_by_direction(building_object)

            # 3) Reconstruct helpers after aggregation
            bui_eln = len(building_object["building_surface"])
            typology_elements = np.array([s["ISO52016_type_string"] for s in building_object["building_surface"]], dtype=object)
            Type_eli = ["EXT" if t not in ("GR", "ADJ", "AD") else t for t in typology_elements]
            orientation_elements = np.array([s["ISO52016_orientation_string"] for s in building_object["building_surface"]], dtype=object)
            heat_convective_elements_internal = np.array(
                    [s["convective_heat_transfer_coefficient_internal"] for s in building_object["building_surface"]],
                    dtype=float
            )
            heat_radiative_elements_internal = np.array(
                [s["radiative_heat_transfer_coefficient_internal"] for s in building_object["building_surface"]],
                dtype=float
            )
            heat_convective_elements_external = np.array(
                [s["convective_heat_transfer_coefficient_external"] for s in building_object["building_surface"]],
                dtype=float
            )
            heat_radiative_elements_external = np.array(
                [s["radiative_heat_transfer_coefficient_external"] for s in building_object["building_surface"]],
                dtype=float
            )

            g_gl_wi_t = np.array(
                [float(s.get("g_value", 0.0)) if s["type"] == "transparent" else 0.0
                for s in building_object["building_surface"]],
                dtype=float
            )

            sky_factor_elements = np.array(
                [float(s.get("sky_view_factor", 0.0)) for s in building_object["building_surface"]],
                dtype=float
            )    
            area_elements = np.array([float(s["area"]) for s in building_object["building_surface"]], dtype=float)
            area_elements_tot = float(np.sum(area_elements))

            pbar.update(1)

            # External temperature ... (to be checked)
            theta_sup = sim_df["T2m"]
            
            # Thermal capacity of the internal environmnet of the thermal zone
            C_int = (c_int_per_A_us * building_object["building"]["net_floor_area"])
            pbar.update(1)

            # mean internal radiative transfer coefficient
            if isinstance(building_object, dict):
                radiative_heat_transfer_coefficient = 5.13
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        radiative_heat_transfer_coefficient * np.ones(bui_eln),
                    )/ area_elements_tot
                )
                for surf in building_object["building_surface"]:
                    surf["radiative_heat_transfer_coefficient_internal"] = (
                        radiative_heat_transfer_coefficient
                    )
            else:
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        building_object.heat_radiative_elements_internal,
                    )
                    / area_elements_tot
                )
            pbar.update(1)

            
            # inizialiazation vectorB and temperature
            nodes = ISO52016().Number_of_nodes_element(building_object)
            Theta_old = 20 * np.ones(nodes.Rn)
            VecB = 20 * np.ones((nodes.Rn, 3))

            surf_has_node = np.array([nodes.Pln[Eli] > 0 for Eli in range(bui_eln)], dtype=bool)
            surf_int_row  = np.full(bui_eln, -1, dtype=int)
            for Eli in range(bui_eln):
                if surf_has_node[Eli]:
                    # Index of internal node of surface Eli
                    surf_int_row[Eli] = nodes.PlnSum[Eli] + nodes.Pln[Eli]

            # Total area of internal surfaces
            area_int_surfaces_tot = float(area_elements[surf_has_node].sum()) or 1.0

            # Sum convective internal towards air (only surfaces with node)
            Ah_ci = float((area_elements[surf_has_node] *
                        heat_convective_elements_internal[surf_has_node]).sum())

            pbar.update(1)

            # Temperature ground and thermal bridges
            t_Th = ISO52016().Temp_calculation_of_ground(building_object, path_weather_file=path_weather_file_)
            #
            pbar.set_postfix({"Info": f"Calculating ground temperature"})
            pbar.update(1)
            h_pli_eli = (ISO52016().Conduttance_node_of_element(building_object).h_pli_eli)

            pbar.set_postfix({"Info": f"Calculating conduttance of elements"})
            pbar.update(1)
            kappa_pli_eli = (ISO52016().Areal_heat_capacity_of_element(building_object).kappa_pli_eli)

            pbar.set_postfix({"Info": f"Calculating aeral heat capacity of elements"})
            pbar.update(1)
            a_sol_pli_eli = (ISO52016().Solar_absorption_of_element(building_object).a_sol_pli_eli)

            pbar.set_postfix({"Info": f"Calculating solar absorption of element"})
            pbar.update(1)

        # ------------------------------------------------------------------
        # LUMP CAPACITY: add once only the capacity of the AD floors
        # ------------------------------------------------------------------
        if isinstance(building_object, dict):
            for _surf in building_object["building_surface"]:
                if _surf.get("ISO52016_type_string") == "AD":
                    C_int += float(_surf.get("thermal_capacity", 0.0))

        """
        CALCULATION OF SENSIBLE HEATING AND COOLING LOAD (following the procedure of poin 6.5.5.2 of UNI ISO 52016)
        For each hour and each zone the actual internal operative temperature θ and the actual int;ac;op;zt;t 6.5.5.2 Sensible heating and cooling load
        heating or cooling load, ΦHC;ld;ztc;t, is calculated using the following step-wise procedure: 
        """
        H_ve_nat_all = [0]
        # Time step for indoor temperature in adjacent zones
        if building_object['building']['adj_zones_present']:
            list_adj_zones = building_object['building']['number_adj_zone']
            if list_adj_zones == 1:
                theta_ztu = np.zeros(Tstepn)
                theta_ztu[0] = 15
            elif list_adj_zones > 1:
                theta_ztu = np.zeros((Tstepn, list_adj_zones))
                theta_ztu[:2] = 15
                
        
        # Generate profiles
        category_profiles = ISO52016().generate_category_profile(
            building_object, 
            occupants_schedule_workdays,
            occupants_schedule_weekend,
            appliances_schedule_workdays,
            appliances_schedule_weekend,
            lighting_schedule_workdays,
            lighting_schedule_weekend,
            )
        country_calendar= get_country_code_from_latlon(building_object["building"]["latitude"], building_object["building"]["longitude"])
        gen = HourlyProfileGenerator(country=country_calendar, num_months=13, category_profiles=category_profiles)
        profile_df = gen.generate()      

        def _has_energy(arrlike):
            a = np.asarray(arrlike, dtype=float)
            return np.isfinite(a).all() and a.max() > 0 and a.sum() > 0

        # fallback: se heating/cooling/ventilation profili sono piatti (tutti 0), usa occupancy
        for cat in ("heating","cooling","ventilation"):
            col = f"{cat}_profile"
            if not _has_energy(profile_df[col].values):
                profile_df[col] = profile_df["occupancy_profile"].values
  
        Tstepn = len(profile_df)
        # ====================================
        # Get info of porfiles
        # ====================================

        # summury_profile = gen.get_summary()

        # fig = gen.plot_annual_profiles(freq="H", include_weekend_shading=True,
        #                        title="Annual Profiles — Hourly")
        # fig.show()

        # # grafico a medie giornaliere solo per alcune categorie
        # fig_day = gen.plot_annual_profiles(categories=["ventilation","heating","cooling","occupancy"],
        #                                 freq="D", include_weekend_shading=True,
        #                                 title="Annual Profiles — Daily Average")
        # fig_day.show()
                            
        # === ACCUMULATORS FOR SANKEY (Wh) ===
        dt_h = 1.0  # hours per timestep (Dtime is in s)
        # NB: the accumulators will be reset before the start index of the analysis (after warm-up)
        E_solar_Wh = 0.0
        E_internal_Wh = 0.0
        E_heating_Wh = 0.0
        E_cooling_Wh = 0.0
        E_vent_loss_Wh = 0.0
        E_tb_loss_Wh = 0.0
        E_ground_loss_Wh = 0.0
        E_storage_Wh = 0.0

        # ------------------------------------------------------------------
        # STATE CAPACITY (J/K) aligned to the nodes of the equation
        # ------------------------------------------------------------------
        C_state = np.zeros(nodes.Rn, dtype=float)
        C_state[0] = float(C_int)  # node air+furniture (+ AD already lumped)
        for Eli in range(bui_eln):
            n_nodes = nodes.Pln[Eli]
            if n_nodes == 0:
                continue
            for Pli in range(n_nodes):
                ri_state = 1 + nodes.PlnSum[Eli] + Pli
                C_state[ri_state] = float(kappa_pli_eli[Pli, Eli])

        # Previous state for storage (°C): initialized ONCE
        Theta_prev_state = np.full(nodes.Rn, 20.0, dtype=float)

        # --- new structures for TRASMISSIONS per element (only OP and W) ---
        surface_names = [surf["name"] for surf in building_object["building_surface"]]
        surface_types  = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        E_trans_loss_by_surface_Wh = {name: 0.0 for name in surface_names}  # riempiamo solo per OP/W


        win_col_for_index = {}
        for i, s in enumerate(building_object["building_surface"]):
            if s.get("type") == "transparent":
                nm = s.get("name")
                if nm:
                    win_col_for_index[i] = f"W_{nm}"

        # ------------------------------------------------------------------
        # ESCLUDI IL MESE DI WARM-UP DAL SANKEY
        # ------------------------------------------------------------------
        start_idx = 0  # 31d * 24h = 744 (from 1 January, if December is warm-up)
        E_solar_Wh = 0.0
        E_internal_Wh = 0.0
        E_heating_Wh = 0.0
        E_cooling_Wh = 0.0
        E_vent_loss_Wh = 0.0
        E_tb_loss_Wh = 0.0
        E_ground_loss_Wh = 0.0
        E_storage_Wh = 0.0

        with tqdm(total=Tstepn) as pbar:
            n_w = 0
            for Tstepi in range(start_idx, Tstepn):

                if profile_df.iloc[Tstepi]["heating_profile"] > 0:
                    Theta_H_set = building_object["building_parameters"]["temperature_setpoints"]["heating_setpoint"]
                else:
                    Theta_H_set = building_object["building_parameters"]["temperature_setpoints"]["heating_setback"]
                if profile_df.iloc[Tstepi]["cooling_profile"] > 0:
                    Theta_C_set = building_object["building_parameters"]["temperature_setpoints"]["cooling_setpoint"]
                else:
                    Theta_C_set = building_object["building_parameters"]["temperature_setpoints"]["cooling_setback"]

                Theta_old = VecB[:, colB_act]

                # firs step:
                # HEATING:
                # if there is no set point for heating (heating system not installed) -> heating power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_H_set < -995:  #
                    power_heating_max_act = 0
                else:
                    # Reasonable caps
                    A_use = building_object["building"]["net_floor_area"]
                    design_P = max(150.0 * A_use, 15_000.0)  # e.g., 150 W/m² or 15 kW minimum
                    warmup_P = 3.0 * design_P
                    

                    if isinstance(building_object, dict):
                        if (Tstepi < 744):  # During warmup, almost unlimited heating power to ensure convergence to setpoint
                            power_heating_max = warmup_P
                        else:
                            power_heating_max = building_object["building_parameters"]["system_capacities"]["heating_capacity"]
                    else:
                        power_heating_max = building_object.power_heating_max  
                    power_heating_max_act = power_heating_max
                    # power_heating_max_act = building_object.power_heating_max

                # COOLING:
                # if there is no set point for heating (cooling system not installed) -> cooling power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_C_set > 995:
                    power_cooling_max_act = 0
                else:
                    if isinstance(building_object, dict):
                        if (Tstepi < 744):  # During warmup, almost unlimited cooling power to ensure convergence to setpoint
                            power_cooling_max = -1e6
                        else:
                            power_cooling_max = -building_object["building_parameters"]["system_capacities"]["cooling_capacity"]
                        power_cooling_max_act = power_cooling_max
                    else:
                        power_cooling_max = building_object.power_cooling_max
                        power_cooling_max_act = power_cooling_max
                    # power_cooling_max_act = building_object.power_cooling_max

                Phi_HC_nd_calc[0] = 0  # the load has three values:  0 no heating e no cooling, 1  heating, 2 cooling
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
                    Phi_sol_dir_zt_t = 0 # inizialize solar gain
 
                    # Solar  heat gain source inside the thermal zone 6.5.13.2
                    for Eli in range(bui_eln):
                        
                        if isinstance(building_object, dict):
                            if (building_object["building_surface"][Eli]["ISO52016_type_string"]== "AD"):
                                continue

                        if Type_eli[Eli] == "EXT" or Type_eli[Eli] == "ADJ":
                            '''
                            Solar gains for each elements, the sim_df['SV' or 'EV', etc.] is calculated based on the
                            UNI 52010:
                            Phi_sol_dir_zt_t: solar gain [W]
                            g_gl_wi_t: g-value of windows
                            sim_df[orientation_elements[Eli]].iloc[Tstepi]: UNI52010
                            '''
                            
                            # case with shading reduction factor
                            Ffr_wi = 0.25 # <- to modify with shading calculation annex F. o.25 is a good approximation
                            transparent_names = [surface["name"] for surface in building_object["building_surface"] if surface.get("type") == "transparent"]
                            
                            if g_gl_wi_t[Eli] != 0:
                                colname = win_col_for_index.get(Eli)  # column for this specific window element
                                if colname and colname in sim_df.columns:
                                    F_sh_obst_wi_t = float(sim_df[colname].iloc[Tstepi])
                                else:
                                    F_sh_obst_wi_t = 1.0
                            else:
                                F_sh_obst_wi_t = 1.0

                            Phi_sol_dir_zt_t += g_gl_wi_t[Eli] * (sim_df[f'I_sol_dif_{orientation_elements[Eli]}'].iloc[Tstepi] + sim_df[f'I_sol_dir_w_{orientation_elements[Eli]}'].iloc[Tstepi] * F_sh_obst_wi_t) * area_elements[Eli] * (1 - Ffr_wi)
                            
                            '''
                            FRAME AREA FRACTION OF THE WINDOW 
                            ----------------------------------
                            Ffr_wi: frame area fraction of window
                            calculated according to Annex E 
                            Ffr_wi = 1 - (Agl_wi/A_wi)
                            where:
                            Agl_wi: glazing area of window
                            A_wi: total area of window
                            if not provided a value of 0.25 is considered according to the table B21 in the annex B of the ISO

                            SHADING REDUCTION FACTOR DUE TO OBSTACLES FOR DIRECT SOLAR IRRADIATION
                            ----------------------------------------------------------------------
                            # Example balcony or obstacles
                            F_sh_dir_k_t = (h_k_sun_t * w_k_sun_t)/(H_k* W_k)
                            where:
                            h_k_sun_t: horizontal distance from the window to the obstacle
                            w_k_sun_t: vertical distance from the window to the obstacle
                            H_k: height of the facade element k, obtained from the geometry data of the element in [m]. if tilted the vertical projection of the height. For example the height of the window under a balcony
                            W_k: width of the facade element k, obtained from the geometry data of the element in [m]. 
                            '''
                            # Phi_sol_dir_zt_t_tot_new.append(Phi_sol_dir_zt_t)

                    ri = 0
                    '''
                    Energy balacne on zone level. Eq. (38) UNI 52016
                    XTemp = Thermal capacity at specific time (t) and for  a specific degree °C [W] +
                    + Ventilation loss (at time t)[W] + Transmission loss (at time t)[W] + intrnal gain[W] + solar gain [W]. Missed the
                    the convective fraction of the heating/cooling system
                    '''

                    H_ve_nat_hour = VentilationInternalGains(building_object).heat_transfer_coefficient_by_ventilation(building_object, Theta_old[ri], sim_df.iloc[Tstepi]["T2m"], sim_df.iloc[Tstepi]["WS10m"], type_ventilation="occupancy", flowrate_person=0.5)
                    # sanifica
                    if not np.isfinite(H_ve_nat_hour) or H_ve_nat_hour < 0:
                        H_ve_nat_hour = 0.0

                    H_ve_nat = float(H_ve_nat_hour) * float(profile_df['ventilation_profile'].iloc[Tstepi])
                    if not np.isfinite(H_ve_nat) or H_ve_nat < 0:
                        H_ve_nat = 0.0
                    H_ve_nat_all.append(H_ve_nat)
                    
                    
                    # ===========================================================================
                    #                       INTERNAL GAINS
                    # ===========================================================================
                    #                       UNCONDITIONED ZONES
                    # ---------------------------------------------------------------------------
                    # Internal gains and solar of unconditioned zone
                    if building_object['building']['adj_zones_present']:
                        # list_adj_zones = list(building_object.adj_zones.keys())
                        list_adj_zones = building_object['building']['number_adj_zone']
                        adj_bui_class = building_object['adjacent_zones'][0]['building_type_class']
                        adj_bui_a_use = building_object['adjacent_zones'][0]['a_use']
                        phi_int_gains_unc_zone = VentilationInternalGains(building_object).internal_gains(
                            building_type_class = adj_bui_class, 
                            a_use = adj_bui_a_use, 
                            unconditioned_zones_nearby = False,
                            h_occup=float(profile_df['occupancy_profile'].iloc[Tstepi]),
                            h_app=float(profile_df['appliances_profile'].iloc[Tstepi])
                        )
                        phi_solar_gains_unc_zone = 0 # <----- TO BE MODIFIED ACCORDING TO THE WIDNOW OF THE UNCODITIONED ZONES !!!!!
                        phi_gn_dir_ztu = phi_int_gains_unc_zone + phi_solar_gains_unc_zone
                    
                        ## CASE OF SINGLE UNCONDITIONED ZONE
                        if list_adj_zones == 1:
                            adj_zone = building_object['adjacent_zones'][0]
                            H_ztu, b_ztu, F_ztc_ztu_m =ISO52016().transmission_heat_transfer_coefficient_ISO13789(adj_zone)
                        else: 
                            H_ztu_zones = np.zeros((4, list_adj_zones))
                            name_zones = []
                            for i in range(list_adj_zones):
                                adj_zone = building_object['adjacent_zones'][i]
                                H_ztu, b_ztu, F_ztc_ztu_m =ISO52016().transmission_heat_transfer_coefficient_ISO13789(adj_zone)
                                H_ztu_zones[0, i] = H_ztu
                                H_ztu_zones[1, i] = b_ztu
                                H_ztu_zones[2, i] = F_ztc_ztu_m
                                H_ztu_zones[3, i] = adj_zone['orientation_zone']['azimuth']
                                name_zones.append(adj_zone['name'])
                            H_ztu_zones_df = pd.DataFrame(H_ztu_zones, columns=name_zones, index = ['H_ztu', 'b_ztu', 'F_ztc_ztu_m', 'orientation'])
                            
                    # ---------------------------------------------------------------------------
                    # Internal gains conditioned and unconditioned zones
                    if building_object['building']['adj_zones_present']:
                        int_gains_with_unconditioned_zones = VentilationInternalGains(building_object).internal_gains(
                                            building_type_class = building_object['building']['building_type_class'], 
                                            a_use=building_object['building']['net_floor_area'], 
                                            unconditioned_zones_nearby = True, 
                                            Fztc_ztu_m=F_ztc_ztu_m,
                                            list_adj_zones=list_adj_zones,
                                            b_ztu=b_ztu,
                                            h_occup=profile_df['occupancy_profile'][Tstepi],
                                            h_app=profile_df['appliances_profile'][Tstepi],
                                            )
                    else:
                        int_gains_conditioned_zone = VentilationInternalGains(building_object).internal_gains(
                                            building_type_class = building_object['building']['building_type_class'], 
                                            a_use=building_object['building']['net_floor_area'], 
                                            unconditioned_zones_nearby = False,
                                            h_occup=profile_df['occupancy_profile'][Tstepi],
                                            h_app=profile_df['appliances_profile'][Tstepi]
                                            )

                    if building_object['building']['adj_zones_present'] and building_object['building']['number_adj_zone']>=1:
                        int_gains = int_gains_with_unconditioned_zones
                    else:
                        int_gains = int_gains_conditioned_zone
                    
                    # Lump adiabatic surface capacitances into the zone node
                    for surf in building_object["building_surface"]:
                        if surf["ISO52016_type_string"] == "AD":
                            C_int += surf["thermal_capacity"]
                    
                    XTemp = (
                        t_Th.thermal_bridge_heat * sim_df.iloc[Tstepi]["T2m"]
                        + H_ve_nat * theta_sup.iloc[Tstepi]
                        + f_int_c * int_gains
                        + f_sol_c * Phi_sol_dir_zt_t
                        + (C_int / Dtime[Tstepi]) * Theta_old[ri]
                    )
                    # X_temp_old.append(int_gains_vent.H_ve)
                    
                    # adding the convective fraction of the heating/cooling system according to the type of system available (heating, cooling and heating and cooling)
                    for cBi in range(nrHCmodes):
                        if Phi_HC_nd_calc[cBi] > 0:
                            f_HC_c = f_H_c
                        else:
                            f_HC_c = f_C_c
                        VecB[ri, cBi] += XTemp + f_HC_c * Phi_HC_nd_calc[cBi]

                    ci = 0

                    '''
                    First part of the equation of energy balance on zone level(38)
                    [C_int/deltaT] +sum(eli=1 to n)(A_eli + h_ci_eli) + sum(vei = 1 to ven)H_ve,_vei_t + Ht_tb_ztc] * theta_int_a_ztc_t -
                    sum(eli=1 to n)(Aeli * h_ci_eli * theta_pln_eli_t) 
                    '''
                    
                    # ==================================================================
                    MatA[ri, ci] += ((C_int / Dtime[Tstepi])+ Ah_ci+ t_Th.thermal_bridge_heat+ H_ve_nat)

                    heat_convective_elements_internal = [
                        surf["convective_heat_transfer_coefficient_internal"]
                        for surf in building_object["building_surface"]
                    ]
                    heat_radiative_elements_internal = [
                        surf["radiative_heat_transfer_coefficient_internal"]
                        for surf in building_object["building_surface"]
                    ]
                    heat_convective_elements_external = [
                        surf["convective_heat_transfer_coefficient_external"]
                        for surf in building_object["building_surface"]
                    ]
                    heat_radiative_elements_external = [
                        surf["radiative_heat_transfer_coefficient_external"]
                        for surf in building_object["building_surface"]
                    ]
                    sky_factor_elements = [
                        surf["sky_view_factor"]
                        for surf in building_object["building_surface"]
                    ]
                    
                    for Eli in range(bui_eln):
                        Pli = nodes.Pln[Eli]
                        if Pli == 0:  # adiabatic element
                            continue
                        ci = nodes.PlnSum[Eli] + Pli 
                        MatA[ri, ci] -= (area_elements[Eli]* heat_convective_elements_internal[Eli])
                    # ==================================================================

                    # ========================================
                    # Temperature of unconditioned space (if any)
                    # ========================================
                    if building_object['building']['adj_zones_present']:
                        c_ztu_h_max = 1 # from table B.16 
                        if Tstepi >0:
                            # Single zones
                            if list_adj_zones == 1:
                                theta_ztu_t = (Theta_int_op[Tstepi-1,0] - b_ztu*(Theta_int_op[Tstepi-1,0] - sim_df["T2m"].iloc[Tstepi]) + (phi_gn_dir_ztu/H_ztu))                        
                                theta_ztu_t_checked =min(sim_df["T2m"].iloc[Tstepi] + c_ztu_h_max*(Theta_int_op[Tstepi-1,0] - sim_df["T2m"].iloc[Tstepi]), theta_ztu_t)
                                theta_ztu[Tstepi] = theta_ztu_t_checked
                            
                            # Multiple zones
                            elif list_adj_zones > 1:
                                for z in range(list_adj_zones):
                                    zone = building_object['adjacent_zones'][z]
                                    H_ztu = H_ztu_zones_df.loc['H_ztu'][zone['name']]
                                    theta_ztu_t = (Theta_int_op[Tstepi-1,0] - b_ztu*(Theta_int_op[Tstepi-1,0] - sim_df["T2m"].iloc[Tstepi]) + (phi_gn_dir_ztu/H_ztu))
                                    theta_ztu_t_checked =min(sim_df["T2m"].iloc[Tstepi] + c_ztu_h_max*(Theta_int_op[Tstepi-1,0] - sim_df["T2m"].iloc[Tstepi]), theta_ztu_t)
                                    theta_ztu[Tstepi,z] = theta_ztu_t_checked
                        theta_ztu_df = pd.DataFrame(theta_ztu, columns=H_ztu_zones_df.columns.tolist())

                    for Eli in range(bui_eln):
                        n_nodes = nodes.Pln[Eli]
                        if n_nodes == 0:  # adiabatic element
                            continue
                        for Pli in range(n_nodes):
                            ri += 1
                            XTemp = (
                                + (kappa_pli_eli[Pli, Eli] / Dtime[Tstepi])
                                * Theta_old[ri]
                            )
                            for cBi in range(nrHCmodes):
                                VecB[ri, cBi] += XTemp
                            
                            if Pli == (n_nodes - 1): 
                                '''
                                Internal surface node 
                                formula (39) from pli=pln (surface node facing calculation zone ztc)
                                '''
                                # XTemp = (1 - f_int_c) * int_gains_vent.Phi_int.iloc[
                                XTemp = (1 - f_int_c) * int_gains + (1 - f_sol_c) * Phi_sol_dir_zt_t
                                for cBi in range(nrHCmodes):
                                    if Phi_HC_nd_calc[cBi] > 0:
                                        f_HC_c = f_H_c
                                    else:
                                        f_HC_c = f_C_c
                                    VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / area_int_surfaces_tot
                                    # VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / area_elements_tot

                            elif Pli == 0:
                                if Type_eli[Eli] == "EXT":
                                    '''
                                    External surface node - formuala (41) 
                                    phi_sky_eli_t:  (extra) thermal radiation to the sky in W/m2 calcualted by formula 6.5.13.3

                                    '''
                                    phi_sky_eli_t = sky_factor_elements[Eli] * heat_radiative_elements_external[Eli] * delta_Theta_er
                                    XTemp = (
                                        (heat_convective_elements_external[Eli]+ heat_radiative_elements_external[Eli]) * sim_df["T2m"].iloc[Tstepi] \
                                        - phi_sky_eli_t
                                        + a_sol_pli_eli[Pli, Eli] * sim_df[f'I_sol_tot_{orientation_elements[Eli]}'].iloc[Tstepi]
                                    )  
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp
                                
                                elif Type_eli[Eli] == "ADJ":
                                    '''
                                    Case Opaque wall is adjacent to unconditioned thermal zone
                                    phi_sky_eli_t = 0
                                    a_sol_pli_eli = 0
                                    '''
                                    if building_object['building']['adj_zones_present']:
                                        list_adj_zones = building_object['building']['number_adj_zone']
                                        if list_adj_zones > 1:
                                            name_adj_zone = name_adjacent_zones[Eli]
                                            XTemp = ((heat_convective_elements_external[Eli]+ heat_radiative_elements_external[Eli]) * theta_ztu_df[name_adj_zone].iloc[Tstepi]) 
                                        else:
                                            XTemp = ((heat_convective_elements_external[Eli]+ heat_radiative_elements_external[Eli]) * theta_ztu[Tstepi]) 
                                    
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp                                    

                                elif Type_eli[Eli] == "GR":
                                    XTemp = (1 / t_Th.R_gr_ve) * t_Th.Theta_gr_ve[sim_df.index.month[Tstepi] - 1]
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp

                            ci = 1 + nodes.PlnSum[Eli] + Pli
                            MatA[ri, ci] += kappa_pli_eli[Pli, Eli] / Dtime[Tstepi]
                            
                            if Pli == (n_nodes - 1):
                                '''
                                in formula (39) Internal surface node - sum/elk=1 to eln)(A_elk/Atot * h_ri_eli  * Teta_pli_elk_t)
                                '''
                      
                                Area_ratio = 0.0
                                for Elk in range(bui_eln):
                                    if nodes.Pln[Elk] == 0:
                                        continue  # no internal node -> no radiative exchange with the zone
                                    Plk = nodes.Pln[Elk] - 1
                                    ck = 1 + nodes.PlnSum[Elk] + Plk
                                    Area_ratio += area_elements[Elk] / area_int_surfaces_tot
                                    MatA[ri, ck] -= (area_elements[Elk] / area_int_surfaces_tot) * heat_radiative_elements_internal[Elk]
                                '''
                                in formula 39  [.. + h_ci_eli  + h_re_eli * sum(elk=1 to eln)(A_elk/Atot) + ..]
                                '''
                                MatA[ri, ci] += (
                                    heat_convective_elements_internal[Eli] + 
                                    heat_radiative_elements_internal[Eli] * Area_ratio
                                )
                                MatA[ri, 0] -= heat_convective_elements_internal[Eli]


                            elif Pli == 0:
                                
                                if Type_eli[Eli] == "EXT":
                                    MatA[ri, ci] += (
                                        heat_convective_elements_external[Eli]+ 
                                        heat_radiative_elements_external[Eli]
                                    )
                                
                                elif Type_eli[Eli] == "ADJ":
                                    MatA[ri, ci] += (
                                        heat_convective_elements_external[Eli]+ 
                                        heat_radiative_elements_external[Eli]
                                    )
                                
                                elif Type_eli[Eli] == "GR":
                                    MatA[ri, ci] += 1 / t_Th.R_gr_ve
                            
                            if Pli > 0:
                                MatA[ri, ci] += h_pli_eli[Pli - 1, Eli] # hpli-1,eli * teta,pli,eli,t
                                MatA[ri, ci - 1] -= h_pli_eli[Pli - 1, Eli] # - hpli-1,eli * teta,pli-1,eli,t
                            
                            if Pli < n_nodes - 1:
                                MatA[ri, ci] += h_pli_eli[Pli, Eli] # hpli,eli * teta,pli,eli,t
                                MatA[ri, ci + 1] -= h_pli_eli[Pli, Eli] # - hpli,eli * teta,pli+1,eli,t
                    
                    '''
                    Temperature calculation of:
                    - internal air
                    - mean radiant temperature
                    - operative temperature

                    '''
                    ######## solve system of equations #######
                    if np.linalg.matrix_rank(MatA) < MatA.shape[0]:
                        print("⚠️ Warning: MatA is singular or ill-conditioned")
                        print("Rank:", np.linalg.matrix_rank(MatA), "Expected:", MatA.shape[0])
                        np.set_printoptions(precision=3, suppress=True)
                        print("MatA diagonal:", np.diag(MatA))
                    
                    # --- Safe diagonal regularization (avoid read-only error) ---
                    diag_min = max(1e-6, 1e-9 * np.linalg.norm(MatA, ord=np.inf))
                    d = np.diag(MatA).copy()
                    np.maximum(d, diag_min, out=d)
                    np.fill_diagonal(MatA, d)
                    # -------------------------------------------------------------
                    theta = np.linalg.solve(MatA, VecB)
                    VecB = theta
                    
                    # Air Temperature
                    Theta_int_air[Tstepi, :] = VecB[0, :]
                    
                    # --- Mean Radiant Temperature (only internal surfaces) ---
                    Theta_int_r_mn[Tstepi, :] = 0.0
                    A_sum = 0.0
                    for Eli in range(bui_eln):
                        n_nodes_Eli = nodes.Pln[Eli]
                        if n_nodes_Eli == 0:
                            continue  # exclude AD and any surface without node
                        ri_surf = nodes.PlnSum[Eli] + n_nodes_Eli  # index of internal node of surface Eli
                        Theta_int_r_mn[Tstepi, :] += area_elements[Eli] * VecB[ri_surf, :]
                        A_sum += area_elements[Eli]

                    # uses ONLY the area of internal surfaces; fallback for safety
                    if A_sum == 0.0:
                        A_sum = 1.0
                    Theta_int_r_mn[Tstepi, :] /= A_sum

                    # Operative Temperature
                    Theta_int_op[Tstepi, :] = 0.5 * (Theta_int_air[Tstepi, :] + Theta_int_r_mn[Tstepi, :])
                                        
                    '''
                    STEP 2: ISO 
                    Case heating: Determinates if the heating or the cooling temperature set-point applies and calcualte the heating or cooling load: 
                    Use formaula (27):
                    Phi_HC_ld_zd = Phi_HC_upper*((theta_int_op_set - thet_int_op_0)/(theta_int_op_upper - theta_int_op_0))
                    where:
                    Phi_HC_ld_zd: unrestricted heating or cooling load to reach the required setpoint in W
                    Phi_HC_upper: is the upper value of the heating or cooling load in W  
                    theta_int_op_set: required internal operative setpoint temperature in °C
                    thet_int_op_0: operating temperature in free floating condition in °C
                    theta_int_op_upper: is the internal operational temperature, obtained for the upper heating or cooling load °C
                    '''
                    def _safe_load(P_max, T_set, T0, Tupper, clip_factor=2000.0):
                        # ritorna carico (W) usando la (27) con fallback
                        denom = (Tupper - T0)
                        if not np.isfinite(denom) or abs(denom) < 0.05:
                            # fallback lineare limitato
                            return 0.0 if T_set <= T0 else min(P_max, clip_factor * (T_set - T0))
                        return P_max * (T_set - T0) / denom
                        
                    if nrHCmodes > 1:
                        # HEATING
                        if Theta_int_op[Tstepi, 0] < Theta_H_set:
                            Theta_op_set = Theta_H_set
                            Phi_HC_nd_act[Tstepi] = _safe_load(power_heating_max, Theta_op_set,
                                                            Theta_int_op[Tstepi, 0], Theta_int_op[Tstepi, colB_H])
                            if Phi_HC_nd_act[Tstepi] > power_heating_max:
                                Phi_HC_nd_act[Tstepi] = power_heating_max
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_H]
                                colB_act = colB_H
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True

                        # COOLING
                        elif Theta_int_op[Tstepi, 0] > Theta_C_set:
                            Theta_op_set = Theta_C_set
                            Phi_HC_nd_act[Tstepi] = _safe_load(power_cooling_max, Theta_op_set,
                                                            Theta_int_op[Tstepi, 0], Theta_int_op[Tstepi, colB_C])
                            if Phi_HC_nd_act[Tstepi] < power_cooling_max:
                                Phi_HC_nd_act[Tstepi] = power_cooling_max
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_C]
                                colB_act = colB_C
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True

                        else:
                            Phi_HC_nd_act[Tstepi] = 0.0
                            Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                            colB_act = 0
                    else:
                        Phi_HC_nd_act[Tstepi] = Phi_HC_nd_calc[0]
                        Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                        colB_act = 0

                # =========================
                # === SANKEY (per timestep)
                # =========================
                dt_h = float(Dtime[Tstepi]) / 3600.0

                # 1) Storage (air + envelope)
                Theta_curr_state = VecB[:, colB_act]
                dTheta_state = Theta_curr_state - Theta_prev_state
                E_storage_Wh += float(np.dot(C_state, dTheta_state)) / 3600.0
                Theta_prev_state = Theta_curr_state

                # 2) Direct inputs
                phi_solar = float(Phi_sol_dir_zt_t)
                phi_int   = float(int_gains)
                E_solar_Wh    += phi_solar * dt_h
                E_internal_Wh += phi_int   * dt_h

                # 3) Heating/Cooling (uses current load)
                phi_hc = float(Phi_HC_nd_act[Tstepi])
                if   phi_hc > 0: E_heating_Wh +=  phi_hc * dt_h
                elif phi_hc < 0: E_cooling_Wh += (-phi_hc) * dt_h

                # 4) Ventilation
                T_in  = float(Theta_int_air[Tstepi, 0])
                T_out = float(sim_df["T2m"].iloc[Tstepi])
                q_vent = float(H_ve_nat) * (T_in - T_out)
                if q_vent > 0:  E_vent_loss_Wh += q_vent * dt_h
                else:           E_solar_Wh     += (-q_vent) * dt_h

                # 5) Thermal bridges
                q_tb = float(t_Th.thermal_bridge_heat) * (T_in - T_out)
                if q_tb > 0:  E_tb_loss_Wh += q_tb * dt_h
                else:         E_solar_Wh   += (-q_tb) * dt_h

                # 6) Ground
                T_gr = float(t_Th.Theta_gr_ve[sim_df.index.month[Tstepi] - 1])
                R_gr = float(t_Th.R_gr_ve) if float(t_Th.R_gr_ve) != 0 else 1e9
                q_ground = (T_in - T_gr) / R_gr
                if q_ground > 0:  E_ground_loss_Wh += q_ground * dt_h
                else:             E_solar_Wh       += (-q_ground) * dt_h

                # 7) Transmission for element (OP, W)
                T_air = float(Theta_int_air[Tstepi, 0])
                T_rad = float(Theta_int_r_mn[Tstepi, 0])
                for Eli in range(bui_eln):
                    if surface_types[Eli] not in ("OP", "W"):  continue
                    n_nodes_Eli = nodes.Pln[Eli]
                    if n_nodes_Eli == 0:                       continue
                    vecb_row_surface = nodes.PlnSum[Eli] + n_nodes_Eli
                    T_surf_int = float(VecB[vecb_row_surface, colB_act])
                    A   = float(area_elements[Eli])
                    hci = float(heat_convective_elements_internal[Eli])
                    hri = float(heat_radiative_elements_internal[Eli])
                    q_cond = A * (hci * (T_air - T_surf_int) + hri * (T_rad - T_surf_int))
                    if   q_cond > 0: E_trans_loss_by_surface_Wh[surface_names[Eli]] += q_cond * dt_h
                    elif q_cond < 0: E_solar_Wh += (-q_cond) * dt_h


                if Tstepi < 6:  # primi 6 passi di debug
                    print(f"[t={Tstepi}] T_op0={Theta_int_op[Tstepi,0]:.2f}  Phi_HC={Phi_HC_nd_act[Tstepi]:.1f}  "
                        f"int_gains={float(int_gains):.1f}  Phi_solar={float(Phi_sol_dir_zt_t):.1f}  "
                        f"H_ve_nat={float(H_ve_nat):.3f}")
                pbar.update(1)
            n_w=n_w+1

        # =========================
        #  Close balance
        # =========================
        Tstep_first_act = start_idx  # = 744 (dopo warm-up)

        # numeric clamp to avoid -0.0 or microscopically negative values
        def _clamp(x: float) -> float:
            return 0.0 if abs(x) < 1e-9 else x

        # total inputs (Wh)
        inputs_Wh = _clamp(E_heating_Wh) + _clamp(E_internal_Wh) + _clamp(E_solar_Wh)

        # total losses (Wh)
        E_transmission_surfaces_Wh = sum(max(0.0, v) for v in E_trans_loss_by_surface_Wh.values())

        # total outputs (Wh)
        outputs_Wh = (
            _clamp(E_cooling_Wh)      # extracted energy
            + _clamp(E_vent_loss_Wh)  # ventilation
            + _clamp(E_tb_loss_Wh)    # thermal bridges
            + _clamp(E_ground_loss_Wh)# ground
            + _clamp(E_transmission_surfaces_Wh)  # transmission OP/W
        )

        # balance residual (Wh)
        E_transmission_residual_Wh = inputs_Wh - outputs_Wh - _clamp(E_storage_Wh)

        # if the residual is small (<1% of the input) I absorb it into the storage to close the balance
        if inputs_Wh > 0 and abs(E_transmission_residual_Wh) < 0.01 * inputs_Wh:
            E_storage_Wh += E_transmission_residual_Wh
            E_transmission_residual_Wh = 0.0

        # =========================
        #  DATA FOR SANKEY
        # =========================
        sankey_inputs = {
            "Heating": _clamp(E_heating_Wh),
            "Internal gains": _clamp(E_internal_Wh),
            "Solar & free-gain": _clamp(E_solar_Wh),
        }

        sankey_outputs = {
            "Cooling (extracted energy)": _clamp(E_cooling_Wh),
            "Ventilation (losses)": _clamp(E_vent_loss_Wh),
            "Thermal bridges": _clamp(E_tb_loss_Wh),
            "Ground": _clamp(E_ground_loss_Wh),
        }

        # add transmission for each element (only positive branches)
        for name, E_Wh in E_trans_loss_by_surface_Wh.items():
            if E_Wh > 0:
                sankey_outputs[f"Transmission - {name}"] = _clamp(E_Wh)

        # display a non-zero residual (pathological case)
        if E_transmission_residual_Wh > 0:
            sankey_outputs["Transmission (residual)"] = _clamp(E_transmission_residual_Wh)

        sankey_data = {
            "inputs": sankey_inputs,
            "outputs": sankey_outputs,
            "energy_accumulated_zone": _clamp(E_storage_Wh),  # can be non-zero on hourly basis, ~0 on annual basis
        }

        # numeric check
        _inputs = inputs_Wh
        _outs_plus_storage = outputs_Wh + _clamp(E_storage_Wh)
        _res = _inputs - _outs_plus_storage
        _rel = _res / max(1.0, _inputs)
        print(f"SANKEY CHECK  inputs={_inputs:.1f}  outputs+storage={_outs_plus_storage:.1f}  residual={_res:.1f} Wh ({100*_rel:.3f}%)")

        # =========================
        #  HOURLY AND ANNUAL RESULTS
        # =========================
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

        # separate H/C
        hourly_results["Q_H"] = 0.0
        hourly_results.loc[hourly_results["Q_HC"] > 0, "Q_H"] = hourly_results.loc[hourly_results["Q_HC"] > 0, "Q_HC"]

        hourly_results["Q_C"] = 0.0
        hourly_results.loc[hourly_results["Q_HC"] < 0, "Q_C"] = -hourly_results.loc[hourly_results["Q_HC"] < 0, "Q_HC"]

        Q_H_annual = float(hourly_results["Q_H"].sum())
        Q_C_annual = float(hourly_results["Q_C"].sum())
        A_use = float(building_object['building']['net_floor_area'])

        annual_results_dic = {
            "Q_H_annual": Q_H_annual,
            "Q_C_annual": Q_C_annual,
            "Q_H_annual_per_sqm": Q_H_annual / A_use if A_use > 0 else 0.0,
            "Q_C_annual_per_sqm": Q_C_annual / A_use if A_use > 0 else 0.0,
        }
        annual_results_df = pd.DataFrame([annual_results_dic])

        # Sankey
        fig = plot_sankey_building(sankey_data)
        fig.show()

        return hourly_results, annual_results_df
