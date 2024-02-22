__author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"

#%%
import requests
import pandas as pd
import datetime as dt
from timezonefinder import TimezoneFinder
from pytz import timezone
import numpy as np
from dataclasses import dataclass
from src.functions import Equation_of_time, Hour_angle_calc, Air_mass_calc,Get_positions,Filter_list_by_indices
# from functions import Equation_of_time, Hour_angle_calc, Air_mass_calc,get_positions,Filter_list_by_indices
from tqdm import tqdm


#%%

@dataclass
class WeatherDataResult:
    elevation: float
    weather_data: pd.DataFrame
    utc_offset:int

    # def __getitem__(self, item):
    #     return getattr(self, item)

@dataclass
class Solar_irradiance:
    solar_irradiance: pd.DataFrame

@dataclass
class _52010:
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
class solar_abs_elements:
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
    H_tb: float

@dataclass
class h_vent_and_int_gains:
    H_ve: pd.Series
    Phi_int: pd.Series
    sim_df_update: pd.DataFrame


#%%
# ===============================================================================================
#                                       MODULE SIMULATIONS   
# ===============================================================================================
class __ISO52010__:
    n_timesteps = 8760 # number of hours in a year
    solar_constant = 1370  # [W/m2]
    K_eps = 1.104  # [rad^-3]
    Perez_coefficients_matrix = np.array(
    [[1.065, -0.008, 0.588, -0.062, -0.060, 0.072, -0.022], [1.230, 0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
    [1.500, 0.330, 0.487, -0.221, 0.055, -0.064, -0.026], [1.950, 0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
    [2.800, 0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [4.500, 1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
    [6.200, 1.060, -1.600, -0.359, 0.264, -1.127, 0.131], [99999, 0.678, -0.327, -0.250, 0.156, -1.377,
                                                                    0.251]])  # # Values for clearness index and brightness coefficients as function of clearness parameter

    def __init__(self, inputs:dict):
        self.inputs = inputs

    
    def get_tmy_data(self) -> WeatherDataResult:
        '''
        GET weather data from PVGIS API 
        '''
        # Connection to PVGIS API to get weather data 
        latitude = self.inputs['latitude']
        longitude= self.inputs['longitude']
        url = f'https://re.jrc.ec.europa.eu/api/tmy?lat={latitude}&lon={longitude}&outputformat=json&browser=1'
        response = requests.request("GET", url, allow_redirects=True)
        data = response.json()
        df_weather = pd.DataFrame(data['outputs']['tmy_hourly'])
        
        # Time data into UTC
        df_weather['time(UTC)']=[dt.datetime.strptime(x, "%Y%m%d:%H%M") for x in df_weather['time(UTC)']]
        # Order data in date ascending order 
        df_weather = df_weather.sort_values(by = 'time(UTC)')
        df_weather.index = df_weather['time(UTC)']
        del(df_weather['time(UTC)'])
        
        # Elevation is not needed for the energy demand calculation, only for the PV optimization
        loc_elevation = data['inputs']['location']['elevation']
    
        # TIMEZONE FINDER
        tf = TimezoneFinder()
        utcoffset_in_hours = int(timezone(tf.timezone_at(lng=longitude, lat=latitude)).localize(df_weather.index[0]).utcoffset().total_seconds() / 3600.0)

        # return {'elevation': elevation, 'Weather data': df_weather, 'UTC offset': utcoffset_in_hours}
        return WeatherDataResult(
            elevation=loc_elevation,
            weather_data=df_weather,
            utc_offset=utcoffset_in_hours
        )
    
    def Solar_irradiance_calc(self, timezone_utc, beta_ic_deg, gamma_ic_deg, DHI, DNI, ground_solar_reflectivity,
                    calendar, n_timesteps=n_timesteps, solar_constant=solar_constant, K_eps = K_eps,
                    Perez_coefficients_matrix = Perez_coefficients_matrix)->Solar_irradiance:
        """
        ISO 52010-1:2017 specifies a calculation procedure for the conversion of climatic data for energy calculations.
        The main element in ISO 52010-1:2017 is the calculation of solar irradiance on a surface with arbitrary orientation and tilt. 
        A simple method for conversion of solar irradiance to illuminance is also provided.
        The solar irradiance and illuminance on an arbitrary surface are applicable as input for energy and daylighting calculations, for building elements 
        (such as roofs, facades and windows) and for components of technical building systems (such as thermal solar collectors, PV panels)


        Parameters
        ------------
        timezone_utc: The UTC offset (or time offset) is an amount of time subtracted from or added to Coordinated Universal Time (UTC) time to specify the local solar time.
        beta_ic_deg=Tilt angle of inclined surface from horizontal is measured upwards facing (hor.facing upw=0, vert=90, hor.facing down=180).
        gamma_ic_deg=Orientation angle of the inclined (ic) surface, expressed as the geographical azimuth angle of the horizontal projection of the inclined (S=0, E=pos., W=neg.) surface normal.
        DHI = Diffuse horizontal irradiance - [W/m2]
        DNI = Direct (beam) irradiance - [W/m2]
        ground_solar_reflectivity = solar reflectivity of the ground
        calendar: dataframe with days of the year (from 1 to 365) and hour of day (1 to 24), 
        n_timesteps: number of hour in a year = 8760
        Return 
        --------

        Examples
        --------
        .. code-block:: python

        calendar: 
                            day of year	hour of day
        2015-05-01 00:00:00	121	1
        2015-05-01 01:00:00	121	2
        2015-05-01 02:00:00	121	3
        2015-05-01 03:00:00	121	4
        2015-05-01 04:00:00	121	5
        ...	...	...
        2015-09-30 19:00:00	273	20
        2015-09-30 20:00:00	273	21
        2015-09-30 21:00:00	273	22
        2015-09-30 22:00:00	273	23
        2015-09-30 23:00:00	273	24
        """
        
        beta_ic = np.radians(beta_ic_deg)
        gamma_ic = np.radians(gamma_ic_deg)
        latitude = np.radians(self.inputs['latitude'])
        # DayWeekJan1 = 1  # doy of week of 1 January; 1=Monday, 7=Sunday
        #
        # Earth Orbit Deviation. 
        earth_orbit_deviation_deg = 360 / 365 * calendar['day of year']  # [deg]
        earth_orbit_deviation = np.radians(earth_orbit_deviation_deg)  # [rad]
        # Declination - Formula. 6.4.1.1. Unit:deg. Value: hourly
        declination_deg = 0.33281 - 22.984 * np.cos(earth_orbit_deviation) - 0.3499 * np.cos(
            2 * earth_orbit_deviation) - 0.1398 * np.cos(3 * earth_orbit_deviation) + 3.7872 * np.sin(
            earth_orbit_deviation) + 0.03205 * np.sin(2 * earth_orbit_deviation) + 0.07187 * np.sin(
            3 * earth_orbit_deviation)  # [deg]
        declination = np.radians(declination_deg)
        #
        t_eq = Equation_of_time(calendar['day of year'])  # [min]
        time_shift = timezone_utc - (self.inputs['longitude'] / 15)  # [h]
        solar_time = calendar['hour of day'] - t_eq / 60 - time_shift  # [h]
        hour_angle_deg = Hour_angle_calc(solar_time)  # [deg]
        hour_angle = np.radians(hour_angle_deg)
        #
        solar_altitude_angle_sin = np.sin(declination) * np.sin(latitude) + np.cos(declination) * np.cos(
            latitude) * np.cos(hour_angle)
        solar_altitude_angle = np.arcsin(
            np.sin(declination) * np.sin(latitude) + np.cos(declination) * np.cos(latitude) * np.cos(hour_angle))
        solar_altitude_angle[solar_altitude_angle < 1e-4] = 0
        # Solar angle of incidence on the inclined surface - Formula 6.4.1.8
        solar_incidence_angle_ic_cos = np.sin(declination) * np.sin(latitude) * np.cos(
            beta_ic) - np.sin(declination) * np.cos(latitude) * np.sin(beta_ic) * np.cos(
            gamma_ic) + np.cos(declination) * np.cos(latitude) * np.cos(beta_ic) * np.cos(hour_angle) + np.cos(
            declination) * np.sin(latitude) * np.sin(beta_ic) * np.cos(gamma_ic) * np.cos(hour_angle) + np.cos(
            declination) * np.sin(beta_ic) * np.sin(gamma_ic) * np.sin(hour_angle)  # ic=inclined surface
        solar_incidence_angle_ic = np.arccos(solar_incidence_angle_ic_cos)
        #
        air_mass = Air_mass_calc(solar_altitude_angle)  # [-]
        I_ext = solar_constant * (1 + 0.033 * np.cos(earth_orbit_deviation))  # extra-terrestrial radiation [W/m2]
        solar_zenith_angle = np.pi / 2 - solar_altitude_angle
        solar_azimuth_angle_aux_1_sin = (np.cos(declination) * np.sin(np.pi - hour_angle)) / np.cos(
            np.arcsin(solar_altitude_angle_sin))
        solar_azimuth_angle_aux_1_cos = (np.cos(latitude) * np.sin(declination) + np.sin(latitude) * np.cos(
            declination) * np.cos(np.pi - hour_angle)) / np.cos(np.arcsin(solar_altitude_angle_sin))
        solar_azimuth_angle_aux_2 = np.arcsin(
            np.cos(declination) * np.sin(np.pi - hour_angle) / np.cos(np.arcsin(solar_altitude_angle_sin)))
        solar_azimuth_angle = -(np.pi + solar_azimuth_angle_aux_2)
        mask = (solar_azimuth_angle_aux_1_cos > 0) & (solar_azimuth_angle_aux_1_sin >= 0)
        solar_azimuth_angle[mask] = np.pi - solar_azimuth_angle_aux_2[mask]
        mask = solar_azimuth_angle_aux_1_cos < 0
        solar_azimuth_angle[mask] = solar_azimuth_angle_aux_2[mask]
        a_perez = pd.Series(np.zeros(n_timesteps), index=calendar.index)
        
        ###### TO BE CHECKED
        # mask = solar_incidence_angle_ic_cos > 0
        # a_perez[mask] = solar_incidence_angle_ic_cos[mask]
        mask = solar_incidence_angle_ic > 0
        a_perez[mask] = solar_incidence_angle_ic[mask]
        #########
        
        b_perez = np.maximum(np.cos(np.radians(85)) * np.ones(n_timesteps), np.cos(solar_zenith_angle))
        clearness = pd.Series(999 * np.ones(n_timesteps), index=calendar.index)  # [-]
        mask = DHI > 0  # DHI=diffuse horizontal irradiance; DNI=direct (beam) normal irradiance; GHI=global horizontal irradiance
        clearness[mask] = ((DHI[mask] + DNI[mask]) / DHI[mask] + K_eps * np.float_power(solar_altitude_angle[mask], 3)) / (
                    1 + K_eps * np.float_power(solar_altitude_angle[mask], 3))
        sky_brightness = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [-]
        sky_brightness[mask] = air_mass[mask] * DHI[mask] / I_ext[mask]
        
        PerezF = np.zeros((n_timesteps, 6))  # columns in the order: F11, F12, F13, F21, F22, F23
        #
        mask_DHI = DHI > 0
        for i in range(6):
            mask_clearness = clearness < Perez_coefficients_matrix[0, 0]
            mask = mask_DHI & mask_clearness
            PerezF[mask, i] = Perez_coefficients_matrix[0, i + 1]
            for j in range(6):
                mask_clearness = (clearness >= Perez_coefficients_matrix[j, 0]) & (
                            clearness < Perez_coefficients_matrix[j + 1, 0])
                mask = mask_DHI & mask_clearness
                PerezF[mask, i] = Perez_coefficients_matrix[j + 1, i + 1]
            mask_clearness = clearness >= Perez_coefficients_matrix[6, 0]
            mask = mask_DHI & mask_clearness
            PerezF[mask, i] = Perez_coefficients_matrix[7, i + 1]
        #
        PerezF1 = np.maximum(0, PerezF[:, 0] + PerezF[:, 1] * sky_brightness + PerezF[:, 2] * solar_zenith_angle)
        PerezF2 = PerezF[:, 3] + PerezF[:, 4] * sky_brightness + PerezF[:, 5] * solar_zenith_angle
        I_dir = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [W/m2]
        mask = solar_incidence_angle_ic_cos > 0
        I_dir[mask] = DNI[mask] * solar_incidence_angle_ic_cos[mask]
        I_dif = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [W/m2]
        mask = DHI > 0
        I_dif[mask] = DHI[mask] * (
                        (1 - PerezF1[mask]) * (1 + np.cos(beta_ic)) / 2 + PerezF1[mask] * a_perez[mask] / b_perez[mask] + PerezF2[mask] * np.sin(beta_ic))
        I_dif_ground = (DHI + DNI * np.sin(solar_altitude_angle)) * ground_solar_reflectivity * (
                    1 - np.cos(beta_ic)) / 2  # [W/m2]
        I_circum = DHI * PerezF1 * a_perez / b_perez  # [W/m2]
        I_dif_tot = I_dif - I_circum + I_dif_ground
        I_dir_tot = I_dir + I_circum
        I_tot = I_dif_tot + I_dir_tot
        
        return Solar_irradiance(solar_irradiance=I_tot)


def Calc_52010(inputs) ->_52010:
    '''
    Param:
    -------
    weatherData: dictionary with severl inputs of weather data from pvgis obtained using the get_ext_data input 
    
    '''
    
    # get weather dataframe
    # weatherData = get_ext_data(lat,long)
    weatherData = __ISO52010__(inputs).get_tmy_data()
    # sim_df = weatherData['Weather data']
    sim_df = weatherData.weather_data
    # timezoneW = weatherData['UTC offset']
    timezoneW = weatherData.utc_offset
    # Change time index
    sim_df.index.year.unique().values
    # sim_df.loc[sim_df.index.year == 2015,:]
    sim_df.index = pd.to_datetime({'year': 2009, 'month': sim_df.index.month, 'day': sim_df.index.day,
                                    'hour': sim_df.index.hour})
    for column in sim_df:
        sim_df[column] = np.roll(sim_df[column], timezoneW)
    sim_df.rename_axis(index={'time(UTC)': 'time(local)'}, inplace=True)
    sim_df['day of year'] = sim_df.index.dayofyear
    sim_df['hour of day'] = sim_df.index.hour + 1  # 1 to 24
    or_tilt_azim_dic = {'HOR': (0, 0), 'SV': (90, 0), 'EV': (90, 90), 'NV': (90, 180), 'WV': (90,-90)}  # dictionary mapping orientation in or_eli with (beta_ic_deg=elevation/tilt, gamma_ic_deg=azimuth), see util.util.ISO52010_calc()

    for orientation in set(inputs['or_eli']):
        sim_df[orientation] = __ISO52010__(inputs).Solar_irradiance_calc(
            timezone_utc = timezoneW, 
            beta_ic_deg = or_tilt_azim_dic[orientation][0], 
            gamma_ic_deg = or_tilt_azim_dic[orientation][1], 
            DHI=sim_df['Gd(h)'], 
            DNI=sim_df['Gb(n)'], 
            ground_solar_reflectivity=0.2,
            calendar= sim_df[['day of year', 'hour of day']]).solar_irradiance
        
    sim_df = pd.concat([sim_df[sim_df.index.month == 12],sim_df])  # weather_data augmented by warmup period consisting of December month copied at the beginning
    

    return _52010(sim_df=sim_df)



class __ISO52016__:
    
    or_tilt_azim_dic = {'HOR': (0, 0), 'SV': (90, 0), 'EV': (90, 90), 'NV': (90, 180), 'WV': (90, -90)}  # dictionary mapping orientation in or_eli with (beta_ic_deg=elevation/tilt, gamma_ic_deg=azimuth), see util.util.ISO52010_calc() 
   

    __slots__=("inputs","sim_df")
    def __init__(self, inputs:dict):
        '''
        Simulate the Final Energy consumption of the building using the ISO52016
        Parameters
        -------------
        lat : latitude of the building location
        long : longitude of the building location
        '''
        self.inputs = inputs
        self.sim_df = Calc_52010(inputs).sim_df

    

    def Number_of_nodes_element(self,**kwargs) ->numb_nodes_facade_elements:
        '''
        Calculation of the number of nodes for each element.
        If OPACQUE, or ADIABATIC -> n_nodes = 5
        If TRANSPARENT-> n_nodes = 2

        Param
        -----
        type: list of facade elements type ['GR', 'OP', 'OP', 'OP', 'OP', 'OP', 'W', 'W', 'W', 'W']

        Return
        -------
        Rn: value of last node to be used in the definition of the element vector
        Pln: inizial number of nodes according to the type of element (5 - opaque element, 2 - transparent element)
        PlnSum: sequential number of nodes based on the list of opaque and transparent elements 
        '''
        # Number of envelop building elements
        el_list = len(self.inputs['TypeSub_eli'])
        # Initialize Pln with all elements as 5
        Pln = np.full(el_list, 5)
        # Replace elements with value 2 where type is "W"
        Pln[self.inputs['TypeSub_eli'] == "W"] = 2
        # Calculation fo number of nodes for each building element (wall, roof, window)
        PlnSum = np.array([0] * el_list)        
        for Eli in range(1, el_list):
            PlnSum[Eli] = PlnSum[Eli - 1] + Pln[Eli - 1] # Index of matrix , each row is a node 
    
        Rn = PlnSum[-1] + Pln[-1] + 1 # value of last node to be used in the definition of the vector
    
        return numb_nodes_facade_elements(Rn, Pln, PlnSum)

    def Conduttance_node_of_element(self, lambda_gr=2.0, **kwargs) -> conduttance_elements:
        '''
        Calculation of the conduttance between node "pli" adn node "pli-1", as determined per type of construction 
        element in 6.5.7 in W/m2K    
        
        Parameter:
        ----------- 
        lambda_gr: thermal conductivity of ground [W/(m K)]. Default=2

        Required from dict:
        ------------------   
        df: dataframe with: 
            index = name of elements 
            columns:
                - type = type of building element OPAQUE -'OP', TRANSPARENT - 'W', GROUND -'GR' 
                - res = theraml ressistance of opaque building element 
                - kappa_m = heat_capacity of the element in Table B.14 
                - solar_absorption_coeff = solar absorption coefficient of element provided by user or using values of \
                    Table A.15 and B.15 of ISO 52016
                - area: area of each element [m2]
                - ori_tilt: orientation and tilt values
                - g_value: onyl for window
        
        Result:
        -------
        h_pli_eli: conduttance coefficient between nodes (W/m2K)
        '''
        R_gr = 0.5 / lambda_gr  # thermal resistance of 0.5 m of ground [m2 K/W]
        # Number of envelop building elements
        el_type = self.inputs['TypeSub_eli']
        # Initialization of conduttance coefficient calcualation
        h_pli_eli = np.zeros((4,len(el_type)))
        
        # layer = 1 
        layer_no = 0
        for i in range(len(el_type)):
            if self.inputs["R_eli"][i] != 0:
                if el_type[i] == 'OP':
                    h_pli_eli[0, i] = 6 / self.inputs["R_eli"][i]
                elif el_type[i] == 'W':
                    h_pli_eli[0, i] = 1 / self.inputs["R_eli"][i]
                elif el_type[i] == 'GR':
                    h_pli_eli[0, i] = 2 / R_gr
        
        # layer = 2
        layer_no = 1
        for i in range(len(el_type)):
            if self.inputs["R_eli"][i] != 0:
                if el_type[i] == 'OP':
                    h_pli_eli[layer_no, i] = 3 / self.inputs["R_eli"][i]
                elif el_type[i] == 'GR':
                    h_pli_eli[layer_no, i] = 1 / (self.inputs["R_eli"][i] / 4 + R_gr / 2)
        
        # layer = 3
        layer_no = 2
        for i in range(len(el_type)):
            if self.inputs["R_eli"][i] != 0:
                if el_type[i] == 'OP':
                    h_pli_eli[layer_no, i] = 3 / self.inputs["R_eli"][i]
                elif el_type[i] == 'GR':
                    h_pli_eli[layer_no, i] = 2 / self.inputs["R_eli"][i]
        
        # layer = 4
        layer_no = 3
        for i in range(len(el_type)):
            if self.inputs["R_eli"][i] != 0:
                if el_type[i] == 'OP':
                    h_pli_eli[layer_no, i] = 6 / self.inputs["R_eli"][i]
                elif el_type[i] == 'GR':
                    h_pli_eli[layer_no, i] = 4 / self.inputs["R_eli"][i] 
        
        return conduttance_elements(h_pli_eli=h_pli_eli)

    def Solar_absorption_of_element(self, **kwargs) -> solar_abs_elements:
        '''
        Calculation of solar absorption for each single elements
        Param
        -------
        type: list of elements type
        a_sol: coefficients of solar absorption for each elements
        
        Return
        ------
        a_sol_pli_eli: solar absorption of each single nodes

        EXAMPLE
        -------
        inputs = {
            "type": ["GR", "OP", "OP", "OP", "OP", "OP", "W", "W", "W", "W"],
            "a_sol": [0, 0.6, 0.6, 0.6, 0.6, 0.6, 0, 0, 0, 0],
        }
        
        array([[0. , 0.6, 0.6, 0.6, 0.6, 0.6, 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
        '''
        # Number of envelop building elements
        el_list = len(self.inputs['TypeSub_eli'])
        # Coefficient list of elements
        # a_sol_eli = df['solar_absorption_coeff'].to_list()
        a_sol_eli = self.inputs['a_sol_eli']
        
        # Initialization of solar_abs_coeff
        a_sol_pli_eli = np.zeros((5, el_list))
        a_sol_pli_eli[0, :] = a_sol_eli
        
        return solar_abs_elements(a_sol_pli_eli=a_sol_pli_eli)
    
    def Areal_heat_capacity_of_element(self, **kwargs) -> aeral_heat_capacity:
        '''
        Calculation of the aeral heat capacity of the node "pli" and node "pli-1" as 
        determined per type of construction element [W/m2K] - 6.5.7 ISO 52016
        
        Param:
        ------ 
        type: list of elements type
        kappa_m: heat_capacity of the element in Table B.14 (Col name: 'kappa_m')
        construction_class: class of construction with respect to the distribution of the mass in the construction
                        Table B.13. Possible choice: class_i, class_e, class_ie, class_d, class_m

        Return 
        ------
        aeral_heat_capacity: array of aeral heat capacity of each facade element
        '''
        # Number of envelop building elements
        # el_list = len(df)
        el_type = self.inputs['TypeSub_eli']
        # List of heat capacyit of building envelope elements
        # list_kappa_el = df['kappa_m']
        list_kappa_el = self.inputs['kappa_m_eli']

        # Initialization of heat capacity of nodes
        kappa_pli_eli_ = np.zeros((5, len(el_type)))

        #
        if self.inputs["construction_class"] == "class_i":   # Mass concetrated at internal side     
            # OPAQUE: kpl5 = km_eli ; kpl1=kpl2=kpl3=kpl4=0
            # GROUND: kpl5 = km_eli ; kpl3=kpl4=0
            node = 4
            for i in range(len(el_type)):
                if self.inputs['TypeSub_eli'][i] == 'OP':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                elif self.inputs['TypeSub_eli'][i] == 'GR':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
            
            node = 1
            for i in range(len(el_type)):
                if self.inputs['TypeSub_eli'][i] == 'GR':
                    kappa_pli_eli_[node, i] = 1e6 # heat capacity of the ground            
            
        elif self.inputs["construction_class"] == "class_e": # mass concentrated at external side
            # OPAQUE: kpl1 = km_eli ; kpl2=kpl3=kpl4=kpl5=0
            # GROUND: kpl3 = km_eli ; kpl4=kpl5=0
            node = 0
            for i in range(len(el_type)):
                if self.inputs['TypeSub_eli'][i] == 'OP':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                elif self.inputs['TypeSub_eli'][i] == 'GR':
                    node = 2
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                                
        elif self.inputs["construction_class"] == "class_ie": # mass divided over internal and external side)        
            # OPAQUE: kpl1 = kpl5 = km_eli/2 ; kpl2=kpl3=kpl4=0
            # GROUND: kpl1 = kp5 =km_eli/2; kpl4=0
            node = 0
            for i in range(len(el_type)):
                if self.inputs['TypeSub_eli'][i] == 'OP':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]/2
                elif self.inputs['TypeSub_eli'][i] == 'GR':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]/2
            node = 4
            for i in range(len(el_type)):
                if self.inputs['TypeSub_eli'][i] == 'OP':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]/2
                elif self.inputs['TypeSub_eli'][i] == 'GR':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]/2

        elif self.inputs["construction_class"] == "class_d": # (mass equally distributed)
            # OPAQUE: kpl2=kpl3=kpl4=km_eli/4
            # GROUND: kpl3=km_eli/4; kpl4=km_eli/2
            node_list_1 = [1,2,3]
            for node in node_list_1:            
                for i in range(len(el_type)):
                    if self.inputs['TypeSub_eli'][i] == 'OP':
                        kappa_pli_eli_[node, i] = list_kappa_el[i]/4
                    if self.inputs['TypeSub_eli'][i] == 'GR':
                        if node==2: 
                            kappa_pli_eli_[node, i] = list_kappa_el[i]/4
                        if node==3: 
                            kappa_pli_eli_[node, i] = list_kappa_el[i]/2
                                            
            # OPAQUE kpl1=kpl5= km_eli/8
            # GROUND:kpl5=km_eli/4
            node_list_2 = [0,4]
            for node in node_list_2:
                for i in range(len(el_type)):
                    if self.inputs['TypeSub_eli'][i] == 'OP':
                        kappa_pli_eli_[node, i] = list_kappa_el[i]/8
                    if self.inputs['TypeSub_eli'][i] == 'GR':
                        if node == 4:
                            kappa_pli_eli_[node, i] = list_kappa_el[i]/4
                
        elif self.inputs["construction_class"] == "class_m": # mass concentrated inside
            # OPAQUE: kpl1=kpl2=kpl4=kpl5=0; kpl3= km_eli
            # GROUND: kpl4=km_eli; kpl3=kpl5=0
            node = 2
            for i in range(len(el_type)):
                if self.inputs['TypeSub_eli'][i] == 'OP':
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                if self.inputs['TypeSub_eli'][i] == 'GR':
                    node = 3
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
        
        return aeral_heat_capacity(kappa_pli_eli=kappa_pli_eli_)

    def Temp_calculation_of_ground(self, lambda_gr=2.0, R_si=0.17, R_se=0.04, psi_k=0.05, **kwargs) -> temp_ground:
        '''
        Virtual ground temperature calculation of ground according to ISO 13370-1:2017 
        for salb-on-ground (sog) floor
        
        Param:
        ------       
        R_se: external surface resistance (for ground floor calculation). Default 0.04
        R_si: internal surface resistance (for ground floor calculation). Default 0.17             
        psi_k : linear thermal transmittance associated with wall/floor junction [W/(m K)]. Default 0.05
        Required from dict
        ------------------
        heating: TRUE or FALSE. Is there a heating system?
        cooling: TRUE or FALSE. Is there a cooling system?
        H_setpoint: setpoint for the heating system (default 20°C)
        C_setpoint: setpoint for cooling system (default 26°C)
        latitude_deg: latitude of location in degrees
        slab_on_ground_area: area of the building in contact with the ground
        perimeter: perimeter of the building [m]
        wall_thickness: thickness of the wall [m]
        R_floor_construction: resitance of the floor [m2K/W]
        H_tb: Thermal bridges heat transfer coefficient - sum of thermal bridges (clause 6.6.5.3)
        coldest_month: coldest month, if not provided automatically selected according to the hemisphere
        
        Return:
        -------
        R_gr_ve: Thermal resistance of virtual layer (floor_slab)
        H_tb: Heat transfer coefficient of overall thermal briges
        Theta_gr_ve: Internal Temperature of the ground
        '''
        
        # ============================ 
        '''
        In
        DA INTEGRARE CODICE RELATIVO A DIVERSE TIPOLOGIE DI CONTATTO IN BASE ALLA PRESENZA DI 
        LOCALRE NON RISCALADATO, O ALTRO
        '''
        # 
        R_gr = 0.5 / lambda_gr  # thermal resistance of 0.5 m of ground [m2 K/W]
        
        # ============================ 
        # GET MIN, MAX AND MEAN of External temperature values at monthly(M) resolution
        external_temperature_monthly_averages = self.sim_df['T2m'].resample('ME').mean()
        external_temperature_monthly_minima = self.sim_df['T2m'].resample('ME').min()
        external_temperature_monthly_maxima = self.sim_df['T2m'].resample('ME').max()
        # amplitude of external temperature variations
        amplitude_of_external_temperature_variations = (external_temperature_monthly_maxima - external_temperature_monthly_minima).mean() / 2
        # annual mean of external temperature
        annual_mean_external_temperature = external_temperature_monthly_averages.mean()
        # ============================ 
        '''
        Calculation of annual_mean_internal_temperature and its amplitude variations
        if heating and colling are selected: 
            - the annual mean internal temperature is the average between Heating and Cooling setpoints
            - the amplitude variations is the mean of the difference between Heating and Cooling setpoints
        if not heating and cooling the value should be provided by the user:
            - if the user doesnÄt provide any value, the following values are used:
                annual_mean_internal_temperature = 23 <- ((26 (standard C set point) + 20 (standard H setpoint))/2)
                amplitude_of_internal_temperature_variations = 3 <- (26-20)/2
        '''
        # ============================ 
        if self.inputs['heating'] and self.inputs['cooling']:
            if self.inputs['H_setpoint'] is not None and self.inputs['C_setpoint'] is not None: 
                annual_mean_internal_temperature = (self.inputs['H_setpoint'] + self.inputs['C_setpoint']) / 2  # [deg C]
                amplitude_of_internal_temperature_variations = (self.inputs['C_setpoint'] - self.inputs['H_setpoint']) / 2  # [K]
        else:
            if annual_mean_internal_temperature == None:
                '''
                Expert imput: da inserire dall'utente se non li inserisce fornire dei dati default
                '''
                annual_mean_internal_temperature = 23  # estimate, user input #<-- 
                amplitude_of_internal_temperature_variations = 3  # estimate, user input
        # ============================ 
        
        # ============================ 
        '''
        Defintion of the coldest month accoriding to the position.
        If the user doesn't provide a value between 1 (January) and 12 (Decemebr)
        the default values: 1 for northern hemisphere or 7 in southern hemisphere are used 
        '''
        if not self.inputs['coldest_month']:
            if self.inputs['latitude'] >= 0:
                self.inputs['coldest_month'] = 1  # 1..12; 
            else:
                self.inputs['coldest_month'] = 7
        
        internal_temperature_by_month = np.zeros(12)    
        for month in range(12):
            internal_temperature_by_month[month] = annual_mean_internal_temperature - amplitude_of_internal_temperature_variations * np.cos(2*np.pi * (month + 1 - self.inputs['coldest_month']) / 12)  # estimate
        # ============================ 
        
        # ============================ 
        '''
        Area in contact with the ground. 
        If the value is nor provided by the user 
        ''' 
        sog_area = self.inputs['slab_on_ground_area']
        if sog_area == -999:
            sog_area = sum(Filter_list_by_indices(self.inputs['area'],Get_positions(self.inputs['TypeSub_eli'],'GR')))
        # ============================ 
        
        # ============================ 
        '''
        Calcualtion of the perimeter.
        If the value is not provided by the user a rectangluar shape of the building is considered.
        The perimeter is calcuated according to the area of the south and east facade
        '''
        if self.inputs['exposed_perimeter'] == None:
            # SOUTH FACADE
            south_facade_area = sum(Filter_list_by_indices(self.inputs['area'],Get_positions(self.inputs['or_eli'],'SV')))
            # EAST FACADE
            east_facade_area = sum(Filter_list_by_indices(self.inputs['area'],Get_positions(self.inputs['or_eli'],'EV')))
            #
            facade_height = np.sqrt(east_facade_area * south_facade_area / sog_area)
            sog_width = south_facade_area / facade_height
            sog_length = sog_area / sog_width
            exposed_perimeter = 2 * (sog_length + sog_width)
        else: 
            exposed_perimeter = self.inputs['exposed_perimeter']
        characteristic_floor_dimension = sog_area / (0.5 * exposed_perimeter)
        # ============================ 

        # ============================ 
        '''
        Calculation of temperature of the ground using:
            1. the thermal Resistance (R) and Transmittance (U) of the floor
            2. External Temperature [°C]
        '''  
        if not self.inputs['wall_thickness']:
            self.inputs['wall_thickness'] = 0.35  # [m]
        
        if not self.inputs['R_floor_construction']:
            self.inputs['R_floor_construction'] = 5.3  # Floor construction thermal resistance (excluding effect of ground) [m2 K/W]
        
        # The thermal transmittance depends on the characteristic dimension of the floor, B' [see 8.1 and Equation (2)], and the total equivalent thickness, dt (see 8.2), defined by Equation (3):
        equivalent_ground_thickness = self.inputs['wall_thickness'] + lambda_gr * (self.inputs['R_floor_construction'] + R_se)  # [m]
        
        if equivalent_ground_thickness < characteristic_floor_dimension:  # uninsulated and moderately insulated floors
            U_sog = 2 * lambda_gr / (np.pi * characteristic_floor_dimension + equivalent_ground_thickness) * np.log(np.pi * characteristic_floor_dimension / equivalent_ground_thickness + 1)   # thermal transmittance of slab on ground including effect of ground [W/(m2 K)]
        else:  # well-insulated floors
            U_sog = lambda_gr / (0.457 * characteristic_floor_dimension + equivalent_ground_thickness)

        # calcualtion of thermal resistance of virtual layer
        R_gr_ve = 1 / U_sog - R_si - self.inputs['R_floor_construction'] - R_gr  
        # R_sog_eff = 1 / U_sog - R_si  # effective thermal resistance of floor construction (including effect of ground) [m2 K/W]

        # Adding thermal bridges
        if not self.inputs['H_tb']:
            self.inputs['H_tb'] = exposed_perimeter * psi_k
        else:
            self.inputs['H_tb'] += exposed_perimeter * psi_k
        # Calculation of steady-state  ground  heat  transfer  coefficients  are  related  to  the  ratio  of  equivalent  thickness 
        # to  characteristic floor dimension, and the periodic heat transfer coefficients are related to the ratio 
        # of equivalent thickness to periodic penetration depth
        steady_state_heat_transfer_coefficient = sog_area * U_sog + exposed_perimeter * psi_k  # [W/K]
        periodic_penetration_depth = 3.2  # [m]
        H_pi = sog_area * lambda_gr / equivalent_ground_thickness * np.sqrt(2 / (np.float_power(1 + periodic_penetration_depth / equivalent_ground_thickness, 2) + 1))  # periodic heat transfer coefficient related to internal temperature variations [W/K]
        H_pe = 0.37 * exposed_perimeter * lambda_gr * np.log(periodic_penetration_depth / equivalent_ground_thickness + 1)  # periodic heat transfer coefficient related to external temperature variations [W/K]
        annual_average_heat_flow_rate = steady_state_heat_transfer_coefficient * (annual_mean_internal_temperature - annual_mean_external_temperature)  # [W]
        periodic_heat_flow_due_to_internal_temperature_variation = np.zeros(12)
        a_tl = 0  # time lead of the heat flow cycle compared with that of the internal temperature [months]
        b_tl = 1  # time lag of the heat flow cycle compared with that of the external temperature [months]
        for month in range(12):
            periodic_heat_flow_due_to_internal_temperature_variation[month] = -H_pi * amplitude_of_internal_temperature_variations * np.cos(2 * np.pi * (month + 1 - self.inputs['coldest_month'] + a_tl) / 12)
        periodic_heat_flow_due_to_external_temperature_variation = np.zeros(12)
        for month in range(12):
            periodic_heat_flow_due_to_external_temperature_variation[month] = H_pe * amplitude_of_external_temperature_variations * np.cos(2 * np.pi * (month + 1 - self.inputs['coldest_month'] - b_tl) / 12)
        average_heat_flow_rate = annual_average_heat_flow_rate + periodic_heat_flow_due_to_internal_temperature_variation + periodic_heat_flow_due_to_external_temperature_variation
        Theta_gr_ve = internal_temperature_by_month - (average_heat_flow_rate - exposed_perimeter * psi_k * (annual_mean_internal_temperature - annual_mean_external_temperature)) / (sog_area * U_sog)
        
        return temp_ground(R_gr_ve=R_gr_ve,Theta_gr_ve=Theta_gr_ve, H_tb=self.inputs['H_tb'])

    def Occupancy_profile(self, **kwargs) -> simulation_df:
        '''
        Definition of occupancy profile for:
        1) Internal gains 
        2) temperature control and ventilation
        The data is divided in weekend and workday

        Param
        ------
        occ_level_wd: occupancy profile of workday for modification of internal gains
        occ_level_we: occupancy profile of weekend for modification of internal gains
        comf_level_wd: occupancy profile of workday for modification of ventilation
        comf_level_we: occupancy profile of weekend for modification of ventilation
        H_setpoint: value of heating setpoint. e.g 20 
        H_setback: value of heating setback. eg.10 
        C_setpoint: value of cooling setpoint. e.g 26 
        C_setback: value of cooling setback. eg.20

        Return
        ------
        sim_df: dataframe with inputs for simulation having information of weather, occupancy, heating and cooling setpoint and setback
        
        '''
        # WEATHER DATA
        # sim_df = pd.DataFrame(self.get_tmy_data().weather_data)
        sim_df = pd.DataFrame(self.sim_df)
        sim_df.index = pd.DatetimeIndex(sim_df.index)
        # number of days of simulation (13 months)
        number_of_days_with_warmup_period = len(sim_df) // 24
        # Inizailization occupancy for Internal Gain
        sim_df['occupancy level'] = np.nan # 9504 are the numbers of hours in one year + December month for warmup period
        # Inizialization occupancy for Indoor Temperature and Ventilation control
        sim_df['comfort level'] = np.nan

        # Occupation (both for gain and ventilation) workday and weekend according to schedule
        occ_level_wd = self.inputs['occ_level_wd']
        occ_level_we = self.inputs['occ_level_we']
        comf_level_wd = self.inputs['comf_level_wd']
        comf_level_we = self.inputs['comf_level_wd']
        
        ''' WORKDAY '''
        # Nnumber of workdays during the entire simulation period
        wd_mask = sim_df.index.weekday < 5
        # number of workdays for the entire period of simulation (year + warmup: 13 months)
        number_of_weekdays_with_warmup_period = sum(wd_mask) // 24   
        # Associate the occupancy profile to simulation hourly time of workdays
        sim_df.loc[wd_mask, 'occupancy level'] = np.tile(occ_level_wd, number_of_weekdays_with_warmup_period)
        sim_df.loc[wd_mask, 'comfort level'] = np.tile(comf_level_wd, number_of_weekdays_with_warmup_period)
        
        ''' WEEKEND '''
        # number of weekend days for the entire period of simulation (year + warmup: 13 months)
        number_of_weekend_days_with_warmup_period = number_of_days_with_warmup_period - number_of_weekdays_with_warmup_period
        # Number of workdays during the entire simulation period
        we_mask = (sim_df.index.weekday >= 5)
        # Associate the occupancy profile to simulation hourly time of weekends
        sim_df.loc[we_mask, 'occupancy level'] = np.tile(occ_level_we, number_of_weekend_days_with_warmup_period)
        sim_df.loc[we_mask, 'comfort level'] = np.tile(comf_level_we,number_of_weekend_days_with_warmup_period)

        ''' HEATING AND COOLING '''
        # periods where occupancy is =1 and comfort is required and occupancy=0 comfort is not required
        comfort_hi_mask = (sim_df['comfort level'] == 1)
        comfort_lo_mask = (sim_df['comfort level'] == 0)
        
        ''' HEATING '''
        #Associate setback and setpoint of heating to occupancy profile for comfort
        sim_df['Heating'] = np.nan
        sim_df.loc[comfort_hi_mask, 'Heating'] = self.inputs['H_setpoint']
        sim_df.loc[comfort_lo_mask, 'Heating'] = self.inputs['H_setback']
        
        ''' COOLING '''
        #Associate setback and setpoint of cooling to occupancy profile for comfort
        sim_df['Cooling'] = np.nan
        sim_df.loc[comfort_hi_mask, 'Cooling'] = self.inputs['C_setpoint']
        sim_df.loc[comfort_lo_mask, 'Cooling'] = self.inputs['C_setback']

        return simulation_df(simulation_df = sim_df)

    def Vent_heat_transf_coef_and_Int_gains(self, c_air=1006, rho_air=1.204, **kwargs) -> h_vent_and_int_gains:
        '''
        Calculation of heat transfer coefficient (section 8 - ISO 13789:2017 and 6.6.6 ISO 52016:2017 ) and internal gains
        
        Param:
        ------     
        c_air: specific heat of air at constant pressure [J/(kg K)]. Default: 1006
        rho_air: ir density at 20 °C [kg/m3]. Default: 1.204
        
        Required from dict:
        ------------------       
        air_change_rate_base_value: ventilation air change rate [m3/h]
        air_change_rate_extra: extra iar change in case of comfort values [m3/h] 
        a_use: useful area of the building [m2]
        internal_gains_base_value: value fo internal gains [W/m2]
        internal_gains_extra: eventual extra gains during [W]

        Result
        -------
        H_ve: heat transfer coefficient for ventilation [W/K]
        Phi_int: internal gains [W]
        '''
        # VENTILATION (CONDUTTANCE)
        sim_df = __ISO52016__(self.inputs).Occupancy_profile(**self.inputs).simulation_df
        comfort_hi_mask = (sim_df['comfort level'] == 1)
        sim_df['air flow rate'] = self.inputs['air_change_rate_base_value'] * self.inputs['a_use'] # [m3/h]
        sim_df.loc[comfort_hi_mask, 'air flow rate'] += self.inputs['air_change_rate_extra'] * self.inputs['a_use'] 
        air_flow_rate = sim_df['air flow rate']
        H_ve = c_air * rho_air / 3600 * air_flow_rate #[W/K]

        # INTERNAL GAINS
        occ_hi_mask = (sim_df['occupancy level'] == 1)
        sim_df['internal gains'] = self.inputs['internal_gains_base_value'] * self.inputs['a_use'] #[W]
        sim_df.loc[occ_hi_mask, 'internal gains'] += self.inputs['internal_gains_extra'] * self.inputs['a_use'] #[W] 
        Phi_int = sim_df['internal gains']

        return h_vent_and_int_gains(H_ve=H_ve, Phi_int=Phi_int, sim_df_update=sim_df)

    def Temperature_and_Energy_needs_calculation(self, nrHCmodes=2, c_int_per_A_us=10000, f_int_c=0.4, f_sol_c=0.1, 
                                                 f_H_c=1, f_C_c=1, delta_Theta_er=11, **kwargs):
        '''
        Calcualation fo energy needs according to the equation (37) of ISO 52016:2017. Page 60. 
        
        [Matrix A] x [Node temperature vector X] = [State vector B] 
        
        the 
        
        where:
        Theta_int_air: internal air temperature [°C]
        Theta_op_act: Actual operative temperature [°C]
        
        Param
        ------
        nrHCmodes:  inizailization of system mode: 0 for Heating, 1 for Cooling, 2 for Heating and Cooling. Default: 2
        c_int_per_A_us: areal thermal capacity of air and furniture per thermally conditioned zone. Default: 10000
        f_int_c: convective fraction of the internal gains into the zone. Default: 0.4
        f_sol_c: convective fraction of the solar radiation into the zone. Default: 0.1
        f_H_c: convective fraction of the heating system per thermally conditioned zone (if system specific). Deafult: 1
        f_C_c: convective fraction of the cooling system per thermally conditioned zone (if system specific). Default: 1
        delta_Theta_er: Average difference between external air temperature and sky temperature. Default: 11


        INPUT: 
        df: dataframe with: 
            index = name of elements 
            columns:
                - type = type of building element OPAQUE -'OP', TRANSPARENT - 'W', GROUND -'GR' 
                - res = thermal ressistance of opaque building element 
                - kappa_m = heat_capacity of the element in Table B.14 
                - solar_absorption_coeff = solar absorption coefficient of element provided by user or using values of \
                    Table A.15 and B.15 of ISO 52016
                - area: area of each element [m2]
                - ori_tilt: orientation and tilt values
                - g_value: onyl for window
                    
        sim_df: dataframe with: 
            - index: time of simulation on hourly resolution and timeindex typology (13 months on hourly resolution)
            - T2m: Exteranl temperarture [°C]
            - RH: External humidity [%]
            - G(h): 
            - Gb(n):
            - Gd(h):
            - IR(h):
            - WS10m: 
            - WD10m:
            - SP:
            - day of year:
            - hour of day:
            - HOR: 
            - NV:
            - WV: 
            - EV: 
            - SV:
            - occupancy_level:
            - comfort_level:
            - Heating:
            - Cooling:
            - air_flow_rate:
            - internal_gains
        
        Phi_H_nd_max: max power of the heating system (provided by the user) in W
        Phi_C_nd_max: max power of the cooling system (provided by the user) in W
        Rn: ... coming from function: Number_of_nodes_element
        Htb: Heat transmission coefficient for Thermal bridges (provided by the user)
        H_ve: ... coming from function: Ventilation_heat_transfer_coefficient
        Phi_int: ... coming from function: Internal_heat_gains
        a_use: building area [m2]
        Pln: ... coming from function: Number_of_nodes_element
        PlnSum: ... coming from function: Number_of_nodes_element
        a_sol_pli_eli: ... coming from function: Solar_absorption_of_elment
        kappa_pli_eli: ... coming from function:  Areal_heat_capacity_of_element
        h_ci_eli: internal convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
        h_ce_eli: external convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
        h_re_eli: external radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
        h_ri_eli: internal radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
        F_sk_eli: View factor between element and sky
        R_gr_ve : ... coming from function: Temp_calculation_of_ground (Thermal Resitance of virtual layer)
        Theta_gr_ve: ... coming from function: Temp_calculation_of_ground,
        h_pli_eli: ... coming from function: Conduttance_node_of_element
        '''
        i=1
        with tqdm(total=15) as pbar:
            
            pbar.set_postfix({"Info": f"Inizailization {i}"})
            
            # INIZIALIZATION 
            int_gains_vent = __ISO52016__(self.inputs).Vent_heat_transf_coef_and_Int_gains()
            sim_df = int_gains_vent.sim_df_update
            Tstepn = len(sim_df) # number of hours to perform the simulation 

            # HEating and cooling Load
            Phi_HC_nd_calc = np.zeros(3) # Load of Heating or Cooling needed to heat/cool the zone - calculated
            Phi_HC_nd_act = np.zeros(Tstepn) # Load of Heating or Cooling needed to heat/cool the zone - actual

            # Temperature (indoor and operative)
            Theta_int_air = np.zeros((Tstepn, 3))
            Theta_int_r_mn = np.zeros((Tstepn, 3)) # <---
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
            colB_act = 0 #the vector B has 3 columns (1st column actual value, 2nd: maximum value reachable in heating, 3rd: maximum value reachbale in cooling)
            #
            pbar.update(1)
            # Number of building element 
            # bui_eln = len(df)
            bui_eln = len(self.inputs['TypeSub_eli'])
            #
            pbar.update(1)
            # Type of element position if ground or external
            TypeSub_eli = np.array(self.inputs['TypeSub_eli'])
            Type_eli = bui_eln * ['EXT']
            Type_eli[np.where(TypeSub_eli == 'GR')[0][0]] = 'GR'
            #
            pbar.update(1)
            # Window g-value
            # tau_sol_eli = df['g_value'].to_numpy()
            tau_sol_eli = np.array(self.inputs['g_w_eli'])
            # Building Area of elements
            # A_eli = df['area'].to_numpy()
            A_eli = np.array(self.inputs['A_eli'])
            A_eli_tot = np.sum(A_eli) # Sum of all areas
            #
            pbar.update(1)
            # Orientation and tilt
            or_eli = np.array(self.inputs['or_eli'])
            #
            pbar.update(1)
            # External temperature ... (to be cheked)
            theta_sup = sim_df['T2m']
            # Internal capacity 
            C_int = c_int_per_A_us * self.inputs['a_use']
            #
            pbar.update(1)
            # HEat Transfer coefficient for each element Area
            Ah_ci = np.dot(A_eli,self.inputs['h_ci_eli'])
            #
            pbar.update(1)
            # mean internal radiative transfer coefficient 
            h_ri_eli_mn = np.dot(A_eli, self.inputs['h_ri_eli']) / A_eli_tot
            #
            pbar.update(1)
            # inizialiazation vectorB and temperature
            nodes = __ISO52016__(self.inputs).Number_of_nodes_element()
            Theta_old = 20 * np.ones(nodes.Rn)
            VecB = 20 * np.ones((nodes.Rn, 3))
            #
            pbar.update(1)
            #
            # Temperature ground and thermal bridges
            t_Th = __ISO52016__(self.inputs).Temp_calculation_of_ground()
            #
            pbar.set_postfix({"Info": f"Calculating ground temperature"})
            pbar.update(1)
            #
            h_pli_eli=__ISO52016__(self.inputs).Conduttance_node_of_element().h_pli_eli
            #
            pbar.set_postfix({"Info": f"Calculating conduttance of elements"})
            pbar.update(1)
            kappa_pli_eli=__ISO52016__(self.inputs).Areal_heat_capacity_of_element().kappa_pli_eli
            
            #
            pbar.set_postfix({"Info": f"Calculating aeral heat capacity of elements"})
            pbar.update(1)
            
            a_sol_pli_eli=__ISO52016__(self.inputs).Solar_absorption_of_element().a_sol_pli_eli
            #
            pbar.set_postfix({"Info": f"Calculating solar absorption of element"})
            pbar.update(1)
        
        
        '''
        CALCULATION OF SENSIBLE HEATING AND COOLING LOAD (following the procedure of poin 6.5.5.2 of UNI ISO 52016)
        For each hour and each zone the actual internal operative temperature θ and the actual int;ac;op;zt;t 6.5.5.2 Sensible heating and cooling load
        heating or cooling load, ΦHC;ld;ztc;t, is calculated using the following step-wise procedure: 
        '''

        with tqdm(total=Tstepn) as pbar:
            for Tstepi in range(Tstepn):
                # Theta_H_set = sim_df['Heating'][Tstepi]
                Theta_H_set = sim_df.iloc[Tstepi]['Heating']
                # Theta_C_set = sim_df['Cooling'][Tstepi]
                Theta_C_set = sim_df.iloc[Tstepi]['Cooling']
                Theta_old = VecB[:, colB_act]

                # firs step: 
                # HEATING:
                # if there is no set point for heating (heating system not installed) -> heating power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_H_set < -995: # 
                    Phi_H_nd_max_act = 0
                else:
                    Phi_H_nd_max_act = self.inputs['Phi_H_nd_max'] # 
                
                # COOLING: 
                # if there is no set point for heating (cooling system not installed) -> cooling power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_C_set > 995:
                    Phi_C_nd_max_act = 0
                else:
                    Phi_C_nd_max_act = self.inputs['Phi_C_nd_max']

                Phi_HC_nd_calc[0] = 0 # the load has three values:  0 no heating e no cooling, 1  heating, 2 cooling
                if Phi_H_nd_max_act == 0 and Phi_C_nd_max_act == 0: # 
                    nrHCmodes = 1
                elif Phi_C_nd_max_act == 0:
                    colB_H = 1
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_H] = Phi_H_nd_max_act
                elif Phi_H_nd_max_act == 0:
                    colB_C = 1
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_C] = Phi_C_nd_max_act
                else:
                    nrHCmodes = 3
                    colB_H = 1
                    colB_C = 2
                    Phi_HC_nd_calc[colB_H] = Phi_H_nd_max_act
                    Phi_HC_nd_calc[colB_C] = Phi_C_nd_max_act

                iterate = True
                while iterate:

                    iterate = False

                    VecB = np.zeros((nodes.Rn, 3))
                    MatA = np.zeros((nodes.Rn, nodes.Rn))

                    Phi_sol_zi = 0

                    for Eli in range(bui_eln):
                        if Type_eli[Eli] == 'EXT':    
                        # Solar gains for each elements, the sim_df['SV' or 'EV', etc.] is calculated based on the 
                        # UNI 52010:
                        # Phi_sol_zi: solar gain [W]
                        # tu_sol_ei: g-value of windows
                        # sim_df[or_eli[Eli]].iloc[Tstepi]: UNI52010                                                         
                            Phi_sol_zi += tau_sol_eli[Eli] * A_eli[Eli] * sim_df[or_eli[Eli]].iloc[Tstepi]

                    ri = 0
                    # Energy balacne on zone level. Eq. (38) UNI 52016
                    # XTemp = Thermal capacity at specific time (t) and for  a specific degree °C [W] +
                    # + Ventilation loss (at time t)[W] + Transmission loss (at time t)[W] + intrnal gain[W] + solar gain [W]. Missed the 
                    # the convective fraction of the heating/cooling system
                    XTemp = t_Th.H_tb * sim_df.iloc[Tstepi]['T2m'] + int_gains_vent.H_ve.iloc[Tstepi] * theta_sup.iloc[Tstepi] + f_int_c * int_gains_vent.Phi_int.iloc[
                    Tstepi] + f_sol_c * Phi_sol_zi + (C_int / Dtime[Tstepi]) * Theta_old[ri]

                    # adding the convective fraction of the heating/cooling system according to the type of system available (heating, cooling and heating and cooling)
                    for cBi in range(nrHCmodes):
                        if Phi_HC_nd_calc[cBi] > 0:
                            f_HC_c = f_H_c
                        else:
                            f_HC_c = f_C_c
                        VecB[ri, cBi] += XTemp + f_HC_c * Phi_HC_nd_calc[cBi]

                    ci = 0

                    # FIrst part of the equation on the square bracket(38) 
                    MatA[ri, ci] += (C_int / Dtime[Tstepi]) + Ah_ci + t_Th.H_tb + int_gains_vent.H_ve.iloc[Tstepi]

                    for Eli in range(bui_eln):
                        Pli = nodes.Pln[Eli]
                        ci = nodes.PlnSum[Eli] + Pli
                        MatA[ri, ci] -= A_eli[Eli] * self.inputs['h_ci_eli'][Eli]

                    # block_in_iterate_start = time.perf_counter()

                    for Eli in range(bui_eln):
                        for Pli in range(nodes.Pln[Eli]):
                            ri += 1
                            XTemp = a_sol_pli_eli[Pli, Eli] * sim_df[or_eli[Eli]].iloc[Tstepi] + (
                                        kappa_pli_eli[Pli, Eli] / Dtime[Tstepi]) * Theta_old[ri]
                            for cBi in range(nrHCmodes):
                                VecB[ri, cBi] += XTemp
                            if Pli == (nodes.Pln[Eli] - 1):
                                XTemp = ((1 - f_int_c) * int_gains_vent.Phi_int.iloc[Tstepi] + (1 - f_sol_c) * Phi_sol_zi)
                                for cBi in range(nrHCmodes):
                                    if Phi_HC_nd_calc[cBi] > 0:
                                        f_HC_c = f_H_c
                                    else:
                                        f_HC_c = f_C_c
                                    VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / A_eli_tot
                            elif Pli == 0:
                                if Type_eli[Eli] == 'EXT':
                                    XTemp = (self.inputs['h_ce_eli'][Eli] + self.inputs['h_re_eli'][Eli]) * sim_df['T2m'].iloc[Tstepi] - self.inputs['F_sk_eli'][Eli] * \
                                            self.inputs['h_re_eli'][Eli] * delta_Theta_er
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp
                                elif Type_eli[Eli] == 'GR':
                                    XTemp = (1 / t_Th.R_gr_ve) * t_Th.Theta_gr_ve[sim_df.index.month[Tstepi] - 1]
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp

                            ci = 1 + nodes.PlnSum[Eli] + Pli
                            MatA[ri, ci] += kappa_pli_eli[Pli, Eli] / Dtime[Tstepi]
                            if Pli == (nodes.Pln[Eli] - 1):
                                MatA[ri, ci] += self.inputs['h_ci_eli'][Eli] + h_ri_eli_mn
                                MatA[ri, 0] -= self.inputs['h_ci_eli'][Eli]
                                for Elk in range(bui_eln):
                                    Plk = nodes.Pln[Elk] - 1
                                    ck = 1 + nodes.PlnSum[Elk] + Plk
                                    MatA[ri, ck] -= (A_eli[Elk] / A_eli_tot) * self.inputs['h_ri_eli'][Elk]
                            elif Pli == 0:
                                if Type_eli[Eli] == 'EXT':
                                    MatA[ri, ci] += self.inputs['h_ce_eli'][Eli] + self.inputs['h_re_eli'][Eli]
                                elif Type_eli[Eli] == 'GR':
                                    MatA[ri, ci] += 1 / t_Th.R_gr_ve
                            if Pli > 0:
                                MatA[ri, ci] += h_pli_eli[Pli - 1, Eli]
                                MatA[ri, ci - 1] -= h_pli_eli[Pli - 1, Eli]
                            if Pli < nodes.Pln[Eli] - 1:
                                MatA[ri, ci] += h_pli_eli[Pli, Eli]
                                MatA[ri, ci + 1] -= h_pli_eli[Pli, Eli]

                    # total_time_block_in_iterate += (time.perf_counter() - block_in_iterate_start)
                    # pd.DataFrame(MatA).to_csv("MatA_Notworks.csv")
                    # pd.DataFrame(VecB).to_csv("VecB_Notworks.csv")
                    # start_time_solver = time.perf_counter()
                    theta = np.linalg.solve(MatA, VecB)
                    # end_time_solver = time.perf_counter()
                    # total_time_solver += (end_time_solver - start_time_solver)
                    VecB = theta

                    Theta_int_air[Tstepi, :] = VecB[0, :]
                    Theta_int_r_mn[Tstepi, :] = 0
                    for Eli in range(bui_eln):
                        ri = nodes.PlnSum[Eli] + nodes.Pln[Eli]
                        Theta_int_r_mn[Tstepi, :] += A_eli[Eli] * VecB[ri, :]
                    Theta_int_r_mn[Tstepi, :] /= A_eli_tot
                    Theta_int_op[Tstepi, :] = 0.5 * (Theta_int_air[Tstepi, :] + Theta_int_r_mn[Tstepi, :])

                    if nrHCmodes > 1: # se 
                        if Theta_int_op[Tstepi, 0] < Theta_H_set:
                            Theta_op_set = Theta_H_set
                            Phi_HC_nd_act[Tstepi] = self.inputs['Phi_H_nd_max'] * (Theta_op_set - Theta_int_op[Tstepi, 0]) / (
                                        Theta_int_op[Tstepi, colB_H] - Theta_int_op[Tstepi, 0])
                            if Phi_HC_nd_act[Tstepi] > self.inputs['Phi_H_nd_max']:
                                Phi_HC_nd_act[Tstepi] = self.inputs['Phi_H_nd_max']
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
                            Phi_HC_nd_act[Tstepi] = self.inputs['Phi_C_nd_max'] * (Theta_op_set - Theta_int_op[Tstepi, 0]) / (
                                    Theta_int_op[Tstepi, colB_C] - Theta_int_op[Tstepi, 0])
                            if Phi_HC_nd_act[Tstepi] < self.inputs['Phi_C_nd_max']:
                                Phi_HC_nd_act[Tstepi] = self.inputs['Phi_C_nd_max']
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
        # Tstepn_act = 8760

        # simulation_calendar_year = weather_data.index[0].year  # arbitrary non-leap year
        # index = pd.date_range(start='2018-12-01', end='2020-01-01', freq='H', inclusive='left')
        hourly_results = pd.DataFrame(data=np.vstack((Phi_HC_nd_act[Tstep_first_act:], Theta_op_act[Tstep_first_act:], sim_df['T2m'][Tstep_first_act:])).T, index=sim_df[Tstep_first_act:].index,
                                    columns=['Q_HC', 'T_op', 'T_ext'])
        # hourly_results.drop(index=pd.date_range(start='2018-12-01', end='2019-01-01', freq='H', inclusive='left'),
        #                     inplace=True)  # remove warmup period
        hourly_results['Q_H'] = 0
        mask = hourly_results['Q_HC'] > 0
        hourly_results.loc[mask, 'Q_H'] = hourly_results.loc[mask, 'Q_HC'].astype('int64')
        hourly_results['Q_C'] = 0
        mask = hourly_results['Q_HC'] < 0
        hourly_results.loc[mask, 'Q_C'] = -hourly_results.loc[mask, 'Q_HC'].astype('int64')

        # heat pump: transformation of heating and cooling thermal load into electric load
        # C_to_K = 273.15
        # COP = 0.09756 * (hourly_results['T_ext'] + C_to_K) - 24.02587  # ref load: 45°C
        # EER = 52.81889 + (hourly_results['T_ext'] + C_to_K) * -0.16294  # ref load: 7°C
        # hourly_results['P'] = 0  # electric demand (power)
        # mask = hourly_results['Q_HC'] > 0  # heating
        # hourly_results.loc[mask, 'P'] = hourly_results.loc[mask, 'Q_H'] / COP[mask]
        # mask = hourly_results['Q_HC'] < 0  # cooling
        # hourly_results.loc[mask, 'P'] = hourly_results.loc[mask, 'Q_C'] / EER[mask]

        return hourly_results    
    



#%%
# ========================================================================
# TEST 

bui_item = {
        'url_api': "http://127.0.0.1:8000/api/v1", 
        'latitude':46.66345144066082,
        'longitude':9.71636944229362,
        'Eln':10, #
        'a_use': 100, 
        "slab_on_ground_area":100,#
        'H_setpoint':20,     
        'C_setpoint':26,
        'Phi_H_nd_max':30000,            
        'Phi_C_nd_max':-10000,
        'air_change_rate_base_value':1.107 ,
        'air_change_rate_extra':0.0,
        'internal_gains_base_value':1.452,
        'internal_gains_extra':0.0,        
        'H_tb' : 2.0,
        'H_setback':10,
        'C_setback':26,
        'TypeSub_eli': np.array(['W', 'OP', 'W', 'GR', 'OP', 'W', 'OP', 'OP', 'W', 'OP'],dtype=object), 
        'kappa_m_eli': np.array([0.0, 25000.0, 0.0, 279852.0, 25000.0, 0.0, 279852.0, 25000.0, 0.0, 25000.0], dtype=object), 
        'A_eli': np.array([1.0, 30.0, 10.0, 100.0, 30.0, 3.0, 100.0, 30.0, 6.0, 30.0],dtype=object), 
        'a_sol_eli': np.array([1.0, 0.6, 1.0, 0.0, 0.6, 1.0, 0.0, 0.6, 1.0, 0.6], dtype=object),
        'g_w_eli': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=object),
        'or_eli': np.array(['NV', 'NV', 'SV', 'HOR', 'SV', 'EV', 'HOR', 'EV', 'WV', 'WV'],dtype=object),
        'h_ci_eli': np.array([2.5, 2.5, 2.5, 0.7, 2.5, 2.5, 5.0, 2.5, 2.5, 2.5], dtype=object),
        'h_ri_eli': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
        'h_ce_eli': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
        'h_re_eli': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
        'U_eli': np.array([0.8, 0.8, 0.8, 0.4, 0.8, 0.8, 0.4, 0.8, 0.8, 0.8], dtype=object),
        'R_eli': np.array([1.25, 1.25, 1.25, 2.5, 1.25, 1.25, 2.5, 1.25, 1.25, 1.25],dtype=object), 
        'F_sk_eli': np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5], dtype=object), 
        'occ_level_wd': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
        'occ_level_we': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
        'comf_level_wd': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
        'comf_level_we': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
        'sog_area': 100.0, 
        'exposed_perimeter': 40.0, 
        'wall_thickness': 0.3, 
        'R_floor_construction': 2.5, 
        'baseline_hci': np.array([2.5, 2.5, 2.5, 0.7, 2.5, 2.5, 5.0, 2.5, 2.5, 2.5], dtype=object), 
        'baseline_hce': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object), 
        "occ_level_wd": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
        "occ_level_we": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
        "comf_level_wd": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
        "comf_level_we": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
        'coldest_month': 1, 
        'uuid': '9cdbfbeb-f2f7-467c-9a69-3e0cc3a181ee',
        "heating": True,
        "cooling": True,
        "construction_class": "class_i",
    }
# bui_item = {
#         'url_api': "http://127.0.0.1:8000/api/v1", 
#         'latitude':46.66345144066082,
#         'longitude':9.71636944229362,
        
#         'Eln':10, #
#         'a_use': 100, 
#         "slab_on_ground_area":100,#
#         'H_setpoint':20,     
#         'C_setpoint':26,
#         'Phi_H_nd_max':30000,            
#         'Phi_C_nd_max':10000,
#         'air_change_rate_base_value':1.107 ,
#         'air_change_rate_extra':0.0,
#         'internal_gains_base_value':1.452,
#         'internal_gains_extra':0.0,        
#         'H_tb' : 2.0,
#         'H_setback':10,
#         'C_setback':26,
#         'TypeSub_eli': np.array(['W', 'OP', 'W', 'GR', 'OP', 'W', 'OP', 'OP', 'W', 'OP'],dtype=object), 
#         'kappa_m_eli': np.array([0.0, 25000.0, 0.0, 279852.0, 25000.0, 0.0, 279852.0, 25000.0, 0.0, 25000.0], dtype=object), 
#         'A_eli': np.array([1.0, 30.0, 10.0, 100.0, 30.0, 3.0, 100.0, 30.0, 6.0, 30.0],dtype=object), 
#         'a_sol_eli': np.array([1.0, 0.6, 1.0, 0.0, 0.6, 1.0, 0.0, 0.6, 1.0, 0.6], dtype=object),
#         'g_w_eli': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=object),
#         'or_eli': np.array(['NV', 'NV', 'SV', 'HOR', 'SV', 'EV', 'HOR', 'EV', 'WV', 'WV'],dtype=object),
#         'h_ci_eli': np.array([2.5, 2.5, 2.5, 0.7, 2.5, 2.5, 5.0, 2.5, 2.5, 2.5], dtype=object),
#         'h_ri_eli': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
#         'h_ce_eli': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
#         'h_re_eli': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
#         'U_eli': np.array([0.8, 0.8, 0.8, 0.4, 0.8, 0.8, 0.4, 0.8, 0.8, 0.8], dtype=object),
#         "R_eli": np.array([1.25, 1.25, 1.25, 2.5, 1.25,1.25,1.25,2.5,1.25,1.25,1.25], dtype=object),
#         'R_c_eli': np.array([1.25, 1.25, 1.25, 2.5, 1.25, 1.25, 2.5, 1.25, 1.25, 1.25],dtype=object), 
#         'F_sk_eli': np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5], dtype=object), 
#         'occ_level_wd': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     dtype=object), 
#         'occ_level_we': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     dtype=object), 
#         'comf_level_wd': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     dtype=object), 
#         'comf_level_we': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     dtype=object), 
#         'sog_area': 100.0, 
#         'exposed_perimeter': 40.0, 
#         'wall_thickness': 0.3, 
#         'R_floor_construction': 2.5, 
#         'baseline_hci': np.array([2.5, 2.5, 2.5, 0.7, 2.5, 2.5, 5.0, 2.5, 2.5, 2.5], dtype=object), 
#         'baseline_hce': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object), 
#         "occ_level_wd": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
#         "occ_level_we": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
#         "comf_level_wd": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
#         "comf_level_we": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 0],
#         'coldest_month': 1, 
#         'uuid': '9cdbfbeb-f2f7-467c-9a69-3e0cc3a181ee',
#         "heating": True,
#         "cooling": True,
#         "construction_class": "class_i",
#     }

#%%


#%%
# TEST:
# inizialize_weather_class = __ISO52010__(46.66345144066082, 9.71636944229362, bui_item)
# hourly_sim = __ISO52016__(bui_item).Temperature_and_Energy_needs_calculation()
# hourly_sim
# inizialize.Temperature_and_Energy_needs_calculation(**bui_item)

# ==========================================================================

# %%
# import plotly.graph_objects as go

# # Create trace for variable 1
# trace1 = go.Scatter(x=hourly_sim.index, y=hourly_sim['Q_H'], mode='lines', name='Variable 1')

# # Create layout
# layout = go.Layout(title='Time Series Chart',
#                    xaxis=dict(title='Timestamp'),
#                    yaxis=dict(title='Value'))

# # Create figure
# fig = go.Figure(data=[trace1], layout=layout)

# # Show the plot
# fig.show()
# %%
