__author__ = "Daniele Antonucci, Ulrich Filippi Oberegger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberegger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"

# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from pybuildingenergy.global_inputs import main_directory_
import pickle


# ===========================================================================================
def is_float(value):
    """
    Check if a value is float
    """
    try:
        float_value = float(value)
        return isinstance(float_value, float)
    except ValueError:
        return False


def capitalize_first_letter(s):
    """
    Capitalize first letter of string
    """
    if not s:
        return s
    return s[0].upper() + s[1:]


# ===========================================================================================
def Equation_of_time(day_of_year: pd.Series) -> pd.Series:
    """
    Equation of time according to formula 6.4.1.2 of ISO52010

    :param day_of_year: series of days in a year
    :return **t_eq**: equation of time
    """
    t_eq = 0.45 * (day_of_year - 359)
    mask = day_of_year < 21
    t_eq[mask] = 2.6 + 0.44 * day_of_year[mask]
    mask = (day_of_year >= 21) & (day_of_year < 136)
    t_eq[mask] = 5.2 + 9 * np.cos((day_of_year[mask] - 43) * 0.0357)
    mask = (day_of_year >= 136) & (day_of_year < 241)
    t_eq[mask] = 1.4 - 5 * np.cos((day_of_year[mask] - 135) * 0.0449)
    mask = (day_of_year >= 241) & (day_of_year < 336)
    t_eq[mask] = -6.3 - 10 * np.cos((day_of_year[mask] - 306) * 0.036)

    return t_eq


def Hour_angle_calc(solar_time: pd.Series) -> pd.Series:
    """
    Equation of time according to formula 6.4.1.5 of ISO52010

    :param solar_time: defined by formula 6.4.1.4 of ISO52010

    .. note::
        South is 0, East pos; (because of shading sectors: keep between -180 and + 180)

    :return **hour_angle_deg** hourly angle degrees
    
    """
    hour_angle_deg = 180 / 12 * (12.5 - solar_time)
    hour_angle_deg[hour_angle_deg > 180] -= 360
    hour_angle_deg[hour_angle_deg < -180] += 360

    return hour_angle_deg


def Air_mass_calc(solar_altitude_angle: pd.Series) -> pd.Series:
    """
    Calculation of air mass

    :param solar_altitude_angle: the angular distance between the rays of Sun and the horizon of the Earth

    :return **air_mass**

    """
    solar_altitude_angle_deg = np.degrees(solar_altitude_angle)
    air_mass = 1 / (
        np.sin(solar_altitude_angle)
        + 0.15 * np.power(solar_altitude_angle_deg + 3.885, -1.253)
    )
    mask = solar_altitude_angle_deg >= 10
    air_mass[mask] = 1 / np.sin(solar_altitude_angle[mask])

    return air_mass


def Get_positions(lst, value):
    """
    Ger position of a value
    """
    positions = []
    for i, item in enumerate(lst):
        if item == value:
            positions.append(i)
    return positions


def Filter_list_by_indices(lst, indices):
    """
    Filter a list by indices
    """
    return [lst[i] for i in indices]


def Perimeter_from_area(area, side):
    """
    Perimeter from area assuming 10m of side

    :param area: gross surface area of the slab on ground floor in [m]
    :param side: side of a rectangular shape of the building
    """
    base = area / side
    perimeter = 2 * (base + side)
    return perimeter


def Area_roof(leng_small, leng_roof):
    """
    Calculation of the roof area. According to the following formula:

    formula = ((length_small*cos(28)+0.5*cos(28))*(length_roof+(0.5*2)))*2

    **where**:

    cos(28°) = 0.88

    .. note:: 
        The calculation assumes that the building has a rectangular floor plan.

    :param leng_small: small length of roof side in [m]
    :param leng_roof: length of roof side in [m]

    """
    Area = 2 * ((leng_small * 0.88) + (0.5 * 0.88)) * (leng_roof + (0.5 * 2))
    return Area


def Internal_gains(bui_type: str, area: float):
    """
    Calculation of internal gains according to the building typology

    :param bui_type: type of building. Possible choice: residential, office, commercial
    :param area: gross area of the building

    :return: int_gains: value of internal gains in W/m2

    .. note::
        Only for residential building the data is avaialble.
        The result is a sum of contribution given by equipments, people heat and lights
    """
    if bui_type == "residential":
        int_gains = (120 + 100 + 2000 + 800 + 5 + 4 * 100) / area + 5 * 5

    else:
        int_gains = 5
    return int_gains


def Power_heating_system(bui_volume, bui_class):
    """
    Approssimative calculation of generator power, according to the following formula:
    p = Voume[m3] * energy needs[kW/m3]

    :param bui_class: could be:
        * *'old'*: No or very low insulated building
        * *'new'*: very well insulated building
        * *'average'*:medium insulated building

    :return heat_power: power of generator [W]
    """
    if bui_class == "old":
        p = bui_volume * 0.12
    elif bui_class == "gold":
        p = bui_volume * 0.03
    else:
        p = bui_volume * 0.05

    return p * 1000


def Check_area(elements_area: list):
    """
    Check the areas of the opaque and transparent components for the same orientation.
    If the opaque component is smaller than the transparent one, then the two areas are equalized.
    If the area of the floor slab on ground is higher than the area of the roof, that the two areas are equalized.

    The check follows the order of entry of the component as per the following reference.
    ["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"]
    ['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV']

    **where**:

    * OP: Opaque elements
    * GR: Ground Floor
    * W: Transaprent elements
    * NV: Vertical North
    * SV: South Vertical
    * EV: East Vertical
    * WV: West Vertical
    * HOR: Horizontal for ground floor(HOR-GR), Horizontal or Slope for Roof(HOR-OP)

    :param elements_area: List of areas for each individual facade element.

    :return check_el_area: list of corrected areas
    """
    check_el_area = elements_area.copy()
    for i in range(4):
        area_wall = elements_area[i]
        area_window = elements_area[i + 6]
        if area_wall < area_window:
            check_el_area[i] = area_window

    area_floor = check_el_area[4]
    area_roof = check_el_area[5]
    if area_roof < area_floor:
        check_el_area[4] = area_roof

    return check_el_area


#
# df_101_heating_hdd = df_101.loc[:,['energy_consumption_m3','heating_degree_days']]
# data_101 = [list(tuple(x)) for x in df_101_heating_hdd.itertuples(index=False, name=None)]


def Heating_Degree_days(out_temp: list, base_temperature: int) -> list:
    """
    Getting heating degree days from daily outdoor temperature


    :param base_temperature: outoddor base temperature to be used as limit for calculation of Heating Degree days.
            e.g.  18  in Europe
            e.g   20  in Italy
    :return HDD: Heating Degree days
    """

    HDD = []
    for temp in out_temp:
        if temp < base_temperature:
            HDD_value = base_temperature - temp
        else:
            HDD_value = 0

        HDD.append(HDD_value)
    return HDD


def check_dict_format(**kwargs):
    for arg_name, arg_value in kwargs.items():
        if not isinstance(arg_value, dict):
            raise TypeError(f"Argument '{arg_name}' must be a dictionary.")
    # If all inputs are in dictionary format, return True or do other operations
    return True


# ==========
def ePlus_shape_data(ep_result: pd.DataFrame, A_use: float):
    """
    Reshape dataset from energyplus:

    :param ep_result: dataframe of energy plus result
    :param A_use: net useful area in m2

    :return
        * **EnergyPlus_monthly_heating_in_kWh_per_sqm**: Monthly heating consumption of the building in kWh per m2
        * **EnergyPlus_monthly_cooling_in_kWh_per_sqm**: Monthly cooling consumption of the building in kWh per m2
        * **EnergyPlus_annual_heating_in_kWh_per_sqm**: Annual heating consumption of the building in kWh per m2
        * **EnergyPlus_annual_cooling_in_kWh_per_sqm**: Annual cooling consumption of the building in kWh per m2
    """

    ep_hourly_heating_in_J = ep_result[
        "ZONE ONE:Zone Air System Sensible Heating Energy [J](Hourly)"
    ]
    ep_hourly_cooling_in_J = ep_result[
        "ZONE ONE:Zone Air System Sensible Cooling Energy [J](Hourly) "
    ]  # Note: for some reason, EnergyPlus saves last output in csv with an extra space :-(
    # ep_hourly_T_op = ep_result['ZONE ONE:Zone Operative Temperature [C](Hourly)']

    month_start_hours = [
        0,
        744,
        1488,
        2160,
        2904,
        3624,
        4368,
        5088,
        5832,
        6576,
        7296,
        8040,
        8760,
    ]

    ep_monthly_heating_in_J = np.array(
        [
            ep_hourly_heating_in_J[
                month_start_hours[i] : month_start_hours[i + 1]
            ].sum()
            for i in range(12)
        ]
    )
    ep_monthly_cooling_in_J = np.array(
        [
            ep_hourly_cooling_in_J[
                month_start_hours[i] : month_start_hours[i + 1]
            ].sum()
            for i in range(12)
        ]
    )

    ep_annual_heating_in_J = ep_hourly_heating_in_J.sum()
    ep_annual_cooling_in_J = ep_hourly_cooling_in_J.sum()

    ep_annual_heating_in_kWh = ep_annual_heating_in_J / 3.6e6
    ep_annual_cooling_in_kWh = ep_annual_cooling_in_J / 3.6e6

    ep_monthly_heating_in_kWh = ep_monthly_heating_in_J / 3.6e6
    ep_monthly_cooling_in_kWh = ep_monthly_cooling_in_J / 3.6e6

    # ep_monthly_T_op = np.array([ep_hourly_T_op[month_start_hours[i]:month_start_hours[i + 1]].mean() for i in range(12)])

    EnergyPlus_annual_heating_in_kWh_per_sqm = ep_annual_heating_in_kWh / A_use
    EnergyPlus_annual_cooling_in_kWh_per_sqm = ep_annual_cooling_in_kWh / A_use

    EnergyPlus_monthly_heating_in_kWh_per_sqm = ep_monthly_heating_in_kWh / A_use
    EnergyPlus_monthly_cooling_in_kWh_per_sqm = ep_monthly_cooling_in_kWh / A_use

    return (
        EnergyPlus_monthly_heating_in_kWh_per_sqm,
        EnergyPlus_monthly_cooling_in_kWh_per_sqm,
        EnergyPlus_annual_heating_in_kWh_per_sqm,
        EnergyPlus_annual_cooling_in_kWh_per_sqm,
    )


def get_buildings_demos():
    """
    Get archetypes and demo buildings
    """
    # pickle_file_path = main_directory_ + "/pybuildingenergy/pybuildingenergy/data/archetypes.pickle"
    pickle_file_path = main_directory_ + "/archetypes.pickle"
    with open(pickle_file_path, "rb") as f:
        archetypes = pickle.load(f)

    return archetypes


def Simple_regeression(x_data: list, y_data: list, x_data_name: str):
    """
    Simple regression between two variable

    :param x_data: data on x-axes
    :param y_data: data on y-axes

    :return
        * **R2**: coefficinet of determination
        * **equation**: linear regression equation
        * **residual**: residual !!! *Not yet included* !!!
    
    """
    #
    # Sample data
    x = np.array(x_data).reshape(-1, 1)  # Example independent variable
    y = np.array(y_data)  # Example dependent variable

    # Fit linear regression model
    model = LinearRegression().fit(x, y)

    # Predict y values using the fitted model
    y_pred = model.predict(x)

    # Calculate R-squared
    r2 = r2_score(y, y_pred)

    # Get the slope (coefficient) and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_

    # Print R-squared and equation of the regression line
    # print("R-squared:", r2)
    # print("Equation of the regression line: y =", slope, "* x +", intercept)
    equation = f"y ={round(intercept,2)} + {x_data_name}*{round(slope,2)}"

    return (round(r2, 2), equation)

# ============================================================================================================
#                           CALIBRATION PROCESS
# ============================================================================================================



# def ISO52016_calibration_objective_function_monthly_heating_PSO(x, building_object, real_months_values, tmy_data):
#     '''
#     Objective function:
#     adjusted root mean sqaure deviation
#     '''
#     inputs_for_simulation = building_object.copy()
#     new_input_hci = (x.T[0].reshape(-1,1)* inputs_for_simulation['heat_convective_elements_internal'])[0]
#     new_input_hce = (x.T[1].reshape(-1,1)* inputs_for_simulation['heat_convective_elements_external'])[0]

#     inputs_for_simulation['h_ci_eli'] = new_input_hci
#     inputs_for_simulation['h_ce_eli'] = new_input_hce
#     print(inputs_for_simulation['h_ci_eli'], inputs_for_simulation['h_ce_eli'])

#     # GET HOURLY RESULT OF ISO 52016 SIMULATION FOR HEATING
#     hourly_results, annual_results_df = bep_calc(inputs_for_simulation,tmy_data)
#     # print(hourly_results)
#     ISO52016_monthly_heating_in_kWh_per_sqm = hourly_results['Q_H'].resample('M').sum() / (1e3 * inputs_for_simulation['A_use'])
#     print(ISO52016_monthly_heating_in_kWh_per_sqm)
#     # ISO52016_monthly_heating_in_kWh_per_sqm = hourly_results['Q_H'].resample('M').sum() / 1000/ inputs_for_simulation['A_use']
   
#     # ENERGY PLUS
#     n = 12 # number of real_months_values
#     obj_funct = math.sqrt(sum((real_months_values - ISO52016_monthly_heating_in_kWh_per_sqm.to_numpy())**2)/(n-1))
#     print(obj_funct)

#     return obj_funct

# def PSO_optimizer(inputs, tmy_data, monitored_monthly_data, number_iter, ftol_PSO, ftol_iter, bounds):
#     options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}    
    
#     optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds, 
#                           ftol=ftol_PSO, ftol_iter=ftol_iter)

#     cost, pos = optimizer.optimize(ISO52016_calibration_objective_function_monthly_heating_PSO, 
#                                 iters= number_iter,
#                                 inputs= inputs, 
#                                 real_months_values= monitored_monthly_data,
#                                 tmy_data= tmy_data)
#     return cost, pos 

# def RMSE_calculation(actual, calibrated_monthly_data, not_calibrated_monthly_data):
#     '''
#     Calcualte the improvement of the calibration in %, considering the RMSE of the calibrated and not calibrated data
#     Parameters:
#     -----------
#     actual: monitored monthly real data of the building in kWh/m2
#     calibrated_monthly_data: data calibrated by the PSO optimizartion using the calculation of the enrgy consumption according to ISO 52016 in kWh/m2
#     not_calibrated_monthy_data: calculation of monthly data according to the ISO 52016
    
#     Returns:
#     --------
#     1. RMSE of calibrated [0] and not calibrated model [1]
#     2. Percentage of improvement [2]
#     '''
    
#     #define Actual and Predicted Array
#     pred_not_calibrated = np.array(not_calibrated_monthly_data)
#     pred_calibrated = np.array(calibrated_monthly_data)

#     #Calculate RMSE
#     result_calibrated = sqrt(mean_squared_error(actual,pred_calibrated))
#     result_not_calibrated = sqrt(mean_squared_error(actual,pred_not_calibrated))
#     improvement = round((result_not_calibrated - result_calibrated)/result_not_calibrated * 100,2)   
    
#     # PRINT 
#     print("RMSE - not calibrated:",  result_not_calibrated,  "RMSE - calibrated:", result_calibrated)
#     print("Improvement %: ",improvement)
    
#     return (round(result_calibrated,2), round(result_not_calibrated,2), improvement)



# def runCalib(inputs:dict,monthly_real_data:list, bounds, number_iter, ftol_PSO, ftol_iter):

    
#     # DATA from ISO 
#     tmy_data = get_tmy_data(inputs['latitude'], inputs['longitude'])
#     hourly_data, yearly_data = bep_calc(inputs_data=inputs, year_weather=2015, tmy_data=tmy_data)
#     not_calibrated_monthly_data = hourly_data['Q_H'].resample('M').sum() / (1e3 * inputs['A_use'])
    
#     # CALIBRATION USING PSO
#     # Real data
#     monitored_monthly_data = np.array(monthly_real_data, dtype=object)
    
#     # PSO CALIBRATION
#     bounds = (np.array([0.4,0.4]), np.array([2,2]))
#     ftol_PSO = 0.3
#     ftol_iter = 100
#     options = {'c1': 0.7, 'c2': 0.5, 'w':1}
#     number_iter = 50

#     optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds, 
#                         ftol=ftol_PSO, ftol_iter=ftol_iter)

#     cost, options = optimizer.optimize(ISO52016_calibration_objective_function_monthly_heating_PSO, 
#                                 iters = number_iter,
#                                 inputs= inputs, 
#                                 real_months_values= monitored_monthly_data,
#                                 tmy_data= tmy_data)
#     # cost, options = PSO_optimizer(inputs, tmy_data, monitored_monthly_data,
#     #                               number_iter, ftol_PSO, ftol_iter, bounds=bounds)

#     # CALIBRATION MODEL
#     inputs_calibrated = inputs.copy()
#     inputs_calibrated['h_ci_eli'] = options[0]*inputs['h_ci_eli']
#     inputs_calibrated['h_ce_eli'] = options[1]*inputs['h_ce_eli']
#     calibrated_hourly_result, calibrated_yearly_result = bep_calc(inputs_data = inputs_calibrated, year_weather=2015, tmy_data=tmy_data)
#     calibrated_monthly_data = calibrated_hourly_result['Q_H'].resample('M').sum() / (1e3 * inputs['A_use'])

#     # ASSESSING result
#     RMSE_value= RMSE_calculation(monthly_real_data,not_calibrated_monthly_data, calibrated_monthly_data )
    
#     # PLOT 
#     df_energy = pd.DataFrame({
#         'calibrated_data': calibrated_monthly_data.reset_index(drop=True).to_list(),
#         'real_data': monitored_monthly_data,
#         'not_calibrated_data': not_calibrated_monthly_data
#     })

#     return df_energy, RMSE_value, inputs_calibrated
