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
from pybuildingenergy.constants import main_directory_
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

# ====================================================================================================
#                                       DHW Table of ISO 12831
# ====================================================================================================

# Hourly breakdown of relative demand for hot water by volume. Table B.1
table_B_1 = pd.DataFrame(
    {
        "XXS" : [0, 0, 0,0, 0, 0, 0, 10, 5, 5, 0, 10, 15, 0, 0, 0, 0, 0, 15, 10, 10, 20, 0, 0],
        "XS" : [0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0],
        "S" : [0, 0, 0, 0, 0, 0, 0, 10, 5, 5, 0, 10, 15, 0, 0, 0, 0, 0, 10, 0, 20, 25, 0, 0],
        "M": [0, 0, 0, 0, 0, 0, 0, 27.5, 7.2, 3.6, 1.8, 3.6, 5.4, 0, 1.8, 1.8, 1.8, 0, 5.4, 1.8, 12.6, 25.7, 0, 0],
        "L": [0, 0, 0, 0, 0, 0, 0, 14.7, 33.6, 1.8, 0.9, 1.8, 2.7, 0, 0.9, 0.9, 0.9, 0, 2.7, 0.9, 6.3, 31.8, 0, 0],
        "XL": [0, 0, 0, 0, 0, 0, 0, 33.8, 2.2, 1.1, 1.1, 1.7, 3.9, 0, 0.6, 1.1, 1.1, 0.6, 1.7, 0.6, 27, 23.7, 0, 0 ],
        "XXL": [0, 0, 0, 0, 0, 0, 0, 33.7, 1.7, 0.9, 0.9, 1.3, 3, 0, 0.4, 0.9, 0.9, 0.4, 1.3, 0.4, 28.4, 25.9, 0, 0]
     },
     index = [list(range(1,25))]
)

#Hourly distribution of relative volume-based hot water demand across various building categories. Table B.2
table_B_2 = pd.DataFrame(
    {
        "single_family_dwelling" : [1.8, 1, 0.6, 0.3, 0.4, 0.6, 2.4, 4.7, 6.8, 5.7, 6.1, 6.1, 6.3, 6.4, 5.1, 4.4, 4.3, 4.7, 5.7, 6.5, 6.6, 5.8, 4.5, 3.1],
        "appartment dwelling" : [1, 1, 1, 0, 0, 1, 3, 6, 8, 6, 5, 5, 6, 6, 5, 4, 4, 5, 6, 7, 7, 6, 5, 2],
        "residential_home_for_the_elderly" : [0.3, 0.3, 0.4, 0.7, 1.0, 1.8, 9.3, 15.7, 8.1, 7.5, 7.0, 6.6, 7.1, 5.1, 3.8, 3.3, 4.1, 2.9, 6.1, 4.1, 1.4, 1.8, 0.9, 0.4],
        "student_residence": [1.4, 1.0, 0.5, 0.6, 1.3, 3.4, 5.8, 5.8, 6.2, 5.4, 5.1, 4.7, 4.2, 4.5, 4.1, 4.3, 5.3, 6.0, 6.6, 6.0, 5.6, 5.4, 3.9, 2.8],
        "Hospital": [0.4, 0.4, 0.5, 0.8, 1.2, 2.8, 7.5, 10.5, 8.0, 7.5, 7.5, 7.0, 7.5, 5.5, 4.3, 3.7, 4.5, 3.2, 7.0, 4.5, 2.0, 2.0, 1.2, 0.5]
     },
     index = [list(range(1,25))]
)

#  Net energy demand for water heating per day. Table B.3  
table_B_3 = pd.DataFrame({
    'type_of_usage' : ['Office_buildings', "Hospital ward or patient's room", 
                       'School_without_showers', 'School_with_shower',
                       'Retail shop/department store', 'Workshop, industrial facility (for washing snd showering)',
                       'Modest hotel', 'Medium-class hotel',
                       'Luxury-class hotel', 'Restaurant, inn/pub',
                       'Home(for the aged, orphanage, etc.)', 'Barracks',
                       'Sport faciltiy with showers', 'Commercial catering kitchen',
                       'Bakery', 'Hairdresser/barber',
                       'Butcher with production',
                       'Laundry', 'Brewery', 'Dairy'],
    'Usage' : ['person', 'bed', 'person', 'person', 'employee', 'employee', 
               'bed', 'bed', 'bed', 'seat', 'person', 'person', 'person', 'meal',
               'employee', 'employee', 'employee', '100 Kg laundry', '100 liters beer', 
               '100 liters of milk'],
    'Usage dependent' : [0.4, 8, 0.5, 1.5, 1, 1.5, 1.5, 4.5, 7, 1.5, 3.5, 1.5, 1.5, 0.4, 5, 8, 18, 20, 15, 10],
    'Area specific - Wh/m2d': [30, 530, 170, 500, 10, 75, 190, 450, 580, 1250, 230, 150, None, None, None, None, None, None, None, None],
    'Reference Area': ['office floor area', 'wards and room', 'classrooms', 'classrooms',
                       'sales areas', 'area of workshop/works', 'hotel bedrooms',
                       'hotel bedrooms', 'hotel bedrooms', 'public rooms', 'rooms', 'rooms', 
                       None, None, None, None, None, None, None, None], 

})

# Values for the calculation of domestic hot watrer requirements per day
table_B_4 = pd.DataFrame(
    {
    'type_of_activity'  : [
        'Accomodation', 'Health establishment wihtout accomodation', 
        'Health establishment without accomodation',
        'Health establishment without accomodation - without laundry',
        'Catering, 2 meals per day. Traditional cusine',
        'Catering, 2 meals per day. Self service', 
        'Catering, 1 meals per day. Tradional cusine', 
        'Catering, 1 meals per day. Self service',
        'Hotel, 1-star without laundry', 'Hotel, 1-star withlaundry', 
        'Hotel, 2-star without laundry', 'Hotel, 2-star withlaundry', 
        'Hotel, 3-star without laundry', 'Hotel, 3-star withlaundry', 
        'Hotel, 4-star and GC without laundry', 'Hotel, 4-star and GC withlaundry', 
        'Sport_establishment'
        ],
    'V_W_f_day' : [28, 10, 56, 88, 21, 8, 10, 4, 56, 70, 76, 90, 97, 111, 118, 132, 101 ], # [l/d]
    'f' : ['number of beds'] * 4 + ['number of guest per meal'] * 4 + ['number of beds'] * 8 + ['number_of_showers_installed']
    }
)

# values for calculation of domestic hot water requirements per day
table_B_5_Standard = pd.DataFrame({
    'type_pf_building': ['residential_buildings (simple_housing)', 'residential_buildings (luxury_building)',
                         'single_family_dwellings', 'apartment_dwellings'],
    'V_W_p_day':['25-60','60-100', '40-70', '25-30']
})

# values for calculation of domestic hot water requirements per day as avergae of the values from table_B_5_standard
table_B_5_modified = pd.DataFrame({
    'type_of_building': [
        'residential_building - simple housing - MIN', 
        'residential_building - simple housing - AVG', 
        'residential_building - simple housing - MAX', 
        'residential_building - luxury housing - MIN', 
        'residential_building - luxury housing - AVG', 
        'residential_building - luxury housing - MAX', 
        'single_family_dwellings - MIN', 
        'single_family_dwellings - AVG', 
        'single_family_dwellings - MAX', 
        'apartments_dwellings - MIN', 
        'apartments_dwellings - AVG', 
        'apartments_dwellings - MAX', 
        ],
    'liters/person_per_day':[25, 45, 60, 60, 80, 100, 40, 55, 70, 25, 27.5, 30]
})