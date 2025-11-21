'''
Calculation of DHW needs according to the ISO 12831-3
The aim is to evalute and demostrate the validity of the calculation proposed by the ISO.
This work doesn't replace the standard but it should be used along with the EPB standard.

Acknowledgments: The work was developed using the regulations and results obtained from the spreadsheet created by the EPBCenter.

Authour: Daniele Antonucci
'''

# ======================================================================================================
#                               Energy Need curve
# ======================================================================================================

import pandas as pd
import calendar
from datetime import date
from typing import Optional


from pybuildingenergy.source.functions import table_B_3, table_B_4, table_B_5_modified
from pybuildingenergy.global_inputs import WATER_DENSITY, WATER_SPECIFIC_HEAT_CAPACITY

# ================================================================================
#                           CALENDAR RESOLUTION (by nation name)
# ================================================================================

def get_calendar_by_name(nation_name: str):
    """
    Return a Workalendar country instance given a human-readable nation name,
    e.g. 'Italy'. Uses the registry when available; falls back to a small map.
    """
    try:
        from workalendar.registry import registry
        cal_cls = registry.get(nation_name)
        if cal_cls is None:
            raise KeyError(nation_name)
        return cal_cls()
    except Exception:
        # Fallback for common EU countries (extend as needed)
        try:
            from workalendar.europe import Italy, Germany, France, Spain, Austria, Switzerland
            fallback = {
                "Italy": Italy,
                "Germany": Germany,
                "France": France,
                "Spain": Spain,
                "Austria": Austria,
                "Switzerland": Switzerland,
            }
            return fallback[nation_name]()
        except Exception as e:
            raise ValueError(
                f"Cannot resolve calendar for nation '{nation_name}'. "
                "Ensure 'workalendar' is installed and the name is valid."
            ) from e


def generate_daily_calendar(year: int, month: int, country_calendar) -> dict:
    """Generates daily status (Holiday, Working, Non-Working) for a given month/year."""
    daily_calendar = {}
    holidays = [h[0] for h in country_calendar.holidays(year)]
    for day in range(1, 32):
        try:
            current_date = date(year, month, day)
        except ValueError:
            break  # stop when month runs out of days
        if current_date in holidays:
            status = "Holiday"
        elif country_calendar.is_working_day(current_date):
            status = "Working"
        else:
            status = "Non-Working"
        daily_calendar[current_date.strftime("%Y-%m-%d")] = status
    return daily_calendar


def generate_calendar(nation_name: str, year: int) -> pd.DataFrame:
    """
    Create a calendar DataFrame for the full year for a given nation name.
    Returns columns: ['days' (Timestamp), 'values' in {'Working','Non-Working','Holiday'}].
    """
    cal = get_calendar_by_name(nation_name)

    rows = []
    for month in range(1, 13):
        monthly = generate_daily_calendar(year, month, cal)
        for day, val in sorted(monthly.items()):
            rows.append((pd.to_datetime(day), val))

    df = pd.DataFrame(rows, columns=["days", "values"]).dropna(subset=["days"])
    df = df.sort_values("days").drop_duplicates(subset=["days"], keep="last").reset_index(drop=True)

    # Ensure full coverage (if anything missing)
    full_range = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    existing = set(df["days"].dt.normalize())
    missing = [d for d in full_range if d.normalize() not in existing]
    if missing:
        def label_for(d):
            return "Working" if cal.is_working_day(d) else "Holiday"
        add_rows = [(d, label_for(d)) for d in missing]
        df = pd.concat([df, pd.DataFrame(add_rows, columns=["days", "values"])], ignore_index=True)
        df = df.sort_values("days").reset_index(drop=True)
    return df



def calc_V_w_day_trough_V_w_p_day(method: Optional[str] = None,
                                  building_area: Optional[float] = None,
                                  building_type: Optional[str] = None,
                                  V_w_p_day: Optional[float] = None):
    '''
    for single family dwellings and apartments dwellings the V_W_p_day is calcualted based on the number of equivalent persons (adults)

    :param mode: two methods can be used: 
        1) 'number_of_person' providing the number of person' 
        2) 'building_area': providing the area of the building 
        3) 'number_of_units': provinding specific units according to the table B_4
    :param method: possible selection: 
        1) 'correlation': using correlation of B.5
        2) 'table': using table of B.5
    :param building_area:  gross building area [m2]
    :param building_type:  type of building, possible choice: 'Single_family_house', 'Attached_house', 'Dwelling'
    :param num_person: number of person inhabiting the dwelling
    :param V_w_f_day: value taken from the dataframe "table_b_5"
    :param V_w_p_day: value of liters of DHW per person taken from table b_5 modified

    '''

    if building_type in {'Single_family_house', 'Attached_house'}:
        if building_area < 30: 
            n_p_eq_max = 1
        elif 30 <=building_area <= 70:
            n_p_eq_max = 1.775-0.01875*(70-building_area)
        else:
            n_p_eq_max = 0.025*building_area

    elif building_type == 'Dwelling':
        if building_area < 10: 
            n_p_eq_max = 1
        elif 10 <=building_area <= 50:
            n_p_eq_max = 1.775-0.01875*(50-building_area)
        else:
            n_p_eq_max = 0.035*building_area
    else: 
        raise ValueError("Error")

    if n_p_eq_max < 1.75:
        np_eq = n_p_eq_max
    else: 
        np_eq = 1.75+0.3*(n_p_eq_max -1.75)

    if  method == 'correlation':
        # For these residential cases and at the level of one dwelling, requirements can be expressd by :
        V_w_nd_ref = max(40.71, (3.26*building_area/np_eq))/1000 #[m3]
    elif method == 'table':
        V_w_nd_ref = V_w_p_day*np_eq/1000
    else:
        raise ValueError("Invalid method. Choose either 'correlation' or 'table'.")

    return V_w_nd_ref



# ======================================================================
'''
Calculation based on usage and the floor area
'''

def get_days(year):
    """Calculate the number of days per month for a given year excluding days from adjoining months represented as '0'.
    
    Args:
    :param: year (int): The year for which to calculate days per month.

    Returns:
        list: A list containing the number of valid days for each month of the specified year.
    """
    days_per_month = []
    for month in range(1, 13):  # Looping through each month (1 to 12)
        # Getting the week list for each month
        month_weeks = calendar.monthcalendar(year, month)
        # Counting days excluding '0' which represents days of the adjoining month
        days_count = sum(day != 0 for week in month_weeks for day in week)
        days_per_month.append(days_count) 

    return days_per_month



def Volume_and_energy_DHW_calculation(
    n_workdays: int,
    n_weekends: int,
    n_holidays: int,
    sum_fractions: pd.DataFrame,
    total_days: int,
    hourly_fractions: pd.DataFrame,
    teta_W_draw: float,
    teta_w_c_ref: float,
    teta_w_h_ref: float,
    teta_W_cold: float,
    mode_calc: str,
    building_type_B3: str,
    building_area: float,
    unit_count: int,
    building_type_B5: str,
    residential_typology: str,
    calculation_method: str,
    year: int,
    country_calendar: pd.DataFrame
    ):
    '''
    Calculate the daily, monthly, and yearly energy and volume needs for Domestic Hot Water (DHW)
    based on the building parameters and usage.

    Param
    ------
    :param: n_workdays (int): Number of workdays in the year.
    :param: n_weekends (int): Number of weekend days in the year.
    :param: n_holidays (int): Number of holidays in the year.
    :param: sum_fractions (DataFrame): Sum of hourly usage fractions.
    :param: total_days (int): Total number of days in the year.
    :param: hourly_fractions (DataFrame): Hourly usage fractions for different day types.
    :param: teta_W_draw: Water temperature of the mixed (cold nad hot) water drawn at the tap
    :param: teta_w_c_ref:
    :param: teta_w_h_ref:
    :param: teta_W_cold: cold water temperature
    :param: mode (str): calculation mode to be used to get the volume and energy for DHW:
        1. 'Area': using the area of the building
        2. 'number_of_units': number of units according to table B4, B5 
        3. 'volume_type_bui': if building is 'Single_family_house', 'Attached_house', 'Dwellings' the calcuation
            is based on the number of equivalent persons (adults)
    :param: building_type_B3 (str): Building usage type as defined in table B3.
    :param: building_area (float): Building area in square meters.
    :param: unit_count (int): Number of units (e.g., beds, rooms).
    :param: building_type_B5 (str): Building type as defined in table B5.
    :param: residential_typology (str): Specific residential typology.
    :param: calculation_method (str): Method used for calculations ('correlation', etc.).
    :param: year (int): Year for which the calculations are performed.
    :param: country_calendar (DataFrame): Calendar with 'values' column providing
        'Working', 'Non-Working', or 'Holiday' labels for each day.

    Return
    -------
        tuple: Contains detailed DHW needs calculations including yearly consumption,
            monthly and yearly volume, and detailed hourly breakdowns.
    '''
    
    if not isinstance(country_calendar, pd.DataFrame) or "values" not in country_calendar.columns:
        raise TypeError("country_calendar must be a DataFrame with a 'values' column.")

    if building_type_B5 in {'Single_family_house', 'Attached_house', 'Dwelling'}:
        selection_B5 = table_B_5_modified.loc[
            table_B_5_modified['type_of_building'] == residential_typology, :
        ]
        if selection_B5.empty:
            raise ValueError(
                f"Residential typology '{residential_typology}' not found in table_B_5_modified."
            )
        if len(selection_B5) > 1:
            raise ValueError(
                f"Multiple entries found for residential typology '{residential_typology}' in table_B_5_modified."
            )
        liters_per_person = float(selection_B5.iloc[0]['liters/person_per_day'])
        V_nd_d_ref = calc_V_w_day_trough_V_w_p_day(
            method = calculation_method, 
            building_type = building_type_B5, 
            building_area=building_area, 
            V_w_p_day = liters_per_person
            )

        V_W_nd_d = V_nd_d_ref*(teta_w_h_ref-teta_w_c_ref)/(teta_W_draw-teta_w_c_ref)
        
    elif building_type_B5 in {
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
        }: 
        
        if mode_calc == 'area':
            selection_B3 =  table_B_3.loc[table_B_3['type_of_usage']==building_type_B3, :]
            if selection_B3.empty:
                raise ValueError(f"Building usage '{building_type_B3}' not found in table_B_3.")
            if len(selection_B3) > 1:
                raise ValueError(f"Multiple entries found for usage '{building_type_B3}' in table_B_3.")
            # Area specific in kWh/m2d
            area_specific = float(selection_B3.iloc[0]['Area specific - Wh/m2d'])/1000
            # Q_W_calculation: daily energy need at reference condition
            Q_W_nd_d_ref = area_specific * building_area
            # Daily Volume need at  delivery temperature
            V_W_nd_d = Q_W_nd_d_ref/(WATER_DENSITY*WATER_SPECIFIC_HEAT_CAPACITY*(teta_W_draw-teta_w_c_ref))
        elif mode_calc == 'number_of_units':
            selection_B3 =  table_B_3.loc[table_B_3['type_of_usage']==building_type_B3, :]
            if selection_B3.empty:
                raise ValueError(f"Building usage '{building_type_B3}' not found in table_B_3.")
            if len(selection_B3) > 1:
                raise ValueError(f"Multiple entries found for usage '{building_type_B3}' in table_B_3.")
            energy_need = float(selection_B3.iloc[0]['Usage dependent'])
            Q_W_nd_d_ref = unit_count * energy_need 
            # Daily Volume need at  delivery temperature
            V_W_nd_d = Q_W_nd_d_ref/(WATER_DENSITY*WATER_SPECIFIC_HEAT_CAPACITY*(teta_W_draw-teta_w_c_ref))
        elif mode_calc == 'volume_type_bui':
            
            selection_B4 = table_B_4.loc[table_B_4['type_of_activity']==building_type_B5,:]
            if selection_B4.empty:
                raise ValueError(f"Activity '{building_type_B5}' not found in table_B_4.")
            if len(selection_B4) > 1:
                raise ValueError(f"Multiple entries found for activity '{building_type_B5}' in table_B_4.")
            # Daily volume'
            V_nd_d_ref = (unit_count*float(selection_B4.iloc[0]['V_W_f_day']))/1000
            # Daily volume need at delivery temperature
            V_W_nd_d = V_nd_d_ref*(teta_w_h_ref-teta_w_c_ref)/(teta_W_draw-teta_w_c_ref)
        else: 
            raise ValueError("select the right calculation mode from 'area', 'number_of_units', 'volume_type_bui'")
    else: 
        raise ValueError("select the building typology according to those defined in the table")

    # Daily energy need
    V_W_nd_d = float(V_W_nd_d)
    Q_W_nd_d = float(V_W_nd_d * WATER_DENSITY * WATER_SPECIFIC_HEAT_CAPACITY * (teta_W_draw-teta_W_cold))
    # Q_W = area_specific * building_area * n_day
    
    # Monthly Energy Need Calculation
    monthly_cons = [ days * Q_W_nd_d for days in get_days(year)]
    # Yearly Energy Need total of monthly energy needs
    yearly_cons = sum(monthly_cons)

    # Monthly Volume needs
    monthly_volume = [ days * V_W_nd_d for days in get_days(year)]
    # Yearly Volume needs
    yearly_volume = sum(monthly_volume)

    # Totally yearly fractions for workdays
    fractions_workday = n_workdays*sum_fractions.T['Workday'].values[0]
    # Totally yearly fractions for weekend
    fractions_weekend = n_weekends*sum_fractions.T['Weekend'].values[0]
    # Totally yearly fractions for holidays
    fractions_holiday = n_holidays*sum_fractions.T['Holiday'].values[0]
    # 
    tot_fractions = fractions_workday + fractions_weekend + fractions_holiday
    # AVerage correction fractor
    fx_avg = total_days / tot_fractions 
    # Hourly needs, corrected fractions of daily needs
    x_q_h_i_coor = hourly_fractions * fx_avg # dataframe with corrected fraction for workdays weekend and holidays
    
    # Hourly need as a volume at teta_draw
    V_W_nd_h_i = x_q_h_i_coor * V_W_nd_d  # dataframe with corrected fraction for workdays weekend and holidays
    # Hourly need as energy
    Q_W_nd_h_i = x_q_h_i_coor * Q_W_nd_d
    # Hourly result
    # check day category (workday, weekend and holidays): 
    daily_cons_volume = [] 
    daily_cons_energy = []
    for i, day_type in country_calendar.iterrows():
        if day_type['values'] == 'Working':
            col = 'Workday'
        elif day_type['values'] == 'Non-Working':
            col = 'Weekend'
        else:
            col = 'Holiday'
        for item_V in V_W_nd_h_i[col].values.tolist():
            daily_cons_volume.append(item_V)
        for item_Q in Q_W_nd_h_i[col].values.tolist():
            daily_cons_energy.append(item_Q)
    
    return yearly_cons, V_W_nd_d, monthly_volume, yearly_volume, Q_W_nd_d, V_W_nd_h_i, daily_cons_volume, daily_cons_energy