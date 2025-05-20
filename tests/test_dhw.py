import pandas as pd
from src.pybuildingenergy.source.DHW import generate_calendar, Volume_and_energy_DHW_calculation
import plotly.express as px

from workalendar.europe import Italy
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
building_type = 'Dwelling'
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
# calendar_nation= Italy()
Italy_calendar = generate_calendar(Italy, 2023)
n_workdays = sum(Italy_calendar['values'] == 'Working')
n_weekends = sum(Italy_calendar['values'] == 'Non-Working')
n_holidays = sum(Italy_calendar['values'] == 'Holiday')
Italy_calendar['values'].unique()
total_days = Italy_calendar.count().values[0]

def test_energy_needs(snapshot):

    # DHW needs
    DHW_calc = Volume_and_energy_DHW_calculation(
        n_workdays, n_weekends, n_holidays,sum_fractions, total_days, hourly_fractions,
        teta_W_draw, 
        teta_w_c_ref,
        teta_w_h_ref,
        teta_W_cold,
        mode_calc= 'number_of_units', 
        building_type_B3= 'Residential', 
        building_area= building_area, 
        unit_count= 10, 
        building_type_B5= building_type,
        residential_typology= 'residential_building - simple housing - AVG', # table B_4
        calculation_method= 'table',
        year= 2015,
        country_calendar=Italy_calendar
        )
    
    t_start=24
    t_end = 48
    df_DHW_calc = pd.DataFrame(dict(
        x = list(range(0,len(DHW_calc[6])))[t_start:t_end],
        y = DHW_calc[6][t_start:t_end],
    ))
    print(df_DHW_calc)
        # Plot data
    
    df = pd.DataFrame(dict(
        x = list(range(0,len(DHW_calc[6])))[t_start:t_end],
        y = DHW_calc[6][t_start:t_end],
    ))
    fig = px.line(df, x="x", y="y", title="Unsorted Input") 
    fig.show()

    df['z'] = df['y'].cumsum()
    fig_1 = px.line(df, x="x", y="z", title="Unsorted Input") 
    fig_1.show()
    return snapshot.assert_match(str(df_DHW_calc.__dict__), "dhw_test.yml")

