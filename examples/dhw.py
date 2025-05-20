import pandas as pd
from pybuildingenergy.source.DHW import generate_calendar, Volume_and_energy_DHW_calculation
import plotly.express as px
from workalendar.europe import Italy

# List of building Name according to the Table B3
bulding_name_B3 = [
    'Office_buildings', "Hospital ward or patient's room", 
    'School_without_showers', 'School_with_shower',
    'Retail shop/department store', 'Workshop, industrial facility (for washing snd showering)',
    'Modest hotel', 'Medium-class hotel',
    'Luxury-class hotel', 'Restaurant, inn/pub',
    'Home(for the aged, orphanage, etc.)', 'Barracks',
    'Sport faciltiy with showers', 'Commercial catering kitchen',
    'Bakery', 'Hairdresser/barber',
    'Butcher with production',
    'Laundry', 'Brewery', 'Dairy'
]
building_type = [
    'Single_family_house', 'Attached_house', 'Dwelling',
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
]

# If the selection is one of 'Single_family_house', 'Attached_house', 'Dwelling' in building_type, a residential type should be selected
residential_type = [
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
]


# ================================================================================================================
#                                   DHW NEEDS - INPUTS
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
units=10
building_type = 'Dwelling' 
residential_type = 'residential_building - simple housing - AVG'
# Use Profiles
hourly_fractions_examples = pd.DataFrame({
    "Workday" : [0,0,0,0,0,0,0,0,5,10,10,10,20,10,10,10,10,5,0,0,0,0,0,0],
    "Weekend" : [0,0,0,0,0,0,0,0,5,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0],
    "Holiday" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
)
sum_fractions = pd.DataFrame(hourly_fractions_examples.sum())
sum_fractions.columns= ["fractions"]
hourly_fractions = hourly_fractions_examples

# National calendar: import the calendar according to the nation 
Italy_calendar = generate_calendar(Italy, 2023)
n_workdays = sum(Italy_calendar['values'] == 'Working')
n_weekends = sum(Italy_calendar['values'] == 'Non-Working')
n_holidays = sum(Italy_calendar['values'] == 'Holiday')
Italy_calendar['values'].unique()
total_days = Italy_calendar.count().values[0]

# ================================================================================================================
#                                   DHW NEEDS - CALCULATION
# ================================================================================================================

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
    unit_count= units, 
    building_type_B5= building_type,
    residential_typology= residential_type, # table B_4
    calculation_method= 'table',
    year= 2015,
    country_calendar=Italy_calendar
    )

# ================================================================================================================
#                                   DHW NEEDS - SIMPLE PLOT
# ================================================================================================================


# Plot a specific day. Set t_end to 744 to visualize a week.
t_start=24
t_end = 48
df_DHW_calc = pd.DataFrame(dict(
    x = list(range(0,len(DHW_calc[6])))[t_start:t_end],
    y = DHW_calc[6][t_start:t_end],
))
fig = px.line(df_DHW_calc, x="x", y="y", title="Unsorted Input") 
fig.show()

df_DHW_calc['z'] = df_DHW_calc['y'].cumsum()
fig_1 = px.line(df_DHW_calc, x="x", y="z", title="Unsorted Input") 
fig_1.show()

