"""Main module."""


from src.utils import __ISO52010__, __ISO52016__
from data.building_archetype import Selected_bui_archetype, Buildings_from_dictionary
from src.graphs import __Graphs__
from data.best_tests import bt_600
import numpy as np
from global_inputs import bui_types, periods
import argparse
import os
import pandas as pd

# # GET OBJECT ARCHETYPE
## inputs_user = {
#     'latitude':46.66345144066082,
#     'longitude':9.71636944229362,
#     'a_use': 100, 
#     "slab_on_ground":100,#
#     'H_setpoint':22,     
#     'C_setpoint':24,
#     'Phi_H_nd_max':40000,            
#     'Phi_C_nd_max':-10000,
#     'volume':400,
#     'A_eli': [0, 76.69, 53.3, 53.3, 69.5, 73.46, 1.74, 5.22]
# }
# BUI.update_values(inputs_user)
# BUI.inputs_validation()
# BUI.A_eli


# bui_ = {
#     # BUILDING FEATURE
#     'type': 'BestTest600', # building type
#     'year': 2024, # year of construction 
#     'latitude': 39.76,
#     'longitude': -104.86, 
#     'volume' : 129.6, # in m3
#     'exposed_perimeter': 28, # perimeter in m
#     'slab_on_ground': 48, # Area slab on ground in m2
#     'wall_thickness' :  0.087, # in m
#     'coldest_month': 1, 
#     'a_use': 48,
#     'surface_envelope': 48+48+21.6+9.6+12+16.2+16.2,
#     'surface_envelope_model': 48+48+21.6+9.6+12+16.2+16.2,
#     'annual_mean_internal_temperature': 39.76,
#     'annual_mean_external_temperature': 7,
#     # SYSTEMS 
#     'side': 4.8,
#     "heating_mode": True,
#     "cooling_mode": True,
#     'heating_setpoint': 20, # in °c
#     'cooling_setpoint': 27, # in °c
#     'heating_setback':20, # in °c
#     'cooling_setback':27, # in °c
#     'power_heating_max':1000000, # in W
#     # 'power_heating_max':30000, # in W
#     'power_cooling_max':-1000000, # in W
#     # INTERNAL GAINS and VENTILATION LOSSES
#     'air_change_rate_base_value':1.35 , # in m3/h*m2
#     # 'air_change_rate_base_value':1.1 , # in m3/h*m2
#     'air_change_rate_extra':0.0, # in m3/h*m2
#     'internal_gains_base_value':4.1667, # in W/m2
#     # 'internal_gains_base_value':5, # in W/m2
#     'internal_gains_extra':0, # in W/m2
#     # THERMAL BRIDGES
#     'thermal_bridge_heat' : 0.0, # in W/m
#     # FEATURES OF FAACDE ELEMENTS:
#     'thermal_resistance_floor': 0.039, 
#     # "Wall North", "Wall South", "Wall East", "Wall West", "Floor slab on gorund", "Roof", "Window North", "Window South", "Window East", "Window West"
#     'typology_elements': np.array(["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"],dtype=object), 
#     'orientation_elements': np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV'],dtype=object),
#     'solar_area_elements': np.array([1.0,1.0,1.0,1.0,0.0,1.0,0.6,0.6,0.6,0.6], dtype=object),
#     'area_elements': [21.6, 9.6, 16.2,16.2, 48, 48,0.009999, 12,0.009999, 0.009999 ],
#     'transmittance_U_elments' : [0.514, 0.514, 0.514, 0.514, 0.04, 0.318, 10, 3, 10, 10],
#     'thermal_resistance_R_elements' : [1.945, 1.945, 1.945, 1.945, 25.374, 3.1466, 10, 0.33333, 10, 10],
#     'thermal_capacity_elements' : [175000, 175000,175000,175000,110000, 50000, 0, 0, 0, 0],
#     'g_factor_windows' : [0]*6 +[0,0.71,0,0],
#     'heat_convective_elements_internal': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
#     'heat_radiative_elements_internal': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
#     'heat_convective_elements_external': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
#     'heat_radiative_elements_external': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
#     'sky_factor_elements': np.array([0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object), 
#     'occ_level_wd':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#     'occ_level_we': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#     'comf_level_wd': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#     'comf_level_we': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#     'baseline_hci': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
#     'baseline_hce': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
#     "construction_class": "class_i",
#     "weather_source": 'epw',
#     "tmy_filename": "tmy_39.783_-104.892_2005_2015.csv",
#     "location": None
# }
# bui = Buildings_from_dictionary(bui_)




def is_float(value):
    try:
        float_value = float(value)
        return isinstance(float_value, float)
    except ValueError:
        return False
    


def main(archetype=False, best_test=False):
    print(archetype, best_test)
    if archetype:
        building_archetype = input("select type of building archetype (e.g. 'single_family_house'):")
        period_archetype = input("Year of building construction(e.g.'before 1900'): ")
        latitude = input('latitude of the building location in decimal:')
        longitude = input('longitude fo the building location in decimal:')
        # check building archetype:
        if building_archetype in bui_types and period_archetype in periods:
            
            if is_float(latitude) and is_float(longitude):
                inizialize_building = Selected_bui_archetype(building_archetype,period_archetype,float(latitude), float(longitude))
                BUI = inizialize_building.Get_bui_archetype()
                # print(BUI.__getattribute__('occ_level_wd'), bui.__getattribute__('a_use'))
                hourly_sim = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI, weather_source ='pvgis') 
                # hourly_sim = __ISO52016__().Temperature_and_Energy_needs_calculation(bui) 
                print(hourly_sim['Q_H'].sum())
                # __Graphs__(hourly_sim,'heating_cooling').bui_analysis_page()
                print("Report created! check in the charts folder")
            else:
                raise TypeError("Check if latitude and longitude are written in a proper way (as float)")
        
        else:
            raise TypeError (f"check if building archetype is in {bui_types} or periods in {periods}")
    
    elif best_test:

        # besttest_model = input("select best-test model (actually available 'model_940'):")
        # Weather_data = __ISO52010__.get_tmy(BUI=bui, tmy_filename="tmy_39.783_-104.892_2005_2015.csv")
        # Weather_data_pvgis = __ISO52010__.get_tmy_data(BUI=bui)
        
        # # print(Weather_data)
        # # Weather_data['Weather data'].to_csv("weather_epw.csv")
        # Weather_data_pvgis.weather_data.to_csv("weather_epw_pvgis.csv")
        # print(Weather_data_pvgis.utc_offset)
        hourly_sim = __ISO52016__().Temperature_and_Energy_needs_calculation(Buildings_from_dictionary(bt_600)) 
        hourly_sim.to_csv('test_with_epw.csv')
        print(hourly_sim.loc[:,'Q_H'].sum())
        # __Graphs__(hourly_sim,'heating_cooling').bui_analysis_page()
        # print(hourly_sim['Q_H'].sum())
    
    else:
        print("Running example..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--archetype', action='store_true', help='Run in test mode')
    parser.add_argument('--best_test', action='store_true', help='Run in test mode')
    # parser.add_argument('--from_archetype_to_my_building', action='store_true', help='Run in test mode')
    # parser.add_argument('--my_building', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    main(archetype=args.archetype, best_test=args.best_test)

# main(best_test='--best_test')


#%%
# import pandas as pd
# data = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/pybuildingenergy/pybuildingenergy/test_with_epw.csv", index_col=0)
# data.loc[:,'Q_H'].sum()

# pvgis_=pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/pybuildingenergy/pybuildingenergy/weather_epw_pvgis.csv", index_col=0)
# pvgis_.index = pd.DatetimeIndex(pvgis_.index)

# sim_df = pvgis_
# sim_df.index = pd.DatetimeIndex(sim_df.index)
# # timezoneW = weatherData['UTC offset']
# timezoneW = 1
# # Change time index
# sim_df.index.year.unique().values
# # sim_df.loc[sim_df.index.year == 2015,:]
# sim_df.index = pd.to_datetime({'year': 2009, 'month': sim_df.index.month, 'day': sim_df.index.day,
#                                 'hour': sim_df.index.hour})
# sim_df_sorted = sim_df.sort_index(ascending=True)
# sim_df_sorted.loc['2009-01-01':'2009-01-31']

# #%%%%
# epw = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/pybuildingenergy/pybuildingenergy/weather_epw.csv", index_col=0)
# epw.index = pd.DatetimeIndex(epw.index)
# timezoneW = 1
# # Change time index
# epw.index.year.unique().values
# # sim_df.loc[sim_df.index.year == 2015,:]
# epw.index = pd.to_datetime({'year': 2009, 'month': epw.index.month, 'day': epw.index.day,
#                                 'hour': epw.index.hour})
# epw_sorted = epw.sort_index(ascending=True)
# epw_sorted.loc['2009-01-01':'2009-01-31']

