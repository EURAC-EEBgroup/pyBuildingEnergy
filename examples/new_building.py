import numpy as np
from pybuildingenergy.source.utils import ISO52016
from pybuildingenergy.source.graphs import Graphs_and_report
from pybuildingenergy.data.building_archetype import Buildings_from_dictionary
import os

# Inputs
file_dir = os.path.dirname(os.path.realpath(__file__))
'''
Provide the directory data where to save the results and charts; if a new one is not provided, a directory named 'result' is created
'''

file_dir = os.path.dirname(os.path.realpath(__file__))
# Check directory if it is not available create it
def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists.
    If it doesn't exist, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

ensure_directory_exists(file_dir+"/Result")


# Building
user_bui = {
    # BUILDING FEATURE
    'building_type': 'myBui', # building type
    'periods': 2024, # year of construction 
    'latitude': 39.76,
    'longitude': -104.86, 
    'volume' : 129.6, # in m3
    'exposed_perimeter': 28, # perimeter in m
    'slab_on_ground': 48, # Area slab on ground in m2
    'wall_thickness' :  0.087, # in m
    'coldest_month': 1, 
    'a_use': 48,
    'surface_envelope': 48+48+21.6+9.6+12+16.2+16.2,
    'surface_envelope_model': 48+48+21.6+9.6+12+16.2+16.2,
    'annual_mean_internal_temperature': 39.76,
    'annual_mean_external_temperature': 7,
    # SYSTEMS 
    "heating_mode": True,
    "cooling_mode": True,
    'heating_setpoint': 20, # in °c
    'cooling_setpoint': 27, # in °c
    'heating_setback':10, # in °c
    'cooling_setback':27, # in °c
    'power_heating_max':10000, # in W
    'power_cooling_max':-10000, # in W
    # INTERNAL GAINS and VENTILATION LOSSES
    'air_change_rate_base_value':1.35 , # in m3/h*m2
    'air_change_rate_extra':0.0, # in m3/h*m2
    'internal_gains_base_value':4.1667, # in W/m2
    'internal_gains_extra':0, # in W/m2
    # THERMAL BRIDGES
    'thermal_bridge_heat' : 0.0, # in W/m
    # FEATURES OF FAACDE ELEMENTS:
    'thermal_resistance_floor': 0.039, 
    'typology_elements': np.array(["OP", "OP", "OP", "OP", "GR", "OP", "W"],dtype=object), 
    'orientation_elements': np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'SV'],dtype=object),
    'solar_abs_elements': np.array([0.6,0.6,0.6,0.6,0.0,0.6,0.0], dtype=object),
    'area_elements': [21.6, 9.6, 16.2,16.2, 48, 48, 12 ],
    'transmittance_U_elements' : [0.514, 0.514, 0.514, 0.514, 0.04, 0.318, 3],
    'thermal_resistance_R_elements' : [1.77303867, 1.77303867, 1.77303867, 1.77303867, 25.374, 3.00451238,0.16084671],
    'thermal_capacity_elements' : [14534.28, 14534.28,14534.28,14534.28,19500, 18169.944, 0],
    'g_factor_windows' : [0]*6 +[0.71],
    'heat_convective_elements_internal': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50], dtype=object),
    'heat_radiative_elements_internal': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
    'heat_convective_elements_external': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
    'heat_radiative_elements_external': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
    'sky_factor_elements': np.array([0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50], dtype=object), 
    'occ_level_wd': np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=object),
    'occ_level_we': np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=object),
    'comf_level_wd': np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=object),
    'comf_level_we': np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=object),
    "construction_class": "class_i",
    # WEATHER FILE
    "weather_source": 'pvgis'
}




def main(bui_new:dict, weather_type:str, path_weather_file_:str, 
         path_hourly_sim_result: str, path_annual_sim_result:str, 
         dir_chart_folder:str, name_report: str):
    '''
    Param
    ------
    building_archetype: type of building. Possible choice 'single_family_house'
    period_archetype: Period of building construction. Possible choised: 'before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today').
    bui_new: dictionary with inputs of own building to be changed in the archetype building inputs 
    weather_type: Specify the data source for weather data. If using the PVGIS website, indicate 'pvgis'; if loading an EPW file from the path_weather_file_, indicate 'epw'.
    latitude: mandatory if weather_type is 'pvgis'
    longitude:  mandatory if weather_type is 'epw'
    path_weather_file_: if weather_type ='epw', specify the folder where the epw file is uploaded.
    archetype_file_path: pickel file in whcihe tere are all available building archetypes
    dir_chart_folder: directory where charts files are created. Some pre-set charts are saved within the folder.
    name_report: name of the main report to be saved in the dir_chart_folder 
    '''

    # Create Building object
    BUI = Buildings_from_dictionary(bui_new)

    # Run Simulation 
    hourly_sim, annual_results_df = ISO52016().Temperature_and_Energy_needs_calculation(BUI, weather_source=weather_type, path_weather_file=path_weather_file_) 
    hourly_sim.to_csv(path_hourly_sim_result)
    annual_results_df.to_csv(path_annual_sim_result)
    
    # Generate Graphs
    Graphs_and_report(df = hourly_sim,season ='heating_cooling').bui_analysis_page(
        folder_directory=dir_chart_folder,
        name_file=name_report)
    
    return print(f"Simulation eneded!check results in {path_hourly_sim_result} and {path_annual_sim_result}")


if __name__ == "__main__":
    main(
        bui_new = user_bui,
        weather_type = 'pvgis',
        path_weather_file_ = None,   
        path_hourly_sim_result = file_dir + "/Result/hourly_sim__arch.csv",
        path_annual_sim_result = file_dir + "/Result/annual_sim__arch.csv",
        dir_chart_folder = file_dir+ "/Result",
        name_report = "main_report_2"
    )