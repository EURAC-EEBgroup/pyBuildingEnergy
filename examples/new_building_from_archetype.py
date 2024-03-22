import numpy as np
from pybuildingenergy.source.utils import ISO52016
from pybuildingenergy.source.graphs import Graphs_and_report
from pybuildingenergy.data.building_archetype import Selected_bui_archetype
from pybuildingenergy.global_inputs import main_directory_
import os

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


# Building inputs to be uses in the archetype
user_bui = {
    # BUILDING FEATURE
    'volume' : 129.6, # in m3
    'exposed_perimeter': 28, # perimeter in m
    'slab_on_ground': 48, # Area slab on ground in m2
    'wall_thickness' :  0.087, # in m
    'a_use': 48,
    'surface_envelope': 171.6,
    # SYSTEMS 
    "heating_mode": True,
    "cooling_mode": True,
    'heating_setpoint': 20, # in °c
    'heating_setback':15, # in °c
    'power_heating_max':27000, # in W
    # 'power_heating_max':30000, # in W
    # INTERNAL GAINS and VENTILATION LOSSES
    # "Wall North", "Wall South", "Wall East", "Wall West", "Floor slab on gorund", "Roof", "Window North", "Window South", "Window East", "Window West"
    'typology_elements': np.array(["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"],dtype=object), 
    'orientation_elements': np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV'],dtype=object),
    'solar_abs_elements': np.array([1.0,1.0,1.0,1.0,0.0,1.0,0.6,0.6,0.6,0.6], dtype=object),
    'area_elements': [21.6, 9.6, 16.2,16.2, 48, 48,0, 12,0, 0],
    'transmittance_U_elements' : [0.514, 0.514, 0.514, 0.514, 0.04, 0.318, 1, 1, 1, 1],    
    'thermal_capacity_elements' : [175000, 175000,175000,175000,110000, 50000, 0, 0, 0, 0],
    'g_factor_windows' : [0]*6 +[0,0.71,0,0],
    'heat_convective_elements_internal': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
    'heat_radiative_elements_internal': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
    'heat_convective_elements_external': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
    'heat_radiative_elements_external': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
    'sky_factor_elements': np.array([0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object), 
    'occ_level_wd':np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], dtype=object),
    'occ_level_we': np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], dtype=object),
    'comf_level_wd': np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], dtype=object),
    'comf_level_we': np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], dtype=object),
    "construction_class": "class_i",
    "weather_source": 'pvgis',
    "tmy_filename": None,
    "location": None
}



def main(building_archetype:str, period_archetype: str, bui_new:dict, weather_type:str, path_weather_file_:str, latitude:float, longitude:float, 
         path_hourly_sim_result: str, path_annual_sim_result:str, 
         dir_chart_folder:str, name_report: str, 
         archetype_file_path:str=main_directory_ + "/archetypes.pickle"):
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

    # Get Archetype
    BUI = Selected_bui_archetype(building_archetype,period_archetype,float(latitude), float(longitude)).get_archetype(archetype_file_path)
    # Update values of won building
    BUI.update_values(bui_new)

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
        building_archetype = 'single_family_house',
        period_archetype = 'before 1900',
        bui_new = user_bui,
        weather_type = 'pvgis',
        path_weather_file_ = None,
        latitude = 44.78,
        longitude = 9.78,
        path_hourly_sim_result = file_dir + "/Result/hourly_sim__arch.csv",
        path_annual_sim_result = file_dir + "/Result/annual_sim__arch.csv",
        dir_chart_folder = file_dir+ "/Result",
        name_report = "main_report"
    )