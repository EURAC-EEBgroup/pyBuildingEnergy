'''
Simulate a building archetype by choosing from the available ones in the archetype.pickle. 
The archetypes are generated from the Tabula dataset. 
For now, only single_family_house are available in Italy for the following periods:
'before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today'
'''

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



def main(building_archetype:str, period_archetype: str, weather_type:str, path_weather_file_:str, latitude:float, longitude:float, 
         path_hourly_sim_result: str, path_annual_sim_result:str, 
         dir_chart_folder:str, name_report: str, 
         archetype_file_path:str=main_directory_ + "/archetypes.pickle"):
    '''
    Param
    ------
    building_archetype: type of building. Possible choice 'single_family_house'
    period_archetype: Period of building construction. Possible choised: 'before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today').
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
        weather_type = 'pvgis',
        path_weather_file_ = None,
        latitude = 44.78,
        longitude = 9.78,
        path_hourly_sim_result = file_dir + "/Result/hourly_sim__arch.csv",
        path_annual_sim_result = file_dir + "/Result/annual_sim__arch.csv",
        dir_chart_folder = file_dir+ "/Result",
        name_report = "main_report"
    )