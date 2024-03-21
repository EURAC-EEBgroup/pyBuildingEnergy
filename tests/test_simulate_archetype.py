import numpy as np
from src.pybuildingenergy.source.utils import ISO52016
from src.pybuildingenergy.source.graphs import Graphs_and_report
from src.pybuildingenergy.data.building_archetype import Selected_bui_archetype
from src.pybuildingenergy.global_inputs import main_directory_
import os

# Inputs
file_dir = os.path.dirname(os.path.realpath(__file__))

building_archetype = 'single_family_house'
period_archetype = 'before 1900'
weather_type = 'pvgis'
path_weather_file_ = None
latitude = 44.78
longitude = 9.78
path_hourly_sim_result = "/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/Result_test" + "/hourly_sim__arch.csv"
path_annual_sim_result = "/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/Result_test" + "/annual_sim__arch.csv"
dir_chart_folder = "/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/Result_test"
name_report = "main_report"
archetype_file_path =main_directory_ + "/archetypes.pickle"

def test_simulate_archetype(snapshot):
    '''
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
    # Simulate building
    hourly_sim, annual_results_df = ISO52016().Temperature_and_Energy_needs_calculation(BUI, weather_source=weather_type, path_weather_file=path_weather_file_) 
    
    # Generate Graphs
    report = Graphs_and_report(df = hourly_sim,season ='heating_cooling').bui_analysis_page(
        folder_directory=dir_chart_folder,
        name_file=name_report)
    
    print(f"Simulation eneded!check results in {path_hourly_sim_result} and {path_annual_sim_result}")
    return snapshot.assert_match(report, "report_from_archetype_generated.yml")