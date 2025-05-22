"""Main module."""
# Quick solution for debugging only
import sys
from pathlib import Path

# Add the project root to Python path (absolute path)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Prioritize this path

from src.pybuildingenergy.source.utils import ISO52016
# --- end of quick solution

from src.pybuildingenergy.source.utils import ISO52016
from src.pybuildingenergy.data.building_archetype import Selected_bui_archetype
from src.pybuildingenergy.source.graphs import Graphs_and_report
from src.pybuildingenergy.source.functions import is_float
from src.pybuildingenergy.global_inputs import bui_types, periods
import os
import argparse
import subprocess

main_directory_ = os.path.dirname(os.path.realpath(__file__))

def main(archetype=True, best_test=False, folder_dir="", name_file_="", archetype_db_path=""):
    
    if archetype:
        # select_archetyp = int(input("to select archetype type 1; for demo, type 2:"))
        select_archetype = 2
        if select_archetype == 1:

            building_archetype = input("select type of building archetype (e.g. 'single_family_house'):")
            period_archetype = input("Period of building construction(e.g. 'before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today'): ")
            latitude = input('latitude of the building location in decimal:')
            longitude = input('longitude fo the building location in decimal:')
        # check building archetype:
            if building_archetype in bui_types and period_archetype in periods:
                
                if is_float(latitude) and is_float(longitude):
                    BUI = Selected_bui_archetype(building_archetype,period_archetype,float(latitude), float(longitude)).get_archetype(archetype_db_path)
                    hourly_sim, annual_results_df = ISO52016().Temperature_and_Energy_needs_calculation(BUI, weather_source ='pvgis',  path_weather_file=main_directory_ + "/data/examples/weatherdata/2020_Athens.epw") 
                    Graphs_and_report(hourly_sim,'heating_cooling').bui_analysis_page(folder_directory=folder_dir, name_file=name_file_)
                
                else:
                    raise TypeError("Check if latitude and longitude are written in a proper way (as float)")
            
            else:
                raise TypeError (f"check if building archetype is in {bui_types} or periods in {periods}")
        
        elif select_archetype == 2:
            # DEMO FILE
            
            BUI = Selected_bui_archetype('single_family_house','before 1900',45.071321703968124, 7.642963669564985).get_archetype(archetype_db_path)
            hourly_sim, annual_results_df = ISO52016().Temperature_and_Energy_needs_calculation(BUI, weather_source ='pvgis', path_weather_file=main_directory_ + "/data/examples/weatherdata/2020_Athens.epw") 
            Graphs_and_report(hourly_sim,'heating_cooling').bui_analysis_page(folder_directory=folder_dir, name_file=name_file_)
        
        else:
            raise TypeError ("Select one archetype( type: 1) or a demo (type:2)")


    elif best_test:
        # RUN EXAMPLE BEST TEST 600
        try:
            # Execute the command using subprocess
            subprocess.run(['python3', '-m', 'besttest600'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
        print("results are available in examples/BESTEST600_iso_vs_energyplus.html")
    
    else:
        print("Running example..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--archetype', action='store_true', help='Run in test mode')
    parser.add_argument('--best_test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    folder_dir=main_directory_ + "/charts"
    name_file="new_building_from_archetype"
    archetypes_path = "{}/{}".format(main_directory_, "archetypes.pickle")
    main(archetype=True, best_test=False, folder_dir=folder_dir, name_file_=name_file, archetype_db_path=archetypes_path)
