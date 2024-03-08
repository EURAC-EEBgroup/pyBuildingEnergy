"""Main module."""


from src.utils import __ISO52010__, __ISO52016__
from data.building_archetype import Selected_bui_archetype
from src.graphs import __Graphs__
from src.functions import is_float
from global_inputs import bui_types, periods
import argparse
import subprocess
from global_inputs import main_directory_



def main(archetype=False, best_test=False, folder_dir="", name_file_="", archetype_db_path=""):
    
    if archetype:
        select_archetype = int(input("to select archetype type 1; for demo, type 2:"))
        if select_archetype == 1:

            building_archetype = input("select type of building archetype (e.g. 'single_family_house'):")
            period_archetype = input("Year of building construction(e.g.'before 1900'): ")
            latitude = input('latitude of the building location in decimal:')
            longitude = input('longitude fo the building location in decimal:')
        # check building archetype:
            if building_archetype in bui_types and period_archetype in periods:
                
                if is_float(latitude) and is_float(longitude):
                    BUI = Selected_bui_archetype(building_archetype,period_archetype,float(latitude), float(longitude)).get_archetype(archetype_db_path)
                    hourly_sim, annual_results_df = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI, weather_source ='pvgis') 
                    __Graphs__(hourly_sim,'heating_cooling').bui_analysis_page(folder_directory=folder_dir, name_file=name_file_)
                    print("Report created! check in the charts folder")
                
                else:
                    raise TypeError("Check if latitude and longitude are written in a proper way (as float)")
            
            else:
                raise TypeError (f"check if building archetype is in {bui_types} or periods in {periods}")
        
        elif select_archetype == 2:
            # DEMO FILE
            
            BUI = Selected_bui_archetype('single_family_house','before 1900',45.071321703968124, 7.642963669564985).get_archetype(archetype_db_path)
            hourly_sim, annual_results_df = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI, weather_source ='pvgis') 
            __Graphs__(hourly_sim,'heating_cooling').bui_analysis_page(folder_directory=folder_dir, name_file=name_file_)
        
        else:
            raise TypeError ("Select one archetype( type: 1) or a demo (type:2)")


    elif best_test:
        # RUN EXAMPLE BEST TEST 600
        try:
            # Execute the command using subprocess
            subprocess.run(['python3', '-m', 'examples.besttest600'], check=True)
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
    folder_dir="/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/pybuildingenergy/pybuildingenergy/charts"
    name_file="new_building_from_archetype"
    archetypes_path = file_path = "{}/{}".format(main_directory_, "archetypes.pickle")
    main(archetype=args.archetype, best_test=args.best_test, folder_dir=folder_dir, name_file_=name_file, archetype_db_path=archetypes_path)

