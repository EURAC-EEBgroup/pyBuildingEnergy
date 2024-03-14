import numpy as np
from pybuildingenergy.source.utils import __ISO52010__, __ISO52016__
from pybuildingenergy.source.graphs import __Graphs__
from pybuildingenergy.data.building_archetype import Selected_bui_archetype
from pybuildingenergy.global_inputs import main_directory_

#1 GET DATA FROM ARCHETYPE
building_archetype = 'single_family_house'
period_archetype = 'before 1900'
latitude=  45.071321703968124 
longitude = 7.642963669564985
pickle_file_path = main_directory_ + "/archetypes.pickle"
BUI = Selected_bui_archetype(building_archetype,period_archetype,float(latitude), float(longitude)).get_archetype(pickle_file_path)

# Data to be changed by new building
bui_new = {
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
    'solar_area_elements': np.array([1.0,1.0,1.0,1.0,0.0,1.0,0.6,0.6,0.6,0.6], dtype=object),
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

# UPDATE BUILDING
BUI.update_values(bui_new)

# VALIDATE BUILDING 
# BUI.inputs_validation()

# Run Simulation and generate graphs 
hourly_sim, annual_results_df = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI, weather_source ='pvgis') 
__Graphs__(df = hourly_sim,season ='heating_cooling').bui_analysis_page(folder_directory="/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/MODERATE/pyBuildingEnergy/pybuildingenergy/pybuildingenergy/charts",
                                                                        name_file="new_building_from_archetype")
