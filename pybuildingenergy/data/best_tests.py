import numpy as np

# ==========================================================================================
#                               INPUTS BEST_TEST600
# ==========================================================================================
bt_600 = {
    # BUILDING FEATURE
    'type': 'BestTest600', # building type
    'year': 2024, # year of construction 
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
    'side': 4.8,
    "heating_mode": True,
    "cooling_mode": True,
    'heating_setpoint': 20, # in °c
    'cooling_setpoint': 27, # in °c
    'heating_setback':20, # in °c
    'cooling_setback':27, # in °c
    'power_heating_max':1000000, # in W
    'power_cooling_max':-1000000, # in W
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
    'solar_area_elements': np.array([0.6,0.6,0.6,0.6,0.0,0.6,0.0], dtype=object),
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
    'occ_level_wd':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    'occ_level_we': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    'comf_level_wd': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    'comf_level_we': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "construction_class": "class_i",
    # WEATHER FILE
    "weather_source": 'epw',
    "tmy_filename": "tmy_39.783_-104.892_2005_2015.csv",
    "location": None,
    # OPTIMIZATION
    'baseline_hci': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50], dtype=object),
    'baseline_hce': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
}




