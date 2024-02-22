'''
Description and properties of building elements
-----
The calculation of the heat capacity of each opaque element is based on the following formula:
density[kg/m3] * Specific heat capacity[J/kgK] * tichkness[m]
transmittance: reference Tabula/Episcope datasheet

ASSUMPTION:
- windowed surface 1/8 of the gross area
- windowed surface devided in 30% South  30% East 30% Weat 10% North 
- Rectangle building with:
    a side of 10 meters and oriented to the north.

UNIT:
thickness : centimeter
transmittance: W/m2K
thermal_resistance: m2K/W
heat capacity: J/kgK

STANDARD INPUTS
                                                    WALL       ROOF     WINDOW   FLOOR-Slab on ground
convective_heat_transfer_coefficient_internal:      2.50       5.00       2.50      0.70
convective_heat_transfer_coefficient_external:      5.13       5.13       5.13      5.13
radiative_heat_transfer_coefficient_internal:       20.0       5.13       20.0      20.0
radiative_heat_transfer_coefficient_external:       4.14       4.14       4.14      4.14
solar_factor*:                                      0.50       1.00       0.50      0.00
*Solar factor for vertical element is 0.5, 
 horizontal 1.0, 0.0 for floor slab on ground.

solar absorption of elements                        0.60       0.60       1.00      0.00

AREA ROOF
type of roof: gable roof with eaves overhang.
Area roof calcuated with slope of 28°C from the perimeter + 0.5m of eaves overhang

formula = ((length_small*cos(28)+0.5*cos(28))*length_roof)*2
cos(28°) = 0.88

INTERNAL GAINS
reference: 339487912_Optimization_of_passive_design_features_for_a_naturally_ventilated_residential_building_according_to_the_bioclimatic_architecture_concept_and_considering_the_northern_Morocco_climate

Zone                |       Equipment           |       Power       | Unit
Living                          TV                      120             W
                                Light                   5               W/m2
Bedroom                         Light                   5               W/m2
Bedroom2                        Light                   5               W/m2
Bedroom3                        Light                   5               W/m2
Kirtchen                        Refrigerator            100             W
                                Washing                 2000            W
                                Stove                   800             W
                                Light                   5               W/m2
People - 100/each               4                       100             W                                             

AIR CHANGE RATE
Residential                     0.5 m3/h 

HEATING SYSTEM
POWER:
    The power is calculated using the following formula with some approssimations
    For newly well-insulated buildings:
    - the standard energy requirement value can be 0.03 kW/m³, 
    for older buildings with high thermal losses:
    - the standard energy requirement value can be 0.12 kW/m³

COOLING SYSTEM
for the building archetype the cooling system is set off.
In this case the values of cooling setpoint and setback are the same



TABLE B.13 - DISTRIBUTION OF MASS OPAQUE AND GROUND FLOOR ELEMENTS
df_TB13 = pd.DataFrame({
    "Class":["Class I (mass concentrated at internal side)",
             "Class E (mass concentrated at external side)",
             "Class IE (mass divided over internal and external side)",
             "Class D (mass equally distributed)"
             ],
    "Specification_of_the_class": [
        "Construction with external thermal insulation (main mass component near inside surface) , or equivalent",
        "Construction with internal thermal insulation (main mass component near outside surface) , or equivalent",
        "Construction with thermal insulation in between two main mass components, or equivalent",
        "Uninsulated construction (e.g. solid or hollow bricks, heavy or lightweight concrete, or lightweight \
            construction with negligible mass (e.g. steel sandwich panel), or equivalent"]
    
})


TABLE B.14 - SPECIFIC HEAT CAPACITY OF OPAQUE AND GROUND FLOOR ELEMENTS
df_TB_14 = pd.DataFrame({
    "Class" : ["Very light","Light", "Medium", "Heavy", "Very heavy"],
    "kappa_m_op": [50000, 75000, 110000, 175000, 250000],
    "Specification of the class": [
        "Construction containing no mass components, other than e.g. plastic board and/or wood siding, or equivalent",
        "Construction containing no mass components other than 5 to 10 cm lightweight brick or concrete, or equivalent",
        "Construction containing no mass components other than 10 to 20 cm lightweight brick or concrete, or less than 7 cm solid brick or heavy weight concrete, or equivalent",
        "Construction containing 7 to 12 cm solid brick or heavy weight concrete, or equivalent",
        "Construction containing more than 12 cm solid brick or heavy weight concrete, or equivalent"
    ]
})



TABLE 25 - CONVENTIONAL HEAT TRANSFER COEFFICIENT
df_Tb_25 = pd.DataFrame({
    "Heat_transfer_coefficient":["convective coefficient; internal surface",
                                "convective coefficient; external surface",
                                "radiative coefficient, internal surface",
                                "radiative coefficient, external surface"],
    "Symbol":["hc_i", "hc_e", "hlr_i", "hlr_e"],
    "Direction_of_heat_flow_Upwards": [5,20,5.13,4.14],
    "Direction_of_heat_flow_Horizontal": [2.5,20,5.13,4.14],
    "Direction_of_heat_flow_Downwards": [0.7,20,5.13,4.14],
})


'''

import numpy as np
from data.profiles import profile_residential_1
from src.functions import Perimeter_from_area, Area_roof, Internal_gains, Power_heating_system
# def Perimeter_from_area(Area, side):
#     '''
#     Perimeter form area assuming 10m of side
#     '''
#     base = Area/side
#     perimeter = 2*(base + side)
#     return perimeter

# def Area_roof(leng_small, leng_roof):
#     '''
#     Area roof calculated according to: 
#     formula = ((length_small*cos(28)+0.5*cos(28))*(length_roof+(0.5*2)))*2
#     cos(28°) = 0.88
#     '''
#     Area = 2*((leng_small*0.88)+(0.5*0.88))*(leng_roof+(0.5*2))
#     return Area

# def Internal_gains(bui_type:str,area: float):
#     '''
#     Calcualtion of internal gains according to the building typology 
#     Param
#     --------
#     bui_type: type of building. Possible choice: residential, office, commercial
#     area: gross area of the building

#     Return 
#     --------
#     int_gains: value of internal gains in W/m2

#     Note
#     -----
#     Power value defined in the table on the top of the file 
#     Only for rsedintial the data is avaialble. 
#     '''
#     if bui_type ==  "residential":
#         # sum of power of equipments and people heat + lights
#         int_gains = (120+100+2000+800+5+4*100)/area + 5*5
        
#     else:
#         int_gains=5
#     return int_gains

# def Power_heating_system(bui_volume, bui_class):
#     '''
#     Approssimative calculation of generator power
#     p = Voume[m3] * energy needs[kW/m3]
#     Param
#     ------
#     bui_class: could be:
#         'old': No or very low insulated building
#         'new': very well insulated building
#         'average':medium insulated building 
    
#     Return 
#     ------
#     heat_power: power og the generator in Watt
#     '''
#     if bui_class == 'old':
#         p = bui_volume * 0.12
#     elif bui_class == 'gold':
#         p = bui_volume * 0.03
#     else:
#         p = bui_volume * 0.05

#     return p*1000

# latitude = 44.68045685667568
# longitude = 10.323822015417987
# profile_residential_1 = {
#     'code': 'profile01',
#     'type': 'residential',
#     'profile_workdays_internal_gains': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     dtype=object),
#     'profile_weekend_internal_gains': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     dtype=object),
#     'profile_workdays_ventilation': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     dtype=object),
#     'profile_weekend_ventilation': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     dtype=object)
# }

def main(latitude, longitude):

    # ========================================================================================================================
    # @Italy
    # WALL
    code_wall = ['wall01', 'wall02', 'wall03']
    description_wall = ['Masonry with lists of stones and bricks (40cm)', 'solid brick masonry', 'hollow brick masonry']
    thickness_wall = [0.40, 0.38, 0.40]
    heat_capacity_wall= [665658, 523248, 319500]
    U_wall= [ 1.61, 1.48, 1.26]
    R_wall= [1/value for value in U_wall]

    # ROOF 
    code_roof = ['roof01','roof02']
    description_roof = ['Pitched roof with wood structure and planking', 'Pitched roof with brick-concrete slab']
    thickness_roof = [0.34, 0.34]
    heat_capacity_roof= [278856, 390606]
    U_roof= [1.8, 2.2]
    R_roof= [1/value for value in U_roof]

    #FLOOR
    code_floor= ['floor01','floor02']
    description_floor = ['Concrete floor on soil', 'floor with reinforced brick-concreate slab, low insulation']
    thickness_floor = [0.27, 0.34]
    heat_capacity_floor= [463800, 448050]
    U_floor= [2.0, 0.98]
    R_floor= [1/value for value in U_floor]

    # WINDOW
    code_window = ['window01','window02']
    description_window = ['Single glass, methal frame without thermal break','single glasss wood frame']
    U_window = [5.7, 4.9]
    R_window = [1/value for value in U_window]
    g_window = [0.85, 0.85]

    # ========================================================================================================================
    #                                       SINGLE FAMILY HOUSE
    # ========================================================================================================================

    periods = ['before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today']
    type = ['single_fammily_house']*len(periods)
    area = [139, 115, 116, 162, 156, 199, 172, 174]
    window_area = [17.4, 14.4, 14.5, 20.3, 19.5, 24.9, 21.5, 21.8] # 1/8 della superificie
    volume = [533,448,455,583,679,725,605,607]
    S_V = [0.77, 0.82, 0.81, 0.75, 0.73, 0.72, 0.73, 0.72]
    S_envelope = [S*volume for S,volume in zip(S_V, volume)]
    number_of_floor = [2,2,2,2,2,2,2,2]
    bui_height=[x / (y/z) for x, y,z in zip(volume, area, number_of_floor)]
    base= [(value/number_of_floor)/10 for value,number_of_floor in zip(area,number_of_floor)]
    perimeter =[Perimeter_from_area(value, 10/2) for value in area]
    area_north = [round(10 * heights,2) for heights in bui_height]
    area_south = area_north
    area_west = [round(bases * heights,2) for bases,heights in zip(base,bui_height)]
    area_east = area_west
    area_roof = [round(Area_roof(10, leng_roof)/2,2) for leng_roof in base]
    w_code = ['wall01','wall01','wall02','wall02','wall03','wall01','wall01','wall01']
    r_code = ['roof01','roof01','roof01','roof02','roof02','roof01','roof01','roof01']
    win_code= ['window01','window01','window02','window02','window02','window02','window02','window02']
    f_code = ['floor01','floor01','floor01','floor01','floor01','floor01','floor01','floor01']
    building_category_const = ['old','old','old','old','old','old','medium','medium']
    air_change_rate_base_value = [0.11,0.14,0.14, 0.1, 0.1, 0.1, 0.1,0.1]
    bui_inputs_aryhetype = []
    for i,year in enumerate(periods):
        bui_ = {
            # BUILDING FEATURE
            'type': type[i], # building type
            'year': year, # year of construction 
            'latitude': latitude,
            'longitude': longitude, 
            'volume' : volume[i], # in m3
            'exposed_perimeter': perimeter[i], # perimeter in m
            'slab_on_ground_area': area[i]/number_of_floor[i], # Area slab on ground in m2
            'wall_thickness' :  thickness_wall[code_wall.index(w_code[i])], # in m
            'coldest_month': 1, 
            'a_use': area[i],
            'surface_envelope': area_north[i]+area_south[i]+area_east[i]+area_west[i]+area[i]/2+area_roof[i],
            'surface_envelope_model': S_envelope[i],
            # SYSTEMS 
            'base': base[i],
            "heating": True,
            "cooling": True,
            'H_setpoint': 20, # in °c
            'C_setpoint': 26, # in °c
            'H_setback':10, # in °c
            'C_setback':26, # in °c
            'Phi_H_nd_max':Power_heating_system(volume[i], building_category_const[i]), # in W
            # 'Phi_H_nd_max':30000, # in W
            'Phi_C_nd_max':-10000, # in W
            # INTERNAL GAINS and VENTILATION LOSSES
            'air_change_rate_base_value':air_change_rate_base_value[i]*area[i] , # in m3/h*m2
            # 'air_change_rate_base_value':1.1 , # in m3/h*m2
            'air_change_rate_extra':0.0, # in m3/h*m2
            'internal_gains_base_value':Internal_gains('residential', area[i]), # in W/m2
            # 'internal_gains_base_value':5, # in W/m2
            'internal_gains_extra':0.0, # in W/m2
            # THERMAL BRIDGES
            'H_tb' : 10.0, # in W/m
            # FEATURES OF FAACDE ELEMENTS:
            'R_floor_construction': R_floor[code_floor.index(f_code[i])], 
            # "Wall North", "Wall South", "Wall East", "Wall West", "Floor slab on gorund", "Roof", "Window North", "Window South", "Window East", "Window West"
            'TypeSub_eli': np.array(["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"],dtype=object), 
            'or_eli': np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV'],dtype=object),
            'a_sol_eli': np.array([1.0,1.0,1.0,1.0,0.0,1.0,0.6,0.6,0.6,0.6], dtype=object),
            'A_eli': [area_north[i], area_south[i], area_east[i],area_west[i],area[i]/2, area_roof[i], 
                    round(0.1*window_area[i],2),round(0.3*window_area[i],2),
                    round(0.3*window_area[i],2),round(0.3*window_area[i],2),
                    ],
            'U_eli' : [U_wall[code_wall.index(w_code[i])]]*4+ [U_floor[code_floor.index(f_code[i])]] + [U_roof[code_roof.index(r_code[i])]] +[U_window[code_window.index(win_code[i])]]*4,
            'R_eli' : [R_wall[code_wall.index(w_code[i])]]*4+ [R_floor[code_floor.index(f_code[i])]] + [R_roof[code_roof.index(r_code[i])]] +[R_window[code_window.index(win_code[i])]]*4,
            'kappa_m_eli' : [heat_capacity_wall[code_wall.index(w_code[i])]]*4+ [heat_capacity_floor[code_floor.index(f_code[i])]] + [heat_capacity_roof[code_roof.index(r_code[i])]] +[0]*4,
            'g_w_eli' : [0]*6 +[g_window[code_window.index(win_code[i])]]*4,
            'h_ci_eli': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
            'h_ri_eli': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
            'h_ce_eli': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
            'h_re_eli': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
            'F_sk_eli': np.array([0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object), 
            'occ_level_wd': profile_residential_1['profile_workdays_internal_gains'],
            'occ_level_we': profile_residential_1['profile_weekend_internal_gains'],
            'comf_level_wd': profile_residential_1['profile_workdays_ventilation'],
            'comf_level_we': profile_residential_1['profile_weekend_ventilation'],
            'baseline_hci': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
            'baseline_hce': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
            "construction_class": "class_i"
        }

        bui_inputs_aryhetype.append(bui_)

    return bui_inputs_aryhetype
    
    # Save the JSON data to a file
    # with open("building_archetype.json", "w") as file:
    #     file.write(json.dumps(bui_inputs_aryhetype))


if __name__ == "__main__":
    main()
    
# bii_1 = bui_inputs_aryhetype[0]


