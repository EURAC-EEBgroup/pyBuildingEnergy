__author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"

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

Definition of reference tecnology and building archetypo for each nation.
Reference: 
- EIPISCOPE/TABULA
- Building stock Observatory 

# Acutal limitation
Vedere test area edificio e area del solaio controterra differenti 
# Italy:
- tasso di ventilazione fissato a 0.3  h-1
- considerato solo edifici tipo zona climatica media (E)

'''

import numpy as np
# from pybuildingenergy.data.profiles import profile_residential_1
from data.profiles import profile_residential_1
# from pybuildingenergy.src.functions import Perimeter_from_area, Area_roof, Internal_gains, Power_heating_system
from src.functions import Perimeter_from_area, Area_roof, Internal_gains, Power_heating_system, filter_list_by_index, Check_area

# ================================================================================================
#                           COMPONENTS ARCHETYPE            
# ================================================================================================

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
#                                       INPUTS: SINGLE FAMILY HOUSE
# ========================================================================================================================

periods = ['before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today']
bui_types = ['single_fammily_house']*len(periods)
area = [139, 115, 116, 162, 156, 199, 172, 174]
window_area = [17.4, 14.4, 14.5, 20.3, 19.5, 24.9, 21.5, 21.8] # 1/8 della superificie
volume = [533,448,455,583,679,725,605,607]
coldest_month = [1]*len(periods)
S_V = [0.77, 0.82, 0.81, 0.75, 0.73, 0.72, 0.73, 0.72]
S_envelope = [S*volume for S,volume in zip(S_V, volume)]
number_of_floor = [2,2,2,2,2,2,2,2]
height = [round(volume_i/(area_i/number_of_floor_i),2) for volume_i, area_i, number_of_floor_i in zip(volume, area, number_of_floor)]
bui_height=[x / (y/z) for x, y,z in zip(volume, area, number_of_floor)]
base= [(value/number_of_floor)/10 for value,number_of_floor in zip(area,number_of_floor)]
perimeter =[Perimeter_from_area(value, 10/2) for value in area]
area_north = [round(10 * heights,2) for heights in bui_height]
area_south = area_north
area_west = [round(bases * heights,2) for bases,heights in zip(base,bui_height)]
area_east = area_west
area_roof = [round(Area_roof(10, leng_roof)/2,2) for leng_roof in base]
thermal_bridge_heat  =[10]*len(periods)
w_code = ['wall01','wall01','wall02','wall02','wall03','wall01','wall01','wall01']
r_code = ['roof01','roof01','roof01','roof02','roof02','roof01','roof01','roof01']
win_code= ['window01','window01','window02','window02','window02','window02','window02','window02']
f_code = ['floor01','floor01','floor01','floor01','floor01','floor01','floor01','floor01']
building_category_const = ['old','old','old','old','old','old','medium','medium']
air_change_rate_base_value = [0.08,0.14,0.14, 0.1, 0.1, 0.1, 0.1,0.1]
# GLOBAL INPUTS
typology_elements = np.array(["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"],dtype=object)
orientation_elements =  np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV'],dtype=object)
solar_area_elements =  np.array([1.0,1.0,1.0,1.0,0.0,1.0,0.6,0.6,0.6,0.6], dtype=object)
heat_convective_elements_internal= np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object)
heat_radiative_elements_internal= np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object)
heat_convective_elements_external= np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object)
heat_radiative_elements_external= np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object)
sky_factor_elements= np.array([0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object)
baseline_hci= np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object)
baseline_hce= np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object)

profile_residential_1 = {
    'code': 'profile01',
    'type': 'residential',
    'profile_workdays_internal_gains': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'profile_weekend_internal_gains': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'profile_workdays_ventilation': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    'profile_weekend_ventilation': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
}

heating_installed = [True]*len(periods)
cooling_installed = [False]*len(periods)

# ========================================================================================================================
#                                       GENERATE BUILDING ARCHETYPE
# ========================================================================================================================

class BuildingArchetype(object):

    def __init__(self, 
        building_type: str, 
        periods: str,
        latitude: float,
        longitude: float, 
        exposed_perimeter: float,
        area: float, 
        number_of_floor: int,
        height: float,
        volume:float,
        slab_on_ground:float=None,
        wall_thickness= float,
        coldest_month = int,
        surface_envelope = float,
        surface_envelope_model = float,
        side = float, 
        heating_mode = bool, # True or False if heating syste is 
        cooling_mode =  bool, 
        heating_setpoint = float, # Heating set-point in °C
        cooling_setpoint = float, # Cooling set-point in °
        heating_setback = float, # Heating set-back in °C
        cooling_setback = float, # Cooling set-back in °C
        power_heating_max = float, # Max Power of the heating system
        power_cooling_max = float, # Max Power of the cooling system
        air_change_rate_base_value = float, # air change rate
        air_change_rate_extra = float,
        internal_gains_base_value = float,
        internal_gains_extra = float,
        thermal_bridge_heat = float, # value of thermal bridges
        thermal_resistance_floor = float,
        area_elements = list, # Area of each facade elements 
        transmittance_U_elments = list,
        thermal_resistance_R_elements = list,
        thermal_capacity_elements = list,
        g_factor_windows = list,        
        occ_level_wd= np.array,
        occ_level_we=np.array,
        comf_level_wd= np.array,
        construction_class= "class_i"
        ):

        self.building_type = building_type
        self.periods = periods
        self.latitude = latitude
        self.longitude = longitude
        self.a_use = area
        self.exposed_perimeter = exposed_perimeter
        self.height=  height
        self.number_of_floor = number_of_floor
        self.volume = volume  # Initialize attribute to None
        self.slab_on_ground = slab_on_ground
        self.wall_thickness = wall_thickness
        self.coldest_month = coldest_month
        self.surface_envelope = surface_envelope # calculated from sum of surfaces + floor  + roof
        self.surface_envelope_model = surface_envelope_model # calculated from S/V coefficient
        self.base = side # side of rectangular shape of building, a side of 10 meters is hypothesized
        self.heating_mode = heating_mode
        self.cooling_mode = cooling_mode
        self.heating_setpoint = heating_setpoint
        self.heating_setback = heating_setback
        self.cooling_setpoint = cooling_setpoint
        self.cooling_setback = cooling_setback
        self.power_cooling_max = -power_cooling_max
        self.power_heating_max = power_heating_max
        self.air_change_rate_base_value = air_change_rate_base_value
        self.air_change_rate_extra = air_change_rate_extra
        self.internal_gains_base_value = internal_gains_base_value
        self.internal_gains_extra = internal_gains_extra
        self.thermal_bridge_heat = thermal_bridge_heat
        self.thermal_resistance_floor = thermal_resistance_floor
        self.typology_elements = typology_elements
        self.orientation_elements = orientation_elements
        self.solar_area_elements = solar_area_elements
        self.area_elements = area_elements
        self.transmittance_U_elments =transmittance_U_elments 
        self.thermal_resistance_R_elements = thermal_resistance_R_elements
        self.thermal_capacity_elements = thermal_capacity_elements
        self.g_factor_windows = g_factor_windows
        self.heat_convective_elements_internal = heat_convective_elements_internal
        self.heat_radiative_elements_internal = heat_radiative_elements_internal
        self.heat_convective_elements_external = heat_convective_elements_external
        self.heat_radiative_elements_external = heat_radiative_elements_external
        self.sky_factor_elements = sky_factor_elements
        self.occ_level_wd = occ_level_wd
        self.occ_level_we = occ_level_we
        self.comf_level_wd = comf_level_wd
        self.baseline_hci = baseline_hci
        self.baseline_hce = baseline_hce
        self.construction_class = construction_class

    @property
    def slab_on_ground(self):
        return self._slab_on_ground
    
    @slab_on_ground.setter
    def slab_on_ground(self, value):
        '''
        Calculation of the slab on the ground area
        Aslab = Area/number_of_floor [m2]
        '''
        if value is None:
            self._slab_on_ground= self.a_use / self.number_of_floor
        else:
            self._slab_on_ground = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        if value is None:
            self._volume = (self.a_use / self.number_of_floor) * self.height
        else:
            self._volume = value

    # Area of building elements can not be 0. Se
    @property
    def area_elements(self):
        return self._area_elements
    
    @area_elements.setter
    def area_elements(self, values):
        self._area_elements = Check_area(values)



    def update_values(self, new_values:dict) -> object:
        '''
        Update characteristic of a building providing a dictionary with new values
        Param
        -------
        new_values: dictioanry with new values of a building feature

        Return 
        -------
        building object updated

        Example
        --------
        new_inputs = {
            'latitude':46.66345144066082,
            'longitude':9.71636944229362,
            'Eln':10, #
            'a_use': 100, 
            "slab_on_ground":100,#
            'heating_setpoint':22,     
            'cooling_setpoint':24,
            'power_heating_max':40000,            
            'power_cooling_max':-10000
        }
        '''
        for key, value in new_values.items():
            if not hasattr(self, key):
                pass
            if isinstance(value, list):
                if isinstance(getattr(self, key), list):
                    if len(value) != len(getattr(self, key)):
                        raise ValueError(f"The length of '{key}' must match the original length")
                    setattr(self, key, value)    
                else:
                    raise ValueError(f"The length of '{key}' must match the original length")
            else: 
                if isinstance(getattr(self, key), list):
                    raise ValueError(f"The length of '{key}' must match the original length")
                setattr(self, key, value)    
    
    def inputs_validation(self):
        '''
        Validate inputs according to define rules and provide list of possible errors
        Rules:
            - Perimeter should be lower than area of building. Limitation building higher than 16m2
            - Transmittance values too hight or too low
            - Check area of the wall should be higher than the area of the window for the same orientation
            - Area of the floor slab on gorund should be lower than the area of the roof
        '''
        quality_check_errors = []
        # 1. Check perimeter and area
        if self.a_use >= 16:
            if self.exposed_perimeter > self.a_use:
                quality_check_errors.append("Possible error. Check the value of perimeter and area if they are correct.")
        
        #2. Check value of envelope transmittance 
        for i,u_value in enumerate(self.transmittance_U_elments):
            element = self.typology_elements[i]
            if element == 'OP':
                nameElement = "Opaque Element"
            elif element == 'W':
                nameElement = "Transaprent Element"
            elif element == 'HOR':
                nameElement = "Floor or Roof"
            
            orient_elment = self.orientation_elements[i]

            if u_value > 8 or u_value <=0.1:
                 quality_check_errors.append(f"Possible error. Transmittance of the element {nameElement} oriented to {orient_elment} too low or too hight")

        #3. Check area of wall and windows
        for i in range(4):
            area_wall = self.area_elements[i]
            area_window = self.area_elements[i+6]
            if area_wall == area_window:
                quality_check_errors.append(f"Error!. The area of the {self.orientation_elements[i]} wall is equal to the area of the {self.orientation_elements[i]}windows. ")

        # 4. Check area roof and floor slab on ground
        area_roof = self.area_elements[5]
        area_floor = self.area_elements[6]
        if area_floor == area_roof:
            quality_check_errors.append(f"Warning!. The area of the floor slab on ground is higher than the area of the roof")            


        print(quality_check_errors)
        return quality_check_errors
        

# ===================================================================================================
#                           GET INPUTS FROM SPECIFIC ARCHETYPE
# ===================================================================================================
class Selected_bui_archetype:
    def __init__(self, bui_archetype, period, latitude, longitude):
        self.bui_archetype = bui_archetype
        self.period = period
        self.latitude = latitude, 
        self.longitude = longitude

    
        if bui_archetype in bui_types:
                self.bui_archetype = bui_archetype
        else:
            raise ValueError(f"Invalid choice for archetype. Possible choices are: {', '.join(bui_types)}")

        if period in periods:
                self.built_year = period
        else:
            raise ValueError(f"Invalid choice for possible periods. Possible choices are: {', '.join(periods)}")
        
        if isinstance(latitude, float):
            self.latitude = latitude
        else: 
            raise ValueError("latitude should be a float")

        if isinstance(longitude, float):
            self.longitude = longitude            
        else: 
            raise ValueError("longitude should be a float")

    def Number_archetype(self):
        '''
        Get index of selected archetype
        '''
        # Filter accordiin gto building typology
        indices = [index for index, element in enumerate(bui_types) if element == self.bui_archetype]
        # Filter according to index position
        periods_selected_buiType = filter_list_by_index(periods, indices)
        index_sel = periods_selected_buiType.index(self.period)
        return index_sel


    def Get_bui_archetype(self):
        index_archetype = self.Number_archetype()
        return BuildingArchetype(
            building_type = self.bui_archetype,
            periods = self.period,
            latitude=self.latitude, 
            longitude=self.longitude,
            exposed_perimeter= perimeter[index_archetype],
            area=area[index_archetype], 
            height = height[index_archetype],
            number_of_floor=number_of_floor[index_archetype],
            volume= None,
            slab_on_ground = None, 
            wall_thickness = thickness_wall[code_wall.index(w_code[index_archetype])],
            coldest_month = coldest_month[index_archetype],
            surface_envelope = area_north[index_archetype]+area_south[index_archetype]+area_east[index_archetype]+area_west[index_archetype]+area[index_archetype]/number_of_floor[index_archetype]+area_roof[index_archetype],
            surface_envelope_model = S_envelope[index_archetype],
            side = base[index_archetype], 
            heating_mode = heating_installed[index_archetype], # True or False if heating syste is 
            cooling_mode =  cooling_installed[index_archetype], 
            heating_setpoint = 20, # Heating set-point in °C
            cooling_setpoint = 26, # Cooling set-point in °
            heating_setback = 12, # Heating set-back in °C
            cooling_setback = 26, # Cooling set-back in °C
            power_heating_max = Power_heating_system(volume[index_archetype], building_category_const[index_archetype]), # Max Power of the heating system
            power_cooling_max = 10000, # Max Power of the cooling system
            air_change_rate_base_value = air_change_rate_base_value[index_archetype]*area[index_archetype], # air change rate
            air_change_rate_extra = 0,
            internal_gains_base_value = Internal_gains('residential', area[index_archetype]),
            internal_gains_extra = 20,
            thermal_bridge_heat = thermal_bridge_heat[index_archetype], # value of thermal bridges
            thermal_resistance_floor = R_floor[code_floor.index(f_code[index_archetype])],
            area_elements = list((area_north[index_archetype], area_south[index_archetype], 
                     area_east[index_archetype],area_west[index_archetype],
                     area[index_archetype]/2, area_roof[index_archetype], 
                     round(0.1*window_area[index_archetype],2),round(0.3*window_area[index_archetype],2),
                     round(0.3*window_area[index_archetype],2),round(0.3*window_area[index_archetype],2))
            ), # Area of each facade elements 
            transmittance_U_elments = [U_wall[code_wall.index(w_code[index_archetype])]]*4+ [U_floor[code_floor.index(f_code[index_archetype])]] + [U_roof[code_roof.index(r_code[index_archetype])]] +[U_window[code_window.index(win_code[index_archetype])]]*4,
            thermal_resistance_R_elements = [R_wall[code_wall.index(w_code[index_archetype])]]*4+ [R_floor[code_floor.index(f_code[index_archetype])]] + [R_roof[code_roof.index(r_code[index_archetype])]] +[R_window[code_window.index(win_code[index_archetype])]]*4,
            thermal_capacity_elements = [heat_capacity_wall[code_wall.index(w_code[index_archetype])]]*4+ [heat_capacity_floor[code_floor.index(f_code[index_archetype])]] + [heat_capacity_roof[code_roof.index(r_code[index_archetype])]] +[0]*4,
            g_factor_windows = [0]*6 +[g_window[code_window.index(win_code[index_archetype])]]*4,     
            occ_level_wd= profile_residential_1['profile_workdays_internal_gains'],
            occ_level_we= profile_residential_1['profile_workdays_internal_gains'],
            comf_level_wd= profile_residential_1['profile_workdays_internal_gains'],
            construction_class= "class_i"
        )
    

# def main(latitude, longitude):
#     bui_inputs_archetype = []
#     for i,year in enumerate(periods):
#         bui_ = {
#             # BUILDING FEATURE
#             'type': bui_types[i], # building type
#             'year': year, # year of construction 
#             'latitude': latitude,
#             'longitude': longitude, 
#             'volume' : volume[i], # in m3
#             'exposed_perimeter': perimeter[i], # perimeter in m
#             'slab_on_ground_area': area[i]/number_of_floor[i], # Area slab on ground in m2
#             'wall_thickness' :  thickness_wall[code_wall.index(w_code[i])], # in m
#             'coldest_month': 1, 
#             'a_use': area[i],
#             'surface_envelope': area_north[i]+area_south[i]+area_east[i]+area_west[i]+area[i]/number_of_floor[i]+area_roof[i],
#             'surface_envelope_model': S_envelope[i],
#             # SYSTEMS 
#             'base': base[i],
#             "heating": True,
#             "cooling": True,
#             'heating_setpoint': 20, # in °c
#             'cooling_setpoint': 26, # in °c
#             'heating_setback':10, # in °c
#             'cooling_setback':26, # in °c
#             'power_heating_max':Power_heating_system(volume[i], building_category_const[i]), # in W
#             # 'power_heating_max':30000, # in W
#             'power_cooling_max':-10000, # in W
#             # INTERNAL GAINS and VENTILATION LOSSES
#             'air_change_rate_base_value':air_change_rate_base_value[i]*area[i] , # in m3/h*m2
#             # 'air_change_rate_base_value':1.1 , # in m3/h*m2
#             'air_change_rate_extra':0.0, # in m3/h*m2
#             'internal_gains_base_value':Internal_gains('residential', area[i]), # in W/m2
#             # 'internal_gains_base_value':5, # in W/m2
#             'internal_gains_extra':0.0, # in W/m2
#             # THERMAL BRIDGES
#             'thermal_bridge_heat' : 10.0, # in W/m
#             # FEATURES OF FAACDE ELEMENTS:
#             'thermal_resistance_floor': R_floor[code_floor.index(f_code[i])], 
#             # "Wall North", "Wall South", "Wall East", "Wall West", "Floor slab on gorund", "Roof", "Window North", "Window South", "Window East", "Window West"
#             'typology_elements': np.array(["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"],dtype=object), 
#             'orientation_elements': np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV'],dtype=object),
#             'solar_area_elements': np.array([1.0,1.0,1.0,1.0,0.0,1.0,0.6,0.6,0.6,0.6], dtype=object),
#             'area_elements': [area_north[i], area_south[i], area_east[i],area_west[i],area[i]/2, area_roof[i], 
#                     round(0.1*window_area[i],2),round(0.3*window_area[i],2),
#                     round(0.3*window_area[i],2),round(0.3*window_area[i],2),
#                     ],
#             'transmittance_U_elments' : [U_wall[code_wall.index(w_code[i])]]*4+ [U_floor[code_floor.index(f_code[i])]] + [U_roof[code_roof.index(r_code[i])]] +[U_window[code_window.index(win_code[i])]]*4,
#             'thermal_resistance_R_elements' : [R_wall[code_wall.index(w_code[i])]]*4+ [R_floor[code_floor.index(f_code[i])]] + [R_roof[code_roof.index(r_code[i])]] +[R_window[code_window.index(win_code[i])]]*4,
#             'thermal_capacity_elements' : [heat_capacity_wall[code_wall.index(w_code[i])]]*4+ [heat_capacity_floor[code_floor.index(f_code[i])]] + [heat_capacity_roof[code_roof.index(r_code[i])]] +[0]*4,
#             'g_factor_windows' : [0]*6 +[g_window[code_window.index(win_code[i])]]*4,
#             'heat_convective_elements_internal': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
#             'heat_radiative_elements_internal': np.array([5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13],dtype=object),
#             'heat_convective_elements_external': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
#             'heat_radiative_elements_external': np.array([4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14, 4.14],dtype=object),
#             'sky_factor_elements': np.array([0.50, 0.50, 0.50, 0.50, 0.00, 1.00, 0.50, 0.50, 0.50, 0.50], dtype=object), 
#             'occ_level_wd': profile_residential_1['profile_workdays_internal_gains'],
#             'occ_level_we': profile_residential_1['profile_weekend_internal_gains'],
#             'comf_level_wd': profile_residential_1['profile_workdays_ventilation'],
#             'baseline_hci': np.array([2.50, 2.50, 2.50, 2.50, 0.70, 5.00, 2.50, 2.50, 2.50, 2.50], dtype=object),
#             'baseline_hce': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],dtype=object),
#             "construction_class": "class_i"
#         }

#         bui_inputs_archetype.append(bui_)
        
    
#     return bui_inputs_archetype
    
#     # Save the JSON data to a file
#     # with open("building_archetype.json", "w") as file:
#     #     file.write(json.dumps(bui_inputs_aryhetype))


# if __name__ == "__main__":
#     main()
    
# bii_1 = bui_inputs_aryhetype[0]


