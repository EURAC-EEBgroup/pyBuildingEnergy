"""Main module."""


# #%%
# # import sys
# # print(sys.path)
# from pybuildingenergy.src.utils import __ISO52010__, __ISO52016__, bui_item
# from pybuildingenergy.src.building_stock import Building_archetype

# from pybuildingenergy.src.functions import Filter_list_by_indices, Scatter_with_regression,Simple_regeression
# from scipy.stats import linregress

# # GET ARCHETYPE
# inizialize = Building_archetype('single_fammily_house','before 1900',44.66345144066082, 10.323822015417987)
# inputs_archetype = inizialize.Get_archetype()

# inputs_user = {
#     'url_api': "http://127.0.0.1:8000/api/v1", 
#     'latitude':46.66345144066082,
#     'longitude':9.71636944229362,
#     'Eln':10, #
#     'a_use': 100, 
#     "slab_on_ground_area":100,#
#     'H_setpoint':22,      
#     'C_setpoint':24,
#     'Phi_H_nd_max':40000,            
#     'Phi_C_nd_max':-10000,
# }

# inputs_user = []

# new_inputs = inizialize.Set_own_values(inputs_archetype,inputs_user)
# #%%
# # SIMULATE ARCHETYPE
# hourly_sim = __ISO52016__(inputs_archetype).Temperature_and_Energy_needs_calculation() 






#%%
from pybuildingenergy.src.utils import __ISO52010__, __ISO52016__, bui_item, MyClass
from pybuildingenergy.data.building_archetype import Selected_bui_archetype
from pybuildingenergy.src.graphs import __Graphs__

# GET OBJECT ARCHETYPE
inizialize_building = Selected_bui_archetype('single_fammily_house','before 1900',45.47450206750902, 9.166056263772619)
BUI = inizialize_building.Get_bui_archetype()
## inputs_user = {
#     'latitude':46.66345144066082,
#     'longitude':9.71636944229362,
#     'a_use': 100, 
#     "slab_on_ground":100,#
#     'H_setpoint':22,     
#     'C_setpoint':24,
#     'Phi_H_nd_max':40000,            
#     'Phi_C_nd_max':-10000,
#     'volume':400,
#     'A_eli': [0, 76.69, 53.3, 53.3, 69.5, 73.46, 1.74, 5.22]
# }
# BUI.update_values(inputs_user)
# BUI.inputs_validation()
# BUI.A_eli

#%%
hourly_sim = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI) 
__Graphs__(hourly_sim,'heating_cooling').bui_analysis_page()


