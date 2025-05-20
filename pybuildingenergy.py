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


# %%
# import src.utils

from src.utils import __ISO52010__, __ISO52016__

from data.building_archetype import Selected_bui_archetype
from src.graphs import __Graphs__
import json

# Load building archetype
with open("pybuildingenergy/data/bui.json", "r") as f:
    BUI_JSON = json.load(f)

slab_on_ground_area = next((d for d in BUI_JSON["building_surface"] if d["name"] == "Slab on ground"), None)["area"]

# Create BuildingArchetype from BUI_JSON
BUI = BuildingArchetype(
            building_type=BUI_JSON["building"]["name"],
            latitude=BUI_JSON["building"]["latitude"],
            longitude=BUI_JSON["building"]["longitude"],
            exposed_perimeter=BUI_JSON["building"]["exposed_perimeter"],
            area=slab_on_ground_area,
            height=BUI_JSON["building"]["height"],
            number_of_floor=number_of_floor[index_archetype],
            volume=None,
            slab_on_ground=None,
            wall_thickness=thickness_wall[code_wall.index(w_code[index_archetype])],
            coldest_month=coldest_month[index_archetype],
            surface_envelope=area_north[index_archetype]
            + area_south[index_archetype]
            + area_east[index_archetype]
            + area_west[index_archetype]
            + area[index_archetype] / number_of_floor[index_archetype]
            + area_roof[index_archetype],
            surface_envelope_model=S_envelope[index_archetype],
            side=base[index_archetype],
            heating=heating_installed[
                index_archetype
            ],  # True or False if heating syste is
            cooling=cooling_installed[index_archetype],
            H_setpoint=20,  # Heating set-point in °C
            C_setpoint=26,  # Cooling set-point in °
            H_setback=12,  # Heating set-back in °C
            C_setback=26,  # Cooling set-back in °C
            Phi_H_nd_max=Power_heating_system(
                volume[index_archetype], building_category_const[index_archetype]
            ),  # Max Power of the heating system
            Phi_C_nd_max=10000,  # Max Power of the cooling system
            air_change_rate_base_value=air_change_rate_base_value[index_archetype]
            * area[index_archetype],  # air change rate
            air_change_rate_extra=0,
            internal_gains_base_value=Internal_gains(
                "residential", area[index_archetype]
            ),
            internal_gains_extra=20,
            H_tb=H_tb[index_archetype],  # value of thermal bridges
            R_floor_construction=R_floor[code_floor.index(f_code[index_archetype])],
            A_eli=list(
                (
                    area_north[index_archetype],
                    area_south[index_archetype],
                    area_east[index_archetype],
                    area_west[index_archetype],
                    area[index_archetype] / 2,
                    area_roof[index_archetype],
                    round(0.1 * window_area[index_archetype], 2),
                    round(0.3 * window_area[index_archetype], 2),
                    round(0.3 * window_area[index_archetype], 2),
                    round(0.3 * window_area[index_archetype], 2),
                )
            ),  # Area of each facade elements
            U_eli=[U_wall[code_wall.index(w_code[index_archetype])]] * 4
            + [U_floor[code_floor.index(f_code[index_archetype])]]
            + [U_roof[code_roof.index(r_code[index_archetype])]]
            + [U_window[code_window.index(win_code[index_archetype])]] * 4,
            R_eli=[R_wall[code_wall.index(w_code[index_archetype])]] * 4
            + [R_floor[code_floor.index(f_code[index_archetype])]]
            + [R_roof[code_roof.index(r_code[index_archetype])]]
            + [R_window[code_window.index(win_code[index_archetype])]] * 4,
            kappa_m_eli=[heat_capacity_wall[code_wall.index(w_code[index_archetype])]]
            * 4
            + [heat_capacity_floor[code_floor.index(f_code[index_archetype])]]
            + [heat_capacity_roof[code_roof.index(r_code[index_archetype])]]
            + [0] * 4,
            g_w_eli=[0] * 6
            + [g_window[code_window.index(win_code[index_archetype])]] * 4,
            occ_level_wd=profile_residential_1["profile_workdays_internal_gains"],
            occ_level_we=profile_residential_1["profile_workdays_internal_gains"],
            comf_level_wd=profile_residential_1["profile_workdays_internal_gains"],
            construction_class="class_i",
        )


inizialize_building = Selected_bui_archetype(
    "single_fammily_house", "before 1900", lat, lon
)
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

# %%
hourly_sim = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI)
__Graphs__(hourly_sim, "heating_cooling").bui_analysis_page()