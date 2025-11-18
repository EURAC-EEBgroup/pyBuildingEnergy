'''
Evaluation of heat transfer coefficent by ventilation using:
- the iso 16798-7 based on natural ventilation through windows using wind velocity and temperature difference (internal - external) inputs
- simplified method that take into account the occupancy profile

# To integrate:
- leaks 
- cross-ventilation 
- mechanical ventilation
'''

from dataclasses import dataclass
import numpy as np
import warnings
from source.table_iso_16798_1 import *

@dataclass
class h_natural_vent:
    H_ve_nat: np.ndarray

class VentilationInternalGains:
    def __init__(self, building_object):
        self.building_object = building_object

    @staticmethod
    def heat_transfer_coefficient_by_ventilation(
        building_object, Tz, Te, u_site, Rw_arg_i=None, c_air=1006, 
        rho_air=1.204, C_wnd=0.001, C_st=0.0035, rho_a_ref=1.204, altitude=None, type_ventilation="temp_wind", 
        flowrate_person=1.4
    ) -> h_natural_vent:
        """
        Calculate the heat transfer coefficient by ventilation  for air flow element K, Hve_k_t.
        (section 6.5.10.1 of ISO52016-1:2017)

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))
        :param c_air: specific heat of air at constant pressure [J/(kg K)]. Default: 1006
        :param rho_air: ir density at 20 °C [kg/m3]. Default: 1.204
        :param Tz: indoor temperature [°C] at time t
        :param Te: external temperature [°C] at time t
        :param u_site: air velocity of wind in the building site [m/s] at time t
        :param altitude: altitude of the building [m]
        :param Rw_arg_i: ratio of window opening area to maximum window opening area [0-1], type List
        :param type_ventilation: type of ventilation, "temp_wind" or "occupancy". Default: "temp_wind"
        :param flowrate_person: ventilation rate per person per m2 [l/(s m2)]

        : return 
            * **Hve_k_t**: heat transfer coefficient by ventilation  for air flow element K, Hve_k_t.

        """

        """
        According to the ISO 16798-7:2017
        calculation of air flow rate through window using wind velocity and temperature difference (6.4.3.5.4)
        
        # Single side ventilation 
        qv_arg_in = 3600*rho_a_ref/rho_a_e*Aw_tot/2*max(C_wnd*u_site^2, C_st*hw_st*abs(Tz-Te))^0.5 
        
        where: 
        qv_arg_in: air flow rate in [m3/s]
        rho_a_ref: air density at reference temperature [kg/m3]
        rho_a_e: air density at external temperature [kg/m3]
        Aw_tot: total area of the window [m2]
                calculated as : 
                Aw_tot = sum(i=1 to Nw)Aw_i
                Aw_i = Aw_tot_i * Rw_arg_i

                where:
                Rw_arg_i: Ratio of window opening area to maximum window opening area [0-1]
        C_wnd: coefficient taking into account wind speed in airing calculaton 1/[m/s] = 0.001 - Table 11
        u_site: wind velocity at the site [m/s]
        C_st: Coefficent taking into account stack effect in airing calculation (m/s)/(m*K) =  0.0035 - Table 11
        hw_st: useful height for stack effect for airing calculated as: 
            
            hw_st = max(i=1 to Nw)(hw_path_i + hw_fa_i/2) - min(i=1 to Nw)(hw_path_i - hw_fa_i/2)
            
            where:
            hw_path_i: height of the vertical center of the window, measured from the floor of the zone to which the window belongs [m]
            hw_fa_i: height of the vertical center of the window, measured from the floor of the zone to which the window belongs [m]

        Tz: internal temperature [°C]
        Te: external temperature [°C]



        """
        if type_ventilation == "temp_wind":
            # 1) Get number of windows
            n_windows = len([surface["name"] for surface in building_object["building_surface"] if surface.get("type") == "transparent"])
            
            # 2-3) Calculate hw_path_i and hw_fa_i
            width_window = []
            height_parapet = []
            height_window = []
            area_window = []

            for surface in building_object["building_surface"]:
                if surface.get("type") == "transparent":
                    # collect if the key exists
                    if "parapet" in surface:
                        height_parapet.append(surface["parapet"])
                    if "height" in surface:
                        height_window.append(surface["height"])
                    if "width" in surface:
                        width_window.append(surface["width"])
                    if "height" in surface and "width" in surface:
                        area_window.append(surface["height"] * surface["width"])

            hw_path_i = height_parapet + height_window/2
            hw_fa_i = height_window/2
        
            # 4) Calculate hw_st
            hw_st = float(np.max(hw_path_i + hw_fa_i/2) - np.min(hw_path_i - hw_fa_i/2))
        
            # 5) Adjust rho_a_ref for altitude
            if altitude is not None:
                rho_a_ref = round(1.204 * (1 - (0.00651 * altitude)/293)**4.255, 3)
        
            # 6) Density of air at external temperature
            rho_a_e = (291.15/(273.15 + Te)) * rho_a_ref
        
            # 7) Effective window area
            if Rw_arg_i is None:
                Aw_i = area_window * 0.9
            else:
                if len(Rw_arg_i) != n_windows:
                    warnings.warn("Length of Rw_arg_i != number of windows. Using 0.9 for all.")
                    Aw_i = area_window * 0.9
                else:
                    Aw_i = area_window * np.array(Rw_arg_i)
        
            # 8) Total window area
            Aw_tot = float(np.sum(Aw_i))
        
            # 9) Calculate flow rate (CORRECTED)
            wind_term = C_wnd * (u_site**2)
            stack_term = C_st * hw_st * np.abs(Tz - Te)
            qv_arg_in = 3600 * rho_a_ref / rho_a_e * Aw_tot / 2 * (np.maximum(wind_term, stack_term)**0.5)
        
            # 10) Heat transfer coefficient
            Hve_k_t = c_air * rho_air * qv_arg_in / 3600  # [W/K]
        
        elif type_ventilation == "occupancy":
            zone_area = building_object["building"]['net_floor_area']
            Hve_k_t = zone_area * (3.6 * flowrate_person) * rho_air * c_air / 3600
        
        return np.array(Hve_k_t)


    def internal_gains(
        cls, building_type_class, a_use, unconditioned_zones_nearby=False, list_adj_zones=None, Fztc_ztu_m: float=1, b_ztu: float=1,
        h_occup: float=1, h_app: float=1, h_light: float=1, h_dhw: float=1, h_hvac: float=1, h_proc: float=1
        ):
        """
        Calculation of internal gains (6.5.12.1 Overall internal heat gains into thermally conditioned zone )

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

        Formula: 
        
        Phi_int_z_t [W] = (q_int_occ + q_int_app + q_int_light + q_int_dhw + q_int_hvac + q_int_proc) [W/m2] * A_use [m2]

        where: 
        
        q_int_occ: internal heat flow due to metabloic heat M1-6 - iso 16798-1
        q_int_app: internal heat flow due to equipment M1-6 - iso 16798-1
        q_int_light: internal heat flow due to lighting M9-1 - iso 151931-1
        q_int_dhw: internal heat flow due to domestic hot water M3-1, M8-1 - iso 15316-1
        q_int_hvac: internal heat flow due to HVAC M10 - iso 15316-1. 16798-9, 16798-3
        q_int_proc: internal heat flow due to processes heat M1-6 - iso 16798-1
        A_use: useful area of the building [m2]
        
        :return:
            * Phi_int: internal gains [W]
        """

        q_int_occ = internal_gains_occupants[building_type_class]['occupants']
        q_int_app = internal_gains_occupants[building_type_class]['appliances']
        # q_int_light = internal_gains_occupants[building_type_class]['lighting']
        Phi_int_z_t = (q_int_occ * h_occup + q_int_app * h_app) * a_use

        if unconditioned_zones_nearby:
            Phi_int_dir_z_t = (q_int_occ * h_occup + q_int_app * h_app) * a_use
            for zones in range(list_adj_zones):
                Phi_int_z_t +=Phi_int_dir_z_t + (1-b_ztu)*Fztc_ztu_m * Phi_int_dir_z_t
        
        return float(Phi_int_z_t)


def transmission_heat_transfer_coefficient_ISO13789(adj_zone, n_ue=0.5, qui=0):
        '''
        Calcualtion of heat transfer coefficient, Htr calculated as
        Htr = Hd + Hg + Hu + Ha
        where:
        Hd: direct transmission heat transfer coefficient between the heated and cooled space and exterior trough the building envelope in W/K
        Hg:transmission trasnfer coefficient through the ground in W/K
        Hu:transmission heat transfer coefficent through unconditioned space
        Ha: transmision heat transfer coefficient to adjacent buildings
        '''
        # Sngle zone 
        # list_adj_zones = list(building_object.adj_zones.keys())
        # adj_zones = building_object.adj_zones
        # volume_all_zones = [] 
        # orient_all_zones= []
        # for zones in list_adj_zones:
        #     orient_all_zones.append(adj_zones[zones]['orientation'])    
        #     volume_all_zones.append(adj_zones[zones]['volume'])    

        '''
        1. Calculation of transmittance of wall closed to the unconditioned zone
        '''

        area_elements_zt = building_object.area_elements
        # eli_type_zt = building_object.typology_elements
        ori_type_zt = building_object.orientation_elements
        transmittance_eli_zt = building_object.transmittance_U_elements
        
        for orient_eli in orient_all_zones:
            area_eli_zt_ztu = np.array([a for a, t in zip(area_elements_zt, ori_type_zt) if t == orient_eli])
            transmittance_eli_zt_ztu = np.array([a for a, t in zip(transmittance_eli_zt, ori_type_zt) if t == orient_eli])
        
        Hd_zt_ztu = np.sum(area_eli_zt_ztu * transmittance_eli_zt_ztu)
            
        '''   
        2. Calculate the losses for transmission of the walls to the external environment of the non conditioned zone
        '''
        for orient_eli in orient_all_zones:
            area_eli_ztu_ext = np.array([a for a, t in zip(area_elements_zt, ori_type_zt) if t != orient_eli])
            transmittance_eli_ztu_ext = np.array([a for a, t in zip(transmittance_eli_zt, ori_type_zt) if t != orient_eli])

        Hd_ztu_ext = np.sum(area_eli_ztu_ext * transmittance_eli_ztu_ext)


        '''
        4. Calculate the losses for ventilation of the non conditioned zone Hve,iu and Hve,ue
        
        1) Hve,iu = rho*cp*qiu
        2) Hve,ue = rho*cp*que
        
        rho: density of air
        cp: specific heat capacity of air
        qiu: air flow rate in m3/h between conditioned and unconditioned zone   
        que: air flow rate in m3/h between unconditioned zone and external environment
        Note:
        rho_cp = 0.33 Wh/(m3K) if qiu in m3/h

        # air change rate of unconditioned spaces:
        in order to not understimate the transmission heat transfer, the air flow rate between a conditioned space and an unconditioned space 
        shall be assumed to be zero
        qiu= 0
        que = Vu * n_ue
        Vu: volume of the air in the unconditioned space
        n_ue: air change rate between the unconditioned space and the external environment in h-1. Can be taken from the table df_n_ue that is table 7  of ISO 13789

        default n_ue = 0.5 that is for building having All joints between components well-sealed, no ventilation opening provided

        '''    
        que = volume_all_zones[0] * n_ue
        Hve_ue =  0.33 * que
        Hve_iu = 0.33 * qui
        # 
        Hue = Hd_ztu_ext + Hve_ue
        Hiu = Hd_zt_ztu + Hve_iu
        H_ztu_tot = float(round(Hiu +Hue,3))
        
        '''
        Calculation of the adjustment factor for the thermally uncononditioned afjacent zone ztu for in month m
        b = Hztu_e_m/Hztu_tot_m

        where:
        H_ztu_tot_m = sum(j=1 to n)(H_ztc_j_ztu) + Hztu_e_m
        where:
        H_ztc_j_ztu: heat transfer coefficient between the thermally unconditioned zone and the adjacent thermally conditioned zone j  = Hiu (ISO 13789)
        Hztu_e_m: heat transfer coefficient between the thermally unconditioned zone and the external environment for month m =  Hue (ISO 13789)
        '''
        b_ztu_m = float(round(Hue/(Hue + Hiu),3))
        
        return H_ztu_tot, b_ztu_m

    # def Vent_heat_transf_coef_and_Int_gains(
    #     cls, building_object, path_weather_file, c_air=1006, rho_air=1.204
    # ) -> h_vent_and_int_gains:
    #     """
    #     Calculation of heat transfer coefficient (section 8 - ISO 13789:2017 and 6.6.6 ISO 52016:2017 ) and internal gains

    #     :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
    #     :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))
    #     :param c_air: specific heat of air at constant pressure [J/(kg K)]. Default: 1006
    #     :param rho_air: ir density at 20 °C [kg/m3]. Default: 1.204

    #     .. note:: Required parameters of building_object:

    #         * air_change_rate_base_value: ventilation air change rate [m3/h]
    #         * air_change_rate_extra: extra iar change in case of comfort values [m3/h]
    #         * a_use: useful area of the building [m2]
    #         * internal_gains_base_value: value fo internal gains [W/m2]
    #         * internal_gains_extra: eventual extra gains during [W]

    #     :return:
    #         * H_ve: heat transfer coefficient for ventilation [W/K]
    #         * Phi_int: internal gains [W]

    #     """
    #     # VENTILATION (CONDUTTANCE)
    #     sim_df = (
    #         ISO52016()
    #         .Occupancy_profile(building_object, path_weather_file)
    #         .simulation_df
    #     )
    #     comfort_hi_mask = sim_df["comfort level"] == 1
    #     sim_df["air flow rate"] = building_object.__getattribute__(
    #         "air_change_rate_base_value"
    #     ) * building_object.__getattribute__(
    #         "a_use"
    #     )  # [m3/h]
    #     sim_df.loc[comfort_hi_mask, "air flow rate"] += building_object.__getattribute__("air_change_rate_extra") * building_object.__getattribute__(
    #         "a_use"
    #     )
    #     air_flow_rate = sim_df["air flow rate"]
    #     H_ve = c_air * rho_air / 3600 * air_flow_rate  # [W/K]

    #     # INTERNAL GAINS
    #     occ_hi_mask = sim_df["occupancy level"] == 1
    #     sim_df["internal gains"] = building_object.__getattribute__(
    #         "internal_gains_base_value"
    #     ) * building_object.__getattribute__(
    #         "a_use"
    #     )  # [W]
    #     sim_df.loc[occ_hi_mask, "internal gains"] += building_object.__getattribute__(
    #         "internal_gains_extra"
    #     ) * building_object.__getattribute__(
    #         "a_use"
    #     )  # [W]
    #     Phi_int = sim_df["internal gains"]

    #     return h_vent_and_int_gains(H_ve=H_ve, Phi_int=Phi_int, sim_df_update=sim_df)
