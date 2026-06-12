'''
Ventilation heat-transfer coefficient and affine boundary for ISO 52016-1 §6.5.10.

Methods implemented:
- Natural ventilation through windows (EN 16798-7 wind/temperature-difference method)
- Occupancy-based simplified method
- Custom, EnergyPlus infiltration, and Sherman-Grimsrud infiltration models
- Affine ventilation boundary: additive VentilationStream objects and VentilationBoundary

Affine boundary
---------------
ISO 52016-1 §6.5.10 inserts ventilation into the zone energy balance as:

    A_air,air += H_ve
    B_air     += S_ve

where:

    H_ve = sum_k(H_k)                 [W/K]  aggregate conductance
    S_ve = sum_k(H_k * T_source,k)    [W]    weighted source term

The zone sensible ventilation heat flow (positive = heat leaving the zone) is:

    Q_ve = H_ve * T_zone - S_ve

No additional matrix temperature node is required.  T_supply_equivalent may be
reported as S_ve / H_ve when H_ve > 0, but the solver must retain the pair
(H_ve, S_ve) rather than reducing it to a single temperature or a
state-dependent effective conductance.

Existing five ventilation_type configurations resolve to one outdoor-air stream
(backward-compatible).  The optional ``components`` configuration enables
independent streams with per-component operation schedules.

To integrate (deferred):
- Detailed leakage and duct networks
- Cross-ventilation and inter-zone airflow
- Latent AHU processes, recirculation, duct losses, and unbalanced flow
'''

from dataclasses import dataclass
import math
import numpy as np
import warnings
from .table_iso_16798_1 import *

@dataclass
class h_natural_vent:
    H_ve_nat: np.ndarray


@dataclass(frozen=True)
class VentilationStream:
    """Single additive airflow element for the ISO 52016-1 §6.5.10 ventilation term.

    Sensible heat delivered to the zone from this stream (positive = heat into zone):
        Q_k = H_k * (T_source,k - T_zone)
    """
    name: str
    heat_transfer_coefficient_w_k: float   # H_k [W/K], must be >= 0 and finite
    source_temperature_c: float            # T_source,k [°C], must be finite
    category: str = "outdoor_air"

    def __post_init__(self):
        if not self.name:
            raise ValueError("VentilationStream name must be non-empty")
        if not math.isfinite(self.heat_transfer_coefficient_w_k):
            raise ValueError(
                f"Stream {self.name!r}: H_k must be finite, "
                f"got {self.heat_transfer_coefficient_w_k}"
            )
        if self.heat_transfer_coefficient_w_k < 0.0:
            raise ValueError(
                f"Stream {self.name!r}: H_k must be >= 0, "
                f"got {self.heat_transfer_coefficient_w_k}"
            )
        if not math.isfinite(self.source_temperature_c):
            raise ValueError(
                f"Stream {self.name!r}: source_temperature_c must be finite, "
                f"got {self.source_temperature_c}"
            )


@dataclass(frozen=True)
class VentilationBoundary:
    """Aggregate affine ventilation boundary for one ISO 52016-1 zone at one timestep.

    ISO 52016-1 §6.5.10 zone-air matrix assembly:
        A_air,air += H_ve          (= sum of all stream H_k)
        B_air     += S_ve          (= sum of H_k * T_source,k across all streams)

    Zone sensible ventilation heat flow, positive = heat leaving the zone:
        Q_ve = H_ve * T_zone - S_ve
    """
    streams: tuple  # tuple[VentilationStream, ...]

    def __post_init__(self):
        # Coerce any iterable to tuple so the boundary is truly immutable
        object.__setattr__(self, "streams", tuple(self.streams))
        for s in self.streams:
            if not isinstance(s, VentilationStream):
                raise TypeError(
                    f"streams must contain VentilationStream instances, got {type(s)}"
                )
        names = [s.name for s in self.streams]
        if len(names) != len(set(names)):
            dupes = [n for n in set(names) if names.count(n) > 1]
            raise ValueError(f"Duplicate stream names in boundary: {dupes}")

    @property
    def heat_transfer_coefficient_w_k(self) -> float:
        """Aggregate H_ve = sum(H_k) [W/K]."""
        return sum(s.heat_transfer_coefficient_w_k for s in self.streams)

    @property
    def source_term_w(self) -> float:
        """Aggregate S_ve = sum(H_k * T_source,k) [W]."""
        return sum(
            s.heat_transfer_coefficient_w_k * s.source_temperature_c
            for s in self.streams
        )

    @property
    def equivalent_supply_temperature_c(self):
        """Flow-weighted equivalent source temperature S_ve / H_ve [°C], or None when H_ve = 0."""
        h = self.heat_transfer_coefficient_w_k
        if h == 0.0:
            return None
        return self.source_term_w / h

    def sensible_heat_flow_w(self, zone_temperature_c: float) -> float:
        """Zone ventilation heat flow [W]; positive = heat leaving the zone."""
        return self.heat_transfer_coefficient_w_k * zone_temperature_c - self.source_term_w

class VentilationInternalGains:
    def __init__(self, building_object):
        self.building_object = building_object

    @staticmethod


    def heat_transfer_coefficient_by_ventilation(
            building_object, Tz, Te, u_site, Rw_arg_i=None, c_air=1006,
            rho_air=1.204, C_wnd=0.001, C_st=0.0035, rho_a_ref=1.204, altitude=None,
            type_ventilation="temp_wind", flowrate_per_area=1.4, custom_Hve_k_t=3,
            flowrate_person=None
        ):
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
        :param type_ventilation: type of ventilation, "temp_wind", "occupancy",
            "custom", "eplus_infiltration_ext_area", or
            "sherman_grimsrud_like". Default: "temp_wind"
        :param flowrate_per_area: ventilation rate per unit area [l/(s m2)]
        :param flowrate_person: backward-compatible alias of flowrate_per_area.

        : return 
            * **Hve_k_t**: heat transfer coefficient by ventilation  for air flow element K, Hve_k_t. in W/K

        """

        """
        According to EN 16798-7:2017
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

        # Backward compatibility: older call sites use `flowrate_person`.
        if flowrate_person is not None:
            flowrate_per_area = flowrate_person

        # Parameter lookup precedence:
        # 1) top-level keys on provided dict (zone-like object)
        # 2) building_parameters.ventilation (building/global object)
        def _vent_param(key, default):
            if isinstance(building_object, dict):
                if key in building_object and building_object.get(key) is not None:
                    return building_object.get(key)
                vent = building_object.get("building_parameters", {}).get("ventilation", {})
                if isinstance(vent, dict) and key in vent and vent.get(key) is not None:
                    return vent.get(key)
            return default

        if type_ventilation == "temp_wind":
            c_wnd_eff = float(_vent_param("temp_wind_c_wnd", C_wnd))
            if not np.isfinite(c_wnd_eff):
                c_wnd_eff = float(C_wnd)
            c_wnd_eff = max(0.0, c_wnd_eff)

            c_st_eff = float(_vent_param("temp_wind_c_st", C_st))
            if not np.isfinite(c_st_eff):
                c_st_eff = float(C_st)
            c_st_eff = max(0.0, c_st_eff)

            rho_a_ref_cfg = float(_vent_param("temp_wind_rho_a_ref", rho_a_ref))
            if not np.isfinite(rho_a_ref_cfg) or rho_a_ref_cfg <= 0.0:
                rho_a_ref_cfg = float(rho_a_ref)

            rw_default = float(_vent_param("temp_wind_opening_ratio", 0.9))
            if not np.isfinite(rw_default):
                rw_default = 0.9
            rw_default = min(1.0, max(0.0, rw_default))

            # --- Collect transparent surfaces as "windows"
            windows = [s for s in building_object.get("building_surface", []) if s.get("type") == "transparent"]
            n_windows = len(windows)

            if n_windows == 0:
                return np.array(0.0)

            # Build per-window arrays (robust to missing keys)
            parapet_list = []
            height_list = []
            width_list = []

            for s in windows:
                h = s.get("height", None)
                w = s.get("width", None)

                # Need height+width to compute area; if missing, skip and warn
                if h is None or w is None:
                    warnings.warn(
                        f"Transparent surface '{s.get('name','<unnamed>')}' missing 'height' or 'width'. Skipped in ventilation calc."
                    )
                    continue

                parapet_list.append(float(s.get("parapet", 0.0)))
                height_list.append(float(h))
                width_list.append(float(w))

            # If all windows were skipped
            if len(height_list) == 0:
                return np.array(0.0)

            parapet = np.asarray(parapet_list, dtype=float)
            height = np.asarray(height_list, dtype=float)
            width = np.asarray(width_list, dtype=float)

            # EN 16798-7: hw_path = center height from floor; hw_fa = opening height (use full height)
            hw_path_i = parapet + 0.5 * height
            hw_fa_i = height

            # Useful height for stack effect
            hw_st = float(np.max(hw_path_i + 0.5 * hw_fa_i) - np.min(hw_path_i - 0.5 * hw_fa_i))

            # Adjust rho_a_ref for altitude (if provided)
            if altitude is not None:
                rho_a_ref_eff = round(rho_a_ref_cfg * (1 - (0.00651 * altitude) / 293.0) ** 4.255, 3)
            else:
                rho_a_ref_eff = float(rho_a_ref_cfg)

            # Air density at external temperature
            rho_a_e = (293.15 / (273.15 + float(Te))) * rho_a_ref_eff

            # Max opening area per window
            Aw_max = height * width  # [m2]

            # Effective opening ratio
            if Rw_arg_i is None:
                Rw = np.full(Aw_max.shape, rw_default, dtype=float)
            else:
                Rw_arg_i = list(Rw_arg_i)
                if len(Rw_arg_i) != Aw_max.size:
                    warnings.warn(
                        "Length of Rw_arg_i != number of usable windows. "
                        f"Using {rw_default:.3f} for all."
                    )
                    Rw = np.full(Aw_max.shape, rw_default, dtype=float)
                else:
                    Rw = np.asarray(Rw_arg_i, dtype=float)
            Rw = np.clip(Rw, 0.0, 1.0)

            Aw_i = Aw_max * Rw
            Aw_tot = float(np.sum(Aw_i))

            # Airflow (kept consistent with your existing implementation: result in m3/h)
            wind_term = c_wnd_eff * (float(u_site) ** 2)
            stack_term = c_st_eff * hw_st * abs(float(Tz) - float(Te))
            qv_m3_s = (rho_a_ref_eff / rho_a_e) * (Aw_tot / 2.0) * (max(wind_term, stack_term) ** 0.5)

            # Heat transfer coefficient [W/K]
            Hve_k_t = c_air * rho_air * qv_m3_s

        elif type_ventilation == "occupancy":
            # Legacy naming: flow_rate_per_person is used as an area-based
            # specific airflow [L/(s m2)] in this branch.
            flowrate_per_area = float(_vent_param("flow_rate_per_person", flowrate_per_area))
            if not np.isfinite(flowrate_per_area):
                flowrate_per_area = 0.0
            flowrate_per_area = max(0.0, flowrate_per_area)
            zone_area = float(building_object["building"]["net_floor_area"])
            if not np.isfinite(zone_area):
                zone_area = 0.0
            zone_area = max(0.0, zone_area)
            if altitude is not None:
                rho_air_eff = 1.204 * (1 - (0.00651 * altitude) / 293.0) ** 4.255
            else:
                rho_air_eff = rho_air
            qv_m3_s = flowrate_per_area * zone_area / 1000.0
            Hve_k_t = rho_air_eff * c_air * qv_m3_s

        elif type_ventilation == "eplus_infiltration_ext_area":
            # Emulates EnergyPlus ZoneInfiltration:DesignFlowRate (Flow/ExteriorArea)
            # q = q_design * (A + B*|Tz-Te| + C*V + D*V^2)
            q_per_ext = float(_vent_param("infiltration_flow_per_exterior_area_m3_s_m2", 0.0))
            coef_a = float(_vent_param("infiltration_coeff_constant", 0.0))
            coef_b = float(_vent_param("infiltration_coeff_temperature", 0.0))
            coef_c = float(_vent_param("infiltration_coeff_velocity", 0.0))
            coef_d = float(_vent_param("infiltration_coeff_velocity_squared", 0.0))
            include_transparent = bool(_vent_param("infiltration_include_transparent_area", True))
            wind_reduction_factor = float(_vent_param("infiltration_wind_reduction_factor", 1.0))
            if not np.isfinite(wind_reduction_factor):
                wind_reduction_factor = 1.0
            wind_reduction_factor = max(0.0, wind_reduction_factor)
            ext_area_mode_raw = _vent_param("infiltration_exterior_area_mode", "outdoors_only")
            ext_area_mode = str(ext_area_mode_raw).strip().lower()
            if ext_area_mode in ("", "default", "solver_default", "outdoors"):
                ext_area_mode = "outdoors_only"
            elif ext_area_mode in ("energyplus", "energyplus_like", "include_ground_like"):
                ext_area_mode = "energyplus_like"
            elif ext_area_mode != "outdoors_only":
                warnings.warn(
                    "Unknown infiltration_exterior_area_mode="
                    f"{ext_area_mode_raw!r}; falling back to 'outdoors_only'."
                )
                ext_area_mode = "outdoors_only"
            schedule_mult = float(_vent_param("infiltration_schedule_multiplier", 1.0))
            if not np.isfinite(schedule_mult):
                schedule_mult = 1.0
            schedule_mult = max(0.0, schedule_mult)

            # Optional zone scoping (defensive):
            # when available, restrict exterior area to the requested thermal zone.
            target_zone = building_object.get("zone_name", None)
            if target_zone is None:
                target_zone = building_object.get("building", {}).get("zone_name", None)
            target_zone = None if target_zone is None else str(target_zone)

            surfaces = building_object.get("building_surface", [])
            zones_in_surfaces = sorted(
                {
                    str(s.get("zone"))
                    for s in surfaces
                    if isinstance(s, dict) and s.get("zone") is not None
                }
            )
            if target_zone is None and len(zones_in_surfaces) == 1:
                # If the provided object already contains one zone only, scope automatically.
                target_zone = zones_in_surfaces[0]
            elif target_zone is None and len(zones_in_surfaces) > 1:
                warnings.warn(
                    "eplus_infiltration_ext_area called without zone_name on a multi-zone "
                    "surface set; using building-level exterior area. Pass zone_name (or a "
                    "zone-filtered building_surface) for per-zone infiltration."
                )

            ext_area = 0.0
            allowed_boundaries = {"OUTDOORS"}
            if ext_area_mode == "energyplus_like":
                allowed_boundaries.update(
                    {
                        "GROUND",
                        "FOUNDATION",
                        "OTHERSIDECONDITIONSMODEL",
                        "GROUNDFCFACTORMETHOD",
                        "GROUNDSLABPREPROCESSORAVERAGE",
                        "GROUNDSLABPREPROCESSORCORE",
                        "GROUNDSLABPREPROCESSORPERIMETER",
                        "GROUNDBASEMENTPREPROCESSORAVERAGEWALL",
                        "GROUNDBASEMENTPREPROCESSORAVERAGEFLOOR",
                        "GROUNDBASEMENTPREPROCESSORUPPERWALL",
                        "GROUNDBASEMENTPREPROCESSORLOWERWALL",
                    }
                )
            for surf in surfaces:
                if target_zone is not None:
                    surf_zone = surf.get("zone", None)
                    if surf_zone is not None and str(surf_zone) != target_zone:
                        continue
                boundary = str(surf.get("boundary", "")).upper().replace(" ", "")
                if boundary not in allowed_boundaries:
                    continue
                if (not include_transparent) and str(surf.get("type", "")).lower() == "transparent":
                    continue
                try:
                    ext_area += max(0.0, float(surf.get("area", 0.0)))
                except Exception:
                    continue

            q_design_m3_s = q_per_ext * ext_area
            u_site_eff = float(u_site) * wind_reduction_factor
            multiplier = (
                coef_a
                + coef_b * abs(float(Tz) - float(Te))
                + coef_c * u_site_eff
                + coef_d * (u_site_eff ** 2)
            )
            if not np.isfinite(multiplier):
                multiplier = 0.0
            qv_m3_s = max(0.0, q_design_m3_s * multiplier * schedule_mult)
            Hve_k_t = rho_air * c_air * qv_m3_s

        elif type_ventilation in ("sherman_grimsrud_like", "sherman_grimsrud", "ela_stack_wind"):
            # Sherman-Grimsrud-like leakage model:
            #   Vdot_inf = F(t) * ELA * sqrt(Cs * |Tz-Te| + Cw * WS^2)
            # Units must be coherent so that Vdot_inf is in m3/s.
            ela_m2 = float(_vent_param("infiltration_effective_leakage_area_m2", 0.0))
            if not np.isfinite(ela_m2):
                ela_m2 = 0.0
            ela_m2 = max(0.0, ela_m2)

            c_stack = float(_vent_param("infiltration_stack_coefficient", 0.0))
            if not np.isfinite(c_stack):
                c_stack = 0.0
            c_stack = max(0.0, c_stack)

            c_wind = float(_vent_param("infiltration_wind_coefficient", 0.0))
            if not np.isfinite(c_wind):
                c_wind = 0.0
            c_wind = max(0.0, c_wind)

            f_t = float(_vent_param("infiltration_schedule_multiplier", 1.0))
            if not np.isfinite(f_t):
                f_t = 1.0
            f_t = max(0.0, f_t)

            delta_t = abs(float(Tz) - float(Te))
            ws = max(0.0, float(u_site))
            root_term = c_stack * delta_t + c_wind * (ws ** 2)
            if not np.isfinite(root_term):
                root_term = 0.0
            root_term = max(0.0, root_term)

            qv_m3_s = f_t * ela_m2 * (root_term ** 0.5)
            Hve_k_t = rho_air * c_air * max(0.0, qv_m3_s)

        elif type_ventilation == "custom":
            custom_h_ve = float(
                _vent_param("custom_heat_transfer_coefficient_ventilation", custom_Hve_k_t)
            )
            if not np.isfinite(custom_h_ve):
                custom_h_ve = 0.0
            Hve_k_t = max(0.0, custom_h_ve)

        else:
            raise ValueError(
                "type_ventilation must be one of: 'temp_wind', 'occupancy', "
                "'custom', 'eplus_infiltration_ext_area', "
                "'sherman_grimsrud_like'."
            )

        return np.array(Hve_k_t)

    # def heat_transfer_coefficient_by_ventilation(
    #     building_object, Tz, Te, u_site, Rw_arg_i=None, c_air=1006, 
    #     rho_air=1.204, C_wnd=0.001, C_st=0.0035, rho_a_ref=1.204, altitude=None, type_ventilation="temp_wind", 
    #     flowrate_per_area=1.4, custom_Hve_k_t=3
    # ) -> h_natural_vent:
        
    #     if type_ventilation == "temp_wind":
    #         # 1) Get number of windows
    #         n_windows = len([surface["name"] for surface in building_object["building_surface"] if surface.get("type") == "transparent"])
            
    #         # 2-3) Calculate hw_path_i and hw_fa_i
    #         width_window = []
    #         height_parapet = []
    #         height_window = []
    #         area_window = []

    #         for surface in building_object["building_surface"]:
    #             if surface.get("type") == "transparent":
    #                 # collect if the key exists
    #                 if "parapet" in surface:
    #                     height_parapet.append(surface["parapet"])
    #                 if "height" in surface:
    #                     height_window.append(surface["height"])
    #                 if "width" in surface:
    #                     width_window.append(surface["width"])
    #                 if "height" in surface and "width" in surface:
    #                     area_window.append(surface["height"] * surface["width"])

    #         hw_path_i = height_parapet + height_window/2
    #         hw_fa_i = height_window/2
        
    #         # 4) Calculate hw_st
    #         hw_st = float(np.max(hw_path_i + hw_fa_i/2) - np.min(hw_path_i - hw_fa_i/2))
        
    #         # 5) Adjust rho_a_ref for altitude
    #         if altitude is not None:
    #             rho_a_ref = round(1.204 * (1 - (0.00651 * altitude)/293)**4.255, 3)
        
    #         # 6) Density of air at external temperature
    #         rho_a_e = (291.15/(273.15 + Te)) * rho_a_ref
        
    #         # 7) Effective window area
    #         if Rw_arg_i is None:
    #             Aw_i = area_window * 0.9
    #         else:
    #             if len(Rw_arg_i) != n_windows:
    #                 warnings.warn("Length of Rw_arg_i != number of windows. Using 0.9 for all.")
    #                 Aw_i = area_window * 0.9
    #             else:
    #                 Aw_i = area_window * np.array(Rw_arg_i)
        
    #         # 8) Total window area
    #         Aw_tot = float(np.sum(Aw_i))
        
    #         # 9) Calculate flow rate (CORRECTED)
    #         wind_term = C_wnd * (u_site**2)
    #         stack_term = C_st * hw_st * np.abs(Tz - Te)
    #         qv_arg_in = 3600 * rho_a_ref / rho_a_e * Aw_tot / 2 * (np.maximum(wind_term, stack_term)**0.5)
        
    #         # 10) Heat transfer coefficient
    #         Hve_k_t = c_air * rho_air * qv_arg_in / 3600  # [W/K]
        
    #     elif type_ventilation == "occupancy":
    #         zone_area = building_object["building"]['net_floor_area']
    #         Hve_k_t = zone_area * (3.6 * flowrate_per_area) * rho_air * c_air / 3600

    #     elif type_ventilation == "custom":
    #         Hve_k_t = custom_Hve_k_t
        
    #     return np.array(Hve_k_t)


    def internal_gains(
        self, building_type_class, a_use, unconditioned_zones_nearby=False, list_adj_zones=None, Fztc_ztu_m: float=1, b_ztu: float=1,
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

        # Default full-load gains [W/m2] from ISO table
        gains_row = internal_gains_occupants[building_type_class]
        q_int_occ = float(gains_row.get("occupants", 0.0))
        q_int_app = float(gains_row.get("appliances", 0.0))
        q_int_light = float(gains_row.get("lighting", 0.0))

        # Safety against NaN values in the source table
        if not np.isfinite(q_int_occ):
            q_int_occ = 0.0
        if not np.isfinite(q_int_app):
            q_int_app = 0.0
        if not np.isfinite(q_int_light):
            q_int_light = 0.0

        # Optional full-load overrides from BUI -> internal_gains
        building_object = getattr(self, "building_object", None)
        if building_object:
            for gain in building_object.get("internal_gains", []):
                gname = gain.get("name")
                full_load = gain.get("full_load")
                if full_load is None:
                    continue
                try:
                    full_load = float(full_load)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(full_load):
                    continue
                if gname == "occupants":
                    q_int_occ = full_load
                elif gname == "appliances":
                    q_int_app = full_load
                elif gname == "lighting":
                    q_int_light = full_load

        q_int_total = q_int_occ * h_occup + q_int_app * h_app + q_int_light * h_light
        Phi_int_z_t = q_int_total * a_use

        if unconditioned_zones_nearby:
            Phi_int_dir_z_t = q_int_total * a_use
            for zones in range(list_adj_zones):
                Phi_int_z_t +=Phi_int_dir_z_t + (1-b_ztu)*Fztc_ztu_m * Phi_int_dir_z_t
        
        return float(Phi_int_z_t)


# ---------------------------------------------------------------------------
# Affine ventilation boundary resolver (ISO 52016-1 §6.5.10)
# ---------------------------------------------------------------------------

def _parse_source_temperature(spec, outdoor_temperature_c: float) -> float:
    """Resolve a source_temperature specification to a float [°C].

    Accepts:
    - the string "outdoor" (case-insensitive) → outdoor_temperature_c
    - any numeric value → that value cast to float
    """
    if isinstance(spec, str):
        if spec.strip().lower() == "outdoor":
            return outdoor_temperature_c
        try:
            return float(spec)
        except ValueError:
            raise ValueError(f"Unknown source_temperature string: {spec!r}")
    return float(spec)


def _resolve_component_streams(
    components: list,
    outdoor_temperature_c: float,
    zone_volume_m3,
    c_air: float,
    rho_air: float,
    component_multipliers: dict = None,
    default_multiplier: float = 1.0,
    zone_temperature_c: float = 20.0,
    ahu_outputs_collector: dict = None,
) -> list:
    """Convert a ventilation components list to VentilationStream objects.

    Supported component ventilation_type values:
    - 'constant_ach': infiltration or natural ventilation via air-change rate.
      Requires zone_volume_m3. source_temperature defaults to "outdoor".
    - 'prescribed': fixed H_k and source temperature supplied by caller (e.g. AHU).
    - 'mechanical_supply': physics-based AHU step via EN 16798-5-1 sensible model.
      Requires supply_flow_m3_h, sensible_heat_recovery_efficiency,
      supply_temperature_setpoint_c. zone_temperature_c is used as the extract
      temperature (previous-timestep zone AIR node temperature, not operative).

    component_multipliers: optional per-component name -> float.
      For 'mechanical_supply': a continuous operating-flow fraction relative to
      design flow. It is NOT timestep duty-cycle averaging; operation_fraction is
      always 1.0. A value of 0.25 means the AHU runs at 25% of design airflow
      continuously, with fan power following the EN 16798-5-1 reduced-flow relation.
      Values outside [0, 1] raise ValueError for 'mechanical_supply'.
      For 'constant_ach' and 'prescribed': scales H_k linearly and is not
      restricted to [0, 1]. The resulting H_k must still be finite and
      non-negative, as enforced by VentilationStream.
    default_multiplier: applied to any component not in component_multipliers.
      Infiltration components typically use 1.0 so they remain active
      independently of the mechanical ventilation schedule.
    """
    streams = []
    seen_names: set = set()
    cm = component_multipliers or {}
    for comp in components:
        name = str(comp.get("name", "")).strip()
        if not name:
            raise ValueError("Each ventilation component must have a non-empty 'name'")
        if name in seen_names:
            raise ValueError(f"Duplicate ventilation component name: {name!r}")
        seen_names.add(name)

        comp_type = str(comp.get("ventilation_type", "")).strip().lower()
        component_fraction = float(cm.get(name, default_multiplier))

        if comp_type == "constant_ach":
            if zone_volume_m3 is None or float(zone_volume_m3) <= 0.0:
                raise ValueError(
                    f"Component {name!r}: ventilation_type='constant_ach' "
                    "requires a positive zone_volume_m3"
                )
            ach = float(comp["air_changes_per_hour"])
            q_m3_s = ach * float(zone_volume_m3) / 3600.0
            h_k = rho_air * c_air * q_m3_s * component_fraction
            source_temp = _parse_source_temperature(
                comp.get("source_temperature", "outdoor"), outdoor_temperature_c
            )
            category = "outdoor_air"

        elif comp_type == "prescribed":
            h_k = float(comp["heat_transfer_coefficient_w_k"]) * component_fraction
            source_temp = float(comp["source_temperature_c"])
            category = comp.get("category", "supply")

        elif comp_type == "mechanical_supply":
            # Validate flow fraction here (before AHUStepInputs) so the error
            # names the component rather than producing a generic dataclass message.
            if not math.isfinite(component_fraction) or not (0.0 <= component_fraction <= 1.0):
                raise ValueError(
                    f"Component {name!r}: mechanical-supply flow fraction must be finite "
                    f"and in [0, 1], got {component_fraction!r}"
                )
            # Imported lazily: the boundary module stays usable without the
            # AHU module unless a mechanical_supply component is configured.
            from .ventilation_16798_5_1 import (  # noqa: PLC0415
                AHUStepInputs,
                ahu_outputs_to_ventilation_stream,
                calculate_sensible_ahu_step,
                sensible_ahu_config_from_dict,
            )
            supply_m3_h = float(comp["supply_flow_m3_h"])
            extract_m3_h = float(comp.get("extract_flow_m3_h", supply_m3_h))
            ahu_cfg = sensible_ahu_config_from_dict(comp)
            # extract_air_temperature_c: previous-timestep zone AIR node temperature,
            # not operative. Explicit lag avoids the T_zone→T_extract→T_supply→T_zone loop.
            extract_air_temperature_c = zone_temperature_c
            ahu_inp = AHUStepInputs(
                outdoor_temperature_c=outdoor_temperature_c,
                extract_temperature_c=extract_air_temperature_c,
                required_supply_flow_m3_h=supply_m3_h,
                required_extract_flow_m3_h=extract_m3_h,
                operation_fraction=1.0,
                # component_fraction is a continuous flow fraction, not timestep duty-cycle.
                flow_fraction=component_fraction,
            )
            ahu_out = calculate_sensible_ahu_step(ahu_cfg, ahu_inp)
            if ahu_outputs_collector is not None:
                ahu_outputs_collector[name] = ahu_out
            _s = ahu_outputs_to_ventilation_stream(ahu_out, name=name)
            h_k = _s.heat_transfer_coefficient_w_k
            source_temp = _s.source_temperature_c
            category = "supply"

        else:
            raise ValueError(
                f"Component {name!r}: unknown ventilation_type={comp_type!r}. "
                "Supported component types: 'constant_ach', 'mechanical_supply', 'prescribed'."
            )

        streams.append(
            VentilationStream(
                name=name,
                heat_transfer_coefficient_w_k=h_k,
                source_temperature_c=source_temp,
                category=category,
            )
        )
    return streams


def resolve_ventilation_boundary(
    building_object,
    zone_temperature_c: float,
    outdoor_temperature_c: float,
    wind_speed_m_s: float,
    profile_multiplier: float = 1.0,
    component_multipliers: dict = None,
    zone_volume_m3=None,
    altitude_m=None,
    extra_streams=(),
    c_air: float = 1006.0,
    rho_air: float = 1.204,
    ahu_outputs_collector: dict = None,
) -> VentilationBoundary:
    """Resolve building/zone configuration into an ISO 52016-1 VentilationBoundary.

    When the ventilation configuration contains a 'components' list, each entry
    becomes an independent VentilationStream with its own source temperature.

    When no 'components' key is present, the legacy scalar ventilation_type path
    is used, producing a single outdoor-air stream scaled by profile_multiplier.
    All five existing ventilation_type calculations (temp_wind, occupancy, custom,
    eplus_infiltration_ext_area, sherman_grimsrud_like) are supported via this path.

    profile_multiplier: operation fraction applied to the legacy single stream.
    component_multipliers: optional dict mapping component name -> float.
      For 'mechanical_supply' components: a continuous flow fraction (not duty-cycle);
      values outside [0, 1] raise ValueError.
      For other component types: scales H_k linearly and is not restricted to
      [0, 1]. The resulting H_k must still be finite and non-negative.
      Components not in the dict default to 1.0 (full capacity), so infiltration
      components remain active independently of the mechanical ventilation schedule.
      Pass {"ahu_supply": 0.0} to turn off the AHU while infiltration runs
      unaffected. When None, all components run at 1.0.
    extra_streams: additional pre-built VentilationStream objects added by the caller
    (e.g. a summer night purge stream resolved by the timestep loop).

    Raises RuntimeError or ValueError on invalid configuration; does not silently
    substitute zero for invalid H_k or non-finite temperatures.
    """
    vent_cfg: dict = {}
    if isinstance(building_object, dict):
        vent_cfg = (
            building_object.get("building_parameters", {}).get("ventilation", {})
            or building_object.get("ventilation", {})
            or {}
        )

    components = vent_cfg.get("components", None)

    if components is not None:
        # For component streams, each stream uses its own schedule (1.0 default)
        # so infiltration components remain active when a mechanical schedule is
        # off.  profile_multiplier is only applied to the legacy single stream.
        streams = _resolve_component_streams(
            components,
            outdoor_temperature_c,
            zone_volume_m3,
            c_air,
            rho_air,
            component_multipliers=component_multipliers,
            default_multiplier=1.0,
            zone_temperature_c=zone_temperature_c,
            ahu_outputs_collector=ahu_outputs_collector,
        )
    else:
        vent_type = str(vent_cfg.get("ventilation_type", "none")).strip().lower()
        if vent_type in ("none", "off", "disabled", ""):
            streams = []
        else:
            try:
                h_ve = VentilationInternalGains.heat_transfer_coefficient_by_ventilation(
                    building_object,
                    zone_temperature_c,
                    outdoor_temperature_c,
                    wind_speed_m_s,
                    type_ventilation=vent_type,
                    flowrate_person=float(vent_cfg.get("flow_rate_per_person", 0.0)),
                    custom_Hve_k_t=float(
                        vent_cfg.get("custom_heat_transfer_coefficient_ventilation", 0.0)
                    ),
                    altitude=altitude_m,
                )
                h_ve = float(np.ravel(np.asarray(h_ve))[0])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to resolve ventilation boundary for "
                    f"ventilation_type={vent_type!r}: {exc}"
                ) from exc
            if not math.isfinite(h_ve) or h_ve < 0.0:
                raise ValueError(
                    f"Ventilation resolved to invalid H_ve={h_ve!r} "
                    f"for ventilation_type={vent_type!r}"
                )
            h_ve_scaled = h_ve * profile_multiplier
            streams = (
                [VentilationStream(
                    name="ventilation",
                    heat_transfer_coefficient_w_k=h_ve_scaled,
                    source_temperature_c=outdoor_temperature_c,
                    category="outdoor_air",
                )]
                if h_ve_scaled > 0.0
                else []
            )

    return VentilationBoundary(streams=tuple(streams) + tuple(extra_streams))


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
