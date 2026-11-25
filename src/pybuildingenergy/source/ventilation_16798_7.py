"""EN 16798-7:2017 (module M5-5) - air flow rates in buildings including infiltration.

Method 1 (clause 6): air flow rates of one ventilation zone from the building
characteristics.  Mechanical, passive duct, combustion, window (airing), vent and envelope
leakage flows are combined in a mass balance on the internal reference pressure
``p_z;ref`` (67).  Air entering the zone is positive, air leaving it negative.  Flows in
m3/h, pressures in Pa, temperatures in degC (K where the standard uses K), heat in kWh.

Method 2 (clause 7) is a national statistical approach with no default data (Table A.14,
B.4): only the totals ``q_V;tot;in`` / ``q_V;tot;out`` are exchanged, see :func:`method2_totals`.

Main entry points:

* :func:`mechanical_flows` - required flows at the air terminal devices (9)-(24);
* :func:`required_supply_temperature` - LOAD_COMP supply temperature (7)-(8);
* :class:`ZoneAirflow` - the zone, with :meth:`ZoneAirflow.solve` for (1)-(6), (25)-(69);
* :func:`ahu_inputs` - the outputs in the form needed by the EN 16798-5-1 AHU model.

Notes on the printed text:

* (47)/(48): the scanned formula is cut at the right edge; the stack term is taken with the
  exponent 0.5 as in (45)/(46).
* Table 7 gives the airtightness index ``q_V;dPlea;ref`` in m3/h; (A.1)/(B.4) multiply it by
  the reference area ``A_lea``, so it is used per m2 of ``A_lea``.  For a total leakage in
  m3/h give ``a_lea_m2 = 1``.
* Table B.7 has no values for High facades with Normal/Shielded exposure: the Medium values
  are used.
* (30): with a wind term two roots can exist (hysteresis of the sign); the one with the
  smaller flow is taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# --------------------------------------------------------------------------- Table 11
C_A_KWH_KG_K = 0.000279
C_ST = 0.0035
C_WND = 0.001
G = 9.81
P_ATM_PA = 101_325.0
T0_ABS = 273.15
T_E_REF_K = 293.15
RHO_A_REF = 1.204
DP_LEA_REF_PA = 50.0

# --------------------------------------------------------------------------- Annex B tables
VENT_SYS_OP = ("BALANCED_OP", "EXTRACT_OP", "SUPPLY_OP", "NATURAL_OP")             # Table B.14
TABLE_B2_F_AS = {"room_air_flue_into_room": 0.0, "room_air_separate_duct": 1.0,
                 "sealed": 0.0}                                                    # f_as
TABLE_B3_F_FF = {"wood_open_fire": 2.8, "gas_closed_fan": 0.38, "gas_open_flue_balancer": 0.78,
                 "gas_open_kitchen_stove": 3.35, "gas_open_effect_fire": 3.35,
                 "oil_closed": 0.32, "coal_closed": 0.52}                          # l/(s kW)
TABLE_B4_F_CTRL = {("people_count_co2", "non_residential"): 0.8,
                   ("occupancy", "non_residential"): 0.9,
                   ("other", "non_residential"): 1.0, ("any", "residential"): 1.0}
TABLE_B5_F_SYS = {"certified_ctrl_lt_1": 1.0, "certified_ctrl_1": 1.1, "non_certified": 1.2}
# Table B.7 (cross-ventilated zone): (windward, leeward, roof <10, roof 10-30, roof >30)
TABLE_B7_CP = {
    ("low", "open"): (0.50, -0.70, -0.70, -0.60, -0.20),
    ("low", "normal"): (0.25, -0.50, -0.60, -0.50, -0.20),
    ("low", "shielded"): (0.05, -0.30, -0.50, -0.40, -0.20),
    ("medium", "open"): (0.65, -0.70, -0.70, -0.60, -0.20),
    ("medium", "normal"): (0.45, -0.50, -0.60, -0.50, -0.20),
    ("medium", "shielded"): (0.25, -0.30, -0.50, -0.40, -0.20),
    ("high", "open"): (0.80, -0.70, -0.70, -0.60, -0.20),
}
TABLE_B8_CP = (0.05, -0.05, 0.0)                    # windward, leeward, roof (f_cros = 0)
TABLE_B9_DC_COWL = ((0.5, 0.0), (1.0, -0.1), (math.inf, -0.2))   # roof outlet height [m]
TABLE_B10_F_CROS = {"single_family": 1, "apartment_cross": 1, "apartment_other": 0, "office": 0,
                    "education": 1, "hospital": 0, "hotel_restaurant": 0, "sport": 1,
                    "retail": 1, "other": 0}
TABLE_B13_C_RGH_10 = {"open": 1.0, "country": 0.9, "urban": 0.8}
B_DEFAULTS = {"C_D_vent": 0.6, "C_D_ATD": 0.6, "C_D_w": 0.67, "n_vent": 0.5, "n_ATD": 0.5,
              "n_w": 0.5, "n_lea": 0.667, "f_arg": 1.8, "eps_v": 1.0, "n_w_div": 1,
              "dCp": 0.75, "Cp_cowl_roof": 0.0, "u10_site_max": 3.0, "A_door_min": 1.4}


def f_ctrl(sensor: str, building: str = "non_residential") -> float:
    """Table B.4."""
    if building == "residential":
        return 1.0
    return TABLE_B4_F_CTRL.get((sensor, "non_residential"), 1.0)


def f_sys(system: str) -> float:
    """Table B.5."""
    return TABLE_B5_F_SYS[system]


def cp_table(height_m: float, shield: str, kind: str, cross: bool = True,
             roof_slope_deg: float = 0.0) -> float:
    """Wind pressure coefficient (Tables B.7/B.8).  ``kind``: windward, leeward, roof."""
    if kind not in ("windward", "leeward", "roof"):
        raise ValueError("kind must be 'windward', 'leeward' or 'roof'")
    if not cross:
        return TABLE_B8_CP[("windward", "leeward", "roof").index(kind)]
    cls = "low" if height_m < 15.0 else ("medium" if height_m < 50.0 else "high")
    key = (cls, str(shield).lower())
    if key not in TABLE_B7_CP:                      # High + Normal/Shielded: not tabulated
        key = ("medium", str(shield).lower())
    row = TABLE_B7_CP[key]
    if kind == "windward":
        return row[0]
    if kind == "leeward":
        return row[1]
    return row[2] if roof_slope_deg < 10.0 else (row[3] if roof_slope_deg <= 30.0 else row[4])


def d_c_cowl_height(height_above_roof_m: float) -> float:
    """Table B.9."""
    for lim, v in TABLE_B9_DC_COWL:
        if height_above_roof_m < lim or (height_above_roof_m == lim and lim == 1.0):
            return v
    return -0.2


# --------------------------------------------------------------------------- 6.4.2
def air_density_ref(altitude_m: float = 0.0) -> float:
    """Table 11, note a: rho_a;ref adjusted to the altitude."""
    return RHO_A_REF * (1.0 - 0.00651 * float(altitude_m) / 293.0) ** 4.255


def air_density(t_c: float, rho_ref: float = RHO_A_REF) -> float:
    """(1): rho_T = T_ref / T * rho_a;ref."""
    return T_E_REF_K / (t_c + T0_ABS) * rho_ref


def wind_speed_site(u10: float, c_rgh_site=1.0, c_top_site=1.0, c_rgh_met=1.0, c_top_met=1.0) -> float:
    """(2): wind speed at zone level."""
    return c_rgh_site * c_top_site / (c_rgh_met * c_top_met) * u10


def wind_speed_10_site(u10: float, c_rgh_10_site=1.0, c_top_10_site=1.0, c_rgh_met=1.0,
                       c_top_met=1.0) -> float:
    """(3): wind speed at 10 m at the site."""
    return c_rgh_10_site * c_top_10_site / (c_rgh_met * c_top_met) * u10


def path_pressure_difference(cp: float, h_path: float, u_site: float, t_e_c: float, t_z_c: float,
                             p_z_ref: float, rho_ref: float = RHO_A_REF) -> float:
    """(4)-(6): pressure difference outside - inside of an air flow path at height h_path."""
    t_e, t_z = t_e_c + T0_ABS, t_z_c + T0_ABS
    p_e = rho_ref * T_E_REF_K / t_e * (0.5 * cp * u_site ** 2 - h_path * G)             # (5)
    p_z = p_z_ref - rho_ref * h_path * G * T_E_REF_K / t_z                               # (6)
    return p_e - p_z                                                                     # (4)


def required_supply_temperature(theta_zt_c: float, q_h_req_kwh: float, q_c_req_kwh: float,
                                q_sup_dis_m3h: float, t_ci_h: float, d_theta_ctr_syst_k: float = 0.0,
                                theta_e_comb_c: Optional[float] = None,
                                rho_a: float = RHO_A_REF) -> float:
    """(7)-(8): supply air temperature for SUP_AIR_TEMP_CTRL = LOAD_COMP.

    Either the heating or the cooling energy is 0 (NOTE 2); the thermal and ventilation zone
    must coincide (NOTE 3)."""
    if q_h_req_kwh > 0.0 and q_c_req_kwh > 0.0:
        raise ValueError("(7) assumes either the heating or the cooling requirement is 0")
    for nm, val in (("q_sup_dis_m3h", q_sup_dis_m3h), ("t_ci_h", t_ci_h), ("rho_a", rho_a)):
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"{nm} must be finite and > 0 (got {val})")
    for nm, val in (("d_theta_ctr_syst_k", d_theta_ctr_syst_k), ("theta_e_comb_c", theta_e_comb_c)):
        if val is not None and not math.isfinite(val):
            raise ValueError(f"{nm} must be finite (got {val})")

    if not (math.isfinite(q_h_req_kwh) and math.isfinite(q_c_req_kwh) and q_h_req_kwh >= 0.0
            and q_c_req_kwh >= 0.0 and math.isfinite(theta_zt_c)):
        raise ValueError("q_h_req_kwh and q_c_req_kwh must be finite and >= 0; theta_zt_c finite")

    def ls(q):
        if q <= 0.0 or not d_theta_ctr_syst_k:
            return 0.0
        if theta_e_comb_c is None:
            raise ValueError("theta_e_comb_c (corrected outdoor temperature, (8)) is required when "
                             "d_theta_ctr_syst_k is not 0 and there is a heating/cooling need")
        den = theta_zt_c - theta_e_comb_c
        if abs(den) <= 1e-9:
            raise ValueError("(8) is undefined: theta_zt_c equals theta_e_comb_c")
        return q * d_theta_ctr_syst_k / den                                              # (8)

    k = rho_a * C_A_KWH_KG_K * q_sup_dis_m3h * t_ci_h
    return (theta_zt_c + (q_h_req_kwh + ls(q_h_req_kwh)) / k
            - (q_c_req_kwh + ls(q_c_req_kwh)) / k)                                       # (7)


# --------------------------------------------------------------------------- 6.4.3.2
def required_outdoor_air(q_oda_des_m3h: float, f_ctrl_: float = 1.0, f_sys_: float = 1.0,
                         eps_v: float = 1.0) -> float:
    """(9): q_V;ODA;req = f_ctrl f_sys / eps_V q_V;ODA;req;des."""
    if eps_v <= 0.0:
        raise ValueError("eps_v must be > 0")
    for name, v in (("q_oda_des_m3h", q_oda_des_m3h), ("f_ctrl_", f_ctrl_), ("f_sys_", f_sys_),
                    ("eps_v", eps_v)):
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"{name} must be finite and >= 0 (got {v})")
    return f_ctrl_ * f_sys_ / eps_v * q_oda_des_m3h


def terminal_required_flows(vent_sys_op: str, q_oda_req_m3h: float) -> tuple:
    """(10)-(17): (q_V;SUP;req, q_V;ETA;req) at the air terminal devices; extract negative."""
    op = str(vent_sys_op).upper()
    if op not in VENT_SYS_OP:
        raise ValueError(f"vent_sys_op must be one of {VENT_SYS_OP}")
    if op == "BALANCED_OP":
        return q_oda_req_m3h, -q_oda_req_m3h
    if op == "EXTRACT_OP":
        return 0.0, -q_oda_req_m3h
    if op == "SUPPLY_OP":
        return q_oda_req_m3h, 0.0
    return 0.0, 0.0


def mechanical_flows(vent_sys_op: str, q_oda_req_m3h: float, f_op_v: float = 1.0,
                     flow_control: str = "ODA", q_h_req_kwh: float = 0.0, q_c_req_kwh: float = 0.0,
                     q_h_em_ls_kwh: float = 0.0, q_c_em_ls_kwh: float = 0.0,
                     theta_sup_out_c: Optional[float] = None, t_z_c: Optional[float] = None,
                     t_ci_h: float = 1.0, rho_a: float = RHO_A_REF,
                     q_sup_max_m3h: Optional[float] = None) -> dict:
    """(18)-(22): mechanical flows through the air terminal devices.

    ``flow_control`` "ODA": ``f_op;V`` times the outdoor air requirement.  "LOAD": the flow
    that covers the heating or cooling need at the supply temperature ``theta_sup_out_c``,
    at least the outdoor air flow; VENT_SYS_OP must be BALANCED_OP or SUPPLY_OP."""
    if not math.isfinite(q_oda_req_m3h) or q_oda_req_m3h < 0.0:
        raise ValueError(f"q_oda_req_m3h must be finite and >= 0 (got {q_oda_req_m3h})")
    if not math.isfinite(f_op_v) or not 0.0 <= f_op_v <= 1.0:
        raise ValueError(f"f_op_v is an operation signal and must be in [0, 1] (got {f_op_v})")
    if q_sup_max_m3h is not None and (not math.isfinite(q_sup_max_m3h) or q_sup_max_m3h < 0.0):
        raise ValueError("q_sup_max_m3h must be finite and >= 0")
    for nm, val in (("q_h_req_kwh", q_h_req_kwh), ("q_c_req_kwh", q_c_req_kwh),
                    ("q_h_em_ls_kwh", q_h_em_ls_kwh), ("q_c_em_ls_kwh", q_c_em_ls_kwh)):
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"{nm} must be finite and >= 0 (got {val})")
    q_sup_req, q_eta_req = terminal_required_flows(vent_sys_op, q_oda_req_m3h)
    op = str(vent_sys_op).upper()
    if str(flow_control).upper() == "ODA":
        return {"q_sup_dis_req": f_op_v * q_sup_req, "q_eta_dis_req": f_op_v * q_eta_req}   # (18), (19)
    if op not in ("BALANCED_OP", "SUPPLY_OP"):
        raise ValueError("SUP_AIR_FLW_CTRL = LOAD needs BALANCED_OP or SUPPLY_OP")
    if theta_sup_out_c is None or t_z_c is None:
        raise ValueError("LOAD flow control needs theta_sup_out_c and t_z_c")
    for nm, val in (("theta_sup_out_c", theta_sup_out_c), ("t_z_c", t_z_c)):
        if not math.isfinite(val):
            raise ValueError(f"{nm} must be finite (got {val})")
    if not (math.isfinite(t_ci_h) and t_ci_h > 0.0 and math.isfinite(rho_a) and rho_a > 0.0):
        raise ValueError("t_ci_h and rho_a must be finite and > 0")
    k = rho_a * C_A_KWH_KG_K * t_ci_h
    terms = [q_sup_req]
    for q_req, ls, dt in ((q_h_req_kwh, q_h_em_ls_kwh, theta_sup_out_c - t_z_c),
                          (q_c_req_kwh, q_c_em_ls_kwh, t_z_c - theta_sup_out_c)):
        if q_req + ls > 0.0:
            if dt * k <= 1e-12:
                if q_sup_max_m3h is None:
                    raise ValueError("the supply temperature cannot cover the load; give q_sup_max_m3h")
                terms.append(q_sup_max_m3h)
            else:
                terms.append((q_req + ls) / (k * dt))                                         # (20)
    q_sup = f_op_v * max(terms)
    if q_sup_max_m3h is not None:
        q_sup = min(q_sup, q_sup_max_m3h)
    return {"q_sup_dis_req": q_sup, "q_eta_dis_req": -q_sup if op == "BALANCED_OP" else 0.0}  # (21), (22)


def design_supply_flow(q_h_des_kwh: float, q_c_des_kwh: float, theta_sup_out_c: float, t_z_c: float,
                       q_oda_req_des_m3h: float, t_ci_h: float = 1.0, rho_a: float = RHO_A_REF) -> float:
    """(23): maximum design supply flow to the zone."""
    if not (q_oda_req_des_m3h >= 0.0 and q_h_des_kwh >= 0.0 and q_c_des_kwh >= 0.0) \
            or not math.isfinite(q_oda_req_des_m3h + q_h_des_kwh + q_c_des_kwh):
        raise ValueError("design flow and design energies must be >= 0")
    if not (math.isfinite(t_ci_h) and t_ci_h > 0.0 and math.isfinite(rho_a) and rho_a > 0.0):
        raise ValueError("t_ci_h and rho_a must be finite and > 0")
    if not (math.isfinite(theta_sup_out_c) and math.isfinite(t_z_c)):
        raise ValueError("theta_sup_out_c and t_z_c must be finite")
    k = rho_a * C_A_KWH_KG_K * t_ci_h
    terms = [q_oda_req_des_m3h]
    if q_h_des_kwh > 0.0 and theta_sup_out_c > t_z_c:
        terms.append(q_h_des_kwh / (k * (theta_sup_out_c - t_z_c)))
    if q_c_des_kwh > 0.0 and t_z_c > theta_sup_out_c:
        terms.append(q_c_des_kwh / (k * (t_z_c - theta_sup_out_c)))
    return max(terms)


def supply_outdoor_air_flow(f_oda: float, q_sup_dis_m3h: float) -> float:
    """(24): q_V;ODA;dis = f_ODA q_V;SUP;dis."""
    if not 0.0 <= f_oda <= 1.0 or q_sup_dis_m3h < 0.0:
        raise ValueError("f_oda must be in [0, 1] and q_sup_dis_m3h >= 0")
    return f_oda * q_sup_dis_m3h


# --------------------------------------------------------------------------- 6.4.3.3
def atd_coefficient(a_atd_cm2: float, c_d: float = 0.6, dp_ref_pa: float = 10.0, n: float = 0.5,
                    rho_ref: float = RHO_A_REF) -> float:
    """(26): C_ATD from the equivalent area."""
    if a_atd_cm2 < 0.0 or c_d <= 0.0 or dp_ref_pa <= 0.0 or n <= 0.0:
        raise ValueError("a_atd_cm2 must be >= 0; c_d, dp_ref_pa and n must be > 0")
    return 3600.0 / 10000.0 * c_d * a_atd_cm2 * (2.0 / rho_ref) ** 0.5 * (1.0 / dp_ref_pa) ** (n - 0.5)


def vent_coefficient(a_vent_cm2: float, c_d: float = 0.6, dp_ref_pa: float = 10.0, n: float = 0.5,
                     rho_ref: float = RHO_A_REF) -> float:
    """(59): C_vent;i from the equivalent area."""
    return atd_coefficient(a_vent_cm2, c_d, dp_ref_pa, n, rho_ref)


def combustion_air_flow(f_op_comb: float, f_as: float, f_ff: float, p_h_fi_kw: float) -> float:
    """(35): q_V;comb = 3.6 f_op;comb f_as f_ff P_h;fi (f_ff in l/(s kW))."""
    if min(f_op_comb, f_as, f_ff, p_h_fi_kw) < 0.0:
        raise ValueError("f_op_comb, f_as, f_ff and p_h_fi_kw must be >= 0")
    return 3.6 * f_op_comb * f_as * f_ff * p_h_fi_kw


def sgn(x: float) -> float:
    return (x > 0.0) - (x < 0.0)


@dataclass(frozen=True)
class PassiveDuct:
    """Passive or hybrid duct (6.4.3.3)."""
    c_atd: float                                  # C_ATD [m3/(h Pa^n)]
    n_atd: float = B_DEFAULTS["n_ATD"]
    height_m: float = 3.0                         # h_pdu
    zone_height_m: float = 3.0                    # h_z (B.2)
    levels_exchange_air: bool = False             # air transfer between levels: B.2 instead of B.1
    cross_section_m2: float = 0.01                # A_s;pdu
    zeta_cowl: float = 0.0                        # pressure loss coefficient without wind
    c_cowl: float = 0.0                           # pressure factor of the cowl
    cp_cowl_roof: float = B_DEFAULTS["Cp_cowl_roof"]
    d_c_cowl_height: float = 0.0
    duct_dp_pa: float = 0.0                       # Delta p_pdu at ``duct_q_ref_m3h``
    duct_q_ref_m3h: float = 100.0

    def __post_init__(self) -> None:
        if self.c_atd <= 0.0 or self.n_atd <= 0.0 or self.duct_q_ref_m3h <= 0.0:
            raise ValueError("PassiveDuct: c_atd, n_atd and duct_q_ref_m3h must be > 0")
        if self.height_m < 0.0 or self.zone_height_m < 0.0 or self.cross_section_m2 <= 0.0 \
                or self.duct_dp_pa < 0.0:
            raise ValueError("PassiveDuct: heights and duct_dp_pa must be >= 0, cross_section_m2 > 0")

    @property
    def h_stack(self) -> float:
        """(B.1)/(B.2): useful height for the stack effect."""
        return (self.zone_height_m if self.levels_exchange_air and self.zone_height_m > 3.0
                else self.height_m) + 2.0


def _pdu_residual(pdu: PassiveDuct, q: float, rhs: float, u_site: float, rho_ref: float) -> float:
    """(30) left side minus right side for a flow ``q``."""
    s = sgn(q)
    d_atd = -s * (abs(q) / pdu.c_atd) ** (1.0 / pdu.n_atd)                               # (25)
    d_pdu = -s * pdu.duct_dp_pa * (abs(q) / pdu.duct_q_ref_m3h) ** 2
    if u_site == 0.0:
        d_cowl = -0.5 * s * pdu.zeta_cowl * rho_ref * (abs(q) / (3600.0 * pdu.cross_section_m2)) ** 2  # (28)
    else:
        c_tot = pdu.c_cowl + pdu.cp_cowl_roof + pdu.d_c_cowl_height                      # (27)
        d_cowl = -0.5 * s * c_tot * rho_ref * u_site ** 2                                # (29)
    return d_atd + d_pdu + d_cowl - rhs


def passive_duct_flow(pdu: PassiveDuct, p_z_ref: float, u_site: float, t_e_c: float, t_z_c: float,
                      rho_ref: float = RHO_A_REF) -> float:
    """(30): flow through the passive duct for a given reference pressure (positive in)."""
    rho_e, rho_z = air_density(t_e_c, rho_ref), air_density(t_z_c, rho_ref)
    rhs = p_z_ref + pdu.h_stack * G * (rho_e - rho_z)

    def root(sign):
        f0 = _pdu_residual(pdu, sign * 1e-9, rhs, u_site, rho_ref)
        if sign > 0 and f0 <= 0.0:
            return None
        if sign < 0 and f0 >= 0.0:
            return None
        lo, hi = 0.0, 1.0
        while sign * _pdu_residual(pdu, sign * hi, rhs, u_site, rho_ref) > 0.0 and hi < 1e7:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if sign * _pdu_residual(pdu, sign * mid, rhs, u_site, rho_ref) > 0.0:
                lo = mid
            else:
                hi = mid
        return sign * 0.5 * (lo + hi)

    roots = [r for r in (root(+1), root(-1)) if r is not None]
    return min(roots, key=abs) if roots else 0.0


# --------------------------------------------------------------------------- zone
@dataclass(frozen=True)
class WindowData:
    """A window for the airing calculation (Table 7)."""
    max_area_m2: float                            # A_w;max,i
    height_m: float = 1.5                         # h_w;path,i
    free_area_height_m: float = 1.0               # h_w;fa,i
    orientation_deg: float = 0.0                  # 0 S, 90 W, 180 N, 270 E
    tilt_deg: float = 90.0                        # 0 horizontal, 90 vertical
    opening_ratio: float = 1.0                    # R_w;arg,i
    facade: str = "windward"                      # for the pressure method (53)
    cp: Optional[float] = None


@dataclass(frozen=True)
class ZoneAirflow:
    """A ventilation zone (6.4.3.1) with its air flow paths."""
    height_m: float = 3.0                         # h_z
    facade_area_m2: float = 100.0                 # A_fde
    roof_area_m2: float = 0.0                     # A_roof
    q50_per_m2: float = 0.0                       # airtightness index [m3/(h m2)] at 50 Pa
    a_lea_m2: float = 1.0                         # reference area of the index
    n_lea: float = B_DEFAULTS["n_lea"]
    c_vent_total: float = 0.0                     # C_vent (B.3) [m3/(h Pa^n)]
    n_vent: float = B_DEFAULTS["n_vent"]
    shield: str = "normal"
    cross_ventilation: bool = False               # f_cros
    roof_slope_deg: float = 0.0
    windows: Sequence[WindowData] = field(default_factory=tuple)
    window_method: str = "wind_temperature"       # simplified, wind_temperature, pressure
    c_d_w: float = B_DEFAULTS["C_D_w"]
    n_w: float = B_DEFAULTS["n_w"]
    n_w_div: int = B_DEFAULTS["n_w_div"]
    f_arg: float = B_DEFAULTS["f_arg"]
    d_cp: float = B_DEFAULTS["dCp"]
    u10_site_max: float = B_DEFAULTS["u10_site_max"]
    passive_duct: Optional[PassiveDuct] = None
    altitude_m: float = 0.0
    c_rgh_site: float = 1.0
    c_top_site: float = 1.0
    c_rgh_10_site: float = 1.0
    c_top_10_site: float = 1.0
    c_rgh_met: float = 1.0
    c_top_met: float = 1.0

    def __post_init__(self) -> None:
        if str(self.window_method).lower() not in ("simplified", "wind_temperature", "pressure"):
            raise ValueError("window_method must be 'simplified', 'wind_temperature' or 'pressure'")
        if str(self.shield).lower() not in ("open", "normal", "shielded"):
            raise ValueError("shield must be 'open', 'normal' or 'shielded'")
        if self.height_m <= 0.0:
            raise ValueError("height_m must be > 0")
        for name in ("facade_area_m2", "roof_area_m2", "q50_per_m2", "a_lea_m2", "c_vent_total",
                     "u10_site_max", "d_cp"):
            if not getattr(self, name) >= 0.0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be >= 0 (got {getattr(self, name)})")
        for name in ("n_lea", "n_vent", "n_w", "c_d_w", "f_arg", "c_rgh_site", "c_top_site",
                     "c_rgh_10_site", "c_top_10_site", "c_rgh_met", "c_top_met"):
            if not getattr(self, name) > 0.0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be > 0 (got {getattr(self, name)})")
        if int(self.n_w_div) < 1:
            raise ValueError("n_w_div must be >= 1")
        for w in self.windows:
            if w.max_area_m2 < 0.0 or not 0.0 <= w.opening_ratio <= 1.0:
                raise ValueError("window max_area_m2 must be >= 0 and opening_ratio in [0, 1]")
        if self.window_method == "pressure" and self.cross_ventilation is False and not self.windows:
            pass

    # ------------------------------------------------------------------ paths
    @property
    def c_lea(self) -> float:
        """(B.4): leakage coefficient of the zone."""
        return self.q50_per_m2 * self.a_lea_m2 / DP_LEA_REF_PA ** self.n_lea

    def _facade_paths(self):
        """Table B.11/B.12 paths: (kind, height, share) on the windward/leeward facades."""
        hz = self.height_m
        return [(k, f * hz) for k in ("windward", "leeward") for f in (0.25, 0.75)]

    def _cp(self, kind, h):
        return cp_table(h, self.shield, kind, self.cross_ventilation, self.roof_slope_deg)

    def _leak_vent_flows(self, p_z, u_site, t_e, t_z, rho_ref):
        """(58), (62): vents and leaks through the Table B.11/B.12 paths."""
        out = {"vent_in": 0.0, "vent_out": 0.0, "lea_in": 0.0, "lea_out": 0.0}
        tot_area = self.facade_area_m2 + self.roof_area_m2
        c_fde = self.c_lea * self.facade_area_m2 / tot_area if tot_area > 0.0 else 0.0     # (B.5)
        c_roof = self.c_lea * self.roof_area_m2 / tot_area if tot_area > 0.0 else 0.0      # (B.6)
        paths = []
        for kind, h in self._facade_paths():
            paths.append(("lea", c_fde * 0.25, self.n_lea, self._cp(kind, h), h))
            paths.append(("vent", self.c_vent_total * 0.25, self.n_vent, self._cp(kind, h), h))
        if c_roof > 0.0:
            paths.append(("lea", c_roof, self.n_lea,
                          self._cp("roof", self.height_m), self.height_m))
        for what, c, n, cp, h in paths:
            if c <= 0.0:
                continue
            dp = path_pressure_difference(cp, h, u_site, t_e, t_z, p_z, rho_ref)
            q = c * sgn(dp) * abs(dp) ** n                                                # (58)/(62)
            out[f"{what}_in" if q > 0 else f"{what}_out"] += q
        return out

    def _window_pressure_flows(self, p_z, u_site, t_e, t_z, rho_ref):
        """(53)-(57)."""
        q_in = q_out = 0.0
        nd = int(self.n_w_div)
        for w in self.windows:
            a = w.opening_ratio * w.max_area_m2                                           # (40)
            if a <= 0.0:
                continue
            c = 3600.0 * self.c_d_w * a * (2.0 / rho_ref) ** self.n_w                     # (54)
            cp = w.cp if w.cp is not None else self._cp(w.facade, w.height_m)
            for j in range(1, nd + 2):
                h = (w.height_m - w.free_area_height_m / 2.0
                     + w.free_area_height_m / (2.0 * (nd + 1))
                     + w.free_area_height_m / (nd + 1) * (j - 1))                         # (55)
                dp = path_pressure_difference(cp, h, u_site, t_e, t_z, p_z, rho_ref)
                q = c / (nd + 1) * sgn(dp) * abs(dp) ** self.n_w                          # (53)
                if q > 0:
                    q_in += q
                else:
                    q_out += q
        return q_in, q_out

    # ------------------------------------------------------------------ windows (fixed)
    def window_stack_height(self) -> float:
        """(44): useful height for the stack effect for airing."""
        if not self.windows:
            return 0.0
        return (max(w.height_m + w.free_area_height_m / 2.0 for w in self.windows)
                - min(w.height_m - w.free_area_height_m / 2.0 for w in self.windows))

    def cross_window_area(self) -> float:
        """(49)-(52): equivalent window area for cross ventilation."""
        a_w = [w.opening_ratio * w.max_area_m2 for w in self.windows]
        a_tot = sum(a_w)
        res = []
        for i in (1, 2):
            terms = []
            for j in (1, 2, 3, 4):
                a_ori = 0.0
                a_ref = (i - 1) * 45.0 + (j - 1) * 90.0                                    # (49)
                for w, a in zip(self.windows, a_w):
                    rel = (w.orientation_deg - a_ref + 180.0) % 360.0 - 180.0
                    if -45.0 <= rel < 45.0 and w.tilt_deg >= 60.0:                         # (50)
                        a_ori += a
                if a_ori > 0.0:
                    rest = a_tot - a_ori
                    terms.append(0.0 if rest <= 0.0 else
                                 1.0 / math.sqrt(1.0 / a_ori ** 2 + 1.0 / rest ** 2))
            res.append(0.25 * sum(terms))                                                  # (51)
        return min(res)                                                                    # (52)

    def _window_fixed_flows(self, q_oda_req, u10_site, t_e, t_z, rho_e, rho_z, rho_ref):
        """(42)-(48): airing flows that do not depend on the reference pressure."""
        m = self.window_method
        if m == "simplified":
            return (rho_ref / rho_e * self.f_arg * q_oda_req,
                    -rho_ref / rho_z * self.f_arg * q_oda_req)                             # (42), (43)
        a_tot = sum(w.opening_ratio * w.max_area_m2 for w in self.windows)
        if a_tot <= 0.0:
            return 0.0, 0.0
        h_st = self.window_stack_height()
        stack = C_ST * h_st * abs(t_z - t_e)
        if not self.cross_ventilation:
            core = max(C_WND * u10_site ** 2, stack) ** 0.5                                # (45), (46)
            return (3600.0 * rho_ref / rho_e * a_tot / 2.0 * core,
                    -3600.0 * rho_ref / rho_z * a_tot / 2.0 * core)
        a_cros = self.cross_window_area()
        wind = self.c_d_w * a_cros * min(u10_site, self.u10_site_max) * self.d_cp ** 0.5
        core = max(wind, a_tot / 2.0 * stack ** 0.5)                                       # (47), (48)
        return 3600.0 * rho_ref / rho_e * core, -3600.0 * rho_ref / rho_z * core

    # ------------------------------------------------------------------ solve
    def solve(self, t_e_c: float, t_z_c: float, u10: float = 0.0, q_sup_dis: float = 0.0,
              q_eta_dis: float = 0.0, q_oda_req: float = 0.0, windows_open: bool = True,
              combustion_flow: float = 0.0, tol: float = 1e-9) -> dict:
        """Air flow rates of the zone for one state of operation (6.4.1).

        ``q_sup_dis`` / ``q_eta_dis``: mechanical flows through the air terminal devices
        (extract negative); ``q_oda_req``: outdoor air requirement for the simplified airing
        (42)/(43); ``windows_open``: airing on/off; ``combustion_flow``: (35) when the
        appliance is on.  Returns flows by path, the reference pressure ``p_z_ref`` (67), the
        totals (68)/(69) and the mass balance residual."""
        if q_sup_dis < 0.0 or q_eta_dis > 0.0:
            raise ValueError("q_sup_dis must be >= 0 and q_eta_dis <= 0")
        if not (q_oda_req >= 0.0 and combustion_flow >= 0.0 and u10 >= 0.0) or not math.isfinite(q_oda_req + combustion_flow + u10 + t_e_c + t_z_c):
            raise ValueError("q_oda_req, combustion_flow and u10 must be >= 0")
        rho_ref = air_density_ref(self.altitude_m)
        rho_e, rho_z = air_density(t_e_c, rho_ref), air_density(t_z_c, rho_ref)
        u_site = wind_speed_site(u10, self.c_rgh_site, self.c_top_site, self.c_rgh_met, self.c_top_met)
        u10_site = wind_speed_10_site(u10, self.c_rgh_10_site, self.c_top_10_site,
                                      self.c_rgh_met, self.c_top_met)
        arg_in = arg_out = 0.0
        pressure_windows = self.window_method == "pressure" and windows_open
        if windows_open and not pressure_windows:
            arg_in, arg_out = self._window_fixed_flows(q_oda_req, u10_site, t_e_c, t_z_c,
                                                       rho_e, rho_z, rho_ref)
        comb_out = -abs(combustion_flow)                                                   # (37), (38)

        def parts(p):
            lv = self._leak_vent_flows(p, u_site, t_e_c, t_z_c, rho_ref)
            a_in, a_out = arg_in, arg_out
            if pressure_windows:
                a_in, a_out = self._window_pressure_flows(p, u_site, t_e_c, t_z_c, rho_ref)
            pdu = (passive_duct_flow(self.passive_duct, p, u_site, t_e_c, t_z_c, rho_ref)
                   if self.passive_duct is not None else 0.0)
            return lv, a_in, a_out, pdu

        def mass(p):
            lv, a_in, a_out, pdu = parts(p)
            m = rho_e * (q_sup_dis + lv["vent_in"] + lv["lea_in"] + a_in + max(pdu, 0.0))
            m += rho_z * (q_eta_dis + lv["vent_out"] + lv["lea_out"] + a_out + comb_out + min(pdu, 0.0))
            return m                                                                       # (65)-(67)

        lo, hi = -1e3, 1e3
        if mass(lo) == 0.0 and mass(hi) == 0.0:
            lo = hi = 0.0                                  # no flow at all: p_z = 0
        else:
            for _ in range(12):                            # up to +-4e9 Pa
                if mass(lo) > 0.0 > mass(hi):
                    break
                lo, hi = lo * 4.0, hi * 4.0
            else:
                raise ValueError(
                    "EN 16798-7: the mass balance (67) has no solution: the mechanical, "
                    "combustion or airing flows cannot be balanced by the openings of the zone "
                    "(for example an extract flow with no leak, vent or window to supply the "
                    "air). Add air flow paths or reduce the mechanical flow.")
        p_z = 0.5 * (lo + hi)
        for _ in range(200):
            p_z = 0.5 * (lo + hi)
            if mass(p_z) > 0.0:
                lo = p_z
            else:
                hi = p_z
            if hi - lo < tol:
                break
        lv, a_in, a_out, pdu = parts(p_z)
        flows_in = q_sup_dis + lv["vent_in"] + lv["lea_in"] + a_in + max(pdu, 0.0)
        flows_out = q_eta_dis + lv["vent_out"] + lv["lea_out"] + a_out + comb_out + min(pdu, 0.0)
        m_in = rho_e * flows_in
        m_out = rho_z * flows_out
        scale = max(abs(m_in), abs(m_out), 1.0)
        if abs(m_in + m_out) > 1e-4 * scale:
            raise ValueError(f"EN 16798-7: the mass balance (67) did not converge "
                             f"(residual {m_in + m_out:.3g} kg/h)")
        return {
            "p_z_ref_pa": p_z, "q_sup_dis": q_sup_dis, "q_eta_dis": q_eta_dis,
            "q_arg_in": a_in, "q_arg_out": a_out, "q_comb_in": 0.0, "q_comb_out": comb_out,
            "q_pdu_in": max(pdu, 0.0), "q_pdu_out": min(pdu, 0.0),
            "q_vent_in": lv["vent_in"], "q_vent_out": lv["vent_out"],
            "q_lea_in": lv["lea_in"], "q_lea_out": lv["lea_out"],
            "q_tot_in": m_in / rho_e,                                                      # (68)
            "q_tot_out": m_out / rho_z,                                                    # (69)
            "mass_residual_kg_h": m_in + m_out, "rho_e": rho_e, "rho_z": rho_z,
            "u_site": u_site, "u10_site": u10_site,
        }


def method2_totals(q_tot_in_m3h: float, q_tot_out_m3h: float) -> dict:
    """Method 2 (clause 7): the national statistical approach only delivers the totals."""
    if q_tot_in_m3h < 0.0 or q_tot_out_m3h > 0.0:
        raise ValueError("q_tot_in must be >= 0 and q_tot_out <= 0")
    return {"q_tot_in": q_tot_in_m3h, "q_tot_out": q_tot_out_m3h}


def ahu_inputs(zone_result: dict, q_oda_req_m3h: Optional[float] = None,
               theta_sup_req_c: Optional[float] = None) -> dict:
    """Mechanical outputs as input of the EN 16798-5-1 AHU model
    (:class:`AHUStepInputs`: required supply / extract flow, outdoor air, LOAD_COMP temperature)."""
    return {"required_supply_flow_m3_h": float(zone_result["q_sup_dis"]),
            "required_extract_flow_m3_h": float(-zone_result["q_eta_dis"]),
            "required_outdoor_air_flow_m3_h": q_oda_req_m3h,
            "required_supply_temperature_c": theta_sup_req_c}


__all__ = [
    "ZoneAirflow", "WindowData", "PassiveDuct", "mechanical_flows", "terminal_required_flows",
    "required_outdoor_air", "required_supply_temperature", "design_supply_flow",
    "supply_outdoor_air_flow", "air_density", "air_density_ref", "wind_speed_site",
    "wind_speed_10_site", "path_pressure_difference", "cp_table", "atd_coefficient",
    "vent_coefficient", "combustion_air_flow", "passive_duct_flow", "method2_totals", "ahu_inputs",
    "f_ctrl", "f_sys", "d_c_cowl_height",
]
