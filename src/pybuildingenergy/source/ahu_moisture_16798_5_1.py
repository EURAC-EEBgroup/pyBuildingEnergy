"""EN 16798-5-1:2017 air handling unit: heat recovery corrections and moisture processes.

Formulae of 6.4.3.2.6 (heat and moisture recovery, Formulae 40-45), 6.4.3.2.9 (cooling and
dehumidification 53-63, humidification 65-66, heating 67-69), 6.4.3.3 (energy 77-84) and
Annex D (rotary heat exchangers, D.1-D.10), with the default values of Tables B.6, B.7 and
B.8 and the constants of Table 27.

All functions are pure; :func:`ventilation_16798_5_1.calculate_sensible_ahu_step` calls them
when the AHU configuration declares the corresponding data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Table 27
P_ATM_PA = 101_325.0
R_W_J_KG = 0.680 * 3.6e6            # latent heat of vaporisation, 0.680 kWh/kg
C_A_J_KG_K = 0.000279 * 3.6e6        # 0.000279 kWh/(kg K) = 1004.4 J/(kg K)

HR_TYPES = ("PLATE_CROSS", "PLATE_COUNTER", "ROT_NH", "ROT_HYG", "ROT_SORP", "PUMP_CIRC", "OTHER")
ROTARY = ("ROT_NH", "ROT_HYG", "ROT_SORP")

# Table B.6: nominal temperature ratio and velocity constants (v_hr;nom = 3.5 m/s, C2 = 1.0)
TABLE_B6 = {
    "PLATE_CROSS": {"eta_hr_nom": 0.60, "C1": -0.0201},
    "PLATE_COUNTER": {"eta_hr_nom": 0.85, "C1": -0.0201},
    "ROT_NH": {"eta_hr_nom": 0.69, "C1": -0.0643},
    "ROT_HYG": {"eta_hr_nom": 0.67, "C1": -0.0684},
    "ROT_SORP": {"eta_hr_nom": 0.69, "C1": -0.0665},
    "PUMP_CIRC": {"eta_hr_nom": 0.71, "C1": -0.0491},
    "OTHER": {"eta_hr_nom": None, "C1": 0.0},
}
V_HR_NOM_M_S = 3.5
C2_DEFAULT = 1.0

# Table B.7: nominal moisture recovery efficiency (at 2 m/s)
TABLE_B7_ETA_XR_NOM = {"ROT_NH": 0.30, "ROT_HYG": 0.42, "ROT_SORP": 0.69}

# Table D.1 as printed in EN 16798-5-1:2017
TABLE_D1_2017 = {
    "C3": 1.0182, "C4": 0.0352, "C5": 0.276, "e1": -2.7,
    "C6": {"ROT_NH": 248.0, "ROT_HYG": 129.0, "ROT_SORP": 16.4},
    "C7": {"ROT_NH": -0.240, "ROT_HYG": 0.357, "ROT_SORP": 0.918},
    "C8": {"ROT_HYG": 23.8},
    "C9": 0.1,
    "C10": {"ROT_NH": -0.200, "ROT_HYG": -0.152, "ROT_SORP": -0.098},
    "C11": 1.0, "C12": 1.0533, "C13": 80_000.0, "C14": 15.0, "e2": -4.0,
}
# Table D.1 with the amendment, as in the EPB Center spreadsheet Demo_EN_16798-5-1
# (2026-05-12): C7 ROT_HYG, C10 ROT_SORP and C11 per heat recovery type.  Default.
TABLE_D1 = {
    **TABLE_D1_2017,
    "C7": {"ROT_NH": -0.240, "ROT_HYG": 0.476, "ROT_SORP": 0.918},
    "C10": {"ROT_NH": -0.200, "ROT_HYG": -0.152, "ROT_SORP": 0.098},
    "C11": {"ROT_NH": 1.7011666666666667, "ROT_HYG": 1.5333333333333334,
            "ROT_SORP": 1.3444202898550726},
    "dx_e_nom": 0.005,                 # nominal condensation potential, now used in (D.5b)
}
ANNEX_D_VERSIONS = ("amended", "2017")

# Table B.8: cooling/heating coil constants
TABLE_B8 = {"CC1": 11.9083, "CC2": 4110.34, "CC3": 235.0, "f_C_bp": 0.1, "eta_coil_C": 0.8, "eta_coil_H": 0.8}


# --------------------------------------------------------------------- psychrometrics
def saturation_pressure_pa(theta_c: float) -> float:
    """Formula (D.7)."""
    return 611.2 * math.exp(17.62 * theta_c / (243.12 + theta_c))


def saturation_humidity_ratio(theta_c: float, p_atm: float = P_ATM_PA) -> float:
    """Formula (D.6) [kg/kg dry air]."""
    p = saturation_pressure_pa(theta_c)
    return 0.622 * p / (p_atm - p)


def coil_surface_humidity_ratio(theta_coil_c: float, cc: Optional[dict] = None) -> float:
    """Formula (56)/(63): saturated humidity ratio on the coil surface."""
    c = cc or TABLE_B8
    return math.exp(c["CC1"] - c["CC2"] / (theta_coil_c + c["CC3"]))


def coil_temperature_for_humidity_ratio(x_coil: float, cc: Optional[dict] = None) -> float:
    """Formula (59): coil surface temperature giving the saturated humidity ratio x_coil."""
    c = cc or TABLE_B8
    return c["CC2"] / (c["CC1"] - math.log(max(x_coil, 1e-9))) - c["CC3"]


# --------------------------------------------------------------------- heat recovery
@dataclass(frozen=True)
class HeatRecoveryData:
    """Product/design data of the heat recovery (Table A.6 / B.6, Annex D)."""

    hr_type: str                                # one of HR_TYPES
    eta_hr_nom: Optional[float] = None          # default Table B.6
    v_hr_des_m_s: Optional[float] = None        # maximum design air velocity
    q_sup_hr_nom_m3_h: Optional[float] = None   # air flow at design conditions
    v_hr_nom_m_s: float = V_HR_NOM_M_S
    C1: Optional[float] = None
    C2: float = C2_DEFAULT
    rotor_speed_rpm: Optional[float] = None     # rotary: actual speed (None = maximum)
    rotor_speed_max_rpm: float = 20.0           # Table B.6
    eta_xr_nom: Optional[float] = None          # default Table B.7 (0 for non-rotary)
    f_xr_other: float = 0.0                     # Formula (45b)
    # Annex D: "amended" (EPB Center spreadsheet 2026, default) or "2017" (printed text)
    annex_d_version: str = "amended"

    def __post_init__(self) -> None:
        t = str(self.hr_type).upper()
        if t not in HR_TYPES:
            raise ValueError(f"heat recovery type must be one of {HR_TYPES}, got {self.hr_type!r}")
        object.__setattr__(self, "hr_type", t)
        v = str(self.annex_d_version).strip().lower()
        if v not in ANNEX_D_VERSIONS:
            raise ValueError(f"annex_d_version must be one of {ANNEX_D_VERSIONS}")
        object.__setattr__(self, "annex_d_version", v)

    @property
    def nominal_temperature_ratio(self) -> float:
        v = self.eta_hr_nom if self.eta_hr_nom is not None else TABLE_B6[self.hr_type]["eta_hr_nom"]
        if v is None:
            raise ValueError("heat recovery type OTHER needs eta_hr_nom")
        return float(v)


def effective_velocity(data: HeatRecoveryData, q_sup_m3_h: float, f_oda_min: float = 1.0) -> Optional[float]:
    """Formula (43); None when the design data are not given."""
    if data.v_hr_des_m_s is None or not data.q_sup_hr_nom_m3_h:
        return None
    return data.v_hr_des_m_s * q_sup_m3_h * f_oda_min / data.q_sup_hr_nom_m3_h


def temperature_ratio(data: HeatRecoveryData, q_sup_m3_h: float, q_eta_m3_h: float,
                      f_oda_min: float = 1.0) -> dict:
    """Formula (40) eta_hr = eta_hr;nom * f_q * f_v * f_n with (41), (42), (D.1)."""
    eta_nom = data.nominal_temperature_ratio
    f_q = 1.0
    if q_sup_m3_h > 0.0:
        base = (q_eta_m3_h - q_sup_m3_h) / (q_sup_m3_h * f_oda_min) + 1.0
        f_q = max(base, 0.0) ** 0.4                                     # (41)
    v_eff = effective_velocity(data, q_sup_m3_h, f_oda_min)
    c1 = data.C1 if data.C1 is not None else TABLE_B6[data.hr_type]["C1"]
    f_v = c1 * (v_eff - data.v_hr_nom_m_s) + data.C2 if v_eff is not None else 1.0   # (42)
    f_n = 1.0
    if data.hr_type in ROTARY and data.rotor_speed_rpm is not None:
        d = TABLE_D1_2017
        ratio = max(data.rotor_speed_rpm / data.rotor_speed_max_rpm, 1e-6)
        f_n = d["C3"] - d["C4"] * (ratio + d["C5"]) ** d["e1"]          # (D.1)
    eta = max(0.0, min(1.0, eta_nom * f_q * f_v * f_n))
    return {"eta_hr": eta, "eta_hr_nom": eta_nom, "f_q": f_q, "f_v": f_v, "f_n": f_n, "v_hr_eff": v_eff}


def moisture_recovery_efficiency(data: HeatRecoveryData, x_eta_in: float, theta_e_c: float,
                                 x_e: float, q_sup_m3_h: float, q_eta_m3_h: float,
                                 f_oda_min: float = 1.0) -> dict:
    """Formulae (45a)-(45c) with Annex D (D.5)-(D.10)."""
    t = data.hr_type
    amended = str(getattr(data, "annex_d_version", "amended")).lower() != "2017"
    if t == "OTHER":
        eta_nom = float(data.eta_xr_nom or 0.0)
        return {"eta_xr": max(0.0, min(1.0, eta_nom * data.f_xr_other)), "eta_xr_nom": eta_nom}
    if t not in ROTARY:
        return {"eta_xr": 0.0, "eta_xr_nom": 0.0}                       # (45c)
    d = TABLE_D1 if amended else TABLE_D1_2017
    eta_nom = float(data.eta_xr_nom if data.eta_xr_nom is not None else TABLE_B7_ETA_XR_NOM[t])
    x_e_sat = saturation_humidity_ratio(theta_e_c)                      # (D.6)
    dx = x_eta_in - x_e_sat
    if t == "ROT_HYG":
        if amended:
            # (D.5b) amended; (D.5c) no longer needed
            dxn = dx - d["dx_e_nom"]
            f_dx = max(0.0, 1.0 + d["C6"][t] * dxn, d["C7"][t] + d["C8"][t] * dxn)
        elif (x_e - x_e_sat) > 0.0:
            f_dx = d["C6"][t] * dx + d["C7"][t]                         # (D.5b) 2017
        else:
            f_dx = max(0.0, d["C8"][t] * dx + d["C7"][t])               # (D.5c) 2017
    else:
        f_dx = max(0.0, d["C6"][t] * dx + d["C7"][t])                   # (D.5a)
    f_q = 1.0
    if q_sup_m3_h > 0.0 and f_oda_min > 0.0:
        imbalance = (q_eta_m3_h - q_sup_m3_h) / (q_sup_m3_h * f_oda_min)
        # (D.8): amended 1 - C9 * imbalance (1 for balanced flow); printed 2017 adds "+1"
        f_q = 1.0 - d["C9"] * (imbalance if amended else imbalance + 1.0)
    v_eff = effective_velocity(data, q_sup_m3_h, f_oda_min)
    c11 = d["C11"][t] if isinstance(d["C11"], dict) else d["C11"]
    f_v = d["C10"][t] * (v_eff - data.v_hr_nom_m_s) + c11 if v_eff is not None else 1.0  # (D.9)
    f_n = 1.0
    if data.rotor_speed_rpm is not None:
        f_n = max(0.0, d["C12"] - d["C13"] * (data.rotor_speed_rpm + d["C14"]) ** d["e2"])    # (D.10)
    eta = max(0.0, min(1.0, eta_nom * f_dx * f_q * f_v * f_n))
    return {"eta_xr": eta, "eta_xr_nom": eta_nom, "f_dx": f_dx, "f_q_x": f_q, "f_v_x": f_v, "f_n_x": f_n}


# --------------------------------------------------------------------- coils
@dataclass(frozen=True)
class CoilData:
    """Cooling coil constants (Table A.8 / B.8)."""

    CC1: float = TABLE_B8["CC1"]
    CC2: float = TABLE_B8["CC2"]
    CC3: float = TABLE_B8["CC3"]
    f_C_bp: float = TABLE_B8["f_C_bp"]

    def as_dict(self) -> dict:
        return {"CC1": self.CC1, "CC2": self.CC2, "CC3": self.CC3}


def cooling_and_dehumidification(theta_rca: float, x_rca: float, theta_sup_req: float,
                                 x_sup_max: Optional[float], coil: CoilData = CoilData()) -> dict:
    """Required state after the cooling coil, Formulae (53b)-(60) and (63).

    ``theta_rca``/``x_rca``: air entering the coil; ``theta_sup_req``: supply temperature
    required by the temperature control; ``x_sup_max``: maximum supply humidity ratio
    (None = no dehumidification control).  Returns the temperature and humidity ratio after
    the coil, the coil surface temperature and the dehumidification-driven temperature.
    """
    bp = coil.f_C_bp
    cc = coil.as_dict()
    theta_dhu_req = theta_rca                                           # (60b)
    if x_sup_max is not None and x_sup_max < x_rca:
        x_dhu_coil = (x_sup_max - bp * x_rca) / (1.0 - bp)              # (58)
        theta_dhu_coil = coil_temperature_for_humidity_ratio(x_dhu_coil, cc)   # (59)
        theta_dhu_req = bp * theta_rca + (1.0 - bp) * theta_dhu_coil    # (60a)
    theta_c_req = min(theta_dhu_req, theta_sup_req, theta_rca)          # (53b)
    return _coil_outlet(theta_rca, x_rca, theta_c_req, coil) | {"theta_SUP_DHU_req": theta_dhu_req}


def _coil_outlet(theta_rca: float, x_rca: float, theta_c: float, coil: CoilData) -> dict:
    """Coil surface temperature (57) and outlet humidity ratio (63) for an outlet theta_c."""
    bp = coil.f_C_bp
    if theta_c >= theta_rca - 1e-12:
        return {"theta_SUP_C": theta_rca, "x_SUP_C": x_rca, "theta_C_coil": None}
    theta_coil = (theta_c - bp * theta_rca) / (1.0 - bp)               # (57)
    x_c = min(bp * x_rca + (1.0 - bp) * coil_surface_humidity_ratio(theta_coil, coil.as_dict()), x_rca)  # (63)
    return {"theta_SUP_C": theta_c, "x_SUP_C": x_c, "theta_C_coil": theta_coil}


def coil_power_w(q_m3_s: float, rho: float, theta_in: float, x_in: float,
                 theta_out: float, x_out: float) -> tuple:
    """Formula (80) as power: (sensible, latent) [W] removed by the cooling coil."""
    sens = q_m3_s * rho * C_A_J_KG_K * (theta_in - theta_out)
    lat = q_m3_s * rho * R_W_J_KG * (x_in - x_out)
    return max(sens, 0.0), max(lat, 0.0)


def limit_cooling_coil(theta_rca: float, x_rca: float, theta_c_req: float, q_m3_s: float,
                       rho: float, cap_w: Optional[float], coil: CoilData = CoilData()) -> dict:
    """Outlet state when the total (sensible + latent) coil power is limited to ``cap_w``."""
    out = _coil_outlet(theta_rca, x_rca, theta_c_req, coil)
    s, l = coil_power_w(q_m3_s, rho, theta_rca, x_rca, out["theta_SUP_C"], out["x_SUP_C"])
    if cap_w is None or s + l <= cap_w + 1e-9:
        return out | {"Q_sens_w": s, "Q_lat_w": l, "limited": False}
    lo, hi = theta_c_req, theta_rca                                     # bisection on theta_c
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        o = _coil_outlet(theta_rca, x_rca, mid, coil)
        s, l = coil_power_w(q_m3_s, rho, theta_rca, x_rca, o["theta_SUP_C"], o["x_SUP_C"])
        if s + l > cap_w:
            lo = mid
        else:
            hi = mid
    o = _coil_outlet(theta_rca, x_rca, hi, coil)
    s, l = coil_power_w(q_m3_s, rho, theta_rca, x_rca, o["theta_SUP_C"], o["x_SUP_C"])
    return o | {"Q_sens_w": s, "Q_lat_w": l, "limited": True}


def humidification(x_sup_c: float, x_sup_min: Optional[float], hum_type: str = "STEAM") -> dict:
    """Formulae (65), (66a)/(66b): humidity ratio after the humidifier and temperature drop."""
    x_hu = x_sup_c if x_sup_min is None else max(x_sup_min, x_sup_c)
    dx = x_hu - x_sup_c
    if str(hum_type).upper() == "STEAM":
        d_theta = 0.0                                                   # (66a)
    else:
        d_theta = max(0.0, R_W_J_KG / C_A_J_KG_K * dx)                  # (66b)
    return {"x_SUP_HU": x_hu, "dx_HU": dx, "d_theta_SUP_HU": d_theta}


# ---------------------------------------------------------------- distribution and leakage
# Table 10: duct leakage factor f_lea;du by duct airtightness class (B.3: new B, existing unknown)
TABLE_10_DUCT_LEAKAGE = {"UNKNOWN": 1.45, "A": 1.18, "B": 1.06, "C": 1.02, "D": 1.0}
# Table 11: AHU leakage factor f_lea;ahu by AHU airtightness class (B.4: new L2, existing L3)
TABLE_11_AHU_LEAKAGE = {"L3": 1.10, "L2": 1.04, "L1": 1.01}


def _hr_family(hr_type: Optional[str]) -> str:
    t = str(hr_type or "PLATE").upper()
    if t.startswith("PLATE"):
        return "PLATE"
    if t in ROTARY:
        return "ROT"
    if t == "PUMP_CIRC":
        return "PUMP_CIRC"
    return "PLATE"


# Table B.5: (SUP_FAN_LOC, ETA_FAN_LOC) -> {family: (f_lea;hr;SUP, f_lea;hr;ETA, f_ODA;hr)}.
# PLATE: the printed f_ODA;hr = 1.02 would be an outdoor air share above 1; it is set to 1.0
# as in the EPB Center reference spreadsheet (Demo_EN_16798-5-1, 2026-05-12).
TABLE_B5 = {
    ("UP_HR", "UP_HR"):     {"PLATE": (1.02, 1.02, 1.0), "ROT": (1.04, 1.0, 1.0),   "PUMP_CIRC": (1.0, 1.0, 1.0)},
    ("UP_HR", "DOWN_HR"):   {"PLATE": (1.02, 1.02, 1.0), "ROT": (1.10, 1.10, 1.0),  "PUMP_CIRC": (1.0, 1.0, 1.0)},
    ("DOWN_HR", "UP_HR"):   {"PLATE": (1.02, 1.02, 1.0), "ROT": (1.10, 1.10, 0.909), "PUMP_CIRC": (1.0, 1.0, 1.0)},
    ("DOWN_HR", "DOWN_HR"): {"PLATE": (1.02, 1.02, 1.0), "ROT": (1.02, 1.0, 0.98),  "PUMP_CIRC": (1.0, 1.0, 1.0)},
}


def hr_leakage_factors(hr_type: Optional[str], sup_fan_loc: str, eta_fan_loc: str) -> tuple:
    """Table B.5: (f_lea;hr;SUP, f_lea;hr;ETA, f_ODA;hr) for the fan arrangement."""
    key = (str(sup_fan_loc).upper(), str(eta_fan_loc).upper())
    if key not in TABLE_B5:
        raise ValueError(f"fan locations must be UP_HR or DOWN_HR, got {key}")
    return TABLE_B5[key][_hr_family(hr_type)]


# Table B.14: exhaust air temperature limit after the heat recovery, indirect frost control
TABLE_B14_ETA_HR_LIM = {
    ("RESIDENTIAL", "PLATE"): 5.0, ("RESIDENTIAL", "ROT_HYG"): 0.0,
    ("NON_RESIDENTIAL", "PLATE"): 0.0, ("NON_RESIDENTIAL", "ROT_HYG"): -5.0,
    ("NON_RESIDENTIAL", "ROT_NH"): -10.0, ("NON_RESIDENTIAL", "ROT_SORP"): -5.0,
    ("NON_RESIDENTIAL", "PUMP_CIRC"): 5.0,
}
# Table B.14, direct frost control: outdoor limit temperatures
FROST_DIRECT_T_LIM1_C = -10.0
FROST_DIRECT_T_LIM2_C = -5.0


def frost_exhaust_limit(building_use: str, hr_type: Optional[str]) -> Optional[float]:
    """Table B.14 limit for indirect frost control, or None when the table has no entry."""
    use = str(building_use).upper().replace("-", "_")
    t = str(hr_type or "PLATE").upper()
    if t.startswith("PLATE"):
        t = "PLATE"
    return TABLE_B14_ETA_HR_LIM.get((use, t))


# Table B (6.4.3.2.8): humidification efficiency of the adiabatic cooling humidifier
ETA_HU_AC_DEFAULT = 0.9

_CP_V_J_KG_K = 1860.0
_H_FG0_J_KG = 2_501_000.0


def wet_bulb_temperature(theta_c: float, x: float, p_atm: float = P_ATM_PA) -> float:
    """Wet-bulb (adiabatic saturation) temperature [°C] of air at theta_c, x [kg/kg].

    Solves h(theta, x) = h(theta_wb, x_sat(theta_wb)) by bisection (6.4.3.2.8 asks for a
    psychrometric method with iteration)."""
    def h(t, xx):
        return C_A_J_KG_K * t + xx * (_H_FG0_J_KG + _CP_V_J_KG_K * t)
    target = h(theta_c, x)
    lo, hi = -40.0, float(theta_c)
    if saturation_humidity_ratio(hi, p_atm) <= x:
        return float(theta_c)                       # already saturated
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if h(mid, saturation_humidity_ratio(mid, p_atm)) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- air density (Table 27)
RHO_A_REF_KG_M3 = 1.204


def air_density(altitude_m: float = 0.0) -> float:
    """Table 27, note a: rho_a = rho_a;ref * (1 - 0.00651 h_alt / 293)^4.255."""
    return RHO_A_REF_KG_M3 * (1.0 - 0.00651 * float(altitude_m) / 293.0) ** 4.255


# ---------------------------------------------------------------- Annex C: ground preheating
# Table C.1: month with the lowest monthly mean outdoor temperature -> t_an;min [h], f_t
TABLE_C1 = {1: (350.0, 1.08 * math.pi), 2: (1100.0, 1.25 * math.pi), 3: (1800.0, 1.41 * math.pi),
            11: (7650.0, 2.75 * math.pi), 12: (8400.0, 2.92 * math.pi)}
# Table C.2: soil material -> (lambda W/(m K), rho kg/m3, c J/(kg K))
TABLE_C2 = {"moist_soil": (1.5, 1400.0, 1400.0), "dry_sand": (0.7, 1500.0, 920.0),
            "moist_sand": (1.88, 1500.0, 1200.0), "moist_clay": (1.45, 1800.0, 1340.0),
            "wet_clay": (2.9, 1800.0, 1590.0)}


@dataclass(frozen=True)
class GroundExchangerData:
    """Earth-to-air heat exchanger (Annex C).

    ``u_w_m2_k`` is U_du; when None it is computed with (C.5)-(C.6) from the duct
    conductivity, diameters and air velocity.  ``control``: "no_ctrl" (32a/d) or "bypass"
    (GND_PREH_CTRL = BYPASS, (32b/c/e/f)).
    """
    area_m2: float                                   # A_s, inner surface
    mean_annual_temperature_c: float                 # theta_e;mn;an
    max_monthly_mean_temperature_c: float            # theta_e;max;m
    depth_m: float = 1.5                             # z
    soil: str = "moist_soil"                         # Table C.2
    coldest_month: int = 1                           # Table C.1
    u_w_m2_k: Optional[float] = None
    duct_conductivity_w_m_k: float = 0.4             # lambda_du (C.5)
    duct_inner_diameter_m: float = 0.2               # d_i
    duct_outer_diameter_m: float = 0.22              # d_o
    air_velocity_m_s: float = 2.0                    # v (C.6)
    max_relative_humidity_pct: float = 100.0         # phi_ODA;gnd;max (C.9)
    control: str = "no_ctrl"

    def __post_init__(self) -> None:
        if self.soil not in TABLE_C2:
            raise ValueError(f"soil must be one of {list(TABLE_C2)}")
        if int(self.coldest_month) not in TABLE_C1:
            raise ValueError(f"coldest_month must be one of {list(TABLE_C1)} (Table C.1)")
        if self.area_m2 <= 0.0 or self.depth_m < 0.0:
            raise ValueError("area_m2 must be > 0 and depth_m >= 0")
        if str(self.control).lower() not in ("no_ctrl", "bypass"):
            raise ValueError("ground exchanger control must be 'no_ctrl' or 'bypass'")


def ground_temperature(data: GroundExchangerData, hour_of_year: float) -> float:
    """(C.2)-(C.4): soil temperature at depth z and hour of the year."""
    lam, rho, c = TABLE_C2[data.soil]
    xi = data.depth_m * math.sqrt(math.pi * rho * c / (lam * 8760.0 * 3600.0))      # (C.3)
    _, f_t = TABLE_C1[int(data.coldest_month)]                                      # (C.4)
    return (data.mean_annual_temperature_c
            + (data.max_monthly_mean_temperature_c - data.mean_annual_temperature_c)
            * math.exp(-xi) * math.cos(2.0 * math.pi * hour_of_year / 8760.0 - xi - f_t))  # (C.2)


def ground_exchanger_u(data: GroundExchangerData, theta_e_c: float) -> float:
    """(C.5)-(C.6) with theta_m = theta_e (as allowed to avoid iteration)."""
    if data.u_w_m2_k is not None:
        return float(data.u_w_m2_k)
    tm = theta_e_c / 100.0
    h_i = ((4.13 + 0.23 * tm - 0.0077 * tm ** 2) * data.air_velocity_m_s ** 0.75
           / data.duct_inner_diameter_m ** 0.25)                                    # (C.6)
    r = (math.log(data.duct_outer_diameter_m / data.duct_inner_diameter_m)
         / (2.0 * math.pi * data.duct_conductivity_w_m_k))
    return 1.0 / (r + 1.0 / h_i)                                                     # (C.5)


def ground_preheating(data: GroundExchangerData, theta_e_c: float, x_e: Optional[float],
                      hour_of_year: float, q_m3_h: float, rho_cp_j_m3_k: float,
                      p_atm: float = P_ATM_PA) -> dict:
    """(C.1), (C.7)-(C.10): temperature and moisture change of the outdoor air."""
    if q_m3_h <= 0.0:
        return {"d_theta": 0.0, "d_x": 0.0, "theta_gnd": ground_temperature(data, hour_of_year)}
    th_g = ground_temperature(data, hour_of_year)
    u = ground_exchanger_u(data, theta_e_c)
    d_theta = (th_g - theta_e_c) * (1.0 - math.exp(-u * data.area_m2
                                                   / (q_m3_h / 3600.0 * rho_cp_j_m3_k)))  # (C.1)
    d_x = 0.0
    if x_e is not None:
        p_p = saturation_pressure_pa(theta_e_c + d_theta) * data.max_relative_humidity_pct / 100.0  # (C.9)-(C.10)
        x_gnd = min(x_e, 0.622 * p_p / (p_atm - p_p))                                # (C.8)
        d_x = x_gnd - x_e                                                            # (C.7)
    return {"d_theta": d_theta, "d_x": d_x, "theta_gnd": th_g}


# ---------------------------------------------------------------- national data (Annex A / NA)
# Default data sets: "IT" = UNI EN 16798-5-1:2018/NA:2025 (Italian national annex, draft for
# public enquiry); "EN" = informative Annex B of EN 16798-5-1:2017.  Values that only
# depend on the product are taken from "design or inspection" in both and are not listed.
# The specific pump energies are given in the annex as kWh/m3; the EPB Center spreadsheet
# notes that they are too large by a factor 1000, so they are used here as Wh/m3.
_HR_DP_DES = {"PLATE_CROSS": 300.0, "PLATE_COUNTER": 500.0, "ROT_NH": 400.0, "ROT_HYG": 400.0,
              "ROT_SORP": 400.0, "PUMP_CIRC": 500.0}          # Table NA.6 / B.6 [Pa]
_HUM_PUMP_WH_M3 = {("CONTACT", "no_ctrl"): 0.01, ("ROT_SPRAY", "no_ctrl"): 0.20,
                   ("ROT_SPRAY", "on_off"): 0.20, ("ROT_SPRAY", "speed"): 0.20,
                   ("HI_PRES", "speed"): 0.04, ("HYBRID", "on_off"): 0.02}   # Table NA.10 / B.10
NATIONAL_DATA = {
    "IT": {
        "source": "UNI EN 16798-5-1:2018/NA:2025",
        "duct_leakage_class": {"new": "B", "existing": "UNKNOWN"},     # Table NA.2
        "ahu_leakage_class": {"new": "L2", "existing": "L3"},         # Table NA.3
        "hr_design_pressure_drop_pa": _HR_DP_DES,                      # Table NA.6
        "humidifier_pump_wh_m3": _HUM_PUMP_WH_M3,                      # Table NA.10
        "fan_heat_recovery_factor": {"IN_AIR": 1.0, "OUTS_AIR": 0.6},  # Table NA.12
        "pressure_control_fraction": {"ventilation": (0.4, 0.67),      # Table NA.13
                                      "air_conditioning": (0.3, 0.5)},
        "controls_fan_power_fraction": 0.05,                          # Table NA.9
        "eta_hu_ac": 0.9,                                              # NA.4.6
    },
    "EN": {
        "source": "EN 16798-5-1:2017, Annex B",
        "duct_leakage_class": {"new": "B", "existing": "UNKNOWN"},     # Table B.3
        "ahu_leakage_class": {"new": "L2", "existing": "L3"},         # Table B.4
        "hr_design_pressure_drop_pa": _HR_DP_DES,                      # Table B.6
        "humidifier_pump_wh_m3": _HUM_PUMP_WH_M3,                      # Table B.10
        "fan_heat_recovery_factor": {"IN_AIR": 1.0, "OUTS_AIR": 0.6},  # Table B.12
        "pressure_control_fraction": {"ventilation": (0.4, 0.67),
                                      "air_conditioning": (0.3, 0.5)},
        "controls_fan_power_fraction": None,                          # no default in Annex B
        "eta_hu_ac": 0.9,
    },
}
HUMIDIFIER_TYPES = ("STEAM", "ADIABATIC", "CONTACT", "ROT_SPRAY", "HI_PRES", "HYBRID", "OTHER")
