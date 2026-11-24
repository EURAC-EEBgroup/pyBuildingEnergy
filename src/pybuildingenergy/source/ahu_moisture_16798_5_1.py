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

# Table D.1
TABLE_D1 = {
    "C3": 1.0182, "C4": 0.0352, "C5": 0.276, "e1": -2.7,
    "C6": {"ROT_NH": 248.0, "ROT_HYG": 129.0, "ROT_SORP": 16.4},
    "C7": {"ROT_NH": -0.240, "ROT_HYG": 0.357, "ROT_SORP": 0.918},
    "C8": {"ROT_HYG": 23.8},
    "C9": 0.1,
    "C10": {"ROT_NH": -0.200, "ROT_HYG": -0.152, "ROT_SORP": -0.098},
    "C11": 1.0, "C12": 1.0533, "C13": 80_000.0, "C14": 15.0, "e2": -4.0,
}

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

    def __post_init__(self) -> None:
        t = str(self.hr_type).upper()
        if t not in HR_TYPES:
            raise ValueError(f"heat recovery type must be one of {HR_TYPES}, got {self.hr_type!r}")
        object.__setattr__(self, "hr_type", t)

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
        d = TABLE_D1
        ratio = max(data.rotor_speed_rpm / data.rotor_speed_max_rpm, 1e-6)
        f_n = d["C3"] - d["C4"] * (ratio + d["C5"]) ** d["e1"]          # (D.1)
    eta = max(0.0, min(1.0, eta_nom * f_q * f_v * f_n))
    return {"eta_hr": eta, "eta_hr_nom": eta_nom, "f_q": f_q, "f_v": f_v, "f_n": f_n, "v_hr_eff": v_eff}


def moisture_recovery_efficiency(data: HeatRecoveryData, x_eta_in: float, theta_e_c: float,
                                 x_e: float, q_sup_m3_h: float, q_eta_m3_h: float,
                                 f_oda_min: float = 1.0) -> dict:
    """Formulae (45a)-(45c) with Annex D (D.5)-(D.10)."""
    t = data.hr_type
    if t == "OTHER":
        eta_nom = float(data.eta_xr_nom or 0.0)
        return {"eta_xr": max(0.0, min(1.0, eta_nom * data.f_xr_other)), "eta_xr_nom": eta_nom}
    if t not in ROTARY:
        return {"eta_xr": 0.0, "eta_xr_nom": 0.0}                       # (45c)
    d = TABLE_D1
    eta_nom = float(data.eta_xr_nom if data.eta_xr_nom is not None else TABLE_B7_ETA_XR_NOM[t])
    x_e_sat = saturation_humidity_ratio(theta_e_c)                      # (D.6)
    dx = x_eta_in - x_e_sat
    if t == "ROT_HYG":
        if (x_e - x_e_sat) > 0.0:
            f_dx = d["C6"][t] * dx + d["C7"][t]                         # (D.5b)
        else:
            f_dx = max(0.0, d["C8"][t] * dx + d["C7"][t])               # (D.5c)
    else:
        f_dx = max(0.0, d["C6"][t] * dx + d["C7"][t])                   # (D.5a)
    f_q = 1.0
    if q_sup_m3_h > 0.0:
        # (D.8) as printed in EN 16798-5-1:2017: gives 1 - C9 = 0.9 for balanced flow
        # (the correction is not normalised to 1, unlike Formula 41)
        f_q = 1.0 - d["C9"] * ((q_eta_m3_h - q_sup_m3_h) / (q_sup_m3_h * f_oda_min) + 1.0)
    v_eff = effective_velocity(data, q_sup_m3_h, f_oda_min)
    f_v = d["C10"][t] * (v_eff - data.v_hr_nom_m_s) + d["C11"] if v_eff is not None else 1.0  # (D.9)
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
