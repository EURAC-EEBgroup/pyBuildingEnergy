"""EN 16798-5-2:2017 - Ventilation systems, Method 2: distribution and generation.

Monthly, yearly or bin calculation of small packaged ventilation units (typically
residential), with ventilation stages ``k``, heat recovery or recirculation, preheating /
precooling (frost protection, ground heat exchanger, solar air panel), reheating, duct and
casing losses, fan and auxiliary energy, and the extract air heat pump for domestic hot
water, space heating and space cooling.  Humidification, dehumidification and the fan
temperature rise are outside the scope of the method (Clause 1, 6.4.2.2 NOTE 7).

Units: flows in m3/h, temperatures in degC, times in h, energies in kWh, powers in kW
(as in the standard); ``(rho c)_a`` in kWh/(m3 K).

Deviation from the printed text (as in EN 16798-5-1 (18)): in (20) the extract duct heat
loss lowers the extract air temperature (printed ``+``); otherwise a duct crossing a cold
attic would warm the extract air.

Main entry points: :func:`calculate_interval` (one month, year or bin) and
:func:`monthly_from_hourly` (intervals built from an hourly EN ISO 52016-1 result).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

# --------------------------------------------------------------------------- Table 9
P_ATM_PA = 101_325.0
RHO_A_SEA_KG_M3 = 1.204
C_A_KWH_KG_K = 0.000279
R_W_KWH_KG = 0.680


def air_density(altitude_m: float = 0.0) -> float:
    """Table 9, note a: rho_a = rho_a;sea (1 - 0.00651 h / 288)^4.255."""
    return RHO_A_SEA_KG_M3 * (1.0 - 0.00651 * float(altitude_m) / 288.0) ** 4.255


# --------------------------------------------------------------------------- Annex B tables
TABLE_B2_DUCT_LEAKAGE = {"A": 1.06, "B": 1.02, "C": 1.00, "D": 1.00}          # f_lea;du
TABLE_B3_DUCT_CLASS = {"new": "B", "existing": "A"}
TABLE_B4_AHU_LEAKAGE = {"1": 1.00, "2": 1.01, "3": 1.02}  # A1/B1/C1, A2/B2/C2, A3/B3/C3
TABLE_B5_AHU_CLASS = {"new": "A2", "existing": "A3"}
TABLE_B7_DEFROST_HOURS = {"cold": 1003.0, "average": 168.0, "warm": 0.0}      # h/a at -4 degC
TABLE_B8_DEFROST_DT = {"cold": 5.2, "average": 2.4, "warm": 0.0}               # K at -4 degC
TABLE_B9_DEFROST_CTRL = {"temperature_time": 0.8, "electronic": 1.0, "two_step": 1.6,
                         "one_step": 3.1}
TABLE_B10_F_GEN_HP = {
    "unheated": {"air_air_hp": 0.02, "air_water_hp": 0.02, "air_air_water_hp": 0.02,
                 "electric_coil": 0.00, "water_coil": 0.01},
    "heated": {"air_air_hp": 0.0, "air_water_hp": 0.0, "air_air_water_hp": 0.0,
               "electric_coil": 0.0, "water_coil": 0.0},
}
TABLE_B11_F_GEN_CU = {"cooled": 0.02, "uncooled": 0.00}
TABLE_B12_DP_DEFROST = {"none": 0.0, "ground": 20.0, "electric": 10.0, "water": 20.0,
                        "brine": 40.0}
TABLE_B13_ETA_FAN = {"ac": 0.20, "ec_forward": 0.26, "ec_backward": 0.35}
TABLE_B14_FLOW_CTRL = {"manual": 1.00, "clock": 0.95, "central_dcv": 0.85, "local_dcv": 0.65}
TABLE_B15_EXPONENT = {"single_speed": 1.0, "two_speed": 1.2, "three_speed": 1.5,
                      "variable_speed": 2.0}
TABLE_B16_F_ETA = ((19.0, 0.98), (20.0, 1.00), (21.0, 1.02), (22.0, 1.04), (23.0, 1.06),
                   (24.0, 1.08), (25.0, 1.10), (26.0, 1.12))
TABLE_B16_F_H = {"30/25": 1.154, "35/28": 1.068, "40/32": 1.0, "50/40": 0.894, "55/45": 0.853}
TABLE_B16_F_V = ((0.3, 0.71), (0.4, 0.79), (0.5, 0.86), (0.6, 0.93), (0.7, 1.0), (0.8, 1.09),
                 (0.9, 1.18), (1.0, 1.24), (1.5, 1.62), (2.0, 2.0), (2.5, 2.39), (3.0, 2.78),
                 (3.5, 3.16), (4.0, 3.55))
TABLE_B17_LIMITS = {"preheating": 0.0, "precooling": 26.0}
# B.3.1 heat pump defaults
HP_DEFAULTS = {"p_W": 0.0025, "COP_W": 3.4, "p_H": 0.0027, "COP_H": 3.9, "p_C": 0.0027,
               "EER": 2.6, "f_eta": 1.0, "f_h": 1.068, "f_v": 1.0}


def _interp(table, x):
    pts = sorted(table)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def insulation_factor(r_cas_m2k_w: float) -> float:
    """Table B.6: AHU insulation factor f_ins;ahu from the casing thermal resistance."""
    r = float(r_cas_m2k_w)
    if r > 1.0:
        return 0.00
    if r >= 0.75:
        return 0.01
    if r >= 0.50:
        return 0.02
    return 0.03


def correction_extract_temperature(theta_eta_c: float) -> float:
    """Table B.16, f_dtheta-ETA (interpolated, clamped to 19-26 degC)."""
    return _interp(TABLE_B16_F_ETA, theta_eta_c)


def correction_flow(q_mean_over_q_max: float) -> float:
    """Table B.16, f_V (interpolated, clamped to 0.3-4.0 q_v,max)."""
    return _interp(TABLE_B16_F_V, q_mean_over_q_max)


# --------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class HeatPumpData:
    """Extract air heat pump (B.3.1 defaults); ``None`` disables a service."""
    p_W_kw_per_m3h: Optional[float] = None       # domestic hot water
    COP_W: float = HP_DEFAULTS["COP_W"]
    p_H_kw_per_m3h: Optional[float] = None       # space heating
    COP_H: float = HP_DEFAULTS["COP_H"]
    p_C_kw_per_m3h: Optional[float] = None       # space cooling
    EER: float = HP_DEFAULTS["EER"]
    f_dtheta_h: float = HP_DEFAULTS["f_h"]       # Table B.16, heating circuit temperatures
    f_gen_hp: float = 0.0                        # Table B.10
    f_gen_cu: float = 0.0                        # Table B.11
    # (46) as printed removes the DHW time from t_on;H;max;k AND subtracts Q_W;gen;hp;out; the
    # second deduction counts the DHW share twice.  True keeps the printed text.
    deduct_dhw_energy_46: bool = True


@dataclass(frozen=True)
class Unit52Config:
    """Product and system data of the ventilation unit (Tables 6, 7, Annex B)."""
    duct_leakage_factor: float = 1.0             # f_lea;du (Table B.2); 1 if ducts in ventilated area
    ahu_leakage_factor: float = 1.0              # f_lea;ahu (Table B.4); 1 if unit in ventilated area
    ahu_insulation_factor: float = 0.0           # f_ins;ahu (Table B.6)
    heat_recovery: str = "plate"                 # plate, rotary, none (recirculation units)
    h_du_sup_nc_kw_k: float = 0.0                # H_du;SUP;nc
    h_du_sup_cnd_kw_k: float = 0.0               # H_du;SUP;cnd
    h_du_eta_nc_kw_k: Optional[float] = None     # H_du;ETA;nc, default H_du;SUP;nc (23f)
    ahu_au_kw_k: float = 0.0                     # (A U)ahu (28)
    ahu_in_conditioned_zone: bool = True         # (28) NOTE, (30)
    spi_kw_per_m3h: Optional[Sequence[float]] = None  # SPI_k per stage (31a)
    spi_single_kw_per_m3h: Optional[float] = None     # single SPI (31b)
    q_sup_max_m3h: Optional[float] = None        # q_V;SUP;dis;max for (31b)
    flow_control_factor: float = 1.0             # f_flow,ctrl (Table B.14)
    motor_exponent: float = 1.0                  # x (Table B.15)
    fan_efficiency: float = TABLE_B13_ETA_FAN["ec_backward"]  # eta_fan (Table B.13)
    dp_defrost_pa: float = 0.0                   # Table B.12
    rotor_power_max_kw: float = 0.0              # P_hr;rot;max
    rotor_speed_max_rpm: float = 20.0
    controls_power_kw: float = 0.0               # P_el;V;ctrl
    defrost_control_factor: float = 1.0          # f_defrost,ctrl (Table B.9)
    preheating_limit_c: float = TABLE_B17_LIMITS["preheating"]
    precooling_limit_c: float = TABLE_B17_LIMITS["precooling"]
    altitude_m: float = 0.0
    heat_pump: Optional[HeatPumpData] = None
    # (20) as printed adds the extract duct heat loss term (+).  Default False: the loss
    # lowers the extract air temperature (physical direction, as in EN 16798-5-1 (18)).
    extract_duct_sign_as_printed: bool = False

    def __post_init__(self) -> None:
        if self.duct_leakage_factor < 1.0 or self.ahu_leakage_factor < 1.0:
            raise ValueError("leakage factors must be >= 1")
        if str(self.heat_recovery).lower() not in ("plate", "rotary", "none"):
            raise ValueError("heat_recovery must be 'plate', 'rotary' or 'none'")
        if not (0.0 < self.fan_efficiency <= 1.0):
            raise ValueError("fan_efficiency must be in (0, 1]")

    @property
    def rho_c(self) -> float:
        """(rho c)_a in kWh/(m3 K)."""
        return air_density(self.altitude_m) * C_A_KWH_KG_K


@dataclass(frozen=True)
class Stage:
    """Ventilation stage ``k`` in the interval (Table 8)."""
    q_sup_req_m3h: float                          # q_V;SUP;dis;req;k
    q_eta_req_m3h: float                          # q_V;ETA;dis;req;k
    hours: float                                  # t_V;k
    eta_hr: float = 0.0                           # eta_hr;k (EN 13141-7/-8, EN 13142)
    q_rca_m3h: float = 0.0                        # recirculation air flow q_V;rca;k
    theta_eta_c: Optional[float] = None           # mean extract air temperature (default theta_i)
    d_theta_preh_k: float = 0.0                   # preheating (ground HX, solar panel, defrost coil)
    d_theta_prec_k: float = 0.0                   # precooling (ground HX), <= 0
    hours_hr: Optional[float] = None              # t_V;hr;k (time without bypass), default t_V;k
    hours_preh: Optional[float] = None            # t_V;preh;k
    hours_prec: Optional[float] = None            # t_V;prec;k
    rotor_speed_rpm: Optional[float] = None       # n_rot;k (default maximum)


@dataclass(frozen=True)
class IntervalInputs:
    """Operating conditions of the interval (month, year or bin; Table 8)."""
    total_hours: float                            # t_total
    theta_e_c: float                              # mean external temperature
    theta_i_c: float                              # internal temperature
    stages: Sequence[Stage]
    theta_sur_c: Optional[float] = None           # unconditioned surroundings (default theta_e)
    q_h_v_in_kwh: Optional[float] = None          # AHU heating coil input (M3-1), (15)
    supply_temperature_setpoint_c: Optional[float] = None  # reheating requirement, (40)
    defrost_hours: float = 0.0                    # t_defrost (Table B.7)
    defrost_dt_k: float = 0.0                     # Delta T_defrost (Table B.8)
    q_w_hp_req_kwh: float = 0.0                   # Q_W;V;hp;out;req
    q_h_hp_req_kwh: float = 0.0                   # Q_H;V;hp;out;req
    q_c_hp_req_kwh: float = 0.0                   # Q_C;V;hp;out;req


# --------------------------------------------------------------------------- calculation
def _duct_drop(theta_in, theta_sur, h_kw_k, q_m3h, rho_c):
    if h_kw_k <= 0.0 or q_m3h <= 0.0:
        return 0.0
    return (theta_in - theta_sur) * (1.0 - math.exp(-h_kw_k / (rho_c * q_m3h)))


def _validate_interval(cfg: Unit52Config, inp: IntervalInputs) -> None:
    """Reject inputs the method cannot represent instead of losing them silently."""
    import dataclasses
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f"{f.name} must be finite (got {val})")
    if cfg.rho_c <= 0.0 or cfg.duct_leakage_factor <= 0.0 or cfg.ahu_leakage_factor <= 0.0:
        raise ValueError("rho_c and the leakage factors must be > 0")
    t_tot = float(inp.total_hours)
    if not math.isfinite(t_tot) or t_tot <= 0.0:
        raise ValueError("total_hours (t_total) must be finite and > 0")
    if not inp.stages:
        raise ValueError("at least one ventilation stage is required")
    t_v = 0.0
    for k, s in enumerate(inp.stages):
        if not math.isfinite(s.hours) or s.hours < 0.0:
            raise ValueError(f"stage {k}: hours (t_V;k) must be finite and >= 0")
        for nm in ("q_sup_req_m3h", "q_eta_req_m3h", "q_rca_m3h"):
            if not math.isfinite(getattr(s, nm)) or getattr(s, nm) < 0.0:
                raise ValueError(f"stage {k}: {nm} must be finite and >= 0")
        if not math.isfinite(s.eta_hr) or not 0.0 <= s.eta_hr <= 1.0:
            raise ValueError(f"stage {k}: eta_hr must be finite and in [0, 1] (got {s.eta_hr})")
        for nm in ("d_theta_preh_k", "d_theta_prec_k"):
            if not math.isfinite(getattr(s, nm)):
                raise ValueError(f"stage {k}: {nm} must be finite")
        if s.theta_eta_c is not None and not math.isfinite(s.theta_eta_c):
            raise ValueError(f"stage {k}: theta_eta_c must be finite")
        t_v += s.hours
    for nm in ("theta_e_c", "theta_i_c", "theta_sur_c", "q_h_v_in_kwh",
               "supply_temperature_setpoint_c"):
        val = getattr(inp, nm)
        if val is not None and not math.isfinite(val):
            raise ValueError(f"{nm} must be finite")
    if t_v > t_tot * (1.0 + 1e-9):
        raise ValueError(f"the stage operating times ({t_v:g} h) exceed t_total ({t_tot:g} h)")
    if not math.isfinite(inp.defrost_hours) or not math.isfinite(inp.defrost_dt_k):
        raise ValueError("defrost_hours and defrost_dt_k must be finite")
    if inp.defrost_hours < 0.0 or inp.defrost_hours > t_tot * (1.0 + 1e-9):
        raise ValueError("defrost_hours (t_defrost) must be between 0 and t_total")
    for nm in ("q_w_hp_req_kwh", "q_h_hp_req_kwh", "q_c_hp_req_kwh"):
        if not math.isfinite(getattr(inp, nm)) or getattr(inp, nm) < 0.0:
            raise ValueError(f"{nm} must be finite and >= 0")
    if inp.q_h_hp_req_kwh > 0.0 and inp.q_c_hp_req_kwh > 0.0:
        raise ValueError(
            "the method takes either space heating or space cooling in an interval "
            "(6.4.3.10.1.3 NOTE 1): q_h_hp_req_kwh and q_c_hp_req_kwh cannot both be > 0; "
            "use shorter intervals or bins")


def calculate_interval(cfg: Unit52Config, inp: IntervalInputs) -> dict:
    """EN 16798-5-2 for one calculation interval; returns the Table 5 outputs and the
    intermediate values (per stage under ``"stages"``)."""
    _validate_interval(cfg, inp)
    rc = cfg.rho_c
    f_du, f_ahu = cfg.duct_leakage_factor, cfg.ahu_leakage_factor
    t_tot = float(inp.total_hours)
    th_e = float(inp.theta_e_c)
    th_i = float(inp.theta_i_c)
    th_sur = th_e if inp.theta_sur_c is None else float(inp.theta_sur_c)
    h_eta_nc = cfg.h_du_sup_nc_kw_k if cfg.h_du_eta_nc_kw_k is None else cfg.h_du_eta_nc_kw_k  # (23f)
    hr_on = str(cfg.heat_recovery).lower() != "none"

    st_out = []
    t_v = sum(max(0.0, s.hours) for s in inp.stages)                                     # (8)
    q_sup_mean = q_eta_mean = 0.0
    q_ls_dis = 0.0
    q_preh = q_prec = q_hr = q_rca = q_h_req = 0.0
    f_oda_num = 0.0
    th_am_num = 0.0
    e_fan = 0.0
    w_hr = w_ctrl = 0.0
    for k, s in enumerate(inp.stages):
        t_k = max(0.0, float(s.hours))
        q_sup = s.q_sup_req_m3h * f_du * f_ahu                                         # (4)
        q_eta = s.q_eta_req_m3h * f_du * f_ahu                                         # (6)
        q_lea_sup = (f_du - 1.0) * q_sup                                               # (9)
        q_lea_eta = (f_du - 1.0) * q_eta                                               # (10)
        th_eta = th_i if s.theta_eta_c is None else float(s.theta_eta_c)
        # extract duct (20)-(21): leakage from the surroundings and transmission loss
        d_du_eta = _duct_drop(th_eta, th_sur, h_eta_nc, q_eta, rc)                     # (21a)
        sign = 1.0 if cfg.extract_duct_sign_as_printed else -1.0
        th_eta_out = th_eta / f_du + (f_du - 1.0) / f_du * th_sur + sign * d_du_eta    # (20)
        th_eta_hr_in = th_eta_out                                                      # (23c)
        # supply air treatment (11)-(15)
        th_preh = th_e + s.d_theta_preh_k                                              # (13)
        th_oda = th_preh + s.d_theta_prec_k
        t_hr = t_k if s.hours_hr is None else min(t_k, max(0.0, s.hours_hr))
        share_hr = t_hr / t_k if t_k > 0.0 else 0.0
        d_hr_full = ((s.eta_hr - (f_ahu - 1.0) - cfg.ahu_insulation_factor)
                     * (th_eta_out - th_preh)) if hr_on else 0.0                        # (12)
        d_hr = d_hr_full * share_hr          # mean over the stage time (bypass time excluded)
        th_sup_hr = th_oda + d_hr                                                      # (23b)
        q_out = q_sup                                                                  # q_V;SUP;RCA;k
        d_rca = (s.q_rca_m3h / q_out * (th_eta_out - th_sup_hr)) if q_out > 0.0 else 0.0  # (14)
        th_sup_rca = th_sup_hr + d_rca                                                 # (23a)
        # reheating: from the coil input (15), else from the supply set-point
        d_du_nc_guess = _duct_drop(th_sup_rca, th_sur, cfg.h_du_sup_nc_kw_k, q_sup, rc)
        if inp.q_h_v_in_kwh is not None and q_sup > 0.0 and t_tot > 0.0:
            d_rh = inp.q_h_v_in_kwh / (q_sup * rc * t_tot)                             # (15)
        elif inp.supply_temperature_setpoint_c is not None:
            d_rh = max(0.0, inp.supply_temperature_setpoint_c + d_du_nc_guess - th_sup_rca)
        else:
            d_rh = 0.0
        th_in = th_sup_rca + d_rh                                                      # (11)
        # supply duct (16)-(19), sections in sequence
        d_nc = _duct_drop(th_in, th_sur, cfg.h_du_sup_nc_kw_k, q_sup, rc)              # (18)
        d_cnd = _duct_drop(th_in - d_nc, th_i, cfg.h_du_sup_cnd_kw_k, q_sup, rc)       # (19)
        th_out = th_in - d_nc - d_cnd                                                  # (16)-(17)
        # outdoor air fraction (22)
        den = th_sup_hr - th_eta_hr_in
        f_oda_k = ((th_sup_rca - th_eta_hr_in) / den if abs(den) > 1e-9
                   else (1.0 - s.q_rca_m3h / q_out if q_out > 0.0 else 1.0))
        f_oda_k = max(0.0, min(1.0, f_oda_k))
        # energies
        q_sup_mean += q_sup * t_k / t_tot                                              # (5)
        q_eta_mean += q_eta * t_k / t_tot                                              # (7)
        f_oda_num += f_oda_k * t_k                                                     # (23g)
        q_ls_dis += rc * (q_sup * d_nc + q_lea_sup * (th_in - th_sur)) * t_k           # (25)
        q_base = q_sup / f_du
        t_preh = (inp.defrost_hours if s.hours_preh is None and inp.defrost_hours > 0.0
                  else (t_k if s.hours_preh is None and th_e < cfg.preheating_limit_c
                        else (s.hours_preh or 0.0)))
        t_prec = (t_k if s.hours_prec is None and th_e > cfg.precooling_limit_c
                  else (s.hours_prec or 0.0))
        q_preh += rc * f_oda_k * q_base * s.d_theta_preh_k * min(t_preh, t_k)          # (36)
        q_prec += rc * f_oda_k * q_base * s.d_theta_prec_k * min(t_prec, t_k)          # (37)
        q_hr += rc * f_oda_k * q_base * d_hr_full * t_hr                               # (38)
        q_rca += rc * q_base * d_rca * t_k                                             # (39)
        # (40): d_rh is a mean over the stage time (d_hr already includes the bypass share), so the
        # coil runs for the whole stage, also when the recovery is fully bypassed
        q_h_req += rc * q_base * d_rh * t_k
        th_am_num += 0.5 * ((th_e + s.d_theta_preh_k) + max(th_in, th_eta_out)) * t_k  # (29)
        # fans (31a): SPI per stage
        if cfg.spi_kw_per_m3h is not None:
            spi = float(cfg.spi_kw_per_m3h[min(k, len(cfg.spi_kw_per_m3h) - 1)])
            e_fan += t_k * q_sup * (spi + 2.78e-7 * cfg.dp_defrost_pa / cfg.fan_efficiency)
        # auxiliary (33), (35)
        if str(cfg.heat_recovery).lower() == "rotary":
            n = cfg.rotor_speed_max_rpm if s.rotor_speed_rpm is None else s.rotor_speed_rpm
            w_hr += t_k * cfg.rotor_power_max_kw * n / cfg.rotor_speed_max_rpm
        w_ctrl += t_k * cfg.controls_power_kw
        st_out.append({
            "q_sup_dis_m3h": q_sup, "q_eta_dis_m3h": q_eta, "q_lea_sup_m3h": q_lea_sup,
            "q_lea_eta_m3h": q_lea_eta, "theta_eta_dis_out_c": th_eta_out,
            "d_theta_hr_k": d_hr, "d_theta_rca_k": d_rca, "d_theta_rh_k": d_rh,
            "theta_sup_dis_in_c": th_in, "theta_sup_dis_out_c": th_out, "f_oda": f_oda_k,
        })

    if cfg.spi_kw_per_m3h is None and cfg.spi_single_kw_per_m3h is not None:
        q_max = cfg.q_sup_max_m3h or max((s.q_sup_req_m3h for s in inp.stages), default=0.0) * f_du * f_ahu
        e_fan = sum(max(0.0, s.hours) for s in inp.stages) * (
            cfg.spi_single_kw_per_m3h + 2.78e-7 * cfg.dp_defrost_pa / cfg.fan_efficiency
        ) * q_max * cfg.flow_control_factor ** cfg.motor_exponent                      # (31b)

    f_oda = f_oda_num / t_tot if t_tot > 0.0 else 0.0                                  # (23g)
    w_preh = (rc * q_sup_mean / f_du * f_oda * inp.defrost_hours * inp.defrost_dt_k
              * cfg.defrost_control_factor)                                            # (34)
    w_aux = w_hr + w_preh + w_ctrl                                                     # (32)
    th_am = th_am_num / t_v if t_v > 0.0 else th_i                                     # (29)
    q_ls_ahu = (0.0 if cfg.ahu_in_conditioned_zone
                else cfg.ahu_au_kw_k * (th_am - th_sur) * t_v)                         # (28)

    hp = _heat_pump(cfg, inp, q_sup_mean)
    q_ls_gen = q_ls_ahu
    if cfg.heat_pump is not None:
        q_ls_gen += (cfg.heat_pump.f_gen_hp * (hp["Q_W_gen_hp_out"] + hp["Q_H_gen_hp_out"])
                     + cfg.heat_pump.f_gen_cu * hp["Q_C_gen_hp_in"])                   # (27)
    q_ls_gen_rbl = q_ls_gen if cfg.ahu_in_conditioned_zone else 0.0                    # (30)

    return {
        "q_V_SUP_dis_m3h": q_sup_mean, "q_V_ETA_dis_m3h": q_eta_mean,
        "q_V_lea_SUP_dis_m3h": (f_du - 1.0) * q_sup_mean,
        "q_V_lea_ETA_dis_m3h": (f_du - 1.0) * q_eta_mean,
        "theta_SUP_dis_out_c": [s["theta_sup_dis_out_c"] for s in st_out],
        "f_ODA": f_oda, "t_V_h": t_v,
        "E_V_gen_fan_kwh": e_fan, "W_V_aux_kwh": w_aux, "W_V_aux_hr_kwh": w_hr,
        "W_V_aux_preh_kwh": w_preh, "W_V_aux_ctrl_kwh": w_ctrl,
        "Q_ls_V_dis_kwh": q_ls_dis, "Q_ls_V_dis_rbl_kwh": 0.0,                          # (26)
        "Q_ls_V_ahu_kwh": q_ls_ahu, "Q_ls_V_gen_kwh": q_ls_gen, "Q_ls_V_gen_rbl_kwh": q_ls_gen_rbl,
        "Q_preh_kwh": q_preh, "Q_prec_kwh": q_prec, "Q_hr_kwh": q_hr, "Q_rca_kwh": q_rca,
        "Q_H_V_in_req_kwh": q_h_req, "theta_a_m_c": th_am,
        **hp, "stages": st_out,
    }


def _heat_pump(cfg: Unit52Config, inp: IntervalInputs, q_sup_mean: float) -> dict:
    """Extract air heat pump (41)-(58)."""
    out = {k: 0.0 for k in ("Q_W_gen_hp_out", "Q_W_residual", "Q_H_gen_hp_out", "Q_H_residual",
                            "Q_C_gen_hp_in", "Q_C_residual", "E_W_hp_in", "E_H_hp_in",
                            "E_C_hp_in")}
    hp = cfg.heat_pump
    # demand with no heat pump service to cover it stays in the residual (never lost)
    services = {"W": (hp is not None and hp.p_W_kw_per_m3h is not None, inp.q_w_hp_req_kwh),
                "H": (hp is not None and hp.p_H_kw_per_m3h is not None, inp.q_h_hp_req_kwh),
                "C": (hp is not None and hp.p_C_kw_per_m3h is not None, inp.q_c_hp_req_kwh)}
    for key, (configured, req) in services.items():
        if req > 0.0 and not configured:
            out[f"Q_{key}_residual"] = req
            warnings.warn(
                f"EN 16798-5-2: {req:g} kWh of {'domestic hot water' if key == 'W' else 'space heating' if key == 'H' else 'space cooling'}"
                " requested but the heat pump has no such service configured: reported as residual",
                stacklevel=3)
    if hp is None:
        return out
    f_du, f_ahu = cfg.duct_leakage_factor, cfg.ahu_leakage_factor
    stages = list(inp.stages)
    q_eta = [s.q_eta_req_m3h * f_du * f_ahu for s in stages]
    t_vk = [max(0.0, s.hours) for s in stages]
    th_eta = [inp.theta_i_c if s.theta_eta_c is None else s.theta_eta_c for s in stages]
    t_tot = float(inp.total_hours)
    q_max = cfg.q_sup_max_m3h or max((s.q_sup_req_m3h for s in stages), default=0.0) * f_du * f_ahu
    q_mean = sum(qe * t for qe, t in zip(q_eta, t_vk)) / t_tot if t_tot > 0.0 else 0.0
    f_v = correction_flow(q_mean / q_max) if q_max > 0.0 else 1.0                      # Table B.16
    t_on_w = [0.0] * len(stages)
    # domestic hot water (41)-(45)
    if hp.p_W_kw_per_m3h is not None:
        q_w_max = sum(hp.p_W_kw_per_m3h * qe * hp.COP_W * t * correction_extract_temperature(te)
                      for qe, t, te in zip(q_eta, t_vk, th_eta)) * f_v                  # (41)
        req = inp.q_w_hp_req_kwh
        if req > q_w_max:
            out["Q_W_gen_hp_out"], out["Q_W_residual"] = q_w_max, req - q_w_max        # (42a), (43a)
            t_on_w = list(t_vk)                                                        # (44a), (45a)
        else:
            out["Q_W_gen_hp_out"] = req                                                # (42b), (43b)
            ratio = req / q_w_max if q_w_max > 0.0 else 0.0
            t_on_w = [ratio * t for t in t_vk]                                         # (45b)
        out["E_W_hp_in"] = sum(hp.p_W_kw_per_m3h * qe * t for qe, t in zip(q_eta, t_on_w))  # (56)
    t_rest = [t - tw for t, tw in zip(t_vk, t_on_w)]
    # space heating (46)-(50) - either heating or cooling in the interval (NOTE 1)
    if hp.p_H_kw_per_m3h is not None and inp.q_h_hp_req_kwh > 0.0:
        q_h_max = sum(hp.p_H_kw_per_m3h * qe * hp.COP_H * t * correction_extract_temperature(te)
                      for qe, t, te in zip(q_eta, t_rest, th_eta)) * hp.f_dtheta_h * f_v \
            - (out["Q_W_gen_hp_out"] if hp.deduct_dhw_energy_46 else 0.0)              # (46)
        q_h_max = max(0.0, q_h_max)
        req = inp.q_h_hp_req_kwh
        if req > q_h_max:
            out["Q_H_gen_hp_out"], out["Q_H_residual"] = q_h_max, req - q_h_max        # (47a), (48a)
            t_on_h = t_rest                                                            # (50a)
        else:
            out["Q_H_gen_hp_out"] = req                                                # (47b)
            ratio = req / q_h_max if q_h_max > 0.0 else 0.0
            t_on_h = [ratio * t for t in t_rest]                                       # (50b)
        out["E_H_hp_in"] = sum(hp.p_H_kw_per_m3h * qe * t for qe, t in zip(q_eta, t_on_h))  # (57)
    elif hp.p_C_kw_per_m3h is not None and inp.q_c_hp_req_kwh > 0.0:
        # space cooling (51)-(55)
        q_c_max = sum(hp.p_C_kw_per_m3h * qe * hp.EER * t for qe, t in zip(q_eta, t_rest))  # (51)
        req = inp.q_c_hp_req_kwh
        if req > q_c_max:
            out["Q_C_gen_hp_in"], out["Q_C_residual"] = q_c_max, req - q_c_max        # (52a), (53a)
            t_on_c = t_rest                                                            # (55a)
        else:
            out["Q_C_gen_hp_in"] = req                                                 # (52b)
            ratio = req / q_c_max if q_c_max > 0.0 else 0.0
            t_on_c = [ratio * t for t in t_rest]                                       # (55b)
        out["E_C_hp_in"] = sum(hp.p_C_kw_per_m3h * qe * t for qe, t in zip(q_eta, t_on_c))  # (58)
    return out


def combine_by_area(results: Sequence[dict], areas_m2: Sequence[float], floor_area_m2: float) -> dict:
    """(24): energies of several ventilation systems of one zone weighted by area."""
    keys = [k for k, v in results[0].items() if isinstance(v, (int, float)) and k.endswith("_kwh")]
    return {k: sum(r[k] * a / floor_area_m2 for r, a in zip(results, areas_m2)) for k in keys}


def defrost_defaults(climate: str = "average") -> tuple:
    """Tables B.7 / B.8 (defrost set-point -4 degC): (t_defrost [h/a], Delta T_defrost [K])."""
    c = str(climate).lower()
    if c not in TABLE_B7_DEFROST_HOURS:
        raise ValueError("climate must be 'cold', 'average' or 'warm'")
    return TABLE_B7_DEFROST_HOURS[c], TABLE_B8_DEFROST_DT[c]


def monthly_from_hourly(hourly, cfg: Unit52Config, stage_template: Stage,
                        defrost_climate: Optional[str] = None,
                        heating_coil_col: Optional[str] = None) -> list:
    """Monthly intervals from an hourly result (columns ``T_ext`` and ``T_air``, e.g. the
    single-zone EN ISO 52016-1 output) and :func:`calculate_interval` for each month.

    ``stage_template`` gives the flows and ratios; its ``hours`` is the daily operating
    time (24 for continuous operation) and is scaled to the days of each month.  The
    annual defrost hours of Table B.7 are shared among the months in proportion to the
    hours below the defrost set-point (-4 degC), per calendar year (scaled to the data length
    when the year is incomplete).  Months of different years are separate intervals.
    Returns a list of ``((year, month), result)``.
    """
    import pandas as pd

    df = hourly.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("hourly must have a DatetimeIndex")
    for col in ("T_ext", "T_air"):
        if col not in df.columns:
            raise ValueError(f"hourly needs the column {col!r}")
    if df.empty:
        raise ValueError("hourly is empty")
    t_ext, t_air = df["T_ext"], df["T_air"]
    # one interval per (year, month): the same month of different years is not mixed
    year, month = df.index.year, df.index.month
    periods = sorted(set(zip(year, month)))
    frost_h = {p: float(((t_ext < -4.0) & (year == p[0]) & (month == p[1])).sum()) for p in periods}
    hours_y = {y: float((year == y).sum()) for y in set(year)}
    frost_y = {y: sum(h for (yy, _), h in frost_h.items() if yy == y) for y in hours_y}
    t_def_a, dt_def = defrost_defaults(defrost_climate) if defrost_climate else (0.0, 0.0)
    out = []
    for p in periods:
        sel = (year == p[0]) & (month == p[1])
        hours = float(sel.sum())
        days = hours / 24.0
        st = Stage(**{**stage_template.__dict__, "hours": min(hours, stage_template.hours * days)})
        q_h_in = (float(df.loc[sel, heating_coil_col].sum()) / 1000.0
                  if heating_coil_col and heating_coil_col in df.columns else None)
        # annual defrost hours (Table B.7) scaled to the data length of the year and shared
        # by the frost hours; never more than the operating time of the interval
        t_def_year = t_def_a * min(1.0, hours_y[p[0]] / 8760.0)
        t_def = (t_def_year * frost_h[p] / frost_y[p[0]] if frost_y[p[0]] > 0.0 else 0.0)
        t_def = min(t_def, st.hours)
        res = calculate_interval(cfg, IntervalInputs(
            total_hours=hours, theta_e_c=float(t_ext[sel].mean()),
            theta_i_c=float(t_air[sel].mean()), stages=[st], q_h_v_in_kwh=q_h_in,
            defrost_hours=t_def, defrost_dt_k=dt_def))
        out.append((p, res))
    return out


__all__ = [
    "Unit52Config", "Stage", "IntervalInputs", "HeatPumpData", "calculate_interval",
    "combine_by_area", "defrost_defaults", "monthly_from_hourly", "insulation_factor",
    "correction_extract_temperature", "correction_flow", "air_density",
]
