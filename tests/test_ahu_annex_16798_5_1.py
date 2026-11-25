"""EN 16798-5-1: actual coil states (61)-(69), fan control (72)-(75), AHU casing losses
(85)-(86), auxiliary energy (87)-(93), ground preheating (Annex C), supply temperature
control NO_CTRL / ODA_COMP / LOAD_COMP and air density (Table 27).

Reference values from the EPB Center spreadsheet Demo_EN_16798-5-1 (2026-05-12), sheet
"Method_calculation" (summer hour: theta_e 21.8 degC, ROT_NH, f_ODA;min 400/1350).
"""

import math

import pytest

from pybuildingenergy.source import ahu_moisture_16798_5_1 as mx
from pybuildingenergy.source.central_ahu_16798_5_1 import CentralAHU
from pybuildingenergy.source.ventilation_16798_5_1 import (
    AHUStepInputs,
    _fan_power_and_temperature_rise,
    calculate_sensible_ahu_step,
    sensible_ahu_config_from_dict,
)

# --------------------------------------------------------------------------- reference sheet
Q_SUP_AHU = 567.5318518518519          # (25) of the reference hour
F_ODA_MIN = 400.0 / 1350.0             # (35b)


def test_reference_sheet_flows_and_heat_recovery_ratio():
    assert 1.04 * ((1 - F_ODA_MIN) + F_ODA_MIN * 1.1) * 1.06 * 500 == pytest.approx(Q_SUP_AHU)
    hr = mx.HeatRecoveryData(hr_type="ROT_NH", eta_hr_nom=0.75, v_hr_des_m_s=3.5,
                             q_sup_hr_nom_m3_h=1350, C1=-0.0643, C2=1.0, eta_xr_nom=0.3)
    r = mx.temperature_ratio(hr, Q_SUP_AHU, Q_SUP_AHU, f_oda_min=F_ODA_MIN)
    assert r["v_hr_eff"] == pytest.approx(0.435964, abs=1e-6)      # (43)
    assert r["f_v"] == pytest.approx(1.19702, abs=1e-5)            # (42)
    assert r["eta_hr"] == pytest.approx(0.897763, abs=1e-6)        # (40)
    m = mx.moisture_recovery_efficiency(hr, 0.01369, 21.8, 0.0115, Q_SUP_AHU, Q_SUP_AHU,
                                        f_oda_min=F_ODA_MIN)
    assert mx.saturation_humidity_ratio(21.8) == pytest.approx(0.0164160249504918, rel=1e-9)  # (D.6)
    assert m["f_dx"] == 0.0                                        # (D.5a)


def test_reference_sheet_dehumidification_requirement_56_to_60():
    req = mx.cooling_and_dehumidification(21.8, 0.0130411, 21.8, 0.011, mx.CoilData())
    assert req["theta_C_coil"] == pytest.approx(15.036, abs=1e-3)          # (59)
    assert req["theta_SUP_C"] == pytest.approx(15.7124, abs=1e-4)          # (60a)
    assert req["x_SUP_C"] == pytest.approx(0.011)                          # (54)


def _ref_coil_step(**kw):
    """The reference hour from the mixed air on: theta_RCA 21.8 degC, x_RCA 0.0130411,
    NO_CTRL, coil medium at 14.4606 degC, 1.263 kWh cooling and 0.3415 kWh heating."""
    d = {"sensible_heat_recovery_efficiency": 0.0, "supply_temperature_mode": "none",
         "cooling_coil_enabled": True, "supply_humidity_max_kg_kg": 0.011, "frost_control": "none",
         "heat_recovery_control": "always_on", "altitude_m": 400.0}
    d.update(kw)
    return calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(
        outdoor_temperature_c=21.8, extract_temperature_c=21.8,
        outdoor_humidity_ratio=0.0130411, extract_humidity_ratio=0.0130411,
        required_supply_flow_m3_h=Q_SUP_AHU, required_extract_flow_m3_h=Q_SUP_AHU,
        cooling_coil_medium_temperature_c=14.460593388997411,
        cooling_coil_output_w=1263.024683258067, heating_coil_input_w=341.5318641178524))


def test_reference_sheet_actual_coil_states_62_to_64_and_69():
    o = _ref_coil_step()
    assert o.cooling_coil_temperature_c == pytest.approx(15.2761, abs=1e-4)   # (64)
    assert o.supply_humidity_ratio == pytest.approx(0.0111541, abs=1e-7)      # (63)
    # (62), (69): differences with the sheet only from rho (1.1591 vs 1.1584) and c_a
    # (1006 vs 1004.4 J/(kg K)) -> within 0.02 K
    assert o.actual_cooling_coil_power_w == pytest.approx(1263.024683258067)
    theta_sup_c = o.actual_supply_temperature_c - 341.5318641178524 / (
        o.air_density_kg_m3 * 1006.0 * Q_SUP_AHU / 3600.0)
    assert theta_sup_c == pytest.approx(19.513, abs=0.02)
    assert o.actual_supply_temperature_c == pytest.approx(21.3751, abs=0.03)
    assert o.air_density_kg_m3 == pytest.approx(1.158356, abs=1e-3)           # Table 27, 400 m


def test_coil_medium_temperature_required_61_and_68():
    o = calculate_sensible_ahu_step(
        sensible_ahu_config_from_dict({"sensible_heat_recovery_efficiency": 0.0,
                                       "supply_temperature_setpoint_c": 30.0, "frost_control": "none"}),
        AHUStepInputs(outdoor_temperature_c=10.0, extract_temperature_c=20.0,
                      required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    # (68): theta_H;ahu;in;req = (30 - 10 (1 - 0.8)) / 0.8
    assert o.heating_coil_medium_temperature_required_c == pytest.approx((30.0 - 10.0 * 0.2) / 0.8)
    assert o.cooling_coil_medium_temperature_required_c is None
    c = calculate_sensible_ahu_step(
        sensible_ahu_config_from_dict({"sensible_heat_recovery_efficiency": 0.0, "cooling_coil_enabled": True,
                                       "supply_temperature_setpoint_c": 16.0, "frost_control": "none"}),
        AHUStepInputs(outdoor_temperature_c=30.0, extract_temperature_c=26.0,
                      required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    assert c.cooling_coil_medium_temperature_required_c == pytest.approx((16.0 - 30.0 * 0.2) / 0.8)  # (61)


def test_dehumidification_requirement_81():
    o = _ref_coil_step(supply_temperature_mode="fixed", supply_temperature_setpoint_c=21.8)
    assert o.dehumidification_power_required_w > 0.0
    assert o.dehumidification_power_required_w <= o.required_cooling_coil_power_w + 1e-9


# --------------------------------------------------------------------------- supply temperature
def _simple(extra=None, **kw):
    d = {"sensible_heat_recovery_efficiency": 0.0, "frost_control": "none", "cooling_coil_enabled": True}
    d.update(extra or {})
    k = dict(outdoor_temperature_c=21.8, extract_temperature_c=24.0,
             required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0)
    k.update(kw)
    return calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(**k))


def test_oda_compensated_supply_temperature_29b():
    o = _simple({"supply_temperature_mode": "oda_compensated", "supply_temperature_slope": -0.44,
                 "supply_temperature_offset_c": 27.5, "supply_temperature_minimum_c": 16.5,
                 "supply_temperature_maximum_c": 31.5})
    assert o.requested_supply_temperature_c == pytest.approx(-0.44 * 21.8 + 27.5)
    o = _simple({"supply_temperature_mode": "oda_compensated", "supply_temperature_slope": -0.44,
                 "supply_temperature_offset_c": 27.5, "supply_temperature_minimum_c": 16.5,
                 "supply_temperature_maximum_c": 31.5}, outdoor_temperature_c=-15.0)
    assert o.requested_supply_temperature_c == pytest.approx(31.5)


def test_load_compensated_supply_temperature_29c():
    o = _simple({"supply_temperature_mode": "load_compensated"}, required_supply_temperature_c=17.0)
    assert o.requested_supply_temperature_c == 17.0
    assert o.actual_supply_temperature_c == pytest.approx(17.0)
    with pytest.raises(ValueError, match="requires"):
        _simple({"supply_temperature_mode": "load_compensated"})


def test_no_supply_temperature_control_53a_67a():
    o = _simple({"supply_temperature_mode": "none"}, outdoor_temperature_c=5.0)
    assert o.requested_supply_temperature_c is None
    assert o.actual_heating_coil_power_w == 0.0 and o.actual_cooling_coil_power_w == 0.0
    assert o.actual_supply_temperature_c == pytest.approx(5.0)


# --------------------------------------------------------------------------- fans (72)-(75)
def _fan(control, **kw):
    args = dict(model=sensible_ahu_config_from_dict({"sensible_heat_recovery_efficiency": 0.0,
                                                     "supply_temperature_setpoint_c": 18.0}).fan_performance_model,
                operating_flow_m3_h=500.0, design_flow_m3_h=1000.0, operation_fraction=1.0,
                nominal_specific_power_w_per_m3_s=0.0, design_pressure_pa=500.0,
                nominal_efficiency=0.6, reference_flow_m3_h=None, reference_efficiency=None,
                heat_fraction_to_air=1.0, fan_control=control, efficiency_curve=(0.0, 0.0, 1.0))
    args.update(kw)
    return _fan_power_and_temperature_rise(**args)[0]


def test_fan_pressure_by_fan_control_74():
    q = 500.0 / 3600.0
    assert _fan("direct") == pytest.approx(q * 500.0 * 0.25 / 0.6)                          # (74b)
    assert _fan("const_pres", pressure_control_fraction=0.4) == pytest.approx(
        q * 500.0 * (0.6 * 0.25 + 0.4) / 0.6)                                                # (74d)
    assert _fan("min_pres", pressure_control_fraction=0.4, max_zone_flow_ratio=0.7) == pytest.approx(
        q * 500.0 * (0.6 * 0.25 + 0.4 * 0.49) / 0.6)                                         # (74e)
    assert _fan("no_ctrl", pressure_curve=(-0.5, 0.0, 1.5)) == pytest.approx(
        q * 500.0 * (1.5 - 0.5 * 0.25) / 0.6)                                                # (74a)
    # (73): efficiency from the curve
    assert _fan("direct", efficiency_curve=(0.5, 0.5)) == pytest.approx(q * 500.0 * 0.25 / (0.6 * 0.75))


def test_no_ctrl_fan_requires_the_curves():
    with pytest.raises(ValueError, match="pressure_curve"):
        sensible_ahu_config_from_dict({"sensible_heat_recovery_efficiency": 0.0,
                                       "supply_temperature_setpoint_c": 18.0, "fan_control": "no_ctrl"})


# --------------------------------------------------------------------------- casing (85)-(86)
def test_ahu_casing_losses_85_86():
    base = {"sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 18.0,
            "frost_control": "none", "ahu_casing_au_w_k": 17.5, "duct_surrounding_temperature_c": 5.0}
    nc = calculate_sensible_ahu_step(sensible_ahu_config_from_dict(base), AHUStepInputs(
        outdoor_temperature_c=0.0, extract_temperature_c=22.0,
        required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    # no recovery: theta_SUP;hr = 0, theta_ETA;hr;in = 22
    assert nc.ahu_heat_loss_w == pytest.approx(17.5 * (0.0 - 5.0) + 17.5 * (22.0 - 5.0))   # (85b)
    assert nc.ahu_heat_loss_recoverable_w == 0.0                                           # (86b)
    cond = calculate_sensible_ahu_step(sensible_ahu_config_from_dict({**base, "ahu_location": "conditioned"}),
                                       AHUStepInputs(outdoor_temperature_c=0.0, extract_temperature_c=22.0,
                                                     required_supply_flow_m3_h=1000.0,
                                                     required_extract_flow_m3_h=1000.0))
    assert cond.ahu_heat_loss_w == pytest.approx(17.5 * (0.0 - 22.0))                     # (85a)
    assert cond.ahu_heat_loss_recoverable_w == pytest.approx(cond.ahu_heat_loss_w)         # (86a)


# --------------------------------------------------------------------------- auxiliary (87)-(93)
def test_auxiliary_energy_rotor_controls_and_fan_share():
    d = {"sensible_heat_recovery_efficiency": 0.7, "supply_temperature_setpoint_c": 18.0,
         "frost_control": "none", "heat_recovery_type": "ROT_NH", "heat_recovery_control": "always_on",
         "controls_power_w": 100.0, "supply_fan_design_pressure_pa": 500.0,
         "extract_fan_design_pressure_pa": 400.0, "heat_recovery_design_pressure_drop_pa": 400.0,
         "supply_fan_specific_power_w_per_m3_s": 1000.0, "extract_fan_specific_power_w_per_m3_s": 800.0}
    o = calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(
        outdoor_temperature_c=0.0, extract_temperature_c=22.0,
        required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    assert o.rotor_speed_rpm == pytest.approx(20.0)                       # no control: full speed
    assert o.auxiliary_heat_recovery_w == pytest.approx(120.0)            # (88a)
    assert o.auxiliary_controls_w == pytest.approx(100.0)                 # (91)
    assert o.auxiliary_power_w == pytest.approx(220.0)                    # (87)
    assert o.fan_power_heat_recovery_w == pytest.approx(o.fan_electric_power_w * 400.0 / 900.0)  # (89)
    # modulating control: the rotor slows down when the recovery is bypassed (D.2)-(D.4)
    m = calculate_sensible_ahu_step(sensible_ahu_config_from_dict({**d, "heat_recovery_control": "modulating_bypass"}),
                                    AHUStepInputs(outdoor_temperature_c=14.0, extract_temperature_c=22.0,
                                                  required_supply_flow_m3_h=1000.0,
                                                  required_extract_flow_m3_h=1000.0))
    assert m.rotor_speed_rpm < 20.0
    assert m.auxiliary_heat_recovery_w < 120.0


def test_auxiliary_energy_pump_88b_and_humidifier_92_93():
    p = calculate_sensible_ahu_step(sensible_ahu_config_from_dict({
        "sensible_heat_recovery_efficiency": 0.6, "supply_temperature_setpoint_c": 30.0,
        "frost_control": "none", "heat_recovery_type": "PUMP_CIRC", "heat_recovery_control": "always_on",
        "heat_recovery_power_max_w": 20000.0}), AHUStepInputs(
        outdoor_temperature_c=0.0, extract_temperature_c=22.0,
        required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    load = max(0.5, p.heat_recovery_power_w / 20000.0)
    assert p.auxiliary_heat_recovery_w == pytest.approx(1000.0 * 0.03 * load ** 2.5)
    h = calculate_sensible_ahu_step(sensible_ahu_config_from_dict({
        "sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 22.0,
        "frost_control": "none", "supply_humidity_min_kg_kg": 0.006, "humidifier_type": "ADIABATIC",
        "humidifier_pump_specific_power_wh_m3": 0.2, "humidifier_design_water_kg_h": 10.0,
        "humidifier_control": "on_off"}), AHUStepInputs(
        outdoor_temperature_c=0.0, extract_temperature_c=22.0, outdoor_humidity_ratio=0.002,
        extract_humidity_ratio=0.006, required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    f_pl = min(1.0, h.humidification_water_kg_h / 10.0)                    # (93b)
    assert h.humidifier_auxiliary_w == pytest.approx(1000.0 * 0.2 * f_pl)  # (92b)


# --------------------------------------------------------------------------- Annex C
def test_ground_temperature_c2_to_c4():
    g = mx.GroundExchangerData(area_m2=40.0, mean_annual_temperature_c=10.0,
                               max_monthly_mean_temperature_c=20.0, depth_m=2.0, soil="moist_clay")
    lam, rho, c = mx.TABLE_C2["moist_clay"]
    xi = 2.0 * math.sqrt(math.pi * rho * c / (lam * 8760 * 3600))
    t = 3000.0
    assert mx.ground_temperature(g, t) == pytest.approx(
        10.0 + 10.0 * math.exp(-xi) * math.cos(2 * math.pi * t / 8760 - xi - 1.08 * math.pi))


def test_ground_preheating_in_the_ahu_step_32_33_76():
    ge = {"area_m2": 60.0, "mean_annual_temperature_c": 10.0, "max_monthly_mean_temperature_c": 20.0,
          "u_w_m2_k": 5.0}
    d = {"sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 20.0,
         "frost_control": "none", "ground_exchanger": ge}
    o = calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(
        outdoor_temperature_c=-5.0, extract_temperature_c=21.0, hour_of_year=350.0,
        required_supply_flow_m3_h=500.0, required_extract_flow_m3_h=500.0))
    g = mx.ground_preheating(mx.GroundExchangerData(**ge), -5.0, None, 350.0, 500.0, 1.204 * 1006.0)
    assert g["d_theta"] > 0.0
    assert o.ground_heat_transfer_w == pytest.approx(1.204 * 1006.0 * 500.0 / 3600.0 * g["d_theta"])  # (76)
    with pytest.raises(ValueError, match="hour_of_year"):
        calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(
            outdoor_temperature_c=-5.0, extract_temperature_c=21.0,
            required_supply_flow_m3_h=500.0, required_extract_flow_m3_h=500.0))


def test_outdoor_air_duct_and_altitude():
    d = {"sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 20.0,
         "frost_control": "none", "duct_outdoor_h_w_k": 20.0, "duct_surrounding_temperature_c": 15.0}
    o = calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(
        outdoor_temperature_c=0.0, extract_temperature_c=21.0,
        required_supply_flow_m3_h=500.0, required_extract_flow_m3_h=500.0))
    base = calculate_sensible_ahu_step(sensible_ahu_config_from_dict({**d, "duct_outdoor_h_w_k": 0.0}),
                                       AHUStepInputs(outdoor_temperature_c=0.0, extract_temperature_c=21.0,
                                                     required_supply_flow_m3_h=500.0,
                                                     required_extract_flow_m3_h=500.0))
    assert o.actual_heating_coil_power_w < base.actual_heating_coil_power_w
    hi = calculate_sensible_ahu_step(sensible_ahu_config_from_dict({**d, "altitude_m": 1500.0}),
                                     AHUStepInputs(outdoor_temperature_c=0.0, extract_temperature_c=21.0,
                                                   required_supply_flow_m3_h=500.0,
                                                   required_extract_flow_m3_h=500.0))
    assert hi.air_density_kg_m3 == pytest.approx(1.204 * (1 - 0.00651 * 1500 / 293) ** 4.255)
    assert hi.air_density_kg_m3 < 1.204


# --------------------------------------------------------------------------- central AHU
def test_central_ahu_fan_power_follows_the_vav_flow():
    cfg = {"name": "c", "sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 14.0,
           "frost_control": "none", "supply_fan_specific_power_w_per_m3_s": 1000.0,
           "extract_fan_specific_power_w_per_m3_s": 800.0, "min_flow_ratio": 0.2,
           "zones": [{"zone": "a", "supply_flow_max_m3_h": 1000.0, "supply_flow_min_m3_h": 200.0}]}
    c = CentralAHU(cfg, ["a"])
    full, _ = c.step(12, 30.0, {"a": 26.0}, {})
    half_c = CentralAHU(cfg, ["a"])
    half_c._t_src_prev = 14.0
    # previous step: the zone needed 2 kW of heating on top of the VAV cooling -> less flow
    half, z = half_c.step(12, 30.0, {"a": 26.0}, {"a": 2000.0})
    ratio = z["a"]["q"] / 1000.0
    assert ratio < 1.0
    # EN 16798-5-1 direct control: dp ~ x^2, eta ~ sqrt(x) -> P ~ x^2.5
    assert half.fan_electric_power_w == pytest.approx(full.fan_electric_power_w * ratio ** 2.5, rel=1e-6)


def test_central_load_compensated_supply_temperature():
    cfg = {"name": "c", "sensible_heat_recovery_efficiency": 0.0, "supply_temperature_mode": "load_compensated",
           "supply_temperature_minimum_c": 14.0, "supply_temperature_maximum_c": 35.0,
           "cooling_coil_enabled": True, "frost_control": "none",
           "zones": [{"zone": "a", "supply_flow_max_m3_h": 1000.0, "vav": False}]}
    c = CentralAHU(cfg, ["a"])
    c.step(12, 30.0, {"a": 26.0}, {})                         # first step: neutral (zone air)
    out, _ = c.step(13, 30.0, {"a": 26.0}, {"a": -2000.0})    # previous step: 2 kW cooling still needed
    rc = c._rho_cp * 1000.0 / 3600.0
    t_req = 26.0 + (-2000.0 + rc * (26.0 - 26.0)) / rc
    assert out.requested_supply_temperature_c == pytest.approx(max(14.0, t_req))
    with pytest.raises(ValueError, match="VAV"):
        CentralAHU({**cfg, "zones": [{"zone": "a", "supply_flow_max_m3_h": 1000.0}]}, ["a"])


# --------------------------------------------------------------------------- Annex D amended
def test_annex_d_amended_matches_the_reference_sheet():
    hr = mx.HeatRecoveryData(hr_type="ROT_NH", eta_hr_nom=0.75, v_hr_des_m_s=3.5,
                             q_sup_hr_nom_m3_h=1350, C1=-0.0643, eta_xr_nom=0.3)
    m = mx.moisture_recovery_efficiency(hr, 0.01369, 21.8, 0.0115, Q_SUP_AHU, Q_SUP_AHU,
                                        f_oda_min=F_ODA_MIN)
    assert m["f_q_x"] == pytest.approx(1.0)                         # (D.8), sheet: 1
    assert m["f_v_x"] == pytest.approx(2.3139738444342832)          # (D.9), sheet value
    old = mx.HeatRecoveryData(hr_type="ROT_NH", eta_hr_nom=0.75, v_hr_des_m_s=3.5,
                              q_sup_hr_nom_m3_h=1350, C1=-0.0643, eta_xr_nom=0.3,
                              annex_d_version="2017")
    m17 = mx.moisture_recovery_efficiency(old, 0.01369, 21.8, 0.0115, Q_SUP_AHU, Q_SUP_AHU,
                                          f_oda_min=F_ODA_MIN)
    assert m17["f_q_x"] == pytest.approx(0.9) and m17["f_v_x"] == pytest.approx(1.6128071777676167)


def test_annex_d_amended_d5b_rotary_hygroscopic():
    hr = mx.HeatRecoveryData(hr_type="ROT_HYG")
    t_e, x_eta = 0.0, 0.009
    dxn = x_eta - mx.saturation_humidity_ratio(t_e) - 0.005
    m = mx.moisture_recovery_efficiency(hr, x_eta, t_e, 0.003, 1000.0, 1000.0)
    assert m["f_dx"] == pytest.approx(max(0.0, 1.0 + 129.0 * dxn, 0.476 + 23.8 * dxn))
    with pytest.raises(ValueError, match="annex_d_version"):
        mx.HeatRecoveryData(hr_type="ROT_HYG", annex_d_version="1999")


# --------------------------------------------------------------------------- national data
def test_italian_national_annex_defaults():
    base = {"sensible_heat_recovery_efficiency": 0.7, "supply_temperature_setpoint_c": 18.0}
    c = sensible_ahu_config_from_dict({**base, "heat_recovery_type": "ROT_HYG",
                                       "building_status": "existing", "fan_motor_location": "OUTS_AIR",
                                       "system_type": "air_conditioning", "humidifier_type": "HI_PRES",
                                       "humidifier_control": "speed"})
    assert c.national_data == "IT"
    assert c.duct_leakage_factor_supply == 1.45 and c.ahu_leakage_factor_supply == 1.10   # NA.2, NA.3
    assert c.heat_recovery_design_pressure_drop_pa == 400.0                            # NA.6
    assert c.supply_fan_heat_fraction_to_air == 0.6                                    # NA.12
    assert (c.supply_fan_pressure_control_fraction, c.extract_fan_pressure_control_fraction) == (0.3, 0.5)  # NA.13
    assert c.humidifier_pump_specific_power_wh_m3 == 0.04                              # NA.10
    assert c.controls_fan_power_fraction == 0.05                                       # NA.9
    o = calculate_sensible_ahu_step(
        sensible_ahu_config_from_dict({**base, "supply_fan_specific_power_w_per_m3_s": 1000.0,
                                       "extract_fan_specific_power_w_per_m3_s": 800.0}),
        AHUStepInputs(outdoor_temperature_c=0.0, extract_temperature_c=21.0,
                      required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    assert o.auxiliary_controls_w == pytest.approx(0.05 * o.fan_electric_power_w)
    # Annex B data set: no default control power; an explicit power always wins
    en = sensible_ahu_config_from_dict({**base, "national_data": "EN"})
    assert en.controls_fan_power_fraction is None
    assert sensible_ahu_config_from_dict({**base, "controls_power_w": 80.0}).controls_fan_power_fraction is None


# --------------------------------------------------------------------------- fan locations
def _fan_step(**loc):
    d = {"sensible_heat_recovery_efficiency": 0.8, "supply_temperature_setpoint_c": 20.0,
         "frost_control": "none", "heat_recovery_control": "always_on",
         "supply_fan_specific_power_w_per_m3_s": 1500.0, "extract_fan_specific_power_w_per_m3_s": 1200.0,
         "fan_performance_model": "constant_sfp",
         # declaring a fan location also applies the Table B.5 leakage: neutralised here
         # to test the thermal effect alone
         "hr_leakage_factor_supply": 1.0, "hr_leakage_factor_extract": 1.0,
         "outdoor_air_fraction_hr": 1.0, **loc}
    return calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(
        outdoor_temperature_c=-5.0, extract_temperature_c=21.0,
        required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))


def test_supply_fan_upstream_of_the_heat_recovery_32a_counted_once():
    rc = 1.204 * 1006.0
    dt_s = 1500.0 * 0.9 / rc                      # supply fan temperature rise
    dt_e = 1200.0 * 0.9 / rc
    down = _fan_step()
    up = _fan_step(supply_fan_location="UP_HR")
    # downstream: HR from -5 degC, fan heat added after the coils
    t_hr_down = -5.0 + 0.8 * (21.0 + dt_e + 5.0)
    assert down.actual_heating_coil_power_w == pytest.approx(rc * 1000 / 3600 * (20.0 - dt_s - t_hr_down))
    # upstream: the fan warms the outdoor air before the HR, nothing added afterwards
    t_pre = -5.0 + dt_s
    t_hr_up = t_pre + 0.8 * (21.0 + dt_e - t_pre)
    assert up.actual_heating_coil_power_w == pytest.approx(rc * 1000 / 3600 * (20.0 - t_hr_up))
    assert up.actual_supply_temperature_c == pytest.approx(20.0)
    assert up.actual_heating_coil_power_w > down.actual_heating_coil_power_w


def test_extract_fan_downstream_of_the_heat_recovery_31b():
    rc = 1.204 * 1006.0
    dt_s = 1500.0 * 0.9 / rc
    o = _fan_step(extract_fan_location="DOWN_HR")
    t_hr = -5.0 + 0.8 * (21.0 + 5.0)              # extract fan heat not in the HR
    assert o.actual_heating_coil_power_w == pytest.approx(rc * 1000 / 3600 * (20.0 - dt_s - t_hr))


# --------------------------------------------------------------------------- LOAD_COMP in zones
def test_load_compensated_supply_temperature_helper():
    from pybuildingenergy.source.ventilation import _load_compensated_supply_temperature
    assert _load_compensated_supply_temperature("a", None, 21.0, 500.0, 1.204, 1006.0) == 21.0
    prev = _simple({"supply_temperature_mode": "fixed", "supply_temperature_setpoint_c": 18.0},
                   outdoor_temperature_c=18.0, required_supply_flow_m3_h=500.0,
                   required_extract_flow_m3_h=500.0)
    t = _load_compensated_supply_temperature(
        "a", {"q_hvac_prev_w": 400.0, "ahu_prev": {"a": prev}}, 21.0, 500.0, 1.204, 1006.0)
    h = 1.204 * 1006.0 * 500.0 / 3600.0
    assert t == pytest.approx(21.0 + (400.0 + h * (prev.actual_supply_temperature_c - 21.0)) / h)


def _lc_engine(control_mode):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_ventilation_boundary import _legacy_building, _run_single_zone_engine
    comp = {"name": "ahu", "ventilation_type": "mechanical_supply", "supply_flow_m3_h": 600.0,
            "sensible_heat_recovery_efficiency": 0.0, "supply_temperature_mode": "load_compensated",
            "supply_temperature_minimum_c": 16.0, "supply_temperature_maximum_c": 35.0,
            "frost_control": "none", "heat_recovery_control": "always_on"}
    return _run_single_zone_engine(_legacy_building([comp]), control_mode=control_mode, n=48, t_out=-5.0)


@pytest.mark.parametrize("mode", ["standard", "ahu_causal"])
def test_zone_component_load_compensated_supply_temperature(mode):
    h = _lc_engine(mode)
    t_req = h["T_ahu_sup_req_ahu"]
    assert t_req.notna().all()
    # in winter the AHU supplies warm air to cover the zone heating load, within the limits
    assert t_req.max() > 21.0 and t_req.max() <= 35.0 + 1e-9 and t_req.min() >= 16.0 - 1e-9
