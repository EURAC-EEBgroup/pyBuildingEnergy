"""EN 16798-5-2:2017 (Method 2, monthly / yearly / bin): hand calculations of the formulae."""

import math

import pandas as pd
import pytest

from pybuildingenergy.source import ventilation_16798_5_2 as v

RC = 1.204 * 0.000279        # (rho c)_a, kWh/(m3 K), sea level


def _month(**kw):
    st = dict(q_sup_req_m3h=150.0, q_eta_req_m3h=150.0, hours=744.0, eta_hr=0.85)
    st.update(kw.pop("stage", {}))
    base = dict(total_hours=744.0, theta_e_c=2.0, theta_i_c=20.0, stages=[v.Stage(**st)])
    base.update(kw)
    return v.IntervalInputs(**base)


def test_tables_and_density():
    assert v.air_density(500.0) == pytest.approx(1.204 * (1 - 0.00651 * 500 / 288) ** 4.255)  # Table 9
    assert v.TABLE_B2_DUCT_LEAKAGE["A"] == 1.06 and v.TABLE_B4_AHU_LEAKAGE["3"] == 1.02
    assert v.insulation_factor(0.6) == 0.02                                    # Table B.6
    assert v.defrost_defaults("cold") == (1003.0, 5.2)                         # Tables B.7, B.8
    assert v.correction_extract_temperature(22.5) == pytest.approx(1.05)       # Table B.16
    assert v.correction_flow(0.75) == pytest.approx(1.045)


def test_flows_leakage_and_heat_recovery_4_to_12():
    cfg = v.Unit52Config(duct_leakage_factor=1.06, ahu_leakage_factor=1.02, ahu_insulation_factor=0.01)
    r = v.calculate_interval(cfg, _month())
    q = 150.0 * 1.06 * 1.02
    assert r["q_V_SUP_dis_m3h"] == pytest.approx(q)                            # (4), (5)
    assert r["q_V_lea_SUP_dis_m3h"] == pytest.approx(0.06 * q)                 # (9)
    st = r["stages"][0]
    # (20) extract duct with leakage from the surroundings (outdoor air)
    th_eta_out = 20.0 / 1.06 + 0.06 / 1.06 * 2.0
    assert st["theta_eta_dis_out_c"] == pytest.approx(th_eta_out)
    # (12) heat recovery
    d_hr = (0.85 - 0.02 - 0.01) * (th_eta_out - 2.0)
    assert st["d_theta_hr_k"] == pytest.approx(d_hr)
    assert st["theta_sup_dis_in_c"] == pytest.approx(2.0 + d_hr)               # (11)
    assert r["Q_hr_kwh"] == pytest.approx(RC * 1.0 * q / 1.06 * d_hr * 744.0)  # (38)


def test_bypass_time_reduces_the_recovery_38():
    cfg = v.Unit52Config()
    full = v.calculate_interval(cfg, _month())
    half = v.calculate_interval(cfg, _month(stage={"hours_hr": 372.0}))
    assert half["Q_hr_kwh"] == pytest.approx(full["Q_hr_kwh"] / 2.0)


def test_recirculation_14_and_outdoor_air_fraction_22():
    cfg = v.Unit52Config(heat_recovery="none")
    r = v.calculate_interval(cfg, _month(stage={"q_rca_m3h": 60.0, "eta_hr": 0.0}))
    st = r["stages"][0]
    assert st["d_theta_rca_k"] == pytest.approx(60.0 / 150.0 * (20.0 - 2.0))   # (14)
    assert st["f_oda"] == pytest.approx(0.6)                                   # (22)
    assert r["f_ODA"] == pytest.approx(0.6)                                    # (23g)
    assert r["Q_rca_kwh"] == pytest.approx(RC * 150.0 * st["d_theta_rca_k"] * 744.0)  # (39)


def test_reheating_15_and_ducts_16_to_19_and_25():
    cfg = v.Unit52Config(h_du_sup_nc_kw_k=0.005, h_du_sup_cnd_kw_k=0.003)
    r = v.calculate_interval(cfg, _month(q_h_v_in_kwh=200.0, theta_sur_c=8.0))
    st = r["stages"][0]
    assert st["d_theta_rh_k"] == pytest.approx(200.0 / (150.0 * RC * 744.0))   # (15)
    th_in = st["theta_sup_dis_in_c"]
    d_nc = (th_in - 8.0) * (1 - math.exp(-0.005 / (RC * 150.0)))               # (18)
    d_cnd = (th_in - d_nc - 20.0) * (1 - math.exp(-0.003 / (RC * 150.0)))      # (19)
    assert st["theta_sup_dis_out_c"] == pytest.approx(th_in - d_nc - d_cnd)    # (16)-(17)
    assert r["Q_ls_V_dis_kwh"] == pytest.approx(RC * 150.0 * d_nc * 744.0)     # (25)
    assert r["Q_ls_V_dis_rbl_kwh"] == 0.0                                      # (26)


def test_fan_energy_31a_31b_and_auxiliary_32_to_35():
    cfg = v.Unit52Config(spi_kw_per_m3h=[0.00025], dp_defrost_pa=10.0, fan_efficiency=0.35,
                         heat_recovery="rotary", rotor_power_max_kw=0.05, controls_power_kw=0.002)
    r = v.calculate_interval(cfg, _month(stage={"rotor_speed_rpm": 10.0}, defrost_hours=50.0,
                                         defrost_dt_k=2.4))
    assert r["E_V_gen_fan_kwh"] == pytest.approx(744.0 * 150.0 * (0.00025 + 2.78e-7 * 10.0 / 0.35))
    assert r["W_V_aux_hr_kwh"] == pytest.approx(744.0 * 0.05 * 10.0 / 20.0)   # (33)
    assert r["W_V_aux_ctrl_kwh"] == pytest.approx(744.0 * 0.002)              # (35)
    assert r["W_V_aux_preh_kwh"] == pytest.approx(RC * 150.0 * 1.0 * 50.0 * 2.4 * 1.0)  # (34)
    assert r["W_V_aux_kwh"] == pytest.approx(r["W_V_aux_hr_kwh"] + r["W_V_aux_ctrl_kwh"]
                                             + r["W_V_aux_preh_kwh"])      # (32)
    single = v.calculate_interval(
        v.Unit52Config(spi_single_kw_per_m3h=0.0003, flow_control_factor=0.85, motor_exponent=2.0,
                       q_sup_max_m3h=200.0), _month())
    assert single["E_V_gen_fan_kwh"] == pytest.approx(744.0 * 0.0003 * 200.0 * 0.85 ** 2)  # (31b)


def test_casing_losses_28_29_30():
    cfg = v.Unit52Config(ahu_au_kw_k=0.002, ahu_in_conditioned_zone=False)
    r = v.calculate_interval(cfg, _month(theta_sur_c=5.0))
    st = r["stages"][0]
    th_am = 0.5 * (2.0 + max(st["theta_sup_dis_in_c"], st["theta_eta_dis_out_c"]))   # (29)
    assert r["theta_a_m_c"] == pytest.approx(th_am)
    assert r["Q_ls_V_ahu_kwh"] == pytest.approx(0.002 * (th_am - 5.0) * 744.0)  # (28)
    assert r["Q_ls_V_gen_rbl_kwh"] == 0.0                                       # (30)
    inside = v.calculate_interval(v.Unit52Config(ahu_au_kw_k=0.002), _month())
    assert inside["Q_ls_V_ahu_kwh"] == 0.0                                      # (28) NOTE


def test_extract_air_heat_pump_41_to_58():
    hp = v.HeatPumpData(p_W_kw_per_m3h=0.0025, COP_W=2.9, p_H_kw_per_m3h=0.0027, COP_H=3.4)
    cfg = v.Unit52Config(heat_pump=hp)
    # small hot water demand: covered, with part of the time (42b)-(45b)
    r = v.calculate_interval(cfg, _month(q_w_hp_req_kwh=100.0, q_h_hp_req_kwh=2000.0,
                                         stage={"q_sup_req_m3h": 150.0}))
    f_eta = v.correction_extract_temperature(20.0)
    f_v = v.correction_flow(1.0)
    q_w_max = 0.0025 * 150.0 * 2.9 * 744.0 * f_eta * f_v                       # (41)
    assert r["Q_W_gen_hp_out"] == pytest.approx(100.0) and r["Q_W_residual"] == 0.0
    t_on_w = 100.0 / q_w_max * 744.0                                            # (44b)
    assert r["E_W_hp_in"] == pytest.approx(0.0025 * 150.0 * t_on_w)             # (56)
    q_h_max = 0.0027 * 150.0 * 3.4 * (744.0 - t_on_w) * f_eta * 1.068 * f_v - 100.0  # (46)
    assert r["Q_H_gen_hp_out"] == pytest.approx(min(2000.0, q_h_max))
    assert r["Q_H_residual"] == pytest.approx(max(0.0, 2000.0 - q_h_max))       # (48a)
    # cooling only when there is no heating requirement (NOTE 1)
    c = v.calculate_interval(v.Unit52Config(heat_pump=v.HeatPumpData(p_C_kw_per_m3h=0.0027, EER=2.6)),
                             _month(theta_e_c=28.0, theta_i_c=26.0, q_c_hp_req_kwh=50.0))
    assert c["Q_C_gen_hp_in"] == pytest.approx(50.0)
    q_c_max = 0.0027 * 150.0 * 2.6 * 744.0                                      # (51)
    assert c["E_C_hp_in"] == pytest.approx(0.0027 * 150.0 * 50.0 / q_c_max * 744.0)  # (58)


def test_preheating_and_precooling_36_37():
    cfg = v.Unit52Config()
    pre = v.calculate_interval(cfg, _month(theta_e_c=-3.0, stage={"d_theta_preh_k": 4.0}))
    assert pre["Q_preh_kwh"] == pytest.approx(RC * 1.0 * 150.0 * 4.0 * 744.0)   # (36), theta_e < 0
    pc = v.calculate_interval(cfg, _month(theta_e_c=28.0, theta_i_c=26.0, stage={"d_theta_prec_k": -3.0}))
    assert pc["Q_prec_kwh"] == pytest.approx(RC * 1.0 * 150.0 * -3.0 * 744.0)   # (37), theta_e > 26


def test_combine_by_area_24_and_monthly_from_hourly():
    a = v.calculate_interval(v.Unit52Config(), _month())
    b = v.calculate_interval(v.Unit52Config(), _month(stage={"eta_hr": 0.0}))
    comb = v.combine_by_area([a, b], [60.0, 40.0], 100.0)
    assert comb["Q_hr_kwh"] == pytest.approx(0.6 * a["Q_hr_kwh"])
    idx = pd.date_range("2025-01-01", periods=24 * 59, freq="h")
    hourly = pd.DataFrame({"T_ext": [-6.0 if i < 24 * 10 else 3.0 for i in range(len(idx))],
                           "T_air": 20.0}, index=idx)
    res = v.monthly_from_hourly(hourly, v.Unit52Config(),
                                v.Stage(q_sup_req_m3h=150.0, q_eta_req_m3h=150.0, hours=24.0, eta_hr=0.8),
                                defrost_climate="average")
    assert [m for m, _ in res] == [(2025, 1), (2025, 2)]
    jan, feb = res[0][1], res[1][1]
    assert jan["t_V_h"] == pytest.approx(31 * 24) and feb["t_V_h"] == pytest.approx(28 * 24)
    # all frost hours in January: the annual defrost energy is assigned to January
    assert jan["W_V_aux_preh_kwh"] > 0.0 and feb["W_V_aux_preh_kwh"] == 0.0


def test_unserved_demand_is_kept_as_residual():
    cfg = v.Unit52Config(heat_pump=v.HeatPumpData(p_H_kw_per_m3h=0.0027))
    with pytest.warns(UserWarning, match="domestic hot water"):
        r = v.calculate_interval(cfg, _month(q_w_hp_req_kwh=20.0))
    assert r["Q_W_gen_hp_out"] == 0.0 and r["Q_W_residual"] == 20.0
    with pytest.warns(UserWarning, match="space cooling"):
        c = v.calculate_interval(v.Unit52Config(), _month(q_c_hp_req_kwh=7.0))
    assert c["Q_C_residual"] == 7.0


def test_invalid_intervals_are_rejected():
    cfg = v.Unit52Config(heat_pump=v.HeatPumpData(p_H_kw_per_m3h=0.0027, p_C_kw_per_m3h=0.0027))
    with pytest.raises(ValueError, match="either space heating or space cooling"):
        v.calculate_interval(cfg, _month(q_h_hp_req_kwh=10.0, q_c_hp_req_kwh=5.0))
    with pytest.raises(ValueError, match="total_hours"):
        v.calculate_interval(cfg, _month(total_hours=0.0))
    with pytest.raises(ValueError, match="exceed t_total"):
        v.calculate_interval(cfg, _month(total_hours=500.0))          # stage has 744 h
    with pytest.raises(ValueError, match="defrost_hours"):
        v.calculate_interval(cfg, _month(defrost_hours=800.0))
    with pytest.raises(ValueError, match=">= 0"):
        v.calculate_interval(cfg, _month(q_w_hp_req_kwh=-1.0))


def test_monthly_from_hourly_keeps_years_apart_and_caps_defrost():
    idx = pd.date_range("2024-01-01", periods=24 * 31, freq="h").append(
        pd.date_range("2025-01-01", periods=24 * 31, freq="h"))
    hourly = pd.DataFrame({"T_ext": -6.0, "T_air": 20.0}, index=idx)
    res = v.monthly_from_hourly(hourly, v.Unit52Config(),
                                v.Stage(q_sup_req_m3h=150.0, q_eta_req_m3h=150.0, hours=24.0),
                                defrost_climate="cold")
    assert [p for p, _ in res] == [(2024, 1), (2025, 1)]
    for _, r in res:
        assert r["t_V_h"] == pytest.approx(31 * 24)
        # 1003 h/a scaled to the 744 h of data of the year, never above the interval hours
        assert r["W_V_aux_preh_kwh"] <= RC * 150.0 * 744.0 * 5.2 + 1e-9


def test_extract_duct_sign_as_printed_option():
    kw = dict(h_du_eta_nc_kw_k=0.01, duct_leakage_factor=1.0)
    lowered = v.calculate_interval(v.Unit52Config(**kw), _month(theta_sur_c=5.0))
    printed = v.calculate_interval(v.Unit52Config(extract_duct_sign_as_printed=True, **kw),
                                   _month(theta_sur_c=5.0))
    t_l = lowered["stages"][0]["theta_eta_dis_out_c"]
    t_p = printed["stages"][0]["theta_eta_dis_out_c"]
    assert t_l < 20.0 < t_p                   # printed sign warms the extract air
    assert t_p - 20.0 == pytest.approx(20.0 - t_l)


@pytest.mark.parametrize("eta", [float("nan"), 1.5, -0.1])
def test_invalid_heat_recovery_efficiency_is_rejected(eta):
    with pytest.raises(ValueError, match="eta_hr"):
        v.calculate_interval(v.Unit52Config(), _month(stage={"eta_hr": eta}))


def test_nan_defrost_hours_is_rejected():
    with pytest.raises(ValueError, match="defrost"):
        v.calculate_interval(v.Unit52Config(), _month(defrost_hours=float("nan")))


def test_invalid_unit_configuration_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        v.calculate_interval(v.Unit52Config(h_du_sup_nc_kw_k=float("nan")), _month())


def test_reheating_runs_for_the_whole_stage_when_recovery_is_bypassed():
    cfg = v.Unit52Config()
    r = v.calculate_interval(cfg, _month(theta_e_c=0.0, supply_temperature_setpoint_c=20.0,
                                         stage={"hours_hr": 0.0, "q_eta_req_m3h": 150.0}))
    assert r["theta_SUP_dis_out_c"][0] == pytest.approx(20.0)
    assert r["Q_H_V_in_req_kwh"] == pytest.approx(RC * 150.0 * 20.0 * 744.0, rel=1e-3)


def test_dhw_energy_deduction_in_46_is_switchable():
    kw = dict(p_W_kw_per_m3h=0.0025, COP_W=2.9, p_H_kw_per_m3h=0.0027, COP_H=3.4)
    inp = _month(q_w_hp_req_kwh=100.0, q_h_hp_req_kwh=5000.0)
    printed = v.calculate_interval(v.Unit52Config(heat_pump=v.HeatPumpData(**kw)), inp)
    single = v.calculate_interval(
        v.Unit52Config(heat_pump=v.HeatPumpData(deduct_dhw_energy_46=False, **kw)), inp)
    assert single["Q_H_gen_hp_out"] == pytest.approx(printed["Q_H_gen_hp_out"] + 100.0)
