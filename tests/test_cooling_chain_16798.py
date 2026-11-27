"""EN 16798-9 / -13 / -15 cooling modules: options, formulae and chaining checks."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.cooling_16798_9 import CoolingSystemCalculator
from pybuildingenergy.source.cooling_generation_16798_13 import CoolingGenerationSystemCalculator
from pybuildingenergy.source.cooling_storage_16798_15 import CoolingStorageSystemCalculator

IDX = pd.date_range("2026-07-01", periods=3, freq="h")


def _loads(**cols):
    base = {"T_ext": [24.0, 30.0, 35.0], "Q_C_kWh": [1.0, 2.0, 4.0]}
    base.update(cols)
    return pd.DataFrame(base, index=IDX)


# ---------------------------------------------------------------- EN 16798-9
def test_max_tmp_takes_the_minimum_circuit_requirement():
    """(17)/(20): MAX_TMP = min of the circuit / AHU coil requirements, within limits."""
    calc = CoolingSystemCalculator({
        "distribution_temperature_control": "MAX_TMP",
        "theta_C_dis_flw_set_min_C": 6.0, "theta_C_dis_flw_set_max_C": 18.0,
        "design_deltaT_K": 5.0,
    })
    ts = calc.run_timeseries(_loads(
        theta_C_ahu_in_req_C_1=[10.0, 9.0, 4.0],
        theta_C_dis_in_flw_req_C_2=[16.0, 8.0, 12.0],
    )).timeseries
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_req_C"], [10.0, 8.0, 6.0])


def test_max_tmp_without_circuit_requirements_raises():
    calc = CoolingSystemCalculator({"distribution_temperature_control": "MAX_TMP"})
    with pytest.raises(ValueError, match="MAX_TMP"):
        calc.run_timeseries(_loads())


def test_flow_temperature_follows_generator_and_storage_when_not_reached():
    """(23): theta_C;dis;in;flw = max(gen out, sto out, required); return from (24)."""
    calc = CoolingSystemCalculator({"theta_C_dis_flw_set_C": 7.0, "design_deltaT_K": 5.0,
                                    "design_flow_m3_h": 1.0, "design_cooling_load_kW": 4.0})
    ts = calc.run_timeseries(_loads(theta_C_gen_out_C=[6.0, 9.0, 7.0],
                                    theta_C_sto_out_C=[6.5, 8.0, 10.0])).timeseries
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_req_C"], 7.0)
    np.testing.assert_allclose(ts["theta_C_dis_supply_C"], [7.0, 9.0, 10.0])
    dt = ts["Q_C_dis_out_tot_req_kWh"] / (1.16 * ts["q_V_C_dis_m3_h"] * ts["hours"])
    np.testing.assert_allclose(ts["theta_C_dis_return_C"], ts["theta_C_dis_supply_C"] + dt)


def test_flow_temperature_defaults_to_required_without_actual_temperatures():
    ts = CoolingSystemCalculator({"theta_C_dis_flw_set_C": 7.0}).run_timeseries(_loads()).timeseries
    np.testing.assert_allclose(ts["theta_C_dis_supply_C"], 7.0)


# ---------------------------------------------------------------- EN 16798-13
_MAP = pd.DataFrame({
    "source_temperature_C": [25.0, 35.0, 25.0, 35.0],
    "sink_temperature_C": [7.0, 7.0, 12.0, 12.0],
    "capacity_kW": [4.0, 3.0, 4.0, 3.0],
    "eer": [4.0, 3.0, 4.2, 3.2],
})


def test_backup_electricity_is_in_the_rejected_heat():
    calc = CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP,
                                              "cooling_backup_eer": 2.5})
    ts = calc.run_timeseries(_loads(Q_C_kWh=[1.0, 2.0, 6.0])).timeseries
    assert ts["Q_C_gen_backup_in_kWh"].iloc[-1] > 0.0
    np.testing.assert_allclose(
        ts["Q_C_gen_out_kWh"],
        ts["Q_C_gen_in_kWh"] + ts["E_C_gen_el_in_kWh"] + ts["E_C_backup_in_kWh"],
    )


def test_hybrid_free_cooling_uses_wet_bulb():
    """(6c): WET and HYBRID compare the required outlet temperature with the wet bulb."""
    cfg = {"cooling_performance_map": _MAP, "free_cooling_enabled": True,
           "heat_rejection_type": "HYBRID", "free_cooling_deltaT_K": 4.0}
    loads = _loads(T_ext=[10.0, 10.0, 10.0], T_ext_wb_C=[2.0, 2.0, 2.0],
                   T_C_sink_C=[12.0, 12.0, 12.0])
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).timeseries
    # 12 - 10 (dry bulb) = 2 K < 4 K, but 12 - 2 (wet bulb) = 10 K > 4 K
    np.testing.assert_allclose(ts["f_op_fc"], 1.0)


def test_minimum_part_load_ratio_is_f_c_pl_min():
    """(20b): below f_C;PL;min the EER stays at its value at f_C;PL;min."""
    loads = _loads(Q_C_kWh=[0.4, 0.4, 0.4], T_ext=[30.0] * 3)
    base = {"cooling_performance_map": _MAP, "part_load_performance_method": "en14825"}
    a = CoolingGenerationSystemCalculator({**base, "minimum_part_load_ratio": 0.3}).run_timeseries(loads)
    b = CoolingGenerationSystemCalculator({**base, "part_load_minimum_capacity_ratio": 0.3}).run_timeseries(loads)
    np.testing.assert_allclose(a.timeseries["EER_C_gen"], b.timeseries["EER_C_gen"])
    np.testing.assert_allclose(a.timeseries["f_C_PL_for_performance"], 0.3)


def test_conflicting_minimum_part_load_ratios_raise():
    with pytest.raises(ValueError, match="f_C;PL;min"):
        CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP,
                                           "minimum_part_load_ratio": 0.3,
                                           "part_load_minimum_capacity_ratio": 0.2})


def test_minimum_part_load_ratio_with_simple_method_warns():
    with pytest.warns(RuntimeWarning, match="no effect"):
        CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP,
                                           "minimum_part_load_ratio": 0.3})


def test_stale_demand_column_warns_and_is_reported():
    """The generator prefers Q_C_gen_in_req_kWh over Q_C_kWh: a stale copy is flagged."""
    loads = _loads(Q_C_gen_in_req_kWh=[9.0, 9.0, 9.0])
    with pytest.warns(RuntimeWarning, match="Q_C_gen_in_req_kWh"):
        ts = CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP}).run_timeseries(loads).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_in_req_kWh"], 9.0)


def test_consistent_demand_columns_do_not_warn():
    loads = _loads(Q_C_gen_in_req_kWh=[1.0, 2.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP}).run_timeseries(loads)


# ---------------------------------------------------------------- EN 16798-15
def test_disabled_storage_passes_the_sink_temperature():
    ts = CoolingStorageSystemCalculator({"enabled": False}).run_timeseries(
        _loads(T_C_sink_C=[7.0, 8.0, 9.0])).timeseries
    np.testing.assert_allclose(ts["T_C_sink_C"], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(ts["Q_C_sto_in_kWh"], ts["Q_C_sto_out_kWh"])




# state of charge -------------------------------------------------------------
DAY = pd.date_range("2026-07-01", periods=48, freq="h")
DAY_HOURS = (DAY.hour >= 12) & (DAY.hour < 18)
SCHEDULE = [1 if 12 <= h < 18 else 0 for h in range(24)]
STORE = {"storage_volume_l": 2000.0, "theta_C_sto_C": 6.0, "theta_C_sto_return_C": 12.0,
         "H_C_sto_tot_ls_W_K": 3.0, "ambient_temperature_C": 25.0,
         "H_C_sto_out_ls_W_K": 0.0, "H_C_sto_in_ls_W_K": 0.0}
Q_MAX = 1.16 * 2.0 * (12.0 - 6.0)          # rho c V (theta_ret - theta_set) = 13.92 kWh


def _day_loads(**cols):
    data = {"T_ext": np.where(DAY_HOURS, 32.0, 20.0),
            "Q_C_kWh": np.where(DAY_HOURS, 4.0, 0.5), "T_C_sink_C": 6.0}
    data.update(cols)
    return pd.DataFrame(data, index=DAY)


def _energy_balance(ts):
    """Generator request = demand + absorbed losses + loops + pump heat + stored change."""
    stored0 = ts["Q_C_sto_max_kWh"].iloc[0]
    rhs = (ts["Q_C_sto_out_kWh"].sum() + ts["Q_C_sto_ls_kWh"].sum()
           + ts["Q_C_sto_out_ls_kWh"].sum() + ts["Q_C_sto_in_ls_kWh"].sum()
           - ts["Q_C_sto_aux_rvd_kWh"].sum()
           + ts["Q_C_sto_stored_kWh"].iloc[-1] - stored0)
    assert ts["Q_C_sto_in_kWh"].sum() == pytest.approx(rhs, rel=1e-9)


def test_cont_keeps_the_tank_full_and_matches_the_buffer_balance():
    ts = CoolingStorageSystemCalculator(STORE).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(ts["Q_C_sto_stored_kWh"], Q_MAX)
    np.testing.assert_allclose(ts["theta_C_sto_C"], 6.0)
    np.testing.assert_allclose(ts["Q_C_sto_in_kWh"], ts["Q_C_sto_out_kWh"] + ts["Q_C_sto_ls_tot_kWh"])
    np.testing.assert_allclose(ts["T_C_sink_C"], 6.0)      # no recharge beyond losses
    _energy_balance(ts)


def test_time_control_shifts_the_load_to_the_night():
    ts = CoolingStorageSystemCalculator({**STORE, "control_mode": "TIME",
                                         "discharge_schedule": SCHEDULE}).run_timeseries(_day_loads()).timeseries
    day, night = ts[DAY_HOURS], ts[~DAY_HOURS]
    # day: discharge limited by the stored energy (5); no charging
    assert day["Q_C_sto_dis_kWh"].sum() > 0.0
    assert (day["Q_C_sto_chg_kWh"] == 0.0).all()
    assert day["Q_C_sto_dis_kWh"].groupby(day.index.date).sum().max() <= Q_MAX + 1e-9
    # 6 h x 4 kWh = 24 kWh > Q_MAX: the tank empties, the rest comes from the generator
    assert day["Q_C_sto_stored_kWh"].min() == pytest.approx(0.0, abs=1e-9)
    assert day["Q_C_sto_direct_kWh"].sum() > 0.0
    # night: the generator recharges, nothing is discharged
    assert (night["Q_C_sto_dis_kWh"] == 0.0).all()
    assert night["Q_C_sto_chg_kWh"].sum() > 0.0
    # generator request in the afternoon lower than the demand
    assert day["Q_C_sto_in_kWh"].sum() < day["Q_C_sto_out_kWh"].sum()
    # storage temperature rises while discharging, from the set point to the return
    assert ts["theta_C_sto_C"].max() == pytest.approx(12.0, abs=1e-6)
    assert ts["theta_C_sto_C"].min() == pytest.approx(6.0, abs=1e-6)
    # net contribution (EN 16798-9 (26) sign): > 0 discharging, < 0 charging
    np.testing.assert_allclose(ts["Q_C_sto_net_kWh"], ts["Q_C_sto_dis_kWh"] - ts["Q_C_sto_chg_kWh"])
    _energy_balance(ts)


def test_recharge_uses_the_formula_1_flow_temperature():
    ts = CoolingStorageSystemCalculator({**STORE, "control_mode": "TIME", "delta_theta_C_sto_gen_flw_K": 2.0,
                                         "discharge_schedule": SCHEDULE}).run_timeseries(_day_loads()).timeseries
    recharge = ts["Q_C_sto_chg_kWh"] > ts["Q_C_sto_ls_kWh"] + 1e-6
    assert recharge.any()
    np.testing.assert_allclose(ts.loc[recharge, "T_C_sink_C"], ts.loc[recharge, "theta_C_sto_C"] - 2.0)
    np.testing.assert_allclose(ts.loc[~recharge, "T_C_sink_C"], 6.0)


def test_charge_power_limits_the_recharge():
    ts = CoolingStorageSystemCalculator({**STORE, "control_mode": "TIME", "charge_power_kW": 1.0,
                                         "discharge_schedule": SCHEDULE}).run_timeseries(_day_loads()).timeseries
    assert ts["Q_C_sto_chg_kWh"].max() <= 1.0 + 1e-9
    _energy_balance(ts)


def test_temp_control_discharges_above_the_outdoor_limit():
    ts = CoolingStorageSystemCalculator({**STORE, "control_mode": "TEMP", "theta_e_lim_C": 25.0}
                                        ).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(ts["f_C_sto_ctrl"], np.where(DAY_HOURS, 1.0, 0.0))
    assert (ts.loc[~DAY_HOURS, "Q_C_sto_dis_kWh"] == 0.0).all()


def test_control_column_drives_load_pred():
    f = np.where(DAY_HOURS, 1.0, 0.0)
    a = CoolingStorageSystemCalculator({**STORE, "control_mode": "LOAD_PRED"}).run_timeseries(
        _day_loads(f_C_sto_ctrl=f)).timeseries
    b = CoolingStorageSystemCalculator({**STORE, "control_mode": "TIME", "discharge_schedule": SCHEDULE}
                                       ).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(a["Q_C_sto_in_kWh"], b["Q_C_sto_in_kWh"])


@pytest.mark.parametrize("cfg,match", [
    ({"control_mode": "TIME"}, "discharge_schedule"),
    ({"control_mode": "TEMP"}, "theta_e_lim_C"),
    ({"control_mode": "TIME", "discharge_schedule": [1] * 23}, "24 hourly"),
    ({"charge_power_kW": 0.0}, "charge_power_kW"),
])
def test_invalid_control_configuration_raises(cfg, match):
    with pytest.raises(ValueError, match=match):
        CoolingStorageSystemCalculator(cfg)


def test_load_pred_without_control_column_raises():
    with pytest.raises(ValueError, match="f_C_sto_ctrl"):
        CoolingStorageSystemCalculator({"control_mode": "LOAD_PRED"}).run_timeseries(_day_loads())


def test_zero_volume_supplies_everything_directly():
    ts = CoolingStorageSystemCalculator({**STORE, "storage_volume_l": 0.0, "H_C_sto_tot_ls_W_K": 0.0,
                                         "control_mode": "TIME", "discharge_schedule": SCHEDULE}
                                        ).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(ts["Q_C_sto_dis_kWh"], 0.0)
    np.testing.assert_allclose(ts["Q_C_sto_in_kWh"], ts["Q_C_sto_out_kWh"])


def test_storage_inlet_requirement_is_the_distribution_requirement():
    ts = CoolingStorageSystemCalculator({}).run_timeseries(
        _loads(theta_C_dis_in_flw_req_C=[7.0, 8.0, 9.0])).timeseries
    np.testing.assert_allclose(ts["theta_C_sto_in_req_C"], [7.0, 8.0, 9.0])


# ---------------------------------------------------------------- chain
def test_chain_energy_boundaries():
    """9 -> 15 -> 13 with Q_C_kWh handed over: each stage adds its own losses/heat."""
    loads = _loads(Q_C_kWh=[1.0, 2.0, 3.0])
    s9 = CoolingSystemCalculator({"f_C_ls_dis": 0.05, "f_C_aux_dis": 0.02,
                                  "f_wat_C_aux_dis": 1.0}).run_timeseries(loads).timeseries
    np.testing.assert_allclose(s9["Q_C_gen_in_req_kWh"], loads["Q_C_kWh"] * (1 + 0.05 + 0.02))

    stage = loads.copy()
    stage["Q_C_kWh"] = s9["Q_C_gen_in_req_kWh"]
    stage["T_C_sink_C"] = s9["T_C_sink_C"]
    s15 = CoolingStorageSystemCalculator({"H_C_sto_tot_ls_W_K": 2.0, "ambient_temperature_C": 25.0,
                                          "input_pump_power_kW": 0.02, "input_pump_flow_m3_h": 0.8,
                                          }).run_timeseries(stage).timeseries
    np.testing.assert_allclose(
        s15["Q_C_sto_in_kWh"],
        stage["Q_C_kWh"] + s15["Q_C_sto_ls_tot_kWh"] + s15["W_C_sto_aux_kWh"],
    )

    stage["Q_C_kWh"] = s15["Q_C_sto_in_kWh"]
    stage["T_C_sink_C"] = s15["T_C_sink_C"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s13 = CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP}).run_timeseries(stage).timeseries
    np.testing.assert_allclose(s13["Q_C_gen_in_req_kWh"], s15["Q_C_sto_in_kWh"])
    np.testing.assert_allclose(s13["Q_C_gen_in_kWh"] + s13["Q_C_gen_unmet_kWh"], s13["Q_C_gen_in_req_kWh"])


# stratification ----------------------------------------------------------------
def _time_store(**extra):
    return {**STORE, "control_mode": "TIME", "discharge_schedule": SCHEDULE, **extra}


def test_single_node_is_the_mixed_tank():
    a = CoolingStorageSystemCalculator(_time_store()).run_timeseries(_day_loads()).timeseries
    b = CoolingStorageSystemCalculator(_time_store(stratification_nodes=1)).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(a["Q_C_sto_in_kWh"], b["Q_C_sto_in_kWh"])


def test_stratified_tank_keeps_the_outlet_cold():
    mixed = CoolingStorageSystemCalculator(_time_store()).run_timeseries(_day_loads()).timeseries
    strat = CoolingStorageSystemCalculator(_time_store(stratification_nodes=20)).run_timeseries(_day_loads()).timeseries
    _energy_balance(strat)
    assert {"theta_C_sto_bottom_C", "theta_C_sto_top_C"} <= set(strat.columns)
    # same stored energy, but the stratified outlet stays near the set point longer
    first = DAY_HOURS & (DAY.day == 1)
    assert strat.loc[first, "theta_C_sto_out_act_C"].iloc[0] == pytest.approx(6.0, abs=1e-6)
    assert strat.loc[first, "theta_C_sto_out_act_C"].mean() < mixed.loc[first, "theta_C_sto_out_act_C"].mean()
    assert strat["theta_C_sto_bottom_C"].min() >= 6.0 - 1e-9
    assert strat["theta_C_sto_top_C"].max() <= 12.0 + 1e-9


def test_outlet_temperature_limit_favours_stratification():
    """With a 9 C limit the mixed tank stops at half charge, the stratified one later."""
    lim = {"theta_C_sto_out_max_C": 9.0}
    mixed = CoolingStorageSystemCalculator(_time_store(**lim)).run_timeseries(_day_loads()).timeseries
    strat = CoolingStorageSystemCalculator(_time_store(stratification_nodes=20, **lim)
                                           ).run_timeseries(_day_loads()).timeseries
    assert strat["Q_C_sto_dis_kWh"].sum() > mixed["Q_C_sto_dis_kWh"].sum()
    _energy_balance(mixed)
    _energy_balance(strat)


# ice and PCM -------------------------------------------------------------------
ICE = {"storage_type": "STO_TYPE_ICE", "storage_volume_l": 1000.0, "theta_C_sto_C": -4.0,
       "theta_C_sto_return_C": 12.0, "H_C_sto_tot_ls_W_K": 5.0, "ambient_temperature_C": 25.0,
       "H_C_sto_out_ls_W_K": 0.0, "H_C_sto_in_ls_W_K": 0.0}
# Table B.2: Q_lat 93 Wh/kg, Cp sld 0.54, Cp lqd 1.16 Wh/(kg K); M = 1000 kg
E_L = 1000 * 1.16e-3 * 12.0          # liquid 12 -> 0 C
E_LAT = 1000 * 0.093
E_S = 1000 * 0.54e-3 * 0.5 * 4.0      # ice, linear profile 0 -> -4 C


def test_ice_capacity_from_the_three_bands():
    ts = CoolingStorageSystemCalculator(ICE).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(ts["Q_C_sto_max_kWh"], E_L + E_LAT + E_S)
    assert ts["m_C_sto_sld_kg"].iloc[0] == pytest.approx(1000.0)


def test_ice_on_coil_mass_limited_by_the_maximum_thickness():
    cfg = {**ICE, "tube_outer_diameter_m": 0.025, "tube_length_m": 100.0}
    ts = CoolingStorageSystemCalculator(cfg).run_timeseries(_day_loads()).timeseries
    m_max = 900 * np.pi * 100.0 * ((0.0125 + 0.035) ** 2 - 0.0125 ** 2)        # (13)
    np.testing.assert_allclose(ts["Q_C_sto_max_kWh"],
                               E_L + m_max * 0.093 + m_max * 0.54e-3 * 0.5 * 4.0)
    assert ts["d_C_sto_m"].max() == pytest.approx(0.035)
    assert ts["d_C_sto_m"].min() >= 0.0


def test_melting_rate_limits_the_discharge():
    loads = pd.DataFrame({"T_ext": 30.0, "Q_C_kWh": [40.0], "T_C_sink_C": 6.0},
                         index=pd.date_range("2026-07-01 12:00", periods=1, freq="h"))
    cfg = {**ICE, "control_mode": "LOAD_PRED", "H_C_sto_tot_ls_W_K": 0.0}
    ts = CoolingStorageSystemCalculator(cfg).run_timeseries(loads.assign(f_C_sto_ctrl=1.0)).timeseries
    melt = 2.1 * 1.0 * (12.0 - 0.0)                     # K_v,fu V (theta_ret - theta_tr)
    assert ts["Q_C_sto_dis_kWh"].iloc[0] == pytest.approx(E_S + melt)
    assert ts["Q_C_sto_direct_kWh"].iloc[0] == pytest.approx(40.0 - E_S - melt)


def test_freezing_rate_limits_the_charge():
    cfg = {**ICE, "control_mode": "TIME", "discharge_schedule": SCHEDULE, "H_C_sto_tot_ls_W_K": 0.0}
    ts = CoolingStorageSystemCalculator(cfg).run_timeseries(_day_loads(Q_C_kWh=np.where(DAY_HOURS, 30.0, 0.0))
                                                            ).timeseries
    freeze = 0.55 * 1.0 * (0.0 - (-4.0))               # K_v,cr V (theta_tr - theta_charge)
    # steps that start and end in the latent band (the liquid band charges unlimited)
    start = ts["Q_C_sto_stored_kWh"].shift(1)
    in_latent = (start >= E_L - 1e-9) & (ts["Q_C_sto_stored_kWh"] <= E_L + E_LAT) & ~DAY_HOURS
    assert in_latent.any()
    np.testing.assert_allclose(ts.loc[in_latent, "Q_C_sto_chg_kWh"], freeze)
    # first night hour: liquid band (13.92 kWh) + one hour of freezing
    assert ts["Q_C_sto_chg_kWh"].iloc[18] == pytest.approx(E_L + freeze)
    _energy_balance(ts)


def test_ice_needs_a_charging_temperature_below_freezing():
    ts = CoolingStorageSystemCalculator({**ICE, "theta_C_sto_C": 2.0}).run_timeseries(_day_loads()).timeseries
    np.testing.assert_allclose(ts["Q_C_sto_max_kWh"], 1000 * 1.16e-3 * (12.0 - 2.0))
    np.testing.assert_allclose(ts["m_C_sto_sld_kg"], 0.0)


def test_pcm_capacity_and_conservation():
    cfg = {"storage_type": "STO_TYPE_PCM", "theta_sto_tr_C": 8.0, "theta_C_sto_C": 5.0,
           "storage_volume_l": 500.0, "theta_C_sto_return_C": 12.0, "H_C_sto_tot_ls_W_K": 1.0,
           "ambient_temperature_C": 25.0, "H_C_sto_out_ls_W_K": 0.0, "H_C_sto_in_ls_W_K": 0.0,
           "control_mode": "TIME", "discharge_schedule": SCHEDULE}
    ts = CoolingStorageSystemCalculator(cfg).run_timeseries(_day_loads(Q_C_kWh=np.where(DAY_HOURS, 2.0, 0.0))
                                                            ).timeseries
    m = 560 * 0.5                                        # Table B.2 PCM density
    q_max = m * 0.616e-3 * (12 - 8) + m * 0.0271 + m * 0.392e-3 * (8 - 5)
    np.testing.assert_allclose(ts["Q_C_sto_max_kWh"], q_max)
    assert ts["theta_C_sto_C"].min() == pytest.approx(5.0)
    _energy_balance(ts)


@pytest.mark.parametrize("cfg,match", [
    ({"storage_type": "STO_TYPE_PCM"}, "theta_sto_tr_C"),
    ({"storage_type": "STO_TYPE_ICE", "stratification_nodes": 4}, "STO_TYPE_CW only"),
    ({"stratification_nodes": 0}, "stratification_nodes"),
    ({"storage_type": "STO_TYPE_ICE", "tube_length_m": 10.0}, "tube_outer_diameter_m"),
])
def test_invalid_medium_configuration_raises(cfg, match):
    with pytest.raises(ValueError, match=match):
        CoolingStorageSystemCalculator(cfg)


# several circuits (EN 16798-9 7.4.2-7.4.3) ------------------------------------
CIRCUITS = [
    {"name": "fancoil", "priority": 1, "demand_columns": ["Q_z1", "Q_z2"],
     "theta_C_dis_flw_set_C": 7.0, "design_flow_m3_h": 1.0, "design_cooling_load_kW": 5.0},
    {"name": "ahu", "priority": 2, "demand_columns": "Q_ahu",
     "distribution_temperature_control": "MAX_TMP", "ahu_required_temperature_columns": ["t_coil"],
     "design_deltaT_K": 4.0},
]


def _circuit_loads():
    return pd.DataFrame({"T_ext": [25.0, 30.0, 35.0], "Q_z1": [0.5, 1.0, 1.5], "Q_z2": [0.5, 1.0, 0.5],
                         "Q_ahu": [1.0, 4.0, 0.0], "t_coil": [10.0, 8.0, 12.0]}, index=IDX)


def _circuit_calc(**cfg):
    return CoolingSystemCalculator({"distribution_temperature_control": "MAX_TMP",
                                    "circuits": CIRCUITS, **cfg})


def test_circuits_total_request_and_required_temperatures():
    ts = _circuit_calc().run_timeseries(_circuit_loads()).timeseries
    np.testing.assert_allclose(ts["Q_C_dis_in_req_kWh_fancoil"], [1.0, 2.0, 2.0])     # (28)
    np.testing.assert_allclose(ts["Q_C_dis_out_tot_req_kWh"], [2.0, 6.0, 2.0])         # (27)
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_req_C_ahu"], [10.0, 8.0, 12.0])  # (20)
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_req_C"], [7.0, 7.0, 7.0])        # (17) min


def test_circuit_flows_returns_and_mixed_return():
    ts = _circuit_calc().run_timeseries(_circuit_loads()).timeseries
    q_fc = 1.0 * np.array([1.0, 2.0, 2.0]) / 5.0                                       # (22)
    np.testing.assert_allclose(ts["q_V_C_dis_m3_h_fancoil"], q_fc)
    q_ahu = np.array([1.0, 4.0, 0.0]) / (1.16 * 4.0)
    np.testing.assert_allclose(ts["q_V_C_dis_m3_h_ahu"], q_ahu)
    # (23): the AHU circuit raises the 7 C network water to its own requirement
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_C_ahu"], [10.0, 8.0, 12.0])
    t_ret_fc = 7.0 + np.array([1.0, 2.0, 2.0]) / (1.16 * q_fc)                       # (24)
    np.testing.assert_allclose(ts["theta_C_dis_out_ret_C_fancoil"], t_ret_fc)
    mix = (q_fc * t_ret_fc + q_ahu * ts["theta_C_dis_out_ret_C_ahu"]) / (q_fc + q_ahu)
    np.testing.assert_allclose(ts["theta_C_dis_out_ret_req_C"], mix)
    np.testing.assert_allclose(ts["q_V_C_dis_m3_h"], q_fc + q_ahu)


def test_shortfall_hits_the_lowest_priority_first():
    calc = _circuit_calc()
    res9 = calc.run_timeseries(_circuit_loads())
    out = calc.distribute_shortfall(res9, [1.5, 5.0, 10.0])
    # step 1: AHU (priority 2) loses 1.0, fancoil 0.5
    np.testing.assert_allclose(out["Q_C_dis_in_kWh_ahu"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out["Q_C_dis_in_kWh_fancoil"], [0.5, 1.0, 0.0])
    np.testing.assert_allclose(out["Q_C_dis_unmet_tot_kWh"], [1.5, 5.0, 2.0])    # capped at the demand
    # zones of a circuit share pro rata
    np.testing.assert_allclose(out["Q_C_del_kWh_Q_z1"], [0.25, 0.5, 0.0])
    np.testing.assert_allclose(out["Q_C_del_kWh_Q_z2"], [0.25, 0.5, 0.0])
    np.testing.assert_allclose(out["f_C_sup_fancoil"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(out["Q_C_dis_in_tot_kWh"] + out["Q_C_dis_unmet_tot_kWh"],
                               res9.timeseries["Q_C_dis_out_tot_req_kWh"])


def test_equal_priority_shares_pro_rata():
    circuits = [dict(c, priority=1) for c in CIRCUITS]
    calc = _circuit_calc(circuits=circuits)
    res9 = calc.run_timeseries(_circuit_loads())
    out = calc.distribute_shortfall(res9, 3.0)
    # step 1: demand 2 kWh < 3 kWh shortfall -> nothing delivered; step 2: 3 of 6 kWh -> half each
    np.testing.assert_allclose(out["f_C_sup_fancoil"][:2], [0.0, 0.5])
    np.testing.assert_allclose(out["f_C_sup_ahu"][:2], [0.0, 0.5])


def test_no_shortfall_delivers_everything():
    calc = _circuit_calc()
    res9 = calc.run_timeseries(_circuit_loads())
    out = calc.distribute_shortfall(res9, 0.0)
    np.testing.assert_allclose(out["Q_C_dis_in_tot_kWh"], res9.timeseries["Q_C_dis_out_tot_req_kWh"])


def test_single_circuit_feedback_without_circuits():
    calc = CoolingSystemCalculator({})
    res9 = calc.run_timeseries(_loads())
    out = calc.distribute_shortfall(res9, [0.0, 1.0, 5.0])
    np.testing.assert_allclose(out["Q_C_dis_in_kWh_main"], [1.0, 1.0, 0.0])


def test_feedback_from_the_generator_shortfall():
    """9 -> 13 with an undersized generator: the shortfall goes back to the AHU circuit."""
    calc = _circuit_calc()
    res9 = calc.run_timeseries(_circuit_loads())
    stage = _circuit_loads()[["T_ext"]].assign(Q_C_kWh=res9.timeseries["Q_C_gen_in_req_kWh"],
                                                T_C_sink_C=res9.timeseries["T_C_sink_C"])
    res13 = CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP}).run_timeseries(stage)  # 3-4 kW
    unmet = res13.timeseries["Q_C_gen_unmet_kWh"]
    assert unmet.iloc[1] > 0.0
    out = calc.distribute_shortfall(res9, unmet)
    assert out["Q_C_dis_unmet_kWh_ahu"].iloc[1] == pytest.approx(unmet.iloc[1])
    assert out["Q_C_dis_unmet_kWh_fancoil"].iloc[1] == pytest.approx(0.0)


@pytest.mark.parametrize("circuits,match", [
    ([{"name": "a", "demand_columns": "x"}, {"name": "a", "demand_columns": "y"}], "unique"),
    ([{"name": "a"}], "demand_columns"),
    ([{"name": "a", "demand_columns": "x", "distribution_temperature_control": "MAX_TMP"}], "ahu_required"),
    ([{"name": "a", "demand_columns": "x", "priority": 0}], "priority"),
    ([], "non-empty"),
])
def test_invalid_circuits_raise(circuits, match):
    with pytest.raises(ValueError, match=match):
        CoolingSystemCalculator({"circuits": circuits})


def test_missing_circuit_column_raises():
    with pytest.raises(ValueError, match="not found"):
        _circuit_calc().run_timeseries(_circuit_loads().drop(columns="Q_z2"))


# EN 16798-13: absorption chillers --------------------------------------------
def _zc(e, c, h):
    return (273.15 + e) * (h - c) / ((273.15 + h) * (c - e))


def _gen_loads(**cols):
    data = {"T_ext": [30.0, 30.0, 30.0], "Q_C_kWh": [2.0, 5.0, 12.0],
            "T_C_sink_C": 7.0, "T_C_source_C": 30.0}
    data.update(cols)
    return pd.DataFrame(data, index=IDX)


ABS = {"generation_type": "ABS", "nominal_capacity_kW": 10.0, "zeta_n": 0.7,
       "absorption_electric_ratio": 0.01}


def test_absorption_at_nominal_conditions():
    ts = CoolingGenerationSystemCalculator(ABS).run_timeseries(_gen_loads()).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], 10.0)                 # (9d) at nominal
    np.testing.assert_allclose(ts["zeta_C_gen"], 0.7)                           # (22)
    q = np.array([2.0, 5.0, 10.0])
    np.testing.assert_allclose(ts["Q_H_C_gen_abs_in_kWh"], q / 0.7)             # (21)
    np.testing.assert_allclose(ts["E_C_gen_el_in_kWh"], 0.01 * q)
    np.testing.assert_allclose(ts["Q_C_gen_out_kWh"], q + q / 0.7 + 0.01 * q)   # (16)
    np.testing.assert_allclose(ts["Q_C_gen_unmet_kWh"], [0.0, 0.0, 2.0])


def test_absorption_follows_the_carnot_heat_ratio():
    ts = CoolingGenerationSystemCalculator(ABS).run_timeseries(
        _gen_loads(theta_H_C_gen_abs_in_C=80.0)).timeseries
    ratio = _zc(7, 30, 80) / _zc(7, 30, 90)                                     # (23)
    np.testing.assert_allclose(ts["zeta_C_gen"], 0.7 * ratio)
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], 10.0 * ratio)


def test_absorption_with_coefficients():
    c = {"C5": -0.05, "C6": 0.4, "C7": 0.1}
    ts = CoolingGenerationSystemCalculator({**ABS, **c}).run_timeseries(
        _gen_loads(T_C_source_C=27.0)).timeseries
    zc, zc_n = _zc(7, 27, 90), _zc(7, 30, 90)
    p = lambda x: -0.05 * x * x + 0.4 * x + 0.1
    np.testing.assert_allclose(ts["zeta_C_gen"], p(zc))
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], 10.0 * p(zc) / p(zc_n))


def test_absorption_without_driving_temperature_lift_has_no_capacity():
    ts = CoolingGenerationSystemCalculator(ABS).run_timeseries(
        _gen_loads(theta_H_C_gen_abs_in_C=25.0)).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], 0.0)
    np.testing.assert_allclose(ts["Q_C_gen_unmet_kWh"], [2.0, 5.0, 12.0])


@pytest.mark.parametrize("cfg,match", [
    ({**ABS, "C5": 0.1}, "C5, C6, C7"),
    ({**ABS, "cooling_performance_map": _MAP}, "performance map"),
    ({**ABS, "part_load_performance_method": "curve", "part_load_curve_coefficients": [0.1, 0.5, 0.1]},
     "a \\+ b \\+ c = 1"),
    ({**ABS, "part_load_performance_method": "curve"}, "part_load_curve_coefficients"),
    ({"generation_type": "XYZ", "nominal_capacity_kW": 5.0}, "COMP, ABS or OTHER"),
])
def test_invalid_absorption_configuration(cfg, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        CoolingGenerationSystemCalculator(cfg)


# EN 16798-13: partial free cooling ---------------------------------------------
FC = {"cooling_performance_map": _MAP, "free_cooling_enabled": True, "heat_rejection_type": "DRY",
      "free_cooling_model": "partial", "free_cooling_ua_kW_K": 2.0}


def test_partial_free_cooling_covers_up_to_the_exchanger_capacity():
    loads = _loads(T_ext=[8.0, 8.0, 14.0], Q_C_kWh=[5.0, 16.0, 4.0], T_C_sink_C=7.0,
                   theta_C_dis_return_C=12.0)
    ts = CoolingGenerationSystemCalculator(FC).run_timeseries(loads).timeseries
    # UA (theta_ret - theta_e) t = 2 x (12 - 8) = 8 kWh; at 14 C no free cooling
    np.testing.assert_allclose(ts["Q_C_gen_fc_max_kWh"], [8.0, 8.0, 0.0])
    np.testing.assert_allclose(ts["Q_C_gen_free_kWh"], [5.0, 8.0, 0.0])
    np.testing.assert_allclose(ts["f_op_fc"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(ts["Q_C_gen_mech_in_kWh"] + ts["Q_C_gen_unmet_kWh"], [0.0, 8.0, 4.0])


def test_partial_free_cooling_default_return_temperature():
    ts = CoolingGenerationSystemCalculator({**FC, "free_cooling_return_deltaT_K": 4.0}).run_timeseries(
        _loads(T_ext=[8.0] * 3, Q_C_kWh=[50.0] * 3, T_C_sink_C=7.0)).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_fc_max_kWh"], 2.0 * (7.0 + 4.0 - 8.0))


def test_binary_free_cooling_is_the_default():
    cfg = {k: v for k, v in FC.items() if k not in {"free_cooling_model", "free_cooling_ua_kW_K"}}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(
        _loads(T_ext=[2.0, 8.0, 14.0], T_C_sink_C=7.0)).timeseries
    np.testing.assert_allclose(ts["f_op_fc"], [1.0, 0.0, 0.0])        # 7 - 2 > 4 K only


def test_partial_free_cooling_needs_ua():
    with pytest.raises(ValueError, match="free_cooling_ua_kW_K"):
        CoolingGenerationSystemCalculator({**FC, "free_cooling_ua_kW_K": None})


# EN 16798-13: several generators with priorities -------------------------------
GENS = [
    {"name": "small", "priority": 1, "nominal_capacity_kW": 3.0, "nominal_eer": 4.0},
    {"name": "big", "priority": 2, "nominal_capacity_kW": 10.0, "nominal_eer": 3.0},
]


def _nominal_loads(q):
    return pd.DataFrame({"T_ext": 35.0, "Q_C_kWh": q, "T_C_sink_C": 7.0, "T_C_source_C": 35.0},
                        index=pd.date_range("2026-07-01", periods=len(q), freq="h"))


def test_priority_cascade():
    ts = CoolingGenerationSystemCalculator({"generators": GENS}).run_timeseries(
        _nominal_loads([2.0, 5.0, 15.0])).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_small"], [2.0, 3.0, 3.0])       # (10)/(11)
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_big"], [0.0, 2.0, 10.0])
    np.testing.assert_allclose(ts["Q_C_gen_unmet_kWh"], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(ts["E_C_gen_el_in_kWh"],
                               ts["Q_C_gen_in_kWh_small"] / 4.0 + ts["Q_C_gen_in_kWh_big"] / 3.0)
    np.testing.assert_allclose(ts["Q_C_gen_in_max_kWh"], 13.0)
    np.testing.assert_allclose(ts["Q_C_gen_out_kWh"], ts["Q_C_gen_in_kWh"] + ts["E_C_gen_el_in_kWh"])


def test_reversed_priorities():
    gens = [dict(GENS[0], priority=2), dict(GENS[1], priority=1)]
    ts = CoolingGenerationSystemCalculator({"generators": gens}).run_timeseries(
        _nominal_loads([5.0, 12.0])).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_big"], [5.0, 10.0])
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_small"], [0.0, 2.0])


def test_equal_priority_shares_pro_rata_to_capacity():
    gens = [dict(g, priority=1) for g in GENS]
    ts = CoolingGenerationSystemCalculator({"generators": gens}).run_timeseries(
        _nominal_loads([6.5])).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_small"], 6.5 * 3 / 13)
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_big"], 6.5 * 10 / 13)


def test_backup_takes_the_remainder_of_the_cascade():
    ts = CoolingGenerationSystemCalculator({"generators": GENS, "cooling_backup_eer": 2.0}).run_timeseries(
        _nominal_loads([15.0])).timeseries
    assert ts["Q_C_gen_backup_in_kWh"].iloc[0] == pytest.approx(2.0)
    assert ts["E_C_backup_in_kWh"].iloc[0] == pytest.approx(1.0)
    assert ts["Q_C_gen_unmet_kWh"].iloc[0] == pytest.approx(0.0)


def test_absorption_first_then_compression():
    gens = [{"name": "abs", "priority": 1, **ABS, "theta_H_C_gen_abs_in_C": 90.0},
            {"name": "comp", "priority": 2, "nominal_capacity_kW": 10.0, "nominal_eer": 3.0}]
    loads = _nominal_loads([6.0, 14.0]).assign(T_C_source_C=30.0)
    ts = CoolingGenerationSystemCalculator({"generators": gens}).run_timeseries(loads).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_abs"], [6.0, 10.0])
    np.testing.assert_allclose(ts["Q_H_C_gen_abs_in_kWh"], ts["Q_C_gen_in_kWh_abs"] / 0.7)
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_comp"], [0.0, 4.0])


def test_single_generator_list_equals_the_plain_calculator():
    loads = _gen_loads(T_C_source_C=[25.0, 30.0, 35.0])
    a = CoolingGenerationSystemCalculator({"cooling_performance_map": _MAP}).run_timeseries(loads).timeseries
    b = CoolingGenerationSystemCalculator({"generators": [{"name": "g", "cooling_performance_map": _MAP}]}
                                          ).run_timeseries(loads).timeseries
    for col in ("Q_C_gen_in_kWh", "E_C_gen_el_in_kWh", "Q_C_gen_out_kWh", "Q_C_gen_unmet_kWh"):
        np.testing.assert_allclose(a[col], b[col])


def test_free_cooling_before_the_cascade():
    cfg = {"generators": GENS, **{k: v for k, v in FC.items() if k != "cooling_performance_map"}}
    loads = _nominal_loads([10.0]).assign(T_ext=8.0, theta_C_dis_return_C=12.0)
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).timeseries
    assert ts["Q_C_gen_free_kWh"].iloc[0] == pytest.approx(8.0)
    assert ts["Q_C_gen_in_kWh_small"].iloc[0] == pytest.approx(2.0)
    assert ts["Q_C_gen_in_kWh_big"].iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize("gens,match", [
    ([{"name": "a", "nominal_capacity_kW": 1.0}, {"name": "a", "nominal_capacity_kW": 1.0}], "unique"),
    ([{"name": "a", "nominal_capacity_kW": 1.0, "priority": 0}], "priority"),
    ([], "non-empty"),
])
def test_invalid_generators(gens, match):
    with pytest.raises(ValueError, match=match):
        CoolingGenerationSystemCalculator({"generators": gens})


# EN 16798-13: absorption at part load (library extension) ----------------------
def test_absorption_en14825_part_load():
    cfg = {**ABS, "part_load_performance_method": "en14825", "part_load_degradation_coefficient": 0.9}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_gen_loads(Q_C_kWh=[5.0, 10.0, 1.0])).timeseries
    plf = lambda x: x / (0.9 * x + 0.1)
    np.testing.assert_allclose(ts["zeta_C_gen_full_load"], 0.7)
    np.testing.assert_allclose(ts["zeta_C_gen"], 0.7 * np.array([plf(0.5), 1.0, plf(0.1)]))
    np.testing.assert_allclose(ts["Q_H_C_gen_abs_in_kWh"], np.array([5.0, 10.0, 1.0]) / ts["zeta_C_gen"])
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], 10.0)            # capacity unchanged


def test_absorption_part_load_curve_with_minimum():
    cfg = {**ABS, "part_load_performance_method": "curve", "part_load_curve_coefficients": [0.2, 0.8, 0.0],
           "minimum_part_load_ratio": 0.3}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_gen_loads(Q_C_kWh=[5.0, 1.0, 0.0])).timeseries
    # factor 0.2 + 0.8 x; at 10 % load the factor stays at its 30 % value (20b)
    np.testing.assert_allclose(ts["f_C_PLF"], [0.6, 0.44, 1.0])
    np.testing.assert_allclose(ts["zeta_C_gen"], 0.7 * np.array([0.6, 0.44, 1.0]))
    np.testing.assert_allclose(ts["Q_H_C_gen_abs_in_kWh"], [5.0 / 0.42, 1.0 / 0.308, 0.0])


def test_part_load_curve_for_compression():
    cfg = {"nominal_capacity_kW": 10.0, "nominal_eer": 4.0, "theta_cond_in_nominal_C": 30.0,
           "part_load_performance_method": "curve", "part_load_curve_coefficients": [0.5, 0.5, 0.0]}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_gen_loads(Q_C_kWh=[5.0, 10.0, 2.0])).timeseries
    np.testing.assert_allclose(ts["EER_C_gen"], 4.0 * np.array([0.75, 1.0, 0.6]))


# EN 16798-13: generators of type OTHER (9g), (24) ------------------------------
OTH = {"generation_type": "OTHER", "nominal_capacity_kW": 5.0, "other_electric_ratio": 0.05}


def test_other_generator_with_constants():
    ts = CoolingGenerationSystemCalculator(OTH).run_timeseries(_gen_loads(Q_C_kWh=[2.0, 5.0, 8.0])).timeseries
    q = np.array([2.0, 5.0, 5.0])
    np.testing.assert_allclose(ts["Q_C_gen_mech_in_kWh"], q)
    np.testing.assert_allclose(ts["Q_C_gen_unmet_kWh"], [0.0, 0.0, 3.0])
    np.testing.assert_allclose(ts["E_C_gen_el_in_kWh"], 0.05 * q)                 # (24)
    np.testing.assert_allclose(ts["EER_C_gen"], 20.0)
    np.testing.assert_allclose(ts["Q_C_gen_out_kWh"], q + 0.05 * q)              # (16)
    np.testing.assert_allclose(ts["Q_H_C_gen_abs_in_kWh"], 0.0)


def test_other_generator_with_functions_of_the_sink_and_part_load():
    seen = []

    def capacity(theta_out_req, theta_sk):
        return 1.0 - 0.05 * (theta_sk - 10.0)

    def electric(theta_out, theta_sk, f_pl):
        seen.append((theta_out, theta_sk))
        return 0.02 + 0.03 * f_pl

    cfg = {**OTH, "other_capacity_factor": capacity, "other_electric_ratio": electric}
    loads = _gen_loads(Q_C_kWh=[3.0, 3.0, 3.0], theta_sk_C=[10.0, 14.0, 18.0], T_C_sink_C=[4.0, 7.0, 7.0])
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], [5.0, 4.0, 3.0])        # (9g)
    fpl = np.array([0.6, 0.75, 1.0])
    np.testing.assert_allclose(ts["f_C_PL"], fpl)
    np.testing.assert_allclose(ts["E_C_gen_el_in_kWh"], 3.0 * (0.02 + 0.03 * fpl))
    assert seen[0] == (5.0, 10.0)                       # theta_C;gen;out = max(req, limit 5 C)


@pytest.mark.parametrize("cfg,match", [
    ({k: v for k, v in OTH.items() if k != "other_electric_ratio"}, "other_electric_ratio"),
    ({**OTH, "part_load_performance_method": "en14825"}, "simple"),
    ({**OTH, "cooling_performance_map": _MAP}, "performance map"),
    ({**OTH, "other_electric_ratio": -0.1}, "non-negative"),
])
def test_invalid_other_configuration(cfg, match):
    with pytest.raises(ValueError, match=match):
        CoolingGenerationSystemCalculator(cfg)


def test_other_function_returning_negative_values_raises():
    cfg = {**OTH, "other_electric_ratio": lambda t, sk, f: -1.0}
    with pytest.raises(ValueError, match="negative"):
        CoolingGenerationSystemCalculator(cfg).run_timeseries(_gen_loads())


def test_other_generator_first_then_compression():
    gens = [{"name": "ground", "priority": 1, **OTH},
            {"name": "comp", "priority": 2, "nominal_capacity_kW": 10.0, "nominal_eer": 3.0}]
    ts = CoolingGenerationSystemCalculator({"generators": gens}).run_timeseries(
        _nominal_loads([4.0, 9.0])).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_ground"], [4.0, 5.0])
    np.testing.assert_allclose(ts["Q_C_gen_in_kWh_comp"], [0.0, 4.0])
    el = np.array([0.2, 0.25 + 4.0 / 3.0])
    np.testing.assert_allclose(ts["E_C_gen_el_in_kWh"], el)
    np.testing.assert_allclose(ts["EER_C_gen"], np.array([4.0, 9.0]) / el)


# EN 16798-13: heat-rejection operation 6.4.2.4 ---------------------------------
def _hr_loads(**cols):
    data = {"T_ext": [20.0, 30.0], "T_ext_wb_C": [15.0, 22.0], "Q_C_kWh": [2.0, 2.0], "T_C_sink_C": 7.0}
    data.update(cols)
    return pd.DataFrame(data, index=IDX[:2])


HR = {"nominal_capacity_kW": 10.0, "nominal_eer": 4.0, "heat_rejection_deltaT_K": 5.0,
      "performance_includes_heat_rejection_aux": False, "p_hr_el_dry": 0.02, "p_hr_el_wet": 0.01}


def test_hybrid_temp_control_switches_at_the_limit():
    cfg = {**HR, "heat_rejection_type": "HYBRID", "hybrid_heat_rejection_control": "TEMP",
           "theta_lim_wet_hr_C": 25.0}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_hr_loads()).timeseries
    np.testing.assert_allclose(ts["f_op_hr_wet"], [0.0, 1.0])                     # (13d)/(13e)
    np.testing.assert_allclose(ts["theta_cond_in_C"], [25.0, 27.0])               # (14d)/(14e), (3a)/(3b)
    np.testing.assert_allclose(ts["W_C_aux_gen_kWh"], ts["Q_C_gen_out_kWh"] * [0.02, 0.01])   # (26b)


def test_hybrid_max_power_goes_wet_when_the_dry_capacity_is_short():
    cfg = {"cooling_performance_map": _MAP, "heat_rejection_type": "HYBRID", "heat_rejection_deltaT_K": 2.0}
    loads = _hr_loads(T_ext=[33.0, 33.0], T_ext_wb_C=[23.0, 23.0], Q_C_kWh=[2.0, 3.5])
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).timeseries
    # dry: 35 C -> 3 kW; wet: 25 C -> 4 kW
    np.testing.assert_allclose(ts["f_op_hr_wet"], [0.0, 1.0])                     # (13f)/(13g)
    np.testing.assert_allclose(ts["theta_cond_in_C"], [35.0, 25.0])
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], [3.0, 4.0])             # (8b)/(8c)
    np.testing.assert_allclose(ts["Q_C_gen_mech_in_kWh"], [2.0, 3.5])
    assert CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).summary["t_C_hr_wet_h"] == 1.0


def test_hybrid_max_power_decided_on_all_generators():
    gens = [{"name": "a", "cooling_performance_map": _MAP}, {"name": "b", "cooling_performance_map": _MAP}]
    cfg = {"generators": gens, "heat_rejection_type": "HYBRID", "heat_rejection_deltaT_K": 2.0}
    loads = _hr_loads(T_ext=[33.0, 33.0], T_ext_wb_C=[23.0, 23.0], Q_C_kWh=[5.0, 7.0])
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).timeseries
    np.testing.assert_allclose(ts["f_op_hr_wet"], [0.0, 1.0])                     # 7 > 3 + 3 kWh
    np.testing.assert_allclose(ts["Q_C_gen_in_max_kWh_a"], [3.0, 4.0])
    np.testing.assert_allclose(ts["Q_C_gen_in_max_kWh"], [6.0, 8.0])
    np.testing.assert_allclose(ts["Q_C_gen_unmet_kWh"], 0.0)


def test_wet_and_dry_heat_rejection_use_their_own_rates():
    wet = CoolingGenerationSystemCalculator({**HR, "heat_rejection_type": "WET"}).run_timeseries(_hr_loads())
    dry = CoolingGenerationSystemCalculator({**HR, "heat_rejection_type": "DRY"}).run_timeseries(_hr_loads())
    np.testing.assert_allclose(wet.timeseries["theta_cond_in_C"], [20.0, 27.0])
    np.testing.assert_allclose(dry.timeseries["theta_cond_in_C"], [25.0, 35.0])
    np.testing.assert_allclose(wet.timeseries["W_C_aux_gen_kWh"], wet.timeseries["Q_C_gen_out_kWh"] * 0.01)
    np.testing.assert_allclose(dry.timeseries["W_C_aux_gen_kWh"], dry.timeseries["Q_C_gen_out_kWh"] * 0.02)


def test_condenser_inlet_limit():
    cfg = {**HR, "heat_rejection_type": "DRY", "theta_cond_in_lim_C": 28.0}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_hr_loads()).timeseries
    np.testing.assert_allclose(ts["theta_cond_in_C"], [28.0, 35.0])               # (14b)


def test_other_heat_rejection_uses_the_sink_temperature_series():
    cfg = {**HR, "heat_rejection_type": "OTHER", "p_hr_el_other": 0.03}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_hr_loads(theta_sk_C=[11.0, 13.0])).timeseries
    np.testing.assert_allclose(ts["theta_cond_in_C"], [11.0, 13.0])               # (14h)
    np.testing.assert_allclose(ts["W_C_aux_gen_kWh"], ts["Q_C_gen_out_kWh"] * 0.03)   # (26a)


def test_capacity_temperature_correction():
    cfg = {"nominal_capacity_kW": 10.0, "nominal_eer": 4.0, "capacity_temperature_correction": True}
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(_gen_loads(T_C_source_C=[30.0, 35.0, 40.0])).timeseries
    expected = [10.0 * (280.15 / (c - 7.0)) / (280.15 / 28.0) for c in (30.0, 35.0, 40.0)]   # (9b), dtheta_corr 0
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], expected)


@pytest.mark.parametrize("cfg,match", [
    ({**HR, "heat_rejection_type": "HYBRID", "hybrid_heat_rejection_control": "TEMP"}, "theta_lim_wet_hr_C"),
    ({**HR, "heat_rejection_type": "HYBRID", "hybrid_heat_rejection_control": "X"}, "TEMP or MAX_POWER"),
    ({"cooling_performance_map": _MAP, "capacity_temperature_correction": True}, "nominal-EER"),
])
def test_invalid_heat_rejection_configuration(cfg, match):
    with pytest.raises(ValueError, match=match):
        CoolingGenerationSystemCalculator(cfg)


# EN 16798-13 (20a)/(20b) from EN 14825 points, 6.3.2.2.2.1 ---------------------
_TRUE = (-0.05, 0.06, 0.05, 0.054, 4.0)      # synthetic C1-C4, delta_theta_corr (not product data)
_EER_N = 3.5


def _f_eer(cond, evap, x, c=_TRUE):
    return (273.15 + evap) / (cond - evap + x * c[4]) * (c[0] * x ** 3 + c[1] * x ** 2 + c[2] * x + c[3])


def _points(n=5):
    pts = [(1.0, 34.0), (0.74, 29.0), (0.47, 24.0), (0.21, 19.0), (0.47, 34.0)][:n]
    return [{"f_C_PL": x, "EER": _EER_N * _f_eer(tc, 7.0, x), "theta_C_evap_out_C": 7.0, "theta_cond_in_C": tc}
            for x, tc in pts]


def test_en16798_13_fit_recovers_the_coefficients():
    from pybuildingenergy.source.cooling_generation_16798_13 import _fit_en16798_13
    np.testing.assert_allclose(_fit_en16798_13(_points(), _EER_N), _TRUE, rtol=1e-9, atol=1e-12)


def test_en16798_13_fourth_point_estimate_uses_formula_2():
    from pybuildingenergy.source.cooling_generation_16798_13 import _fit_en16798_13
    pts = _points(4)
    c = _fit_en16798_13(pts, _EER_N)
    eer5 = pts[2]["EER"] * (24.0 - 7.0) / (34.0 - 7.0)                         # (2)
    assert _EER_N * _f_eer(34.0, 7.0, 0.47, c) == pytest.approx(eer5)
    for p in pts:                                                              # (1) holds at A-D
        assert _EER_N * _f_eer(p["theta_cond_in_C"], 7.0, p["f_C_PL"], c) == pytest.approx(p["EER"])


def test_en16798_13_part_load_eer_and_capacity():
    cfg = {"nominal_capacity_kW": 10.0, "nominal_eer": _EER_N, "part_load_performance_method": "en16798_13",
           "en14825_points": _points(), "minimum_part_load_ratio": 0.3}
    cap = 10.0 * (280.15 / (23.0 + 4.0)) / (280.15 / (28.0 + 4.0))           # (9b) with delta_theta_corr
    loads = _gen_loads(Q_C_kWh=[0.5 * cap, 0.1 * cap, 2 * cap])
    ts = CoolingGenerationSystemCalculator(cfg).run_timeseries(loads).timeseries
    np.testing.assert_allclose(ts["Q_C_gen_capacity_kW"], cap)
    np.testing.assert_allclose(ts["EER_C_gen"], _EER_N * np.array(
        [_f_eer(30.0, 7.0, 0.5), _f_eer(30.0, 7.0, 0.3), _f_eer(30.0, 7.0, 1.0)]))     # (20a), (20b), full
    np.testing.assert_allclose(ts["EER_C_gen_full_load"], _EER_N * _f_eer(30.0, 7.0, 1.0))


@pytest.mark.parametrize("cfg,match", [
    ({"nominal_capacity_kW": 10.0, "part_load_performance_method": "en16798_13", "en14825_points": _points()},
     "nominal_eer"),
    ({"nominal_capacity_kW": 10.0, "nominal_eer": 3.0, "part_load_performance_method": "en16798_13"},
     "en14825_points"),
    ({**ABS, "part_load_performance_method": "en16798_13", "en14825_points": _points()}, "compression"),
    ({"nominal_capacity_kW": 10.0, "nominal_eer": 3.0, "part_load_performance_method": "en16798_13",
      "en14825_points": _points()[:3]}, "A, B, C, D"),
])
def test_invalid_en16798_13_part_load(cfg, match):
    with pytest.raises(ValueError, match=match):
        CoolingGenerationSystemCalculator(cfg)


def test_indices_per_energy_carrier():
    """SEER_C_gen counts electricity only; absorption reports its seasonal heat ratio separately."""
    s = CoolingGenerationSystemCalculator(ABS).run_timeseries(_gen_loads()).summary
    q = 2.0 + 5.0 + 10.0
    assert s["SEER_C_gen"] == pytest.approx(q / (0.01 * q))
    assert s["SEER_C_gen_el"] == s["SEER_C_gen"]
    assert s["zeta_C_gen_seasonal"] == pytest.approx(0.7)
    assert s["EER_C_gen_all_inputs"] == pytest.approx(q / (0.01 * q + q / 0.7))
