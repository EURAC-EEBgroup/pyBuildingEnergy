"""Coupled cooling chain EN 16798-9 -> -15 -> -13 against independent reference balances.

The references below are written from scratch (plain energy bookkeeping with a
fully mixed tank, no losses), not with the library code, so they check the coupled
balance: storage state, charge received, shortfall to the users and to the charge,
and the priorities of the circuits.
"""

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.cooling_chain_16798 import CoolingChainCalculator
from pybuildingenergy.source.cooling_storage_16798_15 import CoolingStorageSystemCalculator


DAYS = 3
IDX = pd.date_range("2026-07-01", periods=24 * DAYS, freq="h")
DAY_HOURS = (IDX.hour >= 10) & (IDX.hour < 18)
SCHEDULE = [1.0 if 10 <= h < 18 else 0.0 for h in range(24)]

# 2000 l between 7 and 12 C: 1.16 x 2 x 5 = 11.6 kWh
TANK = {"storage_volume_l": 2000.0, "theta_C_sto_C": 7.0, "theta_C_sto_return_C": 12.0,
        "theta_C_gen_out_C": 7.0, "H_C_sto_tot_ls_W_K": 0.0, "H_C_sto_out_ls_W_K": 0.0,
        "H_C_sto_in_ls_W_K": 0.0, "control_mode": "TIME", "discharge_schedule": SCHEDULE}
CAP_TANK = 1.16 * 2.0 * 5.0


def _gen(capacity_kW, **extra):
    return {"nominal_capacity_kW": capacity_kW, "nominal_eer": 3.0, **extra}


def _loads(day_load=4.0, night_load=0.0):
    q = np.where(DAY_HOURS, day_load, night_load)
    return pd.DataFrame({"T_ext": 30.0, "T_C_source_C": 30.0, "Q_C_kWh": q, "T_C_sink_C": 7.0}, index=IDX)


def _reference(demand, f_ctrl, avl, cap, start=None, emergency=True):
    """Fully mixed tank, no losses: discharge (5), users before charging, charge when f < 1."""
    e = cap if start is None else start
    rows = []
    for d, f, a in zip(demand, f_ctrl, avl):
        req = f * d
        if emergency and d - a > req:
            req = d - a
        dis = min(e, req)
        e -= dis
        direct = min(d - dis, a)
        unmet = d - dis - direct
        wanted = charge = 0.0
        if f < 1.0:
            wanted = cap - e
            charge = min(wanted, a - direct)
            e += charge
        rows.append((dis, direct, unmet, charge, wanted - charge, e))
    return pd.DataFrame(rows, columns=["dis", "direct", "unmet", "chg", "chg_short", "stored"])


def _check(res, ref):
    sto = res.storage.timeseries
    np.testing.assert_allclose(sto["Q_C_sto_dis_kWh"], ref["dis"], atol=1e-9)
    np.testing.assert_allclose(sto["Q_C_sto_direct_kWh"], ref["direct"], atol=1e-9)
    np.testing.assert_allclose(sto["Q_C_sto_unmet_kWh"], ref["unmet"], atol=1e-9)
    np.testing.assert_allclose(sto["Q_C_sto_chg_kWh"], ref["chg"], atol=1e-9)
    np.testing.assert_allclose(sto["Q_C_sto_chg_unmet_kWh"], ref["chg_short"], atol=1e-9)
    np.testing.assert_allclose(sto["Q_C_sto_stored_kWh"], ref["stored"], atol=1e-9)
    np.testing.assert_allclose(res.delivered["Q_C_dis_unmet_tot_kWh"], ref["unmet"], atol=1e-9)
    assert res.converged
    assert res.generation.summary["QC_gen_unmet_kWh"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- reference cases
def test_multi_day_cycle_with_an_adequate_generator():
    res = CoolingChainCalculator(_gen(3.0), storage=TANK).run_timeseries(_loads())
    ref = _reference(_loads()["Q_C_kWh"], np.array(SCHEDULE * DAYS), np.full(len(IDX), 3.0), CAP_TANK)
    _check(res, ref)
    # each day: 11.6 kWh from the tank, 0.4 kWh direct at 12:00, then 3 kWh/h direct and 1 kWh/h unmet
    assert ref["unmet"].sum() == pytest.approx(5.0 * DAYS)


def test_undersized_generator_cannot_recharge_the_tank_overnight():
    """0.5 kW x 16 night hours = 8 kWh < 11.6 kWh: the charge shortfall carries over to the next day."""
    res = CoolingChainCalculator(_gen(0.5), storage=TANK).run_timeseries(_loads())
    ref = _reference(_loads()["Q_C_kWh"], np.array(SCHEDULE * DAYS), np.full(len(IDX), 0.5), CAP_TANK)
    _check(res, ref)
    assert ref["chg_short"].sum() > 0.0
    days = res.storage.timeseries["Q_C_sto_unmet_kWh"].groupby(IDX.date).sum().to_numpy()
    assert days[1] > days[0]                        # day 2 starts with a partly charged tank


def test_several_generators_feed_the_storage_like_one_with_their_total_capacity():
    gens = {"generators": [{"name": "a", "priority": 1, **_gen(1.0)},
                           {"name": "b", "priority": 2, **_gen(0.5)}]}
    res = CoolingChainCalculator(gens, storage=TANK).run_timeseries(_loads(day_load=3.0))
    ref = _reference(_loads(day_load=3.0)["Q_C_kWh"], np.array(SCHEDULE * DAYS), np.full(len(IDX), 1.5), CAP_TANK)
    _check(res, ref)
    gen = res.generation.timeseries
    # priority 1 runs first: the second generator only when the first one is at its maximum
    assert ((gen["Q_C_gen_in_kWh_b"] > 1e-9) <= (gen["Q_C_gen_in_kWh_a"] >= 1.0 - 1e-9)).all()


def test_zero_availability_empties_the_store_and_leaves_the_rest_unmet():
    loads = _loads().assign(f_op_C=0.0)
    res = CoolingChainCalculator(_gen(3.0), storage=TANK).run_timeseries(loads)
    ref = _reference(loads["Q_C_kWh"], np.array(SCHEDULE * DAYS), np.zeros(len(IDX)), CAP_TANK)
    _check(res, ref)
    assert res.summary["QC_users_unmet_kWh"] == pytest.approx(32.0 * DAYS - CAP_TANK)


def test_without_emergency_discharge_the_control_is_kept():
    cfg = {**TANK, "discharge_on_generator_shortfall": False}
    loads = _loads(night_load=2.0)                      # night load above the 1 kW generator
    res = CoolingChainCalculator(_gen(1.0), storage=cfg).run_timeseries(loads)
    ref = _reference(loads["Q_C_kWh"], np.array(SCHEDULE * DAYS), np.full(len(IDX), 1.0), CAP_TANK,
                     emergency=False)
    _check(res, ref)
    night = ~DAY_HOURS
    assert (res.storage.timeseries.loc[night, "Q_C_sto_dis_kWh"] == 0.0).all()


def test_no_demand_no_energy():
    res = CoolingChainCalculator(_gen(3.0), storage=TANK).run_timeseries(_loads(day_load=0.0))
    assert res.summary["QC_users_unmet_kWh"] == 0.0
    assert res.storage.timeseries["Q_C_sto_in_kWh"].sum() == pytest.approx(0.0)
    assert res.generation.summary["EC_total_kWh"] == pytest.approx(0.0)


def test_without_storage_the_generator_shortfall_reaches_the_users():
    res = CoolingChainCalculator(_gen(3.0)).run_timeseries(_loads())
    np.testing.assert_allclose(res.delivered["Q_C_dis_unmet_tot_kWh"], np.where(DAY_HOURS, 1.0, 0.0))
    assert res.storage is None


# --------------------------------------------------------------- circuits and zones
CIRCUITS = [
    {"name": "zones", "priority": 1, "demand_columns": ["Q_z1", "Q_z2"],
     "theta_C_dis_flw_set_C": 7.0, "design_flow_m3_h": 1.0, "design_cooling_load_kW": 5.0},
    {"name": "ahu", "priority": 2, "demand_columns": "Q_ahu", "theta_C_dis_flw_set_C": 7.0,
     "design_deltaT_K": 4.0},
]


def _circuit_loads():
    q = lambda v: np.where(DAY_HOURS, v, 0.0)
    return pd.DataFrame({"T_ext": 30.0, "T_C_source_C": 30.0, "Q_z1": q(1.5), "Q_z2": q(0.5), "Q_ahu": q(2.0)},
                        index=IDX)


def test_shortfall_reaches_the_circuits_by_priority_and_the_zones_pro_rata():
    chain = CoolingChainCalculator(_gen(3.0), distribution={"circuits": CIRCUITS}, storage=TANK)
    res = chain.run_timeseries(_circuit_loads())
    ref = _reference(np.where(DAY_HOURS, 4.0, 0.0), np.array(SCHEDULE * DAYS), np.full(len(IDX), 3.0), CAP_TANK)
    _check(res, ref)
    d = res.delivered
    # the AHU (priority 2) loses first, the zone circuit keeps everything while unmet <= 2 kWh
    np.testing.assert_allclose(d["Q_C_dis_unmet_kWh_ahu"], np.minimum(ref["unmet"], 2.0), atol=1e-9)
    np.testing.assert_allclose(d["Q_C_dis_unmet_kWh_zones"], 0.0, atol=1e-9)
    np.testing.assert_allclose(d["Q_C_del_kWh_Q_z1"], np.where(DAY_HOURS, 1.5, 0.0), atol=1e-9)
    total = res.distribution.timeseries["Q_C_dis_out_tot_req_kWh"]
    np.testing.assert_allclose(d["Q_C_dis_in_tot_kWh"] + d["Q_C_dis_unmet_tot_kWh"], total)


# --------------------------------------------------------------- latent store and temperatures
ICE = {"storage_type": "STO_TYPE_ICE", "storage_volume_l": 500.0, "theta_C_sto_return_C": 12.0,
       "H_C_sto_tot_ls_W_K": 0.0, "H_C_sto_out_ls_W_K": 0.0, "H_C_sto_in_ls_W_K": 0.0,
       "control_mode": "TIME", "discharge_schedule": SCHEDULE}


def test_ice_is_not_rebuilt_when_the_generator_cannot_go_below_freezing():
    """Generator limited to 5 C: after the first day only the water band (12 -> 5 C) is recharged."""
    gen = _gen(20.0, theta_C_gen_out_limit_C=5.0)
    res = CoolingChainCalculator(gen, storage=ICE).run_timeseries(_loads(day_load=8.0))
    sto = res.storage.timeseries
    water_band = 500.0 * 1.16e-3 * (12.0 - 5.0)
    last_night = (IDX.day == IDX[-1].day) & (IDX.hour == 9)
    assert sto.loc[last_night, "m_C_sto_sld_kg"].iloc[0] == pytest.approx(0.0)
    assert sto.loc[last_night, "Q_C_sto_stored_kWh"].iloc[0] == pytest.approx(water_band)
    assert sto["Q_C_sto_chg_unmet_kWh"].sum() == pytest.approx(0.0)     # nothing more is reachable


def test_ice_is_rebuilt_when_the_generator_reaches_the_charging_temperature():
    gen = _gen(20.0, theta_C_gen_out_limit_C=-5.0)
    res = CoolingChainCalculator(gen, storage=ICE).run_timeseries(_loads(day_load=8.0))
    sto = res.storage.timeseries
    last_night = (IDX.day == IDX[-1].day) & (IDX.hour == 9)
    assert CoolingStorageSystemCalculator(ICE).theta_C_sto_C == pytest.approx(-2.0)
    # the day empties the store; the night recharges the water band 12 -> 0 C at once
    # (500 kg x 1.16 Wh/kgK x 12 K = 6.96 kWh) and then freezes at K_cr V (0 - (-2)) =
    # 0.55 x 0.5 x 2 = 0.55 kWh/h for 16 h (Table B.2 rate limit)
    assert sto.loc[last_night, "Q_C_sto_stored_kWh"].iloc[0] == pytest.approx(6.96 + 16 * 0.55)
    assert sto.loc[last_night, "m_C_sto_sld_kg"].iloc[0] == pytest.approx(16 * 0.55 / 0.093)


def test_latent_charge_follows_the_given_generator_temperature():
    """Stand-alone storage: a 'theta_C_gen_out_C' column above 0 C stops the ice build-up."""
    cfg = {**ICE, "control_mode": "CONT"}
    # melting 12.6 kWh/h (K_fu V (12 - 0)) empties the 53.7 kWh store within five hours
    loads = pd.DataFrame({"Q_C_sto_out_kWh": [12.6] * 5 + [0.0, 0.0],
                          "theta_C_gen_out_C": [4.0] * 6 + [-3.0]}, index=IDX[:7])
    sto = CoolingStorageSystemCalculator(cfg).run_timeseries(loads).timeseries
    assert sto["m_C_sto_sld_kg"].iloc[5] == pytest.approx(0.0)
    assert sto["Q_C_sto_stored_kWh"].iloc[5] == pytest.approx(500.0 * 1.16e-3 * (12.0 - 4.0))
    # at -3 C the charge temperature is theta_C_sto_C = -2 C: freezing at 0.55 x 0.5 x 2 kWh/h
    assert sto["m_C_sto_sld_kg"].iloc[6] == pytest.approx(0.55 / 0.093)


# --------------------------------------------------------------- temperature-dependent availability
def test_charging_at_a_lower_temperature_reduces_the_capacity_and_the_loop_converges():
    """(9a)-(9c): the capacity depends on the required temperature, which depends on the state
    of charge (formula (1): theta_sto - 2 K while recharging): storage and generator iterate."""
    gen = _gen(1.0, capacity_temperature_correction=True)
    res = CoolingChainCalculator(gen, storage=TANK).run_timeseries(_loads())
    assert res.converged
    gen_ts = res.generation.timeseries
    assert res.generation.summary["QC_gen_unmet_kWh"] == pytest.approx(0.0, abs=1e-6)
    # capacity (9b) with delta_theta_corr = 0 at the temperature required by the storage
    t = res.storage.timeseries["theta_C_sto_in_req_C"].clip(lower=5.0)
    expected = 1.0 * ((273.15 + t) / (30.0 - t)) / (280.15 / 28.0)
    np.testing.assert_allclose(gen_ts["Q_C_gen_capacity_kW"], expected)
    charging = res.storage.timeseries["Q_C_sto_chg_kWh"] > 1e-9
    assert charging.any()
    np.testing.assert_allclose(res.storage.timeseries["Q_C_sto_in_kWh"],
                               gen_ts["Q_C_gen_in_kWh"], atol=1e-6)


def test_invalid_chain():
    with pytest.raises(ValueError, match="generation"):
        CoolingChainCalculator(None)
    with pytest.raises(ValueError, match="Q_C_kWh"):
        CoolingChainCalculator(_gen(1.0)).run_timeseries(pd.DataFrame({"T_ext": [30.0]}, index=IDX[:1]))


@pytest.mark.parametrize("mode", ["demand", "continuous"])
def test_pumps_and_loop_losses_fit_inside_the_generator_availability(mode):
    """The storage reserves the loop losses (7), (8) and the pump heat before direct supply and
    charging, so its request never exceeds what the generator can give."""
    cfg = {**TANK, "H_C_sto_out_ls_W_K": 0.5, "H_C_sto_in_ls_W_K": 0.5, "H_C_sto_tot_ls_W_K": 2.0,
           "input_pump_power_kW": 0.05, "input_pump_flow_m3_h": 0.5, "output_pump_power_kW": 0.05,
           "output_pump_flow_m3_h": 0.5, "operation_mode": mode, "ambient_temperature_C": 25.0}
    res = CoolingChainCalculator(_gen(0.8), storage=cfg).run_timeseries(_loads())
    sto = res.storage.timeseries
    assert res.converged and res.iterations <= 2
    assert (sto["Q_C_sto_in_kWh"] <= 0.8 + 1e-9).all()
    assert res.generation.summary["QC_gen_unmet_kWh"] == pytest.approx(0.0, abs=1e-9)
    # balance of the generator request (7), (8), pump heat, direct supply and charging
    np.testing.assert_allclose(
        sto["Q_C_sto_in_kWh"],
        sto["Q_C_sto_direct_kWh"] + sto["Q_C_sto_chg_kWh"] + sto["Q_C_sto_out_ls_kWh"]
        + sto["Q_C_sto_in_ls_kWh"] + sto["W_C_sto_aux_kWh"], atol=1e-12)
    # users, store and shortfall add up to the demand
    np.testing.assert_allclose(
        sto["Q_C_sto_dis_kWh"] + sto["Q_C_sto_direct_kWh"] + sto["Q_C_sto_unmet_kWh"], sto["Q_C_sto_out_kWh"])


# --------------------------------------------------------------- temperatures back to the distribution
def _hot_loads():
    idx = pd.date_range("2026-07-01", periods=4, freq="h")
    return pd.DataFrame({"T_ext": 30.0, "T_op": 26.0, "T_C_source_C": 30.0,
                         "Q_C_kWh": [2.0, 4.0, 6.0, 0.0]}, index=idx)


HOT_GEN = {"nominal_capacity_kW": 10.0, "nominal_eer": 3.0, "theta_C_gen_out_limit_C": 12.0}


def test_actual_generator_temperature_reaches_the_distribution():
    """Required 7 C, generator not below 12 C: EN 16798-9 reports 12 C (23) and its return (24)."""
    res = CoolingChainCalculator(HOT_GEN, distribution={"theta_C_dis_flw_set_C": 7.0}).run_timeseries(_hot_loads())
    ts = res.distribution.timeseries
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_req_C"], 7.0)
    np.testing.assert_allclose(ts["theta_C_dis_in_flw_C"], 12.0)
    np.testing.assert_allclose(ts["theta_C_dis_out_ret_req_C"], [17.0, 17.0, 17.0, 12.0])   # deltaT 5 K
    assert res.summary["h_C_dis_flw_not_reached"] == 3.0
    assert res.summary["theta_C_dis_flw_dev_max_K"] == pytest.approx(5.0)
    assert res.summary["QC_users_unmet_kWh"] == 0.0          # no terminal data: energy only


def test_terminals_turn_a_warm_supply_into_a_temperature_shortfall():
    """Terminal 6 kW at 7 C / 26 C, n = 1: at 12 C it gives 6 x (26 - 12 - 2.5) / (26 - 7 - 2.5) kW."""
    dist = {"theta_C_dis_flw_set_C": 7.0, "terminal_design_capacity_kW": 6.0}
    res = CoolingChainCalculator(HOT_GEN, distribution=dist).run_timeseries(_hot_loads())
    cap = 6.0 * 11.5 / 16.5
    d = res.delivered
    np.testing.assert_allclose(d["Q_C_dis_unmet_temp_kWh_main"], [0.0, 0.0, 6.0 - cap, 0.0])
    np.testing.assert_allclose(d["Q_C_dis_unmet_gen_kWh_main"], 0.0)
    np.testing.assert_allclose(d["Q_C_dis_in_kWh_main"], [2.0, 4.0, cap, 0.0])
    # the generator only produces what the terminals can deliver
    np.testing.assert_allclose(res.generation.timeseries["Q_C_gen_in_kWh"], [2.0, 4.0, cap, 0.0])
    assert res.converged and res.outer_iterations == 2


def test_temperature_and_energy_shortfalls_are_separated():
    gen = dict(HOT_GEN, nominal_capacity_kW=3.0)
    dist = {"theta_C_dis_flw_set_C": 7.0, "terminal_design_capacity_kW": 6.0}
    res = CoolingChainCalculator(gen, distribution=dist).run_timeseries(_hot_loads())
    cap = 6.0 * 11.5 / 16.5
    d = res.delivered
    np.testing.assert_allclose(d["Q_C_dis_unmet_temp_kWh_main"], [0.0, 0.0, 6.0 - cap, 0.0])
    np.testing.assert_allclose(d["Q_C_dis_unmet_gen_kWh_main"], [0.0, 1.0, cap - 3.0, 0.0])
    np.testing.assert_allclose(d["Q_C_dis_in_kWh_main"], [2.0, 3.0, 3.0, 0.0])
    assert res.summary["QC_users_unmet_kWh"] == pytest.approx(res.summary["QC_users_unmet_temp_kWh"]
                                                              + res.summary["QC_users_unmet_gen_kWh"])


def test_terminal_limit_per_circuit_with_its_own_requirement():
    """Only the zone circuit has terminals; the AHU circuit asks for 12 C and is not limited."""
    circuits = [dict(CIRCUITS[0], terminal_design_capacity_kW=1.5),
                dict(CIRCUITS[1], theta_C_dis_flw_set_C=12.0)]
    res = CoolingChainCalculator(_gen(10.0, theta_C_gen_out_limit_C=10.0),
                                 distribution={"circuits": circuits}).run_timeseries(_circuit_loads())
    d = res.delivered
    day = DAY_HOURS
    cap = 1.5 * (26.0 - 10.0 - 2.5) / (26.0 - 7.0 - 2.5)          # zone circuit at 10 C
    np.testing.assert_allclose(d.loc[day, "Q_C_dis_in_kWh_zones"], cap)
    np.testing.assert_allclose(d.loc[day, "Q_C_dis_unmet_temp_kWh_ahu"], 0.0)
    np.testing.assert_allclose(d.loc[day, "Q_C_del_kWh_Q_z1"], cap * 1.5 / 2.0)    # zones pro rata
    np.testing.assert_allclose(res.distribution.timeseries.loc[day, "theta_C_dis_in_flw_C_ahu"], 12.0)


def test_storage_outlet_temperature_reaches_the_distribution():
    """Generator limited to 12 C: once the initial 7 C water is used, the tank only holds 12 C water."""
    tank = {**TANK, "theta_C_sto_return_C": 17.0}
    res = CoolingChainCalculator(_gen(3.0, theta_C_gen_out_limit_C=12.0),
                                 distribution={"theta_C_dis_flw_set_C": 7.0}, storage=tank).run_timeseries(_loads())
    ts = res.distribution.timeseries
    sto = res.storage.timeseries
    day2 = (IDX.day == 2) & DAY_HOURS & (sto["Q_C_sto_dis_kWh"] > 1e-9).to_numpy()
    assert day2.any()
    np.testing.assert_allclose(ts.loc[day2, "theta_C_dis_in_flw_C"], sto.loc[day2, "theta_C_sto_out_act_C"])
    assert (ts.loc[day2, "theta_C_dis_in_flw_C"] >= 12.0 - 1e-9).all()


def test_non_convergence_is_reported():
    gen = _gen(1.0, capacity_temperature_correction=True)
    with pytest.warns(RuntimeWarning, match="no convergence"):
        res = CoolingChainCalculator(gen, storage=TANK, max_iterations=1).run_timeseries(_loads())
    assert not res.converged
    assert res.summary["converged"] == 0.0
    assert res.history[-1] > 0.0


def test_generators_with_different_outlet_limits():
    """(4) per generator: priority 1 reaches 7 C, priority 2 only 10 C; the outlet is the energy mix."""
    gens = {"generators": [{"name": "a", "priority": 1, **_gen(3.0, theta_C_gen_out_limit_C=5.0)},
                           {"name": "b", "priority": 2, **_gen(5.0, theta_C_gen_out_limit_C=10.0)}]}
    from pybuildingenergy.source.cooling_generation_16798_13 import CoolingGenerationSystemCalculator
    calc = CoolingGenerationSystemCalculator(gens)
    assert calc.minimum_outlet_temperature_C == 5.0
    loads = pd.DataFrame({"T_ext": 30.0, "T_C_source_C": 30.0, "T_C_sink_C": 7.0,
                          "Q_C_kWh": [2.0, 6.0, 0.0]}, index=IDX[:3])
    ts = calc.run_timeseries(loads).timeseries
    np.testing.assert_allclose(ts["theta_C_gen_out_C_b"], 10.0)
    np.testing.assert_allclose(ts["theta_C_gen_out_C"], [7.0, (3 * 7.0 + 3 * 10.0) / 6.0, 7.0])
    res = CoolingChainCalculator(gens, distribution={"theta_C_dis_flw_set_C": 7.0}).run_timeseries(loads)
    np.testing.assert_allclose(res.distribution.timeseries["theta_C_dis_in_flw_C"], [7.0, 8.5, 7.0])
