"""prEN 15316-5:2024 Method A (multi-volume storage), 6.4.3, and its coupling to the
EN 15316-4-3 Method 3 solar iteration."""

import numpy as np
import pandas as pd
import pytest

import pybuildingenergy as pybui
from pybuildingenergy.source.storage_15316_5 import LayeredStorage, _rearrange_layers

KWH_PER_L_K = 4186.0 / 3600.0 * 1000.0 / 1e6  # rho*cp, kWh/(l K)


def test_rearrange_mixes_inverted_layers_by_volume():
    t = _rearrange_layers(np.array([60.0, 40.0, 30.0]), np.array([100.0, 100.0, 100.0]))
    assert t == pytest.approx([43.3333333] * 3)
    # volume weighting, and an already stratified profile is left alone
    assert _rearrange_layers(np.array([50.0, 20.0]), np.array([100.0, 300.0])) == pytest.approx([27.5, 27.5])
    assert _rearrange_layers(np.array([10.0, 30.0, 60.0]), np.ones(3)) == pytest.approx([10.0, 30.0, 60.0])


def test_dhw_draw_off_pushes_layers_up_eq_9_to_22():
    # 4 x 50 l, hot (60) in the top two layers; ask for exactly one hot layer's energy
    one_layer = KWH_PER_L_K * 50.0 * (60.0 - 10.0)
    st = LayeredStorage([50] * 4, cold_water_C=10.0, initial_temperatures_C=[10, 10, 60, 60],
                        dhw_min_out_C=45.0, dhw_demand_kWh=[one_layer], heater_layer=4)
    st(0, 0.0, True)
    row = st.history[0]
    assert row["V_W_withdrawn_l"] == pytest.approx(50.0)
    assert row["Q_W_sto_out_kWh"] == pytest.approx(one_layer)
    assert st.temps == pytest.approx([10, 10, 10, 60])


def test_draw_off_is_limited_to_usable_energy_and_reports_deficiency():
    st = LayeredStorage([100, 100], cold_water_C=10.0, initial_temperatures_C=[10.0, 60.0],
                        dhw_demand_kWh=[100.0], dhw_min_out_C=40.0, heater_layer=2)
    st(0, 0.0, True)
    usable = KWH_PER_L_K * 100.0 * 50.0
    row = st.history[0]
    assert row["Q_W_sto_out_kWh"] == pytest.approx(usable)
    assert row["Q_W_sto_nreq_kWh"] == pytest.approx(100.0 - usable)  # eq. 17


def test_energy_balance_with_solar_heater_draw_and_losses():
    st = LayeredStorage([100, 100, 100, 100], set_point_C=55.0, cold_water_C=10.0, ambient_C=20.0,
                        loss_W_K=2.0, heater_power_kW=3.0, heater_layer=3,
                        initial_temperatures_C=[10, 20, 30, 40],
                        dhw_demand_kWh=[1.0, 0.0, 2.0, 0.0], ambient_series_C=[20.0] * 4)
    e0 = st.stored_energy_kWh()
    for i, q_sol in enumerate([0.0, 1.5, 0.0, 0.8]):
        st(i, q_sol, True)
    h = pd.DataFrame(st.history)
    e1 = st.stored_energy_kWh()
    # change of stored energy = inputs - draw-off - losses (draw-off removes water at its
    # temperature and refills with cold water: the energy above cold water leaves)
    delta = (h["Q_solar_in_kWh"] + h["Q_backup_in_kWh"] - h["Q_W_sto_out_kWh"] - h["Q_sto_ls_kWh"]).sum()
    assert e1 - e0 == pytest.approx(delta, rel=1e-6)
    assert (np.diff(st.temps) >= -1e-9).all()  # stratified: never hotter below


def test_heater_only_runs_below_set_point_and_is_power_limited():
    st = LayeredStorage([100, 100], set_point_C=55.0, heater_power_kW=1.0, heater_layer=2,
                        initial_temperatures_C=[20.0, 30.0])
    st(0, 0.0, True)
    assert st.history[0]["Q_backup_in_kWh"] == pytest.approx(1.0)  # needs more than 1 kWh
    hot = LayeredStorage([100, 100], set_point_C=55.0, heater_power_kW=1.0, heater_layer=2,
                         initial_temperatures_C=[60.0, 60.0])
    hot(0, 0.0, True)
    assert hot.history[0]["Q_backup_in_kWh"] == 0.0


def test_solar_input_on_layer_1_and_exchanger_inlet_eq_32_and_44():
    st = LayeredStorage([100, 100], initial_temperatures_C=[20.0, 20.0], solar_exchanger_W_K=500.0,
                        solar_mass_flow_kg_s=0.1, heater_layer=2)
    theta_in, acc = st(0, 1.0, False)  # trial: no state change
    assert st.temps == pytest.approx([20.0, 20.0]) and acc == 1.0
    d_exh = 1.0 * 1000.0 / 500.0
    expected = max(20.0 + d_exh - 1.0 / (3.6 * 0.1 * 4186.0 / 3600.0), 20.0)
    assert theta_in == pytest.approx(expected)
    st(0, 1.0, True)
    # 1 kWh into layer 1 heats it by 1/(0.1163) K, then mixes with the cooler layer above
    assert st.history[0]["Q_solar_in_kWh"] == 1.0
    assert st.stored_energy_kWh(20.0) == pytest.approx(1.0)


def test_layered_storage_drives_the_method_3_iteration():
    I = [0.0] * 8 + [300.0, 500.0, 700.0, 850.0, 900.0, 850.0, 700.0, 500.0, 300.0] + [0.0] * 7
    data = pd.DataFrame({"I_sol_collector_plane_W_m2": I, "T_ext_C": 10.0, "E_site_el_load_kWh": 0.0})
    st = LayeredStorage([75] * 4, cold_water_C=10.0, initial_temperatures_C=[15, 15, 15, 15],
                        solar_mass_flow_kg_s=0.05, heater_layer=4)
    cfg = {"solar_thermal": {"enabled": True, "area_m2": 4.0, "collector_area_m2": 4.0,
                             "method": "method_3", "mass_flow_kg_s": 0.05,
                             "initial_loop_inlet_temperature_C": 15.0}}
    ts = pybui.RenewableEnergySystemCalculator(cfg, storage_response=st).run_timeseries(data).timeseries
    accepted = ts["Q_solar_thermal_available_kWh"].sum()
    assert accepted > 0.0
    assert len(st.history) == 24  # one committed step per hour
    assert st.stored_energy_kWh(15.0) == pytest.approx(accepted, rel=1e-6)
    assert ts["T_solar_loop_in_C"].iloc[16] > ts["T_solar_loop_in_C"].iloc[8]


def test_storage_calculator_method_a_runs_both_services_through_the_layered_tank():
    n = 48
    idx = pd.date_range("2021-03-01", periods=n, freq="h")
    q_w = np.where(idx.hour.isin([7, 8, 19, 20]), 1.5, 0.0)
    q_h = np.where(idx.hour.isin(range(6, 22)), 0.6, 0.0)
    sol = np.where(idx.hour.isin(range(10, 16)), 2.0, 0.0)
    data = pd.DataFrame({"Q_W_kWh": q_w, "Q_H_kWh": q_h, "Q_sol_to_sto_kWh": sol}, index=idx)
    calc = pybui.StorageSystemCalculator({
        "dhw": {"set_temperature_C": 55.0, "ambient_temperature_C": 16.0, "standby_loss_coefficient_W_K": 1.5},
        "method_a": {"layer_volumes_l": [100, 100, 100, 100], "heater_power_kW": 4.0,
                     "heater_layer": 3, "space_heating_layer": 3},
    })
    res = calc.run_timeseries(data)
    ts, summ = res.timeseries, res.summary
    assert {"T_layer_1_C", "T_layer_4_C", "Q_sto_solar_in_kWh", "Q_W_sto_delivered_kWh"} <= set(ts.columns)
    assert ts["Q_sto_solar_in_kWh"].sum() > 0.0
    # energy balance over the run: back-up + solar = delivered + losses + change of stored energy
    tank_in = ts["Q_W_sto_in_kWh"].sum() + ts["Q_H_sto_in_kWh"].sum() + ts["Q_sto_solar_in_kWh"].sum()
    delivered = ts["Q_W_sto_delivered_kWh"].sum() + ts["Q_H_sto_delivered_kWh"].sum()
    losses = ts["Q_W_sto_ls_kWh"].sum() + ts["Q_H_sto_ls_kWh"].sum()
    temps = np.array([ts[f"T_layer_{k}_C"].iloc[-1] for k in range(1, 5)])
    e_end = float(np.sum(KWH_PER_L_K * 100.0 * (temps - 10.0)))
    e_start = float(np.sum(KWH_PER_L_K * 100.0 * (np.array([10.0, 10.0, 55.0, 55.0]) - 10.0)))
    assert tank_in - delivered - losses == pytest.approx(e_end - e_start, abs=1e-6)
    assert summ["Q_sto_ls_kWh"] == pytest.approx(losses)


def test_storage_calculator_method_a_validates_its_input():
    with pytest.raises(ValueError, match="layer_volumes_l"):
        pybui.StorageSystemCalculator({"method_a": {"heater_power_kW": 1.0}})
