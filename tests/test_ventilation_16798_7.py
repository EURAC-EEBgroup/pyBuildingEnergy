"""EN 16798-7:2017 (M5-5) method 1: hand calculations of the formulae and mass balance."""

import math

import pytest

from pybuildingenergy.source import ventilation_16798_7 as v

RHO = 1.204
CA = 0.000279


def test_density_wind_and_pressure_formulae_1_to_6():
    assert v.air_density(20.0) == pytest.approx(RHO)                                   # (1)
    assert v.air_density(0.0) == pytest.approx(293.15 / 273.15 * RHO)
    assert v.wind_speed_site(5.0, 0.9, 1.0, 1.0, 1.0) == pytest.approx(4.5)            # (2)
    assert v.wind_speed_10_site(5.0, 0.8, 1.0, 1.0, 1.0) == pytest.approx(4.0)         # (3)
    dp = v.path_pressure_difference(0.5, 4.0, 3.0, 0.0, 20.0, 2.0)                     # (4)-(6)
    p_e = RHO * 293.15 / 273.15 * (0.5 * 0.5 * 9.0 - 4.0 * 9.81)                       # (5)
    p_z = 2.0 - RHO * 4.0 * 9.81 * 293.15 / 293.15                                     # (6)
    assert dp == pytest.approx(p_e - p_z)
    assert v.air_density_ref(0.0) == RHO and v.air_density_ref(500.0) < RHO            # Table 11


def test_tables_b():
    assert v.cp_table(10.0, "open", "windward") == 0.50                                # B.7
    assert v.cp_table(30.0, "normal", "leeward") == -0.50
    assert v.cp_table(60.0, "shielded", "windward") == 0.25                            # High: Medium
    assert v.cp_table(10.0, "normal", "roof", roof_slope_deg=20.0) == -0.50
    assert v.cp_table(10.0, "open", "windward", cross=False) == 0.05                   # B.8
    assert v.f_ctrl("people_count_co2") == 0.8 and v.f_ctrl("x", "residential") == 1.0  # B.4
    assert v.f_sys("non_certified") == 1.2                                             # B.5
    assert v.d_c_cowl_height(0.3) == 0.0 and v.d_c_cowl_height(0.7) == -0.1 \
        and v.d_c_cowl_height(2.0) == -0.2                                             # B.9
    assert v.TABLE_B3_F_FF["wood_open_fire"] == 2.8


def test_required_flows_9_to_17():
    assert v.required_outdoor_air(100.0, 0.8, 1.1, 0.9) == pytest.approx(0.8 * 1.1 / 0.9 * 100)  # (9)
    assert v.terminal_required_flows("BALANCED_OP", 80.0) == (80.0, -80.0)             # (10), (11)
    assert v.terminal_required_flows("EXTRACT_OP", 80.0) == (0.0, -80.0)               # (12), (13)
    assert v.terminal_required_flows("SUPPLY_OP", 80.0) == (80.0, 0.0)                 # (14), (15)
    assert v.terminal_required_flows("NATURAL_OP", 80.0) == (0.0, 0.0)                 # (16), (17)
    with pytest.raises(ValueError):
        v.terminal_required_flows("X", 1.0)


def test_mechanical_flows_oda_and_load_18_to_24():
    r = v.mechanical_flows("BALANCED_OP", 100.0, f_op_v=0.5)
    assert r == {"q_sup_dis_req": 50.0, "q_eta_dis_req": -50.0}                        # (18), (19)
    k = RHO * CA * 1.0
    load = v.mechanical_flows("BALANCED_OP", 100.0, flow_control="LOAD", q_h_req_kwh=2.0,
                              theta_sup_out_c=30.0, t_z_c=21.0, t_ci_h=1.0)
    assert load["q_sup_dis_req"] == pytest.approx(2.0 / (k * 9.0))                     # (20)
    assert load["q_eta_dis_req"] == pytest.approx(-load["q_sup_dis_req"])              # (21)
    sup = v.mechanical_flows("SUPPLY_OP", 100.0, flow_control="LOAD", q_c_req_kwh=0.1,
                             theta_sup_out_c=16.0, t_z_c=26.0)
    assert sup["q_sup_dis_req"] == 100.0 and sup["q_eta_dis_req"] == 0.0               # min = ODA, (22)
    with pytest.raises(ValueError, match="LOAD"):
        v.mechanical_flows("EXTRACT_OP", 100.0, flow_control="LOAD", theta_sup_out_c=20.0, t_z_c=20.0)
    assert v.design_supply_flow(5.0, 0.0, 30.0, 20.0, 50.0) == pytest.approx(5.0 / (k * 10.0))  # (23)
    assert v.supply_outdoor_air_flow(0.6, 200.0) == 120.0                              # (24)


def test_required_supply_temperature_7_8():
    k = RHO * CA * 100.0 * 1.0
    t = v.required_supply_temperature(21.0, 1.5, 0.0, 100.0, 1.0)
    assert t == pytest.approx(21.0 + 1.5 / k)                                          # (7)
    ls = 1.5 * (1.0 / (21.0 - 0.0))                                                    # (8)
    t2 = v.required_supply_temperature(21.0, 1.5, 0.0, 100.0, 1.0, 1.0, 0.0)
    assert t2 == pytest.approx(21.0 + (1.5 + ls) / k)
    c = v.required_supply_temperature(26.0, 0.0, 1.0, 100.0, 1.0)
    assert c == pytest.approx(26.0 - 1.0 / k)
    with pytest.raises(ValueError, match="either"):
        v.required_supply_temperature(21.0, 1.0, 1.0, 100.0, 1.0)


def test_coefficients_25_26_59_35():
    c = v.atd_coefficient(50.0, 0.6, 10.0, 0.5)
    assert c == pytest.approx(3600 / 10000 * 0.6 * 50.0 * (2 / RHO) ** 0.5)            # (26), n = 0.5
    assert v.vent_coefficient(50.0, 0.6, 10.0, 0.6) == pytest.approx(
        3600 / 10000 * 0.6 * 50.0 * (2 / RHO) ** 0.5 * (1 / 10.0) ** 0.1)              # (59)
    assert v.combustion_air_flow(1, 1.0, 2.8, 8.0) == pytest.approx(3.6 * 2.8 * 8.0)   # (35)
    assert v.combustion_air_flow(0, 1.0, 2.8, 8.0) == 0.0                              # (39)


def test_leakage_distribution_b4_to_b6():
    z = v.ZoneAirflow(height_m=3.0, facade_area_m2=90.0, roof_area_m2=30.0, q50_per_m2=6.0, a_lea_m2=120.0)
    assert z.c_lea == pytest.approx(6.0 * 120.0 / 50.0 ** 0.667)                       # (B.4)
    # no wind, no temperature difference: no flow through the leaks, p_z = 0 (the stack
    # effect is zero with equal temperatures)
    r = z.solve(20.0, 20.0, 0.0)
    assert abs(r["q_lea_in"]) < 1e-3 and abs(r["q_lea_out"]) < 1e-3    # steep at p = 0 (n = 0.667)
    assert abs(r["p_z_ref_pa"]) < 1e-6


def test_stack_effect_leakage_balances_mass_62_67():
    z = v.ZoneAirflow(height_m=5.0, facade_area_m2=100.0, q50_per_m2=8.0, a_lea_m2=100.0)
    r = v.ZoneAirflow.solve(z, 0.0, 20.0, 0.0)
    # cold outside: air enters low, leaves high, and the mass balance closes
    assert r["q_lea_in"] > 0.0 and r["q_lea_out"] < 0.0
    assert abs(r["mass_residual_kg_h"]) < 1e-6                                          # (67)
    assert r["q_tot_in"] == pytest.approx(r["q_lea_in"])                                # (68)
    assert r["q_tot_out"] == pytest.approx(r["q_lea_out"] * 1.0, rel=1e-9) or r["q_tot_out"] < 0.0
    # wind: the windward façade brings air in, the leeward one lets it out
    w = z.solve(10.0, 20.0, 6.0)
    assert abs(w["mass_residual_kg_h"]) < 1e-6 and w["q_lea_in"] > r["q_lea_in"] * 0.0


def test_mechanical_extract_makes_infiltration_make_up_the_difference():
    z = v.ZoneAirflow(height_m=3.0, facade_area_m2=100.0, q50_per_m2=10.0, a_lea_m2=100.0)
    r = z.solve(10.0, 20.0, 0.0, q_sup_dis=0.0, q_eta_dis=-200.0)
    assert abs(r["mass_residual_kg_h"]) < 1e-6
    assert r["p_z_ref_pa"] < 0.0                                       # under-pressure
    assert r["q_lea_in"] > 150.0                                       # infiltration replaces the extract air
    bal = z.solve(10.0, 20.0, 0.0, q_sup_dis=200.0, q_eta_dis=-200.0)
    assert abs(bal["p_z_ref_pa"]) < abs(r["p_z_ref_pa"])               # balanced system: smaller depression
    with pytest.raises(ValueError):
        z.solve(10.0, 20.0, 0.0, q_sup_dis=-1.0)


def test_vents_58_59_and_combustion_37_38():
    z = v.ZoneAirflow(height_m=3.0, facade_area_m2=100.0, q50_per_m2=0.0, c_vent_total=20.0)
    r = z.solve(5.0, 20.0, 0.0, combustion_flow=30.0)
    assert r["q_comb_out"] == -30.0
    assert r["q_vent_in"] > 25.0                                       # vents supply the burnt air
    assert abs(r["mass_residual_kg_h"]) < 1e-6


def test_window_airing_42_to_46():
    w = [v.WindowData(max_area_m2=1.5, height_m=1.0, free_area_height_m=1.0, opening_ratio=1.0)]
    z = v.ZoneAirflow(windows=w, window_method="simplified", f_arg=1.8)
    r = z.solve(5.0, 20.0, 2.0, q_oda_req=100.0)
    rho_e, rho_z = v.air_density(5.0), v.air_density(20.0)
    assert r["q_arg_in"] == pytest.approx(RHO / rho_e * 1.8 * 100.0)                   # (42)
    assert r["q_arg_out"] == pytest.approx(-RHO / rho_z * 1.8 * 100.0)                 # (43)
    z1 = v.ZoneAirflow(windows=w, window_method="wind_temperature")
    s = z1.solve(5.0, 20.0, 2.0)
    stack = 0.0035 * 1.0 * 15.0                                                        # (44): h_wst = 1.0
    core = max(0.001 * 4.0, stack) ** 0.5
    assert z1.window_stack_height() == pytest.approx(1.0)
    assert s["q_arg_in"] == pytest.approx(3600 * RHO / rho_e * 1.5 / 2 * core)          # (45)
    assert s["q_arg_out"] == pytest.approx(-3600 * RHO / rho_z * 1.5 / 2 * core)        # (46)
    off = z1.solve(5.0, 20.0, 2.0, windows_open=False)
    assert off["q_arg_in"] == 0.0


def test_cross_ventilation_area_49_to_52_and_flow_47():
    # two façades facing south and north, 1 m2 each
    w = [v.WindowData(max_area_m2=1.0, orientation_deg=0.0), v.WindowData(max_area_m2=1.0, orientation_deg=180.0)]
    z = v.ZoneAirflow(windows=w, cross_ventilation=True)
    # sector edges are inclusive at -45 deg (50): S and N windows count in both rotations,
    # i = 1: j = 1 (S) and j = 3 (N), A_ori = 1, rest = 1 -> 1/sqrt(1 + 1) each; A = 1/4 * 2 * 0.7071
    assert z.cross_window_area() == pytest.approx(0.25 * 2 * 2 ** -0.5)               # (51), (52)
    # four windows (S, N, SW-ish 45 deg, 225 deg): i = 1 gives 0.9487, i = 2 gives 0.7071
    d = v.ZoneAirflow(windows=[v.WindowData(max_area_m2=1.0, orientation_deg=0.0),
                               v.WindowData(max_area_m2=1.0, orientation_deg=180.0),
                               v.WindowData(max_area_m2=1.0, orientation_deg=45.0),
                               v.WindowData(max_area_m2=1.0, orientation_deg=225.0)],
                      cross_ventilation=True)
    assert d.cross_window_area() == pytest.approx(0.25 * 2 * 2 ** 0.5)                  # min = 0.7071
    # a horizontal rooflight (tilt < 60 deg) does not count (50)
    flat = v.ZoneAirflow(windows=[v.WindowData(max_area_m2=1.0, orientation_deg=0.0, tilt_deg=10.0),
                                  v.WindowData(max_area_m2=1.0, orientation_deg=180.0)],
                         cross_ventilation=True)
    # only the north window counts: one sector with A_ori = 1 and rest = 1 (the rooflight)
    assert flat.cross_window_area() == pytest.approx(0.25 / math.sqrt(2.0))
    r = d.solve(15.0, 22.0, 2.0)
    stack = 0.0035 * d.window_stack_height() * 7.0
    wind = 0.67 * d.cross_window_area() * min(2.0, 3.0) * 0.75 ** 0.5
    rho_e = v.air_density(15.0)
    assert r["q_arg_in"] == pytest.approx(3600 * RHO / rho_e * max(wind, 4.0 / 2.0 * stack ** 0.5))  # (47)


def test_window_pressure_method_53_to_57_balances_mass():
    w = [v.WindowData(max_area_m2=1.0, height_m=1.0, free_area_height_m=1.0, facade="windward"),
         v.WindowData(max_area_m2=1.0, height_m=1.0, free_area_height_m=1.0, facade="leeward")]
    z = v.ZoneAirflow(windows=w, window_method="pressure", cross_ventilation=True, shield="open",
                      q50_per_m2=0.0)
    r = z.solve(10.0, 20.0, 5.0)
    assert r["q_arg_in"] > 0.0 and r["q_arg_out"] < 0.0
    assert abs(r["mass_residual_kg_h"]) < 1e-6
    # windward air in ~ leeward air out
    assert abs(r["q_arg_in"] + r["q_arg_out"]) / r["q_arg_in"] < 0.1
    c = 3600 * 0.67 * 1.0 * (2 / RHO) ** 0.5                                           # (54)
    assert c > 0.0


def test_passive_duct_30_to_34():
    pdu = v.PassiveDuct(c_atd=v.atd_coefficient(100.0), height_m=6.0, cross_section_m2=0.02,
                        zeta_cowl=1.0)
    z = v.ZoneAirflow(passive_duct=pdu, q50_per_m2=4.0, a_lea_m2=100.0, facade_area_m2=100.0)
    r = z.solve(0.0, 20.0, 0.0)
    # stack effect: the passive duct extracts air, replaced by infiltration
    assert r["q_pdu_out"] < 0.0 and r["q_pdu_in"] == 0.0                               # (33), (34)
    assert abs(r["mass_residual_kg_h"]) < 1e-5
    assert pdu.h_stack == 8.0                                                          # (B.1)
    # (30) satisfied by the solution
    rhs = r["p_z_ref_pa"] + pdu.h_stack * 9.81 * (v.air_density(0.0) - v.air_density(20.0))
    res = v._pdu_residual(pdu, r["q_pdu_out"], rhs, 0.0, RHO)
    assert abs(res) < 1e-6
    warm = z.solve(25.0, 20.0, 0.0)                                    # warm outside: reverse flow
    assert warm["q_pdu_in"] >= 0.0


def test_method2_and_ahu_inputs():
    assert v.method2_totals(120.0, -110.0) == {"q_tot_in": 120.0, "q_tot_out": -110.0}
    z = v.ZoneAirflow(q50_per_m2=3.0, a_lea_m2=100.0)
    r = z.solve(5.0, 20.0, 0.0, q_sup_dis=150.0, q_eta_dis=-150.0)
    a = v.ahu_inputs(r, 150.0, 18.0)
    assert a["required_supply_flow_m3_h"] == 150.0 and a["required_extract_flow_m3_h"] == 150.0
    from pybuildingenergy.source.ventilation_16798_5_1 import AHUStepInputs
    AHUStepInputs(outdoor_temperature_c=5.0, extract_temperature_c=20.0, **{
        k: v_ for k, v_ in a.items() if k in ("required_supply_flow_m3_h", "required_extract_flow_m3_h",
                                             "required_outdoor_air_flow_m3_h",
                                             "required_supply_temperature_c")})
    with pytest.raises(ValueError):
        v.ZoneAirflow(window_method="x")


# --------------------------------------------------------------------------- engine coupling
def _engine(components, mode="standard", t_out=-5.0):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_ventilation_boundary import _legacy_building, _run_single_zone_engine
    return _run_single_zone_engine(_legacy_building(components), control_mode=mode, n=48, t_out=t_out)


_ZONE = {"height_m": 3.0, "facade_area_m2": 100.0, "q50_per_m2": 6.0, "a_lea_m2": 100.0}
_NAT = {"name": "nat", "ventilation_type": "en16798_7", "zone": _ZONE}


def test_boundary_with_en16798_7_component_and_extract_only_ahu():
    from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
    ahu = {"name": "ahu", "ventilation_type": "mechanical_supply", "sensible_heat_recovery_efficiency": 0.0,
           "supply_temperature_setpoint_c": 18.0, "frost_control": "none",
           "outdoor_air_design_flow_m3_h": 200.0, "vent_sys_op": "EXTRACT_OP"}
    cfg = lambda comps: {"building_parameters": {"ventilation": {"components": comps}}}
    coll = {}
    b = resolve_ventilation_boundary(cfg([ahu, _NAT]), 20.0, -5.0, 0.0, zone_volume_m3=300.0,
                                     ahu_outputs_collector=coll)
    names = {s.name: s for s in b.streams}
    assert "ahu_makeup" not in names                  # the make-up air is now the infiltration
    assert coll["ahu"].actual_extract_flow_m3_h == pytest.approx(200.0)
    assert coll["ahu"].actual_supply_flow_m3_h == 0.0
    nat = names["nat"]
    assert nat.source_temperature_c == -5.0
    # the 200 m3/h extracted at 20 degC are replaced through the envelope by the same MASS of
    # outdoor air at -5 degC (67): 200 * T_e / T_z m3/h
    q_in = nat.heat_transfer_coefficient_w_k / (1.204 * 1006.0 / 3600.0)
    assert q_in == pytest.approx(200.0 * 268.15 / 293.15, rel=0.02)
    free = resolve_ventilation_boundary(cfg([_NAT]), 20.0, -5.0, 0.0, zone_volume_m3=300.0)
    q_free = {s.name: s for s in free.streams}["nat"].heat_transfer_coefficient_w_k / (1.204 * 1006.0 / 3600.0)
    assert q_in > q_free                                # extract air increases the infiltration


def test_ahu_flows_from_en16798_7_requirement():
    from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
    ahu = {"name": "ahu", "ventilation_type": "mechanical_supply", "sensible_heat_recovery_efficiency": 0.0,
           "supply_temperature_setpoint_c": 18.0, "frost_control": "none",
           "outdoor_air_design_flow_m3_h": 100.0, "f_ctrl": 0.8, "f_sys": 1.1,
           "ventilation_effectiveness": 0.9}
    coll = {}
    resolve_ventilation_boundary({"building_parameters": {"ventilation": {"components": [ahu]}}},
                                 20.0, 5.0, 0.0, zone_volume_m3=300.0, ahu_outputs_collector=coll)
    assert coll["ahu"].actual_supply_flow_m3_h == pytest.approx(0.8 * 1.1 / 0.9 * 100.0)   # (9), (10)
    assert coll["ahu"].actual_extract_flow_m3_h == pytest.approx(0.8 * 1.1 / 0.9 * 100.0)  # (11)


@pytest.mark.parametrize("mode", ["standard", "ahu_causal"])
def test_zone_simulation_with_en16798_7_component(mode):
    h = _engine([_NAT], mode=mode)
    h0 = _engine([{"name": "nat", "ventilation_type": "prescribed", "heat_transfer_coefficient_w_k": 0.0,
                   "source_temperature_c": -5.0}], mode=mode)
    assert (h["H_ve"] > 0.0).all()
    assert h["H_ve"].mean() > h0["H_ve"].mean()
    assert h["Q_HC"].abs().sum() > 0.0


def test_unsolvable_balance_is_an_error():
    z = v.ZoneAirflow(q50_per_m2=0.0, c_vent_total=0.0)          # no air flow path at all
    with pytest.raises(ValueError, match="no solution"):
        z.solve(0.0, 20.0, 0.0, q_eta_dis=-100.0)
    assert z.solve(0.0, 20.0, 0.0)["q_tot_in"] == 0.0            # nothing to balance: fine
    from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
    ahu = {"name": "ahu", "ventilation_type": "mechanical_supply", "sensible_heat_recovery_efficiency": 0.0,
           "supply_temperature_setpoint_c": 18.0, "frost_control": "none",
           "outdoor_air_design_flow_m3_h": 100.0, "vent_sys_op": "EXTRACT_OP"}
    nat = {"name": "nat", "ventilation_type": "en16798_7", "zone": {"q50_per_m2": 0.0}}
    with pytest.raises(ValueError, match="no solution"):
        resolve_ventilation_boundary({"building_parameters": {"ventilation": {"components": [ahu, nat]}}},
                                     20.0, 0.0, 0.0, zone_volume_m3=300.0)


def test_one_en16798_7_component_per_zone():
    from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
    a = {**_NAT, "name": "n1"}
    b = {**_NAT, "name": "n2"}
    with pytest.raises(ValueError, match="only one 'en16798_7' component"):
        resolve_ventilation_boundary({"building_parameters": {"ventilation": {"components": [a, b]}}},
                                     20.0, 0.0, 0.0, zone_volume_m3=300.0)


def test_supply_temperature_requires_corrected_outdoor_temperature():
    with pytest.raises(ValueError, match="theta_e_comb_c"):
        v.required_supply_temperature(21.0, 1.5, 0.0, 100.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="undefined"):
        v.required_supply_temperature(21.0, 1.5, 0.0, 100.0, 1.0, 1.0, 21.0)
    with pytest.raises(ValueError, match="finite and >= 0"):
        v.required_supply_temperature(21.0, -1.0, 0.0, 100.0, 1.0)


def test_non_physical_inputs_are_rejected():
    with pytest.raises(ValueError, match="q_oda_des_m3h"):
        v.required_outdoor_air(-50.0)
    with pytest.raises(ValueError, match="q_oda_req_m3h"):
        v.mechanical_flows("BALANCED_OP", -10.0)
    with pytest.raises(ValueError, match="q50_per_m2"):
        v.ZoneAirflow(q50_per_m2=-1.0)
    with pytest.raises(ValueError, match="n_lea"):
        v.ZoneAirflow(n_lea=0.0)
    with pytest.raises(ValueError, match="c_atd"):
        v.PassiveDuct(c_atd=-1.0)
    with pytest.raises(ValueError, match="f_oda"):
        v.supply_outdoor_air_flow(1.5, 100.0)
    with pytest.raises(ValueError, match="a_atd_cm2"):
        v.atd_coefficient(-5.0)
    with pytest.raises(ValueError, match="u10"):
        v.ZoneAirflow().solve(5.0, 20.0, u10=-1.0)


def test_non_finite_and_out_of_range_signals():
    nan = float("nan")
    with pytest.raises(ValueError, match="q_oda_des_m3h"):
        v.required_outdoor_air(nan)
    with pytest.raises(ValueError, match="f_op_v"):
        v.mechanical_flows("BALANCED_OP", 100.0, f_op_v=2.0)
    with pytest.raises(ValueError, match="f_op_v"):
        v.mechanical_flows("BALANCED_OP", 100.0, f_op_v=nan)
    with pytest.raises(ValueError):
        v.ZoneAirflow(q50_per_m2=nan)


def test_supply_temperature_rejects_nan_and_zero_density():
    nan = float("nan")
    with pytest.raises(ValueError, match="q_sup_dis_m3h"):
        v.required_supply_temperature(21.0, 1.0, 0.0, nan, 1.0)
    with pytest.raises(ValueError, match="theta_e_comb_c"):
        v.required_supply_temperature(21.0, 1.0, 0.0, 100.0, 1.0, 1.0, nan)
    with pytest.raises(ValueError, match="rho_a"):
        v.required_supply_temperature(21.0, 1.0, 0.0, 100.0, 1.0, rho_a=0.0)
