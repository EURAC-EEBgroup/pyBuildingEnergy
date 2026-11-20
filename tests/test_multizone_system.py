"""Multizone system topologies: central, per zone, combined H/C/DHW with tank dispatch."""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.multizone_system import dispatch_combined, simulate_multizone_system

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json"
EMISSION = ROOT / "examples" / "example_building_emission_only.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"
STORAGE = {"volume_l": 200.0, "set_temperature_C": 55.0, "min_temperature_C": 45.0,
           "cold_water_C": 10.0, "ambient_C": 20.0, "H_sto_ls_W_K": 1.5}


def _idx(n):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def _s(values):
    return pd.Series(np.asarray(values, dtype=float), index=_idx(len(values)))


# ------------------------------------------------------------------ dispatch (unit)
def test_tank_energy_balance_without_space_demand():
    n = 48
    draw = _s(np.tile([0, 0, 0, 0, 0, 0, 1.5, 1.5] + [0] * 16, 2))
    d = dispatch_combined(_s(np.zeros(n)), _s(np.zeros(n)), draw, 1.0, STORAGE,
                          {"capacity_heating_kW": 5.0})
    e0 = 200 * 0.001163 * 45.0
    lhs = d["Q_W_charge_kWh"].sum()
    rhs = draw.sum() + d["Q_sto_ls_kWh"].sum() + d["E_sto_kWh"].iloc[-1] - e0
    assert lhs == pytest.approx(rhs, abs=1e-9)
    assert d["Q_W_unmet_kWh"].sum() == 0.0
    assert d["theta_sto"].max() <= 55.0 + 1e-9


def test_dhw_priority_displaces_space_heating():
    # empty tank (initial 0) and a heating demand equal to the capacity
    st = dict(STORAGE, initial_fraction=0.0)
    d = dispatch_combined(_s([5.0, 5.0, 5.0]), _s([0, 0, 0]), _s([0, 0, 0]), 1.0, st,
                          {"capacity_heating_kW": 5.0})
    assert d["priority_active"].iloc[0] == 1.0
    assert d["Q_H_displaced_kWh"].iloc[0] > 0.0
    assert d["Q_H_unmet_kWh"].iloc[0] == pytest.approx(d["Q_H_displaced_kWh"].iloc[0])
    # without priority the heating is served and the tank waits
    d2 = dispatch_combined(_s([5.0, 5.0, 5.0]), _s([0, 0, 0]), _s([0, 0, 0]), 1.0, st,
                           {"capacity_heating_kW": 5.0, "dhw_priority": False})
    assert d2["Q_H_unmet_kWh"].sum() == 0.0
    assert d2["Q_W_charge_kWh"].sum() == 0.0


def test_cooling_time_share_versus_heat_recovery():
    st = dict(STORAGE, initial_fraction=0.5)
    qc = _s([4.0] * 6)
    draw = _s([1.0] * 6)
    share = dispatch_combined(_s([0] * 6), qc, draw, 1.0, st,
                              {"capacity_heating_kW": 5.0, "capacity_cooling_kW": 4.0})
    rec = dispatch_combined(_s([0] * 6), qc, draw, 1.0, st,
                            {"capacity_heating_kW": 5.0, "capacity_cooling_kW": 4.0,
                             "cooling_dhw": "heat_recovery"})
    # time share: charging during cooling hours costs cooling (capacity fully used)
    assert share["Q_C_unmet_kWh"].sum() > 0.0
    # heat recovery: the condenser charges the tank, cooling is fully served
    assert rec["Q_C_unmet_kWh"].sum() == pytest.approx(0.0)
    assert rec["Q_W_charge_recovery_kWh"].sum() > 0.0
    assert rec["Q_W_unmet_kWh"].sum() == 0.0


def test_simultaneous_heating_and_cooling_never_disappear():
    """One generator, 2 kWh heating and 3 kWh cooling in the same hour (two zones)."""
    full = dict(STORAGE, initial_fraction=1.0)
    caps = {"capacity_heating_kW": 4.0, "capacity_cooling_kW": 4.0, "capacity_dhw_kW": 4.0}
    for prio, first in (("heating", "H"), ("cooling", "C"), ("larger", "C")):
        d = dispatch_combined(_s([2.0]), _s([3.0]), _s([0.0]), 1.0, full,
                              dict(caps, space_priority=prio))
        r = d.iloc[0]
        # balance per service: served + unmet = demand
        assert r["Q_H_served_kWh"] + r["Q_H_unmet_kWh"] == pytest.approx(2.0)
        assert r["Q_C_served_kWh"] + r["Q_C_unmet_kWh"] == pytest.approx(3.0)
        assert r["simultaneous"] == 1.0 and r["first_service"] == first
        # time share: f_heating + f_cooling <= 1
        assert r["f_heating"] + r["f_cooling"] <= 1.0 + 1e-12
    # heating first: 2 kWh = half the hour, cooling gets the other half = 2 of 3 kWh
    d = dispatch_combined(_s([2.0]), _s([3.0]), _s([0.0]), 1.0, full, caps)
    assert d["Q_H_served_kWh"].iloc[0] == pytest.approx(2.0)
    assert d["Q_C_served_kWh"].iloc[0] == pytest.approx(2.0)
    assert d["Q_C_unmet_kWh"].iloc[0] == pytest.approx(1.0)
    assert d["Q_C_displaced_by_heating_kWh"].iloc[0] == pytest.approx(1.0)


def test_undersized_generator_leaves_dhw_unmet():
    d = dispatch_combined(_s([0] * 5), _s([0] * 5), _s([3.0] * 5), 1.0,
                          dict(STORAGE, initial_fraction=0.0), {"capacity_heating_kW": 1.0})
    assert d["Q_W_unmet_kWh"].sum() > 5.0


# -------------------------------------------------------------- topologies (engine)
pytestmark_engine = pytest.mark.skipif(
    not (CONFIG.exists() and EMISSION.exists() and WEATHER.exists()), reason="inputs missing")


def _config():
    cfg = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    cfg["weather"] = {"source": "epw", "file": str(WEATHER)}
    for z in cfg["building"]["zones"]:
        z["cooling_capacity"] = 1e6
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
    return cfg


def _emitter():
    em = copy.deepcopy(json.loads(EMISSION.read_text(encoding="utf-8"))["systems"]["emission"]["config"])
    for svc in ("heating", "cooling"):
        em.setdefault(svc, {}).pop("nominal_power_kW", None)      # unlimited: no capacity effect
    return em


IDEAL = {"kind": "ideal", "config": {"cop_heating": 3.0, "cop_dhw": 2.5, "eer": 3.5}}
DIST = {"heating": {"operation_mode": "demand", "nominal_power_kW": 30.0, "design_deltaT_K": 10.0,
                    "supply_temperature_C": 40.0, "return_temperature_C": 30.0,
                    "pipe_sections": [{"length_m": 40.0, "equivalent_length_m": 8.0,
                                       "linear_thermal_transmittance_W_mK": 0.25,
                                       "ambient_temperature_C": 15.0, "recoverable": False}],
                    "pump_control_code": 4, "eei": 0.23}}


def _run(system, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulate_multizone_system(_config(), system, emission_inc="approximate", **kw)


@pytestmark_engine
def test_central_topology_sums_the_zone_emitters_on_one_circuit():
    em = {zn: _emitter() for zn in ("Z_ground", "Z_first")}
    res = _run({"topology": "central", "distribution": DIST, "generation": IDEAL},
               emission_by_zone=em)
    row = res.annual.iloc[0]
    zone_sum = sum(res.emission.emission[zn].summary["QH_em_in_kWh"] for zn in em)
    assert row["Q_H_em_in_kWh"] == pytest.approx(zone_sum, rel=1e-6)
    assert row["Q_H_dis_in_kWh"] > row["Q_H_em_in_kWh"]                 # pipe losses
    expected_el = row["Q_H_dis_in_kWh"] / 3.0 + row["Q_C_dis_in_kWh"] / 3.5
    assert row["electricity_kWh"] == pytest.approx(expected_el, rel=1e-6)
    # one electricity total: generator + distribution + emitter auxiliaries of all zones
    w_em = sum(res.emission.emission[zn].summary["WH_em_aux_kWh"]
               + res.emission.emission[zn].summary["WC_em_aux_kWh"] for zn in em)
    assert row["W_em_aux_kWh"] == pytest.approx(w_em)
    assert row["electricity_total_kWh"] == pytest.approx(
        row["electricity_kWh"] + row["W_dis_aux_kWh"] + w_em)


@pytestmark_engine
def test_per_zone_topology_one_generator_per_zone():
    em = {zn: _emitter() for zn in ("Z_ground", "Z_first")}
    gas = {"kind": "ideal", "capacity_heating_kW": 3.0,       # small boiler in Z_first
           "config": {"cop_heating": 0.9, "eer": 3.0, "carrier": "natural_gas"}}
    res = _run({"topology": "per_zone", "zones": {
        "Z_ground": {"distribution": None, "generation": IDEAL},
        "Z_first": {"distribution": None, "generation": gas}}}, emission_by_zone=em)
    a = res.annual.set_index("circuit")
    assert set(a.index) == {"Z_ground", "Z_first", "TOTAL"}
    for zn in em:                                                      # no distribution losses
        assert a.loc[zn, "Q_H_dis_in_kWh"] == pytest.approx(
            res.emission.emission[zn].summary["QH_em_in_kWh"], rel=1e-6)
    assert a.loc["Z_first", "natural_gas_kWh"] == pytest.approx(a.loc["Z_first", "Q_H_dis_in_kWh"] / 0.9)
    assert a.loc["Z_ground", "natural_gas_kWh"] == 0.0
    assert a.loc["TOTAL", "Q_H_dis_in_kWh"] == pytest.approx(
        a.loc["Z_ground", "Q_H_dis_in_kWh"] + a.loc["Z_first", "Q_H_dis_in_kWh"])
    # the generator capacity reaches the thermal balance of its zone
    q_first = res.emission.hourly["Q_HVAC_Z_first"]
    assert q_first.max() <= 3000.0 + 1e-6
    assert res.emission.hourly["cap_limited_Z_first"].sum() > 0
    assert res.emission.comfort().loc["Z_first", "Kh_below_heating_setpoint"] > 0.0


def test_per_zone_dhw_shares_must_sum_to_one():
    em = {zn: _emitter() for zn in ("Z_ground", "Z_first")}
    sys_ = {"topology": "per_zone", "zones": {
        "Z_ground": {"generation": IDEAL, "dhw_share": 0.5},
        "Z_first": {"generation": IDEAL}}}
    with pytest.raises(ValueError, match="dhw_share"):
        simulate_multizone_system(_config(), sys_, emission_by_zone=em,
                                  dhw={"daily_volume_l": 100.0, "draw_temperature_C": 45.0,
                                       "cold_water_temperature_C": 10.0})


@pytestmark_engine
def test_combined_heating_cooling_dhw_with_tank():
    system = {
        "topology": "combined",
        "emission": _emitter(),                     # one radiant/fan-coil type for all zones
        "distribution": None,
        "generation": IDEAL,
        "storage": STORAGE,
        "dispatch": {"capacity_heating_kW": 12.0, "capacity_cooling_kW": 10.0,
                     "capacity_dhw_kW": 8.0, "dhw_priority": True, "cooling_dhw": "time_share"},
    }
    dhw = {"daily_volume_l": 200.0, "draw_temperature_C": 45.0, "cold_water_temperature_C": 10.0}
    res = _run(system, dhw=dhw)
    s = res.dhw_dispatch_summary()
    d = res.dispatch
    e0 = 200 * 0.001163 * 45.0
    # tank balance over the year
    assert s["Q_W_charge_kWh"] + s["Q_W_charge_recovery_kWh"] == pytest.approx(
        res.hourly["Q_W_gen_req_kWh"].sum() - s["Q_W_unmet_kWh"] + s["Q_sto_ls_kWh"]
        + d["E_sto_kWh"].iloc[-1] - e0, abs=1e-6)
    # the tank is charged both in heating and in cooling hours, and the evaluation reports it
    assert s["hours_charging_during_heating"] > 0
    assert s["hours_charging_during_cooling"] > 0
    assert s["Q_W_unmet_kWh"] == pytest.approx(0.0, abs=1e-6)
    # the generator produced the space energy served plus the tank charge
    gen = res.generators["combined"]
    assert gen["QW_gen_out_kWh"] == pytest.approx(s["Q_W_charge_kWh"])
    with pytest.raises(ValueError):
        _run(dict(system), emission_by_zone={"Z_ground": _emitter()})


@pytestmark_engine
def test_combined_tank_is_recalculated_on_the_generator_production():
    """The ideal generator can give only 2 kW at DHW temperature while the dispatch was told
    8 kW: the dispatch (tank state) is recalculated on what the generator produces.  The
    zones have different terminals (emission_by_zone) on the one generator."""
    em = {"Z_ground": _emitter(), "Z_first": _emitter()}
    em["Z_first"]["heating"]["control_K"] = 1.0                    # a different terminal
    gen = {"kind": "ideal", "config": {"cop_heating": 3.0, "cop_dhw": 2.5, "eer": 3.5,
                                         "max_dhw_kW": 2.0}}
    system = {"topology": "combined", "distribution": None, "generation": gen,
              "storage": STORAGE,
              "dispatch": {"capacity_heating_kW": 12.0, "capacity_cooling_kW": 10.0,
                           "capacity_dhw_kW": 8.0}}
    dhw = {"daily_volume_l": 200.0, "draw_temperature_C": 45.0, "cold_water_temperature_C": 10.0}
    res = _run(system, emission_by_zone=em, dhw=dhw)
    row = res.annual.iloc[0]
    assert row["dispatch_iterations"] > 1
    assert res.dispatch["Q_W_charge_kWh"].max() <= 2.0 + 1e-6      # within the real production
    assert res.generators["combined"]["QW_gen_unmet_kWh"] == pytest.approx(0.0, abs=0.05)
    assert res.emission.delta_theta_K["Z_first"][0] != res.emission.delta_theta_K["Z_ground"][0]


@pytestmark_engine
def test_shared_generator_power_is_shared_hourly_by_request():
    """central topology, 5 kW generator for two zones: every hour the zones get the power in
    proportion to their request (same reduction factor), never more than 5 kW in total."""
    em = {zn: _emitter() for zn in ("Z_ground", "Z_first")}
    gen = dict(IDEAL, capacity_heating_kW=5.0)
    res = _run({"topology": "central", "distribution": None, "generation": gen}, emission_by_zone=em)
    h = res.emission.hourly
    assert any("dynamic" in str(n) for n in res.notes)
    q = {zn: h[f"Q_HVAC_{zn}"].clip(lower=0.0) for zn in em}
    req = {zn: h[f"Q_HVAC_req_{zn}"].clip(lower=0.0) for zn in em}
    total = q["Z_ground"] + q["Z_first"]
    assert total.max() <= 5000.0 + 1e-3
    limited = (h["cap_limited_Z_ground"] > 0) & (h["cap_limited_Z_first"] > 0) & \
              (req["Z_ground"] > 1.0) & (req["Z_first"] > 1.0)
    assert limited.sum() > 50
    # same share of the request for both zones, and the generator fully used
    r_g = q["Z_ground"][limited] / req["Z_ground"][limited]
    r_f = q["Z_first"][limited] / req["Z_first"][limited]
    np.testing.assert_allclose(r_g, r_f, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(total[limited], 5000.0, rtol=1e-6)
    # the static option keeps a fixed split instead
    res_s = _run({"topology": "central", "distribution": None, "generation": gen,
                  "capacity_allocation": "static"}, emission_by_zone=em)
    bld = res_s.emission.config["building"]
    assert "generator_groups" not in bld
    caps = [float(z["heating_capacity"]) for z in bld["zones"]]
    assert sum(caps) == pytest.approx(5000.0)                       # fixed split of 5 kW
    hs = res_s.emission.hourly
    for z, c in zip(bld["zones"], caps):
        assert hs[f"Q_HVAC_{z['name']}"].max() <= c + 1e-3
