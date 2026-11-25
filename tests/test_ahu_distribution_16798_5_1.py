"""EN 16798-5-1: distribution and leakage (6.4.2), unbalanced flows, recirculation (6.4.3.2.5),
frost protection (6.4.3.2.7), adiabatic cooling (6.4.3.2.8) and the central multizone AHU."""

import math

import pytest

from pybuildingenergy.source import ahu_moisture_16798_5_1 as mx
from pybuildingenergy.source.central_ahu_16798_5_1 import CentralAHU
from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
from pybuildingenergy.source.ventilation_16798_5_1 import (
    AHUStepInputs,
    calculate_sensible_ahu_step,
    sensible_ahu_config_from_dict,
)

RHO_CP = 1.204 * 1006.0


def _step(extra=None, **kw):
    d = {"sensible_heat_recovery_efficiency": 0.8, "supply_temperature_setpoint_c": 18.0,
         "heat_recovery_control": "modulating_bypass", "frost_control": "none",
         "cooling_coil_enabled": True}
    d.update(extra or {})
    k = dict(outdoor_temperature_c=-10.0, extract_temperature_c=22.0,
             required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0)
    k.update(kw)
    return calculate_sensible_ahu_step(sensible_ahu_config_from_dict(d), AHUStepInputs(**k))


# --------------------------------------------------------------------------- tables / parser
def test_tables_10_11_b5_b14():
    c = sensible_ahu_config_from_dict({
        "sensible_heat_recovery_efficiency": 0.7, "supply_temperature_setpoint_c": 18.0,
        "duct_leakage_class": "A", "ahu_leakage_class": "L3", "heat_recovery_type": "ROT_HYG",
        "supply_fan_location": "DOWN_HR", "extract_fan_location": "UP_HR",
        "building_use": "non_residential", "frost_control": "indirect_bypass"})
    assert c.duct_leakage_factor_supply == c.duct_leakage_factor_extract == 1.18
    assert c.ahu_leakage_factor_supply == 1.10
    assert (c.hr_leakage_factor_supply, c.hr_leakage_factor_extract, c.outdoor_air_fraction_hr) \
        == (1.10, 1.10, 0.909)
    assert c.frost_exhaust_limit_c == -5.0          # Table B.14, non-residential ROT_HYG
    assert mx.frost_exhaust_limit("residential", "PLATE_CROSS") == 5.0


def test_wet_bulb_temperature():
    # 26 degC, x = 0.010 kg/kg (about 48 % RH): wet bulb about 18.3 degC
    assert mx.wet_bulb_temperature(26.0, 0.010) == pytest.approx(18.3, abs=0.2)
    xs = mx.saturation_humidity_ratio(20.0)
    assert mx.wet_bulb_temperature(20.0, xs) == pytest.approx(20.0, abs=1e-6)


# --------------------------------------------------------------------------- unbalanced flows
def test_unbalanced_flow_corrects_the_ratio_and_the_exhaust():
    bal = _step({"heat_recovery_control": "always_on"})
    low = _step({"heat_recovery_control": "always_on"}, required_extract_flow_m3_h=700.0)
    # (41): f_q = (q_ETA / q_SUP)^0.4
    assert low.heat_recovery_temperature_ratio == pytest.approx(0.8 * 0.7 ** 0.4)
    # exhaust energy balance: q_eha (T_eta - T_eha) = q_oda (T_hr - T_oda)
    t_hr = -10.0 + low.heat_recovery_temperature_ratio * 32.0
    assert low.exhaust_temperature_c == pytest.approx(22.0 - (1000 / 700) * (t_hr + 10.0))
    assert low.actual_heating_coil_power_w > bal.actual_heating_coil_power_w
    assert low.makeup_outdoor_air_flow_m3_h == 0.0


def test_extract_above_supply_brings_outdoor_makeup_air_into_the_zone():
    comp = {"name": "ahu", "ventilation_type": "mechanical_supply", "supply_flow_m3_h": 600.0,
            "extract_flow_m3_h": 900.0, "sensible_heat_recovery_efficiency": 0.7,
            "supply_temperature_setpoint_c": 18.0, "frost_control": "none"}
    coll = {}
    bdy = resolve_ventilation_boundary(
        {"building_parameters": {"ventilation": {"components": [comp]}}},
        20.0, -5.0, 0.0, zone_volume_m3=500.0, ahu_outputs_collector=coll)
    names = {s.name: s for s in bdy.streams}
    assert "ahu_makeup" in names
    mk = names["ahu_makeup"]
    assert mk.source_temperature_c == -5.0
    assert mk.heat_transfer_coefficient_w_k == pytest.approx(1.204 * 1006.0 * 300.0 / 3600.0)
    assert coll["ahu"].makeup_outdoor_air_flow_m3_h == pytest.approx(300.0)


# --------------------------------------------------------------------------- distribution
def test_duct_leakage_raises_the_fan_flow_and_reaches_the_zone():
    o = _step({"duct_leakage_class": "unknown", "ahu_leakage_class": "L2"})
    assert o.supply_fan_flow_m3_h == pytest.approx(1000.0 * 1.45 * 1.04)      # (4a), (25)
    assert o.actual_supply_flow_m3_h == pytest.approx(1450.0)                # leakage into zone
    lost = _step({"duct_leakage_class": "unknown", "duct_location": "unconditioned"})
    assert lost.actual_supply_flow_m3_h == pytest.approx(1000.0)
    assert lost.duct_heat_loss_w > 0.0


def test_duct_heat_loss_formulae_12_to_15_and_21():
    h_nc = 40.0
    o = _step({"duct_supply_h_unconditioned_w_k": h_nc, "duct_surrounding_temperature_c": 5.0})
    a = h_nc / (RHO_CP * 1000.0 / 3600.0)
    t_in = o.ahu_outlet_temperature_c
    # (12): the AHU target is raised by the duct drop computed at the required temperature
    assert t_in == pytest.approx(18.0 + (18.0 - 5.0) * (1 - math.exp(-a)))
    d_nc = (t_in - 5.0) * (1 - math.exp(-a))                                   # (14)
    assert o.duct_outlet_temperature_c == pytest.approx(t_in - d_nc)          # (13)
    assert o.duct_heat_loss_w == pytest.approx(RHO_CP * 1000.0 / 3600.0 * d_nc)  # (21)
    assert o.actual_supply_temperature_c == pytest.approx(o.duct_outlet_temperature_c)


def test_extract_duct_lowers_the_extract_temperature():
    base = _step({"heat_recovery_control": "always_on"})
    duct = _step({"heat_recovery_control": "always_on", "duct_extract_h_unconditioned_w_k": 50.0})
    assert duct.exhaust_temperature_c < base.exhaust_temperature_c
    assert duct.actual_heating_coil_power_w > base.actual_heating_coil_power_w


# --------------------------------------------------------------------------- recirculation
def test_recirculation_fixed_share_and_energy_79():
    o = _step({"recirculation_control": "fix", "outdoor_air_fraction_min": 0.3,
               "heat_recovery_control": "always_on", "cooling_coil_enabled": False})
    assert o.outdoor_air_fraction == pytest.approx(0.3)
    t_hr = -10.0 + 0.8 * 32.0
    t_rca = 0.7 * 22.0 + 0.3 * t_hr                                            # (36a)
    assert o.actual_supply_temperature_c == pytest.approx(t_rca)               # no cooling coil
    assert o.recirculation_power_w == pytest.approx(RHO_CP * 1000.0 / 3600.0 * 0.7 * 32.0)


def test_recirculation_control_raises_the_outdoor_share_to_meet_the_setpoint():
    o = _step({"recirculation_control": "fix", "outdoor_air_fraction_min": 0.3})
    assert 0.3 < o.outdoor_air_fraction <= 1.0                                 # (38b)
    assert o.actual_supply_temperature_c == pytest.approx(18.0)
    assert o.actual_heating_coil_power_w == pytest.approx(0.0, abs=1e-6)


def test_recirculation_variable_uses_the_required_outdoor_air():
    o = _step({"recirculation_control": "variable", "heat_recovery_control": "always_on"},
              required_outdoor_air_flow_m3_h=250.0)
    assert o.outdoor_air_fraction == pytest.approx(0.25)                       # (35a)


# --------------------------------------------------------------------------- frost
@pytest.mark.parametrize("mode", ["exhaust_limit", "indirect_bypass", "indirect_preheat", "indirect_recirc"])
def test_indirect_frost_keeps_the_exhaust_at_the_limit(mode):
    o = _step({"frost_control": mode, "frost_exhaust_limit_c": 0.0,
               "heat_recovery_control": "always_on"}, outdoor_temperature_c=-15.0)
    assert o.frost_protection_required
    assert o.exhaust_temperature_c == pytest.approx(0.0, abs=1e-9)


def test_frost_preheat_energy_90a_and_recirc_share():
    pre = _step({"frost_control": "indirect_preheat", "frost_exhaust_limit_c": 0.0,
                 "heat_recovery_control": "always_on"}, outdoor_temperature_c=-15.0)
    t_fp = 22.0 - 22.0 / 0.8                       # exhaust at 0 with full ratio
    assert pre.frost_preheat_power_w == pytest.approx(RHO_CP * 1000.0 / 3600.0 * (t_fp + 15.0))
    rec = _step({"frost_control": "indirect_recirc", "frost_exhaust_limit_c": 0.0,
                 "heat_recovery_control": "always_on"}, outdoor_temperature_c=-15.0)
    assert rec.frost_preheat_power_w == 0.0
    assert rec.outdoor_air_fraction == pytest.approx(1.0 - (t_fp + 15.0) / 37.0)


def test_direct_frost_control_formula_46():
    cfg = {"frost_control": "direct", "frost_direct_ratio_minus15": 0.5,
           "frost_direct_ratio_minus7": 0.65, "heat_recovery_control": "always_on"}
    cold = _step(cfg, outdoor_temperature_c=-12.0)
    mid = _step(cfg, outdoor_temperature_c=-7.0)
    mild = _step(cfg, outdoor_temperature_c=0.0)
    assert cold.exhaust_temperature_c == pytest.approx(22.0 - 0.5 * 34.0)
    assert mid.exhaust_temperature_c == pytest.approx(22.0 - 0.65 * 29.0)
    assert mild.exhaust_temperature_c == pytest.approx(22.0 - 0.8 * 22.0)
    assert cold.frost_protection_required and not mild.frost_protection_required
    with pytest.raises(ValueError, match="frost_direct_ratio"):
        _step({"frost_control": "direct"})


# --------------------------------------------------------------------------- adiabatic cooling
def test_adiabatic_cooling_on_the_extract_side():
    hot = dict(outdoor_temperature_c=32.0, extract_temperature_c=26.0,
               outdoor_humidity_ratio=0.012, extract_humidity_ratio=0.010)
    ac = _step({"adiabatic_cooling": True, "supply_temperature_setpoint_c": 20.0}, **hot)
    no = _step({"supply_temperature_setpoint_c": 20.0}, **hot)
    assert ac.adiabatic_cooling_active and not no.adiabatic_cooling_active
    assert ac.actual_cooling_coil_power_w < no.actual_cooling_coil_power_w
    assert ac.adiabatic_cooling_water_kg_h > 0.0
    assert ac.outdoor_air_fraction == 1.0
    # (51a): extract air cooled toward the wet bulb, not below it
    t_wb = mx.wet_bulb_temperature(26.0, 0.010)
    t_ac = 26.0 - 0.9 * (26.0 - t_wb)
    t_hr = 32.0 + 0.8 * (t_ac - 32.0)
    # exhaust warmed by what the outdoor air gave up in the recovery
    assert ac.exhaust_temperature_c == pytest.approx(t_ac + (32.0 - t_hr))
    # off in heating
    win = _step({"adiabatic_cooling": True}, outdoor_humidity_ratio=0.002, extract_humidity_ratio=0.007)
    assert not win.adiabatic_cooling_active


# --------------------------------------------------------------------------- central AHU / VAV
def _central(**kw):
    cfg = {"name": "c", "sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 14.0,
           "cooling_coil_enabled": True, "frost_control": "none",
           "zones": [{"zone": "a", "supply_flow_max_m3_h": 1000.0, "supply_flow_min_m3_h": 200.0},
                     {"zone": "b", "supply_flow_max_m3_h": 500.0, "vav": False}]}
    cfg.update(kw)
    return CentralAHU(cfg, ["a", "b"])


def test_vav_terminal_follows_the_zone_load():
    c = _central()
    _, z = c.step(12, 30.0, {"a": 26.0, "b": 26.0}, {})
    assert z["a"]["q"] == pytest.approx(1000.0) and z["b"]["q"] == pytest.approx(500.0)
    # previous step: VAV gave rho c q (14 - 26) and the zone still needed 1 kW of cooling
    _, z = c.step(13, 30.0, {"a": 26.0, "b": 26.0}, {"a": -1000.0, "b": 0.0})
    need = 1000.0 + RHO_CP * 1000.0 / 3600.0 * 12.0
    assert z["a"]["q_req"] == pytest.approx(min(1000.0, need / (RHO_CP * 12.0) * 3600.0))
    # heating load with cold supply: minimum flow; constant volume terminal unchanged
    _, z = c.step(14, 0.0, {"a": 20.0, "b": 20.0}, {"a": 3000.0, "b": 3000.0})
    assert z["a"]["q_req"] == pytest.approx(200.0)
    assert z["b"]["q_req"] == pytest.approx(500.0)


def test_central_air_flow_control_23():
    zones = {"a": 22.0, "b": 22.0}
    c = _central(air_flow_control="no_ctrl", supply_flow_nominal_m3_h=1200.0)
    out, z = c.step(0, 0.0, zones, {})
    assert out.actual_supply_flow_m3_h == pytest.approx(1200.0)
    assert z["a"]["q"] / z["b"]["q"] == pytest.approx(2.0)                     # (6)
    c = _central(air_flow_control="on_off", operation_schedule=[0.0] * 6 + [1.0] * 18)
    out, _ = c.step(3, 0.0, zones, {})
    assert out.actual_supply_flow_m3_h == 0.0
    c = _central(air_flow_control="multi_stage", flow_stages_m3_h=[800.0, 1600.0])
    out, _ = c.step(9, 0.0, zones, {})
    # stage 1600 chosen for 1500 required; the terminals deliver at most 1000 + 500
    assert out.actual_supply_flow_m3_h == pytest.approx(1500.0)
    c = _central(air_flow_control="variable", min_flow_ratio=0.9, supply_flow_nominal_m3_h=1500.0)
    c._t_src_prev = 14.0
    out, z = c.step(9, 0.0, {"a": 20.0, "b": 22.0}, {"a": 3000.0})       # requirements 200 + 500
    assert out.actual_supply_flow_m3_h == pytest.approx(1350.0)                # f_q;min * nom
    assert z["a"]["q"] == pytest.approx(850.0) and z["b"]["q"] == pytest.approx(500.0)


def test_central_extract_is_the_flow_weighted_zone_mix():
    c = _central(heat_recovery_control="always_on", sensible_heat_recovery_efficiency=0.5,
                 zones=[{"zone": "a", "supply_flow_max_m3_h": 1000.0, "vav": False},
                        {"zone": "b", "supply_flow_max_m3_h": 1000.0, "vav": False}])
    out, _ = c.step(9, 0.0, {"a": 20.0, "b": 24.0}, {})
    assert out.exhaust_temperature_c == pytest.approx(22.0 - 0.5 * 22.0)


def test_multizone_central_ahu_serves_its_zones():
    import copy
    import json
    import warnings
    from pathlib import Path

    from pybuildingenergy.source.utils import ISO52016

    root = Path(__file__).resolve().parents[1]
    cfg, wea = root / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json", root / "examples" / "2020_Athens.epw"
    if not (cfg.exists() and wea.exists()):
        pytest.skip("inputs missing")
    bui = copy.deepcopy(json.loads(cfg.read_text(encoding="utf-8"))["building"])
    names = [z["name"] for z in bui["zones"]]
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6

    def run(with_ahu):
        b = copy.deepcopy(bui)
        if with_ahu:
            b.setdefault("building_parameters", {})["central_ahus"] = [{
                "name": "central", "sensible_heat_recovery_efficiency": 0.75,
                "supply_temperature_setpoint_c": 16.0, "cooling_coil_enabled": True,
                "frost_control": "exhaust_limit", "supply_humidity_max_kg_kg": 0.009,
                "duct_leakage_class": "B",
                "zones": [{"zone": zn, "supply_flow_max_m3_h": 600.0, "outdoor_air_flow_m3_h": 150.0}
                          for zn in names]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ISO52016.Temperature_and_Energy_needs_calculation_multizone(
                b, path_weather_file=str(wea), weather_source="epw", include_total=False)

    h, a = run(True)
    _, a0 = run(False)
    jul = h.index.month == 7
    for zn in names:
        q = h[f"q_vav_m3h_central_{zn}"]
        assert q.max() <= 600.0 * 1.06 + 1e-6 and q[jul].mean() > q[h.index.month == 1].mean()
        # the AHU supply humidity reaches the zone moisture balance
        assert f"X_int_ss_kg_kg_da_{zn}" in h.columns
    assert h.loc[jul, "Q_ahu_cool_lat_central"].sum() > 0.0
    # the cooled supply air takes part of the zone cooling need
    assert a["Q_C_annual_kWh"].sum() < a0["Q_C_annual_kWh"].sum()


def test_central_flow_never_exceeds_the_terminal_maxima():
    c = _central(air_flow_control="no_ctrl", supply_flow_nominal_m3_h=2000.0)
    out, z = c.step(0, 0.0, {"a": 22.0, "b": 22.0}, {})
    assert z["a"]["q"] == pytest.approx(1000.0) and z["b"]["q"] == pytest.approx(500.0)
    assert out.actual_supply_flow_m3_h == pytest.approx(1500.0)
    # excess above the requirements goes to the terminals below their maximum
    c = _central(air_flow_control="no_ctrl", supply_flow_nominal_m3_h=1300.0)
    c._t_src_prev = 14.0
    _, z = c.step(0, 0.0, {"a": 20.0, "b": 22.0}, {"a": 3000.0})     # a at its minimum (200)
    assert z["b"]["q"] == pytest.approx(500.0)
    assert z["a"]["q"] == pytest.approx(800.0)


def test_central_ahu_moisture_with_a_zone_outside_the_moisture_path():
    """A served zone that opts out of the AHU moisture path still gives the central AHU
    its moisture state, so the AHU treats the humidity for the other zones."""
    import copy
    import json
    import warnings
    from pathlib import Path

    from pybuildingenergy.source.utils import ISO52016

    root = Path(__file__).resolve().parents[1]
    cfg, wea = root / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json", root / "examples" / "2020_Athens.epw"
    if not (cfg.exists() and wea.exists()):
        pytest.skip("inputs missing")
    bui = copy.deepcopy(json.loads(cfg.read_text(encoding="utf-8"))["building"])
    names = [z["name"] for z in bui["zones"]]
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
    bui["zones"][0]["latent_ahu_moisture"] = False
    bui.setdefault("building_parameters", {})["central_ahus"] = [{
        "name": "central", "sensible_heat_recovery_efficiency": 0.75,
        "supply_temperature_setpoint_c": 16.0, "cooling_coil_enabled": True,
        "frost_control": "exhaust_limit", "supply_humidity_max_kg_kg": 0.009,
        "zones": [{"zone": zn, "supply_flow_max_m3_h": 600.0} for zn in names]}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h, _ = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(wea), weather_source="epw", include_total=False)
    jul = h.index.month == 7
    assert h.loc[jul, "Q_ahu_cool_lat_central"].sum() > 0.0
    assert h["X_ahu_sup_central"].notna().any()
    # the opted-out zone is not coupled, the other one is
    assert f"X_int_ahu_{names[0]}" not in h.columns
    assert f"X_int_ahu_{names[1]}" in h.columns


def _oda_central(q_b, oda_b=100.0, **kw):
    cfg = {"name": "c", "sensible_heat_recovery_efficiency": 0.7, "supply_temperature_setpoint_c": 18.0,
           "recirculation_control": "variable", "heat_recovery_control": "always_on",
           "frost_control": "none",
           "zones": [{"zone": "a", "supply_flow_max_m3_h": 1000.0, "outdoor_air_flow_m3_h": 100.0, "vav": False},
                     {"zone": "b", "supply_flow_max_m3_h": q_b, "outdoor_air_flow_m3_h": oda_b, "vav": False}]}
    cfg.update(kw)
    return CentralAHU(cfg, ["a", "b"])


def test_central_outdoor_air_share_follows_the_most_demanding_zone_35a():
    c = _oda_central(200.0)
    out, z = c.step(0, 0.0, {"a": 22.0, "b": 22.0}, {})
    assert out.outdoor_air_fraction == pytest.approx(0.5)            # max(100/1000, 100/200)
    assert z["b"]["q_oda"] == pytest.approx(100.0)
    assert z["a"]["q_oda"] == pytest.approx(500.0)
    assert z["a"]["q_oda_deficit"] == z["b"]["q_oda_deficit"] == 0.0
    c = _oda_central(100.0)                                          # the reported case
    out, z = c.step(0, 0.0, {"a": 22.0, "b": 22.0}, {})
    assert z["b"]["q_oda"] >= 100.0 - 1e-9 and z["a"]["q_oda"] >= 100.0 - 1e-9
    assert c.oda_deficit_hours == {"a": 0, "b": 0}


def test_central_outdoor_air_deficit_is_reported():
    c = _oda_central(80.0)                                           # 80 m3/h cannot carry 100
    with pytest.warns(UserWarning, match="less than the required"):
        _, z = c.step(0, 0.0, {"a": 22.0, "b": 22.0}, {})
    assert z["b"]["q_oda"] == pytest.approx(80.0)
    assert z["b"]["q_oda_deficit"] == pytest.approx(20.0)
    assert c.oda_deficit_hours["b"] == 1
    # a fixed share below the requirement is also reported
    c = _oda_central(200.0, recirculation_control="fix", outdoor_air_fraction_min=0.2)
    with pytest.warns(UserWarning):
        _, z = c.step(0, 0.0, {"a": 22.0, "b": 22.0}, {})
    assert z["b"]["q_oda_deficit"] == pytest.approx(60.0)


def test_extract_only_operation_keeps_the_extract_fan_power():
    o = _step({"extract_fan_specific_power_w_per_m3_s": 800.0,
               "supply_fan_specific_power_w_per_m3_s": 1000.0},
              required_supply_flow_m3_h=0.0, required_extract_flow_m3_h=360.0)
    assert o.actual_supply_flow_m3_h == 0.0
    assert o.actual_extract_flow_m3_h == pytest.approx(360.0)
    assert o.fan_electric_power_w == pytest.approx(800.0 * 360.0 / 3600.0)    # (72), extract part
    assert o.makeup_outdoor_air_flow_m3_h == pytest.approx(360.0)


@pytest.mark.parametrize("control", ["always_on", "modulating_bypass"])
def test_full_recirculation_with_no_outdoor_air_required(control):
    o = _step({"recirculation_control": "variable", "heat_recovery_control": control,
               "cooling_coil_enabled": False},
              outdoor_temperature_c=-5.0, extract_temperature_c=21.0,
              required_supply_flow_m3_h=360.0, required_extract_flow_m3_h=500.0,
              required_outdoor_air_flow_m3_h=0.0)
    assert math.isfinite(o.actual_supply_temperature_c)
    if control == "always_on":
        # no outdoor air: the supply is the recirculated extract air
        assert o.outdoor_air_fraction == 0.0
        assert o.actual_supply_temperature_c == pytest.approx(21.0)
        assert o.heat_recovery_power_w == 0.0
    else:
        assert 0.0 <= o.outdoor_air_fraction <= 1.0


def test_plate_heat_recovery_with_fan_locations_is_accepted():
    """Table B.5: the printed f_ODA;hr = 1.02 for PLATE is an outdoor share above 1; 1.0 is
    used (EPB Center reference spreadsheet)."""
    c = sensible_ahu_config_from_dict({
        "sensible_heat_recovery_efficiency": 0.7, "supply_temperature_setpoint_c": 18.0,
        "heat_recovery_type": "PLATE_CROSS", "supply_fan_location": "UP_HR"})
    assert c.outdoor_air_fraction_hr == 1.0
    assert c.hr_leakage_factor_supply == 1.02
