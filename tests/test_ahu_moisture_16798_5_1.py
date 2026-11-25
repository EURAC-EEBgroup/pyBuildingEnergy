"""EN 16798-5-1:2017 heat recovery corrections and moisture processes of the AHU."""

import math

import numpy as np

import pytest

from pybuildingenergy.source import ahu_moisture_16798_5_1 as mx
from pybuildingenergy.source.ventilation_16798_5_1 import (
    AHUStepInputs,
    calculate_sensible_ahu_step,
    sensible_ahu_config_from_dict,
)


def test_saturation_humidity_ratio_d6_d7():
    p = 611.2 * math.exp(17.62 * 20.0 / (243.12 + 20.0))
    assert mx.saturation_pressure_pa(20.0) == pytest.approx(p)
    assert mx.saturation_humidity_ratio(20.0) == pytest.approx(0.622 * p / (101325 - p))
    assert 0.0145 < mx.saturation_humidity_ratio(20.0) < 0.0150


def test_temperature_ratio_formulae_40_to_43():
    d = mx.HeatRecoveryData("PLATE_COUNTER", v_hr_des_m_s=2.5, q_sup_hr_nom_m3_h=1000.0)
    r = mx.temperature_ratio(d, 1000.0, 1000.0)
    assert r["f_q"] == pytest.approx(1.0)
    assert r["v_hr_eff"] == pytest.approx(2.5)                        # (43)
    assert r["f_v"] == pytest.approx(-0.0201 * (2.5 - 3.5) + 1.0)      # (42)
    assert r["eta_hr"] == pytest.approx(0.85 * r["f_v"])               # (40), Table B.6
    u = mx.temperature_ratio(d, 1000.0, 1200.0)
    assert u["f_q"] == pytest.approx(1.2 ** 0.4)                       # (41)
    # without design velocity data f_v = 1 (nominal ratio only)
    assert mx.temperature_ratio(mx.HeatRecoveryData("PLATE_CROSS"), 500, 500)["eta_hr"] == pytest.approx(0.60)


def test_rotor_speed_correction_d1():
    d = mx.HeatRecoveryData("ROT_NH", rotor_speed_rpm=10.0)
    r = mx.temperature_ratio(d, 1000.0, 1000.0)
    assert r["f_n"] == pytest.approx(1.0182 - 0.0352 * (0.5 + 0.276) ** -2.7)


def test_moisture_recovery_rotary_sorption_d5_to_d10():
    d = mx.HeatRecoveryData("ROT_SORP")
    x_eta, t_e, x_e = 0.008, 0.0, 0.003
    r = mx.moisture_recovery_efficiency(d, x_eta, t_e, x_e, 1000.0, 1000.0)
    f_dx = max(0.0, 16.4 * (x_eta - mx.saturation_humidity_ratio(t_e)) + 0.918)   # (D.5a)
    assert r["f_dx"] == pytest.approx(f_dx)
    assert r["f_q_x"] == pytest.approx(1.0)                 # (D.8) amended: 1 for balanced flow
    assert r["eta_xr"] == pytest.approx(0.69 * f_dx)
    # printed 2017 text: (D.8) with "+1" gives 0.9
    r17 = mx.moisture_recovery_efficiency(mx.HeatRecoveryData("ROT_SORP", annex_d_version="2017"),
                                          x_eta, t_e, x_e, 1000.0, 1000.0)
    assert r17["f_q_x"] == pytest.approx(0.9)
    # plate exchanger: no moisture recovery (45c)
    assert mx.moisture_recovery_efficiency(mx.HeatRecoveryData("PLATE_CROSS"), x_eta, t_e, x_e,
                                           1000, 1000)["eta_xr"] == 0.0


def test_dehumidification_reaches_the_humidity_set_point_formulae_57_to_63():
    out = mx.cooling_and_dehumidification(28.0, 0.012, 18.0, 0.008)
    assert out["x_SUP_C"] == pytest.approx(0.008, abs=1e-9)
    assert out["theta_SUP_C"] < 18.0              # colder than the temperature set-point
    # (60a): theta_DHU = f_bp * theta_RCA + (1 - f_bp) * theta_coil
    assert out["theta_SUP_DHU_req"] == pytest.approx(0.1 * 28.0 + 0.9 * out["theta_C_coil"])


def _cfg(**kw):
    d = {"sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 16.0,
         "cooling_coil_enabled": True, "frost_control": "none",
         "heat_recovery_control": "always_on"}
    d.update(kw)
    return sensible_ahu_config_from_dict(d)


def _inp(t_oda, x_oda, t_eta=26.0, x_eta=0.010, q=1000.0):
    return AHUStepInputs(outdoor_temperature_c=t_oda, extract_temperature_c=t_eta,
                         required_supply_flow_m3_h=q, required_extract_flow_m3_h=q,
                         outdoor_humidity_ratio=x_oda, extract_humidity_ratio=x_eta)


def test_ahu_cooling_coil_removes_sensible_and_latent_heat():
    o = calculate_sensible_ahu_step(_cfg(supply_humidity_max_kg_kg=0.009), _inp(30.0, 0.014))
    q = 1000.0 / 3600.0
    assert o.supply_humidity_ratio == pytest.approx(0.009, abs=1e-6)
    lat = q * 1.204 * mx.R_W_J_KG * (0.014 - 0.009)
    assert o.latent_cooling_coil_power_w == pytest.approx(lat, rel=1e-6)
    assert o.actual_cooling_coil_power_w > o.latent_cooling_coil_power_w
    # after dehumidification the air is reheated to the supply set-point (67b)
    assert o.actual_heating_coil_power_w > 0.0
    assert o.actual_supply_temperature_c == pytest.approx(16.0, abs=1e-6)


def test_ahu_steam_and_adiabatic_humidification_formulae_65_66_82():
    steam = calculate_sensible_ahu_step(
        _cfg(supply_humidity_min_kg_kg=0.006, supply_temperature_setpoint_c=20.0),
        _inp(0.0, 0.002, t_eta=21.0, x_eta=0.005))
    q = 1000.0 / 3600.0
    assert steam.supply_humidity_ratio == pytest.approx(0.006)
    assert steam.humidification_power_w == pytest.approx(q * 1.204 * mx.R_W_J_KG * 0.004)
    assert steam.humidification_water_kg_h == pytest.approx(q * 1.204 * 0.004 * 3600)
    adia = calculate_sensible_ahu_step(
        _cfg(supply_humidity_min_kg_kg=0.006, supply_temperature_setpoint_c=20.0,
             humidifier_type="adiabatic"),
        _inp(0.0, 0.002, t_eta=21.0, x_eta=0.005))
    assert adia.humidification_power_w == 0.0
    d_theta = mx.R_W_J_KG / mx.C_A_J_KG_K * 0.004                        # (66b)
    extra = adia.actual_heating_coil_power_w - steam.actual_heating_coil_power_w
    assert extra == pytest.approx(1.204 * 1006.0 * q * d_theta, rel=1e-6)


def test_heat_recovery_data_changes_the_ratio_and_recovers_moisture():
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, heat_recovery_type="ROT_SORP",
               supply_temperature_setpoint_c=30.0, cooling_coil_enabled=False)
    o = calculate_sensible_ahu_step(cfg, _inp(-5.0, 0.002, t_eta=21.0, x_eta=0.007))
    assert o.heat_recovery_temperature_ratio == pytest.approx(0.75)    # declared eta_hr;nom
    assert o.moisture_recovery_efficiency > 0.0
    assert 0.002 < o.supply_humidity_ratio < 0.007


def test_without_humidity_inputs_nothing_changes():
    cfg = _cfg(sensible_heat_recovery_efficiency=0.7)
    a = calculate_sensible_ahu_step(cfg, AHUStepInputs(
        outdoor_temperature_c=30.0, extract_temperature_c=26.0,
        required_supply_flow_m3_h=1000.0, required_extract_flow_m3_h=1000.0))
    assert a.supply_humidity_ratio is None
    assert a.latent_cooling_coil_power_w == 0.0


# ------------------------------------------------- coupling with the zone moisture balance
def _engine_run(components, t_out=30.0, n=48):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_ventilation_boundary import _legacy_building, _run_legacy
    return _run_legacy(_legacy_building(components), n=n, t_out=t_out)


def _ahu(**kw):
    d = {"name": "ahu", "ventilation_type": "mechanical_supply", "supply_flow_m3_h": 600.0,
         "sensible_heat_recovery_efficiency": 0.0, "supply_temperature_setpoint_c": 18.0,
         "cooling_coil_enabled": True, "frost_control": "none", "heat_recovery_control": "always_on"}
    d.update(kw)
    return d


def test_ahu_dehumidification_reaches_the_zone_moisture_balance():
    dry = _engine_run([_ahu(supply_humidity_max_kg_kg=0.008)])
    assert "X_ahu_sup_ahu" in dry.columns
    on = dry["q_sup_m3h_ahu"] > 0
    assert on.any()
    # the supply leaves the coil at the dehumidification set-point
    np.testing.assert_allclose(dry.loc[on, "X_ahu_sup_ahu"], 0.008, atol=1e-6)
    assert (dry.loc[on, "Q_ahu_cool_lat_ahu"] > 0).all()
    # eq. (80): the zone is drier than with outdoor air only
    assert (dry["X_int_ss_kg_kg_da"].iloc[-1] < dry["X_ext_kg_kg_da"].iloc[-1])


def test_without_moisture_data_the_balance_is_unchanged():
    plain = _engine_run([_ahu(cooling_coil_enabled=False)])
    assert plain["X_ahu_sup_ahu"].isna().all()            # no moisture data: no supply humidity
    assert (plain["Q_ahu_cool_lat_ahu"] == 0).all()
    assert "X_int_ss_kg_kg_da" not in plain.columns


def test_multizone_zone_ahu_dehumidifies_only_its_zone():
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
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
        if z["name"] == "Z_first":
            z["ventilation"] = {"components": [_ahu(supply_flow_m3_h=300.0, supply_humidity_max_kg_kg=0.008)]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(wea), weather_source="epw", include_total=False)
    h = h[0] if isinstance(h, tuple) else h
    assert "X_ahu_sup_ahu_Z_first" in h.columns and "X_ahu_sup_ahu_Z_ground" not in h.columns
    summer = h.index.month.isin([7, 8])
    on = (h["q_sup_m3h_ahu_Z_first"] > 0) & summer
    assert (h.loc[on, "X_ahu_sup_ahu_Z_first"] <= 0.008 + 1e-6).all()
    assert h.loc[on, "Q_ahu_cool_lat_ahu_Z_first"].sum() > 0.0
    # coupled: the moisture balance uses the humidity computed in the heat balance hour
    np.testing.assert_allclose(h.loc[on, "X_sup_ss_kg_kg_da_Z_first"],
                               h.loc[on, "X_ahu_sup_ahu_Z_first"], atol=1e-12)
    # the zone moisture content with the AHU (eq. 80) is reported for Z_first only
    assert "X_int_ss_kg_kg_da_Z_first" in h.columns and "X_int_ss_kg_kg_da_Z_ground" not in h.columns


def test_moisture_is_coupled_to_the_heat_balance_of_the_same_hour():
    """The AHU called by the heat balance already treats the humidity.  With the coil
    limited to 70 % of what dehumidification to 0.008 needs, sensible + latent stay within
    the capacity, the supply stays wetter than the set-point, and the moisture balance
    uses that same supply humidity."""
    full = _engine_run([_ahu(supply_humidity_max_kg_kg=0.008)])
    on = full["q_sup_m3h_ahu"] > 0
    cap = 0.7 * float(full.loc[on, "Q_ahu_cool_req_ahu"].min())
    h = _engine_run([_ahu(supply_humidity_max_kg_kg=0.008, cooling_coil_max_power_w=cap)])
    on = h["q_sup_m3h_ahu"] > 0
    assert (h.loc[on, "Q_ahu_cool_ahu"] <= cap + 1e-6).all()
    assert (h.loc[on, "Q_ahu_cool_lat_ahu"] > 0).all()
    assert (h.loc[on, "X_ahu_sup_ahu"] > 0.008 + 1e-5).all()          # incomplete dehumidification
    np.testing.assert_allclose(h.loc[on, "X_sup_ss_kg_kg_da"], h.loc[on, "X_ahu_sup_ahu"], atol=1e-12)
    # and the latent part reduces the sensible cooling the coil can give (same total)
    assert (h.loc[on, "Q_ahu_cool_ahu"] - h.loc[on, "Q_ahu_cool_lat_ahu"]
            < full.loc[on, "Q_ahu_cool_ahu"] - full.loc[on, "Q_ahu_cool_lat_ahu"]).all()


def test_two_ahus_are_mixed_by_flow():
    h = _engine_run([
        _ahu(name="a1", supply_flow_m3_h=300.0, supply_humidity_max_kg_kg=0.008),
        _ahu(name="a2", supply_flow_m3_h=300.0, supply_humidity_max_kg_kg=0.010),
    ])
    on = (h["q_sup_m3h_a1"] > 0) & (h["q_sup_m3h_a2"] > 0)
    mix = (h["q_sup_m3h_a1"] * h["X_ahu_sup_a1"] + h["q_sup_m3h_a2"] * h["X_ahu_sup_a2"]) \
        / (h["q_sup_m3h_a1"] + h["q_sup_m3h_a2"])
    np.testing.assert_allclose(h.loc[on, "X_sup_ss_kg_kg_da"], mix[on], atol=1e-12)


def _causal_run(components, t_out=30.0, n=48, **kw):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_ventilation_boundary import _legacy_building, _run_single_zone_engine
    return _run_single_zone_engine(_legacy_building(components), control_mode="ahu_causal",
                                   n=n, t_out=t_out, **kw)


def test_ahu_causal_mode_is_coupled_like_the_standard_mode():
    comps = [_ahu(supply_humidity_max_kg_kg=0.008)]
    std = _engine_run(comps)
    cau = _causal_run(comps)
    on = cau["q_sup_m3h_ahu"] > 0
    assert on.any()
    assert cau["Q_ahu_cool_lat_ahu"].max() > 0.0
    np.testing.assert_allclose(cau.loc[on, "X_ahu_sup_ahu"], 0.008, atol=1e-6)
    np.testing.assert_allclose(cau.loc[on, "X_sup_ahu_mix"], cau.loc[on, "X_ahu_sup_ahu"], atol=1e-12)
    assert np.isfinite(cau["X_int_ahu"]).all()
    assert std["Q_ahu_cool_lat_ahu"].max() > 0.0


def test_cooling_coil_without_moisture_keys_condenses():
    """An enabled cooling coil follows the moisture path even without humidity set-points:
    with a coil below the dew point the air is dehumidified."""
    h = _engine_run([_ahu(supply_temperature_setpoint_c=12.0)])
    on = h["q_sup_m3h_ahu"] > 0
    assert np.isfinite(h.loc[on, "X_ahu_sup_ahu"]).all()
    assert h.loc[on, "Q_ahu_cool_lat_ahu"].max() > 0.0


def test_two_ahus_coupled_do_not_warn_about_the_first_only():
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _engine_run([
            _ahu(name="a1", supply_flow_m3_h=300.0, supply_humidity_max_kg_kg=0.008),
            _ahu(name="a2", supply_flow_m3_h=300.0, supply_humidity_max_kg_kg=0.010),
        ])
    assert not any("only the first mechanical_supply" in str(w.message) for w in rec)


def test_multizone_reported_moisture_is_the_one_seen_by_the_ahu():
    """Global latent options (moisture production and absorption) enter the in-loop
    balance, and the reported zone moisture is the one given to the AHU."""
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
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
        if z["name"] == "Z_first":
            z["ventilation"] = {"components": [_ahu(supply_flow_m3_h=300.0, supply_humidity_max_kg_kg=0.008)]}

    def run(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
                bui, path_weather_file=str(wea), weather_source="epw", include_total=False, **kw)
        return h[0] if isinstance(h, tuple) else h

    base = run()
    h = run(moisture_production_kg_s=2e-4, moisture_absorption_kg_s=5e-5, latent_internal_fraction=0.2)
    np.testing.assert_allclose(h["X_int_ss_kg_kg_da_Z_first"], h["X_int_ahu_Z_first"], atol=0.0)
    # the options change the zone moisture that the AHU sees (net source > 0: wetter zone)
    assert h["X_int_ahu_Z_first"].mean() > base["X_int_ahu_Z_first"].mean()


def _rotary(**kw):
    return _ahu(heat_recovery_type="ROT_SORP", heat_recovery_nominal_ratio=0.75,
                sensible_heat_recovery_efficiency=0.75, supply_humidity_max_kg_kg=0.008, **kw)


def test_single_zone_reported_moisture_is_the_one_seen_by_the_ahu():
    """Standard engine: X_int_ss_kg_kg_da is the zone state given to the AHU, from the
    first hour (same initialisation)."""
    h = _engine_run([_rotary()])
    np.testing.assert_allclose(h["X_int_ss_kg_kg_da"], h["X_int_ahu"], atol=0.0)
    np.testing.assert_allclose(h["X_sup_ss_kg_kg_da"], h["X_sup_ahu_mix"], atol=0.0)


def test_ahu_causal_exposes_the_latent_balance_columns():
    comps = [_rotary()]
    std = _engine_run(comps)
    cau = _causal_run(comps)
    lat_cols = [c for c in std.columns
                if c.startswith(("X_", "G_", "Phi_HU", "Phi_DHU", "Phi_ss_", "Q_latent", "Q_H_latent",
                                 "Q_int_latent", "Q_C_", "m_dot_vent", "q_sup_ahu_moist"))]
    assert lat_cols
    assert not [c for c in lat_cols if c not in cau.columns]
    np.testing.assert_allclose(cau["X_int_ss_kg_kg_da"], cau["X_int_ahu"], atol=0.0)
