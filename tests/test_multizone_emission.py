"""Multizone need + EN 15316-2 emission per zone (equivalent set-point recalculated together)."""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybuildingenergy.source.composer import _shift_zone_setpoints, simulate_multizone_emission

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json"
EMISSION = ROOT / "examples" / "example_building_emission_only.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"
pytestmark = pytest.mark.skipif(
    not (CONFIG.exists() and EMISSION.exists() and WEATHER.exists()), reason="inputs missing")


def _config():
    cfg = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    cfg["weather"] = {"source": "epw", "file": str(WEATHER)}
    for z in cfg["building"]["zones"]:
        z["cooling_capacity"] = 1e6
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
    return cfg


def _emitter():
    return copy.deepcopy(json.loads(EMISSION.read_text(encoding="utf-8"))["systems"]["emission"]["config"])


@pytest.fixture(scope="module")
def first_only():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulate_multizone_emission(_config(), {"Z_first": _emitter()}, unlisted_zones="ideal")


def test_shift_moves_only_the_zones_with_an_emitter():
    bui = _config()["building"]
    bui["zones"][1].pop("heating_setback", None)            # falls back to the global/engine value
    out = _shift_zone_setpoints(bui, {"Z_first": (0.65, -1.7)})
    g0, f0 = bui["zones"]
    g1, f1 = out["zones"]
    assert g1["heating_setpoint"] == g0["heating_setpoint"]
    assert f1["heating_setpoint"] == pytest.approx(f0["heating_setpoint"] + 0.65)
    assert f1["cooling_setpoint"] == pytest.approx(f0["cooling_setpoint"] - 1.7)
    global_sb = (bui.get("building_parameters") or {}).get("temperature_setpoints", {}).get("heating_setback", 16.0)
    assert f1["heating_setback"] == pytest.approx(float(global_sb) + 0.65)   # global value, else default


def test_equivalent_setpoint_of_one_zone_changes_the_neighbour(first_only):
    """Raising the heating set-point of Z_first raises its need and, through the partition,
    gives more heat to Z_ground: its heating need drops although its set-point is unchanged."""
    a = first_only.annual.set_index("zone")
    assert first_only.delta_theta_K["Z_first"][0] > 0.0
    assert "Z_ground" not in first_only.delta_theta_K
    assert a.loc["Z_first", "Q_H_need_inc_kWh"] > a.loc["Z_first", "Q_H_need_kWh"]
    d_ground = a.loc["Z_ground", "Q_H_need_inc_kWh"] - a.loc["Z_ground", "Q_H_need_kWh"]
    d_first = a.loc["Z_first", "Q_H_need_inc_kWh"] - a.loc["Z_first", "Q_H_need_kWh"]
    assert d_ground < -1.0                       # kWh: the neighbour needs less heating
    assert abs(d_ground) < d_first               # ... but less than the extra need of Z_first


def test_emission_results_only_for_zones_with_an_emitter(first_only):
    a = first_only.annual.set_index("zone")
    assert set(first_only.emission) == {"Z_first"}
    assert np.isnan(a.loc["Z_ground", "QH_em_in_kWh"])
    # emission input >= need at the equivalent set-point (losses and embedded effects >= 0 here)
    assert a.loc["Z_first", "QH_em_in_kWh"] >= a.loc["Z_first", "Q_H_need_inc_kWh"] - 1e-6
    assert a.loc["TOTAL", "Q_H_need_kWh"] == pytest.approx(
        a.loc["Z_first", "Q_H_need_kWh"] + a.loc["Z_ground", "Q_H_need_kWh"])
    assert "Z_first" in first_only.report()


def test_approximate_method_skips_the_recalculation():
    with pytest.warns(UserWarning, match="coupling between zones"):
        res = simulate_multizone_emission(_config(), {"Z_first": _emitter()}, emission_inc="approximate")
    assert res.hourly_inc is None


def test_unknown_zone_is_rejected():
    with pytest.raises(ValueError, match="unknown zones"):
        simulate_multizone_emission(_config(), {"Nowhere": _emitter()})


def test_common_hydraulic_node_uses_the_highest_zone_requirement():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_iso_15316_1_hydraulic import _c2_config

    hyd = {
        zn: _c2_config(selected_emm_cont_circuit=i, mixing_valve=True,
                       flow_temp_control_type="Type 1 - Based on demand")
        for i, zn in enumerate(("Z_ground", "Z_first"))
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simulate_multizone_emission(_config(), {"Z_first": _emitter()}, hydraulic_systems=hyd,
                                          unlisted_zones="ideal")
    system = res.hydraulic["system"]
    assert set(res.hydraulic["zones"]) == set(hyd)
    req = system["zone_temperature_requirements(°C)"]
    node = system["θH_nod_out(°C)"]
    checked = 0
    for r, n in zip(req, node):
        vals = [v for v in (r.values() if isinstance(r, dict) else []) if v is not None]
        if vals:
            assert n == pytest.approx(max(vals))             # EN 15316-1 Annex C, eq. (C.1)
            checked += 1
    assert checked > 100                                     # hours with a heating requirement

    # Full balance: the circuit of the zone with an emitter reuses the emission result
    # (one EN 15316-2 calculation, no second emission correction).
    zone = res.hydraulic["zones"]["Z_first"]
    assert set(zone["emission_calculation_mode"]) == {"en15316-2"}
    ts = res.emission["Z_first"].timeseries
    expected = (ts["Q_H_em_in_kWh"] + ts["Q_H_em_back_out_kWh"]).to_numpy()
    np.testing.assert_allclose(zone["QH_em_i_in(kWh)"].to_numpy(dtype=float), expected, atol=1e-9)


def test_emitter_capacity_reaches_the_thermal_balance():
    from pybuildingenergy.source.composer import _apply_emitter_capacities

    em = _emitter()
    em["cooling"]["nominal_power_kW"] = 0.0                  # no cooling emitter in Z_first
    bui, applied = _apply_emitter_capacities(_config()["building"], {"Z_first": em})
    assert applied["Z_first"]["cooling_capacity"] == 0.0
    assert bui["zones"][0]["cooling_capacity"] == 1e6        # Z_ground untouched

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coupled = simulate_multizone_emission(_config(), {"Z_first": em})
        detached = simulate_multizone_emission(
            _config(), {"Z_first": em}, couple_emitter_capacity=False
        )
    t_c = coupled.hourly["T_op_Z_first"]
    t_d = detached.hourly["T_op_Z_first"]
    # coupled: no cooling at all in Z_first, so the zone runs warmer in summer
    assert (coupled.hourly["Q_HVAC_Z_first"] >= -1e-6).all()
    assert t_c.max() > t_d.max() + 0.5
    # the shortage is a temperature deviation, not unmet energy, when coupled
    sc = coupled.emission["Z_first"].summary
    sd = detached.emission["Z_first"].summary
    unmet_key = next(k for k in sd if k.startswith("QC") and "unmet" in k)
    assert sd[unmet_key] > 1.0
    assert sc[unmet_key] == pytest.approx(0.0, abs=1e-6)
    # ... and the shortage is reported by the comfort/capacity indicators (EN 16798-1)
    table = coupled.comfort(method="adaptive")
    zf = table.loc["Z_first"]
    assert zf["hours_limited_cooling"] > 100
    assert zf["Q_C_not_delivered_kWh"] > 1.0
    assert zf["Kh_above_cooling_setpoint"] > 1.0
    assert table.loc["Z_ground", "hours_limited"] == 0


def test_zone_without_emitter_is_free_floating_by_default(first_only):
    with pytest.warns(UserWarning, match="free-floating"):
        res = simulate_multizone_emission(_config(), {"Z_first": _emitter()})
    assert res.free_floating_zones == ["Z_ground"]
    q = res.hourly["Q_HVAC_Z_ground"].to_numpy(dtype=float)
    assert np.abs(q).max() < 1e-6                           # no heating or cooling at all
    a = res.annual.set_index("zone")
    assert a.loc["Z_ground", "Q_H_need_kWh"] == pytest.approx(0.0, abs=1e-6)
    # Z_first now borders an unheated zone: more heating than with an ideal Z_ground
    ideal = first_only.annual.set_index("zone")
    assert a.loc["Z_first", "Q_H_need_kWh"] > ideal.loc["Z_first", "Q_H_need_kWh"]
    # the ideal option and an explicit zone flag keep the zone conditioned
    cfg = _config()
    cfg["building"]["zones"][0]["free_floating"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explicit = simulate_multizone_emission(cfg, {"Z_first": _emitter()})
    assert explicit.free_floating_zones == []
    assert np.abs(explicit.hourly["Q_HVAC_Z_ground"]).max() > 0.0
    with pytest.raises(ValueError):
        simulate_multizone_emission(_config(), {"Z_first": _emitter()}, unlisted_zones="off")
