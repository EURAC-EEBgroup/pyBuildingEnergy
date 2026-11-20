"""EN ISO 52016-1 adjacent thermally unconditioned zones (6.4.5, 6.5.8.4, 6.5.9, 6.5.12, E.3)."""

import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
from pybuildingenergy.source.unconditioned_zone_52016_1 import (
    UnconditionedZoneSet,
    compute_zone_coefficients,
    corrected_thermal_resistance,
    ztu_theta_external,
)
from pybuildingenergy.source.utils import ISO52016
from pybuildingenergy.source.ventilation import VentilationInternalGains

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adjacent_unheated_zone import BASE_CONFIG, WEATHER, _adjacent_floor_config  # noqa: E402


def _zone(**kw):
    z = {
        "name": "Z1",
        "orientation_zone": {"azimuth": 0},
        "area_facade_elements": [20.0, 60.0, 30.0, 30.0, 50.0],
        "typology_elements": ["OP", "OP", "OP", "OP", "OP"],
        "transmittance_U_elements": [1.0, 1.0, 1.0, 1.0, 1.0],
        "orientation_elements": ["NV", "SV", "EV", "WV", "HOR"],
        "volume": 300.0,
        "building_type_class": "Residential_apartment",
        "a_use": 50.0,
    }
    z.update(kw)
    return z


# ----------------------------------------------------------------- coefficients


def test_formula_6_default_ventilation_contribution():
    c = compute_zone_coefficients(_zone())
    assert c.H_iu == pytest.approx(20.0)                 # elements facing the zone azimuth (NV)
    assert c.H_tr_ue == pytest.approx(170.0)
    assert c.H_ue == pytest.approx(1.5 * 170.0)          # (1 + c_ztu;ve) * H_tr;ue, c = 0.5
    assert c.b == pytest.approx(c.H_ue / (c.H_ue + c.H_iu))
    assert c.F == pytest.approx(1.0)


def test_legacy_iso13789_ventilation_when_n_ue_given():
    c = compute_zone_coefficients(_zone(), n_ue=0.5)
    assert c.H_ue == pytest.approx(170.0 + 0.33 * 300.0 * 0.5)


def test_distribution_factor_with_two_conditioned_zones():
    c = compute_zone_coefficients(_zone(), other_conditioned_H_iu=[60.0])
    assert c.F == pytest.approx(20.0 / 80.0)
    assert c.H_tot == pytest.approx(80.0 + c.H_ue)
    assert c.b == pytest.approx(c.H_ue / c.H_tot)


def test_default_b_ztu_overrides_and_zeroes_gains():
    zs = UnconditionedZoneSet({"building": {"adj_zones_present": True, "number_adj_zone": 1},
                               "adjacent_zones": [_zone(b_ztu=0.4)]})
    gains = zs.gains_direct(lambda cls, a: 999.0, {"NV": {"dif": 100.0, "dir": 100.0}})
    assert zs.coeffs[0].b == 0.4
    assert gains == [{"int": 0.0, "sol": 0.0}]


# ------------------------------------------------------------------- formulae


def test_formula_58_59():
    # no gains: theta = op - b (op - e)
    assert ztu_theta_external(20.0, 0.0, 0.8, 0.0, 100.0) == pytest.approx(4.0)
    # gains raise the temperature ... but Formula 59 caps it at e + c (op - e)
    assert ztu_theta_external(20.0, 0.0, 0.8, 100.0, 100.0, c_h_max=1.0) == pytest.approx(5.0)
    assert ztu_theta_external(20.0, 0.0, 0.8, 1e6, 100.0, c_h_max=1.0) == pytest.approx(20.0)
    assert ztu_theta_external(20.0, 0.0, 0.8, 1e6, 100.0, c_h_max=0.5) == pytest.approx(10.0)


def test_formula_54_corrected_resistance():
    r, b, hsi, hse = 2.0, 0.6, 10.0, 25.0
    rc = corrected_thermal_resistance(r, b, hsi, hse)
    assert rc == pytest.approx((r + b * (1 / hsi + 1 / hse)) / (1 - b))
    # the total resistance is divided by (1 - b): U_corrected = (1 - b) U
    assert (rc + 1 / hsi + 1 / hse) == pytest.approx((r + 1 / hsi + 1 / hse) / (1 - b))
    with pytest.raises(ValueError):
        corrected_thermal_resistance(r, 1.0, hsi, hse)


def test_e3_solar_gains_use_reduction_factor_and_absorbing_surfaces():
    z = _zone(
        typology_elements=["OP", "W", "OP", "OP", "OP"],      # SV glazing in the external partition
        g_value=0.5,
    )
    c = compute_zone_coefficients(z)
    # E.5: F_sol;ue = g (1 - F_fr), F_fr = opaque / (opaque + transparent) of the external partition
    f_fr = (30.0 + 30.0 + 50.0) / (30.0 + 30.0 + 50.0 + 60.0)
    assert c.f_sol_ue == pytest.approx(0.5 * (1.0 - f_fr))
    zs = UnconditionedZoneSet({"building": {"adj_zones_present": True, "number_adj_zone": 1},
                               "adjacent_zones": [z]})
    gains = zs.gains_direct(lambda cls, a: 0.0, {"NV": {"dif": 100.0, "dir": 200.0}})
    # E.6: F_sol * sum alpha A (I_dif + I_dir F_sh): shared opaque wall 20 m2, alpha 0.6
    assert gains[0]["sol"] == pytest.approx(c.f_sol_ue * 0.6 * 20.0 * 300.0)


def test_no_glazing_means_no_solar_gains():
    zs = UnconditionedZoneSet({"building": {"adj_zones_present": True, "number_adj_zone": 1},
                               "adjacent_zones": [_zone()]})
    gains = zs.gains_direct(lambda cls, a: 0.0, {"NV": {"dif": 100.0, "dir": 200.0}})
    assert gains[0]["sol"] == 0.0


# ------------------------------------------------------------- internal gains


def _vig():
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    return VentilationInternalGains(cfg["building"])


@pytest.mark.skipif(not BASE_CONFIG.exists(), reason="example config not available")
def test_internal_gains_not_inflated_by_adjacent_zones():
    vig = _vig()
    plain = vig.internal_gains("Residential_apartment", 100.0)
    # legacy arguments no longer add the conditioned zone's own gains again
    legacy = vig.internal_gains("Residential_apartment", 100.0, unconditioned_zones_nearby=True,
                                list_adj_zones=2, Fztc_ztu_m=1.0, b_ztu=0.8)
    assert legacy == pytest.approx(plain)


@pytest.mark.skipif(not BASE_CONFIG.exists(), reason="example config not available")
def test_formula_65_attribution_only_for_internal_type():
    vig = _vig()
    plain = vig.internal_gains("Residential_apartment", 100.0)
    phi_ztu = vig.internal_gains("Residential_apartment", 40.0, use_overrides=False)
    got = vig.internal_gains("Residential_apartment", 100.0,
                             ztu_contributions=[(phi_ztu, 0.8, 1.0)])
    assert got == pytest.approx(plain + 0.2 * phi_ztu)
    states = [
        {"type": "external", "b": 0.8, "F": 1.0, "phi_int_dir": 500.0, "phi_sol_dir": 300.0},
        {"type": "internal", "b": 0.5, "F": 1.0, "phi_int_dir": 100.0, "phi_sol_dir": 40.0},
    ]
    att = UnconditionedZoneSet.attributed_gains(states)
    assert att == {"int": pytest.approx(50.0), "sol": pytest.approx(20.0)}


# -------------------------------------------------------------- engine level


def _run(mod=None):
    cfg = _adjacent_floor_config()
    if mod:
        mod(cfg)
    bui, _ = sanitize_and_validate_BUI(copy.deepcopy(cfg), fix=True)
    with contextlib.redirect_stdout(io.StringIO()):
        hourly, _ = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(WEATHER)
        )
    return hourly, bui


@pytest.fixture(scope="module")
def two_zone_run():
    if not BASE_CONFIG.exists():
        pytest.skip("FH_Poland_DC_baseline_two_floors.json not available (gitignored local input)")

    def mod(cfg):
        cfg["building"]["number_adj_zone"] = 2
        cfg["adjacent_zones"][0]["b_ztu"] = 0.3
        second = copy.deepcopy(cfg["adjacent_zones"][0])
        second["name"] = "Z_other"
        second["b_ztu"] = 0.7
        cfg["adjacent_zones"].append(second)

    return _run(mod)


def test_each_zone_uses_its_own_adjustment_factor(two_zone_run):
    hourly, _ = two_zone_run
    op = hourly["T_op"].to_numpy()[:-1]
    te = hourly["T_ext"].to_numpy()[1:]
    for name, b in (("Z_first", 0.3), ("Z_other", 0.7)):
        th = hourly[f"theta_ztu_{name}"].to_numpy()[1:]
        expected = op - b * (op - te)
        expected = np.minimum(te + 1.0 * (op - te), expected)
        assert np.allclose(th, expected, atol=1e-6), name
    assert not np.allclose(hourly["theta_ztu_Z_first"], hourly["theta_ztu_Z_other"])


@pytest.fixture(scope="module")
def runs_external_internal():
    if not BASE_CONFIG.exists():
        pytest.skip("FH_Poland_DC_baseline_two_floors.json not available (gitignored local input)")
    ext, bui = _run()
    internal, _ = _run(lambda cfg: cfg["adjacent_zones"][0].__setitem__("type", "internal"))
    return ext, internal, bui


def test_internal_type_transmits_with_corrected_resistance(runs_external_internal):
    ext, internal, bui = runs_external_internal
    surf = [s for s in bui["building_surface"] if s["type"] == "adjacent"][0]
    ua = surf["u_value"] * surf["area"]
    c = compute_zone_coefficients(bui["adjacent_zones"][0])
    # Formula 54: transmission to the exterior with U_corrected = (1 - b) U
    ratio = internal["Q_adj"].sum() / ((1 - c.b) * ua * (internal["T_op"] - internal["T_ext"]).sum())
    assert 0.85 < ratio < 1.25
    # the temperature of an internal-type zone is the operative temperature of the zone (6.4.5.3)
    assert np.allclose(internal["theta_ztu"].to_numpy()[1:], internal["T_op"].to_numpy()[:-1])


def test_external_type_exchanges_with_ztu_temperature(runs_external_internal):
    ext, _, bui = runs_external_internal
    surf = [s for s in bui["building_surface"] if s["type"] == "adjacent"][0]
    ua = surf["u_value"] * surf["area"]
    ratio = ext["Q_adj"].sum() / (ua * (ext["T_op"] - ext["theta_ztu"]).sum())
    assert 0.85 < ratio < 1.25


# ---------------------------------------------------- ventilation from the ztu


def test_ventilation_from_adjacent_zone_supply_temperature_and_formula_62():
    from pybuildingenergy.source.ventilation import _resolve_component_streams

    comp = [{"name": "from_attic", "ventilation_type": "constant_ach", "air_changes_per_hour": 0.5,
             "from_adjacent_zone": "A"}]
    kw = dict(outdoor_temperature_c=0.0, zone_volume_m3=360.0, c_air=1006.0, rho_air=1.204)
    base = 0.5 * 360.0 / 3600.0 * 1.204 * 1006.0

    ext = _resolve_component_streams(
        comp, adjacent_zone_state={"A": {"type": "external", "theta": 12.0, "b": 0.8}}, **kw)[0]
    assert ext.source_temperature_c == 12.0                       # 6.5.10.2: supply = theta_ztu
    assert ext.heat_transfer_coefficient_w_k == pytest.approx(base)

    internal = _resolve_component_streams(
        comp, adjacent_zone_state={"A": {"type": "internal", "theta": 20.0, "b": 0.8}}, **kw)[0]
    assert internal.source_temperature_c == 0.0                   # exterior air
    assert internal.heat_transfer_coefficient_w_k == pytest.approx(0.8 * base)   # Formula 62

    with pytest.raises(ValueError):
        _resolve_component_streams(comp, adjacent_zone_state={}, **kw)


def test_c_ztu_h_max_none_means_no_limit():
    """TR 52016-2, 6.5.9 says 'no maximum'; the formula with c = 0 caps at the outdoor temperature."""
    assert ztu_theta_external(20.0, 0.0, 0.8, 1e6, 100.0, c_h_max=None) > 20.0
    assert ztu_theta_external(20.0, 0.0, 0.8, 1e6, 100.0, c_h_max=0.0) == pytest.approx(0.0)
    c = compute_zone_coefficients(_zone(c_ztu_h_max=None))
    assert c.c_ztu_h_max is None


def test_internal_type_with_glazing_warns_about_sunspaces():
    z = _zone(type="internal", typology_elements=["OP", "W", "OP", "OP", "OP"], g_value=0.5)
    with pytest.warns(UserWarning, match="sunspace"):
        compute_zone_coefficients(z)


# ------------------------------------------- previous-hour boundary (6.5.9, D.2)


@pytest.fixture(scope="module")
def previous_hour_run():
    """One run that records what the engine passes to the adjacent zone at every hour."""
    if not BASE_CONFIG.exists():
        pytest.skip("FH_Poland_DC_baseline_two_floors.json not available (gitignored local input)")
    calls = []
    original = UnconditionedZoneSet.step

    def spy(self, t, theta_op_prev, theta_e, gains):
        states = original(self, t, theta_op_prev, theta_e, gains)
        calls.append((t, theta_op_prev, theta_e, [dict(s) for s in states]))
        return states

    mp = pytest.MonkeyPatch()
    mp.setattr(UnconditionedZoneSet, "step", spy)
    try:
        hourly, bui = _run(lambda cfg: cfg["adjacent_zones"][0].__setitem__("b_ztu", 0.5))
    finally:
        mp.undo()
    return hourly, bui, calls


def _first_call_per_step(calls):
    first = {}
    for t, op_prev, e, states in calls:
        if t in first:
            # HVAC re-solves within the hour must not move the boundary
            assert first[t][0] == op_prev, f"step {t}: boundary changed within the hour"
        else:
            first[t] = (op_prev, e, states)
    return first


def test_adjacent_zone_uses_previous_hour_operative_temperature(previous_hour_run):
    """EN ISO 52016-1 6.5.9 (Formulae 58-59): the conditions of the adjacent zone at hour t
    come from the operative temperature of the conditioned zone at hour t-1 and the
    external temperature at hour t.  The first simulated hour (start of the warm-up period
    that precedes the reported year) uses the initial value of 20 C."""
    from pybuildingenergy.source.utils import _ZTU_INITIAL_OPERATIVE_C

    hourly, _, calls = previous_hour_run
    op = hourly["T_op"].to_numpy(dtype=float)
    te = hourly["T_ext"].to_numpy(dtype=float)
    th = hourly["theta_ztu"].to_numpy(dtype=float)
    n = len(op)

    first = _first_call_per_step(calls)
    n_steps = len(first)
    assert sorted(first) == list(range(n_steps))
    warmup = n_steps - n                                     # hours simulated before the year
    assert warmup >= 0

    assert first[0][0] == pytest.approx(_ZTU_INITIAL_OPERATIVE_C)

    # reported hour i is step warmup + i
    passed = np.array([first[warmup + i][0] for i in range(n)])
    np.testing.assert_allclose(passed[1:], op[:-1], atol=1e-9)        # theta_op(t-1)
    np.testing.assert_allclose([first[warmup + i][1] for i in range(n)], te, atol=1e-9)
    # ... and not the same-hour operative temperature
    assert np.abs(passed[1:] - op[1:]).max() > 0.1

    # Formula 58 with b_ztu = 0.5 (a default b zeroes the gains), capped by Formula 59
    expected = np.minimum(te + 1.0 * (passed - te), passed - 0.5 * (passed - te))
    np.testing.assert_allclose(th, expected, atol=1e-6)


def test_ventilation_from_adjacent_zone_sees_the_previous_hour_state(previous_hour_run):
    """The adjacent state used in hour t (transmission and ventilation from the adjacent
    zone, 6.5.10.2) is the one built from theta_op(t-1): a change of the conditioned zone
    at hour t reaches the adjacent zone only at hour t+1."""
    hourly, _, calls = previous_hour_run
    op = hourly["T_op"].to_numpy(dtype=float)
    th = hourly["theta_ztu"].to_numpy(dtype=float)
    te = hourly["T_ext"].to_numpy(dtype=float)
    first = _first_call_per_step(calls)
    warmup = len(first) - len(th)
    np.testing.assert_allclose(
        [first[warmup + i][2][0]["theta"] for i in range(len(th))], th, atol=1e-9
    )
    # with b = 0.5 and Formula 59 inactive (theta_e <= theta_op):
    # theta_ztu(t+1) - 0.5 theta_e(t+1) = 0.5 theta_op(t) exactly
    lhs = th[1:] - 0.5 * te[1:]
    free = te[1:] <= op[:-1]
    assert free.sum() > 8000
    np.testing.assert_allclose(lhs[free], 0.5 * op[:-1][free], atol=1e-6)
    # the same identity with the same-hour operative temperature does not hold
    assert np.abs(lhs[free] - 0.5 * op[1:][free]).max() > 0.05
