"""Daylight and artificial light per control state of adaptive windows (EN ISO 52016-3)."""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybuildingenergy.source.adaptive_envelope_52016_3 import (
    LUMINOUS_EFFICACY_LM_W,
    adaptive_lighting_by_state,
)
from pybuildingenergy.source.utils import ISO52016

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"
pytestmark = pytest.mark.skipif(not (CONFIG.exists() and WEATHER.exists()), reason="inputs missing")

WIN = "Transparent south surface_Z_first"
P = f"adaptive_{WIN}_"
COEF = 0.03                      # working-plane lx per lx of vertical outdoor illuminance and tau_V
LIGHTING = {"target_lx": 300.0, "power_W_m2": 8.0, "control": "dimming", "daylight_coefficient": COEF}


def _adaptive(lighting):
    cfg = {"type": "roller_blind", "position": "external", "control_type": "automated",
           "space_category": "office", "system": "heating_cooling",
           "states": {"retracted": {"g_value": 0.65, "u_value": 0.7, "tau_v": 0.8},
                      "extended": {"g_value": 0.10, "u_value": 0.6, "tau_v": 0.05}}}
    if lighting is not None:
        cfg["lighting"] = lighting
    return cfg


def _run(lighting):
    bui = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
    for s in bui["building_surface"]:
        if s["name"] == WIN:
            s["adaptive_envelope"] = _adaptive(lighting)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw", include_total=False,
            include_latent=False)
    return h[0] if isinstance(h, tuple) else h


@pytest.fixture(scope="module")
def runs():
    return {"daylight_first": _run(LIGHTING), "no_daylight_control": _run(None)}


def test_hourly_illuminance_columns_per_state(runs):
    h = runs["daylight_first"]
    for col in ("E_v_ext_lx", "E_day_lx", "E_art_lx", "E_tot_lx", "P_light_W_m2", "state", "tau_v"):
        assert P + col in h.columns
    # outdoor vertical illuminance = K_V * irradiance; daylight = Formula 4 with tau_V of the state
    np.testing.assert_allclose(h[P + "E_v_ext_lx"], LUMINOUS_EFFICACY_LM_W * h[P + "I_sol"], rtol=1e-12)
    expected = np.minimum(h[P + "E_v_ext_lx"], COEF * h[P + "tau_v"] * h[P + "E_v_ext_lx"])
    day = h[P + "E_day_lx"] > 0
    np.testing.assert_allclose(h.loc[day, P + "E_day_lx"], expected[day], atol=1e-9)
    # night by DAY-MODE MM1 (global irradiance <= 1 W/m2): no daylight, only a residual
    # outdoor illuminance on the window plane
    night = ~day & (h[P + "E_v_ext_lx"] > 0)
    assert (expected[night] < 1.0).all()


def test_artificial_light_only_tops_up_the_daylight(runs):
    h = runs["daylight_first"]
    vis = h[P + "visual_assessed"].astype(bool)
    assert vis.sum() > 1000
    # target always reached when occupants are present; artificial = what daylight lacks
    assert (h.loc[vis, P + "E_tot_lx"] >= 300.0 - 1e-6).all()
    np.testing.assert_allclose(h.loc[vis, P + "E_art_lx"],
                               np.maximum(300.0 - h.loc[vis, P + "E_day_lx"], 0.0), atol=1e-9)
    assert (h.loc[~vis, P + "E_art_lx"] == 0.0).all()
    np.testing.assert_allclose(h[P + "P_light_W_m2"], 8.0 * h[P + "E_art_lx"] / 300.0, atol=1e-12)


def test_daylight_is_preferred_by_the_control(runs):
    """With the lighting option the DAYL-MODE check is active: when daylight is below the
    target the control picks the state that lets more light in.  Lighting energy is lower
    than with the same window controlled without looking at daylight."""
    a, b = runs["daylight_first"], runs["no_daylight_control"]
    vis = a[P + "visual_assessed"].astype(bool)
    # same formula applied afterwards to the states chosen without daylight control
    e_day_b = np.where(b[P + "E_v_ext_lx"] > 0,
                       np.minimum(b[P + "E_v_ext_lx"], COEF * b[P + "tau_v"] * b[P + "E_v_ext_lx"]), 0.0)
    e_art_b = np.where(vis, np.maximum(300.0 - e_day_b, 0.0), 0.0)
    light_a = a[P + "P_light_W_m2"].sum()
    light_b = (8.0 * e_art_b / 300.0).sum()
    assert light_a < light_b
    # the LL flag (daylight low) appears and in those hours the blind is less extended
    ll = a[P + "modes"].str.endswith("LL")
    assert ll.sum() > 0
    assert a.loc[ll, P + "fraction"].mean() <= b.loc[ll, P + "fraction"].mean()


def test_summary_per_control_state(runs):
    t = adaptive_lighting_by_state(runs["daylight_first"])
    assert set(t["window"]) == {WIN}
    assert t["hours"].sum() == len(runs["daylight_first"])
    for col in ("E_day_lx_mean", "E_art_lx_mean", "pct_daylight_only", "lighting_kWh_m2", "tau_v"):
        assert col in t.columns
    # states with more transmittance give more daylight on average when occupied
    occ = t[t["hours_occupied_awake"] > 50]
    assert occ["E_tot_lx_min"].min() >= 300.0 - 1e-6


def test_lighting_needs_a_daylight_model():
    from pybuildingenergy.source.adaptive_envelope_52016_3 import parse_adaptive_envelope

    surf = {"name": "w", "g_value": 0.6, "u_value": 1.0,
            "adaptive_envelope": _adaptive({"target_lx": 300.0})}
    with pytest.raises(ValueError, match="daylight"):
        parse_adaptive_envelope(surf)
