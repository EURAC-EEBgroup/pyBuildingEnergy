"""EN ISO 52016-1 Annex F shading: combination of the shadows, hanging overhangs, window geometry
and the shading of windows aggregated by orientation."""

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
from pybuildingenergy.source.functions import shading_reduction_factor as srf
from pybuildingenergy.source.utils import ISO52010, ISO52016

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))
from case_definitions import build_case, WEATHER_FILE  # noqa: E402

SOUTH = dict(phi_sol_t=180.0, beta_k_t=90.0, gamma_k_t=180.0)


def _obstacle(top=7.0, distance=10.0, sill=0.0):
    return {"kind": "obstacle", "azimuth_start_deg": 170.0, "azimuth_end_deg": 190.0,
            "distance_m": distance, "top_height_m": top, "window_sill_height_m": sill}


# --- F.16-F.18: overhang and obstacle shadows add up ------------------------------------------
def test_overhang_and_obstacle_shadows_add_up_f18():
    alpha = 30.0
    t = np.tan(np.radians(alpha))
    f, h_sun = srf(alpha_sol_t=alpha, **SOUTH, D_k_ovh_q=1.0, L_k_ovh_q=0.0,
                   elements_shading_type=["horizontal_overhang", "obstacles"], H_k=2.0, W_k=1.0,
                   annex_f_shading_objects=[_obstacle(top=7.0)])
    obstacle, overhang = 7.0 - 10.0 * t, 1.0 * t  # F.10 from below, F.4 from above
    assert h_sun == pytest.approx(2.0 - (obstacle + overhang))
    assert f == pytest.approx(h_sun / 2.0)


def test_each_family_is_capped_by_the_window_height():
    f, h_sun = srf(alpha_sol_t=30.0, **SOUTH, D_k_ovh_q=0.0, L_k_ovh_q=0.0, elements_shading_type="obstacles",
                   H_k=2.0, W_k=1.0, annex_f_shading_objects=[_obstacle(top=50.0)])
    assert h_sun == 0.0 and f == 0.0


# --- F.11: hanging overhang of the detailed method -----------------------------------------------
def test_hanging_overhang_uses_f11():
    f, h_sun = srf(alpha_sol_t=45.0, **SOUTH, D_k_ovh_q=0.0, L_k_ovh_q=0.0, elements_shading_type="obstacles",
                   H_k=2.0, W_k=1.0, annex_f_shading_objects=[{
                       "kind": "overhang", "azimuth_start_deg": 170.0, "azimuth_end_deg": 190.0,
                       "distance_m": 1.0, "lowest_height_m": 3.2, "window_sill_height_m": 0.9}])
    # F.11: H + H0 - H_ovh + L tan(alpha) = 2 + 0.9 - 3.2 + 1.0 = 0.7 m shaded from the top
    assert h_sun == pytest.approx(2.0 - 0.7)
    assert f == pytest.approx(1.3 / 2.0)


def test_hanging_overhang_above_the_sunbeam_does_not_shade():
    f, _ = srf(alpha_sol_t=20.0, **SOUTH, D_k_ovh_q=0.0, L_k_ovh_q=0.0, elements_shading_type="obstacles",
               H_k=2.0, W_k=1.0, annex_f_shading_objects=[{
                   "kind": "overhang", "azimuth_start_deg": 170.0, "azimuth_end_deg": 190.0,
                   "distance_m": 1.0, "lowest_height_m": 5.0, "window_sill_height_m": 0.0}])
    assert f == 1.0


# --- simple overhang: F.4 and the input ranges of Table 15 --------------------------------------
def test_simple_overhang_f4():
    f, h_sun = srf(alpha_sol_t=60.0, **SOUTH, D_k_ovh_q=0.8, L_k_ovh_q=0.5,
                   elements_shading_type="horizontal_overhang", H_k=1.4, W_k=1.4)
    shadow = 0.8 * np.tan(np.radians(60.0)) - 0.5
    assert f == pytest.approx((1.4 - shadow) / 1.4)


@pytest.mark.parametrize("D, L", [(-0.1, 0.0), (0.5, -0.3), (float("nan"), 0.0)])
def test_overhang_inputs_must_be_non_negative(D, L):
    with pytest.raises(ValueError, match=">= 0"):
        srf(alpha_sol_t=40.0, **SOUTH, D_k_ovh_q=D, L_k_ovh_q=L,
            elements_shading_type="horizontal_overhang", H_k=1.4, W_k=1.4)


def test_side_fin_inputs_must_be_non_negative():
    with pytest.raises(ValueError, match=">= 0"):
        srf(alpha_sol_t=40.0, phi_sol_t=210.0, beta_k_t=90.0, gamma_k_t=180.0, D_k_ovh_q=0, L_k_ovh_q=0,
            elements_shading_type=None, H_k=1.4, W_k=1.4, side_fins={"right": {"depth": -0.5, "distance": 0.0}})


# --- window geometry (F.3.3.1) ----------------------------------------------------------------------
@pytest.mark.parametrize("H, W", [(1.4, None), (None, 1.4), (1.4, 0.0), (0.0, 1.4)])
def test_overhang_needs_height_and_width(H, W):
    with pytest.raises(ValueError, match="height"):
        srf(alpha_sol_t=60.0, **SOUTH, D_k_ovh_q=0.8, L_k_ovh_q=0.5,
            elements_shading_type="horizontal_overhang", H_k=H, W_k=W)


def test_obstacles_only_do_not_need_the_width_f3_3_2():
    f, _ = srf(alpha_sol_t=10.0, **SOUTH, D_k_ovh_q=0, L_k_ovh_q=0, elements_shading_type="obstacles",
               H_k=2.0, W_k=None, annex_f_shading_objects=[_obstacle(top=3.0)])
    assert 0.0 < f < 1.0


def test_window_without_shading_elements_is_never_blocked_by_missing_dimensions():
    # before the fix a missing width returned 0: the whole direct sun was lost silently
    f, _ = srf(alpha_sol_t=60.0, **SOUTH, D_k_ovh_q=0, L_k_ovh_q=0, elements_shading_type=None,
               H_k=1.4, W_k=None)
    assert f == 1.0
    f, _ = srf(alpha_sol_t=60.0, phi_sol_t=0.0, beta_k_t=90.0, gamma_k_t=180.0, D_k_ovh_q=0, L_k_ovh_q=0,
               elements_shading_type=None, H_k=1.4, W_k=None)
    assert f == 0.0  # sun behind the facade (F.3)


# --- misspelled key of the overhang data --------------------------------------------------------------
def test_misspelled_overhang_key_is_accepted_with_a_warning():
    win = {"name": "w", "shading_type": "horizontal_overhang",
           "overhang_proprieties": {"width_of_horizontal_overhangs": 0.8}}
    with pytest.warns(UserWarning, match="misspelling"):
        props = ISO52010._overhang_properties_of_window(win)
    assert props["width_of_horizontal_overhangs"] == 0.8


def test_overhang_without_depth_is_reported():
    with pytest.warns(UserWarning, match="depth D"):
        ISO52010._overhang_properties_of_window({"name": "w", "shading_type": "horizontal_overhang"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ISO52010._overhang_properties_of_window({"name": "w", "shading_type": "none"})


def _bui_with_window(**window):
    bui = build_case("light")
    for s in bui["building_surface"]:
        if s["name"] == "South window":
            s.update(window)
    return bui


def test_check_input_renames_the_misspelled_key_and_rejects_missing_geometry():
    base = dict(shading_type="horizontal_overhang", width_or_distance_of_shading_elements=0.0,
                overhang_proprieties={"width_of_horizontal_overhangs": 1.0}, height=2.0, width=6.0)
    fixed, issues = sanitize_and_validate_BUI(_bui_with_window(**base), fix=True)
    win = next(s for s in fixed["building_surface"] if s["name"] == "South window")
    assert win["overhang_properties"]["width_of_horizontal_overhangs"] == 1.0 and "overhang_proprieties" not in win
    assert any("overhang_proprieties" in i["path"] and i["fix_applied"] for i in issues)

    no_width = {**base, "overhang_properties": base["overhang_proprieties"]}
    del no_width["overhang_proprieties"]
    no_width["width"] = None
    _, issues = sanitize_and_validate_BUI(_bui_with_window(**no_width), fix=True)
    assert any(i["level"] == "ERROR" and i["path"].endswith(".width") and not i["fix_applied"] for i in issues)

    negative = {**no_width, "width": 6.0, "width_or_distance_of_shading_elements": -0.2}
    _, issues = sanitize_and_validate_BUI(_bui_with_window(**negative), fix=True)
    assert any(i["level"] == "ERROR" and "width_or_distance" in i["path"] for i in issues)


# --- windows aggregated by orientation (single-zone engine) --------------------------------------------
def _run(windows):
    """Case 600 with the 12 m2 south window replaced by ``windows`` (list of (name, area, overhang depth))."""
    bui = build_case("light")
    surfaces = []
    for s in bui["building_surface"]:
        if s["name"] != "South window":
            surfaces.append(s)
            continue
        for name, area, depth in windows:
            w = copy.deepcopy(s)
            w.update(name=name, area=area, height=2.0, width=area / 2.0)
            if depth:
                w.update(shading_type="horizontal_overhang", width_or_distance_of_shading_elements=0.0,
                         overhang_properties={"width_of_horizontal_overhangs": depth})
            surfaces.append(w)
    bui["building_surface"] = surfaces
    out = ISO52016.Temperature_and_Energy_needs_calculation(
        bui, weather_source="epw", path_weather_file=str(WEATHER_FILE),
        c_int_per_A_us=0.0, hvac_control_variable="air")
    df = out[0]
    return df["Q_solar_gains"].sum() / 1000.0, df["Q_C"].sum() / 1000.0


@pytest.fixture(scope="module")
def runs():
    return {
        "single_open": _run([("A", 12.0, 0.0)]),
        "single_overhang": _run([("A", 12.0, 2.0)]),
        "split_overhang": _run([("A1", 6.0, 2.0), ("A2", 6.0, 2.0)]),
        "mixed": _run([("A", 6.0, 2.0), ("B", 6.0, 0.0)]),
    }


def test_split_windows_keep_their_overhang(runs):
    # before the fix the aggregated element "A1 + A2" found no shading column and lost the overhang
    assert runs["split_overhang"][0] == pytest.approx(runs["single_overhang"][0], rel=1e-3)
    assert runs["split_overhang"][1] == pytest.approx(runs["single_overhang"][1], rel=2e-2)
    assert runs["split_overhang"][0] < 0.6 * runs["single_open"][0]


def test_mixed_windows_give_the_area_weighted_solar_gain(runs):
    # the solar gain is linear in the shading factor of each window (Formula 69)
    expected = 0.5 * (runs["single_overhang"][0] + runs["single_open"][0])
    assert runs["mixed"][0] == pytest.approx(expected, rel=1e-3)
