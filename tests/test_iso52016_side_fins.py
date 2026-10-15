"""Side fins in the direct-beam shading factor (ISO 52016-1 Annex F, Method 1).

F.5 / F.6: the fin on the sun's side casts a shadow of width
w = D_fin * tan(|phi_sol - gamma_k|) - L_fin; F.19-F.22 combine it with the
overhang shadow height as F_sh;dir = h_sun * w_sun / (H * W).
"""

import numpy as np
import pytest

from pybuildingenergy.source.functions import shading_reduction_factor

FINS = {"left": {"depth": 1.0, "distance": 0.0}, "right": {"depth": 1.0, "distance": 0.0}}


def _f(phi, alpha=10.0, gamma=180.0, side_fins=None, shading_type=None, D=None, L=None,
       H=2.0, W=3.0):
    return shading_reduction_factor(
        alpha_sol_t=alpha, phi_sol_t=phi, beta_k_t=90.0, gamma_k_t=gamma,
        D_k_ovh_q=D, L_k_ovh_q=L, elements_shading_type=shading_type,
        H_k=H, W_k=W, side_fins=side_fins,
    )[0]


def test_fin_on_sun_side_shades_formula_f5():
    # Sun 30 deg west of a south window: the right-hand (west) fin shades.
    expected = (3.0 - np.tan(np.radians(30.0))) / 3.0
    assert _f(210.0, side_fins=FINS) == pytest.approx(expected)
    assert _f(210.0, side_fins={"right": FINS["right"]}) == pytest.approx(expected)
    # Only a left (east) fin: no shadow from the west.
    assert _f(210.0, side_fins={"left": FINS["left"]}) == pytest.approx(1.0)


def test_left_fin_and_fin_distance_formula_f6():
    # Sun 45 deg east of south, left fin 0.4 m away from the window edge.
    fins = {"left": {"depth": 1.0, "distance": 0.4}}
    assert _f(135.0, side_fins=fins) == pytest.approx((3.0 - (1.0 - 0.4)) / 3.0)


def test_fin_shadow_is_capped_at_window_width():
    assert _f(180.0 + 85.0, side_fins=FINS) == pytest.approx(0.0)


def test_overhang_and_fins_combine_as_area_product_f22():
    alpha, phi = 50.0, 200.0
    d = np.radians(phi - 180.0)
    h_shadow = 1.0 * np.tan(np.radians(alpha)) / np.cos(d) - 0.5
    w_shadow = 1.0 * np.tan(d)
    expected = (2.0 - h_shadow) * (3.0 - w_shadow) / 6.0
    got = _f(phi, alpha=alpha, side_fins=FINS, shading_type="horizontal_overhang", D=1.0, L=0.5)
    assert got == pytest.approx(expected)


def test_no_fins_keeps_previous_behaviour():
    with_none = _f(210.0, alpha=40.0, shading_type="horizontal_overhang", D=1.0, L=0.5)
    legacy_alias = _f(210.0, alpha=40.0, shading_type="vertical_overhang", D=1.0, L=0.5)
    assert with_none == pytest.approx(legacy_alias)
    assert _f(210.0) == pytest.approx(1.0)
