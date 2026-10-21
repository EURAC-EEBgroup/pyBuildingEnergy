"""Annex F Method 2: diffuse/reflected shading (ISO/TR 52016-2:2017, F.6.3, eq. F.8-F.18)."""

import numpy as np
import pytest

from pybuildingenergy.source.functions import diffuse_shading_reduction_factors as f_dif

FINS = {"left": {"depth": 1.0, "distance": 0.0}, "right": {"depth": 1.0, "distance": 0.0}}


def test_unshaded_vertical_window_is_unreduced():
    assert f_dif(90, 2.0, 3.0) == (pytest.approx(1.0), pytest.approx(1.0))


def test_remote_skyline_f18():
    # alpha = 30 deg: F_w-sky = (1 - sin(30)) / 2 = 0.25 -> F_dif = 0.25 / 0.5
    d, r = f_dif(90, 2.0, 3.0, obstruction_angle_deg=30.0)
    assert d == pytest.approx(0.5)
    assert r == pytest.approx(1.0)  # (1 - 0.25) / 0.5 -> capped at 1


def test_overhang_f13_f17():
    D, L, H = 1.0, 0.2, 2.0
    p1, p2 = D / H, L / H
    f_w_o = 0.3282 * (1 - p2 / np.hypot(p1, p2))
    d, r = f_dif(90, H, 3.0, D_k_ovh_q=D, L_k_ovh_q=L, has_overhang=True)
    assert d == pytest.approx((0.5 - f_w_o) / 0.5)
    assert r == pytest.approx(1.0)


def test_fins_f11_f16():
    W = 3.0
    p1 = 1.0 / W
    f_w_s = 0.6514 * (1 - 0.0 / p1)
    d, _ = f_dif(90, 2.0, W, side_fins=FINS)
    assert d == pytest.approx(1.0 - f_w_s)


def test_smallest_factor_is_kept_and_bounded():
    both, _ = f_dif(90, 2.0, 3.0, D_k_ovh_q=1.0, L_k_ovh_q=0.0, has_overhang=True, side_fins=FINS)
    only_ovh, _ = f_dif(90, 2.0, 3.0, D_k_ovh_q=1.0, L_k_ovh_q=0.0, has_overhang=True)
    only_fin, _ = f_dif(90, 2.0, 3.0, side_fins=FINS)
    assert both == pytest.approx(min(only_ovh, only_fin))
    assert 0.0 <= both <= 1.0


def test_horizontal_window_does_not_divide_by_zero():
    d, r = f_dif(0, 2.0, 3.0)
    assert np.isfinite(d) and np.isfinite(r)
