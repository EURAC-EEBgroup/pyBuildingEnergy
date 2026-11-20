"""EN 16798-1 Annex B comfort categories and capacity-shortage indicators."""

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source.comfort_16798_1 import (
    capacity_shortage_indicators,
    comfort_indicators,
    running_mean_outdoor_temperature,
)


def _frame(t_op, t_ext, **extra):
    idx = pd.date_range("2026-07-01", periods=len(t_op), freq="h")
    return pd.DataFrame({"T_op_Z": t_op, "T_ext": t_ext, **extra}, index=idx)


def test_running_mean_formula_b1():
    idx = pd.date_range("2026-07-01", periods=72, freq="h")
    t = pd.Series(np.repeat([20.0, 30.0, 10.0], 24), index=idx)
    rm = running_mean_outdoor_temperature(t)
    assert rm.iloc[0] == pytest.approx(20.0)                      # first day: own mean
    assert rm.iloc[30] == pytest.approx(0.2 * 20.0 + 0.8 * 20.0)  # day 2 from day 1
    assert rm.iloc[60] == pytest.approx(0.2 * 30.0 + 0.8 * 20.0)  # day 3 from day 2


def test_adaptive_categories_with_constant_climate():
    # theta_rm = 25 -> theta_c = 0.33 * 25 + 18.8 = 27.05; II upper 30.05, III upper 31.05
    n = 24 * 5
    t_op = np.full(n, 27.0)
    t_op[-24:] = 30.5                                             # last day in category III
    h = _frame(t_op, np.full(n, 25.0))
    out = comfort_indicators(h, "Z", method="adaptive")
    assert out["hours_cat_I"] == n - 24
    assert out["hours_cat_III"] == 24
    assert out["Kh_above_cat_II"] == pytest.approx(24 * (30.5 - 30.05))
    assert out["Kh_below_cat_II"] == 0.0


def test_adaptive_not_applicable_below_10c_and_occupancy():
    n = 48
    h = _frame(np.full(n, 22.0), np.full(n, 5.0))
    out = comfort_indicators(h, "Z", method="adaptive")
    assert out["occupied_hours"] == 0 and out["not_applicable_hours"] == n
    occ = np.r_[np.ones(24), np.zeros(24)]
    h2 = _frame(np.full(n, 27.0), np.full(n, 25.0))
    assert comfort_indicators(h2, "Z", occupancy=occ)["occupied_hours"] == 24


def test_static_categories_heating_and_cooling_season():
    n = 48
    t_ext = np.r_[np.full(24, 5.0), np.full(24, 28.0)]            # winter day, summer day
    t_op = np.r_[np.full(24, 19.5), np.full(24, 26.5)]
    out = comfort_indicators(_frame(t_op, t_ext), "Z", method="static")
    assert out["hours_cat_III"] == 48                             # 19.5 >= 19; 26.5 <= 27
    assert out["Kh_below_cat_II"] == pytest.approx(24 * 0.5)
    assert out["Kh_above_cat_II"] == pytest.approx(24 * 0.5)


def test_capacity_shortage_indicators():
    h = _frame(
        np.array([18.0, 20.0, 28.0, 26.0]), np.zeros(4),
        Q_HVAC_Z=[3000.0, 1000.0, -2000.0, -500.0],
        Q_HVAC_req_Z=[5000.0, 1000.0, -3500.0, -500.0],
        cap_limited_Z=[1, 0, 1, 0],
        T_set_H_Z=[20.0] * 4, T_set_C_Z=[26.0] * 4,
    )
    out = capacity_shortage_indicators(h, "Z")
    assert out["hours_limited"] == 2
    assert out["Q_H_not_delivered_kWh"] == pytest.approx(2.0)
    assert out["Q_C_not_delivered_kWh"] == pytest.approx(1.5)
    assert out["Kh_below_heating_setpoint"] == pytest.approx(1.9)
    assert out["Kh_above_cooling_setpoint"] == pytest.approx(1.9)
