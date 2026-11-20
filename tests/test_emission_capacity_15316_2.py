"""EN 15316-2 emitter capacity: deficit only above P_n * t; P_n = 0 means no emitter."""

import warnings

import pandas as pd
import pytest

from pybuildingenergy.source.emission_15316_2 import EmissionSystemCalculator


def _loads(q_h, q_h_inc):
    return pd.DataFrame(
        {"T_ext": [0.0], "T_op": [20.0], "Q_H_kWh": [q_h], "Q_C_kWh": [0.0],
         "Q_H_em_out_inc_kWh": [q_h_inc]},
        index=pd.date_range("2026-01-01", periods=1, freq="h"),
    )


def _calc(p_kw):
    return EmissionSystemCalculator({
        "demand_unit": "kWh",
        "heating": {"control_K": 0.5, "nominal_power_kW": p_kw},
        "cooling": {"nominal_power_kW": 0.0},
    })


def test_lower_recalculated_need_is_not_a_deficit():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ts = _calc(100.0).run_timeseries(_loads(5.0, 4.0)).timeseries
    assert ts["Q_H_em_unmet_kWh"].iloc[0] == pytest.approx(0.0)
    assert ts["Q_H_em_in_kWh"].iloc[0] == pytest.approx(4.0)


def test_deficit_is_the_part_above_capacity():
    with pytest.warns(RuntimeWarning, match="undersized"):
        ts = _calc(3.0).run_timeseries(_loads(5.0, 5.5)).timeseries
    assert ts["Q_H_em_unmet_kWh"].iloc[0] == pytest.approx(2.5)


def test_zero_nominal_power_means_no_emitter():
    with pytest.warns(RuntimeWarning, match="no emitter"):
        ts = _calc(0.0).run_timeseries(_loads(5.0, 5.0)).timeseries
    assert ts["Q_H_em_unmet_kWh"].iloc[0] == pytest.approx(5.0)
    assert ts["Q_H_em_in_kWh"].iloc[0] == pytest.approx(0.0)


def test_negative_nominal_power_is_rejected():
    with pytest.raises(ValueError):
        _calc(-1.0).run_timeseries(_loads(5.0, 5.0))
