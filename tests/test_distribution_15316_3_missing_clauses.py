"""Numeric checks for the EN 15316-3:2017 clauses added to DistributionSystemCalculator:

- Eq.(8)-(9): open-circuited DHW stub losses during tapping.
- Eq.(10) (via the Eq.(15) stand-in): DHW circulation-loop loss while not tapping.
- Eq.(15): simplified method for a DHW pipe section without circulation.
- Eq.(27): "all other pumps" efficiency factor, with EEI forced to 0.25.
- Eq.(29)/(30): setback/boost auxiliary energy for intermittent pump operation.
- Eq.(31): ribbon heater auxiliary energy.
- Table 10: externally supplied beta_{H,C,W}_dis and a time-varying ambient
  temperature (referenced by column name) instead of a single static value.

Each test hand-computes the expected value from the standard's formula so a
mismatch points at the implementation, not at a snapshot.
"""

import math

import pandas as pd
import pytest

from pybuildingenergy.source.distribution_15316_3 import DistributionSystemCalculator

_CW = 1.163e-3  # kWh/(kg*K), Table 11
_RHO_W = 990.0  # kg/m3, Table 11


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="h")


def test_dhw_circulation_standby_loss_uses_eq15_when_not_tapping():
    """Eq.(10): loss of a circulating DHW section while NOT tapping, using the
    Eq.(15) stand-in theta_W,mean = 25*Psi^-0.2 for the standby temperature."""

    idx = _hourly_index(2)
    data = pd.DataFrame(
        {
            "Q_W_kWh": [1.0, 0.0],  # hour 0: tapping: hour 1: no demand
        },
        index=idx,
    )
    psi, length, ambient = 0.3, 10.0, 20.0
    calc = DistributionSystemCalculator(
        {
            "dhw": {
                "operation_mode": "demand",
                "dhw_temperature_C": 55.0,
                "dhw_return_deltaT_K": 5.0,
                "pipe_sections": [
                    {
                        "length_m": length,
                        "linear_thermal_transmittance_W_mK": psi,
                        "ambient_temperature_C": ambient,
                        "recoverable": False,
                        "circulating": True,
                    }
                ],
            }
        }
    )
    result = calc.run_timeseries(data)

    mean_temp = 55.0 - 0.5 * 5.0  # Eq.(2)
    expected_tapping = psi * length * (mean_temp - ambient) * 1.0 / 1000.0
    theta_standby = 25.0 * psi ** -0.2  # Eq.(15)
    expected_standby = psi * length * (theta_standby - ambient) * 1.0 / 1000.0

    assert result.timeseries["Q_W_dis_ls_kWh"].iloc[0] == pytest.approx(expected_tapping)
    assert result.timeseries["Q_W_dis_ls_kWh"].iloc[1] == pytest.approx(expected_standby)
    # Sanity: a 20 K-ish standby temperature drop still carries a real loss,
    # not zero - this is exactly the term the old implementation omitted.
    assert expected_standby > 0.0


def test_dhw_non_circulating_section_uses_eq15_for_the_whole_step():
    """Eq.(15): a pipe section without circulation loses heat continuously,
    not just while tapping - unlike the circulating-loop case above."""

    idx = _hourly_index(2)
    data = pd.DataFrame({"Q_W_kWh": [1.0, 0.0]}, index=idx)
    psi, length, ambient = 0.3, 10.0, 20.0
    calc = DistributionSystemCalculator(
        {
            "dhw": {
                "operation_mode": "demand",
                "pipe_sections": [
                    {
                        "length_m": length,
                        "linear_thermal_transmittance_W_mK": psi,
                        "ambient_temperature_C": ambient,
                        "circulating": False,
                    }
                ],
            }
        }
    )
    result = calc.run_timeseries(data)

    theta_mean = 25.0 * psi ** -0.2
    expected = psi * length * (theta_mean - ambient) * 1.0 / 1000.0
    # Same loss on the tapping hour and the idle hour: Eq.(15) does not care
    # whether the hour had a tap or not.
    assert result.timeseries["Q_W_dis_ls_kWh"].iloc[0] == pytest.approx(expected)
    assert result.timeseries["Q_W_dis_ls_kWh"].iloc[1] == pytest.approx(expected)


def test_dhw_open_stub_loss_eq8_eq9():
    idx = _hourly_index(2)
    data = pd.DataFrame({"Q_W_kWh": [1.0, 0.0]}, index=idx)
    stub_volume, taps_per_hour, ambient, hot = 0.001, 2.0, 20.0, 55.0
    calc = DistributionSystemCalculator(
        {
            "dhw": {
                "operation_mode": "demand",
                "dhw_temperature_C": hot,
                "pipe_sections": [],
                "stub_volume_m3": stub_volume,
                "stub_ambient_temperature_C": ambient,
                "taps_per_hour": taps_per_hour,
            }
        }
    )
    result = calc.run_timeseries(data)

    m_stub_kg_h = stub_volume * _RHO_W * taps_per_hour  # Eq.(9)
    expected_tapping = m_stub_kg_h * _CW * (hot - ambient) * 1.0  # Eq.(8)

    assert result.timeseries["Q_W_dis_stub_kWh"].iloc[0] == pytest.approx(expected_tapping)
    # No tapping in hour 1 -> no stub loss (Eq.8 only applies "during operation").
    assert result.timeseries["Q_W_dis_stub_kWh"].iloc[1] == pytest.approx(0.0)
    assert result.summary["QW_dis_stub_kWh"] == pytest.approx(expected_tapping)


def test_pump_eq27_forces_eei_to_0_25_for_large_pumps():
    """Eq.(27): pumps outside the EU-regulation 0.001-2.5 kW range (and with
    no label power) must use EEI = 0.25 regardless of the configured EEI."""

    idx = _hourly_index(1)
    data = pd.DataFrame({"Q_H_kWh": [5.0]}, index=idx)
    calc = DistributionSystemCalculator(
        {
            "heating": {
                "operation_mode": "continuous",
                "part_load_mode": "constant_when_on",
                "pipe_sections": [],
                "design_flow_m3_h": 180.0,
                "design_delta_pressure_kPa": 100.0,  # P_hydr = 100*180/3600 = 5 kW
                "pump_control_code": 0,  # Table B.5: CP1=0.25, CP2=0.75
                "pump_selection_factor": 1.2,
                "eei": 0.5,  # must be IGNORED in favour of EEI=0.25
            }
        }
    )
    result = calc.run_timeseries(data)

    f_e = (1.25 + math.sqrt(0.2 / 5.0)) * 1.2  # Eq.(27), b=1.2
    expected_epsilon = f_e * (0.25 + 0.75 / 1.0) * 0.25 / 0.25  # Eq.(24), EEI forced to 0.25
    assert result.timeseries["epsilon_H_dis"].iloc[0] == pytest.approx(expected_epsilon)
    assert result.timeseries["P_H_hydr_des_kW"].iloc[0] == pytest.approx(5.0)


def test_pump_setback_and_boost_modes_eq29_eq30():
    idx = _hourly_index(3)
    data = pd.DataFrame(
        {
            "Q_H_kWh": [4.0, 0.0, 0.0],
            "pump_mode_H": ["regular", "setback", "boost"],
        },
        index=idx,
    )
    calc = DistributionSystemCalculator(
        {
            "heating": {
                "operation_mode": "continuous",
                "part_load_mode": "constant_when_on",
                "pipe_sections": [],
                "design_flow_m3_h": 100.0,
                "design_delta_pressure_kPa": 72.0,  # P_hydr = 72*100/3600 = 2 kW
                "pump_control_code": 0,
                "eei": 0.23,
            }
        }
    )
    result = calc.run_timeseries(data)
    p_hydr_kW = 2.0

    assert result.timeseries["W_H_dis_aux_kWh"].iloc[1] == pytest.approx(p_hydr_kW * 1.0)  # Eq.(29)
    assert result.timeseries["W_H_dis_aux_kWh"].iloc[2] == pytest.approx(3.3 * p_hydr_kW * 1.0)  # Eq.(30)
    # The regular-mode row must still go through the normal Eq.(22)-(24) path
    # (a positive, but different, value - not accidentally re-using Eq. 29/30).
    regular = result.timeseries["W_H_dis_aux_kWh"].iloc[0]
    assert regular > 0.0
    assert regular not in (pytest.approx(p_hydr_kW * 1.0), pytest.approx(3.3 * p_hydr_kW * 1.0))


def test_ribbon_heater_eq31_uses_only_flagged_sections():
    idx = _hourly_index(1)
    data = pd.DataFrame({"Q_W_kWh": [1.0]}, index=idx)
    calc = DistributionSystemCalculator(
        {
            "dhw": {
                "operation_mode": "demand",
                "dhw_temperature_C": 55.0,
                "dhw_return_deltaT_K": 5.0,
                "ribbon_heater": True,
                "pipe_sections": [
                    {
                        "length_m": 10.0,
                        "linear_thermal_transmittance_W_mK": 0.3,
                        "ambient_temperature_C": 20.0,
                        "ribbon_target": True,
                    },
                    {
                        "length_m": 50.0,
                        "linear_thermal_transmittance_W_mK": 0.4,
                        "ambient_temperature_C": 20.0,
                        "ribbon_target": False,
                    },
                ],
            }
        }
    )
    result = calc.run_timeseries(data)

    mean_temp = 55.0 - 0.5 * 5.0
    expected_rib = 0.3 * 10.0 * (mean_temp - 20.0) * 1.0 / 1000.0  # only the flagged section
    assert result.timeseries["W_W_dis_rib_kWh"].iloc[0] == pytest.approx(expected_rib)
    # It must be strictly less than the total loss (which includes the second,
    # non-ribbon section).
    assert expected_rib < result.timeseries["Q_W_dis_ls_kWh"].iloc[0]


def test_beta_override_bypasses_the_internal_heuristic():
    idx = _hourly_index(1)
    data = pd.DataFrame({"Q_H_kWh": [4.0], "beta_H_dis": [0.4]}, index=idx)
    calc = DistributionSystemCalculator(
        {
            "heating": {
                "operation_mode": "continuous",
                "pipe_sections": [],
                "nominal_power_kW": 8.0,  # heuristic would give beta = 4/8 = 0.5
                "design_flow_m3_h": 0.7,
                "design_delta_pressure_kPa": 20.0,
            }
        }
    )
    result = calc.run_timeseries(data)
    assert result.timeseries["beta_H_dis"].iloc[0] == pytest.approx(0.4)


def test_ambient_temperature_can_reference_a_time_varying_column():
    idx = _hourly_index(2)
    data = pd.DataFrame(
        {
            "Q_H_kWh": [2.0, 2.0],
            "theta_H_dis_supply_C": [50.0, 50.0],
            "theta_H_dis_return_C": [40.0, 40.0],
            "boiler_room_C": [10.0, 25.0],  # arbitrary name, resolved by reference
        },
        index=idx,
    )
    psi, length = 0.3, 20.0
    calc = DistributionSystemCalculator(
        {
            "heating": {
                "operation_mode": "continuous",
                "pipe_sections": [
                    {
                        "length_m": length,
                        "linear_thermal_transmittance_W_mK": psi,
                        "ambient_temperature_C": "boiler_room_C",
                    }
                ],
            }
        }
    )
    result = calc.run_timeseries(data)

    mean_temp = 45.0  # (50+40)/2
    expected_0 = psi * length * (mean_temp - 10.0) * 1.0 / 1000.0
    expected_1 = psi * length * (mean_temp - 25.0) * 1.0 / 1000.0
    assert result.timeseries["Q_H_dis_ls_kWh"].iloc[0] == pytest.approx(expected_0)
    assert result.timeseries["Q_H_dis_ls_kWh"].iloc[1] == pytest.approx(expected_1)
    assert expected_0 != pytest.approx(expected_1)
