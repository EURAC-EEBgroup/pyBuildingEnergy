import pandas as pd
import pytest

import pybuildingenergy as pybui


def _loads():
    index = pd.date_range("2020-01-01", periods=24, freq="h")
    return pd.DataFrame(
        {
            "T_ext": 5.0,
            "Q_H_kWh": 1.0,
            "Q_W_kWh": 0.2,
            "Q_C_kWh": 0.1,
            "GHI": [0.0] * 8 + [300.0] * 8 + [0.0] * 8,
            "E_site_el_load_kWh": 0.3,
        },
        index=index,
    )


def test_combustion_boiler_15316_4_1_summary_balance():
    result = pybui.CombustionBoilerSystemCalculator(
        {"nominal_power_kW": 20.0, "full_load_efficiency": 0.94}
    ).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["EHW_gen_in_kWh"] > result.summary["QHW_gen_out_kWh"]
    assert result.summary["eta_HW_gen"] > 0.9


def test_district_15316_4_5_can_include_cooling_branch():
    result = pybui.DistrictEnergySystemCalculator(
        {"cooling_enabled": True, "heating_substation_efficiency": 0.97}
    ).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["QC_gen_out_kWh"] == pytest.approx(2.4)
    assert result.summary["E_total_district_kWh"] > result.summary["QHW_gen_out_kWh"]


def test_cogeneration_15316_4_4_reports_electricity():
    result = pybui.CogenerationSystemCalculator(
        {
            "nominal_thermal_power_kW": 5.0,
            "thermal_efficiency": 0.56,
            "electrical_efficiency": 0.30,
        }
    ).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["E_chp_el_generated_kWh"] > 0.0
    assert result.summary["E_chp_el_self_consumed_kWh"] > 0.0


def test_renewables_15316_4_3_4_6_reduce_dhw_and_generate_pv():
    result = pybui.RenewableEnergySystemCalculator(
        {
            "solar_thermal": {
                "enabled": True,
                "area_m2": 2.0,
                "annual_yield_kWh_m2": 400.0,
            },
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 1200.0,
            },
        }
    ).run_timeseries(_loads())

    assert result.summary["Q_solar_thermal_used_kWh"] > 0.0
    assert result.summary["QW_after_solar_kWh"] < result.summary["QW_before_solar_kWh"]
    assert result.summary["E_PV_gen_kWh"] == pytest.approx(2400.0)
