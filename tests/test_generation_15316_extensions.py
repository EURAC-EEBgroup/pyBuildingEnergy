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


def test_primary_energy_52000_1_reports_delivered_and_export_credit():
    result = pybui.PrimaryEnergyAccountingCalculator(
        {
            "export_credit_method": "symmetric",
            "primary_energy_factors": {
                "electricity": {"nonrenewable": 2.0, "renewable": 0.0},
                "natural_gas": {"nonrenewable": 1.1, "renewable": 0.0},
            },
            "carriers": ["electricity", "natural_gas"],
        }
    ).run_annual(
        {
            "E_delivered_electricity_kWh": 10.0,
            "E_delivered_natural_gas_kWh": 20.0,
            "E_exported_electricity_kWh": 3.0,
        }
    )

    assert result.summary["PE_delivered_nonren_kWh"] == pytest.approx(42.0)
    assert result.summary["PE_export_credit_nonren_kWh"] == pytest.approx(6.0)
    assert result.summary["PE_net_nonren_kWh"] == pytest.approx(36.0)


def test_lighting_15193_1_and_bacs_reduce_service_energy():
    loads = _loads()
    lighting = pybui.LightingSystemCalculator(
        {
            "area_m2": 100.0,
            "installed_power_density_W_m2": 5.0,
            "daylight_control_fraction": 0.4,
            "occupancy_control_fraction": 0.2,
        }
    ).run_timeseries(loads.assign(lighting_profile=1.0, occupancy_profile=1.0))

    bacs = pybui.BACSControlFactorCalculator(
        {"bacs_class": "B"}
    ).run_timeseries(
        pd.DataFrame({"E_L_total_kWh": lighting.timeseries["E_L_total_kWh"]})
    )

    assert lighting.summary["E_L_total_kWh"] > 0.0
    assert lighting.summary["LENI_kWh_m2a"] > 0.0
    assert bacs.summary["lighting_factor"] < 1.0
    assert (
        bacs.summary["lighting_E_L_total_kWh_after_kWh"]
        < bacs.summary["lighting_E_L_total_kWh_before_kWh"]
    )


def test_ventilation_16798_5_7_heat_recovery_and_fans():
    result = pybui.VentilationSystemCalculator(
        {
            "volume_m3": 300.0,
            "air_change_rate_h": 0.5,
            "heat_recovery_efficiency": 0.75,
            "specific_fan_power_W_s_m3": 800.0,
        }
    ).run_timeseries(_loads().assign(T_zone=20.0))

    assert result.summary["Q_vent_heat_recovered_kWh"] > 0.0
    assert (
        result.summary["Q_vent_heat_loss_after_HRV_kWh"]
        < result.summary["Q_vent_heat_loss_before_HRV_kWh"]
    )
    assert result.summary["E_vent_fan_kWh"] > 0.0


def test_cost_optimality_15459_1_global_cost():
    result = pybui.CostOptimalityCalculator(
        {
            "calculation_period_years": 30,
            "real_discount_rate": 0.03,
            "floor_area_m2": 100.0,
            "energy_prices": {"electricity": 0.30},
            "export_prices": {"electricity": 0.05},
        }
    ).calculate(
        components=[
            {
                "investment_EUR": 10_000.0,
                "annual_maintenance_EUR": 100.0,
                "lifetime_years": 15,
            }
        ],
        annual_energy={"electricity": 1000.0},
        annual_exports={"electricity": 100.0},
    )

    assert result.summary["Annual_energy_cost_EUR"] == pytest.approx(300.0)
    assert result.summary["Annual_export_value_EUR"] == pytest.approx(5.0)
    assert result.summary["Global_cost_EUR_m2"] > 100.0


def test_biomass_15316_4_wrapper_reports_biomass_delivered():
    result = pybui.BiomassBoilerSystemCalculator(
        {"nominal_power_kW": 20.0}
    ).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["E_biomass_delivered_kWh"] > result.summary["QHW_gen_out_kWh"]


def test_renewables_battery_dispatch_reduces_grid_import():
    without_battery = pybui.RenewableEnergySystemCalculator(
        {
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 100.0,
                "self_consumption_fraction": 1.0,
            }
        }
    ).run_timeseries(_loads())
    with_battery = pybui.RenewableEnergySystemCalculator(
        {
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 100.0,
                "self_consumption_mode": "dispatch",
                "battery": {"capacity_kWh": 2.0},
            }
        }
    ).run_timeseries(_loads())

    assert with_battery.summary["E_battery_discharge_to_load_kWh"] > 0.0
    assert (
        with_battery.summary["E_grid_after_PV_kWh"]
        < without_battery.summary["E_grid_after_PV_kWh"]
    )
