import numpy as np
import pandas as pd
import pytest
import os
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace


# ==============================================================================
#                           FIXTURES
# ==============================================================================

@pytest.fixture
def building_data():
    """Fixture per i dati dell'edificio"""
    return {
        "building": {
            "name": "test-cy",
            "azimuth_relative_to_true_north": 41.8,
            "latitude": 46.49018685497359,
            "longitude": 11.327028776009655,
            "exposed_perimeter": 40,
            "height": 3,
            "wall_thickness": 0.3,
            "n_floors": 1,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 2,
            "net_floor_area": 100,
            "construction_class": "class_i"
        },
        "adjacent_zones": [
            {
                "name": "adj_1",
                "orientation_zone": {"azimuth": 0},
                "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
                "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
                "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
                "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
                'volume': 300,
                'building_type_class': 'Residential_apartment',
                'a_use': 50
            },
            {
                "name": "adj_2",
                "orientation_zone": {"azimuth": 180},
                "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
                "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
                "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
                "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
                'volume': 300,
                'building_type_class': 'Residential_apartment',
                'a_use': 50
            }
        ],
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": 130,
                "sky_view_factor": 1.0,
                "u_value": 2.2,
                "solar_absorptance": 0.4,
                "thermal_capacity": 741500.0,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None
            },
            {
                "name": "Opaque north surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.4,
                "solar_absorptance": 0.4,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 0, "tilt": 90},
                "name_adj_zone": "adj_1"
            },
            {
                "name": "Opaque south surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.4,
                "solar_absorptance": 0.4,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 180, "tilt": 90},
                "name_adj_zone": "adj_2"
            },
            {
                "name": "Opaque east surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.2,
                "solar_absorptance": 0.6,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 90, "tilt": 90},
                "name_adj_zone": None
            },
            {
                "name": "Opaque west surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.2,
                "solar_absorptance": 0.7,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 270, "tilt": 90},
                "name_adj_zone": None
            },
            {
                "name": "Slab to ground",
                "type": "opaque",
                "area": 100,
                "sky_view_factor": 0.0,
                "u_value": 1.6,
                "solar_absorptance": 0.6,
                "thermal_capacity": 405801,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None
            },
            {
                "name": "Transparent east surface",
                "type": "transparent",
                "area": 4,
                "sky_view_factor": 0.5,
                "u_value": 5,
                "g_value": 0.726,
                "height": 2,
                "width": 1,
                "parapet": 1.1,
                "orientation": {"azimuth": 90, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
                "name_adj_zone": None
            },
            {
                "name": "Transparent west surface",
                "type": "transparent",
                "area": 4,
                "sky_view_factor": 0.5,
                "u_value": 5,
                "g_value": 0.726,
                "height": 2,
                "width": 1,
                "parapet": 1.1,
                "orientation": {"azimuth": 270, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
                "name_adj_zone": None
            }
        ],
        "units": {
            "area": "m²",
            "u_value": "W/m²K",
            "thermal_capacity": "J/kgK",
            "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
            "tilt": "degrees (0=horizontal, 90=vertical)",
            "internal_gain": "W/m²",
            "internal_gain_profile": "Normalized to 0-1",
            "HVAC_profile": "0: off, 1: on"
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "°C"
            },
            "system_capacities": {
                "heating_capacity": 10000000.0,
                "cooling_capacity": 12000000.0,
                "units": "W"
            },
            "airflow_rates": {
                "infiltration_rate": 1.0,
                "units": "ACH (air changes per hour)"
            },
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": 4.2,
                    "weekday": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1.0, 1.0],
                    "weekend": [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
                },
                {
                    "name": "appliances",
                    "full_load": 3,
                    "weekday": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                    "weekend": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                },
                {
                    "name": "lighting",
                    "full_load": 3,
                    "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                    "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                }
            ],
            "construction": {
                "wall_thickness": 0.3,
                "thermal_bridges": 2,
                "units": "m (for thickness), W/mK (for thermal bridges)"
            },
            "climate_parameters": {
                "coldest_month": 1,
                "units": "1-12 (January-December)"
            },
            "heating_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            },
            "cooling_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            },
            "ventilation_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            }
        }
    }


@pytest.fixture
def hvac_system_config():
    """Fixture per la configurazione del sistema HVAC"""
    return {
        'emitter_type': 'Floor heating',
        'nominal_power': 8,
        'emission_efficiency': 90,
        'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
        'selected_emm_cont_circuit': 0,
        'mixing_valve': True,
        'mixing_valve_delta': 2,
        'heat_losses_recovered': True,
        'distribution_loss_recovery': 90,
        'simplified_approach': 80,
        'distribution_aux_recovery': 80,
        'distribution_aux_power': 30,
        'distribution_loss_coeff': 48,
        'distribution_operation_time': 1,
        'full_load_power': 27,
        'max_monthly_load_factor': 100,
        'tH_gen_i_ON': 1,
        'auxiliary_power_generator': 0,
        'fraction_of_auxiliary_power_generator': 40,
        'generator_circuit': 'independent',
        'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',
        'gen_outdoor_temp_data': pd.DataFrame({
            "θext_min_gen": [-7],
            "θext_max_gen": [15],
            "θflw_gen_max": [60],
            "θflw_gen_min": [35],
        }, index=["Generator curve"]),
        'speed_control_generator_pump': 'variable',
        'generator_nominal_deltaT': 20,
        'efficiency_model': 'simple',
        'calc_when_QH_positive_only': False,
        'off_compute_mode': 'full',
    }


@pytest.fixture
def output_dir(tmp_path):
    """Fixture per la directory di output temporanea"""
    test_output = tmp_path / "result_test"
    test_output.mkdir()
    return str(test_output)


# ==============================================================================
#                           TESTS
# ==============================================================================

def test_import_package():
    """Test per verificare che il package sia importabile"""
    import pybuildingenergy as pybui
    assert hasattr(pybui, "__version__")


def test_check_heating_system_inputs(hvac_system_config):
    """Test per la validazione degli input del sistema di riscaldamento"""
    import pybuildingenergy as pybui
    
    res = pybui.check_heating_system_inputs(hvac_system_config)
    
    assert "emitter_type" in res
    assert "messages" in res
    assert "config" in res
    assert res["emitter_type"] == hvac_system_config["emitter_type"]


def test_heating_system_calculator(hvac_system_config):
    """Test per il calcolatore del sistema di riscaldamento"""
    import pybuildingenergy as pybui
    
    calc = pybui.HeatingSystemCalculator(hvac_system_config)
    assert calc is not None


def test_heat_pump_system_calculator_heating_cooling_dhw():
    """Test heat pump generation for heating, cooling and DHW demand."""
    import pybuildingenergy as pybui

    heating_map = pd.DataFrame({
        "source_temperature_C": [-7, -7, 2, 2, 7, 7],
        "sink_temperature_C": [35, 55, 35, 55, 35, 55],
        "capacity_kW": [5.0, 4.0, 6.0, 5.0, 7.0, 6.0],
        "cop": [3.2, 2.4, 3.8, 2.8, 4.2, 3.2],
    })
    cooling_map = pd.DataFrame({
        "source_temperature_C": [25, 25, 35, 35],
        "sink_temperature_C": [7, 18, 7, 18],
        "capacity_kW": [5.0, 6.0, 4.0, 5.0],
        "eer": [3.0, 3.6, 2.5, 3.1],
    })
    loads = pd.DataFrame({
        "T_ext": [-5.0, 0.0, 5.0, 12.0, 25.0, 30.0],
        "Q_H_kWh": [4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
        "Q_C_kWh": [0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
        "Q_W_kWh": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }, index=pd.date_range("2026-01-01", periods=6, freq="h"))

    calc = pybui.HeatPumpSystemCalculator({
        "heating_performance_map": heating_map,
        "dhw_performance_map": heating_map,
        "cooling_performance_map": cooling_map,
        "source_type": "air",
        "time_step_hours": 1.0,
        "demand_unit": "kWh",
        "hp_operating_limit_C": 58.0,
        "dhw_target_temperature_C": 55.0,
        "dhw_sink_temperature_C": 55.0,
        "external_auxiliary_power_W": 100.0,
        "standby_power_W": 5.0,
        "heating_storage_loss_kWh_per_day": 0.1,
        "dhw_storage_loss_kWh_per_day": 0.2,
    })

    result = calc.run_timeseries(loads)

    assert result.bins is not None
    assert len(result.bins) > 0
    assert result.summary["QH_gen_out_kWh"] == pytest.approx(10.0)
    assert result.summary["QW_gen_out_kWh"] == pytest.approx(3.0)
    assert result.summary["QC_gen_out_kWh"] == pytest.approx(5.0)
    assert result.summary["E_total_electricity_kWh"] > 0
    assert result.summary["SPF_HW_gen"] > 1.0
    assert result.summary["SEER_C_gen"] > 1.0


def test_heat_pump_performance_data_14511_14825_part_load():
    """Test EN 14511 rating normalization and EN 14825 water-side correction."""
    import pybuildingenergy as pybui

    result = pybui.HeatPumpPerformanceDataCalculator(
        {
            "unit_type": "air-to-water",
            "capacity_control": "fixed",
            "heating_design_load_kW": 8.0,
            "cooling_design_load_kW": 6.0,
            "heating_degradation_coefficient": 0.9,
            "cooling_degradation_coefficient": 0.9,
            "heating_rating_points": [
                {
                    "rating_condition": "EN 14511 A7/W35",
                    "source_temperature_C": 7.0,
                    "sink_temperature_C": 35.0,
                    "capacity_kW": 10.0,
                    "cop": 4.5,
                    "part_load_ratio": 0.5,
                }
            ],
            "cooling_rating_points": [
                {
                    "rating_condition": "EN 14511 A35/W7",
                    "source_temperature_C": 35.0,
                    "sink_temperature_C": 7.0,
                    "capacity_kW": 8.0,
                    "eer": 3.5,
                    "part_load_ratio": 0.5,
                }
            ],
        }
    ).run()

    assert pybui.en14825_part_load_factor(1.0, 0.9, "air-to-water") == pytest.approx(1.0)
    assert pybui.en14825_part_load_factor(0.5, 0.9, "air-to-water") < 1.0
    assert result.heating_map["cop"].iloc[0] == pytest.approx(4.5)
    assert result.cooling_map["eer"].iloc[0] == pytest.approx(3.5)
    assert result.rating_points["performance_at_part_load"].min() < 4.5
    assert result.summary["rating_point_count"] == pytest.approx(2.0)
    assert pybui.HeatPumpPerformanceDataCalculator is not None


def test_heat_pump_uses_en14825_part_load_correction():
    """Test optional EN 14825 part-load correction in heat-pump generation."""
    import pybuildingenergy as pybui

    heating_map = pd.DataFrame(
        {
            "source_temperature_C": [7.0],
            "sink_temperature_C": [35.0],
            "capacity_kW": [10.0],
            "cop": [4.0],
        }
    )
    loads = pd.DataFrame(
        {
            "T_ext": [7.0],
            "Q_H_kWh": [2.0],
            "Q_C_kWh": [0.0],
            "Q_W_kWh": [0.0],
            "T_H_sink_C": [35.0],
        },
        index=pd.date_range("2026-01-01", periods=1, freq="h"),
    )
    calc = pybui.HeatPumpSystemCalculator(
        {
            "heating_performance_map": heating_map,
            "dhw_performance_map": heating_map,
            "source_type": "air",
            "time_step_hours": 1.0,
            "demand_unit": "kWh",
            "hp_operating_limit_C": 58.0,
            "part_load_performance_method": "en14825",
            "part_load_unit_type": "air-to-water",
            "part_load_degradation_coefficient": 0.9,
            "external_auxiliary_power_W": 0.0,
            "standby_power_W": 0.0,
            "heating_storage_loss_kWh_per_day": 0.0,
            "dhw_storage_loss_kWh_per_day": 0.0,
        }
    )

    result = calc.run_timeseries(loads)
    row = result.bins.iloc[0]

    assert row["H_part_load_ratio"] == pytest.approx(0.2)
    assert row["H_part_load_factor"] < 1.0
    assert row["H_performance"] < row["H_performance_full_load"]
    assert result.summary["EH_hp_in_kWh"] > 2.0 / 4.0


def test_emission_system_calculator_heating_cooling_effects():
    """Test EN 15316-2 emission losses and auxiliary electricity."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [-5.0, 0.0, 30.0, 34.0],
            "T_H_int_ini_C": [20.0, 20.0, 20.0, 20.0],
            "T_C_int_ini_C": [26.0, 26.0, 26.0, 26.0],
            "Q_H_kWh": [4.0, 2.0, 0.0, 0.0],
            "Q_C_kWh": [0.0, 0.0, 2.0, 3.0],
            "Q_H_em_out_inc_kWh": [4.2, 2.1, 0.0, 0.0],
            "Q_C_em_out_inc_kWh": [0.0, 0.0, 2.15, 3.2],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    calc = pybui.EmissionSystemCalculator(
        {
            "demand_unit": "kWh",
            "heating": {
                "stratification_K": 0.35,
                "control_K": 0.70,
                "hydraulic_balancing_K": 0.10,
                "room_automation_K": -0.50,
                "fan_power_W": 10.0,
                "fan_count": 2.0,
                "convective_fraction": 0.95,
            },
            "cooling": {
                "stratification_K": 0.40,
                "control_K": 0.70,
                "hydraulic_balancing_K": 0.10,
                "room_automation_K": -0.50,
                "fan_power_W": 10.0,
                "fan_count": 2.0,
                "convective_fraction": 0.95,
            },
        }
    )

    result = calc.run_timeseries(loads)

    assert result.timeseries["Q_H_em_in_kWh"].sum() > loads["Q_H_kWh"].sum()
    assert result.timeseries["Q_C_em_in_kWh"].sum() > loads["Q_C_kWh"].sum()
    assert result.summary["QH_em_ls_kWh"] == pytest.approx(0.3)
    assert result.summary["QC_em_ls_kWh"] == pytest.approx(0.35)
    assert result.summary["WH_em_aux_kWh"] > 0
    assert result.summary["WC_em_aux_kWh"] > 0
    assert result.summary["e_H_em_ls_an"] > 1.0
    assert result.summary["e_C_em_ls_an"] > 1.0
    assert pybui.EmissionSystemCalculator is not None


def test_distribution_system_calculator_heating_cooling_dhw():
    """Test EN 15316-3 distribution losses and pump auxiliaries."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [-5.0, 0.0, 30.0, 34.0],
            "T_op": [20.0, 20.0, 26.0, 26.0],
            "Q_H_kWh": [4.0, 2.0, 0.0, 0.0],
            "Q_C_kWh": [0.0, 0.0, 2.0, 3.0],
            "Q_W_kWh": [0.5, 0.5, 0.5, 0.5],
            "theta_H_dis_supply_C": [50.0, 50.0, 50.0, 50.0],
            "theta_H_dis_return_C": [40.0, 40.0, 40.0, 40.0],
            "theta_C_dis_supply_C": [8.0, 8.0, 8.0, 8.0],
            "theta_C_dis_return_C": [14.0, 14.0, 14.0, 14.0],
            "theta_W_dis_hot_C": [58.0, 58.0, 58.0, 58.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    common_section = {
        "length_m": 20.0,
        "equivalent_length_m": 0.0,
        "linear_thermal_transmittance_W_mK": 0.30,
        "ambient_temperature_C": 20.0,
        "recoverable": True,
    }
    calc = pybui.DistributionSystemCalculator(
        {
            "demand_unit": "kWh",
            "heating": {
                "pipe_sections": [common_section],
                "supply_temperature_C": 45.0,
                "return_temperature_C": 35.0,
                "nominal_power_kW": 8.0,
                "design_flow_m3_h": 0.7,
                "design_delta_pressure_kPa": 20.0,
                "pump_control_code": 4,
                "eei": 0.23,
            },
            "cooling": {
                "pipe_sections": [{**common_section, "ambient_temperature_C": 26.0}],
                "supply_temperature_C": 7.0,
                "return_temperature_C": 12.0,
                "nominal_power_kW": 6.0,
                "design_flow_m3_h": 0.8,
                "design_delta_pressure_kPa": 20.0,
                "pump_control_code": 3,
                "eei": 0.23,
            },
            "dhw": {
                "pipe_sections": [common_section],
                "dhw_temperature_C": 55.0,
                "dhw_return_deltaT_K": 5.0,
                "design_flow_m3_h": 0.12,
                "design_delta_pressure_kPa": 15.0,
                "pump_control_code": 3,
                "pump_label_power_kW": 0.02,
                "part_load_mode": "constant_when_on",
            },
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QH_dis_ls_kWh"] > 0
    assert result.summary["QC_dis_ls_kWh"] > 0
    assert result.summary["QW_dis_ls_kWh"] > 0
    assert result.summary["WH_dis_aux_kWh"] > 0
    assert result.summary["WC_dis_aux_kWh"] > 0
    assert result.summary["WW_dis_aux_kWh"] > 0
    assert result.summary["QH_dis_in_kWh"] > loads["Q_H_kWh"].sum()
    assert result.summary["QC_dis_in_kWh"] > loads["Q_C_kWh"].sum()
    assert result.summary["QW_dis_in_kWh"] > loads["Q_W_kWh"].sum()
    assert result.summary["QH_dis_rbl_kWh"] > 0
    assert result.summary["QC_dis_rbl_kWh"] < 0
    assert result.timeseries["theta_H_dis_mean_C"].iloc[0] == pytest.approx(45.0)
    assert result.timeseries["theta_C_dis_mean_C"].iloc[0] == pytest.approx(11.0)
    assert result.timeseries["theta_W_dis_mean_C"].iloc[0] == pytest.approx(55.5)
    assert pybui.DistributionSystemCalculator is not None


def test_storage_system_calculator_heating_dhw():
    """Test EN 15316-5 storage losses and pump auxiliaries."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [-5.0, 0.0, 12.0, 14.0],
            "Q_H_kWh": [4.0, 2.0, 0.0, 0.0],
            "Q_W_kWh": [0.5, 0.5, 0.5, 0.5],
            "theta_H_sto_set_C": [45.0, 45.0, 45.0, 45.0],
            "theta_W_sto_set_C": [55.0, 55.0, 55.0, 55.0],
            "storage_ambient_temperature_C": [20.0, 20.0, 20.0, 20.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    calc = pybui.StorageSystemCalculator(
        {
            "demand_unit": "kWh",
            "heating": {
                "storage_volume_l": 80.0,
                "standby_loss_coefficient_W_K": 2.0,
                "set_temperature_C": 45.0,
                "ambient_temperature_C": 20.0,
                "input_pump_power_kW": 0.03,
                "input_pump_flow_m3_h": 0.7,
                "input_pump_deltaT_K": 10.0,
                "thermal_loss_room_fraction": 0.75,
                "auxiliary_to_medium_fraction": 0.25,
            },
            "dhw": {
                "storage_volume_l": 180.0,
                "standby_loss_kWh_per_day_ref": 0.84,
                "standby_set_temperature_ref_C": 55.0,
                "standby_ambient_temperature_ref_C": 20.0,
                "set_temperature_C": 55.0,
                "ambient_temperature_C": 20.0,
                "input_pump_power_kW": 0.025,
                "input_pump_flow_m3_h": 0.4,
                "input_pump_deltaT_K": 5.0,
            },
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QH_sto_ls_kWh"] == pytest.approx(0.2)
    assert result.summary["QW_sto_ls_kWh"] == pytest.approx(0.14)
    assert result.summary["WH_sto_aux_kWh"] > 0
    assert result.summary["WW_sto_aux_kWh"] > 0
    assert result.summary["QH_sto_in_kWh"] > loads["Q_H_kWh"].sum()
    assert result.summary["QW_sto_in_kWh"] > loads["Q_W_kWh"].sum()
    assert result.summary["QH_sto_ls_rbl_kWh"] == pytest.approx(0.15)
    assert result.summary["QH_sto_ls_nrbl_kWh"] == pytest.approx(0.05)
    assert result.timeseries["theta_H_sto_set_C"].iloc[0] == pytest.approx(45.0)
    assert result.timeseries["theta_W_sto_set_C"].iloc[0] == pytest.approx(55.0)
    assert pybui.StorageSystemCalculator is not None


def test_cooling_system_calculator_16798_9_operating_conditions():
    """Test EN 16798-9 cooling operating temperatures and flow."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [24.0, 30.0, 35.0],
            "Q_C_kWh": [0.0, 2.0, 4.0],
        },
        index=pd.date_range("2026-07-01", periods=3, freq="h"),
    )
    calc = pybui.CoolingSystemCalculator(
        {
            "demand_unit": "kWh",
            "distribution_temperature_control": "CONST",
            "distribution_flow_control": "VARIABLE",
            "theta_C_dis_flw_set_C": 7.0,
            "design_deltaT_K": 5.0,
            "design_cooling_load_kW": 8.0,
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QC_dis_out_tot_req_kWh"] == pytest.approx(6.0)
    assert result.timeseries["theta_C_dis_supply_C"].dropna().iloc[-1] == pytest.approx(7.0)
    assert result.timeseries["theta_C_dis_return_C"].iloc[-1] > result.timeseries["theta_C_dis_supply_C"].iloc[-1]
    assert result.summary["q_V_C_dis_max_m3_h"] > 0
    assert pybui.CoolingSystemCalculator is not None


def test_cooling_storage_system_calculator_16798_15():
    """Test EN 16798-15 chilled storage heat gains and auxiliaries."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [30.0, 31.0, 32.0],
            "Q_C_kWh": [2.0, 3.0, 0.0],
            "T_C_sink_C": [7.0, 7.0, 7.0],
            "theta_C_dis_return_C": [12.0, 12.0, 12.0],
        },
        index=pd.date_range("2026-07-01", periods=3, freq="h"),
    )
    calc = pybui.CoolingStorageSystemCalculator(
        {
            "demand_unit": "kWh",
            "storage_volume_l": 80.0,
            "H_C_sto_tot_ls_W_K": 0.8,
            "generator_loop_loss_coefficient_W_K": 0.2,
            "distribution_loop_loss_coefficient_W_K": 0.2,
            "ambient_temperature_C": 20.0,
            "input_pump_power_kW": 0.02,
            "input_pump_flow_m3_h": 0.8,
            "input_pump_deltaT_K": 5.0,
            "output_pump_power_kW": 0.02,
            "output_pump_flow_m3_h": 0.8,
            "output_pump_deltaT_K": 5.0,
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QC_sto_out_kWh"] == pytest.approx(5.0)
    assert result.summary["QC_sto_ls_tot_kWh"] > 0
    assert result.summary["WC_sto_aux_kWh"] > 0
    assert result.summary["QC_sto_in_kWh"] > result.summary["QC_sto_out_kWh"]
    assert result.summary["QC_sto_ls_tot_rbl_kWh"] < 0
    assert pybui.CoolingStorageSystemCalculator is not None


def test_cooling_generation_system_calculator_16798_13():
    """Test EN 16798-13 compression cooling generation."""
    import pybuildingenergy as pybui

    cooling_map = pd.DataFrame(
        {
            "source_temperature_C": [25.0, 35.0, 25.0, 35.0],
            "sink_temperature_C": [7.0, 7.0, 12.0, 12.0],
            "capacity_kW": [8.0, 7.0, 8.5, 7.5],
            "eer": [4.0, 3.3, 4.2, 3.5],
        }
    )
    loads = pd.DataFrame(
        {
            "T_ext": [25.0, 35.0, 30.0],
            "Q_C_kWh": [2.0, 4.0, 0.0],
            "T_C_sink_C": [7.0, 7.0, 7.0],
        },
        index=pd.date_range("2026-07-01", periods=3, freq="h"),
    )
    calc = pybui.CoolingGenerationSystemCalculator(
        {
            "demand_unit": "kWh",
            "cooling_performance_map": cooling_map,
            "nominal_capacity_kW": 8.0,
            "control_power_kW": 0.01,
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QC_gen_in_req_kWh"] == pytest.approx(6.0)
    assert result.summary["QC_gen_in_kWh"] == pytest.approx(6.0)
    assert result.summary["EC_gen_el_in_kWh"] > 0
    assert result.summary["WC_aux_gen_kWh"] > 0
    assert result.summary["SEER_C_gen"] > 2.5
    assert result.summary["QC_gen_out_kWh"] > result.summary["QC_gen_in_kWh"]
    assert pybui.CoolingGenerationSystemCalculator is not None


def test_cooling_generation_uses_en14825_part_load_correction():
    """Test optional EN 14825 part-load correction in cooling generation."""
    import pybuildingenergy as pybui

    cooling_map = pd.DataFrame(
        {
            "source_temperature_C": [30.0],
            "sink_temperature_C": [7.0],
            "capacity_kW": [10.0],
            "eer": [4.0],
        }
    )
    loads = pd.DataFrame(
        {
            "T_ext": [30.0],
            "Q_C_kWh": [2.0],
            "T_C_sink_C": [7.0],
        },
        index=pd.date_range("2026-07-01", periods=1, freq="h"),
    )
    calc = pybui.CoolingGenerationSystemCalculator(
        {
            "demand_unit": "kWh",
            "cooling_performance_map": cooling_map,
            "part_load_performance_method": "en14825",
            "part_load_unit_type": "air-to-water",
            "part_load_degradation_coefficient": 0.9,
        }
    )

    result = calc.run_timeseries(loads)
    row = result.timeseries.iloc[0]

    assert row["f_C_PL"] == pytest.approx(0.2)
    assert row["f_C_PLF"] < 1.0
    assert row["EER_C_gen"] < row["EER_C_gen_full_load"]
    assert result.summary["EC_gen_el_in_kWh"] > 2.0 / 4.0


@pytest.mark.parametrize("fix", [True, False])
def test_sanitize_and_validate_bui(building_data, fix):
    """Test per la validazione dei dati dell'edificio"""
    import pybuildingenergy as pybui
    
    bui_result, report = pybui.sanitize_and_validate_BUI(building_data, fix=fix)
    
    assert bui_result is not None
    assert isinstance(report, list)
    
    # Verifica che non ci siano errori critici
    errors = [e for e in report if e["level"] == "ERROR"]
    assert len(errors) == 0, f"Errori trovati: {errors}"


def test_transparent_aggregation_uses_equivalent_geometry():
    """Le finestre aggregate mantengono area coerente e parapetto non cumulato."""
    import pybuildingenergy as pybui

    building_object = {
        "building_surface": [
            {
                "name": "W1",
                "type": "transparent",
                "boundary": "OUTDOORS",
                "area": 2.0,
                "u_value": 1.2,
                "g_value": 0.6,
                "width": 1.0,
                "height": 2.0,
                "parapet": 1.0,
                "orientation": {"azimuth": 90.0, "tilt": 90.0},
                "ISO52016_type_string": "W",
                "ISO52016_orientation_string": "EV",
            },
            {
                "name": "W2",
                "type": "transparent",
                "boundary": "OUTDOORS",
                "area": 2.0,
                "u_value": 1.2,
                "g_value": 0.6,
                "width": 1.0,
                "height": 2.0,
                "parapet": 1.0,
                "orientation": {"azimuth": 90.0, "tilt": 90.0},
                "ISO52016_type_string": "W",
                "ISO52016_orientation_string": "EV",
            },
        ]
    }

    aggregated = pybui.ISO52016._aggregate_surfaces_by_direction(building_object)
    assert len(aggregated["building_surface"]) == 1

    window = aggregated["building_surface"][0]
    assert pytest.approx(window["area"]) == 4.0
    assert pytest.approx(window["width"]) == 2.0
    assert pytest.approx(window["height"]) == 2.0
    assert pytest.approx(window["parapet"]) == 1.0


def test_transparent_aggregation_preserves_shading_for_identical_side_by_side_windows():
    """L'ombreggiamento della finestra equivalente deve coincidere con quello della finestra tipo."""
    import pybuildingenergy as pybui
    from pybuildingenergy.source.functions import shading_reduction_factor

    building_object = {
        "building_surface": [
            {
                "name": "W1",
                "type": "transparent",
                "boundary": "OUTDOORS",
                "area": 2.0,
                "u_value": 1.2,
                "g_value": 0.6,
                "width": 1.0,
                "height": 2.0,
                "parapet": 1.0,
                "orientation": {"azimuth": 90.0, "tilt": 90.0},
                "ISO52016_type_string": "W",
                "ISO52016_orientation_string": "EV",
            },
            {
                "name": "W2",
                "type": "transparent",
                "boundary": "OUTDOORS",
                "area": 2.0,
                "u_value": 1.2,
                "g_value": 0.6,
                "width": 1.0,
                "height": 2.0,
                "parapet": 1.0,
                "orientation": {"azimuth": 90.0, "tilt": 90.0},
                "ISO52016_type_string": "W",
                "ISO52016_orientation_string": "EV",
            },
        ]
    }

    aggregated = pybui.ISO52016._aggregate_surfaces_by_direction(building_object)
    window_eq = aggregated["building_surface"][0]

    F_single, _ = shading_reduction_factor(
        alpha_sol_t=45.0,
        phi_sol_t=90.0,
        beta_k_t=90.0,
        gamma_k_t=90.0,
        D_k_ovh_q=1.0,
        L_k_ovh_q=0.0,
        elements_shading_type="horizontal_overhang",
        H_k=2.0,
        W_k=1.0,
    )
    F_equiv, _ = shading_reduction_factor(
        alpha_sol_t=45.0,
        phi_sol_t=90.0,
        beta_k_t=90.0,
        gamma_k_t=90.0,
        D_k_ovh_q=1.0,
        L_k_ovh_q=0.0,
        elements_shading_type="horizontal_overhang",
        H_k=window_eq["height"],
        W_k=window_eq["width"],
    )

    assert pytest.approx(F_equiv) == F_single


def test_shading_window_uses_geographical_gamma_for_north(monkeypatch):
    """Per NV il gamma passato allo shading deve usare convenzione geografica (N=0)."""
    from pybuildingenergy.source import utils as utils_module

    captured_calls = []

    def _fake_shading_reduction_factor(*args, **kwargs):
        captured_calls.append(kwargs)
        return 1.0, 1.0

    monkeypatch.setattr(utils_module, "shading_reduction_factor", _fake_shading_reduction_factor)

    idx = pd.RangeIndex(1)
    calendar = pd.DataFrame({"day of year": [1], "hour of day": [12]}, index=idx)
    solar_altitude_angle = pd.Series([np.radians(45.0)], index=idx)
    solar_azimuth_angle = pd.Series([np.radians(180.0)], index=idx)
    I_dir_tot = pd.Series([500.0], index=idx)
    I_dif_tot = pd.Series([100.0], index=idx)

    building_object = {
        "building_surface": [
            {
                "name": "North Window",
                "type": "transparent",
                "orientation": {"azimuth": 0.0, "tilt": 90.0},
                "height": 1.5,
                "width": 1.2,
            }
        ]
    }

    result = utils_module.ISO52010.Shading_reduction_factor_window(
        solar_altitude_angle=solar_altitude_angle,
        solar_azimuth_angle=solar_azimuth_angle,
        I_dir_tot=I_dir_tot,
        I_dif_tot=I_dif_tot,
        calendar=calendar,
        n_timesteps=1,
        orientation="NV",
        building_object=building_object,
    )

    assert result is not None
    assert len(captured_calls) == 1
    assert captured_calls[0]["gamma_k_t"] == pytest.approx(0.0)
    # input solar azimuth in ISO convention (N=180) must be converted to geographical (N=0)
    assert captured_calls[0]["phi_sol_t"] == pytest.approx(0.0)


def test_shading_window_filters_west_with_geographical_gamma(monkeypatch):
    """Con gamma WV=270, una finestra a azimuth geografico 270 deve essere selezionata."""
    from pybuildingenergy.source import utils as utils_module

    captured_calls = []

    def _fake_shading_reduction_factor(*args, **kwargs):
        captured_calls.append(kwargs)
        return 1.0, 1.0

    monkeypatch.setattr(utils_module, "shading_reduction_factor", _fake_shading_reduction_factor)

    idx = pd.RangeIndex(1)
    calendar = pd.DataFrame({"day of year": [1], "hour of day": [16]}, index=idx)
    solar_altitude_angle = pd.Series([np.radians(30.0)], index=idx)
    solar_azimuth_angle = pd.Series([np.radians(-90.0)], index=idx)
    I_dir_tot = pd.Series([400.0], index=idx)
    I_dif_tot = pd.Series([120.0], index=idx)

    building_object = {
        "building_surface": [
            {
                "name": "West Window",
                "type": "transparent",
                "orientation": {"azimuth": 270.0, "tilt": 90.0},
                "height": 1.5,
                "width": 1.2,
            },
            {
                "name": "East Window",
                "type": "transparent",
                "orientation": {"azimuth": 90.0, "tilt": 90.0},
                "height": 1.5,
                "width": 1.2,
            },
        ]
    }

    result = utils_module.ISO52010.Shading_reduction_factor_window(
        solar_altitude_angle=solar_altitude_angle,
        solar_azimuth_angle=solar_azimuth_angle,
        I_dir_tot=I_dir_tot,
        I_dif_tot=I_dif_tot,
        calendar=calendar,
        n_timesteps=1,
        orientation="WV",
        building_object=building_object,
    )

    assert result is not None
    assert len(captured_calls) == 1
    assert captured_calls[0]["gamma_k_t"] == pytest.approx(270.0)
    # input solar azimuth in ISO convention (W=-90) -> geographical west (270)
    assert captured_calls[0]["phi_sol_t"] == pytest.approx(270.0)


def test_shading_reduction_factor_handles_wrapped_azimuth():
    """L'azimuth 270 e -90 devono essere trattati come la stessa direzione."""
    from pybuildingenergy.source.functions import shading_reduction_factor

    f_neg90, _ = shading_reduction_factor(
        alpha_sol_t=45.0,
        phi_sol_t=-90.0,
        beta_k_t=90.0,
        gamma_k_t=-90.0,
        D_k_ovh_q=0.0,
        L_k_ovh_q=0.0,
        elements_shading_type=None,
        H_k=1.5,
        W_k=1.2,
    )
    f_270, _ = shading_reduction_factor(
        alpha_sol_t=45.0,
        phi_sol_t=270.0,
        beta_k_t=90.0,
        gamma_k_t=-90.0,
        D_k_ovh_q=0.0,
        L_k_ovh_q=0.0,
        elements_shading_type=None,
        H_k=1.5,
        W_k=1.2,
    )

    assert f_neg90 == pytest.approx(1.0)
    assert f_270 == pytest.approx(f_neg90)


def test_shading_reduction_factor_is_invariant_if_phi_gamma_share_same_rotation():
    """La funzione dipende dal relativo phi-gamma, non dal riferimento assoluto."""
    from pybuildingenergy.source.functions import shading_reduction_factor

    f_geo, h_geo = shading_reduction_factor(
        alpha_sol_t=45.0,
        phi_sol_t=0.0,     # geografico: Nord
        beta_k_t=90.0,
        gamma_k_t=0.0,     # finestra Nord
        D_k_ovh_q=0.8,
        L_k_ovh_q=0.1,
        elements_shading_type="horizontal_overhang",
        H_k=1.8,
        W_k=1.2,
    )
    f_iso_shifted, h_iso_shifted = shading_reduction_factor(
        alpha_sol_t=45.0,
        phi_sol_t=180.0,   # stessa direzione ma con offset +180
        beta_k_t=90.0,
        gamma_k_t=180.0,   # stessa direzione ma con offset +180
        D_k_ovh_q=0.8,
        L_k_ovh_q=0.1,
        elements_shading_type="horizontal_overhang",
        H_k=1.8,
        W_k=1.2,
    )

    assert f_iso_shifted == pytest.approx(f_geo)
    assert h_iso_shifted == pytest.approx(h_geo)


def test_multizone_longwave_exchange_matrix_is_symmetric_and_conservative():
    """L'operatore radiativo multizone deve essere simmetrico e a somma-riga nulla."""
    from pybuildingenergy.source.utils import ISO52016

    A = np.zeros((5, 5), dtype=float)
    faces = [
        (0, 2.0, 5.0),
        (1, 1.0, 7.0),
        (2, 3.0, 4.0),
    ]
    ISO52016._add_zone_longwave_radiative_exchange(A, faces)

    rad_block = A[:3, :3]
    assert np.allclose(rad_block, rad_block.T, atol=1e-12)
    assert np.allclose(rad_block.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(A[3:, :], 0.0, atol=1e-12)
    assert np.allclose(A[:, 3:], 0.0, atol=1e-12)


def test_multizone_surface_shading_factor_area_weighted_by_zone_orientation():
    """Per superficie aggregata usa F_sh area-pesato delle finestre originali."""
    from pybuildingenergy.source.utils import ISO52016

    sim_df = pd.DataFrame(
        {
            "W_win_1": [0.2],
            "W_win_2": [0.8],
        }
    )
    shading_groups = {
        ("zone_a", "EV"): [("win_1", 1.0), ("win_2", 3.0)],
    }
    surf = {
        "name": "win_1 + win_2",
        "zone": "zone_a",
        "ISO52016_orientation_string": "EV",
    }

    f_sh = ISO52016._surface_shading_factor_from_timeseries(
        sim_df=sim_df,
        tstep=0,
        surface=surf,
        shading_components_by_zone_orientation=shading_groups,
        default_zone="zone_a",
    )
    assert f_sh == pytest.approx((0.2 * 1.0 + 0.8 * 3.0) / 4.0)

    # Fallback: no column/group available -> neutral factor
    f_sh_default = ISO52016._surface_shading_factor_from_timeseries(
        sim_df=sim_df,
        tstep=0,
        surface={"name": "unknown", "zone": "zone_b", "ISO52016_orientation_string": "WV"},
        shading_components_by_zone_orientation={},
        default_zone="zone_b",
    )
    assert f_sh_default == pytest.approx(1.0)


def test_multizone_example_ground_exchange_uses_ground_area(monkeypatch):
    """L'output orario del terreno nell'esempio deve scalare con l'area a contatto col suolo."""
    module_path = Path(__file__).resolve().parents[1] / "examples" / "multizone_free_floating_example.py"
    spec = importlib.util.spec_from_file_location("multizone_free_floating_example_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class _GroundData:
        R_gr_ve = 2.0
        Theta_gr_ve = np.arange(1.0, 13.0, dtype=float)
        thermal_bridge_heat = 0.0

    monkeypatch.setattr(
        module.ISO52016,
        "Temp_calculation_of_ground",
        lambda self, building_object, path_weather_file=None, weather_source="epw": _GroundData(),
    )

    building_object = {
        "building": {"net_floor_area": 120.0},
        "zones": [
            {"name": "Z1", "net_floor_area": 80.0},
            {"name": "Z2", "net_floor_area": 40.0},
        ],
        "building_surface": [
            {"name": "Floor_Z1", "boundary": "GROUND", "zone": "Z1", "area": 80.0},
            {"name": "Floor_Z2", "boundary": "GROUND", "zone": "Z2", "area": 40.0},
            {"name": "Roof_Z1", "boundary": "OUTDOORS", "zone": "Z1", "area": 80.0},
        ],
    }
    hourly = pd.DataFrame(
        {
            "T_air_Z1": [20.0, 18.0],
            "T_air_Z2": [21.0, 19.0],
        },
        index=pd.to_datetime(["2024-01-15 12:00:00", "2024-02-15 12:00:00"]),
    )

    out = module._compute_ground_temperature_and_exchanges(
        building_object=building_object,
        hourly_results=hourly,
        weather_path="unused.epw",
        weather_source="epw",
    )

    assert list(out["T_ground_virtual"]) == pytest.approx([1.0, 2.0])
    assert out["H_ground_Z1"].iloc[0] == pytest.approx(40.0)
    assert out["H_ground_Z2"].iloc[0] == pytest.approx(20.0)
    assert out["Q_ground_Z1"].iloc[0] == pytest.approx(40.0 * (20.0 - 1.0))
    assert out["Q_ground_Z2"].iloc[1] == pytest.approx(20.0 * (19.0 - 2.0))


def test_multizone_ground_flux_helpers_return_true_boundary_fluxes():
    """I helper del solver devono aggregare il flusso reale sulla frontiera terreno."""
    from pybuildingenergy.source.utils import (
        _build_multizone_ground_flux_links,
        _ground_fluxes_from_state,
    )

    surfaces = [
        {"name": "Floor Z1", "ISO52016_type_string": "GR", "zone": "Z1", "area": 80.0},
        {"name": "Floor Z2", "ISO52016_type_string": "GR", "zone": "Z2", "area": 40.0},
    ]
    nodes = SimpleNamespace(
        Pln=np.array([1, 1], dtype=int),
        PlnSum=np.array([0, 1], dtype=int),
    )
    ground_data = SimpleNamespace(
        R_gr_ve=2.0,
        Theta_gr_ve=np.array([15.0] * 12, dtype=float),
    )
    zone_names = ["Z1", "Z2"]
    z_idx = {"Z1": 0, "Z2": 1}

    links, zone_h = _build_multizone_ground_flux_links(
        surfaces=surfaces,
        nodes=nodes,
        zone_names=zone_names,
        z_idx=z_idx,
        ground_data=ground_data,
        sys_row_from_surface_ri=lambda ri: int(ri),
    )

    theta_state = np.array([0.0, 19.0, 17.0], dtype=float)
    t_gr, zone_flux, surface_flux = _ground_fluxes_from_state(
        theta_state=theta_state,
        month_index=0,
        ground_data=ground_data,
        ground_links=links,
        zone_names=zone_names,
    )

    assert t_gr == pytest.approx(15.0)
    assert zone_h[0] == pytest.approx(40.0)
    assert zone_h[1] == pytest.approx(20.0)
    assert zone_flux["Z1"] == pytest.approx(160.0)
    assert zone_flux["Z2"] == pytest.approx(40.0)
    assert surface_flux["Floor_Z1"] == pytest.approx(160.0)
    assert surface_flux["Floor_Z2"] == pytest.approx(40.0)


def test_multizone_solver_exports_ground_flux_columns(monkeypatch):
    """Il solver multizona deve esportare temperatura virtuale e flussi reali del terreno."""
    from pybuildingenergy.source.utils import ISO52016

    sim_df = pd.DataFrame(
        {
            "T2m": [5.0, 5.0],
            "WS10m": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
    )

    monkeypatch.setattr(
        ISO52016,
        "Weather_data_bui",
        lambda self, building_object, path_weather_file=None, weather_source="epw": SimpleNamespace(
            simulation_df=sim_df
        ),
    )
    monkeypatch.setattr(ISO52016, "_aggregate_surfaces_by_direction", lambda self, bui: bui)
    monkeypatch.setattr(
        ISO52016,
        "Number_of_nodes_element",
        lambda self, building_object: SimpleNamespace(
            Rn=2,
            Pln=np.array([1], dtype=int),
            PlnSum=np.array([0], dtype=int),
        ),
    )
    monkeypatch.setattr(
        ISO52016,
        "Conduttance_node_of_element",
        lambda self, building_object: SimpleNamespace(h_pli_eli=np.zeros((1, 1), dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Areal_heat_capacity_of_element",
        lambda self, building_object: SimpleNamespace(kappa_pli_eli=np.zeros((1, 1), dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Solar_absorption_of_element",
        lambda self, building_object: SimpleNamespace(a_sol_pli_eli=np.zeros((1, 1), dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Temp_calculation_of_ground",
        lambda self, building_object, path_weather_file=None, weather_source="epw": SimpleNamespace(
            R_gr_ve=2.0,
            Theta_gr_ve=np.array([10.0] * 12, dtype=float),
            thermal_bridge_heat=0.0,
            ground_contact_area=10.0,
        ),
    )

    building_object = {
        "building": {
            "net_floor_area": 10.0,
            "building_type_class": "Residential_apartment",
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": -100.0,
                "cooling_setpoint": 100.0,
                "heating_setback": -100.0,
                "cooling_setback": 100.0,
            },
            "system_capacities": {},
            "ventilation": {},
        },
        "zones": [
            {
                "name": "Z1",
                "net_floor_area": 10.0,
                "building_type_class": "Residential_apartment",
                "heating_setpoint": -100.0,
                "cooling_setpoint": 100.0,
                "heating_setback": -100.0,
                "cooling_setback": 100.0,
            }
        ],
        "building_surface": [
            {
                "name": "Slab Ground",
                "type": "opaque",
                "boundary": "GROUND",
                "zone": "Z1",
                "area": 10.0,
                "u_value": 1.0,
                "thermal_capacity": 0.0,
                "solar_absorptance": 0.0,
                "orientation": {"azimuth": 0.0, "tilt": 180.0},
                "sky_view_factor": 0.0,
                "convective_heat_transfer_coefficient_internal": 2.5,
                "radiative_heat_transfer_coefficient_internal": 0.0,
            }
        ],
    }

    out = ISO52016.simulate_envelope_multizone_free_floating(
        building_object=building_object,
        path_weather_file="unused.epw",
        weather_source="epw",
        include_solar=False,
        warmup_hours=0,
        use_profiles=False,
        include_internal_gains=False,
        include_ventilation=False,
        include_thermal_bridges=False,
    )

    assert "T_ground_virtual" in out.columns
    assert "H_ground_Z1" in out.columns
    assert "Q_ground_Z1" in out.columns
    assert "Q_ground_surface_Slab_Ground" in out.columns
    assert np.allclose(pd.to_numeric(out["T_ground_virtual"], errors="coerce"), 10.0)
    assert np.allclose(pd.to_numeric(out["H_ground_Z1"], errors="coerce"), 5.0)
    assert np.allclose(
        pd.to_numeric(out["Q_ground_Z1"], errors="coerce"),
        pd.to_numeric(out["Q_ground_surface_Slab_Ground"], errors="coerce"),
    )
    assert pd.to_numeric(out["Q_ground_Z1"], errors="coerce").iloc[0] > 0.0


def test_build_multizone_opaque_inside_flux_links_and_fluxes():
    """I flussi opachi lato interno devono essere esportati con segno +surface->zone."""
    from pybuildingenergy.source.utils import (
        _build_multizone_opaque_inside_flux_links,
        _opaque_inside_fluxes_from_state,
    )

    surfaces = [
        {"name": "Roof Z1", "type": "opaque", "boundary": "OUTDOORS", "zone": "Z1", "area": 10.0},
        {"name": "Slab Z1", "type": "opaque", "boundary": "GROUND", "zone": "Z1", "area": 8.0},
        {
            "name": "Wall Int",
            "type": "opaque",
            "boundary": "INTERNAL",
            "zone": "Z1",
            "adjacent_zone": "Z2",
            "area": 6.0,
        },
    ]
    nodes = SimpleNamespace(
        Pln=np.array([2, 2, 2], dtype=int),
        PlnSum=np.array([0, 2, 4], dtype=int),
    )
    h_pli_eli = np.array([[2.0, 3.0, 4.0]], dtype=float)

    links = _build_multizone_opaque_inside_flux_links(
        surfaces=surfaces,
        nodes=nodes,
        zone_names=["Z1", "Z2"],
        z_idx={"Z1": 0, "Z2": 1},
        sys_row_from_surface_ri=lambda ri: int(ri),
        h_pli_eli=h_pli_eli,
    )

    assert [link["surface_token"] for link in links] == ["Roof_Z1", "Slab_Z1"]

    theta_state = np.array([0.0, 15.0, 18.0, 20.0, 16.0], dtype=float)
    surface_flux = _opaque_inside_fluxes_from_state(
        theta_state=theta_state,
        opaque_inside_links=links,
    )

    assert surface_flux["Roof_Z1"] == pytest.approx(-60.0)
    assert surface_flux["Slab_Z1"] == pytest.approx(96.0)


def test_multizone_solver_exports_opaque_inside_flux_columns(monkeypatch):
    """Il solver multizona deve esportare i flussi opachi lato interno per superficie."""
    from pybuildingenergy.source.utils import ISO52016

    sim_df = pd.DataFrame(
        {
            "T2m": [5.0, 5.0],
            "WS10m": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
    )

    monkeypatch.setattr(
        ISO52016,
        "Weather_data_bui",
        lambda self, building_object, path_weather_file=None, weather_source="epw": SimpleNamespace(
            simulation_df=sim_df
        ),
    )
    monkeypatch.setattr(ISO52016, "_aggregate_surfaces_by_direction", lambda self, bui: bui)
    monkeypatch.setattr(
        ISO52016,
        "Number_of_nodes_element",
        lambda self, building_object: SimpleNamespace(
            Rn=3,
            Pln=np.array([2], dtype=int),
            PlnSum=np.array([0], dtype=int),
        ),
    )
    monkeypatch.setattr(
        ISO52016,
        "Conduttance_node_of_element",
        lambda self, building_object: SimpleNamespace(h_pli_eli=np.array([[2.0]], dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Areal_heat_capacity_of_element",
        lambda self, building_object: SimpleNamespace(kappa_pli_eli=np.zeros((2, 1), dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Solar_absorption_of_element",
        lambda self, building_object: SimpleNamespace(a_sol_pli_eli=np.zeros((2, 1), dtype=float)),
    )

    building_object = {
        "building": {
            "net_floor_area": 10.0,
            "building_type_class": "Residential_apartment",
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": -100.0,
                "cooling_setpoint": 100.0,
                "heating_setback": -100.0,
                "cooling_setback": 100.0,
            },
            "system_capacities": {},
            "ventilation": {},
        },
        "zones": [
            {
                "name": "Z1",
                "net_floor_area": 10.0,
                "building_type_class": "Residential_apartment",
                "heating_setpoint": -100.0,
                "cooling_setpoint": 100.0,
                "heating_setback": -100.0,
                "cooling_setback": 100.0,
            }
        ],
        "building_surface": [
            {
                "name": "Roof Test",
                "type": "opaque",
                "boundary": "OUTDOORS",
                "zone": "Z1",
                "area": 10.0,
                "u_value": 1.0,
                "thermal_capacity": 0.0,
                "solar_absorptance": 0.0,
                "orientation": {"azimuth": 0.0, "tilt": 0.0},
                "sky_view_factor": 1.0,
                "convective_heat_transfer_coefficient_internal": 1.0,
                "radiative_heat_transfer_coefficient_internal": 0.0,
                "convective_heat_transfer_coefficient_external": 1.0,
                "radiative_heat_transfer_coefficient_external": 0.0,
            }
        ],
    }

    out = ISO52016.simulate_envelope_multizone_free_floating(
        building_object=building_object,
        path_weather_file="unused.epw",
        weather_source="epw",
        include_solar=False,
        warmup_hours=0,
        use_profiles=False,
        include_internal_gains=False,
        include_ventilation=False,
        include_thermal_bridges=False,
    )

    assert "Q_opaque_inside_surface_Roof_Test" in out.columns
    q_inside = pd.to_numeric(out["Q_opaque_inside_surface_Roof_Test"], errors="coerce")
    assert np.isfinite(q_inside).all()
    assert q_inside.iloc[0] < 0.0


def test_multizone_solver_does_not_cache_global_ventilation_fallbacks_in_zones(monkeypatch):
    """Changing the global ventilation model between runs must affect the next run."""
    from pybuildingenergy.source.utils import ISO52016

    sim_df = pd.DataFrame(
        {
            "T2m": [25.0, 25.0],
            "WS10m": [1.0, 1.0],
        },
        index=pd.to_datetime(["2024-07-01 00:00:00", "2024-07-01 01:00:00"]),
    )

    monkeypatch.setattr(
        ISO52016,
        "Weather_data_bui",
        lambda self, building_object, path_weather_file=None, weather_source="epw": SimpleNamespace(
            simulation_df=sim_df
        ),
    )
    monkeypatch.setattr(ISO52016, "_aggregate_surfaces_by_direction", lambda self, bui: bui)
    monkeypatch.setattr(
        ISO52016,
        "Number_of_nodes_element",
        lambda self, building_object: SimpleNamespace(
            Rn=2,
            Pln=np.array([1], dtype=int),
            PlnSum=np.array([0], dtype=int),
        ),
    )
    monkeypatch.setattr(
        ISO52016,
        "Conduttance_node_of_element",
        lambda self, building_object: SimpleNamespace(h_pli_eli=np.zeros((1, 1), dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Areal_heat_capacity_of_element",
        lambda self, building_object: SimpleNamespace(kappa_pli_eli=np.zeros((1, 1), dtype=float)),
    )
    monkeypatch.setattr(
        ISO52016,
        "Solar_absorption_of_element",
        lambda self, building_object: SimpleNamespace(a_sol_pli_eli=np.zeros((1, 1), dtype=float)),
    )

    building_object = {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": -100.0,
                "cooling_setpoint": 100.0,
                "heating_setback": -100.0,
                "cooling_setback": 100.0,
            },
            "system_capacities": {},
            "ventilation": {
                "ventilation_type": "custom",
                "custom_heat_transfer_coefficient_ventilation": 0.5,
                "infiltration_flow_per_exterior_area_m3_s_m2": 3.0e-4,
                "infiltration_coeff_constant": 0.0,
                "infiltration_coeff_temperature": 0.0,
                "infiltration_coeff_velocity": 0.224,
                "infiltration_coeff_velocity_squared": 0.0,
                "infiltration_include_transparent_area": True,
                "infiltration_exterior_area_mode": "outdoors_only",
                "infiltration_schedule_multiplier": 1.0,
            },
        },
        "zones": [
            {
                "name": "Z1",
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "heating_setpoint": -100.0,
                "cooling_setpoint": 100.0,
                "heating_setback": -100.0,
                "cooling_setback": 100.0,
            }
        ],
        "building_surface": [
            {
                "name": "Roof Z1",
                "type": "opaque",
                "boundary": "OUTDOORS",
                "zone": "Z1",
                "area": 100.0,
                "u_value": 1.0,
                "thermal_capacity": 0.0,
                "solar_absorptance": 0.0,
                "orientation": {"azimuth": 0.0, "tilt": 0.0},
                "sky_view_factor": 1.0,
                "convective_heat_transfer_coefficient_internal": 2.5,
                "radiative_heat_transfer_coefficient_internal": 0.0,
                "convective_heat_transfer_coefficient_external": 2.5,
                "radiative_heat_transfer_coefficient_external": 0.0,
            }
        ],
    }

    out_custom = ISO52016.simulate_envelope_multizone_free_floating(
        building_object=building_object,
        path_weather_file="unused.epw",
        weather_source="epw",
        include_solar=False,
        warmup_hours=0,
        use_profiles=False,
        include_internal_gains=False,
        include_ventilation=True,
        include_thermal_bridges=False,
    )

    assert "ventilation_type" not in building_object["zones"][0]
    assert pd.to_numeric(out_custom["H_ve_Z1"], errors="coerce").iloc[0] == pytest.approx(0.5)

    building_object["building_parameters"]["ventilation"]["ventilation_type"] = "eplus_infiltration_ext_area"
    out_eplus = ISO52016.simulate_envelope_multizone_free_floating(
        building_object=building_object,
        path_weather_file="unused.epw",
        weather_source="epw",
        include_solar=False,
        warmup_hours=0,
        use_profiles=False,
        include_internal_gains=False,
        include_ventilation=True,
        include_thermal_bridges=False,
    )

    assert pd.to_numeric(out_eplus["H_ve_Z1"], errors="coerce").iloc[0] > 1.0
    assert pd.to_numeric(out_eplus["H_ve_Z1"], errors="coerce").iloc[0] != pytest.approx(
        pd.to_numeric(out_custom["H_ve_Z1"], errors="coerce").iloc[0]
    )


def test_temp_calculation_of_ground_supports_optional_energyplus_monthly_override(monkeypatch):
    """L'override EnergyPlus deve essere opzionale e non alterare il ramo ISO di default."""
    import pybuildingenergy.source.utils as utils_module
    from pybuildingenergy.source.utils import ISO52016

    sim_df = pd.DataFrame(
        {"T2m": np.linspace(5.0, 16.0, 12, dtype=float)},
        index=pd.date_range("2024-01-31", periods=12, freq="ME"),
    )

    monkeypatch.setattr(
        utils_module,
        "Calculation_ISO_52010",
        lambda building_object, path_weather_file, weather_source="epw": SimpleNamespace(sim_df=sim_df),
    )

    base_building = {
        "building": {
            "net_floor_area": 100.0,
            "exposed_perimeter": 40.0,
            "wall_thickness": 0.3,
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "cooling_setpoint": 26.0,
            },
            "simulation_options": {},
        },
        "building_surface": [
            {"name": "Floor", "boundary": "GROUND", "area": 100.0},
        ],
    }

    iso_out = ISO52016.Temp_calculation_of_ground(
        building_object=copy.deepcopy(base_building),
        path_weather_file="unused.epw",
        weather_source="epw",
    )

    monthly_ground = np.linspace(11.0, 22.0, 12, dtype=float)
    for model_name in ("monthly", "energyplus"):
        monthly_building = copy.deepcopy(base_building)
        monthly_building["building_parameters"]["simulation_options"] = {
            "ground_temperature_model": model_name,
            "ground_temperature_monthly": monthly_ground.tolist(),
        }
        monthly_out = ISO52016.Temp_calculation_of_ground(
            building_object=monthly_building,
            path_weather_file="unused.epw",
            weather_source="epw",
        )

        assert np.allclose(monthly_out.Theta_gr_ve, monthly_ground)
        assert not np.allclose(iso_out.Theta_gr_ve, monthly_ground)
        assert monthly_out.R_gr_ve == pytest.approx(iso_out.R_gr_ve)
        assert monthly_out.thermal_bridge_heat == pytest.approx(iso_out.thermal_bridge_heat)


def test_prepend_december_warmup_to_previous_year_keeps_multizone_warmup_explicit():
    """Il warm-up di dicembre deve vivere nell'anno precedente, non sovrapporsi al dicembre attivo."""
    from pybuildingenergy.source.utils import _prepend_december_warmup_to_previous_year

    idx = pd.date_range("2020-01-01 00:00:00", "2020-12-31 23:00:00", freq="h")
    df = pd.DataFrame({"T2m": np.arange(len(idx), dtype=float)}, index=idx)

    out = _prepend_december_warmup_to_previous_year(df)

    assert len(out) == len(df) + (31 * 24)
    assert out.index.is_unique
    assert out.index[0] == pd.Timestamp("2019-12-01 00:00:00")
    assert out.index[(31 * 24) - 1] == pd.Timestamp("2019-12-31 23:00:00")
    assert out.index[31 * 24] == pd.Timestamp("2020-01-01 00:00:00")


def test_eplus_infiltration_ext_area_supports_optional_energyplus_like_area_mode():
    """L'opzione energyplus_like deve includere anche superfici ground-like nell'ext_area."""
    from pybuildingenergy.source.ventilation import VentilationInternalGains

    building_outdoors_only = {
        "zone_name": "Z1",
        "building_surface": [
            {"name": "Roof", "zone": "Z1", "boundary": "OUTDOORS", "type": "opaque", "area": 80.0},
            {"name": "Wall", "zone": "Z1", "boundary": "OUTDOORS", "type": "opaque", "area": 30.0},
            {
                "name": "Slab",
                "zone": "Z1",
                "boundary": "OtherSideConditionsModel",
                "type": "opaque",
                "area": 80.0,
            },
        ],
        "building_parameters": {
            "ventilation": {
                "infiltration_flow_per_exterior_area_m3_s_m2": 1.0,
                "infiltration_coeff_constant": 1.0,
                "infiltration_coeff_temperature": 0.0,
                "infiltration_coeff_velocity": 0.0,
                "infiltration_coeff_velocity_squared": 0.0,
                "infiltration_schedule_multiplier": 1.0,
                "infiltration_include_transparent_area": True,
                "infiltration_exterior_area_mode": "outdoors_only",
            }
        },
    }

    building_energyplus_like = copy.deepcopy(building_outdoors_only)
    building_energyplus_like["building_parameters"]["ventilation"][
        "infiltration_exterior_area_mode"
    ] = "energyplus_like"

    h_outdoors_only = VentilationInternalGains.heat_transfer_coefficient_by_ventilation(
        building_outdoors_only,
        Tz=20.0,
        Te=10.0,
        u_site=0.0,
        rho_air=1.0,
        c_air=1.0,
        type_ventilation="eplus_infiltration_ext_area",
    )
    h_energyplus_like = VentilationInternalGains.heat_transfer_coefficient_by_ventilation(
        building_energyplus_like,
        Tz=20.0,
        Te=10.0,
        u_site=0.0,
        rho_air=1.0,
        c_air=1.0,
        type_ventilation="eplus_infiltration_ext_area",
    )

    assert h_outdoors_only == pytest.approx(110.0)
    assert h_energyplus_like == pytest.approx(190.0)


def test_eplus_infiltration_ext_area_supports_optional_wind_reduction_factor():
    """La riduzione del vento deve essere opzionale e lasciare invariato il default."""
    from pybuildingenergy.source.ventilation import VentilationInternalGains

    base_building = {
        "zone_name": "Z1",
        "building_surface": [
            {"name": "Roof", "zone": "Z1", "boundary": "OUTDOORS", "type": "opaque", "area": 100.0},
        ],
        "building_parameters": {
            "ventilation": {
                "infiltration_flow_per_exterior_area_m3_s_m2": 1.0,
                "infiltration_coeff_constant": 0.0,
                "infiltration_coeff_temperature": 0.0,
                "infiltration_coeff_velocity": 1.0,
                "infiltration_coeff_velocity_squared": 0.0,
                "infiltration_schedule_multiplier": 1.0,
                "infiltration_include_transparent_area": True,
                "infiltration_wind_reduction_factor": 1.0,
            }
        },
    }

    reduced_building = copy.deepcopy(base_building)
    reduced_building["building_parameters"]["ventilation"]["infiltration_wind_reduction_factor"] = 0.25

    h_default = VentilationInternalGains.heat_transfer_coefficient_by_ventilation(
        base_building,
        Tz=20.0,
        Te=10.0,
        u_site=4.0,
        rho_air=1.0,
        c_air=1.0,
        type_ventilation="eplus_infiltration_ext_area",
    )
    h_reduced = VentilationInternalGains.heat_transfer_coefficient_by_ventilation(
        reduced_building,
        Tz=20.0,
        Te=10.0,
        u_site=4.0,
        rho_air=1.0,
        c_air=1.0,
        type_ventilation="eplus_infiltration_ext_area",
    )

    assert h_default == pytest.approx(400.0)
    assert h_reduced == pytest.approx(100.0)


def test_transmission_heat_transfer_coefficient_uses_geographical_orientation_mapping():
    """Con azimuth 0 deve selezionare NV (non SV) nel calcolo Hd_zt_ztu."""
    from pybuildingenergy.source.utils import ISO52016

    adj_zone = {
        "orientation_zone": {"azimuth": 0.0},
        "area_facade_elements": np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
        "transmittance_U_elements": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        "orientation_elements": np.array(["NV", "SV", "EV", "WV"], dtype=object),
        "volume": 0.0,
    }

    H_ztu_tot, b_ztu_m, F_ztc_ztu_m = ISO52016.transmission_heat_transfer_coefficient_ISO13789(
        adj_zone, n_ue=0.0, qui=0.0
    )

    # Selected side is NV (10), others are 20+30+40 = 90.
    assert H_ztu_tot == pytest.approx(100.0)
    assert b_ztu_m == pytest.approx(0.9, abs=1e-3)
    assert F_ztc_ztu_m == pytest.approx(1.0)


@pytest.mark.slow
def test_iso52016_calculation(building_data, output_dir):
    """Test per il calcolo ISO52016 (può richiedere tempo)"""
    import pybuildingenergy as pybui
    
    # Validazione dati
    bui_checked, issues = pybui.sanitize_and_validate_BUI(building_data, fix=True)
    errors = [e for e in issues if e["level"] == "ERROR"]
    
    assert len(errors) == 0, "Errori nella validazione dei dati"
    
    # Esegui calcolo
    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui_checked,
        weather_source="pvgis"
    )
    
    # Verifica risultati
    assert hourly_sim is not None
    assert annual_results_df is not None
    assert len(hourly_sim) > 0
    assert len(annual_results_df) > 0
    
    # Salva risultati
    hourly_sim.to_csv(os.path.join(output_dir, "hourly_sim_test.csv"))
    annual_results_df.to_csv(os.path.join(output_dir, "annual_results_test.csv"))
    
    # Verifica che i file siano stati creati
    assert os.path.exists(os.path.join(output_dir, "hourly_sim_test.csv"))
    assert os.path.exists(os.path.join(output_dir, "annual_results_test.csv"))

def test_iso52016_calculation_climatedata(building_data, output_dir):
    """Test for the ISO 52016 calculation (may take time)"""
    import pybuildingenergy as pybui
    
    # Data validation
    bui_checked, issues = pybui.sanitize_and_validate_BUI(building_data, fix=True)
    errors = [e for e in issues if e["level"] == "ERROR"]
    
    assert len(errors) == 0, "Errors in data validation"
    
    # Run calculation
    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui_checked,
        weather_source="climatedata"
    )
    
    # Check results
    assert hourly_sim is not None
    assert annual_results_df is not None
    assert len(hourly_sim) > 0
    assert len(annual_results_df) > 0
    
    # Save results
    hourly_sim.to_csv(os.path.join(output_dir, "hourly_sim_test.csv"))
    annual_results_df.to_csv(os.path.join(output_dir, "annual_results_test.csv"))
    
    # Verify that the files were created
    assert os.path.exists(os.path.join(output_dir, "hourly_sim_test.csv"))
    assert os.path.exists(os.path.join(output_dir, "annual_results_test.csv"))



def test_dhw_calculation():
    """Test per il calcolo del fabbisogno di acqua calda sanitaria"""
    import pybuildingenergy as pybui
    
    # Parametri
    theta_W_draw = 42
    theta_W_cold = 11.2
    theta_w_h_ref = 60
    theta_w_c_ref = 13.5
    
    hourly_fractions = pd.DataFrame({
        "Workday": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 20, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0],
        "Weekend": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Holiday": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    
    sum_fractions = pd.DataFrame(hourly_fractions.sum())
    sum_fractions.columns = ["fractions"]
    
    # Genera calendario
    calendar_nation = "Italy"
    italy_calendar = pybui.generate_calendar(calendar_nation, 2023)
    
    n_workdays = sum(italy_calendar['values'] == 'Working')
    n_weekends = sum(italy_calendar['values'] == 'Non-Working')
    n_holidays = sum(italy_calendar['values'] == 'Holiday')
    total_days = italy_calendar.count().values[0]
    
    # Calcolo DHW
    dhw_result = pybui.Volume_and_energy_DHW_calculation(
        n_workdays, n_weekends, n_holidays, sum_fractions, total_days, hourly_fractions,
        theta_W_draw,
        theta_w_c_ref,
        theta_w_h_ref,
        theta_W_cold,
        mode_calc='number_of_units',
        building_type_B3='Residential',
        building_area=142,
        unit_count=10,
        building_type_B5='Dwelling',
        residential_typology='residential_building - simple housing - AVG',
        calculation_method='table',
        year=2015,
        country_calendar=italy_calendar
    )
    
    assert dhw_result is not None
    assert len(dhw_result) > 0
    assert len(dhw_result[6]) == total_days * 24
    assert len(dhw_result[7]) == total_days * 24


def test_dhw_en12831_3_residential_energy_need_matches_annex_b_formula():
    import pybuildingenergy as pybui
    from pybuildingenergy.global_inputs import (
        WATER_DENSITY,
        WATER_SPECIFIC_HEAT_CAPACITY,
    )

    hourly_fractions = pd.DataFrame(
        {
            "Workday": [0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 5, 5, 10, 5, 5, 5, 10, 10, 5, 5, 0, 0, 0, 0],
            "Weekend": [0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 5, 5, 10, 5, 5, 5, 10, 10, 5, 5, 0, 0, 0, 0],
            "Holiday": [0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 5, 5, 10, 5, 5, 5, 10, 10, 5, 5, 0, 0, 0, 0],
        }
    )
    sum_fractions = pd.DataFrame(hourly_fractions.sum(), columns=["fractions"])
    calendar = pybui.generate_calendar("Italy", 2023)
    n_workdays = int((calendar["values"] == "Working").sum())
    n_weekends = int((calendar["values"] == "Non-Working").sum())
    n_holidays = int((calendar["values"] == "Holiday").sum())
    total_days = int(len(calendar))

    dhw_result = pybui.Volume_and_energy_DHW_calculation(
        n_workdays=n_workdays,
        n_weekends=n_weekends,
        n_holidays=n_holidays,
        sum_fractions=sum_fractions,
        total_days=total_days,
        hourly_fractions=hourly_fractions,
        theta_W_draw=42.0,
        theta_w_c_ref=13.5,
        theta_w_h_ref=60.0,
        theta_W_cold=11.2,
        mode_calc="volume_type_bui",
        building_type_B3="Residential",
        building_area=120.0,
        unit_count=1,
        building_type_B5="Dwelling",
        residential_typology="residential_building - simple housing - AVG",
        calculation_method="table",
        year=2023,
        country_calendar=calendar,
    )

    n_p_eq_max = 0.035 * 120.0
    n_p_eq = 1.75 + 0.3 * (n_p_eq_max - 1.75)
    v_ref_m3_day = 45.0 * n_p_eq / 1000.0
    expected_v_draw_m3_day = v_ref_m3_day * (60.0 - 13.5) / (42.0 - 13.5)
    expected_q_day = (
        expected_v_draw_m3_day
        * WATER_DENSITY
        * WATER_SPECIFIC_HEAT_CAPACITY
        * (42.0 - 11.2)
    )

    assert dhw_result[1] == pytest.approx(expected_v_draw_m3_day)
    assert dhw_result[4] == pytest.approx(expected_q_day)
    assert dhw_result[0] == pytest.approx(expected_q_day * total_days)
    assert len(dhw_result[6]) == total_days * 24
    assert len(dhw_result[7]) == total_days * 24
    assert sum(dhw_result[7]) == pytest.approx(dhw_result[0])


def test_dhw_en12831_3_direct_flow_design_power_matches_formula_18():
    import pybuildingenergy as pybui

    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    dhw = pd.Series([0.0] * 8 + [0.5, 1.0, 0.5] + [0.0] * 13, index=idx)
    calc = pybui.DHWDesignLoadCalculator(
        {
            "system_type": "direct_flow",
            "design_flow_l_s": 0.12,
            "draw_temperature_C": 42.0,
            "cold_water_temperature_C": 10.0,
        }
    )

    result = calc.run_timeseries(pd.DataFrame({"Q_W_kWh": dhw}))

    expected_phi = 0.12 * 1.0 * 4.176 * (42.0 - 10.0)
    assert result.summary["phi_eff_nominal_kW"] == pytest.approx(expected_phi)
    assert result.summary["design_flow_m3_h"] == pytest.approx(0.12 * 3.6)
    assert result.summary["sizing_satisfied"] is True


def test_dhw_en12831_3_annex_b8_standby_loss_interpolates_for_180_l():
    import pybuildingenergy as pybui
    from pybuildingenergy.source.DHW import _annex_b_standby_loss_kWh_d

    expected = 1.35 + (180.0 - 150.0) / (200.0 - 150.0) * (1.56 - 1.35)
    calc = pybui.DHWDesignLoadCalculator(
        {
            "system_type": "loading_storage",
            "storage_volume_l": 180.0,
            "nominal_power_kW": 6.0,
        }
    )

    assert _annex_b_standby_loss_kWh_d(180.0) == pytest.approx(expected)
    assert calc.storage_standby_loss_kWh_d == pytest.approx(expected)
    assert calc.storage_standby_loss_source == "EN 12831-3 Annex B Table B.8"


def test_dhw_en12831_3_explicit_standby_loss_overrides_annex_b8():
    import pybuildingenergy as pybui

    calc = pybui.DHWDesignLoadCalculator(
        {
            "system_type": "loading_storage",
            "storage_volume_l": 180.0,
            "standby_loss_kWh_per_day": 0.95,
            "nominal_power_kW": 6.0,
        }
    )

    assert calc.storage_standby_loss_kWh_d == pytest.approx(0.95)
    assert calc.storage_standby_loss_source == "standby_loss_kWh_per_day"


def test_dhw_en12831_3_storage_switch_points_follow_standard_formulas():
    import pybuildingenergy as pybui
    from pybuildingenergy.global_inputs import WATER_SPECIFIC_HEAT_CAPACITY

    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    dhw = pd.Series([0.0] * 8 + [0.2, 0.4, 0.4, 0.2, 0.8, 0.3, 0.3, 0.2, 0.2] + [0.0] * 7, index=idx)
    calc = pybui.DHWDesignLoadCalculator(
        {
            "system_type": "mixed_storage",
            "sizing_mode": "check",
            "storage_volume_l": 180.0,
            "storage_height_m": 1.2,
            "sensor_relative_height": 0.5,
            "loading_factor": 0.96,
            "standby_loss_kWh_per_day": 1.0,
            "nominal_power_kW": 7.0,
            "draw_temperature_C": 42.0,
            "cold_water_temperature_C": 10.0,
            "storage_max_temperature_C": 55.0,
        }
    )

    result = calc.run_timeseries(pd.DataFrame({"Q_W_kWh": dhw}))
    c = WATER_SPECIFIC_HEAT_CAPACITY
    q_max = 180.0 * c * (55.0 - 10.0) * 0.96
    q_min = 180.0 * c * (42.0 - 10.0) * 0.96 * (1.0 - 0.5 * 0.5)

    assert len(result.timeseries) == 1440
    assert result.summary["Q_sto_max_kWh"] == pytest.approx(q_max)
    assert result.summary["Q_sto_on_kWh"] == pytest.approx(0.5 * q_max)
    assert result.summary["Q_sto_min_kWh"] == pytest.approx(q_min)
    assert result.summary["q_sb_sto_kWh_d"] == pytest.approx(1.0)
    assert result.summary["q_sb_sto_source"] == "standby_loss_kWh_per_day"
    assert result.summary["sizing_satisfied"] is True


def test_dhw_en12831_3_size_storage_increases_insufficient_tank():
    import pybuildingenergy as pybui

    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    dhw = pd.Series([0.0] * 7 + [0.6, 1.2, 1.2, 0.8, 0.6, 1.5, 1.0, 0.8, 0.6, 0.4] + [0.0] * 7, index=idx)
    calc = pybui.DHWDesignLoadCalculator(
        {
            "system_type": "loading_storage",
            "sizing_mode": "size_storage",
            "storage_volume_l": 30.0,
            "nominal_power_kW": 0.5,
            "draw_temperature_C": 42.0,
            "cold_water_temperature_C": 10.0,
            "storage_max_temperature_C": 55.0,
            "standby_loss_kWh_per_day": 0.6,
            "max_storage_volume_l": 500.0,
        }
    )

    result = calc.run_timeseries(pd.DataFrame({"Q_W_kWh": dhw}))

    assert result.summary["V_sto_selected_l"] > 30.0
    assert result.summary["sizing_satisfied"] is True


def test_dhw_en12831_3_design_sizing_aggregates_duplicate_timestamps():
    import pybuildingenergy as pybui

    idx = list(pd.date_range("2024-01-01", periods=24, freq="h"))
    idx.insert(9, idx[9])
    dhw = pd.Series([0.0] * len(idx), index=pd.DatetimeIndex(idx))
    dhw.loc["2024-01-01 09:00:00"] = [0.4, 0.6]
    dhw.loc["2024-01-01 10:00:00"] = 0.5
    calc = pybui.DHWDesignLoadCalculator(
        {
            "system_type": "loading_storage",
            "sizing_mode": "check",
            "storage_volume_l": 180.0,
            "nominal_power_kW": 6.0,
            "draw_temperature_C": 42.0,
            "cold_water_temperature_C": 10.0,
            "storage_max_temperature_C": 55.0,
        }
    )

    result = calc.run_timeseries(pd.DataFrame({"Q_W_kWh": dhw}))

    assert len(result.timeseries) == 1440
    assert result.summary["Q_W_design_day_kWh"] == pytest.approx(1.5)


def test_heat_pump_example_geometry_uses_useful_area_and_footprint_consistently():
    from examples.heat_pump_15316_4_2_example import (
        building_geometry_summary,
        example_building,
    )

    for scenario in ("athens", "bolzano"):
        building = example_building(scenario)
        summary = building_geometry_summary(building)
        b = building["building"]
        surfaces = {surface["name"]: surface for surface in building["building_surface"]}
        length = float(b["footprint_length"])
        width = float(b["footprint_width"])
        height = float(b["height"])

        assert summary["net_floor_area_m2"] == pytest.approx(
            summary["footprint_area_m2"] * summary["n_floors"]
        )
        assert surfaces["Roof surface"]["area"] == pytest.approx(summary["footprint_area_m2"])
        assert surfaces["Slab to ground"]["area"] == pytest.approx(summary["footprint_area_m2"])
        assert surfaces["North wall"]["area"] == pytest.approx(length * height)
        assert (
            surfaces["South wall"]["area"] + surfaces["South glazing"]["area"]
        ) == pytest.approx(length * height)
        assert (
            surfaces["East wall"]["area"] + surfaces["East glazing"]["area"]
        ) == pytest.approx(width * height)
        assert (
            surfaces["West wall"]["area"] + surfaces["West glazing"]["area"]
        ) == pytest.approx(width * height)


def test_heat_pump_example_dhw_design_uses_annex_b8_standby_loss_by_default():
    import pybuildingenergy as pybui
    from examples.heat_pump_15316_4_2_example import (
        dhw_design_system_config,
        heat_pump_maps,
        storage_system_config,
    )

    heating_map, _ = heat_pump_maps("bolzano")
    config = dhw_design_system_config(
        scenario="bolzano",
        distribution_config={"dhw": {"pipe_sections": [], "max_length_m": 0.0}},
        storage_config=storage_system_config("bolzano"),
        heating_map=heating_map,
        sizing_mode="check",
    )
    calc = pybui.DHWDesignLoadCalculator(config)

    assert "standby_loss_kWh_per_day" not in config
    assert calc.storage_standby_loss_kWh_d == pytest.approx(1.476)


def test_heat_pump_example_dhw_profile_aligns_cross_year_index():
    from examples.heat_pump_15316_4_2_example import make_dhw_profile

    index = pd.date_range("2009-01-01 02:00:00", periods=8760, freq="h")

    profile = make_dhw_profile(
        pd.DatetimeIndex(index),
        building_area=120.0,
        country="Greece",
        cold_water_temperature_C=11.2,
    )

    assert isinstance(profile, pd.Series)
    assert profile.index.equals(index)
    assert len(profile) == len(index)
    assert profile.notna().all()
    assert float(profile.sum()) > 0.0


def test_heat_pump_example_dhw_profile_accepts_empty_index():
    from examples.heat_pump_15316_4_2_example import make_dhw_profile

    profile = make_dhw_profile(
        pd.DatetimeIndex([]),
        building_area=120.0,
        country="Greece",
        cold_water_temperature_C=11.2,
    )

    assert isinstance(profile, pd.Series)
    assert profile.empty
    assert profile.name == "Q_W_kWh"


def test_heat_pump_example_dhw_annex_b_table_b2_profiles_are_normalized():
    from examples.heat_pump_15316_4_2_example import (
        DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT,
        DHW_DAY_TYPE_COLUMNS,
        dhw_annex_b_table_b2_hourly_fractions,
    )

    for profile_key, raw_percent in DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT.items():
        hourly_fractions = dhw_annex_b_table_b2_hourly_fractions(profile_key)

        assert len(raw_percent) == 24
        assert list(hourly_fractions.columns) == list(DHW_DAY_TYPE_COLUMNS)
        assert len(hourly_fractions) == 24
        assert (hourly_fractions >= 0.0).all().all()
        for column in DHW_DAY_TYPE_COLUMNS:
            assert hourly_fractions[column].sum() == pytest.approx(1.0)


def test_hourly_profile_generator_accepts_weekend_alias():
    from pybuildingenergy.source.generate_profile import HourlyProfileGenerator

    wd = np.zeros(24, dtype=float)
    hd = np.ones(24, dtype=float)
    category_profiles = {"ventilation": {"weekday": wd, "weekend": hd}}

    gen = HourlyProfileGenerator(
        country="IT",
        num_months=1,
        start_year=2024,
        category_profiles=category_profiles,
    )

    assert np.allclose(gen.profiles["ventilation"]["weekday"], wd)
    assert np.allclose(gen.profiles["ventilation"]["holiday"], hd)


def test_generate_category_profile_accepts_holiday_alias_from_bui():
    from pybuildingenergy.source.utils import ISO52016

    holiday_24 = [0.25] * 24
    weekday_24 = [0.75] * 24
    building_object = {
        "building": {"building_type_class": "Residential_apartment"},
        "building_parameters": {
            "internal_gains": [
                {"name": "occupants", "weekday": weekday_24, "holiday": holiday_24},
                {"name": "appliances", "weekday": weekday_24, "holiday": holiday_24},
                {"name": "lighting", "weekday": weekday_24, "holiday": holiday_24},
            ],
            "heating_profile": {"weekday": weekday_24, "holiday": holiday_24},
            "cooling_profile": {"weekday": weekday_24, "holiday": holiday_24},
            "ventilation_profile": {"weekday": weekday_24, "holiday": holiday_24},
        },
    }

    default_wd = {"Residential_apartment": [1.0] * 24}
    default_we = {"Residential_apartment": [1.0] * 24}

    category_profiles = ISO52016.generate_category_profile(
        building_object,
        default_wd,
        default_we,
        default_wd,
        default_we,
        default_wd,
        default_we,
    )

    assert np.allclose(category_profiles["ventilation"]["weekday"], weekday_24)
    assert np.allclose(category_profiles["ventilation"]["holiday"], holiday_24)
    assert np.allclose(category_profiles["occupancy"]["holiday"], holiday_24)


def test_occupancy_ventilation_uses_zone_area_and_liters_to_m3_conversion():
    from pybuildingenergy.source.ventilation import VentilationInternalGains

    base = {
        "building": {"net_floor_area": 80.0},
        "building_parameters": {"ventilation": {"flow_rate_per_person": 0.05}},
    }
    h_ve_80 = VentilationInternalGains(base).heat_transfer_coefficient_by_ventilation(
        base,
        Tz=26.0,
        Te=30.0,
        u_site=0.0,
        type_ventilation="occupancy",
    )
    expected_80 = 1.204 * 1006.0 * (0.05 * 80.0 / 1000.0)
    assert float(h_ve_80) == pytest.approx(expected_80)

    smaller = copy.deepcopy(base)
    smaller["building"]["net_floor_area"] = 40.0
    h_ve_40 = VentilationInternalGains(smaller).heat_transfer_coefficient_by_ventilation(
        smaller,
        Tz=26.0,
        Te=30.0,
        u_site=0.0,
        type_ventilation="occupancy",
    )
    assert float(h_ve_40) == pytest.approx(expected_80 / 2.0)


# ==============================================================================
#                           MARKERS
# ==============================================================================

# To run only fast tests: pytest -v -m "not slow"
# To run all tests: pytest -v
