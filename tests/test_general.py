import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path


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
        'emitter_type': 'Floor heating 1',
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
    assert res["emitter_type"] == "Floor heating 1"


def test_heating_system_calculator(hvac_system_config):
    """Test per il calcolatore del sistema di riscaldamento"""
    import pybuildingenergy as pybui
    
    calc = pybui.HeatingSystemCalculator(hvac_system_config)
    assert calc is not None


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


def test_dhw_calculation():
    """Test per il calcolo del fabbisogno di acqua calda sanitaria"""
    import pybuildingenergy as pybui
    
    # Parametri
    teta_W_draw = 42
    teta_W_cold = 11.2
    teta_w_h_ref = 60
    teta_w_c_ref = 13.5
    
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
        teta_W_draw,
        teta_w_c_ref,
        teta_w_h_ref,
        teta_W_cold,
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


# ==============================================================================
#                           MARKERS
# ==============================================================================

# To run only fast tests: pytest -v -m "not slow"
# To run all tests: pytest -v