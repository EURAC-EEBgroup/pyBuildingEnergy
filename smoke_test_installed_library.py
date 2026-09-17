"""Smoke test for the INSTALLED pybuildingenergy package.

Run this from OUTSIDE the repo (or after `pip install` in a clean venv) so
`import pybuildingenergy` resolves to the installed wheel, not to the repo's
`src/` tree.

It exercises several building envelope configurations and several HVAC
plant configurations (emission/distribution + six generator technologies)
using the same inputs already validated by the project's own test suite.

Usage:
    python smoke_test_installed_library.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import pybuildingenergy as pybui

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    try:
        fn()
        RESULTS.append((label, True, ""))
        print(f"[PASS] {label}")
    except Exception as exc:  # noqa: BLE001 - smoke test wants to catch everything
        RESULTS.append((label, False, repr(exc)))
        print(f"[FAIL] {label}: {exc!r}")
        traceback.print_exc(limit=2)


# ---------------------------------------------------------------------------
# 1. Building envelope configurations
# ---------------------------------------------------------------------------

def _base_building(**overrides) -> dict:
    building = {
        "name": "smoke-test",
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
        "construction_class": "class_i",
    }
    building.update(overrides)

    surfaces = [
        {"name": "Roof surface", "type": "opaque", "area": 130, "sky_view_factor": 1.0,
         "u_value": 2.2, "solar_absorptance": 0.4, "thermal_capacity": 741500.0,
         "orientation": {"azimuth": 0, "tilt": 0}, "name_adj_zone": None},
        {"name": "Opaque north surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5,
         "u_value": 1.4, "solar_absorptance": 0.4, "thermal_capacity": 1416240.0,
         "orientation": {"azimuth": 0, "tilt": 90}, "name_adj_zone": None},
        {"name": "Opaque south surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5,
         "u_value": 1.4, "solar_absorptance": 0.4, "thermal_capacity": 1416240.0,
         "orientation": {"azimuth": 180, "tilt": 90}, "name_adj_zone": None},
        {"name": "Opaque east surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5,
         "u_value": 1.2, "solar_absorptance": 0.6, "thermal_capacity": 1416240.0,
         "orientation": {"azimuth": 90, "tilt": 90}, "name_adj_zone": None},
        {"name": "Opaque west surface", "type": "opaque", "area": 30, "sky_view_factor": 0.5,
         "u_value": 1.2, "solar_absorptance": 0.7, "thermal_capacity": 1416240.0,
         "orientation": {"azimuth": 270, "tilt": 90}, "name_adj_zone": None},
        {"name": "Slab to ground", "type": "opaque", "area": 100, "sky_view_factor": 0.0,
         "u_value": 1.6, "solar_absorptance": 0.6, "thermal_capacity": 405801,
         "orientation": {"azimuth": 0, "tilt": 0}, "name_adj_zone": None},
        {"name": "Transparent east surface", "type": "transparent", "area": 4, "sky_view_factor": 0.5,
         "u_value": 5, "g_value": 0.726, "height": 2, "width": 1, "parapet": 1.1,
         "orientation": {"azimuth": 90, "tilt": 90}, "shading": False,
         "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5,
         "overhang_properties": {"width_of_horizontal_overhangs": 1}, "name_adj_zone": None},
        {"name": "Transparent west surface", "type": "transparent", "area": 4, "sky_view_factor": 0.5,
         "u_value": 5, "g_value": 0.726, "height": 2, "width": 1, "parapet": 1.1,
         "orientation": {"azimuth": 270, "tilt": 90}, "shading": False,
         "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5,
         "overhang_properties": {"width_of_horizontal_overhangs": 1}, "name_adj_zone": None},
    ]

    return {
        "building": building,
        "adjacent_zones": [],
        "building_surface": surfaces,
        "units": {
            "area": "m2", "u_value": "W/m2K", "thermal_capacity": "J/kgK",
            "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
            "tilt": "degrees (0=horizontal, 90=vertical)",
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0, "heating_setback": 17.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0, "units": "C",
            },
            "system_capacities": {
                "heating_capacity": 10000000.0, "cooling_capacity": 12000000.0, "units": "W",
            },
            "airflow_rates": {"infiltration_rate": 1.0, "units": "ACH"},
            "internal_gains": [
                {"name": "occupants", "full_load": 4.2,
                 "weekday": [1.0] * 6 + [0.5] * 3 + [0.1] * 4 + [0.2] * 3 + [0.5] * 3 + [0.8] * 3 + [1.0] * 2,
                 "weekend": [1.0, 1.0] + [0.8] * 20 + [1.0, 1.0]},
            ],
            "construction": {"wall_thickness": 0.3, "thermal_bridge_heat_W_K": 2, "units": "m, W/K"},
            "climate_parameters": {"coldest_month": 1, "units": "1-12"},
            "heating_profile": {"weekday": [0.0] * 5 + [1.0] * 18 + [0.0], "weekend": [0.0] * 5 + [1.0] * 18 + [0.0]},
            "cooling_profile": {"weekday": [0.0] * 5 + [1.0] * 18 + [0.0], "weekend": [0.0] * 7 + [1.0] * 15 + [0.0] * 2},
            "ventilation_profile": {"weekday": [0.0] * 5 + [1.0] * 18 + [0.0], "weekend": [0.0] * 7 + [1.0] * 15 + [0.0] * 2},
        },
    }


BUILDING_CONFIGS = {
    "alpine_heavy_100m2_1floor": _base_building(),
    "alpine_light_250m2_2floors": _base_building(
        construction_class="class_e", net_floor_area=250, n_floors=2,
    ),
    "mediterranean_60m2_apartment": _base_building(
        latitude=38.115688, longitude=13.361267,  # Palermo
        net_floor_area=60, exposed_perimeter=28,
    ),
}


def run_building_configs(weather_source: str = "pvgis"):
    for name, bui in BUILDING_CONFIGS.items():
        def _run(bui=bui):
            bui_checked, issues = pybui.sanitize_and_validate_BUI(bui, fix=True)
            errors = [i for i in issues if i["level"] == "ERROR"]
            assert not errors, f"validation errors: {errors}"
            hourly, annual = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui_checked, weather_source=weather_source
            )
            assert hourly is not None and len(hourly) > 0
            assert annual is not None and len(annual) > 0

        check(f"building envelope [{name}] -> ISO52016", _run)


# ---------------------------------------------------------------------------
# 2. Emission + distribution configurations (per-emitter HVAC terminal unit)
# ---------------------------------------------------------------------------

BASE_HVAC = {
    "nominal_power": 8,
    "emission_efficiency": 90,
    "flow_temp_control_type": "Type 2 - Based on outdoor temperature",
    "selected_emm_cont_circuit": 0,
    "mixing_valve": True,
    "mixing_valve_delta": 2,
    "heat_losses_recovered": True,
    "distribution_loss_recovery": 90,
    "simplified_approach": 80,
    "distribution_aux_recovery": 80,
    "distribution_aux_power": 30,
    "distribution_loss_coeff": 48,
    "distribution_operation_time": 1,
    "full_load_power": 27,
    "max_monthly_load_factor": 100,
    "tH_gen_i_ON": 1,
    "auxiliary_power_generator": 0,
    "fraction_of_auxiliary_power_generator": 40,
    "generator_circuit": "independent",
    "gen_flow_temp_control_type": "Type A - Based on outdoor temperature",
    "gen_outdoor_temp_data": pd.DataFrame({
        "Text_min_gen": [-7], "Text_max_gen": [15],
        "Tflw_gen_max": [60], "Tflw_gen_min": [35],
    }, index=["Generator curve"]),
    "speed_control_generator_pump": "variable",
    "generator_nominal_deltaT": 20,
    "efficiency_model": "simple",
    "calc_when_QH_positive_only": False,
    "off_compute_mode": "full",
}

EMITTER_CONFIGS = {
    "floor_heating": {**BASE_HVAC, "emitter_type": "Floor heating"},
    "radiator": {**BASE_HVAC, "emitter_type": "Radiator"},
    "fan_coil": {**BASE_HVAC, "emitter_type": "Fan coil"},
}


def run_emitter_configs():
    for name, cfg in EMITTER_CONFIGS.items():
        def _run(cfg=cfg):
            checked = pybui.check_heating_system_inputs(cfg)
            assert checked["config"]["emitter_type"] == cfg["emitter_type"]
            calc = pybui.HeatingSystemCalculator(cfg)
            out = calc.compute_step(4.0, 20.0, 5.0)
            assert out["QH_dis_i_in(kWh)"] >= 0.0

        check(f"HVAC emitter [{name}]", _run)


# ---------------------------------------------------------------------------
# 3. Generation-side technologies (heat/DHW/cooling generators)
# ---------------------------------------------------------------------------

def _sample_loads() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=24, freq="h")
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


def run_generation_configs():
    loads = _sample_loads()

    def _condensing_boiler():
        result = pybui.BoilerGeneratorCalculator({
            "boiler_type": "condensing", "fuel_type": "natural_gas",
            "rated_power_kW": 24.0, "intermediate_load_fraction": 0.30,
            "eta_Pn_test_pct": 98.0, "eta_Pint_test_pct": 106.0,
            "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 40.0,
            "f_corr_pct_per_K": 0.04, "P_gen_ls_P0_W": 100.0,
            "P_aux_on_W": 80.0, "P_aux_off_W": 5.0, "f_jacket": 0.40,
            "f_location": 1.0, "f_aux_recoverable": 0.75, "dew_point_C": 55.0,
            "condensing_gain_pct": 11.0, "boiler_location": "inside_heated",
            "efficiency_table": {
                "condensing": {
                    "eta_Pn_test_pct": 98.0, "eta_Pint_test_pct": 106.0,
                    "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 40.0,
                }
            },
            "loss_table": {"condensing": {"P_gen_ls_P0_W": 100.0}},
        }).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)
        assert result["E_gen_in(kWh)"] > 10.0

    def _heat_pump():
        heating_map = pd.DataFrame({
            "source_temperature_C": [-7, -7, 2, 2, 7, 7],
            "sink_temperature_C": [35, 55, 35, 55, 35, 55],
            "capacity_kW": [5.0, 4.0, 6.0, 5.0, 7.0, 6.0],
            "cop": [3.2, 2.4, 3.8, 2.8, 4.2, 3.2],
        })
        hp_loads = pd.DataFrame({
            "T_ext": [-5.0, 0.0, 5.0, 12.0, 25.0, 30.0],
            "Q_H_kWh": [4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
            "Q_C_kWh": [0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
            "Q_W_kWh": [0.5] * 6,
        }, index=pd.date_range("2026-01-01", periods=6, freq="h"))
        result = pybui.HeatPumpSystemCalculator({
            "heating_performance_map": heating_map, "dhw_performance_map": heating_map,
            "cooling_performance_map": heating_map.rename(columns={"cop": "eer"}),
            "source_type": "air", "time_step_hours": 1.0, "demand_unit": "kWh",
            "hp_operating_limit_C": 58.0, "dhw_target_temperature_C": 55.0,
            "dhw_sink_temperature_C": 55.0, "external_auxiliary_power_W": 100.0,
            "standby_power_W": 5.0, "heating_storage_loss_kWh_per_day": 0.1,
            "dhw_storage_loss_kWh_per_day": 0.2,
        }).run_timeseries(hp_loads)
        assert result.summary["E_total_electricity_kWh"] > 0

    def _biomass_boiler():
        result = pybui.BiomassBoilerSystemCalculator(
            {"nominal_power_kW": 20.0}
        ).run_timeseries(loads)
        assert result.summary["E_biomass_delivered_kWh"] > result.summary["QHW_gen_out_kWh"]

    def _combustion_boiler():
        result = pybui.CombustionBoilerSystemCalculator(
            {"nominal_power_kW": 20.0, "full_load_efficiency": 0.94}
        ).run_timeseries(loads)
        assert result.summary["eta_HW_gen"] > 0.9

    def _district_heating():
        result = pybui.DistrictEnergySystemCalculator(
            {"cooling_enabled": True, "heating_substation_efficiency": 0.97}
        ).run_timeseries(loads)
        assert result.summary["E_total_district_kWh"] > result.summary["QHW_gen_out_kWh"]

    def _cogeneration():
        result = pybui.CogenerationSystemCalculator({
            "nominal_thermal_power_kW": 5.0, "thermal_efficiency": 0.56,
            "electrical_efficiency": 0.30,
        }).run_timeseries(loads)
        assert result.summary["E_chp_el_generated_kWh"] > 0.0

    generators = {
        "condensing_boiler": _condensing_boiler,
        "heat_pump": _heat_pump,
        "biomass_boiler": _biomass_boiler,
        "combustion_boiler": _combustion_boiler,
        "district_heating": _district_heating,
        "cogeneration": _cogeneration,
    }
    for name, fn in generators.items():
        check(f"HVAC generation [{name}]", fn)


# ---------------------------------------------------------------------------
# 4. Ventilation (bonus: not strictly "generation" but part of the HVAC chain)
# ---------------------------------------------------------------------------

def run_ventilation_config():
    def _run():
        loads = _sample_loads().assign(T_zone=20.0)
        result = pybui.VentilationSystemCalculator({
            "volume_m3": 300.0, "air_change_rate_h": 0.5,
            "heat_recovery_efficiency": 0.75, "specific_fan_power_W_s_m3": 800.0,
        }).run_timeseries(loads)
        assert result.summary["E_vent_fan_kWh"] > 0.0

    check("HVAC ventilation [heat-recovery unit]", _run)


if __name__ == "__main__":
    print(f"pybuildingenergy version: {getattr(pybui, '__version__', 'unknown')}")
    print(f"pybuildingenergy path:    {pybui.__file__}\n")

    weather_source = sys.argv[1] if len(sys.argv) > 1 else "pvgis"
    print(f"--- Building envelope configurations (weather_source={weather_source}) ---")
    run_building_configs(weather_source=weather_source)

    print("\n--- HVAC emission/distribution configurations ---")
    run_emitter_configs()

    print("\n--- HVAC generation-side configurations ---")
    run_generation_configs()

    print("\n--- HVAC ventilation configuration ---")
    run_ventilation_config()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"\n=== {n_pass} passed, {n_fail} failed (of {len(RESULTS)}) ===")
    if n_fail:
        for name, ok, err in RESULTS:
            if not ok:
                print(f"  FAILED: {name} -> {err}")
        sys.exit(1)
