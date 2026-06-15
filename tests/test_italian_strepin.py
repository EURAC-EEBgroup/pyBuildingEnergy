from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from pybuildingenergy.data.italian_strepin import (
    apply_extra_measure_specs_to_bui,
    load_italian_strepin_tables,
)


WORKBOOK = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "strepin_archetypes"
    / "Italian_STREPIN_archetype_renovation_cost_models_visuals_reworked_fixed.xlsx"
)


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_workbook_tables_load():
    tables = load_italian_strepin_tables(WORKBOOK)

    assert len(tables.archetypes) == 18
    assert len(tables.packages) == 13
    assert len(tables.measure_levels) == 270
    assert "RMF_E1_E" in tables.archetype_ids
    assert "P09" in tables.package_ids


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_package_applies_to_bui_surfaces():
    tables = load_italian_strepin_tables(WORKBOOK)
    baseline = tables.make_case("RMF_E1_E", "P00")
    retrofit = tables.make_case("RMF_E1_E", "P09")

    def u_values(case, keyword):
        return [
            s["u_value"]
            for s in case.bui["building_surface"]
            if keyword in s["name"].lower()
        ]

    assert min(u_values(retrofit, "wall")) < min(u_values(baseline, "wall"))
    assert min(u_values(retrofit, "windows")) < min(u_values(baseline, "windows"))
    assert min(u_values(retrofit, "roof")) < min(u_values(baseline, "roof"))
    assert min(u_values(retrofit, "slab")) < min(u_values(baseline, "slab"))
    assert (
        retrofit.bui["strepin"]["heating_system"]["system_type"]
        == "air_to_water_heat_pump"
    )


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
@pytest.mark.parametrize(
    ("preset", "expected_ach_h", "expected_hve_w_k"),
    [
        ("normative_reference", 0.30, 48.114),
        ("realistic_central", 0.55, 88.209),
        ("leaky_high_behavioral", 0.75, 120.285),
    ],
)
def test_italian_strepin_ventilation_presets_for_rmf_e1_e(
    preset,
    expected_ach_h,
    expected_hve_w_k,
):
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_case("RMF_E1_E", "P00", ventilation_preset=preset)

    metadata = case.bui["strepin"]
    parameters = case.bui["building_parameters"]
    ventilation = parameters["ventilation"]
    airflow_rates = parameters["airflow_rates"]

    assert case.bui["building"]["volume"] == pytest.approx(486.0)
    assert metadata["ventilation_preset"] == preset
    assert metadata["effective_air_change_rate_h"] == pytest.approx(expected_ach_h)
    assert metadata["air_change_rate_h"] == pytest.approx(expected_ach_h)
    assert metadata["air_change_rate_override_h"] is None
    assert metadata["ventilation_hve_W_K"] == pytest.approx(expected_hve_w_k)
    assert ventilation["custom_heat_transfer_coefficient_ventilation"] == pytest.approx(
        expected_hve_w_k
    )
    assert airflow_rates["total_effective_air_change_rate_h"] == pytest.approx(
        expected_ach_h
    )
    assert airflow_rates["infiltration_rate"] == pytest.approx(0.0)
    assert parameters["ventilation_profile"]["weekday"] == [1.0] * 24
    assert parameters["ventilation_profile"]["weekend"] == [1.0] * 24
    assert "components" not in ventilation
    assert "ventilation_system" not in metadata


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_air_change_rate_override_wins_over_preset():
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_case(
        "RMF_E1_E",
        "P00",
        ventilation_preset="realistic_central",
        air_change_rate_h=0.42,
    )

    metadata = case.bui["strepin"]
    ventilation = case.bui["building_parameters"]["ventilation"]

    assert metadata["ventilation_preset"] == "realistic_central"
    assert metadata["effective_air_change_rate_h"] == pytest.approx(0.42)
    assert metadata["air_change_rate_override_h"] == pytest.approx(0.42)
    assert metadata["ventilation_hve_W_K"] == pytest.approx(67.3596)
    assert ventilation["custom_heat_transfer_coefficient_ventilation"] == pytest.approx(
        67.3596
    )


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_engine_summary_uses_engine_demands():
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_case("RMF_E1_B", "P10")
    annual = pd.DataFrame(
        [
            {
                "Q_H_annual_kWh": 12_000.0,
                "Q_C_annual_kWh": 900.0,
            }
        ]
    )

    from pybuildingenergy.data.italian_strepin import summarize_engine_performance

    out = summarize_engine_performance(case, annual, tables)

    assert out["Q_H_engine_kWh_a"] == pytest.approx(12_000.0)
    assert out["Q_C_engine_kWh_a"] == pytest.approx(900.0)
    assert out["Final_energy_kWh_m2a"] > 0.0
    assert out["Primary_nonren_kWh_m2a"] > 0.0
    assert out["Reference_workbook_primary_kWh_m2a"] is not None


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_arbitrary_measure_case_and_extra_overlays():
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_measure_case(
        "RMF_E1_B",
        {"wall": 2, "roof_floor": 2, "windows": 2, "heating": 3, "pv": 2},
    )
    case_bui = apply_extra_measure_specs_to_bui(
        case.bui,
        [
            {"measure_code": "lighting", "parameters": {"installed_power_density_W_m2": 4.0}},
            {"measure_code": "ventilation", "parameters": {"heat_recovery_efficiency": 0.75}},
            {"measure_code": "bacs", "parameters": {"bacs_class": "B"}},
            {"measure_code": "solar_thermal", "parameters": {"area_m2": 4.0}},
            {"measure_code": "battery", "parameters": {"capacity_kWh": 5.0}},
        ],
    )

    strepin = case_bui["strepin"]
    assert case.package["Heating_Level"] == 3
    assert case.package["PV_Level"] == 2
    assert strepin["heating_system"]["carrier"] == "electricity"
    assert strepin["lighting_system"]["installed_power_density_W_m2"] == pytest.approx(4.0)
    assert strepin["ventilation_system"]["heat_recovery_efficiency"] == pytest.approx(0.75)
    assert strepin["bacs"]["bacs_class"] == "B"
    assert strepin["solar_thermal"]["area_m2"] == pytest.approx(4.0)
    assert strepin["battery"]["capacity_kWh"] == pytest.approx(5.0)


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_eem6_solar_shading_overlay_sets_window_factor():
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_case("RMF_E1_E", "P00")
    case_bui = apply_extra_measure_specs_to_bui(
        case.bui,
        [
            {
                "measure_code": "solar_shading",
                "parameters": {
                    "level": "mobile",
                    "solar_transmission_factor": 0.40,
                    "activation_irradiance_W_m2": 180.0,
                },
            }
        ],
    )

    strepin = case_bui["strepin"]
    windows = [
        surface
        for surface in case_bui["building_surface"]
        if surface.get("type") == "transparent"
    ]
    shaded_windows = [
        window
        for window in windows
        if window.get("ISO52016_orientation_string") in {"EV", "SV", "WV"}
    ]
    north_windows = [
        window for window in windows if window.get("ISO52016_orientation_string") == "NV"
    ]

    assert strepin["solar_shading"]["level"] == "mobile"
    assert strepin["solar_shading"]["solar_transmission_factor"] == pytest.approx(0.40)
    assert strepin["solar_shading"]["applied_orientations"] == ["EV", "SV", "WV"]
    assert strepin["solar_shading"]["affected_window_area_m2"] == pytest.approx(
        sum(float(window["area"]) for window in shaded_windows)
    )
    assert windows
    assert shaded_windows
    assert north_windows
    assert all(window["shading_type"] == "solar_transmission_factor" for window in shaded_windows)
    assert all(
        window["solar_shading_transmittance"] == pytest.approx(0.40)
        for window in shaded_windows
    )
    assert all(
        window["solar_shading_activation_irradiance_W_m2"] == pytest.approx(180.0)
        for window in shaded_windows
    )
    assert all(window["shading_type"] == "none" for window in north_windows)


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_eem6_defaults_follow_rds_shading_assumptions():
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_case("RMF_E1_E", "P00")
    case_bui = apply_extra_measure_specs_to_bui(
        case.bui,
        [{"measure_code": "solar_shading", "parameters": {"level": "mobile"}}],
    )

    strepin = case_bui["strepin"]["solar_shading"]
    expected_cost_eur_m2 = -0.0484 * 160.0 + 31.755

    assert strepin["solar_transmission_factor"] == pytest.approx(0.20)
    assert strepin["activation_irradiance_W_m2"] is None
    assert strepin["applied_orientations"] == ["EV", "SV", "WV"]
    assert strepin["cost_EUR_m2"] == pytest.approx(expected_cost_eur_m2)
    assert strepin["investment_EUR"] == pytest.approx(
        expected_cost_eur_m2 * strepin["affected_window_area_m2"]
    )


def test_strepin_hourly_export_adds_units_to_header():
    from examples.strepin_archetypes.run_italian_strepin_engine import (
        _hourly_output_with_units,
    )

    hourly = pd.DataFrame(
        {
            "Q_HC": [1.0],
            "T_op": [20.0],
            "H_ve": [48.114],
            "S_ve": [86.6],
            "T_ve_source_eq": [1.8],
            "Q_H": [1.0],
            "Q_C": [0.0],
        },
        index=pd.to_datetime(["2020-01-01 00:00:00"]),
    )
    buffer = StringIO()

    _hourly_output_with_units(hourly).to_csv(buffer, index_label="time [local]")

    assert buffer.getvalue().splitlines()[0] == (
        "time [local],Q_HC [W],T_op [degC],H_ve [W/K],S_ve [W],"
        "T_ve_source_eq [degC],Q_H [W],Q_C [W]"
    )
