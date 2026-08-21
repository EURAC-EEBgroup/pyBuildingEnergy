import pandas as pd
import pytest

import pybuildingenergy as pybui


def _en442_config(**overrides):
    config = {
        "emitter_type": "Radiator",
        "emitter_calculation_method": "en442",
        "emitter_rating": {
            "phi_50_kW": 1.0,
            "exponent_n": 1.3,
            "phi_30_kW": (30.0 / 50.0) ** 1.3,
            "maximum_operating_temperature_C": 80.0,
            "product_reference": "Test radiator R-1",
        },
        "circuit_design": {"design_water_deltaT_K": 20.0},
        "emission_efficiency": 100.0,
        "mixing_valve": False,
        "flow_temp_control_type": "Type 1 - Based on demand",
        "selected_emm_cont_circuit": 0,
        "distribution_loss_coeff": 0.0,
        "distribution_aux_power": 0.0,
        "calc_when_QH_positive_only": False,
        "off_compute_mode": "full",
    }
    config.update(overrides)
    return config


def test_en442_characteristic_reproduces_phi50_phi30_and_inverse():
    rating = pybui.EN442RadiatorCharacteristic(
        phi_50_kW=2.0,
        characteristic_exponent_n=1.3,
        phi_30_kW=2.0 * (30.0 / 50.0) ** 1.3,
        declared_maximum_operating_temperature_C=110.0,
    )

    assert rating.output_at_excess_temperature(50.0) == pytest.approx(2.0)
    assert rating.output_at_excess_temperature(30.0) == pytest.approx(
        rating.phi_30_kW
    )
    assert rating.required_excess_temperature(1.0) == pytest.approx(
        50.0 * 0.5 ** (1.0 / 1.3)
    )
    assert rating.coefficient_Km == pytest.approx(2.0 / 50.0**1.3)


def test_en442_rating_validates_low_temperature_output_and_scope():
    with pytest.raises(ValueError, match="phi_30_kW is inconsistent"):
        pybui.EN442RadiatorCharacteristic(
            phi_50_kW=1.0,
            characteristic_exponent_n=1.3,
            phi_30_kW=0.8,
        )

    with pytest.raises(ValueError, match="below 120"):
        pybui.EN442RadiatorCharacteristic(
            phi_50_kW=1.0,
            characteristic_exponent_n=1.3,
            declared_maximum_operating_temperature_C=120.0,
        )


def test_en442_rating_accepts_catalogue_watts_and_pressure_drop():
    rating = pybui.EN442RadiatorCharacteristic.from_dict(
        {
            "phi_50_W": 1500.0,
            "exponent_n": 1.25,
            "standard_water_flow_kg_s": 0.02,
            "pressure_drop_at_standard_flow_kPa": 4.0,
            "pressure_drop_exponent": 2.0,
        }
    )

    assert rating.phi_50_kW == pytest.approx(1.5)
    assert rating.pressure_drop_kPa(0.01) == pytest.approx(1.0)


def test_en442_optional_water_flow_characteristic_is_reversible():
    rating = pybui.EN442RadiatorCharacteristic(
        phi_50_kW=2.0,
        characteristic_exponent_n=1.3,
        water_flow_exponent=0.2,
        standard_water_flow_kg_s=0.02,
    )

    output = rating.output_at_excess_temperature(50.0, water_flow_ratio=0.5)

    assert output == pytest.approx(2.0 * 0.5**0.2)
    assert rating.required_excess_temperature(
        output, water_flow_ratio=0.5
    ) == pytest.approx(50.0)
    assert rating.output_at_excess_temperature(
        50.0, water_flow_ratio=0.0
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="positive water flow"):
        rating.required_excess_temperature(1.0, water_flow_ratio=0.0)


def test_en442_characteristic_object_selects_radiator_defaults_and_rejects_conflicts():
    rating = pybui.EN442RadiatorCharacteristic(
        phi_50_kW=1.0,
        characteristic_exponent_n=1.3,
    )
    config = _en442_config()
    config.pop("emitter_type")
    config.pop("emitter_calculation_method")
    config.pop("emitter_rating")
    config["emitter_characteristic"] = rating

    calc = pybui.HeatingSystemCalculator(config)
    checked = pybui.check_heating_system_inputs(config)

    assert calc.emitter_calculation_method == "en442"
    assert calc.EMitter_type == "Radiator"
    assert checked["config"]["emitter_calculation_method"] == "en442"
    assert checked["config"]["emitter_type"] == "Radiator"

    config["emitter_calculation_method"] = "en15316_default"
    with pytest.raises(ValueError, match="conflicts"):
        pybui.HeatingSystemCalculator(config)


def test_legacy_tb14_path_preserves_existing_radiator_curve():
    calc = pybui.HeatingSystemCalculator(
        {
            "emitter_type": "Radiator",
            "nominal_power": 1.0,
            "emission_efficiency": 100.0,
            "mixing_valve": False,
            "selected_emm_cont_circuit": 0,
        }
    )

    common = calc.calculate_common_emission_parameters(0.5, 20.0)
    emitter = calc.calculate_type_C2(common, 20.0)

    assert calc.emitter_calculation_method == "en15316_default"
    assert common["ΔθH_em_n"] == pytest.approx(20.0)
    assert emitter["ΔθH_em_air_eff"] == pytest.approx(
        50.0 * 0.5 ** (1.0 / 1.3)
    )
    assert emitter["θH_em_flow"] == pytest.approx(54.33651150011566)
    assert emitter["θH_em_ret"] == pytest.approx(44.33651150011566)


def test_en442_product_limit_caps_generation_input_and_reports_unmet_load():
    calc = pybui.HeatingSystemCalculator(_en442_config())

    out = calc.compute_step(2.0, 20.0, 5.0)

    assert out["emitter_calculation_method"] == "en442"
    assert out["emitter_product_reference"] == "Test radiator R-1"
    assert out["ΦH_em_requested(kW)"] == pytest.approx(2.0)
    assert out["ΦH_em_available(kW)"] == pytest.approx(1.0, abs=1e-8)
    assert out["ΦH_em_available_at_operating_conditions(kW)"] == pytest.approx(
        1.0, abs=1e-8
    )
    assert out["ΦH_em_eff(kW)"] == pytest.approx(1.0, abs=1e-8)
    assert out["QH_em_unmet(kWh)"] == pytest.approx(1.0, abs=1e-8)
    assert out["QH_em_i_in(kWh)"] == pytest.approx(1.0, abs=1e-8)
    assert out["QH_dis_i_req(kWh)"] == pytest.approx(1.0, abs=1e-8)
    assert out["θH_em_flow(°C)"] <= 80.0 + 1e-8
    assert out["emitter_limited"] is True


def test_en442_constant_temperature_control_limits_output_at_actual_water_temperature():
    rating = {
        "phi_50_kW": 8.0,
        "exponent_n": 1.3,
        "maximum_operating_temperature_C": 110.0,
    }
    calc = pybui.HeatingSystemCalculator(
        _en442_config(
            emitter_rating=rating,
            flow_temp_control_type="Type 3 - Constant temperature",
            constant_flow_temp=[60.0],
        )
    )

    out = calc.compute_step(8.0, 20.0, 5.0)

    assert out["ΦH_em_available(kW)"] > out["ΦH_em_requested(kW)"]
    assert out["ΦH_em_eff(kW)"] < out["ΦH_em_requested(kW)"]
    assert out["ΦH_em_eff(kW)"] == pytest.approx(
        out["ΦH_em_available_at_operating_conditions(kW)"], abs=1e-7
    )
    assert out["θH_em_flow(°C)"] == pytest.approx(60.0)
    assert out["QH_em_unmet(kWh)"] > 0.0


def test_en442_rating_and_circuit_design_delta_are_independent():
    ten_kelvin = pybui.HeatingSystemCalculator(
        _en442_config(circuit_design={"design_water_deltaT_K": 10.0})
    )
    twenty_kelvin = pybui.HeatingSystemCalculator(_en442_config())

    common_10 = ten_kelvin.calculate_common_emission_parameters(1.0, 20.0)
    common_20 = twenty_kelvin.calculate_common_emission_parameters(1.0, 20.0)
    emitter_10 = ten_kelvin.calculate_type_C2(common_10, 20.0)
    emitter_20 = twenty_kelvin.calculate_type_C2(common_20, 20.0)

    assert emitter_10["θH_em_avg"] == pytest.approx(70.0)
    assert emitter_20["θH_em_avg"] == pytest.approx(70.0)
    assert emitter_10["θH_em_flow"] == pytest.approx(75.0)
    assert emitter_20["θH_em_flow"] == pytest.approx(80.0)
    assert common_10["V_H_em_nom"] == pytest.approx(2.0 * common_20["V_H_em_nom"])


def test_en442_input_check_validates_rating_without_tb14_catalogue_name():
    config = _en442_config(emitter_type="Panel radiator catalogue model")

    checked = pybui.check_heating_system_inputs(config)

    assert checked["config"]["emitter_type"] == "Panel radiator catalogue model"
    assert checked["config"]["emitter_calculation_method"] == "en442"
    assert any("EN 442 product rating validated" in msg for msg in checked["messages"])


def test_en442_rejects_emitters_outside_scope():
    with pytest.raises(ValueError, match="not applicable"):
        pybui.HeatingSystemCalculator(_en442_config(emitter_type="Floor heating"))


def test_en442_timeseries_keeps_limit_outputs_for_idle_rows():
    calc = pybui.HeatingSystemCalculator(
        _en442_config(calc_when_QH_positive_only=True, off_compute_mode="idle")
    )
    loads = pd.DataFrame(
        {"Q_H_kWh": [0.0, 2.0], "T_op": [20.0, 20.0], "T_ext": [5.0, 5.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="h"),
    )

    result = calc.run_timeseries(loads)

    assert "QH_em_unmet(kWh)" in result
    assert result.iloc[0]["QH_em_unmet(kWh)"] == pytest.approx(0.0)
    assert result.iloc[1]["QH_em_unmet(kWh)"] == pytest.approx(1.0, abs=1e-8)


def test_en15316_2_output_is_capacity_checked_by_en442_before_distribution():
    calc = pybui.HeatingSystemCalculator(
        _en442_config(
            emission_calculation_mode="en15316-2",
            emission_15316_2_config={
                "demand_unit": "kWh",
                "heating": {
                    "stratification_K": 0.0,
                    "control_K": 0.0,
                    "hydraulic_balancing_K": 0.0,
                    "room_automation_K": 0.0,
                    "embedded_K": 0.0,
                },
            },
        )
    )
    loads = pd.DataFrame(
        {"Q_H_kWh": [2.0], "T_op": [20.0], "T_ext": [5.0]},
        index=pd.date_range("2026-01-01", periods=1, freq="h"),
    )

    result = calc.run_timeseries(loads).iloc[0]

    assert result["emission_calculation_mode"] == "en15316-2"
    assert result["emitter_calculation_method"] == "en442"
    assert result["QH_em_i_in(kWh)"] == pytest.approx(1.0, abs=1e-8)
    assert result["QH_dis_i_req(kWh)"] == pytest.approx(1.0, abs=1e-8)
    assert result["QH_em_unmet(kWh)"] == pytest.approx(1.0, abs=1e-8)


def test_en442_product_pressure_drop_enters_derived_en15316_3_configuration():
    rating = {
        "phi_50_kW": 1.0,
        "exponent_n": 1.3,
        "maximum_operating_temperature_C": 80.0,
        "standard_water_flow_kg_s": 0.02,
        "pressure_drop_at_standard_flow_kPa": 4.0,
    }
    calc = pybui.HeatingSystemCalculator(
        _en442_config(
            emitter_rating=rating,
            distribution_calculation_mode="analytical",
        )
    )

    config = calc._distribution_analytical_config(80.0, 60.0)
    design_flow_m3_h = 1.0 / (1.16 * 20.0)
    expected = 4.0 * ((design_flow_m3_h * 1000.0 / 3600.0) / 0.02) ** 2

    assert config["heating"]["additional_pressure_kPa"] == pytest.approx(expected)
