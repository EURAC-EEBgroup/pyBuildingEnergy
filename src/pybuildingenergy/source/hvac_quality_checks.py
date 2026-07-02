from __future__ import annotations


def validate_hvac_input_coherence(input_system_hvac: dict) -> dict:
    """Validate HVAC input coherence across emission, distribution and generation.

    The function raises the same ValueError conditions that were previously
    embedded in examples/new_building.py and returns a small summary of the
    evaluated values for optional logging.
    """

    emission_nominal = float(input_system_hvac.get("nominal_power", 0.0))
    emission_15316_2_nominal = float(
        input_system_hvac.get("emission_15316_2_config", {})
        .get("heating", {})
        .get("nominal_power_kW", 0.0)
    )
    distribution_nominal = float(
        input_system_hvac.get("distribution_15316_3_config", {})
        .get("heating", {})
        .get("nominal_power_kW", 0.0)
    )
    generator_full_load = float(input_system_hvac.get("full_load_power", 0.0))
    boiler_rated = float(input_system_hvac.get("boiler_generation_config", {}).get("rated_power_kW", 0.0))
    same_generator_for_heating_and_dhw = bool(input_system_hvac.get("same_generator_for_heating_and_dhw", True))
    boiler_type = str(input_system_hvac.get("boiler_generation_config", {}).get("boiler_type", "")).lower()
    fuel_type = str(input_system_hvac.get("boiler_generation_config", {}).get("fuel_type", "")).lower()
    boiler_location = str(input_system_hvac.get("boiler_generation_config", {}).get("boiler_location", "")).lower()
    speed_control_generator_pump = str(input_system_hvac.get("speed_control_generator_pump", "")).lower()
    efficiency_model = str(input_system_hvac.get("efficiency_model", "")).lower()
    off_compute_mode = str(input_system_hvac.get("off_compute_mode", "")).lower()
    heating_generator_type = str(
        input_system_hvac.get("heating_generator_type", input_system_hvac.get("generation_calculation_mode", "boiler_15316_4_1"))
    ).lower()

    allowed_heating_generator_types = {"boiler_15316_4_1", "heat_pump_15316_4_2"}
    if heating_generator_type not in allowed_heating_generator_types:
        raise ValueError(
            "HVAC input error: heating_generator_type is not supported. "
            f"Expected one of {sorted(allowed_heating_generator_types)}, got '{heating_generator_type}'."
        )

    tolerance_pct = 0.10
    emission_block_diff = abs(emission_nominal - emission_15316_2_nominal)
    emission_distribution_diff = abs(emission_nominal - distribution_nominal)
    generator_boiler_diff = abs(generator_full_load - boiler_rated)
    distribution_flow = float(
        input_system_hvac.get("distribution_15316_3_config", {})
        .get("heating", {})
        .get("design_flow_m3_h", 0.0)
    )
    distribution_deltaT = float(
        input_system_hvac.get("distribution_15316_3_config", {})
        .get("heating", {})
        .get("design_deltaT_K", 0.0)
    )
    water_heat_capacity_density = 1.15
    expected_flow = (
        emission_nominal / (water_heat_capacity_density * distribution_deltaT)
        if distribution_deltaT > 0
        else 0.0
    )
    flow_diff = abs(distribution_flow - expected_flow)

    if (
        emission_nominal <= 0
        or emission_15316_2_nominal <= 0
        or distribution_nominal <= 0
        or generator_full_load <= 0
        or (heating_generator_type == "boiler_15316_4_1" and boiler_rated <= 0)
    ):
        raise ValueError(
            "HVAC input error: nominal_power, emission_15316_2_config.heating.nominal_power_kW, "
            "distribution nominal_power_kW and full_load_power must be positive. "
            "boiler rated_power_kW must be positive when heating_generator_type='boiler_15316_4_1'."
        )

    if emission_block_diff > 0.01 * max(emission_nominal, 1e-9):
        raise ValueError(
            "HVAC input error: nominal_power and emission_15316_2_config.heating.nominal_power_kW "
            f"must match closely. Got nominal_power={emission_nominal:.3f} kW and "
            f"emission_15316_2_config.heating.nominal_power_kW={emission_15316_2_nominal:.3f} kW."
        )

    generator_power_diff = abs(generator_full_load - boiler_rated)
    if heating_generator_type == "boiler_15316_4_1" and generator_power_diff > 0.01 * max(generator_full_load, 1e-9):
        raise ValueError(
            "HVAC input error: full_load_power and boiler_generation_config.rated_power_kW "
            f"must match closely. Got full_load_power={generator_full_load:.3f} kW and "
            f"rated_power_kW={boiler_rated:.3f} kW."
        )

    dhw_nominal = float(input_system_hvac.get("dhw_generator_config", {}).get("nominal_power_kW", 0.0))
    if not same_generator_for_heating_and_dhw and dhw_nominal <= 0:
        raise ValueError(
            "HVAC input error: dhw_generator_config.nominal_power_kW must be positive "
            "when same_generator_for_heating_and_dhw is False."
        )

    allowed_fuel_map = {
        "standard": {"natural_gas"},
        "low_temperature": {"natural_gas"},
        "condensing": {"natural_gas"},
        "biomass_log": {"wood_log"},
        "biomass_pellet": {"wood_pellet"},
    }

    if heating_generator_type == "boiler_15316_4_1" and boiler_type not in allowed_fuel_map:
        raise ValueError(
            "HVAC input error: boiler_type is not supported. "
            f"Expected one of {sorted(allowed_fuel_map)}."
        )

    if heating_generator_type == "boiler_15316_4_1" and fuel_type not in allowed_fuel_map[boiler_type]:
        raise ValueError(
            "HVAC input error: boiler_type and fuel_type are not compatible. "
            f"For boiler_type='{boiler_type}' expected one of {sorted(allowed_fuel_map[boiler_type])}, "
            f"got fuel_type='{fuel_type}'."
        )

    allowed_boiler_locations = {
        "inside_heated",
        "adjacent_unheated",
        "outside_building",
    }

    if heating_generator_type == "boiler_15316_4_1" and boiler_location not in allowed_boiler_locations:
        raise ValueError(
            "HVAC input error: boiler_location is not supported. "
            f"Expected one of {sorted(allowed_boiler_locations)}, got '{boiler_location}'."
        )

    allowed_speed_control_generator_pump = {"variable", "deltaT_constant"}
    if speed_control_generator_pump not in allowed_speed_control_generator_pump:
        raise ValueError(
            "HVAC input error: speed_control_generator_pump is not supported. "
            f"Expected one of {sorted(allowed_speed_control_generator_pump)}, "
            f"got '{speed_control_generator_pump}'."
        )

    allowed_efficiency_model = {"simple", "parametric"}
    if efficiency_model not in allowed_efficiency_model:
        raise ValueError(
            "HVAC input error: efficiency_model is not supported. "
            f"Expected one of {sorted(allowed_efficiency_model)}, got '{efficiency_model}'."
        )

    allowed_off_compute_mode = {"idle", "temps", "full"}
    if off_compute_mode not in allowed_off_compute_mode:
        raise ValueError(
            "HVAC input error: off_compute_mode is not supported. "
            f"Expected one of {sorted(allowed_off_compute_mode)}, got '{off_compute_mode}'."
        )

    if emission_distribution_diff > tolerance_pct * emission_nominal:
        raise ValueError(
            "HVAC input error: emission nominal_power and distribution nominal_power_kW "
            "are not aligned. The distribution must use the same reference power as the emitter."
        )

    if heating_generator_type == "boiler_15316_4_1" and generator_boiler_diff > tolerance_pct * generator_full_load:
        raise ValueError(
            "HVAC input error: full_load_power and boiler rated_power_kW are not aligned. "
            "The boiler configuration must match the generator sizing."
        )

    if distribution_deltaT <= 0:
        raise ValueError("HVAC input error: design_deltaT_K must be positive to validate the design flow.")

    if distribution_flow <= 0:
        raise ValueError("HVAC input error: design_flow_m3_h must be positive.")

    if flow_diff > tolerance_pct * max(expected_flow, 1e-9):
        raise ValueError(
            "HVAC input error: design_flow_m3_h is not coherent with nominal_power and "
            f"design_deltaT_K. Expected design_flow_m3_h is about {expected_flow:.3f} m3/h "
            f"from Vdot = P / (c * ΔT) = {emission_nominal:.1f} / "
            f"({water_heat_capacity_density:.2f} * {distribution_deltaT:.1f}). "
            "Check the hydraulic sizing of the distribution circuit."
        )

    return {
        "emission_nominal": emission_nominal,
        "emission_15316_2_nominal": emission_15316_2_nominal,
        "distribution_nominal": distribution_nominal,
        "generator_full_load": generator_full_load,
        "boiler_rated": boiler_rated,
        "dhw_nominal": dhw_nominal,
        "same_generator_for_heating_and_dhw": same_generator_for_heating_and_dhw,
        "heating_generator_type": heating_generator_type,
        "expected_flow_m3_h": expected_flow,
        "distribution_flow_m3_h": distribution_flow,
    }
