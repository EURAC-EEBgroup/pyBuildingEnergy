"""Hydraulic-temperature coherence checks for EN 15316-1 C.2."""

import pytest

from pybuildingenergy.source.iso_15316_1 import (
    HeatingSystemCalculator,
    MultiZoneHeatingSystemCalculator,
)


def _c2_config(**overrides):
    config = {
        "emitter_type": "Floor heating",
        "nominal_power": 8.0,
        "emission_efficiency": 90.0,
        "selected_emm_cont_circuit": 0,
        "mixing_valve": False,
        "flow_temp_control_type": "Type 3 - Constant temperature",
        "constant_flow_temp": [42.0],
        "generator_circuit": "independent",
        "gen_flow_temp_control_type": "Type B",
    }
    config.update(overrides)
    return config


def _c2_emission(calc):
    common = calc.calculate_common_emission_parameters(2.0, 20.0)
    return common, calc.calculate_type_C2(common, 20.0)


def test_c2_raises_an_insufficient_node_setpoint_to_emitter_requirement():
    calc = HeatingSystemCalculator(_c2_config(constant_flow_temp=[20.0]))
    _, emission = _c2_emission(calc)

    node_temperature = calc.calculate_circuit_node_temperature(5.0, emission)

    assert node_temperature == pytest.approx(emission["θH_em_flw_min"])
    assert calc._hydraulic_alarms[0]["code"] == "C2_NODE_SETPOINT_RAISED"


def test_c2_requires_mixing_when_a_direct_circuit_node_is_too_hot():
    calc = HeatingSystemCalculator(_c2_config(constant_flow_temp=[60.0]))
    common, emission = _c2_emission(calc)
    node_temperature = calc.calculate_circuit_node_temperature(5.0, emission)

    operating = calc.calculate_operating_conditions(
        emission, common, 20.0, node_temperature
    )

    assert operating["hydraulic_status"] == "mixing_valve_required"
    assert operating["θH_dis_flw"] == pytest.approx(emission["θH_em_flow"])
    assert operating["hydraulic_alarms"][-1]["code"] == "C2_MIXING_VALVE_REQUIRED"


def test_c2_direct_circuit_is_coherent_when_node_matches_emitter_requirement():
    calc = HeatingSystemCalculator(_c2_config())
    common, emission = _c2_emission(calc)

    operating = calc.calculate_operating_conditions(
        emission, common, 20.0, emission["θH_em_flw_min"]
    )

    assert operating["hydraulic_status"] == "direct_circuit_coherent"
    assert operating["hydraulic_requirement_met"] is True


def test_multizone_c1_node_uses_maximum_requirement_for_c2_to_c5():
    systems = {
        f"zone_c{circuit}": _c2_config(
            selected_emm_cont_circuit=circuit,
            mixing_valve=True,
            flow_temp_control_type="Type 1 - Based on demand",
        )
        for circuit in range(4)
    }
    calculator = MultiZoneHeatingSystemCalculator(systems)

    result = calculator.compute_step(
        {
            zone: {"Q_H_kWh": 1.5 + circuit, "T_op": 20.0, "T_ext": 3.0}
            for circuit, zone in enumerate(systems)
        }
    )

    requirements = result["zone_temperature_requirements(°C)"]
    assert result["θH_nod_out(°C)"] == pytest.approx(max(requirements.values()))
    assert set(result["zones"]) == set(systems)
    assert all(
        zone_result["θH_nod_out(°C)"] == pytest.approx(result["θH_nod_out(°C)"])
        for zone_result in result["zones"].values()
    )
