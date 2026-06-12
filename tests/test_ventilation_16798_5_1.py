"""Tests for the solver-independent EN 16798-5-1 sensible AHU model."""

import math

import pytest

from pybuildingenergy.source.ventilation_16798_5_1 import (
    AHUStepInputs,
    CompensationPoint,
    FanPerformanceModel,
    FrostControlMode,
    HeatRecoveryControl,
    SensibleAHUConfig,
    SupplyTemperatureControl,
    SupplyTemperatureControlMode,
    _RHO_CP_J_M3_K,
    ahu_outputs_to_ventilation_stream,
    calculate_sensible_ahu_step,
    resolve_supply_temperature_setpoint,
    sensible_ahu_config_from_dict,
)
from pybuildingenergy.source.ventilation import VentilationBoundary, VentilationStream


def _point(reference_c, supply_c):
    return CompensationPoint(reference_c, supply_c)


# ---------------------------------------------------------------------------
# Supply-temperature controller tests
# ---------------------------------------------------------------------------

def test_fixed_supply_temperature_control():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.FIXED,
        setpoint_c=18.0,
    )

    assert resolve_supply_temperature_setpoint(control, -10.0, 22.0) == 18.0


def test_scheduled_supply_temperature_control():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.SCHEDULED,
    )

    assert (
        resolve_supply_temperature_setpoint(
            control,
            outdoor_temperature_c=-10.0,
            extract_temperature_c=22.0,
            scheduled_setpoint_c=17.5,
        )
        == 17.5
    )


def test_scheduled_control_requires_runtime_setpoint():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.SCHEDULED,
    )

    with pytest.raises(ValueError, match="requires scheduled_setpoint_c"):
        resolve_supply_temperature_setpoint(control, -10.0, 22.0)


def test_outdoor_compensation_interpolates_multiple_segments():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
        points=(
            _point(10.0, 18.0),
            _point(-10.0, 22.0),
            _point(0.0, 20.0),
        ),
    )

    assert resolve_supply_temperature_setpoint(control, -5.0, 22.0) == 21.0
    assert resolve_supply_temperature_setpoint(control, 5.0, 22.0) == 19.0


@pytest.mark.parametrize(
    ("extract_temperature_c", "expected_supply_temperature_c"),
    [
        (20.0, 19.0),
        (21.0, 19.0),
        (22.0, 18.0),
        (23.0, 17.0),
        (24.0, 17.0),
    ],
)
def test_extract_compensation_interpolates_and_clamps(
    extract_temperature_c,
    expected_supply_temperature_c,
):
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
        points=(
            _point(21.0, 19.0),
            _point(23.0, 17.0),
        ),
    )

    assert (
        resolve_supply_temperature_setpoint(
            control,
            outdoor_temperature_c=-5.0,
            extract_temperature_c=extract_temperature_c,
        )
        == expected_supply_temperature_c
    )


def test_compensation_limits_apply_after_interpolation():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
        points=(
            _point(-10.0, 25.0),
            _point(10.0, 15.0),
        ),
        minimum_c=17.0,
        maximum_c=21.0,
    )

    assert resolve_supply_temperature_setpoint(control, -10.0, 22.0) == 21.0
    assert resolve_supply_temperature_setpoint(control, 10.0, 22.0) == 17.0


def test_duplicate_compensation_reference_temperature_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
            points=(
                _point(21.0, 19.0),
                _point(21.0, 17.0),
            ),
        )


@pytest.mark.parametrize(
    "mode",
    [
        SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
        SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
    ],
)
def test_compensated_control_requires_two_points(mode):
    with pytest.raises(ValueError, match="at least two points"):
        SupplyTemperatureControl(
            mode=mode,
            points=(_point(21.0, 19.0),),
        )


def test_invalid_minimum_and_maximum_rejected():
    with pytest.raises(ValueError, match="minimum_c must be <= maximum_c"):
        SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.FIXED,
            setpoint_c=18.0,
            minimum_c=20.0,
            maximum_c=19.0,
        )


def test_unsupported_extrapolation_rejected():
    with pytest.raises(ValueError, match="endpoint-clamped"):
        SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
            points=(
                _point(21.0, 19.0),
                _point(23.0, 17.0),
            ),
            extrapolation="linear",
        )


@pytest.mark.parametrize(
    "bad_value",
    [math.nan, math.inf, -math.inf],
)
def test_non_finite_compensation_input_rejected(bad_value):
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
        points=(
            _point(21.0, 19.0),
            _point(23.0, 17.0),
        ),
    )

    with pytest.raises(ValueError, match="extract_temperature_c must be finite"):
        resolve_supply_temperature_setpoint(control, -5.0, bad_value)


def test_control_accepts_string_mode_and_freezes_sorted_points():
    source_points = [
        _point(23.0, 17.0),
        _point(21.0, 19.0),
    ]
    control = SupplyTemperatureControl(
        mode="extract_compensated",
        points=source_points,
    )
    source_points.append(_point(25.0, 16.0))

    assert control.mode is SupplyTemperatureControlMode.EXTRACT_COMPENSATED
    assert tuple(point.reference_temperature_c for point in control.points) == (
        21.0,
        23.0,
    )


# ---------------------------------------------------------------------------
# AHU model test helpers
# ---------------------------------------------------------------------------

_Q_M3_H = 1350.0  # nominal flow rate used in most tests
_Q_M3_S = _Q_M3_H / 3600.0

_FIXED_18 = SupplyTemperatureControl(
    mode=SupplyTemperatureControlMode.FIXED,
    setpoint_c=18.0,
)


def _cfg(**kw) -> SensibleAHUConfig:
    defaults = dict(
        sensible_heat_recovery_efficiency=0.75,
        supply_temperature_control=_FIXED_18,
        heat_recovery_control="modulating_bypass",
        frost_control="exhaust_limit",
        heating_coil_max_power_w=None,
        cooling_coil_enabled=False,
        supply_fan_specific_power_w_per_m3_s=0.0,
        extract_fan_specific_power_w_per_m3_s=0.0,
        supply_fan_heat_fraction_to_air=1.0,
        extract_fan_heat_fraction_to_air=0.0,
        fan_performance_model="en16798_5_1",
    )
    defaults.update(kw)
    return SensibleAHUConfig(**defaults)


def _inp(**kw) -> AHUStepInputs:
    defaults = dict(
        outdoor_temperature_c=-9.7,
        extract_temperature_c=21.0,
        required_supply_flow_m3_h=_Q_M3_H,
        required_extract_flow_m3_h=_Q_M3_H,
        operation_fraction=1.0,
    )
    defaults.update(kw)
    return AHUStepInputs(**defaults)


def _rho_cp_q(q_m3_s: float = _Q_M3_S) -> float:
    return _RHO_CP_J_M3_K * q_m3_s


# ---------------------------------------------------------------------------
# SensibleAHUConfig validation
# ---------------------------------------------------------------------------

def test_config_accepts_string_enum_values():
    cfg = _cfg(heat_recovery_control="always_on", frost_control="none")
    assert cfg.heat_recovery_control is HeatRecoveryControl.ALWAYS_ON
    assert cfg.frost_control is FrostControlMode.NONE


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "match"),
    [
        ({"heat_recovery_control": "turbo"}, ValueError, "heat_recovery_control"),
        ({"frost_control": "preheat"}, ValueError, "frost_control"),
        ({"sensible_heat_recovery_efficiency": 1.01}, ValueError, "sensible_heat_recovery_efficiency"),
        ({"sensible_heat_recovery_efficiency": -0.1}, ValueError, "sensible_heat_recovery_efficiency"),
        ({"sensible_heat_recovery_efficiency": math.nan}, ValueError, None),
        ({"supply_fan_specific_power_w_per_m3_s": -1.0}, ValueError, "supply_fan_specific_power"),
        ({"supply_fan_heat_fraction_to_air": 1.1}, ValueError, "supply_fan_heat_fraction"),
        ({"heating_coil_max_power_w": -100.0}, ValueError, "heating_coil_max_power_w"),
        ({"cooling_coil_enabled": True, "cooling_coil_max_power_w": -100.0}, ValueError, "cooling_coil_max_power_w"),
        ({"supply_temperature_control": "fixed:18"}, TypeError, "supply_temperature_control"),
    ],
)
def test_config_rejects_invalid_input(kwargs, exc_type, match):
    with pytest.raises(exc_type, match=match):
        _cfg(**kwargs)


# ---------------------------------------------------------------------------
# AHUStepInputs validation
# ---------------------------------------------------------------------------

def test_inputs_default_to_full_timestep_operation():
    inp = AHUStepInputs(
        outdoor_temperature_c=-5.0,
        extract_temperature_c=21.0,
        required_supply_flow_m3_h=3600.0,
        required_extract_flow_m3_h=3600.0,
    )
    assert inp.operation_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"required_supply_flow_m3_h": -100.0}, "required_supply_flow_m3_h"),
        ({"operation_fraction": 1.5}, "operation_fraction"),
        ({"operation_fraction": -0.1}, "operation_fraction"),
        ({"flow_fraction": 1.1}, "flow_fraction"),
        ({"flow_fraction": -0.1}, "flow_fraction"),
        ({"outdoor_temperature_c": math.nan}, "outdoor_temperature_c"),
        ({"scheduled_setpoint_c": math.inf}, "scheduled_setpoint_c"),
        ({"required_extract_flow_m3_h": 0.0}, "balanced-flow"),
    ],
)
def test_inputs_rejects_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _inp(**kwargs)


def test_inputs_both_zero_flows_accepted():
    inp = _inp(required_supply_flow_m3_h=0.0, required_extract_flow_m3_h=0.0)
    assert inp.required_supply_flow_m3_h == 0.0
    assert inp.required_extract_flow_m3_h == 0.0


# ---------------------------------------------------------------------------
# calculate_sensible_ahu_step: type guards
# ---------------------------------------------------------------------------

def test_step_type_guards():
    with pytest.raises(TypeError, match="config"):
        calculate_sensible_ahu_step("not a config", _inp())
    with pytest.raises(TypeError, match="inputs"):
        calculate_sensible_ahu_step(_cfg(), "not inputs")


# ---------------------------------------------------------------------------
# Zero operation
# ---------------------------------------------------------------------------

def test_zero_operation_produces_zero_flow_and_energy():
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.0))
    assert out.actual_supply_flow_m3_h == 0.0
    assert out.actual_extract_flow_m3_h == 0.0
    assert out.fan_electric_power_w == 0.0
    assert out.heat_recovery_power_w == 0.0
    assert out.required_heating_coil_power_w == 0.0
    assert out.actual_heating_coil_power_w == 0.0
    assert out.bypass_fraction == 0.0
    assert not out.frost_protection_required


def test_zero_operation_requested_setpoint_is_none():
    # AHU is off: no setpoint resolved; None signals "not applicable".
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.0))
    assert out.requested_supply_temperature_c is None


def test_scheduled_control_off_ahu_does_not_require_setpoint():
    # Scheduled mode with no setpoint must not raise when AHU is off.
    ctrl = SupplyTemperatureControl(mode="scheduled")
    cfg = _cfg(supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=0.0))
    assert out.actual_supply_flow_m3_h == 0.0
    assert out.actual_heating_coil_power_w == pytest.approx(0.0)


def test_zero_required_flow_equivalent_to_zero_operation():
    out = calculate_sensible_ahu_step(
        _cfg(), _inp(required_supply_flow_m3_h=0.0, required_extract_flow_m3_h=0.0)
    )
    assert out.actual_supply_flow_m3_h == 0.0
    assert out.fan_electric_power_w == 0.0


# ---------------------------------------------------------------------------
# Actual flows scale with operation_fraction
# ---------------------------------------------------------------------------

def test_actual_flows_scale_with_operation_fraction():
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.5))
    assert out.actual_supply_flow_m3_h == pytest.approx(0.5 * _Q_M3_H)
    assert out.actual_extract_flow_m3_h == pytest.approx(0.5 * _Q_M3_H)


# ---------------------------------------------------------------------------
# Heat recovery
# ---------------------------------------------------------------------------

def test_zero_hr_efficiency_no_heat_recovery():
    cfg = _cfg(sensible_heat_recovery_efficiency=0.0)
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-9.7))
    assert out.heat_recovery_power_w == pytest.approx(0.0, abs=1e-9)
    # All heating must come from the coil
    expected_coil = _rho_cp_q() * (18.0 - (-9.7))
    assert out.required_heating_coil_power_w == pytest.approx(expected_coil, rel=1e-6)
    assert out.actual_heating_coil_power_w == pytest.approx(expected_coil, rel=1e-6)


def test_nominal_hr_efficiency_warms_supply():
    # T_outdoor=-9.7, T_extract=21, eta=0.75
    # T_hr = -9.7 + 0.75*(21 - (-9.7)) = -9.7 + 23.025 = 13.325°C
    out = calculate_sensible_ahu_step(_cfg(), _inp())

    t_hr = -9.7 + 0.75 * (21.0 - (-9.7))
    expected_hr = _rho_cp_q() * (t_hr - (-9.7))
    assert out.heat_recovery_power_w == pytest.approx(expected_hr, rel=1e-6)

    expected_coil = _rho_cp_q() * (18.0 - t_hr)
    assert out.required_heating_coil_power_w == pytest.approx(expected_coil, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-9)


def test_sensible_energy_balance_closes():
    """Q_hr + Q_coil = rho*cp*q*(T_supply - T_outdoor)"""
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    total_supply = _rho_cp_q() * (
        out.actual_supply_temperature_c - (-9.7)
    )
    assert total_supply == pytest.approx(
        out.heat_recovery_power_w + out.actual_heating_coil_power_w, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Modulating bypass
# ---------------------------------------------------------------------------

def test_bypass_activates_when_hr_overshoots_setpoint_in_heating_mode():
    # T_outdoor=10, T_extract=22, eta=0.75 → T_hr=10+0.75*12=19°C > setpoint=16°C
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=16.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    t_hr_full = 10.0 + 0.75 * 12.0  # 19.0°C
    expected_bp = (t_hr_full - 16.0) / (t_hr_full - 10.0)
    assert out.bypass_fraction == pytest.approx(expected_bp, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(16.0, abs=1e-6)
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_bypass_activates_in_economizer_mode_to_avoid_overcooling():
    # T_outdoor=25, T_extract=21, eta=0.75 → T_hr=25+0.75*(21-25)=22°C < setpoint=23°C < 25°C
    # HR overcools; bypass warms supply by mixing in outdoor air
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=23.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=25.0, extract_temperature_c=21.0)
    )

    t_hr_full = 25.0 + 0.75 * (21.0 - 25.0)  # 22.0°C
    expected_bp = (t_hr_full - 23.0) / (t_hr_full - 25.0)  # (22-23)/(22-25) = 1/3
    assert out.bypass_fraction == pytest.approx(expected_bp, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(23.0, abs=1e-6)


def test_no_bypass_when_hr_undershoots_setpoint():
    # HR cannot reach setpoint (T_hr < setpoint); heating coil required, no bypass
    out = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=-9.7))
    t_hr = -9.7 + 0.75 * (21.0 - (-9.7))  # 13.325°C < 18°C setpoint
    assert out.bypass_fraction == pytest.approx(0.0, abs=1e-9)
    assert out.heat_recovery_power_w > 0.0
    assert out.required_heating_coil_power_w > 0.0


def test_bypass_zero_when_hr_output_equals_setpoint():
    # T_outdoor=10, T_extract=22, eta=0.75 → T_hr=19°C; setpoint=19°C → no bypass
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=19.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    assert out.bypass_fraction == pytest.approx(0.0, abs=1e-9)


def test_always_on_hr_does_not_bypass_even_when_setpoint_below_hr_output():
    # HR overshoots setpoint but always_on → no bypass
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=16.0)
    cfg = _cfg(
        heat_recovery_control="always_on",
        supply_temperature_control=ctrl,
        sensible_heat_recovery_efficiency=0.75,
    )
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    assert out.bypass_fraction == pytest.approx(0.0, abs=1e-9)
    t_hr_full = 10.0 + 0.75 * 12.0  # 19.0°C
    assert out.actual_supply_temperature_c == pytest.approx(t_hr_full, abs=1e-9)


# ---------------------------------------------------------------------------
# Frost control
# ---------------------------------------------------------------------------

def test_frost_control_activates_when_exhaust_would_freeze():
    # T_extract=21, T_outdoor=-15, eta=0.75
    # T_EHA_nominal = 21 - 0.75*(21-(-15)) = 21 - 27 = -6°C < limit -5°C → frost
    out = calculate_sensible_ahu_step(
        _cfg(frost_exhaust_limit_c=-5.0),
        _inp(outdoor_temperature_c=-15.0, extract_temperature_c=21.0),
    )
    assert out.frost_protection_required


def test_frost_control_reduces_hr_efficiency_to_keep_exhaust_at_limit():
    t_ext, t_oda = 21.0, -15.0
    dT = t_ext - t_oda  # 36 K
    frost_limit = -5.0
    eta_frost = (t_ext - frost_limit) / dT  # 26/36

    out = calculate_sensible_ahu_step(
        _cfg(sensible_heat_recovery_efficiency=0.75, frost_exhaust_limit_c=frost_limit),
        _inp(outdoor_temperature_c=t_oda, extract_temperature_c=t_ext),
    )

    t_hr_expected = t_oda + eta_frost * dT
    expected_hr_power = _rho_cp_q() * (t_hr_expected - t_oda)
    assert out.heat_recovery_power_w == pytest.approx(expected_hr_power, rel=1e-5)


def test_frost_control_not_active_when_exhaust_above_limit():
    # T_extract=21, T_outdoor=-5, eta=0.75
    # T_EHA = 21 - 0.75*(21-(-5)) = 21 - 19.5 = 1.5°C > -5°C → no frost
    out = calculate_sensible_ahu_step(
        _cfg(frost_exhaust_limit_c=-5.0),
        _inp(outdoor_temperature_c=-5.0, extract_temperature_c=21.0),
    )
    assert not out.frost_protection_required


def test_frost_mode_none_never_activates():
    cfg = _cfg(frost_control="none", sensible_heat_recovery_efficiency=0.9)
    # Very cold outdoor, would normally trigger frost
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-20.0))
    assert not out.frost_protection_required


# ---------------------------------------------------------------------------
# Heating coil
# ---------------------------------------------------------------------------

def test_unlimited_coil_always_reaches_setpoint():
    cfg = _cfg(heating_coil_max_power_w=None)
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-20.0))
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-9)
    assert out.actual_heating_coil_power_w == pytest.approx(
        out.required_heating_coil_power_w, rel=1e-9
    )


def test_capacity_limited_coil_misses_setpoint():
    # Limit to 100 W — far below what is needed for cold outdoor
    cfg = _cfg(heating_coil_max_power_w=100.0)
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-9.7))
    assert out.actual_heating_coil_power_w == pytest.approx(100.0, rel=1e-9)
    assert out.actual_heating_coil_power_w < out.required_heating_coil_power_w
    assert out.actual_supply_temperature_c < 18.0


def test_no_heating_needed_when_hr_meets_setpoint_exactly():
    # T_outdoor=10, T_extract=22, eta=0.5 → T_hr=10+0.5*12=16°C; setpoint=16°C
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=16.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.5, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)
    assert out.actual_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_coil_capacity_scales_with_operation_fraction():
    """Time-averaged available coil power = rated_power * operation_fraction.

    For ON/OFF scheduling the coil runs at rated power during the ON period.
    The supply temperature rise during that period is coil_max / rho_cp_q_nominal,
    identical to full operation — only the time-averaged energy differs.
    """
    # Choose coil_max < required_at_full so the coil is capacity-limited.
    t_oda = -9.7
    t_hr = t_oda + 0.75 * (21.0 - t_oda)  # 13.325°C (no bypass, setpoint > t_hr)
    rho_cp_q_nom = _RHO_CP_J_M3_K * _Q_M3_S
    required_at_full = rho_cp_q_nom * (18.0 - t_hr)  # ~2 233 W
    coil_max = required_at_full * 0.6  # definitely capacity-limited at any op fraction

    cfg = _cfg(heating_coil_max_power_w=coil_max)

    out_full = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=1.0))
    out_half = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=0.5))

    assert out_full.actual_heating_coil_power_w == pytest.approx(coil_max, rel=1e-9)
    assert out_half.actual_heating_coil_power_w == pytest.approx(coil_max * 0.5, rel=1e-9)
    assert out_half.actual_supply_temperature_c == pytest.approx(
        out_full.actual_supply_temperature_c, abs=1e-6
    )


def test_zero_coil_max_delivers_no_heat():
    cfg = _cfg(heating_coil_max_power_w=0.0)
    out = calculate_sensible_ahu_step(cfg, _inp())
    assert out.actual_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)
    assert out.required_heating_coil_power_w > 0.0  # coil is needed but absent


# ---------------------------------------------------------------------------
# Cooling coil — delivery, capacity, and shortfall
# ---------------------------------------------------------------------------

def test_disabled_cooling_full_bypass_when_setpoint_below_outdoor():
    # T_outdoor=5, T_extract=22, setpoint=2°C < T_outdoor — unattainable without cooling.
    # MODULATING_BYPASS: bp = (17.75 - 2) / (17.75 - 5) = 1.235 → clamped to 1.0.
    # Full bypass delivers T_outdoor (5°C); coil has nothing to do.
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=2.0)
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.75,
        supply_temperature_control=ctrl,
        cooling_coil_enabled=False,
    )
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=5.0, extract_temperature_c=22.0)
    )
    assert out.actual_supply_temperature_c == pytest.approx(5.0, abs=1e-6)
    assert out.bypass_fraction == pytest.approx(1.0, abs=1e-9)
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)
    # Cooling shortfall: full bypass delivers T_outdoor=5°C; setpoint=2°C needs 3 K more
    expected_cooling = _rho_cp_q() * (5.0 - 2.0)
    assert out.required_cooling_coil_power_w == pytest.approx(expected_cooling, rel=1e-6)
    # Disabled: required reports the shortfall but nothing is delivered.
    assert out.actual_cooling_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_enabled_cooling_reaches_setpoint_below_outdoor():
    # Same boundary as the disabled case, but with the cooling coil active the
    # residual 3 K (after full bypass to T_outdoor=5°C) is removed to reach 2°C.
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=2.0)
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.75,
        supply_temperature_control=ctrl,
        cooling_coil_enabled=True,
    )
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=5.0, extract_temperature_c=22.0)
    )
    expected_cooling = _rho_cp_q() * (5.0 - 2.0)
    assert out.required_cooling_coil_power_w == pytest.approx(expected_cooling, rel=1e-6)
    assert out.actual_cooling_coil_power_w == pytest.approx(expected_cooling, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(2.0, abs=1e-6)
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_cooling_capacity_limit_caps_delivered_power_and_supply():
    # Cooling coil capped at 1 K worth of power: it removes 1 K of the 3 K
    # required, so supply settles at 4°C and the required value still reports 3 K.
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=2.0)
    coil_max = _rho_cp_q() * 1.0
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.75,
        supply_temperature_control=ctrl,
        cooling_coil_enabled=True,
        cooling_coil_max_power_w=coil_max,
    )
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=5.0, extract_temperature_c=22.0)
    )
    assert out.actual_cooling_coil_power_w == pytest.approx(coil_max, rel=1e-9)
    assert out.required_cooling_coil_power_w == pytest.approx(
        _rho_cp_q() * 3.0, rel=1e-6
    )
    assert out.actual_supply_temperature_c == pytest.approx(4.0, abs=1e-6)


def test_cooling_unmet_load_zero_in_normal_heating_mode():
    # Normal winter: HR undershoots setpoint → heating coil needed, no cooling shortfall
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    assert out.required_cooling_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_enabled_cooling_idle_in_heating_mode():
    # Cooling coil enabled but boundary calls for heating: cooling stays at zero
    # and the heating coil does the work — the two coils never co-activate.
    out = calculate_sensible_ahu_step(_cfg(cooling_coil_enabled=True), _inp())
    assert out.actual_cooling_coil_power_w == pytest.approx(0.0, abs=1e-9)
    assert out.required_cooling_coil_power_w == pytest.approx(0.0, abs=1e-9)
    assert out.actual_heating_coil_power_w > 0.0


# ---------------------------------------------------------------------------
# Fan heat and electricity
# ---------------------------------------------------------------------------

def test_extract_fan_heat_warms_extract_before_hr():
    """Extract fan heat increases HR recovery by warming the extract air."""
    sfp = 500.0  # W/(m³/s)
    p_eta = _Q_M3_S * sfp
    dt_eta = p_eta / (_RHO_CP_J_M3_K * _Q_M3_S)

    cfg_with = _cfg(
        extract_fan_specific_power_w_per_m3_s=sfp,
        extract_fan_heat_fraction_to_air=1.0,
        sensible_heat_recovery_efficiency=0.75,
    )
    cfg_without = _cfg(extract_fan_specific_power_w_per_m3_s=0.0)

    out_with = calculate_sensible_ahu_step(cfg_with, _inp())
    out_without = calculate_sensible_ahu_step(cfg_without, _inp())

    assert out_with.heat_recovery_power_w > out_without.heat_recovery_power_w

    # Precise check: T_ext_at_hr increases by dt_eta
    t_ext_at_hr = 21.0 + dt_eta
    dT = t_ext_at_hr - (-9.7)
    t_hr = -9.7 + 0.75 * dT
    expected_hr = _rho_cp_q() * (t_hr - (-9.7))
    assert out_with.heat_recovery_power_w == pytest.approx(expected_hr, rel=1e-5)


def test_extract_fan_heat_zero_fraction_does_not_affect_hr():
    """Extract fan heat fraction=0 means no heat added to extract airstream."""
    sfp = 500.0
    cfg_zero_frac = _cfg(
        extract_fan_specific_power_w_per_m3_s=sfp,
        extract_fan_heat_fraction_to_air=0.0,
    )
    cfg_no_fan = _cfg(extract_fan_specific_power_w_per_m3_s=0.0)

    out_zero_frac = calculate_sensible_ahu_step(cfg_zero_frac, _inp())
    out_no_fan = calculate_sensible_ahu_step(cfg_no_fan, _inp())

    # Same HR power; electricity differs
    assert out_zero_frac.heat_recovery_power_w == pytest.approx(
        out_no_fan.heat_recovery_power_w, rel=1e-9
    )
    assert out_zero_frac.fan_electric_power_w > out_no_fan.fan_electric_power_w


def test_supply_fan_heat_absorbed_by_coil_target_adjustment():
    """Coil targets setpoint minus fan heat rise so delivered equals setpoint."""
    sfp = 500.0
    p_sup = _Q_M3_S * sfp
    dt_sup = p_sup / (_RHO_CP_J_M3_K * _Q_M3_S)

    cfg = _cfg(
        supply_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=1.0,
    )
    out = calculate_sensible_ahu_step(cfg, _inp())

    # Fan is downstream of the coil; the coil targets setpoint - dt_sup so that
    # actual delivered temperature equals the 18°C setpoint after fan heat.
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-5)
    assert out.fan_electric_power_w == pytest.approx(p_sup, rel=1e-9)


def test_supply_fan_zero_heat_fraction_does_not_raise_supply():
    sfp = 500.0
    cfg = _cfg(
        supply_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=0.0,
    )
    out = calculate_sensible_ahu_step(cfg, _inp())
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-6)
    assert out.fan_electric_power_w == pytest.approx(_Q_M3_S * sfp, rel=1e-9)


def test_fan_electricity_is_separate_from_thermal_heating():
    sfp = 300.0
    cfg = _cfg(
        supply_fan_specific_power_w_per_m3_s=sfp,
        extract_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=0.0,
        extract_fan_heat_fraction_to_air=0.0,
    )
    out = calculate_sensible_ahu_step(cfg, _inp())

    # Fan electricity = both fans, no heat to airstreams
    expected_electric = 2.0 * _Q_M3_S * sfp
    assert out.fan_electric_power_w == pytest.approx(expected_electric, rel=1e-9)

    # Coil power is same as without fans (no thermal interaction)
    out_no_fan = calculate_sensible_ahu_step(_cfg(), _inp())
    assert out.required_heating_coil_power_w == pytest.approx(
        out_no_fan.required_heating_coil_power_w, rel=1e-6
    )


def test_fan_electricity_zero_when_no_fans_configured():
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    assert out.fan_electric_power_w == pytest.approx(0.0, abs=1e-9)


def test_en16798_fan_part_load_matches_reference_relations():
    """Part-load pressure follows r² and fan efficiency follows sqrt(r)."""
    flow_fraction = 0.25
    sfp = 500.0
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.0,
        heating_coil_max_power_w=0.0,
        supply_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=1.0,
        fan_performance_model="en16798_5_1",
    )

    out_design = calculate_sensible_ahu_step(cfg, _inp())
    out_part = calculate_sensible_ahu_step(
        cfg,
        _inp(flow_fraction=flow_fraction),
    )

    # With q_ref=q_design and eta_ref=eta_nom:
    # P/P_design = r * r² / sqrt(r) = r**2.5.
    assert out_part.fan_electric_power_w == pytest.approx(
        out_design.fan_electric_power_w * flow_fraction ** 2.5,
        rel=1e-9,
    )

    dt_design = out_design.actual_supply_temperature_c - (-9.7)
    dt_part = out_part.actual_supply_temperature_c - (-9.7)
    # Delta-T = delta_p/(rho*cp*eta), hence r²/sqrt(r) = r**1.5.
    assert dt_part == pytest.approx(dt_design * flow_fraction ** 1.5, rel=1e-9)


def test_en16798_fan_explicit_product_data_matches_part_load_equations():
    flow_fraction = 0.4
    design_pressure_pa = 500.0
    eta_nom = 0.72
    q_ref_m3_h = 900.0
    eta_ref = 0.68
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.0,
        heating_coil_max_power_w=0.0,
        supply_fan_specific_power_w_per_m3_s=0.0,
        supply_fan_design_pressure_pa=design_pressure_pa,
        supply_fan_nominal_efficiency=eta_nom,
        supply_fan_reference_flow_m3_h=q_ref_m3_h,
        supply_fan_reference_efficiency=eta_ref,
        supply_fan_heat_fraction_to_air=1.0,
        fan_performance_model="en16798_5_1",
    )
    out = calculate_sensible_ahu_step(cfg, _inp(flow_fraction=flow_fraction))

    q_actual_m3_h = _Q_M3_H * flow_fraction
    q_actual_m3_s = q_actual_m3_h / 3600.0
    pressure_pa = design_pressure_pa * flow_fraction ** 2
    efficiency = eta_nom * (eta_ref / eta_nom) * math.sqrt(
        q_actual_m3_h / q_ref_m3_h
    )
    expected_power_w = q_actual_m3_s * pressure_pa / efficiency
    expected_dt_k = pressure_pa / (_RHO_CP_J_M3_K * efficiency)

    assert out.fan_electric_power_w == pytest.approx(expected_power_w, rel=1e-9)
    assert out.actual_supply_temperature_c - (-9.7) == pytest.approx(
        expected_dt_k,
        rel=1e-9,
    )


def test_constant_sfp_mode_remains_available():
    flow_fraction = 0.25
    sfp = 500.0
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.0,
        heating_coil_max_power_w=0.0,
        supply_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=1.0,
        fan_performance_model="constant_sfp",
    )
    out_design = calculate_sensible_ahu_step(cfg, _inp())
    out_part = calculate_sensible_ahu_step(
        cfg,
        _inp(flow_fraction=flow_fraction),
    )

    assert out_part.fan_electric_power_w == pytest.approx(
        out_design.fan_electric_power_w * flow_fraction,
        rel=1e-9,
    )
    assert out_part.actual_supply_temperature_c == pytest.approx(
        out_design.actual_supply_temperature_c,
        abs=1e-9,
    )


def test_operation_fraction_averages_power_without_changing_fan_temperature_rise():
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.0,
        heating_coil_max_power_w=0.0,
        supply_fan_specific_power_w_per_m3_s=500.0,
        supply_fan_heat_fraction_to_air=1.0,
    )
    out_full = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=1.0))
    out_half_time = calculate_sensible_ahu_step(
        cfg,
        _inp(operation_fraction=0.5),
    )

    assert out_half_time.fan_electric_power_w == pytest.approx(
        0.5 * out_full.fan_electric_power_w,
        rel=1e-9,
    )
    assert out_half_time.actual_supply_temperature_c == pytest.approx(
        out_full.actual_supply_temperature_c,
        abs=1e-9,
    )


def test_extract_fan_zero_heat_fraction_supported_in_en16798_mode_at_part_load():
    cfg_with_fan = _cfg(
        extract_fan_specific_power_w_per_m3_s=500.0,
        extract_fan_heat_fraction_to_air=0.0,
        fan_performance_model="en16798_5_1",
    )
    cfg_without_fan = _cfg(extract_fan_specific_power_w_per_m3_s=0.0)
    inp = _inp(flow_fraction=0.25)

    out_with_fan = calculate_sensible_ahu_step(cfg_with_fan, inp)
    out_without_fan = calculate_sensible_ahu_step(cfg_without_fan, inp)

    assert out_with_fan.fan_electric_power_w > 0.0
    assert out_with_fan.heat_recovery_power_w == pytest.approx(
        out_without_fan.heat_recovery_power_w,
        rel=1e-9,
    )


# ---------------------------------------------------------------------------
# Scheduled supply-temperature control in AHU step
# ---------------------------------------------------------------------------

def test_scheduled_control_uses_scheduled_setpoint_c_from_inputs():
    ctrl = SupplyTemperatureControl(mode="scheduled")
    cfg = _cfg(supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(cfg, _inp(scheduled_setpoint_c=20.0))
    assert out.requested_supply_temperature_c == pytest.approx(20.0)
    assert out.actual_supply_temperature_c == pytest.approx(20.0, abs=1e-6)


def test_scheduled_control_raises_without_setpoint():
    ctrl = SupplyTemperatureControl(mode="scheduled")
    cfg = _cfg(supply_temperature_control=ctrl)
    with pytest.raises(ValueError, match="requires scheduled_setpoint_c"):
        calculate_sensible_ahu_step(cfg, _inp())


# ---------------------------------------------------------------------------
# Requested setpoint always reported
# ---------------------------------------------------------------------------

def test_requested_setpoint_matches_controller_output():
    ctrl = SupplyTemperatureControl(
        mode="extract_compensated",
        points=(_point(21.0, 19.0), _point(23.0, 17.0)),
    )
    cfg = _cfg(supply_temperature_control=ctrl)
    inp = _inp(extract_temperature_c=22.0)
    out = calculate_sensible_ahu_step(cfg, inp)
    assert out.requested_supply_temperature_c == pytest.approx(18.0)  # midpoint


# ---------------------------------------------------------------------------
# Operation fraction partial load
# ---------------------------------------------------------------------------

def test_partial_load_scales_all_power_outputs():
    out_full = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=1.0))
    out_half = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.5))

    assert out_half.actual_supply_flow_m3_h == pytest.approx(
        0.5 * out_full.actual_supply_flow_m3_h
    )
    assert out_half.heat_recovery_power_w == pytest.approx(
        0.5 * out_full.heat_recovery_power_w, rel=1e-6
    )
    assert out_half.required_heating_coil_power_w == pytest.approx(
        0.5 * out_full.required_heating_coil_power_w, rel=1e-6
    )


# ---------------------------------------------------------------------------
# VentilationStream adapter tests
# ---------------------------------------------------------------------------

def test_adapter_stream_attributes():
    """Adapter produces a VentilationStream with correct type, name, category, and physics."""
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    stream = ahu_outputs_to_ventilation_stream(out)
    assert isinstance(stream, VentilationStream)
    assert stream.name == "mechanical_supply"
    assert stream.category == "supply"
    assert stream.heat_transfer_coefficient_w_k == pytest.approx(
        _RHO_CP_J_M3_K * out.actual_supply_flow_m3_h / 3600.0
    )
    assert stream.source_temperature_c == pytest.approx(out.actual_supply_temperature_c)
    named = ahu_outputs_to_ventilation_stream(out, name="ahu_zone_1")
    assert named.name == "ahu_zone_1"


def test_adapter_zero_flow():
    """AHU off: H_k=0 so stream contributes nothing; source_temperature_c stays finite."""
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.0))
    stream = ahu_outputs_to_ventilation_stream(out)
    assert stream.heat_transfer_coefficient_w_k == 0.0
    assert math.isfinite(stream.source_temperature_c)


def test_adapter_zone_heat_flow_formula():
    """VentilationBoundary source_term_w equals H_k * actual_supply_temperature_c.

    Exercises the full coupling path: AHU step → stream → VentilationBoundary,
    then verifies both source_term_w and sensible_heat_flow_w close correctly.
    """
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    stream = ahu_outputs_to_ventilation_stream(out)
    boundary = VentilationBoundary((stream,))
    t_zone = 21.0

    # S_ve must equal H_k * T_actual_supply (not outdoor or requested temperature)
    expected_s_ve = stream.heat_transfer_coefficient_w_k * out.actual_supply_temperature_c
    assert boundary.source_term_w == pytest.approx(expected_s_ve)

    # Q_ve = H_ve * T_zone - S_ve; supply at 18 °C < zone 21 °C → positive Q_ve
    expected_q_ve = stream.heat_transfer_coefficient_w_k * t_zone - expected_s_ve
    assert boundary.sensible_heat_flow_w(t_zone) == pytest.approx(expected_q_ve)


# ---------------------------------------------------------------------------
# Boundary integration tests (AHU → VentilationStream → VentilationBoundary)
# ---------------------------------------------------------------------------

def _infiltration_stream(t_outdoor_c: float, h_w_k: float = 50.0) -> VentilationStream:
    """Outdoor-air infiltration stream at fixed conductance."""
    return VentilationStream(
        name="infiltration",
        heat_transfer_coefficient_w_k=h_w_k,
        source_temperature_c=t_outdoor_c,
        category="outdoor_air",
    )


def test_integration_boundary_h_ve_is_sum_of_streams():
    """VentilationBoundary H_ve equals infiltration plus AHU stream conductances."""
    t_outdoor = -5.0
    inf = _infiltration_stream(t_outdoor, h_w_k=50.0)
    out = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=t_outdoor))
    ahu = ahu_outputs_to_ventilation_stream(out)
    bnd = VentilationBoundary([inf, ahu])

    assert bnd.heat_transfer_coefficient_w_k == pytest.approx(
        inf.heat_transfer_coefficient_w_k + ahu.heat_transfer_coefficient_w_k
    )


def test_integration_actual_supply_temperature_enters_s_ve():
    """S_ve includes H_ahu * actual_supply_temperature_c, not the outdoor or requested value."""
    t_outdoor = -5.0
    t_zone = 21.0
    inf = _infiltration_stream(t_outdoor, h_w_k=50.0)
    out = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=t_outdoor))
    ahu = ahu_outputs_to_ventilation_stream(out)
    bnd = VentilationBoundary([inf, ahu])

    expected_s_ve = (
        inf.heat_transfer_coefficient_w_k * t_outdoor
        + ahu.heat_transfer_coefficient_w_k * out.actual_supply_temperature_c
    )
    assert bnd.source_term_w == pytest.approx(expected_s_ve)

    # Supply temperature is warmer than outdoor, so S_ve > H_ve * T_outdoor.
    s_ve_if_outdoor = bnd.heat_transfer_coefficient_w_k * t_outdoor
    assert bnd.source_term_w > s_ve_if_outdoor


def test_integration_infiltration_active_when_ahu_off():
    """With AHU off, infiltration stream still contributes its full conductance."""
    t_outdoor = -5.0
    inf = _infiltration_stream(t_outdoor, h_w_k=50.0)
    out = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=t_outdoor,
                                                    operation_fraction=0.0))
    ahu_off = ahu_outputs_to_ventilation_stream(out)
    bnd = VentilationBoundary([inf, ahu_off])

    assert ahu_off.heat_transfer_coefficient_w_k == 0.0
    assert bnd.heat_transfer_coefficient_w_k == pytest.approx(
        inf.heat_transfer_coefficient_w_k
    )
    assert bnd.source_term_w == pytest.approx(
        inf.heat_transfer_coefficient_w_k * t_outdoor
    )


def test_integration_ahu_off_does_not_affect_zone_heat_flow():
    """When AHU is off, sensible_heat_flow_w equals infiltration-only heat flow."""
    t_outdoor = -5.0
    t_zone = 21.0
    inf = _infiltration_stream(t_outdoor, h_w_k=50.0)

    out_on = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=t_outdoor))
    out_off = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=t_outdoor,
                                                        operation_fraction=0.0))
    bnd_on = VentilationBoundary([inf, ahu_outputs_to_ventilation_stream(out_on)])
    bnd_off = VentilationBoundary([inf, ahu_outputs_to_ventilation_stream(out_off)])

    # AHU off → boundary = infiltration only
    assert bnd_off.sensible_heat_flow_w(t_zone) == pytest.approx(
        inf.heat_transfer_coefficient_w_k * t_zone - inf.heat_transfer_coefficient_w_k * t_outdoor
    )
    # AHU on with supply at 18 °C < zone 21 °C: supply air removes additional
    # heat from the zone (positive Q_ve increases), so total Q_ve is higher.
    assert bnd_on.sensible_heat_flow_w(t_zone) > bnd_off.sensible_heat_flow_w(t_zone)


def test_integration_previous_step_extract_temperature_pattern():
    """Verify one-step-lag coupling: T_zone_prev is passed as extract_temperature_c.

    The extract temperature only affects HR heat recovery, not zone balance algebra.
    A warmer previous-step zone temperature increases HR output and reduces coil load.
    """
    t_outdoor = -5.0
    t_zone_prev_cold = 18.0
    t_zone_prev_warm = 24.0

    out_cold = calculate_sensible_ahu_step(
        _cfg(), _inp(outdoor_temperature_c=t_outdoor,
                     extract_temperature_c=t_zone_prev_cold)
    )
    out_warm = calculate_sensible_ahu_step(
        _cfg(), _inp(outdoor_temperature_c=t_outdoor,
                     extract_temperature_c=t_zone_prev_warm)
    )

    # Warmer previous-step zone → more HR heat → less coil required
    assert out_warm.heat_recovery_power_w > out_cold.heat_recovery_power_w
    assert out_warm.required_heating_coil_power_w < out_cold.required_heating_coil_power_w


def test_integration_q_ve_sign_convention_positive_is_heat_leaving_zone():
    """Q_ve = H_ve*T_zone - S_ve; positive means heat leaving the zone.

    When supply is colder than zone, net ventilation removes heat from zone
    (positive Q_ve).  When supply is warmer than zone, net flow adds heat
    (negative Q_ve = heat into zone).
    """
    t_outdoor = -5.0
    t_zone = 21.0

    # Supply at 18 °C < zone 21 °C → ventilation removes heat → positive Q_ve
    out = calculate_sensible_ahu_step(
        _cfg(), _inp(outdoor_temperature_c=t_outdoor)
    )
    ahu = ahu_outputs_to_ventilation_stream(out)
    bnd = VentilationBoundary([ahu])
    assert out.actual_supply_temperature_c < t_zone
    assert bnd.sensible_heat_flow_w(t_zone) > 0.0

    # Supply above zone → ventilation adds heat → negative Q_ve
    out_hot = calculate_sensible_ahu_step(
        _cfg(supply_temperature_control=SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.FIXED, setpoint_c=30.0
        )),
        _inp(outdoor_temperature_c=t_outdoor)
    )
    ahu_hot = ahu_outputs_to_ventilation_stream(out_hot)
    bnd_hot = VentilationBoundary([ahu_hot])
    assert out_hot.actual_supply_temperature_c > t_zone
    assert bnd_hot.sensible_heat_flow_w(t_zone) < 0.0


# ---------------------------------------------------------------------------
# sensible_ahu_config_from_dict — component-dict factory
# ---------------------------------------------------------------------------

def _minimal_ahu_dict(**overrides):
    d = {
        "sensible_heat_recovery_efficiency": 0.784,
        "supply_temperature_setpoint_c": 18.0,
    }
    d.update(overrides)
    return d


def test_from_dict_required_fields():
    cfg = sensible_ahu_config_from_dict(_minimal_ahu_dict())
    assert cfg.sensible_heat_recovery_efficiency == pytest.approx(0.784)
    assert cfg.supply_temperature_control.setpoint_c == pytest.approx(18.0)


def test_from_dict_defaults():
    cfg = sensible_ahu_config_from_dict(_minimal_ahu_dict())
    assert cfg.heat_recovery_control is HeatRecoveryControl.MODULATING_BYPASS
    assert cfg.frost_control is FrostControlMode.EXHAUST_LIMIT
    assert cfg.frost_exhaust_limit_c == pytest.approx(-5.0)
    assert cfg.heating_coil_max_power_w is None
    assert cfg.cooling_coil_enabled is False
    assert cfg.supply_fan_specific_power_w_per_m3_s == pytest.approx(0.0)
    assert cfg.extract_fan_specific_power_w_per_m3_s == pytest.approx(0.0)
    assert cfg.supply_fan_heat_fraction_to_air == pytest.approx(0.90)
    assert cfg.extract_fan_heat_fraction_to_air == pytest.approx(0.90)
    assert cfg.fan_performance_model is FanPerformanceModel.EN16798_5_1
    assert cfg.supply_fan_nominal_efficiency == pytest.approx(0.72)
    assert cfg.extract_fan_nominal_efficiency == pytest.approx(0.78)


def test_from_dict_explicit_optional_fields():
    cfg = sensible_ahu_config_from_dict(_minimal_ahu_dict(
        heat_recovery_control="always_on",
        frost_control="none",
        frost_exhaust_limit_c=-3.0,
        heating_coil_max_power_w=5000.0,
        supply_fan_specific_power_w_per_m3_s=200.0,
        extract_fan_specific_power_w_per_m3_s=150.0,
        supply_fan_heat_fraction_to_air=0.9,
        extract_fan_heat_fraction_to_air=0.8,
        fan_performance_model="constant_sfp",
        supply_fan_design_pressure_pa=450.0,
        extract_fan_design_pressure_pa=350.0,
        supply_fan_nominal_efficiency=0.70,
        extract_fan_nominal_efficiency=0.75,
        supply_fan_reference_flow_m3_h=1000.0,
        extract_fan_reference_flow_m3_h=900.0,
        supply_fan_reference_efficiency=0.68,
        extract_fan_reference_efficiency=0.73,
    ))
    assert cfg.heat_recovery_control is HeatRecoveryControl.ALWAYS_ON
    assert cfg.frost_control is FrostControlMode.NONE
    assert cfg.frost_exhaust_limit_c == pytest.approx(-3.0)
    assert cfg.heating_coil_max_power_w == pytest.approx(5000.0)
    assert cfg.supply_fan_specific_power_w_per_m3_s == pytest.approx(200.0)
    assert cfg.extract_fan_specific_power_w_per_m3_s == pytest.approx(150.0)
    assert cfg.supply_fan_heat_fraction_to_air == pytest.approx(0.9)
    assert cfg.extract_fan_heat_fraction_to_air == pytest.approx(0.8)
    assert cfg.fan_performance_model is FanPerformanceModel.CONSTANT_SFP
    assert cfg.supply_fan_design_pressure_pa == pytest.approx(450.0)
    assert cfg.extract_fan_design_pressure_pa == pytest.approx(350.0)
    assert cfg.supply_fan_nominal_efficiency == pytest.approx(0.70)
    assert cfg.extract_fan_nominal_efficiency == pytest.approx(0.75)
    assert cfg.supply_fan_reference_flow_m3_h == pytest.approx(1000.0)
    assert cfg.extract_fan_reference_flow_m3_h == pytest.approx(900.0)
    assert cfg.supply_fan_reference_efficiency == pytest.approx(0.68)
    assert cfg.extract_fan_reference_efficiency == pytest.approx(0.73)


def test_from_dict_missing_setpoint_raises():
    with pytest.raises(KeyError, match="supply_temperature_setpoint_c"):
        sensible_ahu_config_from_dict({"sensible_heat_recovery_efficiency": 0.8})


def test_from_dict_missing_efficiency_raises():
    with pytest.raises(KeyError):
        sensible_ahu_config_from_dict({"supply_temperature_setpoint_c": 18.0})


def test_from_dict_produces_runnable_config():
    cfg = sensible_ahu_config_from_dict(_minimal_ahu_dict())
    inp = AHUStepInputs(
        outdoor_temperature_c=-5.0,
        extract_temperature_c=21.0,
        required_supply_flow_m3_h=3600.0,
        required_extract_flow_m3_h=3600.0,
        operation_fraction=1.0,
    )
    out = calculate_sensible_ahu_step(cfg, inp)
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-9)


def test_from_dict_outdoor_compensated_mode():
    """outdoor_compensated mode with two-point curve."""
    # Points: T_oda=-20 → T_sup=19, T_oda=+15 → T_sup=17
    cfg = sensible_ahu_config_from_dict({
        "sensible_heat_recovery_efficiency": 0.784,
        "supply_temperature_mode": "outdoor_compensated",
        "supply_temperature_points": [[-20, 19], [15, 17]],
    })
    assert cfg.supply_temperature_control.mode is SupplyTemperatureControlMode.OUTDOOR_COMPENSATED
    assert len(cfg.supply_temperature_control.points) == 2


def test_from_dict_outdoor_compensated_delivers_interpolated_setpoint():
    """At T_oda=0, interpolated setpoint = 17 + (15-0)/(15-(-20)) * (19-17) = 17 + 15/35*2 ≈ 17.857."""
    cfg = sensible_ahu_config_from_dict({
        "sensible_heat_recovery_efficiency": 0.784,
        "supply_temperature_mode": "outdoor_compensated",
        "supply_temperature_points": [[-20, 19], [15, 17]],
    })
    inp = AHUStepInputs(
        outdoor_temperature_c=0.0,
        extract_temperature_c=21.0,
        required_supply_flow_m3_h=3600.0,
        required_extract_flow_m3_h=3600.0,
        operation_fraction=1.0,
    )
    out = calculate_sensible_ahu_step(cfg, inp)
    expected_setpoint = 17.0 + (15.0 - 0.0) / (15.0 - (-20.0)) * (19.0 - 17.0)
    assert out.requested_supply_temperature_c == pytest.approx(expected_setpoint, rel=1e-6)
    # Actual supply reaches the setpoint (HR+coil can cover T_oda=0 → setpoint≈17.86)
    assert out.actual_supply_temperature_c == pytest.approx(expected_setpoint, abs=0.1)


def test_from_dict_outdoor_compensated_missing_points_raises():
    with pytest.raises(KeyError, match="supply_temperature_points"):
        sensible_ahu_config_from_dict({
            "sensible_heat_recovery_efficiency": 0.784,
            "supply_temperature_mode": "outdoor_compensated",
        })


def test_from_dict_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unsupported supply_temperature_mode"):
        sensible_ahu_config_from_dict({
            "sensible_heat_recovery_efficiency": 0.784,
            "supply_temperature_mode": "magic",
            "supply_temperature_setpoint_c": 18.0,
        })
