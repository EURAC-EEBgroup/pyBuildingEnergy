"""Sensible air-handling calculations based on EN 16798-5-1.

The aim is to evaluate and demonstrate the calculation proposed by the standard.
This work does not replace the standard; it should be used alongside the EPB standard.

Acknowledgments: The work was developed from the standard and the spreadsheet created by EPB Center.

The module is intentionally independent of the ISO 52016 zone solver. Supply
temperature control is resolved before the EN 16798-5-1 thermodynamic
calculation so that fixed, scheduled, outdoor-compensated, and
extract-compensated supervisory controls share one explicit and testable
interface.

Standards note on extract-compensated control
----------------------------------------------
The EXTRACT_COMPENSATED controller computes a requested supply-temperature
setpoint as a piecewise-linear function of zone (extract) air temperature.
This is a supervisory strategy supported by the architecture of EN 16798-5-1
but is NOT enumerated as one of the core controller modes in EN 16798-5-1
Table B.2 (which covers fixed, outdoor-compensated, and load-compensated
strategies). It is implemented here as a generic piecewise-linear controller
that produces the required supply-air temperature for the AHU calculation.

Physical constants
------------------
Standard air at sea level, approximately 20 °C:
  rho = 1.204 kg/m³,  cp = 1006 J/(kg·K)  →  rho*cp = 1211.224 J/(m³·K)

Fan heat placement
------------------
The configured extract-fan heat fraction is added to extract air before the
heat exchanger. Set ``extract_fan_heat_fraction_to_air=0`` for an extract fan
located downstream of heat recovery or whose heat is otherwise not returned to
the HR inlet. The configured supply-fan heat fraction is added after the heat
exchanger and heating coil, just before delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .ventilation import VentilationStream


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_RHO_AIR_KG_M3: float = 1.204
_CP_AIR_J_KG_K: float = 1006.0
_RHO_CP_J_M3_K: float = _RHO_AIR_KG_M3 * _CP_AIR_J_KG_K  # 1211.224 J/(m³·K)


# ---------------------------------------------------------------------------
# Supply-temperature controller
# ---------------------------------------------------------------------------

class SupplyTemperatureControlMode(str, Enum):
    """Available supply-air temperature setpoint strategies."""

    FIXED = "fixed"
    SCHEDULED = "scheduled"
    OUTDOOR_COMPENSATED = "outdoor_compensated"
    EXTRACT_COMPENSATED = "extract_compensated"


@dataclass(frozen=True)
class CompensationPoint:
    """One point on a supply-temperature compensation curve."""

    reference_temperature_c: float
    supply_temperature_c: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_temperature_c):
            raise ValueError("reference_temperature_c must be finite")
        if not math.isfinite(self.supply_temperature_c):
            raise ValueError("supply_temperature_c must be finite")


@dataclass(frozen=True)
class SupplyTemperatureControl:
    """Configuration for a requested AHU supply-temperature setpoint.

    Compensation curves are sorted by reference temperature and use
    piecewise-linear interpolation. Values outside the curve are clamped to
    the nearest endpoint.
    """

    mode: SupplyTemperatureControlMode
    setpoint_c: float | None = None
    points: tuple[CompensationPoint, ...] = ()
    minimum_c: float | None = None
    maximum_c: float | None = None
    extrapolation: str = "clamp"

    def __post_init__(self) -> None:
        try:
            mode = SupplyTemperatureControlMode(self.mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported supply-temperature control mode: {self.mode!r}"
            ) from exc
        object.__setattr__(self, "mode", mode)

        points = tuple(self.points)
        if any(not isinstance(point, CompensationPoint) for point in points):
            raise TypeError("points must contain CompensationPoint instances")
        points = tuple(sorted(points, key=lambda point: point.reference_temperature_c))
        object.__setattr__(self, "points", points)

        references = [point.reference_temperature_c for point in points]
        if len(references) != len(set(references)):
            raise ValueError("Compensation reference temperatures must be unique")

        for name, value in (
            ("setpoint_c", self.setpoint_c),
            ("minimum_c", self.minimum_c),
            ("maximum_c", self.maximum_c),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")

        if (
            self.minimum_c is not None
            and self.maximum_c is not None
            and self.minimum_c > self.maximum_c
        ):
            raise ValueError("minimum_c must be <= maximum_c")
        if self.extrapolation != "clamp":
            raise ValueError("Only endpoint-clamped extrapolation is supported")

        if mode is SupplyTemperatureControlMode.FIXED:
            if self.setpoint_c is None:
                raise ValueError("Fixed control requires setpoint_c")
            if points:
                raise ValueError("Fixed control does not accept compensation points")
        elif mode is SupplyTemperatureControlMode.SCHEDULED:
            if self.setpoint_c is not None:
                raise ValueError("Scheduled control receives its setpoint at runtime")
            if points:
                raise ValueError("Scheduled control does not accept compensation points")
        else:
            if self.setpoint_c is not None:
                raise ValueError("Compensated control does not accept setpoint_c")
            if len(points) < 2:
                raise ValueError("Compensated control requires at least two points")


def resolve_supply_temperature_setpoint(
    control: SupplyTemperatureControl,
    outdoor_temperature_c: float,
    extract_temperature_c: float,
    scheduled_setpoint_c: float | None = None,
) -> float:
    """Return the requested AHU supply-air temperature in degrees Celsius.

    The requested setpoint is distinct from the actual delivered supply
    temperature. Heat recovery, coil capacity, frost control, and disabled
    cooling can make the requested condition unattainable.
    """

    if not isinstance(control, SupplyTemperatureControl):
        raise TypeError("control must be a SupplyTemperatureControl")

    mode = control.mode
    if mode is SupplyTemperatureControlMode.FIXED:
        requested_c = control.setpoint_c
    elif mode is SupplyTemperatureControlMode.SCHEDULED:
        if scheduled_setpoint_c is None:
            raise ValueError("Scheduled control requires scheduled_setpoint_c")
        if not math.isfinite(scheduled_setpoint_c):
            raise ValueError("scheduled_setpoint_c must be finite")
        requested_c = scheduled_setpoint_c
    elif mode is SupplyTemperatureControlMode.OUTDOOR_COMPENSATED:
        requested_c = _interpolate_compensation_curve(
            control.points,
            outdoor_temperature_c,
            "outdoor_temperature_c",
        )
    else:
        requested_c = _interpolate_compensation_curve(
            control.points,
            extract_temperature_c,
            "extract_temperature_c",
        )

    if control.minimum_c is not None:
        requested_c = max(control.minimum_c, requested_c)
    if control.maximum_c is not None:
        requested_c = min(control.maximum_c, requested_c)
    return requested_c


def _interpolate_compensation_curve(
    points: tuple[CompensationPoint, ...],
    reference_temperature_c: float,
    input_name: str,
) -> float:
    if not math.isfinite(reference_temperature_c):
        raise ValueError(f"{input_name} must be finite")

    if reference_temperature_c <= points[0].reference_temperature_c:
        return points[0].supply_temperature_c
    if reference_temperature_c >= points[-1].reference_temperature_c:
        return points[-1].supply_temperature_c

    for lower, upper in zip(points, points[1:]):
        if reference_temperature_c <= upper.reference_temperature_c:
            fraction = (
                (reference_temperature_c - lower.reference_temperature_c)
                / (upper.reference_temperature_c - lower.reference_temperature_c)
            )
            return lower.supply_temperature_c + fraction * (
                upper.supply_temperature_c - lower.supply_temperature_c
            )

    raise RuntimeError("Compensation interpolation failed")


# ---------------------------------------------------------------------------
# AHU thermodynamic model
# ---------------------------------------------------------------------------

class HeatRecoveryControl(str, Enum):
    """Available heat-recovery control strategies."""

    ALWAYS_ON = "always_on"
    MODULATING_BYPASS = "modulating_bypass"


class FrostControlMode(str, Enum):
    """Available frost-protection strategies.

    NONE         — no frost protection; HR efficiency is not reduced.
    EXHAUST_LIMIT — directly reduces the effective HR efficiency so the
                   exhaust leaving the HR core stays at or above
                   ``frost_exhaust_limit_c``.  The continuous exhaust-
                   temperature-limit formulation follows the direct
                   frost-protection approach in EN 16798-5-1 formula (46):
                   heat-recovery efficiency is reduced when frost risk is
                   present.  It does not implement the indirect treatment
                   in formulas (47b) through (49), which calculates
                   frost-protected outdoor-air and limited heat-recovery
                   supply temperatures.  The physical defrost mechanism
                   (bypass, preheat, or recirculation) is a separate
                   configuration dimension not modelled here.
    """

    NONE = "none"
    EXHAUST_LIMIT = "exhaust_limit"


class FanPerformanceModel(str, Enum):
    """Fan power and temperature-rise calculation method.

    EN16798_5_1 applies the EN 16798-5-1 single-zone no-control fan
    relations: pressure varies with the square of the flow ratio and fan
    efficiency follows the product-data square-root correction.

    CONSTANT_SFP applies the constant-SFP relation P = q * SFP at every flow.
    """

    EN16798_5_1 = "en16798_5_1"
    CONSTANT_SFP = "constant_sfp"


@dataclass(frozen=True)
class SensibleAHUConfig:
    """Configuration for the sensible EN 16798-5-1 AHU timestep model.

    Covers balanced single-zone airflow with sensible heat recovery,
    EXHAUST_LIMIT frost protection, modulating-bypass economizer, and
    heating and cooling coils. Cooling is sensible only (no
    dehumidification); set ``cooling_coil_enabled=True`` to deliver
    mechanical cooling, optionally capped by ``cooling_coil_max_power_w``.
    When the cooling coil is disabled, any cooling load is still reported
    in ``required_cooling_coil_power_w`` as an unmet shortfall.

    Fan model:
      ``supply_fan_specific_power_w_per_m3_s`` and
      ``extract_fan_specific_power_w_per_m3_s`` are the **nominal** SFP values
      at design flow. The default ``en16798_5_1`` model derives design pressure
      from nominal SFP and nominal fan efficiency unless an explicit design
      pressure is supplied. At part load it applies the EN 16798-5-1
      single-zone no-control pressure relation and product-data efficiency
      correction:

      ``delta_p = delta_p_design * (q / q_design)**2``
      ``eta = eta_ref * sqrt(q / q_ref)``

      ``constant_sfp`` applies ``P = q * SFP`` at every flow.
    """

    sensible_heat_recovery_efficiency: float
    supply_temperature_control: SupplyTemperatureControl
    heat_recovery_control: HeatRecoveryControl | str
    frost_control: FrostControlMode | str
    heating_coil_max_power_w: float | None
    cooling_coil_enabled: bool
    supply_fan_specific_power_w_per_m3_s: float
    extract_fan_specific_power_w_per_m3_s: float
    supply_fan_heat_fraction_to_air: float
    extract_fan_heat_fraction_to_air: float
    # Cooling coil capacity [W]; None = unlimited. Ignored when
    # cooling_coil_enabled is False. Mirrors heating_coil_max_power_w.
    cooling_coil_max_power_w: float | None = None
    frost_exhaust_limit_c: float = -5.0
    fan_performance_model: FanPerformanceModel | str = FanPerformanceModel.EN16798_5_1
    supply_fan_design_pressure_pa: float | None = None
    extract_fan_design_pressure_pa: float | None = None
    supply_fan_nominal_efficiency: float = 0.72
    extract_fan_nominal_efficiency: float = 0.78
    supply_fan_reference_flow_m3_h: float | None = None
    extract_fan_reference_flow_m3_h: float | None = None
    supply_fan_reference_efficiency: float | None = None
    extract_fan_reference_efficiency: float | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self, "heat_recovery_control",
                HeatRecoveryControl(self.heat_recovery_control),
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported heat_recovery_control: {self.heat_recovery_control!r}"
            ) from exc

        try:
            object.__setattr__(
                self, "frost_control",
                FrostControlMode(self.frost_control),
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported frost_control: {self.frost_control!r}"
            ) from exc

        try:
            object.__setattr__(
                self,
                "fan_performance_model",
                FanPerformanceModel(self.fan_performance_model),
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported fan_performance_model: {self.fan_performance_model!r}"
            ) from exc

        eta = self.sensible_heat_recovery_efficiency
        if not math.isfinite(eta) or eta < 0.0 or eta > 1.0:
            raise ValueError(
                "sensible_heat_recovery_efficiency must be finite and in [0, 1]"
            )

        if not isinstance(self.supply_temperature_control, SupplyTemperatureControl):
            raise TypeError(
                "supply_temperature_control must be a SupplyTemperatureControl"
            )

        if self.heating_coil_max_power_w is not None:
            if (
                not math.isfinite(self.heating_coil_max_power_w)
                or self.heating_coil_max_power_w < 0.0
            ):
                raise ValueError(
                    "heating_coil_max_power_w must be finite and non-negative when provided"
                )

        if self.cooling_coil_max_power_w is not None:
            if (
                not math.isfinite(self.cooling_coil_max_power_w)
                or self.cooling_coil_max_power_w < 0.0
            ):
                raise ValueError(
                    "cooling_coil_max_power_w must be finite and non-negative when provided"
                )

        for name, value in (
            ("supply_fan_specific_power_w_per_m3_s",
             self.supply_fan_specific_power_w_per_m3_s),
            ("extract_fan_specific_power_w_per_m3_s",
             self.extract_fan_specific_power_w_per_m3_s),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        for name, value in (
            ("supply_fan_design_pressure_pa", self.supply_fan_design_pressure_pa),
            ("extract_fan_design_pressure_pa", self.extract_fan_design_pressure_pa),
            ("supply_fan_reference_flow_m3_h", self.supply_fan_reference_flow_m3_h),
            ("extract_fan_reference_flow_m3_h", self.extract_fan_reference_flow_m3_h),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when provided")

        for name, value in (
            ("supply_fan_nominal_efficiency", self.supply_fan_nominal_efficiency),
            ("extract_fan_nominal_efficiency", self.extract_fan_nominal_efficiency),
            ("supply_fan_reference_efficiency", self.supply_fan_reference_efficiency),
            ("extract_fan_reference_efficiency", self.extract_fan_reference_efficiency),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0.0 or value > 1.0
            ):
                raise ValueError(f"{name} must be in (0, 1] when provided")

        for name, value in (
            ("supply_fan_heat_fraction_to_air", self.supply_fan_heat_fraction_to_air),
            ("extract_fan_heat_fraction_to_air", self.extract_fan_heat_fraction_to_air),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        if not math.isfinite(self.frost_exhaust_limit_c):
            raise ValueError("frost_exhaust_limit_c must be finite")


@dataclass(frozen=True)
class AHUStepInputs:
    """Time-varying inputs for one AHU timestep.

    ``operation_fraction`` is the fraction of the timestep for which the AHU
    operates. ``flow_fraction`` is the continuous operating-flow ratio relative
    to the required/design flow. Keeping these separate distinguishes ON/OFF
    duty averaging from true fan-speed reduction. The ISO 52016 component
    adapter maps schedules only to ``flow_fraction`` and fixes
    ``operation_fraction`` at 1.0. Direct users of this standalone API may
    supply a duty fraction explicitly.
    """

    outdoor_temperature_c: float
    extract_temperature_c: float
    required_supply_flow_m3_h: float
    required_extract_flow_m3_h: float
    operation_fraction: float = 1.0
    scheduled_setpoint_c: float | None = None
    flow_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("outdoor_temperature_c", self.outdoor_temperature_c),
            ("extract_temperature_c", self.extract_temperature_c),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        for name, value in (
            ("required_supply_flow_m3_h", self.required_supply_flow_m3_h),
            ("required_extract_flow_m3_h", self.required_extract_flow_m3_h),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        q_sup = self.required_supply_flow_m3_h
        q_ext = self.required_extract_flow_m3_h
        if q_sup != q_ext:
            larger = max(q_sup, q_ext)
            if larger == 0.0 or abs(q_sup - q_ext) / larger > 1e-6:
                raise ValueError(
                    "required_supply_flow_m3_h and required_extract_flow_m3_h must be "
                    "equal: unequal supply and extract flow is outside the balanced-flow "
                    "scope"
                )

        if (
            not math.isfinite(self.operation_fraction)
            or not (0.0 <= self.operation_fraction <= 1.0)
        ):
            raise ValueError("operation_fraction must be finite and in [0, 1]")

        if (
            not math.isfinite(self.flow_fraction)
            or not (0.0 <= self.flow_fraction <= 1.0)
        ):
            raise ValueError("flow_fraction must be finite and in [0, 1]")

        if (
            self.scheduled_setpoint_c is not None
            and not math.isfinite(self.scheduled_setpoint_c)
        ):
            raise ValueError("scheduled_setpoint_c must be finite when provided")


@dataclass(frozen=True)
class AHUStepOutputs:
    """Computed outputs for one AHU timestep.

    Power quantities are in watts, averaged over the timestep. The caller is
    responsible for converting power to energy using its timestep duration.

    requested_supply_temperature_c: setpoint from the supply-temperature controller,
                                    or None when the AHU is off (flow = 0).
    actual_supply_temperature_c:    delivered supply temperature including fan heat.
    heat_recovery_power_w:          rho*cp*q*(T_hr_outlet − T_outdoor).  Equals the
                                    EN 16798-5-1 formula (77) heat-recovery term only
                                    when all of the following hold:
                                    balanced flow, no recirculation, no ODA preheating, and
                                    frost correction inactive.  During active EXHAUST_LIMIT
                                    frost control, the total supply heating relative to
                                    outdoor air may differ from preheat-based defrost
                                    configurations because the effective HR outlet temperature
                                    is determined by a different algorithm.
                                    Positive in heating mode; negative in economizer mode.
    required_heating_coil_power_w:  heating coil power needed to reach setpoint.
    actual_heating_coil_power_w:    heating coil power actually delivered (≤ required).
    required_cooling_coil_power_w:  sensible cooling coil power needed to pull the
                                    post-HR/bypass supply air down to the setpoint.
                                    Non-zero when the requested supply temperature is
                                    below the minimum achievable by bypass alone.
                                    Placed before the heating coil in the air path,
                                    per EN 16798-5-1.
    actual_cooling_coil_power_w:    cooling coil power actually delivered (≤ required).
                                    Zero when cooling_coil_enabled is False — the
                                    required value then reports the unmet shortfall.
                                    Capacity-limited by cooling_coil_max_power_w.
    fan_electric_power_w:           total fan electrical power (separate from thermal).
    bypass_fraction:                effective fraction of the HR bypassed, derived from
                                    actual vs nominal recovery [0, 1].  Reflects efficiency
                                    reduction rather than a physical bypass damper position.
    frost_protection_required:      True when exhaust temperature would fall below
                                    frost_exhaust_limit_c at nominal HR efficiency.
    """

    requested_supply_temperature_c: float | None
    actual_supply_temperature_c: float
    actual_supply_flow_m3_h: float
    actual_extract_flow_m3_h: float
    required_heating_coil_power_w: float
    actual_heating_coil_power_w: float
    required_cooling_coil_power_w: float
    actual_cooling_coil_power_w: float
    fan_electric_power_w: float
    heat_recovery_power_w: float
    bypass_fraction: float
    frost_protection_required: bool


def _fan_power_and_temperature_rise(
    *,
    model: FanPerformanceModel,
    operating_flow_m3_h: float,
    design_flow_m3_h: float,
    operation_fraction: float,
    nominal_specific_power_w_per_m3_s: float,
    design_pressure_pa: float | None,
    nominal_efficiency: float,
    reference_flow_m3_h: float | None,
    reference_efficiency: float | None,
    heat_fraction_to_air: float,
) -> tuple[float, float]:
    """Return timestep-average fan electric power [W] and operating delta-T [K].

    The EN 16798-5-1 mode derives the design pressure from nominal specific
    fan power and efficiency unless given explicitly, then applies
    ``delta_p = delta_p_design * (q / q_design)**2`` and
    ``eta = eta_ref * sqrt(q / q_ref)`` at part load. The temperature rise is
    evaluated while the fan is operating; electric power is averaged over the
    timestep using ``operation_fraction``.
    """
    if (
        operating_flow_m3_h <= 0.0
        or design_flow_m3_h <= 0.0
        or operation_fraction <= 0.0
    ):
        return 0.0, 0.0

    q_operating_m3_s = operating_flow_m3_h / 3600.0
    if model is FanPerformanceModel.CONSTANT_SFP:
        power_operating_w = (
            q_operating_m3_s * nominal_specific_power_w_per_m3_s
        )
    else:
        pressure_design_pa = (
            design_pressure_pa
            if design_pressure_pa is not None
            else nominal_specific_power_w_per_m3_s * nominal_efficiency
        )
        if pressure_design_pa <= 0.0:
            return 0.0, 0.0

        q_ratio = operating_flow_m3_h / design_flow_m3_h
        pressure_pa = pressure_design_pa * q_ratio ** 2

        q_ref_m3_h = (
            reference_flow_m3_h
            if reference_flow_m3_h is not None
            else design_flow_m3_h
        )
        eta_ref = (
            reference_efficiency
            if reference_efficiency is not None
            else nominal_efficiency
        )
        efficiency = (
            nominal_efficiency
            * (eta_ref / nominal_efficiency)
            * math.sqrt(operating_flow_m3_h / q_ref_m3_h)
        )
        power_operating_w = q_operating_m3_s * pressure_pa / efficiency

    temperature_rise_k = (
        power_operating_w * heat_fraction_to_air
        / (_RHO_CP_J_M3_K * q_operating_m3_s)
    )
    return power_operating_w * operation_fraction, temperature_rise_k


def calculate_sensible_ahu_step(
    config: SensibleAHUConfig,
    inputs: AHUStepInputs,
) -> AHUStepOutputs:
    """Calculate one timestep of the sensible EN 16798-5-1 AHU model.

    Air-path order: extract fan heat -> heat recovery (frost-limited
    efficiency, modulating bypass) -> cooling coil -> heating coil ->
    supply fan heat.  The coil target is pre-compensated for downstream
    supply-fan heat so the delivered temperature meets the requested
    setpoint when capacity allows.

    Power outputs are timestep averages (scaled by operation_fraction);
    temperature outputs describe the operating airstream.  When
    operation_fraction, flow_fraction, or required supply flow is zero, all
    flow and energy outputs are zero and the actual supply temperature
    equals outdoor.
    """
    if not isinstance(config, SensibleAHUConfig):
        raise TypeError("config must be a SensibleAHUConfig")
    if not isinstance(inputs, AHUStepInputs):
        raise TypeError("inputs must be an AHUStepInputs")

    # Operating flow vs timestep-average flow: keeps reduced fan speed
    # (flow_fraction) separate from ON/OFF duty averaging (operation_fraction).
    op = inputs.operation_fraction
    flow_fraction = inputs.flow_fraction
    operating_supply_m3_h = flow_fraction * inputs.required_supply_flow_m3_h
    operating_extract_m3_h = flow_fraction * inputs.required_extract_flow_m3_h
    actual_supply_m3_h = op * operating_supply_m3_h
    actual_extract_m3_h = op * operating_extract_m3_h
    q_sup = operating_supply_m3_h / 3600.0  # m³/s while operating
    q_eta = operating_extract_m3_h / 3600.0  # m³/s while operating

    if op <= 0.0 or q_sup <= 0.0:
        return AHUStepOutputs(
            requested_supply_temperature_c=None,
            actual_supply_temperature_c=inputs.outdoor_temperature_c,
            actual_supply_flow_m3_h=0.0,
            actual_extract_flow_m3_h=0.0,
            required_heating_coil_power_w=0.0,
            actual_heating_coil_power_w=0.0,
            required_cooling_coil_power_w=0.0,
            actual_cooling_coil_power_w=0.0,
            fan_electric_power_w=0.0,
            heat_recovery_power_w=0.0,
            bypass_fraction=0.0,
            frost_protection_required=False,
        )

    # Requested setpoint (only resolved when the AHU is actually running)
    requested_t_sup = resolve_supply_temperature_setpoint(
        config.supply_temperature_control,
        inputs.outdoor_temperature_c,
        inputs.extract_temperature_c,
        inputs.scheduled_setpoint_c,
    )

    rho_cp_q = _RHO_CP_J_M3_K * q_sup  # W/K for supply stream

    # Fan electricity and temperature rises. Power outputs are timestep
    # averages; temperature rises describe the operating airstream.
    p_sup_fan, dt_sup_fan = _fan_power_and_temperature_rise(
        model=config.fan_performance_model,
        operating_flow_m3_h=operating_supply_m3_h,
        design_flow_m3_h=inputs.required_supply_flow_m3_h,
        operation_fraction=op,
        nominal_specific_power_w_per_m3_s=(
            config.supply_fan_specific_power_w_per_m3_s
        ),
        design_pressure_pa=config.supply_fan_design_pressure_pa,
        nominal_efficiency=config.supply_fan_nominal_efficiency,
        reference_flow_m3_h=config.supply_fan_reference_flow_m3_h,
        reference_efficiency=config.supply_fan_reference_efficiency,
        heat_fraction_to_air=config.supply_fan_heat_fraction_to_air,
    )
    p_eta_fan, dt_eta_fan = _fan_power_and_temperature_rise(
        model=config.fan_performance_model,
        operating_flow_m3_h=operating_extract_m3_h,
        design_flow_m3_h=inputs.required_extract_flow_m3_h,
        operation_fraction=op,
        nominal_specific_power_w_per_m3_s=(
            config.extract_fan_specific_power_w_per_m3_s
        ),
        design_pressure_pa=config.extract_fan_design_pressure_pa,
        nominal_efficiency=config.extract_fan_nominal_efficiency,
        reference_flow_m3_h=config.extract_fan_reference_flow_m3_h,
        reference_efficiency=config.extract_fan_reference_efficiency,
        heat_fraction_to_air=config.extract_fan_heat_fraction_to_air,
    )
    fan_electric_power_w = p_sup_fan + p_eta_fan

    t_outdoor = inputs.outdoor_temperature_c
    t_ext_at_hr = inputs.extract_temperature_c + dt_eta_fan
    eta_nom = config.sensible_heat_recovery_efficiency
    dT = t_ext_at_hr - t_outdoor  # K; positive in heating mode

    # EXHAUST_LIMIT frost protection — EN 16798-5-1 formula (46) direct
    # effectiveness control.  For balanced flow the exhaust leaving the HR is
    # T_EHA = T_extract_at_hr - eta * dT; if it would fall below
    # frost_exhaust_limit_c, reduce eta_eff so T_EHA sits exactly at the limit.
    frost_protection_req = False
    eta_eff = eta_nom
    if (
        config.frost_control is FrostControlMode.EXHAUST_LIMIT
        and eta_nom > 0.0
        and abs(dT) > 1e-9
    ):
        t_eha = t_ext_at_hr - eta_nom * dT  # exhaust at HR outlet
        if t_eha < config.frost_exhaust_limit_c:
            frost_protection_req = True
            eta_frost = (t_ext_at_hr - config.frost_exhaust_limit_c) / dT
            eta_eff = max(0.0, min(eta_nom, eta_frost))

    # HR outlet temperature (before bypass modulation)
    t_after_hr = t_outdoor + eta_eff * dT

    # Internal coil/bypass target: subtract downstream supply fan heat so the
    # delivered temperature equals requested_t_sup after the fan addition below.
    t_target = requested_t_sup - dt_sup_fan

    # Bypass formula: bp = (T_hr - T_target) / (T_hr - T_outdoor), clamped to [0, 1].
    # bp = 1 → full bypass, delivers T_outdoor.  bp = 0 → no bypass, delivers T_hr.
    # Clamping handles setpoints outside the achievable mixing range: when the
    # setpoint is below T_outdoor (heating mode, no cooling), full bypass delivers
    # the closest attainable temperature; when the setpoint is above T_outdoor
    # (economizer mode), full bypass delivers T_outdoor.
    #
    # Design choice: in economizer mode (T_outdoor > T_extract) we modulate the HR
    # continuously to track the setpoint rather than switching to full outdoor bypass.
    # This keeps supply delivery consistent across all conditions without a mode switch.
    if config.heat_recovery_control is HeatRecoveryControl.MODULATING_BYPASS:
        dT_hr_oda = t_after_hr - t_outdoor  # = eta_eff * dT
        if abs(dT_hr_oda) > 1e-9:
            bp = (t_after_hr - t_target) / dT_hr_oda
            bp = max(0.0, min(1.0, bp))
            t_after_hr = t_outdoor * bp + t_after_hr * (1.0 - bp)

    # Total effective bypass fraction: fraction of supply air bypassing the HR
    # core due to both frost protection and setpoint modulation.
    if eta_nom > 0.0 and abs(dT) > 1e-9:
        bypass_fraction = max(
            0.0, min(1.0, 1.0 - (t_after_hr - t_outdoor) / (eta_nom * dT))
        )
    else:
        bypass_fraction = 0.0

    # Heat recovery power: net thermal gain of the supply stream vs outdoor air
    hr_power_operating_w = rho_cp_q * (t_after_hr - t_outdoor)

    # Cooling coil — precedes the heating coil, following the EN 16798-5-1 air
    # path.  In this sensible single-setpoint model the two coils are mutually
    # exclusive within a timestep, so the ordering does not change the result.
    #
    # required_cooling is the sensible load needed to pull the post-HR/bypass air
    # down to t_target.  When the coil is disabled it is reported as the unmet
    # cooling shortfall (actual delivered = 0); when enabled it is met up to the
    # optional capacity limit.  This mirrors the heating coil below.
    required_cooling_operating_w = rho_cp_q * max(0.0, t_after_hr - t_target)
    if not config.cooling_coil_enabled:
        actual_cooling_operating_w = 0.0
    elif config.cooling_coil_max_power_w is None:
        actual_cooling_operating_w = required_cooling_operating_w
    else:
        actual_cooling_operating_w = min(
            required_cooling_operating_w,
            config.cooling_coil_max_power_w,
        )

    t_after_cooling = t_after_hr - actual_cooling_operating_w / rho_cp_q

    # Heating coil targeting t_target so the delivered temperature equals
    # requested_t_sup after supply-fan heat is added below.  It sees the
    # post-cooling temperature; with a single setpoint at most one coil is active.
    required_heating_operating_w = rho_cp_q * max(0.0, t_target - t_after_cooling)
    if config.heating_coil_max_power_w is None:
        actual_heating_operating_w = required_heating_operating_w
    else:
        actual_heating_operating_w = min(
            required_heating_operating_w,
            config.heating_coil_max_power_w,
        )

    t_after_coil = t_after_cooling + actual_heating_operating_w / rho_cp_q

    # Supply fan heat (after HR and coils, before zone delivery)
    t_actual_supply = t_after_coil + dt_sup_fan

    return AHUStepOutputs(
        requested_supply_temperature_c=requested_t_sup,
        actual_supply_temperature_c=t_actual_supply,
        actual_supply_flow_m3_h=actual_supply_m3_h,
        actual_extract_flow_m3_h=actual_extract_m3_h,
        required_heating_coil_power_w=required_heating_operating_w * op,
        actual_heating_coil_power_w=actual_heating_operating_w * op,
        required_cooling_coil_power_w=required_cooling_operating_w * op,
        actual_cooling_coil_power_w=actual_cooling_operating_w * op,
        fan_electric_power_w=fan_electric_power_w,
        heat_recovery_power_w=hr_power_operating_w * op,
        bypass_fraction=bypass_fraction,
        frost_protection_required=frost_protection_req,
    )


# ---------------------------------------------------------------------------
# ISO 52016-1 zone adapter
# ---------------------------------------------------------------------------

def _opt_float(v) -> "float | None":
    return None if v is None else float(v)


def _parse_bool(v, *, key: str) -> bool:
    """Parse a config flag robustly.

    Native bools pass through; recognised strings are mapped explicitly so a
    JSON/CSV value of ``"false"`` does not become ``True`` via ``bool("false")``.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    raise ValueError(
        f"{key!r} must be a boolean or a recognised string "
        f"(true/false/yes/no/on/off/1/0); got {v!r}"
    )


def sensible_ahu_config_from_dict(d: dict) -> SensibleAHUConfig:
    """Build a SensibleAHUConfig from a ventilation component dict.

    Required keys:
      ``sensible_heat_recovery_efficiency`` — float in [0, 1]

    Supply temperature control — ``supply_temperature_mode`` selects one mode:
      "fixed" (default):        ``supply_temperature_setpoint_c`` [°C]
      "outdoor_compensated" /
      "extract_compensated":    ``supply_temperature_points`` — list of
                                [T_reference, T_supply] pairs, e.g.
                                [[-20, 19], [15, 17]]
      All modes accept optional ``supply_temperature_minimum_c`` and
      ``supply_temperature_maximum_c`` clamps [°C].

    Optional keys and their defaults:
      ``heat_recovery_control``             — "modulating_bypass"
      ``frost_control``                     — "exhaust_limit"
      ``frost_exhaust_limit_c``             — -5.0 [°C]
      ``heating_coil_max_power_w``          — None (unlimited)
      ``cooling_coil_enabled``              — False
      ``cooling_coil_max_power_w``          — None (unlimited; ignored when disabled)
      ``supply_fan_specific_power_w_per_m3_s`` — 0.0  (nominal SFP at design flow)
      ``extract_fan_specific_power_w_per_m3_s`` — 0.0  (nominal SFP at design flow)
      ``supply_fan_heat_fraction_to_air``   — 0.90
      ``extract_fan_heat_fraction_to_air``  — 0.90
      ``fan_performance_model``              — "en16798_5_1" (see FanPerformanceModel)
      ``supply_fan_design_pressure_pa``      — derived from nominal SFP and efficiency
      ``extract_fan_design_pressure_pa``     — derived from nominal SFP and efficiency
      ``supply_fan_nominal_efficiency``      — 0.72
      ``extract_fan_nominal_efficiency``     — 0.78
      ``*_fan_reference_flow_m3_h``          — required/design flow for the timestep
      ``*_fan_reference_efficiency``         — corresponding nominal efficiency

    See docs/ventilation_ahu_audit.html for the full key reference and the
    discussion of the defaulted fan product data.
    """
    heating_coil_max = d.get("heating_coil_max_power_w")
    if heating_coil_max is not None:
        heating_coil_max = float(heating_coil_max)

    cooling_coil_max = d.get("cooling_coil_max_power_w")
    if cooling_coil_max is not None:
        cooling_coil_max = float(cooling_coil_max)

    ctrl_mode_str = str(d.get("supply_temperature_mode", "fixed")).strip().lower()
    if ctrl_mode_str in ("fixed", ""):
        raw_setpoint = d.get("supply_temperature_setpoint_c")
        if raw_setpoint is None:
            raise KeyError(
                "fixed supply control requires 'supply_temperature_setpoint_c'"
            )
        supply_ctrl = SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.FIXED,
            setpoint_c=float(raw_setpoint),
            minimum_c=_opt_float(d.get("supply_temperature_minimum_c")),
            maximum_c=_opt_float(d.get("supply_temperature_maximum_c")),
        )
    elif ctrl_mode_str in ("outdoor_compensated", "extract_compensated"):
        mode_enum = (
            SupplyTemperatureControlMode.OUTDOOR_COMPENSATED
            if ctrl_mode_str == "outdoor_compensated"
            else SupplyTemperatureControlMode.EXTRACT_COMPENSATED
        )
        raw_pts = d.get("supply_temperature_points")
        if not raw_pts:
            raise KeyError(
                f"{ctrl_mode_str} control requires 'supply_temperature_points' "
                "(list of [T_ref, T_sup] pairs, e.g. [[21, 19], [24, 17]])"
            )
        points = tuple(
            CompensationPoint(
                reference_temperature_c=float(p[0]),
                supply_temperature_c=float(p[1]),
            )
            for p in raw_pts
        )
        supply_ctrl = SupplyTemperatureControl(
            mode=mode_enum,
            points=points,
            minimum_c=_opt_float(d.get("supply_temperature_minimum_c")),
            maximum_c=_opt_float(d.get("supply_temperature_maximum_c")),
        )
    else:
        raise ValueError(
            f"Unsupported supply_temperature_mode: {ctrl_mode_str!r}. "
            "Supported: 'fixed', 'outdoor_compensated', 'extract_compensated'."
        )

    return SensibleAHUConfig(
        sensible_heat_recovery_efficiency=float(d["sensible_heat_recovery_efficiency"]),
        supply_temperature_control=supply_ctrl,
        heat_recovery_control=str(d.get("heat_recovery_control", "modulating_bypass")),
        frost_control=str(d.get("frost_control", "exhaust_limit")),
        frost_exhaust_limit_c=float(d.get("frost_exhaust_limit_c", -5.0)),
        heating_coil_max_power_w=heating_coil_max,
        cooling_coil_enabled=_parse_bool(
            d.get("cooling_coil_enabled", False), key="cooling_coil_enabled"
        ),
        cooling_coil_max_power_w=cooling_coil_max,
        supply_fan_specific_power_w_per_m3_s=float(
            d.get("supply_fan_specific_power_w_per_m3_s", 0.0)
        ),
        extract_fan_specific_power_w_per_m3_s=float(
            d.get("extract_fan_specific_power_w_per_m3_s", 0.0)
        ),
        supply_fan_heat_fraction_to_air=float(
            d.get("supply_fan_heat_fraction_to_air", 0.90)
        ),
        extract_fan_heat_fraction_to_air=float(
            d.get("extract_fan_heat_fraction_to_air", 0.90)
        ),
        fan_performance_model=str(
            d.get("fan_performance_model", "en16798_5_1")
        ),
        supply_fan_design_pressure_pa=_opt_float(
            d.get("supply_fan_design_pressure_pa")
        ),
        extract_fan_design_pressure_pa=_opt_float(
            d.get("extract_fan_design_pressure_pa")
        ),
        supply_fan_nominal_efficiency=float(
            d.get("supply_fan_nominal_efficiency", 0.72)
        ),
        extract_fan_nominal_efficiency=float(
            d.get("extract_fan_nominal_efficiency", 0.78)
        ),
        supply_fan_reference_flow_m3_h=_opt_float(
            d.get("supply_fan_reference_flow_m3_h")
        ),
        extract_fan_reference_flow_m3_h=_opt_float(
            d.get("extract_fan_reference_flow_m3_h")
        ),
        supply_fan_reference_efficiency=_opt_float(
            d.get("supply_fan_reference_efficiency")
        ),
        extract_fan_reference_efficiency=_opt_float(
            d.get("extract_fan_reference_efficiency")
        ),
    )


def ahu_outputs_to_ventilation_stream(
    outputs: AHUStepOutputs,
    name: str = "mechanical_supply",
) -> VentilationStream:
    """Convert AHU step outputs into a VentilationStream for ISO 52016-1 §6.5.10.

    The stream conductance and source temperature couple the AHU into the
    zone affine ventilation boundary:

        H_k = rho_air * cp_air * actual_supply_flow [W/K]
        T_source,k = actual_supply_temperature [°C]

    When the AHU is off (zero flow), H_k = 0 so the stream contributes
    nothing to the zone balance regardless of source temperature.
    """
    h_w_k = _RHO_CP_J_M3_K * outputs.actual_supply_flow_m3_h / 3600.0
    return VentilationStream(
        name=name,
        heat_transfer_coefficient_w_k=h_w_k,
        source_temperature_c=outputs.actual_supply_temperature_c,
        category="supply",
    )
