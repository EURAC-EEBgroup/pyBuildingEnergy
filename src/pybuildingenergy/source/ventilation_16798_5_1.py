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
from . import ahu_moisture_16798_5_1 as _mx


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
    # EN 16798-5-1 SUP_AIR_TEMP_CTRL (Table 19)
    NONE = "none"                          # NO_CTRL: no supply temperature control (53a), (67a)
    ODA_COMPENSATED = "oda_compensated"    # ODA_COMP (29b): min(max_c, max(min_c, f_e*theta_e + offset))
    LOAD_COMPENSATED = "load_compensated"  # LOAD_COMP (29c): required supply temperature given at runtime


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
    slope: float | None = None              # f_e, ODA_COMP (29b)
    offset_c: float | None = None           # delta_theta_off, ODA_COMP (29b)

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
        elif mode in (SupplyTemperatureControlMode.SCHEDULED,
                      SupplyTemperatureControlMode.LOAD_COMPENSATED,
                      SupplyTemperatureControlMode.NONE):
            if self.setpoint_c is not None:
                raise ValueError(f"{mode.value} control does not accept setpoint_c")
            if points:
                raise ValueError(f"{mode.value} control does not accept compensation points")
        elif mode is SupplyTemperatureControlMode.ODA_COMPENSATED:
            if self.slope is None or self.offset_c is None:
                raise ValueError("oda_compensated control requires slope (f_e) and offset_c")
            if not (math.isfinite(self.slope) and math.isfinite(self.offset_c)):
                raise ValueError("slope and offset_c must be finite")
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
) -> float | None:
    """Return the requested AHU supply-air temperature in degrees Celsius.

    The requested setpoint is distinct from the actual delivered supply
    temperature. Heat recovery, coil capacity, frost control, and disabled
    cooling can make the requested condition unattainable.
    """

    if not isinstance(control, SupplyTemperatureControl):
        raise TypeError("control must be a SupplyTemperatureControl")

    mode = control.mode
    if mode is SupplyTemperatureControlMode.NONE:
        return None                                                     # NO_CTRL
    if mode is SupplyTemperatureControlMode.FIXED:
        requested_c = control.setpoint_c
    elif mode in (SupplyTemperatureControlMode.SCHEDULED,
                  SupplyTemperatureControlMode.LOAD_COMPENSATED):
        if scheduled_setpoint_c is None:
            raise ValueError(f"{mode.value.capitalize()} control requires scheduled_setpoint_c "
                             "(or required_supply_temperature_c for load_compensated)")
        if not math.isfinite(scheduled_setpoint_c):
            raise ValueError("scheduled_setpoint_c must be finite")
        requested_c = scheduled_setpoint_c                               # (29c)
    elif mode is SupplyTemperatureControlMode.ODA_COMPENSATED:
        requested_c = control.slope * outdoor_temperature_c + control.offset_c   # (29b)
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
    # EN 16798-5-1 6.4.3.2.7
    DIRECT = "direct"                      # (46): corrected ratios below the outdoor limits
    INDIRECT_BYPASS = "indirect_bypass"    # (47c)/(48c): supply after HR limited
    INDIRECT_PREHEAT = "indirect_preheat"  # (47b)/(48b), (90a): outdoor air preheated
    INDIRECT_RECIRC = "indirect_recirc"    # DEFR_TYPE RECIRC: exhaust mixed into outdoor air


class RecirculationControl(str, Enum):
    """EN 16798-5-1 RCA_CTRL (6.4.3.2.5, Table 21)."""

    NONE = "none"          # no recirculation: f_ODA;min = f_ODA;hr (35c)
    FIX = "fix"            # fixed outdoor air fraction (35b)
    VARIABLE = "variable"  # outdoor air fraction from the required outdoor air flow (35a)


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

    Covers balanced or unbalanced airflow, heat recovery (with the Annex D
    corrections when ``heat_recovery_data`` is given), frost protection (exhaust
    limit, direct, indirect bypass / preheat / recirculation), modulating-bypass
    economizer, recirculation (fixed or variable outdoor air share), adiabatic
    cooling on the extract side, duct and AHU leakage, duct heat losses, cooling
    with dehumidification, humidification and heating coils.  Set
    ``cooling_coil_enabled=True`` to deliver
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
    # EN 16798-5-1 6.4.3.2.6 / Annex D: heat recovery type and design data.  When given,
    # the temperature ratio is eta_hr;nom * f_q * f_v * f_n (Formula 40) and the moisture
    # recovery follows Formulae 44-45; None keeps the constant nominal efficiency.
    heat_recovery_data: "_mx.HeatRecoveryData | None" = None
    # 6.4.3.2.9: moisture processes (used when the step inputs carry humidity ratios)
    coil_data: "_mx.CoilData" = _mx.CoilData()
    supply_humidity_max_kg_kg: float | None = None    # dehumidification set-point
    supply_humidity_min_kg_kg: float | None = None    # humidification set-point
    humidifier_type: str = "STEAM"                    # STEAM or ADIABATIC (66a/66b)
    # EN 16798-5-1 6.4.2 distribution: leakage factors (Table 10, Table 11, Table B.5) and
    # duct heat transfer coefficients [W/K]
    duct_leakage_factor_supply: float = 1.0           # f_lea;du;SUP
    duct_leakage_factor_extract: float = 1.0          # f_lea;du;ETA
    duct_leakage_to_zone: bool = True                 # supply ducts inside the zone (8)/(22)
    ahu_leakage_factor_supply: float = 1.0            # f_lea;ahu;SUP (3)
    ahu_leakage_factor_extract: float = 1.0           # f_lea;ahu;ETA
    hr_leakage_factor_supply: float = 1.0             # f_lea;hr;SUP (Table B.5)
    hr_leakage_factor_extract: float = 1.0            # f_lea;hr;ETA
    outdoor_air_fraction_hr: float = 1.0              # f_ODA;hr (Table B.5)
    duct_supply_h_unconditioned_w_k: float = 0.0      # H_du;SUP;nc (14)
    duct_supply_h_zone_w_k: float = 0.0               # H_du;SUP;zt (15)
    duct_extract_h_unconditioned_w_k: float = 0.0     # H_du;ETA;nc (19)
    duct_surrounding_temperature_c: float | None = None  # theta_sur;nc (None: outdoor air)
    # 6.4.3.2.5 recirculation
    recirculation_control: RecirculationControl | str = RecirculationControl.NONE
    outdoor_air_fraction_min: float | None = None     # RCA_CTRL = FIX (35b)
    # 6.4.3.2.7 frost protection, direct control (46) and Table B.14 limits
    frost_direct_ratio_minus15: float | None = None   # corrected ratio below theta_e;lim1
    frost_direct_ratio_minus7: float | None = None    # corrected ratio below theta_e;lim2
    frost_direct_t_lim1_c: float = _mx.FROST_DIRECT_T_LIM1_C
    frost_direct_t_lim2_c: float = _mx.FROST_DIRECT_T_LIM2_C
    # 6.4.3.2.8 adiabatic cooling (humidifier in the extract air before the HR)
    adiabatic_cooling: bool = False
    adiabatic_cooling_efficiency: float = _mx.ETA_HU_AC_DEFAULT
    # Table 27: air density at the site altitude
    altitude_m: float = 0.0
    # 6.4.3.3.1 fans: FAN_CTRL (74)-(75) and characteristic curves (73)
    fan_control: str = "direct"                      # no_ctrl, direct, const_pres, min_pres
    supply_fan_pressure_curve: tuple | None = None   # f_dp(q/q_nom), polynomial, highest power first
    extract_fan_pressure_curve: tuple | None = None
    supply_fan_efficiency_curve: tuple | None = None  # f_eta(q/q_nom)
    extract_fan_efficiency_curve: tuple | None = None
    supply_fan_pressure_control_fraction: float = 0.4   # f_dp;SUP;ctrl
    extract_fan_pressure_control_fraction: float = 0.67  # f_dp;ETA;ctrl
    # (61), (64), (68): coil temperature ratios (Table B.8)
    cooling_coil_efficiency: float = _mx.TABLE_B8["eta_coil_C"]
    heating_coil_efficiency: float = _mx.TABLE_B8["eta_coil_H"]
    # 6.4.3.3.8-9: AHU casing heat losses
    ahu_location: str = "unconditioned"              # AHU_LOC: conditioned / unconditioned
    ahu_supply_casing_au_w_k: float = 0.0            # (A U)ahu;SUP
    ahu_extract_casing_au_w_k: float = 0.0           # (A U)ahu;ETA
    # 6.4.3.3.10 auxiliary energy
    rotor_drive_power_max_w: float = 120.0           # P_hr;rot;max (Table B.6)
    rotor_speed_max_rpm: float = 20.0                # n_rot;max (Table B.6)
    pump_specific_power_max_wh_m3: float = 0.03      # p_el;hr;pu;max (3e-5 kWh/m3)
    heat_recovery_power_max_w: float = 500.0         # Phi_hr;max (Table B.6)
    heat_recovery_part_load_min: float = 0.5         # f_pl;hr;min (Table B.6)
    heat_recovery_design_pressure_drop_pa: float | None = None  # dp_SUP+ETA;des;hr (89)
    controls_power_w: float = 0.0                    # sum P_el;V;ctrl (91)
    controls_fan_power_fraction: float | None = None  # NA.9 (IT): sum P_el;V;ctrl = 0.05 P_el;fan
    national_data: str = "IT"                        # default data set (Table NA / Annex B)
    controls_operation_factor: float = 1.0           # f_op;ctrl
    humidifier_pump_specific_power_wh_m3: float = 0.0  # p_el;HU;des (92b)
    humidifier_design_flow_m3_h: float | None = None   # q_V;SUP;HU;des
    humidifier_design_water_kg_h: float | None = None  # q_m;w;HU;des
    humidifier_control: str = "on_off"               # HUM_CTRL: no_ctrl, on_off, speed (93)
    # Annex C, (32)-(33): ground preheating and outdoor air duct
    ground_exchanger: "_mx.GroundExchangerData | None" = None
    duct_outdoor_h_w_k: float = 0.0                  # outdoor air duct, analogous to (14)
    # SUP_FAN_LOC / ETA_FAN_LOC (Tables 12-13): position of the fans relative to the heat
    # recovery.  DOWN_HR supply fan: its heat reaches the supply after the coils (70);
    # UP_HR: it warms the outdoor air before the heat recovery (32a) and is counted once.
    # UP_HR extract fan: its heat enters the heat recovery (31a); DOWN_HR: it leaves with
    # the exhaust air (31b).
    supply_fan_location: str = "DOWN_HR"
    extract_fan_location: str = "UP_HR"

    def __post_init__(self) -> None:
        if str(self.humidifier_type).upper() not in _mx.HUMIDIFIER_TYPES:
            raise ValueError(f"humidifier_type must be one of {_mx.HUMIDIFIER_TYPES} "
                             "(all but STEAM are adiabatic, (66b))")
        for _k in ("supply_humidity_max_kg_kg", "supply_humidity_min_kg_kg"):
            _v = getattr(self, _k)
            if _v is not None and (not math.isfinite(_v) or _v < 0.0):
                raise ValueError(f"{_k} must be finite and >= 0")
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

        try:
            _rc = self.recirculation_control
            object.__setattr__(self, "recirculation_control",
                               _rc if isinstance(_rc, RecirculationControl)
                               else RecirculationControl(str(_rc).strip().lower()))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported recirculation_control: {self.recirculation_control!r}") from exc
        for name in ("duct_leakage_factor_supply", "duct_leakage_factor_extract",
                     "ahu_leakage_factor_supply", "ahu_leakage_factor_extract",
                     "hr_leakage_factor_supply", "hr_leakage_factor_extract"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 1.0:
                raise ValueError(f"{name} must be finite and >= 1")
        if not math.isfinite(self.outdoor_air_fraction_hr) or not (0.0 < self.outdoor_air_fraction_hr <= 1.0):
            raise ValueError("outdoor_air_fraction_hr must be in (0, 1]")
        for name in ("duct_supply_h_unconditioned_w_k", "duct_supply_h_zone_w_k",
                     "duct_extract_h_unconditioned_w_k"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.duct_surrounding_temperature_c is not None and not math.isfinite(
                self.duct_surrounding_temperature_c):
            raise ValueError("duct_surrounding_temperature_c must be finite")
        if self.recirculation_control is RecirculationControl.FIX:
            f = self.outdoor_air_fraction_min
            if f is None or not math.isfinite(f) or not (0.0 < f <= 1.0):
                raise ValueError("recirculation_control='fix' requires outdoor_air_fraction_min in (0, 1]")
        if self.frost_control is FrostControlMode.DIRECT:
            for name in ("frost_direct_ratio_minus15", "frost_direct_ratio_minus7"):
                v = getattr(self, name)
                if v is None or not math.isfinite(v) or not (0.0 <= v <= 1.0):
                    raise ValueError(f"frost_control='direct' requires {name} in [0, 1] (EN 13053)")
        if not (0.0 < self.adiabatic_cooling_efficiency <= 1.0):
            raise ValueError("adiabatic_cooling_efficiency must be in (0, 1]")
        for name in ("supply_fan_location", "extract_fan_location"):
            loc_f = str(getattr(self, name)).strip().upper()
            if loc_f not in ("UP_HR", "DOWN_HR"):
                raise ValueError(f"{name} must be 'UP_HR' or 'DOWN_HR'")
            object.__setattr__(self, name, loc_f)
        fcl = str(self.fan_control).strip().lower()
        if fcl not in ("no_ctrl", "direct", "const_pres", "min_pres"):
            raise ValueError("fan_control must be 'no_ctrl', 'direct', 'const_pres' or 'min_pres'")
        object.__setattr__(self, "fan_control", fcl)
        if fcl == "no_ctrl" and not (self.supply_fan_pressure_curve and self.extract_fan_pressure_curve):
            raise ValueError("fan_control='no_ctrl' requires supply_fan_pressure_curve and "
                             "extract_fan_pressure_curve (manufacturer data, EN ISO 5801)")
        loc = str(self.ahu_location).strip().lower()
        if loc not in ("conditioned", "unconditioned"):
            raise ValueError("ahu_location must be 'conditioned' or 'unconditioned'")
        object.__setattr__(self, "ahu_location", loc)
        hc = str(self.humidifier_control).strip().lower()
        if hc not in ("no_ctrl", "on_off", "speed"):
            raise ValueError("humidifier_control must be 'no_ctrl', 'on_off' or 'speed'")
        object.__setattr__(self, "humidifier_control", hc)
        for name in ("cooling_coil_efficiency", "heating_coil_efficiency"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")
        if not (-500.0 < self.altitude_m < 9000.0):
            raise ValueError("altitude_m out of range")
        for name in ("ahu_supply_casing_au_w_k", "ahu_extract_casing_au_w_k", "rotor_drive_power_max_w",
                     "pump_specific_power_max_wh_m3", "controls_power_w",
                     "humidifier_pump_specific_power_wh_m3", "duct_outdoor_h_w_k"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")


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
    # humidity ratios [kg/kg dry air]; both given -> moisture processes are calculated
    outdoor_humidity_ratio: float | None = None
    extract_humidity_ratio: float | None = None
    # outdoor air flow required by the zone(s) at design flow (RCA_CTRL = VARIABLE, (35a))
    required_outdoor_air_flow_m3_h: float | None = None
    # LOAD_COMP (29c): supply temperature required by the zone (M5-5)
    required_supply_temperature_c: float | None = None
    # inputs from the generation modules (M4-1, M3-1): actual coil states (62)-(64), (69)
    cooling_coil_medium_temperature_c: float | None = None   # theta_C;ahu;in
    cooling_coil_output_w: float | None = None               # Q_C;ahu;out / t_ci
    heating_coil_input_w: float | None = None                # Q_H;ahu;in / t_ci
    # Annex C: hour of the year (t_an = 0 at the start of the year)
    hour_of_year: float | None = None
    # (11): maximum zone part load factor f_V;max, for FAN_CTRL = MIN_PRES (74e)
    max_zone_flow_ratio: float | None = None

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

        # supply and extract flows may differ (EN 16798-5-1 (41), (25)-(26)): the zone
        # adapter adds the outdoor make-up air when the extract exceeds the supply
        if self.required_outdoor_air_flow_m3_h is not None and (
                not math.isfinite(self.required_outdoor_air_flow_m3_h)
                or self.required_outdoor_air_flow_m3_h < 0.0):
            raise ValueError("required_outdoor_air_flow_m3_h must be finite and non-negative")

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
    # EN 16798-5-1 moisture processes (None / 0 when no humidity input)
    supply_humidity_ratio: float | None = None
    latent_cooling_coil_power_w: float = 0.0          # part of actual_cooling_coil_power_w
    humidification_power_w: float = 0.0               # steam humidifier energy (82a)
    humidification_water_kg_h: float = 0.0            # (83)
    heat_recovery_temperature_ratio: float | None = None
    moisture_recovery_efficiency: float | None = None
    # EN 16798-5-1 6.4.2 / 6.4.3: distribution, recirculation, frost, adiabatic cooling
    makeup_outdoor_air_flow_m3_h: float = 0.0         # extract > supply: outdoor air into the zone
    supply_fan_flow_m3_h: float = 0.0                 # q_V;SUP;ahu (25)
    extract_fan_flow_m3_h: float = 0.0                # q_V;ETA;ahu (26)
    outdoor_air_fraction: float = 1.0                 # f_ODA (38)
    recirculation_power_w: float = 0.0                # Q_RCA (79)
    frost_preheat_power_w: float = 0.0                # W_V;preh (90a)
    duct_heat_loss_w: float = 0.0                     # Q_V;ls;dis (21)
    duct_heat_loss_recoverable_w: float = 0.0         # Q_V;ls;dis;rbl (22)
    ahu_outlet_temperature_c: float | None = None     # theta_SUP;dis;in
    duct_outlet_temperature_c: float | None = None    # theta_SUP;dis;out (13)
    exhaust_temperature_c: float | None = None        # exhaust after the heat recovery
    adiabatic_cooling_active: bool = False
    adiabatic_cooling_water_kg_h: float = 0.0
    # (61), (64), (68), (81), (76), (85)-(93)
    cooling_coil_medium_temperature_required_c: float | None = None  # theta_C;ahu;in;req (61)
    heating_coil_medium_temperature_required_c: float | None = None  # theta_H;ahu;in;req (68)
    cooling_coil_temperature_c: float | None = None                   # theta_C;coil (64)
    dehumidification_power_required_w: float = 0.0                    # Q_DHU;ahu;out;req (81)
    ground_heat_transfer_w: float = 0.0                               # Q_gnd (76)
    ahu_heat_loss_w: float = 0.0                                      # Q_V;ls;gen (85)
    ahu_heat_loss_recoverable_w: float = 0.0                          # Q_V;ls;gen;rbl (86)
    auxiliary_power_w: float = 0.0                                    # W_V;aux (87)
    auxiliary_heat_recovery_w: float = 0.0                            # W_V;aux;hr (88)
    fan_power_heat_recovery_w: float = 0.0                            # E_V;gen;in;el;hr (89)
    auxiliary_controls_w: float = 0.0                                 # W_V;aux;ctrl (91)
    humidifier_auxiliary_w: float = 0.0                               # W_HU;aux (92)
    rotor_speed_rpm: float | None = None                              # n_rot (D.3)
    air_density_kg_m3: float = _RHO_AIR_KG_M3


def _polyval(coeffs, x: float) -> float:
    """Polynomial with coefficients from the highest power down (numpy.polyval order)."""
    v = 0.0
    for c in coeffs:
        v = v * x + float(c)
    return v


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
    rho_cp_j_m3_k: float = None,
    fan_control: str = "direct",
    pressure_curve: tuple | None = None,
    efficiency_curve: tuple | None = None,
    pressure_control_fraction: float = 0.0,
    max_zone_flow_ratio: float | None = None,
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
        # EN 16798-5-1 (74)-(75): pressure difference by FAN_CTRL
        if fan_control == "no_ctrl":
            if not pressure_curve:
                raise ValueError("fan_control='no_ctrl' requires the fan pressure curve f_dp(q/q_nom)")
            pressure_pa = pressure_design_pa * _polyval(pressure_curve, q_ratio)      # (74a)/(74c)
        elif fan_control == "const_pres":
            f_c = pressure_control_fraction
            pressure_pa = pressure_design_pa * ((1.0 - f_c) * q_ratio ** 2 + f_c)       # (74d)
        elif fan_control == "min_pres":
            f_c = pressure_control_fraction
            fv = q_ratio if max_zone_flow_ratio is None else max_zone_flow_ratio
            pressure_pa = pressure_design_pa * ((1.0 - f_c) * q_ratio ** 2 + f_c * fv ** 2)  # (74e)
        else:
            pressure_pa = pressure_design_pa * q_ratio ** 2                            # (74b)

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
        if efficiency_curve:
            efficiency = nominal_efficiency * _polyval(efficiency_curve, q_ratio)      # (73)
        else:
            efficiency = (
                nominal_efficiency
                * (eta_ref / nominal_efficiency)
                * math.sqrt(operating_flow_m3_h / q_ref_m3_h)
            )
        if efficiency <= 0.0:
            raise ValueError("fan efficiency from the efficiency curve must be > 0")
        power_operating_w = q_operating_m3_s * max(pressure_pa, 0.0) / efficiency      # (72)

    temperature_rise_k = (
        power_operating_w * heat_fraction_to_air
        / ((rho_cp_j_m3_k or _RHO_CP_J_M3_K) * q_operating_m3_s)
    )
    return power_operating_w * operation_fraction, temperature_rise_k


def _fan_kw(config, side: str, rho_cp: float, inputs) -> dict:
    """FAN_CTRL settings of the supply or extract fan for _fan_power_and_temperature_rise."""
    return {
        "rho_cp_j_m3_k": rho_cp,
        "fan_control": config.fan_control,
        "pressure_curve": getattr(config, f"{side}_fan_pressure_curve"),
        "efficiency_curve": getattr(config, f"{side}_fan_efficiency_curve"),
        "pressure_control_fraction": getattr(config, f"{side}_fan_pressure_control_fraction"),
        "max_zone_flow_ratio": inputs.max_zone_flow_ratio,
    }


def calculate_sensible_ahu_step(
    config: SensibleAHUConfig,
    inputs: AHUStepInputs,
) -> AHUStepOutputs:
    """Calculate one timestep of the EN 16798-5-1 AHU model.

    Air path (Figure 1): extract duct (18)-(20) -> extract fan heat -> frost protection
    (46)-(49) or adiabatic cooling (50)-(52) -> heat recovery (39)-(45) -> recirculation
    (34)-(38) -> cooling/dehumidification (53)-(63) -> humidification (65)-(66) ->
    heating coil -> supply fan heat -> supply duct (12)-(17).  Flows follow (4a)-(10) and
    (25)-(28): ``required_supply_flow_m3_h`` / ``required_extract_flow_m3_h`` are the
    flows required by the zone; the AHU moves them increased by the duct, AHU and heat
    recovery leakage factors.  The coil target is pre-compensated for the supply-fan heat
    and the supply duct losses (12), so the delivered temperature meets the requested
    setpoint when capacity allows.

    With the default configuration (no leakage, no duct losses, no recirculation,
    balanced flow) this reduces to the sensible single-zone model.

    Deviations from the printed formulae, chosen for energy consistency:
      * (18)/(20): the extract duct loss lowers the extract temperature (printed ``+``)
        and the leakage air is mixed by flow, ``(x_zv + (f-1) x_sur) / f``;
      * (47b)/(48c): the allowed supply-side temperature rise is the exhaust drop times
        ``q_exhaust,hr / q_outdoor,hr`` (printed ratio inverted; equal for balanced flow);
      * (50): with adiabatic cooling the supply after recovery is the recovered
        temperature, never colder than the requirement given by (51a).

    Power outputs are timestep averages (scaled by operation_fraction); temperature
    outputs describe the operating airstream.  When operation_fraction, flow_fraction,
    or required supply flow is zero, all flow and energy outputs are zero and the actual
    supply temperature equals outdoor.
    """
    if not isinstance(config, SensibleAHUConfig):
        raise TypeError("config must be a SensibleAHUConfig")
    if not isinstance(inputs, AHUStepInputs):
        raise TypeError("inputs must be an AHUStepInputs")

    op = inputs.operation_fraction
    flow_fraction = inputs.flow_fraction
    rho = _mx.air_density(config.altitude_m)               # Table 27, note a
    rho_cp = rho * _CP_AIR_J_KG_K
    # flows required by the zone (operating) - (6), (7)
    q_zv_m3_h = flow_fraction * inputs.required_supply_flow_m3_h
    q_eta_zv_m3_h = flow_fraction * inputs.required_extract_flow_m3_h

    if op <= 0.0 or q_zv_m3_h <= 0.0:
        # no supply: an extract-only operation still runs the extract fan (72); the zone
        # takes the extracted air from outdoors
        p_eta_only = 0.0
        q_eta_only = 0.0
        if op > 0.0 and q_eta_zv_m3_h > 0.0:
            q_eta_only = (config.ahu_leakage_factor_extract * config.duct_leakage_factor_extract
                          * q_eta_zv_m3_h)                                      # (5a), (26)
            p_eta_only, _ = _fan_power_and_temperature_rise(
                model=config.fan_performance_model,
                operating_flow_m3_h=q_eta_only,
                design_flow_m3_h=(q_eta_only / flow_fraction if flow_fraction > 0.0 else q_eta_only),
                operation_fraction=op,
                nominal_specific_power_w_per_m3_s=config.extract_fan_specific_power_w_per_m3_s,
                design_pressure_pa=config.extract_fan_design_pressure_pa,
                nominal_efficiency=config.extract_fan_nominal_efficiency,
                reference_flow_m3_h=config.extract_fan_reference_flow_m3_h,
                reference_efficiency=config.extract_fan_reference_efficiency,
                heat_fraction_to_air=config.extract_fan_heat_fraction_to_air,
                **_fan_kw(config, "extract", rho_cp, inputs),
            )
        return AHUStepOutputs(
            requested_supply_temperature_c=None,
            actual_supply_temperature_c=inputs.outdoor_temperature_c,
            actual_supply_flow_m3_h=0.0,
            actual_extract_flow_m3_h=op * q_eta_zv_m3_h if op > 0.0 else 0.0,
            required_heating_coil_power_w=0.0,
            actual_heating_coil_power_w=0.0,
            required_cooling_coil_power_w=0.0,
            actual_cooling_coil_power_w=0.0,
            fan_electric_power_w=p_eta_only,
            heat_recovery_power_w=0.0,
            bypass_fraction=0.0,
            frost_protection_required=False,
            makeup_outdoor_air_flow_m3_h=op * q_eta_zv_m3_h if op > 0.0 else 0.0,
            outdoor_air_fraction=0.0,
            extract_fan_flow_m3_h=op * q_eta_only,
            auxiliary_controls_w=config.controls_power_w * config.controls_operation_factor,
            auxiliary_power_w=config.controls_power_w * config.controls_operation_factor,
            air_density_kg_m3=rho,
        )

    requested_t_sup = resolve_supply_temperature_setpoint(
        config.supply_temperature_control,
        inputs.outdoor_temperature_c,
        inputs.extract_temperature_c,
        (inputs.scheduled_setpoint_c if inputs.scheduled_setpoint_c is not None
         else inputs.required_supply_temperature_c),
    )
    no_ctrl = requested_t_sup is None                      # SUP_AIR_TEMP_CTRL = NO_CTRL
    sup_fan_up = config.supply_fan_location == "UP_HR"

    t_oda = inputs.outdoor_temperature_c
    t_zone = inputs.extract_temperature_c         # zone air (previous step)
    t_sur = (config.duct_surrounding_temperature_c
             if config.duct_surrounding_temperature_c is not None else t_oda)
    humid = (inputs.outdoor_humidity_ratio is not None
             and inputs.extract_humidity_ratio is not None)
    x_oda = float(inputs.outdoor_humidity_ratio) if humid else None
    x_zone = float(inputs.extract_humidity_ratio) if humid else None

    # ---------------------------------------------------------------- distribution flows
    f_du_s = config.duct_leakage_factor_supply
    f_du_e = config.duct_leakage_factor_extract
    q_dis_in_m3_h = f_du_s * q_zv_m3_h                                     # (4a)
    q_eta_dis_out_m3_h = f_du_e * q_eta_zv_m3_h                            # (5a)
    q_lea_m3_h = q_dis_in_m3_h - q_zv_m3_h                                 # (8)/(10)
    q_lea_zone_m3_h = q_lea_m3_h if config.duct_leakage_to_zone else 0.0
    q_lea_nc_m3_h = q_lea_m3_h - q_lea_zone_m3_h
    q_delivered_m3_h = q_zv_m3_h + q_lea_zone_m3_h
    q_makeup_m3_h = max(0.0, q_eta_zv_m3_h - q_delivered_m3_h)

    # ---------------------------------------------------------------- recirculation share
    f_hr = config.outdoor_air_fraction_hr
    adiabatic = bool(config.adiabatic_cooling) and humid
    rc = config.recirculation_control
    if rc is RecirculationControl.FIX:
        f_min = min(f_hr, float(config.outdoor_air_fraction_min))          # (35b)
    elif rc is RecirculationControl.VARIABLE and inputs.required_outdoor_air_flow_m3_h is not None:
        q_oda_req = flow_fraction * inputs.required_outdoor_air_flow_m3_h
        f_min = min(f_hr, max(0.0, min(1.0, q_oda_req / q_zv_m3_h)))       # (35a)
    else:
        f_min = f_hr                                                       # (35c)

    def _ahu_flows(f):
        q_s = config.ahu_leakage_factor_supply * (
            (1.0 - f) + f * config.hr_leakage_factor_supply) * q_dis_in_m3_h      # (25)
        q_e = config.ahu_leakage_factor_extract * (
            (1.0 - f) + f * config.hr_leakage_factor_extract) * q_eta_dis_out_m3_h  # (26)
        return q_s, q_e

    # ---------------------------------------------------------------- fans
    q_sup_ahu_m3_h, q_eta_ahu_m3_h = _ahu_flows(f_min)
    k_s = q_sup_ahu_m3_h / q_zv_m3_h
    k_e = q_eta_ahu_m3_h / q_eta_zv_m3_h if q_eta_zv_m3_h > 0.0 else 1.0
    p_sup_fan, dt_sup_fan = _fan_power_and_temperature_rise(
        model=config.fan_performance_model,
        operating_flow_m3_h=q_sup_ahu_m3_h,
        design_flow_m3_h=k_s * inputs.required_supply_flow_m3_h,
        operation_fraction=op,
        nominal_specific_power_w_per_m3_s=config.supply_fan_specific_power_w_per_m3_s,
        design_pressure_pa=config.supply_fan_design_pressure_pa,
        nominal_efficiency=config.supply_fan_nominal_efficiency,
        reference_flow_m3_h=config.supply_fan_reference_flow_m3_h,
        reference_efficiency=config.supply_fan_reference_efficiency,
        heat_fraction_to_air=config.supply_fan_heat_fraction_to_air,
        **_fan_kw(config, "supply", rho_cp, inputs),
    )
    p_eta_fan, dt_eta_fan = _fan_power_and_temperature_rise(
        model=config.fan_performance_model,
        operating_flow_m3_h=q_eta_ahu_m3_h,
        design_flow_m3_h=k_e * inputs.required_extract_flow_m3_h,
        operation_fraction=op,
        nominal_specific_power_w_per_m3_s=config.extract_fan_specific_power_w_per_m3_s,
        design_pressure_pa=config.extract_fan_design_pressure_pa,
        nominal_efficiency=config.extract_fan_nominal_efficiency,
        reference_flow_m3_h=config.extract_fan_reference_flow_m3_h,
        reference_efficiency=config.extract_fan_reference_efficiency,
        heat_fraction_to_air=config.extract_fan_heat_fraction_to_air,
        **_fan_kw(config, "extract", rho_cp, inputs),
    )
    fan_electric_power_w = p_sup_fan + p_eta_fan

    # ---------------------------------------------------------------- extract duct (18)-(20)
    if q_eta_dis_out_m3_h > 0.0:
        a_e = config.duct_extract_h_unconditioned_w_k / (rho_cp * q_eta_dis_out_m3_h / 3600.0)
        d_eta_du = (t_zone - t_sur) * (1.0 - math.exp(-a_e))                       # (19)
    else:
        d_eta_du = 0.0
    t_eta_dis_out = t_zone / f_du_e + (f_du_e - 1.0) / f_du_e * t_sur - d_eta_du  # (18)
    x_eta_dis_out = ((x_zone + (f_du_e - 1.0) * x_oda) / f_du_e) if humid else None  # (20)

    # ---------------------------------------------------------------- supply target (12)
    q_dis_in_m3_s = q_dis_in_m3_h / 3600.0
    a_nc = config.duct_supply_h_unconditioned_w_k / (rho_cp * q_dis_in_m3_s)
    a_zt = config.duct_supply_h_zone_w_k / (rho_cp * q_dis_in_m3_s)
    if no_ctrl:
        t_target = None                     # set to theta_SUP;RCA after the mixing (53a)
    else:
        d_nc_req = (requested_t_sup - t_sur) * (1.0 - math.exp(-a_nc))             # (14)
        d_zt_req = (requested_t_sup - d_nc_req - t_zone) * (1.0 - math.exp(-a_zt))  # (15)
        t_dis_in_req = requested_t_sup + d_nc_req + d_zt_req                       # (12)
        # (29): the coil target leaves room for the supply fan heat added downstream
        t_target = t_dis_in_req - (dt_sup_fan if not sup_fan_up else 0.0)

    # ---------------------------------------------------------------- outdoor air preheating
    # outdoor air duct (32)-(33), analogous to (14), and ground heat exchanger (Annex C)
    q_oda_m3_h = f_min * q_sup_ahu_m3_h
    if config.duct_outdoor_h_w_k > 0.0 and q_oda_m3_h > 0.0:
        d_oda_du = (t_sur - t_oda) * (1.0 - math.exp(
            -config.duct_outdoor_h_w_k / (rho_cp * q_oda_m3_h / 3600.0)))
    else:
        d_oda_du = 0.0
    d_gnd = dx_gnd = 0.0
    if config.ground_exchanger is not None:
        if inputs.hour_of_year is None:
            raise ValueError("a ground heat exchanger needs AHUStepInputs.hour_of_year (Annex C)")
        g = _mx.ground_preheating(config.ground_exchanger, t_oda, x_oda, inputs.hour_of_year,
                                  q_oda_m3_h, rho_cp)
        d_gnd, dx_gnd = g["d_theta"], g["d_x"]
    # (32a)/(32d): with the supply fan upstream of the heat recovery its heat warms the
    # outdoor air here, and is not added again after the coils (counted once)
    t_pre = t_oda + d_gnd + d_oda_du + (dt_sup_fan if sup_fan_up else 0.0)
    if (config.ground_exchanger is not None and not no_ctrl
            and str(config.ground_exchanger.control).lower() == "bypass"):
        t_pre = (min(t_target, t_pre) if t_target > t_oda else max(t_target, t_pre))  # (32e)/(32f)
        d_sum = d_gnd + d_oda_du + (dt_sup_fan if sup_fan_up else 0.0)
        if (t_pre - t_oda) * d_sum <= 0.0:
            t_pre = t_oda if abs(d_sum) > 0.0 else t_pre
    x_pre = (x_oda + dx_gnd) if humid else None                                    # (33)

    # ---------------------------------------------------------------- heat recovery inlets
    t_eta_hr_in = t_eta_dis_out + (dt_eta_fan if config.extract_fan_location == "UP_HR"
                                   else 0.0)                                       # (31a)/(31b)
    x_eta_hr_in = x_eta_dis_out
    adiabatic_on = False
    if adiabatic and not no_ctrl and t_target < t_pre and t_eta_hr_in < t_pre:
        adiabatic_on = True
        f_min = f_hr                                     # 6.4.3.2.8: recirculation off
        q_sup_ahu_m3_h, q_eta_ahu_m3_h = _ahu_flows(f_min)

    # Heat recovery flows at f_ODA;min (41), (43).  Full recirculation (f_ODA;min = 0,
    # e.g. RCA_CTRL VARIABLE with no outdoor air required): no outdoor air crosses the
    # recovery, unless the modulating control can open the outdoor air share up to
    # f_ODA;hr (38b), in which case the recovery is evaluated at f_ODA;hr.
    modulating = (config.heat_recovery_control is HeatRecoveryControl.MODULATING_BYPASS
                  and not adiabatic_on and not no_ctrl)
    f_eval = f_min if f_min > 1e-12 else (f_hr if modulating else 0.0)
    q_oda_hr = f_eval * q_sup_ahu_m3_h
    q_eha_hr = q_eta_ahu_m3_h - (1.0 - f_eval) * q_sup_ahu_m3_h
    hr_flows = q_oda_hr > 1e-12 and q_eha_hr > 1e-12
    if q_eha_hr > 1e-12:
        r_flow = q_oda_hr / q_eha_hr                     # supply / exhaust in the HR
    else:
        r_flow = math.inf if q_oda_hr > 1e-12 else 0.0

    hr_ratio = None
    if not hr_flows:
        eta_nom = 0.0
    elif config.heat_recovery_data is not None:
        # (40)-(43): eta_hr = eta_hr;nom * f_q * f_v * f_n
        hr_ratio = _mx.temperature_ratio(config.heat_recovery_data, q_sup_ahu_m3_h,
                                         q_eta_ahu_m3_h, f_oda_min=f_eval)
        eta_nom = hr_ratio["eta_hr"]
    else:
        f_q = (q_eha_hr / q_oda_hr) ** 0.4 if abs(q_eha_hr - q_oda_hr) > 1e-9 else 1.0  # (41)
        eta_nom = max(0.0, min(1.0, config.sensible_heat_recovery_efficiency * f_q))

    if adiabatic_on:
        # (51a)/(52a): extract air cooled by the humidifier, no colder than needed
        t_wb = _mx.wet_bulb_temperature(t_eta_dis_out, x_eta_dis_out)
        x_wb = _mx.saturation_humidity_ratio(t_wb)
        eta_hu = config.adiabatic_cooling_efficiency
        t_ac = t_eta_dis_out - eta_hu * (t_eta_dis_out - t_wb)
        t_need = (t_target - t_pre) / eta_nom + t_pre if eta_nom > 0.0 else t_ac
        t_eta_hr_in = max(t_need, t_ac)
        if abs(t_eta_dis_out - t_wb) > 1e-9:
            x_eta_hr_in = x_eta_dis_out + (x_wb - x_eta_dis_out) * (
                t_eta_dis_out - t_eta_hr_in) / (t_eta_dis_out - t_wb)
        t_eta_hr_in = min(t_eta_hr_in, t_eta_dis_out)
        x_eta_hr_in = max(x_eta_hr_in, x_eta_dis_out)

    # ---------------------------------------------------------------- frost protection
    t_fp, x_fp = t_pre, x_pre
    frost_protection_req = False
    frost_preheat_operating_w = 0.0
    oda_mix = 0.0                                  # DEFR_TYPE RECIRC: exhaust share in ODA
    eta_eff = eta_nom
    dT = t_eta_hr_in - t_pre
    fc = config.frost_control
    lim = config.frost_exhaust_limit_c
    if fc is FrostControlMode.DIRECT:
        if t_oda < config.frost_direct_t_lim1_c:                                     # (46)
            eta_eff, frost_protection_req = min(eta_nom, config.frost_direct_ratio_minus15), True
        elif t_oda < config.frost_direct_t_lim2_c:
            eta_eff, frost_protection_req = min(eta_nom, config.frost_direct_ratio_minus7), True
    elif fc is not FrostControlMode.NONE and eta_nom > 0.0 and dT > 1e-9 and math.isfinite(r_flow):
        t_eha = t_eta_hr_in - r_flow * eta_nom * dT
        if t_eha < lim:
            frost_protection_req = True
            if fc in (FrostControlMode.EXHAUST_LIMIT, FrostControlMode.INDIRECT_BYPASS):
                # (48c): supply after HR limited so the exhaust stays at the limit (49)
                eta_eff = max(0.0, min(eta_nom, (t_eta_hr_in - lim) / (r_flow * dT)))
            else:
                # (47b): outdoor air raised so the exhaust stays at the limit
                t_fp = min(t_eta_hr_in, t_eta_hr_in - (t_eta_hr_in - lim) / (r_flow * eta_nom))
                t_fp = max(t_fp, t_pre)
                if fc is FrostControlMode.INDIRECT_PREHEAT:
                    frost_preheat_operating_w = (rho_cp * q_oda_hr / 3600.0
                                                 * (t_fp - t_pre))                 # (90a)
                else:   # RECIRC: exhaust air mixed into the outdoor air
                    oda_mix = (t_fp - t_pre) / dT
                    if humid:
                        x_fp = x_pre + oda_mix * (x_eta_hr_in - x_pre)

    t_hr_full = t_fp + eta_eff * (t_eta_hr_in - t_fp)                               # (39a)

    # ---------------------------------------------------------------- control: f_ODA, bypass
    f_oda = f_min
    t_hr = t_hr_full
    bp = 0.0
    if modulating:
        # (38b): outdoor air share between f_ODA;min and f_ODA;hr to approach the target
        if f_hr - f_min > 1e-12 and abs(t_hr_full - t_eta_hr_in) > 1e-9:
            f_oda = max(f_min, min(f_hr, (t_target - t_eta_hr_in) / (t_hr_full - t_eta_hr_in)))
        # (39b)/(39c): heat recovery bypass with the outdoor share fixed
        if f_oda > 0.0 and abs(t_hr_full - t_fp) > 1e-9:
            t_hr_needed = (t_target - (1.0 - f_oda) * t_eta_hr_in) / f_oda
            bp = max(0.0, min(1.0, (t_hr_full - t_hr_needed) / (t_hr_full - t_fp)))
            t_hr = t_fp * bp + t_hr_full * (1.0 - bp)
    t_rca = (1.0 - f_oda) * t_eta_hr_in + f_oda * t_hr                              # (36a)
    if no_ctrl:
        t_target = t_rca                                                           # (53a)

    if eta_nom > 0.0 and abs(t_eta_hr_in - t_fp) > 1e-9:
        bypass_fraction = max(0.0, min(1.0, 1.0 - (t_hr - t_fp) / (eta_nom * (t_eta_hr_in - t_fp))))
    else:
        bypass_fraction = 0.0

    # coils work on the AHU supply flow (25)
    q_sup = q_sup_ahu_m3_h / 3600.0
    rho_cp_q = rho_cp * q_sup
    hr_power_operating_w = rho_cp_q * f_oda * (t_hr - t_fp)                         # (77)
    # (76): ground / outdoor air duct only, without the upstream supply fan heat
    ground_operating_w = rho_cp_q * f_oda * (t_pre - t_oda - (dt_sup_fan if sup_fan_up else 0.0))
    recirc_operating_w = (rho_cp * q_eta_ahu_m3_h / 3600.0 * (1.0 - f_oda)
                          * (t_eta_hr_in - t_oda))                                  # (79)

    # ---------------------------------------------------------------- moisture recovery
    x_after_hr = x_actual = None
    eta_xr = None
    ac_water_kg_h = 0.0
    if humid:
        eta_xr = 0.0
        if config.heat_recovery_data is not None and hr_flows:
            eta_xr = _mx.moisture_recovery_efficiency(
                config.heat_recovery_data, x_eta_hr_in, t_oda, x_oda,
                q_sup_ahu_m3_h, q_eta_ahu_m3_h, f_oda_min=f_eval)["eta_xr"]
        eta_xr_eff = eta_xr * (1.0 - bypass_fraction)
        x_sup_hr = x_fp + eta_xr_eff * (x_eta_hr_in - x_fp)                         # (44)
        hr_power_operating_w += (q_sup * f_oda * rho * _mx.R_W_J_KG
                                 * (x_sup_hr - x_fp))                               # (77)
        x_after_hr = (1.0 - f_oda) * x_eta_dis_out + f_oda * x_sup_hr               # (37)
        if adiabatic_on:
            ac_water_kg_h = q_eha_hr * rho * (x_eta_hr_in - x_eta_dis_out) * op
    t_after_hr = t_rca

    # ---------------------------------------------------------------- cooling coil
    latent_cooling_operating_w = 0.0
    cooling_coil_temperature = None
    if humid:
        # (53b)-(63): required outlet state, with dehumidification to x_max if set
        req = _mx.cooling_and_dehumidification(
            t_after_hr, x_after_hr, t_target, config.supply_humidity_max_kg_kg, config.coil_data)
        s_req, l_req = _mx.coil_power_w(q_sup, rho, t_after_hr, x_after_hr,
                                        req["theta_SUP_C"], req["x_SUP_C"])
        required_cooling_operating_w = s_req + l_req                            # (80)
        t_c_req, x_c_req = req["theta_SUP_C"], req["x_SUP_C"]
        if not config.cooling_coil_enabled:
            actual_cooling_operating_w = 0.0
            t_after_cooling, x_after_cooling = t_after_hr, x_after_hr
        elif inputs.cooling_coil_medium_temperature_c is not None:
            # (62)-(64): actual state from the coil medium temperature and the delivered
            # cooling (input from M4-1; default: the requirement within the coil capacity)
            eta_c, f_bp = config.cooling_coil_efficiency, config.coil_data.f_C_bp
            theta_coil = (eta_c * (inputs.cooling_coil_medium_temperature_c - t_after_hr)
                          / (1.0 - f_bp) + t_after_hr)                              # (64)
            x_after_cooling = min(f_bp * x_after_hr + (1.0 - f_bp)
                                  * _mx.coil_surface_humidity_ratio(theta_coil, config.coil_data.as_dict()),
                                  x_after_hr)                                       # (63)
            q_c = inputs.cooling_coil_output_w
            if q_c is None:
                q_c = required_cooling_operating_w
                if config.cooling_coil_max_power_w is not None:
                    q_c = min(q_c, config.cooling_coil_max_power_w)
            actual_cooling_operating_w = max(0.0, float(q_c))
            t_after_cooling = (t_after_hr - actual_cooling_operating_w / (q_sup * rho_cp)
                               - _mx.R_W_J_KG / _CP_AIR_J_KG_K * (x_after_cooling - x_after_hr))  # (62)
            latent_cooling_operating_w = q_sup * rho * _mx.R_W_J_KG * (x_after_hr - x_after_cooling)
            cooling_coil_temperature = theta_coil
        else:
            act = _mx.limit_cooling_coil(
                t_after_hr, x_after_hr, req["theta_SUP_C"], q_sup, rho,
                config.cooling_coil_max_power_w, config.coil_data)
            actual_cooling_operating_w = act["Q_sens_w"] + act["Q_lat_w"]
            latent_cooling_operating_w = act["Q_lat_w"]
            t_after_cooling, x_after_cooling = act["theta_SUP_C"], act["x_SUP_C"]
            cooling_coil_temperature = act.get("theta_C_coil")
        hum = _mx.humidification(x_after_cooling, config.supply_humidity_min_kg_kg,
                                 config.humidifier_type)
        x_actual = hum["x_SUP_HU"]
        d_theta_hu = hum["d_theta_SUP_HU"]
        humidification_operating_w = (q_sup * rho * _mx.R_W_J_KG * hum["dx_HU"]
                                      if str(config.humidifier_type).upper() == "STEAM" else 0.0)  # (82a)
        humidification_water_kg_h = q_sup * rho * hum["dx_HU"] * 3600.0 * op       # (83)
        # adiabatic humidifier cools the air (66b): the heating coil compensates (69)
        t_after_cooling = t_after_cooling - d_theta_hu
    else:
        humidification_operating_w = 0.0
        humidification_water_kg_h = 0.0
        t_c_req, x_c_req = min(t_after_hr, t_target), None
        required_cooling_operating_w = rho_cp_q * max(0.0, t_after_hr - t_target)
        if not config.cooling_coil_enabled:
            actual_cooling_operating_w = 0.0
        elif inputs.cooling_coil_output_w is not None:
            actual_cooling_operating_w = max(0.0, float(inputs.cooling_coil_output_w))   # (62)
        elif config.cooling_coil_max_power_w is None:
            actual_cooling_operating_w = required_cooling_operating_w
        else:
            actual_cooling_operating_w = min(
                required_cooling_operating_w,
                config.cooling_coil_max_power_w,
            )
        t_after_cooling = t_after_hr - actual_cooling_operating_w / rho_cp_q

    # ---------------------------------------------------------------- heating coil
    # (67): with NO_CTRL the heating only follows the cooling requirement (67a)
    t_h_req = t_c_req if no_ctrl else max(t_target, t_c_req)
    required_heating_operating_w = rho_cp_q * max(0.0, t_h_req - t_after_cooling)       # (84)
    if inputs.heating_coil_input_w is not None:
        actual_heating_operating_w = max(0.0, float(inputs.heating_coil_input_w))       # (69)
    elif config.heating_coil_max_power_w is None:
        actual_heating_operating_w = required_heating_operating_w
    else:
        actual_heating_operating_w = min(required_heating_operating_w,
                                         config.heating_coil_max_power_w)
    t_after_coil = t_after_cooling + actual_heating_operating_w / rho_cp_q

    # ---------------------------------------------------------------- supply fan, duct (13)-(15)
    t_dis_in = t_after_coil + (0.0 if sup_fan_up else dt_sup_fan)                  # (70)
    d_nc = (t_dis_in - t_sur) * (1.0 - math.exp(-a_nc))                            # (14)
    d_zt = (t_dis_in - d_nc - t_zone) * (1.0 - math.exp(-a_zt))                    # (15)
    t_dis_out = t_dis_in - d_nc - d_zt                                             # (13)
    # the zone receives the supply at theta_SUP;dis;out plus the heat the duct gives to the
    # zone (15), i.e. the air at theta_SUP;dis;in - delta_nc, and the leakage into the zone
    t_zone_source = ((q_zv_m3_h * (t_dis_in - d_nc) + q_lea_zone_m3_h * t_dis_in)
                     / q_delivered_m3_h)
    rc_w = rho_cp / 3600.0
    duct_loss_operating_w = rc_w * (q_dis_in_m3_h * (d_nc + d_zt)
                                    + q_eta_dis_out_m3_h * d_eta_du
                                    + q_lea_zone_m3_h * (t_dis_in - t_zone)
                                    + q_lea_nc_m3_h * (t_dis_in - t_sur))          # (21)
    duct_rbl_operating_w = rc_w * (q_dis_in_m3_h * d_zt
                                   + q_lea_zone_m3_h * (t_dis_in - t_zone))        # (22)
    # exhaust after the recovery with the actual outdoor share: q_oda (T_hr - T_fp) is
    # taken from the exhaust stream q_eha
    q_oda_act = f_oda * q_sup_ahu_m3_h
    q_eha_act = q_eta_ahu_m3_h - (1.0 - f_oda) * q_sup_ahu_m3_h
    t_exhaust = (t_eta_hr_in - q_oda_act / q_eha_act * (t_hr - t_fp)
                 if q_eha_act > 1e-12 else None)

    # ---------------------------------------------------------------- coil medium (61), (68)
    eta_c, eta_h = config.cooling_coil_efficiency, config.heating_coil_efficiency
    t_c_medium_req = ((t_c_req - t_after_hr * (1.0 - eta_c)) / eta_c
                      if required_cooling_operating_w > 0.0 else None)              # (61)
    t_sup_c_actual = t_after_coil - actual_heating_operating_w / rho_cp_q
    t_h_medium_req = ((t_h_req - t_sup_c_actual * (1.0 - eta_h)) / eta_h
                      if required_heating_operating_w > 0.0 else None)              # (68)
    # (81): dehumidification part of the cooling requirement
    dhu_operating_w = 0.0
    if humid and config.supply_humidity_max_kg_kg is not None and x_c_req is not None:
        t_ref = t_after_hr if no_ctrl else min(t_after_hr, t_target)
        sens_only = _mx.cooling_and_dehumidification(t_after_hr, x_after_hr, t_ref, None,
                                                     config.coil_data)
        dx_c = max(0.0, x_after_hr - sens_only["x_SUP_C"])                          # (55)
        dhu_operating_w = max(0.0, q_sup * rho * (_mx.C_A_J_KG_K * (t_ref - t_c_req)
                                                  + _mx.R_W_J_KG * (x_after_hr - dx_c - x_c_req)))

    # ---------------------------------------------------------------- AHU casing (85)-(86)
    t_amb_ahu = t_zone if config.ahu_location == "conditioned" else t_sur
    q_lea_ahu_s = (config.ahu_leakage_factor_supply - 1.0) * q_dis_in_m3_h          # (27)
    q_lea_ahu_e = (config.ahu_leakage_factor_extract - 1.0) * q_eta_dis_out_m3_h    # (28)
    ahu_loss_operating_w = (config.ahu_supply_casing_au_w_k * (t_hr - t_amb_ahu)
                            + config.ahu_extract_casing_au_w_k * (t_eta_hr_in - t_amb_ahu)
                            + rc_w * q_lea_ahu_s * (t_hr - t_amb_ahu))
    if config.ahu_location != "conditioned":
        ahu_loss_operating_w += rc_w * q_lea_ahu_e * (t_eta_hr_in - t_amb_ahu)      # (85b)
    ahu_loss_rbl_operating_w = ahu_loss_operating_w if config.ahu_location == "conditioned" else 0.0  # (86)

    # ---------------------------------------------------------------- auxiliary (87)-(93)
    hr_type = (config.heat_recovery_data.hr_type if config.heat_recovery_data is not None else None)
    aux_hr_w = 0.0
    n_rot = None
    if hr_type in _mx.ROTARY and hr_flows:
        n_max = config.rotor_speed_max_rpm
        if modulating and abs(t_eta_hr_in - t_fp) > 1e-9 and eta_nom > 0.0:
            eta_req = max(0.0, (t_hr - t_fp) / (t_eta_hr_in - t_fp))               # (D.2)
            base = (hr_ratio["eta_hr_nom"] * hr_ratio["f_q"] * hr_ratio["f_v"]
                    if hr_ratio else eta_nom)
            d1 = _mx.TABLE_D1
            f_n_req = min(d1["C3"] - 1e-6, eta_req / base) if base > 0.0 else 0.0   # (D.4)
            n_rot = n_max * min(1.0, max(0.0, (-(f_n_req - d1["C3"]) / d1["C4"]) ** (1.0 / d1["e1"])
                                         - d1["C5"]))                                # (D.3)
        else:
            n_rot = n_max
        aux_hr_w = config.rotor_drive_power_max_w * n_rot / n_max                   # (88a)
    elif hr_type == "PUMP_CIRC" and hr_flows:
        load = max(config.heat_recovery_part_load_min,
                   abs(hr_power_operating_w) / max(config.heat_recovery_power_max_w, 1e-9))
        aux_hr_w = (q_sup_ahu_m3_h * f_oda * config.pump_specific_power_max_wh_m3
                    * load ** 2.5)                                                  # (88b)
    dp_hr = config.heat_recovery_design_pressure_drop_pa
    dp_s = (config.supply_fan_design_pressure_pa
            or config.supply_fan_specific_power_w_per_m3_s * config.supply_fan_nominal_efficiency)
    dp_e = (config.extract_fan_design_pressure_pa
            or config.extract_fan_specific_power_w_per_m3_s * config.extract_fan_nominal_efficiency)
    fan_hr_w = (fan_electric_power_w * dp_hr / (dp_s + dp_e)
                if (dp_hr and (dp_s + dp_e) > 0.0) else 0.0)                         # (89)
    p_ctrl = config.controls_power_w
    if p_ctrl <= 0.0 and config.controls_fan_power_fraction:
        p_ctrl = config.controls_fan_power_fraction * fan_electric_power_w          # NA.9
    ctrl_w = p_ctrl * config.controls_operation_factor                              # (91)
    hu_aux_w = 0.0
    if humid and str(config.humidifier_type).upper() != "STEAM" and config.humidifier_pump_specific_power_wh_m3 > 0.0:
        q_hu_des = config.humidifier_design_flow_m3_h or inputs.required_supply_flow_m3_h
        mw = humidification_water_kg_h / op if op > 0.0 else 0.0
        mw_des = config.humidifier_design_water_kg_h
        if config.humidifier_control == "no_ctrl":
            f_pl_hu = 1.0                                                           # (93a)
        else:
            ratio = (mw / mw_des) if mw_des else (1.0 if mw > 0.0 else 0.0)
            f_pl_hu = ratio if config.humidifier_control == "on_off" else ratio ** 2.5  # (93b)/(93c)
        hu_aux_w = q_hu_des * config.humidifier_pump_specific_power_wh_m3 * min(f_pl_hu, 1.0) * op  # (92b)
    aux_total_w = aux_hr_w * op + frost_preheat_operating_w * op + ctrl_w            # (87)

    return AHUStepOutputs(
        requested_supply_temperature_c=requested_t_sup,
        actual_supply_temperature_c=t_zone_source,
        actual_supply_flow_m3_h=op * q_delivered_m3_h,
        actual_extract_flow_m3_h=op * q_eta_zv_m3_h,
        required_heating_coil_power_w=required_heating_operating_w * op,
        actual_heating_coil_power_w=actual_heating_operating_w * op,
        required_cooling_coil_power_w=required_cooling_operating_w * op,
        actual_cooling_coil_power_w=actual_cooling_operating_w * op,
        fan_electric_power_w=fan_electric_power_w,
        heat_recovery_power_w=hr_power_operating_w * op,
        bypass_fraction=bypass_fraction,
        frost_protection_required=frost_protection_req,
        supply_humidity_ratio=x_actual,
        latent_cooling_coil_power_w=latent_cooling_operating_w * op,
        humidification_power_w=humidification_operating_w * op,
        humidification_water_kg_h=humidification_water_kg_h,
        heat_recovery_temperature_ratio=(hr_ratio["eta_hr"] if hr_ratio else eta_nom),
        moisture_recovery_efficiency=eta_xr,
        makeup_outdoor_air_flow_m3_h=op * q_makeup_m3_h,
        supply_fan_flow_m3_h=op * q_sup_ahu_m3_h,
        extract_fan_flow_m3_h=op * q_eta_ahu_m3_h,
        outdoor_air_fraction=f_oda * (1.0 - oda_mix),
        recirculation_power_w=recirc_operating_w * op,
        frost_preheat_power_w=frost_preheat_operating_w * op,
        duct_heat_loss_w=duct_loss_operating_w * op,
        duct_heat_loss_recoverable_w=duct_rbl_operating_w * op,
        ahu_outlet_temperature_c=t_dis_in,
        duct_outlet_temperature_c=t_dis_out,
        exhaust_temperature_c=t_exhaust,
        adiabatic_cooling_active=adiabatic_on,
        adiabatic_cooling_water_kg_h=ac_water_kg_h,
        cooling_coil_medium_temperature_required_c=t_c_medium_req,
        heating_coil_medium_temperature_required_c=t_h_medium_req,
        cooling_coil_temperature_c=cooling_coil_temperature,
        dehumidification_power_required_w=dhu_operating_w * op,
        ground_heat_transfer_w=ground_operating_w * op,
        ahu_heat_loss_w=ahu_loss_operating_w * op,
        ahu_heat_loss_recoverable_w=ahu_loss_rbl_operating_w * op,
        auxiliary_power_w=aux_total_w,
        auxiliary_heat_recovery_w=aux_hr_w * op,
        fan_power_heat_recovery_w=fan_hr_w,
        auxiliary_controls_w=ctrl_w,
        humidifier_auxiliary_w=hu_aux_w,
        rotor_speed_rpm=n_rot,
        air_density_kg_m3=rho,
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


AHU_MOISTURE_KEYS = ("supply_humidity_max_kg_kg", "supply_humidity_min_kg_kg", "heat_recovery_type")


def ahu_declares_moisture(component: dict) -> bool:
    """True when a ``mechanical_supply`` component needs the EN 16798-5-1 moisture path.

    Active when the component declares moisture data (``supply_humidity_max_kg_kg``,
    ``supply_humidity_min_kg_kg``, ``heat_recovery_type``) or an enabled cooling coil,
    which can condense whenever the coil surface falls below the dew point.
    """
    if not isinstance(component, dict):
        return False
    if str(component.get("ventilation_type", "")).lower() != "mechanical_supply":
        return False
    if any(component.get(k) is not None for k in AHU_MOISTURE_KEYS):
        return True
    if component.get("adiabatic_cooling") not in (None, False, "false", "False", 0):
        return True
    cc = component.get("cooling_coil_enabled", False)
    if isinstance(cc, str):
        return cc.strip().lower() in ("1", "true", "yes", "on")
    return bool(cc)


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

    # EN 16798-5-1 6.4.3.2.6 / Annex D: heat recovery type and design data (optional)
    hr_data = None
    if d.get("heat_recovery_type") is not None:
        hr_data = _mx.HeatRecoveryData(
            hr_type=str(d["heat_recovery_type"]),
            # the declared nominal efficiency is eta_hr;nom (Table B.6 only as fallback)
            eta_hr_nom=_opt_float(d.get("heat_recovery_nominal_ratio",
                                        d.get("sensible_heat_recovery_efficiency"))),
            v_hr_des_m_s=_opt_float(d.get("heat_recovery_design_velocity_m_s")),
            q_sup_hr_nom_m3_h=_opt_float(d.get("heat_recovery_nominal_flow_m3_h")),
            v_hr_nom_m_s=float(d.get("heat_recovery_nominal_velocity_m_s", _mx.V_HR_NOM_M_S)),
            C1=_opt_float(d.get("heat_recovery_C1")),
            C2=float(d.get("heat_recovery_C2", _mx.C2_DEFAULT)),
            rotor_speed_rpm=_opt_float(d.get("rotor_speed_rpm")),
            rotor_speed_max_rpm=float(d.get("rotor_speed_max_rpm", 20.0)),
            eta_xr_nom=_opt_float(d.get("moisture_recovery_nominal_efficiency")),
            annex_d_version=str(d.get("annex_d_version", "amended")),
        )
    coil_data = _mx.CoilData(
        CC1=float(d.get("cooling_coil_CC1", _mx.TABLE_B8["CC1"])),
        CC2=float(d.get("cooling_coil_CC2", _mx.TABLE_B8["CC2"])),
        CC3=float(d.get("cooling_coil_CC3", _mx.TABLE_B8["CC3"])),
        f_C_bp=float(d.get("cooling_coil_bypass_factor", _mx.TABLE_B8["f_C_bp"])),
    )

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
    elif ctrl_mode_str in ("none", "no_ctrl"):
        supply_ctrl = SupplyTemperatureControl(mode=SupplyTemperatureControlMode.NONE)
    elif ctrl_mode_str in ("oda_compensated", "oda_comp"):
        for k in ("supply_temperature_slope", "supply_temperature_offset_c"):
            if d.get(k) is None:
                raise KeyError(f"oda_compensated control requires {k!r}")
        supply_ctrl = SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.ODA_COMPENSATED,
            slope=float(d["supply_temperature_slope"]),
            offset_c=float(d["supply_temperature_offset_c"]),
            minimum_c=_opt_float(d.get("supply_temperature_minimum_c")),
            maximum_c=_opt_float(d.get("supply_temperature_maximum_c")),
        )
    elif ctrl_mode_str in ("load_compensated", "load_comp", "scheduled"):
        supply_ctrl = SupplyTemperatureControl(
            mode=(SupplyTemperatureControlMode.SCHEDULED if ctrl_mode_str == "scheduled"
                  else SupplyTemperatureControlMode.LOAD_COMPENSATED),
            minimum_c=_opt_float(d.get("supply_temperature_minimum_c")),
            maximum_c=_opt_float(d.get("supply_temperature_maximum_c")),
        )
    else:
        raise ValueError(
            f"Unsupported supply_temperature_mode: {ctrl_mode_str!r}. Supported: 'fixed', "
            "'outdoor_compensated', 'extract_compensated', 'none', 'oda_compensated', "
            "'load_compensated', 'scheduled'."
        )

    dist = _distribution_kwargs(d)
    return SensibleAHUConfig(
        sensible_heat_recovery_efficiency=float(d["sensible_heat_recovery_efficiency"]),
        supply_temperature_control=supply_ctrl,
        heat_recovery_control=str(d.get("heat_recovery_control", "modulating_bypass")),
        frost_control=str(d.get("frost_control", "exhaust_limit")),
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
            d.get("supply_fan_heat_fraction_to_air",
                  dist.pop("supply_fan_heat_fraction_to_air", 0.90))
        ),
        extract_fan_heat_fraction_to_air=float(
            d.get("extract_fan_heat_fraction_to_air",
                  dist.pop("extract_fan_heat_fraction_to_air", 0.90))
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
        heat_recovery_data=hr_data,
        coil_data=coil_data,
        supply_humidity_max_kg_kg=_opt_float(d.get("supply_humidity_max_kg_kg")),
        supply_humidity_min_kg_kg=_opt_float(d.get("supply_humidity_min_kg_kg")),
        humidifier_type=str(d.get("humidifier_type", "STEAM")).upper(),
        **{"frost_exhaust_limit_c": float(d.get("frost_exhaust_limit_c", -5.0)),
           **dist},
    )


def _distribution_kwargs(d: dict) -> dict:
    """EN 16798-5-1 distribution, recirculation, frost and adiabatic cooling keys.

    Leakage: ``duct_leakage_class`` (A, B, C, D, unknown; Table 10) or
    ``duct_leakage_factor_supply`` / ``duct_leakage_factor_extract``; ``duct_location``
    ("conditioned": supply leakage into the zone, default; "unconditioned": lost);
    ``ahu_leakage_class`` (L1, L2, L3; Table 11) or ``ahu_leakage_factor_supply`` /
    ``ahu_leakage_factor_extract``; ``supply_fan_location`` / ``extract_fan_location``
    (UP_HR, DOWN_HR; Table B.5) or ``hr_leakage_factor_supply`` /
    ``hr_leakage_factor_extract`` / ``outdoor_air_fraction_hr``.
    Duct heat losses [W/K]: ``duct_supply_h_unconditioned_w_k``, ``duct_supply_h_zone_w_k``,
    ``duct_extract_h_unconditioned_w_k``; ``duct_surrounding_temperature_c`` (default:
    outdoor air).  Recirculation: ``recirculation_control`` (none, fix, variable),
    ``outdoor_air_fraction_min`` (fix) and, on the component, ``outdoor_air_flow_m3_h``
    (variable).  Frost: ``frost_control`` (none, exhaust_limit, direct, indirect_bypass,
    indirect_preheat, indirect_recirc); ``frost_direct_ratio_minus15`` /
    ``frost_direct_ratio_minus7``, ``frost_direct_t_lim1_c`` / ``frost_direct_t_lim2_c``;
    ``building_use`` (residential, non_residential) sets ``frost_exhaust_limit_c`` from
    Table B.14 when not given.  Adiabatic cooling: ``adiabatic_cooling`` (bool),
    ``adiabatic_cooling_efficiency`` (default 0.9).
    Air density: ``altitude_m`` (Table 27).  Fans (72)-(75): ``fan_control`` (no_ctrl,
    direct, const_pres, min_pres), ``*_fan_pressure_curve`` / ``*_fan_efficiency_curve``
    (polynomial coefficients in q/q_nom, highest power first), ``*_fan_pressure_control_fraction``.
    Coils (61), (64), (68): ``cooling_coil_efficiency``, ``heating_coil_efficiency``.
    AHU casing (85)-(86): ``ahu_location``, ``ahu_supply_casing_au_w_k``,
    ``ahu_extract_casing_au_w_k`` (or ``ahu_casing_au_w_k``).  Auxiliary energy (87)-(93):
    ``rotor_drive_power_max_w``, ``rotor_speed_max_rpm``, ``pump_specific_power_max_wh_m3``,
    ``heat_recovery_power_max_w``, ``heat_recovery_part_load_min``,
    ``heat_recovery_design_pressure_drop_pa``, ``controls_power_w``,
    ``controls_operation_factor``, ``humidifier_pump_specific_power_wh_m3``,
    ``humidifier_design_flow_m3_h``, ``humidifier_design_water_kg_h``,
    ``humidifier_control``.  Ground preheating (Annex C): ``ground_exchanger`` (dict of
    :class:`GroundExchangerData` fields); outdoor air duct: ``duct_outdoor_h_w_k``.
    """
    out = {}
    nat = str(d.get("national_data", "IT")).strip().upper()
    if nat not in _mx.NATIONAL_DATA:
        raise ValueError(f"national_data must be one of {list(_mx.NATIONAL_DATA)}")
    nd = _mx.NATIONAL_DATA[nat]
    out["national_data"] = nat
    # building status -> default tightness classes (Table NA.2 / NA.3, B.3 / B.4)
    status = d.get("building_status")
    if status is not None:
        st = str(status).strip().lower()
        if st not in ("new", "existing"):
            raise ValueError("building_status must be 'new' or 'existing'")
        d = {"duct_leakage_class": nd["duct_leakage_class"][st],
             "ahu_leakage_class": nd["ahu_leakage_class"][st], **d}
    # heat recovery design pressure drop (Table NA.6 / B.6), used in (89)
    hrt = d.get("heat_recovery_type")
    if d.get("heat_recovery_design_pressure_drop_pa") is None and hrt is not None:
        dp = nd["hr_design_pressure_drop_pa"].get(str(hrt).upper())
        if dp is not None:
            out["heat_recovery_design_pressure_drop_pa"] = dp
    # fan motor location -> recovered ratio of the fan power f_fan;rd (Table NA.12 / B.12)
    fml = d.get("fan_motor_location")
    if fml is not None:
        key = str(fml).strip().upper()
        if key not in nd["fan_heat_recovery_factor"]:
            raise ValueError("fan_motor_location must be 'IN_AIR' or 'OUTS_AIR'")
        f_rd = nd["fan_heat_recovery_factor"][key]
        out.setdefault("supply_fan_heat_fraction_to_air", f_rd)
        out.setdefault("extract_fan_heat_fraction_to_air", f_rd)
    # system type -> controlled share of the design pressure (Table NA.13 / B.13)
    syst = d.get("system_type")
    if syst is not None:
        key = str(syst).strip().lower()
        if key not in nd["pressure_control_fraction"]:
            raise ValueError("system_type must be 'ventilation' or 'air_conditioning'")
        fs, fe = nd["pressure_control_fraction"][key]
        out["supply_fan_pressure_control_fraction"] = fs
        out["extract_fan_pressure_control_fraction"] = fe
    # controls (Table NA.9): 5 % of the fan power when no control power is given
    if d.get("controls_power_w") is None and nd["controls_fan_power_fraction"]:
        out["controls_fan_power_fraction"] = nd["controls_fan_power_fraction"]
    # humidifier pump (Table NA.10 / B.10)
    if d.get("humidifier_pump_specific_power_wh_m3") is None:
        hp = nd["humidifier_pump_wh_m3"].get(
            (str(d.get("humidifier_type", "STEAM")).upper(),
             str(d.get("humidifier_control", "on_off")).lower()))
        if hp is not None:
            out["humidifier_pump_specific_power_wh_m3"] = hp
    cls = d.get("duct_leakage_class")
    if cls is not None:
        key = str(cls).strip().upper()
        if key not in _mx.TABLE_10_DUCT_LEAKAGE:
            raise ValueError(f"duct_leakage_class must be one of {list(_mx.TABLE_10_DUCT_LEAKAGE)}")
        out["duct_leakage_factor_supply"] = out["duct_leakage_factor_extract"] = \
            _mx.TABLE_10_DUCT_LEAKAGE[key]
    cls = d.get("ahu_leakage_class")
    if cls is not None:
        key = str(cls).strip().upper()
        if key not in _mx.TABLE_11_AHU_LEAKAGE:
            raise ValueError(f"ahu_leakage_class must be one of {list(_mx.TABLE_11_AHU_LEAKAGE)}")
        out["ahu_leakage_factor_supply"] = out["ahu_leakage_factor_extract"] = \
            _mx.TABLE_11_AHU_LEAKAGE[key]
    for k in ("supply_fan_location", "extract_fan_location"):
        if d.get(k) is not None:
            out[k] = str(d[k]).strip().upper()
    if d.get("supply_fan_location") is not None or d.get("extract_fan_location") is not None:
        fs, fe, fo = _mx.hr_leakage_factors(d.get("heat_recovery_type"),
                                            d.get("supply_fan_location", "UP_HR"),
                                            d.get("extract_fan_location", "UP_HR"))
        out.update(hr_leakage_factor_supply=fs, hr_leakage_factor_extract=fe,
                   outdoor_air_fraction_hr=fo)
    for k in ("duct_leakage_factor_supply", "duct_leakage_factor_extract",
              "ahu_leakage_factor_supply", "ahu_leakage_factor_extract",
              "hr_leakage_factor_supply", "hr_leakage_factor_extract", "outdoor_air_fraction_hr",
              "duct_supply_h_unconditioned_w_k", "duct_supply_h_zone_w_k",
              "duct_extract_h_unconditioned_w_k", "frost_direct_t_lim1_c",
              "frost_direct_t_lim2_c", "adiabatic_cooling_efficiency"):
        if d.get(k) is not None:
            out[k] = float(d[k])
    for k in ("duct_surrounding_temperature_c", "outdoor_air_fraction_min",
              "frost_direct_ratio_minus15", "frost_direct_ratio_minus7"):
        if d.get(k) is not None:
            out[k] = float(d[k])
    loc = str(d.get("duct_location", "conditioned")).strip().lower()
    if loc not in ("conditioned", "unconditioned"):
        raise ValueError("duct_location must be 'conditioned' or 'unconditioned'")
    out["duct_leakage_to_zone"] = loc == "conditioned"
    if d.get("recirculation_control") is not None:
        out["recirculation_control"] = str(d["recirculation_control"]).strip().lower()
    if d.get("adiabatic_cooling") is not None:
        out["adiabatic_cooling"] = _parse_bool(d["adiabatic_cooling"], key="adiabatic_cooling")
    # Table 27 / fans / coils / casing / auxiliary energy / ground preheating
    for k in ("altitude_m", "supply_fan_pressure_control_fraction",
              "extract_fan_pressure_control_fraction", "cooling_coil_efficiency",
              "heating_coil_efficiency", "ahu_supply_casing_au_w_k", "ahu_extract_casing_au_w_k",
              "rotor_drive_power_max_w", "rotor_speed_max_rpm", "pump_specific_power_max_wh_m3",
              "heat_recovery_power_max_w", "heat_recovery_part_load_min", "controls_power_w",
              "controls_operation_factor", "humidifier_pump_specific_power_wh_m3",
              "duct_outdoor_h_w_k"):
        if d.get(k) is not None:
            out[k] = float(d[k])
    for k in ("heat_recovery_design_pressure_drop_pa", "humidifier_design_flow_m3_h",
              "humidifier_design_water_kg_h"):
        if d.get(k) is not None:
            out[k] = float(d[k])
    for k in ("fan_control", "ahu_location", "humidifier_control"):
        if d.get(k) is not None:
            out[k] = str(d[k]).strip().lower()
    for k in ("supply_fan_pressure_curve", "extract_fan_pressure_curve",
              "supply_fan_efficiency_curve", "extract_fan_efficiency_curve"):
        if d.get(k) is not None:
            out[k] = tuple(float(c) for c in d[k])
    if d.get("ahu_casing_au_w_k") is not None:          # same (A U) for both sides
        out.setdefault("ahu_supply_casing_au_w_k", float(d["ahu_casing_au_w_k"]))
        out.setdefault("ahu_extract_casing_au_w_k", float(d["ahu_casing_au_w_k"]))
    g = d.get("ground_exchanger")
    if g:
        if not isinstance(g, dict):
            raise ValueError("ground_exchanger must be a dict (Annex C data)")
        out["ground_exchanger"] = _mx.GroundExchangerData(**g)
    fc = str(d.get("frost_control", "exhaust_limit")).strip().lower()
    if (d.get("frost_exhaust_limit_c") is None and d.get("building_use") is not None
            and fc.startswith(("indirect", "exhaust"))):
        lim = _mx.frost_exhaust_limit(d["building_use"], d.get("heat_recovery_type"))
        if lim is not None:
            out["frost_exhaust_limit_c"] = lim
    return out


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
    h_w_k = outputs.air_density_kg_m3 * _CP_AIR_J_KG_K * outputs.actual_supply_flow_m3_h / 3600.0
    return VentilationStream(
        name=name,
        heat_transfer_coefficient_w_k=h_w_k,
        source_temperature_c=outputs.actual_supply_temperature_c,
        category="supply",
    )
