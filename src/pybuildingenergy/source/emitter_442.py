"""Emitter characteristic curves, including EN 442:2014 radiator ratings.

EN 442 is a product rating standard.  This module therefore models the
boundary between certified product data and the EN 15316 system calculation;
it does not attempt to reproduce the EN 442 laboratory test procedure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


_EPS = 1e-12


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


class EmitterCharacteristic(ABC):
    """Common interface used by EN 15316 emitter circuit calculations."""

    method: str
    product_reference: str | None

    @property
    @abstractmethod
    def nominal_power_kW(self) -> float:
        """Thermal output at the nominal excess temperature."""

    @property
    @abstractmethod
    def nominal_excess_temperature_K(self) -> float:
        """Mean water-to-room temperature difference at nominal output."""

    @property
    @abstractmethod
    def exponent_n(self) -> float:
        """Exponent of the thermal-output characteristic."""

    @property
    def maximum_operating_temperature_C(self) -> float | None:
        """Maximum declared water temperature, when available."""

        return None

    @abstractmethod
    def output_at_excess_temperature(
        self,
        excess_temperature_K: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        """Return thermal output in kW for a mean excess temperature."""

    @abstractmethod
    def required_excess_temperature(
        self,
        output_kW: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        """Return the mean excess temperature required for an output."""

    def relative_temperature_factor(
        self,
        output_kW: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        """Return required excess temperature divided by its nominal value."""

        return self.required_excess_temperature(
            output_kW, water_flow_ratio=water_flow_ratio
        ) / max(
            self.nominal_excess_temperature_K, _EPS
        )

    def pressure_drop_kPa(self, water_flow_kg_s: float) -> float | None:
        """Return product pressure drop when the rating contains such data."""

        return None


@dataclass(frozen=True)
class GenericEmitterCharacteristic(EmitterCharacteristic):
    """Generic EN 15316/custom power-law emitter characteristic.

    This class preserves the historical ``nominal_power`` + ``TB14`` pathway.
    It is intentionally not labelled as EN 442 product data.
    """

    rated_power_kW: float
    rated_excess_temperature_K: float
    rated_exponent_n: float
    product_reference: str | None = None
    method: str = "en15316_default"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rated_power_kW", _positive(self.rated_power_kW, "rated_power_kW")
        )
        object.__setattr__(
            self,
            "rated_excess_temperature_K",
            _positive(
                self.rated_excess_temperature_K,
                "rated_excess_temperature_K",
            ),
        )
        object.__setattr__(
            self,
            "rated_exponent_n",
            _positive(self.rated_exponent_n, "rated_exponent_n"),
        )

    @property
    def nominal_power_kW(self) -> float:
        return self.rated_power_kW

    @property
    def nominal_excess_temperature_K(self) -> float:
        return self.rated_excess_temperature_K

    @property
    def exponent_n(self) -> float:
        return self.rated_exponent_n

    def output_at_excess_temperature(
        self,
        excess_temperature_K: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        excess = max(float(excess_temperature_K), 0.0)
        if excess <= _EPS:
            return 0.0
        return self.nominal_power_kW * (
            excess / self.nominal_excess_temperature_K
        ) ** self.exponent_n

    def required_excess_temperature(
        self,
        output_kW: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        output = max(float(output_kW), 0.0)
        if output <= _EPS:
            return 0.0
        return self.nominal_excess_temperature_K * (
            output / self.nominal_power_kW
        ) ** (1.0 / self.exponent_n)


@dataclass(frozen=True)
class EN442RadiatorCharacteristic(EmitterCharacteristic):
    """EN 442-2:2014 characteristic for a rated radiator or convector.

    The core equation is EN 442-2:2014, 5.5.1.1::

        phi = phi_50 * (delta_T / 50) ** n

    Optional flow and pressure-drop terms support manufacturer declarations
    without changing the radiator default, for which the flow exponent is zero.
    """

    phi_50_kW: float
    characteristic_exponent_n: float
    phi_30_kW: float | None = None
    declared_maximum_operating_temperature_C: float | None = None
    water_flow_exponent: float = 0.0
    standard_water_flow_kg_s: float | None = None
    pressure_drop_at_standard_flow_kPa: float | None = None
    pressure_drop_exponent: float = 2.0
    output_consistency_tolerance: float = 0.02
    product_reference: str | None = None
    method: str = "en442"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "phi_50_kW", _positive(self.phi_50_kW, "phi_50_kW")
        )
        object.__setattr__(
            self,
            "characteristic_exponent_n",
            _positive(
                self.characteristic_exponent_n,
                "characteristic_exponent_n",
            ),
        )
        object.__setattr__(
            self,
            "water_flow_exponent",
            _nonnegative(self.water_flow_exponent, "water_flow_exponent"),
        )
        object.__setattr__(
            self,
            "pressure_drop_exponent",
            _positive(self.pressure_drop_exponent, "pressure_drop_exponent"),
        )
        tolerance = _nonnegative(
            self.output_consistency_tolerance,
            "output_consistency_tolerance",
        )
        if tolerance > 1.0:
            raise ValueError("output_consistency_tolerance must be <= 1.0.")
        object.__setattr__(self, "output_consistency_tolerance", tolerance)

        if self.declared_maximum_operating_temperature_C is not None:
            maximum = _positive(
                self.declared_maximum_operating_temperature_C,
                "maximum_operating_temperature_C",
            )
            if maximum >= 120.0:
                raise ValueError(
                    "EN 442:2014 applies to products operated below 120 degC; "
                    "maximum_operating_temperature_C must be < 120."
                )
            object.__setattr__(
                self, "declared_maximum_operating_temperature_C", maximum
            )

        if self.standard_water_flow_kg_s is not None:
            object.__setattr__(
                self,
                "standard_water_flow_kg_s",
                _positive(
                    self.standard_water_flow_kg_s,
                    "standard_water_flow_kg_s",
                ),
            )
        if self.water_flow_exponent > 0.0 and self.standard_water_flow_kg_s is None:
            raise ValueError(
                "standard_water_flow_kg_s is required when water_flow_exponent > 0."
            )

        if self.pressure_drop_at_standard_flow_kPa is not None:
            object.__setattr__(
                self,
                "pressure_drop_at_standard_flow_kPa",
                _positive(
                    self.pressure_drop_at_standard_flow_kPa,
                    "pressure_drop_at_standard_flow_kPa",
                ),
            )
            if self.standard_water_flow_kg_s is None:
                raise ValueError(
                    "standard_water_flow_kg_s is required with "
                    "pressure_drop_at_standard_flow_kPa."
                )

        if self.phi_30_kW is not None:
            declared = _positive(self.phi_30_kW, "phi_30_kW")
            object.__setattr__(self, "phi_30_kW", declared)
            expected = self.calculated_phi_30_kW
            relative_error = abs(declared - expected) / max(expected, _EPS)
            if relative_error > tolerance:
                raise ValueError(
                    "phi_30_kW is inconsistent with phi_50_kW and exponent_n: "
                    f"declared={declared:.6g}, calculated={expected:.6g}, "
                    f"relative_error={relative_error:.3%}."
                )

    @property
    def nominal_power_kW(self) -> float:
        return self.phi_50_kW

    @property
    def nominal_excess_temperature_K(self) -> float:
        return 50.0

    @property
    def exponent_n(self) -> float:
        return self.characteristic_exponent_n

    @property
    def maximum_operating_temperature_C(self) -> float | None:
        return self.declared_maximum_operating_temperature_C

    @property
    def calculated_phi_30_kW(self) -> float:
        return self.phi_50_kW * (30.0 / 50.0) ** self.exponent_n

    @property
    def coefficient_Km(self) -> float:
        return self.phi_50_kW / 50.0**self.exponent_n

    def _flow_factor(self, water_flow_ratio: float) -> float:
        ratio = _nonnegative(water_flow_ratio, "water_flow_ratio")
        if self.water_flow_exponent <= _EPS:
            return 1.0
        return ratio**self.water_flow_exponent

    def output_at_excess_temperature(
        self,
        excess_temperature_K: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        excess = max(float(excess_temperature_K), 0.0)
        if excess <= _EPS:
            return 0.0
        return (
            self.phi_50_kW
            * (excess / 50.0) ** self.exponent_n
            * self._flow_factor(water_flow_ratio)
        )

    def required_excess_temperature(
        self,
        output_kW: float,
        *,
        water_flow_ratio: float = 1.0,
    ) -> float:
        output = max(float(output_kW), 0.0)
        if output <= _EPS:
            return 0.0
        flow_factor = self._flow_factor(water_flow_ratio)
        if flow_factor <= _EPS:
            raise ValueError(
                "Positive emitter output requires a positive water flow ratio."
            )
        normalized_output = output / (
            self.phi_50_kW * flow_factor
        )
        return 50.0 * normalized_output ** (1.0 / self.exponent_n)

    def pressure_drop_kPa(self, water_flow_kg_s: float) -> float | None:
        if self.pressure_drop_at_standard_flow_kPa is None:
            return None
        flow = max(float(water_flow_kg_s), 0.0)
        return self.pressure_drop_at_standard_flow_kPa * (
            flow / max(float(self.standard_water_flow_kg_s), _EPS)
        ) ** self.pressure_drop_exponent

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EN442RadiatorCharacteristic":
        """Build a rating while accepting explicit W or kW catalogue fields."""

        cfg = dict(data or {})

        def power_kW(kwh_names: tuple[str, ...], watt_names: tuple[str, ...]) -> float | None:
            values_kW = [float(cfg[name]) for name in kwh_names if cfg.get(name) is not None]
            values_W = [float(cfg[name]) / 1000.0 for name in watt_names if cfg.get(name) is not None]
            values = values_kW + values_W
            if not values:
                return None
            reference = values[0]
            if any(abs(value - reference) > max(abs(reference), 1.0) * 1e-9 for value in values[1:]):
                raise ValueError("Conflicting W and kW emitter rating values were provided.")
            return reference

        phi_50 = power_kW(
            ("phi_50_kW", "phi50_kW"),
            ("phi_50_W", "phi50_W"),
        )
        if phi_50 is None:
            raise ValueError("emitter_rating must define phi_50_kW or phi_50_W.")
        phi_30 = power_kW(
            ("phi_30_kW", "phi30_kW"),
            ("phi_30_W", "phi30_W"),
        )
        exponent = cfg.get(
            "exponent_n",
            cfg.get("characteristic_exponent_n", cfg.get("n")),
        )
        if exponent is None:
            raise ValueError("emitter_rating must define exponent_n.")

        return cls(
            phi_50_kW=phi_50,
            characteristic_exponent_n=exponent,
            phi_30_kW=phi_30,
            declared_maximum_operating_temperature_C=cfg.get(
                "maximum_operating_temperature_C"
            ),
            water_flow_exponent=cfg.get("water_flow_exponent", 0.0),
            standard_water_flow_kg_s=cfg.get("standard_water_flow_kg_s"),
            pressure_drop_at_standard_flow_kPa=cfg.get(
                "pressure_drop_at_standard_flow_kPa"
            ),
            pressure_drop_exponent=cfg.get("pressure_drop_exponent", 2.0),
            output_consistency_tolerance=cfg.get(
                "output_consistency_tolerance", 0.02
            ),
            product_reference=cfg.get(
                "product_reference", cfg.get("catalogue_reference")
            ),
        )


def normalize_emitter_characteristic_method(value: Any) -> str:
    """Normalize public configuration aliases."""

    method = str(value or "en15316_default").strip().lower().replace("-", "_")
    if method in {"legacy", "simplified", "en15316", "en15316_default"}:
        return "en15316_default"
    if method in {"en442", "en_442", "en442_product", "product"}:
        return "en442"
    if method in {"custom", "custom_curve", "power_law"}:
        return "custom"
    raise ValueError(
        "emitter_calculation_method must be 'en15316_default', 'en442', or 'custom'."
    )
