"""EN 15459-1 style economic calculation for EPB renovation options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_EPS = 1e-12


@dataclass
class EconomicSimulationResult:
    """Container returned by :class:`CostOptimalityCalculator`."""

    summary: dict[str, float]
    inputs: dict[str, Any]


class CostOptimalityCalculator:
    """Global cost and annualised cost calculation.

    Components are dictionaries with at least ``investment_EUR``. Optional keys:
    ``annual_maintenance_EUR``, ``lifetime_years``, ``replacement_cost_EUR`` and
    ``residual_value_EUR``. Energy flows are annual kWh by carrier.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def calculate(
        self,
        *,
        components: list[dict[str, Any]] | None = None,
        annual_energy: dict[str, float] | None = None,
        annual_exports: dict[str, float] | None = None,
    ) -> EconomicSimulationResult:
        components = list(components or self.components)
        annual_energy = dict(annual_energy or self.annual_energy)
        annual_exports = dict(annual_exports or self.annual_exports)

        investment = sum(_float(c.get("investment_EUR"), 0.0) for c in components)
        maintenance = sum(_float(c.get("annual_maintenance_EUR"), 0.0) for c in components)
        replacement_pv = 0.0
        residual_pv = 0.0
        for component in components:
            replacement_pv += self._replacement_present_value(component)
            residual_pv += self._residual_present_value(component)

        energy_cost = sum(
            max(float(kwh), 0.0) * self.energy_prices.get(str(carrier), 0.0)
            for carrier, kwh in annual_energy.items()
        )
        export_value = sum(
            max(float(kwh), 0.0) * self.export_prices.get(str(carrier), 0.0)
            for carrier, kwh in annual_exports.items()
        )
        annual_running = energy_cost + maintenance - export_value
        pv_running = annual_running * self.present_value_factor
        global_cost = investment + replacement_pv + pv_running - residual_pv
        annuity_factor = 1.0 / self.present_value_factor if self.present_value_factor > _EPS else 0.0
        annualized_global_cost = global_cost * annuity_factor

        summary = {
            "Calculation_period_years": float(self.calculation_period_years),
            "Real_discount_rate": self.real_discount_rate,
            "Investment_EUR": investment,
            "Annual_maintenance_EUR": maintenance,
            "Annual_energy_cost_EUR": energy_cost,
            "Annual_export_value_EUR": export_value,
            "Annual_running_cost_EUR": annual_running,
            "Present_value_factor_annual": self.present_value_factor,
            "Replacement_present_value_EUR": replacement_pv,
            "Residual_present_value_EUR": residual_pv,
            "Running_cost_present_value_EUR": pv_running,
            "Global_cost_EUR": global_cost,
            "Annualized_global_cost_EUR": annualized_global_cost,
        }
        if self.reference_global_cost_EUR is not None:
            summary["Delta_global_cost_EUR"] = global_cost - self.reference_global_cost_EUR
        if self.floor_area_m2 > _EPS:
            summary["Global_cost_EUR_m2"] = global_cost / self.floor_area_m2
            summary["Annualized_global_cost_EUR_m2a"] = annualized_global_cost / self.floor_area_m2
        return EconomicSimulationResult(summary=summary, inputs=dict(self.input_data))

    def _load_options(self) -> None:
        cfg = self.input_data
        self.calculation_period_years = int(
            max(round(float(cfg.get("calculation_period_years", 30))), 1)
        )
        self.real_discount_rate = float(cfg.get("real_discount_rate", 0.03))
        self.energy_prices = {
            "natural_gas": 0.12,
            "gas": 0.12,
            "electricity": 0.30,
            "biomass": 0.08,
            "biomass_pellets": 0.08,
            "district_heat": 0.11,
            **dict(cfg.get("energy_prices", {})),
        }
        self.export_prices = {
            "electricity": 0.06,
            **dict(cfg.get("export_prices", {})),
        }
        self.components = list(cfg.get("components", []))
        self.annual_energy = dict(cfg.get("annual_energy", {}))
        self.annual_exports = dict(cfg.get("annual_exports", {}))
        self.floor_area_m2 = max(float(cfg.get("floor_area_m2", cfg.get("area_m2", 0.0))), 0.0)
        reference = cfg.get("reference_global_cost_EUR")
        self.reference_global_cost_EUR = None if reference is None else float(reference)
        self.present_value_factor = _present_value_factor(
            self.real_discount_rate,
            self.calculation_period_years,
        )

    def _replacement_present_value(self, component: dict[str, Any]) -> float:
        lifetime = int(round(_float(component.get("lifetime_years"), self.calculation_period_years)))
        if lifetime <= 0 or lifetime >= self.calculation_period_years:
            return 0.0
        replacement = _float(
            component.get(
                "replacement_cost_EUR",
                component.get("replacement_EUR", component.get("investment_EUR", 0.0)),
            ),
            0.0,
        )
        total = 0.0
        year = lifetime
        while year < self.calculation_period_years:
            total += replacement / ((1.0 + self.real_discount_rate) ** year)
            year += lifetime
        return total

    def _residual_present_value(self, component: dict[str, Any]) -> float:
        explicit = component.get("residual_value_EUR", component.get("salvage_value_EUR"))
        if explicit is not None:
            return _float(explicit, 0.0) / (
                (1.0 + self.real_discount_rate) ** self.calculation_period_years
            )

        lifetime = int(round(_float(component.get("lifetime_years"), 0.0)))
        if lifetime <= 0:
            return 0.0
        investment = _float(component.get("investment_EUR"), 0.0)
        age_at_end = self.calculation_period_years % lifetime
        remaining_fraction = (lifetime - age_at_end) / lifetime if age_at_end else 0.0
        residual = investment * max(remaining_fraction, 0.0)
        return residual / ((1.0 + self.real_discount_rate) ** self.calculation_period_years)


def _present_value_factor(rate: float, years: int) -> float:
    if abs(rate) <= _EPS:
        return float(years)
    return (1.0 - (1.0 + rate) ** (-years)) / rate


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out
