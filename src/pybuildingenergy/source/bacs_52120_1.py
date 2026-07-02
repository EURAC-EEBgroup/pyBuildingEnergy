"""Building automation and control factors for EN 15232-1/EN ISO 52120-1.

The standard workflow is represented as service-specific control factors. The
defaults are deliberately editable: national annexes, detailed controls and
building-use classes can replace them without changing the engine code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


DEFAULT_BACS_FACTORS: dict[str, dict[str, float]] = {
    "A": {
        "heating": 0.81,
        "cooling": 0.85,
        "dhw": 0.95,
        "lighting": 0.80,
        "ventilation": 0.70,
        "auxiliary": 0.90,
    },
    "B": {
        "heating": 0.88,
        "cooling": 0.93,
        "dhw": 0.98,
        "lighting": 0.90,
        "ventilation": 0.85,
        "auxiliary": 0.95,
    },
    "C": {
        "heating": 1.00,
        "cooling": 1.00,
        "dhw": 1.00,
        "lighting": 1.00,
        "ventilation": 1.00,
        "auxiliary": 1.00,
    },
    "D": {
        "heating": 1.10,
        "cooling": 1.10,
        "dhw": 1.02,
        "lighting": 1.10,
        "ventilation": 1.10,
        "auxiliary": 1.05,
    },
}


SERVICE_ALIASES: dict[str, list[str]] = {
    "heating": ["Q_H_kWh", "Q_H_after_solar_kWh", "E_heating_kWh"],
    "cooling": ["Q_C_kWh", "E_cooling_kWh"],
    "dhw": ["Q_W_kWh", "Q_W_after_solar_kWh", "E_dhw_kWh"],
    "lighting": ["E_L_total_kWh", "E_lighting_kWh"],
    "ventilation": ["E_vent_fan_kWh", "E_ventilation_kWh"],
    "auxiliary": ["E_aux_kWh", "Aux_electricity_kWh"],
}


@dataclass
class BACSSimulationResult:
    """Container returned by :class:`BACSControlFactorCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class BACSControlFactorCalculator:
    """Apply service-specific BACS factors to time-series or annual balances."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> BACSSimulationResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")
        results = self._simulate(data.copy())
        return BACSSimulationResult(
            timeseries=results,
            summary=self._summarize(data, results),
            inputs=dict(self.input_data),
        )

    def run_annual(self, values: dict[str, Any]) -> BACSSimulationResult:
        return self.run_timeseries(pd.DataFrame([values]))

    def factor_for(self, service: str) -> float:
        return float(self.factors.get(service, 1.0))

    def _load_options(self) -> None:
        cfg = self.input_data
        self.bacs_class = str(cfg.get("bacs_class", cfg.get("class", "C"))).upper()
        default_factors = DEFAULT_BACS_FACTORS.get(self.bacs_class, DEFAULT_BACS_FACTORS["C"])
        self.factors = {**default_factors, **dict(cfg.get("service_factors", {}))}
        self.aliases = {
            **SERVICE_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("service_aliases", {})).items()},
        }

    def _simulate(self, out: pd.DataFrame) -> pd.DataFrame:
        for service, aliases in self.aliases.items():
            factor = self.factor_for(service)
            for col in aliases:
                if col not in out.columns:
                    continue
                adjusted = f"{col}_bacs"
                savings = f"{col}_bacs_saving"
                out.loc[:, adjusted] = out[col].astype(float).clip(lower=0.0) * factor
                out.loc[:, savings] = out[col].astype(float).clip(lower=0.0) - out[adjusted]
        return out

    def _summarize(self, original: pd.DataFrame, results: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {"BACS_class": self.bacs_class}
        for service, aliases in self.aliases.items():
            for col in aliases:
                if col not in original.columns or f"{col}_bacs" not in results.columns:
                    continue
                before = float(original[col].astype(float).clip(lower=0.0).sum())
                after = float(results[f"{col}_bacs"].sum())
                out[f"{service}_{col}_before_kWh"] = before
                out[f"{service}_{col}_after_kWh"] = after
                out[f"{service}_{col}_saving_kWh"] = before - after
                break
            out[f"{service}_factor"] = self.factor_for(service)
        return out
