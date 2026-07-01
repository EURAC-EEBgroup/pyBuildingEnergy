"""Biomass/solid-fuel generation defaults for EN 15316-4 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .combustion_15316_4_1 import (
    CombustionBoilerSimulationResult,
    CombustionBoilerSystemCalculator,
)


@dataclass
class BiomassBoilerSimulationResult:
    """Container returned by :class:`BiomassBoilerSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class BiomassBoilerSystemCalculator:
    """Biomass boiler wrapper around the combustion-generator boundary.

    The wrapper supplies typical solid-fuel defaults while reusing the generic
    combustion balance: useful heat output, generator fuel input, standby losses
    and auxiliaries are all reported at the same EN 15316 generation boundary.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        defaults = {
            "fuel": "biomass_pellets",
            "condensing": False,
            "full_load_efficiency": 0.86,
            "part_load_efficiency": 0.78,
            "minimum_part_load_ratio": 0.30,
            "standby_loss_kWh_per_h": 0.020,
            "auxiliary_power_kW": 0.040,
            "standby_auxiliary_power_kW": 0.006,
            "thermal_loss_room_fraction": 0.20,
        }
        self.input_data = {**defaults, **dict(input_data or {})}
        self._delegate = CombustionBoilerSystemCalculator(self.input_data)

    def run_timeseries(self, data: pd.DataFrame) -> BiomassBoilerSimulationResult:
        result: CombustionBoilerSimulationResult = self._delegate.run_timeseries(data)
        summary = dict(result.summary)
        summary["E_biomass_delivered_kWh"] = summary.get("EHW_fuel_in_kWh", 0.0)
        summary["eta_biomass_gen"] = summary.get("eta_HW_gen", 0.0)
        return BiomassBoilerSimulationResult(
            timeseries=result.timeseries,
            summary=summary,
            inputs=dict(self.input_data),
        )
