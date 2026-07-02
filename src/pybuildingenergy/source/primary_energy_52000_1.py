"""EN ISO 52000-1 style delivered/exported energy accounting.

The calculator keeps the primary-energy balance explicit at carrier level. It
does not bake in a national annex: factors for delivered and exported energy
are input data, with conservative defaults for the STREPIN examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


DEFAULT_PRIMARY_FACTORS: dict[str, dict[str, float]] = {
    "natural_gas": {"nonrenewable": 1.05, "renewable": 0.00},
    "gas": {"nonrenewable": 1.05, "renewable": 0.00},
    "electricity": {"nonrenewable": 2.17, "renewable": 0.47},
    "district_heat": {"nonrenewable": 0.80, "renewable": 0.20},
    "district_cooling": {"nonrenewable": 0.70, "renewable": 0.20},
    "biomass": {"nonrenewable": 0.20, "renewable": 1.00},
    "biomass_pellets": {"nonrenewable": 0.20, "renewable": 1.00},
}


DEFAULT_CARRIER_ALIASES: dict[str, list[str]] = {
    "natural_gas": [
        "E_delivered_natural_gas_kWh",
        "E_gas_delivered_kWh",
        "Gas_final_kWh",
        "Gas_final_kWh_a",
        "EHW_fuel_in_kWh",
    ],
    "electricity": [
        "E_delivered_electricity_kWh",
        "E_grid_kWh",
        "E_grid_after_PV_kWh",
        "Grid_electricity_kWh",
        "Grid_electricity_kWh_a",
        "E_electricity_delivered_kWh",
    ],
    "district_heat": [
        "E_delivered_district_heat_kWh",
        "E_district_heat_delivered_kWh",
    ],
    "district_cooling": [
        "E_delivered_district_cooling_kWh",
        "E_district_cooling_delivered_kWh",
    ],
    "biomass": [
        "E_delivered_biomass_kWh",
        "E_biomass_delivered_kWh",
        "Biomass_final_kWh",
        "Biomass_final_kWh_a",
    ],
}


DEFAULT_EXPORT_ALIASES: dict[str, list[str]] = {
    "electricity": [
        "E_exported_electricity_kWh",
        "E_PV_export_kWh",
        "PV_export_kWh",
        "PV_export_kWh_a",
    ],
}


@dataclass
class PrimaryEnergyAccountingResult:
    """Container returned by :class:`PrimaryEnergyAccountingCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class PrimaryEnergyAccountingCalculator:
    """Delivered/exported energy balance for EN ISO 52000-1 workflows."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> PrimaryEnergyAccountingResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return PrimaryEnergyAccountingResult(
            timeseries=results,
            summary=self._summarize(results),
            inputs=dict(self.input_data),
        )

    def run_annual(self, values: dict[str, Any]) -> PrimaryEnergyAccountingResult:
        """Run the same accounting on a one-row annual balance."""

        return self.run_timeseries(pd.DataFrame([values]))

    def _load_options(self) -> None:
        cfg = self.input_data
        self.carriers = list(cfg.get("carriers", DEFAULT_CARRIER_ALIASES.keys()))
        self.export_credit_method = str(cfg.get("export_credit_method", "symmetric")).lower()
        if self.export_credit_method not in {"symmetric", "none"}:
            raise ValueError("export_credit_method must be 'symmetric' or 'none'.")

        self.primary_factors = _normalise_factor_map(
            cfg.get("primary_energy_factors", cfg.get("factors", {}))
        )
        self.export_factors = _normalise_factor_map(
            cfg.get("export_primary_energy_factors", cfg.get("export_factors", {}))
        )
        self.carrier_aliases = {
            **DEFAULT_CARRIER_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("carrier_aliases", {})).items()},
        }
        self.export_aliases = {
            **DEFAULT_EXPORT_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("export_aliases", {})).items()},
        }

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        out = pd.DataFrame(index=data.index)
        for carrier in self.carriers:
            out.loc[:, f"E_delivered_{carrier}_kWh"] = _series_from_aliases(
                data, self.carrier_aliases.get(carrier, []), default=0.0
            ).astype(float).fillna(0.0).clip(lower=0.0)
            out.loc[:, f"E_exported_{carrier}_kWh"] = _series_from_aliases(
                data, self.export_aliases.get(carrier, []), default=0.0
            ).astype(float).fillna(0.0).clip(lower=0.0)
        return out

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy()
        total_delivered = pd.Series(0.0, index=out.index)
        total_exported = pd.Series(0.0, index=out.index)
        total_pe_nren = pd.Series(0.0, index=out.index)
        total_pe_ren = pd.Series(0.0, index=out.index)
        total_export_credit_nren = pd.Series(0.0, index=out.index)
        total_export_credit_ren = pd.Series(0.0, index=out.index)

        for carrier in self.carriers:
            delivered = out[f"E_delivered_{carrier}_kWh"]
            exported = out[f"E_exported_{carrier}_kWh"]
            factors = self._factors_for(carrier, exported=False)
            export_factors = self._factors_for(carrier, exported=True)

            pe_nren = delivered * factors["nonrenewable"]
            pe_ren = delivered * factors["renewable"]
            if self.export_credit_method == "none":
                credit_nren = pd.Series(0.0, index=out.index)
                credit_ren = pd.Series(0.0, index=out.index)
            else:
                credit_nren = exported * export_factors["nonrenewable"]
                credit_ren = exported * export_factors["renewable"]

            out.loc[:, f"PE_delivered_{carrier}_nonren_kWh"] = pe_nren
            out.loc[:, f"PE_delivered_{carrier}_ren_kWh"] = pe_ren
            out.loc[:, f"PE_export_credit_{carrier}_nonren_kWh"] = credit_nren
            out.loc[:, f"PE_export_credit_{carrier}_ren_kWh"] = credit_ren

            total_delivered += delivered
            total_exported += exported
            total_pe_nren += pe_nren
            total_pe_ren += pe_ren
            total_export_credit_nren += credit_nren
            total_export_credit_ren += credit_ren

        out.loc[:, "E_delivered_total_kWh"] = total_delivered
        out.loc[:, "E_exported_total_kWh"] = total_exported
        out.loc[:, "PE_delivered_nonren_kWh"] = total_pe_nren
        out.loc[:, "PE_delivered_ren_kWh"] = total_pe_ren
        out.loc[:, "PE_export_credit_nonren_kWh"] = total_export_credit_nren
        out.loc[:, "PE_export_credit_ren_kWh"] = total_export_credit_ren
        out.loc[:, "PE_net_nonren_kWh"] = (
            total_pe_nren - total_export_credit_nren
        ).clip(lower=0.0)
        out.loc[:, "PE_net_ren_kWh"] = (total_pe_ren - total_export_credit_ren).clip(
            lower=0.0
        )
        out.loc[:, "PE_net_total_kWh"] = out["PE_net_nonren_kWh"] + out["PE_net_ren_kWh"]
        return out

    def _factors_for(self, carrier: str, *, exported: bool) -> dict[str, float]:
        defaults = DEFAULT_PRIMARY_FACTORS.get(carrier, {"nonrenewable": 1.0, "renewable": 0.0})
        factors = self.primary_factors.get(carrier, defaults)
        if exported:
            factors = self.export_factors.get(carrier, factors)
        nren = max(float(factors.get("nonrenewable", factors.get("nren", 0.0))), 0.0)
        ren = max(float(factors.get("renewable", factors.get("ren", 0.0))), 0.0)
        return {"nonrenewable": nren, "renewable": ren}

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        out = {
            "E_delivered_total_kWh": s("E_delivered_total_kWh"),
            "E_exported_total_kWh": s("E_exported_total_kWh"),
            "PE_delivered_nonren_kWh": s("PE_delivered_nonren_kWh"),
            "PE_delivered_ren_kWh": s("PE_delivered_ren_kWh"),
            "PE_export_credit_nonren_kWh": s("PE_export_credit_nonren_kWh"),
            "PE_export_credit_ren_kWh": s("PE_export_credit_ren_kWh"),
            "PE_net_nonren_kWh": s("PE_net_nonren_kWh"),
            "PE_net_ren_kWh": s("PE_net_ren_kWh"),
            "PE_net_total_kWh": s("PE_net_total_kWh"),
        }
        for carrier in self.carriers:
            delivered = s(f"E_delivered_{carrier}_kWh")
            exported = s(f"E_exported_{carrier}_kWh")
            out[f"E_delivered_{carrier}_kWh"] = delivered
            out[f"E_exported_{carrier}_kWh"] = exported
            out[f"PE_net_{carrier}_nonren_kWh"] = (
                s(f"PE_delivered_{carrier}_nonren_kWh")
                - s(f"PE_export_credit_{carrier}_nonren_kWh")
            )
        if out["E_delivered_total_kWh"] > _KWH_EPS:
            out["PEF_net_nonren"] = out["PE_net_nonren_kWh"] / out["E_delivered_total_kWh"]
            out["PEF_net_total"] = out["PE_net_total_kWh"] / out["E_delivered_total_kWh"]
        else:
            out["PEF_net_nonren"] = float("nan")
            out["PEF_net_total"] = float("nan")
        return out


def _normalise_factor_map(raw: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for carrier, values in dict(raw or {}).items():
        if isinstance(values, (int, float, np.number)):
            out[str(carrier)] = {"nonrenewable": float(values), "renewable": 0.0}
        else:
            out[str(carrier)] = {
                "nonrenewable": float(
                    values.get("nonrenewable", values.get("nren", values.get("total", 0.0)))
                ),
                "renewable": float(values.get("renewable", values.get("ren", 0.0))),
            }
    return out


def _series_from_aliases(
    df: pd.DataFrame, aliases: list[str], default: Any | None
) -> pd.Series | None:
    for name in aliases:
        if name in df.columns:
            return df[name]
    if default is None:
        return None
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)
