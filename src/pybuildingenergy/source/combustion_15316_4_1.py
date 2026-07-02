"""Combustion generation calculation for EN 15316-4-1 workflows.

This module implements a time-series combustion-generator boundary for space
heating and domestic hot water. It is intended to be used after the emission,
distribution and storage modules have calculated the thermal output required
from generation.

The calculator is standards-aware rather than national-annex-specific: product
or project data are supplied through efficiencies, standby losses, modulation
limits and auxiliary powers. The output contract follows the other EN 15316
modules in this package: a detailed time series, an annual/period summary and
the normalized inputs used in the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


@dataclass
class CombustionBoilerSimulationResult:
    """Container returned by :class:`CombustionBoilerSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class CombustionBoilerSystemCalculator:
    """EN 15316-4-1 style combustion generator calculator.

    The implementation uses the generator efficiency method commonly needed in
    EN 15316-4-1 workflows: the useful output requested from the boiler is
    divided by an operating efficiency corrected for part-load and return
    temperature, then generator losses and auxiliaries are reported at the same
    boundary. It supports heating and DHW as one shared boiler.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> CombustionBoilerSimulationResult:
        """Run the combustion-generator calculation."""

        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        summary = self._summarize(results)
        return CombustionBoilerSimulationResult(
            timeseries=results,
            summary=summary,
            inputs=dict(self.input_data),
        )

    def _load_options(self) -> None:
        cfg = self.input_data
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.demand_unit = str(cfg.get("demand_unit", "kWh")).lower()
        if self.demand_unit not in {"wh", "kwh"}:
            raise ValueError("demand_unit must be 'Wh' or 'kWh'.")

        self.fuel = str(cfg.get("fuel", cfg.get("energy_carrier", "natural_gas")))
        self.condensing = bool(cfg.get("condensing", True))
        self.nominal_power_kW = max(
            float(cfg.get("nominal_power_kW", cfg.get("rated_power_kW", 0.0))),
            0.0,
        )
        self.minimum_part_load_ratio = _fraction(
            cfg.get("minimum_part_load_ratio", cfg.get("min_modulation_ratio", 0.0)),
            "minimum_part_load_ratio",
        )
        self.heating_enabled = bool(cfg.get("heating_enabled", True))
        self.dhw_enabled = bool(cfg.get("dhw_enabled", True))

        eta_nom = _fraction(
            cfg.get(
                "efficiency",
                cfg.get("eta_gen_nominal", cfg.get("seasonal_efficiency", 0.93)),
            ),
            "efficiency",
        )
        self.full_load_efficiency = _fraction(
            cfg.get("full_load_efficiency", cfg.get("eta_gen_full_load", eta_nom)),
            "full_load_efficiency",
        )
        default_part = min(1.10, eta_nom + 0.03) if self.condensing else max(0.50, eta_nom - 0.02)
        self.part_load_efficiency = _fraction(
            cfg.get("part_load_efficiency", cfg.get("eta_gen_part_load", default_part)),
            "part_load_efficiency",
        )
        self.efficiency_min = _fraction(
            cfg.get("efficiency_min", 0.50),
            "efficiency_min",
        )
        self.efficiency_max = float(cfg.get("efficiency_max", 1.10 if self.condensing else 1.0))
        if self.efficiency_max <= 0.0:
            raise ValueError("efficiency_max must be positive.")

        self.part_load_reference_ratio = max(
            float(cfg.get("part_load_reference_ratio", 0.30)),
            _KWH_EPS,
        )
        self.return_temperature_reference_C = float(
            cfg.get("return_temperature_reference_C", 45.0 if self.condensing else 60.0)
        )
        default_slope = 0.0020 if self.condensing else 0.0004
        self.return_temperature_efficiency_slope_per_K = float(
            cfg.get("return_temperature_efficiency_slope_per_K", default_slope)
        )

        self.standby_loss_kWh_per_h = max(
            float(cfg.get("standby_loss_kWh_per_h", cfg.get("Q_gen_stby_kWh_per_h", 0.0))),
            0.0,
        )
        self.auxiliary_power_kW = max(
            float(cfg.get("auxiliary_power_kW", cfg.get("P_aux_active_kW", 0.0))),
            0.0,
        )
        self.standby_auxiliary_power_kW = max(
            float(cfg.get("standby_auxiliary_power_kW", cfg.get("P_aux_stby_kW", 0.0))),
            0.0,
        )
        self.thermal_loss_room_fraction = _fraction(
            cfg.get("thermal_loss_room_fraction", cfg.get("f_gen_ls_rbl", 0.0)),
            "thermal_loss_room_fraction",
        )
        self.auxiliary_to_medium_fraction = _fraction(
            cfg.get("auxiliary_to_medium_fraction", cfg.get("f_aux_rvd", 0.0)),
            "auxiliary_to_medium_fraction",
        )

        self.heating_supply_temperature_C = float(cfg.get("heating_supply_temperature_C", 55.0))
        self.heating_return_temperature_C = float(cfg.get("heating_return_temperature_C", 45.0))
        self.dhw_sink_temperature_C = float(cfg.get("dhw_sink_temperature_C", 60.0))
        self.dhw_return_temperature_C = float(cfg.get("dhw_return_temperature_C", 50.0))

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        out = pd.DataFrame(index=df.index)
        out.loc[:, "hours"] = self._time_step_hours(df)
        out.loc[:, "T_ext_C"] = _series_from_aliases(
            df, ["T_ext", "theta_ext", "outdoor_temperature_C", "T_external_C"], default=np.nan
        ).astype(float)

        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        out.loc[:, "Q_H_gen_out_req_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_H_gen_out_kWh",
                "Q_H_sto_in_kWh",
                "Q_H_dis_in_kWh",
                "Q_H_kWh",
                "QH_kWh",
                "space_heating_kWh",
            ],
            raw_aliases=["Q_H", "Q_h", "Heating_needs"],
            fallback=np.maximum(q_hc, 0.0),
        )
        out.loc[:, "Q_W_gen_out_req_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_W_gen_out_kWh",
                "Q_W_sto_in_kWh",
                "Q_W_dis_in_kWh",
                "Q_W_kWh",
                "QW_kWh",
                "Q_DHW_kWh",
                "DHW_kWh",
                "dhw_kWh",
            ],
            raw_aliases=["Q_W", "Q_DHW"],
            fallback=0.0,
            raw_default_unit="kwh",
        )
        if not self.heating_enabled:
            out.loc[:, "Q_H_gen_out_req_kWh"] = 0.0
        if not self.dhw_enabled:
            out.loc[:, "Q_W_gen_out_req_kWh"] = 0.0

        out.loc[:, "T_H_supply_C"] = _series_from_aliases(
            df,
            ["T_H_sink_C", "theta_H_supply_C", "heating_supply_temperature_C"],
            default=self.heating_supply_temperature_C,
        ).astype(float)
        out.loc[:, "T_H_return_C"] = _series_from_aliases(
            df,
            ["T_H_return_C", "theta_H_return_C", "heating_return_temperature_C"],
            default=self.heating_return_temperature_C,
        ).astype(float)
        out.loc[:, "T_W_sink_C"] = _series_from_aliases(
            df,
            ["T_W_sink_C", "dhw_sink_temperature_C", "dhw_tank_temperature_C"],
            default=self.dhw_sink_temperature_C,
        ).astype(float)
        out.loc[:, "T_W_return_C"] = _series_from_aliases(
            df,
            ["T_W_return_C", "dhw_return_temperature_C"],
            default=self.dhw_return_temperature_C,
        ).astype(float)

        for col in ["Q_H_gen_out_req_kWh", "Q_W_gen_out_req_kWh"]:
            out.loc[:, col] = out[col].fillna(0.0).clip(lower=0.0)
        return out

    def _demand_from_columns(
        self,
        df: pd.DataFrame,
        kwh_aliases: list[str],
        raw_aliases: list[str],
        fallback: float | pd.Series,
        raw_default_unit: str | None = None,
    ) -> pd.Series:
        kwh = _series_from_aliases(df, kwh_aliases, default=None)
        if kwh is not None:
            return kwh.astype(float)

        raw = _series_from_aliases(df, raw_aliases, default=None)
        if raw is None:
            if isinstance(fallback, pd.Series):
                raw = fallback.astype(float)
                unit = self.demand_unit
            else:
                return pd.Series(float(fallback), index=df.index)
        else:
            raw = raw.astype(float)
            unit = raw_default_unit or self.demand_unit

        if unit == "wh":
            return raw / 1000.0
        return raw

    def _time_step_hours(self, df: pd.DataFrame) -> pd.Series:
        step = _series_from_aliases(df, ["time_step_hours", "dt_h"], default=np.nan)
        if not step.isna().all():
            return step.astype(float).clip(lower=0.0)
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            diffs = df.index.to_series().diff().dt.total_seconds().dropna() / 3600.0
            if not diffs.empty and np.isfinite(diffs.median()) and diffs.median() > 0:
                return pd.Series(float(diffs.median()), index=df.index)
        return pd.Series(self.default_time_step_h, index=df.index)

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy()
        q_req_total = out["Q_H_gen_out_req_kWh"] + out["Q_W_gen_out_req_kWh"]
        hours = out["hours"].clip(lower=0.0)
        if self.nominal_power_kW > _KWH_EPS:
            capacity = self.nominal_power_kW * hours
        else:
            capacity = q_req_total.copy()

        q_served_total = np.minimum(q_req_total, capacity).clip(lower=0.0)
        q_unmet_total = (q_req_total - q_served_total).clip(lower=0.0)
        share_h = _safe_ratio_series(out["Q_H_gen_out_req_kWh"], q_req_total)
        share_w = _safe_ratio_series(out["Q_W_gen_out_req_kWh"], q_req_total)

        out.loc[:, "Q_H_gen_out_kWh"] = q_served_total * share_h
        out.loc[:, "Q_W_gen_out_kWh"] = q_served_total * share_w
        out.loc[:, "Q_HW_gen_out_kWh"] = q_served_total
        out.loc[:, "Q_H_gen_unmet_kWh"] = q_unmet_total * share_h
        out.loc[:, "Q_W_gen_unmet_kWh"] = q_unmet_total * share_w
        out.loc[:, "Q_HW_gen_unmet_kWh"] = q_unmet_total

        active = q_served_total > _KWH_EPS
        runtime = pd.Series(0.0, index=out.index)
        if self.nominal_power_kW > _KWH_EPS:
            runtime = (q_served_total / self.nominal_power_kW).clip(lower=0.0)
            runtime = np.minimum(runtime, hours)
        else:
            runtime.loc[active] = hours.loc[active]
        out.loc[:, "t_HW_gen_runtime_h"] = runtime

        plr = pd.Series(0.0, index=out.index)
        if self.nominal_power_kW > _KWH_EPS:
            plr = _safe_ratio_series(q_served_total, self.nominal_power_kW * hours)
        plr = plr.clip(lower=0.0, upper=1.0)
        plr_for_eff = plr.clip(lower=self.minimum_part_load_ratio)
        out.loc[:, "f_HW_gen_part_load"] = plr
        out.loc[:, "f_HW_gen_part_load_for_efficiency"] = plr_for_eff

        return_temp = _safe_ratio_series(
            out["Q_H_gen_out_kWh"] * out["T_H_return_C"]
            + out["Q_W_gen_out_kWh"] * out["T_W_return_C"],
            q_served_total,
            default=(self.heating_return_temperature_C + self.dhw_return_temperature_C) / 2.0,
        )
        out.loc[:, "theta_HW_gen_return_C"] = return_temp
        efficiency = self._efficiency(plr_for_eff, return_temp)
        out.loc[:, "eta_HW_gen"] = efficiency.where(active, np.nan)

        standby_loss = self.standby_loss_kWh_per_h * hours
        fuel_input = pd.Series(0.0, index=out.index)
        fuel_input.loc[active] = (
            q_served_total.loc[active] / efficiency.loc[active].clip(lower=_KWH_EPS)
            + standby_loss.loc[active]
        )
        out.loc[:, "E_HW_fuel_in_kWh"] = fuel_input
        out.loc[:, "E_H_fuel_in_kWh"] = fuel_input * share_h
        out.loc[:, "E_W_fuel_in_kWh"] = fuel_input * share_w
        out.loc[:, "EHW_gen_in_kWh"] = fuel_input

        out.loc[:, "Q_HW_gen_loss_kWh"] = (fuel_input - q_served_total).clip(lower=0.0)
        out.loc[:, "Q_HW_gen_loss_rbl_kWh"] = (
            out["Q_HW_gen_loss_kWh"] * self.thermal_loss_room_fraction
        )
        out.loc[:, "Q_HW_gen_loss_nrbl_kWh"] = (
            out["Q_HW_gen_loss_kWh"] - out["Q_HW_gen_loss_rbl_kWh"]
        )

        aux = self.auxiliary_power_kW * runtime + self.standby_auxiliary_power_kW * (
            hours - runtime
        ).clip(lower=0.0)
        out.loc[:, "W_HW_gen_aux_kWh"] = aux
        out.loc[:, "Q_HW_gen_aux_rvd_kWh"] = aux * self.auxiliary_to_medium_fraction
        out.loc[:, "Q_HW_gen_aux_rbl_kWh"] = (
            aux - out["Q_HW_gen_aux_rvd_kWh"]
        ) * self.thermal_loss_room_fraction
        return out

    def _efficiency(self, part_load_ratio: pd.Series, return_temperature_C: pd.Series) -> pd.Series:
        ratio = part_load_ratio.clip(lower=0.0, upper=1.0)
        eta = pd.Series(self.full_load_efficiency, index=ratio.index, dtype=float)
        low = ratio <= self.part_load_reference_ratio
        eta.loc[low] = self.part_load_efficiency
        high = ~low
        if high.any():
            span = max(1.0 - self.part_load_reference_ratio, _KWH_EPS)
            weight = (ratio.loc[high] - self.part_load_reference_ratio) / span
            eta.loc[high] = (
                self.part_load_efficiency * (1.0 - weight)
                + self.full_load_efficiency * weight
            )
        eta += (
            self.return_temperature_reference_C - return_temperature_C.astype(float)
        ) * self.return_temperature_efficiency_slope_per_K
        return eta.clip(lower=self.efficiency_min, upper=self.efficiency_max)

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        q_h = s("Q_H_gen_out_kWh")
        q_w = s("Q_W_gen_out_kWh")
        q_hw = s("Q_HW_gen_out_kWh")
        e_fuel = s("E_HW_fuel_in_kWh")
        aux = s("W_HW_gen_aux_kWh")
        eta = _ratio(q_hw, e_fuel)
        return {
            "QH_gen_out_kWh": q_h,
            "QW_gen_out_kWh": q_w,
            "QHW_gen_out_kWh": q_hw,
            "QH_unmet_kWh": s("Q_H_gen_unmet_kWh"),
            "QW_unmet_kWh": s("Q_W_gen_unmet_kWh"),
            "QHW_unmet_kWh": s("Q_HW_gen_unmet_kWh"),
            "EH_fuel_in_kWh": s("E_H_fuel_in_kWh"),
            "EW_fuel_in_kWh": s("E_W_fuel_in_kWh"),
            "EHW_gen_in_kWh": e_fuel,
            "EHW_fuel_in_kWh": e_fuel,
            "WHW_gen_aux_kWh": aux,
            "E_total_fuel_kWh": e_fuel,
            "E_total_electricity_kWh": aux,
            "QHW_gen_loss_kWh": s("Q_HW_gen_loss_kWh"),
            "QHW_gen_loss_rbl_kWh": s("Q_HW_gen_loss_rbl_kWh"),
            "eta_HW_gen": eta,
            "SPF_HW_gen": eta,
            "e_HW_gen": _ratio(e_fuel, q_hw),
            "f_HW_gen_part_load_mean": _weighted_mean(
                results["f_HW_gen_part_load"], results["Q_HW_gen_out_kWh"] + _KWH_EPS
            ),
            "theta_HW_gen_return_mean_C": _weighted_mean(
                results["theta_HW_gen_return_C"], results["Q_HW_gen_out_kWh"] + _KWH_EPS
            ),
        }


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


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _fraction(value: Any, name: str) -> float:
    try:
        fraction = float(value)
    except Exception as exc:  # pragma: no cover - defensive type context
        raise ValueError(f"{name} must be a fraction.") from exc
    return float(np.clip(fraction, 0.0, 1.0))


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_ratio_series(
    numerator: pd.Series,
    denominator: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    den = denominator.astype(float)
    out = pd.Series(default, index=den.index, dtype=float)
    mask = den.abs() > _KWH_EPS
    out.loc[mask] = numerator.astype(float).loc[mask] / den.loc[mask]
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.Series(values, dtype=float)
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    mask = v.notna() & w.notna()
    if not mask.any() or float(w[mask].sum()) <= _KWH_EPS:
        return float(v[mask].mean()) if mask.any() else 0.0
    return float(np.average(v[mask], weights=w[mask]))
