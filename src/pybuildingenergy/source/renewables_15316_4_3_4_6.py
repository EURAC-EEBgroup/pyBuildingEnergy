"""Solar thermal and PV calculation for EN 15316-4-3/4-6 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


@dataclass
class RenewableEnergySimulationResult:
    """Container returned by :class:`RenewableEnergySystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class RenewableEnergySystemCalculator:
    """Solar thermal and PV pre/post-generator calculator.

    Solar thermal useful heat is allocated to DHW first by default and reduces
    the downstream generator heat request. PV generation is calculated on the
    same time axis and self-consumed against any supplied site electricity load.
    If irradiance columns are unavailable, the annual yield inputs are
    distributed with a deterministic daylight/month profile.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> RenewableEnergySimulationResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return RenewableEnergySimulationResult(
            timeseries=results,
            summary=self._summarize(results),
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

        solar = dict(cfg.get("solar_thermal", cfg.get("thermal_solar", {})) or {})
        self.solar_thermal_enabled = bool(solar.get("enabled", solar.get("area_m2", 0.0) > 0.0))
        self.solar_thermal_area_m2 = max(float(solar.get("area_m2", 0.0)), 0.0)
        self.solar_thermal_yield_kWh_m2_a = max(
            float(solar.get("annual_yield_kWh_m2", solar.get("yield_kWh_m2_a", 450.0))),
            0.0,
        )
        self.solar_thermal_efficiency = _fraction(
            solar.get("collector_loop_efficiency", solar.get("efficiency", 1.0)),
            "solar_thermal.efficiency",
        )
        self.solar_priority = str(solar.get("priority", "dhw_first")).lower()
        if self.solar_priority not in {"dhw_first", "heating_first", "proportional"}:
            raise ValueError("solar_thermal.priority must be dhw_first, heating_first or proportional.")

        pv = dict(cfg.get("pv", cfg.get("photovoltaic", {})) or {})
        self.pv_enabled = bool(pv.get("enabled", pv.get("capacity_kWp", pv.get("pv_kWp", 0.0)) > 0.0))
        self.pv_capacity_kWp = max(
            float(pv.get("capacity_kWp", pv.get("pv_kWp", 0.0))),
            0.0,
        )
        self.pv_yield_kWh_kWp_a = max(
            float(pv.get("annual_yield_kWh_kWp", pv.get("yield_kWh_kWp_a", 1100.0))),
            0.0,
        )
        self.pv_performance_ratio = _fraction(
            pv.get("performance_ratio", 0.80), "pv.performance_ratio"
        )
        self.pv_self_consumption_fraction = _fraction(
            pv.get("self_consumption_fraction", 1.0), "pv.self_consumption_fraction"
        )

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        out = pd.DataFrame(index=df.index)
        out.loc[:, "hours"] = self._time_step_hours(df)
        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        out.loc[:, "Q_H_before_solar_kWh"] = self._demand_from_columns(
            df,
            ["Q_H_gen_out_kWh", "Q_H_sto_in_kWh", "Q_H_dis_in_kWh", "Q_H_kWh", "QH_kWh"],
            ["Q_H", "Q_h", "Heating_needs"],
            np.maximum(q_hc, 0.0),
        )
        out.loc[:, "Q_W_before_solar_kWh"] = self._demand_from_columns(
            df,
            ["Q_W_gen_out_kWh", "Q_W_sto_in_kWh", "Q_W_dis_in_kWh", "Q_W_kWh", "Q_DHW_kWh"],
            ["Q_W", "Q_DHW"],
            0.0,
            raw_default_unit="kwh",
        )
        out.loc[:, "E_site_el_load_kWh"] = _series_from_aliases(
            df,
            [
                "E_site_el_load_kWh",
                "electricity_load_kWh",
                "E_electricity_load_kWh",
                "E_total_electricity_kWh",
            ],
            default=0.0,
        ).astype(float).clip(lower=0.0)
        irradiance = _series_from_aliases(
            df,
            [
                "GHI",
                "ghi",
                "ghi_W_m2",
                "global_horizontal_irradiance_W_m2",
                "I_sol_global_W_m2",
                "G(h)",
            ],
            default=np.nan,
        )
        out.loc[:, "GHI_W_m2"] = irradiance.astype(float)
        for col in ["Q_H_before_solar_kWh", "Q_W_before_solar_kWh"]:
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
        return raw / 1000.0 if unit == "wh" else raw

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
        profile = self._solar_profile(out)
        q_solar_available = pd.Series(0.0, index=out.index)
        if self.solar_thermal_enabled and self.solar_thermal_area_m2 > 0.0:
            annual = (
                self.solar_thermal_area_m2
                * self.solar_thermal_yield_kWh_m2_a
                * self.solar_thermal_efficiency
            )
            q_solar_available = profile * annual
        out.loc[:, "Q_solar_thermal_available_kWh"] = q_solar_available

        q_h = out["Q_H_before_solar_kWh"]
        q_w = out["Q_W_before_solar_kWh"]
        if self.solar_priority == "heating_first":
            q_h_solar = np.minimum(q_h, q_solar_available)
            q_remaining = (q_solar_available - q_h_solar).clip(lower=0.0)
            q_w_solar = np.minimum(q_w, q_remaining)
        elif self.solar_priority == "proportional":
            q_total = q_h + q_w
            q_used = np.minimum(q_total, q_solar_available)
            q_h_solar = q_used * _safe_ratio_series(q_h, q_total)
            q_w_solar = q_used * _safe_ratio_series(q_w, q_total)
        else:
            q_w_solar = np.minimum(q_w, q_solar_available)
            q_remaining = (q_solar_available - q_w_solar).clip(lower=0.0)
            q_h_solar = np.minimum(q_h, q_remaining)

        out.loc[:, "Q_H_solar_thermal_used_kWh"] = q_h_solar
        out.loc[:, "Q_W_solar_thermal_used_kWh"] = q_w_solar
        out.loc[:, "Q_solar_thermal_unused_kWh"] = (
            q_solar_available - q_h_solar - q_w_solar
        ).clip(lower=0.0)
        out.loc[:, "Q_H_after_solar_kWh"] = (q_h - q_h_solar).clip(lower=0.0)
        out.loc[:, "Q_W_after_solar_kWh"] = (q_w - q_w_solar).clip(lower=0.0)

        pv_generation = pd.Series(0.0, index=out.index)
        if self.pv_enabled and self.pv_capacity_kWp > 0.0:
            if out["GHI_W_m2"].notna().any() and out["GHI_W_m2"].clip(lower=0.0).sum() > 0:
                raw = out["GHI_W_m2"].fillna(0.0).clip(lower=0.0) / 1000.0
                raw = raw * out["hours"] * self.pv_capacity_kWp * self.pv_performance_ratio
                expected = self.pv_capacity_kWp * self.pv_yield_kWh_kWp_a
                scale = expected / max(float(raw.sum()), _KWH_EPS)
                pv_generation = raw * scale
            else:
                pv_generation = profile * self.pv_capacity_kWp * self.pv_yield_kWh_kWp_a
        out.loc[:, "E_PV_gen_kWh"] = pv_generation
        self_consumption_limit = pv_generation * self.pv_self_consumption_fraction
        out.loc[:, "E_PV_self_consumed_kWh"] = np.minimum(
            out["E_site_el_load_kWh"], self_consumption_limit
        )
        out.loc[:, "E_PV_export_kWh"] = (pv_generation - out["E_PV_self_consumed_kWh"]).clip(lower=0.0)
        out.loc[:, "E_grid_after_PV_kWh"] = (
            out["E_site_el_load_kWh"] - out["E_PV_self_consumed_kWh"]
        ).clip(lower=0.0)
        return out

    def _solar_profile(self, out: pd.DataFrame) -> pd.Series:
        ghi = out["GHI_W_m2"].fillna(0.0).clip(lower=0.0)
        if float(ghi.sum()) > 0.0:
            raw = ghi * out["hours"].clip(lower=0.0)
        else:
            if isinstance(out.index, pd.DatetimeIndex):
                month_weight = pd.Series(
                    np.clip(
                        np.sin((out.index.month - 1) / 12.0 * np.pi),
                        0.05,
                        None,
                    ),
                    index=out.index,
                    dtype=float,
                )
                hour_angle = (out.index.hour + 0.5 - 12.0) / 12.0 * np.pi
                daylight = pd.Series(np.clip(np.cos(hour_angle), 0.0, None), index=out.index)
                raw = month_weight * daylight * out["hours"].clip(lower=0.0)
            else:
                raw = pd.Series(1.0, index=out.index) * out["hours"].clip(lower=0.0)
        total = float(raw.sum())
        if total <= _KWH_EPS:
            return pd.Series(0.0, index=out.index)
        return raw / total

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        thermal_used = s("Q_H_solar_thermal_used_kWh") + s("Q_W_solar_thermal_used_kWh")
        thermal_available = s("Q_solar_thermal_available_kWh")
        pv_gen = s("E_PV_gen_kWh")
        return {
            "QH_before_solar_kWh": s("Q_H_before_solar_kWh"),
            "QW_before_solar_kWh": s("Q_W_before_solar_kWh"),
            "QH_after_solar_kWh": s("Q_H_after_solar_kWh"),
            "QW_after_solar_kWh": s("Q_W_after_solar_kWh"),
            "QH_solar_thermal_used_kWh": s("Q_H_solar_thermal_used_kWh"),
            "QW_solar_thermal_used_kWh": s("Q_W_solar_thermal_used_kWh"),
            "Q_solar_thermal_available_kWh": thermal_available,
            "Q_solar_thermal_used_kWh": thermal_used,
            "Q_solar_thermal_unused_kWh": s("Q_solar_thermal_unused_kWh"),
            "f_solar_thermal_used": thermal_used / thermal_available
            if thermal_available > _KWH_EPS
            else float("nan"),
            "E_PV_gen_kWh": pv_gen,
            "E_PV_self_consumed_kWh": s("E_PV_self_consumed_kWh"),
            "E_PV_export_kWh": s("E_PV_export_kWh"),
            "E_grid_after_PV_kWh": s("E_grid_after_PV_kWh"),
            "f_PV_self_consumed": s("E_PV_self_consumed_kWh") / pv_gen
            if pv_gen > _KWH_EPS
            else float("nan"),
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
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{name} must be a fraction.") from exc
    return float(np.clip(fraction, 0.0, 1.0))


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
