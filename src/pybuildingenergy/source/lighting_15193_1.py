"""EN 15193-1 style lighting energy calculation.

This module calculates lighting electricity and associated internal gains from
installed power, operation schedules and optional daylight/occupancy controls.
It is intentionally data-driven so national annex values and project-specific
LENI parameters can be supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


@dataclass
class LightingSimulationResult:
    """Container returned by :class:`LightingSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class LightingSystemCalculator:
    """Lighting electricity, parasitic energy and heat gains."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> LightingSimulationResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return LightingSimulationResult(
            timeseries=results,
            summary=self._summarize(results),
            inputs=dict(self.input_data),
        )

    def _load_options(self) -> None:
        cfg = self.input_data
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.area_m2 = max(float(cfg.get("area_m2", cfg.get("floor_area_m2", 0.0))), 0.0)
        self.installed_power_density_W_m2 = max(
            float(cfg.get("installed_power_density_W_m2", cfg.get("LPD_W_m2", 7.0))),
            0.0,
        )
        self.daylight_dependency_factor = _fraction(
            cfg.get("daylight_dependency_factor", cfg.get("F_D", 1.0)),
            "daylight_dependency_factor",
        )
        self.occupancy_dependency_factor = _fraction(
            cfg.get("occupancy_dependency_factor", cfg.get("F_O", 1.0)),
            "occupancy_dependency_factor",
        )
        self.constant_illuminance_factor = _fraction(
            cfg.get("constant_illuminance_factor", cfg.get("F_C", 1.0)),
            "constant_illuminance_factor",
        )
        self.daylight_control_fraction = _fraction(
            cfg.get("daylight_control_fraction", 0.0),
            "daylight_control_fraction",
        )
        self.occupancy_control_fraction = _fraction(
            cfg.get("occupancy_control_fraction", 0.0),
            "occupancy_control_fraction",
        )
        self.parasitic_power_W_m2 = max(
            float(cfg.get("parasitic_power_W_m2", cfg.get("standby_power_W_m2", 0.0))),
            0.0,
        )
        self.emergency_lighting_kWh_m2_a = max(
            float(cfg.get("emergency_lighting_kWh_m2_a", 0.0)),
            0.0,
        )
        self.internal_gain_fraction = _fraction(
            cfg.get("internal_gain_fraction", 1.0),
            "internal_gain_fraction",
        )

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        out = pd.DataFrame(index=data.index)
        out.loc[:, "hours"] = self._time_step_hours(data)
        out.loc[:, "area_m2"] = _series_from_aliases(
            data,
            ["area_m2", "floor_area_m2", "Af_m2"],
            default=self.area_m2,
        ).astype(float).fillna(self.area_m2).clip(lower=0.0)
        out.loc[:, "lighting_profile"] = _series_from_aliases(
            data,
            ["lighting_profile", "profile_lighting", "lighting_schedule"],
            default=self._default_lighting_profile(data.index),
        ).astype(float).fillna(0.0).clip(lower=0.0, upper=1.0)
        out.loc[:, "occupancy_profile"] = _series_from_aliases(
            data,
            ["occupancy_profile", "people_profile", "profile_occupancy"],
            default=out["lighting_profile"],
        ).astype(float).fillna(0.0).clip(lower=0.0, upper=1.0)
        out.loc[:, "GHI_W_m2"] = _series_from_aliases(
            data,
            [
                "GHI",
                "ghi",
                "ghi_W_m2",
                "global_horizontal_irradiance_W_m2",
                "I_sol_global_W_m2",
            ],
            default=np.nan,
        ).astype(float)
        return out

    def _time_step_hours(self, df: pd.DataFrame) -> pd.Series:
        step = _series_from_aliases(df, ["time_step_hours", "dt_h"], default=np.nan)
        if not step.isna().all():
            return step.astype(float).clip(lower=0.0)
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            diffs = df.index.to_series().diff().dt.total_seconds().dropna() / 3600.0
            if not diffs.empty and np.isfinite(diffs.median()) and diffs.median() > 0:
                return pd.Series(float(diffs.median()), index=df.index)
        return pd.Series(self.default_time_step_h, index=df.index)

    def _default_lighting_profile(self, index: pd.Index) -> pd.Series:
        if isinstance(index, pd.DatetimeIndex):
            hour = index.hour
            occupied = ((hour >= 6) & (hour <= 8)) | ((hour >= 17) & (hour <= 23))
            profile = np.where(occupied, 1.0, 0.15)
            return pd.Series(profile, index=index, dtype=float)
        return pd.Series(1.0, index=index, dtype=float)

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy()
        daylight_reduction = self._daylight_reduction(out)
        occupancy_reduction = self.occupancy_control_fraction * (1.0 - out["occupancy_profile"])
        control_multiplier = (
            self.daylight_dependency_factor
            * self.occupancy_dependency_factor
            * self.constant_illuminance_factor
            * (1.0 - daylight_reduction)
            * (1.0 - occupancy_reduction)
        )
        control_multiplier = control_multiplier.clip(lower=0.0, upper=1.0)
        out.loc[:, "P_light_W"] = (
            out["area_m2"]
            * self.installed_power_density_W_m2
            * out["lighting_profile"]
            * control_multiplier
        )
        out.loc[:, "E_L_lighting_kWh"] = out["P_light_W"] * out["hours"] / 1000.0
        out.loc[:, "E_L_parasitic_kWh"] = (
            out["area_m2"] * self.parasitic_power_W_m2 * out["hours"] / 1000.0
        )
        emergency_total = self.emergency_lighting_kWh_m2_a * self.area_m2
        if len(out) > 0:
            out.loc[:, "E_L_emergency_kWh"] = emergency_total / len(out)
        else:  # pragma: no cover
            out.loc[:, "E_L_emergency_kWh"] = 0.0
        out.loc[:, "E_L_total_kWh"] = (
            out["E_L_lighting_kWh"]
            + out["E_L_parasitic_kWh"]
            + out["E_L_emergency_kWh"]
        )
        out.loc[:, "Q_L_internal_gain_kWh"] = (
            out["E_L_lighting_kWh"] * self.internal_gain_fraction
        )
        out.loc[:, "F_light_controls"] = control_multiplier
        return out

    def _daylight_reduction(self, out: pd.DataFrame) -> pd.Series:
        if self.daylight_control_fraction <= 0.0:
            return pd.Series(0.0, index=out.index)
        ghi = out["GHI_W_m2"].fillna(0.0).clip(lower=0.0)
        if float(ghi.max()) <= _KWH_EPS:
            return pd.Series(0.0, index=out.index)
        daylight_signal = (ghi / max(float(ghi.quantile(0.95)), _KWH_EPS)).clip(0.0, 1.0)
        return daylight_signal * self.daylight_control_fraction

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        area = float(results["area_m2"].mean()) if "area_m2" in results else self.area_m2
        total = s("E_L_total_kWh")
        return {
            "E_L_lighting_kWh": s("E_L_lighting_kWh"),
            "E_L_parasitic_kWh": s("E_L_parasitic_kWh"),
            "E_L_emergency_kWh": s("E_L_emergency_kWh"),
            "E_L_total_kWh": total,
            "Q_L_internal_gain_kWh": s("Q_L_internal_gain_kWh"),
            "LENI_kWh_m2a": total / area if area > _KWH_EPS else float("nan"),
            "P_light_peak_W": float(results["P_light_W"].max())
            if "P_light_W" in results
            else 0.0,
            "F_light_controls_mean": float(results["F_light_controls"].mean())
            if "F_light_controls" in results
            else 1.0,
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
