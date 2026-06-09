"""Cogeneration calculation for EN 15316-4-4 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


@dataclass
class CogenerationSimulationResult:
    """Container returned by :class:`CogenerationSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class CogenerationSystemCalculator:
    """Thermal-led CHP generator calculator.

    The model is intended for the EN 15316-4-4 system boundary. It covers a
    thermal-led unit, optional auxiliary/standby electricity, useful electricity
    generation, self-consumption and backup heat for capacity shortfalls.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> CogenerationSimulationResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return CogenerationSimulationResult(
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
        self.fuel = str(cfg.get("fuel", cfg.get("energy_carrier", "natural_gas")))
        self.nominal_thermal_power_kW = max(
            float(cfg.get("nominal_thermal_power_kW", cfg.get("rated_thermal_power_kW", 0.0))),
            0.0,
        )
        self.thermal_efficiency = _fraction(
            cfg.get("thermal_efficiency", cfg.get("eta_th", 0.55)),
            "thermal_efficiency",
        )
        self.electrical_efficiency = _fraction(
            cfg.get("electrical_efficiency", cfg.get("eta_el", 0.30)),
            "electrical_efficiency",
        )
        if self.thermal_efficiency + self.electrical_efficiency > 1.15:
            raise ValueError("thermal_efficiency + electrical_efficiency is implausibly high.")
        self.backup_efficiency = _fraction(
            cfg.get("backup_efficiency", cfg.get("eta_backup", 0.92)),
            "backup_efficiency",
        )
        self.minimum_part_load_ratio = _fraction(
            cfg.get("minimum_part_load_ratio", 0.0), "minimum_part_load_ratio"
        )
        self.auxiliary_power_kW = max(float(cfg.get("auxiliary_power_kW", 0.0)), 0.0)
        self.standby_auxiliary_power_kW = max(
            float(cfg.get("standby_auxiliary_power_kW", 0.0)), 0.0
        )
        self.loss_recoverable_fraction = _fraction(
            cfg.get("loss_recoverable_fraction", 0.0), "loss_recoverable_fraction"
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
        out.loc[:, "Q_H_gen_out_req_kWh"] = self._demand_from_columns(
            df,
            ["Q_H_gen_out_kWh", "Q_H_sto_in_kWh", "Q_H_dis_in_kWh", "Q_H_kWh", "QH_kWh"],
            ["Q_H", "Q_h", "Heating_needs"],
            np.maximum(q_hc, 0.0),
        )
        out.loc[:, "Q_W_gen_out_req_kWh"] = self._demand_from_columns(
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
        hours = out["hours"].clip(lower=0.0)
        q_req = out["Q_H_gen_out_req_kWh"] + out["Q_W_gen_out_req_kWh"]
        if self.nominal_thermal_power_kW > _KWH_EPS:
            capacity = self.nominal_thermal_power_kW * hours
        else:
            capacity = q_req.copy()
        q_chp = np.minimum(q_req, capacity).clip(lower=0.0)
        q_backup = (q_req - q_chp).clip(lower=0.0)
        share_h = _safe_ratio_series(out["Q_H_gen_out_req_kWh"], q_req)
        share_w = _safe_ratio_series(out["Q_W_gen_out_req_kWh"], q_req)

        out.loc[:, "Q_H_chp_out_kWh"] = q_chp * share_h
        out.loc[:, "Q_W_chp_out_kWh"] = q_chp * share_w
        out.loc[:, "Q_H_backup_out_kWh"] = q_backup * share_h
        out.loc[:, "Q_W_backup_out_kWh"] = q_backup * share_w
        out.loc[:, "Q_H_gen_out_kWh"] = out["Q_H_chp_out_kWh"] + out["Q_H_backup_out_kWh"]
        out.loc[:, "Q_W_gen_out_kWh"] = out["Q_W_chp_out_kWh"] + out["Q_W_backup_out_kWh"]
        out.loc[:, "Q_HW_gen_out_kWh"] = out["Q_H_gen_out_kWh"] + out["Q_W_gen_out_kWh"]

        runtime = pd.Series(0.0, index=out.index)
        if self.nominal_thermal_power_kW > _KWH_EPS:
            runtime = (q_chp / self.nominal_thermal_power_kW).clip(lower=0.0)
            runtime = np.minimum(runtime, hours)
        else:
            runtime.loc[q_chp > _KWH_EPS] = hours.loc[q_chp > _KWH_EPS]
        out.loc[:, "t_chp_runtime_h"] = runtime
        plr = pd.Series(0.0, index=out.index)
        if self.nominal_thermal_power_kW > _KWH_EPS:
            plr = _safe_ratio_series(q_chp, self.nominal_thermal_power_kW * hours)
        out.loc[:, "f_chp_part_load"] = plr.clip(lower=0.0, upper=1.0)
        out.loc[:, "f_chp_part_load_for_efficiency"] = out["f_chp_part_load"].clip(
            lower=self.minimum_part_load_ratio
        )

        chp_fuel = q_chp / max(self.thermal_efficiency, _KWH_EPS)
        backup_fuel = q_backup / max(self.backup_efficiency, _KWH_EPS)
        out.loc[:, "E_chp_fuel_in_kWh"] = chp_fuel
        out.loc[:, "E_backup_fuel_in_kWh"] = backup_fuel
        out.loc[:, "EHW_gen_in_kWh"] = chp_fuel + backup_fuel
        out.loc[:, "E_chp_el_generated_kWh"] = chp_fuel * self.electrical_efficiency
        out.loc[:, "E_chp_el_self_consumed_kWh"] = np.minimum(
            out["E_chp_el_generated_kWh"], out["E_site_el_load_kWh"]
        )
        out.loc[:, "E_chp_el_export_kWh"] = (
            out["E_chp_el_generated_kWh"] - out["E_chp_el_self_consumed_kWh"]
        ).clip(lower=0.0)
        out.loc[:, "Q_HW_gen_loss_kWh"] = (
            out["EHW_gen_in_kWh"]
            - out["Q_HW_gen_out_kWh"]
            - out["E_chp_el_generated_kWh"]
        ).clip(lower=0.0)
        out.loc[:, "Q_HW_gen_loss_rbl_kWh"] = out["Q_HW_gen_loss_kWh"] * self.loss_recoverable_fraction
        out.loc[:, "W_HW_gen_aux_kWh"] = (
            self.auxiliary_power_kW * runtime
            + self.standby_auxiliary_power_kW * (hours - runtime).clip(lower=0.0)
        )
        return out

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        q_hw = s("Q_HW_gen_out_kWh")
        fuel = s("EHW_gen_in_kWh")
        aux = s("W_HW_gen_aux_kWh")
        el_gen = s("E_chp_el_generated_kWh")
        return {
            "QH_gen_out_kWh": s("Q_H_gen_out_kWh"),
            "QW_gen_out_kWh": s("Q_W_gen_out_kWh"),
            "QHW_gen_out_kWh": q_hw,
            "QHW_chp_out_kWh": s("Q_H_chp_out_kWh") + s("Q_W_chp_out_kWh"),
            "QHW_backup_out_kWh": s("Q_H_backup_out_kWh") + s("Q_W_backup_out_kWh"),
            "EHW_gen_in_kWh": fuel,
            "E_chp_fuel_in_kWh": s("E_chp_fuel_in_kWh"),
            "E_backup_fuel_in_kWh": s("E_backup_fuel_in_kWh"),
            "E_chp_el_generated_kWh": el_gen,
            "E_chp_el_self_consumed_kWh": s("E_chp_el_self_consumed_kWh"),
            "E_chp_el_export_kWh": s("E_chp_el_export_kWh"),
            "WHW_gen_aux_kWh": aux,
            "QHW_gen_loss_kWh": s("Q_HW_gen_loss_kWh"),
            "eta_HW_gen": _ratio(q_hw, fuel),
            "eta_chp_total": _ratio(q_hw + el_gen, fuel),
            "SPF_HW_gen": _ratio(q_hw, fuel + aux),
            "E_total_fuel_kWh": fuel,
            "E_total_electricity_kWh": aux,
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


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return float(numerator / denominator)
