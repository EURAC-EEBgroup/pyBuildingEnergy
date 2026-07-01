"""District heating and cooling calculation for EN 15316-4-5 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


@dataclass
class DistrictSystemSimulationResult:
    """Container returned by :class:`DistrictEnergySystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class DistrictEnergySystemCalculator:
    """District heating/cooling substation calculator.

    The EN 15316-4-5 boundary is represented with delivered thermal loads,
    substation/network loss factors and auxiliary electricity. Heating and DHW
    are grouped on the district-heating branch; cooling can be calculated on a
    district-cooling branch when present.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> DistrictSystemSimulationResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return DistrictSystemSimulationResult(
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
        self.heating_enabled = bool(cfg.get("heating_enabled", True))
        self.dhw_enabled = bool(cfg.get("dhw_enabled", True))
        self.cooling_enabled = bool(cfg.get("cooling_enabled", False))
        self.heating_efficiency = _fraction(
            cfg.get("heating_substation_efficiency", cfg.get("eta_H_district", 0.97)),
            "heating_substation_efficiency",
        )
        self.dhw_efficiency = _fraction(
            cfg.get("dhw_substation_efficiency", cfg.get("eta_W_district", self.heating_efficiency)),
            "dhw_substation_efficiency",
        )
        self.cooling_efficiency = _fraction(
            cfg.get("cooling_substation_efficiency", cfg.get("eta_C_district", 0.95)),
            "cooling_substation_efficiency",
        )
        self.heating_fixed_loss_kWh_per_h = max(
            float(cfg.get("heating_fixed_loss_kWh_per_h", 0.0)), 0.0
        )
        self.cooling_fixed_loss_kWh_per_h = max(
            float(cfg.get("cooling_fixed_loss_kWh_per_h", 0.0)), 0.0
        )
        self.heating_auxiliary_power_kW = max(
            float(cfg.get("heating_auxiliary_power_kW", 0.0)), 0.0
        )
        self.cooling_auxiliary_power_kW = max(
            float(cfg.get("cooling_auxiliary_power_kW", 0.0)), 0.0
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
        out.loc[:, "Q_C_gen_out_req_kWh"] = self._demand_from_columns(
            df,
            ["Q_C_gen_out_kWh", "Q_C_sto_in_kWh", "Q_C_dis_in_kWh", "Q_C_kWh", "QC_kWh"],
            ["Q_C", "Cooling_needs"],
            np.maximum(-q_hc, 0.0),
        )
        if not self.heating_enabled:
            out.loc[:, "Q_H_gen_out_req_kWh"] = 0.0
        if not self.dhw_enabled:
            out.loc[:, "Q_W_gen_out_req_kWh"] = 0.0
        if not self.cooling_enabled:
            out.loc[:, "Q_C_gen_out_req_kWh"] = 0.0
        for col in ["Q_H_gen_out_req_kWh", "Q_W_gen_out_req_kWh", "Q_C_gen_out_req_kWh"]:
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
        out.loc[:, "Q_H_gen_out_kWh"] = out["Q_H_gen_out_req_kWh"]
        out.loc[:, "Q_W_gen_out_kWh"] = out["Q_W_gen_out_req_kWh"]
        out.loc[:, "Q_C_gen_out_kWh"] = out["Q_C_gen_out_req_kWh"]
        out.loc[:, "E_H_district_in_kWh"] = out["Q_H_gen_out_kWh"] / max(self.heating_efficiency, _KWH_EPS)
        out.loc[:, "E_W_district_in_kWh"] = out["Q_W_gen_out_kWh"] / max(self.dhw_efficiency, _KWH_EPS)
        out.loc[:, "E_C_district_in_kWh"] = out["Q_C_gen_out_kWh"] / max(self.cooling_efficiency, _KWH_EPS)
        active_hw = (out["Q_H_gen_out_kWh"] + out["Q_W_gen_out_kWh"]) > _KWH_EPS
        active_c = out["Q_C_gen_out_kWh"] > _KWH_EPS
        out.loc[:, "Q_HW_district_fixed_loss_kWh"] = self.heating_fixed_loss_kWh_per_h * hours * active_hw
        out.loc[:, "Q_C_district_fixed_loss_kWh"] = self.cooling_fixed_loss_kWh_per_h * hours * active_c
        out.loc[:, "E_HW_district_in_kWh"] = (
            out["E_H_district_in_kWh"]
            + out["E_W_district_in_kWh"]
            + out["Q_HW_district_fixed_loss_kWh"]
        )
        out.loc[:, "E_C_district_in_kWh"] = out["E_C_district_in_kWh"] + out["Q_C_district_fixed_loss_kWh"]
        out.loc[:, "Q_HW_gen_loss_kWh"] = (
            out["E_HW_district_in_kWh"] - out["Q_H_gen_out_kWh"] - out["Q_W_gen_out_kWh"]
        ).clip(lower=0.0)
        out.loc[:, "Q_C_gen_loss_kWh"] = (
            out["E_C_district_in_kWh"] - out["Q_C_gen_out_kWh"]
        ).clip(lower=0.0)
        out.loc[:, "Q_HW_gen_loss_rbl_kWh"] = out["Q_HW_gen_loss_kWh"] * self.loss_recoverable_fraction
        out.loc[:, "Q_C_gen_loss_rbl_kWh"] = out["Q_C_gen_loss_kWh"] * self.loss_recoverable_fraction
        out.loc[:, "W_HW_gen_aux_kWh"] = self.heating_auxiliary_power_kW * hours * active_hw
        out.loc[:, "W_C_gen_aux_kWh"] = self.cooling_auxiliary_power_kW * hours * active_c
        return out

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        q_hw = s("Q_H_gen_out_kWh") + s("Q_W_gen_out_kWh")
        e_hw = s("E_HW_district_in_kWh")
        q_c = s("Q_C_gen_out_kWh")
        e_c = s("E_C_district_in_kWh")
        aux_hw = s("W_HW_gen_aux_kWh")
        aux_c = s("W_C_gen_aux_kWh")
        return {
            "QH_gen_out_kWh": s("Q_H_gen_out_kWh"),
            "QW_gen_out_kWh": s("Q_W_gen_out_kWh"),
            "QC_gen_out_kWh": q_c,
            "QHW_gen_out_kWh": q_hw,
            "EHW_gen_in_kWh": e_hw,
            "EC_gen_in_kWh": e_c,
            "EHW_district_in_kWh": e_hw,
            "EC_district_in_kWh": e_c,
            "WHW_gen_aux_kWh": aux_hw,
            "WC_gen_aux_kWh": aux_c,
            "QHW_gen_loss_kWh": s("Q_HW_gen_loss_kWh"),
            "QC_gen_loss_kWh": s("Q_C_gen_loss_kWh"),
            "eta_HW_gen": _ratio(q_hw, e_hw),
            "SEER_C_gen": _ratio(q_c, e_c + aux_c),
            "SPF_HW_gen": _ratio(q_hw, e_hw + aux_hw),
            "E_total_district_kWh": e_hw + e_c,
            "E_total_electricity_kWh": aux_hw + aux_c,
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
    return float(np.clip(fraction, _KWH_EPS, 1.0))


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return float(numerator / denominator)
