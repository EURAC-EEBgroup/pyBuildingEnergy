"""EN 16798-5/7 style ventilation and heat-recovery calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


AIR_HEAT_CAPACITY_WH_M3K = 0.33
_KWH_EPS = 1e-12


@dataclass
class VentilationSystemSimulationResult:
    """Container returned by :class:`VentilationSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class VentilationSystemCalculator:
    """Mechanical/natural ventilation balance with heat recovery and fans."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> VentilationSystemSimulationResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return VentilationSystemSimulationResult(
            timeseries=results,
            summary=self._summarize(results),
            inputs=dict(self.input_data),
        )

    def effective_heat_transfer_coefficient_W_K(self) -> float:
        """Return the nominal ventilation heat-transfer coefficient after HRV."""

        h_outdoor = AIR_HEAT_CAPACITY_WH_M3K * self.nominal_flow_m3_h
        return h_outdoor * (1.0 - self.heat_recovery_efficiency * self.heat_recovery_fraction)

    def _load_options(self) -> None:
        cfg = self.input_data
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.volume_m3 = max(float(cfg.get("volume_m3", 0.0)), 0.0)
        self.air_change_rate_h = max(float(cfg.get("air_change_rate_h", 0.30)), 0.0)
        nominal_flow = cfg.get("nominal_flow_m3_h", cfg.get("supply_air_flow_m3_h", None))
        if nominal_flow is None:
            nominal_flow = self.volume_m3 * self.air_change_rate_h
        self.nominal_flow_m3_h = max(float(nominal_flow), 0.0)
        self.heat_recovery_efficiency = _fraction(
            cfg.get("heat_recovery_efficiency", cfg.get("temperature_efficiency", 0.0)),
            "heat_recovery_efficiency",
        )
        self.heat_recovery_fraction = _fraction(
            cfg.get("heat_recovery_fraction", 1.0),
            "heat_recovery_fraction",
        )
        self.mechanical_fraction = _fraction(
            cfg.get("mechanical_fraction", 1.0),
            "mechanical_fraction",
        )
        self.specific_fan_power_W_s_m3 = max(
            float(cfg.get("specific_fan_power_W_s_m3", cfg.get("SFP_W_s_m3", 900.0))),
            0.0,
        )
        self.fan_power_W_per_m3_h = max(
            float(cfg.get("fan_power_W_per_m3_h", 0.0)),
            0.0,
        )
        self.supply_temperature_setpoint_C = float(
            cfg.get("supply_temperature_setpoint_C", 18.0)
        )
        self.heat_recovery_bypass_temperature_C = float(
            cfg.get("heat_recovery_bypass_temperature_C", 24.0)
        )
        self.fan_heat_to_air_fraction = _fraction(
            cfg.get("fan_heat_to_air_fraction", 0.6),
            "fan_heat_to_air_fraction",
        )

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        out = pd.DataFrame(index=data.index)
        out.loc[:, "hours"] = self._time_step_hours(data)
        out.loc[:, "T_ext_C"] = _series_from_aliases(
            data, ["T_ext", "theta_ext", "outdoor_temperature_C"], default=0.0
        ).astype(float)
        out.loc[:, "T_zone_C"] = _series_from_aliases(
            data, ["T_zone", "T_int", "theta_int", "indoor_temperature_C"], default=20.0
        ).astype(float)
        out.loc[:, "ventilation_profile"] = _series_from_aliases(
            data,
            ["ventilation_profile", "profile_ventilation", "operation_profile"],
            default=1.0,
        ).astype(float).fillna(0.0).clip(lower=0.0, upper=1.0)
        out.loc[:, "flow_m3_h"] = _series_from_aliases(
            data, ["flow_m3_h", "ventilation_flow_m3_h"], default=self.nominal_flow_m3_h
        ).astype(float).fillna(self.nominal_flow_m3_h).clip(lower=0.0)
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

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy()
        flow = out["flow_m3_h"] * out["ventilation_profile"]
        mechanical_flow = flow * self.mechanical_fraction
        natural_flow = flow - mechanical_flow
        h_outdoor = AIR_HEAT_CAPACITY_WH_M3K * flow
        h_mech = AIR_HEAT_CAPACITY_WH_M3K * mechanical_flow
        h_nat = AIR_HEAT_CAPACITY_WH_M3K * natural_flow

        hrv_active = out["T_ext_C"] < self.heat_recovery_bypass_temperature_C
        h_recovered = h_mech * self.heat_recovery_efficiency * self.heat_recovery_fraction
        h_recovered = h_recovered.where(hrv_active, 0.0)
        h_effective = (h_outdoor - h_recovered).clip(lower=0.0)
        delta_t = (out["T_zone_C"] - out["T_ext_C"]).clip(lower=0.0)

        fan_power_from_sfp_W = mechanical_flow / 3600.0 * self.specific_fan_power_W_s_m3
        fan_power_from_m3h_W = mechanical_flow * self.fan_power_W_per_m3_h
        fan_power_W = fan_power_from_sfp_W + fan_power_from_m3h_W

        out.loc[:, "flow_m3_h"] = flow
        out.loc[:, "mechanical_flow_m3_h"] = mechanical_flow
        out.loc[:, "natural_flow_m3_h"] = natural_flow
        out.loc[:, "H_ve_outdoor_W_K"] = h_outdoor
        out.loc[:, "H_ve_recovered_W_K"] = h_recovered
        out.loc[:, "H_ve_effective_W_K"] = h_effective
        out.loc[:, "Q_vent_heat_loss_before_HRV_kWh"] = h_outdoor * delta_t * out["hours"] / 1000.0
        out.loc[:, "Q_vent_heat_recovered_kWh"] = h_recovered * delta_t * out["hours"] / 1000.0
        out.loc[:, "Q_vent_heat_loss_after_HRV_kWh"] = h_effective * delta_t * out["hours"] / 1000.0
        out.loc[:, "P_vent_fan_W"] = fan_power_W
        out.loc[:, "E_vent_fan_kWh"] = fan_power_W * out["hours"] / 1000.0
        out.loc[:, "Q_fan_heat_to_air_kWh"] = (
            out["E_vent_fan_kWh"] * self.fan_heat_to_air_fraction
        )
        return out

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        q_before = s("Q_vent_heat_loss_before_HRV_kWh")
        q_recovered = s("Q_vent_heat_recovered_kWh")
        return {
            "V_vent_m3": float((results["flow_m3_h"] * results["hours"]).sum()),
            "H_ve_outdoor_mean_W_K": float(results["H_ve_outdoor_W_K"].mean()),
            "H_ve_effective_mean_W_K": float(results["H_ve_effective_W_K"].mean()),
            "Q_vent_heat_loss_before_HRV_kWh": q_before,
            "Q_vent_heat_recovered_kWh": q_recovered,
            "Q_vent_heat_loss_after_HRV_kWh": s("Q_vent_heat_loss_after_HRV_kWh"),
            "f_heat_recovered": q_recovered / q_before if q_before > _KWH_EPS else float("nan"),
            "E_vent_fan_kWh": s("E_vent_fan_kWh"),
            "P_vent_fan_peak_W": float(results["P_vent_fan_W"].max())
            if "P_vent_fan_W" in results
            else 0.0,
            "Q_fan_heat_to_air_kWh": s("Q_fan_heat_to_air_kWh"),
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
