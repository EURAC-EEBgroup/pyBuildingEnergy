"""Space emission calculation according to EN 15316-2:2017.

The module implements the hourly calculation path for space-heating and
water-based space-cooling emission systems, modules M3-5 and M4-5. It is meant
to sit between building energy needs and generation: users provide the thermal
output required in the room and, when available, the output recalculated with
the equivalent internal temperature caused by emitter and control effects.

The implementation follows the EN 15316-2:2017 hourly structure:

* equivalent internal temperature variation from stratification, control,
  radiation, hydraulic balancing and room automation;
* optional recalculated heating/cooling output with that equivalent internal
  temperature;
* embedded-emitter losses;
* emission input energy, auxiliary fan/control electricity and annual
  expenditure factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warnings

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


@dataclass
class EmissionSimulationResult:
    """Container returned by :class:`EmissionSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class EmissionSystemCalculator:
    """EN 15316-2:2017 space emission calculator.

    Parameters
    ----------
    input_data:
        Dictionary containing service-specific heating/cooling emission
        assumptions. Heating data are read from ``heating`` and cooling data
        from ``cooling``. See the example script for a compact configuration.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> EmissionSimulationResult:
        """Run the hourly emission calculation."""

        prepared = self._prepare_timeseries(data)
        results = prepared.copy()
        results = pd.concat(
            [
                results,
                self._simulate_service(prepared, "H"),
                self._simulate_service(prepared, "C"),
            ],
            axis=1,
        )
        summary = self._summarize(results)
        return EmissionSimulationResult(
            timeseries=results,
            summary=summary,
            inputs=dict(self.input_data),
        )

    def temperature_increase_K(self, service: str) -> float:
        """Return the equivalent internal-temperature variation for a service."""

        service_key = self._service_key(service)
        return float(self.services[service_key]["temperature_increase_K"])

    def equivalent_heating_setpoint_C(self, setpoint_C: float) -> float:
        """Heating setpoint modified by the EN 15316-2 equivalent temperature."""

        return float(setpoint_C) + self.temperature_increase_K("H")

    def equivalent_cooling_setpoint_C(self, setpoint_C: float) -> float:
        """Cooling setpoint modified by the EN 15316-2 equivalent temperature."""

        return float(setpoint_C) + self.temperature_increase_K("C")

    def _load_options(self) -> None:
        cfg = self.input_data
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.demand_unit = str(cfg.get("demand_unit", "kWh")).lower()
        if self.demand_unit not in {"wh", "kwh"}:
            raise ValueError("demand_unit must be 'Wh' or 'kWh'.")

        self.cooling_solar_gain_temperature_C = float(
            cfg.get("cooling_solar_gain_temperature_C", 8.0)
        )
        self.default_heating_internal_temperature_C = float(
            cfg.get("default_heating_internal_temperature_C", 20.0)
        )
        self.default_cooling_internal_temperature_C = float(
            cfg.get("default_cooling_internal_temperature_C", 26.0)
        )

        self.services = {
            "H": self._service_options("heating", cfg.get("heating", {})),
            "C": self._service_options("cooling", cfg.get("cooling", {})),
        }

    def _service_options(self, name: str, data: dict[str, Any]) -> dict[str, float]:
        cfg = dict(data or {})
        components = {
            "stratification_K": float(cfg.get("stratification_K", cfg.get("str_K", 0.0))),
            "control_K": float(cfg.get("control_K", cfg.get("ctr_K", 0.0))),
            "radiation_K": float(cfg.get("radiation_K", cfg.get("rad_K", 0.0))),
            "hydraulic_balancing_K": float(
                cfg.get("hydraulic_balancing_K", cfg.get("hydr_K", 0.0))
            ),
            "room_automation_K": float(
                cfg.get("room_automation_K", cfg.get("roomaut_K", 0.0))
            ),
        }

        return {
            **components,
            "temperature_increase_K": float(sum(components.values())),
            "embedded_K": float(cfg.get("embedded_K", cfg.get("emb_K", 0.0))),
            "nominal_power_kW": float(cfg.get("nominal_power_kW", np.inf)),
            "fan_power_W": max(float(cfg.get("fan_power_W", 0.0)), 0.0),
            "fan_count": max(float(cfg.get("fan_count", cfg.get("n_fans", 0.0))), 0.0),
            "electric_control_aux": bool(cfg.get("electric_control_aux", False)),
            "backup_available": bool(cfg.get("backup_available", False)),
            "notify_inc_fallback": bool(cfg.get("notify_inc_fallback", True)),
            "undersize_tolerance": float(cfg.get("undersize_tolerance", 0.05)),
            "control_power_W": max(float(cfg.get("control_power_W", 0.0)), 0.0),
            "control_count": max(
                float(cfg.get("control_count", cfg.get("n_controls", 0.0))), 0.0
            ),
            "convective_fraction": _fraction(
                cfg.get("convective_fraction", cfg.get("f_em_conv", 0.7)),
                f"{name}.convective_fraction",
            ),
        }

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        out = pd.DataFrame(index=df.index)
        out["hours"] = self._time_step_hours(df)
        out["T_ext_C"] = _required_series(
            df,
            ["T_ext", "theta_ext", "outdoor_temperature_C", "T_external_C"],
            "outdoor temperature",
        ).astype(float)

        out["T_H_int_ini_C"] = _series_from_aliases(
            df,
            ["T_H_int_ini_C", "theta_H_int_ini_C", "T_op", "operative_temperature_C"],
            default=self.default_heating_internal_temperature_C,
        ).astype(float)
        out["T_C_int_ini_C"] = _series_from_aliases(
            df,
            ["T_C_int_ini_C", "theta_C_int_ini_C", "T_op", "operative_temperature_C"],
            default=self.default_cooling_internal_temperature_C,
        ).astype(float)

        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        out["Q_H_em_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_H_kWh", "QH_kWh", "heating_kWh", "space_heating_kWh"],
            raw_aliases=["Q_H", "Q_h", "Heating_needs"],
            fallback=np.maximum(q_hc, 0.0),
        )
        out["Q_C_em_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_C_kWh", "QC_kWh", "cooling_kWh", "space_cooling_kWh"],
            raw_aliases=["Q_C", "Cooling_needs"],
            fallback=np.maximum(-q_hc, 0.0),
        )

        out["Q_H_em_out_inc_input_kWh"] = _series_from_aliases(
            df,
            ["Q_H_em_out_inc_kWh", "Q_H_inc_kWh", "QH_inc_kWh", "heating_inc_kWh"],
            default=np.nan,
        ).astype(float)
        out["Q_C_em_out_inc_input_kWh"] = _series_from_aliases(
            df,
            ["Q_C_em_out_inc_kWh", "Q_C_inc_kWh", "QC_inc_kWh", "cooling_inc_kWh"],
            default=np.nan,
        ).astype(float)

        for col in ["Q_H_em_out_kWh", "Q_C_em_out_kWh"]:
            out.loc[:, col] = out[col].fillna(0.0).clip(lower=0.0)
        if "t_h_rl" in df.columns:
            out["t_h_rl"] = pd.to_numeric(df["t_h_rl"], errors="coerce").fillna(0.0).clip(lower=0.0)
        return out

    def _demand_from_columns(
        self,
        df: pd.DataFrame,
        kwh_aliases: list[str],
        raw_aliases: list[str],
        fallback: pd.Series | np.ndarray | float,
    ) -> pd.Series:
        series = _series_from_aliases(df, kwh_aliases, default=None)
        if series is not None:
            return series.astype(float).clip(lower=0.0)

        raw = _series_from_aliases(df, raw_aliases, default=None)
        if raw is not None:
            values = raw.astype(float)
            if self.demand_unit == "wh":
                values = values / 1000.0
            return values.clip(lower=0.0)

        if isinstance(fallback, pd.Series):
            return fallback.reindex(df.index).astype(float).clip(lower=0.0)
        return pd.Series(fallback, index=df.index, dtype=float).clip(lower=0.0)

    def _simulate_service(self, prepared: pd.DataFrame, service: str) -> pd.DataFrame:
        opts = self.services[service]
        prefix = "H" if service == "H" else "C"
        is_heating = service == "H"

        q_out = prepared[f"Q_{prefix}_em_out_kWh"].astype(float)
        hours = prepared["hours"].astype(float)
        theta_ini = prepared[f"T_{prefix}_int_ini_C"].astype(float)
        delta = float(opts["temperature_increase_K"])
        emb = float(opts["embedded_K"])
        theta_inc = theta_ini + delta
        e_comb = prepared["T_ext_C"].astype(float)
        if not is_heating:
            e_comb = e_comb + self.cooling_solar_gain_temperature_C

        denom_base = theta_ini - e_comb
        denom_inc = theta_inc - e_comb
        denom_base = denom_base.where(denom_base.abs() > _KWH_EPS, np.nan)
        denom_inc = denom_inc.where(denom_inc.abs() > _KWH_EPS, np.nan)

        q_inc_input = prepared[f"Q_{prefix}_em_out_inc_input_kWh"]
        q_inc_approx = q_out * (denom_inc / denom_base)
        fallback_rows = q_inc_input.isna() & (q_out > _KWH_EPS)
        if bool(fallback_rows.any()) and bool(opts["notify_inc_fallback"]):
            warnings.warn(
                f"EN 15316-2 {service}: Q_{prefix}_em_out_inc_kWh not supplied for {int(fallback_rows.sum())} "
                "step(s); the emission output at the modified set point is estimated by temperature ratios "
                "(engineering fallback, not the M2-2 recalculation required by the standard).",
                UserWarning,
                stacklevel=2,
            )
        q_inc = q_inc_input.where(q_inc_input.notna(), q_inc_approx)
        q_inc = q_inc.replace([np.inf, -np.inf], np.nan).fillna(q_out).clip(lower=0.0)

        nominal_power = float(opts["nominal_power_kW"])
        if np.isfinite(nominal_power) and nominal_power > 0:
            q_inc = np.minimum(q_inc, nominal_power * hours)

        # EN 15316-1 eq. 16: when the emitter cannot deliver the expected output,
        # the difference is backup energy (or unmet load if no backup exists).
        q_act = np.minimum(q_out, q_inc)
        q_deficit = (q_out - q_act).clip(lower=0.0)
        backup = bool(opts["backup_available"])
        q_back = q_deficit if backup else q_deficit * 0.0
        q_unmet = q_deficit * 0.0 if backup else q_deficit
        undersized = (q_deficit > float(opts["undersize_tolerance"]) * q_out) & (q_out > _KWH_EPS)
        if not backup and bool(undersized.any()):
            warnings.warn(
                f"EN 15316-2 {service}: emitter undersized in {int(undersized.sum())} hour(s); "
                f"largest unmet load {float(q_deficit.max()):.3f} kWh with no backup or supplementary "
                "system declared (backup_available=false).",
                RuntimeWarning,
                stacklevel=2,
            )

        q_emb_ls = (q_inc * emb / denom_inc).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        q_temp_effect = q_inc - q_out
        q_em_ls = q_temp_effect + q_emb_ls
        q_em_in = q_out + q_em_ls

        # EN 15316-2 eq. 14: W_fan = n_fan * P_fan * t_h,rL / 1000, where t_h,rL is the
        # operation time of the system in the period (EN 15316-1 operating time). If not given,
        # the run time is estimated from the load factor (emitted energy / nominal power).
        if "t_h_rl" in prepared.columns:
            operating_hours = prepared["t_h_rl"].astype(float).clip(upper=hours)
        elif np.isfinite(nominal_power) and nominal_power > 0:
            capacity = (nominal_power * hours).replace(0.0, np.nan)
            operating_hours = hours * (q_em_in / capacity).clip(lower=0.0, upper=1.0).fillna(0.0)
        else:
            operating_hours = hours
        operating_hours = operating_hours.where(q_em_in > _KWH_EPS, 0.0)
        w_fan = opts["fan_count"] * opts["fan_power_W"] * operating_hours / 1000.0
        if opts["electric_control_aux"]:
            w_control = (
                opts["control_count"] * opts["control_power_W"] * operating_hours / 1000.0
            )
        else:
            w_control = pd.Series(0.0, index=prepared.index)
        w_aux = w_fan + w_control

        out = pd.DataFrame(index=prepared.index)
        out[f"theta_{prefix}_int_inc_delta_K"] = delta
        out[f"theta_{prefix}_int_inc_C"] = theta_inc
        out[f"theta_{prefix}_e_comb_C"] = e_comb
        out[f"Q_{prefix}_em_out_inc_kWh"] = q_inc
        out[f"Q_{prefix}_em_temp_effect_kWh"] = q_temp_effect
        out[f"Q_{prefix}_emb_ls_kWh"] = q_emb_ls
        out[f"Q_{prefix}_em_ls_kWh"] = q_em_ls
        out[f"Q_{prefix}_em_back_out_kWh"] = q_back
        out[f"Q_{prefix}_em_unmet_kWh"] = q_unmet
        out[f"Q_{prefix}_em_in_kWh"] = q_em_in
        out[f"W_{prefix}_em_fan_aux_kWh"] = w_fan
        out[f"W_{prefix}_em_control_aux_kWh"] = w_control
        out[f"W_{prefix}_em_aux_kWh"] = w_aux
        out[f"f_{prefix}_em_conv"] = float(opts["convective_fraction"])
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

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        q_h_out = s("Q_H_em_out_kWh")
        q_c_out = s("Q_C_em_out_kWh")
        q_h_ls = s("Q_H_em_ls_kWh")
        q_c_ls = s("Q_C_em_ls_kWh")

        return {
            "QH_em_out_kWh": q_h_out,
            "QH_em_out_inc_kWh": s("Q_H_em_out_inc_kWh"),
            "QH_em_temp_effect_kWh": s("Q_H_em_temp_effect_kWh"),
            "QH_emb_ls_kWh": s("Q_H_emb_ls_kWh"),
            "QH_em_ls_kWh": q_h_ls,
            "QH_em_in_kWh": s("Q_H_em_in_kWh"),
            "WH_em_aux_kWh": s("W_H_em_aux_kWh"),
            "theta_H_int_inc_K": self.temperature_increase_K("H"),
            "theta_H_int_inc_mean_C": _weighted_mean(
                results["theta_H_int_inc_C"], results["Q_H_em_out_kWh"] + _KWH_EPS
            ),
            "fH_em_conv": float(self.services["H"]["convective_fraction"]),
            "e_H_em_ls_an": _ratio(q_h_out + q_h_ls, q_h_out),
            "QH_em_back_out_kWh": s("Q_H_em_back_out_kWh"),
            "QH_em_unmet_kWh": s("Q_H_em_unmet_kWh"),
            "QC_em_out_kWh": q_c_out,
            "QC_em_out_inc_kWh": s("Q_C_em_out_inc_kWh"),
            "QC_em_temp_effect_kWh": s("Q_C_em_temp_effect_kWh"),
            "QC_emb_ls_kWh": s("Q_C_emb_ls_kWh"),
            "QC_em_ls_kWh": q_c_ls,
            "QC_em_in_kWh": s("Q_C_em_in_kWh"),
            "WC_em_aux_kWh": s("W_C_em_aux_kWh"),
            "theta_C_int_inc_K": self.temperature_increase_K("C"),
            "theta_C_int_inc_mean_C": _weighted_mean(
                results["theta_C_int_inc_C"], results["Q_C_em_out_kWh"] + _KWH_EPS
            ),
            "fC_em_conv": float(self.services["C"]["convective_fraction"]),
            "e_C_em_ls_an": _ratio(q_c_out + q_c_ls, q_c_out),
            "QC_em_back_out_kWh": s("Q_C_em_back_out_kWh"),
            "QC_em_unmet_kWh": s("Q_C_em_unmet_kWh"),
            "W_em_aux_kWh": s("W_H_em_aux_kWh") + s("W_C_em_aux_kWh"),
            "Q_em_ls_kWh": q_h_ls + q_c_ls,
        }

    @staticmethod
    def _service_key(service: str) -> str:
        service_lower = str(service).lower()
        if service_lower in {"h", "heating", "space_heating"}:
            return "H"
        if service_lower in {"c", "cooling", "space_cooling"}:
            return "C"
        raise ValueError("service must be 'H'/'heating' or 'C'/'cooling'.")


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


def _required_series(df: pd.DataFrame, aliases: list[str], label: str) -> pd.Series:
    series = _series_from_aliases(df, aliases, default=None)
    if series is None:
        raise ValueError(f"{label} is required. Provide one of: {aliases}.")
    return series


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _fraction(value: Any, name: str) -> float:
    try:
        fraction = float(value)
    except Exception as exc:  # pragma: no cover - defensive type context
        raise ValueError(f"{name} must be a fraction between 0 and 1.") from exc
    return float(np.clip(fraction, 0.0, 1.0))


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return float(numerator / denominator)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.Series(values, dtype=float)
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    mask = v.notna() & w.notna()
    if not mask.any() or float(w[mask].sum()) <= _KWH_EPS:
        return float(v[mask].mean()) if mask.any() else 0.0
    return float(np.average(v[mask], weights=w[mask]))
