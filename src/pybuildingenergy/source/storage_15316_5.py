"""Heating and DHW storage calculation according to EN 15316-5:2017.

The module implements the single-volume Method B calculation for space-heating
and domestic hot water storage systems, modules M3-7 and M8-7. It is designed to
sit between distribution and generation: users provide the thermal energy
required downstream of the storage and the calculator returns the thermal input
that has to be supplied by the generator, storage standing losses, recoverable
losses and storage pump auxiliary energy.

The implementation follows EN 15316-5:2017 for storage systems controlled at a
constant set temperature during the calculation step:

* storage thermal losses from H_sto_ls, setpoint, ambient temperature and time;
* optional derivation of H_sto_ls from a declared daily standby loss;
* auxiliary pump operation time from transferred heat, water flow and
  generator temperature difference;
* recoverable and medium-recovered auxiliary energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12
_WATER_HEAT_CAPACITY_DENSITY_KWH_M3K = 1.15


@dataclass
class StorageSimulationResult:
    """Container returned by :class:`StorageSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class StorageSystemCalculator:
    """EN 15316-5:2017 heating and DHW storage calculator."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> StorageSimulationResult:
        """Run the hourly storage calculation."""

        prepared = self._prepare_timeseries(data)
        output_columns = []
        for service in ["H", "W"]:
            output_columns.extend(
                [
                    f"theta_{service}_sto_set_C",
                    f"theta_{service}_sto_out_C",
                    f"theta_{service}_sto_amb_C",
                    f"V_{service}_sto_l",
                    f"H_{service}_sto_ls_W_K",
                    f"Q_{service}_sto_ls_kWh",
                    f"Q_{service}_sto_ls_rbl_kWh",
                    f"Q_{service}_sto_ls_nrbl_kWh",
                    f"W_{service}_sto_aux_kWh",
                    f"Q_{service}_sto_aux_rvd_kWh",
                    f"Q_{service}_sto_aux_rbl_kWh",
                    f"Q_{service}_sto_rbl_kWh",
                    f"Q_{service}_sto_in_kWh",
                    f"t_{service}_sto_pmp_in_h",
                    f"t_{service}_sto_pmp_out_h",
                ]
            )
        results = prepared.drop(columns=output_columns, errors="ignore")
        results = pd.concat(
            [
                results,
                self._simulate_service(prepared, "H"),
                self._simulate_service(prepared, "W"),
            ],
            axis=1,
        )
        summary = self._summarize(results)
        return StorageSimulationResult(
            timeseries=results,
            summary=summary,
            inputs=dict(self.input_data),
        )

    def _load_options(self) -> None:
        cfg = self.input_data
        self.dynamic_calculation = bool(cfg.get("dynamic_calculation", False))
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.demand_unit = str(cfg.get("demand_unit", "kWh")).lower()
        if self.demand_unit not in {"wh", "kwh"}:
            raise ValueError("demand_unit must be 'Wh' or 'kWh'.")

        self.services = {
            "H": self._service_options("heating", cfg.get("heating", {})),
            "W": self._service_options("dhw", cfg.get("dhw", cfg.get("domestic_hot_water", {}))),
        }

    def _service_options(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(data or {})
        set_default = 45.0 if name == "heating" else 55.0
        setpoint = float(cfg.get("set_temperature_C", cfg.get("storage_setpoint_C", set_default)))
        h_loss = self._standby_loss_coefficient(cfg, setpoint)

        return {
            "enabled": bool(cfg.get("enabled", True)),
            "set_temperature_C": setpoint,
            "output_temperature_C": float(cfg.get("output_temperature_C", setpoint)),
            "ambient_temperature_C": float(cfg.get("ambient_temperature_C", 16.0)),
            "storage_volume_l": max(float(cfg.get("storage_volume_l", 0.0)), 0.0),
            "storage_height_m": max(float(cfg.get("storage_height_m", 1.2)), 0.1),
            "sensor_height_m": max(float(cfg.get("sensor_height_m", 0.6)), 0.0),
            "cold_inlet_temperature_C": float(cfg.get("cold_inlet_temperature_C", 10.0)),
            "stratification_efficiency": _fraction(
                cfg.get("stratification_efficiency", 0.85),
                f"{name}.stratification_efficiency",
            ),
            "dhw_switch_temperature_C": float(cfg.get("dhw_switch_temperature_C", 45.0)),
            "recharge_power_kW": max(
                float(cfg.get("recharge_power_kW", cfg.get("nominal_power_kW", 0.0))),
                0.0,
            ),
            "recharge_cop": max(float(cfg.get("recharge_cop", 3.0)), _KWH_EPS),
            "standby_loss_coefficient_W_K": h_loss,
            "standby_loss_adaptation_factor": _fraction(
                cfg.get("standby_loss_adaptation_factor", cfg.get("f_sto_bac_acc", 1.0)),
                f"{name}.standby_loss_adaptation_factor",
            ),
            "connection_loss_factor": max(
                float(cfg.get("connection_loss_factor", cfg.get("f_sto_dis_ls", 1.0))),
                0.0,
            ),
            "thermal_loss_room_fraction": _fraction(
                cfg.get("thermal_loss_room_fraction", cfg.get("f_sto_m", 0.75)),
                f"{name}.thermal_loss_room_fraction",
            ),
            "auxiliary_to_medium_fraction": _fraction(
                cfg.get("auxiliary_to_medium_fraction", cfg.get("f_rvd_aux", 0.25)),
                f"{name}.auxiliary_to_medium_fraction",
            ),
            "input_pump_power_kW": max(float(cfg.get("input_pump_power_kW", 0.0)), 0.0),
            "input_pump_flow_m3_h": max(float(cfg.get("input_pump_flow_m3_h", 0.0)), 0.0),
            "input_pump_deltaT_K": max(float(cfg.get("input_pump_deltaT_K", 10.0)), _KWH_EPS),
            "output_pump_power_kW": max(float(cfg.get("output_pump_power_kW", 0.0)), 0.0),
            "output_pump_flow_m3_h": max(float(cfg.get("output_pump_flow_m3_h", 0.0)), 0.0),
            "output_pump_deltaT_K": max(float(cfg.get("output_pump_deltaT_K", 10.0)), _KWH_EPS),
            "operation_mode": str(cfg.get("operation_mode", "demand")).lower(),
        }

    def _standby_loss_coefficient(self, cfg: dict[str, Any], setpoint: float) -> float:
        direct = cfg.get("standby_loss_coefficient_W_K", cfg.get("H_sto_ls_W_K"))
        if direct is not None:
            return max(float(direct), 0.0)

        loss_ref = cfg.get("standby_loss_kWh_per_day_ref", cfg.get("standby_loss_kWh_per_day"))
        if loss_ref is not None:
            theta_ref = float(cfg.get("standby_set_temperature_ref_C", setpoint))
            theta_amb_ref = float(cfg.get("standby_ambient_temperature_ref_C", 20.0))
            delta_ref = max(theta_ref - theta_amb_ref, _KWH_EPS)
            return max(float(loss_ref), 0.0) * 1000.0 / (24.0 * delta_ref)

        storage_type = str(cfg.get("storage_type", "")).lower()
        volume_l = max(float(cfg.get("storage_volume_l", 0.0)), 0.0)
        if storage_type and volume_l > 0.0:
            return _annex_b_loss_coefficient(storage_type, volume_l)

        return 2.0

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        out = pd.DataFrame(index=df.index)
        out["hours"] = self._time_step_hours(df)
        out["T_ext_C"] = _series_from_aliases(
            df, ["T_ext", "theta_ext", "outdoor_temperature_C", "T_external_C"], default=np.nan
        ).astype(float)

        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        out["Q_H_sto_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_H_sto_out_kWh",
                "Q_H_dis_in_kWh",
                "Q_H_kWh",
                "QH_kWh",
                "space_heating_kWh",
            ],
            raw_aliases=["Q_H", "Q_h", "Heating_needs"],
            fallback=np.maximum(q_hc, 0.0),
        )
        out["Q_W_sto_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_W_sto_out_kWh",
                "Q_W_dis_in_kWh",
                "Q_W_kWh",
                "QW_kWh",
                "Q_DHW_kWh",
                "DHW_kWh",
                "dhw_kWh",
            ],
            raw_aliases=["Q_W", "Q_DHW"],
            fallback=0.0,
        )

        for col in ["Q_H_sto_out_kWh", "Q_W_sto_out_kWh"]:
            out.loc[:, col] = out[col].fillna(0.0).clip(lower=0.0)

        temperature_columns = [
            "theta_H_sto_set_C",
            "theta_W_sto_set_C",
            "theta_H_sto_amb_C",
            "theta_W_sto_amb_C",
            "T_H_sink_C",
            "T_W_sink_C",
            "dhw_sink_temperature_C",
            "storage_ambient_temperature_C",
        ]
        for col in temperature_columns:
            if col in df.columns:
                out.loc[:, col] = df[col].astype(float)

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
        q_out = prepared[f"Q_{service}_sto_out_kWh"].astype(float)
        hours = prepared["hours"].astype(float)

        if not opts["enabled"]:
            return self._pass_through(prepared.index, service, q_out)

        if self.dynamic_calculation and service == "W":
            return self._simulate_service_dynamic(prepared, service)

        setpoint = self._setpoint_temperature(prepared, service, opts)
        ambient = self._ambient_temperature(prepared, service, opts)
        loss = self._thermal_loss(setpoint, ambient, hours, opts)
        pump = self._pump_auxiliary(q_out, loss, hours, opts)

        aux_rvd = pump["W_aux_kWh"] * opts["auxiliary_to_medium_fraction"]
        aux_rbl = (
            pump["W_aux_kWh"]
            * (1.0 - opts["auxiliary_to_medium_fraction"])
            * opts["thermal_loss_room_fraction"]
        )
        loss_rbl = loss * opts["thermal_loss_room_fraction"]
        loss_nrbl = loss - loss_rbl
        q_in = (q_out + loss - aux_rvd).clip(lower=0.0)

        out = pd.DataFrame(index=prepared.index)
        out[f"theta_{service}_sto_set_C"] = setpoint
        out[f"theta_{service}_sto_out_C"] = self._output_temperature(prepared, service, opts)
        out[f"theta_{service}_sto_amb_C"] = ambient
        out[f"V_{service}_sto_l"] = opts["storage_volume_l"]
        out[f"H_{service}_sto_ls_W_K"] = opts["standby_loss_coefficient_W_K"]
        out[f"Q_{service}_sto_ls_kWh"] = loss
        out[f"Q_{service}_sto_ls_rbl_kWh"] = loss_rbl
        out[f"Q_{service}_sto_ls_nrbl_kWh"] = loss_nrbl
        out[f"W_{service}_sto_aux_kWh"] = pump["W_aux_kWh"]
        out[f"Q_{service}_sto_aux_rvd_kWh"] = aux_rvd
        out[f"Q_{service}_sto_aux_rbl_kWh"] = aux_rbl
        out[f"Q_{service}_sto_rbl_kWh"] = loss_rbl + aux_rbl
        out[f"Q_{service}_sto_in_kWh"] = q_in
        out[f"t_{service}_sto_pmp_in_h"] = pump["t_input_h"]
        out[f"t_{service}_sto_pmp_out_h"] = pump["t_output_h"]
        return out

    def _simulate_service_dynamic(self, prepared: pd.DataFrame, service: str) -> pd.DataFrame:
        opts = self.services[service]
        q_out = prepared[f"Q_{service}_sto_out_kWh"].astype(float)
        hours = prepared["hours"].astype(float)
        setpoint = self._setpoint_temperature(prepared, service, opts)
        ambient = self._ambient_temperature(prepared, service, opts)
        cold_inlet = _series_from_aliases(
            prepared,
            ["cold_inlet_temperature_C", "T_W_cold_C", "T_cold_C"],
            default=opts["cold_inlet_temperature_C"],
        ).astype(float)

        total_volume = max(opts["storage_volume_l"], 0.0)
        if total_volume <= _KWH_EPS:
            return self._pass_through(prepared.index, service, q_out)

        current_volume_l = total_volume
        top_temp = float(setpoint.iloc[0])
        bottom_temp = float(cold_inlet.iloc[0])
        rho_cp = _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K

        rows = []
        for idx, demand_kWh, step_h, sp, amb, cold in zip(
            prepared.index,
            q_out.to_numpy(dtype=float),
            hours.to_numpy(dtype=float),
            setpoint.to_numpy(dtype=float),
            ambient.to_numpy(dtype=float),
            cold_inlet.to_numpy(dtype=float),
        ):
            avg_temp_before = (top_temp + bottom_temp) / 2.0
            current_volume_m3 = max(current_volume_l / 1000.0, _KWH_EPS)
            loss_kWh = max(opts["standby_loss_coefficient_W_K"] * max(avg_temp_before - amb, 0.0) * step_h / 1000.0, 0.0)
            if loss_kWh > 0.0:
                avg_temp_before = max(cold, avg_temp_before - loss_kWh / max(rho_cp * current_volume_m3, _KWH_EPS))
                stratification_gap = max(avg_temp_before - cold, 0.0) * (1.0 - opts["stratification_efficiency"])
                top_temp = avg_temp_before + stratification_gap / 2.0
                bottom_temp = max(cold, avg_temp_before - stratification_gap / 2.0)

            demand_energy = max(demand_kWh, 0.0)
            demand_volume_m3 = demand_energy / max(rho_cp * max(sp - cold, _KWH_EPS), _KWH_EPS)
            demand_volume_l = demand_volume_m3 * 1000.0
            draw_volume_l = min(demand_volume_l, current_volume_l)
            current_volume_l = max(current_volume_l - draw_volume_l, 0.0)

            pre_draw_top_temp = top_temp
            pre_draw_bottom_temp = bottom_temp

            if draw_volume_l > 0.0:
                energy_removed_kWh = min(demand_energy, rho_cp * max((current_volume_l + draw_volume_l) / 1000.0, _KWH_EPS) * max(avg_temp_before - cold, 0.0))
                temp_drop_K = energy_removed_kWh / max(rho_cp * max(current_volume_m3, _KWH_EPS), _KWH_EPS)
                avg_temp_after_draw = max(cold, avg_temp_before - temp_drop_K)
                stratification_gap = max(avg_temp_after_draw - cold, 0.0) * (1.0 - opts["stratification_efficiency"])
                top_temp = avg_temp_after_draw + stratification_gap / 2.0
                bottom_temp = max(cold, avg_temp_after_draw - stratification_gap / 2.0)
            else:
                avg_temp_after_draw = avg_temp_before

            post_draw_top_temp = top_temp
            post_draw_bottom_temp = bottom_temp

            volume_before_dhw_mode_l = current_volume_l
            dhw_mode_flag = avg_temp_after_draw < opts["dhw_switch_temperature_C"]
            avg_temp_before_recharge = avg_temp_after_draw
            if dhw_mode_flag:
                current_volume_l = total_volume
                stratification_gap = max(avg_temp_before_recharge - cold, 0.0) * (1.0 - opts["stratification_efficiency"])
                top_temp = avg_temp_before_recharge + stratification_gap / 2.0
                bottom_temp = max(cold, avg_temp_before_recharge - stratification_gap / 2.0)

            current_volume_m3 = max(current_volume_l / 1000.0, _KWH_EPS)
            energy_to_setpoint_kWh = max((sp - avg_temp_before_recharge), 0.0) * current_volume_m3 * rho_cp if dhw_mode_flag else 0.0
            recharge_power_kW = max(opts["recharge_power_kW"], _KWH_EPS)
            recharge_time_h = min(step_h, energy_to_setpoint_kWh / recharge_power_kW) if energy_to_setpoint_kWh > 0 else 0.0
            recharge_energy_kWh = min(energy_to_setpoint_kWh, recharge_power_kW * step_h)
            if recharge_energy_kWh > 0.0:
                temp_rise_K = recharge_energy_kWh / max(rho_cp * current_volume_m3, _KWH_EPS)
                avg_temp_after_recharge = min(sp, avg_temp_before_recharge + temp_rise_K)
                stratification_gap = max(avg_temp_after_recharge - cold, 0.0) * (1.0 - opts["stratification_efficiency"])
                top_temp = avg_temp_after_recharge + stratification_gap / 2.0
                bottom_temp = max(cold, avg_temp_after_recharge - stratification_gap / 2.0)
            else:
                avg_temp_after_recharge = avg_temp_before_recharge

            recharge_electricity_kWh = recharge_energy_kWh / opts["recharge_cop"] if recharge_energy_kWh > 0.0 else 0.0

            rows.append(
                {
                    f"theta_{service}_sto_set_C": sp,
                    f"theta_{service}_sto_out_C": avg_temp_after_recharge,
                    f"theta_{service}_sto_amb_C": amb,
                    f"V_{service}_sto_l": current_volume_l,
                    f"H_{service}_sto_ls_W_K": opts["standby_loss_coefficient_W_K"],
                    f"Q_{service}_sto_ls_kWh": loss_kWh,
                    f"Q_{service}_sto_ls_rbl_kWh": loss_kWh * opts["thermal_loss_room_fraction"],
                    f"Q_{service}_sto_ls_nrbl_kWh": loss_kWh * (1.0 - opts["thermal_loss_room_fraction"]),
                    f"W_{service}_sto_aux_kWh": recharge_electricity_kWh,
                    f"Q_{service}_sto_aux_rvd_kWh": 0.0,
                    f"Q_{service}_sto_aux_rbl_kWh": 0.0,
                    f"Q_{service}_sto_rbl_kWh": loss_kWh * opts["thermal_loss_room_fraction"],
                    f"Q_{service}_sto_in_kWh": recharge_energy_kWh,
                    f"t_{service}_sto_pmp_in_h": recharge_time_h,
                    f"t_{service}_sto_pmp_out_h": 0.0,
                    f"theta_{service}_sto_pre_draw_C": pre_draw_top_temp,
                    f"theta_{service}_sto_post_draw_C": avg_temp_after_draw,
                    f"theta_{service}_sto_top_C": top_temp,
                    f"theta_{service}_sto_bottom_C": bottom_temp,
                    f"Q_{service}_sto_recharge_time_h": recharge_time_h,
                    f"Q_{service}_sto_recharge_energy_kWh": recharge_energy_kWh,
                    f"Q_{service}_sto_stratification_K": max(top_temp - bottom_temp, 0.0),
                    f"V_{service}_sto_before_dhw_mode_l": volume_before_dhw_mode_l,
                    f"V_{service}_sto_refill_l": 0.0,
                    f"V_{service}_sto_remaining_l": current_volume_l,
                    f"dhw_dhw_mode_flag": bool(dhw_mode_flag),
                }
            )

        return pd.DataFrame(rows, index=prepared.index)

    def _pass_through(
        self, index: pd.Index, service: str, q_out: pd.Series
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=index)
        out[f"theta_{service}_sto_set_C"] = np.nan
        out[f"theta_{service}_sto_out_C"] = np.nan
        out[f"theta_{service}_sto_amb_C"] = np.nan
        out[f"V_{service}_sto_l"] = 0.0
        out[f"H_{service}_sto_ls_W_K"] = 0.0
        out[f"Q_{service}_sto_ls_kWh"] = 0.0
        out[f"Q_{service}_sto_ls_rbl_kWh"] = 0.0
        out[f"Q_{service}_sto_ls_nrbl_kWh"] = 0.0
        out[f"W_{service}_sto_aux_kWh"] = 0.0
        out[f"Q_{service}_sto_aux_rvd_kWh"] = 0.0
        out[f"Q_{service}_sto_aux_rbl_kWh"] = 0.0
        out[f"Q_{service}_sto_rbl_kWh"] = 0.0
        out[f"Q_{service}_sto_in_kWh"] = q_out
        out[f"t_{service}_sto_pmp_in_h"] = 0.0
        out[f"t_{service}_sto_pmp_out_h"] = 0.0
        return out

    def _setpoint_temperature(
        self, prepared: pd.DataFrame, service: str, opts: dict[str, Any]
    ) -> pd.Series:
        aliases = [f"theta_{service}_sto_set_C"]
        if service == "H":
            aliases += ["T_H_sink_C"]
        else:
            aliases += ["T_W_sink_C", "dhw_sink_temperature_C"]
        return _series_from_aliases(
            prepared, aliases, default=opts["set_temperature_C"]
        ).astype(float)

    def _output_temperature(
        self, prepared: pd.DataFrame, service: str, opts: dict[str, Any]
    ) -> pd.Series:
        aliases = [f"theta_{service}_sto_out_C"]
        if service == "H":
            aliases += ["T_H_sink_C"]
        else:
            aliases += ["T_W_sink_C", "dhw_sink_temperature_C"]
        return _series_from_aliases(
            prepared, aliases, default=opts["output_temperature_C"]
        ).astype(float)

    def _ambient_temperature(
        self, prepared: pd.DataFrame, service: str, opts: dict[str, Any]
    ) -> pd.Series:
        return _series_from_aliases(
            prepared,
            [f"theta_{service}_sto_amb_C", "storage_ambient_temperature_C"],
            default=opts["ambient_temperature_C"],
        ).astype(float)

    def _thermal_loss(
        self,
        setpoint: pd.Series,
        ambient: pd.Series,
        hours: pd.Series,
        opts: dict[str, Any],
    ) -> pd.Series:
        delta = (setpoint - ambient).clip(lower=0.0)
        return (
            opts["standby_loss_coefficient_W_K"]
            * opts["standby_loss_adaptation_factor"]
            * opts["connection_loss_factor"]
            * delta
            * hours
            / 1000.0
        )

    def _pump_auxiliary(
        self,
        q_out: pd.Series,
        loss: pd.Series,
        hours: pd.Series,
        opts: dict[str, Any],
    ) -> dict[str, pd.Series]:
        q_to_storage = q_out + loss
        t_input = self._pump_runtime(
            q_to_storage,
            hours,
            opts["input_pump_flow_m3_h"],
            opts["input_pump_deltaT_K"],
            opts["operation_mode"],
        )
        t_output = self._pump_runtime(
            q_out,
            hours,
            opts["output_pump_flow_m3_h"],
            opts["output_pump_deltaT_K"],
            opts["operation_mode"],
        )
        w_aux = (
            t_input * opts["input_pump_power_kW"]
            + t_output * opts["output_pump_power_kW"]
        )
        return {"W_aux_kWh": w_aux, "t_input_h": t_input, "t_output_h": t_output}

    def _pump_runtime(
        self,
        q_kWh: pd.Series,
        hours: pd.Series,
        flow_m3_h: float,
        deltaT_K: float,
        operation_mode: str,
    ) -> pd.Series:
        if flow_m3_h <= _KWH_EPS:
            return pd.Series(0.0, index=q_kWh.index)
        if operation_mode == "continuous":
            return hours
        power_kW = _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * flow_m3_h * deltaT_K
        return (q_kWh / power_kW).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        summary: dict[str, float] = {"hours": float(results["hours"].sum())}
        for service in ["H", "W"]:
            prefix = "QH" if service == "H" else "QW"
            wprefix = "WH" if service == "H" else "WW"
            summary[f"{prefix}_sto_out_kWh"] = float(
                results[f"Q_{service}_sto_out_kWh"].sum()
            )
            summary[f"{prefix}_sto_in_kWh"] = float(
                results[f"Q_{service}_sto_in_kWh"].sum()
            )
            summary[f"{prefix}_sto_ls_kWh"] = float(
                results[f"Q_{service}_sto_ls_kWh"].sum()
            )
            summary[f"{prefix}_sto_ls_rbl_kWh"] = float(
                results[f"Q_{service}_sto_ls_rbl_kWh"].sum()
            )
            summary[f"{prefix}_sto_ls_nrbl_kWh"] = float(
                results[f"Q_{service}_sto_ls_nrbl_kWh"].sum()
            )
            summary[f"{wprefix}_sto_aux_kWh"] = float(
                results[f"W_{service}_sto_aux_kWh"].sum()
            )
            summary[f"{prefix}_sto_aux_rvd_kWh"] = float(
                results[f"Q_{service}_sto_aux_rvd_kWh"].sum()
            )
            summary[f"{prefix}_sto_aux_rbl_kWh"] = float(
                results[f"Q_{service}_sto_aux_rbl_kWh"].sum()
            )
            summary[f"{prefix}_sto_rbl_kWh"] = float(
                results[f"Q_{service}_sto_rbl_kWh"].sum()
            )
            summary[f"{prefix}_sto_X_in_kWh"] = float(results[f"Q_{service}_sto_in_kWh"].sum())
            summary[f"theta_{service}_sto_set_mean_C"] = _weighted_mean(
                results[f"theta_{service}_sto_set_C"],
                results["hours"],
            )
            summary[f"theta_{service}_sto_amb_mean_C"] = _weighted_mean(
                results[f"theta_{service}_sto_amb_C"],
                results["hours"],
            )

        summary.update(
            {
                "Q_sto_out_kWh": summary["QH_sto_out_kWh"] + summary["QW_sto_out_kWh"],
                "Q_sto_in_kWh": summary["QH_sto_in_kWh"] + summary["QW_sto_in_kWh"],
                "Q_sto_ls_kWh": summary["QH_sto_ls_kWh"] + summary["QW_sto_ls_kWh"],
                "Q_sto_ls_rbl_kWh": summary["QH_sto_ls_rbl_kWh"]
                + summary["QW_sto_ls_rbl_kWh"],
                "Q_sto_ls_nrbl_kWh": summary["QH_sto_ls_nrbl_kWh"]
                + summary["QW_sto_ls_nrbl_kWh"],
                "W_sto_aux_kWh": summary["WH_sto_aux_kWh"] + summary["WW_sto_aux_kWh"],
                "Q_sto_aux_rvd_kWh": summary["QH_sto_aux_rvd_kWh"]
                + summary["QW_sto_aux_rvd_kWh"],
                "Q_sto_aux_rbl_kWh": summary["QH_sto_aux_rbl_kWh"]
                + summary["QW_sto_aux_rbl_kWh"],
                "Q_sto_rbl_kWh": summary["QH_sto_rbl_kWh"] + summary["QW_sto_rbl_kWh"],
            }
        )

        for service in ["H", "W"]:
            prefix = "QH" if service == "H" else "QW"
            out = summary[f"{prefix}_sto_out_kWh"]
            summary[f"e_{service}_sto"] = (
                summary[f"{prefix}_sto_in_kWh"] / out if out > _KWH_EPS else np.nan
            )
        return summary

    def _time_step_hours(self, df: pd.DataFrame) -> pd.Series:
        if "time_step_hours" in df.columns:
            return df["time_step_hours"].astype(float).clip(lower=0.0)
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            delta = df.index.to_series().diff().dt.total_seconds().div(3600.0)
            inferred = float(delta.dropna().median())
            if np.isfinite(inferred) and inferred > 0:
                return pd.Series(inferred, index=df.index, dtype=float)
        return pd.Series(self.default_time_step_h, index=df.index, dtype=float)


def _annex_b_loss_coefficient(storage_type: str, volume_l: float) -> float:
    """Return H_sto_ls from EN 15316-5 Annex B storage-type correlations."""

    if "horizontal" in storage_type:
        c1, c2, c3, c4, c5 = 0.939, 0.0104, 1.0, 45.0, 24.0
    elif "small" in storage_type or volume_l < 75.0:
        c1, c2, c3, c4, c5 = 0.1474, 0.0719, 2.0 / 3.0, 45.0, 24.0
    elif "solar" in storage_type:
        c1, c2, c3, c4, c5 = 0.0, 0.16, 0.5, 1000.0, 1.0
    else:
        c1, c2, c3, c4, c5 = 0.224, 0.0663, 2.0 / 3.0, 45.0, 24.0
    return max((c1 + c2 * volume_l**c3) / (c4 * c5) * 1000.0, 0.0)


def _series_from_aliases(
    df: pd.DataFrame,
    aliases: list[str],
    default: float | int | pd.Series | None,
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
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _fraction(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return number


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))
