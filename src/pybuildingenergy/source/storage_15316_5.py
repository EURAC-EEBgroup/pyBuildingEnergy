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
        if self.method_a is not None:
            service_results = [self._simulate_method_a(prepared, data)]
        else:
            service_results = [
                self._simulate_service(prepared, "H"),
                self._simulate_service(prepared, "W"),
            ]
        results = pd.concat([results, *service_results], axis=1)
        summary = self._summarize(results)
        return StorageSimulationResult(
            timeseries=results,
            summary=summary,
            inputs=dict(self.input_data),
        )

    def _load_options(self) -> None:
        cfg = self.input_data
        self.dynamic_calculation = bool(cfg.get("dynamic_calculation", False))
        # Method A (prEN 15316-5:2024, 6.4.3): one multi-layer tank serving both
        # services; see LayeredStorage for the accepted keys.
        method_a = cfg.get("method_a")
        if method_a is not None and not isinstance(method_a, dict):
            raise TypeError("method_a must be a dict of LayeredStorage options.")
        if method_a is not None and "layer_volumes_l" not in method_a:
            raise ValueError("method_a needs layer_volumes_l (bottom to top).")
        self.method_a = None if method_a is None else dict(method_a)
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

    def _simulate_method_a(self, prepared: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Both services through one :class:`LayeredStorage` (Method A).

        Optional input columns: ``Q_sol_to_sto_kWh`` (heat offered by the solar loop,
        accepted up to the tank's capacity) and ``Q_W_dis_ncons_kWh`` (circulation
        loss drawn from the top layer). ``Q_{H,W}_sto_in_kWh`` is the back-up heater
        input, split between the services in proportion to the energy each
        delivered; solar heat is reported separately in ``Q_sto_solar_in_kWh``.
        Pump auxiliary energy is not modelled on this path.
        """
        cfg = dict(self.method_a)
        w = self.services["W"]
        hours = prepared["hours"].astype(float)
        if not np.allclose(hours, hours.iloc[0]):
            raise ValueError("Method A needs a constant calculation time step.")
        cfg.setdefault("set_point_C", w["set_temperature_C"])
        cfg.setdefault("cold_water_C", w["cold_inlet_temperature_C"])
        cfg.setdefault("ambient_C", w["ambient_temperature_C"])
        cfg.setdefault("loss_W_K", w["standby_loss_coefficient_W_K"])
        cfg["step_hours"] = float(hours.iloc[0])
        circ = _series_from_aliases(data, ["Q_W_dis_ncons_kWh"], default=0.0).astype(float).fillna(0.0)
        solar = _series_from_aliases(data, ["Q_sol_to_sto_kWh", "Q_sto_sol_in_req_kWh"], default=0.0).astype(float).fillna(0.0)
        cfg["dhw_demand_kWh"] = prepared["Q_W_sto_out_kWh"].to_numpy()
        cfg["space_heating_kWh"] = prepared["Q_H_sto_out_kWh"].to_numpy()
        cfg["circulation_kWh"] = circ.to_numpy()
        tank = LayeredStorage(**cfg)
        accepted = np.zeros(len(prepared))
        for i, q_sol in enumerate(solar.to_numpy()):
            _, accepted[i] = tank(i, float(q_sol), True)
        hist = pd.DataFrame(tank.history, index=prepared.index)
        out = pd.DataFrame(index=prepared.index)
        q_w, q_h = hist["Q_W_sto_out_kWh"], hist["Q_H_sto_out_kWh"]
        delivered = (q_w + q_h).replace(0.0, np.nan)
        share_w = (q_w / delivered).fillna(0.5 if (q_w.sum() + q_h.sum()) == 0 else 0.0)
        for service, share in (("W", share_w), ("H", 1.0 - share_w)):
            opts = self.services[service]
            loss = hist["Q_sto_ls_kWh"] * share
            out[f"theta_{service}_sto_set_C"] = tank.set_point_C
            out[f"theta_{service}_sto_out_C"] = opts["output_temperature_C"]
            out[f"theta_{service}_sto_amb_C"] = tank.ambient_C
            out[f"V_{service}_sto_l"] = float(tank.volumes.sum())
            out[f"H_{service}_sto_ls_W_K"] = float(tank.h_ls.sum()) * float(share.mean())
            out[f"Q_{service}_sto_ls_kWh"] = loss
            out[f"Q_{service}_sto_ls_rbl_kWh"] = loss * opts["thermal_loss_room_fraction"]
            out[f"Q_{service}_sto_ls_nrbl_kWh"] = loss * (1.0 - opts["thermal_loss_room_fraction"])
            out[f"W_{service}_sto_aux_kWh"] = 0.0
            out[f"Q_{service}_sto_aux_rvd_kWh"] = 0.0
            out[f"Q_{service}_sto_aux_rbl_kWh"] = 0.0
            out[f"Q_{service}_sto_rbl_kWh"] = out[f"Q_{service}_sto_ls_rbl_kWh"]
            out[f"Q_{service}_sto_in_kWh"] = hist["Q_backup_in_kWh"] * share
            out[f"t_{service}_sto_pmp_in_h"] = 0.0
            out[f"t_{service}_sto_pmp_out_h"] = 0.0
        out["Q_W_sto_delivered_kWh"] = q_w
        out["Q_H_sto_delivered_kWh"] = q_h
        out["Q_W_sto_nreq_kWh"] = hist["Q_W_sto_nreq_kWh"]
        out["Q_H_sto_nreq_kWh"] = hist["Q_H_sto_nreq_kWh"]
        out["Q_sto_solar_in_kWh"] = hist["Q_solar_in_kWh"]
        out["Q_sto_solar_offered_kWh"] = solar.to_numpy()
        for col in hist.columns:
            if col.startswith("T_layer_"):
                out[col] = hist[col]
        return out

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


# ----------------------------------------------------------------------------
# Method A: storage modelled with multiple volumes (layers)
# prEN 15316-5:2024, 6.4.3 (draft; EN 15316-5:2017 has Method B only)
# ----------------------------------------------------------------------------

_RHO_W = 1000.0  # kg/m3
_CP_W_WH = 4186.0 / 3600.0  # Wh/(kg K)


def _rearrange_layers(temps: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Step 7 (6.4.3.8, eq. 38 / Annex C): mix a layer with the one above while
    it is hotter, until temperatures do not decrease upwards.

    Solved as the fixed point of the mixing rule (volume-weighted pooling of
    adjacent violating layers). Annex C reaches the same stratified profile in a
    fixed number of passes; this routine does not reproduce its intermediate
    matrix, only the final profile.
    """
    blocks: list[list[float]] = []  # [energy (V*T), volume, n_layers]
    for t, v in zip(temps, volumes):
        blocks.append([t * v, v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1] + 1e-12:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] += top[2]
    return np.concatenate([np.full(n, e / v) for e, v, n in blocks])


class LayeredStorage:
    """Hot-water storage with N layers of arbitrary volume, EN 15316-5 Method A.

    Layers are numbered from the bottom (1) to the top (N). Per calculation step
    (hour) it applies the 8 steps of 6.4.3: draw-off for domestic hot water
    (eq. 9-22), circulation (23-24), space heating (25-31), solar input on layer 1
    (32), back-up heater on layer ``heater_layer`` (33-37), re-arranging of the
    temperatures (38) and thermal losses (39-42).

    The instance is callable as ``storage(i, q_req_kWh, commit)`` and so can be
    passed as ``storage_response`` to
    :class:`~pybuildingenergy.source.renewables_15316_4_3_4_6.RenewableEnergySystemCalculator`
    (EN 15316-4-3 Method 3). The solar loop heat ``q_req_kWh`` enters on layer 1;
    the loop inlet temperature follows eq. 44. A trial call (``commit=False``) never
    changes the state; the committing call runs the whole step and stores its
    results in ``history``. Call ``storage(i, 0.0, True)`` for steps without solar.

    Assumptions where the draft gives no number: ``heater_layer`` defaults to
    N-1 (the heater sits just below the top layer), ``space_heating_layer`` to N-1
    (note 1 of 6.4.3.6: 3 of 4), the minimum draw-off temperatures to 40 / 30 degC,
    and the maximum temperature that bounds the accepted solar heat to 95 degC.
    Pump auxiliary energy and heat-exchanger limits on the load side are not
    modelled; the solar-loop heat exchanger is (``solar_exchanger_W_K``).
    """

    def __init__(
        self,
        layer_volumes_l,
        set_point_C: float = 60.0,
        cold_water_C: float = 10.0,
        ambient_C: float = 20.0,
        loss_W_K: float = 0.0,
        connection_loss_factor: float = 1.0,
        additional_loss_W_K=0.0,
        heater_power_kW: float = 0.0,
        heater_layer: int | None = None,
        heater_set_point_C: float | None = None,
        space_heating_layer: int | None = None,
        dhw_min_out_C: float = 40.0,
        heating_min_out_C: float = 30.0,
        max_temperature_C: float = 95.0,
        solar_exchanger_W_K: float | None = None,
        solar_mass_flow_kg_s: float | None = None,
        initial_temperatures_C=None,
        step_hours: float = 1.0,
        dhw_demand_kWh=None,
        circulation_kWh=None,
        space_heating_kWh=None,
        ambient_series_C=None,
        rearrange_after_losses: bool = False,
    ):
        self.volumes = np.asarray(layer_volumes_l, dtype=float)
        if self.volumes.ndim != 1 or self.volumes.size < 1 or np.any(self.volumes <= 0.0):
            raise ValueError("layer_volumes_l must be a list of positive volumes (bottom to top).")
        n = self.volumes.size
        self.n = n
        self.set_point_C = float(set_point_C)
        self.cold_C = float(cold_water_C)
        self.ambient_C = float(ambient_C)
        total_v = float(self.volumes.sum())
        # eq. 39: (f_conn * H_ls,i + H_add,i); the standing loss is split by volume
        self.h_ls = (
            float(connection_loss_factor) * float(loss_W_K) * self.volumes / total_v
            + np.broadcast_to(np.asarray(additional_loss_W_K, dtype=float), (n,))
        )
        self.heater_kW = max(float(heater_power_kW), 0.0)
        self.heater_layer = n - 1 if heater_layer is None else int(heater_layer)
        self.sh_layer = max(n - 1, 1) if space_heating_layer is None else int(space_heating_layer)
        for name, val in (("heater_layer", self.heater_layer), ("space_heating_layer", self.sh_layer)):
            if not 1 <= val <= n:
                raise ValueError(f"{name} must be between 1 and {n}.")
        self.heater_set_C = self.set_point_C if heater_set_point_C is None else float(heater_set_point_C)
        self.dhw_min_C = float(dhw_min_out_C)
        self.heating_min_C = float(heating_min_out_C)
        self.max_C = float(max_temperature_C)
        self.h_exh_sol = None if solar_exchanger_W_K is None else float(solar_exchanger_W_K)
        if self.h_exh_sol is not None and self.h_exh_sol <= 0.0:
            raise ValueError("solar_exchanger_W_K must be > 0 (omit it for a direct connection).")
        self.m_sol = None if solar_mass_flow_kg_s is None else float(solar_mass_flow_kg_s)
        self.step_hours = float(step_hours)
        self.dhw = None if dhw_demand_kWh is None else np.asarray(dhw_demand_kWh, dtype=float)
        self.circ = None if circulation_kWh is None else np.asarray(circulation_kWh, dtype=float)
        self.sh = None if space_heating_kWh is None else np.asarray(space_heating_kWh, dtype=float)
        self.ambient_series = None if ambient_series_C is None else np.asarray(ambient_series_C, dtype=float)
        self.rearrange_after_losses = bool(rearrange_after_losses)
        if initial_temperatures_C is not None:
            temps = np.asarray(initial_temperatures_C, dtype=float)
            if temps.shape != (n,):
                raise ValueError("initial_temperatures_C must have one value per layer.")
        else:  # 6.4.3.2: layers affected by the heater at set point, the rest at cold water
            temps = np.where(np.arange(1, n + 1) >= self.heater_layer, self.set_point_C, self.cold_C)
        self.temps = np.asarray(temps, dtype=float).copy()
        self.history: list[dict[str, float]] = []
        self._work: tuple[int, dict] | None = None

    # heat capacity of a layer, kWh/K (eq. 32: rho * cp * V / 1e6 with V in l)
    def _cap(self, volume_l):
        return _RHO_W * _CP_W_WH * volume_l / 1e6

    def stored_energy_kWh(self, reference_C: float | None = None) -> float:
        ref = self.cold_C if reference_C is None else reference_C
        return float(np.sum(self._cap(self.volumes) * (self.temps - ref)))

    # -- steps 2-5: withdrawals, on a working copy -------------------------
    def _withdraw(self, i: int) -> dict:
        t = self.temps.copy()
        n, v, cap = self.n, self.volumes, self._cap(self.volumes)
        ambient = self.ambient_C if self.ambient_series is None else float(self.ambient_series[i])
        q_req_w = 0.0 if self.dhw is None else max(float(self.dhw[i]), 0.0)
        q_circ = 0.0 if self.circ is None else max(float(self.circ[i]), 0.0)
        q_req_h = 0.0 if self.sh is None else max(float(self.sh[i]), 0.0)
        res = {"ambient": ambient, "q_w_req": q_req_w, "q_w_out": 0.0, "v_w": 0.0, "q_h_req": q_req_h}
        # Step 2 (eq. 9-17): usable energy per layer, drawn from the top down
        if q_req_w > 0.0:
            q_use = np.where(t >= self.dhw_min_C, cap * np.maximum(t - self.cold_C, 0.0), 0.0)  # (9)-(10)
            q_out = min(q_req_w, float(q_use.sum()))  # (11)-(12)
            q_i = np.zeros(n)
            q_max = q_out
            for k in range(n - 1, -1, -1):  # (13)-(14), from layer N downwards
                q_i[k] = min(q_max, q_use[k])
                q_max -= q_i[k]
            vol_i = np.where(t > self.cold_C, q_i / (_RHO_W * _CP_W_WH / 1e6 * np.maximum(t - self.cold_C, 1e-12)), 0.0)  # (15)
            v_w = min(float(vol_i.sum()), float(v.sum()))  # (16)
            # Step 3 (eq. 18-22): cold water enters at the bottom and pushes every
            # layer up; what overflows the top is the water drawn off
            sources = [(v_w, self.cold_C)] + [(float(v[k]), float(t[k])) for k in range(n)]
            new_t = np.zeros(n)
            si, left = 0, sources[0][0]
            for k in range(n):
                need, acc = float(v[k]), 0.0
                while need > 1e-12:
                    if left <= 1e-12:
                        si += 1
                        left = sources[si][0]
                        continue
                    take = min(need, left)
                    acc += take * sources[si][1]
                    need -= take
                    left -= take
                new_t[k] = acc / v[k]  # (22)
            t = new_t
            res.update(q_w_out=q_out, v_w=v_w)
        res["q_w_nreq"] = q_req_w - res["q_w_out"]  # (17)
        # Step 4 (eq. 23-24): circulation losses cool the top layer
        if q_circ > 0.0 and res["q_w_out"] <= 0.0:
            t[-1] = max(t[-1] - q_circ / cap[-1], self.cold_C)
        # Step 5 (eq. 25-31): space heating drawn at layer N_vol,sh
        q_out_h = 0.0
        if q_req_h > 0.0:
            usable = float(np.sum(np.maximum(cap[: self.sh_layer] * (t[: self.sh_layer] - self.heating_min_C), 0.0)))  # (25)-(27)
            q_out_h = min(q_req_h, usable + self.heater_kW * self.step_hours)  # (28)
            k = self.sh_layer - 1
            t[k] = max(t[k] - q_out_h / cap[k], self.cold_C)  # (29)-(30)
        res.update(q_h_out=q_out_h, q_h_nreq=q_req_h - q_out_h, temps=t)
        return res

    def _solar_inlet(self, work: dict, q_kWh: float) -> float:
        """Eq. 44: loop inlet temperature from the temperature of layer 1."""
        t1 = float(work["temps"][0])
        if q_kWh <= 0.0 or self.h_exh_sol is None:
            return t1
        d_exh = q_kWh * 1000.0 / (self.h_exh_sol * self.step_hours)
        t_mean = t1 + d_exh
        if self.m_sol is None or self.m_sol <= 0.0:
            return t_mean
        return max(t_mean - q_kWh / (3.6 * self.m_sol * _CP_W_WH * self.step_hours), t1)

    def __call__(self, i: int, q_req_kWh: float, commit: bool) -> tuple[float, float]:
        if self._work is None or self._work[0] != i:
            self._work = (i, self._withdraw(i))
        work = self._work[1]
        cap = self._cap(self.volumes)
        headroom = float(np.sum(cap * np.maximum(self.max_C - work["temps"], 0.0)))
        q_acc = min(max(float(q_req_kWh), 0.0), headroom)
        theta_in = self._solar_inlet(work, q_acc)
        if commit:
            self._finish_step(i, work, q_acc)
            self._work = None
        return theta_in, q_acc

    # -- steps 6-8: inputs, re-arranging, losses; then commit ---------------
    def _finish_step(self, i: int, work: dict, q_solar: float) -> None:
        t, v, cap = work["temps"].copy(), self.volumes, self._cap(self.volumes)
        if q_solar > 0.0:  # (32), then re-arrange after the input (note 2)
            t[0] += q_solar / cap[0]
            t = _rearrange_layers(t, v)
        q_bu = 0.0
        k = self.heater_layer - 1
        if self.heater_kW > 0.0 and t[k] <= self.heater_set_C:  # note 4: ON only below set point
            need = float(np.sum(cap[k:] * np.maximum(self.heater_set_C - t[k:], 0.0)))  # (33)-(34)
            q_bu = min(need, self.heater_kW * self.step_hours)  # (35)
            t[k] += q_bu / cap[k]  # (36)-(37)
            t = _rearrange_layers(t, v)  # (38)
        loss = self.h_ls * (t - work["ambient"]) * self.step_hours / 1000.0  # (39)
        t = t - loss / cap  # (41)-(42)
        if self.rearrange_after_losses:
            t = _rearrange_layers(t, v)
        self.history.append({
            "step": i,
            "Q_W_sto_out_kWh": work["q_w_out"],
            "Q_W_sto_nreq_kWh": work["q_w_nreq"],
            "V_W_withdrawn_l": work["v_w"],
            "Q_H_sto_out_kWh": work["q_h_out"],
            "Q_H_sto_nreq_kWh": work["q_h_nreq"],
            "Q_solar_in_kWh": q_solar,
            "Q_backup_in_kWh": q_bu,
            "Q_sto_ls_kWh": float(loss.sum()),
            **{f"T_layer_{j + 1}_C": float(t[j]) for j in range(self.n)},
        })
        self.temps = t
