"""Cooling-system operating conditions according to EN 16798-9:2017.

The module covers the water-based cooling-system hand-off calculations from
EN 16798-9, Modules M4-1 and M4-4. It is intentionally small: emission,
distribution, storage and generation are calculated by their own modules, while
this calculator supplies the required chilled-water temperatures, volume flow
and generation-side cooling request used to connect those modules.

Implemented clauses:

* 6.4.2 / 7.4.2: required generation outlet and distribution flow
  temperatures for constant or outdoor-compensated control;
* 7.4.2.3 and 7.4.2.4: distribution volume flow, flow temperature (23) and
  return temperature (24);
* 7.4.3.1 and 7.4.3.2: cooling request passed towards storage/generation.

Scope and assumptions (one distribution circuit per calculator):

* MAX_TMP (17)/(20): the minimum of the required temperatures of the circuits or
  AHU cooling coils, given as input columns ``theta_C_dis_in_flw_req_C_<i>`` or
  ``theta_C_ahu_in_req_C[_<k>]``, limited to [set_min, set_max]; without them a
  ValueError is raised.
* (23): the flow temperature is the highest of the required one and the actual
  generator / storage outlet temperatures when given (``theta_C_gen_out_C``,
  ``theta_C_sto_out_C``); otherwise the required temperature is assumed reached.
* (21)-(22): the variable flow is limited to the design flow; with constant flow
  the pump is assumed off (flow 0) when there is no demand.
* (26): this calculator passes the distribution request plus the simplified losses
  and auxiliary heat; the storage contribution Q_C;sto;in is subtracted by the
  EN 16798-15 module (its ``Q_C_sto_in_kWh`` is the generator request, its
  ``Q_C_sto_net_kWh`` the net storage contribution, > 0 when discharging).

Several distribution circuits (``circuits`` option, 7.4.2.2-7.4.2.4, 7.4.3.2):
each circuit i has its own demand columns (28), required flow temperature
(18)-(20), flow (21)/(22), flow temperature (23) and return temperature (24); the
total request is their sum (27), MAX_TMP takes the minimum of the circuit
requirements (17), the water returned to the generator is the flow-weighted mix
of the circuit returns. Without ``circuits`` the calculator is one circuit.

Terminal capacity (``terminal_limits``, library extension): with
``terminal_design_capacity_kW`` (per circuit or at top level) the cooling the
terminals can deliver at the actual flow temperature is
Phi_des ((theta_room - theta_m) / (theta_room;des - theta_m;des))^n, theta_m = flow +
deltaT/2. ``CoolingChainCalculator`` feeds back the actual generator / storage
temperatures (23)/(24) and uses it to separate a temperature shortfall from an
energy shortfall.

Energy actually delivered (``distribute_shortfall``): the energy the generation
(or storage) side could not supply is taken from the circuits by priority
(priority 1 served first; the circuits with the highest number lose first; equal
priority shares pro rata), then from the zones of each circuit pro rata. The
distribution losses and auxiliary heat are assumed supplied (they occur whenever
the network runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warnings

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12
_WATER_HEAT_CAPACITY_DENSITY_KWH_M3K = 1.16


@dataclass
class CoolingSystemSimulationResult:
    """Container returned by :class:`CoolingSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class CoolingSystemCalculator:
    """EN 16798-9:2017 water-based cooling-system connector."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> CoolingSystemSimulationResult:
        """Run the cooling-system operating-condition calculation."""

        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        summary = self._summarize(results)
        return CoolingSystemSimulationResult(
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

        self.system_type = str(cfg.get("system_type", "water")).lower()
        if self.system_type not in {"water", "dx", "air"}:
            raise ValueError("system_type must be 'water', 'dx' or 'air'.")

        self.generator_temperature_control = str(
            cfg.get("generator_temperature_control", "VARIABLE")
        ).upper()
        self.distribution_temperature_control = str(
            cfg.get("distribution_temperature_control", "CONST")
        ).upper()
        self.distribution_flow_control = str(
            cfg.get("distribution_flow_control", "VARIABLE")
        ).upper()

        allowed_temp = {"CONST", "ODA_COMP", "MAX_TMP"}
        if self.distribution_temperature_control not in allowed_temp:
            raise ValueError(
                "distribution_temperature_control must be CONST, ODA_COMP or MAX_TMP."
            )
        if self.distribution_flow_control not in {"CONST", "VARIABLE"}:
            raise ValueError("distribution_flow_control must be CONST or VARIABLE.")

        self.theta_C_gen_out_set_C = float(cfg.get("theta_C_gen_out_set_C", 7.0))
        self.theta_C_dis_flw_set_C = float(
            cfg.get("theta_C_dis_flw_set_C", cfg.get("supply_temperature_C", 7.0))
        )
        self.theta_C_dis_flw_set_min_C = float(
            cfg.get("theta_C_dis_flw_set_min_C", 6.0)
        )
        self.theta_C_dis_flw_set_max_C = float(
            cfg.get("theta_C_dis_flw_set_max_C", 18.0)
        )
        self.outdoor_compensation_slope = float(
            cfg.get("outdoor_compensation_slope", cfg.get("f_e", 0.0))
        )
        self.outdoor_compensation_offset_K = float(
            cfg.get("outdoor_compensation_offset_K", cfg.get("delta_theta_off_K", 7.0))
        )
        self.design_deltaT_K = _positive_float(
            cfg.get("design_deltaT_K", 5.0), "design_deltaT_K"
        )
        self.design_flow_m3_h = max(float(cfg.get("design_flow_m3_h", 0.0)), 0.0)
        self.design_cooling_load_kW = max(
            float(cfg.get("design_cooling_load_kW", cfg.get("nominal_power_kW", 0.0))),
            0.0,
        )
        self.f_wat_C_aux_dis = _fraction(
            cfg.get("f_wat_C_aux_dis", cfg.get("cooling_aux_to_water_fraction", 0.0)),
            "f_wat_C_aux_dis",
        )
        self.simplified_distribution_loss_factor = _fraction(
            cfg.get("f_C_ls_dis", cfg.get("simplified_distribution_loss_factor", 0.0)),
            "f_C_ls_dis",
        )
        self.simplified_auxiliary_factor = _fraction(
            cfg.get("f_C_aux_dis", cfg.get("simplified_auxiliary_factor", 0.0)),
            "f_C_aux_dis",
        )
        self.terminal = self._load_terminal(cfg, self.theta_C_dis_flw_set_C, self.design_deltaT_K, {})
        self.circuits = self._load_circuits(cfg.get("circuits"))

    def _load_terminal(self, c: dict, flow_set: float, delta_t: float, inherit: dict) -> dict | None:
        """Terminal capacity at the actual water temperature (library extension, see ``terminal_limits``)."""
        cap = c.get("terminal_design_capacity_kW")
        if cap is None:
            return None
        get = lambda k, d: c.get(k, inherit.get(k, d))
        t = {
            "capacity": _positive_float(cap, "terminal_design_capacity_kW"),
            "flow_des": float(get("terminal_design_flow_temperature_C", flow_set)),
            "room_des": float(get("terminal_design_room_temperature_C", 26.0)),
            "room": float(get("terminal_room_temperature_C", 26.0)),
            "n": _positive_float(get("terminal_exponent", 1.0), "terminal_exponent"),
            "deltaT": delta_t,
        }
        if t["room_des"] - t["flow_des"] - delta_t / 2.0 <= 0.0:
            raise ValueError("terminal: the design room temperature must exceed the design mean water "
                             "temperature (flow + deltaT/2).")
        return t

    def _load_circuits(self, raw) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("circuits must be a non-empty list of dictionaries.")
        if self.system_type != "water":
            raise ValueError("circuits apply to system_type 'water' only.")
        circuits, names = [], set()
        for c in raw:
            name = str(c.get("name", "")).strip()
            if not name or name in names:
                raise ValueError("each circuit needs a unique non-empty 'name'.")
            names.add(name)
            cols = c.get("demand_columns", c.get("demand_column"))
            cols = [cols] if isinstance(cols, str) else list(cols or [])
            if not cols:
                raise ValueError(f"circuit '{name}': give demand_columns (EN 16798-9 (28)).")
            priority = int(c.get("priority", 1))
            if priority < 1:
                raise ValueError(f"circuit '{name}': priority must be an integer >= 1.")
            ctrl = str(c.get("distribution_temperature_control", "CONST")).upper()
            if ctrl not in {"CONST", "ODA_COMP", "MAX_TMP"}:
                raise ValueError(f"circuit '{name}': temperature control CONST, ODA_COMP or MAX_TMP.")
            ahu = c.get("ahu_required_temperature_columns", [])
            ahu = [ahu] if isinstance(ahu, str) else list(ahu)
            if ctrl == "MAX_TMP" and not ahu:
                raise ValueError(f"circuit '{name}': MAX_TMP needs ahu_required_temperature_columns (20).")
            flow_ctrl = str(c.get("distribution_flow_control", self.distribution_flow_control)).upper()
            if flow_ctrl not in {"CONST", "VARIABLE"}:
                raise ValueError(f"circuit '{name}': distribution_flow_control CONST or VARIABLE.")
            circuits.append({
                "name": name, "priority": priority, "demand_columns": cols,
                "control": ctrl, "ahu_columns": ahu, "flow_control": flow_ctrl,
                "set": float(c.get("theta_C_dis_flw_set_C", self.theta_C_dis_flw_set_C)),
                "set_min": float(c.get("theta_C_dis_flw_set_min_C", self.theta_C_dis_flw_set_min_C)),
                "set_max": float(c.get("theta_C_dis_flw_set_max_C", self.theta_C_dis_flw_set_max_C)),
                "slope": float(c.get("outdoor_compensation_slope", c.get("f_e", self.outdoor_compensation_slope))),
                "offset": float(c.get("outdoor_compensation_offset_K",
                                      c.get("delta_theta_off_K", self.outdoor_compensation_offset_K))),
                "design_flow": max(float(c.get("design_flow_m3_h", 0.0)), 0.0),
                "design_load": max(float(c.get("design_cooling_load_kW", 0.0)), 0.0),
                "deltaT": _positive_float(c.get("design_deltaT_K", self.design_deltaT_K), "design_deltaT_K"),
            })
            circuits[-1]["terminal"] = self._load_terminal(
                c, circuits[-1]["set"], circuits[-1]["deltaT"], self.input_data)
        return circuits

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
        out.loc[:, "T_op_C"] = _series_from_aliases(
            df, ["T_op", "operative_temperature_C", "T_zone_C"], default=np.nan
        ).astype(float)
        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        if self.circuits:
            df = self._prepare_circuits(df, out)
            out.loc[:, "Q_C_dis_out_tot_req_kWh"] = sum(            # (27)
                out[f"Q_C_dis_in_req_kWh_{c['name']}"] for c in self.circuits
            )
        else:
            out.loc[:, "Q_C_dis_out_tot_req_kWh"] = self._demand_from_columns(
                df,
                kwh_aliases=[
                    "Q_C_dis_out_tot_req_kWh",
                    "Q_C_dis_out_kWh",
                    "Q_C_em_in_kWh",
                    "Q_C_kWh",
                    "QC_kWh",
                    "cooling_kWh",
                    "space_cooling_kWh",
                ],
                raw_aliases=["Q_C", "Cooling_needs"],
                fallback=np.maximum(-q_hc, 0.0),
            )
        if self.distribution_temperature_control == "MAX_TMP":
            out.loc[:, "theta_C_dis_in_flw_req_circuits_min_C"] = _circuit_required_minimum(df)
        for col in ("theta_C_gen_out_C", "theta_C_sto_out_C"):
            if col in df.columns:
                out.loc[:, col + "_avl"] = df[col].astype(float)
        if self.system_type in {"dx", "air"}:
            out.loc[:, "theta_C_int_inc_C"] = _series_from_aliases(
                df, ["theta_C_int_inc_C", "cooling_setpoint_C"], default=26.0
            ).astype(float)
            out.loc[:, "theta_SUP_C_req_C"] = _series_from_aliases(
                df, ["theta_SUP_C_req_C", "supply_air_temperature_C"], default=14.0
            ).astype(float)
        return out

    def _prepare_circuits(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """Circuit demands (28) and required flow temperatures (18)-(20)."""
        df = df.copy()
        t_e = out["T_ext_C"].astype(float)
        for c in self.circuits:
            n = c["name"]
            missing = [col for col in c["demand_columns"] if col not in df.columns]
            if missing:
                raise ValueError(f"circuit '{n}': demand columns not found: {missing}.")
            q = sum(df[col].astype(float) for col in c["demand_columns"])
            if self.demand_unit == "wh":
                q = q / 1000.0
            out.loc[:, f"Q_C_dis_in_req_kWh_{n}"] = q.clip(lower=0.0)
            for col in c["demand_columns"]:
                zq = df[col].astype(float) / (1000.0 if self.demand_unit == "wh" else 1.0)
                out.loc[:, f"Q_C_zone_req_kWh_{col}"] = zq.clip(lower=0.0)
            if c["control"] == "CONST":                                          # (18)
                req = pd.Series(c["set"], index=df.index)
            elif c["control"] == "ODA_COMP":                                     # (19)
                if t_e.isna().any():
                    raise ValueError(f"circuit '{n}': ODA_COMP needs the outdoor temperature.")
                req = (c["slope"] * t_e + c["offset"]).clip(lower=c["set_min"], upper=c["set_max"])
            else:                                                                # (20)
                missing = [col for col in c["ahu_columns"] if col not in df.columns]
                if missing:
                    raise ValueError(f"circuit '{n}': AHU temperature columns not found: {missing}.")
                req = df[c["ahu_columns"]].astype(float).min(axis=1).clip(
                    lower=c["set_min"], upper=c["set_max"])
            out.loc[:, f"theta_C_dis_in_flw_req_C_{n}"] = req
            df[f"theta_C_dis_in_flw_req_C_{n}"] = req      # visible to MAX_TMP (17)
        return df

    def _demand_from_columns(
        self,
        df: pd.DataFrame,
        kwh_aliases: list[str],
        raw_aliases: list[str],
        fallback: pd.Series | np.ndarray | float,
    ) -> pd.Series:
        series = _demand_series(df, kwh_aliases, "EN 16798-9")
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

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy(deep=True)

        if self.system_type == "dx":
            out.loc[:, "theta_C_gen_out_req_C"] = out["theta_C_int_inc_C"]
            out.loc[:, "theta_C_dis_in_flw_req_C"] = np.nan
            out.loc[:, "theta_C_dis_out_ret_req_C"] = np.nan
            out.loc[:, "q_V_C_dis_m3_h"] = 0.0
        elif self.system_type == "air":
            out.loc[:, "theta_C_gen_out_req_C"] = out["theta_SUP_C_req_C"]
            out.loc[:, "theta_C_dis_in_flw_req_C"] = np.nan
            out.loc[:, "theta_C_dis_out_ret_req_C"] = np.nan
            out.loc[:, "q_V_C_dis_m3_h"] = 0.0
        else:
            flow_temp = self._distribution_flow_temperature(out)
            out.loc[:, "theta_C_dis_in_flw_req_C"] = flow_temp
            if self.generator_temperature_control == "CONST":
                gen_out = pd.Series(self.theta_C_gen_out_set_C, index=out.index)
            else:
                gen_out = flow_temp
            out.loc[:, "theta_C_gen_out_req_C"] = gen_out
            out.loc[:, "theta_C_dis_in_flw_C"] = self._actual_flow_temperature(out)
            if self.circuits:
                self._simulate_circuits(out)
            else:
                out.loc[:, "q_V_C_dis_m3_h"] = self._volume_flow(out)
                out.loc[:, "theta_C_dis_out_ret_req_C"] = self._return_temperature(out)

        useful = out["Q_C_dis_out_tot_req_kWh"]
        out.loc[:, "Q_C_dis_ls_simplified_kWh"] = (
            useful * self.simplified_distribution_loss_factor
        )
        out.loc[:, "W_C_aux_dis_simplified_kWh"] = useful * self.simplified_auxiliary_factor
        out.loc[:, "Q_C_gen_in_req_kWh"] = (
            useful
            + out["Q_C_dis_ls_simplified_kWh"]
            + self.f_wat_C_aux_dis * out["W_C_aux_dis_simplified_kWh"]
        )

        if "theta_C_dis_in_flw_C" not in out.columns:
            out.loc[:, "theta_C_dis_in_flw_C"] = out["theta_C_dis_in_flw_req_C"]
        out.loc[:, "theta_C_dis_supply_C"] = out["theta_C_dis_in_flw_C"]
        out.loc[:, "theta_C_dis_return_C"] = out["theta_C_dis_out_ret_req_C"]
        out.loc[:, "T_C_supply_C"] = out["theta_C_dis_in_flw_C"]
        out.loc[:, "T_C_return_C"] = out["theta_C_dis_out_ret_req_C"]
        out.loc[:, "T_C_sink_C"] = out["theta_C_gen_out_req_C"]
        return out

    def _distribution_flow_temperature(self, df: pd.DataFrame) -> pd.Series:
        if self.distribution_temperature_control == "CONST":
            return pd.Series(self.theta_C_dis_flw_set_C, index=df.index)
        if self.distribution_temperature_control == "ODA_COMP":
            raw = (
                self.outdoor_compensation_slope * df["T_ext_C"].astype(float)
                + self.outdoor_compensation_offset_K
            )
            return raw.clip(
                lower=self.theta_C_dis_flw_set_min_C,
                upper=self.theta_C_dis_flw_set_max_C,
            )
        # MAX_TMP (17)/(20): minimum of the circuit / AHU coil requirements
        return df["theta_C_dis_in_flw_req_circuits_min_C"].clip(
            lower=self.theta_C_dis_flw_set_min_C,
            upper=self.theta_C_dis_flw_set_max_C,
        )

    def _actual_flow_temperature(self, df: pd.DataFrame) -> pd.Series:
        """Formula (23): max(theta_C;gen;out, theta_C;sto;out, required)."""
        temps = [df["theta_C_dis_in_flw_req_C"].astype(float)]
        for col in ("theta_C_gen_out_C_avl", "theta_C_sto_out_C_avl"):
            if col in df.columns:
                temps.append(df[col].astype(float).fillna(-np.inf))
        return pd.concat(temps, axis=1).max(axis=1)

    def _simulate_circuits(self, out: pd.DataFrame) -> None:
        """Per-circuit flow (21)/(22), flow temperature (23), return (24); mixed return."""
        hours = out["hours"]
        total_flow = pd.Series(0.0, index=out.index)
        mix = pd.Series(0.0, index=out.index)
        for c in self.circuits:
            n = c["name"]
            q = out[f"Q_C_dis_in_req_kWh_{n}"]
            flow = _volume_flow(q, hours, c["flow_control"], c["design_flow"], c["design_load"], c["deltaT"])
            # (23): the circuit receives the network water, raised to its own requirement
            t_flw = pd.concat([out["theta_C_dis_in_flw_C"], out[f"theta_C_dis_in_flw_req_C_{n}"]],
                              axis=1).max(axis=1)
            t_ret = _return_temperature(q, hours, flow, t_flw)
            out.loc[:, f"q_V_C_dis_m3_h_{n}"] = flow
            out.loc[:, f"theta_C_dis_in_flw_C_{n}"] = t_flw
            out.loc[:, f"theta_C_dis_out_ret_C_{n}"] = t_ret
            total_flow += flow
            mix += flow * t_ret
        out.loc[:, "q_V_C_dis_m3_h"] = total_flow
        out.loc[:, "theta_C_dis_out_ret_req_C"] = (mix / total_flow.replace(0.0, np.nan)).fillna(
            out["theta_C_dis_in_flw_C"])

    def distribute_shortfall(
        self,
        result: CoolingSystemSimulationResult,
        unmet_kWh: pd.Series | np.ndarray | float,
    ) -> pd.DataFrame:
        """Energy actually delivered to the circuits and zones (7.4.3).

        ``unmet_kWh``: cooling energy the generation side could not supply in each
        step (e.g. ``Q_C_gen_unmet_kWh`` of EN 16798-13). With a storage between, pass
        the shortfall towards the users ``Q_C_sto_unmet_kWh`` of a storage run with the
        generator availability; :class:`CoolingChainCalculator` does both automatically.
        """
        ts = result.timeseries
        unmet = pd.Series(unmet_kWh, index=ts.index, dtype=float) if np.isscalar(unmet_kWh) \
            else pd.Series(np.asarray(unmet_kWh, dtype=float), index=ts.index)
        if not np.all(np.isfinite(unmet)) or (unmet < 0).any():
            raise ValueError("unmet_kWh must be finite and >= 0.")
        circuits = self.circuits or [{"name": "main", "priority": 1, "demand_columns": []}]
        req = {c["name"]: (ts[f"Q_C_dis_in_req_kWh_{c['name']}"] if self.circuits
                           else ts["Q_C_dis_out_tot_req_kWh"]).to_numpy(dtype=float)
               for c in circuits}
        short = np.minimum(unmet.to_numpy(), sum(req.values()))
        cut = {n: np.zeros(len(ts)) for n in req}
        remaining = short.copy()
        for prio in sorted({c["priority"] for c in circuits}, reverse=True):   # lowest priority first
            group = [c["name"] for c in circuits if c["priority"] == prio]
            demand = sum(req[n] for n in group)
            take = np.minimum(remaining, demand)
            for n in group:
                share = np.divide(req[n], demand, out=np.zeros_like(demand), where=demand > _KWH_EPS)
                cut[n] += take * share
            remaining -= take
        res = pd.DataFrame(index=ts.index)
        for c in circuits:
            n = c["name"]
            delivered = req[n] - cut[n]
            res[f"Q_C_dis_in_req_kWh_{n}"] = req[n]
            res[f"Q_C_dis_in_kWh_{n}"] = delivered
            res[f"Q_C_dis_unmet_kWh_{n}"] = cut[n]
            res[f"f_C_sup_{n}"] = np.divide(delivered, req[n], out=np.ones_like(delivered),
                                            where=req[n] > _KWH_EPS)
            for col in c["demand_columns"]:                  # zones of the circuit, pro rata
                res[f"Q_C_del_kWh_{col}"] = res[f"f_C_sup_{n}"].to_numpy() * \
                    self._last_zone_demand(result, col)
        res["Q_C_dis_in_tot_kWh"] = sum(res[f"Q_C_dis_in_kWh_{c['name']}"] for c in circuits)
        res["Q_C_dis_unmet_tot_kWh"] = short
        return res

    def terminal_limits(self, result: CoolingSystemSimulationResult) -> pd.DataFrame:
        """Cooling the terminals of each circuit can deliver at the actual flow temperature.

        Library extension (EN 16798-9 gives the temperatures, not the terminal output):
        Phi_max = Phi_des ((theta_room - theta_m) / (theta_room;des - theta_m;des))^n with the
        mean water temperature theta_m = theta_flw + deltaT/2, theta_flw the actual circuit flow
        temperature (23) and theta_room the column ``T_op`` (else ``terminal_room_temperature_C``).
        Options per circuit (or at top level without circuits): ``terminal_design_capacity_kW``,
        ``terminal_design_flow_temperature_C`` (default: the set flow temperature),
        ``terminal_design_room_temperature_C`` (26), ``terminal_exponent`` n (1.0).
        Returns ``Q_C_term_max_kWh_<circuit>`` and ``f_C_term_<circuit>`` = min(1, max / demand)
        for the circuits with terminal data.
        """
        ts = result.timeseries
        res = pd.DataFrame(index=ts.index)
        circuits = self.circuits or [{"name": "main", "terminal": self.terminal}]
        for c in circuits:
            t = c["terminal"]
            if t is None:
                continue
            n = c["name"]
            t_flw = ts[f"theta_C_dis_in_flw_C_{n}"] if self.circuits else ts["theta_C_dis_in_flw_C"]
            demand = ts[f"Q_C_dis_in_req_kWh_{n}"] if self.circuits else ts["Q_C_dis_out_tot_req_kWh"]
            room = ts["T_op_C"].fillna(t["room"]) if "T_op_C" in ts else pd.Series(t["room"], index=ts.index)
            d_act = (room - t_flw.astype(float) - t["deltaT"] / 2.0).clip(lower=0.0)
            d_des = t["room_des"] - t["flow_des"] - t["deltaT"] / 2.0
            q_max = t["capacity"] * (d_act / d_des) ** t["n"] * ts["hours"]
            res[f"Q_C_term_max_kWh_{n}"] = q_max
            res[f"f_C_term_{n}"] = np.divide(q_max.to_numpy(), demand.to_numpy(dtype=float),
                                             out=np.ones(len(ts)), where=demand.to_numpy() > _KWH_EPS).clip(0.0, 1.0)
        return res

    def demand_columns_by_circuit(self) -> dict[str, list[str]]:
        """Input columns that carry the demand of each circuit ('main' without circuits)."""
        if self.circuits:
            return {c["name"]: list(c["demand_columns"]) for c in self.circuits}
        return {"main": ["Q_C_dis_out_tot_req_kWh", "Q_C_dis_out_kWh", "Q_C_em_in_kWh", "Q_C_kWh",
                         "QC_kWh", "cooling_kWh", "space_cooling_kWh", "Q_C", "Cooling_needs", "Q_HC"]}

    @staticmethod
    def _last_zone_demand(result: CoolingSystemSimulationResult, col: str) -> np.ndarray:
        return result.timeseries[f"Q_C_zone_req_kWh_{col}"].to_numpy(dtype=float)

    def _volume_flow(self, df: pd.DataFrame) -> pd.Series:
        return _volume_flow(df["Q_C_dis_out_tot_req_kWh"], df["hours"], self.distribution_flow_control,
                            self.design_flow_m3_h, self.design_cooling_load_kW, self.design_deltaT_K)

    def _return_temperature(self, df: pd.DataFrame) -> pd.Series:
        return _return_temperature(df["Q_C_dis_out_tot_req_kWh"], df["hours"], df["q_V_C_dis_m3_h"],
                                   df["theta_C_dis_in_flw_C"])

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        q = results["Q_C_dis_out_tot_req_kWh"]
        weights = q + _KWH_EPS
        summary = {
            "hours": float(results["hours"].sum()),
            "QC_dis_out_tot_req_kWh": float(q.sum()),
            "QC_gen_in_req_simplified_kWh": float(results["Q_C_gen_in_req_kWh"].sum()),
            "QC_dis_ls_simplified_kWh": float(
                results["Q_C_dis_ls_simplified_kWh"].sum()
            ),
            "WC_aux_dis_simplified_kWh": float(
                results["W_C_aux_dis_simplified_kWh"].sum()
            ),
            "theta_C_gen_out_req_mean_C": _weighted_mean(
                results["theta_C_gen_out_req_C"], weights
            ),
            "theta_C_dis_supply_mean_C": _weighted_mean(
                results["theta_C_dis_in_flw_req_C"], weights
            ),
            "theta_C_dis_return_mean_C": _weighted_mean(
                results["theta_C_dis_out_ret_req_C"], weights
            ),
            "q_V_C_dis_mean_m3_h": _weighted_mean(results["q_V_C_dis_m3_h"], weights),
            "q_V_C_dis_max_m3_h": float(results["q_V_C_dis_m3_h"].max()),
        }
        return summary

    def _time_step_hours(self, df: pd.DataFrame) -> pd.Series:
        step = _series_from_aliases(df, ["time_step_hours", "dt_h"], default=np.nan)
        if not step.isna().all():
            return step.astype(float).clip(lower=0.0)

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            diffs = df.index.to_series().diff().dt.total_seconds().dropna() / 3600.0
            if not diffs.empty and np.isfinite(diffs.median()) and diffs.median() > 0:
                return pd.Series(float(diffs.median()), index=df.index)

        return pd.Series(self.default_time_step_h, index=df.index)


def _volume_flow(q: pd.Series, hours: pd.Series, control: str, design_flow: float,
                 design_load: float, deltaT: float) -> pd.Series:
    """(21) constant / (22) variable flow of one circuit."""
    q = q.astype(float)
    h = hours.replace(0.0, np.nan).astype(float)
    if control == "CONST" and design_flow > _KWH_EPS:
        return pd.Series(design_flow, index=q.index).where(q > _KWH_EPS, 0.0)
    if design_flow > _KWH_EPS and design_load > _KWH_EPS:
        flow = design_flow * (q / h) / design_load
        return flow.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=design_flow)
    flow = q / (_WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * deltaT * h)
    return flow.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)


def _return_temperature(q: pd.Series, hours: pd.Series, flow: pd.Series, t_flw: pd.Series) -> pd.Series:
    """(24) return temperature of one circuit."""
    h = hours.replace(0.0, np.nan).astype(float)
    delta = q.astype(float) / (_WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * flow.replace(0.0, np.nan) * h)
    return t_flw.astype(float) + delta.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _circuit_required_minimum(df: pd.DataFrame) -> pd.Series:
    cols = [
        c for c in df.columns
        if str(c).startswith("theta_C_dis_in_flw_req_C_") or str(c).startswith("theta_C_ahu_in_req_C")
    ]
    if not cols:
        raise ValueError(
            "distribution_temperature_control = MAX_TMP needs the required temperatures of "
            "the circuits or AHU cooling coils as input columns "
            "'theta_C_dis_in_flw_req_C_<i>' or 'theta_C_ahu_in_req_C[_<k>]' (EN 16798-9 (17), (20))."
        )
    return df[cols].astype(float).min(axis=1)


def _demand_series(df: pd.DataFrame, aliases: list[str], module: str) -> pd.Series | None:
    """First demand column found; warn if other demand columns hold different values."""
    present = [a for a in aliases if a in df.columns]
    if not present:
        return None
    chosen = df[present[0]].astype(float)
    differing = [
        a for a in present[1:]
        if not np.allclose(df[a].astype(float).to_numpy(), chosen.to_numpy(), equal_nan=True)
    ]
    if differing:
        warnings.warn(
            f"{module}: cooling demand read from '{present[0]}'; the columns {differing} hold "
            "different values and are ignored. Drop stale demand columns before chaining "
            "the cooling modules.",
            RuntimeWarning,
            stacklevel=4,
        )
    return chosen


def _series_from_aliases(
    df: pd.DataFrame,
    aliases: list[str],
    default: float | None,
) -> pd.Series | None:
    for alias in aliases:
        if alias in df.columns:
            return df[alias]
    if default is None:
        return None
    return pd.Series(default, index=df.index, dtype=float)


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return result


def _fraction(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return result


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float)
    w = weights.astype(float)
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))
