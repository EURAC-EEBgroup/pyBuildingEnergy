"""Cooling storage calculation according to EN 16798-15:2017.

The module implements the hourly chilled-water storage balance for Module M4-7.
It is designed to sit between cooling distribution and cooling generation:
users provide the cooling energy required by the distribution side and the
calculator returns the cooling energy that must be removed by the generator,
including storage heat gains and storage pump auxiliary energy.

Implemented clauses:

* 6.5.2: required storage inlet temperature;
* 6.5.3.4: heat gains from the generator loop, distribution loop and storage
  vessel;
* 6.5.3.6: storage pump auxiliary energy from transferred cooling energy;
* 6.5.3.7: recoverable thermal losses, with the cooling sign convention used by
  EN 16798-15.

State of charge (chilled water, sensible storage only):

* the state is the mean storage temperature theta_C;sto; the stored energy is (2)
  referred to the design return temperature ``theta_C_sto_return_C``:
  Q_stored = rho c V (theta_ret - theta_C;sto), between 0 and
  Q_max = rho c V (theta_ret - theta_full), with theta_full = max(set point
  ``theta_C_sto_C``, generator outlet temperature). Initial state: full (6.5.3.2);
* discharge to the distribution, (5): min(Q_stored, f_C;sto;ctrl x request); the
  rest of the request is supplied directly by the generator;
* control factor f_C;sto;ctrl, (6a)-(6c): CONT = 1; TIME from ``discharge_schedule``
  (24 hourly values 0/1); TEMP = 0 when theta_e < ``theta_e_lim_C``; LOAD_PRED (M10-12)
  and any other strategy through an input column ``f_C_sto_ctrl``;
* charging: with CONT the generator refills the tank at every step (a buffer kept
  full: same generator request as without state); with TIME/TEMP/LOAD_PRED the tank
  is charged only while discharge is not authorised (f_C;sto;ctrl = 0), so the load
  is shifted; ``charge_power_kW`` limits the charging rate (default unlimited);
* storage losses (9) are taken from the stored energy with the temperature at the
  start of the step; loop losses (7), (8) and the pump heat go to the generator;
* the request to the generator is direct supply + charging + (7) + (8) + pump heat,
  i.e. (26) of EN 16798-9 with the net storage contribution
  ``Q_C_sto_net_kWh`` = discharge - charging (EN 16798-9 sign: > 0 discharging);
* formula (1): while the tank is recharged beyond its losses the generator flow
  temperature is theta_C;sto - delta_theta_C;sto;gen;flw, otherwise the
  distribution requirement.

Generator availability (coupling, library extension): an input column
``Q_C_sto_gen_avl_kWh`` is the cooling the generator can supply in the step
(default unlimited). The loop losses (7), (8) and the pump heat are covered first,
then the users (direct supply), then the charging; with
``discharge_on_generator_shortfall`` (default True) the store also discharges, even
outside the authorised hours, what the generator cannot supply. Outputs:
``Q_C_sto_unmet_kWh`` (shortfall towards the users), ``Q_C_sto_chg_unmet_kWh``
(charge wanted but not received), ``Q_C_sto_out_del_kWh``. The actual generator
outlet temperature ``theta_C_gen_out_C`` limits the charging (chilled water: down to
max(theta_C_sto_C, theta_C_gen_out_C); ice / PCM: no latent charge while it is above
theta_sto;tr); for ice / PCM, without that column or config value, the generator is
assumed to reach theta_C_sto_C. ``CoolingChainCalculator`` sets both columns.

Stratification (STO_TYPE_CW): ``stratification_nodes`` = N > 1 divides the tank
into N equal nodes with plug flow. Discharge draws the coldest water at the
bottom and returns the warm water at the top; charging enters at the bottom and
leaves at the top; losses are shared equally. The water sent to the
distribution is at the bottom-node temperature (``theta_C_sto_out_act_C``) and,
with ``theta_C_sto_out_max_C``, the discharge stops when the bottom node is
warmer than that limit. N = 1 is the fully mixed tank.

Ice and PCM (STO_TYPE_ICE, STO_TYPE_PCM, 6.5.3.5): enthalpy model of the medium
(mass rho_lqd x V) with three bands of stored cold, (2)-(4): sensible liquid from
the return temperature down to theta_sto;tr, latent Q_lat per kg of solid, sensible
solid down to the charging temperature ``theta_C_sto_C`` (ice: linear temperature
profile in the ice, mean (theta_tr + theta_coil)/2 as in (12); PCM: uniform solid
temperature as in (18)). The solid mass is limited by the whole medium or, for
ice on coil, by the maximum ice thickness d_C;sto;max on a tube of outer diameter
``tube_outer_diameter_m`` and length ``tube_length_m``, (11)/(13). The phase change
rate is limited by the crystallisation / melting conductances K_v;cr, K_v;fu
(Table 5): charging <= K_v;cr V (theta_tr - theta_charge), melting <= K_v;fu V
(theta_ret - theta_tr). Defaults from Table B.2 (its Q_lat and Cp are in Wh/kg and
Wh/(kg K): 93 Wh/kg = 334 kJ/kg for ice). The PCM transition temperature has to be
given (``theta_sto_tr_C``).

Not modelled: hysteresis and supercooling of PCM, heat transfer fluid inside the
latent store, heat release to an ambient colder than the medium (gains are
clipped at zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import copy
import warnings

import numpy as np
import pandas as pd

from .cooling_16798_9 import _demand_series


_KWH_EPS = 1e-12
_WATER_HEAT_CAPACITY_DENSITY_KWH_M3K = 1.16


@dataclass
class CoolingStorageSimulationResult:
    """Container returned by :class:`CoolingStorageSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class CoolingStorageSystemCalculator:
    """EN 16798-15:2017 chilled-water storage calculator."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> CoolingStorageSimulationResult:
        """Run the hourly cooling-storage calculation."""

        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        summary = self._summarize(results)
        return CoolingStorageSimulationResult(
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

        self.enabled = bool(cfg.get("enabled", True))
        self.storage_type = str(cfg.get("storage_type", "STO_TYPE_CW")).upper()
        if self.storage_type not in {"STO_TYPE_CW", "STO_TYPE_ICE", "STO_TYPE_PCM"}:
            raise ValueError(
                "storage_type must be STO_TYPE_CW, STO_TYPE_ICE or STO_TYPE_PCM."
            )
        self.location = str(cfg.get("location", cfg.get("CLG_STO_LOC", "NC"))).upper()
        self.theta_C_sto_C = float(cfg.get("theta_C_sto_C", cfg.get("storage_temperature_C", 7.0)))
        self.theta_C_gen_out_C = float(
            cfg.get("theta_C_gen_out_C", cfg.get("generator_outlet_temperature_C", 7.0))
        )
        self.theta_C_sto_out_C = float(
            cfg.get("theta_C_sto_out_C", cfg.get("storage_output_temperature_C", 7.0))
        )
        self.theta_C_sto_return_C = float(
            cfg.get("theta_C_sto_return_C", cfg.get("storage_return_temperature_C", 12.0))
        )
        self.delta_theta_C_sto_gen_flw_K = float(
            cfg.get("delta_theta_C_sto_gen_flw_K", 2.0)
        )
        self.ambient_temperature_C = float(cfg.get("ambient_temperature_C", 20.0))
        self.conditioned_ambient_temperature_C = float(
            cfg.get("conditioned_ambient_temperature_C", 26.0)
        )
        self.storage_volume_l = max(float(cfg.get("storage_volume_l", 100.0)), 0.0)

        h_default = 0.01 * self.storage_volume_l
        self.H_C_sto_tot_ls_W_K = max(
            float(cfg.get("H_C_sto_tot_ls_W_K", cfg.get("standby_loss_coefficient_W_K", h_default))),
            0.0,
        )
        self.H_C_sto_out_ls_W_K = max(
            float(
                cfg.get(
                    "H_C_sto_out_ls_W_K",
                    cfg.get("generator_loop_loss_coefficient_W_K", 0.3 * self.H_C_sto_tot_ls_W_K),
                )
            ),
            0.0,
        )
        self.H_C_sto_in_ls_W_K = max(
            float(
                cfg.get(
                    "H_C_sto_in_ls_W_K",
                    cfg.get("distribution_loop_loss_coefficient_W_K", 0.3 * self.H_C_sto_tot_ls_W_K),
                )
            ),
            0.0,
        )

        self.f_C_sto_ls_rbl = _fraction(
            cfg.get("f_C_sto_ls_rbl", cfg.get("thermal_loss_recoverable_fraction", 0.0)),
            "f_C_sto_ls_rbl",
        )
        self.f_C_aux_ls_rbl = _fraction(
            cfg.get("f_C_aux_ls_rbl", cfg.get("auxiliary_loss_recoverable_fraction", 0.75)),
            "f_C_aux_ls_rbl",
        )
        self.auxiliary_to_medium_fraction = _fraction(
            cfg.get("auxiliary_to_medium_fraction", 1.0),
            "auxiliary_to_medium_fraction",
        )

        self.input_pump_power_kW = max(float(cfg.get("input_pump_power_kW", 0.0)), 0.0)
        self.input_pump_flow_m3_h = max(float(cfg.get("input_pump_flow_m3_h", 0.0)), 0.0)
        self.input_pump_deltaT_K = _positive_float(
            cfg.get("input_pump_deltaT_K", 5.0), "input_pump_deltaT_K"
        )
        self.output_pump_power_kW = max(float(cfg.get("output_pump_power_kW", 0.0)), 0.0)
        self.output_pump_flow_m3_h = max(float(cfg.get("output_pump_flow_m3_h", 0.0)), 0.0)
        self.output_pump_deltaT_K = _positive_float(
            cfg.get("output_pump_deltaT_K", 5.0), "output_pump_deltaT_K"
        )
        self.operation_mode = str(cfg.get("operation_mode", "demand")).lower()
        self.control_mode = str(cfg.get("control_mode", cfg.get("CLG_STO_CTRL", "CONT"))).upper()
        if self.control_mode not in {"CONT", "TIME", "TEMP", "LOAD_PRED"}:
            raise ValueError("control_mode must be CONT, TIME, TEMP or LOAD_PRED.")
        schedule = cfg.get("discharge_schedule")
        self.discharge_schedule = None
        if schedule is not None:
            arr = np.asarray(schedule, dtype=float)
            if arr.shape != (24,) or not np.all(np.isin(arr, (0.0, 1.0))):
                raise ValueError("discharge_schedule must be 24 hourly values, each 0 or 1.")
            self.discharge_schedule = arr
        lim = cfg.get("theta_e_lim_C")
        self.theta_e_lim_C = None if lim is None else float(lim)
        if self.enabled and self.control_mode == "TIME" and self.discharge_schedule is None:
            raise ValueError("control_mode TIME needs discharge_schedule (24 hourly values 0/1) "
                             "or an input column 'f_C_sto_ctrl'.")
        if self.enabled and self.control_mode == "TEMP" and (
                self.theta_e_lim_C is None or not np.isfinite(self.theta_e_lim_C)):
            raise ValueError("control_mode TEMP needs theta_e_lim_C (EN 16798-15 (6c)).")
        self._load_medium_options(cfg)
        self.discharge_on_generator_shortfall = bool(cfg.get("discharge_on_generator_shortfall", True))
        self._gen_temp_given = "theta_C_gen_out_C" in cfg or "generator_outlet_temperature_C" in cfg
        cp = cfg.get("charge_power_kW")
        self.charge_power_kW = None if cp is None else _positive_float(cp, "charge_power_kW")
        if not np.isfinite(self.theta_C_sto_return_C) or not np.isfinite(self.theta_C_sto_C):
            raise ValueError("theta_C_sto_C and theta_C_sto_return_C must be finite.")

    def _load_medium_options(self, cfg: dict[str, Any]) -> None:
        """Stratification (CW) and latent medium data (ICE, PCM), Table 5 / Table B.2."""
        self.stratification_nodes = int(cfg.get("stratification_nodes", 1))
        if self.stratification_nodes < 1:
            raise ValueError("stratification_nodes must be an integer >= 1.")
        lim = cfg.get("theta_C_sto_out_max_C")
        self.theta_C_sto_out_max_C = None if lim is None else float(lim)
        self.latent = self.storage_type in {"STO_TYPE_ICE", "STO_TYPE_PCM"}
        if self.latent and self.stratification_nodes > 1:
            raise ValueError("stratification_nodes > 1 applies to STO_TYPE_CW only "
                             "(EN 16798-15 6.5.3.5.3: no stratification in latent storage).")
        if not self.latent:
            return
        ice = self.storage_type == "STO_TYPE_ICE"
        d = (dict(theta_sto_tr_C=0.0, rho_lqd=1000.0, rho_sld=900.0, Q_lat_Wh_kg=93.0,
                  Cp_sens_sld_Wh_kgK=0.54, Cp_sens_lqd_Wh_kgK=1.16, K_v_cr_kW_Km3=0.55,
                  K_v_fu_kW_Km3=2.1, d_C_sto_max_m=0.035)
             if ice else
             dict(theta_sto_tr_C=None, rho_lqd=560.0, rho_sld=560.0, Q_lat_Wh_kg=27.1,
                  Cp_sens_sld_Wh_kgK=0.392, Cp_sens_lqd_Wh_kgK=0.616, K_v_cr_kW_Km3=1.15,
                  K_v_fu_kW_Km3=1.85, d_C_sto_max_m=None))
        if not ice and cfg.get("theta_sto_tr_C") is None and self.enabled:
            raise ValueError("STO_TYPE_PCM needs the transition temperature theta_sto_tr_C.")
        vals = {}
        for key, default in d.items():
            v = cfg.get(key, default)
            vals[key] = None if v is None else float(v)
            if v is not None and (not np.isfinite(vals[key]) or (key != "theta_sto_tr_C" and vals[key] < 0)):
                raise ValueError(f"{key} must be finite and >= 0.")
        self.theta_sto_tr_C = vals["theta_sto_tr_C"]
        self.rho_lqd, self.rho_sld = vals["rho_lqd"], vals["rho_sld"]
        self.Q_lat_kWh_kg = vals["Q_lat_Wh_kg"] / 1000.0
        self.cp_sld_kWh_kgK = vals["Cp_sens_sld_Wh_kgK"] / 1000.0
        self.cp_lqd_kWh_kgK = vals["Cp_sens_lqd_Wh_kgK"] / 1000.0
        self.K_v_cr, self.K_v_fu = vals["K_v_cr_kW_Km3"], vals["K_v_fu_kW_Km3"]
        self.d_C_sto_max_m = vals["d_C_sto_max_m"]
        tube_d, tube_l = cfg.get("tube_outer_diameter_m"), cfg.get("tube_length_m")
        self.tube_outer_diameter_m = None if tube_d is None else _positive_float(tube_d, "tube_outer_diameter_m")
        self.tube_length_m = None if tube_l is None else _positive_float(tube_l, "tube_length_m")
        if (self.tube_outer_diameter_m is None) != (self.tube_length_m is None):
            raise ValueError("give both tube_outer_diameter_m and tube_length_m (ice on coil, (11)/(13)).")
        if "theta_C_sto_C" not in cfg and "storage_temperature_C" not in cfg and self.theta_sto_tr_C is not None:
            # charging temperature: transition temperature minus the generator approach
            self.theta_C_sto_C = self.theta_sto_tr_C - self.delta_theta_C_sto_gen_flw_K

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
        out.loc[:, "Q_C_sto_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_C_sto_out_kWh",
                "Q_C_dis_in_kWh",
                "Q_C_kWh",
                "QC_kWh",
                "cooling_kWh",
                "space_cooling_kWh",
            ],
            raw_aliases=["Q_C", "Cooling_needs"],
            fallback=np.maximum(-q_hc, 0.0),
        )
        out.loc[:, "theta_C_dis_in_req_C"] = _series_from_aliases(
            df,
            ["theta_C_dis_in_flw_req_C", "theta_C_dis_supply_C", "T_C_sink_C"],
            default=self.theta_C_sto_out_C,
        ).astype(float)
        out.loc[:, "theta_C_gen_out_C"] = _series_from_aliases(
            df,
            ["theta_C_gen_out_C", "theta_C_gen_out_req_C", "T_C_sink_C"],
            default=self.theta_C_gen_out_C,
        ).astype(float)
        if self.latent and "theta_C_gen_out_C" not in df.columns and not self._gen_temp_given:
            # latent store: unless the actual generator outlet temperature is given, the
            # generator is assumed to reach the charging temperature theta_C_sto_C
            out.loc[:, "theta_C_gen_out_C"] = self.theta_C_sto_C
        elif self.latent and "theta_C_gen_out_C" not in df.columns:
            out.loc[:, "theta_C_gen_out_C"] = self.theta_C_gen_out_C
        out.loc[:, "Q_C_sto_gen_avl_kWh"] = _series_from_aliases(
            df, ["Q_C_sto_gen_avl_kWh", "Q_C_gen_avl_kWh"], default=np.inf).astype(float)
        if (out["Q_C_sto_gen_avl_kWh"] < 0).any() or out["Q_C_sto_gen_avl_kWh"].isna().any():
            raise ValueError("Q_C_sto_gen_avl_kWh must be >= 0.")
        out.loc[:, "theta_C_sto_out_C"] = _series_from_aliases(
            df,
            ["theta_C_sto_out_C", "theta_C_dis_supply_C", "T_C_sink_C"],
            default=self.theta_C_sto_out_C,
        ).astype(float)
        out.loc[:, "theta_C_sto_return_C"] = _series_from_aliases(
            df,
            ["theta_C_sto_return_C", "theta_C_dis_return_C", "T_C_return_C"],
            default=self.theta_C_sto_return_C,
        ).astype(float)
        if any(c in df.columns for c in ("theta_C_sto_C", "storage_temperature_C")) and self.enabled:
            warnings.warn(
                "EN 16798-15: the storage temperature is now a state variable; the input "
                "column 'theta_C_sto_C' / 'storage_temperature_C' is ignored (set point: "
                "config 'theta_C_sto_C').",
                RuntimeWarning,
                stacklevel=3,
            )
        out.loc[:, "f_C_sto_ctrl"] = self._control_factor(df, out)
        out.loc[:, "theta_C_sto_amb_C"] = self._ambient_temperature(df, out)
        return out

    def _demand_from_columns(
        self,
        df: pd.DataFrame,
        kwh_aliases: list[str],
        raw_aliases: list[str],
        fallback: pd.Series | np.ndarray | float,
    ) -> pd.Series:
        series = _demand_series(df, kwh_aliases, "EN 16798-15")
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

    def _control_factor(self, source: pd.DataFrame, prepared: pd.DataFrame) -> pd.Series:
        """f_C;sto;ctrl, EN 16798-15 (6a)-(6c); an input column overrides the option."""
        if "f_C_sto_ctrl" in source.columns:
            f = source["f_C_sto_ctrl"].astype(float)
            if not np.all(np.isfinite(f)) or (f < 0).any() or (f > 1).any():
                raise ValueError("f_C_sto_ctrl must be finite and within [0, 1].")
            return f
        if self.control_mode == "LOAD_PRED" and self.enabled:
            raise ValueError("control_mode LOAD_PRED needs an input column 'f_C_sto_ctrl' "
                             "(defined by module M10-12).")
        if self.control_mode == "TIME" and self.discharge_schedule is not None:
            if not isinstance(source.index, pd.DatetimeIndex):
                raise ValueError("control_mode TIME needs a DatetimeIndex (or 'f_C_sto_ctrl').")
            return pd.Series(self.discharge_schedule[source.index.hour], index=source.index)
        if self.control_mode == "TEMP" and self.theta_e_lim_C is not None:
            t_e = prepared["T_ext_C"]
            if t_e.isna().any():
                raise ValueError("control_mode TEMP needs the outdoor temperature 'T_ext'.")
            return pd.Series(np.where(t_e < self.theta_e_lim_C, 0.0, 1.0), index=source.index)
        return pd.Series(1.0, index=source.index)

    def _ambient_temperature(self, source: pd.DataFrame, prepared: pd.DataFrame) -> pd.Series:
        direct = _series_from_aliases(
            source, ["theta_C_sto_amb_C", "cooling_storage_ambient_temperature_C"], default=None
        )
        if direct is not None:
            return direct.astype(float)
        if self.location in {"EXT", "CLG_STO_LOC_EXT", "EXTERNAL", "OUTDOOR"}:
            return prepared["T_ext_C"].fillna(self.ambient_temperature_C).astype(float)
        if self.location in {"CND", "CLG_STO_LOC_CND", "CONDITIONED"}:
            return pd.Series(self.conditioned_ambient_temperature_C, index=source.index)
        return pd.Series(self.ambient_temperature_C, index=source.index)

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy(deep=True)
        q_out = out["Q_C_sto_out_kWh"].astype(float)

        if not self.enabled:
            out.loc[:, "theta_C_sto_in_req_C"] = out["theta_C_dis_in_req_C"]
            out.loc[:, "Q_C_sto_out_ls_kWh"] = 0.0
            out.loc[:, "Q_C_sto_in_ls_kWh"] = 0.0
            out.loc[:, "Q_C_sto_ls_kWh"] = 0.0
            out.loc[:, "Q_C_sto_ls_tot_kWh"] = 0.0
            out.loc[:, "W_C_sto_aux_in_kWh"] = 0.0
            out.loc[:, "W_C_sto_aux_out_kWh"] = 0.0
            out.loc[:, "W_C_sto_aux_kWh"] = 0.0
            out.loc[:, "Q_C_sto_aux_rvd_kWh"] = 0.0
            out.loc[:, "Q_C_sto_aux_ls_rbl_kWh"] = 0.0
            out.loc[:, "Q_C_sto_ls_tot_rbl_kWh"] = 0.0
            out.loc[:, "Q_C_sto_in_kWh"] = q_out
            out.loc[:, "t_C_sto_pmp_in_h"] = 0.0
            out.loc[:, "t_C_sto_pmp_out_h"] = 0.0
            out.loc[:, "V_C_sto_l"] = self.storage_volume_l
            out.loc[:, "H_C_sto_tot_ls_W_K"] = 0.0
            out.loc[:, "H_C_sto_out_ls_W_K"] = 0.0
            out.loc[:, "H_C_sto_in_ls_W_K"] = 0.0
            out.loc[:, "T_C_sink_C"] = out["theta_C_sto_in_req_C"]
            for col in ("Q_C_sto_dis_kWh", "Q_C_sto_chg_kWh", "Q_C_sto_net_kWh",
                        "Q_C_sto_stored_kWh", "Q_C_sto_max_kWh"):
                out.loc[:, col] = 0.0
            supplied = np.minimum(q_out, out["Q_C_sto_gen_avl_kWh"])
            out.loc[:, "Q_C_sto_direct_kWh"] = supplied
            out.loc[:, "Q_C_sto_in_kWh"] = supplied
            out.loc[:, "Q_C_sto_unmet_kWh"] = q_out - supplied
            out.loc[:, "Q_C_sto_out_del_kWh"] = supplied
            out.loc[:, "Q_C_sto_chg_unmet_kWh"] = 0.0
            out.loc[:, "theta_C_sto_C"] = np.nan
            out.loc[:, "theta_C_sto_end_C"] = np.nan
            return out

        hours = out["hours"].astype(float)
        ambient = out["theta_C_sto_amb_C"].astype(float)
        gen_temp = out["theta_C_gen_out_C"].astype(float)
        out_temp = out["theta_C_sto_out_C"].astype(float)

        # loop losses (7), (8): supplied by the generator
        out.loc[:, "Q_C_sto_out_ls_kWh"] = (
            self.H_C_sto_out_ls_W_K * (ambient - gen_temp).clip(lower=0.0) * hours / 1000.0
        )
        out.loc[:, "Q_C_sto_in_ls_kWh"] = (
            self.H_C_sto_in_ls_W_K * (ambient - out_temp).clip(lower=0.0) * hours / 1000.0
        )

        out.loc[:, "Q_C_sto_avl_dir_chg_kWh"] = self._available_for_direct_and_charge(out)
        state = self._state_of_charge(out)
        for col, values in state.items():
            out.loc[:, col] = values
        out.loc[:, "Q_C_sto_ls_tot_kWh"] = (
            out["Q_C_sto_out_ls_kWh"]
            + out["Q_C_sto_in_ls_kWh"]
            + out["Q_C_sto_ls_kWh"]
        )
        # formula (1): generator flow temperature while recharging beyond the losses
        recharging = (out["Q_C_sto_chg_kWh"] - out["Q_C_sto_ls_kWh"] - out["Q_C_sto_dis_kWh"]) > 1e-9
        out.loc[:, "theta_C_sto_in_req_C"] = out["theta_C_dis_in_req_C"].where(
            ~recharging, out["theta_C_sto_C"] - self.delta_theta_C_sto_gen_flw_K
        )

        out.loc[:, "Q_C_sto_out_del_kWh"] = q_out - out["Q_C_sto_unmet_kWh"]
        t_out = self._pump_runtime(out["Q_C_sto_out_del_kWh"], hours, self.output_pump_flow_m3_h,
                                   self.output_pump_deltaT_K)
        q_to_generator = (
            out["Q_C_sto_direct_kWh"] + out["Q_C_sto_chg_kWh"]
            + out["Q_C_sto_out_ls_kWh"] + out["Q_C_sto_in_ls_kWh"]
        )
        t_in = self._pump_runtime(
            q_to_generator,
            hours,
            self.input_pump_flow_m3_h,
            self.input_pump_deltaT_K,
        )
        out.loc[:, "t_C_sto_pmp_in_h"] = t_in
        out.loc[:, "t_C_sto_pmp_out_h"] = t_out
        out.loc[:, "W_C_sto_aux_in_kWh"] = t_in * self.input_pump_power_kW
        out.loc[:, "W_C_sto_aux_out_kWh"] = t_out * self.output_pump_power_kW
        out.loc[:, "W_C_sto_aux_kWh"] = out["W_C_sto_aux_in_kWh"] + out["W_C_sto_aux_out_kWh"]

        aux_to_fluid = out["W_C_sto_aux_kWh"] * self.auxiliary_to_medium_fraction
        out.loc[:, "Q_C_sto_aux_rvd_kWh"] = -aux_to_fluid
        out.loc[:, "Q_C_sto_aux_ls_rbl_kWh"] = -out["W_C_sto_aux_kWh"] * self.f_C_aux_ls_rbl
        out.loc[:, "Q_C_sto_ls_tot_rbl_kWh"] = (
            -out["Q_C_sto_ls_tot_kWh"] * self.f_C_sto_ls_rbl
            + out["Q_C_sto_aux_ls_rbl_kWh"]
        )
        out.loc[:, "Q_C_sto_in_kWh"] = (q_to_generator + aux_to_fluid).clip(lower=0.0)
        out.loc[:, "V_C_sto_l"] = self.storage_volume_l
        out.loc[:, "H_C_sto_tot_ls_W_K"] = self.H_C_sto_tot_ls_W_K
        out.loc[:, "H_C_sto_out_ls_W_K"] = self.H_C_sto_out_ls_W_K
        out.loc[:, "H_C_sto_in_ls_W_K"] = self.H_C_sto_in_ls_W_K
        out.loc[:, "T_C_sink_C"] = out["theta_C_sto_in_req_C"]
        return out

    def _available_for_direct_and_charge(self, df: pd.DataFrame) -> pd.Series:
        """Generator energy left for direct supply and charging once the loop losses (7), (8)
        and the pump heat are covered. The pump heat is estimated without the runtime clip,
        so the estimate never exceeds the real margin."""
        avl = df["Q_C_sto_gen_avl_kWh"].astype(float)
        if np.isinf(avl).all():
            return avl
        hours = df["hours"].astype(float)
        f = self.auxiliary_to_medium_fraction
        loop = df["Q_C_sto_out_ls_kWh"] + df["Q_C_sto_in_ls_kWh"]
        fixed = pd.Series(0.0, index=df.index)
        per_kwh = 0.0
        if self.output_pump_flow_m3_h > _KWH_EPS:
            if self.operation_mode == "continuous":
                fixed = fixed + f * hours * self.output_pump_power_kW
            else:
                fixed = fixed + f * df["Q_C_sto_out_kWh"] * self.output_pump_power_kW / (
                    _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * self.output_pump_flow_m3_h * self.output_pump_deltaT_K)
        if self.input_pump_flow_m3_h > _KWH_EPS:
            if self.operation_mode == "continuous":
                fixed = fixed + f * hours * self.input_pump_power_kW
            else:
                per_kwh = f * self.input_pump_power_kW / (
                    _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * self.input_pump_flow_m3_h * self.input_pump_deltaT_K)
        # avl = (direct + charge + loop) (1 + per_kwh) + fixed: the input pump also moves the loop losses
        return ((avl - fixed) / (1.0 + per_kwh) - loop).clip(lower=0.0)

    def _make_store(self):
        if self.latent:
            return _LatentStore(self)
        return _StratifiedTank(self)

    def _state_of_charge(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Hourly balance of the stored cold, EN 16798-15 (2)-(5), (9), (10)."""
        n = len(df)
        hours = df["hours"].to_numpy(dtype=float)
        demand = df["Q_C_sto_out_kWh"].to_numpy(dtype=float)
        f_ctrl = df["f_C_sto_ctrl"].to_numpy(dtype=float)
        ambient = df["theta_C_sto_amb_C"].to_numpy(dtype=float)
        t_gen = df["theta_C_gen_out_C"].to_numpy(dtype=float)
        avl = df["Q_C_sto_avl_dir_chg_kWh"].to_numpy(dtype=float)
        charge_always = self.control_mode == "CONT"
        store = self._make_store()

        cols: dict[str, np.ndarray] = {}

        def put(key, i, value):
            cols.setdefault(key, np.full(n, np.nan))[i] = value

        for i in range(n):
            store.set_charge_limit(t_gen[i])
            theta = store.mean_temperature()
            put("theta_C_sto_C", i, theta)
            store.snapshot()                      # losses with the start-of-step temperatures
            request = f_ctrl[i] * demand[i]                                            # (5)
            if self.discharge_on_generator_shortfall and demand[i] - avl[i] > request:
                request = demand[i] - avl[i]      # generator short: the store covers what it can
            discharge, t_out = store.discharge(request, hours[i])
            direct = min(demand[i] - discharge, avl[i])          # users before charging
            unmet = demand[i] - discharge - direct
            loss = self.H_C_sto_tot_ls_W_K * hours[i] / 1000.0
            absorbed = store.add_gains(loss, ambient[i], allow_negative=True)          # (9)
            charge, charge_short = 0.0, 0.0
            if charge_always or f_ctrl[i] < 1.0:
                limit = np.inf if self.charge_power_kW is None else self.charge_power_kW * hours[i]
                if not charge_always:
                    limit *= 1.0 - f_ctrl[i]
                margin = max(avl[i] - direct, 0.0)
                if margin < limit:
                    wanted = copy.deepcopy(store).charge(limit, hours[i])
                    charge = store.charge(margin, hours[i])
                    charge_short = max(wanted - charge, 0.0)
                else:
                    charge = store.charge(limit, hours[i])
            absorbed -= store.clip_empty()      # no cold left: medium at the return temperature
            put("theta_C_sto_end_C", i, store.mean_temperature())
            put("theta_C_sto_out_act_C", i, t_out)
            put("Q_C_sto_stored_kWh", i, store.stored())
            put("Q_C_sto_max_kWh", i, store.q_max())
            put("Q_C_sto_dis_kWh", i, discharge)
            put("Q_C_sto_direct_kWh", i, direct)
            put("Q_C_sto_unmet_kWh", i, unmet)
            put("Q_C_sto_chg_kWh", i, charge)
            put("Q_C_sto_chg_unmet_kWh", i, charge_short)
            put("Q_C_sto_ls_kWh", i, absorbed)
            put("Q_C_sto_net_kWh", i, discharge - charge)
            for key, value in store.diagnostics().items():
                put(key, i, value)
        return cols

    def _pump_runtime(
        self,
        q_kWh: pd.Series,
        hours: pd.Series,
        flow_m3_h: float,
        deltaT_K: float,
    ) -> pd.Series:
        if flow_m3_h <= _KWH_EPS:
            return pd.Series(0.0, index=q_kWh.index)
        if self.operation_mode == "continuous":
            return hours
        power_kW = _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * flow_m3_h * deltaT_K
        runtime = q_kWh / power_kW
        return runtime.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(
            lower=0.0,
            upper=hours,
        )

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        q = results["Q_C_sto_out_kWh"]
        weights = q + _KWH_EPS
        summary = {
            "hours": float(results["hours"].sum()),
            "QC_sto_out_kWh": float(results["Q_C_sto_out_kWh"].sum()),
            "QC_sto_in_kWh": float(results["Q_C_sto_in_kWh"].sum()),
            "QC_sto_ls_kWh": float(results["Q_C_sto_ls_kWh"].sum()),
            "QC_sto_out_ls_kWh": float(results["Q_C_sto_out_ls_kWh"].sum()),
            "QC_sto_in_ls_kWh": float(results["Q_C_sto_in_ls_kWh"].sum()),
            "QC_sto_ls_tot_kWh": float(results["Q_C_sto_ls_tot_kWh"].sum()),
            "QC_sto_ls_tot_rbl_kWh": float(results["Q_C_sto_ls_tot_rbl_kWh"].sum()),
            "WC_sto_aux_kWh": float(results["W_C_sto_aux_kWh"].sum()),
            "QC_sto_aux_rvd_kWh": float(results["Q_C_sto_aux_rvd_kWh"].sum()),
            "QC_sto_dis_kWh": float(results["Q_C_sto_dis_kWh"].sum()),
            "QC_sto_chg_kWh": float(results["Q_C_sto_chg_kWh"].sum()),
            "QC_sto_direct_kWh": float(results["Q_C_sto_direct_kWh"].sum()),
            "QC_sto_max_kWh": float(results["Q_C_sto_max_kWh"].max()),
            "QC_sto_unmet_kWh": float(results["Q_C_sto_unmet_kWh"].sum()),
            "QC_sto_chg_unmet_kWh": float(results["Q_C_sto_chg_unmet_kWh"].sum()),
            "theta_C_sto_mean_C": _weighted_mean(results["theta_C_sto_C"], weights),
            "theta_C_sto_amb_mean_C": _weighted_mean(results["theta_C_sto_amb_C"], weights),
            "e_C_sto": _ratio(
                float(results["Q_C_sto_in_kWh"].sum()),
                float(results["Q_C_sto_out_kWh"].sum()),
            ),
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


class _StratifiedTank:
    """Chilled-water tank with N equal nodes and plug flow (N = 1: fully mixed).

    Node 0 is the bottom (coldest, outlet to the distribution), node N-1 the top.
    Stored cold = sum m_i c (theta_ret - theta_i), (2).
    """

    def __init__(self, calc: "CoolingStorageSystemCalculator"):
        self.n = calc.stratification_nodes
        self.cap = _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * calc.storage_volume_l / 1000.0  # kWh/K
        self.cap_node = self.cap / self.n
        self.t_ret = calc.theta_C_sto_return_C
        self.t_set = calc.theta_C_sto_C
        self.t_out_max = calc.theta_C_sto_out_max_C
        self.t_charge = self.t_set
        self.temps = np.full(self.n, self.t_set)       # step 0: tank at the set point

    def set_charge_limit(self, t_gen: float) -> None:
        self.t_charge = max(self.t_set, t_gen)

    def snapshot(self) -> None:
        self._snap = self.temps.copy()

    def q_max(self) -> float:
        return max(self.cap * (self.t_ret - self.t_charge), 0.0)

    def stored(self) -> float:
        return float(self.cap_node * np.sum(self.t_ret - self.temps))

    def mean_temperature(self) -> float:
        return float(np.mean(self.temps)) if self.cap > _KWH_EPS else self.t_set

    def diagnostics(self) -> dict[str, float]:
        if self.n == 1:
            return {}
        return {"theta_C_sto_bottom_C": float(self.temps[0]), "theta_C_sto_top_C": float(self.temps[-1])}

    def discharge(self, request: float, hours: float) -> tuple[float, float]:
        """Draw from the bottom, warm return water enters at the top."""
        delivered, vol = 0.0, 0.0
        t_out_sum = 0.0
        if self.cap <= _KWH_EPS:
            return 0.0, float("nan")
        for _ in range(100 * self.n):
            remaining = request - delivered
            bottom = self.temps[0]
            if remaining <= 1e-12 or bottom >= self.t_ret - 1e-9:
                break
            if self.t_out_max is not None and bottom > self.t_out_max + 1e-9:
                break
            f = min(1.0, remaining / (self.cap_node * (self.t_ret - bottom)))
            delivered += f * self.cap_node * (self.t_ret - bottom)
            t_out_sum += f * bottom
            vol += f
            upper = np.append(self.temps[1:], self.t_ret)
            self.temps = (1.0 - f) * self.temps + f * upper
        t_out = t_out_sum / vol if vol > 0 else float(self.temps[0])
        return delivered, t_out

    def add_gains(self, h_kwh_k: float, ambient: float, allow_negative: bool) -> float:
        """Storage losses (9), shared equally by the nodes; returns the energy absorbed."""
        if self.cap <= _KWH_EPS:
            return 0.0
        gains = (h_kwh_k / self.n) * np.maximum(ambient - self._snap, 0.0)
        self.temps = self.temps + gains / self.cap_node
        return float(gains.sum())

    def charge(self, limit: float, hours: float) -> float:
        """Cold water at t_charge enters at the bottom, the top node leaves."""
        if self.cap <= _KWH_EPS:
            return 0.0
        charged = 0.0
        for _ in range(100 * self.n):
            remaining = limit - charged
            top = self.temps[-1]
            if remaining <= 1e-12 or top <= self.t_charge + 1e-12:
                break
            f = min(1.0, remaining / (self.cap_node * (top - self.t_charge)))
            charged += f * self.cap_node * (top - self.t_charge)
            lower = np.insert(self.temps[:-1], 0, self.t_charge)
            self.temps = (1.0 - f) * self.temps + f * lower
        # numerical floor: nodes never colder than the charging temperature
        return charged

    def clip_empty(self) -> float:
        """Nodes warmer than the return temperature are reset to it; returns the excess."""
        excess = float(self.cap_node * np.sum(np.maximum(self.temps - self.t_ret, 0.0)))
        self.temps = np.minimum(self.temps, self.t_ret)
        return excess


class _LatentStore:
    """Ice / PCM store, enthalpy model with liquid, latent and solid bands, (2)-(4), (11)-(22)."""

    def __init__(self, calc: "CoolingStorageSystemCalculator"):
        self.ice = calc.storage_type == "STO_TYPE_ICE"
        self.vol_m3 = calc.storage_volume_l / 1000.0
        self.mass = calc.rho_lqd * self.vol_m3
        self.t_ret = calc.theta_C_sto_return_C
        self.t_tr = calc.theta_sto_tr_C
        self.t_min = calc.theta_C_sto_C
        self.cp_l, self.cp_s, self.lat = calc.cp_lqd_kWh_kgK, calc.cp_sld_kWh_kgK, calc.Q_lat_kWh_kg
        self.k_cr, self.k_fu = calc.K_v_cr, calc.K_v_fu
        self.rho_s = calc.rho_sld
        self.tube_d, self.tube_l = calc.tube_outer_diameter_m, calc.tube_length_m
        self.d_max = calc.d_C_sto_max_m
        m_max = self.mass
        if self.ice and self.tube_d is not None and self.d_max is not None:
            r = self.tube_d / 2.0
            m_max = min(m_max, self.rho_s * np.pi * self.tube_l * ((r + self.d_max) ** 2 - r ** 2))  # (13)
        freezing = self.t_min < self.t_tr and self.t_ret > self.t_tr
        self.m_max = m_max if freezing else 0.0
        floor = self.t_tr if freezing else min(self.t_min, self.t_ret)
        self.e_l = self.mass * self.cp_l * max(self.t_ret - floor, 0.0)                 # (2)
        self.e_lat = self.m_max * self.lat                                              # (3)
        k = 0.5 if self.ice else 1.0              # ice: linear profile in the ice, (12)
        self.e_s = self.m_max * self.cp_s * k * max(self.t_tr - self.t_min, 0.0) if freezing else 0.0  # (4)
        self.e_max = self.e_l + self.e_lat + self.e_s
        self.energy = self.e_max                  # step 0: charged to the set point
        self.t_charge = self.t_min

    def set_charge_limit(self, t_gen: float) -> None:
        """Charging temperature: theta_C_sto_C or the warmer actual generator outlet."""
        self.t_charge = max(self.t_min, t_gen)

    def _reachable(self) -> float:
        """Stored cold reachable with the current charging temperature."""
        tc = self.t_charge
        if self.m_max <= 0.0 or tc >= self.t_tr:
            floor = self.t_tr if self.m_max > 0.0 else min(self.t_min, self.t_ret)
            return self.mass * self.cp_l * max(self.t_ret - max(tc, floor), 0.0)
        k = 0.5 if self.ice else 1.0
        return self.e_l + self.e_lat + self.m_max * self.cp_s * k * (self.t_tr - tc)

    def snapshot(self) -> None:
        self._snap = self.mean_temperature()

    def q_max(self) -> float:
        return self.e_max

    def stored(self) -> float:
        return self.energy

    def solid_mass(self) -> float:
        if self.m_max <= 0.0:
            return 0.0
        return float(min(max((self.energy - self.e_l) / self.lat, 0.0), self.m_max))

    def mean_temperature(self) -> float:
        e = self.energy
        if e <= self.e_l or self.m_max <= 0.0:
            return self.t_ret - e / (self.mass * self.cp_l) if self.mass > 0 else self.t_min
        if e <= self.e_l + self.e_lat or self.ice:
            return self.t_tr                      # ice: water around the ice at 0 C
        return self.t_tr - (e - self.e_l - self.e_lat) / (self.m_max * self.cp_s)   # PCM (18)

    def diagnostics(self) -> dict[str, float]:
        m = self.solid_mass()
        out = {"m_C_sto_sld_kg": m, "f_C_sto_sld": m / self.mass if self.mass > 0 else 0.0}
        if self.ice and self.tube_d is not None:
            r = self.tube_d / 2.0
            out["d_C_sto_m"] = float(np.sqrt(r ** 2 + m / (self.rho_s * np.pi * self.tube_l)) - r)  # (13)
        return out

    def _move(self, amount: float, latent_limit: float, sign: int) -> float:
        """Move energy through the bands (sign -1: discharge, +1: charge)."""
        bands = [(0.0, self.e_l, np.inf),
                 (self.e_l, self.e_l + self.e_lat, latent_limit),
                 (self.e_l + self.e_lat, self.e_max, np.inf)]
        if sign < 0:
            bands = bands[::-1]
        moved = 0.0
        for lo, hi, cap in bands:
            if amount - moved <= 1e-12:
                break
            if sign < 0:
                avail = max(min(self.energy, hi) - lo, 0.0) if self.energy > lo else 0.0
            else:
                avail = max(hi - max(self.energy, lo), 0.0) if self.energy < hi else 0.0
            step = min(amount - moved, avail, cap)
            self.energy += sign * step
            moved += step
            if step < avail - 1e-12:              # band not crossed: stop here
                break
        return moved

    def discharge(self, request: float, hours: float) -> tuple[float, float]:
        melt = self.k_fu * self.vol_m3 * max(self.t_ret - self.t_tr, 0.0) * hours
        delivered = self._move(max(request, 0.0), melt, -1)
        return delivered, self.mean_temperature()

    def add_gains(self, h_kwh_k: float, ambient: float, allow_negative: bool) -> float:
        gain = h_kwh_k * max(ambient - self._snap, 0.0)
        self.energy -= gain
        return gain

    def charge(self, limit: float, hours: float) -> float:
        if self.energy < 0.0:                     # losses beyond the stored cold first
            deficit = min(-self.energy, limit)
            self.energy += deficit
            limit -= deficit
        else:
            deficit = 0.0
        limit = min(limit, max(self._reachable() - self.energy, 0.0))
        freeze = self.k_cr * self.vol_m3 * max(self.t_tr - self.t_charge, 0.0) * hours
        return deficit + self._move(limit, freeze, +1)

    def clip_empty(self) -> float:
        excess = max(-self.energy, 0.0)
        self.energy = max(self.energy, 0.0)
        return excess


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


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return numerator / denominator
