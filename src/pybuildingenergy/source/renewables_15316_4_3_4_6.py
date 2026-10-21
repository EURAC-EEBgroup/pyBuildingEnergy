"""Solar thermal and PV calculation for EN 15316-4-3:2017 workflows.

Status against the standard
---------------------------
* Method 3 (solar thermal, hourly, 6.1.3): implemented, ``solar_thermal.method =
  "method_3"``. Eq. (49)-(61) and the Table 14 outputs (``Q_sol_gen_kWh``,
  ``Q_sol_col_out_kWh``, ``Q_sol_loop_out_kWh``, ``Q_sol_loop_out_req_kWh``,
  ``Q_sto_sol_in_kWh``, ``Q_sol_loop_ls_kWh``, ``Q_sol_loop_ls_rbl_kWh``,
  ``E_sol_aux_kWh``, ``N_sol_it``, ``SOL_FINAL``). With a ``storage_response`` the
  collector/storage iteration of 6.1.3.4.2 runs (4 passes by default); without it
  the mean collector temperature is the fixed ``mean_temperature_C`` and
  ``N_sol_it`` is 0. ``Q_sol_loop_out_kWh`` is the
  proposed loop output (eq. 57); ``Q_sol_loop_out_req_kWh`` the request of the final
  storage call, capped at what the previous pass placed; ``Q_sto_sol_in_kWh`` the
  heat the storage accepted. The legacy column ``Q_solar_thermal_collector_kWh`` is the
  collector *output* ``Q_sol_col_out_kWh`` (eq. 55), not the absorber heat
  ``Q_sol_gen_kWh`` (eq. 54). Irradiance must be on the collector plane: pass it
  as ``I_sol_collector_plane_W_m2`` or give DNI/DHI plus the site and orientation
  (EN ISO 52010-1); a horizontal ``GHI`` fallback warns.
* Eq. (60) as printed adds the pump to the controller consumption when
  ``Q_sol;loop;out = 0``. That reads as a typo (a pump idles when nothing is
  delivered), so by default the pump is counted while the loop delivers heat.
  ``solar_thermal.aux_literal_eq60 = True`` applies the printed text. No
  corrigendum or interpretation has been obtained from CEN; treat the default as
  a documented deviation.
* Eq. (56) loop losses are floored at zero by default (``loop_loss_clip``);
  ``False`` applies the formula literally.
* Eq. (61): recoverable loop losses equal the loop losses only in hours with a
  space-heating demand.
* ``solar_thermal.method = "annual_yield"`` (the default) is NOT Method 1 or 2 of
  the standard: it scales ``area * annual_yield * efficiency * (1 - losses)`` over
  a daylight profile. It uses no EN 12976/12977 system test data, no monthly
  f-chart and no Annex correlation coefficients. Methods 1 and 2 are not
  implemented.
* PV Method 4 (``pv.method = "method_4"``, eq. 62-64), Method 5 (``"method_5"``,
  eq. 65-67, monthly) and Method 6 (``"method_6"``/``"hourly"``, eq. 68), with the
  Table 25 outputs (``E_el_pv_out_kWh`` and zero thermal output, losses,
  recoverable losses, auxiliary and energy input). Method 6 needs the irradiation
  on the module plane (``E_sol_pv_kWh_m2`` or ``G_pv_W_m2``) or DNI/DHI with the
  site and orientation; otherwise it warns and uses the horizontal-irradiance
  profile scaled to the annual yield, which is not the standard's "solar module".
  ``pv.method = "annual_yield"`` is the project's own annual estimate, valid when
  ``annual_yield_kWh_kWp`` already refers to the module plane and contains the
  losses. Self-consumption, battery and export are additions beyond the standard.
* Not available: a comparison with numerical examples of the standard (the
  document has none), and shading of the PV/collector plane inside this module
  (supply a shaded irradiance).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

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

    def __init__(
        self,
        input_data: dict[str, Any] | None = None,
        storage_response: Callable[[int, float, bool], tuple[float, float]] | None = None,
    ):
        """
        :param storage_response: optional storage model for the Method 3 iteration
            (EN 15316-4-3:2017, 6.1.3.4.2). Called as
            ``storage_response(step_index, q_req_kWh, commit)`` and returns
            ``(theta_loop_in_C, q_accepted_kWh)``: the temperature returned to the
            collector loop and the heat the storage accepts. ``commit`` is False in
            the trial iterations (the model must not change state) and True on the
            final one. Without it Method 3 uses the fixed ``mean_temperature_C``.
        """
        self.input_data = dict(input_data or {})
        self.storage_response = storage_response
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
        self.solar_thermal_shading_loss_fraction = _fraction(
            solar.get("shading_loss_fraction", solar.get("collector_shading_loss_fraction", 0.0)),
            "solar_thermal.shading_loss_fraction",
        )
        self.solar_thermal_storage_loss_fraction = _fraction(
            solar.get("storage_loss_fraction", 0.0),
            "solar_thermal.storage_loss_fraction",
        )
        self.solar_thermal_usable_fraction = _fraction(
            solar.get("usable_fraction", 1.0),
            "solar_thermal.usable_fraction",
        )
        self.solar_thermal_method = str(solar.get("method", "annual_yield")).lower()
        if self.solar_thermal_method not in {"annual_yield", "hourly", "method_3"}:
            raise ValueError("solar_thermal.method must be annual_yield or hourly.")
        self.solar_collector_eta_0 = _fraction(solar.get("eta_0", solar.get("peak_efficiency", 0.75)), "solar_thermal.eta_0")
        self.solar_collector_a1 = max(float(solar.get("a1_W_m2K", 3.5)), 0.0)
        self.solar_collector_a2 = max(float(solar.get("a2_W_m2K2", 0.015)), 0.0)
        self.solar_collector_khem50 = max(float(solar.get("khem_50", 1.0)), 0.0)
        self.solar_collector_area_m2 = max(float(solar.get("collector_area_m2", self.solar_thermal_area_m2)), 0.0)
        self.solar_loop_heat_loss_W_K = max(float(solar.get("loop_heat_loss_W_K", 0.0)), 0.0)
        self.solar_ambient_location = str(solar.get("location", "OUT")).upper()
        self.solar_indoor_temperature_C = float(solar.get("indoor_temperature_C", 20.0))
        self.solar_mean_temperature_given = solar.get("mean_temperature_C") is not None
        self.solar_collector_mean_temperature_C = float(solar.get("mean_temperature_C", 50.0))
        # True: refuse the horizontal-irradiance fallback instead of warning
        self.solar_require_plane_irradiance = bool(solar.get("require_plane_irradiance", False))
        self.solar_aux_power_W = max(float(solar.get("pump_power_W", 0.0)), 0.0)
        # EN 15316-4-3:2017 eq. (56) has no lower limit: with the collector colder
        # than its surroundings the loop "loss" is negative (a gain). True (default)
        # floors it at zero; False applies the formula literally.
        self.solar_loop_loss_clip = bool(solar.get("loop_loss_clip", True))
        # Collector-plane irradiance from the beam/diffuse components (EN ISO 52010-1)
        # when no collector-plane column is supplied: needs the site and the
        # collector orientation. azimuth_deg is geographical (N=0, E=90, S=180, W=270).
        self.solar_site = {
            key: solar.get(key)
            for key in ("latitude_deg", "longitude_deg", "timezone_h", "tilt_deg", "azimuth_deg")
        }
        self.solar_ground_reflectivity = _fraction(
            solar.get("ground_reflectivity", 0.2), "solar_thermal.ground_reflectivity"
        )
        # Method 3 iteration (needs a storage_response): collector mass flow, number
        # of storage calls (4, Annex A allows alternatives) and the starting inlet temperature
        mass_flow = solar.get("mass_flow_kg_s")
        if mass_flow is None and solar.get("mass_flow_kg_s_m2") is not None:
            mass_flow = float(solar["mass_flow_kg_s_m2"]) * self.solar_collector_area_m2
        self.solar_mass_flow_kg_s = None if mass_flow is None else max(float(mass_flow), 0.0)
        self.solar_iterations = max(int(solar.get("iterations", 4)), 1)
        self.solar_initial_inlet_temperature_C = float(solar.get("initial_loop_inlet_temperature_C", 20.0))
        # Eq. (60) as printed puts the pump on the Q_sol;loop;out = 0 branch; False
        # (default) runs it while the loop delivers heat, True applies the text literally.
        self.solar_aux_literal_eq60 = bool(solar.get("aux_literal_eq60", False))
        self.solar_controller_power_W = max(float(solar.get("controller_power_W", 0.0)), 0.0)
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
        self.pv_shading_loss_fraction = _fraction(
            pv.get("shading_loss_fraction", pv.get("pv_shading_loss_fraction", 0.0)),
            "pv.shading_loss_fraction",
        )
        self.pv_orientation_loss_fraction = _fraction(
            pv.get("orientation_loss_fraction", 0.0),
            "pv.orientation_loss_fraction",
        )
        self.pv_degradation_loss_fraction = _fraction(
            pv.get("degradation_loss_fraction", 0.0),
            "pv.degradation_loss_fraction",
        )
        self.pv_self_consumption_fraction = _fraction(
            pv.get("self_consumption_fraction", 1.0), "pv.self_consumption_fraction"
        )
        self.pv_self_consumption_mode = str(
            pv.get("self_consumption_mode", "fixed_fraction")
        ).lower()
        self.pv_method = str(pv.get("method", "annual_yield")).lower()
        if self.pv_method not in {"annual_yield", "hourly", "method_4", "method_5", "method_6"}:
            raise ValueError("pv.method must be annual_yield, method_4, method_5 or hourly/method_6.")
        # EN 15316-4-3 Methods 4/5: horizontal irradiation, tilt/orientation factor f_tlt,
        # peak power Ppk = Kpk * A (eq. 64/67) and system performance factor f_perf.
        self.pv_require_plane_irradiation = bool(pv.get("require_plane_irradiation", False))
        self.pv_perf_factor = _fraction(
            pv.get("performance_factor", pv.get("f_perf", pv.get("performance_ratio", 0.80))),
            "pv.performance_factor",
        )
        self.pv_annual_horizontal_kWh_m2 = pv.get("annual_irradiation_horizontal_kWh_m2")
        self.pv_monthly_horizontal_kWh_m2 = pv.get("monthly_irradiation_horizontal_kWh_m2")
        self.pv_monthly_tilt_factor = pv.get("tilt_orientation_factor_monthly")
        self.pv_site = {
            key: pv.get(key)
            for key in ("latitude_deg", "longitude_deg", "timezone_h", "tilt_deg", "azimuth_deg")
        }
        self.pv_ground_reflectivity = _fraction(pv.get("ground_reflectivity", 0.2), "pv.ground_reflectivity")
        self.pv_area_m2 = max(float(pv.get("area_m2", 0.0)), 0.0)
        self.pv_peak_power_coefficient_kW_m2 = max(float(pv.get("peak_power_coefficient_kW_m2", pv.get("Kpk_kW_m2", 0.0))), 0.0)
        self.pv_tilt_orientation_factor = max(float(pv.get("tilt_orientation_factor", 1.0)), 0.0)
        if self.pv_self_consumption_mode not in {"fixed_fraction", "dispatch"}:
            raise ValueError("pv.self_consumption_mode must be fixed_fraction or dispatch.")
        battery = dict(pv.get("battery", cfg.get("battery", {})) or {})
        self.battery_capacity_kWh = max(float(battery.get("capacity_kWh", 0.0)), 0.0)
        self.battery_roundtrip_efficiency = _fraction(
            battery.get("roundtrip_efficiency", 0.90),
            "battery.roundtrip_efficiency",
        )
        self.battery_initial_soc_fraction = _fraction(
            battery.get("initial_soc_fraction", 0.50),
            "battery.initial_soc_fraction",
        )
        self.battery_min_soc_fraction = _fraction(
            battery.get("min_soc_fraction", 0.0),
            "battery.min_soc_fraction",
        )
        self.battery_max_charge_power_kW = max(
            float(battery.get("max_charge_power_kW", self.battery_capacity_kWh)),
            0.0,
        )
        self.battery_max_discharge_power_kW = max(
            float(battery.get("max_discharge_power_kW", self.battery_capacity_kWh)),
            0.0,
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
        for target, aliases in {
            "E_sol_pv_kWh_m2": ["E_sol_pv_kWh_m2", "solar_irradiation_pv_kWh_m2", "G_pv_kWh_m2"],
            "G_pv_W_m2": ["G_pv_W_m2", "solar_irradiance_pv_W_m2"],
            "DNI_W_m2": ["DNI", "dni_W_m2", "Gb(n)", "direct_normal_irradiance_W_m2"],
            "DHI_W_m2": ["DHI", "dhi_W_m2", "Gd(h)", "diffuse_horizontal_irradiance_W_m2"],
            "I_sol_collector_W_m2": [
                "I_sol_collector_plane_W_m2",
                "I_sol_collector_W_m2",
                "solar_irradiance_collector_W_m2",
            ],
            "T_ext_C": ["T_ext_C", "T_ext", "outdoor_temperature_C"],
        }.items():
            out.loc[:, target] = _series_from_aliases(df, aliases, default=np.nan).astype(float)
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
        if self.solar_thermal_enabled and self.solar_thermal_area_m2 > 0.0 and self.solar_thermal_method in {"hourly", "method_3"}:
            q_solar_available = self._simulate_solar_thermal_hourly(out)
        elif self.solar_thermal_enabled and self.solar_thermal_area_m2 > 0.0:
            annual = (
                self.solar_thermal_area_m2
                * self.solar_thermal_yield_kWh_m2_a
                * self.solar_thermal_efficiency
                * (1.0 - self.solar_thermal_shading_loss_fraction)
                * (1.0 - self.solar_thermal_storage_loss_fraction)
                * self.solar_thermal_usable_fraction
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
        if self.pv_enabled and (self.pv_capacity_kWp > 0.0 or self.pv_area_m2 > 0.0):
            plane = out["E_sol_pv_kWh_m2"].copy()
            if plane.isna().all() and out["G_pv_W_m2"].notna().any():
                plane = out["G_pv_W_m2"].fillna(0.0).clip(lower=0.0) * out["hours"] / 1000.0
            ppk = self.pv_capacity_kWp
            if ppk <= _KWH_EPS and self.pv_area_m2 > 0 and self.pv_peak_power_coefficient_kW_m2 > 0:
                ppk = self.pv_area_m2 * self.pv_peak_power_coefficient_kW_m2
            if self.pv_method in {"hourly", "method_6"} and plane.isna().all():
                # no plane-of-array input: derive it from DNI/DHI with the site and module orientation
                derived = self._collector_plane_irradiance(out, self.pv_site, self.pv_ground_reflectivity)
                if derived is not None:
                    plane = derived * out["hours"] / 1000.0
                elif self.pv_require_plane_irradiation:
                    raise ValueError(
                        "EN 15316-4-3 Method 6 needs the irradiation on the PV plane "
                        "(E_sol_pv_kWh_m2, G_pv_W_m2, or DNI/DHI with site and orientation); "
                        "require_plane_irradiation is set."
                    )
                else:
                    warnings.warn(
                        "EN 15316-4-3 Method 6: no PV-plane irradiation (E_sol_pv_kWh_m2, G_pv_W_m2, "
                        "or DNI/DHI with latitude/longitude/timezone/tilt/azimuth); falling back to "
                        "the horizontal-irradiance profile scaled to the annual yield.",
                        UserWarning,
                        stacklevel=2,
                    )
            if self.pv_method in {"hourly", "method_6"} and not plane.isna().all() and ppk > 0:
                # eq. (68)
                pv_generation = plane.fillna(0.0).clip(lower=0.0) * ppk * self.pv_perf_factor
            elif self.pv_method == "method_4":
                pv_generation = profile * self._pv_method_4_annual(ppk)
            elif self.pv_method == "method_5":
                pv_generation = self._pv_method_5_monthly(out, profile, ppk)
            else:
                # EN 15316-4-3:2017, Formula (69): when the peak power is
                # derived from module area, use that derived Ppk in the
                # annual method as well (not only in Method 6).
                annual_ppk = ppk
                expected = (
                    annual_ppk
                    * self.pv_yield_kWh_kWp_a
                    * self.pv_tilt_orientation_factor
                    * (1.0 - self.pv_shading_loss_fraction)
                    * (1.0 - self.pv_orientation_loss_fraction)
                    * (1.0 - self.pv_degradation_loss_fraction)
                )
                if out["GHI_W_m2"].notna().any() and out["GHI_W_m2"].clip(lower=0.0).sum() > 0:
                    raw = out["GHI_W_m2"].fillna(0.0).clip(lower=0.0) / 1000.0
                    raw = (
                        raw
                        * out["hours"]
                        * annual_ppk
                        * self.pv_performance_ratio
                    )
                    scale = expected / max(float(raw.sum()), _KWH_EPS)
                    pv_generation = raw * scale
                else:
                    pv_generation = profile * expected
        out.loc[:, "E_PV_gen_kWh"] = pv_generation
        # Table 25 outputs (Methods 4-6): only the electricity is produced; no
        # thermal output, losses, recoverable losses, auxiliary or energy input.
        out.loc[:, "E_el_pv_out_kWh"] = pv_generation
        for col in ("Q_pv_gen_out_kWh", "Q_pv_gen_ls_kWh", "Q_pv_gen_ls_rbl_kWh",
                    "W_pv_gen_aux_kWh", "E_pv_gen_in_kWh"):
            out.loc[:, col] = 0.0
        if self.battery_capacity_kWh > _KWH_EPS or self.pv_self_consumption_mode == "dispatch":
            dispatch = self._dispatch_pv_with_battery(out)
            for col in dispatch.columns:
                out.loc[:, col] = dispatch[col]
        else:
            self_consumption_limit = pv_generation * self.pv_self_consumption_fraction
            out.loc[:, "E_PV_direct_self_consumed_kWh"] = np.minimum(
                out["E_site_el_load_kWh"], self_consumption_limit
            )
            out.loc[:, "E_battery_charge_kWh"] = 0.0
            out.loc[:, "E_battery_discharge_to_load_kWh"] = 0.0
            out.loc[:, "E_battery_losses_kWh"] = 0.0
            out.loc[:, "E_battery_state_of_charge_kWh"] = 0.0
            out.loc[:, "E_PV_self_consumed_kWh"] = out["E_PV_direct_self_consumed_kWh"]
            out.loc[:, "E_PV_export_kWh"] = (
                pv_generation - out["E_PV_self_consumed_kWh"]
            ).clip(lower=0.0)
            out.loc[:, "E_grid_after_PV_kWh"] = (
                out["E_site_el_load_kWh"] - out["E_PV_self_consumed_kWh"]
            ).clip(lower=0.0)
        return out

    def _simulate_solar_thermal_hourly(self, out: pd.DataFrame) -> pd.Series:
        """EN 15316-4-3:2017 Method 3, formulas (49), (52), (55)-(60).

        This is the collector-loop boundary; storage acceptance remains the
        responsibility of the storage module. Inlet/mean collector temperature
        can be supplied in the configuration for a standalone calculation.
        """
        # Eq. (52)-(55) need I_sol,h on the collector plane (tilt, azimuth and
        # shading already applied, EN ISO 52010-1 / module M1-9).
        if not out["I_sol_collector_W_m2"].notna().any():
            plane = self._collector_plane_irradiance(out)
            if plane is not None:
                out.loc[:, "I_sol_collector_W_m2"] = plane
        if out["I_sol_collector_W_m2"].notna().any():
            irradiance = out["I_sol_collector_W_m2"].fillna(0.0).clip(lower=0.0)
        elif self.solar_require_plane_irradiance:
            raise ValueError(
                "EN 15316-4-3 Method 3 needs the irradiance on the collector plane "
                "(I_sol_collector_plane_W_m2, or DNI/DHI with latitude_deg, longitude_deg, "
                "timezone_h, tilt_deg, azimuth_deg); require_plane_irradiance is set."
            )
        else:
            warnings.warn(
                "EN 15316-4-3 Method 3: no collector-plane irradiance column "
                "(I_sol_collector_plane_W_m2); using horizontal irradiance GHI, "
                "so collector tilt and azimuth are not accounted for.",
                UserWarning,
                stacklevel=2,
            )
            irradiance = out["GHI_W_m2"].fillna(0.0).clip(lower=0.0)
        ambient = out["T_ext_C"].fillna(0.0)
        if self.solar_ambient_location == "HS":
            ambient = pd.Series(self.solar_indoor_temperature_C, index=out.index)
        elif self.solar_ambient_location == "NHS":
            ambient = (ambient + self.solar_indoor_temperature_C) / 2.0
        area = self.solar_collector_area_m2
        if self.storage_response is not None:
            return self._simulate_solar_thermal_iterative(out, irradiance, ambient)
        if not self.solar_mean_temperature_given:
            warnings.warn(
                "EN 15316-4-3 Method 3 without a storage model: the collector runs at a fixed "
                f"mean temperature of {self.solar_collector_mean_temperature_C:g} degC (default) and "
                "there is no collector/storage iteration (N_sol_it = 0). Pass storage_response=... "
                "for the complete method, or set solar_thermal.mean_temperature_C explicitly.",
                UserWarning,
                stacklevel=2,
            )
        delta = self.solar_collector_mean_temperature_C - ambient
        reduced = delta / irradiance.replace(0.0, np.nan)
        # Formula (52): eta = eta_0 * Khem(50) - a1 * T* - a2 * T*^2 * I
        eta = (self.solar_collector_eta_0 * self.solar_collector_khem50
               - self.solar_collector_a1 * reduced
               - self.solar_collector_a2 * reduced.pow(2) * irradiance).fillna(0.0).clip(lower=0.0)
        q_col = eta * irradiance * area * out["hours"] / 1000.0
        # Eq. (56); by default heat losses cannot become gains when the collector
        # temperature is below ambient (see loop_loss_clip).
        loss_delta = delta.clip(lower=0.0) if self.solar_loop_loss_clip else delta
        q_loop_loss = self.solar_loop_heat_loss_W_K * loss_delta * out["hours"] / 1000.0
        q_loop = (q_col - q_loop_loss).clip(lower=0.0)
        # EN 15316-4-3:2017 Method 3: below the minimum pump-energy
        # threshold (Psol;pmp * tci * 3), collector-loop output is zero.
        pump_threshold = self.solar_aux_power_W * out["hours"] * 3.0 / 1000.0
        q_loop = q_loop.where(q_loop >= pump_threshold, 0.0)
        q_ls = q_loop_loss.clip(lower=0.0) if self.solar_loop_loss_clip else q_loop_loss
        out.loc[:, "Q_solar_thermal_collector_kWh"] = q_col  # == Q_sol_col_out_kWh
        out.loc[:, "Q_solar_thermal_loop_losses_kWh"] = q_ls
        self._write_method_3_outputs(
            out, irradiance, q_col, q_ls, q_out=q_loop, q_req=q_loop, q_acc=q_loop, n_it=0
        )
        return q_loop

    def _write_method_3_outputs(
        self, out, irradiance, q_col, q_ls, q_out, q_req, q_acc, n_it
    ) -> None:
        """Hourly outputs of EN 15316-4-3:2017 Table 14 (Method 3).

        ``Q_sol_*`` / ``E_sol_aux_kWh`` / ``N_sol_it`` / ``SOL_FINAL`` carry the
        standard's names. Without a storage model there is no iteration
        (``N_sol_it`` = 0): the request is the loop output and everything is accepted.
        """
        hours = out["hours"]
        out.loc[:, "Q_sol_gen_kWh"] = (
            self.solar_collector_eta_0 * irradiance * self.solar_collector_area_m2 * hours / 1000.0
        )  # (54): heat generated in the absorber
        out.loc[:, "Q_sol_col_out_kWh"] = q_col  # (55)
        out.loc[:, "Q_sol_loop_out_kWh"] = q_out  # (57), after the pump threshold
        out.loc[:, "Q_sol_loop_out_req_kWh"] = q_req  # asked of the storage
        out.loc[:, "Q_sto_sol_in_kWh"] = q_acc  # accepted by the storage
        out.loc[:, "Q_sol_loop_ls_kWh"] = q_ls  # (56)
        # (61): losses are recoverable only while there is a space-heating demand
        heating = out["Q_H_before_solar_kWh"].fillna(0.0) > 0.0
        out.loc[:, "Q_sol_loop_ls_rbl_kWh"] = q_ls.where(heating, 0.0)
        # (59)-(60)
        ctr = self.solar_controller_power_W * hours / 1000.0
        pump = self.solar_aux_power_W * hours / 1000.0
        pump_on = (q_out <= 0.0) if self.solar_aux_literal_eq60 else (q_out > 0.0)
        out.loc[:, "E_sol_aux_kWh"] = ctr + pump * pump_on.astype(float)
        out.loc[:, "E_solar_thermal_aux_kWh"] = out["E_sol_aux_kWh"]
        out.loc[:, "N_sol_it"] = int(n_it)
        out.loc[:, "SOL_FINAL"] = True

    def _simulate_solar_thermal_iterative(
        self, out: pd.DataFrame, irradiance: pd.Series, ambient: pd.Series
    ) -> pd.Series:
        """EN 15316-4-3:2017 Method 3 with the collector/storage iteration.

        Per hour: first estimate of the mean collector temperature (51), then
        ``iterations`` passes of efficiency (52)-(53), collector output (55),
        loop losses (56), loop output (57) and a storage call, updating the mean
        temperature with (58). The last pass commits the storage state and is
        limited to what the previous pass could place in the storage.
        """
        if not self.solar_mass_flow_kg_s or self.solar_mass_flow_kg_s <= 0.0:
            raise ValueError(
                "Method 3 with a storage model needs solar_thermal.mass_flow_kg_s "
                "(or mass_flow_kg_s_m2)."
            )
        c_w = 4186.0  # J/(kg K), Table 18
        m_dot = self.solar_mass_flow_kg_s
        area = self.solar_collector_area_m2
        n = len(out)
        cols = {k: np.zeros(n) for k in ("q_col", "q_ls", "q_req", "q_out", "q_acc", "t_avg", "t_in", "pump")}
        theta_in_prev = self.solar_initial_inlet_temperature_C
        for i in range(n):
            I = float(irradiance.iloc[i])
            t_ci = float(out["hours"].iloc[i])
            t_e = float(ambient.iloc[i])
            theta_avg = theta_in_prev + 0.4 * I * area / (m_dot * c_w * 2.0)  # (51)
            theta_in = theta_in_prev
            q_col = q_ls = q_out = q_acc = 0.0
            for it in range(self.solar_iterations):
                if I > 0.0:
                    t_star = (theta_avg - t_e) / I  # (53)
                    eta = max(
                        self.solar_collector_eta_0 * self.solar_collector_khem50
                        - self.solar_collector_a1 * t_star
                        - self.solar_collector_a2 * t_star**2 * I,
                        0.0,
                    )  # (52)
                else:
                    eta = 0.0
                q_col = eta * I * area * t_ci / 1000.0  # (55)
                d = theta_avg - t_e
                q_ls = self.solar_loop_heat_loss_W_K * (max(d, 0.0) if self.solar_loop_loss_clip else d) * t_ci / 1000.0  # (56)
                q_out = max(q_col - q_ls, 0.0)  # (57)
                if q_out < self.solar_aux_power_W * t_ci * 3.0 / 1000.0:
                    q_out = 0.0
                last = it == self.solar_iterations - 1
                q_req = min(q_out, q_acc) if last and it > 0 else q_out
                theta_in, q_acc = self.storage_response(i, q_req, last)
                theta_in, q_acc = float(theta_in), max(float(q_acc), 0.0)
                theta_avg = (theta_in_prev + theta_in) / 2.0 + (
                    q_out * 1000.0 / max(t_ci, 1e-12) / (m_dot * c_w) / 2.0
                )  # (58)
            # Q_sol;loop;out;req is what the final storage call was asked for: Q_sol;loop;out,
            # capped at what the previous pass could place in the storage (6.1.3.4.2)
            cols["q_col"][i], cols["q_ls"][i], cols["q_req"][i] = q_col, q_ls, q_req
            cols["q_out"][i] = q_out
            cols["q_acc"][i], cols["t_avg"][i], cols["t_in"][i] = q_acc, theta_avg, theta_in
            cols["pump"][i] = 1.0 if q_out > 0.0 else 0.0
            theta_in_prev = theta_in
        idx = out.index
        out.loc[:, "Q_solar_thermal_collector_kWh"] = pd.Series(cols["q_col"], index=idx)
        out.loc[:, "Q_solar_thermal_loop_losses_kWh"] = pd.Series(cols["q_ls"], index=idx)
        out.loc[:, "Q_solar_thermal_loop_out_req_kWh"] = pd.Series(cols["q_req"], index=idx)
        out.loc[:, "T_solar_collector_avg_C"] = pd.Series(cols["t_avg"], index=idx)
        out.loc[:, "T_solar_loop_in_C"] = pd.Series(cols["t_in"], index=idx)
        self._write_method_3_outputs(
            out, irradiance, pd.Series(cols["q_col"], index=idx), pd.Series(cols["q_ls"], index=idx),
            q_out=pd.Series(cols["q_out"], index=idx), q_req=pd.Series(cols["q_req"], index=idx),
            q_acc=pd.Series(cols["q_acc"], index=idx), n_it=self.solar_iterations,
        )
        return pd.Series(cols["q_acc"], index=idx)

    def _pv_method_4_annual(self, ppk: float) -> float:
        """Eq. (62)-(63): E_el;pv;out = E_sol;hor * f_tlt * Ppk * f_perf / I_ref (I_ref = 1 kW/m2)."""
        if self.pv_annual_horizontal_kWh_m2 is None:
            raise ValueError("pv.method 'method_4' needs pv.annual_irradiation_horizontal_kWh_m2.")
        e_sol = float(self.pv_annual_horizontal_kWh_m2) * self.pv_tilt_orientation_factor
        return e_sol * ppk * self.pv_perf_factor

    def _pv_method_5_monthly(self, out: pd.DataFrame, profile: pd.Series, ppk: float) -> pd.Series:
        """Eq. (65)-(66), monthly: E_el;pv;out,m = E_sol;hor,m * f_tlt,m * Ppk * f_perf / I_ref,
        spread over the hours of each month with the horizontal-irradiance profile."""
        monthly = self.pv_monthly_horizontal_kWh_m2
        if monthly is None or len(monthly) != 12:
            raise ValueError("pv.method 'method_5' needs 12 values in pv.monthly_irradiation_horizontal_kWh_m2.")
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("pv.method 'method_5' needs a DatetimeIndex.")
        f_tlt = self.pv_monthly_tilt_factor
        if f_tlt is None:
            f_tlt = [self.pv_tilt_orientation_factor] * 12
        elif np.isscalar(f_tlt):
            f_tlt = [float(f_tlt)] * 12
        if len(f_tlt) != 12:
            raise ValueError("pv.tilt_orientation_factor_monthly needs 12 values (or one).")
        result = pd.Series(0.0, index=out.index)
        for m in range(1, 13):
            mask = out.index.month == m
            weight = float(profile[mask].sum())
            if not mask.any() or weight <= _KWH_EPS:
                continue
            e_month = float(monthly[m - 1]) * float(f_tlt[m - 1]) * ppk * self.pv_perf_factor
            result[mask] = profile[mask] / weight * e_month
        return result

    def _collector_plane_irradiance(
        self, out: pd.DataFrame, site: dict | None = None, ground_reflectivity: float | None = None
    ) -> pd.Series | None:
        """Total irradiance on the collector plane from DNI/DHI, EN ISO 52010-1.

        Returns None when the site/orientation inputs or the DNI/DHI columns are
        missing. Needs a DatetimeIndex with an hourly step (the ISO 52010-1 hour
        angle is built from the hour of day).
        """
        site = self.solar_site if site is None else site
        if ground_reflectivity is None:
            ground_reflectivity = self.solar_ground_reflectivity
        if any(v is None for v in site.values()):
            return None
        if not (out["DNI_W_m2"].notna().any() and out["DHI_W_m2"].notna().any()):
            return None
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("Collector-plane irradiance needs a DatetimeIndex.")
        if not np.allclose(out["hours"].to_numpy(dtype=float), 1.0):
            raise ValueError("Collector-plane irradiance (EN ISO 52010-1) needs hourly steps.")
        from .utils import ISO52010  # local import: utils is heavy

        calendar = pd.DataFrame(
            {"day of year": out.index.dayofyear, "hour of day": out.index.hour + 1},
            index=out.index,
        )
        gamma = (180.0 - float(site["azimuth_deg"]) + 180.0) % 360.0 - 180.0  # ISO 52010: S=0, E=+90
        n_days = 366 if len(out) > 8760 else 365
        result = ISO52010.Solar_irradiance_calculation(
            latitude_deg=float(site["latitude_deg"]),
            longitude_deg=float(site["longitude_deg"]),
            timezone=float(site["timezone_h"]),
            beta_ic_deg=float(site["tilt_deg"]),
            gamma_ic_deg=gamma,
            DHI=out["DHI_W_m2"].fillna(0.0).clip(lower=0.0),
            DNI=out["DNI_W_m2"].fillna(0.0).clip(lower=0.0),
            ground_solar_reflectivity=ground_reflectivity,
            calendar=calendar,
            n_timesteps=len(out),
            n_days=n_days,
            building_object=None,
        )[0]
        return result.solar_irradiance["I_sol_tot"].clip(lower=0.0).astype(float)

    def _dispatch_pv_with_battery(self, out: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=out.index)
        charge_eff = self.battery_roundtrip_efficiency**0.5
        discharge_eff = self.battery_roundtrip_efficiency**0.5
        capacity = self.battery_capacity_kWh
        min_soc = capacity * self.battery_min_soc_fraction
        soc = capacity * self.battery_initial_soc_fraction

        direct_values: list[float] = []
        charge_values: list[float] = []
        discharge_to_load_values: list[float] = []
        losses_values: list[float] = []
        export_values: list[float] = []
        grid_values: list[float] = []
        soc_values: list[float] = []

        for _, row in out.iterrows():
            pv = max(float(row["E_PV_gen_kWh"]), 0.0)
            load = max(float(row["E_site_el_load_kWh"]), 0.0)
            hours = max(float(row["hours"]), 0.0)
            direct = min(load, pv)
            surplus = max(pv - direct, 0.0)
            remaining_load = max(load - direct, 0.0)

            charge_from_pv = 0.0
            charge_loss = 0.0
            if capacity > _KWH_EPS and surplus > _KWH_EPS:
                charge_power_limit = (
                    self.battery_max_charge_power_kW * hours
                    if self.battery_max_charge_power_kW > _KWH_EPS
                    else surplus
                )
                room_from_pv = max((capacity - soc) / max(charge_eff, _KWH_EPS), 0.0)
                charge_from_pv = min(surplus, charge_power_limit, room_from_pv)
                stored = charge_from_pv * charge_eff
                charge_loss = charge_from_pv - stored
                soc += stored
                surplus -= charge_from_pv

            discharge_to_load = 0.0
            discharge_loss = 0.0
            if capacity > _KWH_EPS and remaining_load > _KWH_EPS:
                discharge_power_limit = (
                    self.battery_max_discharge_power_kW * hours
                    if self.battery_max_discharge_power_kW > _KWH_EPS
                    else remaining_load
                )
                available_to_load = max(soc - min_soc, 0.0) * discharge_eff
                discharge_to_load = min(remaining_load, discharge_power_limit, available_to_load)
                soc_decrease = discharge_to_load / max(discharge_eff, _KWH_EPS)
                discharge_loss = soc_decrease - discharge_to_load
                soc -= soc_decrease
                remaining_load -= discharge_to_load

            direct_values.append(direct)
            charge_values.append(charge_from_pv)
            discharge_to_load_values.append(discharge_to_load)
            losses_values.append(charge_loss + discharge_loss)
            export_values.append(max(surplus, 0.0))
            grid_values.append(max(remaining_load, 0.0))
            soc_values.append(soc)

        result.loc[:, "E_PV_direct_self_consumed_kWh"] = direct_values
        result.loc[:, "E_battery_charge_kWh"] = charge_values
        result.loc[:, "E_battery_discharge_to_load_kWh"] = discharge_to_load_values
        result.loc[:, "E_battery_losses_kWh"] = losses_values
        result.loc[:, "E_battery_state_of_charge_kWh"] = soc_values
        result.loc[:, "E_PV_self_consumed_kWh"] = (
            result["E_PV_direct_self_consumed_kWh"]
            + result["E_battery_discharge_to_load_kWh"]
        )
        result.loc[:, "E_PV_export_kWh"] = export_values
        result.loc[:, "E_grid_after_PV_kWh"] = grid_values
        return result

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
            "E_PV_direct_self_consumed_kWh": s("E_PV_direct_self_consumed_kWh"),
            "E_PV_self_consumed_kWh": s("E_PV_self_consumed_kWh"),
            "E_PV_export_kWh": s("E_PV_export_kWh"),
            "E_grid_after_PV_kWh": s("E_grid_after_PV_kWh"),
            "E_battery_charge_kWh": s("E_battery_charge_kWh"),
            "E_battery_discharge_to_load_kWh": s("E_battery_discharge_to_load_kWh"),
            "E_battery_losses_kWh": s("E_battery_losses_kWh"),
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



class MixedSolarTank:
    """Fully mixed single-volume tank usable as ``storage_response`` for Method 3.

    A simple stand-in for the EN 15316-5 storage module (which does not yet expose
    the bottom-volume temperature): one node at uniform temperature, heated by the
    collector loop up to ``max_temperature_C``, drawn down by ``draw_kWh`` per step
    and losing ``loss_W_K * (T - ambient)``.
    """

    def __init__(
        self,
        volume_l: float,
        initial_temperature_C: float = 20.0,
        max_temperature_C: float = 90.0,
        loss_W_K: float = 0.0,
        ambient_C: float = 20.0,
        step_hours: float = 1.0,
        draw_kWh: Any = None,
    ):
        self.capacity_kWh_K = 1.15 * volume_l / 1000.0  # rho*cp, kWh/(m3 K) * m3
        if self.capacity_kWh_K <= 0.0:
            raise ValueError("volume_l must be > 0.")
        self.temperature_C = float(initial_temperature_C)
        self.max_temperature_C = float(max_temperature_C)
        self.loss_W_K = float(loss_W_K)
        self.ambient_C = float(ambient_C)
        self.step_hours = float(step_hours)
        self.draw_kWh = None if draw_kWh is None else np.asarray(draw_kWh, dtype=float)
        self.history: list[float] = []

    def __call__(self, i: int, q_req_kWh: float, commit: bool) -> tuple[float, float]:
        headroom = self.capacity_kWh_K * max(self.max_temperature_C - self.temperature_C, 0.0)
        q_acc = min(max(q_req_kWh, 0.0), headroom)
        if commit:
            draw = 0.0 if self.draw_kWh is None else float(self.draw_kWh[i])
            loss = self.loss_W_K * (self.temperature_C - self.ambient_C) * self.step_hours / 1000.0
            self.temperature_C += (q_acc - draw - loss) / self.capacity_kWh_K
            self.temperature_C = max(self.temperature_C, 5.0)
            self.history.append(self.temperature_C)
        return self.temperature_C, q_acc
