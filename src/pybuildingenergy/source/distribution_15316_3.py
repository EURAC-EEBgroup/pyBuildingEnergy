"""Water-based distribution calculation according to EN 15316-3:2017.

The module implements the hourly distribution calculation for space heating,
water-based space cooling and domestic hot water distribution systems, modules
M3-6, M4-6 and M8-6. It is designed to sit between emission/tapping demand and
generation: users provide the useful thermal output required downstream of the
distribution network and the calculator returns the thermal input that has to
be supplied by the generator, distribution thermal losses and pump auxiliary
electricity.

The implementation follows the EN 15316-3:2017 hourly structure:

* mean water temperature in the distribution system;
* pipe thermal losses from linear thermal transmittance, pipe length,
  surrounding temperature and operation time;
* recoverable thermal losses in conditioned spaces;
* hydraulic design pump power, pump expenditure factor and auxiliary energy;
* recoverable and fluid-recovered auxiliary energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12
_WATER_HEAT_CAPACITY_DENSITY_KWH_M3K = 1.15
_WATER_DENSITY_KG_M3 = 990.0
_WATER_SPECIFIC_HEAT_KWH_KGK = 1.163e-3


@dataclass
class DistributionSimulationResult:
    """Container returned by :class:`DistributionSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class DistributionSystemCalculator:
    """EN 15316-3:2017 water-based distribution calculator."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> DistributionSimulationResult:
        """Run the hourly distribution calculation."""

        prepared = self._prepare_timeseries(data)
        service_frames = [
            self._simulate_service(prepared, service) for service in ("H", "C", "W")
        ]
        # `prepared` passes caller columns through untouched (see
        # _prepare_timeseries), which lets e.g. a "beta_H_dis" override be
        # read by _simulate_service above. That same name is also an output
        # column of the H service frame, so it must be dropped from the base
        # frame here or the two would collide into a duplicate column.
        produced = {col for frame in service_frames for col in frame.columns}
        base = prepared.drop(columns=[c for c in prepared.columns if c in produced])
        results = pd.concat([base, *service_frames], axis=1)
        summary = self._summarize(results)
        return DistributionSimulationResult(
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

        self.services = {
            "H": self._service_options("heating", cfg.get("heating", {})),
            "C": self._service_options("cooling", cfg.get("cooling", {})),
            "W": self._service_options("dhw", cfg.get("dhw", cfg.get("domestic_hot_water", {}))),
        }

    def _service_options(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(data or {})
        sections = [_section_options(item) for item in cfg.get("pipe_sections", [])]

        # Derive max_length_m from pipe_sections when not provided explicitly.
        # EN 15316-3 §6 uses the total hydraulic path length (straight + equivalent)
        # for the head-loss calculation; falling back to 0 would zero out the
        # pump auxiliary energy, which is non-physical.
        _sections_length = sum(
            float(s.get("length_m", 0.0)) + float(s.get("equivalent_length_m", 0.0))
            for s in sections
        )
        _max_length_default = max(_sections_length, 0.0)

        return {
            "pipe_sections": sections,
            "operation_mode": str(cfg.get("operation_mode", "demand")).lower(),
            "nominal_power_kW": max(float(cfg.get("nominal_power_kW", 0.0)), 0.0),
            "design_flow_m3_h": max(float(cfg.get("design_flow_m3_h", 0.0)), 0.0),
            "design_deltaT_K": max(float(cfg.get("design_deltaT_K", 10.0)), _KWH_EPS),
            "supply_temperature_C": float(cfg.get("supply_temperature_C", 45.0)),
            "return_temperature_C": float(cfg.get("return_temperature_C", 35.0)),
            "dhw_temperature_C": float(cfg.get("dhw_temperature_C", cfg.get("hot_water_temperature_C", 55.0))),
            "dhw_return_deltaT_K": max(float(cfg.get("dhw_return_deltaT_K", 5.0)), 0.0),
            "max_length_m": max(float(cfg.get("max_length_m", _max_length_default)), 0.0),
            "pressure_loss_per_m_kPa": max(float(cfg.get("pressure_loss_per_m_kPa", 0.10)), 0.0),
            "additional_pressure_kPa": max(float(cfg.get("additional_pressure_kPa", 0.0)), 0.0),
            "resistance_ratio": max(float(cfg.get("resistance_ratio", cfg.get("f_comp", 0.30))), 0.0),
            "design_delta_pressure_kPa": max(
                float(cfg.get("design_delta_pressure_kPa", 0.0)), 0.0
            ),
            "pump_control_code": int(cfg.get("pump_control_code", cfg.get("control_code", 4))),
            "pump_selection_factor": max(float(cfg.get("pump_selection_factor", 1.0)), 0.0),
            "pump_label_power_kW": max(float(cfg.get("pump_label_power_kW", 0.0)), 0.0),
            "eei": _fraction(cfg.get("eei", 0.23), f"{name}.eei"),
            "hydraulic_correction_factor": max(
                float(cfg.get("hydraulic_correction_factor", cfg.get("f_corr", 1.0))),
                0.0,
            ),
            "part_load_mode": str(cfg.get("part_load_mode", "load")).lower(),
            "recoverable_aux_fraction": _fraction(
                cfg.get("recoverable_aux_fraction", cfg.get("f_aux_rbl", 0.25)),
                f"{name}.recoverable_aux_fraction",
            ),
            # EN 15316-3 Eq.(8)-(9): open-circuited DHW stubs, lumped per zone.
            "stub_volume_m3": max(float(cfg.get("stub_volume_m3", 0.0)), 0.0),
            "stub_ambient_temperature_C": cfg.get("stub_ambient_temperature_C", 20.0),
            "taps_per_hour": max(float(cfg.get("taps_per_hour", 1.0)), 0.0),
            "stub_recoverable": bool(cfg.get("stub_recoverable", False)),
            # EN 15316-3 Eq.(31): ribbon heater auxiliary energy for DHW.
            "ribbon_heater": bool(cfg.get("ribbon_heater", False)),
        }

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        # Start from a copy of the caller's own data (not a blank frame) so that
        # columns this module does not itself recognize - a named ambient-
        # temperature series (Table 10: ambient temperatures are "Varying:
        # YES"), a "beta_{H,C,W}_dis" override, or a "pump_mode_{H,C}" column
        # for Eq.(29)/(30) - pass through untouched and are still resolvable
        # by name downstream.
        out = df.copy()
        out["hours"] = self._time_step_hours(df)
        out["T_ext_C"] = _series_from_aliases(
            df, ["T_ext", "theta_ext", "outdoor_temperature_C", "T_external_C"], default=np.nan
        ).astype(float)
        out["T_op_C"] = _series_from_aliases(
            df, ["T_op", "operative_temperature_C", "T_zone_C"], default=20.0
        ).astype(float)

        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        out["Q_H_dis_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_H_kWh", "QH_kWh", "Q_H_em_in_kWh", "space_heating_kWh"],
            raw_aliases=["Q_H", "Q_h", "Heating_needs"],
            fallback=np.maximum(q_hc, 0.0),
        )
        out["Q_C_dis_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_C_kWh", "QC_kWh", "Q_C_em_in_kWh", "space_cooling_kWh"],
            raw_aliases=["Q_C", "Cooling_needs"],
            fallback=np.maximum(-q_hc, 0.0),
        )
        out["Q_W_dis_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_W_kWh", "QW_kWh", "Q_DHW_kWh", "DHW_kWh", "dhw_kWh"],
            raw_aliases=["Q_W", "Q_DHW"],
            fallback=0.0,
        )

        for col in ["Q_H_dis_out_kWh", "Q_C_dis_out_kWh", "Q_W_dis_out_kWh"]:
            out.loc[:, col] = out[col].fillna(0.0).clip(lower=0.0)

        temperature_columns = [
            "theta_H_dis_supply_C",
            "theta_H_dis_return_C",
            "theta_C_dis_supply_C",
            "theta_C_dis_return_C",
            "theta_W_dis_hot_C",
            "T_H_supply_C",
            "T_H_return_C",
            "T_C_supply_C",
            "T_C_return_C",
            "T_W_supply_C",
            "dhw_temperature_C",
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
        opts = dict(self.services[service])
        q_out = prepared[f"Q_{service}_dis_out_kWh"].astype(float)
        hours = prepared["hours"].astype(float)
        op_hours = self._operation_hours(q_out, hours, opts)
        mean_temp = self._mean_water_temperature(prepared, service, opts)

        thermal = self._thermal_losses(prepared, mean_temp, op_hours, hours, service, opts)

        # Table 10 lists beta_H,dis / beta_C,dis as external "Varying: YES"
        # operating-conditions inputs (sourced from the emission module,
        # M3-5), not something this module derives on its own. When the
        # caller supplies it directly, use it as-is; otherwise fall back to
        # the Q/nominal-power heuristic.
        beta_col = f"beta_{service}_dis"
        if beta_col in prepared.columns:
            beta = prepared[beta_col].astype(float).clip(0.0, 1.0)
        else:
            beta = self._part_load(q_out, hours, op_hours, opts)

        mode_col = f"pump_mode_{service}"
        opts["pump_mode"] = prepared[mode_col] if mode_col in prepared.columns else None

        pump = self._pump_auxiliary(q_out, hours, op_hours, beta, opts, service)

        aux_rbl = pump["W_aux_kWh"] * opts["recoverable_aux_fraction"]
        aux_rvd = pump["W_aux_kWh"] * (1.0 - opts["recoverable_aux_fraction"])

        if service == "C":
            thermal_rbl = -thermal["Q_loss_recoverable_kWh"]
            aux_rbl = -aux_rbl
            aux_rvd = -aux_rvd
        else:
            thermal_rbl = thermal["Q_loss_recoverable_kWh"]

        q_in = (q_out + thermal["Q_loss_kWh"] - aux_rvd).clip(lower=0.0)

        out = pd.DataFrame(index=prepared.index)
        out[f"theta_{service}_dis_mean_C"] = mean_temp
        out[f"Q_{service}_dis_ls_kWh"] = thermal["Q_loss_kWh"]
        out[f"Q_{service}_dis_th_rbl_kWh"] = thermal_rbl
        out[f"W_{service}_dis_hydr_kWh"] = pump["W_hydr_kWh"]
        out[f"W_{service}_dis_aux_kWh"] = pump["W_aux_kWh"]
        out[f"Q_{service}_dis_aux_rbl_kWh"] = aux_rbl
        out[f"Q_{service}_dis_aux_rvd_kWh"] = aux_rvd
        out[f"Q_{service}_dis_rbl_kWh"] = thermal_rbl + aux_rbl
        out[f"Q_{service}_dis_in_kWh"] = q_in
        out[f"beta_{service}_dis"] = beta
        out[f"epsilon_{service}_dis"] = pump["epsilon"]
        out[f"P_{service}_hydr_des_kW"] = pump["P_hydr_des_kW"]
        out[f"delta_p_{service}_des_kPa"] = pump["delta_p_des_kPa"]
        out[f"V_{service}_des_m3_h"] = pump["design_flow_m3_h"]
        out[f"t_{service}_dis_op_h"] = op_hours
        if service == "W":
            out["Q_W_dis_stub_kWh"] = thermal["Q_stub_kWh"]
            if opts["ribbon_heater"]:
                out["W_W_dis_rib_kWh"] = thermal["Q_rib_kWh"]
        return out

    def _operation_hours(
        self, q_out: pd.Series, hours: pd.Series, opts: dict[str, Any]
    ) -> pd.Series:
        mode = opts["operation_mode"]
        if mode == "continuous":
            return hours
        if mode.startswith("fixed_fraction:"):
            fraction = _fraction(mode.split(":", 1)[1], "operation fixed fraction")
            return hours * fraction
        return hours.where(q_out > _KWH_EPS, 0.0)

    def _mean_water_temperature(
        self, prepared: pd.DataFrame, service: str, opts: dict[str, Any]
    ) -> pd.Series:
        if service == "W":
            hot = _series_from_aliases(
                prepared,
                ["theta_W_dis_hot_C", "T_W_supply_C", "dhw_temperature_C"],
                default=opts["dhw_temperature_C"],
            ).astype(float)
            return hot - 0.5 * float(opts["dhw_return_deltaT_K"])

        supply_default = opts["supply_temperature_C"]
        return_default = opts["return_temperature_C"]
        supply = _series_from_aliases(
            prepared,
            [f"theta_{service}_dis_supply_C", f"T_{service}_supply_C"],
            default=supply_default,
        ).astype(float)
        ret = _series_from_aliases(
            prepared,
            [f"theta_{service}_dis_return_C", f"T_{service}_return_C"],
            default=return_default,
        ).astype(float)
        return (supply + ret) / 2.0

    def _thermal_losses(
        self,
        prepared: pd.DataFrame,
        mean_temp: pd.Series,
        op_hours: pd.Series,
        hours: pd.Series,
        service: str,
        opts: dict[str, Any],
    ) -> dict[str, pd.Series]:
        total = pd.Series(0.0, index=mean_temp.index)
        recoverable = pd.Series(0.0, index=mean_temp.index)
        rib_flagged = pd.Series(0.0, index=mean_temp.index)
        rib_all = pd.Series(0.0, index=mean_temp.index)
        non_op_hours = (hours - op_hours).clip(lower=0.0)
        any_rib_target = False

        for section in opts["pipe_sections"]:
            ambient = self._resolve_series(prepared, section["ambient_temperature_C"])
            length = float(section["length_m"]) + float(section["equivalent_length_m"])
            psi = float(section["linear_thermal_transmittance_W_mK"])
            circulating = bool(section["circulating"])

            if service == "W" and not circulating:
                # Eq.(15): simplified hourly method for a distribution pipe
                # without circulation (a branch/stub run) - applies for the
                # WHOLE time step, since such a pipe never carries hot water
                # in steady state the way a circulation loop does.
                theta_mean = ambient if psi <= _KWH_EPS else 25.0 * (psi ** -0.2)
                delta = (theta_mean - ambient).clip(lower=0.0)
                op_loss = psi * length * delta * hours / 1000.0
                section_total = op_loss
            else:
                if service == "C":
                    delta_op = (ambient - mean_temp).clip(lower=0.0)
                else:
                    delta_op = (mean_temp - ambient).clip(lower=0.0)
                op_loss = psi * length * delta_op * op_hours / 1000.0
                section_total = op_loss

                if service == "W":
                    # Eq.(10): circulation-loop loss while NOT tapping. The
                    # standard's own Eq.(15) stand-in for the average/decayed
                    # temperature is used in place of the full Eq.(11)-(14)
                    # per-tap decay model (both are sanctioned for hourly
                    # time steps; Eq.(11)-(14) additionally needs pipe mass/
                    # specific heat and inter-tap timing this module does not
                    # collect).
                    theta_standby = ambient if psi <= _KWH_EPS else 25.0 * (psi ** -0.2)
                    delta_standby = (theta_standby - ambient).clip(lower=0.0)
                    section_total = section_total + psi * length * delta_standby * non_op_hours / 1000.0

            total = total + section_total
            if bool(section["recoverable"]):
                recoverable = recoverable + section_total

            if service == "W":
                rib_all = rib_all + op_loss
                if bool(section["ribbon_target"]):
                    any_rib_target = True
                    rib_flagged = rib_flagged + op_loss

        stub_loss = pd.Series(0.0, index=mean_temp.index)
        if service == "W":
            stub_loss = self._dhw_stub_loss(prepared, op_hours, opts)
            total = total + stub_loss
            if opts["stub_recoverable"]:
                recoverable = recoverable + stub_loss

        return {
            "Q_loss_kWh": total,
            "Q_loss_recoverable_kWh": recoverable,
            "Q_stub_kWh": stub_loss,
            # Eq.(31): ribbon heater energy = Eq.(7) taking into account only
            # the hot-water pipe length. Sections flagged "ribbon_target" are
            # used when given; otherwise this falls back to the whole
            # service's operating-time loss (an approximation when the
            # supply-only pipe length is not singled out).
            "Q_rib_kWh": rib_flagged if any_rib_target else rib_all,
        }

    def _resolve_series(self, prepared: pd.DataFrame, value: Any) -> pd.Series:
        if isinstance(value, str):
            if value not in prepared.columns:
                raise KeyError(
                    f"ambient_temperature_C references column {value!r}, "
                    "which is not present in the input data."
                )
            return prepared[value].astype(float)
        return pd.Series(float(value), index=prepared.index)

    def _dhw_stub_loss(
        self, prepared: pd.DataFrame, op_hours: pd.Series, opts: dict[str, Any]
    ) -> pd.Series:
        """Eq.(8)-(9): additional loss of open-circuited DHW stubs while tapping."""

        volume_m3 = float(opts["stub_volume_m3"])
        if volume_m3 <= _KWH_EPS:
            return pd.Series(0.0, index=op_hours.index)

        ambient = self._resolve_series(prepared, opts["stub_ambient_temperature_C"])
        hot_water = _series_from_aliases(
            prepared,
            ["theta_W_dis_hot_C", "T_W_supply_C", "dhw_temperature_C"],
            default=self.services["W"]["dhw_temperature_C"],
        ).astype(float)

        # Eq.(9): mass flow of hot water in the open-circuited stub(s) during
        # tapping [kg/h], from the stub pipe volume, water density and the
        # number of tappings per hour.
        m_stub_kg_h = volume_m3 * _WATER_DENSITY_KG_M3 * float(opts["taps_per_hour"])
        delta = (hot_water - ambient).clip(lower=0.0)
        return m_stub_kg_h * _WATER_SPECIFIC_HEAT_KWH_KGK * delta * op_hours

    def _part_load(
        self,
        q_out: pd.Series,
        hours: pd.Series,
        op_hours: pd.Series,
        opts: dict[str, Any],
    ) -> pd.Series:
        mode = opts["part_load_mode"]
        if mode == "constant_when_on":
            return pd.Series(1.0, index=q_out.index).where(op_hours > _KWH_EPS, 0.0)

        nominal = float(opts["nominal_power_kW"])
        if nominal <= _KWH_EPS:
            power = q_out / hours.replace(0.0, np.nan)
            nominal = max(float(power.max(skipna=True) or 0.0), _KWH_EPS)
        beta = q_out / hours.replace(0.0, np.nan) / nominal
        beta = beta.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        return beta.where(op_hours > _KWH_EPS, 0.0)

    def _pump_auxiliary(
        self,
        q_out: pd.Series,
        hours: pd.Series,
        op_hours: pd.Series,
        beta: pd.Series,
        opts: dict[str, Any],
        service: str,
    ) -> dict[str, pd.Series | float]:
        design_flow = self._design_flow(q_out, hours, opts)
        delta_p = self._design_delta_pressure(opts)
        p_hydr = delta_p * design_flow / 3600.0
        if p_hydr <= _KWH_EPS or design_flow <= _KWH_EPS or delta_p <= _KWH_EPS:
            zero = pd.Series(0.0, index=q_out.index)
            return {
                "W_hydr_kWh": zero,
                "W_aux_kWh": zero,
                "epsilon": zero,
                "P_hydr_des_kW": 0.0,
                "delta_p_des_kPa": delta_p,
                "design_flow_m3_h": design_flow,
            }

        epsilon = self._pump_expenditure_factor(beta, p_hydr, opts, service)
        w_hydr = p_hydr * beta * op_hours * float(opts["hydraulic_correction_factor"])
        w_aux = w_hydr * epsilon

        pump_mode = opts.get("pump_mode")
        if pump_mode is not None:
            # Eq.(29)/(30): intermittent circulation-pump operation in space
            # heating/cooling systems has a setback phase (minimum speed, a
            # 30% mean pump efficiency is assumed) and a boost phase (full
            # electrical design power), on top of the regular mode above.
            mode = pump_mode.reindex(q_out.index).astype(str).str.lower()
            is_setback = mode == "setback"
            is_boost = mode == "boost"
            w_setback = p_hydr * hours
            w_boost = 3.3 * p_hydr * hours
            w_hydr = w_hydr.where(~is_setback, w_setback).where(~is_boost, w_boost)
            w_aux = w_aux.where(~is_setback, w_setback).where(~is_boost, w_boost)

        return {
            "W_hydr_kWh": w_hydr,
            "W_aux_kWh": w_aux,
            "epsilon": epsilon,
            "P_hydr_des_kW": p_hydr,
            "delta_p_des_kPa": delta_p,
            "design_flow_m3_h": design_flow,
        }

    def _design_flow(
        self, q_out: pd.Series, hours: pd.Series, opts: dict[str, Any]
    ) -> float:
        direct = float(opts["design_flow_m3_h"])
        if direct > _KWH_EPS:
            return direct

        nominal_power = float(opts["nominal_power_kW"])
        if nominal_power <= _KWH_EPS:
            power = q_out / hours.replace(0.0, np.nan)
            nominal_power = max(float(power.max(skipna=True) or 0.0), 0.0)
        if nominal_power <= _KWH_EPS:
            return 0.0
        return nominal_power / (
            _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K * float(opts["design_deltaT_K"])
        )

    def _design_delta_pressure(self, opts: dict[str, Any]) -> float:
        direct = float(opts["design_delta_pressure_kPa"])
        if direct > _KWH_EPS:
            return direct
        return (
            (1.0 + float(opts["resistance_ratio"]))
            * float(opts["pressure_loss_per_m_kPa"])
            * float(opts["max_length_m"])
            + float(opts["additional_pressure_kPa"])
        )

    def _pump_expenditure_factor(
        self,
        beta: pd.Series,
        p_hydr_kW: float,
        opts: dict[str, Any],
        service: str,
    ) -> pd.Series:
        cp1, cp2 = _pump_control_constants(service, int(opts["pump_control_code"]))
        f_e, eei_override = self._pump_efficiency_factor(p_hydr_kW, opts)
        eei = eei_override if eei_override is not None else float(opts["eei"])
        beta_pos = beta.where(beta > _KWH_EPS, np.nan)
        epsilon = f_e * (cp1 + cp2 / beta_pos) * eei / 0.25
        return epsilon.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _pump_efficiency_factor(
        self, p_hydr_kW: float, opts: dict[str, Any]
    ) -> tuple[float, float | None]:
        label_power = float(opts["pump_label_power_kW"])
        if label_power > _KWH_EPS:
            return label_power / p_hydr_kW, None  # Eq.(28): existing installation

        p_hydr_W = p_hydr_kW * 1000.0
        if 1.0 < p_hydr_W < 2500.0:
            # Eq.(26): EU Regulation 622/2012 reference power for wet-running
            # circulation pumps. The formula is evaluated in W (the "17"
            # constant is not scale-invariant); the final x1000 conversion to
            # kW in the standard cancels in this ratio, so it is omitted here.
            p_ref_W = 1.7 * p_hydr_W + 17.0 * (1.0 - np.exp(-0.3 * p_hydr_W))
            return float(p_ref_W / p_hydr_W), None

        # Eq.(27): all other pumps. The standard requires EEI = 0.25 here,
        # which makes the EEI/0.25 term in Eq.(24) equal to 1.
        b = float(opts["pump_selection_factor"])
        f_e = (1.25 + (0.2 / p_hydr_kW) ** 0.5) * b
        return float(f_e), 0.25

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

        summary: dict[str, float] = {}
        for prefix in ["H", "C", "W"]:
            q_out = s(f"Q_{prefix}_dis_out_kWh")
            q_in = s(f"Q_{prefix}_dis_in_kWh")
            q_ls = s(f"Q_{prefix}_dis_ls_kWh")
            w_aux = s(f"W_{prefix}_dis_aux_kWh")
            summary.update(
                {
                    f"Q{prefix}_dis_out_kWh": q_out,
                    f"Q{prefix}_dis_ls_kWh": q_ls,
                    f"Q{prefix}_dis_th_rbl_kWh": s(f"Q_{prefix}_dis_th_rbl_kWh"),
                    f"W{prefix}_dis_aux_kWh": w_aux,
                    f"Q{prefix}_dis_aux_rbl_kWh": s(f"Q_{prefix}_dis_aux_rbl_kWh"),
                    f"Q{prefix}_dis_aux_rvd_kWh": s(f"Q_{prefix}_dis_aux_rvd_kWh"),
                    f"Q{prefix}_dis_rbl_kWh": s(f"Q_{prefix}_dis_rbl_kWh"),
                    f"Q{prefix}_dis_in_kWh": q_in,
                    f"e_{prefix}_dis_an": _ratio(q_in, q_out),
                    f"beta_{prefix}_dis_mean": _weighted_mean(
                        results[f"beta_{prefix}_dis"], results[f"t_{prefix}_dis_op_h"]
                    ),
                    f"epsilon_{prefix}_dis_mean": _weighted_mean(
                        results[f"epsilon_{prefix}_dis"], results[f"t_{prefix}_dis_op_h"]
                    ),
                    f"t_{prefix}_dis_op_h": s(f"t_{prefix}_dis_op_h"),
                }
            )
        summary["QW_dis_stub_kWh"] = s("Q_W_dis_stub_kWh")
        if "W_W_dis_rib_kWh" in results:
            summary["WW_dis_rib_kWh"] = s("W_W_dis_rib_kWh")

        summary["Q_dis_ls_kWh"] = (
            summary["QH_dis_ls_kWh"]
            + summary["QC_dis_ls_kWh"]
            + summary["QW_dis_ls_kWh"]
        )
        summary["W_dis_aux_kWh"] = (
            summary["WH_dis_aux_kWh"]
            + summary["WC_dis_aux_kWh"]
            + summary["WW_dis_aux_kWh"]
        )
        summary["Q_dis_rbl_kWh"] = (
            summary["QH_dis_rbl_kWh"]
            + summary["QC_dis_rbl_kWh"]
            + summary["QW_dis_rbl_kWh"]
        )
        return summary


def _section_options(data: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(data or {})
    return {
        "length_m": max(float(cfg.get("length_m", 0.0)), 0.0),
        "equivalent_length_m": max(float(cfg.get("equivalent_length_m", 0.0)), 0.0),
        "linear_thermal_transmittance_W_mK": max(
            float(
                cfg.get(
                    "linear_thermal_transmittance_W_mK",
                    cfg.get("psi_W_mK", 0.0),
                )
            ),
            0.0,
        ),
        # A number is a constant ambient temperature; a string names a column
        # in the input timeseries, for EN 15316-3 Table 10's "Varying: YES"
        # ambient temperatures (e.g. different value in/out of heating season).
        "ambient_temperature_C": cfg.get("ambient_temperature_C", 20.0),
        "recoverable": bool(cfg.get("recoverable", True)),
        # DHW only. False = distribution pipe without circulation (a branch or
        # stub run): EN 15316-3 Eq.(15) applies for its whole duration, not
        # just while not tapping.
        "circulating": bool(cfg.get("circulating", True)),
        # DHW only, used by Eq.(31) (ribbon heater): if no section in the
        # service is flagged, Eq.(31) falls back to the service's total
        # operating-time loss.
        "ribbon_target": bool(cfg.get("ribbon_target", False)),
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
    except Exception as exc:  # pragma: no cover - defensive type context
        raise ValueError(f"{name} must be a fraction between 0 and 1.") from exc
    return float(np.clip(fraction, 0.0, 1.0))


def _pump_control_constants(service: str, control_code: int) -> tuple[float, float]:
    if service == "H":
        return {0: (0.25, 0.75), 3: (0.75, 0.25), 4: (0.90, 0.10)}.get(
            control_code, (0.90, 0.10)
        )
    if service == "C":
        return {0: (0.25, 0.75), 3: (0.85, 0.15), 4: (0.85, 0.15)}.get(
            control_code, (0.85, 0.15)
        )
    return {0: (0.25, 0.94), 3: (0.50, 0.63), 4: (0.50, 0.63)}.get(
        control_code, (0.50, 0.63)
    )


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
