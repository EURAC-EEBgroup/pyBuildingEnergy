"""EN 15316-4-1 boiler generation calculator.

This module implements the cycle / expenditure-factor method for combustion
generators in a table-driven way. It intentionally avoids hardcoded default
performance values: all generator-case data must be provided through the input
configuration or a custom table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

_EPS = 1e-9

GENERATOR_TYPES = {
    "standard",
    "low_temperature",
    "condensing",
    "biomass_log",
    "biomass_pellet",
}

LOCATION_TYPES = {
    "inside_heated",
    "adjacent_unheated",
    "outside_building",
}


@dataclass
class BoilerConfig:
    boiler_type: str
    fuel_type: str
    rated_power_kW: float
    intermediate_load_fraction: float
    eta_Pn_test_pct: float
    eta_Pint_test_pct: float
    theta_test_Pn_C: float
    theta_test_Pint_C: float
    f_corr_pct_per_K: float
    P_gen_ls_P0_W: float
    P_aux_on_W: float
    P_aux_off_W: float
    f_jacket: float
    f_location: float
    f_aux_recoverable: float
    dew_point_C: float
    condensing_gain_pct: float


class BoilerGeneratorCalculator:
    """EN 15316-4-1 boiler generator calculator."""

    def __init__(self, input_data: Optional[dict[str, Any]] = None):
        self.input_data = dict(input_data or {})
        self._load_parameters()

    def _require(self, key: str, default: Any = None) -> Any:
        if key in self.input_data:
            return self.input_data[key]
        if default is not None:
            return default
        raise ValueError(f"Missing required input parameter: {key}")

    def _load_parameters(self) -> None:
        inp = self.input_data

        btype = str(self._require("boiler_type")).lower()
        if btype not in GENERATOR_TYPES:
            raise ValueError(f"Unknown boiler_type '{btype}'. Expected one of {sorted(GENERATOR_TYPES)}.")

        loc_key = str(self._require("boiler_location")).lower()
        if loc_key not in LOCATION_TYPES:
            raise ValueError(f"Unknown boiler_location '{loc_key}'. Expected one of {sorted(LOCATION_TYPES)}.")

        self.cfg = BoilerConfig(
            boiler_type=btype,
            fuel_type=str(self._require("fuel_type")).lower(),
            rated_power_kW=float(self._require("rated_power_kW")),
            intermediate_load_fraction=float(self._require("intermediate_load_fraction")),
            eta_Pn_test_pct=float(self._require("eta_Pn_test_pct")),
            eta_Pint_test_pct=float(self._require("eta_Pint_test_pct")),
            theta_test_Pn_C=float(self._require("theta_test_Pn_C")),
            theta_test_Pint_C=float(self._require("theta_test_Pint_C")),
            f_corr_pct_per_K=float(self._require("f_corr_pct_per_K")),
            P_gen_ls_P0_W=float(self._require("P_gen_ls_P0_W")),
            P_aux_on_W=float(self._require("P_aux_on_W")),
            P_aux_off_W=float(self._require("P_aux_off_W")),
            f_jacket=float(self._require("f_jacket")),
            f_location=float(self._require("f_location")),
            f_aux_recoverable=float(self._require("f_aux_recoverable")),
            dew_point_C=float(self._require("dew_point_C")),
            condensing_gain_pct=float(self._require("condensing_gain_pct")),
        )

        self.eta_table = self._load_case_table("efficiency_table")
        self.loss_table = self._load_case_table("loss_table")

        assert 0.0 < self.cfg.rated_power_kW
        assert 0.0 < self.cfg.intermediate_load_fraction < 1.0
        assert 0.0 <= self.cfg.f_jacket <= 1.0
        assert 0.0 <= self.cfg.f_location <= 1.0
        assert 0.0 <= self.cfg.f_aux_recoverable <= 1.0

    def _load_case_table(self, key: str) -> dict[str, Any]:
        table = self._require(key)
        if not isinstance(table, dict):
            raise ValueError(f"{key} must be a dict with case definitions.")
        return table

    def _case_values(self, table: dict[str, Any], fallback_key: str | None = None) -> dict[str, float]:
        case = table.get(self.cfg.boiler_type)
        if case is None and fallback_key is not None:
            case = table.get(fallback_key)
        if case is None:
            raise ValueError(f"Missing case data for boiler_type '{self.cfg.boiler_type}'.")
        if not isinstance(case, dict):
            raise ValueError(f"Case data for '{self.cfg.boiler_type}' must be a dict.")
        return case

    def _eta_corrected_pct(self, eta_test_pct: float, theta_test_C: float, theta_avg_C: float,
                           theta_return_C: Optional[float]) -> float:
        eta = float(eta_test_pct) - self.cfg.f_corr_pct_per_K * (float(theta_avg_C) - float(theta_test_C))
        if self.cfg.boiler_type == "condensing" and theta_return_C is not None:
            tr = float(theta_return_C)
            if tr < self.cfg.dew_point_C:
                ratio = max(0.0, min(1.0, (self.cfg.dew_point_C - tr) / max(self.cfg.dew_point_C, _EPS)))
                eta += self.cfg.condensing_gain_pct * ratio
        return float(max(min(eta, 110.0), 1.0))

    def _eta_triplet(self, theta_avg_C: float, theta_return_C: Optional[float]) -> dict[str, float]:
        eff_case = self._case_values(self.eta_table)
        return {
            "eta_Pn_corr(%)": self._eta_corrected_pct(
                eff_case["eta_Pn_test_pct"], eff_case["theta_test_Pn_C"], theta_avg_C, theta_return_C
            ),
            "eta_Pint_corr(%)": self._eta_corrected_pct(
                eff_case["eta_Pint_test_pct"], eff_case["theta_test_Pint_C"], theta_avg_C, theta_return_C
            ),
        }

    def _loss_powers_kW(self, theta_avg_C: float, theta_return_C: Optional[float]) -> dict[str, float]:
        triplet = self._eta_triplet(theta_avg_C, theta_return_C)
        case = self._case_values(self.loss_table)
        Pn = self.cfg.rated_power_kW
        Pint = self.cfg.intermediate_load_fraction * Pn
        return {
            **triplet,
            "P_gen_ls_Pn(kW)": Pn * max((100.0 / triplet["eta_Pn_corr(%)"] - 1.0), 0.0),
            "P_gen_ls_Pint(kW)": Pint * max((100.0 / triplet["eta_Pint_corr(%)"] - 1.0), 0.0),
            "P_gen_ls_P0(kW)": float(case["P_gen_ls_P0_W"]) / 1000.0,
        }

    def _loss_power_at_load_kW(self, beta: float, P_ls_Pn: float, P_ls_Pint: float, P_ls_P0: float) -> float:
        beta = max(min(float(beta), 1.0), 0.0)
        b_int = self.cfg.intermediate_load_fraction
        if beta <= _EPS:
            return P_ls_P0
        if beta <= b_int:
            return P_ls_P0 + (beta / b_int) * (P_ls_Pint - P_ls_P0)
        return P_ls_Pint + ((beta - b_int) / max(1.0 - b_int, _EPS)) * (P_ls_Pn - P_ls_Pint)

    def _auxiliary_energy_kWh(self, t_run_h: float, t_use_h: float) -> dict[str, float]:
        t_off = max(float(t_use_h) - float(t_run_h), 0.0)
        W_on = self.cfg.P_aux_on_W * float(t_run_h) / 1000.0
        W_off = self.cfg.P_aux_off_W * t_off / 1000.0
        return {
            "W_gen_aux_on(kWh)": W_on,
            "W_gen_aux_off(kWh)": W_off,
            "W_gen_aux(kWh)": W_on + W_off,
        }

    def compute_step(
        self,
        Q_gen_out_kWh: float,
        t_use_h: float = 1.0,
        theta_avg_C: float = 45.0,
        theta_return_C: Optional[float] = None,
        Q_H_gen_out_kWh: Optional[float] = None,
        Q_W_gen_out_kWh: Optional[float] = None,
    ) -> dict[str, float]:
        Q_out = max(float(Q_gen_out_kWh), 0.0)
        t_use = max(float(t_use_h), _EPS)
        Pn = self.cfg.rated_power_kW

        triplet = self._loss_powers_kW(theta_avg_C, theta_return_C)
        beta = min((Q_out / t_use) / max(Pn, _EPS), 1.0)
        P_gen_ls_beta = self._loss_power_at_load_kW(
            beta,
            triplet["P_gen_ls_Pn(kW)"],
            triplet["P_gen_ls_Pint(kW)"],
            triplet["P_gen_ls_P0(kW)"],
        )

        t_run = min(Q_out / max(Pn, _EPS), t_use)
        t_off = t_use - t_run

        Q_gen_ls_on = P_gen_ls_beta * t_run
        Q_gen_ls_off = triplet["P_gen_ls_P0(kW)"] * t_off
        Q_gen_ls = Q_gen_ls_on + Q_gen_ls_off
        E_gen_in = Q_out + Q_gen_ls
        e_gen = (E_gen_in / Q_out) if Q_out > _EPS else float("nan")

        aux = self._auxiliary_energy_kWh(t_run, t_use)
        Q_gen_ls_env = Q_gen_ls * self.cfg.f_jacket
        Q_gen_ls_rbl = Q_gen_ls_env * self.cfg.f_location
        W_gen_aux_rbl = aux["W_gen_aux(kWh)"] * self.cfg.f_aux_recoverable

        if (Q_H_gen_out_kWh is not None) or (Q_W_gen_out_kWh is not None):
            qh = max(float(Q_H_gen_out_kWh or 0.0), 0.0)
            qw = max(float(Q_W_gen_out_kWh or 0.0), 0.0)
            qsum = qh + qw
            if qsum > _EPS:
                E_H_gen_in = E_gen_in * qh / qsum
                E_W_gen_in = E_gen_in * qw / qsum
            else:
                E_H_gen_in = 0.0
                E_W_gen_in = 0.0
        else:
            E_H_gen_in = E_gen_in
            E_W_gen_in = 0.0

        return {
            "Q_gen_out(kWh)": Q_out,
            "t_use(h)": t_use,
            "theta_avg(°C)": float(theta_avg_C),
            "theta_return(°C)": float(theta_return_C) if theta_return_C is not None else float("nan"),
            "beta_gen(-)": beta,
            "t_run(h)": t_run,
            "t_off(h)": t_off,
            **triplet,
            "P_gen_ls_beta(kW)": P_gen_ls_beta,
            "Q_gen_ls_on(kWh)": Q_gen_ls_on,
            "Q_gen_ls_off(kWh)": Q_gen_ls_off,
            "Q_gen_ls(kWh)": Q_gen_ls,
            "E_gen_in(kWh)": E_gen_in,
            "e_gen(-)": e_gen,
            "E_H_gen_in(kWh)": E_H_gen_in,
            "E_W_gen_in(kWh)": E_W_gen_in,
            **aux,
            "Q_gen_ls_env(kWh)": Q_gen_ls_env,
            "Q_gen_ls_rbl(kWh)": Q_gen_ls_rbl,
            "W_gen_aux_rbl(kWh)": W_gen_aux_rbl,
        }

    def run_timeseries(self, df: pd.DataFrame, column_map: Optional[dict[str, str]] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        alias = {
            "Q_gen_out_kWh": ["Q_gen_out_kWh", "Q_gen_out", "Q_HW_dis_in_kWh", "Q_HW_dis_in", "Q_gen", "Q_out_kWh"],
            "Q_H_gen_out_kWh": ["Q_H_gen_out_kWh", "QH_gen_out_kWh", "QH_gen_out", "Q_H_gen_out"],
            "Q_W_gen_out_kWh": ["Q_W_gen_out_kWh", "QW_gen_out_kWh", "QW_gen_out", "Q_W_gen_out", "Q_W_dis_i_in_kWh", "QW_dis_i_in"],
            "t_use_h": ["t_use_h", "t_use", "dt_h", "delta_t_h"],
            "theta_avg_C": ["theta_avg_C", "theta_avg", "theta_gen_avg_C", "θ_avg_C", "theta_gen_mean"],
            "theta_return_C": ["theta_return_C", "theta_return", "theta_gen_ret_C", "θ_ret_C", "theta_X_gen_cr_ret"],
        }
        if column_map:
            for key, name in column_map.items():
                if key in alias:
                    alias[key] = [name, *alias[key]]

        def pick(row: pd.Series, names: list[str], default: Any = None) -> Any:
            for n in names:
                if n in row.index:
                    v = row[n]
                    if v is None:
                        continue
                    try:
                        if pd.isna(v):
                            continue
                    except (TypeError, ValueError):
                        pass
                    return v
            return default

        records = []
        for idx, row in df.iterrows():
            Q_out = pick(row, alias["Q_gen_out_kWh"])
            if Q_out is None:
                qh = pick(row, alias["Q_H_gen_out_kWh"], default=0.0) or 0.0
                qw = pick(row, alias["Q_W_gen_out_kWh"], default=0.0) or 0.0
                Q_out = float(qh) + float(qw)

            res = self.compute_step(
                Q_gen_out_kWh=float(Q_out),
                t_use_h=float(pick(row, alias["t_use_h"], default=1.0) or 1.0),
                theta_avg_C=float(pick(row, alias["theta_avg_C"], default=45.0) or 45.0),
                theta_return_C=pick(row, alias["theta_return_C"], default=None),
                Q_H_gen_out_kWh=pick(row, alias["Q_H_gen_out_kWh"], default=None),
                Q_W_gen_out_kWh=pick(row, alias["Q_W_gen_out_kWh"], default=None),
            )
            res["_index"] = idx
            records.append(res)

        out = pd.DataFrame.from_records(records)
        if "_index" in out.columns:
            out = out.set_index("_index")
            out.index.name = df.index.name
        return out


__all__ = ["BoilerConfig", "BoilerGeneratorCalculator", "GENERATOR_TYPES", "LOCATION_TYPES"]
