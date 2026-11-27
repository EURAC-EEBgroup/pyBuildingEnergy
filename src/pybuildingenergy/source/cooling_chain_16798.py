"""Coupled cooling chain EN 16798-9 -> EN 16798-15 -> EN 16798-13.

The three cooling modules can be run one after the other, each stage reading the
request of the previous one. This module closes the loop between them:

1. EN 16798-9 gives the request at the distribution inlet ``Q_C_gen_in_req_kWh``
   (circuit demands, losses) and the required flow temperatures;
2. EN 16798-15 turns it into the request to the generator (direct supply, charging,
   losses, pump heat) and the required generator outlet temperature (formula (1));
3. EN 16798-13 gives, for that temperature, the cooling the generation system can
   actually supply in the step (free cooling + generators (7)-(9) + backup).

When the generator cannot supply the storage request, the storage is simulated again
with the available energy as a limit (``Q_C_sto_gen_avl_kWh``): the users are served
before the charging, the store discharges what it holds to cover a generator
shortfall (``discharge_on_generator_shortfall``), and the state of charge follows
the charge really received. Since the available energy depends on the required
temperature, which depends on the state of charge, storage and generator are
iterated until the storage request fits the generator availability.

The shortfall towards the users (``Q_C_sto_unmet_kWh``, or the generator unmet
without storage) is then distributed to the circuits and zones of EN 16798-9 with
:meth:`CoolingSystemCalculator.distribute_shortfall`: the circuits with the lowest
priority (largest ``priority`` number) lose first, equal priorities pro rata.

Temperatures back to the distribution: once storage and generator agree, EN 16798-9
is run again with the actual generator outlet temperature (steps with direct supply)
and storage outlet temperature (steps with discharge), so the flow temperature (23)
and the return temperature (24) are the effective ones; ``h_C_dis_flw_not_reached``
counts the hours with demand in which the required flow temperature is not reached.
With terminal data in EN 16798-9 (``terminal_design_capacity_kW``, see
``CoolingSystemCalculator.terminal_limits``) the circuit demand is limited to what the
terminals deliver at that temperature and the chain is run again (outer iteration),
so the shortfall is split into ``Q_C_dis_unmet_temp_kWh_<circuit>`` (water too warm
for the terminals) and ``Q_C_dis_unmet_gen_kWh_<circuit>`` (energy the storage /
generator could not supply). A RuntimeWarning is raised when an iteration does not
converge within ``max_iterations``.

Every stage receives a single, explicit demand column (``Q_C_sto_out_kWh`` for the
storage, ``Q_C_gen_in_req_kWh`` for the generator), so the alias precedence of the
modules never picks a stale column.

Assumptions (library choices, not in the standards):

* the generator availability is the cooling the generation system supplies when
  asked for an unlimited request at the temperature required by the storage;
  with a backup (``cooling_backup_eer``) the availability is unlimited;
* the charging temperature the generator can reach is
  max(theta_C_sto_C, ``theta_C_gen_out_limit_C`` of the generator): an ice / PCM
  store whose charging temperature is below the generator limit is not charged
  in the latent band;
* a residual generator shortfall left after ``max_iterations`` is assigned to the
  users up to the direct supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import warnings

import numpy as np
import pandas as pd

from .cooling_16798_9 import CoolingSystemCalculator, CoolingSystemSimulationResult
from .cooling_generation_16798_13 import CoolingGenerationSimulationResult, CoolingGenerationSystemCalculator
from .cooling_storage_16798_15 import CoolingStorageSimulationResult, CoolingStorageSystemCalculator


_BIG_KWH = 1e9
_PASS_THROUGH = (
    "T_ext", "T_ext_wb_C", "theta_sk_C", "theta_H_C_gen_abs_in_C", "f_op_C", "f_op_ctrl",
    "T_C_source_C", "time_step_hours", "theta_C_sto_amb_C", "f_C_sto_ctrl", "f_op_hr_wet",
)


@dataclass
class CoolingChainResult:
    """Results of :class:`CoolingChainCalculator`."""

    distribution: CoolingSystemSimulationResult | None
    storage: CoolingStorageSimulationResult | None
    generation: CoolingGenerationSimulationResult
    delivered: pd.DataFrame
    summary: dict[str, float]
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)
    outer_iterations: int = 1


class CoolingChainCalculator:
    """Distribution (EN 16798-9), optional storage (EN 16798-15) and generation
    (EN 16798-13) with the storage-generator balance closed in time.

    Each stage accepts a configuration dictionary or an already built calculator.
    ``distribution=None`` reads the demand from ``Q_C_kWh`` (e.g. the inlet of an
    EN 15316-3 distribution) and the required temperatures from
    ``theta_C_gen_out_req_C`` / ``theta_C_dis_in_flw_req_C`` (or
    ``theta_C_dis_supply_C``) / ``T_C_sink_C`` (default 7 C); the shortfall can then be
    sent to the EN 16798-9 circuits with ``distribute_shortfall``.
    """

    def __init__(
        self,
        generation: dict[str, Any] | CoolingGenerationSystemCalculator,
        distribution: dict[str, Any] | CoolingSystemCalculator | None = None,
        storage: dict[str, Any] | CoolingStorageSystemCalculator | None = None,
        max_iterations: int = 20,
        tolerance_kWh: float = 1e-6,
    ):
        self.distribution = _build(distribution, CoolingSystemCalculator)
        self.storage = _build(storage, CoolingStorageSystemCalculator)
        if self.storage is not None and not self.storage.enabled:
            self.storage = None
        self.generation = _build(generation, CoolingGenerationSystemCalculator)
        if self.generation is None:
            raise ValueError("the cooling chain needs a generation (EN 16798-13) configuration.")
        if int(max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1.")
        self.max_iterations = int(max_iterations)
        self.tolerance_kWh = float(tolerance_kWh)

    def run_timeseries(self, data: pd.DataFrame) -> CoolingChainResult:
        dist = self.distribution
        terminals = dist is not None and (
            any(c.get("terminal") for c in dist.circuits) if dist.circuits else dist.terminal is not None)
        original = dist.run_timeseries(data) if terminals else None
        f_term: dict[str, np.ndarray] | None = None
        outer_converged, outer = True, 0
        for outer in range(1, (self.max_iterations if terminals else 1) + 1):
            data_k = data if f_term is None else self._scaled(data, f_term)
            dist_req, base = self._distribution_stage(data_k)
            demand = base.pop("_demand")
            sto_res, gen_res, iterations, converged, history, users_unmet = self._supply(base, demand)
            dist_res = self._effective_distribution(data_k, dist_req, sto_res, gen_res)
            if not terminals:
                break
            new_f = self._terminal_fractions(dist_res, original)
            change = 0.0 if f_term is None else max(np.max(np.abs(new_f[k] - f_term[k])) for k in new_f)
            if f_term is not None and change <= 1e-9:
                break
            if f_term is None and all(np.all(v >= 1.0 - 1e-12) for v in new_f.values()):
                f_term = new_f
                break
            f_term = new_f
        else:
            outer_converged = not terminals
        if not (converged and outer_converged):
            warnings.warn(
                f"CoolingChainCalculator: no convergence after {self.max_iterations} iterations "
                f"(storage-generator residual {history[-1] if history else 0.0:.3g} kWh, generator "
                f"residual {gen_res.summary['QC_gen_unmet_kWh']:.3g} kWh); check 'converged'.",
                RuntimeWarning, stacklevel=2)
        delivered = self._delivered(dist_res, original, f_term, demand, users_unmet)
        summary = self._summary(demand, delivered, dist_res, sto_res, gen_res, iterations,
                                converged and outer_converged, outer)
        return CoolingChainResult(dist_res, sto_res, gen_res, delivered, summary, iterations,
                                  converged and outer_converged, history, outer)

    def _supply(self, base: pd.DataFrame, demand: pd.Series):
        if self.storage is None:
            gen_res = self.generation.run_timeseries(base.assign(Q_C_gen_in_req_kWh=demand))
            users_unmet = gen_res.timeseries["Q_C_gen_unmet_kWh"].to_numpy(dtype=float)
            sto_res, iterations, converged, history = None, 1, True, []
        else:
            sto_res, gen_res, iterations, converged, history = self._storage_loop(base, demand)
            sto = sto_res.timeseries
            residual = gen_res.timeseries["Q_C_gen_unmet_kWh"].to_numpy(dtype=float)
            users_unmet = sto["Q_C_sto_unmet_kWh"].to_numpy(dtype=float) + np.minimum(
                residual, sto["Q_C_sto_direct_kWh"].to_numpy(dtype=float))
        users_unmet = np.clip(users_unmet, 0.0, demand.to_numpy(dtype=float))
        users_unmet[users_unmet < 1e-9] = 0.0
        return sto_res, gen_res, iterations, converged, history, users_unmet

    def _effective_distribution(self, data, dist_req, sto_res, gen_res):
        """EN 16798-9 again with the actual generator / storage outlet temperatures, (23)/(24)."""
        if self.distribution is None:
            return None
        gen_t = gen_res.timeseries["theta_C_gen_out_C"].astype(float)
        if sto_res is None:
            return self.distribution.run_timeseries(data.assign(theta_C_gen_out_C=gen_t))
        sto = sto_res.timeseries
        return self.distribution.run_timeseries(data.assign(
            theta_C_gen_out_C=gen_t.where(sto["Q_C_sto_direct_kWh"] > 1e-9),
            theta_C_sto_out_C=sto["theta_C_sto_out_act_C"].where(sto["Q_C_sto_dis_kWh"] > 1e-9)))

    def _terminal_fractions(self, dist_res, original) -> dict[str, np.ndarray]:
        lim = self.distribution.terminal_limits(dist_res)
        ts0 = original.timeseries
        out = {}
        for name in self.distribution.demand_columns_by_circuit():
            col = f"Q_C_term_max_kWh_{name}"
            if col not in lim:
                continue
            req = (ts0[f"Q_C_dis_in_req_kWh_{name}"] if self.distribution.circuits
                   else ts0["Q_C_dis_out_tot_req_kWh"]).to_numpy(dtype=float)
            out[name] = np.divide(lim[col].to_numpy(), req, out=np.ones(len(req)),
                                  where=req > 1e-12).clip(0.0, 1.0)
        return out

    def _scaled(self, data: pd.DataFrame, f_term: dict[str, np.ndarray]) -> pd.DataFrame:
        scaled = data.copy()
        for name, cols in self.distribution.demand_columns_by_circuit().items():
            if name not in f_term:
                continue
            for col in cols:
                if col in scaled.columns:
                    scaled[col] = scaled[col].astype(float) * f_term[name]
        return scaled

    # ------------------------------------------------------------------ stages
    def _distribution_stage(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame.")
        base = pd.DataFrame(index=data.index)
        for col in _PASS_THROUGH:
            if col in data.columns:
                base[col] = data[col]
        if "T_ext" not in base and "theta_ext" in data.columns:
            base["T_ext"] = data["theta_ext"]
        if self.distribution is None:
            if "Q_C_kWh" not in data.columns:
                raise ValueError("without a distribution stage the data need 'Q_C_kWh'.")
            base["_demand"] = data["Q_C_kWh"].astype(float).clip(lower=0.0)
            sink = data["T_C_sink_C"] if "T_C_sink_C" in data.columns else 7.0
            base["theta_C_gen_out_req_C"] = data.get("theta_C_gen_out_req_C", sink)
            base["theta_C_dis_in_flw_req_C"] = data.get(
                "theta_C_dis_in_flw_req_C", data.get("theta_C_dis_supply_C", sink))
            if "theta_C_dis_return_C" in data.columns and data["theta_C_dis_return_C"].notna().all():
                base["theta_C_dis_return_C"] = data["theta_C_dis_return_C"]
            return None, base
        res = self.distribution.run_timeseries(data)
        ts = res.timeseries
        base["_demand"] = ts["Q_C_gen_in_req_kWh"].astype(float)
        base["theta_C_gen_out_req_C"] = ts["theta_C_gen_out_req_C"]
        base["theta_C_dis_in_flw_req_C"] = ts["theta_C_dis_in_flw_req_C"]
        if ts["theta_C_dis_return_C"].notna().all():
            base["theta_C_dis_return_C"] = ts["theta_C_dis_return_C"]
        return res, base

    def _storage_loop(self, base: pd.DataFrame, demand: pd.Series):
        sto_calc, gen_calc = self.storage, self.generation
        t_charge = max(sto_calc.theta_C_sto_C, gen_calc.minimum_outlet_temperature_C)
        sto_in = pd.DataFrame(index=base.index)
        for col in ("T_ext", "time_step_hours", "theta_C_sto_amb_C", "f_C_sto_ctrl",
                    "theta_C_dis_in_flw_req_C"):
            if col in base:
                sto_in[col] = base[col]
        sto_in["Q_C_sto_out_kWh"] = demand
        sto_in["theta_C_gen_out_C"] = t_charge
        gen_base = base.drop(columns=["theta_C_dis_in_flw_req_C"], errors="ignore")

        available = None
        history: list[float] = []
        converged = False
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            frame = sto_in if available is None else sto_in.assign(Q_C_sto_gen_avl_kWh=available)
            sto_res = sto_calc.run_timeseries(frame)
            sto = sto_res.timeseries
            gen_in = gen_base.assign(theta_C_gen_out_req_C=sto["theta_C_sto_in_req_C"])
            cap = gen_calc.run_timeseries(gen_in.assign(Q_C_gen_in_req_kWh=_BIG_KWH)).timeseries
            new_available = cap["Q_C_gen_in_kWh"].to_numpy(dtype=float)
            excess = np.maximum(sto["Q_C_sto_in_kWh"].to_numpy(dtype=float) - new_available, 0.0)
            history.append(float(excess.sum()))
            if excess.max() <= self.tolerance_kWh:
                converged = True
                break
            # keep the tighter limit seen so far: the state can only lose charge
            available = new_available if available is None else np.minimum(available, new_available)
        gen_res = gen_calc.run_timeseries(gen_in.assign(Q_C_gen_in_req_kWh=sto["Q_C_sto_in_kWh"]))
        return sto_res, gen_res, iterations, converged, history

    def _delivered(self, dist_res, original, f_term, demand: pd.Series, users_unmet: np.ndarray) -> pd.DataFrame:
        if self.distribution is None:
            out = pd.DataFrame(index=demand.index)
            out["Q_C_dis_in_req_kWh_main"] = demand
            out["Q_C_dis_in_kWh_main"] = demand - users_unmet
            out["Q_C_dis_unmet_kWh_main"] = users_unmet
            out["Q_C_dis_unmet_gen_kWh_main"] = users_unmet
            out["Q_C_dis_unmet_temp_kWh_main"] = 0.0
            out["Q_C_dis_in_tot_kWh"] = out["Q_C_dis_in_kWh_main"]
            out["Q_C_dis_unmet_tot_kWh"] = users_unmet
            out["Q_C_dis_unmet_gen_tot_kWh"] = users_unmet
            out["Q_C_dis_unmet_temp_tot_kWh"] = 0.0
            return out
        out = self.distribution.distribute_shortfall(dist_res, users_unmet)
        names = list(self.distribution.demand_columns_by_circuit())
        ref = original.timeseries if original is not None else dist_res.timeseries
        temp_tot = np.zeros(len(out))
        for n in names:
            req0 = (ref[f"Q_C_dis_in_req_kWh_{n}"] if self.distribution.circuits
                    else ref["Q_C_dis_out_tot_req_kWh"]).to_numpy(dtype=float)
            gen_cut = out[f"Q_C_dis_unmet_kWh_{n}"].to_numpy(dtype=float)
            temp_cut = req0 - out[f"Q_C_dis_in_req_kWh_{n}"].to_numpy(dtype=float)
            temp_cut = np.where(temp_cut > 1e-12, temp_cut, 0.0)
            delivered = out[f"Q_C_dis_in_kWh_{n}"].to_numpy(dtype=float)
            out[f"Q_C_dis_in_req_kWh_{n}"] = req0
            out[f"Q_C_dis_unmet_gen_kWh_{n}"] = gen_cut
            out[f"Q_C_dis_unmet_temp_kWh_{n}"] = temp_cut
            out[f"Q_C_dis_unmet_kWh_{n}"] = gen_cut + temp_cut
            f_sup = np.divide(delivered, req0, out=np.ones(len(req0)), where=req0 > 1e-12)
            out[f"f_C_sup_{n}"] = f_sup
            if self.distribution.circuits:
                circuit = next(c for c in self.distribution.circuits if c["name"] == n)
                for col in circuit["demand_columns"]:
                    out[f"Q_C_del_kWh_{col}"] = f_sup * ref[f"Q_C_zone_req_kWh_{col}"].to_numpy(dtype=float)
            temp_tot += temp_cut
        out["Q_C_dis_unmet_gen_tot_kWh"] = out["Q_C_dis_unmet_tot_kWh"].to_numpy(dtype=float)
        out["Q_C_dis_unmet_temp_tot_kWh"] = temp_tot
        out["Q_C_dis_unmet_tot_kWh"] = out["Q_C_dis_unmet_gen_tot_kWh"] + temp_tot
        return out

    def _summary(self, demand, delivered, dist_res, sto_res, gen_res, iterations, converged, outer) -> dict[str, float]:
        g = gen_res.summary
        summary = {
            "QC_dis_req_kWh": float(delivered["Q_C_dis_in_tot_kWh"].sum() + delivered["Q_C_dis_unmet_tot_kWh"].sum()),
            "QC_users_unmet_kWh": float(delivered["Q_C_dis_unmet_tot_kWh"].sum()),
            "QC_users_unmet_gen_kWh": float(delivered["Q_C_dis_unmet_gen_tot_kWh"].sum()),
            "QC_users_unmet_temp_kWh": float(delivered["Q_C_dis_unmet_temp_tot_kWh"].sum()),
            "QC_users_delivered_kWh": float(delivered["Q_C_dis_in_tot_kWh"].sum()),
            "QC_gen_in_kWh": g["QC_gen_in_kWh"],
            "QC_gen_unmet_residual_kWh": g["QC_gen_unmet_kWh"],
            "EC_total_kWh": g["EC_total_kWh"],
            "QH_C_gen_abs_in_kWh": g["QH_C_gen_abs_in_kWh"],
            "iterations": float(iterations),
            "outer_iterations": float(outer),
            "converged": float(converged),
        }
        if dist_res is not None:
            ts = dist_res.timeseries
            dev = (ts["theta_C_dis_in_flw_C"] - ts["theta_C_dis_in_flw_req_C"]).where(
                ts["Q_C_dis_out_tot_req_kWh"] > 1e-9, 0.0).fillna(0.0)
            summary["h_C_dis_flw_not_reached"] = float(ts["hours"][dev > 0.01].sum())
            summary["theta_C_dis_flw_dev_max_K"] = float(dev.max())
        if sto_res is not None:
            s = sto_res.summary
            summary.update({
                "QC_sto_dis_kWh": s["QC_sto_dis_kWh"],
                "QC_sto_chg_kWh": s["QC_sto_chg_kWh"],
                "QC_sto_chg_unmet_kWh": s["QC_sto_chg_unmet_kWh"],
                "QC_sto_ls_tot_kWh": s["QC_sto_ls_tot_kWh"],
                "WC_sto_aux_kWh": s["WC_sto_aux_kWh"],
            })
            summary["EC_total_kWh"] += s["WC_sto_aux_kWh"]
        return summary


def _build(cfg, cls):
    if cfg is None:
        return None
    if isinstance(cfg, cls):
        return cfg
    if isinstance(cfg, dict):
        return cls(cfg)
    raise TypeError(f"expected a configuration dict or a {cls.__name__}.")
