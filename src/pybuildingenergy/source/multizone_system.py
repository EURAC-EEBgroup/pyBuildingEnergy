"""System topologies for a multizone building: emission per zone -> distribution -> generation.

Three topologies (``system["topology"]``):

``"central"``
    one generator and one distribution system; one emission system per zone (EN 15316-2,
    ``emission_by_zone``).  The emitter inputs of the zones are summed at the
    distribution (EN 15316-1 Annex C: one common circuit) and the distribution input feeds
    the generator.

``"per_zone"``
    for every zone its own emission, distribution and generator
    (``system["zones"][zone] = {"distribution": ..., "generation": ...}``).  The results
    are summed for the building.

``"combined"``
    one system for space heating, space cooling and domestic hot water: one emission type
    for all the zones (``system["emission"]``, e.g. radiant floor), one distribution with
    heating/cooling/DHW blocks, one generator and a DHW storage tank.  The generator
    serves one service at a time inside the hour (time share), so the DHW tank charging
    competes with the space heating and cooling demand: see :func:`dispatch_combined`.

Space heating and cooling come from :func:`simulate_multizone_emission` (equivalent
set-points, emitter capacity in the thermal balance).  Distribution uses
``DistributionSystemCalculator`` (EN 15316-3) and generation the calculator of the
generator kind (``heat_pump`` EN 15316-4-2, ``boiler_15316_4_1``, ``district``
EN 15316-4-5, or ``ideal`` with constant efficiencies).

Generator capacity (``couple_generator_capacity=True``): the declared generator power
(``capacity_heating_kW`` / ``capacity_cooling_kW`` on the generation, or the dispatch
capacities of ``combined``) limits the zones in the thermal balance, so a shortage changes
the zone temperatures and the adaptive-facade control.

* shared generator (``central``, ``combined``), ``system["capacity_allocation"]``:
  ``"dynamic"`` (default) - every hour, inside the thermal solver, the generator power is
  shared among the zones in proportion to their request of that hour (building
  ``generator_groups``); heating and cooling requested together share the generator time
  in the order ``space_priority``.  ``"static"`` - fixed share by emitter nominal power
  (or net floor area).
* ``per_zone``: each zone is limited by its own generator.

Not in the thermal balance (post-processed: unmet energy, displaced energy): the
distribution losses between generator and emitters, and the generator time taken by the
DHW tank in ``combined``.

``combined``: one emission type for all the zones (``system["emission"]``) or terminals per
zone (``emission_by_zone``) on the one generator.  The tank dispatch is recalculated on the
generator production (hourly for the models with hourly results, by the annual delivered
share for the bin method of the heat pump).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TOPOLOGIES = ("central", "per_zone", "combined")
WATER_KWH_PER_L_K = 0.001163


# --------------------------------------------------------------------- building blocks
def _series(df: pd.DataFrame, col: str, index) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series(0.0, index=index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).reindex(index).fillna(0.0)


def _distribution(cfg: Optional[dict], q_h, q_c, q_w, t_ext, dt) -> Dict[str, pd.Series]:
    """EN 15316-3 distribution; ``None`` = no distribution losses (ideal)."""
    idx = q_h.index
    zero = pd.Series(0.0, index=idx)
    if not cfg:
        return {"Q_H_dis_in": q_h, "Q_C_dis_in": q_c, "Q_W_dis_in": q_w, "W_dis_aux": zero.copy()}
    from .distribution_15316_3 import DistributionSystemCalculator

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("demand_unit", "kWh")
    ts = DistributionSystemCalculator(cfg).run_timeseries(pd.DataFrame(
        {"T_ext": t_ext, "Q_H_kWh": q_h, "Q_C_kWh": q_c, "Q_W_kWh": q_w, "time_step_hours": dt},
        index=idx)).timeseries
    out = {}
    for svc, q in (("H", q_h), ("C", q_c), ("W", q_w)):
        block = {"H": "heating", "C": "cooling", "W": "dhw"}[svc]
        col = f"Q_{svc}_dis_in_kWh"
        out[f"Q_{svc}_dis_in"] = _series(ts, col, idx) if (block in cfg and col in ts.columns) else q
    aux = zero.copy()
    for svc in ("H", "C", "W"):
        aux = aux + _series(ts, f"W_{svc}_dis_aux_kWh", idx).abs()
    out["W_dis_aux"] = aux
    return out


def _generation(gen: dict, q_h, q_w, q_c, t_ext, dt) -> Dict[str, Any]:
    """Run the generator on the hourly loads; returns served energy and carriers."""
    kind = gen.get("kind", "ideal")
    cfg = copy.deepcopy(gen.get("config", {}))
    idx = q_h.index
    carriers = {"electricity_kWh": 0.0, "natural_gas_kWh": 0.0, "biomass_kWh": 0.0,
                "district_heat_kWh": 0.0, "district_cooling_kWh": 0.0}
    q_req_h, q_req_w, q_req_c = q_h, q_w, q_c
    hourly_served = None                 # hourly served energy, when the model gives it
    if kind == "ideal":
        # optional hourly limits: what exceeds them is unmet
        q_h = np.minimum(q_h, float(cfg.get("max_heating_kW", np.inf)) * dt)
        q_w = np.minimum(q_w, float(cfg.get("max_dhw_kW", np.inf)) * dt)
        q_c = np.minimum(q_c, float(cfg.get("max_cooling_kW", np.inf)) * dt)
        cop_h = float(cfg.get("cop_heating", 1.0))
        cop_w = float(cfg.get("cop_dhw", cop_h))
        eer = float(cfg.get("eer", 3.0))
        carrier = str(cfg.get("carrier", "electricity")) + "_kWh"
        carriers.setdefault(carrier, 0.0)
        carriers[carrier] += float(q_h.sum() / cop_h + q_w.sum() / cop_w)
        carriers["electricity_kWh"] += float(q_c.sum() / eer)
        out_h, out_w, out_c = float(q_h.sum()), float(q_w.sum()), float(q_c.sum())
        summary = {"cop_heating": cop_h, "cop_dhw": cop_w, "eer": eer}
        hourly_served = {"H": pd.Series(q_h, index=idx), "W": pd.Series(q_w, index=idx),
                         "C": pd.Series(q_c, index=idx)}
        q_h, q_w, q_c = q_req_h, q_req_w, q_req_c
    elif kind == "boiler_15316_4_1":
        from .generation_15316_4_1 import BoilerGeneratorCalculator

        if float(q_c.sum()) > 0.5:
            raise ValueError("a boiler cannot serve cooling: use topology per_zone/central with a "
                             "heat pump, or set cooling to zero")
        gr = BoilerGeneratorCalculator(cfg).run_timeseries(pd.DataFrame({
            "Q_H_gen_out_kWh": q_h, "Q_W_gen_out_kWh": q_w, "t_use_h": dt,
            "theta_avg_C": float(cfg.get("theta_avg_C", 45.0)),
            "theta_return_C": float(cfg.get("theta_return_C", 35.0))}, index=idx))
        summary = gr.summary
        out_h, out_w, out_c = summary["QH_gen_out_kWh"], summary["QW_gen_out_kWh"], 0.0
        fuel = "natural_gas_kWh" if str(cfg.get("fuel_type", "natural_gas")).lower() in (
            "natural_gas", "gas", "lpg") else "biomass_kWh"
        carriers[fuel] = float(summary["EHW_gen_in_kWh"])
        carriers["electricity_kWh"] = float(summary["WHW_gen_aux_kWh"])
    elif kind in ("heat_pump", "district"):
        if kind == "heat_pump":
            from .heat_pump_15316_4_2 import HeatPumpSystemCalculator as Calc
        else:
            from .district_15316_4_5 import DistrictEnergySystemCalculator as Calc
        cfg.setdefault("demand_unit", "kWh")
        gr = Calc(cfg).run_timeseries(pd.DataFrame(
            {"T_ext": t_ext, "Q_H_kWh": q_h, "Q_W_kWh": q_w, "Q_C_kWh": q_c, "time_step_hours": dt},
            index=idx))
        summary = gr.summary
        out_h = float(summary.get("QH_gen_out_kWh", 0.0))
        out_w = float(summary.get("QW_gen_out_kWh", 0.0))
        out_c = float(summary.get("QC_gen_out_kWh", 0.0))
        carriers["electricity_kWh"] = float(summary.get("E_total_electricity_kWh", 0.0))
        if kind == "district":
            carriers["district_heat_kWh"] = float(summary.get("EHW_district_in_kWh", 0.0))
            carriers["district_cooling_kWh"] = float(summary.get("EC_district_in_kWh", 0.0))
    else:
        raise ValueError(f"unknown generator kind {kind!r} (ideal, heat_pump, boiler_15316_4_1, district)")
    return {
        "kind": kind,
        "QH_gen_out_kWh": out_h, "QW_gen_out_kWh": out_w, "QC_gen_out_kWh": out_c,
        "QH_gen_unmet_kWh": max(float(q_h.sum()) - out_h, 0.0),
        "QW_gen_unmet_kWh": max(float(q_w.sum()) - out_w, 0.0),
        "QC_gen_unmet_kWh": max(float(q_c.sum()) - out_c, 0.0),
        "carriers": carriers, "summary": summary, "hourly_served": hourly_served,
    }


# ------------------------------------------------------------------ DHW tank dispatch
def dispatch_combined(
    q_h: pd.Series, q_c: pd.Series, q_w_draw: pd.Series, dt: float, storage: dict, dispatch: dict,
) -> pd.DataFrame:
    """Hourly time share of ONE generator between space heating, space cooling and the
    charging of a DHW storage tank (single-volume tank, energy above the cold water).

    The generator serves one service at a time: inside the hour the services share the
    time, in this order:

    1. DHW charging, when the tank is below ``min_temperature_C`` and ``dhw_priority``;
    2. the space services in the order ``space_priority``: ``"heating"`` (default),
       ``"cooling"``, or ``"larger"`` (the larger demand of the hour first).  Heating and
       cooling requested in the same hour (different zones) are both served by time
       share if the capacity allows; what does not fit is unmet and reported as
       displaced by the other service - it never disappears from the balance;
    3. DHW charging with the time left, when the tank is below the reload threshold.

    Inputs are the energies required at the generator outlet [kWh]: ``q_h``/``q_c`` space
    heating/cooling (after distribution), ``q_w_draw`` DHW drawn from the tank (DHW
    distribution included).

    ``storage``: ``volume_l``, ``set_temperature_C`` (default 55), ``min_temperature_C``
    (45), ``cold_water_C`` (10), ``ambient_C`` (20), ``H_sto_ls_W_K`` (default EN 15316-5
    Annex B for the volume), ``initial_fraction`` (1).

    ``dispatch``: ``capacity_heating_kW``, ``capacity_cooling_kW``, ``capacity_dhw_kW``
    (charging power at DHW temperature, default the heating capacity), ``dhw_priority``
    (True), ``space_priority``, ``reload_deadband`` (0.1), ``cooling_dhw``
    (``"time_share"``, or ``"heat_recovery"``: condenser heat recovery charges the tank
    while cooling, up to ``heat_recovery_ratio`` (1.25) times the cooling served, without
    using generator time).

    Unmet space energy is not shifted to the following hours.
    """
    from .storage_15316_5 import _annex_b_loss_coefficient

    vol = float(storage["volume_l"])
    t_set = float(storage.get("set_temperature_C", 55.0))
    t_min = float(storage.get("min_temperature_C", 45.0))
    t_cold = float(storage.get("cold_water_C", 10.0))
    t_amb = float(storage.get("ambient_C", 20.0))
    h_ls = float(storage.get("H_sto_ls_W_K", _annex_b_loss_coefficient("dhw", vol)))
    c = vol * WATER_KWH_PER_L_K                                   # kWh/K
    e_max = c * (t_set - t_cold)
    e_min = c * (t_min - t_cold)
    e = e_max * float(storage.get("initial_fraction", 1.0))

    n_steps = len(q_h)

    def _cap_array(value):
        arr = np.broadcast_to(np.asarray(value, dtype=float), (n_steps,)).astype(float)
        return arr * dt

    cap_h_arr = _cap_array(dispatch.get("capacity_heating_kW", np.inf))
    cap_c_arr = _cap_array(dispatch.get("capacity_cooling_kW", np.inf))
    cap_w_arr = _cap_array(dispatch.get("capacity_dhw_kW", dispatch.get("capacity_heating_kW", np.inf)))
    if not np.all(np.isfinite(cap_w_arr)):
        raise ValueError("dispatch needs capacity_dhw_kW or capacity_heating_kW")
    cap = {"H": 0.0, "C": 0.0}
    priority = bool(dispatch.get("dhw_priority", True))
    space_priority = str(dispatch.get("space_priority", "heating"))
    if space_priority not in ("heating", "cooling", "larger"):
        raise ValueError("space_priority must be 'heating', 'cooling' or 'larger'")
    deadband = float(dispatch.get("reload_deadband", 0.1))
    mode_c = str(dispatch.get("cooling_dhw", "time_share"))
    if mode_c not in ("time_share", "heat_recovery"):
        raise ValueError("cooling_dhw must be 'time_share' or 'heat_recovery'")
    hr_ratio = float(dispatch.get("heat_recovery_ratio", 1.25)) if mode_c == "heat_recovery" else 0.0

    def _space(demand: dict, f: float, order) -> tuple:
        served = {"H": 0.0, "C": 0.0}
        for s in order:
            if demand[s] <= 0.0:
                continue
            if not np.isfinite(cap[s]):
                served[s] = demand[s]
                continue
            if cap[s] <= 0.0:
                continue
            served[s] = min(demand[s], f * cap[s])
            f -= served[s] / cap[s]
        return served, max(f, 0.0)

    n = len(q_h)
    names = ("Q_H_served", "Q_C_served", "Q_W_charge", "Q_W_charge_recovery", "Q_H_unmet",
             "Q_C_unmet", "Q_W_unmet", "Q_sto_ls", "E_sto", "theta_sto", "f_dhw", "f_heating",
             "f_cooling", "priority_active", "Q_H_displaced", "Q_C_displaced",
             "Q_H_displaced_by_cooling", "Q_C_displaced_by_heating", "simultaneous")
    cols = {k: np.zeros(n) for k in names}
    first_service = np.empty(n, dtype=object)
    qh, qc, qw = (np.asarray(s, dtype=float) for s in (q_h, q_c, q_w_draw))
    for t in range(n):
        cap["H"], cap["C"] = cap_h_arr[t], cap_c_arr[t]
        cap_w = max(cap_w_arr[t], 1e-12)
        theta = t_cold + e / c
        loss = max(h_ls * (theta - t_amb), 0.0) * dt / 1000.0
        e_avail = e - loss - qw[t]
        need_full = max(e_max - e_avail, 0.0)
        demand = {"H": max(qh[t], 0.0), "C": max(qc[t], 0.0)}
        if space_priority == "larger":
            order = ("H", "C") if demand["H"] >= demand["C"] else ("C", "H")
        else:
            order = ("H", "C") if space_priority == "heating" else ("C", "H")
        both = demand["H"] > 0.0 and demand["C"] > 0.0
        first_service[t] = (order[0] if demand[order[0]] > 0 else order[1]) if (demand["H"] or demand["C"]) else ""

        # provisional condenser recovery (cooling served as if alone)
        rec0 = min(need_full, hr_ratio * min(demand["C"], cap["C"])) if hr_ratio else 0.0
        prio = priority and (e_avail + rec0) < e_min
        f = 1.0
        charge = 0.0
        if prio:
            charge = min(max(need_full - rec0, 0.0), cap_w)
            f -= charge / cap_w
        served, f = _space(demand, f, order)
        recovery = min(max(need_full - charge, 0.0), hr_ratio * served["C"]) if hr_ratio else 0.0
        if not prio and (e_avail + charge + recovery) < max((1.0 - deadband) * e_max, e_min) + 1e-12:
            extra = min(max(need_full - charge - recovery, 0.0), f * cap_w)
            charge += extra
            f -= extra / cap_w

        # what the space would get without DHW, and each service alone
        no_dhw, _ = _space(demand, 1.0, order)
        alone = {s: (min(demand[s], cap[s]) if np.isfinite(cap[s]) else demand[s]) for s in ("H", "C")}

        e_new = e_avail + charge + recovery
        unmet_w = 0.0
        if e_new < 0.0:
            unmet_w, e_new = -e_new, 0.0
        e = min(e_new, e_max)
        cols["Q_H_served"][t], cols["Q_C_served"][t] = served["H"], served["C"]
        cols["Q_H_unmet"][t] = demand["H"] - served["H"]
        cols["Q_C_unmet"][t] = demand["C"] - served["C"]
        cols["Q_H_displaced"][t] = no_dhw["H"] - served["H"]          # by the DHW charging
        cols["Q_C_displaced"][t] = no_dhw["C"] - served["C"]
        cols["Q_H_displaced_by_cooling"][t] = alone["H"] - no_dhw["H"]  # by the other service
        cols["Q_C_displaced_by_heating"][t] = alone["C"] - no_dhw["C"]
        cols["simultaneous"][t] = float(both)
        cols["Q_W_charge"][t] = charge
        cols["Q_W_charge_recovery"][t] = recovery
        cols["Q_W_unmet"][t] = unmet_w
        cols["Q_sto_ls"][t] = loss
        cols["E_sto"][t] = e
        cols["theta_sto"][t] = t_cold + e / c
        cols["f_dhw"][t] = charge / cap_w
        cols["f_heating"][t] = served["H"] / cap["H"] if np.isfinite(cap["H"]) and cap["H"] > 0 else 0.0
        cols["f_cooling"][t] = served["C"] / cap["C"] if np.isfinite(cap["C"]) and cap["C"] > 0 else 0.0
        cols["priority_active"][t] = float(prio)
    out = pd.DataFrame({f"{k}_kWh" if k.startswith("Q_") or k == "E_sto" else k: v
                        for k, v in cols.items()}, index=q_h.index)
    out["first_service"] = first_service
    return out


# ------------------------------------------------------------------------- top level
@dataclass
class MultizoneSystemResult:
    topology: str
    emission: Any                                  # MultizoneEmissionResult
    hourly: pd.DataFrame                           # building-level system flows [kWh]
    annual: pd.DataFrame                           # one row per circuit (+ TOTAL)
    generators: Dict[str, Any]                     # circuit -> generator result
    dispatch: Optional[pd.DataFrame] = None        # combined topology: tank dispatch
    notes: list = field(default_factory=list)

    def dhw_dispatch_summary(self) -> Dict[str, float]:
        """Combined topology: how the tank charging interacts with heating and cooling."""
        if self.dispatch is None:
            return {}
        d = self.dispatch
        heat = self.hourly["Q_H_gen_req_kWh"] > 0
        cool = self.hourly["Q_C_gen_req_kWh"] > 0
        return {
            "Q_W_charge_kWh": float(d["Q_W_charge_kWh"].sum()),
            "Q_W_charge_recovery_kWh": float(d["Q_W_charge_recovery_kWh"].sum()),
            "Q_sto_ls_kWh": float(d["Q_sto_ls_kWh"].sum()),
            "Q_W_unmet_kWh": float(d["Q_W_unmet_kWh"].sum()),
            "hours_charging": int((d["Q_W_charge_kWh"] > 1e-9).sum()),
            "hours_charging_during_heating": int(((d["Q_W_charge_kWh"] > 1e-9) & heat).sum()),
            "hours_charging_during_cooling": int(((d["Q_W_charge_kWh"] > 1e-9) & cool).sum()),
            "hours_dhw_priority": int(d["priority_active"].sum()),
            "Q_H_displaced_by_dhw_kWh": float(d["Q_H_displaced_kWh"].sum()),
            "Q_C_displaced_by_dhw_kWh": float(d["Q_C_displaced_kWh"].sum()),
            "Q_H_unmet_kWh": float(d["Q_H_unmet_kWh"].sum()),
            "Q_C_unmet_kWh": float(d["Q_C_unmet_kWh"].sum()),
            "theta_sto_min_C": float(d["theta_sto"].min()),
        }


def _zone_emission_inputs(em_res, zone, index):
    if zone not in em_res.emission:
        # zone without emitter: free-floating or ideal (unlisted_zones); its need, if any,
        # reaches the circuit without emission losses
        q = pd.to_numeric(em_res.hourly[f"Q_HVAC_{zone}"], errors="coerce").fillna(0.0)
        dt = float(pd.Series(index).diff().dt.total_seconds().median() / 3600.0)
        return q.clip(lower=0) / 1000 * dt, (-q).clip(lower=0) / 1000 * dt
    ts = em_res.emission[zone].timeseries
    q_h = _series(ts, "Q_H_em_in_kWh", index) + _series(ts, "Q_H_em_back_out_kWh", index)
    q_c = _series(ts, "Q_C_em_in_kWh", index) + _series(ts, "Q_C_em_back_out_kWh", index)
    return q_h, q_c


def _apply_generator_capacities(config: dict, system: dict, emission_by_zone: dict):
    """Copy of ``config`` with the zone capacities limited by the generator power [W]."""
    topology = system["topology"]
    zones = config["building"]["zones"]
    groups = []                          # (zone names, heating kW, cooling kW)
    if topology == "per_zone":
        for zn, spec in (system.get("zones") or {}).items():
            g = (spec.get("generation") or {})
            groups.append(([zn], g.get("capacity_heating_kW"), g.get("capacity_cooling_kW")))
    else:
        src = system.get("dispatch", {}) if topology == "combined" else (system.get("generation") or {})
        groups.append(([str(z["name"]) for z in zones],
                       src.get("capacity_heating_kW"), src.get("capacity_cooling_kW")))
    if not any(h is not None or c is not None for _, h, c in groups):
        return config, {}
    out = copy.deepcopy(config)
    by_name = {str(z["name"]): z for z in out["building"]["zones"]}
    alloc = {}
    allocation = str(system.get("capacity_allocation", "dynamic"))
    if allocation not in ("dynamic", "static"):
        raise ValueError("system.capacity_allocation must be 'dynamic' or 'static'")
    if topology != "per_zone" and allocation == "dynamic":
        # hourly sharing inside the thermal solver, in proportion to the zone requests
        names, cap_h, cap_c = groups[0]
        grp = {"zones": names,
               "space_priority": (system.get("dispatch") or {}).get("space_priority", "heating")}
        if cap_h is not None:
            grp["heating_capacity"] = float(cap_h) * 1000.0
        if cap_c is not None:
            grp["cooling_capacity"] = float(cap_c) * 1000.0
        out["building"]["generator_groups"] = [grp]
        return out, {"generator_group": {k: v for k, v in grp.items() if k != "zones"},
                     "allocation": "dynamic, proportional to the hourly zone requests"}
    for names, cap_h, cap_c in groups:
        names = [n for n in names if n in by_name]
        for svc, cap_kw, key in (("heating", cap_h, "heating_capacity"), ("cooling", cap_c, "cooling_capacity")):
            if cap_kw is None:
                continue
            p_em = [((emission_by_zone.get(n) or {}).get(svc) or {}).get("nominal_power_kW") for n in names]
            if all(p is not None and float(p) > 0 for p in p_em):
                w = np.array([float(p) for p in p_em])
            else:
                w = np.array([float(by_name[n].get("net_floor_area", 1.0) or 1.0) for n in names])
            w = w / w.sum()
            for n, share in zip(names, w):
                cap_w = float(cap_kw) * 1000.0 * float(share)
                cur = by_name[n].get(key)
                by_name[n][key] = cap_w if cur is None else min(float(cur), cap_w)
                alloc.setdefault(n, {})[key] = round(by_name[n][key], 1)
    return out, alloc


def simulate_multizone_system(
    config: dict,
    system: dict,
    emission_by_zone: Optional[dict] = None,
    dhw: Optional[dict] = None,
    couple_generator_capacity: bool = True,
    **emission_kwargs,
) -> MultizoneSystemResult:
    """Multizone building with its heating/cooling(/DHW) system, see the module docstring.

    ``dhw``: demand spec of ``composer.dhw_demand_kwh`` (``daily_volume_l``,
    ``draw_temperature_C``, ``cold_water_temperature_C``) or an hourly series
    ``{"Q_W_kWh": series}``; used by the ``central`` (DHW on the same generator, no tank)
    and ``combined`` (DHW tank) topologies.
    """
    from .composer import dhw_demand_kwh, simulate_multizone_emission

    topology = system.get("topology")
    if topology not in TOPOLOGIES:
        raise ValueError(f"system.topology must be one of {TOPOLOGIES}, got {topology!r}")
    zones = [str(z["name"]) for z in config["building"]["zones"]]
    notes = []
    if topology == "combined":
        if emission_by_zone and "emission" in system:
            raise ValueError("topology 'combined': give either system['emission'] (one terminal "
                             "type for all the zones) or emission_by_zone (terminals per zone)")
        if not emission_by_zone:
            if "emission" not in system:
                raise ValueError("topology 'combined' needs system['emission'] or emission_by_zone")
            emission_by_zone = {zn: copy.deepcopy(system["emission"]) for zn in zones}
            notes.append("combined: the same emission system serves all the zones")
        else:
            notes.append("combined: terminals per zone (emission_by_zone) on one generator")
    if not emission_by_zone:
        raise ValueError("emission_by_zone is required for the central and per_zone topologies")

    if topology == "per_zone" and dhw is not None:
        shares = {zn: float(spec.get("dhw_share", 0.0)) for zn, spec in (system.get("zones") or {}).items()}
        if any(v < 0.0 for v in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"per_zone: the dhw_share of the zones must be >= 0 and sum to 1, got {shares}")

    # Generator capacity in the thermal balance: the zone heating/cooling capacities are
    # limited by the generator power, so a generator shortage changes the zone temperatures
    # (and the adaptive-facade control).  A shared generator is allocated to its zones by
    # the emitter nominal power (or the net floor area when not given): a static share.
    if couple_generator_capacity:
        config, alloc = _apply_generator_capacities(config, system, emission_by_zone)
        if alloc:
            notes.append(f"generator capacity applied to the zones [W]: {alloc}")

    em = simulate_multizone_emission(config, emission_by_zone, **emission_kwargs)
    idx = em.hourly.index
    dt = float(pd.Series(idx).diff().dt.total_seconds().median() / 3600.0)
    t_ext = pd.to_numeric(em.hourly["T_ext"], errors="coerce").reindex(idx)
    per_zone = {zn: _zone_emission_inputs(em, zn, idx) for zn in zones}

    if dhw is None:
        q_w = pd.Series(0.0, index=idx)
    elif "Q_W_kWh" in dhw:
        q_w = pd.Series(dhw["Q_W_kWh"], index=idx, dtype=float)
    else:
        q_w = dhw_demand_kwh(idx, dhw)

    hourly = pd.DataFrame(index=idx)
    hourly["Q_H_em_in_kWh"] = sum(v[0] for v in per_zone.values())
    hourly["Q_C_em_in_kWh"] = sum(v[1] for v in per_zone.values())
    hourly["Q_W_use_kWh"] = q_w
    generators, rows, dispatch = {}, [], None

    def _em_aux(zone_list):
        tot = 0.0
        for zn in zone_list:
            if zn in em.emission:
                sm = em.emission[zn].summary
                tot += float(sm.get("WH_em_aux_kWh", 0.0)) + float(sm.get("WC_em_aux_kWh", 0.0))
        return tot

    def _row(name, dis, gen, q_h_em, q_c_em, q_w_use, zone_list):
        r = {"circuit": name,
             "Q_H_em_in_kWh": float(q_h_em.sum()), "Q_C_em_in_kWh": float(q_c_em.sum()),
             "Q_W_use_kWh": float(q_w_use.sum()),
             "Q_H_dis_in_kWh": float(dis["Q_H_dis_in"].sum()),
             "Q_C_dis_in_kWh": float(dis["Q_C_dis_in"].sum()),
             "Q_W_dis_in_kWh": float(dis["Q_W_dis_in"].sum()),
             "W_dis_aux_kWh": float(dis["W_dis_aux"].sum())}
        r.update({k: v for k, v in gen.items() if k.endswith("_kWh")})
        r.update(gen["carriers"])
        r["W_em_aux_kWh"] = _em_aux(zone_list)
        # one electricity total: generator + distribution auxiliaries + emitter auxiliaries
        r["electricity_total_kWh"] = (float(gen["carriers"]["electricity_kWh"])
                                      + r["W_dis_aux_kWh"] + r["W_em_aux_kWh"])
        return r

    if topology == "central":
        dis = _distribution(system.get("distribution"), hourly["Q_H_em_in_kWh"],
                            hourly["Q_C_em_in_kWh"], q_w, t_ext, dt)
        gen = _generation(system["generation"], dis["Q_H_dis_in"], dis["Q_W_dis_in"],
                          dis["Q_C_dis_in"], t_ext, dt)
        generators["central"] = gen
        rows.append(_row("central", dis, gen, hourly["Q_H_em_in_kWh"], hourly["Q_C_em_in_kWh"], q_w, zones))
        hourly["Q_H_gen_req_kWh"] = dis["Q_H_dis_in"]
        hourly["Q_C_gen_req_kWh"] = dis["Q_C_dis_in"]
        hourly["Q_W_gen_req_kWh"] = dis["Q_W_dis_in"]
    elif topology == "per_zone":
        zsys = system.get("zones") or {}
        missing = [zn for zn in zones if zn in emission_by_zone and zn not in zsys]
        if missing:
            raise ValueError(f"per_zone: system['zones'] has no distribution/generation for {missing}")
        tot = {k: pd.Series(0.0, index=idx) for k in ("H", "C", "W")}
        for zn, spec in zsys.items():
            if zn not in per_zone:
                raise ValueError(f"per_zone: unknown zone {zn!r}")
            q_h_em, q_c_em = per_zone[zn]
            q_w_zn = q_w * float(spec.get("dhw_share", 0.0))
            dis = _distribution(spec.get("distribution"), q_h_em, q_c_em, q_w_zn, t_ext, dt)
            gen = _generation(spec["generation"], dis["Q_H_dis_in"], dis["Q_W_dis_in"],
                              dis["Q_C_dis_in"], t_ext, dt)
            generators[zn] = gen
            rows.append(_row(zn, dis, gen, q_h_em, q_c_em, q_w_zn, [zn]))
            tot["H"] += dis["Q_H_dis_in"]
            tot["C"] += dis["Q_C_dis_in"]
            tot["W"] += dis["Q_W_dis_in"]
        hourly["Q_H_gen_req_kWh"], hourly["Q_C_gen_req_kWh"], hourly["Q_W_gen_req_kWh"] = tot["H"], tot["C"], tot["W"]
    else:  # combined
        if "storage" not in system or "dispatch" not in system:
            raise ValueError("topology 'combined' needs system['storage'] and system['dispatch']")
        dis = _distribution(system.get("distribution"), hourly["Q_H_em_in_kWh"],
                            hourly["Q_C_em_in_kWh"], q_w, t_ext, dt)
        # The dispatch schedules the tank charging before the generator performance is
        # known.  If the generator delivers less than scheduled, the dispatch capacities are
        # scaled by the delivered/scheduled ratio of each service and the dispatch (tank
        # state included) is recalculated, until the generator serves what is scheduled.
        disp_cfg = copy.deepcopy(system["dispatch"])
        disp_cfg.setdefault("capacity_dhw_kW", disp_cfg.get("capacity_heating_kW"))
        iterations = 0
        for iterations in range(1, 31):
            dispatch = dispatch_combined(dis["Q_H_dis_in"], dis["Q_C_dis_in"], dis["Q_W_dis_in"],
                                         dt, system["storage"], disp_cfg)
            # heat recovered from the condenser is not produced by the heating mode
            gen = _generation(system["generation"], dispatch["Q_H_served_kWh"],
                              dispatch["Q_W_charge_kWh"], dispatch["Q_C_served_kWh"], t_ext, dt)
            short = {}
            for key, svc, col in (("capacity_heating_kW", "QH", "Q_H_served_kWh"),
                                  ("capacity_cooling_kW", "QC", "Q_C_served_kWh"),
                                  ("capacity_dhw_kW", "QW", "Q_W_charge_kWh")):
                sched = float(dispatch[col].sum())
                served_h = (gen.get("hourly_served") or {}).get(svc[1])
                if served_h is not None:
                    # hourly model: no hour may exceed what the generator produced
                    if np.any(dispatch[col].to_numpy() > served_h.to_numpy() + 1e-6):
                        short[key] = (sched - gen[f"{svc}_gen_unmet_kWh"]) / max(sched, 1e-12)
                elif sched > 1e-6 and gen[f"{svc}_gen_unmet_kWh"] > max(1e-3 * sched, 0.01):
                    short[key] = (sched - gen[f"{svc}_gen_unmet_kWh"]) / sched
            if not short:
                break
            col_of = {"capacity_heating_kW": ("Q_H_served_kWh", "H"),
                      "capacity_cooling_kW": ("Q_C_served_kWh", "C"),
                      "capacity_dhw_kW": ("Q_W_charge_kWh", "W")}
            for key, ratio in short.items():
                col, svc = col_of[key]
                sched_h = dispatch[col].to_numpy(dtype=float)
                cur = np.broadcast_to(np.asarray(disp_cfg.get(key, np.inf), dtype=float),
                                      sched_h.shape).astype(float)
                cur = np.where(np.isfinite(cur), cur, sched_h.max() / dt)
                served_h = (gen.get("hourly_served") or {}).get(svc)
                if served_h is not None:
                    # hourly: the capacity of each hour becomes what the generator produced
                    sv = served_h.to_numpy(dtype=float)
                    over = sched_h > sv + 1e-9
                    cur = np.where(over, sv / dt, cur)
                else:
                    # annual results only (bin method): scale by the delivered share
                    cur = cur * max(ratio, 0.05)
                disp_cfg[key] = cur
        else:
            notes.append("combined: dispatch and generator not reconciled after 30 iterations")
        if iterations > 1:
            used = {k: (float(np.min(disp_cfg[k])) if np.ndim(disp_cfg.get(k)) else disp_cfg.get(k))
                    for k in ("capacity_heating_kW", "capacity_cooling_kW", "capacity_dhw_kW")}
            notes.append(f"combined: dispatch recalculated {iterations - 1} time(s) on the generator "
                         f"production, capacities used {used}")
        generators["combined"] = gen
        r = _row("combined", dis, gen, hourly["Q_H_em_in_kWh"], hourly["Q_C_em_in_kWh"], q_w, zones)
        r["QH_gen_unmet_kWh"] += float(dispatch["Q_H_unmet_kWh"].sum())
        r["QC_gen_unmet_kWh"] += float(dispatch["Q_C_unmet_kWh"].sum())
        r["QW_gen_unmet_kWh"] += float(dispatch["Q_W_unmet_kWh"].sum())
        r["Q_sto_ls_kWh"] = float(dispatch["Q_sto_ls_kWh"].sum())
        r["dispatch_iterations"] = iterations
        r["Q_H_displaced_by_cooling_kWh"] = float(dispatch["Q_H_displaced_by_cooling_kWh"].sum())
        r["Q_C_displaced_by_heating_kWh"] = float(dispatch["Q_C_displaced_by_heating_kWh"].sum())
        rows.append(r)
        hourly["Q_H_gen_req_kWh"] = dis["Q_H_dis_in"]
        hourly["Q_C_gen_req_kWh"] = dis["Q_C_dis_in"]
        hourly["Q_W_gen_req_kWh"] = dis["Q_W_dis_in"]
        hourly = pd.concat([hourly, dispatch.add_prefix("dispatch_")], axis=1)
        notes.append("combined: one service at a time per hour (time share); DHW tank charged "
                     "with the spare capacity, first when below the minimum temperature")

    annual = pd.DataFrame(rows)
    if len(annual) > 1:
        tot = {"circuit": "TOTAL"}
        for col in annual.columns:
            if col != "circuit" and np.issubdtype(annual[col].dtype, np.number):
                tot[col] = float(annual[col].sum())
        annual = pd.concat([annual, pd.DataFrame([tot])], ignore_index=True)
    return MultizoneSystemResult(topology=topology, emission=em, hourly=hourly, annual=annual,
                                 generators=generators, dispatch=dispatch, notes=notes)
