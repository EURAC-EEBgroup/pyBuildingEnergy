"""Thermal comfort and capacity-shortage indicators per zone (EN 16798-1:2019, Annex B).

Two methods for the operative temperature:

* ``"adaptive"`` (B.2.2, buildings without mechanical cooling, or for the free-running
  periods): comfort temperature theta_c = 0.33 theta_rm + 18.8, with the running mean of
  the daily mean outdoor temperature theta_rm = (1 - alpha) theta_ed-1 + alpha theta_rm-1,
  alpha = 0.8.  Upper limits theta_c + 2/3/4 K and lower limits theta_c - 3/4/5 K for the
  categories I/II/III.  Outside 10 C <= theta_rm <= 30 C (upper) and 15 C <= theta_rm <= 30 C
  (lower) the limits are not defined; those hours are counted as ``not_applicable``.
* ``"static"`` (B.2.1, Table B.1, mechanically heated and cooled buildings): fixed ranges of
  the operative temperature for the heating and cooling seasons.  The defaults are those
  for residential and office spaces (sedentary activity); pass ``static_limits`` for others.

Each occupied hour is assigned to the best category it satisfies (I, II, III, else IV).
The degree-hours outside category II are also given.

The capacity-shortage indicators use the multizone output: ``cap_limited_<zone>`` (the
emitter or HVAC capacity limited the power in that hour), ``Q_HVAC_req_<zone>`` (ideal
request at the set-point) and ``T_set_H/C_<zone>`` (set-points active in the hour).  With
the capacity in the thermal balance, a shortage shows as a temperature off the set-point
and as the energy not delivered (request - delivered); both are reported.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

CATEGORIES = ("I", "II", "III")
ADAPTIVE_UPPER_K = {"I": 2.0, "II": 3.0, "III": 4.0}
ADAPTIVE_LOWER_K = {"I": 3.0, "II": 4.0, "III": 5.0}
RUNNING_MEAN_ALPHA = 0.8

# Table B.1 (residential living spaces and offices, sedentary): (heating min, cooling max) degC
STATIC_LIMITS_DEFAULT = {
    "I": (21.0, 25.5),
    "II": (20.0, 26.0),
    "III": (19.0, 27.0),
}


def running_mean_outdoor_temperature(t_ext: pd.Series, alpha: float = RUNNING_MEAN_ALPHA) -> pd.Series:
    """Hourly series of theta_rm (EN 16798-1 B.2.2, Formula B.1) from hourly outdoor data.

    theta_rm of a day uses the daily means of the previous days; the first day starts
    from its own daily mean.
    """
    if not isinstance(t_ext.index, pd.DatetimeIndex):
        raise ValueError("t_ext needs a DatetimeIndex")
    daily = t_ext.astype(float).groupby(t_ext.index.normalize()).mean()
    d = daily.to_numpy(dtype=float)
    rm = np.empty(len(d))
    rm[0] = d[0]
    for i in range(1, len(d)):
        rm[i] = (1.0 - alpha) * d[i - 1] + alpha * rm[i - 1]     # Formula B.1
    rm_daily = pd.Series(rm, index=daily.index)
    return rm_daily.reindex(t_ext.index.normalize()).set_axis(t_ext.index)


def _occupied_mask(n: int, occupancy) -> np.ndarray:
    if occupancy is None:
        return np.ones(n, dtype=bool)
    occ = np.asarray(occupancy, dtype=float).reshape(-1)[:n]
    if len(occ) < n:
        occ = np.pad(occ, (0, n - len(occ)))
    return occ > 0.0


def comfort_indicators(
    hourly: pd.DataFrame,
    zone: str,
    method: str = "adaptive",
    occupancy=None,
    heating_season: Optional[Iterable[bool]] = None,
    static_limits: Optional[Dict[str, tuple]] = None,
    time_step_h: float = 1.0,
) -> Dict[str, float]:
    """Comfort category of each occupied hour and degree-hours outside category II.

    ``heating_season`` (static method): hourly booleans; by default an hour belongs to the
    heating season when the daily mean outdoor temperature is below 15 C.
    """
    t_op = pd.to_numeric(hourly[f"T_op_{zone}"], errors="coerce").to_numpy(dtype=float)
    n = len(t_op)
    occ = _occupied_mask(n, occupancy) & np.isfinite(t_op)
    t_ext = pd.to_numeric(hourly["T_ext"], errors="coerce")

    upper = {c: np.full(n, np.inf) for c in CATEGORIES}
    lower = {c: np.full(n, -np.inf) for c in CATEGORIES}
    applicable = np.ones(n, dtype=bool)
    if method == "adaptive":
        rm = running_mean_outdoor_temperature(t_ext).to_numpy(dtype=float)
        t_c = 0.33 * rm + 18.8
        up_ok = (rm >= 10.0) & (rm <= 30.0)
        lo_ok = (rm >= 15.0) & (rm <= 30.0)
        for c in CATEGORIES:
            upper[c] = np.where(up_ok, t_c + ADAPTIVE_UPPER_K[c], np.inf)
            lower[c] = np.where(lo_ok, t_c - ADAPTIVE_LOWER_K[c], -np.inf)
        applicable = up_ok | lo_ok
    elif method == "static":
        limits = static_limits or STATIC_LIMITS_DEFAULT
        if heating_season is None:
            daily = t_ext.groupby(t_ext.index.normalize()).transform("mean").to_numpy(dtype=float)
            heating = daily < 15.0
        else:
            heating = np.asarray(list(heating_season), dtype=bool)[:n]
        for c in CATEGORIES:
            h_min, c_max = limits[c]
            lower[c] = np.where(heating, h_min, -np.inf)
            upper[c] = np.where(heating, np.inf, c_max)
    else:
        raise ValueError("method must be 'adaptive' or 'static'")

    cat = np.full(n, "IV", dtype=object)
    for c in reversed(CATEGORIES):              # best category last so it wins
        ok = (t_op <= upper[c]) & (t_op >= lower[c])
        cat[ok] = c
    counted = occ & applicable
    n_occ = int(counted.sum())
    out: Dict[str, float] = {"method": method, "occupied_hours": n_occ,
                             "not_applicable_hours": int((occ & ~applicable).sum())}
    for c in CATEGORIES + ("IV",):
        h = int((counted & (cat == c)).sum())
        out[f"hours_cat_{c}"] = h
        out[f"pct_cat_{c}"] = 100.0 * h / n_occ if n_occ else float("nan")
    over = np.where(counted & np.isfinite(upper["II"]), np.maximum(t_op - upper["II"], 0.0), 0.0)
    under = np.where(counted & np.isfinite(lower["II"]), np.maximum(lower["II"] - t_op, 0.0), 0.0)
    out["Kh_above_cat_II"] = float(over.sum() * time_step_h)
    out["Kh_below_cat_II"] = float(under.sum() * time_step_h)
    return out


def capacity_shortage_indicators(
    hourly: pd.DataFrame, zone: str, tolerance_k: float = 0.1, time_step_h: float = 1.0
) -> Dict[str, float]:
    """Hours, degree-hours and energy of the capacity shortage of one zone.

    * ``hours_limited``: hours in which the capacity limited the heating/cooling power;
    * ``Kh_below_heating_setpoint`` / ``Kh_above_cooling_setpoint``: operative temperature
      off the active set-point (beyond ``tolerance_k``) in those hours;
    * ``Q_H_not_delivered_kWh`` / ``Q_C_not_delivered_kWh``: ideal request at the set-point
      minus the power delivered, in those hours (the theoretical missing demand of the hour;
      it does not include the effect of the lower temperature on the following hours).
    """
    needed = [f"cap_limited_{zone}", f"Q_HVAC_req_{zone}", f"T_set_H_{zone}", f"T_set_C_{zone}"]
    missing = [c for c in needed if c not in hourly.columns]
    if missing:
        raise KeyError(f"multizone output without {missing}: rerun with the current engine")
    lim = pd.to_numeric(hourly[f"cap_limited_{zone}"], errors="coerce").fillna(0).to_numpy() > 0
    q_req = pd.to_numeric(hourly[f"Q_HVAC_req_{zone}"], errors="coerce").fillna(0.0).to_numpy()
    q = pd.to_numeric(hourly[f"Q_HVAC_{zone}"], errors="coerce").fillna(0.0).to_numpy()
    t_op = pd.to_numeric(hourly[f"T_op_{zone}"], errors="coerce").to_numpy(dtype=float)
    th = pd.to_numeric(hourly[f"T_set_H_{zone}"], errors="coerce").to_numpy(dtype=float)
    tc = pd.to_numeric(hourly[f"T_set_C_{zone}"], errors="coerce").to_numpy(dtype=float)
    heat = lim & (q_req > 0.0)
    cool = lim & (q_req < 0.0)
    below = np.where(heat, np.maximum(th - tolerance_k - t_op, 0.0), 0.0)
    above = np.where(cool, np.maximum(t_op - tc - tolerance_k, 0.0), 0.0)
    return {
        "hours_limited": int(lim.sum()),
        "hours_limited_heating": int(heat.sum()),
        "hours_limited_cooling": int(cool.sum()),
        "Kh_below_heating_setpoint": float(np.nansum(below) * time_step_h),
        "Kh_above_cooling_setpoint": float(np.nansum(above) * time_step_h),
        "Q_H_not_delivered_kWh": float(np.sum(np.where(heat, q_req - q, 0.0)) * time_step_h / 1000.0),
        "Q_C_not_delivered_kWh": float(np.sum(np.where(cool, q - q_req, 0.0)) * time_step_h / 1000.0),
    }


def multizone_comfort_table(
    hourly: pd.DataFrame,
    zones: Iterable[str],
    method: str = "adaptive",
    occupancy_by_zone: Optional[Dict[str, Iterable[float]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """One row per zone: comfort categories and capacity-shortage indicators."""
    rows = []
    for zn in zones:
        row = {"zone": zn}
        row.update(comfort_indicators(
            hourly, zn, method=method,
            occupancy=(occupancy_by_zone or {}).get(zn), **kwargs))
        if f"cap_limited_{zn}" in hourly.columns:
            row.update(capacity_shortage_indicators(hourly, zn))
        rows.append(row)
    return pd.DataFrame(rows).set_index("zone")
