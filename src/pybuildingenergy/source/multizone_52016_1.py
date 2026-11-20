"""Multi-zone support for EN ISO 52016-1:2017 (6.4.2, 6.4.6, 6.4.7, Annex D).

* :func:`parse_interzone_airflows` - Annex D.2 (1) / D.4: air flow between thermal zones
* :func:`validate_multizone_input` - structural checks of a multizone building object
* :func:`check_zoning_rules` - the quantitative zoning criteria of 6.4.2 (steps 2, 3, 5, 9, 10)
* :func:`spatial_setpoint_reduction` - 6.4.6 Option B (Formulae 7-8)
* :func:`annual_with_total` - building total row for the per-zone annual results
* :func:`multizone_report` / :func:`format_multizone_report` - description of the partitioning
  and of the input data, required by Annex D ("shall be described in the report")

Air flow between zones (``building_object["interzone_airflows"]``)
------------------------------------------------------------------
Each entry describes the *net* air flow from zone ``from`` to zone ``to`` (D.4)::

    {"from": "Z_ground", "to": "Z_first", "q_v_m3_h": 120.0}      # or q_v_m3_s / ach_from_zone
    {"from": "A", "to": "B", "q_v_m3_h": 80.0, "makeup_from_outdoor": true,
     "follow_ventilation_profile": true}

``H_ve,z->y = rho_a c_a q_V,z->y`` (D.2).  The receiving zone ``y`` sees air at the
temperature of the source zone ``z`` (Annex D.2 1).  The air leaving ``z`` is replaced by
the zone's own ventilation inputs, unless ``makeup_from_outdoor`` is true, in which case an
equal outdoor air flow is added to ``z``.  An air flow in both directions between two zones
is not allowed: the standard treats such zones as a single zone (D.2, NOTE 1).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RHO_AIR = 1.204          # kg/m3, as in the ventilation boundary
C_AIR = 1006.0           # J/(kg K)
RHO_C_AIR = RHO_AIR * C_AIR

# 6.4.2 thresholds
SMALL_ZONE_FRACTION = 0.05          # step 9
VERY_SMALL_ZONE_FRACTION = 0.01     # step 10
SIMILAR_SETPOINT_DIFF_K = 4.0       # step 5
SIMILAR_OPERATION_HOURS = 3.0       # step 5
LARGE_OPENING_RATIO = 0.003         # m2 per m2 of useful floor area (steps 2 and 3)
HIGH_VENTILATION_M3_H_PER_M2 = 3.0 * 3.6   # 3 dm3/s per m2 (step 2)


@dataclass(frozen=True)
class Issue:
    """A finding of the input validation."""

    level: str      # "ERROR" | "WARN" | "INFO"
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.path}: {self.message}"


# ----------------------------------------------------------------------------- helpers


def _zones(building_object: dict) -> List[dict]:
    zones = building_object.get("zones") or []
    return list(zones)


def _zone_names(building_object: dict) -> List[str]:
    return [str(z.get("name")) for z in _zones(building_object)]


def _surface_boundary(surface: dict) -> str:
    return str(surface.get("boundary", "OUTDOORS")).strip().upper()


# ------------------------------------------------------------------ inter-zone air flows


def parse_interzone_airflows(building_object: dict) -> List[Dict[str, Any]]:
    """Normalised list of inter-zone air flows (Annex D.2 1, D.4).

    Each result has ``from``, ``to``, ``q_v_m3_s`` (base value), ``makeup_from_outdoor`` and
    ``follow_ventilation_profile``.  Raises ``ValueError`` for invalid input, including an
    air flow in both directions between the same two zones.
    """
    raw = building_object.get("interzone_airflows") or []
    if not raw:
        return []
    zones = {str(z.get("name")): z for z in _zones(building_object)}
    if not zones:
        raise ValueError("interzone_airflows requires a multizone building object (zones)")
    flows: List[Dict[str, Any]] = []
    pairs = set()
    for k, f in enumerate(raw):
        path = f"interzone_airflows[{k}]"
        src, dst = f.get("from"), f.get("to")
        if src not in zones or dst not in zones:
            raise ValueError(f"{path}: 'from' and 'to' must be zone names, got {src!r} -> {dst!r}")
        if src == dst:
            raise ValueError(f"{path}: 'from' and 'to' are the same zone ({src!r})")
        if (dst, src) in pairs:
            raise ValueError(
                f"{path}: air flow in both directions between {src!r} and {dst!r}. "
                "ISO 52016-1 Annex D.2: zones exchanging air in both directions are considered "
                "a single zone; merge them (6.4.2) or give the net flow in one direction only."
            )
        if (src, dst) in pairs:
            raise ValueError(f"{path}: duplicate air flow {src!r} -> {dst!r}; sum the flows")
        pairs.add((src, dst))
        if f.get("q_v_m3_s") is not None:
            q = float(f["q_v_m3_s"])
        elif f.get("q_v_m3_h") is not None:
            q = float(f["q_v_m3_h"]) / 3600.0
        elif f.get("ach_from_zone") is not None:
            vol = float(zones[src].get("volume") or 0.0)
            if vol <= 0.0:
                raise ValueError(f"{path}: ach_from_zone needs a positive volume for zone {src!r}")
            q = float(f["ach_from_zone"]) * vol / 3600.0
        else:
            raise ValueError(f"{path}: give q_v_m3_s, q_v_m3_h or ach_from_zone")
        if not math.isfinite(q) or q < 0.0:
            raise ValueError(f"{path}: the air flow must be finite and >= 0, got {q}")
        flows.append(
            {
                "from": str(src),
                "to": str(dst),
                "q_v_m3_s": q,
                "makeup_from_outdoor": bool(f.get("makeup_from_outdoor", False)),
                "follow_ventilation_profile": bool(f.get("follow_ventilation_profile", False)),
            }
        )
    return flows


def interzone_coupling(
    flows: Sequence[Dict[str, Any]],
    zone_index: Dict[str, int],
    n_zones: int,
    t_out: float,
    profile_value=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Air-node coupling terms of the inter-zone flows for one time step.

    Returns ``(Hmat, s)`` such that the heat balance of zone ``i`` gets
    ``+ sum_j Hmat[i, j] * theta_air[j] - s[i]`` on the left-hand side (same convention as
    the ventilation term ``H_ve * theta - S_ve``).

    Receiving zone ``y`` of a flow ``z -> y``:  ``Hmat[y, y] += H`` and ``Hmat[y, z] -= H``.
    Source zone with ``makeup_from_outdoor``:  ``Hmat[z, z] += H`` and ``s[z] += H * t_out``.
    """
    hmat = np.zeros((n_zones, n_zones), dtype=float)
    s = np.zeros(n_zones, dtype=float)
    for f in flows:
        z, y = zone_index[f["from"]], zone_index[f["to"]]
        mult = 1.0
        if f["follow_ventilation_profile"] and profile_value is not None:
            mult = float(profile_value(f["from"], "ventilation_profile"))
        h = RHO_C_AIR * f["q_v_m3_s"] * mult
        hmat[y, y] += h
        hmat[y, z] -= h
        if f["makeup_from_outdoor"]:
            hmat[z, z] += h
            s[z] += h * float(t_out)
    return hmat, s


# ------------------------------------------------------------------------- validation


def _operation_hours(zone: dict, key: str) -> Optional[float]:
    prof = zone.get(key)
    if not isinstance(prof, dict):
        return None
    wd = prof.get("weekday")
    if wd is None:
        return None
    try:
        return float(sum(1 for v in wd if float(v) > 0.0))
    except (TypeError, ValueError):
        return None


def validate_multizone_input(building_object: dict) -> List[Issue]:
    """Structural validation of a multizone building object (before the calculation)."""
    issues: List[Issue] = []
    zones = _zones(building_object)
    if not zones:
        issues.append(Issue("ERROR", "zones", "no zones defined"))
        return issues

    names = [str(z.get("name")) for z in zones]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        issues.append(Issue("ERROR", "zones", f"duplicate zone names: {dup}"))
    for i, z in enumerate(zones):
        p = f"zones[{i}]"
        if not z.get("name"):
            issues.append(Issue("ERROR", p, "zone without name"))
        area = z.get("net_floor_area")
        if area is None:
            issues.append(Issue("WARN", p, "net_floor_area missing: the building floor area is used"))
        elif not (float(area) > 0.0):
            issues.append(Issue("ERROR", p, "net_floor_area must be positive"))
        vol = z.get("volume")
        if vol is not None and not (float(vol) > 0.0):
            issues.append(Issue("ERROR", p, "volume must be positive"))
        th, tc = z.get("heating_setpoint"), z.get("cooling_setpoint")
        if (th is not None and tc is not None and float(th) > -990.0 and float(tc) < 990.0
                and float(th) > float(tc)):
            issues.append(
                Issue("ERROR", p, f"heating_setpoint ({th}) is above cooling_setpoint ({tc})")
            )
        for cap in ("heating_capacity", "cooling_capacity"):
            if z.get(cap) is not None and float(z[cap]) < 0.0:
                issues.append(Issue("ERROR", p, f"{cap} must be >= 0"))

    name_set = set(names)
    surfaces = building_object.get("building_surface") or []
    touched = {n: 0 for n in names}
    internal_keys: Dict[Tuple[str, str, str], int] = {}
    for i, s in enumerate(surfaces):
        p = f"building_surface[{i}] ({s.get('name')})"
        zone = s.get("zone", names[0])
        if zone not in name_set:
            issues.append(Issue("ERROR", p, f"unknown zone {zone!r}"))
            continue
        touched[zone] += 1
        if _surface_boundary(s) == "INTERNAL":
            adj = s.get("adjacent_zone")
            if adj is None or adj not in name_set:
                issues.append(Issue("ERROR", p, f"INTERNAL surface with unknown adjacent_zone {adj!r}"))
                continue
            if adj == zone:
                issues.append(Issue("ERROR", p, "adjacent_zone equals the zone of the surface"))
                continue
            touched[adj] += 1
            key = (str(s.get("name")), *sorted((zone, adj)))
            internal_keys[key] = internal_keys.get(key, 0) + 1
    for key, n in internal_keys.items():
        if n > 1:
            issues.append(
                Issue("ERROR", "building_surface",
                      f"partition {key[0]!r} between {key[1]!r} and {key[2]!r} is defined {n} times; "
                      "define each INTERNAL partition once (it is modelled with both faces)")
            )
    for n, c in touched.items():
        if c == 0:
            issues.append(Issue("WARN", f"zones[{n}]", "no building surface is attached to this zone"))

    total_area = sum(float(z.get("net_floor_area") or 0.0) for z in zones)
    bld_area = (building_object.get("building") or {}).get("net_floor_area")
    if bld_area and total_area > 0.0 and abs(total_area - float(bld_area)) / float(bld_area) > 0.02:
        issues.append(
            Issue("WARN", "zones",
                  f"sum of the zone floor areas ({total_area:.1f} m2) differs from "
                  f"building.net_floor_area ({float(bld_area):.1f} m2) by more than 2 %")
        )

    try:
        parse_interzone_airflows(building_object)
    except ValueError as exc:
        issues.append(Issue("ERROR", "interzone_airflows", str(exc)))
    return issues


def raise_on_errors(issues: Iterable[Issue]) -> None:
    errors = [i for i in issues if i.level == "ERROR"]
    if errors:
        raise ValueError("invalid multizone input:\n  " + "\n  ".join(str(e) for e in errors))


# ------------------------------------------------------------------- zoning rules (6.4.2)


def _adjacent_pairs(building_object: dict) -> Dict[Tuple[str, str], float]:
    """Zone pairs sharing an INTERNAL partition -> shared area."""
    out: Dict[Tuple[str, str], float] = {}
    for s in building_object.get("building_surface") or []:
        if _surface_boundary(s) != "INTERNAL":
            continue
        a, b = s.get("zone"), s.get("adjacent_zone")
        if a is None or b is None or a == b:
            continue
        key = tuple(sorted((str(a), str(b))))
        out[key] = out.get(key, 0.0) + float(s.get("area", 0.0) or 0.0)
    return out


def check_zoning_rules(building_object: dict) -> List[Issue]:
    """Quantitative zoning criteria of ISO 52016-1, 6.4.2 (advisory, never raises).

    * step 2: a highly ventilated zone (>= 3 dm3/s per m2) is thermally unconditioned
    * step 3: permanent large openings (>= 0.003 m2 per m2) -> one thermal zone
    * step 5: adjacent zones with similar conditions of use (settings within 4 K,
      daily operation periods within 3 h) may be combined
    * steps 9, 10: zones below 5 % / 1 % of the total useful floor area may be combined
    """
    issues: List[Issue] = []
    zones = _zones(building_object)
    if len(zones) < 2:
        return issues
    by_name = {str(z["name"]): z for z in zones}
    total = sum(float(z.get("net_floor_area") or 0.0) for z in zones)

    if total > 0.0:
        for z in zones:
            frac = float(z.get("net_floor_area") or 0.0) / total
            if frac < VERY_SMALL_ZONE_FRACTION:
                issues.append(Issue("WARN", f"zones[{z['name']}]",
                                    f"useful floor area is {100 * frac:.2f} % of the total (< 1 %): "
                                    "6.4.2.11 (step 10) allows recombining it with an adjacent zone"))
            elif frac < SMALL_ZONE_FRACTION:
                issues.append(Issue("WARN", f"zones[{z['name']}]",
                                    f"useful floor area is {100 * frac:.1f} % of the total (< 5 %): "
                                    "6.4.2.10 (step 9) allows recombining it with an adjacent zone"))

    for z in zones:
        area = float(z.get("net_floor_area") or 0.0)
        vent = (z.get("ventilation") or {}) if isinstance(z.get("ventilation"), dict) else {}
        comps = vent.get("components") or []
        vol = float(z.get("volume") or 0.0)
        if area > 0.0 and vol > 0.0 and comps:
            q = sum(float(c.get("air_changes_per_hour", 0.0)) * vol
                    for c in comps if str(c.get("ventilation_type")) == "constant_ach")
            if q / area >= HIGH_VENTILATION_M3_H_PER_M2:
                issues.append(Issue("WARN", f"zones[{z['name']}]",
                                    f"permanent ventilation of {q / area:.1f} m3/h per m2 is >= 3 dm3/s "
                                    "per m2: 6.4.2.2 treats such a space as thermally unconditioned"))

    pairs = _adjacent_pairs(building_object)
    for (a, b), shared in pairs.items():
        za, zb = by_name[a], by_name[b]
        dh = abs(float(za.get("heating_setpoint", 0.0)) - float(zb.get("heating_setpoint", 0.0)))
        dc = abs(float(za.get("cooling_setpoint", 0.0)) - float(zb.get("cooling_setpoint", 0.0)))
        ha, hb = _operation_hours(za, "heating_profile"), _operation_hours(zb, "heating_profile")
        hours_ok = ha is None or hb is None or abs(ha - hb) <= SIMILAR_OPERATION_HOURS
        if dh < SIMILAR_SETPOINT_DIFF_K and dc < SIMILAR_SETPOINT_DIFF_K and hours_ok:
            issues.append(Issue("INFO", f"zones[{a}, {b}]",
                                "adjacent zones with similar thermal conditions of use "
                                f"(set-point differences {dh:.1f} K / {dc:.1f} K, operation periods "
                                "within 3 h): 6.4.2.6 (step 5) allows combining them. A coupled "
                                "calculation is only to be used with care (Annex D, Table B.7)"))

    for k, op in enumerate(building_object.get("zone_openings") or []):
        a, b = (op.get("zones") or [None, None])[:2]
        if a not in by_name or b not in by_name:
            issues.append(Issue("ERROR", f"zone_openings[{k}]", f"unknown zones {a!r}, {b!r}"))
            continue
        area = float(op.get("area_m2", 0.0))
        smaller = min(float(by_name[a].get("net_floor_area") or 0.0),
                      float(by_name[b].get("net_floor_area") or 0.0))
        if smaller > 0.0 and area >= LARGE_OPENING_RATIO * smaller:
            issues.append(Issue("WARN", f"zone_openings[{k}]",
                                f"permanent opening of {area:.2f} m2 between {a!r} and {b!r} is "
                                f">= 0.003 m2 per m2 of useful floor area: 6.4.2.4 (step 3) requires "
                                "combining the two spaces into one thermal zone (this includes a "
                                "sunspace with a permanent opening, ISO/TR 52016-2, 6.4.2.2)"))
    return issues


def emit_zoning_warnings(issues: Iterable[Issue], stacklevel: int = 3) -> None:
    for i in issues:
        if i.level in ("WARN", "ERROR"):
            warnings.warn(str(i), stacklevel=stacklevel)


# --------------------------------------------------------------- 6.4.6 Option B (7)-(8)


def spatial_setpoint_reduction(
    h_tr: float,
    h_ve: float,
    a_use: float,
    theta_set_h: float,
    theta_e: float,
    f_mod_t: float = 0.8,
    f_mod_sp: float = 0.5,
    h_int_spec: float = 2.0,
) -> float:
    """Decrease of the heating set-point for moderately conditioned spaces (Formulae 7-8).

    ``H_e,spec = (H_tr + H_ve) / A_use`` (8) in W/(m2 K) and::

        d_theta = (f_mod,t f_mod,sp) (f_mod,sp H_e,spec) (theta_set - theta_e)
                  / (f_mod,sp H_e,spec + H_int,spec)                          (7)

    Defaults are those of Table B.6 (f_mod,t = 0.8, f_mod,sp = 0.5, H_int,spec = 2.0 W/(m2 K)).
    The result is never negative: when the outdoor temperature exceeds the set-point no
    reduction applies.
    """
    if a_use <= 0.0:
        raise ValueError("a_use must be positive")
    for name, v in (("f_mod_t", f_mod_t), ("f_mod_sp", f_mod_sp)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if h_int_spec < 0.0:
        raise ValueError("h_int_spec must be >= 0")
    h_e_spec = (float(h_tr) + float(h_ve)) / float(a_use)
    num = (f_mod_t * f_mod_sp) * (f_mod_sp * h_e_spec) * (float(theta_set_h) - float(theta_e))
    den = f_mod_sp * h_e_spec + h_int_spec
    if den <= 0.0:
        return 0.0
    return max(0.0, num / den)


# ---------------------------------------------------------------------- annual total


def annual_with_total(annual: pd.DataFrame, zones: Optional[Sequence[dict]] = None) -> pd.DataFrame:
    """Append a ``zone == "TOTAL"`` row (sum of the zones; specific values over the total area)."""
    if annual is None or len(annual) == 0 or "zone" not in annual.columns:
        return annual
    body = annual[annual["zone"] != "TOTAL"].copy()
    tot: Dict[str, Any] = {"zone": "TOTAL"}
    area_by_zone = {}
    if zones:
        area_by_zone = {str(z["name"]): float(z.get("net_floor_area") or 0.0) for z in zones}
    total_area = sum(area_by_zone.get(str(n), 0.0) for n in body["zone"])
    for col in body.columns:
        if col == "zone":
            continue
        if col == "time_step_h":
            tot[col] = body[col].iloc[0]
        elif "per_sqm" in col:
            base = col.replace("_per_sqm", "")
            if base in body.columns and total_area > 0.0:
                tot[col] = float(body[base].sum()) / total_area
            else:
                tot[col] = np.nan
        elif pd.api.types.is_numeric_dtype(body[col]):
            tot[col] = float(body[col].sum())
        else:
            tot[col] = np.nan
    return pd.concat([body, pd.DataFrame([tot])], ignore_index=True)


# --------------------------------------------------------------------------- report


COUPLING_SCHEME = (
    "Thermal coupling between zones is solved simultaneously: all zone air and surface nodes "
    "of the building form one system of equations per hour. Annex D.2 prescribes that the "
    "boundary condition of each adjacent zone is the temperature of the previous hour (with "
    "iteration and relaxation if the coupling is strong). ISO/TR 52016-2, 6.3.6, names two "
    "ways to handle outputs that feed inputs: use the previous interval (simplest, may "
    "oscillate) or repeat the calculation within the interval; the simultaneous solution is "
    "the second, has no lag and no oscillation, and is a deliberate deviation from the "
    "letter of D.2."
)

PARTITION_CONDUCTANCE_NOTE = (
    "H_tr,zy = U * A of the INTERNAL partitions is referred to the zone operative temperatures; "
    "with the air temperatures the apparent conductance is lower."
)


def multizone_report(
    building_object: dict,
    hourly: Optional[pd.DataFrame] = None,
    annual: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Description of the partitioning into zones and of the input data (Annex D, D.2/D.3)."""
    zones = _zones(building_object)
    flows = parse_interzone_airflows(building_object)
    zones_tbl = [
        {
            "zone": z.get("name"),
            "net_floor_area_m2": z.get("net_floor_area"),
            "volume_m3": z.get("volume"),
            "heating_setpoint_C": z.get("heating_setpoint"),
            "cooling_setpoint_C": z.get("cooling_setpoint"),
            "heating_capacity_W": z.get("heating_capacity"),
            "cooling_capacity_W": z.get("cooling_capacity"),
            "free_floating": bool(z.get("free_floating", False)),
        }
        for z in zones
    ]
    partitions = []
    pair_h: Dict[Tuple[str, str], float] = {}
    for s in building_object.get("building_surface") or []:
        if _surface_boundary(s) != "INTERNAL":
            continue
        a, b = str(s.get("zone")), str(s.get("adjacent_zone"))
        area, u = float(s.get("area", 0.0) or 0.0), float(s.get("u_value", 0.0) or 0.0)
        partitions.append({"surface": s.get("name"), "zone_a": a, "zone_b": b,
                           "area_m2": area, "u_value": u, "H_tr_zy_W_K": u * area})
        key = tuple(sorted((a, b)))
        pair_h[key] = pair_h.get(key, 0.0) + u * area
    flows_tbl = [
        {"from": f["from"], "to": f["to"], "q_v_m3_h": f["q_v_m3_s"] * 3600.0,
         "H_ve_W_K": RHO_C_AIR * f["q_v_m3_s"], "makeup_from_outdoor": f["makeup_from_outdoor"]}
        for f in flows
    ]
    exchange = []
    if hourly is not None:
        for p in partitions:
            ca = f"Q_opaque_inside_surface_{_token(p['surface'])}_to_{p['zone_a']}"
            cb = f"Q_opaque_inside_surface_{_token(p['surface'])}_to_{p['zone_b']}"
            if ca in hourly.columns and cb in hourly.columns:
                dt_h = _dt_hours(hourly)
                qa, qb = hourly[ca].to_numpy(float), hourly[cb].to_numpy(float)
                exchange.append({
                    "surface": p["surface"],
                    "to_zone_a_kWh": float(qa.sum() * dt_h / 1000.0),
                    "to_zone_b_kWh": float(qb.sum() * dt_h / 1000.0),
                    "closure_kWh": float((qa + qb).sum() * dt_h / 1000.0),
                })
    ztu = []
    try:
        from .unconditioned_zone_52016_1 import UnconditionedZoneSet

        ztu = UnconditionedZoneSet(building_object).describe()
    except Exception:
        ztu = []
    issues = validate_multizone_input(building_object) + check_zoning_rules(building_object)
    return {
        "standard": "EN ISO 52016-1:2017, Annex D (thermally coupled zones, hourly method)",
        "coupled_zones": True,
        "coupling_scheme": COUPLING_SCHEME,
        "partition_conductance_note": PARTITION_CONDUCTANCE_NOTE,
        "zones": zones_tbl,
        "partitions": partitions,
        "zone_pair_conductances_W_K": {f"{a} <-> {b}": h for (a, b), h in pair_h.items()},
        "interzone_airflows": flows_tbl,
        "unconditioned_adjacent_zones": ztu,
        "partition_exchange": exchange,
        "issues": [str(i) for i in issues],
        "annual": annual_with_total(annual, zones) if annual is not None else None,
    }


def _token(name: Any) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")


def _dt_hours(hourly: pd.DataFrame) -> float:
    try:
        idx = pd.DatetimeIndex(hourly.index)
        if len(idx) > 1:
            return float((idx[1] - idx[0]).total_seconds() / 3600.0)
    except Exception:
        pass
    return 1.0


def format_multizone_report(report: Dict[str, Any]) -> str:
    """Plain-text rendering of :func:`multizone_report`."""
    L: List[str] = []
    L.append(report["standard"])
    L.append("=" * len(report["standard"]))
    L.append("")
    L.append("Zones")
    for z in report["zones"]:
        L.append(
            f"  {z['zone']}: A = {z['net_floor_area_m2']} m2, V = {z['volume_m3']} m3, "
            f"set-points {z['heating_setpoint_C']}/{z['cooling_setpoint_C']} C, "
            f"capacity {z['heating_capacity_W']}/{z['cooling_capacity_W']} W"
            + (", free-floating" if z["free_floating"] else "")
        )
    L.append("")
    L.append("Internal partitions (transmission coupling H_tr,zy)")
    for p in report["partitions"]:
        L.append(f"  {p['surface']}: {p['zone_a']} <-> {p['zone_b']}, A = {p['area_m2']:.2f} m2, "
                 f"U = {p['u_value']:.3f}, H_tr,zy = {p['H_tr_zy_W_K']:.1f} W/K")
    if not report["partitions"]:
        L.append("  (none)")
    L.append("")
    L.append("Air flow between zones (H_ve,z->y)")
    for f in report["interzone_airflows"]:
        L.append(f"  {f['from']} -> {f['to']}: {f['q_v_m3_h']:.1f} m3/h, H_ve = {f['H_ve_W_K']:.1f} W/K"
                 + (", outdoor make-up in the source zone" if f["makeup_from_outdoor"] else ""))
    if not report["interzone_airflows"]:
        L.append("  (none)")
    if report["unconditioned_adjacent_zones"]:
        L.append("")
        L.append("Adjacent thermally unconditioned zones (6.4.5)")
        for u in report["unconditioned_adjacent_zones"]:
            L.append(f"  {u['name']} ({u['type']}): b = {u['b']:.3f}, H_ztu;e = {u['H_ue']:.1f} W/K, "
                     f"H_ztc;ztu = {u['H_iu']:.1f} W/K")
    L.append("")
    L.append("Coupling scheme")
    L.append("  " + report["coupling_scheme"])
    L.append("  " + report["partition_conductance_note"])
    if report["partition_exchange"]:
        L.append("")
        L.append("Heat exchanged through the partitions over the period (kWh)")
        for e in report["partition_exchange"]:
            L.append(f"  {e['surface']}: to {e['to_zone_a_kWh']:+.1f} / {e['to_zone_b_kWh']:+.1f} "
                     f"(closure {e['closure_kWh']:+.2f})")
    if report["issues"]:
        L.append("")
        L.append("Checks")
        L.extend("  " + i for i in report["issues"])
    ann = report.get("annual")
    if ann is not None and len(ann):
        L.append("")
        L.append("Annual energy need")
        for _, r in ann.iterrows():
            L.append(f"  {r['zone']}: heating {r['Q_H_annual_kWh']:.0f} kWh, "
                     f"cooling {r['Q_C_annual_kWh']:.0f} kWh")
    return "\n".join(L)
