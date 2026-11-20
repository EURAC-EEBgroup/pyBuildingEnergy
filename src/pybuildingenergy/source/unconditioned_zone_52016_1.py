"""Thermally unconditioned adjacent zones according to EN ISO 52016-1:2017.

Implemented clauses
-------------------
* 6.4.5.1  two zone types: *external* (``ztue``, default) and *internal* (``ztui``)
* 6.4.5.4  adjustment factor ``b_ztu`` (Formulae 2-3), distribution factor
  ``F_ztc;ztu`` (Formulae 4-5), ``H_ztu;e`` with the default contribution of
  ventilation (Formula 6, Table B.5: ``c_ztu;ve = 0.5``)
* 6.4.5.5  default ``b_ztu`` values (Table A.4); the gains are then set to zero (E.3.1)
* 6.5.8.4  corrected thermal resistance for ``ztui`` (Formulae 54-56)
* 6.5.9    hourly temperature of an external ``ztu`` (Formulae 58-60, Table B.16)
* 6.5.10   ventilation from the ``ztu``: supply temperature (external type) or
  ``H_ve = b * rho * c * q`` (internal type, Formula 62)
* 6.5.12.1 / 6.5.13.1  gains attributed to the conditioned zone (Formulae 65, 68):
  only for the internal type
* E.3      internal and solar gains of the ``ztu`` itself (Formulae E.5-E.6)

Input (one entry of ``building_object["adjacent_zones"]``)
----------------------------------------------------------
Required (as before): ``name``, ``orientation_zone.azimuth``, ``area_facade_elements``,
``typology_elements``, ``transmittance_U_elements``, ``orientation_elements``,
``volume``, ``building_type_class``, ``a_use``.

Optional:

``type`` / ``ztu_type``
    ``"external"`` (default) or ``"internal"``.
``b_ztu``
    Default adjustment factor (Table A.4). Overrides the calculated value and, as
    required by E.3.1, the gains of the ``ztu`` are set to zero.
``c_ztu_ve``
    Coefficient of Formula 6 (default 0.5, Table B.5).
``n_ue``
    Only for the legacy ISO 13789 ventilation term ``0.33 * V * n_ue``. When given,
    Formula 6 is not used.
``c_ztu_h_max``
    Coefficient of Formula 59 (default 1.0, Table B.16).  ``None`` disables the limit.
    NOTE: ISO/TR 52016-2, 6.5.9, says that ``c = 0`` means "no maximum", but with Formula 59
    ``c = 0`` caps the temperature at the outdoor temperature.  The formula is followed for
    numbers; use ``None`` for "no maximum".
``g_value`` / ``g_value_elements``
    Total solar energy transmittance of the glazing of the external partition
    (``typology_elements == "W"``); default 0.6 (flagged in ``notes``).
``absorbing_surfaces``
    List of ``{"area", "orientation", "absorptance"}`` of the opaque surfaces inside the
    ``ztu`` (Formula E.6). Default: the opaque elements shared with the conditioned
    zone, absorptance 0.6 (Table B.15, category 2).
``obstacle_shading_factor``
    ``F_sh;obst`` of E.6 (default 1.0).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# rho_a * c_a in Wh/(m3 K), as used by ISO 13789 (flows in m3/h).
RHO_C_AIR_WH_M3K = 0.33

DEFAULT_C_ZTU_VE = 0.5          # Table B.5
DEFAULT_C_ZTU_H_MAX = 1.0       # Table B.16
DEFAULT_ABSORPTANCE = 0.6       # Table B.15, category 2
DEFAULT_G_VALUE = 0.6

_TRANSPARENT_LABELS = {"W", "TR", "WINDOW", "TRANSPARENT"}
_CARDINAL_AZIMUTHS = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
_CARDINAL_LABELS = np.array(["NV", "EV", "SV", "WV"], dtype=object)


def _zone_orientation_label(adj_zone: dict) -> str:
    """Cardinal label (NV=0, EV=90, SV=180, WV=270) of the wall shared with the zone."""
    try:
        azimuth = float(adj_zone["orientation_zone"]["azimuth"]) % 360.0
    except Exception:
        azimuth = 0.0
    diffs = np.abs(((azimuth - _CARDINAL_AZIMUTHS + 180.0) % 360.0) - 180.0)
    return str(_CARDINAL_LABELS[int(np.argmin(diffs))])


@dataclass
class UnconditionedZoneCoefficients:
    """Time-independent coefficients of one thermally unconditioned adjacent zone."""

    name: str
    ztu_type: str                    # "external" | "internal"
    orientation: str                 # cardinal label of the shared partition
    H_tr_ue: float                   # transmission to the exterior [W/K]
    H_ve_ue: float                   # ventilation to the exterior [W/K]
    H_ue: float                      # H_ztu;e [W/K]
    H_iu: float                      # H_ztc;ztu [W/K]
    H_tot: float                     # H_ztu;tot [W/K] (Formula 3)
    b: float                         # adjustment factor (Formula 2)
    F: float                         # distribution factor (Formulae 4-5)
    default_b_used: bool = False     # b taken from Table A.4 -> gains are zero
    c_ztu_h_max: Optional[float] = DEFAULT_C_ZTU_H_MAX
    f_sol_ue: float = 0.0            # F_sol;ue,ztu without time dependence (E.5)
    absorbing: List[Dict[str, Any]] = field(default_factory=list)
    obstacle_shading_factor: float = 1.0
    internal_gain_class: Optional[str] = None
    internal_gain_area: float = 0.0
    notes: List[str] = field(default_factory=list)


def compute_zone_coefficients(
    adj_zone: dict,
    n_ue: Optional[float] = None,
    qui: float = 0.0,
    c_ztu_ve: Optional[float] = None,
    other_conditioned_H_iu: Sequence[float] = (),
) -> UnconditionedZoneCoefficients:
    """Coefficients of an adjacent thermally unconditioned zone (6.4.5).

    ``H_ztu;e`` follows Formula 6, ``(1 + c_ztu;ve) * H_tr;ue`` (default
    ``c_ztu;ve = 0.5``), unless ``n_ue`` is given: then the ISO 13789 expression
    ``H_tr;ue + 0.33 * V * n_ue`` is used (legacy behaviour).

    ``other_conditioned_H_iu`` lists the ``H_ztc;ztu`` of any further conditioned zone
    adjacent to the same ``ztu``; it is used for ``H_ztu;tot`` and ``F_ztc;ztu``.
    """
    notes: List[str] = []
    ztu_type = str(adj_zone.get("type", adj_zone.get("ztu_type", "external"))).strip().lower()
    if ztu_type not in ("external", "internal"):
        raise ValueError(
            f"adjacent zone {adj_zone.get('name')!r}: type must be 'external' or 'internal', "
            f"got {ztu_type!r}"
        )

    areas = np.asarray(adj_zone["area_facade_elements"], dtype=float)
    u_values = np.asarray(adj_zone["transmittance_U_elements"], dtype=float)
    orientations = np.asarray(adj_zone["orientation_elements"], dtype=object)
    typology = np.asarray(adj_zone.get("typology_elements", ["OP"] * len(areas)), dtype=object)
    if not (len(areas) == len(u_values) == len(orientations)):
        raise ValueError(
            f"adjacent zone {adj_zone.get('name')!r}: area_facade_elements, "
            "transmittance_U_elements and orientation_elements must have the same length"
        )

    label = _zone_orientation_label(adj_zone)
    mask_shared = orientations == label
    mask_external = ~mask_shared

    H_iu_tr = float(np.sum(areas[mask_shared] * u_values[mask_shared]))
    H_tr_ue = float(np.sum(areas[mask_external] * u_values[mask_external]))

    if n_ue is None and adj_zone.get("n_ue") is not None:
        n_ue = float(adj_zone["n_ue"])
    if c_ztu_ve is None:
        c_ztu_ve = float(adj_zone.get("c_ztu_ve", DEFAULT_C_ZTU_VE))

    if n_ue is not None:
        H_ve_ue = RHO_C_AIR_WH_M3K * float(adj_zone["volume"]) * float(n_ue)
        H_ue = H_tr_ue + H_ve_ue
        notes.append("H_ztu;e from ISO 13789 ventilation (n_ue given), not Formula 6")
    else:
        H_ve_ue = c_ztu_ve * H_tr_ue
        H_ue = (1.0 + c_ztu_ve) * H_tr_ue                       # Formula 6

    H_iu = H_iu_tr + RHO_C_AIR_WH_M3K * float(qui)
    H_iu_sum = H_iu + float(sum(other_conditioned_H_iu))
    H_tot = H_iu_sum + H_ue                                      # Formula 3
    if H_tot <= 0.0:
        raise ValueError(f"adjacent zone {adj_zone.get('name')!r}: H_ztu;tot must be positive")
    b = H_ue / H_tot                                             # Formula 2
    F = H_iu / H_iu_sum if H_iu_sum > 0.0 else 1.0               # Formulae 4-5

    default_b_used = False
    if adj_zone.get("b_ztu") is not None:
        b_default = float(adj_zone["b_ztu"])
        if not 0.0 <= b_default <= 1.0:
            raise ValueError(f"adjacent zone {adj_zone.get('name')!r}: b_ztu must be in [0, 1]")
        b = b_default
        default_b_used = True
        notes.append("b_ztu from Table A.4 default; gains of the zone set to zero (E.3.1)")

    # --- E.3: reduction factor of the external partition and absorbing surfaces ---
    is_transparent = np.array(
        [str(t).strip().upper() in _TRANSPARENT_LABELS for t in typology], dtype=bool
    )
    ext_opaque = float(np.sum(areas[mask_external & ~is_transparent]))
    ext_transp = float(np.sum(areas[mask_external & is_transparent]))
    f_sol_ue = 0.0
    if ext_transp > 0.0:
        f_fr = ext_opaque / (ext_opaque + ext_transp)            # E.5 definition
        g_elems = adj_zone.get("g_value_elements")
        if g_elems is not None:
            g_arr = np.asarray(g_elems, dtype=float)
            sel = mask_external & is_transparent
            g_gl = float(np.sum(g_arr[sel] * areas[sel]) / ext_transp)
        elif adj_zone.get("g_value") is not None:
            g_gl = float(adj_zone["g_value"])
        else:
            g_gl = DEFAULT_G_VALUE
            notes.append(f"glazing g-value not given: default {DEFAULT_G_VALUE} used")
        f_sol_ue = g_gl * (1.0 - f_fr)                           # Formula E.5

    absorbing = adj_zone.get("absorbing_surfaces")
    if absorbing is None:
        sel = mask_shared & ~is_transparent
        absorbing = [
            {"area": float(a), "orientation": str(o), "absorptance": DEFAULT_ABSORPTANCE}
            for a, o in zip(areas[sel], orientations[sel])
        ]
    else:
        absorbing = [
            {
                "area": float(s["area"]),
                "orientation": str(s["orientation"]),
                "absorptance": float(s.get("absorptance", DEFAULT_ABSORPTANCE)),
            }
            for s in absorbing
        ]

    if ztu_type == "internal" and f_sol_ue > 0.0:
        warnings.warn(
            f"adjacent zone {adj_zone.get('name')!r}: the internal type is not suitable for "
            "spaces with high solar gains (sunspace, atrium): the gains in the conditioned zone "
            "can be overestimated (ISO/TR 52016-2, 6.4.5). Use type 'external'.",
            stacklevel=3,
        )
        notes.append("internal type with glazing: not suitable for sunspaces/atria (TR 52016-2, 6.4.5)")

    return UnconditionedZoneCoefficients(
        name=str(adj_zone.get("name", "")),
        ztu_type=ztu_type,
        orientation=label,
        H_tr_ue=H_tr_ue,
        H_ve_ue=H_ve_ue,
        H_ue=H_ue,
        H_iu=H_iu,
        H_tot=H_tot,
        b=b,
        F=F,
        default_b_used=default_b_used,
        c_ztu_h_max=(
            None if ("c_ztu_h_max" in adj_zone and adj_zone["c_ztu_h_max"] is None)
            else float(adj_zone.get("c_ztu_h_max", DEFAULT_C_ZTU_H_MAX))
        ),
        f_sol_ue=f_sol_ue,
        absorbing=absorbing,
        obstacle_shading_factor=float(adj_zone.get("obstacle_shading_factor", 1.0)),
        internal_gain_class=adj_zone.get("building_type_class"),
        internal_gain_area=float(adj_zone.get("a_use", 0.0) or 0.0),
        notes=notes,
    )


def corrected_thermal_resistance(
    r_c: float, b: float, h_si: float, h_se: float
) -> float:
    """Formula 54: R_c corrected for an internal unconditioned zone (6.5.8.4).

    ``h_si`` and ``h_se`` are the total (convective + radiative) surface coefficients.
    """
    if not 0.0 <= b < 1.0:
        raise ValueError(f"adjustment factor b must be in [0, 1) for Formula 54, got {b}")
    return (r_c + b * (1.0 / h_si + 1.0 / h_se)) / (1.0 - b)


def ztu_theta_external(
    theta_op_prev: float,
    theta_e: float,
    b: float,
    phi_gn_dir: float,
    h_tot: float,
    c_h_max: Optional[float] = DEFAULT_C_ZTU_H_MAX,
) -> float:
    """Hourly temperature of an external ztu: Formulae 58 and 59 (``c_h_max=None``: no limit)."""
    theta = theta_op_prev - b * (theta_op_prev - theta_e) + phi_gn_dir / h_tot   # (58)
    if c_h_max is None:
        return theta
    theta_max = theta_e + c_h_max * (theta_op_prev - theta_e)                   # (59)
    return min(theta_max, theta)


class UnconditionedZoneSet:
    """All the thermally unconditioned adjacent zones of a single-zone building."""

    def __init__(self, building_object: dict):
        bld = building_object.get("building", {}) or {}
        zones = building_object.get("adjacent_zones", []) or []
        self.active = bool(bld.get("adj_zones_present")) and len(zones) > 0
        self.zone_dicts: List[dict] = list(zones) if self.active else []
        self.names: List[str] = []
        self.coeffs: List[UnconditionedZoneCoefficients] = []
        if not self.active:
            return
        declared = bld.get("number_adj_zone")
        if declared is not None and int(declared) != len(zones):
            warnings.warn(
                f"building.number_adj_zone={declared} but {len(zones)} adjacent_zones are "
                "defined: the adjacent_zones list is used",
                stacklevel=3,
            )
        for z in zones:
            c = compute_zone_coefficients(z)
            self.names.append(c.name)
            self.coeffs.append(c)
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate adjacent zone names: {self.names}")
        self._theta_hist: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ lookups
    def __len__(self) -> int:
        return len(self.coeffs)

    def index_of(self, name: Optional[str]) -> Optional[int]:
        """Index of a zone by name; a single zone is the fallback for unnamed surfaces."""
        if not self.active:
            return None
        if name is not None and name in self.names:
            return self.names.index(name)
        if len(self.names) == 1:
            return 0
        return None

    def coefficients_for(self, name: Optional[str]) -> Optional[UnconditionedZoneCoefficients]:
        i = self.index_of(name)
        return None if i is None else self.coeffs[i]

    # ----------------------------------------------------------------- hourly
    def init_history(self, n_steps: int) -> None:
        self._theta_hist = np.zeros((n_steps, len(self.coeffs)), dtype=float)

    @property
    def theta_history(self) -> Optional[np.ndarray]:
        return self._theta_hist

    def gains_direct(
        self,
        internal_gain_fn,
        solar_series: Optional[Dict[str, Dict[str, float]]],
    ) -> List[Dict[str, float]]:
        """Phi_int;dir;ztu and Phi_sol;dir;ztu of every zone for one hour.

        ``internal_gain_fn(building_type_class, a_use)`` returns Phi_int;dir [W] (6.5.12.2).
        ``solar_series`` maps an orientation label to ``{"dif": W/m2, "dir": W/m2}``.
        """
        out = []
        for c in self.coeffs:
            if c.default_b_used:
                out.append({"int": 0.0, "sol": 0.0})
                continue
            phi_int = float(internal_gain_fn(c.internal_gain_class, c.internal_gain_area))
            phi_sol = 0.0
            if c.f_sol_ue > 0.0 and solar_series is not None:
                for s in c.absorbing:
                    irr = solar_series.get(s["orientation"])
                    if irr is None:
                        continue
                    phi_sol += (
                        s["absorptance"]
                        * s["area"]
                        * (irr["dif"] + irr["dir"] * c.obstacle_shading_factor)
                    )
                phi_sol *= c.f_sol_ue                                            # (E.6)
            out.append({"int": phi_int, "sol": phi_sol})
        return out

    def step(
        self,
        t: int,
        theta_op_prev,
        theta_e: float,
        gains: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """State of every zone at hour ``t``: temperature, boundary temperature, b, gains.

        ``theta_op_prev`` is the operative temperature of the conditioned zone at hour t-1:
        one value for all the zones, or one value per zone (multizone building, where each
        unconditioned zone borders its own conditioned zones).
        """
        states = []
        per_zone = np.ndim(theta_op_prev) > 0
        all_op = theta_op_prev
        for k, c in enumerate(self.coeffs):
            theta_op_prev = float(all_op[k]) if per_zone else float(all_op)
            phi_int = gains[k]["int"]
            phi_sol = gains[k]["sol"]
            if c.ztu_type == "external":
                theta = ztu_theta_external(
                    theta_op_prev, theta_e, c.b, phi_int + phi_sol, c.H_tot, c.c_ztu_h_max
                )
                theta_boundary = theta
            else:
                # 6.4.5.3: internal type -> operative temperature of the conditioned zone,
                # without the effect of gains. The partition sees the exterior through the
                # corrected resistance (Formula 54).
                theta = float(theta_op_prev)
                theta_boundary = float(theta_e)
            if self._theta_hist is not None:
                self._theta_hist[t, k] = theta
            states.append(
                {
                    "name": c.name,
                    "type": c.ztu_type,
                    "theta": float(theta),
                    "theta_boundary": float(theta_boundary),
                    "b": c.b,
                    "F": c.F,
                    "phi_int_dir": phi_int,
                    "phi_sol_dir": phi_sol,
                }
            )
        return states

    @staticmethod
    def attributed_gains(states: List[Dict[str, Any]]) -> Dict[str, float]:
        """Gains of internal-type zones attributed to the conditioned zone.

        Formula 65: sum (1 - b) * F * Phi_int;dir;ztu
        Formula 68: sum (1 - b) * F * Phi_sol;dir;ztu
        External-type zones attribute nothing (their effect is in the temperature).
        """
        w_int = 0.0
        w_sol = 0.0
        for s in states:
            if s["type"] != "internal":
                continue
            w = (1.0 - s["b"]) * s["F"]
            w_int += w * s["phi_int_dir"]
            w_sol += w * s["phi_sol_dir"]
        return {"int": w_int, "sol": w_sol}

    @staticmethod
    def ventilation_state(states: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """State dictionary understood by the ventilation boundary resolver."""
        return {
            s["name"]: {"type": s["type"], "theta": s["theta"], "b": s["b"]} for s in states
        }

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": c.name, "type": c.ztu_type, "H_tr_ue": c.H_tr_ue, "H_ve_ue": c.H_ve_ue,
                "H_ue": c.H_ue, "H_iu": c.H_iu, "H_tot": c.H_tot, "b": c.b, "F": c.F,
                "f_sol_ue": c.f_sol_ue, "notes": list(c.notes),
            }
            for c in self.coeffs
        ]


def legacy_iso13789_coefficients(adj_zone: dict, n_ue: Optional[float] = None, qui: float = 0.0):
    """Backward-compatible ``(H_ztu_tot, b_ztu, F_ztc_ztu)`` tuple."""
    c = compute_zone_coefficients(adj_zone, n_ue=n_ue, qui=qui)
    return float(round(c.H_tot, 3)), float(round(c.b, 3)), float(c.F)
