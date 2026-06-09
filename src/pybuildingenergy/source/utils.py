__author__ = "Daniele Antonucci, Ulrich Filippi Oberegger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich Filippi Oberegger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"

import requests
import pandas as pd
import datetime as dt

import math
import io
from pathlib import Path
import tempfile

import copy
import re
import time
# timezonefinder imported lazily -- avoids ~0.7s startup cost
from pytz import timezone
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
# pvlib imported lazily -- avoids ~1.9s startup cost
from .ventilation import (
    VentilationInternalGains,
    VentilationStream,
    VentilationBoundary,
    resolve_ventilation_boundary,
)
from .functions import *
# generate_profile imported lazily -- avoids ~0.8s startup cost
from .table_iso_16798_1 import * 
from . import table_iso_16798_1 as iso16798_profiles


@dataclass
class WeatherDataResult:
    elevation: float
    weather_data: pd.DataFrame
    utc_offset: int
    latitude: float
    longitude: float


@dataclass
class Solar_irradiance:
    """
    Hourly solar irradiance
    Output of `Solar_irradiance_calculation`
    """

    solar_irradiance: pd.DataFrame

@dataclass
class Shading_Reduction_factor_window :
    """
    Hourly shading reduction factor calculated as Annex F
    """

    shading_reduction_factor_window: pd.DataFrame

@dataclass
class simdf_52010:
    sim_df: pd.DataFrame


@dataclass
class numb_nodes_facade_elements:
    Rn: int
    Pln: np.array
    PlnSum: np.array


@dataclass
class conduttance_elements:
    h_pli_eli: np.array


@dataclass
class solar_absorption_elements:
    a_sol_pli_eli: np.array


@dataclass
class aeral_heat_capacity:
    kappa_pli_eli: np.array


@dataclass
class simulation_df:
    simulation_df: pd.DataFrame


@dataclass
class temp_ground:
    R_gr_ve: float
    Theta_gr_ve: np.array
    thermal_bridge_heat: float
    ground_contact_area: float = 0.0


@dataclass
class h_vent_and_int_gains:
    H_ve: pd.Series
    # Phi_int: pd.Series
    sim_df_update: pd.DataFrame


@dataclass
class h_natural_vent:
    H_ve_nat: np.array


def _make_sched_resolver(kwargs, iso16798_profiles_obj):
    """Return a schedule-kwarg resolver that handles explicit None correctly.

    ``kwargs.get(key, default)`` only returns *default* when the key is absent;
    it returns ``None`` when the caller passes the key explicitly as ``None``
    (e.g. the multizone hybrid caller always passes ``occupants_schedule_workdays=None``).
    Using an explicit ``is None`` check avoids this and also avoids the ``v or default``
    anti-pattern, which silently replaces falsey-but-valid inputs such as ``{}`` and
    raises ``ValueError`` for array-like inputs whose truth value is ambiguous.
    """
    def _sched(key, attr):
        v = kwargs.get(key)
        return (
            getattr(iso16798_profiles_obj, attr, {"Residential_apartment": [1.0] * 24})
            if v is None else v
        )
    return _sched


def _resolve_single_zone_vent_boundary(building_object, T_zone, Tstepi, sim_df, profile_df):
    """Build the affine ventilation boundary for legacy and causal single-zone solvers.

    Reads zone volume and ventilation config from *building_object*, resolves per-component
    profile multipliers from *profile_df* at timestep *Tstepi*, and delegates to
    resolve_ventilation_boundary.  Centralises logic that is otherwise duplicated in the
    legacy and causal solver timestep loops.
    """
    _bld = building_object.get("building", {})
    _zone_vol = float(
        _bld.get("zone_volume_m3") or _bld.get("zone_volume") or _bld.get("volume") or 0.0
    )
    _vent_cfg = building_object.get("building_parameters", {}).get("ventilation", {}) or {}
    _comp_mult: dict = {}
    for _comp in _vent_cfg.get("components", []):
        _cname = str(_comp.get("name", "")).strip()
        _cprof = _comp.get("profile")
        if _cname and _cprof is not None:
            _col = str(_cprof)
            if _col in profile_df.columns:
                _comp_mult[_cname] = float(profile_df[_col].iloc[Tstepi])
            else:
                import warnings as _w
                _w.warn(
                    f"Component {_cname!r}: profile column {_col!r} not in "
                    f"profile_df; available columns: {list(profile_df.columns)}. Using 1.0.",
                    stacklevel=3,
                )
    return resolve_ventilation_boundary(
        building_object,
        float(T_zone),
        float(sim_df.iloc[Tstepi]["T2m"]),
        float(sim_df.iloc[Tstepi].get("WS10m", 0.0) or 0.0),
        profile_multiplier=float(profile_df["ventilation_profile"].iloc[Tstepi]),
        component_multipliers=_comp_mult if _comp_mult else None,
        zone_volume_m3=_zone_vol if _zone_vol > 0.0 else None,
    )


def _infer_timestep_hours_from_index(index, default=1.0):
    """
    Infer representative timestep duration [h] from an index.
    Fallback is `default` when index is not datetime-like or irregular/empty.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return float(default)

    dt_h = (
        index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600.0)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    dt_h = dt_h[dt_h > 0.0]
    if dt_h.empty:
        return float(default)
    return float(dt_h.median())


def _prepend_december_warmup_to_previous_year(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepend a copy of December shifted to the previous year.

    This keeps the warm-up block explicit and avoids timestamp overlap with the
    actual simulation December, which would otherwise be collapsed by duplicate
    index handling in downstream solvers.
    """
    if not isinstance(sim_df, pd.DataFrame) or sim_df.empty:
        return sim_df

    idx = pd.DatetimeIndex(sim_df.index)
    if len(idx) == 0:
        return sim_df

    dec_mask = idx.month == 12
    if not bool(np.any(dec_mask)):
        return sim_df

    warmup_df = sim_df.loc[dec_mask].copy()
    warmup_df.index = pd.DatetimeIndex(warmup_df.index) - pd.DateOffset(years=1)
    return pd.concat([warmup_df, sim_df], axis=0)


def _integrate_power_series_to_energy_wh(power_series, default_dt_h=1.0):
    """
    Integrate a power series [W] into energy [Wh].
    Uses per-step datetime deltas when available, otherwise assumes `default_dt_h`.
    """
    p = pd.to_numeric(power_series, errors="coerce").fillna(0.0)
    if p.empty:
        return 0.0

    idx = p.index
    if not isinstance(idx, pd.DatetimeIndex):
        return float(p.sum() * float(default_dt_h))

    if len(p) < 2:
        return float(p.iloc[0] * float(default_dt_h))

    idx_series = idx.to_series(index=p.index)
    dt_h = idx_series.shift(-1).sub(idx_series).dt.total_seconds().div(3600.0)
    fallback_dt_h = _infer_timestep_hours_from_index(idx, default=default_dt_h)
    dt_h = dt_h.fillna(fallback_dt_h)
    dt_h = dt_h.where(np.isfinite(dt_h) & (dt_h > 0.0), fallback_dt_h)
    return float((p * dt_h).sum())


def _ground_contact_area(building_object):
    """
    Total slab-on-ground area [m2].
    Prefer explicit GROUND surfaces; fall back to legacy downward surfaces;
    finally use building net floor area when no surface metadata is available.
    """
    surfaces = building_object.get("building_surface", []) if isinstance(building_object, dict) else []

    area_ground = 0.0
    area_fallback = 0.0
    for surf in surfaces:
        if not isinstance(surf, dict):
            continue
        try:
            area = float(surf.get("area", 0.0))
        except Exception:
            continue
        if not np.isfinite(area) or area <= 0.0:
            continue

        boundary = str(surf.get("boundary", "")).upper()
        iso_type = str(surf.get("ISO52016_type_string", "")).upper()
        if boundary == "GROUND" or iso_type == "GR":
            area_ground += area
            continue

        # Legacy fallback for dictionaries that do not carry an explicit boundary.
        orientation = surf.get("orientation", {}) or {}
        tilt = orientation.get("tilt", None)
        sky_view_factor = surf.get("sky_view_factor", None)
        try:
            tilt = float(tilt) if tilt is not None else None
        except Exception:
            tilt = None
        try:
            sky_view_factor = (
                float(sky_view_factor) if sky_view_factor is not None else None
            )
        except Exception:
            sky_view_factor = None

        if (
            sky_view_factor is not None
            and abs(sky_view_factor) < 1e-9
            and (tilt is None or tilt > 170.0)
        ):
            area_fallback += area

    if area_ground > 0.0:
        return float(area_ground)
    if area_fallback > 0.0:
        return float(area_fallback)

    floor_area = None
    if isinstance(building_object, dict):
        floor_area = building_object.get("building", {}).get("net_floor_area")
    try:
        floor_area = float(floor_area)
    except Exception:
        floor_area = None
    if floor_area is not None and np.isfinite(floor_area) and floor_area > 0.0:
        return float(floor_area)

    raise ValueError(
        "Ground-contact area is missing: provide GROUND surfaces or set building.net_floor_area."
    )


def _ground_conductance_w_per_k(area_m2: float, ground_data: temp_ground | None) -> float:
    """Convert area-specific virtual-ground resistance [m2K/W] into conductance [W/K]."""
    if ground_data is None:
        return 0.0
    try:
        area = float(area_m2)
        r_gr = float(ground_data.R_gr_ve)
    except Exception:
        return 0.0
    if not np.isfinite(area) or area <= 0.0:
        return 0.0
    if not np.isfinite(r_gr) or r_gr <= 0.0:
        return 0.0
    return area / r_gr


def _sanitize_result_column_token(value) -> str:
    """Build a conservative ASCII-like token suitable for result column names."""
    token = []
    for ch in str(value):
        token.append(ch if ch.isalnum() else "_")
    out = "".join(token).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unnamed"


def _ground_temperature_at_month_index(ground_data: temp_ground | None, month_index: int) -> float:
    """Return the virtual ground temperature [C] at the requested 0-based month index."""
    if ground_data is None:
        return float("nan")
    try:
        theta_gr = np.asarray(ground_data.Theta_gr_ve, dtype=float).reshape(-1)
    except Exception:
        return float("nan")
    if theta_gr.size == 0:
        return float("nan")
    try:
        month = int(month_index)
    except Exception:
        month = 0
    if month < 0 or month >= theta_gr.size:
        month = 0
    return float(theta_gr[month])


def _build_multizone_ground_flux_links(
    surfaces,
    nodes,
    zone_names,
    z_idx,
    ground_data: temp_ground | None,
    sys_row_from_surface_ri,
):
    """
    Build ground-boundary links for multizone post-processing of true solver fluxes.

    Returns:
      - ground_links: list of dicts with zone index, external-node row, conductance and column token
      - zone_h_ground: ndarray [W/K] aggregated by zone
    """
    zone_h_ground = np.zeros(len(zone_names), dtype=float)
    ground_links = []
    used_tokens = set()

    for Eli, surf in enumerate(surfaces):
        if str(surf.get("ISO52016_type_string", "")).upper() != "GR":
            continue

        try:
            n_nodes = int(nodes.Pln[Eli])
        except Exception:
            continue
        if n_nodes <= 0:
            continue

        zname = surf.get("zone", zone_names[0] if zone_names else "main")
        if zname not in z_idx:
            continue

        try:
            area_s = float(surf.get("area", 0.0))
        except Exception:
            continue
        h_gr = _ground_conductance_w_per_k(area_s, ground_data)
        if not np.isfinite(h_gr) or h_gr <= 0.0:
            continue

        try:
            ri_ext = 1 + int(nodes.PlnSum[Eli])
            row_ext = int(sys_row_from_surface_ri(ri_ext))
        except Exception:
            continue

        base_token = _sanitize_result_column_token(surf.get("name", f"ground_surface_{Eli}"))
        token = base_token
        counter = 2
        while token in used_tokens:
            token = f"{base_token}_{counter}"
            counter += 1
        used_tokens.add(token)

        zi = int(z_idx[zname])
        zone_h_ground[zi] += h_gr
        ground_links.append(
            {
                "zone_index": zi,
                "zone_name": zname,
                "surface_name": str(surf.get("name", f"ground_surface_{Eli}")),
                "surface_token": token,
                "row_ext": row_ext,
                "h_gr": float(h_gr),
            }
        )

    return ground_links, zone_h_ground


def _ground_fluxes_from_state(
    theta_state: np.ndarray,
    month_index: int,
    ground_data: temp_ground | None,
    ground_links,
    zone_names,
):
    """
    Compute actual ground-boundary fluxes from the solved state.

    Sign convention:
      - positive: heat leaves the building towards the ground
      - negative: heat enters the building from the ground
    """
    T_gr = _ground_temperature_at_month_index(ground_data, month_index)
    zone_flux = {str(name): 0.0 for name in zone_names}
    surface_flux = {}

    if not np.isfinite(T_gr):
        for link in ground_links:
            surface_flux[link["surface_token"]] = float("nan")
        return T_gr, zone_flux, surface_flux

    for link in ground_links:
        row_ext = int(link["row_ext"])
        if row_ext < 0 or row_ext >= len(theta_state):
            continue
        q_gr = float(link["h_gr"]) * (float(theta_state[row_ext]) - float(T_gr))
        zone_flux[str(link["zone_name"])] += q_gr
        surface_flux[link["surface_token"]] = q_gr

    return T_gr, zone_flux, surface_flux


def _build_multizone_opaque_inside_flux_links(
    surfaces,
    nodes,
    zone_names,
    z_idx,
    sys_row_from_surface_ri,
    h_pli_eli,
):
    """
    Build per-surface links for true inside-face opaque conduction fluxes.

    Exported surfaces are opaque envelope faces with OUTDOORS or GROUND boundary.

    Sign convention for the associated flux:
      - positive: heat enters the zone from the opaque surface
      - negative: heat leaves the zone towards the opaque surface
    """
    links = []
    used_tokens = set()

    for Eli, surf in enumerate(surfaces):
        if str(surf.get("type", "")).lower() != "opaque":
            continue

        boundary = str(surf.get("boundary", "")).upper()
        if boundary not in {"OUTDOORS", "GROUND"}:
            continue

        try:
            n_nodes = int(nodes.Pln[Eli])
        except Exception:
            continue
        if n_nodes <= 1:
            continue

        zname = surf.get("zone", zone_names[0] if zone_names else "main")
        if zname not in z_idx:
            continue

        try:
            area_s = float(surf.get("area", 0.0))
            h_cond_face = float(h_pli_eli[n_nodes - 2, Eli]) * area_s
        except Exception:
            continue
        if not np.isfinite(h_cond_face) or h_cond_face <= 0.0:
            continue

        try:
            ri_in = 1 + int(nodes.PlnSum[Eli]) + (n_nodes - 1)
            row_in = int(sys_row_from_surface_ri(ri_in))
            row_prev = int(sys_row_from_surface_ri(ri_in - 1))
        except Exception:
            continue

        base_token = _sanitize_result_column_token(surf.get("name", f"opaque_surface_{Eli}"))
        token = base_token
        counter = 2
        while token in used_tokens:
            token = f"{base_token}_{counter}"
            counter += 1
        used_tokens.add(token)

        links.append(
            {
                "zone_index": int(z_idx[zname]),
                "zone_name": str(zname),
                "surface_name": str(surf.get("name", f"opaque_surface_{Eli}")),
                "surface_token": token,
                "row_in": row_in,
                "row_prev": row_prev,
                "h_cond_face": float(h_cond_face),
            }
        )

    return links


def _opaque_inside_fluxes_from_state(
    theta_state: np.ndarray,
    opaque_inside_links,
):
    """
    Compute true inside-face conduction fluxes for opaque envelope surfaces.

    Sign convention:
      - positive: heat enters the zone from the opaque surface
      - negative: heat leaves the zone towards the opaque surface
    """
    surface_flux = {}
    for link in opaque_inside_links:
        row_in = int(link["row_in"])
        row_prev = int(link["row_prev"])
        token = str(link["surface_token"])
        if (
            row_in < 0
            or row_prev < 0
            or row_in >= len(theta_state)
            or row_prev >= len(theta_state)
        ):
            surface_flux[token] = float("nan")
            continue
        q_in = float(link["h_cond_face"]) * (
            float(theta_state[row_prev]) - float(theta_state[row_in])
        )
        surface_flux[token] = q_in
    return surface_flux


def h_ci_tarp(
    T_air_C: float | np.ndarray,
    T_surf_C: float | np.ndarray,
    tilt_deg: float,
    h_min: float = 0.1,
) -> float | np.ndarray:
    """
    Internal natural convection coefficient using TARP/Walton (EnergyPlus default).

    Base law: h = C * |T_air - T_surf|^(1/3), where C depends on tilt and buoyancy stability.
    Inputs in degC, output in W/(m2 K).
    """
    dT = np.asarray(T_air_C, dtype=float) - np.asarray(T_surf_C, dtype=float)
    adT = np.abs(dT)
    coeff = np.where(adT > 1e-6, np.power(adT, 1.0 / 3.0), 0.0)

    if abs(float(tilt_deg) - 90.0) < 10.0:
        h = 1.31 * coeff
    elif float(tilt_deg) < 10.0:
        unstable = dT < 0.0
        h = np.where(unstable, 1.52, 0.76) * coeff
    elif float(tilt_deg) > 170.0:
        unstable = dT > 0.0
        h = np.where(unstable, 1.52, 0.76) * coeff
    else:
        h = 1.31 * coeff

    h = np.maximum(h, float(h_min))
    if np.ndim(h) == 0:
        return float(h)
    return h


SIGMA = 5.67e-8


def h_re_sky(
    T_surf_C: float | np.ndarray,
    T_sky_C: float | np.ndarray,
    epsilon: float = 0.9,
) -> float | np.ndarray:
    """
    Dynamic linearized external long-wave radiative coefficient.

    h_re = epsilon * sigma * (Ts^2 + Tref^2) * (Ts + Tref)  [W/(m2 K)]
    with temperatures in Kelvin.
    """
    Ts = np.asarray(T_surf_C, dtype=float) + 273.15
    Tsk = np.asarray(T_sky_C, dtype=float) + 273.15
    h = float(epsilon) * SIGMA * (Ts ** 2 + Tsk ** 2) * (Ts + Tsk)
    if np.ndim(h) == 0:
        return float(h)
    return h


def T_sky_berdahl_fromberg(
    T_air_C: float | np.ndarray,
    T_dew_C: float | np.ndarray,
) -> float | np.ndarray:
    """
    Equivalent sky temperature from Berdahl & Fromberg (1982).
    """
    eps_sky = 0.787 + 0.764 * np.log((np.asarray(T_dew_C, dtype=float) + 273.15) / 273.15)
    eps_sky = np.clip(eps_sky, 0.05, 1.0)
    T_sky_K = np.power(eps_sky, 0.25) * (np.asarray(T_air_C, dtype=float) + 273.15)
    out = T_sky_K - 273.15
    if np.ndim(out) == 0:
        return float(out)
    return out


def T_sky_swinbank(T_air_C: float | np.ndarray) -> float | np.ndarray:
    """
    Equivalent sky temperature from Swinbank (1963), clear-sky correlation.
    """
    T_air_K = np.asarray(T_air_C, dtype=float) + 273.15
    out = 0.0553 * np.power(T_air_K, 1.5) - 273.15
    if np.ndim(out) == 0:
        return float(out)
    return out


def q_re_surface(
    T_surf_C: float | np.ndarray,
    T_sky_C: float | np.ndarray,
    T_air_C: float | np.ndarray,
    tilt_deg: float,
    epsilon: float = 0.9,
    T_gnd_C: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Net external long-wave radiative heat flux [W/m2], positive as heat loss from surface.
    """
    if T_gnd_C is None:
        T_gnd_C = T_air_C

    tilt_r = np.radians(float(tilt_deg))
    F_sky = (1.0 + np.cos(tilt_r)) / 2.0
    F_gnd = (1.0 - np.cos(tilt_r)) / 2.0

    q_sky = F_sky * h_re_sky(T_surf_C, T_sky_C, epsilon) * (np.asarray(T_surf_C) - np.asarray(T_sky_C))
    q_gnd = F_gnd * h_re_sky(T_surf_C, T_gnd_C, epsilon) * (np.asarray(T_surf_C) - np.asarray(T_gnd_C))
    out = q_sky + q_gnd
    if np.ndim(out) == 0:
        return float(out)
    return out


def _get_simulation_options(building_object) -> dict:
    if not isinstance(building_object, dict):
        return {}
    bp = building_object.get("building_parameters", {}) or {}
    sim_opt = bp.get("simulation_options", {}) or {}
    return sim_opt if isinstance(sim_opt, dict) else {}


def _normalize_ground_temperature_model(model_raw) -> str:
    if model_raw is None:
        return "iso13370"
    model = str(model_raw).strip().lower().replace("-", "_")
    aliases = {
        "default": "iso13370",
        "iso": "iso13370",
        "iso_13370": "iso13370",
        "monthly": "monthly",
        "monthly_profile": "monthly",
        "custom_monthly": "monthly",
        "energy_plus": "monthly",
        "eplus": "monthly",
        "energyplus": "monthly",
        "energyplus_monthly": "monthly",
        "fixed_monthly": "monthly",
        "site_ground_temperature_building_surface": "monthly",
    }
    model = aliases.get(model, model)
    if model not in {"iso13370", "monthly"}:
        raise ValueError("ground_temperature_model must be 'iso13370' or 'monthly'.")
    return model


def _normalize_ground_temperature_monthly(values_raw) -> np.ndarray:
    if values_raw is None:
        raise ValueError(
            "ground_temperature_monthly is required when ground_temperature_model='monthly' "
            "(or legacy alias 'energyplus')."
        )

    raw_values = values_raw
    if isinstance(values_raw, str):
        tokens = [token for token in re.split(r"[\s,;]+", values_raw.strip()) if token]
        raw_values = tokens

    try:
        values = np.asarray(raw_values, dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(
            "ground_temperature_monthly must contain 12 numeric monthly values [degC]."
        ) from exc

    if values.size != 12:
        raise ValueError("ground_temperature_monthly must contain exactly 12 monthly values [degC].")
    if not np.all(np.isfinite(values)):
        raise ValueError("ground_temperature_monthly must contain only finite numeric values.")
    return values.astype(float, copy=True)


def _resolve_ground_temperature_model(building_object, model_override=None) -> str:
    if model_override is not None:
        return _normalize_ground_temperature_model(model_override)

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in ("ground_temperature_model", "soil_temperature_model"):
            if sim_opt.get(key) is not None:
                return _normalize_ground_temperature_model(sim_opt.get(key))
        bp = building_object.get("building_parameters", {}) or {}
        for key in ("ground_temperature_model", "soil_temperature_model"):
            if bp.get(key) is not None:
                return _normalize_ground_temperature_model(bp.get(key))

    return "iso13370"


def _resolve_ground_temperature_monthly(building_object, monthly_override=None) -> np.ndarray | None:
    if monthly_override is not None:
        return _normalize_ground_temperature_monthly(monthly_override)

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in (
            "ground_temperature_monthly",
            "energyplus_ground_temperature_monthly",
            "site_ground_temperature_building_surface",
        ):
            if sim_opt.get(key) is not None:
                return _normalize_ground_temperature_monthly(sim_opt.get(key))
        bp = building_object.get("building_parameters", {}) or {}
        for key in (
            "ground_temperature_monthly",
            "energyplus_ground_temperature_monthly",
            "site_ground_temperature_building_surface",
        ):
            if bp.get(key) is not None:
                return _normalize_ground_temperature_monthly(bp.get(key))

    return None


def _normalize_internal_convection_model(model_raw) -> str:
    if model_raw is None:
        return "table"
    model = str(model_raw).strip().lower().replace("-", "_")
    aliases = {
        "default": "table",
        "tabella": "table",
        "fixed": "table",
        "constant": "table",
        "tarp_walton": "tarp",
        "walton": "tarp",
    }
    model = aliases.get(model, model)
    if model not in {"table", "tarp"}:
        raise ValueError("internal_convection_model must be 'table' or 'tarp'.")
    return model


def _resolve_internal_convection_model(building_object, model_override=None) -> str:
    if model_override is not None:
        return _normalize_internal_convection_model(model_override)

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in ("internal_convection_model", "h_ci_model", "convective_internal_model"):
            if sim_opt.get(key) is not None:
                return _normalize_internal_convection_model(sim_opt.get(key))
        bp = building_object.get("building_parameters", {}) or {}
        for key in ("internal_convection_model", "h_ci_model", "convective_internal_model"):
            if bp.get(key) is not None:
                return _normalize_internal_convection_model(bp.get(key))

    return "table"


def _surface_tilt_for_internal_convection(surface: dict) -> float:
    ori = surface.get("orientation", {}) if isinstance(surface, dict) else {}
    if isinstance(ori, dict):
        tilt = ori.get("tilt", None)
        if tilt is not None:
            try:
                return float(tilt)
            except Exception:
                pass

    bnd = str(surface.get("boundary", "OUTDOORS")).upper() if isinstance(surface, dict) else "OUTDOORS"
    ori_tag = str(surface.get("ISO52016_orientation_string", "")).upper() if isinstance(surface, dict) else ""
    if bnd == "GROUND":
        return 180.0
    if ori_tag in {"NV", "EV", "SV", "WV"}:
        return 90.0
    if ori_tag in {"HF"}:
        return 180.0
    if ori_tag in {"HR"}:
        return 0.0
    if ori_tag == "HOR":
        return 0.0
    return 90.0


def _surface_tilt_for_external_radiation(surface: dict) -> float:
    return _surface_tilt_for_internal_convection(surface)


def _surface_roughness_index(surface: dict) -> int:
    """
    Convert roughness tag to EnergyPlus-like roughness index (1..6).
    1=VeryRough, 2=Rough, 3=MediumRough, 4=MediumSmooth, 5=Smooth, 6=VerySmooth.
    """
    if not isinstance(surface, dict):
        return 3
    tag = str(
        surface.get("roughness", surface.get("surface_roughness", "MediumRough"))
    ).strip().lower()
    table = {
        "veryrough": 1,
        "very_rough": 1,
        "rough": 2,
        "mediumrough": 3,
        "medium_rough": 3,
        "mediumsmooth": 4,
        "medium_smooth": 4,
        "smooth": 5,
        "verysmooth": 6,
        "very_smooth": 6,
    }
    return int(table.get(tag, 3))


_DOE2_COEFF = {
    1: (11.58, 5.894, 0.0),    # VeryRough
    2: (12.49, 4.065, 0.028),  # Rough
    3: (10.79, 4.192, 0.0),    # MediumRough
    4: (8.23, 4.0, -0.057),    # MediumSmooth
    5: (10.22, 3.1, 0.0),      # Smooth
    6: (8.23, 3.33, -0.036),   # VerySmooth
}

_BLAST_COEFF = {
    1: (5.15, 3.78, 0.0),
    2: (3.17, 3.78, 0.0),
    3: (2.19, 3.78, 0.0),
    4: (1.67, 3.78, 0.0),
    5: (1.52, 3.78, 0.0),
    6: (1.13, 3.78, 0.0),
}

_MOWITT_VERT = (0.84, 1.7)   # Ct, a
_MOWITT_HORIZ = (9.4, 2.0)   # Ct, a


def _natural_convection_external_w_m2k(dT_abs: float, tilt_deg: float) -> float:
    dT = max(0.0, float(dT_abs))
    if dT <= 1.0e-6:
        return 0.0
    tilt = float(tilt_deg)
    coeff = dT ** (1.0 / 3.0)
    if abs(tilt - 90.0) < 10.0:
        return 1.31 * coeff
    if tilt < 10.0:
        return 1.52 * coeff
    return 0.76 * coeff


def _normalize_external_convection_model(model_raw) -> str:
    if model_raw is None:
        return "table"
    model = str(model_raw).strip().lower().replace("-", "_")
    aliases = {
        "default": "table",
        "iso": "table",
        "fixed": "table",
        "constant": "table",
        "simple": "simplecombined",
        "simple_combined": "simplecombined",
        "doe_2": "doe2",
        "mo_witt": "mowitt",
    }
    model = aliases.get(model, model)
    if model not in {"table", "doe2", "mowitt", "blast", "simplecombined"}:
        raise ValueError(
            "external_convection_model must be one of: "
            "'table', 'doe2', 'mowitt', 'blast', 'simplecombined'."
        )
    return model


def _resolve_external_convection_model(building_object, model_override=None) -> str:
    if model_override is not None:
        return _normalize_external_convection_model(model_override)

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in (
            "external_convection_model",
            "h_ce_model",
            "convective_external_model",
            "external_convection_algorithm",
        ):
            if sim_opt.get(key) is not None:
                return _normalize_external_convection_model(sim_opt.get(key))
        bp = building_object.get("building_parameters", {}) or {}
        for key in (
            "external_convection_model",
            "h_ce_model",
            "convective_external_model",
            "external_convection_algorithm",
        ):
            if bp.get(key) is not None:
                return _normalize_external_convection_model(bp.get(key))

    return "table"


def _resolve_external_convection_h_min(building_object, h_min_override=None) -> float:
    if h_min_override is not None:
        try:
            return max(0.1, float(h_min_override))
        except Exception:
            return 2.0

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in ("external_convection_h_min", "h_ce_min"):
            if sim_opt.get(key) is not None:
                try:
                    return max(0.1, float(sim_opt.get(key)))
                except Exception:
                    pass
        bp = building_object.get("building_parameters", {}) or {}
        for key in ("external_convection_h_min", "h_ce_min"):
            if bp.get(key) is not None:
                try:
                    return max(0.1, float(bp.get(key)))
                except Exception:
                    pass
    return 2.0


def _dynamic_external_convection_h(
    surface: dict,
    T_surf_C: float,
    T_air_C: float,
    u_wind_ms: float,
    model: str,
    h_min: float = 2.0,
    fallback_h_ce: float = 20.0,
) -> float:
    model_n = _normalize_external_convection_model(model)
    h_fallback = max(0.0, float(fallback_h_ce))
    if model_n == "table":
        return h_fallback

    u = max(0.0, float(u_wind_ms))
    dT = abs(float(T_air_C) - float(T_surf_C))
    tilt = _surface_tilt_for_external_radiation(surface)
    h = np.nan

    try:
        if model_n == "doe2":
            ri = _surface_roughness_index(surface)
            D, E, F = _DOE2_COEFF.get(ri, _DOE2_COEFF[3])
            h_forced = D + E * u + F * u ** 2
            h_nat = _natural_convection_external_w_m2k(dT, tilt)
            h = max(h_forced, h_nat)
        elif model_n == "mowitt":
            if abs(float(tilt) - 90.0) < 30.0:
                Ct, a = _MOWITT_VERT
            else:
                Ct, a = _MOWITT_HORIZ
            h_nat_sq = (Ct * dT ** (1.0 / 3.0)) ** 2 if dT > 1.0e-6 else 0.0
            h_forced_sq = (a * u ** 0.89) ** 2
            h = np.sqrt(h_nat_sq + h_forced_sq)
        elif model_n == "blast":
            ri = _surface_roughness_index(surface)
            W1, W2, _ = _BLAST_COEFF.get(ri, _BLAST_COEFF[3])
            h_forced = W1 + W2 * u
            if float(tilt) < 10.0 and dT > 1.0e-6:
                denom = 7.238 - abs(np.cos(np.radians(float(tilt))))
                h_nat = 9.482 * (dT ** (1.0 / 3.0)) / max(0.5, denom)
            else:
                h_nat = _natural_convection_external_w_m2k(dT, tilt)
            h = h_forced + h_nat
        elif model_n == "simplecombined":
            h = 4.0 + 4.0 * u
    except Exception:
        h = np.nan

    if not np.isfinite(h):
        h = h_fallback
    return max(float(h_min), float(h))


def _normalize_external_radiation_model(model_raw) -> str:
    if model_raw is None:
        return "table"
    model = str(model_raw).strip().lower().replace("-", "_")
    aliases = {
        "default": "table",
        "iso": "table",
        "fixed": "table",
        "constant": "table",
        "dynamic_h_re": "dynamic",
        "h_re_dynamic": "dynamic",
    }
    model = aliases.get(model, model)
    if model not in {"table", "dynamic"}:
        raise ValueError("external_radiation_model must be 'table' or 'dynamic'.")
    return model


def _resolve_external_radiation_model(building_object, model_override=None) -> str:
    if model_override is not None:
        return _normalize_external_radiation_model(model_override)

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in ("external_radiation_model", "h_re_model", "radiative_external_model"):
            if sim_opt.get(key) is not None:
                return _normalize_external_radiation_model(sim_opt.get(key))
        bp = building_object.get("building_parameters", {}) or {}
        for key in ("external_radiation_model", "h_re_model", "radiative_external_model"):
            if bp.get(key) is not None:
                return _normalize_external_radiation_model(bp.get(key))

    return "table"


def _normalize_sky_temperature_model(model_raw) -> str:
    if model_raw is None:
        return "berdahl_fromberg"
    model = str(model_raw).strip().lower().replace("-", "_")
    aliases = {
        "default": "berdahl_fromberg",
        "berdahl": "berdahl_fromberg",
        "fromberg": "berdahl_fromberg",
        "bf": "berdahl_fromberg",
        "epw": "epw_ir",
        "ir": "epw_ir",
        "infrared": "epw_ir",
    }
    model = aliases.get(model, model)
    if model not in {"berdahl_fromberg", "swinbank", "epw_ir"}:
        raise ValueError("sky_temperature_model must be 'berdahl_fromberg', 'swinbank', or 'epw_ir'.")
    return model


def _resolve_sky_temperature_model(building_object, model_override=None) -> str:
    if model_override is not None:
        return _normalize_sky_temperature_model(model_override)

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in ("sky_temperature_model", "t_sky_model"):
            if sim_opt.get(key) is not None:
                return _normalize_sky_temperature_model(sim_opt.get(key))
        bp = building_object.get("building_parameters", {}) or {}
        for key in ("sky_temperature_model", "t_sky_model"):
            if bp.get(key) is not None:
                return _normalize_sky_temperature_model(bp.get(key))

    return "berdahl_fromberg"


def _resolve_default_external_emissivity(building_object, epsilon_override=None) -> float:
    if epsilon_override is not None:
        try:
            return float(np.clip(float(epsilon_override), 0.01, 1.0))
        except Exception:
            return 0.9

    if isinstance(building_object, dict):
        sim_opt = _get_simulation_options(building_object)
        for key in ("external_emissivity_default", "surface_emissivity_external_default"):
            if sim_opt.get(key) is not None:
                try:
                    return float(np.clip(float(sim_opt.get(key)), 0.01, 1.0))
                except Exception:
                    pass
        bp = building_object.get("building_parameters", {}) or {}
        for key in ("external_emissivity_default", "surface_emissivity_external_default"):
            if bp.get(key) is not None:
                try:
                    return float(np.clip(float(bp.get(key)), 0.01, 1.0))
                except Exception:
                    pass

    return 0.9


def _surface_external_emissivity(surface: dict, epsilon_default: float = 0.9) -> float:
    if isinstance(surface, dict):
        for key in ("external_emissivity", "emissivity_external", "epsilon_external", "epsilon"):
            if surface.get(key) is not None:
                try:
                    return float(np.clip(float(surface.get(key)), 0.01, 1.0))
                except Exception:
                    pass
    return float(np.clip(float(epsilon_default), 0.01, 1.0))


def _dewpoint_from_t_rh(T_air_C, RH_pct):
    T = np.asarray(T_air_C, dtype=float)
    RH = np.asarray(RH_pct, dtype=float)
    RH = np.clip(RH, 1.0e-3, 100.0)
    a = 17.625
    b = 243.04
    gamma = np.log(RH / 100.0) + (a * T) / (b + T)
    return (b * gamma) / (a - gamma)


def _sky_temperature_from_weather(
    sim_df: pd.DataFrame,
    tstep: int,
    sky_temperature_model: str,
) -> float:
    T_air = float(pd.to_numeric(sim_df["T2m"], errors="coerce").iloc[tstep])
    if not np.isfinite(T_air):
        T_air = 0.0

    model = _normalize_sky_temperature_model(sky_temperature_model)
    if model == "epw_ir":
        if "IR(h)" in sim_df.columns:
            ir_h = float(pd.to_numeric(sim_df["IR(h)"], errors="coerce").iloc[tstep])
            if np.isfinite(ir_h) and ir_h > 0.0:
                return float(np.power(ir_h / SIGMA, 0.25) - 273.15)
        model = "berdahl_fromberg"

    if model == "swinbank":
        return float(T_sky_swinbank(T_air))

    # Berdahl-Fromberg fallback.
    dew_cols = ("Tdew", "T_dew", "T_dew_C", "dew_point", "temp_dew")
    T_dew = np.nan
    for c in dew_cols:
        if c in sim_df.columns:
            T_dew = float(pd.to_numeric(sim_df[c], errors="coerce").iloc[tstep])
            break
    if not np.isfinite(T_dew):
        if "RH" in sim_df.columns:
            RH = float(pd.to_numeric(sim_df["RH"], errors="coerce").iloc[tstep])
            if not np.isfinite(RH):
                RH = 70.0
            T_dew = float(_dewpoint_from_t_rh(T_air, RH))
        else:
            T_dew = T_air - 10.0
    return float(T_sky_berdahl_fromberg(T_air, T_dew))


def _dynamic_external_radiative_h_and_ref(
    surface: dict,
    T_surf_C: float,
    T_sky_C: float,
    T_air_C: float,
    epsilon: float = 0.9,
    T_gnd_C: float | None = None,
) -> tuple[float, float]:
    if T_gnd_C is None:
        T_gnd_C = T_air_C
    tilt = _surface_tilt_for_external_radiation(surface)
    tilt_r = np.radians(float(tilt))
    F_sky = (1.0 + np.cos(tilt_r)) / 2.0
    F_gnd = (1.0 - np.cos(tilt_r)) / 2.0
    h_sky = float(h_re_sky(T_surf_C, T_sky_C, epsilon))
    h_gnd = float(h_re_sky(T_surf_C, T_gnd_C, epsilon))
    h_re_eq = max(0.0, F_sky * h_sky + F_gnd * h_gnd)
    if h_re_eq <= 0.0:
        return 0.0, float(T_air_C)
    T_ref = (F_sky * h_sky * float(T_sky_C) + F_gnd * h_gnd * float(T_gnd_C)) / h_re_eq
    return float(h_re_eq), float(T_ref)


def _internal_h_ci_value(
    surface: dict,
    model: str,
    t_air_c: float,
    t_surf_c: float,
    fallback_h_ci: float,
) -> float:
    h_table = max(0.0, float(fallback_h_ci))
    if model != "tarp":
        return h_table
    try:
        tilt = _surface_tilt_for_internal_convection(surface)
        return float(h_ci_tarp(t_air_c, t_surf_c, tilt))
    except Exception:
        return h_table

# ===============================================================================================
#                                       MODULES SIMULATIONS
# ===============================================================================================

#                                       ISO 52010
# ===============================================================================================


class ISO52010:
    solar_constant = 1370  # [W/m2]
    K_eps = 1.104  # [rad^-3]
    Perez_coefficients_matrix = np.array(
        [
            [1.065, -0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
            [1.230, 0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
            [1.500, 0.330, 0.487, -0.221, 0.055, -0.064, -0.026],
            [1.950, 0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
            [2.800, 0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
            [4.500, 1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
            [6.200, 1.060, -1.600, -0.359, 0.264, -1.127, 0.131],
            [99999, 0.678, -0.327, -0.250, 0.156, -1.377, 0.251],
        ]
    )  # # Values for clearness index and brightness coefficients as function of clearness parameter

    def __init__(self):
        pass

    # GET DATA FROM PVGIS
    @classmethod
    def get_tmy_data_pvgis(cls, building_object) -> WeatherDataResult:
        from timezonefinder import TimezoneFinder  # lazy: avoids ~0.7s cold import
        """
        Get Weather data from pvgis API

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        :return:
            * *elevation*: altitude of specifici location (type: **float**)
            * *weather_data*: dataframe with wetaher parameters (e.g. outdoor temperature, outdoor relative humidity, etc.) (type: **pd.DataFrame**)
            * *utc_offset*: refers to the difference in time between Coordinated Universal Time (UTC) and the local time of a specific location (type: **int**)
            * *latitude*: latitude of the building place (type: **float**)
            * *longitude*: longitude of the building place (type: **float**)

        .. note::
            In case only weather data is desired, the ``building_object`` can only have the **latitude** and **longitude** parameters.

        """
        from timezonefinder import TimezoneFinder  # lazy: avoids ~0.7s cold import

        # Connection to PVGIS API to get weather data
        if isinstance(building_object, dict):
            latitude = building_object["building"]["latitude"]
            longitude = building_object["building"]["longitude"]
        else:
            latitude = building_object.__getattribute__("latitude")
            longitude = building_object.__getattribute__("longitude")
        url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={latitude}&lon={longitude}&outputformat=json&browser=1"
        response = requests.request("GET", url, allow_redirects=True)
        data = response.json()
        df_weather = pd.DataFrame(data["outputs"]["tmy_hourly"])

        # Time data into UTC
        df_weather["time(UTC)"] = [
            dt.datetime.strptime(x, "%Y%m%d:%H%M") for x in df_weather["time(UTC)"]
        ]

        # Change year to 2019 before sorting by date, because the months in the tmy file are stitched together from different years
        df_weather["time(UTC)"] = df_weather["time(UTC)"].apply(
            lambda x: x.replace(year=2019)
        )

        # Order data in date ascending order
        df_weather = df_weather.sort_values(by="time(UTC)")
        df_weather.index = df_weather["time(UTC)"]
        del df_weather["time(UTC)"]

        # Elevation is not needed for the energy demand calculation, only for the PV optimization
        loc_elevation = data["inputs"]["location"]["elevation"]
        latitude_ = data["inputs"]["location"]["latitude"]
        longitude_ = data["inputs"]["location"]["longitude"]
        # TIMEZONE FINDER
        tf = TimezoneFinder()
        utcoffset_in_hours = int(
            timezone(tf.timezone_at(lng=longitude, lat=latitude))
            .localize(df_weather.index[0])
            .utcoffset()
            .total_seconds()
            / 3600.0
        )

        return WeatherDataResult(
            elevation=loc_elevation,
            weather_data=df_weather,
            utc_offset=utcoffset_in_hours,
            latitude=latitude_,
            longitude=longitude_,
        )

    @classmethod
    def get_tmy_data_climatedataforbuildings(
        cls,
        building_object,
        dataset="EU",
        period="1991-2020",
        data_type="tmy",
        out_dir=None,
        in_memory=True, 
    ):
        """
        Get Weather data from climatedataforbuildings.eu and save to an EPW file.

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param dataset: dataset to use ("EU" or "NO").
        :param period: reference period ("1991-2020" or "2006-2020").
        :param data_type: data type to use (only "tmy" supported).
        :param out_dir: output directory for the downloaded EPW file.
        :param in_memory: if True, return an in-memory EPW buffer instead of saving to disk. True tends to be faster. 
        :param user_agent: custom User-Agent header for requests.

        :return: path of the downloaded EPW file or an in-memory buffer (type: **Path** or **io.BytesIO**)
        """
        # Extract latitude and longitude from building object
        if isinstance(building_object, dict):
            latitude = building_object["building"]["latitude"]
            longitude = building_object["building"]["longitude"]
        else:
            latitude = building_object.__getattribute__("latitude")
            longitude = building_object.__getattribute__("longitude")

        dataset = str(dataset).upper()
        data_type = str(data_type).lower()
        period = str(period)
        # Validate inputs
        valid_types = {"tmy"}
        valid_datasets = {"EU", "NO"}
        valid_periods = {"1991-2020", "2006-2020"}
        
        if data_type not in valid_types:
            raise ValueError("type must be tmy")
        if dataset not in valid_datasets:
            raise ValueError("dataset must be EU or NO")
        if period not in valid_periods:
            raise ValueError("period must be 1991-2020 or 2006-2020")

        # Set User-Agent header
        user_agent = f"pybuildingenergy"
        headers = {"User-Agent": user_agent}
        # Haversine formula to calculate distance between two lat/lon points
        def _haversine_km(lat1, lon1, lat2, lon2):
            radius_km = 6371
            to_rad = math.radians
            dlat = to_rad(lat2 - lat1)
            dlon = to_rad(lon2 - lon1)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(to_rad(lat1))
                * math.cos(to_rad(lat2))
                * math.sin(dlon / 2) ** 2
            )
            return 2 * radius_km * math.asin(math.sqrt(a))
        # Fetch index file for locations
        index_url = f"https://www.climatedataforbuildings.eu/api/{data_type}-{dataset.lower()}.json"
        response = requests.get(index_url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch index file from {index_url}: HTTP Error {response.status_code}"
            )
        points = response.json()
        # Find closest point
        best = None
        best_dist = float("inf")
        for point in points:
            dist = _haversine_km(latitude, longitude, point["lat"], point["lon"])
            if dist < best_dist:
                best_dist = dist
                best = point

        if best is None:
            raise RuntimeError(f"No points returned by {index_url}")
        # Download EPW file for closest point
        file_path = best["files"].get(period)
        if not file_path:
            raise RuntimeError(f"No file available for period {period} in {index_url}")

        epw_url = f"https://www.climatedataforbuildings.eu/FA{data_type.upper()}/{file_path}"
        response = requests.get(epw_url, headers=headers, stream=True, timeout=60)
        # Save to disk or return in-memory buffer
        try:
            response.raise_for_status()
            # Return in-memory buffer
            if in_memory:
                # EPW is a text format; provide a text buffer
                return io.StringIO(response.content.decode(errors="ignore"))
            # Save to disk
            if out_dir is None:
                out_dir = Path.home() / "Downloads"
            else:
                out_dir = Path(out_dir)
            # Create output directory if it doesn't exist
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / Path(file_path).name
            # Write to file with permission error handling
            try:
                # Write to specified output directory
                with out_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
            except PermissionError: # Fallback to temp directory
                out_dir = Path(tempfile.gettempdir()) / "pybuildingenergy"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / Path(file_path).name
                with out_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
        finally: # Ensure response is closed
            response.close()

        return out_path

    # GET WEATHER DATA FROM .epw FILE
    @classmethod
    def get_tmy_data_epw(cls, path_weather_file):
        from pvlib.iotools import epw  # lazy: avoids ~1.9s cold import
        """
        Get Wetaher data from epw file

        :param path_weather_file: path of the .epw weather file or an in-memory buffer.

        :return:
            * *elevation*: altitude of specifici location (type: **float**)
            * *weather_data*: dataframe with wetaher parameters (e.g. outdoor temperature, outdoor relative humidity, etc.) (type: **pd.DataFrame**)
            * *utc_offset*: refers to the difference in time between Coordinated Universal Time (UTC) and the local time of a specific location (type: **int**)
            * *latitude*: latitude of the building place (type: **float**)
            * *longitude*: longitude of the building place (type: **float**)
        """
        from pvlib.iotools import epw  # lazy: avoids ~1.9s cold import

        # Read EPW file
        if isinstance(path_weather_file, (bytes, bytearray)):
            path_weather_file = io.StringIO(
                bytes(path_weather_file).decode(errors="ignore")
            )
        elif isinstance(path_weather_file, io.BytesIO):
            path_weather_file.seek(0)
            path_weather_file = io.StringIO(
                path_weather_file.read().decode(errors="ignore")
            )
        weather_data = epw.read_epw(path_weather_file)

        # Weather data filter in a format to be used by ISO52016 in a csv
        df_weather_time_series = weather_data[0]
        tmy_weather_data = df_weather_time_series.loc[
            :,
            [
                "temp_air",
                "relative_humidity",
                "ghi",
                "dni",
                "dhi",
                "ghi_infrared",
                "wind_speed",
                "wind_direction",
                "atmospheric_pressure",
            ],
        ]
        tmy_weather_data.index.name = "time(UTC)"

        # Convert DatetimeIndex to the desired format
        tmy_weather_data.index = df_weather_time_series.index.tz_convert(
            None
        )  # Remove timezone information
        tmy_weather_data.index = df_weather_time_series.index.strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Format datetime index
        tmy_weather_data.index = pd.DatetimeIndex(tmy_weather_data.index)
        #
        tmy_weather_data.columns = [
            "T2m",
            "RH",
            "G(h)",
            "Gb(n)",
            "Gd(h)",
            "IR(h)",
            "WS10m",
            "WD10m",
            "SP",
        ]

        #
        location_info = weather_data[1]
        elevation = location_info["altitude"]
        utcoffset_in_hours = int(location_info["TZ"])
        latitude_ = location_info["latitude"]
        longitude_ = location_info["longitude"]

        return WeatherDataResult(
            elevation=elevation,
            weather_data=tmy_weather_data,
            utc_offset=utcoffset_in_hours,
            latitude=latitude_,
            longitude=longitude_,
        )

    @classmethod
    def Shading_reduction_factor_window(
        cls,
        solar_altitude_angle,
        solar_azimuth_angle,
        I_dir_tot,
        I_dif_tot,
        calendar,
        n_timesteps,
        orientation,
        building_object
    ):
        """
        Calculates the shading reduction factor for each window.
        Returns a Shading_Reduction_factor_window object, or None if not applicable.
        """

        orientation = str(orientation).upper() if orientation is not None else None

        # 1) Trivial / invalid cases
        if orientation is None:
            # No orientation specified: either returns None, or you may decide to iterate over all of them.
            return None

        if orientation == "HOR":
            # Horizontal: no vertical window to calculate
            return None

        # 2) Use the project-wide geographical azimuth convention:
        #    N=0, E=90, S=180, W=270
        #    (solar_azimuth_angle from ISO52010 is converted below)
        orientation_lookup = {
            "NV": 0.0,
            "EV": 90.0,
            "SV": 180.0,
            "WV": 270.0,
        }
        if orientation not in orientation_lookup:
            raise ValueError(f"Unknown orientation '{orientation}' passed to shading calculation.")

        orientation_angle = float(orientation_lookup[orientation])

        # 3) Transparent window filter by cardinal orientation label.
        #    This keeps filtering independent from the azimuth convention
        #    used for gamma in shading_reduction_factor.
        def _surface_orientation_label(surface):
            ori_tag = str(surface.get("ISO52016_orientation_string", "")).upper()
            if ori_tag in {"HOR", "NV", "EV", "SV", "WV"}:
                return ori_tag

            ori = surface.get("orientation", {}) or {}
            az = ori.get("azimuth", None)
            tilt = ori.get("tilt", None)
            try:
                az_f = float(az) % 360.0
                tilt_f = float(tilt)
            except (TypeError, ValueError):
                return None

            if np.isclose(tilt_f, 0.0, atol=1e-6):
                return "HOR"
            if np.isclose(tilt_f, 90.0, atol=1e-6):
                candidates = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
                labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
                diffs = np.abs(((az_f - candidates + 180.0) % 360.0) - 180.0)
                return str(labels[int(np.argmin(diffs))])
            return "HOR" if tilt_f < 45.0 else "SV"

        def _matches_orientation(surface):
            return _surface_orientation_label(surface) == orientation

        filtered_windows = [
            s for s in building_object.get("building_surface", [])
            if s.get("type") == "transparent" and _matches_orientation(s)
        ]
        if not filtered_windows:
            return None

        # 4) Calculate factors for each window
        F_sh_dir_df = pd.DataFrame(index=calendar.index)

        for window in filtered_windows:
            F_sh_dir_k_ts_hour = pd.Series(np.zeros(n_timesteps), index=calendar.index, dtype=float)
            h_k_sun_t_hour     = pd.Series(np.zeros(n_timesteps), index=calendar.index, dtype=float)

            for i in range(n_timesteps):
                # Protect from NaN in angle values
                alpha_deg = float(np.degrees(solar_altitude_angle.iloc[i])) if pd.notna(solar_altitude_angle.iloc[i]) else 0.0
                # solar_azimuth_angle is produced by ISO52010 in ISO convention
                # (S=0, E=+90, W=-90). Convert to geographical (N=0, E=90, S=180, W=270).
                if pd.notna(solar_azimuth_angle.iloc[i]):
                    phi_iso_deg = float(np.degrees(solar_azimuth_angle.iloc[i]))
                    phi_deg = (180.0 - phi_iso_deg) % 360.0
                else:
                    phi_deg = 0.0

                F_sh_dir_k_t = shading_reduction_factor(
                    alpha_sol_t=alpha_deg,
                    phi_sol_t=phi_deg,
                    beta_k_t=90,
                    gamma_k_t=orientation_angle,
                    D_k_ovh_q=window.get("overhang_proprieties", {}).get("width_of_horizontal_overhangs"),
                    L_k_ovh_q=window.get("width_or_distance_of_shading_elements"),
                    elements_shading_type=window.get("shading_type"),
                    H_k=window.get("height"),
                    W_k=window.get("width"),
                )

                # Warning: I assume F_sh_dir_k_t = (dir_factor, sun_height)
                fatt_dir = F_sh_dir_k_t[0] if F_sh_dir_k_t is not None else 0.0
                if fatt_dir > 0:
                    I_dir = float(I_dir_tot.iloc[i]) if pd.notna(I_dir_tot.iloc[i]) else 0.0
                    I_dif = float(I_dif_tot.iloc[i]) if pd.notna(I_dif_tot.iloc[i]) else 0.0
                    denom = I_dir + I_dif
                    F_sh_obst_k_t = 0.0 if denom == 0 else (fatt_dir * I_dir + I_dif) / denom
                else:
                    F_sh_obst_k_t = 0.0

                F_sh_dir_k_ts_hour.iat[i] = F_sh_obst_k_t
                h_k_sun_t_hour.iat[i]     = F_sh_dir_k_t[1] if F_sh_dir_k_t is not None else 0.0

            name = window.get('name', 'unknown')
            F_sh_dir_df[f"W_{name}"]     = F_sh_dir_k_ts_hour
            F_sh_dir_df[f"H_sun_{name}"] = h_k_sun_t_hour

        return Shading_Reduction_factor_window(shading_reduction_factor_window=F_sh_dir_df)



    # @classmethod
    # def Shading_reduction_factor_window(
    #     cls,
    #     solar_altitude_angle,
    #     solar_azimuth_angle,
    #     I_dir_tot,
    #     I_dif_tot,
    #     calendar,
    #     n_timesteps,
    #     orientation,
    #     building_object
    # ) -> pd.DataFrame:
    #     """
    #     Calculate the shading reduction factor for each window.
    #     :param solar_altitude_angle: Solar altitude angle
    #     :param solar_azimuth_angle: Solar azimuth angle
    #     :param I_dir_tot: Direct solar radiation
    #     :param I_dif_tot: Diffuse solar radiation
    #     :param calendar: Calendar
    #     :param n_timesteps: Number of timesteps
    #     :param orientation: Orientation of the window
    #     :param building_object: Building object

    #     :return: Shading reduction factor for each window
    #     """
    #     orientation_df  =pd.DataFrame({
    #         "name": ['NV', 'SV', 'EV', 'WV'],
    #         "angle": [0,180,90,270]
    #     })
    #     try:
    #         # Extract data from trasparent surfaces 
    #         orientation_angle = orientation_df.loc[orientation_df['name'] == orientation,'angle'].values[0]
    #         filtered_windows = [
    #             s for s in building_object.get("building_surface", [])
    #             if s.get("type") == "transparent"
    #             and s.get("orientation", {}).get("azimuth") == orientation_angle
    #         ]
    #         F_sh_dir_df = pd.DataFrame(index=calendar.index)
    #         for window in filtered_windows:
    #             # orientation_angle = orientation_df.loc[orientation_df['name'] == orientation_w,'angle'].values[0]
    #             F_sh_dir_k_ts_hour = pd.Series(np.zeros(n_timesteps), index=calendar.index)
    #             h_k_sun_t_hour = pd.Series(np.zeros(n_timesteps), index=calendar.index)                    
    #             for i in range(n_timesteps):
    #                 F_sh_dir_k_t = shading_reduction_factor(
    #                     alpha_sol_t=np.degrees(solar_altitude_angle.iloc[i]),
    #                     phi_sol_t=np.degrees(solar_azimuth_angle.iloc[i]),
    #                     beta_k_t=90,
    #                     gamma_k_t=orientation_angle,
    #                     D_k_ovh_q=window.get("overhang_proprieties", {}).get("width_of_horizontal_overhangs"),
    #                     L_k_ovh_q=window.get("width_or_distance_of_shading_elements"),
    #                     elements_shading_type=window.get("shading_type"),
    #                     H_k=window.get("height"),
    #                     W_k=window.get("width")
    #                 )
    #                 if F_sh_dir_k_t[0] > 0:
    #                     if (I_dir_tot.iloc[i] + I_dif_tot.iloc[i]) == 0:
    #                         F_sh_obst_k_t = 0
    #                     else:       
    #                         F_sh_obst_k_t = (F_sh_dir_k_t[0] * I_dir_tot.iloc[i] + I_dif_tot.iloc[i]) / (I_dir_tot.iloc[i] + I_dif_tot.iloc[i])
    #                         # F_sh_obst_k_t = (F_sh_dir_k_t[0] * I_dir_tot[i] + I_dif_tot[i]) / (I_dir_tot[i] + I_dif_tot[i])
    #                 else:
    #                     F_sh_obst_k_t = 0
    #                 F_sh_dir_k_ts_hour.iloc[i] = F_sh_obst_k_t
    #                 h_k_sun_t_hour.iloc[i] = F_sh_dir_k_t[1]
    #             F_sh_dir_df[f"W_{window.get('name')}"] = F_sh_dir_k_ts_hour
    #             F_sh_dir_df[f"H_sun_{window.get('name')}"] = h_k_sun_t_hour
    #         return Shading_Reduction_factor_window(shading_reduction_factor_window=F_sh_dir_df)
    #     except:
    #         pass

 

    @classmethod
    def Solar_irradiance_calculation(
        cls,
        latitude_deg,
        longitude_deg,
        timezone,
        beta_ic_deg,
        gamma_ic_deg,
        DHI,
        DNI,
        ground_solar_reflectivity,
        calendar,
        n_timesteps,
        n_days,
        building_object
    ):
        """
        ISO 52010-1:2017 specifies a calculation procedure for the conversion of climatic data for energy calculations.
        The main element in ISO 52010-1:2017 is the calculation of solar irradiance on a surface with arbitrary orientation and tilt.
        A simple method for conversion of solar irradiance to illuminance is also provided.
        The solar irradiance and illuminance on an arbitrary surface are applicable as input for energy and daylighting calculations, for building elements
        (such as roofs, facades and windows) and for components of technical building systems (such as thermal solar collectors, PV panels)

        :param timezone_utc: The UTC offset (or time offset) is an amount of time subtracted from or added to Coordinated Universal Time (UTC) time to specify the local solar time. (type: **int**)
        :param beta_ic_deg: Tilt angle of inclined surface from horizontal is measured upwards facing (hor.facing upw=0, vert=90, hor.facing down=180). (type: **int**)
        :param gamma_ic_deg: Orientation angle of the inclined (ic) surface, expressed as the geographical azimuth angle of the horizontal projection of the inclined (S=0, E=pos., W=neg.) surface normal. (type: **int**)
        :param DHI: Diffuse horizontal irradiance - [W/m2] (type: **float**)
        :param DNI: Direct (beam) irradiance - [W/m2] (type: **float**)
        :param ground_solar_reflectivity: solar reflectivity of the ground (type: **float**)
        :param calendar: dataframe with days of the year (from 1 to 365) and hour of day (1 to 24) (type: **int**)
        :param n_timesteps: number of hour in a year = 8760 (type: **int**)

        :return: **solar_irradiance**: hourly solar irradiance (type: **pd.DataFrame**)
        """

        beta_ic = np.radians(beta_ic_deg)
        gamma_ic = np.radians(gamma_ic_deg)
        DayWeekJan1 = 1  # doy of week of 1 January; 1=Monday, 7=Sunday
        latitude = np.radians(latitude_deg)
        earth_orbit_deviation_deg = 360 / n_days * calendar["day of year"]  # [deg]
        earth_orbit_deviation = np.radians(earth_orbit_deviation_deg)  # [rad]
        declination_deg = (
            0.33281
            - 22.984 * np.cos(earth_orbit_deviation)
            - 0.3499 * np.cos(2 * earth_orbit_deviation)
            - 0.1398 * np.cos(3 * earth_orbit_deviation)
            + 3.7872 * np.sin(earth_orbit_deviation)
            + 0.03205 * np.sin(2 * earth_orbit_deviation)
            + 0.07187 * np.sin(3 * earth_orbit_deviation)
        )  # [deg]
        declination = np.radians(declination_deg)
        t_eq = Equation_of_time(calendar["day of year"])  # [min]
        time_shift = timezone - (longitude_deg / 15)  # [h]
        solar_time = calendar["hour of day"] - t_eq / 60 - time_shift  # [h]
        hour_angle_deg = Hour_angle_calc(solar_time)  # [deg]
        hour_angle = np.radians(hour_angle_deg)
        solar_altitude_angle_sin = np.sin(declination) * np.sin(latitude) + np.cos(declination) * np.cos(latitude) * np.cos(hour_angle)
        solar_altitude_angle = np.arcsin(np.sin(declination) * np.sin(latitude)+ np.cos(declination) * np.cos(latitude) * np.cos(hour_angle))
        solar_altitude_angle[solar_altitude_angle < 1e-4] = 0
        solar_incidence_angle_ic_cos = (
            np.sin(declination) * np.sin(latitude) * np.cos(beta_ic)
            - np.sin(declination)
            * np.cos(latitude)
            * np.sin(beta_ic)
            * np.cos(gamma_ic)
            + np.cos(declination)
            * np.cos(latitude)
            * np.cos(beta_ic)
            * np.cos(hour_angle)
            + np.cos(declination)
            * np.sin(latitude)
            * np.sin(beta_ic)
            * np.cos(gamma_ic)
            * np.cos(hour_angle)
            + np.cos(declination)
            * np.sin(beta_ic)
            * np.sin(gamma_ic)
            * np.sin(hour_angle)
        )  # ic=inclined surface
        solar_incidence_angle_ic = np.arccos(solar_incidence_angle_ic_cos)
        air_mass = Air_mass_calc(solar_altitude_angle)  # [-]
        solar_constant = 1370  # [W/m2]
        I_ext = solar_constant * (
            1 + 0.033 * np.cos(earth_orbit_deviation)
        )  # extra-terrestrial radiation [W/m2]
        solar_zenith_angle = np.pi / 2 - solar_altitude_angle
        solar_azimuth_angle_aux_1_sin = (
            np.cos(declination) * np.sin(np.pi - hour_angle)
        ) / np.cos(np.arcsin(solar_altitude_angle_sin))
        solar_azimuth_angle_aux_1_cos = (
            np.cos(latitude) * np.sin(declination)
            + np.sin(latitude) * np.cos(declination) * np.cos(np.pi - hour_angle)
        ) / np.cos(np.arcsin(solar_altitude_angle_sin))
        solar_azimuth_angle_aux_2 = np.arcsin(
            np.cos(declination)
            * np.sin(np.pi - hour_angle)
            / np.cos(np.arcsin(solar_altitude_angle_sin))
        )
        solar_azimuth_angle = -(np.pi + solar_azimuth_angle_aux_2)
        mask = (solar_azimuth_angle_aux_1_cos > 0) & (
            solar_azimuth_angle_aux_1_sin >= 0
        )
        solar_azimuth_angle[mask] = np.pi - solar_azimuth_angle_aux_2[mask]
        mask = solar_azimuth_angle_aux_1_cos < 0
        solar_azimuth_angle[mask] = solar_azimuth_angle_aux_2[mask]
        a_perez = pd.Series(np.zeros(n_timesteps), index=calendar.index)
        mask = solar_incidence_angle_ic_cos > 0
        a_perez[mask] = solar_incidence_angle_ic_cos[mask]
        b_perez = np.maximum(
            np.cos(np.radians(85)) * np.ones(n_timesteps), np.cos(solar_zenith_angle)
        )
        clearness = pd.Series(999 * np.ones(n_timesteps), index=calendar.index)  # [-]
        K_eps = 1.104  # [rad^-3]
        mask = (
            DHI > 0
        )  # DHI=diffuse horizontal irradiance; DNI=direct (beam) normal irradiance; GHI=global horizontal irradiance
        clearness[mask] = (
            (DHI[mask] + DNI[mask]) / DHI[mask]
            + K_eps * np.float_power(solar_altitude_angle[mask], 3)
        ) / (1 + K_eps * np.float_power(solar_altitude_angle[mask], 3))
        sky_brightness = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [-]
        sky_brightness[mask] = air_mass[mask] * DHI[mask] / I_ext[mask]
        Perez_coefficients_matrix = np.array(
            [
                [1.065, -0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
                [1.230, 0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
                [1.500, 0.330, 0.487, -0.221, 0.055, -0.064, -0.026],
                [1.950, 0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
                [2.800, 0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
                [4.500, 1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
                [6.200, 1.060, -1.600, -0.359, 0.264, -1.127, 0.131],
                [99999, 0.678, -0.327, -0.250, 0.156, -1.377, 0.251],
            ]
        )  # see Excel implementation of ISO 52010-1; columns in this order: epsilon, f11, f12, f13, f21, f22, f23
        PerezF = np.zeros(
            (n_timesteps, 6)
        )  # columns in the order: F11, F12, F13, F21, F22, F23

        mask_DHI = DHI > 0
        for i in range(6):
            mask_clearness = clearness < Perez_coefficients_matrix[0, 0]
            mask = mask_DHI & mask_clearness
            PerezF[mask, i] = Perez_coefficients_matrix[0, i + 1]
            for j in range(6):
                mask_clearness = (clearness >= Perez_coefficients_matrix[j, 0]) & (
                    clearness < Perez_coefficients_matrix[j + 1, 0]
                )
                mask = mask_DHI & mask_clearness
                PerezF[mask, i] = Perez_coefficients_matrix[j + 1, i + 1]
            mask_clearness = clearness >= Perez_coefficients_matrix[6, 0]
            mask = mask_DHI & mask_clearness
            PerezF[mask, i] = Perez_coefficients_matrix[7, i + 1]

        PerezF1 = np.maximum(
            0,
            PerezF[:, 0]
            + PerezF[:, 1] * sky_brightness
            + PerezF[:, 2] * solar_zenith_angle,
        )
        PerezF2 = (
            PerezF[:, 3]
            + PerezF[:, 4] * sky_brightness
            + PerezF[:, 5] * solar_zenith_angle
        )
        I_dir = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [W/m2]
        mask = solar_incidence_angle_ic_cos > 0
        I_dir[mask] = DNI[mask] * solar_incidence_angle_ic_cos[mask]
        I_dif = pd.Series(np.zeros(n_timesteps), index=calendar.index)  # [W/m2]
        mask = DHI > 0
        I_dif[mask] = DHI[mask] * (
            (1 - PerezF1[mask]) * (1 + np.cos(beta_ic)) / 2
            + PerezF1[mask] * a_perez[mask] / b_perez[mask]
            + PerezF2[mask] * np.sin(beta_ic)
        )
        I_dif_ground = (
            (DHI + DNI * np.sin(solar_altitude_angle))
            * ground_solar_reflectivity
            * (1 - np.cos(beta_ic))
            / 2
        )  # [W/m2]
        I_circum = DHI * PerezF1 * a_perez / b_perez  # [W/m2]
        I_dif_tot = I_dif - I_circum + I_dif_ground
        I_dir_tot = I_dir + I_circum
        #
        I_tot = pd.DataFrame({"I_sol_tot":I_dif_tot + I_dir_tot})
        I_tot['I_sol_dif'] = I_dif_tot
        I_tot['I_sol_dir'] = I_dir_tot
        
        return Solar_irradiance(solar_irradiance=I_tot), solar_altitude_angle, solar_azimuth_angle, I_dir_tot, I_dif_tot


def Calculation_ISO_52010(building_object, path_weather_file, weather_source="pvgis") -> simdf_52010:
    """
    Calculation procedure for the conversion of climatic data for energy calculation.
    The main element in ISO 52010-1:2017 is the calculation of solar irradiance on a surface with arbitrary orientation and tilt


    :param building_object:  Building object create according to the method ``Building``or ``Buildings_from_dictionary``
    :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

    :return: **sim_df**: climatic data for energy simulation (type: **pd.DataFrame**)
    """

    # Get weather dataframe
    if weather_source == "pvgis":
        weatherData = ISO52010.get_tmy_data_pvgis(building_object)
    elif weather_source == "epw":
        weatherData = ISO52010.get_tmy_data_epw(path_weather_file)
    elif weather_source == "climatedata":
        if path_weather_file is None:
            path_weather_file = ISO52010.get_tmy_data_climatedataforbuildings(
                building_object, in_memory=True
            )
        weatherData = ISO52010.get_tmy_data_epw(path_weather_file)
    else:
        raise ValueError("select the right weather source: 'epw', 'pvgis', or 'climatedata'")

    sim_df = weatherData.weather_data
    timezoneW = weatherData.utc_offset

    # Change time index
    if len(sim_df) > 8760:  # In the case of a leap year, (ITA: anno bisestile)
        pass
    else:
        sim_df.index = pd.to_datetime(
            {
                "year": 2009,
                "month": sim_df.index.month,
                "day": sim_df.index.day,
                "hour": sim_df.index.hour,
            }
        )

    # Time handling by weather source:
    # - PVGIS data are UTC -> convert to local civil time using site timezone.
    # - EPW read via pvlib is already in local weather-file time -> keep as-is.
    if weather_source == "pvgis":
        from timezonefinder import TimezoneFinder  # lazy: only needed for PVGIS timezone conversion

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=weatherData.longitude, lat=weatherData.latitude) or "UTC"
        idx = pd.DatetimeIndex(sim_df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        sim_df.index = idx.tz_convert(tz_name).tz_localize(None)
    elif weather_source == "epw":
        idx = pd.DatetimeIndex(sim_df.index)
        if idx.tz is not None:
            sim_df.index = idx.tz_localize(None)

    sim_df.rename_axis(index={"time(UTC)": "time(local)"}, inplace=True)
    sim_df["day of year"] = sim_df.index.dayofyear
    sim_df["hour of day"] = sim_df.index.hour + 1  # 1 to 24
    or_tilt_azim_dic = {
        "HOR": (0, 0),
        "SV": (90, 0),
        "EV": (90, 90),
        "NV": (90, 180),
        "WV": (90, -90),
    }  # dictionary mapping orientation in orientation_elements with (beta_ic_deg=elevation/tilt, gamma_ic_deg=azimuth), see util.util.ISO52010_calc()

    if len(sim_df) > 8760:
        n_tsteps = 8784
        n_days_year = 366
    else:
        n_tsteps = 8760
        n_days_year = 365
    # Convert the NumPy array to a tuple
    if isinstance(building_object, dict):
        orientation_elements = ["EV", "HOR", "SV", "NV", "WV"]
    else:
        orientation_elements = building_object.__getattribute__("orientation_elements")

    for orientation in set(orientation_elements):

        Solar_irradiance, alt, az, I_dir_tot, I_dif_tot = ISO52010.Solar_irradiance_calculation(
            n_timesteps=n_tsteps,
            n_days=n_days_year,
            latitude_deg=weatherData.latitude,
            longitude_deg=weatherData.longitude,
            timezone=timezoneW,
            beta_ic_deg=or_tilt_azim_dic[orientation][0],
            gamma_ic_deg=or_tilt_azim_dic[orientation][1],
            DHI=sim_df["Gd(h)"],
            DNI=sim_df["Gb(n)"],
            ground_solar_reflectivity=0.2,
            calendar=sim_df[["day of year", "hour of day"]],
            building_object=building_object
        )
        Solar_irradiance.solar_irradiance.columns = [f'I_sol_tot_{orientation}',f'I_sol_dif_{orientation}',f'I_sol_dir_w_{orientation}']
        sim_df = pd.concat([sim_df, Solar_irradiance.solar_irradiance], axis=1)

        Shading_factor = ISO52010.Shading_reduction_factor_window(
            solar_altitude_angle=alt,
            solar_azimuth_angle=az,
            I_dir_tot=I_dir_tot,
            I_dif_tot=I_dif_tot,
            calendar=sim_df[["day of year", "hour of day"]],
            n_timesteps=n_tsteps,
            building_object=building_object,
            orientation=orientation
        )
        if Shading_factor != None:
            sim_df = pd.concat([sim_df, Shading_factor.shading_reduction_factor_window], axis=1)

    # Add an explicit December warm-up block at the beginning, shifted to the
    # previous year so it remains distinguishable from the actual December of
    # the simulation year.
    sim_df = _prepend_december_warmup_to_previous_year(sim_df)

    # sim_df.to_csv('sim_df_with_shadingf.csv')
    return simdf_52010(sim_df=sim_df)


#                                       ISO 52016
# ===============================================================================================


def _series_to_float_array(df, col: str, default: float | None = None):
    """Return a column of df as a float64 NumPy array.
    Used to replace hot-path .iloc[t] accesses with O(1) array indexing.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if default is None:
        raise KeyError(f"Missing required weather/profile column: {col!r}")
    return np.full(len(df), float(default), dtype=float)


def _safe_load(P_max, T_set, T0, Tupper, clip_factor=2000.0):
    """Return HVAC load [W] using ISO 52016 Eq. (27) with robust fallback.
    Positive = heating, negative = cooling.
    Defined at module level so it is not recreated on every timestep.
    """
    try:
        pmax = float(P_max)
        t_set = float(T_set)
        t0 = float(T0)
        t_upper = float(Tupper)
    except (TypeError, ValueError):
        return 0.0
    dT = t_set - t0
    if not np.isfinite(dT) or abs(dT) < 1e-9:
        return 0.0
    denom = t_upper - t0
    if np.isfinite(denom) and abs(denom) >= 0.05:
        load = pmax * dT / denom
    else:
        load = clip_factor * dT
    if not np.isfinite(load):
        return 0.0
    if pmax >= 0.0:
        return float(min(pmax, max(0.0, load)))
    return float(max(pmax, min(0.0, load)))


class ISO52016:

    or_tilt_azim_dic = {
        "HOR": (0, 0),
        "SV": (90, 0),
        "EV": (90, 90),
        "NV": (90, 180),
        "WV": (90, -90),
    }  # dictionary mapping orientation in orientation_elements with (beta_ic_deg=elevation/tilt, gamma_ic_deg=azimuth), see util.util.ISO52010_calc()

    def __init__(self):
        pass

        """
        Simulate the Final Energy consumption of the building using the ISO52016
        """

    @staticmethod
    def _add_zone_longwave_radiative_exchange(A_matrix: np.ndarray, faces) -> None:
        """
        Add long-wave radiative exchange terms for one zone-facing surface network.

        The assembly uses one area-weighted equivalent h_ri for the zone and
        the coupling form:
            + A_i * h_ri_eq on diagonal,
            - A_i * (A_k / A_tot) * h_ri_eq on all columns k
        which yields:
          - symmetric radiative sub-matrix
          - row-sum zero (energy-conservative exchange operator)
        """
        if not faces:
            return

        valid_faces = []
        for (R_i, A_i, h_ri_i) in faces:
            A_i_f = float(A_i)
            h_ri_i_f = float(h_ri_i)
            if (
                A_i_f > 0.0
                and np.isfinite(A_i_f)
                and np.isfinite(h_ri_i_f)
                and h_ri_i_f >= 0.0
            ):
                valid_faces.append((R_i, A_i_f, h_ri_i_f))

        if not valid_faces:
            return

        area_tot = float(sum(A_i_f for _, A_i_f, _ in valid_faces))
        if area_tot <= 0.0:
            return

        h_ri_eq = float(
            sum(A_i_f * h_ri_i_f for _, A_i_f, h_ri_i_f in valid_faces) / area_tot
        )
        if not np.isfinite(h_ri_eq) or h_ri_eq < 0.0:
            return

        for (R_i, A_i_f, _) in valid_faces:
            A_matrix[R_i, R_i] += A_i_f * h_ri_eq
            for (R_k, A_k_f, _) in valid_faces:
                A_matrix[R_i, R_k] -= A_i_f * (A_k_f / area_tot) * h_ri_eq

    @staticmethod
    def _surface_shading_factor_from_timeseries(
        sim_df: pd.DataFrame,
        tstep: int,
        surface: dict,
        shading_components_by_zone_orientation: dict,
        default_zone: str = "main",
    ) -> float:
        """
        Return shading reduction factor [0..1] for a (possibly aggregated) window surface.

        Priority:
        1) area-weighted average from original window columns W_<name> grouped by (zone, orientation)
        2) direct column W_<surface_name> (non-aggregated case)
        3) fallback 1.0 (no shading data available)
        """

        def _column_value(col_name: str):
            if not col_name or col_name not in sim_df.columns:
                return None
            try:
                raw = sim_df[col_name].iloc[tstep]
                arr = np.asarray(raw, dtype=float).reshape(-1)
            except Exception:
                return None
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            return float(np.clip(np.mean(arr), 0.0, 1.0))

        zname = surface.get("zone", default_zone)
        if zname is None:
            zname = default_zone
        ori = str(surface.get("ISO52016_orientation_string", "SV")).upper()
        key = (zname, ori)

        components = shading_components_by_zone_orientation.get(key, [])
        if components:
            num = 0.0
            den = 0.0
            for win_name, win_area in components:
                try:
                    area_f = float(win_area)
                except Exception:
                    continue
                if not np.isfinite(area_f) or area_f <= 0.0:
                    continue
                f_val = _column_value(f"W_{win_name}")
                if f_val is None:
                    continue
                num += area_f * f_val
                den += area_f
            if den > 0.0:
                return float(np.clip(num / den, 0.0, 1.0))

        surface_name = str(surface.get("name", "")).strip()
        f_direct = _column_value(f"W_{surface_name}")
        if f_direct is not None:
            return f_direct

        return 1.0

    @classmethod
    def Number_of_nodes_element(cls, building_object) -> numb_nodes_facade_elements:
        """
        Calculation of the number of nodes for each element.
        If OPACQUE, or ADIABATIC -> n_nodes = 5
        If TRANSPARENT-> n_nodes = 2

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``
        :return:
            * *Rn*: value of last node to be used in the definition of the element vector
            * *Pln*: inizial number of nodes according to the type of element (5 - opaque element, 2 - transparent element)
            * *PlnSum*: sequential number of nodes based on the list of opaque and transparent elements
        """
        # Number of envelop building elements
        el_list = len(building_object["building_surface"])
        # Initialize Pln with all elements as 5
        Pln = np.full(el_list, 5)
        # Replace elements with value 2 where type is "transparent",
        # 0 if adiabatic (type or boundary).
        for i, surf in enumerate(building_object["building_surface"]):
            surf_type = str(surf.get("type", "")).lower()
            surf_boundary = str(surf.get("boundary", "")).upper()
            if surf_type == "transparent":
                Pln[i] = 2
            elif surf_type == "adiabatic" or surf_boundary == "ADIABATIC":
                Pln[i] = 0
        # Calculation fo number of nodes for each building element (wall, roof, window)
        PlnSum = np.array([0] * el_list)
        for Eli in range(1, el_list):
            PlnSum[Eli] = (
                PlnSum[Eli - 1] + Pln[Eli - 1]
            )  # Index of matrix , each row is a node

        Rn = (
            PlnSum[-1] + Pln[-1] + 1
        )  # value of last node to be used in the definition of the vector

        return numb_nodes_facade_elements(Rn, Pln, PlnSum)

    @classmethod
    def Conduttance_node_of_element(
        cls, building_object, lambda_gr=2.0
    ) -> conduttance_elements:
        """
        Calculation of the conductance between node "pli" and node "pli-1", as determined per type of construction
        element in 6.5.7 in W/m2K

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param lambda_gr: hermal conductivity of ground [W/(m K)]. Default=2

        .. note:: Required parameters of building_object:

            * type: type of building element opaque, transparent or adiabatic
            * res: theraml ressistance of opaque building element
            * kappa_m: heat_capacity of the element in Table B.14
            * solar_absorption_coeff: solar absorption coefficient of element provided by user or using values of Table A.15 and B.15 of ISO 52016
            * area: area of each element [m2]
            * ori_tilt: orientation and tilt values
            * g_value: onyl for window

        :return: h_pli_eli: conductance coefficient between nodes (W/m2K). *type*: np.array
        """
        R_gr = 0.5 / lambda_gr  # thermal resistance of 0.5 m of ground [m2 K/W]
        # Number of envelop building elements
        el_type = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        # Initialization of conductance coefficient calculation
        h_pli_eli = np.zeros((4, len(el_type)))
        U_eli = [surf["u_value"] for surf in building_object["building_surface"]]
        R_c_eli = [0.0] * len(el_type)
        h_ci_eli = [
            surf["convective_heat_transfer_coefficient_internal"]
            for surf in building_object["building_surface"]
        ]
        h_ri_eli = [
            surf["radiative_heat_transfer_coefficient_internal"]
            for surf in building_object["building_surface"]
        ]
        convective_heat_transfer_coefficient_external = 20.0  # See ISO 13789
        h_ce_eli = [convective_heat_transfer_coefficient_external] * len(el_type)
        for i, surf in enumerate(building_object["building_surface"]):
            if surf["ISO52016_type_string"] == "AD":
                h_ce_eli[i] = 0.0
                surf["convective_heat_transfer_coefficient_external"] = 0.0
            else:
                surf["convective_heat_transfer_coefficient_external"] = (
                    convective_heat_transfer_coefficient_external
                )
        radiative_heat_transfer_coefficient_external = 4.14  # See ISO 13789
        h_re_eli = [radiative_heat_transfer_coefficient_external] * len(el_type)
        for i, surf in enumerate(building_object["building_surface"]):
            if surf["ISO52016_type_string"] == "AD":
                h_re_eli[i] = 0.0
                surf["radiative_heat_transfer_coefficient_external"] = 0.0
            else:
                surf["radiative_heat_transfer_coefficient_external"] = (
                    radiative_heat_transfer_coefficient_external
                )

        for i in range(0, len(el_type)):
            if el_type[i] == "AD":
                R_c_eli[i] = float("inf")
            if R_c_eli[i] == 0.0:
                R_c_eli[i] = (
                    1 / U_eli[i]
                    - 1 / (h_ci_eli[i] + h_ri_eli[i])
                    - 1 / (h_ce_eli[i] + h_re_eli[i])
                )

        # layer = 1
        layer_no = 0
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    h_pli_eli[0, i] = 6 / R_c_eli[i]
                elif el_type[i] == "W":
                    h_pli_eli[0, i] = 1 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[0, i] = 2 / R_gr

        # layer = 2
        layer_no = 1
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":                    
                    h_pli_eli[layer_no, i] = 3 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[layer_no, i] = 1 / (R_c_eli[i] / 4 + R_gr / 2)

        # layer = 3
        layer_no = 2
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    h_pli_eli[layer_no, i] = 3 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[layer_no, i] = 2 / R_c_eli[i]

        # layer = 4
        layer_no = 3
        for i in range(len(el_type)):
            if R_c_eli[i] != 0:
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    h_pli_eli[layer_no, i] = 6 / R_c_eli[i]
                elif el_type[i] == "GR":
                    h_pli_eli[layer_no, i] = 4 / R_c_eli[i]

        return conduttance_elements(h_pli_eli=h_pli_eli)

    @classmethod
    def Solar_absorption_of_element(cls, building_object) -> solar_absorption_elements:
        """
        Calculation of solar absorption for each single elements

        :param building_object: building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        :return: a_sol_pli_eli: solar absorption of each single nodes (type: *np.array*)

        .. note:: 
            EXAMPLE:

            a_sol_pli_eli = array(
                [
                    [0. , 0.6, 0.6, 0.6, 0.6, 0.6, 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]
                ]
            )

        """
        # Number of envelop building elements
        el_list = len(building_object["building_surface"])

        # Coefficient list of elements
        solar_abs_elements = [0.0] * el_list
        for i, surf in enumerate(building_object["building_surface"]):
            if "solar_absorptance" in surf:  # Opaque element
                if surf["ISO52016_type_string"] == "AD" or surf["ISO52016_type_string"] == "ADJ":
                    solar_abs_elements[i] = 0.0
                else:
                    solar_abs_elements[i] = surf["solar_absorptance"]
            else:  # Transparent element
                solar_abs_elements[i] = surf["g_value"]

        # Initialization of solar_abs_coeff
        a_sol_pli_eli = np.zeros((5, el_list))
        a_sol_pli_eli[0, :] = solar_abs_elements

        return solar_absorption_elements(a_sol_pli_eli=a_sol_pli_eli)

    @classmethod
    def Areal_heat_capacity_of_element(cls, building_object) -> aeral_heat_capacity:
        """
        Calculation of the areal heat capacity of the node "pli" and node "pli-1" as
        determined per type of construction element [W/m2K] - 6.5.7 ISO 52016

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.

        .. note:: Required parameters of building_object:

            * type: type of building element OPAQUE -'OP', TRANSPARENT - 'W', GROUND -'GR'
            * kappa_m: heat_capacity of the element in Table B.14
            * construction_class: lass of construction with respect to the distribution of the mass in the construction
                        Table B.13. Possible choice: class_i, class_e, class_ie, class_d, class_m

        :return: aeral_heat_capacity: areal heat capacity of each facade element (type: *np.array*)
        """
        # Number of envelop building elements
        el_type = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        list_kappa_el = [0] * len(el_type)
        for i, surf in enumerate(building_object["building_surface"]):
            if "thermal_capacity" in surf:
                list_kappa_el[i] = surf["thermal_capacity"]

        # Initialization of heat capacity of nodes
        kappa_pli_eli_ = np.zeros((5, len(el_type)))

        if building_object['building']['construction_class'] == "class_i":
            # Mass concentrated at internal side
            # OPAQUE: kpl5 = km_eli ; kpl1=kpl2=kpl3=kpl4=0
            # GROUND: kpl5 = km_eli ; kpl3=kpl4=0
            node = 1
            for i in range(len(el_type)):
                if el_type[i] == "GR":
                    kappa_pli_eli_[node, i] = 1e6  # heat capacity of the ground
            node = 4
            for i in range(len(el_type)):
                if el_type[i] != "W":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        elif (
            building_object['building']['construction_class'] == "class_e"
        ):  # mass concentrated at external side
            # OPAQUE: kpl1 = km_eli ; kpl2=kpl3=kpl4=kpl5=0
            # GROUND: kpl3 = km_eli ; kpl4=kpl5=0
            node = 0
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                elif el_type[i] == "GR":
                    node = 2
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        elif (
            building_object['building']['construction_class'] == "class_ie"
        ):  # mass divided over internal and external side)
            # OPAQUE: kpl1 = kpl5 = km_eli/2 ; kpl2=kpl3=kpl4=0
            # GROUND: kpl1 = kp5 =km_eli/2; kpl4=0
            node = 0
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
                elif el_type[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
            node = 4
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2
                elif el_type[i] == "GR":
                    kappa_pli_eli_[node, i] = list_kappa_el[i] / 2

        elif (
            building_object['building']["construction_class"] == "class_d"
        ):  # (mass equally distributed)
            # OPAQUE: kpl2=kpl3=kpl4=km_eli/4
            # GROUND: kpl3=km_eli/4; kpl4=km_eli/2
            node_list_1 = [1, 2, 3]
            for node in node_list_1:
                for i in range(len(el_type)):
                    if el_type[i] == "OP" or el_type[i] == "ADJ":
                        kappa_pli_eli_[node, i] = list_kappa_el[i] / 4
                    if el_type[i] == "GR":
                        if node == 2:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 4
                        if node == 3:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 2

            # OPAQUE kpl1=kpl5= km_eli/8
            # GROUND:kpl5=km_eli/4
            node_list_2 = [0, 4]
            for node in node_list_2:
                for i in range(len(el_type)):
                    if el_type[i] == "OP" or el_type[i] == "ADJ":
                        kappa_pli_eli_[node, i] = list_kappa_el[i] / 8
                    if el_type[i] == "GR":
                        if node == 4:
                            kappa_pli_eli_[node, i] = list_kappa_el[i] / 4

        elif (
            building_object['building']["construction_class"] == "class_m"
        ):  # mass concentrated inside
            # OPAQUE: kpl1=kpl2=kpl4=kpl5=0; kpl3= km_eli
            # GROUND: kpl4=km_eli; kpl3=kpl5=0
            node = 2
            for i in range(len(el_type)):
                if el_type[i] == "OP" or el_type[i] == "ADJ":
                    kappa_pli_eli_[node, i] = list_kappa_el[i]
                if el_type[i] == "GR":
                    node = 3
                    kappa_pli_eli_[node, i] = list_kappa_el[i]

        return aeral_heat_capacity(kappa_pli_eli=kappa_pli_eli_)

    @classmethod
    def Temp_calculation_of_ground(
        cls, building_object, lambda_gr=2.0, R_si=0.17, R_se=0.04, psi_k=0.05, **kwargs
    ) -> temp_ground:
        """
        Virtual ground temperature calculation of ground according to ISO 13370-1:2017
        for salb-on-ground (sog) floor

        :param building_object: Building object create according to the method ``Building``or ``Buildings_from_dictionary``.
        :param lambda_gr: hermal conductivity of ground [W/(m K)]. Default=2
        :param R_se: external surface resistance (for ground floor calculation). Default 0.04
        :param R_si: internal surface resistance (for ground floor calculation). Default 0.17
        :param psi_k: linear thermal transmittance associated with wall/floor junction [W/(m K)]. Default 0.05

        .. note:: Required parameters of building_object:

            * heating: TRUE or FALSE. Is there a heating system?
            * cooling: TRUE or FALSE. Is there a cooling system?
            * heating_setpoint: setpoint for the heating system (default 20°C)
            * cooling_setpoint: setpoint for cooling system (default 26°C)
            * latitude_deg: latitude of location in degrees
            * slab_on_ground_area: area of the building in contact with the ground
            * perimeter: perimeter of the building [m]
            * wall_thickness: thickness of the wall [m]
            * thermal_resistance_floor: resitance of the floor [m2K/W]
            * thermal_bridge_heat: Thermal bridges heat transfer coefficient - sum of thermal bridges (clause 6.6.5.3)
            * coldest_month: coldest month, if not provided automatically selected according to the hemisphere

        :return:
            * **R_gr_ve**: Thermal resistance of virtual layer (floor_slab)
            * **thermal_bridge_heat**: Heat transfer coefficient of overall thermal briges
            * **Theta_gr_ve**: Internal Temperature of the ground

        .. caution:: 
            The calculation is only for buildings with a ground floor slab. That is, in direct contact with the ground and not for unheated rooms.
            To be integrated: code related to different types of contacts based on the presence of an unheated room or other.

        .. note:: 
            Calculation of annual_mean_internal_temperature and its amplitude variations
            if heating and colling are selected:

                * the annual mean internal temperature is the average between Heating and Cooling setpoints
                * the amplitude variations is the mean of the difference between Heating and Cooling setpoints

            if not heating and cooling the value should be provided by the user:

                * if the user doesn't provide any value, the following values are used: 
                    * annual_mean_internal_temperature = 23 <- ((26 (standard C set point) + 20 (standard H setpoint))/2)
                    * amplitude_of_internal_temperature_variations = 3 <- (26-20)/2

        .. note:: 
            Defintion of the coldest month accoriding to the position.
            If the user doesn't provide a value between 1 (January) and 12 (Decemebr)
            the default values: 1 for northern hemisphere or 7 in southern hemisphere are used

        .. note::
            Optional override through ``building_parameters.simulation_options``:

                * ``ground_temperature_model = "iso13370"`` (default)
                * ``ground_temperature_model = "monthly"`` plus
                  ``ground_temperature_monthly = [T_jan, ..., T_dec]`` [degC]
                * ``ground_temperature_model = "energyplus"`` is still accepted
                  as backward-compatible alias of ``"monthly"``

        """

        # ============================
        #
        R_gr = 0.5 / lambda_gr  # thermal resistance of 0.5 m of ground [m2 K/W]

        ground_temperature_model = _resolve_ground_temperature_model(
            building_object,
            model_override=kwargs.get("ground_temperature_model"),
        )
        ground_temperature_monthly = None
        if ground_temperature_model == "monthly":
            ground_temperature_monthly = _resolve_ground_temperature_monthly(
                building_object,
                monthly_override=kwargs.get(
                    "ground_temperature_monthly",
                    kwargs.get("energyplus_ground_temperature_monthly"),
                ),
            )
            if ground_temperature_monthly is None:
                raise ValueError(
                    "ground_temperature_model='monthly' requires "
                    "simulation_options.ground_temperature_monthly with 12 monthly values [degC]."
                )

        # ============================
        # GET MIN, MAX AND MEAN of External temperature values at monthly(M) resolution
        path_weather_file_ = kwargs.get("path_weather_file")
        weather_source = kwargs.get("weather_source", "pvgis")
        sim_df = Calculation_ISO_52010(building_object, path_weather_file_, weather_source=weather_source).sim_df

        try:
            external_temperature_monthly_averages = sim_df["T2m"].resample("ME").mean()
            external_temperature_monthly_minima = sim_df["T2m"].resample("ME").min()
            external_temperature_monthly_maxima = sim_df["T2m"].resample("ME").max()
        except:
            external_temperature_monthly_averages = sim_df["T2m"].resample("M").mean()
            external_temperature_monthly_minima = sim_df["T2m"].resample("M").min()
            external_temperature_monthly_maxima = sim_df["T2m"].resample("M").max()

        # amplitude of external temperature variations
        amplitude_of_external_temperature_variations = (
            external_temperature_monthly_maxima - external_temperature_monthly_minima
        ).mean() / 2
        # annual mean of external temperature
        # Use monthly average external temperature as fallback
        annual_mean_external_temperature = (external_temperature_monthly_averages.mean())

        # ============================

        # ============================
        # For the Grins Beat project, we assume active heating and cooling setpoints
        annual_mean_internal_temperature = (
            building_object["building_parameters"]["temperature_setpoints"][
                "heating_setpoint"
            ]
            + building_object["building_parameters"]["temperature_setpoints"][
                "cooling_setpoint"
            ]
        ) / 2
        amplitude_of_internal_temperature_variations = (
            building_object["building_parameters"]["temperature_setpoints"][
                "cooling_setpoint"
            ]
            - building_object["building_parameters"]["temperature_setpoints"][
                "heating_setpoint"
            ]
        ) / 2
        
        # ============================

        # ============================

        coldest_month = 1
        building_object["building_parameters"]["coldest_month"] = coldest_month

        internal_temperature_by_month = np.zeros(12)
        for month in range(12):
            internal_temperature_by_month[month] = (
                annual_mean_internal_temperature
                - amplitude_of_internal_temperature_variations
                * np.cos(2 * np.pi * (month + 1 - coldest_month) / 12)
            )  # estimate
        # ============================

        # ============================
        # Total slab-on-ground area across all ground-contact surfaces.
        sog_area = _ground_contact_area(building_object)
        # ============================

        # ============================
        """
        Calculation of the perimeter.
        If the value is not provided by the user a rectangluar shape of the building is considered.
        The perimeter is calcuated according to the area of the south and east facade
        """
        
        exposed_perimeter = building_object["building"]["exposed_perimeter"]
        characteristic_floor_dimension = sog_area / (0.5 * exposed_perimeter)
        # ============================

        # ============================
        """
        Calculation of temperature of the ground using:
            1. the thermal Resistance (R) and Transmittance (U) of the floor
            2. External Temperature [°C]
        """
        wall_thickness = building_object["building"]["wall_thickness"]
        thermal_resistance_floor = 5.3
        # building_object.thermal_resistance_floor = 5.3  # Floor construction thermal resistance (excluding effect of ground) [m2 K/W]

        # The thermal transmittance depends on the characteristic dimension of the floor, B' [see 8.1 and Equation (2)], and the total equivalent thickness, dt (see 8.2), defined by Equation (3):
        equivalent_ground_thickness = wall_thickness + lambda_gr * (thermal_resistance_floor + R_se)  # [m]

        if (
            equivalent_ground_thickness < characteristic_floor_dimension
        ):  # uninsulated and moderately insulated floors
            U_sog = (2 * lambda_gr/ (np.pi * characteristic_floor_dimension + equivalent_ground_thickness)
                * np.log(
                    np.pi * characteristic_floor_dimension / equivalent_ground_thickness
                    + 1
                )
            )  # thermal transmittance of slab on ground including effect of ground [W/(m2 K)]
        else:  # well-insulated floors
            U_sog = lambda_gr / (0.457 * characteristic_floor_dimension + equivalent_ground_thickness)

        # calcualtion of thermal resistance of virtual layer
        R_gr_ve_raw = 1 / U_sog - R_si - thermal_resistance_floor - R_gr
        # Keep the virtual-ground resistance physically valid; negative/zero values
        # cause non-physical ground coupling and unstable/biased energy needs.
        if (not np.isfinite(R_gr_ve_raw)) or (R_gr_ve_raw <= 0.0):
            R_gr_ve = max(R_gr, 0.05)
        else:
            R_gr_ve = float(R_gr_ve_raw)

        # Adding thermal bridges
        thermal_bridge_heat = exposed_perimeter * psi_k

        # Calculation of steady-state  ground  heat  transfer  coefficients  are  related  to  the  ratio  of  equivalent  thickness
        # to  characteristic floor dimension, and the periodic heat transfer coefficients are related to the ratio
        # of equivalent thickness to periodic penetration depth
        steady_state_heat_transfer_coefficient = (
            sog_area * U_sog + exposed_perimeter * psi_k
        )  # [W/K]
        periodic_penetration_depth = 3.2  # [m]
        H_pi = (
            sog_area
            * lambda_gr
            / equivalent_ground_thickness
            * np.sqrt(
                2
                / (
                    np.float_power(
                        1 + periodic_penetration_depth / equivalent_ground_thickness, 2
                    )
                    + 1
                )
            )
        )  # periodic heat transfer coefficient related to internal temperature variations [W/K]
        H_pe = (
            0.37
            * exposed_perimeter
            * lambda_gr
            * np.log(periodic_penetration_depth / equivalent_ground_thickness + 1)
        )  # periodic heat transfer coefficient related to external temperature variations [W/K]
        annual_average_heat_flow_rate = steady_state_heat_transfer_coefficient * (
            annual_mean_internal_temperature - annual_mean_external_temperature
        )  # [W]
        periodic_heat_flow_due_to_internal_temperature_variation = np.zeros(12)
        a_tl = 0  # time lead of the heat flow cycle compared with that of the internal temperature [months]
        b_tl = 1  # time lag of the heat flow cycle compared with that of the external temperature [months]
        for month in range(12):
            periodic_heat_flow_due_to_internal_temperature_variation[month] = (
                -H_pi
                * amplitude_of_internal_temperature_variations
                * np.cos(2 * np.pi * (month + 1 - coldest_month + a_tl) / 12)
            )
        periodic_heat_flow_due_to_external_temperature_variation = np.zeros(12)
        for month in range(12):
            periodic_heat_flow_due_to_external_temperature_variation[month] = (
                H_pe
                * amplitude_of_external_temperature_variations
                * np.cos(2 * np.pi * (month + 1 - coldest_month - b_tl) / 12)
            )
        average_heat_flow_rate = (
            annual_average_heat_flow_rate
            + periodic_heat_flow_due_to_internal_temperature_variation
            + periodic_heat_flow_due_to_external_temperature_variation
        )
        Theta_gr_ve = internal_temperature_by_month - (
            average_heat_flow_rate
            - exposed_perimeter
            * psi_k
            * (annual_mean_internal_temperature - annual_mean_external_temperature)
        ) / (sog_area * U_sog)
        if ground_temperature_model == "monthly":
            Theta_gr_ve = ground_temperature_monthly.copy()

        return temp_ground(
            R_gr_ve=R_gr_ve,
            Theta_gr_ve=Theta_gr_ve,
            thermal_bridge_heat=thermal_bridge_heat,
            ground_contact_area=float(sog_area),
        )

    @classmethod
    def Weather_data_bui(cls, building_object, path_weather_file, weather_source="pvgis") -> simulation_df:
        """
        Get weather data for the building object.

        :param path_weather_file: path of the .epw weather file. (e.g (../User/documents/epw/athens.epw))

        :retrun: sim_df: dataframe with inputs for simulation having information of weather, occupancy, heating and cooling setpoint and setback
        """
        # WEATHER DATA
        if weather_source == "pvgis":
            sim_df = pd.DataFrame(Calculation_ISO_52010(building_object, path_weather_file, weather_source=weather_source).sim_df)
        
        elif weather_source == "epw":
            sim_df = pd.DataFrame(Calculation_ISO_52010(building_object, path_weather_file, weather_source=weather_source).sim_df)

        elif weather_source == "climatedata":
            sim_df = pd.DataFrame(Calculation_ISO_52010(building_object, path_weather_file, weather_source=weather_source).sim_df)
        
        sim_df.index = pd.DatetimeIndex(sim_df.index)
        
        return simulation_df(simulation_df=sim_df)
    
    @classmethod
    def transmission_heat_transfer_coefficient_ISO13789(cls,adj_zone, n_ue=0.5, qui=0):
        '''
        Calculation of heat transfer coefficient, Htr calculated as
        Htr = Hd + Hg + Hu + Ha
        where:
        Hd: direct transmission heat transfer coefficient between the heated and cooled space and exterior trough the building envelope in W/K
        Hg: transmission trasnfer coefficient through the ground in W/K
        Hu: transmission heat transfer coefficent through unconditioned space
        Ha: transmision heat transfer coefficient to adjacent buildings
        '''

        '''
        1. From the adj_zone dictionary, get the orientation, U and area of the facade elements
        '''
        # Geographical azimuth convention used across checks/surfaces:
        # NV=0, EV=90, SV=180, WV=270.
        # Use nearest cardinal to be robust to non-exact azimuths.
        try:
            azimuth = float(adj_zone["orientation_zone"]["azimuth"]) % 360.0
        except Exception:
            azimuth = 0.0
        candidates = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
        labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
        diffs = np.abs(((azimuth - candidates + 180.0) % 360.0) - 180.0)
        orientation_name = str(labels[int(np.argmin(diffs))])

        # --- Step 2: Extract arrays
        areas = adj_zone["area_facade_elements"].astype(float)
        U_values = adj_zone["transmittance_U_elements"].astype(float)
        orientations = np.asarray(adj_zone["orientation_elements"], dtype=object)

        # --- Step 3: Boolean masks
        # mask_selected = orientation_name == orientation_name
        # mask_others = orientation_name != orientation_name
        mask_selected = orientations == orientation_name
        mask_others = orientations != orientation_name

        # --- Step 4: Calculate sums
        '''
        Transmission losses of walls attached to the unconditioned zone
        '''
        Hd_zt_ztu = np.sum(areas[mask_selected] * U_values[mask_selected])

        '''   
        Transmission losses of external walls of unconditioned zone
        '''
        Hd_ztu_ext = np.sum(areas[mask_others] * U_values[mask_others])

        '''
        4. Calculate ventilation losses Hve,iu and Hve,ue
        
        1) Hve,iu = rho*cp*qiu
        2) Hve,ue = rho*cp*que
        
        rho: density of air
        cp: specific heat capacity of air
        qiu: air flow rate in m3/h between conditioned and unconditioned zone   
        que: air flow rate in m3/h between unconditioned zone and external environment
        Note:
        rho_cp = 0.33 Wh/(m3K) if qiu in m3/h

        # air change rate of unconditioned spaces:
        in order to not understimate the transmission heat transfer, the air flow rate between a conditioned space and an unconditioned space 
        shall be assumed to be zero
        qiu= 0
        que = Vu * n_ue
        Vu: volume of the air in the unconditioned space
        n_ue: air change rate between the unconditioned space and the external environment in h-1. Can be taken from the table df_n_ue that is table 7  of ISO 13789

        default n_ue = 0.5 that is for building having All joints between components well-sealed, no ventilation opening provided

        '''    
        volume_zone = adj_zone["volume"]
        que = volume_zone * n_ue
        Hve_ue =  0.33 * que
        Hve_iu = 0.33 * qui
        # 
        Hue = Hd_ztu_ext + Hve_ue
        Hiu = Hd_zt_ztu + Hve_iu
        H_ztu_tot = float(round(Hiu +Hue,3))
        
        '''
        Calculation of the adjustment factor for the thermally uncononditioned afjacent zone ztu for in month m
        b = Hztu_e_m/Hztu_tot_m

        where:
        H_ztu_tot_m = sum(j=1 to n)(H_ztc_j_ztu) + Hztu_e_m
        where:
        H_ztc_j_ztu: heat transfer coefficient between the thermally unconditioned zone and the adjacent thermally conditioned zone j  = Hiu (ISO 13789)
        Hztu_e_m: heat transfer coefficient between the thermally unconditioned zone and the external environment for month m =  Hue (ISO 13789)
        '''
        b_ztu_m = float(round(Hue/(Hue + Hiu),3))

        '''
        Calculation of the distribution factor for the heat transfer between thermally conditioned zone i and the adjacent thermally unconditioned zone ztu, for month m
    
        if multiple thermally conditioned zones:
            F_ztc_ztu_m = H_ztc_i_ztu_m / sum(j=1 to n)(H_ztc_j_ztu)
        where:
            H_ztc_i_ztu_m: heat transfer coefficient between the single thermally conditioned zone and the adjacent thermally unconditioned zone for month m
        if only 1 thermally conditioned zones:
            F_ztc_ztu_m = 1

        '''
        F_ztc_ztu_m = 1 #<<<< write the code formultiple thermally consitioned zones
        
        return H_ztu_tot, b_ztu_m, F_ztc_ztu_m


    @classmethod
    def _aggregate_surfaces_by_direction(cls, building_object):
        """
        Collapse multiple surfaces that share the same direction into a single
        equivalent surface so transmission & solar terms are computed on aggregates.

        Key = (
            ISO52016_type_string,
            ISO52016_orientation_string,
            type,
            boundary,
            zone,
            adjacent_zone,
        )

        - Additive: area, thermal_capacity
        - Area-weighted: u_value, g_value, sky_view_factor,
                        h_conv/rad (int & ext), solar_absorptance
        - Non-aggregated fields: keep first reasonable 'name' with a suffix
        """
        if not isinstance(building_object, dict):
            # Only implemented for dict-style building_object here
            return building_object

        # If the tags are not present yet, do nothing
        for s in building_object.get("building_surface", []):
            if "ISO52016_type_string" not in s or "ISO52016_orientation_string" not in s:
                return building_object

        from collections import defaultdict

        buckets = defaultdict(lambda: {
            "name": [],
            "type": None,
            "boundary": None,
            "zone": None,
            "adjacent_zone": None,
            "name_adj_zone": None,
            "orientation": None,
            "area": 0.0,
            "uA": 0.0,            # accumulator for U*A
            "gA": 0.0,            # accumulator for g*A (windows only)
            "svfA": 0.0,          # sky_view_factor * A
            "hciA": 0.0,          # convective int * A
            "hriA": 0.0,          # radiative  int * A
            "hceA": 0.0,          # convective ext * A
            "hreA": 0.0,          # radiative  ext * A
            "a_sol_A": 0.0,       # solar_absorptance * A
            "thermal_capacity": 0.0,
            "ISO52016_type_string": None,
            "ISO52016_orientation_string": None,
            "width_sum": 0.0,
            "height_sum": 0.0,
            "parapet_sum": 0.0,
            "parapetA": 0.0,
        })

        for s in building_object["building_surface"]:
            tstr = s["ISO52016_type_string"]
            ostr = s["ISO52016_orientation_string"]
            bnd = str(s.get("boundary", "OUTDOORS")).upper()
            zone = s.get("zone", None)
            adjacent_zone = s.get("adjacent_zone", None)
            key = (tstr, ostr, s["type"], bnd, zone, adjacent_zone)

            A = float(s.get("area", 0.0))
            U = float(s.get("u_value", 0.0))
            width_ = float(s.get("width", 0.0))
            height_ = float(s.get("height", 0.0))
            parapet_ = float(s.get("parapet", 0.0))
            g = float(s.get("g_value", 0.0)) if s.get("type") == "transparent" else 0.0
            svf = float(s.get("sky_view_factor", 0.0))
            hci = float(s.get("convective_heat_transfer_coefficient_internal", 0.0))
            hri = float(s.get("radiative_heat_transfer_coefficient_internal", 0.0))
            hce = float(s.get("convective_heat_transfer_coefficient_external", 0.0))
            hre = float(s.get("radiative_heat_transfer_coefficient_external", 0.0))
            a_sol = float(s.get("solar_absorptance", 0.0))
            Cth = float(s.get("thermal_capacity", 0.0))

            b = buckets[key]
            b["name"].append(s.get("name", "surface"))
            b["type"] = s["type"]
            b["ISO52016_type_string"] = tstr
            b["ISO52016_orientation_string"] = ostr
            b["boundary"] = bnd
            b["zone"] = zone
            b["adjacent_zone"] = adjacent_zone
            if b["name_adj_zone"] is None and s.get("name_adj_zone", None) is not None:
                b["name_adj_zone"] = s.get("name_adj_zone")
            if b["orientation"] is None and isinstance(s.get("orientation", None), dict):
                b["orientation"] = copy.deepcopy(s["orientation"])

            b["area"] += A
            b["uA"]   += U * A
            b["gA"]   += g * A
            b["svfA"] += svf * A
            b["hciA"] += hci * A
            b["hriA"] += hri * A
            b["hceA"] += hce * A
            b["hreA"] += hre * A
            b["a_sol_A"] += a_sol * A
            b["thermal_capacity"] += Cth
            b["width_sum"] += width_
            b["height_sum"] += height_
            b["parapet_sum"] += parapet_
            if b["type"] == "transparent":
                b["parapetA"] += parapet_ * A

        # Build new surfaces list
        new_surfaces = []
        for key, b in buckets.items():
            area_tot = float(b["area"])
            A = area_tot if area_tot > 0 else 1.0  # safety for area-weighted averages

            # Equivalent geometry for aggregated transparent surfaces:
            # assume side-by-side windows -> widths add; keep area coherence.
            if b["type"] == "transparent":
                width_equiv = float(b["width_sum"])
                if width_equiv > 0.0 and area_tot > 0.0:
                    height_equiv = area_tot / width_equiv
                else:
                    height_equiv = float(b["height_sum"])

                if area_tot > 0.0:
                    parapet_equiv = float(b["parapetA"]) / area_tot
                else:
                    parapet_equiv = float(b["parapet_sum"])
            else:
                width_equiv = float(b["width_sum"])
                height_equiv = float(b["height_sum"])
                parapet_equiv = float(b["parapet_sum"])

            agg = {
                "name": " + ".join(b["name"])[:120],  # trimmed
                "type": b["type"],
                "boundary": b["boundary"],
                "zone": b["zone"],
                "area": b["area"],
                "u_value": b["uA"] / A,
                "sky_view_factor": b["svfA"] / A,
                "solar_absorptance": b["a_sol_A"] / A,
                "thermal_capacity": b["thermal_capacity"],
                "ISO52016_type_string": b["ISO52016_type_string"],
                "ISO52016_orientation_string": b["ISO52016_orientation_string"],
                # carry typical window fields if present (area-weighted g)
                "g_value": b["gA"] / A if b["type"] == "transparent" else 0.0,
                # re-attach heat transfer coefficients (area-weighted)
                "convective_heat_transfer_coefficient_internal": b["hciA"] / A,
                "radiative_heat_transfer_coefficient_internal": b["hriA"] / A,
                "convective_heat_transfer_coefficient_external": b["hceA"] / A,
                "radiative_heat_transfer_coefficient_external": b["hreA"] / A,
                "width": width_equiv,
                "height": height_equiv,
                "parapet": parapet_equiv,
            }
            if b["adjacent_zone"] is not None:
                agg["adjacent_zone"] = b["adjacent_zone"]
            if b["name_adj_zone"] is not None:
                agg["name_adj_zone"] = b["name_adj_zone"]

            # Preserve the first valid orientation seen in the bucket.
            if isinstance(b["orientation"], dict):
                agg["orientation"] = copy.deepcopy(b["orientation"])
            else:
                orientation_map = {
                    "HOR": {"azimuth": 0, "tilt": 0},
                    "NV": {"azimuth": 0, "tilt": 90},
                    "EV": {"azimuth": 90, "tilt": 90},
                    "SV": {"azimuth": 180, "tilt": 90},
                    "WV": {"azimuth": 270, "tilt": 90},
                }
                agg["orientation"] = orientation_map.get(
                    b["ISO52016_orientation_string"],
                    {"azimuth": 0, "tilt": 90},
                )

            new_surfaces.append(agg)

        # Return a shallow-copied object with compacted surfaces
        new_bui = dict(building_object)
        new_bui["building_surface"] = new_surfaces
        return new_bui

    
    
    @classmethod
    def generate_category_profile(
            cls,
            building_object,
            occupants_schedule_workdays,
            occupants_schedule_weekend,
            appliances_schedule_workdays,
            appliances_schedule_weekend,
            lighting_schedule_workdays,
            lighting_schedule_weekend,
        ):
        """
        Generate category_profiles using the profiles defined in the BUI if present.
        If there are NO ventilation/heating/cooling profiles, use the default
        OCCUPANCY profiles (based on building_type_class).
        :param building_object: building object
        :param occupants_schedule_workdays: occupants schedule workdays
        :param occupants_schedule_weekend: occupants schedule weekend
        :param appliances_schedule_workdays: appliances schedule workdays
        :param appliances_schedule_weekend: appliances schedule weekend
        :param lighting_schedule_workdays: lighting schedule workdays
        :param lighting_schedule_weekend: lighting schedule weekend

        :return: dict category_profiles
        """

        import numpy as np

        bt = building_object["building"]["building_type_class"]

        # ---------------- helpers locali ----------------
        def _get_schedule_pair_for_bt(bt_key, workdays_map, weekend_map, name):
            """Give (weekday, weekend) from default dictionaries for the type bt_key."""
            if bt_key not in workdays_map or bt_key not in weekend_map:
                raise KeyError(f"{name}: '{bt_key}' not present in default profiles.")
            wd = np.asarray(workdays_map[bt_key], dtype=float)
            hd = np.asarray(weekend_map[bt_key], dtype=float)
            if wd.shape != (24,) or hd.shape != (24,):
                raise ValueError(f"{name}: default profiles must have 24 values.")
            return wd, hd

        def _pair_from_bui(profile_dict, name):
            """
            If profile_dict is present and valid, returns (weekday, weekend) as np.array(24).
            Altrimenti ritorna None.
            """
            if not profile_dict:
                return None
            hd_raw = profile_dict.get("weekend", profile_dict.get("holiday"))
            if "weekday" not in profile_dict or hd_raw is None:
                return None
            try:
                wd = np.asarray(profile_dict["weekday"], dtype=float)
                hd = np.asarray(hd_raw, dtype=float)
            except Exception:
                return None
            if wd.shape != (24,) or hd.shape != (24,):
                return None
            return wd, hd

        # ---------- OCCUPANCY ----------
        # 1) try to read from BUI (internal_gains -> 'occupants'); 2) otherwise use default for bt
        occ_entry = None
        for g in building_object["building_parameters"].get("internal_gains", []):
            if g.get("name") == "occupants":
                occ_entry = g
                break

        if occ_entry is not None:
            occ_pair = _pair_from_bui(occ_entry, "occupancy")
            if occ_pair is None:
                raise ValueError("occupancy: profilo BUI deve avere 24 valori.")
            occ_wd, occ_hd = occ_pair
        else:
            occ_wd, occ_hd = _get_schedule_pair_for_bt(
                bt, occupants_schedule_workdays, occupants_schedule_weekend, "occupancy"
            )

        # ---------- APPLIANCES ----------
        app_entry = None
        for g in building_object["building_parameters"].get("internal_gains", []):
            if g.get("name") == "appliances":
                app_entry = g
                break

        if app_entry is not None:
            app_pair = _pair_from_bui(app_entry, "appliances")
            if app_pair is None:
                raise ValueError("appliances: profilo BUI deve avere 24 valori.")
            app_wd, app_hd = app_pair
        else:
            app_wd, app_hd = _get_schedule_pair_for_bt(
                bt, appliances_schedule_workdays, appliances_schedule_weekend, "appliances"
            )

        # ---------- LIGHTING ----------
        lig_entry = None
        for g in building_object["building_parameters"].get("internal_gains", []):
            if g.get("name") == "lighting":
                lig_entry = g
                break

        if lig_entry is not None:
            lig_pair = _pair_from_bui(lig_entry, "lighting")
            if lig_pair is None:
                raise ValueError("lighting: profilo BUI deve avere 24 valori.")
            lig_wd, lig_hd = lig_pair
        else:
            lig_wd, lig_hd = _get_schedule_pair_for_bt(
                bt, lighting_schedule_workdays, lighting_schedule_weekend, "lighting"
            )

        # ---------- VENTILATION / HEATING / COOLING ----------
        # Rule: if no profiles in the BUI → use the OCCUPANCY default profiles (occ_wd/occ_hd)
        bp = building_object["building_parameters"]

        # ventilation
        pair = _pair_from_bui(bp.get("ventilation_profile"), "ventilation_profile")
        if pair is None:
            vent_wd, vent_hd = occ_wd.copy(), occ_hd.copy()   # <-- fallback a OCCUPANCY default
        else:
            vent_wd, vent_hd = pair

        # heating
        pair = _pair_from_bui(bp.get("heating_profile"), "heating_profile")
        if pair is None:
            heat_wd, heat_hd = occ_wd.copy(), occ_hd.copy()   # <-- fallback a OCCUPANCY default
        else:
            heat_wd, heat_hd = pair

        # cooling
        pair = _pair_from_bui(bp.get("cooling_profile"), "cooling_profile")
        if pair is None:
            cool_wd, cool_hd = occ_wd.copy(), occ_hd.copy()   # <-- fallback a OCCUPANCY default
        else:
            cool_wd, cool_hd = pair

        # ---------- build category_profiles ----------
        category_profiles = {
            "ventilation": {"weekday": vent_wd, "holiday": vent_hd},
            "heating":     {"weekday": heat_wd, "holiday": heat_hd},
            "cooling":     {"weekday": cool_wd, "holiday": cool_hd},
            "occupancy":   {"weekday": occ_wd,  "holiday": occ_hd},
            "lighting":    {"weekday": lig_wd,  "holiday": lig_hd},
            "appliances":  {"weekday": app_wd,  "holiday": app_hd},
        }

        return category_profiles

    @classmethod
    def simulate_envelope_multizone_free_floating(
        cls,
        building_object: dict,
        path_weather_file=None,
        weather_source="epw",
        include_solar=True,
        delta_Theta_er=11.0,
        dt_s=3600.0,
        warmup_hours=744,
        f_int_c=0.4,
        f_sol_c=0.1,
        use_profiles=True,
        include_internal_gains=True,
        include_ventilation=True,
        include_thermal_bridges=True,
        occupants_schedule_workdays=None,
        occupants_schedule_weekend=None,
        appliances_schedule_workdays=None,
        appliances_schedule_weekend=None,
        lighting_schedule_workdays=None,
        lighting_schedule_weekend=None,
        hvac_control_variable="operative",
        internal_convection_model=None,
        external_convection_model=None,
        external_convection_h_min=None,
        external_radiation_model=None,
        sky_temperature_model=None,
        external_emissivity_default=None,
        progress_log_every_steps=None,
        progress_logger=None,
    ):
        """
        Multizone envelope simulation with coupled internal partitions and ideal HVAC.
        Supports profile-driven setpoints, internal gains and ventilation terms.

        Output columns per zone:
          - T_air_<zone>
          - T_rad_<zone>
          - T_op_<zone>
          - Q_HVAC_<zone>  [W, +heating, -cooling]
          - mode_<zone>    ["H", "C", "0"]
          - Phi_int_<zone> [W]
          - H_ve_<zone>    [W/K]
          - H_ground_<zone> [W/K, ground-equivalent conductance]
          - Q_ground_<zone> [W, +building -> ground, -ground -> building]
          - night_purge_factor_<zone>  [1.0 when inactive]
          - night_purge_active_<zone>  [0/1]

        Additional global columns when ground-contact surfaces are present:
          - T_ground_virtual [degC]
          - Q_ground_surface_<surface> [W, +building -> ground]
          - Q_opaque_inside_surface_<surface> [W, +surface -> zone]

        HVAC control variable:
          - "operative": setpoint check/enforcement on T_op (default).
          - "air":       setpoint check/enforcement on T_air (EnergyPlus-like thermostat variable).

        Internal convection model:
          - "table": fixed h_ci from surface/table defaults (default).
          - "tarp":  dynamic h_ci (TARP/Walton), evaluated from previous-step air/surface temperatures.

        External convection model:
          - "table": fixed h_ce from surface/table defaults (default).
          - "doe2":  EnergyPlus DOE-2 wind-dependent correlation.
          - "mowitt": MoWiTT model (often used for glazing).
          - "blast": BLAST mixed forced+natural approximation.
          - "simplecombined": simple linear model h_ce = 4 + 4 * v.

        External long-wave radiation model:
          - "table": fixed h_re from surface/table defaults (default ISO-like behavior).
          - "dynamic": dynamic h_re(T_surf, T_sky) with geometric sky/ground view factors.

        Progress logging:
          - set `progress_log_every_steps` to an integer > 0 to emit periodic progress messages.
          - `progress_logger` can be any callable accepting a single string.
        """

        # ------------------------
        # 0) Weather
        # ------------------------
        sim_df = cls().Weather_data_bui(
            building_object, path_weather_file, weather_source=weather_source
        ).simulation_df
        sim_df = sim_df.copy()
        sim_df.index = pd.DatetimeIndex(sim_df.index)
        sim_df = sim_df.loc[~sim_df.index.duplicated(keep="first")].copy()
        sim_df = sim_df.sort_index()
        T2m_arr = _series_to_float_array(sim_df, "T2m")

        # ------------------------
        # 0.1) Time profiles (occupancy/heating/cooling/ventilation)
        # ------------------------
        zones_for_profiles = building_object.get("zones", None)
        if not zones_for_profiles:
            zones_for_profiles = [
                {
                    "name": "main",
                    "net_floor_area": float(building_object["building"]["net_floor_area"]),
                }
            ]

        zone_profile_columns = [
            "occupancy_profile",
            "appliances_profile",
            "lighting_profile",
            "heating_profile",
            "cooling_profile",
            "ventilation_profile",
        ]
        zone_profiles = {
            str(z["name"]): pd.DataFrame(
                {
                    "occupancy_profile": np.ones(len(sim_df), dtype=float),
                    "appliances_profile": np.ones(len(sim_df), dtype=float),
                    "lighting_profile": np.ones(len(sim_df), dtype=float),
                    "heating_profile": np.ones(len(sim_df), dtype=float),
                    "cooling_profile": np.ones(len(sim_df), dtype=float),
                    "ventilation_profile": np.ones(len(sim_df), dtype=float),
                },
                index=sim_df.index,
            )
            for z in zones_for_profiles
        }

        if occupants_schedule_workdays is None:
            occupants_schedule_workdays = getattr(
                iso16798_profiles,
                "occupants_schedule_workdays",
                {"Residential_apartment": [1.0] * 24},
            )
        if occupants_schedule_weekend is None:
            occupants_schedule_weekend = getattr(
                iso16798_profiles,
                "occupants_schedule_weekend",
                {"Residential_apartment": [1.0] * 24},
            )
        if appliances_schedule_workdays is None:
            appliances_schedule_workdays = getattr(
                iso16798_profiles,
                "appliances_schedule_workdays",
                {"Residential_apartment": [1.0] * 24},
            )
        if appliances_schedule_weekend is None:
            appliances_schedule_weekend = getattr(
                iso16798_profiles,
                "appliances_schedule_weekend",
                {"Residential_apartment": [1.0] * 24},
            )
        if lighting_schedule_workdays is None:
            lighting_schedule_workdays = getattr(
                iso16798_profiles,
                "lighting_schedule_workdays",
                {"Residential_apartment": [1.0] * 24},
            )
        if lighting_schedule_weekend is None:
            lighting_schedule_weekend = getattr(
                iso16798_profiles,
                "lighting_schedule_weekend",
                {"Residential_apartment": [1.0] * 24},
            )

        def _clone_profile_dict(profile_dict):
            if not isinstance(profile_dict, dict):
                return None
            hd_raw = profile_dict.get("weekend", profile_dict.get("holiday"))
            if "weekday" not in profile_dict or hd_raw is None:
                return None
            return {
                "weekday": list(profile_dict["weekday"]),
                # Keep "weekend" as canonical key in BUI-like payloads.
                "weekend": list(hd_raw),
            }

        def _upsert_internal_gain_profile(internal_gains_list, gain_name, profile_dict):
            profile_clone = _clone_profile_dict(profile_dict)
            if profile_clone is None:
                return
            found = False
            for gain in internal_gains_list:
                if gain.get("name") == gain_name:
                    gain["weekday"] = profile_clone["weekday"]
                    gain["weekend"] = profile_clone["weekend"]
                    found = True
                    break
            if not found:
                internal_gains_list.append(
                    {
                        "name": gain_name,
                        "weekday": profile_clone["weekday"],
                        "weekend": profile_clone["weekend"],
                    }
                )

        def _build_zone_profile_input(zone_obj):
            def _normalize_bt(bt_value):
                if bt_value is None:
                    return None
                bt_str = str(bt_value).strip()
                if bt_str == "" or bt_str.lower() in {"none", "null", "nan"}:
                    return None
                return bt_str

            zone_building_type = zone_obj.get(
                "building_type_class",
                building_object.get("building", {}).get("building_type_class"),
            )
            zone_building_type = _normalize_bt(zone_building_type) or "Residential_apartment"
            zone_profile_bui = {
                "building": {"building_type_class": zone_building_type},
                "building_parameters": copy.deepcopy(building_object.get("building_parameters", {})),
            }
            bp = zone_profile_bui["building_parameters"]
            bp.setdefault("internal_gains", [])

            for key in ("ventilation_profile", "heating_profile", "cooling_profile"):
                cloned = _clone_profile_dict(zone_obj.get(key))
                if cloned is not None:
                    bp[key] = cloned

            if isinstance(zone_obj.get("internal_gains"), list):
                bp["internal_gains"] = copy.deepcopy(zone_obj.get("internal_gains"))
            elif not isinstance(bp.get("internal_gains"), list):
                bp["internal_gains"] = []

            _upsert_internal_gain_profile(
                bp["internal_gains"],
                "occupants",
                zone_obj.get("occupants_profile", zone_obj.get("occupancy_profile")),
            )
            _upsert_internal_gain_profile(
                bp["internal_gains"],
                "appliances",
                zone_obj.get("appliances_profile"),
            )
            _upsert_internal_gain_profile(
                bp["internal_gains"],
                "lighting",
                zone_obj.get("lighting_profile", zone_obj.get("ligthing_profile")),
            )

            return zone_profile_bui

        def _align_profile_raw_to_weather(profile_raw: pd.DataFrame):
            if profile_raw is None or len(profile_raw) == 0:
                return None
            aligned = profile_raw.copy().reset_index(drop=True)
            target_len = len(sim_df)
            warmup_h = int(max(0, warmup_hours or 0))

            # Common case: weather has 13 months (Dec warm-up + full year), profile has 12 months.
            if warmup_h > 0 and (len(aligned) + warmup_h) == target_len and len(aligned) >= warmup_h:
                aligned = pd.concat([aligned.iloc[-warmup_h:].copy(), aligned], ignore_index=True)
            elif len(aligned) < target_len:
                reps = int(np.ceil(target_len / len(aligned)))
                aligned = pd.concat([aligned] * reps, ignore_index=True).iloc[:target_len].copy()
            elif len(aligned) > target_len:
                aligned = aligned.iloc[:target_len].copy()

            aligned.index = sim_df.index
            return aligned

        if use_profiles:
            try:
                country_calendar = get_country_code_from_latlon(
                    building_object["building"]["latitude"],
                    building_object["building"]["longitude"],
                )
            except Exception:
                country_calendar = "IT"

            for zone_obj in zones_for_profiles:
                zname = str(zone_obj.get("name", "main"))
                if zname not in zone_profiles:
                    zone_profiles[zname] = pd.DataFrame(
                        {k: np.ones(len(sim_df), dtype=float) for k in zone_profile_columns},
                        index=sim_df.index,
                    )
                try:
                    profile_input = _build_zone_profile_input(zone_obj)
                    category_profiles = cls().generate_category_profile(
                        profile_input,
                        occupants_schedule_workdays,
                        occupants_schedule_weekend,
                        appliances_schedule_workdays,
                        appliances_schedule_weekend,
                        lighting_schedule_workdays,
                        lighting_schedule_weekend,
                    )
                    gen = HourlyProfileGenerator(
                        country=country_calendar,
                        num_months=13,
                        category_profiles=category_profiles,
                    )
                    profile_raw = gen.generate()
                    aligned = _align_profile_raw_to_weather(profile_raw)
                    if aligned is None:
                        continue
                    for col in zone_profile_columns:
                        if col in aligned.columns:
                            arr = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).clip(lower=0.0)
                            zone_profiles[zname][col] = arr.values
                except Exception:
                    # Fallback to flat profiles for this zone only.
                    continue

        def _profile_value(zone_name: str, col: str, tstep: int, default: float = 1.0) -> float:
            if zone_name not in zone_profiles:
                zone_name = next(iter(zone_profiles.keys()), None)
            if zone_name is None:
                return float(default)
            zone_profile_df = zone_profiles[zone_name]
            if col not in zone_profile_df.columns:
                return float(default)
            v = float(zone_profile_df[col].iloc[tstep])
            if not np.isfinite(v):
                return float(default)
            return max(0.0, v)

        # ------------------------
        # 1) Zone indexing (air nodes)
        # ------------------------
        zones_input = building_object.get("zones", None)
        zones = copy.deepcopy(zones_input) if zones_input else None
        if not zones:
            zones = [
                {
                    "name": "main",
                    "net_floor_area": float(building_object["building"]["net_floor_area"]),
                }
            ]

        zone_names = [z["name"] for z in zones]
        Z = len(zone_names)
        z_idx = {name: i for i, name in enumerate(zone_names)}
        h_ci_model = _resolve_internal_convection_model(
            building_object,
            internal_convection_model,
        )
        h_ce_model = _resolve_external_convection_model(
            building_object,
            external_convection_model,
        )
        h_ce_min = _resolve_external_convection_h_min(
            building_object,
            external_convection_h_min,
        )
        h_re_model = _resolve_external_radiation_model(
            building_object,
            external_radiation_model,
        )
        sky_t_model = _resolve_sky_temperature_model(
            building_object,
            sky_temperature_model,
        )
        eps_ext_default = _resolve_default_external_emissivity(
            building_object,
            external_emissivity_default,
        )
        hvac_control_mode = str(hvac_control_variable).strip().lower()
        if hvac_control_mode not in ("operative", "air"):
            raise ValueError("hvac_control_variable must be 'operative' or 'air'")

        # setpoints/capacities/ventilation: zone override -> global fallback -> default
        global_sp = building_object.get("building_parameters", {}).get("temperature_setpoints", {})
        global_caps = building_object.get("building_parameters", {}).get("system_capacities", {})
        global_vent = building_object.get("building_parameters", {}).get("ventilation", {})
        def _normalize_building_type(bt_value):
            if bt_value is None:
                return None
            bt_str = str(bt_value).strip()
            if bt_str == "" or bt_str.lower() in {"none", "null", "nan"}:
                return None
            return bt_str

        global_building_type = _normalize_building_type(
            building_object.get("building", {}).get("building_type_class")
        )
        if global_building_type is None:
            global_building_type = "Residential_apartment"

        def _zone_sp(zone_obj, key, default):
            if key in zone_obj and zone_obj[key] is not None:
                return float(zone_obj[key])
            if key in global_sp and global_sp[key] is not None:
                return float(global_sp[key])
            return float(default)

        def _zone_cap(zone_obj, key, default=np.inf):
            if key in zone_obj and zone_obj[key] is not None:
                return max(0.0, float(zone_obj[key]))
            if key in global_caps and global_caps[key] is not None:
                return max(0.0, float(global_caps[key]))
            return float(default)

        def _zone_vent(zone_obj, key, default):
            if key in zone_obj and zone_obj[key] is not None:
                return zone_obj[key]
            if key in global_vent and global_vent[key] is not None:
                return global_vent[key]
            return default

        for zone in zones:
            zone.setdefault("heating_setpoint", _zone_sp(zone, "heating_setpoint", 20.0))
            zone.setdefault("cooling_setpoint", _zone_sp(zone, "cooling_setpoint", 26.0))
            zone.setdefault("heating_setback", _zone_sp(zone, "heating_setback", 16.0))
            zone.setdefault("cooling_setback", _zone_sp(zone, "cooling_setback", 30.0))
            zone.setdefault("heating_capacity", _zone_cap(zone, "heating_capacity", np.inf))
            zone.setdefault("cooling_capacity", _zone_cap(zone, "cooling_capacity", np.inf))
            zone_bt = _normalize_building_type(zone.get("building_type_class", global_building_type))
            zone["building_type_class"] = zone_bt if zone_bt is not None else global_building_type
            zone.setdefault("ventilation_type", _zone_vent(zone, "ventilation_type", "none"))
            zone.setdefault("flow_rate_per_person", float(_zone_vent(zone, "flow_rate_per_person", 0.0)))
            zone.setdefault(
                "custom_heat_transfer_coefficient_ventilation",
                float(_zone_vent(zone, "custom_heat_transfer_coefficient_ventilation", 0.0)),
            )
            zone.setdefault(
                "infiltration_flow_per_exterior_area_m3_s_m2",
                float(_zone_vent(zone, "infiltration_flow_per_exterior_area_m3_s_m2", 0.0)),
            )
            zone.setdefault(
                "infiltration_coeff_constant",
                float(_zone_vent(zone, "infiltration_coeff_constant", 0.0)),
            )
            zone.setdefault(
                "infiltration_coeff_temperature",
                float(_zone_vent(zone, "infiltration_coeff_temperature", 0.0)),
            )
            zone.setdefault(
                "infiltration_coeff_velocity",
                float(_zone_vent(zone, "infiltration_coeff_velocity", 0.0)),
            )
            zone.setdefault(
                "infiltration_coeff_velocity_squared",
                float(_zone_vent(zone, "infiltration_coeff_velocity_squared", 0.0)),
            )
            zone.setdefault(
                "infiltration_include_transparent_area",
                bool(_zone_vent(zone, "infiltration_include_transparent_area", True)),
            )
            zone.setdefault(
                "infiltration_exterior_area_mode",
                str(_zone_vent(zone, "infiltration_exterior_area_mode", "outdoors_only")),
            )
            zone.setdefault(
                "infiltration_wind_reduction_factor",
                float(_zone_vent(zone, "infiltration_wind_reduction_factor", 1.0)),
            )
            zone.setdefault(
                "infiltration_effective_leakage_area_m2",
                float(_zone_vent(zone, "infiltration_effective_leakage_area_m2", 0.0)),
            )
            zone.setdefault(
                "infiltration_stack_coefficient",
                float(_zone_vent(zone, "infiltration_stack_coefficient", 0.0)),
            )
            zone.setdefault(
                "infiltration_wind_coefficient",
                float(_zone_vent(zone, "infiltration_wind_coefficient", 0.0)),
            )
            zone.setdefault(
                "infiltration_schedule_multiplier",
                float(_zone_vent(zone, "infiltration_schedule_multiplier", 1.0)),
            )

        # ------------------------
        # 2) Surface preprocessing
        # ------------------------
        surfaces = building_object["building_surface"]
        Nsurf = len(surfaces)

        # Keep compatibility with existing helper methods:
        # OP/W for outdoors, GR for ground, AD for adiabatic, ADJ for internal partition
        for surf in surfaces:
            bnd = surf.get("boundary", "OUTDOORS").upper()
            if bnd == "GROUND":
                surf["ISO52016_type_string"] = "GR"
            elif bnd == "ADIABATIC":
                surf["ISO52016_type_string"] = "AD"
            elif bnd == "INTERNAL":
                surf["ISO52016_type_string"] = "ADJ"
            else:
                surf["ISO52016_type_string"] = (
                    "W" if surf.get("type", "").lower() == "transparent" else "OP"
                )

        def _orientation_string(surf):
            ori_existing = str(surf.get("ISO52016_orientation_string", "")).upper()
            if ori_existing in {"HOR", "NV", "EV", "SV", "WV"}:
                return ori_existing
            ori = surf.get("orientation", {}) or {}
            az = ori.get("azimuth", None)
            tilt = ori.get("tilt", None)
            try:
                az_f = float(az) % 360.0
                tilt_f = float(tilt)
            except Exception:
                return "SV"
            if abs(tilt_f) < 1e-6:
                return "HOR"
            if abs(tilt_f - 90.0) < 1e-6:
                candidates = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
                labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
                diffs = np.abs(((az_f - candidates + 180.0) % 360.0) - 180.0)
                return str(labels[int(np.argmin(diffs))])
            return "HOR" if tilt_f < 45.0 else "SV"

        # coefficients and orientation defaults
        for surf in surfaces:
            surf.setdefault("convective_heat_transfer_coefficient_internal", 2.5)
            # Keep coherence with single-zone core defaults.
            surf.setdefault("radiative_heat_transfer_coefficient_internal", 5.13)
            bnd = surf.get("boundary", "OUTDOORS").upper()
            if bnd == "INTERNAL":
                surf.setdefault(
                    "convective_heat_transfer_coefficient_external",
                    float(surf.get("convective_heat_transfer_coefficient_internal", 2.5)),
                )
                surf.setdefault(
                    "radiative_heat_transfer_coefficient_external",
                    float(surf.get("radiative_heat_transfer_coefficient_internal", 5.13)),
                )
            else:
                surf.setdefault("convective_heat_transfer_coefficient_external", 20.0)
                surf.setdefault("radiative_heat_transfer_coefficient_external", 4.14)
            surf.setdefault("sky_view_factor", 0.0)
            surf["ISO52016_orientation_string"] = _orientation_string(surf)

        # Keep link to pre-aggregation windows so multizone transmitted solar gains
        # can use area-weighted shading factors from per-window W_<name> columns.
        shading_components_by_zone_orientation = {}
        for surf in surfaces:
            if str(surf.get("type", "")).lower() != "transparent":
                continue
            if str(surf.get("boundary", "OUTDOORS")).upper() != "OUTDOORS":
                continue
            win_name = str(surf.get("name", "")).strip()
            if not win_name:
                continue
            try:
                win_area = float(surf.get("area", 0.0))
            except Exception:
                continue
            if not np.isfinite(win_area) or win_area <= 0.0:
                continue
            zname = surf.get("zone", zone_names[0])
            if zname not in z_idx:
                zname = zone_names[0]
            ori = str(surf.get("ISO52016_orientation_string", "SV")).upper()
            key = (zname, ori)
            shading_components_by_zone_orientation.setdefault(key, []).append((win_name, win_area))

        # Keep alignment with single-zone core:
        # aggregate surfaces with equal ISO type/orientation while preserving
        # zone/boundary partitioning.
        building_object = cls()._aggregate_surfaces_by_direction(building_object)
        surfaces = building_object["building_surface"]
        Nsurf = len(surfaces)

        def _get(surf, key, default):
            value = surf.get(key, None)
            return float(value) if value is not None else float(default)

        # ------------------------
        # 3) Nodes for surfaces
        # ------------------------
        nodes = cls().Number_of_nodes_element(building_object)

        # force boundary ADIABATIC -> 0 nodes
        for i, surf in enumerate(surfaces):
            if surf.get("boundary", "").upper() == "ADIABATIC":
                nodes.Pln[i] = 0

        # rebuild PlnSum and Rn coherently
        PlnSum = np.array([0] * Nsurf, dtype=int)
        for i in range(1, Nsurf):
            PlnSum[i] = PlnSum[i - 1] + int(nodes.Pln[i - 1])
        Rn = int(PlnSum[-1] + int(nodes.Pln[-1]) + 1)
        nodes.PlnSum = PlnSum
        nodes.Rn = Rn

        # ------------------------
        # 4) Coefficients (ISO52016)
        # ------------------------
        h_pli_eli = cls().Conduttance_node_of_element(building_object).h_pli_eli
        kappa_pli_eli = cls().Areal_heat_capacity_of_element(building_object).kappa_pli_eli
        a_sol_pli_eli = cls().Solar_absorption_of_element(building_object).a_sol_pli_eli

        # ------------------------
        # 5) Total system size: air nodes + surface nodes
        # ------------------------
        Ntot = Z + (nodes.Rn - 1)

        def _sys_row_from_surface_ri(ri: int) -> int:
            return Z + (ri - 1)  # ri in [1..Rn-1]

        # ------------------------
        # 6) Initial conditions
        # ------------------------
        Tstepn = len(sim_df)
        Theta = np.full((Ntot,), 20.0, dtype=float)  # state at t-1
        zone_output_buffers = []
        for name in zone_names:
            zone_output_buffers.append(
                {
                    "name": name,
                    "zi": int(z_idx[name]),
                    "T_air": np.empty(Tstepn, dtype=float),
                    "T_rad": np.empty(Tstepn, dtype=float),
                    "T_op": np.empty(Tstepn, dtype=float),
                    "Q_HVAC": np.empty(Tstepn, dtype=float),
                    "mode": np.empty(Tstepn, dtype=object),
                    "Phi_int": np.empty(Tstepn, dtype=float),
                    "H_ve": np.empty(Tstepn, dtype=float),
                    "S_ve": np.empty(Tstepn, dtype=float),
                    "T_ve_source_eq": np.full(Tstepn, np.nan, dtype=float),
                    "Q_ve": np.empty(Tstepn, dtype=float),
                    "H_ground": np.empty(Tstepn, dtype=float),
                    "Q_ground": np.empty(Tstepn, dtype=float),
                    "night_purge_factor": np.empty(Tstepn, dtype=float),
                    "night_purge_active": np.empty(Tstepn, dtype=np.int8),
                }
            )

        # ------------------------
        # 7) Ground virtual temperature (optional)
        # ------------------------
        has_ground = any(surf["ISO52016_type_string"] == "GR" for surf in surfaces)
        t_Th = (
            cls().Temp_calculation_of_ground(
                building_object,
                path_weather_file=path_weather_file,
                weather_source=weather_source,
            )
            if has_ground
            else None
        )
        ground_links, zone_h_ground = _build_multizone_ground_flux_links(
            surfaces=surfaces,
            nodes=nodes,
            zone_names=zone_names,
            z_idx=z_idx,
            ground_data=t_Th,
            sys_row_from_surface_ri=_sys_row_from_surface_ri,
        )
        ground_temperature_buffer = np.full(Tstepn, np.nan, dtype=float)
        ground_surface_output_buffers = [
            {
                "surface_token": str(link["surface_token"]),
                "Q_ground": np.empty(Tstepn, dtype=float),
            }
            for link in ground_links
        ]
        opaque_inside_links = _build_multizone_opaque_inside_flux_links(
            surfaces=surfaces,
            nodes=nodes,
            zone_names=zone_names,
            z_idx=z_idx,
            sys_row_from_surface_ri=_sys_row_from_surface_ri,
            h_pli_eli=h_pli_eli,
        )
        opaque_inside_surface_output_buffers = [
            {
                "surface_token": str(link["surface_token"]),
                "Q_inside": np.empty(Tstepn, dtype=float),
            }
            for link in opaque_inside_links
        ]

        # Thermal bridges distributed by zone area share
        zone_area_vec = np.array([float(z.get("net_floor_area", 50.0)) for z in zones], dtype=float)
        zone_area_tot = float(np.sum(zone_area_vec)) if float(np.sum(zone_area_vec)) > 0 else 1.0
        H_tb_tot = float(t_Th.thermal_bridge_heat) if (t_Th is not None and include_thermal_bridges) else 0.0
        H_tb_zone = H_tb_tot * (zone_area_vec / zone_area_tot)
        C_air_zone = 10000.0 * zone_area_vec
        # Align with core behavior: adiabatic element capacity is lumped in zone air node once.
        C_ad_zone = np.zeros(Z, dtype=float)
        for surf in surfaces:
            if surf.get("ISO52016_type_string") != "AD":
                continue
            zname = surf.get("zone", zone_names[0])
            if zname in z_idx:
                kappa_ad = float(surf.get("thermal_capacity", 0.0))  # [J/(m2 K)]
                area_ad = float(surf.get("area", 1.0))               # [m2]
                if np.isfinite(kappa_ad) and np.isfinite(area_ad) and area_ad > 0.0:
                    C_ad_zone[z_idx[zname]] += kappa_ad * area_ad    # [J/K]
        C_air_zone += C_ad_zone

        # Internal-node maps for radiative distribution of transmitted solar gains.
        zone_rad_nodes = [[] for _ in range(Z)]  # (row, area)
        zone_rad_area = np.zeros(Z, dtype=float)
        for Eli, surf in enumerate(surfaces):
            n_nodes = int(nodes.Pln[Eli])
            if n_nodes == 0:
                continue
            area_s = float(surf.get("area", 0.0))
            zA = surf.get("zone", zone_names[0])
            if zA in z_idx:
                r_in = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + (n_nodes - 1))
                ziA = z_idx[zA]
                zone_rad_nodes[ziA].append((r_in, area_s))
                zone_rad_area[ziA] += area_s
            if surf.get("boundary", "OUTDOORS").upper() == "INTERNAL":
                zB = surf.get("adjacent_zone", None)
                if zB in z_idx:
                    r_out = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + 0)
                    ziB = z_idx[zB]
                    zone_rad_nodes[ziB].append((r_out, area_s))
                    zone_rad_area[ziB] += area_s

        # Zone-facing surface nodes used for long-wave radiative exchange between surfaces.
        # Entries are (row, area, h_ri_face), with coefficients on the side facing that zone.
        zone_radiative_faces = [[] for _ in range(Z)]
        zone_radiative_area = np.zeros(Z, dtype=float)
        for Eli, surf in enumerate(surfaces):
            n_nodes = int(nodes.Pln[Eli])
            if n_nodes == 0:
                continue
            area_s = float(surf.get("area", 0.0))
            if area_s <= 0.0:
                continue

            h_ri_a = _get(surf, "radiative_heat_transfer_coefficient_internal", 5.13)
            zA = surf.get("zone", zone_names[0])
            if zA in z_idx:
                r_in = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + (n_nodes - 1))
                ziA = z_idx[zA]
                zone_radiative_faces[ziA].append((r_in, area_s, h_ri_a))
                zone_radiative_area[ziA] += area_s

            if surf.get("boundary", "OUTDOORS").upper() == "INTERNAL":
                zB = surf.get("adjacent_zone", None)
                if zB in z_idx:
                    h_rb = _get(
                        surf,
                        "radiative_heat_transfer_coefficient_external",
                        h_ri_a,
                    )
                    r_out = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + 0)
                    ziB = z_idx[zB]
                    zone_radiative_faces[ziB].append((r_out, area_s, h_rb))
                    zone_radiative_area[ziB] += area_s

        # Dynamic air/surface links used in zone air balances.
        # With TARP, h_ci is re-evaluated every timestep from previous temperatures.
        def _build_zone_air_links_from_theta(theta_ref: np.ndarray):
            zone_air_links_t = [[] for _ in range(Z)]  # (surface_row, h_tot_WK)
            h_ci_internal_t = np.full(Nsurf, np.nan, dtype=float)

            for Eli, surf in enumerate(surfaces):
                n_nodes = int(nodes.Pln[Eli])
                if n_nodes == 0:
                    continue
                area_s = float(surf.get("area", 0.0))
                if area_s <= 0.0:
                    continue

                h_ci_tab = _get(surf, "convective_heat_transfer_coefficient_internal", 2.5)
                zA = surf.get("zone", zone_names[0])
                if zA in z_idx:
                    r_in = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + (n_nodes - 1))
                    h_ci_a = _internal_h_ci_value(
                        surf,
                        h_ci_model,
                        t_air_c=float(theta_ref[z_idx[zA]]),
                        t_surf_c=float(theta_ref[r_in]),
                        fallback_h_ci=h_ci_tab,
                    )
                    h_ci_internal_t[Eli] = h_ci_a
                    # Long-wave radiation is handled separately by
                    # _add_zone_longwave_radiative_exchange().
                    h_tot_a = h_ci_a * area_s
                    zone_air_links_t[z_idx[zA]].append((r_in, h_tot_a))

                if surf.get("boundary", "OUTDOORS").upper() == "INTERNAL":
                    zB = surf.get("adjacent_zone", None)
                    if zB in z_idx:
                        r_out = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + 0)
                        h_cb = _get(
                            surf,
                            "convective_heat_transfer_coefficient_external",
                            h_ci_tab,
                        )
                        h_tot_b = h_cb * area_s
                        zone_air_links_t[z_idx[zB]].append((r_out, h_tot_b))

            zone_hsurf_sum_t = np.array(
                [sum(float(h) for _, h in links) for links in zone_air_links_t],
                dtype=float,
            )
            return zone_air_links_t, zone_hsurf_sum_t, h_ci_internal_t

        def _zone_radiant_temperature(theta_vec: np.ndarray) -> np.ndarray:
            t_rad = np.zeros(Z, dtype=float)
            for zi in range(Z):
                area_tot = float(zone_rad_area[zi])
                if area_tot <= 0.0:
                    t_rad[zi] = float(theta_vec[zi])
                    continue
                num = 0.0
                for (R_surf, area_surf) in zone_rad_nodes[zi]:
                    if R_surf < 0 or R_surf >= len(theta_vec):
                        continue
                    num += float(area_surf) * float(theta_vec[R_surf])
                t_rad[zi] = num / area_tot
            return t_rad

        def _zone_air_radiant_operative(theta_vec: np.ndarray):
            t_air = np.asarray(theta_vec[:Z], dtype=float)
            t_rad = _zone_radiant_temperature(theta_vec)
            t_op = 0.5 * (t_air + t_rad)
            return t_air, t_rad, t_op

        def _apply_operative_setpoint_constraint(
            A_mat: np.ndarray,
            B_vec: np.ndarray,
            zi: int,
            t_set: float,
        ) -> None:
            A_mat[zi, :] = 0.0
            area_tot = float(zone_rad_area[zi])
            if area_tot <= 0.0 or len(zone_rad_nodes[zi]) == 0:
                # Fallback for zones without radiative faces.
                A_mat[zi, zi] = 1.0
                B_vec[zi] = float(t_set)
                return

            # Enforce operative setpoint:
            # T_air + T_rad = 2*T_set, where
            # T_rad = sum_j(w_j * T_surf_j), w_j = A_j / A_tot.
            A_mat[zi, zi] = 1.0
            for (R_surf, area_surf) in zone_rad_nodes[zi]:
                if float(area_surf) <= 0.0:
                    continue
                A_mat[zi, R_surf] += float(area_surf) / area_tot
            B_vec[zi] = 2.0 * float(t_set)

        def _apply_air_setpoint_constraint(
            A_mat: np.ndarray,
            B_vec: np.ndarray,
            zi: int,
            t_set: float,
        ) -> None:
            A_mat[zi, :] = 0.0
            A_mat[zi, zi] = 1.0
            B_vec[zi] = float(t_set)

        # Per-zone proxy objects for internal gains / ventilation models
        zone_surfaces = {name: [] for name in zone_names}
        for surf in surfaces:
            zn = surf.get("zone", zone_names[0])
            bnd = surf.get("boundary", "OUTDOORS").upper()
            if zn in zone_surfaces and bnd != "INTERNAL":
                zone_surfaces[zn].append(surf)

        internal_gains_input = building_object.get("building_parameters", {}).get("internal_gains", [])
        zone_proxies = {}
        for zone in zones:
            zname = zone["name"]
            bld = dict(building_object.get("building", {}))
            bld["net_floor_area"] = float(zone.get("net_floor_area", 50.0))
            bld["zone_name"] = zname
            if zone.get("building_type_class") is not None:
                bld["building_type_class"] = zone.get("building_type_class")
            zone_internal_gains = zone.get("internal_gains", internal_gains_input)
            zone_vent_params = {
                "ventilation_type": str(zone.get("ventilation_type", "none")).strip().lower(),
                "flow_rate_per_person": float(zone.get("flow_rate_per_person", 0.0)),
                "custom_heat_transfer_coefficient_ventilation": float(
                    zone.get("custom_heat_transfer_coefficient_ventilation", 0.0)
                ),
                "infiltration_flow_per_exterior_area_m3_s_m2": float(
                    zone.get("infiltration_flow_per_exterior_area_m3_s_m2", 0.0)
                ),
                "infiltration_coeff_constant": float(zone.get("infiltration_coeff_constant", 0.0)),
                "infiltration_coeff_temperature": float(zone.get("infiltration_coeff_temperature", 0.0)),
                "infiltration_coeff_velocity": float(zone.get("infiltration_coeff_velocity", 0.0)),
                "infiltration_coeff_velocity_squared": float(
                    zone.get("infiltration_coeff_velocity_squared", 0.0)
                ),
                "infiltration_include_transparent_area": bool(
                    zone.get("infiltration_include_transparent_area", True)
                ),
                "infiltration_exterior_area_mode": str(
                    zone.get("infiltration_exterior_area_mode", "outdoors_only")
                ),
                "infiltration_wind_reduction_factor": float(
                    zone.get("infiltration_wind_reduction_factor", 1.0)
                ),
                "infiltration_effective_leakage_area_m2": float(
                    zone.get("infiltration_effective_leakage_area_m2", 0.0)
                ),
                "infiltration_stack_coefficient": float(
                    zone.get("infiltration_stack_coefficient", 0.0)
                ),
                "infiltration_wind_coefficient": float(
                    zone.get("infiltration_wind_coefficient", 0.0)
                ),
                "infiltration_schedule_multiplier": float(
                    zone.get("infiltration_schedule_multiplier", 1.0)
                ),
            }
            # Components: zone-local overrides global; global is the fallback
            _zone_vent_obj = zone.get("ventilation", {})
            if isinstance(_zone_vent_obj, dict) and "components" in _zone_vent_obj:
                zone_vent_params["components"] = _zone_vent_obj["components"]
            elif "components" in global_vent:
                zone_vent_params["components"] = global_vent["components"]

            # Store zone-specific volume so constant_ach uses per-zone air not
            # building-total air.  Must be set before building the proxy so the
            # bld dict carries the correct value.
            _zone_vol_raw = (
                zone.get("zone_volume_m3")
                or zone.get("zone_volume")
                or zone.get("volume")
            )
            if _zone_vol_raw is not None:
                bld["zone_volume_m3"] = float(_zone_vol_raw)

            zone_proxies[zname] = {
                "zone_name": zname,
                "building": bld,
                "building_surface": list(zone_surfaces.get(zname, [])),
                "internal_gains": copy.deepcopy(zone_internal_gains),
                "building_parameters": {"ventilation": zone_vent_params},
            }

        def _zone_internal_gain_w(zone_obj: dict, tstep: int) -> float:
            if not include_internal_gains:
                return 0.0

            bt = zone_obj.get("building_type_class", None)
            if bt is None:
                return 0.0

            # fallback to first known type if an unknown building class is provided
            if "internal_gains_occupants" in globals() and bt not in internal_gains_occupants:
                bt = next(iter(internal_gains_occupants.keys()), None)
            if bt is None:
                return 0.0

            zname = str(zone_obj.get("name", zone_names[0]))
            if zname not in zone_proxies:
                zname = zone_names[0]
            a_use = float(zone_obj.get("net_floor_area", 50.0))
            h_occ = _profile_value(zname, "occupancy_profile", tstep, 1.0)
            h_app = _profile_value(zname, "appliances_profile", tstep, 1.0)
            h_light = _profile_value(zname, "lighting_profile", tstep, 1.0)
            try:
                phi = VentilationInternalGains(zone_proxies[zname]).internal_gains(
                    building_type_class=bt,
                    a_use=a_use,
                    unconditioned_zones_nearby=False,
                    h_occup=h_occ,
                    h_app=h_app,
                    h_light=h_light,
                )
            except Exception:
                phi = 0.0
            phi = float(phi) if np.isfinite(phi) else 0.0
            return max(0.0, phi)

        def _zone_ventilation_h_wk(
            zone_obj: dict,
            theta_air_prev: float,
            tstep: int,
            T_out: float,
        ) -> tuple:  # (VentilationBoundary, float, int)
            _empty = VentilationBoundary(streams=())
            if not include_ventilation:
                return _empty, 1.0, 0

            zname = str(zone_obj.get("name", zone_names[0]))
            if zname not in zone_proxies:
                zname = zone_names[0]

            proxy = zone_proxies[zname]
            vent_cfg = proxy.get("building_parameters", {}).get("ventilation", {})
            vent_type = str(vent_cfg.get("ventilation_type", "none")).strip().lower()
            has_components = "components" in vent_cfg

            if not has_components and vent_type in ("none", "off", "disabled", ""):
                return _empty, 1.0, 0

            ws = float(sim_df["WS10m"].iloc[tstep]) if "WS10m" in sim_df.columns else 0.0
            profile_mult = _profile_value(zname, "ventilation_profile", tstep, 1.0)

            # Zone volume: zone-specific keys only; do NOT fall back to the
            # global building volume (proxy["building"] is a copy of the global
            # building dict and would give full-building air to every zone).
            _bld = proxy.get("building", {})
            zone_vol = float(
                zone_obj.get("zone_volume_m3")
                or zone_obj.get("zone_volume")
                or zone_obj.get("volume")
                or _bld.get("zone_volume_m3")  # set from zone data during proxy build
                or 0.0
            )

            # Per-component schedules: each component may have a "profile" key
            # naming a profile in the profile registry.  Components without a
            # profile key always run at full capacity (1.0) so infiltration
            # remains active independently of the mechanical schedule.
            # Only the six standard profile columns are supported; unknown names
            # warn once and fall back to 1.0.
            _known_profiles = {
                "occupancy_profile", "appliances_profile", "lighting_profile",
                "heating_profile", "cooling_profile", "ventilation_profile",
            }
            _comp_mult: dict = {}
            for _comp in vent_cfg.get("components", []):
                _cname = str(_comp.get("name", "")).strip()
                _prof = _comp.get("profile")
                if _cname and _prof is not None:
                    if _prof not in _known_profiles:
                        import warnings as _w
                        _w.warn(
                            f"Zone {zname!r} component {_cname!r}: profile {_prof!r} is not "
                            f"one of the supported profile columns {sorted(_known_profiles)}. "
                            "Using 1.0.",
                            stacklevel=2,
                        )
                    _comp_mult[_cname] = _profile_value(zname, _prof, tstep, 1.0)

            try:
                base_bdy = resolve_ventilation_boundary(
                    proxy,
                    float(theta_air_prev),
                    float(T_out),
                    float(ws),
                    profile_multiplier=profile_mult,
                    component_multipliers=_comp_mult if _comp_mult else None,
                    zone_volume_m3=zone_vol if zone_vol > 0.0 else None,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Zone {zname!r}: ventilation boundary resolution failed "
                    f"(ventilation_type={vent_type!r}): {exc}"
                ) from exc

            purge_factor_applied = 1.0
            purge_active = 0

            # Optional summer night purge: represented as an additional outdoor-air stream.
            # H_purge = max(0, boost_factor - 1) * H_base so total equals boost_factor * H_base.
            purge_cfg = zone_obj.get("summer_night_purge", None)
            if not isinstance(purge_cfg, dict):
                purge_cfg = (
                    building_object.get("building_parameters", {}).get("summer_night_purge", None)
                    if isinstance(building_object, dict)
                    else None
                )
            if isinstance(purge_cfg, dict) and bool(purge_cfg.get("enabled", False)):
                month_start, month_end = 6, 9
                months = purge_cfg.get("months", None)
                if isinstance(months, (list, tuple)) and len(months) == 2:
                    try:
                        month_start = int(months[0])
                        month_end = int(months[1])
                    except Exception:
                        month_start, month_end = 6, 9
                else:
                    try:
                        month_start = int(purge_cfg.get("month_start", purge_cfg.get("start_month", 6)))
                    except Exception:
                        month_start = 6
                    try:
                        month_end = int(purge_cfg.get("month_end", purge_cfg.get("end_month", 9)))
                    except Exception:
                        month_end = 9
                month_start = max(1, min(12, month_start))
                month_end = max(1, min(12, month_end))

                hour_start, hour_end = 22, 7
                hours = purge_cfg.get("hours", None)
                if isinstance(hours, (list, tuple)) and len(hours) == 2:
                    try:
                        hour_start = int(hours[0])
                        hour_end = int(hours[1])
                    except Exception:
                        hour_start, hour_end = 22, 7
                else:
                    try:
                        hour_start = int(purge_cfg.get("hour_start", 22))
                    except Exception:
                        hour_start = 22
                    try:
                        hour_end = int(purge_cfg.get("hour_end", 7))
                    except Exception:
                        hour_end = 7
                hour_start = hour_start % 24
                hour_end = hour_end % 24

                try:
                    delta_t_min = float(purge_cfg.get("delta_t_min", 0.5))
                except Exception:
                    delta_t_min = 0.5
                if not np.isfinite(delta_t_min):
                    delta_t_min = 0.5

                try:
                    boost_factor = float(purge_cfg.get("boost_factor", 1.0))
                except Exception:
                    boost_factor = 1.0
                if not np.isfinite(boost_factor) or boost_factor <= 1.0:
                    boost_factor = 1.0

                ts = pd.to_datetime(sim_df.index[tstep], errors="coerce")
                if pd.notna(ts):
                    month_now = int(ts.month)
                    hour_now = int(ts.hour)
                else:
                    month_now = int(_month_at_tstep(tstep) + 1)
                    hour_now = int(tstep % 24)

                if month_start <= month_end:
                    month_ok = month_start <= month_now <= month_end
                else:
                    month_ok = (month_now >= month_start) or (month_now <= month_end)

                if hour_start == hour_end:
                    # Equal start/end means full-day availability.
                    hour_ok = True
                elif hour_start <= hour_end:
                    hour_ok = hour_start <= hour_now < hour_end
                else:
                    hour_ok = (hour_now >= hour_start) or (hour_now < hour_end)

                if month_ok and hour_ok:
                    if float(theta_air_prev) - float(T_out) >= delta_t_min:
                        H_base = base_bdy.heat_transfer_coefficient_w_k
                        H_purge = max(0.0, boost_factor - 1.0) * H_base
                        if H_purge > 0.0:
                            purge_stream = VentilationStream(
                                name="summer_night_purge",
                                heat_transfer_coefficient_w_k=H_purge,
                                source_temperature_c=float(T_out),
                                category="outdoor_air",
                            )
                            base_bdy = VentilationBoundary(
                                streams=base_bdy.streams + (purge_stream,)
                            )
                        purge_factor_applied = float(boost_factor)
                        purge_active = 1

            return base_bdy, purge_factor_applied, purge_active

        def _zone_solar_transmitted_w(tstep: int) -> np.ndarray:
            phi_sol = np.zeros(Z, dtype=float)
            if not include_solar:
                return phi_sol
            for surf in surfaces:
                if surf.get("ISO52016_type_string") != "W":
                    continue
                if surf.get("boundary", "OUTDOORS").upper() != "OUTDOORS":
                    continue
                zname = surf.get("zone", zone_names[0])
                if zname not in z_idx:
                    continue
                ori = surf.get("ISO52016_orientation_string", "SV")
                col = f"I_sol_tot_{ori}"
                if col not in sim_df.columns:
                    continue
                g_val = float(surf.get("g_value", 0.0))
                area = float(surf.get("area", 0.0))
                i_tot = float(sim_df[col].iloc[tstep])
                try:
                    ffr_wi = float(surf.get("frame_area_fraction", 0.25))
                except Exception:
                    ffr_wi = 0.25
                if not np.isfinite(ffr_wi):
                    ffr_wi = 0.25
                ffr_wi = min(1.0, max(0.0, ffr_wi))
                f_sh = cls._surface_shading_factor_from_timeseries(
                    sim_df=sim_df,
                    tstep=tstep,
                    surface=surf,
                    shading_components_by_zone_orientation=shading_components_by_zone_orientation,
                    default_zone=zone_names[0],
                )
                gain = max(0.0, g_val * area * i_tot * f_sh * (1.0 - ffr_wi))
                phi_sol[z_idx[zname]] += gain
            return phi_sol

        def _month_at_tstep(tstep: int) -> int:
            try:
                m = int(sim_df.index.month[tstep])
                if 1 <= m <= 12:
                    return m - 1
            except Exception:
                pass
            if "day of year" in sim_df.columns:
                try:
                    doy = int(sim_df["day of year"].iloc[tstep])
                    doy = max(1, min(366, doy))
                    base_year = 2008 if doy == 366 else 2009
                    m = int((pd.Timestamp(base_year, 1, 1) + pd.Timedelta(days=doy - 1)).month)
                    return m - 1
                except Exception:
                    pass
            ts = pd.to_datetime(sim_df.index[tstep], errors="coerce")
            if pd.notna(ts):
                return int(ts.month) - 1
            return 0

        # ------------------------
        # helper: assemble linear system
        # ------------------------
        def assemble_system(
            theta_prev: np.ndarray,
            tstep: int,
            phi_int_z: np.ndarray,
            h_ve_z: np.ndarray,
            s_ve_z: np.ndarray,
            phi_sol_z: np.ndarray,
        ):
            A = np.zeros((Ntot, Ntot), dtype=float)
            B = np.zeros((Ntot,), dtype=float)
            zone_air_links_t, zone_hsurf_sum_t, h_ci_internal_t = _build_zone_air_links_from_theta(theta_prev)

            T_out = float(T2m_arr[tstep])
            u_wind = 0.0
            if "WS10m" in sim_df.columns:
                u_wind = float(pd.to_numeric(sim_df["WS10m"], errors="coerce").iloc[tstep])
            if not np.isfinite(u_wind):
                u_wind = 0.0
            T_sky = _sky_temperature_from_weather(sim_df, tstep, sky_t_model)
            for zi in range(Z):
                C_air = float(C_air_zone[zi])
                h_ve = float(h_ve_z[zi])
                s_ve = float(s_ve_z[zi])
                h_tb = float(H_tb_zone[zi])
                phi_int = float(phi_int_z[zi])
                phi_sol_conv = float(f_sol_c) * float(phi_sol_z[zi])

                A[zi, zi] += C_air / dt_s + h_ve + h_tb
                B[zi] += (C_air / dt_s) * float(theta_prev[zi])
                B[zi] += s_ve + h_tb * T_out
                B[zi] += float(f_int_c) * phi_int + phi_sol_conv

            # surfaces
            for Eli, surf in enumerate(surfaces):
                n_nodes = int(nodes.Pln[Eli])
                if n_nodes == 0:
                    continue

                A_s = float(surf["area"])
                h_ci = h_ci_internal_t[Eli]
                if not np.isfinite(h_ci):
                    h_ci = _get(surf, "convective_heat_transfer_coefficient_internal", 2.5)
                h_ce_tab = _get(surf, "convective_heat_transfer_coefficient_external", 20.0)
                h_re_tab = _get(surf, "radiative_heat_transfer_coefficient_external", 4.14)
                svf = _get(surf, "sky_view_factor", 0.0)

                zone_A = surf.get("zone", zone_names[0])
                if zone_A not in z_idx:
                    raise ValueError(
                        f"Surface '{surf.get('name')}' has unknown zone '{zone_A}'"
                    )
                air_A = z_idx[zone_A]

                bnd = surf.get("boundary", "OUTDOORS").upper()
                is_internal = bnd == "INTERNAL"
                if is_internal:
                    zone_B = surf.get("adjacent_zone", None)
                    if zone_B is None or zone_B not in z_idx:
                        raise ValueError(
                            f"Internal surface '{surf.get('name')}' missing/invalid adjacent_zone"
                        )
                    air_B = z_idx[zone_B]
                else:
                    air_B = None

                for Pli in range(n_nodes):
                    ri = 1 + nodes.PlnSum[Eli] + Pli
                    R = _sys_row_from_surface_ri(ri)

                    # Surface equations assembled in absolute power [W].
                    cap = (float(kappa_pli_eli[Pli, Eli]) * A_s) / dt_s
                    A[R, R] += cap
                    B[R] += cap * float(theta_prev[R])

                    # conduction to neighboring nodes
                    if Pli > 0:
                        h = float(h_pli_eli[Pli - 1, Eli]) * A_s
                        r_prev = _sys_row_from_surface_ri(ri - 1)
                        A[R, R] += h
                        A[R, r_prev] -= h
                    if Pli < n_nodes - 1:
                        h = float(h_pli_eli[Pli, Eli]) * A_s
                        r_next = _sys_row_from_surface_ri(ri + 1)
                        A[R, R] += h
                        A[R, r_next] -= h

                    # internal face towards zone_A (last node)
                    if Pli == n_nodes - 1:
                        # Surface-to-surface long-wave radiation is assembled
                        # separately below; air/surface coupling stays convective.
                        h_tot_a = h_ci * A_s
                        A[R, R] += h_tot_a
                        A[R, air_A] -= h_tot_a
                        A[air_A, air_A] += h_tot_a
                        A[air_A, R] -= h_tot_a

                    # external/adjacent face (first node)
                    if Pli == 0:
                        if is_internal:
                            # Side B can use its own "external" coefficients.
                            h_cb = _get(
                                surf,
                                "convective_heat_transfer_coefficient_external",
                                h_ci,
                            )
                            h_tot_b = h_cb * A_s
                            A[R, R] += h_tot_b
                            A[R, air_B] -= h_tot_b
                            A[air_B, air_B] += h_tot_b
                            A[air_B, R] -= h_tot_b
                        else:
                            tstr = surf["ISO52016_type_string"]
                            if tstr == "OP" or tstr == "W":
                                h_ce_use = _dynamic_external_convection_h(
                                    surface=surf,
                                    T_surf_C=float(theta_prev[R]),
                                    T_air_C=float(T_out),
                                    u_wind_ms=float(u_wind),
                                    model=h_ce_model,
                                    h_min=h_ce_min,
                                    fallback_h_ce=h_ce_tab,
                                )
                                if h_re_model == "dynamic":
                                    eps_ext = _surface_external_emissivity(
                                        surf,
                                        eps_ext_default,
                                    )
                                    h_re_use, T_ref_re = _dynamic_external_radiative_h_and_ref(
                                        surface=surf,
                                        T_surf_C=float(theta_prev[R]),
                                        T_sky_C=float(T_sky),
                                        T_air_C=float(T_out),
                                        epsilon=eps_ext,
                                    )
                                    h_ext = (h_ce_use + h_re_use) * A_s
                                    A[R, R] += h_ext
                                    B[R] += (h_ce_use * T_out + h_re_use * T_ref_re) * A_s
                                else:
                                    h_ext = (h_ce_use + h_re_tab) * A_s
                                    A[R, R] += h_ext
                                    B[R] += h_ext * T_out
                                    phi_sky = svf * h_re_tab * float(delta_Theta_er) * A_s
                                    B[R] += -phi_sky
                                # solar absorption on external node: opaque only
                                if include_solar and tstr == "OP":
                                    ori = surf.get("ISO52016_orientation_string", "SV")
                                    col = f"I_sol_tot_{ori}"
                                    if col in sim_df.columns:
                                        B[R] += float(a_sol_pli_eli[Pli, Eli]) * A_s * float(
                                            sim_df[col].iloc[tstep]
                                        )
                            elif tstr == "GR" and t_Th is not None:
                                month = _month_at_tstep(tstep)
                                T_gr = float(t_Th.Theta_gr_ve[month])
                                # R_gr_ve is area-specific [m2K/W]; convert it to a
                                # per-surface conductance so total ground coupling
                                # scales with the actual slab area of each zone.
                                h_gr = _ground_conductance_w_per_k(A_s, t_Th)
                                A[R, R] += h_gr
                                B[R] += h_gr * T_gr

            # Long-wave radiative exchange between zone-facing internal surfaces.
            # Analogous to the single-zone core formulation, but in absolute-power [W] rows.
            for zi in range(Z):
                cls._add_zone_longwave_radiative_exchange(A, zone_radiative_faces[zi])

            # Radiative part of transmitted solar gains distributed on internal surface nodes.
            for zi in range(Z):
                phi_sol_rad = (1.0 - float(f_sol_c)) * float(phi_sol_z[zi])
                area_tot = float(zone_rad_area[zi])
                if area_tot <= 0.0 or abs(phi_sol_rad) < 1e-12:
                    continue
                for (R_surf, area_surf) in zone_rad_nodes[zi]:
                    if float(area_surf) <= 0.0:
                        continue
                    # Distribute zone-level radiant gain [W] by internal-surface area share.
                    B[R_surf] += phi_sol_rad * (float(area_surf) / area_tot)

            return A, B, zone_air_links_t, zone_hsurf_sum_t

        def _solve_with_diag_guard(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            diag_min = max(1e-6, 1e-9 * np.linalg.norm(A, ord=np.inf))
            d = np.diag(A).copy()
            np.maximum(d, diag_min, out=d)
            np.fill_diagonal(A, d)
            return np.linalg.solve(A, B)

        def _compute_q_hvac_from_balance(
            theta_new: np.ndarray,
            theta_prev: np.ndarray,
            mode_vec,
            h_ve_z: np.ndarray,
            s_ve_z: np.ndarray,
            phi_int_z: np.ndarray,
            phi_sol_z: np.ndarray,
            T_out: float,
            zone_air_links_step,
            zone_hsurf_sum_step: np.ndarray,
        ) -> np.ndarray:
            q = np.zeros(Z, dtype=float)
            for zi in range(Z):
                if mode_vec[zi] == "0":
                    continue
                C = float(C_air_zone[zi])
                h_ve = float(h_ve_z[zi])
                s_ve = float(s_ve_z[zi])
                h_tb = float(H_tb_zone[zi])
                T_air = float(theta_new[zi])
                T_prev = float(theta_prev[zi])
                sum_h = float(zone_hsurf_sum_step[zi])
                sum_hTs = 0.0
                for (R_surf, h_tot) in zone_air_links_step[zi]:
                    sum_hTs += float(h_tot) * float(theta_new[R_surf])

                gains_conv = float(f_int_c) * float(phi_int_z[zi]) + float(f_sol_c) * float(phi_sol_z[zi])
                q[zi] = (
                    (C / dt_s + sum_h + h_ve + h_tb) * T_air
                    - sum_hTs
                    - (C / dt_s) * T_prev
                    - s_ve
                    - h_tb * T_out
                    - gains_conv
                )
            return np.asarray(q, dtype=float)

        def _clamp_q_hvac_by_mode(q_vec: np.ndarray, mode_vec) -> np.ndarray:
            q_out = np.asarray(q_vec, dtype=float).copy()
            for zi in range(Z):
                m = str(mode_vec[zi])
                if m == "H":
                    q_out[zi] = max(0.0, float(q_out[zi]))
                elif m == "C":
                    q_out[zi] = min(0.0, float(q_out[zi]))
                else:
                    q_out[zi] = 0.0
            return q_out

        # ------------------------
        # 8) Time loop (free-float -> setpoint solve -> capacity-limited solve)
        # ------------------------
        start_idx = 0
        if warmup_hours:
            warmup_h = int(max(0, warmup_hours))
            if len(sim_df) > warmup_h:
                try:
                    # Warm-up present when a December block is prepended before January.
                    if int(sim_df.index[0].month) == 12 and int(sim_df.index[warmup_h].month) == 1:
                        start_idx = warmup_h
                except Exception:
                    if len(sim_df) >= (8760 + warmup_h):
                        start_idx = warmup_h

        for t in range(Tstepn):
            # compute timestep forcings once (used by all solves)
            T_out = float(sim_df["T2m"].iloc[t])
            phi_int_z_t = np.zeros(Z, dtype=float)
            h_ve_z_t = np.zeros(Z, dtype=float)
            s_ve_z_t = np.zeros(Z, dtype=float)
            purge_factor_z_t = np.ones(Z, dtype=float)
            purge_active_z_t = np.zeros(Z, dtype=int)
            for zi, zone in enumerate(zones):
                phi_int_z_t[zi] = _zone_internal_gain_w(zone, t)
                vent_bdy, purge_factor_z_t[zi], purge_active_z_t[zi] = _zone_ventilation_h_wk(
                    zone, Theta[zi], t, T_out
                )
                h_ve_z_t[zi] = vent_bdy.heat_transfer_coefficient_w_k
                s_ve_z_t[zi] = vent_bdy.source_term_w
            phi_sol_z_t = _zone_solar_transmitted_w(t)

            # Base matrix for this timestep
            A_base, B_base, zone_air_links_t, zone_hsurf_sum_t = assemble_system(
                Theta,
                t,
                phi_int_z_t,
                h_ve_z_t,
                s_ve_z_t,
                phi_sol_z_t,
            )

            # 8.1 free-floating solve
            Theta_ff = _solve_with_diag_guard(A_base.copy(), B_base.copy())
            Theta_air_ff, _, Theta_op_ff = _zone_air_radiant_operative(Theta_ff)

            # 8.2 decide mode per zone
            mode = ["0"] * Z
            T_set = np.full(Z, np.nan, dtype=float)
            for zi, zone in enumerate(zones):
                zname_mode = str(zone.get("name", zone_names[zi]))
                if use_profiles:
                    heat_active = _profile_value(zname_mode, "heating_profile", t, 1.0) > 0.0
                    cool_active = _profile_value(zname_mode, "cooling_profile", t, 1.0) > 0.0
                    Th = float(zone["heating_setpoint"] if heat_active else zone["heating_setback"])
                    Tc = float(zone["cooling_setpoint"] if cool_active else zone["cooling_setback"])
                else:
                    Th = float(zone["heating_setpoint"])
                    Tc = float(zone["cooling_setpoint"])
                if hvac_control_mode == "air":
                    T0 = float(Theta_air_ff[zi])
                else:
                    T0 = float(Theta_op_ff[zi])
                if T0 < Th:
                    mode[zi] = "H"
                    T_set[zi] = Th
                elif T0 > Tc:
                    mode[zi] = "C"
                    T_set[zi] = Tc
                else:
                    mode[zi] = "0"

            # 8.3 setpoint solve with ideal clamps where mode != 0
            A_sp = A_base.copy()
            B_sp = B_base.copy()

            for zi in range(Z):
                if mode[zi] != "0":
                    if hvac_control_mode == "air":
                        _apply_air_setpoint_constraint(A_sp, B_sp, zi, float(T_set[zi]))
                    else:
                        _apply_operative_setpoint_constraint(A_sp, B_sp, zi, float(T_set[zi]))

            Theta_sp = _solve_with_diag_guard(A_sp, B_sp)
            Q_req = _compute_q_hvac_from_balance(
                Theta_sp,
                Theta,
                mode,
                h_ve_z_t,
                s_ve_z_t,
                phi_int_z_t,
                phi_sol_z_t,
                T_out,
                zone_air_links_t,
                zone_hsurf_sum_t,
            )
            Q_req = _clamp_q_hvac_by_mode(Q_req, mode)

            # 8.4 check capacities (if provided) and, if needed, re-solve with limited HVAC power
            capacity_limited = np.zeros(Z, dtype=bool)
            Q_target = np.zeros(Z, dtype=float)
            for zi in range(Z):
                if mode[zi] == "0":
                    Q_target[zi] = 0.0
                    continue
                if mode[zi] == "H":
                    cap_h = float(zones[zi].get("heating_capacity", np.inf))
                    if np.isfinite(cap_h) and Q_req[zi] > cap_h:
                        capacity_limited[zi] = True
                        Q_target[zi] = cap_h
                    else:
                        Q_target[zi] = Q_req[zi]
                elif mode[zi] == "C":
                    cap_c = float(zones[zi].get("cooling_capacity", np.inf))
                    q_c_min = -max(0.0, cap_c)
                    if np.isfinite(cap_c) and Q_req[zi] < q_c_min:
                        capacity_limited[zi] = True
                        Q_target[zi] = q_c_min
                    else:
                        Q_target[zi] = Q_req[zi]

            if np.any(capacity_limited):
                A_cap = A_base.copy()
                B_cap = B_base.copy()

                for zi in range(Z):
                    if mode[zi] == "0":
                        continue
                    if capacity_limited[zi]:
                        # Air equation right-hand side includes known HVAC term
                        B_cap[zi] += float(Q_target[zi])
                    else:
                        # Keep ideal operative setpoint where capacity is sufficient
                        if hvac_control_mode == "air":
                            _apply_air_setpoint_constraint(A_cap, B_cap, zi, float(T_set[zi]))
                        else:
                            _apply_operative_setpoint_constraint(A_cap, B_cap, zi, float(T_set[zi]))

                Theta_new = _solve_with_diag_guard(A_cap, B_cap)
                Q_hvac = _compute_q_hvac_from_balance(
                    Theta_new,
                    Theta,
                    mode,
                    h_ve_z_t,
                    s_ve_z_t,
                    phi_int_z_t,
                    phi_sol_z_t,
                    T_out,
                    zone_air_links_t,
                    zone_hsurf_sum_t,
                )
            else:
                Theta_new = Theta_sp
                Q_hvac = Q_req

            Q_hvac = _clamp_q_hvac_by_mode(Q_hvac, mode)

            # advance state
            Theta = Theta_new
            Theta_air_t, Theta_rad_t, Theta_op_t = _zone_air_radiant_operative(Theta_new)
            month_t = _month_at_tstep(t)
            T_ground_t, q_ground_zone_t, q_ground_surface_t = _ground_fluxes_from_state(
                theta_state=Theta_new,
                month_index=month_t,
                ground_data=t_Th,
                ground_links=ground_links,
                zone_names=zone_names,
            )
            q_opaque_inside_surface_t = _opaque_inside_fluxes_from_state(
                theta_state=Theta_new,
                opaque_inside_links=opaque_inside_links,
            )
            ground_temperature_buffer[t] = float(T_ground_t) if np.isfinite(T_ground_t) else np.nan

            # save outputs
            ts = sim_df.index[t]
            for zone_out in zone_output_buffers:
                zi = zone_out["zi"]
                zone_out["T_air"][t] = float(Theta_air_t[zi])
                zone_out["T_rad"][t] = float(Theta_rad_t[zi])
                zone_out["T_op"][t] = float(Theta_op_t[zi])
                zone_out["Q_HVAC"][t] = float(Q_hvac[zi])
                zone_out["mode"][t] = mode[zi]
                zone_out["Phi_int"][t] = float(phi_int_z_t[zi])
                zone_out["H_ve"][t] = float(h_ve_z_t[zi])
                zone_out["S_ve"][t] = float(s_ve_z_t[zi])
                _h_ve_i = float(h_ve_z_t[zi])
                _s_ve_i = float(s_ve_z_t[zi])
                zone_out["T_ve_source_eq"][t] = (
                    _s_ve_i / _h_ve_i if _h_ve_i > 0.0 else np.nan
                )
                zone_out["Q_ve"][t] = _h_ve_i * float(Theta_air_t[zi]) - _s_ve_i
                zone_out["H_ground"][t] = float(zone_h_ground[zi]) if zi < len(zone_h_ground) else 0.0
                zone_out["Q_ground"][t] = float(q_ground_zone_t.get(str(zone_out["name"]), 0.0))
                zone_out["night_purge_factor"][t] = float(purge_factor_z_t[zi])
                zone_out["night_purge_active"][t] = int(purge_active_z_t[zi])
            for surface_out in ground_surface_output_buffers:
                surface_out["Q_ground"][t] = float(
                    q_ground_surface_t.get(surface_out["surface_token"], 0.0)
                )
            for surface_out in opaque_inside_surface_output_buffers:
                surface_out["Q_inside"][t] = float(
                    q_opaque_inside_surface_t.get(surface_out["surface_token"], 0.0)
                )

            if progress_logger is not None:
                try:
                    log_every = int(progress_log_every_steps)
                except Exception:
                    log_every = 0
                if log_every > 0:
                    step_i = t + 1
                    if step_i == 1 or step_i == Tstepn or (step_i % log_every) == 0:
                        progress_logger(
                            "Multizone V1 progress: "
                            f"{step_i}/{Tstepn} "
                            f"({100.0 * step_i / max(1, Tstepn):.1f}%), "
                            f"timestamp={ts}"
                        )

        out_data = {}
        if len(ground_links) > 0:
            out_data["T_ground_virtual"] = ground_temperature_buffer
        for zone_out in zone_output_buffers:
            name = str(zone_out["name"])
            out_data[f"T_air_{name}"] = zone_out["T_air"]
            out_data[f"T_rad_{name}"] = zone_out["T_rad"]
            out_data[f"T_op_{name}"] = zone_out["T_op"]
            out_data[f"Q_HVAC_{name}"] = zone_out["Q_HVAC"]
            out_data[f"mode_{name}"] = zone_out["mode"]
            out_data[f"Phi_int_{name}"] = zone_out["Phi_int"]
            out_data[f"H_ve_{name}"] = zone_out["H_ve"]
            out_data[f"S_ve_{name}"] = zone_out["S_ve"]
            out_data[f"T_ve_source_eq_{name}"] = zone_out["T_ve_source_eq"]
            out_data[f"Q_ve_{name}"] = zone_out["Q_ve"]
            out_data[f"H_ground_{name}"] = zone_out["H_ground"]
            out_data[f"Q_ground_{name}"] = zone_out["Q_ground"]
            out_data[f"night_purge_factor_{name}"] = zone_out["night_purge_factor"]
            out_data[f"night_purge_active_{name}"] = zone_out["night_purge_active"]
        for surface_out in ground_surface_output_buffers:
            out_data[f"Q_ground_surface_{surface_out['surface_token']}"] = surface_out["Q_ground"]
        for surface_out in opaque_inside_surface_output_buffers:
            out_data[f"Q_opaque_inside_surface_{surface_out['surface_token']}"] = surface_out["Q_inside"]

        out = pd.DataFrame(out_data, index=sim_df.index)
        out = out.iloc[start_idx:].copy()
        return out

    @classmethod
    def Temperature_and_Energy_needs_calculation_multizone(
        cls,
        building_object,
        path_weather_file=None,
        weather_source="epw",
        include_solar=True,
        warmup_hours=744,
        **kwargs,
    ):
        """
        Multizone hourly/annual energy-need calculation.
        This extends the multizone coupled model with profile-driven setpoints,
        ventilation and internal gains, and returns annual KPIs per zone.

        Returns:
            hourly_results (pd.DataFrame):
                Contains T_air_*, T_rad_*, T_op_*, Q_HVAC_*, mode_*, Phi_int_*, H_ve_*,
                H_ground_*, Q_ground_* and, when present, T_ground_virtual plus
                per-surface Q_ground_surface_* and Q_opaque_inside_surface_* columns.
            annual_results (pd.DataFrame):
                One row per zone with heating/cooling annual needs in Wh
                (plus explicit kWh columns) and specific values.
        """
        from .generate_profile import HourlyProfileGenerator, get_country_code_from_latlon  # lazy
        hourly_results = cls.simulate_envelope_multizone_free_floating(
            building_object=building_object,
            path_weather_file=path_weather_file,
            weather_source=weather_source,
            include_solar=include_solar,
            warmup_hours=warmup_hours,
            **kwargs,
        )

        zones = building_object.get("zones", None)
        if not zones:
            zones = [
                {
                    "name": "main",
                    "net_floor_area": float(building_object["building"]["net_floor_area"]),
                }
            ]

        annual_rows = []
        dt_h = _infer_timestep_hours_from_index(hourly_results.index, default=1.0)
        for zone in zones:
            zname = zone["name"]
            qcol = f"Q_HVAC_{zname}"
            if qcol not in hourly_results.columns:
                continue

            area = float(zone.get("net_floor_area", building_object["building"].get("net_floor_area", 0.0)))
            q_h_w = pd.to_numeric(hourly_results[qcol], errors="coerce").fillna(0.0).clip(lower=0.0)
            q_c_w = (-pd.to_numeric(hourly_results[qcol], errors="coerce").fillna(0.0).clip(upper=0.0))
            q_h = _integrate_power_series_to_energy_wh(q_h_w, default_dt_h=dt_h)
            q_c = _integrate_power_series_to_energy_wh(q_c_w, default_dt_h=dt_h)

            annual_rows.append(
                {
                    "zone": zname,
                    "Q_H_annual": q_h,
                    "Q_C_annual": q_c,
                    "Q_H_annual_per_sqm": (q_h / area) if area > 0 else 0.0,
                    "Q_C_annual_per_sqm": (q_c / area) if area > 0 else 0.0,
                    "Q_H_annual_kWh": q_h / 1000.0,
                    "Q_C_annual_kWh": q_c / 1000.0,
                    "Q_H_annual_kWh_per_sqm": (q_h / 1000.0 / area) if area > 0 else 0.0,
                    "Q_C_annual_kWh_per_sqm": (q_c / 1000.0 / area) if area > 0 else 0.0,
                    "time_step_h": dt_h,
                }
            )

        annual_results = pd.DataFrame(annual_rows)
        return hourly_results, annual_results

    @classmethod
    def _build_single_zone_building_object_for_core(cls, building_object: dict, zone_name: str) -> dict:
        """
        Build a single-zone BUI view for the legacy `_Temperature_and_Energy_needs_calculation_core`.
        INTERNAL partitions are converted to adiabatic surfaces for the per-zone core run.
        """
        def _normalize_building_type(bt_value):
            if bt_value is None:
                return None
            bt_str = str(bt_value).strip()
            if bt_str == "" or bt_str.lower() in {"none", "null", "nan"}:
                return None
            return bt_str

        bui = copy.deepcopy(building_object)
        zones = building_object.get("zones", []) or [
            {
                "name": "main",
                "net_floor_area": float(building_object["building"]["net_floor_area"]),
            }
        ]
        zone = next((z for z in zones if z.get("name") == zone_name), None)
        if zone is None:
            raise ValueError(f"Unknown zone '{zone_name}'")

        zone_area = float(zone.get("net_floor_area", building_object["building"].get("net_floor_area", 50.0)))

        bui["building"] = copy.deepcopy(building_object.get("building", {}))
        bui["zone_name"] = zone_name
        bui["building"]["zone_name"] = zone_name
        bui["building"]["net_floor_area"] = zone_area
        bui["building"]["adj_zones_present"] = False
        bui["building"]["number_adj_zone"] = 0
        # Zone-specific volume: set from zone dict if present, then remove any
        # inherited global keys so the legacy lookup cannot silently use building-total
        # airflow for per-zone constant_ach.
        _hybrid_zone_vol = (
            zone.get("zone_volume_m3")
            or zone.get("zone_volume")
            or zone.get("volume")
        )
        # Remove ALL inherited global volume keys before optionally setting
        # the zone-specific one. zone_volume_m3 must be cleared too, otherwise
        # a global "zone_volume_m3" in building_object["building"] survives the
        # deepcopy and is preferred by the legacy lookup at building.zone_volume_m3.
        for _vk in ("volume", "zone_volume", "zone_volume_m3"):
            bui["building"].pop(_vk, None)
        if _hybrid_zone_vol is not None:
            bui["building"]["zone_volume_m3"] = float(_hybrid_zone_vol)

        zone_bt = _normalize_building_type(zone.get("building_type_class"))
        global_bt = _normalize_building_type(building_object.get("building", {}).get("building_type_class"))
        bui["building"]["building_type_class"] = zone_bt or global_bt or "Residential_apartment"

        bui["adjacent_zones"] = []
        selected_surfaces = []
        for surf in building_object.get("building_surface", []):
            s = copy.deepcopy(surf)
            bnd = str(s.get("boundary", "OUTDOORS")).upper()
            s_zone = s.get("zone", zone_name)
            s_adj = s.get("adjacent_zone", None)

            if bnd == "INTERNAL":
                if s_zone == zone_name or s_adj == zone_name:
                    s["boundary"] = "ADIABATIC"
                    s["type"] = "adiabatic"
                    s["zone"] = zone_name
                    s.pop("adjacent_zone", None)
                    s["sky_view_factor"] = 0.0
                    s["name"] = f"{s.get('name', 'internal')}_{zone_name}_adiabatic"
                    s.setdefault("name_adj_zone", None)
                    selected_surfaces.append(s)
            elif s_zone == zone_name:
                s.setdefault("name_adj_zone", None)
                selected_surfaces.append(s)

        if len(selected_surfaces) == 0:
            raise ValueError(f"Zone '{zone_name}' has no surfaces for core simulation")

        bui["building_surface"] = selected_surfaces

        bp = bui.setdefault("building_parameters", {})
        sp = bp.setdefault("temperature_setpoints", {})
        sp["heating_setpoint"] = float(zone.get("heating_setpoint", sp.get("heating_setpoint", 20.0)))
        sp["cooling_setpoint"] = float(zone.get("cooling_setpoint", sp.get("cooling_setpoint", 26.0)))
        sp["heating_setback"] = float(zone.get("heating_setback", sp.get("heating_setback", 16.0)))
        sp["cooling_setback"] = float(zone.get("cooling_setback", sp.get("cooling_setback", 30.0)))

        caps = bp.setdefault("system_capacities", {})
        if zone.get("heating_capacity") is not None:
            caps["heating_capacity"] = float(zone["heating_capacity"])
        if zone.get("cooling_capacity") is not None:
            caps["cooling_capacity"] = float(zone["cooling_capacity"])

        vent = bp.setdefault("ventilation", {})
        vent_type = str(zone.get("ventilation_type", "none")).strip().lower()
        if vent_type in ("none", "off", "disabled", ""):
            vent_type = "custom"
        vent["ventilation_type"] = vent_type
        vent["flow_rate_per_person"] = float(zone.get("flow_rate_per_person", 0.0))
        vent["custom_heat_transfer_coefficient_ventilation"] = float(
            zone.get("custom_heat_transfer_coefficient_ventilation", 0.0)
        )
        vent["infiltration_flow_per_exterior_area_m3_s_m2"] = float(
            zone.get("infiltration_flow_per_exterior_area_m3_s_m2", 0.0)
        )
        vent["infiltration_coeff_constant"] = float(zone.get("infiltration_coeff_constant", 0.0))
        vent["infiltration_coeff_temperature"] = float(zone.get("infiltration_coeff_temperature", 0.0))
        vent["infiltration_coeff_velocity"] = float(zone.get("infiltration_coeff_velocity", 0.0))
        vent["infiltration_coeff_velocity_squared"] = float(
            zone.get("infiltration_coeff_velocity_squared", 0.0)
        )
        vent["infiltration_include_transparent_area"] = bool(
            zone.get("infiltration_include_transparent_area", True)
        )
        vent["infiltration_exterior_area_mode"] = str(
            zone.get("infiltration_exterior_area_mode", "outdoors_only")
        )
        vent["infiltration_wind_reduction_factor"] = float(
            zone.get("infiltration_wind_reduction_factor", 1.0)
        )
        vent["infiltration_effective_leakage_area_m2"] = float(
            zone.get("infiltration_effective_leakage_area_m2", 0.0)
        )
        vent["infiltration_stack_coefficient"] = float(
            zone.get("infiltration_stack_coefficient", 0.0)
        )
        vent["infiltration_wind_coefficient"] = float(
            zone.get("infiltration_wind_coefficient", 0.0)
        )
        vent["infiltration_schedule_multiplier"] = float(
            zone.get("infiltration_schedule_multiplier", 1.0)
        )
        # Pass through new components list when present in "ventilation" sub-key
        _zone_vent_obj = zone.get("ventilation", {})
        if isinstance(_zone_vent_obj, dict) and "components" in _zone_vent_obj:
            vent["components"] = _zone_vent_obj["components"]

        # Forward zone-level ventilation/heating/cooling profiles when present.
        # The legacy profile generator reads these top-level building_parameters
        # keys; occupancy/appliances/lighting go through internal_gains instead.
        for _pkey in ("ventilation_profile", "heating_profile", "cooling_profile"):
            _zone_prof = zone.get(_pkey)
            if _zone_prof is not None:
                bp[_pkey] = copy.deepcopy(_zone_prof)

        # Forward zone-level internal_gains (occupancy/appliances/lighting
        # schedules) when present.  Direct assignment is required because bp is
        # already deep-copied from global building_parameters, so setdefault()
        # would be a no-op when the global already has the key.
        _zone_gains = zone.get("internal_gains")
        _global_gains = building_object.get("building_parameters", {}).get("internal_gains", [])
        bp["internal_gains"] = copy.deepcopy(
            _zone_gains if _zone_gains is not None else _global_gains
        )

        return bui

    @classmethod
    def compute_internal_partition_coupling_from_zone_temperatures(
        cls,
        building_object: dict,
        zone_temperature_df: pd.DataFrame,
        path_weather_file=None,
        weather_source="epw",
        include_solar=True,
        delta_Theta_er=11.0,
        dt_s=3600.0,
        internal_convection_model=None,
        external_convection_model=None,
        external_convection_h_min=None,
        external_radiation_model=None,
        sky_temperature_model=None,
        external_emissivity_default=None,
    ) -> pd.DataFrame:
        """
        Compute INTERNAL partition heat exchange by clamping zone air temperatures and
        solving envelope nodes. Returns `Q_cpl_<zone>` [W], positive when heat enters zone air.
        """
        sim_df = cls().Weather_data_bui(
            building_object, path_weather_file, weather_source=weather_source
        ).simulation_df
        sim_df = sim_df.copy()
        sim_df.index = pd.DatetimeIndex(sim_df.index)
        sim_df = sim_df.loc[~sim_df.index.duplicated(keep="first")].copy()
        sim_df = sim_df.sort_index()
        T2m_arr = _series_to_float_array(sim_df, "T2m")

        zones = building_object.get("zones", None)
        if not zones:
            zones = [
                {
                    "name": "main",
                    "net_floor_area": float(building_object["building"]["net_floor_area"]),
                }
            ]
        zone_names = [z["name"] for z in zones]
        Z = len(zone_names)
        z_idx = {name: i for i, name in enumerate(zone_names)}
        h_ci_model = _resolve_internal_convection_model(
            building_object,
            internal_convection_model,
        )
        h_ce_model = _resolve_external_convection_model(
            building_object,
            external_convection_model,
        )
        h_ce_min = _resolve_external_convection_h_min(
            building_object,
            external_convection_h_min,
        )
        h_re_model = _resolve_external_radiation_model(
            building_object,
            external_radiation_model,
        )
        sky_t_model = _resolve_sky_temperature_model(
            building_object,
            sky_temperature_model,
        )
        eps_ext_default = _resolve_default_external_emissivity(
            building_object,
            external_emissivity_default,
        )

        # Accept columns as bare zone name or prefixed names
        temp_df = zone_temperature_df.copy()
        temp_df.index = pd.DatetimeIndex(temp_df.index)
        temp_df = temp_df.loc[~temp_df.index.duplicated(keep="first")].copy()
        temp_df = temp_df.sort_index()
        zone_t = pd.DataFrame(index=temp_df.index)
        for name in zone_names:
            if name in temp_df.columns:
                col = name
            elif f"T_air_{name}" in temp_df.columns:
                col = f"T_air_{name}"
            elif f"T_op_core_{name}" in temp_df.columns:
                col = f"T_op_core_{name}"
            elif f"T_op_{name}" in temp_df.columns:
                col = f"T_op_{name}"
            else:
                raise ValueError(f"Missing temperature column for zone '{name}' in zone_temperature_df")
            zone_t[name] = pd.to_numeric(temp_df[col], errors="coerce")

        common_idx = sim_df.index.intersection(zone_t.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping timestamps between weather and zone temperatures")
        sim_df = sim_df.loc[common_idx].copy()
        zone_t = zone_t.loc[common_idx].copy().ffill().bfill()
        zone_t = zone_t.loc[~zone_t.index.duplicated(keep="first")].copy()
        n_steps = min(len(sim_df), len(zone_t))
        sim_df = sim_df.iloc[:n_steps].copy()
        zone_t = zone_t.iloc[:n_steps].copy()

        surfaces = building_object["building_surface"]
        Nsurf = len(surfaces)
        for surf in surfaces:
            bnd = surf.get("boundary", "OUTDOORS").upper()
            if bnd == "GROUND":
                surf["ISO52016_type_string"] = "GR"
            elif bnd == "ADIABATIC":
                surf["ISO52016_type_string"] = "AD"
            elif bnd == "INTERNAL":
                surf["ISO52016_type_string"] = "ADJ"
            else:
                surf["ISO52016_type_string"] = (
                    "W" if surf.get("type", "").lower() == "transparent" else "OP"
                )

        def _orientation_string(surf):
            ori_existing = str(surf.get("ISO52016_orientation_string", "")).upper()
            if ori_existing in {"HOR", "NV", "EV", "SV", "WV"}:
                return ori_existing
            ori = surf.get("orientation", {}) or {}
            az = ori.get("azimuth", None)
            tilt = ori.get("tilt", None)
            try:
                az_f = float(az) % 360.0
                tilt_f = float(tilt)
            except Exception:
                return "SV"
            if abs(tilt_f) < 1e-6:
                return "HOR"
            if abs(tilt_f - 90.0) < 1e-6:
                candidates = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
                labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
                diffs = np.abs(((az_f - candidates + 180.0) % 360.0) - 180.0)
                return str(labels[int(np.argmin(diffs))])
            return "HOR" if tilt_f < 45.0 else "SV"

        for surf in surfaces:
            surf.setdefault("convective_heat_transfer_coefficient_internal", 2.5)
            surf.setdefault("radiative_heat_transfer_coefficient_internal", 5.13)
            if surf.get("boundary", "OUTDOORS").upper() == "INTERNAL":
                surf.setdefault(
                    "convective_heat_transfer_coefficient_external",
                    float(surf.get("convective_heat_transfer_coefficient_internal", 2.5)),
                )
                surf.setdefault(
                    "radiative_heat_transfer_coefficient_external",
                    float(surf.get("radiative_heat_transfer_coefficient_internal", 5.13)),
                )
            else:
                surf.setdefault("convective_heat_transfer_coefficient_external", 20.0)
                surf.setdefault("radiative_heat_transfer_coefficient_external", 4.14)
            surf.setdefault("sky_view_factor", 0.0)
            surf["ISO52016_orientation_string"] = _orientation_string(surf)

        def _get(surf, key, default):
            v = surf.get(key, None)
            return float(v) if v is not None else float(default)

        nodes = cls().Number_of_nodes_element(building_object)
        for i, surf in enumerate(surfaces):
            if surf.get("boundary", "").upper() == "ADIABATIC":
                nodes.Pln[i] = 0

        PlnSum = np.array([0] * Nsurf, dtype=int)
        for i in range(1, Nsurf):
            PlnSum[i] = PlnSum[i - 1] + int(nodes.Pln[i - 1])
        nodes.PlnSum = PlnSum
        nodes.Rn = int(PlnSum[-1] + int(nodes.Pln[-1]) + 1)

        h_pli_eli = cls().Conduttance_node_of_element(building_object).h_pli_eli
        kappa_pli_eli = cls().Areal_heat_capacity_of_element(building_object).kappa_pli_eli
        a_sol_pli_eli = cls().Solar_absorption_of_element(building_object).a_sol_pli_eli

        Ntot = Z + (nodes.Rn - 1)

        def _sys_row_from_surface_ri(ri: int) -> int:
            return Z + (ri - 1)

        def _solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            diag_min = max(1e-6, 1e-9 * np.linalg.norm(A, ord=np.inf))
            d = np.diag(A).copy()
            np.maximum(d, diag_min, out=d)
            np.fill_diagonal(A, d)
            return np.linalg.solve(A, B)

        def _month_at_tstep(tstep: int) -> int:
            try:
                m = int(sim_df.index.month[tstep])
                if 1 <= m <= 12:
                    return m - 1
            except Exception:
                pass
            if "day of year" in sim_df.columns:
                try:
                    doy = int(sim_df["day of year"].iloc[tstep])
                    doy = max(1, min(366, doy))
                    base_year = 2008 if doy == 366 else 2009
                    m = int((pd.Timestamp(base_year, 1, 1) + pd.Timedelta(days=doy - 1)).month)
                    return m - 1
                except Exception:
                    pass
            ts = pd.to_datetime(sim_df.index[tstep], errors="coerce")
            if pd.notna(ts):
                return int(ts.month) - 1
            return 0

        has_ground = any(surf["ISO52016_type_string"] == "GR" for surf in surfaces)
        t_Th = (
            cls().Temp_calculation_of_ground(
                building_object,
                path_weather_file=path_weather_file,
                weather_source=weather_source,
            )
            if has_ground
            else None
        )

        def assemble(theta_prev: np.ndarray, tstep: int, t_air_vec: np.ndarray):
            A = np.zeros((Ntot, Ntot), dtype=float)
            B = np.zeros((Ntot,), dtype=float)
            zone_internal_links = [[] for _ in range(Z)]
            T_out = float(T2m_arr[tstep])
            u_wind = 0.0
            if "WS10m" in sim_df.columns:
                u_wind = float(pd.to_numeric(sim_df["WS10m"], errors="coerce").iloc[tstep])
            if not np.isfinite(u_wind):
                u_wind = 0.0
            T_sky = _sky_temperature_from_weather(sim_df, tstep, sky_t_model)

            # Clamp air nodes to provided zone temperatures
            for zi in range(Z):
                A[zi, zi] = 1.0
                B[zi] = float(t_air_vec[zi])

            for Eli, surf in enumerate(surfaces):
                n_nodes = int(nodes.Pln[Eli])
                if n_nodes == 0:
                    continue

                A_s = float(surf["area"])
                h_ci_tab = _get(surf, "convective_heat_transfer_coefficient_internal", 2.5)
                h_ri = _get(surf, "radiative_heat_transfer_coefficient_internal", 5.13)
                h_ce_tab = _get(surf, "convective_heat_transfer_coefficient_external", 20.0)
                h_re_tab = _get(surf, "radiative_heat_transfer_coefficient_external", 4.14)
                svf = _get(surf, "sky_view_factor", 0.0)

                zone_A = surf.get("zone", zone_names[0])
                if zone_A not in z_idx:
                    continue
                air_A = z_idx[zone_A]
                r_in = _sys_row_from_surface_ri(1 + nodes.PlnSum[Eli] + (n_nodes - 1))
                h_ci = _internal_h_ci_value(
                    surf,
                    h_ci_model,
                    t_air_c=float(t_air_vec[air_A]),
                    t_surf_c=float(theta_prev[r_in]),
                    fallback_h_ci=h_ci_tab,
                )

                bnd = surf.get("boundary", "OUTDOORS").upper()
                is_internal = bnd == "INTERNAL"
                if is_internal:
                    zone_B = surf.get("adjacent_zone", None)
                    if zone_B is None or zone_B not in z_idx:
                        continue
                    air_B = z_idx[zone_B]
                else:
                    air_B = None

                for Pli in range(n_nodes):
                    ri = 1 + nodes.PlnSum[Eli] + Pli
                    R = _sys_row_from_surface_ri(ri)

                    cap = (float(kappa_pli_eli[Pli, Eli]) * A_s) / dt_s
                    A[R, R] += cap
                    B[R] += cap * theta_prev[R]

                    if Pli > 0:
                        h = float(h_pli_eli[Pli - 1, Eli]) * A_s
                        r_prev = _sys_row_from_surface_ri(ri - 1)
                        A[R, R] += h
                        A[R, r_prev] -= h
                    if Pli < n_nodes - 1:
                        h = float(h_pli_eli[Pli, Eli]) * A_s
                        r_next = _sys_row_from_surface_ri(ri + 1)
                        A[R, R] += h
                        A[R, r_next] -= h

                    if Pli == n_nodes - 1:
                        h_tot = (h_ci + h_ri) * A_s
                        A[R, R] += h_tot
                        A[R, air_A] -= h_tot
                        if is_internal:
                            zone_internal_links[air_A].append((R, h_tot))

                    if Pli == 0:
                        if is_internal:
                            h_cb = _get(
                                surf,
                                "convective_heat_transfer_coefficient_external",
                                h_ci_tab,
                            )
                            h_rb = _get(
                                surf,
                                "radiative_heat_transfer_coefficient_external",
                                h_ri,
                            )
                            h_tot = (h_cb + h_rb) * A_s
                            A[R, R] += h_tot
                            A[R, air_B] -= h_tot
                            zone_internal_links[air_B].append((R, h_tot))
                        else:
                            tstr = surf["ISO52016_type_string"]
                            if tstr == "OP" or tstr == "W":
                                h_ce_use = _dynamic_external_convection_h(
                                    surface=surf,
                                    T_surf_C=float(theta_prev[R]),
                                    T_air_C=float(T_out),
                                    u_wind_ms=float(u_wind),
                                    model=h_ce_model,
                                    h_min=h_ce_min,
                                    fallback_h_ce=h_ce_tab,
                                )
                                if h_re_model == "dynamic":
                                    eps_ext = _surface_external_emissivity(
                                        surf,
                                        eps_ext_default,
                                    )
                                    h_re_use, T_ref_re = _dynamic_external_radiative_h_and_ref(
                                        surface=surf,
                                        T_surf_C=float(theta_prev[R]),
                                        T_sky_C=float(T_sky),
                                        T_air_C=float(T_out),
                                        epsilon=eps_ext,
                                    )
                                    h_ext = (h_ce_use + h_re_use) * A_s
                                    A[R, R] += h_ext
                                    B[R] += (h_ce_use * T_out + h_re_use * T_ref_re) * A_s
                                else:
                                    h_ext = (h_ce_use + h_re_tab) * A_s
                                    A[R, R] += h_ext
                                    B[R] += h_ext * T_out
                                    B[R] += -(svf * h_re_tab * float(delta_Theta_er) * A_s)
                                if include_solar and tstr == "OP":
                                    ori = surf.get("ISO52016_orientation_string", "SV")
                                    col = f"I_sol_tot_{ori}"
                                    if col in sim_df.columns:
                                        B[R] += float(a_sol_pli_eli[Pli, Eli]) * A_s * float(sim_df[col].iloc[tstep])
                            elif tstr == "GR" and t_Th is not None:
                                month = _month_at_tstep(tstep)
                                T_gr = float(t_Th.Theta_gr_ve[month])
                                # R_gr_ve is area-specific [m2K/W]; convert it to a
                                # per-surface conductance so total ground coupling
                                # scales with the actual slab area of each zone.
                                h_gr = _ground_conductance_w_per_k(A_s, t_Th)
                                A[R, R] += h_gr
                                B[R] += h_gr * T_gr

            return A, B, zone_internal_links

        Theta = np.full((Ntot,), 20.0, dtype=float)
        for zi, name in enumerate(zone_names):
            Theta[zi] = float(zone_t[name].iloc[0])

        out = pd.DataFrame(index=sim_df.index)
        for t in range(len(sim_df)):
            t_air_vec = np.array([float(zone_t[name].iloc[t]) for name in zone_names], dtype=float)
            A, B, internal_links = assemble(Theta, t, t_air_vec)
            Theta_new = _solve(A, B)

            for zi, name in enumerate(zone_names):
                q_cpl = 0.0
                for (R_surf, h_tot) in internal_links[zi]:
                    q_cpl += float(h_tot) * (float(Theta_new[R_surf]) - float(t_air_vec[zi]))
                out.loc[sim_df.index[t], f"Q_cpl_{name}"] = float(q_cpl)

            Theta = Theta_new

        out = out.loc[~out.index.duplicated(keep="first")].copy()
        out = out.sort_index()
        return out

    @classmethod
    def Temperature_and_Energy_needs_calculation_multizone_hybrid(
        cls,
        building_object,
        path_weather_file=None,
        weather_source="epw",
        include_solar=True,
        max_iterations=6,
        tolerance_w=10.0,
        relaxation=0.6,
        occupants_schedule_workdays=None,
        occupants_schedule_weekend=None,
        appliances_schedule_workdays=None,
        appliances_schedule_weekend=None,
        lighting_schedule_workdays=None,
        lighting_schedule_weekend=None,
        **kwargs,
    ):
        """
        Hybrid iterative multizone:
          1) run `_Temperature_and_Energy_needs_calculation_core` per zone
          2) compute INTERNAL coupling from coupled envelope nodes
          3) feed coupling back as external internal gain series
          4) iterate until convergence.

        Returns:
            hourly_hybrid, annual_hybrid, iterations_df
        """
        zones = building_object.get("zones", None)
        if not zones:
            zones = [
                {
                    "name": "main",
                    "net_floor_area": float(building_object["building"]["net_floor_area"]),
                }
            ]
        zone_names = [z["name"] for z in zones]
        coupling_h_ci_model = kwargs.get("internal_convection_model", None)
        coupling_h_ce_model = kwargs.get("external_convection_model", None)
        coupling_h_ce_min = kwargs.get("external_convection_h_min", None)
        coupling_h_re_model = kwargs.get("external_radiation_model", None)
        coupling_sky_model = kwargs.get("sky_temperature_model", None)
        coupling_eps_ext = kwargs.get("external_emissivity_default", None)

        zone_models = {
            name: cls._build_single_zone_building_object_for_core(building_object, name)
            for name in zone_names
        }

        coupling_map = {name: None for name in zone_names}
        uncoupled_hourly = {}
        iteration_rows = []

        final_zone_hourly = None
        for it in range(1, int(max_iterations) + 1):
            zone_hourly = {}
            for name in zone_names:
                ext_series = coupling_map[name]
                ext_values = None if ext_series is None else np.asarray(ext_series.values, dtype=float)
                hourly, _, _ = cls.Temperature_and_Energy_needs_calculation(
                    zone_models[name],
                    weather_source=weather_source,
                    path_weather_file=path_weather_file,
                    external_internal_gains_series=ext_values,
                    occupants_schedule_workdays=occupants_schedule_workdays,
                    occupants_schedule_weekend=occupants_schedule_weekend,
                    appliances_schedule_workdays=appliances_schedule_workdays,
                    appliances_schedule_weekend=appliances_schedule_weekend,
                    lighting_schedule_workdays=lighting_schedule_workdays,
                    lighting_schedule_weekend=lighting_schedule_weekend,
                    **kwargs,
                )
                hourly = hourly.loc[~hourly.index.duplicated(keep="first")].copy()
                hourly = hourly.sort_index()
                zone_hourly[name] = hourly
                if it == 1:
                    uncoupled_hourly[name] = hourly.copy()

            common_idx = None
            for name in zone_names:
                common_idx = zone_hourly[name].index if common_idx is None else common_idx.intersection(zone_hourly[name].index)
            if common_idx is None or len(common_idx) == 0:
                raise ValueError("No overlapping timestamps across zone core simulations")
            common_idx = common_idx.sort_values()

            zone_temp_df = pd.DataFrame(index=common_idx)
            for name in zone_names:
                zone_temp_df[name] = pd.to_numeric(
                    zone_hourly[name].reindex(common_idx)["T_op"], errors="coerce"
                ).ffill().bfill()

            coupling_now = cls.compute_internal_partition_coupling_from_zone_temperatures(
                building_object=building_object,
                zone_temperature_df=zone_temp_df,
                path_weather_file=path_weather_file,
                weather_source=weather_source,
                include_solar=include_solar,
                internal_convection_model=coupling_h_ci_model,
                external_convection_model=coupling_h_ce_model,
                external_convection_h_min=coupling_h_ce_min,
                external_radiation_model=coupling_h_re_model,
                sky_temperature_model=coupling_sky_model,
                external_emissivity_default=coupling_eps_ext,
            )
            coupling_now = coupling_now.loc[~coupling_now.index.duplicated(keep="first")].copy()
            coupling_now = coupling_now.reindex(common_idx).fillna(0.0)

            max_delta = 0.0
            mean_delta = 0.0
            new_coupling_map = {}
            for name in zone_names:
                col = f"Q_cpl_{name}"
                now = pd.to_numeric(coupling_now[col], errors="coerce").fillna(0.0)
                prev = coupling_map[name]
                if prev is None:
                    prev = pd.Series(0.0, index=common_idx)
                else:
                    prev = pd.to_numeric(prev.reindex(common_idx), errors="coerce").fillna(0.0)

                delta = (now - prev).abs()
                max_delta = max(max_delta, float(delta.max()))
                mean_delta += float(delta.mean())
                new_coupling_map[name] = (1.0 - float(relaxation)) * prev + float(relaxation) * now

            mean_delta = mean_delta / max(len(zone_names), 1)
            iteration_rows.append(
                {
                    "iteration": it,
                    "max_abs_delta_coupling_W": max_delta,
                    "mean_abs_delta_coupling_W": mean_delta,
                }
            )

            coupling_map = new_coupling_map
            final_zone_hourly = zone_hourly
            if max_delta <= float(tolerance_w):
                break

        # Final coherent pass with latest coupling map
        final_zone_hourly = {}
        for name in zone_names:
            ext_values = np.asarray(coupling_map[name].values, dtype=float) if coupling_map[name] is not None else None
            hourly, _, _ = cls.Temperature_and_Energy_needs_calculation(
                zone_models[name],
                weather_source=weather_source,
                path_weather_file=path_weather_file,
                external_internal_gains_series=ext_values,
                occupants_schedule_workdays=occupants_schedule_workdays,
                occupants_schedule_weekend=occupants_schedule_weekend,
                appliances_schedule_workdays=appliances_schedule_workdays,
                appliances_schedule_weekend=appliances_schedule_weekend,
                lighting_schedule_workdays=lighting_schedule_workdays,
                lighting_schedule_weekend=lighting_schedule_weekend,
                **kwargs,
            )
            hourly = hourly.loc[~hourly.index.duplicated(keep="first")].copy()
            hourly = hourly.sort_index()
            final_zone_hourly[name] = hourly

        common_idx = None
        for name in zone_names:
            common_idx = final_zone_hourly[name].index if common_idx is None else common_idx.intersection(final_zone_hourly[name].index)
        if common_idx is None or len(common_idx) == 0:
            raise ValueError("No overlapping timestamps in final hybrid zone runs")
        common_idx = common_idx.sort_values()

        zone_temp_df = pd.DataFrame(index=common_idx)
        for name in zone_names:
            zone_temp_df[name] = pd.to_numeric(
                final_zone_hourly[name].reindex(common_idx)["T_op"], errors="coerce"
            ).ffill().bfill()

        coupling_final = cls.compute_internal_partition_coupling_from_zone_temperatures(
            building_object=building_object,
            zone_temperature_df=zone_temp_df,
            path_weather_file=path_weather_file,
            weather_source=weather_source,
            include_solar=include_solar,
            internal_convection_model=coupling_h_ci_model,
            external_convection_model=coupling_h_ce_model,
            external_convection_h_min=coupling_h_ce_min,
            external_radiation_model=coupling_h_re_model,
            sky_temperature_model=coupling_sky_model,
            external_emissivity_default=coupling_eps_ext,
        )
        coupling_final = coupling_final.loc[~coupling_final.index.duplicated(keep="first")].copy()
        coupling_final = coupling_final.reindex(common_idx).fillna(0.0)

        hourly_hybrid = pd.DataFrame(index=common_idx)
        annual_rows = []
        dt_h = _infer_timestep_hours_from_index(common_idx, default=1.0)
        for zone in zones:
            name = zone["name"]
            h_fin = final_zone_hourly[name].reindex(common_idx)
            h_unc = uncoupled_hourly.get(name, h_fin).reindex(common_idx)
            q_col = f"Q_HC_hybrid_{name}"
            hourly_hybrid[f"T_op_core_{name}"] = pd.to_numeric(h_fin["T_op"], errors="coerce")
            hourly_hybrid[f"Q_HC_uncoupled_{name}"] = pd.to_numeric(h_unc["Q_HC"], errors="coerce")
            hourly_hybrid[f"Q_cpl_{name}"] = pd.to_numeric(coupling_final[f"Q_cpl_{name}"], errors="coerce")
            hourly_hybrid[q_col] = pd.to_numeric(h_fin["Q_HC"], errors="coerce")

            area = float(zone.get("net_floor_area", building_object["building"].get("net_floor_area", 0.0)))
            q = pd.to_numeric(hourly_hybrid[q_col], errors="coerce").fillna(0.0)
            q_unc = pd.to_numeric(hourly_hybrid[f"Q_HC_uncoupled_{name}"], errors="coerce").fillna(0.0)
            q_h = _integrate_power_series_to_energy_wh(q.clip(lower=0.0), default_dt_h=dt_h)
            q_c = _integrate_power_series_to_energy_wh((-q.clip(upper=0.0)), default_dt_h=dt_h)
            q_h_unc = _integrate_power_series_to_energy_wh(q_unc.clip(lower=0.0), default_dt_h=dt_h)
            q_c_unc = _integrate_power_series_to_energy_wh((-q_unc.clip(upper=0.0)), default_dt_h=dt_h)
            annual_rows.append(
                {
                    "zone": name,
                    "Q_H_annual_hybrid": q_h,
                    "Q_C_annual_hybrid": q_c,
                    "Q_H_annual_uncoupled": q_h_unc,
                    "Q_C_annual_uncoupled": q_c_unc,
                    "Delta_Q_H_annual": q_h - q_h_unc,
                    "Delta_Q_C_annual": q_c - q_c_unc,
                    "Q_H_annual_hybrid_per_sqm": (q_h / area) if area > 0 else 0.0,
                    "Q_C_annual_hybrid_per_sqm": (q_c / area) if area > 0 else 0.0,
                    "Q_H_annual_hybrid_kWh": q_h / 1000.0,
                    "Q_C_annual_hybrid_kWh": q_c / 1000.0,
                    "Q_H_annual_uncoupled_kWh": q_h_unc / 1000.0,
                    "Q_C_annual_uncoupled_kWh": q_c_unc / 1000.0,
                    "Delta_Q_H_annual_kWh": (q_h - q_h_unc) / 1000.0,
                    "Delta_Q_C_annual_kWh": (q_c - q_c_unc) / 1000.0,
                    "time_step_h": dt_h,
                }
            )

        annual_hybrid = pd.DataFrame(annual_rows)
        iterations_df = pd.DataFrame(iteration_rows)
        hourly_hybrid = hourly_hybrid.sort_index()
        return hourly_hybrid, annual_hybrid, iterations_df

    @classmethod
    def compare_multizone_methods(
        cls,
        building_object,
        path_weather_file=None,
        weather_source="epw",
        include_solar=True,
        warmup_hours=744,
        hybrid_max_iterations=6,
        hybrid_tolerance_w=10.0,
        hybrid_relaxation=0.6,
        occupants_schedule_workdays=None,
        occupants_schedule_weekend=None,
        appliances_schedule_workdays=None,
        appliances_schedule_weekend=None,
        lighting_schedule_workdays=None,
        lighting_schedule_weekend=None,
        **kwargs,
    ):
        """
        Run and compare:
          - Version 1: `Temperature_and_Energy_needs_calculation_multizone`
          - Version 2: `Temperature_and_Energy_needs_calculation_multizone_hybrid`

        Returns dict with hourly/annual outputs and difference summaries.
        """
        compare_start_s = time.perf_counter()
        building_object_v1 = copy.deepcopy(building_object)
        building_object_v2 = copy.deepcopy(building_object)

        v1_start_s = time.perf_counter()
        v1_hourly, v1_annual = cls.Temperature_and_Energy_needs_calculation_multizone(
            building_object=building_object_v1,
            path_weather_file=path_weather_file,
            weather_source=weather_source,
            include_solar=include_solar,
            warmup_hours=warmup_hours,
            occupants_schedule_workdays=occupants_schedule_workdays,
            occupants_schedule_weekend=occupants_schedule_weekend,
            appliances_schedule_workdays=appliances_schedule_workdays,
            appliances_schedule_weekend=appliances_schedule_weekend,
            lighting_schedule_workdays=lighting_schedule_workdays,
            lighting_schedule_weekend=lighting_schedule_weekend,
            **kwargs,
        )
        v1_elapsed_s = float(time.perf_counter() - v1_start_s)

        v2_start_s = time.perf_counter()
        v2_hourly, v2_annual, v2_iter = cls.Temperature_and_Energy_needs_calculation_multizone_hybrid(
            building_object=building_object_v2,
            path_weather_file=path_weather_file,
            weather_source=weather_source,
            include_solar=include_solar,
            max_iterations=hybrid_max_iterations,
            tolerance_w=hybrid_tolerance_w,
            relaxation=hybrid_relaxation,
            occupants_schedule_workdays=occupants_schedule_workdays,
            occupants_schedule_weekend=occupants_schedule_weekend,
            appliances_schedule_workdays=appliances_schedule_workdays,
            appliances_schedule_weekend=appliances_schedule_weekend,
            lighting_schedule_workdays=lighting_schedule_workdays,
            lighting_schedule_weekend=lighting_schedule_weekend,
            **kwargs,
        )
        v2_elapsed_s = float(time.perf_counter() - v2_start_s)
        post_start_s = time.perf_counter()

        zones = building_object.get("zones", None)
        if not zones:
            zones = [
                {
                    "name": "main",
                    "net_floor_area": float(building_object["building"]["net_floor_area"]),
                }
            ]
        zone_names = [z["name"] for z in zones]

        common_idx = v1_hourly.index.intersection(v2_hourly.index)
        v1 = v1_hourly.reindex(common_idx)
        v2 = v2_hourly.reindex(common_idx)

        hourly_diff = pd.DataFrame(index=common_idx)
        dt_h = _infer_timestep_hours_from_index(common_idx, default=1.0)
        summary_rows = []
        for zone in zones:
            name = zone["name"]
            q1_col = f"Q_HVAC_{name}"
            q2_col = f"Q_HC_hybrid_{name}"
            t1_col = f"T_air_{name}"
            t2_col = f"T_op_core_{name}"
            if q1_col not in v1.columns or q2_col not in v2.columns:
                continue

            q1 = pd.to_numeric(v1[q1_col], errors="coerce").fillna(0.0)
            q2 = pd.to_numeric(v2[q2_col], errors="coerce").fillna(0.0)
            dq = q2 - q1
            hourly_diff[f"dQ_{name}"] = dq

            if t1_col in v1.columns and t2_col in v2.columns:
                t1 = pd.to_numeric(v1[t1_col], errors="coerce").ffill().bfill()
                t2 = pd.to_numeric(v2[t2_col], errors="coerce").ffill().bfill()
                dt = t2 - t1
                hourly_diff[f"dT_{name}"] = dt
                max_abs_dt = float(dt.abs().max())
                mean_abs_dt = float(dt.abs().mean())
            else:
                max_abs_dt = np.nan
                mean_abs_dt = np.nan

            qh1 = _integrate_power_series_to_energy_wh(q1.clip(lower=0.0), default_dt_h=dt_h)
            qc1 = _integrate_power_series_to_energy_wh((-q1.clip(upper=0.0)), default_dt_h=dt_h)
            qh2 = _integrate_power_series_to_energy_wh(q2.clip(lower=0.0), default_dt_h=dt_h)
            qc2 = _integrate_power_series_to_energy_wh((-q2.clip(upper=0.0)), default_dt_h=dt_h)

            summary_rows.append(
                {
                    "zone": name,
                    "Q_H_annual_v1": qh1,
                    "Q_C_annual_v1": qc1,
                    "Q_H_annual_v2": qh2,
                    "Q_C_annual_v2": qc2,
                    "Delta_Q_H_annual_v2_minus_v1": qh2 - qh1,
                    "Delta_Q_C_annual_v2_minus_v1": qc2 - qc1,
                    "Q_H_annual_v1_kWh": qh1 / 1000.0,
                    "Q_C_annual_v1_kWh": qc1 / 1000.0,
                    "Q_H_annual_v2_kWh": qh2 / 1000.0,
                    "Q_C_annual_v2_kWh": qc2 / 1000.0,
                    "Delta_Q_H_annual_v2_minus_v1_kWh": (qh2 - qh1) / 1000.0,
                    "Delta_Q_C_annual_v2_minus_v1_kWh": (qc2 - qc1) / 1000.0,
                    "Mean_abs_hourly_dQ_W": float(dq.abs().mean()),
                    "Max_abs_hourly_dQ_W": float(dq.abs().max()),
                    "Mean_abs_hourly_dT_C": mean_abs_dt,
                    "Max_abs_hourly_dT_C": max_abs_dt,
                    "time_step_h": dt_h,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        post_elapsed_s = float(time.perf_counter() - post_start_s)
        total_elapsed_s = float(time.perf_counter() - compare_start_s)
        return {
            "version1_hourly": v1_hourly,
            "version1_annual": v1_annual,
            "version2_hourly": v2_hourly,
            "version2_annual": v2_annual,
            "hybrid_iterations": v2_iter,
            "hourly_difference": hourly_diff,
            "summary_difference": summary_df,
            "timing_seconds": {
                "fully_integrated_v1": v1_elapsed_s,
                "hybrid_v2": v2_elapsed_s,
                "postprocessing": post_elapsed_s,
                "total": total_elapsed_s,
            },
        }


    @classmethod
    def Temperature_and_Energy_needs_calculation(
        cls,
        building_object,
        nrHCmodes=2,
        c_int_per_A_us=10000,
        f_int_c=0.4,
        f_sol_c=0.1,
        f_H_c=1,
        f_C_c=1,
        delta_Theta_er=11,
        **kwargs,
    ):
        """
        Wrapper that runs the core calculation and surfaces any simulation error.
        """
        try:
            return cls._Temperature_and_Energy_needs_calculation_core(
                building_object,
                nrHCmodes=nrHCmodes,
                c_int_per_A_us=c_int_per_A_us,
                f_int_c=f_int_c,
                f_sol_c=f_sol_c,
                f_H_c=f_H_c,
                f_C_c=f_C_c,
                delta_Theta_er=delta_Theta_er,
                **kwargs,
            )
        except Exception as exc:
            print(f"❌ Simulation error in Temperature_and_Energy_needs_calculation: {exc}")
            raise

    @classmethod
    def _Temperature_and_Energy_needs_calculation_core(
        cls,
        building_object,
        nrHCmodes=2,
        c_int_per_A_us=10000,
        f_int_c=0.4,
        f_sol_c=0.1,
        f_H_c=1,
        f_C_c=1,
        delta_Theta_er=11,
        sankey_graph=False,
        **kwargs,
    ):
        from .generate_profile import HourlyProfileGenerator, get_country_code_from_latlon  # lazy
        """
        Calcualation fo energy needs according to the equation (37) of ISO 52016:2017. Page 60.

        [Matrix A] x [Node temperature vector X] = [State vector B]

        where:
        Theta_int_air: internal air temperature [°C]
        Theta_op_act: Actual operative temperature [°C]

        :param building_object: Building object create according to the method ``Building`` or ``Buildings_from_dictionary``.
        :param nrHCmodes:  inizailization of system mode: 0 for Heating, 1 for Cooling, 2 for Heating and Cooling. Default: 2
        :param k_m_int_a_zt: areal thermal capacity of air and furniture per thermally conditioned zone. Default: 10000 J/m2K
        :param f_int_c: convective fraction of the internal gains into the zone. Default: 0.4
        :param f_sol_c: convective fraction of the solar radiation into the zone. Default: 0.1
        :param f_H_c: convective fraction of the heating system per thermally conditioned zone (if system specific). Deafult: 1
        :param f_C_c: convective fraction of the cooling system per thermally conditioned zone (if system specific). Default: 1
        :param delta_Theta_er: Average difference between external air temperature and sky temperature. Default: 11 fro intermediate zones, 13 Tropics and 9 Sub polar areas

        .. note:: 
            INPUT:
            **sim_df*: dataframe with:

                * index: time of simulation on hourly resolution and timeindex typology (13 months on hourly resolution)
                * T2m: Exteranl temperarture [°C]
                * RH: External humidity [%]
                * G(h):
                * Gb(n):
                * Gd(h):
                * IR(h):
                * WS10m:
                * WD10m:
                * SP:
                * day of year:
                * hour of day:
                * HOR:
                * NV:
                * WV:
                * EV:
                * SV:
                * occupancy_level:
                * comfort_level:
                * Heating:
                * Cooling:
                * air_flow_rate:
                * internal_gains

            * **power_heating_max**: max power of the heating system (provided by the user) in W
            * **power_cooling_max**: max power of the cooling system (provided by the user) in W
            * **Rn**: ... result of function ``Number_of_nodes_element``
            * **Htb**: Heat transmission coefficient for Thermal bridges (provided by the user)
            * **H_ve**: ... result of function ``Ventilation_heat_transfer_coefficient``
            * **Phi_int**: ... result of function ``Internal_heat_gains``
            * **a_use**: building area [m2]
            * **Pln**: ... result of function ``Number_of_nodes_element``
            * **PlnSum**: ... result of function ``Number_of_nodes_element``
            * **a_sol_pli_eli**: ... result of function ``Solar_absorption_of_elment``
            * **kappa_pli_eli**: ... result of function  ``Areal_heat_capacity_of_element``
            * **heat_convective_elements_internal**: internal convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_convective_elements_external**: external convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_radiative_elements_external**: external radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_radiative_elements_internal**: internal radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **sky_factor_elements**: View factor between element and sky
            * **R_gr_ve**: ... result of function ``Temp_calculation_of_ground`` (Thermal Resitance of virtual layer)
            * **Theta_gr_ve**: ... result of function ``Temp_calculation_of_ground``
            * **h_pli_eli**: ... result of function ``Conduttance_node_of_element``

        """
        from .generate_profile import HourlyProfileGenerator, get_country_code_from_latlon  # lazy
        _sched = _make_sched_resolver(kwargs, iso16798_profiles)
        occupants_schedule_workdays = _sched("occupants_schedule_workdays", "occupants_schedule_workdays")
        occupants_schedule_weekend = _sched("occupants_schedule_weekend", "occupants_schedule_weekend")
        appliances_schedule_workdays = _sched("appliances_schedule_workdays", "appliances_schedule_workdays")
        appliances_schedule_weekend = _sched("appliances_schedule_weekend", "appliances_schedule_weekend")
        lighting_schedule_workdays = _sched("lighting_schedule_workdays", "lighting_schedule_workdays")
        lighting_schedule_weekend = _sched("lighting_schedule_weekend", "lighting_schedule_weekend")

        h_ci_model = _resolve_internal_convection_model(
            building_object,
            kwargs.get("internal_convection_model", None),
        )
        h_re_model = _resolve_external_radiation_model(
            building_object,
            kwargs.get("external_radiation_model", None),
        )
        sky_t_model = _resolve_sky_temperature_model(
            building_object,
            kwargs.get("sky_temperature_model", None),
        )
        eps_ext_default = _resolve_default_external_emissivity(
            building_object,
            kwargs.get("external_emissivity_default", None),
        )
        i = 1
        with tqdm(total=13) as pbar:

            pbar.set_postfix({"Info": f"Initialization {i}"})

            # INIZIALIZATION
            path_weather_file_ = kwargs.get("path_weather_file", None)
            if kwargs["weather_source"] == "pvgis":
                path_weather_file_ = None
            sim_df = ISO52016().Weather_data_bui(building_object, path_weather_file_, weather_source=kwargs["weather_source"]).simulation_df
            Tstepn = len(sim_df)  # number of hours to perform the simulation

            # Heating and cooling Load
            Phi_HC_nd_calc = np.zeros(3)  # Load of Heating or Cooling needed to heat/cool the zone - calculated
            Phi_HC_nd_act = np.zeros(Tstepn)  # Load of Heating or Cooling needed to heat/cool the zone - actual

            # Name adjacent zones (optional in many BUI inputs)
            name_adjacent_zones = [surface.get("name_adj_zone") for surface in building_object["building_surface"]]

            # Temperature (indoor and operative)
            '''
            Initialize vector temperature
            Theta_int_air: internal air temperature
            Theta_int_r_mn: mean radiant temperature
                caluclated as:
                Theta_int_r_min = sum(eli=1 to n)(A_eli * teta_pli=oln,eli,t)/sum(eli=1 to n)(A_eli)
                where:
                A_eli: area of element eli
                teta_pli=oln,eli,t: is the temperature at node pli=pln of the building element eli

            Theta_int_op: operative temperature
            '''
            Theta_int_air = np.zeros((Tstepn, 3))
            Theta_int_r_mn = np.zeros((Tstepn, 3))  # <---
            Theta_int_op = np.zeros((Tstepn, 3))
            Theta_op_act = np.zeros(Tstepn)
            pbar.update(1)

            # Time
            Dtime = 3600.0 * np.ones(Tstepn)
            pbar.update(1)

            # Mode
            colB_act = 0  # the vector B has 3 columns (1st column actual value, 2nd: maximum value reachable in heating, 3rd: maximum value reachbale in cooling)
            pbar.update(1)
            
            # # Number of building element
            bui_eln = len(building_object["building_surface"])

            # Element types and orientations
            typology_elements = np.array(bui_eln * ["EXT"], dtype="object")
            for i, surf in enumerate(building_object["building_surface"]):
                if surf["type"] == "opaque":
                    if surf["sky_view_factor"] == 0:
                        typology_elements[i] = "GR"
                    else:
                        typology_elements[i] = "OP"
                elif surf["type"] == "adiabatic":
                    typology_elements[i] = "AD"
                elif surf["type"] == "transparent":
                    typology_elements[i] = "W"
                elif surf["type"] == "adjacent":
                    typology_elements[i] = "ADJ"
                surf["ISO52016_type_string"] = typology_elements[i]

            Type_eli = bui_eln * ["EXT"]
            for i, t in enumerate(typology_elements):
                if t == "GR":
                    Type_eli[i] = "GR"
                elif t == "ADJ":
                    Type_eli[i] = "ADJ"
                elif t == "AD":
                    Type_eli[i] = "AD"
                else:
                    Type_eli[i] = "EXT"
            
            # --- HYDRATION: set the coefficients for the surfaces if missing ---
            if isinstance(building_object, dict):
                # Typical values (ok for default robust)
                hci_facade = 2.5   # convective internal walls
                hci_ground = 0.7   # convective internal towards ground
                hci_roof   = 5.0   # convective internal roofs
                hce_facade = 20.0  # convective external walls
                hce_roof   = 25.0  # convective external roofs
                hce_ground = 4.0   # convective external ground/adiabatic ≈ negligible
                hre_int    = 5.13  # radiative internal
                hre_ext    = 5.13  # radiative external

                for i, surf in enumerate(building_object["building_surface"]):
                    svf = surf.get("sky_view_factor", 1)
                    tstr = surf.get("ISO52016_type_string", "EXT")

                    # --- convective internal ---
                    if tstr == "AD":         # adiabatic: no exchange with the zone
                        hci = 0.0
                    elif svf == 0:           # towards ground
                        hci = hci_ground
                    elif svf == 1:           # roof
                        hci = hci_roof
                    else:                    # facade
                        hci = hci_facade
                    surf.setdefault("convective_heat_transfer_coefficient_internal", hci)

                    # --- radiative internal ---
                    surf.setdefault("radiative_heat_transfer_coefficient_internal", hre_int)

                    # --- convective external ---
                    if tstr == "AD":
                        hce = 0.0            # external side does not exist for AD: we cancel it
                    elif Type_eli[i] == "GR":
                        hce = hce_ground
                    elif svf == 1:
                        hce = hce_roof
                    else:
                        hce = hce_facade
                    surf.setdefault("convective_heat_transfer_coefficient_external", hce)

                    # --- radiative external ---
                    surf.setdefault("radiative_heat_transfer_coefficient_external", hre_ext)

            #
            pbar.update(1)
            if isinstance(building_object, dict):
                g_values = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    if surf["type"] == "transparent":
                        g_values[i] = surf["g_value"]
                g_gl_wi_t = g_values
            else:
                g_gl_wi_t = np.array(building_object.g_factor_windows)
            
            # Building Area of elements
            if isinstance(building_object, dict):
                area_elements = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    area_elements[i] = surf["area"]
            else:
                area_elements = np.array(building_object.area_elements)
            
            area_elements_tot = np.sum(area_elements)  # Sum of all areas
            pbar.update(1)

            # Orientation and tilt
        
            # 1) Assign the orientation string to all surfaces (without aggregating here)
            orientation_elements = np.empty(bui_eln, dtype=object)
            for i, surf in enumerate(building_object["building_surface"]):
                azimuth = float(surf["orientation"]["azimuth"])
                tilt = float(surf["orientation"]["tilt"])

                # Tolerances for robustness
                def is_close(x, target, tol=1e-6):
                    return abs(x - target) <= tol

                if is_close(tilt, 0.0):
                    orientation_elements[i] = "HOR"
                elif is_close(tilt, 90.0):
                    # normalizza azimuth in [0, 360)
                    az = azimuth % 360.0
                    if is_close(az, 0.0) or is_close(az, 360.0):
                        orientation_elements[i] = "NV"
                    elif is_close(az, 90.0):
                        orientation_elements[i] = "EV"
                    elif is_close(az, 180.0):
                        orientation_elements[i] = "SV"
                    elif is_close(az, 270.0):
                        orientation_elements[i] = "WV"
                    else:
                        # fallback: choose the closest cardinal point
                        # (NV=0, EV=90, SV=180, WV=270)
                        candidates = np.array([0.0, 90.0, 180.0, 270.0])
                        labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
                        orientation_elements[i] = labels[np.argmin(np.abs((az - candidates) % 360.0))]
                else:
                    # if tilt is not exactly 0 or 90, decide the logic (here we map for threshold)
                    orientation_elements[i] = "HOR" if tilt < 45.0 else "NV"

                surf["ISO52016_orientation_string"] = orientation_elements[i]

            # 2) Aggregate (once only) and recalculate helpers
            building_object = ISO52016()._aggregate_surfaces_by_direction(building_object)

            # 3) Reconstruct helpers after aggregation
            bui_eln = len(building_object["building_surface"])
            typology_elements = np.array([s["ISO52016_type_string"] for s in building_object["building_surface"]], dtype=object)
            Type_eli = ["EXT" if t not in ("GR", "ADJ", "AD") else t for t in typology_elements]
            orientation_elements = np.array([s["ISO52016_orientation_string"] for s in building_object["building_surface"]], dtype=object)
            heat_convective_elements_internal = np.array(
                    [s["convective_heat_transfer_coefficient_internal"] for s in building_object["building_surface"]],
                    dtype=float
            )
            heat_radiative_elements_internal = np.array(
                [s["radiative_heat_transfer_coefficient_internal"] for s in building_object["building_surface"]],
                dtype=float
            )
            heat_convective_elements_external = np.array(
                [s["convective_heat_transfer_coefficient_external"] for s in building_object["building_surface"]],
                dtype=float
            )
            heat_radiative_elements_external = np.array(
                [s["radiative_heat_transfer_coefficient_external"] for s in building_object["building_surface"]],
                dtype=float
            )

            g_gl_wi_t = np.array(
                [float(s.get("g_value", 0.0)) if s["type"] == "transparent" else 0.0
                for s in building_object["building_surface"]],
                dtype=float
            )

            sky_factor_elements = np.array(
                [float(s.get("sky_view_factor", 0.0)) for s in building_object["building_surface"]],
                dtype=float
            )    
            area_elements = np.array([float(s["area"]) for s in building_object["building_surface"]], dtype=float)
            area_elements_tot = float(np.sum(area_elements))

            # W

            pbar.update(1)

            # External temperature ... (to be checked)
            theta_sup = sim_df["T2m"]
            
            # Thermal capacity of the internal environment of the thermal zone
            C_int = (c_int_per_A_us * building_object["building"]["net_floor_area"])
            pbar.update(1)

            # mean internal radiative transfer coefficient
            if isinstance(building_object, dict):
                radiative_heat_transfer_coefficient = 5.13
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        radiative_heat_transfer_coefficient * np.ones(bui_eln),
                    )/ area_elements_tot
                )
                for surf in building_object["building_surface"]:
                    surf["radiative_heat_transfer_coefficient_internal"] = (
                        radiative_heat_transfer_coefficient
                    )
            else:
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        building_object.heat_radiative_elements_internal,
                    )
                    / area_elements_tot
                )
            pbar.update(1)

            
            # Initialization vectorB and temperature
            nodes = ISO52016().Number_of_nodes_element(building_object)
            Theta_old = 20 * np.ones(nodes.Rn)
            VecB = 20 * np.ones((nodes.Rn, 3))

            surf_has_node = np.array([nodes.Pln[Eli] > 0 for Eli in range(bui_eln)], dtype=bool)
            surf_int_row  = np.full(bui_eln, -1, dtype=int)
            surf_ext_row  = np.full(bui_eln, -1, dtype=int)
            for Eli in range(bui_eln):
                if surf_has_node[Eli]:
                    # Index of the internal face node in the solved vector:
                    # row 0 is air node; surface nodes start at 1.
                    # last node of element Eli = 1 + PlnSum[Eli] + (Pln[Eli]-1)
                    surf_int_row[Eli] = 1 + int(nodes.PlnSum[Eli]) + (int(nodes.Pln[Eli]) - 1)
                    surf_ext_row[Eli] = 1 + int(nodes.PlnSum[Eli])

            # Total area of internal surfaces
            area_int_surfaces_tot = float(area_elements[surf_has_node].sum()) or 1.0

            heat_convective_elements_internal_tab = np.asarray(
                heat_convective_elements_internal,
                dtype=float,
            )
            surfaces_for_hci = (
                building_object["building_surface"]
                if isinstance(building_object, dict)
                else [None] * bui_eln
            )
            heat_radiative_elements_external_tab = np.asarray(
                heat_radiative_elements_external,
                dtype=float,
            )

            def _compute_h_ci_internal_t(theta_air_prev: float, theta_state_prev: np.ndarray) -> np.ndarray:
                h_ci_t = heat_convective_elements_internal_tab.copy()
                if h_ci_model != "tarp" or not isinstance(building_object, dict):
                    return h_ci_t
                for Eli in range(bui_eln):
                    if not surf_has_node[Eli]:
                        continue
                    ri_int = surf_int_row[Eli]
                    if ri_int < 0 or ri_int >= len(theta_state_prev):
                        continue
                    h_ci_t[Eli] = _internal_h_ci_value(
                        surfaces_for_hci[Eli],
                        h_ci_model,
                        t_air_c=float(theta_air_prev),
                        t_surf_c=float(theta_state_prev[ri_int]),
                        fallback_h_ci=float(heat_convective_elements_internal_tab[Eli]),
                    )
                return h_ci_t

            def _compute_h_re_external_t(theta_state_prev: np.ndarray, tstep: int):
                h_re_t = heat_radiative_elements_external_tab.copy()
                t_ref_t = np.full(bui_eln, float(T2m_arr[tstep]), dtype=float)
                if h_re_model != "dynamic" or not isinstance(building_object, dict):
                    return h_re_t, t_ref_t

                T_out_t = float(T2m_arr[tstep])
                T_sky_t = _sky_temperature_from_weather(sim_df, tstep, sky_t_model)
                for Eli in range(bui_eln):
                    if Type_eli[Eli] != "EXT" or not surf_has_node[Eli]:
                        continue
                    r_ext = surf_ext_row[Eli]
                    if r_ext < 0 or r_ext >= len(theta_state_prev):
                        continue
                    surf_obj = building_object["building_surface"][Eli]
                    eps_ext = _surface_external_emissivity(surf_obj, eps_ext_default)
                    h_re_dyn, t_ref_dyn = _dynamic_external_radiative_h_and_ref(
                        surface=surf_obj,
                        T_surf_C=float(theta_state_prev[r_ext]),
                        T_sky_C=float(T_sky_t),
                        T_air_C=float(T_out_t),
                        epsilon=eps_ext,
                    )
                    h_re_t[Eli] = h_re_dyn
                    t_ref_t[Eli] = t_ref_dyn
                return h_re_t, t_ref_t

            pbar.update(1)

            # Temperature ground and thermal bridges
            _tth_kw = {"path_weather_file": path_weather_file_,
                        "weather_source": kwargs["weather_source"]}
            t_Th = ISO52016().Temp_calculation_of_ground(building_object, **_tth_kw)
            #
            pbar.set_postfix({"Info": f"Calculating ground temperature"})
            pbar.update(1)
            h_pli_eli = (ISO52016().Conduttance_node_of_element(building_object).h_pli_eli)

            pbar.set_postfix({"Info": f"Calculating conductance of elements"})
            pbar.update(1)
            kappa_pli_eli = (ISO52016().Areal_heat_capacity_of_element(building_object).kappa_pli_eli)

            pbar.set_postfix({"Info": f"Calculating areal heat capacity of elements"})
            pbar.update(1)
            a_sol_pli_eli = (ISO52016().Solar_absorption_of_element(building_object).a_sol_pli_eli)

            pbar.set_postfix({"Info": f"Calculating solar absorption of element"})
            pbar.update(1)

        # ------------------------------------------------------------------
        # LUMP CAPACITY: add once only the capacity of the AD floors
        # ------------------------------------------------------------------
        if isinstance(building_object, dict):
            for _surf in building_object["building_surface"]:
                if _surf.get("ISO52016_type_string") == "AD":
                    C_int += float(_surf.get("thermal_capacity", 0.0))

        """
        CALCULATION OF SENSIBLE HEATING AND COOLING LOAD (following the procedure of poin 6.5.5.2 of UNI ISO 52016)
        For each hour and each zone the actual internal operative temperature θ and the actual int;ac;op;zt;t 6.5.5.2 Sensible heating and cooling load
        heating or cooling load, ΦHC;ld;ztc;t, is calculated using the following step-wise procedure: 
        """
        H_ve_nat_all = [0]
        S_ve_nat_all = [0.0]
        # Time step for indoor temperature in adjacent zones
        if building_object['building']['adj_zones_present']:
            list_adj_zones = building_object['building']['number_adj_zone']
            if list_adj_zones == 1:
                theta_ztu = np.zeros(Tstepn)
                theta_ztu[0] = 15
            elif list_adj_zones > 1:
                theta_ztu = np.zeros((Tstepn, list_adj_zones))
                theta_ztu[:2] = 15
                
        
        # Generate profiles
        category_profiles = ISO52016().generate_category_profile(
            building_object, 
            occupants_schedule_workdays,
            occupants_schedule_weekend,
            appliances_schedule_workdays,
            appliances_schedule_weekend,
            lighting_schedule_workdays,
            lighting_schedule_weekend,
            )
        try:
            country_calendar = get_country_code_from_latlon(
                building_object["building"]["latitude"],
                building_object["building"]["longitude"],
            )
        except Exception:
            country_calendar = "IT"
        gen = HourlyProfileGenerator(country=country_calendar, num_months=13, category_profiles=category_profiles)
        profile_df = gen.generate()

        def _has_energy(arrlike):
            a = np.asarray(arrlike, dtype=float)
            return np.isfinite(a).all() and a.max() > 0 and a.sum() > 0

        # fallback: se heating/cooling/ventilation profili sono piatti (tutti 0), usa occupancy
        for cat in ("heating","cooling","ventilation"):
            col = f"{cat}_profile"
            if not _has_energy(profile_df[col].values):
                profile_df[col] = profile_df["occupancy_profile"].values

        # keep weather and profiles strictly aligned on the same hourly horizon
        n_common = min(len(sim_df), len(profile_df))
        if n_common <= 0:
            raise ValueError("No common timesteps between weather data and generated profiles.")
        sim_df = sim_df.iloc[:n_common].copy()
        profile_df = profile_df.iloc[:n_common].copy()
        profile_df.index = sim_df.index
        Tstepn = int(n_common)
        external_internal_gains_series = kwargs.get("external_internal_gains_series", None)
        if external_internal_gains_series is None:
            external_internal_gains_series = np.zeros(Tstepn, dtype=float)
        else:
            ext_arr = np.asarray(external_internal_gains_series, dtype=float).reshape(-1)
            if ext_arr.size < Tstepn:
                ext_arr = np.pad(ext_arr, (0, Tstepn - ext_arr.size), mode="constant", constant_values=0.0)
            external_internal_gains_series = ext_arr[:Tstepn]

        # ====================================
        # Get info of porfiles
        # ====================================

        # summury_profile = gen.get_summary()

        # fig = gen.plot_annual_profiles(freq="H", include_weekend_shading=True,
        #                        title="Annual Profiles — Hourly")
        # fig.show()

        # # grafico a medie giornaliere solo per alcune categorie
        # fig_day = gen.plot_annual_profiles(categories=["ventilation","heating","cooling","occupancy"],
        #                                 freq="D", include_weekend_shading=True,
        #                                 title="Annual Profiles — Daily Average")
        # fig_day.show()
                            
        # === ACCUMULATORS FOR SANKEY (Wh) ===
        dt_h = 1.0  # hours per timestep (Dtime is in s)
        # NB: the accumulators will be reset before the start index of the analysis (after warm-up)
        E_solar_Wh = 0.0
        E_internal_Wh = 0.0
        E_heating_Wh = 0.0
        E_cooling_Wh = 0.0
        E_vent_loss_Wh = 0.0
        E_tb_loss_Wh = 0.0
        E_ground_loss_Wh = 0.0
        E_storage_Wh = 0.0

        # ------------------------------------------------------------------
        # STATE CAPACITY (J/K) aligned to the nodes of the equation
        # ------------------------------------------------------------------
        C_state = np.zeros(nodes.Rn, dtype=float)
        C_state[0] = float(C_int)  # node air+furniture (+ AD already lumped)
        for Eli in range(bui_eln):
            n_nodes = nodes.Pln[Eli]
            if n_nodes == 0:
                continue
            for Pli in range(n_nodes):
                ri_state = 1 + nodes.PlnSum[Eli] + Pli
                C_state[ri_state] = float(kappa_pli_eli[Pli, Eli])

        # Previous state for storage (°C): initialized ONCE
        Theta_prev_state = np.full(nodes.Rn, 20.0, dtype=float)

        # --- new structures for TRASMISSIONS per element (only OP and W) ---
        surface_names = [surf["name"] for surf in building_object["building_surface"]]
        surface_types  = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        E_trans_loss_by_surface_Wh = {name: 0.0 for name in surface_names}  # riempiamo solo per OP/W


        win_col_for_index = {}
        for i, s in enumerate(building_object["building_surface"]):
            if s.get("type") == "transparent":
                nm = s.get("name")
                if nm:
                    win_col_for_index[i] = f"W_{nm}"

        # ------------------------------------------------------------------
        # ESCLUDI IL MESE DI WARM-UP DAL SANKEY
        # ------------------------------------------------------------------
        warmup_hours = int(kwargs.get("warmup_hours", 744))
        Tstep_first_act = max(0, min(warmup_hours, Tstepn))
        start_idx = 0
        E_solar_Wh = 0.0
        E_internal_Wh = 0.0
        E_heating_Wh = 0.0
        E_cooling_Wh = 0.0
        E_vent_loss_Wh = 0.0
        E_tb_loss_Wh = 0.0
        E_ground_loss_Wh = 0.0
        E_storage_Wh = 0.0

        # --- Commit-1 pre-loop allocations ---
        vig = VentilationInternalGains(building_object)
        _MatA = np.zeros((nodes.Rn, nodes.Rn), dtype=float)
        _VecB = np.zeros((nodes.Rn, 3), dtype=float)
        # -------------------------------------

        # --- Commit-2: pre-extract time-series as NumPy arrays ---
        T2m_arr       = _series_to_float_array(sim_df, "T2m")
        WS10m_arr     = _series_to_float_array(sim_df, "WS10m", default=0.0)
        heat_prof_arr = _series_to_float_array(profile_df, "heating_profile", default=1.0)
        cool_prof_arr = _series_to_float_array(profile_df, "cooling_profile", default=1.0)
        vent_prof_arr = _series_to_float_array(profile_df, "ventilation_profile", default=1.0)
        occ_prof_arr  = _series_to_float_array(profile_df, "occupancy_profile", default=0.0)
        app_prof_arr  = _series_to_float_array(profile_df, "appliances_profile", default=0.0)
        light_prof_arr= _series_to_float_array(profile_df, "lighting_profile", default=0.0)
        month_arr     = (sim_df.index.month.to_numpy() - 1).astype(int)
        # solar irradiance and shading factor pre-indexed per element
        _Tstepn_c2 = len(sim_df)
        I_sol_dif_el = np.zeros((_Tstepn_c2, bui_eln), dtype=float)
        I_sol_dir_el = np.zeros((_Tstepn_c2, bui_eln), dtype=float)
        I_sol_tot_el = np.zeros((_Tstepn_c2, bui_eln), dtype=float)
        F_sh_el      = np.ones((_Tstepn_c2, bui_eln), dtype=float)
        for _c2_Eli in range(bui_eln):
            _c2_ori  = orientation_elements[_c2_Eli]
            _c2_dif  = f"I_sol_dif_{_c2_ori}"
            _c2_dir  = f"I_sol_dir_w_{_c2_ori}"
            _c2_tot  = f"I_sol_tot_{_c2_ori}"
            if _c2_dif in sim_df.columns:
                I_sol_dif_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_dif], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            if _c2_dir in sim_df.columns:
                I_sol_dir_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_dir], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            if _c2_tot in sim_df.columns:
                I_sol_tot_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_tot], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            _c2_wincol = win_col_for_index.get(_c2_Eli)
            if _c2_wincol and _c2_wincol in sim_df.columns:
                F_sh_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_wincol], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                )
        # -----------------------------------------------------------

        with tqdm(total=Tstepn) as pbar:
            n_w = 0
            for Tstepi in range(start_idx, Tstepn):

                if heat_prof_arr[Tstepi] > 0:
                    Theta_H_set = building_object["building_parameters"]["temperature_setpoints"]["heating_setpoint"]
                else:
                    Theta_H_set = building_object["building_parameters"]["temperature_setpoints"]["heating_setback"]
                if cool_prof_arr[Tstepi] > 0:
                    Theta_C_set = building_object["building_parameters"]["temperature_setpoints"]["cooling_setpoint"]
                else:
                    Theta_C_set = building_object["building_parameters"]["temperature_setpoints"]["cooling_setback"]

                Theta_old = VecB[:, colB_act].copy()

                # firs step:
                # HEATING:
                # if there is no set point for heating (heating system not installed) -> heating power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_H_set < -995:  #
                    power_heating_max_act = 0
                else:
                    # Reasonable caps
                    A_use = building_object["building"]["net_floor_area"]
                    design_P = max(150.0 * A_use, 15_000.0)  # e.g., 150 W/m² or 15 kW minimum
                    warmup_P = 3.0 * design_P

                    if isinstance(building_object, dict):
                        if Tstepi < warmup_hours:  # During warmup, almost unlimited heating power to ensure convergence to setpoint
                            power_heating_max = warmup_P
                        else:
                            power_heating_max = building_object["building_parameters"]["system_capacities"]["heating_capacity"]
                    else:
                        power_heating_max = building_object.power_heating_max
                    power_heating_max_act = power_heating_max

                # COOLING:
                # if there is no set point for heating (cooling system not installed) -> cooling power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_C_set > 995:
                    power_cooling_max_act = 0
                else:
                    if isinstance(building_object, dict):
                        if Tstepi < warmup_hours:  # During warmup, almost unlimited cooling power to ensure convergence to setpoint
                            power_cooling_max = -1e6
                        else:
                            power_cooling_max = -building_object["building_parameters"]["system_capacities"]["cooling_capacity"]
                        power_cooling_max_act = power_cooling_max
                    else:
                        power_cooling_max = building_object.power_cooling_max
                        power_cooling_max_act = power_cooling_max

                # Default HVAC mode columns (0 = free-float, 1 = heating candidate, 2 = cooling candidate).
                # Keep them always defined to avoid accidental references in single-mode branches.
                colB_H = 1
                colB_C = 2
                Phi_HC_nd_calc[0] = 0  # the load has three values:  0 no heating e no cooling, 1  heating, 2 cooling
                if power_heating_max_act == 0 and power_cooling_max_act == 0:  #
                    nrHCmodes = 1
                elif power_cooling_max_act == 0:
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_H] = power_heating_max_act
                elif power_heating_max_act == 0:
                    colB_C = 1
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_C] = power_cooling_max_act
                else:
                    nrHCmodes = 3
                    Phi_HC_nd_calc[colB_H] = power_heating_max_act
                    Phi_HC_nd_calc[colB_C] = power_cooling_max_act

                iterate = True
                _H_ve_nat_tstep = 0.0
                _S_ve_nat_tstep = 0.0
                while iterate:

                    iterate = False

                    _VecB.fill(0.0)
                    VecB = _VecB
                    _MatA.fill(0.0)
                    MatA = _MatA
                    Phi_sol_dir_zt_t = 0 # inizialize solar gain

                    # Solar  heat gain source inside the thermal zone 6.5.13.2
                    for Eli in range(bui_eln):

                        if isinstance(building_object, dict):
                            if (building_object["building_surface"][Eli]["ISO52016_type_string"]== "AD"):
                                continue

                        if Type_eli[Eli] == "EXT" or Type_eli[Eli] == "ADJ":
                            '''
                            Solar gains for each elements, the sim_df['SV' or 'EV', etc.] is calculated based on the
                            UNI 52010:
                            Phi_sol_dir_zt_t: solar gain [W]
                            g_gl_wi_t: g-value of windows
                            sim_df[orientation_elements[Eli]].iloc[Tstepi]: UNI52010
                            '''
                            
                            # case with shading reduction factor
                            Ffr_wi = 0.25 # <- to modify with shading calculation annex F. o.25 is a good approximation
                            F_sh_obst_wi_t = F_sh_el[Tstepi, Eli] if g_gl_wi_t[Eli] != 0 else 1.0

                            Phi_sol_dir_zt_t += (
                                g_gl_wi_t[Eli]
                                * (I_sol_dif_el[Tstepi, Eli] + I_sol_dir_el[Tstepi, Eli] * F_sh_obst_wi_t)
                                * area_elements[Eli] * (1 - Ffr_wi)
                            )
                            
                            '''
                            FRAME AREA FRACTION OF THE WINDOW 
                            ----------------------------------
                            Ffr_wi: frame area fraction of window
                            calculated according to Annex E 
                            Ffr_wi = 1 - (Agl_wi/A_wi)
                            where:
                            Agl_wi: glazing area of window
                            A_wi: total area of window
                            if not provided a value of 0.25 is considered according to the table B21 in the annex B of the ISO

                            SHADING REDUCTION FACTOR DUE TO OBSTACLES FOR DIRECT SOLAR IRRADIATION
                            ----------------------------------------------------------------------
                            # Example balcony or obstacles
                            F_sh_dir_k_t = (h_k_sun_t * w_k_sun_t)/(H_k* W_k)
                            where:
                            h_k_sun_t: horizontal distance from the window to the obstacle
                            w_k_sun_t: vertical distance from the window to the obstacle
                            H_k: height of the facade element k, obtained from the geometry data of the element in [m]. if tilted the vertical projection of the height. For example the height of the window under a balcony
                            W_k: width of the facade element k, obtained from the geometry data of the element in [m]. 
                            '''
                            # Phi_sol_dir_zt_t_tot_new.append(Phi_sol_dir_zt_t)

                    ri = 0
                    '''
                    Energy balacne on zone level. Eq. (38) UNI 52016
                    XTemp = Thermal capacity at specific time (t) and for  a specific degree °C [W] +
                    + Ventilation loss (at time t)[W] + Transmission loss (at time t)[W] + intrnal gain[W] + solar gain [W]. Missed the
                    the convective fraction of the heating/cooling system
                    '''

                    _vent_bdy = _resolve_single_zone_vent_boundary(
                        building_object, float(Theta_old[ri]), Tstepi, sim_df, profile_df,
                    )
                    H_ve_nat = _vent_bdy.heat_transfer_coefficient_w_k
                    S_ve_nat = _vent_bdy.source_term_w
                    # Record for this timestep; do NOT append here — the while
                    # loop may iterate again for HVAC re-solving and we must
                    # emit exactly one entry per timestep to keep diagnostics
                    # aligned.
                    _H_ve_nat_tstep = H_ve_nat
                    _S_ve_nat_tstep = S_ve_nat
                    
                    
                    
                    # ===========================================================================
                    #                       INTERNAL GAINS
                    # ===========================================================================
                    #                       UNCONDITIONED ZONES
                    # ---------------------------------------------------------------------------
                    # Internal gains and solar of unconditioned zone
                    if building_object['building']['adj_zones_present']:
                        # list_adj_zones = list(building_object.adj_zones.keys())
                        list_adj_zones = building_object['building']['number_adj_zone']
                        adj_bui_class = building_object['adjacent_zones'][0]['building_type_class']
                        adj_bui_a_use = building_object['adjacent_zones'][0]['a_use']
                        phi_int_gains_unc_zone = vig.internal_gains(
                            building_type_class = adj_bui_class, 
                            a_use = adj_bui_a_use, 
                            unconditioned_zones_nearby = False,
                            h_occup=occ_prof_arr[Tstepi],
                            h_app=app_prof_arr[Tstepi],
                            h_light=light_prof_arr[Tstepi],
                        )
                        phi_solar_gains_unc_zone = 0 # <----- TO BE MODIFIED ACCORDING TO THE WIDNOW OF THE UNCODITIONED ZONES !!!!!
                        phi_gn_dir_ztu = phi_int_gains_unc_zone + phi_solar_gains_unc_zone
                    
                        ## CASE OF SINGLE UNCONDITIONED ZONE
                        if list_adj_zones == 1:
                            adj_zone = building_object['adjacent_zones'][0]
                            H_ztu, b_ztu, F_ztc_ztu_m =ISO52016().transmission_heat_transfer_coefficient_ISO13789(adj_zone)
                        else: 
                            H_ztu_zones = np.zeros((4, list_adj_zones))
                            name_zones = []
                            for i in range(list_adj_zones):
                                adj_zone = building_object['adjacent_zones'][i]
                                H_ztu, b_ztu, F_ztc_ztu_m =ISO52016().transmission_heat_transfer_coefficient_ISO13789(adj_zone)
                                H_ztu_zones[0, i] = H_ztu
                                H_ztu_zones[1, i] = b_ztu
                                H_ztu_zones[2, i] = F_ztc_ztu_m
                                H_ztu_zones[3, i] = adj_zone['orientation_zone']['azimuth']
                                name_zones.append(adj_zone['name'])
                            H_ztu_zones_df = pd.DataFrame(H_ztu_zones, columns=name_zones, index = ['H_ztu', 'b_ztu', 'F_ztc_ztu_m', 'orientation'])
                            
                    # ---------------------------------------------------------------------------
                    # Internal gains conditioned and unconditioned zones
                    if building_object['building']['adj_zones_present']:
                        int_gains_with_unconditioned_zones = vig.internal_gains(
                                            building_type_class = building_object['building']['building_type_class'], 
                                            a_use=building_object['building']['net_floor_area'], 
                                            unconditioned_zones_nearby = True, 
                                            Fztc_ztu_m=F_ztc_ztu_m,
                                            list_adj_zones=list_adj_zones,
                                            b_ztu=b_ztu,
                                            h_occup=occ_prof_arr[Tstepi],
                                            h_app=app_prof_arr[Tstepi],
                                            h_light=light_prof_arr[Tstepi],
                                            )
                    else:
                        int_gains_conditioned_zone = vig.internal_gains(
                                            building_type_class = building_object['building']['building_type_class'], 
                                            a_use=building_object['building']['net_floor_area'], 
                                            unconditioned_zones_nearby = False,
                                            h_occup=occ_prof_arr[Tstepi],
                                            h_app=app_prof_arr[Tstepi],
                                            h_light=light_prof_arr[Tstepi],
                                            )

                    if building_object['building']['adj_zones_present'] and building_object['building']['number_adj_zone']>=1:
                        int_gains = int_gains_with_unconditioned_zones
                    else:
                        int_gains = int_gains_conditioned_zone

                    ext_int_gain = float(external_internal_gains_series[Tstepi])
                    if not np.isfinite(ext_int_gain):
                        ext_int_gain = 0.0
                    
                    XTemp = (
                        t_Th.thermal_bridge_heat * sim_df.iloc[Tstepi]["T2m"]
                        + S_ve_nat
                        + f_int_c * int_gains
                        + ext_int_gain
                        + f_sol_c * Phi_sol_dir_zt_t
                        + (C_int / Dtime[Tstepi]) * Theta_old[ri]
                    )
                    # X_temp_old.append(int_gains_vent.H_ve)
                    
                    # adding the convective fraction of the heating/cooling system according to the type of system available (heating, cooling and heating and cooling)
                    for cBi in range(nrHCmodes):
                        if Phi_HC_nd_calc[cBi] > 0:
                            f_HC_c = f_H_c
                        else:
                            f_HC_c = f_C_c
                        VecB[ri, cBi] += XTemp + f_HC_c * Phi_HC_nd_calc[cBi]

                    ci = 0

                    '''
                    First part of the equation of energy balance on zone level(38)
                    [C_int/deltaT] +sum(eli=1 to n)(A_eli + h_ci_eli) + sum(vei = 1 to ven)H_ve,_vei_t + Ht_tb_ztc] * theta_int_a_ztc_t -
                    sum(eli=1 to n)(Aeli * h_ci_eli * theta_pln_eli_t) 
                    '''
                    
                    # ==================================================================
                    heat_convective_elements_internal_t = _compute_h_ci_internal_t(
                        theta_air_prev=float(Theta_old[ri]),
                        theta_state_prev=Theta_old,
                    )
                    heat_radiative_elements_external_t, ext_rad_ref_temp_t = _compute_h_re_external_t(
                        theta_state_prev=Theta_old,
                        tstep=Tstepi,
                    )
                    Ah_ci_t = float(
                        (
                            area_elements[surf_has_node]
                            * heat_convective_elements_internal_t[surf_has_node]
                        ).sum()
                    )
                    MatA[ri, ci] += (
                        (C_int / Dtime[Tstepi])
                        + Ah_ci_t
                        + t_Th.thermal_bridge_heat
                        + H_ve_nat
                    )
                    
                    for Eli in range(bui_eln):
                        Pli = nodes.Pln[Eli]
                        if Pli == 0:  # adiabatic element
                            continue
                        ci = nodes.PlnSum[Eli] + Pli 
                        MatA[ri, ci] -= (
                            area_elements[Eli] * heat_convective_elements_internal_t[Eli]
                        )
                    # ==================================================================

                    # ========================================
                    # Temperature of unconditioned space (if any)
                    # ========================================
                    if building_object['building']['adj_zones_present']:
                        c_ztu_h_max = 1 # from table B.16 
                        if Tstepi >0:
                            # Single zones
                            if list_adj_zones == 1:
                                theta_ztu_t = (Theta_int_op[Tstepi-1,0] - b_ztu*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]) + (phi_gn_dir_ztu/H_ztu))
                                theta_ztu_t_checked = min(T2m_arr[Tstepi] + c_ztu_h_max*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]), theta_ztu_t)
                                theta_ztu[Tstepi] = theta_ztu_t_checked
                            
                            # Multiple zones
                            elif list_adj_zones > 1:
                                for z in range(list_adj_zones):
                                    zone = building_object['adjacent_zones'][z]
                                    H_ztu = H_ztu_zones_df.loc['H_ztu'][zone['name']]
                                    theta_ztu_t = (Theta_int_op[Tstepi-1,0] - b_ztu*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]) + (phi_gn_dir_ztu/H_ztu))
                                    theta_ztu_t_checked = min(T2m_arr[Tstepi] + c_ztu_h_max*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]), theta_ztu_t)
                                    theta_ztu[Tstepi,z] = theta_ztu_t_checked
                        theta_ztu_df = pd.DataFrame(theta_ztu, columns=H_ztu_zones_df.columns.tolist())

                    for Eli in range(bui_eln):
                        n_nodes = nodes.Pln[Eli]
                        if n_nodes == 0:  # adiabatic element
                            continue
                        for Pli in range(n_nodes):
                            ri += 1
                            XTemp = (
                                + (kappa_pli_eli[Pli, Eli] / Dtime[Tstepi])
                                * Theta_old[ri]
                            )
                            for cBi in range(nrHCmodes):
                                VecB[ri, cBi] += XTemp
                            
                            if Pli == (n_nodes - 1): 
                                '''
                                Internal surface node 
                                formula (39) from pli=pln (surface node facing calculation zone ztc)
                                '''
                                # XTemp = (1 - f_int_c) * int_gains_vent.Phi_int.iloc[
                                XTemp = (1 - f_int_c) * int_gains + (1 - f_sol_c) * Phi_sol_dir_zt_t
                                for cBi in range(nrHCmodes):
                                    if Phi_HC_nd_calc[cBi] > 0:
                                        f_HC_c = f_H_c
                                    else:
                                        f_HC_c = f_C_c
                                    VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / area_int_surfaces_tot
                                    # VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / area_elements_tot

                            elif Pli == 0:
                                if Type_eli[Eli] == "EXT":
                                    '''
                                    External surface node - formuala (41) 
                                    phi_sky_eli_t:  (extra) thermal radiation to the sky in W/m2 calculated by formula 6.5.13.3

                                    '''
                                    if h_re_model == "dynamic":
                                        XTemp = (
                                            heat_convective_elements_external[Eli] * T2m_arr[Tstepi]
                                            + heat_radiative_elements_external_t[Eli] * ext_rad_ref_temp_t[Eli]
                                            + a_sol_pli_eli[Pli, Eli] * I_sol_tot_el[Tstepi, Eli]
                                        )
                                    else:
                                        phi_sky_eli_t = (
                                            sky_factor_elements[Eli]
                                            * heat_radiative_elements_external_t[Eli]
                                            * delta_Theta_er
                                        )
                                        XTemp = (
                                            (heat_convective_elements_external[Eli] + heat_radiative_elements_external_t[Eli])
                                            * T2m_arr[Tstepi]
                                            - phi_sky_eli_t
                                            + a_sol_pli_eli[Pli, Eli] * I_sol_tot_el[Tstepi, Eli]
                                        )
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp
                                
                                elif Type_eli[Eli] == "ADJ":
                                    '''
                                    Case Opaque wall is adjacent to unconditioned thermal zone
                                    phi_sky_eli_t = 0
                                    a_sol_pli_eli = 0
                                    '''
                                    XTemp = 0.0
                                    if building_object['building']['adj_zones_present']:
                                        list_adj_zones = building_object['building']['number_adj_zone']
                                        if list_adj_zones > 1:
                                            name_adj_zone = name_adjacent_zones[Eli]
                                            XTemp = (
                                                (heat_convective_elements_external[Eli] + heat_radiative_elements_external_t[Eli])
                                                * theta_ztu_df[name_adj_zone].iloc[Tstepi]
                                            )
                                        else:
                                            XTemp = (
                                                (heat_convective_elements_external[Eli] + heat_radiative_elements_external_t[Eli])
                                                * theta_ztu[Tstepi]
                                            )
                                    
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp                                    

                                elif Type_eli[Eli] == "GR":
                                    # Legacy single-zone core is assembled in areal form [W/m2]:
                                    # R_gr_ve is area-specific [m2K/W], so 1 / R_gr_ve is the
                                    # correct areal conductance here. The total-area scaling is
                                    # applied later only in post-processing energy balances.
                                    XTemp = (1 / t_Th.R_gr_ve) * t_Th.Theta_gr_ve[month_arr[Tstepi]]
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp

                            ci = 1 + nodes.PlnSum[Eli] + Pli
                            MatA[ri, ci] += kappa_pli_eli[Pli, Eli] / Dtime[Tstepi]
                            
                            if Pli == (n_nodes - 1):
                                '''
                                in formula (39) Internal surface node - sum/elk=1 to eln)(A_elk/Atot * h_ri_eli  * Teta_pli_elk_t)
                                '''
                      
                                Area_ratio = 0.0
                                for Elk in range(bui_eln):
                                    if nodes.Pln[Elk] == 0:
                                        continue  # no internal node -> no radiative exchange with the zone
                                    Plk = nodes.Pln[Elk] - 1
                                    ck = 1 + nodes.PlnSum[Elk] + Plk
                                    Area_ratio += area_elements[Elk] / area_int_surfaces_tot
                                    MatA[ri, ck] -= (area_elements[Elk] / area_int_surfaces_tot) * heat_radiative_elements_internal[Elk]
                                '''
                                in formula 39  [.. + h_ci_eli  + h_re_eli * sum(elk=1 to eln)(A_elk/Atot) + ..]
                                '''
                                MatA[ri, ci] += (
                                    heat_convective_elements_internal_t[Eli] + 
                                    heat_radiative_elements_internal[Eli] * Area_ratio
                                )
                                MatA[ri, 0] -= heat_convective_elements_internal_t[Eli]


                            elif Pli == 0:
                                
                                if Type_eli[Eli] == "EXT":
                                    MatA[ri, ci] += (
                                        heat_convective_elements_external[Eli]+ 
                                        heat_radiative_elements_external_t[Eli]
                                    )
                                
                                elif Type_eli[Eli] == "ADJ":
                                    MatA[ri, ci] += (
                                        heat_convective_elements_external[Eli]+ 
                                        heat_radiative_elements_external_t[Eli]
                                    )
                                
                                elif Type_eli[Eli] == "GR":
                                    # Areal conductance [W/m2K], consistent with the legacy core.
                                    MatA[ri, ci] += 1 / t_Th.R_gr_ve
                            
                            if Pli > 0:
                                MatA[ri, ci] += h_pli_eli[Pli - 1, Eli] # hpli-1,eli * teta,pli,eli,t
                                MatA[ri, ci - 1] -= h_pli_eli[Pli - 1, Eli] # - hpli-1,eli * teta,pli-1,eli,t
                            
                            if Pli < n_nodes - 1:
                                MatA[ri, ci] += h_pli_eli[Pli, Eli] # hpli,eli * teta,pli,eli,t
                                MatA[ri, ci + 1] -= h_pli_eli[Pli, Eli] # - hpli,eli * teta,pli+1,eli,t
                    
                    '''
                    Temperature calculation of:
                    - internal air
                    - mean radiant temperature
                    - operative temperature

                    '''
                    ######## solve system of equations #######
                    # --- Safe diagonal regularization (avoid read-only error) ---
                    diag_min = max(1e-6, 1e-9 * np.linalg.norm(MatA, ord=np.inf))
                    d = np.diag(MatA).copy()
                    np.maximum(d, diag_min, out=d)
                    np.fill_diagonal(MatA, d)
                    # -------------------------------------------------------------
                    try:
                        theta = np.linalg.solve(MatA, VecB)
                    except np.linalg.LinAlgError:
                        rank = np.linalg.matrix_rank(MatA)
                        print(f"⚠️ MatA solve failed at t={Tstepi}: rank={rank}/{MatA.shape[0]}")
                        print("MatA diagonal:", np.diag(MatA))
                        raise
                    VecB[:, :] = theta
                    
                    # Air Temperature
                    Theta_int_air[Tstepi, :] = VecB[0, :]
                    
                    # --- Mean Radiant Temperature (only internal surfaces) ---
                    Theta_int_r_mn[Tstepi, :] = 0.0
                    A_sum = 0.0
                    for Eli in range(bui_eln):
                        n_nodes_Eli = nodes.Pln[Eli]
                        if n_nodes_Eli == 0:
                            continue  # exclude AD and any surface without node
                        ri_surf = int(surf_int_row[Eli])
                        if ri_surf <= 0 or ri_surf >= VecB.shape[0]:
                            continue
                        Theta_int_r_mn[Tstepi, :] += area_elements[Eli] * VecB[ri_surf, :]
                        A_sum += area_elements[Eli]

                    # uses ONLY the area of internal surfaces; fallback for safety
                    if A_sum == 0.0:
                        A_sum = 1.0
                    Theta_int_r_mn[Tstepi, :] /= A_sum

                    # Operative Temperature
                    Theta_int_op[Tstepi, :] = 0.5 * (Theta_int_air[Tstepi, :] + Theta_int_r_mn[Tstepi, :])
                                        
                    '''
                    STEP 2: ISO 
                    Case heating: Determinates if the heating or the cooling temperature set-point applies and calcualte the heating or cooling load: 
                    Use formaula (27):
                    Phi_HC_ld_zd = Phi_HC_upper*((theta_int_op_set - thet_int_op_0)/(theta_int_op_upper - theta_int_op_0))
                    where:
                    Phi_HC_ld_zd: unrestricted heating or cooling load to reach the required setpoint in W
                    Phi_HC_upper: is the upper value of the heating or cooling load in W  
                    theta_int_op_set: required internal operative setpoint temperature in °C
                    thet_int_op_0: operating temperature in free floating condition in °C
                    theta_int_op_upper: is the internal operational temperature, obtained for the upper heating or cooling load °C
                    '''
                    if nrHCmodes > 1:
                        # HEATING
                        if Theta_int_op[Tstepi, 0] < Theta_H_set:
                            Theta_op_set = Theta_H_set
                            Phi_HC_nd_act[Tstepi] = _safe_load(power_heating_max_act, Theta_op_set,
                                                            Theta_int_op[Tstepi, 0], Theta_int_op[Tstepi, colB_H])
                            if Phi_HC_nd_act[Tstepi] > power_heating_max_act:
                                Phi_HC_nd_act[Tstepi] = power_heating_max_act
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_H]
                                colB_act = colB_H
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True

                        # COOLING
                        elif Theta_int_op[Tstepi, 0] > Theta_C_set:
                            Theta_op_set = Theta_C_set
                            Phi_HC_nd_act[Tstepi] = _safe_load(power_cooling_max_act, Theta_op_set,
                                                            Theta_int_op[Tstepi, 0], Theta_int_op[Tstepi, colB_C])
                            if Phi_HC_nd_act[Tstepi] < power_cooling_max_act:
                                Phi_HC_nd_act[Tstepi] = power_cooling_max_act
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_C]
                                colB_act = colB_C
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True

                        else:
                            Phi_HC_nd_act[Tstepi] = 0.0
                            Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                            colB_act = 0
                    else:
                        Phi_HC_nd_act[Tstepi] = Phi_HC_nd_calc[0]
                        Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                        colB_act = 0

                # Append ventilation diagnostics once per timestep (outside the
                # while loop so HVAC re-solving iterations do not duplicate entries).
                H_ve_nat_all.append(_H_ve_nat_tstep)
                S_ve_nat_all.append(_S_ve_nat_tstep)
                H_ve_nat = _H_ve_nat_tstep
                S_ve_nat = _S_ve_nat_tstep

                # =========================
                # === SANKEY (per timestep)
                # =========================
                dt_h = float(Dtime[Tstepi]) / 3600.0

                # 1) Storage (air + envelope): always update state tracking,
                # but accumulate only after warm-up.
                Theta_curr_state = VecB[:, colB_act]
                dTheta_state = Theta_curr_state - Theta_prev_state
                if Tstepi >= Tstep_first_act:
                    E_storage_Wh += float(np.dot(C_state, dTheta_state)) / 3600.0
                Theta_prev_state = Theta_curr_state

                if Tstepi >= Tstep_first_act:
                    # 2) Direct inputs
                    phi_solar = float(Phi_sol_dir_zt_t)
                    phi_int   = float(int_gains)
                    E_solar_Wh    += phi_solar * dt_h
                    E_internal_Wh += phi_int   * dt_h

                    # 3) Heating/Cooling (uses current load)
                    phi_hc = float(Phi_HC_nd_act[Tstepi])
                    if   phi_hc > 0: E_heating_Wh +=  phi_hc * dt_h
                    elif phi_hc < 0: E_cooling_Wh += (-phi_hc) * dt_h

                    # 4) Ventilation: Q_ve = H_ve * T_in - S_ve (positive = heat leaving zone)
                    T_in  = float(Theta_int_air[Tstepi, 0])
                    T_out = float(sim_df["T2m"].iloc[Tstepi])
                    q_vent = float(H_ve_nat) * T_in - float(S_ve_nat)
                    if q_vent > 0:  E_vent_loss_Wh += q_vent * dt_h
                    else:           E_solar_Wh     += (-q_vent) * dt_h

                    # 5) Thermal bridges
                    q_tb = float(t_Th.thermal_bridge_heat) * (T_in - T_out)
                    if q_tb > 0:  E_tb_loss_Wh += q_tb * dt_h
                    else:         E_solar_Wh   += (-q_tb) * dt_h

                    # 6) Ground
                    T_gr = float(t_Th.Theta_gr_ve[month_arr[Tstepi]])
                    h_ground = _ground_conductance_w_per_k(
                        float(getattr(t_Th, "ground_contact_area", 0.0)),
                        t_Th,
                    )
                    q_ground = h_ground * (T_in - T_gr)
                    if q_ground > 0:  E_ground_loss_Wh += q_ground * dt_h
                    else:             E_solar_Wh       += (-q_ground) * dt_h

                    # 7) Transmission for element (OP, W)
                    T_air = float(Theta_int_air[Tstepi, 0])
                    T_rad = float(Theta_int_r_mn[Tstepi, 0])
                    for Eli in range(bui_eln):
                        if surface_types[Eli] not in ("OP", "W"):  continue
                        n_nodes_Eli = nodes.Pln[Eli]
                        if n_nodes_Eli == 0:                       continue
                        vecb_row_surface = nodes.PlnSum[Eli] + n_nodes_Eli
                        T_surf_int = float(VecB[vecb_row_surface, colB_act])
                        A   = float(area_elements[Eli])
                        hci = float(heat_convective_elements_internal[Eli])
                        hri = float(heat_radiative_elements_internal[Eli])
                        q_cond = A * (hci * (T_air - T_surf_int) + hri * (T_rad - T_surf_int))
                        if   q_cond > 0: E_trans_loss_by_surface_Wh[surface_names[Eli]] += q_cond * dt_h
                        elif q_cond < 0: E_solar_Wh += (-q_cond) * dt_h


                if Tstepi < 6:  # primi 6 passi di debug
                    print(f"[t={Tstepi}] T_op0={Theta_int_op[Tstepi,0]:.2f}  Phi_HC={Phi_HC_nd_act[Tstepi]:.1f}  "
                        f"int_gains={float(int_gains):.1f}  Phi_solar={float(Phi_sol_dir_zt_t):.1f}  "
                        f"H_ve_nat={float(H_ve_nat):.3f}")
                pbar.update(1)
            n_w=n_w+1

        # =========================
        #  Close balance
        # =========================
        # numeric clamp to avoid -0.0 or microscopically negative values
        def _clamp(x: float) -> float:
            return 0.0 if abs(x) < 1e-9 else x

        # total inputs (Wh)
        inputs_Wh = _clamp(E_heating_Wh) + _clamp(E_internal_Wh) + _clamp(E_solar_Wh)

        # total losses (Wh)
        E_transmission_surfaces_Wh = sum(max(0.0, v) for v in E_trans_loss_by_surface_Wh.values())

        # total outputs (Wh)
        outputs_Wh = (
            _clamp(E_cooling_Wh)      # extracted energy
            + _clamp(E_vent_loss_Wh)  # ventilation
            + _clamp(E_tb_loss_Wh)    # thermal bridges
            + _clamp(E_ground_loss_Wh)# ground
            + _clamp(E_transmission_surfaces_Wh)  # transmission OP/W
        )

        # balance residual (Wh)
        E_transmission_residual_Wh = inputs_Wh - outputs_Wh - _clamp(E_storage_Wh)

        # if the residual is small (<1% of the input) I absorb it into the storage to close the balance
        if inputs_Wh > 0 and abs(E_transmission_residual_Wh) < 0.01 * inputs_Wh:
            E_storage_Wh += E_transmission_residual_Wh
            E_transmission_residual_Wh = 0.0

        # =========================
        #  DATA FOR SANKEY
        # =========================
        sankey_inputs = {
            "Heating": _clamp(E_heating_Wh),
            "Internal gains": _clamp(E_internal_Wh),
            "Solar & free-gain": _clamp(E_solar_Wh),
        }

        sankey_outputs = {
            "Cooling (extracted energy)": _clamp(E_cooling_Wh),
            "Ventilation (losses)": _clamp(E_vent_loss_Wh),
            "Thermal bridges": _clamp(E_tb_loss_Wh),
            "Ground": _clamp(E_ground_loss_Wh),
        }

        # add transmission for each element (only positive branches)
        for name, E_Wh in E_trans_loss_by_surface_Wh.items():
            if E_Wh > 0:
                sankey_outputs[f"Transmission - {name}"] = _clamp(E_Wh)

        # display a non-zero residual (pathological case)
        if E_transmission_residual_Wh > 0:
            sankey_outputs["Transmission (residual)"] = _clamp(E_transmission_residual_Wh)

        sankey_data = {
            "inputs": sankey_inputs,
            "outputs": sankey_outputs,
            "energy_accumulated_zone": _clamp(E_storage_Wh),  # can be non-zero on hourly basis, ~0 on annual basis
        }

        # numeric check
        _inputs = inputs_Wh
        _outs_plus_storage = outputs_Wh + _clamp(E_storage_Wh)
        _res = _inputs - _outs_plus_storage
        _rel = _res / max(1.0, _inputs)
        print(f"SANKEY CHECK  inputs={_inputs:.1f}  outputs+storage={_outs_plus_storage:.1f}  residual={_res:.1f} Wh ({100*_rel:.3f}%)")

        # =========================
        #  HOURLY AND ANNUAL RESULTS
        # =========================
        act_slice = slice(Tstep_first_act, Tstepn)
        sim_df_act = sim_df.iloc[Tstep_first_act:Tstepn]
        hourly_results = pd.DataFrame(
            data=np.vstack(
                (
                    Phi_HC_nd_act[act_slice],
                    Theta_op_act[act_slice],
                    sim_df_act["T2m"].to_numpy(),
                )
            ).T,
            index=sim_df_act.index,
            columns=["Q_HC", "T_op", "T_ext"],
        )

        # Zone air and ventilation diagnostics
        _h_ve_arr = np.array(H_ve_nat_all[1:Tstepn + 1], dtype=float)[act_slice]
        _s_ve_arr = np.array(S_ve_nat_all[1:Tstepn + 1], dtype=float)[act_slice]
        _t_air_arr = Theta_int_air[act_slice, 0]
        hourly_results["T_air"] = _t_air_arr
        hourly_results["H_ve"] = _h_ve_arr
        hourly_results["S_ve"] = _s_ve_arr
        _t_eq = np.full_like(_h_ve_arr, np.nan, dtype=float)
        np.divide(
            _s_ve_arr,
            _h_ve_arr,
            out=_t_eq,
            where=_h_ve_arr > 0.0,
        )
        hourly_results["T_ve_source_eq"] = _t_eq
        hourly_results["Q_ve"] = _h_ve_arr * _t_air_arr - _s_ve_arr

        # separate H/C
        hourly_results["Q_H"] = 0.0
        hourly_results.loc[hourly_results["Q_HC"] > 0, "Q_H"] = hourly_results.loc[hourly_results["Q_HC"] > 0, "Q_HC"]

        hourly_results["Q_C"] = 0.0
        hourly_results.loc[hourly_results["Q_HC"] < 0, "Q_C"] = -hourly_results.loc[hourly_results["Q_HC"] < 0, "Q_HC"]

        dt_h_annual = _infer_timestep_hours_from_index(hourly_results.index, default=1.0)
        Q_H_annual = _integrate_power_series_to_energy_wh(hourly_results["Q_H"], default_dt_h=dt_h_annual)
        Q_C_annual = _integrate_power_series_to_energy_wh(hourly_results["Q_C"], default_dt_h=dt_h_annual)
        A_use = float(building_object['building']['net_floor_area'])

        annual_results_dic = {
            "Q_H_annual": Q_H_annual,
            "Q_C_annual": Q_C_annual,
            "Q_H_annual_per_sqm": Q_H_annual / A_use if A_use > 0 else 0.0,
            "Q_C_annual_per_sqm": Q_C_annual / A_use if A_use > 0 else 0.0,
            "Q_H_annual_kWh": Q_H_annual / 1000.0,
            "Q_C_annual_kWh": Q_C_annual / 1000.0,
            "Q_H_annual_kWh_per_sqm": (Q_H_annual / 1000.0 / A_use) if A_use > 0 else 0.0,
            "Q_C_annual_kWh_per_sqm": (Q_C_annual / 1000.0 / A_use) if A_use > 0 else 0.0,
            "time_step_h": dt_h_annual,
        }
        annual_results_df = pd.DataFrame([annual_results_dic])

        # Sankey
        if sankey_graph:
            fig = plot_sankey_building(sankey_data)
            fig.show()

        return hourly_results, annual_results_df, sankey_data

    @classmethod
    def _Temperature_and_Energy_needs_calculation_core_ahu_causal(
        cls,
        building_object,
        nrHCmodes=2,
        c_int_per_A_us=10000,
        f_int_c=0.4,
        f_sol_c=0.1,
        f_H_c=1,
        f_C_c=1,
        delta_Theta_er=11,
        sankey_graph=False,
        **kwargs,
    ):
        from .generate_profile import HourlyProfileGenerator, get_country_code_from_latlon  # lazy
        """
        Calcualation of energy needs according to the equation (37) of ISO 52016:2017. Page 60.

        [Matrix A] x [Node temperature vector X] = [State vector B]

        where:
        Theta_int_air: internal air temperature [°C]
        Theta_op_act: Actual operative temperature [°C]

        :param building_object: Building object create according to the method ``Building`` or ``Buildings_from_dictionary``.
        :param nrHCmodes:  inizailization of system mode: 0 for Heating, 1 for Cooling, 2 for Heating and Cooling. Default: 2
        :param k_m_int_a_zt: areal thermal capacity of air and furniture per thermally conditioned zone. Default: 10000 J/m2K
        :param f_int_c: convective fraction of the internal gains into the zone. Default: 0.4
        :param f_sol_c: convective fraction of the solar radiation into the zone. Default: 0.1
        :param f_H_c: convective fraction of the heating system per thermally conditioned zone (if system specific). Deafult: 1
        :param f_C_c: convective fraction of the cooling system per thermally conditioned zone (if system specific). Default: 1
        :param delta_Theta_er: Average difference between external air temperature and sky temperature. Default: 11 fro intermediate zones, 13 Tropics and 9 Sub polar areas

        .. note:: 
            INPUT:
            **sim_df*: dataframe with:

                * index: time of simulation on hourly resolution and timeindex typology (13 months on hourly resolution)
                * T2m: Exteranl temperarture [°C]
                * RH: External humidity [%]
                * G(h):
                * Gb(n):
                * Gd(h):
                * IR(h):
                * WS10m:
                * WD10m:
                * SP:
                * day of year:
                * hour of day:
                * HOR:
                * NV:
                * WV:
                * EV:
                * SV:
                * occupancy_level:
                * comfort_level:
                * Heating:
                * Cooling:
                * air_flow_rate:
                * internal_gains

            * **power_heating_max**: max power of the heating system (provided by the user) in W
            * **power_cooling_max**: max power of the cooling system (provided by the user) in W
            * **Rn**: ... result of function ``Number_of_nodes_element``
            * **Htb**: Heat transmission coefficient for Thermal bridges (provided by the user)
            * **H_ve**: ... result of function ``Ventilation_heat_transfer_coefficient``
            * **Phi_int**: ... result of function ``Internal_heat_gains``
            * **a_use**: building area [m2]
            * **Pln**: ... result of function ``Number_of_nodes_element``
            * **PlnSum**: ... result of function ``Number_of_nodes_element``
            * **a_sol_pli_eli**: ... result of function ``Solar_absorption_of_elment``
            * **kappa_pli_eli**: ... result of function  ``Areal_heat_capacity_of_element``
            * **heat_convective_elements_internal**: internal convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_convective_elements_external**: external convective heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_radiative_elements_external**: external radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **heat_radiative_elements_internal**: internal radiative  heat transfer coefficient for each element as defined int Table 25 of UNI 52016 - 7.2.2.10
            * **sky_factor_elements**: View factor between element and sky
            * **R_gr_ve**: ... result of function ``Temp_calculation_of_ground`` (Thermal Resitance of virtual layer)
            * **Theta_gr_ve**: ... result of function ``Temp_calculation_of_ground``
            * **h_pli_eli**: ... result of function ``Conduttance_node_of_element``

        """
        from .generate_profile import HourlyProfileGenerator, get_country_code_from_latlon  # lazy
        _sched = _make_sched_resolver(kwargs, iso16798_profiles)
        occupants_schedule_workdays = _sched("occupants_schedule_workdays", "occupants_schedule_workdays")
        occupants_schedule_weekend = _sched("occupants_schedule_weekend", "occupants_schedule_weekend")
        appliances_schedule_workdays = _sched("appliances_schedule_workdays", "appliances_schedule_workdays")
        appliances_schedule_weekend = _sched("appliances_schedule_weekend", "appliances_schedule_weekend")
        lighting_schedule_workdays = _sched("lighting_schedule_workdays", "lighting_schedule_workdays")
        lighting_schedule_weekend = _sched("lighting_schedule_weekend", "lighting_schedule_weekend")

        external_heating_power_fn = kwargs.get("external_heating_power_fn", None)
        if external_heating_power_fn is not None and not callable(external_heating_power_fn):
            raise TypeError("external_heating_power_fn must be callable or None.")
        external_heating_power_causal_fn = kwargs.get("external_heating_power_causal_fn", None)
        if external_heating_power_causal_fn is not None and not callable(external_heating_power_causal_fn):
            raise TypeError("external_heating_power_causal_fn must be callable or None.")
        external_heating_power_series = kwargs.get("external_heating_power_series", None)
        h_ci_model = _resolve_internal_convection_model(
            building_object,
            kwargs.get("internal_convection_model", None),
        )
        h_re_model = _resolve_external_radiation_model(
            building_object,
            kwargs.get("external_radiation_model", None),
        )
        sky_t_model = _resolve_sky_temperature_model(
            building_object,
            kwargs.get("sky_temperature_model", None),
        )
        eps_ext_default = _resolve_default_external_emissivity(
            building_object,
            kwargs.get("external_emissivity_default", None),
        )
        
        i = 1
        with tqdm(total=13) as pbar:

            pbar.set_postfix({"Info": f"Initialization {i}"})

            # INIZIALIZATION
            path_weather_file_ = kwargs.get("path_weather_file", None)
            if kwargs["weather_source"] == "pvgis":
                path_weather_file_ = None

            elif kwargs["weather_source"] == "epw":
                path_weather_file_ = (kwargs["path_weather_file"] if "path_weather_file" in kwargs else None)

            elif kwargs["weather_source"] == "climatedata":
                path_weather_file_ = None


            sim_df = ISO52016().Weather_data_bui(building_object, path_weather_file_, weather_source=kwargs["weather_source"]).simulation_df
            Tstepn = len(sim_df)  # number of hours to perform the simulation

            # Heating and cooling Load
            Phi_HC_nd_calc = np.zeros(3)  # Load of Heating or Cooling needed to heat/cool the zone - calculated
            Phi_HC_nd_act = np.zeros(Tstepn)  # Load of Heating or Cooling needed to heat/cool the zone - actual
            Phi_H_AHU_act = np.zeros(Tstepn)  # External AHU heating power effectively used [W]

            # Name adjacent zones (optional in many BUI inputs)
            name_adjacent_zones = [surface.get("name_adj_zone") for surface in building_object["building_surface"]]

            # Temperature (indoor and operative)
            '''
            Initialize vector temperature
            Theta_int_air: internal air temperature
            Theta_int_r_mn: mean radiant temperature
                caluclated as:
                Theta_int_r_min = sum(eli=1 to n)(A_eli * teta_pli=oln,eli,t)/sum(eli=1 to n)(A_eli)
                where:
                A_eli: area of element eli
                teta_pli=oln,eli,t: is the temperature at node pli=pln of the building element eli

            Theta_int_op: operative temperature
            '''
            Theta_int_air = np.zeros((Tstepn, 3))
            Theta_int_r_mn = np.zeros((Tstepn, 3))  # <---
            Theta_int_op = np.zeros((Tstepn, 3))
            Theta_op_act = np.zeros(Tstepn)
            pbar.update(1)

            # Time
            Dtime = 3600.0 * np.ones(Tstepn)
            pbar.update(1)

            # Mode
            colB_act = 0  # the vector B has 3 columns (1st column actual value, 2nd: maximum value reachable in heating, 3rd: maximum value reachbale in cooling)
            pbar.update(1)
            
            # # Number of building element
            bui_eln = len(building_object["building_surface"])

            # Element types and orientations
            typology_elements = np.array(bui_eln * ["EXT"], dtype="object")
            for i, surf in enumerate(building_object["building_surface"]):
                if surf["type"] == "opaque":
                    if surf["sky_view_factor"] == 0:
                        typology_elements[i] = "GR"
                    else:
                        typology_elements[i] = "OP"
                elif surf["type"] == "adiabatic":
                    typology_elements[i] = "AD"
                elif surf["type"] == "transparent":
                    typology_elements[i] = "W"
                elif surf["type"] == "adjacent":
                    typology_elements[i] = "ADJ"
                surf["ISO52016_type_string"] = typology_elements[i]

            Type_eli = bui_eln * ["EXT"]
            for i, t in enumerate(typology_elements):
                if t == "GR":
                    Type_eli[i] = "GR"
                elif t == "ADJ":
                    Type_eli[i] = "ADJ"
                elif t == "AD":
                    Type_eli[i] = "AD"
                else:
                    Type_eli[i] = "EXT"
            
            # --- HYDRATION: set the coefficients for the surfaces if missing ---
            if isinstance(building_object, dict):
                # Typical values (ok for default robust)
                hci_facade = 2.5   # convective internal walls
                hci_ground = 0.7   # convective internal towards ground
                hci_roof   = 5.0   # convective internal roofs
                hce_facade = 20.0  # convective external walls
                hce_roof   = 25.0  # convective external roofs
                hce_ground = 4.0   # convective external ground/adiabatic ≈ negligible
                hre_int    = 5.13  # radiative internal
                hre_ext    = 5.13  # radiative external

                for i, surf in enumerate(building_object["building_surface"]):
                    svf = surf.get("sky_view_factor", 1)
                    tstr = surf.get("ISO52016_type_string", "EXT")

                    # --- convective internal ---
                    if tstr == "AD":         # adiabatic: no exchange with the zone
                        hci = 0.0
                    elif svf == 0:           # towards ground
                        hci = hci_ground
                    elif svf == 1:           # roof
                        hci = hci_roof
                    else:                    # facade
                        hci = hci_facade
                    surf.setdefault("convective_heat_transfer_coefficient_internal", hci)

                    # --- radiative internal ---
                    surf.setdefault("radiative_heat_transfer_coefficient_internal", hre_int)

                    # --- convective external ---
                    if tstr == "AD":
                        hce = 0.0            # external side does not exist for AD: we cancel it
                    elif Type_eli[i] == "GR":
                        hce = hce_ground
                    elif svf == 1:
                        hce = hce_roof
                    else:
                        hce = hce_facade
                    surf.setdefault("convective_heat_transfer_coefficient_external", hce)

                    # --- radiative external ---
                    surf.setdefault("radiative_heat_transfer_coefficient_external", hre_ext)

            #
            pbar.update(1)
            if isinstance(building_object, dict):
                g_values = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    if surf["type"] == "transparent":
                        g_values[i] = surf["g_value"]
                g_gl_wi_t = g_values
            else:
                g_gl_wi_t = np.array(building_object.g_factor_windows)
            
            # Building Area of elements
            if isinstance(building_object, dict):
                area_elements = np.zeros(bui_eln, dtype=float)
                for i, surf in enumerate(building_object["building_surface"]):
                    area_elements[i] = surf["area"]
            else:
                area_elements = np.array(building_object.area_elements)
            
            area_elements_tot = np.sum(area_elements)  # Sum of all areas
            pbar.update(1)

            # Orientation and tilt
        
            # 1) Assign the orientation string to all surfaces (without aggregating here)
            orientation_elements = np.empty(bui_eln, dtype=object)
            for i, surf in enumerate(building_object["building_surface"]):
                azimuth = float(surf["orientation"]["azimuth"])
                tilt = float(surf["orientation"]["tilt"])

                # Tolerances for robustness
                def is_close(x, target, tol=1e-6):
                    return abs(x - target) <= tol

                if is_close(tilt, 0.0):
                    orientation_elements[i] = "HOR"
                elif is_close(tilt, 90.0):
                    # normalizza azimuth in [0, 360)
                    az = azimuth % 360.0
                    if is_close(az, 0.0) or is_close(az, 360.0):
                        orientation_elements[i] = "NV"
                    elif is_close(az, 90.0):
                        orientation_elements[i] = "EV"
                    elif is_close(az, 180.0):
                        orientation_elements[i] = "SV"
                    elif is_close(az, 270.0):
                        orientation_elements[i] = "WV"
                    else:
                        # fallback: choose the closest cardinal point
                        # (NV=0, EV=90, SV=180, WV=270)
                        candidates = np.array([0.0, 90.0, 180.0, 270.0])
                        labels = np.array(["NV", "EV", "SV", "WV"], dtype=object)
                        orientation_elements[i] = labels[np.argmin(np.abs((az - candidates) % 360.0))]
                else:
                    # if tilt is not exactly 0 or 90, decide the logic (here we map for threshold)
                    orientation_elements[i] = "HOR" if tilt < 45.0 else "NV"

                surf["ISO52016_orientation_string"] = orientation_elements[i]

            # 2) Aggregate (once only) and recalculate helpers
            building_object = ISO52016()._aggregate_surfaces_by_direction(building_object)

            # 3) Reconstruct helpers after aggregation
            bui_eln = len(building_object["building_surface"])
            typology_elements = np.array([s["ISO52016_type_string"] for s in building_object["building_surface"]], dtype=object)
            Type_eli = ["EXT" if t not in ("GR", "ADJ", "AD") else t for t in typology_elements]
            orientation_elements = np.array([s["ISO52016_orientation_string"] for s in building_object["building_surface"]], dtype=object)
            heat_convective_elements_internal = np.array(
                    [s["convective_heat_transfer_coefficient_internal"] for s in building_object["building_surface"]],
                    dtype=float
            )
            heat_radiative_elements_internal = np.array(
                [s["radiative_heat_transfer_coefficient_internal"] for s in building_object["building_surface"]],
                dtype=float
            )
            heat_convective_elements_external = np.array(
                [s["convective_heat_transfer_coefficient_external"] for s in building_object["building_surface"]],
                dtype=float
            )
            heat_radiative_elements_external = np.array(
                [s["radiative_heat_transfer_coefficient_external"] for s in building_object["building_surface"]],
                dtype=float
            )

            g_gl_wi_t = np.array(
                [float(s.get("g_value", 0.0)) if s["type"] == "transparent" else 0.0
                for s in building_object["building_surface"]],
                dtype=float
            )

            sky_factor_elements = np.array(
                [float(s.get("sky_view_factor", 0.0)) for s in building_object["building_surface"]],
                dtype=float
            )    
            area_elements = np.array([float(s["area"]) for s in building_object["building_surface"]], dtype=float)
            area_elements_tot = float(np.sum(area_elements))

            # W

            pbar.update(1)

            # External temperature ... (to be checked)
            theta_sup = sim_df["T2m"]
            
            # Thermal capacity of the internal environment of the thermal zone
            C_int = (c_int_per_A_us * building_object["building"]["net_floor_area"])
            pbar.update(1)

            # mean internal radiative transfer coefficient
            if isinstance(building_object, dict):
                radiative_heat_transfer_coefficient = 5.13
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        radiative_heat_transfer_coefficient * np.ones(bui_eln),
                    )/ area_elements_tot
                )
                for surf in building_object["building_surface"]:
                    surf["radiative_heat_transfer_coefficient_internal"] = (
                        radiative_heat_transfer_coefficient
                    )
            else:
                heat_radiative_elements_internal_mn = (
                    np.dot(
                        area_elements,
                        building_object.heat_radiative_elements_internal,
                    )
                    / area_elements_tot
                )
            pbar.update(1)

            
            # inizialiazation vectorB and temperature
            nodes = ISO52016().Number_of_nodes_element(building_object)
            Theta_old = 20 * np.ones(nodes.Rn)
            VecB = 20 * np.ones((nodes.Rn, 3))

            surf_has_node = np.array([nodes.Pln[Eli] > 0 for Eli in range(bui_eln)], dtype=bool)
            surf_int_row  = np.full(bui_eln, -1, dtype=int)
            surf_ext_row  = np.full(bui_eln, -1, dtype=int)
            for Eli in range(bui_eln):
                if surf_has_node[Eli]:
                    # Index of the internal face node in the solved vector:
                    # row 0 is air node; surface nodes start at 1.
                    # last node of element Eli = 1 + PlnSum[Eli] + (Pln[Eli]-1)
                    surf_int_row[Eli] = 1 + int(nodes.PlnSum[Eli]) + (int(nodes.Pln[Eli]) - 1)
                    surf_ext_row[Eli] = 1 + int(nodes.PlnSum[Eli])

            # Total area of internal surfaces
            area_int_surfaces_tot = float(area_elements[surf_has_node].sum()) or 1.0

            heat_convective_elements_internal_tab = np.asarray(
                heat_convective_elements_internal,
                dtype=float,
            )
            surfaces_for_hci = (
                building_object["building_surface"]
                if isinstance(building_object, dict)
                else [None] * bui_eln
            )
            heat_radiative_elements_external_tab = np.asarray(
                heat_radiative_elements_external,
                dtype=float,
            )

            def _compute_h_ci_internal_t(theta_air_prev: float, theta_state_prev: np.ndarray) -> np.ndarray:
                h_ci_t = heat_convective_elements_internal_tab.copy()
                if h_ci_model != "tarp" or not isinstance(building_object, dict):
                    return h_ci_t
                for Eli in range(bui_eln):
                    if not surf_has_node[Eli]:
                        continue
                    ri_int = surf_int_row[Eli]
                    if ri_int < 0 or ri_int >= len(theta_state_prev):
                        continue
                    h_ci_t[Eli] = _internal_h_ci_value(
                        surfaces_for_hci[Eli],
                        h_ci_model,
                        t_air_c=float(theta_air_prev),
                        t_surf_c=float(theta_state_prev[ri_int]),
                        fallback_h_ci=float(heat_convective_elements_internal_tab[Eli]),
                    )
                return h_ci_t

            def _compute_h_re_external_t(theta_state_prev: np.ndarray, tstep: int):
                h_re_t = heat_radiative_elements_external_tab.copy()
                t_ref_t = np.full(bui_eln, float(T2m_arr[tstep]), dtype=float)
                if h_re_model != "dynamic" or not isinstance(building_object, dict):
                    return h_re_t, t_ref_t

                T_out_t = float(T2m_arr[tstep])
                T_sky_t = _sky_temperature_from_weather(sim_df, tstep, sky_t_model)
                for Eli in range(bui_eln):
                    if Type_eli[Eli] != "EXT" or not surf_has_node[Eli]:
                        continue
                    r_ext = surf_ext_row[Eli]
                    if r_ext < 0 or r_ext >= len(theta_state_prev):
                        continue
                    surf_obj = building_object["building_surface"][Eli]
                    eps_ext = _surface_external_emissivity(surf_obj, eps_ext_default)
                    h_re_dyn, t_ref_dyn = _dynamic_external_radiative_h_and_ref(
                        surface=surf_obj,
                        T_surf_C=float(theta_state_prev[r_ext]),
                        T_sky_C=float(T_sky_t),
                        T_air_C=float(T_out_t),
                        epsilon=eps_ext,
                    )
                    h_re_t[Eli] = h_re_dyn
                    t_ref_t[Eli] = t_ref_dyn
                return h_re_t, t_ref_t

            pbar.update(1)

            # Temperature ground and thermal bridges
            _tth_kw = {"path_weather_file": path_weather_file_,
                        "weather_source": kwargs["weather_source"]}
            t_Th = ISO52016().Temp_calculation_of_ground(building_object, **_tth_kw)
            #
            pbar.set_postfix({"Info": f"Calculating ground temperature"})
            pbar.update(1)
            h_pli_eli = (ISO52016().Conduttance_node_of_element(building_object).h_pli_eli)

            pbar.set_postfix({"Info": f"Calculating conductance of elements"})
            pbar.update(1)
            kappa_pli_eli = (ISO52016().Areal_heat_capacity_of_element(building_object).kappa_pli_eli)

            pbar.set_postfix({"Info": f"Calculating areal heat capacity of elements"})
            pbar.update(1)
            a_sol_pli_eli = (ISO52016().Solar_absorption_of_element(building_object).a_sol_pli_eli)

            pbar.set_postfix({"Info": f"Calculating solar absorption of element"})
            pbar.update(1)

        # ------------------------------------------------------------------
        # LUMP CAPACITY: add once only the capacity of the AD floors
        # ------------------------------------------------------------------
        if isinstance(building_object, dict):
            for _surf in building_object["building_surface"]:
                if _surf.get("ISO52016_type_string") == "AD":
                    C_int += float(_surf.get("thermal_capacity", 0.0))

        """
        CALCULATION OF SENSIBLE HEATING AND COOLING LOAD (following the procedure of poin 6.5.5.2 of UNI ISO 52016)
        For each hour and each zone the actual internal operative temperature θ and the actual int;ac;op;zt;t 6.5.5.2 Sensible heating and cooling load
        heating or cooling load, ΦHC;ld;ztc;t, is calculated using the following step-wise procedure: 
        """
        H_ve_nat_all = [0]
        S_ve_nat_all = [0.0]
        # Time step for indoor temperature in adjacent zones
        if building_object['building']['adj_zones_present']:
            list_adj_zones = building_object['building']['number_adj_zone']
            if list_adj_zones == 1:
                theta_ztu = np.zeros(Tstepn)
                theta_ztu[0] = 15
            elif list_adj_zones > 1:
                theta_ztu = np.zeros((Tstepn, list_adj_zones))
                theta_ztu[:2] = 15
                
        
        # Generate profiles
        category_profiles = ISO52016().generate_category_profile(
            building_object, 
            occupants_schedule_workdays,
            occupants_schedule_weekend,
            appliances_schedule_workdays,
            appliances_schedule_weekend,
            lighting_schedule_workdays,
            lighting_schedule_weekend,
            )
        try:
            country_calendar = get_country_code_from_latlon(
                building_object["building"]["latitude"],
                building_object["building"]["longitude"],
            )
        except Exception:
            country_calendar = "IT"
        gen = HourlyProfileGenerator(country=country_calendar, num_months=13, category_profiles=category_profiles)
        profile_df = gen.generate()

        def _has_energy(arrlike):
            a = np.asarray(arrlike, dtype=float)
            return np.isfinite(a).all() and a.max() > 0 and a.sum() > 0

        # fallback: se heating/cooling/ventilation profili sono piatti (tutti 0), usa occupancy
        for cat in ("heating","cooling","ventilation"):
            col = f"{cat}_profile"
            if not _has_energy(profile_df[col].values):
                profile_df[col] = profile_df["occupancy_profile"].values

        # keep weather and profiles strictly aligned on the same hourly horizon
        n_common = min(len(sim_df), len(profile_df))
        if n_common <= 0:
            raise ValueError("No common timesteps between weather data and generated profiles.")
        sim_df = sim_df.iloc[:n_common].copy()
        profile_df = profile_df.iloc[:n_common].copy()
        profile_df.index = sim_df.index
        Tstepn = int(n_common)
        if external_heating_power_series is not None:
            ext_heat = np.asarray(external_heating_power_series, dtype=float).reshape(-1)
            if ext_heat.size < Tstepn:
                ext_heat = np.pad(ext_heat, (0, Tstepn - ext_heat.size), mode="constant", constant_values=0.0)
            external_heating_power_series = ext_heat[:Tstepn]
        external_internal_gains_series = kwargs.get("external_internal_gains_series", None)
        if external_internal_gains_series is None:
            external_internal_gains_series = np.zeros(Tstepn, dtype=float)
        else:
            ext_arr = np.asarray(external_internal_gains_series, dtype=float).reshape(-1)
            if ext_arr.size < Tstepn:
                ext_arr = np.pad(ext_arr, (0, Tstepn - ext_arr.size), mode="constant", constant_values=0.0)
            external_internal_gains_series = ext_arr[:Tstepn]

        # ====================================
        # Get info of porfiles
        # ====================================

        # summury_profile = gen.get_summary()

        # fig = gen.plot_annual_profiles(freq="H", include_weekend_shading=True,
        #                        title="Annual Profiles — Hourly")
        # fig.show()

        # # grafico a medie giornaliere solo per alcune categorie
        # fig_day = gen.plot_annual_profiles(categories=["ventilation","heating","cooling","occupancy"],
        #                                 freq="D", include_weekend_shading=True,
        #                                 title="Annual Profiles — Daily Average")
        # fig_day.show()
                            
        # === ACCUMULATORS FOR SANKEY (Wh) ===
        dt_h = 1.0  # hours per timestep (Dtime is in s)
        # NB: the accumulators will be reset before the start index of the analysis (after warm-up)
        E_solar_Wh = 0.0
        E_internal_Wh = 0.0
        E_heating_Wh = 0.0
        E_cooling_Wh = 0.0
        E_vent_loss_Wh = 0.0
        E_tb_loss_Wh = 0.0
        E_ground_loss_Wh = 0.0
        E_storage_Wh = 0.0

        # ------------------------------------------------------------------
        # STATE CAPACITY (J/K) aligned to the nodes of the equation
        # ------------------------------------------------------------------
        C_state = np.zeros(nodes.Rn, dtype=float)
        C_state[0] = float(C_int)  # node air+furniture (+ AD already lumped)
        for Eli in range(bui_eln):
            n_nodes = nodes.Pln[Eli]
            if n_nodes == 0:
                continue
            for Pli in range(n_nodes):
                ri_state = 1 + nodes.PlnSum[Eli] + Pli
                C_state[ri_state] = float(kappa_pli_eli[Pli, Eli])

        # Previous state for storage (°C): initialized ONCE
        Theta_prev_state = np.full(nodes.Rn, 20.0, dtype=float)
        # Causal controller state: previous-step operative/air/radiant temperatures.
        theta_air_prev_causal = 20.0
        theta_op_prev_causal = 20.0
        theta_rad_prev_causal = 20.0

        # --- new structures for TRASMISSIONS per element (only OP and W) ---
        surface_names = [surf["name"] for surf in building_object["building_surface"]]
        surface_types  = [surf["ISO52016_type_string"] for surf in building_object["building_surface"]]
        E_trans_loss_by_surface_Wh = {name: 0.0 for name in surface_names}  # riempiamo solo per OP/W


        win_col_for_index = {}
        for i, s in enumerate(building_object["building_surface"]):
            if s.get("type") == "transparent":
                nm = s.get("name")
                if nm:
                    win_col_for_index[i] = f"W_{nm}"

        # ------------------------------------------------------------------
        # ESCLUDI IL MESE DI WARM-UP DAL SANKEY
        # ------------------------------------------------------------------
        warmup_hours = int(kwargs.get("warmup_hours", 744))
        Tstep_first_act = max(0, min(warmup_hours, Tstepn))
        start_idx = 0
        E_solar_Wh = 0.0
        E_internal_Wh = 0.0
        E_heating_Wh = 0.0
        E_cooling_Wh = 0.0
        E_vent_loss_Wh = 0.0
        E_tb_loss_Wh = 0.0
        E_ground_loss_Wh = 0.0
        E_storage_Wh = 0.0

        # --- Commit-1 pre-loop allocations ---
        vig = VentilationInternalGains(building_object)
        _MatA = np.zeros((nodes.Rn, nodes.Rn), dtype=float)
        _VecB = np.zeros((nodes.Rn, 3), dtype=float)
        # -------------------------------------

        # --- Commit-2: pre-extract time-series as NumPy arrays ---
        T2m_arr       = _series_to_float_array(sim_df, "T2m")
        WS10m_arr     = _series_to_float_array(sim_df, "WS10m", default=0.0)
        heat_prof_arr = _series_to_float_array(profile_df, "heating_profile", default=1.0)
        cool_prof_arr = _series_to_float_array(profile_df, "cooling_profile", default=1.0)
        vent_prof_arr = _series_to_float_array(profile_df, "ventilation_profile", default=1.0)
        occ_prof_arr  = _series_to_float_array(profile_df, "occupancy_profile", default=0.0)
        app_prof_arr  = _series_to_float_array(profile_df, "appliances_profile", default=0.0)
        light_prof_arr= _series_to_float_array(profile_df, "lighting_profile", default=0.0)
        month_arr     = (sim_df.index.month.to_numpy() - 1).astype(int)
        # solar irradiance and shading factor pre-indexed per element
        _Tstepn_c2 = len(sim_df)
        I_sol_dif_el = np.zeros((_Tstepn_c2, bui_eln), dtype=float)
        I_sol_dir_el = np.zeros((_Tstepn_c2, bui_eln), dtype=float)
        I_sol_tot_el = np.zeros((_Tstepn_c2, bui_eln), dtype=float)
        F_sh_el      = np.ones((_Tstepn_c2, bui_eln), dtype=float)
        for _c2_Eli in range(bui_eln):
            _c2_ori  = orientation_elements[_c2_Eli]
            _c2_dif  = f"I_sol_dif_{_c2_ori}"
            _c2_dir  = f"I_sol_dir_w_{_c2_ori}"
            _c2_tot  = f"I_sol_tot_{_c2_ori}"
            if _c2_dif in sim_df.columns:
                I_sol_dif_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_dif], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            if _c2_dir in sim_df.columns:
                I_sol_dir_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_dir], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            if _c2_tot in sim_df.columns:
                I_sol_tot_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_tot], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            _c2_wincol = win_col_for_index.get(_c2_Eli)
            if _c2_wincol and _c2_wincol in sim_df.columns:
                F_sh_el[:, _c2_Eli] = (
                    pd.to_numeric(sim_df[_c2_wincol], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                )
        # -----------------------------------------------------------

        with tqdm(total=Tstepn) as pbar:
            n_w = 0
            for Tstepi in range(start_idx, Tstepn):

                if heat_prof_arr[Tstepi] > 0:
                    Theta_H_set = building_object["building_parameters"]["temperature_setpoints"]["heating_setpoint"]
                else:
                    Theta_H_set = building_object["building_parameters"]["temperature_setpoints"]["heating_setback"]
                if cool_prof_arr[Tstepi] > 0:
                    Theta_C_set = building_object["building_parameters"]["temperature_setpoints"]["cooling_setpoint"]
                else:
                    Theta_C_set = building_object["building_parameters"]["temperature_setpoints"]["cooling_setback"]

                Theta_old = VecB[:, colB_act].copy()

                # firs step:
                # HEATING:
                # if there is no set point for heating (heating system not installed) -> heating power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_H_set < -995:  #
                    power_heating_max_act = 0
                else:
                    # Reasonable caps
                    A_use = building_object["building"]["net_floor_area"]
                    design_P = max(150.0 * A_use, 15_000.0)  # e.g., 150 W/m² or 15 kW minimum
                    warmup_P = 3.0 * design_P


                    if isinstance(building_object, dict):
                        if Tstepi < warmup_hours:  # During warmup, almost unlimited heating power to ensure convergence to setpoint
                            power_heating_max = warmup_P
                        else:
                            power_heating_max = building_object["building_parameters"]["system_capacities"]["heating_capacity"]
                    else:
                        power_heating_max = building_object.power_heating_max  
                    power_heating_max_act = power_heating_max
                    # power_heating_max_act = building_object.power_heating_max

                # COOLING:
                # if there is no set point for heating (cooling system not installed) -> cooling power = 0
                # otherwise the actual power is equal to the maximum one
                if Theta_C_set > 995:
                    power_cooling_max_act = 0
                else:
                    if isinstance(building_object, dict):
                        if Tstepi < warmup_hours:  # During warmup, almost unlimited cooling power to ensure convergence to setpoint
                            power_cooling_max = -1e6
                        else:
                            power_cooling_max = -building_object["building_parameters"]["system_capacities"]["cooling_capacity"]
                        power_cooling_max_act = power_cooling_max
                    else:
                        power_cooling_max = building_object.power_cooling_max
                        power_cooling_max_act = power_cooling_max
                    # power_cooling_max_act = building_object.power_cooling_max

                # Default HVAC mode columns (0 = free-float, 1 = heating candidate, 2 = cooling candidate).
                # Keep them always defined to avoid accidental references in single-mode branches.
                colB_H = 1
                colB_C = 2
                ahu_causal_heat_t = None
                if external_heating_power_causal_fn is not None:
                    try:
                        ahu_causal_heat_t = external_heating_power_causal_fn(
                            tstep=Tstepi,
                            theta_air_prev=float(theta_air_prev_causal),
                            theta_op_prev=float(theta_op_prev_causal),
                            theta_rad_prev=float(theta_rad_prev_causal),
                            theta_h_set=float(Theta_H_set),
                            theta_c_set=float(Theta_C_set),
                            weather_row=sim_df.iloc[Tstepi],
                            profile_row=profile_df.iloc[Tstepi],
                            power_heating_max=float(power_heating_max_act),
                        )
                    except TypeError:
                        # Backward-compatible minimal signature:
                        # fn(tstep, theta_air_prev)
                        ahu_causal_heat_t = external_heating_power_causal_fn(
                            Tstepi,
                            float(theta_air_prev_causal),
                        )

                    try:
                        ahu_causal_heat_t = float(ahu_causal_heat_t)
                    except (TypeError, ValueError):
                        ahu_causal_heat_t = 0.0
                    if (not np.isfinite(ahu_causal_heat_t)) or (ahu_causal_heat_t < 0.0):
                        ahu_causal_heat_t = 0.0
                    ahu_causal_heat_t = min(ahu_causal_heat_t, float(power_heating_max_act))

                Phi_HC_nd_calc[0] = 0  # the load has three values:  0 no heating e no cooling, 1  heating, 2 cooling
                if ahu_causal_heat_t is not None and ahu_causal_heat_t > 0.0:
                    # Causal control mode: use the AHU heating power computed from k-1 state
                    # directly for timestep k, then solve temperatures for k.
                    nrHCmodes = 1
                    Phi_HC_nd_calc[0] = ahu_causal_heat_t
                    Phi_H_AHU_act[Tstepi] = ahu_causal_heat_t
                elif power_heating_max_act == 0 and power_cooling_max_act == 0:  #
                    nrHCmodes = 1
                elif power_cooling_max_act == 0:
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_H] = power_heating_max_act
                elif power_heating_max_act == 0:
                    colB_C = 1
                    nrHCmodes = 2
                    Phi_HC_nd_calc[colB_C] = power_cooling_max_act
                else:
                    nrHCmodes = 3
                    Phi_HC_nd_calc[colB_H] = power_heating_max_act
                    Phi_HC_nd_calc[colB_C] = power_cooling_max_act

                iterate = True
                _H_ve_nat_tstep = 0.0
                _S_ve_nat_tstep = 0.0
                while iterate:

                    iterate = False

                    _VecB.fill(0.0)
                    VecB = _VecB
                    _MatA.fill(0.0)
                    MatA = _MatA
                    Phi_sol_dir_zt_t = 0 # inizialize solar gain

                    # Solar  heat gain source inside the thermal zone 6.5.13.2
                    for Eli in range(bui_eln):

                        if isinstance(building_object, dict):
                            if (building_object["building_surface"][Eli]["ISO52016_type_string"]== "AD"):
                                continue

                        if Type_eli[Eli] == "EXT" or Type_eli[Eli] == "ADJ":
                            '''
                            Solar gains for each elements, the sim_df['SV' or 'EV', etc.] is calculated based on the
                            UNI 52010:
                            Phi_sol_dir_zt_t: solar gain [W]
                            g_gl_wi_t: g-value of windows
                            sim_df[orientation_elements[Eli]].iloc[Tstepi]: UNI52010
                            '''
                            
                            # case with shading reduction factor
                            Ffr_wi = 0.25 # <- to modify with shading calculation annex F. o.25 is a good approximation
                            F_sh_obst_wi_t = F_sh_el[Tstepi, Eli] if g_gl_wi_t[Eli] != 0 else 1.0

                            Phi_sol_dir_zt_t += (
                                g_gl_wi_t[Eli]
                                * (I_sol_dif_el[Tstepi, Eli] + I_sol_dir_el[Tstepi, Eli] * F_sh_obst_wi_t)
                                * area_elements[Eli] * (1 - Ffr_wi)
                            )
                            
                            '''
                            FRAME AREA FRACTION OF THE WINDOW 
                            ----------------------------------
                            Ffr_wi: frame area fraction of window
                            calculated according to Annex E 
                            Ffr_wi = 1 - (Agl_wi/A_wi)
                            where:
                            Agl_wi: glazing area of window
                            A_wi: total area of window
                            if not provided a value of 0.25 is considered according to the table B21 in the annex B of the ISO

                            SHADING REDUCTION FACTOR DUE TO OBSTACLES FOR DIRECT SOLAR IRRADIATION
                            ----------------------------------------------------------------------
                            # Example balcony or obstacles
                            F_sh_dir_k_t = (h_k_sun_t * w_k_sun_t)/(H_k* W_k)
                            where:
                            h_k_sun_t: horizontal distance from the window to the obstacle
                            w_k_sun_t: vertical distance from the window to the obstacle
                            H_k: height of the facade element k, obtained from the geometry data of the element in [m]. if tilted the vertical projection of the height. For example the height of the window under a balcony
                            W_k: width of the facade element k, obtained from the geometry data of the element in [m]. 
                            '''
                            # Phi_sol_dir_zt_t_tot_new.append(Phi_sol_dir_zt_t)

                    ri = 0
                    '''
                    Energy balacne on zone level. Eq. (38) UNI 52016
                    XTemp = Thermal capacity at specific time (t) and for  a specific degree °C [W] +
                    + Ventilation loss (at time t)[W] + Transmission loss (at time t)[W] + intrnal gain[W] + solar gain [W]. Missed the
                    the convective fraction of the heating/cooling system
                    '''

                    _vent_bdy = _resolve_single_zone_vent_boundary(
                        building_object, float(Theta_old[ri]), Tstepi, sim_df, profile_df,
                    )
                    H_ve_nat = _vent_bdy.heat_transfer_coefficient_w_k
                    S_ve_nat = _vent_bdy.source_term_w
                    # Record for this timestep; do NOT append here — the while
                    # loop may iterate again for HVAC re-solving and we must
                    # emit exactly one entry per timestep to keep diagnostics
                    # aligned.
                    _H_ve_nat_tstep = H_ve_nat
                    _S_ve_nat_tstep = S_ve_nat
                    
                    
                    
                    # ===========================================================================
                    #                       INTERNAL GAINS
                    # ===========================================================================
                    #                       UNCONDITIONED ZONES
                    # ---------------------------------------------------------------------------
                    # Internal gains and solar of unconditioned zone
                    if building_object['building']['adj_zones_present']:
                        # list_adj_zones = list(building_object.adj_zones.keys())
                        list_adj_zones = building_object['building']['number_adj_zone']
                        adj_bui_class = building_object['adjacent_zones'][0]['building_type_class']
                        adj_bui_a_use = building_object['adjacent_zones'][0]['a_use']
                        phi_int_gains_unc_zone = vig.internal_gains(
                            building_type_class = adj_bui_class, 
                            a_use = adj_bui_a_use, 
                            unconditioned_zones_nearby = False,
                            h_occup=occ_prof_arr[Tstepi],
                            h_app=app_prof_arr[Tstepi],
                            h_light=light_prof_arr[Tstepi],
                        )
                        phi_solar_gains_unc_zone = 0 # <----- TO BE MODIFIED ACCORDING TO THE WIDNOW OF THE UNCODITIONED ZONES !!!!!
                        phi_gn_dir_ztu = phi_int_gains_unc_zone + phi_solar_gains_unc_zone
                    
                        ## CASE OF SINGLE UNCONDITIONED ZONE
                        if list_adj_zones == 1:
                            adj_zone = building_object['adjacent_zones'][0]
                            H_ztu, b_ztu, F_ztc_ztu_m =ISO52016().transmission_heat_transfer_coefficient_ISO13789(adj_zone)
                        else: 
                            H_ztu_zones = np.zeros((4, list_adj_zones))
                            name_zones = []
                            for i in range(list_adj_zones):
                                adj_zone = building_object['adjacent_zones'][i]
                                H_ztu, b_ztu, F_ztc_ztu_m =ISO52016().transmission_heat_transfer_coefficient_ISO13789(adj_zone)
                                H_ztu_zones[0, i] = H_ztu
                                H_ztu_zones[1, i] = b_ztu
                                H_ztu_zones[2, i] = F_ztc_ztu_m
                                H_ztu_zones[3, i] = adj_zone['orientation_zone']['azimuth']
                                name_zones.append(adj_zone['name'])
                            H_ztu_zones_df = pd.DataFrame(H_ztu_zones, columns=name_zones, index = ['H_ztu', 'b_ztu', 'F_ztc_ztu_m', 'orientation'])
                            
                    # ---------------------------------------------------------------------------
                    # Internal gains conditioned and unconditioned zones
                    if building_object['building']['adj_zones_present']:
                        int_gains_with_unconditioned_zones = vig.internal_gains(
                                            building_type_class = building_object['building']['building_type_class'], 
                                            a_use=building_object['building']['net_floor_area'], 
                                            unconditioned_zones_nearby = True, 
                                            Fztc_ztu_m=F_ztc_ztu_m,
                                            list_adj_zones=list_adj_zones,
                                            b_ztu=b_ztu,
                                            h_occup=occ_prof_arr[Tstepi],
                                            h_app=app_prof_arr[Tstepi],
                                            h_light=light_prof_arr[Tstepi],
                                            )
                    else:
                        int_gains_conditioned_zone = vig.internal_gains(
                                            building_type_class = building_object['building']['building_type_class'], 
                                            a_use=building_object['building']['net_floor_area'], 
                                            unconditioned_zones_nearby = False,
                                            h_occup=occ_prof_arr[Tstepi],
                                            h_app=app_prof_arr[Tstepi],
                                            h_light=light_prof_arr[Tstepi],
                                            )

                    if building_object['building']['adj_zones_present'] and building_object['building']['number_adj_zone']>=1:
                        int_gains = int_gains_with_unconditioned_zones
                    else:
                        int_gains = int_gains_conditioned_zone

                    ext_int_gain = float(external_internal_gains_series[Tstepi])
                    if not np.isfinite(ext_int_gain):
                        ext_int_gain = 0.0
                    
                    XTemp = (
                        t_Th.thermal_bridge_heat * sim_df.iloc[Tstepi]["T2m"]
                        + S_ve_nat
                        + f_int_c * int_gains
                        + ext_int_gain
                        + f_sol_c * Phi_sol_dir_zt_t
                        + (C_int / Dtime[Tstepi]) * Theta_old[ri]
                    )
                    # X_temp_old.append(int_gains_vent.H_ve)
                    
                    # adding the convective fraction of the heating/cooling system according to the type of system available (heating, cooling and heating and cooling)
                    for cBi in range(nrHCmodes):
                        if Phi_HC_nd_calc[cBi] > 0:
                            f_HC_c = f_H_c
                        else:
                            f_HC_c = f_C_c
                        VecB[ri, cBi] += XTemp + f_HC_c * Phi_HC_nd_calc[cBi]

                    ci = 0

                    '''
                    First part of the equation of energy balance on zone level(38)
                    [C_int/deltaT] +sum(eli=1 to n)(A_eli + h_ci_eli) + sum(vei = 1 to ven)H_ve,_vei_t + Ht_tb_ztc] * theta_int_a_ztc_t -
                    sum(eli=1 to n)(Aeli * h_ci_eli * theta_pln_eli_t) 
                    '''
                    
                    # ==================================================================
                    heat_convective_elements_internal_t = _compute_h_ci_internal_t(
                        theta_air_prev=float(Theta_old[ri]),
                        theta_state_prev=Theta_old,
                    )
                    heat_radiative_elements_external_t, ext_rad_ref_temp_t = _compute_h_re_external_t(
                        theta_state_prev=Theta_old,
                        tstep=Tstepi,
                    )
                    Ah_ci_t = float(
                        (
                            area_elements[surf_has_node]
                            * heat_convective_elements_internal_t[surf_has_node]
                        ).sum()
                    )
                    MatA[ri, ci] += (
                        (C_int / Dtime[Tstepi])
                        + Ah_ci_t
                        + t_Th.thermal_bridge_heat
                        + H_ve_nat
                    )
                    
                    for Eli in range(bui_eln):
                        Pli = nodes.Pln[Eli]
                        if Pli == 0:  # adiabatic element
                            continue
                        ci = nodes.PlnSum[Eli] + Pli 
                        MatA[ri, ci] -= (
                            area_elements[Eli] * heat_convective_elements_internal_t[Eli]
                        )
                    # ==================================================================

                    # ========================================
                    # Temperature of unconditioned space (if any)
                    # ========================================
                    if building_object['building']['adj_zones_present']:
                        c_ztu_h_max = 1 # from table B.16 
                        if Tstepi >0:
                            # Single zones
                            if list_adj_zones == 1:
                                theta_ztu_t = (Theta_int_op[Tstepi-1,0] - b_ztu*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]) + (phi_gn_dir_ztu/H_ztu))
                                theta_ztu_t_checked = min(T2m_arr[Tstepi] + c_ztu_h_max*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]), theta_ztu_t)
                                theta_ztu[Tstepi] = theta_ztu_t_checked
                            
                            # Multiple zones
                            elif list_adj_zones > 1:
                                for z in range(list_adj_zones):
                                    zone = building_object['adjacent_zones'][z]
                                    H_ztu = H_ztu_zones_df.loc['H_ztu'][zone['name']]
                                    theta_ztu_t = (Theta_int_op[Tstepi-1,0] - b_ztu*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]) + (phi_gn_dir_ztu/H_ztu))
                                    theta_ztu_t_checked = min(T2m_arr[Tstepi] + c_ztu_h_max*(Theta_int_op[Tstepi-1,0] - T2m_arr[Tstepi]), theta_ztu_t)
                                    theta_ztu[Tstepi,z] = theta_ztu_t_checked
                        theta_ztu_df = pd.DataFrame(theta_ztu, columns=H_ztu_zones_df.columns.tolist())

                    for Eli in range(bui_eln):
                        n_nodes = nodes.Pln[Eli]
                        if n_nodes == 0:  # adiabatic element
                            continue
                        for Pli in range(n_nodes):
                            ri += 1
                            XTemp = (
                                + (kappa_pli_eli[Pli, Eli] / Dtime[Tstepi])
                                * Theta_old[ri]
                            )
                            for cBi in range(nrHCmodes):
                                VecB[ri, cBi] += XTemp
                            
                            if Pli == (n_nodes - 1): 
                                '''
                                Internal surface node 
                                formula (39) from pli=pln (surface node facing calculation zone ztc)
                                '''
                                # XTemp = (1 - f_int_c) * int_gains_vent.Phi_int.iloc[
                                XTemp = (1 - f_int_c) * int_gains + (1 - f_sol_c) * Phi_sol_dir_zt_t
                                for cBi in range(nrHCmodes):
                                    if Phi_HC_nd_calc[cBi] > 0:
                                        f_HC_c = f_H_c
                                    else:
                                        f_HC_c = f_C_c
                                    VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / area_int_surfaces_tot
                                    # VecB[ri, cBi] += (XTemp + (1 - f_HC_c) * Phi_HC_nd_calc[cBi]) / area_elements_tot

                            elif Pli == 0:
                                if Type_eli[Eli] == "EXT":
                                    '''
                                    External surface node - formuala (41) 
                                    phi_sky_eli_t:  (extra) thermal radiation to the sky in W/m2 calculated by formula 6.5.13.3

                                    '''
                                    if h_re_model == "dynamic":
                                        XTemp = (
                                            heat_convective_elements_external[Eli] * T2m_arr[Tstepi]
                                            + heat_radiative_elements_external_t[Eli] * ext_rad_ref_temp_t[Eli]
                                            + a_sol_pli_eli[Pli, Eli] * I_sol_tot_el[Tstepi, Eli]
                                        )
                                    else:
                                        phi_sky_eli_t = (
                                            sky_factor_elements[Eli]
                                            * heat_radiative_elements_external_t[Eli]
                                            * delta_Theta_er
                                        )
                                        XTemp = (
                                            (heat_convective_elements_external[Eli] + heat_radiative_elements_external_t[Eli])
                                            * T2m_arr[Tstepi]
                                            - phi_sky_eli_t
                                            + a_sol_pli_eli[Pli, Eli] * I_sol_tot_el[Tstepi, Eli]
                                        )
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp
                                
                                elif Type_eli[Eli] == "ADJ":
                                    '''
                                    Case Opaque wall is adjacent to unconditioned thermal zone
                                    phi_sky_eli_t = 0
                                    a_sol_pli_eli = 0
                                    '''
                                    XTemp = 0.0
                                    if building_object['building']['adj_zones_present']:
                                        list_adj_zones = building_object['building']['number_adj_zone']
                                        if list_adj_zones > 1:
                                            name_adj_zone = name_adjacent_zones[Eli]
                                            XTemp = (
                                                (heat_convective_elements_external[Eli] + heat_radiative_elements_external_t[Eli])
                                                * theta_ztu_df[name_adj_zone].iloc[Tstepi]
                                            )
                                        else:
                                            XTemp = (
                                                (heat_convective_elements_external[Eli] + heat_radiative_elements_external_t[Eli])
                                                * theta_ztu[Tstepi]
                                            )
                                    
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp                                    

                                elif Type_eli[Eli] == "GR":
                                    # Legacy single-zone core is assembled in areal form [W/m2]:
                                    # R_gr_ve is area-specific [m2K/W], so 1 / R_gr_ve is the
                                    # correct areal conductance here. The total-area scaling is
                                    # applied later only in post-processing energy balances.
                                    XTemp = (1 / t_Th.R_gr_ve) * t_Th.Theta_gr_ve[month_arr[Tstepi]]
                                    for cBi in range(nrHCmodes):
                                        VecB[ri, cBi] += XTemp

                            ci = 1 + nodes.PlnSum[Eli] + Pli
                            MatA[ri, ci] += kappa_pli_eli[Pli, Eli] / Dtime[Tstepi]
                            
                            if Pli == (n_nodes - 1):
                                '''
                                in formula (39) Internal surface node - sum/elk=1 to eln)(A_elk/Atot * h_ri_eli  * Teta_pli_elk_t)
                                '''
                      
                                Area_ratio = 0.0
                                for Elk in range(bui_eln):
                                    if nodes.Pln[Elk] == 0:
                                        continue  # no internal node -> no radiative exchange with the zone
                                    Plk = nodes.Pln[Elk] - 1
                                    ck = 1 + nodes.PlnSum[Elk] + Plk
                                    Area_ratio += area_elements[Elk] / area_int_surfaces_tot
                                    MatA[ri, ck] -= (area_elements[Elk] / area_int_surfaces_tot) * heat_radiative_elements_internal[Elk]
                                '''
                                in formula 39  [.. + h_ci_eli  + h_re_eli * sum(elk=1 to eln)(A_elk/Atot) + ..]
                                '''
                                MatA[ri, ci] += (
                                    heat_convective_elements_internal_t[Eli] + 
                                    heat_radiative_elements_internal[Eli] * Area_ratio
                                )
                                MatA[ri, 0] -= heat_convective_elements_internal_t[Eli]


                            elif Pli == 0:
                                
                                if Type_eli[Eli] == "EXT":
                                    MatA[ri, ci] += (
                                        heat_convective_elements_external[Eli]+ 
                                        heat_radiative_elements_external_t[Eli]
                                    )
                                
                                elif Type_eli[Eli] == "ADJ":
                                    MatA[ri, ci] += (
                                        heat_convective_elements_external[Eli]+ 
                                        heat_radiative_elements_external_t[Eli]
                                    )
                                
                                elif Type_eli[Eli] == "GR":
                                    # Areal conductance [W/m2K], consistent with the legacy core.
                                    MatA[ri, ci] += 1 / t_Th.R_gr_ve
                            
                            if Pli > 0:
                                MatA[ri, ci] += h_pli_eli[Pli - 1, Eli] # hpli-1,eli * teta,pli,eli,t
                                MatA[ri, ci - 1] -= h_pli_eli[Pli - 1, Eli] # - hpli-1,eli * teta,pli-1,eli,t
                            
                            if Pli < n_nodes - 1:
                                MatA[ri, ci] += h_pli_eli[Pli, Eli] # hpli,eli * teta,pli,eli,t
                                MatA[ri, ci + 1] -= h_pli_eli[Pli, Eli] # - hpli,eli * teta,pli+1,eli,t
                    
                    '''
                    Temperature calculation of:
                    - internal air
                    - mean radiant temperature
                    - operative temperature

                    '''
                    ######## solve system of equations #######
                    # --- Safe diagonal regularization (avoid read-only error) ---
                    diag_min = max(1e-6, 1e-9 * np.linalg.norm(MatA, ord=np.inf))
                    d = np.diag(MatA).copy()
                    np.maximum(d, diag_min, out=d)
                    np.fill_diagonal(MatA, d)
                    # -------------------------------------------------------------
                    try:
                        theta = np.linalg.solve(MatA, VecB)
                    except np.linalg.LinAlgError:
                        rank = np.linalg.matrix_rank(MatA)
                        print(f"⚠️ MatA solve failed at t={Tstepi}: rank={rank}/{MatA.shape[0]}")
                        print("MatA diagonal:", np.diag(MatA))
                        raise
                    VecB[:, :] = theta
                    
                    # Air Temperature
                    Theta_int_air[Tstepi, :] = VecB[0, :]
                    
                    # --- Mean Radiant Temperature (only internal surfaces) ---
                    Theta_int_r_mn[Tstepi, :] = 0.0
                    A_sum = 0.0
                    for Eli in range(bui_eln):
                        n_nodes_Eli = nodes.Pln[Eli]
                        if n_nodes_Eli == 0:
                            continue  # exclude AD and any surface without node
                        ri_surf = int(surf_int_row[Eli])
                        if ri_surf <= 0 or ri_surf >= VecB.shape[0]:
                            continue
                        Theta_int_r_mn[Tstepi, :] += area_elements[Eli] * VecB[ri_surf, :]
                        A_sum += area_elements[Eli]

                    # uses ONLY the area of internal surfaces; fallback for safety
                    if A_sum == 0.0:
                        A_sum = 1.0
                    Theta_int_r_mn[Tstepi, :] /= A_sum

                    # Operative Temperature
                    Theta_int_op[Tstepi, :] = 0.5 * (Theta_int_air[Tstepi, :] + Theta_int_r_mn[Tstepi, :])
                                        
                    '''
                    STEP 2: ISO 
                    Case heating: Determinates if the heating or the cooling temperature set-point applies and calcualte the heating or cooling load: 
                    Use formaula (27):
                    Phi_HC_ld_zd = Phi_HC_upper*((theta_int_op_set - thet_int_op_0)/(theta_int_op_upper - theta_int_op_0))
                    where:
                    Phi_HC_ld_zd: unrestricted heating or cooling load to reach the required setpoint in W
                    Phi_HC_upper: is the upper value of the heating or cooling load in W  
                    theta_int_op_set: required internal operative setpoint temperature in °C
                    thet_int_op_0: operating temperature in free floating condition in °C
                    theta_int_op_upper: is the internal operational temperature, obtained for the upper heating or cooling load °C
                    '''
                    if nrHCmodes > 1:
                        # HEATING
                        if Theta_int_op[Tstepi, 0] < Theta_H_set:
                            Theta_op_set = Theta_H_set
                            ahu_heat_t = None
                            if external_heating_power_fn is not None:
                                try:
                                    ahu_heat_t = external_heating_power_fn(
                                        tstep=Tstepi,
                                        theta_op=float(Theta_int_op[Tstepi, 0]),
                                        theta_air=float(Theta_int_air[Tstepi, 0]),
                                        theta_rad=float(Theta_int_r_mn[Tstepi, 0]),
                                        theta_h_set=float(Theta_H_set),
                                        theta_c_set=float(Theta_C_set),
                                        weather_row=sim_df.iloc[Tstepi],
                                        profile_row=profile_df.iloc[Tstepi],
                                        power_heating_max=float(power_heating_max_act),
                                    )
                                except TypeError:
                                    # Backward-compatible minimal signature:
                                    # fn(tstep, theta_op, theta_air)
                                    ahu_heat_t = external_heating_power_fn(
                                        Tstepi,
                                        float(Theta_int_op[Tstepi, 0]),
                                        float(Theta_int_air[Tstepi, 0]),
                                    )
                            elif external_heating_power_series is not None:
                                ahu_heat_t = external_heating_power_series[Tstepi]

                            if ahu_heat_t is None:
                                Phi_HC_nd_act[Tstepi] = _safe_load(power_heating_max_act, Theta_op_set,
                                                                Theta_int_op[Tstepi, 0], Theta_int_op[Tstepi, colB_H])
                            else:
                                try:
                                    ahu_heat_t = float(ahu_heat_t)
                                except (TypeError, ValueError):
                                    ahu_heat_t = 0.0
                                if not np.isfinite(ahu_heat_t):
                                    ahu_heat_t = 0.0
                                ahu_heat_t = min(max(ahu_heat_t, 0.0), float(power_heating_max_act))
                                Phi_HC_nd_act[Tstepi] = ahu_heat_t
                                Phi_H_AHU_act[Tstepi] = ahu_heat_t

                            if Phi_HC_nd_act[Tstepi] > power_heating_max_act:
                                Phi_HC_nd_act[Tstepi] = power_heating_max_act
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_H]
                                colB_act = colB_H
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True

                        # COOLING
                        elif Theta_int_op[Tstepi, 0] > Theta_C_set:
                            Theta_op_set = Theta_C_set
                            Phi_HC_nd_act[Tstepi] = _safe_load(power_cooling_max_act, Theta_op_set,
                                                            Theta_int_op[Tstepi, 0], Theta_int_op[Tstepi, colB_C])
                            if Phi_HC_nd_act[Tstepi] < power_cooling_max_act:
                                Phi_HC_nd_act[Tstepi] = power_cooling_max_act
                                Theta_op_act[Tstepi] = Theta_int_op[Tstepi, colB_C]
                                colB_act = colB_C
                            else:
                                Phi_HC_nd_calc[0] = Phi_HC_nd_act[Tstepi]
                                Theta_op_act[Tstepi] = Theta_op_set
                                colB_act = 0
                                nrHCmodes = 1
                                iterate = True

                        else:
                            Phi_HC_nd_act[Tstepi] = 0.0
                            Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                            colB_act = 0
                    else:
                        Phi_HC_nd_act[Tstepi] = Phi_HC_nd_calc[0]
                        Theta_op_act[Tstepi] = Theta_int_op[Tstepi, 0]
                        colB_act = 0

                # Append ventilation diagnostics once per timestep (outside the
                # while loop so HVAC re-solving iterations do not duplicate entries).
                H_ve_nat_all.append(_H_ve_nat_tstep)
                S_ve_nat_all.append(_S_ve_nat_tstep)
                H_ve_nat = _H_ve_nat_tstep
                S_ve_nat = _S_ve_nat_tstep

                # =========================
                # === SANKEY (per timestep)
                # =========================
                dt_h = float(Dtime[Tstepi]) / 3600.0

                # 1) Storage (air + envelope): always update state tracking,
                # but accumulate only after warm-up.
                Theta_curr_state = VecB[:, colB_act]
                dTheta_state = Theta_curr_state - Theta_prev_state
                if Tstepi >= Tstep_first_act:
                    E_storage_Wh += float(np.dot(C_state, dTheta_state)) / 3600.0
                Theta_prev_state = Theta_curr_state

                if Tstepi >= Tstep_first_act:
                    # 2) Direct inputs
                    phi_solar = float(Phi_sol_dir_zt_t)
                    phi_int   = float(int_gains)
                    E_solar_Wh    += phi_solar * dt_h
                    E_internal_Wh += phi_int   * dt_h

                    # 3) Heating/Cooling (uses current load)
                    phi_hc = float(Phi_HC_nd_act[Tstepi])
                    if   phi_hc > 0: E_heating_Wh +=  phi_hc * dt_h
                    elif phi_hc < 0: E_cooling_Wh += (-phi_hc) * dt_h

                    # 4) Ventilation: Q_ve = H_ve * T_in - S_ve (positive = heat leaving zone)
                    T_in  = float(Theta_int_air[Tstepi, 0])
                    T_out = float(sim_df["T2m"].iloc[Tstepi])
                    q_vent = float(H_ve_nat) * T_in - float(S_ve_nat)
                    if q_vent > 0:  E_vent_loss_Wh += q_vent * dt_h
                    else:           E_solar_Wh     += (-q_vent) * dt_h

                    # 5) Thermal bridges
                    q_tb = float(t_Th.thermal_bridge_heat) * (T_in - T_out)
                    if q_tb > 0:  E_tb_loss_Wh += q_tb * dt_h
                    else:         E_solar_Wh   += (-q_tb) * dt_h

                    # 6) Ground
                    T_gr = float(t_Th.Theta_gr_ve[month_arr[Tstepi]])
                    h_ground = _ground_conductance_w_per_k(
                        float(getattr(t_Th, "ground_contact_area", 0.0)),
                        t_Th,
                    )
                    q_ground = h_ground * (T_in - T_gr)
                    if q_ground > 0:  E_ground_loss_Wh += q_ground * dt_h
                    else:             E_solar_Wh       += (-q_ground) * dt_h

                    # 7) Transmission for element (OP, W)
                    T_air = float(Theta_int_air[Tstepi, 0])
                    T_rad = float(Theta_int_r_mn[Tstepi, 0])
                    for Eli in range(bui_eln):
                        if surface_types[Eli] not in ("OP", "W"):  continue
                        n_nodes_Eli = nodes.Pln[Eli]
                        if n_nodes_Eli == 0:                       continue
                        vecb_row_surface = nodes.PlnSum[Eli] + n_nodes_Eli
                        T_surf_int = float(VecB[vecb_row_surface, colB_act])
                        A   = float(area_elements[Eli])
                        hci = float(heat_convective_elements_internal[Eli])
                        hri = float(heat_radiative_elements_internal[Eli])
                        q_cond = A * (hci * (T_air - T_surf_int) + hri * (T_rad - T_surf_int))
                        if   q_cond > 0: E_trans_loss_by_surface_Wh[surface_names[Eli]] += q_cond * dt_h
                        elif q_cond < 0: E_solar_Wh += (-q_cond) * dt_h


                if Tstepi < 6:  # primi 6 passi di debug
                    print(f"[t={Tstepi}] T_op0={Theta_int_op[Tstepi,0]:.2f}  Phi_HC={Phi_HC_nd_act[Tstepi]:.1f}  "
                        f"int_gains={float(int_gains):.1f}  Phi_solar={float(Phi_sol_dir_zt_t):.1f}  "
                        f"H_ve_nat={float(H_ve_nat):.3f}")
                theta_air_prev_causal = float(Theta_int_air[Tstepi, colB_act])
                theta_op_prev_causal = float(Theta_int_op[Tstepi, colB_act])
                theta_rad_prev_causal = float(Theta_int_r_mn[Tstepi, colB_act])
                pbar.update(1)
            n_w=n_w+1

        # =========================
        #  Close balance
        # =========================
        # numeric clamp to avoid -0.0 or microscopically negative values
        def _clamp(x: float) -> float:
            return 0.0 if abs(x) < 1e-9 else x

        # total inputs (Wh)
        inputs_Wh = _clamp(E_heating_Wh) + _clamp(E_internal_Wh) + _clamp(E_solar_Wh)

        # total losses (Wh)
        E_transmission_surfaces_Wh = sum(max(0.0, v) for v in E_trans_loss_by_surface_Wh.values())

        # total outputs (Wh)
        outputs_Wh = (
            _clamp(E_cooling_Wh)      # extracted energy
            + _clamp(E_vent_loss_Wh)  # ventilation
            + _clamp(E_tb_loss_Wh)    # thermal bridges
            + _clamp(E_ground_loss_Wh)# ground
            + _clamp(E_transmission_surfaces_Wh)  # transmission OP/W
        )

        # balance residual (Wh)
        E_transmission_residual_Wh = inputs_Wh - outputs_Wh - _clamp(E_storage_Wh)

        # if the residual is small (<1% of the input) I absorb it into the storage to close the balance
        if inputs_Wh > 0 and abs(E_transmission_residual_Wh) < 0.01 * inputs_Wh:
            E_storage_Wh += E_transmission_residual_Wh
            E_transmission_residual_Wh = 0.0

        # =========================
        #  DATA FOR SANKEY
        # =========================
        sankey_inputs = {
            "Heating": _clamp(E_heating_Wh),
            "Internal gains": _clamp(E_internal_Wh),
            "Solar & free-gain": _clamp(E_solar_Wh),
        }

        sankey_outputs = {
            "Cooling (extracted energy)": _clamp(E_cooling_Wh),
            "Ventilation (losses)": _clamp(E_vent_loss_Wh),
            "Thermal bridges": _clamp(E_tb_loss_Wh),
            "Ground": _clamp(E_ground_loss_Wh),
        }

        # add transmission for each element (only positive branches)
        for name, E_Wh in E_trans_loss_by_surface_Wh.items():
            if E_Wh > 0:
                sankey_outputs[f"Transmission - {name}"] = _clamp(E_Wh)

        # display a non-zero residual (pathological case)
        if E_transmission_residual_Wh > 0:
            sankey_outputs["Transmission (residual)"] = _clamp(E_transmission_residual_Wh)

        sankey_data = {
            "inputs": sankey_inputs,
            "outputs": sankey_outputs,
            "energy_accumulated_zone": _clamp(E_storage_Wh),  # can be non-zero on hourly basis, ~0 on annual basis
        }

        # numeric check
        _inputs = inputs_Wh
        _outs_plus_storage = outputs_Wh + _clamp(E_storage_Wh)
        _res = _inputs - _outs_plus_storage
        _rel = _res / max(1.0, _inputs)
        print(f"SANKEY CHECK  inputs={_inputs:.1f}  outputs+storage={_outs_plus_storage:.1f}  residual={_res:.1f} Wh ({100*_rel:.3f}%)")

        # =========================
        #  HOURLY AND ANNUAL RESULTS
        # =========================
        act_slice = slice(Tstep_first_act, Tstepn)
        sim_df_act = sim_df.iloc[Tstep_first_act:Tstepn]
        hourly_results = pd.DataFrame(
            data=np.vstack(
                (
                    Phi_HC_nd_act[act_slice],
                    Theta_op_act[act_slice],
                    sim_df_act["T2m"].to_numpy(),
                )
            ).T,
            index=sim_df_act.index,
            columns=["Q_HC", "T_op", "T_ext"],
        )
        hourly_results["T_air"] = Theta_int_air[act_slice, 0]
        hourly_results["T_rad"] = Theta_int_r_mn[act_slice, 0]

        # Zone air and ventilation diagnostics
        _h_ve_arr = np.array(H_ve_nat_all[1:Tstepn + 1], dtype=float)[act_slice]
        _s_ve_arr = np.array(S_ve_nat_all[1:Tstepn + 1], dtype=float)[act_slice]
        _t_air_arr = Theta_int_air[act_slice, 0]
        hourly_results["T_air"] = _t_air_arr
        hourly_results["H_ve"] = _h_ve_arr
        hourly_results["S_ve"] = _s_ve_arr
        _t_eq = np.full_like(_h_ve_arr, np.nan, dtype=float)
        np.divide(
            _s_ve_arr,
            _h_ve_arr,
            out=_t_eq,
            where=_h_ve_arr > 0.0,
        )
        hourly_results["T_ve_source_eq"] = _t_eq
        hourly_results["Q_ve"] = _h_ve_arr * _t_air_arr - _s_ve_arr

        # separate H/C
        hourly_results["Q_H"] = 0.0
        hourly_results.loc[hourly_results["Q_HC"] > 0, "Q_H"] = hourly_results.loc[hourly_results["Q_HC"] > 0, "Q_HC"]

        hourly_results["Q_C"] = 0.0
        hourly_results.loc[hourly_results["Q_HC"] < 0, "Q_C"] = -hourly_results.loc[hourly_results["Q_HC"] < 0, "Q_HC"]
        if (
            external_heating_power_fn is not None
            or external_heating_power_causal_fn is not None
            or external_heating_power_series is not None
        ):
            hourly_results["Q_H_AHU_used"] = Phi_H_AHU_act[act_slice]

        dt_h_annual = _infer_timestep_hours_from_index(hourly_results.index, default=1.0)
        Q_H_annual = _integrate_power_series_to_energy_wh(hourly_results["Q_H"], default_dt_h=dt_h_annual)
        Q_C_annual = _integrate_power_series_to_energy_wh(hourly_results["Q_C"], default_dt_h=dt_h_annual)
        A_use = float(building_object['building']['net_floor_area'])

        annual_results_dic = {
            "Q_H_annual": Q_H_annual,
            "Q_C_annual": Q_C_annual,
            "Q_H_annual_per_sqm": Q_H_annual / A_use if A_use > 0 else 0.0,
            "Q_C_annual_per_sqm": Q_C_annual / A_use if A_use > 0 else 0.0,
            "Q_H_annual_kWh": Q_H_annual / 1000.0,
            "Q_C_annual_kWh": Q_C_annual / 1000.0,
            "Q_H_annual_kWh_per_sqm": (Q_H_annual / 1000.0 / A_use) if A_use > 0 else 0.0,
            "Q_C_annual_kWh_per_sqm": (Q_C_annual / 1000.0 / A_use) if A_use > 0 else 0.0,
            "time_step_h": dt_h_annual,
        }
        annual_results_df = pd.DataFrame([annual_results_dic])

        # Sankey
        if sankey_graph:
            fig = plot_sankey_building(sankey_data)
            fig.show()

        return hourly_results, annual_results_df, sankey_data
