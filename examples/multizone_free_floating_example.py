import argparse
import os
import sys
import json
import copy
import time
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure local package import works when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pybuildingenergy.source.utils import ISO52016
from pybuildingenergy.source.functions import plot_sankey_building


def infer_timestep_hours(index: pd.Index, default: float = 1.0) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return float(default)
    dt_h = (
        index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600.0)
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
    )
    dt_h = dt_h[dt_h > 0.0]
    if dt_h.empty:
        return float(default)
    return float(dt_h.median())


def power_series_to_monthly_energy_kwh(power_w: pd.Series) -> pd.Series:
    p = pd.to_numeric(power_w, errors="coerce").fillna(0.0)
    if p.empty:
        return pd.Series(dtype=float)
    if not isinstance(p.index, pd.DatetimeIndex):
        return (p / 1000.0).groupby((pd.Series(range(len(p)), index=p.index) // 720)).sum()

    idx_series = p.index.to_series(index=p.index)
    fallback_dt_h = infer_timestep_hours(p.index, default=1.0)
    dt_h = idx_series.shift(-1).sub(idx_series).dt.total_seconds().div(3600.0)
    dt_h = dt_h.fillna(fallback_dt_h)
    dt_h = dt_h.where((dt_h > 0.0) & dt_h.notna(), fallback_dt_h)
    e_kwh_step = p * dt_h / 1000.0
    return e_kwh_step.resample("ME").sum()


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _get_first_matching_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    values = df.loc[:, col]
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return None
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def _ground_conductance_from_area(area_m2: float, r_gr_ve_m2k_w: float) -> float:
    try:
        area = float(area_m2)
        r_gr = float(r_gr_ve_m2k_w)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(area) or area <= 0.0:
        return 0.0
    if not np.isfinite(r_gr) or r_gr <= 0.0:
        return 0.0
    return area / r_gr


def _compute_ground_temperature_and_exchanges(
    building_object: dict,
    hourly_results: pd.DataFrame,
    weather_path: str,
    weather_source: str = "epw",
) -> pd.DataFrame:
    hourly = hourly_results.copy()
    hourly.index = pd.DatetimeIndex(hourly.index)
    hourly = hourly.loc[~hourly.index.duplicated(keep="first")].sort_index()

    zones = building_object.get("zones", [])
    if not zones:
        zones = [{"name": "main", "net_floor_area": float(building_object["building"]["net_floor_area"])}]
    zone_names = [str(z["name"]) for z in zones]
    default_zone = zone_names[0]

    out = pd.DataFrame(index=hourly.index)
    if hourly.empty:
        return out

    required_ground_cols = ["T_ground_virtual"]
    for zname in zone_names:
        required_ground_cols.extend([f"H_ground_{zname}", f"Q_ground_{zname}"])
    if all(col in hourly.columns for col in required_ground_cols):
        return hourly[required_ground_cols].copy()

    has_ground = False
    for surf in building_object.get("building_surface", []):
        bnd = str(surf.get("boundary", "OUTDOORS")).upper()
        iso_type = str(surf.get("ISO52016_type_string", "")).upper()
        if bnd == "GROUND" or iso_type == "GR":
            has_ground = True
            break

    if not has_ground:
        out["T_ground_virtual"] = np.nan
        for zname in zone_names:
            out[f"H_ground_{zname}"] = 0.0
            out[f"Q_ground_{zname}"] = 0.0
        return out

    t_Th = ISO52016().Temp_calculation_of_ground(
        copy.deepcopy(building_object),
        path_weather_file=weather_path,
        weather_source=weather_source,
    )
    theta_gr_monthly = np.asarray(t_Th.Theta_gr_ve, dtype=float)
    if theta_gr_monthly.size == 0:
        theta_gr_monthly = np.full(12, np.nan, dtype=float)
    elif theta_gr_monthly.size < 12:
        theta_gr_monthly = np.resize(theta_gr_monthly, 12)

    zone_ground_h = {zname: 0.0 for zname in zone_names}
    for surf in building_object.get("building_surface", []):
        bnd = str(surf.get("boundary", "OUTDOORS")).upper()
        iso_type = str(surf.get("ISO52016_type_string", "")).upper()
        if bnd != "GROUND" and iso_type != "GR":
            continue
        try:
            area = float(surf.get("area", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(area) or area <= 0.0:
            continue
        zname = surf.get("zone", default_zone)
        if zname not in zone_ground_h:
            zname = default_zone
        zone_ground_h[zname] += _ground_conductance_from_area(area, float(t_Th.R_gr_ve))

    month_idx = pd.DatetimeIndex(hourly.index).month.to_numpy(dtype=int)
    t_ground = np.array(
        [theta_gr_monthly[m - 1] if 1 <= int(m) <= 12 else theta_gr_monthly[0] for m in month_idx],
        dtype=float,
    )
    out["T_ground_virtual"] = t_ground

    for zname in zone_names:
        h_ground = float(zone_ground_h.get(zname, 0.0))
        out[f"H_ground_{zname}"] = h_ground
        t_air_col = f"T_air_{zname}"
        if t_air_col in hourly.columns:
            t_air = pd.to_numeric(hourly[t_air_col], errors="coerce")
            out[f"Q_ground_{zname}"] = h_ground * (t_air - out["T_ground_virtual"])
        else:
            out[f"Q_ground_{zname}"] = np.nan

    return out


def _monthly_hvac_energy_by_zone(
    df: pd.DataFrame,
    zone_names: list[str],
    hvac_col_candidates: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_h = pd.DataFrame()
    monthly_c = pd.DataFrame()
    for zname in zone_names:
        q_col = _pick_first_existing_col(df, hvac_col_candidates.get(zname, []))
        if q_col is None:
            continue
        q_series = pd.to_numeric(df[q_col], errors="coerce").fillna(0.0)
        monthly_h[zname] = power_series_to_monthly_energy_kwh(q_series.clip(lower=0.0))
        monthly_c[zname] = power_series_to_monthly_energy_kwh(-q_series.clip(upper=0.0))
    return monthly_h.fillna(0.0), monthly_c.fillna(0.0)


def _write_temperature_plot(
    df: pd.DataFrame,
    zone_names: list[str],
    temp_col_candidates: dict[str, list[str]],
    out_html_path: str,
    title: str,
) -> None:
    fig_temp = go.Figure()
    for zname in zone_names:
        temp_col = _pick_first_existing_col(df, temp_col_candidates.get(zname, []))
        if temp_col is None:
            continue
        fig_temp.add_trace(
            go.Scatter(
                x=df.index,
                y=pd.to_numeric(df[temp_col], errors="coerce"),
                mode="lines",
                name=f"{zname} ({temp_col}) [degC]",
                line={"width": 1.2},
                hovertemplate=f"{temp_col}: %{{y:.2f}} degC<extra></extra>",
            )
        )
    fig_temp.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Temperature [degC]",
        template="plotly_white",
        hovermode="x unified",
    )
    if isinstance(df.index, pd.DatetimeIndex):
        fig_temp.update_xaxes(rangeslider={"visible": True})
    fig_temp.write_html(out_html_path, include_plotlyjs="cdn", full_html=True)


def _write_monthly_hvac_plot(
    monthly_h: pd.DataFrame,
    monthly_c: pd.DataFrame,
    zone_names: list[str],
    out_html_path: str,
    title_h: str,
    title_c: str,
    trace_suffix: str,
) -> None:
    fig_cons = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(title_h, title_c),
    )
    month_idx = monthly_h.index if not monthly_h.empty else monthly_c.index
    month_labels = [ts.strftime("%Y-%m") if hasattr(ts, "strftime") else str(ts) for ts in month_idx]
    for zname in zone_names:
        if zname in monthly_h.columns:
            fig_cons.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_h[zname].values,
                    mode="lines+markers",
                    name=f"{zname} H {trace_suffix} [kWh]",
                ),
                row=1,
                col=1,
            )
        if zname in monthly_c.columns:
            fig_cons.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_c[zname].values,
                    mode="lines+markers",
                    name=f"{zname} C {trace_suffix} [kWh]",
                ),
                row=2,
                col=1,
            )
    fig_cons.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis2_title="Month",
        yaxis_title="Energy [kWh/month]",
        yaxis2_title="Energy [kWh/month]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig_cons.write_html(out_html_path, include_plotlyjs="cdn", full_html=True)


def _write_ground_exchange_plot(
    df: pd.DataFrame,
    zone_names: list[str],
    out_html_path: str,
    title_temp: str,
    title_flux: str,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(title_temp, title_flux),
    )
    t_ground = _get_first_matching_series(df, "T_ground_virtual")
    if t_ground is not None:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=t_ground,
                mode="lines",
                name="T_ground_virtual [degC]",
                line={"width": 1.8, "color": "#6b4f2a"},
                hovertemplate="T_ground_virtual: %{y:.2f} degC<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for zname in zone_names:
        q_col = f"Q_ground_{zname}"
        q_ground = _get_first_matching_series(df, q_col)
        if q_ground is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=q_ground,
                mode="lines",
                name=f"{zname} ({q_col}) [W]",
                line={"width": 1.2},
                hovertemplate=f"{q_col}: %{{y:.1f}} W<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis2_title="Time",
        yaxis_title="Temperature [degC]",
        yaxis2_title="Heat exchange [W]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    if isinstance(df.index, pd.DatetimeIndex):
        fig.update_xaxes(rangeslider={"visible": True}, row=2, col=1)
    fig.write_html(out_html_path, include_plotlyjs="cdn", full_html=True)


def _build_v1_vs_hybrid_annual_summary(
    annual_v1: pd.DataFrame,
    annual_v2: pd.DataFrame,
) -> pd.DataFrame:
    cols_v1 = [c for c in ["zone", "Q_H_annual_kWh", "Q_C_annual_kWh"] if c in annual_v1.columns]
    cols_v2 = [c for c in ["zone", "Q_H_annual_hybrid_kWh", "Q_C_annual_hybrid_kWh"] if c in annual_v2.columns]
    if "zone" not in cols_v1 or "zone" not in cols_v2:
        return pd.DataFrame()

    merged = annual_v1.loc[:, cols_v1].merge(
        annual_v2.loc[:, cols_v2],
        on="zone",
        how="outer",
    )
    if "Q_H_annual_kWh" in merged.columns and "Q_H_annual_hybrid_kWh" in merged.columns:
        merged["Delta_Q_H_hybrid_minus_v1_kWh"] = (
            pd.to_numeric(merged["Q_H_annual_hybrid_kWh"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["Q_H_annual_kWh"], errors="coerce").fillna(0.0)
        )
    if "Q_C_annual_kWh" in merged.columns and "Q_C_annual_hybrid_kWh" in merged.columns:
        merged["Delta_Q_C_hybrid_minus_v1_kWh"] = (
            pd.to_numeric(merged["Q_C_annual_hybrid_kWh"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["Q_C_annual_kWh"], errors="coerce").fillna(0.0)
        )
    return merged


def _write_v1_vs_hybrid_report(
    v1_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    annual_summary_df: pd.DataFrame,
    iterations_df: pd.DataFrame,
    zone_names: list[str],
    out_html_path: str,
) -> None:
    common_idx = v1_df.index.intersection(hybrid_df.index).sort_values()
    v1 = v1_df.reindex(common_idx)
    hybrid = hybrid_df.reindex(common_idx)

    monthly_h_v1, monthly_c_v1 = _monthly_hvac_energy_by_zone(
        df=v1,
        zone_names=zone_names,
        hvac_col_candidates={z: [f"Q_HVAC_{z}"] for z in zone_names},
    )
    monthly_h_v2, monthly_c_v2 = _monthly_hvac_energy_by_zone(
        df=hybrid,
        zone_names=zone_names,
        hvac_col_candidates={z: [f"Q_HC_hybrid_{z}"] for z in zone_names},
    )

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.22, 0.22, 0.16, 0.16, 0.14],
        subplot_titles=(
            "Hourly operative temperature: V1 vs Hybrid",
            "Hourly HVAC power: V1 vs Hybrid",
            "Monthly heating energy: V1 vs Hybrid",
            "Monthly cooling energy: V1 vs Hybrid",
            "Hybrid iteration convergence",
        ),
    )

    for zname in zone_names:
        v1_t_col = _pick_first_existing_col(v1, [f"T_op_{zname}", f"T_air_{zname}"])
        v2_t_col = _pick_first_existing_col(hybrid, [f"T_op_core_{zname}", f"T_air_core_{zname}"])
        if v1_t_col is not None:
            fig.add_trace(
                go.Scatter(
                    x=v1.index,
                    y=pd.to_numeric(v1[v1_t_col], errors="coerce"),
                    mode="lines",
                    name=f"{zname} V1 temp",
                    legendgroup=f"{zname}_temp",
                ),
                row=1,
                col=1,
            )
        if v2_t_col is not None:
            fig.add_trace(
                go.Scatter(
                    x=hybrid.index,
                    y=pd.to_numeric(hybrid[v2_t_col], errors="coerce"),
                    mode="lines",
                    name=f"{zname} Hybrid temp",
                    legendgroup=f"{zname}_temp",
                    line={"dash": "dash"},
                ),
                row=1,
                col=1,
            )

        v1_q_col = _pick_first_existing_col(v1, [f"Q_HVAC_{zname}"])
        v2_q_col = _pick_first_existing_col(hybrid, [f"Q_HC_hybrid_{zname}"])
        if v1_q_col is not None:
            fig.add_trace(
                go.Scatter(
                    x=v1.index,
                    y=pd.to_numeric(v1[v1_q_col], errors="coerce"),
                    mode="lines",
                    name=f"{zname} V1 HVAC",
                    legendgroup=f"{zname}_hvac",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        if v2_q_col is not None:
            fig.add_trace(
                go.Scatter(
                    x=hybrid.index,
                    y=pd.to_numeric(hybrid[v2_q_col], errors="coerce"),
                    mode="lines",
                    name=f"{zname} Hybrid HVAC",
                    legendgroup=f"{zname}_hvac",
                    line={"dash": "dash"},
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    month_idx = monthly_h_v1.index if not monthly_h_v1.empty else monthly_h_v2.index
    if len(month_idx) == 0:
        month_idx = monthly_c_v1.index if not monthly_c_v1.empty else monthly_c_v2.index
    month_labels = [ts.strftime("%Y-%m") if hasattr(ts, "strftime") else str(ts) for ts in month_idx]

    for zname in zone_names:
        if zname in monthly_h_v1.columns:
            fig.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_h_v1.reindex(month_idx)[zname].values,
                    mode="lines+markers",
                    name=f"{zname} H V1",
                    legendgroup=f"{zname}_monthly_h",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        if zname in monthly_h_v2.columns:
            fig.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_h_v2.reindex(month_idx)[zname].values,
                    mode="lines+markers",
                    name=f"{zname} H Hybrid",
                    legendgroup=f"{zname}_monthly_h",
                    line={"dash": "dash"},
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        if zname in monthly_c_v1.columns:
            fig.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_c_v1.reindex(month_idx)[zname].values,
                    mode="lines+markers",
                    name=f"{zname} C V1",
                    legendgroup=f"{zname}_monthly_c",
                    showlegend=False,
                ),
                row=4,
                col=1,
            )
        if zname in monthly_c_v2.columns:
            fig.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_c_v2.reindex(month_idx)[zname].values,
                    mode="lines+markers",
                    name=f"{zname} C Hybrid",
                    legendgroup=f"{zname}_monthly_c",
                    line={"dash": "dash"},
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

    if not iterations_df.empty and "iteration" in iterations_df.columns:
        if "max_abs_delta_coupling_W" in iterations_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=iterations_df["iteration"],
                    y=pd.to_numeric(iterations_df["max_abs_delta_coupling_W"], errors="coerce"),
                    mode="lines+markers",
                    name="Max abs delta coupling [W]",
                    legendgroup="hybrid_iter",
                    showlegend=False,
                ),
                row=5,
                col=1,
            )
        if "mean_abs_delta_coupling_W" in iterations_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=iterations_df["iteration"],
                    y=pd.to_numeric(iterations_df["mean_abs_delta_coupling_W"], errors="coerce"),
                    mode="lines+markers",
                    name="Mean abs delta coupling [W]",
                    legendgroup="hybrid_iter",
                    line={"dash": "dash"},
                    showlegend=False,
                ),
                row=5,
                col=1,
            )

    fig.update_layout(
        title="Multizone comparison: fully integrated V1 vs hybrid iterative V2",
        template="plotly_white",
        hovermode="x unified",
        height=1800,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_yaxes(title_text="Temperature [degC]", row=1, col=1)
    fig.update_yaxes(title_text="Power [W]", row=2, col=1)
    fig.update_yaxes(title_text="Energy [kWh/month]", row=3, col=1)
    fig.update_yaxes(title_text="Energy [kWh/month]", row=4, col=1)
    fig.update_yaxes(title_text="Coupling delta [W]", row=5, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_xaxes(title_text="Month", row=4, col=1)
    fig.update_xaxes(title_text="Iteration", row=5, col=1)

    sections = [fig.to_html(include_plotlyjs="cdn", full_html=False)]
    if not annual_summary_df.empty:
        table = annual_summary_df.copy()
        for col in table.columns:
            if col == "zone":
                continue
            table[col] = pd.to_numeric(table[col], errors="coerce").map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        sections.append("<h2>Annual summary [kWh]</h2>")
        sections.append(table.to_html(index=False, border=0))
    if not iterations_df.empty:
        table = iterations_df.copy()
        for col in table.columns:
            if col == "iteration":
                continue
            table[col] = pd.to_numeric(table[col], errors="coerce").map(
                lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
            )
        sections.append("<h2>Hybrid iteration diagnostics</h2>")
        sections.append(table.to_html(index=False, border=0))

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Multizone V1 vs Hybrid V2</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;}th,td{padding:6px 8px;border:1px solid #ddd;}"
        "th{background:#f5f5f5;}</style></head><body>"
        "<h1>Multizone comparison: fully integrated V1 vs hybrid iterative V2</h1>"
        + "".join(sections)
        + "</body></html>"
    )
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _build_shading_components_by_zone_orientation(building_object: dict, zone_names: list[str]) -> dict:
    shading_components = {}
    default_zone = zone_names[0] if zone_names else "main"
    for surf in building_object.get("building_surface", []):
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
        zname = surf.get("zone", default_zone)
        if zname not in zone_names:
            zname = default_zone
        ori = str(surf.get("ISO52016_orientation_string", "SV")).upper()
        key = (zname, ori)
        shading_components.setdefault(key, []).append((win_name, win_area))
    return shading_components


def _surface_orientation_signature(surf: dict) -> tuple[str, float, float]:
    ori_tag = str(surf.get("ISO52016_orientation_string", "")).strip().upper()
    orientation = surf.get("orientation", {}) if isinstance(surf.get("orientation", {}), dict) else {}
    try:
        azimuth = float(orientation.get("azimuth", 0.0)) % 360.0
    except Exception:
        azimuth = 0.0
    try:
        tilt = float(orientation.get("tilt", 90.0))
    except Exception:
        tilt = 90.0
    return ori_tag, azimuth, tilt


def _normalize_idf_gross_opaque_areas(building_object: dict) -> tuple[dict, list[dict]]:
    bui = copy.deepcopy(building_object)
    building_meta = bui.get("building", {}) if isinstance(bui, dict) else {}
    if not isinstance(building_meta, dict):
        return bui, []
    if str(building_meta.get("model_source", "")).strip().lower() != "idf":
        return bui, []
    if bool(building_meta.get("_opaque_areas_normalized", False)):
        return bui, []

    area_mode = str(building_meta.get("opaque_surface_area_mode", "net")).strip().lower()
    if area_mode not in {"gross", "gross_from_idf", "gross_including_windows"}:
        return bui, []

    surfaces = bui.get("building_surface", [])
    if not isinstance(surfaces, list) or not surfaces:
        return bui, []

    windows_by_group: dict[tuple[str, str, float, float], float] = {}
    opaque_by_group: dict[tuple[str, str, float, float], list[dict]] = {}

    for idx, surf in enumerate(surfaces):
        if not isinstance(surf, dict):
            continue
        if str(surf.get("boundary", "OUTDOORS")).upper() != "OUTDOORS":
            continue
        surf_type = str(surf.get("type", "")).strip().lower()
        try:
            area = float(surf.get("area", 0.0))
        except Exception:
            continue
        if not np.isfinite(area) or area <= 0.0:
            continue
        zone_name = str(surf.get("zone", "main")).strip() or "main"
        key = (zone_name, *_surface_orientation_signature(surf))
        if surf_type == "transparent":
            windows_by_group[key] = windows_by_group.get(key, 0.0) + area
        elif surf_type == "opaque":
            opaque_by_group.setdefault(key, []).append(
                {
                    "idx": idx,
                    "name": str(surf.get("name", f"surface_{idx}")),
                    "area": area,
                }
            )

    adjustments: list[dict] = []
    for key, opaque_items in opaque_by_group.items():
        window_area = float(windows_by_group.get(key, 0.0))
        if window_area <= 0.0:
            continue
        gross_opaque_area = float(sum(item["area"] for item in opaque_items))
        if gross_opaque_area <= 0.0:
            continue

        net_opaque_area = max(0.0, gross_opaque_area - window_area)
        scale = net_opaque_area / gross_opaque_area if gross_opaque_area > 0.0 else 0.0

        for item in opaque_items:
            surf = surfaces[item["idx"]]
            gross_area = float(item["area"])
            net_area = gross_area * scale
            surf["gross_area"] = gross_area
            surf["window_area_subtracted"] = gross_area - net_area
            surf["area"] = net_area

        adjustments.append(
            {
                "zone": key[0],
                "orientation": key[1] or f"az={key[2]:.0f},tilt={key[3]:.0f}",
                "surface_names": [item["name"] for item in opaque_items],
                "gross_opaque_area": gross_opaque_area,
                "window_area": window_area,
                "net_opaque_area": net_opaque_area,
                "overflow_window_area": max(0.0, window_area - gross_opaque_area),
            }
        )

    building_meta["_opaque_areas_normalized"] = True
    building_meta["opaque_surface_area_mode_effective"] = "net"
    return bui, adjustments


def _compute_multizone_sankey_by_zone(
    building_object: dict,
    hourly_results: pd.DataFrame,
    weather_path: str,
    weather_source: str = "epw",
) -> tuple[dict, pd.DataFrame]:
    zones = building_object.get("zones", [])
    if not zones:
        zones = [{"name": "main", "net_floor_area": float(building_object["building"]["net_floor_area"])}]
    zone_names = [z["name"] for z in zones]
    default_zone = zone_names[0]

    # Weather and time alignment
    sim_df = ISO52016().Weather_data_bui(
        copy.deepcopy(building_object),
        weather_path,
        weather_source=weather_source,
    ).simulation_df
    sim_df = sim_df.copy()
    sim_df.index = pd.DatetimeIndex(sim_df.index)
    sim_df = sim_df.loc[~sim_df.index.duplicated(keep="first")].sort_index()

    hourly = hourly_results.copy()
    hourly.index = pd.DatetimeIndex(hourly.index)
    hourly = hourly.loc[~hourly.index.duplicated(keep="first")].sort_index()

    common_idx = hourly.index.intersection(sim_df.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping timestamps between hourly multizone results and weather data.")
    hourly = hourly.reindex(common_idx)
    sim_df = sim_df.reindex(common_idx)

    t_out = pd.to_numeric(sim_df["T2m"], errors="coerce").interpolate(limit_direction="both")
    dt_h = infer_timestep_hours(common_idx, default=1.0)

    ground_df = _compute_ground_temperature_and_exchanges(
        building_object=building_object,
        hourly_results=hourly,
        weather_path=weather_path,
        weather_source=weather_source,
    ).reindex(common_idx)

    # Ground and thermal bridges (same helper as core)
    t_Th = ISO52016().Temp_calculation_of_ground(
        copy.deepcopy(building_object),
        path_weather_file=weather_path,
        weather_source=weather_source,
        psi_k=0
    )

    zone_areas = {
        z["name"]: float(z.get("net_floor_area", building_object["building"].get("net_floor_area", 0.0)))
        for z in zones
    }
    area_tot = float(sum(max(0.0, v) for v in zone_areas.values())) or 1.0
    H_tb_tot = float(t_Th.thermal_bridge_heat)
    H_tb_zone = {zn: H_tb_tot * (max(0.0, zone_areas.get(zn, 0.0)) / area_tot) for zn in zone_names}

    shading_components = _build_shading_components_by_zone_orientation(building_object, zone_names)

    # Surface buckets per zone
    windows_by_zone = {zn: [] for zn in zone_names}
    trans_ext_by_zone = {zn: [] for zn in zone_names}  # OP/W towards outdoors (excluding ground)
    for surf in building_object.get("building_surface", []):
        zn = surf.get("zone", default_zone)
        if zn not in zone_names:
            zn = default_zone
        bnd = str(surf.get("boundary", "OUTDOORS")).upper()
        typ = str(surf.get("type", "")).lower()
        if typ == "transparent" and bnd == "OUTDOORS":
            windows_by_zone[zn].append(surf)
            trans_ext_by_zone[zn].append(surf)
        elif typ == "opaque" and bnd == "OUTDOORS":
            trans_ext_by_zone[zn].append(surf)

    # Energy accumulators [Wh] per zone
    acc = {
        zn: {
            "heating_Wh": 0.0,
            "cooling_Wh": 0.0,
            "internal_Wh": 0.0,
            "solar_free_Wh": 0.0,
            "vent_loss_Wh": 0.0,
            "tb_loss_Wh": 0.0,
            "ground_loss_Wh": 0.0,
            "trans_loss_Wh": 0.0,
        }
        for zn in zone_names
    }

    for tstep, ts in enumerate(common_idx):
        Tout = float(t_out.iloc[tstep]) if np.isfinite(t_out.iloc[tstep]) else 0.0

        for zn in zone_names:
            t_air_col = f"T_air_{zn}"
            q_hvac_col = f"Q_HVAC_{zn}"
            phi_int_col = f"Phi_int_{zn}"
            h_ve_col = f"H_ve_{zn}"

            if t_air_col not in hourly.columns:
                continue

            Tair = float(pd.to_numeric(hourly[t_air_col], errors="coerce").iloc[tstep])
            if not np.isfinite(Tair):
                continue

            # Inputs
            q_hvac = 0.0
            if q_hvac_col in hourly.columns:
                q_hvac = float(pd.to_numeric(hourly[q_hvac_col], errors="coerce").fillna(0.0).iloc[tstep])
            if q_hvac > 0.0:
                acc[zn]["heating_Wh"] += q_hvac * dt_h
            elif q_hvac < 0.0:
                acc[zn]["cooling_Wh"] += (-q_hvac) * dt_h

            phi_int = 0.0
            if phi_int_col in hourly.columns:
                phi_int = float(pd.to_numeric(hourly[phi_int_col], errors="coerce").fillna(0.0).iloc[tstep])
            if phi_int > 0.0:
                acc[zn]["internal_Wh"] += phi_int * dt_h

            # Solar transmitted through windows (with shading factor)
            q_solar = 0.0
            for surf in windows_by_zone[zn]:
                ori = str(surf.get("ISO52016_orientation_string", "SV")).upper()
                col = f"I_sol_tot_{ori}"
                if col not in sim_df.columns:
                    continue
                try:
                    g_val = float(surf.get("g_value", 0.0))
                    area = float(surf.get("area", 0.0))
                    i_tot = float(pd.to_numeric(sim_df[col], errors="coerce").fillna(0.0).iloc[tstep])
                except Exception:
                    continue
                f_sh = ISO52016._surface_shading_factor_from_timeseries(
                    sim_df=sim_df,
                    tstep=tstep,
                    surface=surf,
                    shading_components_by_zone_orientation=shading_components,
                    default_zone=default_zone,
                )
                q_solar += max(0.0, g_val * area * i_tot * float(f_sh))
            acc[zn]["solar_free_Wh"] += q_solar * dt_h

            # Ventilation
            h_ve = 0.0
            if h_ve_col in hourly.columns:
                h_ve = float(pd.to_numeric(hourly[h_ve_col], errors="coerce").fillna(0.0).iloc[tstep])
            q_vent = h_ve * (Tair - Tout)
            if q_vent > 0.0:
                acc[zn]["vent_loss_Wh"] += q_vent * dt_h
            else:
                acc[zn]["solar_free_Wh"] += (-q_vent) * dt_h

            # Thermal bridges
            q_tb = float(H_tb_zone.get(zn, 0.0)) * (Tair - Tout)
            if q_tb > 0.0:
                acc[zn]["tb_loss_Wh"] += q_tb * dt_h
            else:
                acc[zn]["solar_free_Wh"] += (-q_tb) * dt_h

            # Ground
            q_ground_col = f"Q_ground_{zn}"
            q_ground = 0.0
            if q_ground_col in ground_df.columns:
                q_ground = float(pd.to_numeric(ground_df[q_ground_col], errors="coerce").fillna(0.0).iloc[tstep])
            if q_ground > 0.0:
                acc[zn]["ground_loss_Wh"] += q_ground * dt_h
            else:
                acc[zn]["solar_free_Wh"] += (-q_ground) * dt_h

            # Transmission through external OP/W surfaces (UA * dT)
            q_trans = 0.0
            for surf in trans_ext_by_zone[zn]:
                try:
                    ua = float(surf.get("u_value", 0.0)) * float(surf.get("area", 0.0))
                except Exception:
                    ua = 0.0
                if not np.isfinite(ua) or ua <= 0.0:
                    continue
                q = ua * (Tair - Tout)
                if q > 0.0:
                    q_trans += q
                else:
                    acc[zn]["solar_free_Wh"] += (-q) * dt_h
            acc[zn]["trans_loss_Wh"] += q_trans * dt_h

    def _clamp(x: float) -> float:
        return 0.0 if abs(float(x)) < 1e-9 else float(x)

    sankey_by_zone = {}
    summary_rows = []
    for zn in zone_names:
        a = acc[zn]
        inputs = _clamp(a["heating_Wh"]) + _clamp(a["internal_Wh"]) + _clamp(a["solar_free_Wh"])
        outputs = (
            _clamp(a["cooling_Wh"])
            + _clamp(a["vent_loss_Wh"])
            + _clamp(a["tb_loss_Wh"])
            + _clamp(a["ground_loss_Wh"])
            + _clamp(a["trans_loss_Wh"])
        )
        storage = _clamp(inputs - outputs)

        sankey_inputs = {
            "Heating": _clamp(a["heating_Wh"]),
            "Internal gains": _clamp(a["internal_Wh"]),
            "Solar & free-gain": _clamp(a["solar_free_Wh"]),
        }
        sankey_outputs = {
            "Cooling (extracted energy)": _clamp(a["cooling_Wh"]),
            "Ventilation (losses)": _clamp(a["vent_loss_Wh"]),
            "Thermal bridges": _clamp(a["tb_loss_Wh"]),
            "Ground": _clamp(a["ground_loss_Wh"]),
            "Transmission (envelope)": _clamp(a["trans_loss_Wh"]),
        }

        sankey_by_zone[zn] = {
            "inputs": sankey_inputs,
            "outputs": sankey_outputs,
            "energy_accumulated_zone": storage,
        }

        summary_rows.append(
            {
                "zone": zn,
                "Heating_Wh": sankey_inputs["Heating"],
                "Internal_gains_Wh": sankey_inputs["Internal gains"],
                "Solar_free_Wh": sankey_inputs["Solar & free-gain"],
                "Cooling_Wh": sankey_outputs["Cooling (extracted energy)"],
                "Ventilation_losses_Wh": sankey_outputs["Ventilation (losses)"],
                "Thermal_bridges_Wh": sankey_outputs["Thermal bridges"],
                "Ground_Wh": sankey_outputs["Ground"],
                "Transmission_envelope_Wh": sankey_outputs["Transmission (envelope)"],
                "Storage_Wh": storage,
            }
        )

    return sankey_by_zone, pd.DataFrame(summary_rows)


def _write_multizone_sankey_html(
    sankey_by_zone: dict,
    summary_df: pd.DataFrame,
    out_html_path: str,
    building_object: dict | None = None,
    hourly_results: pd.DataFrame | None = None,
) -> None:
    sections = []
    sections.append("<h1>Multizone Annual Energy Sankey by Thermal Zone</h1>")
    sections.append("<p>Convention: values in Wh, cooling is extracted energy.</p>")
    sections.append("<h2>Summary</h2>")
    sections.append(summary_df.to_html(index=False, border=0))

    include_js = True
    for zone_name, sankey_data in sankey_by_zone.items():
        fig = plot_sankey_building(sankey_data)
        fig.update_layout(title=f"Annual energy balance — Sankey ({zone_name})", font_size=12)
        sections.append(f"<h2>Zone {zone_name}</h2>")
        sections.append(fig.to_html(include_plotlyjs="cdn" if include_js else False, full_html=False))
        include_js = False

    if building_object is not None and hourly_results is not None:
        sections.append("<h2>Heat exchanged between thermal zones</h2>")
        sections.append(_interzone_heat_exchange_html_fragment(building_object, hourly_results))

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Multizone Sankey by Zone</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;} "
        "table{border-collapse:collapse;} th,td{padding:6px 8px;border:1px solid #ddd;} "
        "th{background:#f5f5f5;}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _sanitize_result_column_token(value: object) -> str:
    """Match the token convention used by the multizone simulation outputs."""
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "unnamed"


def _interzone_heat_exchange_html_fragment(
    building_object: dict,
    hourly_results: pd.DataFrame,
 ) -> str:
    """Build an HTML fragment for annual/monthly internal-partition transfer.

    The two face-flux outputs of an internal partition can differ because its
    thermal mass may temporarily store energy.  The symmetric value
    ``(Q_to_B - Q_to_A) / 2`` is therefore used as the net A -> B transfer.
    """
    hourly = hourly_results.copy()
    hourly.index = pd.DatetimeIndex(hourly.index)
    dt_h = infer_timestep_hours(hourly.index, default=1.0)
    links = []

    for surface in building_object.get("building_surface", []):
        if str(surface.get("boundary", "")).upper() != "INTERNAL":
            continue
        zone_a = str(surface.get("zone", ""))
        zone_b = str(surface.get("adjacent_zone", ""))
        if not zone_a or not zone_b:
            continue

        token = _sanitize_result_column_token(surface.get("name", "internal_partition"))
        col_a = f"Q_opaque_inside_surface_{token}_to_{_sanitize_result_column_token(zone_a)}"
        col_b = f"Q_opaque_inside_surface_{token}_to_{_sanitize_result_column_token(zone_b)}"
        if col_a not in hourly.columns or col_b not in hourly.columns:
            continue

        q_to_a = pd.to_numeric(hourly[col_a], errors="coerce").fillna(0.0)
        q_to_b = pd.to_numeric(hourly[col_b], errors="coerce").fillna(0.0)
        q_a_to_b = (q_to_b - q_to_a) / 2.0  # +: A -> B, -: B -> A
        links.append(
            {
                "surface": str(surface.get("name", token)),
                "zone_a": zone_a,
                "zone_b": zone_b,
                "q_a_to_b": q_a_to_b,
                "energy_to_a_kwh": float((q_to_a * dt_h).sum() / 1000.0),
                "energy_to_b_kwh": float((q_to_b * dt_h).sum() / 1000.0),
                "net_a_to_b_kwh": float((q_a_to_b * dt_h).sum() / 1000.0),
            }
        )

    if not links:
        return "<p>No INTERNAL partitions with paired face-flux outputs were found.</p>"

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.48, 0.52],
        vertical_spacing=0.14,
        specs=[[{"type": "domain"}], [{"type": "xy"}]],
        subplot_titles=("Annual net heat transfer", "Monthly net heat transfer"),
    )
    node_labels = []
    node_index = {}
    sources, targets, values, labels = [], [], [], []
    for link in links:
        energy = link["net_a_to_b_kwh"]
        source = link["zone_a"] if energy >= 0 else link["zone_b"]
        target = link["zone_b"] if energy >= 0 else link["zone_a"]
        for zone in (source, target):
            if zone not in node_index:
                node_index[zone] = len(node_labels)
                node_labels.append(zone)
        sources.append(node_index[source])
        targets.append(node_index[target])
        values.append(abs(energy))
        labels.append(f"{link['surface']}: {abs(energy):.1f} kWh")

    fig.add_trace(
        go.Sankey(
            node={"label": node_labels, "pad": 24, "thickness": 22},
            link={"source": sources, "target": targets, "value": values, "label": labels},
            valueformat=".1f",
            valuesuffix=" kWh",
        ),
        row=1,
        col=1,
    )

    for link in links:
        monthly = (link["q_a_to_b"] * dt_h / 1000.0).resample("ME").sum()
        fig.add_trace(
            go.Bar(
                x=monthly.index.strftime("%Y-%m"),
                y=monthly.values,
                name=f"{link['surface']} ({link['zone_a']} → {link['zone_b']})",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Heat exchanged between thermal zones",
        template="plotly_white",
        height=900,
        barmode="relative",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_yaxes(
        title_text="Net energy [kWh]; positive = first zone → second zone",
        row=2,
        col=1,
        zeroline=True,
        zerolinewidth=1,
    )
    fig.update_xaxes(title_text="Month", row=2, col=1)

    table = pd.DataFrame(
        [
            {
                "partition": link["surface"],
                "zone A": link["zone_a"],
                "zone B": link["zone_b"],
                "net A → B [kWh]": link["net_a_to_b_kwh"],
                "into A face [kWh]": link["energy_to_a_kwh"],
                "into B face [kWh]": link["energy_to_b_kwh"],
            }
            for link in links
        ]
    )
    note = (
        "<p>Positive monthly values mean heat flows from zone A to zone B; negative values mean the reverse. "
        "The annual Sankey shows the net direction. The two face values can differ because the partition itself "
        "has thermal mass.</p>"
    )
    return (
        fig.to_html(include_plotlyjs="cdn", full_html=False)
        + note
        + table.to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}")
    )




# Writing a JSON export of the building dictionary with computed U-values and occupants expressed as W/person.
building_object = {
  "building": {
    "name": "TwoZoneBuilding",
"model_source": "idf",
"opaque_surface_area_mode": "net",
"latitude": 41.8,
"longitude": 12.58,
"net_floor_area": 120.0,
"exposed_perimeter": 44.0,
"wall_thickness": 0.43,
"building_type_class": "Residential_apartment",
"construction_class": "class_e"
  },
"building_parameters": {
    "temperature_setpoints": {
      "heating_setpoint": 20.0,
"cooling_setpoint": 26.0,
"heating_setback": -100,
"cooling_setback": 100.0
    },
"system_capacities": {
      "heating_capacity": 1000000000.0,
"cooling_capacity": 1000000000.0
    },
"ventilation": {
      # Single source of truth for ventilation/infiltration model.
      # Valid types: occupancy | temp_wind | custom |
      #              eplus_infiltration_ext_area | sherman_grimsrud_like
      "ventilation_type": "eplus_infiltration_ext_area",
#         # Keep disabled by default: the reference IDF uses A = 0.0.
      "infiltration_coeff_constant_auto_by_latitude": False,
#         # Occupancy model [L/(s m2)].
      "flow_rate_per_person": 0.3,
#         # Custom fixed H_ve [W/K].
      "custom_heat_transfer_coefficient_ventilation": 0.5,
#         # Temp-wind ISO 16798-7 coefficients.
      "temp_wind_c_wnd": 0.001,
    "temp_wind_c_st": 0.0035,
    "temp_wind_rho_a_ref": 1.204,
    "temp_wind_opening_ratio": 0.9,
#         # EnergyPlus DesignFlowRate(Flow/ExteriorArea)-like infiltration.
      "infiltration_flow_per_exterior_area_m3_s_m2": 3.0e-4,
#         # "infiltration_coeff_constant": 0.0,
      "infiltration_coeff_temperature": 0.0,
"infiltration_coeff_velocity": 0.224,
"infiltration_coeff_velocity_squared": 0.0,
"infiltration_include_transparent_area": True,
#         # "outdoors_only" = only OUTDOORS surfaces.
#         # "energyplus_like" = also include ground-like boundaries in ext_area.
"infiltration_exterior_area_mode": "energyplus_like",
#         # Optional reduction factor applied to EPW wind speed before using
#         # it in the EnergyPlus-like infiltration coefficients.
"infiltration_wind_reduction_factor": 1.0,
#         # Sherman-Grimsrud-like parameters.
      "infiltration_effective_leakage_area_m2": 0.5,
"infiltration_stack_coefficient": 0.0,
"infiltration_wind_coefficient": 0.0,
#         # Optional multiplier F(t).
      "infiltration_schedule_multiplier": 1.0
    },
"ventilation_profile": {
      "weekday": [1.0] * 24,
"weekend": [1.0] * 24
    },
"simulation_options": {
    "internal_convection_model": "tarp",
#         # "table" | "tarp"
    "external_convection_model": "doe2",
#         # "table" | "doe2" | "mowitt" | "blast" | "simplecombined"
    "external_convection_h_min": 2.0,
#         # W/(m2 K), lower bound for dynamic h_ce
    "external_radiation_model": "dynamic",
#         # "table" | "dynamic"
    "sky_temperature_model": "epw_ir",
#         # "berdahl_fromberg" | "swinbank" | "epw_ir"
    "external_emissivity_default": 0.9,
#         # Optional fixed monthly ground temperatures from building_object:
    "ground_temperature_model": "monthly",
    # Backward-compatible alias: "energyplus"
    "ground_temperature_monthly": [9.867, 9.199, 9.773, 11.434, 13.737, 16.066, 17.795, 18.462, 17.889, 16.228, 13.924, 11.596]
},
"internal_gains": [ 
    {
        "name": "occupants",
        "full_load": 6.0,
        "w_per_person": 120.0,
        "weekday": [ 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.5333333333333333, 0.8, 0.4333333333333333, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.3333333333333333, 0.13333333333333333, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.4333333333333333, 0.6666666666666666, 0.8333333333333333, 0.9333333333333333, 1.0, 0.8666666666666667, 0.3333333333333333 ],
        "weekend": [ 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.4666666666666667, 0.6, 0.6333333333333333, 0.6, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6, 0.6, 0.6, 0.6666666666666666, 0.7, 0.7333333333333333, 0.8333333333333333, 0.9333333333333333, 1.0, 0.8666666666666667, 0.3333333333333333 ] }, 
        { "name": "appliances",
        "full_load": 0.0,
        "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] }, 
        { "name": "lighting",
        "full_load": 3.3333,
        "weekday": [ 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.475, 0.8, 0.45, 0.07500000000000001, 0.07500000000000001, 0.15000000000000002, 0.375, 0.15000000000000002, 0.07500000000000001, 0.07500000000000001, 0.15000000000000002, 0.475, 0.7250000000000001, 0.875, 0.95, 1.0, 0.8500000000000001, 0.25 ],
        "weekend": [ 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.4, 0.575, 0.675, 0.675, 0.75, 0.75, 0.75, 0.675, 0.675, 0.675, 0.75, 0.775, 0.8, 0.875, 0.95, 1.0, 0.8500000000000001, 0.25 ] } ],
    },
    "zones": [ { 
        "name": "Z1",
        "net_floor_area": 80.0,
        "building_type_class": "Residential_apartment",
        "heating_setpoint": 20.0,
        "cooling_setpoint": 26.0,
        "cooling_setback": 100.0,
        "summer_night_purge": { "enabled": False,
        "months": [6, 8],
        "hours": [22, 7],
        "delta_t_min": 0.1,
        "boost_factor": 7.0 },
        "internal_gains": [ { "name": "occupants",
        "full_load": 6.0,
        "w_per_person": 120.0 }, { "name": "appliances",
        "full_load": 0.0 }, { "name": "lighting",
        "full_load": 3.75 } ],
        "occupants_profile": { "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 0.5, 0.1, 0.1, 0.2, 0.5, 0.2, 0.1, 0.1, 0.2, 0.6, 0.9, 1.0, 1.0, 1.0, 0.8, 0.0 ],
        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.0 ] },
        "appliances_profile": { "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] },
        "lighting_profile": { "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 0.5, 0.1, 0.1, 0.2, 0.5, 0.2, 0.1, 0.1, 0.2, 0.6, 0.9, 1.0, 1.0, 1.0, 0.8, 0.0 ],
        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.0 ] },
        "heating_profile": { "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 ],
        "weekend": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0  ] },
        "cooling_profile": { "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0 ],
        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0 ] },
        "ventilation_profile": { "weekday": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
        "weekend": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ] } }, { "name": "Z2",
        "net_floor_area": 40.0,
        "building_type_class": "Residential_apartment",
        "heating_setpoint": 18.0,
        "cooling_setpoint": 26.0,
        "cooling_setback": 100.0,
        "summer_night_purge": { 
            "enabled": False,
            "months": [6, 8],
            "hours": [22, 7],
            "delta_t_min": 0.1,
            "boost_factor": 7.0 },
            "internal_gains": [ 
                {"name": "occupants","full_load": 6.0,"w_per_person": 120.0 }, 
                { "name": "appliances","full_load": 0.0 }, 
                { "name": "lighting","full_load": 2.5 } ],
                "occupants_profile": { 
                    "weekday": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0 ],
                    "weekend": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0 ] 
                    },
                    "appliances_profile": { 
                        "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] 
                    },
                    "lighting_profile": { 
                        "weekday": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0 ],
                        "weekend": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0 ] 
                    },
                    "heating_profile": { 
                        "weekday": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 ],
                        "weekend": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 ] 
                    },
                    "cooling_profile": { 
                        "weekday": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0 ],
                        "weekend": [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0 ] 
                    },
                    "ventilation_profile": { 
                        "weekday": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                        "weekend": [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ] 
                    } 
                } 
            ],
            "building_surface": [ { "name": "ZoneA_Floor",
            "type": "opaque",
            "boundary": "GROUND",
"zone": "Z1",
"area": 80.0,
"u_value": 1.73,
"thermal_capacity": 429600.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 0.0,
"tilt": 180.0 },
"ISO52016_orientation_string": "HF",
"sky_view_factor": 0.0 }, { "name": "ZoneA_Roof",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z1",
"area": 80.0,
"u_value": 0.307,
"thermal_capacity": 398520.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 0.0,
"tilt": 0.0 },
"ISO52016_orientation_string": "HR",
"sky_view_factor": 1.0 }, { "name": "ZoneA_Wall_S",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z1",
"area": 30.0,
"u_value": 0.2891,
"thermal_capacity": 248700.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 180.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "SV",
"sky_view_factor": 0.5 }, { "name": "ZoneA_Wall_E",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z1",
"area": 24.0,
"u_value": 0.2891,
"thermal_capacity": 248700.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 90.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "EV",
"sky_view_factor": 0.5 }, { "name": "ZoneA_Wall_W",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z1",
"area": 24.0,
"u_value": 0.2891,
"thermal_capacity": 248700.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 270.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "WV",
"sky_view_factor": 0.5 }, { "name": "ZoneA_Wall_N_Int",
"type": "opaque",
"boundary": "INTERNAL",
"zone": "Z1",
"area": 30.0,
"u_value": 1.8421,
"thermal_capacity": 136800.0,
"solar_absorptance": 0.0,
"orientation": { "azimuth": 0.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "NV",
"sky_view_factor": 0.0,
"adjacent_zone": "Z1",
"convective_heat_transfer_coefficient_internal": 2.5,
"radiative_heat_transfer_coefficient_internal": 5.13,
"convective_heat_transfer_coefficient_external": 2.5,
"radiative_heat_transfer_coefficient_external": 5.13 
}, 

{ "name": "ZoneB_Floor",
"type": "opaque",
"boundary": "GROUND",
"zone": "Z2",
"area": 40.0,
"u_value": 1.73,
"thermal_capacity": 429600.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 0.0,
"tilt": 180.0 },
"ISO52016_orientation_string": "HF",
"sky_view_factor": 0.0 }, { "name": "ZoneB_Roof",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z2",
"area": 40.0,
"u_value": 0.307,
"thermal_capacity": 398520.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 0.0,
"tilt": 0.0 },
"ISO52016_orientation_string": "HR",
"sky_view_factor": 1.0 }, { "name": "ZoneB_Wall_N",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z2",
"area": 30.0,
"u_value": 0.2891,
"thermal_capacity": 248700.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 0.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "NV",
"sky_view_factor": 0.5 }, { "name": "ZoneB_Wall_E",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z2",
"area": 12.0,
"u_value": 0.2891,
"thermal_capacity": 248700.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 90.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "EV",
"sky_view_factor": 0.5 }, { "name": "ZoneB_Wall_W",
"type": "opaque",
"boundary": "OUTDOORS",
"zone": "Z2",
"area": 12.0,
"u_value": 0.2891,
"thermal_capacity": 248700.0,
"solar_absorptance": 0.6,
"orientation": { "azimuth": 270.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "WV",
"sky_view_factor": 0.5 }, { "name": "ZoneA_Win_S_L",
"type": "transparent",
"boundary": "OUTDOORS",
"zone": "Z1",
"area": 4.5,
"u_value": 2.0,
"g_value": 0.6,
"orientation": { "azimuth": 180.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "SV",
"sky_view_factor": 0.5,
"frame_area_fraction": 0,
"height": 1.5,
"width": 3,
"parapet": 0.9,
"shading": False,
         }, { "name": "ZoneA_Win_S_R",
"type": "transparent",
"boundary": "OUTDOORS",
"zone": "Z1",
"area": 4.5,
"u_value": 2.0,
"g_value": 0.6,
"orientation": { "azimuth": 180.0,
"tilt": 90.0 },
"ISO52016_orientation_string": "SV",
"sky_view_factor": 0.5,
"frame_area_fraction": 0,
"height": 1.5,
"width": 3,
"parapet": 0.9,
"shading": False,
         }, { "name": "ZoneB_Win_N",
"type": "transparent",
"boundary": "OUTDOORS",
"zone": "Z2",
"area": 6.0,
"u_value": 2.0,
"g_value": 0.6,
"orientation": { "azimuth": 0.0,
"tilt": 90.0 },
"frame_area_fraction": 0.0,
"height": 1.5,
"width": 4,
"parapet": 0.9,
"shading": False,
"ISO52016_orientation_string": "NV",
"sky_view_factor": 0.5 } ]
}


SUMMER_NIGHT_PURGE_PRESETS = {
    # Disabled (baseline infiltration only).
    "off": {
        "enabled": False,
        "months": [6, 8],
        "hours": [22, 7],
        "delta_t_min": 0.5,
        "boost_factor": 1.0,
    },
    # Mild purge: little impact on comfort/heating shoulder months.
    "conservative": {
        "enabled": True,
        "months": [6, 9],
        "hours": [23, 6],
        "delta_t_min": 1.0,
        "boost_factor": 1.8,
    },
    # Recommended default for first calibration cycle.
    "balanced": {
        "enabled": True,
        "months": [6, 8],
        "hours": [22, 7],
        "delta_t_min": 0.5,
        "boost_factor": 3.0,
    },
    # Calibrated on current EP benchmark (best among tested presets/overrides).
    "calibrated": {
        "enabled": True,
        "months": [5, 10],
        "hours": [20, 9],
        "delta_t_min": 0.1,
        "boost_factor": 6.0,
    },
    # Single robust preset across cities (balanced yearly compromise).
    "global_robust": {
        "enabled": True,
        "months": [6, 9],
        "hours": [20, 9],
        "delta_t_min": 0.2,
        "boost_factor": 4.5,
    },
    # Strong night flushing.
    "aggressive": {
        "enabled": True,
        "months": [5, 9],
        "hours": [0, 0],
        "delta_t_min": 0.1,
        "boost_factor": 7.5,
    },
}

ENERGYPLUS_REFERENCE_GROUND_TEMPERATURES_ATHENS = [
    9.867,
    9.199,
    9.773,
    11.434,
    13.737,
    16.066,
    17.795,
    18.462,
    17.889,
    16.228,
    13.924,
    11.596,
]


def _resolve_summer_night_purge_config(
    preset: str = "calibrated",
    *,
    enabled: bool | None = None,
    month_start: int | None = None,
    month_end: int | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
    delta_t_min: float | None = None,
    boost_factor: float | None = None,
) -> dict:
    cfg = copy.deepcopy(SUMMER_NIGHT_PURGE_PRESETS.get(preset, SUMMER_NIGHT_PURGE_PRESETS["calibrated"]))
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if month_start is not None:
        cfg["months"][0] = max(1, min(12, int(month_start)))
    if month_end is not None:
        cfg["months"][1] = max(1, min(12, int(month_end)))
    if hour_start is not None:
        cfg["hours"][0] = int(hour_start) % 24
    if hour_end is not None:
        cfg["hours"][1] = int(hour_end) % 24
    if delta_t_min is not None:
        cfg["delta_t_min"] = max(0.0, float(delta_t_min))
    if boost_factor is not None:
        cfg["boost_factor"] = max(1.0, float(boost_factor))
    return cfg


def _read_epw_latitude_from_header(weather_path: str) -> float:
    """Reads latitude from EPW LOCATION header line."""
    with open(weather_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError("Empty EPW file.")

    parts = first_line.split(",")
    # EPW LOCATION fields:
    # 0 LOCATION, 1 City, 2 State, 3 Country, 4 Source, 5 WMO, 6 Latitude, ...
    if len(parts) < 7:
        raise ValueError("Invalid EPW header; latitude field not found.")
    return float(parts[6])


def _auto_purge_preset_from_latitude(lat_abs: float) -> str:
    """Select night purge preset from absolute latitude."""
    lat_abs = abs(float(lat_abs))
    if lat_abs < 43.0:
        return "aggressive"    # Mediterranean climates (Athens, Madrid, central Italy)
    if lat_abs < 50.0:
        return "balanced"      # temperate central Europe
    if lat_abs <= 60.0:
        return "calibrated"    # northern Europe
    return "balanced"


def _epw_latitude_abs(weather_path: str) -> float:
    """Returns absolute latitude from EPW metadata/header."""
    lat_abs = None
    # Preferred reader.
    try:
        from pvlib.iotools import epw as pvepw

        _, meta = pvepw.read_epw(weather_path)
        lat_abs = abs(float(meta["latitude"]))
    except Exception:
        lat_abs = abs(float(_read_epw_latitude_from_header(weather_path)))
    return float(lat_abs)


def _auto_infiltration_coeff_constant_from_epw(weather_path: str) -> float:
    """Select infiltration A coefficient from EPW latitude bands."""
    lat_abs = _epw_latitude_abs(weather_path)
    if lat_abs > 37.0 and lat_abs <= 40.0 :
        return 0.30  # Mediterranean hot
    if lat_abs > 40.0 and lat_abs <= 42.0:
        return 0.30  # Mediterranean
    if lat_abs > 42.0 and lat_abs <= 50.0:
        return 0.20 # Central Europe
    return 0.00      # Northern Europe (Berlin-like)


def apply_summer_night_purge_to_zones(bui: dict, summer_night_purge: dict | None = None) -> None:
    """Apply the same summer night purge config to all zones and building params."""
    purge_cfg = copy.deepcopy(
        summer_night_purge
        if isinstance(summer_night_purge, dict)
        else SUMMER_NIGHT_PURGE_PRESETS["calibrated"]
    )
    bui.setdefault("building_parameters", {})["summer_night_purge"] = copy.deepcopy(purge_cfg)
    for zone in bui.get("zones", []):
        zone["summer_night_purge"] = copy.deepcopy(purge_cfg)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multizone free-floating example with optional quick diagnostic settings."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter settings for faster diagnostic runs.",
    )
    parser.add_argument(
        "--warmup-hours",
        type=int,
        default=None,
        help="Override warmup hours (default: 744, or 168 with --quick).",
    )
    parser.add_argument(
        "--weather-path",
        default=None,
        help="Path to EPW weather file (default: examples/2020_Athens.epw).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <project_root>/result_test).",
    )
    parser.add_argument(
        "--no-solar",
        action="store_true",
        help="Disable solar gains in comparison step.",
    )
    parser.add_argument(
        "--ground-temperature-model",
        choices=["iso13370", "monthly", "energyplus"],
        default="iso13370",
        help=(
            "Ground temperature model for slab-on-ground exchange "
            "(default: iso13370). Use 'monthly' to read the 12 monthly "
            "ground temperatures from building_object or CLI."
        ),
    )
    parser.add_argument(
        "--ground-temperature-monthly",
        type=float,
        nargs=12,
        default=None,
        metavar=("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        help=(
            "Override the 12 monthly ground temperatures [degC] used when "
            "--ground-temperature-model energyplus is selected."
        ),
    )
    parser.add_argument(
        "--infiltration-ext-area-mode",
        choices=["outdoors_only", "energyplus_like"],
        default=None,
        help=(
            "Reference area mode for eplus_infiltration_ext_area. "
            "'outdoors_only' uses only OUTDOORS surfaces; "
            "'energyplus_like' also counts ground-like boundaries."
        ),
    )
    parser.add_argument(
        "--infiltration-wind-reduction-factor",
        type=float,
        default=None,
        help=(
            "Optional multiplier applied to wind speed in eplus_infiltration_ext_area "
            "(default from building_object: 1.0). Values < 1 reduce infiltration."
        ),
    )
    parser.add_argument(
        "--night-purge-preset",
        choices=sorted(list(SUMMER_NIGHT_PURGE_PRESETS.keys()) + ["auto_geo"]),
        default="off",
        help=(
            "Summer night purge preset (default: off). "
            "Use 'auto_geo' to select preset from EPW latitude."
        ),
    )
    parser.add_argument(
        "--night-purge-disable",
        action="store_true",
        help="Disable summer night purge regardless of preset.",
    )
    parser.add_argument(
        "--night-purge-month-start",
        type=int,
        default=None,
        help="Override purge start month [1..12].",
    )
    parser.add_argument(
        "--night-purge-month-end",
        type=int,
        default=None,
        help="Override purge end month [1..12].",
    )
    parser.add_argument(
        "--night-purge-hour-start",
        type=int,
        default=None,
        help="Override purge start hour [0..23].",
    )
    parser.add_argument(
        "--night-purge-hour-end",
        type=int,
        default=None,
        help="Override purge end hour [0..23].",
    )
    parser.add_argument(
        "--night-purge-delta-t-min",
        type=float,
        default=None,
        help="Override minimum (T_in - T_out) [C] for purge activation.",
    )
    parser.add_argument(
        "--night-purge-boost-factor",
        type=float,
        default=None,
        help="Override H_ve multiplier when purge is active (>=1).",
    )
    parser.add_argument(
        "--progress-log-hours",
        type=int,
        default=720,
        help="Emit a progress log every N simulated hours during the annual V1 run (default: 720). Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-sankey",
        action="store_true",
        help="Skip Sankey post-processing and HTML export.",
    )
    parser.add_argument(
        "--hybrid-max-iterations",
        type=int,
        default=6,
        help="Maximum number of coupling iterations for the hybrid multizone method.",
    )
    parser.add_argument(
        "--hybrid-tolerance-w",
        type=float,
        default=10.0,
        help="Convergence tolerance [W] for the hybrid multizone coupling iteration.",
    )
    parser.add_argument(
        "--hybrid-relaxation",
        type=float,
        default=0.6,
        help="Relaxation factor [0..1] for the hybrid multizone coupling iteration.",
    )
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip the iterative hybrid multizone run and related report exports.",
    )
    return parser


def _log_progress(run_start_t: float, message: str) -> None:
    elapsed_s = time.perf_counter() - run_start_t
    print(f"[{elapsed_s:8.1f}s] {message}", flush=True)



if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    run_start_t = time.perf_counter()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(project_root, "result_test")
    os.makedirs(output_dir, exist_ok=True)

    default_weather_path = os.path.join(os.path.dirname(__file__), "2020_Athens.epw")
    weather_path = os.path.abspath(args.weather_path) if args.weather_path else default_weather_path
    if not os.path.exists(weather_path):
        raise FileNotFoundError(
            f"Weather file not found: {weather_path}. Use --weather-path <file.epw>."
        )

    building_object, opaque_area_adjustments = _normalize_idf_gross_opaque_areas(building_object)
    sim_opts = building_object.setdefault("building_parameters", {}).setdefault("simulation_options", {})
    ground_temperature_monthly_source = "iso13370"
    selected_ground_temperature_model = str(args.ground_temperature_model).strip().lower().replace("-", "_")
    if selected_ground_temperature_model == "energyplus":
        selected_ground_temperature_model = "monthly"
    sim_opts["ground_temperature_model"] = selected_ground_temperature_model
    if selected_ground_temperature_model == "monthly":
        existing_ground_monthly = sim_opts.get("ground_temperature_monthly")
        if args.ground_temperature_monthly is not None:
            sim_opts["ground_temperature_monthly"] = [float(v) for v in args.ground_temperature_monthly]
            ground_temperature_monthly_source = "cli"
        elif existing_ground_monthly is not None:
            ground_temperature_monthly_source = "building_object"
        else:
            sim_opts["ground_temperature_monthly"] = list(ENERGYPLUS_REFERENCE_GROUND_TEMPERATURES_ATHENS)
            ground_temperature_monthly_source = "energyplus_reference_athens"

    warmup_hours = args.warmup_hours if args.warmup_hours is not None else (168 if args.quick else 744)
    include_solar = not args.no_solar
    selected_purge_preset = str(args.night_purge_preset)
    auto_geo_lat = None
    if selected_purge_preset == "auto_geo":
        auto_geo_lat = _epw_latitude_abs(weather_path)
        selected_purge_preset = _auto_purge_preset_from_latitude(auto_geo_lat)

    night_purge_cfg = _resolve_summer_night_purge_config(
        preset=selected_purge_preset,
        enabled=False if args.night_purge_disable else None,
        month_start=args.night_purge_month_start,
        month_end=args.night_purge_month_end,
        hour_start=args.night_purge_hour_start,
        hour_end=args.night_purge_hour_end,
        delta_t_min=args.night_purge_delta_t_min,
        boost_factor=args.night_purge_boost_factor,
    )

    _log_progress(run_start_t, f"Output dir: {output_dir}")
    _log_progress(run_start_t, f"Weather EPW: {weather_path}")
    if opaque_area_adjustments:
        _log_progress(
            run_start_t,
            "Opaque area normalization: applied gross->net conversion for IDF-derived outdoor opaque surfaces.",
        )
        for adj in opaque_area_adjustments:
            _log_progress(
                run_start_t,
                "Opaque net area: "
                f"zone={adj['zone']}, orientation={adj['orientation']}, "
                f"gross={adj['gross_opaque_area']:.2f} m2, "
                f"windows={adj['window_area']:.2f} m2, "
                f"net={adj['net_opaque_area']:.2f} m2, "
                f"surfaces={', '.join(adj['surface_names'])}",
            )
    elif str(building_object.get("building", {}).get("model_source", "")).strip().lower() == "idf":
        _log_progress(
            run_start_t,
            "Opaque area normalization: model_source=idf but no gross opaque surfaces required adjustment.",
        )
    _log_progress(
        run_start_t,
        "Run settings: "
        f"quick={args.quick}, warmup_hours={warmup_hours}, "
        f"include_solar={include_solar}",
    )
    if selected_ground_temperature_model == "monthly":
        _log_progress(
            run_start_t,
            "Ground temperature model: "
            f"monthly, monthly_source={ground_temperature_monthly_source}, "
            f"monthly_values={sim_opts.get('ground_temperature_monthly')}",
        )
    else:
        _log_progress(run_start_t, "Ground temperature model: iso13370")
    if auto_geo_lat is not None:
        _log_progress(
            run_start_t,
            "Auto-geo night purge selection: "
            f"lat_abs={auto_geo_lat:.3f} -> preset={selected_purge_preset}",
        )
    _log_progress(
        run_start_t,
        "Summer night purge: "
        f"preset={selected_purge_preset}, "
        f"enabled={night_purge_cfg.get('enabled')}, "
        f"months={night_purge_cfg.get('months')}, "
        f"hours={night_purge_cfg.get('hours')}, "
        f"delta_t_min={night_purge_cfg.get('delta_t_min')}, "
        f"boost_factor={night_purge_cfg.get('boost_factor')}",
    )

    _log_progress(run_start_t, "Applying summer night purge config to all zones...")
    apply_summer_night_purge_to_zones(
        building_object,
        summer_night_purge=night_purge_cfg,
    )
    vent_cfg = (
        building_object.get("building_parameters", {}).get("ventilation", {})
        if isinstance(building_object, dict)
        else {}
    )
    if args.infiltration_ext_area_mode is not None:
        vent_cfg["infiltration_exterior_area_mode"] = str(args.infiltration_ext_area_mode)
    if args.infiltration_wind_reduction_factor is not None:
        vent_cfg["infiltration_wind_reduction_factor"] = float(args.infiltration_wind_reduction_factor)
    lat_abs = _epw_latitude_abs(weather_path)
    a_effective = vent_cfg.get("infiltration_coeff_constant", 0.0)
    if a_effective is None:
        a_effective = 0.0
    if bool(vent_cfg.get("infiltration_coeff_constant_auto_by_latitude", False)):
        a_effective = _auto_infiltration_coeff_constant_from_epw(weather_path)
        vent_cfg["infiltration_coeff_constant"] = a_effective
    _log_progress(
        run_start_t,
        "Infiltration latitude setup: "
        f"lat_abs={lat_abs:.2f}, "
        f"A_auto_enabled={bool(vent_cfg.get('infiltration_coeff_constant_auto_by_latitude', False))}, "
        f"A_effective={a_effective}",
    )
    _log_progress(
        run_start_t,
        "Ventilation from building_object: "
        f"type={vent_cfg.get('ventilation_type')}, "
        f"flow_rate_per_person={vent_cfg.get('flow_rate_per_person')}, "
        f"q_inf_ext_area={vent_cfg.get('infiltration_flow_per_exterior_area_m3_s_m2')}, "
        f"A={a_effective}, "
        f"B={vent_cfg.get('infiltration_coeff_temperature')}, "
        f"C={vent_cfg.get('infiltration_coeff_velocity')}, "
        f"D={vent_cfg.get('infiltration_coeff_velocity_squared')}, "
        f"ext_area_mode={vent_cfg.get('infiltration_exterior_area_mode', 'outdoors_only')}, "
        f"wind_reduction_factor={vent_cfg.get('infiltration_wind_reduction_factor', 1.0)}, "
        f"ela={vent_cfg.get('infiltration_effective_leakage_area_m2')}, "
        f"cs={vent_cfg.get('infiltration_stack_coefficient')}, "
        f"cw={vent_cfg.get('infiltration_wind_coefficient')}, "
        f"f_schedule={vent_cfg.get('infiltration_schedule_multiplier')}",
    )
    sim_opts = (
        building_object.get("building_parameters", {}).get("simulation_options", {})
        if isinstance(building_object, dict)
        else {}
    )
    _log_progress(
        run_start_t,
        "Simulation options: "
        f"internal_convection_model={sim_opts.get('internal_convection_model', 'table')}, "
        f"external_convection_model={sim_opts.get('external_convection_model', 'table')}, "
        f"external_convection_h_min={sim_opts.get('external_convection_h_min', 2.0)}, "
        f"external_radiation_model={sim_opts.get('external_radiation_model', 'table')}, "
        f"sky_temperature_model={sim_opts.get('sky_temperature_model', 'berdahl_fromberg')}, "
        f"external_emissivity_default={sim_opts.get('external_emissivity_default', 0.9)}, "
        f"ground_temperature_model={sim_opts.get('ground_temperature_model', 'iso13370')}",
    )
    _log_progress(run_start_t, "Starting multizone simulation: V1 fully integrated...")
    v1_start_t = time.perf_counter()
    res_v1, annual_v1 = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
        building_object=building_object,
        path_weather_file=weather_path,
        weather_source="epw",
        include_solar=include_solar,
        warmup_hours=warmup_hours,
        hvac_control_variable="air",
        progress_log_every_steps=max(0, int(args.progress_log_hours)),
        progress_logger=lambda message: _log_progress(run_start_t, message),
    )
    t_v1_s = float(time.perf_counter() - v1_start_t)
    t_total_s = float(time.perf_counter() - run_start_t)
    _log_progress(run_start_t, f"Full integrated simulation completed in {t_v1_s:.1f}s.")
    _log_progress(
        run_start_t,
        "Simulation timing: "
        f"fully_integrated={t_v1_s:.1f}s, "
        f"total_run_so_far={t_total_s:.1f}s",
    )

    res_v2 = None
    annual_v2 = None
    hybrid_iterations = pd.DataFrame()
    t_v2_s = 0.0
    if args.skip_hybrid:
        _log_progress(run_start_t, "Skipping hybrid iterative calculation (--skip-hybrid).")
    else:
        _log_progress(run_start_t, "Starting multizone simulation: V2 hybrid iterative...")
        v2_start_t = time.perf_counter()
        res_v2, annual_v2, hybrid_iterations = ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(
            building_object=building_object,
            path_weather_file=weather_path,
            weather_source="epw",
            include_solar=include_solar,
            max_iterations=int(args.hybrid_max_iterations),
            tolerance_w=float(args.hybrid_tolerance_w),
            relaxation=float(args.hybrid_relaxation),
            warmup_hours=warmup_hours,
            hvac_control_variable="air",
        )
        t_v2_s = float(time.perf_counter() - v2_start_t)
        if not hybrid_iterations.empty:
            last_iter = hybrid_iterations.iloc[-1]
            _log_progress(
                run_start_t,
                "Hybrid iterative simulation completed in "
                f"{t_v2_s:.1f}s, iterations={int(last_iter.get('iteration', len(hybrid_iterations)))}, "
                f"max_abs_delta_coupling_W={float(last_iter.get('max_abs_delta_coupling_W', np.nan)):.2f}",
            )
        else:
            _log_progress(run_start_t, f"Hybrid iterative simulation completed in {t_v2_s:.1f}s.")

    zone_names = [z["name"] for z in building_object["zones"]]
    for z in zone_names:
        c_act = f"night_purge_active_{z}"
        c_fac = f"night_purge_factor_{z}"
        if c_act not in res_v1.columns:
            continue
        act_hours = int(pd.to_numeric(res_v1[c_act], errors="coerce").fillna(0).clip(lower=0).sum())
        if c_fac in res_v1.columns:
            fac_s = pd.to_numeric(res_v1[c_fac], errors="coerce").fillna(1.0)
            fac_active = fac_s[pd.to_numeric(res_v1[c_act], errors="coerce").fillna(0) > 0]
            mean_fac = float(fac_active.mean()) if len(fac_active) > 0 else 1.0
        else:
            mean_fac = 1.0
        _log_progress(
            run_start_t,
            f"Night purge diagnostics {z}: active_hours={act_hours}, mean_factor_active={mean_fac:.2f}",
        )

    _log_progress(run_start_t, "Computing hourly ground temperatures and ground exchanges...")
    ground_df = _compute_ground_temperature_and_exchanges(
        building_object=building_object,
        hourly_results=res_v1,
        weather_path=weather_path,
        weather_source="epw",
    )
    if not ground_df.empty:
        missing_ground_cols = [col for col in ground_df.columns if col not in res_v1.columns]
        if missing_ground_cols:
            res_v1 = pd.concat([res_v1, ground_df.loc[:, missing_ground_cols]], axis=1)
        t_ground_valid = _get_first_matching_series(res_v1, "T_ground_virtual")
        if t_ground_valid is not None:
            t_ground_valid = t_ground_valid.dropna()
            if len(t_ground_valid) > 0:
                _log_progress(
                    run_start_t,
                    "Ground temperature range over calculation period: "
                    f"min={t_ground_valid.min():.2f} degC, "
                    f"max={t_ground_valid.max():.2f} degC",
                )

    # Save full outputs
    csv_v1_path = os.path.join(output_dir, "multizone_v1_hourly.csv")
    annual_v1_path = os.path.join(output_dir, "multizone_v1_annual.csv")
    timings_path = os.path.join(output_dir, "multizone_v1_timings_seconds.csv")
    csv_v2_path = os.path.join(output_dir, "multizone_v2_hybrid_hourly.csv")
    annual_v2_path = os.path.join(output_dir, "multizone_v2_hybrid_annual.csv")
    iterations_v2_path = os.path.join(output_dir, "multizone_v2_hybrid_iterations.csv")
    summary_v1_v2_path = os.path.join(output_dir, "multizone_v1_vs_v2_hybrid_summary.csv")

    _log_progress(run_start_t, "Saving CSV result files...")
    res_v1.to_csv(csv_v1_path)
    annual_v1.to_csv(annual_v1_path, index=False)
    timing_row = {
        "fully_integrated_v1_s": t_v1_s,
        "hybrid_v2_s": t_v2_s,
        "total_run_s": float(time.perf_counter() - run_start_t),
    }
    pd.DataFrame([timing_row]).to_csv(timings_path, index=False)
    annual_v1_v2_summary = pd.DataFrame()
    if res_v2 is not None and annual_v2 is not None:
        res_v2.to_csv(csv_v2_path)
        annual_v2.to_csv(annual_v2_path, index=False)
        hybrid_iterations.to_csv(iterations_v2_path, index=False)
        annual_v1_v2_summary = _build_v1_vs_hybrid_annual_summary(annual_v1, annual_v2)
        if not annual_v1_v2_summary.empty:
            annual_v1_v2_summary.to_csv(summary_v1_v2_path, index=False)

    # ---- Plot 1 (HTML): Operative temperatures (V1) ----
    _log_progress(run_start_t, "Building HTML chart: operative temperatures (V1)...")
    zone_names = [z["name"] for z in building_object["zones"]]
    temp_plot_path = os.path.join(output_dir, "multizone_v1_temperatures.html")
    _write_temperature_plot(
        df=res_v1,
        zone_names=zone_names,
        temp_col_candidates={z: [f"T_op_{z}", f"T_air_{z}"] for z in zone_names},
        out_html_path=temp_plot_path,
        title="Operative Temperature by Zone (Full Integrated V1)",
    )

    ground_plot_path = os.path.join(output_dir, "multizone_v1_ground_exchange.html")
    _log_progress(run_start_t, "Building HTML chart: ground temperature and exchanges...")
    _write_ground_exchange_plot(
        df=res_v1,
        zone_names=zone_names,
        out_html_path=ground_plot_path,
        title_temp="Virtual Ground Temperature",
        title_flux="Ground Heat Exchange by Zone",
    )

    # ---- Plot 2 (HTML): Monthly consumptions (V1) ----
    dt_h_common = infer_timestep_hours(res_v1.index, default=1.0)
    monthly_h_v1, monthly_c_v1 = _monthly_hvac_energy_by_zone(
        df=res_v1,
        zone_names=zone_names,
        hvac_col_candidates={z: [f"Q_HVAC_{z}"] for z in zone_names},
    )

    _log_progress(run_start_t, "Building HTML chart: monthly consumptions (V1)...")
    cons_compare_path = os.path.join(output_dir, "multizone_v1_monthly_consumptions.html")
    _write_monthly_hvac_plot(
        monthly_h=monthly_h_v1,
        monthly_c=monthly_c_v1,
        zone_names=zone_names,
        out_html_path=cons_compare_path,
        title_h="Monthly Heating Energy (V1)",
        title_c="Monthly Cooling Energy (V1)",
        trace_suffix="V1",
    )
    hybrid_temp_plot_path = None
    hybrid_cons_compare_path = None
    hybrid_report_path = None
    if res_v2 is not None and annual_v2 is not None:
        _log_progress(run_start_t, "Building HTML chart: operative temperatures (V2 hybrid)...")
        hybrid_temp_plot_path = os.path.join(output_dir, "multizone_v2_hybrid_temperatures.html")
        _write_temperature_plot(
            df=res_v2,
            zone_names=zone_names,
            temp_col_candidates={z: [f"T_op_core_{z}", f"T_air_core_{z}"] for z in zone_names},
            out_html_path=hybrid_temp_plot_path,
            title="Operative Temperature by Zone (Hybrid Iterative V2)",
        )

        monthly_h_v2, monthly_c_v2 = _monthly_hvac_energy_by_zone(
            df=res_v2,
            zone_names=zone_names,
            hvac_col_candidates={z: [f"Q_HC_hybrid_{z}"] for z in zone_names},
        )
        _log_progress(run_start_t, "Building HTML chart: monthly consumptions (V2 hybrid)...")
        hybrid_cons_compare_path = os.path.join(output_dir, "multizone_v2_hybrid_monthly_consumptions.html")
        _write_monthly_hvac_plot(
            monthly_h=monthly_h_v2,
            monthly_c=monthly_c_v2,
            zone_names=zone_names,
            out_html_path=hybrid_cons_compare_path,
            title_h="Monthly Heating Energy (Hybrid V2)",
            title_c="Monthly Cooling Energy (Hybrid V2)",
            trace_suffix="Hybrid V2",
        )

        _log_progress(run_start_t, "Building HTML report: V1 vs hybrid iterative V2...")
        hybrid_report_path = os.path.join(output_dir, "multizone_v1_vs_v2_hybrid_report.html")
        _write_v1_vs_hybrid_report(
            v1_df=res_v1,
            hybrid_df=res_v2,
            annual_summary_df=annual_v1_v2_summary,
            iterations_df=hybrid_iterations,
            zone_names=zone_names,
            out_html_path=hybrid_report_path,
        )

    print("Version 1 (multizone extended) head:")
    print(res_v1.head())
    print("\nAnnual results V1:")
    print(annual_v1)
    if annual_v2 is not None:
        print("\nAnnual results V2 hybrid:")
        print(annual_v2)
    print(f"\n[units] common timestep used for energy integration: {dt_h_common:.3f} h")

    print(f"\nSaved hourly V1: {csv_v1_path}")
    print(f"Saved annual V1: {annual_v1_path}")
    print(f"Saved timings: {timings_path}")
    if res_v2 is not None:
        print(f"Saved hourly V2 hybrid: {csv_v2_path}")
        print(f"Saved annual V2 hybrid: {annual_v2_path}")
        print(f"Saved hybrid iterations: {iterations_v2_path}")
        if not annual_v1_v2_summary.empty:
            print(f"Saved V1 vs V2 summary: {summary_v1_v2_path}")

    # ---- Sankey per zona termica (Version 1 multizone) ----
    if args.skip_sankey:
        _log_progress(run_start_t, "Skipping Sankey post-processing (--skip-sankey).")
    else:
        sankey_start_t = time.perf_counter()
        _log_progress(run_start_t, "Computing and saving Sankey by zone...")
        sankey_by_zone, sankey_summary = _compute_multizone_sankey_by_zone(
            building_object=building_object,
            hourly_results=res_v1,
            weather_path=weather_path,
            weather_source="epw",
        )
        sankey_summary_path = os.path.join(output_dir, "multizone_v1_sankey_by_zone_summary.csv")
        sankey_json_path = os.path.join(output_dir, "multizone_v1_sankey_by_zone.json")
        sankey_html_path = os.path.join(output_dir, "multizone_v1_sankey_by_zone.html")

        sankey_summary.to_csv(sankey_summary_path, index=False)
        with open(sankey_json_path, "w", encoding="utf-8") as f:
            json.dump(sankey_by_zone, f, indent=2)
        _write_multizone_sankey_html(
            sankey_by_zone=sankey_by_zone,
            summary_df=sankey_summary,
            out_html_path=sankey_html_path,
            building_object=building_object,
            hourly_results=res_v1,
        )
        _log_progress(
            run_start_t,
            f"Sankey post-processing completed in {time.perf_counter() - sankey_start_t:.1f}s.",
        )
        print(f"Saved multizone Sankey summary: {sankey_summary_path}")
        print(f"Saved multizone Sankey JSON: {sankey_json_path}")
        print(f"Saved multizone Sankey HTML: {sankey_html_path}")
    print(f"Saved HTML plot (temperatures V1): {temp_plot_path}")
    print(f"Saved HTML plot (ground exchange V1): {ground_plot_path}")
    print(f"Saved HTML plot (monthly consumptions V1): {cons_compare_path}")
    if hybrid_temp_plot_path is not None:
        print(f"Saved HTML plot (temperatures V2 hybrid): {hybrid_temp_plot_path}")
    if hybrid_cons_compare_path is not None:
        print(f"Saved HTML plot (monthly consumptions V2 hybrid): {hybrid_cons_compare_path}")
    if hybrid_report_path is not None:
        print(f"Saved HTML report (V1 vs V2 hybrid): {hybrid_report_path}")
    _log_progress(run_start_t, "Run completed.")
