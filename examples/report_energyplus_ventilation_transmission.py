#!/usr/bin/env python3
"""Generate HTML report for EnergyPlus ventilation/transmission and optional ISO comparison.

What this script does:
1) Reads EnergyPlus CSV (`two_zone_ideal_24_2_fixedout.csv`-like).
2) Builds ventilation hourly/annual tables and HTML charts.
3) Builds transmission-by-surface table/chart if conduction-energy outputs are available.
4) Optionally compares ventilation vs ISO multizone outputs (`H_ve_*`, `T_air_*`) when EPW is provided.
5) If required variables are missing, prints/saves the IDF `Output:Variable` snippet to add.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report HTML/tabella su ventilazione e trasmissione da output EnergyPlus, "
            "con confronto opzionale verso ISO multizone."
        )
    )
    parser.add_argument(
        "--energyplus-csv",
        default=str(SCRIPT_DIR / "two_zone_ideal_24_2_fixedout.csv"),
        help="Percorso CSV output EnergyPlus.",
    )
    parser.add_argument(
        "--iso-csv",
        default=str(PROJECT_ROOT / "result_test" / "multizone_v1_hourly.csv"),
        help="Percorso CSV hourly ISO multizone (opzionale).",
    )
    parser.add_argument(
        "--epw-path",
        default=str(SCRIPT_DIR / "2020_Milan.epw"),
        help="Percorso EPW per stimare i carichi di ventilazione (se non presenti direttamente in CSV).",
    )
    parser.add_argument(
        "--idf-path",
        default="",
        help="Percorso IDF (usato solo come riferimento nel report).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Anno base per parsing Date/Time di EnergyPlus.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "result_test"),
        help="Directory output report.",
    )
    parser.add_argument(
        "--out-prefix",
        default="energyplus_ventilation_transmission_report",
        help="Prefisso file output.",
    )
    return parser.parse_args()


def parse_energyplus_datetime(series: pd.Series, *, year: int, shift_interval_end: bool = True) -> pd.DatetimeIndex:
    raw = series.astype(str).str.strip()
    parts = raw.str.extract(
        r"(?P<month>\d{1,2})/(?P<day>\d{1,2})\s+(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})"
    )
    if parts.isna().any().any():
        bad = raw[parts.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"Date/Time EnergyPlus non parseabile. Esempi: {bad}")
    parts = parts.astype(int)
    day_roll = (parts["hour"] == 24).astype(int)
    parts.loc[day_roll == 1, "hour"] = 0
    ts = pd.to_datetime(
        {
            "year": year,
            "month": parts["month"],
            "day": parts["day"],
            "hour": parts["hour"],
            "minute": parts["minute"],
            "second": parts["second"],
        },
        errors="coerce",
    )
    ts = ts + pd.to_timedelta(day_roll, unit="D")
    if shift_interval_end:
        # E+ Hourly output is typically end-of-interval.
        ts = ts - pd.Timedelta(hours=1)
    return pd.DatetimeIndex(ts)


def infer_timestep_hours(index: pd.DatetimeIndex, default: float = 1.0) -> float:
    if len(index) < 2:
        return float(default)
    dt_h = (
        pd.Series(index).diff().dt.total_seconds().div(3600.0).replace([np.inf, -np.inf], np.nan).dropna()
    )
    dt_h = dt_h[dt_h > 0]
    if dt_h.empty:
        return float(default)
    return float(dt_h.median())


def read_epw_drybulb(epw_path: Path, year: int = 2020) -> pd.Series:
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW non trovato: {epw_path}")

    rows: List[Tuple[pd.Timestamp, float]] = []
    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            # EPW data starts at line 9 (0-index -> i >= 8)
            if i < 8:
                continue
            vals = [v.strip() for v in line.strip().split(",")]
            if len(vals) < 7:
                continue
            try:
                month = int(vals[1])
                day = int(vals[2])
                hour = int(vals[3])  # 1..24 end-of-hour
                dry_bulb = float(vals[6])
            except Exception:
                continue
            hour0 = 0 if hour == 24 else hour
            ts = pd.Timestamp(year=year, month=month, day=day, hour=hour0, minute=0, second=0)
            if hour == 24:
                ts = ts + pd.Timedelta(days=1)
            # Shift to begin-of-interval for consistency with parsed E+ timestamps.
            ts = ts - pd.Timedelta(hours=1)
            rows.append((ts, dry_bulb))

    if not rows:
        raise ValueError("Impossibile leggere temperatura esterna da EPW.")
    out = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([t for t, _ in rows]), name="T_out_epw_C")
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def detect_columns_by_pattern(columns: Iterable[str], patterns: Iterable[str]) -> List[str]:
    cols = []
    for c in columns:
        cl = c.lower()
        if any(p.lower() in cl for p in patterns):
            cols.append(c)
    return cols


def zone_from_col(col: str) -> str:
    return col.split(":", 1)[0].strip() if ":" in col else col


def build_ventilation_from_energyplus_energy(df_ep: pd.DataFrame, dt_h: float) -> Optional[pd.DataFrame]:
    loss_cols = detect_columns_by_pattern(
        df_ep.columns,
        [
            "Zone Infiltration Sensible Heat Loss Energy [J](Hourly)",
            "Zone Ventilation Sensible Heat Loss Energy [J](Hourly)",
        ],
    )
    gain_cols = detect_columns_by_pattern(
        df_ep.columns,
        [
            "Zone Infiltration Sensible Heat Gain Energy [J](Hourly)",
            "Zone Ventilation Sensible Heat Gain Energy [J](Hourly)",
        ],
    )
    if not loss_cols and not gain_cols:
        return None

    out = pd.DataFrame(index=df_ep.index)
    zones = sorted({zone_from_col(c) for c in (loss_cols + gain_cols)})
    for z in zones:
        z_loss = [c for c in loss_cols if zone_from_col(c) == z]
        z_gain = [c for c in gain_cols if zone_from_col(c) == z]
        loss_w = (
            df_ep[z_loss].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            / max(dt_h, 1e-9)
            / 3600.0
        )
        gain_w = (
            df_ep[z_gain].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            / max(dt_h, 1e-9)
            / 3600.0
        )
        out[f"Q_vent_loss_{z}_W"] = loss_w
        out[f"Q_vent_gain_{z}_W"] = gain_w
        out[f"Q_vent_net_{z}_W"] = gain_w - loss_w
    out["Q_vent_loss_total_W"] = out[[c for c in out.columns if c.startswith("Q_vent_loss_") and c.endswith("_W")]].sum(axis=1)
    out["Q_vent_gain_total_W"] = out[[c for c in out.columns if c.startswith("Q_vent_gain_") and c.endswith("_W")]].sum(axis=1)
    out["Q_vent_net_total_W"] = out["Q_vent_gain_total_W"] - out["Q_vent_loss_total_W"]
    out.attrs["source"] = "energyplus_zone_ventilation_energy"
    return out


def build_ventilation_from_flow(
    df_ep: pd.DataFrame,
    t_out: pd.Series,
    dt_h: float,
    rho_air: float = 1.204,
    cp_air: float = 1006.0,
) -> Optional[pd.DataFrame]:
    flow_cols = detect_columns_by_pattern(
        df_ep.columns,
        ["Zone Infiltration Current Density Volume Flow Rate [m3/s](Hourly)"],
    )
    if not flow_cols:
        return None

    zone_air_cols = {zone_from_col(c): c for c in detect_columns_by_pattern(df_ep.columns, ["Zone Mean Air Temperature [C](Hourly)"])}
    if not zone_air_cols:
        return None

    t_out_al = pd.to_numeric(t_out.reindex(df_ep.index), errors="coerce").interpolate(limit_direction="both")
    out = pd.DataFrame(index=df_ep.index)
    for c in flow_cols:
        z = zone_from_col(c)
        tcol = zone_air_cols.get(z)
        if tcol is None:
            continue
        vdot = pd.to_numeric(df_ep[c], errors="coerce").fillna(0.0)
        t_air = pd.to_numeric(df_ep[tcol], errors="coerce").interpolate(limit_direction="both")
        q_w = rho_air * cp_air * vdot * (t_air - t_out_al)  # + means ventilation loss from zone
        out[f"Q_vent_loss_{z}_W"] = q_w.clip(lower=0.0)
        out[f"Q_vent_gain_{z}_W"] = (-q_w).clip(lower=0.0)
        out[f"Q_vent_net_{z}_W"] = out[f"Q_vent_gain_{z}_W"] - out[f"Q_vent_loss_{z}_W"]
    if out.empty:
        return None
    out["Q_vent_loss_total_W"] = out[[c for c in out.columns if c.startswith("Q_vent_loss_") and c.endswith("_W")]].sum(axis=1)
    out["Q_vent_gain_total_W"] = out[[c for c in out.columns if c.startswith("Q_vent_gain_") and c.endswith("_W")]].sum(axis=1)
    out["Q_vent_net_total_W"] = out["Q_vent_gain_total_W"] - out["Q_vent_loss_total_W"]
    out.attrs["source"] = "estimated_from_infiltration_flow_and_epw"
    return out


def build_transmission_surface_table(df_ep: pd.DataFrame) -> Optional[pd.DataFrame]:
    conduction_cols = detect_columns_by_pattern(
        df_ep.columns,
        [
            "Surface Inside Face Conduction Heat Transfer Energy [J](Hourly)",
            "Surface Outside Face Conduction Heat Transfer Energy [J](Hourly)",
            "Surface Average Face Conduction Heat Transfer Energy [J](Hourly)",
        ],
    )
    if not conduction_cols:
        return None

    rows = []
    for c in conduction_cols:
        s = pd.to_numeric(df_ep[c], errors="coerce").fillna(0.0)
        net_kWh = s.sum() / 3.6e6
        pos_kWh = s.clip(lower=0.0).sum() / 3.6e6
        neg_kWh = (-s.clip(upper=0.0)).sum() / 3.6e6
        rows.append(
            {
                "surface_column": c,
                "surface_name": c.split(":")[0].strip() if ":" in c else c,
                "net_kWh": float(net_kWh),
                "positive_kWh": float(pos_kWh),
                "negative_kWh": float(neg_kWh),
                "abs_kWh": float(pos_kWh + neg_kWh),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_kWh", ascending=False).reset_index(drop=True)
    return out


def build_iso_ventilation_estimate(df_iso: pd.DataFrame, t_out: pd.Series) -> Optional[pd.DataFrame]:
    h_cols = [c for c in df_iso.columns if c.startswith("H_ve_")]
    if not h_cols:
        return None

    out = pd.DataFrame(index=df_iso.index)
    t_out_al = pd.to_numeric(t_out.reindex(df_iso.index), errors="coerce").interpolate(limit_direction="both")
    for h_col in h_cols:
        z = h_col.removeprefix("H_ve_")
        t_col = f"T_air_{z}"
        if t_col not in df_iso.columns:
            continue
        h_ve = pd.to_numeric(df_iso[h_col], errors="coerce").fillna(0.0)
        t_air = pd.to_numeric(df_iso[t_col], errors="coerce").interpolate(limit_direction="both")
        q_w = h_ve * (t_air - t_out_al)  # + loss
        out[f"Q_vent_loss_{z}_W"] = q_w.clip(lower=0.0)
        out[f"Q_vent_gain_{z}_W"] = (-q_w).clip(lower=0.0)
        out[f"Q_vent_net_{z}_W"] = out[f"Q_vent_gain_{z}_W"] - out[f"Q_vent_loss_{z}_W"]
    if out.empty:
        return None
    out["Q_vent_loss_total_W"] = out[[c for c in out.columns if c.startswith("Q_vent_loss_") and c.endswith("_W")]].sum(axis=1)
    out["Q_vent_gain_total_W"] = out[[c for c in out.columns if c.startswith("Q_vent_gain_") and c.endswith("_W")]].sum(axis=1)
    out["Q_vent_net_total_W"] = out["Q_vent_gain_total_W"] - out["Q_vent_loss_total_W"]
    return out


def suggested_idf_outputs_block() -> str:
    return (
        "! --- Add these outputs for ventilation and transmission breakdown ---\n"
        "Output:Variable, *, Zone Infiltration Sensible Heat Loss Energy, Hourly;\n"
        "Output:Variable, *, Zone Infiltration Sensible Heat Gain Energy, Hourly;\n"
        "Output:Variable, *, Zone Ventilation Sensible Heat Loss Energy, Hourly;\n"
        "Output:Variable, *, Zone Ventilation Sensible Heat Gain Energy, Hourly;\n"
        "Output:Variable, *, Surface Inside Face Conduction Heat Transfer Energy, Hourly;\n"
        "Output:Variable, *, Surface Outside Face Conduction Heat Transfer Energy, Hourly;\n"
        "Output:Variable, *, Surface Average Face Conduction Heat Transfer Energy, Hourly;\n"
        "! Optional split for windows/convection/radiation if needed:\n"
        "! Output:Variable, *, Surface Inside Face Convection Heat Gain Rate, Hourly;\n"
        "! Output:Variable, *, Surface Window Heat Gain Energy, Hourly;\n"
    )


def make_report_html(
    out_html: Path,
    vent_ep: Optional[pd.DataFrame],
    vent_iso: Optional[pd.DataFrame],
    trans_table: Optional[pd.DataFrame],
    missing_notes: List[str],
    summary_table: pd.DataFrame,
    idf_path: str,
) -> None:
    sections: List[str] = []
    sections.append("<h1>EnergyPlus Ventilation / Transmission Report</h1>")
    sections.append(f"<p><b>IDF reference:</b> {idf_path}</p>")

    sections.append("<h2>Summary</h2>")
    sections.append(summary_table.to_html(index=False, border=0, classes="table table-striped"))

    if vent_ep is not None:
        fig_vent = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Ventilation - Total Loss/Gain Power", "Ventilation - Net Power (Gain - Loss)"),
        )
        fig_vent.add_trace(go.Scatter(x=vent_ep.index, y=vent_ep["Q_vent_loss_total_W"], name="EP Loss [W]"), row=1, col=1)
        fig_vent.add_trace(go.Scatter(x=vent_ep.index, y=vent_ep["Q_vent_gain_total_W"], name="EP Gain [W]"), row=1, col=1)
        fig_vent.add_trace(go.Scatter(x=vent_ep.index, y=vent_ep["Q_vent_net_total_W"], name="EP Net [W]"), row=2, col=1)
        if vent_iso is not None:
            common = vent_ep.index.intersection(vent_iso.index)
            fig_vent.add_trace(
                go.Scatter(x=common, y=vent_iso.loc[common, "Q_vent_loss_total_W"], name="ISO Loss [W]", line={"dash": "dash"}),
                row=1,
                col=1,
            )
            fig_vent.add_trace(
                go.Scatter(x=common, y=vent_iso.loc[common, "Q_vent_gain_total_W"], name="ISO Gain [W]", line={"dash": "dash"}),
                row=1,
                col=1,
            )
            fig_vent.add_trace(
                go.Scatter(x=common, y=vent_iso.loc[common, "Q_vent_net_total_W"], name="ISO Net [W]", line={"dash": "dash"}),
                row=2,
                col=1,
            )
        fig_vent.update_layout(template="plotly_white", hovermode="x unified", height=750)
        sections.append("<h2>Ventilation</h2>")
        sections.append(fig_vent.to_html(include_plotlyjs="cdn", full_html=False))
        sections.append(
            "<p><i>Convention: Loss > 0 means heat removed from zone air by ventilation/infiltration.</i></p>"
        )

    if trans_table is not None and not trans_table.empty:
        fig_tr = go.Figure()
        top = trans_table.head(20)
        fig_tr.add_trace(
            go.Bar(x=top["surface_name"], y=top["positive_kWh"], name="Positive kWh")
        )
        fig_tr.add_trace(
            go.Bar(x=top["surface_name"], y=top["negative_kWh"], name="Negative kWh")
        )
        fig_tr.update_layout(
            template="plotly_white",
            barmode="group",
            title="Transmission by Surface (top 20 by abs energy)",
            xaxis_title="Surface",
            yaxis_title="Energy [kWh]",
        )
        sections.append("<h2>Transmission by Surface</h2>")
        sections.append(fig_tr.to_html(include_plotlyjs=False, full_html=False))
        sections.append(trans_table.to_html(index=False, border=0, classes="table table-striped"))
    else:
        sections.append("<h2>Transmission by Surface</h2>")
        sections.append("<p>Not available in current CSV. Add conduction output variables in IDF.</p>")

    if missing_notes:
        sections.append("<h2>Missing / Notes</h2>")
        sections.append("<ul>" + "".join(f"<li>{n}</li>" for n in missing_notes) + "</ul>")
        sections.append("<h3>Suggested IDF Outputs</h3>")
        sections.append(f"<pre>{suggested_idf_outputs_block()}</pre>")

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>EnergyPlus Ventilation/Transmission Report</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;} "
        "th,td{padding:6px 8px;border:1px solid #ddd;} th{background:#f5f5f5;}</style>"
        "</head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_csv = Path(args.energyplus_csv).expanduser().resolve()
    if not ep_csv.exists():
        raise FileNotFoundError(f"CSV EnergyPlus non trovato: {ep_csv}")
    df_ep = pd.read_csv(ep_csv)
    if "Date/Time" not in df_ep.columns:
        raise KeyError("Colonna 'Date/Time' non trovata nel CSV EnergyPlus.")
    df_ep.index = parse_energyplus_datetime(df_ep["Date/Time"], year=args.year, shift_interval_end=True)
    df_ep = df_ep.loc[~df_ep.index.duplicated(keep="first")].sort_index()
    dt_h = infer_timestep_hours(df_ep.index, default=1.0)

    missing_notes: List[str] = []

    epw_path = Path(args.epw_path).expanduser().resolve()
    t_out = None
    if epw_path.exists():
        try:
            t_out = read_epw_drybulb(epw_path, year=args.year)
        except Exception as exc:
            missing_notes.append(f"EPW letto con errore ({epw_path}): {exc}")
    else:
        missing_notes.append(f"EPW non trovato: {epw_path}")

    vent_ep = build_ventilation_from_energyplus_energy(df_ep, dt_h=dt_h)
    if vent_ep is None:
        if t_out is not None:
            vent_ep = build_ventilation_from_flow(df_ep, t_out=t_out, dt_h=dt_h)
            if vent_ep is None:
                missing_notes.append("Ventilazione non calcolabile: mancano sia energy variables che flow/temperature.")
            else:
                missing_notes.append(
                    "Ventilazione EnergyPlus stimata da portata infiltrazione + T aria zona + T esterna EPW (non da energia diretta)."
                )
        else:
            missing_notes.append("Ventilazione non calcolabile: mancano variabili energia e EPW per stima da portata.")
    else:
        missing_notes.append("Ventilazione EnergyPlus calcolata da variabili energia dirette (J/h).")

    trans_table = build_transmission_surface_table(df_ep)
    if trans_table is None:
        missing_notes.append("Trasmissione per parete non disponibile: mancano colonne 'Surface ... Conduction Heat Transfer Energy'.")

    # Optional ISO comparison (ventilation only)
    vent_iso = None
    iso_csv = Path(args.iso_csv).expanduser().resolve()
    if iso_csv.exists() and t_out is not None and vent_ep is not None:
        try:
            df_iso = pd.read_csv(iso_csv)
            tcol = df_iso.columns[0]
            df_iso.index = pd.to_datetime(df_iso[tcol], errors="coerce")
            df_iso = df_iso.loc[~df_iso.index.isna()].set_index(df_iso.index).sort_index()
            vent_iso = build_iso_ventilation_estimate(df_iso, t_out=t_out)
            if vent_iso is None:
                missing_notes.append("Confronto ISO ventilazione non disponibile: colonne H_ve_*/T_air_* mancanti.")
        except Exception as exc:
            missing_notes.append(f"Errore lettura ISO CSV ({iso_csv}): {exc}")
    elif not iso_csv.exists():
        missing_notes.append(f"CSV ISO non trovato (opzionale): {iso_csv}")
    elif t_out is None:
        missing_notes.append("Confronto ISO ventilazione saltato: manca T esterna (EPW).")

    # Summary table
    summary_rows = []
    if vent_ep is not None:
        summary_rows.append(
            {
                "metric": "EP ventilation loss [kWh]",
                "value": float((vent_ep["Q_vent_loss_total_W"] * dt_h).sum() / 1000.0),
            }
        )
        summary_rows.append(
            {
                "metric": "EP ventilation gain [kWh]",
                "value": float((vent_ep["Q_vent_gain_total_W"] * dt_h).sum() / 1000.0),
            }
        )
        summary_rows.append(
            {
                "metric": "EP ventilation net [kWh]",
                "value": float((vent_ep["Q_vent_net_total_W"] * dt_h).sum() / 1000.0),
            }
        )
    if vent_iso is not None:
        summary_rows.append(
            {
                "metric": "ISO ventilation loss [kWh]",
                "value": float((vent_iso["Q_vent_loss_total_W"] * dt_h).sum() / 1000.0),
            }
        )
        summary_rows.append(
            {
                "metric": "ISO ventilation gain [kWh]",
                "value": float((vent_iso["Q_vent_gain_total_W"] * dt_h).sum() / 1000.0),
            }
        )
        summary_rows.append(
            {
                "metric": "ISO ventilation net [kWh]",
                "value": float((vent_iso["Q_vent_net_total_W"] * dt_h).sum() / 1000.0),
            }
        )
    if trans_table is not None:
        summary_rows.append(
            {
                "metric": "EP transmission abs total [kWh]",
                "value": float(trans_table["abs_kWh"].sum()),
            }
        )
    summary_table = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame([{"metric": "No data", "value": np.nan}])

    # Save CSV outputs
    if vent_ep is not None:
        vent_ep_out = out_dir / f"{args.out_prefix}_ventilation_energyplus_hourly.csv"
        vent_ep.to_csv(vent_ep_out)
    if vent_iso is not None:
        common = vent_ep.index.intersection(vent_iso.index)
        cmp = pd.DataFrame(index=common)
        cmp["EP_loss_W"] = vent_ep.loc[common, "Q_vent_loss_total_W"]
        cmp["ISO_loss_W"] = vent_iso.loc[common, "Q_vent_loss_total_W"]
        cmp["EP_gain_W"] = vent_ep.loc[common, "Q_vent_gain_total_W"]
        cmp["ISO_gain_W"] = vent_iso.loc[common, "Q_vent_gain_total_W"]
        cmp["delta_loss_W_ISO_minus_EP"] = cmp["ISO_loss_W"] - cmp["EP_loss_W"]
        cmp_out = out_dir / f"{args.out_prefix}_ventilation_ep_vs_iso_hourly.csv"
        cmp.to_csv(cmp_out)
    if trans_table is not None:
        trans_out = out_dir / f"{args.out_prefix}_transmission_surfaces.csv"
        trans_table.to_csv(trans_out, index=False)

    missing_out = out_dir / f"{args.out_prefix}_missing_outputs_and_notes.txt"
    missing_out.write_text(
        "\n".join(missing_notes + ["", "Suggested IDF outputs:", suggested_idf_outputs_block()]),
        encoding="utf-8",
    )

    # HTML report
    out_html = out_dir / f"{args.out_prefix}.html"
    make_report_html(
        out_html=out_html,
        vent_ep=vent_ep,
        vent_iso=vent_iso,
        trans_table=trans_table,
        missing_notes=missing_notes,
        summary_table=summary_table,
        idf_path=args.idf_path,
    )

    print(f"EnergyPlus CSV: {ep_csv}")
    print(f"ISO CSV       : {iso_csv} ({'found' if iso_csv.exists() else 'missing'})")
    print(f"EPW           : {epw_path} ({'found' if epw_path.exists() else 'missing'})")
    print(f"Report HTML   : {out_html}")
    if vent_ep is not None:
        print(f"Ventilation   : {out_dir / f'{args.out_prefix}_ventilation_energyplus_hourly.csv'}")
    if trans_table is not None:
        print(f"Transmission  : {out_dir / f'{args.out_prefix}_transmission_surfaces.csv'}")
    if vent_iso is not None:
        print(f"EP vs ISO vent: {out_dir / f'{args.out_prefix}_ventilation_ep_vs_iso_hourly.csv'}")
    print(f"Notes         : {missing_out}")


if __name__ == "__main__":
    main()
