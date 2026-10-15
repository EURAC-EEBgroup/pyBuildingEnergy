"""Generate one HTML report per ASHRAE Standard 140-2023 Section 7 case
(600-695, 900-995 and the six free-float cases) plus an index page.

Each report shows the inputs applied, the annual results against the
Annex A3 range that applies to the case (Tables A3-1 to A3-4), the monthly
loads, the hourly outputs for the days required by Table 7-47 and the
annual hourly profiles.

Usage:
    PYTHONPATH=src python examples/ashrae140_bestest/generate_reports.py
Output: examples/ashrae140_bestest/reports/index.html
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR.parent.parent / "src"))
sys.path.insert(0, str(EXAMPLES_DIR.parent))
sys.path.insert(0, str(EXAMPLES_DIR))

import case_definitions as cd  # noqa: E402
from geometry_3d import building_figure, geometry_caption  # noqa: E402
from run_comparison import FREE_FLOAT_CASES, range_status, verdict_text  # noqa: E402
from pybuildingenergy.source.utils import ISO52016  # noqa: E402
import validation_report_html as vr  # noqa: E402

REPORT_DIR = EXAMPLES_DIR / "reports"
SURFACE_COEFFICIENTS = "ashrae_convective"

DESCRIPTIONS = {
    600: ("Low-mass base case, 12 m2 of south windows, 20/27 degC thermostat", "7.2.1"),
    610: ("Case 600 with a 1.0 m south overhang at roof level", "7.2.2.1.1"),
    620: ("Case 600 with 6 m2 east and 6 m2 west windows", "7.2.2.1.2"),
    630: ("Case 620 with overhang and side fins on the east/west windows", "7.2.2.1.3"),
    640: ("Case 600 with heating setback to 10 degC from 23:00 to 07:00", "7.2.2.1.4"),
    650: ("Case 600 with night ventilation 18:00-07:00, no heating", "7.2.2.1.5"),
    660: ("Case 600 with low-e argon-filled windows", "7.2.2.1.6"),
    670: ("Case 600 with single-pane windows", "7.2.2.1.7"),
    680: ("Case 600 with increased wall and roof insulation", "7.2.2.1.8"),
    685: ("Case 600 with a 20/20 degC thermostat", "7.2.2.1.9"),
    695: ("Case 680 with a 20/20 degC thermostat", "7.2.2.1.10"),
    900: ("High-mass base case", "7.2.2.2.1"),
    910: ("Case 900 with a 1.0 m south overhang at roof level", "7.2.2.2.2"),
    920: ("Case 900 with east and west windows", "7.2.2.2.3"),
    930: ("Case 920 with overhang and side fins", "7.2.2.2.4"),
    940: ("Case 900 with heating setback", "7.2.2.2.5"),
    950: ("Case 900 with night ventilation, no heating", "7.2.2.2.6"),
    960: ("Sunspace: lightweight conditioned back zone + free-floating high-mass sun zone", "7.2.2.2.7"),
    980: ("Case 900 with increased wall and roof insulation", "7.2.2.2.8"),
    985: ("Case 900 with a 20/20 degC thermostat", "7.2.2.2.9"),
    995: ("Case 980 with a 20/20 degC thermostat", "7.2.2.2.10"),
}
FF_DESCRIPTIONS = {
    600: ("Case 600 without heating or cooling", "7.2.2.3.1"),
    650: ("Case 650 without heating or cooling, night vent fan kept", "7.2.2.3.2"),
    680: ("Case 680 without heating or cooling", "7.2.2.3.3"),
    900: ("Case 900 without heating or cooling", "7.2.2.3.4"),
    950: ("Case 950 without heating or cooling, night vent fan kept", "7.2.2.3.5"),
    980: ("Case 980 without heating or cooling", "7.2.2.3.6"),
}

# Table 7-47: days with required hourly output.
LOAD_DAYS = {600: ["02-01", "07-14"], 660: ["02-01", "07-14"], 670: ["02-01", "07-14"],
             680: ["02-01", "07-14"], 685: ["02-01", "07-14"], 695: ["02-01", "07-14"],
             900: ["02-01", "07-14"], 980: ["02-01", "07-14"], 985: ["02-01", "07-14"],
             995: ["02-01", "07-14"], 640: ["02-01"], 940: ["02-01"]}
FF_DAYS = {600: ["02-01"], 680: ["02-01"], 900: ["02-01"], 980: ["02-01"],
           650: ["07-14"], 950: ["07-14"]}
DEFAULT_DAYS = ["02-01", "07-14"]  # diagnostic only for cases not listed in Table 7-47


def _run_single_zone(bui) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.Temperature_and_Energy_needs_calculation(
            bui, weather_source="epw", path_weather_file=str(cd.WEATHER_FILE),
            c_int_per_A_us=0.0, hvac_control_variable="air")
    return out[0] if isinstance(out, tuple) else out


def _run_960() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.simulate_envelope_multizone_free_floating(
            cd.build_case_960(SURFACE_COEFFICIENTS), path_weather_file=str(cd.WEATHER_FILE),
            weather_source="epw", c_int_per_A_us=0.0, hvac_control_variable="air")
    df = out[0] if isinstance(out, tuple) else out
    q = df["Q_HVAC_back"]
    epw = pd.read_csv(cd.WEATHER_FILE, skiprows=8, header=None)
    res = pd.DataFrame(index=df.index)
    res["Q_H"] = q.clip(lower=0.0)
    res["Q_C"] = (-q).clip(lower=0.0)
    res["T_air"] = df["T_air_back"]
    res["T_op"] = df["T_op_back"]
    res["T_air_sun"] = df["T_air_sun"]
    res["T_ext"] = epw[6].to_numpy(float)[: len(res)]
    return res


def simulate(case_number: int, free_float: bool = False) -> pd.DataFrame:
    if case_number == 960:
        return _run_960()
    bui = cd.build_bestest_case(case_number, surface_coefficients=SURFACE_COEFFICIENTS)
    if free_float:
        bui["building_parameters"]["system_capacities"].update(
            {"heating_capacity": 0.0, "cooling_capacity": 0.0})
    return _run_single_zone(bui)


def summary(df: pd.DataFrame) -> dict:
    return {
        "heating_MWh": df["Q_H"].sum() / 1e6,
        "cooling_MWh": df["Q_C"].sum() / 1e6,
        "peak_heating_kW": df["Q_H"].max() / 1e3,
        "peak_cooling_kW": df["Q_C"].max() / 1e3,
        "peak_heating_time": df["Q_H"].idxmax(),
        "peak_cooling_time": df["Q_C"].idxmax(),
        "T_mean": df["T_air"].mean(), "T_min": df["T_air"].min(), "T_max": df["T_air"].max(),
        "T_min_time": df["T_air"].idxmin(), "T_max_time": df["T_air"].idxmax(),
    }


def acceptance(case_number: int):
    return (cd.LOW_MASS_ACCEPTANCE | cd.HIGH_MASS_ACCEPTANCE).get(case_number)


def range_rows(case_number: int, results: dict) -> list[dict]:
    """One row per Annex A3 range that applies to the case."""
    crit = acceptance(case_number)
    if crit is None:
        return []
    rows = []
    parent = crit["parent"]
    for metric in ("heating", "cooling"):
        limits = crit[metric]
        value = results[case_number][f"{metric}_MWh"]
        label = f"Case {case_number}"
        if parent is not None:
            value = value - results[parent][f"{metric}_MWh"]
            label = f"Case {case_number} − Case {parent}"
            if crit.get("reverse"):
                value, label = -value, f"Case {parent} − Case {case_number}"
        rows.append({"case": case_number, "metric": metric, "label": label, "value": value,
                     "limits": limits, "status": range_status(value, limits),
                     "table": cd.annex_a3_table(case_number, metric)})
    return rows


def verdict(r: dict) -> str:
    """Result phrase without the range values, citing where the standard prints them, e.g.
    "+4.417 — PASS (within the ASHRAE 140 acceptance range, Case 600; 2020 edition,
    Addendum b — ANSI/ASHRAE 140-2023, Annex A3, Table A3-1)"; wording from
    run_comparison.verdict_text."""
    text = vr.esc(verdict_text(r["value"], r["status"], r["label"], r["table"]))
    if r["status"] in ("PASS", "FAIL"):
        text = text.replace(f" — {r['status']} (", f" — {vr.badge(r['status'])} (", 1)
    return f"<span class='verdict'>{text}</span>"


def _fmt_time(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %d, %H:00")


def _inputs_section(case_number: int) -> str:
    if case_number == 960:
        bui = cd.build_case_960(SURFACE_COEFFICIENTS)
    else:
        bui = cd.build_bestest_case(case_number, surface_coefficients=SURFACE_COEFFICIENTS)
    rows = []
    for s in bui["building_surface"]:
        shading = []
        if s.get("width_or_distance_of_shading_elements") is not None:
            shading.append(f"overhang D={s['overhang_properties']['width_of_horizontal_overhangs']} m, "
                           f"L={s['width_or_distance_of_shading_elements']} m")
        if s.get("side_fins"):
            shading.append("fins D=1.0 m")
        rows.append([
            s["name"], s.get("zone", "-"), s["type"], vr.fmt(s["area"], 1), vr.fmt(s["u_value"], 3),
            vr.fmt(s.get("thermal_capacity", 0) / 1000, 1),
            vr.fmt(s["convective_heat_transfer_coefficient_internal"], 1),
            vr.fmt(s["convective_heat_transfer_coefficient_external"], 1),
            vr.fmt(s.get("solar_absorptance", 0), 2),
            vr.fmt(s.get("g_value"), 3) if s["type"] == "transparent" else "-",
            vr.fmt(s.get("solar_transmittance_correction_factor"), 3) if s["type"] == "transparent" else "-",
            "; ".join(shading) or "-",
        ])
    sp = bui["building_parameters"]["temperature_setpoints"]
    zones = bui.get("zones")
    if zones:
        ctrl = "back zone 20/27 degC (air); sun zone free-floating"
    else:
        hs = sp["heating_setpoint"]
        ctrl = ("no heating" if hs < -100 else f"heating {hs:g} degC") + f", cooling {sp['cooling_setpoint']:g} degC"
        if sp.get("heating_setback") not in (None, hs) and hs > -100:
            ctrl += f", heating setback {sp['heating_setback']:g} degC 23:00-07:00"
    vent = bui["building_parameters"]["ventilation"]["components"]
    vent_txt = "; ".join(f"{c['name']} {c['air_changes_per_hour']:.3f} ach" for c in vent)
    return (
        "<h2>Inputs applied</h2>"
        f"<p>Thermostat (zone air temperature, Section 7.2.1.13): {vr.esc(ctrl)}. "
        f"Air flows (altitude-corrected, Tables 7-4/7-16): {vr.esc(vent_txt)}. "
        "Surface coefficients: Tables 7-7/7-9 convective values. Weather: 725650TY (Denver Intl, TMY3).</p>"
        + vr.table(["Surface", "Zone", "Type", "Area m2", "U W/m2K", "kappa kJ/m2K",
                    "h_ci", "h_ce", "a_sol", "g_n", "F_w", "Shading"], rows)
    )


def _monthly_section(df: pd.DataFrame) -> str:
    g = df.groupby(df.index.month)
    rows = [[vr.MONTHS[m - 1], vr.fmt(g["Q_H"].sum()[m] / 1e3, 1), vr.fmt(g["Q_C"].sum()[m] / 1e3, 1),
             vr.fmt(g["Q_H"].max()[m] / 1e3, 3), vr.fmt(g["Q_C"].max()[m] / 1e3, 3),
             vr.fmt(g["T_air"].mean()[m], 2)] for m in range(1, 13)]
    fig = go.Figure()
    fig.add_bar(x=vr.MONTHS, y=(g["Q_H"].sum() / 1e3).to_list(), name="Heating kWh",
                marker_color=vr.COLORS["heating"])
    fig.add_bar(x=vr.MONTHS, y=(g["Q_C"].sum() / 1e3).to_list(), name="Cooling kWh",
                marker_color=vr.COLORS["cooling"])
    fig.update_layout(title="Monthly heating and sensible cooling loads", barmode="group",
                      yaxis_title="kWh")
    return ("<h2>Monthly results</h2>" + vr.figure(fig)
            + vr.table(["Month", "Heating kWh", "Cooling kWh", "Peak heating kW",
                        "Peak cooling kW", "Mean T_air degC"], rows))


def _daily_section(df: pd.DataFrame, days, required: bool, free_float: bool) -> str:
    out = ["<h2>Hourly outputs on " + ("the days required by Table 7-47" if required
                                         else "two diagnostic days (not required by Table 7-47)") + "</h2>"]
    for day in days:
        month, dom = (int(x) for x in day.split("-"))
        d = df[(df.index.month == month) & (df.index.day == dom)]
        label = pd.Timestamp(2009, month, dom).strftime("%B %d")
        fig = go.Figure()
        hours = list(range(1, len(d) + 1))
        if not free_float:
            fig.add_bar(x=hours, y=(d["Q_H"] / 1e3).to_list(), name="Heating kWh (+)",
                        marker_color=vr.COLORS["heating"])
            fig.add_bar(x=hours, y=(-d["Q_C"] / 1e3).to_list(), name="Sensible cooling kWh (-)",
                        marker_color=vr.COLORS["cooling"])
        fig.add_scatter(x=hours, y=d["T_air"].to_list(), name="Zone air T degC", yaxis="y2",
                        line=dict(color=vr.COLORS["temperature"]))
        fig.add_scatter(x=hours, y=d["T_ext"].to_list(), name="Outdoor T degC", yaxis="y2",
                        line=dict(color=vr.COLORS["outdoor"], dash="dot"))
        fig.update_layout(title=f"{label}: hourly integrated values (hour 1 = 00:00-01:00)",
                          xaxis_title="Hour", yaxis_title="kWh",
                          yaxis2=dict(title="degC", overlaying="y", side="right"), barmode="relative")
        rows = [[h, vr.fmt(q_h / 1e3, 3), vr.fmt(-q_c / 1e3, 3), vr.fmt(t, 2), vr.fmt(te, 1)]
                for h, q_h, q_c, t, te in zip(hours, d["Q_H"], d["Q_C"], d["T_air"], d["T_ext"])]
        out.append(f"<h3>{label}</h3>" + vr.figure(fig)
                   + vr.table(["Hour", "Heating kWh", "Cooling kWh (-)", "T_air degC", "T_ext degC"], rows))
    return "".join(out)


def _annual_profile_section(df: pd.DataFrame, free_float: bool, sun_zone: bool) -> str:
    """Daily values over the year (hourly series would make each page several MB)."""
    daily = df.resample("D")
    days = daily["T_air"].mean().index
    fig = go.Figure()
    if not free_float:
        fig.add_scatter(x=days, y=(daily["Q_H"].sum() / 1e3).to_list(), name="Daily heating kWh",
                        line=dict(color=vr.COLORS["heating"]))
        fig.add_scatter(x=days, y=(daily["Q_C"].sum() / 1e3).to_list(), name="Daily cooling kWh",
                        line=dict(color=vr.COLORS["cooling"]))
    fig.add_scatter(x=days, y=daily["T_air"].max().to_list(), name="Daily max zone air T",
                    yaxis="y2", line=dict(color=vr.COLORS["temperature"], width=0.5), showlegend=False)
    fig.add_scatter(x=days, y=daily["T_air"].min().to_list(), name="Daily min/max zone air T",
                    yaxis="y2", line=dict(color=vr.COLORS["temperature"], width=0.5),
                    fill="tonexty", fillcolor="rgba(149,96,0,0.15)")
    fig.add_scatter(x=days, y=daily["T_air"].mean().to_list(), name="Daily mean zone air T",
                    yaxis="y2", line=dict(color=vr.COLORS["temperature"]))
    if sun_zone:
        fig.add_scatter(x=days, y=daily["T_air_sun"].mean().to_list(), name="Daily mean sun zone air T",
                        yaxis="y2", line=dict(color="#7a3e9d"))
    fig.add_scatter(x=days, y=daily["T_ext"].mean().to_list(), name="Daily mean outdoor T",
                    yaxis="y2", line=dict(color=vr.COLORS["outdoor"], dash="dot"))
    fig.update_layout(title="Annual profile (daily values)", yaxis_title="kWh/day",
                      yaxis2=dict(title="degC", overlaying="y", side="right"))
    return "<h2>Annual profile</h2>" + vr.figure(fig, height=420)


def _balance_section(df: pd.DataFrame) -> str:
    cols = [("Q_solar_gains", "Solar gains"), ("Q_internal_gains", "Internal gains"),
            ("Q_tr_opaque", "Transmission, opaque"), ("Q_tr_window", "Transmission, windows"),
            ("Q_ve", "Ventilation / infiltration"), ("Q_storage", "Storage")]
    rows = [[label, vr.fmt(df[c].sum() / 1e3, 1)] for c, label in cols if c in df.columns]
    if not rows:
        return ""
    return ("<h2>Annual energy balance terms</h2>"
            "<p class='muted'>Annual sums of the engine's hourly balance columns, as reported by "
            "pybuildingenergy (sign convention of the engine).</p>"
            + vr.table(["Term", "kWh/year"], rows))


def case_page(case_number: int, df: pd.DataFrame, results: dict, free_float: bool = False) -> str:
    s = summary(df)
    if free_float:
        desc, section = FF_DESCRIPTIONS[case_number]
        title = f"ASHRAE 140-2023 Case {case_number}FF"
        days, required = FF_DAYS[case_number], True
        body = vr.cards([
            ("Annual mean zone air T", f"{s['T_mean']:.2f} degC", "Section 7.3.6.1"),
            ("Annual minimum", f"{s['T_min']:.2f} degC", _fmt_time(s["T_min_time"])),
            ("Annual maximum", f"{s['T_max']:.2f} degC", _fmt_time(s["T_max_time"])),
            ("Annex A3 range", vr.badge("N/A"), "Free-float cases have no A3 range"),
        ])
        if case_number == 900:
            bins = np.arange(-50, 99, 1)
            freq, _ = np.histogram(df["T_air"].to_numpy(), bins=bins)
            fig = go.Figure(go.Bar(x=bins[:-1], y=freq, marker_color=vr.COLORS["temperature"]))
            fig.update_layout(title="Annual 1 degC zone air temperature bin frequencies (Section 7.3.7)",
                              xaxis_title="Bin lower bound degC", yaxis_title="hours")
            body += "<h2>Temperature bins</h2>" + vr.figure(fig)
    else:
        desc, section = DESCRIPTIONS[case_number]
        title = f"ASHRAE 140-2023 Case {case_number}"
        days = LOAD_DAYS.get(case_number, DEFAULT_DAYS)
        required = case_number in LOAD_DAYS
        rows = range_rows(case_number, results)
        status_cells = [[{"heating": "Annual heating [MWh]", "cooling": "Annual sensible cooling [MWh]"}[r["metric"]],
                         verdict(r)] for r in rows]
        body = vr.cards([
            ("Annual heating", f"{s['heating_MWh']:.3f} MWh", "Section 7.3.1.1"),
            ("Annual sensible cooling", f"{s['cooling_MWh']:.3f} MWh", "Section 7.3.1.2"),
            ("Peak heating", f"{s['peak_heating_kW']:.3f} kW",
             _fmt_time(s["peak_heating_time"]) if s["peak_heating_kW"] > 0 else "-"),
            ("Peak sensible cooling", f"{s['peak_cooling_kW']:.3f} kW", _fmt_time(s["peak_cooling_time"])),
        ])
        body += ("<h2>Annex A3 acceptance</h2>"
                 + vr.table(["Load", "Result"], status_cells, raw_columns=(1,))
                 + "<p class='muted'>Range values: ANSI/ASHRAE Standard 140-2023, Annex A3 (not "
                   "reproduced, copyright ASHRAE). Peak loads have no Annex A3 range and are diagnostic only.</p>")
        if case_number == 960:
            body += vr.cards([
                ("Sun zone mean air T", f"{df['T_air_sun'].mean():.2f} degC", "free-floating"),
                ("Sun zone minimum", f"{df['T_air_sun'].min():.2f} degC", _fmt_time(df['T_air_sun'].idxmin())),
                ("Sun zone maximum", f"{df['T_air_sun'].max():.2f} degC", _fmt_time(df['T_air_sun'].idxmax())),
            ])
    body += ("<h2>Geometry</h2>"
             f"<p class='muted'>{vr.esc(geometry_caption(case_number, free_float))} Drag to rotate, scroll to zoom.</p>"
             + vr.figure(building_figure(case_number, heavy=case_number >= 900, free_float=free_float),
                         height=520))
    body += _inputs_section(case_number)
    if not free_float:
        body += _monthly_section(df)
    body += _daily_section(df, days, required, free_float)
    body += _annual_profile_section(df, free_float, case_number == 960)
    body += _balance_section(df)
    subtitle = (f"{vr.esc(desc)} — Section {section}. "
                f"<a href='index.html'>All cases</a>")
    footer = ("pybuildingenergy single-zone ISO 52016-1 engine"
              + (" (multizone engine for Case 960)" if case_number == 960 else "")
              + ". Generated by examples/ashrae140_bestest/generate_reports.py.")
    return vr.page(title, subtitle, body, footer)


def index_page(results: dict, ff: dict, rows: list[dict]) -> str:
    low = [r for r in rows if r["case"] in cd.LOW_MASS_ACCEPTANCE and r["status"] in ("PASS", "FAIL")]
    high = [r for r in rows if r["case"] in cd.HIGH_MASS_ACCEPTANCE and r["status"] in ("PASS", "FAIL")]
    n_low, n_high = sum(r["status"] == "PASS" for r in low), sum(r["status"] == "PASS" for r in high)
    if cd.ANNEX_A3_AVAILABLE:
        body = vr.cards([
            ("Thermal Fabric Low Mass", f"{n_low}/{len(low)}",
             vr.badge("PASS" if n_low >= 18 else "FAIL") + " Table A3-14 requires 18 of 21"),
            ("Thermal Fabric High Mass", f"{n_high}/{len(high)}",
             vr.badge("PASS" if n_high >= 17 else "FAIL") + " Table A3-14 requires 17 of 19"),
        ])
    else:
        body = ("<div class='callout'>The Annex A3 range values were not available when this page "
                "was generated, so the range checks are not shown.</div>")
    body += ("<div class='callout'><b>Acceptance-range values are not reproduced.</b> They are copyright "
             "ASHRAE and are printed in ANSI/ASHRAE Standard 140-2023, Normative Annex A3, Tables A3-1 "
             "(low-mass heating), A3-2 (low-mass sensible cooling), A3-3 (high-mass heating) and A3-4 "
             "(high-mass sensible cooling); the group pass criteria are in Table A3-14. Annex A3 was "
             "introduced by Addendum b to Standard 140-2020. Owning the standard is enough: the values can be entered "
             "in <code>Standards/local_reference_data/ashrae140_2023_annex_a3.json</code> (format: "
             "<code>examples/ashrae140_bestest/annex_a3_ranges.template.json</code>).</div>")
    table_rows = []
    for case_number in list(cd.LOW_MASS_ACCEPTANCE) + list(cd.HIGH_MASS_ACCEPTANCE):
        r = results[case_number]
        case_rows = {x["metric"]: x for x in rows if x["case"] == case_number}
        cells = [f"<a href='case_{case_number}.html'>{case_number}</a>", vr.esc(DESCRIPTIONS[case_number][0]),
                 vr.fmt(r["heating_MWh"], 3), vr.fmt(r["cooling_MWh"], 3)]
        for metric in ("heating", "cooling"):
            x = case_rows[metric]
            cells.append(verdict(x))
        table_rows.append(cells)
    ff_rows = [[f"<a href='case_{n}FF.html'>{n}FF</a>", vr.esc(FF_DESCRIPTIONS[n][0]),
                vr.fmt(v["T_mean"], 2), vr.fmt(v["T_min"], 2), vr.fmt(v["T_max"], 2)] for n, v in ff.items()]
    body += ("<h2>Cases with Annex A3 ranges</h2>"
             "<p class='muted'>Cases 600 and 900 are checked on their absolute annual loads; every other "
             "case on its difference from the parent case, as Tables A3-1 to A3-4 prescribe.</p>"
             + vr.table(["Case", "Description", "Heating MWh", "Cooling MWh", "Heating (Annex A3)",
                         "Cooling (Annex A3)"], table_rows, raw_columns=(0, 1, 4, 5))
             + "<h2>Free-float cases (no Annex A3 range)</h2>"
             + vr.table(["Case", "Description", "Mean T_air degC", "Min degC", "Max degC"], ff_rows,
                        raw_columns=(0, 1))
             + "<div class='callout'>Inputs follow ANSI/ASHRAE Standard 140-2023 Section 7 and its "
               "accompanying files; see <code>examples/ashrae140_bestest/README.md</code> for the "
               "mapping onto the ISO 52016-1 engine and the documented deviations (interior solar "
               "distribution of Table 7-13 not applied; ISO 52016-1 Annex F Method 1 shading, diffuse "
               "radiation not shaded).</div>")
    return vr.page("ASHRAE Standard 140-2023 — Section 7 validation",
                   "Building thermal envelope and fabric load tests, pybuildingenergy", body,
                   "Generated by examples/ashrae140_bestest/generate_reports.py.")


def main() -> None:
    frames, results = {}, {}
    for case_number in list(cd.LOW_MASS_ACCEPTANCE) + list(cd.HIGH_MASS_ACCEPTANCE):
        print(f"simulating Case {case_number}", flush=True)
        frames[case_number] = simulate(case_number)
        results[case_number] = summary(frames[case_number])
    ff_frames, ff = {}, {}
    for case_number in FREE_FLOAT_CASES:
        print(f"simulating Case {case_number}FF", flush=True)
        ff_frames[case_number] = simulate(case_number, free_float=True)
        ff[case_number] = summary(ff_frames[case_number])
    rows = [r for n in results for r in range_rows(n, results)]
    for case_number, df in frames.items():
        vr.write(REPORT_DIR / f"case_{case_number}.html", case_page(case_number, df, results))
    for case_number, df in ff_frames.items():
        vr.write(REPORT_DIR / f"case_{case_number}FF.html", case_page(case_number, df, results, free_float=True))
    index = vr.write(REPORT_DIR / "index.html", index_page(results, ff, rows))
    print(f"written {index}")


if __name__ == "__main__":
    main()
