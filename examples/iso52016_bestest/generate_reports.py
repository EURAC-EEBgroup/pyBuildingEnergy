"""Generate one HTML report per EN ISO 52016-1:2017 verification case
(clause 7.2.3, Table 27: 600, 640, 900, 940, 600FF, 900FF) plus an index page.

Each report shows the clause 7.2.4 outputs (monthly and annual heating and
cooling needs, monthly mean operative temperature, peak loads, hourly
values on January 4 and July 27).

The reference values of Tables 28-34 are ISO copyright and are not
distributed (see iso_reference_results.py). By default the pages show the
simulation results, the annual deviations from the reference and the table
of the standard each reference is printed in, but not the reference values.
With the standard and the local reference file, ``--include-reference-values``
writes the full side-by-side comparison for local use; do not publish
those pages.

ISO 52016-1 defines no pass/fail tolerance; the +/-10% flag on annual needs
is the tolerance of tests/test_iso52016_bestest.py.

Usage:
    PYTHONPATH=src python examples/iso52016_bestest/generate_reports.py [--include-reference-values]
Output: examples/iso52016_bestest/reports/index.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR.parent.parent / "src"))
sys.path.insert(0, str(EXAMPLES_DIR.parent))
sys.path.insert(0, str(EXAMPLES_DIR))

import iso_reference_results as ref  # noqa: E402
from iso_case_definitions import CASE_IDS, build_case  # noqa: E402
from run_comparison import ANNUAL_TOLERANCE, CASE_INPUTS, simulate  # noqa: E402
import validation_report_html as vr  # noqa: E402
from geometry_3d import building_figure, geometry_caption  # noqa: E402

REPORT_DIR = EXAMPLES_DIR / "reports"
STD = "EN ISO 52016-1:2017"
DESCRIPTIONS = {
    "600": "Test 1 — lightweight, continuous thermostat 20/27 degC",
    "640": "Test 2 — lightweight, intermittent heating (10 degC from 23:00 to 07:00)",
    "900": "Test 3 — heavyweight, continuous thermostat 20/27 degC",
    "940": "Test 4 — heavyweight, intermittent heating (10 degC from 23:00 to 07:00)",
    "600FF": "Test 5 — lightweight, free floating",
    "900FF": "Test 6 — heavyweight, free floating",
}
NOT_REPRODUCED = (
    "<div class='callout'><b>Reference values are not reproduced.</b> They are ISO copyright and are "
    f"printed in {STD}, clause 7.2.4: Table 28 (heating needs), Table 29 (cooling needs), Table 30 "
    "(mean operative temperature), Table 31 (peak loads), Table 32 (free-floating extremes), Table 33 "
    "(hourly loads, January 4) and Table 34 (hourly operative temperature, January 4). Owning the "
    "standard is enough: enter them in <code>Standards/local_reference_data/iso52016_1_2017_tables_28_34.json</code> "
    "(format: <code>examples/iso52016_bestest/iso_reference_tables.template.json</code>) and run "
    "<code>generate_reports.py --include-reference-values</code> for the full comparison.</div>")


def dev_pct(value, reference):
    return 100.0 * (value - reference) / reference if reference else float("nan")


def annual_status(value, reference) -> str:
    if reference is None:
        return "N/A"
    return "PASS" if abs(dev_pct(value, reference)) <= 100 * ANNUAL_TOLERANCE else "FAIL"


def tol_badge(status: str) -> str:
    if status == "N/A":
        return vr.badge("N/A", "reference not installed")
    return vr.badge(status, "within +/-10%" if status == "PASS" else "outside +/-10%")


def summary(case_id: str, df: pd.DataFrame) -> dict:
    s = {"heating_kWh": df["Q_H"].sum() / 1e3, "cooling_kWh": df["Q_C"].sum() / 1e3,
         "T_op_mean": df["T_op"].mean(), "T_op_max": df["T_op"].max(), "T_op_min": df["T_op"].min(),
         "peak_heating_kW": df["Q_H"].max() / 1e3, "peak_cooling_kW": df["Q_C"].max() / 1e3}
    s["checks"] = []
    if not case_id.endswith("FF"):
        for metric, table, tab in (("heating", ref.ANNUAL_HEATING_KWH_ALL, "Table 28"),
                                   ("cooling", ref.ANNUAL_COOLING_KWH_ALL, "Table 29")):
            value, r = s[f"{metric}_kWh"], table.get(case_id)
            s["checks"].append((metric, value, r, dev_pct(value, r) if r else None, annual_status(value, r), tab))
    t_ref = ref.ANNUAL_OPERATIVE_TEMPERATURE_C.get(case_id)
    s["T_op_dev"] = None if t_ref is None else s["T_op_mean"] - t_ref
    return s


def _day(df, month, day):
    return df[(df.index.month == month) & (df.index.day == day)]


def _annual_note(check) -> str:
    metric, value, r, d, status, tab = check
    if r is None:
        return f"{STD} {tab} (reference not installed)"
    return f"{d:+.1f}% vs {STD} {tab} " + tol_badge(status)


def case_page(case_id: str, df: pd.DataFrame, exact_climate: bool, show_ref: bool) -> str:
    s = summary(case_id, df)
    free_float = case_id.endswith("FF")
    g = df.groupby(df.index.month)
    body = ""
    if free_float:
        t_dev = "" if s["T_op_dev"] is None else f"{s['T_op_dev']:+.2f} K vs "
        ff_ref = ref.FREE_FLOAT_OPERATIVE_TEMPERATURE_C.get(case_id) if show_ref else None
        body += vr.cards([
            ("Annual mean T_op", f"{s['T_op_mean']:.2f} degC", f"{t_dev}{STD} Table 30"),
            ("Annual maximum T_op", f"{s['T_op_max']:.2f} degC",
             f"Table 32: {ff_ref[0]} degC" if ff_ref else f"reference: {STD} Table 32"),
            ("Annual minimum T_op", f"{s['T_op_min']:.2f} degC",
             f"Table 32: {ff_ref[1]} degC" if ff_ref else f"reference: {STD} Table 32"),
            ("Acceptance", vr.badge("N/A"), "ISO 52016-1 gives no tolerance"),
        ])
    else:
        peaks = ref.PEAK_LOAD_KW.get(case_id) if show_ref else None
        body += vr.cards([
            ("Annual heating", f"{s['heating_kWh']:,.0f} kWh", _annual_note(s["checks"][0])),
            ("Annual cooling", f"{s['cooling_kWh']:,.0f} kWh", _annual_note(s["checks"][1])),
            ("Peak heating", f"{s['peak_heating_kW']:.3f} kW",
             f"Table 31: {peaks[0]} kW" if peaks else f"reference: {STD} Table 31"),
            ("Peak cooling", f"{s['peak_cooling_kW']:.3f} kW",
             f"Table 31: {peaks[1]} kW" if peaks else f"reference: {STD} Table 31"),
        ])
        if case_id == "900":
            body += ("<div class='callout'>Case 900 is compared with corrected references for Tables 28/29: "
                     "the printed Case 900 heating and cooling columns are swapped (see "
                     "<code>iso_reference_results.py</code> and the EPB Center collection of comments on "
                     "EN ISO 52016-1:2017, comment 28).</div>")
    if not show_ref:
        body += NOT_REPRODUCED

    # 3D view of the test cell: ISO cases 600/640/600FF and 900/940/900FF share the Case 600 / 900 cell.
    mass, control = CASE_INPUTS[case_id]
    cell = 600 if mass == "light" else 900
    free_floating = control == "free_floating"
    body += ("<h2>Geometry</h2>"
             f"<p class='muted'>{vr.esc(geometry_caption(cell, free_floating))} "
             + ("Intermittent thermostat: heating 10 degC from 23:00 to 07:00 (no geometry change). "
                if control == "intermittent" else "")
             + "Drag to rotate, scroll to zoom.</p>"
             + vr.figure(building_figure(cell, heavy=mass == "heavy", free_float=free_floating), height=520))

    # Monthly needs (Tables 28-29)
    if not free_float:
        ref_h = ref.MONTHLY_HEATING_KWH.get(case_id) if show_ref else None
        ref_c = ref.MONTHLY_COOLING_KWH.get(case_id) if show_ref else None
        fig = go.Figure()
        fig.add_bar(x=vr.MONTHS, y=(g["Q_H"].sum() / 1e3).to_list(), name="Heating (pybuildingenergy)",
                    marker_color=vr.COLORS["heating"])
        fig.add_bar(x=vr.MONTHS, y=(g["Q_C"].sum() / 1e3).to_list(), name="Cooling (pybuildingenergy)",
                    marker_color=vr.COLORS["cooling"])
        if ref_h and ref_c:
            fig.add_scatter(x=vr.MONTHS, y=ref_h, name="Heating (Table 28)", mode="markers+lines",
                            line=dict(color=vr.COLORS["heating"], dash="dot"))
            fig.add_scatter(x=vr.MONTHS, y=ref_c, name="Cooling (Table 29)", mode="markers+lines",
                            line=dict(color=vr.COLORS["cooling"], dash="dot"))
        fig.update_layout(title="Monthly sensible energy needs", barmode="group", yaxis_title="kWh")
        if ref_h and ref_c:
            headers = ["Month", "Heating kWh", "Table 28", "Diff", "Cooling kWh", "Table 29", "Diff"]
            rows = [[vr.MONTHS[m - 1], vr.fmt(g["Q_H"].sum()[m] / 1e3, 0), ref_h[m - 1],
                     vr.fmt(g["Q_H"].sum()[m] / 1e3 - ref_h[m - 1], 0), vr.fmt(g["Q_C"].sum()[m] / 1e3, 0),
                     ref_c[m - 1], vr.fmt(g["Q_C"].sum()[m] / 1e3 - ref_c[m - 1], 0)] for m in range(1, 13)]
        else:
            headers = ["Month", "Heating kWh", "Cooling kWh"]
            rows = [[vr.MONTHS[m - 1], vr.fmt(g["Q_H"].sum()[m] / 1e3, 0), vr.fmt(g["Q_C"].sum()[m] / 1e3, 0)]
                    for m in range(1, 13)]
            rows.append(["Annual", vr.fmt(s["heating_kWh"], 0), vr.fmt(s["cooling_kWh"], 0)])
        body += (f"<h2>Monthly heating and cooling needs (reference: {STD} Tables 28-29)</h2>"
                 + vr.figure(fig) + vr.table(headers, rows))

    # Monthly operative temperature (Table 30)
    t_sim = g["T_op"].mean().to_list()
    t_ref = ref.MONTHLY_OPERATIVE_TEMPERATURE_C.get(case_id) if show_ref else None
    fig = go.Figure()
    fig.add_scatter(x=vr.MONTHS, y=t_sim, name="pybuildingenergy", line=dict(color=vr.COLORS["temperature"]))
    if t_ref:
        fig.add_scatter(x=vr.MONTHS, y=t_ref, name="Table 30", mode="markers+lines",
                        line=dict(color=vr.COLORS["reference"], dash="dot"))
    fig.update_layout(title="Monthly mean operative temperature", yaxis_title="degC")
    if t_ref:
        rows = [[vr.MONTHS[i], vr.fmt(t_sim[i], 2), t_ref[i], vr.fmt(t_sim[i] - t_ref[i], 2)] for i in range(12)]
        headers = ["Month", "T_op degC", "Table 30", "Diff K"]
    else:
        rows = [[vr.MONTHS[i], vr.fmt(t_sim[i], 2)] for i in range(12)]
        rows.append(["Annual", vr.fmt(s["T_op_mean"], 2)])
        headers = ["Month", "T_op degC"]
    body += (f"<h2>Monthly mean operative temperature (reference: {STD} Table 30)</h2>"
             + vr.figure(fig) + vr.table(headers, rows))

    # Hourly January 4 (Tables 33-34) and July 27 (no reference printed)
    for (month, day, label) in ((1, 4, "January 4"), (7, 27, "July 27")):
        d = _day(df, month, day)
        hours = list(range(1, len(d) + 1))
        ref_loads = ref.JANUARY_4_HOURLY_LOAD_WH.get(case_id) if (month == 1 and show_ref) else None
        ref_temp = ref.JANUARY_4_HOURLY_OPERATIVE_TEMPERATURE_C.get(case_id) if (month == 1 and show_ref) else None
        fig = go.Figure()
        if not free_float:
            net = (d["Q_H"] - d["Q_C"]) / 1e3
            fig.add_bar(x=hours, y=net.to_list(), name="Heating (+) / cooling (-) kWh",
                        marker_color=[vr.COLORS["heating"] if v >= 0 else vr.COLORS["cooling"] for v in net])
            if ref_loads:
                fig.add_scatter(x=hours, y=[v / 1e3 for v in ref_loads], name="Table 33", mode="markers",
                                marker=dict(color=vr.COLORS["reference"], size=8, symbol="x"))
        fig.add_scatter(x=hours, y=d["T_op"].to_list(), name="T_op degC", yaxis="y2",
                        line=dict(color=vr.COLORS["temperature"]))
        if ref_temp:
            fig.add_scatter(x=hours, y=ref_temp, name="Table 34 T_op", yaxis="y2", mode="markers",
                            marker=dict(color=vr.COLORS["reference"], size=7))
        fig.add_scatter(x=hours, y=d["T_ext"].to_list(), name="Outdoor T degC", yaxis="y2",
                        line=dict(color=vr.COLORS["outdoor"], dash="dot"))
        fig.update_layout(title=f"{label}: hourly values (hour 1 = 00:00-01:00)", xaxis_title="Hour",
                          yaxis_title="kWh", yaxis2=dict(title="degC", overlaying="y", side="right"))
        headers, cols = ["Hour"], []
        if not free_float:
            headers.append("Load kWh (+H / -C)")
            cols.append([vr.fmt(v, 3) for v in (d["Q_H"] - d["Q_C"]) / 1e3])
            if ref_loads:
                headers.append("Table 33 kWh")
                cols.append([vr.fmt(v / 1e3, 3) for v in ref_loads])
        headers.append("T_op degC")
        cols.append([vr.fmt(v, 2) for v in d["T_op"]])
        if ref_temp:
            headers.append("Table 34 degC")
            cols.append([vr.fmt(v, 2) for v in ref_temp])
        rows = [[h] + [c[i] for c in cols] for i, h in enumerate(hours)]
        if month == 1:
            note = (f"<p class='muted'>Reference: {STD} Table 33 (loads) and Table 34 "
                    "(free-floating T_op).</p>")
        else:
            note = ("<p class='muted'>Clause 7.2.4 requires these hourly values; the standard prints no "
                    "reference for July 27.</p>")
        body += f"<h2>Hourly outputs, {label}</h2>" + note + vr.figure(fig) + vr.table(headers, rows)

    mass, control = CASE_INPUTS[case_id]
    bui = build_case(mass, control)
    rows = [[x["name"], vr.fmt(x["area"], 1), vr.fmt(x["u_value"], 4), vr.fmt(x.get("thermal_capacity", 0) / 1000, 1),
             x.get("construction_class", "-"), vr.fmt(x.get("sky_view_factor"), 2), vr.fmt(x.get("solar_absorptance"), 2)]
            for x in bui["building_surface"]]
    climate = ("ISO 52016-1 BESTEST climate spreadsheet (clause 7.2.2.12)" if exact_climate
               else "Denver TMY3 EPW approximation, not the standard climate")
    body += ("<h2>Inputs applied</h2><p>Operative-temperature control (engine default, clause 7.2.2.1); "
             "infiltration 0.41 ach (clause 7.2.2.14); ISO 52016-1 Table 25 surface coefficients; window "
             "g_gl;n = 0.789, F_w = 0.90, U_w = 2.984 W/(m2 K); floor to outdoor air (Table 23 footnote a). "
             f"Weather: {climate}.</p>"
             + vr.table(["Surface", "Area m2", "U W/m2K", "kappa kJ/m2K", "Class", "F_sky", "a_sol"], rows))
    subtitle = f"{vr.esc(DESCRIPTIONS[case_id])} — clause 7.2.3, Table 27. <a href='index.html'>All cases</a>"
    return vr.page(f"ISO 52016-1:2017 Case {case_id}", subtitle, body,
                   "Generated by examples/iso52016_bestest/generate_reports.py.")


def index_page(summaries: dict, exact_climate: bool, show_ref: bool) -> str:
    rows = []
    for case_id, s in summaries.items():
        cells = [f"<a href='case_{case_id}.html'>{case_id}</a>", vr.esc(DESCRIPTIONS[case_id])]
        if s["checks"]:
            for metric, value, r, d, status, tab in s["checks"]:
                if r is None:
                    cells.append(f"{value:,.0f} (reference: {tab})")
                elif show_ref:
                    cells.append(f"{value:,.0f} vs {r:,} ({d:+.1f}%, {tab}) {tol_badge(status)}")
                else:
                    cells.append(f"{value:,.0f} ({d:+.1f}% vs {tab}) {tol_badge(status)}")
        else:
            cells += ["—", "—"]
        dev = "" if s["T_op_dev"] is None else f" ({s['T_op_dev']:+.2f} K vs Table 30)"
        cells.append(f"{s['T_op_mean']:.2f}{dev}")
        rows.append(cells)
    checks = [c for s in summaries.values() for c in s["checks"] if c[2] is not None]
    devs = [abs(s["T_op_dev"]) for s in summaries.values() if s["T_op_dev"] is not None]
    body = vr.cards([
        ("Annual needs within +/-10%", f"{sum(c[4] == 'PASS' for c in checks)}/{len(checks)}" if checks else "—",
         f"Cases 600, 640, 900, 940 vs {STD} Tables 28-29"),
        ("Max |Delta| annual mean T_op", f"{max(devs):.2f} K" if devs else "—",
         f"All six cases vs {STD} Table 30"),
        ("Weather", "ISO climate" if exact_climate else "TMY3 approx.", "clause 7.2.2.12"),
    ])
    body += ("<div class='callout'>ISO 52016-1 clause 7.2 defines no pass/fail tolerance: the results and "
             "their deviations from the reference values are to be reported. The +/-10% flag is the "
             "tolerance used by <code>tests/test_iso52016_bestest.py</code>.</div>")
    if not show_ref:
        body += NOT_REPRODUCED
    body += vr.table(["Case", "Description", "Annual heating kWh", "Annual cooling kWh",
                      "Annual mean T_op degC"], rows, raw_columns=(0, 1, 2, 3))
    return vr.page("ISO 52016-1:2017 — hourly method verification cases",
                   "Clause 7.2, Tests 1-6 (BESTEST 600, 640, 900, 940, 600FF, 900FF), pybuildingenergy", body,
                   "Generated by examples/iso52016_bestest/generate_reports.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-reference-values", action="store_true",
                        help="show the Table 28-34 values (local use only; ISO copyright)")
    args = parser.parse_args()
    show_ref = bool(args.include_reference_values and ref.REFERENCES_AVAILABLE)
    summaries, exact = {}, True
    for case_id in CASE_IDS.values():
        print(f"simulating ISO Case {case_id}", flush=True)
        df, exact_climate = simulate(case_id)
        exact = exact and exact_climate
        summaries[case_id] = summary(case_id, df)
        vr.write(REPORT_DIR / f"case_{case_id}.html", case_page(case_id, df, exact_climate, show_ref))
    order = {c: i for i, c in enumerate(ref.CASES)}
    summaries = dict(sorted(summaries.items(), key=lambda kv: order[kv[0]]))
    index = vr.write(REPORT_DIR / "index.html", index_page(summaries, exact, show_ref))
    print(f"written {index}")


if __name__ == "__main__":
    main()
