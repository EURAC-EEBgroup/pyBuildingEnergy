"""HTML report of the ASHRAE Standard 140-2023 Section 7.2.3 in-depth cases.

One page, ``reports/in_depth.html``: inputs of the 25 cases, annual loads and peak
loads, the differences between each case and its base (Informative Table 7-33),
and a rotatable 3D view of each distinct test-cell geometry. The standard gives no
acceptance range for these cases, so nothing is marked pass or fail.

Usage:
    PYTHONPATH=src python examples/ashrae140_bestest/generate_in_depth_report.py [--solar-distribution area|table]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR.parent.parent / "src"))
sys.path.insert(0, str(EXAMPLES_DIR.parent))
sys.path.insert(0, str(EXAMPLES_DIR))

from geometry_3d import _IN_DEPTH_GEOMETRY, in_depth_caption, in_depth_figure  # noqa: E402
from in_depth_cases import CASE_SPECS, DIFFERENCES, IN_DEPTH_CASES  # noqa: E402
from run_in_depth import differences, run_all  # noqa: E402
import validation_report_html as vr  # noqa: E402

REPORT_DIR = EXAMPLES_DIR / "reports"

FILL_TEXT = {"hc": "High-conductance elements", "solid": "Solid wall", "south": "South windows",
             "eastwest": "East/west windows"}
SHADE_TEXT = {None: "—", "overhang": "South overhang", "overhang_fins": "Overhang + fins"}
COEF_TEXT = {"default": "Tables 7-7 / 7-9 (convective)", "table_7_40": "Table 7-40",
             "table_7_41": "Table 7-41", "table_7_46": "Table 7-46 (combined)"}
DIST_TEXT = {None: "—", "270": "Table 7-43", "280": "Table 7-44", "300": "Table 7-45"}


def _coefficients_text(spec) -> str:
    interior = COEF_TEXT[spec["int_coef"]]
    exterior = COEF_TEXT[spec["ext_coef"]]
    if spec["int_coef"] == spec["ext_coef"] == "default":
        return interior
    return f"int. {interior}; ext. {exterior}"


def inputs_table() -> str:
    rows = []
    for n in IN_DEPTH_CASES:
        s = CASE_SPECS[n]
        rows.append([
            n, "High" if s["mass"] == "heavy" else "Low", FILL_TEXT[s["fill"]], SHADE_TEXT[s.get("shade")],
            f"{s['ach']:g}", f"{s['gains']:g}", f"{s['ext_abs']:g}",
            f"{s['eps_int']:g} / {s['eps_ext']:g}", s["tstat"], _coefficients_text(s),
            DIST_TEXT[s["dist"]],
        ])
    return vr.table(["Case", "Mass", "South wall 12 m2 / openings", "Shading", "Infiltration ach",
                     "Internal gains W", "Ext. solar abs.", "IR emittance int / ext", "Thermostat",
                     "Surface coefficients", "Interior solar distribution"], rows)


def results_table(results: dict) -> str:
    rows = [[n, vr.fmt(r["annual_heating_MWh"], 3), vr.fmt(r["annual_cooling_MWh"], 3),
             vr.fmt(r["peak_heating_kW"], 3), vr.fmt(r["peak_cooling_kW"], 3)]
            for n, r in sorted(results.items())]
    return vr.table(["Case", "Heating MWh", "Cooling MWh", "Peak heating kW", "Peak cooling kW"], rows)


def differences_section(results: dict) -> str:
    rows_data = differences(results)
    rows = [[f"{r['case']} − {r['base']}", r["feature"],
             vr.fmt(r["d_heating_MWh"], 3), vr.fmt(r["d_cooling_MWh"], 3)] for r in rows_data]
    labels = [f"{r['case']} − {r['base']}" for r in rows_data]
    fig = go.Figure([
        go.Bar(name="Δ Heating", x=labels, y=[r["d_heating_MWh"] for r in rows_data],
               marker_color=vr.COLORS["heating"]),
        go.Bar(name="Δ Cooling", x=labels, y=[r["d_cooling_MWh"] for r in rows_data],
               marker_color=vr.COLORS["cooling"]),
    ])
    fig.update_layout(barmode="group", yaxis_title="MWh/year", title="Difference from the base case")
    return (vr.table(["Case − base", "Tested feature", "Δ Heating MWh", "Δ Cooling MWh"], rows)
            + vr.figure(fig, 420))


def geometry_section() -> str:
    groups: dict[tuple, list[int]] = {}
    for n in IN_DEPTH_CASES:
        key = (*_IN_DEPTH_GEOMETRY[n], n in (800, 810))
        groups.setdefault(key, []).append(n)
    parts = []
    for cases in groups.values():
        label = ", ".join(str(n) for n in cases)
        fig = in_depth_figure(cases[0])
        parts.append(f"<details><summary><b>Cases {label}</b></summary>"
                     f"<p class='muted'>{vr.esc(in_depth_caption(cases[0]))}</p>"
                     f"{vr.figure(fig, 460)}</details>")
    return "".join(parts)


def page(results: dict, solar_distribution: str) -> str:
    note = ("The interior solar distribution fractions of Tables 7-43 to 7-45 are applied "
            "(Sections 7.2.3.9.3.2, 7.2.3.10.2.2, 7.2.3.12.2.2): the engine has no interior "
            "solar absorptance input, so this is the only way to represent Cases 270, 280, 440 and 810."
            if solar_distribution == "table" else
            "The engine's area-weighted interior solar distribution is used: the interior solar "
            "absorptance cannot be told apart, so 280 equals 270, 440 equals 600 and 810 equals 900 "
            "in the solar part of the calculation.")
    body = (
        "<div class='callout'><b>Diagnostic cases.</b> ASHRAE 140-2023 gives no acceptance range for "
        "Section 7.2.3: each case changes one input from a base case (Informative Table 7-33) and what "
        "matters is the difference. The example results of other programs published with the standard "
        "are not reproduced here. " + vr.esc(note) + "</div>"
        "<h2>Inputs</h2>" + inputs_table()
        + "<h2>Annual results</h2>"
          "<p class='muted'>Cases 600 and 900 (Section 7.2.1 / 7.2.2.2) are included as bases.</p>"
        + results_table(results)
        + "<h2>Differences from the base cases</h2>" + differences_section(results)
        + "<h2>Geometry</h2>" + geometry_section()
    )
    return vr.page("ASHRAE Standard 140-2023 — In-depth cases (Section 7.2.3)",
                   "Single-feature diagnostic cases, pybuildingenergy. <a href='index.html'>Section 7 cases</a>",
                   body, "Generated by examples/ashrae140_bestest/generate_in_depth_report.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solar-distribution", choices=("area", "table"), default="table")
    args = parser.parse_args()
    results = run_all(args.solar_distribution)
    path = vr.write(REPORT_DIR / "in_depth.html", page(results, args.solar_distribution))
    print(f"written {path}")


if __name__ == "__main__":
    main()
