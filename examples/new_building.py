import sys
from pathlib import Path

# Ensure local package import works when running the script directly
# (package sources live in ../src).
EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (SRC_DIR, PROJECT_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import pybuildingenergy as pybui
import numpy as np
import pandas as pd 
import plotly.express as px 
from pyecharts import options as pye_opts
from pyecharts.charts import Line, Page, Sankey
from pybuildingenergy.source.utils import *
from pybuildingenergy.source.check_input import sanitize_and_validate_BUI, check_heating_system_inputs
from pybuildingenergy.source.graphs import Graphs_and_report
from pybuildingenergy.source.graphs import *
from pybuildingenergy.source.iso_15316_1 import HeatingSystemCalculator
from pybuildingenergy.source.generate_profile import HourlyProfileGenerator, get_country_code_from_latlon
from pybuildingenergy.source.DHW import *
from pybuildingenergy.source.ventilation import *
from pybuildingenergy.source.table_iso_16798_1 import *

WEATHER_CANDIDATES = [
    EXAMPLES_DIR / "2050_Athens.epw",
    EXAMPLES_DIR / "2020_Milan.epw",
]
# WEATHER_FILE = next((p for p in WEATHER_CANDIDATES if p.exists()), None)
WEATHER_FILE = None
WEATHER_SOURCE = "epw" if WEATHER_FILE is not None else "pvgis"
print(WEATHER_FILE)

GENERATE_EXTRA_REPORTS = False

def _run_iso52016(building_obj):
    kwargs = {"weather_source": WEATHER_SOURCE}
    if WEATHER_SOURCE == "epw":
        kwargs["path_weather_file"] = str(WEATHER_FILE)

    out = ISO52016.Temperature_and_Energy_needs_calculation(building_obj, **kwargs)
    if isinstance(out, tuple) and len(out) == 3:
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1], {}
    raise RuntimeError("Unexpected output format from Temperature_and_Energy_needs_calculation")


def _export_hvac_flow_results(hourly_sim: pd.DataFrame, hvac_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Export a compact stage-by-stage HVAC results table.

    The output combines:
    - building energy needs from ISO52016;
    - emission, distribution and generation outputs from ISO 15316-1 / EN 15316 blocks.
    """

    if not isinstance(hourly_sim, pd.DataFrame) or not isinstance(hvac_df, pd.DataFrame):
        raise TypeError("hourly_sim and hvac_df must be pandas DataFrames.")

    stage_df = pd.DataFrame(index=hourly_sim.index)

    building_cols = [
        "Q_HC",
        "Q_H",
        "Q_C",
        "T_op0",
        "T_air",
        "T_op",
        "T_ext",
    ]
    for col in building_cols:
        if col in hourly_sim.columns:
            stage_df[f"building_{col}"] = hourly_sim[col]

    hvac_cols = [
        "Q_h(kWh)",
        "QH_em_i_in(kWh)",
        "QH_dis_i_req(kWh)",
        "QH_dis_i_in(kWh)",
        "QH_gen_out(kWh)",
        "EHW_gen_in(kWh)",
        "EHW_gen_aux(kWh)",
        "QW_gen_i_ls_rbl_H(kWh)",
        "Q_w_dis_i_ls(kWh)",
        "Q_w_dis_i_aux(kWh)",
        "Q_w_dis_i_ls_rbl_H(kWh)",
        "ΦH_em_eff(kW)",
        "θH_em_flow(°C)",
        "θH_em_ret(°C)",
        "θH_dis_flw(°C)",
        "θH_dis_ret(°C)",
        "θX_gen_cr_flw(°C)",
        "θX_gen_cr_ret(°C)",
        "V_H_em_eff(m3/h)",
        "V_H_dis(m3/h)",
        "V_H_gen(m3/h)",
        "efficiency_gen(%)",
        "emission_calculation_mode",
    ]
    for col in hvac_cols:
        if col in hvac_df.columns:
            stage_df[f"hvac_{col}"] = hvac_df[col]

    out_path = Path(output_dir) / "hvac_stage_results.csv"
    stage_df.to_csv(out_path)
    print(f"[info] HVAC stage results written to {out_path}")
    return stage_df


def _build_sankey_consumption_report(
    hourly_sim: pd.DataFrame,
    hvac_df: pd.DataFrame,
    output_dir: str,
    dhw_useful_df: pd.DataFrame | None = None,
    dhw_distribution_df: pd.DataFrame | None = None,
    dhw_storage_df: pd.DataFrame | None = None,
) -> Path:
    """Create a Sankey HTML report for the HVAC heating and DHW energy flow chains.

    Heating and DHW are shown as separate branches when DHW data are provided.
    Each branch tracks demand -> emission/distribution/storage -> generator
    input, with losses attached at every subsystem stage.
    """

    def _sum_col(df: pd.DataFrame, col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()) if col in df.columns else 0.0

    same_generator_for_heating_and_dhw = bool(
        INPUT_SYSTEM_HVAC.get("same_generator_for_heating_and_dhw", True)
    )

    q_h_iso = _sum_col(hvac_df, "Q_h(kWh)")
    q_h_fuel = _sum_col(hvac_df, "EHW_gen_in(kWh)")
    q_h_gen_aux = _sum_col(hvac_df, "EHW_gen_aux(kWh)")
    q_h_gen_out = _sum_col(hvac_df, "QH_gen_out(kWh)")
    q_h_dis_ls = _sum_col(hvac_df, "Q_w_dis_i_ls(kWh)")
    q_h_dis_aux = _sum_col(hvac_df, "Q_w_dis_i_aux(kWh)")
    q_h_em_in = _sum_col(hvac_df, "QH_em_i_in(kWh)")
    q_h_em_ls = _sum_col(hvac_df, "QH_em_ls(kWh)") if "QH_em_ls(kWh)" in hvac_df.columns else 0.0
    q_h_em_aux = _sum_col(hvac_df, "W_H_em_aux(kWh)") if "W_H_em_aux(kWh)" in hvac_df.columns else 0.0
    q_h_gen_standby = max(q_h_fuel - q_h_gen_out - q_h_gen_aux, 0.0)
    q_h_uncovered = max(q_h_iso - q_h_em_in, 0.0)
    coverage_pct = 100.0 * q_h_em_in / q_h_iso if q_h_iso > 0 else 0.0

    q_w_demand = _sum_col(dhw_useful_df, "Q_W_kWh") if dhw_useful_df is not None else 0.0
    q_w_dis_in = _sum_col(dhw_distribution_df, "Q_W_dis_in_kWh") if dhw_distribution_df is not None else q_w_demand
    q_w_dis_ls = _sum_col(dhw_distribution_df, "Q_W_dis_ls_kWh") if dhw_distribution_df is not None else 0.0
    q_w_dis_aux = _sum_col(dhw_distribution_df, "W_W_dis_aux_kWh") if dhw_distribution_df is not None else 0.0
    if dhw_storage_df is not None:
        q_w_sto_in = _sum_col(dhw_storage_df, "Q_W_sto_in_kWh")
        q_w_sto_out = _sum_col(dhw_storage_df, "Q_W_sto_out_kWh")
        q_w_sto_ls = _sum_col(dhw_storage_df, "Q_W_sto_ls_kWh")
        q_w_sto_aux = _sum_col(dhw_storage_df, "W_W_sto_aux_kWh")
    else:
        q_w_sto_in = q_w_dis_in
        q_w_sto_out = q_w_dis_in
        q_w_sto_ls = 0.0
        q_w_sto_aux = 0.0
    q_w_final_energy = q_w_sto_in if dhw_storage_df is not None else q_w_dis_in
    nodes = [
        {"name": "Heating useful energy"},
        {"name": "Heating emission"},
        {"name": "Heating distribution"},
        {"name": "Heating generation"},
        {"name": "DHW useful energy"},
        {"name": "DHW distribution"},
        {"name": "DHW storage"},
        {"name": "DHW generation"},
        {"name": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating"},
    ]
    if not same_generator_for_heating_and_dhw:
        nodes.append({"name": "Final energy DHW"})
    if q_h_em_ls > 0.1:
        nodes.append({"name": "Heating emission losses"})
    if q_h_em_aux > 0.1:
        nodes.append({"name": "Heating emission auxiliaries"})
    if q_h_gen_standby > 1.0:
        nodes.append({"name": "Heating generation losses"})
    if q_h_gen_aux > 0.1:
        nodes.append({"name": "Heating auxiliaries"})
    if q_h_dis_ls > 0.1:
        nodes.append({"name": "Heating distribution losses"})
    if q_h_dis_aux > 0.1:
        nodes.append({"name": "Heating distribution auxiliaries"})
    if q_w_sto_ls > 0.1:
        nodes.append({"name": "DHW storage losses"})
    if q_w_sto_aux > 0.1:
        nodes.append({"name": "DHW storage auxiliaries"})
    if q_w_dis_ls > 0.1:
        nodes.append({"name": "DHW distribution losses"})
    if q_w_dis_aux > 0.1:
        nodes.append({"name": "DHW distribution auxiliaries"})

    if same_generator_for_heating_and_dhw:
        links = [
            {"source": "Heating useful energy", "target": "Heating emission", "value": round(q_h_iso, 1)},
            {"source": "Heating emission", "target": "Heating distribution", "value": round(q_h_em_in, 1)},
            {"source": "Heating distribution", "target": "Heating generation", "value": round(q_h_gen_out, 1)},
            {"source": "Heating generation", "target": "Final energy", "value": round(q_h_fuel, 1)},
        ]
    else:
        links = [
            {"source": "Heating useful energy", "target": "Heating emission", "value": round(q_h_iso, 1)},
            {"source": "Heating emission", "target": "Heating distribution", "value": round(q_h_em_in, 1)},
            {"source": "Heating distribution", "target": "Heating generation", "value": round(q_h_gen_out, 1)},
            {"source": "Heating generation", "target": "Final energy Heating", "value": round(q_h_fuel, 1)},
        ]
    if q_h_em_ls > 0.1:
        links.append({"source": "Heating emission", "target": "Heating emission losses", "value": round(q_h_em_ls, 1)})
        links.append({"source": "Heating emission losses", "target": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating", "value": round(q_h_em_ls, 1)})
    if q_h_em_aux > 0.1:
        links.append({"source": "Heating emission", "target": "Heating emission auxiliaries", "value": round(q_h_em_aux, 1)})
        links.append({"source": "Heating emission auxiliaries", "target": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating", "value": round(q_h_em_aux, 1)})
    if q_h_gen_standby > 1.0:
        links.append({"source": "Heating generation", "target": "Heating generation losses", "value": round(q_h_gen_standby, 1)})
        links.append({"source": "Heating generation losses", "target": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating", "value": round(q_h_gen_standby, 1)})
    if q_h_gen_aux > 0.1:
        links.append({"source": "Heating generation", "target": "Heating auxiliaries", "value": round(q_h_gen_aux, 1)})
        links.append({"source": "Heating auxiliaries", "target": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating", "value": round(q_h_gen_aux, 1)})
    if q_h_dis_ls > 0.1:
        links.append({"source": "Heating distribution", "target": "Heating distribution losses", "value": round(q_h_dis_ls, 1)})
        links.append({"source": "Heating distribution losses", "target": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating", "value": round(q_h_dis_ls, 1)})
    if q_h_dis_aux > 0.1:
        links.append({"source": "Heating distribution", "target": "Heating distribution auxiliaries", "value": round(q_h_dis_aux, 1)})
        links.append({"source": "Heating distribution auxiliaries", "target": "Final energy" if same_generator_for_heating_and_dhw else "Final energy Heating", "value": round(q_h_dis_aux, 1)})

    final_dhw_target = "Final energy" if same_generator_for_heating_and_dhw else "Final energy DHW"
    if dhw_distribution_df is not None:
        links.append({"source": "DHW useful energy", "target": "DHW distribution", "value": round(q_w_demand, 1)})
        if dhw_storage_df is not None:
            links.append({"source": "DHW distribution", "target": "DHW storage", "value": round(q_w_dis_in, 1)})
            links.append({"source": "DHW storage", "target": "DHW generation", "value": round(q_w_sto_out, 1)})
            if q_w_sto_ls > 0.1:
                links.append({"source": "DHW storage", "target": "DHW storage losses", "value": round(q_w_sto_ls, 1)})
                links.append({"source": "DHW storage losses", "target": final_dhw_target, "value": round(q_w_sto_ls, 1)})
            if q_w_sto_aux > 0.1:
                links.append({"source": "DHW storage", "target": "DHW storage auxiliaries", "value": round(q_w_sto_aux, 1)})
                links.append({"source": "DHW storage auxiliaries", "target": final_dhw_target, "value": round(q_w_sto_aux, 1)})
            links.append({"source": "DHW generation", "target": final_dhw_target, "value": round(q_w_final_energy, 1)})
        else:
            links.append({"source": "DHW distribution", "target": "DHW generation", "value": round(q_w_final_energy, 1)})
            links.append({"source": "DHW generation", "target": final_dhw_target, "value": round(q_w_final_energy, 1)})
        if q_w_dis_ls > 0.1:
            links.append({"source": "DHW distribution", "target": "DHW distribution losses", "value": round(q_w_dis_ls, 1)})
            links.append({"source": "DHW distribution losses", "target": final_dhw_target, "value": round(q_w_dis_ls, 1)})
        if q_w_dis_aux > 0.1:
            links.append({"source": "DHW distribution", "target": "DHW distribution auxiliaries", "value": round(q_w_dis_aux, 1)})
            links.append({"source": "DHW distribution auxiliaries", "target": final_dhw_target, "value": round(q_w_dis_aux, 1)})

    subtitle = (
        f"Heating useful energy: {q_h_iso:.0f} kWh  |  "
        f"Heating emission input: {q_h_em_in:.0f} kWh ({coverage_pct:.1f}%)  |  "
        f"Heating emission losses: {q_h_em_ls:.0f} kWh  |  "
        f"Heating final energy: {q_h_fuel:.0f} kWh  |  "
        f"DHW useful energy: {q_w_demand:.0f} kWh  |  "
        f"DHW final energy: {q_w_final_energy:.0f} kWh"
    )

    used_nodes = {link["source"] for link in links} | {link["target"] for link in links}
    nodes = [node for node in nodes if node["name"] in used_nodes]

    sankey = (
        Sankey(init_opts=pye_opts.InitOpts(width="1150px", height="720px", page_title="HVAC Energy Sankey"))
        .add(
            series_name="Heating + DHW",
            nodes=nodes,
            links=links,
            linestyle_opt=pye_opts.LineStyleOpts(opacity=0.4, curve=0.5, color="source"),
            label_opts=pye_opts.LabelOpts(position="right", font_size=11),
            node_gap=14,
            node_width=16,
            pos_left="8%",
            pos_right="8%",
            pos_top="8%",
            pos_bottom="10%",
        )
        .set_global_opts(
            title_opts=pye_opts.TitleOpts(
                title="Energy flow heating + DHW — ISO 52016 + EN 15316",
                subtitle=subtitle,
            ),
            toolbox_opts=pye_opts.ToolboxOpts(is_show=True),
        )
    )

    out_path = Path(output_dir) / "hvac_energy_sankey.html"
    sankey.render(str(out_path))
    html = out_path.read_text(encoding="utf-8")
    html = html.replace(
        "</head>",
        """
<style>
html, body { width: 100%; height: 100%; margin: 0; padding: 0; background: #faf7f2; }
body { display: flex; justify-content: center; align-items: flex-start; }
.chart-container { margin: 0 auto !important; }
</style>
</head>
""",
    )
    html = html.replace('<body>', '<body><div style="width: 100%; display: flex; justify-content: center;">', 1)
    html = html.replace('</body>', '</div></body>', 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"[info] Sankey report written to {out_path}")
    return out_path


def _build_temperature_report(hvac_df: pd.DataFrame, output_dir: str) -> Path:
    """Create an HTML report with the water temperatures across subsystems."""

    if "timestamp" in hvac_df.columns:
        index = pd.to_datetime(hvac_df["timestamp"], errors="coerce")
    else:
        index = pd.RangeIndex(len(hvac_df))

    temp_cols = [
        ("hvac_θH_em_flow(°C)", "Emission supply"),
        ("hvac_θH_em_ret(°C)", "Emission return"),
        ("hvac_θH_dis_flw(°C)", "Distribution supply"),
        ("hvac_θH_dis_ret(°C)", "Distribution return"),
        ("hvac_θX_gen_cr_flw(°C)", "Generator supply"),
        ("hvac_θX_gen_cr_ret(°C)", "Generator return"),
    ]

    page = Page(layout=Page.SimplePageLayout)
    x_data = [str(x) for x in index]
    for col, title in temp_cols:
        if col not in hvac_df.columns:
            continue
        vals = pd.to_numeric(hvac_df[col], errors="coerce").fillna(method="ffill").fillna(method="bfill").tolist()
        chart = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis(title, vals, is_smooth=True, is_symbol_show=False)
            .set_global_opts(
                title_opts=pye_opts.TitleOpts(title=f"Water temperature - {title}"),
                xaxis_opts=pye_opts.AxisOpts(axislabel_opts=pye_opts.LabelOpts(rotate=45)),
                yaxis_opts=pye_opts.AxisOpts(name="°C"),
                datazoom_opts=[pye_opts.DataZoomOpts(type_="inside"), pye_opts.DataZoomOpts()],
            )
        )
        page.add(chart)

    out_path = Path(output_dir) / "hvac_temperatures.html"
    page.render(str(out_path))
    print(f"[info] Temperature report written to {out_path}")
    return out_path

# BUI = {
#     "building": {
#         "name": "test-cy",
#         "azimuth_relative_to_true_north": 41.8,
#         "latitude": 37.98880066730187,
#         "longitude": 23.733531819066098,
#         "exposed_perimeter": 40,
#         "height": 3,
#         "wall_thickness": 0.3,
#         "n_floors": 1,
#         "building_type_class": "Residential_apartment",
#         "adj_zones_present": False,
#         "number_adj_zone":2,
#         "net_floor_area": 100,
#         "construction_class": "class_i",
#     },
#     "adjacent_zones": [
#         {
#             "name":"adj_1",
#             "orientation_zone": {
#                 "azimuth": 0,
#             },
#             "area_facade_elements": np.array([20,60,30,30,50,50], dtype=object),
#             "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
#             "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
#             "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
#             'volume': 300, 
#             'building_type_class':'Residential_apartment',
#             'a_use':50 
#         },
#         {
#             "name":"adj_2",
#             "orientation_zone": {
#                 "azimuth": 180,
#             },
#             "area_facade_elements": np.array([20,60,30,30,50,50], dtype=object),
#             "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
#             "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
#             "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
#             'volume': 300, 
#             'building_type_class':'Residential_apartment',
#             'a_use':50 
#         }
#     ],
#     "building_surface": [
#         {
#             "name": "Roof surface",
#             "type": "opaque",
#             "area": 130,
#             "sky_view_factor": 1.0,
#             "u_value": 2.2,
#             "solar_absorptance": 0.4,
#             "thermal_capacity": 741500.0,
#             "orientation": {
#                 "azimuth": 0,
#                 "tilt": 0
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Opaque north surface",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.4,
#             "solar_absorptance": 0.4,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 0,
#                 "tilt": 90
#             },
#             "name_adj_zone": "adj_1"
#         },
#         {
#             "name": "Opaque south surface",
#             "type": "opaque",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.4,
#             "solar_absorptance": 0.4,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 180,
#                 "tilt": 90
#             },
#             "name_adj_zone": "adj_2"
#         },
#         {
#             "name": "Opaque east surface",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.2,
#             "solar_absorptance": 0.6,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 90,
#                 "tilt": 90
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Opaque west surface",
#             "type": "opaque",
#             "area": 30,
#             "sky_view_factor": 0.5,
#             "u_value": 1.2,
#             "solar_absorptance": 0.7,
#             "thermal_capacity": 1416240.0,
#             "orientation": {
#                 "azimuth": 270,
#                 "tilt": 90
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Slab to ground",
#             "type": "opaque",
#             "area": 100,
#             "sky_view_factor": 0.0,
#             "u_value": 1.6,
#             "solar_absorptance": 0.6,
#             "thermal_capacity": 405801,
#             "orientation": {
#                 "azimuth": 0,
#                 "tilt": 0
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Transparent east surface",
#             "type": "transparent",
#             "area": 4,
#             "sky_view_factor": 0.5,
#             "u_value": 5,
#             "g_value": 0.726,
#             "height": 2,
#             "width": 1,
#             "parapet": 1.1,
#             "orientation": {
#                 "azimuth": 90,
#                 "tilt": 90
#             },
#             "shading": False,
#             "shading_type": "horizontal_overhang",
#             "width_or_distance_of_shading_elements": 0.5,
#             "overhang_proprieties": {
#                 "width_of_horizontal_overhangs":1
#             },
#             "name_adj_zone": None
#         },
#         {
#             "name": "Transparent west surface",
#             "type": "transparent",
#             "area": 4,
#             "sky_view_factor": 0.5,
#             "u_value": 5,
#             "g_value": 0.726,
#             "height": 2,
#             "width": 1,
#             "parapet": 1.1,
#             "orientation": {
#                 "azimuth": 270,
#                 "tilt": 90
#             },
#             "shading": False,
#             "shading_type": "horizontal_overhang",
#             "width_or_distance_of_shading_elements": 0.5,
#             "overhang_proprieties": {
#                 "width_of_horizontal_overhangs":1
#             },
#             "name_adj_zone": None
#         }
#     ],
#     "units": {
#         "area": "m²",
#         "u_value": "W/m²K",
#         "thermal_capacity": "J/kgK",
#         "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
#         "tilt": "degrees (0=horizontal, 90=vertical)",
#         "internal_gain": "W/m²",
#         "internal_gain_profile": "Normalized to 0-1",
#         "HVAC_profile": "0: off, 1: on"
#     },
#     "building_parameters": {
#         "temperature_setpoints": {
#             "heating_setpoint": 20.0,
#             "heating_setback": 17.0,
#             "cooling_setpoint": 26.0,
#             "cooling_setback": 30.0,
#             "units": "°C"
#         },
#         "system_capacities": {
#             "heating_capacity": 10000000.0,
#             "cooling_capacity": 12000000.0,
#             "units": "W"
#         },
#         "airflow_rates": {
#             "infiltration_rate": 1.0,
#             "ventilation_rate_extra": 1.0,
#             "units": "ACH (air changes per hour)"
#         },
#         "internal_gains": [
#             {
#                 "name": "occupants",
#                 "full_load": 4.2,
#                 "weekday": [1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1.0,1.0],
#                 "weekend": [1.0,1.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1.0,1.0]
#             },
#             {
#                 "name": "appliances",
#                 "full_load": 3,
#                 "weekday": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
#                 "weekend": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
#             },
#             {
#                 "name": "lighting",
#                 "full_load": 3,
#                 "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
#                 "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
#             }
#         ],
#         "construction": {
#             "wall_thickness": 0.3,
#             "thermal_bridges": 2,
#             "units": "m (for thickness), W/mK (for thermal bridges)"
#         },
#         "climate_parameters": {
#             "coldest_month": 1,
#             "units": "1-12 (January-December)"
#         },
#         "heating_profile": {
#             "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#             "weekend": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#         },
#         "cooling_profile": {
#             "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#             "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
#         },
#         "ventilation_profile": {
#             "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
#             "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
#         }
#     }
# }
BUI = {
    "building": {
        "name": "Archetype_ITA_SFH_2010",
        "azimuth_relative_to_true_north": 0,
        "latitude": 41.9,
        "longitude": 12.5,
        "exposed_perimeter": 40,
        "height": 6,
        "wall_thickness": 0.35,
        "n_floors": 2,
        "building_type_class": "Residential_apartment",
        "adj_zones_present": False,
        "number_adj_zone": 0,
        "net_floor_area": 120,
        "construction_class": "class_i",
        "construction_year": "2010-today",
        "country": "Italy"
    },
    "adjacent_zones": [
        {
            "name": "adj_1",
            "orientation_zone": {
                "azimuth": 0.0
            },
            "area_facade_elements": [
                20,
                60,
                30,
                30,
                50,
                50
            ],
            "typology_elements": [
                "OP",
                "OP",
                "OP",
                "OP",
                "GR",
                "OP"
            ],
            "transmittance_U_elements": [
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.5156683855612851,
                1.162633192818565
            ],
            "orientation_elements": [
                "NV",
                "SV",
                "EV",
                "WV",
                "HOR",
                "HOR"
            ],
            "volume": 300.0,
            "building_type_class": "Residential_apartment",
            "a_use": 50.0
        },
        {
            "name": "adj_2",
            "orientation_zone": {
                "azimuth": 180.0
            },
            "area_facade_elements": [
                20,
                60,
                30,
                30,
                50,
                50
            ],
            "typology_elements": [
                "OP",
                "OP",
                "OP",
                "OP",
                "GR",
                "OP"
            ],
            "transmittance_U_elements": [
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.8196721311475411,
                0.5156683855612851,
                1.162633192818565
            ],
            "orientation_elements": [
                "NV",
                "SV",
                "EV",
                "WV",
                "HOR",
                "HOR"
            ],
            "volume": 300.0,
            "building_type_class": "Residential_apartment",
            "a_use": 50.0
        }
    ],
    "building_surface": [
        {
            "name": "Roof surface",
            "type": "opaque",
            "area": 130.0,
            "sky_view_factor": 1.0,
            "u_value": 2.2,
            "solar_absorptance": 0.4,
            "thermal_capacity": 741500.0,
            "orientation": {
                "azimuth": 0.0,
                "tilt": 0.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 13.0
        },
        {
            "name": "Opaque north surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 0.0,
                "tilt": 90.0
            },
            "name_adj_zone": "adj_1",
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Opaque south surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.4,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 180.0,
                "tilt": 90.0
            },
            "name_adj_zone": "adj_2",
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Opaque east surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.2,
            "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 90.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Opaque west surface",
            "type": "opaque",
            "area": 30.0,
            "sky_view_factor": 0.5,
            "u_value": 1.2,
            "solar_absorptance": 0.7,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 270.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 3.0
        },
        {
            "name": "Slab to ground",
            "type": "opaque",
            "area": 100.0,
            "sky_view_factor": 0.5,
            "u_value": 1.6,
            "solar_absorptance": 0.6,
            "thermal_capacity": 405801.0,
            "orientation": {
                "azimuth": 0.0,
                "tilt": 0.0
            },
            "name_adj_zone": None,
            "height": 10.0,
            "length": 10.0
        },
        {
            "name": "Transparent east surface",
            "type": "transparent",
            "area": 3.0,
            "sky_view_factor": 0.5,
            "u_value": 5.0,
            "solar_absorptance": 0.5,
            "thermal_capacity": 0.0,
            "orientation": {
                "azimuth": 90.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 2.0,
            # "length": 1.0,
            "g_value": 0.726,
            "width": 1.0,
            "parapet": 1.1,
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs": 1.0
            }
        },
        {
            "name": "Transparent west surface",
            "type": "transparent",
            "area": 5.0,
            "sky_view_factor": 0.5,
            "u_value": 5.0,
            "solar_absorptance": 0.5,
            "thermal_capacity": 0.0,
            "orientation": {
                "azimuth": 270.0,
                "tilt": 90.0
            },
            "name_adj_zone": None,
            "height": 2.0,
            # "length": 1.0,
            "g_value": 0.726,
            "width": 1.0,
            "parapet": 1.1,
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs": 1.0
            }
        }
    ],
    "units": {
        "area": "m\u00b2",
        "u_value": "W/m\u00b2K",
        "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
        "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m\u00b2",
        "internal_gain_profile": "Normalized to 0-1",
        "HVAC_profile": "0: off, 1: on"
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0,
            "heating_setback": 17.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "units": "\u00b0C"
        },
        "system_capacities": {
            "heating_capacity": 10000000.0,
            "cooling_capacity": 12000000.0,
            "units": "W"
        },
        "ventilation": {
            "ventilation_type": "occupancy",
            "flow_rate_per_person": 0.3,
            "units": "l/(s m2)",
            # Keep a numeric fallback because ventilation.py always casts this field to float().
            "custom_heat_transfer_coefficient_ventilation": 0.0,
            "info": "ventilation type could be: 1) Occupancy 2) occupancy 3)custom. If custum the value of custom_heat_transfer_coefficient_ventilation is used"
        },
        "internal_gains": [
            {
                "name": "occupants",
                "full_load": 4.2,
                "weekday": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.2,
                    0.5,
                    0.5,
                    0.5,
                    0.8,
                    0.8,
                    0.8,
                    1.0,
                    1.0
                ],
                "weekend": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.2,
                    0.5,
                    0.5,
                    0.5,
                    0.8,
                    0.8,
                    0.8,
                    1.0,
                    1.0
                ]
            },
            {
                "name": "appliances",
                "full_load": 3.0,
                "weekday": [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.5,
                    0.5,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.8,
                    0.8,
                    0.8,
                    0.6,
                    0.6
                ],
                "weekend": [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.5,
                    0.5,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.5,
                    0.5,
                    0.7,
                    0.7,
                    0.8,
                    0.8,
                    0.8,
                    0.6,
                    0.6
                ]
            },
            {
                "name": "lighting",
                "full_load": 3.0,
                "weekday": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.15,
                    0.15
                ],
                "weekend": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.15,
                    0.15
                ]
            }
        ],
        "construction": {
            "wall_thickness": 0.35,
            "thermal_bridges": 2.0,
            "units": "m (for thickness), W/mK (for thermal bridges)"
        },
        "climate_parameters": {
            "coldest_month": 1,
            "units": "1-12 (January-December)"
        },
        "heating_profile": {
            "weekday": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ],
            "weekend": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ]
        },
        "cooling_profile": {
            "weekday": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ],
            "weekend": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0
            ]
        },
        "ventilation_profile": {
            "weekday": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ],
            "weekend": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0
            ]
        }
    }
}


INPUT_SYSTEM_HVAC = {
    # ------------------------------------------------------------------
    # EN 15316-2 - EMISSION SYSTEM
    # ------------------------------------------------------------------
    # Legacy ISO 15316-1 emission inputs.
    'emitter_type': 'Floor heating',
    'nominal_power': 15.0,
    'emission_efficiency': 90.0,
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
    'selected_emm_cont_circuit': 0,
    'mixing_valve': True,
    'emission_calculation_mode': 'en15316-2',
    'emission_time_step_hours': 1.0,

    # 'TB14': custom_TB14, #  <- Uncomment and provide your emitter table; otherwise the default stored in global_inputs.py is used
    # 'heat_emission_data' : pd.DataFrame({ <- Uncomment and provide your emitter table; otherwise the default stored in global_inputs.py is used
    #         "θH_em_flw_max_sahz_i": [45],
    #         "ΔθH_em_w_max_sahz_i": [8],
    #         "θH_em_ret_req_sahz_i": [20],
    #         "βH_em_req_sahz_i": [80],
    #         "θH_em_flw_min_tz_i": [28],
    #     }, index=[
    #         "Max flow temperature HZ1",
    #         "Max Δθ flow / return HZ1",
    #         "Desired return temperature HZ1",
    #         "Desired load factor with ON-OFF for HZ1",
    #         "Minimum flow temperature for HZ1"
    #     ]),
    # 'outdoor_temp_data': pd.DataFrame({
    #         # If this table is provided, iso_15316_1.py uses it to build the
    #         # secondary outdoor-temperature curve; otherwise it falls back to
    #         # its internal _default_outdoor_data() table.
    #         "θext_min_sahz_i": [-10],
    #         "θext_max_sahz_i": [16],
    #         "θem_flw_max_sahz_i": [45],
    #         "θem_flw_min_sahz_i": [28],
    #     }, index=[
    #         "Minimum outdoor temperature",
    #         "Maximum outdoor temperature",
    #         "Maximum flow temperature",
    #         "Minimum flow temperature"
    #     ]),
    'mixing_valve_delta':2,
    # 'constant_flow_temp':42,

    # Detailed EN 15316-2 emission configuration.
    'emission_15316_2_config': {
        # Global EN 15316-2 settings.
        'time_step_hours': 1.0,
        'demand_unit': 'kWh',
        'default_heating_internal_temperature_C': 20.0,
        'default_cooling_internal_temperature_C': 26.0,

        # Heating emission parameters used by EmissionSystemCalculator.
        'heating': {
            'stratification_K': 0.0,
            'control_K': 0.0,
            'radiation_K': 0.0,
            'hydraulic_balancing_K': 0.0,
            'room_automation_K': 0.0,
            'embedded_K': 0.0,
            'nominal_power_kW': 15.0,
            'fan_power_W': 0.0,
            'fan_count': 0,
            'control_power_W': 0.0,
            'control_count': 0,
            'convective_fraction': 0.7,
        },

        # Cooling emission parameters used by EmissionSystemCalculator.
        'cooling': {
            'stratification_K': 0.0,
            'control_K': 0.0,
            'radiation_K': 0.0,
            'hydraulic_balancing_K': 0.0,
            'room_automation_K': 0.0,
            'embedded_K': 0.0,
            'nominal_power_kW': 15.0,
            'fan_power_W': 0.0,
            'fan_count': 0,
            'control_power_W': 0.0,
            'control_count': 0,
            'convective_fraction': 0.7,
        },
    },

    # ------------------------------------------------------------------
    # EN 15316-3 - DISTRIBUTION SYSTEM
    # ------------------------------------------------------------------
    # Inputs used by the distribution block.
    'heat_losses_recovered': True,
    'distribution_loss_recovery': 90,
    'simplified_approach': 80,
    'distribution_aux_recovery': 80,
    'distribution_aux_power': 30,
    'distribution_loss_coeff': 48,
    'distribution_operation_time': 1,
    'distribution_calculation_mode': 'analytical', # or simplified for aggregated approach
    'distribution_15316_3_config': {
        # EN 15316-3 step resolution and service-level configuration.
        'time_step_hours': 1.0,
        'demand_unit': 'kWh',
        'heating': {
            # Heating distribution service.
            'operation_mode': 'demand',
            'nominal_power_kW': 15.0,
            'design_flow_m3_h': 1.304,
            'design_deltaT_K': 10.0,
            'pipe_sections': [
                {
                    # Single aggregated pipe section for the example building.
                    'length_m': 12.0,
                    'equivalent_length_m': 2.0,
                    'linear_thermal_transmittance_W_mK': 0.45,
                    'ambient_temperature_C': 20.0,
                    'recoverable': True,
                }
            ],
            # Pump model assumptions.
            'pump_control_code': 4, # 0 uncontrolled, 3 = variable speed based on p∆ constant, 4 = variable speed based on p∆ variable
            'eei': 0.23,
            'hydraulic_correction_factor': 1.0,
            'recoverable_aux_fraction': 0.25,
            'pressure_loss_per_m_kPa': 0.10,
            'additional_pressure_kPa': 0.0,
            'resistance_ratio': 0.30,
            'pump_selection_factor': 1.0,
            'pump_label_power_kW': 0.0,
            'part_load_mode': 'load',
        },
        'dhw': {
            'operation_mode': 'demand',
            'nominal_power_kW': 15.0,
            'design_flow_m3_h': 1.304,
            'design_deltaT_K': 10.0,
            'dhw_temperature_C': 55.0,
            'dhw_return_deltaT_K': 5.0,
            'max_length_m': 25.0,
            'pressure_loss_per_m_kPa': 0.10,
            'additional_pressure_kPa': 0.0,
            'resistance_ratio': 0.30,
            'pump_control_code': 4,
            'eei': 0.23,
            'hydraulic_correction_factor': 1.0,
            'recoverable_aux_fraction': 0.25,
            'pipe_sections': [
                {
                    'length_m': 12.0,
                    'equivalent_length_m': 2.0,
                    'linear_thermal_transmittance_W_mK': 0.45,
                    'ambient_temperature_C': 20.0,
                    'recoverable': True,
                }
            ],
        },
    },

    # DHW branch selection.
    'same_generator_for_heating_and_dhw': True,
    'dhw_storage_enabled': True,
    'dhw_storage_config': {
        'time_step_hours': 1.0,
        'demand_unit': 'kWh',
        'dhw': {
            'enabled': True,
            'storage_volume_l': 180.0,
            'storage_setpoint_C': 55.0,
            'output_temperature_C': 55.0,
            'ambient_temperature_C': 20.0,
            'standby_loss_kWh_per_day_ref': 0.90,
            'standby_set_temperature_ref_C': 55.0,
            'standby_ambient_temperature_ref_C': 20.0,
            'standby_loss_adaptation_factor': 1.0,
            'connection_loss_factor': 1.0,
            'thermal_loss_room_fraction': 0.75,
            'auxiliary_to_medium_fraction': 0.25,
            'input_pump_power_kW': 0.0,
            'input_pump_flow_m3_h': 0.0,
            'input_pump_deltaT_K': 10.0,
            'output_pump_power_kW': 0.0,
            'output_pump_flow_m3_h': 0.0,
            'output_pump_deltaT_K': 10.0,
            'operation_mode': 'demand',
        },
    },
    'dhw_generator_config': {
        'nominal_power_kW': 8.0,
        'rated_power_kW': 8.0,
        'boiler_type': 'condensing',
        'fuel_type': 'natural_gas',
        'boiler_location': 'inside_heated',
    },
    
    # ------------------------------------------------------------------
    # EN 15316-4-1 - GENERATION SYSTEM
    # ------------------------------------------------------------------
    # Inputs used by the boiler/generator block.
    'full_load_power': 27,                  # kW
    'max_monthly_load_factor': 100,         # %
    'tH_gen_i_ON': 1,                       # h
    'auxiliary_power_generator': 0,         # %
    'fraction_of_auxiliary_power_generator': 40,   # %
    'generator_circuit': 'independent',     # 'direct' | 'independent'

    # Primary flow temperature control and outdoor reset curve.
    'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',
    'gen_outdoor_temp_data': pd.DataFrame({
        "θext_min_gen": [-7],
        "θext_max_gen": [15],
        "θflw_gen_max": [60],
        "θflw_gen_min": [35],
    }, index=["Generator curve"]),

    'speed_control_generator_pump': 'variable',
    'generator_nominal_deltaT': 20,         # °C
    'mixing_valve_delta':2,
    # 
    'generation_calculation_mode': 'boiler_15316_4_1',
    'boiler_generation_config': {
        # Boiler-specific EN 15316-4-1 performance data.
        'boiler_type': 'condensing',
        'fuel_type': 'natural_gas',
        'rated_power_kW': 27.0,
        'intermediate_load_fraction': 0.30,
        'eta_Pn_test_pct': 98.0,
        'eta_Pint_test_pct': 106.0,
        'theta_test_Pn_C': 60.0,
        'theta_test_Pint_C': 40.0,
        'f_corr_pct_per_K': 0.04,
        'P_gen_ls_P0_W': 100.0,
        'P_aux_on_W': 80.0,
        'P_aux_off_W': 5.0,
        'f_jacket': 0.40,
        'f_location': 1.0,
        'f_aux_recoverable': 0.75,
        'dew_point_C': 55.0,
        'condensing_gain_pct': 11.0,
        'efficiency_table': {
            'condensing': {
                'eta_Pn_test_pct': 98.0,
                'eta_Pint_test_pct': 106.0,
                'theta_test_Pn_C': 60.0,
                'theta_test_Pint_C': 40.0,
            }
        },
        'loss_table': {
            'condensing': {
                'P_gen_ls_P0_W': 100.0,
            }
        },
        'boiler_location': 'inside_heated',
    },

    # Optional explicit generator setpoints (commented by default)
    # 'θHW_gen_flw_set': 50,
    # 'θHW_gen_ret_set': 40,

    # Efficiency model
    'efficiency_model': 'simple',

    # Calculation options
    'calc_when_QH_positive_only': False,
    'off_compute_mode': 'full',




}

# ============================================================
#           HVAC INPUT CONSISTENCY CHECK
# ============================================================

# This check verifies that the main thermal powers are coherent across
# emission, distribution, and generation before the simulation starts.
# If the values are not aligned, the script stops with a clear error.
_emission_nominal = float(INPUT_SYSTEM_HVAC.get('nominal_power', 0.0))
_emission_15316_2_nominal = float(
    INPUT_SYSTEM_HVAC.get('emission_15316_2_config', {})
    .get('heating', {})
    .get('nominal_power_kW', 0.0)
)
_distribution_nominal = float(
    INPUT_SYSTEM_HVAC.get('distribution_15316_3_config', {})
    .get('heating', {})
    .get('nominal_power_kW', 0.0)
)
_generator_full_load = float(INPUT_SYSTEM_HVAC.get('full_load_power', 0.0))
_boiler_rated = float(INPUT_SYSTEM_HVAC.get('boiler_generation_config', {}).get('rated_power_kW', 0.0))
_same_generator_for_heating_and_dhw = bool(INPUT_SYSTEM_HVAC.get('same_generator_for_heating_and_dhw', True))
_dhw_storage_enabled = bool(INPUT_SYSTEM_HVAC.get('dhw_storage_enabled', False))
_boiler_type = str(INPUT_SYSTEM_HVAC.get('boiler_generation_config', {}).get('boiler_type', '')).lower()
_fuel_type = str(INPUT_SYSTEM_HVAC.get('boiler_generation_config', {}).get('fuel_type', '')).lower()
_boiler_location = str(INPUT_SYSTEM_HVAC.get('boiler_generation_config', {}).get('boiler_location', '')).lower()
_speed_control_generator_pump = str(INPUT_SYSTEM_HVAC.get('speed_control_generator_pump', '')).lower()
_efficiency_model = str(INPUT_SYSTEM_HVAC.get('efficiency_model', '')).lower()
_off_compute_mode = str(INPUT_SYSTEM_HVAC.get('off_compute_mode', '')).lower()

_tolerance_pct = 0.10
_emission_block_diff = abs(_emission_nominal - _emission_15316_2_nominal)
_emission_distribution_diff = abs(_emission_nominal - _distribution_nominal)
_generator_boiler_diff = abs(_generator_full_load - _boiler_rated)
_distribution_flow = float(
    INPUT_SYSTEM_HVAC.get('distribution_15316_3_config', {})
    .get('heating', {})
    .get('design_flow_m3_h', 0.0)
)
_distribution_deltaT = float(
    INPUT_SYSTEM_HVAC.get('distribution_15316_3_config', {})
    .get('heating', {})
    .get('design_deltaT_K', 0.0)
)
_water_heat_capacity_density = 1.15
_expected_flow = (
    _emission_nominal / (_water_heat_capacity_density * _distribution_deltaT)
    if _distribution_deltaT > 0
    else 0.0
)
_flow_diff = abs(_distribution_flow - _expected_flow)

if (
    _emission_nominal <= 0
    or _emission_15316_2_nominal <= 0
    or _distribution_nominal <= 0
    or _generator_full_load <= 0
    or _boiler_rated <= 0
):
    raise ValueError(
        "HVAC input error: nominal_power, emission_15316_2_config.heating.nominal_power_kW, "
        "distribution nominal_power_kW, full_load_power and boiler rated_power_kW must all be positive."
    )

if _emission_block_diff > 0.01 * max(_emission_nominal, 1e-9):
    raise ValueError(
        "HVAC input error: nominal_power and emission_15316_2_config.heating.nominal_power_kW "
        f"must match closely. Got nominal_power={_emission_nominal:.3f} kW and "
        f"emission_15316_2_config.heating.nominal_power_kW={_emission_15316_2_nominal:.3f} kW."
    )

_generator_power_diff = abs(_generator_full_load - _boiler_rated)
if _generator_power_diff > 0.01 * max(_generator_full_load, 1e-9):
    raise ValueError(
        "HVAC input error: full_load_power and boiler_generation_config.rated_power_kW "
        f"must match closely. Got full_load_power={_generator_full_load:.3f} kW and "
        f"rated_power_kW={_boiler_rated:.3f} kW."
    )

_dhw_nominal = float(INPUT_SYSTEM_HVAC.get('dhw_generator_config', {}).get('nominal_power_kW', 0.0))
if not _same_generator_for_heating_and_dhw and _dhw_nominal <= 0:
    raise ValueError(
        "HVAC input error: dhw_generator_config.nominal_power_kW must be positive "
        "when same_generator_for_heating_and_dhw is False."
    )

# Boiler/fuel compatibility check: this blocks incoherent generator configurations
# before the EN 15316-4-1 model is executed.
_allowed_fuel_map = {
    "standard": {"natural_gas"},
    "low_temperature": {"natural_gas"},
    "condensing": {"natural_gas"},
    "biomass_log": {"wood_log"},
    "biomass_pellet": {"wood_pellet"},
}

if _boiler_type not in _allowed_fuel_map:
    raise ValueError(
        "HVAC input error: boiler_type is not supported. "
        f"Expected one of {sorted(_allowed_fuel_map)}."
    )

if _fuel_type not in _allowed_fuel_map[_boiler_type]:
    raise ValueError(
        "HVAC input error: boiler_type and fuel_type are not compatible. "
        f"For boiler_type='{_boiler_type}' expected one of {sorted(_allowed_fuel_map[_boiler_type])}, "
        f"got fuel_type='{_fuel_type}'."
    )

# Boiler location check: the EN 15316-4-1 calculator only supports a closed
# set of installation positions, so invalid values are rejected up front.
_allowed_boiler_locations = {
    "inside_heated",
    "adjacent_unheated",
    "outside_building",
}

if _boiler_location not in _allowed_boiler_locations:
    raise ValueError(
        "HVAC input error: boiler_location is not supported. "
        f"Expected one of {sorted(_allowed_boiler_locations)}, got '{_boiler_location}'."
    )

# Generator control checks: these fields select the internal calculation branch
# used by the EN 15316-4-1 / ISO 15316-1 interface, so invalid values must be
# rejected before the solver starts.
_allowed_speed_control_generator_pump = {
    "variable",
    "deltaT_constant",
}
if _speed_control_generator_pump not in _allowed_speed_control_generator_pump:
    raise ValueError(
        "HVAC input error: speed_control_generator_pump is not supported. "
        f"Expected one of {sorted(_allowed_speed_control_generator_pump)}, "
        f"got '{_speed_control_generator_pump}'."
    )

_allowed_efficiency_model = {
    "simple",
    "parametric",
}
if _efficiency_model not in _allowed_efficiency_model:
    raise ValueError(
        "HVAC input error: efficiency_model is not supported. "
        f"Expected one of {sorted(_allowed_efficiency_model)}, got '{_efficiency_model}'."
    )

_allowed_off_compute_mode = {
    "idle",
    "temps",
    "full",
}
if _off_compute_mode not in _allowed_off_compute_mode:
    raise ValueError(
        "HVAC input error: off_compute_mode is not supported. "
        f"Expected one of {sorted(_allowed_off_compute_mode)}, got '{_off_compute_mode}'."
    )

if _emission_distribution_diff > _tolerance_pct * _emission_nominal:
    raise ValueError(
        "HVAC input error: emission nominal_power and distribution nominal_power_kW "
        "are not aligned. The distribution must use the same reference power as the emitter."
    )

if _generator_boiler_diff > _tolerance_pct * _generator_full_load:
    raise ValueError(
        "HVAC input error: full_load_power and boiler rated_power_kW are not aligned. "
        "The boiler configuration must match the generator sizing."
    )

# Hydraulic sizing check: validates that the design flow is coherent with the
# nominal emitter power and the design temperature difference of the circuit.
if _distribution_deltaT <= 0:
    raise ValueError(
        "HVAC input error: design_deltaT_K must be positive to validate the design flow."
    )

if _distribution_flow <= 0:
    raise ValueError(
        "HVAC input error: design_flow_m3_h must be positive."
    )

if _flow_diff > _tolerance_pct * max(_expected_flow, 1e-9):
    raise ValueError(
        "HVAC input error: design_flow_m3_h is not coherent with nominal_power and "
        f"design_deltaT_K. Expected design_flow_m3_h is about {_expected_flow:.3f} m3/h "
        f"from Vdot = P / (c * ΔT) = {_emission_nominal:.1f} / "
        f"({_water_heat_capacity_density:.2f} * {_distribution_deltaT:.1f}). "
        "Check the hydraulic sizing of the distribution circuit."
    )

# ============================================================
#           QUALITY CHECK SYSTEM INPUT HVAC
# ============================================================

# res = pybui.check_heating_system_inputs(INPUT_SYSTEM_HVAC)
res = check_heating_system_inputs(INPUT_SYSTEM_HVAC)


print("Selected Emitter:", res["emitter_type"])
print("Messages:")
for m in res["messages"]:
    print("-", m)
INPUT_SYSTEM_HVAC = res["config"]

# ============================================================
#           CALCULATION HVAC
# ============================================================

calc = HeatingSystemCalculator(INPUT_SYSTEM_HVAC)



bui_fixed, report = sanitize_and_validate_BUI(BUI, fix=True)


# print issues
for r in report:
    lvl = r["level"]
    print(f"[{lvl}] {r['path']}: {r['msg']}" + (" (fix applied)" if r["fix_applied"] else ""))

# validate BUI
bui_checked, issues = sanitize_and_validate_BUI(BUI, fix=False)
bui_checked['building_surface']
# extract only errors (level "ERROR")
errors = [e for e in issues if e["level"] == "ERROR"]



# def process_building(building_archetype, output_dir="result_test"):
#     """Process a single building archetype and save results"""
#     try:

#         # Process the building
#         (
#             hourly_sim,
#             annual_results_df,
#             _,
#         ) = _run_iso52016(building_archetype)

#         # Generate unique filenames for each building
#         building_name = building_archetype["building"].get("name", "unknown")
#         hourly_file = os.path.join(output_dir, f"hourly_sim_{building_name}.csv")
#         annual_file = os.path.join(output_dir, f"annual_results_{building_name}.csv")

#         # Save results with unique filenames
#         hourly_sim.to_csv(hourly_file)
#         annual_results_df.to_csv(annual_file, index=False)

#         # Calculate metrics
#         heating_kWh = hourly_sim[hourly_sim["Q_HC"] > 0]["Q_HC"].sum() / 1000
#         cooling_kWh = -hourly_sim[hourly_sim["Q_HC"] < 0]["Q_HC"].sum() / 1000
#         treated_floor_area = building_archetype["building"]["treated_floor_area"]
#         heating_kWh_per_sqm = heating_kWh / treated_floor_area
#         cooling_kWh_per_sqm = cooling_kWh / treated_floor_area

#         return {
#             "building_name": building_name,
#             "heating_kWh": heating_kWh,
#             "cooling_kWh": cooling_kWh,
#             "heating_kWh_per_sqm": heating_kWh_per_sqm,
#             "cooling_kWh_per_sqm": cooling_kWh_per_sqm,
#             "status": "success",
#         }

#     except Exception as e:
#         return {
#             "building_name": building_archetype["building"].get("name", "unknown"),
#             "error": str(e),
#             "status": "failed",
#         }


if errors:
    print("❌ Errors in BUI input data — simulation interrupted:\n")
    for e in errors:
        print(f" - {e['path']}: {e['msg']}")
    raise ValueError("Invalid BUI input: correct the data and retry.")
else:
    print("✅ BUI valid — starting ISO52016 simulation...\n")
    if WEATHER_SOURCE == "epw":
        print(f"[info] Weather source: epw ({WEATHER_FILE})")
    else:
        print("[info] Weather source: pvgis (no local EPW found)")
    file_dir = "/Users/dantonucci/Documents/GitHub/pybuildingenergy/result_test"
    # hourly_sim,annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(bui_checked, weather_source="epw", path_weather_file=str(WEATHER_FILE))
    hourly_sim, annual_results_df, sankey_data = _run_iso52016(bui_checked)

    # ISO 15316-1 calculation
    df_in = calc.load_csv_data(hourly_sim)  # columns: Q_H, T_op, T_ext (or aliases)
    df_out = calc.run_timeseries()
    _export_hvac_flow_results(hourly_sim, df_out, file_dir)
    _build_sankey_consumption_report(hourly_sim, df_out, file_dir)
    if GENERATE_EXTRA_REPORTS:
        _build_temperature_report(df_out, file_dir)
        Graphs_and_report(df=hourly_sim, season='heating_cooling', building_area=BUI['building']['net_floor_area']).bui_analysis_page(
            folder_directory=file_dir,
            name_file="main_report",
        )
    print("[info] Column legend: examples/hvac_stage_results_legend_it.md")

    # ============================================================
    #           DHW CHAIN: optional storage + distribution
    # ============================================================
    year_for_dhw = int(pd.DatetimeIndex(hourly_sim.index).year.min()) if len(hourly_sim.index) else 2023
    italy_calendar = generate_calendar("Italy", year_for_dhw)
    n_workdays = int((italy_calendar["values"] == "Working").sum())
    n_weekends = int((italy_calendar["values"] == "Non-Working").sum())
    n_holidays = int((italy_calendar["values"] == "Holiday").sum())
    total_days = int(italy_calendar["values"].count())

    hourly_fractions_examples = pd.DataFrame({
        "Workday": [0,0,0,0,0,0,0,0,5,10,10,10,20,10,10,10,10,5,0,0,0,0,0,0],
        "Weekend": [0,0,0,0,0,0,0,0,5,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0],
        "Holiday": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    })
    sum_fractions = pd.DataFrame(hourly_fractions_examples.sum())
    sum_fractions.columns = ["fractions"]
    dhw_calc = Volume_and_energy_DHW_calculation(
        n_workdays,
        n_weekends,
        n_holidays,
        sum_fractions,
        total_days,
        hourly_fractions_examples,
        42,
        13.5,
        60,
        11.2,
        mode_calc='number_of_units',
        building_type_B3='Residential',
        building_area=120,
        unit_count=10,
        building_type_B5='Dwelling',
        residential_typology='residential_building - simple housing - AVG',
        calculation_method='table',
        year=year_for_dhw,
        country_calendar=italy_calendar
    )

    dhw_hourly = pd.DataFrame(
        {"Q_W_kWh": pd.Series(dhw_calc[7], index=hourly_sim.index, dtype=float)},
        index=hourly_sim.index,
    )
    dhw_distribution_calc = pybui.DistributionSystemCalculator(INPUT_SYSTEM_HVAC.get('distribution_15316_3_config', {}))
    dhw_distribution_result = dhw_distribution_calc.run_timeseries(dhw_hourly)
    dhw_distribution_result.timeseries.to_csv(f"{file_dir}/dhw_distribution_15316_3_hourly_results.csv")
    pd.DataFrame([dhw_distribution_result.summary]).to_csv(
        f"{file_dir}/dhw_distribution_15316_3_summary.csv",
        index=False,
    )

    dhw_storage_result = None
    if _dhw_storage_enabled:
        dhw_storage_input = pd.DataFrame(
            {"Q_W_kWh": dhw_distribution_result.timeseries["Q_W_dis_in_kWh"].astype(float)},
            index=hourly_sim.index,
        )
        dhw_storage_calc = pybui.StorageSystemCalculator(INPUT_SYSTEM_HVAC.get('dhw_storage_config', {}))
        dhw_storage_result = dhw_storage_calc.run_timeseries(dhw_storage_input)
        dhw_storage_result.timeseries.to_csv(f"{file_dir}/dhw_storage_15316_5_hourly_results.csv")
        pd.DataFrame([dhw_storage_result.summary]).to_csv(
            f"{file_dir}/dhw_storage_15316_5_summary.csv",
            index=False,
        )

    combined_parts = [hourly_sim.reset_index(drop=True)]
    if isinstance(df_out, pd.DataFrame):
        combined_parts.append(df_out.add_prefix("hvac_").reset_index(drop=True))
    combined_parts.append(dhw_hourly.add_prefix("dhw_").reset_index(drop=True))
    if hasattr(dhw_distribution_result, "timeseries"):
        combined_parts.append(dhw_distribution_result.timeseries.add_prefix("dhw_dis_").reset_index(drop=True))
    if dhw_storage_result is not None:
        combined_parts.append(dhw_storage_result.timeseries.add_prefix("dhw_sto_").reset_index(drop=True))
    combined_hourly = pd.concat(combined_parts, axis=1)
    combined_hourly.to_csv(f"{file_dir}/hvac_dhw_hourly_results.csv")

    heating_generator = float(df_out.get("QH_gen_out(kWh)", pd.Series(0.0, index=df_out.index)).sum())
    if dhw_storage_result is not None:
        dhw_generator = float(
            pd.to_numeric(
                dhw_storage_result.timeseries["Q_W_sto_in_kWh"],
                errors="coerce",
            ).fillna(0.0).sum()
        )
    else:
        dhw_generator = float(
            pd.to_numeric(
                dhw_distribution_result.timeseries["Q_W_dis_in_kWh"],
                errors="coerce",
            ).fillna(0.0).sum()
        )

    if _same_generator_for_heating_and_dhw:
        print(
            f"[info] Same generator for heating and DHW: heating={heating_generator:.2f} kWh, "
            f"DHW after distribution/storage={dhw_generator:.2f} kWh, combined={heating_generator + dhw_generator:.2f} kWh"
        )
    else:
        dhw_gen_cfg = INPUT_SYSTEM_HVAC.get('dhw_generator_config', {})
        dhw_gen_power = float(dhw_gen_cfg.get('rated_power_kW', dhw_gen_cfg.get('nominal_power_kW', 0.0)))
        if dhw_gen_power <= 0:
            raise ValueError(
                "HVAC input error: dhw_generator_config must define rated_power_kW or nominal_power_kW when DHW is separate."
            )
        print(
            "[info] Separate DHW generator selected. "
            f"Use dhw_generator_config for generator sizing; DHW demand after distribution/storage is {dhw_generator:.2f} kWh."
        )

    _build_sankey_consumption_report(
        hourly_sim,
        df_out,
        file_dir,
        dhw_useful_df=dhw_hourly,
        dhw_distribution_df=dhw_distribution_result.timeseries,
        dhw_storage_df=(dhw_storage_result.timeseries if dhw_storage_result is not None else None),
    )

    raise SystemExit(0)


# ================================================================================================================
#                                   DHW NEEDS
# ================================================================================================================

# Water temperature of the mixed (cold and hot) water drawn at the tap
teta_W_draw = 42 
# Cold water temperature
teta_W_cold = 11.2 
# Hot water delivery temperature 60°C
teta_w_h_ref = 60
# Cold water supply temperature 13.5°C
teta_w_c_ref = 13.5
# Physical constant
# Building inputs
building_area = 120
building_type = 'Dwellings'
# Use Profiles
hourly_fractions_examples = pd.DataFrame({
    "Workday" : [0,0,0,0,0,0,0,0,5,10,10,10,20,10,10,10,10,5,0,0,0,0,0,0],
    "Weekend" : [0,0,0,0,0,0,0,0,5,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0],
    "Holiday" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
)
sum_fractions = pd.DataFrame(hourly_fractions_examples.sum())
sum_fractions.columns= ["fractions"]
# National calendar
hourly_fractions = hourly_fractions_examples
calendar_nation = "Italy"
Italy_calendar = generate_calendar(calendar_nation, 2023)
n_workdays = sum(Italy_calendar['values'] == 'Working')
n_weekends = sum(Italy_calendar['values'] == 'Non-Working')
n_holidays = sum(Italy_calendar['values'] == 'Holiday')
Italy_calendar['values'].unique()
total_days = Italy_calendar.count().values[0]

# DHW needs
DHW_calc = Volume_and_energy_DHW_calculation(
    n_workdays, n_weekends, n_holidays,sum_fractions, total_days, hourly_fractions,
    teta_W_draw, 
    teta_w_c_ref,
    teta_w_h_ref,
    teta_W_cold,
    mode_calc= 'number_of_units', 
    building_type_B3= 'Residential', 
    building_area= 142, 
    unit_count= 10, 
    building_type_B5= 'Dwelling',
    residential_typology= 'residential_building - simple housing - AVG', # table B_4
    calculation_method= 'table',
    year= 2015,
    country_calendar=Italy_calendar
    )

# Plot data
t_start=24
t_end = 48
df = pd.DataFrame(dict(
    x = list(range(0,len(DHW_calc[6])))[t_start:t_end],
    y = DHW_calc[6][t_start:t_end],
))
fig = px.line(df, x="x", y="y", title = "DHW profile") 
fig.show()

df['z'] = df['y'].cumsum()
fig_1 = px.line(df, x="x", y="z", title = "DHW profile cumulative") 
fig_1.show()



# PRIMARY ENERGY CALCULATION 
