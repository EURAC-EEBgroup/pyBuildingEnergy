from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyecharts import options as pye_opts
from pyecharts.charts import Sankey


def _make_pyecharts_report_local(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    html = html.replace(
        "https://assets.pyecharts.org/assets/v5/echarts.min.js",
        "./echarts.min.js",
    )
    html_path.write_text(html, encoding="utf-8")


def _sum_col(df: pd.DataFrame | None, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()) if df is not None and col in df.columns else 0.0


def _build_sankey_consumption_report(
    hourly_sim: pd.DataFrame,
    hvac_df: pd.DataFrame,
    output_dir: str,
    same_generator_for_heating_and_dhw: bool = True,
    dhw_useful_df: pd.DataFrame | None = None,
    dhw_distribution_df: pd.DataFrame | None = None,
    dhw_storage_df: pd.DataFrame | None = None,
) -> Path:
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
    _make_pyecharts_report_local(out_path)
    return out_path


def _build_cooling_sankey_report(
    hourly_sim: pd.DataFrame,
    output_dir: str,
    cooling_system_result=None,
    cooling_storage_result=None,
    cooling_generation_result=None,
) -> Path | None:
    missing = []
    if not isinstance(hourly_sim, pd.DataFrame):
        missing.append("ISO 52016 hourly results dataframe")
    if cooling_system_result is None or not hasattr(cooling_system_result, "timeseries"):
        missing.append("cooling_system_result.timeseries from EN 16798-9")
    if cooling_generation_result is None or not hasattr(cooling_generation_result, "timeseries"):
        missing.append("cooling_generation_result.timeseries from EN 16798-13")
    if missing:
        return None

    iso_df = hourly_sim
    sys_df = cooling_system_result.timeseries
    sto_df = cooling_storage_result.timeseries if cooling_storage_result is not None else None
    gen_df = cooling_generation_result.timeseries

    q_c_sensible = _sum_col(iso_df, "Q_C_sensible") / 1000.0 if "Q_C_sensible" in iso_df.columns else _sum_col(iso_df, "Q_C") / 1000.0
    q_c_latent = _sum_col(iso_df, "Q_C_latent") / 1000.0 if "Q_C_latent" in iso_df.columns else 0.0
    q_c_16798_out = _sum_col(sys_df, "Q_C_dis_out_tot_req_kWh")
    q_c_dis_losses = _sum_col(sys_df, "Q_C_dis_ls_simplified_kWh")
    q_c_dis_aux = _sum_col(sys_df, "W_C_aux_dis_simplified_kWh")
    q_c_16798_gen_req = _sum_col(sys_df, "Q_C_gen_in_req_kWh")
    if sto_df is not None:
        q_c_generation_req = _sum_col(sto_df, "Q_C_sto_in_kWh")
        q_c_storage_losses = _sum_col(sto_df, "Q_C_sto_ls_tot_kWh")
        q_c_storage_aux = _sum_col(sto_df, "W_C_sto_aux_kWh")
    else:
        q_c_generation_req = q_c_16798_gen_req
        q_c_storage_losses = 0.0
        q_c_storage_aux = 0.0

    q_c_gen_req = _sum_col(gen_df, "Q_C_gen_in_req_kWh")
    q_c_gen_delivered = _sum_col(gen_df, "Q_C_gen_in_kWh")
    q_c_gen_free = _sum_col(gen_df, "Q_C_gen_free_kWh")
    q_c_gen_mech = _sum_col(gen_df, "Q_C_gen_mech_in_kWh")
    q_c_gen_backup = _sum_col(gen_df, "Q_C_gen_backup_in_kWh")
    q_c_gen_unmet = _sum_col(gen_df, "Q_C_gen_unmet_kWh")
    e_c_el = _sum_col(gen_df, "E_C_gen_el_in_kWh")
    e_c_backup = _sum_col(gen_df, "E_C_backup_in_kWh")
    w_c_aux_gen = _sum_col(gen_df, "W_C_aux_gen_kWh")
    e_c_total = _sum_col(gen_df, "E_C_total_kWh")

    q_c_to_system = q_c_16798_out
    q_c_sensible_to_system = min(q_c_sensible, q_c_to_system)
    q_c_latent_to_system = max(0.0, min(q_c_latent, q_c_to_system - q_c_sensible_to_system))
    q_c_sensible_not_passed = max(q_c_sensible - q_c_sensible_to_system, 0.0)
    q_c_latent_not_passed = max(q_c_latent - q_c_latent_to_system, 0.0)

    links = []
    def _positive_link(source: str, target: str, value: float) -> None:
        if value > 0.05:
            links.append({"source": source, "target": target, "value": round(float(value), 1)})

    _positive_link("Cooling sensible need - ISO 52016", "Cooling useful energy passed to system", q_c_sensible_to_system)
    _positive_link("Cooling latent need - ISO 52016", "Cooling useful energy passed to system", q_c_latent_to_system)
    _positive_link("Cooling sensible need - ISO 52016", "Cooling sensible not passed to system", q_c_sensible_not_passed)
    _positive_link("Cooling latent need - ISO 52016", "Cooling latent not passed to system", q_c_latent_not_passed)
    _positive_link("Cooling useful energy passed to system", "Cooling distribution - EN 16798-9", q_c_to_system)
    _positive_link("Cooling distribution - EN 16798-9", "Cooling storage - EN 16798-15" if sto_df is not None else "Cooling generation - EN 16798-13", q_c_16798_gen_req)
    _positive_link("Cooling distribution - EN 16798-9", "Cooling distribution losses", q_c_dis_losses)
    _positive_link("Cooling distribution - EN 16798-9", "Cooling distribution auxiliaries", q_c_dis_aux)
    if sto_df is not None:
        _positive_link("Cooling storage - EN 16798-15", "Cooling generation - EN 16798-13", q_c_generation_req)
        _positive_link("Cooling storage - EN 16798-15", "Cooling storage heat gains", q_c_storage_losses)
        _positive_link("Cooling storage - EN 16798-15", "Cooling storage auxiliaries", q_c_storage_aux)
    _positive_link("Cooling generation - EN 16798-13", "Mechanical cooling delivered (kWh termici)", q_c_gen_mech)
    _positive_link("Cooling generation - EN 16798-13", "Free cooling delivered (kWh termici)", q_c_gen_free)
    _positive_link("Cooling generation - EN 16798-13", "Backup cooling delivered (kWh termici)", q_c_gen_backup)
    _positive_link("Cooling generation - EN 16798-13", "Unmet cooling need (kWh termici)", q_c_gen_unmet)
    _positive_link("Final cooling energy (kWh elettrici)", "Cooling generation - EN 16798-13", e_c_total)
    if not links:
        return None

    used_node_names = {link["source"] for link in links} | {link["target"] for link in links}
    nodes = [{"name": name} for name in used_node_names]
    seer = q_c_gen_delivered / e_c_total if e_c_total > 0 else 0.0
    subtitle = (
        f"ISO 52016 cooling: {q_c_sensible + q_c_latent:.0f} kWh "
        f"(sensible {q_c_sensible:.0f}, latent {q_c_latent:.0f}) | "
        f"EN 16798-9 request: {q_c_16798_out:.0f} kWh | "
        f"Latent not passed: {q_c_latent_not_passed:.0f} kWh | "
        f"Final cooling energy: {e_c_total:.0f} kWh | SEER {seer:.2f}"
    )
    sankey = (
        Sankey(init_opts=pye_opts.InitOpts(width="2400px", height="800px", page_title="Cooling Energy Sankey"))
        .add(
            series_name="Cooling",
            nodes=nodes,
            links=links,
            linestyle_opt=pye_opts.LineStyleOpts(opacity=0.45, curve=0.5, color="source"),
            label_opts=pye_opts.LabelOpts(position="right", font_size=12),
            node_gap=14,
            node_width=15,
            pos_left="6%",
            pos_right="7%",
            pos_top="10%",
            pos_bottom="10%",
        )
        .set_global_opts(
            title_opts=pye_opts.TitleOpts(
                title="Cooling energy flow - ISO 52016 + EN 16798",
                subtitle=subtitle,
            ),
            toolbox_opts=pye_opts.ToolboxOpts(is_show=True),
        )
    )
    out_path = Path(output_dir) / "cooling_energy_sankey.html"
    sankey.render(str(out_path))
    _make_pyecharts_report_local(out_path)
    return out_path
