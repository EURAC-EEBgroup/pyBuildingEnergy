import sys
import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (SRC_DIR, PROJECT_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pandas as pd

import pybuildingenergy as pybui
from pybuildingenergy.source.check_input import sanitize_and_validate_BUI, check_heating_system_inputs
from pybuildingenergy.source.example_inputs import get_example_hvac_input
from pybuildingenergy.source.hvac_quality_checks import validate_hvac_input_coherence
from pybuildingenergy.source.hvac_sankey_reports import _build_sankey_consumption_report
from pybuildingenergy.source.iso_15316_1 import HeatingSystemCalculator
from pybuildingenergy.source.DHW import Volume_and_energy_DHW_calculation

WEATHER_CANDIDATES = [
    EXAMPLES_DIR / "2050_Athens.epw",
    EXAMPLES_DIR / "2020_Milan.epw",
]
WEATHER_FILE = EXAMPLES_DIR / "2020_Milan.epw"
WEATHER_SOURCE = "epw"
GENERATE_EXTRA_REPORTS = False
DHW_DYNAMIC_CALCULATION = True


def _run_iso52016(building_obj, input_hvac):
    kwargs = {
        "weather_source": WEATHER_SOURCE,
        "latent_indoor_rh_pct": input_hvac.get("latent_indoor_rh_pct", 50.0),
    }
    if WEATHER_SOURCE == "epw":
        kwargs["path_weather_file"] = str(WEATHER_FILE)
    out = pybui.ISO52016.Temperature_and_Energy_needs_calculation(building_obj, **kwargs)
    if isinstance(out, tuple) and len(out) == 3:
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1], {}
    raise RuntimeError("Unexpected output format from Temperature_and_Energy_needs_calculation")


def _apply_heating_heat_pump_generation(hourly_sim, hvac_df, input_hvac, output_dir):
    hp_cfg = dict(input_hvac.get("heating_heat_pump_15316_4_2_config", {}))
    hp_cfg.setdefault("source_type", "air")
    hp_cfg.setdefault("heating_enabled", True)
    hp_cfg.setdefault("cooling_enabled", False)
    hp_cfg.setdefault("dhw_enabled", False)
    hp_cfg.setdefault("demand_unit", "kWh")

    loads = pd.DataFrame(index=hvac_df.index)
    loads["T_ext"] = pd.to_numeric(hvac_df.get("T_ext(°C)", hourly_sim["T_ext"]), errors="coerce")
    loads["Q_H_kWh"] = pd.to_numeric(hvac_df["QH_gen_out(kWh)"], errors="coerce").fillna(0.0).clip(lower=0.0)
    loads["Q_C_kWh"] = 0.0
    loads["Q_W_kWh"] = 0.0
    if "θH_dis_flw(°C)" in hvac_df.columns:
        loads["T_H_sink_C"] = pd.to_numeric(hvac_df["θH_dis_flw(°C)"], errors="coerce")
    if "θH_dis_ret(°C)" in hvac_df.columns:
        loads["T_H_return_C"] = pd.to_numeric(hvac_df["θH_dis_ret(°C)"], errors="coerce")

    hp_calc = pybui.HeatPumpSystemCalculator(hp_cfg)
    hp_result = hp_calc.run_timeseries(loads)
    summary = dict(hp_result.summary)
    q_h = float(summary.get("QH_gen_out_kWh", loads["Q_H_kWh"].sum()))
    e_hp = float(summary.get("EH_hp_in_kWh", 0.0))
    e_backup = float(summary.get("EHW_backup_in_kWh", 0.0))
    e_gen = float(summary.get("EHW_gen_in_kWh", e_hp + e_backup))
    w_aux = float(summary.get("WHW_gen_aux_kWh", 0.0))
    spf = float(summary.get("SPF_HW_gen", q_h / (e_gen + w_aux) if (e_gen + w_aux) > 0 else 0.0))

    out = hvac_df.copy()
    weights = pd.to_numeric(out["QH_gen_out(kWh)"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = weights / q_h if q_h > 0 else 0.0
    out["generation_type"] = "heat_pump_15316_4_2"
    out["EHW_gen_in(kWh)"] = weights * e_gen
    out["EH_gen_in(kWh)"] = weights * e_gen
    out["EHW_gen_aux(kWh)"] = weights * w_aux
    out["efficiency_gen(%)"] = spf * 100.0
    out["HP_EH_hp_in(kWh)"] = weights * e_hp
    out["HP_EH_backup_in(kWh)"] = weights * e_backup
    out["HP_SPF_HW_gen(-)"] = spf

    out_dir = Path(output_dir)
    hp_result.bins.to_csv(out_dir / "heating_heat_pump_15316_4_2_bin_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "heating_heat_pump_15316_4_2_summary.csv", index=False)
    loads.to_csv(out_dir / "heating_heat_pump_15316_4_2_loads.csv")
    return out, hp_result


def _default_heat_pump_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
    heating_rows = []
    for source in [-15, -7, 2, 7, 12, 20]:
        for sink in [35, 45, 55]:
            capacity = 8.8 + 0.11 * (source - 7.0) - 0.055 * (sink - 35.0)
            cop = 4.60 + 0.055 * (source - 7.0) - 0.045 * (sink - 35.0)
            heating_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "cop": max(cop, 1.6),
                }
            )

    dhw_rows = []
    for source in [-15, -7, 2, 7, 12, 20]:
        for sink in [45, 50, 55, 60]:
            capacity = 8.4 + 0.10 * (source - 7.0) - 0.06 * (sink - 55.0)
            cop = 3.85 + 0.045 * (source - 7.0) - 0.050 * (sink - 55.0)
            dhw_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.0),
                    "cop": max(cop, 1.4),
                }
            )

    return pd.DataFrame(heating_rows), pd.DataFrame(dhw_rows)


def _build_hourly_report(hourly_sim, hvac_df, dhw_hourly, dhw_distribution_result, dhw_storage_result, output_dir):
    report_df = pd.DataFrame(index=hourly_sim.index)
    report_df["timestamp"] = pd.Index(hourly_sim.index)
    report_df["heating_load_kWh"] = pd.to_numeric(hvac_df.get("QH_gen_out(kWh)", 0.0), errors="coerce").fillna(0.0)
    report_df["dhw_load_kWh"] = pd.to_numeric(dhw_hourly["Q_W_kWh"], errors="coerce").fillna(0.0)
    report_df["hp_electricity_heating_kWh"] = pd.to_numeric(
        hvac_df.get("EHW_gen_in(kWh)", 0.0), errors="coerce"
    ).fillna(0.0)
    zone_temp = None
    for temp_col in ("T_op", "T_air", "T_op0"):
        if temp_col in hourly_sim.columns:
            zone_temp = pd.to_numeric(hourly_sim[temp_col], errors="coerce")
            break
    report_df["zone_internal_temperature_C"] = (
        zone_temp.reindex(report_df.index).fillna(np.nan)
        if zone_temp is not None
        else pd.Series(np.nan, index=report_df.index)
    )
    if dhw_storage_result is not None and "W_W_sto_aux_kWh" in dhw_storage_result.timeseries.columns:
        report_df["hp_electricity_dhw_kWh"] = pd.to_numeric(
            dhw_storage_result.timeseries["W_W_sto_aux_kWh"], errors="coerce"
        ).fillna(0.0)
        report_df["dhw_tank_temperature_C"] = pd.to_numeric(
            dhw_storage_result.timeseries.get("theta_W_sto_post_draw_C", dhw_storage_result.timeseries["theta_W_sto_out_C"]),
            errors="coerce",
        ).fillna(np.nan)
        report_df["dhw_storage_temperature_C"] = pd.to_numeric(
            dhw_storage_result.timeseries["theta_W_sto_set_C"], errors="coerce"
        ).fillna(np.nan)
        if "theta_W_sto_top_C" in dhw_storage_result.timeseries.columns:
            report_df["dhw_tank_top_temperature_C"] = pd.to_numeric(
                dhw_storage_result.timeseries["theta_W_sto_top_C"], errors="coerce"
            ).fillna(np.nan)
        if "theta_W_sto_bottom_C" in dhw_storage_result.timeseries.columns:
            report_df["dhw_tank_bottom_temperature_C"] = pd.to_numeric(
                dhw_storage_result.timeseries["theta_W_sto_bottom_C"], errors="coerce"
            ).fillna(np.nan)
        if "Q_W_sto_recharge_time_h" in dhw_storage_result.timeseries.columns:
            report_df["dhw_recharge_time_h"] = pd.to_numeric(
                dhw_storage_result.timeseries["Q_W_sto_recharge_time_h"], errors="coerce"
            ).fillna(0.0)
        if "Q_W_sto_recharge_energy_kWh" in dhw_storage_result.timeseries.columns:
            report_df["dhw_recharge_energy_kWh"] = pd.to_numeric(
                dhw_storage_result.timeseries["Q_W_sto_recharge_energy_kWh"], errors="coerce"
            ).fillna(0.0)
        if "Q_W_sto_stratification_K" in dhw_storage_result.timeseries.columns:
            report_df["dhw_stratification_K"] = pd.to_numeric(
                dhw_storage_result.timeseries["Q_W_sto_stratification_K"], errors="coerce"
            ).fillna(0.0)
        if "theta_W_sto_pre_draw_C" in dhw_storage_result.timeseries.columns:
            report_df["dhw_temp_pre_draw_C"] = pd.to_numeric(
                dhw_storage_result.timeseries["theta_W_sto_pre_draw_C"], errors="coerce"
            ).fillna(np.nan)
        if "theta_W_sto_post_draw_C" in dhw_storage_result.timeseries.columns:
            report_df["dhw_temp_post_draw_C"] = pd.to_numeric(
                dhw_storage_result.timeseries["theta_W_sto_post_draw_C"], errors="coerce"
            ).fillna(np.nan)
        if "V_W_sto_l" in dhw_storage_result.timeseries.columns:
            report_df["dhw_storage_volume_l"] = pd.to_numeric(
                dhw_storage_result.timeseries["V_W_sto_l"], errors="coerce"
            ).fillna(0.0)
        if "V_W_sto_refill_l" in dhw_storage_result.timeseries.columns:
            report_df["dhw_reheat_volume_l"] = pd.to_numeric(
                dhw_storage_result.timeseries["V_W_sto_refill_l"], errors="coerce"
            ).fillna(0.0)
        if "V_W_sto_before_dhw_mode_l" in dhw_storage_result.timeseries.columns:
            report_df["dhw_tank_before_reheat_l"] = pd.to_numeric(
                dhw_storage_result.timeseries["V_W_sto_before_dhw_mode_l"], errors="coerce"
            ).fillna(0.0)
        if "V_W_sto_remaining_l" in dhw_storage_result.timeseries.columns:
            report_df["dhw_tank_remaining_volume_l"] = pd.to_numeric(
                dhw_storage_result.timeseries["V_W_sto_remaining_l"], errors="coerce"
            ).fillna(0.0)
        if "dhw_dhw_mode_flag" in dhw_storage_result.timeseries.columns:
            report_df["dhw_dhw_mode_flag"] = dhw_storage_result.timeseries["dhw_dhw_mode_flag"].astype(bool)
    else:
        report_df["hp_electricity_dhw_kWh"] = pd.to_numeric(
            dhw_distribution_result.timeseries.get("W_W_dis_aux_kWh", 0.0), errors="coerce"
        ).fillna(0.0)
        report_df["dhw_tank_temperature_C"] = np.nan
        report_df["dhw_storage_temperature_C"] = np.nan
        report_df["dhw_storage_volume_l"] = np.nan
        report_df["dhw_reheat_volume_l"] = np.nan
        report_df["dhw_tank_before_reheat_l"] = np.nan
        report_df["dhw_tank_remaining_volume_l"] = np.nan
        report_df["dhw_dhw_mode_flag"] = False

    dhw_mode_mask = report_df["dhw_dhw_mode_flag"].fillna(False).astype(bool)
    recharge_time_h = pd.to_numeric(
        report_df.get("dhw_recharge_time_h", pd.Series(0.0, index=report_df.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    heating_available_fraction = (1.0 - recharge_time_h).clip(lower=0.0, upper=1.0)
    report_df.loc[dhw_mode_mask, "hp_electricity_heating_kWh"] = (
        report_df.loc[dhw_mode_mask, "hp_electricity_heating_kWh"]
        * heating_available_fraction.loc[dhw_mode_mask]
    )
    heating_active = report_df["hp_electricity_heating_kWh"] > 0.0
    dhw_active = (report_df["hp_electricity_dhw_kWh"] > 0.0) | dhw_mode_mask
    report_df["hp_mode"] = np.select(
        [heating_active & dhw_active, heating_active, dhw_active],
        ["heating + dhw", "heating", "dhw"],
        default="",
    )
    display_df = report_df.copy()
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    if "dhw_mode" not in display_df.columns:
        display_df["dhw_mode"] = np.select(
            [display_df["dhw_dhw_mode_flag"].fillna(False).astype(bool), display_df["dhw_load_kWh"] > 0],
            ["dhw recharge", "draw"],
            default="standby",
        )
    column_order = [
        "timestamp",
        "hp_mode",
        "heating_load_kWh",
        "hp_electricity_heating_kWh",
        "dhw_load_kWh",
        "hp_electricity_dhw_kWh",
        "dhw_mode",
        "dhw_tank_temperature_C",
        "dhw_storage_temperature_C",
        "dhw_tank_top_temperature_C",
        "dhw_tank_bottom_temperature_C",
        "dhw_temp_pre_draw_C",
        "dhw_temp_post_draw_C",
        "dhw_recharge_energy_kWh",
        "dhw_recharge_time_h",
        "dhw_stratification_K",
        "dhw_storage_volume_l",
        "dhw_tank_before_reheat_l",
        "dhw_reheat_volume_l",
        "dhw_tank_remaining_volume_l",
        "dhw_dhw_mode_flag",
        "zone_internal_temperature_C",
    ]
    column_order = [c for c in column_order if c in display_df.columns]
    display_df = display_df[column_order]
    html_table = display_df.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    charts_only_html = ""
    if dhw_storage_result is not None:
        chart_df = display_df.copy()
        if not chart_df.empty:
            chart_df = chart_df.fillna(0.0)
            zero_series = pd.Series(0.0, index=chart_df.index)
            nan_series = pd.Series(np.nan, index=chart_df.index)
            dhw_time_quota_h = (
                pd.to_numeric(chart_df.get("dhw_recharge_time_h", zero_series), errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
            heating_time_quota_h = (1.0 - dhw_time_quota_h).clip(lower=0.0, upper=1.0)
            chart_data = {
                "timestamp": chart_df["timestamp"].astype(str).tolist(),
                "dhw_load_kWh": pd.to_numeric(chart_df.get("dhw_load_kWh", zero_series), errors="coerce").fillna(0.0).tolist(),
                "dhw_reheat_volume_l": pd.to_numeric(chart_df.get("dhw_reheat_volume_l", zero_series), errors="coerce").fillna(0.0).tolist(),
                "dhw_storage_volume_l": pd.to_numeric(chart_df.get("dhw_storage_volume_l", zero_series), errors="coerce").fillna(0.0).tolist(),
                "dhw_temp_pre_draw_C": pd.to_numeric(chart_df.get("dhw_temp_pre_draw_C", nan_series), errors="coerce").tolist(),
                "dhw_temp_post_draw_C": pd.to_numeric(chart_df.get("dhw_temp_post_draw_C", nan_series), errors="coerce").tolist(),
                "dhw_tank_top_temperature_C": pd.to_numeric(chart_df.get("dhw_tank_top_temperature_C", nan_series), errors="coerce").tolist(),
                "dhw_tank_bottom_temperature_C": pd.to_numeric(chart_df.get("dhw_tank_bottom_temperature_C", nan_series), errors="coerce").tolist(),
                "dhw_storage_temperature_C": pd.to_numeric(chart_df.get("dhw_storage_temperature_C", nan_series), errors="coerce").tolist(),
                "dhw_dhw_mode_flag": pd.to_numeric(chart_df.get("dhw_dhw_mode_flag", zero_series), errors="coerce").fillna(0).astype(int).tolist(),
                "dhw_recharge_time_h": pd.to_numeric(chart_df.get("dhw_recharge_time_h", zero_series), errors="coerce").fillna(0.0).tolist(),
                "dhw_time_quota_h": dhw_time_quota_h.tolist(),
                "heating_time_quota_h": heating_time_quota_h.tolist(),
                "dhw_recharge_energy_kWh": pd.to_numeric(chart_df.get("dhw_recharge_energy_kWh", zero_series), errors="coerce").fillna(0.0).tolist(),
                "generator_heating_consumption_kWh": pd.to_numeric(chart_df.get("hp_electricity_heating_kWh", zero_series), errors="coerce").fillna(0.0).tolist(),
                "generator_dhw_consumption_kWh": pd.to_numeric(chart_df.get("hp_electricity_dhw_kWh", zero_series), errors="coerce").fillna(0.0).tolist(),
                "zone_internal_temperature_C": pd.to_numeric(chart_df.get("zone_internal_temperature_C", nan_series), errors="coerce").tolist(),
            }
            chart_json = json.dumps(chart_data)
            charts_only_html = """
            <html>
            <head>
              <meta charset='utf-8'>
              <title>HVAC charts</title>
              <script src='../src/pybuildingenergy/charts/assets/echarts.min.js'></script>
              <style>
                body{font-family:Inter,Segoe UI,Arial,sans-serif;background:#f6f7fb;color:#111827;margin:0;padding:24px;line-height:1.4}
                .wrap{max-width:1700px;margin:0 auto}
                h1{font-size:28px;margin:0 0 8px}
                .sub{color:#6b7280;margin:0 0 24px}
                .card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 1px 2px rgba(0,0,0,.04);padding:18px;margin-bottom:18px}
                .chart{width:100%;height:440px}
                .hint{margin-top:10px;color:#6b7280;font-size:12px}
              </style>
            </head>
            <body>
              <div class='wrap'>
                <h1>HVAC charts</h1>
                <p class='sub'>Grafici offline in ECharts con zoom da toolbox, tooltip dettagliato e serie orarie per ACS, accumulo e riscaldamento.</p>
                <div class='card'><div id='chart_hourly_split' class='chart'></div><div class='hint'>Barre impilate: quota oraria DHW/heating. Linee: consumo elettrico heating e DHW.</div></div>
                <div class='card'><div id='chart_dhw' class='chart'></div><div class='hint'>DHW demand, reheat volume and storage volume.</div></div>
                <div class='card'><div id='chart_temp' class='chart'></div><div class='hint'>Temperature dell'accumulo ACS e setpoint.</div></div>
                <div class='card'><div id='chart_recharge' class='chart'></div><div class='hint'>Tempo di ricarica e segnale di modalita DHW.</div></div>
                <div class='card'><div id='chart_generator_zone' class='chart'></div><div class='hint'>Consumo elettrico heating e DHW della pompa di calore con priorita ACS sotto 45 C.</div></div>
              </div>
              <script>
                const data = __CHART_JSON__;
                const timestamps = data.timestamp || [];
                const commonZoom = [
                  { type: 'inside', start: 0, end: 100 },
                  { type: 'slider', height: 18, bottom: 12 }
                ];
                function baseOption(title, legendTop) {
                  return {
                    title: { text: title, left: 12, top: 8, textStyle: { fontSize: 18, fontWeight: 600, color: '#111827' } },
                    legend: { top: legendTop ?? 42, left: 12, type: 'scroll' },
                    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
                    grid: { left: 60, right: 70, top: 86, bottom: 78, containLabel: true },
                    toolbox: {
                      right: 14,
                      top: 8,
                      feature: {
                        dataZoom: { yAxisIndex: 'none' },
                        restore: {},
                        saveAsImage: {}
                      }
                    },
                    dataZoom: commonZoom,
                    xAxis: {
                      type: 'category',
                      boundaryGap: false,
                      data: timestamps,
                      axisLabel: { hideOverlap: true, rotate: 35 }
                    },
                    animation: false
                  };
                }

                const chartHourlySplit = echarts.init(document.getElementById('chart_hourly_split'));
                chartHourlySplit.setOption({
                  ...baseOption('Hourly DHW/heating split and electricity consumption'),
                  yAxis: [
                    { type: 'value', name: 'h', min: 0, max: 1, axisLabel: { formatter: '{value}' } },
                    { type: 'value', name: 'kWh', axisLabel: { formatter: '{value}' } }
                  ],
                  series: [
                    {
                      name: 'DHW quota (h)',
                      type: 'bar',
                      stack: 'hour quota',
                      data: data.dhw_time_quota_h || [],
                      itemStyle: { color: '#2563eb' },
                      emphasis: { focus: 'series' }
                    },
                    {
                      name: 'Heating quota (h)',
                      type: 'bar',
                      stack: 'hour quota',
                      data: data.heating_time_quota_h || [],
                      itemStyle: { color: '#ea580c' },
                      emphasis: { focus: 'series' }
                    },
                    {
                      name: 'DHW electricity (kWh)',
                      type: 'line',
                      yAxisIndex: 1,
                      data: data.generator_dhw_consumption_kWh || [],
                      showSymbol: false,
                      smooth: true,
                      lineStyle: { width: 2, color: '#0f766e' }
                    },
                    {
                      name: 'Heating electricity (kWh)',
                      type: 'line',
                      yAxisIndex: 1,
                      data: data.generator_heating_consumption_kWh || [],
                      showSymbol: false,
                      smooth: true,
                      lineStyle: { width: 2, color: '#7c3aed' }
                    }
                  ]
                });

                const chartDhw = echarts.init(document.getElementById('chart_dhw'));
                chartDhw.setOption({
                  ...baseOption('DHW demand, refill and storage volume'),
                  yAxis: [
                    { type: 'value', name: 'kWh', axisLabel: { formatter: '{value}' } },
                    { type: 'value', name: 'L', axisLabel: { formatter: '{value}' } }
                  ],
                  series: [
                    { name: 'DHW demand (kWh)', type: 'bar', data: data.dhw_load_kWh || [], itemStyle: { color: '#2563eb' }, emphasis: { focus: 'series' } },
                    { name: 'Reheat volume (L)', type: 'bar', yAxisIndex: 1, data: data.dhw_reheat_volume_l || [], itemStyle: { color: '#dc2626' }, emphasis: { focus: 'series' } },
                    { name: 'Storage volume (L)', type: 'line', yAxisIndex: 1, data: data.dhw_storage_volume_l || [], showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#0f766e' } }
                  ]
                });

                const chartTemp = echarts.init(document.getElementById('chart_temp'));
                chartTemp.setOption({
                  ...baseOption('DHW storage temperatures'),
                  yAxis: { type: 'value', name: '°C', axisLabel: { formatter: '{value}' } },
                  series: [
                    { name: 'Pre-draw temp', type: 'line', data: data.dhw_temp_pre_draw_C || [], showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#7c3aed' } },
                    { name: 'Post-draw temp', type: 'line', data: data.dhw_temp_post_draw_C || [], showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#f59e0b' } },
                    { name: 'Top layer temp', type: 'line', data: data.dhw_tank_top_temperature_C || [], showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#ef4444' } },
                    { name: 'Bottom layer temp', type: 'line', data: data.dhw_tank_bottom_temperature_C || [], showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#0ea5e9' } },
                    { name: 'Setpoint', type: 'line', data: data.dhw_storage_temperature_C || [], showSymbol: false, smooth: false, lineStyle: { width: 2, type: 'dashed', color: '#111827' } }
                  ]
                });

                const chartRecharge = echarts.init(document.getElementById('chart_recharge'));
                chartRecharge.setOption({
                  ...baseOption('DHW recharge time and mode flag'),
                  yAxis: [
                    { type: 'value', name: 'h', axisLabel: { formatter: '{value}' } },
                    { type: 'value', name: 'flag', min: 0, max: 1, interval: 1, axisLabel: { formatter: '{value}' } }
                  ],
                  series: [
                    { name: 'Recharge time (h)', type: 'line', data: data.dhw_recharge_time_h || [], showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#7c3aed' } },
                    {
                      name: 'Recharge energy (kWh)',
                      type: 'bar',
                      data: data.dhw_recharge_energy_kWh || [],
                      itemStyle: { color: '#059669' },
                      emphasis: { focus: 'series' }
                    },
                    {
                      name: 'DHW mode flag',
                      type: 'scatter',
                      yAxisIndex: 1,
                      symbolSize: 8,
                      data: (data.dhw_dhw_mode_flag || []).map((v, idx) => [timestamps[idx], v]),
                      itemStyle: { color: '#dc2626' }
                    }
                  ]
                });

                const chartGenerator = echarts.init(document.getElementById('chart_generator_zone'));
                chartGenerator.setOption({
                  ...baseOption('Generator electricity and zone internal temperature'),
                  yAxis: [
                    { type: 'value', name: 'kWh', axisLabel: { formatter: '{value}' } },
                    { type: 'value', name: '°C', axisLabel: { formatter: '{value}' } }
                  ],
                  series: [
                    {
                      name: 'Heating electricity (kWh)',
                      type: 'bar',
                      data: data.generator_heating_consumption_kWh || [],
                      itemStyle: { color: '#ea580c' },
                      emphasis: { focus: 'series' }
                    },
                    {
                      name: 'DHW electricity (kWh)',
                      type: 'bar',
                      data: data.generator_dhw_consumption_kWh || [],
                      itemStyle: { color: '#2563eb' },
                      emphasis: { focus: 'series' }
                    },
                    {
                      name: 'Zone internal temperature (°C)',
                      type: 'line',
                      yAxisIndex: 1,
                      data: data.zone_internal_temperature_C || [],
                      showSymbol: false,
                      smooth: true,
                      lineStyle: { width: 2, color: '#0f766e' }
                    }
                  ]
                });

                window.addEventListener('resize', () => {
                  chartHourlySplit.resize();
                  chartDhw.resize();
                  chartTemp.resize();
                  chartRecharge.resize();
                  chartGenerator.resize();
                });
              </script>
            </body>
            </html>
            """.replace("__CHART_JSON__", chart_json)
    _dhw_recharge_kWh = float(report_df["dhw_recharge_energy_kWh"].fillna(0.0).sum()) if "dhw_recharge_energy_kWh" in report_df.columns else 0.0
    _dhw_max_recharge_h = float(report_df["dhw_recharge_time_h"].fillna(0.0).max()) if "dhw_recharge_time_h" in report_df.columns else 0.0
    _dhw_storage_vol_l = float(report_df["dhw_storage_volume_l"].dropna().iloc[0]) if "dhw_storage_volume_l" in report_df.columns and not report_df["dhw_storage_volume_l"].dropna().empty else 0.0
    out_html = Path(output_dir) / "hvac_hourly_report.html"
    display_df.to_csv(Path(output_dir) / "hvac_hourly_report.csv", index=False)
    out_html.write_text(
        "<html><head><meta charset='utf-8'><title>HVAC hourly report</title>"
        "<style>"
        "body{font-family:Inter,Segoe UI,Arial,sans-serif;background:#f6f7fb;color:#111827;margin:0;padding:24px;line-height:1.4}"
        ".wrap{max-width:1800px;margin:0 auto}"
        ".table-wrap{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 1px 2px rgba(0,0,0,.04);overflow:auto;max-height:calc(100vh - 48px)}"
        "table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}"
        "th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:right;white-space:nowrap}"
        "th{background:#111827;color:#fff;position:sticky;top:0;z-index:3}"
        "td:first-child,th:first-child{text-align:left}"
        "tr:nth-child(even) td{background:#fafafa}"
        "</style></head><body><div class='wrap'><div class='table-wrap'>"
        f"{html_table}"
        "</div></div></body></html>",
        encoding="utf-8",
    )
    if charts_only_html:
        (Path(output_dir) / "hvac_hourly_charts.html").write_text(charts_only_html, encoding="utf-8")


def _build_bui():
    return {
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
            "country": "Italy",
        },
        "building_surface": [
            {"name": "Roof surface", "type": "opaque", "area": 130.0, "sky_view_factor": 1.0, "u_value": 2.2, "solar_absorptance": 0.4, "thermal_capacity": 741500.0, "orientation": {"azimuth": 0.0, "tilt": 0.0}, "name_adj_zone": None, "height": 10.0, "length": 13.0},
            {"name": "Opaque north surface", "type": "opaque", "area": 30.0, "sky_view_factor": 0.5, "u_value": 1.4, "solar_absorptance": 0.4, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 0.0, "tilt": 90.0}, "name_adj_zone": None, "height": 10.0, "length": 3.0},
            {"name": "Opaque south surface", "type": "opaque", "area": 30.0, "sky_view_factor": 0.5, "u_value": 1.4, "solar_absorptance": 0.4, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 180.0, "tilt": 90.0}, "name_adj_zone": None, "height": 10.0, "length": 3.0},
            {"name": "Opaque east surface", "type": "opaque", "area": 30.0, "sky_view_factor": 0.5, "u_value": 1.2, "solar_absorptance": 0.6, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 90.0, "tilt": 90.0}, "name_adj_zone": None, "height": 10.0, "length": 3.0},
            {"name": "Opaque west surface", "type": "opaque", "area": 30.0, "sky_view_factor": 0.5, "u_value": 1.2, "solar_absorptance": 0.7, "thermal_capacity": 1416240.0, "orientation": {"azimuth": 270.0, "tilt": 90.0}, "name_adj_zone": None, "height": 10.0, "length": 3.0},
            {"name": "Slab to ground", "type": "opaque", "area": 100.0, "sky_view_factor": 0.5, "u_value": 1.6, "solar_absorptance": 0.6, "thermal_capacity": 405801.0, "orientation": {"azimuth": 0.0, "tilt": 0.0}, "name_adj_zone": None, "height": 10.0, "length": 10.0},
            {"name": "Transparent east surface", "type": "transparent", "area": 3.0, "sky_view_factor": 0.5, "u_value": 5.0, "solar_absorptance": 0.5, "thermal_capacity": 0.0, "orientation": {"azimuth": 90.0, "tilt": 90.0}, "name_adj_zone": None, "height": 2.0, "g_value": 0.726, "width": 1.0, "parapet": 1.1, "shading": False, "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5, "overhang_properties": {"width_of_horizontal_overhangs": 1.0}},
            {"name": "Transparent west surface", "type": "transparent", "area": 5.0, "sky_view_factor": 0.5, "u_value": 5.0, "solar_absorptance": 0.5, "thermal_capacity": 0.0, "orientation": {"azimuth": 270.0, "tilt": 90.0}, "name_adj_zone": None, "height": 2.0, "g_value": 0.726, "width": 1.0, "parapet": 1.1, "shading": False, "shading_type": "horizontal_overhang", "width_or_distance_of_shading_elements": 0.5, "overhang_properties": {"width_of_horizontal_overhangs": 1.0}},
        ],
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "°C",
            },
            "system_capacities": {"heating_capacity": 10000000.0, "cooling_capacity": 12000000.0, "units": "W"},
            "ventilation": {
                "ventilation_type": "occupancy",
                "flow_rate_per_person": 0.3,
                "units": "l/(s m2)",
                "custom_heat_transfer_coefficient_ventilation": 0.0,
            },
            "construction": {"wall_thickness": 0.35, "thermal_bridge_heat_W_K": 2.0, "units": "m (for thickness), W/K (for total thermal-bridge coefficient)"},
            "climate_parameters": {"coldest_month": 1, "units": "1-12 (January-December)"},
            "heating_profile": {"weekday": [0.0] * 5 + [1.0] * 18 + [0.0], "weekend": [0.0] * 5 + [1.0] * 18 + [0.0]},
            "cooling_profile": {"weekday": [0.0] * 24, "weekend": [0.0] * 24},
            "ventilation_profile": {"weekday": [0.0] * 5 + [1.0] * 18 + [0.0], "weekend": [0.0] * 5 + [0.0] * 18 + [0.0]},
        },
    }


def _configure_inputs():
    cfg = get_example_hvac_input()
    cfg["emitter_type"] = "Floor heating"
    cfg["heating_generator_type"] = "heat_pump_15316_4_2"
    cfg["generation_calculation_mode"] = "heat_pump_15316_4_2"
    cfg["same_generator_for_heating_and_dhw"] = True
    cfg["dhw_storage_enabled"] = True
    heating_map, dhw_map = _default_heat_pump_maps()
    cfg["heating_heat_pump_15316_4_2_config"]["heating_performance_map"] = heating_map
    cfg["heating_heat_pump_15316_4_2_config"]["dhw_performance_map"] = dhw_map
    cfg["dhw_generator_config"]["nominal_power_kW"] = 8.0
    cfg["dhw_generator_config"]["rated_power_kW"] = 8.0
    cfg["heating_heat_pump_15316_4_2_config"]["source_type"] = "air"
    cfg["heating_heat_pump_15316_4_2_config"]["heating_sink_temp_at_design_C"] = 35.0
    cfg["heating_heat_pump_15316_4_2_config"]["heating_sink_temp_at_cutoff_C"] = 28.0
    cfg["heating_heat_pump_15316_4_2_config"]["hp_operating_limit_C"] = 55.0
    cfg["heating_heat_pump_15316_4_2_config"]["dhw_storage_loss_kWh_per_day"] = 0.0
    cfg["heating_heat_pump_15316_4_2_config"]["dhw_enabled"] = True
    cfg["heating_heat_pump_15316_4_2_config"]["cooling_enabled"] = False
    cfg["heating_heat_pump_15316_4_2_config"]["heating_enabled"] = True
    cfg["dhw_storage_config"]["dhw"]["storage_volume_l"] = 180.0
    cfg["dhw_storage_config"]["dhw"]["storage_setpoint_C"] = 55.0
    cfg["dhw_storage_config"]["dhw"]["output_temperature_C"] = 55.0
    cfg["dhw_storage_config"]["dhw"]["ambient_temperature_C"] = 20.0
    cfg["dhw_storage_config"]["dhw"]["dynamic_calculation"] = DHW_DYNAMIC_CALCULATION
    cfg["dhw_storage_config"]["dynamic_calculation"] = DHW_DYNAMIC_CALCULATION
    cfg["dhw_storage_config"]["dhw"]["storage_height_m"] = 1.2
    cfg["dhw_storage_config"]["dhw"]["sensor_height_m"] = 0.6
    cfg["dhw_storage_config"]["dhw"]["cold_inlet_temperature_C"] = 10.0
    cfg["dhw_storage_config"]["dhw"]["stratification_efficiency"] = 0.85
    cfg["dhw_storage_config"]["dhw"]["dhw_switch_temperature_C"] = 45.0
    cfg["dhw_storage_config"]["dhw"]["recharge_power_kW"] = 8.0
    cfg["dhw_storage_config"]["dhw"]["recharge_cop"] = 3.0
    cfg["distribution_15316_3_config"]["heating"]["nominal_power_kW"] = 15.0
    cfg["distribution_15316_3_config"]["heating"]["design_deltaT_K"] = 5.0
    cfg["distribution_15316_3_config"]["heating"]["design_flow_m3_h"] = 2.609
    cfg["distribution_15316_3_config"]["heating"]["pipe_sections"] = [
        {
            "length_m": 12.0,
            "equivalent_length_m": 2.0,
            "linear_thermal_transmittance_W_mK": 0.45,
            "ambient_temperature_C": 20.0,
            "recoverable": True,
        }
    ]
    cfg["distribution_15316_3_config"]["dhw"]["nominal_power_kW"] = 8.0
    cfg["distribution_15316_3_config"]["dhw"]["design_flow_m3_h"] = 1.0
    cfg["distribution_15316_3_config"]["dhw"]["dhw_temperature_C"] = 55.0
    cfg["distribution_15316_3_config"]["dhw"]["pipe_sections"] = [
        {
            "length_m": 12.0,
            "equivalent_length_m": 2.0,
            "linear_thermal_transmittance_W_mK": 0.45,
            "ambient_temperature_C": 20.0,
            "recoverable": True,
        }
    ]
    cfg["cooling_enabled"] = False
    cfg["cooling_storage_enabled"] = False
    cfg["dhw_generator_config"]["generator_type"] = "heat_pump_15316_4_2"
    return cfg


BUI = _build_bui()
INPUT_SYSTEM_HVAC = _configure_inputs()
INPUT_SYSTEM_COOLING = {"enabled": False, "cooling_storage_enabled": False}

_coherence_summary = validate_hvac_input_coherence(INPUT_SYSTEM_HVAC)
_heating_generator_type = _coherence_summary["heating_generator_type"]
_dhw_storage_enabled = bool(INPUT_SYSTEM_HVAC.get("dhw_storage_enabled", False))

res = check_heating_system_inputs(INPUT_SYSTEM_HVAC)
print("Selected Emitter:", res["emitter_type"])
for m in res["messages"]:
    print("-", m)
INPUT_SYSTEM_HVAC = res["config"]

calc = HeatingSystemCalculator(INPUT_SYSTEM_HVAC)
bui_checked, issues = sanitize_and_validate_BUI(BUI, fix=False)
errors = [e for e in issues if e["level"] == "ERROR"]
if errors:
    print("Invalid BUI input:")
    for e in errors:
        print(f" - {e['path']}: {e['msg']}")
    raise ValueError("Invalid BUI input: correct the data and retry.")

file_dir = str(EXAMPLES_DIR.parent / "result_test")
Path(file_dir).mkdir(parents=True, exist_ok=True)
hourly_sim, annual_results_df, sankey_data = _run_iso52016(bui_checked, INPUT_SYSTEM_HVAC)
df_in = calc.load_csv_data(hourly_sim)
df_out = calc.run_timeseries()
heating_heat_pump_result = None
if _heating_generator_type == "heat_pump_15316_4_2":
    df_out, heating_heat_pump_result = _apply_heating_heat_pump_generation(hourly_sim, df_out, INPUT_SYSTEM_HVAC, file_dir)

hourly_sim.to_csv(Path(file_dir) / "hourly_sim.csv")
annual_results_df.to_csv(Path(file_dir) / "annual_results.csv", index=False)
df_out.to_csv(Path(file_dir) / "hvac_results.csv", index=False)

year_for_dhw = int(pd.DatetimeIndex(hourly_sim.index).year.min()) if len(hourly_sim.index) else 2023
italy_calendar = pybui.generate_calendar("IT", year_for_dhw)
hourly_fractions_examples = pd.DataFrame(
    {
        "Workday": [0,0,0,0,0,0,0,0,5,10,10,10,20,10,10,10,10,5,0,0,0,0,0,0],
        "Weekend": [0,0,0,0,0,0,0,0,5,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0],
        "Holiday": [0] * 24,
    }
)
sum_fractions = pd.DataFrame(hourly_fractions_examples.sum())
sum_fractions.columns = ["fractions"]
dhw_calc = Volume_and_energy_DHW_calculation(
    int((italy_calendar["values"] == "Working").sum()),
    int((italy_calendar["values"] == "Non-Working").sum()),
    int((italy_calendar["values"] == "Holiday").sum()),
    sum_fractions,
    int(italy_calendar["values"].count()),
    hourly_fractions_examples,
    42,
    13.5,
    60,
    11.2,
    mode_calc="number_of_units",
    building_type_B3="Residential",
    building_area=120,
    unit_count=10,
    building_type_B5="Dwelling",
    residential_typology="residential_building - simple housing - AVG",
    calculation_method="table",
    year=year_for_dhw,
    country_calendar=italy_calendar,
)
dhw_values = pd.Series(dhw_calc[7], dtype=float)
if len(dhw_values) >= len(hourly_sim.index):
    dhw_values = dhw_values.iloc[: len(hourly_sim.index)].reset_index(drop=True)
else:
    dhw_values = dhw_values.reindex(range(len(hourly_sim.index)), fill_value=0.0)
dhw_hourly = pd.DataFrame({"Q_W_kWh": dhw_values.to_numpy(dtype=float)}, index=hourly_sim.index)
dhw_distribution_calc = pybui.DistributionSystemCalculator(INPUT_SYSTEM_HVAC.get("distribution_15316_3_config", {}))
dhw_distribution_result = dhw_distribution_calc.run_timeseries(dhw_hourly)
dhw_storage_result = None
if _dhw_storage_enabled:
    dhw_storage_input = pd.DataFrame({"Q_W_kWh": dhw_distribution_result.timeseries["Q_W_dis_in_kWh"].astype(float)}, index=hourly_sim.index)
    dhw_storage_calc = pybui.StorageSystemCalculator(INPUT_SYSTEM_HVAC.get("dhw_storage_config", {}))
    dhw_storage_result = dhw_storage_calc.run_timeseries(dhw_storage_input)

combined_parts = [hourly_sim.reset_index(drop=True), df_out.add_prefix("hvac_").reset_index(drop=True), dhw_hourly.add_prefix("dhw_").reset_index(drop=True), dhw_distribution_result.timeseries.add_prefix("dhw_dis_").reset_index(drop=True)]
if dhw_storage_result is not None:
    combined_parts.append(dhw_storage_result.timeseries.add_prefix("dhw_sto_").reset_index(drop=True))
pd.concat(combined_parts, axis=1).to_csv(Path(file_dir) / "hvac_dhw_hourly_results.csv", index=False)
_build_hourly_report(hourly_sim, df_out, dhw_hourly, dhw_distribution_result, dhw_storage_result, file_dir)

_build_sankey_consumption_report(
    hourly_sim,
    df_out,
    file_dir,
    same_generator_for_heating_and_dhw=True,
)

print("[info] Done. Outputs written to", file_dir)
