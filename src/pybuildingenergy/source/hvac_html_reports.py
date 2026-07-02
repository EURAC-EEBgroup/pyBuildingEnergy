from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyecharts import options as pye_opts
from pyecharts.charts import Line, Page


def _make_pyecharts_report_local(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    html = html.replace(
        "https://assets.pyecharts.org/assets/v5/echarts.min.js",
        "./echarts.min.js",
    )
    html_path.write_text(html, encoding="utf-8")


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
        vals = pd.to_numeric(hvac_df[col], errors="coerce").ffill().bfill().tolist()
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
    _make_pyecharts_report_local(out_path)
    print(f"[info] Temperature report written to {out_path}")
    return out_path
