"""Build an hourly/daily/monthly ECharts HTML report of thermal need, operative
temperature(s), outdoor temperature and outdoor relative humidity.

Uses ``pybuildingenergy.thermal_html_report()``, which plots whatever series
it is given -- so the SAME function works for a single-zone result (one
thermal-need series, one T_op series) and for a multizone result (one series
per zone). The outdoor temperature is already part of the single-zone hourly
output; the multizone engine does not keep it in its own output, so this
script re-fetches the weather data once (same site, same source) to get both
T_ext and outdoor relative humidity for the chart.

Usage:
    python examples/thermal_report_example.py --config examples/FH_Poland_DC_baseline_two_floors.json --mode single-zone
    python examples/thermal_report_example.py --config examples/FH_Poland_DC_multizone_ideal_envelope.json --mode multizone
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

import pybuildingenergy as pybui
from pybuildingenergy.source.utils import ISO52010

EXAMPLES_DIR = Path(__file__).resolve().parent


def _fetch_outdoor_series(building_object: dict, weather_source: str, path_weather_file: str | None, target_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Outdoor temperature + relative humidity, aligned to the simulation's local-time index.

    Mirrors the weather-loading/timezone-conversion steps the engine itself uses
    internally (see utils.py::Calculation_ISO_52010), since the multizone hourly
    output does not carry the outdoor temperature series itself.
    """

    if weather_source == "pvgis":
        weatherData = ISO52010.get_tmy_data_pvgis(building_object)
        sim_df = weatherData.weather_data
        sim_df.index = pd.to_datetime({
            "year": 2009, "month": sim_df.index.month, "day": sim_df.index.day, "hour": sim_df.index.hour,
        })
        from timezonefinder import TimezoneFinder
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=weatherData.longitude, lat=weatherData.latitude) or "UTC"
        idx = pd.DatetimeIndex(sim_df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        sim_df.index = idx.tz_convert(tz_name).tz_localize(None)
        # The DST "fall back" hour maps two different UTC hours onto the same
        # local civil timestamp; keep the first so reindex() below has no
        # duplicate labels to choke on.
        sim_df = sim_df[~sim_df.index.duplicated(keep="first")]
        t_ext = sim_df["T2m"]
        rh_ext = sim_df["RH"]
    elif weather_source == "epw":
        weatherData = ISO52010.get_tmy_data_epw(path_weather_file)
        sim_df = weatherData.weather_data
        t_ext = sim_df["temp_air"] if "temp_air" in sim_df.columns else sim_df.iloc[:, 0]
        rh_ext = sim_df["relative_humidity"] if "relative_humidity" in sim_df.columns else None
    else:
        raise ValueError("weather_source must be 'pvgis' or 'epw'")

    t_ext = t_ext.reindex(target_index, method="nearest")
    rh_ext = rh_ext.reindex(target_index, method="nearest") if rh_ext is not None else None
    return t_ext, rh_ext


def report_single_zone(config_path: str, output_dir: Path) -> str:
    config = pybui.load_config(config_path)
    bui = config["building"]
    weather = config["weather"]

    checked, issues = pybui.sanitize_and_validate_BUI(copy.deepcopy(bui), fix=True)
    errors = [i for i in issues if i["level"] == "ERROR"]
    if errors:
        raise ValueError(f"Building validation errors: {errors}")

    kwargs = {"weather_source": weather["source"]}
    if weather["source"] == "epw":
        kwargs["path_weather_file"] = weather["file"]
    hourly = pybui.ISO52016.Temperature_and_Energy_needs_calculation(checked, **kwargs)[0]

    thermal_need_kWh = {
        "Q_H": hourly["Q_H"].clip(lower=0.0) / 1000.0,
        "Q_C": hourly["Q_C"].clip(lower=0.0) / 1000.0,
    }
    t_op_C = {"T_op": hourly["T_op"]}
    t_ext_C = hourly["T_ext"]

    _, rh_ext = _fetch_outdoor_series(bui, weather["source"], weather.get("file"), hourly.index)

    return pybui.thermal_html_report(
        thermal_need_kWh=thermal_need_kWh,
        t_op_C=t_op_C,
        t_ext_C=t_ext_C,
        rh_ext_pct=rh_ext,
        folder_directory=str(output_dir),
        name_file="thermal_report_single_zone",
    )


def report_multizone(config_path: str, output_dir: Path) -> str:
    config = pybui.load_config(config_path)
    bui = config["building"]
    weather = config["weather"]

    checked, issues = pybui.sanitize_and_validate_BUI(copy.deepcopy(bui), fix=True)
    errors = [i for i in issues if i["level"] == "ERROR"]
    if errors:
        raise ValueError(f"Building validation errors: {errors}")

    hourly, _annual = pybui.ISO52016.Temperature_and_Energy_needs_calculation_multizone(
        building_object=checked, weather_source=weather["source"], path_weather_file=weather.get("file"),
        include_solar=True, warmup_hours=744, hvac_control_variable="operative",
    )

    zone_names = [z["name"] for z in bui.get("zones", [])] or ["main"]
    thermal_need_kWh = {}
    t_op_C = {}
    for zname in zone_names:
        qcol, tcol = f"Q_HVAC_{zname}", f"T_op_{zname}"
        if qcol not in hourly.columns:
            continue
        thermal_need_kWh[f"Q_{zname}"] = hourly[qcol].clip(lower=0.0) / 1000.0
        t_op_C[zname] = hourly[tcol]

    t_ext_C, rh_ext = _fetch_outdoor_series(bui, weather["source"], weather.get("file"), hourly.index)

    return pybui.thermal_html_report(
        thermal_need_kWh=thermal_need_kWh,
        t_op_C=t_op_C,
        t_ext_C=t_ext_C,
        rh_ext_pct=rh_ext,
        folder_directory=str(output_dir),
        name_file="thermal_report_multizone",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="building JSON (single-zone combined config, or multizone config)")
    ap.add_argument("--mode", choices=["single-zone", "multizone"], required=True)
    ap.add_argument("--output-dir", default=str(EXAMPLES_DIR / "outputs" / "thermal_report"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single-zone":
        path = report_single_zone(args.config, out_dir)
    else:
        path = report_multizone(args.config, out_dir)

    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
