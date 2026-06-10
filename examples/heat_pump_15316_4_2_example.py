"""Run European-standard system examples from ISO52016 and DHW demands.

This script demonstrates the complete sequence:

1. Calculate hourly space heating/cooling needs with ISO52016.
2. Apply EN 15316-2 emitter/control effects and emission losses.
3. Calculate DHW energy needs and DHW sizing with EN 12831-3.
4. Apply EN 15316-3 water-based distribution losses and pump auxiliaries.
5. Apply EN 15316-5 heating/DHW storage losses and auxiliaries using the
   EN 12831-3 DHW storage selection where enabled.
6. Apply EN 16798-9, EN 16798-15 and EN 16798-13 to the cooling-side
   operating conditions, chilled storage and generation calculation.
7. Optionally apply EN 15316-4-3/4-6 solar thermal and PV.
8. Run EN 15316-4-2 heat-pump, EN 15316-4-1 boiler, EN 15316-4-4
   cogeneration or EN 15316-4-5 district generation for heating and DHW.
9. Save the intermediate loads, generation balances and summary outputs.

Default weather uses PVGIS for the selected scenario. If network access is not
available, run with ``--weather-source epw --path-weather-file path/to/weather.epw``.
"""

from __future__ import annotations

import argparse
import copy
import html
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pybuildingenergy as pybui  # noqa: E402


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*ChainedAssignmentError.*",
)


SCENARIO_DHW_COUNTRY = {
    "athens": "Greece",
    "bolzano": "Italy",
}


def default_output_dir(
    scenario: str,
    emission_method: str = "en15316-2",
    distribution_method: str = "en15316-3",
    storage_method: str = "en15316-5",
    cooling_system_method: str = "en16798-9",
    cooling_storage_method: str = "en16798-15",
    cooling_generation_method: str = "en16798-13",
    performance_data_method: str = "en14511-14825",
    dhw_design_method: str = "en12831-3",
    heating_generation_method: str = "en15316-4-2",
    renewable_method: str = "simple",
    primary_energy_method: str = "simple",
    lighting_method: str = "simple",
    ventilation_method: str = "simple",
    bacs_method: str = "simple",
    economic_method: str = "simple",
) -> Path:
    suffix = f"heat_pump_15316_4_2_{scenario}"
    if heating_generation_method != "en15316-4-2":
        suffix = f"{scenario}_{heating_generation_method.replace('-', '_')}"
    if (
        emission_method == "simple"
        and distribution_method == "simple"
        and storage_method == "simple"
        and cooling_system_method == "simple"
        and cooling_storage_method == "simple"
        and cooling_generation_method == "heat-pump-simple"
    ):
        suffix = f"{suffix}_simple"
    elif (
        emission_method == "en15316-2"
        and distribution_method == "simple"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_emission_only"
    elif (
        emission_method == "en15316-2"
        and distribution_method == "en15316-3"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_emission_distribution"
    elif (
        emission_method == "en15316-2"
        and distribution_method == "simple"
        and storage_method == "en15316-5"
    ):
        suffix = f"{suffix}_emission_storage"
    elif (
        emission_method == "simple"
        and distribution_method == "en15316-3"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_distribution_only"
    elif (
        emission_method == "simple"
        and distribution_method == "simple"
        and storage_method == "en15316-5"
    ):
        suffix = f"{suffix}_storage_only"
    elif (
        emission_method == "simple"
        and distribution_method == "en15316-3"
        and storage_method == "en15316-5"
    ):
        suffix = f"{suffix}_distribution_storage"
    if cooling_generation_method == "heat-pump-simple" and not suffix.endswith("_simple"):
        suffix = f"{suffix}_heat_pump_cooling"
    elif cooling_storage_method == "simple" and cooling_generation_method == "en16798-13":
        suffix = f"{suffix}_no_cooling_storage"
    if performance_data_method == "simple" and not suffix.endswith("_simple"):
        suffix = f"{suffix}_simple_performance"
    if dhw_design_method == "simple" and not suffix.endswith("_simple"):
        suffix = f"{suffix}_simple_dhw_design"
    if renewable_method == "en15316-4-3-4-6":
        suffix = f"{suffix}_renewables"
    if primary_energy_method == "eniso52000-1":
        suffix = f"{suffix}_primary"
    if lighting_method == "en15193-1":
        suffix = f"{suffix}_lighting"
    if ventilation_method == "en16798-5-7":
        suffix = f"{suffix}_ventilation"
    if bacs_method == "eniso52120-1":
        suffix = f"{suffix}_bacs"
    if economic_method == "en15459-1":
        suffix = f"{suffix}_cost"
    return REPO_ROOT / "examples" / "outputs" / suffix


def example_building(scenario: str = "athens") -> dict:
    """A compact residential building intended to show heating and cooling."""

    if scenario not in SCENARIO_DHW_COUNTRY:
        available = ", ".join(SCENARIO_DHW_COUNTRY)
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {available}")

    weekday_on = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    weekend_cooling = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    n_floors = 2
    footprint_length = 10.0
    footprint_width = 6.0
    footprint_area = footprint_length * footprint_width
    net_floor_area = footprint_area * n_floors
    floor_height = 3.0
    total_height = floor_height * n_floors
    north_south_wall_area = footprint_length * total_height
    east_west_wall_area = footprint_width * total_height
    south_glazing_area = 12.0
    east_glazing_area = 6.0
    west_glazing_area = 6.0
    window_height = 1.6

    building = {
        "building": {
            "name": "Athens_heat_pump_demo",
            "azimuth_relative_to_true_north": 0,
            "latitude": 37.9888,
            "longitude": 23.7335,
            "exposed_perimeter": 2.0 * (footprint_length + footprint_width),
            "height": total_height,
            "floor_height": floor_height,
            "wall_thickness": 0.35,
            "n_floors": n_floors,
            "footprint_length": footprint_length,
            "footprint_width": footprint_width,
            "footprint_area": footprint_area,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "net_floor_area": net_floor_area,
            "construction_class": "class_i",
            "construction_year": "2010-today",
            "country": "Greece",
        },
        "adjacent_zones": [],
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": footprint_area,
                "sky_view_factor": 1.0,
                "u_value": 0.45,
                "solar_absorptance": 0.5,
                "thermal_capacity": 741500.0,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None,
            },
            {
                "name": "North wall",
                "type": "opaque",
                "area": north_south_wall_area,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.5,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 0, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "South wall",
                "type": "opaque",
                "area": north_south_wall_area - south_glazing_area,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.65,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 180, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "East wall",
                "type": "opaque",
                "area": east_west_wall_area - east_glazing_area,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.6,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 90, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "West wall",
                "type": "opaque",
                "area": east_west_wall_area - west_glazing_area,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.65,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 270, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "Slab to ground",
                "type": "opaque",
                "area": footprint_area,
                "sky_view_factor": 0.0,
                "u_value": 0.5,
                "solar_absorptance": 0.6,
                "thermal_capacity": 405801,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None,
            },
            {
                "name": "South glazing",
                "type": "transparent",
                "area": south_glazing_area,
                "sky_view_factor": 0.5,
                "u_value": 1.7,
                "g_value": 0.55,
                "height": window_height,
                "width": south_glazing_area / window_height,
                "parapet": 0.9,
                "orientation": {"azimuth": 180, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.0,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 0.0},
                "name_adj_zone": None,
            },
            {
                "name": "East glazing",
                "type": "transparent",
                "area": east_glazing_area,
                "sky_view_factor": 0.5,
                "u_value": 1.7,
                "g_value": 0.55,
                "height": window_height,
                "width": east_glazing_area / window_height,
                "parapet": 0.9,
                "orientation": {"azimuth": 90, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.0,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 0.0},
                "name_adj_zone": None,
            },
            {
                "name": "West glazing",
                "type": "transparent",
                "area": west_glazing_area,
                "sky_view_factor": 0.5,
                "u_value": 1.7,
                "g_value": 0.55,
                "height": window_height,
                "width": west_glazing_area / window_height,
                "parapet": 0.9,
                "orientation": {"azimuth": 270, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.0,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 0.0},
                "name_adj_zone": None,
            },
        ],
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "C",
            },
            "system_capacities": {
                "heating_capacity": 10000000.0,
                "cooling_capacity": 12000000.0,
                "units": "W",
            },
            "ventilation": {
                "ventilation_type": "custom",
                "flow_rate_per_person": 0.5,
                "units": "W/K when custom ventilation is selected",
                "custom_heat_transfer_coefficient_ventilation": 30.0,
                "info": "ventilation type can be Occupancy, temp_wind or custom.",
            },
            "airflow_rates": {
                "infiltration_rate": 0.7,
                "units": "ACH",
            },
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": 5.0,
                    "weekday": [1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.6, 0.6, 0.8, 0.9, 0.9, 0.9, 1, 1],
                    "weekend": [1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1, 1],
                },
                {
                    "name": "appliances",
                    "full_load": 4.0,
                    "weekday": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 0.8, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.8, 0.8, 1, 1, 1, 0.7, 0.7],
                    "weekend": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 0.8, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.8, 0.8, 1, 1, 1, 0.7, 0.7],
                },
                {
                    "name": "lighting",
                    "full_load": 3.0,
                    "weekday": [0, 0, 0, 0, 0, 0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.15, 0.15],
                    "weekend": [0, 0, 0, 0, 0, 0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.15, 0.15],
                },
            ],
            "construction": {
                "wall_thickness": 0.35,
                "thermal_bridges": 1.5,
                "units": "m and W/K",
            },
            "climate_parameters": {
                "coldest_month": 1,
                "units": "1-12",
            },
            "heating_profile": {"weekday": weekday_on, "weekend": weekday_on},
            "cooling_profile": {"weekday": weekday_on, "weekend": weekend_cooling},
            "ventilation_profile": {"weekday": weekday_on, "weekend": weekend_cooling},
        },
    }

    if scenario == "bolzano":
        building["building"].update(
            {
                "name": "Bolzano_heat_pump_demo",
                "latitude": 46.4983,
                "longitude": 11.3548,
                "construction_year": "1991-2005",
                "construction_class": "class_e",
                "country": "Italy",
            }
        )
        building["building_parameters"]["airflow_rates"]["infiltration_rate"] = 0.30
        building["building_parameters"]["ventilation"]["custom_heat_transfer_coefficient_ventilation"] = 20.0
        building["building_parameters"]["construction"]["thermal_bridges"] = 1.25

        surfaces = {surface["name"]: surface for surface in building["building_surface"]}
        surfaces["Roof surface"]["u_value"] = 0.22
        surfaces["Roof surface"]["solar_absorptance"] = 0.65
        surfaces["North wall"]["u_value"] = 0.30
        surfaces["South wall"]["u_value"] = 0.30
        surfaces["East wall"]["u_value"] = 0.30
        surfaces["West wall"]["u_value"] = 0.30
        surfaces["Slab to ground"]["u_value"] = 0.32

        for name in ["South glazing", "East glazing", "West glazing"]:
            surfaces[name]["u_value"] = 1.2
            surfaces[name]["g_value"] = 0.58

        surfaces["South glazing"]["area"] = 16.0
        surfaces["South glazing"]["width"] = 16.0 / window_height
        surfaces["South wall"]["area"] = north_south_wall_area - 16.0
        surfaces["East glazing"]["area"] = 8.0
        surfaces["East glazing"]["width"] = 8.0 / window_height
        surfaces["East wall"]["area"] = east_west_wall_area - 8.0
        surfaces["West glazing"]["area"] = 8.0
        surfaces["West glazing"]["width"] = 8.0 / window_height
        surfaces["West wall"]["area"] = east_west_wall_area - 8.0

    return building


def run_iso52016(building: dict, weather_source: str, path_weather_file: str | None) -> pd.DataFrame:
    """Run ISO52016 and return the hourly load table."""

    checked, issues = pybui.sanitize_and_validate_BUI(building, fix=True)
    errors = [issue for issue in issues if issue["level"] == "ERROR"]
    if errors:
        details = "\n".join(f"- {item['path']}: {item['msg']}" for item in errors)
        raise RuntimeError(f"The example building is not valid:\n{details}")

    kwargs = {"weather_source": weather_source}
    if weather_source == "epw":
        if not path_weather_file:
            raise ValueError("--path-weather-file is required when --weather-source epw.")
        kwargs["path_weather_file"] = path_weather_file

    result = pybui.ISO52016.Temperature_and_Energy_needs_calculation(checked, **kwargs)
    if not isinstance(result, tuple) or len(result) < 1:
        raise RuntimeError("ISO52016 did not return an hourly result table.")
    return result[0].copy()


def dhw_cold_water_temperature_C(scenario: str) -> float:
    return 9.5 if scenario == "bolzano" else 11.2


DHW_DAY_TYPE_COLUMNS = ("Workday", "Weekend", "Holiday")

# EN 12831-3:2017 Annex B, Table B.2 gives relative hourly hot-water
# volume profiles by building category. The tabulated values are percentages
# rounded to one decimal, so some columns sum to 99.7 or 99.9 rather than
# exactly 100. The helper below normalizes each profile before it is used.
DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT = {
    "single_family_dwelling": [
        1.8, 1.0, 0.6, 0.3, 0.4, 0.6,
        2.4, 4.7, 6.8, 5.7, 6.1, 6.1,
        6.3, 6.4, 5.1, 4.4, 4.3, 4.7,
        5.7, 6.5, 6.6, 5.8, 4.5, 3.1,
    ],
    "apartment_dwelling": [
        1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        3.0, 6.0, 8.0, 6.0, 5.0, 5.0,
        6.0, 6.0, 5.0, 4.0, 4.0, 5.0,
        6.0, 7.0, 7.0, 6.0, 5.0, 2.0,
    ],
    "residential_home_elderly": [
        0.3, 0.3, 0.4, 0.7, 1.0, 1.8,
        9.3, 15.7, 8.1, 7.5, 7.0, 6.6,
        7.1, 5.1, 3.8, 3.3, 4.1, 2.9,
        6.1, 4.1, 1.4, 1.8, 0.9, 0.4,
    ],
    "student_residence": [
        1.4, 1.0, 0.5, 0.6, 1.3, 3.4,
        5.8, 5.8, 6.2, 5.4, 5.1, 4.7,
        4.2, 4.5, 4.1, 4.3, 5.3, 6.0,
        6.6, 6.0, 5.6, 5.4, 3.9, 2.8,
    ],
    "hospital": [
        0.4, 0.4, 0.5, 0.8, 1.2, 2.8,
        7.5, 10.5, 8.0, 7.5, 7.5, 7.0,
        7.5, 5.5, 4.3, 3.7, 4.5, 3.2,
        7.0, 4.5, 2.0, 2.0, 1.2, 0.5,
    ],
}

# Both example buildings are residential, two-floor, 120 m2 useful-area houses.
# The single-family dwelling curve is therefore used for both climates; the
# Workalendar country still gives scenario-specific workday/holiday calendars.
SCENARIO_DHW_ANNEX_B_PROFILE = {
    "athens": "single_family_dwelling",
    "bolzano": "single_family_dwelling",
}


def dhw_annex_b_table_b2_hourly_fractions(
    profile_key: str = "single_family_dwelling",
) -> pd.DataFrame:
    """Return EN 12831-3 Annex B Table B.2 hourly fractions for the examples.

    ``Volume_and_energy_DHW_calculation`` expects one 24-hour vector for each
    day type. Table B.2 is category-based, not day-type-based, so the selected
    category profile is reused for workdays, weekends and holidays. The values
    are normalized to fractions that sum to 1.0 per day type.
    """

    if profile_key not in DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT:
        available = ", ".join(sorted(DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT))
        raise ValueError(
            f"Unknown EN 12831-3 Annex B Table B.2 profile '{profile_key}'. "
            f"Available profiles: {available}."
        )
    percent = pd.Series(
        DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT[profile_key],
        index=range(24),
        dtype=float,
    )
    if len(percent) != 24:
        raise ValueError(f"DHW profile '{profile_key}' must contain 24 hourly values.")
    total = float(percent.sum())
    if total <= 0.0:
        raise ValueError(f"DHW profile '{profile_key}' must have a positive sum.")
    fractions = percent / total
    return pd.DataFrame({column: fractions.values for column in DHW_DAY_TYPE_COLUMNS})


def make_dhw_profile(
    index: pd.DatetimeIndex,
    building_area: float,
    country: str,
    cold_water_temperature_C: float = 11.2,
    annex_b_profile_key: str = "single_family_dwelling",
) -> pd.Series:
    """Calculate hourly DHW needs and align them to an ISO52016 hourly index."""

    target_index = pd.DatetimeIndex(index)
    if target_index.tz is not None:
        target_index = target_index.tz_convert(None)
    if target_index.empty:
        return pd.Series(index=target_index, dtype=float, name="Q_W_kWh")

    hourly_fractions = dhw_annex_b_table_b2_hourly_fractions(
        annex_b_profile_key
    )
    sum_fractions = pd.DataFrame(hourly_fractions.sum(), columns=["fractions"])

    annual_profiles = []
    for year in sorted(set(target_index.year)):
        calendar = pybui.generate_calendar(country, int(year))
        n_workdays = int((calendar["values"] == "Working").sum())
        n_weekends = int((calendar["values"] == "Non-Working").sum())
        n_holidays = int((calendar["values"] == "Holiday").sum())
        total_days = int(calendar["values"].count())

        dhw_result = pybui.Volume_and_energy_DHW_calculation(
            n_workdays,
            n_weekends,
            n_holidays,
            sum_fractions,
            total_days,
            hourly_fractions,
            42.0,
            13.5,
            60.0,
            cold_water_temperature_C,
            mode_calc="number_of_units",
            building_type_B3="Residential",
            building_area=building_area,
            unit_count=4,
            building_type_B5="Dwelling",
            residential_typology="residential_building - simple housing - AVG",
            calculation_method="table",
            year=int(year),
            country_calendar=calendar,
        )
        values = pd.Series(dhw_result[7], name="Q_W_kWh")
        year_index = pd.date_range(f"{year}-01-01 00:00:00", periods=len(values), freq="h")
        values.index = year_index
        annual_profiles.append(values)

    profile = pd.concat(annual_profiles).sort_index()
    if not profile.index.is_unique:
        profile = profile.groupby(level=0).mean().sort_index()

    reindexed = profile.reindex(target_index)
    if reindexed is None:
        lookup = profile.to_dict()
        aligned_values = np.array([lookup.get(ts, np.nan) for ts in target_index], dtype=float)
    else:
        aligned_values = pd.to_numeric(reindexed, errors="coerce").to_numpy(dtype=float)

    missing = np.isnan(aligned_values)
    if missing.any():
        valid_positions = np.flatnonzero(~missing)
        if valid_positions.size == 0:
            raise ValueError("DHW profile alignment produced only missing values.")

        positions = np.arange(len(aligned_values))
        last_valid = np.maximum.accumulate(np.where(~missing, positions, -1))
        has_previous = last_valid >= 0
        filled = aligned_values.copy()
        filled[has_previous] = aligned_values[last_valid[has_previous]]

        next_valid = np.minimum.accumulate(np.where(~missing, positions, len(aligned_values))[::-1])[::-1]
        needs_backfill = np.isnan(filled) & (next_valid < len(aligned_values))
        filled[needs_backfill] = aligned_values[next_valid[needs_backfill]]
        aligned_values = filled

    return pd.Series(aligned_values, index=target_index, name="Q_W_kWh", dtype=float)


def _nearest_capacity_kW(
    performance_map: pd.DataFrame,
    source_temperature_C: float,
    sink_temperature_C: float,
) -> float:
    if performance_map.empty or "capacity_kW" not in performance_map:
        return 0.0
    df = performance_map.copy()
    df["_distance"] = (
        (df["source_temperature_C"].astype(float) - source_temperature_C).abs()
        + (df["sink_temperature_C"].astype(float) - sink_temperature_C).abs()
    )
    return float(df.sort_values("_distance").iloc[0]["capacity_kW"])


def dhw_design_system_config(
    scenario: str,
    distribution_config: dict,
    storage_config: dict,
    heating_map: pd.DataFrame,
    sizing_mode: str,
) -> dict:
    """Build EN 12831-3 DHW design-load inputs aligned with later modules."""

    dhw_storage = storage_config["dhw"]
    dhw_distribution = distribution_config["dhw"]
    source_design_C = -7.0 if scenario == "bolzano" else 2.0
    sink_design_C = float(dhw_storage.get("set_temperature_C", 55.0))
    nominal_power_kW = _nearest_capacity_kW(
        heating_map,
        source_temperature_C=source_design_C,
        sink_temperature_C=sink_design_C,
    )
    if nominal_power_kW <= 0.0:
        nominal_power_kW = 6.0 if scenario == "bolzano" else 7.0

    config = {
        "system_type": "loading_storage",
        "sizing_mode": sizing_mode,
        "demand_unit": "kWh",
        "storage_volume_l": float(dhw_storage.get("storage_volume_l", 180.0)),
        "storage_height_m": 1.25 if scenario == "bolzano" else 1.20,
        "sensor_relative_height": 0.35,
        "loading_factor": 1.0,
        "storage_max_temperature_C": float(dhw_storage.get("set_temperature_C", 55.0)),
        "draw_temperature_C": 42.0,
        "cold_water_temperature_C": dhw_cold_water_temperature_C(scenario),
        "ambient_temperature_C": float(dhw_storage.get("ambient_temperature_C", 16.0)),
        "charging_temperature_C": min(
            60.0,
            float(dhw_storage.get("set_temperature_C", 55.0)) + 5.0,
        ),
        "nominal_power_kW": nominal_power_kW,
        "heat_exchanger_power_kW": nominal_power_kW,
        "heat_generator_type": "heat_pump",
        "distribution_pipe_sections": dhw_distribution.get("pipe_sections", []),
        "distribution_length_m": float(dhw_distribution.get("max_length_m", 0.0)),
        "specific_distribution_loss_W_m": 11.0,
        "pipe_mean_temperature_C": 50.0,
        "mixed_storage_time_constant_min": 55.0 if scenario == "bolzano" else 45.0,
        "max_storage_volume_l": 600.0,
    }
    # EN 12831-3 Annex B Table B.8 is used by default for DHW design standby
    # loss. Provide one of these keys only when project/product data should
    # override the Annex B.8 interpolation.
    for key in (
        "dhw_design_standby_loss_kWh_per_day",
        "standby_loss_kWh_per_day",
        "q_sb_sto_kWh_d",
    ):
        if key in dhw_storage:
            config["standby_loss_kWh_per_day"] = float(dhw_storage[key])
            break
    return config


def apply_dhw_design_to_system_configs(
    dhw_design: pybui.DHWDesignSimulationResult | None,
    distribution_config: dict,
    storage_config: dict,
) -> None:
    """Use EN 12831-3 sizing outputs as downstream EN 15316 inputs."""

    if dhw_design is None:
        return
    summary = dhw_design.summary
    selected_volume_l = float(summary.get("V_sto_selected_l", 0.0) or 0.0)
    if selected_volume_l > 0.0:
        storage_config["dhw"]["storage_volume_l"] = selected_volume_l
    q_sb = float(summary.get("q_sb_sto_kWh_d", 0.0) or 0.0)
    if q_sb > 0.0:
        storage_config["dhw"]["standby_loss_kWh_per_day_ref"] = q_sb
    design_flow = float(summary.get("design_flow_m3_h", 0.0) or 0.0)
    if design_flow > 0.0:
        distribution_config["dhw"]["design_flow_m3_h"] = max(
            float(distribution_config["dhw"].get("design_flow_m3_h", 0.0)),
            design_flow,
        )


def heat_pump_maps(scenario: str = "athens") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Example product maps. Replace with manufacturer/test data for real studies."""

    if scenario == "bolzano":
        heating_capacity_base = 8.4
        heating_capacity_source_factor = 0.13
        heating_capacity_sink_factor = 0.055
        heating_cop_base = 4.05
        cooling_capacity_base = 6.2
        cooling_capacity_source_factor = 0.07
        cooling_capacity_sink_factor = 0.035
        cooling_eer_base = 3.75
    else:
        heating_capacity_base = 9.0
        heating_capacity_source_factor = 0.12
        heating_capacity_sink_factor = 0.05
        heating_cop_base = 4.1
        cooling_capacity_base = 8.0
        cooling_capacity_source_factor = 0.08
        cooling_capacity_sink_factor = 0.04
        cooling_eer_base = 3.9

    heating_rows = []
    for source in [-15, -7, 2, 7, 12, 20]:
        for sink in [35, 45, 55]:
            cop = heating_cop_base + 0.055 * source - 0.045 * (sink - 35)
            capacity = (
                heating_capacity_base
                + heating_capacity_source_factor * source
                - heating_capacity_sink_factor * (sink - 35)
            )
            heating_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "cop": max(cop, 1.6),
                }
            )

    cooling_rows = []
    for source in [20, 25, 30, 35, 40]:
        for sink in [7, 12, 18]:
            eer = cooling_eer_base - 0.055 * (source - 25) + 0.045 * (sink - 7)
            capacity = (
                cooling_capacity_base
                - cooling_capacity_source_factor * (source - 25)
                + cooling_capacity_sink_factor * (sink - 7)
            )
            cooling_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "eer": max(eer, 1.7),
                }
            )

    return pd.DataFrame(heating_rows), pd.DataFrame(cooling_rows)


def heat_pump_performance_data(
    scenario: str = "athens",
) -> pybui.HeatPumpPerformanceDataResult:
    """Build example maps from EN 14511/EN 14825 style rating rows."""

    if scenario == "bolzano":
        heating_design_load_kW = 8.0
        cooling_design_load_kW = 5.8
        heating_capacity_a7w35 = 8.9
        heating_cop_a7w35 = 4.55
        cooling_capacity_a35w7 = 6.2
        cooling_eer_a35w7 = 3.45
    else:
        heating_design_load_kW = 6.7
        cooling_design_load_kW = 7.6
        heating_capacity_a7w35 = 8.8
        heating_cop_a7w35 = 4.60
        cooling_capacity_a35w7 = 8.0
        cooling_eer_a35w7 = 3.25

    heating_part_load_ratios = {
        -7: 0.8846,
        2: 0.5385,
        7: 0.3462,
        12: 0.1538,
    }
    cooling_part_load_ratios = {
        35: 1.0000,
        30: 0.7368,
        25: 0.4737,
        20: 0.2105,
    }

    heating_rows = []
    for source in [-15, -7, 2, 7, 12, 20]:
        for sink in [35, 45, 55]:
            capacity = (
                heating_capacity_a7w35
                + 0.11 * (source - 7.0)
                - 0.055 * (sink - 35.0)
            )
            cop = heating_cop_a7w35 + 0.055 * (source - 7.0) - 0.045 * (sink - 35.0)
            if source == 7:
                condition = f"EN 14511-2 standard A7/W{sink}; EN 14825 C"
            elif source in heating_part_load_ratios:
                condition = f"EN 14825 heating point at {source:+g}C/W{sink}"
            else:
                condition = f"EN 14511-2 application A{source:+g}/W{sink}"
            heating_rows.append(
                {
                    "rating_condition": condition,
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "cop": max(cop, 1.6),
                    "part_load_ratio": heating_part_load_ratios.get(source, 1.0),
                    "test_standard": "EN 14511-2:2022 / EN 14825:2022",
                }
            )

    cooling_rows = []
    for source in [20, 25, 27, 30, 35, 40, 46]:
        for sink in [7, 12, 18]:
            capacity = (
                cooling_capacity_a35w7
                - 0.07 * (source - 35.0)
                + 0.035 * (sink - 7.0)
            )
            eer = cooling_eer_a35w7 - 0.055 * (source - 35.0) + 0.045 * (sink - 7.0)
            if source == 35:
                condition = f"EN 14511-2 standard A35/W{sink}; EN 14825 A"
            elif source in cooling_part_load_ratios:
                condition = f"EN 14825 cooling point at {source:g}C/W{sink}"
            else:
                condition = f"EN 14511-2 application A{source:g}/W{sink}"
            cooling_rows.append(
                {
                    "rating_condition": condition,
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "eer": max(eer, 1.7),
                    "part_load_ratio": cooling_part_load_ratios.get(source, 1.0),
                    "test_standard": "EN 14511-2:2022 / EN 14825:2022",
                }
            )

    return pybui.HeatPumpPerformanceDataCalculator(
        {
            "unit_type": "air-to-water",
            "capacity_control": "fixed",
            "heating_design_load_kW": heating_design_load_kW,
            "cooling_design_load_kW": cooling_design_load_kW,
            "heating_degradation_coefficient": 0.9,
            "cooling_degradation_coefficient": 0.9,
            "heating_rating_points": heating_rows,
            "cooling_rating_points": cooling_rows,
        }
    ).run()


def heat_pump_config(
    scenario: str,
    heating_map: pd.DataFrame,
    cooling_map: pd.DataFrame,
    include_internal_storage_losses: bool = True,
    cooling_enabled: bool = True,
    performance_data_method: str = "en14511-14825",
) -> dict:
    """Scenario-specific generator assumptions for the example heat pump."""

    config = {
        "heating_performance_map": heating_map,
        "dhw_performance_map": heating_map,
        "cooling_performance_map": cooling_map,
        "source_type": "air",
        "demand_unit": "kWh",
        "bin_width_C": 1.0,
        "cooling_enabled": cooling_enabled,
        "design_outdoor_temperature_C": -3.0,
        "heating_cutoff_temperature_C": 16.0,
        "heating_sink_temp_at_design_C": 45.0,
        "heating_sink_temp_at_cutoff_C": 30.0,
        "dhw_target_temperature_C": 60.0,
        "dhw_sink_temperature_C": 55.0,
        "dhw_cold_water_temperature_C": 11.2,
        "cooling_sink_temperature_C": 7.0,
        "hp_operating_limit_C": 55.0,
        "backup_mode": "parallel",
        "heating_backup_efficiency": 1.0,
        "dhw_backup_efficiency": 1.0,
        "external_auxiliary_power_W": 120.0,
        "standby_power_W": 8.0,
        "heating_storage_loss_kWh_per_day": 0.15,
        "dhw_storage_loss_kWh_per_day": 0.90,
        "storage_test_deltaT_K": 45.0,
        "storage_ambient_temperature_C": 20.0,
    }
    if performance_data_method == "en14511-14825":
        config.update(
            {
                "part_load_performance_method": "en14825",
                "part_load_unit_type": "air-to-water",
                "part_load_degradation_coefficient": 0.9,
                "heating_part_load_degradation_coefficient": 0.9,
                "dhw_part_load_degradation_coefficient": 0.9,
                "cooling_part_load_degradation_coefficient": 0.9,
                "heating_part_load_minimum_capacity_ratio": 0.1538,
                "dhw_part_load_minimum_capacity_ratio": 0.1538,
                "cooling_part_load_minimum_capacity_ratio": 0.2105,
            }
        )
    else:
        config["part_load_performance_method"] = "simple"

    if scenario == "bolzano":
        config.update(
            {
                "design_outdoor_temperature_C": -7.0,
                "heating_sink_temp_at_design_C": 45.0,
                "heating_sink_temp_at_cutoff_C": 30.0,
                "dhw_cold_water_temperature_C": 9.5,
                "external_auxiliary_power_W": 130.0,
                "heating_storage_loss_kWh_per_day": 0.20,
                "dhw_storage_loss_kWh_per_day": 0.95,
            }
        )

    if not include_internal_storage_losses:
        config["heating_storage_loss_kWh_per_day"] = 0.0
        config["dhw_storage_loss_kWh_per_day"] = 0.0

    return config


def boiler_generation_config(scenario: str = "athens") -> dict:
    """Scenario-specific EN 15316-4-1 combustion-generator assumptions."""

    config = {
        "demand_unit": "kWh",
        "fuel": "natural_gas",
        "condensing": True,
        "nominal_power_kW": 18.0,
        "full_load_efficiency": 0.94,
        "part_load_efficiency": 0.98,
        "efficiency_max": 1.08,
        "minimum_part_load_ratio": 0.15,
        "return_temperature_reference_C": 45.0,
        "return_temperature_efficiency_slope_per_K": 0.0020,
        "standby_loss_kWh_per_h": 0.010,
        "auxiliary_power_kW": 0.035,
        "standby_auxiliary_power_kW": 0.004,
        "thermal_loss_room_fraction": 0.75,
        "auxiliary_to_medium_fraction": 0.20,
        "heating_supply_temperature_C": 45.0,
        "heating_return_temperature_C": 35.0,
        "dhw_sink_temperature_C": 55.0,
        "dhw_return_temperature_C": 45.0,
    }
    if scenario == "bolzano":
        config.update(
            {
                "nominal_power_kW": 22.0,
                "full_load_efficiency": 0.95,
                "part_load_efficiency": 1.00,
                "standby_loss_kWh_per_h": 0.012,
                "auxiliary_power_kW": 0.040,
            }
        )
    return config


def cogeneration_system_config(scenario: str = "athens") -> dict:
    """Scenario-specific EN 15316-4-4 thermal-led CHP assumptions."""

    config = {
        "demand_unit": "kWh",
        "fuel": "natural_gas",
        "nominal_thermal_power_kW": 6.0,
        "thermal_efficiency": 0.56,
        "electrical_efficiency": 0.30,
        "backup_efficiency": 0.94,
        "minimum_part_load_ratio": 0.35,
        "auxiliary_power_kW": 0.030,
        "standby_auxiliary_power_kW": 0.004,
        "loss_recoverable_fraction": 0.50,
    }
    if scenario == "bolzano":
        config.update(
            {
                "nominal_thermal_power_kW": 8.0,
                "thermal_efficiency": 0.57,
                "electrical_efficiency": 0.29,
                "backup_efficiency": 0.95,
            }
        )
    return config


def district_energy_system_config(scenario: str = "athens") -> dict:
    """Scenario-specific EN 15316-4-5 district energy assumptions."""

    config = {
        "demand_unit": "kWh",
        "heating_enabled": True,
        "dhw_enabled": True,
        "cooling_enabled": False,
        "heating_substation_efficiency": 0.97,
        "dhw_substation_efficiency": 0.96,
        "heating_fixed_loss_kWh_per_h": 0.006,
        "heating_auxiliary_power_kW": 0.025,
        "loss_recoverable_fraction": 0.50,
    }
    if scenario == "bolzano":
        config.update(
            {
                "heating_substation_efficiency": 0.98,
                "dhw_substation_efficiency": 0.97,
                "heating_fixed_loss_kWh_per_h": 0.008,
                "heating_auxiliary_power_kW": 0.030,
            }
        )
    return config


def renewable_system_config(scenario: str, building: dict) -> dict:
    """Scenario-specific EN 15316-4-3/4-6 renewable assumptions."""

    area = float(building["building"]["net_floor_area"])
    solar_area = max(2.0, min(6.0, area / 35.0))
    pv_kWp = max(1.5, min(4.0, area / 35.0))
    if scenario == "bolzano":
        solar_yield = 410.0
        pv_yield = 1120.0
    else:
        solar_yield = 560.0
        pv_yield = 1500.0
    return {
        "demand_unit": "kWh",
        "solar_thermal": {
            "enabled": True,
            "area_m2": solar_area,
            "annual_yield_kWh_m2": solar_yield,
            "collector_loop_efficiency": 0.85,
            "priority": "dhw_first",
        },
        "pv": {
            "enabled": True,
            "capacity_kWp": pv_kWp,
            "annual_yield_kWh_kWp": pv_yield,
            "performance_ratio": 0.80,
            "self_consumption_fraction": 0.70,
        },
    }


def lighting_system_config(scenario: str, building: dict) -> dict:
    """Scenario-specific EN 15193-1 lighting assumptions."""

    area = float(building["building"]["net_floor_area"])
    config = {
        "area_m2": area,
        "installed_power_density_W_m2": 4.5,
        "daylight_dependency_factor": 0.86,
        "occupancy_dependency_factor": 0.90,
        "constant_illuminance_factor": 0.95,
        "daylight_control_fraction": 0.35,
        "occupancy_control_fraction": 0.25,
        "parasitic_power_W_m2": 0.02,
        "emergency_lighting_kWh_m2_a": 0.0,
    }
    if scenario == "bolzano":
        config["daylight_control_fraction"] = 0.30
    return config


def ventilation_system_config(scenario: str, building: dict) -> dict:
    """Scenario-specific EN 16798-5/7 ventilation assumptions."""

    area = float(building["building"]["net_floor_area"])
    volume = area * float(building["building"].get("floor_height", 3.0))
    config = {
        "volume_m3": volume,
        "air_change_rate_h": 0.45,
        "heat_recovery_efficiency": 0.75,
        "mechanical_fraction": 1.0,
        "specific_fan_power_W_s_m3": 850.0,
        "fan_heat_to_air_fraction": 0.6,
    }
    if scenario == "bolzano":
        config.update({"air_change_rate_h": 0.50, "heat_recovery_efficiency": 0.80})
    return config


def apply_ventilation_to_building(building: dict, config: dict) -> dict:
    """Return a building copy with EN 16798-5/7 effective Hve in ISO52016."""

    adjusted = copy.deepcopy(building)
    h_ve = pybui.VentilationSystemCalculator(config).effective_heat_transfer_coefficient_W_K()
    ventilation = adjusted["building_parameters"].setdefault("ventilation", {})
    ventilation["ventilation_type"] = "custom"
    ventilation["custom_heat_transfer_coefficient_ventilation"] = h_ve
    ventilation["units"] = "W/K"
    adjusted["building_parameters"].setdefault("airflow_rates", {})[
        "infiltration_rate"
    ] = float(config.get("air_change_rate_h", 0.45))
    return adjusted


def primary_energy_config() -> dict:
    """Generic EN ISO 52000-1 factors used by the examples."""

    return {
        "carriers": ["natural_gas", "electricity", "district_heat", "biomass"],
        "primary_energy_factors": {
            "natural_gas": {"nonrenewable": 1.05, "renewable": 0.0},
            "electricity": {"nonrenewable": 2.17, "renewable": 0.0},
            "district_heat": {"nonrenewable": 0.80, "renewable": 0.20},
            "biomass": {"nonrenewable": 0.20, "renewable": 1.00},
        },
        "export_primary_energy_factors": {
            "electricity": {"nonrenewable": 2.17, "renewable": 0.0}
        },
        "export_credit_method": "symmetric",
    }


def cost_optimal_config(scenario: str, building: dict) -> dict:
    """Generic EN 15459-1 economics used by the examples."""

    area = float(building["building"]["net_floor_area"])
    prices = {
        "natural_gas": 0.12,
        "electricity": 0.30,
        "district_heat": 0.11,
        "biomass": 0.08,
    }
    if scenario == "bolzano":
        prices["electricity"] = 0.32
        prices["district_heat"] = 0.12
    return {
        "calculation_period_years": 30,
        "real_discount_rate": 0.03,
        "floor_area_m2": area,
        "energy_prices": prices,
        "export_prices": {"electricity": 0.06},
    }


def emission_system_config(scenario: str = "athens") -> dict:
    """Scenario-specific EN 15316-2 emission assumptions.

    The defaults represent water-based fan-coil emission with PI room control,
    balanced hydraulics and stand-alone room automation. Values are taken from
    EN 15316-2:2017 Annex B default tables where available.
    """

    config = {
        "demand_unit": "kWh",
        "cooling_solar_gain_temperature_C": 8.0,
        "heating": {
            "stratification_K": 0.35,
            "control_K": 0.70,
            "radiation_K": 0.0,
            "hydraulic_balancing_K": 0.10,
            "room_automation_K": -0.50,
            "embedded_K": 0.0,
            "nominal_power_kW": 10.0,
            "fan_power_W": 10.0,
            "fan_count": 4.0,
            "control_power_W": 0.1,
            "control_count": 4.0,
            "convective_fraction": 0.95,
        },
        "cooling": {
            "stratification_K": 0.40,
            "control_K": 0.70,
            "radiation_K": 0.0,
            "hydraulic_balancing_K": 0.10,
            "room_automation_K": -0.50,
            "embedded_K": 0.0,
            "nominal_power_kW": 8.0,
            "fan_power_W": 10.0,
            "fan_count": 4.0,
            "control_power_W": 0.1,
            "control_count": 4.0,
            "convective_fraction": 0.95,
        },
    }

    if scenario == "bolzano":
        config["heating"]["nominal_power_kW"] = 11.0
        config["cooling"]["nominal_power_kW"] = 6.5

    return config


def building_with_emission_setpoints(
    building: dict,
    emission_calc: pybui.EmissionSystemCalculator,
) -> dict:
    """Return a copy of the building with EN 15316-2 equivalent setpoints."""

    adjusted = copy.deepcopy(building)
    setpoints = adjusted["building_parameters"]["temperature_setpoints"]
    h_delta = emission_calc.temperature_increase_K("H")
    c_delta = emission_calc.temperature_increase_K("C")

    for key in ["heating_setpoint", "heating_setback"]:
        if key in setpoints:
            setpoints[key] = float(setpoints[key]) + h_delta
    for key in ["cooling_setpoint", "cooling_setback"]:
        if key in setpoints:
            setpoints[key] = float(setpoints[key]) - c_delta

    return adjusted


def prepare_emission_loads(
    hourly_sim: pd.DataFrame,
    hourly_sim_emission: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare baseline and equivalent-setpoint loads for EN 15316-2."""

    loads = pd.DataFrame(index=hourly_sim.index)
    loads["T_ext"] = hourly_sim["T_ext"].astype(float)
    loads["T_op"] = hourly_sim["T_op"].astype(float)

    if "Q_H" in hourly_sim:
        loads["Q_H_kWh"] = hourly_sim["Q_H"].astype(float) / 1000.0
    else:
        loads["Q_H_kWh"] = hourly_sim["Q_HC"].clip(lower=0).astype(float) / 1000.0

    if "Q_C" in hourly_sim:
        loads["Q_C_kWh"] = hourly_sim["Q_C"].astype(float) / 1000.0
    else:
        loads["Q_C_kWh"] = (-hourly_sim["Q_HC"].clip(upper=0)).astype(float) / 1000.0

    emission_aligned = hourly_sim_emission.reindex(loads.index)
    if "Q_H" in emission_aligned:
        loads["Q_H_em_out_inc_kWh"] = emission_aligned["Q_H"].astype(float) / 1000.0
    else:
        loads["Q_H_em_out_inc_kWh"] = (
            emission_aligned["Q_HC"].clip(lower=0).astype(float) / 1000.0
        )

    if "Q_C" in emission_aligned:
        loads["Q_C_em_out_inc_kWh"] = emission_aligned["Q_C"].astype(float) / 1000.0
    else:
        loads["Q_C_em_out_inc_kWh"] = (
            -emission_aligned["Q_HC"].clip(upper=0).astype(float) / 1000.0
        )

    return loads


def _distribution_geometry(building: dict) -> tuple[float, float, float, int]:
    """Approximate rectangular Annex B dimensions from the example building."""

    n_floors = max(int(building["building"].get("n_floors", 1)), 1)
    floor_height = float(building["building"].get("height", 3.0)) / n_floors
    footprint_area = max(
        float(
            building["building"].get(
                "footprint_area",
                float(building["building"]["net_floor_area"]) / n_floors,
            )
        ),
        1.0,
    )
    if "footprint_length" in building["building"] and "footprint_width" in building["building"]:
        return (
            float(building["building"]["footprint_length"]),
            float(building["building"]["footprint_width"]),
            floor_height,
            n_floors,
        )
    aspect_ratio = 1.4
    width = float(np.sqrt(footprint_area / aspect_ratio))
    length = footprint_area / width
    return length, width, floor_height, n_floors


def building_geometry_summary(building: dict) -> dict[str, float]:
    """Return the main area quantities used by the example building."""

    building_data = building["building"]
    n_floors = max(int(building_data.get("n_floors", 1)), 1)
    net_floor_area = float(building_data["net_floor_area"])
    footprint_area = float(building_data.get("footprint_area", net_floor_area / n_floors))
    surfaces = {surface["name"]: surface for surface in building["building_surface"]}
    return {
        "net_floor_area_m2": net_floor_area,
        "n_floors": float(n_floors),
        "footprint_area_m2": footprint_area,
        "roof_area_m2": float(surfaces["Roof surface"]["area"]),
        "ground_slab_area_m2": float(surfaces["Slab to ground"]["area"]),
        "footprint_length_m": float(building_data.get("footprint_length", np.nan)),
        "footprint_width_m": float(building_data.get("footprint_width", np.nan)),
        "total_height_m": float(building_data.get("height", np.nan)),
        "floor_height_m": float(building_data.get("floor_height", np.nan)),
    }


def _pipe_psi_by_scenario(scenario: str) -> tuple[float, float, float]:
    return 0.20, 0.30, 0.30


def _space_distribution_sections(
    scenario: str,
    building: dict,
    service: str,
) -> tuple[list[dict], float]:
    """Build Annex B two-pipe sections V, S and A for the example building."""

    length, width, floor_height, n_floors = _distribution_geometry(building)
    psi_v, psi_s, psi_a = _pipe_psi_by_scenario(scenario)
    compact_routing_factor = 0.75
    section_v = compact_routing_factor * (2.0 * length + 0.0325 * length * width + 6.0)
    section_s = compact_routing_factor * (0.025 * length * width * floor_height * n_floors)
    section_a = compact_routing_factor * (0.55 * length * width * n_floors)
    ambient_v = 13.0 if service == "heating" else 26.0
    ambient_zone = 20.0 if service == "heating" else 26.0
    max_length = length + width + floor_height * n_floors + 10.0
    return (
        [
            {
                "name": "V base distributor",
                "length_m": section_v,
                "linear_thermal_transmittance_W_mK": psi_v,
                "ambient_temperature_C": ambient_v,
                "recoverable": False,
            },
            {
                "name": "S vertical shafts",
                "length_m": section_s,
                "linear_thermal_transmittance_W_mK": psi_s,
                "ambient_temperature_C": ambient_zone,
                "recoverable": True,
            },
            {
                "name": "A connection pipes",
                "length_m": section_a,
                "linear_thermal_transmittance_W_mK": psi_a,
                "ambient_temperature_C": ambient_zone,
                "recoverable": True,
            },
        ],
        max_length,
    )


def _dhw_distribution_sections(scenario: str, building: dict) -> tuple[list[dict], float]:
    """Build compact DHW distribution sections from EN 15316-3 Annex B."""

    length, width, floor_height, n_floors = _distribution_geometry(building)
    psi_v, psi_s, psi_a = _pipe_psi_by_scenario(scenario)
    circulation_fraction = 0.25
    section_v = circulation_fraction * (2.0 * length + 0.0125 * length * width)
    section_s = circulation_fraction * (0.075 * length * width * n_floors * floor_height)
    section_a = 0.075 * length * width * n_floors
    max_length = length + floor_height * n_floors + 2.5
    return (
        [
            {
                "name": "V compact DHW loop",
                "length_m": section_v,
                "linear_thermal_transmittance_W_mK": psi_v,
                "ambient_temperature_C": 13.0,
                "recoverable": False,
            },
            {
                "name": "S compact DHW shaft",
                "length_m": section_s,
                "linear_thermal_transmittance_W_mK": psi_s,
                "ambient_temperature_C": 20.0,
                "recoverable": True,
            },
            {
                "name": "A DHW branch pipes",
                "length_m": section_a,
                "linear_thermal_transmittance_W_mK": psi_a,
                "ambient_temperature_C": 20.0,
                "recoverable": True,
            },
        ],
        max_length,
    )


def distribution_system_config(scenario: str, building: dict) -> dict:
    """Scenario-specific EN 15316-3 distribution assumptions."""

    heating_sections, heating_max_length = _space_distribution_sections(
        scenario,
        building,
        service="heating",
    )
    cooling_sections, cooling_max_length = _space_distribution_sections(
        scenario,
        building,
        service="cooling",
    )
    dhw_sections, dhw_max_length = _dhw_distribution_sections(scenario, building)

    config = {
        "demand_unit": "kWh",
        "heating": {
            "pipe_sections": heating_sections,
            "supply_temperature_C": 45.0,
            "return_temperature_C": 35.0,
            "design_deltaT_K": 10.0,
            "nominal_power_kW": 10.0,
            "max_length_m": heating_max_length,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 6.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 4,
            "eei": 0.23,
            "recoverable_aux_fraction": 0.25,
        },
        "cooling": {
            "pipe_sections": cooling_sections,
            "supply_temperature_C": 7.0,
            "return_temperature_C": 12.0,
            "design_deltaT_K": 5.0,
            "nominal_power_kW": 8.0,
            "max_length_m": cooling_max_length,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 6.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 3,
            "eei": 0.23,
            "recoverable_aux_fraction": 0.25,
        },
        "dhw": {
            "pipe_sections": dhw_sections,
            "dhw_temperature_C": 55.0,
            "dhw_return_deltaT_K": 5.0,
            "design_flow_m3_h": 0.12,
            "max_length_m": dhw_max_length,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 6.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 3,
            "pump_label_power_kW": 0.020,
            "part_load_mode": "constant_when_on",
            "recoverable_aux_fraction": 0.25,
        },
    }

    if scenario == "bolzano":
        config["heating"]["nominal_power_kW"] = 11.0
        config["cooling"]["nominal_power_kW"] = 6.5
        config["dhw"]["dhw_temperature_C"] = 55.0

    return config


def storage_system_config(scenario: str) -> dict:
    """Scenario-specific EN 15316-5 storage assumptions.

    The example keeps EN 15316-5 connection losses at 1.0 because pipe and valve
    losses are already represented explicitly in the EN 15316-3 distribution
    module. Declared daily standby losses are converted internally to H_sto_ls.
    EN 12831-3 DHW design sizing uses Annex B Table B.8 standby losses by
    default; set dhw_design_standby_loss_kWh_per_day for a project-specific
    design override.
    """

    config = {
        "demand_unit": "kWh",
        "heating": {
            "storage_volume_l": 80.0,
            "set_temperature_C": 45.0,
            "output_temperature_C": 45.0,
            "ambient_temperature_C": 16.0,
            "standby_loss_kWh_per_day_ref": 0.15,
            "standby_set_temperature_ref_C": 45.0,
            "standby_ambient_temperature_ref_C": 20.0,
            "connection_loss_factor": 1.0,
            "standby_loss_adaptation_factor": 1.0,
            "thermal_loss_room_fraction": 0.75,
            "auxiliary_to_medium_fraction": 0.25,
            "input_pump_power_kW": 0.025,
            "input_pump_flow_m3_h": 0.90,
            "input_pump_deltaT_K": 10.0,
        },
        "dhw": {
            "storage_volume_l": 180.0,
            "set_temperature_C": 55.0,
            "output_temperature_C": 55.0,
            "ambient_temperature_C": 16.0,
            "standby_loss_kWh_per_day_ref": 0.90,
            "standby_set_temperature_ref_C": 55.0,
            "standby_ambient_temperature_ref_C": 20.0,
            "connection_loss_factor": 1.0,
            "standby_loss_adaptation_factor": 1.0,
            "thermal_loss_room_fraction": 0.75,
            "auxiliary_to_medium_fraction": 0.25,
            "input_pump_power_kW": 0.025,
            "input_pump_flow_m3_h": 0.50,
            "input_pump_deltaT_K": 5.0,
        },
    }

    if scenario == "bolzano":
        config["heating"].update(
            {
                "storage_volume_l": 100.0,
                "standby_loss_kWh_per_day_ref": 0.20,
                "input_pump_power_kW": 0.030,
                "input_pump_flow_m3_h": 1.00,
            }
        )
        config["dhw"]["standby_loss_kWh_per_day_ref"] = 0.95

    return config


def cooling_system_config(scenario: str) -> dict:
    """Scenario-specific EN 16798-9 cooling-system operating assumptions."""

    config = {
        "demand_unit": "kWh",
        "system_type": "water",
        "generator_temperature_control": "VARIABLE",
        "distribution_temperature_control": "CONST",
        "distribution_flow_control": "VARIABLE",
        "theta_C_gen_out_set_C": 7.0,
        "theta_C_dis_flw_set_C": 7.0,
        "theta_C_dis_flw_set_min_C": 6.0,
        "theta_C_dis_flw_set_max_C": 18.0,
        "outdoor_compensation_slope": 0.0,
        "outdoor_compensation_offset_K": 7.0,
        "design_deltaT_K": 5.0,
        "design_cooling_load_kW": 8.0,
        "f_wat_C_aux_dis": 0.25,
    }
    if scenario == "bolzano":
        config["design_cooling_load_kW"] = 6.5
    return config


def cooling_storage_system_config(scenario: str) -> dict:
    """Scenario-specific EN 16798-15 chilled-water buffer assumptions."""

    config = {
        "demand_unit": "kWh",
        "storage_type": "STO_TYPE_CW",
        "location": "NC",
        "storage_volume_l": 80.0,
        "storage_temperature_C": 7.0,
        "storage_output_temperature_C": 7.0,
        "storage_return_temperature_C": 12.0,
        "generator_outlet_temperature_C": 7.0,
        "ambient_temperature_C": 20.0,
        "H_C_sto_tot_ls_W_K": 0.8,
        "generator_loop_loss_coefficient_W_K": 0.24,
        "distribution_loop_loss_coefficient_W_K": 0.24,
        "thermal_loss_recoverable_fraction": 0.0,
        "auxiliary_loss_recoverable_fraction": 0.75,
        "auxiliary_to_medium_fraction": 1.0,
        "input_pump_power_kW": 0.020,
        "input_pump_flow_m3_h": 0.90,
        "input_pump_deltaT_K": 5.0,
        "output_pump_power_kW": 0.020,
        "output_pump_flow_m3_h": 0.90,
        "output_pump_deltaT_K": 5.0,
    }
    if scenario == "bolzano":
        config.update(
            {
                "storage_volume_l": 60.0,
                "H_C_sto_tot_ls_W_K": 0.6,
                "generator_loop_loss_coefficient_W_K": 0.18,
                "distribution_loop_loss_coefficient_W_K": 0.18,
                "input_pump_power_kW": 0.018,
                "input_pump_flow_m3_h": 0.75,
                "output_pump_power_kW": 0.018,
                "output_pump_flow_m3_h": 0.75,
            }
        )
    return config


def cooling_generation_system_config(
    scenario: str,
    cooling_map: pd.DataFrame,
    performance_data_method: str = "en14511-14825",
) -> dict:
    """Scenario-specific EN 16798-13 compression-cooling assumptions."""

    nominal_capacity = 6.2 if scenario == "bolzano" else 8.0
    config = {
        "demand_unit": "kWh",
        "cooling_performance_map": cooling_map,
        "nominal_capacity_kW": nominal_capacity,
        "generation_type": "COMP",
        "heat_rejection_type": "AIR_C_COND",
        "theta_C_gen_out_limit_C": 5.0,
        "theta_C_gen_out_set_C": 7.0,
        "free_cooling_enabled": False,
        "performance_includes_heat_rejection_aux": True,
        "control_power_kW": 0.010 if scenario == "athens" else 0.008,
        "additional_auxiliary_power_kW": 0.0,
        "heat_recovery_fraction": 0.0,
        "minimum_part_load_ratio": 0.0,
    }
    if performance_data_method == "en14511-14825":
        config.update(
            {
                "part_load_performance_method": "en14825",
                "part_load_unit_type": "air-to-water",
                "part_load_degradation_coefficient": 0.9,
                "part_load_minimum_capacity_ratio": 0.2105,
            }
        )
    else:
        config["part_load_performance_method"] = "simple"
    return config


def prepare_heat_pump_loads(hourly_sim: pd.DataFrame, dhw_kWh: pd.Series) -> pd.DataFrame:
    """Convert ISO52016 Wh outputs to the heat-pump module kWh inputs."""

    loads = pd.DataFrame(index=hourly_sim.index)
    loads["T_ext"] = hourly_sim["T_ext"].astype(float)
    loads["T_op"] = hourly_sim["T_op"].astype(float)
    if "Q_H" in hourly_sim:
        loads["Q_H_kWh"] = hourly_sim["Q_H"].astype(float) / 1000.0
    else:
        loads["Q_H_kWh"] = hourly_sim["Q_HC"].clip(lower=0).astype(float) / 1000.0

    if "Q_C" in hourly_sim:
        loads["Q_C_kWh"] = hourly_sim["Q_C"].astype(float) / 1000.0
    else:
        loads["Q_C_kWh"] = (-hourly_sim["Q_HC"].clip(upper=0)).astype(float) / 1000.0

    loads["Q_W_kWh"] = dhw_kWh.reindex(loads.index).astype(float)
    return loads


def ensure_heating_and_cooling(loads: pd.DataFrame) -> None:
    heating = float(loads["Q_H_kWh"].sum())
    cooling = float(loads["Q_C_kWh"].sum())
    dhw = float(loads["Q_W_kWh"].sum())
    if heating <= 0 or cooling <= 0:
        raise RuntimeError(
            "The ISO52016 run did not produce both heating and cooling demand. "
            f"Calculated heating={heating:.2f} kWh, cooling={cooling:.2f} kWh. "
            "Use a warmer/cooler EPW file, lower the cooling setpoint, or adapt "
            "the example building gains/glazing."
        )
    if dhw <= 0:
        raise RuntimeError("The DHW calculation returned zero demand; check the DHW inputs.")


def allocate_bin_outputs_to_hours(loads: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Map bin-method results back to hours for visual inspection.

    EN 15316-4-2 is evaluated by temperature bins. This helper does not replace
    the bin calculation; it distributes each bin total back to its member hours
    in proportion to hourly service demand, so users can inspect when the annual
    electricity and backup energy are effectively caused.
    """

    if bins.empty:
        return loads.copy()

    bin_width = float((bins["bin_upper_C"] - bins["bin_lower_C"]).median())
    out = loads.copy()
    out["bin_lower_C"] = np.floor(out["T_ext"].astype(float) / bin_width) * bin_width

    service_specs = {
        "H": "Q_H_kWh",
        "W": "Q_W_kWh",
        "C": "Q_C_kWh",
    }
    service_result_cols = [
        "Q_hp_out_kWh",
        "Q_backup_out_kWh",
        "Q_unmet_kWh",
        "E_hp_in_kWh",
        "E_backup_in_kWh",
        "Q_environment_in_kWh",
        "Q_rejected_kWh",
    ]

    for prefix in service_specs:
        for col in service_result_cols:
            out[f"{prefix}_{col}"] = 0.0
        out[f"{prefix}_performance"] = np.nan
        out[f"{prefix}_capacity_kW"] = np.nan

    for col in ["W_HW_gen_aux_kWh", "W_C_gen_aux_kWh"]:
        out[col] = 0.0

    for _, bin_row in bins.iterrows():
        mask = out["bin_lower_C"].eq(float(bin_row["bin_lower_C"]))
        if not bool(mask.any()):
            continue

        n_hours = int(mask.sum())
        hour_weight = pd.Series(1.0 / n_hours, index=out.index[mask])

        for prefix, load_col in service_specs.items():
            loads_in_bin = out.loc[mask, load_col].clip(lower=0.0)
            total_load = float(loads_in_bin.sum())
            if total_load > 0:
                weights = loads_in_bin / total_load
            else:
                weights = hour_weight

            for col in service_result_cols:
                bin_value = float(bin_row.get(f"{prefix}_{col}", 0.0) or 0.0)
                out.loc[mask, f"{prefix}_{col}"] = weights * bin_value

            out.loc[mask, f"{prefix}_performance"] = bin_row.get(f"{prefix}_performance")
            out.loc[mask, f"{prefix}_capacity_kW"] = bin_row.get(f"{prefix}_capacity_kW")

        out.loc[mask, "W_HW_gen_aux_kWh"] = hour_weight * float(
            bin_row.get("W_HW_gen_aux_kWh", 0.0) or 0.0
        )
        out.loc[mask, "W_C_gen_aux_kWh"] = hour_weight * float(
            bin_row.get("W_C_gen_aux_kWh", 0.0) or 0.0
        )

    out["E_HW_total_kWh"] = (
        out["H_E_hp_in_kWh"]
        + out["W_E_hp_in_kWh"]
        + out["H_E_backup_in_kWh"]
        + out["W_E_backup_in_kWh"]
        + out["W_HW_gen_aux_kWh"]
    )
    out["E_C_total_kWh"] = out["C_E_hp_in_kWh"] + out["C_E_backup_in_kWh"] + out["W_C_gen_aux_kWh"]
    out["E_total_kWh"] = out["E_HW_total_kWh"] + out["E_C_total_kWh"]
    return out


def merge_cooling_generation_to_allocated(
    allocated: pd.DataFrame,
    cooling_generation: pybui.CoolingGenerationSimulationResult | None,
) -> pd.DataFrame:
    """Replace old reversible-HP cooling columns with EN 16798-13 hourly results."""

    if cooling_generation is None:
        return allocated

    out = allocated.copy()
    hourly = cooling_generation.timeseries.reindex(out.index).fillna(0.0)
    out["C_Q_gen_out_kWh"] = hourly["Q_C_gen_in_req_kWh"]
    out["C_Q_hp_out_kWh"] = hourly["Q_C_gen_in_kWh"]
    out["C_Q_backup_out_kWh"] = hourly["Q_C_gen_backup_in_kWh"]
    out["C_Q_unmet_kWh"] = hourly["Q_C_gen_unmet_kWh"]
    out["C_E_hp_in_kWh"] = hourly["E_C_gen_el_in_kWh"]
    out["C_E_backup_in_kWh"] = hourly["E_C_backup_in_kWh"]
    out["C_Q_rejected_kWh"] = hourly["Q_C_gen_out_kWh"]
    out["C_performance"] = hourly["EER_C_gen"].replace(0.0, np.nan)
    out["C_capacity_kW"] = hourly["Q_C_gen_capacity_kW"].replace(0.0, np.nan)
    out["C_hp_runtime_h"] = hourly["t_C_gen_runtime_h"]
    out["W_C_gen_aux_kWh"] = hourly["W_C_aux_gen_kWh"]
    out["E_C_total_kWh"] = hourly["E_C_total_kWh"]
    out["E_total_kWh"] = out["E_HW_total_kWh"] + out["E_C_total_kWh"]
    return out


def combined_generation_summary(
    heat_pump_summary: dict[str, float],
    cooling_generation: pybui.CoolingGenerationSimulationResult | None,
) -> dict[str, float]:
    """Return one summary dictionary for plots that show all generation services."""

    summary = dict(heat_pump_summary)
    if cooling_generation is None:
        return summary

    cooling = cooling_generation.summary
    summary.update(
        {
            "QC_gen_out_kWh": cooling.get("QC_gen_in_req_kWh", 0.0),
            "QC_hp_out_kWh": cooling.get("QC_gen_in_kWh", 0.0),
            "QC_backup_out_kWh": cooling.get("QC_gen_backup_in_kWh", 0.0),
            "QC_unmet_kWh": cooling.get("QC_gen_unmet_kWh", 0.0),
            "EC_hp_in_kWh": cooling.get("EC_gen_el_in_kWh", 0.0),
            "EC_backup_in_kWh": cooling.get("EC_backup_in_kWh", 0.0),
            "EC_gen_in_kWh": cooling.get("EC_gen_el_in_kWh", 0.0)
            + cooling.get("EC_backup_in_kWh", 0.0),
            "WC_gen_aux_kWh": cooling.get("WC_aux_gen_kWh", 0.0),
            "QC_rejected_kWh": cooling.get("QC_gen_out_kWh", 0.0),
            "SEER_C_gen": cooling.get("SEER_C_gen", np.nan),
        }
    )
    summary["E_total_electricity_kWh"] = (
        summary.get("EHW_gen_in_kWh", 0.0)
        + summary.get("WHW_gen_aux_kWh", 0.0)
        + summary.get("EC_gen_in_kWh", 0.0)
        + summary.get("WC_gen_aux_kWh", 0.0)
    )
    return summary


def combined_system_summary(
    heating_generation_summary: dict[str, float],
    cooling_generation: pybui.CoolingGenerationSimulationResult | None,
) -> dict[str, float]:
    """Combine a non-heat-pump heating/DHW generator with optional cooling."""

    summary = dict(heating_generation_summary)
    if cooling_generation is None:
        summary.setdefault("QC_gen_out_kWh", 0.0)
        summary.setdefault("EC_gen_in_kWh", 0.0)
        summary.setdefault("WC_gen_aux_kWh", 0.0)
        summary.setdefault("SEER_C_gen", np.nan)
    else:
        cooling = cooling_generation.summary
        summary.update(
            {
                "QC_gen_out_kWh": cooling.get("QC_gen_in_req_kWh", 0.0),
                "QC_hp_out_kWh": cooling.get("QC_gen_in_kWh", 0.0),
                "QC_backup_out_kWh": cooling.get("QC_gen_backup_in_kWh", 0.0),
                "QC_unmet_kWh": cooling.get("QC_gen_unmet_kWh", 0.0),
                "EC_hp_in_kWh": cooling.get("EC_gen_el_in_kWh", 0.0),
                "EC_backup_in_kWh": cooling.get("EC_backup_in_kWh", 0.0),
                "EC_gen_in_kWh": cooling.get("EC_gen_el_in_kWh", 0.0)
                + cooling.get("EC_backup_in_kWh", 0.0),
                "WC_gen_aux_kWh": cooling.get("WC_aux_gen_kWh", 0.0),
                "QC_rejected_kWh": cooling.get("QC_gen_out_kWh", 0.0),
                "SEER_C_gen": cooling.get("SEER_C_gen", np.nan),
            }
        )
    summary["E_total_electricity_kWh"] = (
        summary.get("E_total_electricity_kWh", 0.0)
        + summary.get("EC_gen_in_kWh", 0.0)
        + summary.get("WC_gen_aux_kWh", 0.0)
    )
    return summary


def renewable_summary_with_electric_load(
    renewable: pybui.RenewableEnergySimulationResult | None,
    electricity_load_kWh: float,
) -> dict[str, float] | None:
    """Recalculate PV self-consumption against final site electricity."""

    if renewable is None:
        return None
    summary = dict(renewable.summary)
    pv_gen = float(summary.get("E_PV_gen_kWh", 0.0))
    pv_cfg = dict(renewable.inputs.get("pv", renewable.inputs.get("photovoltaic", {})) or {})
    self_fraction = float(pv_cfg.get("self_consumption_fraction", 1.0))
    self_fraction = float(np.clip(self_fraction, 0.0, 1.0))
    self_consumed = min(max(float(electricity_load_kWh), 0.0), pv_gen * self_fraction)
    summary["E_PV_self_consumed_kWh"] = self_consumed
    summary["E_PV_export_kWh"] = max(pv_gen - self_consumed, 0.0)
    summary["E_grid_after_PV_kWh"] = max(float(electricity_load_kWh) - self_consumed, 0.0)
    summary["f_PV_self_consumed"] = self_consumed / pv_gen if pv_gen > 0 else np.nan
    return summary


def _write_plot(fig: go.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=80, b=55),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _optional_summary_value(
    summary: dict[str, float] | None,
    key: str,
) -> float | None:
    if not summary or key not in summary:
        return None
    value = _finite_float(summary[key], default=np.nan)
    return value if np.isfinite(value) else None


def _summary_value(
    summary: dict[str, float] | None,
    key: str,
    default: float = 0.0,
) -> float:
    value = _optional_summary_value(summary, key)
    return default if value is None else value


def _sum_column(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return 0.0
    return _finite_float(df[column].sum())


def _max_column(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return 0.0
    return _finite_float(df[column].max())


def _min_active_capacity(
    allocated: pd.DataFrame,
    load_column: str,
    capacity_column: str,
) -> float:
    if load_column not in allocated or capacity_column not in allocated:
        return 0.0
    active = allocated.loc[allocated[load_column].gt(0.0), capacity_column].replace(0.0, np.nan)
    if active.dropna().empty:
        return 0.0
    return _finite_float(active.min())


def _positive_stage_values(stages: list[tuple[str, float | None]]) -> tuple[list[str], list[float]]:
    clean = [
        (label, float(value))
        for label, value in stages
        if value is not None and np.isfinite(value) and value >= 0.0
    ]
    return [label for label, _ in clean], [value for _, value in clean]


def _relative_href(path: Path, base_dir: Path) -> str:
    try:
        relative = path.relative_to(base_dir)
    except ValueError:
        relative = Path(os.path.relpath(path, base_dir))
    return html.escape(relative.as_posix())


PLOT_LINK_LABELS = {
    "00_system_overview.html": "System overview",
    "00_user_overview.html": "System overview",
    "00_workflow_handoff.html": "Workflow handoff",
    "00_sanity_checks.html": "Sanity checks",
    "00_dhw_design_12831_3.html": "EN 12831-3 DHW sizing and supply curve",
    "00_performance_14511_14825.html": "EN 14511 / EN 14825 performance data",
    "01_inputs_timeseries.html": "Input time series",
    "02_emission_15316_2_timeseries.html": "EN 15316-2 emission time series",
    "03_emission_15316_2_monthly.html": "EN 15316-2 monthly emission summary",
    "04_distribution_15316_3_timeseries.html": "EN 15316-3 distribution time series",
    "05_distribution_15316_3_monthly.html": "EN 15316-3 monthly distribution summary",
    "06_storage_15316_5_timeseries.html": "EN 15316-5 storage time series",
    "07_storage_15316_5_monthly.html": "EN 15316-5 monthly storage summary",
    "08_cooling_16798_9_operating_conditions.html": "EN 16798-9 cooling operating conditions",
    "09_cooling_storage_16798_15.html": "EN 16798-15 cooling storage",
    "10_cooling_generation_16798_13_timeseries.html": "EN 16798-13 cooling generation time series",
    "11_cooling_generation_16798_13_bins.html": "EN 16798-13 cooling generation bins",
    "12_renewables_15316_4_3_4_6.html": "EN 15316-4-3/4-6 renewables",
    "13_en15316_4_1_generation.html": "EN 15316-4-1 boiler generation",
    "13_en15316_4_4_generation.html": "EN 15316-4-4 cogeneration",
    "13_en15316_4_5_generation.html": "EN 15316-4-5 district generation",
    "02_heat_pump_timeseries.html": "Heating/DHW heat-pump time series",
    "03_monthly_summary.html": "Monthly demand, electricity, SPF and SEER",
    "04_bin_energy_balance.html": "EN 15316-4-2 bin energy balance",
    "05_bin_performance.html": "COP/EER, capacity and runtime by bin",
    "06_energy_flow_sankey.html": "Annual energy-flow Sankey",
}


def _plot_link_label(path: Path) -> str:
    if path.name in PLOT_LINK_LABELS:
        return PLOT_LINK_LABELS[path.name]
    stem = path.stem
    prefix, _, rest = stem.partition("_")
    if prefix.isdigit() and rest:
        stem = rest
    return stem.replace("_", " ").title()


def plot_user_overview(
    loads: pd.DataFrame,
    allocated: pd.DataFrame,
    summary: dict[str, float],
    output_dir: Path,
    geometry_summary: dict[str, float] | None = None,
) -> Path:
    """Create a first-read page for end users of the example."""

    monthly_loads = loads[["Q_H_kWh", "Q_W_kWh", "Q_C_kWh"]].resample("ME").sum()
    monthly_electricity = allocated[["E_HW_total_kWh", "E_C_total_kWh"]].resample("ME").sum()
    monthly_total_electricity = monthly_electricity.sum(axis=1)

    service_names = ["Heating", "DHW", "Cooling"]
    demand_values = [
        _summary_value(summary, "QH_gen_out_kWh"),
        _summary_value(summary, "QW_gen_out_kWh"),
        _summary_value(summary, "QC_gen_out_kWh"),
    ]
    electricity_values = [
        _sum_column(allocated, "H_E_hp_in_kWh") + _sum_column(allocated, "H_E_backup_in_kWh"),
        _sum_column(allocated, "W_E_hp_in_kWh") + _sum_column(allocated, "W_E_backup_in_kWh"),
        _sum_column(allocated, "C_E_hp_in_kWh") + _sum_column(allocated, "C_E_backup_in_kWh"),
    ]
    auxiliary_electricity = _sum_column(allocated, "W_HW_gen_aux_kWh") + _sum_column(
        allocated, "W_C_gen_aux_kWh"
    )

    floor_area = (
        _summary_value(geometry_summary, "net_floor_area_m2", default=np.nan)
        if geometry_summary
        else np.nan
    )
    if not np.isfinite(floor_area) or floor_area <= 0.0:
        floor_area = np.nan
    intensity_values = [
        demand_values[0] / floor_area if np.isfinite(floor_area) else np.nan,
        demand_values[1] / floor_area if np.isfinite(floor_area) else np.nan,
        demand_values[2] / floor_area if np.isfinite(floor_area) else np.nan,
        _summary_value(summary, "E_total_electricity_kWh") / floor_area
        if np.isfinite(floor_area)
        else np.nan,
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{}, {"type": "domain"}],
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
        subplot_titles=[
            "Annual demand served and electricity used",
            "Final electricity split",
            "Monthly demand pattern and electricity",
            "Seasonal performance",
            "Useful-area intensities",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=service_names,
            y=demand_values,
            name="Thermal demand served",
            marker_color="#8b5a2b",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=service_names,
            y=electricity_values,
            name="Compressor + backup electricity",
            marker_color="#4f6f8f",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Pie(
            labels=[
                "Heating",
                "DHW",
                "Cooling",
                "Auxiliaries",
            ],
            values=electricity_values + [auxiliary_electricity],
            name="Electricity split",
            marker=dict(colors=["#b23b3b", "#e09f3e", "#2f78b7", "#808080"]),
            hole=0.42,
            textinfo="label+percent",
        ),
        row=1,
        col=2,
    )
    for col, name, color in [
        ("Q_H_kWh", "Heating demand", "#b23b3b"),
        ("Q_W_kWh", "DHW demand", "#e09f3e"),
        ("Q_C_kWh", "Cooling demand", "#2f78b7"),
    ]:
        fig.add_trace(
            go.Bar(
                x=monthly_loads.index,
                y=monthly_loads[col],
                name=name,
                marker_color=color,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=monthly_total_electricity.index,
            y=monthly_total_electricity,
            mode="lines+markers",
            name="Total electricity",
            line=dict(color="#1f2933", width=3),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["Heating + DHW SPF", "Cooling SEER"],
            y=[
                _summary_value(summary, "SPF_HW_gen", default=np.nan),
                _summary_value(summary, "SEER_C_gen", default=np.nan),
            ],
            name="Seasonal performance",
            marker_color=["#a83232", "#2f78b7"],
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["Heating", "DHW", "Cooling", "Total electricity"],
            y=intensity_values,
            name="kWh/m2 useful floor area",
            marker_color=["#b23b3b", "#e09f3e", "#2f78b7", "#4f6f8f"],
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.update_yaxes(title_text="kWh/year", row=1, col=1)
    fig.update_yaxes(title_text="kWh/month", row=2, col=1)
    fig.update_yaxes(title_text="ratio", row=3, col=1)
    fig.update_yaxes(title_text="kWh/m2.year", row=3, col=2)
    fig.update_layout(
        title="System Overview: Loads, Electricity and Performance",
        barmode="group",
        height=980,
    )
    return _write_plot(fig, output_dir / "visuals" / "00_user_overview.html")


def plot_workflow_handoff(
    loads: pd.DataFrame,
    allocated: pd.DataFrame,
    summary: dict[str, float],
    output_dir: Path,
    emission_result: pybui.EmissionSimulationResult | None = None,
    distribution_result: pybui.DistributionSimulationResult | None = None,
    storage_result: pybui.StorageSimulationResult | None = None,
    cooling_storage_result: pybui.CoolingStorageSimulationResult | None = None,
    cooling_generation_result: pybui.CoolingGenerationSimulationResult | None = None,
) -> Path:
    """Create an annual handoff plot following the implemented standards chain."""

    emission = emission_result.summary if emission_result is not None else None
    distribution = distribution_result.summary if distribution_result is not None else None
    storage = storage_result.summary if storage_result is not None else None
    cooling_storage = cooling_storage_result.summary if cooling_storage_result is not None else None
    cooling_generation = (
        cooling_generation_result.summary if cooling_generation_result is not None else None
    )
    iso_heating = _optional_summary_value(emission, "QH_em_out_kWh")
    if iso_heating is None:
        iso_heating = _sum_column(loads, "Q_H_kWh")
    iso_cooling = _optional_summary_value(emission, "QC_em_out_kWh")
    if iso_cooling is None:
        iso_cooling = _sum_column(loads, "Q_C_kWh")
    dhw_need = _optional_summary_value(distribution, "QW_dis_out_kWh")
    if dhw_need is None:
        dhw_need = _sum_column(loads, "Q_W_kWh")

    heating_labels, heating_values = _positive_stage_values(
        [
            ("ISO 52016 room load", iso_heating),
            ("EN 15316-2 emission", _optional_summary_value(emission, "QH_em_in_kWh")),
            ("EN 15316-3 distribution", _optional_summary_value(distribution, "QH_dis_in_kWh")),
            ("EN 15316-5 storage", _optional_summary_value(storage, "QH_sto_in_kWh")),
            ("Generator request", _summary_value(summary, "QH_gen_out_kWh")),
        ]
    )
    cooling_labels, cooling_values = _positive_stage_values(
        [
            ("ISO 52016 room load", iso_cooling),
            ("EN 15316-2 emission", _optional_summary_value(emission, "QC_em_in_kWh")),
            ("EN 15316-3 distribution", _optional_summary_value(distribution, "QC_dis_in_kWh")),
            ("EN 16798-15 storage", _optional_summary_value(cooling_storage, "QC_sto_in_kWh")),
            ("Generator request", _summary_value(summary, "QC_gen_out_kWh")),
        ]
    )
    dhw_labels, dhw_values = _positive_stage_values(
        [
            ("EN 12831-3 tap need", dhw_need),
            ("EN 15316-3 distribution", _optional_summary_value(distribution, "QW_dis_in_kWh")),
            ("EN 15316-5 storage", _optional_summary_value(storage, "QW_sto_in_kWh")),
            ("Generator request", _summary_value(summary, "QW_gen_out_kWh")),
        ]
    )

    loss_items = [
        ("Emission heating", _summary_value(emission, "QH_em_ls_kWh")),
        ("Emission cooling", _summary_value(emission, "QC_em_ls_kWh")),
        ("Distribution heating", _summary_value(distribution, "QH_dis_ls_kWh")),
        ("Distribution cooling", _summary_value(distribution, "QC_dis_ls_kWh")),
        ("Distribution DHW", _summary_value(distribution, "QW_dis_ls_kWh")),
        ("Storage heating", _summary_value(storage, "QH_sto_ls_kWh")),
        ("Storage DHW", _summary_value(storage, "QW_sto_ls_kWh")),
        ("Cooling storage gains", _summary_value(cooling_storage, "QC_sto_ls_tot_kWh")),
    ]
    loss_labels = [label for label, value in loss_items if value > 0.0]
    loss_values = [value for _, value in loss_items if value > 0.0]

    electricity_items = [
        ("Heating compressor", _sum_column(allocated, "H_E_hp_in_kWh")),
        ("DHW compressor", _sum_column(allocated, "W_E_hp_in_kWh")),
        ("Cooling compressor", _sum_column(allocated, "C_E_hp_in_kWh")),
        ("Heating backup", _sum_column(allocated, "H_E_backup_in_kWh")),
        ("DHW backup", _sum_column(allocated, "W_E_backup_in_kWh")),
        ("Cooling backup", _sum_column(allocated, "C_E_backup_in_kWh")),
        ("Heating+DHW auxiliaries", _sum_column(allocated, "W_HW_gen_aux_kWh")),
        ("Cooling auxiliaries", _sum_column(allocated, "W_C_gen_aux_kWh")),
    ]
    electricity_labels = [label for label, value in electricity_items if value > 0.0]
    electricity_values = [value for _, value in electricity_items if value > 0.0]

    qhw_request = _summary_value(summary, "QHW_gen_out_kWh")
    qc_request = _summary_value(summary, "QC_gen_out_kWh")
    backup_share = (
        _summary_value(summary, "QHW_backup_out_kWh") / qhw_request if qhw_request > 0.0 else 0.0
    )
    h_unmet_share = (
        _summary_value(summary, "QHW_unmet_kWh") / qhw_request if qhw_request > 0.0 else 0.0
    )
    c_unmet_share = (
        _summary_value(summary, "QC_unmet_kWh") / qc_request if qc_request > 0.0 else 0.0
    )

    fig = make_subplots(
        rows=4,
        cols=1,
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=[
            "Annual load handoff by service",
            "Annual load increases and thermal losses",
            "Final electricity by component",
            "Performance and edge-case fractions",
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=heating_labels,
            y=heating_values,
            mode="lines+markers",
            name="Heating",
            line=dict(color="#b23b3b", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dhw_labels,
            y=dhw_values,
            mode="lines+markers",
            name="DHW",
            line=dict(color="#e09f3e", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cooling_labels,
            y=cooling_values,
            mode="lines+markers",
            name="Cooling",
            line=dict(color="#2f78b7", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=loss_labels, y=loss_values, name="Losses/gains added to loads", marker_color="#8d99ae"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=electricity_labels, y=electricity_values, name="Electricity", marker_color="#4f6f8f"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["Heating + DHW SPF", "Cooling SEER"],
            y=[
                _summary_value(summary, "SPF_HW_gen", default=np.nan),
                _summary_value(summary, "SEER_C_gen", default=np.nan),
            ],
            name="Seasonal performance",
            marker_color=["#b23b3b", "#2f78b7"],
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=["Heating+DHW backup", "Heating+DHW unmet", "Cooling unmet"],
            y=[backup_share, h_unmet_share, c_unmet_share],
            name="Fraction of request",
            marker_color=["#7f1d1d", "#ff9900", "#00a2ff"],
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(title_text="kWh/year", row=1, col=1)
    fig.update_yaxes(title_text="kWh/year", row=2, col=1)
    fig.update_yaxes(title_text="kWh/year", row=3, col=1)
    fig.update_yaxes(title_text="ratio", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="fraction", tickformat=".0%", row=4, col=1, secondary_y=True)
    fig.update_layout(
        title="Workflow Handoff: From Building Loads to Generator Electricity",
        barmode="group",
        height=1220,
    )
    return _write_plot(fig, output_dir / "visuals" / "00_workflow_handoff.html")


def plot_sanity_checks(
    loads: pd.DataFrame,
    allocated: pd.DataFrame,
    summary: dict[str, float],
    output_dir: Path,
    geometry_summary: dict[str, float] | None = None,
) -> Path:
    """Create compact checks for geometry, sizing and exceptional outputs."""

    geometry_summary = geometry_summary or {}
    floor_area = _summary_value(geometry_summary, "net_floor_area_m2", default=np.nan)
    footprint = _summary_value(geometry_summary, "footprint_area_m2", default=np.nan)
    floors = _summary_value(geometry_summary, "n_floors", default=np.nan)
    expected_floor_area = footprint * floors if np.isfinite(footprint) and np.isfinite(floors) else np.nan
    area_consistency = floor_area / expected_floor_area if expected_floor_area > 0 else np.nan

    annual_electricity = _summary_value(summary, "E_total_electricity_kWh", default=np.nan)
    electricity_intensity = (
        annual_electricity / floor_area
        if np.isfinite(annual_electricity) and np.isfinite(floor_area) and floor_area > 0.0
        else np.nan
    )
    annual_load_intensity = (
        (
            _summary_value(summary, "QH_gen_out_kWh")
            + _summary_value(summary, "QW_gen_out_kWh")
            + _summary_value(summary, "QC_gen_out_kWh")
        )
        / floor_area
        if np.isfinite(floor_area) and floor_area > 0.0
        else np.nan
    )

    qhw_request = _summary_value(summary, "QHW_gen_out_kWh")
    qc_request = _summary_value(summary, "QC_gen_out_kWh")
    backup_share = (
        _summary_value(summary, "QHW_backup_out_kWh") / qhw_request if qhw_request > 0.0 else 0.0
    )
    h_unmet_share = (
        _summary_value(summary, "QHW_unmet_kWh") / qhw_request if qhw_request > 0.0 else 0.0
    )
    c_unmet_share = (
        _summary_value(summary, "QC_unmet_kWh") / qc_request if qc_request > 0.0 else 0.0
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            "Geometry areas used by the example",
            "Peak hourly loads versus available active capacity",
            "Compact plausibility indicators",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=["Useful floor area", "Footprint", "Roof", "Ground slab"],
            y=[
                floor_area,
                footprint,
                _summary_value(geometry_summary, "roof_area_m2", default=np.nan),
                _summary_value(geometry_summary, "ground_slab_area_m2", default=np.nan),
            ],
            name="Area",
            marker_color="#5f7f95",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["Heating", "DHW", "Cooling"],
            y=[
                _max_column(loads, "Q_H_kWh"),
                _max_column(loads, "Q_W_kWh"),
                _max_column(loads, "Q_C_kWh"),
            ],
            name="Peak hourly load",
            marker_color="#b07d62",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["Heating", "DHW", "Cooling"],
            y=[
                _min_active_capacity(allocated, "Q_H_kWh", "H_capacity_kW"),
                _min_active_capacity(allocated, "Q_W_kWh", "W_capacity_kW"),
                _min_active_capacity(allocated, "Q_C_kWh", "C_capacity_kW"),
            ],
            name="Minimum active capacity",
            marker_color="#4f6f8f",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[
                "Floor-area check",
                "Load intensity",
                "Electricity intensity",
                "Backup share",
                "Heating+DHW unmet",
                "Cooling unmet",
            ],
            y=[
                area_consistency,
                annual_load_intensity,
                electricity_intensity,
                backup_share,
                h_unmet_share,
                c_unmet_share,
            ],
            name="Indicator",
            marker_color=["#6aa84f", "#8b5a2b", "#4f6f8f", "#7f1d1d", "#ff9900", "#00a2ff"],
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="m2", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_yaxes(title_text="mixed units", row=3, col=1)
    fig.update_layout(
        title="Sanity Checks: Geometry, Sizing and Exceptional Outputs",
        barmode="group",
        height=940,
    )
    return _write_plot(fig, output_dir / "visuals" / "00_sanity_checks.html")


def plot_dhw_design_12831_3(
    dhw_design: pybui.DHWDesignSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = dhw_design.timeseries
    summary = dhw_design.summary
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=[
            "Representative design-day needs and supply curves",
            "Storage residual capacity and switch points",
            "Effective reheating power and minute energy terms",
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=hourly.index,
            y=hourly["Q_W_need_cum_kWh"],
            mode="lines",
            name="Needs curve",
            line=dict(color="#e09f3e", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly.index,
            y=hourly["Q_W_supply_cum_kWh"],
            mode="lines",
            name="Supply curve",
            line=dict(color="#4f6f8f", width=3),
        ),
        row=1,
        col=1,
    )
    if "Q_W_sto_residual_kWh" in hourly:
        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly["Q_W_sto_residual_kWh"],
                mode="lines",
                name="Residual storage capacity",
                line=dict(color="#6aa84f", width=3),
            ),
            row=2,
            col=1,
        )
        for col, label, color, dash in [
            ("Q_W_sto_max_kWh", "Maximum capacity", "#1f2933", "dot"),
            ("Q_W_sto_on_kWh", "Switch-on capacity", "#b23b3b", "dash"),
            ("Q_W_sto_min_kWh", "Minimum allowed capacity", "#ff9900", "dash"),
        ]:
            if col in hourly:
                fig.add_trace(
                    go.Scatter(
                        x=hourly.index,
                        y=hourly[col],
                        mode="lines",
                        name=label,
                        line=dict(color=color, dash=dash),
                    ),
                    row=2,
                    col=1,
                )
    fig.add_trace(
        go.Scatter(
            x=hourly.index,
            y=hourly["phi_eff_kW"],
            mode="lines",
            name="Effective reheat power",
            line=dict(color="#7f1d1d", width=2),
        ),
        row=3,
        col=1,
    )
    for col, label, color in [
        ("Q_W_need_kWh", "DHW need", "#e09f3e"),
        ("Q_W_reheat_effective_kWh", "Effective reheat energy", "#4f6f8f"),
        ("Q_W_storage_loss_kWh", "Storage loss", "#a6a6a6"),
        ("Q_W_distribution_loss_kWh", "Distribution loss", "#808080"),
    ]:
        if col in hourly:
            fig.add_trace(
                go.Bar(
                    x=hourly.index,
                    y=hourly[col],
                    name=label,
                    marker_color=color,
                    opacity=0.65,
                ),
                row=3,
                col=1,
            )
    satisfied = "satisfied" if bool(summary.get("sizing_satisfied", False)) else "not satisfied"
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kW / kWh min", row=3, col=1)
    fig.update_layout(
        title=(
            "EN 12831-3 DHW Design Load: "
            f"{summary.get('V_sto_selected_l', 0.0):,.0f} l storage, "
            f"{summary.get('phi_N_selected_kW', 0.0):.1f} kW generator, "
            f"supply curve {satisfied}"
        ),
        barmode="overlay",
        height=980,
    )
    return _write_plot(fig, output_dir / "visuals" / "00_dhw_design_12831_3.html")


def create_iso52016_visuals(hourly_sim: pd.DataFrame, output_dir: Path, building_area: float) -> Path | None:
    """Create the existing ISO52016 report page in the visual output folder."""

    iso_dir = output_dir / "visuals" / "iso52016"
    iso_dir.mkdir(parents=True, exist_ok=True)
    report_path = iso_dir / "iso52016_building_report.html"
    try:
        pybui.Graphs_and_report(
            df=hourly_sim,
            season="heating_cooling",
            building_area=building_area,
        ).bui_analysis_page(folder_directory=str(iso_dir), name_file="iso52016_building_report")
    except Exception as exc:
        if report_path.exists():
            print(f"ISO52016 visual report could not be regenerated; keeping existing report: {exc}")
            return report_path
        print(f"ISO52016 visual report could not be generated: {exc}")
        return None
    return report_path


def plot_input_timeseries(
    loads: pd.DataFrame,
    output_dir: Path,
    title: str = "Space Emission Loads and DHW Inputs Sent to the Heat Pump",
) -> Path:
    daily_energy = loads[["Q_H_kWh", "Q_C_kWh", "Q_W_kWh"]].resample("D").sum()
    daily_temp = loads[["T_ext", "T_op"]].resample("D").mean()
    cumulative = daily_energy.cumsum()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily useful thermal demand",
            "Daily mean temperatures",
            "Cumulative useful demand",
        ],
    )
    for col, name, color in [
        ("Q_H_kWh", "Space heating", "#b23b3b"),
        ("Q_C_kWh", "Space cooling", "#2f78b7"),
        ("Q_W_kWh", "DHW", "#e09f3e"),
    ]:
        fig.add_trace(go.Bar(x=daily_energy.index, y=daily_energy[col], name=name, marker_color=color), row=1, col=1)
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[col], mode="lines", name=f"Cumulative {name}", line=dict(color=color)), row=3, col=1)

    fig.add_trace(go.Scatter(x=daily_temp.index, y=daily_temp["T_ext"], mode="lines", name="Outdoor temperature", line=dict(color="#44546a")), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily_temp.index, y=daily_temp["T_op"], mode="lines", name="Operative temperature", line=dict(color="#70ad47")), row=2, col=1)
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="degC", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title=title)
    return _write_plot(fig, output_dir / "visuals" / "01_inputs_timeseries.html")


def plot_emission_timeseries(
    emission: pybui.EmissionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = emission.timeseries
    daily = hourly[
        [
            "Q_H_em_out_kWh",
            "Q_H_em_ls_kWh",
            "Q_H_em_in_kWh",
            "Q_C_em_out_kWh",
            "Q_C_em_ls_kWh",
            "Q_C_em_in_kWh",
            "W_H_em_aux_kWh",
            "W_C_em_aux_kWh",
        ]
    ].resample("D").sum()
    daily_temp = hourly[
        [
            "T_H_int_ini_C",
            "theta_H_int_inc_C",
            "T_C_int_ini_C",
            "theta_C_int_inc_C",
        ]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily building needs and emission input loads",
            "Daily emission losses and auxiliary electricity",
            "Baseline and equivalent internal temperatures",
        ],
    )
    for col, name, color in [
        ("Q_H_em_out_kWh", "Heating building need", "#b23b3b"),
        ("Q_H_em_in_kWh", "Heating emission input", "#7f1d1d"),
        ("Q_C_em_out_kWh", "Cooling building need", "#2f78b7"),
        ("Q_C_em_in_kWh", "Cooling emission input", "#1f4e79"),
    ]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name, line=dict(color=color)), row=1, col=1)

    for col, name, color in [
        ("Q_H_em_ls_kWh", "Heating emission losses", "#d65f5f"),
        ("Q_C_em_ls_kWh", "Cooling emission losses", "#6fa8dc"),
        ("W_H_em_aux_kWh", "Heating emission auxiliaries", "#808080"),
        ("W_C_em_aux_kWh", "Cooling emission auxiliaries", "#5b9bd5"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color, dash in [
        ("T_H_int_ini_C", "Heating base internal temperature", "#b23b3b", "dot"),
        ("theta_H_int_inc_C", "Heating equivalent internal temperature", "#7f1d1d", "solid"),
        ("T_C_int_ini_C", "Cooling base internal temperature", "#2f78b7", "dot"),
        ("theta_C_int_inc_C", "Cooling equivalent internal temperature", "#1f4e79", "solid"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=daily_temp.index,
                y=daily_temp[col],
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash),
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 15316-2 Emission System Time Series")
    return _write_plot(fig, output_dir / "visuals" / "02_emission_15316_2_timeseries.html")


def plot_emission_monthly(
    emission: pybui.EmissionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = emission.timeseries
    monthly = hourly[
        [
            "Q_H_em_out_kWh",
            "Q_H_em_temp_effect_kWh",
            "Q_H_emb_ls_kWh",
            "Q_H_em_in_kWh",
            "Q_C_em_out_kWh",
            "Q_C_em_temp_effect_kWh",
            "Q_C_emb_ls_kWh",
            "Q_C_em_in_kWh",
            "W_H_em_aux_kWh",
            "W_C_em_aux_kWh",
        ]
    ].resample("ME").sum()
    monthly["e_H_em_ls"] = monthly["Q_H_em_in_kWh"] / monthly["Q_H_em_out_kWh"].replace(0, np.nan)
    monthly["e_C_em_ls"] = monthly["Q_C_em_in_kWh"] / monthly["Q_C_em_out_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly EN 15316-2 emission balance",
            "Monthly emission expenditure factors",
        ],
    )
    for col, name, color in [
        ("Q_H_em_out_kWh", "Heating building need", "#b23b3b"),
        ("Q_H_em_temp_effect_kWh", "Heating control/emitter effect", "#d65f5f"),
        ("Q_H_emb_ls_kWh", "Heating embedded loss", "#a64242"),
        ("Q_C_em_out_kWh", "Cooling building need", "#2f78b7"),
        ("Q_C_em_temp_effect_kWh", "Cooling control/emitter effect", "#6fa8dc"),
        ("Q_C_emb_ls_kWh", "Cooling embedded loss", "#1f4e79"),
        ("W_H_em_aux_kWh", "Heating auxiliaries", "#808080"),
        ("W_C_em_aux_kWh", "Cooling auxiliaries", "#5b9bd5"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["e_H_em_ls"],
            mode="lines+markers",
            name="Heating expenditure factor",
            line=dict(color="#7f1d1d"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["e_C_em_ls"],
            mode="lines+markers",
            name="Cooling expenditure factor",
            line=dict(color="#2f78b7"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly EN 15316-2 Emission Summary", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "03_emission_15316_2_monthly.html")


def plot_distribution_timeseries(
    distribution: pybui.DistributionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = distribution.timeseries
    daily = hourly[
        [
            "Q_H_dis_out_kWh",
            "Q_H_dis_ls_kWh",
            "Q_H_dis_in_kWh",
            "Q_C_dis_out_kWh",
            "Q_C_dis_ls_kWh",
            "Q_C_dis_in_kWh",
            "Q_W_dis_out_kWh",
            "Q_W_dis_ls_kWh",
            "Q_W_dis_in_kWh",
            "W_H_dis_aux_kWh",
            "W_C_dis_aux_kWh",
            "W_W_dis_aux_kWh",
        ]
    ].resample("D").sum()
    temps = hourly[
        ["theta_H_dis_mean_C", "theta_C_dis_mean_C", "theta_W_dis_mean_C"]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily distribution output and generator-side input loads",
            "Daily distribution thermal losses and pump electricity",
            "Daily mean distribution water temperatures",
        ],
    )
    for col, name, color in [
        ("Q_H_dis_out_kWh", "Heating downstream output", "#b23b3b"),
        ("Q_H_dis_in_kWh", "Heating generator-side input", "#7f1d1d"),
        ("Q_C_dis_out_kWh", "Cooling downstream output", "#2f78b7"),
        ("Q_C_dis_in_kWh", "Cooling generator-side input", "#1f4e79"),
        ("Q_W_dis_out_kWh", "DHW downstream output", "#e09f3e"),
        ("Q_W_dis_in_kWh", "DHW generator-side input", "#9c6500"),
    ]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name, line=dict(color=color)), row=1, col=1)

    for col, name, color in [
        ("Q_H_dis_ls_kWh", "Heating pipe losses", "#d65f5f"),
        ("Q_C_dis_ls_kWh", "Cooling pipe gains", "#6fa8dc"),
        ("Q_W_dis_ls_kWh", "DHW pipe losses", "#f2bf6d"),
        ("W_H_dis_aux_kWh", "Heating pump electricity", "#808080"),
        ("W_C_dis_aux_kWh", "Cooling pump electricity", "#5b9bd5"),
        ("W_W_dis_aux_kWh", "DHW pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color in [
        ("theta_H_dis_mean_C", "Heating water", "#b23b3b"),
        ("theta_C_dis_mean_C", "Cooling water", "#2f78b7"),
        ("theta_W_dis_mean_C", "DHW water", "#e09f3e"),
    ]:
        fig.add_trace(go.Scatter(x=temps.index, y=temps[col], mode="lines", name=name, line=dict(color=color)), row=3, col=1)

    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 15316-3 Distribution System Time Series")
    return _write_plot(fig, output_dir / "visuals" / "04_distribution_15316_3_timeseries.html")


def plot_distribution_monthly(
    distribution: pybui.DistributionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = distribution.timeseries
    monthly = hourly[
        [
            "Q_H_dis_out_kWh",
            "Q_H_dis_ls_kWh",
            "Q_H_dis_in_kWh",
            "Q_C_dis_out_kWh",
            "Q_C_dis_ls_kWh",
            "Q_C_dis_in_kWh",
            "Q_W_dis_out_kWh",
            "Q_W_dis_ls_kWh",
            "Q_W_dis_in_kWh",
            "W_H_dis_aux_kWh",
            "W_C_dis_aux_kWh",
            "W_W_dis_aux_kWh",
        ]
    ].resample("ME").sum()
    monthly["e_H_dis"] = monthly["Q_H_dis_in_kWh"] / monthly["Q_H_dis_out_kWh"].replace(0, np.nan)
    monthly["e_C_dis"] = monthly["Q_C_dis_in_kWh"] / monthly["Q_C_dis_out_kWh"].replace(0, np.nan)
    monthly["e_W_dis"] = monthly["Q_W_dis_in_kWh"] / monthly["Q_W_dis_out_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly EN 15316-3 distribution balance",
            "Monthly distribution expenditure factors",
        ],
    )
    for col, name, color in [
        ("Q_H_dis_out_kWh", "Heating downstream output", "#b23b3b"),
        ("Q_H_dis_ls_kWh", "Heating pipe losses", "#d65f5f"),
        ("Q_C_dis_out_kWh", "Cooling downstream output", "#2f78b7"),
        ("Q_C_dis_ls_kWh", "Cooling pipe gains", "#6fa8dc"),
        ("Q_W_dis_out_kWh", "DHW downstream output", "#e09f3e"),
        ("Q_W_dis_ls_kWh", "DHW pipe losses", "#f2bf6d"),
        ("W_H_dis_aux_kWh", "Heating pump electricity", "#808080"),
        ("W_C_dis_aux_kWh", "Cooling pump electricity", "#5b9bd5"),
        ("W_W_dis_aux_kWh", "DHW pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("e_H_dis", "Heating distribution factor", "#7f1d1d"),
        ("e_C_dis", "Cooling distribution factor", "#2f78b7"),
        ("e_W_dis", "DHW distribution factor", "#9c6500"),
    ]:
        fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly[col], mode="lines+markers", name=name, line=dict(color=color)),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly EN 15316-3 Distribution Summary", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "05_distribution_15316_3_monthly.html")


def plot_storage_timeseries(
    storage: pybui.StorageSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = storage.timeseries
    daily = hourly[
        [
            "Q_H_sto_out_kWh",
            "Q_H_sto_ls_kWh",
            "Q_H_sto_in_kWh",
            "Q_W_sto_out_kWh",
            "Q_W_sto_ls_kWh",
            "Q_W_sto_in_kWh",
            "W_H_sto_aux_kWh",
            "W_W_sto_aux_kWh",
        ]
    ].resample("D").sum()
    temps = hourly[
        ["theta_H_sto_set_C", "theta_W_sto_set_C", "theta_H_sto_amb_C", "theta_W_sto_amb_C"]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily storage output and generator-side input loads",
            "Daily storage standing losses and pump electricity",
            "Daily storage setpoint and ambient temperatures",
        ],
    )
    for col, name, color in [
        ("Q_H_sto_out_kWh", "Heating storage output", "#b23b3b"),
        ("Q_H_sto_in_kWh", "Heating generator-side input", "#7f1d1d"),
        ("Q_W_sto_out_kWh", "DHW storage output", "#e09f3e"),
        ("Q_W_sto_in_kWh", "DHW generator-side input", "#9c6500"),
    ]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name, line=dict(color=color)), row=1, col=1)

    for col, name, color in [
        ("Q_H_sto_ls_kWh", "Heating tank losses", "#d65f5f"),
        ("Q_W_sto_ls_kWh", "DHW tank losses", "#f2bf6d"),
        ("W_H_sto_aux_kWh", "Heating storage pump electricity", "#808080"),
        ("W_W_sto_aux_kWh", "DHW storage pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color in [
        ("theta_H_sto_set_C", "Heating storage setpoint", "#b23b3b"),
        ("theta_W_sto_set_C", "DHW storage setpoint", "#e09f3e"),
        ("theta_H_sto_amb_C", "Heating storage ambient", "#7f7f7f"),
        ("theta_W_sto_amb_C", "DHW storage ambient", "#a6a6a6"),
    ]:
        fig.add_trace(go.Scatter(x=temps.index, y=temps[col], mode="lines", name=name, line=dict(color=color)), row=3, col=1)

    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 15316-5 Storage System Time Series")
    return _write_plot(fig, output_dir / "visuals" / "06_storage_15316_5_timeseries.html")


def plot_storage_monthly(
    storage: pybui.StorageSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = storage.timeseries
    monthly = hourly[
        [
            "Q_H_sto_out_kWh",
            "Q_H_sto_ls_kWh",
            "Q_H_sto_ls_rbl_kWh",
            "Q_H_sto_ls_nrbl_kWh",
            "Q_H_sto_in_kWh",
            "Q_W_sto_out_kWh",
            "Q_W_sto_ls_kWh",
            "Q_W_sto_ls_rbl_kWh",
            "Q_W_sto_ls_nrbl_kWh",
            "Q_W_sto_in_kWh",
            "W_H_sto_aux_kWh",
            "W_W_sto_aux_kWh",
        ]
    ].resample("ME").sum()
    monthly["e_H_sto"] = monthly["Q_H_sto_in_kWh"] / monthly["Q_H_sto_out_kWh"].replace(0, np.nan)
    monthly["e_W_sto"] = monthly["Q_W_sto_in_kWh"] / monthly["Q_W_sto_out_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly EN 15316-5 storage balance",
            "Monthly storage expenditure factors",
        ],
    )
    for col, name, color in [
        ("Q_H_sto_out_kWh", "Heating storage output", "#b23b3b"),
        ("Q_H_sto_ls_rbl_kWh", "Heating recoverable loss", "#d65f5f"),
        ("Q_H_sto_ls_nrbl_kWh", "Heating non-recoverable loss", "#7f1d1d"),
        ("Q_W_sto_out_kWh", "DHW storage output", "#e09f3e"),
        ("Q_W_sto_ls_rbl_kWh", "DHW recoverable loss", "#f2bf6d"),
        ("Q_W_sto_ls_nrbl_kWh", "DHW non-recoverable loss", "#9c6500"),
        ("W_H_sto_aux_kWh", "Heating storage pump electricity", "#808080"),
        ("W_W_sto_aux_kWh", "DHW storage pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("e_H_sto", "Heating storage factor", "#7f1d1d"),
        ("e_W_sto", "DHW storage factor", "#9c6500"),
    ]:
        fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly[col], mode="lines+markers", name=name, line=dict(color=color)),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly EN 15316-5 Storage Summary", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "07_storage_15316_5_monthly.html")


def plot_cooling_system_16798_9(
    cooling_system: pybui.CoolingSystemSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = cooling_system.timeseries
    daily_energy = hourly[["Q_C_dis_out_tot_req_kWh", "Q_C_gen_in_req_kWh"]].resample("D").sum()
    daily_state = hourly[
        [
            "theta_C_gen_out_req_C",
            "theta_C_dis_in_flw_req_C",
            "theta_C_dis_out_ret_req_C",
            "q_V_C_dis_m3_h",
        ]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily cooling request passed through EN 16798-9",
            "Required chilled-water temperatures",
            "Required distribution volume flow",
        ],
    )
    fig.add_trace(
        go.Bar(x=daily_energy.index, y=daily_energy["Q_C_dis_out_tot_req_kWh"], name="Distribution cooling request", marker_color="#2f78b7"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=daily_energy.index, y=daily_energy["Q_C_gen_in_req_kWh"], name="Generator request before detailed losses", marker_color="#6fa8dc"),
        row=1,
        col=1,
    )
    for col, name, color in [
        ("theta_C_gen_out_req_C", "Generator outlet required", "#1f4e79"),
        ("theta_C_dis_in_flw_req_C", "Distribution supply required", "#2f78b7"),
        ("theta_C_dis_out_ret_req_C", "Distribution return required", "#70ad47"),
    ]:
        fig.add_trace(go.Scatter(x=daily_state.index, y=daily_state[col], mode="lines", name=name, line=dict(color=color)), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=daily_state.index, y=daily_state["q_V_C_dis_m3_h"], mode="lines", name="Cooling flow", line=dict(color="#5b9bd5")),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="degC", row=2, col=1)
    fig.update_yaxes(title_text="m3/h", row=3, col=1)
    fig.update_layout(title="EN 16798-9 Cooling System Operating Conditions")
    return _write_plot(fig, output_dir / "visuals" / "08_cooling_16798_9_operating_conditions.html")


def plot_cooling_storage_16798_15(
    storage: pybui.CoolingStorageSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = storage.timeseries
    daily = hourly[
        [
            "Q_C_sto_out_kWh",
            "Q_C_sto_ls_tot_kWh",
            "Q_C_sto_in_kWh",
            "W_C_sto_aux_kWh",
        ]
    ].resample("D").sum()
    temps = hourly[["theta_C_sto_C", "theta_C_sto_amb_C"]].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily chilled-storage energy balance",
            "Daily storage auxiliary electricity",
            "Daily storage and ambient temperatures",
        ],
    )
    for col, name, color in [
        ("Q_C_sto_out_kWh", "Storage output to distribution", "#2f78b7"),
        ("Q_C_sto_ls_tot_kWh", "Storage heat gains", "#6fa8dc"),
        ("Q_C_sto_in_kWh", "Generator cooling required", "#1f4e79"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=1, col=1)
    fig.add_trace(
        go.Bar(x=daily.index, y=daily["W_C_sto_aux_kWh"], name="Storage pumps", marker_color="#5b9bd5"),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=temps.index, y=temps["theta_C_sto_C"], mode="lines", name="Storage water", line=dict(color="#2f78b7")), row=3, col=1)
    fig.add_trace(go.Scatter(x=temps.index, y=temps["theta_C_sto_amb_C"], mode="lines", name="Storage ambient", line=dict(color="#44546a")), row=3, col=1)
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 16798-15 Cooling Storage")
    return _write_plot(fig, output_dir / "visuals" / "09_cooling_storage_16798_15.html")


def plot_cooling_generation_16798_13(
    generation: pybui.CoolingGenerationSimulationResult,
    output_dir: Path,
) -> list[Path]:
    hourly = generation.timeseries
    daily = hourly[
        [
            "Q_C_gen_in_req_kWh",
            "Q_C_gen_in_kWh",
            "Q_C_gen_unmet_kWh",
            "E_C_gen_el_in_kWh",
            "E_C_backup_in_kWh",
            "W_C_aux_gen_kWh",
        ]
    ].resample("D").sum()
    perf_cols = ["EER_C_gen", "f_C_PL", "theta_C_gen_out_C", "theta_cond_in_C"]
    for optional_col in ["EER_C_gen_full_load", "f_C_PLF"]:
        if optional_col in hourly:
            perf_cols.append(optional_col)
    perf = hourly[perf_cols].resample("D").mean()

    fig_ts = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily EN 16798-13 cooling generation balance",
            "Daily generation electricity and auxiliaries",
            "Daily EER, part-load and operating temperatures",
        ],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    for col, name, color in [
        ("Q_C_gen_in_req_kWh", "Cooling required", "#2f78b7"),
        ("Q_C_gen_in_kWh", "Cooling removed", "#1f4e79"),
        ("Q_C_gen_unmet_kWh", "Unmet cooling", "#00a2ff"),
    ]:
        fig_ts.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=1, col=1)
    for col, name, color in [
        ("E_C_gen_el_in_kWh", "Compressor electricity", "#2f78b7"),
        ("E_C_backup_in_kWh", "Cooling backup electricity", "#1f4e79"),
        ("W_C_aux_gen_kWh", "Generator auxiliaries", "#5b9bd5"),
    ]:
        fig_ts.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)
    fig_ts.add_trace(go.Scatter(x=perf.index, y=perf["EER_C_gen"], mode="lines", name="EER", line=dict(color="#2f78b7")), row=3, col=1, secondary_y=False)
    if "EER_C_gen_full_load" in perf:
        fig_ts.add_trace(go.Scatter(x=perf.index, y=perf["EER_C_gen_full_load"], mode="lines", name="Full-load EER", line=dict(color="#2f78b7", dash="dot")), row=3, col=1, secondary_y=False)
    fig_ts.add_trace(go.Scatter(x=perf.index, y=perf["f_C_PL"], mode="lines", name="Part-load ratio", line=dict(color="#70ad47")), row=3, col=1, secondary_y=False)
    if "f_C_PLF" in perf:
        fig_ts.add_trace(go.Scatter(x=perf.index, y=perf["f_C_PLF"], mode="lines", name="Part-load factor", line=dict(color="#44546a", dash="dash")), row=3, col=1, secondary_y=False)
    fig_ts.add_trace(go.Scatter(x=perf.index, y=perf["theta_C_gen_out_C"], mode="lines", name="Evaporator outlet", line=dict(color="#1f4e79", dash="dot")), row=3, col=1, secondary_y=True)
    fig_ts.add_trace(go.Scatter(x=perf.index, y=perf["theta_cond_in_C"], mode="lines", name="Condenser inlet", line=dict(color="#44546a", dash="dash")), row=3, col=1, secondary_y=True)
    fig_ts.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig_ts.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig_ts.update_yaxes(title_text="ratio", row=3, col=1, secondary_y=False)
    fig_ts.update_yaxes(title_text="degC", row=3, col=1, secondary_y=True)
    fig_ts.update_layout(title="EN 16798-13 Cooling Generation Time Series", barmode="group")
    ts_path = _write_plot(fig_ts, output_dir / "visuals" / "10_cooling_generation_16798_13_timeseries.html")

    bins = hourly.copy()
    bins["bin_center_C"] = np.floor(bins["T_ext_C"].astype(float)) + 0.5
    grouped = bins.groupby("bin_center_C").agg(
        hours=("hours", "sum"),
        q_req=("Q_C_gen_in_req_kWh", "sum"),
        q_supplied=("Q_C_gen_in_kWh", "sum"),
        e_el=("E_C_gen_el_in_kWh", "sum"),
        w_aux=("W_C_aux_gen_kWh", "sum"),
        eer=("EER_C_gen", "mean"),
        plr=("f_C_PL", "mean"),
        plf=("f_C_PLF", "mean") if "f_C_PLF" in bins else ("f_C_PL", "mean"),
        eer_full=("EER_C_gen_full_load", "mean") if "EER_C_gen_full_load" in bins else ("EER_C_gen", "mean"),
        capacity=("Q_C_gen_capacity_kW", "mean"),
    ).reset_index()
    fig_bin = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Cooling removed and electricity by outdoor-temperature bin",
            "Direct EER and part-load ratio by bin",
            "Available cooling capacity and bin hours",
        ],
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )
    fig_bin.add_trace(go.Bar(x=grouped["bin_center_C"], y=grouped["q_supplied"], name="Cooling removed", marker_color="#2f78b7"), row=1, col=1)
    fig_bin.add_trace(go.Bar(x=grouped["bin_center_C"], y=grouped["e_el"] + grouped["w_aux"], name="Cooling electricity + auxiliaries", marker_color="#5b9bd5"), row=1, col=1)
    fig_bin.add_trace(go.Scatter(x=grouped["bin_center_C"], y=grouped["eer"], mode="lines+markers", name="EER", line=dict(color="#2f78b7")), row=2, col=1, secondary_y=False)
    fig_bin.add_trace(go.Scatter(x=grouped["bin_center_C"], y=grouped["eer_full"], mode="lines+markers", name="Full-load EER", line=dict(color="#2f78b7", dash="dot")), row=2, col=1, secondary_y=False)
    fig_bin.add_trace(go.Scatter(x=grouped["bin_center_C"], y=grouped["plr"], mode="lines+markers", name="Part-load ratio", line=dict(color="#70ad47")), row=2, col=1, secondary_y=True)
    fig_bin.add_trace(go.Scatter(x=grouped["bin_center_C"], y=grouped["plf"], mode="lines+markers", name="Part-load factor", line=dict(color="#44546a", dash="dash")), row=2, col=1, secondary_y=True)
    fig_bin.add_trace(go.Scatter(x=grouped["bin_center_C"], y=grouped["capacity"], mode="lines+markers", name="Capacity", line=dict(color="#1f4e79")), row=3, col=1, secondary_y=False)
    fig_bin.add_trace(go.Bar(x=grouped["bin_center_C"], y=grouped["hours"], name="Hours", marker_color="#b7b7b7", opacity=0.55), row=3, col=1, secondary_y=True)
    fig_bin.update_xaxes(title_text="Outdoor-temperature bin center [degC]", row=3, col=1)
    fig_bin.update_yaxes(title_text="kWh", row=1, col=1)
    fig_bin.update_yaxes(title_text="EER", row=2, col=1, secondary_y=False)
    fig_bin.update_yaxes(title_text="part-load", row=2, col=1, secondary_y=True)
    fig_bin.update_yaxes(title_text="kW", row=3, col=1, secondary_y=False)
    fig_bin.update_yaxes(title_text="hours", row=3, col=1, secondary_y=True)
    fig_bin.update_layout(title="EN 16798-13 Cooling Generation Bin Performance", barmode="group")
    bin_path = _write_plot(fig_bin, output_dir / "visuals" / "11_cooling_generation_16798_13_bins.html")
    return [ts_path, bin_path]


def plot_renewables_15316_4_3_4_6(
    renewable: pybui.RenewableEnergySimulationResult,
    output_dir: Path,
) -> Path:
    hourly = renewable.timeseries
    daily = hourly[
        [
            "Q_H_solar_thermal_used_kWh",
            "Q_W_solar_thermal_used_kWh",
            "Q_solar_thermal_unused_kWh",
            "E_PV_gen_kWh",
            "E_PV_self_consumed_kWh",
            "E_PV_export_kWh",
        ]
    ].resample("D").sum()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=[
            "Daily solar thermal contribution",
            "Daily PV generation and allocation",
            "Cumulative renewable contribution",
        ],
    )
    for col, name, color in [
        ("Q_W_solar_thermal_used_kWh", "Solar thermal to DHW", "#e09f3e"),
        ("Q_H_solar_thermal_used_kWh", "Solar thermal to heating", "#b23b3b"),
        ("Q_solar_thermal_unused_kWh", "Unused solar thermal", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=1, col=1)
    for col, name, color in [
        ("E_PV_self_consumed_kWh", "PV self-consumed", "#70ad47"),
        ("E_PV_export_kWh", "PV exported", "#9bbb59"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)
    cumulative = daily.cumsum()
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative["Q_H_solar_thermal_used_kWh"]
            + cumulative["Q_W_solar_thermal_used_kWh"],
            mode="lines",
            name="Cumulative solar thermal used",
            line=dict(color="#e09f3e"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative["E_PV_gen_kWh"],
            mode="lines",
            name="Cumulative PV generation",
            line=dict(color="#70ad47"),
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title="EN 15316-4-3 / EN 15316-4-6 Renewable Systems", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "12_renewables_15316_4_3_4_6.html")


def plot_heating_generation_system(
    generation,
    output_dir: Path,
    heating_generation_method: str,
) -> Path:
    hourly = generation.timeseries
    daily_cols = [
        col
        for col in [
            "Q_H_gen_out_kWh",
            "Q_W_gen_out_kWh",
            "EHW_gen_in_kWh",
            "W_HW_gen_aux_kWh",
            "Q_HW_gen_loss_kWh",
            "E_chp_el_generated_kWh",
            "E_chp_el_self_consumed_kWh",
            "E_chp_el_export_kWh",
        ]
        if col in hourly
    ]
    daily = hourly[daily_cols].resample("D").sum()
    perf_col = "eta_HW_gen" if "eta_HW_gen" in hourly else None
    runtime_col = "t_chp_runtime_h" if "t_chp_runtime_h" in hourly else "t_HW_gen_runtime_h"
    if runtime_col not in hourly:
        runtime_col = None
    perf = pd.DataFrame(index=daily.index)
    if perf_col is not None:
        perf["eta_HW_gen"] = hourly[perf_col].resample("D").mean()
    if runtime_col is not None:
        perf["runtime_h"] = hourly[runtime_col].resample("D").sum()

    title_by_method = {
        "en15316-4-1": "EN 15316-4-1 Combustion Boiler Generation",
        "en15316-4-4": "EN 15316-4-4 Cogeneration",
        "en15316-4-5": "EN 15316-4-5 District Heating",
    }
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=[
            "Daily useful heat supplied",
            "Daily input energy, auxiliaries and losses",
            "Daily efficiency and runtime",
        ],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    for col, name, color in [
        ("Q_H_gen_out_kWh", "Space heating supplied", "#b23b3b"),
        ("Q_W_gen_out_kWh", "DHW supplied", "#e09f3e"),
    ]:
        if col in daily:
            fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=1, col=1)
    for col, name, color in [
        ("EHW_gen_in_kWh", "Heating/DHW generator input", "#595959"),
        ("W_HW_gen_aux_kWh", "Auxiliary electricity", "#808080"),
        ("Q_HW_gen_loss_kWh", "Generator losses", "#a6a6a6"),
        ("E_chp_el_generated_kWh", "CHP electricity generated", "#70ad47"),
        ("E_chp_el_self_consumed_kWh", "CHP electricity self-consumed", "#548235"),
        ("E_chp_el_export_kWh", "CHP electricity exported", "#9bbb59"),
    ]:
        if col in daily:
            fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)
    if "eta_HW_gen" in perf:
        fig.add_trace(
            go.Scatter(
                x=perf.index,
                y=perf["eta_HW_gen"],
                mode="lines",
                name="Generation efficiency",
                line=dict(color="#1f4e79"),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    if "runtime_h" in perf:
        fig.add_trace(
            go.Bar(
                x=perf.index,
                y=perf["runtime_h"],
                name="Runtime",
                marker_color="#b7b7b7",
                opacity=0.65,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="efficiency", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="h/day", row=3, col=1, secondary_y=True)
    fig.update_layout(
        title=title_by_method.get(heating_generation_method, "Heating/DHW Generation"),
        barmode="group",
        height=980,
    )
    filename = f"13_{heating_generation_method.replace('-', '_')}_generation.html"
    return _write_plot(fig, output_dir / "visuals" / filename)


def plot_generic_system_overview(
    loads: pd.DataFrame,
    summary: dict[str, float],
    output_dir: Path,
    heating_generation_method: str,
    geometry_summary: dict[str, float] | None = None,
    renewable_summary: dict[str, float] | None = None,
) -> Path:
    monthly_loads = loads[["Q_H_kWh", "Q_W_kWh", "Q_C_kWh"]].resample("ME").sum()
    monthly_generation_input = pd.Series(
        summary.get("EHW_gen_in_kWh", 0.0) / max(len(monthly_loads), 1),
        index=monthly_loads.index,
    )
    floor_area = (
        _summary_value(geometry_summary, "net_floor_area_m2", default=np.nan)
        if geometry_summary
        else np.nan
    )
    final_grid = (
        renewable_summary.get("E_grid_after_PV_kWh", np.nan)
        if renewable_summary
        else np.nan
    )
    indicators = {
        "Heating demand": summary.get("QH_gen_out_kWh", 0.0),
        "DHW demand": summary.get("QW_gen_out_kWh", 0.0),
        "Heating/DHW input": summary.get("EHW_gen_in_kWh", 0.0),
        "Cooling electricity": summary.get("EC_gen_in_kWh", 0.0)
        + summary.get("WC_gen_aux_kWh", 0.0),
        "PV generation": renewable_summary.get("E_PV_gen_kWh", 0.0)
        if renewable_summary
        else 0.0,
        "Grid electricity after PV": final_grid if np.isfinite(final_grid) else 0.0,
    }
    intensities = [
        value / floor_area if np.isfinite(floor_area) and floor_area > 0 else np.nan
        for value in indicators.values()
    ]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly thermal requests after upstream standards",
            "Annual generation and renewable balance",
            "Useful-area intensities",
        ],
    )
    for col, name, color in [
        ("Q_H_kWh", "Heating request", "#b23b3b"),
        ("Q_W_kWh", "DHW request", "#e09f3e"),
        ("Q_C_kWh", "Cooling request", "#2f78b7"),
    ]:
        fig.add_trace(go.Bar(x=monthly_loads.index, y=monthly_loads[col], name=name, marker_color=color), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=monthly_generation_input.index,
            y=monthly_generation_input,
            mode="lines",
            name="Mean monthly heating/DHW input",
            line=dict(color="#595959"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(indicators.keys()),
            y=list(indicators.values()),
            name="Annual energy",
            marker_color=["#b23b3b", "#e09f3e", "#595959", "#2f78b7", "#70ad47", "#4f6f8f"],
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(indicators.keys()),
            y=intensities,
            name="kWh/m2",
            marker_color=["#b23b3b", "#e09f3e", "#595959", "#2f78b7", "#70ad47", "#4f6f8f"],
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="kWh/year", row=2, col=1)
    fig.update_yaxes(title_text="kWh/m2.year", row=3, col=1)
    fig.update_layout(
        title=f"System Overview: {heating_generation_method} Heating/DHW Generation",
        height=980,
    )
    return _write_plot(fig, output_dir / "visuals" / "00_system_overview.html")


def plot_performance_data_14511_14825(
    performance_data: pybui.HeatPumpPerformanceDataResult,
    output_dir: Path,
) -> Path:
    rating = performance_data.rating_points
    heating = rating[rating["mode"] == "heating"]
    cooling = rating[rating["mode"] == "cooling"]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            "EN 14511 declared heating COP and EN 14825 COPbin",
            "EN 14511 declared cooling EER and EN 14825 EERbin",
            "Declared capacity at rating points",
            "EN 14825 capacity ratio and part-load factor",
        ],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
    )
    for sink in sorted(heating["sink_temperature_C"].unique()):
        data = heating[heating["sink_temperature_C"].eq(sink)].sort_values(
            "source_temperature_C"
        )
        fig.add_trace(
            go.Scatter(
                x=data["source_temperature_C"],
                y=data["cop"],
                mode="lines+markers",
                name=f"COPd W{sink:g}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["source_temperature_C"],
                y=data["performance_at_part_load"],
                mode="lines+markers",
                name=f"COPbin W{sink:g}",
                line=dict(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["source_temperature_C"],
                y=data["capacity_kW"],
                mode="lines+markers",
                name=f"Heating capacity W{sink:g}",
            ),
            row=3,
            col=1,
        )

    for sink in sorted(cooling["sink_temperature_C"].unique()):
        data = cooling[cooling["sink_temperature_C"].eq(sink)].sort_values(
            "source_temperature_C"
        )
        fig.add_trace(
            go.Scatter(
                x=data["source_temperature_C"],
                y=data["eer"],
                mode="lines+markers",
                name=f"EERd W{sink:g}",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["source_temperature_C"],
                y=data["performance_at_part_load"],
                mode="lines+markers",
                name=f"EERbin W{sink:g}",
                line=dict(dash="dot"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["source_temperature_C"],
                y=data["capacity_kW"],
                mode="lines+markers",
                name=f"Cooling capacity W{sink:g}",
                line=dict(dash="dash"),
            ),
            row=3,
            col=1,
        )

    part_load = rating.sort_values(["mode", "source_temperature_C", "sink_temperature_C"])
    fig.add_trace(
        go.Scatter(
            x=part_load["source_temperature_C"],
            y=part_load["capacity_ratio"],
            mode="markers",
            name="Capacity ratio CR",
            marker=dict(color="#70ad47"),
            text=part_load["rating_condition"],
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=part_load["source_temperature_C"],
            y=part_load["part_load_factor"],
            mode="markers",
            name="Part-load factor",
            marker=dict(color="#44546a", symbol="diamond"),
            text=part_load["rating_condition"],
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Source / outdoor temperature [degC]", row=1, col=1)
    fig.update_xaxes(title_text="Source / outdoor temperature [degC]", row=2, col=1)
    fig.update_xaxes(title_text="Source / outdoor temperature [degC]", row=3, col=1)
    fig.update_xaxes(title_text="Source / outdoor temperature [degC]", row=4, col=1)
    fig.update_yaxes(title_text="COP [-]", row=1, col=1)
    fig.update_yaxes(title_text="EER [-]", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    fig.update_yaxes(title_text="CR [-]", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="factor [-]", row=4, col=1, secondary_y=True)
    fig.update_layout(
        title="EN 14511 Rating Data and EN 14825 Part-Load Inspection",
        height=1120,
    )
    return _write_plot(fig, output_dir / "visuals" / "00_performance_14511_14825.html")


def plot_heat_pump_hourly(allocated: pd.DataFrame, output_dir: Path) -> Path:
    daily = allocated[
        [
            "E_HW_total_kWh",
            "E_C_total_kWh",
            "H_E_hp_in_kWh",
            "W_E_hp_in_kWh",
            "C_E_hp_in_kWh",
            "H_E_backup_in_kWh",
            "W_E_backup_in_kWh",
            "W_HW_gen_aux_kWh",
            "W_C_gen_aux_kWh",
        ]
    ].resample("D").sum()

    perf = allocated[["H_performance", "W_performance", "C_performance"]].resample("D").mean()
    cumulative = daily[["E_HW_total_kWh", "E_C_total_kWh"]].cumsum()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily heat-pump electricity by service",
            "Daily mean bin COP/EER assigned to hours",
            "Cumulative final electricity",
        ],
    )
    bars = [
        ("H_E_hp_in_kWh", "Heating compressor", "#b23b3b"),
        ("W_E_hp_in_kWh", "DHW compressor", "#e09f3e"),
        ("C_E_hp_in_kWh", "Cooling compressor", "#2f78b7"),
        ("H_E_backup_in_kWh", "Heating backup", "#7f1d1d"),
        ("W_E_backup_in_kWh", "DHW backup", "#9c6500"),
        ("W_HW_gen_aux_kWh", "Heating+DHW auxiliaries", "#808080"),
        ("W_C_gen_aux_kWh", "Cooling auxiliaries", "#5b9bd5"),
    ]
    for col, name, color in bars:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("H_performance", "Heating COP", "#b23b3b"),
        ("W_performance", "DHW COP", "#e09f3e"),
        ("C_performance", "Cooling EER", "#2f78b7"),
    ]:
        fig.add_trace(go.Scatter(x=perf.index, y=perf[col], mode="lines", name=name, line=dict(color=color)), row=2, col=1)

    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative["E_HW_total_kWh"], mode="lines", name="Cumulative heating+DHW electricity", line=dict(color="#7f1d1d")), row=3, col=1)
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative["E_C_total_kWh"], mode="lines", name="Cumulative cooling electricity", line=dict(color="#2f78b7")), row=3, col=1)
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="COP / EER", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title="Allocated Hourly View of Heat-Pump Outputs")
    return _write_plot(fig, output_dir / "visuals" / "02_heat_pump_timeseries.html")


def plot_monthly_summary(loads: pd.DataFrame, allocated: pd.DataFrame, output_dir: Path) -> Path:
    monthly_loads = loads[["Q_H_kWh", "Q_C_kWh", "Q_W_kWh"]].resample("ME").sum()
    monthly_energy = allocated[["E_HW_total_kWh", "E_C_total_kWh"]].resample("ME").sum()
    monthly = pd.concat([monthly_loads, monthly_energy], axis=1)
    monthly["SPF_HW_month"] = (monthly["Q_H_kWh"] + monthly["Q_W_kWh"]) / monthly["E_HW_total_kWh"].replace(0, np.nan)
    monthly["SEER_C_month"] = monthly["Q_C_kWh"] / monthly["E_C_total_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["Monthly useful demand and final electricity", "Monthly SPF / SEER"],
    )
    for col, name, color in [
        ("Q_H_kWh", "Heating demand", "#b23b3b"),
        ("Q_W_kWh", "DHW demand", "#e09f3e"),
        ("Q_C_kWh", "Cooling demand", "#2f78b7"),
        ("E_HW_total_kWh", "Heating+DHW electricity", "#595959"),
        ("E_C_total_kWh", "Cooling electricity", "#5b9bd5"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["SPF_HW_month"], mode="lines+markers", name="SPF heating+DHW", line=dict(color="#7f1d1d")), row=2, col=1)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["SEER_C_month"], mode="lines+markers", name="SEER cooling", line=dict(color="#2f78b7")), row=2, col=1)
    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly Aggregates")
    return _write_plot(fig, output_dir / "visuals" / "03_monthly_summary.html")


def plot_bin_balance(bins: pd.DataFrame, output_dir: Path) -> Path:
    x = bins["bin_center_C"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Thermal demand/output by outdoor-temperature bin",
            "Input electricity by bin",
            "Auxiliaries, losses and unmet demand",
        ],
    )
    for col, name, color in [
        ("H_Q_gen_out_kWh", "Heating demand", "#b23b3b"),
        ("W_Q_gen_out_kWh", "DHW demand", "#e09f3e"),
        ("C_Q_gen_out_kWh", "Cooling demand", "#2f78b7"),
        ("H_Q_hp_out_kWh", "Heating from HP", "#d65f5f"),
        ("W_Q_hp_out_kWh", "DHW from HP", "#f2bf6d"),
        ("C_Q_hp_out_kWh", "Cooling from HP", "#6fa8dc"),
    ]:
        if col in bins:
            fig.add_trace(go.Bar(x=x, y=bins[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("H_E_hp_in_kWh", "Heating compressor", "#b23b3b"),
        ("W_E_hp_in_kWh", "DHW compressor", "#e09f3e"),
        ("C_E_hp_in_kWh", "Cooling compressor", "#2f78b7"),
        ("H_E_backup_in_kWh", "Heating backup", "#7f1d1d"),
        ("W_E_backup_in_kWh", "DHW backup", "#9c6500"),
    ]:
        if col in bins:
            fig.add_trace(go.Bar(x=x, y=bins[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color in [
        ("W_HW_gen_aux_kWh", "Heating+DHW auxiliaries", "#808080"),
        ("W_C_gen_aux_kWh", "Cooling auxiliaries", "#5b9bd5"),
        ("Q_HW_gen_ls_tot_kWh", "Heating+DHW gen losses", "#a6a6a6"),
        ("H_Q_unmet_kWh", "Heating unmet", "#ff0000"),
        ("W_Q_unmet_kWh", "DHW unmet", "#ff9900"),
        ("C_Q_unmet_kWh", "Cooling unmet", "#00a2ff"),
    ]:
        if col in bins:
            fig.add_trace(go.Bar(x=x, y=bins[col], name=name, marker_color=color), row=3, col=1)

    fig.update_xaxes(title_text="Outdoor-temperature bin center [degC]", row=3, col=1)
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title="Heat-Pump Bin Energy Balance", barmode="group")
    return _write_plot(fig, output_dir / "visuals" / "04_bin_energy_balance.html")


def plot_bin_performance(bins: pd.DataFrame, output_dir: Path) -> Path:
    x = bins["bin_center_C"]
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Heating and DHW COP by bin",
            "Cooling EER by bin",
            "EN 14825 part-load ratio and correction factor",
            "Capacity by bin",
            "Runtime and effective hours",
        ],
        specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{}]],
    )
    for prefix, label, color in [
        ("H", "Heating", "#b23b3b"),
        ("W", "DHW", "#e09f3e"),
        ("C", "Cooling", "#2f78b7"),
    ]:
        performance_row = 2 if prefix == "C" else 1
        performance_name = f"{label} EER" if prefix == "C" else f"{label} COP"
        if f"{prefix}_performance" in bins:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bins[f"{prefix}_performance"],
                    mode="lines+markers",
                    name=performance_name,
                    line=dict(color=color),
                ),
                row=performance_row,
                col=1,
            )
        if f"{prefix}_performance_full_load" in bins:
            full_name = (
                f"{label} full-load EER"
                if prefix == "C"
                else f"{label} full-load COP"
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bins[f"{prefix}_performance_full_load"],
                    mode="lines",
                    name=full_name,
                    line=dict(color=color, dash="dot"),
                ),
                row=performance_row,
                col=1,
            )
        if f"{prefix}_part_load_ratio" in bins:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bins[f"{prefix}_part_load_ratio"],
                    mode="lines+markers",
                    name=f"{label} CR",
                    line=dict(color=color),
                ),
                row=3,
                col=1,
                secondary_y=False,
            )
        if f"{prefix}_part_load_factor" in bins:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bins[f"{prefix}_part_load_factor"],
                    mode="lines",
                    name=f"{label} part-load factor",
                    line=dict(color=color, dash="dash"),
                ),
                row=3,
                col=1,
                secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bins[f"{prefix}_capacity_kW"],
                mode="lines+markers",
                name=f"{label} capacity",
                line=dict(color=color),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=bins[f"{prefix}_hp_runtime_h"],
                name=f"{label} HP runtime",
                marker_color=color,
            ),
            row=5,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=bins["effective_hours"],
            mode="lines",
            name="Effective bin hours",
            line=dict(color="#404040", dash="dash"),
        ),
        row=5,
        col=1,
    )
    fig.update_xaxes(title_text="Outdoor-temperature bin center [degC]", row=5, col=1)
    fig.update_yaxes(title_text="COP [-]", row=1, col=1)
    fig.update_yaxes(title_text="EER [-]", row=2, col=1)
    fig.update_yaxes(title_text="CR [-]", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="factor [-]", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="kW", row=4, col=1)
    fig.update_yaxes(title_text="hours", row=5, col=1)
    fig.update_layout(
        title="Heat-Pump Product Map and Runtime Inspection",
        height=1180,
    )
    return _write_plot(fig, output_dir / "visuals" / "05_bin_performance.html")


def plot_summary_sankey(summary: dict[str, float], output_dir: Path) -> Path:
    labels = [
        "Ambient/source heat",
        "HP electricity",
        "Backup electricity",
        "Aux electricity",
        "Heat pump",
        "Backup heater",
        "Useful heating+DHW",
        "Generator/aux losses",
        "Useful cooling extracted",
        "Cooling electricity",
        "Rejected heat",
    ]
    index = {label: i for i, label in enumerate(labels)}
    flows: list[tuple[str, str, float]] = [
        ("Ambient/source heat", "Heat pump", summary.get("QHW_environment_in_kWh", 0.0)),
        ("HP electricity", "Heat pump", summary.get("EH_hp_in_kWh", 0.0) + summary.get("EW_hp_in_kWh", 0.0)),
        ("Backup electricity", "Backup heater", summary.get("EHW_backup_in_kWh", 0.0)),
        ("Heat pump", "Useful heating+DHW", summary.get("QHW_hp_out_kWh", 0.0)),
        ("Backup heater", "Useful heating+DHW", summary.get("QHW_backup_out_kWh", 0.0)),
        ("Aux electricity", "Generator/aux losses", summary.get("WHW_gen_aux_kWh", 0.0)),
        ("Useful cooling extracted", "Rejected heat", summary.get("QC_hp_out_kWh", 0.0)),
        ("Cooling electricity", "Rejected heat", summary.get("EC_hp_in_kWh", 0.0) + summary.get("WC_gen_aux_kWh", 0.0)),
    ]
    source = []
    target = []
    value = []
    for src, dst, val in flows:
        if val and np.isfinite(val) and val > 0:
            source.append(index[src])
            target.append(index[dst])
            value.append(float(val))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=16, thickness=18),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(title="Heat-Pump Annual Energy Flow Summary")
    return _write_plot(fig, output_dir / "visuals" / "06_energy_flow_sankey.html")


def create_inspection_index(
    output_dir: Path,
    summary: dict[str, float],
    plot_paths: list[Path],
    iso_report: Path | None,
    geometry_summary: dict[str, float] | None = None,
    dhw_design_summary: dict[str, float | str | bool] | None = None,
    performance_data_summary: dict[str, float] | None = None,
    emission_summary: dict[str, float] | None = None,
    distribution_summary: dict[str, float] | None = None,
    storage_summary: dict[str, float] | None = None,
    cooling_system_summary: dict[str, float] | None = None,
    cooling_storage_summary: dict[str, float] | None = None,
    cooling_generation_summary: dict[str, float] | None = None,
    renewable_summary: dict[str, float] | None = None,
    heating_generation_input_label: str = "Heating+DHW electricity",
) -> Path:
    cards = {
        "Heating demand": summary.get("QH_gen_out_kWh", 0.0),
        "DHW demand": summary.get("QW_gen_out_kWh", 0.0),
        "Cooling demand": summary.get("QC_gen_out_kWh", 0.0),
        heating_generation_input_label: summary.get("EHW_gen_in_kWh", 0.0) + summary.get("WHW_gen_aux_kWh", 0.0),
        "Cooling electricity": summary.get("EC_gen_in_kWh", 0.0) + summary.get("WC_gen_aux_kWh", 0.0),
        "SPF heating+DHW": summary.get("SPF_HW_gen", np.nan),
        "SEER cooling": summary.get("SEER_C_gen", np.nan),
    }
    if geometry_summary:
        cards.update(
            {
                "Useful floor area": geometry_summary.get("net_floor_area_m2", np.nan),
                "Footprint area": geometry_summary.get("footprint_area_m2", np.nan),
                "Floors": geometry_summary.get("n_floors", np.nan),
                "Roof area": geometry_summary.get("roof_area_m2", np.nan),
                "Ground slab area": geometry_summary.get("ground_slab_area_m2", np.nan),
            }
        )
    if dhw_design_summary:
        cards.update(
            {
                "EN 12831-3 DHW design day": dhw_design_summary.get("Q_W_design_day_kWh", 0.0),
                "EN 12831-3 selected DHW storage": dhw_design_summary.get("V_sto_selected_l", 0.0),
                "EN 12831-3 DHW design flow": dhw_design_summary.get("design_flow_m3_h", 0.0),
                "EN 12831-3 DHW supply margin": dhw_design_summary.get("supply_margin_min_kWh", 0.0),
            }
        )
    if performance_data_summary:
        cards.update(
            {
                "EN 14511 heating rating points": performance_data_summary.get(
                    "heating_rating_point_count", 0.0
                ),
                "EN 14511 cooling rating points": performance_data_summary.get(
                    "cooling_rating_point_count", 0.0
                ),
                "EN 14825 heating Cd": performance_data_summary.get(
                    "heating_degradation_coefficient", np.nan
                ),
                "EN 14825 cooling Cd": performance_data_summary.get(
                    "cooling_degradation_coefficient", np.nan
                ),
            }
        )
    if emission_summary:
        cards.update(
            {
                "EN 15316-2 heating losses": emission_summary.get("QH_em_ls_kWh", 0.0),
                "EN 15316-2 cooling losses": emission_summary.get("QC_em_ls_kWh", 0.0),
                "EN 15316-2 auxiliaries": emission_summary.get("W_em_aux_kWh", 0.0),
            }
        )
    if distribution_summary:
        cards.update(
            {
                "EN 15316-3 pipe losses": distribution_summary.get("Q_dis_ls_kWh", 0.0),
                "EN 15316-3 pump electricity": distribution_summary.get("W_dis_aux_kWh", 0.0),
                "EN 15316-3 DHW losses": distribution_summary.get("QW_dis_ls_kWh", 0.0),
            }
        )
    if storage_summary:
        cards.update(
            {
                "EN 15316-5 storage losses": storage_summary.get("Q_sto_ls_kWh", 0.0),
                "EN 15316-5 storage pump electricity": storage_summary.get("W_sto_aux_kWh", 0.0),
                "EN 15316-5 DHW storage losses": storage_summary.get("QW_sto_ls_kWh", 0.0),
            }
        )
    if cooling_system_summary:
        cards.update(
            {
                "EN 16798-9 cooling request": cooling_system_summary.get("QC_dis_out_tot_req_kWh", 0.0),
                "EN 16798-9 mean flow": cooling_system_summary.get("q_V_C_dis_mean_m3_h", 0.0),
            }
        )
    if cooling_storage_summary:
        cards.update(
            {
                "EN 16798-15 cooling storage gains": cooling_storage_summary.get("QC_sto_ls_tot_kWh", 0.0),
                "EN 16798-15 storage pump electricity": cooling_storage_summary.get("WC_sto_aux_kWh", 0.0),
            }
        )
    if cooling_generation_summary:
        cards.update(
            {
                "EN 16798-13 cooling electricity": cooling_generation_summary.get("EC_total_kWh", 0.0),
                "EN 16798-13 cooling SEER": cooling_generation_summary.get("SEER_C_gen", np.nan),
            }
        )
    if renewable_summary:
        cards.update(
            {
                "EN 15316-4-3 solar thermal used": renewable_summary.get(
                    "Q_solar_thermal_used_kWh", 0.0
                ),
                "EN 15316-4-6 PV generation": renewable_summary.get("E_PV_gen_kWh", 0.0),
                "PV self-consumed": renewable_summary.get("E_PV_self_consumed_kWh", 0.0),
                "Grid electricity after PV": renewable_summary.get("E_grid_after_PV_kWh", 0.0),
            }
        )
    plot_items = []
    for path in plot_paths:
        plot_items.append(
            f'<li><a href="{_relative_href(path, output_dir)}">'
            f'{html.escape(_plot_link_label(path))}</a></li>'
        )
    start_items = plot_items[:3]
    detailed_items = plot_items[3:]
    if iso_report is not None:
        detailed_items.append(
            f'<li><a href="{_relative_href(iso_report, output_dir)}">'
            "ISO52016 existing building report</a></li>"
        )
    workflow_audit = REPO_ROOT / "docs" / "simulation_workflow_audit.html"
    audit_items = [
        '<li><a href="'
        f'{_relative_href(workflow_audit, output_dir)}'
        '">Whole simulation workflow audit trail</a></li>'
    ]

    card_html = "\n".join(
        f"<div class='card'><span>{html.escape(name)}</span><strong>{value:,.2f}</strong></div>"
        for name, value in cards.items()
        if value is not None and np.isfinite(value)
    )
    page_title = (
        "EN 15316 and EN 16798 Heat-Pump System Inspection"
        if performance_data_summary
        and (cooling_generation_summary or cooling_storage_summary or cooling_system_summary)
        else "EN 15316 and EN 16798 Heat-Pump System Inspection"
        if cooling_generation_summary or cooling_storage_summary or cooling_system_summary
        else "EN 15316-2, EN 15316-3, EN 15316-5 and EN 15316-4-2 Inspection"
        if storage_summary and distribution_summary and emission_summary
        else "EN 15316-3, EN 15316-5 and EN 15316-4-2 Inspection"
        if storage_summary and distribution_summary
        else "EN 15316-5 and EN 15316-4-2 Inspection"
        if storage_summary
        else "EN 15316-2, EN 15316-3 and EN 15316-4-2 Inspection"
        if distribution_summary
        else "EN 15316-2 and EN 15316-4-2 Inspection"
        if emission_summary
        else "Heat Pump EN 15316-4-2 Inspection"
    )
    page_intro = (
        "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, EN 15316-3 distribution losses and pump auxiliaries, "
        "EN 15316-5 heating/DHW storage, EN 16798-9 cooling operating "
        "conditions, EN 16798-15 chilled storage, EN 16798-13 cooling "
        "generation, EN 14511/EN 14825 product data, heat-pump heating/DHW "
        "bin method, electricity use and seasonal performance."
        if performance_data_summary
        and (cooling_generation_summary or cooling_storage_summary or cooling_system_summary)
        else
        "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, EN 15316-3 distribution losses and pump auxiliaries, "
        "EN 15316-5 heating/DHW storage, EN 16798-9 cooling operating "
        "conditions, EN 16798-15 chilled storage, EN 16798-13 cooling "
        "generation, heat-pump heating/DHW bin method, electricity use and "
        "seasonal performance."
        if cooling_generation_summary or cooling_storage_summary or cooling_system_summary
        else
        "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, EN 15316-3 distribution losses and pump auxiliaries, "
        "EN 15316-5 heating/DHW storage losses and auxiliaries, DHW profile, "
        "heat-pump bin method, electricity use, backup energy, losses and "
        "seasonal performance."
        if storage_summary and distribution_summary and emission_summary
        else "Open the plots below to inspect the ISO52016 inputs, EN 15316-3 "
        "distribution losses and pump auxiliaries, EN 15316-5 heating/DHW "
        "storage losses and auxiliaries, DHW profile, heat-pump bin method, "
        "electricity use, backup energy, losses and seasonal performance."
        if storage_summary and distribution_summary
        else "Open the plots below to inspect the ISO52016 and DHW inputs, "
        "EN 15316-5 heating/DHW storage losses and auxiliaries, heat-pump bin "
        "method, electricity use, backup energy, losses and seasonal performance."
        if storage_summary
        else "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, EN 15316-3 distribution losses and pump auxiliaries, "
        "DHW profile, heat-pump bin method, electricity use, backup energy, losses "
        "and seasonal performance."
        if distribution_summary
        else
        "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, DHW profile, heat-pump bin method, electricity use, "
        "backup energy, losses and seasonal performance."
        if emission_summary
        else "Open the plots below to inspect the direct ISO52016 and DHW inputs, "
        "heat-pump bin method, electricity use, backup energy, losses and seasonal "
        "performance."
    )
    if dhw_design_summary:
        page_intro = page_intro.replace(
            "DHW profile",
            "EN 12831-3 DHW profile and sizing",
        )
        page_intro = page_intro.replace(
            "ISO52016 and DHW inputs",
            "ISO52016 and EN 12831-3 DHW inputs",
        )
    page_intro += (
        " Start with the overview, workflow handoff and sanity-check plots for a "
        "quick interpretation, then use the detailed plots for module-level review."
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(page_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 20px 0 28px; }}
    .card {{ border: 1px solid #d7dde5; border-radius: 8px; padding: 14px 16px; background: #f8fafc; }}
    .card span {{ display: block; color: #5b6777; font-size: 13px; margin-bottom: 8px; }}
    .card strong {{ font-size: 22px; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>{html.escape(page_title)}</h1>
  <p>{html.escape(page_intro)}</p>
  <div class="grid">{card_html}</div>
  <h2>Start Here</h2>
  <ul>{"".join(start_items)}</ul>
  <h2>Detailed Interactive Plots</h2>
  <ul>{"".join(detailed_items)}</ul>
  <h2>Audit Documentation</h2>
  <ul>{"".join(audit_items)}</ul>
</body>
</html>
"""
    index_path = output_dir / "inspection_index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def create_visual_outputs(
    hourly_sim: pd.DataFrame,
    loads: pd.DataFrame,
    result: pybui.HeatPumpSimulationResult,
    output_dir: Path,
    building_area: float,
    dhw_design_result: pybui.DHWDesignSimulationResult | None = None,
    emission_result: pybui.EmissionSimulationResult | None = None,
    distribution_result: pybui.DistributionSimulationResult | None = None,
    storage_result: pybui.StorageSimulationResult | None = None,
    cooling_system_result: pybui.CoolingSystemSimulationResult | None = None,
    cooling_storage_result: pybui.CoolingStorageSimulationResult | None = None,
    cooling_generation_result: pybui.CoolingGenerationSimulationResult | None = None,
    performance_data_result: pybui.HeatPumpPerformanceDataResult | None = None,
    renewable_result: pybui.RenewableEnergySimulationResult | None = None,
    geometry_summary: dict[str, float] | None = None,
) -> Path:
    iso_report = create_iso52016_visuals(hourly_sim, output_dir, building_area)
    allocated = allocate_bin_outputs_to_hours(loads, result.bins)
    allocated = merge_cooling_generation_to_allocated(allocated, cooling_generation_result)
    allocated.to_csv(output_dir / "heat_pump_hourly_allocated_results.csv")
    combined_summary = combined_generation_summary(result.summary, cooling_generation_result)

    input_title = (
        "Final Loads Sent to Heating/DHW Heat Pump and EN 16798-13 Cooling Generator"
        if cooling_generation_result is not None
        else "Storage-Adjusted Loads Sent to the Heat Pump"
        if storage_result is not None
        else "Distribution-Adjusted Loads and DHW Inputs Sent to the Heat Pump"
        if distribution_result is not None
        else "Space Emission Loads and DHW Inputs Sent to the Heat Pump"
        if emission_result is not None
        else "ISO52016 and DHW Inputs Sent to the Heat Pump"
    )
    plot_paths = [
        plot_user_overview(loads, allocated, combined_summary, output_dir, geometry_summary),
        plot_workflow_handoff(
            loads,
            allocated,
            combined_summary,
            output_dir,
            emission_result=emission_result,
            distribution_result=distribution_result,
            storage_result=storage_result,
            cooling_storage_result=cooling_storage_result,
            cooling_generation_result=cooling_generation_result,
        ),
        plot_sanity_checks(loads, allocated, combined_summary, output_dir, geometry_summary),
    ]
    if dhw_design_result is not None:
        plot_paths.append(plot_dhw_design_12831_3(dhw_design_result, output_dir))
    if performance_data_result is not None:
        plot_paths.append(plot_performance_data_14511_14825(performance_data_result, output_dir))
    if renewable_result is not None:
        plot_paths.append(plot_renewables_15316_4_3_4_6(renewable_result, output_dir))
    plot_paths.append(plot_input_timeseries(loads, output_dir, title=input_title))
    if emission_result is not None:
        plot_paths.extend(
            [
                plot_emission_timeseries(emission_result, output_dir),
                plot_emission_monthly(emission_result, output_dir),
            ]
        )
    if distribution_result is not None:
        plot_paths.extend(
            [
                plot_distribution_timeseries(distribution_result, output_dir),
                plot_distribution_monthly(distribution_result, output_dir),
            ]
        )
    if storage_result is not None:
        plot_paths.extend(
            [
                plot_storage_timeseries(storage_result, output_dir),
                plot_storage_monthly(storage_result, output_dir),
            ]
        )
    if cooling_system_result is not None:
        plot_paths.append(plot_cooling_system_16798_9(cooling_system_result, output_dir))
    if cooling_storage_result is not None:
        plot_paths.append(plot_cooling_storage_16798_15(cooling_storage_result, output_dir))
    if cooling_generation_result is not None:
        plot_paths.extend(plot_cooling_generation_16798_13(cooling_generation_result, output_dir))
    plot_paths.extend(
        [
            plot_heat_pump_hourly(allocated, output_dir),
            plot_monthly_summary(loads, allocated, output_dir),
            plot_bin_balance(result.bins, output_dir),
            plot_bin_performance(result.bins, output_dir),
            plot_summary_sankey(combined_summary, output_dir),
        ]
    )
    emission_summary = emission_result.summary if emission_result is not None else None
    distribution_summary = (
        distribution_result.summary if distribution_result is not None else None
    )
    storage_summary = storage_result.summary if storage_result is not None else None
    cooling_system_summary = (
        cooling_system_result.summary if cooling_system_result is not None else None
    )
    cooling_storage_summary = (
        cooling_storage_result.summary if cooling_storage_result is not None else None
    )
    cooling_generation_summary = (
        cooling_generation_result.summary if cooling_generation_result is not None else None
    )
    renewable_summary = renewable_summary_with_electric_load(
        renewable_result,
        combined_summary.get("E_total_electricity_kWh", 0.0),
    )
    return create_inspection_index(
        output_dir,
        combined_summary,
        plot_paths,
        iso_report,
        geometry_summary=geometry_summary,
        dhw_design_summary=(
            dhw_design_result.summary if dhw_design_result is not None else None
        ),
        performance_data_summary=(
            performance_data_result.summary if performance_data_result is not None else None
        ),
        emission_summary=emission_summary,
        distribution_summary=distribution_summary,
        storage_summary=storage_summary,
        cooling_system_summary=cooling_system_summary,
        cooling_storage_summary=cooling_storage_summary,
        cooling_generation_summary=cooling_generation_summary,
        renewable_summary=renewable_summary,
    )


def create_non_heat_pump_visual_outputs(
    hourly_sim: pd.DataFrame,
    loads: pd.DataFrame,
    heating_generation_result,
    heating_generation_method: str,
    output_dir: Path,
    building_area: float,
    dhw_design_result: pybui.DHWDesignSimulationResult | None = None,
    emission_result: pybui.EmissionSimulationResult | None = None,
    distribution_result: pybui.DistributionSimulationResult | None = None,
    storage_result: pybui.StorageSimulationResult | None = None,
    cooling_system_result: pybui.CoolingSystemSimulationResult | None = None,
    cooling_storage_result: pybui.CoolingStorageSimulationResult | None = None,
    cooling_generation_result: pybui.CoolingGenerationSimulationResult | None = None,
    performance_data_result: pybui.HeatPumpPerformanceDataResult | None = None,
    renewable_result: pybui.RenewableEnergySimulationResult | None = None,
    geometry_summary: dict[str, float] | None = None,
) -> Path:
    iso_report = create_iso52016_visuals(hourly_sim, output_dir, building_area)
    combined_summary = combined_system_summary(
        heating_generation_result.summary,
        cooling_generation_result,
    )
    renewable_summary = renewable_summary_with_electric_load(
        renewable_result,
        combined_summary.get("E_total_electricity_kWh", 0.0),
    )
    plot_paths = [
        plot_generic_system_overview(
            loads,
            combined_summary,
            output_dir,
            heating_generation_method,
            geometry_summary,
            renewable_summary,
        )
    ]
    if dhw_design_result is not None:
        plot_paths.append(plot_dhw_design_12831_3(dhw_design_result, output_dir))
    if performance_data_result is not None:
        plot_paths.append(plot_performance_data_14511_14825(performance_data_result, output_dir))
    if renewable_result is not None:
        plot_paths.append(plot_renewables_15316_4_3_4_6(renewable_result, output_dir))
    plot_paths.append(
        plot_input_timeseries(
            loads,
            output_dir,
            title="Final Loads Sent to Heating/DHW and Cooling Generators",
        )
    )
    if emission_result is not None:
        plot_paths.extend(
            [
                plot_emission_timeseries(emission_result, output_dir),
                plot_emission_monthly(emission_result, output_dir),
            ]
        )
    if distribution_result is not None:
        plot_paths.extend(
            [
                plot_distribution_timeseries(distribution_result, output_dir),
                plot_distribution_monthly(distribution_result, output_dir),
            ]
        )
    if storage_result is not None:
        plot_paths.extend(
            [
                plot_storage_timeseries(storage_result, output_dir),
                plot_storage_monthly(storage_result, output_dir),
            ]
        )
    if cooling_system_result is not None:
        plot_paths.append(plot_cooling_system_16798_9(cooling_system_result, output_dir))
    if cooling_storage_result is not None:
        plot_paths.append(plot_cooling_storage_16798_15(cooling_storage_result, output_dir))
    if cooling_generation_result is not None:
        plot_paths.extend(plot_cooling_generation_16798_13(cooling_generation_result, output_dir))
    plot_paths.append(
        plot_heating_generation_system(
            heating_generation_result,
            output_dir,
            heating_generation_method,
        )
    )

    return create_inspection_index(
        output_dir,
        combined_summary,
        plot_paths,
        iso_report,
        geometry_summary=geometry_summary,
        dhw_design_summary=(
            dhw_design_result.summary if dhw_design_result is not None else None
        ),
        performance_data_summary=(
            performance_data_result.summary if performance_data_result is not None else None
        ),
        emission_summary=emission_result.summary if emission_result is not None else None,
        distribution_summary=(
            distribution_result.summary if distribution_result is not None else None
        ),
        storage_summary=storage_result.summary if storage_result is not None else None,
        cooling_system_summary=(
            cooling_system_result.summary if cooling_system_result is not None else None
        ),
        cooling_storage_summary=(
            cooling_storage_result.summary if cooling_storage_result is not None else None
        ),
        cooling_generation_summary=(
            cooling_generation_result.summary if cooling_generation_result is not None else None
        ),
        renewable_summary=renewable_summary,
        heating_generation_input_label="Heating+DHW generation input",
    )


def resolve_system_methods(args: argparse.Namespace) -> tuple[str, str, str, str, str, str]:
    if args.calculation_path in {
        "full",
        "full-renewables",
        "boiler-full",
        "chp-full",
        "district-full",
    } or args.calculation_path is None:
        methods = ["en15316-2", "en15316-3", "en15316-5", "en16798-9", "en16798-15", "en16798-13"]
    elif args.calculation_path == "emission-distribution":
        methods = ["en15316-2", "en15316-3", "simple", "en16798-9", "simple", "en16798-13"]
    elif args.calculation_path == "emission-storage":
        methods = ["en15316-2", "simple", "en15316-5", "en16798-9", "en16798-15", "en16798-13"]
    elif args.calculation_path == "distribution-storage":
        methods = ["simple", "en15316-3", "en15316-5", "en16798-9", "en16798-15", "en16798-13"]
    elif args.calculation_path == "storage-only":
        methods = ["simple", "simple", "en15316-5", "en16798-9", "en16798-15", "en16798-13"]
    elif args.calculation_path == "emission-only":
        methods = ["en15316-2", "simple", "simple", "en16798-9", "simple", "en16798-13"]
    elif args.calculation_path == "no-cooling-storage":
        methods = ["en15316-2", "en15316-3", "en15316-5", "en16798-9", "simple", "en16798-13"]
    elif args.calculation_path == "heat-pump-cooling":
        methods = ["en15316-2", "en15316-3", "en15316-5", "en16798-9", "simple", "heat-pump-simple"]
    elif args.calculation_path == "simple":
        methods = ["simple", "simple", "simple", "simple", "simple", "heat-pump-simple"]
    else:
        raise ValueError(f"Unsupported calculation path: {args.calculation_path}")

    if args.emission_method is not None:
        methods[0] = args.emission_method
    if args.distribution_method is not None:
        methods[1] = args.distribution_method
    if args.storage_method is not None:
        methods[2] = args.storage_method
    if args.cooling_system_method is not None:
        methods[3] = args.cooling_system_method
    if args.cooling_storage_method is not None:
        methods[4] = args.cooling_storage_method
    if args.cooling_generation_method is not None:
        methods[5] = args.cooling_generation_method

    return tuple(methods)  # type: ignore[return-value]


def resolve_heating_generation_method(args: argparse.Namespace) -> str:
    if args.heating_generation_method is not None:
        return args.heating_generation_method
    if args.calculation_path == "boiler-full":
        return "en15316-4-1"
    if args.calculation_path == "chp-full":
        return "en15316-4-4"
    if args.calculation_path == "district-full":
        return "en15316-4-5"
    return "en15316-4-2"


def resolve_renewable_method(args: argparse.Namespace) -> str:
    if args.renewable_method is not None:
        return args.renewable_method
    if args.calculation_path == "full-renewables":
        return "en15316-4-3-4-6"
    return "simple"


def resolve_performance_data_method(args: argparse.Namespace) -> str:
    if args.performance_data_method is not None:
        return args.performance_data_method
    if args.calculation_path == "simple":
        return "simple"
    return "en14511-14825"


def resolve_dhw_design_method(args: argparse.Namespace) -> str:
    if args.dhw_design_method is not None:
        return args.dhw_design_method
    if args.calculation_path == "simple":
        return "simple"
    return "en12831-3"


def run_example(args: argparse.Namespace) -> None:
    scenario = args.scenario
    (
        emission_method,
        distribution_method,
        storage_method,
        cooling_system_method,
        cooling_storage_method,
        cooling_generation_method,
    ) = resolve_system_methods(args)
    heating_generation_method = resolve_heating_generation_method(args)
    renewable_method = resolve_renewable_method(args)
    performance_data_method = resolve_performance_data_method(args)
    dhw_design_method = resolve_dhw_design_method(args)
    building = example_building(scenario)
    geometry_summary = building_geometry_summary(building)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            scenario,
            emission_method,
            distribution_method,
            storage_method,
            cooling_system_method,
            cooling_storage_method,
            cooling_generation_method,
            performance_data_method,
            dhw_design_method,
            heating_generation_method,
            renewable_method,
        )
    )
    dhw_country = args.dhw_calendar_country or SCENARIO_DHW_COUNTRY[scenario]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([geometry_summary]).to_csv(
        output_dir / "building_geometry_summary.csv",
        index=False,
    )

    hourly_sim = run_iso52016(building, args.weather_source, args.path_weather_file)
    dhw_kWh = make_dhw_profile(
        pd.DatetimeIndex(hourly_sim.index),
        building_area=float(building["building"]["net_floor_area"]),
        country=dhw_country,
        cold_water_temperature_C=dhw_cold_water_temperature_C(scenario),
        annex_b_profile_key=SCENARIO_DHW_ANNEX_B_PROFILE[scenario],
    )
    loads = prepare_heat_pump_loads(hourly_sim, dhw_kWh)

    performance_data_result = None
    if performance_data_method == "en14511-14825":
        performance_data_result = heat_pump_performance_data(scenario)
        heating_map = performance_data_result.heating_map
        cooling_map = performance_data_result.cooling_map
    elif performance_data_method == "simple":
        heating_map, cooling_map = heat_pump_maps(scenario)
    else:
        raise ValueError(
            "--performance-data-method must be 'en14511-14825' or 'simple'."
        )

    distribution_config = distribution_system_config(scenario, building)
    storage_config = storage_system_config(scenario)
    dhw_design_result = None
    if dhw_design_method == "en12831-3":
        dhw_design_calc = pybui.DHWDesignLoadCalculator(
            dhw_design_system_config(
                scenario,
                distribution_config,
                storage_config,
                heating_map,
                sizing_mode=args.dhw_sizing_mode,
            )
        )
        dhw_design_result = dhw_design_calc.run_timeseries(loads)
        apply_dhw_design_to_system_configs(
            dhw_design_result,
            distribution_config,
            storage_config,
        )
    elif dhw_design_method != "simple":
        raise ValueError("--dhw-design-method must be 'en12831-3' or 'simple'.")

    emission_result = None
    if emission_method == "en15316-2":
        emission_calc = pybui.EmissionSystemCalculator(emission_system_config(scenario))
        emission_building = building_with_emission_setpoints(building, emission_calc)
        hourly_sim_emission = run_iso52016(
            emission_building,
            args.weather_source,
            args.path_weather_file,
        )
        emission_loads = prepare_emission_loads(hourly_sim, hourly_sim_emission)
        emission_result = emission_calc.run_timeseries(emission_loads)
        loads["Q_H_building_kWh"] = emission_result.timeseries["Q_H_em_out_kWh"]
        loads["Q_C_building_kWh"] = emission_result.timeseries["Q_C_em_out_kWh"]
        loads["Q_H_em_loss_kWh"] = emission_result.timeseries["Q_H_em_ls_kWh"]
        loads["Q_C_em_loss_kWh"] = emission_result.timeseries["Q_C_em_ls_kWh"]
        loads["W_H_em_aux_kWh"] = emission_result.timeseries["W_H_em_aux_kWh"]
        loads["W_C_em_aux_kWh"] = emission_result.timeseries["W_C_em_aux_kWh"]
        loads["Q_H_kWh"] = emission_result.timeseries["Q_H_em_in_kWh"]
        loads["Q_C_kWh"] = emission_result.timeseries["Q_C_em_in_kWh"]
    elif emission_method != "simple":
        raise ValueError("--emission-method must be 'en15316-2' or 'simple'.")

    cooling_system_result = None
    if cooling_system_method == "en16798-9":
        cooling_system_calc = pybui.CoolingSystemCalculator(cooling_system_config(scenario))
        cooling_system_result = cooling_system_calc.run_timeseries(loads)
        loads["Q_C_16798_9_out_kWh"] = cooling_system_result.timeseries[
            "Q_C_dis_out_tot_req_kWh"
        ]
        loads["Q_C_gen_in_req_16798_9_kWh"] = cooling_system_result.timeseries[
            "Q_C_gen_in_req_kWh"
        ]
        loads["theta_C_dis_supply_C"] = cooling_system_result.timeseries[
            "theta_C_dis_supply_C"
        ]
        loads["theta_C_dis_return_C"] = cooling_system_result.timeseries[
            "theta_C_dis_return_C"
        ]
        loads["theta_C_gen_out_req_C"] = cooling_system_result.timeseries[
            "theta_C_gen_out_req_C"
        ]
        loads["T_C_sink_C"] = cooling_system_result.timeseries["T_C_sink_C"]
    elif cooling_system_method != "simple":
        raise ValueError("--cooling-system-method must be 'en16798-9' or 'simple'.")

    distribution_result = None
    if distribution_method == "en15316-3":
        distribution_calc = pybui.DistributionSystemCalculator(distribution_config)
        distribution_result = distribution_calc.run_timeseries(loads)
        loads["Q_H_distribution_out_kWh"] = distribution_result.timeseries["Q_H_dis_out_kWh"]
        loads["Q_C_distribution_out_kWh"] = distribution_result.timeseries["Q_C_dis_out_kWh"]
        loads["Q_W_distribution_out_kWh"] = distribution_result.timeseries["Q_W_dis_out_kWh"]
        loads["Q_H_dis_loss_kWh"] = distribution_result.timeseries["Q_H_dis_ls_kWh"]
        loads["Q_C_dis_loss_kWh"] = distribution_result.timeseries["Q_C_dis_ls_kWh"]
        loads["Q_W_dis_loss_kWh"] = distribution_result.timeseries["Q_W_dis_ls_kWh"]
        loads["W_H_dis_aux_kWh"] = distribution_result.timeseries["W_H_dis_aux_kWh"]
        loads["W_C_dis_aux_kWh"] = distribution_result.timeseries["W_C_dis_aux_kWh"]
        loads["W_W_dis_aux_kWh"] = distribution_result.timeseries["W_W_dis_aux_kWh"]
        loads["Q_H_kWh"] = distribution_result.timeseries["Q_H_dis_in_kWh"]
        loads["Q_C_kWh"] = distribution_result.timeseries["Q_C_dis_in_kWh"]
        loads["Q_W_kWh"] = distribution_result.timeseries["Q_W_dis_in_kWh"]
    elif distribution_method != "simple":
        raise ValueError("--distribution-method must be 'en15316-3' or 'simple'.")

    storage_result = None
    if storage_method == "en15316-5":
        storage_calc = pybui.StorageSystemCalculator(storage_config)
        storage_result = storage_calc.run_timeseries(loads)
        loads["Q_H_storage_out_kWh"] = storage_result.timeseries["Q_H_sto_out_kWh"]
        loads["Q_W_storage_out_kWh"] = storage_result.timeseries["Q_W_sto_out_kWh"]
        loads["Q_H_sto_loss_kWh"] = storage_result.timeseries["Q_H_sto_ls_kWh"]
        loads["Q_W_sto_loss_kWh"] = storage_result.timeseries["Q_W_sto_ls_kWh"]
        loads["W_H_sto_aux_kWh"] = storage_result.timeseries["W_H_sto_aux_kWh"]
        loads["W_W_sto_aux_kWh"] = storage_result.timeseries["W_W_sto_aux_kWh"]
        loads["Q_H_kWh"] = storage_result.timeseries["Q_H_sto_in_kWh"]
        loads["Q_W_kWh"] = storage_result.timeseries["Q_W_sto_in_kWh"]
    elif storage_method != "simple":
        raise ValueError("--storage-method must be 'en15316-5' or 'simple'.")

    cooling_storage_result = None
    if cooling_storage_method == "en16798-15":
        cooling_storage_calc = pybui.CoolingStorageSystemCalculator(
            cooling_storage_system_config(scenario)
        )
        cooling_storage_result = cooling_storage_calc.run_timeseries(loads)
        loads["Q_C_storage_out_kWh"] = cooling_storage_result.timeseries[
            "Q_C_sto_out_kWh"
        ]
        loads["Q_C_sto_loss_kWh"] = cooling_storage_result.timeseries[
            "Q_C_sto_ls_tot_kWh"
        ]
        loads["W_C_sto_aux_kWh"] = cooling_storage_result.timeseries[
            "W_C_sto_aux_kWh"
        ]
        loads["Q_C_kWh"] = cooling_storage_result.timeseries["Q_C_sto_in_kWh"]
        loads["T_C_sink_C"] = cooling_storage_result.timeseries["T_C_sink_C"]
        loads["theta_C_gen_out_req_C"] = cooling_storage_result.timeseries[
            "theta_C_sto_in_req_C"
        ]
    elif cooling_storage_method != "simple":
        raise ValueError("--cooling-storage-method must be 'en16798-15' or 'simple'.")

    renewable_result = None
    if renewable_method == "en15316-4-3-4-6":
        renewable_calc = pybui.RenewableEnergySystemCalculator(
            renewable_system_config(scenario, building)
        )
        renewable_result = renewable_calc.run_timeseries(loads)
        loads["Q_H_before_renewables_kWh"] = renewable_result.timeseries[
            "Q_H_before_solar_kWh"
        ]
        loads["Q_W_before_renewables_kWh"] = renewable_result.timeseries[
            "Q_W_before_solar_kWh"
        ]
        loads["Q_H_solar_thermal_used_kWh"] = renewable_result.timeseries[
            "Q_H_solar_thermal_used_kWh"
        ]
        loads["Q_W_solar_thermal_used_kWh"] = renewable_result.timeseries[
            "Q_W_solar_thermal_used_kWh"
        ]
        loads["E_PV_gen_kWh"] = renewable_result.timeseries["E_PV_gen_kWh"]
        loads["Q_H_kWh"] = renewable_result.timeseries["Q_H_after_solar_kWh"]
        loads["Q_W_kWh"] = renewable_result.timeseries["Q_W_after_solar_kWh"]
    elif renewable_method != "simple":
        raise ValueError("--renewable-method must be 'en15316-4-3-4-6' or 'simple'.")

    ensure_heating_and_cooling(loads)

    cooling_generation_result = None
    cooling_enabled_for_heat_pump = heating_generation_method == "en15316-4-2"
    if cooling_generation_method == "en16798-13":
        cooling_generation_calc = pybui.CoolingGenerationSystemCalculator(
            cooling_generation_system_config(
                scenario,
                cooling_map,
                performance_data_method=performance_data_method,
            )
        )
        cooling_generation_result = cooling_generation_calc.run_timeseries(loads)
        cooling_enabled_for_heat_pump = False
    elif cooling_generation_method != "heat-pump-simple":
        raise ValueError(
            "--cooling-generation-method must be 'en16798-13' or 'heat-pump-simple'."
        )
    elif heating_generation_method != "en15316-4-2":
        raise ValueError(
            "--cooling-generation-method heat-pump-simple requires "
            "--heating-generation-method en15316-4-2."
        )

    result = None
    heating_generation_result = None
    if heating_generation_method == "en15316-4-2":
        hp_loads = loads.copy()
        if cooling_generation_result is not None:
            hp_loads["Q_C_kWh"] = 0.0
        calc = pybui.HeatPumpSystemCalculator(
            heat_pump_config(
                scenario,
                heating_map,
                cooling_map,
                include_internal_storage_losses=storage_method == "simple",
                cooling_enabled=cooling_enabled_for_heat_pump,
                performance_data_method=performance_data_method,
            )
        )
        result = calc.run_timeseries(hp_loads)
        heating_generation_result = result
    else:
        generation_loads = loads.copy()
        site_el_cols = [
            "W_H_em_aux_kWh",
            "W_C_em_aux_kWh",
            "W_H_dis_aux_kWh",
            "W_C_dis_aux_kWh",
            "W_W_dis_aux_kWh",
            "W_H_sto_aux_kWh",
            "W_W_sto_aux_kWh",
            "W_C_sto_aux_kWh",
        ]
        generation_loads["E_site_el_load_kWh"] = 0.0
        for col in site_el_cols:
            if col in generation_loads:
                generation_loads["E_site_el_load_kWh"] += generation_loads[col].fillna(0.0)
        if cooling_generation_result is not None:
            generation_loads["E_site_el_load_kWh"] += cooling_generation_result.timeseries[
                "E_C_total_kWh"
            ].reindex(generation_loads.index).fillna(0.0)
        if heating_generation_method == "en15316-4-1":
            heating_generation_result = pybui.CombustionBoilerSystemCalculator(
                boiler_generation_config(scenario)
            ).run_timeseries(generation_loads)
        elif heating_generation_method == "en15316-4-4":
            heating_generation_result = pybui.CogenerationSystemCalculator(
                cogeneration_system_config(scenario)
            ).run_timeseries(generation_loads)
        elif heating_generation_method == "en15316-4-5":
            heating_generation_result = pybui.DistrictEnergySystemCalculator(
                district_energy_system_config(scenario)
            ).run_timeseries(generation_loads)
        else:
            raise ValueError(
                "--heating-generation-method must be en15316-4-2, en15316-4-1, "
                "en15316-4-4 or en15316-4-5."
            )

    loads.to_csv(output_dir / "iso52016_loads_with_dhw.csv")
    if dhw_design_result is not None:
        dhw_design_result.timeseries.to_csv(
            output_dir / "dhw_12831_3_design_timeseries.csv"
        )
        pd.DataFrame([dhw_design_result.summary]).to_csv(
            output_dir / "dhw_12831_3_design_summary.csv",
            index=False,
        )
    if emission_result is not None:
        emission_result.timeseries.to_csv(output_dir / "emission_15316_2_hourly_results.csv")
        pd.DataFrame([emission_result.summary]).to_csv(
            output_dir / "emission_15316_2_summary.csv",
            index=False,
        )
    if distribution_result is not None:
        distribution_result.timeseries.to_csv(
            output_dir / "distribution_15316_3_hourly_results.csv"
        )
        pd.DataFrame([distribution_result.summary]).to_csv(
            output_dir / "distribution_15316_3_summary.csv",
            index=False,
        )
    if storage_result is not None:
        storage_result.timeseries.to_csv(output_dir / "storage_15316_5_hourly_results.csv")
        pd.DataFrame([storage_result.summary]).to_csv(
            output_dir / "storage_15316_5_summary.csv",
            index=False,
        )
    if cooling_system_result is not None:
        cooling_system_result.timeseries.to_csv(
            output_dir / "cooling_16798_9_hourly_results.csv"
        )
        pd.DataFrame([cooling_system_result.summary]).to_csv(
            output_dir / "cooling_16798_9_summary.csv",
            index=False,
        )
    if cooling_storage_result is not None:
        cooling_storage_result.timeseries.to_csv(
            output_dir / "cooling_storage_16798_15_hourly_results.csv"
        )
        pd.DataFrame([cooling_storage_result.summary]).to_csv(
            output_dir / "cooling_storage_16798_15_summary.csv",
            index=False,
        )
    if cooling_generation_result is not None:
        cooling_generation_result.timeseries.to_csv(
            output_dir / "cooling_generation_16798_13_hourly_results.csv"
        )
        pd.DataFrame([cooling_generation_result.summary]).to_csv(
            output_dir / "cooling_generation_16798_13_summary.csv",
            index=False,
        )
    if renewable_result is not None:
        renewable_result.timeseries.to_csv(
            output_dir / "renewables_15316_4_3_4_6_hourly_results.csv"
        )
        pd.DataFrame([renewable_result.summary]).to_csv(
            output_dir / "renewables_15316_4_3_4_6_summary.csv",
            index=False,
        )
    if performance_data_result is not None:
        performance_data_result.rating_points.to_csv(
            output_dir / "performance_14511_14825_rating_points.csv",
            index=False,
        )
        performance_data_result.heating_map.to_csv(
            output_dir / "performance_14511_14825_heating_map.csv",
            index=False,
        )
        performance_data_result.cooling_map.to_csv(
            output_dir / "performance_14511_14825_cooling_map.csv",
            index=False,
        )
        pd.DataFrame([performance_data_result.summary]).to_csv(
            output_dir / "performance_14511_14825_summary.csv",
            index=False,
        )
    if heating_generation_method == "en15316-4-2":
        result.bins.to_csv(output_dir / "heat_pump_bin_results.csv", index=False)
        pd.DataFrame([result.summary]).to_csv(output_dir / "heat_pump_summary.csv", index=False)
        combined_summary = combined_generation_summary(result.summary, cooling_generation_result)
    else:
        generation_slug = heating_generation_method.replace("-", "_")
        heating_generation_result.timeseries.to_csv(
            output_dir / f"heating_generation_{generation_slug}_hourly_results.csv"
        )
        pd.DataFrame([heating_generation_result.summary]).to_csv(
            output_dir / f"heating_generation_{generation_slug}_summary.csv",
            index=False,
        )
        combined_summary = combined_system_summary(
            heating_generation_result.summary,
            cooling_generation_result,
        )
    pd.DataFrame([combined_summary]).to_csv(
        output_dir / "combined_generation_summary.csv",
        index=False,
    )
    if heating_generation_method == "en15316-4-2":
        inspection_index = create_visual_outputs(
            hourly_sim=hourly_sim,
            loads=loads,
            result=result,
            output_dir=output_dir,
            building_area=float(building["building"]["net_floor_area"]),
            dhw_design_result=dhw_design_result,
            emission_result=emission_result,
            distribution_result=distribution_result,
            storage_result=storage_result,
            cooling_system_result=cooling_system_result,
            cooling_storage_result=cooling_storage_result,
            cooling_generation_result=cooling_generation_result,
            performance_data_result=performance_data_result,
            renewable_result=renewable_result,
            geometry_summary=geometry_summary,
        )
    else:
        inspection_index = create_non_heat_pump_visual_outputs(
            hourly_sim=hourly_sim,
            loads=loads,
            heating_generation_result=heating_generation_result,
            heating_generation_method=heating_generation_method,
            output_dir=output_dir,
            building_area=float(building["building"]["net_floor_area"]),
            dhw_design_result=dhw_design_result,
            emission_result=emission_result,
            distribution_result=distribution_result,
            storage_result=storage_result,
            cooling_system_result=cooling_system_result,
            cooling_storage_result=cooling_storage_result,
            cooling_generation_result=cooling_generation_result,
            performance_data_result=performance_data_result,
            renewable_result=renewable_result,
            geometry_summary=geometry_summary,
        )

    print(f"\nSystem example completed for scenario: {scenario}")
    print(f"Emission calculation mode: {emission_method}")
    print(f"Distribution calculation mode: {distribution_method}")
    print(f"Storage calculation mode: {storage_method}")
    print(f"Cooling operating-condition mode: {cooling_system_method}")
    print(f"Cooling storage mode: {cooling_storage_method}")
    print(f"Cooling generation mode: {cooling_generation_method}")
    print(f"Heating/DHW generation mode: {heating_generation_method}")
    print(f"Renewable mode: {renewable_method}")
    print(f"Performance data mode: {performance_data_method}")
    print(f"DHW design mode: {dhw_design_method}")
    if dhw_design_result is not None:
        dhw_summary = dhw_design_result.summary
        print(
            "EN 12831-3 DHW design: "
            f"{dhw_summary['Q_W_design_day_kWh']:,.2f} kWh design day, "
            f"{dhw_summary['V_sto_selected_l']:,.0f} l selected storage, "
            f"{dhw_summary['phi_N_selected_kW']:.2f} kW selected generator power"
        )
        print(
            "EN 12831-3 supply-curve check: "
            f"margin {dhw_summary['supply_margin_min_kWh']:.3f} kWh, "
            f"satisfied={dhw_summary['sizing_satisfied']}"
        )
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Visual inspection page: {inspection_index.resolve()}")
    if performance_data_result is not None:
        print(
            "EN 14511/EN 14825 rating points: "
            f"{int(performance_data_result.summary['rating_point_count'])}"
        )
    if emission_result is not None:
        cooling_input_label = (
            "EN 16798-13 cooling generator input load"
            if cooling_generation_result is not None
            else "Heat-pump space cooling input load"
        )
        heating_input_label = (
            "Heat-pump space heating input load"
            if heating_generation_method == "en15316-4-2"
            else "Heating generator space heating input load"
        )
        print(f"ISO52016 space heating need: {emission_result.summary['QH_em_out_kWh']:,.1f} kWh")
        print(f"EN 15316-2 heating emission losses: {emission_result.summary['QH_em_ls_kWh']:,.1f} kWh")
        print(f"{heating_input_label}: {loads['Q_H_kWh'].sum():,.1f} kWh")
        print(f"ISO52016 space cooling need: {emission_result.summary['QC_em_out_kWh']:,.1f} kWh")
        print(f"EN 15316-2 cooling emission losses: {emission_result.summary['QC_em_ls_kWh']:,.1f} kWh")
        print(f"{cooling_input_label}: {loads['Q_C_kWh'].sum():,.1f} kWh")
        print(f"EN 15316-2 emission auxiliaries: {emission_result.summary['W_em_aux_kWh']:,.1f} kWh")
    else:
        load_label = (
            "Heat-pump input load"
            if distribution_result is not None or storage_result is not None
            else "Demand"
        )
        if heating_generation_method != "en15316-4-2" and load_label == "Heat-pump input load":
            load_label = "Generator input load"
        print(f"Space heating {load_label.lower()}: {loads['Q_H_kWh'].sum():,.1f} kWh")
        print(f"Space cooling {load_label.lower()}: {loads['Q_C_kWh'].sum():,.1f} kWh")
    if distribution_result is not None:
        print(f"EN 15316-3 heating distribution losses: {distribution_result.summary['QH_dis_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-3 cooling distribution losses: {distribution_result.summary['QC_dis_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-3 DHW distribution losses: {distribution_result.summary['QW_dis_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-3 distribution pump auxiliaries: {distribution_result.summary['W_dis_aux_kWh']:,.1f} kWh")
    if storage_result is not None:
        print(f"EN 15316-5 heating storage losses: {storage_result.summary['QH_sto_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-5 DHW storage losses: {storage_result.summary['QW_sto_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-5 storage pump auxiliaries: {storage_result.summary['W_sto_aux_kWh']:,.1f} kWh")
        print(f"DHW storage output demand: {storage_result.summary['QW_sto_out_kWh']:,.1f} kWh")
        print(f"Heating/DHW generator DHW input load: {loads['Q_W_kWh'].sum():,.1f} kWh")
    else:
        print(f"DHW demand: {loads['Q_W_kWh'].sum():,.1f} kWh")
    print(
        "Example building geometry: "
        f"{geometry_summary['net_floor_area_m2']:,.1f} m2 useful floor area, "
        f"{geometry_summary['footprint_area_m2']:,.1f} m2 footprint, "
        f"{int(geometry_summary['n_floors'])} floors"
    )
    if cooling_system_result is not None:
        print(f"EN 16798-9 cooling request: {cooling_system_result.summary['QC_dis_out_tot_req_kWh']:,.1f} kWh")
        print(f"EN 16798-9 mean cooling flow: {cooling_system_result.summary['q_V_C_dis_mean_m3_h']:.3f} m3/h")
    if cooling_storage_result is not None:
        print(f"EN 16798-15 cooling storage heat gains: {cooling_storage_result.summary['QC_sto_ls_tot_kWh']:,.1f} kWh")
        print(f"EN 16798-15 cooling storage auxiliaries: {cooling_storage_result.summary['WC_sto_aux_kWh']:,.1f} kWh")
        print(f"Cooling generator input after storage: {loads['Q_C_kWh'].sum():,.1f} kWh")
    if renewable_result is not None:
        renewable_summary = renewable_summary_with_electric_load(
            renewable_result,
            combined_summary.get("E_total_electricity_kWh", 0.0),
        )
        print(f"EN 15316-4-3 solar thermal used: {renewable_summary['Q_solar_thermal_used_kWh']:,.1f} kWh")
        print(f"EN 15316-4-6 PV generation: {renewable_summary['E_PV_gen_kWh']:,.1f} kWh")
        print(f"Grid electricity after PV: {renewable_summary['E_grid_after_PV_kWh']:,.1f} kWh")
    gen_input_label = (
        "Heating and DHW electricity, including backup"
        if heating_generation_method == "en15316-4-2"
        else "Heating and DHW generation input"
    )
    print(f"{gen_input_label}: {combined_summary['EHW_gen_in_kWh']:,.1f} kWh")
    print(
        "Heating and DHW auxiliaries: "
        f"{heating_generation_result.summary['WHW_gen_aux_kWh']:,.1f} kWh"
    )
    if cooling_generation_result is not None:
        print(f"EN 16798-13 cooling electricity: {cooling_generation_result.summary['EC_total_kWh']:,.1f} kWh")
        print(f"EN 16798-13 cooling rejected heat: {cooling_generation_result.summary['QC_gen_out_kWh']:,.1f} kWh")
    else:
        print(f"Cooling electricity: {result.summary['EC_gen_in_kWh']:,.1f} kWh")
    print(f"SPF_HW_gen: {combined_summary['SPF_HW_gen']:.2f}")
    print(f"SEER_C_gen: {combined_summary['SEER_C_gen']:.2f}")


def parse_args(default_scenario: str = "athens") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_DHW_COUNTRY),
        default=default_scenario,
        help=f"Example scenario to run. Default: {default_scenario}.",
    )
    parser.add_argument(
        "--weather-source",
        choices=["pvgis", "epw"],
        default="pvgis",
        help="Weather source used by ISO52016. Default: pvgis.",
    )
    parser.add_argument(
        "--path-weather-file",
        default=None,
        help="EPW path when --weather-source epw is selected.",
    )
    parser.add_argument(
        "--dhw-calendar-country",
        default=None,
        help="Workalendar country name for the DHW profile. Default: scenario-specific.",
    )
    parser.add_argument(
        "--dhw-design-method",
        choices=["en12831-3", "simple"],
        default=None,
        help=(
            "DHW design-load sizing mode. 'en12831-3' runs the EN 12831-3 "
            "summation-curve sizing check and feeds selected DHW storage/design "
            "flow to the downstream EN 15316 modules; 'simple' keeps the earlier "
            "fixed DHW distribution/storage assumptions. Default: en12831-3, "
            "except --calculation-path simple defaults to simple."
        ),
    )
    parser.add_argument(
        "--dhw-sizing-mode",
        choices=["check", "size_storage", "size_power", "auto"],
        default="size_storage",
        help=(
            "EN 12831-3 sizing target when --dhw-design-method en12831-3 is used. "
            "'check' only checks the configured generator/storage; 'size_storage' "
            "keeps generator power and increases storage if needed; 'size_power' "
            "keeps storage and reports required generator power; 'auto' chooses "
            "from available inputs. Default: size_storage."
        ),
    )
    parser.add_argument(
        "--calculation-path",
        choices=[
            "full",
            "full-renewables",
            "boiler-full",
            "chp-full",
            "district-full",
            "emission-distribution",
            "emission-storage",
            "distribution-storage",
            "storage-only",
            "emission-only",
            "no-cooling-storage",
            "heat-pump-cooling",
            "simple",
        ],
        default=None,
        help=(
            "Shortcut for the subsystem chain. 'full' applies EN 15316-2, "
            "EN 15316-3, EN 15316-5 and the EN 16798 cooling-side modules; "
            "'full-renewables' also applies EN 15316-4-3/4-6 solar/PV; "
            "'boiler-full', 'chp-full' and 'district-full' keep the same "
            "upstream/cooling chain but switch heating/DHW generation to "
            "EN 15316-4-1, EN 15316-4-4 or EN 15316-4-5; "
            "'emission-distribution' applies EN 15316-2 and EN 15316-3 but "
            "uses simple storage; "
            "'emission-storage', 'distribution-storage' and 'storage-only' "
            "isolate selected subsystems; 'emission-only' applies EN 15316-2 "
            "only; 'no-cooling-storage' bypasses EN 16798-15; "
            "'heat-pump-cooling' keeps the older reversible heat-pump cooling "
            "branch instead of EN 16798-13; 'simple' uses the earlier direct "
            "ISO52016/DHW loads and simple heat-pump storage losses. Default: full."
        ),
    )
    parser.add_argument(
        "--heating-generation-method",
        choices=["en15316-4-2", "en15316-4-1", "en15316-4-4", "en15316-4-5"],
        default=None,
        help=(
            "Heating/DHW generation standard. EN 15316-4-2 is the heat-pump "
            "bin method; EN 15316-4-1 is combustion boiler generation; "
            "EN 15316-4-4 is thermal-led CHP; EN 15316-4-5 is district heat. "
            "Default follows --calculation-path."
        ),
    )
    parser.add_argument(
        "--renewable-method",
        choices=["en15316-4-3-4-6", "simple"],
        default=None,
        help=(
            "Solar thermal/PV mode. EN 15316-4-3/4-6 applies the renewable "
            "calculator before heating/DHW generation and reports PV balance; "
            "'simple' bypasses renewables. Default is simple except "
            "--calculation-path full-renewables."
        ),
    )
    parser.add_argument(
        "--emission-method",
        choices=["en15316-2", "simple"],
        default=None,
        help=(
            "Space-emission calculation mode. 'en15316-2' applies emitter/control "
            "effects before heat-pump generation; 'simple' uses the direct ISO52016 "
            "loads as in the earlier example. Default: en15316-2 unless "
            "--calculation-path is set."
        ),
    )
    parser.add_argument(
        "--distribution-method",
        choices=["en15316-3", "simple"],
        default=None,
        help=(
            "Water-based distribution calculation mode. 'en15316-3' applies pipe "
            "losses and pump auxiliaries before heat-pump generation; 'simple' "
            "bypasses distribution. Default: en15316-3 unless --emission-method "
            "simple or --calculation-path is set."
        ),
    )
    parser.add_argument(
        "--storage-method",
        choices=["en15316-5", "simple"],
        default=None,
        help=(
            "Heating/DHW storage calculation mode. 'en15316-5' applies storage "
            "standing losses and storage pump auxiliaries before heat-pump "
            "generation; 'simple' keeps the earlier simplified storage losses "
            "inside the heat-pump module. Default: en15316-5 unless all upstream "
            "methods are simple or --calculation-path is set."
        ),
    )
    parser.add_argument(
        "--cooling-system-method",
        choices=["en16798-9", "simple"],
        default=None,
        help=(
            "Cooling operating-condition connector. 'en16798-9' calculates "
            "required chilled-water temperatures and flow; 'simple' uses fixed "
            "heat-pump defaults. Default: en16798-9 unless --calculation-path "
            "overrides it."
        ),
    )
    parser.add_argument(
        "--cooling-storage-method",
        choices=["en16798-15", "simple"],
        default=None,
        help=(
            "Cooling storage calculation mode. 'en16798-15' applies chilled "
            "storage heat gains and pump auxiliaries; 'simple' bypasses cooling "
            "storage. Default: en16798-15 unless --calculation-path overrides it."
        ),
    )
    parser.add_argument(
        "--cooling-generation-method",
        choices=["en16798-13", "heat-pump-simple"],
        default=None,
        help=(
            "Cooling generation calculation mode. 'en16798-13' uses the cooling "
            "generation module M4-8; 'heat-pump-simple' keeps the earlier "
            "reversible heat-pump cooling treatment. Default: en16798-13 unless "
            "--calculation-path overrides it."
        ),
    )
    parser.add_argument(
        "--performance-data-method",
        choices=["en14511-14825", "simple"],
        default=None,
        help=(
            "Heat-pump product performance data mode. 'en14511-14825' builds "
            "capacity/COP/EER maps from EN 14511 rating points and applies "
            "EN 14825 part-load correction in the generation calculators; "
            "'simple' keeps the earlier synthetic maps and no part-load "
            "correction. Default: en14511-14825, except --calculation-path "
            "simple defaults to simple."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder where CSV and visual outputs are written. "
            "Default: examples/outputs/heat_pump_15316_4_2_<scenario>; simplified "
            "paths add descriptive suffixes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_example(parse_args())
