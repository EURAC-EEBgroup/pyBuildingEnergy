import pandas as pd
from pathlib import Path
import json


def set_nested_value(d, keys, value):
    """Helper function to set value in a nested dictionary"""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_surface_key(surface_type, tilt, sky_view_factor=None):
    """
    Get the key to identify a surface in the building_surfaces list
    based on type, tilt, and sky_view_factor.
    """
    if surface_type == "opaque":
        if tilt == 0:
            return "roof" if sky_view_factor == 1 else "floor"
        elif tilt == 90:
            return "wall"
    elif surface_type == "transparent" and tilt == 90:
        return "windows"
    return None


cwd = Path(__file__).parent  # Get the current working directory

data_folder = cwd / "../src/pybuildingenergy/data"
archetypes_folder = cwd / "archetypes"
LOCATION_CLIMATE_LAT_LONG = json.load(open(data_folder / "locations.json"))

# Mapping of variable names to their properties in the data structure
# Each entry contains the surface type and a function to determine the surface key
variable_mapping = {
    "U-value of the wall": {
        "type": "opaque",
        "tilt": 90,
        "sky_view_factor": None,
        "key_path": ["u_value"],
    },
    "U-value of the roof": {
        "type": "opaque",
        "tilt": 0,
        "sky_view_factor": 1,
        "key_path": ["u_value"],
    },
    "U-value of the floor": {
        "type": "opaque",
        "tilt": 0,
        "sky_view_factor": 0,
        "key_path": ["u_value"],
    },
    "U-value of the windows": {
        "type": "transparent",
        "tilt": 90,
        "sky_view_factor": None,
        "key_path": ["u_value"],
    },
    "Air exchange rate": {
        "key_path": ["building_parameters", "airflow_rates", "infiltration_rate"]
    },
}

with open(archetypes_folder / "default.json", "r", encoding="utf-8") as file:
    default_json = json.load(file)

df = pd.read_parquet(data_folder / "urbem_long_format_data.parquet")

for group in df.groupby(
    ["Location", "Climate", "Building category", "Construction period"]
):
    location, climate, building_category, construction_period = group[0]

    location = location.rstrip(", ")

    data = default_json.copy()

    if "<" in construction_period:
        construction_period = "Before " + construction_period.split("<")[1].strip()
    elif ">" in construction_period:
        construction_period = "After " + construction_period.split(">")[1].strip()

    if "\n" in building_category:
        building_category = building_category.split("\n")[0].strip()

    data["building"][
        "name"
    ] = f"{location}-{climate}-{building_category}-{construction_period}"

    # --- Latitude and longitude
    for location_data in LOCATION_CLIMATE_LAT_LONG[location]:
        if location_data["Climate zone"] == climate:
            data["building"]["latitude"] = location_data["Latitude"]
            data["building"]["longitude"] = location_data["Longitude"]
            break

    # Process each variable in the mapping
    for variable, props in variable_mapping.items():
        df_filtered = group[1].loc[
            (group[1]["Variable"] == variable) & (group[1]["Metric"] == "Mean value")
        ]

        if not df_filtered.empty and df_filtered["Value"].iloc[0] != "nan":
            value = df_filtered["Value"].iloc[0]

            # Handle building parameters (non-surface specific)
            if variable == "Air exchange rate":
                set_nested_value(data, props["key_path"], value)
                continue

            # Handle building surfaces
            surface_key = get_surface_key(
                surface_type=props["type"],
                tilt=props["tilt"],
                sky_view_factor=props.get("sky_view_factor"),
            )

            if surface_key and "building_surfaces" in data:
                # Find the matching surface in building_surfaces
                for surface in data["building_surfaces"]:
                    if (
                        surface["type"] == props["type"]
                        and surface["tilt"] == props["tilt"]
                        and (
                            "sky_view_factor" not in surface
                            or surface.get("sky_view_factor")
                            == props.get("sky_view_factor")
                        )
                        and surface.get("key") == surface_key
                    ):
                        # Set the value in the correct path within the surface
                        current = surface
                        for key in props["key_path"][:-1]:
                            current = current.setdefault(key, {})
                        current[props["key_path"][-1]] = value
                        break

    # Writing the JSON file
    with open(
        archetypes_folder
        / f"{location}-{climate}-{building_category}-{construction_period}.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
