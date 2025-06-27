import pandas as pd
from pathlib import Path
import json


def set_nested_value(d, keys, value):
    """Helper function to set value in a nested dictionary"""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = float(value)


cwd = Path(__file__).parent  # Get the current working directory

data_folder = cwd / "../src/pybuildingenergy/data"
archetypes_folder = cwd / "archetypes"
LOCATION_CLIMATE_LAT_LONG = json.load(open(data_folder / "locations.json"))

# Mapping of variable names in URBEM scorecards dataframe to their properties in the bui json data structure
urbem_df_to_bui_json_mapping = {
    "U-value of the wall": {
        "type": "opaque",
        "tilt": 90,
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
        "key_path": ["u_value"],
    },
    "Air exchange rate": {
        "key_path": ["building_parameters", "airflow_rates", "infiltration_rate"]
    },
    "Heating capacity": {
        "key_path": ["building_parameters", "system_capacities", "heating_capacity"]
    },
    "Cooling capacity": {
        "key_path": ["building_parameters", "system_capacities", "cooling_capacity"]
    },
}

with open(archetypes_folder / "default.json", "r", encoding="utf-8") as file:
    default_json = json.load(file)

df = pd.read_parquet(
    data_folder / "urbem_long_format_data.parquet"
)  # The data of all URBEM scorecards

for group in df.groupby(
    ["Location", "Climate", "Building category", "Construction period"]
):  # group is a tuple (key, data) where key is the group name and data is the data of the group, which is a dataframe containing the data of a single URBEM scorecard
    location, climate, building_category, construction_period = group[0]

    location = location.rstrip(", ")

    if location == "Lombardia":
        location = "Lombardy"

    bui_json = (
        default_json.copy()
    )  # This becomes the final BUI JSON with the values from the URBEM scorecard

    if "<" in construction_period:
        construction_period = "Before " + construction_period.split("<")[1].strip()
    elif ">" in construction_period:
        construction_period = "After " + construction_period.split(">")[1].strip()

    if "\n" in building_category:
        building_category = building_category.split("\n")[0].strip()

    bui_json["building"][
        "name"
    ] = f"{location}-{climate}-{building_category}-{construction_period}"

    # --- Latitude and longitude
    for location_data in LOCATION_CLIMATE_LAT_LONG[location]:
        if location_data["Climate zone"] == climate:
            bui_json["building"]["latitude"] = location_data["Latitude"]
            bui_json["building"]["longitude"] = location_data["Longitude"]
            break

    # Process each variable in the mapping
    for variable, props in urbem_df_to_bui_json_mapping.items():
        df_filtered = group[1].loc[
            (group[1]["Variable"] == variable) & (group[1]["Metric"] == "Mean value")
        ]

        if not df_filtered.empty and df_filtered["Value"].iloc[0] != "nan":
            value = df_filtered["Value"].iloc[0]

            # Handle building parameters (non-surface specific)
            if variable == "Air exchange rate":
                set_nested_value(bui_json, props["key_path"], value)
            elif "U-value" in variable:
                # Find the matching surface in building_surfaces
                for surface in bui_json["building_surface"]:
                    if (
                        surface["type"] == props["type"]
                        and surface["orientation"]["tilt"] == props["tilt"]
                        and (
                            props.get("sky_view_factor") is None
                            or surface["sky_view_factor"] == props["sky_view_factor"]
                        )
                    ):
                        surface["u_value"] = float(value)
            elif variable == "Total heating power":
                set_nested_value(bui_json, props["key_path"], value)
            elif variable == "Total cooling power":
                set_nested_value(bui_json, props["key_path"], value)

    # Writing the JSON file
    with open(
        archetypes_folder
        / f"{location}-{climate}-{building_category}-{construction_period}.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(bui_json, fp, ensure_ascii=False, indent=4)
