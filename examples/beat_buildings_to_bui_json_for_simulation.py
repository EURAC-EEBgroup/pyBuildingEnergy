import json
from pathlib import Path
import pandas as pd

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

with open(data_folder / "json_building_sim.json", "r") as f:
    beat_buildings = json.load(f)

df = pd.read_parquet(
    data_folder / "urbem_long_format_data.parquet"
)  # The data of all URBEM scorecards

with open(cwd / "archetypes/default.json", "r") as f:
    default_bui = json.load(f)

beat_to_urbem_regions = {
    "Emilia-Romagna": "Lombardy",
    "Basilicata": "Sicily",
    "Calabria": "Calabria",
    "Abruzzo": "Trentino",
    "Campania": "Apulia",
}

beat_to_urbem_periods = {
    "2001-2005": "2001-2010",
    "1991-2000": "1991-2000",
    "2011-2015": "2010-2023",
    "2016-2020": "2010-2023",
    "1971-1980": "1971-1980",
    "1981-1990": "1981-1990",
    "2006-2010": "2001-2010",
}

for building in beat_buildings:
    beat_building = default_bui.copy()
    beat_building["building"]["name"] = building["building"]["name"]
    beat_building["building"]["azimuth_relative_to_true_north"] = building["building"][
        "azimuth_relative_to_true_north"
    ]
    beat_building["building"]["latitude"] = building["building"]["latitude"]
    beat_building["building"]["longitude"] = building["building"]["longitude"]
    beat_building["building"]["exposed_perimeter"] = building["building"]["perimeter"]
    beat_building["building"]["height"] = building["building"]["height"]

    # for surface in building["building_surface"]:


# print(df.Location.unique())

regions = []
provinces = []
for building in beat_buildings:
    regions.append(building["building"]["region"])
    provinces.append(building["building"]["province"])

# print(set(regions))
# print(set(provinces))

urbem_to_beat_regions = {
    "Trentino": "Abruzzo",
    "Lombardy": "Emilia-Romagna",
    "Piedmont": "Emilia-Romagna",
    "Aosta": "Abruzzo",
    "Sicily": "Basilicata",
    "Apulia": "Campania",
    "Tuscany": "Emilia-Romagna",
    "Liguria": "Campania",
    "Calabria": "Calabria",
    "Lazio": "Abruzzo",
}


ages = []
for building in beat_buildings:
    ages.append(building["building"]["age"])

print(set(ages))


types = []
for building in beat_buildings:
    types.append(building["building"]["type"])

print(set(types))
