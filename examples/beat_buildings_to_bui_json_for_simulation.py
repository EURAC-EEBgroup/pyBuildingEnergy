import json
from pathlib import Path
import pandas as pd
from timezonefinder import TimezoneFinder
import requests
import datetime as dt
from pytz import timezone


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance (in km) between two points using the Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6373.0  # Earth radius in km

    lat_rad1 = radians(lat1)
    lat_rad2 = radians(lat2)
    delta_lat_rad = radians(lat2 - lat1)
    delta_lon_rad = radians(lon2 - lon1)

    a = (
        sin(delta_lat_rad / 2) ** 2
        + cos(lat_rad1) * cos(lat_rad2) * sin(delta_lon_rad / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def map_building_to_urbem_scorecard(building, df, climate_zone):
    """
    Map building data to the best matching URBEM scorecard row.

    Filtering is done in the following order:
    1. Filter for residential buildings (excluding non-residential)
    2. For SFH: filter for 'Single' in building category
       For other types: filter for 'multifamily' or 'multi-family' in building category
    3. Filter for matching climate zone
    4. If multiple matches, select the one with closest construction period

    Args:
        building (dict): The building data from BEAT
        df (pd.DataFrame): The URBEM scorecard data
        climate_zone (str): The climate zone (A-F) determined from HDD

    Returns:
        pd.DataFrame: The best matching row(s) from the URBEM scorecard
    """
    # Make a copy of the dataframe to avoid modifying the original
    filtered_df = df.copy()

    # 1. Filter for residential buildings (case insensitive, exclude non-residential)
    filtered_df = filtered_df[
        filtered_df["Building category"].str.contains(
            "residential", case=False, na=False
        )
        & ~filtered_df["Building category"].str.contains(
            "non-residential", case=False, na=False
        )
    ]

    if len(filtered_df) == 0:
        return df.head(1)  # Fallback if no residential buildings found

    # 2. Filter based on building type (SFH vs multi-family)
    beat_building_type = building["building"]["type"].lower()
    is_sfh = "single" in beat_building_type or beat_building_type == "sfh"

    if is_sfh:
        # For single family homes, look for 'Single' in building category
        filtered_df = filtered_df[
            filtered_df["Building category"].str.contains(
                "single", case=False, na=False
            )
        ]
    else:
        # For multi-family, look for 'multifamily' or 'multi-family'
        filtered_df = filtered_df[
            filtered_df["Building category"].str.contains(
                "multi[- ]?family", case=False, na=False, regex=True
            )
        ]

    if len(filtered_df) == 0:
        return df.head(1)  # Fallback if no matching building type found

    # 3. Filter for matching climate zone
    climate_filtered = filtered_df[filtered_df["Climate"] == climate_zone]

    # If no exact climate match, keep the original filtered_df
    if len(climate_filtered) > 0:
        filtered_df = climate_filtered

    # 4. If still multiple matches, find the best match based on construction period and location
    if len(filtered_df) > 1:
        # Get the building's construction period and location
        building_lat = building["building"]["latitude"]
        building_lon = building["building"]["longitude"]

        # Extract years from construction period strings
        def get_period_years(period_str):
            try:
                # Handle formats: '1991-2000', '>2010', '<1990', '-1950'
                if period_str.startswith(">") and period_str[1:].isdigit():
                    return int(period_str[1:]), float("inf")
                elif (
                    period_str.startswith("<")
                    or period_str.startswith("-")
                    and period_str[1:].isdigit()
                ):
                    return 0, int(period_str[1:])
                else:
                    years = period_str.split("-")
                    if len(years) == 2 and years[0].isdigit() and years[1].isdigit():
                        return int(years[0]), int(years[1])
            except:
                pass
            return None, None

        # Function to calculate period match score (lower is better)
        def calculate_period_score(target_start, target_end, row_start, row_end):
            if row_start is None or row_end is None:
                return float("inf")

            # If target is within row's period, it's a perfect match
            if target_start >= row_start and target_end <= row_end:
                return 0

            # If row is a '>year' period
            if row_end == float("inf"):
                return max(
                    0, row_start - target_end
                )  # How many years after the period starts

            # If row is a '<year' period
            if row_end < float("inf") and row_start == 0:
                return max(
                    0, target_start - row_end
                )  # How many years before the period ends

            # For regular period ranges, calculate distance to the nearest edge
            return min(
                abs(target_start - row_start)
                + abs(target_end - row_end),  # Distance between ranges
                abs(target_start - row_end),  # Distance to end of row's period
                abs(target_end - row_start),  # Distance to start of row's period
            )

        # Calculate distance score
        cities_with_corresponding_climate_zone = []
        for region, cities in LOCATION_CLIMATE_LAT_LONG.items():
            for city in cities:
                if city["Climate zone"] == climate_zone:
                    city["Region"] = region
                    cities_with_corresponding_climate_zone.append(city)

        cities_with_corresponding_climate_zone = [
            {
                **city,
                "distance_to_building": calculate_distance(
                    building_lat, building_lon, city["Latitude"], city["Longitude"]
                ),
            }
            for city in cities_with_corresponding_climate_zone
        ]

        cities_ordered_by_distance_to_building = sorted(
            cities_with_corresponding_climate_zone,
            key=lambda city: city["distance_to_building"],
        )

        # Calculate scores for each row
        best_score = float("inf")
        best_matches = []

        for _, row in filtered_df.iterrows():
            # Calculate period score
            row_start, row_end = get_period_years(row["Construction period"])
            target_start, target_end = get_period_years(building["building"]["age"])
            period_score = calculate_period_score(
                target_start, target_end, row_start, row_end
            )

            # Combine scores
            period_weight = 0.8
            location_weight = 0.2
            score = (
                period_weight * period_score
                + location_weight
                * cities_ordered_by_distance_to_building[0]["distance_to_building"]
            )

            if score < best_score:
                best_score = score
                best_matches = [row]
            elif score == best_score:
                best_matches.append(row)

        # Return all rows that match the best match criteria
        if best_matches:
            # Get the best match's key attributes
            best_match = best_matches[0]
            best_attrs = {
                "Location": best_match["Location"],
                "Climate": best_match["Climate"],
                "Building category": best_match["Building category"],
                "Construction period": best_match["Construction period"],
            }

            # Filter the original dataframe to get all rows with these attributes
            mask = pd.Series(True, index=df.index)
            for col, val in best_attrs.items():
                mask &= df[col] == val

            return df[mask].copy()

    # If we couldn't find a best match, return the first matching row
    if len(filtered_df) > 0:
        best_match = filtered_df.iloc[0]
        mask = (
            (df["Location"] == best_match["Location"])
            & (df["Climate"] == best_match["Climate"])
            & (df["Building category"] == best_match["Building category"])
            & (df["Construction period"] == best_match["Construction period"])
        )
        return df[mask].copy()

    # Fallback: return the first row of the original dataframe
    return df.head(1).copy()


def calculate_HDD_from_weather_data(weather_data, method="italy"):
    """
    Calculate Heating Degree Days (HDD) from weather data following Eurostat definition:
    https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm
    """
    if method == "eurostat":
        hdd = 0
        for day in range(0, len(weather_data["T2m"]) // 24):
            daily_mean_temp = weather_data["T2m"][day * 24 : (day + 1) * 24].mean()
            if daily_mean_temp <= 15:
                hdd += 18 - daily_mean_temp
        return hdd
    elif method == "italy":
        hdd = 0
        for day in range(0, len(weather_data["T2m"]) // 24):
            daily_mean_temp = weather_data["T2m"][day * 24 : (day + 1) * 24].mean()
            if daily_mean_temp < 20:
                hdd += 20 - daily_mean_temp
        return hdd


def calculate_CDD_from_weather_data(weather_data):
    """
    Calculate Cooling Degree Days (CDD) from weather data following Eurostat definition:
    https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm
    """
    cdd = 0
    for day in range(0, len(weather_data["T2m"]) // 24):
        daily_mean_temp = weather_data["T2m"][day * 24 : (day + 1) * 24].mean()
        if daily_mean_temp >= 24:
            cdd += daily_mean_temp - 21
    return cdd


def italian_climate_zone_from_weather_file_hdd(hdd):
    """
    Determines the Italian climate zone (A-F) based on Heating Degree Days (HDD).

    Parameters:
    - hdd (float): Heating Degree Days

    Returns:
    - str: Climate zone (A, B, C, D, E, F)
    - str: Description if HDD is out of range

    Reference:
    - Presidential Decree 412/1993 (Italy)
    """
    if hdd <= 600:
        return "A"  # Very mild winters (e.g., Lampedusa)
    elif 600 < hdd <= 900:
        return "B"  # Mild winters (e.g., Palermo, Naples)
    elif 900 < hdd <= 1400:
        return "C"  # Moderate winters (e.g., Rome, Florence)
    elif 1400 < hdd <= 2100:
        return "D"  # Cold winters (e.g., Milan, Bologna)
    elif 2100 < hdd <= 3000:
        return "E"  # Very cold winters (e.g., Bolzano, Aosta)
    elif hdd > 3000:
        return "F"  # Alpine/extreme cold (e.g., high-altitude areas)
    else:
        return "Invalid HDD value (must be positive)"


def get_tmy_data(latitude, longitude):
    """
    GET weather data from PVGIS API
    """
    # Connection to PVGIS API to get weather data
    url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={latitude}&lon={longitude}&outputformat=json&browser=1"
    response = requests.request("GET", url, allow_redirects=True)
    data = response.json()
    df_weather = pd.DataFrame(data["outputs"]["tmy_hourly"])

    # Time data into UTC
    df_weather["time(UTC)"] = [
        dt.datetime.strptime(x, "%Y%m%d:%H%M") for x in df_weather["time(UTC)"]
    ]

    # Change year to 2019 before sorting by date, because the months in the tmy file are stitched together from different years
    df_weather["time(UTC)"] = df_weather["time(UTC)"].apply(
        lambda x: x.replace(year=2019)
    )

    # Order data in date ascending order
    df_weather = df_weather.sort_values(by="time(UTC)")
    df_weather.index = df_weather["time(UTC)"]
    del df_weather["time(UTC)"]

    # Elevation is not needed for the energy demand calculation, only for the PV optimization
    loc_elevation = data["inputs"]["location"]["elevation"]
    latitude_ = data["inputs"]["location"]["latitude"]
    longitude_ = data["inputs"]["location"]["longitude"]

    # TIMEZONE FINDER
    tf = TimezoneFinder()
    utcoffset_in_hours = int(
        timezone(tf.timezone_at(lng=longitude, lat=latitude))
        .localize(df_weather.index[0])
        .utcoffset()
        .total_seconds()
        / 3600.0
    )

    return loc_elevation, df_weather, utcoffset_in_hours, latitude_, longitude_


cwd = Path(__file__).parent  # Get the current working directory
data_folder = cwd / "../src/pybuildingenergy/data"

LOCATION_CLIMATE_LAT_LONG = json.load(open(data_folder / "locations.json"))

with open(data_folder / "json_building_sim.json", "r") as f:
    beat_buildings = json.load(f)

df = pd.read_parquet(
    data_folder / "urbem_long_format_data.parquet"
)  # The data of all URBEM scorecards

with open(data_folder / "template flat roof.json", "r") as f:
    default_bui = json.load(f)
    # Print the structure of default_bui to understand the building_surface format
    print("Default BUI structure:", json.dumps(default_bui, indent=2))

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

beat_to_urbem_regions = {
    "Emilia-Romagna": "Lombardy",
    "Basilicata": "Sicily",
    "Calabria": "Calabria",
    "Abruzzo": "Trentino",
    "Campania": "Apulia",
}

for building in beat_buildings:
    # Get weather data from PVGIS
    elevation, weather_data, utcoffset_in_hours, latitude, longitude = get_tmy_data(
        building["building"]["latitude"], building["building"]["longitude"]
    )

    if (
        latitude != building["building"]["latitude"]
        or longitude != building["building"]["longitude"]
    ):
        print(
            "Warning: for building "
            + building["building"]["name"]
            + ", latitude and longitude from input JSON and PVGIS do not match"
        )

    # Calculate HDD and determine climate zone
    hdd = calculate_HDD_from_weather_data(weather_data)
    climate_zone = italian_climate_zone_from_weather_file_hdd(hdd)

    # Get the best matching URBEM scorecard data for this building
    urbem_scorecard = map_building_to_urbem_scorecard(building, df, climate_zone)

    # Print the matched scorecard for debugging
    print(f"\nMatched URBEM scorecard for building {building['building']['name']}:")
    print(
        f"- Location - BEAT: {building['building']['region']} URBEM: {urbem_scorecard['Location'].iloc[0]}"
    )
    print(
        f"- Climate Zone - BEAT: {climate_zone} URBEM: {urbem_scorecard['Climate'].iloc[0]}"
    )
    print(
        f"- Building Category - BEAT: {building['building']['type']} URBEM: {urbem_scorecard['Building category'].iloc[0]}"
    )
    print(
        f"- Construction Period - BEAT: {building['building']['age']} URBEM: {urbem_scorecard['Construction period'].iloc[0]}"
    )

    beat_building = default_bui.copy()
    beat_building["building"]["name"] = building["building"]["name"]
    beat_building["building"]["azimuth_relative_to_true_north"] = building["building"][
        "azimuth_relative_to_true_north"
    ]
    beat_building["building"]["latitude"] = building["building"]["latitude"]
    beat_building["building"]["longitude"] = building["building"]["longitude"]
    beat_building["building"]["exposed_perimeter"] = building["building"]["perimeter"]
    beat_building["building"]["height"] = building["building"]["height"]

    # Area mapping
    # Key: (azimuth, tilt) for walls, or surface name for slab to ground and roof
    area_mapping = {}
    for surface in building["building_surface"]:
        if "orientation" in surface:
            key = (surface["orientation"]["azimuth"], surface["orientation"]["tilt"])
            area_mapping[key] = surface["area"]
        else:  # Slab has only keys "name" and "area"
            area_mapping["slab"] = surface["area"]

    for surface in beat_building["building_surface"]:
        if surface["type"] == "opaque":
            if surface["orientation"]["tilt"] != 0:
                key = (
                    surface["orientation"]["azimuth"],
                    surface["orientation"]["tilt"],
                )
                surface["area"] = area_mapping[key]
            else:
                surface["area"] = area_mapping["slab"]
        elif surface["type"] == "transparent":
            key = (
                surface["orientation"]["azimuth"],
                surface["orientation"]["tilt"],
            )
            surface["area"] = area_mapping[key]

    # Extract relevant parameters from the matched URBEM scorecard
    # We'll use these to update the building parameters
    building_params = {}
    for _, row in urbem_scorecard.iterrows():
        if row["Metric"] == "Value":  # We only want the actual values, not statistics
            param_name = row["Variable"]
            param_value = row["Value"]

            # Map the parameter to the appropriate location in the building JSON
            if param_name in urbem_df_to_bui_json_mapping:
                mapping = urbem_df_to_bui_json_mapping[param_name]
                current = beat_building

                # Navigate the path in the building JSON
                for key in mapping["key_path"][:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                # Set the final value
                final_key = mapping["key_path"][-1]
                current[final_key] = param_value

    region = beat_to_urbem_regions[building["building"]["region"]]
    province = building["building"]["province"]
    beat_building_type = building["building"]["type"]


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
