import json
from pathlib import Path
import pandas as pd
from timezonefinder import TimezoneFinder
import requests
import datetime as dt
from pytz import timezone


def classify_hvac_capacity(
    heating_kw,
    cooling_kw,
    building_area_m2,
    estimated_apartments,
    confidence_threshold=0.7,
):
    """
    Determine if HVAC capacity data refers to a single apartment or entire building.

    Parameters:
    - heating_kw: Heating capacity in kW
    - cooling_kw: Cooling capacity in kW (optional, can be None)
    - building_area_m2: Total building floor area
    - estimated_apartments: Estimated number of apartments
    - confidence_threshold: Minimum confidence for classification (0-1)

    Returns:
    - dict with classification results and confidence scores
    """

    # Typical capacity ranges for Italian apartments (kW)
    APARTMENT_RANGES = {
        "heating": {"min": 2.5, "typical_min": 4, "typical_max": 18, "max": 30},
        "cooling": {"min": 1.5, "typical_min": 3, "typical_max": 15, "max": 20},
    }

    # Calculate building metrics
    avg_apartment_area = building_area_m2 / max(1, estimated_apartments)

    # Expected capacity per apartment based on area (simplified heuristic)
    expected_heating_per_apt = max(3, min(25, 0.08 * avg_apartment_area + 2))
    expected_cooling_per_apt = max(2, min(18, 0.06 * avg_apartment_area + 1.5))

    # Calculate expected total building capacity
    expected_building_heating = expected_heating_per_apt * estimated_apartments
    expected_building_cooling = expected_cooling_per_apt * estimated_apartments

    results = {
        "classification": None,
        "confidence": 0.0,
        "reasoning": [],
        "metrics": {
            "avg_apartment_area_m2": round(avg_apartment_area, 1),
            "expected_heating_per_apt": round(expected_heating_per_apt, 1),
            "expected_cooling_per_apt": round(expected_cooling_per_apt, 1),
            "expected_building_heating": round(expected_building_heating, 1),
            "expected_building_cooling": round(expected_building_cooling, 1),
        },
    }

    # Analyze heating capacity
    heating_scores = {"apartment": 0, "building": 0}
    heating_reasoning = []

    # Check if heating falls in apartment range
    if (
        APARTMENT_RANGES["heating"]["min"]
        <= heating_kw
        <= APARTMENT_RANGES["heating"]["max"]
    ):
        heating_scores["apartment"] += 0.6
        heating_reasoning.append(f"Heating {heating_kw}kW within apartment range")

        if (
            APARTMENT_RANGES["heating"]["typical_min"]
            <= heating_kw
            <= APARTMENT_RANGES["heating"]["typical_max"]
        ):
            heating_scores["apartment"] += 0.3
            heating_reasoning.append("Within typical apartment heating range")

    # Check if heating is too high for single apartment
    if heating_kw > APARTMENT_RANGES["heating"]["max"]:
        heating_scores["building"] += 0.8
        heating_reasoning.append(
            f"Heating {heating_kw}kW too high for single apartment"
        )

    # Compare with expected building capacity
    building_heating_ratio = heating_kw / expected_building_heating
    if 0.5 <= building_heating_ratio <= 2.0:
        heating_scores["building"] += 0.4
        heating_reasoning.append(
            f"Heating matches expected building capacity (ratio: {building_heating_ratio:.2f})"
        )

    # Compare with per-apartment expectation
    apt_heating_ratio = heating_kw / expected_heating_per_apt
    if 0.7 <= apt_heating_ratio <= 1.5:
        heating_scores["apartment"] += 0.4
        heating_reasoning.append(
            f"Heating matches expected apartment capacity (ratio: {apt_heating_ratio:.2f})"
        )

    # Analyze cooling capacity if available
    cooling_scores = {"apartment": 0, "building": 0}
    cooling_reasoning = []

    if cooling_kw is not None:
        # Check cooling ranges
        if (
            APARTMENT_RANGES["cooling"]["min"]
            <= cooling_kw
            <= APARTMENT_RANGES["cooling"]["max"]
        ):
            cooling_scores["apartment"] += 0.6
            cooling_reasoning.append(f"Cooling {cooling_kw}kW within apartment range")

            if (
                APARTMENT_RANGES["cooling"]["typical_min"]
                <= cooling_kw
                <= APARTMENT_RANGES["cooling"]["typical_max"]
            ):
                cooling_scores["apartment"] += 0.3
                cooling_reasoning.append("Within typical apartment cooling range")

        if cooling_kw > APARTMENT_RANGES["cooling"]["max"]:
            cooling_scores["building"] += 0.8
            cooling_reasoning.append(
                f"Cooling {cooling_kw}kW too high for single apartment"
            )

        # Compare with building expectation
        building_cooling_ratio = cooling_kw / expected_building_cooling
        if 0.5 <= building_cooling_ratio <= 2.0:
            cooling_scores["building"] += 0.4
            cooling_reasoning.append(
                f"Cooling matches expected building capacity (ratio: {building_cooling_ratio:.2f})"
            )

        # Compare with apartment expectation
        apt_cooling_ratio = cooling_kw / expected_cooling_per_apt
        if 0.7 <= apt_cooling_ratio <= 1.5:
            cooling_scores["apartment"] += 0.4
            cooling_reasoning.append(
                f"Cooling matches expected apartment capacity (ratio: {apt_cooling_ratio:.2f})"
            )

    # Additional heuristics
    additional_reasoning = []

    # Very small buildings are more likely to be single apartments
    if estimated_apartments <= 2:
        heating_scores["apartment"] += 0.2
        cooling_scores["apartment"] += 0.2
        additional_reasoning.append("Small building favors apartment-level data")

    # Very large buildings with small per-apartment capacity suggest building-level
    if estimated_apartments >= 10 and heating_kw / estimated_apartments < 3:
        heating_scores["building"] += 0.3
        cooling_scores["building"] += 0.3
        additional_reasoning.append(
            "Large building with low per-unit capacity suggests building-level data"
        )

    # Combine scores (weight heating more heavily as it's more reliable)
    heating_weight = 0.7 if cooling_kw is not None else 1.0
    cooling_weight = 0.3 if cooling_kw is not None else 0.0

    final_apartment_score = (
        heating_scores["apartment"] * heating_weight
        + cooling_scores["apartment"] * cooling_weight
    )
    final_building_score = (
        heating_scores["building"] * heating_weight
        + cooling_scores["building"] * cooling_weight
    )

    # Determine classification
    if final_apartment_score > final_building_score:
        if final_apartment_score >= confidence_threshold:
            results["classification"] = "apartment"
            results["confidence"] = min(1.0, final_apartment_score)
        else:
            results["classification"] = "uncertain_apartment"
            results["confidence"] = final_apartment_score
    else:
        if final_building_score >= confidence_threshold:
            results["classification"] = "building"
            results["confidence"] = min(1.0, final_building_score)
        else:
            results["classification"] = "uncertain_building"
            results["confidence"] = final_building_score

    # Compile all reasoning
    results["reasoning"] = heating_reasoning + cooling_reasoning + additional_reasoning
    results["scores"] = {
        "apartment": round(final_apartment_score, 3),
        "building": round(final_building_score, 3),
    }

    return results


# Example usage and test function
def test_classifier(building_area_m2, estimated_apartments, heating_kw, cooling_kw):
    """Test the classifier with example scenarios"""

    result = classify_hvac_capacity(
        heating_kw,
        cooling_kw,
        building_area_m2,
        estimated_apartments,
    )

    print(f"Input: {heating_kw}kW heating, {cooling_kw}kW cooling")
    print(f"Building: {building_area_m2}m², ~{estimated_apartments} apartments")
    print(f"\nClassification: {result['classification'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(
        f"Scores - Apartment: {result['scores']['apartment']:.3f}, Building: {result['scores']['building']:.3f}"
    )

    print(
        f"\nExpected per apartment: {result['metrics']['expected_heating_per_apt']}kW heating, {result['metrics']['expected_cooling_per_apt']}kW cooling"
    )
    print(
        f"Expected for building: {result['metrics']['expected_building_heating']}kW heating, {result['metrics']['expected_building_cooling']}kW cooling"
    )

    print(f"\nReasoning:")
    for reason in result["reasoning"]:
        print(f"  • {reason}")


def estimate_floors(height, interfloor_height=3.0, ground_floor_height=3.5):
    """
    Estimate number of floors from building height.

    Parameters:
    height: float or array-like, building height in meters
    interfloor_height: float, typical height per floor (default 3.0m)
    ground_floor_height: float, ground floor height (default 3.5m, often taller)

    Returns:
    int or array of ints, estimated number of floors
    """
    # Account for ground floor being typically taller
    remaining_height = height - ground_floor_height

    # Calculate additional floors beyond ground floor
    additional_floors = remaining_height / interfloor_height

    # Total floors = 1 (ground) + additional floors, rounded to nearest integer
    total_floors = round(1 + additional_floors)

    # Ensure minimum of 1 floor
    total_floors = max(total_floors, 1)

    return total_floors


def estimate_floors_simple(height, avg_floor_height=3.2):
    """
    Simple estimation using average floor height.

    Parameters:
    height: float or array-like, building height in meters
    avg_floor_height: float, average height per floor (default 3.2m)

    Returns:
    int or array of ints, estimated number of floors
    """
    floors = round(height / avg_floor_height)
    return max(floors, 1)


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


def filter_residential_buildings(df):
    """Filter out non-residential buildings from URBEM scorecard data."""
    filtered_df = df.copy()
    filtered_df = filtered_df[
        filtered_df["Building category"].str.contains(
            "residential", case=False, na=False
        )
        & ~filtered_df["Building category"].str.contains(
            "non-residential", case=False, na=False
        )
    ]
    return filtered_df


def filter_by_building_type(df, building):
    """
    Filter URBEM scorecard data based on building type.

    For single family homes, look for 'Single' in building category.
    For multi-family, look for 'multifamily' or 'multi-family'.
    """
    beat_building_type = building["building"]["type"].lower()
    is_sfh = "single" in beat_building_type or beat_building_type == "sfh"

    if is_sfh:
        filtered_df = df[
            df["Building category"].str.contains("single", case=False, na=False)
        ]
    else:
        filtered_df = df[
            df["Building category"].str.contains(
                "multi[- ]?family", case=False, na=False, regex=True
            )
        ]
    return filtered_df


# Function to split climate zones and expand rows
def expand_climate_zones(df):
    # Create a list to store the expanded rows
    expanded_rows = []

    # Define all possible dash characters that might be used
    dash_chars = [
        "-",
        "‐",
        "–",
        "—",
        "−",
    ]  # regular hyphen, non-breaking hyphen, en dash, em dash, minus

    for _, row in df.iterrows():
        climate = str(row["Climate"])
        # Check for any type of dash in Climate
        if any(dash in climate for dash in dash_chars):
            # Replace all types of dashes with regular dash and split
            for dash in dash_chars[1:]:  # Skip the first one (regular dash)
                climate = climate.replace(dash, "-")
            climate_zones = [z.strip() for z in climate.split("-") if z.strip()]

            # For each zone, create a copy of the row with the single climate zone
            for zone in climate_zones:
                new_row = row.copy()
                new_row["Climate"] = zone
                expanded_rows.append(new_row)
        else:
            # If no dash, keep the original row
            expanded_rows.append(row)

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    # Remove any rows that still contain dashes (just in case)
    expanded_df = expanded_df[
        ~expanded_df["Climate"].str.contains("|".join(dash_chars), na=False)
    ]

    return expanded_df


def map_building_to_urbem_scorecard(building, df, prioritize_period=True):
    """
    Map building data to the best matching URBEM scorecard row.

    Select is done based on priority:
       - If prioritize_period=True (default): Find closest location, then best period
       - If prioritize_period=False: Find best period, then closest location

    Args:
        building (dict): The building data from BEAT
        df (pd.DataFrame): The URBEM scorecard data
        prioritize_period (bool): If True, prioritize matching construction period over location.
                                 If False, prioritize matching location over construction period.
                                 Default is True.

    Returns:
        pd.DataFrame: The best matching row(s) from the URBEM scorecard
    """
    # Find the best match based on construction period and location
    if len(filtered_df) > 1:
        # Get the building's location
        building_lat = building["building"]["latitude"]
        building_lon = building["building"]["longitude"]

        def find_closest_region(regions, building_lat, building_lon):
            """Helper function to find the closest region."""
            region_distances = []
            for region in regions:
                cities = LOCATION_CLIMATE_LAT_LONG[region]
                for city in cities:
                    region_distances.append(
                        {
                            "region": region,
                            "distance": calculate_distance(
                                building_lat,
                                building_lon,
                                city["Latitude"],
                                city["Longitude"],
                            ),
                        }
                    )
            return (
                min(region_distances, key=lambda x: x["distance"])
                if region_distances
                else None
            )

        if prioritize_period:
            # 1. Find best matching period across all regions
            best_period = find_best_period(filtered_df, building["building"]["age"])
            period_filtered = filtered_df[
                filtered_df["Construction period"] == best_period
            ]

            # 2. Among rows with best period, find closest region
            if len(period_filtered) > 1:
                closest_region = find_closest_region(
                    period_filtered["Location"].unique(),
                    building_lat,
                    building_lon,
                )
                if closest_region:
                    return period_filtered[
                        period_filtered["Location"] == closest_region["region"]
                    ].copy()
            return period_filtered

        else:
            # 1. Find closest region first
            closest_region = find_closest_region(
                filtered_df["Location"].unique(),
                building_lat,
                building_lon,
            )

            if closest_region:
                location_filtered = filtered_df[
                    filtered_df["Location"] == closest_region["region"]
                ]

                # 2. Among rows in closest region, find best period
                if len(location_filtered) > 1:
                    best_period = find_best_period(
                        location_filtered, building["building"]["age"]
                    )
                    if best_period:
                        return location_filtered[
                            location_filtered["Construction period"] == best_period
                        ].copy()
                return location_filtered

    # Fallback: return the first row of the filtered dataframe
    if len(filtered_df) > 0:
        return filtered_df.head(1)

    # Final fallback: return the first row of the original dataframe
    return df.head(1).copy()


def get_period_years(period_str, default_start_year=1900, default_end_year=2024):
    """
    Convert period string to start and end years.

    Args:
        period_str (str or float): Period string in formats like:
                                 - "<1945" or "-1945" for periods before 1945
                                 - "2001>" or "2001-" for periods after 2001
                                 - "1946-1960" for ranges between 1946 and 1960
                                 - Single year like "1950" for that specific year
        default_start_year (int, optional): Default start year for open-ended periods.
                                          Defaults to 1900.
        default_end_year (int, optional): Default end year for open-ended periods.
                                        Defaults to 2024.

    Returns:
        tuple: (start_year, end_year) as integers
    """
    if pd.isna(period_str) or period_str == "-":
        return default_start_year, default_end_year

    period_str = str(period_str).strip()

    # Handle periods before a certain year ("<1945" or "-1945")
    if period_str.startswith("<") or period_str.startswith("-"):
        try:
            end_year = int(period_str[1:].strip())
            return default_start_year, end_year
        except (ValueError, IndexError):
            return default_start_year, default_end_year

    # Handle periods after a certain year ("2001>" or "2001-")
    if period_str.endswith(">") or period_str.endswith("-"):
        try:
            start_year = int(period_str[:-1].strip())
            return start_year, default_end_year
        except (ValueError, IndexError):
            return default_start_year, default_end_year

    # Handle ranges ("1946-1960")
    if "-" in period_str:
        try:
            start, end = period_str.split("-")
            start_year = int(start.strip()) if start.strip() else default_start_year
            end_year = int(end.strip()) if end.strip() else default_end_year
            return start_year, end_year
        except (ValueError, IndexError):
            return default_start_year, default_end_year

    # Handle single year
    try:
        year = int(period_str)
        return year, year
    except ValueError:
        return default_start_year, default_end_year


def calculate_period_score(target_start, target_end, row_start, row_end):
    """
    Calculate a score for how well a row's period matches the target.
    Lower score is better.

    Args:
        target_start (int): Start year of target period
        target_end (int): End year of target period
        row_start (int): Start year of row's period
        row_end (int): End year of row's period

    Returns:
        float: Score representing the match quality (lower is better)
    """
    if row_start is None or row_end is None:
        return float("inf")

    # If target is within row's period, it's a perfect match
    if target_start >= row_start and target_end <= row_end:
        return 0

    # If row is a '>year' period
    if row_end == float("inf"):
        return max(0, row_start - target_end)  # How many years after the period starts

    # If row is a '<year' period
    if row_end < float("inf") and row_start == 0:
        return max(0, target_start - row_end)  # How many years before the period ends

    # For regular period ranges, calculate distance to the nearest edge
    return min(
        abs(target_start - row_start)
        + abs(target_end - row_end),  # Distance between ranges
        abs(target_start - row_end),  # Distance to end of row's period
        abs(target_end - row_start),  # Distance to start of row's period
    )


def find_best_climate(climate_period_df, target_climate_zone):
    """
    Find the best matching climate zone by going backwards in the alphabet.
    For example, if target_zone is 'F', check 'F' first, then 'E', 'D', etc.

    Args:
        climate_period_df (pd.DataFrame): DataFrame with 'Climate' column
        target_climate_zone (str): Target climate zone (A-F)

    Returns:
        str: The best matching climate zone found
    """
    # Get unique climate zones from the dataframe
    available_zones = climate_period_df["Climate"].unique()

    # If target zone exists, return it
    if target_climate_zone in available_zones:
        return target_climate_zone

    # If target zone not found, go backwards in the alphabet
    # Create a list of zones in reverse order (e.g., ['F', 'E', 'D', 'C', 'B', 'A'])
    all_zones = ["F", "E", "D", "C", "B", "A"]

    try:
        # Find the position of our target zone in the sequence
        target_idx = all_zones.index(target_climate_zone)

        # Check zones after our target (warmer climates)
        for zone in all_zones[target_idx + 1 :]:
            if zone in available_zones:
                return zone

        # If no warmer climate found, check colder climates
        for zone in reversed(all_zones[:target_idx]):
            if zone in available_zones:
                return zone
    except ValueError:
        # If target_zone is not in all_zones (shouldn't happen with valid input)
        pass

    # If no match found, return the first available zone or None if empty
    return available_zones[0] if len(available_zones) > 0 else None


def find_best_period(rows, building_age):
    """
    Find the best matching construction period from a DataFrame of rows.

    Args:
        rows (pd.DataFrame): DataFrame containing a 'Construction period' column
        building_age (str or int): The building's construction period or age

    Returns:
        str: The best matching construction period
    """
    best_period_score = float("inf")
    best_period = None

    target_start, target_end = get_period_years(building_age)

    for _, row in rows.iterrows():
        row_start, row_end = get_period_years(row["Construction period"])
        period_score = calculate_period_score(
            target_start, target_end, row_start, row_end
        )

        if period_score < best_period_score:
            best_period_score = period_score
            best_period = row["Construction period"]

    return best_period


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
output_folder = cwd / "../examples/archetypes/beat_buildings"

LOCATION_CLIMATE_LAT_LONG = json.load(open(data_folder / "locations.json"))

with open(data_folder / "json_building_sim.json", "r") as f:
    beat_buildings = json.load(f)

with open(data_folder / "construction_properties.json", "r") as f:
    construction_properties = pd.DataFrame(json.load(f))

df = pd.read_parquet(
    data_folder / "urbem_long_format_data.parquet"
)  # The data of all URBEM scorecards

# Expand climate zones
df = expand_climate_zones(df)

# Mapping of variable names in URBEM scorecards dataframe to their properties in the bui json data structure
urbem_df_to_bui_json_mapping = {
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

    # Skip calculation if output JSON for this building already exists
    output_path = output_folder / f"beat_building_{building['building']['name']}.json"
    # if output_path.exists():
    #     print(f"Skipping {output_path.name} because it already exists")
    #     continue

    # Read template at each iteration to make sure no field from before is erroneously kept
    with open(data_folder / "template flat roof.json", "r", encoding="utf-8") as f:
        default_bui = json.load(f)
        # # Print the structure of default_bui to understand the building_surface format
        # print(
        #     "Default BUI structure:", json.dumps(default_bui, indent=2, ensure_ascii=False)
        # )

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

    # Filter URBEM scorecards for residential, building type, and climate zone
    filtered_df = filter_residential_buildings(df)
    filtered_df = filter_by_building_type(filtered_df, building)
    filtered_df = filtered_df[filtered_df.Climate == climate_zone]

    # Get the best matching URBEM scorecard data for this building
    urbem_scorecard = map_building_to_urbem_scorecard(
        building, filtered_df, prioritize_period=False
    )

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

    def get_construction_with_closest_u_value(
        construction_properties, target_u_value, construction_type
    ):
        """Find the construction with U-value closest to the target."""
        construction_properties = construction_properties[
            construction_properties["type"] == construction_type
        ].copy()
        construction_properties["u_value_diff"] = abs(
            construction_properties["u_value"] - target_u_value
        )
        return construction_properties.loc[
            construction_properties["u_value_diff"].idxmin()
        ]

    def get_u_value(
        urbem_scorecard,
        construction_properties,
        building,
        df,
        u_value_var,
        structure_var,
        construction_type,
    ):
        """
        Extract U-value and construction properties for a building component (wall/roof) from URBEM scorecard.

        Args:
            urbem_scorecard (pd.DataFrame): URBEM scorecard data
            construction_properties (pd.DataFrame): Construction properties data
            building (dict): Building data
            df (pd.DataFrame): URBEM dataset
            u_value_var (str): Variable name for U-value (e.g., 'U-value of the wall')
            structure_var (str): Variable name for structure type (e.g., 'External walls structure')
            construction_type (str): Type of the construction (e.g., 'wall' or 'roof')

        Returns:
            tuple: (u_value, construction_properties) where construction_properties is a dict with the closest matching construction
        """
        # Extract U-value if directly available in scorecard

        u_value_series = urbem_scorecard[
            (urbem_scorecard["Variable"] == u_value_var)
            & (urbem_scorecard["Metric"] == "Mean value")
        ].Value
        if u_value_series.empty:
            u_value = ""
        else:
            u_value = u_value_series.iloc[0]

        # Extract construction name if available
        construction_name = urbem_scorecard[
            urbem_scorecard["Variable"] == structure_var
        ].Value.iloc[0]

        if u_value != "":
            u_value = float(u_value)
            # Find closest construction matching this U-value
            construction = get_construction_with_closest_u_value(
                construction_properties, u_value, construction_type
            )
            return u_value, construction.to_dict()

        if construction_name != "no data available":
            # Get the specified construction
            construction = construction_properties[
                construction_properties["construction_name"] == construction_name
            ]
            return (
                float(construction["u_value"].iloc[0]),
                construction.iloc[0].to_dict(),
            )

        # If no direct U-value or construction data, find best matching period
        location_type_period = df.query(
            f"Variable == '{u_value_var}' and Metric == 'Mean value' and Value != ''"
        )[["Location", "Building category", "Construction period"]].drop_duplicates()

        closest_period = find_best_period(
            location_type_period, building["building"]["age"]
        )
        u_value_row = df.query(
            f"`Construction period` == @closest_period and "
            f"Variable == '{u_value_var}' and "
            "Metric == 'Mean value' and Value != ''"
        )
        u_value = float(u_value_row.Value.iloc[0])
        construction = get_construction_with_closest_u_value(
            construction_properties, u_value, construction_type
        )
        return u_value, construction.to_dict()

    # Get wall U-value and construction
    wall_u_value, wall_construction = get_u_value(
        urbem_scorecard=urbem_scorecard,
        construction_properties=construction_properties,
        building=building,
        df=filtered_df,
        u_value_var="U-value of the wall",
        structure_var="External walls structure",
        construction_type="wall",
    )

    # Get roof U-value and construction
    roof_u_value, roof_construction = get_u_value(
        urbem_scorecard=urbem_scorecard,
        construction_properties=construction_properties,
        building=building,
        df=filtered_df,
        u_value_var="U-value of the roof",
        structure_var="Roof slabs structure",
        construction_type="roof",
    )

    # Apply wall construction to vertical opaque surfaces
    print(f"Closest wall construction: {wall_construction}")

    # Process all opaque wall surfaces and collect new adiabatic surfaces
    new_adiabatic_surfaces = []
    for surf in beat_building["building_surface"]:
        if surf["type"] == "opaque" and surf["orientation"]["tilt"] == 90:
            surf["u_value"] = wall_construction["u_value"]
            surf["thermal_capacity"] = wall_construction["thermal_capacity"]
            surf["adiabatic"] = False
            surf["name"] += " - non-adiabatic"

            # Finding exposed percentage of the wall with same azimuth
            exposed_surf = next(
                (
                    s
                    for s in building["building_surface"]
                    if s["orientation"]["azimuth"] == surf["orientation"]["azimuth"]
                ),
                None,
            )

            if exposed_surf and exposed_surf.get("exposed_perc", 1.0) < 1.0:
                exposed_surf_percentage = exposed_surf["exposed_perc"]

                # Create corresponding adiabatic wall surface
                adiabatic_surf = surf.copy()
                adiabatic_surf["adiabatic"] = True
                adiabatic_surf["name"] += " - adiabatic"
                adiabatic_surf["area"] *= 1 - exposed_surf_percentage
                new_adiabatic_surfaces.append(adiabatic_surf)

                # Update the original surface
                surf["area"] *= exposed_surf_percentage

    # Add all new adiabatic surfaces
    if new_adiabatic_surfaces:
        beat_building["building_surface"].extend(new_adiabatic_surfaces)
        print(f"Added {len(new_adiabatic_surfaces)} new adiabatic surfaces")

    # Set wall thickness
    beat_building["building"]["wall_thickness"] = wall_construction["thickness"]
    print(f"Wall thickness: {wall_construction['thickness']}")

    # Apply roof construction to horizontal opaque surfaces with sky_view_factor = 1
    print(f"Closest roof construction: {roof_construction}")
    for surf in beat_building["building_surface"]:
        if surf["type"] == "opaque" and surf["sky_view_factor"] == 1:
            surf["u_value"] = roof_construction["u_value"]
            surf["thermal_capacity"] = roof_construction["thermal_capacity"]
            surf["adiabatic"] = False
            surf["name"] += " - non-adiabatic"

    def get_window_u_value(best_scorecard, all_scorecards, climate_zone, building):
        """
        Get window U-value with fallback logic.

        Args:
            best_scorecard (pd.DataFrame): The best matching scorecard
            all_scorecards (pd.DataFrame): All available scorecards
            climate_zone (str): Target climate zone
            building (dict): Building data containing age information

        Returns:
            float: The window U-value
        """
        default_u_value = 3.1  # Mean in residential URBEM scorecards
        u_value_var = "U-value of the windows"

        # Try to get U-value directly from the best scorecard
        u_value_row = best_scorecard.query(
            f"Variable == @u_value_var and Metric == 'Mean value' and Value != ''"
        )

        if not u_value_row.empty:
            return float(u_value_row.Value.iloc[0])

        # If not found, filter for residential buildings with window U-values
        filtered_df = filter_residential_buildings(all_scorecards).query(
            f"Variable == @u_value_var and Metric == 'Mean value' and Value != ''"
        )

        if not filtered_df.empty:
            # Get unique climate and construction period combinations
            climate_period_df = filtered_df[
                ["Climate", "Construction period"]
            ].drop_duplicates()

            # Find closest climate
            closest_climate = find_best_climate(climate_period_df, climate_zone)
            if closest_climate is None:
                return default_u_value

            # Filter for closest climate
            filtered_df = filtered_df[filtered_df["Climate"] == closest_climate]

            if not filtered_df.empty:
                # Find best matching period
                location_type_period = filtered_df[
                    ["Location", "Building category", "Construction period"]
                ].drop_duplicates()
                closest_period = find_best_period(
                    location_type_period, building["building"]["age"]
                )

                # Get U-value for closest period
                period_u_value = filtered_df[
                    filtered_df["Construction period"] == closest_period
                ].Value.iloc[0]

                if period_u_value:
                    return float(period_u_value)

        return default_u_value

    def get_wwr(best_scorecard, all_scorecards, orientation):
        """
        Get WWR for a specific orientation with fallback to best matching period.

        Args:
            best_scorecard (pd.DataFrame): Best matching scorecard
            all_scorecards (pd.DataFrame): All scorecards
            orientation (str): Orientation ('North', 'South', 'East', 'West')

        Returns:
            float: The WWR value
        """
        default_wwr = {
            "North": 0.15,
            "South": 0.15,
            "East": 0.15,
            "West": 0.15,
        }  # Mean of residential scorecards with existing WWR values

        # Try to get WWR directly from the scorecard
        var_name = f"WWR – {orientation} orientation"
        wwr_row = best_scorecard.query(
            f"Variable == @var_name and Metric == 'Mean value' and Value != ''"
        )

        if not wwr_row.empty:
            return float(wwr_row.Value.iloc[0])

        # If not found:
        # Filter for residential buildings with existing WWR values
        wwr_rows = filter_residential_buildings(all_scorecards).query(
            f"Variable == @var_name and Metric == 'Mean value' and Value != ''"
        )
        # Determine Climate and Construction period combinations
        climate_period_df = wwr_rows[
            ["Climate", "Construction period"]
        ].drop_duplicates()

        # Determine closest climate first
        closest_climate_zone = find_best_climate(climate_period_df, climate_zone)

        # Filter climate_period_df for closest climate zone
        climate_period_df = climate_period_df.query(f"Climate == @closest_climate_zone")

        # Now determine closest period
        closest_period = find_best_period(
            climate_period_df, building["building"]["age"]
        )

        # Filter climate_period_df for closest period
        climate_period_df = climate_period_df.query(
            "`Construction period` == @closest_period"
        )

        return float(
            wwr_rows.query(
                "Climate == @closest_climate_zone and `Construction period` == @closest_period"
            ).Value.iloc[0]
        )

    # Get WWR for each orientation
    wwr_north = get_wwr(urbem_scorecard, df, "North")
    wwr_south = get_wwr(urbem_scorecard, df, "South")
    wwr_east = get_wwr(urbem_scorecard, df, "East")
    wwr_west = get_wwr(urbem_scorecard, df, "West")

    print(
        f"Using WWR values - North: {wwr_north}, South: {wwr_south}, East: {wwr_east}, West: {wwr_west}"
    )

    # Calculate window areas based on WWR and apply to transparent surfaces
    for surface in beat_building["building_surface"]:
        if surface["type"] == "opaque" and surface["orientation"]["tilt"] == 90:
            # Find matching transparent surface (window) for this wall
            orientation = surface["orientation"]["azimuth"]
            match orientation:
                case 0:
                    wwr = wwr_north
                case 90:
                    wwr = wwr_east
                case 180:
                    wwr = wwr_south
                case 270:
                    wwr = wwr_west

            # Find corresponding window surface
            for window_surface in beat_building["building_surface"]:
                if (
                    window_surface["type"] == "transparent"
                    and window_surface["orientation"]["tilt"] == 90
                    and window_surface["orientation"]["azimuth"] == orientation
                ):
                    # Calculate window area
                    wall_area = surface["area"]
                    window_area = wall_area * wwr
                    window_surface["area"] = window_area
                    surface["area"] -= window_area
                    break

    # Get window U-value
    window_u_value = get_window_u_value(urbem_scorecard, df, climate_zone, building)
    print(f"Using window U-value: {window_u_value} W/(m²K)")

    # Estimate window g-value from U-value (linear, simplified)
    window_g_value = 0.45 + 0.08 * window_u_value
    print(f"Using window g-value: {window_g_value}")

    # Apply to all window surfaces
    for surface in beat_building["building_surface"]:
        if surface["type"] == "transparent":
            surface["u_value"] = window_u_value
            surface["g_value"] = window_g_value

    def get_air_change_rate(construction_period):
        """
        Get air change rate (1/h) based on climate zone and building construction year.

        Args:
            construction_period (str): Construction period of the building

        Returns:
            float: Air change rate in 1/h
        """
        ach = pd.DataFrame(
            {
                "Construction period": ["<1980", "1981-2000", ">2000"],
                "ACH": [1.5, 1.0, 0.6],
            }
        )
        closest_period = find_best_period(ach, construction_period)
        air_change_rate = ach.query("`Construction period` == @closest_period")[
            "ACH"
        ].iloc[0]
        return air_change_rate

    # Calculate realistic air change value since URBEM scorecards provide (too low) design values
    air_change_rate = get_air_change_rate(
        urbem_scorecard["Construction period"].iloc[0]
    )
    beat_building["building_parameters"]["airflow_rates"][
        "infiltration_rate"
    ] = air_change_rate
    print(f"Set air change rate to {air_change_rate}/h")

    def get_system_capacity(
        best_scorecard, all_scorecards, climate_zone, building, capacity_type
    ):
        """
        Get system capacity (heating or cooling) with fallback logic.

        Args:
            best_scorecard (pd.DataFrame): The best matching scorecard
            all_scorecards (pd.DataFrame): All available scorecards
            climate_zone (str): Target climate zone
            building (dict): Building data
            capacity_type (str): 'heating' or 'cooling'

        Returns:
            float: The system capacity in kW per apartment
        """
        # Default values - mean of residential URBEM scorecards
        default_values = {"heating": 47.0, "cooling": 19.0}  # kW
        var_names = {"heating": "Total heating power", "cooling": "Total cooling power"}

        var_name = var_names[capacity_type]
        default_value = default_values[capacity_type]

        # Try to get capacity directly from the best scorecard
        capacity_row = best_scorecard.query(
            f"Variable == @var_name and Metric == 'Mean value' and Value != ''"
        )

        if not capacity_row.empty:
            return float(capacity_row.Value.iloc[0])

        # If not found, filter for residential buildings with capacity values
        filtered_df = filter_residential_buildings(all_scorecards).query(
            f"Variable == @var_name and Metric == 'Mean value' and Value != ''"
        )

        if not filtered_df.empty:
            # Get unique climate and construction period combinations
            climate_period_df = filtered_df[
                ["Climate", "Construction period"]
            ].drop_duplicates()

            # Find closest climate
            closest_climate = find_best_climate(climate_period_df, climate_zone)
            if closest_climate is not None:
                # Filter for closest climate
                filtered_df = filtered_df[filtered_df["Climate"] == closest_climate]

                if not filtered_df.empty:
                    # Find best matching period
                    location_type_period = filtered_df[
                        ["Location", "Building category", "Construction period"]
                    ].drop_duplicates()
                    closest_period = find_best_period(
                        location_type_period, building["building"]["age"]
                    )

                    # Get capacity for closest period
                    period_capacity = filtered_df[
                        filtered_df["Construction period"] == closest_period
                    ].Value

                    if not period_capacity.empty:
                        return float(period_capacity.iloc[0])

        return default_value

    # Get and set system capacities
    heating_capacity = get_system_capacity(
        urbem_scorecard, df, climate_zone, building, "heating"
    )
    cooling_capacity = get_system_capacity(
        urbem_scorecard, df, climate_zone, building, "cooling"
    )

    interfloor_height = 3.0
    ground_floor_height = 3.5
    n_floors = estimate_floors(
        height=building["building"]["height"],
        interfloor_height=interfloor_height,
        ground_floor_height=ground_floor_height,
    )
    beat_building["building"]["n_floors"] = n_floors
    print(f"Estimated number of floors: {n_floors}")
    print(
        f"Effective average interfloor height: {building['building']['height'] / n_floors:.3f} (not used in building energy performance simulation, for cross-checking only)"
    )

    if building["building"]["type"] == "SFH":
        n_apartments = 1  # By definition of SFH
    else:
        area_per_apartment = 80
        n_apartments = round(building["building"]["area"] / area_per_apartment)

    print(f"Estimated number of apartments: {n_apartments}")
    print(
        f"Estimated number of apartments per floor: {n_apartments/n_floors:.3f} (not used in building energy performance simulation, for cross-checking only)"
    )
    print(
        f"Apartment size: {building['building']['area'] / n_apartments:.3f} m² (not used in building energy performance simulation, for cross-checking only)"
    )
    print(
        f"Heating capacity derived from URBEM scorecards and fallback logic: {heating_capacity:.1f} kW"
    )
    print(
        f"Cooling capacity derived from URBEM scorecards and fallback logic: {cooling_capacity:.1f} kW"
    )

    if building["building"]["type"] != "SFH":
        # Use HVAC classifier to determine whether heating and cooling capacities are per apartment or per building
        classifier_result = classify_hvac_capacity(
            heating_capacity,
            cooling_capacity,
            building["building"]["area"],
            n_apartments,
        )
    else:
        classifier_result["classification"] = "building"

    # NOTE: In the JSON file for pybuildingenergy, capacities are in W, while in the database they are in kW
    if classifier_result["classification"] in ["apartment", "uncertain_apartment"]:
        print("Heating and cooling capacities classified as at apartment level")
        beat_building["building_parameters"]["system_capacities"][
            "heating_capacity"
        ] = (heating_capacity * n_apartments) * 1e3
        beat_building["building_parameters"]["system_capacities"][
            "cooling_capacity"
        ] = (cooling_capacity * n_apartments) * 1e3
    else:
        print("Heating and cooling capacities classified as at building level")
        beat_building["building_parameters"]["system_capacities"][
            "heating_capacity"
        ] = (heating_capacity * 1e3)
        beat_building["building_parameters"]["system_capacities"][
            "cooling_capacity"
        ] = (cooling_capacity * 1e3)

    print(
        f"Heating capacity: {beat_building['building_parameters']['system_capacities']['heating_capacity'] / 1e3:.1f} kW"
    )
    print(
        f"Cooling capacity: {beat_building['building_parameters']['system_capacities']['cooling_capacity'] / 1e3:.1f} kW"
    )

    # Save BUI JSON
    # Convert to JSON string with proper encoding
    json_str = json.dumps(beat_building, indent=4, ensure_ascii=False)

    # Write with explicit UTF-8 encoding
    output_path.write_text(json_str, encoding="utf-8")
