import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
import os

# Add the project root to Python path (absolute path)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Prioritize this path

from src.pybuildingenergy.source.utils import ISO52016
from src.pybuildingenergy.data.building_archetype import Buildings
from src.pybuildingenergy.source.graphs import Graphs_and_report
import json


def get_building_surface(
    json_data,
    type_surface="opaque",
    sky_view_factor=0.0,
    orientation_tilt=0,
    orientation_azimuth=0,
):
    """Get building surface area"""
    return next(
        (
            d
            for d in json_data["building_surface"]
            if d["type"] == type_surface
            and d["sky_view_factor"] == sky_view_factor
            and d["orientation"]["tilt"] == orientation_tilt
            and d["orientation"]["azimuth"] == orientation_azimuth
        ),
        None,
    )


def process_building(building_archetype, output_dir="results"):
    """Process a single building archetype and save results"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        height = building_archetype["building"]["height"]
        floor_to_floor_height = 3
        number_of_floors = round(height / floor_to_floor_height)

        roof = get_building_surface(
            building_archetype,
            type_surface="opaque",
            sky_view_factor=1.0,
            orientation_tilt=0,
            orientation_azimuth=0,
        )

        slab_to_ground = get_building_surface(
            building_archetype,
            type_surface="opaque",
            sky_view_factor=0.0,
            orientation_tilt=0,
            orientation_azimuth=0,
        )

        opaque_surface_north = get_building_surface(
            building_archetype,
            type_surface="opaque",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=0,
        )

        opaque_surface_south = get_building_surface(
            building_archetype,
            type_surface="opaque",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=180,
        )

        opaque_surface_east = get_building_surface(
            building_archetype,
            type_surface="opaque",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=90,
        )

        opaque_surface_west = get_building_surface(
            building_archetype,
            type_surface="opaque",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=270,
        )

        transparent_surface_north = get_building_surface(
            building_archetype,
            type_surface="transparent",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=0,
        )

        transparent_surface_south = get_building_surface(
            building_archetype,
            type_surface="transparent",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=180,
        )

        transparent_surface_east = get_building_surface(
            building_archetype,
            type_surface="transparent",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=90,
        )

        transparent_surface_west = get_building_surface(
            building_archetype,
            type_surface="transparent",
            sky_view_factor=0.5,
            orientation_tilt=90,
            orientation_azimuth=270,
        )

        volume = number_of_floors * floor_to_floor_height * slab_to_ground["area"]

        surface_envelope = (
            roof["area"]
            + slab_to_ground["area"]
            + opaque_surface_north["area"]
            + opaque_surface_south["area"]
            + opaque_surface_east["area"]
            + opaque_surface_west["area"]
            + transparent_surface_north["area"]
            + transparent_surface_south["area"]
            + transparent_surface_east["area"]
            + transparent_surface_west["area"]
        )

        side = (opaque_surface_east["area"] + transparent_surface_east["area"]) / height

        thermal_transmittances = [
            opaque_surface_north["u_value"],
            opaque_surface_south["u_value"],
            opaque_surface_east["u_value"],
            opaque_surface_west["u_value"],
            slab_to_ground["u_value"],
            roof["u_value"],
            transparent_surface_north["u_value"],
            transparent_surface_south["u_value"],
            transparent_surface_east["u_value"],
            transparent_surface_west["u_value"],
        ]

        thermal_resistances = [
            1 / thermal_transmittance
            for thermal_transmittance in thermal_transmittances
        ]

        heat_capacities = [
            opaque_surface_north["thermal_capacity"],
            opaque_surface_south["thermal_capacity"],
            opaque_surface_east["thermal_capacity"],
            opaque_surface_west["thermal_capacity"],
            slab_to_ground["thermal_capacity"],
            roof["thermal_capacity"],
        ] + [0] * 4

        g_values = [0] * 6 + [
            transparent_surface_north["g_value"],
            transparent_surface_south["g_value"],
            transparent_surface_east["g_value"],
            transparent_surface_west["g_value"],
        ]

        treated_floor_area = slab_to_ground["area"] * number_of_floors

        infiltration_rate_in_ach = building_archetype["building_parameters"][
            "airflow_rates"
        ]["infiltration_rate"]

        ventilation_rate_extra_in_ach = building_archetype["building_parameters"][
            "airflow_rates"
        ]["ventilation_rate_extra"]

        infiltration_rate_in_m3_per_sqm_hr = (
            infiltration_rate_in_ach * volume / treated_floor_area
        )
        ventilation_rate_extra_in_m3_per_sqm_hr = (
            ventilation_rate_extra_in_ach * volume / treated_floor_area
        )

        # Create BuildingArchetype from BUI_JSON
        BUI = Buildings(
            latitude=building_archetype["building"]["latitude"],
            longitude=building_archetype["building"]["longitude"],
            exposed_perimeter=building_archetype["building"]["exposed_perimeter"],
            area=treated_floor_area,
            number_of_floor=number_of_floors,
            height=height,
            volume=volume,
            slab_on_ground=slab_to_ground["area"],
            wall_thickness=building_archetype["building"]["wall_thickness"],
            coldest_month=building_archetype["building_parameters"][
                "climate_parameters"
            ]["coldest_month"],
            surface_envelope=surface_envelope,
            surface_envelope_model=surface_envelope,
            side=side,
            heating_mode=True,  # True or False if heating system is
            cooling_mode=True,
            heating_setpoint=building_archetype["building_parameters"][
                "temperature_setpoints"
            ][
                "heating_setpoint"
            ],  # Heating set-point in °C
            cooling_setpoint=building_archetype["building_parameters"][
                "temperature_setpoints"
            ][
                "cooling_setpoint"
            ],  # Cooling set-point in °C
            heating_setback=building_archetype["building_parameters"][
                "temperature_setpoints"
            ][
                "heating_setback"
            ],  # Heating set-back in °C
            cooling_setback=building_archetype["building_parameters"][
                "temperature_setpoints"
            ][
                "cooling_setback"
            ],  # Cooling set-back in °C
            power_heating_max=building_archetype["building_parameters"][
                "system_capacities"
            ][
                "heating_capacity"
            ],  # Max Power of the heating system
            power_cooling_max=building_archetype["building_parameters"][
                "system_capacities"
            ][
                "cooling_capacity"
            ],  # Max Power of the cooling system
            air_change_rate_base_value=infiltration_rate_in_m3_per_sqm_hr,
            air_change_rate_extra=ventilation_rate_extra_in_m3_per_sqm_hr,
            internal_gains_base_value=building_archetype["building_parameters"][
                "internal_gains"
            ]["unoccupied"],
            internal_gains_extra=building_archetype["building_parameters"][
                "internal_gains"
            ]["occupied_extra"],
            thermal_bridge_heat=building_archetype["building_parameters"][
                "construction"
            ][
                "thermal_bridges"
            ],  # value of thermal bridges
            thermal_resistance_floor=1 / slab_to_ground["u_value"],
            area_elements=list(
                (
                    opaque_surface_north["area"] + transparent_surface_north["area"],
                    opaque_surface_south["area"] + transparent_surface_south["area"],
                    opaque_surface_east["area"] + transparent_surface_east["area"],
                    opaque_surface_west["area"] + transparent_surface_west["area"],
                    slab_to_ground["area"],
                    roof["area"],
                    transparent_surface_north["area"],
                    transparent_surface_south["area"],
                    transparent_surface_east["area"],
                    transparent_surface_west["area"],
                )
            ),  # Areas of each facade element in this specific order: north, south, east, west, slab to ground, roof, north transparent, south transparent, east transparent, west transparent
            transmittance_U_elements=thermal_transmittances,
            thermal_resistance_R_elements=thermal_resistances,
            thermal_capacity_elements=heat_capacities,
            g_factor_windows=g_values,
            occ_level_wd=building_archetype["building_parameters"][
                "occupancy_profiles"
            ]["weekday"],
            occ_level_we=building_archetype["building_parameters"][
                "occupancy_profiles"
            ]["weekend"],
            comf_level_wd=building_archetype["building_parameters"][
                "occupancy_profiles"
            ]["weekday"],
            comf_level_we=building_archetype["building_parameters"][
                "occupancy_profiles"
            ]["weekend"],
        )

        # Process the building
        (
            hourly_sim,
            annual_results_df,
        ) = ISO52016().Temperature_and_Energy_needs_calculation(
            BUI,
            weather_source="pvgis",
        )

        # Generate unique filenames for each building
        building_id = building_archetype.get("_id", "unknown")
        building_name = building_archetype["building"].get("name", "unknown")
        hourly_file = os.path.join(
            output_dir, f"hourly_sim_{building_id}_{building_name}.csv"
        )
        annual_file = os.path.join(
            output_dir, f"annual_results_{building_id}_{building_name}.csv"
        )

        # Save results with unique filenames
        hourly_sim.to_csv(hourly_file)
        annual_results_df.to_csv(annual_file, index=False)

        # Calculate metrics
        heating_kWh = hourly_sim[hourly_sim["Q_HC"] > 0]["Q_HC"].sum() / 1000
        cooling_kWh = -hourly_sim[hourly_sim["Q_HC"] < 0]["Q_HC"].sum() / 1000
        treated_floor_area = slab_to_ground["area"] * number_of_floors
        heating_kWh_per_sqm = heating_kWh / treated_floor_area
        cooling_kWh_per_sqm = cooling_kWh / treated_floor_area

        return {
            "building_id": building_id,
            "building_name": building_name,
            "heating_kWh": heating_kWh,
            "cooling_kWh": cooling_kWh,
            "heating_kWh_per_sqm": heating_kWh_per_sqm,
            "cooling_kWh_per_sqm": cooling_kWh_per_sqm,
            "status": "success",
        }

    except Exception as e:
        return {
            "building_id": building_archetype.get("_id", "unknown"),
            "building_name": building_archetype["building"].get("name", "unknown"),
            "error": str(e),
            "status": "failed",
        }


def worker_init(db_name, collection_name):
    global db, collection
    client = MongoClient("localhost", 27017)
    db = client[db_name]
    collection = db[collection_name]


def worker_task(_):
    results = []
    while True:
        # Find and update the document to mark it as processing
        building_archetype = collection.find_one_and_update(
            {"status": "unprocessed"}, {"$set": {"status": "processing"}}
        )
        if building_archetype is None:
            break
        result = process_building(building_archetype)
        results.append(result)
        # Mark the document as processed
        collection.update_one(
            {"_id": building_archetype["_id"]}, {"$set": {"status": "processed"}}
        )
    return results


def main():
    # Database and collection names
    db_name = "buildings_db"
    collection_name = "buildings"

    # Connect to MongoDB to initialize the status field
    client = MongoClient("localhost", 27017)
    db = client[db_name]
    collection = db[collection_name]

    # Reset the status field for all documents to "unprocessed"
    # collection.update_many(
    #     {"status": {"$exists": True}}, {"$set": {"status": "unprocessed"}}
    # )

    # Initialize the status field for documents that don't have it
    collection.update_many(
        {"status": {"$exists": False}}, {"$set": {"status": "unprocessed"}}
    )

    # Determine number of processes to use (leave one core free)
    num_processes = max(1, cpu_count() - 1)

    # Create output directory
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the worker pool
    with Pool(
        processes=num_processes,
        initializer=worker_init,
        initargs=(db_name, collection_name),
    ) as pool:
        # Use tqdm for progress bar
        results = list(pool.imap(worker_task, range(num_processes)))

    # Flatten the results list
    results = [result for sublist in results for result in sublist]

    # Save summary of results
    summary_df = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, "simulation_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # Remove the "status" field from each document after processing
    # for result in results:
    #     collection.update_one(
    #         {"_id": result["building_id"]}, {"$unset": {"status": ""}}
    #     )

    # Print summary
    success_count = (summary_df["status"] == "success").sum()
    failed_count = len(summary_df) - success_count

    print(f"\nSummary:")
    print(f"Total buildings processed: {len(summary_df)}")
    print(f"Buildings processed successfully: {success_count}")
    print(f"Buildings failed: {failed_count}")

    # Close MongoDB connection
    client.close()


if __name__ == "__main__":
    main()
