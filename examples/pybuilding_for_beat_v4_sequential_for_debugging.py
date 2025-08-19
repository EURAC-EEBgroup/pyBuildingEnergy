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


# def get_building_surface(
#     json_data,
#     type_surface="opaque",
#     adiabatic=False,
#     sky_view_factor=0.0,
#     orientation_tilt=0,
#     orientation_azimuth=0,
# ):
#     """Get building surface area"""
#     return next(
#         (
#             d
#             for d in json_data["building_surface"]
#             if d["type"] == type_surface
#             and d["sky_view_factor"] == sky_view_factor
#             and d["orientation"]["tilt"] == orientation_tilt
#             and d["orientation"]["azimuth"] == orientation_azimuth
#         ),
#         None,
#     )


def process_building(building_archetype, output_dir="results"):
    """Process a single building archetype and save results"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process the building
        (
            hourly_sim,
            annual_results_df,
        ) = ISO52016.Temperature_and_Energy_needs_calculation(
            building_archetype,
            weather_source="pvgis",
        )

        # Generate unique filenames for each building
        building_id = building_archetype.get("_id", "unknown")
        building_name = building_archetype["building"].get("name", "unknown")
        hourly_file = os.path.join(output_dir, f"hourly_sim_{building_name}.csv")
        annual_file = os.path.join(output_dir, f"annual_results_{building_name}.csv")

        # Save results with unique filenames
        hourly_sim.to_csv(hourly_file)
        annual_results_df.to_csv(annual_file, index=False)

        # Calculate metrics
        heating_kWh = hourly_sim[hourly_sim["Q_HC"] > 0]["Q_HC"].sum() / 1000
        cooling_kWh = -hourly_sim[hourly_sim["Q_HC"] < 0]["Q_HC"].sum() / 1000
        treated_floor_area = building_archetype["building"]["treated_floor_area"]
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


def main():
    # Database and collection names
    db_name = "buildings_db"
    collection_name = "buildings"

    # Connect to MongoDB to initialize the status field
    client = MongoClient("localhost", 27017)
    db = client[db_name]
    collection = db[collection_name]

    # Reset the status field for all documents to "unprocessed"
    collection.update_many(
        {"status": {"$exists": True}}, {"$set": {"status": "unprocessed"}}
    )

    # Initialize the status field for documents that don't have it
    collection.update_many(
        {"status": {"$exists": False}}, {"$set": {"status": "unprocessed"}}
    )

    # Create output directory
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize worker (for database connection)
    worker_init(db_name, collection_name)

    # Process buildings one by one (debug mode)
    results = []
    while True:
        # Find and update the document to mark it as processing
        building_archetype = collection.find_one_and_update(
            {"status": "unprocessed"}, {"$set": {"status": "processing"}}
        )
        if building_archetype is None:
            break

        # Internal gains calculation
        # Initialize profiles with zeros
        internal_gains_weekday_profile = [0] * 24
        internal_gains_weekend_profile = [0] * 24

        # Calculate combined profiles
        for gain in building_archetype["building_parameters"]["internal_gains"]:
            for i in range(24):
                internal_gains_weekday_profile[i] += (
                    gain["full_load"] * gain["weekday"][i]
                )
                internal_gains_weekend_profile[i] += (
                    gain["full_load"] * gain["weekend"][i]
                )
        # Update building archetype with internal gains profiles
        building_archetype["building_parameters"]["internal_gains_total"] = {
            "weekday": internal_gains_weekday_profile,
            "weekend": internal_gains_weekend_profile,
        }

        print(f"Processing building: {building_archetype['building']['name']}")
        if building_archetype["building"]["name"] != 1288437:
            continue
        result = process_building(building_archetype)
        results.append(result)
        # Mark the document as processed
        collection.update_one(
            {"_id": building_archetype["_id"]}, {"$set": {"status": "processed"}}
        )

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
