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

    # Get all unprocessed buildings
    unprocessed_buildings = list(collection.find({"status": "unprocessed"}))
    print(f"Total unprocessed buildings: {len(unprocessed_buildings)}")

    for building in unprocessed_buildings:
        # Internal gains calculation
        internal_gains_weekday_profile = [0] * 24
        internal_gains_weekend_profile = [0] * 24

        # Calculate combined profiles
        for gain in building["building_parameters"]["internal_gains"]:
            for i in range(24):
                internal_gains_weekday_profile[i] += (
                    gain["full_load"] * gain["weekday"][i]
                )
                internal_gains_weekend_profile[i] += (
                    gain["full_load"] * gain["weekend"][i]
                )

        # Update building with internal gains profiles
        building["building_parameters"]["internal_gains_total"] = {
            "weekday": internal_gains_weekday_profile,
            "weekend": internal_gains_weekend_profile,
        }

    # Mark all buildings as processing
    collection.update_many(
        {"_id": {"$in": [b["_id"] for b in unprocessed_buildings]}},
        {"$set": {"status": "processing"}},
    )

    # Process buildings
    num_workers = cpu_count()
    print(f"Starting processing with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_building, unprocessed_buildings),
                total=len(unprocessed_buildings),
                desc="Processing buildings",
            )
        )

    # Save summary of results
    summary_df = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, "simulation_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # Update status in the database
    for result in results:
        if result["status"] == "success":
            collection.update_one(
                {"_id": result["building_id"]}, {"$set": {"status": "processed"}}
            )
        else:
            collection.update_one(
                {"_id": result["building_id"]},
                {"$set": {"status": "failed", "error": result.get("error", "")}},
            )

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
