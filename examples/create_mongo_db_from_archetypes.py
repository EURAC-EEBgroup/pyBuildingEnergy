from pymongo import MongoClient
import json
import os


# Connect to MongoDB
client = MongoClient("localhost", 27017)
db = client.buildings_db
collection = db.buildings


def insert_building(building):
    collection.insert_one(building)


# Load building archetypes
archetypes_folder = "examples/archetypes"
for root, _, files in os.walk(archetypes_folder):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                building = json.load(f)
                insert_building(building)

# Verify the data has been inserted
loaded_buildings = list(collection.find())
for building in loaded_buildings:
    print(building)

client.close()
