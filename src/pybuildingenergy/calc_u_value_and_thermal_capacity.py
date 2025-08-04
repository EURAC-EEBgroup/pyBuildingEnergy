import json


def calculate_u_value_and_thermal_capacity(data, R_si=0.13, R_se=0.04):
    # Extract materials and constructions from the JSON data
    materials = {material["name"]: material for material in data["materials"]}
    constructions = data["constructions"]

    results = []

    for construction in constructions:
        construction_name = construction["name"]
        layers = construction["layers"]

        total_thermal_resistance = 0.0
        total_thermal_capacity = 0.0

        for layer_name in layers:
            if layer_name in materials:
                layer = materials[layer_name]
                if "u_factor" in layer:  # if it is a window
                    u_value = layer["u_factor"]
                    total_thermal_capacity = 0
                else:
                    # Calculate U-value with air resistances
                    thickness = layer["thickness"]
                    conductivity = layer["conductivity"]
                    density = layer["density"]
                    specific_heat = layer["specific_heat"]

                    # Calculate thermal resistance for the layer
                    thermal_resistance = thickness / conductivity
                    total_thermal_resistance += thermal_resistance

                    # Calculate thermal capacity for the layer
                    thermal_capacity = thickness * density * specific_heat
                    total_thermal_capacity += thermal_capacity

                    # Calculate Conductance (U-value without air resistances)
                    conductance = (
                        1 / total_thermal_resistance
                        if total_thermal_resistance != 0
                        else 0
                    )

                    # Calculate U-value with air resistances
                    total_resistance_with_air = total_thermal_resistance + R_si + R_se
                    u_value = (
                        1 / total_resistance_with_air
                        if total_resistance_with_air != 0
                        else 0
                    )
            else:
                print(
                    f"Warning: layer {layer_name} for construction {construction_name} not found in materials"
                )

        results.append(
            {
                "construction_name": construction_name,
                "u_value": u_value,
                "thermal_capacity": total_thermal_capacity,
            }
        )

    return results


# Example JSON data
# data = {
#     "materials": [
#         {
#             "name": "25mm Stucco",
#             "roughness": "Smooth",
#             "thickness": 0.0254,
#             "conductivity": 0.71951849592325,
#             "density": 1856.00424372353,
#             "specific_heat": 839.458381452284,
#             "thermal_absorptance": 0.9,
#             "solar_absorptance": 0.7,
#             "visible_absorptance": 0.7
#         },
#         {
#             "name": "Clay brick35",
#             "roughness": "MediumRough",
#             "thickness": 0.35,
#             "conductivity": 0.85,
#             "density": 1600,
#             "specific_heat": 840,
#             "thermal_absorptance": 0.9,
#             "solar_absorptance": 0.7,
#             "visible_absorptance": 0.7
#         },
#         {
#             "name": "Gypsum",
#             "roughness": "Smooth",
#             "thickness": 0.0127,
#             "conductivity": 0.16,
#             "density": 800,
#             "specific_heat": 1090,
#             "thermal_absorptance": 0.9,
#             "solar_absorptance": 0.7,
#             "visible_absorptance": 0.7
#         }
#     ],
#     "constructions": [
#         {
#             "name": "WALL4",
#             "layers": [
#                 "25mm Stucco",
#                 "Clay brick35",
#                 "Gypsum"
#             ]
#         }
#     ]
# }

data = json.load(open("src/pybuildingenergy/data/materials.json"))

# Calculate U-value and thermal capacity
results = calculate_u_value_and_thermal_capacity(data=data, R_si=0.13, R_se=0.04)

# Save construction properties
with open("src/pybuildingenergy/data/construction_properties.json", "w") as f:
    json.dump(results, f, indent=4)
