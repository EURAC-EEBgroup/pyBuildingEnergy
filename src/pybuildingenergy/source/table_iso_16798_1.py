import pandas as pd
import numpy as np

# Schedule from hour 1 to 24
# WORKDAYS
occupants_schedule_workdays = {
    "Office": [0,0,0,0,0,0,0,0.2,0.6,0.6, 0.7, 0.7, 0.4, 0.6, 0.7, 0.7, 0.6, 0.2, 0, 0, 0,  0, 0, 0],
    "School": [0,0,0,0,0,0,0,0,0.6, 0.7, 0.6, 0.4, 0.3, 0.7, 0.6, 0.4, 0.2, 0,0,0,0,0,0,0],
    "Residential_apartment": [1,1,1,1,1,1,0.5,0.5,0.5,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1,1],
    "Kindergarten": [0,0,0,0,0,0,0,0.4,0.8,0.8,0.3,0.3,0.8,0.1,0.1,0.4,0.3, 0.3,0.3, 0, 0, 0, 0, 0 ],
    "Department_store": [0,0,0,0,0,0,0,0,0.1,0.3,0.3,0.7,0.6,0.5,0.6,0.6,0.9,0.9,1,0.9,0.7,0,0,0],
    "Residential_detached_house": [1,1,1,1,1,1,0.5,0.5,0.5,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1,1],
}
appliances_schedule_workdays = {
    "Office": [0,0,0,0,0,0,0,0.2,0.6,0.6, 0.7, 0.7, 0.4, 0.6, 0.7, 0.7, 0.6, 0.2, 0, 0, 0,  0, 0, 0],
    "School": [0,0,0,0,0,0,0,0,0.6, 0.7, 0.6, 0.4, 0.3, 0.7, 0.6, 0.4, 0.2, 0,0,0,0,0,0,0],
    "Residential_apartment": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5, 0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
    "Kindergarten": [0,0,0,0,0,0,0,0.4,0.8,0.8,0.3,0.3,0.8,0.1,0.1,0.4,0.3, 0.3,0.3, 0, 0, 0, 0, 0 ],
    "Department_store": [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    "Residential_detached_house": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5, 0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
}

lighting_schedule_workdays = {
    "Office": [0,0,0,0,0,0,0,0.2,0.6,0.6, 0.7, 0.7, 0.4, 0.6, 0.7, 0.7, 0.6, 0.2, 0, 0, 0,  0, 0, 0],
    "School": [0,0,0,0,0,0,0,0,0.6, 0.7, 0.6, 0.4, 0.3, 0.7, 0.6, 0.4, 0.2, 0,0,0,0,0,0,0],
    "Residential_apartment": [0,0,0,0,0,0,0.15,0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15 ],
    "Kindergarten": [0,0,0,0,0,0,0,0.4,0.8,0.8,0.3,0.3,0.8,0.1,0.1,0.4,0.3, 0.3,0.3, 0, 0, 0, 0, 0 ],
    "Department_store": [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    "Residential_detached_house": [0,0,0,0,0,0,0.15,0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15 ],
}

# WEEKEND
occupants_schedule_weekend = {
    "Office": [0]*24,
    "School": [0]*24,
    "Residential_apartment": [1,1,1,1,1,1,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1,1],
    "Kindergarten": [0]*24,
    "Department_store": [0,0,0,0,0,0,0,0,0.1,0.3,0.6,0.9,1,0.9,0.7,0.5,0.3,0.3,0.45,0.45,0.45,0,0,0],
    "Residential_detached_house": [1,1,1,1,1,1,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1,1],
}
appliances_schedule_weekend = {
    "Office": [0]*24,
    "School": [0]*24,
    "Residential_apartment": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5, 0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
    "Kindergarten": [0]*24,
    "Department_store": [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    "Residential_detached_house": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5, 0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
}
lighting_schedule_weekend = {
    "Office": [0]*24,
    "School": [0]*24,
    "Residential_apartment": [0,0,0,0,0,0,0.15,0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15 ],
    "Kindergarten": [0]*24,
    "Department_store": [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    "Residential_detached_house": [0,0,0,0,0,0,0.15,0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15 ],
}

# ================================================================================================
# Internal gains in W/m2, flow rate, person quantity, min, max setpoint, etc. 
internal_gains_occupants = {
    "Office": {
        "occupants_quantity": 17, # m2/person
        "occupants": 7,
        "appliances": 12,
        "lighting": 0,
        "mositure_production": 3.53, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 20, #°C
        "min_t_op_in_cooling": 26, #°C
        "ventilation_rate(min)": 0.8, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": np.nan,
    },
    "School": {
        "occupants_quantity": 5.4, # m2/person
        "occupants": 21.7, # W/m2
        "appliances": 8, # W/m2
        "lighting": 0, # W/m2
        "mositure_production": 11.11, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 20, #°C
        "min_t_op_in_cooling": 26, #°C
        "ventilation_rate(min)": 3.8, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": 100, # l/(m2 year)
    },
    "Residential_apartment": {
        "occupants_quantity": 28.3, # m2/person
        "occupants": 4.2, # W/m2
        "appliances": 3, # W/m2
        "lighting": np.nan, # W/m2
        "mositure_production": 2.12, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 20, #°C
        "min_t_op_in_cooling": 26, #°C
        "ventilation_rate(min)": 0.5, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": 100, # l/(m2 year)
    },
    "Residential_detached_house": {
        "occupants_quantity": 42.5, # m2/person
        "occupants": 2.8, # W/m2
        "appliances": 2.4, # W/m2
        "lighting": np.nan, # W/m2
        "mositure_production": 1.41, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 20, #°C
        "min_t_op_in_cooling": 26, #°C
        "ventilation_rate(min)": 0.5, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": 100, # l/(m2 year)
    },
    "Kindergarten": {
        "occupants_quantity": 3.8, # m2/person
        "occupants": 33.3, # W/m2
        "appliances": 4, # W/m2
        "lighting": np.nan, # W/m2
        "mositure_production": 15.79, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 17.5, #°C
        "min_t_op_in_cooling": 25.5, #°C
        "ventilation_rate(min)": 4.5, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": 100, # l/(m2 year)
    },
    "Department_store": {
        "occupants_quantity": 3.8, # m2/person
        "occupants": 9.3, # W/m2
        "appliances": 1, # W/m2
        "lighting": np.nan, # W/m2
        "mositure_production": 3.53, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 16, #°C
        "min_t_op_in_cooling": 25, #°C
        "ventilation_rate(min)": 0.53, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": 100, # l/(m2 year)
    },
    "Residential_detached_house": {
        "occupants_quantity": 42.5, # m2/person
        "occupants": 2.8, # W/m2
        "appliances": 2.4, # W/m2
        "lighting": np.nan, # W/m2
        "mositure_production": 1.41, # g/(m2*h)
        "min_t_op_in_unoccupied_hours": 16, #°C
        "max_t_op_in_unoccupied_hours": 32, #°C
        "min_t_op_in_heating": 20, #°C
        "min_t_op_in_cooling": 26, #°C
        "ventilation_rate(min)": 0.5, # l/(s*m2)
        "lighting_in_working_area": 500, # lux
        "min_relative_humidity": 25, # %
        "max_relative_humidity": 60, # %
        "domestic_hot_water_use": 100, # l/(m2 year)
    },
}

