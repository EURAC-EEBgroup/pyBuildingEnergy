import os
import pickle

def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists.
    If it doesn't exist, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def get_buildings_demos(main_directory_):
    """
    Get archetypes and demo buildings
    """
    # pickle_file_path = main_directory_ + "/pybuildingenergy/pybuildingenergy/data/archetypes.pickle"
    pickle_file_path = main_directory_ + "/archetypes.pickle"
    with open(pickle_file_path, "rb") as f:
        archetypes = pickle.load(f)

    return archetypes