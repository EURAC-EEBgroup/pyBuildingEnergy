"""Energy-balance checks for a single-zone building with an adjacent unheated zone.

The adjacent-type surface (type "adjacent", name_adj_zone pointing at an unheated
zone) is assembled in the ADJ branch of the single-zone core. Its heat flow must
appear in the Sankey balance; these tests assert that it does.
"""

import copy
import json
from pathlib import Path

import pytest

from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
from pybuildingenergy.source.utils import ISO52016

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "examples" / "FH_Poland_DC_baseline_two_floors.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"


def _adjacent_floor_config():
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    bui = cfg["building"]
    bui["building"]["adj_zones_present"] = True
    bui["building"]["number_adj_zone"] = 1
    bui["adjacent_zones"] = [{
        "name": "Z_first",
        "orientation_zone": {"azimuth": 0},
        "area_facade_elements": [242.91, 75.64, 52.155, 61.42, 12.975, 8.705],
        "typology_elements": ["OP", "OP", "OP", "OP", "OP", "OP"],
        "transmittance_U_elements": [0.58, 1.14, 1.14, 1.14, 1.14, 1.14],
        "orientation_elements": ["HOR", "NV", "SV", "EV", "WV", "SV"],
        "volume": 2390.234,
        "building_type_class": "Residential_apartment",
        "a_use": 242.91,
    }]
    surfaces = []
    for s in bui["building_surface"]:
        if s["name"] == "Ceiling to unheated attic":
            surfaces.append({
                "name": "Floor_between_Z_ground_and_Z_first",
                "type": "adjacent",
                "area": 242.91,
                "sky_view_factor": 0,
                "u_value": 0.7,
                "solar_absorptance": 0,
                "thermal_capacity": 49265,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": "Z_first",
            })
        else:
            surfaces.append(s)
    bui["building_surface"] = surfaces
    return bui


@pytest.fixture(scope="module")
def adjacent_run():
    if not BASE_CONFIG.exists():
        pytest.skip("FH_Poland_DC_baseline_two_floors.json not available (gitignored local input)")
    bui, _ = sanitize_and_validate_BUI(copy.deepcopy(_adjacent_floor_config()), fix=True)
    _, _, sankey = ISO52016.Temperature_and_Energy_needs_calculation(
        bui,
        weather_source="epw",
        path_weather_file=str(WEATHER),
        return_sankey_data=True,
    )
    return sankey


def test_adjacent_unheated_zone_sankey_closes(adjacent_run):
    inputs = sum(adjacent_run["inputs"].values())
    residual = adjacent_run["outputs"].get("Transmission (residual)", 0.0)
    assert residual / inputs < 0.01


def test_adjacent_floor_flux_is_reported_in_sankey(adjacent_run):
    floor_entries = [k for k in adjacent_run["outputs"] if "Floor_between" in k]
    assert floor_entries, f"no Sankey entry for the adjacent floor; outputs: {list(adjacent_run['outputs'])}"
    assert adjacent_run["outputs"][floor_entries[0]] > 0
