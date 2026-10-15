"""Regression checks for the ASHRAE 140-2023 Case 600 raised floor."""

import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLE_DIR))

from case_definitions import build_case  # noqa: E402


def test_case600_floor_is_raised_and_exposed_to_air():
    building = build_case("light")
    floor = next(s for s in building["building_surface"] if s["name"] == "Floor")

    # ASHRAE 140-2023 Section 7.2.1.5: ambient air below the floor,
    # no ground-coupled boundary and no solar radiation on the underside.
    assert floor["boundary"] == "OUTDOORS"
    assert floor["solar_absorptance"] == 0.0
    assert floor["sky_view_factor"] == 1.0e-6

    # Section 7.2.1.6 / Table 7-4: the 0.5 ach specification is adjusted
    # for this engine's fixed sea-level air properties.
    infiltration = building["building_parameters"]["ventilation"]["components"][0]
    assert infiltration["ventilation_type"] == "constant_ach"
    assert infiltration["air_changes_per_hour"] < 0.5
