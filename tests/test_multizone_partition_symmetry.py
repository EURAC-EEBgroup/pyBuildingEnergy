"""INTERNAL partitions between simulated zones: internal films on both faces (Table 25).

The result must not depend on which zone declares the partition, nor on the *_external
fields (they belong to outdoor/unconditioned boundaries).  Floors/ceilings take the
reversed heat-flow direction on the second face.
"""

import contextlib
import copy
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybuildingenergy.source.utils import ISO52016, _partition_side_b_coefficients

ROOT = Path(__file__).resolve().parents[1]
WEATHER = ROOT / "examples" / "2020_Athens.epw"


def _wall(**kw):
    s = {"name": "p", "type": "opaque", "boundary": "INTERNAL", "zone": "Z1", "adjacent_zone": "Z2",
         "area": 10.0, "u_value": 1.5, "ISO52016_type_string": "ADJ",
         "orientation": {"azimuth": 0, "tilt": 90}}
    s.update(kw)
    return s


def test_vertical_partition_has_the_same_internal_film_on_both_faces():
    s = _wall(convective_heat_transfer_coefficient_internal=2.2,
              radiative_heat_transfer_coefficient_internal=5.13,
              convective_heat_transfer_coefficient_external=11.9,   # wall-row values: ignored
              radiative_heat_transfer_coefficient_external=4.14)
    assert _partition_side_b_coefficients(s) == pytest.approx((2.2, 5.13))


def test_floor_ceiling_takes_the_reversed_heat_flow_on_the_second_face():
    up = _wall(orientation={"azimuth": 0, "tilt": 0})       # upwards seen from the declaring zone
    down = _wall(orientation={"azimuth": 0, "tilt": 180})
    assert _partition_side_b_coefficients(up)[0] == pytest.approx(0.7)
    assert _partition_side_b_coefficients(down)[0] == pytest.approx(5.0)


def test_explicit_second_face_coefficients_are_used():
    s = _wall(convective_heat_transfer_coefficient_internal_b=3.0,
              radiative_heat_transfer_coefficient_internal_b=4.0)
    assert _partition_side_b_coefficients(s) == pytest.approx((3.0, 4.0))


def test_u_value_stays_air_to_air_whatever_the_external_fields():
    base = _wall()
    wallrow = _wall(convective_heat_transfer_coefficient_external=11.9,
                    radiative_heat_transfer_coefficient_external=4.14)
    h0 = ISO52016.Conductance_node_of_element({"building_surface": [base]}).h_pli_eli[:, 0]
    h1 = ISO52016.Conductance_node_of_element({"building_surface": [wallrow]}).h_pli_eli[:, 0]
    np.testing.assert_allclose(h0, h1)


# ------------------------------------------------------------------ engine symmetry
def _example():
    sys.path.insert(0, str(ROOT / "examples"))
    import multizone_free_floating_example as m
    return copy.deepcopy(m.building_object)


def _swap_declaration(bui):
    """Declare the Z1-Z2 partition from Z2 instead of Z1 (same physical wall)."""
    for s in bui["building_surface"]:
        if str(s.get("boundary", "")).upper() == "INTERNAL":
            s["zone"], s["adjacent_zone"] = s["adjacent_zone"], s["zone"]
            ori = s.setdefault("orientation", {})
            ori["azimuth"] = (float(ori.get("azimuth", 0.0)) + 180.0) % 360.0
            s.pop("ISO52016_orientation_string", None)
    return bui


def _wall_row_films(bui):
    for s in bui["building_surface"]:
        if str(s.get("boundary", "")).upper() == "INTERNAL":
            # symmetric construction: with the mass on one face (class_e/class_i) swapping
            # the declaring zone also mirrors the wall, a physical change
            s["construction_class"] = "class_d"
            s["convective_heat_transfer_coefficient_internal"] = 2.2
            s["radiative_heat_transfer_coefficient_internal"] = 5.13
            s["convective_heat_transfer_coefficient_external"] = 11.9
            s["radiative_heat_transfer_coefficient_external"] = 4.14
    return bui


def _run(bui):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        h = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw", include_total=False,
            include_latent=False)
    return h[0] if isinstance(h, tuple) else h


@pytest.mark.skipif(not WEATHER.exists(), reason="weather missing")
def test_result_does_not_depend_on_the_declaring_zone():
    a = _run(_wall_row_films(_example()))
    b = _run(_swap_declaration(_wall_row_films(_example())))
    for z in ("Z1", "Z2"):
        np.testing.assert_allclose(a[f"T_air_{z}"], b[f"T_air_{z}"], atol=1e-6)
        np.testing.assert_allclose(a[f"Q_HVAC_{z}"], b[f"Q_HVAC_{z}"], atol=1e-3)
    # and the wall-row external fields change nothing for an INTERNAL partition
    c = _run(_example())
    bui_c = _example()
    for s in bui_c["building_surface"]:
        if str(s.get("boundary", "")).upper() == "INTERNAL":
            s["convective_heat_transfer_coefficient_external"] = 11.9
            s["radiative_heat_transfer_coefficient_external"] = 4.14
    d = _run(bui_c)
    np.testing.assert_allclose(c["T_air_Z1"], d["T_air_Z1"], atol=1e-6)
