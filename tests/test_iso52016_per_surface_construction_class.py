"""The single-zone ISO 52016 engine honours a per-surface ``construction_class``,
falling back to the building-level default when a surface does not set its own.

ISO 52016-1's own Case 600/900 worked example (Annex-style verification in
clause 7.2.2.3, Tables 23-24) assigns a *different* mass-distribution class
to different element types within the same building -- e.g. Case 600's
wall and roof are "Evenly (D)" (class_d) while its floor is "Internal (I)"
(class_i). Before this fix, ``building_object["building"]["construction_class"]``
was the only input the engine read, applied uniformly to every opaque
surface regardless of type -- which cannot represent that case.
"""

import pytest

from pybuildingenergy.source.utils import ISO52016


def test_per_surface_construction_class_overrides_building_default():
    building_object = {
        "building": {"construction_class": "class_d"},
        "building_surface": [
            {
                "name": "wall",
                "ISO52016_type_string": "OP",
                "thermal_capacity": 1000.0,
                "construction_class": "class_i",
            },
            {
                "name": "roof",
                "ISO52016_type_string": "OP",
                "thermal_capacity": 2000.0,
                # no per-surface override -> falls back to the building default
            },
        ],
    }
    kappa = ISO52016.Areal_heat_capacity_of_element(building_object).kappa_pli_eli

    # wall (class_i): all mass concentrated at the innermost node (index 4)
    assert kappa[:, 0] == pytest.approx([0.0, 0.0, 0.0, 0.0, 1000.0])
    # roof (class_d, from the building default): evenly distributed
    assert kappa[:, 1] == pytest.approx([250.0, 500.0, 500.0, 500.0, 250.0])


def test_per_surface_construction_class_matches_uniform_single_class_result():
    """When every surface shares the same class (the pre-existing, single-class
    usage), the per-surface path must reproduce the old building-level-only
    result exactly."""
    surfaces = [
        {"name": "wall", "ISO52016_type_string": "OP", "thermal_capacity": 1000.0},
        {"name": "floor", "ISO52016_type_string": "GR", "thermal_capacity": 500.0},
    ]
    uniform_global = ISO52016.Areal_heat_capacity_of_element(
        {"building": {"construction_class": "class_d"}, "building_surface": surfaces}
    ).kappa_pli_eli
    explicit_per_surface = ISO52016.Areal_heat_capacity_of_element(
        {
            "building": {"construction_class": "class_d"},
            "building_surface": [dict(s, construction_class="class_d") for s in surfaces],
        }
    ).kappa_pli_eli
    assert explicit_per_surface == pytest.approx(uniform_global)


@pytest.mark.parametrize("cls", ["class_i", "class_e", "class_ie", "class_d", "class_m"])
def test_each_construction_class_still_only_affects_its_own_surfaces(cls):
    """A surface tagged with a given class must be unaffected by *other*
    surfaces present in the same call with a different class (no cross-talk
    between the five per-class branches)."""
    other_classes = [c for c in ("class_i", "class_e", "class_ie", "class_d", "class_m") if c != cls]
    building_object = {
        "building": {"construction_class": cls},
        "building_surface": [
            {"name": "target", "ISO52016_type_string": "OP", "thermal_capacity": 1000.0},
        ]
        + [
            {
                "name": f"other_{other}",
                "ISO52016_type_string": "OP",
                "thermal_capacity": 999.0,
                "construction_class": other,
            }
            for other in other_classes
        ],
    }
    alone = ISO52016.Areal_heat_capacity_of_element(
        {
            "building": {"construction_class": cls},
            "building_surface": [building_object["building_surface"][0]],
        }
    ).kappa_pli_eli
    with_others = ISO52016.Areal_heat_capacity_of_element(building_object).kappa_pli_eli
    assert with_others[:, 0] == pytest.approx(alone[:, 0])


def test_aggregation_preserves_per_surface_construction_class():
    """Two surfaces with the same orientation/type but a *different*
    construction_class must stay in separate aggregated surfaces (a
    categorical class cannot be area-weight-averaged like a U-value)."""
    building_object = {
        "building_surface": [
            {
                "name": "wall1", "type": "opaque", "area": 10.0,
                "ISO52016_type_string": "OP", "ISO52016_orientation_string": "SV",
                "construction_class": "class_i",
            },
            {
                "name": "wall2", "type": "opaque", "area": 10.0,
                "ISO52016_type_string": "OP", "ISO52016_orientation_string": "SV",
                "construction_class": "class_i",
            },
            {
                "name": "wall3", "type": "opaque", "area": 10.0,
                "ISO52016_type_string": "OP", "ISO52016_orientation_string": "SV",
                "construction_class": "class_d",
            },
        ]
    }
    aggregated = ISO52016._aggregate_surfaces_by_direction(building_object)["building_surface"]
    by_class = {s["construction_class"]: s["area"] for s in aggregated}
    assert by_class == {"class_i": 20.0, "class_d": 10.0}


def test_aggregation_without_construction_class_keeps_building_default():
    building_object = {
        "building_surface": [
            {
                "name": "wall1", "type": "opaque", "area": 10.0,
                "ISO52016_type_string": "OP", "ISO52016_orientation_string": "SV",
            },
            {
                "name": "wall2", "type": "opaque", "area": 10.0,
                "ISO52016_type_string": "OP", "ISO52016_orientation_string": "SV",
            },
        ]
    }
    aggregated = ISO52016._aggregate_surfaces_by_direction(building_object)["building_surface"][0]
    assert "construction_class" not in aggregated
    assert aggregated["area"] == pytest.approx(20.0)


KAPPA_GR = 1.0e6  # placeholder for the fixed-ground layer capacity until ISO 13370/13789 is applied


@pytest.mark.parametrize(
    "cls, expected_shares",
    [
        # EN ISO 52016-1 eq. 50a-50e: share of km on nodes pl1..pl5 of a ground floor
        # (node 2 is replaced by the fixed ground capacity, node 1 stays 0)
        ("class_i", [0.0, 0.0, 0.0, 0.0, 1.0]),
        ("class_e", [0.0, 0.0, 1.0, 0.0, 0.0]),
        ("class_ie", [0.0, 0.0, 0.5, 0.0, 0.5]),
        ("class_d", [0.0, 0.0, 0.25, 0.5, 0.25]),
        ("class_m", [0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
)
def test_ground_floor_node_capacities_follow_eq_50(cls, expected_shares):
    km = 200000.0
    building_object = {
        "building": {"construction_class": "class_d"},
        "building_surface": [
            {
                "name": "floor",
                "ISO52016_type_string": "GR",
                "thermal_capacity": km,
                "construction_class": cls,
            },
        ],
    }
    kappa = ISO52016.Areal_heat_capacity_of_element(building_object).kappa_pli_eli[:, 0]

    expected = [km * share for share in expected_shares]
    expected[1] = KAPPA_GR  # eq. 50f: kpl2 = kgr for all classes
    assert kappa == pytest.approx(expected)
    assert kappa[0] == 0.0  # kpl1 = kgr;vi = 0
