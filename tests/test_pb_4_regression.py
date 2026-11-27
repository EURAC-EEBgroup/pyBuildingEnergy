import pytest
from pybuildingenergy.source.utils import ISO52016

@pytest.mark.parametrize("invalid", ["class_ii", "class_iii", "typo", ""])
@pytest.mark.parametrize("surface_override", [False, True])
def test_unknown_construction_class_rejected(invalid, surface_override):
    surface = {"ISO52016_type_string": "OP", "thermal_capacity": 1000.0}
    building = {"construction_class": "class_i" if surface_override else invalid}
    if surface_override:
        surface["construction_class"] = invalid
    with pytest.raises(ValueError, match="unknown construction_class"):
        ISO52016.Areal_heat_capacity_of_element({"building": building, "building_surface": [surface]})

@pytest.mark.parametrize("cls", ["class_i", "class_e", "class_ie", "class_d", "class_m"])
def test_valid_class_conserves_wall_capacity(cls):
    b = {"building": {"construction_class": cls}, "building_surface": [{"ISO52016_type_string": "OP", "thermal_capacity": 1000.0}]}
    assert ISO52016.Areal_heat_capacity_of_element(b).kappa_pli_eli.sum() == pytest.approx(1000.0)
