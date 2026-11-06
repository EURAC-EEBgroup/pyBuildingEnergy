import pytest
from pybuildingenergy.source.utils import ISO52016

@pytest.mark.parametrize("cls,expected", [("class_e", [1000,0,0,0,0]), ("class_m", [0,0,1000,0,0])])
@pytest.mark.parametrize("types", [("OP", "GR"), ("GR", "OP"), ("GR", "OP", "GR", "OP")])
def test_wall_capacity_independent_of_ground_surface_order(cls, expected, types):
    b = {"building": {"construction_class": cls}, "building_surface": [{"ISO52016_type_string": t, "thermal_capacity": 1000.0} for t in types]}
    k = ISO52016.Areal_heat_capacity_of_element(b).kappa_pli_eli
    for i, kind in enumerate(types):
        if kind == "OP":
            assert k[:, i] == pytest.approx(expected)
