"""Areal heat capacity of the nodes must not depend on the order of the surfaces."""

import numpy as np
import pytest

from pybuildingenergy.source.utils import ISO52016

CLASSES = ("class_i", "class_e", "class_ie", "class_d", "class_m")


def _kappa(cls, types):
    b = {"building": {"construction_class": cls},
         "building_surface": [{"ISO52016_type_string": t, "thermal_capacity": 1000.0} for t in types]}
    return ISO52016.Areal_heat_capacity_of_element(b).kappa_pli_eli


@pytest.mark.parametrize("cls", CLASSES)
def test_wall_and_slab_vectors_do_not_depend_on_the_order(cls):
    a = _kappa(cls, ("OP", "GR"))
    b = _kappa(cls, ("GR", "OP"))
    np.testing.assert_allclose(a[:, 0], b[:, 1])     # wall
    np.testing.assert_allclose(a[:, 1], b[:, 0])     # slab
    # the wall keeps its whole capacity
    assert a[:, 0].sum() == pytest.approx(1000.0)


def test_expected_wall_nodes_class_e_and_class_m():
    assert _kappa("class_e", ("GR", "OP"))[:, 1].tolist() == [1000.0, 0.0, 0.0, 0.0, 0.0]
    assert _kappa("class_m", ("GR", "OP"))[:, 1].tolist() == [0.0, 0.0, 1000.0, 0.0, 0.0]
