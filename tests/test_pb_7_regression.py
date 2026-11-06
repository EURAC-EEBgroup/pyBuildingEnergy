import pytest
from pybuildingenergy.source.utils import ISO52016

def building(u):
    return {"building_surface": [{"type": "transparent", "ISO52016_type_string": "W", "u_value": u, "orientation": {"azimuth": 0, "tilt": 90}, "convective_heat_transfer_coefficient_internal": 2.5, "radiative_heat_transfer_coefficient_internal": 5.13, "convective_heat_transfer_coefficient_external": 20., "radiative_heat_transfer_coefficient_external": 4.14}]}

@pytest.mark.parametrize("u", [5.9, -1., float("nan"), float("inf")])
def test_invalid_core_resistance_rejected(u):
    with pytest.raises(ValueError):
        ISO52016.Conductance_node_of_element(building(u))

@pytest.mark.parametrize("u", [1., 5., 5.7])
def test_window_series_conductance_recovers_u(u):
    h = ISO52016.Conductance_node_of_element(building(u)).h_pli_eli[0,0]
    assert h > 0
    assert 1 / (1 / 7.63 + 1 / h + 1 / 24.14) == pytest.approx(u)


def test_zero_core_resistance_rejected():
    b = building(0.5)
    s = b["building_surface"][0]
    s.update(convective_heat_transfer_coefficient_internal=1., radiative_heat_transfer_coefficient_internal=0., convective_heat_transfer_coefficient_external=1., radiative_heat_transfer_coefficient_external=0.)
    with pytest.raises(ValueError):
        ISO52016.Conductance_node_of_element(b)
