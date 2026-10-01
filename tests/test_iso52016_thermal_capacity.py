"""Areal heat capacity (``thermal_capacity``, J/(m2 K)) in the single-zone ISO 52016 path."""

import copy

import numpy as np
import pytest

from pybuildingenergy.source.utils import ISO52016
from tests.test_ventilation_boundary import _legacy_building, _run_single_zone_engine


def _opaque_wall(name, area, kappa):
    return {
        "name": name,
        "type": "opaque",
        "boundary": "OUTDOORS",
        "area": area,
        "u_value": 0.5,
        "thermal_capacity": kappa,
        "orientation": {"azimuth": 180.0, "tilt": 90.0},
        "ISO52016_type_string": "OP",
        "ISO52016_orientation_string": "SV",
    }


def test_aggregation_area_weights_areal_thermal_capacity():
    """Merged surfaces keep the total heat capacity sum(kappa_i * A_i)."""
    building_object = {
        "building_surface": [
            _opaque_wall("Wall 1", 2.0, 100000.0),
            _opaque_wall("Wall 2", 6.0, 200000.0),
        ]
    }

    aggregated = ISO52016._aggregate_surfaces_by_direction(building_object)
    assert len(aggregated["building_surface"]) == 1
    wall = aggregated["building_surface"][0]

    assert wall["area"] == pytest.approx(8.0)
    assert wall["thermal_capacity"] == pytest.approx(175000.0)
    assert wall["thermal_capacity"] * wall["area"] == pytest.approx(2.0 * 100000.0 + 6.0 * 200000.0)


def _building_with_adiabatic_floor(kappa_ad, area_ad=100.0):
    bld = _legacy_building([{
        "name": "inf",
        "ventilation_type": "prescribed",
        "heat_transfer_coefficient_w_k": 50.0,
        "source_temperature_c": -5.0,
    }])
    bld["building_surface"].append({
        "name": "floor_ad",
        "type": "adiabatic",
        "area": area_ad,
        "sky_view_factor": 0.0,
        "u_value": 0.0,
        "solar_absorptance": 0.0,
        "thermal_capacity": kappa_ad,
        "orientation": {"azimuth": 0, "tilt": 0},
        "name_adj_zone": None,
    })
    # Free-floating: the air-node capacity then shapes the temperature response.
    bld["building_parameters"]["system_capacities"] = {
        "heating_capacity": 0.0,
        "cooling_capacity": 0.0,
    }
    return bld


@pytest.mark.parametrize("control_mode", ["standard", "ahu_causal"])
def test_adiabatic_capacity_is_multiplied_by_area(control_mode):
    """kappa * A of an adiabatic element acts like the same capacity in c_int_per_A_us."""
    kappa_ad = 50000.0  # J/(m2 K)
    area_ad = 100.0  # m2
    net_floor_area = 100.0  # m2 (_legacy_building)

    with_ad = _run_single_zone_engine(
        _building_with_adiabatic_floor(kappa_ad, area_ad),
        control_mode=control_mode,
        n=24,
        t_out=-5.0,
    )
    # Same geometry, the capacity moved from the AD element to the air node.
    lumped = _run_single_zone_engine(
        _building_with_adiabatic_floor(0.0, area_ad),
        control_mode=control_mode,
        n=24,
        t_out=-5.0,
        c_int_per_A_us=10000.0 + kappa_ad * area_ad / net_floor_area,
    )
    # Without the capacity at all, for a sanity check that the test is sensitive.
    without = _run_single_zone_engine(
        _building_with_adiabatic_floor(0.0, area_ad),
        control_mode=control_mode,
        n=24,
        t_out=-5.0,
    )

    t_with = with_ad["T_air"].to_numpy()
    np.testing.assert_allclose(t_with, lumped["T_air"].to_numpy(), rtol=0.0, atol=1e-6)
    assert np.max(np.abs(t_with - without["T_air"].to_numpy())) > 0.1


def test_aggregated_adiabatic_capacity_independent_of_splitting():
    """One adiabatic floor or the same floor in two parts gives the same result."""
    one = _building_with_adiabatic_floor(50000.0, 100.0)
    two = _building_with_adiabatic_floor(50000.0, 50.0)
    second = copy.deepcopy(two["building_surface"][-1])
    second["name"] = "floor_ad_2"
    two["building_surface"].append(second)

    out_one = _run_single_zone_engine(one, n=24, t_out=-5.0)
    out_two = _run_single_zone_engine(two, n=24, t_out=-5.0)
    np.testing.assert_allclose(
        out_one["T_air"].to_numpy(), out_two["T_air"].to_numpy(), rtol=0.0, atol=1e-6
    )
