import pytest

from pybuildingenergy.source.daylight_control import lighting_step


def test_daylight_priority_and_electric_heat():
    assert lighting_step(100, True, 300, 400, 200, heat_fraction=0.8) == pytest.approx((100, 200, 300, 100, 80))
    assert lighting_step(500, True, 300, 400, 200) == pytest.approx((500, 0, 500, 0, 0))


def test_occupancy_and_insufficient_installed_light():
    assert lighting_step(20, False, 300, 200, 100) == pytest.approx((20, 0, 20, 0, 0))
    assert lighting_step(20, True, 300, 200, 100) == pytest.approx((20, 200, 220, 100, 100))
    assert lighting_step(100, True, 300, 400, 200, mode="on_off")[1] == 400


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        lighting_step(float("nan"), True, 300, 400, 200)


def test_external_daylight_is_applied_to_multizone_heat_balance():
    from tests.test_ventilation_boundary import _one_zone_building, _patched_weather
    from pybuildingenergy.source.utils import ISO52016

    building = _one_zone_building()
    with _patched_weather(n=48) as weather:
        building["zones"][0]["daylight_lighting"] = {
            "method": "external", "illuminance_lux": {str(t): 100 for t in weather.index},
            "target_lux": 300, "design_lux": 400, "installed_power_w": 200,
            "heat_fraction": 0.8,
        }
        out = ISO52016.simulate_envelope_multizone_free_floating(
            building, use_profiles=False, warmup_hours=0, include_ventilation=False,
        )
    assert (out["daylight_lux_zone1"] == 100).all()
    assert (out["electric_lux_zone1"] == 200).all()
    assert (out["lighting_heat_W_zone1"] == 80).all()
    assert (out["Phi_int_zone1"] >= 80).all()
