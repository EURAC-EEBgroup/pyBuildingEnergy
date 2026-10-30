"""Composer <-> LightingSystemCalculator (EN 15193-1): lighting electricity joins the site load."""

from pathlib import Path

import numpy as np
import pytest

import pybuildingenergy as pybui

EPW = Path(__file__).resolve().parent / "USA_CO_Denver.Intl.AP.725650_TMY3.epw"
LIGHTING = {"installed_power_density_W_m2": 6.0, "parasitic_power_W_m2": 0.1,
            "emergency_lighting_kWh_m2_a": 1.0}


def _scenario(tmp_path, lighting=None, pv=None):
    return {"generation": "boiler_15316_4_1", "weather_source": "epw", "weather_file": str(EPW),
            "lighting": lighting, "pv": pv, "options": {"hourly_cache_dir": str(tmp_path)}}


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("zone_cache")


def test_lighting_electricity_is_added_to_the_site_load_and_final_energy(cache_dir):
    base = pybui.simulate(pybui.compose_config(_scenario(cache_dir)))
    lit = pybui.simulate(pybui.compose_config(_scenario(cache_dir, lighting=LIGHTING)))
    s = lit.summary
    area = s["floor_area_m2"]
    assert "E_lighting_kWh" not in base.summary
    assert s["E_lighting_kWh"] > 0.0
    assert s["E_lighting_kWh"] == pytest.approx(
        s["E_lighting_use_kWh"] + s["E_lighting_parasitic_kWh"] + s["E_lighting_emergency_kWh"])
    assert s["LENI_kWh_m2a"] == pytest.approx(s["E_lighting_kWh"] / area)
    assert s["E_lighting_emergency_kWh"] == pytest.approx(1.0 * area)
    assert s["E_lighting_parasitic_kWh"] == pytest.approx(0.1 * area * 8760 / 1000.0, rel=1e-3)
    # the hourly site load rises by exactly the lighting electricity, nothing else changes
    delta = lit.hourly["E_site_el_load_kWh"] - base.hourly["E_site_el_load_kWh"]
    assert delta.to_numpy() == pytest.approx(lit.hourly["E_lighting_kWh"].to_numpy())
    assert (s["final_energy_by_carrier_kWh"]["electricity_kWh"]
            - base.summary["final_energy_by_carrier_kWh"]["electricity_kWh"]) == pytest.approx(s["E_lighting_kWh"])
    assert s["E_grid_import_kWh"] - base.summary["E_grid_import_kWh"] == pytest.approx(s["E_lighting_kWh"], rel=1e-6)
    assert s["Q_H_need_kWh"] == pytest.approx(base.summary["Q_H_need_kWh"])  # the zone is untouched
    assert s["system_efficiency"] == pytest.approx(base.summary["system_efficiency"])
    assert any("Lighting (EN 15193-1)" in n and "NOT fed back" in n for n in lit.notes)
    assert "Lighting (EN 15193-1)" in lit.report()


def test_lighting_profile_24h_and_weekend_profile(cache_dir):
    cfg = {"installed_power_density_W_m2": 10.0, "profile_24h": [0.0] * 24,
           "weekend_profile_24h": [1.0] * 24}
    res = pybui.simulate(pybui.compose_config(_scenario(cache_dir, lighting=cfg)))
    h = res.hourly["E_lighting_kWh"]
    weekday = h[h.index.dayofweek < 5]
    weekend = h[h.index.dayofweek >= 5]
    assert weekday.sum() == pytest.approx(0.0)
    assert weekend.min() > 0.0
    with pytest.raises(ValueError, match="24 hourly values"):
        pybui.simulate(pybui.compose_config(_scenario(cache_dir, lighting={"profile_24h": [1.0] * 5})))


def test_lighting_is_seen_by_the_pv_self_consumption(cache_dir):
    pv = "pv_with_battery_dispatch"
    no_light = pybui.simulate(pybui.compose_config(_scenario(cache_dir, pv=pv)))
    lit = pybui.simulate(pybui.compose_config(_scenario(cache_dir, lighting=LIGHTING, pv=pv)))
    assert lit.summary["E_PV_self_consumed_kWh"] >= no_light.summary["E_PV_self_consumed_kWh"] - 1e-6
    assert lit.summary["E_PV_gen_kWh"] == pytest.approx(no_light.summary["E_PV_gen_kWh"])


def test_composed_config_carries_the_lighting_block():
    config = pybui.compose_config({"lighting": LIGHTING, "weather_source": "epw", "weather_file": str(EPW)})
    assert config["lighting"] == LIGHTING
    assert pybui.compose_config({"weather_source": "epw", "weather_file": str(EPW)})["lighting"] is None


def test_daylight_control_without_irradiance_is_reported(cache_dir):
    cfg = dict(LIGHTING, daylight_control_fraction=0.5)
    res = pybui.simulate(pybui.compose_config(_scenario(cache_dir, lighting=cfg)))
    assert any("daylight_control_fraction" in w and "no daylight reduction" in w for w in res.warnings)
