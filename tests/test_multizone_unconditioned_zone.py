"""Thermally unconditioned adjacent zones in the multizone engine (EN ISO 52016-1 6.4.5, 6.5.9)."""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pybuildingenergy.source.utils import ISO52016

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"
pytestmark = pytest.mark.skipif(not (CONFIG.exists() and WEATHER.exists()), reason="inputs missing")

WALL = "Opaque north surface_Z_ground"


def _garage(**kw):
    z = {
        "name": "garage", "type": "external",
        "orientation_zone": {"azimuth": 0},
        "area_facade_elements": [40.0, 40.0, 20.0, 20.0, 60.0],
        "typology_elements": ["OP", "OP", "OP", "OP", "OP"],
        "transmittance_U_elements": [1.0, 1.0, 1.0, 1.0, 1.0],
        "orientation_elements": ["NV", "SV", "EV", "WV", "HOR"],
        "volume": 180.0, "building_type_class": "Residential_apartment", "a_use": 60.0,
    }
    z.update(kw)
    return z


def _bui(garage=None, boundary="UNCONDITIONED"):
    bui = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])
    for z in bui["zones"]:
        z["cooling_capacity"] = 1e6
    if garage is not None:
        bui["building"]["adj_zones_present"] = True
        bui["building"]["number_adj_zone"] = 1
        bui["adjacent_zones"] = [garage]
        for s in bui["building_surface"]:
            if s["name"] == WALL:
                s["boundary"] = boundary
                s["name_adj_zone"] = garage["name"]
    return bui


def _run(bui):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw", include_total=False,
            include_latent=False)


def _hourly(res):
    return res[0] if isinstance(res, tuple) else res


@pytest.fixture(scope="module")
def runs():
    assert any(s["name"] == WALL for s in _bui()["building_surface"])
    return {
        "outdoors": _hourly(_run(_bui())),
        "external": _hourly(_run(_bui(_garage(b_ztu=0.5)))),
        "internal": _hourly(_run(_bui(_garage(type="internal", b_ztu=0.5)))),
    }


def _q_heat(h, zone):
    q = h[f"Q_HVAC_{zone}"].to_numpy(dtype=float)
    return q.clip(min=0).sum() / 1000.0


def test_theta_ztu_follows_formula_58_with_previous_hour(runs):
    h = runs["external"]
    th = h["theta_ztu_garage"].to_numpy(dtype=float)
    op = h["T_op_Z_ground"].to_numpy(dtype=float)          # the only zone next to the garage
    te = h["T_ext"].to_numpy(dtype=float)
    expected = np.minimum(te[1:] + 1.0 * (op[:-1] - te[1:]), op[:-1] - 0.5 * (op[:-1] - te[1:]))
    np.testing.assert_allclose(th[1:], expected, atol=1e-6)
    # and not with the same-hour operative temperature
    same = np.minimum(te[1:] + (op[1:] - te[1:]), op[1:] - 0.5 * (op[1:] - te[1:]))
    assert np.abs(th[1:] - same).max() > 0.05


def test_unconditioned_zone_lowers_the_heat_loss_of_its_neighbour_only(runs):
    q_out = _q_heat(runs["outdoors"], "Z_ground")
    q_ext = _q_heat(runs["external"], "Z_ground")
    q_int = _q_heat(runs["internal"], "Z_ground")
    assert q_ext < q_out                                    # a buffer instead of the outdoors
    assert q_int < q_out
    # Z_first is not next to the garage: it changes only through the floor
    d_first = abs(_q_heat(runs["external"], "Z_first") - _q_heat(runs["outdoors"], "Z_first"))
    assert d_first < 0.35 * (q_out - q_ext)


def test_internal_type_uses_operative_temperature_and_formula_54(runs):
    h = runs["internal"]
    th = h["theta_ztu_garage"].to_numpy(dtype=float)
    op = h["T_op_Z_ground"].to_numpy(dtype=float)
    np.testing.assert_allclose(th[1:], op[:-1], atol=1e-9)   # 6.4.5.3
    # with b = 0.5 the wall loses about (1 - b) of its loss to the outdoors
    wall = next(s for s in _bui()["building_surface"] if s["name"] == WALL)
    ua = wall["u_value"] * wall["area"]
    dt = (h["T_op_Z_ground"] - h["T_ext"]).clip(lower=0).sum()
    saved = _q_heat(runs["outdoors"], "Z_ground") - _q_heat(h, "Z_ground")
    assert 0.2 < saved / (0.5 * ua * dt / 1000.0) < 1.6


def test_unknown_unconditioned_zone_is_rejected():
    bui = _bui(_garage(b_ztu=0.5))
    for s in bui["building_surface"]:
        if s["name"] == WALL:
            s["name_adj_zone"] = "cellar"
    with pytest.raises(ValueError, match="name_adj_zone"):
        _run(bui)


def test_hybrid_method_rejects_unconditioned_zones():
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(
                _bui(_garage(b_ztu=0.5)), path_weather_file=str(WEATHER), weather_source="epw")
