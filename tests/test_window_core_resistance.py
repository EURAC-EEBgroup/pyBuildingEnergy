"""Core resistance of an element: 1/U - R_si - R_se must be > 0 (no negative conductance)."""

import json
from pathlib import Path

import pytest

from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
from pybuildingenergy.source.utils import ISO52016


def _bui(u, type_="transparent", tag="W", tilt=90):
    return {"building_surface": [{"name": "w", "type": type_, "ISO52016_type_string": tag,
                                  "u_value": u, "orientation": {"azimuth": 0, "tilt": tilt}}]}


@pytest.mark.parametrize("u", [5.9, 6.047, 6.5, 10.0])
def test_window_above_the_surface_resistances_is_rejected(u):
    with pytest.raises(ValueError, match=r"building_surface\[0\].*core resistance"):
        ISO52016.Conductance_node_of_element(_bui(u))


def test_opaque_element_above_the_limit_is_rejected():
    with pytest.raises(ValueError, match="core resistance"):
        ISO52016.Conductance_node_of_element(_bui(8.0, "opaque", "OP"))


@pytest.mark.parametrize("u", [0.0, -1.0, float("nan")])
def test_non_positive_u_is_rejected(u):
    with pytest.raises(ValueError, match="u_value"):
        ISO52016.Conductance_node_of_element(_bui(u))


@pytest.mark.parametrize("u", [1.0, 5.0, 5.7])
def test_valid_window_reconstructs_u_from_the_series_resistance(u):
    b = _bui(u)
    h = ISO52016.Conductance_node_of_element(b).h_pli_eli[0, 0]
    s = b["building_surface"][0]
    r_si = 1 / (s["convective_heat_transfer_coefficient_internal"] + s["radiative_heat_transfer_coefficient_internal"])
    r_se = 1 / (s["convective_heat_transfer_coefficient_external"] + s["radiative_heat_transfer_coefficient_external"])
    assert h > 0
    assert 1.0 / (r_si + 1.0 / h + r_se) == pytest.approx(u)


def test_input_validation_reports_the_incompatible_u_value():
    bui = {"building": {"name": "x"}, "building_surface": [
        {"name": "w", "type": "transparent", "area": 2.0, "u_value": 6.1, "g_value": 0.8,
         "orientation": {"azimuth": 180, "tilt": 90}, "sky_view_factor": 0.5}]}
    _, issues = sanitize_and_validate_BUI(bui, fix=False)
    text = " ".join(str(i) for i in issues)
    assert "not compatible with the surface resistances" in text


def test_archetype_windows_are_within_the_limit():
    p = Path(__file__).resolve().parents[1] / "examples" / "building_archetypes.json"
    if not p.exists():
        pytest.skip("archetypes not available")
    us = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("type") == "transparent" and isinstance(o.get("u_value"), (int, float)):
                us.append(o["u_value"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(p.read_text(encoding="utf-8")))
    assert us and max(us) <= 5.7
