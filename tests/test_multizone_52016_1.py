"""Multi-zone calculation with thermal coupling (EN ISO 52016-1 6.4.2, 6.4.6, 6.4.7, Annex D)."""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybuildingenergy.source import multizone_52016_1 as mz
from pybuildingenergy.source.utils import ISO52016

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "FH_Poland_DC_multizone_ideal_envelope.json"
WEATHER = ROOT / "examples" / "2020_Athens.epw"

pytestmark = pytest.mark.skipif(
    not (CONFIG.exists() and WEATHER.exists()), reason="example multizone inputs not available"
)

PARTITION_A = "Q_opaque_inside_surface_Floor_between_Z_ground_and_Z_first_to_Z_ground"
PARTITION_B = "Q_opaque_inside_surface_Floor_between_Z_ground_and_Z_first_to_Z_first"


def _building():
    return copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8"))["building"])


def _calc(bui, **kw):
    kw.setdefault("warmup_hours", 744)
    kw.setdefault("hvac_control_variable", "operative")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw", **kw
        )


def _partition_only(bui, ground_set=(10.0, 20.0), first_set=(26.0, 40.0)):
    """Everything adiabatic except the partition; no gains, no ventilation, no solar."""
    surfaces = []
    for s in bui["building_surface"]:
        if s["type"] == "transparent":
            continue
        if s["boundary"] != "INTERNAL":
            s["boundary"] = "ADIABATIC"
        surfaces.append(s)
    bui["building_surface"] = surfaces
    for z in bui["zones"]:
        z["internal_gains"] = []
        z["heating_capacity"] = 1e6
        z["cooling_capacity"] = 1e6
        z["heating_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
        z["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}
        z["ventilation_profile"] = {"weekday": [0] * 24, "weekend": [0] * 24}
        z.pop("heating_setback", None)
        z.pop("cooling_setback", None)
    bui["zones"][0].update(heating_setpoint=ground_set[0], cooling_setpoint=ground_set[1])
    bui["zones"][1].update(heating_setpoint=first_set[0], cooling_setpoint=first_set[1])
    return bui


def _steady(bui, **kw):
    return _calc(bui, include_solar=False, include_internal_gains=False,
                 include_ventilation=False, **kw)


# --------------------------------------------------------------------- full year


@pytest.fixture(scope="module")
def full_year():
    return _calc(_building())


def test_partition_heat_is_conserved(full_year):
    hourly, _ = full_year
    a, b = hourly[PARTITION_A].sum() / 1000.0, hourly[PARTITION_B].sum() / 1000.0
    assert abs(a + b) < 0.5          # kWh over the year; the net exchange is ~90 kWh
    assert abs(a) > 10.0


def test_total_row_is_the_sum_of_the_zones(full_year):
    _, annual = full_year
    zones = annual[annual["zone"] != "TOTAL"]
    total = annual[annual["zone"] == "TOTAL"].iloc[0]
    assert len(annual) == len(zones) + 1
    assert total["Q_H_annual_kWh"] == pytest.approx(zones["Q_H_annual_kWh"].sum())
    assert total["Q_C_annual_kWh"] == pytest.approx(zones["Q_C_annual_kWh"].sum())
    area = sum(z["net_floor_area"] for z in _building()["zones"])
    assert total["Q_H_annual_kWh_per_sqm"] == pytest.approx(total["Q_H_annual_kWh"] / area)


def test_include_total_false_keeps_per_zone_rows_only():
    bui = _partition_only(_building())
    _, annual = _steady(bui, include_total=False)
    assert "TOTAL" not in set(annual["zone"])


# ---------------------------------------------------------------------- steady state


def test_partition_conductance_equals_u_times_a_on_operative_temperature():
    bui = _partition_only(_building())
    hourly, _ = _steady(bui)
    t = hourly.iloc[-2000:]
    s = [x for x in bui["building_surface"] if x["boundary"] == "INTERNAL"][0]
    ua = s["u_value"] * s["area"]
    h_meas = t["Q_HVAC_Z_first"].mean() / (t["T_op_Z_first"] - t["T_op_Z_ground"]).mean()
    assert h_meas == pytest.approx(ua, rel=0.02)
    # whatever one zone gives the other takes
    assert t["Q_HVAC_Z_first"].mean() == pytest.approx(-t["Q_HVAC_Z_ground"].mean(), rel=1e-6)


def test_coupled_zone_leaving_its_range_switches_mode():
    """A zone idle on its own is cooled once the neighbour heats it above its set-point."""
    bui = _partition_only(_building(), ground_set=(10.0, 22.0), first_set=(30.0, 40.0))
    hourly, _ = _steady(bui)
    t = hourly.iloc[-500:]
    assert (t["T_op_Z_ground"] <= 22.0 + 1e-3).all()
    assert (t["mode_Z_ground"] == "C").all()
    assert t["Q_HVAC_Z_ground"].mean() < 0.0


def test_free_floating_zone_gets_no_hvac():
    bui = _partition_only(_building(), ground_set=(10.0, 40.0))
    bui["zones"][0]["free_floating"] = True
    hourly, annual = _steady(bui, include_total=False)
    t = hourly.iloc[-500:]
    assert (t["mode_Z_ground"] == "0").all()
    assert (t["Q_HVAC_Z_ground"] == 0.0).all()
    # the unconditioned zone settles between outdoors-adiabatic and the heated neighbour
    assert t["T_op_Z_ground"].mean() == pytest.approx(t["T_op_Z_first"].mean(), abs=0.2)


# ----------------------------------------------------------- air flow between zones


def test_interzone_airflow_adds_the_expected_heat_transfer():
    q_m3_h = 300.0
    base = _partition_only(_building())
    flow = _partition_only(_building())
    flow["interzone_airflows"] = [{"from": "Z_first", "to": "Z_ground", "q_v_m3_h": q_m3_h}]
    h0, _ = _steady(base)
    h1, _ = _steady(flow)
    t0, t1 = h0.iloc[-1000:], h1.iloc[-1000:]
    h_ve = mz.RHO_C_AIR * q_m3_h / 3600.0
    # definition of the output: H_ve (T_source - T_receiver)
    expected = h_ve * (t1["T_air_Z_first"] - t1["T_air_Z_ground"])
    assert t1["Q_ve_interzone_Z_ground"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-6)
    assert (t1["Q_ve_interzone_Z_first"] == 0.0).all()      # source zone: no make-up by default
    # the cooled zone has to remove the extra heat (more cooling than without the air flow)
    assert t1["Q_HVAC_Z_ground"].mean() < t0["Q_HVAC_Z_ground"].mean() - 0.5 * expected.mean()


def test_interzone_airflow_outdoor_makeup_is_charged_to_the_source_zone():
    flow = _partition_only(_building())
    flow["interzone_airflows"] = [
        {"from": "Z_first", "to": "Z_ground", "q_v_m3_h": 300.0, "makeup_from_outdoor": True}
    ]
    hourly, _ = _steady(flow)
    t = hourly.iloc[-500:]
    h_ve = mz.RHO_C_AIR * 300.0 / 3600.0
    assert (t["Q_ve_interzone_Z_first"] < 0.0).all() or (t["Q_ve_interzone_Z_first"] > 0.0).all()
    assert abs(t["Q_ve_interzone_Z_first"].mean()) > 0.1 * h_ve


def test_air_flow_in_both_directions_is_rejected():
    bui = _partition_only(_building())
    bui["interzone_airflows"] = [
        {"from": "Z_first", "to": "Z_ground", "q_v_m3_h": 100.0},
        {"from": "Z_ground", "to": "Z_first", "q_v_m3_h": 50.0},
    ]
    with pytest.raises(ValueError, match="both directions"):
        mz.parse_interzone_airflows(bui)
    with pytest.raises(ValueError, match="both directions"):
        _steady(bui)


def test_hybrid_method_refuses_interzone_airflows():
    bui = _partition_only(_building())
    bui["interzone_airflows"] = [{"from": "Z_first", "to": "Z_ground", "q_v_m3_h": 100.0}]
    with pytest.raises(NotImplementedError):
        ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(
            bui, path_weather_file=str(WEATHER), weather_source="epw"
        )


def test_interzone_coupling_matrix():
    zi = {"A": 0, "B": 1}
    flows = mz.parse_interzone_airflows(
        {"zones": [{"name": "A"}, {"name": "B"}],
         "interzone_airflows": [{"from": "A", "to": "B", "q_v_m3_s": 0.01, "makeup_from_outdoor": True}]}
    )
    h, s = mz.interzone_coupling(flows, zi, 2, t_out=5.0)
    hh = mz.RHO_C_AIR * 0.01
    assert h[1, 1] == pytest.approx(hh) and h[1, 0] == pytest.approx(-hh)
    assert h[0, 0] == pytest.approx(hh) and s[0] == pytest.approx(5.0 * hh)


# --------------------------------------------------------------------- validation


def test_validation_flags_structural_errors():
    bui = _building()
    bui["zones"][1]["name"] = bui["zones"][0]["name"]
    msgs = " ".join(str(i) for i in mz.validate_multizone_input(bui))
    assert "duplicate zone names" in msgs

    bui = _building()
    for s in bui["building_surface"]:
        if s["boundary"] == "INTERNAL":
            s["adjacent_zone"] = "Nowhere"
    issues = mz.validate_multizone_input(bui)
    assert any(i.level == "ERROR" and "Nowhere" in i.message for i in issues)
    with pytest.raises(ValueError, match="Nowhere"):
        _calc(bui)

    bui = _building()
    internal = [s for s in bui["building_surface"] if s["boundary"] == "INTERNAL"][0]
    bui["building_surface"].append(copy.deepcopy(internal))
    assert any("defined 2 times" in i.message for i in mz.validate_multizone_input(bui))


def test_zoning_rules_small_similar_and_large_opening():
    bui = _building()
    issues = mz.check_zoning_rules(bui)
    # same set-points and profiles in both zones -> may be combined (step 5)
    assert any("6.4.2.6" in i.message for i in issues)

    bui["zones"][0]["net_floor_area"] = 4.0           # < 5 % of the total (step 9)
    assert any("6.4.2.10" in i.message for i in mz.check_zoning_rules(bui))
    bui["zones"][0]["net_floor_area"] = 1.0           # < 1 % (step 10)
    assert any("6.4.2.11" in i.message for i in mz.check_zoning_rules(bui))

    bui = _building()
    bui["zone_openings"] = [{"zones": ["Z_ground", "Z_first"], "area_m2": 1.0}]
    assert any("6.4.2.4" in i.message for i in mz.check_zoning_rules(bui))
    bui["zone_openings"] = [{"zones": ["Z_ground", "Z_first"], "area_m2": 0.1}]
    assert not any("6.4.2.4" in i.message for i in mz.check_zoning_rules(bui))

    bui = _building()
    bui["zones"][0]["heating_setpoint"] = 25.0       # 5 K apart: no longer similar
    assert not any("6.4.2.6" in i.message for i in mz.check_zoning_rules(bui))


def test_zoning_warnings_are_emitted_by_the_calculation():
    bui = _partition_only(_building())
    bui["zones"][0]["net_floor_area"] = 4.0      # < 5 % of the useful floor area (step 9)
    with pytest.warns(UserWarning, match="6.4.2"):
        ISO52016.Temperature_and_Energy_needs_calculation_multizone(
            bui, path_weather_file=str(WEATHER), weather_source="epw",
            include_solar=False, include_internal_gains=False, include_ventilation=False,
            warmup_hours=744,
        )


# ------------------------------------------------------------------------- report


def test_report_describes_partitioning_and_input(full_year):
    hourly, annual = full_year
    rep = mz.multizone_report(_building(), hourly, annual)
    s = [x for x in _building()["building_surface"] if x["boundary"] == "INTERNAL"][0]
    assert rep["partitions"][0]["H_tr_zy_W_K"] == pytest.approx(s["u_value"] * s["area"])
    assert {z["zone"] for z in rep["zones"]} == {"Z_ground", "Z_first"}
    assert abs(rep["partition_exchange"][0]["closure_kWh"]) < 0.5
    assert "simultaneously" in rep["coupling_scheme"]
    text = mz.format_multizone_report(rep)
    assert "H_tr,zy" in text and "Coupling scheme" in text and "TOTAL" in text


# --------------------------------------------------- 6.4.6 Option B (Formulae 7-8)


def test_spatial_setpoint_reduction_formula_7_8():
    h_tr, h_ve, a = 300.0, 100.0, 100.0           # H_e,spec = 4 W/(m2 K)
    d = mz.spatial_setpoint_reduction(h_tr, h_ve, a, 20.0, 0.0)   # Table B.6 defaults
    num = (0.8 * 0.5) * (0.5 * 4.0) * 20.0
    assert d == pytest.approx(num / (0.5 * 4.0 + 2.0))
    # a warmer outdoor air leaves a smaller reduction, never a negative one
    assert mz.spatial_setpoint_reduction(h_tr, h_ve, a, 20.0, 25.0) == 0.0
    with pytest.raises(ValueError):
        mz.spatial_setpoint_reduction(h_tr, h_ve, 0.0, 20.0, 0.0)


def test_spatial_setpoint_adjustment_lowers_the_heating_need():
    cfg = json.loads((ROOT / "examples" / "FH_Poland_DC_baseline_two_floors.json").read_text(encoding="utf-8")) \
        if (ROOT / "examples" / "FH_Poland_DC_baseline_two_floors.json").exists() else None
    if cfg is None:
        pytest.skip("single-zone example not available")
    from pybuildingenergy.source.check_input import sanitize_and_validate_BUI

    def need(adjust):
        bui = copy.deepcopy(cfg["building"] if "building" in cfg and "building_surface" in cfg["building"] else cfg)
        if adjust:
            bui.setdefault("building_parameters", {})["spatial_setpoint_adjustment"] = {"enabled": True}
        bui, _ = sanitize_and_validate_BUI(bui, fix=True)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            _, annual = ISO52016.Temperature_and_Energy_needs_calculation(
                bui, weather_source="epw", path_weather_file=str(WEATHER))
        return float(annual["Q_H_annual_kWh"].iloc[0])

    assert need(True) < 0.97 * need(False)


def test_spatial_setpoint_reduction_reproduces_tr_52016_2_table_2():
    """ISO/TR 52016-2, 6.4.6.4, Table 2: theta_e = 5 C, theta_set = 20 C, Table B.6 values."""
    for h_e_spec, expected in ((2.0, 18.0), (0.8, 19.0), (0.4, 19.5)):
        d = mz.spatial_setpoint_reduction(h_e_spec * 100.0, 0.0, 100.0, 20.0, 5.0)
        assert 20.0 - d == pytest.approx(expected, abs=0.06), h_e_spec


# ------------------------------------------------------ latent balance per zone (6.5.14)


@pytest.fixture(scope="module")
def latent_run():
    bui = _building()
    bui["zones"][0]["moisture_production_kg_s"] = 2.0e-4        # ~0.7 kg/h in the ground zone only
    hourly, annual = _calc(bui, include_total=True)
    return bui, hourly, annual


def test_latent_columns_and_zone_specific_loads(latent_run):
    bui, hourly, annual = latent_run
    for z in ("Z_ground", "Z_first"):
        for col in ("X_int_kg_kg_da", "G_HU_kg_s", "G_DHU_kg_s", "Phi_HU_W", "Phi_DHU_W", "Q_latent_W"):
            assert f"{col}_{z}" in hourly.columns
    z = annual.set_index("zone")
    assert z.loc["Z_ground", "Q_latent_annual_kWh"] > z.loc["Z_first", "Q_latent_annual_kWh"]
    assert z.loc["TOTAL", "Q_latent_annual_kWh"] == pytest.approx(
        z.loc["Z_ground", "Q_latent_annual_kWh"] + z.loc["Z_first", "Q_latent_annual_kWh"])


def test_latent_balance_equals_the_single_zone_function(latent_run):
    from pybuildingenergy.source import utils as U

    bui, hourly, _ = latent_run
    zone = bui["zones"][0]
    mb = U._zone_moisture_balance(
        hourly["H_ve_Z_ground"].to_numpy(float), hourly["T_air_Z_ground"].to_numpy(float),
        hourly["T_ext"].to_numpy(float), hourly["RH_ext"].to_numpy(float),
        float(zone["volume"]), 3600.0, U.DEFAULT_RH_SET_HUMIDIFICATION_PCT,
        U.DEFAULT_RH_SET_DEHUMIDIFICATION_PCT, g_int_kg_s=zone["moisture_production_kg_s"],
    )
    assert hourly["Phi_DHU_W_Z_ground"].to_numpy() == pytest.approx(mb["phi_dhu_w"])
    assert hourly["X_int_kg_kg_da_Z_ground"].to_numpy() == pytest.approx(mb["x_int"])
    # the moisture content stays between the humidification and dehumidification limits
    assert (hourly["X_int_kg_kg_da_Z_ground"] <= mb["x_set_max"] + 1e-9).all()


def test_latent_can_be_disabled():
    bui = _partition_only(_building())
    hourly, annual = _steady(bui, include_latent=False)
    assert not any(c.startswith("Phi_DHU_W_") for c in hourly.columns)
    assert "Q_latent_annual_kWh" not in annual.columns
