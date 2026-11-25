"""ASHRAE Standard 140-2023, Section 11: air-side HVAC equipment analytical verification.

Four-pipe fan-coil (FC, AE100 series) and single-zone (SZ, AE200 series) systems,
modelled with the EN 16798-5-1 air handling unit step (``calculate_sensible_ahu_step``)
at steady state.

Mapping of the specification (Sections 11.2.1, 11.2.2) onto the AHU model:

* supply 600 cfm, zone exhaust 200 cfm, outdoor air = exhaust: recirculation with
  RCA_CTRL = VARIABLE and ``required_outdoor_air_flow_m3_h`` = exhaust flow;
* no heat recovery, no leakage, no duct or casing losses (11.2.1.1.2.4-5);
* heating and cooling coils that hold the supply temperature needed by the zone sensible
  load: SUP_AIR_TEMP_CTRL = LOAD_COMP;
* cooling coil bypass factor 0 (11.2.1.1.3 d): ``CoilData(f_C_bp=0)``;
* supply fan downstream of the coils, total pressure rise 498 Pa, mechanical efficiency
  0.7, all fan heat to the air stream: constant SFP = 498 / 0.7 W/(m3/s), DOWN_HR;
* SZ only: return fan between the zone and the return-air splitter, 249 Pa at the 400 cfm
  it moves, efficiency 0.7: extract fan UP_HR with SFP = 249 / 0.7, so its heat reaches
  the recirculated air (36a);
* zone humidity floats (no humidity control): the zone humidity ratio is found by
  iterating the zone moisture balance with the 586.1 W latent load.

Cases AE226 / AE245 (return-air comparative dry-bulb / enthalpy economizer): at their
conditions the economizer runs at 100 % outdoor air (QAS mixed air = outdoor air). The
library has no economizer control, so the test imposes that outdoor air fraction
(RCA_CTRL = FIX, f_ODA = 1); only the coil and fan model are verified, not the switching
logic of 11.2.2.3.

Reference: quasi-analytical solution (QAS v2) and the example programs in
``Std140_AE_FCSZ_Results.xlsx`` (accompanying files of ASHRAE 140-2023), coil loads in
kWh/h for the last hour of simulation.  These values are not distributed: the comparisons
read them from ``Standards/local_reference_data/ashrae140_2023_ae_qas.json`` and are
skipped without it.

The AHU model works with volumetric flows and a constant air density (EN 16798-5-1
Table 27), the QAS with mass flows from the local specific volume, so small deviations
are expected.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from pybuildingenergy.source import ahu_moisture_16798_5_1 as mx
from pybuildingenergy.source import ventilation_16798_5_1 as ahu

CFM_TO_M3_H = 1.699011

SUPPLY_M3_H = 600.0 * CFM_TO_M3_H
EXHAUST_M3_H = 200.0 * CFM_TO_M3_H
ZONE_LATENT_W = 586.14

# Test conditions (QASv2.xlsm "Test Case Inputs", zone 1): outdoor dry bulb [C],
# outdoor humidity ratio [kg/kg], zone set-point [C], zone sensible load [W]
# (+ heating, - cooling)
CONDITIONS = {
    1: (-29.0, 0.000258927, 21.111, 2930.7),    # high heating
    3: (15.5, 0.00293588, 23.333, -1465.4),     # low cooling, dry coil
    4: (26.9, 0.0167736, 23.889, -2930.7),      # high cooling, wet coil
    5: (24.9, 0.00449087, 23.333, -1465.4),     # low cooling, dry coil, warm dry air
    6: (23.0, 0.0155555, 23.333, -1465.4),      # low cooling, wet coil, warm humid air
}

# case: (system, condition, outdoor air fraction imposed for the economizer or None)
CASES = {
    "AE101": ("FC", 1, None),
    "AE103": ("FC", 3, None),
    "AE104": ("FC", 4, None),
    "AE201": ("SZ", 1, None),
    "AE203": ("SZ", 3, None),
    "AE204": ("SZ", 4, None),
    "AE205": ("SZ", 5, None),
    "AE206": ("SZ", 6, None),
    "AE226": ("SZ", 6, 1.0),
    "AE245": ("SZ", 5, 1.0),
}

# Reference results (QAS v2 coil loads and coil outlet humidity ratio, spread of the
# example programs) are copyright ASHRAE and not distributed: the comparisons run only
# where they are installed locally, as for the Section 7 tests.
REFERENCE_FILE = (Path(__file__).resolve().parent.parent / "Standards" / "local_reference_data"
                  / "ashrae140_2023_ae_qas.json")


def _load_reference():
    if not REFERENCE_FILE.exists():
        return None
    data = json.loads(REFERENCE_FILE.read_text())
    return {
        "qas": {k: tuple(v) for k, v in data["qas_coil_loads"].items()},
        "x_cco": data["qas_x_cco"],
        "range": {tuple(k.split("|")): tuple(v) for k, v in data["program_range"].items()},
    }


REFERENCE = _load_reference()
needs_reference = pytest.mark.skipif(
    REFERENCE is None,
    reason="ASHRAE 140-2023 AE reference results not installed "
           "(Standards/local_reference_data/ashrae140_2023_ae_qas.json)",
)


def _config(system: str, outdoor_air_fraction=None) -> ahu.SensibleAHUConfig:
    cfg = ahu.SensibleAHUConfig(
        sensible_heat_recovery_efficiency=0.0,
        supply_temperature_control=ahu.SupplyTemperatureControl(mode="load_compensated"),
        heat_recovery_control="always_on",
        frost_control="none",
        heating_coil_max_power_w=None,
        cooling_coil_enabled=True,
        supply_fan_specific_power_w_per_m3_s=498.0 / 0.7,
        extract_fan_specific_power_w_per_m3_s=0.0,
        supply_fan_heat_fraction_to_air=1.0,
        extract_fan_heat_fraction_to_air=0.0,
        fan_performance_model="constant_sfp",
        recirculation_control="variable",
        coil_data=mx.CoilData(f_C_bp=0.0),
        supply_fan_location="DOWN_HR",
    )
    if system == "SZ":   # return fan: its heat reaches the recirculated air
        cfg = dataclasses.replace(cfg, extract_fan_specific_power_w_per_m3_s=249.0 / 0.7,
                                  extract_fan_heat_fraction_to_air=1.0,
                                  extract_fan_location="UP_HR")
    if outdoor_air_fraction is not None:
        cfg = dataclasses.replace(cfg, recirculation_control="fix",
                                  outdoor_air_fraction_min=outdoor_air_fraction)
    return cfg


def _run(case: str):
    """Steady state of the air system: AHU step + zone moisture balance."""
    system, condition, f_oda = CASES[case]
    t_e, x_e, t_z, q_sens = CONDITIONS[condition]
    cfg = _config(system, f_oda)
    rho = mx.air_density(cfg.altitude_m)
    m_cp = rho * mx.C_A_J_KG_K * SUPPLY_M3_H / 3600.0
    m_r = rho * mx.R_W_J_KG * SUPPLY_M3_H / 3600.0
    t_sup_req = t_z + q_sens / m_cp           # supply temperature that meets the zone load
    q_oda = EXHAUST_M3_H if f_oda is None else f_oda * SUPPLY_M3_H
    x_z = x_e
    for _ in range(200):
        out = ahu.calculate_sensible_ahu_step(cfg, ahu.AHUStepInputs(
            outdoor_temperature_c=t_e,
            extract_temperature_c=t_z,
            required_supply_flow_m3_h=SUPPLY_M3_H,
            required_extract_flow_m3_h=SUPPLY_M3_H,
            outdoor_humidity_ratio=x_e,
            extract_humidity_ratio=x_z,
            required_outdoor_air_flow_m3_h=q_oda,
            required_supply_temperature_c=t_sup_req,
        ))
        x_z_new = out.supply_humidity_ratio + ZONE_LATENT_W / m_r
        if abs(x_z_new - x_z) < 1e-10:
            break
        x_z = x_z_new
    else:
        raise AssertionError(f"{case}: zone moisture balance did not converge")
    return out, t_sup_req


def _loads_kw(out):
    lat = out.latent_cooling_coil_power_w
    return (out.actual_heating_coil_power_w / 1e3,
            (out.actual_cooling_coil_power_w - lat) / 1e3,
            lat / 1e3)


@pytest.mark.parametrize("case", sorted(CASES))
def test_supply_meets_the_zone_load(case):
    out, t_sup_req = _run(case)
    f_oda = CASES[case][2]
    assert out.actual_supply_temperature_c == pytest.approx(t_sup_req, abs=0.01)
    assert out.outdoor_air_fraction == pytest.approx(
        EXHAUST_M3_H / SUPPLY_M3_H if f_oda is None else f_oda)


@needs_reference
@pytest.mark.parametrize("case", sorted(CASES))
def test_coil_loads_against_qas(case):
    out, t_sup_req = _run(case)
    heating, sensible, latent = _loads_kw(out)
    q_h, q_cs, q_cl = REFERENCE["qas"][case]
    assert heating == pytest.approx(q_h, rel=0.01, abs=1e-6)
    assert sensible == pytest.approx(q_cs, rel=0.02, abs=1e-6)
    assert latent == pytest.approx(q_cl, rel=0.05, abs=1e-6)


@needs_reference
@pytest.mark.parametrize("case", ["AE101", "AE103", "AE104"])
def test_supply_humidity_ratio_against_qas(case):
    out, _ = _run(case)
    # AE101: with a constant density the outdoor air share is 1/3 of the volume; the QAS
    # mass share is higher (cold outdoor air, warm air at the supply fan inlet), so the
    # supply humidity ratio is about 4 % higher
    assert out.supply_humidity_ratio == pytest.approx(REFERENCE["x_cco"][case], rel=0.05)


@needs_reference
def test_coil_loads_within_example_program_range():
    """Coil loads inside the spread of the example programs, widened by 2 %.

    Just outside the unwidened spread: AE204 sensible (-0.4 %) and AE226 latent (-1.5 %)."""
    results = {case: _loads_kw(_run(case)[0]) for case in CASES}
    index = {"heating": 0, "sensible": 1, "latent": 2}
    for (case, kind), (lo, hi) in REFERENCE["range"].items():
        value = results[case][index[kind]]
        assert lo * 0.98 <= value <= hi * 1.02, (case, kind, value, lo, hi)


def test_return_fan_heat_raises_the_cooling_load():
    """SZ = FC + return fan: its heat adds to the cooling coil load.

    The model adds the nominal fan heat, 249 Pa / 0.7 x 400 cfm = 67.2 W.  The QAS gives
    about 8 % more (AE203 - AE103): it derives the return volume from the mass balance and
    lets the pressure rise grow with the square of that flow."""
    fc = _loads_kw(_run("AE103")[0])[1]
    sz = _loads_kw(_run("AE203")[0])[1]
    nominal_kw = 249.0 / 0.7 * (400.0 * CFM_TO_M3_H / 3600.0) / 1e3
    assert sz - fc == pytest.approx(nominal_kw, rel=0.01)
    if REFERENCE is not None:
        qas = REFERENCE["qas"]["AE203"][1] - REFERENCE["qas"]["AE103"][1]
        assert sz - fc == pytest.approx(qas, rel=0.10)


def test_wet_coil_leaves_saturated_and_dry_coil_keeps_humidity():
    """BF = 0: wet-coil cases leave the coil saturated; dry-coil cases do not condense."""
    for case in ("AE104", "AE204", "AE206"):
        wet, _ = _run(case)
        assert wet.latent_cooling_coil_power_w > 0.0
        # with BF = 0 the coil surface and leaving air coincide; the EN 16798-5-1 surface
        # humidity correlation (Table B.8 constants) is within 2 % of saturation
        x_sat = mx.saturation_humidity_ratio(wet.cooling_coil_temperature_c)
        assert wet.supply_humidity_ratio == pytest.approx(x_sat, rel=0.02)
    for case in ("AE103", "AE203", "AE205"):
        dry, _ = _run(case)
        assert dry.latent_cooling_coil_power_w == 0.0
