"""Annotated multizone input: every field to declare (see docs/multizone_declaration.md).

Hourly multizone calculation (EN ISO 52016-1) with coupled thermal zones, an inter-zone
air flow, simplified adaptive windows (EN ISO 52016-3) and a per-zone EN 15316-2
emission system for heating and cooling.

The base geometry comes from ``FH_Poland_DC_multizone_ideal_envelope.json``: a terraced
tenement in Poland (50.04 N, 19.56 E) with two zones, Z_ground and Z_first, and one floor
between them.  The script adds or overrides the fields the declaration needs and labels
each one ``[DECLARE]``.

ILLUSTRATIVE VALUES: the emitter data and the adaptive-state properties are NOT product
data.  For a real case, replace them and fill ``DATA_SOURCES`` with the origin and the
reference conditions of every value.

Run:
    python examples/multizone_declaration_example.py            # PVGIS climate at the site
    python examples/multizone_declaration_example.py --athens   # offline: Athens EPW
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

from pybuildingenergy.source.adaptive_envelope_workflow_52016_3 import (
    compare_adaptive_with_reference_multizone,
)
from pybuildingenergy.source.composer import simulate_multizone_emission
from pybuildingenergy.source.multizone_52016_1 import (
    format_multizone_report,
    multizone_report,
    validate_multizone_input,
)

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--athens", action="store_true",
                    help="use the Athens EPW (offline); the case becomes 'Polish building in the Athens climate'")
args = parser.parse_args()

config = json.loads((HERE / "FH_Poland_DC_multizone_ideal_envelope.json").read_text(encoding="utf-8"))
bui = config["building"]
site = bui["building"]

# ------------------------------------------------------------------ 0. site and climate
# [DECLARE] location, climate source and calendar must belong to the same site.
if args.athens:
    WEATHER = {"source": "epw", "file": str(HERE / "2020_Athens.epw")}
    CASE_LABEL = (f"Building located in {site['country']} ({site['latitude']} N, {site['longitude']} E) "
                  "SIMULATED IN THE ATHENS CLIMATE (2020 EPW): demonstration only, not a real case")
else:
    WEATHER = {"source": "pvgis", "file": None}               # hourly data at the building coordinates
    CASE_LABEL = (f"Building in {site['country']} ({site['latitude']} N, {site['longitude']} E), "
                  "PVGIS climate at the same coordinates")
config["weather"] = WEATHER
engine_weather = (
    {"weather_source": "epw", "path_weather_file": WEATHER["file"]}
    if WEATHER["source"] == "epw" else {"weather_source": "pvgis"}
)

# --------------------------------------------------------------------------- 1. zones
ZONE_USE = "Residential_apartment"                           # [DECLARE] use of both zones
ADAPTIVE_SCENARIO = "residential"                            # [DECLARE] control scenario,
#                                                              consistent with the zone use
for zone in bui["zones"]:
    zone["building_type_class"] = ZONE_USE
    zone["heating_setpoint"] = 20.0                            # [DECLARE] degC
    zone["heating_setback"] = 18.0                             # [DECLARE] degC, outside occupancy
    zone["cooling_setpoint"] = 26.0                            # [DECLARE] degC
    zone["cooling_setback"] = 30.0                             # [DECLARE] degC
    zone["heating_capacity"] = 20_000.0                        # [DECLARE] W set for the zone; the
    zone["cooling_capacity"] = 20_000.0                        #   emitter may lower it (section 5)
    zone["cooling_profile"] = {"weekday": [1] * 24, "weekend": [1] * 24}   # [DECLARE] availability
    zone["free_floating"] = False                              # [DECLARE] conditioned / free-floating
    # net_floor_area, volume, internal_gains[] (full_load + weekday/weekend profiles) and
    # the ventilation/infiltration data are already in the file: declare them as given.
SET_CAPACITIES = {z["name"]: (z["heating_capacity"], z["cooling_capacity"]) for z in bui["zones"]}

# ------------------------------------------------------------- 2. partitions between zones
# Each shared wall/floor appears ONCE: boundary INTERNAL, zone + adjacent_zone, area,
# orientation, U, areal heat capacity, internal surface coefficients.
for s in bui["building_surface"]:
    if s.get("boundary") == "INTERNAL":
        s.setdefault("construction_class", "class_i")          # [DECLARE] mass distribution

# ------------------------------------------------------------- 3. inter-zone air flow
# One direction per pair of zones; sensible heat only (no moisture transport between zones).
bui["interzone_airflows"] = [{
    "from": "Z_ground",                                       # [DECLARE] source zone
    "to": "Z_first",                                          # [DECLARE] receiving zone
    "q_v_m3_h": 60.0,                                         # [DECLARE] or q_v_m3_s / ach_from_zone
    "makeup_from_outdoor": True,                              # [DECLARE] make-up air of the source
    "follow_ventilation_profile": True,                       # [DECLARE] time profile
}]

# ----------------------------------------------------------- 4. windows / adaptive facade
# Geometry: the base file gives the glazed AREA and the WIDTH and HEIGHT of the glazed band
# of each facade (width x height is about twice the area).  The shading calculation uses
# width and height, so the example declares an equivalent window with the same width and
# height = area / width.  [DECLARE] for a real case: the actual window dimensions.
geometry_notes = []
for s in bui["building_surface"]:
    if s.get("type") != "transparent":
        continue
    s.setdefault("frame_area_fraction", 0.25)                 # [DECLARE] F_fr
    s.setdefault("solar_transmittance_correction_factor", 0.90)  # [DECLARE] F_w
    w, h, a = float(s["width"]), float(s["height"]), float(s["area"])
    if w > 0 and abs(w * h - a) > 0.01 * a:
        geometry_notes.append(f"{s['name']}: A = {a} m2, W x H = {w} x {h} = {w*h:.2f} m2 "
                              f"-> H set to {a/w:.3f} m")
        s["height"] = a / w

ADAPTIVE_WINDOWS = ["Transparent south surface_Z_ground", "Transparent south surface_Z_first"]
# [DECLARE] per state: U, g, tau_v of the COMBINED glazing + roller blind in that state
STATES = {
    "retracted": None,                                        # = the window itself (below)
    "extended": {"u_value": 0.60, "g_value": 0.10, "tau_v": 0.02},
}
DATA_SOURCES = {                                              # [DECLARE] origin + conditions
    "retracted": "glazing data of the base file (assumed; EN 410 / EN 673 conditions not stated)",
    "extended": "ILLUSTRATIVE value, not product data: replace with manufacturer/test/"
                "calculation (EN 13363-2 / ISO 52022-3) for the combined system",
}
for s in bui["building_surface"]:
    if s["name"] in ADAPTIVE_WINDOWS:
        s["adaptive_envelope"] = {
            "type": "roller_blind",                           # [DECLARE] device
            "position": "external",                           # [DECLARE] external/internal/between
            "control_type": "automated",                      # [DECLARE] manual/motorized/automated/passive
            "space_category": ADAPTIVE_SCENARIO,              # [DECLARE] control scenario
            "system": "heating_cooling",                      # [DECLARE] services the control serves
            "states": {
                "retracted": {"u_value": s["u_value"], "g_value": s["g_value"], "tau_v": 0.80},
                "extended": dict(STATES["extended"]),
            },
            # [DECLARE] visual comfort: daylight first, artificial light only up to the target
            "lighting": {
                "target_lx": 200.0,                           # minimum on the working plane
                "power_W_m2": 5.0,                            # installed lighting power density
                "control": "dimming",                         # dimming / on_off
                "daylight_coefficient": 0.03,                 # ILLUSTRATIVE: lx on the working
                #   plane per lx of vertical outdoor illuminance and per unit tau_V; or give
                #   control.daylight = {a_dayl, b_dayl, H_gl_1m} (Formula 4)
            },
            # optional [DECLARE] when used: hysteresis, environment_variable, daytime_hours,
            # sleeping_hours, daylight, glare_policy, glare_limit_lx, annex_c_edition
        }

# --------------------------------------------------------------- 5. emission per zone
# EN 15316-2 per service; nominal_power_kW: 0 = no emitter, omitted = unlimited.
def fan_coil(p_heat_kw, p_cool_kw):
    common = {
        "stratification_K": 0.2, "control_K": 0.4, "radiation_K": 0.0,
        "hydraulic_balancing_K": 0.1, "room_automation_K": -0.2, "embedded_K": 0.0,
        "fan_power_W": 25.0, "fan_count": 4,                  # [DECLARE] fans (always counted)
        "control_power_W": 1.0, "control_count": 4,           # [DECLARE] electric controls ...
        "electric_control_aux": True,                         # ... counted only when True
        "convective_fraction": 0.95, "backup_available": False,
    }
    return {
        "demand_unit": "kWh",
        "heating": {**common, "nominal_power_kW": p_heat_kw},  # [DECLARE] at rated conditions
        "cooling": {**common, "nominal_power_kW": p_cool_kw, "control_K": -0.4,
                    "stratification_K": -0.2},
    }

emission_by_zone = {                                          # [DECLARE] terminal per zone
    "Z_ground": fan_coil(12.0, 8.0),                          #   ILLUSTRATIVE ratings
    "Z_first": fan_coil(12.0, 8.0),
}

# ------------------------------------------------------------------- checks and run
print("CASE:", CASE_LABEL)
for note in geometry_notes:
    print("geometry:", note)
for issue in validate_multizone_input(bui):
    print("check:", issue)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = simulate_multizone_emission(
        config,
        emission_by_zone,
        emission_inc="recalculate",                           # [DECLARE] M2-2 recalculation
        couple_emitter_capacity=True,                         # [DECLARE] emitter limit in the solver
        unlisted_zones="free_floating",                       # [DECLARE] zones without emitter
    )
    # EN ISO 52016-3 6.10 / clause 8: same building, climate and profiles, adaptive
    # windows replaced by their fixed window.  The emitter limits are applied to both runs.
    applied_bui = copy.deepcopy(bui)
    for z in applied_bui["zones"]:
        z.update(result.applied_capacities.get(z["name"], {}))
    comparison = compare_adaptive_with_reference_multizone(applied_bui, **engine_weather)

print(result.report())

print("\nCapacities [W]            set (input)            applied in the solver")
for zn, (ch, cc) in SET_CAPACITIES.items():
    ap = result.applied_capacities.get(zn, {})
    print(f"  {zn:10s} H {ch:9.0f}  C {cc:9.0f}    H {ap.get('heating_capacity', ch):9.0f}  "
          f"C {ap.get('cooling_capacity', cc):9.0f}")
print("free-floating zones:", result.free_floating_zones or "none")

# Comfort: with the emitter limit in the solver a shortage appears as a temperature off
# the set-point, not as unmet energy.  Report both.
print("\nComfort (hours outside the set-points, all hours)")
for zn in SET_CAPACITIES:
    t = result.hourly[f"T_op_{zn}"]
    z = next(z for z in bui["zones"] if z["name"] == zn)
    em = result.emission[zn].summary
    unmet = {k: round(v, 1) for k, v in em.items() if "unmet" in k and isinstance(v, (int, float))}
    print(f"  {zn}: T_op < {z['heating_setpoint']} C: {(t < z['heating_setpoint'] - 0.5).sum()} h, "
          f"> {z['cooling_setpoint']} C: {(t > z['cooling_setpoint'] + 0.5).sum()} h; unmet {unmet}")

print("\nAdaptive vs reference (EN ISO 52016-3 Table 6, adaptive - reference, per zone)")
for zn, k in comparison.kpis.items():
    print(f"  {zn} (comfort limits {comparison.comfort_limits[zn]}):")
    for name, value in k.items():
        print(f"    {name:40s} {value:12.3f}")
print("State statistics:", json.dumps(comparison.statistics, indent=1))

# Daylight and artificial light per control state (occupied, awake hours)
from pybuildingenergy.source.adaptive_envelope_52016_3 import adaptive_lighting_by_state
import pandas as pd
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print("\nIlluminance per control state:")
    print(adaptive_lighting_by_state(comparison.adaptive_hourly).round(1).to_string(index=False))
print("Clause 8 report:", json.dumps(comparison.report, indent=1, default=str)[:4000])
print("Data sources of the adaptive states:", json.dumps(DATA_SOURCES, indent=1))

# Multizone report built from the input with the APPLIED capacities
print(format_multizone_report(multizone_report(copy.deepcopy(applied_bui))))
for w in caught:                                              # report the warnings as well
    print("warning:", w.message)
