"""ISO 52016-1:2017's own verification cases (clause 7.2.2-7.2.4, Table 27):
600, 640, 900, 940, 600FF and 900FF.

This is a *different*, non-comparable sibling of
``examples/ashrae140_bestest/``: both start from the same BESTEST Case
600/900 geometry and materials, but ISO 52016-1's own worked verification
example (clause 7.2.2) deliberately *adapts* several inputs away from
ASHRAE 140's exact boundary conditions, and its own reference results
(clause 7.2.4, Tables 28-30) are computed under those adapted conditions
-- not ASHRAE 140's Table B8 band. Mixing the two is a category error:
see the module docstring in ``ashrae140_bestest/case_definitions.py`` and
both package READMEs for the full explanation.

Adaptations this module makes, each cited to its ISO 52016-1:2017 clause:

* **Thermostat control variable**: clause 7.2.2.1 states whole-model
  verification "considers the calculation of the operative temperatures
  and the sensible energy needs" -- i.e. the *default* pybuildingenergy
  control (``hvac_control_variable="operative"``), not the "air" override
  ``ashrae140_bestest`` uses to match ASHRAE 140's EnergyPlus-style ideal
  loads systems. This module does not pass ``hvac_control_variable`` at
  all, so it uses that default.
* **Floor boundary** (clause 7.2.2.3, Table 23 footnote a): "the external
  environment of the construction is assumed to be the outdoor air"
  -- not a fixed-10degC ground boundary. The floor surface here uses
  ``boundary: "OUTDOORS"`` with no ISO 13370 ground model involved. Its
  ``sky_view_factor`` is a negligible 1e-6, not 0: the single-zone engine
  treats a horizontal opaque surface with sky_view_factor == 0 as ground
  contact (this harness did so, unintentionally, before 2026-10-14).
  Clause 7.2.2.8 gives a sky view only for the roof and walls, and the
  underside of the floor receives no solar radiation (as in the ASHRAE 140
  case it derives from), so its solar absorptance is 0. Its coefficients
  are set explicitly to the Table 25 "downwards" values (h_ci = 0.7) --
  with tilt 0 the engine would otherwise apply the "upwards" roof value --
  and its U-value includes r_se so that the construction resistance is
  exactly the Table 23/24 layer resistance.
* **Thermostat control strategy** (clause 7.2.2.15): "continuous" 20/27
  degC (Tests 1 and 3: 600, 900); "intermittent" with heating at 10 degC
  from 23:00 to 07:00 and 20 degC from 07:00 to 23:00, cooling 27 degC at
  all hours (Tests 2 and 4: 640, 940); free floating with no heating or
  cooling (Tests 5 and 6: 600FF, 900FF).
* **Infiltration**: clause 7.2.2.14 gives 0.41 ACH continuously (not
  ASHRAE 140's 0.5 ACH) -- "the specified infiltration rates have been
  adjusted with factor 0.822 to yield mass flows equivalent to those
  occurring at the specified altitude of the weather station at 1609 m".
* **Window solar correction factor**: clause 7.2.2.6 uses ISO 52016-1's
  own generic Table B.22 default, Fw=0.90 (not ASHRAE 140's native 0.907
  that ``ashrae140_bestest`` uses) -- ``g_gl;n=0.789`` and ``Uw=2.984
  W/(m2 K)`` are unchanged (both modules use the same documented values
  for this window).
* **construction_class per surface**: unchanged from ``ashrae140_bestest``
  (wall/roof "Evenly (D)" = class_d, floor "Internal (I)" = class_i for
  Case 600; wall/floor class_i, roof class_d for Case 900) -- this is the
  *same* ISO 52016-1 Tables 23-24 classification both modules share.

Weather data: clause 7.2.2.12 points to ISO 52016-1's own accompanying
climate-data spreadsheet (ISO copyright, not distributed here; see
iso_climate_loader.py). When it is available locally the runner uses it;
otherwise it falls back to the Denver TMY3 EPW of ``ashrae140_bestest``,
which matches the monthly climate summary of Table 26 only in order of
magnitude -- results obtained with it are an approximation, not the
verification case.
"""

from pathlib import Path

# Geometry, materials and window properties are identical to
# ashrae140_bestest's Case 600/900 (both are ISO 52016-1 Tables 23-24 /
# clause 7.2.2.6, see that module's docstring for the full sourcing and
# derivation) -- duplicated here as plain constants rather than imported
# across the two example packages, to keep each one runnable standalone
# (this file is deliberately named differently from
# ashrae140_bestest/case_definitions.py so the two can be imported in the
# same pytest session without one shadowing the other).

EXAMPLES_DIR = Path(__file__).resolve().parent
WEATHER_FILE = EXAMPLES_DIR.parent / "ashrae140_bestest" / "weather" / "USA_CO_Denver.Intl.AP.725650_TMY3.epw"

FLOOR_AREA_M2 = 48.0
VOLUME_M3 = 129.6
PERIMETER_M = 28.0
WALL_HEIGHT_M = 2.7

LATITUDE_DEG = 39.76
LONGITUDE_DEG = -104.86

U_LTWALL = 0.50974
U_LTROOF = 0.31914
U_LTFLOOR = 0.039331
U_HWWALL = 0.50753
U_HWFLOOR = 0.039343

K_LTWALL = 14534.28
K_LTROOF = 18169.98
K_LTFLOOR = 19500.0
K_HWWALL = 145154.0
K_HWFLOOR = 112000.0

U_WINDOW = 2.984
G_WINDOW = 0.789
FW_ISO_DEFAULT = 0.90  # ISO 52016-1 Table B.22 generic default (clause 7.2.2.6)

SOLAR_ABSORPTANCE_OPAQUE = 0.6
INTERNAL_GAIN_W = 200.0
HEATING_SETPOINT_C = 20.0
COOLING_SETPOINT_C = 27.0
INFILTRATION_ACH_ISO = 0.41  # clause 7.2.2.14, altitude-adjusted

# Floor to outdoor air (Table 23 footnote a), Table 25 "downwards" coefficients.
H_CI_DOWN, H_RI, H_CE, H_RE = 0.7, 5.13, 20.0, 4.14
R_LTFLOOR = 0.025 / 0.14 + 1.003 / 0.04  # Table 23
R_HWFLOOR = 0.080 / 1.13 + 1.007 / 0.04  # Table 24

CONTROL_STRATEGIES = ("continuous", "intermittent", "free_floating")
CASE_IDS = {  # Table 27
    ("light", "continuous"): "600", ("light", "intermittent"): "640",
    ("light", "free_floating"): "600FF", ("heavy", "continuous"): "900",
    ("heavy", "intermittent"): "940", ("heavy", "free_floating"): "900FF",
}


def _opaque_wall(
    name: str, azimuth_deg: float, area_m2: float, u_value: float, kappa_m: float,
    construction_class: str,
) -> dict:
    return {
        "name": name,
        "type": "opaque",
        "boundary": "OUTDOORS",
        "area": area_m2,
        "sky_view_factor": 0.5,
        "u_value": u_value,
        "solar_absorptance": SOLAR_ABSORPTANCE_OPAQUE,
        "thermal_capacity": kappa_m,
        "construction_class": construction_class,
        "orientation": {"azimuth": azimuth_deg, "tilt": 90.0},
        "name_adj_zone": None,
        "height": WALL_HEIGHT_M,
        "length": area_m2 / WALL_HEIGHT_M,
    }


def build_case(mass: str = "light", control: str = "continuous") -> dict:
    """Build the ISO 52016-1-adapted lightweight (``mass="light"``: 600,
    640, 600FF) or heavyweight (``mass="heavy"``: 900, 940, 900FF)
    building_object for the thermostat ``control`` strategy of clause
    7.2.2.15 ("continuous", "intermittent" or "free_floating"). Run with
    ``hvac_control_variable`` left at its default ("operative") -- do not
    pass "air" here, that would silently turn this back into the
    ASHRAE-target case.
    """
    if control not in CONTROL_STRATEGIES:
        raise ValueError(f"control must be one of {CONTROL_STRATEGIES}")
    if mass == "light":
        u_wall, k_wall = U_LTWALL, K_LTWALL
        r_floor, k_floor = R_LTFLOOR, K_LTFLOOR
        case_name = "ISO52016_Case600"
        wall_class = "class_d"
    elif mass == "heavy":
        u_wall, k_wall = U_HWWALL, K_HWWALL
        r_floor, k_floor = R_HWFLOOR, K_HWFLOOR
        case_name = "ISO52016_Case900"
        wall_class = "class_i"
    else:
        raise ValueError("mass must be 'light' (Case 600) or 'heavy' (Case 900)")

    window_area = 12.0

    surfaces = [
        _opaque_wall("North wall", 0.0, 8 * WALL_HEIGHT_M, u_wall, k_wall, wall_class),
        _opaque_wall("South wall", 180.0, 8 * WALL_HEIGHT_M - window_area, u_wall, k_wall, wall_class),
        _opaque_wall("East wall", 90.0, 6 * WALL_HEIGHT_M, u_wall, k_wall, wall_class),
        _opaque_wall("West wall", 270.0, 6 * WALL_HEIGHT_M, u_wall, k_wall, wall_class),
        {
            "name": "Roof",
            "type": "opaque",
            "boundary": "OUTDOORS",
            "area": FLOOR_AREA_M2,
            "sky_view_factor": 1.0,
            "u_value": U_LTROOF,
            "solar_absorptance": SOLAR_ABSORPTANCE_OPAQUE,
            "thermal_capacity": K_LTROOF,
            "construction_class": "class_d",
            "orientation": {"azimuth": 0.0, "tilt": 0.0},
            "name_adj_zone": None,
            "height": 6.0,
            "length": 8.0,
        },
        {
            # Clause 7.2.2.3, Table 23 footnote a: modelled as opaque to
            # OUTDOOR AIR, not a fixed-temperature ground boundary.
            "name": "Floor",
            "type": "opaque",
            "boundary": "OUTDOORS",
            "area": FLOOR_AREA_M2,
            "sky_view_factor": 1.0e-6,  # not 0: that would mean ground contact
            "u_value": 1.0 / (1.0 / (H_CI_DOWN + H_RI) + r_floor + 1.0 / (H_CE + H_RE)),
            "convective_heat_transfer_coefficient_internal": H_CI_DOWN,
            "radiative_heat_transfer_coefficient_internal": H_RI,
            "convective_heat_transfer_coefficient_external": H_CE,
            "radiative_heat_transfer_coefficient_external": H_RE,
            "solar_absorptance": 0.0,  # underside: no solar
            "thermal_capacity": k_floor,
            "construction_class": "class_i",
            "orientation": {"azimuth": 0.0, "tilt": 0.0},
            "name_adj_zone": None,
            "height": 6.0,
            "length": 8.0,
        },
        {
            "name": "South window",
            "type": "transparent",
            "boundary": "OUTDOORS",
            "area": window_area,
            "sky_view_factor": 0.5,
            "u_value": U_WINDOW,
            "g_value": G_WINDOW,
            "solar_transmittance_correction_factor": FW_ISO_DEFAULT,
            "solar_absorptance": 0.0,
            "thermal_capacity": 0.0,
            "orientation": {"azimuth": 180.0, "tilt": 90.0},
            "name_adj_zone": None,
            "height": 2.0,
            "width": window_area / 2.0,
            "frame_area_fraction": 0.0,
            "shading": False,
        },
    ]

    case_name = f"ISO52016_Case{CASE_IDS[(mass, control)]}"
    bui = {
        "building": {
            "name": case_name,
            "azimuth_relative_to_true_north": 0,
            "latitude": LATITUDE_DEG,
            "longitude": LONGITUDE_DEG,
            "exposed_perimeter": PERIMETER_M,
            "height": WALL_HEIGHT_M,
            "wall_thickness": 0.1,
            "n_floors": 1,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "net_floor_area": FLOOR_AREA_M2,
            "construction_class": "class_i",  # required fallback; unused (see case_definitions docstring)
            "construction_year": "2010-today",
            "country": "USA",
            "volume": VOLUME_M3,
        },
        "building_surface": surfaces,
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": HEATING_SETPOINT_C,
                "heating_setback": HEATING_SETPOINT_C,
                "cooling_setpoint": COOLING_SETPOINT_C,
                "cooling_setback": COOLING_SETPOINT_C,
                "units": "degC",
            },
            "system_capacities": {
                "heating_capacity": 1.0e9,
                "cooling_capacity": 1.0e9,
                "units": "W",
            },
            "internal_gains": [
                {"name": "occupants", "full_load": 0.0,
                 "weekday": [1.0] * 24, "weekend": [1.0] * 24},
                {"name": "appliances", "full_load": INTERNAL_GAIN_W / FLOOR_AREA_M2,
                 "weekday": [1.0] * 24, "weekend": [1.0] * 24},
                {"name": "lighting", "full_load": 0.0,
                 "weekday": [1.0] * 24, "weekend": [1.0] * 24},
            ],
            "construction": {
                "wall_thickness": 0.1,
                "thermal_bridge_heat_W_K": 0.0,
            },
            "ventilation": {
                "components": [
                    {
                        "name": "infiltration",
                        "ventilation_type": "constant_ach",
                        "air_changes_per_hour": INFILTRATION_ACH_ISO,
                    }
                ]
            },
            # No ground_temperature_model override needed: the floor is an
            # OUTDOORS-boundary surface here, not "GROUND", so ISO 13370 is
            # never invoked.
        },
        "units": {
            "area": "m2", "u_value": "W/m2K", "thermal_capacity": "J/m2K",
        },
    }
    if control == "intermittent":
        sp = bui["building_parameters"]["temperature_setpoints"]
        sp["heating_setback"] = 10.0
        # Index = hour starting at hh:00; 20 degC from 07:00 to 23:00.
        day = [0.0] * 7 + [1.0] * 16 + [0.0]
        bui["building_parameters"]["heating_profile"] = {"weekday": day, "holiday": day}
    elif control == "free_floating":
        bui["building_parameters"]["system_capacities"].update(
            {"heating_capacity": 0.0, "cooling_capacity": 0.0})
    return bui
