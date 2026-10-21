"""ASHRAE Standard 140-2023 Section 7.2.3 in-depth cases.

The 25 cases 195, 200, 210, 215, 220, 230, 240, 250, 270, 280, 290, 300, 310,
320, 395, 400, 410, 420, 430, 440, 450, 460, 470, 800 and 810 isolate single
modelling features (surface infrared exchange, solid conduction, infiltration,
internal gains, exterior and interior solar absorptance, shading, surface
coefficients, thermal mass) by changing one input at a time from a base case
(Informative Table 7-33). The standard gives no acceptance ranges for them: they
are diagnostic, and the useful output is the *difference* between a case and its
base (``DIFFERENCES``).

Mapping onto the engine (the general conventions are in ``case_definitions.py``):

* High-conductance (HC) wall elements (Tables 7-35 to 7-37) replace the 12 m2 of
  south window in the 200-series and 400-series. They are opaque surfaces with
  the layers of Table 7-36 (3.048 mm panel, 12 mm air gap, 3.048 mm panel), zero
  shortwave transmittance and their own construction class, so that they are not
  aggregated with the walls.
* Infrared emittance. The engine has no emittance input: it derives the interior and
  exterior radiative coefficients from them, h_ri = 5.13 x eps/0.9 and
  h_re = 4.14 x eps/0.9 (eps = 0.9 in Case 600). Cases 210/215/200/195 use eps = 0.1 on
  the interior and/or exterior side of every opaque surface, with the convective
  coefficients of Tables 7-40 and 7-41.
* Exterior solar absorptance 0.1 (Case 220 and its descendants), 0.9 (Case 250) or
  0.6 (Cases 430, 800); the raised floor receives no solar radiation in any case.
* Interior solar absorptance (Cases 270, 280, 440, 810) has no engine input: it is
  represented by the interior solar distribution fractions of Tables 7-43 (270),
  7-44 (280) and 7-45 (300), which Sections 7.2.3.9.3.2, 7.2.3.10.2.2 and
  7.2.3.12.2.2 permit when the program does not compute the distribution itself.
  With ``solar_distribution="area"`` the engine distribution is used instead and
  the interior absorptance cannot be told apart (270 = 280, 440 = 600, 810 = 900).
* Combined coefficients (Cases 450, 460, 470; Table 7-46) are set as convective
  coefficients with a null radiative coefficient, as the standard's informative
  note describes.
* Thermostat: "20,20" (heating and cooling setpoint 20 degC) in the 200-series,
  "20,27" in the 400-series and in Case 320.
* Infiltration is altitude-corrected (``ALTITUDE_FACTOR``) and may be zero.
"""

from copy import deepcopy

from case_definitions import (
    ALTITUDE_FACTOR, FLOOR_AREA_M2, HWFLOOR, HWWALL, LTFLOOR,
    LTROOF, LTWALL, OVERHANG_AT_ROOF, SOLAR_DISTRIBUTION_MODES, WALL_HEIGHT_M,
    WINDOW_600, H_RE_ENGINE, H_RI_ENGINE, _opaque, _window, _with_coefficients,
    build_case, interior_solar_distribution, layers_resistance,
)

# Table 7-36: opaque panel / air gap / opaque panel (inside to outside).
HC_LAYERS = (
    (0.003048, 1.00, 2470.0, 750.0),
    (0.012, 0.0625, 1.292498, 1006.103271),
    (0.003048, 1.00, 2470.0, 750.0),
)
HC_AREA_M2 = 12.0  # two 2 m x 3 m units (Table 7-35)

# Convective coefficients [W/(m2 K)] per surface kind.
# Interior: Table 7-9 (Case 600, ceiling = "roof") and Tables 7-39 / 7-40; exterior: Table 7-7
# and Tables 7-38 / 7-41.
CONV_INT_DEFAULT = {"wall": 2.2, "roof": 1.8, "floor": 2.2, "hc": 2.4, "window": 2.4}
CONV_INT_TABLE_7_40 = {"wall": 1.9, "roof": 1.9, "floor": 1.0, "hc": 2.6}
CONV_EXT_DEFAULT = {"wall": 11.9, "roof": 14.4, "floor": 0.8, "hc": 8.0, "window": 8.0}
CONV_EXT_TABLE_7_41 = {"wall": 11.8, "roof": 14.4, "floor": 0.9, "hc": 7.9}
# Table 7-46: constant combined coefficients, applied with a null radiative coefficient.
COMB_INT_TABLE_7_46 = {"wall": 1.8, "roof": 1.7, "floor": 3.7, "window": 4.5}
COMB_EXT_TABLE_7_46 = {"wall": 21.6, "roof": 21.8, "floor": 5.2, "window": 17.8}

IN_DEPTH_CASES = (195, 200, 210, 215, 220, 230, 240, 250, 270, 280, 290, 300, 310,
                  320, 395, 400, 410, 420, 430, 440, 450, 460, 470, 800, 810)

_BASE_200 = dict(mass="light", fill="hc", ach=0.0, gains=0.0, ext_abs=0.1, tstat="20,20",
                 eps_int=0.9, eps_ext=0.9, int_coef="default", ext_coef="default", dist=None)
_BASE_400 = {**_BASE_200, "tstat": "20,27"}
_BASE_600 = {**_BASE_200, "fill": "south", "ach": 0.5, "gains": 200.0, "ext_abs": 0.6,
             "tstat": "20,27"}

CASE_SPECS = {
    220: dict(_BASE_200),
    210: {**_BASE_200, "eps_int": 0.1, "int_coef": "table_7_40"},
    215: {**_BASE_200, "eps_ext": 0.1, "ext_coef": "table_7_41"},
    200: {**_BASE_200, "eps_int": 0.1, "eps_ext": 0.1,
          "int_coef": "table_7_40", "ext_coef": "table_7_41"},
    195: {**_BASE_200, "fill": "solid", "eps_int": 0.1, "eps_ext": 0.1,
          "int_coef": "table_7_40", "ext_coef": "table_7_41"},
    230: {**_BASE_200, "ach": 1.0},
    240: {**_BASE_200, "gains": 200.0},
    250: {**_BASE_200, "ext_abs": 0.9},
    270: {**_BASE_200, "fill": "south", "dist": "270"},
    280: {**_BASE_200, "fill": "south", "dist": "280"},
    290: {**_BASE_200, "fill": "south", "dist": "270", "shade": "overhang"},
    300: {**_BASE_200, "fill": "eastwest", "dist": "300"},
    310: {**_BASE_200, "fill": "eastwest", "dist": "300", "shade": "overhang_fins"},
    320: {**_BASE_200, "fill": "south", "dist": "270", "tstat": "20,27"},
    400: dict(_BASE_400),
    395: {**_BASE_400, "fill": "solid"},
    410: {**_BASE_400, "ach": 0.5},
    420: {**_BASE_400, "ach": 0.5, "gains": 200.0},
    430: {**_BASE_400, "ach": 0.5, "gains": 200.0, "ext_abs": 0.6},
    440: {**_BASE_600, "dist": "280"},
    450: {**_BASE_600, "int_coef": "table_7_46", "ext_coef": "table_7_46"},
    460: {**_BASE_600, "int_coef": "table_7_46"},
    470: {**_BASE_600, "ext_coef": "table_7_46"},
    800: {**_BASE_400, "mass": "heavy", "ach": 0.5, "gains": 200.0, "ext_abs": 0.6},
    810: {**_BASE_600, "mass": "heavy", "dist": "280"},
}

# (case, base case, tested feature): the differences that Informative Table 7-33 designates.
DIFFERENCES = (
    (210, 220, "interior IR emittance 0.9 -> 0.1"),
    (215, 220, "exterior IR emittance 0.9 -> 0.1"),
    (200, 210, "exterior IR emittance with interior 0.1"),
    (200, 215, "interior IR emittance with exterior 0.1"),
    (195, 200, "HC elements -> solid lightweight wall"),
    (230, 220, "infiltration 0 -> 1.0 ach"),
    (240, 220, "internal gains 0 -> 200 W"),
    (250, 220, "exterior solar absorptance 0.1 -> 0.9"),
    (270, 220, "south windows replace the HC elements"),
    (280, 270, "interior solar absorptance 0.9 -> 0.1"),
    (290, 270, "south overhang"),
    (300, 270, "east/west windows"),
    (310, 300, "east/west overhang and fins"),
    (320, 270, "thermostat 20,20 -> 20,27"),
    (395, 400, "HC elements -> solid lightweight wall"),
    (410, 400, "infiltration 0 -> 0.5 ach"),
    (420, 410, "internal gains 0 -> 200 W"),
    (430, 420, "exterior solar absorptance 0.1 -> 0.6"),
    (400, 600, "Case 600 with HC elements, no infiltration, no gains, absorptance 0.1"),
    (440, 600, "interior solar absorptance 0.6 -> 0.1"),
    (450, 600, "constant combined interior and exterior coefficients"),
    (460, 600, "constant combined interior coefficients"),
    (470, 600, "constant combined exterior coefficients"),
    (460, 450, "exterior coefficients: constant combined -> calculated"),
    (470, 450, "interior coefficients: constant combined -> calculated"),
    (800, 430, "heavyweight walls and floor"),
    (810, 900, "interior solar absorptance 0.6 -> 0.1"),
)

# Which case of the main suite (600/900) a difference refers to is resolved by the runner.
MAIN_SUITE_BASES = (600, 900)

def _coeffs(kind, spec):
    """(h_ci, h_ri, h_ce, h_re) for a surface kind under the case's coefficient modes."""
    int_mode, ext_mode = spec["int_coef"], spec["ext_coef"]
    if int_mode == "table_7_46":
        h_ci, h_ri = COMB_INT_TABLE_7_46[kind], 0.0
    else:
        table = CONV_INT_TABLE_7_40 if int_mode == "table_7_40" else CONV_INT_DEFAULT
        h_ci, h_ri = table[kind], H_RI_ENGINE * spec["eps_int"] / 0.9
    if ext_mode == "table_7_46":
        h_ce, h_re = COMB_EXT_TABLE_7_46[kind], 0.0
    else:
        table = CONV_EXT_TABLE_7_41 if ext_mode == "table_7_41" else CONV_EXT_DEFAULT
        h_ce, h_re = table[kind], H_RE_ENGINE * spec["eps_ext"] / 0.9
    return h_ci, h_ri, h_ce, h_re


def _surface(spec, name, azimuth, tilt, area, layers, construction_class, kind,
             height, length, sky_view_factor, solar_absorptance):
    surface = _opaque(name, azimuth, tilt, area, layers, construction_class, "wall",
                      "ashrae_convective", height, length, sky_view_factor)
    surface["solar_absorptance"] = solar_absorptance
    return _with_coefficients(surface, layers_resistance(layers), _coeffs(kind, spec))


def _windows_for(spec, fill):
    """Window surfaces of Cases 270-320, 440, 450-470, 810 (Figures 7-1, 7-4, 7-5, 7-6)."""
    shade = spec.get("shade")
    sc = "ashrae_convective"
    if fill == "south":
        w = _window("South window", 180.0, 12.0, WINDOW_600, sc)
        if shade == "overhang":
            w.update(OVERHANG_AT_ROOF)
        return [w]
    out = []
    for azimuth, name in ((90.0, "East window"), (270.0, "West window")):
        w = _window(name, azimuth, 6.0, WINDOW_600, sc)
        w["width"] = 3.0
        if shade == "overhang_fins":  # Case 310 = Case 630 shading
            w.update(OVERHANG_AT_ROOF)
            w["side_fins"] = {"left": {"depth": 1.0, "distance": 0.0},
                              "right": {"depth": 1.0, "distance": 0.0}}
        out.append(w)
    return out


def _set_window_coefficients(window, spec):
    """Re-apply the case's interior/exterior coefficients to a window."""
    return _with_coefficients(window, WINDOW_600["r_glazing"], _coeffs("window", spec))


def build_in_depth_case(case_number: int, solar_distribution: str = "table") -> dict:
    """Build one Section 7.2.3 case for ``ISO52016.Temperature_and_Energy_needs_calculation``.

    ``solar_distribution``: "table" applies the interior solar distribution fractions of
    Tables 7-43 to 7-45 where the case defines one (the only way to represent the interior
    solar absorptance); "area" uses the engine's area-weighted distribution.
    """
    if solar_distribution not in SOLAR_DISTRIBUTION_MODES:
        raise ValueError(f"solar_distribution must be one of {SOLAR_DISTRIBUTION_MODES}")
    if case_number not in CASE_SPECS:
        raise ValueError(f"Unsupported in-depth case: {case_number}. "
                         f"Supported: {sorted(CASE_SPECS)}")
    spec = CASE_SPECS[case_number]
    heavy = spec["mass"] == "heavy"
    wall, floor = (HWWALL, HWFLOOR) if heavy else (LTWALL, LTFLOOR)
    wall_class = "class_i" if heavy else "class_d"
    fill = spec["fill"]
    absorptance = spec["ext_abs"]

    bui = deepcopy(build_case(spec["mass"]))
    bui["building"]["name"] = f"ASHRAE140_Case{case_number}"

    h, area = WALL_HEIGHT_M, 8 * WALL_HEIGHT_M
    south_area = {"hc": area - HC_AREA_M2, "south": area - 12.0}.get(fill, area)
    east_west_area = 6 * h - (6.0 if fill == "eastwest" else 0.0)
    if fill == "eastwest":
        south_area = area
    surfaces = [
        _surface(spec, "North wall", 0.0, 90.0, area, wall, wall_class, "wall", h, area / h, 0.5, absorptance),
        _surface(spec, "South wall", 180.0, 90.0, south_area, wall, wall_class, "wall", h, south_area / h, 0.5, absorptance),
        _surface(spec, "East wall", 90.0, 90.0, east_west_area, wall, wall_class, "wall", h, east_west_area / h, 0.5, absorptance),
        _surface(spec, "West wall", 270.0, 90.0, east_west_area, wall, wall_class, "wall", h, east_west_area / h, 0.5, absorptance),
        _surface(spec, "Roof", 0.0, 0.0, FLOOR_AREA_M2, LTROOF, "class_d", "roof", 6.0, 8.0, 1.0, absorptance),
        # Raised floor: exterior receives no solar radiation (Section 7.2.1.5.1).
        _surface(spec, "Floor", 0.0, 0.0, FLOOR_AREA_M2, floor, "class_i", "floor", 6.0, 8.0, 1.0e-6, 0.0),
    ]
    if fill == "solid":
        # Case 195/395: the 12 m2 are lightweight exterior wall, as the rest of the south wall.
        surfaces[1] = _surface(spec, "South wall", 180.0, 90.0, area, LTWALL, "class_d", "wall",
                               h, area / h, 0.5, absorptance)
    elif fill == "hc":
        # Own construction class: the HC elements must not be aggregated with the walls.
        surfaces.append(_surface(spec, "High-conductance wall elements", 180.0, 90.0, HC_AREA_M2,
                                 HC_LAYERS, "class_e", "hc", 2.0, 6.0, 0.5, absorptance))
    else:
        for window in _windows_for(spec, fill):
            surfaces.append(_set_window_coefficients(window, spec))
    bui["building_surface"] = surfaces

    params = bui["building_parameters"]
    sp = params["temperature_setpoints"]
    cooling = 20.0 if spec["tstat"] == "20,20" else 27.0
    sp.update({"heating_setpoint": 20.0, "heating_setback": 20.0,
               "cooling_setpoint": cooling, "cooling_setback": cooling})
    params["internal_gains"][1]["full_load"] = spec["gains"] / FLOOR_AREA_M2
    params["ventilation"]["components"][0]["air_changes_per_hour"] = spec["ach"] * ALTITUDE_FACTOR
    if solar_distribution == "table" and spec["dist"] is not None:
        params["interior_solar_distribution"] = interior_solar_distribution(spec["dist"])
    elif solar_distribution == "table" and fill in ("south", "eastwest"):
        params["interior_solar_distribution"] = interior_solar_distribution("600")
    return bui


def describe_case(case_number: int) -> dict:
    """Spec of a case, for the report tables (inputs only, no results)."""
    return dict(CASE_SPECS[case_number])
