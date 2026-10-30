"""ASHRAE Standard 140-2023 Section 7 cases 600-695 and 900-995.

Case 600 and 900 are the lightweight and heavyweight base cases of ASHRAE
140-2023 Section 7 ("Building Thermal Envelope and Fabric Load Tests"): a
single-zone, 8 m x 6 m x 2.7 m box in Denver, CO, with 12 m2 of unshaded
south-facing glazing, constant infiltration, a 200 W internal gain (60%
radiative / 40% convective, always on) and an ideal, air-temperature
controlled heating/cooling system holding 20/27 degC year-round. Case 900
differs from Case 600 only in the wall and floor construction.

Every input below is taken from ANSI/ASHRAE Standard 140-2023 and its
accompanying electronic files; the relevant section or table is cited next
to each value.

How the 140-2023 inputs are mapped onto pybuildingenergy's single-zone
ISO 52016-1 engine:

* Constructions. The engine takes an air-to-air U-value per surface and
  recovers the construction resistance as R_c = 1/U - r_si - r_se, using
  the *same surface's* heat-transfer coefficients (``Conductance_node_of_
  element``). Each U-value here is therefore built from the standard's
  layer-by-layer surface-to-surface resistance (Tables 7-2, 7-25, 7-27)
  plus the surface coefficients set explicitly on that surface, so R_c
  round-trips exactly to the standard's value whatever coefficient set is
  chosen. kappa_m [J/(m2 K)] = sum(density x cp x thickness) of all layers.
* Surface heat-transfer coefficients (Sections 7.2.1.9.3 and 7.2.1.10.3).
  pybuildingenergy uses constant convective coefficients but computes
  long-wave exchange every time step (interior: linearised exchange between
  surfaces through h_ri; exterior: h_re plus the sky term). That is case
  (b)(1) of both sections, which prescribes the *convective* coefficients of
  Table 7-7 (exterior) and Table 7-9 (interior); this is the default
  (``surface_coefficients="ashrae_convective"``). The standard also permits
  the combined coefficients of Tables 7-7 and 7-9; that option is exposed as
  ``"ashrae_combined"`` for a documented Standard 140 sensitivity run.
* Infiltration and night ventilation (Section 7.2.1.6.1, Table 7-4; Section
  7.2.2.1.5, Table 7-16; Annex B3.1). The engine's ``constant_ach``
  component uses a fixed sea-level air heat capacity (1.204 kg/m3 x 1006
  J/(kg K)) with no altitude correction, so the specified rates are scaled
  to give the same mass flow (heat-capacity rate) as at the 1650 m site,
  where the standard gives rho = 1.0156 kg/m3 and cp = 1015 J/(kg K):
  0.5 ach -> 0.426 ach (Table 7-4 lists 0.414 for an assumed 1.2255 kg/m3
  sea-level density; 0.426 is the same correction for this engine's own
  sea-level density) and the 1700 m3/h vent fan -> 11.17 ach.
* Raised floor (Section 7.2.1.5). The floor is exposed to outdoor air on
  its underside, receives no solar radiation and sees zero wind. It is an
  OUTDOORS opaque surface with solar absorptance 0 and the floor
  coefficients of Tables 7-7/7-9. Its ``sky_view_factor`` is a negligible
  1e-6 rather than 0, because the single-zone engine treats any horizontal
  opaque surface with sky_view_factor == 0 as ground contact. Its tilt is
  given as 0 (not 180): the engine's orientation classifier maps tilt 180
  to a vertical bucket, which would merge the floor with the north wall in
  Case 900 (same construction class); with solar absorptance 0, a
  negligible sky view and explicit coefficients, the tilt has no other
  effect.
* Window (Section 7.2.1.11, Tables 7-10 to 7-12). The glazing resistance
  (two 3.048 mm panes + 12 mm air gap) is 1/5.208 + 2 x 0.003048 m2K/W.
  ``g_value`` is the normal-incidence SHGC (0.769) and
  ``solar_transmittance_correction_factor`` (ISO 52016-1 Formula E.3, F_w)
  carries its incidence-angle dependence: F_w is the irradiance-weighted
  ratio SHGC_eff / SHGC_n over the year for the window's orientation,
  using Table 7-12's angular SHGC (interpolated in cos(incidence), as
  Section 7.2.1.11.3 permits) for beam radiation and its hemispherical SHGC
  for sky- and ground-diffuse radiation (isotropic sky, ground reflectance
  0.2 per Section 7.2.1.1.2.5), computed from the normative weather data
  (``effective_window_fw``). The same is done for the 660/670 windows
  (Tables 7-17 to 7-24).
* Site (Table A1-1): Denver International Airport, 39.833 N, 104.650 W,
  1650 m, UTC-7. ``weather/USA_CO_Denver.Intl.AP.725650_TMY3.epw`` is
  identical, hour by hour, to the normative ``725650TY.CSV`` (dry-bulb,
  GHI, DNI, DHI, pressure and wind speed all match exactly).
* construction_class is set per surface to represent the thermal-mass
  placement of the Section 7 constructions.
* Shading devices (Figures 7-3, 7-5, 7-6), modelled with ISO 52016-1 Annex F
  Method 1 (direct radiation shaded, diffuse unaffected). Case 610: a 1.0 m
  deep overhang at roof level, 0.5 m above the top of the windows (sill
  0.2 m + 2.0 m window, wall 2.7 m), across the whole south wall, i.e. the
  "infinite length" overhang of F.3.3.1 (formula F.4). Case 630: on each
  east/west window (3 m x 2 m, centred, sill 0.2 m) a 1.0 m deep, 3 m wide
  overhang at roof level (0.5 m above the window) and two 1.0 m deep fins
  flush with the window edges from the roof to the ground (formulas F.5/F.6,
  combined with F.19-F.22). The window-wide overhang combined with fins at
  its ends is reproduced exactly by F.22: the part of the window the finite
  overhang leaves unshaded at oblique sun is the strip already shaded by
  the fin on the sun's side.
* Case 960 (Section 7.2.2.2.7) is a two-zone building run through the
  multizone engine (``build_case_960``): a conditioned lightweight back zone
  (Case 600 with the south wall and windows replaced by the common wall), a
  free-floating heavyweight sun zone (2 m x 8 m x 2.7 m, two 3 m x 2 m
  windows raised 0.3 m, no internal gains) and a 0.20 m concrete common wall
  (Table 7-29) with interior surface coefficients on both faces.

Known remaining simplification: pybuildingenergy has no input for the
interior solar distribution fractions of Table 7-13 (e.g. 64.2% of the
transmitted solar absorbed by the floor in Case 600); the engine
distributes transmitted solar over the interior surfaces by area, with a
fraction going directly to the air. Section 7.2.1.12.2 allows this only as
a documented deviation.
"""

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parent
WEATHER_FILE = EXAMPLES_DIR / "weather" / "USA_CO_Denver.Intl.AP.725650_TMY3.epw"

FLOOR_AREA_M2 = 48.0  # 8 m x 6 m (Section 7.2.1.3)
VOLUME_M3 = 129.6  # 48 m2 x 2.7 m
PERIMETER_M = 28.0  # 2 x (8 + 6)
WALL_HEIGHT_M = 2.7

# Site, Table A1-1 (725650TY.CSV, Denver International Airport).
LATITUDE_DEG = 39.833
LONGITUDE_DEG = -104.650
TIME_ZONE_H = -7.0
GROUND_REFLECTANCE = 0.2  # Section 7.2.1.1.2.5

# --- Constructions: (thickness m, conductivity W/(m K), density kg/m3, cp J/(kg K)),
# inside to outside.
LTWALL = ((0.012, 0.16, 950, 840), (0.066, 0.04, 12, 840), (0.009, 0.14, 530, 900))  # Table 7-2
LTFLOOR = ((0.025, 0.14, 650, 1200), (1.003, 0.04, 0, 0))  # Table 7-2
LTROOF = ((0.010, 0.16, 950, 840), (0.1118, 0.04, 12, 840), (0.019, 0.14, 530, 900))  # Table 7-2
HWWALL = ((0.100, 0.51, 1400, 1000), (0.0615, 0.04, 10, 1400), (0.009, 0.14, 530, 900))  # Table 7-27
HWFLOOR = ((0.080, 1.13, 1400, 1000), (1.007, 0.04, 0, 0))  # Table 7-27
LTWALL_680 = ((0.012, 0.16, 950, 840), (0.250, 0.04, 10, 1400), (0.009, 0.14, 530, 900))  # Table 7-25
LTROOF_680 = ((0.010, 0.16, 950, 840), (0.400, 0.04, 12, 840), (0.019, 0.14, 530, 900))  # Table 7-25
HWWALL_980 = ((0.100, 0.51, 1400, 1000), (0.2452, 0.04, 10, 1400), (0.009, 0.14, 530, 900))  # Table 7-31
COMMON_WALL_960 = ((0.20, 0.51, 1400, 1000),)  # Table 7-29


def layers_resistance(layers) -> float:
    """Surface-to-surface resistance [m2 K/W]."""
    return sum(d / k for d, k, _, _ in layers)


def layers_heat_capacity(layers) -> float:
    """Areal heat capacity kappa_m [J/(m2 K)]."""
    return sum(d * rho * cp for d, _, rho, cp in layers)


# --- Windows: glazing (pane + gap) resistance [m2 K/W], normal-incidence
# SHGC, angular SHGC at 0..90 deg in 10 deg steps, hemispherical SHGC.
WINDOW_600 = {  # Tables 7-10, 7-11, 7-12
    "r_glazing": 1.0 / 5.208 + 2 * 0.003048 / 1.0,
    "shgc_n": 0.769,
    "shgc_angular": (0.769, 0.768, 0.766, 0.761, 0.748, 0.718, 0.651, 0.509, 0.267, 0.0),
    "shgc_hemis": 0.670,
    "h_comb_int": 4.5, "h_comb_ext": 17.8,
}
WINDOW_660 = {  # Tables 7-17, 7-18, 7-19 (low-e, argon)
    "r_glazing": 0.55814 + 0.00318 + 0.00305,
    "shgc_n": 0.440,
    "shgc_angular": (0.440, 0.443, 0.438, 0.432, 0.422, 0.403, 0.361, 0.278, 0.148, 0.0),
    "shgc_hemis": 0.377,
    "h_comb_int": 4.5, "h_comb_ext": 17.8,
}
WINDOW_670 = {  # Tables 7-21, 7-22, 7-23 (single pane)
    "r_glazing": 0.00305,
    "shgc_n": 0.864,
    "shgc_angular": (0.864, 0.864, 0.862, 0.859, 0.851, 0.831, 0.785, 0.673, 0.424, 0.0),
    "shgc_hemis": 0.787,
    "h_comb_int": 7.8, "h_comb_ext": 16.0,
}

# Overhang at roof level, 1.0 m deep: its underside is 2.7 - (0.2 + 2.0) = 0.5 m
# above the top of the 2 m windows (Figures 7-1, 7-3, 7-5). ISO 52016-1 F.3.3.1:
# D_ovh = depth, L_ovh = vertical distance from the window edge.
OVERHANG_AT_ROOF = {
    "shading": True,
    "shading_type": "horizontal_overhang",
    "overhang_properties": {"width_of_horizontal_overhangs": 1.0},
    "width_or_distance_of_shading_elements": 0.5,
}

# Interior solar distribution (fractions of the transmitted solar radiation absorbed
# by each interior surface after all reflections, and lost back through the windows).
# Alternative constant inputs permitted by Sections 7.2.1.12.2, 7.2.2.1.2.2.2,
# 7.2.3.9.3.2, 7.2.3.10.2.2 and 7.2.3.12.2.2 for programs that do not compute the
# distribution themselves; a wall's share includes its windows. Used with
# ``solar_distribution="table"`` through ``interior_solar_distribution``.
SOLAR_DISTRIBUTION_TABLES = {
    "600": dict(floor=0.642, ceiling=0.167, east=0.038, west=0.038, north=0.053, south=0.027, lost=0.035),   # Table 7-13
    "620": dict(floor=0.642, ceiling=0.167, east=0.025, west=0.025, north=0.053, south=0.053, lost=0.035),   # Table 7-15
    "660": dict(floor=0.645, ceiling=0.170, east=0.039, west=0.039, north=0.054, south=0.026, lost=0.027),   # Table 7-20
    "670": dict(floor=0.641, ceiling=0.166, east=0.038, west=0.038, north=0.052, south=0.025, lost=0.040),   # Table 7-24
    "270": dict(floor=0.903, ceiling=0.050, east=0.010, west=0.010, north=0.014, south=0.007, lost=0.006),   # Table 7-43
    "280": dict(floor=0.243, ceiling=0.191, east=0.057, west=0.057, north=0.077, south=0.063, lost=0.312),   # Table 7-44
    "300": dict(floor=0.903, ceiling=0.050, east=0.0065, west=0.0065, north=0.014, south=0.014, lost=0.006), # Table 7-45
}
SOLAR_DISTRIBUTION_MODES = ("area", "table")


def interior_solar_distribution(table_key: str) -> dict:
    return dict(SOLAR_DISTRIBUTION_TABLES[table_key])


SOLAR_ABSORPTANCE_OPAQUE = 0.6  # Table 7-5
INTERNAL_GAIN_W = 200.0  # Section 7.2.1.7: 60% radiative / 40% convective
HEATING_SETPOINT_C = 20.0
COOLING_SETPOINT_C = 27.0

# Infiltration / vent fan, altitude-corrected to this engine's fixed
# sea-level air heat capacity (see module docstring).
RHO_CP_SITE = 1.0156 * 1015.0  # Table 7-3 footnote f / Annex B3.1
RHO_CP_ENGINE = 1.204 * 1006.0  # ventilation.py constant_ach defaults
ALTITUDE_FACTOR = RHO_CP_SITE / RHO_CP_ENGINE
INFILTRATION_ACH = 0.5 * ALTITUDE_FACTOR  # Section 7.2.1.6
VENT_FAN_ACH = 1700.0 / VOLUME_M3 * ALTITUDE_FACTOR  # Section 7.2.2.1.5.2

# --- Surface heat-transfer coefficient sets [W/(m2 K)]
# (h_ci, h_ri, h_ce, h_re) per surface kind.
H_RI_ENGINE = 5.13  # ISO 52016-1 Table 25, epsilon = 0.9 (Table 7-5)
H_RE_ENGINE = 4.14
SURFACE_COEFFICIENTS = {
    # Tables 7-7 / 7-9, convective columns (Sections 7.2.1.9.3 b(1), 7.2.1.10.3 b(1)).
    "ashrae_convective": {
        "wall": (2.2, H_RI_ENGINE, 11.9, H_RE_ENGINE),
        "roof": (1.8, H_RI_ENGINE, 14.4, H_RE_ENGINE),
        "floor": (2.2, H_RI_ENGINE, 0.8, H_RE_ENGINE),
        "window": (2.4, H_RI_ENGINE, 8.0, H_RE_ENGINE),
    },
    # Tables 7-7 / 7-9, combined columns (b(2)); window values per glazing type.
    "ashrae_combined": {
        "wall": (1.8, 0.0, 21.6, 0.0),
        "roof": (1.7, 0.0, 21.8, 0.0),
        "floor": (3.7, 0.0, 5.2, 0.0),
        "window": None,
    },
}


def _coefficients(kind: str, surface_coefficients: str, window: dict | None = None):
    try:
        table = SURFACE_COEFFICIENTS[surface_coefficients]
    except KeyError:
        raise ValueError(
            f"surface_coefficients must be one of {sorted(SURFACE_COEFFICIENTS)}"
        ) from None
    if kind == "window" and table["window"] is None:
        return (window["h_comb_int"], 0.0, window["h_comb_ext"], 0.0)
    return table[kind]


def _with_coefficients(surface: dict, r_c: float, coeffs) -> dict:
    """Set explicit coefficients and the air-to-air U that round-trips to r_c."""
    h_ci, h_ri, h_ce, h_re = coeffs
    surface.update({
        "convective_heat_transfer_coefficient_internal": h_ci,
        "radiative_heat_transfer_coefficient_internal": h_ri,
        "convective_heat_transfer_coefficient_external": h_ce,
        "radiative_heat_transfer_coefficient_external": h_re,
        "u_value": 1.0 / (1.0 / (h_ci + h_ri) + r_c + 1.0 / (h_ce + h_re)),
    })
    return surface


@lru_cache(maxsize=None)
def _hourly_solar_geometry():
    """Hourly sun position and irradiance from the normative weather data."""
    epw = pd.read_csv(WEATHER_FILE, skiprows=8, header=None)
    doy = pd.to_datetime(
        {"year": 2009, "month": epw[1], "day": epw[2]}
    ).dt.dayofyear.to_numpy()
    hour = epw[3].to_numpy(float) - 0.5  # hour-ending averages -> mid-hour, standard time
    ghi, dni, dhi = (epw[c].to_numpy(float) for c in (13, 14, 15))
    b = np.radians(360.0 * (doy - 81) / 364.0)
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = hour + (4.0 * (LONGITUDE_DEG - 15.0 * TIME_ZONE_H) + eot_min) / 60.0
    w = np.radians(15.0 * (solar_time - 12.0))
    d = np.radians(23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0)))
    phi = np.radians(LATITUDE_DEG)
    cos_z = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(w)
    sin_z = np.sqrt(np.clip(1.0 - cos_z ** 2, 0.0, 1.0))
    # Solar azimuth, clockwise from north.
    az = np.arctan2(-np.cos(d) * np.sin(w),
                    np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(w))
    return cos_z, sin_z, az, ghi, dni, dhi


def effective_window_fw(window: dict, azimuth_deg: float) -> float:
    """Irradiance-weighted F_w = SHGC_eff / SHGC_n of a vertical window."""
    cos_z, sin_z, az, ghi, dni, dhi = _hourly_solar_geometry()
    cos_inc = sin_z * np.cos(az - np.radians(azimuth_deg))
    cos_inc = np.where(cos_z > 0.0, np.clip(cos_inc, 0.0, 1.0), 0.0)
    beam = dni * cos_inc
    diffuse = 0.5 * dhi + 0.5 * GROUND_REFLECTANCE * ghi  # isotropic sky + ground
    cos_tab = np.cos(np.radians(np.arange(0, 100, 10)))[::-1]
    shgc_tab = np.asarray(window["shgc_angular"], dtype=float)[::-1]
    shgc_beam = np.interp(cos_inc, cos_tab, shgc_tab)
    gained = float(np.sum(beam * shgc_beam) + np.sum(diffuse) * window["shgc_hemis"])
    incident = float(np.sum(beam + diffuse))
    return gained / (incident * window["shgc_n"])


def _opaque(name, azimuth_deg, tilt_deg, area_m2, layers, construction_class, kind,
            surface_coefficients, height, length, sky_view_factor):
    surface = {
        "name": name,
        "type": "opaque",
        "boundary": "OUTDOORS",
        "area": area_m2,
        "sky_view_factor": sky_view_factor,
        "solar_absorptance": SOLAR_ABSORPTANCE_OPAQUE,
        "thermal_capacity": layers_heat_capacity(layers),
        "construction_class": construction_class,
        "orientation": {"azimuth": azimuth_deg, "tilt": tilt_deg},
        "name_adj_zone": None,
        "height": height,
        "length": length,
    }
    return _with_coefficients(surface, layers_resistance(layers),
                              _coefficients(kind, surface_coefficients))


def _wall(name, azimuth_deg, area_m2, layers, construction_class, surface_coefficients):
    return _opaque(name, azimuth_deg, 90.0, area_m2, layers, construction_class, "wall",
                   surface_coefficients, WALL_HEIGHT_M, area_m2 / WALL_HEIGHT_M, 0.5)


def _window(name, azimuth_deg, area_m2, window, surface_coefficients):
    surface = {
        "name": name,
        "type": "transparent",
        "boundary": "OUTDOORS",
        "area": area_m2,
        "sky_view_factor": 0.5,
        "g_value": window["shgc_n"],
        "solar_transmittance_correction_factor": effective_window_fw(window, azimuth_deg),
        "solar_absorptance": 0.0,
        "thermal_capacity": 0.0,
        "orientation": {"azimuth": azimuth_deg, "tilt": 90.0},
        "name_adj_zone": None,
        "height": 2.0,
        "width": area_m2 / 2.0,
        "frame_area_fraction": 0.0,  # no sash (Section 7.2.1.11.1)
        "shading": False,
    }
    return _with_coefficients(surface, window["r_glazing"],
                              _coefficients("window", surface_coefficients, window))


def build_case(mass: str = "light", surface_coefficients: str = "ashrae_convective") -> dict:
    """Build the Case 600 (``mass="light"``) or Case 900 (``mass="heavy"``)
    building_object for ``ISO52016.Temperature_and_Energy_needs_calculation``.
    """
    if mass == "light":
        wall, floor = LTWALL, LTFLOOR
        case_name = "ASHRAE140_Case600"
        wall_class = "class_d"  # ISO 52016-1 Table 23: wall "Evenly (D)"
    elif mass == "heavy":
        wall, floor = HWWALL, HWFLOOR
        case_name = "ASHRAE140_Case900"
        wall_class = "class_i"  # ISO 52016-1 Table 24: wall "Internal (I)"
    else:
        raise ValueError("mass must be 'light' (Case 600) or 'heavy' (Case 900)")

    window_area = 12.0
    sc = surface_coefficients
    surfaces = [
        _wall("North wall", 0.0, 8 * WALL_HEIGHT_M, wall, wall_class, sc),
        _wall("South wall", 180.0, 8 * WALL_HEIGHT_M - window_area, wall, wall_class, sc),
        _wall("East wall", 90.0, 6 * WALL_HEIGHT_M, wall, wall_class, sc),
        _wall("West wall", 270.0, 6 * WALL_HEIGHT_M, wall, wall_class, sc),
        # Roof: "Evenly (D)" for both cases (ISO 52016-1 Tables 23-24).
        _opaque("Roof", 0.0, 0.0, FLOOR_AREA_M2, LTROOF, "class_d", "roof", sc, 6.0, 8.0, 1.0),
        # Raised floor exposed to outdoor air (Section 7.2.1.5): no solar, see
        # module docstring for sky_view_factor and tilt.
        {**_opaque("Floor", 0.0, 0.0, FLOOR_AREA_M2, floor, "class_i", "floor", sc,
                   6.0, 8.0, 1.0e-6),
         "solar_absorptance": 0.0},
        _window("South window", 180.0, window_area, WINDOW_600, sc),
    ]

    return {
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
            # Required fallback for any surface without its own construction_class;
            # every surface sets one explicitly, so this value is never used.
            "construction_class": "class_i",
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
            "system_capacities": {  # Section 7.2.1.13.2: 1000 kW, effectively infinite
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
                        "air_changes_per_hour": INFILTRATION_ACH,
                    }
                ]
            },
        },
        "units": {
            "area": "m2", "u_value": "W/m2K", "thermal_capacity": "J/m2K",
        },
    }


# ---------------------------------------------------------------------------
# Annex A3 acceptance ranges.
#
# The range values are NOT distributed with this repository: they are
# copyright ASHRAE (ANSI/ASHRAE Standard 140-2023, Normative Annex A3,
# Tables A3-1 to A3-4, introduced by Addendum b to Standard 140-2020) and
# reproducing them requires ASHRAE's written permission. Anyone with the
# standard can enter them in a local JSON file (see
# ``annex_a3_ranges.template.json`` for the format), by default
#     Standards/local_reference_data/ashrae140_2023_annex_a3.json
# (``Standards/`` is git-ignored), or point ``ASHRAE140_ANNEX_A3_FILE`` at it.
# Without the file the cases still run; range checks report "not available".
#
# Below: only the structure of the standard's range cases -- which case each
# range is a difference from, and in which Annex A3 table it is listed.
# ---------------------------------------------------------------------------
ANNEX_A3_FILE = Path(os.environ.get(
    "ASHRAE140_ANNEX_A3_FILE",
    EXAMPLES_DIR.parent.parent / "Standards" / "local_reference_data" / "ashrae140_2023_annex_a3.json",
))

# Table A3-1 (heating) / A3-2 (cooling); differential cases vs the parent case.
LOW_MASS_CASES = {
    600: {"parent": None}, 610: {"parent": 600}, 620: {"parent": 600}, 630: {"parent": 620},
    640: {"parent": 600}, 650: {"parent": 600, "no_heating_range": True}, 660: {"parent": 600},
    670: {"parent": 600}, 680: {"parent": 600}, 685: {"parent": 600}, 695: {"parent": 685},
}
# Table A3-3 (heating) / A3-4 (cooling). 910 is written as Case 900 - Case 910.
HIGH_MASS_CASES = {
    900: {"parent": None}, 910: {"parent": 900, "reverse": True}, 920: {"parent": 900},
    930: {"parent": 920}, 940: {"parent": 900}, 950: {"parent": 900, "no_heating_range": True},
    960: {"parent": 900}, 980: {"parent": 900}, 985: {"parent": 900}, 995: {"parent": 985},
}
ANNEX_A3_TABLES = {  # where each range is printed in the standard
    "low": {"heating": "Table A3-1", "cooling": "Table A3-2"},
    "high": {"heating": "Table A3-3", "cooling": "Table A3-4"},
}


def load_annex_a3_ranges(path=None) -> dict | None:
    """Return {case: {"heating": (lo, hi) | None, "cooling": ...}} from the
    local file, or None when it is not available."""
    path = Path(path or ANNEX_A3_FILE)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))["ranges"]
    return {int(k): {m: (tuple(v[m]) if v.get(m) is not None else None) for m in ("heating", "cooling")}
            for k, v in data.items()}


_RANGES = load_annex_a3_ranges()
ANNEX_A3_AVAILABLE = _RANGES is not None


def _acceptance(cases: dict) -> dict:
    out = {}
    for case, info in cases.items():
        entry = {k: v for k, v in info.items() if k != "no_heating_range"}
        for metric in ("heating", "cooling"):
            if metric == "heating" and info.get("no_heating_range"):
                entry[metric] = None  # the standard gives no range for this result
            elif _RANGES is None:
                entry[metric] = "unavailable"
            else:
                entry[metric] = _RANGES[case][metric]
        out[case] = entry
    return out


# {case: {"parent": ..., ["reverse": True,] "heating": (lo, hi) | None | "unavailable", "cooling": ...}}
LOW_MASS_ACCEPTANCE = _acceptance(LOW_MASS_CASES)
HIGH_MASS_ACCEPTANCE = _acceptance(HIGH_MASS_CASES)


def annex_a3_table(case_number: int, metric: str) -> str:
    group = "low" if case_number in LOW_MASS_CASES else "high"
    return ANNEX_A3_TABLES[group][metric]


# Cases 980, 985 and 995 are not "600-series + 300": they are built on Case
# 900 directly (Table 7-14). 960 needs two zones (build_case_960).
HIGH_MASS_SINGLE_ZONE = (900, 910, 920, 930, 940, 950, 980, 985, 995)


def build_bestest_case(case_number: int, surface_coefficients: str = "ashrae_convective",
                       solar_distribution: str = "area") -> dict:
    """Build ASHRAE 140 low-mass 600- or high-mass 900-series cases.

    The high-mass cases 900-950 reuse the corresponding 600-series
    perturbation (900=600, 910=610, ..., 950=650) while using the Case 900
    wall/floor constructions, as specified by ASHRAE 140 (Section 7.2.2.2).
    Case 980 is Case 900 with the Table 7-31 wall/roof insulation, 985 is
    Case 900 with the "20,20" thermostat and 995 is 980 with it. Case 960 is
    two-zone: use ``build_case_960``.

    ``solar_distribution``: "area" (default) uses the engine's ISO 52016-1 area-weighted
    distribution of the transmitted solar radiation; "table" applies the constant
    distribution fractions the standard gives for the case (Tables 7-13, 7-15, 7-20 and 7-24).
    """
    if solar_distribution not in SOLAR_DISTRIBUTION_MODES:
        raise ValueError(f"solar_distribution must be one of {SOLAR_DISTRIBUTION_MODES}")
    if case_number == 960:
        raise ValueError("Case 960 has two zones: use build_case_960()")
    is_high_mass = case_number in HIGH_MASS_ACCEPTANCE
    if not is_high_mass and case_number not in LOW_MASS_ACCEPTANCE:
        raise ValueError(f"Unsupported BESTEST case: {case_number}")

    sc = surface_coefficients
    bui = deepcopy(build_case("heavy" if is_high_mass else "light", surface_coefficients=sc))
    source_case = {980: 980, 985: 685, 995: 995}.get(
        case_number, case_number - 300 if is_high_mass else case_number)
    surfaces = bui["building_surface"]
    south_window = next(s for s in surfaces if s["type"] == "transparent")

    def replace_wall_area(name, area):
        wall = next(s for s in surfaces if s["name"] == name)
        wall["area"] = area
        wall["length"] = area / WALL_HEIGHT_M

    def replace_window(window):
        idx = surfaces.index(south_window)
        surfaces[idx] = _window(south_window["name"], 180.0, south_window["area"], window, sc)
        return surfaces[idx]

    if source_case in (620, 630):  # Sections 7.2.2.1.2-7.2.2.1.3
        surfaces.remove(south_window)
        south_wall = next(s for s in surfaces if s["name"] == "South wall")
        south_wall["area"] += 12.0
        south_wall["length"] = south_wall["area"] / WALL_HEIGHT_M
        for azimuth, wall_name, window_name in ((90.0, "East wall", "East window"),
                                                (270.0, "West wall", "West window")):
            replace_wall_area(wall_name, 6 * WALL_HEIGHT_M - 6.0)
            w = _window(window_name, azimuth, 6.0, WINDOW_600, sc)
            w["width"] = 3.0
            surfaces.append(w)
            if source_case == 630:  # Figures 7-5/7-6: overhang + fins, 1.0 m deep
                w.update(OVERHANG_AT_ROOF)
                w["side_fins"] = {
                    "left": {"depth": 1.0, "distance": 0.0},
                    "right": {"depth": 1.0, "distance": 0.0},
                }
    elif source_case == 610:  # Section 7.2.2.1.1, Figure 7-3
        south_window.update(OVERHANG_AT_ROOF)
    elif source_case == 660:  # Section 7.2.2.1.6
        replace_window(WINDOW_660)
    elif source_case == 670:  # Section 7.2.2.1.7
        replace_window(WINDOW_670)
    elif source_case in (680, 695, 980, 995):  # Tables 7-25 / 7-31
        for s in surfaces:
            if s["name"] in {"North wall", "South wall", "East wall", "West wall"}:
                layers = HWWALL_980 if is_high_mass else LTWALL_680
            elif s["name"] == "Roof":
                layers = LTROOF_680
            else:
                continue
            kind = "roof" if s["name"] == "Roof" else "wall"
            s["thermal_capacity"] = layers_heat_capacity(layers)
            _with_coefficients(s, layers_resistance(layers), _coefficients(kind, sc))

    sp = bui["building_parameters"]["temperature_setpoints"]
    if source_case == 640:  # Section 7.2.2.1.4
        sp["heating_setpoint"] = 20.0
        sp["heating_setback"] = 10.0
        sp["cooling_setpoint"] = 27.0
        sp["cooling_setback"] = 27.0
        # 20 degC from 07:00 (reached at 08:00, hourly-time-step rule (2)) to 23:00.
        night_off = [0.0] * 7 + [1.0] * 16 + [0.0]
        bui["building_parameters"]["heating_profile"] = {
            "weekday": night_off, "holiday": night_off,
        }
    elif source_case == 650:  # Section 7.2.2.1.5
        # Night ventilation 18:00-07:00, cooling only 07:00-18:00, heating always off.
        day_cooling = [0.0] * 7 + [1.0] * 11 + [0.0] * 6
        night_vent = [1.0] * 7 + [0.0] * 11 + [1.0] * 6
        sp["heating_setpoint"] = -999.0
        sp["heating_setback"] = -999.0
        sp["cooling_setpoint"] = 27.0
        sp["cooling_setback"] = 27.0
        bui["building_parameters"]["cooling_profile"] = {
            "weekday": day_cooling, "holiday": day_cooling,
        }
        bui["building_parameters"]["ventilation_profile"] = {
            "weekday": night_vent, "holiday": night_vent,
        }
        # Infiltration stays on continuously; the schedule applies only to
        # the additional vent-fan stream (Table 7-16, altitude-corrected).
        bui["building_parameters"]["ventilation"]["components"].append({
            "name": "night_ventilation",
            "ventilation_type": "constant_ach",
            "air_changes_per_hour": VENT_FAN_ACH,
            "source_temperature": "outdoor",
            "profile": "ventilation_profile",
        })
    if source_case in (685, 695, 995):  # Sections 7.2.2.1.9-7.2.2.1.10, 7.2.2.2.9-10
        sp["heating_setpoint"] = 20.0
        sp["heating_setback"] = 20.0
        sp["cooling_setpoint"] = 20.0
        sp["cooling_setback"] = 20.0

    if solar_distribution == "table":
        key = {620: "620", 630: "620", 660: "660", 670: "670"}.get(source_case, "600")
        bui["building_parameters"]["interior_solar_distribution"] = interior_solar_distribution(key)

    bui["building"]["name"] = f"ASHRAE140_Case{case_number}"
    return bui


SUN_ZONE_DEPTH_M = 2.0  # Section 7.2.2.2.7.3
SUN_ZONE_VOLUME_M3 = 43.2


def build_case_960(surface_coefficients: str = "ashrae_convective") -> dict:
    """Case 960 sunspace (Section 7.2.2.2.7) for
    ``ISO52016.simulate_envelope_multizone_free_floating``.

    Zones: "back" (conditioned, lightweight, as Case 600 without the south
    wall and windows) and "sun" (free-floating, heavyweight as Case 900).
    """
    sc = surface_coefficients
    base = build_case("light", surface_coefficients=sc)  # building-level fields

    def zone(surface, name):
        surface["zone"] = name
        return surface

    common_area = 8 * WALL_HEIGHT_M
    common = _opaque("Common wall", 180.0, 90.0, common_area, COMMON_WALL_960, "class_d",
                     "wall", sc, WALL_HEIGHT_M, 8.0, 0.0)
    h_ci, h_ri, _, _ = _coefficients("wall", sc)
    # INTERNAL partition: the "external" coefficients are those of the sun-zone face.
    common.update({
        "boundary": "INTERNAL",
        "adjacent_zone": "sun",
        "solar_absorptance": 0.0,
        "convective_heat_transfer_coefficient_external": h_ci,
        "radiative_heat_transfer_coefficient_external": h_ri,
        "u_value": 1.0 / (2.0 / (h_ci + h_ri) + layers_resistance(COMMON_WALL_960)),
    })

    sun_floor_area = SUN_ZONE_DEPTH_M * 8.0
    sun_end_wall = SUN_ZONE_DEPTH_M * WALL_HEIGHT_M  # 5.4 m2 (7.2.2.2.7.3 d)
    floor_back = {**_opaque("Floor", 0.0, 0.0, FLOOR_AREA_M2, LTFLOOR, "class_i", "floor", sc,
                            6.0, 8.0, 1.0e-6), "solar_absorptance": 0.0}
    floor_sun = {**_opaque("Sun floor", 0.0, 0.0, sun_floor_area, HWFLOOR, "class_i", "floor",
                           sc, SUN_ZONE_DEPTH_M, 8.0, 1.0e-6), "solar_absorptance": 0.0}
    surfaces = [
        zone(_wall("North wall", 0.0, 8 * WALL_HEIGHT_M, LTWALL, "class_d", sc), "back"),
        zone(_wall("East wall", 90.0, 6 * WALL_HEIGHT_M, LTWALL, "class_d", sc), "back"),
        zone(_wall("West wall", 270.0, 6 * WALL_HEIGHT_M, LTWALL, "class_d", sc), "back"),
        zone(_opaque("Roof", 0.0, 0.0, FLOOR_AREA_M2, LTROOF, "class_d", "roof", sc,
                     6.0, 8.0, 1.0), "back"),
        zone(floor_back, "back"),
        zone(common, "back"),
        zone(_wall("Sun south wall", 180.0, 8 * WALL_HEIGHT_M - 12.0, HWWALL, "class_i", sc), "sun"),
        zone(_wall("Sun east wall", 90.0, sun_end_wall, HWWALL, "class_i", sc), "sun"),
        zone(_wall("Sun west wall", 270.0, sun_end_wall, HWWALL, "class_i", sc), "sun"),
        zone(_opaque("Sun roof", 0.0, 0.0, sun_floor_area, LTROOF, "class_d", "roof", sc,
                     SUN_ZONE_DEPTH_M, 8.0, 1.0), "sun"),
        zone(floor_sun, "sun"),
        # Same windows as Case 900, raised 0.3 m (sill 0.5 m): no effect without shading.
        zone(_window("Sun window", 180.0, 12.0, WINDOW_600, sc), "sun"),
    ]

    infiltration = lambda: {"components": [{  # noqa: E731
        "name": "infiltration", "ventilation_type": "constant_ach",
        "air_changes_per_hour": INFILTRATION_ACH,
    }]}
    gains = lambda w: [  # noqa: E731
        {"name": "occupants", "full_load": 0.0, "weekday": [1.0] * 24, "weekend": [1.0] * 24},
        {"name": "appliances", "full_load": w, "weekday": [1.0] * 24, "weekend": [1.0] * 24},
        {"name": "lighting", "full_load": 0.0, "weekday": [1.0] * 24, "weekend": [1.0] * 24},
    ]
    zones = [
        {
            "name": "back", "net_floor_area": FLOOR_AREA_M2, "volume": VOLUME_M3,
            "heating_setpoint": HEATING_SETPOINT_C, "heating_setback": HEATING_SETPOINT_C,
            "cooling_setpoint": COOLING_SETPOINT_C, "cooling_setback": COOLING_SETPOINT_C,
            "heating_capacity": 1.0e9, "cooling_capacity": 1.0e9,
            "internal_gains": gains(INTERNAL_GAIN_W / FLOOR_AREA_M2),
            "ventilation": infiltration(),
        },
        {
            # Free-floating (7.2.2.2.7.5 a), no internal gains (7.2.2.2.7.3 g).
            "name": "sun", "net_floor_area": sun_floor_area, "volume": SUN_ZONE_VOLUME_M3,
            "heating_setpoint": -999.0, "heating_setback": -999.0,
            "cooling_setpoint": 999.0, "cooling_setback": 999.0,
            "heating_capacity": 0.0, "cooling_capacity": 0.0,
            "internal_gains": gains(0.0),
            "ventilation": infiltration(),
        },
    ]
    base["building"].update({
        "name": "ASHRAE140_Case960",
        "net_floor_area": FLOOR_AREA_M2 + sun_floor_area,
        "volume": VOLUME_M3 + SUN_ZONE_VOLUME_M3,
    })
    base["building_surface"] = surfaces
    base["zones"] = zones
    return base
