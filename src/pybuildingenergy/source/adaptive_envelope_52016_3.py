"""EN ISO 52016-3:2023 + A1:2025 - adaptive building envelope elements.

Calculation procedures for transparent envelope elements whose thermal, solar
and daylight properties change during the year: dynamic solar shading
(roller blind, venetian blind, shutter) and chromogenic glazing.  The module
adds, on top of the hourly EN ISO 52016-1 balance, the part of the standard
that decides *which state* the element is in at every hour:

* 6.7  environmentally activated chromogenic glazing (Formula 1, hysteresis);
* 6.8  actively controlled elements: the reference control scenarios built
  from the conditions DAY / HC / TINT / RAD / OCC / GLARE / DAYL (Table 22),
  their identification methods (Tables 24-31), the parameter values
  (Table 33, Table C.1) and the 23 states of Annex C (Tables C.2-C.8) with
  the "occupant fraction" correction of 6.8.6 (approach 1);
* 6.10 key performance indicators against a reference element (annual and
  monthly), state statistics and the clause 8 conformity report; the
  adaptive-versus-reference run is in ``adaptive_envelope_workflow_52016_3``.

Scope and assumptions
---------------------
* Simplified adaptive elements only (6.4.3): U_w;st, g_gl;st and tau_V;gl;st are
  pre-calculated inputs for the retracted state and for the extended state
  (venetian blind: one set for each slat position).  Detailed elements
  (6.4.4) and actively ventilated cavities (6.4.3.3) need ISO 15099 /
  ISO 52022-3 and are not modelled; a ventilated-cavity element can be given
  as a simplified element with pre-calculated properties.
* Not worked out in the standard and therefore not implemented: HC-MODE
  method 3, RAD-MODE method AC2, GLARE-MODE methods AC2/AC3, HFLOW-MODE,
  peak shaving, protection against view-in, user-behaviour approaches 2 and
  3, control type CNTRL_INTEGR.
* DAYL-MODE needs a_dayl and b_dayl of Formula (4); Table 33 gives no default
  for them.  Without them the daylight level is taken as "fine" (LN).
* Occupied-sleeping (residential): the occupancy profile cannot tell awake
  from sleeping occupants, so ``sleeping_hours`` (default 23-07) is an input.
* The extended fraction of the element is ``rule + correction`` (Annex C);
  the element properties are the area-weighted mix of the retracted state and
  the extended state (6.8.6.2: a fraction of the element is left unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (Formula 2, Formula 3, Table 33)
# ---------------------------------------------------------------------------
LUMINOUS_EFFICACY_LM_W = 115.0  # K_V, Formula (2), ISO 52010-1 average sky
GLARE_SLOPE = 0.25 * 6.22e-5  # Formula (3): P = 0,25 * 6,22e-5 * E + 0,184
GLARE_OFFSET = 0.184
GLARE_LIMIT = 0.45  # P_dglare;limGY, Table 33
GLARE_LIMIT_LX_FORMULA = (GLARE_LIMIT - GLARE_OFFSET) / GLARE_SLOPE  # Formula (3) inverted, ~17 106 lx
GLARE_LIMIT_LX_TABLE33_NOTE = 4500.0  # Table 33 note a ("corresponds with 4 500 lx")
DAYLIGHT_DESK_LIMIT_LX = 300.0  # E_V;int;desk;limLL, Table 33
HC_OPERATIVE_LIMITS_C = (20.0, 23.0)  # theta_int;op;24;limHE / limCO (AM1)
HC_OUTDOOR_LIMITS_C = (12.0, 16.0)  # theta_e;a;24;limHE / limCO (AM2)
TINT_LIMITS_C = (22.0, 24.0)  # theta_int;op;limTN / limTH
HYSTERESIS_DEFAULT = {"solar_irradiance": 150.0, "temperature": 30.0}  # 6.4.3.2.1
OCCUPANT_FRACTION_DAY_START_H = 7  # "previous day" starts at 07h00 (Table 25)

# Table C.1 - [NE-TH, NE-TV, CO-TH, CO-TV] x (RL, RH), daylight illuminance in klux
_RAD_LIMITS_KLUX = {
    "MAN-RES": (37.5, 72.5, 37.5, 72.5, 37.5, 72.5, 22.5, 47.5),
    "MOT-RES": (30.0, 60.0, 30.0, 60.0, 30.0, 60.0, 17.5, 33.0),
    "AUT-RES": (30.0, 60.0, 15.0, 30.0, 15.0, 30.0, 10.0, 15.0),
    "MAN-OFFHE": (37.5, 72.5, 37.5, 72.5, 37.5, 72.5, 22.5, 47.5),
    "MOT-OFFHE": (30.0, 60.0, 30.0, 60.0, 30.0, 60.0, 17.5, 33.0),
    "AUT-OFFHE": (30.0, 60.0, 15.0, 30.0, 15.0, 30.0, 10.0, 15.0),
    "MOT-OFFHECO": (15.0, 30.0, 15.0, 30.0, 10.0, 15.0, 10.0, 15.0),
    "AUT-OFFHECO": (10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0),
}
_RAD_COLUMNS = {"NE-TH": 0, "NE-TV": 2, "CO-TH": 4, "CO-TV": 6}

# Annex C: state number -> (manual rule, manual correction, manual ext. state,
#                           auto rule,   auto correction,   auto ext. state)
_SCENARIOS = {
    "C.2": {
        1: (0.0, 0.1, 1, 0.0, 0.0, 1),
        2: (0.0, 0.1, 1, 0.0, 0.0, 1),
        3: (0.0, 0.1, 1, 0.0, 0.0, 1),
        4: (0.0, 0.1, 1, 0.0, 0.0, 1),
        5: (0.0, 0.1, 1, 0.0, 0.0, 1),
        6: (1.0, -0.4, 1, 1.0, 0.0, 1),
        7: (1.0, -0.4, 1, 1.0, -0.2, 1),
        8: (0.0, 0.1, 1, 0.0, 0.0, 1),
        9: (1.0, -0.4, 1, 1.0, 0.0, 1),
        10: (1.0, -0.4, 1, 1.0, -0.2, 1),
        11: (0.0, 0.1, 1, 0.0, 0.0, 1),
        12: (0.0, 0.1, 1, 0.0, 0.0, 1),
        13: (0.0, 0.1, 1, 0.0, 0.0, 1),
        14: (1.0, -0.4, 1, 1.0, 0.0, 1),
        15: (1.0, -0.4, 1, 1.0, -0.2, 1),
        16: (0.0, 0.1, 1, 0.0, 0.0, 1),
        17: (1.0, -0.2, 1, 1.0, 0.0, 1),
        18: (1.0, -0.2, 1, 1.0, -0.2, 1),
        19: (0.0, 0.1, 1, 0.0, 0.0, 1),
        20: (0.0, 0.1, 1, 0.0, 0.0, 1),
        21: (0.0, 0.1, 1, 0.0, 0.0, 1),
        22: (0.0, 0.1, 1, 0.0, 0.0, 1),
        23: (0.0, 0.1, 1, 0.0, 0.0, 1),
    },
    "C.3": {
        1: (0.0, 0.1, 1, 0.0, 0.0, 1),
        2: (0.0, 0.1, 1, 0.0, 0.0, 1),
        3: (0.0, 0.1, 1, 0.0, 0.0, 1),
        4: (0.0, 0.1, 1, 0.0, 0.0, 1),
        5: (0.0, 0.1, 1, 0.0, 0.0, 1),
        6: (1.0, -0.4, 1, 1.0, 0.0, 1),
        7: (1.0, -0.4, 1, 1.0, -0.2, 1),
        8: (0.0, 0.1, 1, 0.0, 0.0, 1),
        9: (1.0, -0.4, 1, 1.0, 0.0, 1),
        10: (1.0, -0.4, 1, 1.0, -0.2, 1),
        11: (0.0, 0.1, 1, 0.0, 0.0, 1),
        12: (0.0, 0.1, 1, 0.0, 0.0, 1),
        13: (0.0, 0.1, 1, 0.0, 0.0, 1),
        14: (1.0, -0.4, 1, 1.0, 0.0, 1),
        15: (1.0, -0.4, 1, 1.0, -0.2, 1),
        16: (0.0, 0.1, 1, 0.0, 0.0, 1),
        17: (1.0, -0.2, 1, 1.0, 0.0, 1),
        18: (1.0, -0.2, 1, 1.0, -0.2, 1),
        19: (1.0, 0.0, 1, 1.0, 0.0, 1),
        20: (1.0, 0.0, 1, 1.0, 0.0, 1),
        21: (1.0, 0.0, 1, 0.5, 0.0, 1),
        22: (1.0, 0.0, 1, 1.0, 0.0, 1),
        23: (0.5, 0.0, 1, 0.5, 0.0, 1),
    },
    "C.4": {
        1: (0.0, 0.1, 1, 0.0, 0.0, 1),
        2: (0.0, 0.1, 1, 0.0, 0.0, 1),
        3: (0.0, 0.1, 1, 0.0, 0.0, 1),
        4: (0.0, 0.1, 1, 0.0, 0.0, 1),
        5: (0.0, 0.1, 1, 0.0, 0.0, 1),
        6: (1.0, -0.4, 2, 1.0, 0.0, 2),
        7: (1.0, -0.4, 2, 1.0, 0.0, 2),
        8: (0.0, 0.1, 1, 0.0, 0.0, 1),
        9: (1.0, -0.4, 2, 1.0, 0.0, 3),
        10: (1.0, -0.4, 2, 1.0, -0.2, 2),
        11: (0.0, 0.1, 1, 0.0, 0.0, 1),
        12: (0.0, 0.1, 1, 0.0, 0.0, 1),
        13: (1.0, -0.4, 1, 0.0, 0.0, 1),
        14: (1.0, -0.4, 2, 1.0, 0.0, 3),
        15: (1.0, -0.4, 2, 1.0, -0.2, 2),
        16: (1.0, -0.4, 1, 0.0, 0.0, 1),
        17: (1.0, -0.2, 3, 1.0, 0.0, 3),
        18: (1.0, -0.2, 3, 1.0, -0.2, 2),
        19: (0.0, 0.1, 3, 1.0, 0.0, 3),
        20: (0.0, 0.1, 3, 1.0, 0.0, 3),
        21: (0.0, 0.1, 3, 0.5, 0.0, 1),
        22: (0.0, 0.1, 3, 1.0, 0.0, 3),
        23: (0.0, 0.1, 2, 0.0, 0.0, 1),
    },
    "C.5": {
        1: (0.0, 0.1, 1, 0.0, 0.0, 1),
        2: (0.0, 0.1, 1, 0.0, 0.0, 1),
        3: (0.0, 0.1, 1, 0.0, 0.0, 1),
        4: (0.0, 0.1, 1, 0.0, 0.0, 1),
        5: (0.0, 0.1, 1, 0.0, 0.0, 1),
        6: (1.0, -0.4, 1, 1.0, 0.0, 1),
        7: (1.0, -0.4, 1, 1.0, -0.2, 1),
        8: (0.0, 0.1, 1, 0.0, 0.0, 1),
        9: (1.0, -0.4, 1, 1.0, 0.0, 1),
        10: (1.0, -0.4, 1, 1.0, -0.2, 1),
        11: (0.0, 0.1, 1, 0.0, 0.0, 1),
        12: (0.0, 0.1, 1, 0.0, 0.0, 1),
        13: (0.0, 0.1, 1, 0.0, 0.0, 1),
        14: (1.0, -0.4, 1, 1.0, 0.0, 1),
        15: (1.0, -0.4, 1, 1.0, -0.2, 1),
        16: (0.0, 0.1, 1, 0.0, 0.0, 1),
        17: (1.0, -0.2, 1, 1.0, 0.0, 1),
        18: (1.0, -0.2, 1, 1.0, -0.2, 1),
        19: (1.0, -0.2, 1, 1.0, 0.0, 1),
        20: (1.0, -0.2, 1, 1.0, 0.0, 1),
        21: (1.0, -0.2, 1, 0.0, 0.0, 1),
        22: (1.0, -0.2, 1, 1.0, 0.0, 1),
        23: (1.0, -0.2, 1, 0.0, 0.0, 1),
    },
    "C.6": {
        1: (1.0, -0.2, 1, 0.0, 0.0, 1),
        2: (1.0, -0.2, 1, 0.0, 0.0, 1),
        3: (1.0, -0.2, 1, 0.0, 0.0, 1),
        4: (1.0, -0.2, 1, 0.0, 0.0, 1),
        5: (1.0, -0.2, 1, 0.0, 0.0, 1),
        6: (1.0, -0.2, 2, 1.0, 0.0, 2),
        7: (1.0, -0.2, 2, 1.0, 0.0, 2),
        8: (1.0, -0.2, 1, 0.0, 0.0, 1),
        9: (1.0, -0.2, 2, 1.0, 0.0, 3),
        10: (1.0, -0.2, 2, 1.0, -0.2, 2),
        11: (1.0, -0.2, 1, 0.0, 0.0, 1),
        12: (1.0, -0.2, 1, 0.0, 0.0, 1),
        13: (1.0, -0.2, 1, 1.0, 0.0, 1),
        14: (1.0, -0.2, 2, 1.0, 0.0, 3),
        15: (1.0, -0.2, 2, 1.0, -0.2, 2),
        16: (1.0, -0.2, 1, 1.0, 0.0, 1),
        17: (1.0, -0.2, 3, 1.0, 0.0, 3),
        18: (1.0, -0.2, 3, 1.0, -0.2, 2),
        19: (1.0, -0.2, 3, 1.0, 0.0, 3),
        20: (1.0, -0.2, 3, 1.0, 0.0, 3),
        21: (1.0, -0.2, 3, 0.5, 0.0, 1),
        22: (1.0, -0.2, 3, 1.0, 0.0, 3),
        23: (1.0, -0.2, 2, 1.0, 0.0, 1),
    },
    "C.7": {
        1: (0.0, 0.2, 1, 0.0, 0.0, 1),
        2: (1.0, -0.5, 1, 1.0, -0.4, 1),
        3: (0.0, 0.2, 1, 0.0, 0.0, 1),
        4: (1.0, -0.5, 1, 1.0, -0.4, 1),
        5: (0.0, 0.2, 1, 0.0, 0.0, 1),
        6: (1.0, -0.5, 1, 1.0, 0.0, 1),
        7: (1.0, -0.5, 1, 1.0, -0.4, 1),
        8: (0.0, 0.2, 1, 0.0, 0.0, 1),
        9: (1.0, -0.5, 1, 1.0, 0.0, 1),
        10: (1.0, -0.5, 1, 1.0, -0.4, 1),
        11: (0.0, 0.2, 1, 0.0, 0.0, 1),
        12: (1.0, -0.5, 1, 1.0, -0.4, 1),
        13: (0.0, 0.2, 1, 0.0, 0.0, 1),
        14: (1.0, -0.5, 1, 1.0, 0.0, 1),
        15: (1.0, -0.5, 1, 1.0, -0.4, 1),
        16: (0.0, 0.2, 1, 0.0, 0.0, 1),
        17: (1.0, -0.5, 1, 1.0, 0.0, 1),
        18: (1.0, -0.5, 1, 1.0, -0.4, 1),
        19: (1.0, -0.5, 1, 1.0, 0.0, 1),
        20: (1.0, -0.5, 1, 1.0, 0.0, 1),
        21: (1.0, -0.5, 1, 1.0, 0.0, 1),
        22: (1.0, -0.5, 1, 1.0, 0.0, 1),
        23: (1.0, -0.5, 1, 1.0, 0.0, 1),
    },
    "C.8": {
        1: (0.0, 0.2, 3, 1.0, 0.0, 1),
        2: (1.0, -0.5, 2, 1.0, 0.0, 2),
        3: (0.0, 0.2, 3, 1.0, 0.0, 1),
        4: (1.0, -0.5, 2, 1.0, 0.0, 2),
        5: (0.0, 0.2, 3, 1.0, 0.0, 1),
        6: (1.0, -0.5, 3, 1.0, 0.0, 3),
        7: (1.0, -0.5, 3, 1.0, 0.0, 2),
        8: (0.0, 0.2, 3, 1.0, 0.0, 1),
        9: (1.0, -0.5, 3, 1.0, 0.0, 3),
        10: (1.0, -0.5, 3, 1.0, 0.0, 2),
        11: (0.0, 0.2, 3, 1.0, 0.0, 1),
        12: (1.0, -0.5, 2, 1.0, 0.0, 2),
        13: (0.0, 0.2, 3, 1.0, 0.0, 1),
        14: (1.0, -0.5, 3, 1.0, 0.0, 3),
        15: (1.0, -0.5, 3, 1.0, 0.0, 2),
        16: (0.0, 0.2, 3, 1.0, 0.0, 1),
        17: (1.0, -0.5, 3, 1.0, 0.0, 3),
        18: (1.0, -0.5, 3, 1.0, 0.0, 2),
        19: (1.0, -0.5, 3, 1.0, 0.0, 3),
        20: (1.0, -0.5, 3, 1.0, 0.0, 3),
        21: (1.0, -0.5, 3, 1.0, 0.0, 3),
        22: (1.0, -0.5, 3, 1.0, 0.0, 3),
        23: (1.0, -0.5, 3, 1.0, 0.0, 3),
    },
}


# Edition 2023 of Annex C (EN ISO 52016-3:2023, used by the EPB Center case
# study and spreadsheet) differs from the 2026 text in Table C.4 only: external
# venetian blinds, manual/motorized control, night states 19-23 are fully
# extended by rule (1,00 / 0,00) instead of 0,00 / 0,10.
_ANNEX_C_2023_DIFFERENCES = {
    ("C.4", 19): (1.0, 0.0, 3), ("C.4", 20): (1.0, 0.0, 3), ("C.4", 21): (1.0, 0.0, 3),
    ("C.4", 22): (1.0, 0.0, 3), ("C.4", 23): (1.0, 0.0, 2),
}


def scenario_rows(table_id: str, edition: str = "2026") -> Dict[int, tuple]:
    """State table of Annex C for the given edition of the standard."""
    rows = dict(_SCENARIOS[table_id])
    if edition == "2023":
        for (tid, number), manual in _ANNEX_C_2023_DIFFERENCES.items():
            if tid == table_id:
                rows[number] = manual + rows[number][3:]
    return rows


# ---------------------------------------------------------------------------
# Identification of the Annex C state (Figure 2, Tables C.2-C.8)
# ---------------------------------------------------------------------------
def identify_state_number(day: str, hc: str, tint: str, rad: str, glare: str, dayl: str) -> int:
    """Return the state number 1..23 of Annex C for a combination of modes.

    ``day`` DA/NI, ``hc`` HE/NE/CO, ``tint`` TN/TH/TV, ``rad`` RL/RH,
    ``glare`` GN/GY, ``dayl`` LN/LL.  Modes that do not influence the state
    (e.g. RAD-MODE in heating mode) are ignored.
    """
    if day == "NI":
        if hc == "HE":
            return 19
        if hc == "NE":
            return 21 if tint == "TV" else 20
        return 23 if tint == "TV" else 22
    if hc == "HE":
        return 2 if glare == "GY" else 1
    base = {"NE": 3, "CO": 11}[hc]
    if tint == "TN":
        return base + (1 if glare == "GY" else 0)
    first = {"NE": 5, "CO": 13}[hc]
    if tint == "TH":
        return first if rad == "RL" else first + (2 if dayl == "LL" else 1)
    first = {"NE": 8, "CO": 16}[hc]
    return first if rad == "RL" else first + (2 if dayl == "LL" else 1)


def scenario_table_id(element_type: str, position: str, space_category: str) -> str:
    """Select the Annex C table for the element, its position and the space."""
    venetian = element_type == "venetian_blind"
    if space_category == "office":
        return "C.8" if venetian else "C.7"
    internal = position == "internal"
    if element_type == "shutter":
        if internal:
            raise ValueError("Annex C defines no reference scenario for internal shutters")
        return "C.3"
    if venetian:
        return "C.6" if internal else "C.4"
    return "C.5" if internal else "C.2"


def radiation_limits_w_m2(control: str, space_key: str, hc: str, tint: str) -> Optional[Tuple[float, float]]:
    """(RL, RH) of Table C.1 as irradiance on the element plane, W/m2.

    Returns ``None`` when RAD-MODE does not affect the state (heating mode,
    neutral mode with fine operative temperature).
    """
    column = _RAD_COLUMNS.get(f"{hc}-{tint}")
    if column is None:
        return None
    row = _RAD_LIMITS_KLUX.get(f"{control}-{space_key}")
    if row is None:
        raise ValueError(
            f"Table C.1 has no reference limits for control '{control}' in space '{space_key}'"
        )
    to_w_m2 = 1000.0 / LUMINOUS_EFFICACY_LM_W
    return row[column] * to_w_m2, row[column + 1] * to_w_m2


# ---------------------------------------------------------------------------
# Element description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpticalState:
    """Thermal, solar and daylight properties of one state (Table 7)."""

    g: float
    u: float
    tau_v: float = float("nan")

    def mixed(self, other: "OpticalState", fraction: float) -> "OpticalState":
        """Area-weighted mix: ``fraction`` of the element in ``other``."""
        w = float(np.clip(fraction, 0.0, 1.0))
        return OpticalState(
            (1.0 - w) * self.g + w * other.g,
            (1.0 - w) * self.u + w * other.u,
            (1.0 - w) * self.tau_v + w * other.tau_v,
        )


_ELEMENT_TYPES = {"roller_blind", "venetian_blind", "shutter", "chromogenic"}
_POSITIONS = {"external", "internal", "integrated"}
_CONTROL_ALIASES = {
    "manual": "MAN", "man": "MAN", "cntrl_man": "MAN",
    "motorized": "MOT", "motorised": "MOT", "mot": "MOT", "cntrl_mot": "MOT",
    "automated": "AUT", "automatic": "AUT", "aut": "AUT", "cntrl_auto": "AUT",
    "environmental": "PAS", "passive": "PAS", "pas": "PAS", "cntrl_pas": "PAS",
}
_SLAT_KEYS = ("slats_open", "slats_half_open", "slats_closed")


def _optical_state(name: str, data) -> OpticalState:
    if not isinstance(data, dict):
        raise ValueError(f"adaptive_envelope state '{name}' must be an object")
    for key in ("g_value", "u_value"):
        if key not in data:
            raise ValueError(f"adaptive_envelope state '{name}': '{key}' is required")
    g, u = float(data["g_value"]), float(data["u_value"])
    tau = float(data.get("tau_v", float("nan")))
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"adaptive_envelope state '{name}': g_value must be between 0 and 1")
    if u <= 0.0:
        raise ValueError(f"adaptive_envelope state '{name}': u_value must be > 0")
    if np.isfinite(tau) and not 0.0 <= tau <= 1.0:
        raise ValueError(f"adaptive_envelope state '{name}': tau_v must be between 0 and 1")
    return OpticalState(g, u, tau)


@dataclass(frozen=True)
class AdaptiveElementConfig:
    """Validated ``adaptive_envelope`` input of a transparent surface."""

    name: str
    element_type: str
    position: str
    control: str  # MAN / MOT / AUT / PAS
    space_category: str  # residential / office
    system: str  # heating / heating_cooling (office only)
    retracted: Optional[OpticalState]
    extended: Tuple[OpticalState, ...]  # 1 entry, or 3 for venetian blinds
    table_id: Optional[str]
    annex_c_edition: str
    hc_method: str
    day_hours: Optional[Tuple[int, int]]
    sleeping_hours: Tuple[int, int]
    daylight: Optional[Tuple[float, float, float]]  # a_dayl, b_dayl, H_gl;1m
    # environmentally activated chromogenic glazing (6.7)
    env_variable: str = "solar_irradiance"
    env_min: Optional[Tuple[float, OpticalState]] = None
    env_max: Optional[Tuple[float, OpticalState]] = None
    env_hysteresis: float = 0.0
    env_discrete: Tuple[Tuple[float, OpticalState], ...] = ()  # discrete states, ascending x
    glare_limit_lx: float = GLARE_LIMIT_LX_FORMULA

    @property
    def is_residential(self) -> bool:
        return self.space_category == "residential"

    @property
    def space_key(self) -> str:
        if self.is_residential:
            return "RES"
        return "OFFHECO" if self.system == "heating_cooling" else "OFFHE"


def _norm(value) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def parse_adaptive_envelope(surface: dict) -> AdaptiveElementConfig:
    """Validate ``surface["adaptive_envelope"]`` (see module docstring)."""
    name = str(surface.get("name", ""))
    data = surface.get("adaptive_envelope")
    if not isinstance(data, dict):
        raise ValueError(f"Window '{name}': adaptive_envelope must be an object")

    etype = _norm(data.get("type", ""))
    if etype not in _ELEMENT_TYPES:
        raise ValueError(
            f"Window '{name}': adaptive_envelope.type must be one of {sorted(_ELEMENT_TYPES)}"
        )
    control = _CONTROL_ALIASES.get(_norm(data.get("control_type", "")))
    if control is None:
        raise ValueError(
            f"Window '{name}': adaptive_envelope.control_type must be manual, motorized, "
            "automated or environmental (CNTRL_MAN/MOT/AUTO/PAS)"
        )
    control_opts = data.get("control", {}) or {}
    if not isinstance(control_opts, dict):
        raise ValueError(f"Window '{name}': adaptive_envelope.control must be an object")
    states = data.get("states")
    if not isinstance(states, dict):
        raise ValueError(f"Window '{name}': adaptive_envelope.states must be an object")

    edition = str(control_opts.get("annex_c_edition", "2026"))
    if edition not in {"2023", "2026"}:
        raise ValueError(f"Window '{name}': control.annex_c_edition must be '2023' or '2026'")
    glare_policy = _norm(control_opts.get("glare_policy", "formula"))
    if glare_policy not in {"formula", "table_33_note"}:
        raise ValueError(f"Window '{name}': control.glare_policy must be 'formula' or 'table_33_note'")
    glare_limit_lx = (
        GLARE_LIMIT_LX_FORMULA if glare_policy == "formula" else GLARE_LIMIT_LX_TABLE33_NOTE
    )
    if control_opts.get("glare_limit_lx") is not None:
        glare_limit_lx = float(control_opts["glare_limit_lx"])
        if glare_limit_lx <= 0.0:
            raise ValueError(f"Window '{name}': control.glare_limit_lx must be > 0")
    hc_method = str(control_opts.get("hc_method", "AM1")).upper()
    if hc_method not in {"AM1", "AM2"}:
        raise ValueError(f"Window '{name}': control.hc_method must be AM1 or AM2")
    day_hours = control_opts.get("daytime_hours")
    if day_hours is not None:
        day_hours = (int(day_hours[0]), int(day_hours[1]))
    sleeping = control_opts.get("sleeping_hours", (23, 7))
    sleeping = (int(sleeping[0]), int(sleeping[1]))

    if control == "PAS":
        if etype != "chromogenic":
            raise ValueError(f"Window '{name}': environmental control applies to chromogenic glazing only")
        var = _norm(control_opts.get("environment_variable", "solar_irradiance"))
        if var not in {"solar_irradiance", "outdoor_air_temperature", "indoor_operative_temperature"}:
            raise ValueError(
                f"Window '{name}': control.environment_variable must be solar_irradiance, "
                "outdoor_air_temperature or indoor_operative_temperature"
            )
        default_h = HYSTERESIS_DEFAULT["solar_irradiance" if var == "solar_irradiance" else "temperature"]
        hyst = float(control_opts.get("hysteresis", default_h))
        if hyst < 0.0:
            raise ValueError(f"Window '{name}': control.hysteresis must be >= 0")
        common = dict(env_variable=var, env_hysteresis=hyst, glare_limit_lx=glare_limit_lx)
        discrete = states.get("discrete")
        if discrete is not None:  # 6.7: discrete change of states
            if not isinstance(discrete, (list, tuple)) or len(discrete) < 2:
                raise ValueError(f"Window '{name}': states.discrete needs at least two states")
            steps = []
            for i, item in enumerate(discrete):
                if not isinstance(item, dict) or "x" not in item:
                    raise ValueError(f"Window '{name}': states.discrete[{i}] needs the value 'x'")
                steps.append((float(item["x"]), _optical_state(f"discrete[{i}]", item)))
            if any(b[0] <= a[0] for a, b in zip(steps, steps[1:])):
                raise ValueError(f"Window '{name}': states.discrete 'x' values must be strictly ascending")
            return AdaptiveElementConfig(
                name, etype, "integrated", control, "residential", "heating", None, (), None,
                "2026", hc_method, day_hours, sleeping, None, env_discrete=tuple(steps), **common,
            )
        lo, hi = states.get("state_min"), states.get("state_max")
        if not isinstance(lo, dict) or not isinstance(hi, dict) or "x" not in lo or "x" not in hi:
            raise ValueError(
                f"Window '{name}': environmental control needs states.state_min and states.state_max "
                "(each with the environmental value 'x') or states.discrete"
            )
        x_lo, x_hi = float(lo["x"]), float(hi["x"])
        if x_lo >= x_hi:
            raise ValueError(
                f"Window '{name}': state_min.x must be lower than state_max.x (6.4.3.2.1)"
            )
        return AdaptiveElementConfig(
            name, etype, "integrated", control, "residential", "heating", None, (), None,
            "2026", hc_method, day_hours, sleeping, None,
            env_min=(x_lo, _optical_state("state_min", lo)),
            env_max=(x_hi, _optical_state("state_max", hi)), **common,
        )

    position = _norm(data.get("position", "external"))
    if position not in _POSITIONS:
        raise ValueError(f"Window '{name}': adaptive_envelope.position must be one of {sorted(_POSITIONS)}")
    space = _norm(data.get("space_category", "residential"))
    if space not in {"residential", "office"}:
        raise ValueError(f"Window '{name}': adaptive_envelope.space_category must be residential or office")
    system = _norm(data.get("system", "heating"))
    if system not in {"heating", "heating_cooling"}:
        raise ValueError(f"Window '{name}': adaptive_envelope.system must be heating or heating_cooling")
    if space == "office" and control == "MAN" and system == "heating_cooling":
        raise ValueError(
            f"Window '{name}': Table C.1 defines no manual control limits for an office with "
            "heating and cooling; use motorized or automated control"
        )

    retracted = _optical_state("retracted", states.get("retracted"))
    ext = states.get("extended")
    if etype == "venetian_blind":
        if not isinstance(ext, dict) or any(k not in ext for k in _SLAT_KEYS):
            raise ValueError(
                f"Window '{name}': venetian_blind needs states.extended with {list(_SLAT_KEYS)}"
            )
        extended = tuple(_optical_state(k, ext[k]) for k in _SLAT_KEYS)
    else:
        extended = (_optical_state("extended", ext),)

    daylight = control_opts.get("daylight")
    if daylight is not None:
        try:
            daylight = (
                float(daylight["a_dayl"]), float(daylight["b_dayl"]), float(daylight["H_gl_1m"])
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Window '{name}': control.daylight needs a_dayl, b_dayl and H_gl_1m (Formula 4)"
            ) from exc

    return AdaptiveElementConfig(
        name, etype, position, control, space, system, retracted, extended,
        scenario_table_id(etype, position, space), edition, hc_method, day_hours, sleeping, daylight,
        glare_limit_lx=glare_limit_lx,
    )


# ---------------------------------------------------------------------------
# Per-window controller
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveState:
    """Result of the control scenario for one window and one hour."""

    modes: str  # e.g. "DA-HE-TN-RL-GN-LN"
    state_number: int  # Annex C number, 0 for environmental control
    fraction: float  # fraction of the element in the extended state
    ext_state: int  # secondary state: slat position 1..3 (venetian), else 1
    held: bool  # manual control, occupant not awake: previous position kept
    properties: OpticalState
    irradiance: float = 0.0  # on the element plane, W/m2 (weight of the mean g and tau_V)

    @property
    def extended(self) -> bool:
        return self.fraction > 0.5


class AdaptiveEnvelopeController:
    """Sequential state machine of one adaptive element (needs hourly order)."""

    def __init__(self, cfg: AdaptiveElementConfig):
        self.cfg = cfg
        self._rows = scenario_rows(cfg.table_id, cfg.annex_c_edition) if cfg.table_id else {}
        self._rad = "RL"
        self._fraction = 0.0
        self._ext = 1
        self._x_prev: Optional[float] = None
        self._direction = 1.0
        self.log: Dict[int, AdaptiveState] = {}

    def state_at(self, ctx: "AdaptiveEnvelopeContext", tstep: int, irradiance: float) -> AdaptiveState:
        cached = self.log.get(tstep)
        if cached is not None:
            return cached
        st = self._environmental(ctx, tstep, irradiance) if self.cfg.control == "PAS" else self._reference(
            ctx, tstep, irradiance
        )
        st.irradiance = max(0.0, float(irradiance))
        self.log[tstep] = st
        return st

    # -- 6.7 -------------------------------------------------------------
    def _environmental(self, ctx, t: int, irradiance: float) -> AdaptiveState:
        cfg = self.cfg
        x = ctx.environment_value(cfg.env_variable, t, irradiance)
        if self._x_prev is not None:
            if x > self._x_prev:
                self._direction = 1.0
            elif x < self._x_prev:
                self._direction = -1.0
        self._x_prev = x
        if cfg.env_discrete:  # 6.7: the state changes when X has reached the next state
            shifted = x + self._direction * cfg.env_hysteresis / 2.0
            index = max([i for i, (x_i, _) in enumerate(cfg.env_discrete) if shifted >= x_i], default=0)
            frac = index / (len(cfg.env_discrete) - 1)
            return AdaptiveState("PAS", 0, frac, 1, False, cfg.env_discrete[index][1])
        (x_lo, y_lo), (x_hi, y_hi) = cfg.env_min, cfg.env_max
        # Formula (1): +dX/2 when X rises, -dX/2 when it falls
        w = (x - x_lo + self._direction * cfg.env_hysteresis / 2.0) / (x_hi - x_lo)
        w = float(np.clip(w, 0.0, 1.0))
        props = y_lo.mixed(y_hi, w)
        return AdaptiveState("PAS", 0, w, 1, False, props)

    def _properties_of_state(self, number: int, manual: bool, awake: bool) -> OpticalState:
        """Properties of an Annex C state (used for tau_V in Formula 4)."""
        cfg = self.cfg
        row = self._rows[number]
        if manual:
            fraction, ext = float(np.clip(row[0] + row[1], 0.0, 1.0)), row[2]
        else:
            fraction = float(np.clip(row[3] + (row[4] if awake else 0.0), 0.0, 1.0))
            ext = row[5]
        return cfg.retracted.mixed(cfg.extended[min(ext, len(cfg.extended)) - 1], fraction)

    # -- 6.8 / Annex C ----------------------------------------------------
    def _reference(self, ctx, t: int, irradiance: float) -> AdaptiveState:
        cfg = self.cfg
        manual = cfg.control in ("MAN", "MOT")
        day = "DA" if ctx.is_day(t, cfg.day_hours) else "NI"
        hc = ctx.hc_mode(t, cfg.hc_method)
        tint = ctx.tint_mode(t)
        e_lx = LUMINOUS_EFFICACY_LM_W * max(0.0, float(irradiance))

        limits = radiation_limits_w_m2(cfg.control, cfg.space_key, hc, tint)
        if limits is not None:  # between RL and RH: no change of mode (Table 22)
            if irradiance <= limits[0]:
                self._rad = "RL"
            elif irradiance >= limits[1]:
                self._rad = "RH"

        occupied = ctx.occupied(t)
        awake = occupied and not (cfg.is_residential and ctx.sleeping(t, cfg.sleeping_hours))
        # visual conditions are only assessed when the control "sees" them
        # (6.8.5.2 c, f): residential automated control has no presence
        # detection, office control only reacts to awake occupants
        assess_visual = awake or (cfg.is_residential and not manual)

        glare = "GN"
        if not cfg.is_residential and awake and day == "DA":
            # Formula (3), P_dglare > 0,45  <=>  E_V;e;w > glare_limit_lx
            glare = "GY" if e_lx > cfg.glare_limit_lx else "GN"
        dayl = "LN"
        if cfg.daylight is not None and day == "DA" and assess_visual:
            a_dayl, b_dayl, h_gl = cfg.daylight
            # tau_V;gl of Formula (4) is that of the state before the daylight
            # check (same hour), which avoids the circular dependency
            interim = identify_state_number(day, hc, tint, self._rad, glare, "LN")
            tau = self._properties_of_state(interim, manual, awake).tau_v
            if np.isfinite(tau):
                e_desk = min(e_lx, a_dayl * tau * h_gl * e_lx + b_dayl)  # Formula (4)
                dayl = "LL" if e_desk < DAYLIGHT_DESK_LIMIT_LX else "LN"

        number = identify_state_number(day, hc, tint, self._rad, glare, dayl)
        row = self._rows[number]
        held = False
        if manual:
            if awake:
                fraction, ext = float(np.clip(row[0] + row[1], 0.0, 1.0)), row[2]
            else:  # same position as the last occupied hour
                fraction, ext, held = self._fraction, self._ext, True
        else:
            correction = row[4] if awake else 0.0  # occupants intervene only if present
            fraction, ext = float(np.clip(row[3] + correction, 0.0, 1.0)), row[5]
        self._fraction, self._ext = fraction, ext

        extended = cfg.extended[min(ext, len(cfg.extended)) - 1]
        props = cfg.retracted.mixed(extended, fraction)
        modes = f"{day}-{hc}-{tint}-{self._rad}-{glare}-{dayl}"
        return AdaptiveState(modes, number, fraction, ext, held, props)


# ---------------------------------------------------------------------------
# Thermal-zone context: shared inputs and the history fed back by the solver
# ---------------------------------------------------------------------------
def _hour_in_window(hour: float, start: int, end: int) -> bool:
    start, end = start % 24, end % 24
    if start == end:
        return True
    return start <= hour < end if start < end else hour >= start or hour < end


class AdaptiveEnvelopeContext:
    """Inputs shared by the adaptive elements of one thermal zone.

    The hourly solver calls :meth:`record` at the end of every time step with
    the operative temperature and the heating(+)/cooling(-) load.  The control
    scenario of step t only uses data of earlier steps (6.8.4, Tables 25-26).
    """

    def __init__(self, surfaces, sim_df: pd.DataFrame, occupancy=None, dt_h: float = 1.0):
        self.index = sim_df.index
        n = len(sim_df)
        self.dt_h = float(dt_h) if dt_h else 1.0
        self.steps_per_day = max(1, int(round(24.0 / self.dt_h)))
        self.theta_op = np.full(n, np.nan)
        self.phi_hc = np.full(n, np.nan)
        self.theta_e = (
            pd.to_numeric(sim_df["T2m"], errors="coerce").to_numpy(dtype=float)
            if "T2m" in sim_df.columns else np.full(n, np.nan)
        )
        self.occupancy = (
            np.asarray(occupancy, dtype=float) if occupancy is not None else np.ones(n, dtype=float)
        )
        if len(self.occupancy) < n:
            self.occupancy = np.pad(self.occupancy, (0, n - len(self.occupancy)), constant_values=1.0)
        self._irradiance_day = self._daylight_flags(sim_df)
        hours = getattr(self.index, "hour", None)
        minutes = getattr(self.index, "minute", None)
        self.clock = (
            np.asarray(hours, dtype=float) + np.asarray(minutes, dtype=float) / 60.0
            if hours is not None else np.arange(n, dtype=float) % 24.0
        )
        self._hc_cache: Dict[Tuple[int, str], str] = {}
        self.controllers: Dict[str, AdaptiveEnvelopeController] = {}
        for surface in surfaces:
            if surface.get("adaptive_envelope") is not None:
                key = str(surface.get("name", ""))
                self.controllers[key] = AdaptiveEnvelopeController(parse_adaptive_envelope(surface))

    # pandas deep-copies DataFrame.attrs on almost every column access; the
    # context is shared mutable state and must never be copied
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    # -- inputs --------------------------------------------------------------
    @staticmethod
    def _daylight_flags(sim_df: pd.DataFrame, threshold_w_m2: float = 1.0) -> np.ndarray:
        """DAY-MODE method MM1: calculated solar irradiance above a low value."""
        if "Gd(h)" in sim_df.columns and "Gb(n)" in sim_df.columns:
            glob = (
                pd.to_numeric(sim_df["Gd(h)"], errors="coerce").fillna(0.0)
                + pd.to_numeric(sim_df["Gb(n)"], errors="coerce").fillna(0.0)
            ).to_numpy(dtype=float)
        else:
            cols = [c for c in sim_df.columns if str(c).startswith("I_sol_tot_")]
            glob = (
                sim_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1).to_numpy(dtype=float)
                if cols else np.zeros(len(sim_df))
            )
        return glob > threshold_w_m2

    def record(self, tstep: int, theta_op: float, phi_hc: float) -> None:
        if 0 <= tstep < len(self.theta_op):
            self.theta_op[tstep] = float(theta_op)
            self.phi_hc[tstep] = float(phi_hc)

    # -- modes -------------------------------------------------------------
    def is_day(self, t: int, fixed_hours: Optional[Tuple[int, int]] = None) -> bool:
        if fixed_hours is not None:
            return _hour_in_window(self.clock[t], fixed_hours[0], fixed_hours[1])
        return bool(self._irradiance_day[t])

    def occupied(self, t: int) -> bool:
        return bool(self.occupancy[t] > 0.0)

    def sleeping(self, t: int, hours: Tuple[int, int]) -> bool:
        return _hour_in_window(self.clock[t], hours[0], hours[1])

    def tint_mode(self, t: int) -> str:
        """Table 26 (MM1/AM1): operative temperature of the previous interval."""
        prev = self.theta_op[t - 1] if t >= 1 else np.nan
        if not np.isfinite(prev):
            return "TN"
        if prev > TINT_LIMITS_C[1]:
            return "TV"
        return "TH" if prev > TINT_LIMITS_C[0] else "TN"

    def hc_mode(self, t: int, method: str = "AM1") -> str:
        """Table 25: previous day = last completed 24 h period starting 07h00."""
        offset = (self.clock[t] - OCCUPANT_FRACTION_DAY_START_H) % 24.0
        end = t - int(round(offset / self.dt_h))
        start = end - self.steps_per_day
        if start < 0:
            return "NE"
        key = (end, method)
        cached = self._hc_cache.get(key)
        if cached is not None:
            return cached
        mode = "NE"
        if method == "AM2":
            t_e = self.theta_e[start:end]
            if np.all(np.isfinite(t_e)):
                mean_e = float(np.mean(t_e))
                mode = "HE" if mean_e < HC_OUTDOOR_LIMITS_C[0] else "CO" if mean_e > HC_OUTDOOR_LIMITS_C[1] else "NE"
        else:
            op, phi = self.theta_op[start:end], self.phi_hc[start:end]
            if np.all(np.isfinite(op)) and np.all(np.isfinite(phi)):
                q_h, q_c = float(np.sum(np.maximum(phi, 0.0))), float(np.sum(np.maximum(-phi, 0.0)))
                mean_op = float(np.mean(op))
                if q_h > q_c or mean_op < HC_OPERATIVE_LIMITS_C[0]:
                    mode = "HE"
                elif q_c > q_h or mean_op > HC_OPERATIVE_LIMITS_C[1]:
                    mode = "CO"
        self._hc_cache[key] = mode
        return mode

    def environment_value(self, variable: str, t: int, irradiance: float) -> float:
        if variable == "solar_irradiance":
            return float(irradiance)
        if variable == "outdoor_air_temperature":
            return float(self.theta_e[t])
        prev = self.theta_op[t - 1] if t >= 1 else np.nan
        return float(prev) if np.isfinite(prev) else 20.0

    # -- results -----------------------------------------------------------
    def properties(self, surface: dict, tstep: int, irradiance: float) -> Tuple[bool, float, float]:
        """``(extended, g, u)`` of a window at ``tstep`` (cached per step)."""
        ctrl = self.controllers[str(surface.get("name", ""))]
        st = ctrl.state_at(self, tstep, irradiance)
        return st.extended, st.properties.g, st.properties.u

    def to_dataframe(self) -> pd.DataFrame:
        """Hourly state log: one block of columns per adaptive window."""
        frames = {}
        for name, ctrl in self.controllers.items():
            if not ctrl.log:
                continue
            steps = sorted(ctrl.log)
            rows = [ctrl.log[s] for s in steps]
            idx = self.index[steps]
            prefix = f"adaptive_{name}"
            frames[f"{prefix}_modes"] = pd.Series([r.modes for r in rows], index=idx)
            frames[f"{prefix}_state"] = pd.Series([r.state_number for r in rows], index=idx)
            frames[f"{prefix}_fraction"] = pd.Series([r.fraction for r in rows], index=idx)
            frames[f"{prefix}_ext_state"] = pd.Series([r.ext_state for r in rows], index=idx)
            frames[f"{prefix}_g"] = pd.Series([r.properties.g for r in rows], index=idx)
            frames[f"{prefix}_u"] = pd.Series([r.properties.u for r in rows], index=idx)
            frames[f"{prefix}_tau_v"] = pd.Series([r.properties.tau_v for r in rows], index=idx)
            frames[f"{prefix}_I_sol"] = pd.Series([r.irradiance for r in rows], index=idx)
        return pd.DataFrame(frames)


def has_adaptive_envelope(surface) -> bool:
    return isinstance(surface, dict) and surface.get("adaptive_envelope") is not None


# ---------------------------------------------------------------------------
# 6.10 Post-processing: key performance indicators (Table 6)
# ---------------------------------------------------------------------------
def adaptive_envelope_kpis(
    adaptive: pd.DataFrame,
    reference: pd.DataFrame,
    useful_floor_area_m2: float,
    occupancy=None,
    low_limit_c: float = 20.0,
    high_limit_c: float = 25.0,
    time_step_h: float = 1.0,
) -> Dict[str, float]:
    """Difference of the thermal zone with the adaptive element vs a reference.

    ``adaptive`` and ``reference`` are hourly results with the columns ``Q_H``
    and ``Q_C`` (W) and ``T_op`` (degC).  Differences are adaptive - reference,
    so a negative value is an improvement.  Accumulated over/under-temperature
    is counted during occupancy (``occupancy`` > 0, default: all hours).
    """
    if useful_floor_area_m2 <= 0.0:
        raise ValueError("useful_floor_area_m2 must be > 0")

    def _annual(df) -> Tuple[float, float]:
        q_h = float(pd.to_numeric(df["Q_H"], errors="coerce").fillna(0.0).sum()) * time_step_h / 1000.0
        q_c = float(pd.to_numeric(df["Q_C"], errors="coerce").fillna(0.0).sum()) * time_step_h / 1000.0
        return q_h / useful_floor_area_m2, q_c / useful_floor_area_m2

    def _kelvin_hours(df) -> Tuple[float, float]:
        t_op = pd.to_numeric(df["T_op"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(t_op)
        if occupancy is not None:
            occ = np.asarray(occupancy, dtype=float)[: len(t_op)] > 0.0
            mask = mask & np.pad(occ, (0, len(t_op) - len(occ)), constant_values=False)
        low = float(np.sum(np.maximum(low_limit_c - t_op[mask], 0.0))) * time_step_h
        high = float(np.sum(np.maximum(t_op[mask] - high_limit_c, 0.0))) * time_step_h
        return low, high

    def _rel(delta: float, ref: float) -> float:
        return 100.0 * delta / ref if abs(ref) > 1e-12 else float("nan")

    h_a, c_a = _annual(adaptive)
    h_r, c_r = _annual(reference)
    low_a, high_a = _kelvin_hours(adaptive)
    low_r, high_r = _kelvin_hours(reference)
    kpi = {
        "dQ_H_nd_kWh_m2": h_a - h_r,
        "dQ_C_nd_kWh_m2": c_a - c_r,
        "dQ_HC_nd_kWh_m2": (h_a + c_a) - (h_r + c_r),
        "dQ_H_nd_pct": _rel(h_a - h_r, h_r),
        "dQ_C_nd_pct": _rel(c_a - c_r, c_r),
        "dQ_HC_nd_pct": _rel((h_a + c_a) - (h_r + c_r), h_r + c_r),
        "dT_low_Kh": low_a - low_r,
        "dT_high_Kh": high_a - high_r,
        "dT_low_pct": _rel(low_a - low_r, low_r),
        "dT_high_pct": _rel(high_a - high_r, high_r),
    }
    return kpi


def adaptive_envelope_mean_properties(
    state_log: pd.DataFrame,
    window: str,
    irradiance_w_m2,
    theta_op,
    theta_e,
) -> Dict[str, float]:
    """Weighted mean U, g and tau_V of one window (Table 6, 6.10).

    U is weighted by |theta_int;op - theta_e;a|, g by the incident solar
    irradiance and tau_V by the incident illuminance (K_V * irradiance).
    """
    prefix = f"adaptive_{window}"
    idx = state_log.index
    irr = pd.Series(np.asarray(irradiance_w_m2, dtype=float), index=idx).clip(lower=0.0)
    dtheta = (
        pd.Series(np.asarray(theta_op, dtype=float), index=idx)
        - pd.Series(np.asarray(theta_e, dtype=float), index=idx)
    ).abs()

    def _w(values: pd.Series, weights: pd.Series) -> float:
        mask = values.notna() & weights.notna()
        total = float(weights[mask].sum())
        return float((values[mask] * weights[mask]).sum() / total) if total > 0.0 else float("nan")

    return {
        "U_w_mn": _w(state_log[f"{prefix}_u"], dtheta),
        "g_gl_mn": _w(state_log[f"{prefix}_g"], irr),
        "tau_V_gl_mn": _w(state_log[f"{prefix}_tau_v"], irr * LUMINOUS_EFFICACY_LM_W),
    }


def _adaptive_windows(hourly: pd.DataFrame) -> list:
    return sorted(
        c[len("adaptive_"):-len("_modes")]
        for c in hourly.columns
        if str(c).startswith("adaptive_") and str(c).endswith("_modes")
    )


def adaptive_envelope_monthly_report(
    adaptive: pd.DataFrame,
    reference: pd.DataFrame,
    useful_floor_area_m2: float,
    occupancy=None,
    **kpi_kwargs,
) -> pd.DataFrame:
    """Monthly and annual key performance indicators of 6.10 (Table 6).

    One row per month (1..12) and one row ``"year"``: the differences of the
    energy needs and of the accumulated under/over-temperature against the
    reference, and for every adaptive window the mean U-value (weighted by
    |theta_int;op - theta_e;a|), g-value (weighted by the incident
    irradiance) and daylight transmittance (weighted by the incident
    illuminance), as columns ``<window>:U_w_mn`` etc.  ``adaptive`` is the
    hourly result of the engine with the ``adaptive_<window>_*`` columns.
    """
    reference = reference.reindex(adaptive.index)
    occ = None if occupancy is None else np.asarray(occupancy, dtype=float)[: len(adaptive)]
    months = np.asarray(adaptive.index.month)
    windows = _adaptive_windows(adaptive)
    rows = {}
    for key, mask in [(m, months == m) for m in range(1, 13)] + [("year", np.ones(len(adaptive), bool))]:
        if not mask.any():
            continue
        a, r = adaptive[mask], reference[mask]
        row = adaptive_envelope_kpis(
            a, r, useful_floor_area_m2, occupancy=None if occ is None else occ[mask], **kpi_kwargs
        )
        for w in windows:
            means = adaptive_envelope_mean_properties(
                a, w, a[f"adaptive_{w}_I_sol"], a["T_op"], a["T_ext"]
            )
            row.update({f"{w}:{name}": value for name, value in means.items()})
        rows[key] = row
    return pd.DataFrame(rows).T


def adaptive_envelope_state_statistics(hourly: pd.DataFrame) -> Dict[str, dict]:
    """Use of the states per adaptive window (6.10 "statistics", Table 8 of the case study).

    ``occurrences`` per Annex C state number, ``changes`` = number of hours in
    which U, g or tau_V differ from the previous hour, and ``mean_hours_without_change``.
    """
    out = {}
    for w in _adaptive_windows(hourly):
        state = hourly[f"adaptive_{w}_state"]
        props = hourly[[f"adaptive_{w}_{p}" for p in ("g", "u", "tau_v")]].round(9)
        changes = int((props.diff().abs().sum(axis=1) > 0.0).sum())
        out[w] = {
            "occurrences": {int(k): int(v) for k, v in state.value_counts().sort_index().items()},
            "changes": changes,
            "mean_hours_without_change": float(len(hourly) / changes) if changes else float("inf"),
        }
    return out


_NOT_IMPLEMENTED = (
    "detailed adaptive building envelope elements (6.4.4, 6.5.2: nodes, long-wave radiation, "
    "convergence iteration)",
    "adaptive elements with actively ventilated cavity (6.4.3.3, Tables 9-11, 34)",
    "RAD-MODE method AC2/AM2; GLARE-MODE methods AC2/AM2 and AC3/AM3; HFLOW-MODE (not worked out "
    "in the standard)",
    "peak shaving and protection against view-in (not part of the reference control scenario)",
    "control type CNTRL_INTEGR (no reference control scenario specified)",
    "user-behaviour approaches 2 and 3 (not worked out in the standard)",
)


def adaptive_envelope_conformity_report(building_object: dict, hourly: Optional[pd.DataFrame] = None) -> dict:
    """Report of the selected options and methods required by clause 8.

    Lists, per adaptive window, the element, control type and every choice of
    6.8.1-6.8.4 and Annex A/B that the calculation applied, the not implemented
    parts, and (when ``hourly`` is given) the state statistics.
    """
    elements = {}
    for surface in building_object.get("building_surface", []):
        if not has_adaptive_envelope(surface):
            continue
        cfg = parse_adaptive_envelope(surface)
        fixed_day = cfg.day_hours is not None
        elements[cfg.name] = {
            "element_type": cfg.element_type,
            "position": cfg.position,
            "control_type": {"MAN": "CNTRL_MAN", "MOT": "CNTRL_MOT", "AUT": "CNTRL_AUTO", "PAS": "CNTRL_PAS"}[cfg.control],
            "space_category": cfg.space_category,
            "system": cfg.system,
            "annex_c_edition": cfg.annex_c_edition,
            "annex_c_table": cfg.table_id,
            "control_scenario": "reference (6.8.7, Annex C)" if cfg.control != "PAS" else "environmentally activated (6.7)",
            "methods": {
                "DAY-MODE": "MM2/AM3 fixed hours %s" % (cfg.day_hours,) if fixed_day else "MM1/AM1 calculated solar irradiance",
                "HC-MODE": f"{cfg.hc_method} ({'operative temperature and loads of the previous day' if cfg.hc_method == 'AM1' else 'outdoor air temperature of the previous day'})",
                "TINT-MODE": "MM1/AM1 operative temperature of the previous interval",
                "RAD-MODE": "MM1/AM1 Table C.1 limits with hysteresis, E_V = 115 lm/W x irradiance",
                "OCC-MODE": "AM1 occupancy schedule of ISO 52016-1; sleeping hours %s (residential)" % (cfg.sleeping_hours,),
                "GLARE-MODE": "AM1 Formula (3), limit %.0f lx (office only)" % cfg.glare_limit_lx,
                "DAYL-MODE": (
                    "AM1 Formula (4), a_dayl=%g, b_dayl=%g, H_gl;1m=%g" % cfg.daylight
                    if cfg.daylight else "not assessed (a_dayl, b_dayl not given): LN"
                ),
                "user behaviour": "approach 1 (fraction of the element unchanged)",
            },
            "states": {
                "retracted": cfg.retracted.__dict__ if cfg.retracted else None,
                "extended": [s.__dict__ for s in cfg.extended],
            },
        }
    report = {"elements": elements, "not_implemented": list(_NOT_IMPLEMENTED)}
    if hourly is not None:
        report["state_statistics"] = adaptive_envelope_state_statistics(hourly)
    return report


def format_conformity_report(report: dict) -> str:
    """Plain-text rendering of :func:`adaptive_envelope_conformity_report`."""
    lines = ["EN ISO 52016-3 - conformity report (clause 8)", ""]
    for name, el in report["elements"].items():
        lines.append(f"Adaptive element '{name}': {el['element_type']}, {el['position']}, "
                     f"{el['control_type']}, {el['space_category']} ({el['system']})")
        lines.append(f"  control scenario: {el['control_scenario']}; Annex C {el['annex_c_edition']}, table {el['annex_c_table']}")
        lines += [f"  {k}: {v}" for k, v in el["methods"].items()]
        for key, stat in report.get("state_statistics", {}).get(name, {}).items():
            lines.append(f"  statistics {key}: {stat}")
        lines.append("")
    lines.append("Not implemented:")
    lines += [f"  - {item}" for item in report["not_implemented"]]
    return "\n".join(lines)
