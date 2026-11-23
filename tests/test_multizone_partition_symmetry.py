"""INTERNAL partitions must be symmetric in the declaring zone.

The face of an INTERNAL partition that points into the adjacent
(simulated) zone is an internal surface of that zone (EN ISO 52016-1):
both faces couple with the internal film coefficients.  Before the fix,
the adjacent-zone face used the surface's *external* coefficient fields,
so a partition declared with the usual wall-row coefficients (internal
2.2/5.13, external 11.9/4.14) got different films on its two faces and
the results depended on which zone declared the partition.

``build_case_960`` masks the issue by overriding the external fields to
the internal values; these tests remove the override and check that

1. declaring the partition from zone A or zone B gives identical hourly
   results (left-right symmetry of the same physical building);
2. the generic declaration reproduces the published (override) case,
   i.e. the override is redundant once both faces use internal films.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ashrae140_bestest"
sys.path.insert(0, str(EXAMPLES_DIR))

from case_definitions import (  # noqa: E402
    COMMON_WALL_960, WEATHER_FILE, _coefficients, build_case_960, layers_resistance,
)

pytestmark = pytest.mark.skipif(not Path(WEATHER_FILE).exists(), reason="weather file missing")

from pybuildingenergy.source.utils import ISO52016  # noqa: E402


def _generic(declare_from="back"):
    """Case 960 without the coefficient override: the partition keeps the
    wall-row external fields; ``u_value`` is adjusted so the core
    resistance stays R_c = 0.20/0.51 (only the face films are at stake)."""
    bui = build_case_960("ashrae_convective")
    h_ci, h_ri, h_ce, h_re = _coefficients("wall", "ashrae_convective")
    rc = layers_resistance(COMMON_WALL_960)
    for s in bui["building_surface"]:
        if s.get("name") == "Common wall":
            s["convective_heat_transfer_coefficient_external"] = h_ce
            s["radiative_heat_transfer_coefficient_external"] = h_re
            s["u_value"] = 1.0 / (1.0 / (h_ci + h_ri) + rc + 1.0 / (h_ce + h_re))
            if declare_from == "sun":
                s["zone"], s["adjacent_zone"] = "sun", "back"
    return bui


def _run(bui):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ISO52016.simulate_envelope_multizone_free_floating(
            bui,
            path_weather_file=str(WEATHER_FILE),
            weather_source="epw",
            c_int_per_A_us=0.0,
            hvac_control_variable="air",
        )
    return out[0] if isinstance(out, tuple) else out


COLS = ["T_air_back", "T_air_sun", "Q_HVAC_back", "Q_HVAC_sun"]


def _assert_same(df_a, df_b):
    # Tolerances: temperatures at 1e-9 K; powers at 1e-6 W (the direct
    # solver's row ordering changes with the declaration order, leaving
    # ~1e-13 relative round-off on kW-scale signals).
    for col in COLS:
        tol = 1e-9 if col.startswith("T_") else 1e-6
        a = np.asarray(df_a[col], dtype=float)
        b = np.asarray(df_b[col], dtype=float)
        assert np.max(np.abs(a - b)) < tol, (
            f"{col}: max |diff| {np.max(np.abs(a - b)):.3e} >= {tol:.0e}"
        )


def test_partition_declaration_is_symmetric():
    _assert_same(_run(_generic("back")), _run(_generic("sun")))


def test_generic_declaration_matches_the_published_override():
    _assert_same(_run(build_case_960("ashrae_convective")), _run(_generic("back")))
