"""Loader for ISO 52016-1:2017's own BESTEST climate data spreadsheet.

Clause 7.2.2.12 of the standard points to this file instead of an EPW:
``ISO_52016_1_BESTEST_ClimData_2016.08.24.xls``, built from the original
BESTEST "DRYCOLD.TMY" weather file (NOT a modern TMY3 -- a different,
older, specifically-curated-for-BESTEST dataset), already converted to
direct+diffuse solar irradiance per orientation (N, E, S, W, horizontal)
via ISO 52010-1, exactly as clause 7.2.2.12 describes.

This file is ISO's own copyrighted standard content (obtained alongside
the standard itself), not redistributable the way the BSD-licensed
NatLabRockies/BESTEST-GSR EPW is -- so it is NOT shipped in this
repository. If you have your own copy, point ``ISO_CLIMATE_XLS_PATH``
(below) at it, or pass its path explicitly to ``load_iso_bestest_climate_df``,
to get the harness closer to ISO 52016-1's own reference values than the
Denver TMY3 EPW approximation gets (see ashrae140_bestest/weather/). This
requires the optional ``xlrd`` package (``pip install xlrd``) to read the
legacy .xls format.

Using this file instead of the TMY3 approximation took Case 900's heating
deviation from +24.1% to +5.2% against ISO 52016-1's own (corrected, see
iso_reference_results.py) reference value -- strong confirmation that the
TMY3 approximation, not a modelling defect, was the dominant source of
that gap, concentrated specifically in the shoulder-season months
(April/November) where heating/cooling need is most sensitive to the
exact day-to-day weather.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Resolved relative to the repo root so it picks up a local copy placed in
# Standards/ (as this repository's own maintainer's checkout has) without
# hardcoding anyone's absolute machine path; stays unresolved (and this
# loader then raises FileNotFoundError) for anyone who clones the repo
# without that file, which is expected -- see the module docstring.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ISO_CLIMATE_XLS_PATH = PROJECT_ROOT / "Standards" / "ISO_52016_1_BESTEST_ClimData_2016.08.24.xls"

_SHEET_NAME = "ISO_FDIS_52016_1_BESTEST_ClimDa"
_DATA_START_ROW = 5  # 0-indexed; rows 0-4 are title/unit/orientation headers
_N_ROWS = 9504  # 744h init month (copy of December) + 8760h real year
_COL = {
    "T2m": 5,       # theta_e;air, degC
    "NV": 6, "EV": 7, "SV": 8, "WV": 9,  # I_sol, W/m2, already ISO 52010-1 converted
    # columns 10-12 are N45/S45/VOID, not used by Case 600/900 (no diagonal surfaces)
    "HOR": 13,
    "v_w": 14,      # wind speed, m/s
}


def load_iso_bestest_climate_df(xls_path=None) -> pd.DataFrame:
    """Return a 13-month (Dec-prepended) hourly DataFrame shaped like
    pybuildingenergy's own ``Weather_data_bui``/``Calculation_ISO_52010``
    output, built directly from ISO 52016-1's BESTEST climate spreadsheet.

    Raises FileNotFoundError with a clear message if the file is not
    available, and ImportError if ``xlrd`` is not installed.
    """
    path = Path(xls_path or ISO_CLIMATE_XLS_PATH or "")
    if not path.is_file():
        raise FileNotFoundError(
            "ISO 52016-1 BESTEST climate spreadsheet not found at "
            f"{path!r}. This is ISO's own copyrighted standard content, not "
            "shipped in this repository -- set ISO_CLIMATE_XLS_PATH in this "
            "module, or pass xls_path explicitly, to your own copy of "
            "ISO_52016_1_BESTEST_ClimData_2016.08.24.xls (or equivalent)."
        )

    import xlrd  # optional dependency, only needed for this loader

    wb = xlrd.open_workbook(str(path))
    sh = wb.sheet_by_name(_SHEET_NAME)
    rows = [
        [sh.cell_value(r, c) for c in range(16)]
        for r in range(_DATA_START_ROW, _DATA_START_ROW + _N_ROWS)
    ]
    arr = np.array(rows, dtype=float)

    index = pd.date_range("2008-12-01 00:00:00", periods=_N_ROWS, freq="h")
    df = pd.DataFrame(index=index)
    df["T2m"] = arr[:, _COL["T2m"]]
    df["WS10m"] = arr[:, _COL["v_w"]]
    # Not provided by this spreadsheet (absolute humidity ratio only, not
    # RH%); latent energy is tracked separately from the Q_H/Q_C this
    # harness compares, see ashrae140_bestest/README.md.
    df["RH"] = 50.0
    df["day of year"] = df.index.dayofyear
    df["hour of day"] = df.index.hour
    df["WD10m"] = 0.0
    df["SP"] = 101325.0

    for ori in ("NV", "EV", "SV", "WV", "HOR"):
        total = arr[:, _COL[ori]]
        df[f"I_sol_tot_{ori}"] = total
        # Case 600/900 have no shading (shading=False -> F_sh=1), and
        # _solar_irradiance_after_method1_shading computes
        # (i_dif + i_dir) * F_sh, so any split summing to `total` is exact
        # when F_sh=1; this spreadsheet only gives the combined total.
        df[f"I_sol_dif_{ori}"] = 0.0
        df[f"I_sol_dir_w_{ori}"] = total
    df["G(h)"] = df["I_sol_tot_HOR"]
    df["Gb(n)"] = 0.0
    df["Gd(h)"] = df["I_sol_tot_HOR"]

    return df


@contextmanager
def iso_bestest_climate(xls_path=None):
    """Context manager: inside the ``with`` block, any
    ``ISO52016.Temperature_and_Energy_needs_calculation(...)`` call (the
    ``path_weather_file``/``weather_source`` arguments are ignored while
    active, but must still be passed since they are required parameters)
    uses ISO 52016-1's own BESTEST climate data instead of an EPW file.

    Raises the same FileNotFoundError/ImportError as
    ``load_iso_bestest_climate_df`` if the spreadsheet or ``xlrd`` is
    unavailable -- callers that want a graceful TMY3 fallback should catch
    those around their own call to this context manager.
    """
    from pybuildingenergy.source.utils import ISO52016

    df = load_iso_bestest_climate_df(xls_path)

    class _FakeWeatherBui:
        simulation_df = df

    class _FakeCalcIso52010:
        sim_df = df

    with patch.object(ISO52016, "Weather_data_bui", return_value=_FakeWeatherBui()), \
         patch("pybuildingenergy.source.utils.Calculation_ISO_52010", return_value=_FakeCalcIso52010()):
        yield df
