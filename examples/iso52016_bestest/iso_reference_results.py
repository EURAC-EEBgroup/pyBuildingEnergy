"""Reference results of EN ISO 52016-1:2017's verification cases (clause
7.2.4, Tables 28-34) for the six tests of Table 27: 600, 640, 900, 940,
600FF and 900FF.

The values are NOT distributed with this repository: they are ISO copyright
content and sharing the tables requires permission. Anyone with the standard
can enter them in a local JSON file (format and table of origin of every
field: ``iso_reference_tables.template.json``), by default

    Standards/local_reference_data/iso52016_1_2017_tables_28_34.json

(``Standards/`` is git-ignored), or point ``ISO52016_REFERENCE_FILE`` at it.
Without the file the cases still run and report results, but no deviation
from the reference.

Where the values are in the standard (EN ISO 52016-1:2017, clause 7.2.4):

* Table 28 - monthly and annual sensible energy needs for heating;
* Table 29 - monthly and annual sensible energy needs for cooling;
* Table 30 - monthly and annual mean operative temperature (all six tests);
* Table 31 - annual hourly peak heating and cooling loads;
* Table 32 - annual maximum, minimum and mean operative temperature (FF);
* Table 33 - hourly heating (+) / cooling (-) loads on January 4;
* Table 34 - hourly operative temperature on January 4 (FF).

Errata for Case 900 in Tables 28 and 29: the printed Case 900 heating and
cooling columns are swapped (the "heating" column has the summer profile of
a cooling need and the real heating column is missing). The local file
should hold the corrected sequences: cooling = the column printed as heating;
heating = the corrected sequence in the EPB Center's public "Collection of
comments and suggestions on EN ISO 52016-1:2017" (Dick van Dijk, 2022-08-15,
comment #28): https://epb.center/support/documents/iso_52016-1_details/ -- an
informal errata collection, not a formal ISO corrigendum.
"""

import json
import os
from pathlib import Path

CASES = ("600", "640", "900", "940", "600FF", "900FF")

REFERENCE_FILE = Path(os.environ.get(
    "ISO52016_REFERENCE_FILE",
    Path(__file__).resolve().parent.parent.parent
    / "Standards" / "local_reference_data" / "iso52016_1_2017_tables_28_34.json",
))

_FIELDS = (
    "MONTHLY_HEATING_KWH",                       # Table 28
    "MONTHLY_COOLING_KWH",                       # Table 29
    "ANNUAL_HEATING_KWH_ALL",                    # Table 28, annual row
    "ANNUAL_COOLING_KWH_ALL",                    # Table 29, annual row
    "MONTHLY_OPERATIVE_TEMPERATURE_C",           # Table 30
    "ANNUAL_OPERATIVE_TEMPERATURE_C",            # Table 30, annual row
    "PEAK_LOAD_KW",                              # Table 31 (heating, cooling)
    "FREE_FLOAT_OPERATIVE_TEMPERATURE_C",        # Table 32 (max, min, mean)
    "JANUARY_4_HOURLY_LOAD_WH",                  # Table 33
    "JANUARY_4_HOURLY_OPERATIVE_TEMPERATURE_C",  # Table 34
)

# Where each field is printed in EN ISO 52016-1:2017.
TABLE_OF_FIELD = {
    "MONTHLY_HEATING_KWH": "Table 28", "ANNUAL_HEATING_KWH_ALL": "Table 28",
    "MONTHLY_COOLING_KWH": "Table 29", "ANNUAL_COOLING_KWH_ALL": "Table 29",
    "MONTHLY_OPERATIVE_TEMPERATURE_C": "Table 30", "ANNUAL_OPERATIVE_TEMPERATURE_C": "Table 30",
    "PEAK_LOAD_KW": "Table 31", "FREE_FLOAT_OPERATIVE_TEMPERATURE_C": "Table 32",
    "JANUARY_4_HOURLY_LOAD_WH": "Table 33", "JANUARY_4_HOURLY_OPERATIVE_TEMPERATURE_C": "Table 34",
}


def load_references(path=None) -> dict | None:
    path = Path(path or REFERENCE_FILE)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {field: data.get(field, {}) for field in _FIELDS}


_DATA = load_references()
REFERENCES_AVAILABLE = _DATA is not None

for _field in _FIELDS:
    globals()[_field] = (_DATA or {}).get(_field, {})

# Annual needs of Tests 1 and 3, keyed by integer case (kept for the tests).
ANNUAL_HEATING_KWH = {int(c): v for c, v in ANNUAL_HEATING_KWH_ALL.items() if c in ("600", "900")}  # noqa: F821
ANNUAL_COOLING_KWH = {int(c): v for c, v in ANNUAL_COOLING_KWH_ALL.items() if c in ("600", "900")}  # noqa: F821
