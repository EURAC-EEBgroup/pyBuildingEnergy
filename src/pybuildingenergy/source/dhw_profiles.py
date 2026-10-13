"""Reusable EN 12831-3 domestic hot-water load profiles."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from importlib.resources import files
from typing import Any

import pandas as pd


DHW_DAY_TYPE_COLUMNS = ("Workday", "Weekend", "Holiday")
_PROFILE_DATA_FILE = "dhw_profiles.json"


@lru_cache(maxsize=1)
def _load_profile_data() -> dict[str, Any]:
    resource = files("pybuildingenergy.data").joinpath(_PROFILE_DATA_FILE)
    data = json.loads(resource.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{_PROFILE_DATA_FILE} does not contain DHW profiles.")
    for name, values in profiles.items():
        if not isinstance(values, list) or len(values) != 24:
            raise ValueError(f"DHW profile {name!r} must contain 24 hourly values.")
    return data


def load_dhw_annex_b_table_b2_profiles() -> dict[str, list[float]]:
    """Return raw percentages from EN 12831-3 Annex B Table B.2."""

    return deepcopy(_load_profile_data()["profiles"])


DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT = load_dhw_annex_b_table_b2_profiles()


def dhw_annex_b_table_b2_hourly_fractions(
    profile_key: str = "single_family_dwelling",
) -> pd.DataFrame:
    """Return a normalized 24-hour profile for workdays, weekends and holidays."""

    profiles = load_dhw_annex_b_table_b2_profiles()
    if profile_key not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown EN 12831-3 Annex B Table B.2 profile {profile_key!r}. "
            f"Available profiles: {available}."
        )
    percent = pd.Series(profiles[profile_key], index=range(24), dtype=float)
    total = float(percent.sum())
    if total <= 0.0:
        raise ValueError(f"DHW profile {profile_key!r} must have a positive sum.")
    fractions = percent / total
    return pd.DataFrame(
        {column: fractions.values for column in DHW_DAY_TYPE_COLUMNS},
        index=range(24),
    )


__all__ = [
    "DHW_DAY_TYPE_COLUMNS",
    "DHW_ANNEX_B_TABLE_B2_VOLUME_PERCENT",
    "load_dhw_annex_b_table_b2_profiles",
    "dhw_annex_b_table_b2_hourly_fractions",
]
