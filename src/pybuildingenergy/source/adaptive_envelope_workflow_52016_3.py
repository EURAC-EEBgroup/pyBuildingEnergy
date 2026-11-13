"""Adaptive element versus reference element (EN ISO 52016-3, 6.10 and clause 8)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .adaptive_envelope_52016_3 import (
    adaptive_envelope_conformity_report,
    adaptive_envelope_monthly_report,
    adaptive_envelope_state_statistics,
    has_adaptive_envelope,
)


@dataclass
class AdaptiveComparison:
    """Result of :func:`compare_adaptive_with_reference`."""

    adaptive_hourly: pd.DataFrame
    reference_hourly: pd.DataFrame
    kpis: dict  # annual indicators of Table 6 (adaptive - reference)
    monthly: pd.DataFrame  # months 1..12 and "year", see adaptive_envelope_monthly_report
    statistics: dict  # use of the states per window
    report: dict  # clause 8 report (options, methods, not implemented parts)


def make_reference_building_object(building_object: dict) -> dict:
    """Copy of the building with the adaptive elements replaced by their fixed window.

    The window keeps its own ``g_value``/``u_value`` (the normal, static
    properties).  To compare with another reference element, pass a different
    ``reference_building_object`` to :func:`compare_adaptive_with_reference`.
    """
    reference = copy.deepcopy(building_object)
    for surface in reference.get("building_surface", []):
        surface.pop("adaptive_envelope", None)
    return reference


def compare_adaptive_with_reference(
    building_object: dict,
    reference_building_object: Optional[dict] = None,
    occupancy=None,
    **engine_kwargs,
) -> AdaptiveComparison:
    """Run the building with and without the adaptive elements and derive the 6.10 indicators.

    ``engine_kwargs`` go to ``ISO52016.Temperature_and_Energy_needs_calculation``
    (``weather_source``, ``path_weather_file``, ...).  ``occupancy`` (hourly,
    > 0 = occupied, same length as the result) restricts the accumulated
    low/high temperatures to the occupied hours; by default all hours count.
    """
    from .utils import ISO52016  # lazy: utils imports the adaptive module

    if not any(has_adaptive_envelope(s) for s in building_object.get("building_surface", [])):
        raise ValueError("the building has no surface with an 'adaptive_envelope'")
    reference = (
        copy.deepcopy(reference_building_object)
        if reference_building_object is not None else make_reference_building_object(building_object)
    )
    adaptive = copy.deepcopy(building_object)

    adaptive_hourly = ISO52016.Temperature_and_Energy_needs_calculation(adaptive, **engine_kwargs)[0]
    reference_hourly = ISO52016.Temperature_and_Energy_needs_calculation(reference, **engine_kwargs)[0]

    area = float(building_object["building"]["net_floor_area"])
    index = adaptive_hourly.index
    step_h = (index[1] - index[0]).total_seconds() / 3600.0 if len(index) > 1 else 1.0
    monthly = adaptive_envelope_monthly_report(
        adaptive_hourly, reference_hourly, area, occupancy=occupancy, time_step_h=step_h
    )
    return AdaptiveComparison(
        adaptive_hourly=adaptive_hourly,
        reference_hourly=reference_hourly,
        kpis={k: float(v) for k, v in monthly.loc["year"].items() if ":" not in k},
        monthly=monthly,
        statistics=adaptive_envelope_state_statistics(adaptive_hourly),
        report=adaptive_envelope_conformity_report(building_object, adaptive_hourly),
    )

