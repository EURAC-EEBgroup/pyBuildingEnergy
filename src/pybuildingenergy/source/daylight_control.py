"""Hourly daylight and electric lighting control at a single reference point.

The simplified calculation is an approximation, not an EN 17037 spatial assessment.
External illuminance series may instead be supplied from a detailed simulation.
"""
import math


def lighting_step(daylight_lux, occupied, target_lux, design_lux, power_w, mode="continuous", heat_fraction=1.0):
    """Return daylight, electric and total illuminance, power and zone heat gain."""
    values = [float(daylight_lux), float(target_lux), float(design_lux), float(power_w), float(heat_fraction)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("lighting values must be finite")
    daylight, target, design, power, heat = values
    if daylight < 0 or target <= 0 or design <= 0 or power < 0 or not 0 <= heat <= 1:
        raise ValueError("invalid daylight, lighting target, design illuminance, power or heat fraction")
    if mode not in {"continuous", "on_off"}:
        raise ValueError("lighting mode must be continuous or on_off")
    fraction = min(1.0, max(0.0, (target - daylight) / design)) if occupied else 0.0
    if mode == "on_off":
        fraction = float(fraction > 0)
    electric = design * fraction
    return daylight, electric, daylight + electric, power * fraction, power * fraction * heat
