"""Hourly profiles in the multizone harness (simulate_envelope_multizone_free_floating).

Up to 2.2.0 ``HourlyProfileGenerator`` and ``get_country_code_from_latlon`` were not
imported in the multizone path: the per-zone profile block raised NameError inside a bare
``except Exception: continue`` and every zone silently ran with flat 1.0 profiles.
"""

import warnings

import numpy as np

from pybuildingenergy.source.utils import ISO52016
from tests.test_ventilation_boundary import _one_zone_building, _patched_weather


def _run(use_profiles):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _patched_weather(72, -5.0):
            out = ISO52016.simulate_envelope_multizone_free_floating(
                building_object=_one_zone_building(custom_h_ve=400.0),
                use_profiles=use_profiles,
            )
    return out, [w for w in caught if "hourly profiles could not be generated" in str(w.message)]


def test_zone_gain_profiles_follow_the_hourly_schedule():
    out, fallback = _run(use_profiles=True)
    assert not fallback, [str(w.message) for w in fallback]
    phi = out["Phi_int_zone1"].to_numpy()
    # occupancy/appliance/lighting schedules make the internal gains vary over the day
    assert np.unique(np.round(phi, 6)).size > 1


def test_flat_profiles_without_use_profiles():
    out, fallback = _run(use_profiles=False)
    assert not fallback
    assert np.unique(np.round(out["Phi_int_zone1"].to_numpy(), 6)).size == 1
