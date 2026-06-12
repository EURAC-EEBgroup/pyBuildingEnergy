"""
Tests for the ISO 52016-1 §6.5.10 affine ventilation boundary.

Unit tests for VentilationStream, VentilationBoundary, and resolve_ventilation_boundary.
Compatibility tests verify that existing ventilation_type configurations produce
unchanged H_ve values and that S_ve = H_ve * T_outdoor for all-outdoor cases.
Solver identity tests verify that H_ve * T_zone - S_ve equals the physical heat flow.
"""

import pytest
import numpy as np
import pandas as pd
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

from pybuildingenergy.source.utils import ISO52016, _make_sched_resolver
from pybuildingenergy.source.generate_profile import HourlyProfileGenerator
from pybuildingenergy.source.ventilation import (
    VentilationStream,
    VentilationBoundary,
    resolve_ventilation_boundary,
)


# ---------------------------------------------------------------------------
# VentilationStream unit tests
# ---------------------------------------------------------------------------

class TestVentilationStream:
    def test_one_outdoor_stream(self):
        s = VentilationStream(
            name="infiltration",
            heat_transfer_coefficient_w_k=50.0,
            source_temperature_c=-5.0,
        )
        assert s.heat_transfer_coefficient_w_k == 50.0
        assert s.source_temperature_c == -5.0
        assert s.category == "outdoor_air"

    def test_prescribed_supply_stream(self):
        s = VentilationStream(
            name="supply",
            heat_transfer_coefficient_w_k=800.0,
            source_temperature_c=18.0,
            category="supply",
        )
        assert s.heat_transfer_coefficient_w_k == 800.0
        assert s.source_temperature_c == 18.0
        assert s.category == "supply"

    def test_zero_conductance_stream_is_valid(self):
        s = VentilationStream(
            name="off",
            heat_transfer_coefficient_w_k=0.0,
            source_temperature_c=20.0,
        )
        assert s.heat_transfer_coefficient_w_k == 0.0

    @pytest.mark.parametrize(
        ("name", "h_k", "t_src", "match"),
        [
            ("x", -1.0, 10.0, "H_k must be >= 0"),
            ("x", float("nan"), 10.0, "H_k must be finite"),
            ("x", float("inf"), 10.0, "H_k must be finite"),
            ("x", 100.0, float("nan"), "source_temperature_c must be finite"),
            ("", 100.0, 5.0, "name must be non-empty"),
        ],
    )
    def test_stream_rejects_invalid(self, name, h_k, t_src, match):
        with pytest.raises(ValueError, match=match):
            VentilationStream(
                name=name,
                heat_transfer_coefficient_w_k=h_k,
                source_temperature_c=t_src,
            )

    def test_frozen_immutable(self):
        s = VentilationStream(
            name="x",
            heat_transfer_coefficient_w_k=10.0,
            source_temperature_c=0.0,
        )
        with pytest.raises(Exception):
            s.heat_transfer_coefficient_w_k = 20.0


# ---------------------------------------------------------------------------
# VentilationBoundary unit tests
# ---------------------------------------------------------------------------

class TestVentilationBoundary:
    def test_empty_boundary(self):
        bdy = VentilationBoundary(streams=())
        assert bdy.heat_transfer_coefficient_w_k == 0.0
        assert bdy.source_term_w == 0.0
        assert bdy.equivalent_supply_temperature_c is None
        assert bdy.sensible_heat_flow_w(20.0) == 0.0

    def test_one_stream_aggregate(self):
        s = VentilationStream("v", 100.0, 5.0)
        bdy = VentilationBoundary(streams=(s,))
        assert bdy.heat_transfer_coefficient_w_k == 100.0
        assert bdy.source_term_w == pytest.approx(500.0)
        assert bdy.equivalent_supply_temperature_c == pytest.approx(5.0)

    def test_source_warmer_than_zone(self):
        # Supply at 22 °C into zone at 18 °C -> heat into zone (Q_ve < 0)
        s = VentilationStream("supply", 200.0, 22.0)
        bdy = VentilationBoundary(streams=(s,))
        q_ve = bdy.sensible_heat_flow_w(18.0)
        assert q_ve == pytest.approx(200.0 * 18.0 - 200.0 * 22.0)  # -800 W
        assert q_ve < 0.0

    def test_source_colder_than_zone(self):
        # Outdoor at -5 °C, zone at 20 °C -> heat leaving zone (Q_ve > 0)
        s = VentilationStream("inf", 100.0, -5.0)
        bdy = VentilationBoundary(streams=(s,))
        q_ve = bdy.sensible_heat_flow_w(20.0)
        assert q_ve == pytest.approx(100.0 * (20.0 - (-5.0)))  # 2500 W
        assert q_ve > 0.0

    def test_zone_equals_outdoor_zero_heat_flow(self):
        s = VentilationStream("v", 100.0, 15.0)
        bdy = VentilationBoundary(streams=(s,))
        assert bdy.sensible_heat_flow_w(15.0) == pytest.approx(0.0)

    def test_multiple_additive_streams(self):
        s1 = VentilationStream("inf", 50.0, -5.0)
        s2 = VentilationStream("mech", 300.0, 18.0)
        bdy = VentilationBoundary(streams=(s1, s2))
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(350.0)
        expected_s_ve = 50.0 * (-5.0) + 300.0 * 18.0
        assert bdy.source_term_w == pytest.approx(expected_s_ve)
        expected_t_eq = expected_s_ve / 350.0
        assert bdy.equivalent_supply_temperature_c == pytest.approx(expected_t_eq)

    def test_heat_flow_identity(self):
        """Q_ve = H_ve * T_zone - S_ve must hold for any stream combination."""
        s1 = VentilationStream("a", 120.0, -10.0)
        s2 = VentilationStream("b", 80.0, 19.0)
        bdy = VentilationBoundary(streams=(s1, s2))
        t_zone = 21.5
        q_direct = sum(
            s.heat_transfer_coefficient_w_k * (t_zone - s.source_temperature_c)
            for s in (s1, s2)
        )
        assert bdy.sensible_heat_flow_w(t_zone) == pytest.approx(q_direct)

    def test_zero_conductance_stream_in_boundary(self):
        s_zero = VentilationStream("off", 0.0, 99.0)
        s_active = VentilationStream("on", 100.0, 5.0)
        bdy = VentilationBoundary(streams=(s_zero, s_active))
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(100.0)
        assert bdy.source_term_w == pytest.approx(500.0)

    def test_equivalent_supply_temperature(self):
        # Weighted average: (50*(-5) + 300*18) / 350 = 5150/350 ≈ 14.71 °C
        s1 = VentilationStream("inf", 50.0, -5.0)
        s2 = VentilationStream("mech", 300.0, 18.0)
        bdy = VentilationBoundary(streams=(s1, s2))
        expected = (50.0 * (-5.0) + 300.0 * 18.0) / 350.0
        assert bdy.equivalent_supply_temperature_c == pytest.approx(expected)

    def test_frozen_immutable(self):
        s = VentilationStream("x", 10.0, 0.0)
        bdy = VentilationBoundary(streams=(s,))
        with pytest.raises(Exception):
            bdy.streams = ()


# ---------------------------------------------------------------------------
# resolve_ventilation_boundary compatibility tests
# ---------------------------------------------------------------------------

class TestResolveVentilationBoundary:
    """Verify that legacy ventilation_type configurations resolve correctly."""

    def _custom_building(self, h_ve):
        """Minimal building dict with custom ventilation type."""
        return {
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "custom",
                    "custom_heat_transfer_coefficient_ventilation": h_ve,
                    "flow_rate_per_person": 0.0,
                }
            },
            "building_surface": [],
            "building": {"net_floor_area": 100.0},
        }

    def test_custom_legacy_h_ve(self):
        bld = self._custom_building(815.0)
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(815.0)

    def test_custom_legacy_s_ve_equals_h_ve_times_t_outdoor(self):
        """For outdoor-air legacy config: S_ve = H_ve * T_outdoor."""
        t_out = -5.0
        bld = self._custom_building(500.0)
        bdy = resolve_ventilation_boundary(bld, 21.0, t_out, 0.0)
        assert bdy.source_term_w == pytest.approx(bdy.heat_transfer_coefficient_w_k * t_out)

    def test_profile_multiplier_scales_h_ve(self):
        bld = self._custom_building(1000.0)
        bdy = resolve_ventilation_boundary(bld, 20.0, -5.0, 0.0, profile_multiplier=0.6)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(600.0)

    def test_none_ventilation_type_returns_empty(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "none",
                    "flow_rate_per_person": 0.0,
                    "custom_heat_transfer_coefficient_ventilation": 0.0,
                }
            },
            "building_surface": [],
            "building": {"net_floor_area": 100.0},
        }
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k == 0.0
        assert bdy.source_term_w == 0.0

    def test_components_constant_ach(self):
        """constant_ach component: H_k = rho * cp * ach * V / 3600."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "infiltration",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                            "source_temperature": "outdoor",
                        }
                    ]
                }
            }
        }
        zone_vol = 4000.0  # m3
        t_out = -10.0
        bdy = resolve_ventilation_boundary(
            bld, 21.0, t_out, 0.0, zone_volume_m3=zone_vol
        )
        rho, cp = 1.204, 1006.0
        q_m3_s = 0.015 * zone_vol / 3600.0
        expected_h = rho * cp * q_m3_s
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4)
        assert bdy.source_term_w == pytest.approx(expected_h * t_out, rel=1e-4)

    def test_constant_ach_outdoor_air_matches_legacy_custom_boundary(self):
        """Outdoor-air-only component gives the same affine boundary as legacy custom H_ve."""
        zone_vol = 486.0
        ach = 0.30
        t_zone = 21.0
        t_out = 4.0
        rho, cp = 1.204, 1006.0
        h_ve = rho * cp * ach * zone_vol / 3600.0

        component_bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [{
                        "name": "outdoor_air",
                        "ventilation_type": "constant_ach",
                        "air_changes_per_hour": ach,
                    }]
                }
            }
        }
        component_bdy = resolve_ventilation_boundary(
            component_bld,
            t_zone,
            t_out,
            0.0,
            zone_volume_m3=zone_vol,
        )
        legacy_bdy = resolve_ventilation_boundary(
            self._custom_building(h_ve),
            t_zone,
            t_out,
            0.0,
        )

        assert component_bdy.heat_transfer_coefficient_w_k == pytest.approx(
            legacy_bdy.heat_transfer_coefficient_w_k,
            rel=1e-9,
        )
        assert component_bdy.source_term_w == pytest.approx(legacy_bdy.source_term_w, rel=1e-9)
        assert component_bdy.sensible_heat_flow_w(t_zone) == pytest.approx(
            legacy_bdy.sensible_heat_flow_w(t_zone),
            rel=1e-9,
        )

    def test_components_prescribed(self):
        """Prescribed stream has arbitrary source temperature."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "ahu_supply",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 500.0,
                            "source_temperature_c": 18.0,
                        }
                    ]
                }
            }
        }
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(500.0)
        assert bdy.source_term_w == pytest.approx(500.0 * 18.0)
        assert bdy.equivalent_supply_temperature_c == pytest.approx(18.0)

    def test_prescribed_non_outdoor_changes_zone_balance(self):
        """A prescribed supply at 18 °C changes zone balance vs outdoor at -5 °C."""
        t_zone = 21.0
        h = 500.0

        bld_outdoor = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "v",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": h,
                            "source_temperature_c": -5.0,
                        }
                    ]
                }
            }
        }
        bld_supply = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "v",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": h,
                            "source_temperature_c": 18.0,
                        }
                    ]
                }
            }
        }
        q_outdoor = resolve_ventilation_boundary(
            bld_outdoor, t_zone, -5.0, 0.0
        ).sensible_heat_flow_w(t_zone)
        q_supply = resolve_ventilation_boundary(
            bld_supply, t_zone, 18.0, 0.0
        ).sensible_heat_flow_w(t_zone)
        # outdoor: 500*(21-(-5)) = 13000 W loss
        assert q_outdoor == pytest.approx(h * (t_zone - (-5.0)))
        # supply: 500*(21-18) = 1500 W loss
        assert q_supply == pytest.approx(h * (t_zone - 18.0))

    def test_duplicate_component_names_rejected(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {"name": "inf", "ventilation_type": "constant_ach",
                         "air_changes_per_hour": 0.015},
                        {"name": "inf", "ventilation_type": "constant_ach",
                         "air_changes_per_hour": 0.020},
                    ]
                }
            }
        }
        with pytest.raises(ValueError, match="Duplicate.*inf"):
            resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=1000.0)

    def test_constant_ach_without_volume_rejected(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {"name": "inf", "ventilation_type": "constant_ach",
                         "air_changes_per_hour": 0.015},
                    ]
                }
            }
        }
        with pytest.raises((ValueError, TypeError)):
            resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=None)

    def test_extra_streams_added(self):
        """extra_streams parameter adds streams to the resolved boundary."""
        bld = self._custom_building(100.0)
        purge = VentilationStream("purge", 50.0, -5.0)
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, extra_streams=(purge,))
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(150.0)

    def test_q_ve_closes_numerically(self):
        """Q_ve = H_ve * T_zone - S_ve must hold for a custom H_ve boundary."""
        t_zone = 20.5
        t_out = -3.7
        bld = self._custom_building(750.0)
        bdy = resolve_ventilation_boundary(bld, t_zone, t_out, 0.0)
        h_ve = bdy.heat_transfer_coefficient_w_k
        s_ve = bdy.source_term_w
        q_ve_formula = h_ve * t_zone - s_ve
        q_ve_method = bdy.sensible_heat_flow_w(t_zone)
        assert q_ve_formula == pytest.approx(q_ve_method)

    def test_occupancy_ventilation_type_returns_nonneg_h(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "occupancy",
                    "flow_rate_per_person": 1.4,
                    "custom_heat_transfer_coefficient_ventilation": 0.0,
                }
            },
            "building_surface": [],
            "building": {"net_floor_area": 200.0},
        }
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k >= 0.0

    def test_legacy_s_ve_uses_outdoor_not_supply(self):
        """Legacy path always uses outdoor temperature as source, not a fixed supply."""
        t_out_1 = -10.0
        t_out_2 = 10.0
        bld = self._custom_building(400.0)
        bdy1 = resolve_ventilation_boundary(bld, 20.0, t_out_1, 0.0)
        bdy2 = resolve_ventilation_boundary(bld, 20.0, t_out_2, 0.0)
        assert bdy1.source_term_w == pytest.approx(400.0 * t_out_1)
        assert bdy2.source_term_w == pytest.approx(400.0 * t_out_2)

    # --- Regression tests for fixed blockers ---

    def test_component_profile_multiplier_does_not_affect_components(self):
        """profile_multiplier only scales the legacy single stream, not components.

        Components use a default_multiplier of 1.0 so infiltration can remain
        active when the global mechanical schedule switches off.  To turn off a
        component, use component_multipliers.
        """
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "ahu",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 500.0,
                            "source_temperature_c": 18.0,
                        }
                    ]
                }
            }
        }
        # profile_multiplier has NO effect on components
        bdy_prof_off = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=0.0)
        bdy_prof_on = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=1.0)
        assert bdy_prof_off.heat_transfer_coefficient_w_k == pytest.approx(500.0)
        assert bdy_prof_on.heat_transfer_coefficient_w_k == pytest.approx(500.0)

        # component_multipliers is the correct way to turn off individual components
        bdy_off = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0, component_multipliers={"ahu": 0.0}
        )
        bdy_half = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0, component_multipliers={"ahu": 0.5}
        )
        assert bdy_off.heat_transfer_coefficient_w_k == pytest.approx(0.0)
        assert bdy_half.heat_transfer_coefficient_w_k == pytest.approx(250.0)

    def test_infiltration_stays_on_when_ahu_is_off(self):
        """Infiltration (no profile key) is always 1.0; AHU off via component_multipliers."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "infiltration",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                        },
                        {
                            "name": "ahu",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 400.0,
                            "source_temperature_c": 18.0,
                        },
                    ]
                }
            }
        }
        vol = 4000.0
        rho, cp = 1.204, 1006.0
        h_inf = rho * cp * 0.015 * vol / 3600.0

        # AHU turned off via component_multipliers; infiltration has no profile
        # so it defaults to 1.0 regardless of profile_multiplier
        bdy = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            profile_multiplier=0.0,
            component_multipliers={"ahu": 0.0},
            zone_volume_m3=vol,
        )
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(h_inf, rel=1e-3)

        # Without any component_multipliers: both streams active at 1.0
        bdy_both = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            profile_multiplier=0.0,  # no effect on components
            zone_volume_m3=vol,
        )
        assert bdy_both.heat_transfer_coefficient_w_k == pytest.approx(h_inf + 400.0, rel=1e-3)

    def test_global_components_fallback(self):
        """building_parameters.ventilation.components is used when zone has no components."""
        # Components defined at global level, zone has no ventilation sub-key
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "inf",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                        }
                    ]
                }
            }
        }
        vol = 5000.0
        rho, cp = 1.204, 1006.0
        expected_h = rho * cp * (0.015 * vol / 3600.0)

        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=vol)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4)

    def test_zone_components_override_global(self):
        """Zone-level ventilation.components takes precedence over global components.

        In the multizone resolver, the zone proxy is built so that zone-level
        components replace global ones.  This unit test uses the resolver directly
        with a zone-proxy-shaped dict (building_parameters.ventilation.components).
        """
        # Simulate the zone proxy after zone-local components win: the proxy's
        # building_parameters.ventilation.components comes from the zone dict,
        # not the global dict.
        zone_proxy = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "zone_inf",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 50.0,
                            "source_temperature_c": -5.0,
                        }
                    ]
                }
            }
        }
        bdy = resolve_ventilation_boundary(zone_proxy, 21.0, -5.0, 0.0)
        names = [s.name for s in bdy.streams]
        assert "zone_inf" in names
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(50.0)

    def test_zone_volume_not_shared_across_zones(self):
        """constant_ach with explicit zone_volume_m3 uses that volume, not building total."""
        rho, cp = 1.204, 1006.0
        ach = 0.015

        def _bld_with_ach():
            return {
                "building_parameters": {
                    "ventilation": {
                        "components": [
                            {"name": "inf", "ventilation_type": "constant_ach",
                             "air_changes_per_hour": ach}
                        ]
                    }
                }
            }

        vol_small = 500.0
        vol_large = 8000.0
        h_small = rho * cp * ach * vol_small / 3600.0
        h_large = rho * cp * ach * vol_large / 3600.0

        bdy_small = resolve_ventilation_boundary(_bld_with_ach(), 21.0, -5.0, 0.0, zone_volume_m3=vol_small)
        bdy_large = resolve_ventilation_boundary(_bld_with_ach(), 21.0, -5.0, 0.0, zone_volume_m3=vol_large)

        assert bdy_small.heat_transfer_coefficient_w_k == pytest.approx(h_small, rel=1e-4)
        assert bdy_large.heat_transfer_coefficient_w_k == pytest.approx(h_large, rel=1e-4)
        # Must differ — if both returned h_large the proxy was leaking global volume
        assert abs(bdy_small.heat_transfer_coefficient_w_k - bdy_large.heat_transfer_coefficient_w_k) > 1.0

    def test_boundary_streams_coerced_to_tuple(self):
        """Passing a list to VentilationBoundary must be silently coerced to tuple."""
        s = VentilationStream("v", 100.0, 5.0)
        bdy = VentilationBoundary(streams=[s])  # list, not tuple
        assert isinstance(bdy.streams, tuple)
        # Subsequent mutation of the original list must not affect the boundary
        original_list = [s]
        bdy2 = VentilationBoundary(streams=original_list)
        original_list.append(VentilationStream("extra", 50.0, 0.0))
        assert len(bdy2.streams) == 1

    def test_boundary_duplicate_stream_names_rejected(self):
        """VentilationBoundary must reject duplicate stream names."""
        s1 = VentilationStream("dup", 100.0, 5.0)
        s2 = VentilationStream("dup", 200.0, 10.0)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            VentilationBoundary(streams=(s1, s2))

    def test_boundary_non_stream_element_rejected(self):
        """VentilationBoundary must reject non-VentilationStream elements."""
        with pytest.raises(TypeError):
            VentilationBoundary(streams=(42,))

    def test_constant_ach_with_zone_volume_m3(self):
        """constant_ach component resolves correctly when zone_volume_m3 is supplied."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "inf",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                        }
                    ]
                }
            }
        }
        vol = 6960.0  # m3
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=vol)
        rho, cp = 1.204, 1006.0
        expected_h = rho * cp * (0.015 * vol / 3600.0)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4)


# ---------------------------------------------------------------------------
# _make_sched_resolver unit tests
# ---------------------------------------------------------------------------

class TestSchedResolver:
    """Direct unit tests for _make_sched_resolver.

    These tests prove the resolver's three required properties without running
    the full solver:
      1. An absent key returns the iso16798_profiles attribute default.
      2. An explicit None returns the default (not None itself).
      3. A falsey-but-valid value such as {} is returned unchanged (not replaced
         by the default as the old ``v or default`` pattern would have done).
      4. A numpy array is returned unchanged (``v or default`` raises ValueError
         on arrays with more than one element due to ambiguous truth value).
    """

    def _resolver(self, kwargs, profile_attr_value="default_value"):
        """Build a resolver backed by a mock iso16798_profiles object."""
        mock_profiles = MagicMock()
        mock_profiles.occupants_schedule_workdays = profile_attr_value
        return _make_sched_resolver(kwargs, mock_profiles)

    def test_absent_key_returns_attribute_default(self):
        sched = self._resolver({}, profile_attr_value="from_profiles")
        assert sched("occupants_schedule_workdays", "occupants_schedule_workdays") == "from_profiles"

    def test_explicit_none_returns_attribute_default(self):
        sched = self._resolver({"occupants_schedule_workdays": None}, profile_attr_value="from_profiles")
        assert sched("occupants_schedule_workdays", "occupants_schedule_workdays") == "from_profiles"

    def test_empty_dict_returned_unchanged(self):
        """An empty dict is falsey; the old ``v or default`` would silently replace it."""
        sched = self._resolver({"occupants_schedule_workdays": {}}, profile_attr_value="from_profiles")
        result = sched("occupants_schedule_workdays", "occupants_schedule_workdays")
        assert result == {}, f"Empty dict must be returned as-is, got {result!r}"

    def test_numpy_array_returned_unchanged(self):
        """``v or default`` raises ValueError for multi-element arrays; is-None check does not."""
        arr = np.array([0.5] * 24)
        sched = self._resolver({"occupants_schedule_workdays": arr}, profile_attr_value="from_profiles")
        result = sched("occupants_schedule_workdays", "occupants_schedule_workdays")
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------


def _minimal_sim_df(n=8760, t_out=-5.0):
    """Return a minimal simulation DataFrame for multizone and legacy solvers.

    Includes zero irradiance for all standard orientations so the legacy
    solver's solar-gain calculation does not raise KeyError.
    """
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame(index=idx)
    df["T2m"] = t_out
    df["WS10m"] = 0.0
    # Solar columns required by the legacy single-zone path
    for _ori in ("HOR", "NV", "EV", "SV", "WV"):
        df[f"I_sol_dif_{_ori}"] = 0.0
        df[f"I_sol_dir_w_{_ori}"] = 0.0
        df[f"I_sol_tot_{_ori}"] = 0.0
    return df


def _minimal_profile_df(n, extra_columns=None):
    """Return a minimal profile DataFrame with all standard solver columns set to 1.0.

    Pass extra_columns as {name: array_like} to add custom component-schedule columns.
    All standard profiles are 1.0 (fully active) so only the injected schedule drives
    component on/off behaviour.
    """
    df = pd.DataFrame({
        "ventilation_profile": np.ones(n),
        "heating_profile": np.ones(n),
        "cooling_profile": np.ones(n),
        "occupancy_profile": np.ones(n),
        "appliances_profile": np.ones(n),
        "lighting_profile": np.ones(n),
    })
    if extra_columns:
        for col, values in extra_columns.items():
            df[col] = np.asarray(values, dtype=float)
    return df


@contextmanager
def _patched_weather(n=48, t_out=-5.0):
    """Patch ISO52016.Weather_data_bui and Calculation_ISO_52010 to return minimal sim_df.

    Replaces EPW loading in all three solver paths (multizone, legacy, causal) and in
    Temp_calculation_of_ground without modifying production code.
    """
    sim_df = _minimal_sim_df(n, t_out)
    mock_wb = MagicMock()
    mock_wb.simulation_df = sim_df
    mock_ci = MagicMock()
    mock_ci.sim_df = sim_df
    with patch.object(ISO52016, 'Weather_data_bui', return_value=mock_wb), \
         patch('pybuildingenergy.source.utils.Calculation_ISO_52010', return_value=mock_ci):
        yield sim_df


@contextmanager
def _patched_profiles(profile_df):
    """Patch HourlyProfileGenerator.generate to return a fresh copy of profile_df.

    Uses side_effect rather than return_value so each call to gen.generate() gets
    an independent copy.  This prevents cross-call mutation when the hybrid solver
    calls the profile generator multiple times across iterations.
    """
    with patch.object(HourlyProfileGenerator, 'generate', side_effect=lambda: profile_df.copy()):
        yield profile_df


def _adiabatic_surface(zone_name, area=100.0):
    """Minimal adiabatic surface: no thermal nodes (Pln=0), just provides zone membership."""
    return {
        "name": f"floor_{zone_name}",
        "type": "opaque",
        "boundary": "ADIABATIC",
        "area": area,
        "zone": zone_name,
        "ISO52016_type_string": "AD",
        "sky_view_factor": 0.0,
        "u_value": 0.0,
        "thermal_capacity": 0.0,
        "solar_absorptance": 0.0,
        "orientation": {"azimuth": 0, "tilt": 0},
    }


def _one_zone_building(
    ventilation_type="custom",
    custom_h_ve=500.0,
    components=None,
    zone_volume_m3=None,
):
    """Minimal single-zone building dict for the multizone solver."""
    zone = {
        "name": "zone1",
        "net_floor_area": 100.0,
        "heating_setpoint": 21.0,
        "heating_setback": 15.0,
        "cooling_setpoint": 26.0,
        "cooling_setback": 30.0,
    }
    if components is not None:
        zone["ventilation"] = {"components": components}
    else:
        zone["ventilation_type"] = ventilation_type
        zone["custom_heat_transfer_coefficient_ventilation"] = custom_h_ve
    if zone_volume_m3 is not None:
        zone["zone_volume_m3"] = zone_volume_m3

    return {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "construction_class": "class_i",
        },
        "building_parameters": {
            "ventilation": {
                "ventilation_type": "custom",
                "custom_heat_transfer_coefficient_ventilation": custom_h_ve,
                "flow_rate_per_person": 0.0,
            },
            "temperature_setpoints": {},
        },
        "building_surface": [_adiabatic_surface("zone1")],
        "zones": [zone],
    }


class TestSolverIntegration:
    """Verify that the affine boundary reaches the ISO52016 multizone solver."""

    def _run(self, building, n=48, t_out=-5.0, use_profiles=False):
        with _patched_weather(n, t_out):
            return ISO52016.simulate_envelope_multizone_free_floating(
                building_object=building,
                use_profiles=use_profiles,
            )

    def test_legacy_custom_h_ve_appears_in_output(self):
        """H_ve column must equal the configured custom H_ve for all timesteps."""
        bld = _one_zone_building(custom_h_ve=815.0)
        out = self._run(bld)
        assert "H_ve_zone1" in out.columns
        np.testing.assert_allclose(out["H_ve_zone1"].to_numpy(), 815.0, rtol=1e-6)

    def test_s_ve_equals_h_ve_times_t_outdoor_for_legacy(self):
        """For legacy outdoor-air config: S_ve = H_ve * T_outdoor at every step."""
        t_out = -5.0
        bld = _one_zone_building(custom_h_ve=400.0)
        out = self._run(bld, t_out=t_out)
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()
        np.testing.assert_allclose(s_ve, h_ve * t_out, rtol=1e-6)

    def test_prescribed_supply_reduces_heating_demand(self):
        """A 18 °C supply requires less heating to hold setpoint than -5 °C outdoor air.

        Both cases use H_k=400 W/K.  With T_zone=21 °C:
          outdoor:   Q_ve = 400 * (21 - (-5)) = 10400 W of heat loss → large heating demand
          supply:    Q_ve = 400 * (21 -  18 ) =  1200 W of heat loss → small heating demand
        """
        h = 400.0
        t_out = -5.0
        bld_outdoor = _one_zone_building(custom_h_ve=h)
        bld_supply = _one_zone_building(
            components=[{
                "name": "mech",
                "ventilation_type": "prescribed",
                "heat_transfer_coefficient_w_k": h,
                "source_temperature_c": 18.0,
            }]
        )
        out_outdoor = self._run(bld_outdoor, n=48, t_out=t_out, use_profiles=True)
        out_supply = self._run(bld_supply, n=48, t_out=t_out, use_profiles=True)

        # Both zones are maintained at 21 °C; heating demand must be lower for supply
        q_heat_outdoor = out_outdoor["Q_HVAC_zone1"].clip(lower=0).mean()
        q_heat_supply = out_supply["Q_HVAC_zone1"].clip(lower=0).mean()
        assert q_heat_supply < q_heat_outdoor - 1000.0, (
            f"Heating demand should be >1 kW lower with 18 °C supply than outdoor {t_out} °C; "
            f"got outdoor={q_heat_outdoor:.0f} W, supply={q_heat_supply:.0f} W"
        )

    def test_q_ve_closes_h_ve_times_tzone_minus_s_ve(self):
        """Q_ve = H_ve * T_zone - S_ve must hold at every timestep."""
        bld = _one_zone_building(custom_h_ve=600.0)
        out = self._run(bld)
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()
        t_air = out["T_air_zone1"].to_numpy()
        q_ve = out["Q_ve_zone1"].to_numpy()
        np.testing.assert_allclose(q_ve, h_ve * t_air - s_ve, atol=1e-8)

    def test_global_components_reach_solver(self):
        """Global building_parameters.ventilation.components must not be dropped."""
        bld = {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "adj_zones_present": False,
            },
            "building_parameters": {
                "ventilation": {
                    "components": [{
                        "name": "inf",
                        "ventilation_type": "prescribed",
                        "heat_transfer_coefficient_w_k": 300.0,
                        "source_temperature_c": -5.0,
                    }]
                },
                "temperature_setpoints": {},
            },
            "building_surface": [],
            "zones": [{
                "name": "zone1",
                "net_floor_area": 100.0,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
            }],
        }
        # Add a surface so the solver doesn't crash with empty surface list
        bld["building"]["construction_class"] = "class_i"
        bld["building_surface"] = [_adiabatic_surface("zone1")]
        out = self._run(bld)
        # If global components were dropped, H_ve would be 0
        np.testing.assert_allclose(out["H_ve_zone1"].to_numpy(), 300.0, rtol=1e-6)

    def test_constant_ach_uses_zone_volume_not_building_total(self):
        """Each zone uses its own volume for constant_ach, not the building total."""
        ach = 0.015
        vol_a, vol_b = 300.0, 1200.0
        rho, cp = 1.204, 1006.0
        h_a = rho * cp * ach * vol_a / 3600.0
        h_b = rho * cp * ach * vol_b / 3600.0
        def _inf():
            return [{"name": "inf", "ventilation_type": "constant_ach",
                     "air_changes_per_hour": ach}]
        bld = {
            "building": {
                "net_floor_area": 150.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "volume": vol_a + vol_b,  # global total — must not be used per-zone
            },
            "building_parameters": {"ventilation": {}, "temperature_setpoints": {}},
            "building_surface": [
                _adiabatic_surface("zone_a", area=50.0),
                _adiabatic_surface("zone_b", area=100.0),
            ],
            "zones": [
                {
                    "name": "zone_a", "net_floor_area": 50.0, "zone_volume_m3": vol_a,
                    "heating_setpoint": 21.0, "heating_setback": 15.0,
                    "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                    "ventilation": {"components": _inf()},
                },
                {
                    "name": "zone_b", "net_floor_area": 100.0, "zone_volume_m3": vol_b,
                    "heating_setpoint": 21.0, "heating_setback": 15.0,
                    "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                    "ventilation": {"components": _inf()},
                },
            ],
        }
        out = self._run(bld)
        np.testing.assert_allclose(out["H_ve_zone_a"].to_numpy(), h_a, rtol=1e-4)
        np.testing.assert_allclose(out["H_ve_zone_b"].to_numpy(), h_b, rtol=1e-4)

    def test_purge_stream_source_is_outdoor(self):
        """Summer-night purge adds an outdoor-air stream: S_ve = H_ve * T_out."""
        t_out = 15.0  # warm — zone will be warmer, purge condition met
        bld = _one_zone_building(custom_h_ve=200.0)
        bld["zones"][0]["summer_night_purge"] = {
            "enabled": True, "boost_factor": 3.0,
            "months": [6, 8], "hours": [22, 6], "delta_t_min": 0.0,
        }
        with _patched_weather(n=48, t_out=t_out):
            out = ISO52016.simulate_envelope_multizone_free_floating(building_object=bld)
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()
        # All streams (base + purge) are outdoor-air, so S_ve == H_ve * T_out
        np.testing.assert_allclose(s_ve, h_ve * t_out, rtol=1e-6)

    def test_purge_h_ve_equals_boost_factor_times_base(self):
        """When purge is active H_ve = boost_factor * H_base and S_ve = H_ve * T_out.

        Uses months=[1,12] and equal hour_start/hour_end (full-day active) so the
        purge activates on the January timestamps in _minimal_sim_df.  Tests the
        PLAN.md item 'purge case: total H_ve unchanged vs all-outdoor pre-purge
        baseline' — the affine sum must equal boost_factor * H_base, not just H_base.
        """
        h_base = 200.0
        boost = 2.0
        t_out = -5.0

        bld = _one_zone_building(custom_h_ve=h_base)
        bld["zones"][0]["summer_night_purge"] = {
            "enabled": True, "boost_factor": boost,
            "months": [1, 12],   # all year
            "hours": [0, 0],     # equal start/end → full-day active
            "delta_t_min": 0.5,
        }
        # use_profiles=False keeps ventilation_profile=1 so H_base is unscaled.
        # With profiles on, the profile multiplier would scale H_base before the
        # purge boost applies, making H_total vary with the hour-of-day schedule.
        out = self._run(bld, n=24, t_out=t_out, use_profiles=False)
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()

        # From step 2: HVAC holds zone at ~21 °C; T_zone - T_out ≈ 26 °C >> 0.5
        np.testing.assert_allclose(
            h_ve[2:], h_base * boost, rtol=1e-4,
            err_msg="Active purge must give H_ve = boost_factor * H_base",
        )
        np.testing.assert_allclose(
            s_ve[2:], h_ve[2:] * t_out, rtol=1e-4,
            err_msg="All purge streams are outdoor-air: S_ve must equal H_ve * T_out",
        )

    def test_q_hvac_analytical_conditioned_supply(self):
        """Q_HVAC difference between supply and outdoor cases equals H_ve*(T_sup-T_out).

        Two otherwise identical zones differ only in ventilation source temperature:
          - Case A: prescribed supply at T_sup (S_ve = H_ve * T_sup)
          - Case B: outdoor air at T_out     (S_ve = H_ve * T_out)

        At any given zone temperature T_zone, the HVAC backsolve gives:
          Q_A = (…) - H_ve * T_sup   (S_ve = H_ve * T_sup)
          Q_B = (…) - H_ve * T_out   (S_ve = H_ve * T_out)
          ΔQ  = Q_B - Q_A = H_ve * (T_sup - T_out)

        Surface coupling, thermal mass, and every other term appear identically in
        both cases and cancel in the difference.  The identity is exact from step 2
        onward (both zones at the same steady T_sp).

        This test would fail if S_ve were replaced by H_ve * T_out in the backsolve,
        because then Q_A = Q_B and ΔQ = 0.
        """
        h = 400.0
        t_sup = 18.0
        t_out = -5.0
        expected_delta_q = h * (t_sup - t_out)  # 9200 W

        bld_supply = _one_zone_building(
            components=[{
                "name": "mech",
                "ventilation_type": "prescribed",
                "heat_transfer_coefficient_w_k": h,
                "source_temperature_c": t_sup,
            }]
        )
        bld_outdoor = _one_zone_building(custom_h_ve=h)

        out_a = self._run(bld_supply, n=24, t_out=t_out, use_profiles=False)
        out_b = self._run(bld_outdoor, n=24, t_out=t_out, use_profiles=False)

        q_a = out_a["Q_HVAC_zone1"].to_numpy()
        q_b = out_b["Q_HVAC_zone1"].to_numpy()

        # Both zones must be in heating mode for the identity to hold.
        # If one zone is in a different mode the difference cannot equal ΔQ.
        mode_a = out_a["mode_zone1"].to_numpy()
        mode_b = out_b["mode_zone1"].to_numpy()
        assert (mode_a[2:] == "H").all(), "Supply case must be in heating mode from step 2"
        assert (mode_b[2:] == "H").all(), "Outdoor case must be in heating mode from step 2"

        # From step 2 both zones are at steady setpoint; surface/mass terms cancel.
        np.testing.assert_allclose(
            q_b[2:] - q_a[2:], expected_delta_q, rtol=1e-4,
            err_msg=(
                f"Q_HVAC difference must equal H*(T_sup-T_out)={expected_delta_q:.0f} W; "
                "a mismatch means S_ve is not entering the HVAC backsolve"
            ),
        )


# ---------------------------------------------------------------------------
# Cross-solver regression tests (hybrid builder + component schedule logic)
# ---------------------------------------------------------------------------

class TestCrossSolverRegressions:
    """Component schedules and zone volume must behave correctly in all solver paths."""

    # ------------------------------------------------------------------
    # Hybrid builder: zone volume
    # ------------------------------------------------------------------

    def _hybrid_bld(self, zone_vol=None, global_vol=2000.0):
        return {
            "building": {
                "net_floor_area": 200.0,
                "building_type_class": "Residential_apartment",
                "adj_zones_present": False,
                "construction_class": "class_i",
                "volume": global_vol,
            },
            "building_parameters": {
                "ventilation": {"ventilation_type": "custom",
                                "custom_heat_transfer_coefficient_ventilation": 0.0,
                                "flow_rate_per_person": 0.0},
                "temperature_setpoints": {},
            },
            "building_surface": [_adiabatic_surface("zone_a")],
            "zones": [{
                "name": "zone_a",
                "net_floor_area": 100.0,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                **({"zone_volume_m3": zone_vol} if zone_vol is not None else {}),
            }],
        }

    def test_hybrid_builder_copies_zone_volume(self):
        """Hybrid per-zone bui must carry zone_volume_m3 from the zone dict."""
        bui = ISO52016._build_single_zone_building_object_for_core(
            self._hybrid_bld(zone_vol=450.0), "zone_a"
        )
        stored = bui["building"].get("zone_volume_m3")
        assert stored == pytest.approx(450.0), f"Expected 450.0, got {stored}"
        assert stored != pytest.approx(2000.0), "Must not be global volume"

    def test_hybrid_builder_no_zone_volume_does_not_promote_global(self):
        """Without a zone volume key, the global building volume must not be promoted."""
        bui = ISO52016._build_single_zone_building_object_for_core(
            self._hybrid_bld(zone_vol=None, global_vol=9999.0), "zone_a"
        )
        stored = bui["building"].get("zone_volume_m3")
        assert stored is None or stored != pytest.approx(9999.0), (
            f"Global volume must not be promoted to zone; got {stored}"
        )

    def _hybrid_bld_with_profiles(self, zone_extras=None, global_gains=None, zone_gains=None):
        """Hybrid building dict with optional per-zone and global profile overrides."""
        zone = {
            "name": "zone_a",
            "net_floor_area": 100.0,
            "heating_setpoint": 21.0, "heating_setback": 15.0,
            "cooling_setpoint": 26.0, "cooling_setback": 30.0,
        }
        if zone_extras:
            zone.update(zone_extras)
        if zone_gains is not None:
            zone["internal_gains"] = zone_gains

        bp = {
            "ventilation": {"ventilation_type": "custom",
                            "custom_heat_transfer_coefficient_ventilation": 0.0,
                            "flow_rate_per_person": 0.0},
            "temperature_setpoints": {},
        }
        if global_gains is not None:
            bp["internal_gains"] = global_gains

        return {
            "building": {
                "net_floor_area": 200.0,
                "building_type_class": "Residential_apartment",
                "adj_zones_present": False,
                "construction_class": "class_i",
            },
            "building_parameters": bp,
            "building_surface": [_adiabatic_surface("zone_a")],
            "zones": [zone],
        }

    def test_hybrid_zone_internal_gains_replace_global(self):
        """Zone internal_gains must override global gains in the hybrid bui."""
        global_gains = [{"name": "occupants",
                         "weekday": [1.0] * 24, "weekend": [1.0] * 24}]
        zone_gains = [{"name": "occupants",
                       "weekday": [0.2] * 24, "weekend": [0.2] * 24}]
        bui = ISO52016._build_single_zone_building_object_for_core(
            self._hybrid_bld_with_profiles(global_gains=global_gains, zone_gains=zone_gains),
            "zone_a",
        )
        result_gains = bui["building_parameters"].get("internal_gains", [])
        occ = next((g for g in result_gains if g.get("name") == "occupants"), None)
        assert occ is not None, "internal_gains must contain 'occupants' entry"
        assert occ["weekday"][0] == pytest.approx(0.2), (
            f"Zone occupants weekday[0] must be 0.2 (zone value), got {occ['weekday'][0]}"
        )

    def test_hybrid_global_internal_gains_used_when_zone_absent(self):
        """Global internal_gains must reach the hybrid bui when the zone has none."""
        global_gains = [{"name": "occupants",
                         "weekday": [0.7] * 24, "weekend": [0.7] * 24}]
        bui = ISO52016._build_single_zone_building_object_for_core(
            self._hybrid_bld_with_profiles(global_gains=global_gains),
            "zone_a",
        )
        result_gains = bui["building_parameters"].get("internal_gains", [])
        occ = next((g for g in result_gains if g.get("name") == "occupants"), None)
        assert occ is not None, "internal_gains must contain 'occupants' entry"
        assert occ["weekday"][0] == pytest.approx(0.7), (
            f"Global occupants weekday[0] must be 0.7, got {occ['weekday'][0]}"
        )

    @pytest.mark.parametrize("profile_key", [
        "ventilation_profile", "heating_profile", "cooling_profile",
    ])
    def test_hybrid_zone_profile_replaces_global(self, profile_key):
        """Zone ventilation/heating/cooling profile must override the global value."""
        global_prof = [0.9] * 24
        zone_prof = [0.3] * 24
        bld = self._hybrid_bld_with_profiles(zone_extras={profile_key: zone_prof})
        bld["building_parameters"][profile_key] = global_prof
        bui = ISO52016._build_single_zone_building_object_for_core(bld, "zone_a")
        stored = bui["building_parameters"].get(profile_key)
        assert stored is not None, f"{profile_key} must be present after hybrid build"
        assert stored[0] == pytest.approx(0.3), (
            f"Zone {profile_key}[0] must be 0.3 (zone value), got {stored[0]}"
        )

    # ------------------------------------------------------------------
    # Component schedule logic (resolver level, path-independent)
    # ------------------------------------------------------------------

    def _bld_with_components(self, comp_list, zone_vol=300.0):
        return {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "zone_volume_m3": zone_vol,
            },
            "building_parameters": {
                "ventilation": {"components": comp_list},
                "temperature_setpoints": {},
            },
        }

    def test_infiltration_unaffected_by_ahu_profile(self):
        """Infiltration (no profile key) stays at full capacity when AHU multiplier=0."""
        bld = self._bld_with_components([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": 0.015},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": 400.0,
             "source_temperature_c": 18.0},
        ], zone_vol=300.0)
        rho, cp = 1.204, 1006.0
        h_inf = rho * cp * 0.015 * 300.0 / 3600.0

        # AHU off via component_multipliers; infiltration has no profile → 1.0
        bdy = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            component_multipliers={"ahu": 0.0},
            zone_volume_m3=300.0,
        )
        inf_h = sum(s.heat_transfer_coefficient_w_k
                    for s in bdy.streams if s.name == "infiltration")
        assert inf_h == pytest.approx(h_inf, rel=1e-4)
        ahu_h = sum(s.heat_transfer_coefficient_w_k
                    for s in bdy.streams if s.name == "ahu")
        assert ahu_h == pytest.approx(0.0)

    def test_ahu_scaled_by_component_multiplier(self):
        """AHU component is scaled by its component_multiplier (0.5 → half H_k)."""
        bld = self._bld_with_components([
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": 600.0,
             "source_temperature_c": 18.0},
        ])
        bdy_half = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0, component_multipliers={"ahu": 0.5}
        )
        assert bdy_half.heat_transfer_coefficient_w_k == pytest.approx(300.0)

    def test_unknown_profile_emits_warning_in_multizone_path(self):
        """A non-AHU unknown component profile keeps the legacy warning fallback."""
        import warnings
        bld = _one_zone_building()
        bld["zones"][0]["ventilation"] = {
            "components": [{
                "name": "mech",
                "ventilation_type": "prescribed",
                "heat_transfer_coefficient_w_k": 300.0,
                "source_temperature_c": 18.0,
                "profile": "nonexistent_column",
            }]
        }
        # Update zone proxy so the component is picked up
        bld["building_parameters"]["ventilation"] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _patched_weather(n=4):
                ISO52016.simulate_envelope_multizone_free_floating(
                    building_object=bld,
                    use_profiles=False,
                )
        assert any("nonexistent_column" in str(w.message) for w in caught), (
            "Expected a warning about the unknown profile column"
        )

    def test_missing_mechanical_supply_profile_raises_in_multizone_path(self):
        """A mechanical_supply schedule typo must fail fast unless explicitly downgraded."""
        bld = _one_zone_building()
        bld["zones"][0]["ventilation"] = {
            "components": [
                _ahu_component(profile="nonexistent_column"),
            ]
        }
        bld["building_parameters"]["ventilation"] = {}

        with pytest.raises(ValueError, match="nonexistent_column"):
            with _patched_weather(n=4):
                ISO52016.simulate_envelope_multizone_free_floating(
                    building_object=bld,
                    use_profiles=False,
                )

    def test_missing_mechanical_supply_profile_can_warn_explicitly(self):
        """missing_profile_policy='warn' documents the fallback to full-flow operation."""
        import warnings
        bld = _one_zone_building()
        bld["zones"][0]["ventilation"] = {
            "components": [
                _ahu_component(profile="nonexistent_column", missing_profile_policy="warn"),
            ]
        }
        bld["building_parameters"]["ventilation"] = {}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _patched_weather(n=4):
                ISO52016.simulate_envelope_multizone_free_floating(
                    building_object=bld,
                    use_profiles=False,
                )
        assert any("missing_profile_policy='warn'" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# True cross-solver tests: legacy, causal, and hybrid paths
# ---------------------------------------------------------------------------

def _legacy_surface():
    """Minimal opaque external surface for the legacy single-zone solver."""
    return {
        "name": "ext_wall",
        "type": "opaque",
        "area": 50.0,
        "sky_view_factor": 0.5,
        "u_value": 0.5,
        "solar_absorptance": 0.6,
        "thermal_capacity": 200000.0,
        "orientation": {"azimuth": 0, "tilt": 90},
        "name_adj_zone": None,
    }


def _legacy_building(components, zone_vol=300.0):
    """Minimal building dict for the legacy single-zone solver.

    Includes all geometry fields that Temp_calculation_of_ground requires.
    """
    return {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
            "construction_class": "class_i",
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "zone_volume_m3": zone_vol,
            # Geometry required by Temp_calculation_of_ground
            "exposed_perimeter": 40.0,
            "wall_thickness": 0.3,
            "slab_on_ground_area": 100.0,
            "height": 3.0,
            "latitude_deg": 63.4,
            "longitude_deg": 10.4,
            "latitude": 63.4,
            "longitude": 10.4,
        },
        "adjacent_zones": [],
        "building_parameters": {
            "ventilation": {"components": components},
            "temperature_setpoints": {
                "heating_setpoint": 21.0,
                "heating_setback": 15.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
            },
            "system_capacities": {
                "heating_capacity": 10000.0,
                "cooling_capacity": 10000.0,
            },
        },
        "building_surface": [_legacy_surface()],
    }


def _run_legacy(building, n=48, t_out=-5.0, profile_df=None):
    """Run the legacy single-zone solver with patched weather and optional profile.

    warmup_hours=0 is required because the minimal sim_df is short
    (n << 744); without it Tstep_first_act == Tstepn and all output is empty.

    Pass profile_df to inject a controlled profile DataFrame and test
    component-schedule behaviour deterministically.
    """
    pf = profile_df if profile_df is not None else _minimal_profile_df(n)
    with _patched_weather(n, t_out), _patched_profiles(pf):
        result = ISO52016.Temperature_and_Energy_needs_calculation(
            building,
            weather_source="epw",
            path_weather_file=None,
            warmup_hours=0,
        )
    # Returns (hourly, annual, sankey); we want hourly
    return result[0] if isinstance(result, tuple) else result


class TestLegacyCausalSolverPaths:
    """Verify the affine boundary end-to-end in the legacy and causal single-zone
    solver cores (via utils.py::_resolve_single_zone_vent_boundary)."""

    def test_legacy_prescribed_h_ve_appears_in_output(self):
        """Legacy solver must emit H_ve matching the prescribed component H_k."""
        h = 500.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": 18.0,
        }])
        out = _run_legacy(bld)
        assert "H_ve" in out.columns, "Legacy output must include H_ve column"
        np.testing.assert_allclose(out["H_ve"].to_numpy(), h, rtol=1e-5)

    def test_legacy_s_ve_equals_h_ve_times_supply_temp(self):
        """Legacy S_ve = H_ve * T_supply for a prescribed non-outdoor component."""
        t_sup = 18.0
        h = 400.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": t_sup,
        }])
        out = _run_legacy(bld)
        assert "S_ve" in out.columns, "Legacy output must include S_ve column"
        np.testing.assert_allclose(
            out["S_ve"].to_numpy(), h * t_sup, rtol=1e-5,
            err_msg="S_ve must equal H_ve * T_supply for a prescribed non-outdoor component",
        )

    def test_legacy_mixed_profile_end_to_end(self):
        """Legacy solver end-to-end: AHU schedule toggles H_ve between two states.

        Runs the actual solver with an injected profile DataFrame so that the
        standard 'ventilation_profile' column alternates 1/0/1/0 over four
        timesteps.  Infiltration has no profile key and must stay at full
        capacity in every step.

        This test would fail if the _comp_mult_leg extraction in utils.py were
        removed, because it calls the real solver rather than resolving the
        boundary directly.
        """
        rho, cp, ach, vol = 1.204, 1006.0, 0.015, 300.0
        h_inf = rho * cp * ach * vol / 3600.0
        h_ahu = 400.0
        n = 4
        ahu_sched = [1.0, 0.0, 1.0, 0.0]

        bld = _legacy_building([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": ach},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": h_ahu, "source_temperature_c": 18.0,
             "profile": "ventilation_profile"},
        ], zone_vol=vol)

        profile_df = _minimal_profile_df(n, extra_columns={"ventilation_profile": ahu_sched})
        out = _run_legacy(bld, n=n, profile_df=profile_df)

        assert "H_ve" in out.columns
        expected = np.array([h_inf + h_ahu, h_inf, h_inf + h_ahu, h_inf])
        np.testing.assert_allclose(
            out["H_ve"].to_numpy(), expected, rtol=1e-4,
            err_msg="H_ve must match ahu_schedule: infiltration+AHU when on, infiltration-only when off",
        )

    def test_legacy_missing_mechanical_supply_profile_raises(self):
        """Single-zone mechanical_supply components are strict about missing profiles."""
        bld = _legacy_building([_ahu_component(profile="nonexistent_column")])
        with pytest.raises(ValueError, match="nonexistent_column"):
            _run_legacy(bld, n=4)

    def test_legacy_missing_mechanical_supply_profile_warn_policy(self):
        """An explicit warn policy keeps the previous full-flow fallback visible."""
        bld = _legacy_building([
            _ahu_component(profile="nonexistent_column", missing_profile_policy="warn")
        ])
        with pytest.warns(UserWarning, match="missing_profile_policy='warn'"):
            out = _run_legacy(bld, n=4)
        assert "H_ve" in out.columns
        assert np.all(out["H_ve"].to_numpy() > 0.0)

    def test_legacy_q_ve_closes(self):
        """Legacy Q_ve = H_ve * T_air - S_ve at every timestep."""
        bld = _legacy_building([{
            "name": "inf",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": 300.0,
            "source_temperature_c": -5.0,
        }])
        out = _run_legacy(bld, n=48)
        assert "Q_ve" in out.columns
        assert "T_air" in out.columns, "Legacy output must expose T_air for Q_ve closure check"
        h_ve = out["H_ve"].to_numpy()
        s_ve = out["S_ve"].to_numpy()
        t_air = out["T_air"].to_numpy()
        q_ve = out["Q_ve"].to_numpy()
        np.testing.assert_allclose(q_ve, h_ve * t_air - s_ve, atol=1e-6)

    def test_hybrid_zone_volume_per_zone_in_legacy_core(self):
        """Hybrid path must use zone-local volume, not global building total."""
        ach = 0.015
        rho, cp = 1.204, 1006.0
        zone_vol = 500.0
        global_vol = 5000.0
        expected_h = rho * cp * ach * zone_vol / 3600.0
        wrong_h = rho * cp * ach * global_vol / 3600.0

        bld = {
            "building": {
                "net_floor_area": 200.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "volume": global_vol,  # global — must NOT leak into zone
            },
            "building_parameters": {
                "ventilation": {"ventilation_type": "custom",
                                "custom_heat_transfer_coefficient_ventilation": 0.0,
                                "flow_rate_per_person": 0.0},
                "temperature_setpoints": {},
            },
            "building_surface": [
                _adiabatic_surface("zone_a"),
                {
                    "name": "wall_a",
                    "type": "opaque", "area": 50.0, "sky_view_factor": 0.5,
                    "u_value": 0.5, "solar_absorptance": 0.6,
                    "thermal_capacity": 200000.0,
                    "orientation": {"azimuth": 0, "tilt": 90},
                    "name_adj_zone": None, "zone": "zone_a",
                },
            ],
            "zones": [{
                "name": "zone_a",
                "net_floor_area": 100.0,
                "zone_volume_m3": zone_vol,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                "ventilation": {"components": [
                    {"name": "inf", "ventilation_type": "constant_ach",
                     "air_changes_per_hour": ach}
                ]},
            }],
        }

        # Build the per-zone bui the hybrid adapter uses, then verify volume
        bui = ISO52016._build_single_zone_building_object_for_core(bld, "zone_a")
        # Global volume must have been removed; zone_volume_m3 must be zone_vol
        assert bui["building"].get("volume") is None, (
            "Global 'volume' must be cleared from hybrid bui['building']"
        )
        assert bui["building"].get("zone_volume_m3") == pytest.approx(zone_vol), (
            f"Expected zone_volume_m3={zone_vol}, got {bui['building'].get('zone_volume_m3')}"
        )

        # Confirm the resolver sees the right volume by running it
        from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
        vent_cfg = bui["building_parameters"]["ventilation"]
        bdy = resolve_ventilation_boundary(
            bui, 20.0, -5.0, 0.0, zone_volume_m3=bui["building"].get("zone_volume_m3")
        )
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4), (
            f"H_ve should be {expected_h:.4f} (zone vol), not {wrong_h:.4f} (global vol)"
        )

    def test_hybrid_global_zone_volume_m3_does_not_leak(self):
        """Global building.zone_volume_m3 must not survive the hybrid adapter."""
        global_zone_vol = 9999.0
        zone_vol = 300.0
        bld = {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                # Global key that must NOT propagate to zone-specific bui
                "zone_volume_m3": global_zone_vol,
            },
            "building_parameters": {
                "ventilation": {"ventilation_type": "custom",
                                "custom_heat_transfer_coefficient_ventilation": 0.0,
                                "flow_rate_per_person": 0.0},
                "temperature_setpoints": {},
            },
            "building_surface": [_adiabatic_surface("z1")],
            "zones": [{
                "name": "z1",
                "net_floor_area": 100.0,
                "zone_volume_m3": zone_vol,     # zone-local value
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
            }],
        }
        bui = ISO52016._build_single_zone_building_object_for_core(bld, "z1")
        stored = bui["building"].get("zone_volume_m3")
        assert stored == pytest.approx(zone_vol), (
            f"Zone-local zone_volume_m3={zone_vol} must win over global {global_zone_vol}; "
            f"got {stored}"
        )

    def _hybrid_zone_bld(self, components, h_ve_outdoor=0.0):
        """Minimal multizone building for the public hybrid API tests."""
        return {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "number_adj_zone": 0,
                "exposed_perimeter": 40.0,
                "wall_thickness": 0.3,
                "slab_on_ground_area": 100.0,
                "height": 3.0,
                "latitude_deg": 63.4,
                "longitude_deg": 10.4,
                "latitude": 63.4,
                "longitude": 10.4,
            },
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "custom",
                    "custom_heat_transfer_coefficient_ventilation": h_ve_outdoor,
                    "flow_rate_per_person": 0.0,
                },
                "temperature_setpoints": {
                    "heating_setpoint": 21.0, "heating_setback": 15.0,
                    "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                },
                "system_capacities": {
                    # 20 kW so the outdoor-air case (≈10 kW at steady state) is not
                    # capacity-capped and the exact affine delta can be asserted.
                    "heating_capacity": 20000.0, "cooling_capacity": 10000.0,
                },
            },
            "building_surface": [{
                "name": "ext_wall_a", "type": "opaque", "area": 50.0,
                "zone": "zone_a", "sky_view_factor": 0.5, "u_value": 0.5,
                "solar_absorptance": 0.6, "thermal_capacity": 200000.0,
                "orientation": {"azimuth": 0, "tilt": 90}, "name_adj_zone": None,
            }],
            "zones": [{
                "name": "zone_a",
                "net_floor_area": 100.0,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                "ventilation": {"components": components},
            }],
        }

    def test_hybrid_public_api_prescribed_supply_end_to_end(self):
        """Temperature_and_Energy_needs_calculation_multizone_hybrid exercises the affine boundary.

        Two runs differ only in ventilation source temperature (18 °C vs −5 °C outdoor).
        Once both zones reach setpoint (step 2 onward) and the capacity is not limiting,
        the steady-state HVAC difference is exactly:

            ΔQ = H * (T_sup - T_out) = 400 * 23 = 9200 W per timestep

        The building uses 20 kW heating capacity so the outdoor-air case (≈10 kW at
        steady state) is not capped and the algebraic identity can be verified.
        """
        h = 400.0
        t_sup = 18.0
        t_out = -5.0
        n = 24
        expected_delta = h * (t_sup - t_out)  # 9200 W

        bld_supply = self._hybrid_zone_bld([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": t_sup,
        }])
        bld_outdoor = self._hybrid_zone_bld(
            components=[{
                "name": "mech",
                "ventilation_type": "prescribed",
                "heat_transfer_coefficient_w_k": h,
                "source_temperature_c": t_out,
            }]
        )

        with _patched_weather(n, t_out=t_out):
            hourly_s, _, _ = ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(
                bld_supply, weather_source="epw", path_weather_file=None, warmup_hours=0
            )
            hourly_o, _, _ = ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(
                bld_outdoor, weather_source="epw", path_weather_file=None, warmup_hours=0
            )

        assert "Q_HC_hybrid_zone_a" in hourly_s.columns, (
            "Hybrid output must contain Q_HC_hybrid_zone_a"
        )
        q_s = hourly_s["Q_HC_hybrid_zone_a"].to_numpy()
        q_o = hourly_o["Q_HC_hybrid_zone_a"].to_numpy()

        assert (q_s[2:] > 0).all(), "Supply case must be in heating mode from step 2"
        assert (q_o[2:] > 0).all(), "Outdoor case must be in heating mode from step 2"
        # From step 2: both zones at steady setpoint; all terms other than S_ve cancel.
        np.testing.assert_allclose(
            q_o[2:] - q_s[2:], expected_delta, rtol=1e-4,
            err_msg=(
                f"ΔQ_HC must equal H*(T_sup-T_out)={expected_delta:.0f} W; "
                "a mismatch means S_ve is not reaching the hybrid HVAC backsolve"
            ),
        )


def _run_causal(building, n=48, t_out=-5.0, profile_df=None):
    """Run the causal single-zone solver with patched weather and optional profile."""
    pf = profile_df if profile_df is not None else _minimal_profile_df(n)
    with _patched_weather(n, t_out), _patched_profiles(pf):
        result = ISO52016._Temperature_and_Energy_needs_calculation_core_ahu_causal(
            building,
            weather_source="epw",
            path_weather_file=None,
            warmup_hours=0,
        )
    return result[0] if isinstance(result, tuple) else result


class TestCausalSolverPath:
    """Verify the affine boundary end-to-end in the causal solver core
    (via utils.py::_resolve_single_zone_vent_boundary)."""

    def test_causal_prescribed_h_ve_in_output(self):
        """Causal solver must emit H_ve matching the prescribed component H_k."""
        h = 500.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": 18.0,
        }])
        out = _run_causal(bld)
        assert "H_ve" in out.columns, "Causal output must include H_ve column"
        np.testing.assert_allclose(out["H_ve"].to_numpy(), h, rtol=1e-5)

    def test_causal_s_ve_non_outdoor(self):
        """Causal S_ve = H_ve * T_supply for a prescribed non-outdoor component."""
        t_sup = 18.0
        h = 300.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": t_sup,
        }])
        out = _run_causal(bld)
        assert "S_ve" in out.columns
        np.testing.assert_allclose(out["S_ve"].to_numpy(), h * t_sup, rtol=1e-5)

    def test_causal_q_ve_closes(self):
        """Causal Q_ve = H_ve * T_air - S_ve at every timestep."""
        bld = _legacy_building([{
            "name": "inf",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": 300.0,
            "source_temperature_c": -5.0,
        }])
        out = _run_causal(bld, n=48)
        assert "Q_ve" in out.columns
        assert "T_air" in out.columns
        h_ve = out["H_ve"].to_numpy()
        s_ve = out["S_ve"].to_numpy()
        t_air = out["T_air"].to_numpy()
        q_ve = out["Q_ve"].to_numpy()
        np.testing.assert_allclose(q_ve, h_ve * t_air - s_ve, atol=1e-6)

    def test_causal_mixed_profile_end_to_end(self):
        """Causal solver end-to-end: AHU schedule toggles H_ve between two states.

        Mirrors test_legacy_mixed_profile_end_to_end for the causal path.
        Would fail if the _comp_mult_leg schedule extraction were removed
        from the causal core.
        """
        rho, cp, ach, vol = 1.204, 1006.0, 0.015, 300.0
        h_inf = rho * cp * ach * vol / 3600.0
        h_ahu = 400.0
        n = 4
        ahu_sched = [1.0, 0.0, 1.0, 0.0]

        bld = _legacy_building([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": ach},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": h_ahu, "source_temperature_c": 18.0,
             "profile": "ventilation_profile"},
        ], zone_vol=vol)

        profile_df = _minimal_profile_df(n, extra_columns={"ventilation_profile": ahu_sched})
        out = _run_causal(bld, n=n, profile_df=profile_df)

        assert "H_ve" in out.columns
        expected = np.array([h_inf + h_ahu, h_inf, h_inf + h_ahu, h_inf])
        np.testing.assert_allclose(
            out["H_ve"].to_numpy(), expected, rtol=1e-4,
            err_msg="H_ve must match ahu_schedule: infiltration+AHU when on, infiltration-only when off",
        )

    def test_causal_explicit_none_schedule_uses_default(self):
        """Causal core: explicit None for schedule kwarg falls back to the built-in default.

        _make_sched_resolver uses ``is None`` rather than ``or`` so a caller that
        passes occupants_schedule_workdays=None explicitly (as the multizone hybrid
        function does) gets the iso16798_profiles default rather than crashing with
        TypeError inside generate_category_profile.
        """
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": 100.0,
            "source_temperature_c": 18.0,
        }])
        n = 4
        pf = _minimal_profile_df(n)
        with _patched_weather(n), _patched_profiles(pf):
            # occupants_schedule_workdays=None is passed explicitly, mirroring the
            # multizone hybrid caller path.
            result = ISO52016._Temperature_and_Energy_needs_calculation_core_ahu_causal(
                bld,
                weather_source="epw",
                path_weather_file=None,
                warmup_hours=0,
                occupants_schedule_workdays=None,
            )
        out = result[0] if isinstance(result, tuple) else result
        assert "H_ve" in out.columns, "Causal core must return H_ve when schedule kwarg is None"


# ---------------------------------------------------------------------------
# mechanical_supply component type (EN 16798-5-1 AHU step wired into boundary)
# ---------------------------------------------------------------------------

def _ahu_component(**overrides):
    """Minimal mechanical_supply component dict."""
    d = {
        "name": "ahu",
        "ventilation_type": "mechanical_supply",
        "supply_flow_m3_h": 3600.0,
        "sensible_heat_recovery_efficiency": 0.784,
        "supply_temperature_setpoint_c": 18.0,
    }
    d.update(overrides)
    return d


def _ahu_building(components):
    return {"building_parameters": {"ventilation": {"components": components}}}


class TestMechanicalSupplyComponent:

    def test_h_ve_equals_rho_cp_times_flow(self):
        """H_ve = rho_cp * q_m3_s for a fully operating AHU."""
        bdy = resolve_ventilation_boundary(
            _ahu_building([_ahu_component()]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
        )
        rho_cp = 1.204 * 1006.0   # independent constant, not imported from module
        expected_h = rho_cp * 3600.0 / 3600.0  # q = 1.0 m³/s
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-5)

    def test_supply_temperature_reaches_setpoint(self):
        """When HR + coil can cover the load, supply temperature equals the setpoint."""
        bdy = resolve_ventilation_boundary(
            _ahu_building([_ahu_component()]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
        )
        # equivalent_supply_temperature_c = S_ve / H_ve
        assert bdy.equivalent_supply_temperature_c == pytest.approx(18.0, abs=1e-9)

    def test_ahu_off_when_flow_fraction_zero(self):
        """An AHU with flow_fraction=0.0 contributes zero H_ve and S_ve."""
        bdy = resolve_ventilation_boundary(
            _ahu_building([_ahu_component()]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
            component_multipliers={"ahu": 0.0},
        )
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(0.0)
        assert bdy.source_term_w == pytest.approx(0.0)

    def test_infiltration_independent_of_ahu_schedule(self):
        """Infiltration (constant_ach, no profile) remains active when AHU is off."""
        components = [
            _ahu_component(),
            {
                "name": "inf",
                "ventilation_type": "constant_ach",
                "air_changes_per_hour": 0.5,
            },
        ]
        bdy_on = resolve_ventilation_boundary(
            _ahu_building(components),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
            component_multipliers={"ahu": 1.0},
            zone_volume_m3=5000.0,
        )
        bdy_off = resolve_ventilation_boundary(
            _ahu_building(components),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
            component_multipliers={"ahu": 0.0},
            zone_volume_m3=5000.0,
        )
        rho_cp = 1.204 * 1006.0
        h_inf = rho_cp * 0.5 * 5000.0 / 3600.0
        # AHU off → only infiltration
        assert bdy_off.heat_transfer_coefficient_w_k == pytest.approx(h_inf, rel=1e-4)
        # AHU on → infiltration + AHU
        assert bdy_on.heat_transfer_coefficient_w_k > h_inf

    def test_extract_flow_defaults_to_supply_flow(self):
        """Omitting extract_flow_m3_h defaults to supply_flow_m3_h (balanced)."""
        bdy_explicit = resolve_ventilation_boundary(
            _ahu_building([_ahu_component(extract_flow_m3_h=3600.0)]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
        )
        bdy_default = resolve_ventilation_boundary(
            _ahu_building([_ahu_component()]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
        )
        assert bdy_explicit.heat_transfer_coefficient_w_k == pytest.approx(
            bdy_default.heat_transfer_coefficient_w_k, rel=1e-9
        )
        assert bdy_explicit.source_term_w == pytest.approx(
            bdy_default.source_term_w, rel=1e-9
        )

    def test_bypass_active_when_hr_would_overheat(self):
        """When T_oda > T_set, bypass mixes to reach setpoint, reducing H_ve."""
        # T_oda=12, T_ext=21, T_set=18 → bypass fraction > 0
        # Effective temperature rise = 18 - 12 = 6 K (not 9 K from full HR)
        bdy = resolve_ventilation_boundary(
            _ahu_building([_ahu_component()]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=12.0,
            wind_speed_m_s=0.0,
        )
        rho_cp = 1.204 * 1006.0
        # Still full flow, so H_ve unchanged; but S_ve = H_ve * T_sup = H_ve * 18
        expected_h = rho_cp * 1.0
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-5)
        assert bdy.equivalent_supply_temperature_c == pytest.approx(18.0, abs=1e-9)

    def test_extract_temperature_affects_hr(self):
        """zone_temperature_c is passed as extract temperature — different T_ext → different supply.

        With no heating coil the supply equals the HR outlet, which depends on extract
        temperature.  An unlimited coil would hide the difference by topping up to 18 °C
        in both cases; this test uses heating_coil_max_power_w=0 to expose the physics.

        T_oda=-10, T_ext=21: T_hr = -10 + 0.784*31 = 14.304 °C  (no frost: T_eha = -3.3 > -5)
        T_oda=-10, T_ext=25: T_hr = -10 + 0.784*35 = 17.44  °C  (no frost: T_eha = -2.4 > -5)
        """
        no_coil = dict(heating_coil_max_power_w=0.0)
        bdy_21 = resolve_ventilation_boundary(
            _ahu_building([_ahu_component(**no_coil)]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-10.0,
            wind_speed_m_s=0.0,
        )
        bdy_25 = resolve_ventilation_boundary(
            _ahu_building([_ahu_component(**no_coil)]),
            zone_temperature_c=25.0,
            outdoor_temperature_c=-10.0,
            wind_speed_m_s=0.0,
        )
        # Without a coil, supply = T_hr exactly
        assert bdy_21.equivalent_supply_temperature_c == pytest.approx(-10 + 0.784 * 31, rel=1e-5)
        assert bdy_25.equivalent_supply_temperature_c == pytest.approx(-10 + 0.784 * 35, rel=1e-5)
        # Higher extract → more HR → warmer supply
        assert bdy_25.equivalent_supply_temperature_c > bdy_21.equivalent_supply_temperature_c

    def test_frost_protection_reduces_hr_at_extreme_cold(self):
        """EXHAUST_LIMIT frost protection actively reduces effective HR efficiency.

        With no heating coil the frost-limited HR outlet is directly observable as
        the delivered supply temperature.  Comparing against frost_control='none'
        confirms the efficiency reduction is real, not masked by the coil.

        T_oda=-15, T_ext=21, eta=0.784, frost_limit=-5 (default):
          T_eha_nom = 21 - 0.784*36 = -7.224 °C  < -5  → frost active
          eta_frost  = (21 - (-5)) / 36 = 26/36
          T_hr_frost = -15 + (26/36)*36 = 11.0 °C
        Without frost (mode='none'):
          T_hr       = -15 + 0.784*36  = 13.224 °C
        """
        no_coil = dict(heating_coil_max_power_w=0.0)
        bdy_frost = resolve_ventilation_boundary(
            _ahu_building([_ahu_component(**no_coil)]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-15.0,
            wind_speed_m_s=0.0,
        )
        bdy_no_frost = resolve_ventilation_boundary(
            _ahu_building([_ahu_component(**no_coil, frost_control="none")]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-15.0,
            wind_speed_m_s=0.0,
        )
        assert bdy_frost.equivalent_supply_temperature_c == pytest.approx(11.0, abs=1e-6)
        assert bdy_no_frost.equivalent_supply_temperature_c == pytest.approx(
            -15 + 0.784 * 36, rel=1e-5
        )
        assert bdy_frost.equivalent_supply_temperature_c < bdy_no_frost.equivalent_supply_temperature_c

    def test_ahu_outputs_collector_captures_step_results(self):
        """ahu_outputs_collector receives AHUStepOutputs for every mechanical_supply component."""
        coll = {}
        resolve_ventilation_boundary(
            _ahu_building([_ahu_component()]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
            ahu_outputs_collector=coll,
        )
        assert "ahu" in coll
        from pybuildingenergy.source.ventilation_16798_5_1 import AHUStepOutputs
        assert isinstance(coll["ahu"], AHUStepOutputs)
        assert coll["ahu"].actual_supply_temperature_c == pytest.approx(18.0, abs=1e-9)
        assert coll["ahu"].actual_heating_coil_power_w >= 0.0
        assert coll["ahu"].heat_recovery_power_w >= 0.0

    def test_component_multiplier_is_flow_fraction_for_en16798_fan(self):
        """Component schedules reduce flow, not timestep availability."""
        component = _ahu_component(
            supply_fan_specific_power_w_per_m3_s=500.0,
            extract_fan_specific_power_w_per_m3_s=0.0,
            fan_performance_model="en16798_5_1",
        )
        coll_full = {}
        bdy_full = resolve_ventilation_boundary(
            _ahu_building([component]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
            component_multipliers={"ahu": 1.0},
            ahu_outputs_collector=coll_full,
        )
        coll_part = {}
        bdy_part = resolve_ventilation_boundary(
            _ahu_building([component]),
            zone_temperature_c=21.0,
            outdoor_temperature_c=-5.0,
            wind_speed_m_s=0.0,
            component_multipliers={"ahu": 0.25},
            ahu_outputs_collector=coll_part,
        )

        assert bdy_part.heat_transfer_coefficient_w_k == pytest.approx(
            0.25 * bdy_full.heat_transfer_coefficient_w_k,
            rel=1e-9,
        )
        assert coll_part["ahu"].fan_electric_power_w == pytest.approx(
            coll_full["ahu"].fan_electric_power_w * 0.25 ** 2.5,
            rel=1e-9,
        )

    def test_ahu_outputs_collector_empty_when_no_mechanical_supply(self):
        """No mechanical_supply components → collector is untouched."""
        coll = {}
        bld = {
            "building_parameters": {"ventilation": {"components": [
                {"name": "inf", "ventilation_type": "constant_ach", "air_changes_per_hour": 0.5},
            ]}}
        }
        resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=1000.0,
                                     ahu_outputs_collector=coll)
        assert coll == {}

    def test_unknown_type_still_raises(self):
        """Unknown ventilation_type still raises ValueError after adding mechanical_supply."""
        bld = _ahu_building([{
            "name": "x",
            "ventilation_type": "magic_box",
        }])
        with pytest.raises(ValueError, match="unknown ventilation_type"):
            resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)


# ---------------------------------------------------------------------------
# _ahu_coll_to_columns integration: column contract, types, and edge cases
# ---------------------------------------------------------------------------

from pybuildingenergy.source.ventilation_16798_5_1 import AHUStepOutputs, calculate_sensible_ahu_step
from pybuildingenergy.source.utils import _ahu_coll_to_columns, _AHU_DIAG_SPEC


def _make_step_outputs(**overrides) -> AHUStepOutputs:
    defaults = dict(
        requested_supply_temperature_c=18.5,
        actual_supply_temperature_c=18.5,
        actual_supply_flow_m3_h=1000.0,
        actual_extract_flow_m3_h=1000.0,
        required_heating_coil_power_w=500.0,
        actual_heating_coil_power_w=500.0,
        required_cooling_coil_power_w=0.0,
        actual_cooling_coil_power_w=0.0,
        fan_electric_power_w=200.0,
        heat_recovery_power_w=3000.0,
        bypass_fraction=0.0,
        frost_protection_required=False,
    )
    defaults.update(overrides)
    return AHUStepOutputs(**defaults)


class TestAhuCollToColumns:
    """_ahu_coll_to_columns — column contract, types, warm-up slicing, off-hours."""

    def test_expected_column_names_single_component(self):
        out = _make_step_outputs()
        cols = _ahu_coll_to_columns([{"ahu": out}])
        expected = {
            "Q_ahu_coil_ahu", "Q_ahu_coil_req_ahu",
            "Q_ahu_cool_ahu", "Q_ahu_cool_req_ahu",
            "Q_ahu_hr_ahu", "P_ahu_fan_ahu",
            "T_ahu_sup_ahu", "T_ahu_sup_req_ahu",
            "q_sup_m3h_ahu", "q_ext_m3h_ahu",
            "ahu_bypass_ahu", "ahu_frost_ahu",
        }
        assert set(cols.keys()) == expected

    def test_frost_is_integer_not_float(self):
        out = _make_step_outputs(frost_protection_required=True)
        cols = _ahu_coll_to_columns([{"ahu": out}])
        assert cols["ahu_frost_ahu"].dtype == np.dtype("int64")
        assert cols["ahu_frost_ahu"][0] == 1

    def test_frost_false_gives_zero(self):
        out = _make_step_outputs(frost_protection_required=False)
        cols = _ahu_coll_to_columns([{"ahu": out}])
        assert cols["ahu_frost_ahu"][0] == 0

    def test_requested_temp_none_when_ahu_off_is_nan(self):
        out_off = _make_step_outputs(requested_supply_temperature_c=None)
        cols = _ahu_coll_to_columns([{"ahu": out_off}])
        assert np.isnan(cols["T_ahu_sup_req_ahu"][0])

    def test_values_match_step_outputs_fields(self):
        out = _make_step_outputs()
        cols = _ahu_coll_to_columns([{"ahu": out}])
        assert cols["Q_ahu_coil_ahu"][0]    == pytest.approx(500.0)
        assert cols["Q_ahu_hr_ahu"][0]      == pytest.approx(3000.0)
        assert cols["P_ahu_fan_ahu"][0]     == pytest.approx(200.0)
        assert cols["T_ahu_sup_ahu"][0]     == pytest.approx(18.5)
        assert cols["T_ahu_sup_req_ahu"][0] == pytest.approx(18.5)
        assert cols["q_sup_m3h_ahu"][0]     == pytest.approx(1000.0)
        assert cols["q_ext_m3h_ahu"][0]     == pytest.approx(1000.0)
        assert cols["ahu_bypass_ahu"][0]    == pytest.approx(0.0)

    def test_multiple_components_produce_separate_suffixes(self):
        out1 = _make_step_outputs(actual_supply_temperature_c=19.0)
        out2 = _make_step_outputs(actual_supply_temperature_c=17.0)
        cols = _ahu_coll_to_columns([{"ahu1": out1, "ahu2": out2}])
        assert "T_ahu_sup_ahu1" in cols
        assert "T_ahu_sup_ahu2" in cols
        assert cols["T_ahu_sup_ahu1"][0] == pytest.approx(19.0)
        assert cols["T_ahu_sup_ahu2"][0] == pytest.approx(17.0)

    def test_warmup_slice_alignment_matches_timestep_count(self):
        """When a warm-up period is sliced away, column arrays must match the slice length."""
        n_warmup = 744   # December warm-up (31 days)
        n_year   = 8760
        on = _make_step_outputs()
        full_coll = [{"ahu": on}] * (n_warmup + n_year)
        act_slice = slice(n_warmup, None)
        cols = _ahu_coll_to_columns(full_coll[act_slice])
        assert len(cols["Q_ahu_coil_ahu"]) == n_year

    def test_missing_component_in_some_timesteps_gives_nan(self):
        """Component absent from a timestep dict should yield NaN for that hour."""
        on  = _make_step_outputs(actual_heating_coil_power_w=999.0)
        off = {}   # AHU off — component not in dict
        cols = _ahu_coll_to_columns([{"ahu": on}, off, {"ahu": on}])
        assert cols["Q_ahu_coil_ahu"][0] == pytest.approx(999.0)
        assert np.isnan(cols["Q_ahu_coil_ahu"][1])
        assert cols["Q_ahu_coil_ahu"][2] == pytest.approx(999.0)

    def test_empty_collector_returns_empty_dict(self):
        assert _ahu_coll_to_columns([]) == {}
        assert _ahu_coll_to_columns([{}, {}]) == {}


# ---------------------------------------------------------------------------
# Helpers for multizone-solver AHU tests
# ---------------------------------------------------------------------------

def _outdoor_wall_surface(zone_name, area=50.0):
    """Minimal outdoor opaque wall that creates creates thermal mass so T_air and T_op diverge."""
    return {
        "name": f"south_wall_{zone_name}",
        "type": "opaque",
        "boundary": "OUTDOORS",
        "area": area,
        "zone": zone_name,
        "sky_view_factor": 0.5,
        "u_value": 0.3,
        "thermal_capacity": 80000.0,
        "solar_absorptance": 0.6,
        "orientation": {"azimuth": 180, "tilt": 90},
    }


def _ahu_zone_building(components):
    """One-zone building with component-list ventilation for the multizone solver."""
    return {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "construction_class": "class_i",
        },
        "building_parameters": {"ventilation": {}, "temperature_setpoints": {}},
        "building_surface": [_adiabatic_surface("zone1")],
        "zones": [{
            "name": "zone1",
            "net_floor_area": 100.0,
            "heating_setpoint": 21.0,
            "heating_setback": 15.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "ventilation": {"components": components},
        }],
    }


def _ahu_zone_building_with_mass(components):
    """One-zone building with an outdoor wall, so T_air != T_op under HVAC control.

    The wall surface nodes (Pln=5) are initialized to 20 °C and lag behind the air
    node after heating begins, giving T_rad < T_air and therefore T_op < T_air.
    """
    return {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "construction_class": "class_i",
        },
        "building_parameters": {"ventilation": {}, "temperature_setpoints": {}},
        "building_surface": [
            _adiabatic_surface("zone1"),
            _outdoor_wall_surface("zone1"),
        ],
        "zones": [{
            "name": "zone1",
            "net_floor_area": 100.0,
            "heating_setpoint": 21.0,
            "heating_setback": 15.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "ventilation": {"components": components},
        }],
    }


# ---------------------------------------------------------------------------
# Mechanical-supply flow-fraction pre-validation
# ---------------------------------------------------------------------------

class TestMechanicalSupplyFlowFractionValidation:
    """component_fraction is validated before AHUStepInputs so the error names the component."""

    @pytest.mark.parametrize("bad_fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_flow_fraction_raises_with_component_name(self, bad_fraction):
        with pytest.raises(ValueError, match="ahu"):
            resolve_ventilation_boundary(
                _ahu_building([_ahu_component()]),
                zone_temperature_c=21.0,
                outdoor_temperature_c=-5.0,
                wind_speed_m_s=0.0,
                component_multipliers={"ahu": bad_fraction},
            )


# ---------------------------------------------------------------------------
# Multizone solver — AHU diagnostic output contract
# ---------------------------------------------------------------------------

class TestMultizoneAhuDiagnostics:
    """Multizone solver output must contain all 12 _AHU_DIAG_SPEC fields per component."""

    def test_all_12_diagnostic_columns_present(self):
        bld = _ahu_zone_building([_ahu_component()])
        with _patched_weather(n=4):
            out = ISO52016.simulate_envelope_multizone_free_floating(
                building_object=bld,
            )
        # Verify each _AHU_DIAG_SPEC prefix appears as a column
        for col_pfx, attr, dtype in _AHU_DIAG_SPEC:
            col = f"{col_pfx}_ahu_zone1"
            assert col in out.columns, (
                f"Expected multizone diagnostic column {col!r} missing from output. "
                f"Present AHU columns: {[c for c in out.columns if 'ahu' in c.lower()]}"
            )

    def test_diagnostic_values_are_finite_for_operating_ahu(self):
        bld = _ahu_zone_building([_ahu_component()])
        with _patched_weather(n=4, t_out=-5.0):
            out = ISO52016.simulate_envelope_multizone_free_floating(
                building_object=bld,
            )
        # Numeric fields (excluding frost int and optional requested_supply_temperature_c)
        float_cols = [
            f"{col_pfx}_ahu_zone1"
            for col_pfx, _, dtype in _AHU_DIAG_SPEC
            if dtype == float and col_pfx != "T_ahu_sup_req"
        ]
        for col in float_cols:
            assert np.all(np.isfinite(out[col].to_numpy())), (
                f"Column {col!r} has non-finite values: {out[col].to_numpy()}"
            )


# ---------------------------------------------------------------------------
# Multizone solver — extract-temperature coupling regression
# ---------------------------------------------------------------------------

class TestMultizoneAhuCouplingLag:
    """AHU receives the previous-timestep zone AIR temperature as extract temperature."""

    def test_extract_temperature_equals_previous_air_not_operative(self):
        """Verify T_extract = T_air[t-1], not T_op[t-1], with T_air != T_op.

        _ahu_zone_building_with_mass provides an outdoor wall whose surface nodes
        (Pln=5, initialized to 20 °C) lag behind the air node after HVAC heats the
        zone.  This gives T_rad < T_air and therefore T_op != T_air by > 0.5 K,
        making the air-vs-operative distinction detectable.
        """
        bld = _ahu_zone_building_with_mass([_ahu_component()])
        captured = []

        def _capture(cfg, inp):
            captured.append(inp)
            return calculate_sensible_ahu_step(cfg, inp)

        with patch(
            "pybuildingenergy.source.ventilation_16798_5_1.calculate_sensible_ahu_step",
            side_effect=_capture,
        ):
            with _patched_weather(n=4):
                out = ISO52016.simulate_envelope_multizone_free_floating(
                    building_object=bld,
                )

        assert len(captured) >= 2, "Expected at least 2 AHU calls (one per timestep)"
        t_air_step0 = float(out["T_air_zone1"].iloc[0])
        t_op_step0 = float(out["T_op_zone1"].iloc[0])

        # Fixture must produce a meaningful T_air / T_op difference
        assert abs(t_air_step0 - t_op_step0) > 0.5, (
            f"Fixture did not produce T_air != T_op: T_air={t_air_step0:.3f}, "
            f"T_op={t_op_step0:.3f}"
        )
        # Extract temperature must equal previous AIR temperature, not operative
        assert captured[1].extract_temperature_c == pytest.approx(t_air_step0, abs=1e-6)
        assert abs(captured[1].extract_temperature_c - t_op_step0) > 0.5
