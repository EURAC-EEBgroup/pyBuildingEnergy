"""Cooling generation calculation according to EN 16798-13:2017.

The module implements the compression (COMP), absorption (ABS) and other (OTHER)
cooling generators of EN 16798-13, Module M4-8, alone or as several generators
with priorities. It is intended to be used for the cooling side of reversible heat
pumps or chillers after emission, distribution and optional cooling storage have
calculated the cooling energy to be removed.

Implemented clauses:

* 6.4.2.1: condenser inlet (3a)/(3b), generation outlet temperatures (4)/(5);
* 6.4.2.2: optional free-cooling operating factor;
* 6.4.2.3: available cooling extraction (7)-(9), priorities (10)/(11), part-load
  ratio (12);
* 6.4.2.4: heat-rejection operation (13a)-(13h), (14a)-(14h);
* 6.4.3.1 to 6.4.3.4: cooling removed, rejected heat, compressor electricity using
  EER, absorption driving heat, electricity of OTHER generators (24);
* 6.4.3.5: generator auxiliary electricity, heat rejection (26a)/(26b).

Scope and assumptions:

* free cooling, ``free_cooling_model="binary"`` (default), is all-or-nothing as in
  (6a)-(6d): the factor is 1 when the required outlet temperature exceeds the
  sink temperature (dry bulb for DRY, wet bulb for WET and HYBRID, theta_sk for
  OTHER) by more than delta_theta_fc, else 0. ``"partial"`` (library extension):
  a free-cooling heat exchanger of conductance ``free_cooling_ua_kW_K`` between the
  chilled-water return and the sink supplies
  min(request, UA (theta_ret - theta_sink) t) and the chiller the rest (series
  pre-cooling); theta_ret from ``theta_C_gen_in_C`` / ``theta_C_dis_return_C`` /
  ``T_C_return_C`` or theta_C;gen;out;req + ``free_cooling_return_deltaT_K``;
* heat rejection (6.4.2.4): theta_cond;in is theta_e (AIR_C_COND), (3a) dry
  theta_e + delta_theta_hr + delta_theta_ls;dis;hr (DRY), (3b) the same on the wet
  bulb (WET), theta_sk (OTHER, column ``theta_sk_C`` or ``other_sink_temperature_C``),
  never below ``theta_cond_in_lim_C`` when given. HYBRID switches between dry and
  wet with ``hybrid_heat_rejection_control`` (HBRD_HEAT_REJ_CTRL, Table 10):
  ``TEMP`` wet when theta_e > ``theta_lim_wet_hr_C``; ``MAX_POWER`` (default) wet
  when the request after free cooling exceeds the sum of the generators' dry
  maxima. A ``T_C_source_C`` column overrides the computed condenser inlet. The
  heat-rejection electricity uses p_hr;el;dry / p_hr;el;wet by f_op;hr;wet (26b);
* ``capacity_temperature_correction=True`` applies (9a)-(9c) to the nominal-EER
  method (``delta_theta_corr_K``, 0 by default as for nominal data); off by default,
  the capacity is then the nominal one;
* absorption (``generation_type="ABS"``): capacity (9d) = Phi_n x p(zeta_c) /
  p(zeta_c;n), heat ratio (22) zeta = p(zeta_c) = C5 zeta_c^2 + C6 zeta_c + C7, Carnot
  heat ratio (23) from the evaporator outlet, condenser inlet and driving hot water
  ``theta_H_C_gen_abs_in_C`` temperatures; driving heat (21) Q_H;C;gen;abs;in =
  Q_C;gen;in / zeta, included in the rejected heat (16). The printed (9d)/(22) mix up
  the indices of C5-C8; they are read as above. Without C5-C7, zeta = zeta_n x
  zeta_c / zeta_c;n (``zeta_n`` default 0.7, single effect; nominal 7/30/90 C).
  Electricity: ``absorption_electric_ratio`` (solution pump, kW_el per kW cooling);
* OTHER generators (ground, groundwater, surface water...): (9g) capacity
  Phi_n x ``other_capacity_factor``(theta_C;gen;out;req, theta_sk) and (24)
  electricity Q x ``other_electric_ratio``(theta_C;gen;out, theta_sk, f_C;PL); each
  is a number or a Python function, as the standard only names the functions;
* ``part_load_performance_method="en16798_13"``: the method of the standard for
  compression chillers. C1-C4 and delta_theta_corr are solved from the EN 14825
  points A, B, C, D and the 5th point (system (1) of 6.3.2.2.2.1, linear in the five
  unknowns; the 5th point estimated with (2) when not given), ``en14825_points``:
  list of {f_C_PL (fraction), EER, theta_C_evap_out_C, theta_cond_in_C}; or C1-C4 and
  ``delta_theta_corr_K`` given directly. EER = EER_n f_EER, (19), with f_EER of (20a)
  and of (20b) below f_C;PL;min; the capacity follows (9a)-(9c) with the same
  delta_theta_corr. Needs ``nominal_eer`` and no performance map;
* several generators: each one has its own outlet limit (4); the system outlet
  temperature ``theta_C_gen_out_C`` is the energy-weighted mix of the running
  generators (``theta_C_gen_out_C_<name>`` per generator);
* indices: ``SEER_C_gen`` (= ``SEER_C_gen_el``) is cooling / electricity only; for
  absorption the driving heat is a separate carrier: ``zeta_C_gen_seasonal`` (cooling of
  the absorption generators / driving heat) and ``EER_C_gen_all_inputs`` (cooling /
  (electricity + driving heat));
* part load: ``part_load_performance_method`` ``simple`` (none), ``en14825``
  (EN 14825 cycling factor) or ``curve`` (factor a + b x + c x^2 with
  ``part_load_curve_coefficients``, = 1 at full load); x = max(f_C;PL, f_C;PL;min)
  as in (20b) (approximations of (20a)/(20b), see ``en16798_13`` above for the
  standard's method). It multiplies the EER (COMP) or, as a library extension, the heat
  ratio zeta of absorption chillers (the standard has no part-load correction for
  them), so the driving heat rises at part load; not used for OTHER;
* several generators (``generators``: list of generator configurations with
  ``name`` and ``priority``), (7)-(11): each step the request after free cooling is
  given to priority 1 up to its maximum (9), the rest to priority 2, and so on;
  generators with the same priority share pro rata to their maxima (extension of
  (10)/(11)); the remainder goes to the backup or is unmet. Free cooling, backup,
  heat-rejection settings and the hybrid dry/wet decision are those of the system;
  each generator inherits the heat-rejection settings unless it gives its own;
* ``minimum_part_load_ratio`` is f_C;PL;min of (20b); the ``simple`` method ignores
  it (warning);
* the optional backup (``cooling_backup_eer``) is a library extension: its cooling
  and its electricity are both counted in the rejected heat Q_C;gen;out.

Manufacturer or project-specific performance maps are preferred. If a map is
not provided, the fallback uses a nominal EER with the EN 16798-13 nominal-EER
temperature correction form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warnings

import numpy as np
import pandas as pd

from .cooling_16798_9 import _demand_series
from .heat_pump_15316_4_2 import _coerce_performance_map, _interpolate_performance, _bilinear_or_linear
from .performance_14511_14825 import en14825_part_load_factor


_KWH_EPS = 1e-12
_T0_ABS_K = 273.15


@dataclass
class CoolingGenerationSimulationResult:
    """Container returned by :class:`CoolingGenerationSystemCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


def _build_pivot_arrays(
    performance_map: "pd.DataFrame",
    column: str,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"] | None:
    """Build sorted pivot arrays for bilinear interpolation.

    Returns (sources, sinks, values) suitable for _bilinear_or_linear,
    or None if the pivot contains NaN (scattered map â€” use IDW fallback).
    Pre-building once avoids rebuilding the pivot on every timestep.
    """
    pivot = performance_map.pivot_table(
        index="source_temperature_C",
        columns="sink_temperature_C",
        values=column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        return None  # scattered map â€” fall back to per-call _interpolate_performance
    return (
        pivot.index.to_numpy(dtype=float),
        pivot.columns.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )

class CoolingGenerationSystemCalculator:
    """EN 16798-13:2017 cooling-generation calculator (COMP, ABS, OTHER)."""

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    @property
    def minimum_outlet_temperature_C(self) -> float:
        """Coldest outlet temperature reachable, (4): the lowest limit of the generators."""
        if self.generators:
            return min(g["calc"].theta_C_gen_out_limit_C for g in self.generators)
        return self.theta_C_gen_out_limit_C

    def run_timeseries(self, data: pd.DataFrame) -> CoolingGenerationSimulationResult:
        """Run the hourly cooling-generation calculation."""

        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        summary = self._summarize(results)
        return CoolingGenerationSimulationResult(
            timeseries=results,
            summary=summary,
            inputs=dict(self.input_data),
        )

    def _load_options(self) -> None:
        cfg = self.input_data
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.demand_unit = str(cfg.get("demand_unit", "kWh")).lower()
        if self.demand_unit not in {"wh", "kwh"}:
            raise ValueError("demand_unit must be 'Wh' or 'kWh'.")

        self.generation_type = str(cfg.get("GEN_TYPE", cfg.get("generation_type", "COMP"))).upper()
        if self.generation_type not in {"COMP", "ABS", "OTHER"}:
            raise ValueError("generation_type must be COMP, ABS or OTHER.")
        self.heat_rejection_type = str(
            cfg.get("HEAT_REJ_TYPE", cfg.get("heat_rejection_type", "AIR_C_COND"))
        ).upper()
        allowed_rejection = {"AIR_C_COND", "DRY", "WET", "HYBRID", "OTHER"}
        if self.heat_rejection_type not in allowed_rejection:
            raise ValueError(
                "heat_rejection_type must be AIR_C_COND, DRY, WET, HYBRID or OTHER."
            )

        self.performance_map = _coerce_performance_map(
            cfg.get("cooling_performance_map", cfg.get("performance_map")),
            performance_column="eer",
            service_name="cooling_generation",
            required=False,
        )
        self.nominal_capacity_kW = max(
            float(cfg.get("nominal_capacity_kW", cfg.get("Phi_C_gen_n_kW", 0.0))),
            0.0,
        )
        self.nominal_eer = _optional_positive_float(
            cfg.get("nominal_eer", cfg.get("EER_n"))
        )
        self.theta_C_gen_out_limit_C = float(
            cfg.get("theta_C_gen_out_limit_C", cfg.get("theta_C_gen_out_lim_C", 5.0))
        )
        self.theta_C_gen_out_set_C = float(
            cfg.get("theta_C_gen_out_set_C", cfg.get("cooling_sink_temperature_C", 7.0))
        )
        self.theta_C_evap_out_nominal_C = float(
            cfg.get("theta_C_evap_out_nominal_C", cfg.get("theta_C_evap_out_n_C", 7.0))
        )
        self.theta_cond_in_nominal_C = float(
            cfg.get("theta_cond_in_nominal_C", cfg.get(
                "theta_cond_in_n_C", 30.0 if self.generation_type == "ABS" else 35.0))
        )
        self.free_cooling_enabled = bool(
            cfg.get("free_cooling_enabled", str(cfg.get("FREE_COOL_OP", "NO")).upper() == "YES")
        )
        self.free_cooling_model = str(cfg.get("free_cooling_model", "binary")).lower()
        if self.free_cooling_model not in {"binary", "partial"}:
            raise ValueError("free_cooling_model must be 'binary' or 'partial'.")
        ua = cfg.get("free_cooling_ua_kW_K")
        self.free_cooling_ua_kW_K = None if ua is None else _positive_float(ua, "free_cooling_ua_kW_K")
        if self.free_cooling_enabled and self.free_cooling_model == "partial" and self.free_cooling_ua_kW_K is None:
            raise ValueError("free_cooling_model 'partial' needs free_cooling_ua_kW_K.")
        self.free_cooling_return_deltaT_K = _positive_float(
            cfg.get("free_cooling_return_deltaT_K", 5.0), "free_cooling_return_deltaT_K")
        self.free_cooling_deltaT_K = max(
            float(cfg.get("free_cooling_deltaT_K", cfg.get("delta_theta_fc_K", 4.0))),
            0.0,
        )
        self.heat_rejection_deltaT_K = float(
            cfg.get("heat_rejection_deltaT_K", cfg.get("delta_theta_hr_K", 0.0))
        )
        self.heat_rejection_loop_deltaT_K = float(
            cfg.get("heat_rejection_loop_deltaT_K", cfg.get("delta_theta_ls_dis_hr_K", 0.0))
        )
        self.wet_bulb_offset_K = float(cfg.get("wet_bulb_offset_K", 3.0))
        self.other_sink_temperature_C = float(cfg.get("other_sink_temperature_C", 12.0))
        lim = cfg.get("theta_cond_in_lim_C")
        self.theta_cond_in_lim_C = None if lim is None else float(lim)       # (14)
        self.hybrid_control = str(cfg.get(
            "HBRD_HEAT_REJ_CTRL", cfg.get("hybrid_heat_rejection_control", "MAX_POWER"))).upper()
        if self.hybrid_control not in {"TEMP", "MAX_POWER"}:
            raise ValueError("hybrid_heat_rejection_control must be TEMP or MAX_POWER (EN 16798-13 Table 10).")
        lim_wet = cfg.get("theta_lim_wet_hr_C")
        self.theta_lim_wet_hr_C = None if lim_wet is None else float(lim_wet)
        if (self.heat_rejection_type == "HYBRID" and self.hybrid_control == "TEMP"
                and self.theta_lim_wet_hr_C is None):
            raise ValueError("hybrid_heat_rejection_control TEMP needs theta_lim_wet_hr_C (EN 16798-13 (13d)).")
        self.minimum_part_load_ratio = max(
            float(cfg.get("minimum_part_load_ratio", cfg.get("f_C_PL_min", 0.0))),
            0.0,
        )
        self.part_load_performance_method = str(
            cfg.get("part_load_performance_method", "simple")
        ).lower()
        if self.part_load_performance_method not in {"simple", "en14825", "curve", "en16798_13"}:
            raise ValueError("part_load_performance_method must be 'simple', 'en14825', 'curve' or "
                             "'en16798_13'.")
        self.part_load_curve = None
        if self.part_load_performance_method == "curve":
            coeffs = cfg.get("part_load_curve_coefficients")
            if coeffs is None or len(coeffs) != 3:
                raise ValueError("part_load_performance_method 'curve' needs part_load_curve_coefficients "
                                 "[a, b, c] (factor = a + b x + c x^2).")
            self.part_load_curve = tuple(float(c) for c in coeffs)
            if abs(_part_load_curve(self.part_load_curve, 1.0) - 1.0) > 1e-6:
                raise ValueError("part_load_curve_coefficients must give a factor of 1 at full load (a + b + c = 1).")
        self.part_load_unit_type = str(cfg.get("part_load_unit_type", "air-to-water")).lower()
        self.part_load_degradation_coefficient = _fraction(
            cfg.get("part_load_degradation_coefficient", 0.9),
            "part_load_degradation_coefficient",
        )
        # f_C;PL;min of (20b): "minimum_part_load_ratio" / "f_C_PL_min" and
        # "part_load_minimum_capacity_ratio" name the same quantity
        self.minimum_part_load_ratio = _fraction(
            self.minimum_part_load_ratio, "minimum_part_load_ratio"
        )
        if "part_load_minimum_capacity_ratio" in cfg:
            self.part_load_minimum_capacity_ratio = _fraction(
                cfg["part_load_minimum_capacity_ratio"], "part_load_minimum_capacity_ratio"
            )
            if (self.minimum_part_load_ratio > 0.0
                    and abs(self.minimum_part_load_ratio - self.part_load_minimum_capacity_ratio) > 1e-12):
                raise ValueError(
                    "minimum_part_load_ratio and part_load_minimum_capacity_ratio are both "
                    "f_C;PL;min (EN 16798-13 (20b)): give one value."
                )
        else:
            self.part_load_minimum_capacity_ratio = self.minimum_part_load_ratio
        if self.part_load_performance_method == "simple" and self.part_load_minimum_capacity_ratio > 0.0:
            warnings.warn(
                "EN 16798-13: the minimum part load ratio has no effect with "
                "part_load_performance_method='simple' (no part-load EER correction); "
                "use 'en14825' or 'curve'.",
                RuntimeWarning,
                stacklevel=3,
            )
        self.cooling_backup_eer = _optional_positive_float(cfg.get("cooling_backup_eer"))
        self.heat_recovery_fraction = _fraction(
            cfg.get("heat_recovery_fraction", 0.0), "heat_recovery_fraction"
        )
        self.performance_includes_heat_rejection_aux = bool(
            cfg.get("performance_includes_heat_rejection_aux", True)
        )
        self.p_hr_el_dry = max(float(cfg.get("p_hr_el_dry", cfg.get("phr_el_dry", 0.0))), 0.0)
        self.p_hr_el_wet = max(float(cfg.get("p_hr_el_wet", cfg.get("phr_el_wet", 0.0))), 0.0)
        self.p_hr_el_other = max(float(cfg.get("p_hr_el_other", cfg.get("phr_el_other", 0.0))), 0.0)
        self.control_power_kW = max(float(cfg.get("control_power_kW", cfg.get("P_el_C_ctrl_kW", 0.0))), 0.0)
        self.additional_auxiliary_power_kW = max(
            float(cfg.get("additional_auxiliary_power_kW", 0.0)),
            0.0,
        )

        self.capacity_temperature_correction = bool(cfg.get("capacity_temperature_correction", False))
        self.delta_theta_corr_K = float(cfg.get("delta_theta_corr_K", 0.0))
        if self.capacity_temperature_correction and self.performance_map is not None:
            raise ValueError("capacity_temperature_correction (EN 16798-13 (9a)-(9c)) applies to the "
                             "nominal-EER method, not to a performance map.")
        self.en16798_coefficients = None
        if self.part_load_performance_method == "en16798_13":
            self._load_en16798_13_part_load(cfg)

        self._load_absorption_options(cfg)
        self._load_other_options(cfg)
        self.generators = self._load_generators(cfg.get("generators"))
        if self.generators:
            self._cap_pivot = self._eer_pivot = None
            return
        if self.performance_map is None and self.nominal_eer is None:
            self.nominal_eer = _default_nominal_eer(self.nominal_capacity_kW)
        if self.performance_map is None and self.nominal_capacity_kW <= _KWH_EPS:
            raise ValueError(
                "Provide either cooling_performance_map or nominal_capacity_kW for EN 16798-13."
            )
        # --- Commit-2A: pre-build pivot arrays so _capacity_and_eer never
        #     calls pivot_table inside the per-timestep loop.
        if self.performance_map is not None:
            self._cap_pivot = _build_pivot_arrays(self.performance_map, "capacity_kW")
            self._eer_pivot = _build_pivot_arrays(self.performance_map, "eer")
        else:
            self._cap_pivot = None
            self._eer_pivot = None

    def _load_absorption_options(self, cfg: dict[str, Any]) -> None:
        self.theta_H_C_gen_abs_in_C = float(cfg.get("theta_H_C_gen_abs_in_C", 90.0))
        self.theta_H_C_gen_abs_in_nominal_C = float(cfg.get("theta_H_C_gen_abs_in_nominal_C", 90.0))
        self.zeta_n = _positive_float(cfg.get("zeta_n", cfg.get("nominal_heat_ratio", 0.7)), "zeta_n")
        self.absorption_electric_ratio = max(float(cfg.get("absorption_electric_ratio", 0.0)), 0.0)
        coeffs = [cfg.get(k) for k in ("C5", "C6", "C7")]
        if any(c is not None for c in coeffs) and not all(c is not None for c in coeffs):
            raise ValueError("give all three absorption coefficients C5, C6, C7 (EN 16798-13 (22)).")
        self.abs_coefficients = None if coeffs[0] is None else tuple(float(c) for c in coeffs)
        if self.generation_type != "ABS":
            return
        if self.performance_map is not None:
            raise ValueError("generation_type ABS uses C5-C7 / zeta_n, not a performance map.")
        zc_n = _carnot_heat_ratio(self.theta_C_evap_out_nominal_C, self.theta_cond_in_nominal_C,
                                  self.theta_H_C_gen_abs_in_nominal_C)
        if zc_n <= 0.0:
            raise ValueError("absorption nominal temperatures need hot water > condenser > evaporator.")
        self._zeta_c_n = zc_n
        if self.abs_coefficients is None:
            self.abs_coefficients = (0.0, self.zeta_n / zc_n, 0.0)
        if _poly(self.abs_coefficients, zc_n) <= 0.0:
            raise ValueError("absorption coefficients give a non-positive heat ratio at nominal conditions.")

    def _load_en16798_13_part_load(self, cfg: dict[str, Any]) -> None:
        """6.3.2.2.2.1: C1-C4 and delta_theta_corr from EN 14825 points, (19), (20a)/(20b), (9a)-(9c)."""
        if self.generation_type != "COMP" or cfg.get("generators") is not None:
            if cfg.get("generators") is None:
                raise ValueError("part_load_performance_method 'en16798_13' applies to compression chillers.")
            return
        if self.performance_map is not None:
            raise ValueError("part_load_performance_method 'en16798_13' uses nominal_eer and EN 14825 "
                             "points, not a performance map.")
        if self.nominal_eer is None:
            raise ValueError("part_load_performance_method 'en16798_13' needs nominal_eer (EER_n, (19)).")
        points = cfg.get("en14825_points")
        direct = [cfg.get(k) for k in ("C1", "C2", "C3", "C4")]
        if points is not None:
            coeffs = _fit_en16798_13(points, self.nominal_eer)
        elif all(c is not None for c in direct):
            coeffs = tuple(float(c) for c in direct) + (float(cfg.get("delta_theta_corr_K", 0.0)),)
        else:
            raise ValueError("part_load_performance_method 'en16798_13' needs en14825_points (A, B, C, D "
                             "and optionally the 5th point) or the coefficients C1-C4.")
        self.en16798_coefficients = coeffs
        self.delta_theta_corr_K = coeffs[4]
        self.capacity_temperature_correction = True      # (9a)-(9c) with the same delta_theta_corr

    def _f_eer(self, cond_C, evap_C, x):
        """(20a): f_EER = (T0 + evap) / (cond - evap + x dtheta_corr) (C1 x^3 + C2 x^2 + C3 x + C4)."""
        c1, c2, c3, c4, dc = self.en16798_coefficients
        lift = np.maximum(np.asarray(cond_C, dtype=float) - np.asarray(evap_C, dtype=float) + x * dc, _KWH_EPS)
        return (_T0_ABS_K + np.asarray(evap_C, dtype=float)) / lift * (c1 * x ** 3 + c2 * x ** 2 + c3 * x + c4)

    def _load_other_options(self, cfg: dict[str, Any]) -> None:
        """GEN_TYPE OTHER: user functions of (9g) and (24)."""
        self.other_capacity_factor = _number_or_callable(
            cfg.get("other_capacity_factor", 1.0), "other_capacity_factor")
        raw_el = cfg.get("other_electric_ratio")
        self.other_electric_ratio = None if raw_el is None else _number_or_callable(raw_el, "other_electric_ratio")
        if self.generation_type != "OTHER" or cfg.get("generators") is not None:
            return
        if self.performance_map is not None:
            raise ValueError("generation_type OTHER uses other_capacity_factor / other_electric_ratio, "
                             "not a performance map.")
        if self.part_load_performance_method != "simple":
            raise ValueError("generation_type OTHER: the part load enters other_electric_ratio(theta_out, "
                             "theta_sk, f_PL) (EN 16798-13 (24)); use part_load_performance_method 'simple'.")
        if self.other_electric_ratio is None:
            raise ValueError("generation_type OTHER needs other_electric_ratio: a number or a function "
                             "f(theta_C_gen_out, theta_sk, f_C_PL) in kWh_el per kWh of cooling (EN 16798-13 (24)).")

    def _load_generators(self, raw) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("generators must be a non-empty list of generator configurations.")
        shared = {k: v for k, v in self.input_data.items() if k in _SHARED_GENERATOR_KEYS}
        gens, names = [], set()
        for g in raw:
            name = str(g.get("name", "")).strip()
            if not name or name in names:
                raise ValueError("each generator needs a unique non-empty 'name'.")
            names.add(name)
            priority = int(g.get("priority", 1))
            if priority < 1:
                raise ValueError(f"generator '{name}': priority must be an integer >= 1.")
            sub_cfg = {**shared, **{k: v for k, v in g.items() if k not in {"name", "priority"}}}
            for k in ("free_cooling_enabled", "FREE_COOL_OP", "cooling_backup_eer", "generators"):
                sub_cfg.pop(k, None)
            gens.append({"name": name, "priority": priority,
                         "calc": CoolingGenerationSystemCalculator(sub_cfg)})
        return gens

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        out = pd.DataFrame(index=df.index)
        out.loc[:, "hours"] = self._time_step_hours(df)
        out.loc[:, "T_ext_C"] = _series_from_aliases(
            df, ["T_ext", "theta_ext", "outdoor_temperature_C", "T_external_C"], default=None
        )
        if out["T_ext_C"].isna().any():
            raise ValueError("Outdoor temperature is required. Provide 'T_ext' or 'theta_ext'.")
        out.loc[:, "T_ext_C"] = out["T_ext_C"].astype(float)
        out.loc[:, "T_ext_wb_C"] = _series_from_aliases(
            df, ["T_ext_wb_C", "theta_ext_wb_C", "wet_bulb_temperature_C"], default=np.nan
        ).astype(float)

        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0).astype(float)
        out.loc[:, "Q_C_gen_in_req_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_C_gen_in_req_kWh",
                "Q_C_sto_in_kWh",
                "Q_C_dis_in_kWh",
                "Q_C_kWh",
                "QC_kWh",
                "cooling_kWh",
                "space_cooling_kWh",
            ],
            raw_aliases=["Q_C", "Cooling_needs"],
            fallback=np.maximum(-q_hc, 0.0),
        )
        out.loc[:, "theta_C_gen_out_req_C"] = _series_from_aliases(
            df,
            ["theta_C_gen_out_req_C", "T_C_sink_C", "cooling_sink_temperature_C"],
            default=self.theta_C_gen_out_set_C,
        ).astype(float)
        out.loc[:, "T_C_source_C"] = _series_from_aliases(
            df,
            ["T_C_source_C", "cooling_source_temperature_C", "source_temperature_C"],
            default=np.nan,
        ).astype(float)
        out.loc[:, "theta_sk_C"] = _series_from_aliases(
            df, ["theta_sk_C", "T_sink_other_C"], default=self.other_sink_temperature_C).astype(float)
        wet_bulb = out["T_ext_wb_C"].fillna(out["T_ext_C"] - self.wet_bulb_offset_K)
        loop = self.heat_rejection_deltaT_K + self.heat_rejection_loop_deltaT_K
        out.loc[:, "theta_cond_in_dry_C"] = out["T_ext_C"] + loop                   # (3a)
        out.loc[:, "theta_cond_in_wet_C"] = wet_bulb + loop                         # (3b)
        out.loc[:, "f_op_hr_wet_in"] = _series_from_aliases(
            df, ["f_op_hr_wet"], default=np.nan).astype(float)
        self._source_given = "T_C_source_C" in df.columns or "cooling_source_temperature_C" in df.columns \
            or "source_temperature_C" in df.columns
        out.loc[:, "theta_H_C_gen_abs_in_C"] = _series_from_aliases(
            df, ["theta_H_C_gen_abs_in_C"], default=self.theta_H_C_gen_abs_in_C).astype(float)
        out.loc[:, "theta_C_gen_ret_C"] = _series_from_aliases(
            df, ["theta_C_gen_in_C", "theta_C_dis_return_C", "T_C_return_C"], default=np.nan
        ).astype(float).fillna(out["theta_C_gen_out_req_C"] + self.free_cooling_return_deltaT_K)
        out.loc[:, "f_op_C"] = _series_from_aliases(
            df, ["f_op_C", "cooling_operation_factor"], default=1.0
        ).astype(float).clip(lower=0.0, upper=1.0)
        out.loc[:, "f_op_ctrl"] = _series_from_aliases(
            df, ["f_op_ctrl", "cooling_control_operation_factor"], default=1.0
        ).astype(float).clip(lower=0.0, upper=1.0)
        return out

    def _demand_from_columns(
        self,
        df: pd.DataFrame,
        kwh_aliases: list[str],
        raw_aliases: list[str],
        fallback: pd.Series | np.ndarray | float,
    ) -> pd.Series:
        series = _demand_series(df, kwh_aliases, "EN 16798-13")
        if series is not None:
            return series.astype(float).clip(lower=0.0)

        raw = _series_from_aliases(df, raw_aliases, default=None)
        if raw is not None:
            values = raw.astype(float)
            if self.demand_unit == "wh":
                values = values / 1000.0
            return values.clip(lower=0.0)

        if isinstance(fallback, pd.Series):
            return fallback.reindex(df.index).astype(float).clip(lower=0.0)
        return pd.Series(fallback, index=df.index, dtype=float).clip(lower=0.0)

    def _condenser_inlet(self, df: pd.DataFrame, f_wet: pd.Series) -> pd.Series:
        """theta_cond;in of (14a)-(14h); a given source temperature column takes precedence."""
        if self.heat_rejection_type == "AIR_C_COND":
            computed = df["T_ext_C"]
        elif self.heat_rejection_type == "OTHER":
            computed = df["theta_sk_C"]
        else:
            computed = df["theta_cond_in_dry_C"].where(f_wet < 0.5, df["theta_cond_in_wet_C"])
        theta = df["T_C_source_C"].astype(float).fillna(computed.astype(float))
        if self.theta_cond_in_lim_C is not None:
            theta = theta.clip(lower=self.theta_cond_in_lim_C)
        return theta

    def _wet_operation(self, df: pd.DataFrame, q_mech_req: pd.Series) -> pd.Series:
        """f_op;hr;wet of (13a)-(13h)."""
        if self.heat_rejection_type == "WET":
            return pd.Series(1.0, index=df.index)
        if self.heat_rejection_type != "HYBRID":
            return pd.Series(0.0, index=df.index)
        if self.hybrid_control == "TEMP":                                         # (13d)/(13e)
            computed = (df["T_ext_C"] > self.theta_lim_wet_hr_C).astype(float)
        else:                                                                     # (13f)/(13g)
            given = df["f_op_hr_wet_in"]
            if given.notna().all():
                return given.clip(0.0, 1.0).round()
            dry = self._condenser_inlet(df, pd.Series(0.0, index=df.index))
            cap_dry, _ = self._capacities(df, dry)
            q_max_dry = cap_dry * df["hours"] * df["f_op_C"]
            computed = (q_mech_req > q_max_dry + _KWH_EPS).astype(float)
        return df["f_op_hr_wet_in"].clip(0.0, 1.0).round().fillna(computed)

    def _capacities(self, df: pd.DataFrame, cond: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Capacity (kW) and full-load EER (COMP) / heat ratio (ABS) per step; NaN for OTHER."""
        if self.generation_type == "OTHER":                                       # (9g)
            factor = _evaluate(self.other_capacity_factor, "other_capacity_factor",
                               df["theta_C_gen_out_req_C"].to_numpy(), df["theta_sk_C"].to_numpy())
            return (pd.Series(self.nominal_capacity_kW * factor, index=df.index),
                    pd.Series(np.nan, index=df.index))
        capacity, perf = [], []
        for c, gen_out, hot in zip(cond.to_numpy(dtype=float), df["theta_C_gen_out_C"].to_numpy(dtype=float),
                                   df["theta_H_C_gen_abs_in_C"].to_numpy(dtype=float)):
            if self.generation_type == "ABS":
                cap, p = self._capacity_and_zeta(c, gen_out, hot)
            else:
                cap, p = self._capacity_and_eer(c, gen_out)
            capacity.append(cap)
            perf.append(p)
        return pd.Series(capacity, index=df.index, dtype=float), pd.Series(perf, index=df.index, dtype=float)

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        if self.generators:
            return self._simulate_generators(prepared)
        out = prepared.copy(deep=True)
        out.loc[:, "theta_C_gen_out_C"] = np.maximum(
            out["theta_C_gen_out_req_C"].astype(float),
            self.theta_C_gen_out_limit_C,
        )
        out.loc[:, "f_op_fc"] = self._free_cooling_factor(out)
        q_req = out["Q_C_gen_in_req_kWh"].astype(float)
        hours = out["hours"].astype(float)
        free_cooling = out["f_op_fc"].astype(float).clip(0.0, 1.0)
        q_free = q_req * free_cooling
        q_mechanical_req = q_req * (1.0 - free_cooling)

        out.loc[:, "f_op_hr_wet"] = self._wet_operation(out, q_mechanical_req)
        out.loc[:, "theta_cond_in_C"] = self._condenser_inlet(out, out["f_op_hr_wet"])
        out.loc[:, "T_C_source_C"] = out["theta_cond_in_C"]
        capacity, perf = self._capacities(out, out["theta_cond_in_C"])
        out.loc[:, "Q_C_gen_capacity_kW"] = capacity
        out.loc[:, "zeta_C_gen"] = perf if self.generation_type == "ABS" else np.nan
        out.loc[:, "EER_C_gen"] = perf if self.generation_type == "COMP" else np.nan

        max_available = out["Q_C_gen_capacity_kW"] * hours * out["f_op_C"]
        out.loc[:, "Q_C_gen_in_max_kWh"] = max_available
        q_mechanical = np.minimum(q_mechanical_req, max_available)
        q_missing = (q_mechanical_req - q_mechanical).clip(lower=0.0)

        if self.cooling_backup_eer is None:
            q_backup = pd.Series(0.0, index=out.index)
            q_unmet = q_missing
        else:
            q_backup = q_missing
            q_unmet = pd.Series(0.0, index=out.index)

        out.loc[:, "Q_C_gen_free_kWh"] = q_free
        out.loc[:, "Q_C_gen_mech_in_kWh"] = q_mechanical
        out.loc[:, "Q_C_gen_backup_in_kWh"] = q_backup
        out.loc[:, "Q_C_gen_unmet_kWh"] = q_unmet
        out.loc[:, "Q_C_gen_in_kWh"] = q_free + q_mechanical + q_backup

        out.loc[:, "f_C_PL"] = _safe_divide(q_mechanical, max_available).clip(0.0, 1.0)   # (12)
        out.loc[:, "EER_C_gen_full_load"] = out["EER_C_gen"]
        out.loc[:, "zeta_C_gen_full_load"] = out["zeta_C_gen"]
        method = self.part_load_performance_method
        if method == "simple":
            out.loc[:, "f_C_PLF"] = 1.0
            out.loc[:, "f_C_PL_for_performance"] = out["f_C_PL"]
            out.loc[:, "part_load_degradation_coefficient"] = 0.0
        else:
            # (20a)/(20b): below f_C;PL;min the performance stays at its f_C;PL;min value
            plr = out["f_C_PL"].clip(lower=self.part_load_minimum_capacity_ratio, upper=1.0)
            loaded = q_mechanical > _KWH_EPS
            if method == "en16798_13":                          # (19), (20a)/(20b)
                eer_pl = self.nominal_eer * self._f_eer(out["theta_cond_in_C"].to_numpy(),
                                                        out["theta_C_gen_out_C"].to_numpy(),
                                                        plr.to_numpy(dtype=float))
                factors = eer_pl / out["EER_C_gen_full_load"].to_numpy(dtype=float)
                if np.any(eer_pl[loaded.to_numpy()] <= 0.0):
                    raise ValueError("EN 16798-13 (20a): non-positive EER; check the EN 14825 points.")
            elif method == "en14825":
                factors = np.array([
                    en14825_part_load_factor(
                        capacity_ratio=x,
                        degradation_coefficient=self.part_load_degradation_coefficient,
                        unit_type=self.part_load_unit_type,
                    ) for x in plr])
            else:
                factors = _part_load_curve(self.part_load_curve, plr.to_numpy(dtype=float))
                if np.any(factors[loaded.to_numpy()] <= 0.0):
                    raise ValueError("part_load_curve_coefficients give a non-positive factor in the "
                                     "operating range.")
            out.loc[:, "f_C_PLF"] = np.where(loaded, factors, 1.0)
            out.loc[:, "f_C_PL_for_performance"] = plr
            out.loc[:, "part_load_degradation_coefficient"] = (
                self.part_load_degradation_coefficient if method == "en14825" else 0.0)
            if self.generation_type == "ABS":                  # library extension: zeta at part load
                out.loc[:, "zeta_C_gen"] = (out["zeta_C_gen_full_load"] * out["f_C_PLF"]).clip(lower=_KWH_EPS)
            else:
                out.loc[:, "EER_C_gen"] = (out["EER_C_gen_full_load"] * out["f_C_PLF"]).clip(lower=_KWH_EPS)
        out.loc[:, "t_C_gen_runtime_h"] = _safe_divide(
            q_mechanical,
            out["Q_C_gen_capacity_kW"].replace(0.0, np.nan),
        ).fillna(0.0).clip(lower=0.0, upper=hours)

        if self.generation_type == "ABS":
            out.loc[:, "E_C_gen_el_in_kWh"] = q_mechanical * self.absorption_electric_ratio
            out.loc[:, "Q_H_C_gen_abs_in_kWh"] = _safe_divide(                    # (21)
                q_mechanical, out["zeta_C_gen"].replace(0.0, np.nan)).fillna(0.0)
            out.loc[:, "Q_C_gen_abs_kWh"] = q_mechanical
        elif self.generation_type == "OTHER":                                     # (24)
            ratio = _evaluate(self.other_electric_ratio, "other_electric_ratio",
                              out["theta_C_gen_out_C"].to_numpy(), out["theta_sk_C"].to_numpy(),
                              out["f_C_PL"].to_numpy())
            out.loc[:, "f_C_gen_oth_el"] = ratio
            out.loc[:, "E_C_gen_el_in_kWh"] = q_mechanical * ratio
            out.loc[:, "EER_C_gen"] = np.where(ratio > 0.0, 1.0 / np.where(ratio > 0.0, ratio, 1.0), np.nan)
            out.loc[:, "EER_C_gen_full_load"] = out["EER_C_gen"]
            out.loc[:, "Q_H_C_gen_abs_in_kWh"] = 0.0
        else:
            out.loc[:, "E_C_gen_el_in_kWh"] = _safe_divide(
                q_mechanical,
                out["EER_C_gen"].replace(0.0, np.nan),
            ).fillna(0.0)
            out.loc[:, "Q_H_C_gen_abs_in_kWh"] = 0.0
        if "Q_C_gen_abs_kWh" not in out.columns:
            out.loc[:, "Q_C_gen_abs_kWh"] = 0.0
        out.loc[:, "E_C_backup_in_kWh"] = (
            q_backup / self.cooling_backup_eer
            if self.cooling_backup_eer is not None
            else 0.0
        )
        # (16) plus the backup electricity (its cooling is already in Q_C_gen_in)
        out.loc[:, "Q_C_gen_out_kWh"] = (
            out["Q_C_gen_in_kWh"] + out["E_C_gen_el_in_kWh"] + out["E_C_backup_in_kWh"]
            + out["Q_H_C_gen_abs_in_kWh"]
        )
        out.loc[:, "Q_C_gen_out_rbl_kWh"] = (
            out["Q_C_gen_out_kWh"] * self.heat_recovery_fraction
        )
        out.loc[:, "theta_C_gen_out_max_C"] = out["theta_cond_in_C"]
        out.loc[:, "W_C_aux_gen_kWh"] = self._auxiliary_energy(out)
        out.loc[:, "E_C_total_kWh"] = (
            out["E_C_gen_el_in_kWh"]
            + out["E_C_backup_in_kWh"]
            + out["W_C_aux_gen_kWh"]
        )
        return out

    def _simulate_generators(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Several generators with priorities, EN 16798-13 (7)-(11)."""
        out = prepared.copy(deep=True)
        out.loc[:, "theta_C_gen_out_C"] = np.maximum(out["theta_C_gen_out_req_C"].astype(float),
                                                    self.theta_C_gen_out_limit_C)
        out.loc[:, "f_op_fc"] = self._free_cooling_factor(out)
        q_req = out["Q_C_gen_in_req_kWh"].astype(float)
        f_fc = out["f_op_fc"].astype(float).clip(0.0, 1.0)
        q_free = q_req * f_fc
        q_mech_req = q_req * (1.0 - f_fc)

        stage = pd.DataFrame({
            "T_ext": out["T_ext_C"], "T_ext_wb_C": out["T_ext_wb_C"],
            "theta_C_gen_out_req_C": out["theta_C_gen_out_req_C"],
            "theta_H_C_gen_abs_in_C": out["theta_H_C_gen_abs_in_C"],
            "theta_sk_C": out["theta_sk_C"],
            "f_op_C": out["f_op_C"], "f_op_ctrl": out["f_op_ctrl"],
            "time_step_hours": out["hours"],
        }, index=out.index)
        if self._source_given:
            stage["T_C_source_C"] = out["T_C_source_C"]

        # (13f)/(13g): the hybrid mode is decided once for the system, on the sum of the dry maxima
        f_wet = out["f_op_hr_wet_in"].clip(0.0, 1.0).round()
        if self.heat_rejection_type == "HYBRID" and f_wet.isna().any():
            if self.hybrid_control == "TEMP":
                computed = (out["T_ext_C"] > self.theta_lim_wet_hr_C).astype(float)
            else:
                dry_stage = stage.assign(Q_C_gen_in_req_kWh=q_mech_req, f_op_hr_wet=0.0)
                q_max_dry = sum(g["calc"].run_timeseries(dry_stage).timeseries["Q_C_gen_in_max_kWh"]
                                for g in self.generators)
                computed = (q_mech_req > q_max_dry + _KWH_EPS).astype(float)
            f_wet = f_wet.fillna(computed)
        else:
            f_wet = f_wet.fillna(self._wet_operation(out, q_mech_req))
        out.loc[:, "f_op_hr_wet"] = f_wet
        out.loc[:, "theta_cond_in_C"] = self._condenser_inlet(out, f_wet)
        stage["f_op_hr_wet"] = f_wet

        maxima = {}
        for g in self.generators:                                           # (8)/(9)
            first = g["calc"].run_timeseries(stage.assign(Q_C_gen_in_req_kWh=q_mech_req)).timeseries
            maxima[g["name"]] = first["Q_C_gen_in_max_kWh"].to_numpy(dtype=float)
        remaining = q_mech_req.to_numpy(dtype=float).copy()
        assigned = {g["name"]: np.zeros(len(out)) for g in self.generators}
        for prio in sorted({g["priority"] for g in self.generators}):      # (10)/(11)
            group = [g["name"] for g in self.generators if g["priority"] == prio]
            cap = sum(maxima[n] for n in group)
            take = np.minimum(remaining, cap)
            for n in group:
                share = np.divide(maxima[n], cap, out=np.zeros_like(cap), where=cap > _KWH_EPS)
                assigned[n] = take * share
            remaining = remaining - take

        totals = {k: np.zeros(len(out)) for k in (
            "Q_C_gen_mech_in_kWh", "E_C_gen_el_in_kWh", "Q_H_C_gen_abs_in_kWh", "W_C_aux_gen_kWh",
            "Q_C_gen_out_kWh", "Q_C_gen_out_rbl_kWh", "Q_C_gen_in_max_kWh", "Q_C_gen_capacity_kW",
            "t_C_gen_runtime_h")}
        q_comp = np.zeros(len(out))
        q_abs = np.zeros(len(out))
        t_mix = np.zeros(len(out))
        for g in self.generators:
            n = g["name"]
            ts = g["calc"].run_timeseries(stage.assign(Q_C_gen_in_req_kWh=assigned[n])).timeseries
            for col in ("Q_C_gen_in_kWh", "Q_C_gen_in_max_kWh", "E_C_gen_el_in_kWh",
                        "Q_H_C_gen_abs_in_kWh", "W_C_aux_gen_kWh", "Q_C_gen_out_kWh", "f_C_PL",
                        "EER_C_gen", "zeta_C_gen"):
                out.loc[:, f"{col}_{n}"] = ts[col].to_numpy()
            totals["Q_C_gen_mech_in_kWh"] += ts["Q_C_gen_in_kWh"].to_numpy()
            for col in ("E_C_gen_el_in_kWh", "Q_H_C_gen_abs_in_kWh", "W_C_aux_gen_kWh", "Q_C_gen_out_kWh",
                        "Q_C_gen_out_rbl_kWh", "Q_C_gen_in_max_kWh", "Q_C_gen_capacity_kW",
                        "t_C_gen_runtime_h"):
                totals[col] += ts[col].to_numpy()
            if g["calc"].generation_type in {"COMP", "OTHER"}:      # electrically driven
                q_comp += ts["Q_C_gen_in_kWh"].to_numpy()
            else:
                q_abs += ts["Q_C_gen_in_kWh"].to_numpy()
            out.loc[:, f"theta_C_gen_out_C_{n}"] = ts["theta_C_gen_out_C"].to_numpy()
            t_mix += ts["Q_C_gen_in_kWh"].to_numpy() * ts["theta_C_gen_out_C"].to_numpy()

        q_missing = pd.Series(np.maximum(remaining, 0.0), index=out.index)
        if self.cooling_backup_eer is None:
            q_backup, q_unmet = pd.Series(0.0, index=out.index), q_missing
        else:
            q_backup, q_unmet = q_missing, pd.Series(0.0, index=out.index)
        e_backup = q_backup / self.cooling_backup_eer if self.cooling_backup_eer else q_backup * 0.0

        # (4) per generator; the system outlet is the energy-weighted mix of the running generators
        q_run = totals["Q_C_gen_mech_in_kWh"]
        idle = np.maximum(out["theta_C_gen_out_req_C"].to_numpy(dtype=float), self.minimum_outlet_temperature_C)
        out.loc[:, "theta_C_gen_out_C"] = np.where(
            q_run > _KWH_EPS, t_mix / np.where(q_run > _KWH_EPS, q_run, 1.0), idle)
        out.loc[:, "Q_C_gen_abs_kWh"] = q_abs
        out.loc[:, "Q_C_gen_free_kWh"] = q_free
        out.loc[:, "Q_C_gen_mech_in_kWh"] = totals["Q_C_gen_mech_in_kWh"]
        out.loc[:, "Q_C_gen_backup_in_kWh"] = q_backup
        out.loc[:, "Q_C_gen_unmet_kWh"] = q_unmet
        out.loc[:, "Q_C_gen_in_kWh"] = q_free + out["Q_C_gen_mech_in_kWh"] + q_backup
        out.loc[:, "Q_C_gen_in_max_kWh"] = totals["Q_C_gen_in_max_kWh"]
        out.loc[:, "Q_C_gen_capacity_kW"] = totals["Q_C_gen_capacity_kW"]
        out.loc[:, "f_C_PL"] = _safe_divide(out["Q_C_gen_mech_in_kWh"], out["Q_C_gen_in_max_kWh"]).clip(0, 1)
        out.loc[:, "f_C_PLF"] = np.nan
        out.loc[:, "E_C_gen_el_in_kWh"] = totals["E_C_gen_el_in_kWh"]
        out.loc[:, "Q_H_C_gen_abs_in_kWh"] = totals["Q_H_C_gen_abs_in_kWh"]
        comp_el = sum(out[f"E_C_gen_el_in_kWh_{g['name']}"] for g in self.generators
                      if g["calc"].generation_type in {"COMP", "OTHER"})
        out.loc[:, "EER_C_gen"] = _safe_divide(pd.Series(q_comp, index=out.index),
                                               pd.Series(comp_el, index=out.index)).replace(0.0, np.nan)
        out.loc[:, "EER_C_gen_full_load"] = out["EER_C_gen"]
        out.loc[:, "t_C_gen_runtime_h"] = totals["t_C_gen_runtime_h"]
        out.loc[:, "E_C_backup_in_kWh"] = e_backup
        # (16): free cooling and backup add to the generators' rejected heat
        out.loc[:, "Q_C_gen_out_kWh"] = totals["Q_C_gen_out_kWh"] + q_free + q_backup + e_backup
        out.loc[:, "Q_C_gen_out_rbl_kWh"] = totals["Q_C_gen_out_rbl_kWh"]
        out.loc[:, "theta_C_gen_out_max_C"] = out["theta_cond_in_C"]
        extra = (q_free + q_backup + e_backup)
        out.loc[:, "W_C_aux_gen_kWh"] = totals["W_C_aux_gen_kWh"] + self._heat_rejection_aux(extra, f_wet)
        out.loc[:, "E_C_total_kWh"] = out["E_C_gen_el_in_kWh"] + e_backup + out["W_C_aux_gen_kWh"]
        return out

    def _heat_rejection_aux(self, q_rejected: pd.Series, f_wet: pd.Series) -> pd.Series:
        """W_hr of (26a)/(26b)."""
        if self.performance_includes_heat_rejection_aux:
            return q_rejected * 0.0
        if self.heat_rejection_type == "OTHER":
            return q_rejected * (1.0 - f_wet) * self.p_hr_el_other
        return q_rejected * ((1.0 - f_wet) * self.p_hr_el_dry + f_wet * self.p_hr_el_wet)

    def _free_cooling_factor(self, df: pd.DataFrame) -> pd.Series:
        if not self.free_cooling_enabled:
            return pd.Series(0.0, index=df.index)
        if self.heat_rejection_type == "AIR_C_COND":
            return pd.Series(0.0, index=df.index)
        if self.heat_rejection_type in {"WET", "HYBRID"}:   # (6c)
            sink = df["T_ext_wb_C"].fillna(df["T_ext_C"] - self.wet_bulb_offset_K)
        elif self.heat_rejection_type == "OTHER":                  # (6d)
            sink = df["theta_sk_C"]
        else:
            sink = df["T_ext_C"]
        if self.free_cooling_model == "partial":
            q_req = df["Q_C_gen_in_req_kWh"].astype(float)
            q_fc_max = self.free_cooling_ua_kW_K * (df["theta_C_gen_ret_C"] - sink).clip(lower=0.0) * df["hours"]
            df.loc[:, "Q_C_gen_fc_max_kWh"] = q_fc_max
            return _safe_divide(q_fc_max, q_req.where(q_req > _KWH_EPS)).clip(0.0, 1.0)
        delta = df["theta_C_gen_out_req_C"] - sink
        return pd.Series(1.0, index=df.index).where(
            delta > self.free_cooling_deltaT_K,
            0.0,
        )

    def _capacity_and_zeta(self, cond_in_C: float, evap_out_C: float, hot_in_C: float) -> tuple[float, float]:
        """Absorption chiller: capacity (9d) and heat ratio (22) from zeta_c (23)."""
        zc = _carnot_heat_ratio(evap_out_C, cond_in_C, hot_in_C)
        if zc <= 0.0:
            return 0.0, _KWH_EPS
        p = _poly(self.abs_coefficients, zc)
        p_n = _poly(self.abs_coefficients, self._zeta_c_n)
        return max(self.nominal_capacity_kW * p / p_n, 0.0), max(p, _KWH_EPS)

    def _capacity_and_eer(self, source_temperature_C: float, sink_temperature_C: float) -> tuple[float, float]:
        if self.performance_map is not None:
            if self._cap_pivot is not None:
                # Fast path: pre-built bilinear grids (the common case)
                cap = _bilinear_or_linear(*self._cap_pivot, source_temperature_C, sink_temperature_C)
                eer = _bilinear_or_linear(*self._eer_pivot, source_temperature_C, sink_temperature_C)
            else:
                # Scattered map fallback: rebuild pivot each call (rare)
                cap = _interpolate_performance(
                    self.performance_map, source_temperature_C, sink_temperature_C, "capacity_kW"
                )
                eer = _interpolate_performance(
                    self.performance_map, source_temperature_C, sink_temperature_C, "eer"
                )
            return max(float(cap), 0.0), max(float(eer), _KWH_EPS)

        capacity = self.nominal_capacity_kW
        nominal_eer = float(self.nominal_eer or _default_nominal_eer(capacity))
        current_lift = max(source_temperature_C - sink_temperature_C, _KWH_EPS)
        nominal_lift = max(
            self.theta_cond_in_nominal_C - self.theta_C_evap_out_nominal_C,
            _KWH_EPS,
        )
        current_carnot = (_T0_ABS_K + sink_temperature_C) / current_lift
        nominal_carnot = (_T0_ABS_K + self.theta_C_evap_out_nominal_C) / nominal_lift
        eer = nominal_eer * current_carnot / max(nominal_carnot, _KWH_EPS)
        if self.en16798_coefficients is not None:                                # (19), (20a) at full load
            eer = nominal_eer * float(self._f_eer(source_temperature_C, sink_temperature_C, 1.0))
        if self.capacity_temperature_correction:                                 # (9a)-(9c)
            dc = self.delta_theta_corr_K
            num = (_T0_ABS_K + sink_temperature_C) / max(
                source_temperature_C - sink_temperature_C + dc, _KWH_EPS)
            den = (_T0_ABS_K + self.theta_C_evap_out_nominal_C) / max(
                self.theta_cond_in_nominal_C - self.theta_C_evap_out_nominal_C + dc, _KWH_EPS)
            capacity = capacity * num / den
        return capacity, max(float(eer), _KWH_EPS)

    def _auxiliary_energy(self, df: pd.DataFrame) -> pd.Series:
        active_hours = df["t_C_gen_runtime_h"].astype(float)
        q_rejected = df["Q_C_gen_out_kWh"].astype(float)

        w_hr = self._heat_rejection_aux(q_rejected, df["f_op_hr_wet"])
        w_ctrl = active_hours * self.control_power_kW * df["f_op_ctrl"].astype(float)
        w_extra = active_hours * self.additional_auxiliary_power_kW
        return w_hr + w_ctrl + w_extra

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        q_req = float(results["Q_C_gen_in_req_kWh"].sum())
        q_in = float(results["Q_C_gen_in_kWh"].sum())
        e_el = float(results["E_C_gen_el_in_kWh"].sum())
        e_backup = float(results["E_C_backup_in_kWh"].sum())
        w_aux = float(results["W_C_aux_gen_kWh"].sum())
        e_total = e_el + e_backup + w_aux
        weights = results["Q_C_gen_in_kWh"] + _KWH_EPS
        summary = {
            "hours": float(results["hours"].sum()),
            "QC_gen_in_req_kWh": q_req,
            "QC_gen_in_kWh": q_in,
            "QC_gen_free_kWh": float(results["Q_C_gen_free_kWh"].sum()),
            "QC_gen_mech_in_kWh": float(results["Q_C_gen_mech_in_kWh"].sum()),
            "QC_gen_backup_in_kWh": float(results["Q_C_gen_backup_in_kWh"].sum()),
            "QC_gen_unmet_kWh": float(results["Q_C_gen_unmet_kWh"].sum()),
            "QC_gen_in_max_kWh": float(results["Q_C_gen_in_max_kWh"].sum()),
            "QC_gen_out_kWh": float(results["Q_C_gen_out_kWh"].sum()),
            "QC_gen_out_rbl_kWh": float(results["Q_C_gen_out_rbl_kWh"].sum()),
            "QH_C_gen_abs_in_kWh": float(results["Q_H_C_gen_abs_in_kWh"].sum()),
            "zeta_C_gen_mean": _weighted_mean(results["zeta_C_gen"], weights)
            if "zeta_C_gen" in results else float("nan"),
            "EC_gen_el_in_kWh": e_el,
            "EC_backup_in_kWh": e_backup,
            "WC_aux_gen_kWh": w_aux,
            "EC_total_kWh": e_total,
            "SEER_C_gen": _ratio(q_in, e_total),
            # indices per energy carrier: SEER_C_gen counts electricity only
            "SEER_C_gen_el": _ratio(q_in, e_total),
            "QC_gen_abs_kWh": float(results["Q_C_gen_abs_kWh"].sum()),
            "zeta_C_gen_seasonal": _ratio(float(results["Q_C_gen_abs_kWh"].sum()),
                                          float(results["Q_H_C_gen_abs_in_kWh"].sum())),
            "EER_C_gen_all_inputs": _ratio(q_in, e_total + float(results["Q_H_C_gen_abs_in_kWh"].sum())),
            "EER_C_gen_mean": _weighted_mean(results["EER_C_gen"], weights),
            "EER_C_gen_full_load_mean": _weighted_mean(
                results["EER_C_gen_full_load"], weights
            )
            if "EER_C_gen_full_load" in results
            else _weighted_mean(results["EER_C_gen"], weights),
            "f_C_PL_mean": _weighted_mean(results["f_C_PL"], weights),
            "f_C_PLF_mean": _weighted_mean(results["f_C_PLF"], weights)
            if "f_C_PLF" in results
            else 1.0,
            "theta_C_gen_out_mean_C": _weighted_mean(results["theta_C_gen_out_C"], weights),
            "theta_cond_in_mean_C": _weighted_mean(results["theta_cond_in_C"], weights),
            "t_C_gen_runtime_h": float(results["t_C_gen_runtime_h"].sum()),
            "t_C_hr_wet_h": float((results["f_op_hr_wet"] * results["hours"]).sum()),
        }
        return summary

    def _time_step_hours(self, df: pd.DataFrame) -> pd.Series:
        step = _series_from_aliases(df, ["time_step_hours", "dt_h"], default=np.nan)
        if not step.isna().all():
            return step.astype(float).clip(lower=0.0)

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            diffs = df.index.to_series().diff().dt.total_seconds().dropna() / 3600.0
            if not diffs.empty and np.isfinite(diffs.median()) and diffs.median() > 0:
                return pd.Series(float(diffs.median()), index=df.index)

        return pd.Series(self.default_time_step_h, index=df.index)


_SHARED_GENERATOR_KEYS = {
    "time_step_hours", "demand_unit", "HEAT_REJ_TYPE", "heat_rejection_type",
    "theta_C_gen_out_limit_C", "theta_C_gen_out_lim_C", "theta_C_gen_out_set_C",
    "cooling_sink_temperature_C", "heat_rejection_deltaT_K", "delta_theta_hr_K",
    "heat_rejection_loop_deltaT_K", "delta_theta_ls_dis_hr_K", "wet_bulb_offset_K",
    "other_sink_temperature_C", "theta_H_C_gen_abs_in_C", "performance_includes_heat_rejection_aux",
    "p_hr_el_dry", "p_hr_el_wet", "p_hr_el_other", "phr_el_dry", "phr_el_wet", "phr_el_other",
    "HBRD_HEAT_REJ_CTRL", "hybrid_heat_rejection_control", "theta_lim_wet_hr_C", "theta_cond_in_lim_C",
}


def _fit_en16798_13(points, eer_n: float) -> tuple[float, float, float, float, float]:
    """EN 16798-13 6.3.2.2.2.1, system (1): C1-C4 and delta_theta_corr from the EN 14825
    points A, B, C, D and the 5th point (estimated with (2) when not given).

    Each point: ``f_C_PL`` (fraction 0-1), ``EER``, ``theta_C_evap_out_C``, ``theta_cond_in_C``.
    """
    pts = [dict(p) for p in points]
    if len(pts) not in (4, 5):
        raise ValueError("en14825_points: give the points A, B, C, D (and optionally the 5th point).")
    for p in pts:
        for k in ("f_C_PL", "EER", "theta_C_evap_out_C", "theta_cond_in_C"):
            if k not in p:
                raise ValueError(f"en14825_points: each point needs {k}.")
        if not 0.0 < float(p["f_C_PL"]) <= 1.0 or float(p["EER"]) <= 0.0:
            raise ValueError("en14825_points: f_C_PL must be in (0, 1] and EER > 0.")
    if len(pts) == 4:                                                     # (2), delta_theta_corr = 0
        a, c = pts[0], pts[2]
        eer5 = float(c["EER"]) * (float(c["theta_cond_in_C"]) - float(c["theta_C_evap_out_C"])) / (
            float(a["theta_cond_in_C"]) - float(c["theta_C_evap_out_C"]))
        pts.append({"f_C_PL": c["f_C_PL"], "EER": eer5, "theta_C_evap_out_C": c["theta_C_evap_out_C"],
                    "theta_cond_in_C": a["theta_cond_in_C"]})
    rows, rhs = [], []
    for p in pts:
        x, r = float(p["f_C_PL"]), float(p["EER"]) / eer_n
        te, tc = float(p["theta_C_evap_out_C"]), float(p["theta_cond_in_C"])
        rows.append([x ** 3, x ** 2, x, 1.0, -r * x / (_T0_ABS_K + te)])
        rhs.append(r * (tc - te) / (_T0_ABS_K + te))
    a_mat = np.array(rows)
    if abs(np.linalg.det(a_mat)) < 1e-14:
        raise ValueError("en14825_points: the system (1) of EN 16798-13 is singular (points not independent).")
    return tuple(float(v) for v in np.linalg.solve(a_mat, np.array(rhs)))


def _number_or_callable(value: Any, name: str):
    if callable(value):
        return value
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative number or a function.")
    return result


def _evaluate(func, name: str, *columns) -> np.ndarray:
    """Evaluate a user number/function row by row; the result must be finite and >= 0."""
    n = len(columns[0])
    if not callable(func):
        return np.full(n, float(func))
    values = np.array([float(func(*(float(c[i]) for c in columns))) for i in range(n)], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} returned a negative or non-finite value.")
    return values


def _part_load_curve(coeffs: tuple[float, float, float], x):
    a, b, c = coeffs
    return a + b * x + c * x * x


def _carnot_heat_ratio(evap_out_C: float, cond_in_C: float, hot_in_C: float) -> float:
    """EN 16798-13 (23); <= 0 when the temperatures do not allow operation."""
    if not (hot_in_C > cond_in_C > evap_out_C):
        return 0.0
    return ((_T0_ABS_K + evap_out_C) * (hot_in_C - cond_in_C)) / (
        (_T0_ABS_K + hot_in_C) * (cond_in_C - evap_out_C))


def _poly(coeffs: tuple[float, float, float], x: float) -> float:
    c5, c6, c7 = coeffs
    return c5 * x * x + c6 * x + c7


def _series_from_aliases(
    df: pd.DataFrame,
    aliases: list[str],
    default: float | None,
) -> pd.Series | None:
    for alias in aliases:
        if alias in df.columns:
            return df[alias]
    if default is None:
        return None
    return pd.Series(default, index=df.index, dtype=float)


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return result


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("optional positive values must be positive when provided.")
    return result


def _fraction(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return result


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float)
    w = weights.astype(float)
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return numerator / denominator


def _default_nominal_eer(capacity_kW: float) -> float:
    if capacity_kW <= 12.0:
        return 2.9
    if capacity_kW <= 100.0:
        return 3.1
    if capacity_kW <= 300.0:
        return 3.2
    if capacity_kW <= 600.0:
        return 3.4
    return 3.5
