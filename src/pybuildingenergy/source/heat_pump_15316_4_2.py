"""Heat-pump generation according to EN 15316-4-2:2017, Method A (hourly/bin).

This module implements EN 15316-4-2:2017's Method A (clause 6, "Generation
with heat pump systems - Energy calculation (hourly, bin)"), REPLACING the
EN 15316-4-2:2008 bin-method part-load treatment this module previously
used. The 2017 edition still steps through hourly/bin time steps (clause
3.7's "bin" - a statistical outdoor-temperature class - and clause 6.4's
"hourly (or lower)"/"bin" time steps are both still core concepts of the
2017 edition; Method A keeps the hourly/bin stepping this module already
used), but the PART-LOAD performance calculation at each step is now the
2017 edition's own Eq.(14)-(30) (clause 6.7.2.6/6.7.2.7), not the 2008
edition's bin-ratio approach.

It is intended to sit after the building/distribution calculations: users
provide the thermal energy requirements for space heating, domestic hot
water and, for reversible heat pumps, space cooling. Product performance
data are supplied as COP/EER and capacity maps as a function of source
(evaporator inlet) and sink (condenser outlet) temperatures - clause
6.7.2's "Path A" (EN 14511 series), whose Table 11/12 bidirectional
temperature-correction matrix (Eq.(3)-(9)) is, by construction, the SAME
bilinear interpolation over a source/sink temperature grid this module
already implements (_interpolate_performance()/_bilinear_or_linear()) -
reused unchanged for full-load COP/capacity.

Formulas implemented from EN 15316-4-2:2017, clause 6:

* Eq.(1): generation sub-system energy balance (E_HW,gen,in x COP_HW,gen
  = Q_HW,gen,out + Q_HW,gen,ls,tot - Q_HW,gen,in - f_gen,auxls,rvd x
  W_HW,gen,aux).
* Eq.(3)-(9) (clause 6.7.2.2/6.7.2.3, Path A): full-load COP/capacity from
  the EN 14511-series rated points, bidirectionally interpolated over
  inlet (evaporator)/outlet (condenser) temperature - reuses this module's
  existing bilinear interpolation over heating_performance_map/
  dhw_performance_map/cooling_performance_map.
* Eq.(10)-(13) (clause 6.7.2.5): energy delivered and operating time at
  full load, compared against the demand.
* Eq.(14)-(24) (clause 6.7.2.6/6.7.2.7.1): part-load ratio LR and COP at
  part load for CONTINUOUSLY-MODULATING (inverter) compressors - net COP
  (auxiliary power P_gen,aux stripped out, Eq.(18)-(20)), then corrected
  for part load between LR_cont,min and 1 (Eq.(21)).
* Eq.(25)-(30) (clause 6.7.2.7.2): part-load COP/energy for ON/OFF
  compressors (or an inverter compressor below its own LR_cont,min), using
  the cycling-inertia parameters tau_eq (product-specific, Table B default)
  and tau_out,em,type (Table 13, by emitter inertia category).
* Eq.(31)-(32): zero-demand step - standby_power_W only accrues while the
  generator is POWERED (Eq.31); the input "powered"/"hp_powered"/
  "operating_season" column (default: always powered, 1.0 - unchanged
  pre-existing behaviour) lets a caller mark specific steps as outside the
  unit's operating season for Eq.(32)'s zero-energy treatment instead.
* Eq.(44) (clause 6.8): auxiliary energy, Eq.(45) (clause 6.9): energy used
  from the heat source, Eq.(46) (clause 6.10): back-up heater energy.

NOT implemented from the 2017 edition (documented gaps, not silently
dropped):

* Clause 6.7.3, "Path B" (EN 14825-series exergetic-efficiency approach,
  Eq.(33)-(40), Annex D) - an ALTERNATIVE way to get COP/capacity from
  EN 14825 A/B/C/D rating points via a Carnot-COP x exergetic-efficiency
  construction, instead of Path A's direct bidirectional interpolation.
  This module's pre-existing, simpler en14825_part_load_factor()-based
  degradation correction (performance_14511_14825.py) remains available
  as the part_load_performance_method="en14825" option, but it is NOT a
  faithful implementation of Path B/Annex D.
* Clause 6.7.4 (absorption/adsorption, combustion-engine-driven heat
  pumps, Eq.(41)-(43), Table B.10/B.11) - this module only implements the
  electrically-driven compressor path (Eq.(14)-(30)).
* Clause 7 (Method B, monthly/annual) - a different time resolution,
  requiring different input/output wiring than this module's hourly/bin
  stepping; not implemented (see docs/standards_errata.md-style scoping
  notes in this module's own tests if this changes).
* Annex A's normative input-data TEMPLATE and Annex B's full default-value
  tables are not transcribed in full; a curated, documented subset of
  defaults is used where the formulas need one (tau_out,em,type from
  Table 13 is transcribed in full - only 5 rows).
* Clause 5.9's genuine SIMULTANEOUS operation (heating + DHW covered AT
  THE SAME TIME) needs its OWN dedicated test data (capacity/COP measured
  specifically for the combined mode - "these test results have to be
  available"), which this module has no input for.
  combined_operation="independent" instead runs the SAME performance map
  for both services, each initially allocated against the full step
  duration independently, then caps their COMBINED runtime at the step
  duration (see _cap_combined_independent_runtime()) so one compressor
  cannot appear to deliver output twice over - a safety cap against
  double-counting, NOT a faithful clause 5.9 calculation.
  combined_operation="alternative" (the default) is the more literally
  faithful mode (clause 3.1's "alternate operation": a diverter switches
  the SAME generator between the two services, never both at once).
* Performance maps are interpolated as given, without checking they
  actually contain EN 14511/14825's own mandatory rating points (see
  _warn_about_performance_map_quality() for the sanity-check warnings
  this module DOES apply - not an EN 14511 conformance check). Conformance
  of the input data remains the caller's responsibility.
* Storage losses (heating_storage_loss_kWh_per_day/
  dhw_storage_loss_kWh_per_day) are modelled as a single normalised daily
  loss value, temperature-corrected (_storage_loss()) - not EN 15316-5's
  own full storage-subsystem model (StorageSystemCalculator). Use that
  calculator upstream/downstream for a full storage-subsystem calculation.

The cooling mode is treated as a reversible heat-pump extension using the
same temperature-bin and product-map structure; EN 15316-4-2 itself is a
heating/DHW generation standard, so cooling is NOT part of the 2017 Method
A formulas above - the pre-existing, simplified cooling treatment is kept
unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .performance_14511_14825 import en14825_part_load_factor


_KWH_EPS = 1e-12

# EN 15316-4-2:2017, Table B.11: default LR_cont,min (clause 6.7.2.6.1/B.1.3)
# by compressor technology - "LR_cont,min = 1 for ON-OFF technology" (i.e.
# no continuous-modulation range at all - Eq.(25)-(30)'s ON/OFF branch
# applies for every LR<1), "LR_cont,min = 0,2 for inverter technology".
DEFAULT_LR_CONT_MIN_BY_COMPRESSOR_TECH = {"on_off": 1.0, "inverter": 0.2}

# Table B.11: TAU_EQ (heat pump cycling inertia, Eq.(28)) default - a
# single value, not product/emitter-specific, unless overridden.
DEFAULT_TAU_EQ_S = 30.0

# Table B.11: TAU_OUT_EM_TYPE_W / _STO (DHW/storage distribution time
# constant, Eq.(28)) - single default values, no emitter-type choice (that
# only applies to space heating, Table 13 below).
DEFAULT_TAU_OUT_EM_TYPE_W_S = 1560.0
DEFAULT_TAU_OUT_EM_TYPE_STO_S = 1560.0

# Clause 6.7.2.7.2 / Table 13: distribution time constant tau_out,em,type
# [s] for SPACE HEATING (Eq.(28)), by emitter inertia category.
# "heavy_embedded" (1920 s) matches Annex B's Table B.11 TAU_OUT_EM_TYPE_H
# default exactly, and is this module's own default category.
TAU_OUT_EM_TYPE_H_S_BY_CATEGORY = {
    "heavy_embedded": 1920.0,
    "light_embedded": 1370.0,
    "low": 360.0,
    "very_low": 120.0,
}


def hp_part_load_cop_continuous(
    cop_full_load: float,
    phi_full_load_kW: float,
    load_ratio: float,
    lr_cont_min: float,
    p_gen_aux_kW: float = 0.0,
) -> float:
    """EN 15316-4-2:2017 Eq.(18)-(21): COP at part load for a continuously-
    modulating (inverter) compressor operating between LR_cont,min and 1.

    cop_full_load/phi_full_load_kW are the COP/thermal capacity AT FULL
    LOAD for this step's inlet/outlet temperature (i.e. the already
    temperature-interpolated Eq.(3)-(9) "Path A" values). Returns
    cop_full_load unchanged at load_ratio>=1 (Eq.(21)'s own blend factor
    is 1 there) and when p_gen_aux_kW<=0 (Eq.(18)'s net-COP correction is
    then a no-op, matching Table B.9's f_gen,Pn,aux,int=0 default for
    electrically-driven heat pumps).
    """

    load_ratio = max(min(float(load_ratio), 1.0), 0.0)
    lr_cont_min = max(min(float(lr_cont_min), 1.0 - 1e-6), _KWH_EPS)
    cop_full_load = float(cop_full_load)
    phi_full_load_kW = float(phi_full_load_kW)
    if cop_full_load <= 0.0 or phi_full_load_kW <= 0.0:
        return cop_full_load
    if load_ratio >= 1.0 - 1e-9:
        return cop_full_load

    p_gen_aux_kW = max(float(p_gen_aux_kW), 0.0)
    p_gen_LR100 = phi_full_load_kW / cop_full_load  # Eq.(9)-derived: driving power at full load
    p_comp_LR100 = max(p_gen_LR100 - p_gen_aux_kW, _KWH_EPS)

    if p_gen_aux_kW > 0.0:
        # Eq.(18): net (auxiliary-free) COP at full load.
        cop_net_100 = (cop_full_load * phi_full_load_kW) / max(
            phi_full_load_kW - p_gen_aux_kW * cop_full_load, _KWH_EPS
        )
        # Eq.(19): minimum-continuous-load net correction factor. t_ci
        # cancels out of both numerator and denominator (it multiplies
        # every term uniformly), so it is not a parameter here.
        f_cont_min_net = (lr_cont_min * p_comp_LR100) / max(
            lr_cont_min * p_gen_LR100 - p_gen_aux_kW, _KWH_EPS
        )
    else:
        cop_net_100 = cop_full_load
        f_cont_min_net = 1.0

    # Eq.(21).
    cop_net_lr = cop_net_100 * (
        1.0 + (f_cont_min_net - 1.0) * (1.0 - load_ratio) / (1.0 - lr_cont_min)
    )
    cop_net_lr = max(cop_net_lr, _KWH_EPS)
    if p_gen_aux_kW <= 0.0:
        return cop_net_lr

    # Eq.(22)-(24): translate the net COP back to a gross COP that
    # includes P_gen,aux (the same correction Eq.(18) applied in reverse).
    phi_LR = phi_full_load_kW * load_ratio
    p_comp_LR = phi_LR / cop_net_lr
    e_gen_in_rate = p_comp_LR + p_gen_aux_kW
    return max(phi_LR / e_gen_in_rate, _KWH_EPS) if e_gen_in_rate > _KWH_EPS else cop_full_load


def hp_part_load_cop_on_off(
    cop_full_load: float,
    phi_full_load_kW: float,
    load_ratio: float,
    lr_cont_min: float,
    tau_eq_s: float = DEFAULT_TAU_EQ_S,
    tau_out_em_type_s: float = TAU_OUT_EM_TYPE_H_S_BY_CATEGORY["heavy_embedded"],
    p_gen_aux_kW: float = 0.0,
) -> float:
    """EN 15316-4-2:2017 Eq.(25)-(30): COP at part load for an ON/OFF
    compressor, or an inverter compressor cycling below its own
    LR_cont,min (load_ratio < lr_cont_min). Includes the cycling-inertia
    power penalty Eq.(28) (zero at load_ratio=0 and at load_ratio=
    lr_cont_min, maximum around the midpoint - the classic ON/OFF
    start/stop loss shape).

    Returns NaN at load_ratio<=0 (no energy delivered, so no meaningful
    COP - see Eq.(31)-(32) for the zero-demand step instead).
    """

    load_ratio = max(min(float(load_ratio), 1.0), 0.0)
    lr_cont_min = max(min(float(lr_cont_min), 1.0), _KWH_EPS)
    cop_full_load = float(cop_full_load)
    phi_full_load_kW = float(phi_full_load_kW)
    if load_ratio <= _KWH_EPS:
        return float("nan")
    if cop_full_load <= 0.0 or phi_full_load_kW <= 0.0:
        return cop_full_load

    p_gen_aux_kW = max(float(p_gen_aux_kW), 0.0)
    p_gen_LR100 = phi_full_load_kW / cop_full_load
    # Eq.(26).
    p_gen_LRcont_min = p_gen_LR100 * lr_cont_min
    # Eq.(25): linear ramp of compressor power from 0 (LR=0) up to
    # p_gen_LRcont_min (LR=LR_cont,min); flat at p_gen_LRcont_min beyond.
    if load_ratio < lr_cont_min:
        p_gen_comp_LR = p_gen_LRcont_min * (1.0 - (lr_cont_min - load_ratio) / lr_cont_min)
    else:
        p_gen_comp_LR = p_gen_LRcont_min
    p_gen_comp_LR = max(p_gen_comp_LR, 0.0)

    # Eq.(28): cycling-inertia power penalty, 0 outside (0, LR_cont,min).
    if _KWH_EPS < load_ratio < lr_cont_min and tau_out_em_type_s > 0.0:
        p_gen_comp_onoff = p_gen_LRcont_min * (
            tau_eq_s * load_ratio * (1.0 - load_ratio) / tau_out_em_type_s
        )
    else:
        p_gen_comp_onoff = 0.0

    # Eq.(29)/(30): COP_LR(t) = Q_out(t)/E_in(t) - t_H cancels between
    # Q_out(t)=Phi_LR x t_H and E_in(t)=(sum of powers) x t_H.
    phi_LR = phi_full_load_kW * load_ratio
    e_rate = p_gen_comp_LR + p_gen_aux_kW + p_gen_comp_onoff
    if e_rate <= _KWH_EPS:
        return cop_full_load
    return max(phi_LR / e_rate, _KWH_EPS)


@dataclass
class HeatPumpSimulationResult:
    """Container returned by :class:`HeatPumpSystemCalculator`.

    Attributes
    ----------
    bins:
        One row per outdoor-temperature bin with service loads, operating
        temperatures, product-map values and energy balances.
    summary:
        Aggregated annual or period totals and seasonal performance factors.
    inputs:
        Normalized configuration used by the calculator.
    """

    bins: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


def _build_pivot_arrays_hp(
    performance_map: "pd.DataFrame | None",
    column: str,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """Pre-build sorted pivot arrays for _bilinear_or_linear.
    Returns None if map is absent or pivot contains NaN (IDW fallback).
    """
    if performance_map is None:
        return None
    pivot = performance_map.pivot_table(
        index="source_temperature_C",
        columns="sink_temperature_C",
        values=column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        return None
    return (
        pivot.index.to_numpy(dtype=float),
        pivot.columns.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )

class HeatPumpSystemCalculator:
    """Case-specific heat-pump generation calculator.

    The implementation follows the EN 15316-4-2 detailed/bin-method sequence:

    1. Create temperature bins from hourly or sub-hourly outdoor/source data.
    2. Allocate heating and DHW thermal requirements to each bin.
    3. Interpolate heat-pump capacity and COP/EER for source/sink conditions.
    4. Check runtime, operating-temperature and backup limits.
    5. Calculate heat-pump input energy, auxiliary energy, losses and SPF.

    Parameters
    ----------
    input_data:
        Dictionary with product maps, operating temperatures and system options.
        See ``Readme.md`` for a compact example.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data: dict[str, Any] = dict(input_data or {})
        self._load_options()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate(self, data: pd.DataFrame) -> HeatPumpSimulationResult:
        """Run the bin-method simulation for a time-series input table."""

        prepared = self._prepare_timeseries(data)
        bins = self._simulate_bins(prepared)
        summary = self._summarize(bins)
        return HeatPumpSimulationResult(bins=bins, summary=summary, inputs=dict(self.input_data))

    def run_timeseries(self, data: pd.DataFrame) -> HeatPumpSimulationResult:
        """Alias for :meth:`simulate`, matching the package naming style."""

        return self.simulate(data)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _load_options(self) -> None:
        cfg = self.input_data

        self.bin_width_C = _positive_float(cfg.get("bin_width_C", 1.0), "bin_width_C")
        self.default_time_step_h = _positive_float(
            cfg.get("time_step_hours", 1.0), "time_step_hours"
        )
        self.demand_unit = str(cfg.get("demand_unit", "Wh")).lower()
        if self.demand_unit not in {"wh", "kwh"}:
            raise ValueError("demand_unit must be 'Wh' or 'kWh'.")

        self.heating_enabled = bool(cfg.get("heating_enabled", True))
        self.cooling_enabled = bool(cfg.get("cooling_enabled", True))
        self.dhw_enabled = bool(cfg.get("dhw_enabled", True))

        self.source_type = str(cfg.get("source_type", "air")).lower()
        self.ground_source_temperature_C = float(cfg.get("ground_source_temperature_C", 10.0))
        self.water_source_temperature_C = float(cfg.get("water_source_temperature_C", 10.0))
        self.exhaust_air_temperature_C = float(cfg.get("exhaust_air_temperature_C", 20.0))

        self.design_outdoor_temperature_C = float(cfg.get("design_outdoor_temperature_C", -7.0))
        self.heating_cutoff_temperature_C = float(cfg.get("heating_cutoff_temperature_C", 16.0))
        self.heating_sink_temp_at_design_C = float(
            cfg.get("heating_sink_temp_at_design_C", 45.0)
        )
        self.heating_sink_temp_at_cutoff_C = float(
            cfg.get("heating_sink_temp_at_cutoff_C", 30.0)
        )
        self.heating_deltaT_K = _positive_float(cfg.get("heating_deltaT_K", 5.0), "heating_deltaT_K")

        self.dhw_target_temperature_C = float(cfg.get("dhw_target_temperature_C", 60.0))
        self.dhw_cold_water_temperature_C = float(cfg.get("dhw_cold_water_temperature_C", 10.0))
        hp_limit_for_dhw_default = cfg.get("hp_operating_limit_C", 55.0)
        if hp_limit_for_dhw_default is None:
            hp_limit_for_dhw_default = self.dhw_target_temperature_C
        self.dhw_sink_temperature_C = float(
            cfg.get(
                "dhw_sink_temperature_C",
                min(self.dhw_target_temperature_C, float(hp_limit_for_dhw_default)),
            )
        )

        self.cooling_sink_temperature_C = float(cfg.get("cooling_sink_temperature_C", 7.0))
        self.cooling_max_source_temperature_C = _optional_float(
            cfg.get("cooling_max_source_temperature_C")
        )

        self.hp_operating_limit_C = _optional_float(cfg.get("hp_operating_limit_C", 55.0))
        self.hp_min_source_temperature_C = _optional_float(cfg.get("hp_min_source_temperature_C"))
        self.hp_cutoff_temperature_C = _optional_float(cfg.get("hp_cutoff_temperature_C"))
        self.bivalent_temperature_C = _optional_float(cfg.get("bivalent_temperature_C"))

        self.backup_mode = str(cfg.get("backup_mode", "parallel")).lower()
        if self.backup_mode not in {"none", "parallel", "alternative", "part_parallel"}:
            raise ValueError(
                "backup_mode must be one of 'none', 'parallel', 'alternative', 'part_parallel'."
            )

        self.heating_backup_efficiency = _positive_float(
            cfg.get("heating_backup_efficiency", cfg.get("backup_efficiency", 1.0)),
            "heating_backup_efficiency",
        )
        self.dhw_backup_efficiency = _positive_float(
            cfg.get("dhw_backup_efficiency", cfg.get("backup_efficiency", 1.0)),
            "dhw_backup_efficiency",
        )
        self.cooling_backup_eer = _optional_float(cfg.get("cooling_backup_eer"))

        # EN 15316-4-2:2017 Eq.(18)-(30) ("method_a_2017") is now the
        # DEFAULT part-load treatment, REPLACING the 2008 edition's bin-
        # ratio approach this module previously defaulted to ("simple" -
        # no part-load correction at all - was the old default). "en14825"
        # keeps this module's pre-existing, simpler multiplicative
        # degradation-coefficient correction (NOT clause 6.7.3/Annex D's
        # own Path B); "none" (alias "simple", for old configs) disables
        # any part-load correction, as before. Cooling is out of EN 15316-
        # 4-2:2017's scope entirely (it only covers heating/DHW) and always
        # keeps using "en14825"/"none" regardless of this setting.
        self.part_load_performance_method = str(
            cfg.get("part_load_performance_method", "method_a_2017")
        ).lower()
        if self.part_load_performance_method == "simple":
            self.part_load_performance_method = "none"
        if self.part_load_performance_method not in {"method_a_2017", "en14825", "none"}:
            raise ValueError(
                "part_load_performance_method must be 'method_a_2017', 'en14825' or 'none'."
            )
        self.part_load_unit_type = str(cfg.get("part_load_unit_type", "air-to-water")).lower()
        default_cd = cfg.get("part_load_degradation_coefficient", 0.9)
        self.heating_part_load_degradation_coefficient = _fraction(
            cfg.get("heating_part_load_degradation_coefficient", default_cd),
            "heating_part_load_degradation_coefficient",
        )
        self.dhw_part_load_degradation_coefficient = _fraction(
            cfg.get("dhw_part_load_degradation_coefficient", default_cd),
            "dhw_part_load_degradation_coefficient",
        )
        self.cooling_part_load_degradation_coefficient = _fraction(
            cfg.get("cooling_part_load_degradation_coefficient", default_cd),
            "cooling_part_load_degradation_coefficient",
        )

        # Table B.11: LR_cont,min DEFAULTS by compressor technology ("on_off"
        # -> 1.0, i.e. no continuous-modulation range at all; "inverter" ->
        # 0.2). An explicit *_part_load_minimum_capacity_ratio still
        # overrides this per service, exactly as before.
        self.compressor_technology = str(cfg.get("compressor_technology", "on_off")).lower()
        if self.compressor_technology not in DEFAULT_LR_CONT_MIN_BY_COMPRESSOR_TECH:
            raise ValueError(
                "compressor_technology must be one of "
                f"{sorted(DEFAULT_LR_CONT_MIN_BY_COMPRESSOR_TECH)}."
            )
        # The tech-based LR_cont,min default only applies to the NEW
        # method_a_2017 path; the pre-existing "en14825"/"none" methods
        # keep their OWN original default (0.0 - no part-load-ratio floor)
        # so they stay byte-for-byte backward compatible on their own.
        default_min_cr = cfg.get(
            "part_load_minimum_capacity_ratio",
            DEFAULT_LR_CONT_MIN_BY_COMPRESSOR_TECH[self.compressor_technology]
            if self.part_load_performance_method == "method_a_2017"
            else 0.0,
        )
        self.heating_part_load_minimum_capacity_ratio = _fraction(
            cfg.get("heating_part_load_minimum_capacity_ratio", default_min_cr),
            "heating_part_load_minimum_capacity_ratio",
        )
        self.dhw_part_load_minimum_capacity_ratio = _fraction(
            cfg.get("dhw_part_load_minimum_capacity_ratio", default_min_cr),
            "dhw_part_load_minimum_capacity_ratio",
        )
        self.cooling_part_load_minimum_capacity_ratio = _fraction(
            cfg.get("cooling_part_load_minimum_capacity_ratio", 0.0),
            "cooling_part_load_minimum_capacity_ratio",
        )

        # Table B.11: TAU_EQ (heat pump cycling inertia) and the Table 13
        # distribution time constant tau_out,em,type for Eq.(28)'s ON/OFF
        # cycling-inertia power penalty. External auxiliary power (already
        # tracked by this module as external_auxiliary_power_W, see below)
        # doubles as Eq.(18)-(30)'s P_gen,aux - the standard's own
        # definition ("auxiliary energy NOT included in the COP standard
        # testing") matches exactly what this field already represents.
        self.tau_eq_s = _positive_float(cfg.get("tau_eq_s", DEFAULT_TAU_EQ_S), "tau_eq_s")
        self.emitter_inertia_category = str(
            cfg.get("emitter_inertia_category", "heavy_embedded")
        ).lower()
        if self.emitter_inertia_category not in TAU_OUT_EM_TYPE_H_S_BY_CATEGORY:
            raise ValueError(
                "emitter_inertia_category must be one of "
                f"{sorted(TAU_OUT_EM_TYPE_H_S_BY_CATEGORY)}."
            )
        self.tau_out_em_type_H_s = _positive_float(
            cfg.get(
                "tau_out_em_type_H_s",
                TAU_OUT_EM_TYPE_H_S_BY_CATEGORY[self.emitter_inertia_category],
            ),
            "tau_out_em_type_H_s",
        )
        self.tau_out_em_type_W_s = _positive_float(
            cfg.get("tau_out_em_type_W_s", DEFAULT_TAU_OUT_EM_TYPE_W_S), "tau_out_em_type_W_s"
        )

        self.combined_operation = str(cfg.get("combined_operation", "alternative")).lower()
        if self.combined_operation not in {"alternative", "independent"}:
            raise ValueError("combined_operation must be 'alternative' or 'independent'.")
        self.dhw_priority = bool(cfg.get("dhw_priority", True))
        self.utility_cutoff_hours_per_day = max(
            0.0, min(float(cfg.get("utility_cutoff_hours_per_day", 0.0)), 24.0)
        )

        self.external_auxiliary_power_W = max(float(cfg.get("external_auxiliary_power_W", 0.0)), 0.0)
        self.standby_power_W = max(float(cfg.get("standby_power_W", 0.0)), 0.0)
        self.auxiliary_loss_to_ambient_fraction = _fraction(
            cfg.get("auxiliary_loss_to_ambient_fraction", 1.0),
            "auxiliary_loss_to_ambient_fraction",
        )
        self.auxiliary_recovered_fraction = _fraction(
            cfg.get("auxiliary_recovered_fraction", 0.0),
            "auxiliary_recovered_fraction",
        )
        self.auxiliary_location_b_factor = _fraction(
            cfg.get("auxiliary_location_b_factor", 0.0), "auxiliary_location_b_factor"
        )

        self.heating_storage_loss_kWh_per_day = max(
            float(cfg.get("heating_storage_loss_kWh_per_day", 0.0)), 0.0
        )
        self.dhw_storage_loss_kWh_per_day = max(
            float(cfg.get("dhw_storage_loss_kWh_per_day", 0.0)), 0.0
        )
        self.storage_test_deltaT_K = _positive_float(
            cfg.get("storage_test_deltaT_K", 45.0), "storage_test_deltaT_K"
        )
        self.storage_ambient_temperature_C = float(
            cfg.get("storage_ambient_temperature_C", 20.0)
        )
        self.storage_location_b_factor = _fraction(
            cfg.get("storage_location_b_factor", 0.0), "storage_location_b_factor"
        )

        self.heating_performance_map = _coerce_performance_map(
            cfg.get("heating_performance_map", cfg.get("performance_map")),
            performance_column="cop",
            service_name="heating",
        )
        self.dhw_performance_map = _coerce_performance_map(
            cfg.get("dhw_performance_map", cfg.get("heating_performance_map", cfg.get("performance_map"))),
            performance_column="cop",
            service_name="dhw",
        )
        self.cooling_performance_map = _coerce_performance_map(
            cfg.get("cooling_performance_map"),
            performance_column="eer",
            service_name="cooling",
            required=False,
        )
        # --- Commit-2B: pre-build pivot arrays (one pivot_table call per map/column) ---
        self._h_perf_pivot = _build_pivot_arrays_hp(self.heating_performance_map, "cop")
        self._h_cap_pivot  = _build_pivot_arrays_hp(self.heating_performance_map, "capacity_kW")
        self._dhw_perf_pivot = _build_pivot_arrays_hp(self.dhw_performance_map, "cop")
        self._dhw_cap_pivot  = _build_pivot_arrays_hp(self.dhw_performance_map, "capacity_kW")
        self._c_perf_pivot = _build_pivot_arrays_hp(self.cooling_performance_map, "eer")
        self._c_cap_pivot  = _build_pivot_arrays_hp(self.cooling_performance_map, "capacity_kW")
        # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Time-series preparation
    # ------------------------------------------------------------------
    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        df = data.copy()
        out = pd.DataFrame(index=df.index)

        out["hours"] = self._time_step_hours(df)
        # EN 15316-4-2:2017 Eq.(31)-(32): the generator sub-system only
        # draws P_gen,aux (here, standby_power_W) while "powered" - its own
        # OPERATING SEASON - and nothing at all outside it. Defaults to
        # always-powered (1.0, the pre-existing behaviour, unchanged) when
        # no "powered"/"operating_season" column is supplied; a caller can
        # mark specific steps as outside the season (0) to get Eq.(32)'s
        # zero-energy treatment there instead of Eq.(31)'s standby draw.
        out["powered"] = _series_from_aliases(
            df, ["powered", "hp_powered", "operating_season", "HP_powered"], default=1.0
        ).astype(float).clip(0.0, 1.0)
        out["T_ext_C"] = _series_from_aliases(
            df,
            ["T_ext", "theta_ext", "outdoor_temperature_C", "T_external_C"],
            default=None,
        )
        if out["T_ext_C"].isna().any():
            raise ValueError("Outdoor temperature is required. Provide 'T_ext' or 'theta_ext'.")
        out = out.assign(T_ext_C=out["T_ext_C"].astype(float))

        q_hc = _series_from_aliases(df, ["Q_HC"], default=0.0)
        out["Q_H_gen_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_H_kWh", "QH_kWh", "heating_kWh", "space_heating_kWh"],
            raw_aliases=["Q_H", "Q_h", "Heating_needs"],
            fallback=np.maximum(q_hc.astype(float), 0.0),
        )
        out["Q_C_gen_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=["Q_C_kWh", "QC_kWh", "cooling_kWh", "space_cooling_kWh"],
            raw_aliases=["Q_C", "Cooling_needs"],
            fallback=np.maximum(-q_hc.astype(float), 0.0),
        )
        out["Q_W_gen_out_kWh"] = self._demand_from_columns(
            df,
            kwh_aliases=[
                "Q_W_kWh",
                "Q_DHW_kWh",
                "DHW_kWh",
                "dhw_kWh",
                "domestic_hot_water_kWh",
            ],
            raw_aliases=["Q_W", "Q_DHW"],
            fallback=0.0,
            raw_default_unit="kwh",
        )

        if not self.heating_enabled:
            out["Q_H_gen_out_kWh"] = 0.0
        if not self.cooling_enabled:
            out["Q_C_gen_out_kWh"] = 0.0
        if not self.dhw_enabled:
            out["Q_W_gen_out_kWh"] = 0.0

        out = out.assign(
            T_H_source_C=self._source_temperature(df, out["T_ext_C"], "H"),
            T_W_source_C=self._source_temperature(df, out["T_ext_C"], "W"),
            T_C_source_C=self._source_temperature(df, out["T_ext_C"], "C"),
            T_H_sink_C=_series_from_aliases(
                df,
                ["T_H_sink_C", "theta_H_supply_C", "heating_supply_temperature_C"],
                default=np.nan,
            ).astype(float),
        )
        missing_h_sink = out["T_H_sink_C"].isna()
        out.loc[missing_h_sink, "T_H_sink_C"] = self._heating_sink_temperature(
            out.loc[missing_h_sink, "T_ext_C"]
        )

        out["T_H_return_C"] = _series_from_aliases(
            df,
            ["T_H_return_C", "theta_H_return_C", "heating_return_temperature_C"],
            default=np.nan,
        ).astype(float)
        missing_h_return = out["T_H_return_C"].isna()
        out.loc[missing_h_return, "T_H_return_C"] = (
            out.loc[missing_h_return, "T_H_sink_C"] - self.heating_deltaT_K
        )

        out["T_W_sink_C"] = _series_from_aliases(
            df,
            ["T_W_sink_C", "dhw_sink_temperature_C", "dhw_tank_temperature_C"],
            default=self.dhw_sink_temperature_C,
        ).astype(float)
        out["T_W_target_C"] = _series_from_aliases(
            df,
            ["T_W_target_C", "dhw_target_temperature_C", "dhw_delivery_temperature_C"],
            default=self.dhw_target_temperature_C,
        ).astype(float)
        out["T_W_cold_C"] = _series_from_aliases(
            df,
            ["T_W_cold_C", "dhw_cold_water_temperature_C", "cold_water_temperature_C"],
            default=self.dhw_cold_water_temperature_C,
        ).astype(float)

        out["T_C_sink_C"] = _series_from_aliases(
            df,
            ["T_C_sink_C", "cooling_sink_temperature_C", "chilled_water_temperature_C"],
            default=self.cooling_sink_temperature_C,
        ).astype(float)

        for col in ["Q_H_gen_out_kWh", "Q_C_gen_out_kWh", "Q_W_gen_out_kWh"]:
            out = out.assign(**{col: out[col].fillna(0.0).clip(lower=0.0)})

        return out

    def _time_step_hours(self, df: pd.DataFrame) -> pd.Series:
        step = _series_from_aliases(df, ["time_step_hours", "dt_h"], default=np.nan)
        if not step.isna().all():
            return step.astype(float).clip(lower=0.0)

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            diffs = df.index.to_series().diff().dt.total_seconds().dropna() / 3600.0
            if not diffs.empty and np.isfinite(diffs.median()) and diffs.median() > 0:
                return pd.Series(float(diffs.median()), index=df.index)

        return pd.Series(self.default_time_step_h, index=df.index)

    def _demand_from_columns(
        self,
        df: pd.DataFrame,
        kwh_aliases: list[str],
        raw_aliases: list[str],
        fallback: float | pd.Series,
        raw_default_unit: str | None = None,
    ) -> pd.Series:
        kwh = _series_from_aliases(df, kwh_aliases, default=None)
        if kwh is not None:
            return kwh.astype(float)

        raw = _series_from_aliases(df, raw_aliases, default=None)
        if raw is None:
            if isinstance(fallback, pd.Series):
                raw = fallback.astype(float)
                unit = self.demand_unit
            else:
                return pd.Series(float(fallback), index=df.index)
        else:
            raw = raw.astype(float)
            unit = raw_default_unit or self.demand_unit

        if unit == "wh":
            return raw / 1000.0
        return raw

    def _source_temperature(
        self, df: pd.DataFrame, outdoor_temperature: pd.Series, service_prefix: str
    ) -> pd.Series:
        service_aliases = {
            "H": ["T_H_source_C", "heating_source_temperature_C"],
            "W": ["T_W_source_C", "dhw_source_temperature_C"],
            "C": ["T_C_source_C", "cooling_source_temperature_C"],
        }[service_prefix]
        source = _series_from_aliases(
            df, service_aliases + ["T_source_C", "source_temperature_C"], default=None
        )
        if source is not None:
            return source.astype(float)

        if self.source_type in {"air", "outdoor_air", "outside_air"}:
            return outdoor_temperature.astype(float)
        if self.source_type in {"ground", "brine", "ground_brine"}:
            return pd.Series(self.ground_source_temperature_C, index=df.index)
        if self.source_type in {"water", "groundwater", "surface_water"}:
            return pd.Series(self.water_source_temperature_C, index=df.index)
        if self.source_type in {"exhaust_air", "exhaust"}:
            return pd.Series(self.exhaust_air_temperature_C, index=df.index)
        raise ValueError(
            "source_type must be air, ground, water or exhaust_air when no source "
            "temperature column is provided."
        )

    def _heating_sink_temperature(self, outdoor_temperature: pd.Series) -> pd.Series:
        if self.heating_cutoff_temperature_C == self.design_outdoor_temperature_C:
            return pd.Series(self.heating_sink_temp_at_cutoff_C, index=outdoor_temperature.index)

        slope = (
            self.heating_sink_temp_at_cutoff_C - self.heating_sink_temp_at_design_C
        ) / (self.heating_cutoff_temperature_C - self.design_outdoor_temperature_C)
        values = self.heating_sink_temp_at_design_C + slope * (
            outdoor_temperature.astype(float) - self.design_outdoor_temperature_C
        )
        lo = min(self.heating_sink_temp_at_design_C, self.heating_sink_temp_at_cutoff_C)
        hi = max(self.heating_sink_temp_at_design_C, self.heating_sink_temp_at_cutoff_C)
        return values.clip(lower=lo, upper=hi)

    # ------------------------------------------------------------------
    # Bin simulation
    # ------------------------------------------------------------------
    def _simulate_bins(self, prepared: pd.DataFrame) -> pd.DataFrame:
        df = prepared.copy()
        bin_low = np.floor(df["T_ext_C"].astype(float) / self.bin_width_C) * self.bin_width_C
        df = df.assign(
            bin_lower_C=bin_low,
            bin_upper_C=bin_low + self.bin_width_C,
            bin_center_C=bin_low + self.bin_width_C / 2.0,
        )

        rows: list[dict[str, float]] = []
        for (lower, upper, center), group in df.groupby(
            ["bin_lower_C", "bin_upper_C", "bin_center_C"], sort=True
        ):
            row = self._simulate_one_bin(float(lower), float(upper), float(center), group)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        bins = pd.DataFrame(rows).sort_values("bin_lower_C").reset_index(drop=True)
        return bins

    def _simulate_one_bin(
        self, lower: float, upper: float, center: float, group: pd.DataFrame
    ) -> dict[str, float]:
        hours = float(group["hours"].sum())
        effective_hours = hours * (24.0 - self.utility_cutoff_hours_per_day) / 24.0
        effective_hours = max(effective_hours, 0.0)

        row: dict[str, float] = {
            "bin_lower_C": lower,
            "bin_upper_C": upper,
            "bin_center_C": center,
            "hours": hours,
            "effective_hours": effective_hours,
            "T_ext_mean_C": _weighted_mean(group["T_ext_C"], group["hours"]),
        }

        heating = self._service_load_and_performance(group, "H", effective_hours)
        dhw = self._service_load_and_performance(group, "W", effective_hours)
        cooling = self._service_load_and_performance(group, "C", effective_hours)

        self._allocate_heating_and_dhw_runtime(heating, dhw, effective_hours)
        self._allocate_cooling_runtime(cooling, effective_hours)

        row.update(_prefix_dict(heating, "H"))
        row.update(_prefix_dict(dhw, "W"))
        row.update(_prefix_dict(cooling, "C"))

        # Eq.(31)-(32): standby_power_W only applies to hours the generator
        # is POWERED (within its operating season) - hours outside it (the
        # "powered" column, default 1.0/always, see _prepare_timeseries())
        # draw nothing at all, matching Eq.(32)'s "not powered" case rather
        # than Eq.(31)'s "powered, idle" one.
        powered_hours = float((group["hours"] * group["powered"]).sum())
        row["powered_hours"] = powered_hours

        hp_active_hours_hw = heating["hp_runtime_h"] + dhw["hp_runtime_h"]
        standby_hours_hw = max(powered_hours - hp_active_hours_hw, 0.0)
        aux_active_hw = self.external_auxiliary_power_W * hp_active_hours_hw / 1000.0
        aux_standby_hw = self.standby_power_W * standby_hours_hw / 1000.0
        aux_hw = aux_active_hw + aux_standby_hw

        hp_active_hours_c = cooling["hp_runtime_h"]
        standby_hours_c = max(powered_hours - hp_active_hours_c, 0.0)
        aux_active_c = self.external_auxiliary_power_W * hp_active_hours_c / 1000.0
        aux_standby_c = self.standby_power_W * standby_hours_c / 1000.0
        aux_c = aux_active_c + aux_standby_c

        aux_loss_hw = aux_hw * self.auxiliary_loss_to_ambient_fraction
        aux_loss_c = aux_c * self.auxiliary_loss_to_ambient_fraction
        aux_recovered_hw = aux_hw * self.auxiliary_recovered_fraction
        aux_recovered_c = aux_c * self.auxiliary_recovered_fraction
        aux_recoverable_hw = aux_loss_hw * (1.0 - self.auxiliary_location_b_factor)
        aux_recoverable_c = aux_loss_c * (1.0 - self.auxiliary_location_b_factor)

        storage_loss_hw = heating["Q_gen_ls_kWh"] + dhw["Q_gen_ls_kWh"]
        storage_recoverable_hw = storage_loss_hw * (1.0 - self.storage_location_b_factor)

        row.update(
            {
                "W_HW_gen_aux_kWh": aux_hw,
                "W_HW_gen_aux_active_kWh": aux_active_hw,
                "W_HW_gen_aux_standby_kWh": aux_standby_hw,
                "Q_HW_gen_aux_recovered_kWh": aux_recovered_hw,
                "Q_HW_gen_aux_ls_kWh": aux_loss_hw,
                "Q_HW_gen_aux_ls_rbl_kWh": aux_recoverable_hw,
                "Q_HW_gen_ls_tot_kWh": storage_loss_hw + aux_loss_hw,
                "Q_HW_gen_ls_rbl_tot_kWh": storage_recoverable_hw + aux_recoverable_hw,
                "W_C_gen_aux_kWh": aux_c,
                "W_C_gen_aux_active_kWh": aux_active_c,
                "W_C_gen_aux_standby_kWh": aux_standby_c,
                "Q_C_gen_aux_recovered_kWh": aux_recovered_c,
                "Q_C_gen_aux_ls_kWh": aux_loss_c,
                "Q_C_gen_aux_ls_rbl_kWh": aux_recoverable_c,
            }
        )
        return row

    def _service_load_and_performance(
        self, group: pd.DataFrame, service: str, effective_hours: float
    ) -> dict[str, float]:
        if service == "H":
            demand_col = "Q_H_gen_out_kWh"
            source_col = "T_H_source_C"
            sink_col = "T_H_sink_C"
            return_col = "T_H_return_C"
            perf_map = self.heating_performance_map
            perf_col = "cop"
        elif service == "W":
            demand_col = "Q_W_gen_out_kWh"
            source_col = "T_W_source_C"
            sink_col = "T_W_sink_C"
            return_col = "T_W_cold_C"
            perf_map = self.dhw_performance_map
            perf_col = "cop"
        else:
            demand_col = "Q_C_gen_out_kWh"
            source_col = "T_C_source_C"
            sink_col = "T_C_sink_C"
            return_col = "T_C_sink_C"
            perf_map = self.cooling_performance_map
            perf_col = "eer"

        demand = float(group[demand_col].sum())
        load_hours = float(group.loc[group[demand_col] > _KWH_EPS, "hours"].sum())
        if service == "H" and demand > _KWH_EPS:
            storage_loss = self._storage_loss(
                group,
                loss_kWh_per_day=self.heating_storage_loss_kWh_per_day,
                temperature_col=sink_col,
            )
        elif service == "W" and self.dhw_enabled:
            storage_loss = self._storage_loss(
                group,
                loss_kWh_per_day=self.dhw_storage_loss_kWh_per_day,
                temperature_col=sink_col,
            )
        else:
            storage_loss = 0.0

        source = _weighted_mean(group[source_col], group["hours"])
        sink = _weighted_mean(group[sink_col], group[demand_col] + _KWH_EPS)
        return_or_cold = _weighted_mean(group[return_col], group[demand_col] + _KWH_EPS)
        load_with_losses = demand + storage_loss

        op_limit_fraction = self._operating_limit_backup_fraction(
            service=service,
            sink_temperature_C=sink,
            return_or_cold_temperature_C=return_or_cold,
            group=group,
        )
        backup_operating_kWh = load_with_losses * op_limit_fraction
        unmet_operating_kWh = 0.0
        if service in {"H", "W"} and self.backup_mode == "none":
            unmet_operating_kWh = backup_operating_kWh
            backup_operating_kWh = 0.0
        hp_requested_kWh = max(load_with_losses - backup_operating_kWh - unmet_operating_kWh, 0.0)

        available = self._hp_available(service, source)
        performance = np.nan
        capacity_kW = 0.0
        if hp_requested_kWh > _KWH_EPS and available:
            if perf_map is None:
                raise ValueError(
                    f"{service} demand is present, but no performance map was provided."
                )
            # Use pre-built pivot grids when available (avoids per-call pivot_table)
            if service == "H":
                _perf_grid = self._h_perf_pivot
                _cap_grid  = self._h_cap_pivot
            elif service == "W":
                _perf_grid = self._dhw_perf_pivot
                _cap_grid  = self._dhw_cap_pivot
            else:
                _perf_grid = self._c_perf_pivot
                _cap_grid  = self._c_cap_pivot
            if _perf_grid is not None:
                performance = _bilinear_or_linear(*_perf_grid, source, sink)
                capacity_kW = _bilinear_or_linear(*_cap_grid, source, sink)
            else:
                performance = _interpolate_performance(perf_map, source, sink, perf_col)
                capacity_kW = _interpolate_performance(perf_map, source, sink, "capacity_kW")
            performance = max(float(performance), _KWH_EPS)
            capacity_kW = max(float(capacity_kW), 0.0)

        required_runtime_h = (
            hp_requested_kWh / capacity_kW if capacity_kW > _KWH_EPS else np.inf
        )
        if hp_requested_kWh <= _KWH_EPS:
            required_runtime_h = 0.0
        if not available:
            required_runtime_h = np.inf if hp_requested_kWh > _KWH_EPS else 0.0

        return {
            "service": service,
            "Q_gen_out_kWh": demand,
            "load_hours": load_hours,
            "Q_gen_ls_kWh": storage_loss,
            "Q_total_required_kWh": load_with_losses,
            "Q_backup_operating_kWh": backup_operating_kWh,
            "Q_hp_requested_kWh": hp_requested_kWh,
            "T_source_C": source,
            "T_sink_C": sink,
            "T_return_or_cold_C": return_or_cold,
            "performance": performance,
            "performance_full_load": performance,
            "capacity_kW": capacity_kW,
            "required_runtime_h": required_runtime_h,
            "part_load_ratio": 0.0,
            "part_load_ratio_for_performance": 0.0,
            "part_load_factor": 1.0,
            "part_load_degradation_coefficient": self._part_load_degradation_coefficient(service),
            "hp_available": 1.0 if available else 0.0,
            "hp_runtime_h": 0.0,
            "Q_hp_out_kWh": 0.0,
            "Q_backup_capacity_kWh": 0.0,
            "Q_backup_out_kWh": backup_operating_kWh,
            "Q_unmet_operating_kWh": unmet_operating_kWh,
            "Q_unmet_kWh": unmet_operating_kWh,
            "E_hp_in_kWh": 0.0,
            "E_backup_in_kWh": 0.0,
            "Q_environment_in_kWh": 0.0,
            "Q_rejected_kWh": 0.0,
        }

    def _allocate_heating_and_dhw_runtime(
        self, heating: dict[str, float], dhw: dict[str, float], effective_hours: float
    ) -> None:
        if self.combined_operation == "independent":
            self._allocate_single_heating_service(heating, effective_hours, "H")
            self._allocate_single_heating_service(dhw, effective_hours, "W")
            self._cap_combined_independent_runtime(heating, dhw, effective_hours)
            return

        if self.dhw_priority:
            first = dhw
            second = heating
        else:
            first = heating
            second = dhw

        remaining = max(effective_hours, 0.0)
        self._allocate_single_heating_service(first, remaining, "W" if self.dhw_priority else "H")
        remaining = max(remaining - first["hp_runtime_h"], 0.0)
        self._allocate_single_heating_service(second, remaining, "H" if self.dhw_priority else "W")

    def _cap_combined_independent_runtime(
        self, heating: dict[str, float], dhw: dict[str, float], effective_hours: float
    ) -> None:
        """combined_operation="independent" allocates heating and DHW each
        against the FULL step duration independently - EN 15316-4-2:2017
        clause 5.9 says genuine SIMULTANEOUS operation needs its OWN
        dedicated test data (capacity/COP measured for the combined mode),
        which this module does not model. Short of that, a SINGLE
        compressor still cannot deliver more than effective_hours worth of
        COMBINED runtime in one step: if the two services' independently-
        allocated runtimes add up to more than that, scale BOTH back
        proportionally - shifting the shortfall to back-up/unmet, the same
        way a single capacity-limited service already does - rather than
        silently double-counting output the unit could not physically
        have delivered in this step."""

        total_runtime = heating["hp_runtime_h"] + dhw["hp_runtime_h"]
        if total_runtime <= effective_hours + _KWH_EPS or total_runtime <= _KWH_EPS:
            return
        scale = effective_hours / total_runtime
        for service_result, backup_eff in (
            (heating, self.heating_backup_efficiency),
            (dhw, self.dhw_backup_efficiency),
        ):
            hp_out_before = service_result["Q_hp_out_kWh"]
            if hp_out_before <= _KWH_EPS:
                continue
            hp_out_after = hp_out_before * scale
            shortfall = hp_out_before - hp_out_after
            service_result["hp_runtime_h"] *= scale
            service_result["Q_hp_out_kWh"] = hp_out_after
            part_load_hours = self._part_load_reference_hours(
                service_result, effective_hours, service_result["hp_runtime_h"]
            )
            service_result["part_load_ratio"] = _ratio(
                hp_out_after, service_result["capacity_kW"] * part_load_hours
            )
            if self.backup_mode == "none":
                service_result["Q_unmet_kWh"] = service_result.get("Q_unmet_kWh", 0.0) + shortfall
            else:
                service_result["Q_backup_capacity_kWh"] = (
                    service_result.get("Q_backup_capacity_kWh", 0.0) + shortfall
                )
                service_result["Q_backup_out_kWh"] += shortfall
            self._finalize_heating_service(service_result, backup_eff)

    def _allocate_single_heating_service(
        self, service_result: dict[str, float], available_hours: float, service: str
    ) -> None:
        requested = service_result["Q_hp_requested_kWh"]
        capacity = service_result["capacity_kW"]
        backup_eff = (
            self.heating_backup_efficiency if service == "H" else self.dhw_backup_efficiency
        )

        if requested <= _KWH_EPS:
            self._finalize_heating_service(service_result, backup_eff)
            return

        if service_result["hp_available"] < 0.5 or capacity <= _KWH_EPS:
            hp_out = 0.0
            runtime = 0.0
        else:
            hp_out = min(requested, capacity * max(available_hours, 0.0))
            runtime = hp_out / capacity if capacity > _KWH_EPS else 0.0

        missing_capacity = max(requested - hp_out, 0.0)
        if self.backup_mode == "none":
            backup_capacity = 0.0
            unmet = missing_capacity
        else:
            backup_capacity = missing_capacity
            unmet = 0.0

        initial_unmet = service_result.get("Q_unmet_operating_kWh", 0.0)
        service_result["hp_runtime_h"] = runtime
        service_result["Q_hp_out_kWh"] = hp_out
        part_load_hours = self._part_load_reference_hours(service_result, available_hours, runtime)
        service_result["part_load_ratio"] = _ratio(
            hp_out,
            capacity * part_load_hours,
        )
        service_result["Q_backup_capacity_kWh"] = backup_capacity
        service_result["Q_backup_out_kWh"] += backup_capacity
        service_result["Q_unmet_kWh"] = initial_unmet + unmet
        self._finalize_heating_service(service_result, backup_eff)

    def _finalize_heating_service(
        self, service_result: dict[str, float], backup_efficiency: float
    ) -> None:
        self._apply_part_load_performance(service_result)
        perf = service_result["performance"]
        hp_out = service_result["Q_hp_out_kWh"]
        service_result["E_hp_in_kWh"] = hp_out / perf if hp_out > _KWH_EPS and perf > 0 else 0.0
        service_result["E_backup_in_kWh"] = (
            service_result["Q_backup_out_kWh"] / backup_efficiency
            if service_result["Q_backup_out_kWh"] > _KWH_EPS
            else 0.0
        )
        service_result["Q_environment_in_kWh"] = max(
            service_result["Q_hp_out_kWh"] - service_result["E_hp_in_kWh"], 0.0
        )

    def _allocate_cooling_runtime(
        self, cooling: dict[str, float], effective_hours: float
    ) -> None:
        requested = cooling["Q_hp_requested_kWh"]
        capacity = cooling["capacity_kW"]
        if requested <= _KWH_EPS:
            return

        if cooling["hp_available"] < 0.5 or capacity <= _KWH_EPS:
            hp_out = 0.0
            runtime = 0.0
        else:
            hp_out = min(requested, capacity * max(effective_hours, 0.0))
            runtime = hp_out / capacity if capacity > _KWH_EPS else 0.0

        missing_capacity = max(requested - hp_out, 0.0)
        if self.cooling_backup_eer is None:
            backup_capacity = 0.0
            unmet = missing_capacity
        else:
            backup_capacity = missing_capacity
            unmet = 0.0

        cooling["hp_runtime_h"] = runtime
        cooling["Q_hp_out_kWh"] = hp_out
        part_load_hours = self._part_load_reference_hours(cooling, effective_hours, runtime)
        cooling["part_load_ratio"] = _ratio(
            hp_out,
            capacity * part_load_hours,
        )
        cooling["Q_backup_capacity_kWh"] = backup_capacity
        cooling["Q_backup_out_kWh"] += backup_capacity
        cooling["Q_unmet_kWh"] = unmet
        self._apply_part_load_performance(cooling)
        perf = cooling["performance"]
        cooling["E_hp_in_kWh"] = hp_out / perf if hp_out > _KWH_EPS and perf > 0 else 0.0
        cooling["E_backup_in_kWh"] = (
            backup_capacity / self.cooling_backup_eer
            if self.cooling_backup_eer and backup_capacity > _KWH_EPS
            else 0.0
        )
        cooling["Q_rejected_kWh"] = cooling["Q_hp_out_kWh"] + cooling["E_hp_in_kWh"]

    def _part_load_reference_hours(
        self,
        service_result: dict[str, float],
        available_hours: float,
        runtime: float,
    ) -> float:
        demand_hours = max(service_result.get("load_hours", 0.0), 0.0)
        if demand_hours > _KWH_EPS:
            return max(min(demand_hours, max(available_hours, 0.0)), runtime, _KWH_EPS)
        return max(runtime, _KWH_EPS)

    def _part_load_degradation_coefficient(self, service: str) -> float:
        if service == "C":
            return self.cooling_part_load_degradation_coefficient
        if service == "W":
            return self.dhw_part_load_degradation_coefficient
        return self.heating_part_load_degradation_coefficient

    def _apply_part_load_performance(self, service_result: dict[str, float]) -> None:
        full_load_performance = service_result.get("performance_full_load", np.nan)
        service_result["performance"] = full_load_performance
        service_result["part_load_factor"] = 1.0
        service_result["part_load_ratio_for_performance"] = service_result.get(
            "part_load_ratio", 0.0
        )
        if not np.isfinite(full_load_performance) or full_load_performance <= 0:
            return
        if service_result.get("Q_hp_out_kWh", 0.0) <= _KWH_EPS:
            return

        service = service_result.get("service")
        # EN 15316-4-2:2017 covers heating/DHW only (clause 1) - cooling
        # always falls back to the pre-existing "en14825"/"none" treatment
        # regardless of part_load_performance_method.
        method = self.part_load_performance_method if service != "C" else (
            "en14825" if self.part_load_performance_method == "method_a_2017" else self.part_load_performance_method
        )
        if method == "none":
            return

        part_load_ratio = service_result.get("part_load_ratio", 1.0)
        minimum_ratio = self._part_load_minimum_capacity_ratio(service_result)

        if method == "method_a_2017":
            # Eq.(18)-(30): continuous (inverter) correction at/above
            # LR_cont,min, ON/OFF cycling-inertia correction below it.
            capacity_kW = service_result.get("capacity_kW", 0.0)
            p_gen_aux_kW = self.external_auxiliary_power_W / 1000.0
            if part_load_ratio >= minimum_ratio - 1e-9:
                new_performance = hp_part_load_cop_continuous(
                    cop_full_load=full_load_performance,
                    phi_full_load_kW=capacity_kW,
                    load_ratio=part_load_ratio,
                    lr_cont_min=minimum_ratio,
                    p_gen_aux_kW=p_gen_aux_kW,
                )
            else:
                tau_out_em_type_s = (
                    self.tau_out_em_type_H_s if service == "H" else self.tau_out_em_type_W_s
                )
                new_performance = hp_part_load_cop_on_off(
                    cop_full_load=full_load_performance,
                    phi_full_load_kW=capacity_kW,
                    load_ratio=part_load_ratio,
                    lr_cont_min=minimum_ratio,
                    tau_eq_s=self.tau_eq_s,
                    tau_out_em_type_s=tau_out_em_type_s,
                    p_gen_aux_kW=p_gen_aux_kW,
                )
            if np.isfinite(new_performance) and new_performance > 0:
                service_result["part_load_ratio_for_performance"] = max(part_load_ratio, minimum_ratio)
                service_result["part_load_factor"] = _ratio(new_performance, full_load_performance)
                service_result["performance"] = max(new_performance, _KWH_EPS)
            return

        # method == "en14825": pre-existing simplified multiplicative
        # degradation-coefficient correction (NOT clause 6.7.3/Annex D).
        performance_ratio = max(part_load_ratio, minimum_ratio)
        coefficient = service_result.get("part_load_degradation_coefficient", 0.9)
        factor = en14825_part_load_factor(
            capacity_ratio=performance_ratio,
            degradation_coefficient=coefficient,
            unit_type=self.part_load_unit_type,
        )
        service_result["part_load_ratio_for_performance"] = performance_ratio
        service_result["part_load_factor"] = factor
        service_result["performance"] = max(full_load_performance * factor, _KWH_EPS)

    def _part_load_minimum_capacity_ratio(self, service_result: dict[str, float]) -> float:
        service = service_result.get("service")
        if service == "C":
            return self.cooling_part_load_minimum_capacity_ratio
        if service == "W":
            return self.dhw_part_load_minimum_capacity_ratio
        return self.heating_part_load_minimum_capacity_ratio

    def _storage_loss(
        self, group: pd.DataFrame, loss_kWh_per_day: float, temperature_col: str
    ) -> float:
        if loss_kWh_per_day <= 0:
            return 0.0
        hours = float(group["hours"].sum())
        avg_temp = _weighted_mean(group[temperature_col], group["hours"])
        delta = max(avg_temp - self.storage_ambient_temperature_C, 0.0)
        return loss_kWh_per_day * (delta / self.storage_test_deltaT_K) * (hours / 24.0)

    def _operating_limit_backup_fraction(
        self,
        service: str,
        sink_temperature_C: float,
        return_or_cold_temperature_C: float,
        group: pd.DataFrame,
    ) -> float:
        if self.hp_operating_limit_C is None:
            return 0.0

        limit = self.hp_operating_limit_C
        if service == "W":
            target = _weighted_mean(group["T_W_target_C"], group["Q_W_gen_out_kWh"] + _KWH_EPS)
            cold = _weighted_mean(group["T_W_cold_C"], group["Q_W_gen_out_kWh"] + _KWH_EPS)
            if target <= limit:
                return 0.0
            return _fraction((target - limit) / max(target - cold, _KWH_EPS), "dhw limit")

        if service == "H" and sink_temperature_C > limit:
            return _fraction(
                (sink_temperature_C - limit)
                / max(sink_temperature_C - return_or_cold_temperature_C, _KWH_EPS),
                "heating limit",
            )

        return 0.0

    def _hp_available(self, service: str, source_temperature_C: float) -> bool:
        if service == "C":
            if self.cooling_max_source_temperature_C is not None:
                return source_temperature_C <= self.cooling_max_source_temperature_C
            return True

        if self.hp_min_source_temperature_C is not None:
            if source_temperature_C < self.hp_min_source_temperature_C:
                return False

        if self.backup_mode == "alternative" and self.bivalent_temperature_C is not None:
            return source_temperature_C >= self.bivalent_temperature_C

        if self.backup_mode == "part_parallel" and self.hp_cutoff_temperature_C is not None:
            return source_temperature_C >= self.hp_cutoff_temperature_C

        return True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _summarize(self, bins: pd.DataFrame) -> dict[str, float]:
        if bins.empty:
            return {}

        def s(col: str) -> float:
            return float(bins[col].sum()) if col in bins else 0.0

        q_h = s("H_Q_gen_out_kWh")
        q_w = s("W_Q_gen_out_kWh")
        q_c = s("C_Q_gen_out_kWh")

        e_h_hp = s("H_E_hp_in_kWh")
        e_w_hp = s("W_E_hp_in_kWh")
        e_c_hp = s("C_E_hp_in_kWh")
        e_h_backup = s("H_E_backup_in_kWh")
        e_w_backup = s("W_E_backup_in_kWh")
        e_c_backup = s("C_E_backup_in_kWh")
        aux_hw = s("W_HW_gen_aux_kWh")
        aux_c = s("W_C_gen_aux_kWh")

        e_hw_gen = e_h_hp + e_w_hp + e_h_backup + e_w_backup
        e_c_gen = e_c_hp + e_c_backup
        q_hw = q_h + q_w
        q_hw_hp = s("H_Q_hp_out_kWh") + s("W_Q_hp_out_kWh")
        e_hw_hp = e_h_hp + e_w_hp

        spf_hw_gen = _ratio(q_hw, e_hw_gen + aux_hw)
        spf_hw_hp = _ratio(q_hw_hp, e_hw_hp + aux_hw)
        seer_c_gen = _ratio(q_c, e_c_gen + aux_c)

        return {
            "QH_gen_out_kWh": q_h,
            "QW_gen_out_kWh": q_w,
            "QC_gen_out_kWh": q_c,
            "QHW_gen_out_kWh": q_hw,
            "QH_hp_out_kWh": s("H_Q_hp_out_kWh"),
            "QW_hp_out_kWh": s("W_Q_hp_out_kWh"),
            "QC_hp_out_kWh": s("C_Q_hp_out_kWh"),
            "QHW_hp_out_kWh": q_hw_hp,
            "QHW_backup_out_kWh": s("H_Q_backup_out_kWh") + s("W_Q_backup_out_kWh"),
            "QC_backup_out_kWh": s("C_Q_backup_out_kWh"),
            "QHW_unmet_kWh": s("H_Q_unmet_kWh") + s("W_Q_unmet_kWh"),
            "QC_unmet_kWh": s("C_Q_unmet_kWh"),
            "EH_hp_in_kWh": e_h_hp,
            "EW_hp_in_kWh": e_w_hp,
            "EC_hp_in_kWh": e_c_hp,
            "EHW_backup_in_kWh": e_h_backup + e_w_backup,
            "EC_backup_in_kWh": e_c_backup,
            "EHW_gen_in_kWh": e_hw_gen,
            "EC_gen_in_kWh": e_c_gen,
            "WHW_gen_aux_kWh": aux_hw,
            "WC_gen_aux_kWh": aux_c,
            "E_total_electricity_kWh": e_hw_gen + aux_hw + e_c_gen + aux_c,
            "QHW_gen_ls_tot_kWh": s("Q_HW_gen_ls_tot_kWh"),
            "QHW_gen_ls_rbl_tot_kWh": s("Q_HW_gen_ls_rbl_tot_kWh"),
            "QHW_environment_in_kWh": s("H_Q_environment_in_kWh") + s("W_Q_environment_in_kWh"),
            "QC_rejected_kWh": s("C_Q_rejected_kWh"),
            "SPF_HW_gen": spf_hw_gen,
            "SPF_HW_hp": spf_hw_hp,
            "e_HW_gen": _ratio(1.0, spf_hw_gen),
            "SEER_C_gen": seer_c_gen,
            "H_part_load_factor_mean": _weighted_mean(
                bins["H_part_load_factor"], bins["H_Q_hp_out_kWh"] + _KWH_EPS
            ),
            "W_part_load_factor_mean": _weighted_mean(
                bins["W_part_load_factor"], bins["W_Q_hp_out_kWh"] + _KWH_EPS
            ),
            "C_part_load_factor_mean": _weighted_mean(
                bins["C_part_load_factor"], bins["C_Q_hp_out_kWh"] + _KWH_EPS
            ),
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _coerce_performance_map(
    data: Any,
    performance_column: str,
    service_name: str,
    required: bool = True,
) -> pd.DataFrame | None:
    if data is None:
        if required:
            return None
        return None

    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if df.empty:
        if required:
            raise ValueError(f"{service_name}_performance_map is empty.")
        return None

    aliases = {
        "source_temperature_C": [
            "source_temperature_C",
            "source_temp_C",
            "T_source_C",
            "T_ext",
            "outdoor_temperature_C",
        ],
        "sink_temperature_C": [
            "sink_temperature_C",
            "sink_temp_C",
            "T_sink_C",
            "supply_temperature_C",
            "T_supply_C",
        ],
        "capacity_kW": [
            "capacity_kW",
            "heating_capacity_kW",
            "cooling_capacity_kW",
            "thermal_capacity_kW",
            "power_kW",
        ],
        performance_column: [performance_column, performance_column.upper()],
    }

    renamed: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in df.columns:
                renamed[name] = canonical
                break

    df = df.rename(columns=renamed)
    required_columns = ["source_temperature_C", "sink_temperature_C", "capacity_kW", performance_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{service_name}_performance_map is missing required columns: {missing}."
        )

    df = df[required_columns].astype(float).dropna()
    if df.empty:
        raise ValueError(f"{service_name}_performance_map has no numeric rows.")
    if (df["capacity_kW"] < 0).any() or (df[performance_column] <= 0).any():
        raise ValueError(
            f"{service_name}_performance_map must use non-negative capacity and positive "
            f"{performance_column.upper()} values."
        )
    _warn_about_performance_map_quality(df, performance_column, service_name)
    return df


def _warn_about_performance_map_quality(
    df: pd.DataFrame, performance_column: str, service_name: str
) -> None:
    """Data-quality sanity checks, NOT an EN 14511 conformance check: this
    module interpolates whatever source/sink-temperature grid it is given,
    without verifying the points actually correspond to EN 14511's own
    mandatory rating conditions (or EN 14825's A/B/C/D/... points, for
    "Path A"/Table 11-12 use). Warns (does not refuse) on data shapes that
    are very likely to give a misleading result; conformance of the
    underlying product data to EN 14511/14825 remains the caller's
    responsibility."""

    n_source = df["source_temperature_C"].nunique()
    n_sink = df["sink_temperature_C"].nunique()
    if n_source < 2 or n_sink < 2:
        warnings.warn(
            f"{service_name}_performance_map has only {n_source} distinct source "
            f"temperature(s) and {n_sink} distinct sink temperature(s) - EN 14511/14825 "
            "rating points normally span several source AND sink temperatures (e.g. air-"
            "source units are typically rated at multiple outdoor temperatures and at "
            "least two sink/flow temperatures). With fewer than 2x2 points this module "
            "cannot bilinearly interpolate across the real operating range and will "
            "effectively hold the performance constant along the missing axis.",
            UserWarning,
            stacklevel=4,
        )

    duplicates = df.duplicated(subset=["source_temperature_C", "sink_temperature_C"], keep=False)
    if duplicates.any():
        conflicting = (
            df.loc[duplicates]
            .groupby(["source_temperature_C", "sink_temperature_C"])[performance_column]
            .nunique()
        )
        if (conflicting > 1).any():
            warnings.warn(
                f"{service_name}_performance_map has multiple rows at the same "
                f"(source_temperature_C, sink_temperature_C) point with DIFFERENT "
                f"{performance_column.upper()} values - ambiguous input data; the "
                "interpolation grid build keeps only one value per point (via a "
                "pivot table), silently discarding the others.",
                UserWarning,
                stacklevel=4,
            )


def _interpolate_performance(
    performance_map: pd.DataFrame, source_temperature_C: float, sink_temperature_C: float, column: str
) -> float:
    pivot = performance_map.pivot_table(
        index="source_temperature_C",
        columns="sink_temperature_C",
        values=column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    if not pivot.isna().any().any():
        sources = pivot.index.to_numpy(dtype=float)
        sinks = pivot.columns.to_numpy(dtype=float)
        values = pivot.to_numpy(dtype=float)
        return _bilinear_or_linear(
            sources,
            sinks,
            values,
            float(source_temperature_C),
            float(sink_temperature_C),
        )

    # Fallback for scattered maps: inverse-distance interpolation with clamped
    # temperature scaling so a nearly matching point dominates.
    points = performance_map[
        ["source_temperature_C", "sink_temperature_C", column]
    ].to_numpy(dtype=float)
    src_range = max(float(np.ptp(points[:, 0])), 1.0)
    sink_range = max(float(np.ptp(points[:, 1])), 1.0)
    distances = np.sqrt(
        ((points[:, 0] - source_temperature_C) / src_range) ** 2
        + ((points[:, 1] - sink_temperature_C) / sink_range) ** 2
    )
    exact = distances < 1e-12
    if exact.any():
        return float(points[exact, 2][0])
    weights = 1.0 / np.maximum(distances, 1e-12) ** 2
    return float(np.sum(weights * points[:, 2]) / np.sum(weights))


def _bilinear_or_linear(
    xs: np.ndarray, ys: np.ndarray, values: np.ndarray, x: float, y: float
) -> float:
    if len(xs) == 1 and len(ys) == 1:
        return float(values[0, 0])
    if len(xs) == 1:
        return float(np.interp(y, ys, values[0, :]))
    if len(ys) == 1:
        return float(np.interp(x, xs, values[:, 0]))

    x1_i, x2_i = _bracket_indices(xs, x)
    y1_i, y2_i = _bracket_indices(ys, y)
    x1, x2 = xs[x1_i], xs[x2_i]
    y1, y2 = ys[y1_i], ys[y2_i]

    q11 = values[x1_i, y1_i]
    q21 = values[x2_i, y1_i]
    q12 = values[x1_i, y2_i]
    q22 = values[x2_i, y2_i]

    tx = 0.0 if x2 == x1 else (x - x1) / (x2 - x1)
    ty = 0.0 if y2 == y1 else (y - y1) / (y2 - y1)
    tx = float(np.clip(tx, 0.0, 1.0))
    ty = float(np.clip(ty, 0.0, 1.0))

    return float(
        q11 * (1 - tx) * (1 - ty)
        + q21 * tx * (1 - ty)
        + q12 * (1 - tx) * ty
        + q22 * tx * ty
    )


def _bracket_indices(values: np.ndarray, target: float) -> tuple[int, int]:
    if target <= values[0]:
        return 0, 1
    if target >= values[-1]:
        return len(values) - 2, len(values) - 1
    upper = int(np.searchsorted(values, target, side="right"))
    return upper - 1, upper


def _series_from_aliases(
    df: pd.DataFrame, aliases: list[str], default: Any | None
) -> pd.Series | None:
    for name in aliases:
        if name in df.columns:
            return df[name]
    if default is None:
        return None
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.Series(values, dtype=float)
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    mask = v.notna() & w.notna()
    if not mask.any() or float(w[mask].sum()) <= _KWH_EPS:
        return float(v[mask].mean()) if mask.any() else 0.0
    return float(np.average(v[mask], weights=w[mask]))


def _prefix_dict(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in values.items()
        if key != "service"
    }


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fraction(value: Any, name: str) -> float:
    try:
        fraction = float(value)
    except Exception as exc:  # pragma: no cover - defensive type context
        raise ValueError(f"{name} must be a fraction between 0 and 1.") from exc
    return float(np.clip(fraction, 0.0, 1.0))


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _KWH_EPS:
        return float("nan")
    return float(numerator / denominator)
