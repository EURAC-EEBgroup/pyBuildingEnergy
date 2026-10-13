"""EN 15316-4-1 boiler generation calculator.

This module is the single implementation of EN 15316-4-1:2017 clause 6 (space
heating and DHW generation, combustion systems - gas/oil boilers and biomass)
in this package. It implements the cycle / expenditure-factor method in a
table-driven way: generator-case data (efficiencies, standby losses, auxiliary
powers) are supplied through the input configuration, not hardcoded.

Clause 6 formulas implemented:

* Eq.(37)-(38): generator thermal loss, linearly interpolated between the
  0 %/intermediate/full-load points (P0, Pint, Pn) by the actual load ratio.
* Eq.(41),(43),(45): temperature-corrected efficiency and the resulting
  thermal loss at full/intermediate load, including the Hs/Hi (gross/net
  calorific value) correction factor Eq.(43)/(45) call for. f_hs_hi_pct
  DEFAULTS from fuel_type via Table B.9 (DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE -
  natural gas 111 %, LPG 109 %, fuel oil 106 %, wood/biomass 108 %,
  electricity/district heat 100 %; an unrecognised fuel_type falls back to
  100 %) rather than a flat 100 % (Hs = Hi) regardless of fuel - no
  combustion fuel in Table B.9 actually has Hs/Hi = 1.00, so a flat 100 %
  understated Eq.(43)/(45)/(75)'s computed loss for every combustion fuel.
  This IS a deliberate default-VALUE change (same precedent as f_aux_rvd's
  default above): an explicit f_hs_hi_pct still always overrides it.
* Tables B.4/B.5: f_corr,Pn and f_corr,Pint are two DISTINCT coefficients
  in the standard (e.g. standard/biomass boilers: f_corr,Pn = 0 %/K,
  f_corr,Pint = 0.04 %/K - NOT the same value). f_corr_pn_pct_per_K/
  f_corr_pint_pct_per_K (both None by default) override the shared
  f_corr_pct_per_K field for the full-load (Eq.41/43) / intermediate-load
  (Eq.44/45) correction respectively when set - unchanged behaviour (one
  shared coefficient) for configs that only ever set f_corr_pct_per_K.
  default_boiler_config() DOES set f_corr_pint_pct_per_K from Table B.5
  for every curated boiler_type (ANNEX_B_DEFAULTS), since that function's
  whole purpose is to reproduce the standard's own default values.
* Eq.(40): opt-in (BoilerConfig.dhw_trace_band, False by default) DHW
  trace-band full-load efficiency correction, REPLACING Eq.(41) for
  eta_Pn only (not Eq.(44)'s eta_Pint), per the standard's own instruction
  ("insert eta_gen;Pn;traceband instead of eta_gen;Pn;corr in Formula
  (43)"). False keeps Eq.(41)'s circulation-loop formula - unchanged
  pre-existing behaviour.
* Eq.(46): opt-in (BoilerConfig.apply_p0_temp_correction, False by
  default) temperature correction of the 0 %-load (stand-by) generator
  loss P0, by the actual circuit/installation-room temperature difference
  versus Table B.3's test reference (theta_test_P0_C, 70 degC default) -
  see _loss_powers_kW(). False keeps P_gen_ls_P0_W as a flat, temperature-
  independent constant - unchanged pre-existing behaviour. Separately,
  Eq.(46) AT TEST CONDITIONS (bracket = 1) is also what
  generator_standby_loss_default_W() now implements for Eq.(5)'s default
  P0: (Pn/eta_Pn) * f_gen,ls,P0 * f_Hs/Hi, via its eta_Pn_pct/f_hs_hi_pct
  parameters (both default 100.0, reducing to the OLD, incomplete f_gen,
  ls,P0 * Pn form for direct callers who don't pass them - unchanged
  behaviour there). default_boiler_config() DOES pass the generator's own
  computed eta_Pn and Table B.9's f_hs_hi_pct, so its P_gen_ls_P0_W is no
  longer understated for a real (eta_Pn < 100 %) generator - a deliberate
  default-VALUE change there, same precedent as f_aux_rvd/f_hs_hi_pct
  above.
* Eq.(42): full-load efficiency of a condensing boiler, as the standard
  itself defines it - linear interpolation/extrapolation between efficiency
  measured at 60 degC and at 30 degC average return temperature - used
  whenever the case data gives both ("eta_Pn_test_60_pct"/"eta_Pn_test_30_pct"
  in the efficiency_table case; measured data, so preferred over either of
  the two condensing-gain methods below when available).
* Eq.(15)-(24) (clause 5.7.3's flue-gas/condensation chemistry - flue gas
  temperature, dry gas composition, combustion-air/flue-gas humidity,
  condensing water mass, latent heat of condensation): implemented as the
  standalone condensing_gain_fraction() function, using Table B.10a/B.12/
  B.13/B.14's curated defaults (natural gas/propane/butane/light_oil_EL -
  the standard's own "condensing boiler (oil/gas)" fuel set). Opt-in via
  BoilerConfig.combustion_fuel_key (None by default - see below); Eq.(25)-
  (26) (chimney loss from a flue-gas O2 measurement, an ALTERNATIVE way to
  derive the loss used for eta_Pn_test_pct itself, not a step in the
  condensing-gain chain) is exposed separately as
  chimney_loss_pct_from_flue_gas_analysis(), a standalone helper NOT wired
  into compute_step().
* Without eta_Pn_test_60_pct/eta_Pn_test_30_pct AND without
  combustion_fuel_key, a simplified condensing-gain correction (return
  temperature vs dew_point_C/condensing_gain_pct) is used instead - the
  pre-existing, still-default behaviour, kept for configs that supply
  neither of the two more rigorous Eq.(42)/Eq.(15)-(24) inputs.
* Clause 5.7.1/5.7.2, Eq.(7)-(14) ("measured values" - deriving boiler
  efficiency from a measured fuel flow rate, chimney-on/chimney-off/
  envelope loss fractions, instead of starting from a product-certificate
  eta_Pn_test_pct or an Annex B correlation): implemented as the standalone
  fuel_power_input_kW() (Eq.13), split_standby_loss_pct() (Eq.9, inverted -
  see its own docstring), boiler_efficiency_from_measured_values() (Eq.7-
  (8)/(10)-(12)) and rated_power_from_measured_efficiency_kW() (Eq.14)
  functions, reusing chimney_loss_pct_from_flue_gas_analysis() (Eq.25-26)
  and Table B.7's f_gen,env split ratio (DEFAULT_F_JACKET_BY_BOILER_TYPE).
  Eq.(27) (envelope loss from measured surface areas/temperatures via
  Figure B.1's heat-transfer-coefficient GRAPH, not a table) is NOT
  implemented - split_standby_loss_pct() derives the envelope/chimney-off
  split from Table B.7's ratio instead, sidestepping the need to digitise
  that graph. None of this is wired into BoilerConfig/compute_step(); feed
  the results into eta_Pn_test_pct/P_gen_ls_P0_W by hand, same as the other
  standalone helpers above.
* Eq.(50): recoverable envelope (jacket) loss, from the standby loss RATE
  over the whole period of use - not from the combined on+off loss, which
  would wrongly include flue-gas losses that cannot leave through the jacket.
* Eq.(34): fuel input also credits auxiliary energy recovered into the
  fluid (Q_gen,aux,rvd) and any Q_gen,ren.
* Eq.(51)-(54): recoverable/recovered auxiliary energy, with f_aux,rbl and
  f_aux,rvd kept as Eq.(51) complements by default (not two independent
  inputs that could double-count the same auxiliary energy). A caller MAY
  still override f_aux_rvd explicitly to decouple the two (e.g.
  independently measured values) - _load_parameters() warns (UserWarning,
  does not refuse) whenever an explicit f_aux_rvd + f_aux_recoverable
  doesn't sum to 1, since that risks double-counting (sum > 1) or losing
  (sum < 1) recoverable auxiliary energy. Eq.(53)'s (1-f_brm) factor
  (Table B.8) is the SAME f_location Eq.(50) already uses for the envelope
  loss (f_location = 1-f_brm: 1.0 inside the heated space, 0.0 outdoors).
* Table B.7/B.8-derived defaults for f_jacket and f_location from
  boiler_type/boiler_location when not given explicitly (see
  DEFAULT_F_JACKET_BY_BOILER_TYPE / DEFAULT_F_LOCATION_BY_LOCATION), and
  Eq.(1),(4)-(6) (generator_efficiency_pn_default_pct and friends) plus a
  curated Annex B default dataset (ANNEX_B_DEFAULTS, default_boiler_config())
  for when product/test data for efficiency_table/loss_table/P_aux/f_corr is
  not available - see that function's docstring for exactly what subset of
  Annex B it covers. Table B.1's full construction-year matrix (ONE Table
  B.1 row per boiler_type, every build-year band the standard gives for
  that row) is available via ANNEX_B_YEAR_BANDS/default_boiler_config()'s
  build_year_band parameter; Table B.3 (stand-by loss) and Table B.6
  (auxiliary power) still only cover one representative value per
  boiler_type (they use a different categorisation than Table B.1's), and
  Table B.2's separate EN 303-5 class-based method is not covered at all.
* Eq.(55)-(57): auxiliary power, linearly interpolated between P0/Pint/Pn by
  load ratio, the same way as the thermal loss - not a flat on/off value.
* Eq.(58): generator thermal output = f_ctrl,ls * heating demand + DHW
  demand (Table B.16's control-strategy factor, 1.00-1.06, inflates ONLY
  the heating duty the generator must produce - not what gets reported as
  served to distribution, see compute_step()). 1.00 (no penalty) by
  default; set f_ctrl_ls or control_type to use a different value.
* Eq.(59)-(67) (clause 6.10, separate per-service operating time/load
  factor): "proportional" (BoilerConfig.priority_mode, the default and the
  UNCHANGED pre-existing behaviour) still costs heating and DHW as ONE
  combined load per time step (one beta, one loss/efficiency evaluation),
  splitting the resulting fuel back to each service by its share of the
  combined demand. Setting priority_mode="dhw_priority" switches to a
  DHW-priority sequential-operation implementation of clause 6.10 (see
  _compute_step_dhw_priority()): DHW is served first out of the step's
  capacity, heating gets whatever capacity/time remains, and each service
  gets its own load ratio (beta_H/beta_W) instead of one blended beta.
  This is a REINTERPRETATION of Eq.(63)-(66), not a literal transcription -
  as printed, their denominator (Pn * t_X, with t_X from Eq.(59)-(62))
  makes beta_X algebraically identically 1 for any demand, which cannot be
  what was intended; see docs/standards_errata.md for the full derivation
  and the correction this module applies instead. Simultaneous (not
  priority-ordered) parallel operation and cooling/ventilation services
  (Eq.(60)-(61), beta_C/beta_V) are out of scope - this calculator only
  ever serves heating and DHW. Eq.(68) (beta_Pint = Pint/Pn) IS captured,
  structurally, as intermediate_load_fraction, in both modes.

Clause 6.11/6.12 (direct-heated DHW appliances):

* Eq.(69): instantaneous electrical water heater (boiler_type=
  "electric_instantaneous") - negligible thermal losses, no auxiliary
  energy, E_gen,in = f_tr,ls * Q_gen,out. f_tr,ls defaults from Table B.17
  by installation_era ("before_1980"=1.1, "since_1980"=1.0, the default).
* 6.11.1's instantaneous GAS water heater ("shall be calculated as low
  temperature combination boilers") needs no dedicated code: pass
  boiler_type="low_temperature" (or the "gas_instantaneous" alias, resolved
  to "low_temperature") and use the existing full boiler method above.
* Eq.(70),(74)-(76) + Formula (B.1)/(B.2): gas-fired domestic storage water
  heater (boiler_type="gas_storage_water_heater") - a stand-by storage loss
  (Eq.70, from the EN 89-measured or Annex-B-default q_B,S) plus a
  combustion loss at the FIXED EN 89 reference efficiency eta_100% (84 %
  default, Eq.74/75 - not load-interpolated between P0/Pint/Pn like the
  full boiler method). Rated output defaults from storage volume via
  Eq.(76) when not given directly. Auxiliary energy is set to zero, per the
  standard's own explicit simplification for this appliance.
* Eq.(77)-(82): the 24 h tapping-cycle test method (clause 6.12) is
  implemented as the standalone dhw_tapping_cycle_energy_kWh() function -
  a DAILY method driven by EN 13203-2 test-cycle data points, structurally
  different from compute_step()/run_timeseries()'s hourly loss-curve
  stepping. boiler_type="dhw_tapping_cycle_appliance"
  (SIMPLE_DHW_GENERATOR_TYPES) IS wired into run_timeseries(): it groups
  the input by calendar day (the df index must be a DatetimeIndex),
  calls dhw_tapping_cycle_energy_kWh() once per day with that day's total
  demand, and distributes the day's fuel/auxiliary energy back across its
  hours proportionally to each hour's own share. compute_step() on this
  boiler_type raises NotImplementedError (no single-hour meaning) -
  dhw_tapping_cycle_energy_kWh() itself remains the right tool for a
  single day's total, called directly.

Capacity is capped at rated_power_kW: demand beyond that is reported as
unmet (Q_H_gen_unmet_kWh / Q_W_gen_unmet_kWh), consistent with the other
generation modules (heat pump, district, cogeneration) in this package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-9

GENERATOR_TYPES = {
    "standard",
    "low_temperature",
    "condensing",
    "biomass_log",
    "biomass_pellet",
    # Clause 6.11: direct-heated DHW appliances (see module docstring).
    "electric_instantaneous",
    "gas_storage_water_heater",
    # Clause 6.12: 24 h tapping-cycle test appliance (see module docstring
    # and dhw_tapping_cycle_energy_kWh()).
    "dhw_tapping_cycle_appliance",
}

# Clause 6.11.1: "Instantaneous gas water heater shall be calculated as low
# temperature combination boilers" - a convenience spelling that resolves to
# the existing "low_temperature" boiler_type rather than needing its own
# calculation path.
GENERATOR_TYPE_ALIASES = {
    "gas_instantaneous": "low_temperature",
    "gas_instantaneous_water_heater": "low_temperature",
}

# boiler_type values that are direct-heated DHW appliances (clause 6.11/
# 6.12), NOT the P0/Pint/Pn loss-curve combustion method clause 6.5-6.9
# define - _load_parameters()/compute_step() branch on this set.
SIMPLE_DHW_GENERATOR_TYPES = {
    "electric_instantaneous", "gas_storage_water_heater", "dhw_tapping_cycle_appliance",
}

LOCATION_TYPES = {
    "inside_heated",
    "adjacent_unheated",
    "outside_building",
}

# Table B.17 - default f_tr,ls for an instantaneous electrical water heater,
# by installation era (Eq.69).
DEFAULT_F_TR_LS_BY_ERA = {
    "before_1980": 1.1,
    "since_1980": 1.0,
}

# BoilerConfig.priority_mode (see its own comment) - how a combined H+W
# demand is shared when it exceeds rated capacity.
PRIORITY_MODES = {"proportional", "dhw_priority"}

# Table B.12 - saturation humidity [kg water vapour / m3 of dry gas] vs
# temperature [degC], at 1,01325 bar, for combustion air or flue gas.
# Linearly interpolated for intermediate temperatures (the standard allows
# linear or polynomial interpolation); clamped to the table's own 0-70 degC
# range outside it.
_SATURATION_HUMIDITY_TABLE_B12_C = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
_SATURATION_HUMIDITY_TABLE_B12_KG_PER_M3 = [0.00493, 0.00986, 0.01912, 0.03521, 0.06331, 0.1112, 0.1975, 0.3596]

# Table B.13a - stoichiometric combustion data for gas/oil fuels, at
# 1,01325 bar and 0 degC: a CURATED subset (the table's "(H)" high-
# calorific natural gas column, not the separate "(L)" low-calorific
# variant; the solid-fuel properties in Table B.13b are not included here,
# since condensation gain in this module only ever applies to
# boiler_type="condensing", which the standard itself restricts to gas/oil
# - see Table B.1's "Generator type: condensing boiler (oil/gas)" heading).
# Units: Hs/Hi in kWh/m3 (gases) or kWh/kg (light_oil_EL); V_air_st_dry/
# V_fg_st_dry in m3/m3 or m3/kg; m_H2O_st in kg/m3 or kg/kg - consistently
# "per unit of fuel" within each fuel's own row.
FUEL_COMBUSTION_PROPERTIES_B13: dict[str, dict[str, float]] = {
    "natural_gas": {"Hs": 11.32, "Hi": 10.22, "V_air_st_dry": 9.79, "V_fg_st_dry": 8.80, "m_H2O_st": 1.47},
    "propane": {"Hs": 28.11, "Hi": 25.88, "V_air_st_dry": 24.36, "V_fg_st_dry": 22.30, "m_H2O_st": 2.94},
    "butane": {"Hs": 37.21, "Hi": 34.36, "V_air_st_dry": 32.08, "V_fg_st_dry": 29.68, "m_H2O_st": 3.75},
    "light_oil_EL": {"Hs": 12.60, "Hi": 11.89, "V_air_st_dry": 11.23, "V_fg_st_dry": 10.49, "m_H2O_st": 1.18},
}

# Table B.10a - Siegert constants (c10, c11) for Eq.(25)'s chimney loss with
# the burner on, by fuel - a CURATED subset (Table B.10a's common-fuel row
# only; Table B.10b-e's moisture-content-dependent solid-fuel variants are
# not transcribed here).
FUEL_SIEGERT_CONSTANTS_B10A: dict[str, tuple[float, float]] = {
    "natural_gas": (0.66, 0.009),
    "lpg": (0.63, 0.008),
    "light_oil_EL": (0.68, 0.007),
    "heavy_fuel_oil": (0.68, 0.007),
    "town_gas": (0.63, 0.011),
}

# Table B.14 - default combustion-air/flue-gas humidity and O2 content,
# "all cases".
DEFAULT_X_AIR_PCT = 50.0
DEFAULT_X_FG_PCT = 100.0
DEFAULT_X_O2_FG_DRY_PCT = 6.0

# Table B.8 - default installation room temperature theta_brm by generator
# location, reusing this module's existing LOCATION_TYPES/f_location
# mapping. "outside_building" is Table B.8's "outdoors" row, whose own
# theta_brm is theta_e (the actual external temperature) - this module has
# no weather input to supply that, so it falls back to a generic placeholder
# (override theta_brm_C explicitly for anything more precise).
DEFAULT_THETA_BRM_C_BY_LOCATION = {
    "inside_heated": 20.0,
    "adjacent_unheated": 13.0,  # "in the boiler room" - see DEFAULT_F_LOCATION_BY_LOCATION's own note
    "outside_building": 5.0,
}

# Table B.9 - ratio of gross to net calorific value (Hs/Hi), by energy
# carrier, on this module's f_hs_hi_pct percent scale (100 * Hs/Hi). No
# combustion fuel in Table B.9 has Hs/Hi = 1.00 - only electricity and
# district heat do - so f_hs_hi_pct DEFAULTS from fuel_type via this table
# (see _load_parameters()), not to a flat 100.0 regardless of fuel, which
# would understate Eq.(43)/(45)/(75)'s computed loss for every combustion
# fuel. An explicit f_hs_hi_pct always overrides this; an unrecognised
# fuel_type falls back to 100.0 (Hs = Hi, no correction - the previous
# behaviour for every fuel_type, kept where this table has no entry).
DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE = {
    "natural_gas": 111.0,
    "gas": 111.0,
    "lpg": 109.0,
    "liquid_petroleum_gas": 109.0,
    "propane": 109.0,
    "butane": 109.0,
    "fuel_oil": 106.0,
    "oil": 106.0,
    "light_oil_el": 106.0,
    "anthracite_coal": 104.0,
    "anthracite": 104.0,
    "lignite_coal": 107.0,
    "lignite": 107.0,
    "wood": 108.0,
    "biomass": 108.0,
    "biomass_pellet": 108.0,
    "biomass_log": 108.0,
    "biomass_woodchip": 108.0,
    "district_heat": 100.0,
    "district_heating": 100.0,
    "electricity": 100.0,
}


def saturation_humidity_kg_per_m3(theta_C: float) -> float:
    """Table B.12: saturation humidity of air or flue gas [kg water vapour
    per m3 of dry gas], as a function of temperature, at 1,01325 bar."""

    theta = min(
        max(float(theta_C), _SATURATION_HUMIDITY_TABLE_B12_C[0]), _SATURATION_HUMIDITY_TABLE_B12_C[-1]
    )
    return float(np.interp(theta, _SATURATION_HUMIDITY_TABLE_B12_C, _SATURATION_HUMIDITY_TABLE_B12_KG_PER_M3))


def condensing_gain_fraction(
    fuel_key: str,
    theta_return_C: float,
    theta_brm_C: float,
    delta_theta_wfg_K: float = 10.0,
    x_air_pct: float = DEFAULT_X_AIR_PCT,
    x_fg_pct: float = DEFAULT_X_FG_PCT,
    X_O2_fg_dry_pct: float = DEFAULT_X_O2_FG_DRY_PCT,
) -> float:
    """Eq.(15)-(24): the recoverable latent-heat-of-condensation fraction
    Q_cond/H_s [-] for a condensing boiler, from first-principles combustion
    chemistry (flue gas temperature, dry gas composition, humidity of
    combustion air/flue gas, condensing water mass, latent heat), instead of
    the flat condensing_gain_pct/dew_point_C approximation this module falls
    back to when combustion data is not available (see BoilerConfig.
    combustion_fuel_key and _eta_corrected_pct()).

    Returns 0.0 when Eq.(20)'s m_H2O;cond comes out negative - the
    standard's own "no condensation at this return temperature" rule.
    """

    if fuel_key not in FUEL_COMBUSTION_PROPERTIES_B13:
        raise ValueError(
            f"No curated Table B.13 combustion data for fuel {fuel_key!r}. "
            f"Available: {sorted(FUEL_COMBUSTION_PROPERTIES_B13)}."
        )
    props = FUEL_COMBUSTION_PROPERTIES_B13[fuel_key]

    # Eq.(15): flue gas temperature at the boiler outlet connection.
    theta_fg = float(theta_return_C) + float(delta_theta_wfg_K)

    # Eq.(16)-(17): actual (vs stoichiometric) dry flue gas/combustion air,
    # from the target flue gas O2 content.
    V_fg_dry = props["V_fg_st_dry"] * 21.0 / max(21.0 - float(X_O2_fg_dry_pct), _EPS)
    V_air_dry = props["V_air_st_dry"] + V_fg_dry - props["V_fg_st_dry"]

    # Eq.(18)-(19): total humidity of combustion air and flue gas.
    m_H2O_air = saturation_humidity_kg_per_m3(theta_brm_C) * V_air_dry * float(x_air_pct) / 100.0
    m_H2O_fg = saturation_humidity_kg_per_m3(theta_fg) * V_fg_dry * float(x_fg_pct) / 100.0

    # Eq.(20): condensing water mass - no condensation if negative.
    m_H2O_cond = props["m_H2O_st"] + m_H2O_air - m_H2O_fg
    if m_H2O_cond <= 0.0:
        return 0.0

    # Eq.(22): specific latent heat of condensation [Wh/kg] -> kWh/kg.
    h_cond_fg_kWh_per_kg = (694.61 - 0.6764 * theta_fg) / 1000.0

    # Eq.(23)-(24): specific condensation heat, as a fraction of the gross
    # calorific value - both are "per unit of fuel", so the fuel's own unit
    # (kg or m3) cancels, leaving a dimensionless efficiency-gain fraction.
    Q_cond = m_H2O_cond * h_cond_fg_kWh_per_kg
    return max(Q_cond / props["Hs"], 0.0)


def chimney_loss_pct_from_flue_gas_analysis(
    fuel_key: str,
    theta_fg_C: float,
    theta_brm_C: float,
    X_O2_measured_pct: float,
    theta_gen_ref_C: Optional[float] = None,
    theta_gen_meas_C: Optional[float] = None,
    f_corr_ch_on_pct_per_K: float = 0.045,
) -> float:
    """Eq.(25)-(26): thermal loss through the chimney with the burner ON
    [%], from a flue gas O2 analysis and the Siegert constants (Table
    B.10a) - an ALTERNATIVE source for the full-load loss, from a site
    measurement, instead of product/Annex-B test efficiency data. Not wired
    into compute_step() (which always derives losses from eta_Pn_test_pct/
    Eq.(1)-(4) correlations); provided as a standalone helper for callers
    who have flue-gas-analysis measurements and want to derive f_ch,on
    themselves to feed in as eta_Pn_test_pct = 100 - f_ch,on - f_gen,env.

    Optionally corrected from measured to reference water temperature
    conditions via Eq.(26) (0,045 %/K default correction factor) when both
    theta_gen_ref_C and theta_gen_meas_C are given.
    """

    if fuel_key not in FUEL_SIEGERT_CONSTANTS_B10A:
        raise ValueError(
            f"No curated Table B.10a Siegert constants for fuel {fuel_key!r}. "
            f"Available: {sorted(FUEL_SIEGERT_CONSTANTS_B10A)}."
        )
    c10, c11 = FUEL_SIEGERT_CONSTANTS_B10A[fuel_key]
    f_ch_on_meas = (float(theta_fg_C) - float(theta_brm_C)) * (
        c10 / max(21.0 - float(X_O2_measured_pct), _EPS) + c11
    )
    if theta_gen_ref_C is None or theta_gen_meas_C is None:
        return float(f_ch_on_meas)
    # Eq.(26): correction from measured to reference water temperature.
    return float(f_ch_on_meas - (float(theta_gen_ref_C) - float(theta_gen_meas_C)) * f_corr_ch_on_pct_per_K)


def fuel_power_input_kW(fuel_flow_rate_per_h: float, Hi_kWh_per_unit: float) -> float:
    """Eq.(13): average fuel power input P_gen,del [kW], from a measured
    fuel flow rate (mass or volume per hour - matching Hi's own unit, see
    FUEL_COMBUSTION_PROPERTIES_B13/Table B.13) and the fuel's net calorific
    value Hi [kWh per unit]. Eq.(13)'s own kJ-to-kWh conversion (its
    "[kWh/3600kJ]" bracket) is absorbed here by working directly in kWh
    units throughout, the same way FUEL_COMBUSTION_PROPERTIES_B13 already
    does for Hs/Hi."""

    return float(fuel_flow_rate_per_h) * float(Hi_kWh_per_unit)


def split_standby_loss_pct(f_gen_ls_P0_pct: float, f_gen_env_fraction: float) -> tuple[float, float]:
    """Splits a TOTAL stand-by loss fraction f_gen,ls,P0 [%] (Eq.(5), or an
    independently measured P0 test) into its chimney-off and envelope
    shares, using Table B.7's f_gen,env - the SAME 0-1 fraction this module
    already uses as f_jacket (DEFAULT_F_JACKET_BY_BOILER_TYPE): f_gen,env
    [%] = f_gen,ls,P0 [%] * f_gen_env_fraction; f_ch,off [%] = the
    remainder.

    This is the inverse of how Eq.(9) (f_gen,ls,P0 = f_ch,off + f_gen,env)
    is phrased, which assumes both terms are already independently
    measured. Clause 5.7.5's own wording for Table B.7 ("part of stand-by
    heat losses attributed to... the generator envelope") is a SPLIT
    RATIO, so this function uses it to derive both Eq.(9) components from
    the already-known total instead - sidestepping Eq.(27)'s graph-based
    (Figure B.1 heat transfer coefficient) direct envelope-loss
    measurement, which this module does not implement.

    Returns (f_ch_off_pct, f_gen_env_pct).
    """

    f_gen_env_pct = float(f_gen_ls_P0_pct) * float(f_gen_env_fraction)
    f_ch_off_pct = float(f_gen_ls_P0_pct) - f_gen_env_pct
    return f_ch_off_pct, f_gen_env_pct


def boiler_efficiency_from_measured_values(
    P_gen_del_kW: float,
    f_ch_on_pct: float,
    f_ch_off_pct: float,
    f_gen_env_pct: float,
    beta_Pint: Optional[float] = None,
    P_cond_kW: float = 0.0,
) -> dict[str, float]:
    """Clause 5.7.1/5.7.2, Eq.(7)-(12): derives boiler efficiency (and the
    underlying loss powers) from measured/derived loss FRACTIONS -
    f_ch,on [%] (Eq.(25)-(26), chimney_loss_pct_from_flue_gas_analysis()),
    f_ch,off [%] and f_gen,env [%] (Table B.11's default, or
    split_standby_loss_pct()) - and the measured fuel power input
    P_gen,del [kW] (Eq.(13), fuel_power_input_kW()). This is clause 5.7's
    "measured values" pathway: an ALTERNATIVE to supplying eta_Pn_test_pct/
    P_gen_ls_P0_W directly from a product certificate, or deriving them
    from Eq.(1)-(6)'s Annex B correlations (default_boiler_config()). Not
    wired into BoilerConfig/compute_step() - feed the returned eta_Pn_pct/
    eta_Pint_pct/P_gen_ls_ch_off_kW+P_gen_ls_env_kW (summed, for
    P_gen_ls_P0_W) into a config by hand, the same way
    chimney_loss_pct_from_flue_gas_analysis()'s result would be.

    eta_Pint_pct (Eq.(8)) is only returned when beta_Pint (Eq.(68),
    Pint/Pn, the load factor/duty cycle at intermediate load) is given;
    P_cond_kW (Eq.(24), condensing_gain_fraction()*P_gen_del_kW) only
    matters for condensing boilers and defaults to 0.
    """

    P_gen_del = float(P_gen_del_kW)
    # Eq.(10)-(12): loss powers from the measured/derived loss fractions.
    P_ch_on = float(f_ch_on_pct) / 100.0 * P_gen_del
    P_ch_off = float(f_ch_off_pct) / 100.0 * P_gen_del
    P_env = float(f_gen_env_pct) / 100.0 * P_gen_del

    result: dict[str, float] = {
        "P_gen_ls_ch_on_kW": P_ch_on,
        "P_gen_ls_ch_off_kW": P_ch_off,
        "P_gen_ls_env_kW": P_env,
        # Eq.(9): the total stand-by loss fraction these two components add
        # up to (a consistency check when f_ch_off_pct/f_gen_env_pct came
        # from an independent measurement, not from split_standby_loss_pct()).
        "f_gen_ls_P0_pct": float(f_ch_off_pct) + float(f_gen_env_pct),
        # Eq.(7).
        "eta_Pn_pct": (P_gen_del - P_ch_on - P_env) / P_gen_del * 100.0 if P_gen_del > _EPS else float("nan"),
    }

    if beta_Pint is not None:
        b = float(beta_Pint)
        denom = P_gen_del * b
        if denom > _EPS:
            # Eq.(8): a time-weighted (duty-cycle) average between the
            # "on" phase (chimney-on loss, minus any condensation gain,
            # plus envelope loss) and the "off" phase (chimney-off loss
            # plus envelope loss - the envelope keeps losing heat
            # regardless of firing state).
            numerator = (
                P_gen_del * b
                - b * (P_ch_on - float(P_cond_kW) + P_env)
                + (1.0 - b) * (P_ch_off + P_env)
            )
            result["eta_Pint_pct"] = numerator / denom * 100.0
        else:
            result["eta_Pint_pct"] = float("nan")

    return result


def rated_power_from_measured_efficiency_kW(eta_gen_Pn_pct: float, P_gen_del_kW: float) -> float:
    """Eq.(14): Pn = eta_gen,Pn * P_gen,del - the generator's rated output,
    from measured full-load efficiency and the measured fuel power input,
    for use when Pn is not otherwise known."""

    return float(eta_gen_Pn_pct) / 100.0 * float(P_gen_del_kW)


# Table B.8 - f_brm (temperature reduction factor) by generator location,
# stored here as its Eq.(50)/(53) complement (1-f_brm) = f_location, so it
# can be used directly where f_location is needed. LOCATION_TYPES has no
# "under roof" option, so "adjacent_unheated" maps to Table B.8's "in the
# boiler room" (f_brm=0.3) as the closest match - a judgement call, not a
# literal table entry.
DEFAULT_F_LOCATION_BY_LOCATION = {
    "inside_heated": 1.0 - 0.0,
    "adjacent_unheated": 1.0 - 0.3,
    "outside_building": 1.0 - 1.0,
}

# Table B.7 - part of stand-by heat losses attributed to the generator
# envelope, by burner type.
DEFAULT_F_JACKET_BY_BOILER_TYPE = {
    "standard": 0.50,  # atmospheric burner, the more common "standard" case
    "low_temperature": 0.50,
    "condensing": 0.75,  # fan-assisted
    "biomass_log": 0.50,
    "biomass_pellet": 0.75,  # automatically-fed, fan-assisted
}

# Table B.16 - default f_ctrl,ls (Eq.58) by boiler/control type. Used only
# when a config supplies "control_type" (one of these keys) instead of
# f_ctrl_ls directly; f_ctrl_ls itself still defaults to 1.00 either way.
DEFAULT_F_CTRL_LS_BY_CONTROL_TYPE = {
    "floor_standing_outdoor_reset": 1.00,
    "wall_hung_outdoor_reset": 1.03,
    "wall_hung_room_control": 1.06,
}


@dataclass
class BoilerConfig:
    boiler_type: str
    fuel_type: str
    rated_power_kW: float
    intermediate_load_fraction: float
    eta_Pn_test_pct: float
    eta_Pint_test_pct: float
    theta_test_Pn_C: float
    theta_test_Pint_C: float
    f_corr_pct_per_K: float
    P_gen_ls_P0_W: float
    P_aux_Pn_W: float
    P_aux_Pint_W: float
    P_aux_P0_W: float
    f_jacket: float
    f_location: float
    f_aux_recoverable: float
    dew_point_C: float
    condensing_gain_pct: float
    # Eq.(43)/(45): f_Hs/Hi, on the SAME percent scale as efficiency. Default
    # 100.0 (Hs = Hi, no gross/net calorific-value correction) makes
    # (f_hs_hi_pct - eta_corr)/eta_corr reduce to (100-eta_corr)/eta_corr,
    # i.e. unchanged behaviour for anyone not setting it.
    f_hs_hi_pct: float = 100.0
    # Eq.(34)/(52): part of the auxiliary energy transmitted to the
    # distribution sub-system/fluid. Field default (in _load_parameters) is
    # the Eq.(51) complement of f_aux_recoverable, 1 - f_aux_recoverable -
    # this dataclass-level 0.0 only applies to direct BoilerConfig(...)
    # construction bypassing _load_parameters.
    f_aux_rvd: float = 0.0
    Q_gen_ren_kWh: float = 0.0
    # Eq.(58): control-strategy factor (Table B.16), 1.00 (floor-standing,
    # outdoor-reset control - no penalty) by default.
    f_ctrl_ls: float = 1.0
    # Clause 6.11 direct-heated DHW appliances (SIMPLE_DHW_GENERATOR_TYPES
    # only; unused - left at these defaults - for the combustion boiler
    # types above).
    f_tr_ls: float = 1.0  # Eq.(69)/Table B.17: instantaneous electrical water heater
    storage_volume_L: Optional[float] = None  # Eq.(70)-(76): gas-fired storage water heater
    q_B_S_kWh_per_day: Optional[float] = None  # Eq.(70): EN 89 (or Formula B.1/B.2) stand-by loss
    eta_100_pct: float = 84.0  # Eq.(74)-(75): EN 89 reference efficiency at the 100 % test point
    # Eq.(59)-(67): how a combined H+W demand that exceeds rated capacity is
    # shared between the two services. "proportional" (default, UNCHANGED
    # pre-existing behaviour) splits the shortfall by each service's share
    # of the combined request. "dhw_priority" implements the clause 6.10
    # sequential-priority method (see _compute_step_dhw_priority() and
    # docs/standards_errata.md for the Eq.(63)-(66) reinterpretation this
    # requires): DHW is served first, heating gets whatever capacity and
    # operating time is left, and each service gets its OWN load ratio
    # (beta_H/beta_W) for the loss/auxiliary-power interpolation instead of
    # one blended beta for the combined load.
    priority_mode: str = "proportional"
    # Eq.(15)-(24): opt-in first-principles combustion-chemistry condensing
    # gain (see condensing_gain_fraction()/_eta_corrected_pct()). None (the
    # default) keeps the pre-existing condensing_gain_pct/dew_point_C flat
    # approximation; set combustion_fuel_key (one of
    # FUEL_COMBUSTION_PROPERTIES_B13's keys) to use it instead.
    combustion_fuel_key: Optional[str] = None
    theta_brm_C: float = 20.0  # Table B.8 default by boiler_location (see _load_parameters)
    x_air_pct: float = DEFAULT_X_AIR_PCT
    x_fg_pct: float = DEFAULT_X_FG_PCT
    X_O2_fg_dry_pct: float = DEFAULT_X_O2_FG_DRY_PCT
    delta_theta_wfg_K: float = 10.0  # Eq.(15): default water-to-flue-gas temperature difference
    # Eq.(40): opt-in DHW trace-band full-load efficiency correction
    # (replaces Eq.(41) for eta_Pn only - see _eta_corrected_pct()). False
    # (the default) keeps Eq.(41)'s pre-existing circulation-loop formula.
    dhw_trace_band: bool = False
    # Eq.(46): opt-in temperature correction of the 0 %-load (stand-by)
    # generator loss P0 - see _loss_powers_kW(). False (the default) keeps
    # P_gen_ls_P0_W as a constant, pre-existing behaviour.
    apply_p0_temp_correction: bool = False
    theta_test_P0_C: float = 70.0  # Table B.3's theta_gen;test;P0 (typical value; override per product data)
    # Clause 6.12 (SIMPLE_DHW_GENERATOR_TYPES' "dhw_tapping_cycle_appliance"
    # only): the test_points/standby_point dhw_tapping_cycle_energy_kWh()
    # takes. Unused - left None - for every other boiler_type.
    tapping_cycle_test_points: Optional[list[dict[str, float]]] = None
    tapping_cycle_standby_point: Optional[dict[str, float]] = None
    # Tables B.4/B.5: the standard gives f_corr,Pn and f_corr,Pint as TWO
    # distinct coefficients per boiler type (e.g. standard/biomass boilers:
    # f_corr,Pn = 0, f_corr,Pint = 0.04 %/K - NOT the same value). None (the
    # default) for either falls back to the shared f_corr_pct_per_K field -
    # unchanged pre-existing behaviour for configs that only ever set that
    # one field. Set these to use the Table B.4/B.5 pair instead; see
    # _eta_corrected_pct().
    f_corr_pn_pct_per_K: Optional[float] = None
    f_corr_pint_pct_per_K: Optional[float] = None


@dataclass
class BoilerGeneratorSimulationResult:
    """Container returned by :class:`BoilerGeneratorCalculator.run_timeseries`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class BoilerGeneratorCalculator:
    """EN 15316-4-1 boiler generator calculator (gas/oil and biomass)."""

    def __init__(self, input_data: Optional[dict[str, Any]] = None):
        self.input_data = dict(input_data or {})
        self._load_parameters()

    def _require(self, key: str, default: Any = None) -> Any:
        if key in self.input_data:
            return self.input_data[key]
        if default is not None:
            return default
        raise ValueError(f"Missing required input parameter: {key}")

    def _load_parameters(self) -> None:
        inp = self.input_data

        btype = str(self._require("boiler_type")).lower()
        btype = GENERATOR_TYPE_ALIASES.get(btype, btype)
        if btype not in GENERATOR_TYPES:
            raise ValueError(f"Unknown boiler_type '{btype}'. Expected one of {sorted(GENERATOR_TYPES)}.")

        loc_key = str(self._require("boiler_location")).lower()
        if loc_key not in LOCATION_TYPES:
            raise ValueError(f"Unknown boiler_location '{loc_key}'. Expected one of {sorted(LOCATION_TYPES)}.")

        if btype in SIMPLE_DHW_GENERATOR_TYPES:
            # Clause 6.11 direct-heated DHW appliances use a completely
            # different, much simpler model than the P0/Pint/Pn combustion
            # loss curve below - none of eta_Pn_test_pct/theta_test_*_C/
            # f_corr_pct_per_K/efficiency_table/loss_table/dew_point_C/etc.
            # apply, so requiring them here would force callers to invent
            # meaningless values just to satisfy _require().
            self._load_simple_dhw_parameters(btype, loc_key)
            return

        # P_aux_Pn_W/P_aux_P0_W accept the legacy P_aux_on_W/P_aux_off_W names
        # (full-load and standby auxiliary power respectively); P_aux_Pint_W
        # is new and, if not given, defaults to the linear midpoint so that
        # existing configs keep working without having to supply it.
        # NOTE: dict.get(key, default_expr) always evaluates default_expr
        # eagerly, even when key is present - self._require(...) must only
        # run as a fallback, so check membership first rather than relying
        # on .get()'s default argument to short-circuit it.
        if "P_aux_Pn_W" in inp:
            p_aux_pn = float(inp["P_aux_Pn_W"])
        else:
            p_aux_pn = float(self._require("P_aux_on_W"))
        if "P_aux_P0_W" in inp:
            p_aux_p0 = float(inp["P_aux_P0_W"])
        else:
            p_aux_p0 = float(self._require("P_aux_off_W"))
        p_aux_pint = float(inp.get("P_aux_Pint_W", (p_aux_pn + p_aux_p0) / 2.0))

        # Eq.(51): f_aux,rbl = 1 - f_aux,rvd - the two fractions are NOT
        # independent, they split the SAME auxiliary energy between the
        # fluid (Eq.52, credited against fuel use) and the heated space
        # (Eq.53). f_aux_recoverable keeps its existing, documented meaning
        # (examples/new_building_input*.md: "share... recoverable in the
        # zone balance") as f_aux,rbl - the space side - and f_aux_rvd
        # DEFAULTS to its Eq.(51) complement so the two cannot be set to
        # inconsistent values (e.g. both close to 1, crediting the same
        # auxiliary energy twice) unless a caller deliberately overrides
        # f_aux_rvd to decouple them.
        f_aux_rbl = float(self._require("f_aux_recoverable"))
        f_aux_rvd = float(inp.get("f_aux_rvd", 1.0 - f_aux_rbl))
        # When a caller supplies BOTH values explicitly (deliberately
        # decoupling them from Eq.(51)'s default complement relationship),
        # nothing above stops them summing to something other than 1 -
        # which would either double-count the same auxiliary energy (sum
        # > 1) or silently lose some of it (sum < 1). Not a hard error
        # (independently measured f_aux,rvd/f_aux,rbl values that genuinely
        # don't sum to 1 are a legitimate, if unusual, input), but worth
        # flagging loudly rather than failing silently.
        if "f_aux_rvd" in inp and abs(f_aux_rvd + f_aux_rbl - 1.0) > 1e-6:
            warnings.warn(
                f"EN 15316-4-1: f_aux_rvd ({f_aux_rvd}) + f_aux_recoverable ({f_aux_rbl}) "
                f"= {f_aux_rvd + f_aux_rbl:.4g}, not 1.0. Eq.(51) defines f_aux,rbl = "
                "1 - f_aux,rvd: the two fractions split the SAME auxiliary energy between "
                "the fluid (Eq.52, credited against fuel use) and the heated space (Eq.53) "
                "- a sum other than 1 either double-counts it or loses some of it. Leave "
                "f_aux_rvd unset to derive it automatically, unless this divergence is "
                "deliberate (e.g. independently measured values).",
                UserWarning,
                stacklevel=2,
            )

        self.cfg = BoilerConfig(
            boiler_type=btype,
            fuel_type=str(self._require("fuel_type")).lower(),
            rated_power_kW=float(self._require("rated_power_kW")),
            intermediate_load_fraction=float(self._require("intermediate_load_fraction")),
            eta_Pn_test_pct=float(self._require("eta_Pn_test_pct")),
            eta_Pint_test_pct=float(self._require("eta_Pint_test_pct")),
            theta_test_Pn_C=float(self._require("theta_test_Pn_C")),
            theta_test_Pint_C=float(self._require("theta_test_Pint_C")),
            f_corr_pct_per_K=float(self._require("f_corr_pct_per_K")),
            f_corr_pn_pct_per_K=(
                float(inp["f_corr_pn_pct_per_K"]) if "f_corr_pn_pct_per_K" in inp else None
            ),
            f_corr_pint_pct_per_K=(
                float(inp["f_corr_pint_pct_per_K"]) if "f_corr_pint_pct_per_K" in inp else None
            ),
            P_gen_ls_P0_W=float(self._require("P_gen_ls_P0_W")),
            P_aux_Pn_W=p_aux_pn,
            P_aux_Pint_W=p_aux_pint,
            P_aux_P0_W=p_aux_p0,
            # Table B.7/B.8 defaults from boiler_type/boiler_location when not
            # given explicitly (boiler_location was previously validated but
            # otherwise unused in any calculation).
            f_jacket=float(inp.get("f_jacket", DEFAULT_F_JACKET_BY_BOILER_TYPE[btype])),
            f_location=float(inp.get("f_location", DEFAULT_F_LOCATION_BY_LOCATION[loc_key])),
            f_aux_recoverable=f_aux_rbl,
            dew_point_C=float(self._require("dew_point_C")),
            condensing_gain_pct=float(self._require("condensing_gain_pct")),
            # Table B.9: no combustion fuel has Hs/Hi = 1.00 - DEFAULT from
            # fuel_type (an explicit f_hs_hi_pct always overrides this).
            f_hs_hi_pct=float(inp.get(
                "f_hs_hi_pct",
                DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE.get(str(inp.get("fuel_type", "")).lower(), 100.0),
            )),
            f_aux_rvd=f_aux_rvd,
            Q_gen_ren_kWh=float(inp.get("Q_gen_ren_kWh", 0.0)),
            f_ctrl_ls=float(inp.get(
                "f_ctrl_ls",
                DEFAULT_F_CTRL_LS_BY_CONTROL_TYPE.get(str(inp.get("control_type", "")).lower(), 1.0),
            )),
            priority_mode=str(inp.get("priority_mode", "proportional")).lower(),
            combustion_fuel_key=inp.get("combustion_fuel_key"),
            theta_brm_C=float(inp.get("theta_brm_C", DEFAULT_THETA_BRM_C_BY_LOCATION[loc_key])),
            x_air_pct=float(inp.get("x_air_pct", DEFAULT_X_AIR_PCT)),
            x_fg_pct=float(inp.get("x_fg_pct", DEFAULT_X_FG_PCT)),
            X_O2_fg_dry_pct=float(inp.get("X_O2_fg_dry_pct", DEFAULT_X_O2_FG_DRY_PCT)),
            delta_theta_wfg_K=float(inp.get("delta_theta_wfg_K", 10.0)),
            dhw_trace_band=bool(inp.get("dhw_trace_band", False)),
            apply_p0_temp_correction=bool(inp.get("apply_p0_temp_correction", False)),
            theta_test_P0_C=float(inp.get("theta_test_P0_C", 70.0)),
        )

        self.eta_table = self._load_case_table("efficiency_table")
        self.loss_table = self._load_case_table("loss_table")

        assert 0.0 < self.cfg.rated_power_kW
        assert 0.0 < self.cfg.intermediate_load_fraction < 1.0
        assert 0.0 <= self.cfg.f_jacket <= 1.0
        assert 0.0 <= self.cfg.f_location <= 1.0
        assert 0.0 <= self.cfg.f_aux_recoverable <= 1.0
        assert 0.0 <= self.cfg.f_aux_rvd <= 1.0
        assert self.cfg.Q_gen_ren_kWh >= 0.0
        assert self.cfg.f_ctrl_ls >= 1.0  # Table B.16: a control penalty, never a credit
        assert self.cfg.f_hs_hi_pct > 0.0
        if self.cfg.priority_mode not in PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{self.cfg.priority_mode}'. Expected one of {sorted(PRIORITY_MODES)}.")
        if self.cfg.combustion_fuel_key is not None and self.cfg.combustion_fuel_key not in FUEL_COMBUSTION_PROPERTIES_B13:
            raise ValueError(
                f"No curated Table B.13 combustion data for combustion_fuel_key "
                f"{self.cfg.combustion_fuel_key!r}. Available: {sorted(FUEL_COMBUSTION_PROPERTIES_B13)}."
            )

    def _load_simple_dhw_parameters(self, btype: str, loc_key: str) -> None:
        """Clause 6.11/6.12 direct-heated DHW appliances
        (electric_instantaneous, gas_storage_water_heater,
        dhw_tapping_cycle_appliance) - see SIMPLE_DHW_GENERATOR_TYPES."""

        inp = self.input_data
        f_tr_ls = 1.0
        storage_volume_L: Optional[float] = None
        q_B_S_kWh_per_day: Optional[float] = None
        eta_100_pct = 84.0
        tapping_cycle_test_points: Optional[list[dict[str, float]]] = None
        tapping_cycle_standby_point: Optional[dict[str, float]] = None

        if btype == "electric_instantaneous":
            # Table B.17: f_tr,ls by installation era.
            era = str(inp.get("installation_era", "since_1980")).lower()
            f_tr_ls = float(inp.get("f_tr_ls", DEFAULT_F_TR_LS_BY_ERA.get(era, 1.0)))
            rated_power_kW = float(self._require("rated_power_kW"))
        elif btype == "dhw_tapping_cycle_appliance":
            tapping_cycle_test_points = self._require("tapping_cycle_test_points")
            if not isinstance(tapping_cycle_test_points, list) or not tapping_cycle_test_points:
                raise ValueError(
                    "dhw_tapping_cycle_appliance needs tapping_cycle_test_points "
                    "(a non-empty list - see dhw_tapping_cycle_energy_kWh())."
                )
            tapping_cycle_standby_point = self._require("tapping_cycle_standby_point")
            if not isinstance(tapping_cycle_standby_point, dict):
                raise ValueError("dhw_tapping_cycle_appliance needs tapping_cycle_standby_point (a dict).")
            # Clause 6.12 has no rated-power/capacity-cap concept of its
            # own - demand above the highest tested cycle is EXTRAPOLATED
            # (Eq.(77)/(80)), not capped. A very large sentinel keeps
            # compute_step()/run_timeseries()'s shared capacity-cap/unmet-
            # energy machinery a no-op for this boiler_type, unless a
            # caller deliberately supplies a real rated_power_kW to cap it.
            rated_power_kW = float(inp.get("rated_power_kW", 1.0e6))
        else:  # "gas_storage_water_heater"
            if "storage_volume_L" in inp:
                storage_volume_L = float(inp["storage_volume_L"])
            if "q_B_S_kWh_per_day" in inp:
                # Measured per EN 89 - takes precedence over the Formula
                # (B.1)/(B.2) default below.
                q_B_S_kWh_per_day = float(inp["q_B_S_kWh_per_day"])
            elif storage_volume_L is not None:
                q_B_S_kWh_per_day = gas_storage_water_heater_standby_loss_default_kWh_per_day(
                    storage_volume_L, bool(inp.get("storage_installed_before_1985", False))
                )
            else:
                raise ValueError(
                    "gas_storage_water_heater needs either q_B_S_kWh_per_day "
                    "(measured per EN 89) or storage_volume_L (Formula B.1/B.2 default)."
                )
            eta_100_pct = float(inp.get("eta_100_pct", 84.0))  # EN 89 default (6.11.2)
            if "rated_power_kW" in inp:
                rated_power_kW = float(inp["rated_power_kW"])
            elif storage_volume_L is not None:
                # Eq.(76): default rated output from storage volume.
                rated_power_kW = direct_dhw_generator_rated_power_default_kW(storage_volume_L)
            else:
                raise ValueError("gas_storage_water_heater needs rated_power_kW or storage_volume_L (Eq.76).")

        self.cfg = BoilerConfig(
            boiler_type=btype,
            fuel_type=str(self._require("fuel_type")).lower(),
            rated_power_kW=rated_power_kW,
            # The fields below are all unused by SIMPLE_DHW_GENERATOR_TYPES'
            # compute_step() branch (see module docstring) - filled with
            # inert placeholders so BoilerConfig stays a single dataclass
            # shared by every boiler_type.
            intermediate_load_fraction=0.5,
            eta_Pn_test_pct=100.0, eta_Pint_test_pct=100.0,
            theta_test_Pn_C=45.0, theta_test_Pint_C=45.0,
            f_corr_pct_per_K=0.0,
            P_gen_ls_P0_W=0.0,
            P_aux_Pn_W=0.0, P_aux_Pint_W=0.0, P_aux_P0_W=0.0,
            f_jacket=0.0,
            f_location=float(DEFAULT_F_LOCATION_BY_LOCATION[loc_key]),
            f_aux_recoverable=0.0,
            dew_point_C=55.0, condensing_gain_pct=0.0,
            # Table B.9 (relevant for gas_storage_water_heater's Eq.(75) -
            # unused by electric_instantaneous/dhw_tapping_cycle_appliance).
            f_hs_hi_pct=float(inp.get(
                "f_hs_hi_pct",
                DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE.get(str(inp.get("fuel_type", "")).lower(), 100.0),
            )),
            f_aux_rvd=0.0,
            Q_gen_ren_kWh=float(inp.get("Q_gen_ren_kWh", 0.0)),
            # Clause 6.11 appliances are direct, non-modulating heaters - the
            # standard defines no control-strategy factor for them.
            f_ctrl_ls=1.0,
            priority_mode="proportional",  # unused: clause 6.11 appliances are DHW-only, nothing to prioritise
            f_tr_ls=f_tr_ls,
            storage_volume_L=storage_volume_L,
            q_B_S_kWh_per_day=q_B_S_kWh_per_day,
            eta_100_pct=eta_100_pct,
            tapping_cycle_test_points=tapping_cycle_test_points,
            tapping_cycle_standby_point=tapping_cycle_standby_point,
        )
        self.eta_table = {}
        self.loss_table = {}

        assert 0.0 < self.cfg.rated_power_kW
        assert self.cfg.f_tr_ls > 0.0
        assert self.cfg.f_hs_hi_pct > 0.0
        assert self.cfg.Q_gen_ren_kWh >= 0.0
        if btype == "gas_storage_water_heater":
            assert self.cfg.q_B_S_kWh_per_day is not None and self.cfg.q_B_S_kWh_per_day >= 0.0
            assert self.cfg.eta_100_pct > 0.0

    def _load_case_table(self, key: str) -> dict[str, Any]:
        table = self._require(key)
        if not isinstance(table, dict):
            raise ValueError(f"{key} must be a dict with case definitions.")
        return table

    def _case_values(self, table: dict[str, Any], fallback_key: str | None = None) -> dict[str, float]:
        case = table.get(self.cfg.boiler_type)
        if case is None and fallback_key is not None:
            case = table.get(fallback_key)
        if case is None:
            raise ValueError(f"Missing case data for boiler_type '{self.cfg.boiler_type}'.")
        if not isinstance(case, dict):
            raise ValueError(f"Case data for '{self.cfg.boiler_type}' must be a dict.")
        return case

    def _f_corr_pct_per_K(self, is_full_load: bool) -> float:
        """Tables B.4/B.5: f_corr,Pn and f_corr,Pint are two DISTINCT
        coefficients (e.g. standard/biomass boilers: f_corr,Pn = 0,
        f_corr,Pint = 0.04 %/K). f_corr_pn_pct_per_K/f_corr_pint_pct_per_K
        override the shared f_corr_pct_per_K for the full-load/intermediate-
        load correction respectively when set; otherwise both fall back to
        f_corr_pct_per_K - unchanged behaviour for configs that only ever
        set that one shared field."""

        specific = self.cfg.f_corr_pn_pct_per_K if is_full_load else self.cfg.f_corr_pint_pct_per_K
        return self.cfg.f_corr_pct_per_K if specific is None else specific

    def _eta_corrected_pct(self, eta_test_pct: float, theta_test_C: float, theta_avg_C: float,
                           theta_return_C: Optional[float], is_full_load: bool = False) -> float:
        f_corr = self._f_corr_pct_per_K(is_full_load)
        if is_full_load and self.cfg.dhw_trace_band:
            # Eq.(40): when DHW uses a trace band instead of circulation,
            # this REPLACES Eq.(41) for the full-load efficiency (the
            # standard's own text: "insert eta_gen;Pn;traceband instead of
            # eta_gen;Pn;corr in Formula (43)") - theta_avg_C is reused here
            # as theta_Wc;mn (mean DHW water temperature), the same way
            # this module already reuses it for other boiler-type-specific
            # meanings elsewhere (e.g. gas_storage_water_heater).
            eta = float(eta_test_pct) + f_corr * (50.0 - float(theta_avg_C))
        else:
            # Eq.(41)/(44): temperature-corrected efficiency.
            eta = float(eta_test_pct) - f_corr * (float(theta_avg_C) - float(theta_test_C))
        if self.cfg.boiler_type == "condensing" and theta_return_C is not None:
            tr = float(theta_return_C)
            if self.cfg.combustion_fuel_key is not None:
                # Eq.(15)-(24): first-principles combustion-chemistry
                # condensing gain (flue gas temperature/composition,
                # humidity of combustion air and flue gas, condensing water
                # mass, latent heat - Table B.10/B.12/B.13/B.14 defaults),
                # preferred over the flat condensing_gain_pct approximation
                # below whenever fuel combustion data is available. Only
                # used when the case data does NOT also provide the
                # Eq.(42) 60/30 degC test pair (see _eta_Pn_corrected_pct) -
                # Eq.(42), being measured data, is preferred over either.
                eta += condensing_gain_fraction(
                    self.cfg.combustion_fuel_key, tr, self.cfg.theta_brm_C,
                    self.cfg.delta_theta_wfg_K, self.cfg.x_air_pct, self.cfg.x_fg_pct,
                    self.cfg.X_O2_fg_dry_pct,
                ) * 100.0
            elif tr < self.cfg.dew_point_C:
                # Simplified stand-in for Eq.(15)-(24): scales the
                # condensing gain by how far the return temperature sits
                # below the dew point, instead of deriving it from flue gas
                # composition and humidity. Used only when combustion_fuel_
                # key is NOT set (see above).
                ratio = max(0.0, min(1.0, (self.cfg.dew_point_C - tr) / max(self.cfg.dew_point_C, _EPS)))
                eta += self.cfg.condensing_gain_pct * ratio
        return float(max(min(eta, 110.0), 1.0))

    def _eta_Pn_corrected_pct(self, eff_case: dict[str, float], theta_avg_C: float,
                              theta_return_C: Optional[float]) -> float:
        """Full-load corrected efficiency. Eq.(42) for condensing boilers
        when the case data gives both reference points it needs (efficiency
        measured at a 60 degC AND a 30 degC average return temperature);
        otherwise Eq.(41) plus the dew-point approximation above."""

        eta_60 = eff_case.get("eta_Pn_test_60_pct")
        eta_30 = eff_case.get("eta_Pn_test_30_pct")
        if self.cfg.boiler_type == "condensing" and eta_60 is not None and eta_30 is not None and theta_return_C is not None:
            # Eq.(42): linear interpolation/extrapolation between the two
            # measured reference points by the ACTUAL return temperature -
            # the standard's own condensing-efficiency method, not a stand-in.
            # eta = eta_60 - [(eta_60-eta_30)/(theta_60-theta_30)] * (theta_60 - theta_RT)
            theta_test_60, theta_test_30 = 60.0, 30.0
            slope = (float(eta_60) - float(eta_30)) / (theta_test_60 - theta_test_30)
            eta = float(eta_60) - slope * (theta_test_60 - float(theta_return_C))
            return float(max(min(eta, 110.0), 1.0))
        return self._eta_corrected_pct(
            eff_case["eta_Pn_test_pct"], eff_case["theta_test_Pn_C"], theta_avg_C, theta_return_C,
            is_full_load=True,
        )

    def _eta_triplet(self, theta_avg_C: float, theta_return_C: Optional[float]) -> dict[str, float]:
        eff_case = self._case_values(self.eta_table)
        return {
            "eta_Pn_corr(%)": self._eta_Pn_corrected_pct(eff_case, theta_avg_C, theta_return_C),
            "eta_Pint_corr(%)": self._eta_corrected_pct(
                eff_case["eta_Pint_test_pct"], eff_case["theta_test_Pint_C"], theta_avg_C, theta_return_C
            ),
        }

    def _loss_powers_kW(self, theta_avg_C: float, theta_return_C: Optional[float]) -> dict[str, float]:
        triplet = self._eta_triplet(theta_avg_C, theta_return_C)
        case = self._case_values(self.loss_table)
        Pn = self.cfg.rated_power_kW
        Pint = self.cfg.intermediate_load_fraction * Pn
        # Eq.(43)/(45): P_ls = [(f_Hs/Hi - eta_corr) / eta_corr] * P, exactly
        # as the standard writes it (both terms on the same percent scale).
        # f_hs_hi_pct defaults to 100 (Hs = Hi), which reduces this to
        # (100-eta_corr)/eta_corr * P - unchanged behaviour by default.
        f = self.cfg.f_hs_hi_pct
        P_gen_ls_P0_kW = float(case["P_gen_ls_P0_W"]) / 1000.0
        if self.cfg.apply_p0_temp_correction:
            # Eq.(46): P_H;gen;ls;P0;corr = (Pn/eta_gen;Pn)*f_gen;ls;P0*
            # f_Hs/Hi * [(theta_Hc;mn-theta_brm)/(theta_gen;test;P0-20)]^1.25.
            # P_gen_ls_P0_W is already taken to BE (Pn/eta_gen;Pn)*f_gen;ls;
            # P0*f_Hs/Hi evaluated AT TEST conditions (where the bracket is
            # 1, by construction - theta_avg_C=theta_test_P0_C and
            # theta_brm_C=20), so only the bracket ratio, raised to 1.25,
            # needs applying here. Off by default (apply_p0_temp_correction
            # =False): the pre-existing constant-P0 behaviour, unchanged.
            ratio = max(
                (float(theta_avg_C) - self.cfg.theta_brm_C) / max(self.cfg.theta_test_P0_C - 20.0, _EPS),
                0.0,
            )
            P_gen_ls_P0_kW *= ratio ** 1.25
        return {
            **triplet,
            "P_gen_ls_Pn(kW)": Pn * max((f - triplet["eta_Pn_corr(%)"]), 0.0) / triplet["eta_Pn_corr(%)"],
            "P_gen_ls_Pint(kW)": Pint * max((f - triplet["eta_Pint_corr(%)"]), 0.0) / triplet["eta_Pint_corr(%)"],
            "P_gen_ls_P0(kW)": P_gen_ls_P0_kW,
        }

    def _interp_at_load(self, beta: float, v_pn: float, v_pint: float, v_p0: float) -> float:
        """Eq.(37)-(38) / Eq.(55)-(56) shared form: linear interpolation of a
        quantity known at 0 %, intermediate and full load, by the actual load
        ratio beta."""

        beta = max(min(float(beta), 1.0), 0.0)
        b_int = self.cfg.intermediate_load_fraction
        if beta <= _EPS:
            return v_p0
        if beta <= b_int:
            return v_p0 + (beta / b_int) * (v_pint - v_p0)
        return v_pint + ((beta - b_int) / max(1.0 - b_int, _EPS)) * (v_pn - v_pint)

    def _auxiliary_energy_kWh(self, beta: float, t_run_h: float, t_use_h: float) -> dict[str, float]:
        t_off = max(float(t_use_h) - float(t_run_h), 0.0)
        # Eq.(55)-(56): auxiliary power while running is interpolated by load
        # ratio, the same way as the thermal loss - NOT a flat value
        # regardless of how hard the generator is working.
        p_aux_run_kW = self._interp_at_load(
            beta, self.cfg.P_aux_Pn_W / 1000.0, self.cfg.P_aux_Pint_W / 1000.0, self.cfg.P_aux_P0_W / 1000.0
        )
        W_on = p_aux_run_kW * float(t_run_h)
        W_off = (self.cfg.P_aux_P0_W / 1000.0) * t_off
        return {
            "W_gen_aux_on_kWh": W_on,
            "W_gen_aux_off_kWh": W_off,
            "W_gen_aux_kWh": W_on + W_off,
        }

    def _capped_demand(
        self,
        Q_gen_out_kWh: float,
        t_use: float,
        Q_H_gen_out_kWh: Optional[float],
        Q_W_gen_out_kWh: Optional[float],
    ) -> tuple[float, float, float, Optional[float], Optional[float]]:
        """Eq.(58) + capacity cap, shared by every boiler_type's compute_step
        branch. Returns (Q_req, Q_out, Q_unmet, qh_req_raw, qw_req_raw)."""

        Pn = self.cfg.rated_power_kW
        # Eq.(58): Q_gen,out = f_ctrl,ls * sum(Q_H,dis,in) + sum(Q_XY,dis,in) -
        # the control-strategy factor (Table B.16: 1.00 outdoor-reset floor-
        # standing, up to 1.06 room-thermostat wall-hung) inflates ONLY the
        # heating duty the generator must produce, not DHW/cooling. Can only
        # be applied when the H/W split is known; with just a combined
        # Q_gen_out_kWh there is no heating-only amount to scale. f_ctrl_ls
        # is forced to 1.0 for SIMPLE_DHW_GENERATOR_TYPES (see
        # _load_simple_dhw_parameters), so this is a no-op for them.
        qh_req_raw = qw_req_raw = None
        if (Q_H_gen_out_kWh is not None) or (Q_W_gen_out_kWh is not None):
            qh_req_raw = max(float(Q_H_gen_out_kWh or 0.0), 0.0)
            qw_req_raw = max(float(Q_W_gen_out_kWh or 0.0), 0.0)
            Q_req = qh_req_raw * self.cfg.f_ctrl_ls + qw_req_raw
        else:
            Q_req = max(float(Q_gen_out_kWh), 0.0)

        # Capacity cap: demand beyond the rated power is unmet, consistent
        # with the heat-pump/district/cogeneration generation modules.
        Q_out = min(Q_req, Pn * t_use)
        Q_unmet = max(Q_req - Q_out, 0.0)
        return Q_req, Q_out, Q_unmet, qh_req_raw, qw_req_raw

    def _split_h_w(
        self,
        Q_out: float,
        Q_req: float,
        Q_unmet: float,
        qh_req_raw: Optional[float],
        qw_req_raw: Optional[float],
        E_gen_in: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Returns (qh_out, qw_out, Q_H_unmet, Q_W_unmet, E_H_gen_in, E_W_gen_in)."""

        if qh_req_raw is not None:
            # Two different splits on purpose: fuel is attributed by the
            # f_ctrl,ls-INFLATED heating share (heating's own control losses
            # are why the extra fuel is burned), but what's reported as
            # served/unmet to the distribution system uses the RAW
            # (un-inflated) requested shares, scaled by whatever fraction of
            # the (possibly capacity-capped) total duty was actually
            # achieved - so a caller comparing qh_out/Q_H_unmet against its
            # own un-inflated request sees a consistent balance, and at
            # f_ctrl_ls=1.0 (the default) this is byte-identical to before.
            qh_req_ctrl = qh_req_raw * self.cfg.f_ctrl_ls
            qsum_ctrl = qh_req_ctrl + qw_req_raw
            share_h_ctrl = qh_req_ctrl / qsum_ctrl if qsum_ctrl > _EPS else 0.0
            share_w_ctrl = qw_req_raw / qsum_ctrl if qsum_ctrl > _EPS else 0.0
            E_H_gen_in = E_gen_in * share_h_ctrl
            E_W_gen_in = E_gen_in * share_w_ctrl

            served_fraction = Q_out / Q_req if Q_req > _EPS else 0.0
            qh_out, qw_out = qh_req_raw * served_fraction, qw_req_raw * served_fraction
            Q_H_unmet = qh_req_raw - qh_out
            Q_W_unmet = qw_req_raw - qw_out
        else:
            qh_out, qw_out = Q_out, 0.0
            E_H_gen_in, E_W_gen_in = E_gen_in, 0.0
            Q_H_unmet, Q_W_unmet = Q_unmet, 0.0
        return qh_out, qw_out, Q_H_unmet, Q_W_unmet, E_H_gen_in, E_W_gen_in

    def compute_step(
        self,
        Q_gen_out_kWh: float,
        t_use_h: float = 1.0,
        theta_avg_C: float = 45.0,
        theta_return_C: Optional[float] = None,
        Q_H_gen_out_kWh: Optional[float] = None,
        Q_W_gen_out_kWh: Optional[float] = None,
    ) -> dict[str, float]:
        if self.cfg.boiler_type == "electric_instantaneous":
            return self._compute_step_electric_instantaneous(
                Q_gen_out_kWh, t_use_h, Q_H_gen_out_kWh, Q_W_gen_out_kWh
            )
        if self.cfg.boiler_type == "gas_storage_water_heater":
            return self._compute_step_gas_storage_water_heater(
                Q_gen_out_kWh, t_use_h, theta_avg_C, theta_return_C, Q_H_gen_out_kWh, Q_W_gen_out_kWh
            )
        if self.cfg.boiler_type == "dhw_tapping_cycle_appliance":
            # Clause 6.12's Eq.(77)-(82) are a DAILY method (one data point
            # per calendar day, from a day's TOTAL demand) - a single hourly
            # step has no meaning on its own. Use run_timeseries() (which
            # aggregates by calendar day and distributes the result back
            # across that day's hours) or call dhw_tapping_cycle_energy_kWh()
            # directly for a single day's total.
            raise NotImplementedError(
                "dhw_tapping_cycle_appliance has no single-step meaning (clause "
                "6.12 is a daily method) - call run_timeseries() with a "
                "DatetimeIndex df, or dhw_tapping_cycle_energy_kWh() directly."
            )
        if self.cfg.priority_mode == "dhw_priority" and (
            (Q_H_gen_out_kWh is not None) or (Q_W_gen_out_kWh is not None)
        ):
            return self._compute_step_dhw_priority(
                Q_H_gen_out_kWh, Q_W_gen_out_kWh, t_use_h, theta_avg_C, theta_return_C
            )

        t_use = max(float(t_use_h), _EPS)
        Q_req, Q_out, Q_unmet, qh_req_raw, qw_req_raw = self._capped_demand(
            Q_gen_out_kWh, t_use, Q_H_gen_out_kWh, Q_W_gen_out_kWh
        )

        Pn = self.cfg.rated_power_kW
        triplet = self._loss_powers_kW(theta_avg_C, theta_return_C)
        beta = min((Q_out / t_use) / max(Pn, _EPS), 1.0)
        P_gen_ls_beta = self._interp_at_load(
            beta, triplet["P_gen_ls_Pn(kW)"], triplet["P_gen_ls_Pint(kW)"], triplet["P_gen_ls_P0(kW)"]
        )

        t_run = min(Q_out / max(Pn, _EPS), t_use)
        t_off = t_use - t_run

        Q_gen_ls_on = P_gen_ls_beta * t_run
        Q_gen_ls_off = triplet["P_gen_ls_P0(kW)"] * t_off
        Q_gen_ls = Q_gen_ls_on + Q_gen_ls_off

        aux = self._auxiliary_energy_kWh(beta, t_run, t_use)
        # Eq.(34): E_gen,in = Q_gen,out - Q_gen,aux,rvd + Q_gen,ls - Q_gen,ren.
        # Auxiliary energy recovered INTO the fluid (Eq.52) offsets fuel use;
        # Q_gen,ren (e.g. a solar/renewable contribution upstream of the
        # boiler) does too. Q_gen_ren_kWh defaults to 0 - the standard itself
        # notes it "is normally zero" for a boiler. f_aux_rvd defaults to the
        # Eq.(51) complement of f_aux_recoverable (see _load_parameters), so
        # this credit is applied by default whenever f_aux_recoverable is set.
        Q_gen_aux_rvd = aux["W_gen_aux_kWh"] * self.cfg.f_aux_rvd
        E_gen_in = max(Q_out - Q_gen_aux_rvd + Q_gen_ls - self.cfg.Q_gen_ren_kWh, 0.0)
        e_gen = (E_gen_in / Q_out) if Q_out > _EPS else float("nan")

        # Eq.(50): the recoverable envelope (jacket) loss is the STANDBY loss
        # RATE (P0, roughly constant regardless of firing state) times the
        # envelope fraction, over the WHOLE period of use - not the combined
        # on+off total, which would wrongly fold in flue-gas/combustion
        # losses that leave through the chimney, not the jacket.
        Q_gen_ls_env = triplet["P_gen_ls_P0(kW)"] * self.cfg.f_jacket * t_use
        Q_gen_ls_rbl = Q_gen_ls_env * self.cfg.f_location
        # Eq.(53): Q_H,gen,aux,rbl = (1-f_brm) * W_H,gen * f_aux,rbl. Table B.8
        # gives f_brm by generator location (0 inside the heated space, up to
        # 1 outdoors) - the SAME (1-f_brm) factor Eq.(50) already applies to
        # the envelope loss above, via f_location (not an independent input:
        # f_location IS (1-f_brm), e.g. 1.0 for "inside_heated" in this
        # package's examples/catalog, matching Table B.8's f_brm=0 there).
        W_gen_aux_rbl = aux["W_gen_aux_kWh"] * self.cfg.f_aux_recoverable * self.cfg.f_location

        qh_out, qw_out, Q_H_unmet, Q_W_unmet, E_H_gen_in, E_W_gen_in = self._split_h_w(
            Q_out, Q_req, Q_unmet, qh_req_raw, qw_req_raw, E_gen_in
        )

        return {
            # Original EN 15316-4-1 key names (parenthesis-suffixed units), kept
            # EXACTLY as before: iso_15316_1.py's HeatingSystemCalculator reads
            # these by name (its own "(kWh)"-style convention, not this
            # package's usual "_kWh" suffix - do not rename them without also
            # auditing every `_gen_value(...)` call there).
            "Q_gen_out(kWh)": Q_out,
            "t_use(h)": t_use,
            "theta_avg(°C)": float(theta_avg_C),
            "theta_return(°C)": float(theta_return_C) if theta_return_C is not None else float("nan"),
            "beta_gen(-)": beta,
            "t_run(h)": t_run,
            "t_off(h)": t_off,
            **triplet,
            "P_gen_ls_beta(kW)": P_gen_ls_beta,
            "Q_gen_ls_on(kWh)": Q_gen_ls_on,
            "Q_gen_ls_off(kWh)": Q_gen_ls_off,
            "Q_gen_ls(kWh)": Q_gen_ls,
            "E_gen_in(kWh)": E_gen_in,
            "e_gen(-)": e_gen,
            "E_H_gen_in(kWh)": E_H_gen_in,
            "E_W_gen_in(kWh)": E_W_gen_in,
            "W_gen_aux_on(kWh)": aux["W_gen_aux_on_kWh"],
            "W_gen_aux_off(kWh)": aux["W_gen_aux_off_kWh"],
            "W_gen_aux(kWh)": aux["W_gen_aux_kWh"],
            "Q_gen_ls_env(kWh)": Q_gen_ls_env,
            "Q_gen_ls_rbl(kWh)": Q_gen_ls_rbl,
            "W_gen_aux_rbl(kWh)": W_gen_aux_rbl,
            # New keys (capacity cap + unmet energy, Eq.55-56 load-dependent
            # auxiliary power): no legacy consumer reads these names, so they
            # use this package's regular "_kWh" convention.
            "Q_gen_out_req_kWh": Q_req,
            "Q_gen_unmet_kWh": Q_unmet,
            "Q_H_gen_out_kWh": qh_out,
            "Q_W_gen_out_kWh": qw_out,
            "Q_H_gen_unmet_kWh": Q_H_unmet,
            "Q_W_gen_unmet_kWh": Q_W_unmet,
            "Q_gen_aux_rvd_kWh": Q_gen_aux_rvd,
            "Q_gen_aux_rbl_kWh": W_gen_aux_rbl,
        }

    def _compute_step_dhw_priority(
        self,
        Q_H_gen_out_kWh: Optional[float],
        Q_W_gen_out_kWh: Optional[float],
        t_use_h: float,
        theta_avg_C: float,
        theta_return_C: Optional[float],
    ) -> dict[str, float]:
        """Clause 6.10 (Eq.(59)-(67)), DHW-priority sequential operation -
        opt-in via priority_mode="dhw_priority" (BoilerConfig.priority_mode).
        See docs/standards_errata.md for why Eq.(63)-(66) are reinterpreted
        here rather than implemented literally (as printed, they reduce to
        beta=1 identically, for any demand - see that file for the algebra).

        DHW is served FIRST out of the step's total capacity (Pn * t_use);
        heating gets whatever capacity and operating time remain - not a
        proportional split of the shortfall (the "proportional" default
        mode's behaviour). Each service gets its OWN load ratio (beta_H,
        beta_W - Q_X,gen,out / (Pn * t_use), the same form this module
        already uses for the single combined beta elsewhere, applied per
        service here instead) for the loss/auxiliary-power interpolation,
        instead of one blended beta for the combined load.
        """

        t_use = max(float(t_use_h), _EPS)
        Pn = self.cfg.rated_power_kW
        total_capacity = Pn * t_use

        qh_req_raw = max(float(Q_H_gen_out_kWh or 0.0), 0.0)
        qw_req_raw = max(float(Q_W_gen_out_kWh or 0.0), 0.0)
        # Eq.(58) still inflates ONLY the heating duty (see compute_step's
        # "proportional" branch for the same rule).
        qh_req_ctrl = qh_req_raw * self.cfg.f_ctrl_ls
        Q_req = qh_req_ctrl + qw_req_raw

        # Eq.(62)/(59), sequential priority: DHW is allocated capacity
        # FIRST; heating gets whatever remains.
        qw_out_ctrl = min(qw_req_raw, total_capacity)
        qh_out_ctrl = min(qh_req_ctrl, max(total_capacity - qw_out_ctrl, 0.0))
        Q_out = qw_out_ctrl + qh_out_ctrl
        Q_unmet = max(Q_req - Q_out, 0.0)

        # Reported/served amounts use RAW (un-inflated) heating, the same
        # convention _split_h_w uses in the "proportional" path.
        qw_out = qw_out_ctrl
        qh_out = qh_out_ctrl / self.cfg.f_ctrl_ls if self.cfg.f_ctrl_ls > _EPS else qh_out_ctrl
        Q_H_unmet = max(qh_req_raw - qh_out, 0.0)
        Q_W_unmet = max(qw_req_raw - qw_out, 0.0)

        # Eq.(59)/(62), corrected: time running at full nominal power needed
        # for each service's ACTUAL (capacity-capped) output.
        t_W_op = qw_out_ctrl / max(Pn, _EPS)
        t_H_op = qh_out_ctrl / max(Pn, _EPS)
        t_off = max(t_use - t_W_op - t_H_op, 0.0)

        # Eq.(63)-(66), corrected (docs/standards_errata.md): each service's
        # OWN load ratio against the full step, not a blended combined beta.
        beta_W = min(qw_out_ctrl / max(total_capacity, _EPS), 1.0)
        beta_H = min(qh_out_ctrl / max(total_capacity, _EPS), 1.0)

        triplet = self._loss_powers_kW(theta_avg_C, theta_return_C)
        P_gen_ls_W = self._interp_at_load(
            beta_W, triplet["P_gen_ls_Pn(kW)"], triplet["P_gen_ls_Pint(kW)"], triplet["P_gen_ls_P0(kW)"]
        )
        P_gen_ls_H = self._interp_at_load(
            beta_H, triplet["P_gen_ls_Pn(kW)"], triplet["P_gen_ls_Pint(kW)"], triplet["P_gen_ls_P0(kW)"]
        )
        Q_gen_ls_W_on = P_gen_ls_W * t_W_op
        Q_gen_ls_H_on = P_gen_ls_H * t_H_op
        Q_gen_ls_off = triplet["P_gen_ls_P0(kW)"] * t_off
        Q_gen_ls_on = Q_gen_ls_W_on + Q_gen_ls_H_on
        Q_gen_ls = Q_gen_ls_on + Q_gen_ls_off

        p_aux_W = self._interp_at_load(
            beta_W, self.cfg.P_aux_Pn_W / 1000.0, self.cfg.P_aux_Pint_W / 1000.0, self.cfg.P_aux_P0_W / 1000.0
        )
        p_aux_H = self._interp_at_load(
            beta_H, self.cfg.P_aux_Pn_W / 1000.0, self.cfg.P_aux_Pint_W / 1000.0, self.cfg.P_aux_P0_W / 1000.0
        )
        W_aux_on = p_aux_W * t_W_op + p_aux_H * t_H_op
        W_aux_off = (self.cfg.P_aux_P0_W / 1000.0) * t_off
        W_gen_aux = W_aux_on + W_aux_off

        Q_gen_aux_rvd = W_gen_aux * self.cfg.f_aux_rvd
        E_gen_in = max(Q_out - Q_gen_aux_rvd + Q_gen_ls - self.cfg.Q_gen_ren_kWh, 0.0)
        e_gen = (E_gen_in / Q_out) if Q_out > _EPS else float("nan")

        # Fuel split by each service's OWN output + on-loss - reflects each
        # service's actual operating point, unlike the "proportional" path's
        # single blended share.
        raw_H = qh_out_ctrl + Q_gen_ls_H_on
        raw_W = qw_out_ctrl + Q_gen_ls_W_on
        denom = raw_H + raw_W
        E_H_gen_in = E_gen_in * (raw_H / denom) if denom > _EPS else 0.0
        E_W_gen_in = E_gen_in * (raw_W / denom) if denom > _EPS else 0.0

        Q_gen_ls_env = triplet["P_gen_ls_P0(kW)"] * self.cfg.f_jacket * t_use
        Q_gen_ls_rbl = Q_gen_ls_env * self.cfg.f_location
        W_gen_aux_rbl = W_gen_aux * self.cfg.f_aux_recoverable * self.cfg.f_location

        t_run = t_W_op + t_H_op
        beta_combined = min(Q_out / max(total_capacity, _EPS), 1.0)
        P_gen_ls_beta = (Q_gen_ls_on / t_run) if t_run > _EPS else triplet["P_gen_ls_P0(kW)"]

        return {
            "Q_gen_out(kWh)": Q_out,
            "t_use(h)": t_use,
            "theta_avg(°C)": float(theta_avg_C),
            "theta_return(°C)": float(theta_return_C) if theta_return_C is not None else float("nan"),
            "beta_gen(-)": beta_combined,
            "t_run(h)": t_run,
            "t_off(h)": t_off,
            **triplet,
            "P_gen_ls_beta(kW)": P_gen_ls_beta,
            "Q_gen_ls_on(kWh)": Q_gen_ls_on,
            "Q_gen_ls_off(kWh)": Q_gen_ls_off,
            "Q_gen_ls(kWh)": Q_gen_ls,
            "E_gen_in(kWh)": E_gen_in,
            "e_gen(-)": e_gen,
            "E_H_gen_in(kWh)": E_H_gen_in,
            "E_W_gen_in(kWh)": E_W_gen_in,
            "W_gen_aux_on(kWh)": W_aux_on,
            "W_gen_aux_off(kWh)": W_aux_off,
            "W_gen_aux(kWh)": W_gen_aux,
            "Q_gen_ls_env(kWh)": Q_gen_ls_env,
            "Q_gen_ls_rbl(kWh)": Q_gen_ls_rbl,
            "W_gen_aux_rbl(kWh)": W_gen_aux_rbl,
            "Q_gen_out_req_kWh": Q_req,
            "Q_gen_unmet_kWh": Q_unmet,
            "Q_H_gen_out_kWh": qh_out,
            "Q_W_gen_out_kWh": qw_out,
            "Q_H_gen_unmet_kWh": Q_H_unmet,
            "Q_W_gen_unmet_kWh": Q_W_unmet,
            "Q_gen_aux_rvd_kWh": Q_gen_aux_rvd,
            "Q_gen_aux_rbl_kWh": W_gen_aux_rbl,
            # New keys specific to this mode: per-service operating time and
            # load ratio (Eq.(59)-(66), corrected - docs/standards_errata.md).
            "t_H_op(h)": t_H_op,
            "t_W_op(h)": t_W_op,
            "beta_H(-)": beta_H,
            "beta_W(-)": beta_W,
        }

    def _compute_step_electric_instantaneous(
        self,
        Q_gen_out_kWh: float,
        t_use_h: float,
        Q_H_gen_out_kWh: Optional[float],
        Q_W_gen_out_kWh: Optional[float],
    ) -> dict[str, float]:
        """Eq.(69): instantaneous electrical water heater - "thermal losses
        will have negligible losses. There is no auxiliary energy needed.",
        E_gen,in = f_tr,ls * Q_gen,out (f_tr,ls from Table B.17)."""

        t_use = max(float(t_use_h), _EPS)
        Pn = self.cfg.rated_power_kW
        Q_req, Q_out, Q_unmet, qh_req_raw, qw_req_raw = self._capped_demand(
            Q_gen_out_kWh, t_use, Q_H_gen_out_kWh, Q_W_gen_out_kWh
        )

        E_gen_in = self.cfg.f_tr_ls * Q_out
        e_gen = (E_gen_in / Q_out) if Q_out > _EPS else float("nan")

        qh_out, qw_out, Q_H_unmet, Q_W_unmet, E_H_gen_in, E_W_gen_in = self._split_h_w(
            Q_out, Q_req, Q_unmet, qh_req_raw, qw_req_raw, E_gen_in
        )
        t_run = min(Q_out / max(Pn, _EPS), t_use)

        return {
            "Q_gen_out(kWh)": Q_out,
            "t_use(h)": t_use,
            "theta_avg(°C)": float("nan"),
            "theta_return(°C)": float("nan"),
            "beta_gen(-)": min((Q_out / t_use) / max(Pn, _EPS), 1.0),
            "t_run(h)": t_run,
            "t_off(h)": t_use - t_run,
            "eta_Pn_corr(%)": 100.0 / self.cfg.f_tr_ls,
            "eta_Pint_corr(%)": 100.0 / self.cfg.f_tr_ls,
            "P_gen_ls_Pn(kW)": 0.0, "P_gen_ls_Pint(kW)": 0.0, "P_gen_ls_P0(kW)": 0.0,
            "P_gen_ls_beta(kW)": 0.0,
            "Q_gen_ls_on(kWh)": 0.0, "Q_gen_ls_off(kWh)": 0.0, "Q_gen_ls(kWh)": 0.0,
            "E_gen_in(kWh)": E_gen_in,
            "e_gen(-)": e_gen,
            "E_H_gen_in(kWh)": E_H_gen_in,
            "E_W_gen_in(kWh)": E_W_gen_in,
            "W_gen_aux_on(kWh)": 0.0, "W_gen_aux_off(kWh)": 0.0, "W_gen_aux(kWh)": 0.0,
            "Q_gen_ls_env(kWh)": 0.0, "Q_gen_ls_rbl(kWh)": 0.0, "W_gen_aux_rbl(kWh)": 0.0,
            "Q_gen_out_req_kWh": Q_req,
            "Q_gen_unmet_kWh": Q_unmet,
            "Q_H_gen_out_kWh": qh_out,
            "Q_W_gen_out_kWh": qw_out,
            "Q_H_gen_unmet_kWh": Q_H_unmet,
            "Q_W_gen_unmet_kWh": Q_W_unmet,
            "Q_gen_aux_rvd_kWh": 0.0,
            "Q_gen_aux_rbl_kWh": 0.0,
        }

    def _compute_step_gas_storage_water_heater(
        self,
        Q_gen_out_kWh: float,
        t_use_h: float,
        theta_avg_C: float,
        theta_return_C: Optional[float],
        Q_H_gen_out_kWh: Optional[float],
        Q_W_gen_out_kWh: Optional[float],
    ) -> dict[str, float]:
        """Clause 6.11.2/6.11.3: gas-fired domestic storage water heater.

        theta_avg_C/theta_return_C are reused generically here (the full
        boiler method's "circuit average/return temperature" signature) as
        theta_Wc;mn (mean water temperature context, default 45 degC) and
        theta_brm (installation room temperature, default 20 degC) for
        Eq.(70) - this module has no dedicated parameter pair for them.
        """

        t_use = max(float(t_use_h), _EPS)
        Pn = self.cfg.rated_power_kW
        Q_req, Q_out, Q_unmet, qh_req_raw, qw_req_raw = self._capped_demand(
            Q_gen_out_kWh, t_use, Q_H_gen_out_kWh, Q_W_gen_out_kWh
        )

        theta_Wc_mn = float(theta_avg_C)
        theta_brm = float(theta_return_C) if theta_return_C is not None else 20.0

        # Eq.(70): storage thermal loss. q_B,S (Table B.1/B.2 or EN 89
        # measured) is a kWh/DAY rate - divided by 24 h/day here so it
        # composes with this step's duration t_use the same way P_gen_ls_P0
        # (a loss RATE) does for the full boiler method elsewhere in this
        # module, instead of assuming t_use itself is a day count.
        q_B_S_kW = float(self.cfg.q_B_S_kWh_per_day) / 24.0
        Q_W_S_ls = max((theta_Wc_mn + 5.0 - theta_brm), 0.0) / 45.0 * t_use * q_B_S_kW

        # Eq.(74)/(75): combustion loss at the FIXED EN 89 reference
        # efficiency eta_100% (84 % default) - a single reference point, not
        # load-interpolated between P0/Pint/Pn like the full boiler method.
        eta_100 = self.cfg.eta_100_pct
        f = self.cfg.f_hs_hi_pct
        P_W_gen_100 = max((f - eta_100), 0.0) / eta_100 * (Q_out / t_use)
        Q_W_gen_ls = P_W_gen_100 * t_use

        Q_gen_ls = Q_W_S_ls + Q_W_gen_ls
        # 6.11.3: "For simplification, the auxiliary energy requirement
        # Q_W,gen,aux can be set at zero."
        E_gen_in = max(Q_out + Q_gen_ls, 0.0)
        e_gen = (E_gen_in / Q_out) if Q_out > _EPS else float("nan")

        qh_out, qw_out, Q_H_unmet, Q_W_unmet, E_H_gen_in, E_W_gen_in = self._split_h_w(
            Q_out, Q_req, Q_unmet, qh_req_raw, qw_req_raw, E_gen_in
        )
        t_run = min(Q_out / max(Pn, _EPS), t_use)

        return {
            "Q_gen_out(kWh)": Q_out,
            "t_use(h)": t_use,
            "theta_avg(°C)": theta_Wc_mn,
            "theta_return(°C)": theta_brm,
            "beta_gen(-)": min((Q_out / t_use) / max(Pn, _EPS), 1.0),
            "t_run(h)": t_run,
            "t_off(h)": t_use - t_run,
            "eta_Pn_corr(%)": eta_100,
            "eta_Pint_corr(%)": eta_100,
            "P_gen_ls_Pn(kW)": P_W_gen_100, "P_gen_ls_Pint(kW)": P_W_gen_100, "P_gen_ls_P0(kW)": q_B_S_kW,
            "P_gen_ls_beta(kW)": P_W_gen_100,
            "Q_gen_ls_on(kWh)": Q_W_gen_ls,
            "Q_gen_ls_off(kWh)": Q_W_S_ls,
            "Q_gen_ls(kWh)": Q_gen_ls,
            "E_gen_in(kWh)": E_gen_in,
            "e_gen(-)": e_gen,
            "E_H_gen_in(kWh)": E_H_gen_in,
            "E_W_gen_in(kWh)": E_W_gen_in,
            "W_gen_aux_on(kWh)": 0.0, "W_gen_aux_off(kWh)": 0.0, "W_gen_aux(kWh)": 0.0,
            "Q_gen_ls_env(kWh)": 0.0, "Q_gen_ls_rbl(kWh)": 0.0, "W_gen_aux_rbl(kWh)": 0.0,
            "Q_gen_out_req_kWh": Q_req,
            "Q_gen_unmet_kWh": Q_unmet,
            "Q_H_gen_out_kWh": qh_out,
            "Q_W_gen_out_kWh": qw_out,
            "Q_H_gen_unmet_kWh": Q_H_unmet,
            "Q_W_gen_unmet_kWh": Q_W_unmet,
            "Q_gen_aux_rvd_kWh": 0.0,
            "Q_gen_aux_rbl_kWh": 0.0,
        }

    def run_timeseries(
        self, df: pd.DataFrame, column_map: Optional[dict[str, str]] = None
    ) -> BoilerGeneratorSimulationResult:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must contain at least one row.")
        if self.cfg.boiler_type == "dhw_tapping_cycle_appliance":
            return self._run_timeseries_dhw_tapping_cycle(df, column_map)

        alias = {
            "Q_gen_out_kWh": ["Q_gen_out_kWh", "Q_gen_out", "Q_HW_dis_in_kWh", "Q_HW_dis_in", "Q_gen", "Q_out_kWh"],
            "Q_H_gen_out_kWh": [
                "Q_H_gen_out_kWh", "QH_gen_out_kWh", "QH_gen_out", "Q_H_gen_out", "Q_H_kWh", "QH_kWh",
            ],
            "Q_W_gen_out_kWh": [
                "Q_W_gen_out_kWh", "QW_gen_out_kWh", "QW_gen_out", "Q_W_gen_out", "Q_W_dis_i_in_kWh", "QW_dis_i_in",
                "Q_W_kWh", "QW_kWh",
            ],
            "t_use_h": ["t_use_h", "t_use", "dt_h", "delta_t_h", "time_step_hours"],
            "theta_avg_C": ["theta_avg_C", "theta_avg", "theta_gen_avg_C", "theta_gen_mean"],
            "theta_return_C": ["theta_return_C", "theta_return", "theta_gen_ret_C", "theta_X_gen_cr_ret"],
        }
        if column_map:
            for key, name in column_map.items():
                if key in alias:
                    alias[key] = [name, *alias[key]]

        def pick(row: pd.Series, names: list[str], default: Any = None) -> Any:
            for n in names:
                if n in row.index:
                    v = row[n]
                    if v is None:
                        continue
                    try:
                        if pd.isna(v):
                            continue
                    except (TypeError, ValueError):
                        pass
                    return v
            return default

        records = []
        for idx, row in df.iterrows():
            Q_out = pick(row, alias["Q_gen_out_kWh"])
            qh = pick(row, alias["Q_H_gen_out_kWh"], default=None)
            qw = pick(row, alias["Q_W_gen_out_kWh"], default=None)
            if Q_out is None:
                Q_out = float(qh or 0.0) + float(qw or 0.0)

            res = self.compute_step(
                Q_gen_out_kWh=float(Q_out),
                t_use_h=float(pick(row, alias["t_use_h"], default=1.0) or 1.0),
                theta_avg_C=float(pick(row, alias["theta_avg_C"], default=45.0) or 45.0),
                theta_return_C=pick(row, alias["theta_return_C"], default=None),
                Q_H_gen_out_kWh=qh,
                Q_W_gen_out_kWh=qw,
            )
            res["_index"] = idx
            records.append(res)

        out = pd.DataFrame.from_records(records)
        if "_index" in out.columns:
            out = out.set_index("_index")
            out.index.name = df.index.name

        # Convenience aliases matching the column names the other generation
        # modules (heat pump, district, cogeneration) use, so generic
        # reporting/plotting code that branches on "kind" can treat any of
        # them the same way.
        out["EHW_gen_in_kWh"] = out["E_gen_in(kWh)"]
        out["W_HW_gen_aux_kWh"] = out["W_gen_aux(kWh)"]
        out["Q_HW_gen_loss_kWh"] = out["Q_gen_ls(kWh)"]
        out["t_HW_gen_runtime_h"] = out["t_run(h)"]
        out["eta_HW_gen"] = np.where(
            out["E_gen_in(kWh)"] > _EPS, out["Q_gen_out(kWh)"] / out["E_gen_in(kWh)"], np.nan
        )

        summary = self._summarize(out)
        return BoilerGeneratorSimulationResult(timeseries=out, summary=summary, inputs=dict(self.input_data))

    def _run_timeseries_dhw_tapping_cycle(
        self, df: pd.DataFrame, column_map: Optional[dict[str, str]] = None
    ) -> BoilerGeneratorSimulationResult:
        """Clause 6.12 (Eq.(77)-(82)): aggregates df's DHW demand by
        CALENDAR DAY, calls dhw_tapping_cycle_energy_kWh() once per day
        (the standard's own granularity - a daily test-cycle characterisation,
        not an hourly loss curve), then distributes that day's fuel/
        auxiliary energy back across its hours in proportion to each hour's
        own share of the day's demand (falling back to an even split on a
        zero-demand day, since a share-of-zero is undefined but the
        appliance's own stand-by energy - Eq.(79)/(82) at Qd=0 - still has
        to land somewhere)."""

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "dhw_tapping_cycle_appliance's run_timeseries() needs a DatetimeIndex "
                "df (clause 6.12 is a daily method - calendar days must be identifiable "
                "from the index)."
            )

        alias = {
            "Q_gen_out_kWh": ["Q_gen_out_kWh", "Q_gen_out", "Q_HW_dis_in_kWh", "Q_HW_dis_in", "Q_gen", "Q_out_kWh"],
            "Q_H_gen_out_kWh": [
                "Q_H_gen_out_kWh", "QH_gen_out_kWh", "QH_gen_out", "Q_H_gen_out", "Q_H_kWh", "QH_kWh",
            ],
            "Q_W_gen_out_kWh": [
                "Q_W_gen_out_kWh", "QW_gen_out_kWh", "QW_gen_out", "Q_W_gen_out", "Q_W_dis_i_in_kWh", "QW_dis_i_in",
                "Q_W_kWh", "QW_kWh",
            ],
            "t_use_h": ["t_use_h", "t_use", "dt_h", "delta_t_h", "time_step_hours"],
        }
        if column_map:
            for key, name in column_map.items():
                if key in alias:
                    alias[key] = [name, *alias[key]]

        def pick(row: pd.Series, names: list[str], default: Any = None) -> Any:
            for n in names:
                if n in row.index:
                    v = row[n]
                    if v is None:
                        continue
                    try:
                        if pd.isna(v):
                            continue
                    except (TypeError, ValueError):
                        pass
                    return v
            return default

        demand: list[float] = []
        t_use_vals: list[float] = []
        for _, row in df.iterrows():
            Q_out = pick(row, alias["Q_gen_out_kWh"])
            if Q_out is None:
                qh = pick(row, alias["Q_H_gen_out_kWh"], default=0.0) or 0.0
                qw = pick(row, alias["Q_W_gen_out_kWh"], default=0.0) or 0.0
                Q_out = float(qh) + float(qw)
            demand.append(max(float(Q_out), 0.0))
            t_use_vals.append(float(pick(row, alias["t_use_h"], default=1.0) or 1.0))

        day_key = df.index.normalize()
        demand_s = pd.Series(demand, index=df.index)
        day_total_s = demand_s.groupby(day_key).transform("sum")
        day_count_s = demand_s.groupby(day_key).transform("size")

        day_results = {
            day: dhw_tapping_cycle_energy_kWh(
                float(total), self.cfg.tapping_cycle_test_points, self.cfg.tapping_cycle_standby_point
            )
            for day, total in demand_s.groupby(day_key).sum().items()
        }

        Pn = self.cfg.rated_power_kW
        records = []
        for i, idx in enumerate(df.index):
            Q_out_h = demand[i]
            t_use_h_val = t_use_vals[i]
            day_total = float(day_total_s.iloc[i])
            day_res = day_results[day_key[i]]
            share = (Q_out_h / day_total) if day_total > _EPS else (1.0 / float(day_count_s.iloc[i]))
            E_gen_in = day_res["E_W_gen_in_kWh"] * share
            W_gen_aux = day_res["W_W_gen_kWh"] * share
            e_gen = (E_gen_in / Q_out_h) if Q_out_h > _EPS else float("nan")

            # Eq.(77)/(80) EXTRAPOLATE beyond the highest tested cycle -
            # there is no hard capacity cap in clause 6.12 itself. Pn
            # defaults to a very large sentinel (see _load_simple_dhw_
            # parameters), making this cap a no-op unless a caller
            # deliberately supplies a real rated_power_kW.
            Q_out_capped = min(Q_out_h, Pn * t_use_h_val)
            Q_unmet = max(Q_out_h - Q_out_capped, 0.0)

            records.append({
                "Q_gen_out(kWh)": Q_out_capped,
                "t_use(h)": t_use_h_val,
                "theta_avg(°C)": float("nan"),
                "theta_return(°C)": float("nan"),
                "beta_gen(-)": float("nan"),
                "t_run(h)": t_use_h_val,
                "t_off(h)": 0.0,
                "eta_Pn_corr(%)": float("nan"),
                "eta_Pint_corr(%)": float("nan"),
                "P_gen_ls_Pn(kW)": 0.0, "P_gen_ls_Pint(kW)": 0.0, "P_gen_ls_P0(kW)": 0.0,
                "P_gen_ls_beta(kW)": 0.0,
                "Q_gen_ls_on(kWh)": 0.0, "Q_gen_ls_off(kWh)": 0.0, "Q_gen_ls(kWh)": 0.0,
                "E_gen_in(kWh)": E_gen_in,
                "e_gen(-)": e_gen,
                "E_H_gen_in(kWh)": 0.0,
                "E_W_gen_in(kWh)": E_gen_in,
                "W_gen_aux_on(kWh)": W_gen_aux, "W_gen_aux_off(kWh)": 0.0, "W_gen_aux(kWh)": W_gen_aux,
                "Q_gen_ls_env(kWh)": 0.0, "Q_gen_ls_rbl(kWh)": 0.0, "W_gen_aux_rbl(kWh)": 0.0,
                "Q_gen_out_req_kWh": Q_out_h,
                "Q_gen_unmet_kWh": Q_unmet,
                "Q_H_gen_out_kWh": 0.0,
                "Q_W_gen_out_kWh": Q_out_capped,
                "Q_H_gen_unmet_kWh": 0.0,
                "Q_W_gen_unmet_kWh": Q_unmet,
                "Q_gen_aux_rvd_kWh": 0.0,
                "Q_gen_aux_rbl_kWh": 0.0,
                "_index": idx,
            })

        out = pd.DataFrame.from_records(records)
        out = out.set_index("_index")
        out.index.name = df.index.name

        out["EHW_gen_in_kWh"] = out["E_gen_in(kWh)"]
        out["W_HW_gen_aux_kWh"] = out["W_gen_aux(kWh)"]
        out["Q_HW_gen_loss_kWh"] = out["Q_gen_ls(kWh)"]
        out["t_HW_gen_runtime_h"] = out["t_run(h)"]
        out["eta_HW_gen"] = np.where(
            out["E_gen_in(kWh)"] > _EPS, out["Q_gen_out(kWh)"] / out["E_gen_in(kWh)"], np.nan
        )

        summary = self._summarize(out)
        return BoilerGeneratorSimulationResult(timeseries=out, summary=summary, inputs=dict(self.input_data))

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        q_h = s("Q_H_gen_out_kWh")
        q_w = s("Q_W_gen_out_kWh")
        e_fuel = s("E_gen_in(kWh)")
        aux = s("W_gen_aux(kWh)")
        q_hw = q_h + q_w
        return {
            "QH_gen_out_kWh": q_h,
            "QW_gen_out_kWh": q_w,
            "QC_gen_out_kWh": 0.0,
            "QHW_gen_out_kWh": q_hw,
            "QH_gen_unmet_kWh": s("Q_H_gen_unmet_kWh"),
            "QW_gen_unmet_kWh": s("Q_W_gen_unmet_kWh"),
            "QHW_unmet_kWh": s("Q_H_gen_unmet_kWh") + s("Q_W_gen_unmet_kWh"),
            "EH_gen_in_kWh": s("E_H_gen_in(kWh)"),
            "EW_gen_in_kWh": s("E_W_gen_in(kWh)"),
            "EHW_gen_in_kWh": e_fuel,
            "E_total_fuel_kWh": e_fuel,
            "WHW_gen_aux_kWh": aux,
            "E_total_electricity_kWh": aux,
            "QHW_gen_ls_kWh": s("Q_gen_ls(kWh)"),
            "QHW_gen_ls_rbl_kWh": s("Q_gen_ls_rbl(kWh)"),
            "Q_gen_aux_rbl_kWh": s("W_gen_aux_rbl(kWh)"),
            "eta_HW_gen": _ratio(q_hw, e_fuel),
            "e_HW_gen": _ratio(e_fuel, q_hw),
        }


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


# --------------------------------------------------------------------------------------
# Clause 5.5 / Annex B: default values when product/test data is not available
# --------------------------------------------------------------------------------------

def generator_efficiency_default_pct(c1: float, c2: float, rated_power_kW: float) -> float:
    """Eq.(1)/(2)/(3)/(4): default generator efficiency [%] from the rated
    power output, using the Annex A/B c1,c2 (or c3,c4 for intermediate load,
    or the condensing-specific 60/30 degC pairs) coefficients. Pn is capped
    at 400 kW, per the standard's own note under Eq.(1)-(4)."""
    pn = min(max(float(rated_power_kW), _EPS), 400.0)
    return float(c1) + float(c2) * float(np.log10(pn))


def generator_standby_loss_default_W(
    c5: float, c6: float, rated_power_kW: float,
    eta_Pn_pct: float = 100.0, f_hs_hi_pct: float = 100.0,
) -> float:
    """Eq.(5)+(46 at test conditions): default stand-by heat loss [W] from
    the rated power output.

    Eq.(5) only defines f_gen,ls,P0 ITSELF (a fraction of Pn, from c5/c6) -
    turning that fraction into an actual loss POWER is Eq.(46):
    P_H;gen;ls;P0;corr = (Pn/eta_gen;Pn) * f_gen;ls;P0 * f_Hs/Hi *
    [(theta_Hc;mn-theta_brm)/(theta_gen;test;P0-20)]^1.25, which at TEST
    conditions (bracket = 1, the nameplate/default value this function
    returns) reduces to (Pn/eta_gen;Pn) * f_gen;ls;P0 * f_Hs/Hi - NOT the
    plain f_gen;ls;P0 * Pn this function used to return (missing BOTH the
    1/eta_gen;Pn factor and the f_Hs/Hi factor, understating P0 whenever
    eta_gen;Pn < 100 % - i.e. always, for a real generator).

    eta_Pn_pct/f_hs_hi_pct default to 100.0 each, which makes this reduce
    EXACTLY to the previous (incomplete) formula - unchanged behaviour for
    direct callers who do not pass them. default_boiler_config() DOES pass
    the generator's own computed eta_Pn and Table B.9's f_hs_hi_pct, since
    its whole purpose is to reproduce the standard's defaults correctly.
    """

    pn = max(float(rated_power_kW), _EPS)
    f_pct = float(c5) * (pn ** float(c6))
    eta_Pn_frac = max(float(eta_Pn_pct), _EPS) / 100.0
    f_hs_hi_frac = float(f_hs_hi_pct) / 100.0
    return (pn / eta_Pn_frac) * (f_pct / 100.0) * f_hs_hi_frac * 1000.0


def auxiliary_power_default_W(c7: float, c8: float, n: float, rated_power_kW: float) -> float:
    """Eq.(6): default auxiliary power [W] from the rated power output (c7,
    c8 are already in W, per Annex B's own table column headers)."""
    pn = max(float(rated_power_kW), _EPS)
    return float(c7) + float(c8) * (pn ** float(n))


def gas_storage_water_heater_standby_loss_default_kWh_per_day(
    storage_volume_L: float, installed_before_1985: bool = False
) -> float:
    """Formula (B.1)/(B.2): default daily stand-by thermal loss q_B,S
    [kWh/day] for a gas-fired domestic storage water heater (clause 6.11.3),
    when it has not been measured per EN 89 (45 K average temperature
    difference between the storage water and the installation space). This
    corresponds to 80 % of EN 89's minimum volume-related nominal heat input
    (0,07 kW/l). Formula (B.2) (the same expression x1,4) applies instead for
    appliances installed before 1985."""

    v = max(float(storage_volume_L), _EPS)
    q = 2.0 + 0.033 * (v ** 1.1)
    return q * 1.4 if installed_before_1985 else q


def direct_dhw_generator_rated_power_default_kW(storage_volume_L: float) -> float:
    """Eq.(76): default rated output [kW] for a direct-fired gas storage
    water heater, from the storage volume, when no product data is
    available for Pn."""

    v = max(float(storage_volume_L), _EPS)
    return 0.84 * (0.027 * v + 2.5)


def dhw_tapping_cycle_energy_kWh(
    Q_w_b_d_kWh: float,
    test_points: Sequence[dict[str, float]],
    standby_point: dict[str, float],
) -> dict[str, float]:
    """Clause 6.12, Eq.(77)-(82): daily fuel and auxiliary energy for a DHW
    appliance characterised by one or more 24 h tapping-cycle tests (EN
    13203-2), input data from Table 6 of the standard.

    This is a DAILY method (one data point per calendar day), structurally
    different from compute_step()'s hourly P0/Pint/Pn loss-curve stepping -
    it is NOT wired into run_timeseries(); call it once per day with that
    day's DHW energy need.

    Parameters
    ----------
    Q_w_b_d_kWh:
        Daily energy need for domestic hot water, Q_w;b;d.
    test_points:
        One dict per available tapping-cycle test (at least one required),
        each with "Q_W_test_kWh" (total delivered daily energy for that
        cycle, Q_W;test,i - EN 13203-2's Q_tot(i)), "E_W_gen_in_test_kWh"
        (daily fuel consumption, NCV - Q_gas;p(i)) and "W_W_gen_test_kWh"
        (daily auxiliary energy incl. stand-by use - E_eleco(i)). Sorted
        internally by Q_W_test_kWh; need not be pre-sorted by the caller.
    standby_point:
        Stand-by-only daily values, {"E_W_gen_in_test_stb_kWh": ...,
        "W_W_gen_test_stb_kWh": ...} (EN 13203-2's Q_gas;stb / E_eleco;stb),
        used when Q_w_b_d_kWh is below the lowest test point (Eq.79/82).

    Returns
    -------
    dict with "E_W_gen_in_kWh" (Eq.(77)-(79)) and "W_W_gen_kWh" (Eq.(80)-(82)).
    """

    if not test_points:
        raise ValueError("dhw_tapping_cycle_energy_kWh needs at least one test point.")
    pts = sorted(test_points, key=lambda p: float(p["Q_W_test_kWh"]))
    Qd = max(float(Q_w_b_d_kWh), 0.0)
    lowest, highest = pts[0], pts[-1]

    def _interp(y_lo: float, y_hi: float, x_lo: float, x_hi: float, x: float) -> float:
        if abs(x_hi - x_lo) <= _EPS:
            return y_lo
        return y_lo + (y_hi - y_lo) * (x - x_lo) / (x_hi - x_lo)

    Q_hi = float(highest["Q_W_test_kWh"])
    Q_lo = float(lowest["Q_W_test_kWh"])
    if Qd >= Q_hi:
        # Eq.(77)/(80): scale the highest available test point by the ratio
        # of actual to tested daily demand - extrapolation beyond the
        # highest tested tapping cycle.
        ratio = Qd / Q_hi if Q_hi > _EPS else 0.0
        E_in = float(highest["E_W_gen_in_test_kWh"]) * ratio
        W = float(highest["W_W_gen_test_kWh"]) * ratio
    elif Qd <= Q_lo:
        # Eq.(79)/(82): interpolate between the stand-by point and the
        # lowest available test point.
        ratio = Qd / Q_lo if Q_lo > _EPS else 0.0
        E_in = float(standby_point["E_W_gen_in_test_stb_kWh"]) + (
            float(lowest["E_W_gen_in_test_kWh"]) - float(standby_point["E_W_gen_in_test_stb_kWh"])
        ) * ratio
        W = float(standby_point["W_W_gen_test_stb_kWh"]) + (
            float(lowest["W_W_gen_test_kWh"]) - float(standby_point["W_W_gen_test_stb_kWh"])
        ) * ratio
    else:
        # Eq.(78)/(81): interpolate between the two bracketing test points.
        lo, hi = pts[0], pts[-1]
        for i in range(len(pts) - 1):
            if float(pts[i]["Q_W_test_kWh"]) <= Qd <= float(pts[i + 1]["Q_W_test_kWh"]):
                lo, hi = pts[i], pts[i + 1]
                break
        lo_x, hi_x = float(lo["Q_W_test_kWh"]), float(hi["Q_W_test_kWh"])
        E_in = _interp(float(lo["E_W_gen_in_test_kWh"]), float(hi["E_W_gen_in_test_kWh"]), lo_x, hi_x, Qd)
        W = _interp(float(lo["W_W_gen_test_kWh"]), float(hi["W_W_gen_test_kWh"]), lo_x, hi_x, Qd)

    return {"E_W_gen_in_kWh": E_in, "W_W_gen_kWh": W}


# Table B.1's build-year bands (Eq.(1)-(4) efficiency coefficients only -
# c1/c2/theta_test_Pn_C/c3/c4/theta_test_Pint_C/[c1_30/c2_30]) for each
# ANNEX_B_DEFAULTS boiler_type, read from the SAME physical Table B.1 row
# ANNEX_B_DEFAULTS' own comment names for that boiler_type (so a
# build_year_band selection never silently swaps to a different boiler
# sub-type). "after_1994" always equals ANNEX_B_DEFAULTS[boiler_type]'s own
# top-level c1/c2/.../c1_30/c2_30 - the pre-existing default,
# unchanged. Table B.3 (c5/c6 stand-by loss) and Table B.6 (auxiliary
# power) are NOT broken out by year band - they use a different
# categorisation than Table B.1's and are not curated per-band here; every
# build_year_band for a given boiler_type shares the SAME c5/c6/aux_*/
# f_corr_pct_per_K from ANNEX_B_DEFAULTS[boiler_type].
ANNEX_B_YEAR_BANDS: dict[str, dict[str, dict[str, float]]] = {
    "standard": {  # Table B.1, "Fan-assisted boiler (fossil and biomass fuel)"
        "before_1978": {"c1": 80.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 75.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "1978_to_1986": {"c1": 82.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 77.5, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "1987_to_1994": {"c1": 84.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 80.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "after_1994": {"c1": 85.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 81.5, "c4": 3.0, "theta_test_Pint_C": 70.0},
    },
    "low_temperature": {  # Table B.1, "atmospheric gas boiler" (low temperature section)
        "1978_to_1994": {"c1": 85.5, "c2": 1.5, "theta_test_Pn_C": 70.0, "c3": 86.0, "c4": 1.5, "theta_test_Pint_C": 40.0},
        "after_1994": {"c1": 88.5, "c2": 1.5, "theta_test_Pn_C": 70.0, "c3": 89.0, "c4": 1.5, "theta_test_Pint_C": 40.0},
    },
    "condensing": {  # Table B.1, "Condensing boiler (oil/gas)" - the before_1987 band has no
        # secondary 30b row in the table (no c1_30/c2_30), so it falls back to the
        # dew-point/combustion-chemistry approximation automatically (see
        # BoilerGeneratorCalculator._eta_Pn_corrected_pct). "improved_from_1999" is
        # the table's own "Condensing boiler, improved, from 1999, oil/gas" row,
        # whose 30b sub-row is split by fuel in the standard (oil c1=102,c2=0,3;
        # gas c1=102,c2=1,0) - this curated entry uses the gas sub-row; override
        # c1_30/c2_30 via **overrides for an oil-fired "improved" boiler.
        "before_1987": {"c1": 89.0, "c2": 1.0, "theta_test_Pn_C": 60.0, "c3": 95.0, "c4": 1.0, "theta_test_Pint_C": 30.0},
        "1987_to_1994": {"c1": 91.0, "c2": 1.0, "theta_test_Pn_C": 60.0, "c3": 97.5, "c4": 1.0, "theta_test_Pint_C": 30.0, "c1_30": 92.0, "c2_30": 1.0},
        "after_1994": {"c1": 92.0, "c2": 1.0, "theta_test_Pn_C": 60.0, "c3": 98.0, "c4": 1.0, "theta_test_Pint_C": 30.0, "c1_30": 93.0, "c2_30": 1.0},
        "improved_from_1999": {"c1": 94.0, "c2": 1.0, "theta_test_Pn_C": 60.0, "c3": 103.0, "c4": 1.0, "theta_test_Pint_C": 30.0, "c1_30": 102.0, "c2_30": 1.0},
    },
    "biomass_log": {  # Table B.1, "Atmospheric solid fuel boiler (fossil and biomass fuel)"
        "before_1978": {"c1": 78.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 72.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "1978_to_1994": {"c1": 80.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 75.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "after_1994": {"c1": 81.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 77.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
    },
    "biomass_pellet": {  # Table B.1, "Fan-assisted boiler (fossil and biomass fuel)" - the SAME row as
        # "standard" (the table does not give pellet boilers a separate efficiency row).
        "before_1978": {"c1": 80.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 75.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "1978_to_1986": {"c1": 82.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 77.5, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "1987_to_1994": {"c1": 84.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 80.0, "c4": 3.0, "theta_test_Pint_C": 70.0},
        "after_1994": {"c1": 85.0, "c2": 2.0, "theta_test_Pn_C": 70.0, "c3": 81.5, "c4": 3.0, "theta_test_Pint_C": 70.0},
    },
}

# Annex B Tables B.1/B.3/B.6, a CURATED subset - the "after 1994" build-year
# band (via ANNEX_B_YEAR_BANDS, see above for the other bands available per
# boiler_type) per boiler_type already in GENERATOR_TYPES - still NOT the
# full Annex B matrix (Table B.2's separate EN 303-5 class-based method,
# and Table B.3/B.6's own, DIFFERENT categorisations, are not curated here -
# see ANNEX_B_YEAR_BANDS' own comment). Annex A (the "normative template"
# counterpart) ships with its coefficient columns blank in the base
# standard, to be filled in by national annexes, so it has no usable
# numeric defaults of its own - this package uses Annex B throughout,
# matching the convention already used for EN 15316-3's Annex B defaults
# (EEI, pressure loss per length, etc.) elsewhere in this package. See
# default_boiler_config()'s docstring for the exact table row each entry is
# read from.
ANNEX_B_DEFAULTS: dict[str, dict[str, Any]] = {
    "standard": {  # Table B.1/B.3: fan-assisted boiler (fossil/biomass), after 1994; B.6: forced draught (oil/gas)
        "c1": 85.0, "c2": 2.0, "theta_test_Pn_C": 70.0,
        "c3": 81.5, "c4": 3.0, "theta_test_Pint_C": 70.0,
        "c5": 8.5, "c6": -0.4,
        "aux_Pn": (0.0, 45.0, 0.48), "aux_Pint": (0.0, 15.0, 0.48), "aux_P0": (15.0, 0.0, 0.0),
        # Table B.4 (f_corr,Pn) and Table B.5 (f_corr,Pint) are TWO DISTINCT
        # coefficients for "Oil/gas boiler: Standard boiler" - 0.0 %/K at
        # full load, 0.04 %/K at intermediate load, not the same value.
        "f_corr_pct_per_K": 0.0, "f_corr_pint_pct_per_K": 0.04,
    },
    "low_temperature": {  # Table B.1/B.3/B.6: atmospheric gas boiler (low-temperature), after 1994
        "c1": 88.5, "c2": 1.5, "theta_test_Pn_C": 70.0,
        "c3": 89.0, "c4": 1.5, "theta_test_Pint_C": 40.0,
        "c5": 6.1, "c6": -0.4,
        "aux_Pn": (40.0, 0.148, 1.0), "aux_Pint": (40.0, 0.148, 1.0), "aux_P0": (15.0, 0.0, 0.0),
        # Table B.4/B.5: "Low temperature boiler" - 0.04 %/K at BOTH full
        # and intermediate load (one of the cases where they coincide).
        "f_corr_pct_per_K": 0.04, "f_corr_pint_pct_per_K": 0.04,
    },
    "condensing": {  # Table B.1/B.3/B.6: condensing boiler (oil/gas), after 1994
        # Main row: c1,c2 for Eq.(2) (Pn @ 60 degC); c3,c4 for Eq.(4) (Pint,
        # tested @ 30 degC per the table's footnote b). The unlabelled
        # secondary row directly below "after 1994" (93.0, 1.0, with "30b"
        # under theta_test;Pn) gives c1,c2 for Eq.(3) (Pn @ 30 degC) instead -
        # each build-year band in Table B.1's condensing section has such a
        # secondary row, giving the SAME two columns a second meaning.
        "c1": 92.0, "c2": 1.0, "theta_test_Pn_C": 60.0,
        "c1_30": 93.0, "c2_30": 1.0,  # Eq.(3): Pn @ 30 degC
        "c3": 98.0, "c4": 1.0, "theta_test_Pint_C": 30.0,
        "c5": 4.0, "c6": -0.4,
        "aux_Pn": (0.0, 45.0, 0.48), "aux_Pint": (0.0, 15.0, 0.48), "aux_P0": (15.0, 0.0, 0.0),
        # Table B.4/B.5: "Gas condensing boiler" - 0.20 %/K at BOTH full and
        # intermediate load (gas; oil condensing is 0.04/0.10 respectively -
        # override f_corr_pct_per_K/f_corr_pint_pct_per_K if oil-fired).
        "f_corr_pct_per_K": 0.20, "f_corr_pint_pct_per_K": 0.20,
    },
    "biomass_log": {  # Table B.1/B.3: atmospheric solid fuel boiler, after 1994; B.6: solid fuel boiler up to 250 kW
        "c1": 81.0, "c2": 2.0, "theta_test_Pn_C": 70.0,
        "c3": 77.0, "c4": 3.0, "theta_test_Pint_C": 70.0,
        "c5": 8.0, "c6": -0.28,
        "aux_Pn": (40.0, 0.35, 1.0), "aux_Pint": (20.0, 0.1, 1.0), "aux_P0": (15.0, 0.0, 0.0),
        # Table B.4/B.5: "Solid fuel boiler (fossil and biomass fuel)" - 0.0
        # %/K at full load, 0.04 %/K at intermediate load.
        "f_corr_pct_per_K": 0.0, "f_corr_pint_pct_per_K": 0.04,
    },
    "biomass_pellet": {  # Table B.1: fan-assisted (fossil/biomass) efficiency; B.3: dedicated "Biomass boilers" row; B.6: pellet central boiler with buffer tank
        "c1": 85.0, "c2": 2.0, "theta_test_Pn_C": 70.0,
        "c3": 81.5, "c4": 3.0, "theta_test_Pint_C": 70.0,
        "c5": 14.0, "c6": -0.28,
        "aux_Pn": (40.0, 2.0, 1.0), "aux_Pint": (40.0, 1.8, 1.0), "aux_P0": (15.0, 0.0, 0.0),
        # Table B.4/B.5: same "Solid fuel boiler" row as biomass_log (pellet
        # boilers are solid-fuel biomass too) - 0.0 %/K Pn, 0.04 %/K Pint.
        "f_corr_pct_per_K": 0.0, "f_corr_pint_pct_per_K": 0.04,
    },
}


def default_boiler_config(
    boiler_type: str,
    fuel_type: str,
    rated_power_kW: float,
    boiler_location: str = "inside_heated",
    build_year_band: Optional[str] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a full BoilerGeneratorCalculator config from Eq.(1),(4)-(6) and
    the curated ANNEX_B_DEFAULTS table above, so a caller without product/
    test data can run from just boiler_type + rated_power_kW instead of
    supplying efficiency_table/loss_table/P_aux_*/f_corr_pct_per_K directly.

    build_year_band selects Table B.1's construction-year band for the
    Eq.(1)-(4) efficiency coefficients (c1/c2/c3/c4/theta_test_*_C/
    [c1_30/c2_30]) - see ANNEX_B_YEAR_BANDS for exactly which bands are
    available per boiler_type (ANNEX_B_YEAR_BANDS[boiler_type]). None (the
    default) keeps using ANNEX_B_DEFAULTS[boiler_type]'s own top-level
    values, i.e. the "after_1994" band - UNCHANGED pre-existing behaviour.

    Limitations (see ANNEX_B_YEAR_BANDS/ANNEX_B_DEFAULTS' own comments for
    what this covers):
    - build_year_band only varies Table B.1 (efficiency). Table B.3 (c5/c6,
      stand-by loss) and Table B.6 (auxiliary power) use a DIFFERENT
      categorisation than Table B.1's and are not curated per build year
      here - every build_year_band for a given boiler_type shares the same
      c5/c6/aux_*/f_corr_pct_per_K. Pass P_gen_ls_P0_W=.../P_aux_*_W=... in
      **overrides for another band's stand-by/auxiliary values.
    - Still not Annex B's full matrix: Table B.2 (the separate EN 303-5
      class-based method) is not covered at all.
    - Condensing boilers DO use Eq.(42)'s 60/30 degC interpolation (not the
      dew-point approximation) whenever the selected band has a c1_30/c2_30
      pair (every band except "before_1987", which the table itself gives
      no secondary 30b row for - that band falls back to the dew-point/
      combustion-chemistry approximation automatically).
    - dew_point_C, condensing_gain_pct and intermediate_load_fraction are not
      in the Annex B tables transcribed here; they keep this module's
      existing defaults (overridable as usual).
    - f_aux_recoverable defaults to Eq.(51)'s complement of B.1.3's stated
      f_aux,rvd default (0.75), i.e. 0.25.

    Any key BoilerConfig/BoilerGeneratorCalculator accepts can be overridden
    via **overrides, e.g. default_boiler_config("standard", "natural_gas",
    18.0, f_corr_pct_per_K=0.02).
    """

    if boiler_type not in ANNEX_B_DEFAULTS:
        raise ValueError(
            f"No curated Annex B default for boiler_type {boiler_type!r}. "
            f"Available: {sorted(ANNEX_B_DEFAULTS)}. Supply efficiency_table/"
            "loss_table/etc. directly instead (see BoilerGeneratorCalculator)."
        )
    d = dict(ANNEX_B_DEFAULTS[boiler_type])
    if build_year_band is not None:
        bands = ANNEX_B_YEAR_BANDS.get(boiler_type, {})
        if build_year_band not in bands:
            raise ValueError(
                f"No Table B.1 data for boiler_type {boiler_type!r}, "
                f"build_year_band {build_year_band!r}. Available bands: {sorted(bands)}."
            )
        band = bands[build_year_band]
        d.pop("c1_30", None)
        d.pop("c2_30", None)
        d.update(band)  # overrides c1/c2/theta_test_Pn_C/c3/c4/theta_test_Pint_C/[c1_30/c2_30]
    eta_Pn = generator_efficiency_default_pct(d["c1"], d["c2"], rated_power_kW)
    eta_Pint = generator_efficiency_default_pct(d["c3"], d["c4"], rated_power_kW)
    # Table B.9 (same source default_boiler_config()'s own f_hs_hi_pct
    # default uses - see BoilerConfig.f_hs_hi_pct's docstring): Eq.(46)
    # needs the generator's OWN eta_Pn and f_Hs/Hi to turn Eq.(5)'s
    # f_gen,ls,P0 fraction into an actual loss power correctly.
    f_hs_hi_pct_default = DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE.get(str(fuel_type).lower(), 100.0)
    p_ls_p0_W = generator_standby_loss_default_W(
        d["c5"], d["c6"], rated_power_kW, eta_Pn_pct=eta_Pn, f_hs_hi_pct=f_hs_hi_pct_default,
    )
    p_aux_pn_W = auxiliary_power_default_W(*d["aux_Pn"], rated_power_kW)
    p_aux_pint_W = auxiliary_power_default_W(*d["aux_Pint"], rated_power_kW)
    p_aux_p0_W = auxiliary_power_default_W(*d["aux_P0"], rated_power_kW)

    eff_case: dict[str, float] = {
        "eta_Pn_test_pct": eta_Pn, "eta_Pint_test_pct": eta_Pint,
        "theta_test_Pn_C": d["theta_test_Pn_C"], "theta_test_Pint_C": d["theta_test_Pint_C"],
    }
    if "c1_30" in d and "c2_30" in d:
        # Eq.(2)/(3): condensing boilers get BOTH reference points, enabling
        # Eq.(42)'s interpolation (see BoilerGeneratorCalculator._eta_Pn_corrected_pct)
        # instead of the dew-point approximation.
        eff_case["eta_Pn_test_60_pct"] = eta_Pn
        eff_case["eta_Pn_test_30_pct"] = generator_efficiency_default_pct(
            d["c1_30"], d["c2_30"], rated_power_kW
        )

    config: dict[str, Any] = {
        "boiler_type": boiler_type,
        "fuel_type": fuel_type,
        "rated_power_kW": rated_power_kW,
        "boiler_location": boiler_location,
        "intermediate_load_fraction": 0.30,
        "eta_Pn_test_pct": eta_Pn,
        "eta_Pint_test_pct": eta_Pint,
        "theta_test_Pn_C": d["theta_test_Pn_C"],
        "theta_test_Pint_C": d["theta_test_Pint_C"],
        "f_corr_pct_per_K": d["f_corr_pct_per_K"],
        "f_corr_pint_pct_per_K": d["f_corr_pint_pct_per_K"],
        "P_gen_ls_P0_W": p_ls_p0_W,
        "P_aux_Pn_W": p_aux_pn_W,
        "P_aux_Pint_W": p_aux_pint_W,
        "P_aux_P0_W": p_aux_p0_W,
        "f_aux_recoverable": 1.0 - 0.75,
        "dew_point_C": 55.0,
        "condensing_gain_pct": 11.0 if boiler_type == "condensing" else 0.0,
        "efficiency_table": {boiler_type: eff_case},
        "loss_table": {boiler_type: {"P_gen_ls_P0_W": p_ls_p0_W}},
    }
    config.update(overrides)
    return config


__all__ = [
    "BoilerConfig",
    "BoilerGeneratorCalculator",
    "BoilerGeneratorSimulationResult",
    "GENERATOR_TYPES",
    "GENERATOR_TYPE_ALIASES",
    "SIMPLE_DHW_GENERATOR_TYPES",
    "PRIORITY_MODES",
    "LOCATION_TYPES",
    "DEFAULT_F_JACKET_BY_BOILER_TYPE",
    "DEFAULT_F_CTRL_LS_BY_CONTROL_TYPE",
    "DEFAULT_F_LOCATION_BY_LOCATION",
    "DEFAULT_F_TR_LS_BY_ERA",
    "DEFAULT_THETA_BRM_C_BY_LOCATION",
    "DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE",
    "DEFAULT_X_AIR_PCT",
    "DEFAULT_X_FG_PCT",
    "DEFAULT_X_O2_FG_DRY_PCT",
    "ANNEX_B_DEFAULTS",
    "ANNEX_B_YEAR_BANDS",
    "FUEL_COMBUSTION_PROPERTIES_B13",
    "FUEL_SIEGERT_CONSTANTS_B10A",
    "generator_efficiency_default_pct",
    "generator_standby_loss_default_W",
    "auxiliary_power_default_W",
    "gas_storage_water_heater_standby_loss_default_kWh_per_day",
    "direct_dhw_generator_rated_power_default_kW",
    "dhw_tapping_cycle_energy_kWh",
    "saturation_humidity_kg_per_m3",
    "condensing_gain_fraction",
    "chimney_loss_pct_from_flue_gas_analysis",
    "fuel_power_input_kW",
    "split_standby_loss_pct",
    "boiler_efficiency_from_measured_values",
    "rated_power_from_measured_efficiency_kW",
    "default_boiler_config",
]
