import warnings

import pandas as pd
import pytest

import pybuildingenergy as pybui


def _loads():
    index = pd.date_range("2020-01-01", periods=24, freq="h")
    return pd.DataFrame(
        {
            "T_ext": 5.0,
            "Q_H_kWh": 1.0,
            "Q_W_kWh": 0.2,
            "Q_C_kWh": 0.1,
            "GHI": [0.0] * 8 + [300.0] * 8 + [0.0] * 8,
            "E_site_el_load_kWh": 0.3,
        },
        index=index,
    )


def _boiler_config(**overrides):
    config = {
        "boiler_type": "condensing", "fuel_type": "natural_gas", "rated_power_kW": 20.0,
        "intermediate_load_fraction": 0.30, "eta_Pn_test_pct": 98.0, "eta_Pint_test_pct": 106.0,
        "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 40.0, "f_corr_pct_per_K": 0.04,
        "P_gen_ls_P0_W": 100.0, "P_aux_on_W": 80.0, "P_aux_off_W": 5.0, "f_jacket": 0.40,
        "f_location": 1.0, "f_aux_recoverable": 0.75, "dew_point_C": 55.0, "condensing_gain_pct": 11.0,
        "efficiency_table": {"condensing": {"eta_Pn_test_pct": 98.0, "eta_Pint_test_pct": 106.0, "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 40.0}},
        "loss_table": {"condensing": {"P_gen_ls_P0_W": 100.0}},
        "boiler_location": "inside_heated",
    }
    config.update(overrides)
    return config


def test_boiler_generator_15316_4_1_summary_balance():
    """BoilerGeneratorCalculator is the SOLE EN 15316-4-1 implementation (the
    combustion_15316_4_1/biomass_15316_4 wrappers have been folded into it);
    it also accepts the generic Q_H_kWh/Q_W_kWh/Q_C_kWh/T_ext loads shape the
    other generation modules (heat pump, district, cogeneration) use."""

    result = pybui.BoilerGeneratorCalculator(_boiler_config()).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["QH_gen_unmet_kWh"] == pytest.approx(0.0)
    assert result.summary["QW_gen_unmet_kWh"] == pytest.approx(0.0)
    assert result.summary["EHW_gen_in_kWh"] > result.summary["QHW_gen_out_kWh"]
    assert result.summary["eta_HW_gen"] > 0.9


def test_boiler_generator_15316_4_1_eq1_eq5_eq6_default_correlations():
    import math

    from pybuildingenergy.source.generation_15316_4_1 import (
        auxiliary_power_default_W,
        generator_efficiency_default_pct,
        generator_standby_loss_default_W,
    )

    # Eq.(1)/(4): eta = c1 + c2*log10(Pn), Pn capped at 400 kW.
    assert generator_efficiency_default_pct(85.0, 2.0, 20.0) == pytest.approx(
        85.0 + 2.0 * math.log10(20.0)
    )
    assert generator_efficiency_default_pct(85.0, 2.0, 5000.0) == pytest.approx(
        generator_efficiency_default_pct(85.0, 2.0, 400.0)
    )  # the 400 kW cap

    # Eq.(5): f_gen,ls,P0 [%] = c5 * Pn^c6. eta_Pn_pct/f_hs_hi_pct default
    # to 100/100, reducing to the OLD (incomplete) P0 [W] = f%/100 * Pn[kW]
    # * 1000 form - unchanged behaviour for direct callers.
    p0_W = generator_standby_loss_default_W(8.5, -0.4, 20.0)
    expected_f_pct = 8.5 * (20.0 ** -0.4)
    assert p0_W == pytest.approx(expected_f_pct / 100.0 * 20.0 * 1000.0)

    # Eq.(46) at test conditions: P0 [W] = (Pn/eta_Pn) * f_gen,ls,P0 *
    # f_Hs/Hi * 1000 - a real generator (eta_Pn < 100 %) always has a
    # LARGER standby loss than the Eq.(5)-only form above.
    p0_W_corrected = generator_standby_loss_default_W(
        8.5, -0.4, 20.0, eta_Pn_pct=85.0, f_hs_hi_pct=111.0
    )
    expected_corrected = (20.0 / 0.85) * (expected_f_pct / 100.0) * 1.11 * 1000.0
    assert p0_W_corrected == pytest.approx(expected_corrected)
    assert p0_W_corrected > p0_W

    # Eq.(6): P_aux [W] = c7 + c8*Pn^n (c7, c8 already in W).
    assert auxiliary_power_default_W(0.0, 45.0, 0.48, 20.0) == pytest.approx(45.0 * 20.0**0.48)


def test_boiler_generator_15316_4_1_default_config_uses_eq42_for_condensing():
    """default_boiler_config()'s curated condensing entry supplies BOTH
    Table B.1 reference points (c1,c2 for Pn @ 60 degC, c1_30,c2_30 for Pn @
    30 degC), so it must activate Eq.(42)'s straight-line interpolation -
    not the dew-point approximation - exactly matching Eq.(2)/(3)'s own
    power-dependent values at each reference temperature."""

    import math

    from pybuildingenergy.source.generation_15316_4_1 import default_boiler_config

    config = default_boiler_config("condensing", "natural_gas", 20.0)
    case = config["efficiency_table"]["condensing"]
    assert case["eta_Pn_test_60_pct"] == pytest.approx(92.0 + 1.0 * math.log10(20.0))
    assert case["eta_Pn_test_30_pct"] == pytest.approx(93.0 + 1.0 * math.log10(20.0))

    calc = pybui.BoilerGeneratorCalculator(config)
    at_60 = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=60.0)
    at_30 = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=30.0)
    assert at_60["eta_Pn_corr(%)"] == pytest.approx(case["eta_Pn_test_60_pct"])
    assert at_30["eta_Pn_corr(%)"] == pytest.approx(case["eta_Pn_test_30_pct"])


def test_default_boiler_config_build_year_band_defaults_to_after_1994_unchanged():
    """build_year_band=None (the default) must stay byte-identical to the
    pre-existing single-band behaviour - selecting "after_1994" explicitly
    must give the exact same result."""

    from pybuildingenergy.source.generation_15316_4_1 import default_boiler_config

    implicit = default_boiler_config("condensing", "natural_gas", 20.0)
    explicit = default_boiler_config("condensing", "natural_gas", 20.0, build_year_band="after_1994")
    assert implicit["eta_Pn_test_pct"] == pytest.approx(explicit["eta_Pn_test_pct"])
    assert implicit["efficiency_table"] == explicit["efficiency_table"]


def test_default_boiler_config_build_year_band_table_b1_full_matrix():
    """Table B.1's build-year bands (Eq.(1)/(4)): selecting an older band
    must use ITS OWN c1/c2/c3/c4 coefficients, for every boiler_type/band
    ANNEX_B_YEAR_BANDS advertises - and condensing's "before_1987" band
    (which Table B.1 gives no 60/30 degC secondary row for) must fall back
    to the dew-point/combustion-chemistry approximation, not Eq.(42)."""

    import math

    from pybuildingenergy.source.generation_15316_4_1 import ANNEX_B_YEAR_BANDS, default_boiler_config

    for boiler_type, bands in ANNEX_B_YEAR_BANDS.items():
        for band_name, band in bands.items():
            fuel = "biomass_pellet" if "biomass" in boiler_type else "natural_gas"
            config = default_boiler_config(boiler_type, fuel, 20.0, build_year_band=band_name)
            case = config["efficiency_table"][boiler_type]
            assert case["eta_Pn_test_pct"] == pytest.approx(
                band["c1"] + band["c2"] * math.log10(20.0)
            )
            assert case["eta_Pint_test_pct"] == pytest.approx(
                band["c3"] + band["c4"] * math.log10(20.0)
            )
            assert case["theta_test_Pn_C"] == pytest.approx(band["theta_test_Pn_C"])
            if "c1_30" in band:
                assert "eta_Pn_test_60_pct" in case
                assert case["eta_Pn_test_30_pct"] == pytest.approx(
                    band["c1_30"] + band["c2_30"] * math.log10(20.0)
                )
            else:
                assert "eta_Pn_test_60_pct" not in case

    before_1987 = default_boiler_config("condensing", "natural_gas", 20.0, build_year_band="before_1987")
    assert "eta_Pn_test_60_pct" not in before_1987["efficiency_table"]["condensing"]
    calc = pybui.BoilerGeneratorCalculator(before_1987)
    result = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=30.0)
    assert result["E_gen_in(kWh)"] >= result["Q_gen_out(kWh)"]  # still runs, via the fallback approximation


def test_default_boiler_config_unknown_build_year_band_rejected():
    from pybuildingenergy.source.generation_15316_4_1 import default_boiler_config

    with pytest.raises(ValueError):
        default_boiler_config("standard", "natural_gas", 20.0, build_year_band="before_time")


def test_boiler_generator_15316_4_1_f_corr_pn_pint_default_to_shared_field_unchanged():
    """Tables B.4/B.5: f_corr_pn_pct_per_K/f_corr_pint_pct_per_K default to
    None, falling back to the shared f_corr_pct_per_K for BOTH Eq.(41)/(43)
    (full load) and Eq.(44)/(45) (intermediate load) - unchanged behaviour
    for any config that only ever sets the one shared field (e.g.
    _boiler_config(), which does not set either split field)."""

    config = _boiler_config(f_corr_pct_per_K=0.04)
    calc = pybui.BoilerGeneratorCalculator(config)
    assert calc.cfg.f_corr_pn_pct_per_K is None
    assert calc.cfg.f_corr_pint_pct_per_K is None

    result = calc.compute_step(10.0, theta_avg_C=80.0, theta_return_C=70.0)
    expected_eta_Pn = config["eta_Pn_test_pct"] - 0.04 * (80.0 - config["theta_test_Pn_C"])
    expected_eta_Pint = config["eta_Pint_test_pct"] - 0.04 * (80.0 - config["theta_test_Pint_C"])
    assert result["eta_Pn_corr(%)"] == pytest.approx(expected_eta_Pn)
    assert result["eta_Pint_corr(%)"] == pytest.approx(expected_eta_Pint)


def test_boiler_generator_15316_4_1_f_corr_pn_pint_split_overrides_shared_field():
    """Setting f_corr_pn_pct_per_K/f_corr_pint_pct_per_K overrides the
    shared f_corr_pct_per_K independently for Eq.(41)/(43) vs Eq.(44)/(45) -
    the standard's Table B.4/B.5 split (e.g. standard/biomass boilers:
    f_corr,Pn=0, f_corr,Pint=0.04 %/K, NOT the same value)."""

    config = _boiler_config(
        boiler_type="standard", f_corr_pct_per_K=0.0,
        f_corr_pint_pct_per_K=0.04,  # f_corr_pn_pct_per_K left unset -> falls back to f_corr_pct_per_K=0.0
    )
    config["efficiency_table"] = {"standard": config["efficiency_table"]["condensing"]}
    config["loss_table"] = {"standard": config["loss_table"]["condensing"]}
    calc = pybui.BoilerGeneratorCalculator(config)
    assert calc.cfg.f_corr_pn_pct_per_K is None
    assert calc.cfg.f_corr_pint_pct_per_K == pytest.approx(0.04)

    result = calc.compute_step(10.0, theta_avg_C=80.0, theta_return_C=70.0)
    # f_corr,Pn effectively 0 (falls back to f_corr_pct_per_K=0.0): NO
    # temperature correction at all for the full-load efficiency.
    assert result["eta_Pn_corr(%)"] == pytest.approx(config["eta_Pn_test_pct"])
    # f_corr,Pint=0.04 (the override): intermediate-load efficiency IS
    # corrected, independently of the (zero) full-load correction.
    expected_eta_Pint = config["eta_Pint_test_pct"] - 0.04 * (80.0 - config["theta_test_Pint_C"])
    assert result["eta_Pint_corr(%)"] == pytest.approx(expected_eta_Pint)
    assert result["eta_Pint_corr(%)"] != pytest.approx(config["eta_Pint_test_pct"])


def test_default_boiler_config_sets_table_b5_f_corr_pint_pct_per_K():
    """default_boiler_config() - whose whole purpose is to reproduce the
    standard's own defaults - DOES set f_corr_pint_pct_per_K from Table B.5
    for every curated boiler_type, since Table B.4 (f_corr_pct_per_K) and
    Table B.5 (f_corr_pint_pct_per_K) are not the same value for standard/
    biomass boilers (0 %/K vs 0.04 %/K)."""

    from pybuildingenergy.source.generation_15316_4_1 import default_boiler_config

    for boiler_type, expected_pint in (
        ("standard", 0.04), ("low_temperature", 0.04), ("condensing", 0.20),
        ("biomass_log", 0.04), ("biomass_pellet", 0.04),
    ):
        config = default_boiler_config(boiler_type, "natural_gas", 20.0)
        assert config["f_corr_pint_pct_per_K"] == pytest.approx(expected_pint)

    # standard/biomass: f_corr,Pn (0) and f_corr,Pint (0.04) genuinely
    # differ - confirm compute_step() actually uses each independently.
    config = default_boiler_config("standard", "natural_gas", 20.0)
    assert config["f_corr_pct_per_K"] == pytest.approx(0.0)
    assert config["f_corr_pint_pct_per_K"] == pytest.approx(0.04)
    calc = pybui.BoilerGeneratorCalculator(config)
    result = calc.compute_step(10.0, theta_avg_C=90.0, theta_return_C=70.0)
    assert result["eta_Pn_corr(%)"] == pytest.approx(config["eta_Pn_test_pct"])  # f_corr,Pn=0 -> no correction
    assert result["eta_Pint_corr(%)"] != pytest.approx(config["eta_Pint_test_pct"])  # f_corr,Pint=0.04 -> corrected


def test_default_boiler_config_eq46_corrects_standby_loss_for_efficiency_and_hs_hi():
    """default_boiler_config()'s P_gen_ls_P0_W must come from Eq.(46) at
    test conditions - (Pn/eta_Pn) * f_gen,ls,P0 * f_Hs/Hi - not Eq.(5)
    alone (f_gen,ls,P0 * Pn), which omits both the 1/eta_Pn factor and the
    f_Hs/Hi factor and so UNDERSTATES the standby loss for any real
    generator (eta_Pn < 100 %)."""

    from pybuildingenergy.source.generation_15316_4_1 import (
        ANNEX_B_DEFAULTS,
        DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE,
        default_boiler_config,
        generator_efficiency_default_pct,
        generator_standby_loss_default_W,
    )

    d = ANNEX_B_DEFAULTS["standard"]
    config = default_boiler_config("standard", "natural_gas", 20.0)
    eta_Pn = generator_efficiency_default_pct(d["c1"], d["c2"], 20.0)
    f_hs_hi = DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE["natural_gas"]
    expected = generator_standby_loss_default_W(
        d["c5"], d["c6"], 20.0, eta_Pn_pct=eta_Pn, f_hs_hi_pct=f_hs_hi
    )
    assert config["P_gen_ls_P0_W"] == pytest.approx(expected)

    # The Eq.(5)-only (pre-fix) value would be strictly smaller, since
    # eta_Pn < 100 % and f_hs_hi > 100 % for a real gas-fired generator.
    eq5_only = generator_standby_loss_default_W(d["c5"], d["c6"], 20.0)
    assert config["P_gen_ls_P0_W"] > eq5_only


def test_boiler_generator_15316_4_1_default_config_runs_for_every_generator_type():
    """default_boiler_config() (Eq.(1),(4)-(6) + the curated Annex B subset)
    must produce a config BoilerGeneratorCalculator can actually run, for
    every boiler_type the combustion loss-curve method covers (ANNEX_B_
    DEFAULTS) - not just the ones already covered by hand-written test
    configs elsewhere in this file. The clause 6.11 direct-heated DHW
    appliance types (SIMPLE_DHW_GENERATOR_TYPES) use a structurally
    different model with no Annex B coefficient table of their own, so
    default_boiler_config() does not (and should not) cover them - see
    the dedicated electric_instantaneous/gas_storage_water_heater tests."""

    from pybuildingenergy.source.generation_15316_4_1 import ANNEX_B_DEFAULTS, default_boiler_config

    for boiler_type in sorted(ANNEX_B_DEFAULTS):
        fuel = "biomass_pellet" if "biomass" in boiler_type else "natural_gas"
        config = default_boiler_config(boiler_type, fuel, rated_power_kW=20.0)
        result = pybui.BoilerGeneratorCalculator(config).compute_step(
            10.0, theta_avg_C=50.0, theta_return_C=40.0
        )
        assert 0.0 < result["eta_Pn_corr(%)"] <= 110.0
        assert result["E_gen_in(kWh)"] >= result["Q_gen_out(kWh)"]


def test_boiler_generator_15316_4_1_f_jacket_f_location_derive_from_type_and_location():
    """Table B.7/B.8: f_jacket and f_location are no longer required inputs -
    they default from boiler_type and boiler_location respectively."""

    config = {
        "boiler_type": "condensing", "fuel_type": "natural_gas", "rated_power_kW": 20.0,
        "intermediate_load_fraction": 0.30, "eta_Pn_test_pct": 92.0, "eta_Pint_test_pct": 98.0,
        "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 30.0, "f_corr_pct_per_K": 0.20,
        "P_gen_ls_P0_W": 100.0, "P_aux_on_W": 80.0, "P_aux_off_W": 5.0,
        "dew_point_C": 55.0, "condensing_gain_pct": 11.0, "f_aux_recoverable": 0.25,
        "efficiency_table": {"condensing": {"eta_Pn_test_pct": 92.0, "eta_Pint_test_pct": 98.0, "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 30.0}},
        "loss_table": {"condensing": {"P_gen_ls_P0_W": 100.0}},
        "boiler_location": "outside_building",
        # f_jacket/f_location deliberately NOT given.
    }
    calc = pybui.BoilerGeneratorCalculator(config)
    assert calc.cfg.f_jacket == pytest.approx(0.75)  # Table B.7, fan-assisted (condensing)
    assert calc.cfg.f_location == pytest.approx(0.0)  # Table B.8, outdoors: f_brm=1 -> 1-f_brm=0

    config["boiler_location"] = "inside_heated"
    calc2 = pybui.BoilerGeneratorCalculator(config)
    assert calc2.cfg.f_location == pytest.approx(1.0)  # Table B.8, inside heated space: f_brm=0


def test_boiler_generator_15316_4_1_eq42_condensing_interpolates_60_30_degC():
    """Eq.(42): when the case data gives efficiency measured at BOTH a 60 degC
    and a 30 degC average return temperature, full-load efficiency is a
    straight line between those two points by the actual return temperature -
    not the dew-point approximation (which is used only when the second test
    point, at 30 degC, is absent)."""

    config = _boiler_config()
    config["efficiency_table"]["condensing"].update(
        {"eta_Pn_test_60_pct": 88.0, "eta_Pn_test_30_pct": 108.0}
    )
    calc = pybui.BoilerGeneratorCalculator(config)

    at_60 = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=60.0)
    at_30 = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=30.0)
    at_45 = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=45.0)

    assert at_60["eta_Pn_corr(%)"] == pytest.approx(88.0)
    assert at_30["eta_Pn_corr(%)"] == pytest.approx(108.0)
    # Midpoint of the two reference temperatures -> midpoint of the two
    # measured efficiencies (straight-line interpolation).
    assert at_45["eta_Pn_corr(%)"] == pytest.approx((88.0 + 108.0) / 2.0)


def test_boiler_generator_15316_4_1_eq43_hs_hi_defaults_from_fuel_type_table_b9():
    """Eq.(43)/(45): P_ls = (f_Hs/Hi - eta_corr)/eta_corr * P, both terms on
    the SAME percent scale. f_hs_hi_pct DEFAULTS from fuel_type via Table
    B.9 (DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE - natural_gas -> 111 %), NOT a
    flat 100 % (Hs = Hi) regardless of fuel: no combustion fuel in Table
    B.9 actually has Hs/Hi = 1.00. An explicit f_hs_hi_pct always overrides
    the default; an unrecognised fuel_type falls back to 100 %."""

    from pybuildingenergy.source.generation_15316_4_1 import DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE

    baseline = pybui.BoilerGeneratorCalculator(_boiler_config()).compute_step(
        10.0, theta_avg_C=70.0, theta_return_C=70.0  # no condensing gain at this return temp
    )
    eta = baseline["eta_Pn_corr(%)"]
    assert eta < 100.0  # otherwise both formulas clip at 0 and the test proves nothing
    default_f = DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE["natural_gas"]  # 111.0
    expected = 20.0 * (default_f - eta) / eta  # rated_power_kW = 20.0 in _boiler_config()
    assert baseline["P_gen_ls_Pn(kW)"] == pytest.approx(expected)

    # Explicit f_hs_hi_pct=100 still works, and still reduces to the Hs=Hi
    # form - the pre-existing formula structure, for whoever asks for it.
    explicit_100 = pybui.BoilerGeneratorCalculator(
        _boiler_config(f_hs_hi_pct=100.0)
    ).compute_step(10.0, theta_avg_C=70.0, theta_return_C=70.0)
    expected_100 = 20.0 * (100.0 - eta) / eta
    assert explicit_100["P_gen_ls_Pn(kW)"] == pytest.approx(expected_100)
    assert explicit_100["P_gen_ls_Pn(kW)"] < baseline["P_gen_ls_Pn(kW)"]

    with_factor = pybui.BoilerGeneratorCalculator(
        _boiler_config(f_hs_hi_pct=111.0)
    ).compute_step(10.0, theta_avg_C=70.0, theta_return_C=70.0)
    expected_with_factor = 20.0 * (111.0 - eta) / eta
    assert with_factor["P_gen_ls_Pn(kW)"] == pytest.approx(expected_with_factor)
    assert with_factor["P_gen_ls_Pn(kW)"] > explicit_100["P_gen_ls_Pn(kW)"]

    # Unrecognised fuel_type falls back to 100 % (Hs = Hi) - the previous
    # behaviour, kept where Table B.9 has no entry.
    unknown_fuel = pybui.BoilerGeneratorCalculator(
        _boiler_config(fuel_type="hydrogen")
    ).compute_step(10.0, theta_avg_C=70.0, theta_return_C=70.0)
    assert unknown_fuel["P_gen_ls_Pn(kW)"] == pytest.approx(expected_100)


def test_boiler_generator_15316_4_1_eq34_subtracts_recovered_aux_and_renewable():
    """Eq.(34): E_gen,in = Q_gen,out - Q_gen,aux,rvd + Q_gen,ls - Q_gen,ren.

    Q_gen_ren_kWh defaults to 0. f_aux_rvd defaults to Eq.(51)'s complement
    of f_aux_recoverable (1 - f_aux_recoverable), not to 0, so the credit is
    applied automatically whenever f_aux_recoverable is already set - the two
    fractions must never be free to sum to more than 1 (which would credit
    the same auxiliary energy to both the fluid and the space)."""

    config = _boiler_config()  # f_aux_recoverable = 0.75
    baseline = pybui.BoilerGeneratorCalculator(config).compute_step(
        10.0, theta_avg_C=45.0, theta_return_C=35.0
    )
    expected_default_credit = baseline["W_gen_aux(kWh)"] * (1.0 - config["f_aux_recoverable"])
    assert baseline["Q_gen_aux_rvd_kWh"] == pytest.approx(expected_default_credit)
    assert baseline["Q_gen_aux_rvd_kWh"] + baseline["Q_gen_aux_rbl_kWh"] == pytest.approx(
        baseline["W_gen_aux(kWh)"]
    )  # Eq.(51): the two fractions split the SAME auxiliary energy, not double-count it

    no_credit = pybui.BoilerGeneratorCalculator(
        _boiler_config(f_aux_rvd=0.0)
    ).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)
    assert no_credit["Q_gen_aux_rvd_kWh"] == pytest.approx(0.0)

    with_aux_credit = pybui.BoilerGeneratorCalculator(
        _boiler_config(f_aux_rvd=0.5)
    ).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)
    expected_credit = with_aux_credit["W_gen_aux(kWh)"] * 0.5
    assert with_aux_credit["Q_gen_aux_rvd_kWh"] == pytest.approx(expected_credit)
    assert no_credit["E_gen_in(kWh)"] - with_aux_credit["E_gen_in(kWh)"] == pytest.approx(expected_credit)

    with_renewable = pybui.BoilerGeneratorCalculator(
        _boiler_config(Q_gen_ren_kWh=1.2)
    ).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)
    assert baseline["E_gen_in(kWh)"] - with_renewable["E_gen_in(kWh)"] == pytest.approx(1.2)


def test_boiler_generator_15316_4_1_warns_on_inconsistent_f_aux_rvd_override():
    """Eq.(51): f_aux,rbl = 1 - f_aux,rvd. Leaving f_aux_rvd unset always
    derives a consistent pair (no warning); an EXPLICIT f_aux_rvd that does
    not sum to 1 with f_aux_recoverable is accepted (deliberate decoupling
    is a supported use case - see the eq34 test above) but must warn, since
    it risks double-counting or losing recoverable auxiliary energy."""

    # Default derivation: never warns, regardless of f_aux_recoverable.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pybui.BoilerGeneratorCalculator(_boiler_config())

    # Consistent explicit override (sums to 1): no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pybui.BoilerGeneratorCalculator(_boiler_config(f_aux_rvd=1.0 - _boiler_config()["f_aux_recoverable"]))

    # Inconsistent explicit override: must warn, but must NOT raise/refuse -
    # this stays a supported (if unusual) input.
    with pytest.warns(UserWarning, match="f_aux_rvd"):
        pybui.BoilerGeneratorCalculator(_boiler_config(f_aux_rvd=0.9))  # 0.9 + 0.75 = 1.65


def test_boiler_generator_15316_4_1_caps_capacity_and_reports_unmet():
    # 1.2 kWh of combined demand per hour against a rated power well below
    # that: Eq.(28)-style capacity cap, reported the same way heat pump/
    # district/cogeneration report unmet demand - not silently delivered.
    result = pybui.BoilerGeneratorCalculator(_boiler_config(rated_power_kW=0.5)).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(0.5 * 24)
    assert result.summary["QH_gen_unmet_kWh"] > 0.0
    assert result.summary["QW_gen_unmet_kWh"] > 0.0


def test_boiler_generator_15316_4_1_eq58_f_ctrl_ls_inflates_fuel_not_served_energy():
    """Eq.(58): Q_gen,out = f_ctrl,ls * sum(Q_H,dis,in) + sum(Q_XY,dis,in).
    f_ctrl_ls defaults to 1.0 (Table B.16 best case), reproducing the
    pre-existing behaviour unchanged. A control-strategy penalty (>1)
    inflates what the generator must produce/burn for heating, but must
    NOT inflate what is reported as served to the distribution system -
    only the fuel attribution and the generator's own capacity load."""

    baseline = pybui.BoilerGeneratorCalculator(_boiler_config()).compute_step(
        10.0, theta_avg_C=45.0, theta_return_C=35.0, Q_H_gen_out_kWh=7.0, Q_W_gen_out_kWh=3.0
    )
    assert baseline["Q_gen_out(kWh)"] == pytest.approx(10.0)
    assert baseline["Q_H_gen_out_kWh"] == pytest.approx(7.0)
    assert baseline["Q_W_gen_out_kWh"] == pytest.approx(3.0)

    penalised = pybui.BoilerGeneratorCalculator(
        _boiler_config(f_ctrl_ls=1.06)
    ).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0, Q_H_gen_out_kWh=7.0, Q_W_gen_out_kWh=3.0)

    # Only the heating share is inflated: 7*1.06 + 3 = 10.42.
    assert penalised["Q_gen_out(kWh)"] == pytest.approx(7.0 * 1.06 + 3.0)
    # Served/reported energy to distribution is unchanged - f_ctrl,ls is a
    # generator-side control loss, not a cut to what gets delivered.
    assert penalised["Q_H_gen_out_kWh"] == pytest.approx(7.0)
    assert penalised["Q_W_gen_out_kWh"] == pytest.approx(3.0)
    assert penalised["Q_H_gen_unmet_kWh"] == pytest.approx(0.0)
    assert penalised["Q_W_gen_unmet_kWh"] == pytest.approx(0.0)
    # The extra fuel burden is attributed entirely to the inflated heating
    # share, so fuel use (E_gen_in) must be strictly higher than baseline.
    assert penalised["E_gen_in(kWh)"] > baseline["E_gen_in(kWh)"]


def test_boiler_generator_15316_4_1_f_ctrl_ls_defaults_from_control_type():
    """Table B.16: f_ctrl_ls also derives from a 'control_type' convenience
    key (floor_standing_outdoor_reset=1.00, wall_hung_outdoor_reset=1.03,
    wall_hung_room_control=1.06) so callers don't need the raw factor."""

    from pybuildingenergy.source.generation_15316_4_1 import DEFAULT_F_CTRL_LS_BY_CONTROL_TYPE

    calc = pybui.BoilerGeneratorCalculator(
        _boiler_config(control_type="wall_hung_room_control")
    )
    assert calc.cfg.f_ctrl_ls == pytest.approx(
        DEFAULT_F_CTRL_LS_BY_CONTROL_TYPE["wall_hung_room_control"]
    )
    assert calc.cfg.f_ctrl_ls == pytest.approx(1.06)


def test_boiler_generator_15316_4_1_f_ctrl_ls_splits_capacity_shortfall_proportionally():
    """When the control-inflated requirement exceeds rated capacity, the
    shortfall reported to heating/DHW is split in proportion to each
    service's OWN (un-inflated) share of the original request - not in
    proportion to the control-inflated shares, which would unfairly
    penalise heating just for carrying the control loss."""

    calc = pybui.BoilerGeneratorCalculator(
        _boiler_config(f_ctrl_ls=1.06, rated_power_kW=10.0)
    )
    result = calc.compute_step(
        10.0, theta_avg_C=45.0, theta_return_C=35.0, Q_H_gen_out_kWh=7.0, Q_W_gen_out_kWh=3.0
    )
    served_fraction = 10.0 / (7.0 * 1.06 + 3.0)
    assert result["Q_H_gen_unmet_kWh"] == pytest.approx(7.0 * (1.0 - served_fraction))
    assert result["Q_W_gen_unmet_kWh"] == pytest.approx(3.0 * (1.0 - served_fraction))


def test_boiler_generator_15316_4_1_eq69_instantaneous_electric_water_heater():
    """Eq.(69): E_gen,in = f_tr,ls * Q_gen,out - negligible thermal losses, no
    auxiliary energy. f_tr,ls defaults from Table B.17 by installation era
    (since_1980 -> 1.0, the pre-existing-equivalent "no penalty" case;
    before_1980 -> 1.1)."""

    base_config = {
        "boiler_type": "electric_instantaneous", "fuel_type": "electricity",
        "rated_power_kW": 24.0, "boiler_location": "inside_heated",
    }
    calc = pybui.BoilerGeneratorCalculator(base_config)
    assert calc.cfg.f_tr_ls == pytest.approx(1.0)
    result = calc.compute_step(5.0)
    assert result["E_gen_in(kWh)"] == pytest.approx(5.0)
    assert result["Q_gen_ls(kWh)"] == pytest.approx(0.0)
    assert result["W_gen_aux(kWh)"] == pytest.approx(0.0)

    old_calc = pybui.BoilerGeneratorCalculator({**base_config, "installation_era": "before_1980"})
    assert old_calc.cfg.f_tr_ls == pytest.approx(1.1)
    old_result = old_calc.compute_step(5.0)
    assert old_result["E_gen_in(kWh)"] == pytest.approx(5.5)

    # Capacity cap still applies (the rated power of the appliance itself).
    capped = pybui.BoilerGeneratorCalculator({**base_config, "rated_power_kW": 2.0}).compute_step(5.0)
    assert capped["Q_gen_out(kWh)"] == pytest.approx(2.0)
    assert capped["Q_gen_unmet_kWh"] == pytest.approx(3.0)


def test_boiler_generator_15316_4_1_gas_instantaneous_alias_resolves_to_low_temperature():
    """6.11.1: 'Instantaneous gas water heater shall be calculated as low
    temperature combination boilers' - the 'gas_instantaneous' spelling is a
    convenience alias for boiler_type='low_temperature', not a separate
    calculation path."""

    config = _boiler_config(boiler_type="low_temperature")
    config["efficiency_table"] = {"low_temperature": config["efficiency_table"]["condensing"]}
    config["loss_table"] = {"low_temperature": config["loss_table"]["condensing"]}
    config["boiler_type"] = "gas_instantaneous"

    calc = pybui.BoilerGeneratorCalculator(config)
    assert calc.cfg.boiler_type == "low_temperature"
    result = calc.compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)
    assert result["E_gen_in(kWh)"] >= result["Q_gen_out(kWh)"]


def test_boiler_generator_15316_4_1_gas_storage_water_heater_eq70_and_eq74():
    """Clause 6.11.2/6.11.3: Eq.(70) storage stand-by loss (from Formula
    (B.1)'s default q_B,S, since none is measured here) plus Eq.(74)/(75)'s
    fixed-eta_100% combustion loss; Eq.(76) sizes Pn from the storage volume
    since rated_power_kW is not given. Auxiliary energy is zero (the
    standard's own explicit simplification for this appliance)."""

    from pybuildingenergy.source.generation_15316_4_1 import (
        direct_dhw_generator_rated_power_default_kW,
        gas_storage_water_heater_standby_loss_default_kWh_per_day,
    )

    config = {
        "boiler_type": "gas_storage_water_heater", "fuel_type": "natural_gas",
        "storage_volume_L": 150.0, "boiler_location": "inside_heated",
    }
    calc = pybui.BoilerGeneratorCalculator(config)

    expected_pn = direct_dhw_generator_rated_power_default_kW(150.0)
    expected_q_B_S = gas_storage_water_heater_standby_loss_default_kWh_per_day(150.0)
    assert calc.cfg.rated_power_kW == pytest.approx(expected_pn)
    assert calc.cfg.q_B_S_kWh_per_day == pytest.approx(expected_q_B_S)
    assert calc.cfg.eta_100_pct == pytest.approx(84.0)
    # Table B.9: f_hs_hi_pct defaults from fuel_type (natural_gas -> 111 %),
    # not a flat 100 % - see DEFAULT_F_HS_HI_PCT_BY_FUEL_TYPE.
    assert calc.cfg.f_hs_hi_pct == pytest.approx(111.0)

    result = calc.compute_step(3.0, theta_avg_C=45.0, theta_return_C=20.0)
    expected_storage_loss = (45.0 + 5.0 - 20.0) / 45.0 * (expected_q_B_S / 24.0)
    expected_combustion_loss = (calc.cfg.f_hs_hi_pct - 84.0) / 84.0 * 3.0
    assert result["Q_gen_ls_off(kWh)"] == pytest.approx(expected_storage_loss)
    assert result["Q_gen_ls_on(kWh)"] == pytest.approx(expected_combustion_loss)
    assert result["E_gen_in(kWh)"] == pytest.approx(3.0 + expected_storage_loss + expected_combustion_loss)
    assert result["W_gen_aux(kWh)"] == pytest.approx(0.0)  # 6.11.3: set at zero by simplification

    # A measured EN 89 q_B,S overrides the Formula (B.1) default.
    measured = pybui.BoilerGeneratorCalculator(
        {**config, "q_B_S_kWh_per_day": 5.0, "rated_power_kW": 10.0}
    )
    assert measured.cfg.q_B_S_kWh_per_day == pytest.approx(5.0)
    assert measured.cfg.rated_power_kW == pytest.approx(10.0)


def test_dhw_tapping_cycle_energy_kWh_eq77_to_82():
    """Clause 6.12: Eq.(77)-(82), the 24 h tapping-cycle appliance test
    method - above the highest test point (extrapolate), between two test
    points (interpolate), and below the lowest (interpolate against the
    stand-by point)."""

    from pybuildingenergy.source.generation_15316_4_1 import dhw_tapping_cycle_energy_kWh

    test_points = [
        {"Q_W_test_kWh": 5.845, "E_W_gen_in_test_kWh": 7.0, "W_W_gen_test_kWh": 0.6},
        {"Q_W_test_kWh": 11.655, "E_W_gen_in_test_kWh": 13.0, "W_W_gen_test_kWh": 1.0},
    ]
    standby_point = {"E_W_gen_in_test_stb_kWh": 1.0, "W_W_gen_test_stb_kWh": 0.2}

    # Eq.(77)/(80): above the highest tested cycle -> scaled by the ratio.
    above = dhw_tapping_cycle_energy_kWh(23.31, test_points, standby_point)
    ratio = 23.31 / 11.655
    assert above["E_W_gen_in_kWh"] == pytest.approx(13.0 * ratio)
    assert above["W_W_gen_kWh"] == pytest.approx(1.0 * ratio)

    # Eq.(78)/(81): between the two test points -> linear interpolation.
    between = dhw_tapping_cycle_energy_kWh(8.75, test_points, standby_point)
    frac = (8.75 - 5.845) / (11.655 - 5.845)
    assert between["E_W_gen_in_kWh"] == pytest.approx(7.0 + (13.0 - 7.0) * frac)
    assert between["W_W_gen_kWh"] == pytest.approx(0.6 + (1.0 - 0.6) * frac)

    # Eq.(79)/(82): below the lowest test point -> interpolate from stand-by.
    below = dhw_tapping_cycle_energy_kWh(2.0, test_points, standby_point)
    frac_lo = 2.0 / 5.845
    assert below["E_W_gen_in_kWh"] == pytest.approx(1.0 + (7.0 - 1.0) * frac_lo)
    assert below["W_W_gen_kWh"] == pytest.approx(0.2 + (0.6 - 0.2) * frac_lo)

    with pytest.raises(ValueError):
        dhw_tapping_cycle_energy_kWh(5.0, [], standby_point)


def test_boiler_generator_15316_4_1_priority_mode_defaults_to_proportional_unchanged():
    """BoilerConfig.priority_mode defaults to 'proportional' - the pre-
    existing, unchanged combined-load behaviour - not the new clause 6.10
    'dhw_priority' sequential-operation mode."""

    calc = pybui.BoilerGeneratorCalculator(_boiler_config())
    assert calc.cfg.priority_mode == "proportional"


def test_boiler_generator_15316_4_1_dhw_priority_serves_dhw_before_heating_under_shortfall():
    """Clause 6.10 (Eq.(59)-(67), reinterpreted per docs/standards_errata.md):
    priority_mode='dhw_priority' serves DHW FIRST out of the step's capacity;
    heating only gets whatever capacity remains - unlike 'proportional',
    which would split the shortfall between both services."""

    config = _boiler_config(priority_mode="dhw_priority", rated_power_kW=5.0)
    calc = pybui.BoilerGeneratorCalculator(config)
    result = calc.compute_step(
        0.0, theta_avg_C=45.0, theta_return_C=35.0, Q_H_gen_out_kWh=7.0, Q_W_gen_out_kWh=3.0
    )

    # Combined demand (10.0) exceeds capacity (5.0): DHW is served in full
    # (3.0), heating gets only what is left (2.0), not a proportional share.
    assert result["Q_gen_out(kWh)"] == pytest.approx(5.0)
    assert result["Q_W_gen_out_kWh"] == pytest.approx(3.0)
    assert result["Q_W_gen_unmet_kWh"] == pytest.approx(0.0)
    assert result["Q_H_gen_out_kWh"] == pytest.approx(2.0)
    assert result["Q_H_gen_unmet_kWh"] == pytest.approx(5.0)

    # Per-service operating time/load ratio (Eq.(59)-(66), reinterpreted):
    # each service's own share of the step, not one blended beta_gen.
    assert result["t_W_op(h)"] == pytest.approx(3.0 / 5.0)
    assert result["t_H_op(h)"] == pytest.approx(2.0 / 5.0)
    assert result["beta_W(-)"] == pytest.approx(3.0 / 5.0)
    assert result["beta_H(-)"] == pytest.approx(2.0 / 5.0)
    assert result["t_run(h)"] == pytest.approx(result["t_W_op(h)"] + result["t_H_op(h)"])


def test_boiler_generator_15316_4_1_dhw_priority_serves_both_in_full_without_shortfall():
    """Without a capacity shortfall, 'dhw_priority' still serves both
    services in full - priority only changes what happens when demand
    exceeds capacity, not demand that fits within it."""

    config = _boiler_config(priority_mode="dhw_priority")  # rated_power_kW=20.0, well above 10.0
    result = pybui.BoilerGeneratorCalculator(config).compute_step(
        0.0, theta_avg_C=45.0, theta_return_C=35.0, Q_H_gen_out_kWh=7.0, Q_W_gen_out_kWh=3.0
    )
    assert result["Q_gen_out(kWh)"] == pytest.approx(10.0)
    assert result["Q_H_gen_out_kWh"] == pytest.approx(7.0)
    assert result["Q_W_gen_out_kWh"] == pytest.approx(3.0)
    assert result["Q_H_gen_unmet_kWh"] == pytest.approx(0.0)
    assert result["Q_W_gen_unmet_kWh"] == pytest.approx(0.0)
    assert result["E_gen_in(kWh)"] >= result["Q_gen_out(kWh)"]


def test_boiler_generator_15316_4_1_unknown_priority_mode_rejected():
    with pytest.raises(ValueError):
        pybui.BoilerGeneratorCalculator(_boiler_config(priority_mode="first_come_first_served"))


def test_saturation_humidity_kg_per_m3_table_b12_lookup_and_interpolation():
    """Table B.12: exact table points plus linear interpolation/clamping."""

    from pybuildingenergy.source.generation_15316_4_1 import saturation_humidity_kg_per_m3

    assert saturation_humidity_kg_per_m3(0.0) == pytest.approx(0.00493)
    assert saturation_humidity_kg_per_m3(20.0) == pytest.approx(0.01912)
    assert saturation_humidity_kg_per_m3(70.0) == pytest.approx(0.3596)
    # Midpoint of two table points -> linear interpolation.
    assert saturation_humidity_kg_per_m3(25.0) == pytest.approx((0.01912 + 0.03521) / 2.0)
    # Outside the table's 0-70 degC range -> clamped to the nearest edge.
    assert saturation_humidity_kg_per_m3(-10.0) == pytest.approx(0.00493)
    assert saturation_humidity_kg_per_m3(90.0) == pytest.approx(0.3596)


def test_condensing_gain_fraction_eq15_to_eq24():
    """Eq.(15)-(24): the combustion-chemistry condensing gain, reproduced
    by hand from Table B.13's natural gas row (Hs=11.32, V_air;st;dry=9.79,
    V_fg;st;dry=8.80, m_H2O;st=1.47), Table B.14's defaults (x_air=50 %,
    x_fg=100 %, X_O2;fg;dry=6 %) and Table B.12's saturation humidity."""

    from pybuildingenergy.source.generation_15316_4_1 import (
        condensing_gain_fraction,
        saturation_humidity_kg_per_m3,
    )

    theta_return, theta_brm = 30.0, 20.0
    gain = condensing_gain_fraction("natural_gas", theta_return, theta_brm)

    theta_fg = theta_return + 10.0  # Eq.(15), default delta_theta_wfg=10 K
    V_fg_dry = 8.80 * 21.0 / (21.0 - 6.0)  # Eq.(16)
    V_air_dry = 9.79 + V_fg_dry - 8.80  # Eq.(17)
    m_H2O_air = saturation_humidity_kg_per_m3(theta_brm) * V_air_dry * 0.50  # Eq.(18)
    m_H2O_fg = saturation_humidity_kg_per_m3(theta_fg) * V_fg_dry * 1.00  # Eq.(19)
    m_H2O_cond = 1.47 + m_H2O_air - m_H2O_fg  # Eq.(20)
    assert m_H2O_cond > 0.0  # otherwise this test proves nothing
    h_cond = (694.61 - 0.6764 * theta_fg) / 1000.0  # Eq.(22), Wh/kg -> kWh/kg
    expected = (m_H2O_cond * h_cond) / 11.32  # Eq.(23)/(24)

    assert gain == pytest.approx(expected)
    assert gain > 0.0

    # Eq.(20): no condensation (m_H2O;cond <= 0) at a high enough return
    # temperature -> the fraction is exactly 0, not negative.
    assert condensing_gain_fraction("natural_gas", 70.0, 20.0) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        condensing_gain_fraction("hydrogen", 30.0, 20.0)


def test_boiler_generator_15316_4_1_combustion_fuel_key_overrides_dew_point_approximation():
    """combustion_fuel_key (None by default) switches _eta_corrected_pct()
    from the flat condensing_gain_pct/dew_point_C approximation to the
    Eq.(15)-(24) combustion-chemistry gain - opt-in, not a default-behaviour
    change."""

    from pybuildingenergy.source.generation_15316_4_1 import condensing_gain_fraction

    config = _boiler_config()  # condensing, dew_point_C=55, condensing_gain_pct=11
    calc_default = pybui.BoilerGeneratorCalculator(config)
    assert calc_default.cfg.combustion_fuel_key is None

    calc_chem = pybui.BoilerGeneratorCalculator({**config, "combustion_fuel_key": "natural_gas"})
    assert calc_chem.cfg.combustion_fuel_key == "natural_gas"
    assert calc_chem.cfg.theta_brm_C == pytest.approx(20.0)  # Table B.8 default, inside_heated

    r_default = calc_default.compute_step(10.0, theta_avg_C=45.0, theta_return_C=30.0)
    r_chem = calc_chem.compute_step(10.0, theta_avg_C=45.0, theta_return_C=30.0)
    assert r_default["eta_Pn_corr(%)"] != pytest.approx(r_chem["eta_Pn_corr(%)"])

    gain = condensing_gain_fraction("natural_gas", 30.0, 20.0)
    eta_corr_base = config["eta_Pn_test_pct"] - config["f_corr_pct_per_K"] * (45.0 - config["theta_test_Pn_C"])
    assert r_chem["eta_Pn_corr(%)"] == pytest.approx(min(eta_corr_base + gain * 100.0, 110.0))


def test_boiler_generator_15316_4_1_theta_brm_defaults_by_location():
    from pybuildingenergy.source.generation_15316_4_1 import DEFAULT_THETA_BRM_C_BY_LOCATION

    for loc, expected in DEFAULT_THETA_BRM_C_BY_LOCATION.items():
        calc = pybui.BoilerGeneratorCalculator(_boiler_config(boiler_location=loc))
        assert calc.cfg.theta_brm_C == pytest.approx(expected)

    overridden = pybui.BoilerGeneratorCalculator(_boiler_config(theta_brm_C=15.0))
    assert overridden.cfg.theta_brm_C == pytest.approx(15.0)


def test_boiler_generator_15316_4_1_unknown_combustion_fuel_key_rejected():
    with pytest.raises(ValueError):
        pybui.BoilerGeneratorCalculator(_boiler_config(combustion_fuel_key="hydrogen"))


def test_chimney_loss_pct_from_flue_gas_analysis_eq25_to_eq26():
    """Eq.(25)-(26): chimney loss with the burner on, from a flue gas O2
    analysis and Table B.10a's Siegert constants - a standalone, optional
    alternative derivation, not wired into compute_step()."""

    from pybuildingenergy.source.generation_15316_4_1 import chimney_loss_pct_from_flue_gas_analysis

    # Eq.(25): natural gas, c10=0.66, c11=0.009.
    measured = chimney_loss_pct_from_flue_gas_analysis(
        "natural_gas", theta_fg_C=180.0, theta_brm_C=20.0, X_O2_measured_pct=4.0
    )
    expected = (180.0 - 20.0) * (0.66 / (21.0 - 4.0) + 0.009)
    assert measured == pytest.approx(expected)

    # Eq.(26): corrected from measured to reference water temperature.
    corrected = chimney_loss_pct_from_flue_gas_analysis(
        "natural_gas", theta_fg_C=180.0, theta_brm_C=20.0, X_O2_measured_pct=4.0,
        theta_gen_ref_C=70.0, theta_gen_meas_C=60.0,
    )
    assert corrected == pytest.approx(measured - (70.0 - 60.0) * 0.045)

    with pytest.raises(ValueError):
        chimney_loss_pct_from_flue_gas_analysis("hydrogen", 180.0, 20.0, 4.0)


def test_fuel_power_input_kW_eq13():
    from pybuildingenergy.source.generation_15316_4_1 import fuel_power_input_kW

    # Eq.(13): P_gen,del = fuel flow rate * Hi (both already in kWh units).
    assert fuel_power_input_kW(2.0, 10.22) == pytest.approx(20.44)


def test_split_standby_loss_pct_eq9_inverted():
    """split_standby_loss_pct() splits a total P0 loss fraction into its
    chimney-off/envelope shares using Table B.7's ratio, and the two
    shares must always sum back to the original total (Eq.(9))."""

    from pybuildingenergy.source.generation_15316_4_1 import split_standby_loss_pct

    f_ch_off, f_gen_env = split_standby_loss_pct(f_gen_ls_P0_pct=8.5, f_gen_env_fraction=0.75)
    assert f_gen_env == pytest.approx(8.5 * 0.75)
    assert f_ch_off == pytest.approx(8.5 * 0.25)
    assert f_ch_off + f_gen_env == pytest.approx(8.5)  # Eq.(9)


def test_boiler_efficiency_from_measured_values_eq7_to_eq12():
    """Eq.(7)-(12): full-load efficiency from measured/derived loss
    fractions and fuel power input, hand-derived from the formulas."""

    from pybuildingenergy.source.generation_15316_4_1 import boiler_efficiency_from_measured_values

    P_gen_del, f_ch_on, f_ch_off, f_gen_env = 20.0, 8.0, 2.0, 3.0

    result = boiler_efficiency_from_measured_values(P_gen_del, f_ch_on, f_ch_off, f_gen_env)
    P_ch_on = f_ch_on / 100.0 * P_gen_del  # Eq.(10)
    P_ch_off = f_ch_off / 100.0 * P_gen_del  # Eq.(11)
    P_env = f_gen_env / 100.0 * P_gen_del  # Eq.(12)
    assert result["P_gen_ls_ch_on_kW"] == pytest.approx(P_ch_on)
    assert result["P_gen_ls_ch_off_kW"] == pytest.approx(P_ch_off)
    assert result["P_gen_ls_env_kW"] == pytest.approx(P_env)
    assert result["f_gen_ls_P0_pct"] == pytest.approx(f_ch_off + f_gen_env)  # Eq.(9)
    expected_eta_Pn = (P_gen_del - P_ch_on - P_env) / P_gen_del * 100.0  # Eq.(7)
    assert result["eta_Pn_pct"] == pytest.approx(expected_eta_Pn)
    assert "eta_Pint_pct" not in result  # beta_Pint not given

    # Eq.(8): part-load efficiency, with a condensation credit.
    beta_Pint, P_cond = 0.3, 0.2
    with_pint = boiler_efficiency_from_measured_values(
        P_gen_del, f_ch_on, f_ch_off, f_gen_env, beta_Pint=beta_Pint, P_cond_kW=P_cond
    )
    numerator = (
        P_gen_del * beta_Pint
        - beta_Pint * (P_ch_on - P_cond + P_env)
        + (1.0 - beta_Pint) * (P_ch_off + P_env)
    )
    expected_eta_Pint = numerator / (P_gen_del * beta_Pint) * 100.0
    assert with_pint["eta_Pint_pct"] == pytest.approx(expected_eta_Pint)

    # A higher condensation credit must raise eta_Pint (more heat recovered
    # during the "on" phase, at the same loss fractions).
    more_credit = boiler_efficiency_from_measured_values(
        P_gen_del, f_ch_on, f_ch_off, f_gen_env, beta_Pint=beta_Pint, P_cond_kW=1.0
    )
    assert more_credit["eta_Pint_pct"] > with_pint["eta_Pint_pct"]


def test_rated_power_from_measured_efficiency_kW_eq14():
    from pybuildingenergy.source.generation_15316_4_1 import rated_power_from_measured_efficiency_kW

    # Eq.(14): Pn = eta_gen,Pn * P_gen,del (eta on this module's 0-100 scale).
    assert rated_power_from_measured_efficiency_kW(90.0, 20.0) == pytest.approx(18.0)


def test_boiler_generator_15316_4_1_eq46_p0_temperature_correction_is_opt_in():
    """Eq.(46): apply_p0_temp_correction=False (the default) keeps P0 a
    flat constant - unchanged pre-existing behaviour. True scales it by
    [(theta_avg-theta_brm)/(theta_test_P0-20)]^1.25, inert exactly at test
    conditions (theta_avg=theta_test_P0, theta_brm=20)."""

    config = _boiler_config(boiler_type="standard", f_corr_pct_per_K=0.0)
    config["efficiency_table"] = {"standard": config["efficiency_table"]["condensing"]}
    config["loss_table"] = {"standard": config["loss_table"]["condensing"]}

    baseline = pybui.BoilerGeneratorCalculator(config).compute_step(
        10.0, theta_avg_C=45.0, theta_return_C=35.0
    )
    assert baseline["P_gen_ls_P0(kW)"] == pytest.approx(config["P_gen_ls_P0_W"] / 1000.0)

    corrected = pybui.BoilerGeneratorCalculator(
        {**config, "apply_p0_temp_correction": True}
    ).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)
    expected_ratio = ((45.0 - 20.0) / (70.0 - 20.0)) ** 1.25  # theta_brm=20 (inside_heated default)
    assert corrected["P_gen_ls_P0(kW)"] == pytest.approx(
        config["P_gen_ls_P0_W"] / 1000.0 * expected_ratio
    )
    assert corrected["P_gen_ls_P0(kW)"] < baseline["P_gen_ls_P0(kW)"]

    at_test = pybui.BoilerGeneratorCalculator(
        {**config, "apply_p0_temp_correction": True}
    ).compute_step(10.0, theta_avg_C=70.0, theta_return_C=35.0)  # theta_test_P0_C default = 70
    assert at_test["P_gen_ls_P0(kW)"] == pytest.approx(config["P_gen_ls_P0_W"] / 1000.0)


def test_boiler_generator_15316_4_1_eq40_dhw_trace_band_is_opt_in():
    """Eq.(40): dhw_trace_band=False (the default) keeps Eq.(41)'s
    circulation-loop full-load efficiency correction - unchanged
    pre-existing behaviour. True replaces it with eta_Pn;traceband =
    eta_Pn + f_corr_Pn*(50-theta_Wc;mn), affecting ONLY eta_Pn - not
    eta_Pint, which keeps using Eq.(44) regardless."""

    config = _boiler_config(boiler_type="standard", f_corr_pct_per_K=0.04)
    config["efficiency_table"] = {"standard": config["efficiency_table"]["condensing"]}
    config["loss_table"] = {"standard": config["loss_table"]["condensing"]}

    baseline = pybui.BoilerGeneratorCalculator(config).compute_step(
        10.0, theta_avg_C=45.0, theta_return_C=35.0
    )
    trace_band = pybui.BoilerGeneratorCalculator(
        {**config, "dhw_trace_band": True}
    ).compute_step(10.0, theta_avg_C=45.0, theta_return_C=35.0)

    eta_test_pn = config["eta_Pn_test_pct"]
    expected_baseline_eta_Pn = eta_test_pn - 0.04 * (45.0 - config["theta_test_Pn_C"])  # Eq.(41)
    expected_trace_band_eta_Pn = eta_test_pn + 0.04 * (50.0 - 45.0)  # Eq.(40)
    assert baseline["eta_Pn_corr(%)"] == pytest.approx(expected_baseline_eta_Pn)
    assert trace_band["eta_Pn_corr(%)"] == pytest.approx(expected_trace_band_eta_Pn)
    assert baseline["eta_Pn_corr(%)"] != pytest.approx(trace_band["eta_Pn_corr(%)"])

    # eta_Pint is unaffected (Eq.(40) only replaces Eq.(41), not Eq.(44)).
    assert baseline["eta_Pint_corr(%)"] == pytest.approx(trace_band["eta_Pint_corr(%)"])


def _tapping_cycle_config(**overrides):
    config = {
        "boiler_type": "dhw_tapping_cycle_appliance", "fuel_type": "natural_gas",
        "boiler_location": "inside_heated",
        "tapping_cycle_test_points": [
            {"Q_W_test_kWh": 5.845, "E_W_gen_in_test_kWh": 7.0, "W_W_gen_test_kWh": 0.6},
            {"Q_W_test_kWh": 11.655, "E_W_gen_in_test_kWh": 13.0, "W_W_gen_test_kWh": 1.0},
        ],
        "tapping_cycle_standby_point": {"E_W_gen_in_test_stb_kWh": 1.0, "W_W_gen_test_stb_kWh": 0.2},
    }
    config.update(overrides)
    return config


def test_boiler_generator_15316_4_1_dhw_tapping_cycle_compute_step_not_single_step():
    """Clause 6.12 is a DAILY method - compute_step() must say so clearly
    rather than silently misapplying a daily formula to one hour."""

    calc = pybui.BoilerGeneratorCalculator(_tapping_cycle_config())
    with pytest.raises(NotImplementedError):
        calc.compute_step(1.0)


def test_boiler_generator_15316_4_1_dhw_tapping_cycle_run_timeseries_needs_datetime_index():
    calc = pybui.BoilerGeneratorCalculator(_tapping_cycle_config())
    with pytest.raises(ValueError):
        calc.run_timeseries(pd.DataFrame({"Q_W_kWh": [1.0] * 24}))


def test_boiler_generator_15316_4_1_dhw_tapping_cycle_run_timeseries_aggregates_by_day():
    """run_timeseries() must aggregate hourly demand by CALENDAR DAY, call
    dhw_tapping_cycle_energy_kWh() once per day with that day's total, and
    distribute the result back across the day's hours proportionally -
    reproducing dhw_tapping_cycle_energy_kWh() called directly."""

    from pybuildingenergy.source.generation_15316_4_1 import dhw_tapping_cycle_energy_kWh

    cfg = _tapping_cycle_config()
    calc = pybui.BoilerGeneratorCalculator(cfg)

    day1_total, day2_total = 8.75, 2.0
    index = pd.date_range("2020-01-01", periods=48, freq="h")
    demand = [day1_total / 24] * 24 + [day2_total / 24] * 24
    df = pd.DataFrame({"Q_W_kWh": demand}, index=index)

    result = calc.run_timeseries(df)
    by_day = result.timeseries.groupby(result.timeseries.index.normalize())

    expected_day1 = dhw_tapping_cycle_energy_kWh(
        day1_total, cfg["tapping_cycle_test_points"], cfg["tapping_cycle_standby_point"]
    )
    expected_day2 = dhw_tapping_cycle_energy_kWh(
        day2_total, cfg["tapping_cycle_test_points"], cfg["tapping_cycle_standby_point"]
    )
    day1_key, day2_key = sorted(by_day.groups)
    assert by_day["E_gen_in(kWh)"].sum()[day1_key] == pytest.approx(expected_day1["E_W_gen_in_kWh"])
    assert by_day["W_gen_aux(kWh)"].sum()[day1_key] == pytest.approx(expected_day1["W_W_gen_kWh"])
    assert by_day["E_gen_in(kWh)"].sum()[day2_key] == pytest.approx(expected_day2["E_W_gen_in_kWh"])
    assert by_day["Q_gen_out(kWh)"].sum()[day1_key] == pytest.approx(day1_total)
    assert result.summary["EHW_gen_in_kWh"] == pytest.approx(
        expected_day1["E_W_gen_in_kWh"] + expected_day2["E_W_gen_in_kWh"]
    )


def test_boiler_generator_15316_4_1_dhw_tapping_cycle_zero_demand_day_splits_standby_evenly():
    """A zero-demand day still has to account for the appliance's own
    stand-by energy (Eq.(79)/(82) at Qd=0) - split evenly across that
    day's hours, since there is no demand to weight the share by."""

    from pybuildingenergy.source.generation_15316_4_1 import dhw_tapping_cycle_energy_kWh

    cfg = _tapping_cycle_config()
    calc = pybui.BoilerGeneratorCalculator(cfg)
    index = pd.date_range("2020-01-01", periods=24, freq="h")
    df = pd.DataFrame({"Q_W_kWh": [0.0] * 24}, index=index)

    result = calc.run_timeseries(df)
    expected = dhw_tapping_cycle_energy_kWh(
        0.0, cfg["tapping_cycle_test_points"], cfg["tapping_cycle_standby_point"]
    )
    assert result.timeseries["E_gen_in(kWh)"].sum() == pytest.approx(expected["E_W_gen_in_kWh"])
    assert result.timeseries["E_gen_in(kWh)"].iloc[0] == pytest.approx(expected["E_W_gen_in_kWh"] / 24.0)


def test_district_15316_4_5_can_include_cooling_branch():
    result = pybui.DistrictEnergySystemCalculator(
        {"cooling_enabled": True, "heating_substation_efficiency": 0.97}
    ).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["QC_gen_out_kWh"] == pytest.approx(2.4)
    assert result.summary["E_total_district_kWh"] > result.summary["QHW_gen_out_kWh"]


def test_cogeneration_15316_4_4_reports_electricity():
    result = pybui.CogenerationSystemCalculator(
        {
            "nominal_thermal_power_kW": 5.0,
            "thermal_efficiency": 0.56,
            "electrical_efficiency": 0.30,
        }
    ).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["E_chp_el_generated_kWh"] > 0.0
    assert result.summary["E_chp_el_self_consumed_kWh"] > 0.0


def test_renewables_15316_4_3_4_6_reduce_dhw_and_generate_pv():
    result = pybui.RenewableEnergySystemCalculator(
        {
            "solar_thermal": {
                "enabled": True,
                "area_m2": 2.0,
                "annual_yield_kWh_m2": 400.0,
            },
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 1200.0,
            },
        }
    ).run_timeseries(_loads())

    assert result.summary["Q_solar_thermal_used_kWh"] > 0.0
    assert result.summary["QW_after_solar_kWh"] < result.summary["QW_before_solar_kWh"]
    assert result.summary["E_PV_gen_kWh"] == pytest.approx(2400.0)


def test_primary_energy_52000_1_reports_delivered_and_export_credit():
    result = pybui.PrimaryEnergyAccountingCalculator(
        {
            "export_credit_method": "symmetric",
            "primary_energy_factors": {
                "electricity": {"nonrenewable": 2.0, "renewable": 0.0},
                "natural_gas": {"nonrenewable": 1.1, "renewable": 0.0},
            },
            "carriers": ["electricity", "natural_gas"],
        }
    ).run_annual(
        {
            "E_delivered_electricity_kWh": 10.0,
            "E_delivered_natural_gas_kWh": 20.0,
            "E_exported_electricity_kWh": 3.0,
        }
    )

    assert result.summary["PE_delivered_nonren_kWh"] == pytest.approx(42.0)
    assert result.summary["PE_export_credit_nonren_kWh"] == pytest.approx(6.0)
    assert result.summary["PE_net_nonren_kWh"] == pytest.approx(36.0)


def test_lighting_15193_1_and_bacs_reduce_service_energy():
    loads = _loads()
    lighting = pybui.LightingSystemCalculator(
        {
            "area_m2": 100.0,
            "installed_power_density_W_m2": 5.0,
            "daylight_control_fraction": 0.4,
            "occupancy_control_fraction": 0.2,
        }
    ).run_timeseries(loads.assign(lighting_profile=1.0, occupancy_profile=1.0))

    bacs = pybui.BACSControlFactorCalculator(
        {"bacs_class": "B"}
    ).run_timeseries(
        pd.DataFrame({"E_L_total_kWh": lighting.timeseries["E_L_total_kWh"]})
    )

    assert lighting.summary["E_L_total_kWh"] > 0.0
    assert lighting.summary["LENI_kWh_m2a"] > 0.0
    assert bacs.summary["lighting_factor"] < 1.0
    assert (
        bacs.summary["lighting_E_L_total_kWh_after_kWh"]
        < bacs.summary["lighting_E_L_total_kWh_before_kWh"]
    )


def test_ventilation_16798_5_7_heat_recovery_and_fans():
    result = pybui.VentilationSystemCalculator(
        {
            "volume_m3": 300.0,
            "air_change_rate_h": 0.5,
            "heat_recovery_efficiency": 0.75,
            "specific_fan_power_W_s_m3": 800.0,
        }
    ).run_timeseries(_loads().assign(T_zone=20.0))

    assert result.summary["Q_vent_heat_recovered_kWh"] > 0.0
    assert (
        result.summary["Q_vent_heat_loss_after_HRV_kWh"]
        < result.summary["Q_vent_heat_loss_before_HRV_kWh"]
    )
    assert result.summary["E_vent_fan_kWh"] > 0.0


def test_cost_optimality_15459_1_global_cost():
    result = pybui.CostOptimalityCalculator(
        {
            "calculation_period_years": 30,
            "real_discount_rate": 0.03,
            "floor_area_m2": 100.0,
            "energy_prices": {"electricity": 0.30},
            "export_prices": {"electricity": 0.05},
        }
    ).calculate(
        components=[
            {
                "investment_EUR": 10_000.0,
                "annual_maintenance_EUR": 100.0,
                "lifetime_years": 15,
            }
        ],
        annual_energy={"electricity": 1000.0},
        annual_exports={"electricity": 100.0},
    )

    assert result.summary["Annual_energy_cost_EUR"] == pytest.approx(300.0)
    assert result.summary["Annual_export_value_EUR"] == pytest.approx(5.0)
    assert result.summary["Global_cost_EUR_m2"] > 100.0


def test_biomass_boiler_15316_4_1_reports_biomass_fuel_delivered():
    config = {
        "boiler_type": "biomass_pellet", "fuel_type": "biomass_pellet", "rated_power_kW": 20.0,
        "intermediate_load_fraction": 0.30, "eta_Pn_test_pct": 90.0, "eta_Pint_test_pct": 92.0,
        "theta_test_Pn_C": 70.0, "theta_test_Pint_C": 50.0, "f_corr_pct_per_K": 0.02,
        "P_gen_ls_P0_W": 250.0, "P_aux_on_W": 90.0, "P_aux_off_W": 8.0, "f_jacket": 0.50,
        "f_location": 1.0, "f_aux_recoverable": 0.75, "dew_point_C": 55.0, "condensing_gain_pct": 0.0,
        "efficiency_table": {"biomass_pellet": {"eta_Pn_test_pct": 90.0, "eta_Pint_test_pct": 92.0, "theta_test_Pn_C": 70.0, "theta_test_Pint_C": 50.0}},
        "loss_table": {"biomass_pellet": {"P_gen_ls_P0_W": 250.0}},
        "boiler_location": "inside_heated",
    }
    result = pybui.BoilerGeneratorCalculator(config).run_timeseries(_loads())

    assert result.summary["QHW_gen_out_kWh"] == pytest.approx(28.8)
    assert result.summary["E_total_fuel_kWh"] > result.summary["QHW_gen_out_kWh"]


def test_renewables_battery_dispatch_reduces_grid_import():
    without_battery = pybui.RenewableEnergySystemCalculator(
        {
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 100.0,
                "self_consumption_fraction": 1.0,
            }
        }
    ).run_timeseries(_loads())
    with_battery = pybui.RenewableEnergySystemCalculator(
        {
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 100.0,
                "self_consumption_mode": "dispatch",
                "battery": {"capacity_kWh": 2.0},
            }
        }
    ).run_timeseries(_loads())

    assert with_battery.summary["E_battery_discharge_to_load_kWh"] > 0.0
    assert (
        with_battery.summary["E_grid_after_PV_kWh"]
        < without_battery.summary["E_grid_after_PV_kWh"]
    )


def test_renewables_pv_area_kpk_is_used_by_annual_method():
    data = pd.DataFrame({"E_site_el_load_kWh": [0.0]})
    result = pybui.RenewableEnergySystemCalculator(
        {
            "pv": {
                "enabled": True,
                "area_m2": 10.0,
                "peak_power_coefficient_kW_m2": 0.20,
                "annual_yield_kWh_kWp": 1000.0,
            }
        }
    ).run_timeseries(data)

    assert result.summary["E_PV_gen_kWh"] == pytest.approx(2000.0)


def test_renewables_pv_annual_orientation_factor_is_applied():
    data = pd.DataFrame({"E_site_el_load_kWh": [0.0]})
    result = pybui.RenewableEnergySystemCalculator(
        {
            "pv": {
                "enabled": True,
                "capacity_kWp": 2.0,
                "annual_yield_kWh_kWp": 1000.0,
                "tilt_orientation_factor": 0.8,
            }
        }
    ).run_timeseries(data)

    assert result.summary["E_PV_gen_kWh"] == pytest.approx(1600.0)


def test_renewables_method_3_clips_negative_losses_and_applies_pump_threshold():
    data = pd.DataFrame(
        {
            "G_pv_W_m2": [100.0, 1000.0],
            "T_ext_C": [60.0, 20.0],
            "E_site_el_load_kWh": [0.0, 0.0],
        }
    )
    result = pybui.RenewableEnergySystemCalculator(
        {
            "solar_thermal": {
                "enabled": True,
                "area_m2": 1.0,
                "collector_area_m2": 1.0,
                "method": "method_3",
                "mean_temperature_C": 50.0,
                "loop_heat_loss_W_K": 10.0,
                "pump_power_W": 100.0,
            }
        }
    ).run_timeseries(data)

    ts = result.timeseries
    assert ts["Q_solar_thermal_loop_losses_kWh"].iloc[0] == pytest.approx(0.0)
    assert ts["Q_solar_thermal_available_kWh"].iloc[0] == pytest.approx(0.0)
