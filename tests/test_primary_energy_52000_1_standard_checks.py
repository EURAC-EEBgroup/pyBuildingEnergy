"""EN ISO 52000-1:2017 - hand calculations from the standard (verification).

Factors: Table B.16 (``factor_set="iso_52000_1_b16"``): electricity delivered and exported
2,3 / 0,2 / 420 g/kWh (nren / ren / CO2), PV on-site 0 / 1 / 0.  Expected values are worked
out by hand from Formulae (2), (19), (20), (23)-(28), (39); tests marked ``xfail`` document
the deviation left open (see the reason).
"""

import pandas as pd
import pytest

from pybuildingenergy.source.primary_energy_52000_1 import (
    ISO_52000_1_TABLE_B16_FACTORS,
    PrimaryEnergyAccountingCalculator as Calc,
)


def _run(df, **cfg):
    return Calc({"factor_set": "iso_52000_1_b16", **cfg}).run_timeseries(df).summary


def _pv_building():
    # delivered 100 kWh, PV production 80 kWh, exported 30 kWh
    return pd.DataFrame({
        "E_delivered_electricity_kWh": [100.0],
        "E_onsite_pv_kWh": [80.0],
        "E_exported_electricity_kWh": [30.0],
    })


def test_table_b16_values():
    f = ISO_52000_1_TABLE_B16_FACTORS
    assert f["natural_gas"] == {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.220}
    assert f["electricity"] == {"nonrenewable": 2.3, "renewable": 0.2, "co2": 0.420}
    assert f["district_heat"] == {"nonrenewable": 1.3, "renewable": 0.0, "co2": 0.260}
    assert f["district_cooling"] == {"nonrenewable": 1.3, "renewable": 0.0, "co2": 0.260}
    assert f["biomass"] == {"nonrenewable": 0.2, "renewable": 1.0, "co2": 0.040}


def test_formula_2_19_20_with_pv_export_k_exp_1():
    s = _run(_pv_building())
    # non-renewable: 100*2,3 - 30*2,3 (PV step A has f_nren = 0, step B 2,3)
    assert s["PE_net_nonren_kWh"] == pytest.approx(161.0)
    # renewable: 100*0,2 + 80*1 - [30*1 + 1*30*(0,2 - 1)] = 94
    assert s["PE_net_ren_kWh"] == pytest.approx(94.0)
    assert s["CO2_net_kg"] == pytest.approx(0.42 * 70.0)


@pytest.mark.parametrize("k_exp, nren, ren", [(0.0, 230.0, 70.0), (0.5, 195.5, 82.0), (1.0, 161.0, 94.0)])
def test_k_exp_table_b17(k_exp, nren, ren):
    s = _run(_pv_building(), k_exp=k_exp)
    assert s["PE_net_nonren_kWh"] == pytest.approx(nren)
    assert s["PE_net_ren_kWh"] == pytest.approx(ren)


def test_formula_39_non_exported_carriers():
    df = pd.DataFrame({"E_delivered_natural_gas_kWh": [100.0], "E_delivered_district_heat_kWh": [50.0]})
    s = _run(df)
    assert s["PE_net_nonren_kWh"] == pytest.approx(100 * 1.1 + 50 * 1.3)
    assert s["CO2_net_kg"] == pytest.approx(100 * 0.22 + 50 * 0.26)


def test_rer_all_perimeters_matches_the_tr():
    # ISO/TR 52000-2:2017, 9.7.4 and Table 12/16: with the distant perimeter included
    # the grid electricity (100*0,2) counts and the export credit is netted (30*1 - 30*0,8 = 6)
    s = _run(_pv_building(), rer_perimeters_ren=("onsite", "nearby", "distant"))
    assert s["PE_RER_ren_kWh"] == pytest.approx(20.0 + 80.0 - 6.0)
    assert s["RER"] == pytest.approx(94.0 / 255.0)


def test_rer_default_perimeters_do_not_net_the_export_credit():
    # TR 9.7.4: gas 190, PV 40 (20 used, 20 exported), factors of the TR
    factors = {"natural_gas": {"nonrenewable": 1.1, "renewable": 0.0},
               "electricity": {"nonrenewable": 2.3, "renewable": 0.2}}
    df = pd.DataFrame({"E_delivered_natural_gas_kWh": [190.0], "E_delivered_electricity_kWh": [0.0],
                       "E_onsite_pv_kWh": [40.0], "E_exported_electricity_kWh": [20.0]})
    assert _run(df, primary_energy_factors=factors)["RER"] == pytest.approx(40.0 / 199.0, abs=5e-4)


_TR_FACTORS = {"natural_gas": {"nonrenewable": 1.1, "renewable": 0.0},
               "electricity": {"nonrenewable": 2.3, "renewable": 0.2}}


def test_tr_9_7_4_single_family_house_examples():
    base = pd.DataFrame({"E_delivered_natural_gas_kWh": [190.0], "E_delivered_electricity_kWh": [20.0]})
    s = _run(base, primary_energy_factors=_TR_FACTORS, rer_perimeters_ren=("onsite", "nearby", "distant"))
    assert s["PE_net_nonren_kWh"] == pytest.approx(255.0)
    assert s["RER"] == pytest.approx(0.015, abs=5e-4)
    assert _run(base, primary_energy_factors=_TR_FACTORS)["RER"] == 0.0

    solar = pd.DataFrame({"E_delivered_natural_gas_kWh": [170.0], "E_delivered_electricity_kWh": [20.0],
                          "E_onsite_solar_thermal_kWh": [20.0]})
    s = _run(solar, primary_energy_factors=_TR_FACTORS)
    assert s["PE_net_nonren_kWh"] == pytest.approx(233.0)
    assert s["RER"] == pytest.approx(0.077, abs=1e-3)
    s = _run(solar, primary_energy_factors=_TR_FACTORS, rer_perimeters_ren=("onsite", "nearby", "distant"))
    assert s["RER"] == pytest.approx(0.093, abs=1e-3)

    pv = pd.DataFrame({"E_delivered_natural_gas_kWh": [190.0], "E_delivered_electricity_kWh": [0.0],
                       "E_onsite_pv_kWh": [40.0], "E_exported_electricity_kWh": [20.0]})
    s = _run(pv, primary_energy_factors=_TR_FACTORS, rer_perimeters_ren=("onsite", "nearby", "distant"))
    assert s["PE_net_nonren_kWh"] == pytest.approx(163.0)
    assert s["PE_RER_tot_kWh"] == pytest.approx(199.0)
    assert s["RER"] == pytest.approx(0.181, abs=5e-4)


def _office():
    # ISO/TR 52000-2:2017, Tables 9 and 11 (office 2 000 m2): grid 53 588 kWh (2,0 / 0,5),
    # gas 119 936 kWh (1,1), PV 30 000 kWh of which 3 659 kWh exported
    factors = {"natural_gas": {"nonrenewable": 1.1, "renewable": 0.0},
               "electricity": {"nonrenewable": 2.0, "renewable": 0.5}}
    df = pd.DataFrame({"E_delivered_natural_gas_kWh": [119936.0], "E_delivered_electricity_kWh": [53588.0],
                       "E_onsite_pv_kWh": [30000.0], "E_exported_electricity_kWh": [3659.0]})
    return df, factors


@pytest.mark.parametrize("k_exp, ren, nren, tot", [(0.0, 53135, 239104, 292239), (1.0, 54964, 231787, 286751)])
def test_tr_table_11_office_step_a_and_step_b(k_exp, ren, nren, tot):
    df, factors = _office()
    s = _run(df, primary_energy_factors=factors, k_exp=k_exp, allow_negative=True)
    assert s["PE_net_ren_kWh"] == pytest.approx(ren, abs=3)
    assert s["PE_net_nonren_kWh"] == pytest.approx(nren, abs=3)
    assert s["PE_net_total_kWh"] == pytest.approx(tot, abs=3)


def test_tr_table_12_office_rer():
    df, factors = _office()
    s = _run(df, primary_energy_factors=factors)
    assert s["PE_RER_ren_kWh"] == pytest.approx(30000.0, abs=3)
    assert s["RER"] == pytest.approx(0.105, abs=5e-4)


def test_tr_figure_15_cogeneration_step_a_b_with_negative_result():
    # ISO/TR 52000-2:2017, 11.6.2.1: Step A factors of the exported cogenerated electricity are
    # input (fA_*); k_exp = 1 gives Epren 4 713 and Epnren -713 kWh (negative: "not wrong values")
    df = pd.DataFrame({
        "E_delivered_biogas_kWh": [5000.0], "E_delivered_natural_gas_kWh": [2222.0],
        "E_delivered_electricity_kWh": [0.0], "E_exported_electricity_kWh": [1435.0],
        "fA_electricity_ren": [3116 / 1435], "fA_electricity_nonren": [156 / 1435], "fA_electricity_co2": [0.0],
    })
    factors = {"biogas": {"nonrenewable": 0.05, "renewable": 1.0},
               "natural_gas": {"nonrenewable": 1.05, "renewable": 0.0},
               "electricity": {"nonrenewable": 2.3, "renewable": 0.2}}
    cfg = dict(primary_energy_factors=factors, allow_negative=True, carriers=["biogas", "natural_gas", "electricity"],
               carrier_aliases={"biogas": ["E_delivered_biogas_kWh"]})
    s = _run(df, k_exp=1.0, **cfg)
    assert s["PE_net_ren_kWh"] == pytest.approx(4713, abs=3)
    assert s["PE_net_nonren_kWh"] == pytest.approx(-713, abs=6)  # inputs of the figure are rounded
    s = _run(df, k_exp=0.3, **cfg)
    assert s["PE_net_ren_kWh"] == pytest.approx(2733, abs=3)
    assert s["PE_net_nonren_kWh"] == pytest.approx(1486, abs=6)
    del cfg["allow_negative"]  # the default keeps the negative result of the TR
    assert _run(df, k_exp=1.0, **cfg)["PE_net_nonren_kWh"] == pytest.approx(-713, abs=6)


def test_default_does_not_clip_intervals_of_net_export():
    hourly = pd.DataFrame({"E_delivered_electricity_kWh": [100.0, 0.0],
                           "E_exported_electricity_kWh": [0.0, 100.0]})
    assert _run(hourly)["PE_net_nonren_kWh"] == pytest.approx(0.0)


def test_allow_negative_gives_the_annual_formula_2():
    hourly = pd.DataFrame({"E_delivered_electricity_kWh": [100.0, 0.0],
                           "E_exported_electricity_kWh": [0.0, 100.0]})
    assert _run(hourly, allow_negative=True)["PE_net_nonren_kWh"] == pytest.approx(0.0)


def test_carrier_without_weighting_factors_is_rejected():
    # Table A.16: the factors are input data; no silent 1,0 / CO2 0
    from pybuildingenergy.source.primary_energy_52000_1 import MissingWeightingFactors

    cfg = {"carriers": ["oil"], "carrier_aliases": {"oil": ["E_delivered_oil_kWh"]}}
    df = pd.DataFrame({"E_delivered_oil_kWh": [100.0]})
    with pytest.raises(MissingWeightingFactors, match="oil"):
        Calc(cfg).run_timeseries(df)
    with pytest.raises(MissingWeightingFactors):
        Calc({**cfg, "factor_set": "default"}).run_timeseries(df)
    given = Calc({**cfg, "primary_energy_factors": {"oil": {"nonrenewable": 1.2, "renewable": 0.0, "co2": 0.3}}})
    assert given.run_timeseries(df).summary["PE_net_nonren_kWh"] == pytest.approx(120.0)


@pytest.mark.parametrize("carrier, nren, ren, co2", [
    ("coal", 1.1, 0.0, 0.360), ("oil", 1.1, 0.0, 0.290), ("fossil_gaseous", 1.1, 0.0, 0.220),
    ("biofuel_solid", 0.2, 1.0, 0.040), ("biofuel_liquid", 0.5, 1.0, 0.070), ("biogas", 0.4, 1.0, 0.100),
])
def test_table_b16_covers_the_classes_of_carriers(carrier, nren, ren, co2):
    calc = Calc({"factor_set": "iso_52000_1_b16", "carriers": [carrier],
                 "carrier_aliases": {carrier: [f"E_{carrier}"]}})
    s = calc.run_timeseries(pd.DataFrame({f"E_{carrier}": [100.0]})).summary
    assert s["PE_net_nonren_kWh"] == pytest.approx(100 * nren)
    assert s["PE_net_ren_kWh"] == pytest.approx(100 * ren)
    assert s["CO2_net_kg"] == pytest.approx(100 * co2)


def test_time_dependent_factors_can_replace_the_static_ones():
    cfg = {"carriers": ["oil"], "carrier_aliases": {"oil": ["E_delivered_oil_kWh"]}}
    cols = {f"f_delivered_oil_{k}": [v, v] for k, v in (("nonren", 1.5), ("ren", 0.0), ("co2", 0.3))}
    df = pd.DataFrame({"E_delivered_oil_kWh": [10.0, 10.0], **cols})
    assert Calc(cfg).run_timeseries(df).summary["PE_net_nonren_kWh"] == pytest.approx(30.0)
    del df["f_delivered_oil_co2"]
    with pytest.raises(ValueError):
        Calc(cfg).run_timeseries(df)


@pytest.mark.xfail(strict=True, reason=(
    "Table B.30: exported wind electricity is not taken into account under Step B "
    "(only Step A). The calculator applies Step B (avoided grid resources) to every "
    "exported electricity, whatever the on-site source."))
def test_wind_export_has_no_step_b():
    wind = pd.DataFrame({"E_delivered_electricity_kWh": [0.0], "E_onsite_wind_kWh": [100.0],
                         "E_exported_electricity_kWh": [100.0]})
    assert _run(wind, allow_negative=True)["PE_net_ren_kWh"] == pytest.approx(0.0)


@pytest.mark.parametrize("k_exp, avl, avl_ren, avl_nren", [(1.0, 0.0, 0.0, 0.0), (0.5, 15.0, 15.0, 0.0), (0.0, 30.0, 30.0, 0.0)])
def test_energy_available_outside_the_building_formulae_21_22(k_exp, avl, avl_ren, avl_nren):
    # exported 30 kWh of PV: E_exp;avl = 30*(1-k_exp); Step A weighted resources 30*(1-k_exp)*f_stepA(PV)
    s = _run(_pv_building(), k_exp=k_exp)
    assert s["E_exp_avl_total_kWh"] == pytest.approx(avl)
    assert s["E_exp_avl_electricity_kWh"] == pytest.approx(avl)
    assert s["PE_exp_avl_ren_kWh"] == pytest.approx(avl_ren)
    assert s["PE_exp_avl_nonren_kWh"] == pytest.approx(avl_nren)
    assert s["PE_exp_avl_total_kWh"] == pytest.approx(avl_ren + avl_nren)
    assert s["CO2_exp_avl_kg"] == pytest.approx(0.0)  # Step A of PV: no emissions


def test_energy_available_follows_the_step_a_factors_of_cogeneration():
    df = pd.DataFrame({"E_delivered_electricity_kWh": [0.0], "E_exported_electricity_kWh": [100.0],
                       "fA_electricity_ren": [0.1], "fA_electricity_nonren": [0.8], "fA_electricity_co2": [0.2]})
    s = _run(df, k_exp=0.25)
    assert s["E_exp_avl_total_kWh"] == pytest.approx(75.0)
    assert s["PE_exp_avl_nonren_kWh"] == pytest.approx(100 * 0.8 * 0.75)
    assert s["PE_exp_avl_ren_kWh"] == pytest.approx(100 * 0.1 * 0.75)
    assert s["CO2_exp_avl_kg"] == pytest.approx(100 * 0.2 * 0.75)
