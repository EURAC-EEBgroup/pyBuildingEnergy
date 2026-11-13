"""EN ISO 52000-1 weighted balance: k_exp, Step A/B, on-site renewables, RER, CO2, composer."""

from pathlib import Path

import pandas as pd
import pytest

import pybuildingenergy as pybui

EPW = Path(__file__).resolve().parent / "USA_CO_Denver.Intl.AP.725650_TMY3.epw"
FACTORS = {"electricity": {"nonrenewable": 2.3, "renewable": 0.2, "co2": 0.42}}
ROW = {"E_delivered_electricity_kWh": 10.0, "E_onsite_pv_kWh": 8.0, "E_exported_electricity_kWh": 3.0}


def _run(row=ROW, **opt):
    cfg = {"carriers": ["electricity"], "primary_energy_factors": FACTORS, **opt}
    return pybui.PrimaryEnergyAccountingCalculator(cfg).run_annual(row).summary


def test_onsite_pv_and_step_a_b_with_k_exp_one():
    s = _run()
    # nren: 2.3 * (10 - 3); ren: 0.2*10 + 8*1 - 3*(1 + (0.2 - 1)) = 9.4   (Formulas 2, 19, 20)
    assert s["PE_net_nonren_kWh"] == pytest.approx(16.1)
    assert s["PE_net_ren_kWh"] == pytest.approx(9.4)
    assert s["PE_export_credit_ren_kWh"] == pytest.approx(0.6)
    # RER (Formula 17, Table B.24): the on-site renewable production counts in full (the export
    # credit belongs to the distant perimeter, ISO/TR 52000-2 9.7.2) / total primary energy
    assert s["PE_RER_ren_kWh"] == pytest.approx(8.0)
    assert s["RER"] == pytest.approx(8.0 / 25.5)
    with_distant = _run(rer_perimeters_ren=["onsite", "nearby", "distant"])
    assert with_distant["PE_RER_ren_kWh"] == pytest.approx(0.2 * 10 + 8.0 - 0.6)
    assert s["CO2_net_kg"] == pytest.approx(0.42 * 7.0)


def test_k_exp_zero_ignores_the_export():
    s = _run(k_exp=0.0)
    # only what the building uses: grid import + the 5 kWh of PV used on site
    assert s["PE_net_nonren_kWh"] == pytest.approx(23.0)
    assert s["PE_net_ren_kWh"] == pytest.approx(0.2 * 10 + 5.0)
    assert _run(export_credit_method="none")["PE_net_nonren_kWh"] == pytest.approx(23.0)


def test_k_exp_scales_the_step_b_term_without_onsite_columns():
    row = {"E_delivered_electricity_kWh": 10.0, "E_exported_electricity_kWh": 4.0}
    assert _run(row)["PE_net_nonren_kWh"] == pytest.approx(2.3 * 6)
    assert _run(row, k_exp=0.5)["PE_net_nonren_kWh"] == pytest.approx(2.3 * 8)


def test_negative_balance_is_valid_and_only_the_total_can_be_floored():
    # ISO/TR 52000-2 11.6.2.3: a negative result with k_exp = 1 is "not a wrong value"
    row = {"E_delivered_electricity_kWh": 1.0, "E_exported_electricity_kWh": 4.0}
    assert _run(row)["PE_net_nonren_kWh"] == pytest.approx(2.3 * (1 - 4))
    assert _run(row, allow_negative=False)["PE_net_nonren_kWh"] == 0.0


def test_intervals_are_never_clipped_so_export_offsets_imports_over_the_balance():
    # Formula (2) is annual: 100 kWh imported in one interval, 100 kWh exported in another
    rows = pd.DataFrame({"E_delivered_electricity_kWh": [100.0, 0.0], "E_exported_electricity_kWh": [0.0, 100.0]})
    calc = pybui.PrimaryEnergyAccountingCalculator({"carriers": ["electricity"], "primary_energy_factors": FACTORS})
    result = calc.run_timeseries(rows)
    assert result.summary["PE_net_nonren_kWh"] == pytest.approx(0.0)
    assert list(result.timeseries["PE_net_nonren_kWh"]) == pytest.approx([230.0, -230.0])
    floored = pybui.PrimaryEnergyAccountingCalculator(
        {"carriers": ["electricity"], "primary_energy_factors": FACTORS, "allow_negative": False}
    ).run_timeseries(rows).summary
    assert floored["PE_net_nonren_kWh"] == pytest.approx(0.0)


def test_nearby_district_heat_counts_in_the_rer_numerator_distant_does_not():
    row = {"E_delivered_district_heat_kWh": 10.0, "E_delivered_electricity_kWh": 10.0}
    s = pybui.PrimaryEnergyAccountingCalculator(
        {"carriers": ["district_heat", "electricity"], "factor_set": "iso_52000_1_b16",
         "primary_energy_factors": {"district_heat": {"nonrenewable": 0.5, "renewable": 0.5}}}
    ).run_annual(row).summary
    assert s["PE_RER_ren_kWh"] == pytest.approx(5.0)                  # nearby only
    assert s["RER"] == pytest.approx(5.0 / (10.0 + 23.0 + 2.0))        # total includes distant electricity


def test_table_b16_factor_set_and_invalid_options():
    s = pybui.PrimaryEnergyAccountingCalculator({"factor_set": "iso_52000_1_b16", "carriers": ["natural_gas"]}).run_annual(
        {"E_delivered_natural_gas_kWh": 100.0}).summary
    assert s["PE_net_total_kWh"] == pytest.approx(110.0)
    assert s["CO2_net_kg"] == pytest.approx(22.0)
    with pytest.raises(ValueError):
        pybui.PrimaryEnergyAccountingCalculator({"factor_set": "nope"})
    with pytest.raises(ValueError):
        pybui.PrimaryEnergyAccountingCalculator({"k_exp": 1.5})
    with pytest.raises(ValueError):
        pybui.PrimaryEnergyAccountingCalculator({"perimeters": {"electricity": "far"}})


def test_per_row_step_a_columns_override_the_onsite_default():
    row = pd.DataFrame({"E_delivered_electricity_kWh": [10.0], "E_exported_electricity_kWh": [2.0],
                        "fA_electricity_nonren": [1.0], "fA_electricity_ren": [0.0]})
    s = pybui.PrimaryEnergyAccountingCalculator(
        {"carriers": ["electricity"], "primary_energy_factors": FACTORS, "k_exp": 0.0}).run_timeseries(row).summary
    assert s["PE_export_credit_nonren_kWh"] == pytest.approx(2.0)   # A only, k_exp = 0


def test_configurable_rer_perimeters_apply_to_numerator_and_denominator():
    row = {
        "E_delivered_district_heat_kWh": 10.0,
        "E_delivered_electricity_kWh": 10.0,
    }
    base = {
        "carriers": ["district_heat", "electricity"],
        "primary_energy_factors": {
            "district_heat": {"nonrenewable": 0.5, "renewable": 0.5},
            "electricity": {"nonrenewable": 2.0, "renewable": 0.0},
        },
    }
    default = pybui.PrimaryEnergyAccountingCalculator(base).run_annual(row).summary
    nearby_only = pybui.PrimaryEnergyAccountingCalculator({
        **base, "rer_perimeters_ren": ["nearby"], "rer_perimeters_tot": ["nearby"],
    }).run_annual(row).summary
    assert default["PE_RER_tot_kWh"] == pytest.approx(30.0)
    assert nearby_only["PE_RER_ren_kWh"] == pytest.approx(5.0)
    assert nearby_only["PE_RER_tot_kWh"] == pytest.approx(10.0)
    assert nearby_only["RER"] == pytest.approx(0.5)


def test_time_dependent_delivered_and_export_factors_are_used():
    row = pd.DataFrame({
        "E_delivered_electricity_kWh": [1.0, 1.0],
        "E_exported_electricity_kWh": [0.0, 1.0],
        "f_delivered_electricity_nonren": [1.0, 3.0],
        "f_exported_electricity_nonren": [2.0, 4.0],
    })
    s = pybui.PrimaryEnergyAccountingCalculator({
        "carriers": ["electricity"],
        "primary_energy_factors": {"electricity": {"nonrenewable": 9.0, "renewable": 0.0}},
        "k_exp": 1.0,
        # Keep the two rows in the same assessment balance for this factor test.
        "allow_negative": True,
    }).run_timeseries(row).summary
    assert s["PE_delivered_nonren_kWh"] == pytest.approx(4.0)
    assert s["PE_export_credit_nonren_kWh"] == pytest.approx(4.0)
    assert s["PE_net_nonren_kWh"] == pytest.approx(0.0)


def test_non_epb_onsite_use_is_a_separate_export_flow():
    s = pybui.PrimaryEnergyAccountingCalculator({
        "carriers": ["electricity"],
        "primary_energy_factors": FACTORS,
        "k_exp": 1.0,
    }).run_annual({
        "E_delivered_electricity_kWh": 10.0,
        "E_exported_electricity_to_grid_kWh": 2.0,
        "E_exported_electricity_to_non_epb_kWh": 3.0,
    }).summary
    assert s["E_exported_electricity_kWh"] == pytest.approx(5.0)
    assert s["E_exported_electricity_grid_kWh"] == pytest.approx(2.0)
    assert s["E_exported_electricity_non_epb_kWh"] == pytest.approx(3.0)
    assert s["PE_net_nonren_kWh"] == pytest.approx(2.3 * 5.0)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("zone_cache_pe")


def _scenario(tmp_path, generation="boiler_15316_4_1", pv=None, **opt):
    return {"generation": generation, "pv": pv, "weather_source": "epw", "weather_file": str(EPW),
            "options": {"hourly_cache_dir": str(tmp_path), **opt}}


def test_composer_reports_rer_co2_and_onsite_pv(cache_dir):
    base = pybui.simulate(pybui.compose_config(_scenario(cache_dir))).summary
    # k_exp = 0: only the PV used on site counts; with k_exp = 1 the export can push EP_tot below 0 and RER is undefined
    pv = pybui.simulate(pybui.compose_config(_scenario(cache_dir, pv="pv_with_battery_dispatch", pe_k_exp=0.0))).summary
    assert base["RER"] == pytest.approx(0.0, abs=0.05) and base["E_onsite_renewable_kWh"] == 0.0
    assert pv["E_onsite_renewable_kWh"] == pytest.approx(pv["E_PV_gen_kWh"])
    assert pv["RER"] > base["RER"] and pv["PE_net_ren_kWh"] > base["PE_net_ren_kWh"]
    assert pv["PE_net_nonren_kWh"] < base["PE_net_nonren_kWh"]  # self-consumed PV avoids grid electricity
    full = pybui.simulate(pybui.compose_config(_scenario(cache_dir, pv="pv_with_battery_dispatch"))).summary
    assert full["PE_net_nonren_kWh"] < pv["PE_net_nonren_kWh"]
    assert full["CO2_net_kg"] < pv["CO2_net_kg"]
    assert "EN ISO 52000-1" in pybui.simulate(pybui.compose_config(_scenario(cache_dir))).report()


def test_composer_heat_pump_environment_heat_is_renewable_onsite(cache_dir):
    s = pybui.simulate(pybui.compose_config(_scenario(cache_dir, generation="air_to_water_heating_dhw_parallel_backup"))).summary
    assert s["E_onsite_renewable_kWh"] > 0.0
    assert s["RER"] > 0.0
