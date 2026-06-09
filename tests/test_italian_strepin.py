from pathlib import Path

import pandas as pd
import pytest

from pybuildingenergy.data.italian_strepin import load_italian_strepin_tables


WORKBOOK = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "strepin_archetypes"
    / "Italian_STREPIN_archetype_renovation_cost_models_visuals_reworked_fixed.xlsx"
)


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_workbook_tables_load():
    tables = load_italian_strepin_tables(WORKBOOK)

    assert len(tables.archetypes) == 18
    assert len(tables.packages) == 13
    assert len(tables.measure_levels) == 270
    assert "RMF_E1_E" in tables.archetype_ids
    assert "P09" in tables.package_ids


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_package_applies_to_bui_surfaces():
    tables = load_italian_strepin_tables(WORKBOOK)
    baseline = tables.make_case("RMF_E1_E", "P00")
    retrofit = tables.make_case("RMF_E1_E", "P09")

    def u_values(case, keyword):
        return [
            s["u_value"]
            for s in case.bui["building_surface"]
            if keyword in s["name"].lower()
        ]

    assert min(u_values(retrofit, "wall")) < min(u_values(baseline, "wall"))
    assert min(u_values(retrofit, "windows")) < min(u_values(baseline, "windows"))
    assert min(u_values(retrofit, "roof")) < min(u_values(baseline, "roof"))
    assert min(u_values(retrofit, "slab")) < min(u_values(baseline, "slab"))
    assert (
        retrofit.bui["strepin"]["heating_system"]["system_type"]
        == "air_to_water_heat_pump"
    )


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Italian STREPIN workbook not present")
def test_italian_strepin_engine_summary_uses_engine_demands():
    tables = load_italian_strepin_tables(WORKBOOK)
    case = tables.make_case("RMF_E1_B", "P10")
    annual = pd.DataFrame(
        [
            {
                "Q_H_annual_kWh": 12_000.0,
                "Q_C_annual_kWh": 900.0,
            }
        ]
    )

    from pybuildingenergy.data.italian_strepin import summarize_engine_performance

    out = summarize_engine_performance(case, annual, tables)

    assert out["Q_H_engine_kWh_a"] == pytest.approx(12_000.0)
    assert out["Q_C_engine_kWh_a"] == pytest.approx(900.0)
    assert out["Final_energy_kWh_m2a"] > 0.0
    assert out["Primary_nonren_kWh_m2a"] > 0.0
    assert out["Reference_workbook_primary_kWh_m2a"] is not None
