"""Composer <-> BoilerGeneratorCalculator integration (EN 15316-4-1).

composer.py's boiler_15316_4_1 branch runs ONE BoilerGeneratorCalculator.
run_timeseries() call per simulation, combining heating and DHW into the
SAME hourly step and blending their design temperatures by load share -
a documented simplification (see the comment above that branch in
composer.py). These tests cover that it stays correct, and that
BoilerConfig's priority_mode option (clause 6.10, added to address part of
that simplification) flows through the composer's generation config dict
unmodified - no composer.py-specific wiring is needed for it.
"""

import pybuildingenergy as pybui


def _run(priority_mode: str | None = None):
    catalog = pybui.build_catalog()
    if priority_mode is not None:
        catalog["generation"]["boiler_15316_4_1"]["config"]["priority_mode"] = priority_mode
    scenario = {"generation": "boiler_15316_4_1", "weather_source": "pvgis", "options": {}}
    config = pybui.compose_config(scenario, catalog=catalog)
    return pybui.simulate(config)


def test_composer_boiler_default_uses_proportional_split_and_says_so():
    res = _run()
    notes = [n for n in res.notes if "fuel attributed" in n]
    assert len(notes) == 1
    assert "split proportionally" in notes[0]
    assert "dhw_priority" not in notes[0]


def test_composer_boiler_priority_mode_flows_through_unmodified():
    """Setting priority_mode on the generation config dict (no composer.py
    change needed - gcfg flows straight into BoilerGeneratorCalculator)
    must both run successfully AND be reflected in the reported note."""

    res = _run(priority_mode="dhw_priority")
    notes = [n for n in res.notes if "fuel attributed" in n]
    assert len(notes) == 1
    assert "dhw_priority" in notes[0]
    assert "split proportionally" not in notes[0]

    # Still a complete, sane result - not just a note-text change.
    assert res.summary["final_energy_by_carrier_kWh"]["natural_gas_kWh"] > 0.0
    assert res.summary["Q_unmet_kWh"] >= 0.0
