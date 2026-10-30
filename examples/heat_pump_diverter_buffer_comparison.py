"""Heat pump + diverter valve + DHW tank, with and without a return buffer.

Demonstrates two common residential heat-pump hydraulic schemes with
pybuildingenergy's composer (EN 15316-4-2 generation via
HeatPumpSystemCalculator, EN 15316-2 floor-heating emission, EN 15316-3
distribution), on the built-in 100 m2 single-zone demo dwelling:

1. "Base, no buffer": HP + 3-way diverter valve + DHW tank + direct floor
   radiant. The simplest, most efficient scheme - HP output goes straight
   to whichever service (heating or DHW) the diverter is switched to, one
   shared circuit. Needs a minimum guaranteed water volume in the "always
   open" zone(s) so the HP always has somewhere to dump its output; fits a
   small, single-zone house well.
   Modelled as combined_operation="alternative" (the diverter: HP output
   serves ONE service per time step, not both) with dhw_priority=True, and
   dhw_storage_loss_kWh_per_day for the DHW tank's own stand-by loss. No
   heating_storage_loss_kWh_per_day (no buffer on the heating side).

2. "Return buffer in series": same scheme, but a buffer tank is added in
   series on the heating return. It still goes through the SAME diverter
   valve and single circuit (combined_operation stays "alternative") - the
   buffer's only job is to guarantee enough water volume for defrost
   cycles and to reduce ON/OFF cycling, at the cost of its own stand-by
   loss. Modelled by adding heating_storage_loss_kWh_per_day (plus the
   storage temperature-correction inputs) on TOP of scenario 1's config.

Run: PYTHONPATH=src python examples/heat_pump_diverter_buffer_comparison.py
"""

from __future__ import annotations

import copy

import pybuildingenergy as pybui

# --------------------------------------------------------------------------
# 1. Start from the catalog's realistic air/water HP + DHW config, adapted
#    to a direct low-temperature floor radiant circuit (no backup, so the
#    comparison below isolates the diverter/buffer effect cleanly - a real
#    installation would normally still carry a small electrical backup).
# --------------------------------------------------------------------------
catalog = pybui.build_catalog()
base_hp_config = copy.deepcopy(
    catalog["generation"]["air_to_water_heating_dhw_parallel_backup"]["config"]
)
base_hp_config.update({
    "backup_mode": "none",
    "heating_sink_temp_at_design_C": 35.0,  # direct floor radiant, not a mixed/high-temp circuit
    "heating_sink_temp_at_cutoff_C": 25.0,
})
base_hp_config.pop("backup_efficiency", None)

# --------------------------------------------------------------------------
# 2. Scenario 1: base, no buffer (combined_operation="alternative" is
#    already the default - the diverter valve - but spelled out here).
# --------------------------------------------------------------------------
no_buffer_config = copy.deepcopy(base_hp_config)
no_buffer_config.update({
    "combined_operation": "alternative",  # single circuit, diverter valve
    "dhw_priority": True,
})

# --------------------------------------------------------------------------
# 3. Scenario 2: same scheme + a return buffer tank on the heating branch
#    only (the DHW tank's own loss is unchanged - it is the SAME bollitore
#    in both scenarios). Values reused from the catalog's own
#    "independent_services_with_buffer" example.
# --------------------------------------------------------------------------
with_buffer_config = copy.deepcopy(no_buffer_config)
with_buffer_config.update({
    "heating_storage_loss_kWh_per_day": 1.2,
    "storage_ambient_temperature_C": 18.0,
    "storage_location_b_factor": 0.5,
    "storage_test_deltaT_K": 45.0,
})

catalog["generation"]["hp_diverter_no_buffer"] = {
    "kind": "heat_pump",
    "description": "PdC + deviatrice + bollitore ACS + radiante diretto, senza puffer.",
    "config": no_buffer_config,
}
catalog["generation"]["hp_diverter_with_buffer"] = {
    "kind": "heat_pump",
    "description": "Come sopra, con puffer in serie sul ritorno riscaldamento.",
    "config": with_buffer_config,
}

# --------------------------------------------------------------------------
# 4. Same building/emission/distribution/weather for both runs.
# --------------------------------------------------------------------------
shared_scenario = {
    "emission": "floor_heating_simplified",
    "distribution": "analytical_detailed_dict",
    "weather_source": "pvgis",
    "options": {},
}

for generation_name in ("hp_diverter_no_buffer", "hp_diverter_with_buffer"):
    config = pybui.compose_config({**shared_scenario, "generation": generation_name}, catalog=catalog)
    result = pybui.simulate(config)
    print(result.report())
    print()
