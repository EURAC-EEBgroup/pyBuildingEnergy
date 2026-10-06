"""Compose a single-zone building with selectable HVAC systems into ONE combined
JSON, then run the pybuildingenergy simulation from that single file.

This is the library-side counterpart of ``examples/compose_single_zone_systems.py``
(which is now a thin CLI wrapper around this module). Any project that does
``pip install pybuildingenergy`` gets this natively - no ``sys.path`` hacks, no
copying example scripts around:

    import pybuildingenergy as pybui
    config = pybui.load_config("example_combined_config.json")
    result = pybui.simulate(config)
    print(result.report())

The catalog of selectable options (used by ``compose_config()`` to build a NEW
combined config from named options) is read from reference JSON files shipped
as package data under ``pybuildingenergy/data/generation_catalog/``:

    emission     <- emission_system_configuration_reference.json
    distribution <- distribution_system_configuration_reference.json
    generation   <- heat_pump_configuration_reference.json (+ a real EN 15316-4-1
                    BoilerGeneratorCalculator config, plus biomass/district/CHP)
    pv           <- pv_system_configuration_reference.json
    ahu          <- ahu_system_configuration_reference.json

Step 1 (compose): building + emission + distribution + generation are merged into
ONE JSON document with top-level keys:

    {
      "building": { ... single-zone BUI, as consumed by ISO 52016 ... },
      "hvac_system": { ... ONE flat dict, the exact shape pybuildingenergy's
                        HeatingSystemCalculator itself consumes (see
                        source/example_inputs.py::get_example_hvac_input) ... },
      "dhw": {...}, "pv": {...} or null
    }

``hvac_system`` merges the emission keys (emitter_type, nominal_power,
selected_emm_cont_circuit, emission_15316_2_config, ...) and the distribution
keys (distribution_calculation_mode, distribution_15316_3_config) at the SAME
flat level HeatingSystemCalculator expects, since that class computes emission +
distribution for SPACE HEATING in one call. Generation is always run
separately, for every kind including the EN 15316-4-1 boiler: the full
generator configuration is namespaced under "external_generation":
{"kind": ..., "config": {...}} so the document stays self-describing.

Why the boiler is NOT run through HeatingSystemCalculator's own
'boiler_15316_4_1' branch: that branch (and compute_step() in general) only
ever receives Q_H - there is no channel for Q_W (DHW), so it cannot couple the
boiler's efficiency curve to DHW load. BoilerGeneratorCalculator - the class
that actually implements EN 15316-4-1 - natively accepts Q_H_gen_out_kWh AND
Q_W_gen_out_kWh in the SAME call: it sizes the loss/efficiency curve on the
COMBINED load and splits the input energy back proportionally to each service
(E_H_gen_in / E_W_gen_in), mirroring the standard's own separate expenditure
factors epsilon_H,gen / epsilon_W,gen (Table 4). simulate() calls it that way.

Step 2 (simulate): ISO 52016 runs the building; HeatingSystemCalculator runs
emission + distribution for space heating; EmissionSystemCalculator /
DistributionSystemCalculator (reusing the SAME config's cooling/dhw blocks)
resolve cooling and DHW; then the chosen generator (from "external_generation")
is run ONCE with heating + DHW (+ cooling, for reversible generators) together.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .biomass_15316_4 import BiomassBoilerSystemCalculator
from .check_input import sanitize_and_validate_BUI
from .cogeneration_15316_4_4 import CogenerationSystemCalculator
from .distribution_15316_3 import DistributionSystemCalculator
from .district_15316_4_5 import DistrictEnergySystemCalculator
from .emission_15316_2 import EmissionSystemCalculator
from .generation_15316_4_1 import BoilerGeneratorCalculator
from .heat_pump_15316_4_2 import HeatPumpSystemCalculator
from .iso_15316_1 import HeatingSystemCalculator
from .primary_energy_52000_1 import PrimaryEnergyAccountingCalculator
from .renewables_15316_4_3_4_6 import RenewableEnergySystemCalculator
from .utils import ISO52016

CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "generation_catalog"

DEFAULT_SCENARIO: dict[str, Any] = {
    "building": None,                       # None = built-in demo; or path to a single-zone BUI JSON
    "weather_source": "pvgis",              # "pvgis" needs no local file; "epw" needs weather_file
    "weather_file": None,
    "emission": "fan_coil_en15316_2_heating_cooling",
    "distribution": "analytical_detailed_dict",
    "generation": "boiler_15316_4_1",
    "pv": None,                             # e.g. "pv_with_battery_dispatch"
    "ahu": None,                            # e.g. "ahu_fixed_supply_basic"
    "dhw": {"enabled": True, "daily_volume_l": 150.0, "draw_temperature_C": 40.0, "cold_water_temperature_C": 12.0},
    "options": {
        "autosize": True,                   # set nominal powers / capacities from the peak load
        "sizing_margin": 1.2,
        "ahu_coils_to_generation": True,    # AHU coil energy is added to the generation demand
        "fallback_cooling_eer": None,       # ideal chiller for cooling the chosen generator cannot serve
        "baseline_electricity_kWh_per_year": 0.0,   # other electric loads (flat profile) seen by the PV
        "primary_energy_factors": None,     # override of PrimaryEnergyAccountingCalculator factors
        "ahu_flow_ach": 0.5,                # AHU supply flow = ach x zone volume (None keeps supply_flow_m3_h of the JSON option)
        "ahu_infiltration_ach": None,       # infiltration kept alongside the AHU (None = building airflow_rates.infiltration_rate)
        "pe_balance": "monthly",            # 'annual' | 'monthly' | 'hourly': period over which export credit offsets deliveries
        "hourly_cache_dir": None,           # reuse the ISO 52016 result when only the systems change
    },
}

# Real EN 15316-4-1 BoilerGeneratorCalculator fields (same shape HeatingSystemCalculator's
# own 'boiler_generation_config' expects; values from tests/test_general.py's
# test_boiler_generator_calculator_table_driven).
BOILER_15316_4_1_CONFIG: dict[str, Any] = {
    "boiler_type": "condensing", "fuel_type": "natural_gas", "rated_power_kW": 24.0,
    "intermediate_load_fraction": 0.30, "eta_Pn_test_pct": 98.0, "eta_Pint_test_pct": 106.0,
    "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 40.0, "f_corr_pct_per_K": 0.04,
    "P_gen_ls_P0_W": 100.0, "P_aux_on_W": 80.0, "P_aux_off_W": 5.0, "f_jacket": 0.40,
    "f_location": 1.0, "f_aux_recoverable": 0.75, "dew_point_C": 55.0, "condensing_gain_pct": 11.0,
    "efficiency_table": {"condensing": {"eta_Pn_test_pct": 98.0, "eta_Pint_test_pct": 106.0, "theta_test_Pn_C": 60.0, "theta_test_Pint_C": 40.0}},
    "loss_table": {"condensing": {"P_gen_ls_P0_W": 100.0}},
    "boiler_location": "inside_heated",
}

# Generators HeatingSystemCalculator has NO internal path for: run externally,
# fed with the heating/cooling/DHW demand produced by the shared emission/distribution config.
EXTERNAL_GENERATORS: dict[str, dict[str, Any]] = {
    "biomass_boiler": {"kind": "biomass_boiler", "description": "Biomass boiler EN 15316-4 wrapper, sized to the peak.", "config": {"nominal_power_kW": 20.0}},
    "district_heating": {"kind": "district", "description": "District heating substation (with cooling branch).", "config": {"cooling_enabled": True, "heating_substation_efficiency": 0.97}},
    "cogeneration": {"kind": "cogeneration", "description": "CHP EN 15316-4-4 (thermal 0.56, electric 0.30), sized at 40% of the peak.", "config": {"nominal_thermal_power_kW": 5.0, "thermal_efficiency": 0.56, "electrical_efficiency": 0.30}},
}

# Defaults completing a HeatingSystemCalculator dict when a catalog entry does not specify them
# (generator-side keys the constructor always reads, and a fallback emission shell for the
# emission-only entry that ships as a standalone EmissionSystemCalculator config).
HVAC_DEFAULTS: dict[str, Any] = {
    "nominal_power": 8.0, "emission_efficiency": 90, "mixing_valve": True, "mixing_valve_delta": 2,
    "selected_emm_cont_circuit": 0, "flow_temp_control_type": "Type 2 - Based on outdoor temperature",
    "full_load_power": 24.0, "generator_circuit": "independent",
    "gen_flow_temp_control_type": "Type A - Based on outdoor temperature",
    "gen_outdoor_temp_data": {"θext_min_gen": -7.0, "θext_max_gen": 15.0, "θflw_gen_max": 60.0, "θflw_gen_min": 35.0},
    "efficiency_model": "simple", "calc_when_QH_positive_only": False, "off_compute_mode": "full",
}


# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------

def _read(name: str) -> dict:
    return json.loads((CATALOG_DIR / name).read_text(encoding="utf-8"))


def _resolve_refs(obj: Any, maps: dict[str, Any]) -> Any:
    if isinstance(obj, str) and obj.startswith("@"):
        return copy.deepcopy(maps[obj[1:]])
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, maps) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, maps) for v in obj]
    return obj


def build_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    """Read the reference JSON files and normalise every usable example into a spec.

    Emission entries keep their RAW flat 'hvac_keys' (emitter_type, nominal_power,
    selected_emm_cont_circuit, emission_15316_2_config, ...) so they can be merged
    verbatim into the single combined hvac_system dict.
    """

    catalog: dict[str, dict[str, dict[str, Any]]] = {
        "emission": {}, "distribution": {}, "generation": {}, "pv": {}, "ahu": {},
    }

    # emission -----------------------------------------------------------------------
    for name, e in _read("emission_system_configuration_reference.json")["example_configurations"].items():
        if name.startswith("_"):
            continue
        if "emission_system_config" in e:
            hvac_keys = {
                "emitter_type": "Fan coil", "nominal_power": HVAC_DEFAULTS["nominal_power"],
                "emission_calculation_mode": "en15316-2",
                "emission_15316_2_config": copy.deepcopy(e["emission_system_config"]),
            }
            catalog["emission"][name] = {"mode": "en15316-2", "description": e["description"], "hvac_keys": hvac_keys}
            continue
        keys = e.get("hvac_emission_keys")
        if not keys:
            continue
        mode = "en15316-2" if str(keys.get("emission_calculation_mode", "simplified")).lower() == "en15316-2" else "simplified"
        catalog["emission"][name] = {"mode": mode, "description": e["description"], "hvac_keys": copy.deepcopy(keys)}

    # distribution -------------------------------------------------------------------
    for name, e in _read("distribution_system_configuration_reference.json")["example_configurations"].items():
        cfg = e.get("distribution_system_config") or (e.get("hvac_distribution_keys", {}) or {}).get("distribution_15316_3_config")
        if cfg:
            catalog["distribution"][name] = {"description": e["description"], "config": copy.deepcopy(cfg)}
    catalog["distribution"]["ideal_no_losses"] = {"description": "Ideal distribution: no losses, no pump energy.", "config": {}}

    # generation ---------------------------------------------------------------------
    catalog["generation"]["boiler_15316_4_1"] = {
        "kind": "boiler_15316_4_1",
        "description": "Condensing gas boiler EN 15316-4-1, computed INSIDE HeatingSystemCalculator (no external call).",
        "config": copy.deepcopy(BOILER_15316_4_1_CONFIG),
    }
    hp = _read("heat_pump_configuration_reference.json")["example_configurations"]
    maps = hp["_shared_maps"]
    for name, e in hp.items():
        if name.startswith("_") or "config" not in e:
            continue
        catalog["generation"][name] = {"kind": "heat_pump", "description": e["description"], "config": _resolve_refs(e["config"], maps)}
    for name, spec in EXTERNAL_GENERATORS.items():
        catalog["generation"][name] = copy.deepcopy(spec)

    # pv -----------------------------------------------------------------------------
    for name, e in _read("pv_system_configuration_reference.json")["example_configurations"].items():
        if "pv" in e.get("config", {}):
            catalog["pv"][name] = {"description": e["description"], "config": copy.deepcopy(e["config"])}

    # ahu ----------------------------------------------------------------------------
    for name, e in _read("ahu_system_configuration_reference.json")["example_configurations"].items():
        comp = e.get("component", {})
        if comp.get("ventilation_type") == "mechanical_supply":
            catalog["ahu"][name] = {"description": e["description"], "component": copy.deepcopy(comp)}
    return catalog


def print_catalog(catalog: dict) -> None:
    for category, items in catalog.items():
        print(f"\n[{category}]")
        for name, spec in items.items():
            tag = spec.get("kind") or spec.get("mode") or ""
            print(f"  {name:52s} {('(' + tag + ') ') if tag else ''}{spec['description'][:80]}")


def _pick(catalog: dict, category: str, name: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(catalog[category][name])
    except KeyError:
        raise KeyError(f"Unknown {category} option {name!r}. Available: {', '.join(catalog[category])}") from None


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------

def demo_building() -> dict[str, Any]:
    """100 m2 single-zone dwelling near Milan with a moderately insulated envelope."""

    def opaque(name, area, u, az, tilt, cap, svf):
        return {"name": name, "type": "opaque", "area": area, "sky_view_factor": svf, "u_value": u, "solar_absorptance": 0.5,
                "thermal_capacity": cap, "orientation": {"azimuth": az, "tilt": tilt}, "name_adj_zone": None}

    def window(name, area, az, h, w):
        return {"name": name, "type": "transparent", "area": area, "sky_view_factor": 0.5, "u_value": 1.4, "g_value": 0.45, "height": h, "width": w,
                "parapet": 0.9, "orientation": {"azimuth": az, "tilt": 90}, "shading": False, "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5, "overhang_properties": {"width_of_horizontal_overhangs": 0.8}, "name_adj_zone": None}

    day = [0.0] * 5 + [1.0] * 18 + [0.0]
    return {
        "building": {"name": "demo_dwelling", "azimuth_relative_to_true_north": 0.0, "latitude": 45.47, "longitude": 9.19,
                     "exposed_perimeter": 40, "height": 3.0, "wall_thickness": 0.35, "n_floors": 1, "building_type_class": "Residential_apartment",
                     "adj_zones_present": False, "number_adj_zone": 0, "net_floor_area": 100, "construction_class": "class_i"},
        "adjacent_zones": [],
        "building_surface": [
            opaque("Roof", 110, 0.25, 0, 0, 741500.0, 1.0),
            opaque("Wall N", 28, 0.30, 0, 90, 1416240.0, 0.5),
            opaque("Wall S", 22, 0.30, 180, 90, 1416240.0, 0.5),
            opaque("Wall E", 27, 0.30, 90, 90, 1416240.0, 0.5),
            opaque("Wall W", 27, 0.30, 270, 90, 1416240.0, 0.5),
            opaque("Slab", 100, 0.30, 0, 0, 405801.0, 0.0),
            window("Window N", 2, 0, 1.2, 1.7), window("Window S", 6, 180, 1.4, 4.3),
            window("Window E", 2, 90, 1.4, 1.4), window("Window W", 2, 270, 1.4, 1.4),
        ],
        "units": {"area": "m2", "u_value": "W/m2K"},
        "building_parameters": {
            "temperature_setpoints": {"heating_setpoint": 20.0, "heating_setback": 17.0, "cooling_setpoint": 26.0, "cooling_setback": 30.0},
            "system_capacities": {"heating_capacity": 1e7, "cooling_capacity": 1e7},
            "airflow_rates": {"infiltration_rate": 0.5},
            "internal_gains": [
                {"name": "occupants", "full_load": 3.0, "weekday": [0.9] * 6 + [0.5] * 3 + [0.1] * 8 + [0.6] * 3 + [0.9] * 4, "weekend": [1.0] * 8 + [0.8] * 14 + [1.0] * 2},
                {"name": "appliances", "full_load": 1.5, "weekday": [0.4] * 24, "weekend": [0.5] * 24},
            ],
            "construction": {"wall_thickness": 0.35, "thermal_bridge_heat_W_K": 8.0},
            "climate_parameters": {"coldest_month": 1},
            "heating_profile": {"weekday": day, "weekend": day},
            "cooling_profile": {"weekday": day, "weekend": day},
            "ventilation_profile": {"weekday": day, "weekend": day},
        },
    }


def load_building(path: str | None) -> dict[str, Any]:
    bui = demo_building() if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    if "zones" in bui:
        raise ValueError("This composer handles single-zone buildings only (the BUI has a 'zones' key).")
    return bui


def attach_ahu(bui: dict[str, Any], component: dict[str, Any], infiltration_ach: float | None = None) -> dict[str, Any]:
    """Add the AHU to the zone as a ventilation component (infiltration is kept as a constant_ach component)."""

    b = bui["building"]
    b.setdefault("zone_volume_m3", float(b["net_floor_area"]) * float(b.get("height", 3.0)))
    vent = bui["building_parameters"].setdefault("ventilation", {})
    ach = float(infiltration_ach if infiltration_ach is not None else bui["building_parameters"].get("airflow_rates", {}).get("infiltration_rate", 0.3))
    vent["components"] = [
        {"name": "infiltration", "ventilation_type": "constant_ach", "air_changes_per_hour": ach, "source_temperature": "outdoor"},
        {**component, "name": "ahu"},
    ]
    return bui


# --------------------------------------------------------------------------------------
# Step 1: compose ONE combined JSON (building + hvac_system + dhw + pv)
# --------------------------------------------------------------------------------------

def build_hvac_system(catalog: dict, emission_name: str, distribution_name: str, generation_name: str) -> dict[str, Any]:
    """Merge the chosen emission + distribution + generation options into ONE flat
    dict, in the exact shape pybuildingenergy.HeatingSystemCalculator consumes
    (see source/example_inputs.py::get_example_hvac_input)."""

    em = _pick(catalog, "emission", emission_name)
    di = _pick(catalog, "distribution", distribution_name)
    ge = _pick(catalog, "generation", generation_name)

    hvac: dict[str, Any] = copy.deepcopy(em["hvac_keys"])
    for k in ("nominal_power", "emission_efficiency", "mixing_valve", "mixing_valve_delta", "selected_emm_cont_circuit", "flow_temp_control_type"):
        hvac.setdefault(k, HVAC_DEFAULTS[k])

    hvac["distribution_calculation_mode"] = "analytical" if di["config"] else "simplified"
    if di["config"]:
        hvac["distribution_15316_3_config"] = copy.deepcopy(di["config"])
    hvac.setdefault("heat_losses_recovered", True)
    hvac.setdefault("distribution_loss_recovery", 90)
    hvac.setdefault("simplified_approach", 80)
    hvac.setdefault("distribution_aux_recovery", 80)
    hvac.setdefault("distribution_aux_power", 30)
    hvac.setdefault("distribution_loss_coeff", 48)
    hvac.setdefault("distribution_operation_time", 1)

    for k in ("full_load_power", "generator_circuit", "gen_flow_temp_control_type", "gen_outdoor_temp_data", "efficiency_model", "calc_when_QH_positive_only", "off_compute_mode"):
        hvac.setdefault(k, HVAC_DEFAULTS[k])

    # HeatingSystemCalculator.compute_step() only ever receives Q_H: it has no channel for
    # Q_W (DHW), so its internal 'boiler_15316_4_1' branch cannot couple the boiler's
    # efficiency curve to DHW load. Every generator - including the boiler - is therefore
    # run EXTERNALLY by simulate(), which for the boiler calls BoilerGeneratorCalculator
    # once per hour with BOTH Q_H_gen_out_kWh and Q_W_gen_out_kWh: the calculator applies
    # its loss/efficiency model to the COMBINED load and splits the input energy back
    # proportionally to each service (E_H_gen_in / E_W_gen_in) - the same idea as the
    # standard's separate expenditure factors epsilon_H,gen / epsilon_W,gen (EN 15316-4-1,
    # Table 4). generation_calculation_mode stays 'legacy' purely as a harmless placeholder;
    # its own (heating-only) output is never read.
    kind = ge["kind"]
    hvac["generation_calculation_mode"] = "legacy"
    hvac["external_generation"] = {"kind": kind, "name": generation_name, "config": copy.deepcopy(ge["config"])}
    return hvac


def compose_config(scenario: dict[str, Any] | None = None, catalog: dict | None = None) -> dict[str, Any]:
    """Build the SINGLE combined JSON document: {"building", "hvac_system", "dhw", "pv", ...}.

    This is the artifact meant to travel between repos: one file that fully
    describes the building, the emission system, the distribution system and
    the generator, ready to be handed to `simulate()`.
    """

    sc = copy.deepcopy(DEFAULT_SCENARIO)
    for k, v in (scenario or {}).items():
        if k == "options":
            sc["options"].update(v or {})
        else:
            sc[k] = v
    opt = sc["options"]
    catalog = catalog or build_catalog()
    notes: list[str] = []

    bui = load_building(sc["building"])
    ahu_spec = _pick(catalog, "ahu", sc["ahu"]) if sc["ahu"] else None
    if ahu_spec:
        comp = ahu_spec["component"]
        volume = float(bui["building"].get("zone_volume_m3") or float(bui["building"]["net_floor_area"]) * float(bui["building"].get("height", 3.0)))
        if opt["ahu_flow_ach"]:
            comp["supply_flow_m3_h"] = round(float(opt["ahu_flow_ach"]) * volume, 1)
            comp.pop("extract_flow_m3_h", None)
            notes.append(f"AHU supply flow set to {opt['ahu_flow_ach']} ach x {volume:.0f} m3 = {comp['supply_flow_m3_h']:.0f} m3/h (options.ahu_flow_ach).")
        attach_ahu(bui, comp, opt["ahu_infiltration_ach"])

    hvac_system = build_hvac_system(catalog, sc["emission"], sc["distribution"], sc["generation"])

    pv_config = _pick(catalog, "pv", sc["pv"])["config"] if sc["pv"] else None

    config = {
        "$schema": "pybuildingenergy-composed-config-v1",
        "selection": {"emission": sc["emission"], "distribution": sc["distribution"], "generation": sc["generation"], "pv": sc["pv"], "ahu": sc["ahu"]},
        "weather": {"source": sc["weather_source"], "file": sc["weather_file"]},
        "building": bui,
        "hvac_system": hvac_system,
        "dhw": sc["dhw"],
        "pv": pv_config,
        "options": opt,
        "compose_notes": notes,
    }
    return config


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, default=float), encoding="utf-8")
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# General component-document schema (building + optional systems)
# --------------------------------------------------------------------------------------

def _component(systems: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a normalized optional component declaration."""
    value = systems.get(name, {})
    if value is None:
        return {"enabled": False}
    if isinstance(value, bool):
        return {"enabled": value}
    if not isinstance(value, dict):
        raise ValueError(f"systems.{name} must be an object, true, false or null.")
    return {"enabled": bool(value.get("enabled", True)), **value}


def normalize_system_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the public modular JSON schema to the established simulator schema.

    The public schema keeps independently selectable subsystems under
    ``systems``.  Only enabled components are passed downstream.  The returned
    document is deliberately an internal compatibility representation; callers
    should retain the modular document as their source configuration.
    """
    if "systems" not in config:
        return copy.deepcopy(config)
    if "building" not in config or "weather" not in config:
        raise ValueError("A system configuration requires top-level 'building' and 'weather'.")

    systems = config["systems"]
    if not isinstance(systems, dict):
        raise ValueError("systems must be an object.")
    emission = _component(systems, "emission")
    control = _component(systems, "control")
    distribution = _component(systems, "distribution")
    generation = _component(systems, "generation")
    dhw = _component(systems, "dhw")
    pv = _component(systems, "pv")
    ahu = _component(systems, "ahu")

    if not emission["enabled"]:
        raise ValueError("systems.emission must be enabled: it is the interface from building load to HVAC.")
    emission_cfg = copy.deepcopy(emission.get("config") or emission.get("emission_system_config"))
    if not isinstance(emission_cfg, dict):
        raise ValueError("systems.emission.config must be an EN 15316-2 emission configuration.")

    # An emission-only document is a valid simulation in its own right.
    active_beyond_emission = any(
        component["enabled"] for component in (distribution, generation, dhw, pv, ahu)
    )
    if not active_beyond_emission:
        return {
            "$schema": "pybuildingenergy-emission-only-v2",
            "building": copy.deepcopy(config["building"]),
            "weather": copy.deepcopy(config["weather"]),
            "emission_system_config": emission_cfg,
        }
    if not generation["enabled"]:
        raise ValueError(
            "A configuration with distribution, DHW, PV or AHU requires systems.generation.enabled=true. "
            "Use only systems.emission for an emission-only simulation."
        )

    hvac = copy.deepcopy(HVAC_DEFAULTS)
    hvac.update(copy.deepcopy(emission.get("hvac_keys", {})))
    if control["enabled"]:
        control_cfg = copy.deepcopy(control.get("config", {}))
        if not isinstance(control_cfg, dict):
            raise ValueError("systems.control.config must be an object when enabled.")
        hvac.update(control_cfg)
    hvac.update({
        "emitter_type": emission.get("emitter_type", hvac.get("emitter_type", "Fan coil")),
        "nominal_power": float(emission.get(
            "nominal_power_kW", emission_cfg.get("heating", {}).get("nominal_power_kW", hvac["nominal_power"])
        )),
        "emission_calculation_mode": "en15316-2",
        "emission_15316_2_config": emission_cfg,
    })
    if distribution["enabled"]:
        distribution_cfg = copy.deepcopy(distribution.get("config"))
        if not isinstance(distribution_cfg, dict):
            raise ValueError("systems.distribution.config must be a distribution configuration when enabled.")
        hvac["distribution_calculation_mode"] = "analytical"
        hvac["distribution_15316_3_config"] = distribution_cfg
    else:
        # Explicit ideal distribution: no thermal or auxiliary contribution.
        hvac.update({"distribution_calculation_mode": "simplified", "distribution_loss_coeff": 0.0,
                     "distribution_aux_power": 0.0})

    generation_cfg = copy.deepcopy(generation.get("config"))
    generation_kind = generation.get("kind")
    if not generation_kind or not isinstance(generation_cfg, dict):
        raise ValueError("systems.generation requires both 'kind' and object 'config' when enabled.")
    hvac["external_generation"] = {
        "kind": generation_kind,
        "name": generation.get("name", generation_kind),
        "config": generation_cfg,
    }
    hvac["generation_calculation_mode"] = "legacy"

    bui = copy.deepcopy(config["building"])
    if ahu["enabled"]:
        ahu_cfg = copy.deepcopy(ahu.get("config") or ahu.get("component"))
        if not isinstance(ahu_cfg, dict):
            raise ValueError("systems.ahu.config must be an AHU component when enabled.")
        attach_ahu(bui, ahu_cfg, ahu.get("infiltration_ach"))

    options = copy.deepcopy(DEFAULT_SCENARIO["options"])
    options.update(copy.deepcopy(config.get("options", {})))
    return {
        "$schema": "pybuildingenergy-composed-config-v1",
        "weather": copy.deepcopy(config["weather"]), "building": bui,
        "hvac_system": hvac,
        "dhw": copy.deepcopy(dhw.get("config", {})) if dhw["enabled"] else {"enabled": False},
        "pv": copy.deepcopy(pv.get("config")) if pv["enabled"] else None,
        "options": options,
        "compose_notes": ["Normalized from pybuildingenergy-system-config-v2."],
    }


def simulate_config(config: dict[str, Any], hourly: pd.DataFrame | None = None):
    """Simulate a modular ``systems`` document or a legacy configuration."""
    normalized = normalize_system_config(config)
    if "hvac_system" not in normalized:
        if hourly is not None:
            raise ValueError("hourly override is not supported by the emission-only simulation path.")
        method = str((normalized.get("options") or {}).get("emission_inc_method", "recalculate"))
        return simulate_emission_only(normalized, emission_inc=method)
    return simulate(normalized, hourly=hourly)


# --------------------------------------------------------------------------------------
# Step 2: simulate FROM the combined JSON
# --------------------------------------------------------------------------------------

def run_zone(bui: dict[str, Any], weather_source: str, weather_file: str | None, cache_dir: str | None) -> pd.DataFrame:
    if weather_source == "epw":
        if not weather_file:
            raise ValueError("weather.source is 'epw' but weather.file is empty. Set an absolute path to an .epw "
                              "file reachable from where simulate() runs, or switch weather.source to 'pvgis'.")
        if not Path(weather_file).exists():
            raise FileNotFoundError(
                f"EPW weather file not found: {weather_file!r}. This path travels with the config JSON as-is - "
                "if you moved the JSON to another machine/repo, either copy the .epw file to that path, edit "
                "weather.file to a path valid there, or set weather.source to 'pvgis' (fetched live, no local file)."
            )

    key = hashlib.sha1(json.dumps([bui, weather_source, weather_file], sort_keys=True, default=str).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"zone_{key}.pkl" if cache_dir else None
    if cache and cache.exists():
        return pd.read_pickle(cache)

    checked, issues = sanitize_and_validate_BUI(copy.deepcopy(bui), fix=True)
    errors = [i for i in issues if i["level"] == "ERROR"]
    if errors:
        raise ValueError(f"Building validation errors: {errors}")
    kwargs: dict[str, Any] = {"weather_source": weather_source}
    if weather_source == "epw":
        kwargs["path_weather_file"] = weather_file
    out = ISO52016.Temperature_and_Energy_needs_calculation(checked, **kwargs)
    hourly = out[0]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_pickle(cache)
    return hourly


def dhw_demand_kwh(index: pd.DatetimeIndex, dhw: dict[str, Any] | None) -> pd.Series:
    """Simplified DHW demand: V * 0.001163 * dT per day, spread on a fixed hourly profile (not EN 12831-3)."""

    if not dhw or not dhw.get("enabled", True):
        return pd.Series(0.0, index=index)
    e_day = float(dhw["daily_volume_l"]) * 0.001163 * (float(dhw["draw_temperature_C"]) - float(dhw["cold_water_temperature_C"]))
    w = np.zeros(24)
    w[[6, 7]] = 0.15
    w[[12, 13]] = 0.075
    w[[18, 19, 20, 21]] = 0.1
    w[[22]] = 0.05
    w = w / w.sum()
    return pd.Series(e_day * w[index.hour], index=index)


def _series(hourly: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(hourly[col], errors="coerce").fillna(default) if col in hourly.columns else pd.Series(default, index=hourly.index)


def _gen_outdoor_df(spec: Any) -> pd.DataFrame:
    if isinstance(spec, pd.DataFrame):
        return spec
    return pd.DataFrame({k: [v] for k, v in spec.items()}, index=["Generator curve"])


def _scale_map_to_peak(map_rows: list[dict] | pd.DataFrame, cfg: dict, service: str, peak_kw: float, margin: float) -> tuple[Any, float]:
    df = pd.DataFrame(map_rows).copy()
    if service == "heating":
        src_row = df["source_temperature_C"].min()
        target_sink = float(cfg.get("heating_sink_temp_at_design_C", 45.0))
    else:
        src_row = df["source_temperature_C"].max()
        target_sink = float(cfg.get("cooling_sink_temperature_C", 7.0))
    rows = df[df["source_temperature_C"] == src_row]
    ref = rows.iloc[(rows["sink_temperature_C"] - target_sink).abs().argsort().iloc[0]]
    design_cap = float(ref["capacity_kW"])
    factor = margin * peak_kw / design_cap if design_cap > 0 else 1.0
    df["capacity_kW"] = df["capacity_kW"] * factor
    return df, factor


@dataclass
class CompositionResult:
    config: dict[str, Any]
    hourly: pd.DataFrame
    summary: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        s = self.summary
        area = s["floor_area_m2"]
        sel = self.config["selection"]
        lines = [
            "=" * 78,
            f"emission: {sel['emission']} | distribution: {sel['distribution']}",
            f"generation: {sel['generation']} | pv: {sel['pv']} | ahu: {sel['ahu']}",
            "=" * 78,
            f"{'Zone need (ISO 52016)':34s} H {s['Q_H_need_kWh']:9.0f}  C {s['Q_C_need_kWh']:8.0f}  DHW {s['Q_W_need_kWh']:7.0f} kWh   ({(s['Q_H_need_kWh'] + s['Q_C_need_kWh']) / area:.1f} kWh/m2a H+C)",
            f"{'Emission input':34s} H {s['Q_H_em_in_kWh']:9.0f}  C {s['Q_C_em_in_kWh']:8.0f}                    aux {s['W_em_aux_kWh']:.1f} kWh_el",
            f"{'Distribution input':34s} H {s['Q_H_dis_in_kWh']:9.0f}  C {s['Q_C_dis_in_kWh']:8.0f}  DHW {s['Q_W_dis_in_kWh']:7.0f} kWh   pumps {s['W_dis_aux_kWh']:.1f} kWh_el",
        ]
        if s.get("ahu_fan_kWh") is not None:
            lines.append(f"{'AHU (system side)':34s} heating coil {s['ahu_coil_heat_kWh']:.0f}  cooling coil {s['ahu_coil_cool_kWh']:.0f} kWh   fans {s['ahu_fan_kWh']:.0f} kWh_el")
        lines += [
            f"{'Generation demand':34s} H {s['Q_H_gen_demand_kWh']:9.0f}  C {s['Q_C_gen_demand_kWh']:8.0f}  DHW {s['Q_W_gen_demand_kWh']:7.0f} kWh   unmet {s['Q_unmet_kWh']:.0f} kWh",
            "-" * 78,
            "Final energy (before PV):  " + "  ".join(f"{k.replace('_kWh', '')} {v:,.0f}" for k, v in s["final_energy_by_carrier_kWh"].items() if abs(v) > 0.5) + "  kWh",
        ]
        if s.get("E_PV_gen_kWh") is not None:
            lines.append(f"PV: generation {s['E_PV_gen_kWh']:.0f} | self-consumed {s['E_PV_self_consumed_kWh']:.0f} ({100 * s['f_PV_self_consumed']:.0f}%) | exported {s['E_PV_export_kWh']:.0f} kWh")
        lines += [
            f"Grid electricity: import {s['E_grid_import_kWh']:.0f}  export {s['E_export_kWh']:.0f} kWh",
            f"Primary energy ({s['PE_balance']} balance): delivered nren {s['PE_delivered_nonren_kWh']:.0f} - export credit {s['PE_export_credit_nonren_kWh']:.0f}"
            f" -> net nren {s['PE_net_nonren_kWh']:.0f}, total {s['PE_net_total_kWh']:.0f} kWh ({s['PE_net_total_kWh'] / area:.1f} kWh/m2a)",
            f"Overall system efficiency (need / final energy): {s['system_efficiency']:.2f}",
        ]
        if self.notes:
            lines += ["-" * 78, "Notes:"] + [f"  * {n}" for n in self.notes]
        if self.warnings:
            lines += ["-" * 78, "WARNINGS:"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def simulate(config: dict[str, Any], hourly: pd.DataFrame | None = None) -> CompositionResult:
    """Run the pybuildingenergy simulation FROM the single combined JSON produced by compose_config()."""

    if "systems" in config and "hvac_system" not in config:
        return simulate_config(config, hourly=hourly)

    bui = config["building"]
    hvac = copy.deepcopy(config["hvac_system"])
    opt = config["options"]
    warnings: list[str] = []
    notes: list[str] = list(config.get("compose_notes", []))
    margin = float(opt["sizing_margin"])

    if hourly is None:
        hourly = run_zone(bui, config["weather"]["source"], config["weather"]["file"], opt["hourly_cache_dir"])
    idx = hourly.index
    dt = float(pd.Series(idx).diff().dt.total_seconds().median() / 3600.0)
    area = float(bui["building"]["net_floor_area"])
    ahu_active = "ventilation" in bui.get("building_parameters", {}) and any(
        c.get("ventilation_type") == "mechanical_supply" for c in bui["building_parameters"]["ventilation"].get("components", [])
    )

    q_h = _series(hourly, "Q_H").clip(lower=0.0) / 1000.0 * dt          # W -> kWh
    q_c = _series(hourly, "Q_C").clip(lower=0.0) / 1000.0 * dt
    q_w = dhw_demand_kwh(idx, config["dhw"])
    t_ext = _series(hourly, "T_ext")
    t_op = _series(hourly, "T_op", 20.0)
    peak_h = float(q_h.max() / dt)
    peak_c = float(q_c.max() / dt)

    # ---- autosize (mutates the local hvac copy only; the saved config keeps the catalog defaults) ----
    if opt["autosize"]:
        hvac["nominal_power"] = round(peak_h * margin, 3)
        if hvac.get("emission_calculation_mode") == "en15316-2":
            cfg2 = hvac.setdefault("emission_15316_2_config", {})
            cfg2.setdefault("heating", {})["nominal_power_kW"] = hvac["nominal_power"]
            if peak_c > 0.05:
                cfg2.setdefault("cooling", {})["nominal_power_kW"] = round(peak_c * margin, 3)
        if "distribution_15316_3_config" in hvac:
            for svc, pk in (("heating", peak_h), ("cooling", peak_c), ("dhw", float(q_w.max() / dt))):
                block = hvac["distribution_15316_3_config"].get(svc)
                if block is not None and pk > 0.05:
                    block["nominal_power_kW"] = round(pk * margin, 3)
        notes.append(f"autosize: emission/distribution nominal power set to {margin:g} x peak (heating {peak_h:.1f} kW, cooling {peak_c:.1f} kW).")

    # ============================================================================
    # HEATING (emission + distribution ONLY): ONE call to HeatingSystemCalculator.
    # Its own generation branch is never read - see build_hvac_system() for why.
    # ============================================================================
    ext_gen = hvac["external_generation"]
    kind = ext_gen["kind"]

    hsc_input = copy.deepcopy(hvac)
    hsc_input["gen_outdoor_temp_data"] = _gen_outdoor_df(hsc_input["gen_outdoor_temp_data"])
    heat_ts = pd.DataFrame({"Q_H_kWh": q_h, "T_op": t_op, "T_ext": t_ext, "time_step_hours": dt}, index=idx)
    heat_out = HeatingSystemCalculator(hsc_input).run_timeseries(heat_ts)

    q_h_em = _series(heat_out, "QH_em_i_in(kWh)")
    w_em = _series(heat_out, "W_H_em_aux(kWh)")
    q_h_dis = _series(heat_out, "QH_dis_i_in(kWh)")
    w_dis_h = _series(heat_out, "Q_w_dis_i_aux(kWh)")

    # ============================================================================
    # COOLING + DHW: HeatingSystemCalculator is heating-only. Reuse the SAME
    # merged config's "cooling"/"dhw" blocks with the standalone EN 15316-2 /
    # EN 15316-3 calculators (a real gap in the library, not something to hide).
    # ============================================================================
    q_c_em = q_c.copy()
    w_c_em = pd.Series(0.0, index=idx)
    if hvac.get("emission_calculation_mode") == "en15316-2" and peak_c > 0.05:
        em_cfg = hvac.get("emission_15316_2_config", {})
        em_side = EmissionSystemCalculator(em_cfg).run_timeseries(
            pd.DataFrame({"T_ext": t_ext, "T_op": t_op, "Q_H_kWh": 0.0, "Q_C_kWh": q_c, "time_step_hours": dt}, index=idx)
        ).timeseries
        q_c_em = em_side["Q_C_em_in_kWh"]
        w_c_em = em_side["W_C_em_aux_kWh"]

    q_c_dis, q_w_dis, w_dis_cw = q_c_em.copy(), q_w.copy(), pd.Series(0.0, index=idx)
    if "distribution_15316_3_config" in hvac and (peak_c > 0.05 or float(q_w.sum()) > 0.5):
        dcfg = hvac["distribution_15316_3_config"]
        dist_side = DistributionSystemCalculator(dcfg).run_timeseries(
            pd.DataFrame({"T_ext": t_ext, "Q_H_kWh": 0.0, "Q_C_kWh": q_c_em, "Q_W_kWh": q_w, "time_step_hours": dt}, index=idx)
        ).timeseries
        if "cooling" in dcfg:
            q_c_dis = dist_side["Q_C_dis_in_kWh"]
            w_dis_cw = w_dis_cw + dist_side["W_C_dis_aux_kWh"].abs()
        if "dhw" in dcfg:
            q_w_dis = dist_side["Q_W_dis_in_kWh"]
            w_dis_cw = w_dis_cw + dist_side["W_W_dis_aux_kWh"].abs()

    w_em = w_em + w_c_em
    w_dis = w_dis_h + w_dis_cw
    for svc, q_in_s, q_out_s in (("heating", q_h_dis, q_h_em), ("cooling", q_c_dis, q_c_em), ("DHW", q_w_dis, q_w)):
        out_sum = float(q_out_s.sum())
        if out_sum > 1.0 and (float(q_in_s.sum()) - out_sum) / out_sum > 0.5:
            warnings.append(f"distribution losses for {svc} are {100 * (float(q_in_s.sum()) - out_sum) / out_sum:.0f}% of the demand ({float(q_in_s.sum()) - out_sum:.0f} kWh): check pipe lengths, psi and operation_mode.")

    # ---- AHU (system side) ----------------------------------------------------------
    ahu_coil_h = ahu_coil_c = ahu_fan = pd.Series(0.0, index=idx)
    if ahu_active:
        ahu_coil_h = _series(hourly, "Q_ahu_coil_ahu") / 1000.0 * dt
        ahu_coil_c = _series(hourly, "Q_ahu_cool_ahu") / 1000.0 * dt
        ahu_fan = _series(hourly, "P_ahu_fan_ahu") / 1000.0 * dt
        req_cool = _series(hourly, "Q_ahu_cool_req_ahu").sum() / 1000 * dt
        if req_cool > ahu_coil_c.sum() + 0.5:
            warnings.append(f"AHU cooling coil disabled or undersized: unmet AHU cooling {req_cool - float(ahu_coil_c.sum()):.0f} kWh (reported, not served).")
        if opt["ahu_coils_to_generation"]:
            notes.append("AHU coil energy is added to the generation demand (it is a system-side diagnostic in the zone solver, not part of Q_H/Q_C).")
        else:
            ahu_coil_h = ahu_coil_c = pd.Series(0.0, index=idx)

    q_h_gen = q_h_dis + ahu_coil_h
    q_c_gen = q_c_dis + ahu_coil_c
    q_w_gen = q_w_dis
    peak_gen = float((q_h_gen + q_w_gen).max() / dt)
    peak_cool = float(q_c_gen.max() / dt)

    # ============================================================================
    # GENERATION: every kind - including the EN 15316-4-1 boiler - is run in ONE
    # call that receives heating AND DHW together, so the generator's loss/
    # efficiency model is applied to the COMBINED load (see build_hvac_system()
    # for why the boiler can no longer go through HeatingSystemCalculator).
    # ============================================================================
    baseline = pd.Series(float(opt["baseline_electricity_kWh_per_year"]) / len(idx), index=idx)
    site_load_pre = w_em + w_dis + ahu_fan + baseline

    gcfg = ext_gen["config"]
    if opt["autosize"]:
        if kind == "heat_pump":
            gcfg["heating_performance_map"], f_h = _scale_map_to_peak(gcfg["heating_performance_map"], gcfg, "heating", peak_gen, margin)
            dhw_note = ""
            if "dhw_performance_map" in gcfg:
                # The DHW map generally has a different design-point capacity than
                # the heating map, so scaling both to the same target peak_gen
                # yields a DIFFERENT factor for each - keep it, don't discard it.
                gcfg["dhw_performance_map"], f_w = _scale_map_to_peak(gcfg["dhw_performance_map"], gcfg, "heating", peak_gen, margin)
                dhw_note = f", DHW map x{f_w:.2f}"
            if "cooling_performance_map" in gcfg and peak_cool > 0.05:
                gcfg["cooling_performance_map"], f_c = _scale_map_to_peak(gcfg["cooling_performance_map"], gcfg, "cooling", peak_cool, margin)
                notes.append(f"heat pump cooling map capacities scaled x{f_c:.2f} to the cooling peak {peak_cool:.1f} kW")
            notes.append(f"heat pump heating map capacities scaled x{f_h:.2f}{dhw_note} so capacity at the design point = {margin:.1f} x peak {peak_gen:.1f} kW")
        elif kind == "biomass_boiler":
            gcfg["nominal_power_kW"] = round(peak_gen * margin, 3)
            notes.append(f"{kind} nominal power = {peak_gen * margin:.1f} kW")
        elif kind == "cogeneration":
            gcfg["nominal_thermal_power_kW"] = round(0.4 * peak_gen, 3)
            notes.append(f"CHP thermal power = 40% of the peak = {0.4 * peak_gen:.1f} kW (heuristic; the remainder is covered by its backup)")
        elif kind == "boiler_15316_4_1":
            gcfg["rated_power_kW"] = round(peak_gen * margin, 3)
            notes.append(f"boiler rated_power_kW sized to {gcfg['rated_power_kW']:.1f} kW (heating + DHW combined peak).")

    carriers = {"electricity_kWh": 0.0, "natural_gas_kWh": 0.0, "biomass_kWh": 0.0, "district_heat_kWh": 0.0, "district_cooling_kWh": 0.0}
    gr_timeseries = None
    gs: dict[str, float] = {}

    if kind == "boiler_15316_4_1":
        # BoilerGeneratorCalculator (the class BEHIND EN 15316-4-1) natively accepts
        # Q_H_gen_out_kWh and Q_W_gen_out_kWh in the SAME call: it sizes the loss/
        # efficiency curve on the combined load and splits E_gen_in back into
        # E_H_gen_in / E_W_gen_in proportionally - the same idea as the standard's
        # separate expenditure factors epsilon_H,gen / epsilon_W,gen (Table 4).
        dcfg = hvac.get("distribution_15316_3_config", {})
        heat_supply = float(dcfg.get("heating", {}).get("supply_temperature_C", 45.0))
        heat_return = float(dcfg.get("heating", {}).get("return_temperature_C", 35.0))
        dhw_temp = float(dcfg.get("dhw", {}).get("dhw_temperature_C", 55.0))
        dhw_return = dhw_temp - float(dcfg.get("dhw", {}).get("dhw_return_deltaT_K", 5.0))
        # compute_step takes one operating temperature per hour, but heating and
        # DHW output can be simultaneous and generally differ in magnitude within
        # the same hour. Blend the two design temperatures by each end-use's
        # share of that hour's combined output, instead of jumping to the DHW
        # setpoint for the WHOLE hour as soon as any DHW is drawn (which would
        # understate the boiler's efficiency - and overstate its fuel use - for
        # the heating share of hours where DHW is only a small fraction of the
        # load). This still reduces to the DHW setpoint for DHW-only hours and
        # to the heating setpoint for heating-only hours.
        q_h_arr = q_h_gen.to_numpy(dtype=float)
        q_w_arr = q_w_gen.to_numpy(dtype=float)
        total_out = q_h_arr + q_w_arr
        w_dhw = np.divide(
            q_w_arr, total_out, out=np.zeros_like(total_out), where=total_out > 1e-9
        )
        theta_avg = heat_supply + w_dhw * (dhw_temp - heat_supply)
        theta_ret = heat_return + w_dhw * (dhw_return - heat_return)
        notes.append(
            f"boiler operating temperature: {heat_supply:.0f}/{heat_return:.0f} degC during space heating, "
            f"{dhw_temp:.0f}/{dhw_return:.0f} degC during DHW, blended hour-by-hour by each end-use's share "
            "of that hour's combined output where both occur together (a simplification since compute_step "
            "models one temperature per hour)."
        )
        boiler_ts = pd.DataFrame({
            "Q_H_gen_out_kWh": q_h_gen, "Q_W_gen_out_kWh": q_w_gen,
            "t_use_h": dt, "theta_avg_C": theta_avg, "theta_return_C": theta_ret,
        }, index=idx)
        gr_timeseries = BoilerGeneratorCalculator(gcfg).run_timeseries(boiler_ts)
        # compute_step does not cap Q_out by rated_power_kW (only the part-load ratio beta is
        # clipped at 1.0), so with autosize on the boiler always meets the requested load.
        out_h, out_w, out_c = float(q_h_gen.sum()), float(q_w_gen.sum()), 0.0
        e_gen_in = float(gr_timeseries["E_gen_in(kWh)"].sum())
        e_h_gen_in = float(gr_timeseries["E_H_gen_in(kWh)"].sum())
        e_w_gen_in = float(gr_timeseries["E_W_gen_in(kWh)"].sum())
        e_gen_aux = float(gr_timeseries["W_gen_aux(kWh)"].sum())
        fuel_carrier = "natural_gas_kWh" if str(gcfg.get("fuel_type", "natural_gas")).lower() in ("natural_gas", "gas", "lpg") else "biomass_kWh"
        carriers[fuel_carrier] = e_gen_in
        carriers["electricity_kWh"] = e_gen_aux
        gs = {"E_gen_in_kWh": e_gen_in, "E_H_gen_in_kWh": e_h_gen_in, "E_W_gen_in_kWh": e_w_gen_in, "W_gen_aux_kWh": e_gen_aux}
        notes.append(f"boiler EN 15316-4-1: {e_h_gen_in:.0f} kWh of fuel attributed to heating, {e_w_gen_in:.0f} kWh to DHW (split proportionally to each service's share of the hourly combined load).")
    else:
        gen_loads = pd.DataFrame({"T_ext": t_ext, "Q_H_kWh": q_h_gen, "Q_W_kWh": q_w_gen, "Q_C_kWh": q_c_gen, "time_step_hours": dt}, index=idx)
        if kind == "cogeneration":
            gen_loads["E_site_el_load_kWh"] = site_load_pre

        cls = {
            "heat_pump": HeatPumpSystemCalculator, "biomass_boiler": BiomassBoilerSystemCalculator,
            "district": DistrictEnergySystemCalculator, "cogeneration": CogenerationSystemCalculator,
        }[kind]
        gr = cls(gcfg).run_timeseries(gen_loads)
        gs = gr.summary
        gr_timeseries = getattr(gr, "timeseries", None)  # HeatPumpSystemCalculator returns bins, not an hourly timeseries
        out_h, out_w, out_c = gs.get("QH_gen_out_kWh", 0.0), gs.get("QW_gen_out_kWh", 0.0), gs.get("QC_gen_out_kWh", 0.0)
        carriers["electricity_kWh"] = float(gs.get("E_total_electricity_kWh", 0.0))
        if kind == "cogeneration":
            carriers["natural_gas_kWh"] = float(gs.get("E_total_fuel_kWh", 0.0))
        elif kind == "biomass_boiler":
            carriers["biomass_kWh"] = float(gs.get("E_biomass_delivered_kWh", 0.0))
        elif kind == "district":
            carriers["district_heat_kWh"] = float(gs.get("EHW_district_in_kWh", 0.0))
            carriers["district_cooling_kWh"] = float(gs.get("EC_district_in_kWh", 0.0))

    unmet_h = max(float(q_h_gen.sum()) - out_h, 0.0)
    unmet_w = max(float(q_w_gen.sum()) - out_w, 0.0)
    unmet_c = max(float(q_c_gen.sum()) - out_c, 0.0)
    if unmet_h > 0.5:
        warnings.append(f"generation does not cover {unmet_h:.0f} kWh of heating demand.")
    if unmet_w > 0.5:
        warnings.append(f"generation does not cover {unmet_w:.0f} kWh of DHW demand.")

    chiller_el = pd.Series(0.0, index=idx)
    if unmet_c > 0.5:
        eer = opt["fallback_cooling_eer"]
        if eer:
            served_share = out_c / max(float(q_c_gen.sum()), 1e-9)
            chiller_el = q_c_gen * (1.0 - served_share) / float(eer)
            carriers["electricity_kWh"] += float(chiller_el.sum())
            notes.append(f"unserved cooling {unmet_c:.0f} kWh covered by an ideal chiller with EER {eer}.")
        else:
            warnings.append(f"{unmet_c:.0f} kWh of cooling is not served (set options.fallback_cooling_eer to add a chiller).")

    # ============================================================================
    # Site electricity balance, PV, primary energy
    # ============================================================================
    thermal_w = (q_h_gen + q_w_gen + q_c_gen).clip(lower=0.0)
    weights = thermal_w / thermal_w.sum() if thermal_w.sum() > 0 else pd.Series(0.0, index=idx)
    if kind == "boiler_15316_4_1":
        gen_el_alloc = gr_timeseries["W_gen_aux(kWh)"].astype(float)  # real hourly series, not an allocation
    elif kind == "cogeneration":
        gen_el_alloc = weights * float(gs.get("E_total_electricity_kWh", 0.0))
    else:
        gen_el_alloc = weights * (carriers["electricity_kWh"] - float(chiller_el.sum()))
    site_load = site_load_pre + gen_el_alloc + chiller_el
    if kind in ("heat_pump", "biomass_boiler"):
        notes.append("hourly generator electricity is an allocation of the annual total in proportion to the hourly thermal demand (the bin method has no hourly series).")

    residual = site_load.copy()
    if kind == "cogeneration" and gr_timeseries is not None and "E_chp_el_self_consumed_kWh" in gr_timeseries.columns:
        residual = (site_load - gr_timeseries["E_chp_el_self_consumed_kWh"]).clip(lower=0.0)

    pv_summary: dict[str, float] = {}
    pv_ts = None
    if config.get("pv"):
        pv_in = pd.DataFrame({"E_site_el_load_kWh": residual, "time_step_hours": dt}, index=idx)
        pvr = RenewableEnergySystemCalculator(config["pv"]).run_timeseries(pv_in)
        pv_summary, pv_ts = pvr.summary, pvr.timeseries
        grid_h, export_h = pv_ts["E_grid_after_PV_kWh"].astype(float), pv_ts["E_PV_export_kWh"].astype(float)
    else:
        grid_h, export_h = residual.astype(float), pd.Series(0.0, index=idx)
    if kind == "cogeneration" and gr_timeseries is not None and "E_chp_el_export_kWh" in gr_timeseries.columns:
        export_h = export_h + gr_timeseries["E_chp_el_export_kWh"].astype(float)

    # The accounting calculator clips the net primary energy at 0 PER ROW it receives, so
    # the balance period decides how far export credit offsets deliveries.
    pe_hourly = pd.DataFrame({
        "E_delivered_electricity_kWh": grid_h, "E_exported_electricity_kWh": export_h,
        "E_delivered_natural_gas_kWh": weights * carriers["natural_gas_kWh"],
        "E_biomass_delivered_kWh": weights * carriers["biomass_kWh"],
        "E_delivered_district_heat_kWh": weights * carriers["district_heat_kWh"],
        "E_delivered_district_cooling_kWh": weights * carriers["district_cooling_kWh"],
    }, index=idx)
    pe_cfg = {"primary_energy_factors": opt["primary_energy_factors"]} if opt["primary_energy_factors"] else {}
    pe_calc = PrimaryEnergyAccountingCalculator(pe_cfg)
    balance = str(opt["pe_balance"]).lower()
    if balance == "hourly":
        pe = pe_calc.run_timeseries(pe_hourly).summary
    elif balance == "monthly":
        shifted = pe_hourly.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=dt)
        pe = pe_calc.run_timeseries(shifted.resample("MS").sum()).summary
    elif balance == "annual":
        pe = pe_calc.run_annual(pe_hourly.sum().to_dict()).summary
    else:
        raise ValueError("options.pe_balance must be 'annual', 'monthly' or 'hourly'")
    if pe["PE_export_credit_nonren_kWh"] > pe["PE_delivered_nonren_kWh"] + 1e-9:
        notes.append(f"export credit ({pe['PE_export_credit_nonren_kWh']:.0f} kWh non-renewable PE) exceeds the deliveries of the {balance} balance: the net is clipped at 0 by the accounting method.")
    grid_import, export_total = float(pe["E_delivered_electricity_kWh"]), float(pe["E_exported_electricity_kWh"])

    need_total = float(q_h.sum() + q_c.sum() + q_w.sum())
    final_total = sum(carriers.values())
    final_by_carrier = dict(carriers)
    final_by_carrier["electricity_kWh"] = carriers["electricity_kWh"] + float(w_em.sum() + w_dis.sum() + ahu_fan.sum() + baseline.sum())

    summary: dict[str, float] = {
        "floor_area_m2": area,
        "Q_H_need_kWh": float(q_h.sum()), "Q_C_need_kWh": float(q_c.sum()), "Q_W_need_kWh": float(q_w.sum()),
        "Q_H_em_in_kWh": float(q_h_em.sum()), "Q_C_em_in_kWh": float(q_c_em.sum()), "W_em_aux_kWh": float(w_em.sum()),
        "Q_H_dis_in_kWh": float(q_h_dis.sum()), "Q_C_dis_in_kWh": float(q_c_dis.sum()), "Q_W_dis_in_kWh": float(q_w_dis.sum()), "W_dis_aux_kWh": float(w_dis.sum()),
        "Q_H_gen_demand_kWh": float(q_h_gen.sum()), "Q_C_gen_demand_kWh": float(q_c_gen.sum()), "Q_W_gen_demand_kWh": float(q_w_gen.sum()),
        "Q_unmet_kWh": unmet_h + unmet_w + (0.0 if opt["fallback_cooling_eer"] else unmet_c),
        "PE_balance": balance, "PE_delivered_nonren_kWh": float(pe["PE_delivered_nonren_kWh"]), "PE_export_credit_nonren_kWh": float(pe["PE_export_credit_nonren_kWh"]),
        "final_energy_by_carrier_kWh": final_by_carrier,
        "E_grid_import_kWh": grid_import, "E_export_kWh": export_total,
        "PE_net_nonren_kWh": float(pe["PE_net_nonren_kWh"]), "PE_net_total_kWh": float(pe["PE_net_total_kWh"]),
        "system_efficiency": need_total / max(final_total + float(w_em.sum() + w_dis.sum() + ahu_fan.sum()), 1e-9),
        "peak_H_kW": peak_h, "peak_C_kW": peak_c,
    }
    if ahu_active:
        summary.update({"ahu_coil_heat_kWh": float(ahu_coil_h.sum()), "ahu_coil_cool_kWh": float(ahu_coil_c.sum()), "ahu_fan_kWh": float(ahu_fan.sum())})
    if pv_summary:
        summary.update({k: float(pv_summary[k]) for k in ("E_PV_gen_kWh", "E_PV_self_consumed_kWh", "E_PV_export_kWh", "f_PV_self_consumed")})
    summary["generator_summary"] = {k: float(v) for k, v in gs.items() if isinstance(v, (int, float, np.floating))}

    hourly_out = pd.DataFrame({
        "T_ext": t_ext, "T_op": t_op, "Q_H_need_kWh": q_h, "Q_C_need_kWh": q_c, "Q_W_need_kWh": q_w,
        "Q_H_em_in_kWh": q_h_em, "Q_C_em_in_kWh": q_c_em, "Q_H_dis_in_kWh": q_h_dis, "Q_C_dis_in_kWh": q_c_dis, "Q_W_dis_in_kWh": q_w_dis,
        "AHU_coil_heat_kWh": ahu_coil_h, "AHU_coil_cool_kWh": ahu_coil_c, "AHU_fan_kWh": ahu_fan,
        "W_aux_kWh": w_em + w_dis, "E_site_el_load_kWh": site_load,
    }, index=idx)
    if pv_ts is not None:
        for col in ("E_PV_gen_kWh", "E_PV_self_consumed_kWh", "E_PV_export_kWh", "E_grid_after_PV_kWh", "E_battery_state_of_charge_kWh"):
            if col in pv_ts.columns:
                hourly_out[col] = pv_ts[col]

    return CompositionResult(config=config, hourly=hourly_out, summary=summary, warnings=warnings, notes=notes)


def run_scenario(scenario: dict[str, Any] | None = None, catalog: dict | None = None, hourly: pd.DataFrame | None = None) -> CompositionResult:
    """Convenience: compose the combined JSON, then simulate it. See compose_config() + simulate()."""

    config = compose_config(scenario, catalog=catalog)
    return simulate(config, hourly=hourly)


def compare_generation(base: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run the same building/emission/distribution/PV/AHU with every generation option."""

    catalog = build_catalog()
    base = copy.deepcopy(base or {})
    base.setdefault("options", {}).setdefault("fallback_cooling_eer", 3.0)
    hourly = None
    rows = []
    for name in catalog["generation"]:
        res = run_scenario({**base, "generation": name}, catalog=catalog, hourly=hourly)
        s = res.summary
        rows.append({
            "generation": name, "final_el_kWh": s["final_energy_by_carrier_kWh"]["electricity_kWh"], "gas_kWh": s["final_energy_by_carrier_kWh"]["natural_gas_kWh"],
            "biomass_kWh": s["final_energy_by_carrier_kWh"]["biomass_kWh"], "district_kWh": s["final_energy_by_carrier_kWh"]["district_heat_kWh"],
            "grid_import_kWh": s["E_grid_import_kWh"], "PE_nonren_kWh": s["PE_net_nonren_kWh"], "PE_total_kWh": s["PE_net_total_kWh"],
            "unmet_kWh": s["Q_unmet_kWh"], "efficiency": s["system_efficiency"],
        })
    return pd.DataFrame(rows).set_index("generation").round(1)


def save_outputs(res: CompositionResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.hourly.to_csv(out_dir / "composed_hourly.csv")
    save_config(res.config, out_dir / "composed_config.json")
    payload = {"summary": res.summary, "warnings": res.warnings, "notes": res.notes}
    (out_dir / "composed_summary.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Minimal chain: building + emission system ONLY (no distribution, no generation, no AHU)
# --------------------------------------------------------------------------------------

EMISSION_INC_METHODS = ("recalculate", "approximate")


@dataclass
class EmissionOnlyResult:
    """Result of simulate_emission_only(): building need -> emission input, nothing further."""

    config: dict[str, Any]
    hourly: pd.DataFrame
    summary: dict[str, float]
    emission_inc_method: str = "recalculate"

    def report(self) -> str:
        s = self.summary
        area = s["floor_area_m2"]
        return "\n".join([
            "=" * 60,
            f"{'Zone need (ISO 52016)':30s} H {s['QH_em_out_kWh']:8.0f}  C {s['QC_em_out_kWh']:8.0f} kWh",
            f"{'Emission input (EN 15316-2)':30s} H {s['QH_em_in_kWh']:8.0f}  C {s['QC_em_in_kWh']:8.0f} kWh",
            f"{'Emission auxiliary electricity':30s} H {s['WH_em_aux_kWh']:8.1f}  C {s['WC_em_aux_kWh']:8.1f} kWh_el",
            f"({(s['QH_em_in_kWh'] + s['QC_em_in_kWh']) / area:.1f} kWh/m2a at the emission input, H+C)",
            "=" * 60,
            "Note: this is energy at the EMITTERS, not final/delivered energy -",
            "no distribution or generation step is modelled here.",
            f"Emission output at modified set point: {self.emission_inc_method}"
            + (" (building need recalculated with ISO 52016)" if self.emission_inc_method == "recalculate"
               else " (temperature-ratio approximation: shorter computation time, approximate results)"),
        ])


def _shift_setpoints(building: dict[str, Any], h_delta: float, c_delta: float) -> dict[str, Any]:
    """Copy of the building with heating/cooling set-points moved by the EN 15316-2 equivalent
    temperature increase (eq. 16: theta_inc = theta_ini + delta, cooling deltas are negative)."""
    shifted = copy.deepcopy(building)
    sp = shifted["building_parameters"]["temperature_setpoints"]
    for key in ("heating_setpoint", "heating_setback"):
        if key in sp:
            sp[key] = float(sp[key]) + h_delta
    for key in ("cooling_setpoint", "cooling_setback"):
        if key in sp:
            sp[key] = float(sp[key]) + c_delta
    return shifted


def simulate_emission_only(config: dict[str, Any], emission_inc: str = "recalculate") -> EmissionOnlyResult:
    """Run ONLY ISO 52016 (building need) -> EmissionSystemCalculator (EN 15316-2).

    ``config`` is a plain ``{"building": {...}, "weather": {...},
    "emission_system_config": {...}}`` document (see
    examples/example_building_emission_only.json) - no "hvac_system", no
    distribution, no generation, no AHU. EmissionSystemCalculator is used
    standalone here, unlike HeatingSystemCalculator (which always bundles
    emission + distribution for space heating) or simulate()/compose_config()
    (which always require a generator under "external_generation").

    ``emission_inc`` selects how the emission output at the modified set point
    (EN 15316-2 Q_em,out,inc) is obtained:
      * ``"recalculate"`` (default): the building need is recalculated with ISO 52016
        at the equivalent set points, as the standard requires (M2-2);
      * ``"approximate"``: scaled by the temperature ratios. Faster, but the results
        are approximate and a notification is raised.
    """
    if emission_inc not in EMISSION_INC_METHODS:
        raise ValueError(f"emission_inc must be one of {EMISSION_INC_METHODS}, got {emission_inc!r}")
    if "emission_system_config" not in config:
        config = normalize_system_config(config)

    bui = config["building"]
    weather = config["weather"]

    checked, issues = sanitize_and_validate_BUI(copy.deepcopy(bui), fix=True)
    errors = [i for i in issues if i["level"] == "ERROR"]
    if errors:
        raise ValueError(f"Building validation errors: {errors}")

    kwargs: dict[str, Any] = {"weather_source": weather["source"]}
    if weather["source"] == "epw":
        kwargs["path_weather_file"] = weather["file"]
    hourly = ISO52016.Temperature_and_Energy_needs_calculation(checked, **kwargs)[0]

    dt = float(pd.Series(hourly.index).diff().dt.total_seconds().median() / 3600.0)
    emission_cfg = copy.deepcopy(config["emission_system_config"])
    if emission_inc == "approximate":
        for svc in ("heating", "cooling"):
            emission_cfg.setdefault(svc, {})["notify_inc_fallback"] = False
    calc = EmissionSystemCalculator(emission_cfg)
    em_input = pd.DataFrame({
        "T_ext": hourly["T_ext"], "T_op": hourly["T_op"],
        "Q_H_kWh": hourly["Q_H"].clip(lower=0.0) / 1000.0 * dt,   # W -> kWh
        "Q_C_kWh": hourly["Q_C"].clip(lower=0.0) / 1000.0 * dt,
        "time_step_hours": dt,
    }, index=hourly.index)

    if emission_inc == "recalculate":
        shifted = _shift_setpoints(checked, calc.temperature_increase_K("H"), calc.temperature_increase_K("C"))
        hourly_inc = ISO52016.Temperature_and_Energy_needs_calculation(shifted, **kwargs)[0]
        em_input["Q_H_em_out_inc_kWh"] = hourly_inc["Q_H"].clip(lower=0.0).reindex(hourly.index).to_numpy() / 1000.0 * dt
        em_input["Q_C_em_out_inc_kWh"] = hourly_inc["Q_C"].clip(lower=0.0).reindex(hourly.index).to_numpy() / 1000.0 * dt
    else:
        warnings.warn(
            "Emission output at the modified set point is approximated by temperature ratios: "
            "shorter computation time, but approximate results (EN 15316-2 requires the building "
            "need to be recalculated at the equivalent set point; use emission_inc='recalculate').",
            UserWarning,
            stacklevel=2,
        )

    result = calc.run_timeseries(em_input)
    summary = dict(result.summary)
    summary["floor_area_m2"] = float(bui["building"]["net_floor_area"])
    return EmissionOnlyResult(config=config, hourly=result.timeseries, summary=summary,
                              emission_inc_method=emission_inc)
