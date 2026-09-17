"""Top-level package for pyBuildingEnergy.

The public API is loaded on first access. Keeping package import lightweight is
important for command-line tools, test discovery and applications that only use
one of the standard-specific calculators.
"""

from importlib import import_module


__author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
__email__ = "daniele.antonucci@eurac.edu"
__version__ = "2.0.3"


_PUBLIC_IMPORTS = {
    "sanitize_and_validate_BUI": (".source.check_input", "sanitize_and_validate_BUI"),
    "check_heating_system_inputs": (".source.check_input", "check_heating_system_inputs"),
    "ISO52016": (".source.utils", "ISO52016"),
    "Graphs_and_report": (".source.graphs", "Graphs_and_report"),
    "HeatingSystemCalculator": (".source.iso_15316_1", "HeatingSystemCalculator"),
    "EmissionSimulationResult": (".source.emission_15316_2", "EmissionSimulationResult"),
    "EmissionSystemCalculator": (".source.emission_15316_2", "EmissionSystemCalculator"),
    "DistributionSimulationResult": (".source.distribution_15316_3", "DistributionSimulationResult"),
    "DistributionSystemCalculator": (".source.distribution_15316_3", "DistributionSystemCalculator"),
    "StorageSimulationResult": (".source.storage_15316_5", "StorageSimulationResult"),
    "StorageSystemCalculator": (".source.storage_15316_5", "StorageSystemCalculator"),
    "BoilerConfig": (".source.generation_15316_4_1", "BoilerConfig"),
    "BoilerGeneratorCalculator": (".source.generation_15316_4_1", "BoilerGeneratorCalculator"),
    "CoolingSystemSimulationResult": (".source.cooling_16798_9", "CoolingSystemSimulationResult"),
    "CoolingSystemCalculator": (".source.cooling_16798_9", "CoolingSystemCalculator"),
    "CoolingStorageSimulationResult": (".source.cooling_storage_16798_15", "CoolingStorageSimulationResult"),
    "CoolingStorageSystemCalculator": (".source.cooling_storage_16798_15", "CoolingStorageSystemCalculator"),
    "CoolingGenerationSimulationResult": (".source.cooling_generation_16798_13", "CoolingGenerationSimulationResult"),
    "CoolingGenerationSystemCalculator": (".source.cooling_generation_16798_13", "CoolingGenerationSystemCalculator"),
    "HeatPumpPerformanceDataCalculator": (".source.performance_14511_14825", "HeatPumpPerformanceDataCalculator"),
    "HeatPumpPerformanceDataResult": (".source.performance_14511_14825", "HeatPumpPerformanceDataResult"),
    "en14825_part_load_factor": (".source.performance_14511_14825", "en14825_part_load_factor"),
    "HeatPumpSimulationResult": (".source.heat_pump_15316_4_2", "HeatPumpSimulationResult"),
    "HeatPumpSystemCalculator": (".source.heat_pump_15316_4_2", "HeatPumpSystemCalculator"),
    "CombustionBoilerSimulationResult": (".source.combustion_15316_4_1", "CombustionBoilerSimulationResult"),
    "CombustionBoilerSystemCalculator": (".source.combustion_15316_4_1", "CombustionBoilerSystemCalculator"),
    "CogenerationSimulationResult": (".source.cogeneration_15316_4_4", "CogenerationSimulationResult"),
    "CogenerationSystemCalculator": (".source.cogeneration_15316_4_4", "CogenerationSystemCalculator"),
    "DistrictEnergySystemCalculator": (".source.district_15316_4_5", "DistrictEnergySystemCalculator"),
    "DistrictSystemSimulationResult": (".source.district_15316_4_5", "DistrictSystemSimulationResult"),
    "RenewableEnergySimulationResult": (".source.renewables_15316_4_3_4_6", "RenewableEnergySimulationResult"),
    "RenewableEnergySystemCalculator": (".source.renewables_15316_4_3_4_6", "RenewableEnergySystemCalculator"),
    "PrimaryEnergyAccountingCalculator": (".source.primary_energy_52000_1", "PrimaryEnergyAccountingCalculator"),
    "PrimaryEnergyAccountingResult": (".source.primary_energy_52000_1", "PrimaryEnergyAccountingResult"),
    "LightingSimulationResult": (".source.lighting_15193_1", "LightingSimulationResult"),
    "LightingSystemCalculator": (".source.lighting_15193_1", "LightingSystemCalculator"),
    "VentilationSystemCalculator": (".source.ventilation_16798_5_7", "VentilationSystemCalculator"),
    "VentilationSystemSimulationResult": (".source.ventilation_16798_5_7", "VentilationSystemSimulationResult"),
    "BACSControlFactorCalculator": (".source.bacs_52120_1", "BACSControlFactorCalculator"),
    "BACSSimulationResult": (".source.bacs_52120_1", "BACSSimulationResult"),
    "CostOptimalityCalculator": (".source.economics_15459_1", "CostOptimalityCalculator"),
    "EconomicSimulationResult": (".source.economics_15459_1", "EconomicSimulationResult"),
    "BiomassBoilerSimulationResult": (".source.biomass_15316_4", "BiomassBoilerSimulationResult"),
    "BiomassBoilerSystemCalculator": (".source.biomass_15316_4", "BiomassBoilerSystemCalculator"),
    "ItalianStrepinTables": (".data.italian_strepin", "ItalianStrepinTables"),
    "StrepinCase": (".data.italian_strepin", "StrepinCase"),
    "apply_extra_measure_specs_to_bui": (".data.italian_strepin", "apply_extra_measure_specs_to_bui"),
    "find_default_workbook": (".data.italian_strepin", "find_default_workbook"),
    "load_italian_strepin_tables": (".data.italian_strepin", "load_italian_strepin_tables"),
    "summarize_engine_performance": (".data.italian_strepin", "summarize_engine_performance"),
    "DHWDesignSimulationResult": (".source.DHW", "DHWDesignSimulationResult"),
    "DHWDesignLoadCalculator": (".source.DHW", "DHWDesignLoadCalculator"),
    "Volume_and_energy_DHW_calculation": (".source.DHW", "Volume_and_energy_DHW_calculation"),
    "generate_calendar": (".source.DHW", "generate_calendar"),
    "HourlyProfileGenerator": (".source.generate_profile", "HourlyProfileGenerator"),
    "get_country_code_from_latlon": (".source.generate_profile", "get_country_code_from_latlon"),
}

# Compatibility for attributes historically exposed by wildcard imports.
_LEGACY_MODULES = (
    ".source.DHW",
    ".source.graphs",
    ".source.utils",
    ".source.ventilation",
    ".source.table_iso_16798_1",
)

__all__ = list(_PUBLIC_IMPORTS)


def __getattr__(name):
    """Load a public or legacy top-level attribute only when it is requested."""
    target = _PUBLIC_IMPORTS.get(name)
    if target is not None:
        module_name, attribute_name = target
        value = getattr(import_module(module_name, __name__), attribute_name)
        globals()[name] = value
        return value

    for module_name in _LEGACY_MODULES:
        module = import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
