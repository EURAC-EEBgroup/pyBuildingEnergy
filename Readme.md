# pyBuildingEnergy

![pyBuildingEnergy Logo](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/Logo_pyBuild.png)

## Citation


*Please cite us if you use this library*: 
[![DOI](https://zenodo.org/badge/761715706.svg)](https://zenodo.org/doi/10.5281/zenodo.10887919)

## Documentation **(New)**

Check our **new documentation** in GitHub Pages: [pybuildingenergy docs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/).

## Find Us
Where to find our works:
- [pysimhub](https://pysimhub.io/) .
- [relife project](https://relife.test.ctic.es/) - Under development
...

## Features

The new EPBD recast provides an update on building performance assessment through a methodology that must take into account various aspects such as the thermal characteristics of the building, the use of energy from renewable sources, building automation and control systems, ventilation, cooling, energy recovery, etc.

The methodology should represent the actual operating conditions, allow for the use of measured energy for accuracy and comparability purposes, and be based on hourly or sub-hourly intervals that take into account the variable conditions significantly impacting the operation and performance of the system, as well as internal conditions.

**pyBuildingEnergy** aims to provide an assessment of building performance both in terms of energy and comfort. In this initial release, it is possible to assess the energy performance of the building using ISO 52016-1:2018. Additional modules will be added for a more comprehensive evaluation of performance, assessing ventilation, renewable energies, systems, etc.

The actual calculation methods for the assessment of building performance are the following:

- [x] the (sensible) energy need for heating and cooling, based on hourly or monthly calculations;
- [ ] the latent energy need for (de-)humidification, based on hourly or monthly calculations;
- [x] the internal temperature, based on hourly calculations;
- [x] the sensible heating and cooling load, based on hourly calculations;
- [ ] the moisture and latent heat load for (de-)humidification, based on hourly calculations;
- [ ] the design sensible heating or cooling load and design latent heat load using an hourly calculation interval;
- [ ] the conditions of the supply air to provide the necessary humidification and dehumidification.

The calculation methods can be used for residential or non-residential buildings, or a part of it, referred to as "the building" or the "assessed object".

ISO 52016-1:2018 also contains specifications for the assessment of thermal zones in the building or in the part of a building. The calculations are performed per thermal zone. In the calculations, the thermal zones can be assumed to be thermally coupled or not. ISO 52016-1:2018 is applicable to buildings at the design stage, to new buildings after construction and to existing buildings in the use phase.

-- 

## Weather Data

The tool can use weather data coming from 2 main sources:

- PVGIS API ([link](https://re.jrc.ec.europa.eu/pvg_tools/en/)) - PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM
- `.epw` file from [Ladybug Tools EPWMap](https://www.ladybug.tools/epwmap/)

More details in the example folder.

## Domestic Hot Water - DHW

- [x] Calculation of volume and energy need for domestic hot water according to EN 12831-3.
- [x] Assessment of DHW system design heat load with EN 12831-3 storage switch points, time lag, effective reheating power and supply-curve sizing.

For the audit trail between the standard, code and outputs, open
[DHW EN 12831-3 Implementation Audit](docs/dhw_12831_3_audit.html).

## Space Emission Systems - EN 15316-2 **(New)**

`EmissionSystemCalculator` evaluates space-heating and water-based space-cooling
emission effects before generation. It can be used to account for emitter and
control effects, equivalent internal-temperature changes, embedded emitter
losses, emission auxiliary electricity and annual emission expenditure factors.

For an audit trail between the standard, code and output files, open
[Emission EN 15316-2 Implementation Audit](docs/emission_15316_2_audit.html).

## Water-Based Distribution Systems - EN 15316-3 **(New)**

`DistributionSystemCalculator` evaluates water-based distribution systems for
space heating, space cooling and DHW. It calculates pipe thermal losses,
recoverable distribution losses, pump auxiliary electricity, recoverable pump
heat and recovered pump heat in the fluid before the generator calculation.

For the audit trail, open
[Distribution EN 15316-3 Implementation Audit](docs/distribution_15316_3_audit.html).

## Heating And DHW Storage Systems - EN 15316-5 **(New)**

`StorageSystemCalculator` evaluates single-volume Method B storage systems for
space heating and DHW. It calculates storage standing losses from product
standby-loss data or a direct heat-loss coefficient, storage charging pump
auxiliary electricity, recoverable storage losses and heat recovered in the
medium before the generator calculation.

For the audit trail, open
[Storage EN 15316-5 Implementation Audit](docs/storage_15316_5_audit.html).

## Heat Pump Generation - EN 15316-4-2 **(New)**

`HeatPumpSystemCalculator` evaluates heat-pump generation for:

- space heating,
- domestic hot water (DHW),
- and an optional simplified reversible space-cooling branch with an EER map.

The heating and DHW calculation follows the detailed EN 15316-4-2 bin-method structure: outdoor/source temperature bins, product heating capacity and COP maps, source/sink temperature operating points, runtime/capacity checks, auxiliary energy, optional simplified storage losses, backup energy and SPF outputs. The runnable examples use the EN 16798 cooling modules by default; the older heat-pump EER cooling branch is still available with `--calculation-path heat-pump-cooling`.

For a clause-by-clause audit trail between the standard, the implementation and the output files, open [Heat Pump EN 15316-4-2 Implementation Audit](docs/heat_pump_15316_4_2_audit.html).

## Additional Generation And Renewable Modules - EN 15316-4-1, EN 15316-4-4, EN 15316-4-5, EN 15316-4-3/4-6 **(New)**

The Athens and Bolzano system workflow can now switch the heating/DHW generator
away from the default heat pump:

- `CombustionBoilerSystemCalculator` implements an EN 15316-4-1 combustion-generator boundary with part-load/return-temperature efficiency, standby losses and auxiliaries.
- `CogenerationSystemCalculator` implements an EN 15316-4-4 thermal-led CHP boundary with fuel input, useful electricity, self-consumption/export and backup heat.
- `DistrictEnergySystemCalculator` implements an EN 15316-4-5 district heating/cooling substation boundary.
- `RenewableEnergySystemCalculator` implements the EN 15316-4-3/4-6 solar thermal and PV accounting branch used by the examples and STREPIN runner.

Audit pages:
[Generation EN 15316-4-1/4-4/4-5](docs/generation_15316_4_1_4_4_4_5_audit.html) and
[Renewables EN 15316-4-3/4-6](docs/renewables_15316_4_3_4_6_audit.html).

## Heat Pump Product Performance - EN 14511 And EN 14825 **(New)**

`HeatPumpPerformanceDataCalculator` normalizes heat-pump rating data from EN
14511-style capacity/COP/EER points and calculates EN 14825 part-load
inspection values. The examples use this path by default and apply the EN 14825
water-based part-load correction to heating/DHW COP and EN 16798-13 cooling EER.

For the audit trail between the standards, code and outputs, open
[EN 14511 / EN 14825 Product Performance Implementation Audit](docs/performance_14511_14825_audit.html).

## Cooling System Modules - EN 16798-9, EN 16798-15 And EN 16798-13 **(New)**

The heat-pump examples now treat space cooling with the cooling-side EN 16798 standards:

- `CoolingSystemCalculator` implements EN 16798-9 operating conditions and the handoff of cooling requests to storage or generation.
- `CoolingStorageSystemCalculator` implements EN 16798-15 chilled-water storage heat gains and pump auxiliary energy.
- `CoolingGenerationSystemCalculator` implements EN 16798-13 compression cooling generation using cooling capacity/EER product maps or nominal EER fallback data.

For the audit trail between the standards, code and outputs, open [Cooling EN 16798 Implementation Audit](docs/cooling_16798_audit.html).

For the whole end-to-end workflow that interconnects EN ISO 52016, EN 12831-3,
EN 15316, EN 16798, EN 14511 and EN 14825 in the Athens and Bolzano examples,
open [Heat-Pump Example Simulation Workflow Audit](docs/simulation_workflow_audit.html).

```python
import pandas as pd
import pybuildingenergy as pybui

heating_map = pd.DataFrame({
    "source_temperature_C": [-7, -7, 2, 2, 7, 7],
    "sink_temperature_C": [35, 55, 35, 55, 35, 55],
    "capacity_kW": [5.0, 4.0, 6.0, 5.0, 7.0, 6.0],
    "cop": [3.2, 2.4, 3.8, 2.8, 4.2, 3.2],
})

cooling_map = pd.DataFrame({
    "source_temperature_C": [25, 25, 35, 35],
    "sink_temperature_C": [7, 18, 7, 18],
    "capacity_kW": [5.0, 6.0, 4.0, 5.0],
    "eer": [3.0, 3.6, 2.5, 3.1],
})

loads = pd.DataFrame({
    "T_ext": [-5, 0, 5, 25],
    "Q_H_kWh": [4.0, 3.0, 1.0, 0.0],
    "Q_C_kWh": [0.0, 0.0, 0.0, 2.0],
    "Q_W_kWh": [0.5, 0.5, 0.5, 0.5],
})

calc = pybui.HeatPumpSystemCalculator({
    "heating_performance_map": heating_map,
    "dhw_performance_map": heating_map,
    "cooling_performance_map": cooling_map,
    "source_type": "air",
    "demand_unit": "kWh",
    "dhw_target_temperature_C": 55,
    "dhw_sink_temperature_C": 55,
    "external_auxiliary_power_W": 100,
})

result = calc.run_timeseries(loads)
print(result.summary["SPF_HW_gen"])
print(result.summary["SEER_C_gen"])
```

### Run The Heat Pump Example

The runnable examples are:

```bash
python -m pip install -r requirements.txt
python examples/heat_pump_15316_4_2_example.py --scenario athens
python examples/heat_pump_15316_4_2_example.py --scenario bolzano
```

There is also a Bolzano convenience wrapper:

```bash
python examples/heat_pump_15316_4_2_bolzano_example.py
```

The Athens scenario uses an Athens PVGIS weather location, a Greece DHW calendar, the EN 12831-3 Annex B Table B.2 single-family dwelling hourly DHW profile and a 26 C cooling setpoint. The Bolzano scenario uses Bolzano coordinates, an Italy DHW calendar, the same residential Annex B hourly DHW profile, a tighter solar-exposed envelope and an air-to-water heat-pump map sized for a 120 m2 useful-floor-area residential building. In both cases the example geometry is a two-floor building with a 60 m2 footprint; roof and ground-slab areas are therefore 60 m2, while `net_floor_area` remains 120 m2.

Each scenario runs ISO52016 for the example building, applies EN 15316-2 emission effects, EN 15316-3 distribution effects, EN 15316-5 heating/DHW storage effects and the EN 16798 cooling-side modules by default, calculates an hourly DHW profile and the EN 12831-3 DHW design sizing, runs the generator calculations and writes:

- `examples/outputs/heat_pump_15316_4_2_<scenario>/iso52016_loads_with_dhw.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/dhw_12831_3_design_timeseries.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/dhw_12831_3_design_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/building_geometry_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/emission_15316_2_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/emission_15316_2_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/distribution_15316_3_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/distribution_15316_3_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/storage_15316_5_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/storage_15316_5_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/cooling_16798_9_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/cooling_16798_9_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/cooling_storage_16798_15_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/cooling_storage_16798_15_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/cooling_generation_16798_13_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/cooling_generation_16798_13_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/performance_14511_14825_rating_points.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/performance_14511_14825_heating_map.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/performance_14511_14825_cooling_map.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/performance_14511_14825_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/heat_pump_hourly_allocated_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/heat_pump_bin_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/heat_pump_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/combined_generation_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/inspection_index.html`

Open `inspection_index.html` in a browser to inspect the visual outputs. The page links to:

- a user-facing overview of annual loads, final electricity, monthly trends,
  seasonal performance and useful-area intensities;
- a workflow handoff plot that shows how heating, cooling and DHW loads move
  through the implemented standards;
- sanity-check plots for geometry, peak loads versus active capacity, backup
  shares and unmet loads;
- EN 12831-3 DHW design-day needs, storage switch curves, residual storage and
  effective reheating power;
- the existing ISO52016 building report generated with `Graphs_and_report`;
- the whole simulation workflow audit trail;
- daily input time series for heating, cooling, DHW and temperatures;
- EN 15316-2 emission time series and monthly aggregate plots;
- EN 15316-3 distribution time series and monthly aggregate plots;
- EN 15316-5 storage time series and monthly aggregate plots;
- EN 16798-9 cooling operating-condition time series;
- EN 16798-15 cooling storage time series and aggregate plots;
- EN 16798-13 cooling generation time series and bin plots;
- EN 14511 / EN 14825 rating and part-load performance plots;
- allocated heat-pump electricity time series;
- monthly demand, electricity, SPF and SEER summaries;
- bin-method energy balance plots;
- bin COP/EER, capacity and runtime plots;
- an annual energy-flow Sankey diagram.

The default calculation path is the full chain:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path full
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path full
```

To exercise the newly added generator and renewable branches:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path full-renewables
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path boiler-full
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path chp-full
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path district-full
```

The same choices can be made explicitly with
`--heating-generation-method en15316-4-2|en15316-4-1|en15316-4-4|en15316-4-5`
and `--renewable-method en15316-4-3-4-6`.

To keep the full subsystem chain but use the previous synthetic product maps
without EN 14825 part-load correction, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path full --performance-data-method simple
```

To combine the direct ISO52016/DHW path with the new EN 14511/EN 14825 product
performance stage, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path simple --performance-data-method en14511-14825
```

By default, the full path also runs EN 12831-3 DHW system sizing and passes the
selected DHW storage volume and design flow into EN 15316-5 and EN 15316-3. To
keep the earlier fixed DHW storage/distribution assumptions, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --dhw-design-method simple
```

The EN 12831-3 sizing target can be changed with `--dhw-sizing-mode check`,
`--dhw-sizing-mode size_storage`, `--dhw-sizing-mode size_power` or
`--dhw-sizing-mode auto`.

To run the previous detailed path with EN 15316-2 and EN 15316-3 but without EN 15316-5 storage, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path emission-distribution
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path emission-distribution
```

To run only EN 15316-5 storage between direct ISO52016/DHW loads and the heat pump, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path storage-only
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path storage-only
```

To bypass only the EN 16798-15 cooling storage module while retaining EN 15316-2, EN 15316-3, EN 15316-5 and EN 16798-13, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path no-cooling-storage
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path no-cooling-storage
```

To retain the earlier reversible heat-pump cooling branch instead of EN 16798-13, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path heat-pump-cooling
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path heat-pump-cooling
```

To run EN 15316-2 emission effects but bypass EN 15316-3 distribution and EN 15316-5 storage, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path emission-only
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path emission-only
```

To reproduce the earlier simple calculation without EN 15316-2, EN 15316-3, EN 15316-5 or EN 16798 effects, use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --calculation-path simple
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --calculation-path simple
```

Simple mode writes to `examples/outputs/heat_pump_15316_4_2_<scenario>_simple`
unless `--output-dir` is specified.

By default the script uses PVGIS weather for the selected scenario, so it needs internet access. To run with a local EPW file instead:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --weather-source epw --path-weather-file path/to/weather.epw
```

## Italian STREPIN Archetypes **(New)**

The STREPIN workbook in `examples/strepin_archetypes/` can be converted into
pyBuildingEnergy BUI cases and simulated with the European-standard generator
chain. Package IDs reproduce the workbook's pre/post cases:

```bash
python examples/strepin_archetypes/run_italian_strepin_engine.py --archetypes all --packages P00 --system-chain en-standards
python examples/strepin_archetypes/run_italian_strepin_engine.py --archetypes RMF_E1_B --packages P00,P09 --system-chain en-standards --write-hourly
python examples/strepin_archetypes/run_italian_strepin_engine.py --archetypes all --packages all --system-chain en-standards
```

Arbitrary combinations of the workbook measure levels can also be simulated
without adding a new workbook package:

```bash
python examples/strepin_archetypes/run_italian_strepin_engine.py --archetypes RMF_E1_B --measure-levels wall=2,roof_floor=2,windows=2,heating=3,pv=2 --system-chain en-standards
```

Additional renovation overlays that are considered in the European EPB suite
but are not explicit STREPIN workbook package levels can be layered on top:

```bash
python examples/strepin_archetypes/run_italian_strepin_engine.py --archetypes RMF_E1_B --measure-levels wall=2,roof_floor=2,windows=2,heating=3,pv=2 --extra-measures lighting-controls,ventilation-hrv,bacs-b,solar-thermal,battery-5 --system-chain en-standards
```

The STREPIN runner now reports EN ISO 52000-1 delivered/exported primary
energy, EN 15193-1 lighting, EN 16798-5/7 ventilation and heat recovery,
EN 15232-1 / EN ISO 52120-1 BACS factors, EN 15459-1 global cost, and deeper
EN 15316-4 options for biomass, CHP and district heat where selected.

Climate zone B now defaults to `examples/Palermo_PVGIS_TMY.epw`, downloaded from
PVGIS and normalized for `pvlib.read_epw`. Open
[Italian STREPIN Archetype Engine Audit](docs/strepin_archetype_engine_audit.html)
for the data and workflow crosswalk.

The script checks that the ISO52016 run produces both heating and cooling demand and that DHW demand is non-zero before running the heat-pump calculation.

## Primary Energy - Heating System **(New)**

The EN 15316 series covers the calculation method for system energy requirements and system efficiencies. This family of standards is an integral part of the EPB set and covers:

## EN 15316 Modular Structure **(New)**

- [x] EN 12831-3: Domestic hot-water energy needs and DHW design heat-load/sizing
- [x] EN 15316-1: General and expression of energy performance (Modules M3-1, M3-4, M3-9, M8-1, M8-4)
- [x] EN 15316-2: Emission systems (heating and cooling)
- [x] EN 15316-3: Distribution systems (DHW, heating, cooling)
- [x] EN 15316-4-X: Heat generation systems:
  - [x] 4-1: Combustion boilers
  - [x] 4-2: Heat pumps
  - [x] 4-3: Solar thermal and photovoltaic systems
  - [x] 4-4: Cogeneration systems
  - [x] 4-5: District heating
  - [x] Biomass/solid-fuel default wrapper for EN 15316-4 generation workflows
- [x] EN 15316-5: Storage systems

## Cross-Cutting EPB Accounting **(New)**

- [x] EN ISO 52000-1: delivered/exported energy and primary-energy accounting
- [x] EN 15193-1: lighting energy and LENI-style indicators
- [x] EN 16798-5/7: mechanical ventilation, heat recovery and fan electricity
- [x] EN 15232-1 / EN ISO 52120-1: BACS service factors
- [x] EN 15459-1: global cost and annualized cost calculation

## EN 16798 Cooling Modular Structure **(New)**

- [x] EN 16798-9: Cooling systems, operating conditions and M4-1/M4-4 handoff
- [x] EN 16798-13: Cooling generation, compression systems
- [x] EN 16798-15: Cooling storage
- [ ] EN 16798-13 non-compression cooling generators, such as absorption, adsorption, desiccant or evaporative cooling

## Heat Pump Product Rating Structure **(New)**

- [x] EN 14511-1: Heat-pump and chiller rating terms, including COP and EER definitions
- [x] EN 14511-2: Rating and application-condition temperature points for air-to-water heat pumps and chillers
- [x] EN 14825: Part-load capacity-ratio and degradation-coefficient correction for heating COP and cooling EER

For space heating, applicable standards include EN 15316-1, EN 15316-2-1, EN 15316-2-3 and the appropriate parts of EN 15316-4 depending on the system type, including losses and control aspects.

## Single zone and Multiple Zones **(New)**
# EN ISO 52016 - Multi-zone Calculation and Adjacent Zones

**EN ISO 52016 defines that:**  
The calculation now allows the definition of several **thermal** and **non-thermal** zones adjacent to the considered zone.


**External Adjacent - Unheated Zone**: It is possible to define an **unheated adjacent zone** in contact with the considered thermal zone.  
The length of the separating wall may be **entirely** or **partially** connected to the considered zone.  

The calculation involves:
1. Determining the **internal temperature** of the non-thermal zone.
2. Evaluating the **heat exchange** with the thermal zone.


**External Adjacent - Heated Zone**: In this case, the wall between the two zones is considered **adiabatic** (no heat exchange).
**Adjusted Coefficient**: To account for the **different temperatures** between zones (e.g., thermal and non-thermal), an **adjusted coefficient** is calculated.


### Assumptions and Simplifications
The standard defines various assumptions specified in section *6.5.3 - Assumptions and specific conditions*.  
In general, it aims to **simplify the zoning** approach by reducing the number of zones to a minimum (ISO EN 52016-2:2018).  

It also emphasizes that:

> *A multi-zone calculation with interactions between the zones requires significant and often arbitrary input data (on transmission properties and air flow direction and size).  
> It can also lead to other technical and procedural complications that add uncertainties to the results.  
> A further complication can be the involvement of different heating, cooling and ventilation systems for different zones, which adds to the complexity and arbitrariness of the input and modelling.*  

**Key Remark**: **Therefore, the benefits of calculations with thermally coupled zones can be smaller than the drawbacks.**

---

## EN 16798-7 & 16798-1 - Natural ventilation and profiles **(New)**

Compute the ventilation heat transfer coefficient [W·K⁻¹] of the thermal zone either: 

- from natural ventilation (ISO 16798-7:2017, single-sided airing via windows, wind/stack), or 
- from occupancy-driven flow (simplified volumetric rate per floor area).

For more detail refers to [natural ventilation](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_ventilation/).

Due to the need to have profiles of occupancy and consumption of buildings for some uses, tables of profiles useful for evaluating, occupancy, lights, heating, cooling, internal gains have been implemented.
These tables are provided by ANNEX A of ISO EN 16798-1. 
In the tool they are available here: [Table](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/source/table_iso_16798_1.py)

## Input Quality check  **(New)**

The data provided before being used for the simulation are processed and evaluated to be considered fit for the simulation. This process includes a series of checks that allow to identify any potential errors. 
For more details refers to [Input Quality check](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_input_check/).

## Limitations

The library is developed with the intent of demonstrating specific elements of calculation procedures in the relevant standards. It is not intended to replace the regulations but to complement them, as the latter are essential for understanding the calculation. This library is meant to be used for demonstration and testing purposes and is therefore provided as open source, without protection against misuse or inappropriate use.

The information and views set out in this document are those of the authors and do not necessarily reflect the official opinion of the European Union.

## Getting Started

Install the latest version of the library:

```bash
pip install pybuildingenergy
```

## Building - System Inputs

- For building inputs refer to [Building Inputs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_input/)
- For heating system inputs (EN 15316-1) refer to [Heating System Input](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_15316_input/)


## Example

**New examples will follow soon...**

## Contributing and Support

**Bug reports / Questions**  
If you encounter a bug, please create an issue detailing it. Provide steps to reproduce and a code snippet if possible.

**Code contributions**  
We welcome and appreciate contributions! Every contribution, no matter how small, makes a difference.

## License

- Free software: BSD 3-Clause License  
- Documentation: [pyBuildingEnergy Docs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/)

## Author

- [Daniele Antonucci](https://www.eurac.edu/it/people/daniele-antonucci)
- [Ulrich Filippi Oberegger](https://www.eurac.edu/it/people/ulrich-filippi)
- [Olga Somova](https://www.eurac.edu/it/people/olga-somova)

## Acknowledgment

This work was carried out within European projects:
- **Infinite** - EU Horizon 2020 (grant agreement No. 958397)
- **Moderate** - Horizon Europe (grant agreement No. 101069834)
- **BREEZE** - Building Renovation Efforts for Zero Emission Buildings, co-funded by the European Union LIFE programme (grant agreement No. 101215197)

The DHW calculation was developed with data and methods from the EPB Center spreadsheet.

## References

- [EPB Center - Energy Performance of Buildings Directive (EPBD)](https://epb.center/epb-standards/the-energy-performance-of-buildings-directive-epbd/)
- [REHVA Journal - EN ISO 52000 family of standards](https://www.rehva.eu/rehva-journal/chapter/the-new-en-iso-52000-family-of-standards-to-assess-the-energy-performance-of-buildings-put-in-practice)
- [European Commission - Energy Performance of Buildings Directive](https://energy.ec.europa.eu/topics/energy-efficiency/energy-performance-buildings/energy-performance-buildings-directive_en)
- Directive (EU) 2024/1275 - Official Journal of the EU, May 8, 2024
- EN ISO 52010-1:2018 - External climatic conditions  
- EN ISO 52016-1:2018 - Energy needs for heating and cooling  
- EN ISO 52016-2:2018 - Explanation and justification of ISO 52016-1 and ISO 52017-1
- EN 12831-3:2017 - DHW systems heat load and characterization  
- EN 14511-1:2022 - Air conditioners, liquid chilling packages and heat pumps: terms and definitions
- EN 14511-2:2022 - Air conditioners, liquid chilling packages and heat pumps: test conditions
- EN 14825:2022 - Seasonal performance and part-load conditions for air conditioners, chillers and heat pumps
- EN 15316-1:2018 - System energy requirements and efficiencies  
- EN 15316-2:2017 - Space emission systems
- EN 15316-3:2017 - Distribution systems
- EN 15316-4-1:2017 - Combustion generation systems
- EN 15316-4-2:2008 - Heat-pump generation systems
- EN 15316-4-3 / EN 15316-4-6 - Thermal solar systems and photovoltaic systems
- EN 15316-4-4:2007/2008 - Cogeneration systems
- EN 15316-4-5:2017 - District heating and cooling
- EN 15316-5:2017 - Storage systems
- EN 16798-9:2017 - Cooling systems
- EN 16798-13:2017 - Cooling generation
- EN 16798-15:2017 - Cooling storage
- EN 16798-7 & 16798-1 - Ventilation standards
