# pyBuildingEnergy

![pyBuildingEnergy Logo](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/Logo_pyBuild.png)

## Citation


*Please cite us if you use this library*: 
[![DOI](https://zenodo.org/badge/761715706.svg)](https://zenodo.org/doi/10.5281/zenodo.10887919)

> [!IMPORTANT]
> **Validation of the ISO 52016-1 hourly engine (October 2026)**
>
> **ANSI/ASHRAE Standard 140-2023, Section 7: both Thermal Fabric test groups pass the Annex A3 criteria (Table A3-14)** —
> **Low Mass 20/21 ranges met (18 required), High Mass 17/19 (17 required), cases 600–695 and 900–995.**
> Scripts: [`examples/ashrae140_bestest/`](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/tree/master/examples/ashrae140_bestest/) · reports: [`examples/ashrae140_bestest/reports/index.html`](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/examples/ashrae140_bestest/reports/index.html)
>
> **EN ISO 52016-1:2017, Section 7.2 verification cases (600, 640, 900, 940, 600FF, 900FF): 7 of 8 annual heating/cooling needs within ±10% of Tables 28–29, annual mean operative temperature within 0.4 K in all six cases**
> (ISO 52016-1 defines no pass/fail tolerance; the one value above ±10% is Case 640 cooling, +10.3%).
> Scripts: [`examples/iso52016_bestest/`](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/tree/master/examples/iso52016_bestest/) · reports: [`examples/iso52016_bestest/reports/index.html`](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/examples/iso52016_bestest/reports/index.html)
>
> **Some sources are omitted on purpose: we are not authorized to share parts, tables or limits of these standards.**
> The ASHRAE 140 acceptance-range values (ANSI/ASHRAE Standard 140-2023, Annex A3, Tables A3-1 to A3-4) and the ISO 52016-1
> reference results (EN ISO 52016-1:2017, clause 7.2.4, Tables 28–34) are not in the repository. **Owning the standards is
> enough to see and check everything locally** — see [Sources not distributed](#sources-not-distributed).

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

**pyBuildingEnergy** aims to provide an assessment of building performance both in terms of energy and comfort. The building energy needs are calculated with ISO 52016-1 (hourly method, single zone and multizone), validated against ANSI/ASHRAE Standard 140-2023 and the ISO 52016-1 verification cases (see [Validation](#validation---ashrae-140-2023-and-iso-52016-1-new)). The EPB system chain (EN 15316, EN 16798, EN 12831-3, EN 14511/14825, EN ISO 52000-1 and related standards) is described in the sections below.

The actual calculation methods for the assessment of building performance are the following:

- [x] the (sensible) energy need for heating and cooling, based on hourly or monthly calculations;
- [x] the latent energy need for (de-)humidification, based on hourly or monthly calculations;
- [x] the internal temperature, based on hourly calculations;
- [x] the sensible heating and cooling load, based on hourly calculations;
- [x] the moisture and latent heat load for (de-)humidification, based on hourly calculations;
- [x] the design sensible heating or cooling load and design latent heat load using an hourly calculation interval;
- [ ] the conditions of the supply air to provide the necessary humidification and dehumidification.

The calculation methods can be used for residential or non-residential buildings, or a part of it, referred to as "the building" or the "assessed object".

ISO 52016-1:2018 also contains specifications for the assessment of thermal zones in the building or in the part of a building. The calculations are performed per thermal zone. In the calculations, the thermal zones can be assumed to be thermally coupled or not. ISO 52016-1:2018 is applicable to buildings at the design stage, to new buildings after construction and to existing buildings in the use phase.

-- 

## Validation - ASHRAE 140-2023 And ISO 52016-1 **(New)**

Two independent validation harnesses run the ISO 52016-1 engine on the
BESTEST test cell (8 m x 6 m x 2.7 m, 12 m2 of glazing, Denver climate) and
write one HTML report per case.

**ANSI/ASHRAE Standard 140-2023, Section 7** — building thermal envelope and
fabric load tests, with the inputs of the 2023 specification and its
accompanying files (raised floor, 2023 glazing, altitude-corrected air flows,
Table 7-7/7-9 surface coefficients, overhangs and side fins per Figures 7-3 to
7-6, two-zone sunspace Case 960 on the multizone engine).

| Annex A3 test group | Ranges met | Required (Table A3-14) | Result |
|---|---:|---:|---|
| Thermal Fabric Low Mass (cases 600–695) | 20 of 21 | 18 | **pass** |
| Thermal Fabric High Mass (cases 900–995) | 17 of 19 | 17 | **pass** (no margin) |

Ranges not met (outside the ASHRAE 140 acceptance range; 2020 edition,
Addendum b): Case 900 annual cooling (2.944 MWh), and the 650−600 and 920−900
cooling differences. The acceptance-range values are not reproduced here: they
are in Tables A3-1 to A3-4 of Annex A3 (ANSI/ASHRAE Standard 140-2023,
incorporating Addendum b to 140-2020). Documented deviations: the interior solar
distribution fractions of Table 7-13 are not applied (the engine distributes
transmitted solar by area) and ISO 52016-1 Annex F Method 1 shades direct
radiation only.

```bash
PYTHONPATH=src python examples/ashrae140_bestest/run_comparison.py     # Annex A3 check of all cases
PYTHONPATH=src python examples/ashrae140_bestest/generate_reports.py   # reports/index.html
```

**EN ISO 52016-1:2017, Section 7.2** — the six verification cases of Table 27
with the standard's own adapted inputs (operative-temperature control, floor
to outdoor air, 0.41 ach, F_w = 0.90) and its BESTEST climate file.

| Case | Annual heating vs Table 28 | Annual cooling vs Table 29 | Annual mean T_op vs Table 30 |
|---|---:|---:|---:|
| 600 | +9.5% | +8.5% | −0.09 K |
| 640 | +9.4% | +10.3% | −0.36 K |
| 900 | +6.2% | +6.4% | +0.01 K |
| 940 | −1.9% | +5.3% | −0.17 K |
| 600FF | — | — | +0.11 K |
| 900FF | — | — | +0.14 K |

ISO 52016-1 defines no acceptance tolerance; the tests use ±10% on the annual
needs. Case 900 is compared with the corrected Tables 28/29 values (the
printed heating and cooling columns are swapped). The ISO climate
spreadsheet is ISO copyright content and is not shipped: place it in
`Standards/` to reproduce these numbers (otherwise a TMY3 approximation is used).

```bash
PYTHONPATH=src python examples/iso52016_bestest/run_comparison.py
PYTHONPATH=src python examples/iso52016_bestest/generate_reports.py    # reports/index.html
```

Details, inputs and remaining gaps: [ASHRAE 140 README](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/examples/ashrae140_bestest/README.md)
and [ISO 52016-1 README](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/examples/iso52016_bestest/README.md).

### Sources not distributed

The following content is copyright ASHRAE / ISO and is **not included** in
this repository, because we are not authorized to share parts, tables or
limits of the standards:

| Content | Where to find it in the standard | Local file (git-ignored) |
|---|---|---|
| ASHRAE 140 acceptance ranges, low-mass heating / sensible cooling | ANSI/ASHRAE Standard 140-2023, Normative Annex A3, Tables A3-1 and A3-2 | `Standards/local_reference_data/ashrae140_2023_annex_a3.json` |
| ASHRAE 140 acceptance ranges, high-mass heating / sensible cooling | Annex A3, Tables A3-3 and A3-4 | same file |
| ASHRAE 140 group pass criteria | Annex A3, Table A3-14 (counts quoted above) | — |
| ISO 52016-1 reference results (monthly/annual needs, operative temperatures, peaks, hourly January 4) | EN ISO 52016-1:2017, clause 7.2.4, Tables 28–34 | `Standards/local_reference_data/iso52016_1_2017_tables_28_34.json` |
| ISO 52016-1 BESTEST climate data | EN ISO 52016-1:2017, clause 7.2.2.12 (accompanying spreadsheet) | `Standards/ISO_52016_1_BESTEST_ClimData_2016.08.24.xls` |

Annex A3 was introduced by Addendum b to ANSI/ASHRAE Standard 140-2020 and is
incorporated in the 2023 edition. **Anyone who owns ANSI/ASHRAE Standard 140
(and EN ISO 52016-1 for the ISO cases) can reproduce every check:** copy the
values into the local files following the templates
[`annex_a3_ranges.template.json`](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/examples/ashrae140_bestest/annex_a3_ranges.template.json) and
[`iso_reference_tables.template.json`](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/examples/iso52016_bestest/iso_reference_tables.template.json),
which name the table each value comes from. Without them the cases still run,
range and reference checks report "not available" and the related tests are
skipped. Reports and console output never print the range or reference
values; the published reports cite the table of the standard instead.

## ISO 52016-1 Engine Options And Fixes **(New)**

Changes to `ISO52016.Temperature_and_Energy_needs_calculation` and the
multizone engine made during the validation work:

- `hvac_control_variable="operative" | "air"`: the ideal heating/cooling
  control acts on the operative temperature (ISO 52016-1 default) or on the
  zone air temperature (EnergyPlus-style thermostat, as required by ASHRAE 140).
- `construction_class` can be set per surface (ISO 52016-1 Table B.13); the
  building-level value remains the fallback.
- Windows: the transmitted solar gain applies the correction factor F_w of
  Formula E.3 (`solar_transmittance_correction_factor`, default 0.90) and the
  window's own `frame_area_fraction`; transparent elements have solar
  absorptance 0 at the external node (clause 6.5.7.3).
- `external_convection_model` (`"doe2"`, `"mowitt"`, `"blast"`,
  `"simplecombined"`) is now applied in the single-zone engine too (it was
  accepted but had no effect); `internal_convection_model="tarp"`,
  `external_radiation_model="dynamic"` and `sky_temperature_model` remain
  available. The default `"table"` models are unchanged.
- Side fins (ISO 52016-1 Annex F, F.5/F.6 combined with F.19–F.22) per window:
  `"side_fins": {"left": {"depth": 1.0, "distance": 0.0}, "right": {...}}`,
  in addition to overhangs (F.4) and Annex F obstacles.
- `simulate_envelope_multizone_free_floating(..., c_int_per_A_us=10000)`: the
  zone air/furniture heat capacity is configurable, as in the single-zone engine.

## Weather Data

The tool can use weather data coming from 2 main sources:

- PVGIS API ([link](https://re.jrc.ec.europa.eu/pvg_tools/en/)) - PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM
- `.epw` file from [Ladybug Tools EPWMap](https://www.ladybug.tools/epwmap/)

More details in the example folder.

## Domestic Hot Water - DHW

- [x] Calculation of volume and energy need for domestic hot water according to EN 12831-3.
- [x] Assessment of DHW system design heat load with EN 12831-3 storage switch points, time lag, effective reheating power and supply-curve sizing.

For the audit trail between the standard, code and outputs, open
[DHW EN 12831-3 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/dhw_12831_3_audit.html).

## Space Emission Systems - EN 15316-2 **(New)**

`EmissionSystemCalculator` evaluates space-heating and water-based space-cooling
emission effects before generation. It can be used to account for emitter and
control effects, equivalent internal-temperature changes, embedded emitter
losses, emission auxiliary electricity and annual emission expenditure factors.

For an audit trail between the standard, code and output files, open
[Emission EN 15316-2 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/emission_15316_2_audit.html).

**Auxiliary energy.** By default the emission auxiliary energy is the fans only
(EN 15316-2 eq. 27-28). Control-device power (`control_power_W`, `control_count`) is
included only when the system is an electrical control system with auxiliary energy
(Table A.12), by setting `electric_control_aux: true`.

**Undersized emitters (EN 15316-1 eq. 16).** If the emitter cannot deliver the expected
output, the deficit is backup energy when `backup_available: true` (`Q_*_em_back_out_kWh`).
Otherwise it is reported as unmet load (`Q_*_em_unmet_kWh`) and a warning is raised when it
exceeds `undersize_tolerance` (default 5 % of the expected output).
With `backup_available: true` the backup energy is added to the heating-system demand passed to
distribution and generation (backup input taken equal to its output; backup losses not modelled).
In undersized hours the literal EN 15316-2 eq. 22 gives a negative `Q_*_em_ls` (the capped
emitter output is below the expected output); the energy identities eq. 22-23 are kept as written.

**Emission output at the modified set point.** `simulate_config` (emission-only path) recalculates
the building need with ISO 52016 at the equivalent set points, as EN 15316-2 requires. Set
`options.emission_inc_method: "approximate"` to use the temperature-ratio approximation instead:
it is faster (about half the time, since ISO 52016 runs once) but the results are approximate,
and a notification is raised. The difference is material for cooling (the example gives about
790 kWh with recalculation and 401 kWh with the approximation). The standalone
`EmissionSystemCalculator` has no building, so it can only use the approximation unless
`Q_*_em_out_inc_kWh` is supplied.

**Fan auxiliary energy (EN 15316-2 eq. 14).** `W_fan = n_fan · P_fan · t_h,rL / 1000`. Provide the
system operation time per step as `t_h_rl` (hours, from the EN 15316-1 operating schedule). If it is
not given, the run time is estimated as the load factor (emitted energy / nominal power) times the
time step, instead of assuming the fan runs the full hour whenever there is output.

**Sign convention for cooling (note on the standard).** The implementation
uses `theta_int,inc = theta_int,ini + SUM(delta)` for both heating and cooling,
as in the hourly procedure (EN 15316-2:2017, eq. 16-17), with cooling deltas
entered as negative values (Table B.11). The standard is not fully consistent
on this point: the general description (section 5.2, eq. 1a/1b) gives
`theta_C,int,inc = theta_C,int,ini - delta`. With the negative Table B.11
values the two formulas give opposite results, so the cooling sign should be
checked against the national annex or the standard's clarification before
using cooling results for compliance.

## Radiator Product Ratings - EN 442:2014

`EN442RadiatorCharacteristic` consumes certified radiator or non-fan-assisted
convector catalogue data (`phi_50`, exponent `n`, optional `phi_30`, maximum
operating temperature and optional hydraulic data). `HeatingSystemCalculator`
uses the characteristic in its EN 15316-1 C.2-C.5 circuit calculations and
reports requested, available, delivered and unmet emitter output.

The older `nominal_power` + `TB14` path remains the default and retains its
existing results for generic radiators, floor heating and fan coils. EN 442
product rating data and EN 15316 circuit design data are separate: for example,
an EN 442 `phi_50` rating does not force the heating circuit to use the 10 K
water temperature difference of the product test.

See [examples/en442_radiator_example.py](examples/en442_radiator_example.py)
for a runnable product-rating and operating-limit example.

## Water-Based Distribution Systems - EN 15316-3 **(New)**

`DistributionSystemCalculator` evaluates water-based distribution systems for
space heating, space cooling and DHW. It calculates pipe thermal losses,
recoverable distribution losses, pump auxiliary electricity, recoverable pump
heat and recovered pump heat in the fluid before the generator calculation.

For the audit trail, open
[Distribution EN 15316-3 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/distribution_15316_3_audit.html).

## Heating And DHW Storage Systems - EN 15316-5 **(New)**

`StorageSystemCalculator` evaluates single-volume Method B storage systems for
space heating and DHW. It calculates storage standing losses from product
standby-loss data or a direct heat-loss coefficient, storage charging pump
auxiliary electricity, recoverable storage losses and heat recovered in the
medium before the generator calculation.

For the audit trail, open
[Storage EN 15316-5 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/storage_15316_5_audit.html).

## Heat Pump Generation - EN 15316-4-2 **(New)**

`HeatPumpSystemCalculator` evaluates heat-pump generation for:

- space heating,
- domestic hot water (DHW),
- and an optional simplified reversible space-cooling branch with an EER map.

The heating and DHW calculation follows EN 15316-4-2:2017's Method A (clause 6, hourly/bin) structure: outdoor/source temperature bins, product heating capacity and COP maps (EN 14511-series, Path A), part-load performance per Eq.(14)-(30), runtime/capacity checks, auxiliary energy, optional simplified storage losses, backup energy and SPF outputs. The runnable examples use the EN 16798 cooling modules by default; the older heat-pump EER cooling branch is still available with `--calculation-path heat-pump-cooling`.

For a clause-by-clause audit trail between the standard, the implementation and the output files, open [Heat Pump EN 15316-4-2 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/heat_pump_15316_4_2_audit.html).

## Additional Generation And Renewable Modules - EN 15316-4-1, EN 15316-4-4, EN 15316-4-5, EN 15316-4-3/4-6 **(New)**

The Athens and Bolzano system workflow can now switch the heating/DHW generator
away from the default heat pump:

- `BoilerGeneratorCalculator` implements EN 15316-4-1 (combustion generators - gas/oil boilers and biomass) with the standard's load-interpolated efficiency/loss method, temperature correction, standby losses and auxiliaries.
- `CogenerationSystemCalculator` implements an EN 15316-4-4 thermal-led CHP boundary with fuel input, useful electricity, self-consumption/export and backup heat.
- `DistrictEnergySystemCalculator` implements an EN 15316-4-5 district heating/cooling substation boundary.
- `RenewableEnergySystemCalculator` implements the EN 15316-4-3/4-6 solar thermal and PV accounting branch used by the examples and STREPIN runner.

Audit pages:
[Generation EN 15316-4-1/4-4/4-5](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/generation_15316_4_1_4_4_4_5_audit.html) and
[Renewables EN 15316-4-3/4-6](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/renewables_15316_4_3_4_6_audit.html).

## Heat Pump Product Performance - EN 14511 And EN 14825 **(New)**

`HeatPumpPerformanceDataCalculator` normalizes heat-pump rating data from EN
14511-style capacity/COP/EER points and calculates EN 14825 part-load
inspection values. The examples use this path by default and apply the EN 14825
water-based part-load correction to heating/DHW COP and EN 16798-13 cooling EER.

For the audit trail between the standards, code and outputs, open
[EN 14511 / EN 14825 Product Performance Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/performance_14511_14825_audit.html).

## Cooling System Modules - EN 16798-9, EN 16798-15 And EN 16798-13 **(New)**

The heat-pump examples now treat space cooling with the cooling-side EN 16798 standards:

- `CoolingSystemCalculator` implements EN 16798-9 operating conditions and the handoff of cooling requests to storage or generation.
- `CoolingStorageSystemCalculator` implements EN 16798-15 chilled-water storage heat gains and pump auxiliary energy.
- `CoolingGenerationSystemCalculator` implements EN 16798-13 compression cooling generation using cooling capacity/EER product maps or nominal EER fallback data.

For the audit trail between the standards, code and outputs, open [Cooling EN 16798 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/cooling_16798_audit.html).

For the whole end-to-end workflow that interconnects EN ISO 52016, EN 12831-3,
EN 15316, EN 16798, EN 14511 and EN 14825 in the Athens and Bolzano examples,
open [Heat-Pump Example Simulation Workflow Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/simulation_workflow_audit.html).

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
[Italian STREPIN Archetype Engine Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/strepin_archetype_engine_audit.html)
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

### EN ISO 52016 - Multi-zone Calculation and Adjacent Zones

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

## EN 16798-5-1 - Mechanical Ventilation AHU **(New)**

A sensible, balanced-airflow air-handling-unit model based on EN 16798-5-1 can
be configured as a `mechanical_supply` ventilation component. It computes heat
recovery, bypass and frost behaviour, capacity-limited heating/cooling coils
and fan electricity per timestep, and couples into the ISO 52016-1 zone
balance as a supply-air stream. Existing configurations are unchanged unless
the component is configured.

```python
"building_parameters": {
    "ventilation": {
        "components": [
            {   # envelope leakage: no profile, always active
                "name": "infiltration",
                "ventilation_type": "constant_ach",
                "air_changes_per_hour": 0.15,
            },
            {   # balanced AHU with sensible heat recovery
                "name": "ahu",
                "ventilation_type": "mechanical_supply",
                "supply_flow_m3_h": 3600.0,
                "sensible_heat_recovery_efficiency": 0.784,
                "supply_temperature_setpoint_c": 18.0,
                "heating_coil_max_power_w": 15000.0,
                "supply_fan_specific_power_w_per_m3_s": 750.0,
                "extract_fan_specific_power_w_per_m3_s": 750.0,
                "profile": "ventilation_profile",  # flow fraction in [0, 1]
            },
        ],
    },
},
```

For the implemented / simplified / not-included scope, all configuration keys
and the hourly diagnostic columns, open
[Ventilation AHU EN 16798-5-1 Implementation Audit](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/docs/ventilation_ahu_audit.html).

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

### Run a simulation with progress and HTML reports

From the repository root (paths are relative, so the commands work on any machine):

```bash
# single zone (building + emission system)
python examples/thermal_report_example.py \
  --config examples/FH_Poland_DC_baseline_two_floors.json \
  --mode single-zone

# multizone (ground floor + first floor)
python examples/thermal_report_example.py \
  --config examples/FH_Poland_DC_multizone_ideal_envelope.json \
  --mode multizone
```

The engine shows a progress bar (tqdm) while it computes the hourly balance.
The reports are written to `examples/outputs/thermal_report/` (use `--output-dir <folder>` to change it):

- `thermal_report_single_zone.html` or `thermal_report_multizone.html`
- hourly, daily and monthly charts (ECharts): thermal need, operative temperature per zone, outdoor temperature and outdoor relative humidity

The Poland case JSON files are not shipped with the package: they must be placed in `examples/` (or the `--config` path must point to them).

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

### EPBD and context

- [EPB Center - Energy Performance of Buildings Directive (EPBD)](https://epb.center/epb-standards/the-energy-performance-of-buildings-directive-epbd/)
- [REHVA Journal - EN ISO 52000 family of standards](https://www.rehva.eu/rehva-journal/chapter/the-new-en-iso-52000-family-of-standards-to-assess-the-energy-performance-of-buildings-put-in-practice)
- [European Commission - Energy Performance of Buildings Directive](https://energy.ec.europa.eu/topics/energy-efficiency/energy-performance-buildings/energy-performance-buildings-directive_en)
- Directive (EU) 2024/1275 - Official Journal of the EU, May 8, 2024

### Building energy needs, climate and validation

- EN ISO 52000-1:2017 - Energy performance of buildings: overarching EPB assessment (delivered/exported and primary energy)
- EN ISO 52010-1:2017 - External climatic conditions: conversion of climatic data for energy calculations
- EN ISO 52016-1:2017 - Energy needs for heating and cooling, internal temperatures and sensible and latent heat loads (hourly method, Annex F shading, Annex G dynamic transparent elements, Section 7.2 verification cases)
- CEN ISO/TR 52016-2:2017 - Explanation and justification of ISO 52016-1 and ISO 52017-1
- ISO 13370:2017 - Heat transfer via the ground (ground model of the ISO 52016-1 engine)
- ISO 13789:2017 - Transmission and ventilation heat transfer coefficients (surface heat-transfer coefficients)
- ANSI/ASHRAE Standard 140-2023 - Method of Test for Evaluating Building Performance Simulation Software (Section 7 test cases; Annex A3 acceptance criteria, introduced by Addendum b to ANSI/ASHRAE Standard 140-2020)

### Domestic hot water

- EN 12831-3:2017 - Domestic hot water systems heat load and characterization of needs
- EN 806-3, EN 12897 and EN 13203-2 - referenced as optional input-data sources for DHW design and product data
- EN 89 - referenced for gas-fired storage water-heater standby losses in the EN 15316-4-1 generator

### System energy requirements - EN 15316 series

- EN 15316-1:2017 - System energy requirements and system efficiencies: general and expression of energy performance
- EN 15316-2:2017 - Space emission systems (heating and cooling)
- EN 15316-3:2017 - Space distribution systems (DHW, heating and cooling)
- EN 15316-4-1:2017 - Space heating and DHW generation systems, combustion systems (boilers, biomass)
- EN 15316-4-2:2017 - Space heating generation systems, heat-pump systems (Method A, clause 6)
- EN 15316-4-3 / EN 15316-4-6 - Thermal solar systems and photovoltaic systems
- EN 15316-4-4:2007/2008 - Cogeneration systems
- EN 15316-4-5:2017 - District heating and cooling
- EN 15316-5:2017 - Space heating and DHW storage systems

### Cooling, ventilation and indoor environment - EN 16798 series

- EN 16798-1:2019 - Indoor environmental input parameters (occupancy, lighting, appliance and setpoint profiles, Annex A)
- EN 16798-5-1:2017 - Calculation methods for energy requirements of ventilation and air-conditioning systems (air-handling unit model)
- EN 16798-7:2017 - Calculation methods for the determination of air flow rates in buildings including infiltration (natural ventilation)
- EN 16798-9:2017 - Calculation methods for energy requirements of cooling systems
- EN 16798-13:2017 - Calculation of cooling systems: generation
- EN 16798-15:2017 - Calculation of cooling systems: storage

### Heat-pump and chiller product performance

- EN 14511-1:2022 - Air conditioners, liquid chilling packages and heat pumps: terms and definitions
- EN 14511-2:2022 - Air conditioners, liquid chilling packages and heat pumps: test conditions
- EN 14825:2022 - Seasonal performance and part-load conditions for air conditioners, chillers and heat pumps

### Lighting, building automation and economics

- EN 15193-1:2017 - Energy requirements for lighting
- EN 15232-1:2017 / EN ISO 52120-1 - Impact of building automation, controls and building management (BACS factors)
- EN 15459-1:2017 - Economic evaluation procedure for energy systems in buildings (global cost)

### Validation data sources

- ANSI/ASHRAE Standard 140-2023 accompanying electronic files (weather file `725650TY.CSV`, output templates)
- [NREL BESTEST-GSR repository](https://github.com/NatLabRockies/BESTEST-GSR) - Denver TMY3 EPW used by the ASHRAE 140 harness (BSD-style license)
- [EPB Center - Collection of comments and suggestions on EN ISO 52016-1:2017](https://epb.center/support/documents/iso_52016-1_details/) - corrected Case 900 values of Tables 28/29 (comment no. 28)
- EN ISO 52016-1:2017 BESTEST climate data spreadsheet (clause 7.2.2.12), not distributed with this repository
