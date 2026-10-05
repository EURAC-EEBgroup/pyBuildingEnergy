# Multizone Free-Floating Simulation with pyBuildingEnergy

## Complete guide to `multizone_free_floating_example.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Requirements and Run Instructions](#2-requirements-and-run-instructions)
3. [Building Structure (`building_object`)](#3-building-structure-building_object)
   - 3.1 [Building metadata (`building`)](#31-building-metadata-building)
   - 3.2 [Simulation parameters (`building_parameters`)](#32-simulation-parameters-building_parameters)
   - 3.3 [Ventilation and infiltration model](#33-ventilation-and-infiltration-model)
   - 3.4 [Advanced simulation options (`simulation_options`)](#34-advanced-simulation-options-simulation_options)
   - 3.5 [Internal gains (`internal_gains`)](#35-internal-gains-internal_gains)
   - 3.6 [Thermal zones (`zones`)](#36-thermal-zones-zones)
   - 3.7 [Envelope surfaces (`building_surface`)](#37-envelope-surfaces-building_surface)
4. [Script execution flow](#4-script-execution-flow)
   - 4.1 [Opaque area normalization (IDF)](#41-opaque-area-normalization-idf)
   - 4.2 [V1 simulation - Fully Integrated](#42-v1-simulation--fully-integrated)
   - 4.3 [V2 simulation - Hybrid Iterative](#43-v2-simulation--hybrid-iterative)
   - 4.4 [Post-processing and output](#44-post-processing-and-output)
5. [All command-line options](#5-all-command-line-options)
6. [Ventilation and infiltration models](#6-ventilation-and-infiltration-models)
7. [Ground temperature models](#7-ground-temperature-models)
8. [External convection and radiation models](#8-external-convection-and-radiation-models)
9. [Summer night purge (`summer_night_purge`)](#9-summer-night-purge-summer_night_purge)
10. [Generated output files](#10-generated-output-files)
11. [Typical usage examples](#11-typical-usage-examples)

---

## 1. Overview

The script `multizone_free_floating_example.py` shows how to run an **annual energy simulation of a building with multiple thermal zones** using the `ISO52016` engine in `pyBuildingEnergy`.

The case study is a two-zone residential building (`Z1` and `Z2`) with opaque surfaces, windows, and a ground-contact floor. The script implements **two simulation methods** and compares them:

- **V1 - Fully Integrated**: simultaneous calculation of all zones in a single coupled simulation.
- **V2 - Hybrid Iterative**: an iterative calculation that alternates simulations for each zone and updates zone-to-zone coupling flows until convergence.

The operating mode is **free-floating**, meaning the HVAC system has essentially unlimited capacity (`1e9 W`) and the system follows the temperature setpoints without installed-power limits.

> **Note**: *free-floating* refers to the absence of HVAC power limits, not to the absence of thermal setpoints. Setpoints remain active, but there is no saturation of delivered heating or cooling power.

---

## 2. Requirements and Run Instructions

### Virtual environment

```bash
# Activate the virtual environment
source venv_pybuildingenergy/bin/activate

# Run the script with default settings
python examples/multizone_free_floating_example.py

# Show all available options
python examples/multizone_free_floating_example.py --help
```

### Required EPW file

By default the script looks for:

```text
examples/2020_Athens.epw
```

To use a different weather file:

```bash
python examples/multizone_free_floating_example.py --weather-path /path/to/myfile.epw
```

### Main dependencies

| Package | Role |
|---|---|
| `pandas` | Hourly and annual DataFrames |
| `numpy` | Vectorized calculations |
| `plotly` | Interactive HTML plots |
| `pvlib` | EPW reading and solar radiation |
| `pybuildingenergy` | ISO 52016 simulation engine |

---

## 3. Building Structure (`building_object`)

`building_object` is a Python dictionary that fully describes the building. It is defined directly in the script body (starting around line ~1007) and can be edited before launching the simulation.

### 3.1 Building metadata (`building`)

```python
"building": {
    "name": "TwoZoneBuilding",         # Identifier name
    "model_source": "idf",             # Model source: "idf" | "manual"
    "opaque_surface_area_mode": "net", # Opaque area mode: "net" | "gross" | "gross_from_idf"
    "latitude": 41.8,                  # Latitude [decimal degrees, north positive]
    "longitude": 12.58,                # Longitude [decimal degrees, east positive]
    "net_floor_area": 120.0,           # Total net floor area [m²]
    "exposed_perimeter": 44.0,         # Exposed perimeter [m] (for thermal bridges)
    "wall_thickness": 0.43,            # Wall thickness [m]
    "building_type_class": "Residential_apartment",  # Usage category
    "construction_class": "class_e"    # Construction class (for thermal inertia)
}
```

**`model_source`**: when set to `"idf"`, automatic normalization of gross opaque areas to net areas is enabled (subtracting the window area for the same zone and orientation).

**`opaque_surface_area_mode`**: controls how opaque surface areas are interpreted:
- `"net"` - the entered areas are already net (windows excluded)
- `"gross"` / `"gross_from_idf"` / `"gross_including_windows"` - the areas are gross; the script normalizes them automatically

### 3.2 Simulation parameters (`building_parameters`)

```python
"building_parameters": {
    "temperature_setpoints": {
        "heating_setpoint": 20.0,    # Heating setpoint [°C]
        "cooling_setpoint": 26.0,    # Cooling setpoint [°C]
        "heating_setback": -100,     # Heating setback temperature [°C] (disabled = -100)
        "cooling_setback": 100.0     # Cooling setback temperature [°C] (disabled = 100)
    },
    "system_capacities": {
        "heating_capacity": 1e9,     # Maximum heating power [W]
        "cooling_capacity": 1e9      # Maximum cooling power [W]
    },
    ...
}
```

> Setpoints defined in `building_parameters` act as **global defaults**. Each zone can override them locally in its own definition.

### 3.3 Ventilation and infiltration model

The `"ventilation"` block defines the physical model used to compute airflow rates and ventilation/infiltration losses.

```python
"ventilation": {
    "ventilation_type": "eplus_infiltration_ext_area",
    ...
}
```

The `ventilation_type` parameter selects the model. All available models are described in detail in [Section 6](#6-ventilation-and-infiltration-models).

### 3.4 Advanced simulation options (`simulation_options`)

```python
"simulation_options": {
    "internal_convection_model": "tarp",
    "external_convection_model": "doe2",
    "external_convection_h_min": 2.0,
    "external_radiation_model": "dynamic",
    "sky_temperature_model": "epw_ir",
    "external_emissivity_default": 0.9,
    "ground_temperature_model": "monthly",
    "ground_temperature_monthly": [9.867, 9.199, ..., 11.596]
}
```

All of these parameters are described in [Sections 7](#7-ground-temperature-models) and [8](#8-external-convection-and-radiation-models).

### 3.5 Internal gains (`internal_gains`)

Internal gains are defined as a list of heat sources. Each entry has:

```python
{
    "name": "occupants",         # "occupants" | "appliances" | "lighting"
    "full_load": 6.0,            # Maximum number of occupants (or maximum power [W/m²])
    "w_per_person": 120.0,       # [W/person] - only for "occupants"
    "weekday": [...],            # Weekday hourly profile (24 values in [0,1])
    "weekend": [...]             # Weekend hourly profile (24 values in [0,1])
}
```

These values in `building_parameters` are the **global defaults**. Each zone can have its own `internal_gains` list with specific values and separate profiles (`occupants_profile`, `appliances_profile`, `lighting_profile`).

### 3.6 Thermal zones (`zones`)

Each zone is a dictionary with the following keys:

```python
{
    "name": "Z1",                          # Unique zone identifier
    "net_floor_area": 80.0,                # Net floor area [m²]
    "building_type_class": "Residential_apartment",
    "heating_setpoint": 20.0,              # Overrides the global setpoint
    "cooling_setpoint": 26.0,
    "cooling_setback": 100.0,
    "summer_night_purge": { ... },         # Summer night purge configuration
    "internal_gains": [ ... ],             # Zone-specific internal gains
    "occupants_profile":  { "weekday": [...], "weekend": [...] },
    "appliances_profile": { "weekday": [...], "weekend": [...] },
    "lighting_profile":   { "weekday": [...], "weekend": [...] },
    "heating_profile":    { "weekday": [...], "weekend": [...] },
    "cooling_profile":    { "weekday": [...], "weekend": [...] },
    "ventilation_profile":{ "weekday": [...], "weekend": [...] }
}
```

All hourly profiles are vectors of 24 values between `0.0` and `1.0`, which multiply the corresponding load/setpoint/maximum flow rate for that hour.

The example building has two zones:

| Zone | Net area | Heating | Cooling | Characteristic |
|------|-----------|---------|---------|-----------------|
| `Z1` | 80 m²     | 20 °C   | 26 °C   | Main living/day-night zone |
| `Z2` | 40 m²     | 18 °C   | 26 °C   | Secondary zone (night) |

### 3.7 Envelope surfaces (`building_surface`)

Each surface is a dictionary with the following mandatory and optional keys:

```python
{
    "name": "ZoneA_Wall_S",                    # Unique identifier
    "type": "opaque",                          # "opaque" | "transparent"
    "boundary": "OUTDOORS",                    # "OUTDOORS" | "GROUND" | "INTERNAL"
    "zone": "Z1",                              # Associated zone
    "area": 30.0,                              # Area [m²]
    "u_value": 0.2891,                         # Thermal transmittance [W/(m²·K)]
    "thermal_capacity": 248700.0,              # Thermal capacity [J/K]
    "solar_absorptance": 0.6,                  # Solar absorptance (opaque surfaces)
    "orientation": {
        "azimuth": 180.0,                      # Azimuth [°, 0 = North, 90 = East, 180 = South, 270 = West]
        "tilt": 90.0                           # Tilt [°, 0 = horizontal, 90 = vertical]
    },
    "ISO52016_orientation_string": "SV",       # ISO 52016 orientation tag
    "sky_view_factor": 0.5                     # Sky view factor [0-1]
}
```

#### ISO 52016 orientation tags (`ISO52016_orientation_string`)

| Tag | Meaning |
|-----|---------|
| `HR` | Horizontal Roof |
| `HF` | Horizontal Floor |
| `SV` | Vertical - South |
| `NV` | Vertical - North |
| `EV` | Vertical - East |
| `WV` | Vertical - West |
| `GR` | Ground |

#### Additional keys for transparent surfaces

```python
{
    "type": "transparent",
    "g_value": 0.6,               # Solar factor (total solar transmittance) [-]
    "frame_area_fraction": 0.0,   # Frame area fraction [-]
    "height": 1.5,                # Window height [m]
    "width": 3.0,                 # Window width [m]
    "parapet": 0.9,               # Parapet height [m]
    "shading": False              # External shading enabled [bool]
}
```

#### Additional keys for internal surfaces (`boundary: "INTERNAL"`)

```python
{
    "boundary": "INTERNAL",
    "adjacent_zone": "Z1",
    "convective_heat_transfer_coefficient_internal": 2.5,   # [W/(m²·K)]
    "radiative_heat_transfer_coefficient_internal": 5.13,
    "convective_heat_transfer_coefficient_external": 2.5,
    "radiative_heat_transfer_coefficient_external": 5.13
}
```

---

## 4. Script execution flow

```text
Start
  |
  |- Parse CLI arguments
  |- Resolve EPW path and output folder
  |- Normalize IDF opaque areas (gross -> net)
  |- Configure ground temperature
  |- Configure summer night purge
  |- Configure infiltration/ventilation
  |
  |- [V1] Annual fully integrated simulation
  |    |- ISO52016.Temperature_and_Energy_needs_calculation_multizone(...)
  |
  |- [V2] Annual hybrid iterative simulation  (skippable with --skip-hybrid)
  |    |- ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(...)
  |
  |- Compute ground heat exchanges
  |- Save CSV files (hourly, annual, timing)
  |- Generate HTML plots (temperatures, monthly consumption, ground exchange)
  |- Generate comparative V1 vs V2 report
  |- Compute and save Sankey diagrams by zone  (skippable with --skip-sankey)
```

### 4.1 Opaque area normalization (IDF)

When `model_source = "idf"` and `opaque_surface_area_mode` is `"gross"` or equivalent, the script automatically applies `_normalize_idf_gross_opaque_areas`. This:

1. Groups opaque and transparent surfaces by zone + orientation
2. Subtracts the total window area from the gross opaque area for the same zone/orientation pair
3. Scales the area of each affected opaque surface proportionally
4. Stores the original values in `gross_area` and `window_area_subtracted`

The console log reports every adjustment that is applied.

### 4.2 V1 simulation - Fully Integrated

```python
res_v1, annual_v1 = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
    building_object=building_object,
    path_weather_file=weather_path,
    weather_source="epw",
    include_solar=True,           # Controlled by --no-solar
    warmup_hours=744,             # Warm-up hours (default 744 = 31 days)
    hvac_control_variable="air",  # HVAC control variable: "air" | "operative"
    progress_log_every_steps=720, # Log every N simulated hours
    progress_logger=...,
)
```

**Returned results:**

- `res_v1` - `pd.DataFrame` with an hourly `DatetimeIndex`, containing for each zone:
  - `T_air_{zone}` - Air temperature [°C]
  - `T_op_{zone}` - Operative temperature [°C]
  - `Q_HVAC_{zone}` - HVAC power [W] (positive = heating, negative = cooling)
  - `H_ve_{zone}` - Ventilation conductance [W/K]
  - `Phi_int_{zone}` - Internal gains [W]
  - `night_purge_active_{zone}` - Night purge active flag [0/1]
  - `night_purge_factor_{zone}` - H_ve multiplier during purge [-]

- `annual_v1` - `pd.DataFrame` with one row per zone containing:
  - `zone` - Zone name
  - `Q_H_annual_kWh` - Annual heating demand [kWh]
  - `Q_C_annual_kWh` - Annual cooling demand [kWh]

### 4.3 V2 simulation - Hybrid Iterative

```python
res_v2, annual_v2, hybrid_iterations = ISO52016.Temperature_and_Energy_needs_calculation_multizone_hybrid(
    building_object=building_object,
    path_weather_file=weather_path,
    weather_source="epw",
    include_solar=True,
    max_iterations=6,      # Controlled by --hybrid-max-iterations
    tolerance_w=10.0,      # Controlled by --hybrid-tolerance-w  [W]
    relaxation=0.6,        # Controlled by --hybrid-relaxation   [0..1]
    warmup_hours=744,
    hvac_control_variable="air",
)
```

**Convergence mechanism:**

The method iterates across the zones, updating coupling flows (heat exchanges between adjacent zones) until the maximum absolute change in the coupling flow `max_abs_delta_coupling_W` falls below `tolerance_w`, or until `max_iterations` is reached.

**Additional results:**

- `res_v2` - hourly DataFrame analogous to `res_v1`, with columns:
  - `T_op_core_{zone}`, `T_air_core_{zone}` - Hybrid core simulation temperatures
  - `Q_HC_hybrid_{zone}` - Hybrid HVAC power [W]

- `annual_v2` - annual DataFrame with:
  - `Q_H_annual_hybrid_kWh`, `Q_C_annual_hybrid_kWh`

- `hybrid_iterations` - diagnostic DataFrame with one row per iteration:
  - `iteration` - Iteration number
  - `max_abs_delta_coupling_W` - Maximum coupling-flow change [W]
  - `mean_abs_delta_coupling_W` - Mean coupling-flow change [W]

The relaxation factor `relaxation ∈ [0, 1]` controls how much the new iteration result affects the current state: `0` = no update, `1` = full update. Values around `0.5-0.7` usually improve numerical stability.

### 4.4 Post-processing and output

After the simulations, the script automatically performs:

1. **Ground temperature calculation** (`_compute_ground_temperature_and_exchanges`): adds the columns `T_ground_virtual`, `H_ground_{zone}`, and `Q_ground_{zone}` to the hourly DataFrame.

2. **CSV export**: all hourly and annual results are written to `result_test/` (or the directory specified with `--output-dir`).

3. **Interactive HTML plots** (via Plotly): operative temperatures, monthly consumptions, ground exchanges, and the V1 vs V2 comparison report.

4. **Zone-wise Sankey diagrams**: annual energy balance for each thermal zone, broken down into heating, cooling, internal gains, solar gains, ventilation losses, thermal bridges, ground, and envelope transmission.

---

## 5. All command-line options

### General options

| Option | Type | Default | Description |
|---|---|---|---|
| `--quick` | flag | `False` | Reduces warm-up to 168 h for fast tests |
| `--warmup-hours N` | int | `744` (or `168` with `--quick`) | Simulation warm-up hours |
| `--weather-path FILE` | str | `examples/2020_Athens.epw` | Path to the EPW climate file |
| `--output-dir DIR` | str | `result_test/` | Output folder for CSV and HTML files |
| `--no-solar` | flag | `False` | Disables solar gains in the comparison step |
| `--progress-log-hours N` | int | `720` | Prints progress every N simulated hours (0 = disabled) |

### Ground temperature

| Option | Type | Default | Description |
|---|---|---|---|
| `--ground-temperature-model MODEL` | str | `iso13370` | Ground temperature model: `iso13370` | `monthly` | `energyplus` |
| `--ground-temperature-monthly JAN FEB ... DEC` | 12 float | - | 12 monthly temperatures [°C] used with `--ground-temperature-model energyplus` |

> `energyplus` is an alias for `monthly`: it uses the 12 provided monthly values or the ones already present in `building_object`.

### Infiltration

| Option | Type | Default | Description |
|---|---|---|---|
| `--infiltration-ext-area-mode MODE` | str | (from `building_object`) | Reference area for the `eplus_infiltration_ext_area` model: `outdoors_only` | `energyplus_like` |
| `--infiltration-wind-reduction-factor F` | float | `1.0` | Multiplier for EPW wind speed (values `< 1` reduce infiltration) |

### Summer night purge

| Option | Type | Default | Description |
|---|---|---|---|
| `--night-purge-preset PRESET` | str | `off` | Night purge preset. Values: `off`, `conservative`, `balanced`, `calibrated`, `global_robust`, `aggressive`, `auto_geo` |
| `--night-purge-disable` | flag | - | Forces purge deactivation regardless of the preset |
| `--night-purge-month-start M` | int | (from preset) | Purge start month [1-12] |
| `--night-purge-month-end M` | int | (from preset) | Purge end month [1-12] |
| `--night-purge-hour-start H` | int | (from preset) | Purge start hour [0-23] |
| `--night-purge-hour-end H` | int | (from preset) | Purge end hour [0-23] |
| `--night-purge-delta-t-min DT` | float | (from preset) | Minimum ΔT (T_indoor - T_outdoor) [°C] to activate purge |
| `--night-purge-boost-factor F` | float | (from preset) | Multiplier for H_ve during purge (≥ 1) |

### Hybrid method

| Option | Type | Default | Description |
|---|---|---|---|
| `--skip-hybrid` | flag | `False` | Skips V2 hybrid simulation |
| `--hybrid-max-iterations N` | int | `6` | Maximum number of coupling iterations |
| `--hybrid-tolerance-w TOL` | float | `10.0` | Convergence tolerance [W] on coupling flow |
| `--hybrid-relaxation R` | float | `0.6` | Relaxation factor [0-1] for iterative updates |

### Sankey

| Option | Type | Default | Description |
|---|---|---|---|
| `--skip-sankey` | flag | `False` | Skips Sankey calculation and export |

---

## 6. Ventilation and infiltration models

The model type is selected with `ventilation_type` in `building_parameters.ventilation`. The relevant parameters change depending on the chosen model.

### `occupancy` - Occupancy-based model

Computes the airflow rate as a function of the number of occupants present during the hour.

```python
"ventilation_type": "occupancy",
"flow_rate_per_person": 0.3   # [L/(s·person)]
```

### `custom` - Fixed conductance

Uses a constant ventilation heat-transfer coefficient for the whole simulation.

```python
"ventilation_type": "custom",
"custom_heat_transfer_coefficient_ventilation": 0.5   # [W/K]
```

### `temp_wind` - ISO 16798-7 temperature/wind model

Computes `H_ve` by combining a temperature-difference term and a wind-speed term:

```python
"ventilation_type": "temp_wind",
"temp_wind_c_wnd": 0.001,          # Wind-dependent coefficient
"temp_wind_c_st": 0.0035,          # Stack coefficient (chimney effect)
"temp_wind_rho_a_ref": 1.204,      # Air density [kg/m³]
"temp_wind_opening_ratio": 0.9     # Opening ratio [-]
```

### `eplus_infiltration_ext_area` - EnergyPlus-style model

Replicates EnergyPlus `DesignFlowRate (Flow/ExteriorArea)` using A, B, C, D coefficients:

```text
Q_inf = q_ref × A_ext × (A + B·|ΔT| + C·v_wind + D·v_wind²) × F_schedule
```

where:
- `q_ref` = `infiltration_flow_per_exterior_area_m3_s_m2` [m³/(s·m²)]
- `A_ext` = total exterior surface area [m²]
- `A`, `B`, `C`, `D` = infiltration coefficients

```python
"ventilation_type": "eplus_infiltration_ext_area",
"infiltration_flow_per_exterior_area_m3_s_m2": 3.0e-4,
"infiltration_coeff_constant": 0.0,               # A - constant
"infiltration_coeff_temperature": 0.0,             # B - ΔT dependent
"infiltration_coeff_velocity": 0.224,              # C - wind-speed dependent
"infiltration_coeff_velocity_squared": 0.0,        # D - wind-speed-squared dependent
"infiltration_include_transparent_area": True,     # Includes windows in A_ext
"infiltration_exterior_area_mode": "energyplus_like",
    # "outdoors_only"    -> only OUTDOORS surfaces
    # "energyplus_like"  -> also includes GROUND contacts in A_ext
"infiltration_wind_reduction_factor": 1.0,         # EPW wind-speed multiplier
"infiltration_schedule_multiplier": 1.0,           # Constant F(t) multiplier
"infiltration_coeff_constant_auto_by_latitude": False
    # True -> selects A automatically from EPW latitude
```

**Automatic selection of coefficient A from latitude:**

| Absolute latitude | Selected A |
|---|---|
| 37° - 42° | 0.30 (Mediterranean climate) |
| 42° - 50° | 0.20 (Central Europe) |
| > 50° | 0.00 (Northern Europe) |

### `sherman_grimsrud_like` - Sherman-Grimsrud model

Based on the effective leakage area and stack/wind coefficients:

```python
"ventilation_type": "sherman_grimsrud_like",
"infiltration_effective_leakage_area_m2": 0.5,   # Effective leakage area [m²]
"infiltration_stack_coefficient": 0.0,            # Stack coefficient
"infiltration_wind_coefficient": 0.0              # Wind coefficient
```

---

## 7. Ground temperature models

Selectable through `simulation_options.ground_temperature_model` or from the CLI with `--ground-temperature-model`.

### `iso13370` (default)

Computes ground temperature according to **EN ISO 13370**, based on building geometry (area, perimeter, wall thickness) and the climate data in the EPW file.

```python
"ground_temperature_model": "iso13370"
```

No additional parameters are required. This is the most physically rigorous method.

### `monthly`

Uses 12 constant monthly temperatures provided directly by the user.

```python
"ground_temperature_model": "monthly",
"ground_temperature_monthly": [9.9, 9.2, 9.8, 11.4, 13.7, 16.1,
                                17.8, 18.5, 17.9, 16.2, 13.9, 11.6]
```

Useful for comparison with EnergyPlus models (the `.idf` file reports the 12 values in `Site:GroundTemperature:BuildingSurface`).

### `energyplus`

Alias of `monthly`, introduced for EnergyPlus naming compatibility. Identical behavior.

**Priority order for monthly values:**

1. Values passed from the CLI with `--ground-temperature-monthly JAN ... DEC`
2. Values already present in `building_object.simulation_options.ground_temperature_monthly`
3. EnergyPlus reference values for Athens (fallback):
   `[9.867, 9.199, 9.773, 11.434, 13.737, 16.066, 17.795, 18.462, 17.889, 16.228, 13.924, 11.596]`

---

## 8. External convection and radiation models

### Internal convection (`internal_convection_model`)

| Value | Description |
|---|---|
| `"table"` | Fixed tabulated coefficients (default ISO 52016) |
| `"tarp"` | TARP model (Temperature Adaptive Radiation and Pressure): depends on orientation and temperature difference |

### External convection (`external_convection_model`)

| Value | Description |
|---|---|
| `"table"` | Fixed tabulated coefficients |
| `"doe2"` | DOE-2 model: combination of forced (wind) and natural terms |
| `"mowitt"` | MoWiTT model: based on experimental building measurements |
| `"blast"` | BLAST model: polynomial wind formula |
| `"simplecombined"` | Simplified combination of natural + forced convection |

The `external_convection_h_min` parameter [W/(m²·K)] enforces a lower bound for the `h_ce` coefficient (default: `2.0`), preventing physically unrealistic values under still-air conditions.

### External radiation (`external_radiation_model`)

| Value | Description |
|---|---|
| `"table"` | Fixed tabulated radiation coefficients |
| `"dynamic"` | Hourly dynamic calculation based on sky temperature |

### Sky temperature model (`sky_temperature_model`)

Used when `external_radiation_model = "dynamic"`:

| Value | Description |
|---|---|
| `"berdahl_fromberg"` | Berdahl-Fromberg correlation (classic default) |
| `"swinbank"` | Swinbank correlation |
| `"epw_ir"` | Uses atmospheric IR irradiance directly from the EPW file (more accurate) |

The `external_emissivity_default` parameter (default: `0.9`) defines the emissivity of external surfaces for the radiative exchange with the sky.

---

## 9. Summer night purge (`summer_night_purge`)

Summer night purge increases the airflow rate during nighttime hours when `T_indoor > T_outdoor + delta_t_min`, pre-cooling the thermal mass of the building. It is activated by multiplying `H_ve` by a `boost_factor`.

### Zone configuration

```python
"summer_night_purge": {
    "enabled": True,         # Enable
    "months": [6, 8],        # Start and end month [1..12]
    "hours": [22, 7],        # Start and end hour [0..23]
    "delta_t_min": 0.1,      # Minimum activation ΔT [°C]
    "boost_factor": 7.0      # H_ve multiplier when active
}
```

### Default presets

| Preset | `enabled` | Months | Hours | `delta_t_min` | `boost_factor` | Recommended use |
|---|---|---|---|---|---|---|
| `off` | No | 6-8 | 22-7 | 0.5 | 1.0 | Default/baseline |
| `conservative` | Yes | 6-9 | 23-6 | 1.0 | 1.8 | Minimal comfort impact |
| `balanced` | Yes | 6-8 | 22-7 | 0.5 | 3.0 | First calibration pass |
| `calibrated` | Yes | 5-10 | 20-9 | 0.1 | 6.0 | Optimized on the EP benchmark |
| `global_robust` | Yes | 6-9 | 20-9 | 0.2 | 4.5 | Robust across multiple climates |
| `aggressive` | Yes | 5-9 | 0-0 | 0.1 | 7.5 | Maximum nighttime cooling |
| `auto_geo` | - | - | - | - | - | Automatic selection from EPW latitude |

### Geographic auto-selection (`auto_geo`)

When `--night-purge-preset auto_geo` is used, the preset is selected based on the absolute latitude of the EPW file:

| Latitude | Selected preset |
|---|---|
| < 43° | `aggressive` (Mediterranean climate) |
| 43° - 50° | `balanced` (temperate Europe) |
| 50° - 60° | `calibrated` (Northern Europe) |
| > 60° | `balanced` |

---

## 10. Generated output files

All files are saved to `result_test/` (or the folder specified by `--output-dir`).

### CSV files

| File | Content |
|---|---|
| `multizone_v1_hourly.csv` | Hourly V1 simulation results (temperatures, HVAC, ventilation) |
| `multizone_v1_annual.csv` | Annual demands per zone for V1 |
| `multizone_v1_timings_seconds.csv` | Computation times [s] for each phase |
| `multizone_v2_hybrid_hourly.csv` | Hourly V2 hybrid simulation results |
| `multizone_v2_hybrid_annual.csv` | Annual demands per zone for V2 hybrid |
| `multizone_v2_hybrid_iterations.csv` | Hybrid iteration convergence diagnostics |
| `multizone_v1_vs_v2_hybrid_summary.csv` | Annual V1 vs V2 comparison with deltas |
| `multizone_v1_sankey_by_zone_summary.csv` | Sankey energy-balance table by zone |
| `multizone_v1_sankey_by_zone.json` | Sankey balance in JSON format |

### Interactive HTML files (Plotly)

| File | Content |
|---|---|
| `multizone_v1_temperatures.html` | Hourly operative temperatures by zone (V1) |
| `multizone_v1_ground_exchange.html` | Virtual ground temperature and zone heat flows (V1) |
| `multizone_v1_monthly_consumptions.html` | Monthly heating/cooling energy by zone (V1) |
| `multizone_v2_hybrid_temperatures.html` | Hourly operative temperatures by zone (V2 hybrid) |
| `multizone_v2_hybrid_monthly_consumptions.html` | Monthly energy by zone (V2 hybrid) |
| `multizone_v1_vs_v2_hybrid_report.html` | Full comparison report V1 vs V2 (5 panels + tables) |
| `multizone_v1_sankey_by_zone.html` | Sankey diagrams of the annual balance for each thermal zone, followed by the annual Sankey and monthly chart of net heat exchanged through internal partitions; a positive monthly value is from zone A to zone B |

### V1 vs V2 comparison report structure

The file `multizone_v1_vs_v2_hybrid_report.html` contains 5 stacked panels:

1. **Hourly operative temperature** - V1 vs V2 for each zone
2. **Hourly HVAC power** - V1 vs V2 for each zone
3. **Monthly heating energy** - V1 vs V2
4. **Monthly cooling energy** - V1 vs V2
5. **Hybrid iteration convergence** - `max_abs_delta_coupling_W` and `mean_abs_delta_coupling_W` for each iteration

These are followed by two HTML tables: the annual summary with deltas and the iteration diagnostics.

### Sankey structure by zone

For each zone, the energy balance is divided into:

**Inputs:**
- Heating (`Q_HVAC > 0`)
- Internal gains (`Phi_int`)
- Solar gains + free gains (solar + favorable wind + favorable thermal bridges)

**Outputs:**
- Cooling (removed energy, `|Q_HVAC < 0|`)
- Ventilation losses (`H_ve · ΔT`)
- Thermal bridges (`H_tb · ΔT`)
- Ground exchanges
- Transmission through the envelope (`UA · ΔT`)

---

## 11. Typical usage examples

### Fast test run

```bash
python examples/multizone_free_floating_example.py --quick
```

Warm-up is reduced to 168 h, which is useful to verify that everything works before launching the full annual simulation.

### Simulation with a custom climate file

```bash
python examples/multizone_free_floating_example.py \
    --weather-path /data/weather/Milan_2022.epw \
    --output-dir /results/Milan
```

### V1 only simulation (without hybrid and without Sankey)

```bash
python examples/multizone_free_floating_example.py \
    --skip-hybrid \
    --skip-sankey
```

This significantly reduces computation time while preserving the full V1 results.

### Aggressive night purge for a Mediterranean climate

```bash
python examples/multizone_free_floating_example.py \
    --night-purge-preset aggressive
```

### Night purge with automatic preset selection from EPW latitude

```bash
python examples/multizone_free_floating_example.py \
    --night-purge-preset auto_geo
```
