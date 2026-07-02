# lighting_15193_1.py Description

`lighting_15193_1.py` implements a practical lighting-energy model inspired by **BS EN 15193-1:2017**.
It estimates lighting electricity use, standby energy, emergency lighting energy, and the resulting internal heat gains.
The module is designed for simulation workflows where hourly or aggregated lighting results are needed as inputs to the broader building energy model.

The implementation is useful and structured, but it is **not a full normative transcription** of EN 15193-1.
It follows the general logic of the standard while simplifying some of the control-factor and daylight-calculation parts.

## What the script does

The script:

1. Accepts a configuration dictionary with lighting-related inputs.
2. Reads time-series data such as floor area, occupancy profile, lighting profile, and optional irradiance.
3. Computes:
   - lighting electric power,
   - lighting electricity use,
   - parasitic/standby energy,
   - emergency lighting energy,
   - total lighting energy,
   - internal heat gains from lighting,
   - LENI.
4. Returns both a full hourly/monthly/yearly time series and a summary dictionary.

## Main objects and functions

### `LightingSimulationResult`

Container for the results returned by the calculator.

- `timeseries`: pandas DataFrame with the detailed results
- `summary`: dictionary with aggregated indicators
- `inputs`: normalized input dictionary used by the calculation

### `LightingSystemCalculator`

Main class that performs the calculation.

### `_load_options()`

Reads the input dictionary and stores:

- area
- installed power density
- daylight dependency factor
- occupancy dependency factor
- constant illuminance factor
- daylight control fraction
- occupancy control fraction
- parasitic power
- emergency lighting annual density
- internal gain fraction

### `_prepare_timeseries()`

Normalizes the input time-series dataframe and extracts:

- time step hours
- floor area
- lighting profile
- occupancy profile
- global horizontal irradiance

### `_simulate()`

Runs the core lighting-energy calculation:

- computes a control multiplier
- computes lighting power
- computes lighting electricity
- computes parasitic energy
- computes emergency energy
- computes total energy
- computes internal gains

### `_daylight_reduction()`

Applies a simplified daylight effect based on irradiance.

### `_summarize()`

Builds the aggregate outputs:

- lighting energy
- parasitic energy
- emergency energy
- total energy
- internal gains
- LENI
- peak lighting power
- mean control factor

## Conformity with BS EN 15193-1:2017

The table below summarizes the relationship between the script and the standard.

| Norma | Cosa richiede | Stato nel codice |
|---|---|---|
| Scope generale | Calcolo dell’energia per illuminazione e LENI per edifici residenziali e non residenziali | **Sì**, a livello generale |
| Method 1 | Output per time step, annual energy `W`, LENI `W/A` | **Sì**, implementato |
| Installed power | Somma delle potenze installate dei luminari | **Sì**, ma in forma semplificata |
| Standby energy | Energia per controllo e battery charging | **Sì**, ma in forma aggregata |
| Occupancy dependency factor `FO` | Derivazione da `FA` e `Foc`, con casi distinti | **Parziale** |
| Daylight dependency factor `FD` | Calcolo tramite daylight supply factor e daylight control factor | **Parziale** |
| Constant illuminance factor `FC` | Legato al maintenance factor e alla constant illuminance efficiency | **Parziale** |
| Method 2 | Calcolo annuale budget con default data | **Solo parziale** |
| Method 3 | Metered energy reale, somma letture meter | **No** come metodo distinto |
| Quality control | Report con metodo usato, somme annuali, dichiarazione tolleranze | **No** come workflow esplicito |

## Parts aligned with the standard

The following parts are aligned with the standard at a structural level:

- annual LENI output
- annual energy normalization by useful floor area
- installed lighting power concept
- standby and emergency energy accounting
- time-step based results

## Parts not fully aligned with the standard

The following parts are simplified or not fully normative:

- `FO` is not computed using the full EN 15193-1 annex procedure.
- `FD` is approximated using irradiance rather than the full daylight-supply formulation.
- `FC` is treated as an input factor instead of being derived from the full maintenance-factor logic.
- method 2 is not separated from method 1 as a distinct algorithm.
- method 3 is not implemented as a true metering workflow.
- compliance and quality-control reporting are not implemented as in the standard.

## Practical interpretation

This module is suitable when the goal is to:

- estimate lighting electricity in a building model,
- pass lighting internal gains to the thermal simulation,
- obtain a LENI-like indicator for reporting or comparison,
- keep the model data-driven and lightweight.

It should not be presented as a strict, fully compliant implementation of BS EN 15193-1:2017 without further work on:

- daylight factor calculations,
- occupancy logic,
- maintenance factor treatment,
- method differentiation,
- compliance reporting.

## Recommended next step

If stricter compliance is needed, the module should be refactored to:

1. separate Method 1, Method 2, and Method 3,
2. implement the annex-based `FO`, `FD`, and `FC` logic more explicitly,
3. add a proper quality-control/reporting layer.
