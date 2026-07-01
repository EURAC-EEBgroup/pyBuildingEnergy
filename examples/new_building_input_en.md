# Chapter: Interpretation of Distribution Inputs According to EN 15316-3

## EN 15316-2 - Emission System

In `new_building.py`, this part is organized on two levels:

- a legacy interface block used by `iso_15316_1.py`
- a detailed block used by `emission_15316_2.py`

EN 15316-2 describes the emission subsystem, namely the transition from the useful zone energy need to the heat actually delivered to the indoor environment. In `new_building.py`, this part is handled by the `emission_15316_2_config` block and is used to calculate, hour by hour, emission losses, equivalent temperature increases caused by terminal components, and any electrical auxiliaries for fans or controls.

In the model, the emission subsystem is the connection point between the zone load and distribution. Here the thermal need is corrected by accounting for stratification, control, radiation, hydraulic balancing, room automation and, when required, intermittent operation effects.

### Interface Parameters Used by `iso_15316_1.py`

These are the inputs read directly by the `HeatingSystemCalculator`.

`emitter_type: 'Floor heating'` identifies the emitter or terminal type. It is not a numeric parameter because it represents a system category. The model uses this value to associate the terminal with a performance curve or with a coherent TB14 table. The parameter must belong to the set of terminals supported by the model catalog. If the type is not recognized, the input check replaces it with a coherent fallback.

With the default `TB14` table supplied by the project, the available types are:

- `Radiator`
- `Floor heating`
- `Fan coil`

If a custom `TB14` table is passed in the input dictionary, the list of possible values changes automatically and matches the indices of that table. Therefore, `emitter_type` does not have a universal closed list: it depends on the active table in the model.

| `emitter_type` | Physical meaning | Typical model behavior |
|---|---|---|
| `Radiator` | Traditional radiator terminal with mainly convective and radiative emission | Usually has a high water-to-air temperature difference and a more pronounced emission exponent |
| `Floor heating` | Radiant floor heating system | Operates with low supply temperatures, limited temperature difference and more stable behavior |
| `Fan coil` | Fan coil terminal with fan and coil | Has faster response and electrical auxiliaries for ventilation |

This list applies to the default `TB14`; if the table is replaced with a custom one, the available types are those listed in the loaded table index.

`nominal_power: 8.0` is the nominal emission power, expressed in `kW`. It represents the reference size of the terminal in the ISO 15316-1 branch. The value must be greater than `0 kW` and coherent with distribution and generation. In the example file, `8.0 kW` is a typical size for a small residential unit.

`emission_efficiency: 90.0` is the emission efficiency, expressed in `%`. The coherent minimum is `0%`; the theoretical maximum is `100%`, although slightly higher values may appear in simplified interface logic. For a real terminal, the value should remain in the `0-100%` range.

`flow_temp_control_type: 'Type 2 - Based on outdoor temperature'` describes the supply temperature control logic of the emission circuit. It is not a numeric parameter, but a mode selector. In `iso_15316_1.py`, the actual options are:

- `Type 1 - Based on demand`: follows the demand
- `Type 2 - Based on outdoor temperature`: follows the outdoor temperature
- `Type 3 - Constant temperature`: keeps an almost constant setpoint

`selected_emm_cont_circuit` selects the type of emission circuit. It is an integer parameter with no unit of measure. In the code, the supported values are those listed below. Any other value raises an error. This parameter represents the hydraulic topology of the emission circuit and must be coherent with the selected terminal:

- `0` = constant circuit
- `1` = variable-flow circuit
- `2` = mixing/control circuit with dedicated control
- `3` = constant-flow circuit with variable emission behavior

`mixing_valve: True` enables or disables the presence of a mixing valve. It is a Boolean parameter and therefore has no unit of measure. If the circuit includes mixing between the primary and secondary branches, the value must be `True`; otherwise it must be `False`.

`emission_calculation_mode` decides whether the emission branch uses the internal simplified model or the hourly EN 15316-2 calculation. The accepted values are:

- `simplified`
- `en15316-2`

The aliases `iso_15316_2`, `15316_2` and `en15316_2` are automatically normalized to `en15316-2`.

`emission_time_step_hours: 1.0` is the emission simulation time step, expressed in `h`. The value must be positive; for hourly simulation the typical value is `1.0 h`.

`heat_emission_data` is the terminal characterization table used by the legacy `iso_15316_1.py` branch. It contains TB14 data or a custom table with the emitter-type parameters from which the model derives the emission behavior.

`outdoor_temp_data` is the emission-side climatic curve table used by the legacy `iso_15316_1.py` branch. If it is passed inside `INPUT_SYSTEM_HVAC`, the constructor stores it in `self.df_out_temp`; otherwise the code uses the default table returned by `_default_outdoor_data()`. The function `calculate_circuit_node_temperature(theta_ext, emission_results)` reads `theta_ext_min_sahz_i`, `theta_ext_max_sahz_i`, `theta_em_flw_min_sahz_i` and `theta_em_flw_max_sahz_i` from this table, then interpolates the emission-circuit supply temperature as a function of `theta_ext`, constraining the result between the minimum and maximum curve limits.

`constant_flow_temp` is the constant supply setpoint of the emission circuit. It is used as a fallback value when no climatic curve applies or when a fixed supply temperature is imposed. In the legacy branch it represents the secondary-side supply temperature kept constant.

`mixing_valve_delta` is the temperature difference introduced by the mixing valve between the primary and secondary branches. It represents the temperature correction applied to the emission circuit when the supply temperature must be adapted relative to the upstream circuit. In the legacy model it is read by `iso_15316_1.py`; in the EN 15316-2 branch it acts as a thermal-interface consistency parameter.

### General Parameters of `emission_15316_2_config`

`time_step_hours: 1.0` indicates the internal calculation time step of the module, in `h`. It must be greater than `0`.

`demand_unit: 'kWh'` indicates the unit of measure of the input demand. The supported options are `Wh` and `kWh`.

`default_heating_internal_temperature_C: 20.0` is the reference indoor temperature for heating, in `°C`. The value must be realistic for the served environment; in practice it is often between `18` and `22 °C`.

`default_cooling_internal_temperature_C: 26.0` is the reference indoor temperature for cooling, in `°C`. In practice it is often between `24` and `28 °C`.

### Parameters of the `heating` Section

The `heating` section collects the equivalent contributions that increase the indoor temperature seen by the terminal.

`stratification_K: 0.0` is the equivalent contribution due to stratification, expressed in `K`. It must be greater than or equal to `0 K`. Typical values are small, often between `0` and `2 K`.

`control_K: 0.0` is the equivalent contribution due to terminal control, expressed in `K`. The minimum value is also `0 K`.

`radiation_K: 0.0` is the equivalent contribution due to radiation, expressed in `K`. It must be greater than or equal to `0 K`.

`hydraulic_balancing_K: 0.0` is the equivalent contribution due to hydraulic balancing, expressed in `K`. It must be greater than or equal to `0 K`.

`room_automation_K: 0.0` is the equivalent contribution of room automation, expressed in `K`. It must be greater than or equal to `0 K`.

`embedded_K: 0.0` represents the thermal contribution of emitters embedded in the building structure, expressed in `K` as an equivalent effect on the calculation. It must be greater than or equal to `0 K`.

`nominal_power_kW: 8.0` is the nominal power of the heating branch, in `kW`. It must be positive and coherent with the upper-level `nominal_power`.

`fan_power_W: 0.0` is the electrical power of the fan, in `W`. The minimum allowed value is `0 W`.

`fan_count: 0` is the number of fans. It is an integer count, so the minimum is `0`.

`control_power_W: 0.0` is the electrical power of the control devices, in `W`. The minimum allowed value is `0 W`.

`control_count: 0` is the number of control devices. The minimum is `0`.

`convective_fraction: 0.7` is the convective fraction of the emission. It has no unit of measure because it is a dimensionless fraction. The value must be between `0` and `1`.

### Parameters of the `cooling` Section

The `cooling` section uses the same structure as the heating part, but with sign and thermal reference consistent with cooling.

`stratification_K`, `control_K`, `radiation_K`, `hydraulic_balancing_K`, `room_automation_K` and `embedded_K` have the same units and limits as in the heating section: `K`, with a minimum value of `0`.

`nominal_power_kW` has the same unit of measure as in the heating part, namely `kW`, and must be positive.

`fan_power_W` and `control_power_W` are in `W` and must be greater than or equal to `0 W`.

`fan_count` and `control_count` are integer counts and must be greater than or equal to `0`.

`convective_fraction` is a dimensionless fraction between `0` and `1`.

### Parameters Shared with EN 15316-1

Some emission coefficients are conceptually shared with EN 15316-1, although they may be read from different blocks or with slightly different names:

- `mixing_valve_delta`: in EN 15316-2 it describes the thermal correction linked to emission-circuit mixing; in EN 15316-1 it is called by the thermal interface logic and may influence the generator-side supply temperature
- `flow_temp_control_type`: in EN 15316-2 it controls the emission branch; in EN 15316-1 the equivalent generator logic is `gen_flow_temp_control_type`
- `nominal_power` / `nominal_power_kW`: these are the same physical quantity viewed at two different model levels, namely the reference power of the terminal or subsystem
- `emission_time_step_hours` and `time_step_hours`: these must be coherent with the simulation time step used by the overall system

The consistency principle is important: the value assigned to the emission part must be compatible with distribution and generation, otherwise the hourly energy balance will not close correctly.

## EN 15316-3 - Distribution System

This chapter describes the heat distribution subsystem, namely the path that carries thermal energy from the generator to the terminals and, in some cases, also accounts for heat losses and auxiliaries. EN 15316-3:2017 specifically covers these aspects: network thermal losses, recoverable loss share, circulator electricity, auxiliary recovery, and dependence on the geometric and hydraulic parameters of the distribution network.

In `new_building.py`, distribution is set with `distribution_calculation_mode = 'analytical'`, meaning that it uses the branch closest to the standard structure, with a network described through sections, lengths, linear transmittances and pump parameters. The alternative `simplified` is an aggregate approximation, useful when the circuit should not be described in detail.

This chapter is intended as part of a wider chain. The previous or parallel section of the document may cover EN 15316-2 for emission, while the following section may cover EN 15316-4-1 for generation. This makes the three blocks readable as successive parts of the same HVAC system.

### Thermal Losses and Recoverability

`heat_losses_recovered: True` means that the model considers part of the distribution thermal losses recoverable when the pipe section is located in a conditioned space. This is coherent with EN 15316-3, which distinguishes total distribution losses, recoverable losses and non-recoverable losses.

`distribution_loss_recovery: 90` indicates the percentage of thermal loss assumed as recoverable by the model. With `90`, 90% of the losses are considered recoverable in the room balance, while 10% remain net losses.

`simplified_approach: 80` is not a direct EN 15316-3 standard input. It is an internal model control used to decide whether the simplified branch considers loss recovery. In the current code its use is very simple: if it is `0`, recovery is disabled; if it differs from `0`, recovery remains enabled. Therefore, `80` does not really mean "80% according to the standard"; in practice it is a historical or conventional value, not a percentage used continuously by the code.

`distribution_aux_recovery: 80` indicates the share of distribution auxiliary electricity that can be recovered as useful heat in the environment. EN 15316-3 also accounts for pump electricity, the part that becomes heat in the rooms and any recoverable part.

`distribution_aux_power: 30` is the nominal distribution auxiliary electrical power, expressed in watts. In the standard, pump auxiliary energy depends on design hydraulic power, pressure differential, design flow rate, pump energy efficiency and operating time.

`distribution_loss_coeff: 48` is a global thermal coefficient used in the simplified branch of the model. In aggregate form, it expresses how sensitive losses are to the temperature difference between pipe and environment. In the standard model, thermal loss is more properly calculated as a sum over sections:

```text
Q = Psi * (T_m - T_amb) * (L + L_equiv) * t / 1000
```

The symbols in the formula mean:

- `Q`: distribution thermal losses, in kWh
- `Psi`: linear thermal transmittance of the section, in W/(m*K)
- `T_m`: mean fluid temperature in the pipe, in °C
- `T_amb`: ambient temperature around the pipe section, in °C
- `L`: actual pipe length, in m
- `L_equiv`: equivalent length of fittings and discontinuities, in m
- `t`: section operating time, in h

Therefore, this parameter is an implementation shortcut: instead of describing each section with its own `Psi`, length and ambient temperature, a single coefficient is used.

`distribution_operation_time: 1` is the calculation time for each step, in hours. With value `1`, the simulation is hourly. EN 15316-3 can be applied at different time scales: hourly, monthly and annual. Here the model is coherent with the hourly step: each simulation row corresponds to one hour.

`distribution_calculation_mode: 'analytical'` tells the model to use the analytical distribution calculation, namely the branch that represents network sections, linear losses, additional losses, pump auxiliaries and recoveries. If it were `simplified`, the model would use a more compact approach, with less geometric detail and less dependence on the physical structure of the circuit.

### Analytical Distribution Configuration

`distribution_15316_3_config` contains the data used by the model to reconstruct the distribution network according to a sectional logic.

`time_step_hours: 1.0` indicates that the simulation time step is one hour.

`demand_unit: 'kWh'` indicates that the thermal demand is expressed in kWh.

The `heating` section describes the circuit branch dedicated to space heating.

`operation_mode: 'demand'` means that distribution is calculated as a function of heat demand. This is the most natural choice for a traditional residential system, because the network operates only when there is a thermal need. See the section `Meaning of operation_mode` for more details.

`nominal_power_kW: 8.0` is the nominal reference power of the subsystem. The value must be coherent across the whole system: emission, distribution and generation must be consistent.

`design_flow_m3_h: 0.5` is the design flow rate of the network, in m3/h. The value must be aligned with the system; see the section `Design flow check`.

`design_deltaT_K: 10.0` is the design temperature difference between supply and return. See `Meaning of design_deltaT_K` for more details.

These parameters describe the hydraulic and energy behavior of the circuit.

Note. `time_step_hours` and `distribution_operation_time` do not indicate the same thing. `time_step_hours` defines the simulation time step, namely the duration of each calculation block. `distribution_operation_time` instead defines how long distribution is actually active within that step. In the simplest case, with hourly step and distribution always operating, both are equal to `1`. If distribution operates only for part of the step, `distribution_operation_time` may be lower than `1` even if `time_step_hours` remains `1.0`.

### Pipe Sections

`pipe_sections` describes the network as one or more homogeneous sections. The standard does not reason on a single generic pipe, but on network sections. Each section may have a length, an equivalent length, a linear transmittance, an ambient temperature and a recoverable or non-recoverable share.

In the `new_building.py` example there is only one section, so the circuit is modeled as one aggregate section.

`length_m` is the physical length of the pipe section considered. In EN 15316-3, length is an essential parameter because losses grow linearly with pipe length: more pipe means more heat loss and more auxiliary energy associated with the circuit. With a single `pipe_section`, `length_m` may represent a well-defined real section or an aggregated equivalent section that condenses several network parts.

`equivalent_length_m` is the additional equivalent length caused by network accessories:

- valves
- fittings
- bends
- supports
- local components that generate additional losses

The code does not define one mandatory tabulated value: the parameter is treated as a design input. If detailed data are not available, it can be left at `0.0`.

`linear_thermal_transmittance_W_mK` is the linear thermal transmittance of the section, namely `Psi`, expressed in W/(m*K). This is one of the central parameters of EN 15316-3: the thermal loss of a section depends on linear transmittance, length and temperature difference with the surroundings.

`ambient_temperature_C` is the temperature of the environment around the pipe. It is not a fixed standard value: it depends on where the pipe is installed. If the pipe passes through a heated room, it may be about `20 C`; if it passes through an unheated space, the temperature may be lower; if it is near external walls or cold spaces, the value changes again.

In the configuration file, this value is treated as static for the specific `pipe_section`. In other words, if `ambient_temperature_C = 20.0`, that section always uses `20.0 C` for every simulation step.

In a more advanced model, this parameter could be linked to a dynamic ambient temperature calculated hour by hour. The reason is physical: distribution losses depend on the difference between pipe temperature and surrounding ambient temperature, so if the surrounding environment changes over time, network losses should also change.

This choice is especially useful when the pipe section passes through a plant room with variable temperature, is located in an unconditioned space, or when network losses should follow the hourly evolution of climate or thermal zone.

`recoverable: True` means that the section is considered to be in a conditioned space, so thermal losses can be recovered in the room balance.

### Distribution Pump

`pump_control_code: 4` indicates a variable-speed control based on variable `Delta p`.

The values explicitly handled by the current code are:

- `0` = uncontrolled
- `3` = constant `Delta p` control
- `4` = variable `Delta p` control

Values different from these fall back to the default implementation behavior and do not introduce a distinct logic. For consistency with the model and the standard, one of these three codes should be used.

`eei: 0.23` is the energy efficiency index of the pump.

`hydraulic_correction_factor: 1.0` is the hydraulic correction factor of the circuit.

`recoverable_aux_fraction: 0.25` is the share of pump auxiliaries recoverable as heat in the room.

`pressure_loss_per_m_kPa: 0.10` is the linear pressure loss per meter of pipe.

`additional_pressure_kPa: 0.0` is the additional pressure loss caused by terminals or special components.

`resistance_ratio: 0.30` is the factor representing distributed and local network resistances.

`pump_selection_factor: 1.0` is the factor linked to how the pump was selected relative to the design point.

`pump_label_power_kW: 0.0` is the nominal nameplate power of the pump, if known.

`part_load_mode: 'load'` is the part-load management mode of the pump in the implemented model.

### Meaning of `operation_mode`

`operation_mode` indicates how long, inside each time step, distribution remains active. In the code, the values actually handled are `demand`, `continuous` and `fixed_fraction:<value>`.

`demand` means that distribution operates only when there is thermal demand. This is the most natural behavior for a traditional residential system: if there is demand the network operates, if there is no demand the network and pump are considered off.

`continuous` means that distribution operates for the entire time step, regardless of demand. This choice makes sense when the circuit is actually operating continuously, for example in systems with constant circulation or very limited intermittent control.

`fixed_fraction:<value>` means that distribution operates for a fixed fraction of the time step. For example, `fixed_fraction:0.5` means that the circuit is active for 50% of the hour. The value must be between `0` and `1`.

For residential heating, the most sensible choice is `demand`. For a system with continuous circulation, the most sensible choice is `continuous`. For DHW, the choice depends on the network type: in general `continuous` is appropriate if domestic hot water circulation is permanent, while `demand` is more suitable if the system operates only during use or service windows.

### Simplified Distribution Branch

The `simplified` mode uses the historical aggregate distribution calculation branch in `iso_15316_1.py`. In this mode, the network is not reconstructed in sections as in the analytical branch; it is represented by a more compact global formula.

The simplified branch directly estimates total distribution thermal losses, distribution auxiliaries, the recoverable loss share and the distribution flow rate derived from the energy balance.

The inputs used are:

- `theta_H_dis_flw`: heating distribution supply temperature, in °C
- `theta_H_dis_ret`: heating distribution return temperature, in °C
- `theta_int`: indoor temperature of the thermal zone, in °C
- `QH_em_i_in`: thermal energy entering the distribution subsystem, in kWh
- `Heat_loss_coefficent_dist`: global distribution heat loss coefficient, in W/K
- `W_H_dist_i_aux`: nominal distribution auxiliary electrical power, in W
- `f_h_dist_i_aux`: percentage share of auxiliaries considered recoverable, in %
- `Heat_losses_recovered`: Boolean flag enabling or disabling heat loss recovery
- `simplified_or_holistic_approach`: internal control parameter enabling or disabling recovery in the simplified branch
- `f_h_dist_i_ls`: percentage share of thermal losses considered recoverable, in %
- `tH_dis_i_ON`: distribution operating time in the step, in h
- `c_w`: volumetric thermal capacity of water assumed by the model, in kWh/(m3*K)

In practice, the model uses the circuit mean temperature, the difference between pipe and indoor environment, a global loss coefficient, an aggregate auxiliary power and fixed recovery factors.

The simplifications are significant: it does not use `pipe_sections`, `length_m`, `equivalent_length_m`, `linear_thermal_transmittance_W_mK`, `ambient_temperature_C`, the detailed pump control of the `DistributionSystemCalculator`, or the EN 15316-3 sectional calculation.

Losses are estimated with an aggregate form such as:

```text
Q_w_dis_i_ls = (T_mean - theta_int) * Heat_loss_coefficent_dist / 1000 * tH_dis_i_ON
```

where:

- `Q_w_dis_i_ls`: distribution thermal losses in the step, in kWh
- `T_mean`: mean distribution circuit temperature, i.e. the average of supply and return, in °C
- `theta_int`: indoor temperature of the thermal zone, in °C
- `Heat_loss_coefficent_dist`: global distribution heat loss coefficient, in W/K
- `1000`: conversion factor from Wh to kWh
- `tH_dis_i_ON`: distribution operating time in the step, in h

This is a shortcut compared with EN 15316-3, which instead uses sections, linear transmittance, pipe length, equivalent length and section ambient temperature.

Auxiliaries are treated directly:

```text
Q_w_dis_i_aux = tH_dis_i_ON * W_H_dist_i_aux / 1000 * f_h_dist_i_aux / 100
```

where:

- `Q_w_dis_i_aux`: distribution auxiliary energy in the step, in kWh
- `tH_dis_i_ON`: distribution operating time in the step, in h
- `W_H_dist_i_aux`: nominal distribution auxiliary electrical power, in W
- `1000`: conversion factor from Wh to kWh
- `f_h_dist_i_aux`: percentage share of auxiliaries considered recoverable, in %
- `100`: percentage-to-fraction conversion factor

Loss recovery is simplified to a fixed percentage: if `Heat_losses_recovered = False` or `simplified_or_holistic_approach = 0`, there is no recovery; otherwise a share equal to `f_h_dist_i_ls` is recovered.

The `simplified` mode is useful when a fast model, few inputs and no detailed geometric description of the network are required. It is less faithful to EN 15316-3 because it approximates the network with a single coefficient, does not distinguish real circuit sections and does not describe the pump in detail.

### Meaning of `design_deltaT_K`

`design_deltaT_K` is not a fixed value of the hourly calculation: it is a design value.

In `distribution_15316_3_config.heating` it is used to size the design flow rate and to derive `design_flow_m3_h` when it is not specified explicitly. The actual temperature difference during the simulation may instead vary hour by hour depending on the temperatures calculated by the model.

In short:

- `design_deltaT_K` is a fixed design input
- `theta_H_dis_flw - theta_H_dis_ret` is the actual hourly-step temperature difference

The first defines the circuit, while the second emerges from the dynamic model calculation.

The reference depends on the calculation branch. In the `simplified` branch, distribution loss is calculated relative to `theta_int`, namely the indoor temperature of the thermal zone. In the `analytical` branch, each section uses `ambient_temperature_C`, the temperature of the environment around the pipe section. This means that the reference may be the thermal zone, a plant room, an unheated room, a wall cavity or another environment. Therefore, the analytical branch does not automatically assume the thermal zone: it depends on how the pipe section is modeled.

### Practical Meaning of `length_m`

`length_m` represents the length of the pipe section described by a specific `pipe_section`.

With a single `pipe_section`, this value may represent a real well-defined section or an aggregated equivalent section that condenses several network parts.

If the model uses several sections, the network should be divided into distinct sections, for example the branch from generator to main distribution, vertical risers, branches to terminals and any sections located in different environments. In that case, each section has its own `length_m`.

If only one `pipe_section` is used, `length_m` becomes an aggregate length: it condenses the whole network into one equivalent section. This choice is useful when detailed network geometry is not known and a simpler model or a limited number of inputs is desired.

For a small residential dwelling, a plausible estimate of `length_m` can be obtained by considering the actual network path:

- generator to main distribution: `4 m`
- vertical section or technical passage: `3 m`
- branch to terminals: `5 m`

The total value becomes:

```text
length_m = 4 + 3 + 5 = 12 m
```

This explains why in `new_building.py`, `length_m = 12.0` is plausible as a single aggregated section for a small dwelling if the network has been condensed into one `pipe_section`.

In the `simplified` branch, `length_m` is not used directly. Length becomes relevant only in the `analytical` branch, where losses are calculated by section.

### Practical Meaning of `equivalent_length_m`

`equivalent_length_m` represents the additional equivalent length caused by network accessories:

- valves
- fittings
- bends
- supports
- local components that generate additional losses

The code does not define one mandatory tabulated value: the parameter is treated as a design input. If detailed data are not available, it can be left at `0.0`.

It is used when the distribution network should be represented more realistically and additional losses not directly included in the physical pipe length should be estimated.

For a small dwelling, a simple estimate can be adopted, for example:

- 4 main bends equivalent to about `0.3 m` each
- 2 valves equivalent to about `0.5 m` each
- 2 fittings equivalent to about `0.2 m` each

The total becomes:

```text
equivalent_length_m = 4 * 0.3 + 2 * 0.5 + 2 * 0.2 = 2.4 m
```

This is a practical and conservative estimate, useful when no detailed survey of the network is available. For a simple model, `equivalent_length_m = 0.0` may be kept; for a slightly more realistic model, the equivalent lengths of the main accessories are summed and the result is used as the aggregate value of the section.

### Consistency Between Emission, Distribution and Generation

In `new_building.py`, the main values are aligned:

- `nominal_power = 8.0 kW`
- `nominal_power_kW = 8.0 kW`
- `full_load_power = 27 kW`
- `rated_power_kW = 27.0 kW`

This alignment is correct because the nominal power of emission and distribution must describe the same reference load of the terminal branch, while generation may have a higher power, i.e. a size coherent with the real system generator. In other words, emission and distribution must be coherent with each other; generation must be coherent with the selected generator; emission and generation do not need to have the same power, but they must represent the same physical system and use case.

### Automatic Consistency Checks in `new_building.py`

Two automatic checks have been added in `new_building.py` before starting the simulation. If one of the checks fails, the calculation is stopped with an explicit error.

#### Power Alignment Check

This check verifies that the main values of the HVAC branch are mutually coherent:

- `nominal_power`
- `distribution_15316_3_config.heating.nominal_power_kW`
- `full_load_power`
- `boiler_generation_config.rated_power_kW`

The check requires all powers to be positive, the nominal emission power and distribution power to be aligned, the emission power and generator power to be coherent, and the generator power and boiler nominal power to match within a fixed tolerance.

#### Design Flow Check

This check verifies that `design_flow_m3_h` is coherent with `nominal_power` and `design_deltaT_K`. The check uses the basic thermal relation:

```text
Vdot = P / (c * DeltaT)
```

In the example case:

- `nominal_power = 8.0 kW`
- `design_deltaT_K = 10 K`

The expected theoretical flow rate is therefore about `0.70 m3/h`. If the inserted value is `0.5 m3/h`, the check considers it incoherent and blocks the simulation.

This check prevents the distribution network from being sized with a flow rate that is incompatible with the thermal load it must carry.

## Domestic Hot Water Flow in `new_building.py`

This chapter describes the DHW flow implemented in `new_building.py`. The DHW branch is calculated separately from the space-heating branch up to the point where the user decides whether both services are supplied by the same generator.

### Calculation Sequence

The annual hourly domestic hot water demand is calculated with `DHW.py` according to EN 12831-3.

If the user enables storage, the DHW profile is passed through the storage calculation according to EN 15316-5. The storage block adds storage losses and any storage auxiliaries, and returns the storage-side input energy required to cover the DHW load.

The load is then passed to the distribution calculation according to EN 15316-3. The distribution block adds pipe losses, hydraulic auxiliary energy and recoverable shares according to the configured DHW distribution network.

The final result of the DHW chain represents the DHW-side generator request. In the current `new_building.py` flow, this value is read from the hourly DHW storage output when storage is enabled, or from the hourly DHW distribution output when storage is disabled.

The physical sequence is therefore:

```text
DHW useful demand -> DHW distribution -> DHW storage -> DHW generator request
```

When storage is disabled, the sequence becomes:

```text
DHW useful demand -> DHW distribution -> DHW generator request
```

### Shared or Separate Generator

The following parameter is available in `INPUT_SYSTEM_HVAC`:

- `same_generator_for_heating_and_dhw`

#### If `True`

Space heating and DHW are supplied by the same generator.

In this case the DHW calculation does not require a separate generator configuration:

- the heating demand is calculated by the main HVAC branch
- the DHW useful demand is calculated with `DHW.py`
- the DHW value after distribution and storage is added to the heating-side generator load to obtain the combined generator request

This is the appropriate configuration when one boiler, heat pump or other generator supplies both space heating and domestic hot water.

#### If `False`

Space heating and DHW are supplied by separate generators.

In this case the user must define the DHW generator parameters in:

- `dhw_generator_config`

With a separate DHW generator, the file does not add DHW to the heating load. The DHW demand remains an independent generator-side request and should be sized and interpreted with the DHW generator configuration.

### Optional Storage

The following parameter enables or disables the EN 15316-5 storage step:

- `dhw_storage_enabled`

#### If `True`

The hourly DHW demand passes through the storage block.

The storage calculation returns:

- storage losses
- storage auxiliaries
- storage input energy on the generator side

The storage input energy is then used as the generator-side DHW request in the output reports and in the Sankey diagram.

#### If `False`

The DHW demand passes directly to distribution.

No storage losses are calculated, and the DHW request after distribution is used as the generator-side DHW request.

### DHW Output Files

The DHW flow produces the following files:

- `hvac_stage_results.csv`
- `dhw_storage_15316_5_hourly_results.csv`, if storage is enabled
- `dhw_storage_15316_5_summary.csv`, if storage is enabled
- `dhw_distribution_15316_3_hourly_results.csv`
- `dhw_distribution_15316_3_summary.csv`
- `hvac_dhw_hourly_results.csv`

The combined file `hvac_dhw_hourly_results.csv` contains the hourly HVAC results together with:

- `dhw_Q_W_kWh` for useful DHW demand from EN 12831-3
- `dhw_dis_...` columns for EN 15316-3 DHW distribution results
- `dhw_sto_...` columns for EN 15316-5 DHW storage results, when storage is enabled

### Main DHW Parameters

The main parameters controlling the DHW chain are:

- `same_generator_for_heating_and_dhw`
- `dhw_storage_enabled`
- `dhw_storage_config`
- `dhw_generator_config`
- `distribution_15316_3_config.dhw`

`same_generator_for_heating_and_dhw` selects whether the DHW and heating loads are combined at the generator level.

`dhw_storage_enabled` selects whether the EN 15316-5 storage block is included in the DHW chain.

`dhw_storage_config` contains the storage volume, setpoint temperature, ambient temperature, standby losses, recoverability factors and auxiliary parameters used by the storage model.

`dhw_generator_config` is used when DHW is supplied by a separate generator. It defines the generator size and basic technology data for the DHW branch.

`distribution_15316_3_config.dhw` defines the DHW distribution network, including operation mode, pipe sections, DHW temperature, return temperature difference, hydraulic data, pump control and recoverable auxiliary fraction.

## EN 15316-4-1 - Generation System

In `new_building.py`, generation is set with `generation_calculation_mode = 'boiler_15316_4_1'`, so the calculation branch follows the EN 15316-4-1 boiler logic. In this setup, the generator is not treated as a simple power block, but as a component with variable efficiency, nominal and part-load losses, electrical auxiliaries and operating conditions linked to supply and return temperature. The parameters in `boiler_generation_config` describe boiler size, test values, jacket and standby losses, the recoverable share of auxiliaries and condensing behavior. In other words, generation applies EN 15316-4-1 to transform thermal demand into generator input energy, accounting for real performance and not only for the useful load requested by the network.

## EN 15316-4-1 Generation Parameters and EN 15316-1 Interface

This section collects the inputs that `new_building.py` uses for EN 15316-4-1 generation and, partly, for the system interface managed by `iso_15316_1.py`. The parameters are not all read from the same sub-block: some are used directly by the generator, others describe the thermal chain between distribution and generation.

### Direct EN 15316-4-1 Inputs - Detailed

The following parameters are used directly by the generation block and the boiler.

`full_load_power: 27` is the generator full-load power, expressed in kW. It represents the main size of the machine and is the reference for efficiency, loss and operating-limit calculations.

`max_monthly_load_factor: 100` is the maximum allowed monthly load factor, expressed as a percentage. It limits generator operation relative to its nominal power under aggregated conditions.

`tH_gen_i_ON: 1` is the generator operating time in the step, in hours. For hourly calculation it is `1`, but the parameter remains explicit to preserve consistency with the model time step.

`auxiliary_power_generator: 0` is the percentage share of generator auxiliaries relative to produced thermal energy. It represents electrical consumption linked to generator operation.

`fraction_of_auxiliary_power_generator: 40` is the percentage share of generator thermal losses that the model considers recoverable.

`generation_calculation_mode: 'boiler_15316_4_1'` activates the EN 15316-4-1 calculation branch. This means generation is not treated with a simplified formula, but with the boiler-generator model.

`boiler_generation_config` contains the boiler-specific data used by the EN 15316-4-1 model.

`boiler_type: 'condensing'` identifies the generator type. In the condensing case, efficiency can benefit from latent heat recovery under suitable temperature conditions.

The allowed `boiler_type` values in the model are:

- `standard`: conventional boiler
- `low_temperature`: low-temperature boiler
- `condensing`: condensing boiler
- `biomass_log`: log biomass generator
- `biomass_pellet`: pellet biomass generator

The boiler type determines the calculation case used by the model and therefore how efficiency, losses and operating conditions are handled. If a different value is set, the code raises an error because the type is not included in the generator set supported by `generation_15316_4_1.py`.

`fuel_type: 'natural_gas'` indicates the fuel used by the generator.

In the current module, `fuel_type` does not have a closed list of allowed values like `boiler_type`. However, `new_building.py` applies a compatibility check between `boiler_type` and `fuel_type` to avoid physically incoherent configurations.

The rule adopted in the file is unambiguous for each boiler type:

- `standard` -> `natural_gas`
- `low_temperature` -> `natural_gas`
- `condensing` -> `natural_gas`
- `biomass_log` -> `wood_log`
- `biomass_pellet` -> `wood_pellet`

If `fuel_type` does not match the expected value for the selected `boiler_type`, the program stops the calculation with an explicit error.

| `boiler_type` | Allowed `fuel_type` | Required `efficiency_table` / `loss_table` | Reading note |
|---|---|---|---|
| `standard` | `natural_gas` | `standard` section with `eta_Pn_test_pct`, `eta_Pint_test_pct`, `theta_test_Pn_C`, `theta_test_Pint_C`; `loss_table.standard` with `P_gen_ls_P0_W` | Conventional boiler, without condensing effect |
| `low_temperature` | `natural_gas` | `low_temperature` section with the same test parameters; `loss_table.low_temperature` with `P_gen_ls_P0_W` | Low-temperature generator, more efficient than standard but not condensing |
| `condensing` | `natural_gas` | `condensing` section with the same test parameters; `loss_table.condensing` with `P_gen_ls_P0_W` | Condensing boiler, with possible efficiency increase below dew point |
| `biomass_log` | `wood_log` | `biomass_log` section with generator performance data; the model still requires a table coherent with the selected case | Log wood generator |
| `biomass_pellet` | `wood_pellet` | `biomass_pellet` section with generator performance data; the model still requires a table coherent with the selected case | Pellet generator |

In practice, every valid configuration must respect three consistency levels:

- `boiler_type` must belong to the set supported by the module
- `fuel_type` must be compatible with the selected `boiler_type`
- `efficiency_table` and `loss_table` must contain the key corresponding to the selected generator type

If one of these three levels is not respected, the calculation does not start and `new_building.py` raises an explicit error.

`rated_power_kW: 27.0` is the nominal boiler power. It must be coherent with `full_load_power`, because it represents the same machine size in the generator-specific block.

`intermediate_load_fraction: 0.30` indicates the intermediate load fraction used to interpolate efficiency between nominal and reduced-load operation. In the module, the value must be strictly between `0.0` and `1.0`: it cannot be `0` and cannot be `1`.

`eta_Pn_test_pct: 98.0` is the measured or assumed efficiency at the nominal-load test point.

`eta_Pint_test_pct: 106.0` is the measured or assumed efficiency at the intermediate-load test point.

`theta_test_Pn_C: 60.0` and `theta_test_Pint_C: 40.0` are the test temperatures associated with nominal load and intermediate load respectively.

`f_corr_pct_per_K: 0.04` is the efficiency correction coefficient with temperature, expressed as percentage per kelvin. In the model it is non-negative. There is no rigid maximum imposed by the code, although very high values would make efficiency too sensitive to temperature and unrealistic.

`P_gen_ls_P0_W: 100.0` is the generator loss power at standstill, i.e. standby or inactive-circuit losses. It must be greater than or equal to `0 W`.

`P_aux_on_W: 80.0` is the generator auxiliary power during operation. The minimum allowed value is `0 W`; in practice it should remain coherent with generator size and type.

`P_aux_off_W: 5.0` is the generator auxiliary power when the machine is off. The minimum allowed value is `0 W`. It represents residual consumption during inactivity or standby.

`f_jacket: 0.40` describes the share of generator jacket losses. In the model the value must be between `0.0` and `1.0`: `0.0` means no loss is assigned to the jacket, while `1.0` means all generator losses are considered jacket losses in the recovery chain.

`f_location: 1.0` represents the generator location factor, namely the installation context and therefore the share of losses recoverable by the surrounding environment. This value must also be between `0.0` and `1.0`.

`f_aux_recoverable: 0.75` is the share of auxiliary or thermal losses considered recoverable in the zone balance. In the code the value must be between `0.0` and `1.0`.

`dew_point_C: 55.0` is the reference dew-point temperature used for condensing behavior. The module does not impose a rigid maximum, but the value must be positive and coherent with the boiler temperature regime.

`condensing_gain_pct: 11.0` is the percentage efficiency gain associated with condensing operation under favorable conditions. The code does not impose a rigid maximum, but the parameter must be non-negative.

`efficiency_table` and `loss_table` are support tables for boiler cases. In the case shown, the efficiency table contains the test values for the `condensing` type, while the loss table contains the base or off-state loss value.

`boiler_location: 'inside_heated'` means that the boiler is located in a heated space. This affects the treatment of some losses and their possible recoverability.

The possible `boiler_location` values are:

- `inside_heated`: boiler installed in a heated room
- `adjacent_unheated`: boiler installed in an adjacent unheated room
- `outside_building`: boiler installed outside the building

If a different value is set, the module raises an error because the position is not recognized by the EN 15316-4-1 calculation.

### Note on `full_load_power` and `rated_power_kW`

`full_load_power` and `rated_power_kW` indicate the same physical generator size, but at two different model levels. `full_load_power` is the interface value used by the overall HVAC system, while `rated_power_kW` is the value read by the EN 15316-4-1 module for the boiler calculation. In `new_building.py`, these two values must almost exactly match; an automatic check blocks the simulation if the difference exceeds a very tight tolerance.

### Interface Inputs with EN 15316-1 - Simplified

The following parameters do not describe the boiler itself, but connect the generation block with the rest of the system modeled by `iso_15316_1.py`.

`generator_circuit: 'independent'` means that the primary circuit is independent from the secondary circuit. In this setup the generator can have its own supply-temperature strategy and hydraulic circuit. The actually supported values are `direct` and `independent`.

`gen_flow_temp_control_type: 'Type A - Based on outdoor temperature'` defines the generator-side supply temperature control logic. In `Type A`, the primary temperature follows a climatic curve based on outdoor temperature.

The options considered by the model are:

- `Type A - Based on outdoor temperature`: generator supply temperature follows an outdoor-temperature climatic curve
- `Type B - Based on indoor temperature`: supply control depends on zone indoor temperature
- `Type C - Fixed flow temperature`: supply temperature remains substantially fixed, except for corrections introduced by other system parameters

In practical terms:

- `Type A` is the natural choice when the generator must follow outdoor climate, for example in a system with a defined climatic curve
- `Type B` makes sense when control is driven by indoor comfort and a room-temperature reference
- `Type C` is suitable when the supply temperature is kept at an almost constant setpoint

If a different value is set, the model has no coherent interpretation logic and the calculation must be considered invalid.

`gen_outdoor_temp_data` contains the parameters of the generator climatic curve. `theta_ext_min_gen` and `theta_ext_max_gen` indicate the reference outdoor temperatures, while `theta_flw_gen_min` and `theta_flw_gen_max` define the generator supply-temperature range.

`speed_control_generator_pump: 'variable'` describes the generator pump control type. In this case the pump speed varies according to demand or operating conditions.

The options actually handled by the model are:

- `variable`: the pump works at variable speed and adapts to the load
- `deltaT_constant`: the flow rate is controlled to keep the primary-circuit temperature difference more stable

In a residential system with climatic control and variable load, `variable` is usually the most natural choice. `deltaT_constant` makes sense when the primary circuit should remain close to a predefined temperature difference.

`generator_nominal_deltaT: 20` is the nominal temperature difference of the primary circuit, expressed in `°C`. It is used to convert thermal power into primary flow rate and to keep the generator hydraulic balance coherent.

`mixing_valve_delta: 2` is the additional temperature difference introduced by the mixing valve, expressed in `K` or equivalently in `°C` as a temperature difference. In practice it represents a small temperature difference between the generator circuit and user circuit. Plausible values are generally about `1-5 K` for light mixing, or higher if the valve introduces a stronger thermal correction.

The optional parameters `theta_HW_gen_flw_set` and `theta_HW_gen_ret_set`, if enabled, allow the generator-side supply and return temperatures to be fixed directly. In `new_building.py` they are left commented to keep climatic control.

`efficiency_model: 'simple'` activates the simple efficiency model in the generator calculation block.

The options considered by the model are:

- `simple`: uses a direct simplified efficiency model
- `parametric`: uses a more detailed parametric formulation when the required parameters are available

`calc_when_QH_positive_only: False` means that the calculation may also be executed when the thermal load is not positive.

`off_compute_mode: 'full'` specifies the model behavior when thermal load is zero or negligible.

The currently supported options are:

- `idle`: returns an off condition with the minimum required detail
- `temps`: calculates only temperatures or essential thermal variables
- `full`: still performs the full calculation at zero load

When the boiler is modeled with EN 15316-4-1, these interface inputs do not describe generator efficiency; they connect the generator correctly to the rest of the system.

**Mandatory or almost always required**

- `generator_circuit`
- `gen_flow_temp_control_type`
- `gen_outdoor_temp_data` if `Type A` climatic control is used
- `generator_nominal_deltaT`
- `speed_control_generator_pump`

**Conditioned by the control strategy**

- `gen_outdoor_temp_data`: required only if control depends on outdoor temperature
- `theta_HW_gen_flw_set` and `theta_HW_gen_ret_set`: required only if supply and return setpoints are imposed directly
- `mixing_valve_delta`: useful if the circuit includes mixing between generator branch and user branch

**Optional or refinement parameters**

- `efficiency_model`
- `calc_when_QH_positive_only`
- `off_compute_mode`

These inputs do not build the main physical boiler connection, but modify the calculation mode or behavior in limit cases.
