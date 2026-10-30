# StorageSystemCalculator

`StorageSystemCalculator` implements the EN 15316-5 storage module for heating and domestic hot water.

## Operating modes

### `dynamic_calculation = False`

This is the current default behavior.

The storage is treated as a single volume controlled at a constant set temperature during each timestep.
The calculator computes:

- standing losses;
- recoverable and non-recoverable losses;
- auxiliary electricity for input/output pumps;
- thermal energy required at storage inlet.

### `dynamic_calculation = True`

This mode adds a simplified dynamic tank model to inspect:

- internal temperature drop after DHW draw-off;
- coarse internal stratification;
- time required to recharge the tank back to setpoint.

The current implementation uses a two-layer representation:

- top layer: hot water available to the user;
- bottom layer: colder incoming water.

The top and bottom temperatures are updated hour by hour from:

- hot water draw-off;
- tank standing losses;
- heat injected by the generator.

The DHW mode is enabled when the average tank temperature falls below `dhw_switch_temperature_C`:

```text
T_tank < dhw_switch_temperature_C
```

The default threshold is `45 C`. The tank is then reheated to `55 C` setpoint.

## Required inputs

### Common inputs

These are used in both modes:

- `storage_volume_l`
- `storage_setpoint_C` or `set_temperature_C`
- `output_temperature_C`
- `ambient_temperature_C`
- `standby_loss_coefficient_W_K` or `standby_loss_kWh_per_day`
- `storage_type` or `H_sto_ls_W_K`
- `time_step_hours`
- `demand_unit`

### Additional inputs for `dynamic_calculation = True`

The dynamic mode needs the following extra inputs:

- `storage_height_m`
- `sensor_height_m`
- `cold_inlet_temperature_C`
- `stratification_efficiency`
- `dhw_switch_temperature_C`, default `45.0`
- `recharge_power_kW`
- `recharge_cop`, default `3.0`
- optionally `nominal_power_kW` as fallback for recharge power

## Equations

### 1. Standing losses

For both modes:

```text
Q_sto,ls = H_sto,ls * f_adapt * f_conn * (T_set - T_amb) * Δt / 1000
```

where:

- `H_sto,ls` is the standby-loss coefficient in `W/K`;
- `f_adapt` is the adaptation factor;
- `f_conn` is the connection-loss factor;
- `Δt` is the timestep in hours.

### 2. Pump auxiliary electricity

The auxiliary operating time is estimated from the energy processed by the pump:

```text
t_pump = Q / (1.15 * Vdot * ΔT)
W_aux = t_in * P_in + t_out * P_out
```

where:

- `Q` is the thermal energy moved in `kWh`;
- `1.15` is the water heat-capacity-density factor in `kWh/(m3*K)`;
- `Vdot` is the flow rate in `m3/h`;
- `ΔT` is the pump design temperature difference in `K`.

### 3. Dynamic tank temperature update

The dynamic mode uses a simplified energy balance:

```text
T_new = (E_hot + E_cold + E_recharge) / C_tank
```

with:

```text
C_tank = m_tank * cp
```

and the draw-off induced temperature drop approximated by:

```text
T_mix = (V_hot * T_old + V_draw * T_cold) / (V_hot + V_draw)
```

### 4. Recharge time

The tank is updated with a simplified energy balance:

```text
T_after_draw = T_before_loss - Q_draw / (m_tank * cp)
Q_recharge = max(0, C_tank * (T_set - T_tank))
t_recharge = min(Δt, Q_recharge / P_recharge)
```

where:

- `P_recharge` is the available recharge power of the generator;
- `Q_recharge` is limited by `P_recharge * Δt` in each timestep.

The electric energy assigned to DHW generation is estimated as:

```text
W_DHW = Q_recharge / COP_recharge
```

where `COP_recharge` is given by `recharge_cop`.

## Output columns in dynamic mode

When `dynamic_calculation = True`, the result table can include:

- `theta_W_sto_top_C`
- `theta_W_sto_bottom_C`
- `Q_W_sto_recharge_time_h`
- `Q_W_sto_recharge_energy_kWh`
- `Q_W_sto_stratification_K`
- `V_W_sto_before_dhw_mode_l`
- `V_W_sto_remaining_l`
- `dhw_dhw_mode_flag`

## Notes

- The dynamic mode is intentionally simplified and should be interpreted as an engineering approximation.
- It is useful to inspect temperature drop and recharge behavior after DHW use.
- For detailed stratified storage physics, a multi-node model would be needed.
