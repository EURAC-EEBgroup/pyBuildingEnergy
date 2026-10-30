# HeatingSystemCalculator

A Python class for simulating **multi-block heating systems** (control - emission → distribution → generation) in hourly resolution, following the logic of **ISO 15316-1**.

The model supports:
- 4 **emission circuit types** (C.2–C.5), Annex C of ISO 15316-1
- 3 **primary flow control strategies** (Type A–C), Annex D of ISO 15316-1. Here we added the possibility to control the the primray flow setpoiint based on 3 additional control variables: 
    - 'Type A - Based on outdoor temperature': uses primary's own weather curve
    - 'Type B - Based on demand'            : follows the secondary demand (θH_dis_flw)
    - 'Type C - Constant temperature'       : uses self.θHW_gen_flw_const
- detailed **heat loss and recovery** accounting,
- and full energy balance per time step.

## 🚀 Quick Start

```python
import pandas as pd
from heating_system_calculator import HeatingSystemCalculator

# 1️⃣ Instantiate with project parameters
calc = HeatingSystemCalculator({
    # --- EMISSION (secondary) ---
    'emitter_type': 'Floor heating',
    'nominal_power': 8,
    'emission_efficiency': 90,
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
    'selected_emm_cont_circuit': 0,     # 0=C.2, 1=C.3, 2=C.4, 3=C.5
    'mixing_valve': True,

    # --- GENERATION (primary) ---
    'full_load_power': 27,
    'generator_circuit': 'independent',
    'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',
    'gen_outdoor_temp_data': pd.DataFrame({
        "θext_min_gen": [-7],
        "θext_max_gen": [15],
        "θflw_gen_max": [60],
        "θflw_gen_min": [35],
    }, index=["Generator curve"]),

    'speed_control_generator_pump': 'variable',
    'generator_nominal_deltaT': 20,
    'efficiency_model': 'simple',

    # --- OPTIONS ---
    'calc_when_QH_positive_only': False,
    'off_compute_mode': 'full',
})

# 2️⃣ Load hourly data (CSV or DataFrame)
# Expected columns: Q_H (Wh or kWh), T_op (°C), T_ext (°C)
df = calc.load_csv_data("inputs_hourly.csv")

# 3️⃣ Run full hourly simulation
results = calc.run_timeseries()

# 4️⃣ Save or analyze
results.to_csv("results_output.csv")
print(results[['QH_gen_out(kWh)', 'EH_gen_in(kWh)', 'efficiency_gen(%)']].head())
```

---

## 📈 Plot Example

Visualize primary and secondary flow/return temperatures:

```python
import matplotlib.pyplot as plt

df = results
plt.figure(figsize=(10,6))
plt.plot(df.index, df['θX_gen_cr_flw(°C)'], label="Primary flow (generator)")
plt.plot(df.index, df['θH_dis_flw(°C)'], label="Secondary flow (distribution)")
plt.plot(df.index, df['θX_gen_cr_ret(°C)'], '--', label="Primary return")
plt.plot(df.index, df['θH_dis_ret(°C)'], '--', label="Secondary return")

plt.title("Heating System Temperatures – Primary vs Secondary")
plt.xlabel("Time")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```

---

## 📘 API Summary

### Emission (C.2–C.5)
| Method | Purpose |
|---------|----------|
| `calculate_common_emission_parameters()` | Compute shared parameters (power, nominal flow, deltas). |
| `calculate_type_C2()` | Constant flow and variable temperature. |
| `calculate_type_C3()` | Variable flow and constant temperature. |
| `calculate_type_C4()` | ON–OFF intermittent control. |
| `calculate_type_C5()` | Constant flow and variable heat exchange (by-pass). |

### Distribution
| Method | Purpose |
|---------|----------|
| `calculate_distribution()` | Compute losses, auxiliaries, recoverable heat, and inlet demand. |

### Generation (Primary)
| Method | Purpose |
|---------|----------|
| `calculate_generation()` | Full generator balance (energy, flow, ΔT, efficiency). |
| `_generator_flow_setpoint()` | Compute primary flow setpoint (Type A/B/C control). |
| `_efficiency_from_model()` | Evaluate generator efficiency (`simple`, `parametric`, or `manufacturer`). |

### System Execution
| Method | Purpose |
|---------|----------|
| `compute_step()` | Run one hourly step through all blocks. |
| `run_timeseries()` | Execute the full hourly time series. |
| `_idle_row()` / `_temps_row_when_off()` | Handle zero-load steps. |

---

## 🧾 Expected Outputs

| Column | Description |
|---------|--------------|
| `QH_gen_out(kWh)` | Thermal energy output of the generator. |
| `EHW_gen_in(kWh)` | Primary energy input (fuel/electricity). |
| `θH_dis_flw(°C)`, `θH_dis_ret(°C)` | Secondary (distribution) flow/return temperatures. |
| `θX_gen_cr_flw(°C)`, `θX_gen_cr_ret(°C)` | Primary (generator) flow/return temperatures. |
| `Q_w_dis_i_ls(kWh)` | Distribution losses. |
| `Q_w_dis_i_ls_rbl_H(kWh)` | Recoverable share of losses. |
| `βH_em(-)` | Emitter load factor. |

---

## ⚙️ Troubleshooting

- **Emission > Generator output:**  
  This can happen if distribution recoverable losses or auxiliaries offset net losses.

- **Efficiency > 100 %:**  
  Correct when using efficiency on **LHV/PCI** for condensing boilers (up to 110 %).

- **Primary temperature independent of secondary:**  
  Set  
  ```python
  'generator_circuit': 'independent',
  'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature'
  ```  
  or fix a constant setpoint:  
  ```python
  'θHW_gen_flw_set': 55
  ```

---

## 🧩 Example Post-Processing

```python
out = calc.run_timeseries()

# Energy totals
emission_in = out['QH_dis_i_req(kWh)'].sum()
generator_out = out['QH_gen_out(kWh)'].sum()
fuel_input = out['EHW_gen_in(kWh)'].sum()
eff_avg = 100 * generator_out / fuel_input

print(f"Emission in: {emission_in:.1f} kWh")
print(f"Generator out: {generator_out:.1f} kWh")
print(f"Overall efficiency: {eff_avg:.1f}%")
```
