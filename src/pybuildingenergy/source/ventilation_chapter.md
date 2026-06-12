# Chapter: Ventilation, Infiltration, and Internal Gains

This module collects the thermal building blocks that turn air exchange into heat transfer and internal loads into usable simulation inputs. In the architecture of `pyBuildingEnergy`, it sits at the boundary between the physical description of the building and the energy balance solved by the ISO 52016 workflow.

The module is intentionally compact, but its role is central. Ventilation changes the effective heat transfer coefficient of a zone, infiltration adds uncontrolled losses, and internal gains provide the heat released by occupants, appliances, and lighting. These three elements are not separate details: together they define how a building responds to weather, occupancy, and control strategy.

## 1. The Purpose of the Module

The `ventilation.py` module supports three recurring tasks:

1. compute ventilation heat transfer coefficients under different assumptions;
2. estimate internal sensible gains from standardized usage classes;
3. evaluate transmission and ventilation losses for adjacent unconditioned zones.

The implementation is designed to be flexible enough for several modeling philosophies. A user can choose a physically inspired airflow model based on wind and stack effects, a simplified occupancy-driven model, a fixed ventilation conductance, or infiltration formulations closer to EnergyPlus and Sherman-Grimsrud style approaches.

## 2. `h_natural_vent`

The module begins with a small dataclass:

```python
@dataclass
class h_natural_vent:
    H_ve_nat: np.ndarray
```

This class is a lightweight container for natural ventilation results. It does not introduce behavior; instead, it gives a typed shape to the output of calculations that may be reused elsewhere in the codebase. In practice, it acts as a simple envelope around a NumPy array holding the ventilation heat transfer coefficient.

Its presence reflects a common pattern in the project: the code favors simple data containers when the primary need is to store a result in a structured way without attaching a larger object model.

## 3. `VentilationInternalGains`

The main class in the module is `VentilationInternalGains`. It is initialized with a `building_object`, which gives the class access to the geometry and the parameters needed by the airflow and gain calculations.

```python
class VentilationInternalGains:
    def __init__(self, building_object):
        self.building_object = building_object
```

The class serves as a thin computational wrapper. It does not model time evolution on its own; instead, it provides methods that are called by the rest of the simulation engine whenever a zone-level ventilation coefficient or internal gain value is needed.

### 3.1 `heat_transfer_coefficient_by_ventilation`

This is the core method of the module:

```python
heat_transfer_coefficient_by_ventilation(
    building_object, Tz, Te, u_site, Rw_arg_i=None, c_air=1006,
    rho_air=1.204, C_wnd=0.001, C_st=0.0035, rho_a_ref=1.204, altitude=None,
    type_ventilation="temp_wind", flowrate_per_area=1.4, custom_Hve_k_t=3,
    flowrate_person=None
)
```

Its goal is to return `Hve_k_t`, the ventilation heat transfer coefficient in `W/K`. In thermal simulation terms, this coefficient turns an airflow rate into a heat exchange capacity by multiplying it with the air density and specific heat.

The method accepts:

- indoor temperature `Tz`;
- outdoor temperature `Te`;
- site wind speed `u_site`;
- optional window opening ratios `Rw_arg_i`;
- several ventilation model choices through `type_ventilation`.

The method also includes backward compatibility logic, which is important in a codebase that evolves over time. The parameter `flowrate_person` is preserved as an alias for `flowrate_per_area`, so older call sites continue to work without modification.

### 3.2 Parameter resolution strategy

One of the most important design choices in the method is the parameter lookup order. The helper `_vent_param` searches first in the top-level `building_object` dictionary and then in `building_object["building_parameters"]["ventilation"]`. This means the function can be called with either:

- a zone-like dictionary that directly contains ventilation keys;
- a full building dictionary that stores ventilation settings in the standard nested location.

This makes the method resilient to different calling conventions. It reduces boilerplate in the rest of the code and allows global defaults to be overridden locally.

### 3.3 The `temp_wind` model

When `type_ventilation == "temp_wind"`, the method evaluates a natural ventilation formulation based on the combined effect of wind and stack pressure, following the EN 16798-7 logic documented in the source comments.

The implementation:

- collects all transparent surfaces and interprets them as windows;
- extracts opening geometry from `height`, `width`, and `parapet`;
- computes the useful stack height `hw_st`;
- adjusts the reference air density if altitude is provided;
- evaluates the effective open area of all windows;
- computes the airflow rate from the strongest of the wind and stack terms;
- transforms the airflow into `Hve_k_t`.

This branch is the most geometric and physically detailed one in the module. It depends on the building envelope definition, especially on the transparent surfaces, which is why missing window dimensions produce a warning and are skipped rather than causing a hard failure.

The model is robust in a practical sense: if no usable windows are found, it returns zero ventilation contribution instead of breaking the simulation.

### 3.4 The `occupancy` model

The occupancy branch is intentionally simpler. Here the airflow is driven by the number of people or, more precisely, by a specific airflow rate per unit area that is interpreted in a legacy-compatible way.

The method:

- reads the ventilation rate associated with occupancy;
- multiplies it by the zone floor area;
- converts the result into `m3/s`;
- transforms that into `Hve_k_t`.

This model is useful when ventilation is not being resolved through openings and wind but instead approximated through use intensity. It is less detailed than the `temp_wind` model, but it is easier to calibrate when only schedule- and occupancy-level information is available.

### 3.5 The EnergyPlus-style infiltration model

The branch `eplus_infiltration_ext_area` reproduces the behavior of an EnergyPlus `DesignFlowRate` infiltration model based on exterior area. Conceptually, it computes:

```text
Q_inf = q_design * (A + B*|ΔT| + C*V + D*V²)
```

where the design flow is scaled by the exterior surface area and by coefficients representing constant leakage, temperature dependence, wind dependence, and squared wind dependence.

Several implementation details are worth highlighting:

- the exterior area can include only outdoors-facing surfaces or also ground-related boundaries, depending on the selected mode;
- transparent surfaces can optionally be included;
- wind speed can be reduced by a calibration factor;
- the final airflow is multiplied by a schedule factor;
- the result is converted into a heat transfer coefficient using air properties.

This makes the method suitable for calibration workflows where an EnergyPlus-like infiltration model is used as a benchmark or reference behavior.

### 3.6 The Sherman-Grimsrud-like model

The `sherman_grimsrud_like` branch is a leakage-based formulation. Instead of using exterior area directly, it works with an effective leakage area and two coefficients describing stack and wind sensitivity.

The method evaluates:

- the effective leakage area;
- the temperature difference between indoor and outdoor air;
- the wind speed;
- a square-root term that combines stack and wind contributions.

This formulation is useful when infiltration is better understood as leakage through the envelope rather than as a flow proportional to exposed area.

### 3.7 The `custom` branch

The simplest path is `custom`. In this case the method does not derive the ventilation coefficient from physics or geometry; it simply returns a fixed `Hve_k_t`.

This is a practical escape hatch. It lets the caller impose a prescribed ventilation conductance when the desired behavior is known externally or when the simulation needs a controlled reference value.

## 4. Affine Ventilation Boundary (ISO 52016-1 §6.5.10)

### 4.1 Mathematical foundation

ISO 52016-1 §6.5.10 inserts ventilation into the zone energy balance through a conductance
term and a source term.  The two dataclasses and one resolver function in this module
are an algebraically equivalent implementation of that formulation.

Each airflow stream *k* (infiltration, mechanical supply, summer purge, etc.) contributes:

```
Q_k = H_k · (T_source,k − T_zone)   [W, positive = heat into zone]
```

Summing over all active streams gives the aggregate boundary passed to the solver:

```
H_ve = Σ H_k          [W/K]   conductance
S_ve = Σ H_k · T_source,k   [W]    weighted source term

Q_ve = H_ve · T_zone − S_ve  (positive = heat leaving zone)
```

The solver receives `(H_ve, S_ve)` and inserts them directly into the zone matrix without
adding a supply-air temperature node.  The equivalent supply temperature `S_ve / H_ve`
is available as a diagnostic but must not replace the pair in the solver.

All existing single-stream configurations (any `ventilation_type` value) produce
`S_ve = H_ve · T_outdoor`, so existing result columns retain their meaning; the new
`S_ve` and `Q_ve` diagnostic columns are additive.

### 4.2 `VentilationStream`

A frozen dataclass representing one additive airflow element:

```python
@dataclass(frozen=True)
class VentilationStream:
    name: str                           # unique label
    heat_transfer_coefficient_w_k: float  # H_k [W/K], must be ≥ 0 and finite
    source_temperature_c: float         # T_source,k [°C], must be finite
    category: str = "outdoor_air"       # "outdoor_air" | "supply" | other
```

`__post_init__` enforces finite, non-negative `H_k` and finite `source_temperature_c`.
Zero conductance is allowed (stream is inactive) but negative conductance is rejected.

### 4.3 `VentilationBoundary`

A frozen dataclass holding the complete set of active streams for one zone at one timestep:

```python
@dataclass(frozen=True)
class VentilationBoundary:
    streams: tuple[VentilationStream, ...]
```

Computed properties:

| Property | Formula | Unit |
|----------|---------|------|
| `heat_transfer_coefficient_w_k` | `Σ H_k` | W/K |
| `source_term_w` | `Σ H_k · T_source,k` | W |
| `equivalent_supply_temperature_c` | `S_ve / H_ve` (None when H_ve = 0) | °C |

An empty `streams` tuple is valid and represents zero ventilation (`H_ve = 0`, `S_ve = 0`).

### 4.4 `resolve_ventilation_boundary`

The main entry point called by the solver at each timestep:

```python
resolve_ventilation_boundary(
    building_object,
    zone_temperature_c,
    outdoor_temperature_c,
    wind_speed_m_s,
    profile_multiplier=1.0,
    component_multipliers=None,
    extra_streams=(),
    zone_volume_m3=None,
    altitude_m=None,
    c_air=1006.0,
    rho_air=1.204,
    ahu_outputs_collector=None,
) -> VentilationBoundary
```

Dispatch logic:

1. If `building_parameters.ventilation.components` is present, each entry becomes an
   independent `VentilationStream` via `_resolve_component_streams`.  Components absent
   from `component_multipliers` default to 1.0 (full capacity), so infiltration streams
   remain active when the mechanical schedule is off.
2. Otherwise the legacy scalar path is used: `ventilation_type` is resolved through
   `VentilationInternalGains.heat_transfer_coefficient_by_ventilation` and scaled by
   `profile_multiplier`, producing one outdoor-air stream.  All five existing
   `ventilation_type` values are supported.
3. `extra_streams` (e.g. a summer night purge) are appended after dispatch.

Raises `RuntimeError` or `ValueError` on invalid configuration; does not silently
substitute zero for a non-finite or negative `H_k`.

### 4.5 `components` configuration schema

The `components` key under `building_parameters.ventilation` enables multi-stream
ventilation.  Each entry is a dict:

```python
{
    "name": "infiltration",               # required, unique label
    "ventilation_type": "constant_ach",   # "constant_ach", "prescribed" or "mechanical_supply"
    "profile": "ventilation_profile",     # optional — names a profile_df column
    # type-specific keys, e.g.:
    "air_changes_per_hour": 0.3,
}
```

The `"prescribed"` type accepts `heat_transfer_coefficient_w_k` and
`source_temperature_c` directly, bypassing the scalar airflow models.  This is the
primary mechanism for modelling a supply-air stream at a fixed temperature (e.g. an
AHU delivering 18 °C supply air):

```python
{
    "name": "ahu_supply",
    "ventilation_type": "prescribed",
    "heat_transfer_coefficient_w_k": 815.0,
    "source_temperature_c": 18.0,
    "profile": "ventilation_profile",   # AHU off outside operating hours
}
```

A `components` list may combine infiltration (no profile → always active) and mechanical
supply (with profile → follows schedule), which is the practical purpose of the additive
stream model and the contract used by the EN 16798-5-1 AHU module (section 4.6).

When a component has a `profile`, the profile name must be resolvable by the
active solver path.  For `mechanical_supply`, a missing or unsupported profile
raises `ValueError` by default because treating a misspelled AHU schedule as
full-flow operation can materially change the result.  To preserve that fallback
intentionally, set `"missing_profile_policy": "warn"` on the component; the
solver will emit a warning and use multiplier `1.0`.  Non-AHU components retain
the warning fallback by default for compatibility.

### 4.6 The `mechanical_supply` component (EN 16798-5-1 sensible AHU)

The `"mechanical_supply"` type replaces the externally known supply condition of
`"prescribed"` with a physics-based per-timestep AHU calculation implemented in
`ventilation_16798_5_1.py`: sensible heat recovery, modulating bypass, frost
protection by exhaust-temperature limit, capacity-limited sensible heating and
cooling coils, and fan electricity/fan heat. The calculation is solver-independent;
its result enters the zone balance only as one additive stream with
`H_k = rho_air · cp_air · q_supply` and `T_source,k` equal to the delivered supply
temperature, so the §6.5.10 formulation above is unchanged.

```python
{
    "name": "ahu",
    "ventilation_type": "mechanical_supply",
    "supply_flow_m3_h": 3600.0,                  # required; extract defaults to supply
    "sensible_heat_recovery_efficiency": 0.784,  # required, in [0, 1]
    "supply_temperature_setpoint_c": 18.0,       # required for the default "fixed" mode
    # optional, with defaults:
    # "supply_temperature_mode": "fixed" | "outdoor_compensated" | "extract_compensated",
    # "supply_temperature_points": [[-20, 19], [15, 17]],   # for compensated modes
    # "heat_recovery_control": "modulating_bypass",
    # "frost_control": "exhaust_limit", "frost_exhaust_limit_c": -5.0,
    # "heating_coil_max_power_w": None, "cooling_coil_enabled": False,
    # "supply_fan_specific_power_w_per_m3_s": 0.0, "fan_performance_model": "en16798_5_1",
    "profile": "ventilation_profile",
}
```

Three behaviours of this component differ from the simpler types and are
deliberate conventions:

1. **Lagged extract-air coupling.** The extract-air temperature entering the
   heat recovery is the zone *air-node* temperature from the previous timestep
   (not the operative temperature). This avoids the algebraic loop
   T_zone → T_extract → heat recovery/coils → T_supply → T_zone and preserves
   the single-pass solver workflow.
2. **Profile fractions are flow fractions.** A fractional profile value scales
   the operating airflow continuously (reduced fan speed, with fan power
   following the EN 16798-5-1 part-load relations). It is *not* ON/OFF
   duty-cycle averaging. Values outside [0, 1] raise `ValueError`.
3. **System energy is diagnostic, not a zone gain.** Coil and fan quantities
   are exported as per-component hourly columns (`Q_ahu_coil*`, `Q_ahu_cool*`,
   `Q_ahu_hr`, `P_ahu_fan`, `T_ahu_sup*`, `q_sup_m3h`, `q_ext_m3h`,
   `ahu_bypass`, `ahu_frost`) but are not added to the zone energy balance —
   the ventilation term already uses the conditioned supply temperature, so
   adding coil energy on the zone side would double-count it.

The initial scope is a sensible, balanced-airflow subset of EN 16798-5-1 with
fixed air properties (`rho·cp = 1211.224 J/(m³·K)`, the existing module
convention; no altitude correction). For the full implemented / simplified /
not-included breakdown, the configuration-key reference, the diagnostic-column
schema and the standards context, see
[`docs/ventilation_ahu_audit.html`](../../../docs/ventilation_ahu_audit.html).
The `"prescribed"` type remains available as the intermediate option when the
supply temperature is known externally.

## 5. `internal_gains`

The second major method is `internal_gains`.

```python
internal_gains(
    self, building_type_class, a_use, unconditioned_zones_nearby=False,
    list_adj_zones=None, Fztc_ztu_m: float=1, b_ztu: float=1,
    h_occup: float=1, h_app: float=1, h_light: float=1,
    h_dhw: float=1, h_hvac: float=1, h_proc: float=1
)
```

Its job is to compute the sensible internal heat gains `Phi_int_z_t` for a zone.

The method starts from a standardized table of gains indexed by `building_type_class`. This gives the model a baseline consistent with ISO-derived schedules and typical building usage categories such as offices, schools, residential apartments, detached houses, kindergartens, and department stores.

The calculation proceeds in layers:

1. read default occupant, appliance, and lighting gains from the table;
2. guard against invalid or missing values;
3. apply optional overrides stored in `building_object["internal_gains"]`;
4. combine the gains using weighting factors like `h_occup`, `h_app`, and `h_light`;
5. multiply the result by the useful area `a_use`.

This design is important because it separates the baseline assumptions from the project-specific tuning. In other words, the module knows the canonical values, but it still allows a higher-level model to replace them when the building under study has more precise data.

### 5.1 Local overrides

The override mechanism is straightforward: if the building object contains an `internal_gains` list, each item can replace one of the default full-load values.

That makes the function compatible with project workflows in which internal gains are specified explicitly rather than inferred from the usage class alone.

### 5.2 Unconditioned adjacent zones

The method also contains an optional branch for nearby unconditioned zones. In that case, internal gains can be redistributed using coupling factors such as `Fztc_ztu_m` and `b_ztu`.

Even though this part of the function is more specialized, the underlying idea is clear: not all internal gains remain confined to the conditioned space where they originate. Some can be transferred to adjacent zones and should therefore be handled in the energy balance.

## 6. `transmission_heat_transfer_coefficient_ISO13789`

The module ends with a helper function for unconditioned-adjacent-zone transmission:

```python
transmission_heat_transfer_coefficient_ISO13789(adj_zone, n_ue=0.5, qui=0)
```

This function estimates the heat transfer coefficient of a thermally unconditioned zone using the ISO 13789 decomposition:

- direct transmission through envelope components;
- transmission through ground-related paths;
- transmission through the unconditioned space;
- transmission to adjacent buildings.

The function also computes an adjustment factor `b_ztu_m`, representing the share of the unconditioned zone heat transfer that goes to the exterior.

Conceptually, this helper is a bridge between the detailed zone formulation and the simplified representation of buffer spaces. It turns geometry, transmittance, and ventilation assumptions into an aggregate thermal conductance that can be used in the broader simulation.

## 7. Design Style of the Module

What makes this module useful is not just the formulas it implements, but the way it is structured:

- it exposes a small number of entry points;
- it supports multiple ventilation paradigms;
- it keeps backward compatibility in parameter naming;
- it accepts both global and local configuration styles;
- it uses warnings instead of hard failures where a missing surface element should not stop the simulation.

This makes the module well suited to engineering workflows where input data are incomplete, heterogeneous, or gradually refined over time.

## 8. Practical Reading of the Code

If one reads the module as part of a thermal simulation pipeline, its logic becomes easy to summarize:

- `temp_wind` models airflow through windows under wind and stack pressure;
- `occupancy` links ventilation to use intensity;
- `eplus_infiltration_ext_area` provides a calibration-friendly infiltration model;
- `sherman_grimsrud_like` adds a leakage-oriented alternative;
- `custom` lets the caller impose a prescribed coefficient;
- `internal_gains` supplies the sensible heat released inside the zone;
- `transmission_heat_transfer_coefficient_ISO13789` closes the loop for buffer spaces and adjacent unconditioned zones.

Taken together, these functions define how the building exchanges heat with the air around it and with the people and equipment inside it. In a simulation context, that is enough to strongly influence both heating and cooling demand.

## 9. Closing Note

The module is small, but it carries a disproportionate share of the physical realism of the model. Ventilation and internal gains are often treated as secondary details in simplified building analyses, yet they are among the first mechanisms that determine whether a zone remains comfortable or drifts away from setpoint.

For that reason, `ventilation.py` should be read not as a utility file, but as a compact statement of the building's interaction with air, occupants, and weather.
