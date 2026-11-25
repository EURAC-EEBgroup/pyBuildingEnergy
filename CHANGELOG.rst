Unreleased
----------

- New test ``tests/test_ashrae140_airside_ae.py``: ASHRAE 140-2023 Section 11 air-side
  cases run with the EN 16798-5-1 AHU step at steady state (recirculation, LOAD_COMP supply
  temperature, bypass factor 0, fan after the coils, zone humidity floating):
  - fan-coil AE101, AE103, AE104 and single-zone AE201, AE203-AE206 (return fan modelled as
    extract fan upstream of the recirculation);
  - AE226 / AE245 with the 100 % outdoor air their economizer reaches imposed by the test
    (the library has no economizer control logic).
  Coil loads within 2 % of the quasi-analytical solution (latent within 3.1 %), inside the
  spread of the example programs widened by 2 % (AE204 sensible -0.4 % and AE226 latent
  -1.5 % just outside it). The constant air density of EN 16798-5-1 gives a 4 % higher
  supply humidity ratio in AE101 and a return fan heat about 8 % lower than in the
  mass-based QAS. The reference results are copyright ASHRAE and are read from
  ``Standards/local_reference_data/ashrae140_2023_ae_qas.json`` (not distributed); without
  it the comparisons are skipped.

- Checked on Python 3.12.9 and 3.13.2 with fresh dependencies (numpy 2.5.5, pandas 3.1.1):
  the whole suite passes. ``test_hc_mode_outdoor_method_am2`` no longer writes into a
  read-only array (pandas 3 returns read-only arrays); library code is unchanged.

2.2.0 (2026-11-25)
------------------

EN 16798-5-1:2017 air handling unit: moisture processes and heat recovery corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New module ``ahu_moisture_16798_5_1`` implements:
  - heat recovery temperature ratio (Formulae 40-43): f_q (flow imbalance), f_v (air
    velocity), and f_n for rotary exchangers (D.1);
  - moisture recovery (44-45) with the Annex D corrections (D.5-D.10);
  - cooling and dehumidification (53-63, coil constants Table B.8);
  - humidification, steam or adiabatic (65-66, 82-83);
  - psychrometrics (D.6-D.7) and constants of Table 27.
- ``mechanical_supply`` components accept new optional keys. Without them the results
  are unchanged.
  - Heat recovery: ``heat_recovery_type`` (PLATE_CROSS, PLATE_COUNTER, ROT_NH, ROT_HYG,
    ROT_SORP, PUMP_CIRC, OTHER), ``heat_recovery_design_velocity_m_s``,
    ``heat_recovery_nominal_flow_m3_h``, ``rotor_speed_rpm``,
    ``moisture_recovery_nominal_efficiency``.
  - Humidity set-points and humidifier: ``supply_humidity_max_kg_kg``,
    ``supply_humidity_min_kg_kg``, ``humidifier_type``.
  - Cooling coil constants: ``cooling_coil_CC1``/``CC2``/``CC3`` and
    ``cooling_coil_bypass_factor``.
- AHU step outputs: supply humidity ratio, latent cooling coil power (included in the
  coil power), steam humidifier energy, humidification water, heat recovery ratio and
  moisture recovery efficiency.
- Zone moisture balance (EN ISO 52016-1 6.5.14, eq. 80-83), single-zone and multizone:
  - When a ``mechanical_supply`` component declares moisture data, each hour's supply
    humidity is computed by the AHU, using the previous hour's zone moisture content
    as extract air (``latent_ahu_moisture``: ``"auto"``, True or False).
  - New columns: ``X_ahu_sup_<ahu>``, ``Q_ahu_cool_lat_<ahu>``, ``Q_ahu_hum_<ahu>``,
    ``m_ahu_hum_water_kg_h_<ahu>``, ``X_sup_ss_kg_kg_da`` and ``X_int_ss_kg_kg_da``.
- Annex D follows the amendment as in the EPB Center spreadsheet (2026-05-12), see below.

- Moisture coupled to the heat balance (single-zone and multizone):
  - The zone moisture content is carried hour by hour inside the thermal simulation
    (eq. 80). The AHU of each hour gets the outdoor humidity ratio and the zone
    moisture content of the previous step.
  - So supply temperature, coils (the latent power takes coil capacity), reheat after
    dehumidification and the adiabatic humidifier are all in the heat balance of the
    same hour.
  - The moisture balance uses the supply humidity recorded in that hour. A zone with
    several AHUs that have moisture data uses their flow-weighted mix (before, only
    the first one).
  - New AHU diagnostic columns: ``X_ahu_sup``, ``Q_ahu_cool_lat``, ``Q_ahu_hum``,
    ``m_ahu_hum_water_kg_h`` (NaN or 0 for an AHU without moisture data).
  - Multizone adds ``X_sup_ahu_mix_<zone>``, ``q_sup_ahu_moist_m3_s_<zone>`` and
    ``X_int_ahu_<zone>``.
  - The ``ahu_causal`` control mode is coupled too. It now has the same latent balance
    columns and annual latent results as the standard engine (shared code), plus
    ``X_sup_ahu_mix``, ``q_sup_ahu_moist_m3_s`` and ``X_int_ahu``.
  - Single-zone: with a coupled AHU, the reported ``X_int_ss_kg_kg_da`` is the zone
    moisture the AHU saw. Before, the balance after the simulation started from a
    different initial state, so the two could differ in the first hours.
  - In multizone, the moisture state inside the simulation uses the same options as the
    final balance (zone value first, then the global option), including moisture
    absorption. The reported ``X_int_ss_kg_kg_da_<zone>`` is the moisture the AHU saw.
  - The moisture path is also active for an AHU with an enabled cooling coil and no
    moisture data, so the coil can condense. Results change for such AHUs: the latent
    power now takes part of the coil capacity.
  - The fallback that rebuilt the supply humidity after the simulation runs only when
    the coupled result is missing. Its "only the first mechanical_supply" warning no
    longer appears for coupled runs.

EN 16798-5-1 distribution, recirculation, frost and central AHU:

- Unbalanced supply and extract flows. The heat recovery ratio gets f_q (41) and the
  exhaust temperature follows the energy balance of the two streams. When the extract is
  larger than the supply, the difference enters the zone as outdoor make-up air (new
  stream ``<ahu>_makeup``).
- Distribution (6.4.2):
  - leakage of ducts (Table 10, ``duct_leakage_class``), AHU (Table 11,
    ``ahu_leakage_class``) and heat recovery (Table B.5, ``supply_fan_location`` /
    ``extract_fan_location``), or explicit factors. The fan flows follow (4a), (25)-(26).
  - Supply duct leakage goes into the zone (``duct_location``: "conditioned", default)
    or is lost ("unconditioned").
  - Duct heat losses (12)-(15), (18)-(19) with ``duct_supply_h_unconditioned_w_k``,
    ``duct_supply_h_zone_w_k``, ``duct_extract_h_unconditioned_w_k`` and
    ``duct_surrounding_temperature_c`` (default: outdoor air). Losses (21) and the
    recoverable part (22) are reported.
- Recirculation (6.4.3.2.5): ``recirculation_control`` "fix" (``outdoor_air_fraction_min``)
  or "variable" (component ``outdoor_air_flow_m3_h``). With the modulating control the
  outdoor air share rises up to f_ODA;hr (38b) before the heat recovery is bypassed.
  Mixing (36a), (37) and energy (79).
- Frost protection (6.4.3.2.7): ``frost_control`` "direct" (46, with the corrected ratios
  ``frost_direct_ratio_minus15`` / ``frost_direct_ratio_minus7``), "indirect_bypass"
  (48c), "indirect_preheat" (47b, preheat energy 90a) and "indirect_recirc" (exhaust mixed
  into the outdoor air). ``building_use`` sets the Table B.14 exhaust limit.
- Adiabatic cooling on the extract side (6.4.3.2.8): ``adiabatic_cooling``,
  ``adiabatic_cooling_efficiency`` (default 0.9). It runs in cooling, with recirculation
  off, formulae (50)-(52) and a psychrometric wet-bulb temperature. Water use is reported.
- Central multizone AHU with VAV (``building_parameters.central_ahus``):
  - One AHU step serves several zones. Each zone has a constant volume or VAV terminal.
  - The VAV flow is sized from the zone load of the previous hour.
  - Air flow control NO_CTRL, ON_OFF, MULTI_STAGE or VARIABLE (23a-d).
  - The AHU flow is shared among the zones in proportion to their requirements (6),
    without exceeding any terminal maximum. Flow above the sum of the maxima (NO_CTRL,
    a stage or the minimum flow ratio) is not delivered.
  - Extract air is the flow-weighted mix of the zones. The AHU moisture is coupled to
    the zone moisture balance.
  - With ``recirculation_control: "variable"`` the outdoor air share follows (35a): the
    largest ratio of required outdoor air to zone supply among the zones. Every zone
    therefore gets its ``outdoor_air_flow_m3_h``, not only the total.
  - When the flow limits make a zone requirement impossible (a fixed share, f_ODA;hr < 1,
    frost recirculation, or a zone supply below its requirement), a warning is issued.
    The hours are counted in ``CentralAHU.oda_deficit_hours``, and
    ``q_oda_m3h_<ahu>_<zone>`` reports the outdoor air each zone receives.
  - A served zone outside the AHU moisture path (``latent_ahu_moisture: false`` or a
    declared ``latent_system_supply_moisture_kg_kg``) still has its moisture content
    carried. That state only feeds the AHU extract mix, so the AHU still treats the
    humidity for the other zones.
  - New columns ``<diag>_<ahu>`` and ``q_vav_m3h_<ahu>_<zone>``.
- New AHU diagnostic columns: ``Q_ahu_duct_loss``, ``Q_ahu_duct_loss_rbl``, ``Q_ahu_rca``,
  ``Q_ahu_frost_preh``, ``f_ahu_oda``, ``q_ahu_makeup_m3h``, ``m_ahu_ac_water_kg_h``.
- An AHU with no supply and a non-zero extract (extract only) still uses extract fan
  power (72), and the zone takes the extracted air from outdoors.
- Full recirculation (``recirculation_control: "variable"`` with no outdoor air required)
  is calculated: no outdoor air crosses the heat recovery. With the modulating control,
  the outdoor air share can still open up to f_ODA;hr (38b). Before, this case raised
  ZeroDivisionError.
- With the default configuration the AHU step gives the same results as before.
- Where the printed formula breaks the energy balance, the implementation keeps the
  balance:
  - (18): the extract duct loss lowers the temperature;
  - (20): the leakage air is mixed by flow;
  - (47b)/(48c): the supply/exhaust flow ratio is inverted;
  - (50): with adiabatic cooling, the supply after recovery is the recovered temperature.

EN 16798-7 (module M5-5, air flow rates), new module ``ventilation_16798_7``:

- The existing ventilation types (``temp_wind``, ``occupancy``, ``sherman_grimsrud_like`` and
  others) are empirical models, not this standard. The new module implements method 1
  (clause 6) as a separate calculation of one ventilation zone.
- Required flows: outdoor air requirement (9) with ``f_ctrl`` / ``f_sys`` (Tables B.4, B.5),
  flows at the air terminal devices by VENT_SYS_OP (10)-(17), ODA or LOAD flow control
  (18)-(22), design flow (23), supply outdoor air (24), and the LOAD_COMP supply
  temperature with emission losses (7)-(8).
- ``ZoneAirflow.solve`` combines in the mass balance on the internal reference pressure (67)
  all the air flow paths:
  - envelope leakage with the Table B.12 distribution (B.4)-(B.6), (62)-(64);
  - vents (58)-(61), (59), Table B.11;
  - airing through windows: simplified (42)-(43), wind and temperature, single-sided and
    cross ventilation (44)-(52), or the window pressure method (53)-(57);
  - passive or hybrid duct with air terminal device, duct and cowl (25)-(34);
  - combustion air (35)-(39);
  - mechanical supply and extract flows.
  It gives the totals ``q_V;tot;in`` / ``q_V;tot;out`` (68)-(69) and the mass balance residual.
- Wind (2)-(3) and pressure (4)-(6) with the Table B.7/B.8 pressure coefficients, Table B.13
  terrain classes and the Annex B defaults (Tables B.2-B.14).
- ``ahu_inputs`` maps the mechanical flows to the input of the EN 16798-5-1 AHU model.
- Method 2 (clause 7) is a national statistical approach with no default data: only the
  totals are exchanged (``method2_totals``).
- Coupling with the hourly zone balance, new ventilation component
  ``ventilation_type: "en16798_7"``:
  - ``zone`` holds the :class:`ZoneAirflow` data (windows and ``passive_duct`` as dicts),
    plus ``outdoor_air_req_m3_h`` (simplified airing) and ``combustion_flow_m3_h``;
  - each hour it solves the mass balance (67) with the outdoor temperature, the zone air
    temperature, the wind speed and the actual flows of the AHU components of the same
    hour, and gives the natural inflow (leaks, vents, windows, passive duct) at the
    outdoor temperature;
  - an extract-only or unbalanced AHU then draws its make-up air through the envelope.
    The "_makeup" stream of the AHU is not added when an ``en16798_7`` component is
    present, so the air is not counted twice;
  - the component schedule multiplier is the window opening fraction (0 closes the windows).
- Inconsistent configurations are errors: a mass balance with no solution (for example
  an extract flow with no leak, vent or window to bring the air) raises ``ValueError``
  instead of returning an absurd pressure, and a zone accepts only one ``en16798_7``
  component, since all its air flow paths are solved together.
- ``required_supply_temperature`` (7)-(8) no longer assumes a 1 K difference when the
  corrected outdoor temperature is missing, nor zero loss when the denominator is null: it
  raises ``ValueError`` naming ``theta_e_comb_c``.
- Non-physical inputs of the EN 16798-7 API (negative design or required flows, negative
  or null airtightness/exponents, invalid ``PassiveDuct``, ``f_oda`` outside [0, 1],
  negative wind speed or combustion data) are rejected with a message naming the parameter.
  NaN/inf values are rejected too (also in ``required_supply_temperature``, whose air density
  must be > 0), and ``f_op_v`` (an operation signal) must be in [0, 1].
- EN 16798-5-2: the reheating energy (40) uses the whole stage time, not the recovery time:
  with the recovery fully bypassed the coil now supplies the missing heat (it was 0 kWh).
- EN 16798-5-2, heat pump (46): the printed formula both removes the domestic hot water time
  from ``t_on;H;max;k`` and subtracts ``Q_W;V;gen;hp;out``, so the DHW share is counted twice.
  The printed text is kept by default; ``HeatPumpData(deduct_dhw_energy_46=False)`` subtracts
  only the time.
- EN 16798-5-2: ``eta_hr`` must be finite and in [0, 1] and temperatures/preheating/precooling
  must be finite, as must ``defrost_hours``/``defrost_dt_k`` and the numeric unit
  configuration; NaN or a recovery above 100 % is an error instead of a NaN or non-physical result.
- A ``mechanical_supply`` component can take its flows from the standard: with
  ``outdoor_air_design_flow_m3_h`` (and optional ``f_ctrl``, ``f_sys``,
  ``ventilation_effectiveness``, ``vent_sys_op``) the supply and extract flows follow
  (9)-(17).
- Limits: the natural flows are not reported as output columns; the flow control
  SUP_AIR_FLW_CTRL = LOAD (20) is available as a function but not yet applied hourly to the
  AHU; single ventilation zone without elementary spaces or flows between zones.
- Table 7 gives the airtightness index in m3/h; with the reference area ``A_lea`` of
  (A.1)/(B.4) it is used per m2 of ``A_lea``. In (47)/(48) the stack term uses the exponent
  0.5 as in (45)/(46), since the printed formula is cut at the edge in the scan.

EN 16798-5-2 (Method 2, monthly / yearly / bin), new module ``ventilation_16798_5_2``:

- For small packaged ventilation units, typically residential.
- ``calculate_interval`` covers, for one month, year or bin:
  - ventilation stages, leakage of ducts and unit (4)-(10), Tables B.2-B.5;
  - supply air temperature with preheating / precooling, heat recovery with the leakage
    and insulation factors (Table B.6), recirculation and reheating (11)-(15);
  - supply and extract duct losses (16)-(21) and the outdoor air fraction (22)-(23);
  - duct and casing heat losses (25)-(30);
  - fan energy with SPI per stage or a single SPI with flow control factor and motor
    exponent (31a)/(31b), Tables B.12-B.15;
  - auxiliary energy of rotor, electric defrost preheating and controls (32)-(35),
    Tables B.7-B.9;
  - preheating, precooling, heat recovery, recirculation and reheating energies
    (36)-(40), Table B.17.
- Extract air heat pump for domestic hot water, space heating and space cooling (41)-(58),
  with the B.3.1 defaults and the Table B.16 corrections.
- ``combine_by_area`` applies (24) to several systems in one zone.
- ``monthly_from_hourly`` builds monthly intervals from an hourly EN ISO 52016-1 result.
  The annual defrost hours of Table B.7 are shared by the hours below -4 degC.
- (20) is applied with the extract duct loss lowering the temperature, as for
  EN 16798-5-1 (18). The air density uses 288 in the altitude correction (Table 9 of
  EN 16798-5-2; EN 16798-5-1 uses 293).
- Input checks: ``total_hours`` > 0, stage times within ``total_hours``, defrost hours within
  the interval, no negative flows or demands. Heating and cooling heat pump demands in the
  same interval are rejected (the method takes one of them, 6.4.3.10.1.3 NOTE 1).
- A demand for a heat pump service that is not configured (for example hot water with a
  heating-only heat pump) is reported as residual (``Q_*_residual``) with a warning.
- ``monthly_from_hourly`` makes one interval per (year, month) and returns
  ``((year, month), result)``. The annual defrost hours are scaled to the data length of the
  year and capped at the operating hours of the interval.
- ``extract_duct_sign_as_printed=True`` applies (20) with the printed ``+`` sign. The default
  keeps the physical direction (the loss lowers the extract temperature). No official erratum
  was found; the EPB Center spreadsheet for EN 16798-5-1 keeps the printed sign in (18).
- The module is independent of the hourly AHU model. Its results are not yet fed back
  into the EN ISO 52016-1 zone balance.

Fan location and load compensated supply temperature in zones:

- ``supply_fan_location`` / ``extract_fan_location`` (UP_HR, DOWN_HR) now also have their
  thermal effect, not only the Table B.5 leakage:
  - supply fan UP_HR: its heat warms the outdoor air before the heat recovery (32a). Most
    of it then leaves with the exhaust. It is counted once: the printed (70) would add
    it again after the coils, and so does the reference spreadsheet;
  - extract fan DOWN_HR: its heat leaves with the exhaust air (31b).
  - Defaults (supply DOWN_HR, extract UP_HR) give the same results as before.
- ``supply_temperature_mode: "load_compensated"`` (29c) now works for zone-level AHU
  components in the standard, ``ahu_causal`` and multizone engines.
  - The required supply temperature covers the zone sensible load of the previous step:
    the ideal zone system power still delivered plus the heat the AHU supplied, with the
    current flow, within ``supply_temperature_minimum_c`` / ``maximum_c``.

Annex D amended and Italian national data:

- Annex D (rotary heat exchangers, moisture recovery) now follows the amendment, as in the
  EPB Center spreadsheet. This changes results for rotary heat recovery with moisture:
  - (D.8) without the "+1": the flow-ratio factor is 1 for balanced flow (before 0.9);
  - Table D.1: C7 ROT_HYG 0.476, C10 ROT_SORP +0.098, C11 per type (1.7012 ROT_NH,
    1.5333 ROT_HYG, 1.3444 ROT_SORP);
  - (D.5b) for ROT_HYG with the nominal condensation potential 0.005; (D.5c) removed.
  - The printed 2017 version stays available with ``annex_d_version: "2017"``.
- National default data, ``national_data``: "IT" (default, UNI EN 16798-5-1:2018/NA:2025)
  or "EN" (Annex B):
  - ``building_status`` ("new" / "existing") sets the duct and AHU tightness classes
    (Tables NA.2 / NA.3);
  - heat recovery design pressure drop by type (Table NA.6), used in (89);
  - ``fan_motor_location`` (IN_AIR / OUTS_AIR) sets f_fan;rd (Table NA.12);
  - ``system_type`` ("ventilation" / "air_conditioning") sets f_dp;ctrl (Table NA.13);
  - humidifier pump energy by type and control (Table NA.10), with the humidifier types
    CONTACT, ROT_SPRAY, HI_PRES, HYBRID, STEAM, OTHER;
  - IT only: control power 5 % of the fan power when no control power is given
    (Table NA.9). This adds auxiliary energy only.
  - The annex gives the pump energies (Tables NA.6, NA.10) in kWh/m3. They are used as
    Wh/m3, since the EPB Center spreadsheet notes the printed values are 1000 times too
    large.
  - Tightness classes, fan motor location and system type are applied only when
    declared. Without them, the thermal results do not change.

EN 16798-5-1 remaining parts, checked against the EPB Center spreadsheet
Demo_EN_16798-5-1 (2026-05-12):

- Supply temperature control (Table 19): ``supply_temperature_mode``
  - "none" (NO_CTRL, (53a)/(67a));
  - "oda_compensated" ((29b), with ``supply_temperature_slope`` and
    ``supply_temperature_offset_c``);
  - "load_compensated" ((29c), with the required supply temperature from
    ``AHUStepInputs.required_supply_temperature_c``, or from the zone loads for a central
    AHU with constant volume terminals).
- Fans (72)-(75): ``fan_control`` "direct" (default, as before), "no_ctrl" (manufacturer
  curves ``*_fan_pressure_curve``), "const_pres" and "min_pres" (with f_V;max from (11a)).
  Fan efficiency curve (73): ``*_fan_efficiency_curve``.
- The central AHU fans now follow the flow fraction of the nominal flow, so VAV operation
  reduces fan power by the part-load law. Before, power was proportional to flow.
  - The AHU flow is limited to the nominal flow (23c)/(23d).
- Actual coil states (62)-(64), (69):
  - from the coil medium temperature and the delivered energy given by the generation
    modules (``cooling_coil_medium_temperature_c``, ``cooling_coil_output_w``,
    ``heating_coil_input_w``);
  - required medium temperatures (61), (68) are reported for the generation modules;
  - the dehumidification requirement (81) is reported.
- AHU casing heat losses (85) and the recoverable part (86): ``ahu_location``,
  ``ahu_supply_casing_au_w_k``, ``ahu_extract_casing_au_w_k``.
- Auxiliary energy (87)-(93):
  - heat recovery rotor (88a, with the rotor speed from D.2-D.4) or pump (88b);
  - fan energy share of the heat recovery (89);
  - controls (91);
  - humidifier pump (92)-(93).
- Ground heat exchanger, Annex C (C.1)-(C.10): ``ground_exchanger`` (soil from Table C.2,
  coldest month from Table C.1, NO_CTRL or BYPASS control (32)), with the moisture change
  (33) and the energy (76).
  - Outdoor air duct (32): ``duct_outdoor_h_w_k``.
  - The engines pass the hour of the year to the AHU.
- Air density at the site altitude (Table 27): ``altitude_m``.
- Supply duct sections are calculated in sequence (15): the duct heat loss to the zone
  uses the temperature after the loss to unconditioned space, as in the reference
  spreadsheet.
- New AHU diagnostic columns: ``Q_ahu_gnd``, ``Q_ahu_casing_loss``,
  ``Q_ahu_casing_loss_rbl``, ``P_ahu_aux``, ``P_ahu_hum_aux``, ``Q_ahu_dhu_req``.
- Table B.5: the printed f_ODA;hr = 1.02 for PLATE (an outdoor air share above 1) is set to
  1.0, as in the reference spreadsheet. Before, a PLATE recovery with declared fan
  locations was rejected.
- Checked against the reference spreadsheet (summer hour; see
  ``tests/test_ahu_annex_16798_5_1.py``):
  - flows, outdoor air share, heat recovery ratio (40)-(43), D.5a/D.6, the
    dehumidification requirement (56)-(60) and the actual coil states (62)-(64) agree;
  - (62) and (69) agree within 0.02 K, from the air properties.
- The reference spreadsheet differs from this implementation in these places:
  - it keeps the printed (18), (20), (47b)/(48c) and (50);
  - with NO_CTRL it gives a supply after recovery below the outdoor temperature, and a
    dehumidification energy (80) without its sensible part.

Limits:

- Central AHU:
  - multizone only;
  - load compensated supply temperature only with constant volume terminals;
  - one supply temperature for all zones, with no per-zone reheat terminal;
  - VAV sizing uses the previous hour, not an iteration within the hour.

2.1.7 (2026-11-23)
------------------

First PyPI release containing the 2.1.6 changes (2.1.6 was tagged but not published).

- Multizone INTERNAL partitions (result-changing): the face that looks into the adjacent
  simulated zone is an internal surface (EN ISO 52016-1 Table 25), in the main and hybrid
  multizone methods.
  - Before, that face used the surface's ``*_external`` coefficients for convection and
    long-wave radiation, so a partition declared with wall-row films (e.g. ASHRAE 140
    external 11.9/4.14) got different films on its two faces, and results depended on
    which zone declared it.
  - Vertical partitions get the declared internal coefficients on both faces.
  - Floors and ceilings get the reversed heat-flow direction on the second face
    (upwards 5.0 / downwards 0.7 W/(m2K)). Before, both faces had the same value.
  - With the TARP model the second face is also updated every hour; before, it stayed
    at the table value.
  - Optional keys ``convective_heat_transfer_coefficient_internal_b`` /
    ``radiative_heat_transfer_coefficient_internal_b``.
  - ``Conductance_node_of_element`` subtracts the second-face film, so the declared U
    stays air-to-air.
  - ``UNCONDITIONED`` boundaries are unchanged. In ``build_case_960`` the override of
    the external fields is now redundant.
  - The surface aggregation keeps the ``*_internal_b`` keys.
  - Effect on the two-floor example (Athens), cooling: Z_ground 17 437 -> 18 752 kWh,
    Z_first 20 265 -> 18 975 kWh. Heating changes by about 2 % and the building totals
    are almost unchanged: less exchange through the slab, so the load moves between the
    floors.

- ``Conductance_node_of_element`` now raises ``ValueError`` (surface index, U, core
  resistance, surface resistances and maximum U) when 1/U - R_si - R_se <= 0. Before:
  - a window with U above about 5.8 W/(m2K) and the default vertical films got a
    negative core conductance (e.g. -334 W/(m2K) at U = 5.9), an unphysical network;
  - a core resistance of exactly 0 left the element without conduction.

  A non-positive or non-finite ``u_value`` is also rejected.
- ``sanitize_and_validate_BUI`` reports the same incompatibility as an ERROR before the
  simulation.
- ``examples/building_archetypes.json``: 20 windows with U = 5.8-6.1 W/(m2K) set to 5.7
  W/(m2K), the single-glazing value compatible with the standard surface resistances.

2.1.6 (2026-11-20)
------------------

Window aggregation and EN 15316-2 emitter capacity (result-changing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Aggregated windows: the equivalent ``g_value`` now keeps the sum of
  g·A·(1−F_fr) of the windows (Formula 69). Before, the product of the area-weighted g
  and F_fr differed when the windows had different g and frame fractions.
- Per-window shading weights: when all weights g·A·(1−F_fr) are zero (base g = 0, e.g.
  an adaptive window whose later state has g > 0), the factor falls back to
  A·(1−F_fr) weights instead of returning 1.0.
- ``EmissionSystemCalculator``: the deficit is only the part of the need at the modified
  set point that exceeds the capacity P_n·t. A recalculated need lower than the base
  need is no longer reported as unmet energy.
- ``nominal_power_kW: 0`` now means no emitter for that service: the whole need is
  backup or unmet energy, with a warning. A negative value raises ``ValueError``.
  Leave the key out for unlimited power.
- ``simulate_multizone_emission`` with ``hydraulic_systems``: for a zone with an emitter,
  the circuit runs in EN 15316-2 mode with the same emitter configuration. Its emission
  input therefore equals the emission result, which a test checks. A zone without an
  emitter gets a warning.
- ``simulate_multizone_emission(couple_emitter_capacity=True)`` (default): the emitter
  ``nominal_power_kW`` limits the zone ``heating_capacity`` / ``cooling_capacity`` in both
  thermal runs. A power shortage now changes the zone temperature and the adaptive-facade
  control that follows. ``False`` restores the post-processing check (unmet energy, zone
  temperature unchanged). The limits used are in ``result.applied_capacities``.
- ``simulate_multizone_emission(unlisted_zones="free_floating")`` (default): a zone missing
  from ``emission_by_zone`` runs without heating or cooling (``free_floating: true``),
  instead of taking an ideal load that no emitter delivers. A warning names these zones,
  which are also listed in ``result.free_floating_zones``. ``"ideal"`` restores the
  ideal system, and an explicit ``free_floating`` key on the zone is respected.
- New ``compare_adaptive_with_reference_multizone`` (EN ISO 52016-3, 6.10 and clause 8)
  compares the adaptive building with its reference on the multizone engine. It gives the
  Table 6 indicators per zone, the state statistics and the clause 8 report. The comfort
  limits default to the zone set-points.
- ``adaptive_envelope_kpis``: ``dT_low_pct`` and ``dT_high_pct`` are NaN when the reference
  is below 1 K·h. Before, numerical noise around the set-point gave percentages such as
  -78 %.
- Multizone engine: thermally unconditioned adjacent zones (EN ISO 52016-1 6.4.5, 6.5.9,
  E.3) through ``boundary: "UNCONDITIONED"`` and ``name_adj_zone``, with the
  ``adjacent_zones`` list as in the single-zone engines.
  - θ_ztu of hour t uses the operative temperature of hour t-1 of the conditioned zones
    next to it, weighted by U·A.
  - The internal type uses the corrected resistance of Formula 54, and its gains are
    attributed to the zones with Formulae 65 and 68.
  - The output includes ``theta_ztu_<name>``. The hybrid method rejects these
    boundaries.
  - The surface aggregation keeps surfaces facing different unconditioned zones apart.
- Multizone output: ``T_set_H_<zone>``, ``T_set_C_<zone>`` (set-points active in the
  hour), ``Q_HVAC_req_<zone>`` (ideal request) and ``cap_limited_<zone>``.
- New ``comfort_16798_1`` module with EN 16798-1 Annex B comfort categories, adaptive
  (B.2.2) or static (Table B.1). For each zone it gives the hours per category and the
  degree-hours outside category II. Capacity-shortage indicators give the hours with the
  capacity limit, the degree-hours off the set-point and the energy not delivered.
  ``MultizoneEmissionResult.comfort()`` returns them per zone.
- ``examples/multizone_free_floating_example.py``: the partition ``ZoneA_Wall_N_Int``
  declared itself as ``adjacent_zone`` and is now adjacent to ``Z2``. The new input
  validation had rejected it.
- New ``multizone_system.simulate_multizone_system`` for system topologies of a
  multizone building:
  - ``central``: one generator and one EN 15316-3 distribution feed one EN 15316-2
    emitter per zone. The zone emitter inputs are summed on the circuit.
  - ``per_zone``: each zone has its own distribution and generator.
  - ``combined``: one system serves space heating, space cooling and DHW, with one
    emission type for all zones and a DHW storage tank.
  - Generator kinds: ``heat_pump``, ``boiler_15316_4_1``, ``district``, or ``ideal``
    (constant efficiencies).
- ``dispatch_combined`` shares the generator hour by hour between space heating,
  cooling and tank charging.
  - The tank is charged from the spare capacity. When it falls below the minimum
    temperature, the charge has priority and the space energy it displaces is reported.
  - In cooling, ``time_share`` stops cooling to charge the tank, while ``heat_recovery``
    uses condenser heat recovery.
  - ``MultizoneSystemResult.dhw_dispatch_summary()`` reports charging hours during
    heating and cooling, displaced energy, tank losses and unmet energy.
- Adaptive windows (EN ISO 52016-3): visual comfort and artificial lighting with the
  ``lighting`` option of ``adaptive_envelope``: ``target_lx``, ``power_W_m2``,
  ``control`` (``dimming``/``on_off``), and ``daylight_coefficient`` or
  ``control.daylight`` (Formula 4).
  - Hourly columns per window: ``E_v_ext_lx``, ``E_day_lx`` (daylight on the working
    plane with the τ_V of the state applied), ``E_art_lx``, ``E_tot_lx``,
    ``P_light_W_m2`` and ``visual_assessed``.
  - Daylight comes first: artificial light only makes up what daylight lacks to reach
    the target, in occupied and awake hours.
  - The DAYL-MODE threshold becomes the target, so the control picks the state that
    lets in more light when daylight is short, within the Annex C scenario.
  - ``adaptive_lighting_by_state`` summarises daylight, artificial light, the share of
    hours on daylight only and lighting energy per window and control state. Lighting
    is also reported in the clause 8 report.
- ``multizone_system`` fixes:
  - Heating and cooling in the same hour (different zones) on one generator are served by
    time share, in the order ``space_priority`` (``heating``/``cooling``/``larger``).
    Before, the cooling demand disappeared from the balance. Each service now satisfies
    served + unmet = demand. The dispatch reports ``first_service``, ``simultaneous`` and
    the energy displaced by the other service.
  - ``couple_generator_capacity=True``: the declared generator power limits the zones
    in the thermal balance.
    - With a shared generator (``capacity_allocation="dynamic"``, the default), each hour
      the solver shares the power among the zones in proportion to their request of that
      hour. This uses the new building key ``generator_groups``.
    - Heating and cooling requested in the same hour share the generator time in the
      order ``space_priority``.
    - ``"static"`` keeps the fixed split by emitter power or floor area.
  - ``combined``: the tank dispatch is recalculated on the actual generator production,
    hourly where the generator model gives hourly results. Terminals per zone are allowed
    through ``emission_by_zone``.
  - ``per_zone``: when DHW is given, the ``dhw_share`` values must be >= 0 and sum to 1.
  - The annual table has ``W_em_aux_kWh`` and ``electricity_total_kWh``
    (generator + distribution + emitter auxiliaries).
  - The ``ideal`` generator accepts ``max_heating_kW``, ``max_dhw_kW`` and
    ``max_cooling_kW``.
- Known issues with the EnergyPlus reference data in ``examples`` (not fixed). These
  files cannot serve as a multizone validation:
  - ``two_zone_ideal_24_2_fixedout_Milan_2020.csv`` is identical to the Athens file.
  - ``2020_Stockholm.epw`` is a copy of the Stockholm EnergyPlus output CSV, not a
    weather file.
  - ``calibrate_multizone_against_energyplus_climate.py`` imports a function that no
    longer exists.
  - With ``multizone_free_floating_example.py`` and Athens, annual heating is -52 % and
    cooling is about 15 times EnergyPlus, with an operative temperature bias of +2.8 K.
    The input is therefore not equivalent to the EnergyPlus model (IDF not in the
    repository).
- ``examples/multizone_declaration_example.py`` and ``docs/multizone_declaration.md``:
  annotated input and declaration of scope for the multizone, adaptive and emission
  calculation.

Thermally unconditioned adjacent zones and multizone (EN ISO 52016-1 6.4.2, 6.4.5-6.4.7, Annex D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Result-changing fixes (single-zone engines with ``adj_zones_present: true``)

* Fix: the internal gains of the conditioned zone were inflated when adjacent zones were present
  (2.2 times with one zone and b = 0.8, 3 times with two zones): Formula 65 added the conditioned
  zone's own gains again.  Formula 65 (and 68 for solar gains) now applies only to *internal*-type
  zones, with the gains of the unconditioned zone itself.
* Fix: with several adjacent zones, Formulae 58-59 used the ``b_ztu`` of the last zone of the loop
  for every zone, and the gains of all zones used class and area of the first one.  Each zone now
  has its own ``b``, ``H_ztu;tot`` and gains.
* Fix: ``H_ztu;e`` follows Formula 6, ``(1 + c_ztu;ve) H_tr;ue`` with ``c_ztu;ve = 0.5`` (Table B.5),
  instead of the ISO 13789 ventilation term (still available with ``n_ue``).
* Fix: ``theta_ztu`` of the first hour is computed from Formula 58 (initial operative temperature
  20 C) instead of a fixed 15 C.

New (module ``unconditioned_zone_52016_1``)

* ``type: "internal"`` for adjacent zones: ``theta_ztu`` equals the operative temperature of the
  conditioned zone (6.4.5.3), corrected resistance of Formula 54 (6.5.8.4), ``H_ve = b rho c q``
  (Formula 62), gains attributed with (1 - b) F (Formulae 65, 68).
* ``b_ztu`` default values (Table A.4; gains then set to zero, E.3.1), ``c_ztu_h_max`` (Table B.16).
* E.3 solar gains of the unconditioned zone (Formulae E.5-E.6): ``g_value``/``g_value_elements``,
  ``absorbing_surfaces``, ``obstacle_shading_factor``.
* Ventilation components with ``from_adjacent_zone``: supply temperature ``theta_ztu`` (6.5.10.2).
* ``check_input`` validates the new zone options.

Multizone engine (``module multizone_52016_1``)

* Air flow between zones (Annex D.2 1, D.4): ``building_object["interzone_airflows"]``; a flow in
  both directions is rejected (D.2: the zones are one zone).  Output ``Q_ve_interzone_<zone>``.
* The HVAC modes are re-checked after coupling: a zone left idle that leaves its set-point range
  because a neighbour is conditioned is switched to heating/cooling, and a conditioned zone that
  needs no energy is released (previously its temperature was held while the energy was clamped to 0).
* ``free_floating: true`` zones (thermally unconditioned space as a zone, 6.4.2.3).
* Input validation (unknown zones, duplicated partitions, area sums) and the zoning criteria of
  6.4.2 (steps 2, 3, 5, 9, 10) as warnings; ``validate=False`` disables them.
* ``annual`` results have a ``zone == "TOTAL"`` row (``include_total=False`` to omit it).  Code that
  sums the column must exclude it.
* ``multizone_report`` / ``format_multizone_report`` / ``MultizoneResult.coupling_report()``:
  partitioning, H_tr,zy, air flows and the declaration of the coupling scheme (all zones solved
  together in each hour, a deliberate deviation from the previous-hour boundary of D.2).
* The hybrid iterative method refuses ``interzone_airflows`` and ``free_floating`` zones.
* Fix (multizone): the external shading factor of a window element was the area-weighted average
  of all the windows of the same zone and orientation, applied to every element of the group, also
  to a movable/adaptive window kept as its own element (two windows with factors 0 and 1 both got
  0.5).  Each element now uses the factors of its own windows only (weighted by g A (1 - F_fr) for
  an aggregated element), as in the single-zone engines (Formula 69).  The hourly output has
  ``F_sh_<window element>`` for every shaded element.
* EN 15316-2 emission per zone in the multizone path: ``simulate_multizone_emission(config,
  emission_by_zone, emission_inc="recalculate", hydraulic_systems=None)``.  Each zone has its own
  equivalent temperature increase; the need at the equivalent set-points is recalculated with one
  multizone run where all the zones are shifted together, so the effect of the partitions is kept
  (raising the heating set-point of one zone lowers the heating need of its neighbour).
  ``"approximate"`` uses the temperature-ratio approximation zone by zone (coupling ignored, with a
  warning).  Optional ``hydraulic_systems``: ``MultiZoneHeatingSystemCalculator`` (EN 15316-1 Annex C,
  common node at the highest required flow temperature).  Result: ``MultizoneEmissionResult`` with
  the annual table per zone (+ TOTAL) and ``report()``.  Tests: ``tests/test_multizone_emission.py``.
* EN ISO 52016-3 adaptive windows in the multizone engine (they raised an error before).  A control
  context is created only for the zones that have at least one adaptive window; each context uses
  the operative temperature, HVAC load and occupancy of its own zone (previous time steps, 6.8.4).
  The zones interact only physically (partition, air flows), never through the controls.  State
  logs ``adaptive_<window>_*`` as in the single-zone engine; an adaptive window name used in two
  zones is rejected.  Tests: ``tests/test_adaptive_envelope_52016_3_multizone.py``.
* Latent balance per zone (6.5.14, ideal humidity control): the multizone engine had no moisture
  calculation.  ``Temperature_and_Energy_needs_calculation_multizone`` now applies the single-zone
  balance to every zone (zone volume, outdoor air flow ``H_ve``, zone air temperature, outdoor
  temperature and relative humidity): ``X_int_kg_kg_da_<zone>``, ``G_HU_kg_s_<zone>``,
  ``G_DHU_kg_s_<zone>``, ``Phi_HU_W_<zone>``, ``Phi_DHU_W_<zone>``, ``Q_latent_W_<zone>``,
  ``Q_H_latent_W_<zone>`` and the annual ``Q_latent_annual_kWh`` / ``Q_H_latent_annual_kWh`` (with the
  ``TOTAL`` row).  Options per zone or as keyword arguments, as in the single-zone engine
  (``latent_rh_humidification_pct``, ``latent_rh_dehumidification_pct``, ``moisture_production_kg_s``,
  ``latent_internal_fraction``, ...); ``include_latent=False`` disables it.  Moisture carried by the
  air flows between zones is not modelled.  The hourly frame also has ``T_ext`` and ``RH_ext``.

Single zone

* 6.4.6 Option B (Formulae 7-8): ``building_parameters.spatial_setpoint_adjustment =
  {"enabled": true, "f_mod_t": 0.8, "f_mod_sp": 0.5, "H_int_spec": 2.0}`` lowers the heating
  set-point month by month; ``spatial_setpoint_reduction`` is also available as a function.

ISO/TR 52016-2 (checked against the implementation)

* 6.4.6.4, Table 2: the Option B formula reproduces the three worked values (18,0 / 19,0 / 19,5 C);
  this also confirms the reading of Formula 7, whose operator is missing in the extracted text.
* 6.4.5, Table 1: the transmission, ventilation and gains rules of the two zone types agree with the
  implementation.  The internal type now warns when the zone has glazing (not suitable for
  sunspaces and atria).
* 6.5.9: the TR text on ``c_ztu,h;max = 0`` contradicts Formula 59 (``docs/standards_errata.md``);
  ``c_ztu_h_max: null`` disables the limit.
* 6.3.6: repeating the calculation within the interval is an accepted alternative to using the
  previous interval, which justifies the simultaneous solution of the coupled zones (declared in
  the report).  7.2.8: thermal coupling and unconditioned spaces are not in the BESTEST cases and
  "can be tested analytically", as done in the new tests.

Tests: ``tests/test_unconditioned_zone_52016_1.py``, ``tests/test_multizone_52016_1.py`` (partition
heat conservation, H_tr,zy = U A on the operative temperature within 2 %, mode re-check, air flows,
validation, report, Option B).

2.1.5 (2026-11-17)
------------------

Solar shading (EN ISO 52016-1 Annex F)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Result-changing fixes found by checking the shading of windows against Annex F.

* Fix: the single-zone engines lost the overhang, side fins and obstacles of windows aggregated by
  orientation.  Two or more windows of the same orientation were merged into one element named
  ``"A + B"`` whose shading column ``W_A + B`` does not exist, so the factor stayed 1.0 (Case 600,
  south window split in two with a 2 m overhang: heating/cooling and solar gains equal to the case
  without overhang).  The factor of the element is now the average of the factors of its windows
  weighted by ``g * A * (1 - F_fr)`` (Formula 69 is linear in the factor, so the gain is exact).
* Fix: the shadow of the overhangs and the shadow of the obstacles are added (F.16-F.18); the maximum
  of the two was used.  Each family is capped by the window height.
* Fix: ``annex_f_shading_objects`` with ``kind: "overhang"`` use F.11 (``lowest_height_m``, sill
  ``window_sill_height_m``) instead of the obstacle formula F.10.
* Fix: a window without shading elements is never blocked by a missing height or width (the direct
  sun was set to zero silently).  A window with an overhang or side fins needs a positive height and
  width (F.3.3.1), an obstacle-only window the height only; ``ValueError`` otherwise.  Overhang depth
  ``D``, distance ``L`` and side fin depth/distance must be >= 0 (Table 15).
* Fix: the key ``overhang_proprieties`` (spelled so in ``examples/building_archetypes.json``,
  ``building_examples.py`` and other examples) was ignored and switched the overhang off.  The
  examples are corrected; the misspelled key is accepted with a warning by the engine and renamed by
  ``sanitize_and_validate_BUI``, which also reports an overhang without depth and rejects missing
  window geometry for windows with overhang or fins.  Results of the corrected examples change
  because their overhangs are now active.
* Documentation of the input fields in ``examples/new_building_input_en.md``: ``width_or_distance_of_shading_elements``
  is the vertical distance L, ``width_of_horizontal_overhangs`` the depth D.
* Not changed: the orientation of the window is still rounded to 45 degrees for the hourly shading
  (Annex F rounds only in the monthly method); several overhangs or fins per window are not supported.
* New tests: ``tests/test_iso52016_shading_annex_f.py``.

2.1.4 (2026-11-13)
------------------

EN ISO 52016-3:2023 + A1:2025 - adaptive building envelope elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transparent elements whose U-value, g-value and daylight transmittance change during the
year (dynamic solar shading and chromogenic glazing) are now part of the hourly
EN ISO 52016-1 balance of the single-zone engines.

* New ``adaptive_envelope`` input on a transparent surface (module
  ``source/adaptive_envelope_52016_3.py``): ``type`` (roller blind, venetian blind, shutter,
  chromogenic), ``position`` (external, internal, integrated), ``control_type`` (manual,
  motorized, automated, environmental), ``space_category`` (residential, office), ``system``
  (heating, heating and cooling) and the properties of the retracted and extended state
  (venetian blind: one set per slat position).  It is a simplified element in the sense of
  6.4.3: the properties of each state are inputs.
* Actively controlled elements follow the reference control scenarios of 6.8 and Annex C:
  the modes DAY, HC, TINT, RAD, OCC, GLARE and DAYL (Table 22) are identified with the default
  methods of Tables 24-31, the radiation limits of Table C.1 and the parameters of Table 33,
  the state (1 to 23) is read from Tables C.2-C.8, and the occupant correction of 6.8.6
  (approach 1) is applied.  The U-value and g-value of the window are the area-weighted mix of
  the retracted state and of the extended state of the scenario.
* Environmentally activated chromogenic glazing: continuous linear transition with hysteresis
  (6.7, Formula 1) or discrete states (``states.discrete``).  ``state_min.x`` must be lower
  than ``state_max.x``.
* Options of the control (``adaptive_envelope.control``): ``hc_method`` (AM1, AM2),
  ``annex_c_edition`` (``"2026"`` default, ``"2023"`` as used by the EPB Center case study:
  five values of Table C.4 differ), ``glare_policy`` (``"formula"``: Formula 3, 17 106 lx,
  default; ``"table_33_note"``: 4 500 lx) or ``glare_limit_lx``, ``daylight`` (Formula 4:
  ``a_dayl``, ``b_dayl``, ``H_gl_1m``), ``sleeping_hours``, ``daytime_hours``.
* Results: the hourly state of each adaptive window is returned in the columns
  ``adaptive_<window>_modes``, ``_state``, ``_fraction``, ``_ext_state``, ``_g``, ``_u``,
  ``_tau_v`` and ``_I_sol``.
* Post-processing (6.10, clause 8): ``compare_adaptive_with_reference``
  (``source/adaptive_envelope_workflow_52016_3.py``) runs the building with and without the
  adaptive element and returns the annual and monthly indicators of Table 6 (differences of
  heating and cooling needs and of under/over-temperature, mean U, g and tau_V per window),
  the use of the states and the conformity report with the selected methods and options.
* Limits: single-zone engines only (the multizone engine raises an error for a window with
  ``adaptive_envelope``).  Not implemented: detailed elements (6.4.4, 6.5.2) and actively
  ventilated cavities (6.4.3.3), which need ISO 15099; the methods the standard does not work
  out (RAD-MODE AC2, GLARE-MODE AC2/AC3, HFLOW-MODE, control type ``CNTRL_INTEGR``,
  user-behaviour approaches 2 and 3).

Verification of the EN ISO 52016-3 implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Hour-by-hour comparison with the EPB Center spreadsheet for EN ISO 52016-3 (case office,
  Athens, external venetian blind, automated control, 9 504 h): 9 501 states equal, U, g and
  tau_V identical on the hours with equal state.  The three differences are at the glare limit
  (the spreadsheet rounds it to 17 250 lx).  The previous-day window of HC-MODE and TINT-MODE
  agree on all hours.
* The state tables of Annex C (322 manual/automated rows) agree with the spreadsheet except the
  five values of Table C.4 (night, manual control) listed in ``docs/standards_errata.md``,
  together with the glare limit of Formula (3) versus Table 33 note a and the sign of the
  hysteresis term of Formula (1).
* Engine check on ASHRAE 140 Case 600 (south window as external roller blind, automated, office
  with heating and cooling): with extended state equal to retracted state the result equals
  the fixed window (4 417 kWh heating, 6 251 kWh cooling); with the blind (g 0.10 when extended)
  heating 4 999 kWh and cooling 1 682 kWh.
* New tests: ``tests/test_adaptive_envelope_52016_3.py`` (logic),
  ``tests/test_adaptive_envelope_52016_3_epb_case_study.py`` (spreadsheet and case study report,
  data in ``tests/data/``) and ``tests/test_adaptive_envelope_52016_3_engine.py`` (engine and
  workflow).

EN ISO 52000-1 weighted balance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checked against ISO/TR 52000-2:2017 (worked examples of 9.7.4, Tables 9-12 and 16, Figure 15);
the balance (Formulae 2, 19, 20, 23-28, 39) reproduces them.  Three of the changes below alter
results (interval clipping, RER numerator, carriers without factors):

* New: perimeters of the RER are configurable (``pe_perimeters``, ``pe_rer_perimeters_ren``,
  ``pe_rer_perimeters_tot``; Table B.24 defaults), time-dependent delivered and export factors
  (``pe_time_factors``; columns ``f_delivered_*`` / ``f_exported_*``) and the on-site electricity
  used by non-EPB services as a separate export flow (``non_epb_electricity_kWh_per_year``,
  Formulae 24 and 27).

* Fix: the balance is no longer clipped at 0 in each calculation interval.  Formula (2) is an
  annual balance and ISO/TR 52000-2 11.6.2.3 states that a negative result is valid (Table 15:
  -103 kWh non-renewable).  An interval of net export now offsets the deliveries of the other
  intervals.  ``allow_negative`` defaults to ``True`` (composer option ``pe_allow_negative``);
  ``False`` floors only the total of the balance at 0.  Present since the clipping option was
  introduced; affects hourly and monthly balances with export (``pe_balance``) and buildings
  with a net export.
* Fix: the export credit (Step A + ``k_exp`` x (B - A)) is attributed to the distant perimeter
  in the RER (Formula 17).  With the Table B.24 defaults the renewable numerator now counts the
  on-site and nearby production in full, as in ISO/TR 52000-2 Tables 12 and 16 (20,1 % instead of
  18,1 % in the 9.7.4 example, 0,105 instead of 0,098 for the office of Table 12); the RER with
  the distant perimeter selected is unchanged.
* Fix: an energy carrier without a weighting factor is rejected (``MissingWeightingFactors``, a
  ``ValueError``) instead of being weighted silently with 1.0 non-renewable and 0 CO2; the
  factors are input data (Table A.16).  Result-changing only for custom carriers that relied on
  the silent default.  The ``iso_52000_1_b16`` set now covers all the classes of Table B.16
  (``coal``/``fossil_solid``, ``oil``/``fossil_liquid``, ``fossil_gaseous``, ``biofuel_solid``,
  ``biofuel_liquid``, ``biofuel_gaseous``/``biogas``).  Time-dependent factors given as columns
  can replace the static ones (the export defaults to the delivered columns).
* New: the exported energy that is not valorised in the performance, ``E_exp;el;avl`` =
  exported x ``(1 - k_exp)``, and the weighted resources used to produce it (Step A) are
  reported (Formulae 21-22): ``E_exp_avl_total_kWh``, ``E_exp_avl_<carrier>_kWh``,
  ``PE_exp_avl_nonren_kWh``, ``PE_exp_avl_ren_kWh``, ``PE_exp_avl_total_kWh``,
  ``CO2_exp_avl_kg`` (hourly columns and summary); the composer adds ``E_exp_avl_kWh``,
  ``PE_exp_avl_nonren_kWh`` and ``PE_exp_avl_ren_kWh`` to its summary.  Zero for ``k_exp = 1``.
* Open: Table B.30 treats exported wind electricity under Step A only; the calculator applies
  Step B to every exported electricity (informative default of Annex B, national choice).

2.1.3 (2026-11-06)
------------------

Engine
~~~~~~

* Fix: ``Areal_heat_capacity_of_element`` shared one node index between the surfaces of
  the ``class_e`` and ``class_m`` branches, so an opaque element listed after a ground
  floor took the ground floor's node (EN ISO 52016-1 eq. 45, 47b, 50b, 50e) and the
  result depended on the order of the surfaces.  The nodes are now fixed per element
  type.  Result-changing for ``class_e`` / ``class_m`` buildings with a ground floor
  listed before an opaque element (present since 2.1.0, also in 2.1.2).

2.1.2 (2026-11-06)
------------------

Engine
~~~~~~

* Moisture content and latent load of the single-zone engine now follow EN ISO 52016-1
  6.5.14 (eq. 71-84) with ideal humidity control: separate humidification and
  dehumidification set-points, the storage term of the zone air volume (the moisture
  content is carried from hour to hour), moisture production ``moisture_production_kg_s``
  and absorption ``moisture_absorption_kg_s``, the moisture content the mechanical supply
  needs to cover the load (eq. 78-79) and, when
  ``latent_system_supply_moisture_kg_kg`` is given, the system-specific content and loads
  (eq. 80-83).  New columns ``Phi_HU_W``, ``Phi_DHU_W``, ``G_HU_kg_s``, ``G_DHU_kg_s``,
  ``X_int_kg_kg_da``, ``X_set_min_kg_kg_da``, ``X_set_max_kg_kg_da``,
  ``X_sup_HU_req_kg_kg_da``, ``X_sup_DHU_req_kg_kg_da``; the ``Q_latent_*`` and
  ``Q_H_latent_*`` columns are kept.
* Result-changing: the constants are those of Table 20 (``ca`` 1006 J/(kg K), ``hwe``
  2.466e6 J/kg) and the default set-points are 25 % (humidification) and 60 %
  (dehumidification), the EN 16798-1 limits, instead of a single 50 %.  Pass
  ``latent_indoor_rh_pct=50`` to reproduce the single set-point.  The multizone engine
  has no moisture calculation.
* Composer: add the ``lighting`` scenario key (EN 15193-1, ``LightingSystemCalculator``
  options plus ``profile_24h`` / ``weekend_profile_24h``).  The lighting electricity
  now joins the site load, so it is seen by the PV self-consumption, the grid import,
  the final energy and the primary energy; new summary keys ``E_lighting_kWh``,
  ``LENI_kWh_m2a`` and an ``E_lighting_kWh`` hourly column.  The lighting heat is not
  fed back to the zone, whose gains stay those of the building JSON.

* Ground floor mass classes (EN ISO 52016-1 6.5.7.3): the fixed-ground heat capacity
  ``kgr`` is now put on node pl2 for every mass-position class (eq. 50f), not only for
  ``class_i``, and the ``class_ie`` ground floor splits ``km`` over nodes pl3 and pl5
  (eq. 50c) instead of pl1 and pl5.  ``kgr`` stays at the placeholder 1e6 J/(m2 K)
  (``KAPPA_GR_FIXED_GROUND``) until it is taken from ISO 13370/13789.
* Result-changing: slab-on-ground floors that are not ``class_i`` (e.g. ASHRAE 140
  Case 900) now include the ground heat capacity.

Primary energy
~~~~~~~~~~~~~~

* ``PrimaryEnergyAccountingCalculator`` now follows EN ISO 52000-1:2017: the exported
  energy is weighted with Step A (resources used to produce it) and Step B (resources
  avoided by the grid), ``E_exp = E_A + k_exp (E_B - E_A)`` (eq. 20-28, option ``k_exp``,
  Table B.17); the on-site renewable production (PV, wind, solar thermal, environment
  heat, columns ``E_onsite_*_kWh``) counts as delivered renewable energy (Table B.30);
  new outputs ``RER`` (eq. 17, perimeters of Table B.24, option ``perimeters``),
  ``PE_net_ren_kWh`` and greenhouse gas emissions ``CO2_net_kg`` (9.6.3, factor key
  ``co2`` in kg/kWh); ``factor_set="iso_52000_1_b16"`` selects the Table B.16 defaults;
  ``allow_negative`` lifts the clip at 0 (eq. 2 allows a negative balance).  Without
  on-site columns and with ``k_exp = 1`` the result is unchanged.
* Composer: the PV generation and the heat pump environment heat are passed as on-site
  renewable energy, the exported CHP electricity gets a Step A factor from the
  fuel allocated to the electricity (eq. 4-7, energy allocation).  New options
  ``pe_k_exp``, ``pe_factor_set``, ``pe_allow_negative``; new summary keys
  ``PE_net_ren_kWh``, ``RER``, ``CO2_net_kg``, ``E_onsite_renewable_kWh``.
* Result-changing: with a PV system or a heat pump the renewable primary energy
  (``PE_net_ren_kWh``, ``PE_net_total_kWh``) now includes the on-site renewable
  energy; the non-renewable figures are unchanged.

2.1.1 (2026-10-21)
------------------

Engine
~~~~~~

* Add the optional ``building_parameters["interior_solar_distribution"]``
  input to the single-zone engine: fractions of the transmitted solar
  radiation absorbed by the floor, ceiling and the four walls (a wall's share
  includes its windows) and a ``lost`` fraction reflected back out through the
  windows.  The fractions must be non-negative and sum to 1 (+/- 0.01).  When
  absent, the area-weighted ISO 52016-1 distribution is unchanged.

Validation
~~~~~~~~~~

* Add the 25 in-depth cases of ASHRAE 140-2023 Section 7.2.3 (195, 200, 210,
  215, 220, 230, 240, 250, 270, 280, 290, 300, 310, 320, 395, 400, 410, 420,
  430, 440, 450, 460, 470, 800, 810): ``in_depth_cases.py``,
  ``run_in_depth.py`` and an HTML report ``reports/in_depth.html``.  The
  standard gives no acceptance ranges for them; the report shows the
  difference of each case from its base case.
* ``build_bestest_case(..., solar_distribution="table")`` applies the interior
  solar distribution fractions of Tables 7-13, 7-15, 7-20 and 7-24.

Shading (ISO/TR 52016-2)
~~~~~~~~~~~~~~~~~~~~~~~~

* Add ISO 52016-1 Annex F Method 2: ``building_parameters["annex_f_shading_method"] = 2``
  also reduces the diffuse and ground-reflected radiation of a window
  (ISO/TR 52016-2:2017, F.6.3, eq. F.8-F.18) for overhangs, side fins and a
  remote skyline (``skyline_elevation_deg``), through the new
  ``diffuse_shading_reduction_factors``.  Method 1 stays the default and is
  unchanged.  The skyline-segment factors of F.6.2 (eq. F.4-F.5) and window
  setbacks are not implemented.

Generation (EN 15316-4-3, EN 15316-5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fix the second-order term of the solar collector efficiency (EN 15316-4-3
  eq. 52): ``a2 * T*^2 * I`` was missing the irradiance factor, so the
  efficiency was overestimated.  Solar thermal Method 3 results change.
* Solar thermal Method 3: add the Table 14 outputs (``Q_sol_gen_kWh``,
  ``Q_sol_col_out_kWh``, ``Q_sol_loop_out_kWh``, ``Q_sol_loop_out_req_kWh``,
  ``Q_sto_sol_in_kWh``, ``Q_sol_loop_ls_kWh``, ``Q_sol_loop_ls_rbl_kWh``,
  ``E_sol_aux_kWh``, ``N_sol_it``, ``SOL_FINAL``), the controller power
  (``controller_power_W``), recoverable loop losses (eq. 61), the
  collector/storage iteration (pass ``storage_response=`` to
  ``RenewableEnergySystemCalculator``; ``MixedSolarTank`` is a simple model) and
  the collector-plane irradiance from DNI/DHI with site and orientation.
  ``loop_loss_clip`` (default True) and ``aux_literal_eq60`` select between the
  documented reading and the literal text of eq. 56 and eq. 60;
  ``require_plane_irradiance`` refuses the horizontal-irradiance fallback.
* PV: add Method 4 (``"method_4"``, eq. 62-64), Method 5 (``"method_5"``,
  monthly, eq. 65-67), Method 6 from DNI/DHI, and the Table 25 outputs.
* Add ``LayeredStorage``, the multi-volume Method A of prEN 15316-5:2024
  (a draft), and ``StorageSystemCalculator`` option ``method_a``.  Pump
  auxiliary energy is not modelled on this path.

2.1.0 (2026-10-15)
------------------

Validation
~~~~~~~~~~

* Add two validation harnesses with one HTML report per case and an index:
  ``examples/ashrae140_bestest/`` (ANSI/ASHRAE Standard 140-2023, Section 7,
  cases 600-695, 900-995 and the six free-float cases, Case 960 on the multizone
  engine) and ``examples/iso52016_bestest/`` (EN ISO 52016-1:2017 clause 7.2,
  cases 600, 640, 900, 940, 600FF and 900FF).  Both Thermal Fabric groups of
  Annex A3 pass (Low Mass 20/21, High Mass 17/19, Table A3-14).
* The Annex A3 acceptance ranges and the ISO 52016-1 Tables 28-34 reference
  values are copyright content and are not distributed: they are read from
  git-ignored files under ``Standards/local_reference_data/`` (templates are
  provided) and the related checks are skipped when absent.
* Add tests for the standards' inputs (``test_ashrae140_2023_inputs``), the
  Annex A3 results, the ISO 52016-1 cases, side fins and the 3D geometry view.

Engine
~~~~~~

* ``Solar_absorption_of_element``: transparent elements always have
  ``a_sol = 0`` (ISO 52016-1 6.5.7.3).  Previously a window without
  ``solar_absorptance`` used ``a_sol = g_value`` (double-counted solar gain).
  Results of models with such windows change.
* ``external_convection_model`` (``doe2``, ``mowitt``, ``blast``,
  ``simplecombined``) and ``external_convection_h_min`` are now applied in the
  single-zone engine; they were accepted but had no effect.  The default
  ``table`` model is unchanged.
* Add ``hvac_control_variable`` (``"operative"`` default, or ``"air"``) to the
  single-zone engine.
* ``construction_class`` can be set per surface (falls back to the building
  value).
* Transmitted solar gains apply the window correction factor
  ``solar_transmittance_correction_factor`` (Formula E.3, default 0.90, Table
  B.22) and the window's ``frame_area_fraction``.
* Add side fins to the direct-beam shading factor (ISO 52016-1 Annex F,
  F.5/F.6 combined with F.19-F.22) through a per-window ``side_fins`` input.
* ``simulate_envelope_multizone_free_floating``: add ``c_int_per_A_us``
  (default 10000, unchanged).
* ``sanitize_and_validate_BUI`` validates ``solar_transmittance_correction_factor``.

Documentation
~~~~~~~~~~~~~

* README: validation notice, validation and engine-options sections, sources
  not distributed, grouped references; links are absolute so they work on the
  PyPI project page.
* ``docs/standards_errata.md``: EN ISO 52016-1 Tables 28/29 (Case 900) and
  ASHRAE 140-2023 Table 7-28.

Other changes in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Generalise the ISO 52016-1 §6.5.10 ventilation term to an affine pair
  ``(H_ve, S_ve)`` where ``S_ve = Σ H_k · T_source,k``.  This replaces the
  previous single outdoor-air conductance ``H_ve · T_outdoor`` and supports
  additive airflow streams (infiltration, mechanical supply, purge) at
  independent source temperatures.
* Add ``VentilationStream`` and ``VentilationBoundary`` frozen dataclasses with
  finite-value and non-negative-conductance validation.
* Add ``resolve_ventilation_boundary()`` resolver supporting the existing five
  ``ventilation_type`` configurations (backward-compatible) and an optional
  ``components`` list with per-component operation schedules
  (``component_multipliers``).
* Add ``constant_ach`` component type (explicit infiltration with per-zone
  volume).  Add ``prescribed`` component type for non-outdoor supply streams.
* Convert ``summer_night_purge`` to an additive outdoor-air stream
  (``H_purge = (boost_factor − 1) · H_base``), preserving the previous total
  for all-outdoor configurations.
* Wire ``S_ve`` through all four solver paths (multizone, legacy, causal,
  hybrid), replacing ``H_ve · T_outdoor`` in matrix assembly, ideal-HVAC
  backsolve, and Sankey heat-flow output.
* Add diagnostic outputs ``S_ve_<zone>``, ``T_ve_source_eq_<zone>``, and
  ``Q_ve_<zone>`` (sign convention: positive = heat leaving the zone).
* Fix ``_build_single_zone_building_object_for_core`` to use zone-level
  ``internal_gains`` schedules and ventilation/heating/cooling profiles with
  correct priority over global values.
* Fix ``Temperature_and_Energy_needs_calculation`` and the causal core to treat
  an explicitly passed ``None`` schedule kwarg the same as an absent key, so
  callers such as ``Temperature_and_Energy_needs_calculation_multizone_hybrid``
  do not cause ``generate_category_profile`` to receive ``None``.

0.0.0 (2024-03-26)
------------------

* First release on PyPI.

0.0.6 (2024-04-03)
------------------

* Modify readme.rst file
* Modify Authors.rst

0.0.7 (2024-04-18)
------------------

* Adding calculation of Domestic Hot Water - DHW NEed according to the ISO 12831-3
* Adding example of DHW calculation

2.0.0 (2025-11-19)
------------------

* Adding multizone calculation considering not - thermal adjacent zones
* Adding adiabtic wall calculation for adjacent zone having the same intended use ad the thermal zone
* Adding calculation of natural ventilation using the EN 16798-7
* Adding categories profiles based on the annex of EN 16798-1
* Adding calculation procedure of primary energy using the EN 15316-1
* Reshape the building and havc inputs in a more user-friendly way
* Adding example of multizone calculation

2.0.1 and 2.0.2 (2025-11-24)
------------------
* Update information regarding authors, license, etc.