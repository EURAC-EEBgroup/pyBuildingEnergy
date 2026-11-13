2.2.0 (2026-11-13)
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

Verification
~~~~~~~~~~~~

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