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