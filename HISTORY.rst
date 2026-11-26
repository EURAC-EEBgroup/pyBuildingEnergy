=======
History
=======

Versions 1.0.7, 2.0.0 and 2.0.1 were published on PyPI without a Git tag;
their exact source commits have not been verified.

0.0.1 (2024-03-26)
------------------

* First release on PyPI.

0.0.5 (2024-03-26)
------------------

* Modfying authors and readme file

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

2.1.6 (2026-11-20)
------------------
* Multizone (EN ISO 52016-1) checked against ISO/TR 52016-2:
  * thermally unconditioned adjacent zones (6.4.5) with Formulae 54, 58-59, 65 and 68,
    in the single-zone and multizone engines;
  * air flows between zones, zoning checks, Option B, free-floating zones;
  * latent (moisture) balance per zone.
* Adaptive windows (EN ISO 52016-3) in the multizone engine:
  * one control context per zone, shading applied per window;
  * comparison of the adaptive building with its reference per zone;
  * daylight and artificial light per control state.
* EN 15316-2 emission per zone:
  * equivalent set-points recalculated together for all zones;
  * emitter capacity included in the thermal balance;
  * zones without an emitter run free-floating.
* System topologies for multizone buildings: central, per zone, and combined
  heating/cooling/DHW with a storage tank. Shared generator power is split between the
  zones hour by hour.
* EN 16798-1 comfort categories and capacity-shortage indicators.
* Details in CHANGELOG.rst.

2.1.7 (2026-11-23)
------------------
* First PyPI release of the 2.1.6 work, plus these fixes:
  * elements with 1/U - R_si - R_se <= 0 are rejected (negative window conductance);
  * archetype windows above U = 5.7 W/(m2K) set to 5.7.
* Details in CHANGELOG.rst.

2.2.0 (2026-11-25)
------------------
* EN 16798-5-1 air handling unit extended: moisture processes, heat recovery Annex D
  (EPB Center spreadsheet version, 2017 print selectable), distribution and leakage,
  recirculation, frost protection, fans and auxiliary energy, ground preheating, central
  multizone AHU with VAV, national data sets (EN, IT).
* New standalone modules: EN 16798-5-2 (monthly/yearly/bin method, extract air heat pump)
  and EN 16798-7 (air flow rates, mass balance of the zone), the latter coupled to the
  zone ventilation through an ``en16798_7`` component.
* Input validation for the new modules (finite values, ranges, inconsistent configurations).
* Details in CHANGELOG.rst.

2.2.1 (2026-11-26)
------------------
* Fix: the multizone simulation (``simulate_envelope_multizone_free_floating``) did not
  import the hourly profile generator, so with ``use_profiles=True`` every zone silently
  ran with flat 1.0 profiles (constant internal gains and ventilation factor). Zone
  profiles are now generated; a profile failure emits a ``RuntimeWarning``.
* ASHRAE 140-2023 Section 11 air-side tests (fan-coil and single-zone systems) for the
  EN 16798-5-1 AHU; checked on Python 3.12 and 3.13.
* Details in CHANGELOG.rst.
