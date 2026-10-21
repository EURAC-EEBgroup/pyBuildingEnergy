# ASHRAE Standard 140-2023 — Section 7, cases 600–695 and 900–995

This folder runs the **Building Thermal Envelope and Fabric Load Tests** of
Section 7 of ANSI/ASHRAE Standard 140-2023 with the ISO 52016-1 engine of
`pybuildingenergy` (single zone; multizone for Case 960).

> [!IMPORTANT]
> **Result (Annex A3, Table A3-14): both Thermal Fabric test groups pass.**
> **Low Mass 20/21 ranges met (18 required) — High Mass 17/19 (17 required).**
> HTML report for every case: [`reports/index.html`](reports/index.html). The acceptance-range values are not reproduced: see [Sources not distributed](#sources-not-distributed).

Reference document: `Standards/ASHRAE-D-86892.pdf` (ANSI/ASHRAE Standard
140-2023) and its accompanying files (`Std140_TF_Files`). Only the normative
ranges of Annex A3 are used as acceptance criteria; the example-program
results of Annex B8 are not criteria and are not used.

## How to run

```bash
# Annex A3 check of all cases (Table 7-7/7-9 convective coefficients)
PYTHONPATH=src python examples/ashrae140_bestest/run_comparison.py
# sensitivity: combined coefficients of the same tables
PYTHONPATH=src python examples/ashrae140_bestest/run_comparison.py ashrae_combined
# HTML report of every case (600–695, 900–995, 600FF–980FF) + index
PYTHONPATH=src python examples/ashrae140_bestest/generate_reports.py
# in-depth cases of Section 7.2.3 (25 diagnostic cases) and their report
PYTHONPATH=src python examples/ashrae140_bestest/run_in_depth.py
PYTHONPATH=src python examples/ashrae140_bestest/generate_in_depth_report.py
# tests
python -m pytest -q tests/test_ashrae140_in_depth.py tests/test_ashrae140_2023_inputs.py tests/test_ashrae140_bestest.py \
    tests/test_ashrae140_low_mass.py tests/test_ashrae140_case600_raised_floor.py \
    tests/test_iso52016_side_fins.py
```

## Files

- `case_definitions.py`: inputs of cases 600–695 and 900–995 (`build_bestest_case`),
  two-zone Case 960 (`build_case_960`), structure of the Annex A3 range cases
  and loading of their values from the local (not distributed) file;
- `annex_a3_ranges.template.json`: template of the local range file;
- `run_comparison.py`: simulation, check against Tables A3-1 to A3-4 and
  Table A3-14 count; `run_free_float_case` for the six free-float cases;
- `generate_reports.py`: one HTML report per case in `reports/`;
- `in_depth_cases.py`, `run_in_depth.py`, `generate_in_depth_report.py`: the 25 in-depth
  cases of Section 7.2.3 (see below), their runner and `reports/in_depth.html`;
- `../geometry_3d.py`: 3D view of the test cell used by the reports (shared with the ISO 52016-1 harness);
- `weather/USA_CO_Denver.Intl.AP.725650_TMY3.epw`: weather, identical hour by
  hour to the normative `725650TY.CSV`;
- tests: `tests/test_ashrae140_2023_inputs.py` (every input reproduces the
  value of the standard: resistances, heat capacities, U-values, infiltration,
  window, site, shading, cases 960/980/995), `tests/test_ashrae140_bestest.py`
  (annual results against Annex A3: 600, 900, 630−620, 960−900),
  `tests/test_ashrae140_low_mass.py`, `tests/test_ashrae140_case600_raised_floor.py`,
  `tests/test_iso52016_side_fins.py` (side fins, ISO 52016-1 F.5, F.6, F.22).

## Acceptance criteria

Annex A3 gives ranges for the annual heating and sensible cooling loads:
absolute values for cases 600 and 900, differences from the parent case for
all other cases (Tables A3-1 to A3-4). A result is within range if it is ≥ the
lower limit and ≤ the upper limit. A test group passes when the number of
ranges met reaches the minimum of Table A3-14:

| Group | Tables | Ranges | Minimum |
|---|---|---:|---:|
| Thermal Fabric Low Mass | A3-1, A3-2 | 21 | 18 |
| Thermal Fabric High Mass | A3-3, A3-4 | 19 | 17 |

Peak loads have no Annex A3 range and are reported as diagnostics only.
Free-float cases have no Annex A3 range.

## Sources not distributed

The acceptance-range values are **not reproduced** in this repository: they
are copyright ASHRAE and we are not authorized to share parts, tables or
limits of the standard. They are printed in:

| Content | Where in the standard |
|---|---|
| Annual heating load ranges, low mass (600–695) | ANSI/ASHRAE Standard 140-2023, Annex A3, **Table A3-1** |
| Annual sensible cooling load ranges, low mass | Annex A3, **Table A3-2** |
| Annual heating load ranges, high mass (900–995) | Annex A3, **Table A3-3** |
| Annual sensible cooling load ranges, high mass | Annex A3, **Table A3-4** |
| Minimum number of ranges to pass each group | Annex A3, **Table A3-14** |

Annex A3 was introduced by Addendum b to Standard 140-2020 and is
incorporated in Standard 140-2023 (Annex C).

**Owning Standard 140 is enough to see and check everything:** copy the
ranges into `Standards/local_reference_data/ashrae140_2023_annex_a3.json`
following the template [`annex_a3_ranges.template.json`](annex_a3_ranges.template.json)
(it names the table each range comes from; `Standards/` is git-ignored, or
point the `ASHRAE140_ANNEX_A3_FILE` environment variable at the file). With
the file, the runner, the tests and the reports give PASS/FAIL for every
range; without it the cases still run and the result reads "not available".
The range values are never printed in the reports or in the output.

## Results

`ashrae_convective` coefficients, thermostat on zone air temperature.

| Case | Heating [MWh] | Cooling [MWh] | | Case | Heating [MWh] | Cooling [MWh] |
|---|---:|---:|---|---|---:|---:|
| 600 | 4.417 | 6.251 | | 900 | 1.779 | 2.944 |
| 610 | 4.475 | 4.400 | | 910 | 2.032 | 1.543 |
| 620 | 4.571 | 4.395 | | 920 | 3.290 | 2.965 |
| 630 | 4.805 | 3.291 | | 930 | 3.878 | 2.246 |
| 640 | 2.696 | 6.015 | | 940 | 1.234 | 2.887 |
| 650 | — | 5.349 | | 950 | — | 1.014 |
| 660 | 3.881 | 3.380 | | 960 (back zone) | 2.824 | 1.140 |
| 670 | 5.935 | 6.537 | | 980 | 0.448 | 4.106 |
| 680 | 2.277 | 6.725 | | 985 | 2.561 | 6.851 |
| 685 | 4.866 | 9.147 | | 995 | 1.067 | 7.537 |
| 695 | 2.742 | 9.211 | | | | |

| Group (Table A3-14) | Ranges met | Required | Result |
|---|---:|---:|---|
| Thermal Fabric Low Mass | 20 of 21 | 18 of 21 | **pass** |
| Thermal Fabric High Mass | 17 of 19 | 17 of 19 | **pass** (no margin) |

Ranges not met:

- 650 − 600 cooling: −0.902 — FAIL (outside the ASHRAE 140 acceptance range, Case 650 − Case 600; 2020 edition, Addendum b — Annex A3, Table A3-2)
- 900 cooling: +2.944 — FAIL (outside the ASHRAE 140 acceptance range, Case 900; 2020 edition, Addendum b — Annex A3, Table A3-4)
- 920 − 900 cooling: +0.022 — FAIL (outside the ASHRAE 140 acceptance range, Case 920 − Case 900; 2020 edition, Addendum b — Annex A3, Table A3-4)

Three ranges are met but very close to the edge of the range: 960 − 900
cooling (−1.804), 985 − 900 heating (+0.782) and 920 − 900 heating (+1.510).
The high-mass group has no margin: one more range out would make it fail.

The reports give the result of every range with the same wording. The range
values are not reproduced in this documentation: they are in Tables A3-1 to
A3-4 of Annex A3 (ANSI/ASHRAE Standard 140-2023, incorporating Addendum b to
Standard 140-2020).

Free-float cases (zone air temperature, diagnostics):

| Case | Mean [°C] | Minimum [°C] | Maximum [°C] |
|---|---:|---:|---:|
| 600FF | 25.67 | −12.94 | 67.56 |
| 650FF | 20.01 | −17.64 | 66.50 |
| 680FF | 32.28 | −8.63 | 77.16 |
| 900FF | 25.73 | 1.25 | 45.93 |
| 950FF | 14.94 | −13.28 | 37.60 |
| 980FF | 32.29 | 9.97 | 52.46 |

Case 960 sun zone (free-floating): mean 28.75 °C, minimum 5.08 °C, maximum
50.82 °C.

Sensitivity to the surface coefficients (same inputs):

| Coefficients | 600 H / C | 900 H / C | Low Mass | High Mass |
|---|---|---|---:|---:|
| `ashrae_convective` (Tables 7-7/7-9, convective) | 4.417 / 6.251 | 1.779 / 2.944 | 20/21 | 17/19 |
| `ashrae_combined` (Tables 7-7/7-9, combined) | 3.921 / 6.611 | 1.595 / 3.678 | 20/21 | 12/19 |

## HTML reports

`generate_reports.py` writes into [`reports/`](reports/index.html) one page
per case (600–695, 900–995, 600FF, 650FF, 680FF, 900FF, 950FF, 980FF) and an
index with the Annex A3 check. Each page shows:

- annual loads and peaks (with date and time), the applicable Annex A3 range
  case and its result;
- a rotatable 3D view of the test cell for the case (windows, overhang and fins, glazing type, increased
  insulation, night-ventilation fan, sunspace), drawn from the geometry of the standard's figures;
- the inputs applied surface by surface (area, U, κ, coefficients,
  absorptance, g, F_w, shading), thermostat and air flows;
- monthly loads and monthly peaks;
- hourly values on the days required by Table 7-47 (February 1, July 14);
- the daily annual profile of loads and temperatures;
- for 900FF, the 1 °C temperature bin frequencies (Section 7.3.7);
- the annual energy-balance terms of the engine.

## Inputs applied

### Base Case 600 (Section 7.2.1)

| Parameter | Specification of the standard |
|---|---:|
| Geometry | 48 m², 8 m × 6 m, height 2.7 m |
| Volume | 129.6 m³ |
| Windows | 12 m² facing south (two 3 × 2 m, sill 0.2 m), unshaded |
| Infiltration | 0.5 ACH continuous at site altitude |
| Site | Denver International Airport, 1650 m (Table A1-1) |
| Internal gains | 200 W continuous, 100% sensible, 60% radiative |
| Setpoints | heating 20 °C, cooling 27 °C |
| Control | zone air temperature, non-proportional thermostat |
| Mechanical system | 100% efficient, 1000 kW, no losses |

Case 900 uses the walls and floor of Table 7-27; the roof does not change.
Cases 910–950 are the corresponding 600-series cases with the Case 900
constructions; 980 uses Table 7-31, 985 is Case 900 with the 20/20 thermostat
and 995 is Case 980 with the 20/20 thermostat (Table 7-14).

### Mapping onto the ISO 52016-1 engine

- **Constructions.** The engine takes an air-to-air U-value per surface and
  recovers the construction resistance as R_c = 1/U − r_si − r_se with the
  coefficients of the same surface. Each U-value is built from the
  layer-by-layer resistance (Tables 7-2, 7-25, 7-27, 7-31) and the
  coefficients set on the surface, so R_c equals the value of the standard.
  κ = Σ ρ·c·d over all layers. For the Case 900 floor, Table 7-28 prints
  25.256 m²K/W, but the layers of Table 7-27 add up to 25.246 (value
  confirmed by Table 7-32): the layers are used.
- **Surface coefficients (Sections 7.2.1.9.3 and 7.2.1.10.3).** The engine
  uses constant convective coefficients but computes long-wave exchange every
  hour (between interior surfaces through h_ri, to the sky through h_re).
  That is case (b)(1): the **convective** coefficients of Tables 7-7 and 7-9
  apply (interior/exterior: walls 2.2/11.9, roof 1.8/14.4, floor 2.2/0.8,
  window 2.4/8.0 W/m²K). The `ashrae_combined` option applies the combined
  coefficients (case (b)(2)) with h_ri = h_re = 0.
- **Raised floor (Section 7.2.1.5).** Outdoor air below the floor, no solar
  radiation on the underside, zero wind, no ground coupling: an `OUTDOORS`
  surface with solar absorptance 0. Its sky view factor is 1e-6 rather than 0,
  because the engine treats a horizontal surface with sky view factor 0 as
  ground contact. Its tilt is 0 rather than 180: the engine's orientation
  classifier maps tilt 180 to a vertical bucket, and in Case 900 the floor
  would be aggregated with the north wall.
- **Infiltration and night ventilation (Tables 7-4 and 7-16, Annex B3.1).**
  The `constant_ach` component uses ρ·c_p = 1.204 × 1006 J/(m³K) with no
  altitude correction. The air flows are scaled to give the same mass flow as
  at the site (ρ = 1.0156 kg/m³, c_p = 1015 J/(kg K)): 0.5 → 0.426 h⁻¹ and
  1700 m³/h → 11.17 h⁻¹. The 0.414 of Table 7-4 is the same correction
  referred to a sea-level density of 1.2255 kg/m³.
- **Window (Tables 7-10 to 7-12).** g_value = normal-incidence SHGC 0.769; the
  incidence-angle dependence enters the factor F_w (ISO 52016-1, Formula
  E.3), computed from the normative weather as an irradiance-weighted annual
  mean: angular SHGC of Table 7-12 (interpolated in cos θ, Section
  7.2.1.11.3) for beam radiation, hemispherical SHGC 0.670 for diffuse
  radiation (isotropic sky, ground reflectance 0.2). F_w = 0.859 south, 0.882
  east, 0.875 west. The 660 and 670 windows (Tables 7-17 to 7-24) are treated
  the same way.
- **Site (Table A1-1).** 39.833 N, 104.650 W, 1650 m, UTC−7.
- **construction_class** per surface: Case 600 walls and roof `class_d`,
  floor `class_i`; Case 900 walls and floor `class_i`, roof `class_d`.

### Overhangs and side fins (cases 610, 630, 910, 930)

Geometry of Figures 7-1, 7-3, 7-5 and 7-6, computed with ISO 52016-1 Annex F,
Method 1 (direct radiation shaded, diffuse radiation not modified):

- **610/910**: a 1.0 m deep overhang at roof level across the whole south
  wall. Its underside is 0.5 m above the top of the windows (sill 0.2 m +
  window 2.0 m, wall 2.7 m): the "infinite length" overhang of F.3.3.1,
  Formula F.4 (D = 1.0 m, L = 0.5 m).
- **630/930**: on each east and west window (3 × 2 m, centred, sill 0.2 m) a
  1.0 m deep, 3 m wide overhang at roof level (L = 0.5 m) and two 1.0 m deep
  fins flush with the window edges from the roof to the ground (Formulas F.5
  and F.6, combined with F.19–F.22). The window-wide overhang closed by the
  fins is reproduced exactly by F.22: the part of the window the finite
  overhang would leave sunlit at oblique sun is the strip already shaded by
  the fin on the sun's side.

The engine accepts side fins per window with:

```python
"side_fins": {
    "left":  {"depth": 1.0, "distance": 0.0},   # left/right as seen looking outwards
    "right": {"depth": 1.0, "distance": 0.0},
}
```

### Case 960 (sunspace, Section 7.2.2.2.7)

Computed with the multizone engine (`simulate_envelope_multizone_free_floating`,
`build_case_960`):

- conditioned back zone: as Case 600, with the south wall and windows replaced
  by the common wall; infiltration and internal gains as Case 600;
- free-floating sun zone: 2 × 8 × 2.7 m (43.2 m³), Case 900 construction,
  5.4 m² east/west walls, two 3 × 2 m windows as in Case 900 raised by 0.3 m,
  no internal gains, 0.5 ACH infiltration corrected for altitude;
- 0.20 m concrete common wall (Table 7-29: R = 0.392 m²K/W, κ = 280 kJ/m²K)
  with interior surface coefficients on both faces;
- no air exchange between the zones (Section 7.2.2.2.7.6).

On a single zone the multizone engine reproduces the single-zone engine
exactly for cases 600 and 900 (4.417/6.251 and 1.779/2.944 MWh, same peaks),
so the 960 − 900 difference compares the same model. The `c_int_per_A_us`
parameter of the multizone engine (default 10000 J/m²K, unchanged) allows 0,
as used for the empty test cell.

### Free-float cases (Section 7.2.2.3)

600FF, 650FF, 680FF, 900FF, 950FF and 980FF are the base cases without heating
or cooling; 650FF and 950FF keep the night vent fan. Outputs: mean, minimum
and maximum zone air temperature (Section 7.3.6), hourly values on February 1
or July 14 (Table 7-47) and, for 900FF, the 1 °C bin frequencies (Section 7.3.7).

## Additional outputs of Case 900

Monthly loads (Section 7.3.2.4):

| Month | Heating MWh | Cooling MWh | Peak heating kW | Peak cooling kW |
|---:|---:|---:|---:|---:|
| January | 0.293 | 0.037 | 2.345 | 1.721 |
| February | 0.313 | 0.008 | 2.638 | 0.794 |
| March | 0.124 | 0.090 | 1.793 | 1.762 |
| April | 0.267 | 0.047 | 2.049 | 1.583 |
| May | 0.029 | 0.188 | 1.284 | 2.048 |
| June | 0.000 | 0.528 | 0.000 | 2.387 |
| July | 0.000 | 0.507 | 0.000 | 2.396 |
| August | 0.000 | 0.612 | 0.000 | 2.688 |
| September | 0.000 | 0.555 | 0.022 | 2.945 |
| October | 0.093 | 0.280 | 1.443 | 2.836 |
| November | 0.321 | 0.050 | 2.375 | 1.813 |
| December | 0.341 | 0.042 | 2.224 | 1.542 |

Annual peaks: heating 2.638 kW (February 9, 05:00), cooling 2.945 kW
(September 11, 14:00). The hourly values of February 1 and July 14 (Section
7.3.8) are in [`reports/case_900.html`](reports/case_900.html).

## In-depth cases (Section 7.2.3)

The 25 cases 195, 200, 210, 215, 220, 230, 240, 250, 270, 280, 290, 300, 310, 320, 395,
400, 410, 420, 430, 440, 450, 460, 470, 800 and 810 change one input at a time from a base
case (Informative Table 7-33). The standard gives **no acceptance range** for them: the
harness prints the annual and peak loads and the difference of each case from its base
(`DIFFERENCES` in `in_depth_cases.py`). Example results of other programs are not reproduced.

Mapping onto the engine:

- **High-conductance (HC) wall elements** (Tables 7-35 to 7-37) replace the south
  windows in the 200- and 400-series: opaque surfaces with the layers of Table 7-36,
  no solar transmission, and their own construction class so that they are not aggregated
  with the walls.
- **Infrared emittance** (210, 215, 200, 195): the engine derives the radiative coefficients
  from the emittance (h_ri = 5.13 eps/0.9, h_re = 4.14 eps/0.9); the convective coefficients
  are those of Tables 7-40 and 7-41.
- **Interior solar absorptance** (270, 280, 440, 810): the engine has no such input, so the
  interior solar distribution fractions of Tables 7-43 to 7-45 are applied, as
  Sections 7.2.3.9.3.2, 7.2.3.10.2.2 and 7.2.3.12.2.2 permit
  (`--solar-distribution area` runs the default distribution, in which 280 = 270).
- **Combined coefficients** (450, 460, 470): Table 7-46 values as convective coefficients with a
  null radiative coefficient.
- Thermostat "20,20" in the 200-series and "20,27" in the 400-series and Case 320; infiltration
  corrected for altitude, zero in the base cases.

Tests: `tests/test_ashrae140_in_depth.py` (inputs as built and the sign of the differences).

## Limitations and documented deviations

- **Interior solar distribution (Table 7-13, Section 7.2.1.12.2).** By default the
  engine distributes transmitted solar by area, with a fraction going directly to
  the air: a deviation to be declared. The optional engine input
  `building_parameters["interior_solar_distribution"]` applies the fractions of
  Tables 7-13, 7-15, 7-20 and 7-24 instead (`build_bestest_case(...,
  solar_distribution="table")`); the Annex A3 results above use the default.
- **Shading of diffuse radiation.** ISO 52016-1 Annex F Method 1 shades direct
  radiation only; Method 2 refers to ISO/TR 52016-2 and is not implemented.
- **Sky temperature (Section 7.3.2.3).** The engine does not expose a `T_sky`
  series; this conditional output is not exported.
- **Other Table A3-14 groups** (cooling equipment, heating equipment, air-side
  equipment) and the Annex A2 documentation are not covered: this example
  alone is not a full declaration of conformity with Standard 140.

## Changes (October 2026)

Compared with the version derived from the 2017 EnergyPlus IDF:

- raised floor exposed to outdoor air instead of a 10 °C ground;
- 2023 window (SHGC 0.769) with F_w from Table 7-12 instead of 0.907;
- infiltration and night vent fan corrected for altitude;
- Table 7-7/7-9 coefficients instead of ISO 52016-1 Table 25;
- Denver International Airport coordinates instead of Denver-Stapleton;
- overhang 0.5 m above the windows instead of 1.0 m; 630/930 side fins
  actually modelled (they were described as `"vertical_overhang"` and the
  engine computed them as a second horizontal overhang);
- cases 960, 980, 985, 995 and the six free-float cases added.

Effect of the overhang and side-fin correction:

| Range case | Before | After |
|---|---|---|
| 610 − 600 cooling | −1.204 — PASS | −1.851 — PASS |
| 630 − 620 cooling | −0.476 — FAIL | −1.103 — PASS |
| 900 − 910 heating | −0.089 — PASS | −0.253 — PASS |
| 930 − 920 heating | +0.134 — FAIL | +0.589 — PASS |
| 930 − 920 cooling | −0.344 — FAIL | −0.720 — PASS |

PASS/FAIL: within/outside the ASHRAE 140 acceptance range of the given range
case (2020 edition, Addendum b).
