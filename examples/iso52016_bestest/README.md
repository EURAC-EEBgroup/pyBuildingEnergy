# ISO 52016-1:2017 verification — hourly method test cases (Section 7.2)

This folder runs the six verification cases of Section 7.2 of
**EN ISO 52016-1:2017** (Table 27: BESTEST 600, 640, 900, 940, 600FF and
900FF) with the single-zone hourly calculation of `pybuildingenergy`, using
the standard's own adapted inputs, and compares the results with Tables 28–34.

> [!IMPORTANT]
> **Result: 7 of the 8 annual energy needs (cases 600, 640, 900, 940) are within ±10% of the reference values (Tables 28–29);**
> **the annual mean operative temperature of all six cases is within 0.4 K (Table 30).** The only value above ±10%: Case 640 cooling (+10.3%).
> ISO 52016-1 defines no acceptance tolerance: Section 7.2 asks for the results and their deviations to be reported.
> HTML report for every case: [`reports/index.html`](reports/index.html). The reference values are not reproduced: see [Sources not distributed](#sources-not-distributed).

## How to run

```bash
PYTHONPATH=src python examples/iso52016_bestest/run_comparison.py      # six cases, annual comparison
PYTHONPATH=src python examples/iso52016_bestest/generate_reports.py    # HTML reports in reports/
python -m pytest -q tests/test_iso52016_bestest.py
```

The runner uses the standard's climate file when it is available locally:

```text
Standards/ISO_52016_1_BESTEST_ClimData_2016.08.24.xls   (requires xlrd)
```

The file is ISO copyright content and is not distributed with the
repository. Without it the runner uses the Denver TMY3 EPW as an
approximation; TMY3 results are not the ones reported here.

## Cases verified (Table 27)

| Test | Case | Construction | Thermostat (7.2.2.15) |
|---:|---|---|---|
| 1 | 600 | lightweight | continuous 20/27 °C |
| 2 | 640 | lightweight | intermittent: heating 10 °C from 23:00 to 07:00, 20 °C from 07:00 to 23:00; cooling 27 °C at all hours |
| 3 | 900 | heavyweight | continuous 20/27 °C |
| 4 | 940 | heavyweight | intermittent |
| 5 | 600FF | lightweight | free floating |
| 6 | 900FF | heavyweight | free floating |

## Inputs applied

- geometry of Section 7.2.2: single zone 8 m × 6 m × 2.7 m, 48 m²,
  129.6 m³, 12 m² of south-facing window;
- operative-temperature control (the engine's ISO default, 7.2.2.1);
- surface coefficients of Table 25 (ISO 13789);
- floor to outdoor air, not to the ground (Table 23 footnote a), with no sky
  view and no solar radiation on the underside, and the "downwards" interior
  coefficient of Table 25 (0.7 W/m²K);
- continuous infiltration of 0.41 ACH (7.2.2.14, corrected for altitude);
- window g_gl;n = 0.789, U_w = 2.984 W/(m² K), F_w = 0.90 (Table B.22);
- continuous internal gain of 200 W (7.2.2.13);
- constructions from Tables 23 and 24, mass-distribution classes per surface;
- temperature and irradiance from the standard's BESTEST climate file.

## Results with the ISO climate

Deviations from the reference values of the standard (the reference values
themselves are not reproduced, see "Sources not distributed"):

| Case | Heating kWh | Cooling kWh | Annual mean T_op °C |
|---|---:|---:|---:|
| 600 | 5,621 — **+9.5%** vs Table 28 | 8,142 — **+8.5%** vs Table 29 | 23.21 — −0.09 K vs Table 30 |
| 640 | 3,404 — **+9.4%** | 7,784 — **+10.3%** | 21.14 — −0.36 K |
| 900 | 1,940 — **+6.2%** | 3,576 — **+6.4%** | 24.21 — +0.01 K |
| 940 | 1,278 — **−1.9%** | 3,435 — **+5.3%** | 23.63 — −0.17 K |
| 600FF | — | — | 26.01 — +0.11 K |
| 900FF | — | — | 26.04 — +0.14 K |

Simulated peak loads (kW; reference in Table 31): Case 600 heating 4.357,
cooling 6.652; 640 6.079 and 6.568; 900 4.013 and 4.078; 940 6.943 and 4.078.
Simulated free-floating operative temperatures (°C; reference in Table 32):
600FF maximum 68.51, minimum −19.86; 900FF maximum 45.31, minimum −4.07.

The published HTML reports show the monthly results (Tables 28–30), the
hourly profiles of January 4 (Tables 33–34) and July 27 (for which the
standard prints no reference), with the table of the standard each reference
is printed in but without the values. With the standard and the local
reference file, `generate_reports.py --include-reference-values` produces the
full comparison for local use (not to be published).

## Errata of Tables 28 and 29 for Case 900

In Tables 28 and 29 the Case 900 heating and cooling columns are swapped:
the "heating" column has the summer profile of a cooling need and the real
heating column is missing. The comparison has to use the corrected values:
cooling = the column printed as heating; heating = the corrected monthly
sequence of the EPB Center's collection of comments on EN ISO 52016-1:2017
(comment no. 28, <https://epb.center/support/documents/iso_52016-1_details/>).
This is an external correction, not an official ISO reprint.

## Sources not distributed

The following ISO copyright content is not distributed with the repository,
because we are not authorized to share it:

- the reference values: EN ISO 52016-1:2017, clause 7.2.4, **Tables 28–34**
  (28 heating needs, 29 cooling needs, 30 mean operative temperature, 31 peak
  loads, 32 free-floating extremes, 33 hourly loads on January 4, 34 hourly
  operative temperature on January 4);
- the standard's BESTEST climate file (clause 7.2.2.12).

**Owning the standard is enough to reproduce everything:** copy the table
values into `Standards/local_reference_data/iso52016_1_2017_tables_28_34.json`
following the template [`iso_reference_tables.template.json`](iso_reference_tables.template.json)
(it names the table of every field) and place the climate file in
`Standards/`. The `Standards/` folder is git-ignored.

## Files

- `iso_case_definitions.py`: inputs of the six cases (`build_case(mass, control)`);
- `iso_climate_loader.py`: loader of the ISO BESTEST climate;
- `iso_reference_results.py`: loads the reference values from the local (not
  distributed) file and names the table of each one;
- `iso_reference_tables.template.json`: template of the local reference file;
- `run_comparison.py`: simulation of the six cases and annual comparison;
- `generate_reports.py`: one HTML report per case in `reports/`, each with a rotatable 3D view of the test cell
  (module `../geometry_3d.py`, shared with the ASHRAE 140 harness).

## Changes (October 2026)

- **Floor**: it was described as `OUTDOORS` but with sky view factor 0, which
  the engine treats as ground contact on a horizontal surface. The floor was
  therefore computed as ground-coupled (ISO 13370, including the perimeter
  thermal bridge of the ground model), against Table 23 footnote a. It now
  faces outdoor air; the results changed by +0.5 percentage points (Case 600
  heating from +9.0% to +9.5%) and +1 point (Case 900 heating from +5.2% to
  +6.2%).
- Cases 640, 940, 600FF and 900FF added.
- The reference values of Tables 28–34 were moved from the code to a local
  git-ignored file.
