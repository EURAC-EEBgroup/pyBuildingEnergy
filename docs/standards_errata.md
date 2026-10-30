# Possible errata in referenced standards

This file tracks places where a formula or table in a standard this library
implements appears to be internally inconsistent, as printed in the edition
on file under `Standards/`. These are not confirmed corrigenda from the
standards body (CEN/CENELEC) - they are observations made while implementing
the formula, kept here so the reasoning is auditable and the implementation
can be revisited if an official corrigendum is published. Each entry says
exactly what the implementation does differently from the literal text, and
why.

## EN 15316-4-1:2017, Eq.(63)-(66) - load factor (beta) definitions

**Location:** Clause 6.10 "Heating time and load factor", p. 25 of the PDF
on file (`Standards/BS EN 15316-4-1_2017.pdf`).

**What the standard prints:**

```
t_H = Q_H;gen;out / Pn                                    (59)
t_C = Q_C;gen;out / Pn                                    (60)
t_V = Q_V;gen;out / Pn                                    (61)
t_W = Q_W;gen;out / Pn                                    (62)

"If they are running parallel with the same priority the heating time
can be calculated with Formulae (63), (64), (65), (66) and (67)"

beta_H = Q_H;gen;out / (Pn . t_H)                          (63)
beta_C = Q_C;gen;out / (Pn . t_C)                          (64)
beta_V = Q_V;gen;out / (Pn . t_V)                          (65)
beta_W = Q_W;gen;out / (Pn . t_W)                          (66)
t_H;op = t_H;use . beta_H - t_C;use . beta_C
         - t_V;use . beta_V - t_W;use . beta_W             (67)
```

**The inconsistency:** Eq.(63)'s denominator uses `t_H`, the SAME symbol
Eq.(59) just defined as `t_H = Q_H;gen;out / Pn`. Substituting Eq.(59) into
Eq.(63) algebraically:

```
beta_H = Q_H;gen;out / (Pn . (Q_H;gen;out / Pn)) = 1
```

...identically, for any value of `Q_H;gen;out` (as long as it is nonzero).
The same substitution makes beta_C, beta_V and beta_W identically 1 too. A
formula whose result is always 1 regardless of its input does not "calculate"
a load factor - it contradicts clause 3.11's own definition of load factor
("ratio between the time with the boiler ON (burner 100 %) and the total
generator operation time"), which is a genuine, generally non-unity ratio.

Eq.(67) separately uses `t_H;use`, `t_C;use`, `t_V;use`, `t_W;use` - symbols
with a `;use` suffix, DIFFERENT from the bare `t_H`/`t_C`/`t_V`/`t_W` Eq.(59)-
(62) define. This is consistent with this standard family's (EN ISO 52000-1)
general convention that `;use` denotes "time of use" (an input - the period
during which a service is required), not a value derived from Eq.(59)-(62).
That strongly suggests Eq.(63)-(66)'s denominators were also meant to read
`t_H;use`, `t_C;use`, `t_V;use`, `t_W;use` (the time available/allotted to
each service), not the bare `t_H`/`t_C`/`t_V`/`t_W` as printed - the ";use"
subscript most plausibly dropped out of Eq.(63)-(66) during drafting/
typesetting while it survived intact in Eq.(67) immediately below.

**What this library does instead:** `BoilerGeneratorCalculator`'s
`priority_mode="dhw_priority"` option (see
`src/pybuildingenergy/source/generation_15316_4_1.py`,
`_compute_step_dhw_priority()`) computes each service's load ratio as

```
beta_X = Q_X_gen_out / (Pn * t_use)
```

i.e. Eq.(63)-(66) with `t_X;use` read as the full calculation step length
(this library's `t_use_h`, since a boiler generator here only ever serves
heating and DHW within one shared hourly step - there is no sub-period
scheduling between services within a step). This is the same `Q/(Pn*t)` form
the rest of this module already uses for its single combined `beta_gen`
elsewhere (the `priority_mode="proportional"` default, and every other
generation module in this package) - applied per service instead of once to
a blended combined load. It gives a genuine, non-degenerate ratio consistent
with clause 3.11's definition, instead of the literal text's identical 1.

Eq.(67)'s intent - DHW/cooling/ventilation running in priority ahead of
heating, heating getting whatever capacity and time is left over - is
realised as explicit sequential capacity allocation (DHW first, then
heating) rather than by evaluating the literal subtraction formula (which,
once beta=1 is accepted, degenerates to `t_H;op = t_H;use - t_C;use -
t_V;use - t_W;use`, a seconds-of-the-period subtraction that does not by
itself decide which service's demand is reduced when the subtraction goes
negative).

**Caveat:** this is this library's best-effort reading, not a verified
corrigendum. If CEN/CENELEC publishes an official correction to EN 15316-4-1
clause 6.10, this implementation should be revisited against it.

**See also:** the module docstring in `generation_15316_4_1.py` and the
`priority_mode` field on `BoilerConfig` for how this is exposed; the default
(`"proportional"`) is unaffected by this reinterpretation and keeps this
library's pre-existing combined-load behaviour unchanged.

## EN ISO 52016-1:2017, Tables 28 and 29 - Case 900 heating and cooling columns swapped

**Location:** Clause 7.2.4 "Results of the verification test cases", Tables
28 (sensible energy needs for heating) and 29 (sensible energy needs for
cooling), column "900".

**What the standard prints:** the Case 900 "heating" column has its largest
values in July-September and its smallest in January-February, and the
"cooling" column is close to zero all year. That is the profile of a cooling
need in the first column and of nothing in the second, for a Denver building.

**The inconsistency:** the printed "heating" column is the cooling need and the
real Case 900 heating sequence is missing; the same column error is not
present for Case 600, 640 and 940. The EPB Center's public "Collection of
comments and suggestions on EN ISO 52016-1:2017" (comment no. 28,
https://epb.center/support/documents/iso_52016-1_details/) gives the corrected
heating sequence and confirms that the printed heating column is the cooling
need.

**What the implementation does:** the validation harness
(`examples/iso52016_bestest/`) compares Case 900 with the corrected
sequences (cooling = the column printed as heating; heating = the sequence
of the EPB Center comment), loaded from a local file that is not
distributed with the repository (ISO copyright). This is an external
correction, not an ISO corrigendum.

## ANSI/ASHRAE Standard 140-2023, Table 7-28 - Case 900 raised-floor resistance

**Location:** Informative Table 7-28, "Total surface-to-surface" resistance of
the "High-Mass Case: Raised Floor", compared with Tables 7-27 and 7-32.

**What the standard prints:** 25.256 m2K/W in Table 7-28 (Case 900) and
25.246 m2K/W in Table 7-32 (Case 980, whose floor "is the same as the Case
900 floor").

**The inconsistency:** the layers of Table 7-27 (80 mm concrete slab,
k = 1.13 W/(m K); 1.007 m insulation, k = 0.04 W/(m K)) add up to 25.246
m2K/W, the value of Table 7-32. Table 7-28 differs by 0.010 m2K/W.

**What the implementation does:** the floor resistance is built from the
layers of Table 7-27 (25.246 m2K/W), in `examples/ashrae140_bestest/`.

