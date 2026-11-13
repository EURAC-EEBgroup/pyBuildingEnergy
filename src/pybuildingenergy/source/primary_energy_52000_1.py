"""EN ISO 52000-1 weighted energy balance (primary energy, CO2, RER).

The calculator keeps the balance explicit at carrier level and follows the structure of
EN ISO 52000-1:2017:

* Formula (2): E_we = E_we;del - E_we;exp, with delivered energy weighted per carrier
  (Formulas 19 and 39) and the contribution of the on-site renewable sources (Table B.30).
* Formulas (20)-(28): exported energy weighted with the resources used to produce it
  (Step A) and with the resources avoided by the grid (Step B):
  ``E_we;exp = E_A + k_exp * (E_B - E_A)``.  ``k_exp`` (Table B.17, default 1) controls how much of
  the export is credited; with ``k_exp = 0`` the export has no effect on the performance.
* Formulae (21)-(22): the exported energy not valorised in the performance, ``(1 - k_exp)``, and the
  resources used to produce it (Step A) are reported as ``E_exp_avl_*`` and ``PE_exp_avl_*``.
* Formula (17): renewable energy ratio RER, for the perimeters of Table B.24.
* 9.6.3: greenhouse gas emissions with the same balance.

Weighting factors are input data (a national annex), not hard-coded: a carrier without a factor is
rejected (``MissingWeightingFactors``).  The default set keeps the
factors of the STREPIN examples, ``factor_set="iso_52000_1_b16"`` selects the informative
Table B.16 values. Time-dependent delivered and Step-B export factors can be supplied in the
input time series. The matching factor of Table B.32 is not separately approximated: the
composer uses actual hourly self-consumption before aggregating a monthly balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_KWH_EPS = 1e-12


# Weighting factors: nonrenewable / renewable primary energy (kWh/kWh) and CO2 (kg/kWh).
DEFAULT_PRIMARY_FACTORS: dict[str, dict[str, float]] = {
    "natural_gas": {"nonrenewable": 1.05, "renewable": 0.00, "co2": 0.22},
    "gas": {"nonrenewable": 1.05, "renewable": 0.00, "co2": 0.22},
    "electricity": {"nonrenewable": 2.17, "renewable": 0.47, "co2": 0.42},
    "district_heat": {"nonrenewable": 0.80, "renewable": 0.20, "co2": 0.26},
    "district_cooling": {"nonrenewable": 0.70, "renewable": 0.20, "co2": 0.26},
    "biomass": {"nonrenewable": 0.20, "renewable": 1.00, "co2": 0.04},
    "biomass_pellets": {"nonrenewable": 0.20, "renewable": 1.00, "co2": 0.04},
}

# EN ISO 52000-1:2017 Table B.16 (informative defaults, gross calorific value).
ISO_52000_1_TABLE_B16_FACTORS: dict[str, dict[str, float]] = {
    "natural_gas": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.220},
    "gas": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.220},
    "electricity": {"nonrenewable": 2.3, "renewable": 0.2, "co2": 0.420},
    "district_heat": {"nonrenewable": 1.3, "renewable": 0.0, "co2": 0.260},
    "district_cooling": {"nonrenewable": 1.3, "renewable": 0.0, "co2": 0.260},
    "biomass": {"nonrenewable": 0.2, "renewable": 1.0, "co2": 0.040},
    "biomass_pellets": {"nonrenewable": 0.2, "renewable": 1.0, "co2": 0.040},
    # Table B.16 rows 1-6 by class of carrier, with the usual names of the common fuels
    "fossil_solid": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.360},
    "coal": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.360},
    "fossil_liquid": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.290},
    "oil": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.290},
    "fossil_gaseous": {"nonrenewable": 1.1, "renewable": 0.0, "co2": 0.220},
    "biofuel_solid": {"nonrenewable": 0.2, "renewable": 1.0, "co2": 0.040},
    "biofuel_liquid": {"nonrenewable": 0.5, "renewable": 1.0, "co2": 0.070},
    "biofuel_gaseous": {"nonrenewable": 0.4, "renewable": 1.0, "co2": 0.100},
    "biogas": {"nonrenewable": 0.4, "renewable": 1.0, "co2": 0.100},
}

FACTOR_SETS: dict[str, dict[str, dict[str, float]]] = {
    "default": DEFAULT_PRIMARY_FACTORS,
    "iso_52000_1_b16": ISO_52000_1_TABLE_B16_FACTORS,
}

# Table B.16 rows 10-13: on-site solar, wind and environment energy.
ONSITE_FACTORS: dict[str, float] = {"nonrenewable": 0.0, "renewable": 1.0, "co2": 0.0}

# Delivered on-site renewable energy counted in the balance (Table B.30), with its input columns.
DEFAULT_ONSITE_ALIASES: dict[str, list[str]] = {
    "pv": ["E_onsite_pv_kWh"],
    "wind": ["E_onsite_wind_kWh"],
    "solar_thermal": ["E_onsite_solar_thermal_kWh"],
    "environment_heat": ["E_onsite_environment_heat_kWh"],
}
_ELECTRIC_ONSITE = ("pv", "wind")

# Perimeter of each delivered carrier (Table B.23); used for the RER (Table B.24).
DEFAULT_PERIMETERS: dict[str, str] = {
    "natural_gas": "distant",
    "gas": "distant",
    "electricity": "distant",
    "biomass": "distant",
    "biomass_pellets": "distant",
    "district_heat": "nearby",
    "district_cooling": "nearby",
}
_PERIMETERS = ("onsite", "nearby", "distant")


DEFAULT_CARRIER_ALIASES: dict[str, list[str]] = {
    "natural_gas": [
        "E_delivered_natural_gas_kWh",
        "E_gas_delivered_kWh",
        "Gas_final_kWh",
        "Gas_final_kWh_a",
        "EHW_fuel_in_kWh",
    ],
    "electricity": [
        "E_delivered_electricity_kWh",
        "E_grid_kWh",
        "E_grid_after_PV_kWh",
        "Grid_electricity_kWh",
        "Grid_electricity_kWh_a",
        "E_electricity_delivered_kWh",
    ],
    "district_heat": [
        "E_delivered_district_heat_kWh",
        "E_district_heat_delivered_kWh",
    ],
    "district_cooling": [
        "E_delivered_district_cooling_kWh",
        "E_district_cooling_delivered_kWh",
    ],
    "biomass": [
        "E_delivered_biomass_kWh",
        "E_biomass_delivered_kWh",
        "Biomass_final_kWh",
        "Biomass_final_kWh_a",
    ],
}


DEFAULT_EXPORT_ALIASES: dict[str, list[str]] = {
    "electricity": [
        "E_exported_electricity_kWh",
        "E_PV_export_kWh",
        "PV_export_kWh",
        "PV_export_kWh_a",
    ],
}

# Electricity produced on-site but used by non-EPB building services is an exported
# flow in 11.6.2.2/11.6.2.3 (Formulae 24 and 27), not delivered EPB energy.
DEFAULT_NON_EPB_EXPORT_ALIASES: dict[str, list[str]] = {
    "electricity": [
        "E_exported_electricity_to_non_epb_kWh",
        "E_exported_electricity_non_epb_kWh",
        "E_non_epb_electricity_from_onsite_kWh",
    ],
}


class MissingWeightingFactors(ValueError):
    """An energy carrier of the balance has no weighting factor (Table A.16 is input data)."""


@dataclass
class PrimaryEnergyAccountingResult:
    """Container returned by :class:`PrimaryEnergyAccountingCalculator`."""

    timeseries: pd.DataFrame
    summary: dict[str, float]
    inputs: dict[str, Any]


class PrimaryEnergyAccountingCalculator:
    """Weighted delivered/exported energy balance of EN ISO 52000-1.

    Options (``input_data``): ``carriers``, ``factor_set`` ('default' | 'iso_52000_1_b16'),
    ``primary_energy_factors`` / ``export_primary_energy_factors`` (per carrier, keys
    ``nonrenewable``, ``renewable``, ``co2`` or ``nren``, ``ren``), ``k_exp`` (Table B.17),
    ``export_credit_method`` ('symmetric' | 'none'; 'none' is ``k_exp = 0``), ``allow_negative``
    (default True: Formula 2 is an annual balance and a negative result is valid, ISO/TR 52000-2
    11.6.2.3; False floors the total of the balance at 0, never the single intervals),
    ``perimeters`` (carrier -> onsite|nearby|distant), ``rer_perimeters_ren`` and
    ``rer_perimeters_tot`` (Table B.24 defaults), ``carrier_aliases``, ``export_aliases`` and
    ``onsite_aliases`` for the input column names. Time-varying factors are optional input
    columns ``f_delivered_<carrier>_<nonren|ren|co2>`` and
    ``f_exported_<carrier>_<nonren|ren|co2>``; ``fD_`` and ``fB_`` are accepted aliases.

    Input columns: delivered energy per carrier, exported electricity, and optionally the
    on-site production ``E_onsite_pv_kWh``, ``E_onsite_wind_kWh``, ``E_onsite_solar_thermal_kWh``,
    ``E_onsite_environment_heat_kWh``.  The on-site production is counted as delivered
    renewable energy (Table B.30); the Step A weighting of the export cancels the part that
    leaves the building.  Without on-site production columns the Step A weighting is 0 and the
    export credit is ``k_exp`` times the Step B weighting.  Per-row Step A factors can be given
    with the columns ``fA_<carrier>_nonren``, ``fA_<carrier>_ren`` and ``fA_<carrier>_co2``
    (Formula 8 average over the generators, for example a cogenerator, Formula 4).
    ``E_exported_electricity_to_non_epb_kWh`` records electricity produced on-site and
    used by non-EPB services; it is reported separately but receives the same Step A/B
    treatment as grid export unless different per-row Step-B factors are supplied.
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        self.input_data = dict(input_data or {})
        self._load_options()

    def run_timeseries(self, data: pd.DataFrame) -> PrimaryEnergyAccountingResult:
        prepared = self._prepare_timeseries(data)
        results = self._simulate(prepared)
        return PrimaryEnergyAccountingResult(
            timeseries=results,
            summary=self._summarize(results),
            inputs=dict(self.input_data),
        )

    def run_annual(self, values: dict[str, Any]) -> PrimaryEnergyAccountingResult:
        """Run the same accounting on a one-row annual balance."""

        return self.run_timeseries(pd.DataFrame([values]))

    def factors(self, carrier: str, *, exported: bool = False) -> dict[str, float]:
        """Weighting factors (nonrenewable, renewable, co2) of a delivered or exported carrier."""

        return self._factors_for(carrier, exported=exported)

    def _load_options(self) -> None:
        cfg = self.input_data
        self.carriers = list(cfg.get("carriers", DEFAULT_CARRIER_ALIASES.keys()))
        self.export_credit_method = str(cfg.get("export_credit_method", "symmetric")).lower()
        if self.export_credit_method not in {"symmetric", "none"}:
            raise ValueError("export_credit_method must be 'symmetric' or 'none'.")
        self.k_exp = 0.0 if self.export_credit_method == "none" else float(cfg.get("k_exp", 1.0))
        if not 0.0 <= self.k_exp <= 1.0:
            raise ValueError("k_exp must be between 0 and 1.")
        self.allow_negative = bool(cfg.get("allow_negative", True))

        factor_set = str(cfg.get("factor_set", "default")).lower()
        if factor_set not in FACTOR_SETS:
            raise ValueError(f"factor_set must be one of {sorted(FACTOR_SETS)}.")
        self.default_factors = FACTOR_SETS[factor_set]
        self.primary_factors = _normalise_factor_map(
            cfg.get("primary_energy_factors", cfg.get("factors", {}))
        )
        self.export_factors = _normalise_factor_map(
            cfg.get("export_primary_energy_factors", cfg.get("export_factors", {}))
        )
        self.onsite_factors = {
            **{name: dict(ONSITE_FACTORS) for name in DEFAULT_ONSITE_ALIASES},
            **_normalise_factor_map(cfg.get("onsite_factors", {})),
        }
        self.carrier_aliases = {
            **DEFAULT_CARRIER_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("carrier_aliases", {})).items()},
        }
        self.export_aliases = {
            **DEFAULT_EXPORT_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("export_aliases", {})).items()},
        }
        self.non_epb_export_aliases = {
            **DEFAULT_NON_EPB_EXPORT_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("non_epb_export_aliases", {})).items()},
        }
        self.onsite_aliases = {
            **DEFAULT_ONSITE_ALIASES,
            **{k: list(v) for k, v in dict(cfg.get("onsite_aliases", {})).items()},
        }
        self.perimeters = {**DEFAULT_PERIMETERS, **dict(cfg.get("perimeters", {}))}
        self.rer_ren = tuple(cfg.get("rer_perimeters_ren", ("onsite", "nearby")))
        self.rer_tot = tuple(cfg.get("rer_perimeters_tot", _PERIMETERS))
        for perimeter in (*self.perimeters.values(), *self.rer_ren, *self.rer_tot):
            if perimeter not in _PERIMETERS:
                raise ValueError(f"perimeter must be one of {_PERIMETERS}, got {perimeter!r}.")

    def _prepare_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")

        out = pd.DataFrame(index=data.index)
        for carrier in self.carriers:
            out.loc[:, f"E_delivered_{carrier}_kWh"] = _series_from_aliases(
                data, self.carrier_aliases.get(carrier, []), default=0.0
            ).astype(float).fillna(0.0).clip(lower=0.0)
            # Historical generic export columns denote grid export. Destination-specific
            # columns make the non-EPB flow visible without changing existing callers.
            grid = _series_from_aliases(
                data,
                [f"E_exported_{carrier}_to_grid_kWh", *self.export_aliases.get(carrier, [])],
                default=0.0,
            ).astype(float).fillna(0.0).clip(lower=0.0)
            non_epb = _series_from_aliases(
                data,
                self.non_epb_export_aliases.get(carrier, [f"E_exported_{carrier}_to_non_epb_kWh"]),
                default=0.0,
            ).astype(float).fillna(0.0).clip(lower=0.0)
            out.loc[:, f"E_exported_{carrier}_grid_kWh"] = grid
            out.loc[:, f"E_exported_{carrier}_non_epb_kWh"] = non_epb
            out.loc[:, f"E_exported_{carrier}_kWh"] = grid + non_epb
        for name, aliases in self.onsite_aliases.items():
            series = _series_from_aliases(data, aliases, default=None)
            if series is not None:
                out.loc[:, f"E_onsite_{name}_kWh"] = series.astype(float).fillna(0.0).clip(lower=0.0)
        for carrier in self.carriers:
            for key in ("nonren", "ren", "co2"):
                for col in (
                    f"fA_{carrier}_{key}",
                    f"f_delivered_{carrier}_{key}", f"fD_{carrier}_{key}",
                    f"f_exported_{carrier}_{key}", f"fB_{carrier}_{key}",
                ):
                    if col in data.columns:
                        out.loc[:, col] = pd.to_numeric(data[col], errors="coerce")
        return out

    def _step_a_factors(self, carrier: str, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Step A weighting of the export (9.6.6.2): per-row columns, else the on-site source, else 0."""

        zeros = pd.Series(0.0, index=df.index)
        out = {"nonren": zeros, "ren": zeros, "co2": zeros}
        if carrier == "electricity":
            produced = {n: df[f"E_onsite_{n}_kWh"] for n in _ELECTRIC_ONSITE if f"E_onsite_{n}_kWh" in df}
            total = sum(produced.values()) if produced else None
            if total is not None:
                for key, fkey in (("nonren", "nonrenewable"), ("ren", "renewable"), ("co2", "co2")):
                    weighted = sum(s * self.onsite_factors[n][fkey] for n, s in produced.items())
                    avg = weighted / total.where(total > _KWH_EPS)
                    fallback = float(np.mean([self.onsite_factors[n][fkey] for n in produced]))  # Formula 8
                    out[key] = avg.fillna(fallback)
        for key in out:
            col = f"fA_{carrier}_{key}"
            if col in df:
                out[key] = df[col].fillna(out[key])
        return out

    def _simulate(self, prepared: pd.DataFrame) -> pd.DataFrame:
        out = prepared.copy()
        idx = out.index
        zero = lambda: pd.Series(0.0, index=idx)  # noqa: E731
        tot = {k: zero() for k in (
            "delivered", "exported", "pe_nren", "pe_ren", "credit_nren", "credit_ren",
            "co2_del", "co2_credit", "rer_ren_del", "rer_ren_credit",
            "rer_tot_del", "rer_tot_credit",
            "avl", "avl_nren", "avl_ren", "avl_co2",
        )}

        for carrier in self.carriers:
            delivered = out[f"E_delivered_{carrier}_kWh"]
            exported = out[f"E_exported_{carrier}_kWh"]
            f = self._time_factors_for(carrier, out, exported=False)
            fb = self._time_factors_for(carrier, out, exported=True)
            fa = self._step_a_factors(carrier, out)

            pe_nren, pe_ren, co2 = delivered * f["nonrenewable"], delivered * f["renewable"], delivered * f["co2"]
            # Formula (20): E_exp = E_A + k_exp * (E_B - E_A); E_B uses the avoided-resource factors.
            credit = {
                key: exported * (fa[key] + self.k_exp * (fbv - fa[key]))
                for key, fbv in (("nonren", fb["nonrenewable"]), ("ren", fb["renewable"]), ("co2", fb["co2"]))
            }

            # Formulae (21)-(22): exported energy not valorised in the performance, (1 - k_exp), and the
            # resources used to produce it (Step A), reported because they are made available outside
            avl_share = 1.0 - self.k_exp
            out.loc[:, f"E_exp_avl_{carrier}_kWh"] = exported * avl_share
            out.loc[:, f"PE_exp_avl_{carrier}_nonren_kWh"] = exported * fa["nonren"] * avl_share
            out.loc[:, f"PE_exp_avl_{carrier}_ren_kWh"] = exported * fa["ren"] * avl_share
            tot["avl"] += exported * avl_share
            tot["avl_nren"] += exported * fa["nonren"] * avl_share
            tot["avl_ren"] += exported * fa["ren"] * avl_share
            tot["avl_co2"] += exported * fa["co2"] * avl_share

            out.loc[:, f"PE_delivered_{carrier}_nonren_kWh"] = pe_nren
            out.loc[:, f"PE_delivered_{carrier}_ren_kWh"] = pe_ren
            out.loc[:, f"PE_export_credit_{carrier}_nonren_kWh"] = credit["nonren"]
            out.loc[:, f"PE_export_credit_{carrier}_ren_kWh"] = credit["ren"]

            tot["delivered"] += delivered
            tot["exported"] += exported
            tot["pe_nren"] += pe_nren
            tot["pe_ren"] += pe_ren
            tot["credit_nren"] += credit["nonren"]
            tot["credit_ren"] += credit["ren"]
            tot["co2_del"] += co2
            tot["co2_credit"] += credit["co2"]
            carrier_perimeter = self.perimeters.get(carrier, "distant")
            if carrier_perimeter in self.rer_ren:
                tot["rer_ren_del"] += pe_ren
            if carrier_perimeter in self.rer_tot:
                tot["rer_tot_del"] += pe_nren + pe_ren
            # The credit of the exported energy (Step A + k_exp * (B - A), resources avoided by the
            # grid) belongs to the distant perimeter: it enters the RER only if that perimeter is
            # selected. With the Table B.24 default the renewable numerator counts the on-site
            # production in full (ISO/TR 52000-2, 9.7.2 and Tables 12 and 16).
            if "distant" in self.rer_ren:
                tot["rer_ren_credit"] += credit["ren"]
            if "distant" in self.rer_tot:
                tot["rer_tot_credit"] += credit["nonren"] + credit["ren"]

        # On-site renewable production, delivered energy of perimeter 'onsite' (Table B.30)
        onsite_total = zero()
        for name in self.onsite_aliases:
            col = f"E_onsite_{name}_kWh"
            if col not in out:
                continue
            f = self.onsite_factors.get(name, ONSITE_FACTORS)
            pe_nren, pe_ren = out[col] * f["nonrenewable"], out[col] * f["renewable"]
            out.loc[:, f"PE_delivered_onsite_{name}_nonren_kWh"] = pe_nren
            out.loc[:, f"PE_delivered_onsite_{name}_ren_kWh"] = pe_ren
            tot["pe_nren"] += pe_nren
            tot["pe_ren"] += pe_ren
            tot["co2_del"] += out[col] * f["co2"]
            if "onsite" in self.rer_ren:
                tot["rer_ren_del"] += pe_ren
            if "onsite" in self.rer_tot:
                tot["rer_tot_del"] += pe_nren + pe_ren
            onsite_total += out[col]

        net_nren = tot["pe_nren"] - tot["credit_nren"]
        net_ren = tot["pe_ren"] - tot["credit_ren"]
        net_co2 = tot["co2_del"] - tot["co2_credit"]
        # RER numerator/denominator before any clipping (Formula 17), perimeters of Table B.24
        out.loc[:, "PE_RER_ren_kWh"] = tot["rer_ren_del"] - tot["rer_ren_credit"]
        out.loc[:, "PE_RER_tot_kWh"] = tot["rer_tot_del"] - tot["rer_tot_credit"]

        out.loc[:, "E_onsite_total_kWh"] = onsite_total
        out.loc[:, "E_delivered_total_kWh"] = tot["delivered"]
        out.loc[:, "E_exported_total_kWh"] = tot["exported"]
        out.loc[:, "PE_delivered_nonren_kWh"] = tot["pe_nren"]
        out.loc[:, "PE_delivered_ren_kWh"] = tot["pe_ren"]
        out.loc[:, "PE_export_credit_nonren_kWh"] = tot["credit_nren"]
        out.loc[:, "PE_export_credit_ren_kWh"] = tot["credit_ren"]
        out.loc[:, "PE_net_nonren_kWh"] = net_nren
        out.loc[:, "PE_net_ren_kWh"] = net_ren
        out.loc[:, "PE_net_total_kWh"] = net_nren + net_ren
        out.loc[:, "CO2_delivered_kg"] = tot["co2_del"]
        out.loc[:, "CO2_export_credit_kg"] = tot["co2_credit"]
        out.loc[:, "CO2_net_kg"] = net_co2
        out.loc[:, "E_exp_avl_total_kWh"] = tot["avl"]
        out.loc[:, "PE_exp_avl_nonren_kWh"] = tot["avl_nren"]
        out.loc[:, "PE_exp_avl_ren_kWh"] = tot["avl_ren"]
        out.loc[:, "PE_exp_avl_total_kWh"] = tot["avl_nren"] + tot["avl_ren"]
        out.loc[:, "CO2_exp_avl_kg"] = tot["avl_co2"]
        return out

    def _time_factors_for(
        self, carrier: str, df: pd.DataFrame, *, exported: bool
    ) -> dict[str, pd.Series]:
        """Return static factors overlaid by optional per-timestep factor columns."""

        prefix = "f_exported" if exported else "f_delivered"
        short_prefix = "fB" if exported else "fD"
        try:
            static = self._factors_for(carrier, exported=exported)
        except MissingWeightingFactors:
            static = None  # acceptable only if every factor is given for every interval
        # without a static export factor the export is weighted like the delivered energy
        base = self._time_factors_for(carrier, df, exported=False) if (static is None and exported) else None
        result: dict[str, pd.Series] = {}
        for key, suffix in (("nonrenewable", "nonren"), ("renewable", "ren"), ("co2", "co2")):
            value = (
                base[key].copy() if base is not None
                else pd.Series(np.nan if static is None else static[key], index=df.index, dtype=float)
            )
            for col in (f"{prefix}_{carrier}_{suffix}", f"{short_prefix}_{carrier}_{suffix}"):
                if col in df:
                    candidate = pd.to_numeric(df[col], errors="coerce")
                    value = candidate.where(candidate.notna(), value)
            if value.isna().any():
                self._factors_for(carrier, exported=exported)  # raises MissingWeightingFactors
            if (value < 0).any():
                raise ValueError(f"{prefix} factors for {carrier!r} must be >= 0.")
            result[key] = value
        return result

    def _factors_for(self, carrier: str, *, exported: bool) -> dict[str, float]:
        if carrier not in self.primary_factors and carrier not in self.default_factors:
            raise MissingWeightingFactors(
                f"No weighting factors for the energy carrier {carrier!r}: they are input data "
                "(EN ISO 52000-1, Table A.16). Give input_data['primary_energy_factors']"
                f"[{carrier!r}] = {{'nonrenewable': ..., 'renewable': ..., 'co2': ...}} or use a "
                "factor_set that defines it (iso_52000_1_b16 covers Table B.16)."
            )
        factors = self.primary_factors.get(carrier, self.default_factors.get(carrier))
        if exported:
            factors = self.export_factors.get(carrier, factors)
        nren = max(float(factors.get("nonrenewable", factors.get("nren", 0.0))), 0.0)
        ren = max(float(factors.get("renewable", factors.get("ren", 0.0))), 0.0)
        co2 = max(float(factors.get("co2", 0.0)), 0.0)
        return {"nonrenewable": nren, "renewable": ren, "co2": co2}

    def _summarize(self, results: pd.DataFrame) -> dict[str, float]:
        def s(col: str) -> float:
            return float(results[col].sum()) if col in results else 0.0

        out = {
            "E_delivered_total_kWh": s("E_delivered_total_kWh"),
            "E_exported_total_kWh": s("E_exported_total_kWh"),
            "E_onsite_total_kWh": s("E_onsite_total_kWh"),
            "PE_delivered_nonren_kWh": s("PE_delivered_nonren_kWh"),
            "PE_delivered_ren_kWh": s("PE_delivered_ren_kWh"),
            "PE_export_credit_nonren_kWh": s("PE_export_credit_nonren_kWh"),
            "PE_export_credit_ren_kWh": s("PE_export_credit_ren_kWh"),
            "PE_net_nonren_kWh": s("PE_net_nonren_kWh"),
            "PE_net_ren_kWh": s("PE_net_ren_kWh"),
            "PE_net_total_kWh": s("PE_net_total_kWh"),
            "CO2_delivered_kg": s("CO2_delivered_kg"),
            "CO2_export_credit_kg": s("CO2_export_credit_kg"),
            "CO2_net_kg": s("CO2_net_kg"),
            "E_exp_avl_total_kWh": s("E_exp_avl_total_kWh"),
            "PE_exp_avl_nonren_kWh": s("PE_exp_avl_nonren_kWh"),
            "PE_exp_avl_ren_kWh": s("PE_exp_avl_ren_kWh"),
            "PE_exp_avl_total_kWh": s("PE_exp_avl_total_kWh"),
            "CO2_exp_avl_kg": s("CO2_exp_avl_kg"),
            "k_exp": self.k_exp,
        }
        for carrier in self.carriers:
            delivered = s(f"E_delivered_{carrier}_kWh")
            exported = s(f"E_exported_{carrier}_kWh")
            out[f"E_delivered_{carrier}_kWh"] = delivered
            out[f"E_exported_{carrier}_kWh"] = exported
            out[f"E_exp_avl_{carrier}_kWh"] = s(f"E_exp_avl_{carrier}_kWh")
            out[f"E_exported_{carrier}_grid_kWh"] = s(f"E_exported_{carrier}_grid_kWh")
            out[f"E_exported_{carrier}_non_epb_kWh"] = s(f"E_exported_{carrier}_non_epb_kWh")
            out[f"PE_net_{carrier}_nonren_kWh"] = (
                s(f"PE_delivered_{carrier}_nonren_kWh")
                - s(f"PE_export_credit_{carrier}_nonren_kWh")
            )
        for name in self.onsite_aliases:
            if f"E_onsite_{name}_kWh" in results:
                out[f"E_onsite_{name}_kWh"] = s(f"E_onsite_{name}_kWh")
        if not self.allow_negative:  # floor of the total of the balance, not of its intervals
            for key in ("PE_net_nonren_kWh", "PE_net_ren_kWh", "CO2_net_kg"):
                out[key] = max(out[key], 0.0)
            out["PE_net_total_kWh"] = out["PE_net_nonren_kWh"] + out["PE_net_ren_kWh"]
        if out["E_delivered_total_kWh"] > _KWH_EPS:
            out["PEF_net_nonren"] = out["PE_net_nonren_kWh"] / out["E_delivered_total_kWh"]
            out["PEF_net_total"] = out["PE_net_total_kWh"] / out["E_delivered_total_kWh"]
        else:
            out["PEF_net_nonren"] = float("nan")
            out["PEF_net_total"] = float("nan")
        # Formula (17): RER = E_Pren,RER / E_Ptot, from the annual (unclipped) weighted energies
        rer_den = s("PE_RER_tot_kWh")
        out["PE_RER_ren_kWh"] = s("PE_RER_ren_kWh")
        out["PE_RER_tot_kWh"] = rer_den
        out["RER"] = out["PE_RER_ren_kWh"] / rer_den if rer_den > _KWH_EPS else float("nan")
        return out


def _normalise_factor_map(raw: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for carrier, values in dict(raw or {}).items():
        if isinstance(values, (int, float, np.number)):
            out[str(carrier)] = {"nonrenewable": float(values), "renewable": 0.0, "co2": 0.0}
        else:
            out[str(carrier)] = {
                "nonrenewable": float(
                    values.get("nonrenewable", values.get("nren", values.get("total", 0.0)))
                ),
                "renewable": float(values.get("renewable", values.get("ren", 0.0))),
                "co2": float(values.get("co2", 0.0)),
            }
    return out


def _series_from_aliases(
    df: pd.DataFrame, aliases: list[str], default: Any | None
) -> pd.Series | None:
    for name in aliases:
        if name in df.columns:
            return df[name]
    if default is None:
        return None
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)
