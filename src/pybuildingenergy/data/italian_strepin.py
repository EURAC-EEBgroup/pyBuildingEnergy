"""Italian STREPIN residential archetypes and renovation packages.

This module treats the STREPIN workbook as input data only. It builds
pyBuildingEnergy BUI dictionaries for the 18 residential archetypes and applies
the package levels by changing the simulated envelope/system metadata. Workbook
performance estimates are kept as comparison metadata; they are not used to
drive the engine-side demand calculation.

STREPIN ventilation presets are total effective outdoor-air exchange rates.
They are converted once to a custom ISO52016 ventilation heat-transfer
coefficient and are not added again as separate infiltration, design ventilation
or mechanical ventilation streams.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKBOOK_NAME = (
    "Italian_STREPIN_archetype_renovation_cost_models_visuals_reworked_fixed.xlsx"
)

ORIENTATION_AZIMUTH = {
    "NV": 0.0,
    "EV": 90.0,
    "SV": 180.0,
    "WV": 270.0,
}

DEFAULT_WINDOW_ORIENTATION_SPLIT = {
    "NV": 0.20,
    "EV": 0.20,
    "SV": 0.40,
    "WV": 0.20,
}

DEFAULT_SOLAR_SHADING_TRANSMITTANCE = {
    "sf": 1.00,
    "fixed": 0.20,
    "mobile": 0.20,
}

DEFAULT_SOLAR_SHADING_ORIENTATIONS = ("EV", "SV", "WV")
DEFAULT_MOBILE_SOLAR_SHADING_BASE_WIDTH_CM = 160.0
DEFAULT_FIXED_SOLAR_SHADING_COST_EUR_M2 = 170.0

DEFAULT_CLIMATE_LOCATIONS = {
    "B": {
        "name": "Palermo",
        "latitude": 38.1157,
        "longitude": 13.3615,
        "coldest_month": 1,
    },
    "E": {
        "name": "Milan",
        "latitude": 45.4642,
        "longitude": 9.1900,
        "coldest_month": 1,
    },
}

DEFAULT_INTERNAL_GAINS = [
    {
        "name": "occupants",
        "full_load": 4.2,
        "weekday": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.1,
            0.1,
            0.1,
            0.1,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.8,
            0.8,
            0.8,
            1.0,
            1.0,
        ],
        "weekend": [
            1.0,
            1.0,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            1.0,
            1.0,
        ],
    },
    {
        "name": "appliances",
        "full_load": 3.0,
        "weekday": [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.7,
            0.7,
            0.5,
            0.5,
            0.6,
            0.6,
            0.6,
            0.6,
            0.5,
            0.5,
            0.7,
            0.7,
            0.8,
            0.8,
            0.8,
            0.6,
            0.6,
        ],
        "weekend": [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.7,
            0.7,
            0.5,
            0.5,
            0.6,
            0.6,
            0.6,
            0.6,
            0.5,
            0.5,
            0.7,
            0.7,
            0.8,
            0.8,
            0.8,
            0.6,
            0.6,
        ],
    },
    {
        "name": "lighting",
        "full_load": 3.0,
        "weekday": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.15,
            0.15,
            0.15,
            0.15,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.15,
            0.15,
        ],
        "weekend": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.15,
            0.15,
            0.15,
            0.15,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.15,
            0.15,
        ],
    },
]

DEFAULT_HVAC_PROFILE = {
    "weekday": [0.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * 18 + [0.0],
    "weekend": [0.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * 18 + [0.0],
}

STREPIN_VENTILATION_PROFILE_ALWAYS_ON = {
    "weekday": [1.0] * 24,
    "weekend": [1.0] * 24,
}

STREPIN_VENTILATION_HEAT_CAPACITY_FACTOR_W_H_M3_K = 0.33

STREPIN_VENTILATION_PRESETS = {
    "normative_reference": {
        "description": "STREPIN reference total effective outdoor-air exchange.",
        "ach_h": 0.30,
    },
    "realistic_central": {
        "description": (
            "Central estimate of real total effective ACH for current Italian "
            "residential operation."
        ),
        "ach_h_by_period_and_climate_zone": {
            "existing": {"B": 0.45, "E": 0.55, "default": 0.50},
            "newer": {"B": 0.30, "E": 0.35, "default": 0.33},
        },
    },
    "leaky_high_behavioral": {
        "description": (
            "High-end sensitivity for leaky envelopes and/or high window airing."
        ),
        "ach_h_by_typology_period_and_climate_zone": {
            "single_family_existing": {"B": 0.65, "E": 0.75, "default": 0.70},
            "multifamily_existing": {"B": 0.55, "E": 0.65, "default": 0.60},
            "newer": {"B": 0.40, "E": 0.45, "default": 0.43},
        },
    },
}


def classify_strepin_period(
    archetype_code: str,
    period_code: str | None = None,
) -> str:
    """Return ``existing`` or ``newer`` for STREPIN ventilation presets."""

    tokens = [
        token.strip().upper()
        for token in [period_code, *str(archetype_code).replace("-", "_").split("_")]
        if token is not None and str(token).strip()
    ]
    if any(token in {"N0", "NEW", "NEWER", "REFERENCE", "REF"} for token in tokens):
        return "newer"
    return "existing"


def classify_strepin_typology(
    archetype_code: str,
    type_code: str | None = None,
) -> str:
    """Return ``single_family`` or ``multifamily`` for STREPIN ventilation presets."""

    tokens = [
        token.strip().upper()
        for token in [type_code, *str(archetype_code).replace("-", "_").split("_")]
        if token is not None and str(token).strip()
    ]
    if any(token in {"RMF", "SINGLE_FAMILY", "SF", "SFD"} for token in tokens):
        return "single_family"
    return "multifamily"


def resolve_strepin_effective_ach(
    archetype_code: str,
    climate_zone: str,
    ventilation_preset: str = "normative_reference",
    air_change_rate_h: float | None = None,
    *,
    period_code: str | None = None,
    type_code: str | None = None,
) -> float:
    """Resolve the total effective ACH used for simplified STREPIN cases.

    The returned value represents the complete outdoor-air exchange assumption
    for the case. It is converted to a custom ventilation Hve and is not added
    again as a separate infiltration, design ventilation or mechanical stream.
    """

    if air_change_rate_h is not None and not _is_missing(air_change_rate_h):
        ach = float(air_change_rate_h)
        if ach < 0.0:
            raise ValueError("air_change_rate_h must be non-negative.")
        return ach

    preset_id = _normalize_ventilation_preset(ventilation_preset)
    preset = STREPIN_VENTILATION_PRESETS[preset_id]
    zone = str(climate_zone).strip().upper()

    if "ach_h" in preset:
        return float(preset["ach_h"])

    period = classify_strepin_period(archetype_code, period_code)
    if preset_id == "realistic_central":
        values = preset["ach_h_by_period_and_climate_zone"][period]
        return float(values.get(zone, values["default"]))

    typology = classify_strepin_typology(archetype_code, type_code)
    group = "newer" if period == "newer" else f"{typology}_existing"
    values = preset["ach_h_by_typology_period_and_climate_zone"][group]
    return float(values.get(zone, values["default"]))


def _normalize_ventilation_preset(ventilation_preset: str) -> str:
    preset_id = str(ventilation_preset or "normative_reference").strip().lower()
    preset_id = preset_id.replace("-", "_")
    if preset_id not in STREPIN_VENTILATION_PRESETS:
        available = ", ".join(sorted(STREPIN_VENTILATION_PRESETS))
        raise ValueError(
            f"Unknown STREPIN ventilation preset {ventilation_preset!r}. "
            f"Available presets: {available}."
        )
    return preset_id


@dataclass(frozen=True)
class StrepinCase:
    """A pyBuildingEnergy-ready STREPIN archetype/package case."""

    archetype_id: str
    package_id: str
    bui: dict[str, Any]
    archetype: dict[str, Any]
    package: dict[str, Any]
    measures: list[dict[str, Any]]
    workbook_reference: dict[str, Any] | None = None


@dataclass
class ItalianStrepinTables:
    """Parsed STREPIN workbook tables plus convenience constructors."""

    workbook_path: Path
    archetypes: pd.DataFrame
    packages: pd.DataFrame
    measure_levels: pd.DataFrame
    results: pd.DataFrame
    assumptions: pd.DataFrame

    @property
    def archetype_ids(self) -> list[str]:
        return self.archetypes["Archetype_ID"].astype(str).tolist()

    @property
    def package_ids(self) -> list[str]:
        return self.packages["Package_ID"].astype(str).tolist()

    def assumption_value(self, key: str, default: Any = None) -> Any:
        values = self.assumptions.loc[self.assumptions["Input"] == key, "Value"]
        if values.empty:
            return default
        value = values.iloc[0]
        if _is_missing(value):
            return default
        return value

    def archetype_row(self, archetype_id: str) -> pd.Series:
        rows = self.archetypes.loc[
            self.archetypes["Archetype_ID"].astype(str) == str(archetype_id)
        ]
        if rows.empty:
            raise KeyError(
                f"Unknown STREPIN archetype {archetype_id!r}. "
                f"Available: {', '.join(self.archetype_ids)}"
            )
        return rows.iloc[0]

    def package_row(self, package_id: str) -> pd.Series:
        rows = self.packages.loc[
            self.packages["Package_ID"].astype(str) == str(package_id)
        ]
        if rows.empty:
            raise KeyError(
                f"Unknown STREPIN package {package_id!r}. "
                f"Available: {', '.join(self.package_ids)}"
            )
        return rows.iloc[0]

    def measure_row(
        self,
        archetype_id: str,
        measure_code: str,
        level: int,
    ) -> pd.Series:
        rows = self.measure_levels.loc[
            (self.measure_levels["Archetype_ID"].astype(str) == str(archetype_id))
            & (self.measure_levels["Measure_Code"].astype(str) == str(measure_code))
            & (self.measure_levels["Level"].astype(int) == int(level))
        ]
        if rows.empty:
            raise KeyError(
                f"Missing measure level for {archetype_id!r}, "
                f"{measure_code!r}, level {level}."
            )
        return rows.iloc[0]

    def package_measure_rows(
        self,
        archetype_id: str,
        package_id: str,
    ) -> list[pd.Series]:
        package = self.package_row(package_id)
        level_by_measure = {
            "wall": int(package["Wall_Level"]),
            "roof_floor": int(package["RoofFloor_Level"]),
            "windows": int(package["Windows_Level"]),
            "heating": int(package["Heating_Level"]),
            "pv": int(package["PV_Level"]),
        }
        return [
            self.measure_row(archetype_id, measure_code, level)
            for measure_code, level in level_by_measure.items()
        ]

    def result_row(self, archetype_id: str, package_id: str) -> dict[str, Any] | None:
        rows = self.results.loc[
            (self.results["Archetype_ID"].astype(str) == str(archetype_id))
            & (self.results["Package_ID"].astype(str) == str(package_id))
        ]
        if rows.empty:
            return None
        return _row_to_dict(rows.iloc[0])

    def make_base_bui(
        self,
        archetype_id: str,
        *,
        climate_locations: dict[str, dict[str, Any]] | None = None,
        window_orientation_split: dict[str, float] | None = None,
        floor_height_m: float = 3.0,
        ventilation_preset: str = "normative_reference",
        air_change_rate_h: float | None = None,
        ideal_hvac_capacity: bool = True,
    ) -> dict[str, Any]:
        return _archetype_to_bui(
            self.archetype_row(archetype_id),
            climate_locations=climate_locations,
            window_orientation_split=window_orientation_split,
            floor_height_m=floor_height_m,
            ventilation_preset=ventilation_preset,
            air_change_rate_h=air_change_rate_h,
            ideal_hvac_capacity=ideal_hvac_capacity,
        )

    def make_case(
        self,
        archetype_id: str,
        package_id: str = "P00",
        *,
        climate_locations: dict[str, dict[str, Any]] | None = None,
        window_orientation_split: dict[str, float] | None = None,
        floor_height_m: float = 3.0,
        ventilation_preset: str = "normative_reference",
        air_change_rate_h: float | None = None,
        ideal_hvac_capacity: bool = True,
    ) -> StrepinCase:
        bui = self.make_base_bui(
            archetype_id,
            climate_locations=climate_locations,
            window_orientation_split=window_orientation_split,
            floor_height_m=floor_height_m,
            ventilation_preset=ventilation_preset,
            air_change_rate_h=air_change_rate_h,
            ideal_hvac_capacity=ideal_hvac_capacity,
        )
        package = self.package_row(package_id)
        measures = self.package_measure_rows(archetype_id, package_id)
        bui = apply_measure_rows_to_bui(bui, measures)
        return StrepinCase(
            archetype_id=str(archetype_id),
            package_id=str(package_id),
            bui=bui,
            archetype=_row_to_dict(self.archetype_row(archetype_id)),
            package=_row_to_dict(package),
            measures=[_row_to_dict(row) for row in measures],
            workbook_reference=self.result_row(archetype_id, package_id),
        )

    def make_measure_case(
        self,
        archetype_id: str,
        measure_levels: dict[str, int],
        *,
        package_id: str = "CUSTOM",
        climate_locations: dict[str, dict[str, Any]] | None = None,
        window_orientation_split: dict[str, float] | None = None,
        floor_height_m: float = 3.0,
        ventilation_preset: str = "normative_reference",
        air_change_rate_h: float | None = None,
        ideal_hvac_capacity: bool = True,
    ) -> StrepinCase:
        """Build a case from arbitrary STREPIN workbook measure levels.

        Missing measures default to baseline level 1. Supported workbook measure
        codes are ``wall``, ``roof_floor``, ``windows``, ``heating`` and ``pv``.
        Engine-side overlays such as BACS or HRV are applied separately.
        """

        levels = {
            "wall": 1,
            "roof_floor": 1,
            "windows": 1,
            "heating": 1,
            "pv": 1,
            **{str(k): int(v) for k, v in measure_levels.items()},
        }
        unknown = sorted(set(levels) - {"wall", "roof_floor", "windows", "heating", "pv"})
        if unknown:
            raise ValueError(f"Unknown STREPIN measure code(s): {', '.join(unknown)}")

        bui = self.make_base_bui(
            archetype_id,
            climate_locations=climate_locations,
            window_orientation_split=window_orientation_split,
            floor_height_m=floor_height_m,
            ventilation_preset=ventilation_preset,
            air_change_rate_h=air_change_rate_h,
            ideal_hvac_capacity=ideal_hvac_capacity,
        )
        measures = [
            self.measure_row(archetype_id, measure_code, level)
            for measure_code, level in levels.items()
        ]
        bui = apply_measure_rows_to_bui(bui, measures)
        package = {
            "Package_ID": package_id,
            "Package_Name": "Custom measure combination",
            "Wall_Level": levels["wall"],
            "RoofFloor_Level": levels["roof_floor"],
            "Windows_Level": levels["windows"],
            "Heating_Level": levels["heating"],
            "PV_Level": levels["pv"],
            "Active_Measures": ", ".join(
                f"{code} L{level}" for code, level in levels.items() if level > 1
            )
            or "none",
            "Package_Type": "Custom measure combination",
        }
        return StrepinCase(
            archetype_id=str(archetype_id),
            package_id=str(package_id),
            bui=bui,
            archetype=_row_to_dict(self.archetype_row(archetype_id)),
            package=package,
            measures=[_row_to_dict(row) for row in measures],
            workbook_reference=None,
        )


def find_default_workbook() -> Path:
    """Return the repository-local STREPIN workbook path when available."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "examples" / "strepin_archetypes" / WORKBOOK_NAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the Italian STREPIN workbook. Pass workbook_path to "
        "load_italian_strepin_tables()."
    )


def load_italian_strepin_tables(
    workbook_path: str | Path | None = None,
) -> ItalianStrepinTables:
    """Load the workbook tables used to define Italian STREPIN cases."""

    path = Path(workbook_path) if workbook_path is not None else find_default_workbook()
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    archetypes = _read_table(path, "Archetype_DB", "Archetype_ID")
    packages = _read_table(path, "Packages", "Package_ID")
    measure_levels = _read_table(path, "Measure_Levels", "Archetype_ID")
    results = _read_table(path, "Results_All", "Archetype_ID")
    assumptions = _read_table(path, "Assumptions", "Input")

    packages = packages.copy()
    measure_levels = measure_levels.copy()
    for col in [
        "Wall_Level",
        "RoofFloor_Level",
        "Windows_Level",
        "Heating_Level",
        "PV_Level",
    ]:
        packages.loc[:, col] = packages[col].astype(int)
    measure_levels.loc[:, "Level"] = measure_levels["Level"].astype(int)

    return ItalianStrepinTables(
        workbook_path=path,
        archetypes=archetypes,
        packages=packages,
        measure_levels=measure_levels,
        results=results,
        assumptions=assumptions,
    )


def apply_measure_rows_to_bui(
    bui: dict[str, Any],
    measure_rows: list[pd.Series],
) -> dict[str, Any]:
    """Return a copied BUI with STREPIN measure rows applied."""

    updated = copy.deepcopy(bui)
    metadata = updated.setdefault("strepin", {})
    applied: list[dict[str, Any]] = []

    for row in measure_rows:
        measure_code = str(row["Measure_Code"])
        level = int(row["Level"])
        target_u1 = _to_float(row.get("Target_U1"), 0.0)
        target_u2 = _to_float(row.get("Target_U2"), 0.0)
        applied_row = _row_to_dict(row)

        if measure_code == "wall" and target_u1 > 0.0:
            _set_surface_u_value(updated, target_u1, kind="wall")
        elif measure_code == "roof_floor":
            if target_u1 > 0.0:
                _set_surface_u_value(updated, target_u1, kind="roof")
            if target_u2 > 0.0:
                _set_surface_u_value(updated, target_u2, kind="ground")
        elif measure_code == "windows" and target_u1 > 0.0:
            _set_surface_u_value(updated, target_u1, kind="window")
            _set_window_g_value(updated, target_u1)
        elif measure_code == "heating":
            metadata["heating_system"] = _heating_measure_metadata(row)
        elif measure_code == "pv":
            metadata["pv_system"] = {
                "level": level,
                "level_label": str(row.get("Level_Label", "")),
                "pv_kWp": _to_float(row.get("PV_kWp"), 0.0),
                "investment_EUR": _to_float(row.get("Investment_EUR"), 0.0),
            }

        applied.append(applied_row)

    metadata["applied_measures"] = applied
    return updated


def apply_extra_measure_specs_to_bui(
    bui: dict[str, Any],
    measure_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply engine-side renovation overlays not present in the workbook.

    Specs are dictionaries with ``measure_code`` and optional ``parameters``.
    The function keeps metadata under ``bui["strepin"]`` so the simulation
    runner can activate the corresponding European-standard modules.
    """

    updated = copy.deepcopy(bui)
    metadata = updated.setdefault("strepin", {})
    extra: list[dict[str, Any]] = list(metadata.get("extra_measures", []))

    for spec in measure_specs:
        measure_code = str(spec.get("measure_code", spec.get("code", ""))).strip().lower()
        params = dict(spec.get("parameters", spec.get("params", {})) or {})
        if not measure_code:
            continue

        if measure_code == "lighting":
            lighting = {
                "installed_power_density_W_m2": _to_float(
                    params.get("installed_power_density_W_m2", params.get("lpd_W_m2")),
                    4.0,
                ),
                "daylight_dependency_factor": _to_float(
                    params.get("daylight_dependency_factor", 0.85),
                    0.85,
                ),
                "occupancy_dependency_factor": _to_float(
                    params.get("occupancy_dependency_factor", 0.90),
                    0.90,
                ),
                "daylight_control_fraction": _to_float(
                    params.get("daylight_control_fraction", 0.30),
                    0.30,
                ),
                "occupancy_control_fraction": _to_float(
                    params.get("occupancy_control_fraction", 0.20),
                    0.20,
                ),
                "investment_EUR": _to_float(params.get("investment_EUR"), 20.0)
                * _building_area(updated),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 0.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 15),
            }
            _set_internal_gain_full_load(
                updated,
                "lighting",
                lighting["installed_power_density_W_m2"],
            )
            metadata["lighting_system"] = lighting
            extra.append({"measure_code": "lighting", **lighting})

        elif measure_code == "ventilation":
            volume_m3 = _building_volume(updated)
            ach = _to_float(params.get("air_change_rate_h"), metadata.get("air_change_rate_h", 0.30))
            heat_recovery = max(
                0.0,
                min(1.0, _to_float(params.get("heat_recovery_efficiency"), 0.75)),
            )
            mechanical_fraction = max(
                0.0,
                min(1.0, _to_float(params.get("mechanical_fraction"), 1.0)),
            )
            flow_m3_h = _to_float(params.get("nominal_flow_m3_h"), volume_m3 * ach)
            hve_base = STREPIN_VENTILATION_HEAT_CAPACITY_FACTOR_W_H_M3_K * flow_m3_h
            hve_effective = hve_base * (1.0 - heat_recovery * mechanical_fraction)
            ventilation = {
                "air_change_rate_h": ach,
                "nominal_flow_m3_h": flow_m3_h,
                "heat_recovery_efficiency": heat_recovery,
                "mechanical_fraction": mechanical_fraction,
                "specific_fan_power_W_s_m3": _to_float(
                    params.get("specific_fan_power_W_s_m3"),
                    900.0,
                ),
                "investment_EUR": _to_float(params.get("investment_EUR"), 35.0)
                * _building_area(updated),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 150.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 20),
            }
            bp_vent = updated.setdefault("building_parameters", {}).setdefault("ventilation", {})
            bp_vent["ventilation_type"] = "custom"
            bp_vent["custom_heat_transfer_coefficient_ventilation"] = hve_effective
            bp_vent["units"] = "W/K"
            metadata["ventilation_system"] = ventilation
            extra.append({"measure_code": "ventilation", **ventilation})

        elif measure_code in {"solar_shading", "window_shading", "eem6"}:
            raw_level = str(
                params.get("level", params.get("type", params.get("level_label", "fixed")))
            ).strip().lower()
            level_aliases = {
                "sf": "sf",
                "state-of-fact": "sf",
                "state_of_fact": "sf",
                "none": "sf",
                "no": "sf",
                "fixed": "fixed",
                "fissa": "fixed",
                "mobile": "mobile",
                "movable": "mobile",
            }
            level = level_aliases.get(raw_level, raw_level)
            if level not in DEFAULT_SOLAR_SHADING_TRANSMITTANCE:
                raise ValueError(
                    "Solar shading level must be one of 'sf', 'fixed'/'fissa' or 'mobile'."
                )

            factor = _to_float(
                params.get(
                    "solar_transmission_factor",
                    params.get(
                        "solar_shading_transmittance",
                        params.get("transmittance", params.get("factor")),
                    ),
                ),
                DEFAULT_SOLAR_SHADING_TRANSMITTANCE[level],
            )
            factor = max(0.0, min(1.0, factor))

            activation_raw = params.get("activation_irradiance_W_m2")
            if activation_raw is None:
                activation_raw = params.get("solar_shading_activation_irradiance_W_m2")
            activation = (
                None
                if activation_raw is None
                else max(0.0, _to_float(activation_raw, 0.0))
            )

            orientations = _solar_shading_orientations(
                params.get("orientations", params.get("apply_to_orientations"))
            )
            affected_area_m2 = _set_window_solar_shading(
                updated,
                level=level,
                solar_transmission_factor=factor,
                activation_irradiance_W_m2=activation,
                orientations=orientations,
            )
            cost_eur_m2 = _to_float(
                params.get("cost_EUR_m2", params.get("cost_eur_m2")),
                _solar_shading_default_cost_eur_m2(level, params),
            )
            investment = _to_float(
                params.get("investment_EUR"),
                0.0 if level == "sf" else cost_eur_m2 * affected_area_m2,
            )
            solar_shading = {
                "level": level,
                "level_label": {
                    "sf": "SF",
                    "fixed": "fissa",
                    "mobile": "mobile",
                }[level],
                "solar_transmission_factor": factor,
                "activation_irradiance_W_m2": activation,
                "applied_orientations": sorted(orientations),
                "affected_window_area_m2": affected_area_m2,
                "cost_EUR_m2": cost_eur_m2,
                "investment_EUR": investment,
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 0.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 20),
            }
            metadata["solar_shading"] = solar_shading
            extra.append({"measure_code": "solar_shading", **solar_shading})

        elif measure_code == "bacs":
            bacs = {
                "bacs_class": str(params.get("bacs_class", params.get("class", "B"))).upper(),
                "investment_EUR": _to_float(params.get("investment_EUR"), 12.0)
                * _building_area(updated),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 50.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 15),
            }
            metadata["bacs"] = bacs
            extra.append({"measure_code": "bacs", **bacs})

        elif measure_code in {"biomass", "biomass_boiler"}:
            heating_system = {
                "level": 99,
                "level_label": "Biomass boiler",
                "system_type": "biomass_pellet_boiler",
                "carrier": "biomass",
                "seasonal_efficiency_or_scop": _to_float(params.get("efficiency"), 0.84),
                "investment_EUR": _to_float(params.get("investment_EUR"), 550.0)
                * max(_to_float(_building(updated).get("design_heating_capacity_kW"), 1.0), 1.0),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 300.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 18),
            }
            metadata["heating_system"] = heating_system
            extra.append({"measure_code": "biomass", **heating_system})

        elif measure_code == "district_heat":
            heating_system = {
                "level": 98,
                "level_label": "District heating",
                "system_type": "district_heat_substation",
                "carrier": "district_heat",
                "seasonal_efficiency_or_scop": _to_float(params.get("efficiency"), 0.97),
                "investment_EUR": _to_float(params.get("investment_EUR"), 180.0)
                * max(_to_float(_building(updated).get("design_heating_capacity_kW"), 1.0), 1.0),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 120.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 25),
            }
            metadata["heating_system"] = heating_system
            extra.append({"measure_code": "district_heat", **heating_system})

        elif measure_code == "chp":
            heating_system = {
                "level": 97,
                "level_label": "Cogeneration",
                "system_type": "cogeneration",
                "carrier": "gas",
                "seasonal_efficiency_or_scop": _to_float(params.get("thermal_efficiency"), 0.55),
                "electrical_efficiency": _to_float(params.get("electrical_efficiency"), 0.30),
                "investment_EUR": _to_float(params.get("investment_EUR"), 1200.0)
                * max(_to_float(_building(updated).get("design_heating_capacity_kW"), 1.0), 1.0),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 450.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 15),
            }
            metadata["heating_system"] = heating_system
            extra.append({"measure_code": "chp", **heating_system})

        elif measure_code == "solar_thermal":
            roof_area = sum(
                _to_float(surface.get("area"))
                for surface in updated.get("building_surface", [])
                if _surface_matches_kind(surface, "roof")
            )
            solar = {
                "enabled": True,
                "area_m2": _to_float(params.get("area_m2"), max(2.0, 0.12 * roof_area)),
                "annual_yield_kWh_m2": _to_float(params.get("annual_yield_kWh_m2"), 450.0),
                "storage_loss_fraction": _to_float(params.get("storage_loss_fraction"), 0.12),
                "shading_loss_fraction": _to_float(params.get("shading_loss_fraction"), 0.0),
                "priority": str(params.get("priority", "dhw_first")),
                "investment_EUR": _to_float(params.get("investment_EUR"), 850.0)
                * _to_float(params.get("area_m2"), max(2.0, 0.12 * roof_area)),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 60.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 25),
            }
            metadata["solar_thermal"] = solar
            extra.append({"measure_code": "solar_thermal", **solar})

        elif measure_code == "battery":
            battery = {
                "capacity_kWh": _to_float(params.get("capacity_kWh"), 5.0),
                "roundtrip_efficiency": _to_float(params.get("roundtrip_efficiency"), 0.90),
                "investment_EUR": _to_float(params.get("investment_EUR"), 550.0)
                * _to_float(params.get("capacity_kWh"), 5.0),
                "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 25.0),
                "lifetime_years": _to_int(params.get("lifetime_years"), 12),
            }
            metadata["battery"] = battery
            extra.append({"measure_code": "battery", **battery})

        elif measure_code == "pv_shading":
            pv = metadata.setdefault("pv_system", {})
            shading = max(0.0, min(1.0, _to_float(params.get("shading_loss_fraction"), 0.10)))
            pv["shading_loss_fraction"] = shading
            extra.append(
                {
                    "measure_code": "pv_shading",
                    "shading_loss_fraction": shading,
                    "investment_EUR": _to_float(params.get("investment_EUR"), 0.0),
                    "annual_maintenance_EUR": _to_float(params.get("annual_maintenance_EUR"), 0.0),
                    "lifetime_years": _to_int(params.get("lifetime_years"), 25),
                }
            )
        else:
            raise ValueError(f"Unknown extra STREPIN measure {measure_code!r}.")

    metadata["extra_measures"] = extra
    return updated


def summarize_engine_performance(
    case: StrepinCase,
    annual_results: pd.DataFrame | dict[str, Any],
    tables: ItalianStrepinTables,
    *,
    include_cooling: bool = True,
    cooling_eer: float = 3.0,
) -> dict[str, Any]:
    """Combine engine demand results with STREPIN package system metadata.

    ISO52016 provides heating/cooling useful needs. This helper derives delivered
    and non-renewable primary energy with the workbook carrier assumptions so the
    engine-side demand can be compared against the workbook estimates. Replace
    this with full EN 15316/EN 16798 system chains when those measures are
    modelled in detail.
    """

    annual = _annual_dict(annual_results)
    archetype = case.archetype
    area_m2 = _to_float(archetype["Af_m2"])
    climate_zone = str(archetype["Climate_Zone"])
    type_code = str(archetype["Type_Code"])

    heating_need_kWh = _to_float(
        annual.get("Q_H_annual_kWh"),
        _to_float(annual.get("Q_H_annual"), 0.0) / 1000.0,
    )
    cooling_need_kWh = _to_float(
        annual.get("Q_C_annual_kWh"),
        _to_float(annual.get("Q_C_annual"), 0.0) / 1000.0,
    )
    dhw_useful_kWh = _to_float(archetype.get("Baseline_DHW_Useful")) * area_m2
    aux_electricity_kWh = _to_float(archetype.get("Baseline_Aux_Elec")) * area_m2

    metadata = case.bui.get("strepin", {})
    heating_system = metadata.get("heating_system", {})
    carrier = heating_system.get("carrier", "gas")
    efficiency = _to_float(
        heating_system.get("seasonal_efficiency_or_scop"),
        _to_float(archetype.get("Baseline_Boiler_Eff"), 0.85),
    )
    useful_thermal_kWh = heating_need_kWh + dhw_useful_kWh

    gas_final_kWh = 0.0
    heat_pump_electricity_kWh = 0.0
    if carrier == "electricity":
        heat_pump_electricity_kWh = useful_thermal_kWh / max(efficiency, 1e-9)
    else:
        gas_final_kWh = useful_thermal_kWh / max(efficiency, 1e-9)

    cooling_electricity_kWh = (
        cooling_need_kWh / max(float(cooling_eer), 1e-9) if include_cooling else 0.0
    )
    gross_electricity_kWh = (
        aux_electricity_kWh + heat_pump_electricity_kWh + cooling_electricity_kWh
    )

    pv = case.bui.get("strepin", {}).get("pv_system", {})
    pv_kWp = _to_float(pv.get("pv_kWp"), 0.0)
    pv_yield_key = f"PV_Yield_Zone_{climate_zone}"
    pv_yield = _to_float(tables.assumption_value(pv_yield_key, 0.0), 0.0)
    pv_generation_kWh = pv_kWp * pv_yield
    self_consumption = _to_float(
        tables.assumption_value("PV_SelfConsumption_Factor", 0.65), 0.65
    )
    pv_self_consumed_kWh = min(
        gross_electricity_kWh,
        pv_generation_kWh * max(0.0, min(1.0, self_consumption)),
    )
    grid_electricity_kWh = max(gross_electricity_kWh - pv_self_consumed_kWh, 0.0)
    pv_export_kWh = max(pv_generation_kWh - pv_self_consumed_kWh, 0.0)

    gas_pef = _to_float(tables.assumption_value("Gas_Primary_Energy_Factor", 1.05))
    electricity_pef = _to_float(
        tables.assumption_value("Electricity_Primary_Energy_Factor", 2.17)
    )
    final_energy_kWh = gas_final_kWh + grid_electricity_kWh
    primary_nonren_kWh = gas_final_kWh * gas_pef + grid_electricity_kWh * electricity_pef

    reference = case.workbook_reference or {}
    out = {
        "Archetype_ID": case.archetype_id,
        "Package_ID": case.package_id,
        "Building_Type": archetype.get("Building_Type"),
        "Type_Code": type_code,
        "Climate_Zone": climate_zone,
        "Period_Code": archetype.get("Period_Code"),
        "Af_m2": area_m2,
        "Ventilation_preset": metadata.get("ventilation_preset"),
        "Effective_ACH_h": metadata.get("effective_air_change_rate_h"),
        "Ventilation_Hve_W_K": metadata.get("ventilation_hve_W_K"),
        "Q_H_engine_kWh_a": heating_need_kWh,
        "Q_C_engine_kWh_a": cooling_need_kWh,
        "Q_H_engine_kWh_m2a": heating_need_kWh / area_m2 if area_m2 > 0 else 0.0,
        "Q_C_engine_kWh_m2a": cooling_need_kWh / area_m2 if area_m2 > 0 else 0.0,
        "DHW_useful_kWh_a": dhw_useful_kWh,
        "Gas_final_kWh_a": gas_final_kWh,
        "HeatPump_electricity_kWh_a": heat_pump_electricity_kWh,
        "Cooling_electricity_kWh_a": cooling_electricity_kWh,
        "Aux_electricity_kWh_a": aux_electricity_kWh,
        "Gross_electricity_kWh_a": gross_electricity_kWh,
        "PV_kWp": pv_kWp,
        "PV_generation_kWh_a": pv_generation_kWh,
        "PV_self_consumed_kWh_a": pv_self_consumed_kWh,
        "PV_export_kWh_a": pv_export_kWh,
        "Grid_electricity_kWh_a": grid_electricity_kWh,
        "Final_energy_kWh_a": final_energy_kWh,
        "Final_energy_kWh_m2a": final_energy_kWh / area_m2 if area_m2 > 0 else 0.0,
        "Primary_nonren_kWh_a": primary_nonren_kWh,
        "Primary_nonren_kWh_m2a": (
            primary_nonren_kWh / area_m2 if area_m2 > 0 else 0.0
        ),
        "Heating_system_carrier": carrier,
        "Heating_efficiency_or_SCOP": efficiency,
        "Reference_workbook_final_kWh_m2a": reference.get(
            "Final_Energy_Intensity_kWh_m2a"
        ),
        "Reference_workbook_primary_kWh_m2a": reference.get(
            "Primary_Energy_Intensity_kWhEPnren_m2a"
        ),
        "Reference_workbook_package_name": reference.get("Package_Name"),
    }
    return out


def _read_table(path: Path, sheet_name: str, required_column: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=2)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    if required_column not in df.columns:
        raise ValueError(f"{sheet_name} is missing required column {required_column}.")
    df = df.loc[~df[required_column].isna()].copy()
    return df.reset_index(drop=True)


def _archetype_to_bui(
    row: pd.Series,
    *,
    climate_locations: dict[str, dict[str, Any]] | None,
    window_orientation_split: dict[str, float] | None,
    floor_height_m: float,
    ventilation_preset: str,
    air_change_rate_h: float | None,
    ideal_hvac_capacity: bool,
) -> dict[str, Any]:
    archetype_id = str(row["Archetype_ID"])
    type_code = str(row["Type_Code"])
    climate_zone = str(row["Climate_Zone"])
    period_code = str(row["Period_Code"])
    locations = climate_locations or DEFAULT_CLIMATE_LOCATIONS
    location = locations.get(climate_zone, DEFAULT_CLIMATE_LOCATIONS["E"])

    floors = max(1, _to_int(row["Floors"], 1))
    area_m2 = _to_float(row["Af_m2"])
    footprint_m2 = _to_float(row["Ground_m2"], area_m2 / floors)
    total_height_m = floor_height_m * floors
    volume_m3 = area_m2 * floor_height_m
    ventilation_preset_id = _normalize_ventilation_preset(ventilation_preset)
    effective_ach_h = resolve_strepin_effective_ach(
        archetype_id,
        climate_zone,
        ventilation_preset_id,
        air_change_rate_h,
        period_code=period_code,
        type_code=type_code,
    )
    ventilation_hve_w_k = (
        STREPIN_VENTILATION_HEAT_CAPACITY_FACTOR_W_H_M3_K
        * effective_ach_h
        * volume_m3
    )
    ach_override_h = (
        float(air_change_rate_h)
        if air_change_rate_h is not None and not _is_missing(air_change_rate_h)
        else None
    )

    heating_capacity_w = (
        10_000_000.0
        if ideal_hvac_capacity
        else max(1.0, _to_float(row["Heating_Capacity_kW"]) * 1000.0)
    )

    surfaces: list[dict[str, Any]] = []
    wall_area_total = _to_float(row["Opaque_Wall_m2"])
    window_area_total = _to_float(row["Window_m2"])
    wall_area_by_orientation = {
        orientation: wall_area_total / 4.0 for orientation in ORIENTATION_AZIMUTH
    }
    window_split = _normalize_split(
        window_orientation_split or DEFAULT_WINDOW_ORIENTATION_SPLIT
    )

    for orientation, azimuth in ORIENTATION_AZIMUTH.items():
        wall_area = wall_area_by_orientation[orientation]
        surfaces.append(
            _surface(
                name=f"{orientation} opaque wall",
                surface_type="opaque",
                area=wall_area,
                u_value=_to_float(row["U_wall"]),
                orientation_code=orientation,
                azimuth=azimuth,
                tilt=90.0,
                boundary="OUTDOORS",
                solar_absorptance=0.60,
                sky_view_factor=0.50,
                thermal_capacity=1_416_240.0,
                height=total_height_m,
                width=wall_area / max(total_height_m, 1e-9),
            )
        )

        window_area = window_area_total * window_split.get(orientation, 0.0)
        if window_area > 0.0:
            window_height = 1.50
            surfaces.append(
                _surface(
                    name=f"{orientation} windows",
                    surface_type="transparent",
                    area=window_area,
                    u_value=_to_float(row["U_window"]),
                    orientation_code=orientation,
                    azimuth=azimuth,
                    tilt=90.0,
                    boundary="OUTDOORS",
                    solar_absorptance=0.50,
                    sky_view_factor=0.50,
                    thermal_capacity=0.0,
                    height=window_height,
                    width=window_area / window_height,
                    g_value=_default_window_g_value(_to_float(row["U_window"])),
                    parapet=0.90,
                    shading=False,
                    shading_type="none",
                    width_or_distance_of_shading_elements=0.0,
                    overhang_properties={"width_of_horizontal_overhangs": 0.0},
                )
            )

    roof_area = _to_float(row["Roof_m2"], footprint_m2)
    ground_area = _to_float(row["Ground_m2"], footprint_m2)
    surfaces.append(
        _surface(
            name="Roof",
            surface_type="opaque",
            area=roof_area,
            u_value=_to_float(row["U_roof"]),
            orientation_code="HOR",
            azimuth=0.0,
            tilt=0.0,
            boundary="OUTDOORS",
            solar_absorptance=0.65,
            sky_view_factor=1.0,
            thermal_capacity=741_500.0,
            height=roof_area**0.5,
            width=roof_area**0.5,
        )
    )
    surfaces.append(
        _surface(
            name="Slab to ground",
            surface_type="opaque",
            area=ground_area,
            u_value=_to_float(row["U_floor"]),
            orientation_code="HOR",
            azimuth=0.0,
            tilt=0.0,
            boundary="GROUND",
            solar_absorptance=0.0,
            sky_view_factor=0.0,
            thermal_capacity=405_801.0,
            height=ground_area**0.5,
            width=ground_area**0.5,
        )
    )

    return {
        "building": {
            "name": archetype_id,
            "azimuth_relative_to_true_north": 0.0,
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
            "weather_location_name": location["name"],
            "climate_zone": climate_zone,
            "country": "Italy",
            "exposed_perimeter": _exposed_perimeter_from_area(footprint_m2),
            "height": total_height_m,
            "floor_height": floor_height_m,
            "wall_thickness": 0.35,
            "n_floors": floors,
            "footprint_area": footprint_m2,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "net_floor_area": area_m2,
            "treated_floor_area": area_m2,
            "volume": volume_m3,
            "construction_class": "class_i" if period_code != "N0" else "class_ie",
            "construction_year": str(row["Period"]),
            "type_code": type_code,
            "period_code": period_code,
            "strepin_archetype_id": archetype_id,
            "design_heating_capacity_kW": _to_float(row["Heating_Capacity_kW"]),
        },
        "adjacent_zones": [],
        "building_surface": surfaces,
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "C",
            },
            "system_capacities": {
                "heating_capacity": heating_capacity_w,
                "cooling_capacity": 10_000_000.0,
                "units": "W",
            },
            "ventilation": {
                "ventilation_type": "custom",
                "custom_heat_transfer_coefficient_ventilation": ventilation_hve_w_k,
                "effective_air_change_rate_h": effective_ach_h,
                "ventilation_preset": ventilation_preset_id,
                "flow_rate_per_person": 0.0,
                "units": "W/K",
            },
            "airflow_rates": {
                "total_effective_air_change_rate_h": effective_ach_h,
                "infiltration_rate": 0.0,
                "units": "ACH",
                "note": (
                    "STREPIN ventilation presets are total effective outdoor-air "
                    "exchange rates represented only by the custom ventilation Hve."
                ),
            },
            "internal_gains": copy.deepcopy(DEFAULT_INTERNAL_GAINS),
            "construction": {
                "wall_thickness": 0.35,
                "thermal_bridges": 0.05 * wall_area_total,
                "units": "m and W/K",
            },
            "climate_parameters": {
                "coldest_month": int(location.get("coldest_month", 1)),
                "units": "1-12",
            },
            "heating_profile": copy.deepcopy(DEFAULT_HVAC_PROFILE),
            "cooling_profile": copy.deepcopy(DEFAULT_HVAC_PROFILE),
            "ventilation_profile": copy.deepcopy(STREPIN_VENTILATION_PROFILE_ALWAYS_ON),
        },
        "strepin": {
            "source": "Italian STREPIN workbook",
            "archetype": _row_to_dict(row),
            "geometry_note": (
                "Workbook areas are aggregate values; vertical wall area is split "
                "equally by orientation and window area follows the configured "
                "orientation split."
            ),
            "ventilation_preset": ventilation_preset_id,
            "ventilation_preset_description": STREPIN_VENTILATION_PRESETS[
                ventilation_preset_id
            ]["description"],
            "ventilation_schedule": "always_on",
            "air_change_rate_h": effective_ach_h,
            "effective_air_change_rate_h": effective_ach_h,
            "air_change_rate_override_h": ach_override_h,
            "ventilation_hve_W_K": ventilation_hve_w_k,
            "ventilation_assumption_note": (
                "This ACH is a total effective outdoor-air exchange rate. It is "
                "not a measured airtightness value, design ventilation rate or "
                "separate mechanical ventilation stream."
            ),
        },
    }


def _surface(
    *,
    name: str,
    surface_type: str,
    area: float,
    u_value: float,
    orientation_code: str,
    azimuth: float,
    tilt: float,
    boundary: str,
    solar_absorptance: float,
    sky_view_factor: float,
    thermal_capacity: float,
    height: float,
    width: float,
    g_value: float = 0.0,
    parapet: float = 0.0,
    shading: bool = False,
    shading_type: str = "none",
    width_or_distance_of_shading_elements: float = 0.0,
    overhang_properties: dict[str, float] | None = None,
) -> dict[str, Any]:
    iso_type = "W" if surface_type == "transparent" else "OP"
    if boundary.upper() == "GROUND":
        iso_type = "GR"
    out = {
        "name": name,
        "type": surface_type,
        "boundary": boundary.upper(),
        "area": float(area),
        "u_value": float(u_value),
        "sky_view_factor": float(sky_view_factor),
        "solar_absorptance": float(solar_absorptance),
        "thermal_capacity": float(thermal_capacity),
        "orientation": {"azimuth": float(azimuth), "tilt": float(tilt)},
        "ISO52016_type_string": iso_type,
        "ISO52016_orientation_string": orientation_code,
        "convective_heat_transfer_coefficient_internal": (
            2.5 if tilt >= 89.0 else 5.0
        ),
        "radiative_heat_transfer_coefficient_internal": 5.13,
        "convective_heat_transfer_coefficient_external": 20.0,
        "radiative_heat_transfer_coefficient_external": 4.14,
        "height": float(height),
        "width": float(width),
    }
    if surface_type == "transparent":
        out.update(
            {
                "g_value": float(g_value),
                "parapet": float(parapet),
                "shading": bool(shading),
                "shading_type": shading_type,
                "width_or_distance_of_shading_elements": float(
                    width_or_distance_of_shading_elements
                ),
                "overhang_properties": overhang_properties
                or {"width_of_horizontal_overhangs": 0.0},
            }
        )
    return out


def _set_surface_u_value(
    bui: dict[str, Any],
    target_u: float,
    *,
    kind: str,
) -> None:
    for surface in bui.get("building_surface", []):
        if not _surface_matches_kind(surface, kind):
            continue
        old_u = _to_float(surface.get("u_value"), target_u)
        new_u = min(old_u, float(target_u))
        surface.setdefault("strepin_pre_retrofit", {})["u_value"] = old_u
        surface["u_value"] = new_u


def _set_window_g_value(bui: dict[str, Any], target_u: float) -> None:
    g_value = _default_window_g_value(float(target_u))
    for surface in bui.get("building_surface", []):
        if _surface_matches_kind(surface, "window"):
            surface.setdefault("strepin_pre_retrofit", {})["g_value"] = surface.get(
                "g_value"
            )
            surface["g_value"] = g_value


def _set_window_solar_shading(
    bui: dict[str, Any],
    *,
    level: str,
    solar_transmission_factor: float,
    activation_irradiance_W_m2: float | None,
    orientations: set[str],
) -> float:
    factor = max(0.0, min(1.0, float(solar_transmission_factor)))
    active = level != "sf" and factor < 1.0
    affected_area_m2 = 0.0
    for surface in bui.get("building_surface", []):
        if not _surface_matches_kind(surface, "window"):
            continue
        orientation = _surface_orientation_code(surface)
        if orientation not in orientations:
            continue
        affected_area_m2 += _to_float(surface.get("area"), 0.0)
        before = surface.setdefault("strepin_pre_retrofit", {})
        for key in [
            "shading",
            "shading_type",
            "solar_shading_level",
            "solar_shading_transmittance",
            "solar_shading_activation_irradiance_W_m2",
        ]:
            before.setdefault(key, surface.get(key))
        surface["shading"] = active
        surface["shading_type"] = "solar_transmission_factor" if active else "none"
        surface["solar_shading_level"] = level
        surface["solar_shading_transmittance"] = factor
        if activation_irradiance_W_m2 is None:
            surface.pop("solar_shading_activation_irradiance_W_m2", None)
        else:
            surface["solar_shading_activation_irradiance_W_m2"] = float(
                activation_irradiance_W_m2
            )
    return affected_area_m2


def _surface_orientation_code(surface: dict[str, Any]) -> str | None:
    tag = str(surface.get("ISO52016_orientation_string", "")).strip().upper()
    if tag in {"NV", "EV", "SV", "WV", "HOR"}:
        return tag

    orientation = surface.get("orientation", {}) or {}
    try:
        azimuth = float(orientation.get("azimuth")) % 360.0
        tilt = float(orientation.get("tilt"))
    except (TypeError, ValueError):
        return None
    if tilt < 45.0:
        return "HOR"
    candidates = [(0.0, "NV"), (90.0, "EV"), (180.0, "SV"), (270.0, "WV")]
    return min(
        candidates,
        key=lambda item: abs(((azimuth - item[0] + 180.0) % 360.0) - 180.0),
    )[1]


def _solar_shading_orientations(raw: Any) -> set[str]:
    if raw is None or raw == "":
        return set(DEFAULT_SOLAR_SHADING_ORIENTATIONS)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.replace(";", ",").split(",")]
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    aliases = {
        "N": "NV",
        "NORTH": "NV",
        "NV": "NV",
        "E": "EV",
        "EAST": "EV",
        "EV": "EV",
        "S": "SV",
        "SOUTH": "SV",
        "SV": "SV",
        "W": "WV",
        "WEST": "WV",
        "WV": "WV",
    }
    out = set()
    for value in values:
        key = str(value).strip().upper()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                "Solar shading orientations must use N/E/S/W or NV/EV/SV/WV labels."
            )
        out.add(aliases[key])
    return out or set(DEFAULT_SOLAR_SHADING_ORIENTATIONS)


def _solar_shading_default_cost_eur_m2(level: str, params: dict[str, Any]) -> float:
    if level == "sf":
        return 0.0
    if level == "fixed":
        return DEFAULT_FIXED_SOLAR_SHADING_COST_EUR_M2

    base_width_cm = _to_float(
        params.get("base_width_cm", params.get("B_cm")),
        DEFAULT_MOBILE_SOLAR_SHADING_BASE_WIDTH_CM,
    )
    element = str(params.get("element", "window")).strip().lower()
    if element in {"door", "door_window", "porta_finestra", "portafinestra"}:
        return max(0.0, -0.066 * base_width_cm + 31.794)
    return max(0.0, -0.0484 * base_width_cm + 31.755)


def _set_internal_gain_full_load(
    bui: dict[str, Any],
    name: str,
    full_load_W_m2: float,
) -> None:
    gains = (
        bui.setdefault("building_parameters", {})
        .setdefault("internal_gains", [])
    )
    for gain in gains:
        if str(gain.get("name", "")).lower() == str(name).lower():
            gain.setdefault("strepin_pre_retrofit", {})["full_load"] = gain.get("full_load")
            gain["full_load"] = float(full_load_W_m2)
            return
    gains.append(
        {
            "name": str(name),
            "full_load": float(full_load_W_m2),
            "weekday": [1.0] * 24,
            "weekend": [1.0] * 24,
        }
    )


def _surface_matches_kind(surface: dict[str, Any], kind: str) -> bool:
    boundary = str(surface.get("boundary", "OUTDOORS")).upper()
    surface_type = str(surface.get("type", "")).lower()
    name = str(surface.get("name", "")).lower()
    tilt = _to_float((surface.get("orientation") or {}).get("tilt"), 90.0)
    if kind == "wall":
        return surface_type == "opaque" and boundary == "OUTDOORS" and tilt >= 89.0
    if kind == "roof":
        return surface_type == "opaque" and boundary == "OUTDOORS" and tilt < 1.0
    if kind == "ground":
        return boundary == "GROUND" or "ground" in name or "slab" in name
    if kind == "window":
        return surface_type == "transparent"
    return False


def _heating_measure_metadata(row: pd.Series) -> dict[str, Any]:
    level = int(row["Level"])
    value = _to_float(row.get("Heating_Eff_or_COP"), 0.0)
    carrier = "electricity" if level >= 3 else "gas"
    system_type = "air_to_water_heat_pump" if level >= 3 else "gas_boiler"
    if level == 1:
        system_type = "baseline_gas_boiler"
    elif level == 2:
        system_type = "condensing_gas_boiler"
    return {
        "level": level,
        "level_label": str(row.get("Level_Label", "")),
        "system_type": system_type,
        "carrier": carrier,
        "seasonal_efficiency_or_scop": value,
        "investment_EUR": _to_float(row.get("Investment_EUR"), 0.0),
    }


def _default_window_g_value(u_value: float) -> float:
    if u_value <= 1.4:
        return 0.50
    if u_value <= 2.0:
        return 0.55
    if u_value <= 3.0:
        return 0.62
    return 0.75


def _normalize_split(split: dict[str, float]) -> dict[str, float]:
    normalized = {k: max(0.0, float(v)) for k, v in split.items()}
    total = sum(normalized.values())
    if total <= 0.0:
        return DEFAULT_WINDOW_ORIENTATION_SPLIT.copy()
    return {k: v / total for k, v in normalized.items()}


def _exposed_perimeter_from_area(area_m2: float) -> float:
    side = max(float(area_m2), 1.0) ** 0.5
    return 4.0 * side


def _building(bui: dict[str, Any]) -> dict[str, Any]:
    return dict(bui.get("building", {}))


def _building_area(bui: dict[str, Any]) -> float:
    building = _building(bui)
    return _to_float(
        building.get("net_floor_area", building.get("treated_floor_area")),
        0.0,
    )


def _building_volume(bui: dict[str, Any]) -> float:
    building = _building(bui)
    volume = _to_float(building.get("volume"), 0.0)
    if volume > 0.0:
        return volume
    return _building_area(bui) * _to_float(building.get("floor_height"), 3.0)


def _annual_dict(annual_results: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
    if isinstance(annual_results, pd.DataFrame):
        if annual_results.empty:
            return {}
        return annual_results.iloc[0].to_dict()
    return dict(annual_results)


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): _json_like(v) for k, v in row.to_dict().items()}


def _json_like(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _to_int(value: Any, default: int = 0) -> int:
    return int(round(_to_float(value, float(default))))


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
