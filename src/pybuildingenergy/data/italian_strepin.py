"""Italian STREPIN residential archetypes and renovation packages.

This module treats the STREPIN workbook as input data only. It builds
pyBuildingEnergy BUI dictionaries for the 18 residential archetypes and applies
the package levels by changing the simulated envelope/system metadata. Workbook
performance estimates are kept as comparison metadata; they are not used to
drive the engine-side demand calculation.
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
        air_change_rate_h: float = 0.30,
        ideal_hvac_capacity: bool = True,
    ) -> dict[str, Any]:
        return _archetype_to_bui(
            self.archetype_row(archetype_id),
            climate_locations=climate_locations,
            window_orientation_split=window_orientation_split,
            floor_height_m=floor_height_m,
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
        air_change_rate_h: float = 0.30,
        ideal_hvac_capacity: bool = True,
    ) -> StrepinCase:
        bui = self.make_base_bui(
            archetype_id,
            climate_locations=climate_locations,
            window_orientation_split=window_orientation_split,
            floor_height_m=floor_height_m,
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

    heating_system = case.bui.get("strepin", {}).get("heating_system", {})
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
    air_change_rate_h: float,
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
    ventilation_hve_w_k = 0.33 * float(air_change_rate_h) * volume_m3

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
                    overhang_proprieties={"width_of_horizontal_overhangs": 0.0},
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
                "flow_rate_per_person": 0.0,
                "units": "W/K",
            },
            "airflow_rates": {
                "infiltration_rate": float(air_change_rate_h),
                "units": "ACH",
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
            "ventilation_profile": copy.deepcopy(DEFAULT_HVAC_PROFILE),
        },
        "strepin": {
            "source": "Italian STREPIN workbook",
            "archetype": _row_to_dict(row),
            "geometry_note": (
                "Workbook areas are aggregate values; vertical wall area is split "
                "equally by orientation and window area follows the configured "
                "orientation split."
            ),
            "air_change_rate_h": float(air_change_rate_h),
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
    overhang_proprieties: dict[str, float] | None = None,
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
                "overhang_proprieties": overhang_proprieties
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
