"""CLI wrapper around ``pybuildingenergy``'s composer module.

All the actual logic - the reference-JSON catalog, ``compose_config()``,
``simulate()``, ``load_config()`` and friends - lives IN the installed library
now (``pybuildingenergy.source.composer``, re-exported from the top-level
package). This script only adds the ``argparse`` CLI on top, plus one
example-only convenience: it points at the bundled ``2020_Milan.epw`` weather
file by default, when present, instead of the library's own pvgis default.

Any other project just does ``pip install pybuildingenergy`` and calls the
functions directly - no ``sys.path`` hacks, no copying this file around:

    import pybuildingenergy as pybui
    config = pybui.load_config("example_combined_config.json")
    result = pybui.simulate(config)
    print(result.report())

Usage:
    python examples/compose_single_zone_systems.py --list
    python examples/compose_single_zone_systems.py \
        --emission floor_heating_en15316_2_neutral \
        --distribution analytical_detailed_dict \
        --generation boiler_15316_4_1 \
        --pv pv_with_battery_dispatch --ahu ahu_fixed_supply_basic

    # Only produce the combined JSON, do not simulate:
    python examples/compose_single_zone_systems.py --save-config-only

    # Simulate FROM a previously composed JSON (round-trip):
    python examples/compose_single_zone_systems.py --from-config out/composed_config.json

    python examples/compose_single_zone_systems.py --compare-generation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pybuildingenergy as pybui

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_EPW = EXAMPLES_DIR / "2020_Milan.epw"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the selectable options and exit")
    ap.add_argument("--scenario", help="JSON file with a scenario dict (see pybuildingenergy.source.composer.DEFAULT_SCENARIO)")
    ap.add_argument("--from-config", help="simulate directly from a previously composed combined JSON")
    ap.add_argument("--save-config-only", action="store_true", help="only write the combined JSON, do not simulate")
    ap.add_argument("--building", help="single-zone BUI JSON (default: built-in demo)")
    ap.add_argument("--weather-source", choices=["epw", "pvgis"]); ap.add_argument("--weather-file")
    for k in ("emission", "distribution", "generation", "pv", "ahu"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--no-autosize", action="store_true"); ap.add_argument("--fallback-cooling-eer", type=float)
    ap.add_argument("--baseline-electricity", type=float, help="other electric loads [kWh/year]")
    ap.add_argument("--compare-generation", action="store_true", help="run all generation options and print a comparison table")
    ap.add_argument("--output-dir", default=str(EXAMPLES_DIR / "outputs" / "composed"))
    args = ap.parse_args()

    catalog = pybui.build_catalog()
    if args.list:
        pybui.print_catalog(catalog)
        return

    if args.from_config:
        config = pybui.load_config(args.from_config)
        res = pybui.simulate(config)
        print(res.report())
        pybui.save_outputs(res, Path(args.output_dir))
        print(f"\nSaved: {args.output_dir}/composed_hourly.csv, composed_config.json and composed_summary.json")
        return

    scenario: dict[str, Any] = json.loads(Path(args.scenario).read_text(encoding="utf-8")) if args.scenario else {}
    scenario.setdefault("options", {})
    if "weather_source" not in scenario and args.weather_source is None and DEFAULT_EPW.exists():
        scenario["weather_source"] = "epw"
        scenario["weather_file"] = str(DEFAULT_EPW)
    for k in ("building", "weather_source", "weather_file", "emission", "distribution", "generation", "pv", "ahu"):
        v = getattr(args, k.replace("-", "_"), None)
        if v is not None:
            scenario[k] = v
    if args.no_autosize:
        scenario["options"]["autosize"] = False
    if args.fallback_cooling_eer:
        scenario["options"]["fallback_cooling_eer"] = args.fallback_cooling_eer
    if args.baseline_electricity is not None:
        scenario["options"]["baseline_electricity_kWh_per_year"] = args.baseline_electricity
    scenario["options"].setdefault("hourly_cache_dir", str(Path(args.output_dir) / "cache"))

    if args.compare_generation:
        print(pybui.compare_generation(scenario).to_string())
        return

    config = pybui.compose_config(scenario, catalog=catalog)
    if args.save_config_only:
        path = pybui.save_config(config, Path(args.output_dir) / "composed_config.json")
        print(f"Combined JSON written to {path}")
        return

    res = pybui.simulate(config)
    print(res.report())
    pybui.save_outputs(res, Path(args.output_dir))
    print(f"\nSaved: {args.output_dir}/composed_hourly.csv, composed_config.json and composed_summary.json")


if __name__ == "__main__":
    main()
