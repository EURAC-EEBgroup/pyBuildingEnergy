"""CLI wrapper around ``pybuildingenergy.simulate_config()``.

The simplest possible pybuildingenergy chain - a single-zone building with
ONLY an emission system attached, no distribution, no generation, no AHU:

    ISO 52016 (building energy need) -> EmissionSystemCalculator (EN 15316-2)

The logic itself lives in the installed library (``pybuildingenergy.source.
composer.simulate_config``); this script just adds the CLI on top.
From any other project, after ``pip install pybuildingenergy``:

    import pybuildingenergy as pybui
    config = pybui.load_config("example_building_emission_only.json")
    result = pybui.simulate_config(config)
    print(result.report())

Usage:
    python examples/building_emission_only.py
    python examples/building_emission_only.py --config path/to/other_config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pybuildingenergy as pybui

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLES_DIR / "example_building_emission_only.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="building+emission JSON (default: example_building_emission_only.json)")
    ap.add_argument("--output-dir", default=str(EXAMPLES_DIR / "outputs" / "building_emission_only"))
    args = ap.parse_args()

    config = pybui.load_config(args.config)
    res = pybui.simulate_config(config)
    print(res.report())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.hourly.to_csv(out_dir / "hourly.csv")
    print(f"\nHourly detail saved to {out_dir / 'hourly.csv'}")


if __name__ == "__main__":
    main()
