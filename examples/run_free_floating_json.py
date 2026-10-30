"""Run the complete free-floating JSON examples.

Usage from the repository root::

    python3 examples/run_free_floating_json.py examples/example_single_zone_free_floating.json
    python3 examples/run_free_floating_json.py examples/example_multizone_free_floating.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep the example runnable from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
from pybuildingenergy.source.composer import simulate_multizone
from pybuildingenergy.source.utils import ISO52016


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    building = config["building"]
    weather = config.get("weather", {})
    simulation = config.get("simulation", {})
    weather_kwargs = {
        "weather_source": weather.get("source", "epw"),
        "path_weather_file": weather.get("file"),
        "include_solar": simulation.get("include_solar", True),
        "include_internal_gains": simulation.get("include_internal_gains", True),
        "include_ventilation": simulation.get("include_ventilation", True),
        "warmup_hours": simulation.get("warmup_hours", 744),
        "hvac_control_variable": simulation.get("hvac_control_variable", "air"),
    }

    if building.get("zones"):
        result = simulate_multizone(config, **weather_kwargs)
        print(result.report())
        return 0

    checked, issues = sanitize_and_validate_BUI(building, fix=True)
    errors = [issue for issue in issues if issue["level"] == "ERROR"]
    if errors:
        raise ValueError(f"Building validation errors: {errors}")
    hourly, _ = ISO52016.Temperature_and_Energy_needs_calculation(
        checked,
        **weather_kwargs,
    )
    print(f"{config_path}: {len(hourly)} timesteps")
    print(f"Free-floating: max T_air={hourly['T_air'].max():.2f} °C; min T_air={hourly['T_air'].min():.2f} °C")
    print("HVAC energy should be zero because both system capacities are 0 W.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
