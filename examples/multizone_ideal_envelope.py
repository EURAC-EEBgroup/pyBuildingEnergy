"""CLI runner for multizone "ideal envelope" configs (schema
``pybuildingenergy-building-config-v2``, top-level keys ``weather`` /
``building`` / ``simulation``), e.g. ``FH_Poland_DC_multizone_ideal_envelope.json``.

There is no dedicated runner for this schema in the library itself: it calls
``ISO52016.Temperature_and_Energy_needs_calculation_multizone`` directly with
``building_object = config["building"]``.

Progress logging is opt-in (off by default) and not part of the standard
library API: it wires ``progress_log_every_steps`` / ``progress_logger``,
two optional kwargs accepted by the core simulation function.

Usage:
    python examples/multizone_ideal_envelope.py --config path/to/config.json
    python examples/multizone_ideal_envelope.py --config path/to/config.json --progress
    python examples/multizone_ideal_envelope.py --config path/to/config.json --progress --progress-log-hours 168
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pybuildingenergy.source.utils import ISO52016

EXAMPLES_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="multizone building-config-v2 JSON")
    ap.add_argument("--warmup-hours", type=int, default=744)
    ap.add_argument("--hvac-control-variable", choices=["operative", "air"], default="operative")
    ap.add_argument("--progress", action="store_true", help="print advancement percentage while simulating (off by default)")
    ap.add_argument("--progress-log-hours", type=int, default=720, help="log every N simulated hours, only used with --progress")
    ap.add_argument("--output-dir", default=None, help="if set, saves hourly.csv and annual.csv here")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    building_object = config["building"]

    kwargs = {}
    if args.progress:
        run_start_t = time.perf_counter()

        def progress_logger(message: str) -> None:
            print(f"[{time.perf_counter() - run_start_t:7.1f}s] {message}", flush=True)

        kwargs["progress_log_every_steps"] = args.progress_log_hours
        kwargs["progress_logger"] = progress_logger

    hourly, annual = ISO52016.Temperature_and_Energy_needs_calculation_multizone(
        building_object=building_object,
        weather_source=config["weather"]["source"],
        path_weather_file=config["weather"].get("file"),
        include_solar=config["simulation"].get("include_solar", True),
        include_internal_gains=config["simulation"].get("include_internal_gains", True),
        include_ventilation=config["simulation"].get("include_ventilation", True),
        warmup_hours=args.warmup_hours,
        hvac_control_variable=args.hvac_control_variable,
        **kwargs,
    )

    print(annual)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(out_dir / "hourly.csv")
        annual.to_csv(out_dir / "annual.csv")
        print(f"\nSaved: {out_dir / 'hourly.csv'}, {out_dir / 'annual.csv'}")


if __name__ == "__main__":
    main()
