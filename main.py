#!/usr/bin/env python3
"""Unified entry point for the Reservoir Intelligence System.

Usage
-----
  python main.py rainfall-pipeline [--fetch] [--engineer] [--train] [--all]
  python main.py predict
  python main.py sediment-pipeline [--preprocess] [--train] [--predict]
  python main.py dashboard
  python main.py all
"""

import sys
import argparse
import subprocess
from pathlib import Path


def _section(title):
    bar = "=" * 70
    print("\n" + bar)
    print("  " + title)
    print(bar + "\n")


def _run_module(args, description):
    print("  -> " + description)
    result = subprocess.run([sys.executable] + args, capture_output=False)
    if result.returncode != 0:
        print("\n  " + description + " exited with code " + str(result.returncode))
        return False
    return True


def cmd_rainfall_pipeline(extra_args):
    _section("RAINFALL FORECASTING PIPELINE")
    ok = _run_module(["pipeline.py"] + extra_args, "pipeline.py")
    return 0 if ok else 1


def cmd_predict(_extra_args):
    _section("REAL-TIME RAINFALL PREDICTIONS")
    ok = _run_module(["river_weather_predictions.py"], "river_weather_predictions.py")
    return 0 if ok else 1


def cmd_sediment_pipeline(extra_args):
    _section("SEDIMENT HOTSPOT PIPELINE")
    ok = _run_module(["sediment_pipeline.py"] + extra_args, "sediment_pipeline.py")
    return 0 if ok else 1


def cmd_dashboard(_extra_args):
    _section("STREAMLIT DASHBOARD")
    dashboard = Path("dashboards/sediment_dashboard.py")
    if not dashboard.exists():
        print("  Dashboard not found at " + str(dashboard))
        return 1
    print("  Starting Streamlit -- open http://localhost:8501 in your browser.\n")
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard)],
        capture_output=False,
    )
    return result.returncode


def cmd_all(_extra_args):
    _section("FULL END-TO-END RUN")
    steps = [
        ("Rainfall pipeline", ["pipeline.py", "--all"]),
        ("Real-time predictions", ["river_weather_predictions.py"]),
    ]
    for description, args in steps:
        if not _run_module(args, description):
            print("\n  Stopped at: " + description)
            return 1
    _section("ALL STEPS COMPLETE")
    print("  rainfall_predictions.csv  -> latest zone predictions")
    print("  rainfall_model.pkl        -> trained XGBoost model")
    print("\n  Run  python main.py dashboard  to open the monitoring dashboard.\n")
    return 0


COMMANDS = {
    "rainfall-pipeline": (cmd_rainfall_pipeline, "Run the rainfall ML pipeline"),
    "predict":           (cmd_predict,           "Run real-time rainfall predictions"),
    "sediment-pipeline": (cmd_sediment_pipeline, "Run the sediment hotspot pipeline"),
    "dashboard":         (cmd_dashboard,         "Launch the Streamlit dashboard"),
    "all":               (cmd_all,               "Full end-to-end pipeline + predictions"),
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Reservoir Intelligence System -- unified backend entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            "  " + cmd.ljust(22) + desc for cmd, (_, desc) in COMMANDS.items()
        ),
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()), help="Subsystem to run")
    return parser


def main():
    if len(sys.argv) < 2:
        build_parser().print_help()
        sys.exit(0)
    command = sys.argv[1]
    if command not in COMMANDS:
        build_parser().print_help()
        sys.exit(1)
    extra_args = sys.argv[2:]
    handler, _ = COMMANDS[command]
    _section("RESERVOIR INTELLIGENCE SYSTEM")
    print("  Subsystem : " + command)
    if extra_args:
        print("  Options   : " + " ".join(extra_args))
    sys.exit(handler(extra_args))


if __name__ == "__main__":
    main()
