"""CLI argument parsing for Signdata."""

import argparse
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Signdata"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = subparsers.add_parser("run", help="Run a preprocessing job")
    run_parser.add_argument(
        "config", nargs="?", default=None,
        help="Path to YAML job config file",
    )
    run_parser.add_argument(
        "--force-all", action="store_true", default=False,
        help="Force rerun of all stages",
    )
    run_parser.add_argument(
        "--run-name", default=None,
        help="Override run_name for path isolation",
    )
    run_parser.add_argument(
        "--list-presets", action="store_true", default=False,
        help="List available keypoint presets and exit",
    )
    run_parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides: key=value (e.g. processing.max_workers=8)",
    )

    # --- experiment subcommand ---
    exp_parser = subparsers.add_parser(
        "experiment", help="Run a multi-job experiment",
    )
    exp_parser.add_argument(
        "config",
        help="Path to YAML experiment config file",
    )
    exp_parser.add_argument(
        "--force-all", action="store_true", default=False,
        help="Force rerun of all stages in every job",
    )

    return parser.parse_args(argv)
