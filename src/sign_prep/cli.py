"""CLI argument parsing for sign_prep."""

import argparse
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign Language Preprocessing Pipeline"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides: key=value (e.g. processing.max_workers=8)",
    )
    return parser.parse_args(argv)
