"""Entry point: python -m sign_prep <config.yaml>"""

import logging
import sys

# Ensure registrations happen on import
import sign_prep.datasets  # noqa: F401
import sign_prep.processors  # noqa: F401
import sign_prep.extractors  # noqa: F401

from sign_prep.cli import parse_args
from sign_prep.config import load_config
from sign_prep.pipeline import PipelineRunner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    config = load_config(args.config, overrides=args.override or None)
    runner = PipelineRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
