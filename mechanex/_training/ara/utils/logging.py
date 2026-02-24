"""Logging configuration for ARA."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for ARA module."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger('ara')
    logger.setLevel(level)
    logger.addHandler(handler)
