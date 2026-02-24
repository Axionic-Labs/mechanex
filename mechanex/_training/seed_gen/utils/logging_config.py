"""
Logging configuration for the Seed Generator.
"""

import logging
import sys
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> None:
    """
    Set up logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        use_rich: Whether to use Rich for console output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = []

    # Console handler
    if use_rich:
        console = Console()
        handlers.append(
            RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False
            )
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

    # Set level for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
