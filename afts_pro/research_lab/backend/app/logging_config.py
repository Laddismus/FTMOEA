"""Logging configuration for the Research Lab backend."""

from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    """Configure standard logging to stdout with a consistent format."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


__all__ = ["configure_logging"]
