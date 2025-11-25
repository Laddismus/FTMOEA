import logging
import os
from typing import Optional


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """
    Configure legacy logging setup (kept for compatibility).
    """
    logging.basicConfig(
        level=level,
        format=fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging globally with support for ENV override.
    """
    env_level = os.getenv("AFTS_LOG_LEVEL")
    resolved_level = (env_level or level or "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
