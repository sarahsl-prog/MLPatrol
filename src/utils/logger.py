"""Logging utilities used across MLPatrol."""

from __future__ import annotations

import logging
import os
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_configured = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once."""
    global _configured
    if _configured:
        return

    log_level = level or os.getenv("MLPATROL_LOG_LEVEL", "INFO")
    logging.basicConfig(level=log_level.upper(), format=DEFAULT_FORMAT)
    # Explicitly set root logger level since basicConfig may not work if already configured
    logging.root.setLevel(log_level.upper())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with MLPatrol defaults applied."""
    configure_logging()
    return logging.getLogger(name)
